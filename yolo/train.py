# Ultralytics YOLO 🚀, AGPL-3.0 license

import sys
from pathlib import Path

file = Path(__file__)
sys.path.append(str(file.parents[2]))

import os
import gc
import math
import time
import warnings
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import random
import pandas as pd

import numpy as np
import torch.nn as nn

import torch
from torch import distributed as dist
from torch import optim
import torch.utils
import torch.utils.data

from loader import build_dataloader, build_yolo_dataset
from valid import DetectionValidator
from tools.autobatch import check_train_batch_size
from tools.torch_utils import (
    init_seeds,
    convert_optimizer_state_dict_to_fp16,
    check_amp,
    one_cycle,
    strip_optimizer,
    ModelEMA,
    EarlyStopping
)

from tools.common import (
    yaml_load,
    check_imgsz,
    IterableSimpleNamespace,
    RANK,
    TQDM,
) 

def yaml_load_dataset(data_cfg):
    try:
        data = yaml_load(data_cfg)
        # Set paths
        path = Path(data["path"])
        for k in "train", "val", "test":
            if data.get(k):  
                x = (path / data[k]).resolve()
                if not x.exists() and data[k].startswith("../"):
                    x = (path / data[k][3:]).resolve()
                data[k] = str(x)
        data['nc'] = len(data['names']) # num class
    
    except Exception as e:
        raise ValueError(f"Dataset '{data_cfg}' error ❌ {e}") from e
    
    return data, data["train"], data.get("val") or data.get("test")
    
class Trainer(object):
    """
    BaseTrainer.

    A base class for creating trainers.
    """

    def __init__(self, model, train_cfg, ckpt_path=None):
        """
        Initializes the BaseTrainer class.

        Args:
            model: training model class module
            cfg (str, optional): Path to a configuration file. Defaults to DEFAULT_CFG.
            overrides (dict, optional): Configuration overrides. Defaults to None.
        """
        # set train cfg
        self.args = yaml_load(train_cfg)
        for k, v in self.args.items():
            if isinstance(v, str) and v.lower() == "none":
                self.args[k] = None
        self.args = IterableSimpleNamespace(**self.args) 
        
        # set device
        if 'cuda' in self.args.device and torch.cuda.is_available():
            self.device = torch.device(self.args.device)
        else:
            self.device = torch.device('cpu')

        if self.device.type in {"cpu", "mps"}:
            self.args.workers = 0  # faster CPU training

        # set random seed
        init_seeds(self.args.seed + 1, deterministic=self.args.deterministic) # cuda deterministic alrigthom 


        # set dirs 
        self.save_dir = Path(self.args.save_dir) 
        self.wdir = self.save_dir / "weights" 
        self.last, self.best = self.wdir / "last.pt", self.wdir / "best.pt" 
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.wdir, exist_ok=True)
        
        # set trin times
        self.save_period = self.args.save_period
        self.batch_size = self.args.batch
        self.epochs = self.args.epochs
        self.start_epoch = 0
        
        # model
        self.model = model
        self.ckpt_path = ckpt_path
        self.resume = self.args.resume
        self.warmup = True 
        self.ema = None

        # dataset
        self.data, self.trainset, self.testset = yaml_load_dataset(self.args.data) # get path

        # Optimization utils init
        self.lf = None
        self.scheduler = None

        # Epoch level metrics
        self.best_fitness = None
        self.fitness = None
        self.tloss = None
        # self.loss_names = ["Loss"]
        self.csv = self.save_dir / "results.csv"
    
    
    def build_optimizer(self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
        """
        Constructs an optimizer for the given model, based on the specified optimizer name, learning rate, momentum,
        weight decay, and number of iterations.

        Args:
            model (torch.nn.Module): The model for which to build an optimizer.
            name (str, optional): The name of the optimizer to use. If 'auto', the optimizer is selected
                based on the number of iterations. Default: 'auto'.
            lr (float, optional): The learning rate for the optimizer. Default: 0.001.
            momentum (float, optional): The momentum factor for the optimizer. Default: 0.9.
            decay (float, optional): The weight decay for the optimizer. Default: 1e-5.
            iterations (float, optional): The number of iterations, which determines the optimizer if
                name is 'auto'. Default: 1e5.

        Returns:
            (torch.optim.Optimizer): The constructed optimizer.
        """

        g = [], [], []  # optimizer parameter groups
        bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)  # normalization layers, i.e. BatchNorm2d()
        if name == "auto":
            nc = getattr(model, "nc", 10)  # number of classes
            lr_fit = round(0.002 * 5 / (4 + nc), 6)  # lr0 fit equation to 6 decimal places
            name, lr, momentum = ("SGD", 0.01, 0.9) if iterations > 10000 else ("AdamW", lr_fit, 0.9)
            self.args.warmup_bias_lr = 0.0  # no higher than 0.01 for Adam

        for module_name, module in model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                fullname = f"{module_name}.{param_name}" if module_name else param_name
                if "bias" in fullname:        # bias (no decay)
                    g[2].append(param)
                elif isinstance(module, bn):  # weight (no decay)
                    g[1].append(param)
                else:                         # weight (with decay)
                    g[0].append(param)

        if name in {"Adam", "Adamax", "AdamW", "NAdam", "RAdam"}:
            optimizer = getattr(optim, name, optim.Adam)(
                                                        g[2], 
                                                        lr=lr, 
                                                        betas=(momentum, 0.999), 
                                                        weight_decay=0.0)
        elif name == "RMSProp":
            optimizer = optim.RMSprop(g[2], 
                                      lr=lr, 
                                      momentum=momentum)
        elif name == "SGD":
            optimizer = optim.SGD(g[2], 
                                  lr=lr, 
                                  momentum=momentum, 
                                  nesterov=True)
        else:
            raise NotImplementedError(
                f"Optimizer '{name}' not found in list of available optimizers "
                f"[Adam, AdamW, NAdam, RAdam, RMSProp, SGD, auto]."
                "To request support for addition optimizers please visit https://github.com/ultralytics/ultralytics."
            )

        optimizer.add_param_group({"params": g[0], "weight_decay": decay})   # add g0 with weight_decay
        optimizer.add_param_group({"params": g[1], "weight_decay": 0.0})     # add g1 (BatchNorm2d weights)
        return optimizer
    
    def setup_optimizer(self):
        """Initialize training learning rate scheduler."""
        self.accumulate = max(round(self.args.nbs / self.batch_size), 1)                           # accumulate loss before optimizing
        weight_decay = self.args.weight_decay * self.batch_size * self.accumulate / self.args.nbs  # scale weight_decay
        iterations = math.ceil(len(self.train_loader.dataset) / max(self.batch_size, self.args.nbs)) * self.epochs
        self.optimizer = self.build_optimizer(
            model=self.model,
            name=self.args.optimizer,
            lr=self.args.lr0,
            momentum=self.args.momentum,
            decay=weight_decay,
            iterations=iterations,
        )
        
        # lr scheduler
        if self.args.cos_lr:
            self.lf = one_cycle(1, self.args.lrf, self.epochs)  # cosine 1->hyp['lrf']
        else:
            self.lf = lambda x: max(1 - x / self.epochs, 0) * (1.0 - self.args.lrf) + self.args.lrf  # linear
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)

    def setup_model(self):
        # pretrain
        ckpt = None
        if self.ckpt_path:
            ckpt = torch.load(self.ckpt_path)
            if ckpt.get('model', None) is None:
                ckpt['model'] = deepcopy(ckpt['ema'])
            self.model.load(ckpt)
        
        # set data info
        self.model.to(self.device)
        self.model.nc = self.data["nc"]             # number of classes to model
        self.model.names = self.data["names"]       # class names to model
        self.model.args = self.args                 # hyperparameters to model

        # freeze layers
        freeze_list = (
            self.args.freeze
            if isinstance(self.args.freeze, list)
            else range(self.args.freeze)
            if isinstance(self.args.freeze, int)
            else []
        )
        always_freeze_names = [".dfl"]  # always freeze these layers
        freeze_layer_names = [f"model.{x}." for x in freeze_list] + always_freeze_names
        for k, v in self.model.named_parameters():
            if any(x in k for x in freeze_layer_names):
                v.requires_grad = False
            elif not v.requires_grad and v.dtype.is_floating_point:  # only floating point Tensor can require gradients
                v.requires_grad = True
        return ckpt
    
    def setup_data(self):
        # Check imgsz
        gs = max(int(self.model.stride.max() if hasattr(self.model, "stride") else 32), 32)  
        self.args.imgsz = check_imgsz(self.args.imgsz, stride=gs, floor=gs, max_dim=1)
        self.stride = gs  # for multiscale training

        # dataloader
        if self.batch_size == -1 :
            self.args.batch = self.batch_size = check_train_batch_size(self.model, 
                                                                       self.args.imgsz, 
                                                                       self.amp)
        self.train_loader = self.build_dataloader(self.trainset, 
                                                 batch_size=self.batch_size, 
                                                 rank=-1, 
                                                 mode="train"
                                                 )
        
        self.test_loader = self.build_dataloader(self.testset, 
                                                 batch_size=self.batch_size, 
                                                 rank=-1, 
                                                 mode="val"
                                                 )
        
    def setup_amp(self):
        self.amp = self.args.amp
        if self.amp:
            self.amp = torch.tensor(check_amp(self.model), device=self.device)                       
        self.amp = bool(self.amp)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)
        """
        scale for grad backward and unscale to get real grad, solve amp overflow or underflow problem; e.g float16 float32 
        """
    
    def setup_validater(self):
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        # self.loss_names = "box_om", "cls_om", "dfl_om", "box_oo", "cls_oo", "dfl_oo" # v10
    
        self.validator = DetectionValidator(self.test_loader, 
                                            save_dir=self.save_dir, 
                                            args=deepcopy(self.args)
                                            )
        metric_keys = self.validator.metrics.keys + self.label_loss_items(prefix="val")
        self.metrics = dict(zip(metric_keys, [0] * len(metric_keys)))

    def _setup_train(self):
        # model
        ckpt = self.setup_model()
        
        # ema
        self.ema = ModelEMA(self.model)
        
        # amp
        self.setup_amp()
        
        # data
        self.setup_data()
        
        # optimizer
        self.setup_optimizer()
        
        # val
        self.setup_validater()    
        
        # early stop
        self.stopper, self.stop = EarlyStopping(patience=self.args.patience), False
        self.scheduler.last_epoch = self.start_epoch - 1
        
        # resume optimizer and training epoch setting
        self.resume_training(ckpt)
        del ckpt


    def train(self):
        """Train completed, evaluate and plot if specified by arguments."""        
        self._setup_train()

        nb = len(self.train_loader)  # num of batches
        nw = max(round(self.args.warmup_epochs * nb), 100) if self.args.warmup_epochs > 0 else -1  # warmup iterations
        last_opt_step = -1

        # valid
        # self.metrics, self.fitness = self.validate()
         
        epoch = self.start_epoch
        self.optimizer.zero_grad()
        while True:
            self.model.train() 
            self.epoch = epoch
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.scheduler.step()                   # update lr

            pbar = enumerate(self.train_loader)
            
            # final n epcoh close mosia
            if epoch == (self.epochs - self.args.close_mosaic):
                self._close_dataloader_mosaic()
                self.train_loader.reset()

            title = ('epoch', 'mem', *self.loss_names, 'batch cls', 'img size')
            title = ("\n" + "%11s" * len(title)) % title
            print(title)
            pbar = TQDM(enumerate(self.train_loader), total=nb, desc=title)
           
            self.tloss = None
            
            for i, batch in pbar:                
                ni = i + nb * epoch
                if self.warmup and ni < nw: # Warmup
                    xi = [0, nw]  
                    self.accumulate = max(1, int(np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round())) # update accumulate , become bigger 
                    # update optimize
                    for j, x in enumerate(self.optimizer.param_groups):
                        x["lr"] = np.interp(
                            ni, xi, [self.args.warmup_bias_lr if j == 0 else 0.0, x["initial_lr"] * self.lf(epoch)]
                        )
                        if "momentum" in x:
                            x["momentum"] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])

                # Forward with amp, just set autocast
                with torch.cuda.amp.autocast(self.amp):
                    batch = self.preprocess_batch(batch)
                    self.loss, self.loss_items = self.model.loss(batch)                                        
                    self.tloss = (
                        (self.tloss * i + self.loss_items) / (i + 1) if self.tloss is not None else self.loss_items
                    ) 

                # Backward
                self.scaler.scale(self.loss).backward()                                     # g' = g * s
                if ni - last_opt_step >= self.accumulate:                                   # accumulate grad update grad
                    self.scaler.unscale_(self.optimizer)                                    # unscale to get real grads, g = g' / s
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                                   max_norm=10.0)                           # clip grad, scale if not bigger than max_norm
                    self.scaler.step(self.optimizer)
                    self.scaler.update()                                                    # reset for next backeward
                    self.optimizer.zero_grad()
                    if self.ema:
                        self.ema.update(self.model)
                    last_opt_step = ni
                    
                # Log
                mem = f"{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G"  # (GB)
                loss_len = self.tloss.shape[0] if len(self.tloss.shape) else 1
                losses = self.tloss if loss_len > 1 else torch.unsqueeze(self.tloss, 0)
                pbar.set_description(
                        ("%11s" * 2 + "%11.4g" * (loss_len + 2))
                        % (f"{epoch + 1}/{self.epochs}", mem, *losses, batch["cls"].shape[0], batch["img"].shape[-1])
                    )
    
            
            # Log enpoch
            self.lr = {f"lr/pg{ir}": x["lr"] for ir, x in enumerate(self.optimizer.param_groups)}  # for loggers
            
            final_epoch = epoch + 1 >= self.epochs
            if self.ema:
                self.ema.update_attr(self.model, include=["yaml", "nc", "args", "names", "stride", "class_weights"])

            # Validation
            if self.args.val or final_epoch or self.stopper.possible_stop or self.stop:
                self.metrics, self.fitness = self.validate()

            self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
            self.stop |= self.stopper(epoch + 1, self.fitness) or final_epoch
    
            # Save model
            if self.args.save or final_epoch:
                self.save_model()
  
            gc.collect()
            torch.cuda.empty_cache()
            epoch += 1
        
        # end
        # self.final_eval() #  checkpoint 更新
        # if self.args.plots:
        #     self.plot_metrics()

    def build_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        """Construct and return dataloader."""
        assert mode in {"train", "val"}, f"Mode must be 'train' or 'val', not {mode}."
        gs = max(int(self.model.stride.max() if self.model else 0), 32)
        dataset = build_yolo_dataset(self.args, 
                                     dataset_path, 
                                     batch_size, 
                                     self.data, 
                                     mode=mode, 
                                     rect=mode == "val", 
                                     stride=gs)
        
        shuffle = mode == "train"
        if getattr(dataset, "rect", False) and shuffle:
            print("WARNING ⚠️ 'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = False

        workers = self.args.workers if mode == "train" else self.args.workers * 2
        return build_dataloader(dataset, batch_size, workers, shuffle, rank)

    def validate(self):
        """
        Runs validation on test set using self.validator.

        The returned dict is expected to contain "fitness" key.
        """
        metrics = self.validator(self)
        # metrics = self.validator(model=self.ema.ema)
        # fitness = metrics.pop("fitness", -self.loss.detach().cpu().numpy())  # use loss as fitness measure if not found
        fitness = metrics.pop("fitness")
        if not self.best_fitness or self.best_fitness < fitness:
            self.best_fitness = fitness
        return metrics, fitness
    
    def resume_training(self, ckpt):
        """Resume YOLO training from given epoch and best fitness."""
        if ckpt is None or not self.resume:
            return
        best_fitness = 0.0
        # start_epoch = ckpt.get("epoch", 1)
        start_epoch = 1
        
        if ckpt.get("optimizer", None) is not None:
            self.optimizer.load_state_dict(ckpt["optimizer"])  # optimizer
            best_fitness = ckpt["best_fitness"]
            self.warmup = False
    
        if self.epochs < start_epoch:
            self.epochs += ckpt["epoch"]  # finetune additional epochs
        self.best_fitness = best_fitness
        self.start_epoch = start_epoch
        if start_epoch > (self.epochs - self.args.close_mosaic):
            self._close_dataloader_mosaic()

    def save_model(self):
        """Save model training checkpoints with additional metadata."""
        ckpt = {
                "epoch": self.epoch,
                "best_fitness": self.best_fitness,
                # "model": deepcopy(self.ema.ema).half().state_dict(),
                "model": deepcopy(self.ema.ema).float().state_dict(),
                "updates": self.ema.updates,
                "optimizer": convert_optimizer_state_dict_to_fp16(deepcopy(self.optimizer.state_dict())),
                "train_args": vars(self.args),  # save as dict
                "train_metrics": {**self.metrics, **{"fitness": self.fitness}},
                "train_results": {k.strip(): v for k, v in pd.read_csv(self.csv).to_dict(orient="list").items()},
                "date": datetime.now().isoformat()
            }
        
        torch.save(ckpt, str(self.last))
        if self.best_fitness == self.fitness:
            torch.save(ckpt, str(self.best))

    def save_metrics(self, metrics):
        """Saves training metrics to a CSV file."""
        keys, vals = list(metrics.keys()), list(metrics.values())
        n = len(metrics) + 1  # number of cols
        s = "" if self.csv.exists() else (("%23s," * n % tuple(["epoch"] + keys)).rstrip(",") + "\n")  # header
        with open(self.csv, "a") as f:
            f.write(s + ("%23.5g," * n % tuple([self.epoch + 1] + vals)).rstrip(",") + "\n")

    def _close_dataloader_mosaic(self):
        """Update dataloaders to stop using mosaic augmentation."""
        if hasattr(self.train_loader.dataset, "mosaic"):
            self.train_loader.dataset.mosaic = False
        if hasattr(self.train_loader.dataset, "close_mosaic"):
            self.train_loader.dataset.close_mosaic(hyp=self.args)

    def final_eval(self):
        """Performs final evaluation and validation for object detection YOLO model."""
        for f in self.last, self.best:
            if f.exists():
                strip_optimizer(f)  # strip optimizers data no need
                if f is self.best:
                    self.validator.args.plots = self.args.plots
                    self.model.load(torch.load(f))
                    self.metrics = self.validator(trainer=self,
                                                  model=self.model)
                    self.metrics.pop("fitness", None)

    def preprocess_batch(self, batch):
        """Preprocesses a batch of images by scaling and converting to float."""
        batch["img"] = batch["img"].to(self.device, non_blocking=True).float() / 255
        if self.args.multi_scale:
            imgs = batch["img"]
            sz = (
                random.randrange(self.args.imgsz * 0.5, self.args.imgsz * 1.5 + self.stride)
                // self.stride
                * self.stride
            )  # size
            sf = sz / max(imgs.shape[2:])  # scale factor
            if sf != 1:
                ns = [
                    math.ceil(x * sf / self.stride) * self.stride for x in imgs.shape[2:]
                ]  # new shape (stretched to gs-multiple)
                imgs = nn.functional.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)
            batch["img"] = imgs
        return batch

    def label_loss_items(self, loss_items=None, prefix="train"):
        """
        Returns a loss dict with labelled training loss items tensor.
        """
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]  # convert tensors to 5 decimal place floats
            return dict(zip(keys, loss_items))
        else:
            return keys

if __name__ == '__main__':

    ########################################## yolov10 training job ##########################################

    # from model import DetectionModelv10
    # model_cfg = '/home/ec2-user/SageMaker/detection_models/yolov10/cfg/yolov10m.yaml'
    # train_cfg = '/home/ec2-user/SageMaker/detection_models/yolov10/cfg/train.yaml'
    # ckpt_path = None
    # ckpt_path = '/home/ec2-user/SageMaker/detection_models/yolov10/data/yolov10m.pt'
    # ckpt_path = '/home/ec2-user/SageMaker/detection_models/yolov10/data/weights/best.pt'
    # model = DetectionModelv10(cfg=model_cfg)
    # trainer = Trainer(model=model, ckpt_path=ckpt_path, train_cfg=train_cfg)
    # trainer.train()



    # ########################################## yolov8 training job ##########################################
    # from model import DetectionModelv8
    # model_cfg = '/home/ec2-user/SageMaker/detection_models/yolov10/cfg/yolov8m.yaml'
    # train_cfg = '/home/ec2-user/SageMaker/detection_models/yolov10/cfg/train.yaml'
    # # ckpt_path = '/home/ec2-user/SageMaker/detection_models/yolov10/data/v8m.pt'
    # ckpt_path = '/home/ec2-user/SageMaker/detection_models/yolov10/data/weights0/best.pt'
    # model = DetectionModelv8(cfg=model_cfg)
    # trainer = Trainer(model=model, ckpt_path=ckpt_path, train_cfg=train_cfg)
    # trainer.train()

    ########################################## yolov8 training job ##########################################
    from model import RTDETRDetectionModel
    model_cfg = '/home/ec2-user/SageMaker/detection_models/yolov10/cfg/rtdetr-l.yaml'
    train_cfg = '/home/ec2-user/SageMaker/detection_models/yolov10/cfg/train.yaml'
    # ckpt_path = '/home/ec2-user/SageMaker/detection_models/yolov10/data/v8m.pt'
    # ckpt_path = '/home/ec2-user/SageMaker/detection_models/yolov10/data/weights0/best.pt'
    ckpt_path = '/home/ec2-user/SageMaker/detection_models/yolov10/data/weights/last.pt'
    model = RTDETRDetectionModel(cfg=model_cfg)
    trainer = Trainer(model=model, ckpt_path=ckpt_path, train_cfg=train_cfg)
    trainer.train()

    """
    TODO:
    多卡训练框架：
        数据加载
        模型加载
        模型梯度收集
        变量广播
        主卡validate
    
    yolov10 validate 对float32和float16敏感; 但原始版本不敏感，原因 TODO
    yolov8 validate 不敏感
    rtdetr规整 : dataset rect参数False
    """
    
 
    """
    nohup ~/anaconda3/bin/python /home/ec2-user/SageMaker/detection_models/yolov10/train.py > /home/ec2-user/SageMaker/detection_models/yolov10/data/train_n.log 2>&1 &
    """