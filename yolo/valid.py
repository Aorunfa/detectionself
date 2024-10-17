# Ultralytics YOLO 🚀, AGPL-3.0 license

import os
from pathlib import Path

import numpy as np
import torch
from tools import ops

from tools.metrics import (
    ConfusionMatrix, 
    DetMetrics, 
    box_iou)

import time
from pathlib import Path
from functools import partial
import numpy as np
import torch
from tools.common import TQDM, check_imgsz
from tools.ops import Profile
from tools.torch_utils import smart_inference_mode
from copy import deepcopy


class DetectionValidator(object):
    """
    BaseValidator.

    A base class for creating validators.

    Attributes:
        args (SimpleNamespace): Configuration for the validator.
        dataloader (DataLoader): Dataloader to use for validation.
        pbar (tqdm): Progress bar to update during validation.
        model (nn.Module): Model to validate.
        data (dict): Data dictionary.
        device (torch.device): Device to use for validation.
        batch_i (int): Current batch index.
        training (bool): Whether the model is in training mode.
        names (dict): Class names.
        seen: Records the number of images seen so far during validation.
        stats: Placeholder for statistics during validation.
        confusion_matrix: Placeholder for a confusion matrix.
        nc: Number of classes.
        iouv: (torch.Tensor): IoU thresholds from 0.50 to 0.95 in spaces of 0.05.
        jdict (dict): Dictionary to store JSON validation results.
        speed (dict): Dictionary with keys 'preprocess', 'inference', 'loss', 'postprocess' and their respective
                      batch processing times in milliseconds.
        save_dir (Path): Directory to save results.
        plots (dict): Dictionary to store plots for visualization.
        callbacks (dict): Dictionary to store various callback functions.
    """

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None):
        """
        Initializes a BaseValidator instance.

        Args:
            dataloader (torch.utils.data.DataLoader): Dataloader to be used for validation.
            save_dir (Path, optional): Directory to save results.
            pbar (tqdm.tqdm): Progress bar for displaying progress.
            args (SimpleNamespace): Configuration for the validator.
            _callbacks (dict): Dictionary to store various callback functions.
        """
        if isinstance(args, dict):
            from tools.common import IterableSimpleNamespace
            args = IterableSimpleNamespace(**args)
        self.args = deepcopy(args)
        self.dataloader = dataloader   
        self.speed = {"preprocess": 0.0, 
                      "inference": 0.0, 
                      "loss": 0.0, 
                      "postprocess": 0.0}

        self.save_dir = save_dir or self.args.save_dir
        if not isinstance(self.save_dir, Path):
            self.save_dir = Path(self.save_dir)
        
        (self.save_dir / "labels" if self.args.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)
        
        if self.args.conf is None:
            self.args.conf = 0.001  # default
        self.args.imgsz = check_imgsz(self.args.imgsz, max_dim=1)

        self.plots = {}
        
        self.args.task = "detect"
        self.metrics = DetMetrics(save_dir=self.save_dir, on_plot=self.on_plot)
        self.iouv = torch.linspace(0.5, 0.95, 10)  # IoU vector for mAP@0.5:0.95
        self.niou = self.iouv.numel()
        self.lb = []  # for autolabelling

    @smart_inference_mode()
    def __call__(self, trainer=None, model=None):
        """
        Supports validation of a pre-trained model if passed or a model being trained if trainer is passed (trainer
        gets priority).
        for training process
        """
        self.training = trainer is not None
        model = trainer.ema.ema if self.training else model        
        self.device = next(model.parameters()).device       
        self.args.half = False # update for v10
        # self.args.half = self.device.type != "cpu" if self.training else False  # force FP16 val during training

        model = model.half() if self.args.half else model.float()
        model.eval()
      
        if self.training:
            self.loss = torch.zeros(len(trainer.loss_names), 
                                         device=trainer.device, 
                                         dtype=torch.float32)
        else:
            self.loss = torch.tensor([0., 0., 0.]) 
        
        # init postprocess
        if hasattr(model, 'postprocess'):
            self.postprocess = partial(model.postprocess, args=dict(self.args))
        # if hasattr(model, 'preprocess'):
        #     # self.preprocess = partial(model.preprocess, self.args.imgsz)
        #     self.preprocess = model.preprocess

        
        # timing
        dt = (
            Profile(device=self.device),
            Profile(device=self.device),
            Profile(device=self.device),
            Profile(device=self.device),
        )
        bar = TQDM(self.dataloader, desc=self.get_desc(), total=len(self.dataloader))
        self.init_metrics(model)

        self.jdict = []  # empty before each val
        for i, batch in enumerate(bar):

            # Preprocess
            with dt[0]:
                batch = self.preprocess(batch)

            # Inference
            with dt[1]:
                # preds = model(batch["img"], augment=augment)
                preds = model(batch["img"])


            # Loss
            with dt[2]:
                if self.training:
                    self.loss += model.loss(batch, preds)[1]

            # Postprocess
            with dt[3]:
                preds = self.postprocess(preds)

            self.update_metrics(preds, batch)

        stats = self.get_stats()
        # self.check_stats(stats)
        self.speed = dict(zip(self.speed.keys(), (x.t / len(self.dataloader.dataset) * 1e3 for x in dt)))
        self.finalize_metrics()
        self.print_results()
        # self.run_callbacks("on_val_end")
        # model.float()
        if self.training:
            results = {**stats, **trainer.label_loss_items(self.loss.cpu() / len(self.dataloader), prefix="val")}
            return {k: round(float(v), 5) for k, v in results.items()}
        return stats

    def match_predictions(self, pred_classes, true_classes, iou, use_scipy=False):
        """
        Matches predictions to ground truth objects (pred_classes, true_classes) using IoU.

        Args:
            pred_classes (torch.Tensor): Predicted class indices of shape(N,).
            true_classes (torch.Tensor): Target class indices of shape(M,).
            iou (torch.Tensor): An NxM tensor containing the pairwise IoU values for predictions and ground of truth
            use_scipy (bool): Whether to use scipy for matching (more precise).

        Returns:
            (torch.Tensor): Correct tensor of shape(N,10) for 10 IoU thresholds.
        """
        # Dx10 matrix, where D - detections, 10 - IoU thresholds
        correct = np.zeros((pred_classes.shape[0], self.iouv.shape[0])).astype(bool)
        # LxD matrix where L - labels (rows), D - detections (columns)
        correct_class = true_classes[:, None] == pred_classes
        iou = iou * correct_class  # zero out the wrong classes
        iou = iou.cpu().numpy()
        for i, threshold in enumerate(self.iouv.cpu().tolist()):
            if use_scipy:
                # WARNING: known issue that reduces mAP in https://github.com/ultralytics/ultralytics/pull/4708
                import scipy  # scope import to avoid importing for all commands

                cost_matrix = iou * (iou >= threshold)
                if cost_matrix.any():
                    labels_idx, detections_idx = scipy.optimize.linear_sum_assignment(cost_matrix, maximize=True)
                    valid = cost_matrix[labels_idx, detections_idx] > 0
                    if valid.any():
                        correct[detections_idx[valid], i] = True
            else:
                matches = np.nonzero(iou >= threshold)  # IoU > threshold and classes match
                matches = np.array(matches).T
                if matches.shape[0]:
                    if matches.shape[0] > 1:
                        matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                        # matches = matches[matches[:, 2].argsort()[::-1]]
                        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                    correct[matches[:, 1].astype(int), i] = True
        return torch.tensor(correct, dtype=torch.bool, device=pred_classes.device)

    def on_plot(self, name, data=None):
        """Registers plots (e.g. to be consumed in callbacks)"""
        self.plots[Path(name)] = {"data": data, "timestamp": time.time()}

    def preprocess(self, batch):
        """Preprocesses batch of images for YOLO training."""
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        batch["img"] = (batch["img"].half() if self.args.half else batch["img"].float()) / 255
        for k in ["batch_idx", "cls", "bboxes"]:
            batch[k] = batch[k].to(self.device)
        return batch

    def init_metrics(self, model):
        self.names = model.names
        self.nc = len(model.names)
        self.metrics.names = self.names
        self.metrics.plot = self.args.plots
        self.confusion_matrix = ConfusionMatrix(nc=self.nc, conf=self.args.conf)
        self.seen = 0
        self.jdict = []
        self.stats = dict(tp=[], conf=[], pred_cls=[], target_cls=[])

    def get_desc(self):
        """Return a formatted string summarizing class metrics of YOLO model."""
        return ("%22s" + "%11s" * 6) % ("Class", "Images", "Instances", "Box(P", "R", "mAP50", "mAP50-95)")

    def postprocess(self, preds): # for yolov8
        """Apply Non-maximum suppression to prediction outputs."""
        return ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            labels=self.lb,
            multi_label=True,
            agnostic=self.args.single_cls,
            max_det=self.args.max_det,
        )

    def _prepare_batch(self, si, batch):
        """Prepares a batch of images and annotations for validation."""
        idx = batch["batch_idx"] == si
        cls = batch["cls"][idx].squeeze(-1)
        bbox = batch["bboxes"][idx]
        ori_shape = batch["ori_shape"][si]
        imgsz = batch["img"].shape[2:]
        ratio_pad = batch["ratio_pad"][si]
        if len(cls):
            bbox = ops.xywh2xyxy(bbox) * torch.tensor(imgsz, device=self.device)[[1, 0, 1, 0]]  # target boxes
            ops.scale_boxes(imgsz, bbox, ori_shape, ratio_pad=ratio_pad)  # native-space labels
        return {"cls": cls, "bbox": bbox, "ori_shape": ori_shape, "imgsz": imgsz, "ratio_pad": ratio_pad}

    def _prepare_pred(self, pred, pbatch):
        """Prepares a batch of images and annotations for validation."""
        predn = pred.clone()
        ops.scale_boxes(
            pbatch["imgsz"], predn[:, :4], pbatch["ori_shape"], ratio_pad=pbatch["ratio_pad"]
        )  # native-space pred
        return predn

    def update_metrics(self, preds, batch):
        """Metrics."""
        for si, pred in enumerate(preds):
            self.seen += 1
            npr = len(pred)
            stat = dict(
                conf=torch.zeros(0, device=self.device),
                pred_cls=torch.zeros(0, device=self.device),
                tp=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device),
            )
            pbatch = self._prepare_batch(si, batch)
            cls, bbox = pbatch.pop("cls"), pbatch.pop("bbox")
            nl = len(cls)
            stat["target_cls"] = cls
            if npr == 0:
                if nl:
                    for k in self.stats.keys():
                        self.stats[k].append(stat[k])
                    if self.args.plots:
                        self.confusion_matrix.process_batch(detections=None, gt_bboxes=bbox, gt_cls=cls)
                continue

            # Predictions
            if self.args.single_cls:
                pred[:, 5] = 0

            predn = self._prepare_pred(pred, pbatch)
            stat["conf"] = predn[:, 4]
            stat["pred_cls"] = predn[:, 5]

            # Evaluate
            if nl:
                stat["tp"] = self._process_batch(predn, bbox, cls)
                if self.args.plots:
                    self.confusion_matrix.process_batch(predn, bbox, cls)
            for k in self.stats.keys():
                self.stats[k].append(stat[k])

    def finalize_metrics(self, *args, **kwargs):
        """Set final values for metrics speed and confusion matrix."""
        self.metrics.speed = self.speed
        self.metrics.confusion_matrix = self.confusion_matrix

    def get_stats(self):
        """Returns metrics statistics and results dictionary."""
        stats = {k: torch.cat(v, 0).cpu().numpy() for k, v in self.stats.items()}  # to numpy
        if len(stats) and stats["tp"].any():
            self.metrics.process(**stats)
        self.nt_per_class = np.bincount(
            stats["target_cls"].astype(int), minlength=self.nc
        )  # number of targets per class
        return self.metrics.results_dict

    def print_results(self):
        """Prints training/validation set metrics per class."""
        pf = "%22s" + "%11i" * 2 + "%11.3g" * len(self.metrics.keys)  # print format
        # LOGGER.info(pf % ("all", self.seen, self.nt_per_class.sum(), *self.metrics.mean_results()))
        
        print(pf % ("all", self.seen, self.nt_per_class.sum(), *self.metrics.mean_results()))
        if self.nt_per_class.sum() == 0:
            print(f"WARNING ⚠️ no labels found in {self.args.task} set, can not compute metrics without labels")
            # LOGGER.warning(f"WARNING ⚠️ no labels found in {self.args.task} set, can not compute metrics without labels")

        # Print results per class
        if self.args.verbose and not self.training and self.nc > 1 and len(self.stats):
            for i, c in enumerate(self.metrics.ap_class_index):
                print(pf % (self.names[c], self.seen, self.nt_per_class[c], *self.metrics.class_result(i)))
                # LOGGER.info(pf % (self.names[c], self.seen, self.nt_per_class[c], *self.metrics.class_result(i)))

    def _process_batch(self, detections, gt_bboxes, gt_cls):
        """
        Return correct prediction matrix.

        Args:
            detections (torch.Tensor): Tensor of shape [N, 6] representing detections.
                Each detection is of the format: x1, y1, x2, y2, conf, class.
            labels (torch.Tensor): Tensor of shape [M, 5] representing labels.
                Each label is of the format: class, x1, y1, x2, y2.

        Returns:
            (torch.Tensor): Correct prediction matrix of shape [N, 10] for 10 IoU levels.
        """
        iou = box_iou(gt_bboxes, detections[:, :4])
        return self.match_predictions(detections[:, 5], gt_cls, iou)

    def save_one_txt(self, predn, save_conf, shape, file):
        """Save YOLO detections to a txt file in normalized coordinates in a specific format."""
        gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh
        for *xyxy, conf, cls in predn.tolist():
            xywh = (ops.xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
            line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
            with open(file, "a") as f:
                f.write(("%g " * len(line)).rstrip() % line + "\n")

    def pred_to_json(self, predn, filename):
        """Serialize YOLO predictions to COCO json format."""
        stem = Path(filename).stem
        image_id = int(stem) if stem.isnumeric() else stem
        box = ops.xyxy2xywh(predn[:, :4])  # xywh
        box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
        for p, b in zip(predn.tolist(), box.tolist()):
            self.jdict.append(
                {
                    "image_id": image_id,
                    "category_id": self.class_map[int(p[5])]
                    + (1 if self.is_lvis else 0),  # index starts from 1 if it's lvis
                    "bbox": [round(x, 3) for x in b],
                    "score": round(p[4], 5),
                }
            )



if __name__ == '__main__':
    # # v10 model
    # from model import DetectionModelv10
    # model_cfg = '/home/ec2-user/SageMaker/detection_models/yolov10/cfg/yolov10m.yaml'
    # train_cfg = '/home/ec2-user/SageMaker/detection_models/yolov10/cfg/train.yaml'
    # ckpt_path = '/home/ec2-user/SageMaker/detection_models/yolov10/data/yolov10m.pt'
    # # ckpt_path = '/home/ec2-user/SageMaker/detection_models/yolov10/data/weights/last.pt'
    
    # model = DetectionModelv10(cfg=model_cfg)
    # model.load(torch.load(ckpt_path))
    # model = model.float()
    # model.eval()
    # model.fuse()
    # model.to('cuda')
    
    # v8 test
    from model import DetectionModelv8
    model_cfg = '/home/ec2-user/SageMaker/detection_models/yolov10/cfg/yolov8m.yaml'
    train_cfg = '/home/ec2-user/SageMaker/detection_models/yolov10/cfg/train.yaml'
    ckpt_path = None
    ckpt_path = '/home/ec2-user/SageMaker/detection_models/yolov10/data/v8m.pt'
    
    model = DetectionModelv8(cfg=model_cfg)
    model.load(torch.load(ckpt_path))
    model.eval()
    model.fuse()
    model.to('cuda')

    # ## rtdetr
    # from model import RTDETRDetectionModel
    # model_cfg = '/home/ec2-user/SageMaker/detection_models/yolov10/cfg/rtdetr-l.yaml'
    # train_cfg = '/home/ec2-user/SageMaker/detection_models/yolov10/cfg/train.yaml'
    # ckpt_path = '/home/ec2-user/SageMaker/rtdetr_l.pt'
    
    # model = RTDETRDetectionModel(cfg=model_cfg)
    # model.load(torch.load(ckpt_path))
    # model.eval()
    # model.fuse()
    # model.to('cuda')


    # args
    from tools.common import yaml_load
    args = yaml_load(train_cfg)

    from copy import deepcopy

    
    # set dataset
    from yolo.code.dataset_old import YoloDataset
    from loader import build_dataloader
    img_path = '/home/ec2-user/SageMaker/detection_models/coco/images/val2017'
    label_path = '/home/ec2-user/SageMaker/detection_models/coco/labels/val2017'
    data_dict = yaml_load('/home/ec2-user/SageMaker/detection_models/yolov10/cfg/coco.yaml')
    dataset = YoloDataset(
        img_path=img_path,
        label_path=label_path,
        imgsz=640,
        batch_size=16,
        augment=False,
        rect=True, # True, False for rtdetr
        stride=32,
        data=data_dict
    )
    test_loader = build_dataloader(dataset, 
                                        batch=16,
                                        workers=8,
                                        shuffle=False)
    
    
    ### validate
    save_dir = '/home/ec2-user/SageMaker/detection_models'
    vd = DetectionValidator(test_loader, save_dir=save_dir, args=args)
    metrics = vd(model=model)
    print(metrics)
