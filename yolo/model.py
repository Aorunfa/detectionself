import sys
from pathlib import Path
from numpy.core.multiarray import array as array
file = Path(__file__)
sys.path.append(str(file.parents[2]))
    
from typing import Union
import thop
import torch
import torch.nn as nn
from pathlib import Path
from copy import deepcopy
import contextlib
import os
import cv2
import numpy as np

from tools.torch_utils import(
    fuse_conv_and_bn,
    fuse_deconv_and_bn,
    initialize_weights,
    intersect_dicts,
    make_divisible,
    model_info,
    scale_img,
    time_sync,
    parse_model  
)


from modules import (
    Detect,
    Conv, 
    Conv2, 
    DWConv,
    ConvTranspose,
    RepConv,
    v10Detect,
    )

from tools.augment import LetterBox
from tools import ops
from tools.visul import plot_box_labl
from tools.common import yaml_load

class BaseModel(nn.Module):
    """
    The BaseModel class serves as a base class for all the models in the Ultralytics YOLO family.
    base functons:
        · forward 
        · loss creterion and loss compute
        · safe load
        · fuse
    """
    
    def forward(self, x, *args, **kwargs):
        """
        Forward pass of the model on a single scale. Wrapper for `_forward_once` method.

        Args:
            x (torch.Tensor | dict): The input image tensor or a dict including image tensor and gt labels.

        Returns:
            (torch.Tensor): The output of the network.
        """
        if isinstance(x, dict):  # for cases of training and validating while training.
            return self.loss(x, *args, **kwargs)
        
        return self.predict(x, *args, **kwargs)

    def predict(self, x, embed=None):
        """
        Perform a forward pass through the network.

        Args:
            x (torch.Tensor): The input tensor to the model.
            profile (bool):  Print the computation time of each layer if True, defaults to False.
            visualize (bool): Save the feature maps of the model if True, defaults to False.
            embed (list, optional): A list of feature vectors/embeddings to return.

        Returns:
            (torch.Tensor): The last output of the model.
        """        
        
        y, embeddings = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            x = m(x)  # run
            y.append(x if m.i in self.save else None)
            if embed and m.i in embed:
                embeddings.append(nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
                if m.i == max(embed):
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        return x

    def info(self, detailed=False, verbose=True, imgsz=640):
        """
        Prints model information.

        Args:
            detailed (bool): if True, prints out detailed information about the model. Defaults to False
            verbose (bool): if True, prints out the model information. Defaults to False
            imgsz (int): the size of the image that the model will be trained on. Defaults to 640
        """
        return model_info(self, detailed=detailed, verbose=verbose, imgsz=imgsz)

    def load(self, weights):
        """
        Load the weights into the model.

        Args:
            weights (dict | torch.nn.Module): The pre-trained weights to be loaded.
            verbose (bool, optional): Whether to log the transfer progress. Defaults to True.
        """
        if isinstance(weights, str):
            weights = torch.load(weights)
        csd = weights["model"] if isinstance(weights, dict) else weights
        csd = intersect_dicts(csd, self.state_dict())  # intersect
        self.load_state_dict(csd, strict=False)  # load
        print(f"Transferred {len(csd)}/{len(self.model.state_dict())} items from pretrained weights")

    def loss(self, batch, preds=None):
        """
        Compute loss.

        Args:
            batch (dict): Batch to compute loss on
            preds (torch.Tensor | List[torch.Tensor]): Predictions.
        """
        if not hasattr(self, "criterion"):
            self.criterion = self.init_criterion()

        preds = self.forward(batch["img"]) if preds is None else preds
        return self.criterion(preds, batch)
    
    def init_criterion(self):
        """Initialize the loss criterion for the DetectionModel."""
        return 
    
    @property
    def device(self):
        return next(self.model.parameters()).device
    
    def fuse(self, verbose=True):
        """
        Fuse the `Conv2d()` and `BatchNorm2d()` layers of the model into a single layer, in order to improve the
        computation efficiency.
        for inference
        Returns:
            (nn.Module): The fused model is returned.
        """
        if not self.is_fused():
            for m in self.model.modules():
                if isinstance(m, (Conv, Conv2, DWConv)) and hasattr(m, "bn"):
                    if isinstance(m, Conv2):
                        m.fuse_convs()
                    m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                    delattr(m, "bn")  # remove batchnorm
                    m.forward = m.forward_fuse  # update forward
                if isinstance(m, ConvTranspose) and hasattr(m, "bn"):
                    m.conv_transpose = fuse_deconv_and_bn(m.conv_transpose, m.bn)
                    delattr(m, "bn")  # remove batchnorm
                    m.forward = m.forward_fuse  # update forward
                if isinstance(m, RepConv):
                    m.fuse_convs()
                    m.forward = m.forward_fuse  # update forward
            self.info(verbose=verbose)

        return self

    def is_fused(self, thresh=10):
        """
        Check if the model has less than a certain threshold of BatchNorm layers.

        Args:
            thresh (int, optional): The threshold number of BatchNorm layers. Default is 10.

        Returns:
            (bool): True if the number of BatchNorm layers in the model is less than the threshold, False otherwise.
        """
        bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)  # normalization layers, i.e. BatchNorm2d()
        return sum(isinstance(v, bn) for v in self.modules()) < thresh  # True if < 'thresh' BatchNorm layers in model

class DetectionModelv10(BaseModel):
    def __init__(self, 
                 cfg="yolov10n.yaml", # pathlike 
                 ch=3, 
                 nc=None, 
                 verbose=True):
        """
        v10 detection model; input:
            model cfg path, inchannels, number of classes
        """
        super().__init__()
        self.yaml = cfg if isinstance(cfg, dict) else yaml_load(cfg)

        # Build model
        ch = self.yaml["ch"] = self.yaml.get("ch", ch)
        if nc and nc != self.yaml["nc"]:
            self.yaml["nc"] = nc
        self.model, self.save = parse_model(deepcopy(self.yaml), 
                                            ch=ch, 
                                            verbose=verbose)
        
        self.names = {i: f"{i}" for i in range(self.yaml["nc"])}
        self.inplace = self.yaml.get("inplace", True)

        # Build strides
        m = self.model[-1]
        m.inplace = self.inplace
        s = 256
        fn = lambda x: self.forward(x)["one2many"]
        m.stride = torch.tensor([s / x.shape[-2] for x in fn(torch.zeros(1, ch, s, s))])
        self.stride = m.stride
        m.bias_init()
     
        initialize_weights(self)
        
        if verbose:
            self.info()
            
    def init_criterion(self):
        """Initialize the loss criterion for the DetectionModel."""
        try:
            from loss import E2EDetectLoss
        except:
            from .loss import E2EDetectLoss
        return E2EDetectLoss(self)
    
    def _pre_transform(self, im, imgsz=(640, 640)):
        """
        Pre-transform input image before inference.

        Args:
            im (List(np.ndarray)): (N, 3, h, w) for tensor, [(h, w, 3) x N] for list.

        Returns:
            (list): A list of transformed images.
        """
        same_shapes = len({x.shape for x in im}) == 1
        letterbox = LetterBox(imgsz, 
                              auto=same_shapes, 
                              stride=self.stride.max().item()
                              )
        return [letterbox(image=x) for x in im]
    

    def preprocess(self, im):
        """
        Prepares input image before inference.

        Args:
            im (torch.Tensor | List(np.ndarray)): BCHW for tensor, [(HWC) x B] for list.
        """
        not_tensor = not isinstance(im, torch.Tensor)
        if not_tensor:
            im = np.stack(self._pre_transform(im))
            im = im[..., ::-1].transpose((0, 3, 1, 2))
            im = np.ascontiguousarray(im)  # contiguous
            im = torch.from_numpy(im)

        im = im.to(self.device)
        im = im.half() if getattr(self.model, 'fp16', False) == True else im.float()  # uint8 to fp16/32
        if not_tensor:
            im /= 255.
        return im


    def postprocess(self, preds, img=None, orig_imgs=None, args=None):
        """
        Post-processes predictions and returns a list of Results objects.
        preds: 模型输出
        img: 原始图片对齐到模型输入维度
        orig_imgs: 原始图片
        """
        if args is None:
            args = {'conf': 0.1,
                    'iou': 0.5,
                    'max_det': 300,
                    'classes': None,
                    'agnostic_nms': False
                    }
        if isinstance(preds, dict):
            preds = preds["one2one"]
        if isinstance(preds, tuple):
            preds = preds[0]
            preds[:, :4, :] = ops.xyxy2xywh(preds[:, :4, :].permute(0, 2, 1)).permute(0, 2, 1)
            

        # nms for preds TODO can not do this for validation
        preds = ops.non_max_suppression(
            preds,
            args['conf'],
            args['iou'],
            agnostic=args['agnostic_nms'],
            max_det=args['max_det'],
            classes=args['classes'],
        )

        if img is None and orig_imgs is None:
            return preds # for validation

        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i]
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            results.append(pred.cpu())
        return results
    
    
    @torch.no_grad()
    def inference(self, im: Union[np.array, list], post_args=None):
        # preprocess
        if not isinstance(im, list):
            ims = [im]
            im = [im]
        else:
            ims = deepcopy(im)
        ims = self.preprocess(ims)
        preds = self.predict(ims)
        preds = self.postprocess(preds, ims, im, args=post_args)
        return preds
    
class DetectionModelv8(BaseModel):
    
    def __init__(self, cfg="yolov8n.yaml", ch=3, nc=None, verbose=True):  
        """
        v8 detection model; input:
            model cfg path, inchannels, number of classes
        """
        super().__init__()
        self.yaml = cfg if isinstance(cfg, dict) else yaml_load(cfg)

        # Build model
        ch = self.yaml["ch"] = self.yaml.get("ch", ch)
        if nc and nc != self.yaml["nc"]:
            self.yaml["nc"] = nc
        self.model, self.save = parse_model(deepcopy(self.yaml), 
                                            ch=ch, 
                                            verbose=verbose) #save layer indx：用于模型融的输入
        
        self.names = {i: f"{i}" for i in range(self.yaml["nc"])}
        self.inplace = self.yaml.get("inplace", True)

        # Build strides 8/16/32
        m = self.model[-1]
        s = 256  # 2x min stride
        m.inplace = self.inplace
        forward = lambda x: self.forward(x)
        m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, ch, s, s))])
        self.stride = m.stride
        m.bias_init()
        initialize_weights(self)
        
        if verbose:
            self.info()
    
    def init_criterion(self):
        try:
            from loss import v8DetectionLoss
        except:
            from .loss import v8DetectionLoss
        return v8DetectionLoss(self) 

    def _pre_transform(self, im, imgsz=(640, 640)):
        """
        Pre-transform input image before inference.

        Args:
            im (List(np.ndarray)): (N, 3, h, w) for tensor, [(h, w, 3) x N] for list.

        Returns:
            (list): A list of transformed images.
            resize maxborder 640 and min border to be padd can be devided by maxstride
        """
        same_shapes = len({x.shape for x in im}) == 1 # true for min padding false return img size (640, 640)
        letterbox = LetterBox(imgsz, 
                              auto=same_shapes, 
                              stride=self.stride.max().item()
                              ) # 32

        return [letterbox(image=x) for x in im]
    
    def preprocess(self, im):
        """
        Prepares input image before inference.

        Args:
            im (torch.Tensor | List(np.ndarray)): BCHW for tensor, [(HWC) x B] for list.
        """
        not_tensor = not isinstance(im, torch.Tensor)
        if not_tensor:
            im = np.stack(self._pre_transform(im))
            im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
            im = np.ascontiguousarray(im)  # contiguous
            im = torch.from_numpy(im)

        im = im.to(self.device)
        im = im.half() if getattr(self.model, 'fp16', False) == True else im.float()  # uint8 to fp16/32
        if not_tensor:
            im /= 255  # 0 - 255 to 0.0 - 1.0
        return im

    def postprocess(self, preds, img=None, orig_imgs=None, args=None):
        """
        Post-processes predictions and returns a list of Results objects.
        preds: 模型输出
        img: 原始图片对齐到模型输入维度
        orig_imgs: 原始图片
        """
        if args is None:
            args = {'conf': 0.1,
                    'iou': 0.5,
                    'max_det': 300,
                    'classes': None,
                    'agnostic_nms': False
                    }

        # nms for preds TODO can not do this for validation
        preds = ops.non_max_suppression(
            preds,
            args['conf'],
            args['iou'],
            agnostic=args['agnostic_nms'],
            max_det=args['max_det'],
            classes=args['classes'],
        )

        if img is None and orig_imgs is None:
            return preds # for validation

        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i]
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            results.append(pred.cpu())
        return results
    
    @torch.no_grad()
    def inference(self, im, post_args=None):
        if not isinstance(im, list):
            ims = [im]
            im = [im]
        else:
            ims = deepcopy(im)

        ims = self.preprocess(ims)
        preds = self.predict(ims)
        preds = self.postprocess(preds, ims, im, args=post_args)
        return preds

class RTDETRDetectionModel(BaseModel):
    """
    RTDETR (Real-time DEtection and Tracking using Transformers) Detection Model class.

    This class is responsible for constructing the RTDETR architecture, defining loss functions, and facilitating both
    the training and inference processes. RTDETR is an object detection and tracking model that extends from the
    DetectionModel base class.

    Attributes:
        cfg (str): The configuration file path or preset string. Default is 'rtdetr-l.yaml'.
        ch (int): Number of input channels. Default is 3 (RGB).
        nc (int, optional): Number of classes for object detection. Default is None.
        verbose (bool): Specifies if summary statistics are shown during initialization. Default is True.

    Methods:
        init_criterion: Initializes the criterion used for loss calculation.
        loss: Computes and returns the loss during training.
        predict: Performs a forward pass through the network and returns the output.
    """

    def __init__(self, cfg="rtdetr-l.yaml", ch=3, nc=None, verbose=True):
        """
        Initialize the RTDETRDetectionModel.

        Args:
            cfg (str): Configuration file name or path.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes. Defaults to None.
            verbose (bool, optional): Print additional information during initialization. Defaults to True.
        """
        super().__init__()
        self.yaml = cfg if isinstance(cfg, dict) else yaml_load(cfg)

        # Build model
        ch = self.yaml["ch"] = self.yaml.get("ch", ch)
        if nc and nc != self.yaml["nc"]:
            self.yaml["nc"] = nc
        self.nc = self.yaml["nc"]
        self.model, self.save = parse_model(deepcopy(self.yaml), 
                                            ch=ch, 
                                            verbose=verbose) #save layer indx：用于模型融的输入
        
        self.names = {i: f"{i}" for i in range(self.yaml["nc"])}
        # self.inplace = self.yaml.get("inplace", True)

        # strides
        self.stride = torch.Tensor([32])
        if verbose:
            self.info()


    def init_criterion(self):
        """Initialize the loss criterion for the RTDETRDetectionModel."""
        try:
            from loss import RTDETRDetectionLoss
        except:
            from .loss import RTDETRDetectionLoss

        return RTDETRDetectionLoss(nc=self.nc, use_vfl=True)

    def loss(self, batch, preds=None):
        """
        Compute the loss for the given batch of data.

        Args:
            batch (dict): Dictionary containing image and label data.
            preds (torch.Tensor, optional): Precomputed model predictions. Defaults to None.

        Returns:
            (tuple): A tuple containing the total loss and main three losses in a tensor.
        """
        if not hasattr(self, "criterion"):
            self.criterion = self.init_criterion()

        img = batch["img"]
        # NOTE: preprocess gt_bbox and gt_labels to list.
        bs = len(img)
        batch_idx = batch["batch_idx"]
        gt_groups = [(batch_idx == i).sum().item() for i in range(bs)]
        targets = {
            "cls": batch["cls"].to(img.device, dtype=torch.long).view(-1),
            "bboxes": batch["bboxes"].to(device=img.device),
            "batch_idx": batch_idx.to(img.device, dtype=torch.long).view(-1),
            "gt_groups": gt_groups,
        }

        preds = self.predict(img, batch=targets) if preds is None else preds
        dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta = preds if self.training else preds[1]
        if dn_meta is None:
            dn_bboxes, dn_scores = None, None
        else:
            dn_bboxes, dec_bboxes = torch.split(dec_bboxes, dn_meta["dn_num_split"], dim=2)
            dn_scores, dec_scores = torch.split(dec_scores, dn_meta["dn_num_split"], dim=2)

        dec_bboxes = torch.cat([enc_bboxes.unsqueeze(0), dec_bboxes])  # (7, bs, 300, 4)
        dec_scores = torch.cat([enc_scores.unsqueeze(0), dec_scores])

        loss = self.criterion(
            (dec_bboxes, dec_scores), targets, dn_bboxes=dn_bboxes, dn_scores=dn_scores, dn_meta=dn_meta
        )
        # NOTE: There are like 12 losses in RTDETR, backward with all losses but only show the main three losses.
        return sum(loss.values()), torch.as_tensor(
            [loss[k].detach() for k in ["loss_giou", "loss_class", "loss_bbox"]], device=img.device
        )


    def postprocess(self, preds, orig_imgs=None, args=None):
        """
        Postprocess the raw predictions from the model to generate bounding boxes and confidence scores.

        The method filters detections based on confidence and class if specified in `self.args`.

        Args:
            preds (list): List of [predictions, extra] from the model.
            img (torch.Tensor): Processed input images.
            orig_imgs (list or torch.Tensor): Original, unprocessed images.

        Returns:
            (list[Results]): A list of Results objects containing the post-processed bounding boxes, confidence scores,
                and class labels.
        """
        if args is None:
            args = {'conf': 0.1,
                    'classes': None,
                    }
        
        if not isinstance(preds, (list, tuple)):  # list for PyTorch inference but list[0] Tensor for export inference
            preds = [preds, None]

        nd = preds[0].shape[-1]
        bboxes, scores = preds[0].split((4, nd - 4), dim=-1)

        # TODO return for training
        if orig_imgs is None:
            bboxes = ops.xywh2xyxy(bboxes)
            bboxes[..., [0, 2]] *= 640
            bboxes[..., [1, 3]] *= 640

            conf, cls = scores.max(dim=-1)
            preds = torch.cat([bboxes, 
                               conf.unsqueeze(-1), 
                               cls.unsqueeze(-1)], dim=-1)
            
            return  preds # for trainning and validate

        #########################
        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for bbox, score, orig_img in zip(bboxes, scores, orig_imgs):  # (300, 4)
            bbox = ops.xywh2xyxy(bbox)
            max_score, cls = score.max(-1, keepdim=True)  # (300, 1)
            idx = max_score.squeeze(-1) > args['conf']  # (300, )
            if args['classes'] is not None:
                idx = (cls == torch.tensor(args['classes'], device=cls.device)).any(1) & idx
            pred = torch.cat([bbox, max_score, cls], dim=-1)[idx]  # filter
            oh, ow = orig_img.shape[:2]
            pred[..., [0, 2]] *= ow
            pred[..., [1, 3]] *= oh
            # results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred))
            results.append(pred)
        return results

    def _pre_transform(self, im, imgsz=(640, 640)):
        """
        Pre-transforms the input images before feeding them into the model for inference. The input images are
        letterboxed to ensure a square aspect ratio and scale-filled. The size must be square(640) and scaleFilled.

        Args:
            im (list[np.ndarray] |torch.Tensor): Input images of shape (N,3,h,w) for tensor, [(h,w,3) x N] for list.

        Returns:
            (list): List of pre-transformed images ready for model inference.
        """
        letterbox = LetterBox(imgsz, auto=False, scaleFill=True)
        return [letterbox(image=x) for x in im]    


    def preprocess(self, im):
        """
        Prepares input image before inference.

        Args:
            im (torch.Tensor | List(np.ndarray)): BCHW for tensor, [(HWC) x B] for list.
        """
        not_tensor = not isinstance(im, torch.Tensor)
        if not_tensor:
            im = np.stack(self._pre_transform(im))
            im = im[..., ::-1].transpose((0, 3, 1, 2)) 
            im = np.ascontiguousarray(im)  # contiguous
            im = torch.from_numpy(im)

        im = im.to(self.device)
        im = im.half() if getattr(self.model, 'fp16', False) == True else im.float()  # uint8 to fp16/32
        if not_tensor:
            im /= 255  # 0 - 255 to 0.0 - 1.0
        return im
    
    

    def predict(self, x, embed=None, batch=None):
        """
        Perform a forward pass through the model.

        Args:
            x (torch.Tensor): The input tensor.
            profile (bool, optional): If True, profile the computation time for each layer. Defaults to False.
            visualize (bool, optional): If True, save feature maps for visualization. Defaults to False.
            batch (dict, optional): Ground truth data for evaluation. Defaults to None.
            augment (bool, optional): If True, perform data augmentation during inference. Defaults to False.
            embed (list, optional): A list of feature vectors/embeddings to return.

        Returns:
            (torch.Tensor): Model's output tensor.
        """
        y, dt, embeddings = [], [], []  # outputs
        for m in self.model[:-1]:  # except the head part
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
            if embed and m.i in embed:
                embeddings.append(nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
                if m.i == max(embed):
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        head = self.model[-1]
        x = head([y[j] for j in head.f], batch)  # head inference
        return x
    
    @torch.no_grad()
    def inference(self, im, args:dict=None):
        if not isinstance(im, list):
            ims = [im]
            im = [im]
        else:
            ims = deepcopy(im)

        ims = self.preprocess(ims)
        preds = self.predict(ims)
        preds = self.postprocess(preds, orig_imgs=im, args=args)
        return preds


class PoseModelv8(DetectionModelv8):
    """YOLOv8 pose model."""

    def __init__(self, cfg="yolov8n-pose.yaml", ch=3, nc=None, data_kpt_shape=(None, None), verbose=True):
        """Initialize YOLOv8 Pose model."""
        if not isinstance(cfg, dict):
            cfg = yaml_load(cfg)  # load model YAML
        if any(data_kpt_shape) and list(data_kpt_shape) != list(cfg["kpt_shape"]):
            print(f"Overriding model.yaml kpt_shape={cfg['kpt_shape']} with kpt_shape={data_kpt_shape}")
            cfg["kpt_shape"] = data_kpt_shape
        self.kpt_shape = cfg["kpt_shape"]
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        try:
            from loss import v8PoseLoss
        except:
            from .loss import v8PoseLoss

        return v8PoseLoss(self)
    
    def postprocess(self, preds, img=None, orig_imgs=None, args=None):
        """
        Post-processes predictions and returns a list of Results objects.
        preds: 模型输出
        img: 原始图片对齐到模型输入维度
        orig_imgs: 原始图片
        """
        if args is None:
            args = {'conf': 0.1,
                    'iou': 0.5,
                    'max_det': 300,
                    'classes': None,
                    'agnostic_nms': False
                    }

        # nms for preds
        preds = ops.non_max_suppression(
            preds,
            args['conf'],
            args['iou'],
            agnostic=args['agnostic_nms'],
            max_det=args['max_det'],
            classes=args['classes'],
            nc=len(self.names)
        )
        
        if img is None and orig_imgs is None:
            return preds # for validation

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i]        
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape).round()
            pred_kpts = pred[:, 6:].view(len(pred), *self.kpt_shape) if len(pred) else pred[:, 6:]
            pred_kpts = ops.scale_coords(img.shape[2:], pred_kpts, orig_img.shape)
            results.append(pred_kpts.cpu())
            
        return results
    
if __name__ == '__main__':
    ############## yolov10 detection
    # yaml_path = '/chaofeng/yolo/code/cfg/yolov10m.yaml'
    # model = DetectionModelv10(cfg=yaml_path)
    # #ckpt_path = '/chaofeng/yolo/code/data/v10m.pt'
    # # ckpt_path = '/chaofeng/yolo/code/data_scrach/weights/best.pt'
    # ckpt_path = '/chaofeng/yolov10m_export.pt'
    # ckpt = torch.load(ckpt_path)
    # model.load(ckpt)
    # model.eval()
    # model.fuse()

    # yaml_path = '/home/ec2-user/SageMaker/codehouse/self/yolov10/cfg/yolov8-pose.yaml'
    # model = PoseModelv8(cfg=yaml_path)
    # ckpt_path = '/home/ec2-user/SageMaker/codehouse/self/yolov10/data/v8post_l.pt'
    # ckpt = torch.load(ckpt_path)
    # model.load(ckpt)
    # model.eval()
    # model.fuse()

    ############## yolov8 detection
    # yaml_path = '/chaofeng/yolo/code/cfg/yolov8m.yaml'
    # model = DetectionModelv8(cfg=yaml_path)
    # ckpt_path = '/chaofeng/yolo/code/data_scrach_v8/weights/best.pt'
    # ckpt = torch.load(ckpt_path)
    # model.load(ckpt)
    # model.eval()
    # model.fuse()

    ############# yolov11 detection
    yaml_path = '/chaofeng/yolo/code/cfg/yolov11m.yaml'
    model = DetectionModelv8(cfg=yaml_path)
    ckpt_path = '/chaofeng/yolo/code/data_scrach_v11/weights/best.pt'
    ckpt = torch.load(ckpt_path)
    model.load(ckpt)
    model.eval()
    model.fuse()

    
    import cv2
    im_path = '/chaofeng/yolo/code/assets/bus.jpg'
    ori_im = cv2.imread(im_path)
 
    args = {'conf': 0.1,
        'iou': 0.5,
        'max_det': 100,
        'classes': None,
        'agnostic_nms': False
        }
    preds = model.inference(ori_im, args)
    print('preds', preds)


    # # save
    save_dir = r'/chaofeng/yolo/code'
    plot_box_labl(preds, [ori_im], save_dir=save_dir)