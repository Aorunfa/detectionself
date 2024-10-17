
import sys
from pathlib import Path

file = Path(__file__)
sys.path.append(str(file.parents[2]))

from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.distributed

from tools.data_utils import NUM_THREADS, IMG_FORMATS
from tools.common import TQDM
from tools.logger import LOGGER

from tools.ops import resample_segments

import math
import os
import random
from copy import deepcopy
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import psutil
from torch.utils.data import Dataset

from tools.augment import(
    Compose,
    Format,
    LetterBox,
    v8_transforms
)
from tools.instance import Instances
from tools.ops import xyxy2xywh

from tools.data_utils import (
    get_hash,
    load_dataset_cache_file,
    save_dataset_cache_file,
    exif_size,
    verify_image_label
)

LOCAL_RANK = int(os.environ['LOCAL_RANK'])
class YoloDataset(Dataset):
    """
    Base dataset class for loading and processing image data.

    Args:
        img_path (str): Path to the folder containing images.
        imgsz (int, optional): Image size. Defaults to 640.
        cache (bool, optional): Cache images to RAM or disk during training. Defaults to False; npy saving for imgs
        augment (bool, optional): If True, data augmentation is applied. Defaults to True.
        hyp (dict, optional): Hyperparameters to apply data augmentation. Defaults to None.
        prefix (str, optional): Prefix to print in log messages. Defaults to ''.
        rect (bool, optional): If True, rectangular training is used. Defaults to False.
        batch_size (int, optional): Size of batches. Defaults to None.
        stride (int, optional): Stride. Defaults to 32.
        pad (float, optional): Padding. Defaults to 0.0.
        single_cls (bool, optional): If True, single class training is used. Defaults to False.
        classes (list): List of included classes. Default is None.
        fraction (float): Fraction of dataset to utilize. Default is 1.0 (use all data).

    Attributes:
        im_files (list): List of image file paths.
        labels (list): List of label data dictionaries.
        ni (int): Number of images in the dataset.
        ims (list): List of loaded images.
        npy_files (list): List of numpy file paths.
        transforms (callable): Image transformation function.
    """

    def __init__(
        self,
        img_path,
        label_path,
        imgsz=640,
        cache=False,
        augment=True,
        hyp={},
        prefix="",
        rect=False,
        batch_size=16,
        stride=32,
        pad=0.,
        single_cls=False,
        classes: Optional[list]=None, # obj class for need
        data: dict=None,              # data yaml_path load dict
        task="detect",
        mode='train'
    ):
        """Initialize BaseDataset with given configuration and options."""
        super().__init__()
        self.use_segments = task == "segment"
        self.use_keypoints = task == "pose"
        self.use_obb = task == "obb"
        self.data = data
        assert not (self.use_segments and self.use_keypoints), "Can not use both segments and keypoints."

        self.img_path = img_path
        self.imgsz = imgsz
        self.augment = augment
        self.single_cls = single_cls
        self.prefix = prefix
        self.rect = rect
        self.batch_size = batch_size
        self.stride = stride
        self.pad = pad

        self.build_img_label(img_path, label_path, include_class=classes)
        self.ni = len(self.labels)  # num of imgs

        if self.rect:
            assert self.batch_size is not None
            self.set_rectangle() # minmize resize loss for batch 

        # Transforms
        self.transforms = self.build_transforms(hyp=hyp)

    def build_img_label(self, img_path, label_path, include_class):
        def scran_files(img_path, label_path):
            """
            Read image files.
            img_path dirlike path
            """
            import pandas as pd
            df_im = pd.DataFrame(data={'im': os.listdir(img_path)})
            df_im['key'] = df_im['im'].apply(lambda x: x.split('.')[0])
            df_im['suffix'] = df_im['im'].apply(lambda x: x.split('.')[-1])
            
            df_lb = pd.DataFrame(data={'lb': os.listdir(label_path)})
            df_lb['key'] = df_lb['lb'].apply(lambda x: x.split('.')[0])
            df_im = pd.merge(df_im, df_lb, on='key', how='left')
            df_im = df_im[(~df_im['lb'].isna()) & (df_im['suffix'].isin(IMG_FORMATS))]
            im_files = [os.path.join(img_path, f) for f in df_im['im']]
            lb_files = [os.path.join(label_path, f) for f in df_im['lb']]
            return im_files, lb_files
        self.im_files, self.label_files = scran_files(img_path, label_path)
        self.labels = self.load_labels()
        self.update_labels(include_class=include_class)
        return

    def load_labels(self):
        """
        Returns dictionary of labels for YOLO training.
        load from catch first
        """
        cache_path = Path(self.label_files[0]).parent.with_suffix(".cache")

        try:
            import gc
            gc.disable() 
            cache, exists = np.load(str(cache_path), allow_pickle=True).item(), True 
            gc.enable()
            assert cache["hash"] == get_hash(self.label_files + self.im_files)  # identical hash
        except (FileNotFoundError, AssertionError, AttributeError):
            cache, exists = self.cache_labels(cache_path), False  # run cache ops

        nf, nm, ne, nc, n = cache.pop("results")  # found, missing, empty, corrupt, total
        if exists and LOCAL_RANK in {-1, 0}:
            LOGGER.info(f"Scanning {cache_path}... {nf} images, {nm + ne} backgrounds, {nc} corrupt")
            
        # Read cache
        [cache.pop(k) for k in ("hash", "msgs")]
        labels = cache["labels"]
        if not labels:
            LOGGER.warning(f"WARNING ⚠️ No images found in {cache_path}, training may not work correctly.")
        if len(labels) == 0:
            LOGGER.warning(f"WARNING ⚠️ No labels found in {cache_path}, training may not work correctly")
        self.im_files = [lb["im_file"] for lb in labels]  # for img read
        return labels
    
    def cache_labels(self, path=Path("./labels.cache")):
        """
        Cache dataset labels, check images and read shapes.

        Args:
            path (Path): Path where to save the cache file. Default is Path('./labels.cache').

        Returns:
            (dict): labels.
        """
        x = {"labels": []}
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
        desc = f"{self.prefix}Scanning {path.parent / path.stem}..."
        total = len(self.im_files)
        nkpt, ndim = self.data.get("kpt_shape", (0, 0))
        if self.use_keypoints and (nkpt <= 0 or ndim not in {2, 3}):
            raise ValueError(
                "'kpt_shape' in data.yaml missing or incorrect. Should be a list with [number of "
                "keypoints, number of dims (2 for x,y or 3 for x,y,visible)], i.e. 'kpt_shape: [17, 3]'"
            )
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(
                func=verify_image_label,
                iterable=zip(
                    self.im_files,
                    self.label_files,
                    repeat(self.prefix),
                    repeat(len(self.data["names"])),
                ),
            )
            
            pbar = TQDM(results, desc=desc, total=total)
            for im_file, lb, shape, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                
                if im_file:
                    x["labels"].append(
                        {
                            "im_file": im_file,
                            "shape": shape,
                            "cls": lb[:, 0:1],  
                            "bboxes": lb[:, 1:], 
                            "normalized": True,
                            "bbox_format": "xywh",
                        }
                    )
                if msg:
                    msgs.append(msg)
                pbar.desc = f"{desc} {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            pbar.close()

        if msgs:
            LOGGER.info("\n".join(msgs))
        if nf == 0:
            LOGGER.warning(f"{self.prefix}WARNING ⚠️ No labels found in {path}")
        x["hash"] = get_hash(self.label_files + self.im_files)
        x["results"] = nf, nm, ne, nc, len(self.im_files)
        x["msgs"] = msgs  # warnings
        save_dataset_cache_file(self.prefix, path, x)
        return x

    def update_labels(self, include_class: Optional[list]):
        """Update labels to include only these classes (optional)."""
        include_class_array = np.array(include_class).reshape(1, -1)
        for i in range(len(self.labels)):
            if include_class is not None:
                cls = self.labels[i]["cls"]
                bboxes = self.labels[i]["bboxes"]
                segments = self.labels[i]["segments"]
                keypoints = self.labels[i]["keypoints"]
                j = (cls == include_class_array).any(1)
                self.labels[i]["cls"] = cls[j]
                self.labels[i]["bboxes"] = bboxes[j]
                if segments:
                    self.labels[i]["segments"] = [segments[si] for si, idx in enumerate(j) if idx]
                if keypoints is not None:
                    self.labels[i]["keypoints"] = keypoints[j]
            if self.single_cls:
                self.labels[i]["cls"][:, 0] = 0

    def set_rectangle(self):
        """
        Sets the shape of bounding boxes for YOLO detections as rectangles.
        resize stratege for minizing img info loss  
        """
        bi = np.floor(np.arange(self.ni) / self.batch_size).astype(int)  # batch index
        nb = bi[-1] + 1  # num of batches

        s = np.array([x.pop("shape") for x in self.labels])
        ar = s[:, 0] / s[:, 1]  # h/w
        irect = ar.argsort()
        # update img and label sort
        self.im_files = [self.im_files[i] for i in irect]
        self.labels = [self.labels[i] for i in irect]
        ar = ar[irect]

        # set training image shapes
        shapes = [[1, 1]] * nb
        for i in range(nb):
            ari = ar[bi == i] # a batch ratio
            mini, maxi = ari.min(), ari.max()
            if maxi < 1:
                shapes[i] = [maxi, 1]
            elif mini > 1:
                shapes[i] = [1, 1 / mini]

        self.batch_shapes = np.ceil(
            np.array(shapes) * self.imgsz / self.stride + self.pad
            ).astype(int) * self.stride
        self.batch = bi  # batch idx

    def load_image(self, i, rect_mode=True):
        """Loads 1 image from dataset index 'i', returns (im, resized hw)."""
        f = self.im_files[i]
        im = cv2.imread(f)
        if im is None:
            raise FileNotFoundError(f"Image Not Found {f}") 
        
        h0, w0 = im.shape[:2]
        if rect_mode:
            r = self.imgsz / max(h0, w0)  # ratio
            if r != 1: 
                w, h = (min(math.ceil(w0 * r), self.imgsz), min(math.ceil(h0 * r), self.imgsz))
                im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
        elif not (h0 == w0 == self.imgsz): 
            im = cv2.resize(im, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
        return im, (h0, w0), im.shape[:2]

    def __getitem__(self, index):
        """Returns transformed label information for given index."""
        return self.transforms(self._load_image_and_label(index))
    
    def _load_image_and_label(self, index):
        label = deepcopy(self.labels[index])
        label.pop("shape", None)  # shape is for rect, remove it
        label["img"], label["ori_shape"], label["resized_shape"] = self.load_image(index)
        label["ratio_pad"] = (
            label["resized_shape"][0] / label["ori_shape"][0],
            label["resized_shape"][1] / label["ori_shape"][1],
        )  # for evaluation
        if self.rect:
            label["rect_shape"] = self.batch_shapes[self.batch[index]]
        label = self._format_label(label)
        return label

    def _format_label(self, label):
        """
        Custom your label format here.

        Note:
            cls is not with bboxes now, classification and semantic segmentation need an independent cls label
            Can also support classification and semantic segmentation by adding or removing dict keys there.
        """
        bboxes = label.pop("bboxes")
        segments = label.pop("segments", [])
        keypoints = label.pop("keypoints", None)
        bbox_format = label.pop("bbox_format")
        normalized = label.pop("normalized")

        segment_resamples = 100 if self.use_obb else 1000
        if len(segments) > 0:
            # list[np.array(1000, 2)] * num_samples
            # (N, 1000, 2)
            segments = np.stack(resample_segments(segments, n=segment_resamples), axis=0)
        else:
            segments = np.zeros((0, segment_resamples, 2), dtype=np.float32)
        label["instances"] = Instances(bboxes, segments, keypoints, bbox_format=bbox_format, normalized=normalized)
        return label

    def __len__(self):
        """Returns the length of the labels list for the dataset."""
        return len(self.labels)

    def build_transforms(self, hyp=None):
        """Builds and appends transforms to the list."""
        if self.augment:
            hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
            hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
            transforms = v8_transforms(self, self.imgsz, hyp)
        else:
            transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)])
        transforms.append(
            Format(
                bbox_format="xywh",
                normalize=True,
                batch_idx=True,
                bgr=hyp.bgr if self.augment else 0.0,  # only affect training.
            )
        )
        return transforms

    def close_mosaic(self, hyp):
        """Sets mosaic, copy_paste and mixup options to 0.0 and builds transformations."""
        hyp.mosaic = 0.0  # set mosaic ratio=0.0
        hyp.copy_paste = 0.0  # keep the same behavior as previous v8 close-mosaic
        hyp.mixup = 0.0  # keep the same behavior as previous v8 close-mosaic
        self.transforms = self.build_transforms(hyp)

    @staticmethod
    def collate_fn(batch):
        """Collates data samples into batches."""
        new_batch = {}
        keys = batch[0].keys()
        values = list(zip(*[list(b.values()) for b in batch]))
        for i, k in enumerate(keys):
            value = values[i]
            if k == "img":
                value = torch.stack(value, 0)
            if k in {"bboxes", "cls"}:
                value = torch.cat(value, 0)
            
            new_batch[k] = value
        
        new_batch["batch_idx"] = list(new_batch["batch_idx"])
        for i in range(len(new_batch["batch_idx"])):
            new_batch["batch_idx"][i] += i  # add target image index for build_targets()
        new_batch["batch_idx"] = torch.cat(new_batch["batch_idx"], 0)
        return new_batch

if __name__ == '__main__':
    from tools.common import yaml_load
    img_path = '/chaofeng/yolo/coco/images/val2017'
    label_path = '/chaofeng/yolo/coco/labels/val2017'
    data_dict = yaml_load('/chaofeng/yolo/code/cfg/coco.yaml')

    from tools.common import IterableSimpleNamespace, yaml_load
    train_cfg = '/chaofeng/yolo/code/cfg/train.yaml'
    hyp = yaml_load(train_cfg)
    hyp = IterableSimpleNamespace(**hyp)

    dataset = YoloDataset(
        img_path=img_path,
        label_path=label_path,
        imgsz=640,
        batch_size=16,
        augment=True,
        rect=False,
        stride=32,
        cache=True, 
        data=data_dict,
        hyp=hyp
    )
    for i in dataset:
        print(i)
