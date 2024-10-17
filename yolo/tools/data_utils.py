import os
import torch
import numpy as np
import cv2
import hashlib
from typing import Union
from tools.logger import LOGGER
from pathlib import Path
from PIL import Image, ImageOps
import contextlib
from tools.ops import xyxy2xywh


NUM_THREADS = min(8, max(1, os.cpu_count() - 1))  # number of YOLO multiprocessing threads
LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))  # https://pytorch.org/docs/stable/elastic/run.html
IMG_FORMATS = {"bmp", "dng", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp", "pfm"}  # image suffixes
DATASET_CACHE_VERSION = "1.0.3"

def polygons2masks_overlap(imgsz, segments, downsample_ratio=1):
    """Return a (640, 640) overlap mask."""
    masks = np.zeros(
        (imgsz[0] // downsample_ratio, imgsz[1] // downsample_ratio),
        dtype=np.int32 if len(segments) > 255 else np.uint8,
    )
    areas = []
    ms = []
    for si in range(len(segments)):
        mask = polygon2mask(imgsz, [segments[si].reshape(-1)], downsample_ratio=downsample_ratio, color=1)
        ms.append(mask)
        areas.append(mask.sum())
    areas = np.asarray(areas)
    index = np.argsort(-areas)
    ms = np.array(ms)[index]
    for i in range(len(segments)):
        mask = ms[i] * (i + 1)
        masks = masks + mask
        masks = np.clip(masks, a_min=0, a_max=i + 1)
    return masks, index

def polygons2masks(imgsz, polygons, color, downsample_ratio=1):
    """
    Convert a list of polygons to a set of binary masks of the specified image size.

    Args:
        imgsz (tuple): The size of the image as (height, width).
        polygons (list[np.ndarray]): A list of polygons. Each polygon is an array with shape [N, M], where
                                     N is the number of polygons, and M is the number of points such that M % 2 = 0.
        color (int): The color value to fill in the polygons on the masks.
        downsample_ratio (int, optional): Factor by which to downsample each mask. Defaults to 1.

    Returns:
        (np.ndarray): A set of binary masks of the specified image size with the polygons filled in.
    """
    return np.array([polygon2mask(imgsz, [x.reshape(-1)], color, downsample_ratio) for x in polygons])

def polygon2mask(imgsz, polygons, color=1, downsample_ratio=1):
    """
    Convert a list of polygons to a binary mask of the specified image size.

    Args:
        imgsz (tuple): The size of the image as (height, width).
        polygons (list[np.ndarray]): A list of polygons. Each polygon is an array with shape [N, M], where
                                     N is the number of polygons, and M is the number of points such that M % 2 = 0.
        color (int, optional): The color value to fill in the polygons on the mask. Defaults to 1.
        downsample_ratio (int, optional): Factor by which to downsample the mask. Defaults to 1.

    Returns:
        (np.ndarray): A binary mask of the specified image size with the polygons filled in.
    """
    mask = np.zeros(imgsz, dtype=np.uint8)
    polygons = np.asarray(polygons, dtype=np.int32)
    polygons = polygons.reshape((polygons.shape[0], -1, 2))
    cv2.fillPoly(mask, polygons, color=color)
    nh, nw = (imgsz[0] // downsample_ratio, imgsz[1] // downsample_ratio)
    # Note: fillPoly first then resize is trying to keep the same loss calculation method when mask-ratio=1
    return cv2.resize(mask, (nw, nh))

def get_hash(paths):
    """Returns a single hash value of a list of paths (files or dirs)."""
    size = sum(os.path.getsize(p) for p in paths if os.path.exists(p))  # sizes
    h = hashlib.sha256(str(size).encode())  # hash sizes
    h.update("".join(paths).encode())  # hash paths
    return h.hexdigest()  # return hash

def load_dataset_cache_file(path):
    """Load an Ultralytics *.cache dictionary from path."""
    import gc
    gc.disable()  # reduce pickle load time https://github.com/ultralytics/ultralytics/pull/1585
    cache = np.load(str(path), allow_pickle=True).item()  # load dict
    gc.enable()
    return cache


def save_dataset_cache_file(prefix, path, x):
    """Save an Ultralytics dataset *.cache dictionary x to path."""
    if is_dir_writeable(path.parent):
        if path.exists():
            path.unlink()  # remove *.cache file if exists
        np.save(str(path), x)  # save cache for next time
        path.with_suffix(".cache.npy").rename(path)  # remove .npy suffix
        LOGGER.info(f"{prefix}New cache created: {path}")
    else:
        LOGGER.warning(f"{prefix}WARNING ⚠️ Cache directory {path.parent} is not writeable, cache not saved.")

def is_dir_writeable(dir_path: Union[str, Path]) -> bool:
    """
    Check if a directory is writeable.

    Args:
        dir_path (str | Path): The path to the directory.

    Returns:
        (bool): True if the directory is writeable, False otherwise.
    """
    return os.access(str(dir_path), os.W_OK)

def exif_size(img: Image.Image):
    """Returns exif-corrected PIL size."""
    s = img.size  # (width, height)
    if img.format == "JPEG":  # only support JPEG images
        with contextlib.suppress(Exception):
            exif = img.getexif()
            if exif:
                rotation = exif.get(274, None)  # the EXIF key for the orientation tag is 274
                if rotation in {6, 8}:  # rotation 270 or 90
                    s = s[1], s[0]
    return s

# def verify_image_label(args):
#     """Verify one image-label pair."""
#     im_file, lb_file, prefix, keypoint, num_cls, nkpt, ndim = args
#     # Number (missing, found, empty, corrupt), message, segments, keypoints
#     nm, nf, ne, nc, msg, segments, keypoints = 0, 0, 0, 0, "", [], None
#     try:
#         # Verify images
#         im = Image.open(im_file)
#         im.verify()  # PIL verify
#         shape = exif_size(im)  # image size
#         shape = (shape[1], shape[0])  # h w
#         assert (shape[0] > 9) & (shape[1] > 9), f"image size {shape} <10 pixels"
#         assert im.format.lower() in IMG_FORMATS, f"invalid image format {im.format}"
#         if im.format.lower() in {"jpg", "jpeg"}:
#             with open(im_file, "rb") as f:
#                 f.seek(-2, 2)
#                 if f.read() != b"\xff\xd9":  # corrupt JPEG
#                     ImageOps.exif_transpose(Image.open(im_file)).save(im_file, "JPEG", subsampling=0, quality=100)
#                     msg = f"{prefix}WARNING ⚠️ {im_file}: corrupt JPEG restored and saved"

#         # Verify labels
#         if os.path.isfile(lb_file):
#             nf = 1  # label found
#             with open(lb_file) as f:
#                 lb = [x.split() for x in f.read().strip().splitlines() if len(x)]         
#                 classes = np.array([x[0] for x in lb], dtype=np.float32)
#                 xywh = [np.array(x[1:], dtype=np.float32) for x in lb]  # for coco
#                 lb = np.concatenate((classes.reshape(-1, 1), xywh), 1)  # (cls, xywh)
#             nl = len(lb)
#             if nl:
#                 if keypoint:
#                     assert lb.shape[1] == (5 + nkpt * ndim), f"labels require {(5 + nkpt * ndim)} columns each"
#                     points = lb[:, 5:].reshape(-1, ndim)[:, :2]
#                 else:
#                     assert lb.shape[1] == 5, f"labels require 5 columns, {lb.shape[1]} columns detected"
#                     points = lb[:, 1:]
#                 assert points.max() <= 1, f"non-normalized or out of bounds coordinates {points[points > 1]}"
#                 assert lb.min() >= 0, f"negative label values {lb[lb < 0]}"

#                 # All labels
#                 max_cls = lb[:, 0].max()  # max label count
#                 assert max_cls <= num_cls, (
#                     f"Label class {int(max_cls)} exceeds dataset class count {num_cls}. "
#                     f"Possible class labels are 0-{num_cls - 1}"
#                 )
#                 _, i = np.unique(lb, axis=0, return_index=True)
#                 if len(i) < nl:  # duplicate row check
#                     lb = lb[i]  # remove duplicates
#                     if segments:
#                         segments = [segments[x] for x in i]
#                     msg = f"{prefix}WARNING ⚠️ {im_file}: {nl - len(i)} duplicate labels removed"
#             else:
#                 ne = 1  # label empty
#                 lb = np.zeros((0, (5 + nkpt * ndim) if keypoint else 5), dtype=np.float32)
#         else:
#             nm = 1  # label missing
#             lb = np.zeros((0, (5 + nkpt * ndim) if keypoints else 5), dtype=np.float32)
#         if keypoint:
#             keypoints = lb[:, 5:].reshape(-1, nkpt, ndim)
#             if ndim == 2:
#                 kpt_mask = np.where((keypoints[..., 0] < 0) | (keypoints[..., 1] < 0), 0.0, 1.0).astype(np.float32)
#                 keypoints = np.concatenate([keypoints, kpt_mask[..., None]], axis=-1)  # (nl, nkpt, 3)
#         lb = lb[:, :5]
#         return im_file, lb, shape, segments, keypoints, nm, nf, ne, nc, msg
    
#     except Exception as e:
#         nc = 1
#         msg = f"{prefix}WARNING ⚠️ {im_file}: ignoring corrupt image/label: {e}"
#         return [None, None, None, None, None, nm, nf, ne, nc, msg]


# def verify_image_label(args):
#     """Verify one image-label pair."""
#     im_file, lb_file, prefix, keypoint, num_cls, nkpt, ndim = args
#     # Number (missing, found, empty, corrupt), message, segments, keypoints
#     nm, nf, ne, nc, msg, segments, keypoints = 0, 0, 0, 0, "", [], None
#     try:
#         # Verify images
#         im = Image.open(im_file)
#         im.verify()  # PIL verify
#         shape = exif_size(im)  # image size
#         shape = (shape[1], shape[0])  # h w
#         assert (shape[0] > 9) & (shape[1] > 9), f"image size {shape} <10 pixels"
#         assert im.format.lower() in IMG_FORMATS, f"invalid image format {im.format}"
#         if im.format.lower() in {"jpg", "jpeg"}:
#             with open(im_file, "rb") as f:
#                 f.seek(-2, 2)
#                 if f.read() != b"\xff\xd9":  # corrupt JPEG
#                     ImageOps.exif_transpose(Image.open(im_file)).save(im_file, "JPEG", subsampling=0, quality=100)
#                     msg = f"{prefix}WARNING ⚠️ {im_file}: corrupt JPEG restored and saved"

#         # Verify labels
#         if os.path.isfile(lb_file):
#             nf = 1  # label found
#             with open(lb_file) as f:
#                 lb = [x.split() for x in f.read().strip().splitlines() if len(x)]         
#                 classes = np.array([x[0] for x in lb], dtype=np.float32)
#                 xywh = [np.array(x[1:], dtype=np.float32) for x in lb]  # for coco
#                 lb = np.concatenate((classes.reshape(-1, 1), xywh), 1)  # (n, 5)
#             nl = len(lb)
#             if nl:
#                 if keypoint:
#                     assert lb.shape[1] == (5 + nkpt * ndim), f"labels require {(5 + nkpt * ndim)} columns each"
#                     points = lb[:, 5:].reshape(-1, ndim)[:, :2]
#                 else:
#                     assert lb.shape[1] == 5, f"labels require 5 columns, {lb.shape[1]} columns detected"
#                     points = lb[:, 1:]
#                 assert points.max() <= 1, f"non-normalized or out of bounds coordinates {points[points > 1]}"
#                 assert lb.min() >= 0, f"negative label values {lb[lb < 0]}"

#                 # All labels
#                 max_cls = lb[:, 0].max()  # max label count
#                 assert max_cls <= num_cls, (
#                     f"Label class {int(max_cls)} exceeds dataset class count {num_cls}. "
#                     f"Possible class labels are 0-{num_cls - 1}"
#                 )
#                 _, i = np.unique(lb, axis=0, return_index=True)
#                 if len(i) < nl:  # duplicate row check
#                     lb = lb[i]  # remove duplicates
#                     if segments:
#                         segments = [segments[x] for x in i]
#                     msg = f"{prefix}WARNING ⚠️ {im_file}: {nl - len(i)} duplicate labels removed"
#             else:
#                 ne = 1  # label empty
#                 lb = np.zeros((0, (5 + nkpt * ndim) if keypoint else 5), dtype=np.float32)
#         else:
#             nm = 1  # label missing
#             lb = np.zeros((0, (5 + nkpt * ndim) if keypoints else 5), dtype=np.float32)
#         if keypoint:
#             keypoints = lb[:, 5:].reshape(-1, nkpt, ndim)
#             if ndim == 2:
#                 kpt_mask = np.where((keypoints[..., 0] < 0) | (keypoints[..., 1] < 0), 0.0, 1.0).astype(np.float32)
#                 keypoints = np.concatenate([keypoints, kpt_mask[..., None]], axis=-1)  # (nl, nkpt, 3)
#         lb = lb[:, :5]
#         return im_file, lb, shape, segments, keypoints, nm, nf, ne, nc, msg
    
#     except Exception as e:
#         nc = 1
#         msg = f"{prefix}WARNING ⚠️ {im_file}: ignoring corrupt image/label: {e}"
#         return [None, None, None, None, None, nm, nf, ne, nc, msg]


def verify_image_label(args):
    """Verify one image-label pair."""
    im_file, lb_file, prefix, num_cls = args
    # Number (missing, found, empty, corrupt), message, segments, keypoints
    nm, nf, ne, nc, msg = 0, 0, 0, 0, ""
    try:
        # Verify images
        im = Image.open(im_file)
        im.verify()  # PIL verify
        shape = exif_size(im)  # image size
        shape = (shape[1], shape[0])  # h w
        assert (shape[0] > 9) & (shape[1] > 9), f"image size {shape} <10 pixels"
        assert im.format.lower() in IMG_FORMATS, f"invalid image format {im.format}"
        if im.format.lower() in {"jpg", "jpeg"}:
            with open(im_file, "rb") as f:
                f.seek(-2, 2)
                if f.read() != b"\xff\xd9":  # corrupt JPEG
                    ImageOps.exif_transpose(Image.open(im_file)).save(im_file, "JPEG", subsampling=0, quality=100)
                    msg = f"{prefix}WARNING ⚠️ {im_file}: corrupt JPEG restored and saved"

        # Verify labels
        if os.path.isfile(lb_file):
            nf = 1  # label found
            with open(lb_file) as f:
                lb = [x.split() for x in f.read().strip().splitlines() if len(x)]         
                classes = np.array([x[0] for x in lb], dtype=np.float32)
                xywh = [np.array(x[1:], dtype=np.float32) for x in lb]  # for coco
                lb = np.concatenate((classes.reshape(-1, 1), xywh), 1)  # (n, 5)
            nl = len(lb)
            if nl:
                assert lb.shape[1] == 5, f"labels require 5 columns, {lb.shape[1]} columns detected"
                points = lb[:, 1:]
                assert points.max() <= 1, f"non-normalized or out of bounds coordinates {points[points > 1]}"
                assert lb.min() >= 0, f"negative label values {lb[lb < 0]}"

                # All labels
                max_cls = lb[:, 0].max()  # max label count
                assert max_cls <= num_cls, (
                    f"Label class {int(max_cls)} exceeds dataset class count {num_cls}. "
                    f"Possible class labels are 0-{num_cls - 1}"
                )
                _, i = np.unique(lb, axis=0, return_index=True)
                if len(i) < nl:  # duplicate row check
                    lb = lb[i]  # remove duplicates
                    msg = f"{prefix}WARNING ⚠️ {im_file}: {nl - len(i)} duplicate labels removed"
            else:
                ne = 1  # label empty
                lb = np.zeros((0, 5), dtype=np.float32)
        else:
            nm = 1  # label missing
            lb = np.zeros((0, 5), dtype=np.float32)
        lb = lb[:, :5]
        return im_file, lb, shape, nm, nf, ne, nc, msg
    
    except Exception as e:
        nc = 1
        msg = f"{prefix}WARNING ⚠️ {im_file}: ignoring corrupt image/label: {e}"
        return [None, None, None, nm, nf, ne, nc, msg]
    