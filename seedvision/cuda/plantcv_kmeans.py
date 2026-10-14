"""Bounded CUDA cluster-mask operation from PlantCV 4.11.3.

This module does not yet implement predict_kmeans or train_kmeans.
"""
from __future__ import annotations

import torch
from seedvision.cuda.tensor_callback import checked_cancel

MAX_PIXELS = 16_000_000
MAX_OUTPUT_BYTES = 256 * 1024 * 1024
MAX_TILE_PIXELS = 65_536
MAX_CAT_LIST = 256
MAX_SCRATCH_BYTES = 2 * MAX_TILE_PIXELS
PLANTCV_VERSION = '4.11.3'
PLANTCV_KMEANS_CLASSIFIER_SHA256 = '09019da00e0713c51cff31ffd240f7bf8aa0b4e228f6d88f2bed9c1d40430392'


class NativeKMeansError(ValueError):
    """Invalid input or an explicit cluster-mask resource limit."""


class NativeKMeansCancelled(RuntimeError):
    """Cancelled before a complete cluster-mask result."""


def _check_cancel(cancel,*tensors):
    if checked_cancel(cancel,tensors):
        raise NativeKMeansCancelled('K-means mask generation was cancelled')


def mask_kmeans(labeled_img, k, cat_list=None, *, cancel=None, tile_pixels=MAX_TILE_PIXELS):
    """Return ordered per-class masks or one union mask from CUDA uint8 labels.

    None selects each ID in range(k); a nonempty category list/tuple selects its
    union. IDs outside 0..255 never match. The public source fails on an empty
    cat_list; this API reports that domain explicitly. Partial outputs stay local.
    """
    if cancel is not None and not callable(cancel):
        raise TypeError('cancel must be a zero-argument callable or None')
    _check_cancel(cancel,labeled_img)
    if not torch.is_tensor(labeled_img) or labeled_img.device.type != 'cuda':
        raise NativeKMeansError('labeled_img must already be a CUDA tensor')
    if (labeled_img.layout != torch.strided or labeled_img.dtype != torch.uint8
            or labeled_img.ndim != 2 or not labeled_img.is_contiguous()):
        raise NativeKMeansError('labeled_img must be a contiguous strided uint8 HW raster')
    pixels = labeled_img.numel()
    if not 1 <= pixels <= MAX_PIXELS:
        raise NativeKMeansError(f'labeled_img must contain 1..{MAX_PIXELS} pixels')
    if type(k) is not int or not 0 <= k <= 256:
        raise NativeKMeansError('k must be an integer in [0,256]')
    if type(tile_pixels) is not int or not 1 <= tile_pixels <= MAX_TILE_PIXELS:
        raise NativeKMeansError(f'tile_pixels must be an integer in [1,{MAX_TILE_PIXELS}]')
    if cat_list is None:
        categories, output_count = None, k
    else:
        if not isinstance(cat_list, (list,tuple)) or not 1 <= len(cat_list) <= MAX_CAT_LIST:
            raise NativeKMeansError(f'cat_list must contain 1..{MAX_CAT_LIST} integer IDs')
        if any(type(value) is not int for value in cat_list):
            raise NativeKMeansError('cat_list entries must be integers')
        categories, output_count = tuple(cat_list), 1
    if pixels*output_count > MAX_OUTPUT_BYTES:
        raise NativeKMeansError('Requested cluster masks exceed the aggregate output-byte limit')
    _check_cancel(cancel,labeled_img)
    source = labeled_img.view(-1)
    if categories is not None:
        _check_cancel(cancel,labeled_img)
        output = torch.empty_like(labeled_img)
        destination = output.view(-1)
        for start in range(0,pixels,tile_pixels):
            _check_cancel(cancel,labeled_img,output)
            stop = min(pixels,start+tile_pixels)
            selected = torch.zeros(stop-start,dtype=torch.bool,device=labeled_img.device)
            for class_id in categories:
                _check_cancel(cancel,labeled_img,output,selected)
                if 0 <= class_id <= 255:
                    selected.logical_or_(source[start:stop] == class_id)
            destination[start:stop].zero_().masked_fill_(selected,255)
        _check_cancel(cancel,labeled_img,output)
        return output
    masks = {}
    for class_id in range(k):
        _check_cancel(cancel,labeled_img,*masks.values())
        output = torch.empty(labeled_img.shape,dtype=torch.uint8,device=labeled_img.device)
        masks[str(class_id)] = output
        destination = output.view(-1)
        for start in range(0,pixels,tile_pixels):
            _check_cancel(cancel,labeled_img,*masks.values())
            stop = min(pixels,start+tile_pixels)
            selected = source[start:stop] == class_id
            destination[start:stop].zero_().masked_fill_(selected,255)
    _check_cancel(cancel,labeled_img,*masks.values())
    return masks
