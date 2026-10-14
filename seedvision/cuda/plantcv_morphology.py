"""CUDA-resident full-canvas PlantCV-compatible morphology primitives.

The audited public wrappers are PlantCV 4.11.3, commit
``77e3322fd0cd5a9bef18bb99402f20c8f2db87a9``. ``erode.py`` and ``dilate.py``
delegate through the installed PlantCV ``_helpers._erode`` and
``_helpers._dilate`` functions; those helpers construct square all-ones uint8
kernels, call OpenCV through ``_rect_filter``, and return its full-canvas output.
The isolated worker accepts
``ksize`` 2..255 and ``iterations`` 1..64, without an ROI. The pinned OpenCV
default anchor for an even kernel is the lower/right central sample
(``ksize // 2``); its default morphology border value is neutral (255 for
erosion, 0 for dilation).

This module reproduces that full-canvas contract on CUDA with separable
horizontal/vertical max-pool passes. It avoids materializing the ``ksize**2``
neighborhood per pixel; its full-resolution intermediates are O(HW), while
runtime depends on the kernel and PyTorch's pooling implementation. Pooling's
implicit negative-infinity border is neutral for nonnegative uint8 values and
their erosion complement, so tall or wide narrow rasters do not allocate
kernel-sized padding. Raster pixel count is bounded to the reference runner's
64,000,000-element limit. Dense CUDA uint8 HxW inputs are accepted, including
non-overlapping strided views. Optional half-open ROI bounds filter an
independent crop and replace it in a fresh full-canvas copy. ROI is an additive
primitive/working-recipe contract; the historical single-call runner stays
full-canvas. Zero iterations remain outside the declared domain.

The source manifest pins the public wrapper files but does not include the
transitive ``_helpers.py`` source. The isolated installed helper observed
during this feature audit had SHA-256
``c05b0c944210657332ce307f2066caab55f2ebc8d8fea11988db3e476e7cdfba``; that
observation is not represented as a committed upstream source pin.
"""

from __future__ import annotations

from math import gcd

import torch
import torch.nn.functional as functional

from seedvision.cuda import tensor_callback as _tensor_callback


PLANTCV_VERSION = "4.11.3"
PLANTCV_COMMIT = "77e3322fd0cd5a9bef18bb99402f20c8f2db87a9"
PLANTCV_ERODE_SOURCE_SHA256 = "6ae07bc734ae7226b9f39f5b970d549319e49b45ca9559b0ae3064e02a67487a"
PLANTCV_DILATE_SOURCE_SHA256 = "e0bf2fcf8ca96e8d7aa5cb05cf3b7e4dbdb163450466126bc8fb124549f2472f"
PLANTCV_HELPER_REFERENCE_ENV_SHA256 = "c05b0c944210657332ce307f2066caab55f2ebc8d8fea11988db3e476e7cdfba"
MIN_KERNEL_SIZE = 2
MAX_KERNEL_SIZE = 255
MIN_ITERATIONS = 1
MAX_ITERATIONS = 64
MAX_RASTER_PIXELS = 64_000_000


class MorphologyResourceError(ValueError):
    """Raised when a raster exceeds the selected runner's explicit element cap."""


class MorphologyCancelledError(RuntimeError):
    """Cancellation withheld a partially filtered raster."""


def _roi_bounds(roi, height: int, width: int):
    if roi is None:
        return (0, 0, width, height)
    if type(roi) is not tuple or len(roi) != 4 or any(type(v) is not int for v in roi):
        raise TypeError('roi must be None or a four-integer (x0, y0, x1, y1) tuple')
    x0, y0, x1, y1 = roi
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError('roi must be a non-empty half-open rectangle inside the image')
    return roi


def _check_cancel(cancel, tensors=()):
    if cancel is None:
        return
    if not callable(cancel):
        raise TypeError('cancel must be a callable returning bool or None')
    if _tensor_callback.checked_cancel(cancel, tensors):
        raise MorphologyCancelledError('Morphology cancelled before returning a complete raster.')


def _validate_gray(gray: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(gray):
        raise TypeError("gray must be a PyTorch tensor")
    if gray.layout != torch.strided:
        raise TypeError("gray must be a dense strided tensor")
    if gray.device.type != "cuda":
        raise ValueError("gray must already reside on a CUDA device")
    if gray.dtype != torch.uint8:
        raise TypeError("gray must have dtype torch.uint8")
    if gray.ndim != 2:
        raise ValueError("gray must be a two-dimensional HxW raster")
    height, width = (int(value) for value in gray.shape)
    if height < 1 or width < 1:
        raise ValueError("gray must have non-empty height and width")
    if height * width > MAX_RASTER_PIXELS:
        raise MorphologyResourceError(
            f"gray exceeds the reference runner limit of {MAX_RASTER_PIXELS:,} pixels"
        )
    stride_y, stride_x = (int(value) for value in gray.stride())
    if (height > 1 and stride_y <= 0) or (width > 1 and stride_x <= 0):
        raise ValueError("gray must not use overlapping or reversed raster strides")
    if height > 1 and width > 1:
        common = gcd(stride_y, stride_x)
        if stride_x // common < height and stride_y // common < width:
            raise ValueError("gray must not use overlapping raster strides")
    return gray


def _positive_integer(value: int, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum} inclusive")
    return value


def _axis_max(image: torch.Tensor, kernel_size: int, *, horizontal: bool) -> torch.Tensor:
    """Apply one max window with implicit neutral border and O(HW) storage."""
    anchor = kernel_size // 2
    if horizontal:
        pooled = functional.max_pool2d(image, (1, kernel_size), stride=1, padding=(0, anchor))
        # Symmetric padding makes one extra output for even kernels. The first
        # width outputs have the required default lower/right anchor.
        return pooled[..., : image.shape[-2], : image.shape[-1]]
    pooled = functional.max_pool2d(image, (kernel_size, 1), stride=1, padding=(anchor, 0))
    return pooled[..., : image.shape[-2], : image.shape[-1]]


def _morphology(gray: torch.Tensor, ksize: int, iterations: int, *, eroding: bool,
                roi=None, cancel=None) -> torch.Tensor:
    gray = _validate_gray(gray)
    ksize = _positive_integer(ksize, "ksize", MIN_KERNEL_SIZE, MAX_KERNEL_SIZE)
    iterations = _positive_integer(iterations, "iterations", MIN_ITERATIONS, MAX_ITERATIONS)
    x0, y0, x1, y1 = _roi_bounds(roi, *map(int, gray.shape))
    _check_cancel(cancel, (gray,))

    # The pool kernels operate on float32, where every uint8 integer is exactly
    # representable. Complementing converts a minimum into a maximum. The
    # implicit negative-infinity border is neutral for either nonnegative
    # uint8 intensities or their erosion complement, even for a one-pixel axis.
    with torch.no_grad():
        values = gray[y0:y1, x0:x1].to(dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        if eroding:
            values = 255.0 - values
        for _ in range(iterations):
            _check_cancel(cancel, (gray, values))
            values = _axis_max(values, ksize, horizontal=True)
            values = _axis_max(values, ksize, horizontal=False)
        if eroding:
            values = 255.0 - values
        result = values[0, 0].to(dtype=torch.uint8)
        _check_cancel(cancel, (gray, values, result))
        if (x0, y0, x1, y1) == (0, 0, int(gray.shape[1]), int(gray.shape[0])):
            return result
        output = gray.clone(memory_format=torch.contiguous_format)
        output[y0:y1, x0:x1] = result
        _check_cancel(cancel, (gray, values, result, output))
        return output


def erode(gray: torch.Tensor, ksize: int, iterations: int = 1, *, roi=None, cancel=None) -> torch.Tensor:
    """Apply square erosion to a CUDA raster or independent half-open ROI."""
    return _morphology(gray, ksize, iterations, eroding=True, roi=roi, cancel=cancel)


def dilate(gray: torch.Tensor, ksize: int, iterations: int = 1, *, roi=None, cancel=None) -> torch.Tensor:
    """Apply square dilation to a CUDA raster or independent half-open ROI."""
    return _morphology(gray, ksize, iterations, eroding=False, roi=roi, cancel=cancel)


__all__ = [
    "MAX_ITERATIONS",
    "MAX_KERNEL_SIZE",
    "MAX_RASTER_PIXELS",
    "MIN_ITERATIONS",
    "MIN_KERNEL_SIZE",
    "MorphologyResourceError",
    "MorphologyCancelledError",
    "PLANTCV_COMMIT",
    "PLANTCV_DILATE_SOURCE_SHA256",
    "PLANTCV_ERODE_SOURCE_SHA256",
    "PLANTCV_HELPER_REFERENCE_ENV_SHA256",
    "PLANTCV_VERSION",
    "dilate",
    "erode",
]
