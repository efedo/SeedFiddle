"""CUDA-resident median filtering compatible with selected PlantCV semantics.

PlantCV 4.11.3's pinned ``median_blur.py`` delegates to SciPy
``ndimage.median_filter`` through ``_helpers._rect_filter`` and
``_helpers._rect_replace``. This primitive preserves the uint8 input domain,
SciPy's default ``reflect`` boundary mode and ``origin=0`` rank placement, and
PlantCV's rectangular crop/filter/replace behavior where the pinned SciPy
implementation has defined in-view boundary coordinates. Kernel tuples are ordered
``(height, width)``. ROI coordinates are explicit half-open
``(x0, y0, x1, y1)`` bounds; filtering reflects at the cropped ROI boundary
and pixels outside the crop are copied unchanged.

The public wrapper source is pinned to PlantCV 4.11.3, commit
``77e3322fd0cd5a9bef18bb99402f20c8f2db87a9``. The installed transitive
``_helpers.py`` SHA observed in the isolated environment is recorded below as
environment evidence, not as a direct source-manifest pin. The runner's
odd-scalar no-ROI domain is narrower than this primitive's 1..255 per-axis
kernel domain. SciPy 1.15.3's ``ni_support.c`` reflect mapping performs only
one reduction for sufficiently negative coordinates. When an axis length is
greater than one and ``kernel_axis // 2 >= 4 * axis_length``, the mapping can
produce coordinate -1 and read outside the logical raster; the result can
depend on excluded storage. Such input/ROI geometry is rejected explicitly
instead of reproducing an out-of-view read or claiming a mathematical
reflection as pinned SciPy behavior.

CUDA work is divided into output tiles. Each temporary neighborhood contains
at most ``MAX_TEMPORARY_WINDOW_VALUES`` uint8 values, regardless of raster
size; there is no whole-image ``H*W*K*K`` materialization or padded-image
growth for narrow rasters. The selected runner-compatible pixel cap is
64,000,000. This is an engineering primitive, not scientific validation.
"""

from __future__ import annotations

from math import gcd
from typing import Callable

import torch


PLANTCV_VERSION = "4.11.3"
PLANTCV_COMMIT = "77e3322fd0cd5a9bef18bb99402f20c8f2db87a9"
PLANTCV_MEDIAN_SOURCE_SHA256 = "66988db126ba849be2c39353ac8bf301ebe0e839932cbdd5eb2cf46269e363ee"
PLANTCV_HELPER_REFERENCE_ENV_SHA256 = "c05b0c944210657332ce307f2066caab55f2ebc8d8fea11988db3e476e7cdfba"
MIN_KERNEL_AXIS = 1
MAX_KERNEL_AXIS = 255
MAX_RASTER_PIXELS = 64_000_000
MAX_TEMPORARY_WINDOW_VALUES = 1_000_000
MAX_OUTPUTS_PER_TILE = 65_536


class MedianResourceError(ValueError):
    """Raised when the raster exceeds the isolated runner's explicit pixel cap."""


class MedianCancelledError(RuntimeError):
    """Filtering was cancelled without returning a partially filtered image."""


class UnsupportedMedianGeometryError(ValueError):
    """Raised when pinned SciPy reflect can read outside the logical raster."""


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
        raise MedianResourceError(
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


def _kernel_shape(ksize: int | tuple[int, int]) -> tuple[int, int]:
    if type(ksize) is int:
        values = (ksize, ksize)
    elif type(ksize) is tuple:
        if len(ksize) != 2:
            raise ValueError("ksize tuple must contain exactly (height, width)")
        values = ksize
    else:
        raise TypeError("ksize must be an integer or a two-integer tuple")
    for value in values:
        if type(value) is not int:
            raise TypeError("each ksize dimension must be an integer")
        if not MIN_KERNEL_AXIS <= value <= MAX_KERNEL_AXIS:
            raise ValueError(
                f"each ksize dimension must be between {MIN_KERNEL_AXIS} and {MAX_KERNEL_AXIS} inclusive"
            )
    return values


def _roi_bounds(roi: tuple[int, int, int, int] | None, height: int, width: int) -> tuple[int, int, int, int]:
    if roi is None:
        return (0, 0, width, height)
    if type(roi) is not tuple or len(roi) != 4:
        raise TypeError("roi must be None or a four-integer (x0, y0, x1, y1) tuple")
    if any(type(value) is not int for value in roi):
        raise TypeError("each ROI coordinate must be an integer")
    x0, y0, x1, y1 = roi
    if not (0 <= x0 <= width and 0 <= x1 <= width and 0 <= y0 <= height and 0 <= y1 <= height):
        raise ValueError("roi coordinates must lie within the image bounds")
    if x0 >= x1 or y0 >= y1:
        raise ValueError("roi must describe a non-empty half-open rectangle")
    return roi


def _validate_scipy_reflect_geometry(
    kernel_height: int,
    kernel_width: int,
    roi_height: int,
    roi_width: int,
) -> None:
    """Reject geometries where SciPy 1.15.3's C reflect map leaves the view."""
    for axis, (kernel, length) in enumerate(
        ((kernel_height, roi_height), (kernel_width, roi_width))
    ):
        # ni_support.c special-cases length <= 1. For longer dimensions its
        # negative-coordinate branch reduces cc only once when cc < -2*length.
        # At cc == -4*length (first reachable when the kernel origin is at
        # least 4*length), that reduction leaves zero and the next branch maps
        # zero to -1. This is an excluded-storage read, not a boundary
        # convention that a CUDA kernel can reproduce deterministically.
        if length > 1 and kernel // 2 >= 4 * length:
            axis_name = "height" if axis == 0 else "width"
            raise UnsupportedMedianGeometryError(
                "pinned SciPy 1.15.3 reflect is source-undefined for this "
                f"{axis_name} geometry (kernel={kernel}, length={length}); "
                "the original C mapping can address outside the logical raster"
            )


def _reflect_indices(indices: torch.Tensor, length: int) -> torch.Tensor:
    """Map arbitrary coordinates using SciPy's half-sample symmetric reflect."""
    if length == 1:
        return torch.zeros_like(indices)
    period = 2 * length
    folded = torch.remainder(indices, period)
    return torch.where(folded < length, folded, period - 1 - folded)


def _median_validated(
    gray: torch.Tensor,
    kernel_height: int,
    kernel_width: int,
    bounds: tuple[int, int, int, int],
    cancel: Callable[[], bool] | None = None,
) -> torch.Tensor:
    """Filter after all public arguments have been checked, using bounded tiles."""
    x0, y0, x1, y1 = bounds
    roi_height, roi_width = y1 - y0, x1 - x0
    source = gray[y0:y1, x0:x1]
    if x0 == 0 and y0 == 0 and x1 == int(gray.shape[1]) and y1 == int(gray.shape[0]):
        output = torch.empty((roi_height, roi_width), dtype=torch.uint8, device=gray.device)
        result = output
    else:
        output = gray.clone(memory_format=torch.contiguous_format)
        result = torch.empty((roi_height, roi_width), dtype=torch.uint8, device=gray.device)

    window_area = kernel_height * kernel_width
    tile_pixels = min(
        MAX_OUTPUTS_PER_TILE,
        max(1, MAX_TEMPORARY_WINDOW_VALUES // window_area),
    )
    rank = window_area // 2 + 1  # SciPy rank_filter's default origin=0 median rank.
    y_offsets = torch.arange(
        -(kernel_height // 2), kernel_height - kernel_height // 2,
        dtype=torch.int64,
        device=gray.device,
    )
    x_offsets = torch.arange(
        -(kernel_width // 2), kernel_width - kernel_width // 2,
        dtype=torch.int64,
        device=gray.device,
    )
    pixel_count = roi_height * roi_width

    with torch.no_grad():
        for start in range(0, pixel_count, tile_pixels):
            if cancel is not None and cancel():
                raise MedianCancelledError("Median filtering cancelled between tiles.")
            stop = min(pixel_count, start + tile_pixels)
            local_flat = torch.arange(start, stop, dtype=torch.int64, device=gray.device)
            local_y = local_flat // roi_width
            local_x = local_flat - local_y * roi_width
            sample_y = _reflect_indices(local_y[:, None] + y_offsets[None, :], roi_height)
            sample_x = _reflect_indices(local_x[:, None] + x_offsets[None, :], roi_width)
            windows = source[sample_y[:, :, None], sample_x[:, None, :]]
            medians = torch.kthvalue(windows.reshape(stop - start, window_area), rank, dim=1).values
            result[local_y, local_x] = medians

    if cancel is not None and cancel():
        raise MedianCancelledError("Median filtering cancelled before returning its result.")
    if result is not output:
        output[y0:y1, x0:x1] = result
    return output


def median_blur(
    gray: torch.Tensor,
    ksize: int | tuple[int, int],
    roi: tuple[int, int, int, int] | None = None,
    *,
    cancel: Callable[[], bool] | None = None,
) -> torch.Tensor:
    """Apply defined SciPy/PlantCV uint8 median semantics to a CUDA raster.

    ``ksize`` is a positive scalar or ``(height, width)`` pair in [1, 255].
    ``roi`` is ``None`` or the half-open rectangle ``(x0, y0, x1, y1)``.
    Filtering reflects at the input or cropped ROI boundary when SciPy 1.15.3
    keeps all reflected coordinates inside that logical region. Source-undefined
    large-kernel geometries raise ``UnsupportedMedianGeometryError``. Pixels
    outside a supplied ROI retain their original values.
    ``cancel`` may be a callback (including ``threading.Event.is_set``). It is
    checked before allocation, between tiles, and before returning the result.
    Cancellation raises ``MedianCancelledError`` and never changes the input.
    """
    gray = _validate_gray(gray)
    kernel_height, kernel_width = _kernel_shape(ksize)
    bounds = _roi_bounds(roi, int(gray.shape[0]), int(gray.shape[1]))
    x0, y0, x1, y1 = bounds
    _validate_scipy_reflect_geometry(
        kernel_height,
        kernel_width,
        y1 - y0,
        x1 - x0,
    )
    if cancel is not None and not callable(cancel):
        raise TypeError("cancel must be a callable or None.")
    if cancel is not None and cancel():
        raise MedianCancelledError("Median filtering cancelled before allocation.")
    return _median_validated(gray, kernel_height, kernel_width, bounds, cancel)


__all__ = [
    "MAX_KERNEL_AXIS",
    "MAX_OUTPUTS_PER_TILE",
    "MAX_RASTER_PIXELS",
    "MAX_TEMPORARY_WINDOW_VALUES",
    "MedianResourceError",
    "MedianCancelledError",
    "PLANTCV_COMMIT",
    "PLANTCV_HELPER_REFERENCE_ENV_SHA256",
    "PLANTCV_MEDIAN_SOURCE_SHA256",
    "PLANTCV_VERSION",
    "UnsupportedMedianGeometryError",
    "median_blur",
]
