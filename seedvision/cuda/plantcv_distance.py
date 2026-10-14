"""CUDA implementation of PlantCV's pinned OpenCV watershed distance map.

PlantCV 4.11.3 calls ``cv2.distanceTransformWithLabels(mask, cv2.DIST_L2,
maskSize=0)[0]``. OpenCV 4.11.0 changes that labeled request to its 5x5 chamfer
mask and runs two raster sweeps with 16-bit fixed-point weights. This is not
the exact Euclidean transform: the effective edge costs are 1.0, 1.4, and
2.1969 pixels, rounded to fixed-point integers before propagation.

The CUDA primitive accepts a dense, non-overlapping ``uint8`` HxW tensor with
values exactly 0 or 255. It keeps the fixed-point work raster and float32
result on the input CUDA device, preserves the input, and supports full-canvas
masks up to the isolated runner's 64,000,000-pixel limit. OpenCV's no-zero
input behavior is retained: an all-foreground mask receives the finite
``DIST_MAX`` sentinel (65533.8046875), while an all-background mask is zero.

The local PlantCV source is pinned to commit
``77e3322fd0cd5a9bef18bb99402f20c8f2db87a9``. Its call is in
``plantcv/plantcv/watershed.py`` at SHA-256
``d5fd90bf4e064117ff47ee333d2b83e62134901fda36f6966ae1b0be4a82a712``.
The effective OpenCV implementation is from the separately locked 4.11.0.86
wheel; its fixed-point sweep is mirrored here rather than substituted with an
exact EDT or jump-flood approximation.
"""

from __future__ import annotations

from math import gcd
from typing import Callable

import torch


PLANTCV_VERSION = "4.11.3"
PLANTCV_COMMIT = "77e3322fd0cd5a9bef18bb99402f20c8f2db87a9"
PLANTCV_WATERSHED_SOURCE_SHA256 = "d5fd90bf4e064117ff47ee333d2b83e62134901fda36f6966ae1b0be4a82a712"
OPENCV_REFERENCE_VERSION = "4.11.0"
OPENCV_REFERENCE_WHEEL = "opencv-python==4.11.0.86"
OPENCV_DISTANCE_SOURCE_URL = (
    "https://github.com/opencv/opencv/blob/4.11.0/modules/imgproc/src/distransform.cpp"
)

MAX_RASTER_PIXELS = 64_000_000
SCAN_CHUNK_PIXELS = 262_144

# OpenCV DIST_SHIFT=16, DIST_MASK_5 metrics (float32) and DIST_MAX.
_DIST_SHIFT = 16
_HV_COST = 65_536
_DIAGONAL_COST = 91_750  # cvRound(float32(1.4) * 65536)
_LONG_COST = 143_976  # cvRound(float32(2.1969) * 65536)
_UINT32_MAX = 4_294_967_295
_DIST_MAX = _UINT32_MAX - _LONG_COST

# Stored signed int32 values preserve unsigned cost ordering by flipping the
# sign bit. Addition by a positive fixed-point edge cost then remains exact up
# through UINT32_MAX, including OpenCV's high no-seed sentinel.
_COST_SIGN_BIT = -(1 << 31)
_INF_ENCODED = _DIST_MAX - (1 << 31)
_ZERO_ENCODED = _COST_SIGN_BIT
_DISTANCE_SCALE = 1.0 / (1 << _DIST_SHIFT)


class DistanceResourceError(ValueError):
    """Raised when a raster exceeds the isolated reference runner's cap."""


class DistanceCancelledError(RuntimeError):
    """Raised when cancellation interrupts a sweep without returning output."""


def _validate_mask(mask: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(mask):
        raise TypeError("mask must be a PyTorch tensor")
    if mask.layout != torch.strided:
        raise TypeError("mask must be a dense strided tensor")
    if mask.device.type != "cuda":
        raise ValueError("mask must already reside on a CUDA device")
    if mask.dtype != torch.uint8:
        raise TypeError("mask must have dtype torch.uint8")
    if mask.ndim != 2:
        raise ValueError("mask must be a two-dimensional HxW raster")
    height, width = (int(value) for value in mask.shape)
    if height < 1 or width < 1:
        raise ValueError("mask must have non-empty height and width")
    if height * width > MAX_RASTER_PIXELS:
        raise DistanceResourceError(
            f"mask exceeds the reference runner limit of {MAX_RASTER_PIXELS:,} pixels"
        )
    stride_y, stride_x = (int(value) for value in mask.stride())
    if (height > 1 and stride_y <= 0) or (width > 1 and stride_x <= 0):
        raise ValueError("mask must not use overlapping or reversed raster strides")
    if height > 1 and width > 1:
        common = gcd(stride_y, stride_x)
        if stride_x // common < height and stride_y // common < width:
            raise ValueError("mask must not use overlapping raster strides")
    return mask


def _check_cancel(cancel: Callable[[], bool] | None) -> None:
    if cancel is not None and cancel():
        raise DistanceCancelledError("Distance-map calculation was cancelled.")


def _scan_from_left(
    base: torch.Tensor,
    carry: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    """Apply the same-row left recurrence to one bounded horizontal chunk."""
    keys = base.to(torch.int64) - offsets
    prefix = torch.cummin(keys, dim=0).values
    carry_key = carry.to(torch.int64) + _HV_COST
    prefix = torch.minimum(prefix, carry_key)
    propagated = prefix + offsets
    return torch.minimum(base.to(torch.int64), propagated).to(torch.int32)


def _forward_sweep(
    work: torch.Tensor,
    mask: torch.Tensor,
    offsets: torch.Tensor,
    weights: torch.Tensor,
    cancel: Callable[[], bool] | None,
) -> None:
    height, width = (int(value) for value in mask.shape)
    for y in range(height):
        _check_cancel(cancel)
        py = y + 2
        for start in range(0, width, SCAN_CHUNK_PIXELS):
            stop = min(width, start + SCAN_CHUNK_PIXELS)
            count = stop - start
            x0 = start + 2
            x1 = stop + 2

            # The eighth entry is the current pixel's initial value. It is
            # DIST_MAX for foreground and zero for background, matching the
            # labeled OpenCV sweep's t0 initialization and source branch.
            neighbors = torch.stack(
                (
                    work[py - 2, x0 - 1 : x1 - 1],
                    work[py - 2, x0 + 1 : x1 + 1],
                    work[py - 1, x0 - 2 : x1 - 2],
                    work[py - 1, x0 - 1 : x1 - 1],
                    work[py - 1, x0 : x1],
                    work[py - 1, x0 + 1 : x1 + 1],
                    work[py - 1, x0 + 2 : x1 + 2],
                    work[py, x0:x1],
                ),
                dim=0,
            )
            candidates = neighbors + weights
            base = torch.amin(candidates, dim=0)
            left_carry = work[py, start + 1]
            updated = _scan_from_left(base, left_carry, offsets[:count])
            work[py, x0:x1].copy_(updated)


def _backward_sweep(
    work: torch.Tensor,
    offsets: torch.Tensor,
    weights: torch.Tensor,
    cancel: Callable[[], bool] | None,
) -> None:
    height, width = int(work.shape[0]) - 4, int(work.shape[1]) - 4
    first_start = ((width - 1) // SCAN_CHUNK_PIXELS) * SCAN_CHUNK_PIXELS
    for y in range(height - 1, -1, -1):
        _check_cancel(cancel)
        py = y + 2
        for start in range(first_start, -1, -SCAN_CHUNK_PIXELS):
            stop = min(width, start + SCAN_CHUNK_PIXELS)
            count = stop - start
            x0 = start + 2
            x1 = stop + 2

            # OpenCV visits lower rows and then the row's right-hand pixels.
            # Reversing each chunk turns that right recurrence into the same
            # parallel prefix-min operation used in the forward sweep.
            neighbors = torch.stack(
                (
                    work[py + 2, x0 + 1 : x1 + 1],
                    work[py + 2, x0 - 1 : x1 - 1],
                    work[py + 1, x0 + 2 : x1 + 2],
                    work[py + 1, x0 + 1 : x1 + 1],
                    work[py + 1, x0 : x1],
                    work[py + 1, x0 - 1 : x1 - 1],
                    work[py + 1, x0 - 2 : x1 - 2],
                    work[py, x0:x1],
                ),
                dim=0,
            )
            base = torch.amin(neighbors + weights, dim=0)
            right_carry = work[py, stop + 2]
            reversed_base = torch.flip(base, dims=(0,))
            updated_reversed = _scan_from_left(reversed_base, right_carry, offsets[:count])
            updated = torch.flip(updated_reversed, dims=(0,))
            work[py, x0:x1].copy_(updated)


def _distance_map_fixed_point(
    mask: torch.Tensor,
    *,
    cancel: Callable[[], bool] | None = None,
) -> torch.Tensor:
    """Mirror OpenCV's two-pass mask-5 transform using fixed-point costs."""
    height, width = (int(value) for value in mask.shape)
    device = mask.device
    work = torch.full(
        (height + 4, width + 4),
        _INF_ENCODED,
        dtype=torch.int32,
        device=device,
    )
    zero_cost = torch.full((), _ZERO_ENCODED, dtype=torch.int32, device=device)
    infinity_cost = torch.full((), _INF_ENCODED, dtype=torch.int32, device=device)
    work[2:-2, 2:-2].copy_(torch.where(mask == 0, zero_cost, infinity_cost))

    offsets = torch.arange(SCAN_CHUNK_PIXELS, dtype=torch.int64, device=device) * _HV_COST
    forward_weights = torch.tensor(
        (_LONG_COST, _LONG_COST, _LONG_COST, _DIAGONAL_COST, _HV_COST,
         _DIAGONAL_COST, _LONG_COST, 0),
        dtype=torch.int32,
        device=device,
    ).view(8, 1)
    backward_weights = torch.tensor(
        (_LONG_COST, _LONG_COST, _LONG_COST, _DIAGONAL_COST, _HV_COST,
         _DIAGONAL_COST, _LONG_COST, 0),
        dtype=torch.int32,
        device=device,
    ).view(8, 1)

    _forward_sweep(work, mask, offsets, forward_weights, cancel)
    _backward_sweep(work, offsets, backward_weights, cancel)
    _check_cancel(cancel)

    encoded = work[2:-2, 2:-2]
    unsigned_fixed = torch.bitwise_xor(encoded, _COST_SIGN_BIT).view(torch.uint32)
    return unsigned_fixed.to(torch.float32).mul_(_DISTANCE_SCALE)


def distance_map(
    mask: torch.Tensor,
    *,
    cancel: Callable[[], bool] | None = None,
) -> torch.Tensor:
    """Calculate PlantCV's CUDA-resident watershed distance map.

    ``mask`` must be a dense CUDA ``uint8`` HxW tensor containing only 0 and
    255. The returned HxW tensor is float32 on that same CUDA device. It
    represents OpenCV 4.11.0's labeled 5x5 chamfer result, not exact Euclidean
    distance. ``cancel`` is checked before raster allocation and between rows;
    an interrupted call raises without returning a partial map. CUDA allocation
    errors propagate unchanged, and masks above the pinned runner's explicit
    pixel cap are rejected before work begins.
    """
    mask = _validate_mask(mask)
    if cancel is not None and not callable(cancel):
        raise TypeError("cancel must be a callable or None")
    _check_cancel(cancel)
    binary_values = (mask == 0) | (mask == 255)
    if not bool(binary_values.all().item()):
        raise ValueError("mask must contain only the binary uint8 values 0 and 255")
    _check_cancel(cancel)
    with torch.no_grad():
        return _distance_map_fixed_point(mask, cancel=cancel)


__all__ = [
    "DistanceCancelledError",
    "DistanceResourceError",
    "MAX_RASTER_PIXELS",
    "OPENCV_DISTANCE_SOURCE_URL",
    "OPENCV_REFERENCE_VERSION",
    "OPENCV_REFERENCE_WHEEL",
    "PLANTCV_COMMIT",
    "PLANTCV_VERSION",
    "PLANTCV_WATERSHED_SOURCE_SHA256",
    "SCAN_CHUNK_PIXELS",
    "distance_map",
]
