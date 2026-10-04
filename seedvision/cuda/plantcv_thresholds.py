"""CUDA-resident compatibility primitives for pinned PlantCV thresholds.

The audited source is PlantCV 4.11.3, commit
``77e3322fd0cd5a9bef18bb99402f20c8f2db87a9``. Its threshold entry points
delegate binary/Otsu to OpenCV ``threshold`` and adaptive mean to OpenCV
``adaptiveThreshold``. This module accepts only CUDA ``uint8`` HW tensors and
returns same-device ``uint8`` masks with values 0 or 255. It is a focused
primitive layer; it does not modify or claim completion of any analysis graph.

Otsu transfers only the 256-bin integer histogram to the CPU. Because the pinned
OpenCV build uses IPP by default and can choose different partitions for tied
or nearly tied maxima, this primitive accepts only a unique exact-score maximum
separated from the runner-up by more than 1e-6; ambiguous histograms raise
``OtsuAmbiguityError``. Candidate partitions with a class probability below
``FLT_EPSILON`` are skipped, and cutoff zero is used when none pass. This is a
conservative supported domain, not a claim of full IPP Otsu parity. Adaptive
mean computes replicated-border sums on CUDA and applies OpenCV's float32 scale
and uint8 rounding behavior. Unsupported input domains fail instead of being
silently coerced.
"""

from __future__ import annotations

import math
import struct
from typing import Literal


PLANTCV_VERSION = "4.11.3"
PLANTCV_COMMIT = "77e3322fd0cd5a9bef18bb99402f20c8f2db87a9"
PLANTCV_THRESHOLD_SOURCE_SHA256 = "04a671632e7e4decf5a662779cd88473b5aff750554afeee98eabe02eaa49907"
MIN_ADAPTIVE_KERNEL = 3
MAX_ADAPTIVE_KERNEL = 255
MAX_ABSOLUTE_OFFSET = 255.0
_OTSU_RELATIVE_GAP_DENOMINATOR = 1_000_000
_OTSU_MIN_CLASS_WEIGHT_DENOMINATOR = 1 << 23  # reciprocal of OpenCV's FLT_EPSILON
MIN_OTSU_RELATIVE_PARTITION_GAP = 1 / _OTSU_RELATIVE_GAP_DENOMINATOR
ObjectType = Literal["light", "dark"]


class OtsuAmbiguityError(ValueError):
    """Raised when distinct Otsu partitions are tied or too close to trust."""


def _validate_gray(gray):
    import torch

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
    if int(gray.shape[0]) < 1 or int(gray.shape[1]) < 1:
        raise ValueError("gray must have non-empty height and width")
    return gray


def _object_type(value: str) -> ObjectType:
    if not isinstance(value, str):
        raise TypeError("object_type must be 'light' or 'dark'")
    normalized = value.upper()
    if normalized == "LIGHT":
        return "light"
    if normalized == "DARK":
        return "dark"
    raise ValueError("object_type must be 'light' or 'dark'")


def _mask_from_condition(condition):
    import torch

    return condition.to(dtype=torch.uint8) * 255


def threshold_binary(gray, threshold: int, object_type: str = "light"):
    """Apply PlantCV ``threshold.binary`` semantics to a CUDA gray raster.

    Light foreground is exactly ``gray > threshold``; dark foreground is
    exactly ``gray <= threshold``. Threshold is an integer in [0, 255].
    """
    gray = _validate_gray(gray)
    if isinstance(threshold, bool) or not isinstance(threshold, int):
        raise TypeError("threshold must be an integer")
    if not 0 <= threshold <= 255:
        raise ValueError("threshold must be between 0 and 255 inclusive")
    polarity = _object_type(object_type)
    condition = gray > threshold if polarity == "light" else gray <= threshold
    return _mask_from_condition(condition)


def _otsu_cutoff_from_histogram(histogram: list[int]) -> int:
    """Return the supported cutoff from 256 integer counts.

    Scores are compared as exact rationals over occupied-bin partitions. Empty
    histogram bins therefore create a harmless threshold plateau, while tied or
    close scores for different partitions are rejected. Partitions with a class
    probability below OpenCV's ``FLT_EPSILON`` guard are skipped; if none pass,
    cutoff zero is returned. The pinned OpenCV 4.11 environment has IPP enabled
    by default; IPP and the generic source formula choose different partitions
    for some tied histograms, so this helper does not silently select either
    implementation's tie result. The returned cutoff
    represents the selected partition; the lowest occupied-bin boundary is
    used across empty-bin plateaus, so its scalar need not equal OpenCV's scalar
    even when the resulting mask is identical.
    """
    if len(histogram) != 256 or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in histogram):
        raise ValueError("Otsu histogram must contain 256 non-negative integer bins")
    total = sum(histogram)
    if total <= 0:
        raise ValueError("Otsu threshold is undefined for an empty raster")
    occupied = [index for index, count in enumerate(histogram) if count]
    if len(occupied) == 1:
        # This matches the pinned constant-image cutoff and avoids treating the
        # absence of any two-class partition as an ambiguous maximum.
        return 0

    total_moment = sum(index * count for index, count in enumerate(histogram))
    lower_weight = 0
    lower_moment = 0
    best_score: tuple[int, int] | None = None
    second_score: tuple[int, int] | None = None
    best_threshold = 0
    best_partition_count = 0

    def compare(left: tuple[int, int], right: tuple[int, int]) -> int:
        """Compare non-negative rational scores without floating-point loss."""
        cross_difference = left[0] * right[1] - right[0] * left[1]
        return (cross_difference > 0) - (cross_difference < 0)

    for intensity in occupied[:-1]:
        count = histogram[intensity]
        lower_weight += count
        lower_moment += intensity * count
        upper_weight = total - lower_weight
        # OpenCV's generic 8-bit Otsu path ignores partitions with a class
        # probability below FLT_EPSILON (1 / 2**23). The pinned default IPP
        # runner also returns its zero fallback for the audited extreme cases.
        if min(lower_weight, upper_weight) * _OTSU_MIN_CLASS_WEIGHT_DENOMINATOR < total:
            continue
        # Between-class variance is proportional to this exact rational score.
        numerator = (lower_moment * total - total_moment * lower_weight) ** 2
        score = (numerator, lower_weight * upper_weight)
        if best_score is None or compare(score, best_score) > 0:
            second_score = best_score
            best_score = score
            best_threshold = intensity
            best_partition_count = 1
        elif compare(score, best_score) == 0:
            best_partition_count += 1
        elif second_score is None or compare(score, second_score) > 0:
            second_score = score

    if best_score is None:
        return 0
    if best_partition_count > 1:
        raise OtsuAmbiguityError(
            "Otsu has tied exact scores for distinct occupied-bin partitions"
        )
    if second_score is not None:
        best_num, best_den = best_score
        second_num, second_den = second_score
        gap_num = best_num * second_den - second_num * best_den
        gap_den = best_num * second_den
        # Eight float32 epsilons is a conservative partition-separation guard.
        # In the pinned 3,240-case occupied-bin probe, all 3,132 unique maxima
        # matched IPP; their smallest relative gap was about 2.36e-3. A broader
        # 15,824-histogram probe also found no unique-maximum mismatch, while
        # large synthetic near-ties motivate rejecting a wider 1e-6 margin.
        if gap_num * _OTSU_RELATIVE_GAP_DENOMINATOR <= gap_den:
            raise OtsuAmbiguityError(
                "Otsu's best partition is within the 1e-6 relative score gap"
            )
    return best_threshold


def threshold_otsu(gray, object_type: str = "light"):
    """Apply the supported unambiguous subset of PlantCV ``threshold.otsu``.

    The input image stays on CUDA. Only the exact 256-bin integer histogram is
    copied to CPU. Tied or insufficiently separated eligible partitions raise
    :class:`OtsuAmbiguityError`; partitions with class probability below
    ``FLT_EPSILON`` are skipped, and zero is used if none remain. Empty-bin
    plateaus may use a different internal threshold scalar from OpenCV while
    producing the same mask. The pinned reference's default IPP path is not
    claimed for the rejected ambiguity region.
    """
    import torch

    gray = _validate_gray(gray)
    polarity = _object_type(object_type)
    histogram = torch.bincount(gray.reshape(-1), minlength=256)
    compact_counts = [int(value) for value in histogram.detach().cpu().tolist()]
    cutoff = _otsu_cutoff_from_histogram(compact_counts)
    condition = gray > cutoff if polarity == "light" else gray <= cutoff
    return _mask_from_condition(condition)


def _validate_mean_parameters(ksize: int, offset: int | float) -> tuple[int, float]:
    if isinstance(ksize, bool) or not isinstance(ksize, int):
        raise TypeError("ksize must be an integer")
    if ksize < MIN_ADAPTIVE_KERNEL or ksize > MAX_ADAPTIVE_KERNEL or ksize % 2 == 0:
        raise ValueError(
            f"ksize must be odd and between {MIN_ADAPTIVE_KERNEL} and {MAX_ADAPTIVE_KERNEL}"
        )
    if isinstance(offset, bool) or not isinstance(offset, (int, float)):
        raise TypeError("offset must be a finite number")
    effective_offset = float(offset)
    if not math.isfinite(effective_offset):
        raise ValueError("offset must be finite")
    if abs(effective_offset) > MAX_ABSOLUTE_OFFSET:
        raise ValueError(f"offset magnitude must not exceed {MAX_ABSOLUTE_OFFSET:g}")
    return ksize, effective_offset


def _replicated_box_sum(values, axis: int, radius: int):
    """Sum odd sliding windows on CUDA with exact integer replicated borders.

    A pair of one-dimensional prefix sums makes the two-dimensional box
    operation O(HW), independent of kernel area. ``values`` and its prefix are
    integer tensors, so all supported uint8 inputs and ksize <= 255 sum exactly.
    """
    import torch

    length = int(values.shape[axis])
    prefix = torch.cumsum(values, dim=axis, dtype=torch.int64)
    positions = torch.arange(length, device=values.device, dtype=torch.int64)
    left = (positions - radius).clamp_min(0)
    right = (positions + radius).clamp_max(length - 1)
    before_left = (left - 1).clamp_min(0)

    window_sum = prefix.index_select(axis, right)
    prior_sum = prefix.index_select(axis, before_left)
    left_is_zero = left == 0
    if axis == 0:
        left_is_zero = left_is_zero[:, None]
        left_repeats = (radius - positions).clamp_min(0)[:, None]
        right_repeats = (positions + radius - (length - 1)).clamp_min(0)[:, None]
    else:
        left_is_zero = left_is_zero[None, :]
        left_repeats = (radius - positions).clamp_min(0)[None, :]
        right_repeats = (positions + radius - (length - 1)).clamp_min(0)[None, :]
    prior_sum.masked_fill_(left_is_zero, 0)
    window_sum.sub_(prior_sum)
    first_edge = values.select(axis, 0).unsqueeze(axis)
    last_edge = values.select(axis, length - 1).unsqueeze(axis)
    window_sum.addcmul_(first_edge, left_repeats)
    window_sum.addcmul_(last_edge, right_repeats)
    return window_sum


def threshold_mean(
    gray,
    ksize: int,
    offset: int | float,
    object_type: str = "light",
):
    """Apply PlantCV ``threshold.mean`` semantics to a CUDA gray raster.

    The supported kernel is odd [3, 255]. The local mean uses exact integer
    sums and replicated borders, then reproduces OpenCV's float32 scale,
    multiplication, and ties-to-even uint8 rounding. For light objects the
    threshold offset is rounded with ``ceil``; for dark objects it is rounded
    with ``floor``. Light pixels use strict ``>`` and dark pixels use ``<=``.
    Finite offsets in [-255, 255] are supported; larger values are rejected.
    """
    gray = _validate_gray(gray)
    ksize, effective_offset = _validate_mean_parameters(ksize, offset)
    polarity = _object_type(object_type)
    delta = math.ceil(effective_offset) if polarity == "light" else math.floor(effective_offset)
    radius = ksize // 2
    import torch

    source = gray
    horizontal = _replicated_box_sum(gray.to(dtype=torch.int64), axis=1, radius=radius)
    window_sum = _replicated_box_sum(horizontal, axis=0, radius=radius)
    del horizontal
    area = ksize * ksize
    # OpenCV's uint8 boxFilter uses a float scale and rounds the resulting
    # float32 mean to nearest-even. Exact integer division differs in rare
    # near-half cases (for example 14,077,913 / 65,025 can become 216.5f).
    scale = struct.unpack("f", struct.pack("f", 1.0 / area))[0]
    rounded_mean = window_sum.to(dtype=torch.float32).mul_(scale).round_()
    local_threshold = rounded_mean - delta
    condition = source > local_threshold if polarity == "light" else source <= local_threshold
    return _mask_from_condition(condition)


__all__ = [
    "MAX_ABSOLUTE_OFFSET",
    "MAX_ADAPTIVE_KERNEL",
    "MIN_ADAPTIVE_KERNEL",
    "MIN_OTSU_RELATIVE_PARTITION_GAP",
    "OtsuAmbiguityError",
    "PLANTCV_COMMIT",
    "PLANTCV_THRESHOLD_SOURCE_SHA256",
    "PLANTCV_VERSION",
    "threshold_binary",
    "threshold_mean",
    "threshold_otsu",
]
