"""CUDA implementation of PlantCV's pinned watershed peak-selection stage.

The public PlantCV 4.11.3 watershed routine calls scikit-image's
``peak_local_max(distance_map, min_distance=distance, labels=mask)``.  This
module ports that internal stage for dense, contiguous CUDA tensors.  It does
not claim that PlantCV's public runner archives these intermediate outputs.

The supported parameter domain is ``1 <= distance <= MAX_DISTANCE``.  The
catalogue currently selects 10; this primitive does not infer that distance
zero is rejected by the reference library.  It rejects unsupported radii,
large tensor-workspace estimates, excessive candidate counts, and excessive
greedy-spacing rounds instead of returning an incomplete peak set.
"""

from __future__ import annotations

from numbers import Integral
from typing import Callable

import torch
import torch.nn.functional as F


PLANTCV_VERSION = "4.11.3"
PLANTCV_COMMIT = "77e3322fd0cd5a9bef18bb99402f20c8f2db87a9"
PLANTCV_WATERSHED_SOURCE_SHA256 = "d5fd90bf4e064117ff47ee333d2b83e62134901fda36f6966ae1b0be4a82a712"
SCIKIT_IMAGE_REFERENCE_VERSION = "0.25.2"
SCIPY_REFERENCE_VERSION = "1.15.3"
NUMPY_REFERENCE_VERSION = "2.2.6"
SCIKIT_IMAGE_PEAK_SOURCE_SHA256 = "193b2cf2d30805cb1250c0427e4a43b2a221cd7efb4ba7e827e85cc3ebb79fcd"
SCIKIT_IMAGE_COORD_SOURCE_SHA256 = "cfc12d50f0cc30f06096ca2701fc89b69592843f0037a4c8184058f8ea50ebe6"
SCIPY_FILTERS_SOURCE_SHA256 = "919dcbfd3cccf9c2b61210214d27c828a49529db75a6432f2d9c99442946ec65"
SCIPY_MORPHOLOGY_SOURCE_SHA256 = "0f6ee388080105895b9d31b96b86912515c37e954f8b153a0fd1eeefeef462d2af"
SCIPY_MEASUREMENTS_SOURCE_SHA256 = "1c7a3878776ba69c6811d12d7a1fd12288e365a0cec74d6285ffebcae9330ba"

MAX_RASTER_PIXELS = 64_000_000
MAX_DISTANCE = 128
MAX_CANDIDATES = 1_000_000
MAX_NMS_ROUNDS = 4_096
# Separate from replay's legacy 8B raster-work model. With separable filters,
# 64B admits the committed full-resolution cases (31 and63 rounds; <=34B).
# It is a source-window/vector estimate, not elapsed time or GPU instructions.
MAX_NMS_SAMPLE_VISITS = 64_000_000_000
MAX_ESTIMATED_TENSOR_WORKSPACE_BYTES = 768 * 1024 * 1024
_WORKSPACE_BYTES_PER_ROI_PIXEL = 64
_WORKSPACE_BYTES_PER_CANDIDATE = 40
_FLOAT32_MIN = torch.finfo(torch.float32).min


class PeakCancelledError(RuntimeError):
    """Raised when cancellation interrupts peak selection without output."""


class PeakResourceError(ValueError):
    """Raised when a documented radius, workspace, or candidate limit is hit."""


class PeakInputError(ValueError):
    """Raised when input tensors do not match the supported stage contract."""


def _check_cancel(cancel: Callable[[], bool] | None) -> None:
    if cancel is not None and cancel():
        raise PeakCancelledError("Peak selection was cancelled.")


def _validate_inputs(
    distance_map: torch.Tensor,
    mask: torch.Tensor,
    distance: int,
    cancel: Callable[[], bool] | None,
) -> tuple[int, int, int]:
    if cancel is not None and not callable(cancel):
        raise TypeError("cancel must be a callable or None")
    _check_cancel(cancel)
    if not torch.is_tensor(distance_map) or not torch.is_tensor(mask):
        raise TypeError("distance_map and mask must be PyTorch tensors")
    if distance_map.layout != torch.strided or mask.layout != torch.strided:
        raise PeakInputError("distance_map and mask must be dense strided tensors")
    if distance_map.device.type != "cuda" or mask.device.type != "cuda":
        raise PeakInputError("distance_map and mask must already reside on CUDA")
    if distance_map.device != mask.device:
        raise PeakInputError("distance_map and mask must use the same CUDA device")
    if distance_map.dtype != torch.float32:
        raise TypeError("distance_map must have dtype torch.float32")
    if mask.dtype != torch.uint8:
        raise TypeError("mask must have dtype torch.uint8")
    if distance_map.ndim != 2 or mask.ndim != 2:
        raise PeakInputError("distance_map and mask must be two-dimensional HxW rasters")
    if tuple(distance_map.shape) != tuple(mask.shape):
        raise PeakInputError("distance_map and mask must have the same frame shape")
    height, width = (int(value) for value in distance_map.shape)
    if height < 1 or width < 1:
        raise PeakInputError("distance_map and mask must have non-empty height and width")
    if height * width > MAX_RASTER_PIXELS:
        raise PeakResourceError(
            f"raster exceeds the supported {MAX_RASTER_PIXELS:,}-pixel limit"
        )
    if not distance_map.is_contiguous() or not mask.is_contiguous():
        raise PeakInputError("distance_map and mask must be contiguous CUDA rasters")
    if isinstance(distance, bool) or not isinstance(distance, Integral):
        raise TypeError("distance must be an integer")
    distance = int(distance)
    if distance < 1:
        raise PeakInputError("supported distance values start at 1")
    if distance > MAX_DISTANCE:
        raise PeakResourceError(
            f"distance exceeds the supported radius limit of {MAX_DISTANCE}"
        )
    return height, width, distance


def _estimated_workspace_bytes(
    roi_height: int,
    roi_width: int,
    distance: int,
    candidate_count: int,
) -> int:
    """Bound known dense tensor temporaries; backend scratch is not measured."""
    roi_pixels = roi_height * roi_width
    padded_pixels = (roi_height + 2 * distance) * (roi_width + 2 * distance)
    return (
        roi_pixels * _WORKSPACE_BYTES_PER_ROI_PIXEL
        + padded_pixels * 4  # float32 padded max-filter input
        + candidate_count * _WORKSPACE_BYTES_PER_CANDIDATE
    )


def _empty_outputs(
    distance_map: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    height, width = (int(value) for value in distance_map.shape)
    coordinates = torch.empty((0, 2), dtype=torch.int64, device=distance_map.device)
    peaks = torch.zeros((height, width), dtype=torch.bool, device=distance_map.device)
    return coordinates, peaks


def _binary_opening_cross(mask: torch.Tensor) -> torch.Tensor:
    """SciPy's 2-D default binary opening (connectivity-one cross, border 0)."""
    kernel = torch.tensor(
        ((0.0, 1.0, 0.0), (1.0, 1.0, 1.0), (0.0, 1.0, 0.0)),
        dtype=torch.float32,
        device=mask.device,
    ).view(1, 1, 3, 3)
    values = mask.to(torch.float32)[None, None]
    eroded = F.conv2d(values, kernel, padding=1) == 5.0
    opened = F.conv2d(eroded.to(torch.float32), kernel, padding=1) > 0.0
    return opened[0, 0]


def _nms_round_visits(roi_shape, distance, candidate_count):
    """Conservative source-window/vector visits, not GPU instructions/time."""
    pixels = roi_shape[0]*roi_shape[1]
    kernel = 2*(distance-1)+1
    # Two separable filters: four one-dimensional source windows per pixel.
    # Twenty-four dense visits cover canvas resets, conversion, intermediate
    # pool outputs and comparison buffers. Sixty-four vector visits cover rank/active/selected reads/writes,
    # indexed gathers/scatters, reductions and final coordinate selection.
    return pixels*(24+4*kernel)+64*candidate_count


def _spacing_maximum(values: torch.Tensor, kernel_size: int, radius: int) -> torch.Tensor:
    """A square maximum is exactly two axis maxima, including -inf padding."""
    horizontal = F.max_pool2d(values[None, None], kernel_size=(1, kernel_size),
        stride=1, padding=(0, radius))
    return F.max_pool2d(horizontal, kernel_size=(kernel_size, 1),
        stride=1, padding=(radius, 0))[0, 0]


def _greedy_spacing(
    ordered_coordinates: torch.Tensor,
    distance: int,
    roi_shape: tuple[int, int],
    cancel: Callable[[], bool] | None,
) -> torch.Tensor:
    """Exact priority-greedy Chebyshev suppression in bounded parallel rounds.

    Coordinates arrive sorted by descending intensity with stable row-major
    tie order.  Each round accepts every currently active point with best
    priority in its conflict neighborhood, then suppresses its neighbors.  A
    float64 priority canvas represents these integer ranks exactly for the
    documented one-million-candidate limit; it never stores flattened pixel
    indices as float values.
    """
    if distance <= 1 or ordered_coordinates.shape[0] == 0:
        return ordered_coordinates

    candidate_count = int(ordered_coordinates.shape[0])
    height, width = roi_shape
    visits = _nms_round_visits(roi_shape,distance,candidate_count)
    rounds = min(MAX_NMS_ROUNDS,MAX_NMS_SAMPLE_VISITS//visits)
    if rounds<1:
        raise PeakResourceError('greedy spacing exceeds its NMS sample-visit budget before allocation')
    radius = distance - 1
    kernel_size = 2 * radius + 1
    device = ordered_coordinates.device
    y_coords = ordered_coordinates[:, 0]
    x_coords = ordered_coordinates[:, 1]
    ranks = torch.arange(candidate_count, dtype=torch.float64, device=device)
    active = torch.ones(candidate_count, dtype=torch.bool, device=device)
    selected = torch.zeros_like(active)
    priority_canvas = torch.empty((height, width), dtype=torch.float64, device=device)
    winner_canvas = torch.zeros((height, width), dtype=torch.bool, device=device)
    worst_rank = float(candidate_count)

    for _round in range(rounds):
        _check_cancel(cancel)
        priority_canvas.fill_(-worst_rank)
        priority_canvas[y_coords, x_coords] = torch.where(active, -ranks, -worst_rank)
        best_rank = _spacing_maximum(priority_canvas, kernel_size, radius)
        winners = active & (best_rank[y_coords, x_coords] == -ranks)
        if not bool(winners.any().item()):
            raise RuntimeError("greedy spacing made no progress")
        selected |= winners

        winner_canvas.zero_()
        winner_canvas[y_coords, x_coords] = winners
        suppressed = _spacing_maximum(winner_canvas.to(torch.float32), kernel_size, radius) > 0.0
        active &= ~suppressed[y_coords, x_coords]
        if not bool(active.any().item()):
            return ordered_coordinates[selected]

    if bool(active.any().item()):
        raise PeakResourceError(
            f"greedy spacing exceeded its {rounds:,}-round/sample-visit limit"
        )
    return ordered_coordinates[selected]


def local_maxima(
    distance_map: torch.Tensor,
    mask: torch.Tensor,
    distance: int,
    *,
    cancel: Callable[[], bool] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return PlantCV's pinned peak coordinates and painted peak mask on CUDA.

    Inputs are same-device, contiguous CUDA ``float32`` and binary ``uint8``
    HxW tensors.  The mask's 255-valued label is treated as one label even
    when its foreground is disconnected, matching scikit-image's
    ``find_objects`` ROI behavior.  The returned int64 coordinates are ordered
    by descending distance with stable row-major ties; the returned bool mask
    paints those exact coordinates.  Inputs are never modified or copied to
    the host.  Empty results are represented by shape ``(0, 2)`` and an
    all-false full-frame mask.
    """
    height, width, distance = _validate_inputs(distance_map, mask, distance, cancel)
    _check_cancel(cancel)
    finite = torch.isfinite(distance_map).all()
    binary = ((mask == 0) | (mask == 255)).all()
    if not bool(finite.item()):
        raise PeakInputError("distance_map must contain only finite float32 values")
    if not bool(binary.item()):
        raise PeakInputError("mask must contain only the binary uint8 values 0 and 255")
    _check_cancel(cancel)

    # scikit-image excludes a width=min_distance border from labels before it
    # computes the label's bounding box.  A surviving foreground pixel must
    # therefore lie inside this rectangular canvas interior.
    if height <= 2 * distance or width <= 2 * distance:
        _check_cancel(cancel)
        return _empty_outputs(distance_map)
    interior = mask[distance : height - distance, distance : width - distance]
    interior_foreground = interior == 255
    row_indices = torch.where(interior_foreground.any(dim=1))[0]
    if row_indices.numel() == 0:
        _check_cancel(cancel)
        return _empty_outputs(distance_map)
    col_indices = torch.where(interior_foreground.any(dim=0))[0]
    top = distance + int(row_indices[0].item())
    bottom = distance + int(row_indices[-1].item()) + 1
    left = distance + int(col_indices[0].item())
    right = distance + int(col_indices[-1].item()) + 1
    roi_height, roi_width = bottom - top, right - left
    if _estimated_workspace_bytes(
        roi_height, roi_width, distance, MAX_CANDIDATES
    ) > MAX_ESTIMATED_TENSOR_WORKSPACE_BYTES:
        raise PeakResourceError(
            "ROI exceeds the estimated CUDA tensor-workspace limit"
        )

    _check_cancel(cancel)
    label_roi = mask[top:bottom, left:right] == 255
    roi_distance = distance_map[top:bottom, left:right]
    threshold = distance_map.min()
    image = torch.where(
        label_roi,
        roi_distance,
        torch.full((), _FLOAT32_MIN, dtype=torch.float32, device=distance_map.device),
    )

    if roi_height * roi_width == 1:
        candidates = image > threshold
    else:
        _check_cancel(cancel)
        padded_image = F.pad(
            image[None, None],
            (distance, distance, distance, distance),
            mode="replicate",
        )
        local_max = F.max_pool2d(
            padded_image,
            kernel_size=2 * distance + 1,
            stride=1,
        )[0, 0]
        candidates = image == local_max
        if bool(candidates[label_roi].all().item()):
            isolated = label_roi ^ _binary_opening_cross(label_roi)
            candidates = isolated
        candidates &= image > threshold

    candidate_count = int(candidates.sum().item())
    if candidate_count > MAX_CANDIDATES:
        raise PeakResourceError(
            f"candidate set exceeds the {MAX_CANDIDATES:,}-point limit"
        )
    if _estimated_workspace_bytes(
        roi_height, roi_width, distance, candidate_count
    ) > MAX_ESTIMATED_TENSOR_WORKSPACE_BYTES:
        raise PeakResourceError(
            "candidate arrays exceed the estimated CUDA tensor-workspace limit"
        )
    _check_cancel(cancel)
    flat_indices = torch.nonzero(candidates.reshape(-1), as_tuple=False).reshape(-1)
    if flat_indices.numel() == 0:
        _check_cancel(cancel)
        return _empty_outputs(distance_map)

    row = torch.div(flat_indices, roi_width, rounding_mode="floor")
    col = flat_indices.remainder(roi_width)
    row_major_coordinates = torch.stack((row, col), dim=1)
    intensities = image.reshape(-1)[flat_indices]
    order = torch.argsort(intensities, descending=True, stable=True)
    ordered_coordinates = row_major_coordinates[order]
    selected = _greedy_spacing(
        ordered_coordinates,
        distance,
        (roi_height, roi_width),
        cancel,
    )
    global_coordinates = selected + torch.tensor(
        (top, left), dtype=torch.int64, device=distance_map.device
    )
    _check_cancel(cancel)
    peak_mask = torch.zeros((height, width), dtype=torch.bool, device=distance_map.device)
    if global_coordinates.shape[0]:
        peak_mask[global_coordinates[:, 0], global_coordinates[:, 1]] = True
    _check_cancel(cancel)
    return global_coordinates.contiguous(), peak_mask


__all__ = [
    "MAX_CANDIDATES",
    "MAX_DISTANCE",
    "MAX_ESTIMATED_TENSOR_WORKSPACE_BYTES",
    "MAX_NMS_ROUNDS",
    "MAX_NMS_SAMPLE_VISITS",
    "MAX_RASTER_PIXELS",
    "NUMPY_REFERENCE_VERSION",
    "PeakCancelledError",
    "PeakInputError",
    "PeakResourceError",
    "PLANTCV_COMMIT",
    "PLANTCV_VERSION",
    "PLANTCV_WATERSHED_SOURCE_SHA256",
    "SCIKIT_IMAGE_COORD_SOURCE_SHA256",
    "SCIKIT_IMAGE_PEAK_SOURCE_SHA256",
    "SCIKIT_IMAGE_REFERENCE_VERSION",
    "SCIPY_FILTERS_SOURCE_SHA256",
    "SCIPY_MEASUREMENTS_SOURCE_SHA256",
    "SCIPY_MORPHOLOGY_SOURCE_SHA256",
    "SCIPY_REFERENCE_VERSION",
    "local_maxima",
]
