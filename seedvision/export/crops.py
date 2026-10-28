"""Explicit, CUDA-resident per-seed image and mask crops.

This module provides a SeedFiddle-native crop artifact. It does not claim to
reproduce unresolved SMPTS crop-padding or image-boundary behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
import re
from numbers import Integral
from typing import Callable, Sequence

import numpy as np


_SUPPORTED_METHODS = {
    "procedural_instances",
    "unet_instances",
    "stardist_instances",
    "cellpose_sam_instances",
}
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
MAX_BATCH_CROP_PIXELS = 12_000_000
MAX_BATCH_CROP_OUTPUT_BYTES = 256 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class SeedCropRecord:
    """One requested seed crop, with honest availability and clipping facts.

    ``image_crop`` is a cloned NCHW BGR tensor on the existing CUDA device.
    ``mask_crop`` is an aligned HW boolean tensor on that same device. Neither
    tensor is created for missing, unavailable, failed, or cancelled records.
    Coordinates use corrected full-frame pixels and half-open bounds.
    """

    seed_id: int
    status: str
    image_crop: object | None
    mask_crop: object | None
    corrected_bbox: tuple[int, int, int, int] | None
    requested_bbox: tuple[int, int, int, int] | None
    source_to_crop: tuple[tuple[float, ...], ...] | None
    source_to_corrected: tuple[tuple[float, ...], ...] | None
    source_sha256: str | None
    result_revision: str
    method: str
    label_family: str | None
    checkpoint_id: str
    label_sha256: str | None
    artifact_identity: str | None
    label_shape: tuple[int, int] | None
    source_shape: tuple[int, int] | None
    analysis_crop_offset: tuple[int, int] | None
    label_to_source_scale_xy: tuple[float, float] | None
    padding_px: int
    touches_corrected_frame: bool | None
    touches_analysis_crop: bool | None
    clipped_by_corrected_frame: bool | None
    source_validity: str
    foreign_seed_ids: tuple[int, ...]
    foreign_pixel_count: int
    reason: str = ""


@dataclass(frozen=True, slots=True)
class SeedCropBatch:
    """Immutable metadata for one extraction request."""

    state: str
    method: str
    result_revision: str
    padding_px: int
    records: tuple[SeedCropRecord, ...]
    source_sha256: str | None
    scientifically_validated: bool = False
    review_state: str = "unreviewed"
    reason: str = ""


@dataclass(frozen=True, slots=True)
class _CropPlan:
    seed_id: int
    status: str
    requested_bbox: tuple[int, int, int, int] | None = None
    corrected_bbox: tuple[int, int, int, int] | None = None
    label_local_bbox: tuple[int, int, int, int] | None = None
    touches_analysis_crop: bool | None = None
    reason: str = ""


def extract_seed_crops(
    analysis,
    *,
    method: str,
    seed_ids: Sequence[int],
    padding_px: int,
    result_revision: str,
    cancel: Callable[[], bool] | object | None = None,
) -> SeedCropBatch:
    """Create CUDA-local corrected-image and isolated-mask crops by label ID.

    ``method`` must name the selected analysis result attribute. Padding is an
    explicit SeedFiddle-native pixel choice. Procedural labels may be at a
    bounded working resolution; label-cell centers are mapped to the source
    crop using the actual x/y raster dimensions, then offset into the corrected
    full frame. Learned labels must already align to the analysis crop.

    The corrected full-frame tensor is only sliced and cloned on its existing
    CUDA device. Label metadata is already CPU-resident; only the local label
    tile needed for each returned mask is copied to CUDA. No full corrected
    image or full-resolution label raster is downloaded.
    """

    ids = _normalize_seed_ids(seed_ids)
    padding = _normalize_padding(padding_px)
    if not isinstance(method, str):
        raise TypeError("method must be a string result attribute name.")
    method_name = method
    if not isinstance(result_revision, str) or _SHA256_PATTERN.fullmatch(result_revision) is None:
        raise ValueError("result_revision must be a lowercase SHA-256 identity.")
    revision = result_revision

    source_digest = getattr(analysis, "source_sha256", None)
    records: list[SeedCropRecord] = []

    if not ids:
        return SeedCropBatch(
            "empty", method_name, revision, padding, (), source_digest,
            reason="No seed IDs were requested.",
        )

    if method_name not in _SUPPORTED_METHODS:
        reason = f"Unsupported instance-result method: {method_name}."
        records = [_unavailable_record(seed_id, method_name, revision, padding, source_digest, reason)
                   for seed_id in ids]
        return _batch(records, method_name, revision, padding, source_digest, reason)

    instance_result = getattr(analysis, method_name, None)
    if instance_result is None:
        reason = f"The selected {method_name} result is unavailable."
        records = [_unavailable_record(seed_id, method_name, revision, padding, source_digest, reason)
                   for seed_id in ids]
        return _batch(records, method_name, revision, padding, source_digest, reason)

    try:
        labels, source_shape, family = _selected_labels_and_source_shape(
            analysis, method_name, instance_result
        )
        crop_offset = _normalize_crop_offset(analysis.crop_offset)
        layers_offset = _normalize_crop_offset(
            (analysis.layers.offset_x, analysis.layers.offset_y)
        )
        if layers_offset != crop_offset:
            raise ValueError("The analysis crop and layer offsets do not match.")
        source_tensor = _cuda_corrected_tensor(analysis)
        source_height, source_width = map(int, source_tensor.shape[-2:])
        affine = np.asarray(analysis.calibration.affine_matrix, dtype=np.float64)
        if (affine.shape != (3, 3) or not np.all(np.isfinite(affine))
                or not np.isfinite(np.linalg.det(affine))
                or abs(float(np.linalg.det(affine))) <= np.finfo(np.float64).eps):
            raise ValueError("The source-to-corrected transform is unavailable or invalid.")
        if len(crop_offset) != 2:
            raise ValueError("The analysis crop offset must contain x and y.")
        offset_x, offset_y = crop_offset
        if (offset_x < 0 or offset_y < 0 or offset_x + source_shape[1] > source_width
                or offset_y + source_shape[0] > source_height):
            raise ValueError("The analysis crop does not fit inside the corrected image frame.")
    except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
        reason = str(exc) or "The selected crop source is unavailable."
        records = [_unavailable_record(seed_id, method_name, revision, padding, source_digest, reason)
                   for seed_id in ids]
        return _batch(records, method_name, revision, padding, source_digest, reason)

    label_shape = tuple(map(int, labels.shape))
    source_to_corrected = _matrix_tuple(affine)
    label_digest = sha256(np.ascontiguousarray(labels).tobytes()).hexdigest()
    checkpoint_id = str(getattr(instance_result, "checkpoint_id", "") or "")
    scale_x = source_shape[1] / label_shape[1]
    scale_y = source_shape[0] / label_shape[0]
    scale_xy = (float(scale_x), float(scale_y))

    plans = []
    maximum_label = int(labels.max(initial=0))
    for seed_id in ids:
        if _cancelled(cancel):
            plans = [_CropPlan(value, "cancelled", reason="Extraction was cancelled.")
                     for value in ids]
            break
        plans.append(_plan_one(
            labels, seed_id, source_shape, label_shape, (offset_x, offset_y),
            (source_width, source_height), padding, maximum_label,
        ))
    if any(plan.status == "cancelled" for plan in plans):
        records = [_cancelled_record(value, method_name, revision, padding, source_digest)
                   for value in ids]
        return _batch(records, method_name, revision, padding, source_digest,
                      "Extraction was cancelled during crop planning.")

    output_pixels = sum(
        (plan.corrected_bbox[2] - plan.corrected_bbox[0])
        * (plan.corrected_bbox[3] - plan.corrected_bbox[1])
        for plan in plans if plan.status == "planned"
    )
    output_bytes = output_pixels * (3 * int(source_tensor.element_size()) + 1)
    if (output_pixels > MAX_BATCH_CROP_PIXELS
            or output_bytes > MAX_BATCH_CROP_OUTPUT_BYTES):
        budget_reason = (
            "Requested seed crops exceed the aggregate CUDA crop budget "
            f"({output_pixels:,} pixels, {output_bytes:,} bytes; limits "
            f"{MAX_BATCH_CROP_PIXELS:,} pixels and {MAX_BATCH_CROP_OUTPUT_BYTES:,} bytes)."
        )
        records = []
        for plan in plans:
            if plan.status == "planned":
                records.append(_failed_record(
                    plan.seed_id, method_name, revision, padding, source_digest,
                    label_digest, checkpoint_id, label_shape, source_shape, scale_xy,
                    family, crop_offset, budget_reason,
                ))
            elif plan.status == "missing":
                records.append(_missing_record(
                    plan.seed_id, method_name, revision, padding, source_digest,
                    label_digest, checkpoint_id, label_shape, source_shape, scale_xy,
                    family=family, crop_offset=crop_offset,
                ))
            else:
                records.append(_empty_record(
                    plan.seed_id, method_name, revision, padding, source_digest,
                    label_digest, checkpoint_id, label_shape, source_shape, scale_xy,
                    family=family, crop_offset=crop_offset, reason=plan.reason,
                ))
        return _batch(records, method_name, revision, padding, source_digest, budget_reason)

    for index, (seed_id, plan) in enumerate(zip(ids, plans, strict=True)):
        if _cancelled(cancel):
            records.extend(
                _cancelled_record(value, method_name, revision, padding, source_digest)
                for value in ids[index:]
            )
            break
        if plan.status == "missing":
            records.append(_missing_record(
                seed_id, method_name, revision, padding, source_digest,
                label_digest, checkpoint_id, label_shape, source_shape, scale_xy,
                family=family, crop_offset=crop_offset,
            ))
            continue

            continue
        if plan.status == "empty":
            records.append(_empty_record(
                seed_id, method_name, revision, padding, source_digest,
                label_digest, checkpoint_id, label_shape, source_shape, scale_xy,
                family=family, crop_offset=crop_offset, reason=plan.reason,
            ))
            continue
        try:
            record = _extract_one(
                analysis=analysis,
                source_tensor=source_tensor,
                labels=labels,
                seed_id=seed_id,
                method=method_name,
                family=family,
                revision=revision,
                source_digest=source_digest,
                label_digest=label_digest,
                checkpoint_id=checkpoint_id,
                label_shape=label_shape,
                source_shape=source_shape,
                scale_xy=scale_xy,
                padding=padding,
                affine=affine,
                crop_offset=crop_offset,
                plan=plan,
            )
            if _cancelled(cancel):
                record = _cancelled_record(seed_id, method_name, revision, padding, source_digest)
        except Exception as exc:
            # Cancellation is based only on the explicit state/callback, never
            # on text in an unrelated runtime error.
            if _cancelled(cancel):
                record = _cancelled_record(seed_id, method_name, revision, padding, source_digest)
            else:
                record = _failed_record(
                    seed_id, method_name, revision, padding, source_digest,
                    label_digest, checkpoint_id, label_shape, source_shape, scale_xy,
                    family, crop_offset,
                    str(exc) or type(exc).__name__,
                )
        records.append(record)
        if record.status == "cancelled":
            records.extend(
                _cancelled_record(value, method_name, revision, padding, source_digest)
                for value in ids[index + 1:]
            )
            break

    return _batch(records, method_name, revision, padding, source_digest)


def _normalize_seed_ids(seed_ids: Sequence[int]) -> tuple[int, ...]:
    result = []
    for value in seed_ids:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
            raise ValueError("Seed IDs must be positive integers.")
        seed_id = int(value)
        if seed_id < 1:
            raise ValueError("Seed IDs must be positive integers.")
        result.append(seed_id)
    if len(set(result)) != len(result):
        raise ValueError("Seed IDs must be unique within one crop request.")
    return tuple(result)


def _normalize_padding(padding_px: int) -> int:
    if isinstance(padding_px, (bool, np.bool_)) or not isinstance(padding_px, Integral):
        raise ValueError("padding_px must be a non-negative integer.")
    padding = int(padding_px)
    if padding < 0:
        raise ValueError("padding_px must be a non-negative integer.")
    return padding


def _normalize_crop_offset(values) -> tuple[int, int]:
    try:
        values = tuple(values)
    except TypeError as exc:
        raise ValueError("The analysis crop offset must contain integer x and y.") from exc
    if (len(values) != 2 or any(
        isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
        for value in values
    )):
        raise ValueError("The analysis crop offset must contain integer x and y.")
    return int(values[0]), int(values[1])


def _selected_labels_and_source_shape(analysis, method: str, result):
    from seedvision.learning.contracts import LearnedInstanceResult
    from seedvision.segmentation.procedural import ProceduralInstanceResult

    labels = np.asarray(getattr(result, "labels", None))
    if labels.ndim != 2 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("The selected result does not contain a 2-D integer label map.")
    if 0 in labels.shape:
        raise ValueError("The selected label map has empty dimensions.")
    if np.any(labels < 0):
        raise ValueError("The selected label map contains negative identities.")
    if int(labels.max(initial=0)) > np.iinfo(np.int32).max:
        raise ValueError("The selected label map exceeds the supported identity range.")

    if method == "procedural_instances":
        if not isinstance(result, ProceduralInstanceResult):
            raise ValueError("procedural_instances must be a ProceduralInstanceResult.")
        source_shape = _normalize_source_shape(result.source_shape)
    else:
        if not isinstance(result, LearnedInstanceResult):
            raise ValueError(f"{method} must be a LearnedInstanceResult.")
        source_shape = _raster_shape(analysis.layers.valid_mask)
        if labels.shape != source_shape:
            raise ValueError("Learned labels must align to the complete analysis crop.")
        expected_family = {
            "unet_instances": "unet_watershed",
            "stardist_instances": "stardist",
            "cellpose_sam_instances": "cellpose_sam",
        }[method]
        family = getattr(result.family, "value", str(result.family))
        if family != expected_family:
            raise ValueError(f"The learned result family does not match {method}.")
        return labels, source_shape, family

    layer_shape = _raster_shape(analysis.layers.valid_mask)
    if min(source_shape, default=0) < 1:
        raise ValueError("The analysis crop dimensions are empty.")
    if source_shape != layer_shape:
        raise ValueError("Procedural source_shape must match the analysis crop dimensions.")
    return labels, source_shape, "procedural"


def _raster_shape(raster) -> tuple[int, int]:
    shape = tuple(getattr(raster, "shape", ()))
    if len(shape) >= 2:
        return _normalize_source_shape(shape[-2:])
    raise ValueError("The analysis crop dimensions are unavailable.")


def _normalize_source_shape(values) -> tuple[int, int]:
    values = tuple(values)
    if (len(values) != 2 or any(
        isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
        for value in values
    )):
        raise ValueError("Analysis crop dimensions must be positive integers.")
    shape = int(values[0]), int(values[1])
    if min(shape) < 1:
        raise ValueError("Analysis crop dimensions must be positive integers.")
    return shape


def _cuda_corrected_tensor(analysis):
    import torch

    source = getattr(analysis.calibration, "gpu_corrected_bgr", None)
    if source is None:
        raise ValueError("CUDA corrected-image tensor is unavailable.")
    if hasattr(source, "gpu_tensor"):
        tensor = source.gpu_tensor()
    else:
        tensor = source
    if not torch.is_tensor(tensor):
        raise ValueError("CUDA corrected-image source is not a tensor.")
    if tensor.device.type != "cuda":
        raise ValueError("CUDA corrected-image tensor is unavailable on this device.")
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 4 or tensor.shape[0] != 1 or tensor.shape[1] != 3:
        raise ValueError("Corrected image tensor must be a single NCHW BGR image.")
    if tensor.dtype not in (torch.float32, torch.uint8):
        raise ValueError("Corrected BGR tensor must use float32 or uint8 samples.")
    corrected = getattr(analysis.calibration, "corrected_bgr", None)
    corrected_shape = tuple(getattr(corrected, "shape", ()))
    if len(corrected_shape) < 2:
        raise ValueError("Corrected CPU image frame dimensions are unavailable.")
    corrected_height, corrected_width = map(int, corrected_shape[:2])
    if (corrected_height, corrected_width) != tuple(map(int, tensor.shape[-2:])):
        raise ValueError(
            "CUDA corrected-image tensor dimensions do not match the corrected CPU image frame."
        )
    return tensor


def _plan_one(labels, seed_id, source_shape, label_shape, crop_offset,
              frame_shape, padding, maximum_label) -> _CropPlan:
    if seed_id > maximum_label:
        return _CropPlan(seed_id, "missing")
    label_y, label_x = np.nonzero(labels == seed_id)
    if label_x.size == 0:
        return _CropPlan(seed_id, "missing")
    source_height, source_width = source_shape
    offset_x, offset_y = crop_offset
    frame_width, frame_height = frame_shape
    label_height, label_width = label_shape
    scale_x = source_width / label_width
    scale_y = source_height / label_height
    local_x0 = min(source_width, max(0, math.ceil(float(label_x.min()) * scale_x - 0.5)))
    local_x1 = min(source_width, max(0, math.ceil(float(label_x.max() + 1) * scale_x - 0.5)))
    local_y0 = min(source_height, max(0, math.ceil(float(label_y.min()) * scale_y - 0.5)))
    local_y1 = min(source_height, max(0, math.ceil(float(label_y.max() + 1) * scale_y - 0.5)))
    if local_x1 <= local_x0 or local_y1 <= local_y0:
        return _CropPlan(seed_id, "empty", reason="The selected label has no source-resolution pixel centers.")
    requested = (
        offset_x + local_x0 - padding,
        offset_y + local_y0 - padding,
        offset_x + local_x1 + padding,
        offset_y + local_y1 + padding,
    )
    corrected = (
        max(0, requested[0]), max(0, requested[1]),
        min(frame_width, requested[2]), min(frame_height, requested[3]),
    )
    if corrected[2] <= corrected[0] or corrected[3] <= corrected[1]:
        return _CropPlan(seed_id, "empty", requested, corrected,
                         reason="The requested crop does not intersect the corrected image frame.")
    touches_analysis = bool(
        np.any(label_x == 0) or np.any(label_x == label_width - 1)
        or np.any(label_y == 0) or np.any(label_y == label_height - 1)
    )
    return _CropPlan(
        seed_id, "planned", requested, corrected,
        (local_x0, local_y0, local_x1, local_y1), touches_analysis,
    )


def _extract_one(
    *, analysis, source_tensor, labels, seed_id, method, family, revision,
    source_digest, label_digest, checkpoint_id, label_shape, source_shape,
    scale_xy, padding, affine,
    crop_offset,
    plan,
) -> SeedCropRecord:
    source_height, source_width = source_shape
    offset_x, offset_y = (int(value) for value in analysis.crop_offset)
    label_height, label_width = label_shape
    scale_x = source_width / label_width
    scale_y = source_height / label_height
    label_y, label_x = np.nonzero(labels == seed_id)
    local_x0, local_y0, local_x1, local_y1 = plan.label_local_bbox
    requested = plan.requested_bbox
    frame_height, frame_width = map(int, source_tensor.shape[-2:])
    x0, y0, x1, y1 = plan.corrected_bbox

    # Clone only the returned crop so consumers cannot mutate the resident
    # corrected tensor through a view.
    image_crop = source_tensor[:, :, y0:y1, x0:x1].clone()
    mask_crop, foreign_ids, foreign_count = _aligned_local_mask(
        labels, seed_id, source_shape, label_shape,
        corrected_bbox=(x0, y0, x1, y1), crop_offset=(offset_x, offset_y),
        device=source_tensor.device,
    )
    if not bool(mask_crop.any().item()):
        return _empty_record(
            seed_id, method, revision, padding, source_digest, label_digest,
            checkpoint_id, label_shape, source_shape, scale_xy,
            family=family, crop_offset=crop_offset,
            reason="The selected label has no aligned source-resolution pixels.",
        )

    source_to_crop_matrix = np.asarray(
        ((1.0, 0.0, -float(x0)), (0.0, 1.0, -float(y0)), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    ) @ affine
    label_touches_analysis_crop = bool(
        np.any(label_x == 0) or np.any(label_x == label_width - 1)
        or np.any(label_y == 0) or np.any(label_y == label_height - 1)
    )
    seed_bbox = (
        offset_x + local_x0, offset_y + local_y0,
        offset_x + local_x1, offset_y + local_y1,
    )
    touches_corrected_frame = bool(
        seed_bbox[0] == 0 or seed_bbox[1] == 0
        or seed_bbox[2] == frame_width or seed_bbox[3] == frame_height
    )
    clipped_by_frame = bool(
        x0 != requested[0] or y0 != requested[1]
        or x1 != requested[2] or y1 != requested[3]
    )
    provenance = {
        "schema": "seedfiddle.seed-crop.v1",
        "method": method,
        "family": family,
        "seed_id": seed_id,
        "result_revision": revision,
        "source_sha256": source_digest,
        "label_sha256": label_digest,
        "checkpoint_id": checkpoint_id,
        "label_shape": label_shape,
        "source_shape": source_shape,
        "scale_xy": scale_xy,
        "padding_px": padding,
        "requested_bbox": requested,
        "corrected_bbox": (x0, y0, x1, y1),
        "source_to_corrected": _matrix_tuple(affine),
        "source_to_crop": _matrix_tuple(source_to_crop_matrix),
    }
    identity = sha256(
        json.dumps(provenance, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()
    return SeedCropRecord(
        seed_id=seed_id,
        status="available",
        image_crop=image_crop,
        mask_crop=mask_crop,
        corrected_bbox=(x0, y0, x1, y1),
        requested_bbox=requested,
        source_to_crop=_matrix_tuple(source_to_crop_matrix),
        source_to_corrected=_matrix_tuple(affine),
        source_sha256=source_digest,
        result_revision=revision,
        method=method,
        label_family=family,
        checkpoint_id=checkpoint_id,
        label_sha256=label_digest,
        artifact_identity=identity,
        label_shape=label_shape,
        source_shape=source_shape,
        analysis_crop_offset=crop_offset,
        label_to_source_scale_xy=scale_xy,
        padding_px=padding,
        touches_corrected_frame=touches_corrected_frame,
        touches_analysis_crop=label_touches_analysis_crop,
        clipped_by_corrected_frame=clipped_by_frame,
        source_validity="unknown",
        foreign_seed_ids=foreign_ids,
        foreign_pixel_count=foreign_count,
    )


def _aligned_local_mask(labels, seed_id, source_shape, label_shape, *, corrected_bbox, crop_offset, device):
    import torch

    x0, y0, x1, y1 = corrected_bbox
    offset_x, offset_y = crop_offset
    source_height, source_width = source_shape
    label_height, label_width = label_shape
    local_x = np.arange(x0, x1, dtype=np.int64) - offset_x
    local_y = np.arange(y0, y1, dtype=np.int64) - offset_y
    valid_x = (local_x >= 0) & (local_x < source_width)
    valid_y = (local_y >= 0) & (local_y < source_height)
    label_x = np.floor((local_x + 0.5) * label_width / source_width).astype(np.int64)
    label_y = np.floor((local_y + 0.5) * label_height / source_height).astype(np.int64)
    label_x = np.clip(label_x, 0, label_width - 1)
    label_y = np.clip(label_y, 0, label_height - 1)

    if not np.any(valid_x) or not np.any(valid_y):
        return torch.zeros((y1 - y0, x1 - x0), dtype=torch.bool, device=device), (), 0

    tile_x0, tile_x1 = int(label_x[valid_x].min()), int(label_x[valid_x].max()) + 1
    tile_y0, tile_y1 = int(label_y[valid_y].min()), int(label_y[valid_y].max()) + 1
    # Copy only the label tile covered by this crop, never the full corrected
    # image or an upscaled full-frame label map.
    tile_values = np.ascontiguousarray(
        labels[tile_y0:tile_y1, tile_x0:tile_x1], dtype=np.int32
    )
    tile = torch.as_tensor(tile_values, device=device)
    x_index = torch.as_tensor(label_x - tile_x0, dtype=torch.long, device=device)
    y_index = torch.as_tensor(label_y - tile_y0, dtype=torch.long, device=device)
    aligned_labels = tile.index_select(0, y_index).index_select(1, x_index)
    valid = (
        torch.as_tensor(valid_y, dtype=torch.bool, device=device)[:, None]
        & torch.as_tensor(valid_x, dtype=torch.bool, device=device)[None, :]
    )
    selected = (aligned_labels == int(seed_id)) & valid
    foreign = (aligned_labels > 0) & (aligned_labels != int(seed_id)) & valid
    if bool(foreign.any().item()):
        foreign_ids = tuple(
            int(value) for value in torch.unique(aligned_labels[foreign]).detach().cpu().tolist()
        )
        foreign_count = int(foreign.sum().item())
    else:
        foreign_ids, foreign_count = (), 0
    return selected, foreign_ids, foreign_count


def _missing_record(seed_id, method, revision, padding, source_digest,
                    label_digest, checkpoint_id, label_shape, source_shape, scale_xy,
                    *, family=None, crop_offset=None):
    return _status_record(
        seed_id, "missing", method, revision, padding, source_digest,
        reason="The selected seed ID is not present in the label map.",
        checkpoint_id=checkpoint_id, label_digest=label_digest,
        label_shape=label_shape, source_shape=source_shape, scale_xy=scale_xy,
        family=family, crop_offset=crop_offset,
    )


def _empty_record(seed_id, method, revision, padding, source_digest,
                  label_digest, checkpoint_id, label_shape, source_shape, scale_xy,
                  *, family, crop_offset, reason):
    return _status_record(
        seed_id, "empty", method, revision, padding, source_digest, reason,
        checkpoint_id=checkpoint_id, label_digest=label_digest,
        label_shape=label_shape, source_shape=source_shape, scale_xy=scale_xy,
        family=family, crop_offset=crop_offset,
    )


def _failed_record(seed_id, method, revision, padding, source_digest,
                   label_digest, checkpoint_id, label_shape, source_shape, scale_xy,
                   family, crop_offset, reason):
    return _status_record(
        seed_id, "failed", method, revision, padding, source_digest, reason,
        checkpoint_id=checkpoint_id, label_digest=label_digest,
        label_shape=label_shape, source_shape=source_shape, scale_xy=scale_xy,
        family=family, crop_offset=crop_offset,
    )


def _unavailable_record(seed_id, method, revision, padding, source_digest, reason):
    return _status_record(seed_id, "unavailable", method, revision, padding,
                          source_digest, reason)


def _cancelled_record(seed_id, method, revision, padding, source_digest):
    return _status_record(seed_id, "cancelled", method, revision, padding,
                          source_digest, "Extraction was cancelled.")


def _status_record(seed_id, status, method, revision, padding, source_digest, reason,
                   *, checkpoint_id="", label_digest=None, label_shape=None,
                   source_shape=None, scale_xy=None, family=None, crop_offset=None):
    return SeedCropRecord(
        seed_id=seed_id,
        status=status,
        image_crop=None,
        mask_crop=None,
        corrected_bbox=None,
        requested_bbox=None,
        source_to_crop=None,
        source_to_corrected=None,
        source_sha256=source_digest,
        result_revision=revision,
        method=method,
        label_family=family,
        checkpoint_id=checkpoint_id,
        label_sha256=label_digest,
        artifact_identity=None,
        label_shape=label_shape,
        source_shape=source_shape,
        analysis_crop_offset=crop_offset,
        label_to_source_scale_xy=scale_xy,
        padding_px=padding,
        touches_corrected_frame=None,
        touches_analysis_crop=None,
        clipped_by_corrected_frame=None,
        source_validity="unknown",
        foreign_seed_ids=(),
        foreign_pixel_count=0,
        reason=reason,
    )


def _batch(records, method, revision, padding, source_digest, reason=""):
    states = {record.status for record in records}
    if "cancelled" in states:
        state = "cancelled"
    elif not states:
        state = "empty"
    elif states <= {"available", "missing", "empty"}:
        state = "complete"
    elif "available" in states:
        state = "partial"
    elif "failed" in states:
        state = "failed"
    else:
        state = "unavailable"
    return SeedCropBatch(
        state=state,
        method=method,
        result_revision=revision,
        padding_px=padding,
        records=tuple(records),
        source_sha256=source_digest,
        reason=reason,
    )


def _cancelled(cancel) -> bool:
    if cancel is None:
        return False
    is_set = getattr(cancel, "is_set", None)
    if callable(is_set):
        return bool(is_set())
    if callable(cancel):
        return bool(cancel())
    raise TypeError("cancel must be a callback, event-like object, or None.")


def _matrix_tuple(matrix) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(value) for value in row) for row in np.asarray(matrix))
