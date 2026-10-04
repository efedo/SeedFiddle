"""Bounded, provenance-bound comparison of benchmark checkpoint outputs.

These comparisons consume only verified checkpoint files. They do not move
production tensors to the CPU and do not treat agreement with a reference as
independent scientific validation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np

from seedvision.benchmarking.checkpoints import (
    BenchmarkCheckpoint,
    CheckpointError,
    CheckpointMismatch,
    CheckpointIdentity,
    CoordinateFrame,
    ExecutionStatus,
    RasterRole,
    load_checkpoint,
)
from seedvision.learning.metrics import (
    _overlap_iou_pairs,
    evaluate_instances,
)
from seedvision.segmentation.procedural_fit import (
    _maximum_weight_assignment,
    _preflight_maximum_weight_assignment,
)


MAX_CONFORMANCE_REPORT_BYTES = 1024 * 1024
MAX_PRIMITIVE_ELEMENTS = 100_000_000
MAX_INSTANCE_PIXELS = 16_000_000
MAX_MATCHED_ID_RECORDS = 4096
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_COMPARE_CHUNK_ELEMENTS = 1_000_000


class ConformanceError(ValueError):
    """Raised when a checkpoint pair is stale, unavailable, or incompatible."""


@dataclass(frozen=True, slots=True)
class CheckpointPairContract:
    """Exact adapter for the two outputs of one stage comparison.

    The output digests pin the complete recipe/backend/environment identities.
    Shared source, frame, and upstream fields make the common stage input
    explicit and are checked against both loaded manifests.
    """

    reference_checkpoint_sha256: str
    candidate_checkpoint_sha256: str
    reference_identity: CheckpointIdentity
    candidate_identity: CheckpointIdentity
    comparison_scope: str = "stage"

    def __post_init__(self) -> None:
        for field in ("reference_checkpoint_sha256", "candidate_checkpoint_sha256"):
            value = getattr(self, field)
            if not isinstance(value, str) or not _SHA256.fullmatch(value):
                raise ValueError(f"{field} must be a lowercase SHA-256 digest.")
        if not isinstance(self.reference_identity, CheckpointIdentity):
            raise ValueError("reference_identity must be the exact expected CheckpointIdentity.")
        if not isinstance(self.candidate_identity, CheckpointIdentity):
            raise ValueError("candidate_identity must be the exact expected CheckpointIdentity.")
        if self.reference_identity.source_sha256 != self.candidate_identity.source_sha256:
            raise ValueError("Reference and candidate identities must bind the same source SHA-256.")
        if self.reference_identity.coordinate_frame != self.candidate_identity.coordinate_frame:
            raise ValueError("Reference and candidate identities must bind the same coordinate frame.")
        if self.comparison_scope not in {"stage", "pipeline"}:
            raise ValueError("comparison_scope must be 'stage' or 'pipeline'.")
        if (
            self.comparison_scope == "stage"
            and self.reference_identity.upstream_checkpoint_sha256 != self.candidate_identity.upstream_checkpoint_sha256
        ):
            raise ValueError("Reference and candidate identities must bind the same upstream checkpoint.")

    @property
    def source_sha256(self) -> str:
        return self.reference_identity.source_sha256

    @property
    def coordinate_frame(self) -> CoordinateFrame:
        return self.reference_identity.coordinate_frame

    @property
    def upstream_checkpoint_sha256(self) -> str | None:
        if self.reference_identity.upstream_checkpoint_sha256 == self.candidate_identity.upstream_checkpoint_sha256:
            return self.reference_identity.upstream_checkpoint_sha256
        return None

    @property
    def reference_stage_id(self) -> str:
        return self.reference_identity.stage_id

    @property
    def candidate_stage_id(self) -> str:
        return self.candidate_identity.stage_id


@dataclass(frozen=True, slots=True)
class PrimitiveArrayContract:
    """Declared numerical output shape, dtype, and optional valid range."""

    dtype: str
    shape: tuple[int, ...]
    value_range: tuple[int | float, int | float] | None = None

    def __post_init__(self) -> None:
        try:
            dtype = np.dtype(self.dtype)
        except (TypeError, ValueError) as error:
            raise ValueError("Primitive output dtype is invalid.") from error
        if dtype.hasobject or dtype.kind not in "biuf":
            raise ValueError("Primitive outputs require a real numeric or boolean dtype.")
        if dtype.itemsize > 8:
            raise ValueError("Primitive output dtypes are limited to 64-bit values.")
        object.__setattr__(self, "dtype", dtype.str)
        if not isinstance(self.shape, (tuple, list)) or len(self.shape) > 8:
            raise ValueError("Primitive output shape must have at most eight dimensions.")
        if any(type(part) is not int or part < 0 for part in self.shape):
            raise ValueError("Primitive output dimensions must be non-negative integers.")
        shape = tuple(self.shape)
        element_count = math.prod(shape) if shape else 1
        if element_count > MAX_PRIMITIVE_ELEMENTS:
            raise ValueError("Primitive output shape exceeds the element safety limit.")
        object.__setattr__(self, "shape", shape)
        if self.value_range is not None:
            if not isinstance(self.value_range, (tuple, list)) or len(self.value_range) != 2:
                raise ValueError("Primitive value_range must be a finite ordered pair.")
            low, high = self.value_range
            if (
                isinstance(low, bool) or isinstance(high, bool)
                or not isinstance(low, (int, float)) or not isinstance(high, (int, float))
            ):
                raise ValueError("Primitive value_range must be finite and ordered.")
            if (
                (isinstance(low, float) and not math.isfinite(low))
                or (isinstance(high, float) and not math.isfinite(high))
                or low > high
            ):
                raise ValueError("Primitive value_range must be finite and ordered.")
            object.__setattr__(
                self,
                "value_range",
                (int(low) if type(low) is int else float(low), int(high) if type(high) is int else float(high)),
            )


def _finite_number(name: str, value: Any, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number.")
    try:
        numeric = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{name} must be a finite number.") from error
    if not math.isfinite(numeric) or (minimum is not None and numeric < minimum):
        qualifier = "finite and non-negative" if minimum == 0 else "finite"
        raise ValueError(f"{name} must be {qualifier}.")
    return numeric


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as error:
        raise ConformanceError("Conformance result contains non-JSON metadata.") from error


def _identity_record(checkpoint: BenchmarkCheckpoint) -> dict[str, Any]:
    identity = checkpoint.identity
    payload = identity.to_dict()
    identity_bytes = _canonical_json(payload)
    environment_bytes = _canonical_json(payload["environment_versions"])
    record = {
        "checkpoint_sha256": checkpoint.checkpoint_sha256,
        "identity_sha256": hashlib.sha256(identity_bytes).hexdigest(),
        "source_sha256": identity.source_sha256,
        "coordinate_frame": identity.coordinate_frame.to_dict(),
        "stage_id": identity.stage_id,
        "recipe": {
            "id": identity.recipe_id,
            "version": identity.recipe_version,
            "sha256": identity.recipe_sha256,
        },
        "graph_sha256": identity.graph_sha256,
        "settings_sha256": identity.settings_sha256,
        "backend": {"id": identity.backend_id, "version": identity.backend_version},
        "environment_versions_sha256": hashlib.sha256(environment_bytes).hexdigest(),
        "environment_versions": (
            payload["environment_versions"] if len(environment_bytes) <= 8192 else None
        ),
        "upstream_checkpoint_sha256": identity.upstream_checkpoint_sha256,
        "execution": checkpoint.execution.to_dict(),
        "measurement": checkpoint.measurement.to_dict(),
    }
    return record


def _load_pair(
    root: Path | str,
    pair: CheckpointPairContract,
) -> tuple[BenchmarkCheckpoint, BenchmarkCheckpoint]:
    if not isinstance(pair, CheckpointPairContract):
        raise TypeError("pair must be a CheckpointPairContract.")
    try:
        reference = load_checkpoint(root, pair.reference_checkpoint_sha256)
        candidate = load_checkpoint(root, pair.candidate_checkpoint_sha256)
    except CheckpointError as error:
        raise ConformanceError(f"Checkpoint pair could not be verified: {error}") from error

    for label, checkpoint, expected_identity in (
        ("reference", reference, pair.reference_identity),
        ("candidate", candidate, pair.candidate_identity),
    ):
        if checkpoint.identity != expected_identity:
            raise CheckpointMismatch(
                f"{label} checkpoint identity differs from the exact stage adapter "
                "(source, frame, stage, recipe, graph, settings, backend, environment, or upstream)."
            )
        if checkpoint.execution.status in {ExecutionStatus.FAILED, ExecutionStatus.UNSUPPORTED}:
            raise ConformanceError(
                f"{label} execution is {checkpoint.execution.status.value}; failed or unsupported outputs cannot be compared."
            )
    return reference, candidate


def _check_array_name(checkpoint: BenchmarkCheckpoint, name: str, label: str) -> np.ndarray:
    if not isinstance(name, str) or name not in checkpoint.arrays:
        raise ConformanceError(f"{label} checkpoint does not contain output array {name!r}.")
    return checkpoint.arrays[name]


def _validate_primitive_raster_binding(
    reference: BenchmarkCheckpoint,
    candidate: BenchmarkCheckpoint,
    *,
    reference_array: str,
    candidate_array: str,
) -> None:
    """Keep primitive comparisons bound to any declared raster semantics."""

    left_role = reference.raster_roles.get(reference_array)
    right_role = candidate.raster_roles.get(candidate_array)
    if left_role is None and right_role is None:
        return
    if left_role is None or right_role is None:
        raise ConformanceError("Raster role is missing from one primitive output checkpoint.")
    if left_role != right_role:
        raise ConformanceError("Raster role dtype, layout, or representation differs between checkpoints.")
    semantic_fields = {
        "image": "image_semantics",
        "mask": "mask_semantics",
        "instances": "instance_semantics",
    }
    field = semantic_fields.get(left_role.kind)
    if field is not None and getattr(reference.identity, field) != getattr(candidate.identity, field):
        label = {"image": "Image", "mask": "Mask", "instances": "Instance"}[left_role.kind]
        raise ConformanceError(f"{label} raster representation semantics differ between checkpoints.")


def _reference_relative_close(
    candidate: np.ndarray,
    reference: np.ndarray,
    *,
    atol: float,
    rtol: float,
) -> np.ndarray:
    """Compare finite float64-representable values without overflow.

    Scaling each pair by its largest magnitude keeps the difference bounded
    by two. The right-hand (reference) value sets the relative tolerance.
    """

    candidate64 = candidate.astype(np.float64, copy=False)
    reference64 = reference.astype(np.float64, copy=False)
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        absolute_error = np.abs(candidate64 - reference64)
        tolerance = atol + rtol * np.abs(reference64)
        direct = np.isfinite(absolute_error) & np.isfinite(tolerance)
        result = np.empty(candidate64.shape, dtype=np.bool_)
        result[direct] = absolute_error[direct] <= tolerance[direct]

        # Direct subtraction/bounds preserve exact ordinary tolerance edges.
        # Normalize only values whose difference or bound overflowed float64.
        fallback = ~direct
        if np.any(fallback):
            scale = np.maximum(
                np.maximum(np.abs(candidate64[fallback]), np.abs(reference64[fallback])),
                atol,
            )
            safe_scale = np.where(scale == 0.0, 1.0, scale)
            scaled_error = np.abs(
                candidate64[fallback] / safe_scale - reference64[fallback] / safe_scale
            )
            scaled_tolerance = (
                atol / safe_scale
                + rtol * np.abs(reference64[fallback] / safe_scale)
            )
            result[fallback] = scaled_error <= scaled_tolerance
    return result


def _report(
    *,
    kind: str,
    pair: CheckpointPairContract,
    reference: BenchmarkCheckpoint,
    candidate: BenchmarkCheckpoint,
    passed: bool,
    conformance: Mapping[str, Any],
) -> dict[str, Any]:
    result = {
        "schema": "seedfiddle.stage_conformance.v1",
        "comparison_kind": kind,
        "input_binding": {
            "comparison_scope": pair.comparison_scope,
            "source_sha256": pair.source_sha256,
            "coordinate_frame": pair.coordinate_frame.to_dict(),
            "upstream_checkpoint_sha256": pair.upstream_checkpoint_sha256,
            "reference_upstream_checkpoint_sha256": pair.reference_identity.upstream_checkpoint_sha256,
            "candidate_upstream_checkpoint_sha256": pair.candidate_identity.upstream_checkpoint_sha256,
        },
        "checkpoint_identities": {
            "reference": _identity_record(reference),
            "candidate": _identity_record(candidate),
        },
        "conformance": {"passed": bool(passed), **dict(conformance)},
        "scientific_metrics": {
            "status": "not_evaluated",
            "independent_truth": False,
            "claim": "Agreement with a reference is conformance evidence only, not independent scientific validation.",
        },
    }
    if len(_canonical_json(result)) > MAX_CONFORMANCE_REPORT_BYTES:
        raise ConformanceError("Conformance result exceeds the one-megabyte JSON report limit.")
    return result


def compare_primitive_outputs(
    root: Path | str,
    pair: CheckpointPairContract,
    *,
    reference_array: str,
    candidate_array: str,
    contract: PrimitiveArrayContract,
    atol: float = 0.0,
    rtol: float = 0.0,
) -> dict[str, Any]:
    """Compare declared primitive arrays with bounded, chunked tolerances."""

    if not isinstance(contract, PrimitiveArrayContract):
        raise TypeError("contract must be a PrimitiveArrayContract.")
    atol = _finite_number("atol", atol, minimum=0.0)
    rtol = _finite_number("rtol", rtol, minimum=0.0)
    if np.dtype(contract.dtype).kind != "f" and (atol != 0 or rtol != 0):
        raise ValueError("Absolute and relative tolerances apply only to floating-point primitive outputs.")
    reference, candidate = _load_pair(root, pair)
    left = _check_array_name(reference, reference_array, "reference")
    right = _check_array_name(candidate, candidate_array, "candidate")
    _validate_primitive_raster_binding(
        reference,
        candidate,
        reference_array=reference_array,
        candidate_array=candidate_array,
    )
    for label, array in (("reference", left), ("candidate", right)):
        if array.dtype.str != contract.dtype:
            raise ConformanceError(f"{label} output dtype {array.dtype.str} differs from declared {contract.dtype}.")
        if tuple(array.shape) != contract.shape:
            raise ConformanceError(f"{label} output shape {array.shape} differs from declared {contract.shape}.")
        if array.size > MAX_PRIMITIVE_ELEMENTS:
            raise ConformanceError("Primitive output exceeds the element comparison limit.")
        if array.dtype.kind == "f":
            flat = array.reshape(-1)
            for start in range(0, flat.size, _COMPARE_CHUNK_ELEMENTS):
                if not np.isfinite(flat[start : start + _COMPARE_CHUNK_ELEMENTS]).all():
                    raise ConformanceError(f"{label} primitive output contains NaN or infinity.")
        if contract.value_range is not None and array.size:
            low, high = contract.value_range
            minimum, maximum = np.min(array).item(), np.max(array).item()
            if minimum < low or maximum > high:
                raise ConformanceError(f"{label} primitive output exceeds the declared value range.")

    count = int(left.size)
    within_tolerance = 0
    max_absolute_error: float | int | None = 0.0
    max_relative_error: float | None = None
    absolute_error_overflow = False
    relative_error_overflow = False
    left_flat = left.reshape(-1)
    right_flat = right.reshape(-1)
    for start in range(0, count, _COMPARE_CHUNK_ELEMENTS):
        end = min(count, start + _COMPARE_CHUNK_ELEMENTS)
        lhs = left_flat[start:end]
        rhs = right_flat[start:end]
        if left.dtype.kind == "f":
            within_tolerance += int(np.count_nonzero(
                _reference_relative_close(rhs, lhs, atol=atol, rtol=rtol)
            ))
            lhs64 = lhs.astype(np.float64, copy=False)
            rhs64 = rhs.astype(np.float64, copy=False)
            scale = np.maximum(np.maximum(np.abs(lhs64), np.abs(rhs64)), atol)
            safe_scale = np.where(scale == 0.0, 1.0, scale)
            scaled_error = np.abs((rhs64 / safe_scale) - (lhs64 / safe_scale))
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                absolute = np.abs(lhs64 - rhs64)
                if absolute.size and np.isfinite(absolute).all():
                    if not absolute_error_overflow:
                        max_absolute_error = max(float(max_absolute_error or 0.0), float(np.max(absolute)))
                elif absolute.size:
                    absolute_error_overflow = True
                nonzero = np.abs(lhs64) > 0.0
                if np.any(nonzero):
                    relative = np.empty(np.count_nonzero(nonzero), dtype=np.float64)
                    finite_absolute = np.isfinite(absolute[nonzero])
                    reference_values = np.abs(lhs64[nonzero])
                    relative[finite_absolute] = (
                        absolute[nonzero][finite_absolute] / reference_values[finite_absolute]
                    )
                    relative[~finite_absolute] = (
                        scaled_error[nonzero][~finite_absolute]
                        / np.abs(lhs64[nonzero][~finite_absolute] / safe_scale[nonzero][~finite_absolute])
                    )
                    if np.isfinite(relative).all():
                        if not relative_error_overflow:
                            observed = float(np.max(relative))
                            max_relative_error = observed if max_relative_error is None else max(max_relative_error, observed)
                    else:
                        relative_error_overflow = True
        else:
            exact = lhs == rhs
            within_tolerance += int(np.count_nonzero(exact))
            if left.dtype.kind == "b":
                difference = np.logical_not(exact)
            elif left.dtype.kind == "i" and left.dtype.itemsize == 8:
                # Flip the sign bit to map signed order to uint64 order; the
                # unsigned distance then represents even INT64_MIN-to-MAX
                # differences without overflow or float precision loss.
                sign_bit = np.uint64(1 << 63)
                left_ordered = np.bitwise_xor(lhs.astype(np.int64, copy=False).view(np.uint64), sign_bit)
                right_ordered = np.bitwise_xor(rhs.astype(np.int64, copy=False).view(np.uint64), sign_bit)
                difference = np.maximum(left_ordered, right_ordered) - np.minimum(left_ordered, right_ordered)
            elif left.dtype.kind == "i":
                left_wide = lhs.astype(np.int64, copy=False)
                right_wide = rhs.astype(np.int64, copy=False)
                difference = np.abs(left_wide - right_wide)
            else:
                left_wide = lhs.astype(np.uint64, copy=False)
                right_wide = rhs.astype(np.uint64, copy=False)
                difference = np.maximum(left_wide, right_wide) - np.minimum(left_wide, right_wide)
            if difference.size:
                observed_absolute = int(np.max(difference))
                max_absolute_error = max(int(max_absolute_error or 0), observed_absolute)
                left64 = lhs.astype(np.float64, copy=False)
                nonzero = lhs != 0
                if np.any(nonzero):
                    relative = difference[nonzero].astype(np.float64) / np.abs(left64[nonzero])
                    observed_relative = float(np.max(relative))
                    max_relative_error = (
                        observed_relative
                        if max_relative_error is None
                        else max(max_relative_error, observed_relative)
                    )

    if absolute_error_overflow:
        max_absolute_error = None
    if relative_error_overflow:
        max_relative_error = None

    total = count
    passed = within_tolerance == total and reference.execution.status == candidate.execution.status
    summary = {
        "reference_array": reference_array,
        "candidate_array": candidate_array,
        "dtype": contract.dtype,
        "shape": list(contract.shape),
        "value_range": None if contract.value_range is None else list(contract.value_range),
        "absolute_tolerance": float(atol),
        "relative_tolerance": float(rtol),
        "element_count": total,
        "within_tolerance_count": within_tolerance,
        "mismatch_count": total - within_tolerance,
        "max_absolute_error": max_absolute_error,
        "absolute_error_exceeds_float64": absolute_error_overflow,
        "max_relative_error_for_nonzero_reference": max_relative_error,
        "relative_error_exceeds_float64": relative_error_overflow,
        "reference_execution_status": reference.execution.status.value,
        "candidate_execution_status": candidate.execution.status.value,
        "execution_status_match": reference.execution.status == candidate.execution.status,
    }
    return _report(
        kind="primitive",
        pair=pair,
        reference=reference,
        candidate=candidate,
        passed=passed,
        conformance=summary,
    )


def _require_raster_pair(
    reference: BenchmarkCheckpoint,
    candidate: BenchmarkCheckpoint,
    *,
    kind: str,
    reference_array: str,
    candidate_array: str,
) -> tuple[np.ndarray, np.ndarray, RasterRole, RasterRole]:
    left = _check_array_name(reference, reference_array, "reference")
    right = _check_array_name(candidate, candidate_array, "candidate")
    left_role = reference.raster_roles.get(reference_array)
    right_role = candidate.raster_roles.get(candidate_array)
    if left_role is None or right_role is None or left_role.kind != kind or right_role.kind != kind:
        raise ConformanceError(f"Both outputs must declare the explicit {kind} raster role.")
    if left_role != right_role:
        raise ConformanceError("Raster role dtype, layout, or representation differs between checkpoints.")
    if left.shape != right.shape or left.dtype != right.dtype:
        raise ConformanceError("Raster shape or dtype differs between checkpoints.")
    if kind == "mask" and reference.identity.mask_semantics != candidate.identity.mask_semantics:
        raise ConformanceError("Mask representation semantics differ between checkpoints.")
    if kind == "instances" and reference.identity.instance_semantics != candidate.identity.instance_semantics:
        raise ConformanceError("Instance label representation semantics differ between checkpoints.")
    return left, right, left_role, right_role


def _require_empty_status_content(
    checkpoint: BenchmarkCheckpoint,
    array: np.ndarray,
    *,
    label: str,
    kind: str,
) -> None:
    if checkpoint.execution.status != ExecutionStatus.EMPTY:
        return
    flat = array.reshape(-1)
    for start in range(0, flat.size, _COMPARE_CHUNK_ELEMENTS):
        if np.any(flat[start : start + _COMPARE_CHUNK_ELEMENTS] != 0):
            raise ConformanceError(f"{label} execution is marked empty but its {kind} raster is non-empty.")


def compare_masks(
    root: Path | str,
    pair: CheckpointPairContract,
    *,
    reference_array: str,
    candidate_array: str,
) -> dict[str, Any]:
    """Compare checkpoint-bound binary masks exactly in their shared frame."""

    reference, candidate = _load_pair(root, pair)
    left, right, _, _ = _require_raster_pair(
        reference,
        candidate,
        kind="mask",
        reference_array=reference_array,
        candidate_array=candidate_array,
    )
    _require_empty_status_content(reference, left, label="reference", kind="mask")
    _require_empty_status_content(candidate, right, label="candidate", kind="mask")
    total = int(left.size)
    mismatch = 0
    left_flat, right_flat = left.reshape(-1), right.reshape(-1)
    for start in range(0, total, _COMPARE_CHUNK_ELEMENTS):
        end = min(total, start + _COMPARE_CHUNK_ELEMENTS)
        mismatch += int(np.count_nonzero(left_flat[start:end] != right_flat[start:end]))
    status_match = reference.execution.status == candidate.execution.status
    summary = {
        "reference_array": reference_array,
        "candidate_array": candidate_array,
        "raster_kind": "mask",
        "dtype": left.dtype.str,
        "shape": list(left.shape),
        "pixel_count": total,
        "mismatch_pixel_count": mismatch,
        "exactly_equal": mismatch == 0,
        "reference_execution_status": reference.execution.status.value,
        "candidate_execution_status": candidate.execution.status.value,
        "execution_status_match": status_match,
    }
    return _report(
        kind="mask",
        pair=pair,
        reference=reference,
        candidate=candidate,
        passed=mismatch == 0 and status_match,
        conformance=summary,
    )


def _compact_instance_ids(labels: np.ndarray) -> tuple[np.ndarray, dict[int, int]]:
    if labels.ndim != 2 or labels.dtype.kind not in "iu":
        raise ConformanceError("Instance rasters must be two-dimensional integer labels.")
    if labels.size > MAX_INSTANCE_PIXELS:
        raise ConformanceError(f"Instance raster exceeds the {MAX_INSTANCE_PIXELS}-pixel metric limit.")
    if labels.size and labels.dtype.kind == "i" and np.any(labels < 0):
        raise ConformanceError("Instance raster contains negative labels.")
    unique, first_positions, inverse = np.unique(
        labels,
        return_index=True,
        return_inverse=True,
    )
    positive_indices = np.flatnonzero(unique != 0)
    # Canonical instance order follows first row-major occurrence, not numeric
    # label value, so tie-breaking is stable under arbitrary ID permutations.
    positive_indices = positive_indices[np.argsort(first_positions[positive_indices], kind="stable")]
    compact_lookup = np.zeros(len(unique), dtype=np.int32)
    compact_lookup[positive_indices] = np.arange(1, len(positive_indices) + 1, dtype=np.int32)
    compact = compact_lookup[inverse].reshape(labels.shape)
    id_map = {
        int(compact_id): int(unique[unique_index])
        for compact_id, unique_index in enumerate(positive_indices, start=1)
    }
    return compact, id_map


def _matched_id_audit(
    reference: np.ndarray,
    candidate: np.ndarray,
    reference_ids: Mapping[int, int],
    candidate_ids: Mapping[int, int],
    *,
    iou_threshold: float,
    record_limit: int,
) -> tuple[list[dict[str, Any]], int]:
    pairs, _, _ = _overlap_iou_pairs(reference, candidate)
    adjacency: dict[int, set[int]] = {}
    scores: dict[tuple[int, int], float] = {}
    for iou, true_id, predicted_id, _ in pairs:
        if iou < iou_threshold:
            continue
        adjacency.setdefault(true_id, set()).add(-predicted_id)
        adjacency.setdefault(-predicted_id, set()).add(true_id)
        scores[(true_id, predicted_id)] = iou

    assignments: list[tuple[int, int, float]] = []
    unseen = set(adjacency)
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        pending = [start]
        component = {start}
        while pending:
            node = pending.pop()
            for neighbour in sorted(adjacency[node]):
                if neighbour not in component:
                    component.add(neighbour)
                    unseen.discard(neighbour)
                    pending.append(neighbour)
        true_ids = sorted(node for node in component if node > 0)
        predicted_ids = sorted(-node for node in component if node < 0)
        try:
            _preflight_maximum_weight_assignment(
                len(true_ids),
                len(predicted_ids),
                include_weights=True,
            )
        except MemoryError as error:
            raise ConformanceError(f"Instance matching exceeds the assignment budget: {error}") from error
        bonus = min(len(true_ids), len(predicted_ids)) + 1
        true_positions = {identifier: index for index, identifier in enumerate(true_ids)}
        predicted_positions = {identifier: index for index, identifier in enumerate(predicted_ids)}
        weights = np.zeros((len(true_ids), len(predicted_ids)), dtype=np.float64)
        for true_id in true_ids:
            row = true_positions[true_id]
            for negative_prediction_id in adjacency[true_id]:
                predicted_id = -negative_prediction_id
                weights[row, predicted_positions[predicted_id]] = bonus + scores[(true_id, predicted_id)]
        for row, column in _maximum_weight_assignment(weights):
            true_id, predicted_id = true_ids[row], predicted_ids[column]
            if (true_id, predicted_id) in scores:
                assignments.append((true_id, predicted_id, scores[(true_id, predicted_id)]))

    assignments.sort(key=lambda item: (item[0], item[1]))
    total = len(assignments)
    records = [
        {
            "reference_instance_id": int(reference_ids[true_id]),
            "candidate_instance_id": int(candidate_ids[predicted_id]),
            "iou": float(iou),
        }
        for true_id, predicted_id, iou in assignments[:record_limit]
    ]
    return records, total


def compare_instance_partitions(
    root: Path | str,
    pair: CheckpointPairContract,
    *,
    reference_array: str,
    candidate_array: str,
    iou_threshold: float = 0.50,
    contact_overlap_fraction: float = 0.10,
    boundary_tolerance_px: float = 2.0,
    max_matched_id_records: int = 256,
) -> dict[str, Any]:
    """Compare instance partitions, ignoring the arbitrary numeric label IDs.

    Aggregate counts and segmentation metrics come directly from
    :func:`seedvision.learning.metrics.evaluate_instances`. The compact match
    audit repeats only its published cardinality-first assignment so that
    original label IDs can accompany matched IoUs.
    """

    iou_threshold = _finite_number("iou_threshold", iou_threshold, minimum=0.0)
    if not 0 < iou_threshold <= 1:
        raise ValueError("iou_threshold must be in (0, 1].")
    contact_overlap_fraction = _finite_number(
        "contact_overlap_fraction",
        contact_overlap_fraction,
        minimum=0.0,
    )
    if not 0 <= contact_overlap_fraction <= 1:
        raise ValueError("contact_overlap_fraction must be in [0, 1].")
    boundary_tolerance_px = _finite_number(
        "boundary_tolerance_px",
        boundary_tolerance_px,
        minimum=0.0,
    )
    if type(max_matched_id_records) is not int or not 0 <= max_matched_id_records <= MAX_MATCHED_ID_RECORDS:
        raise ValueError(f"max_matched_id_records must be between 0 and {MAX_MATCHED_ID_RECORDS}.")

    reference, candidate = _load_pair(root, pair)
    left, right, _, _ = _require_raster_pair(
        reference,
        candidate,
        kind="instances",
        reference_array=reference_array,
        candidate_array=candidate_array,
    )
    _require_empty_status_content(reference, left, label="reference", kind="instance")
    _require_empty_status_content(candidate, right, label="candidate", kind="instance")
    compact_left, left_ids = _compact_instance_ids(left)
    compact_right, right_ids = _compact_instance_ids(right)
    try:
        metrics = evaluate_instances(
            compact_left,
            compact_right,
            iou_threshold=float(iou_threshold),
            contact_overlap_fraction=float(contact_overlap_fraction),
            boundary_tolerance_px=float(boundary_tolerance_px),
        ).to_dict()
        matches, matched_count = _matched_id_audit(
            compact_left,
            compact_right,
            left_ids,
            right_ids,
            iou_threshold=float(iou_threshold),
            record_limit=max_matched_id_records,
        )
    except (ValueError, MemoryError, OverflowError) as error:
        raise ConformanceError(f"Instance partition evaluator could not compare this pair: {error}") from error

    foreground_equal = np.array_equal(left > 0, right > 0)
    object_matches = metrics["true_positives"]
    partition_equivalent = bool(
        foreground_equal
        and metrics["true_instances"] == metrics["predicted_instances"] == object_matches
        and metrics["false_positives"] == 0
        and metrics["false_negatives"] == 0
        and metrics["panoptic_quality"] == 1.0
    )
    status_match = reference.execution.status == candidate.execution.status
    summary = {
        "reference_array": reference_array,
        "candidate_array": candidate_array,
        "raster_kind": "instances",
        "dtype": left.dtype.str,
        "shape": list(left.shape),
        "label_ids_ignored": True,
        "partition_equivalent": partition_equivalent,
        "foreground_equal": foreground_equal,
        "iou_threshold": float(iou_threshold),
        "contact_overlap_fraction": float(contact_overlap_fraction),
        "boundary_tolerance_px": float(boundary_tolerance_px),
        "reference_execution_status": reference.execution.status.value,
        "candidate_execution_status": candidate.execution.status.value,
        "execution_status_match": status_match,
        "partition_agreement_metrics": metrics,
        "metric_interpretation": "Existing instance metrics treat the reference output as the comparison target; it is not independent truth.",
        "matched_id_audit": {
            "assignment": "maximum-cardinality-then-summed-IoU",
            "matched_count": matched_count,
            "records": matches,
            "records_truncated": matched_count > len(matches),
        },
    }
    return _report(
        kind="instances",
        pair=pair,
        reference=reference,
        candidate=candidate,
        passed=partition_equivalent and status_match,
        conformance=summary,
    )


__all__ = [
    "MAX_CONFORMANCE_REPORT_BYTES",
    "MAX_INSTANCE_PIXELS",
    "MAX_MATCHED_ID_RECORDS",
    "MAX_PRIMITIVE_ELEMENTS",
    "CheckpointPairContract",
    "ConformanceError",
    "PrimitiveArrayContract",
    "compare_instance_partitions",
    "compare_masks",
    "compare_primitive_outputs",
]
