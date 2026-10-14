"""One-call, checkpoint-backed PlantCV image and binary-mask stages.

This bridge executes only the pinned public median-blur, fill, erode, and
dilate callables. It does not replay a recipe graph or certify scientific
accuracy. Checkpoint serialization is an explicit CPU boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from functools import lru_cache
import json
import math
from pathlib import Path
import re
import time
from typing import Any, Mapping

import numpy as np

from seedvision.benchmarking.checkpoints import (
    CheckpointCancelled,
    CheckpointIdentity,
    CheckpointLimits,
    CheckpointReference,
    CoordinateFrame,
    ExecutionRecord,
    ExecutionStatus,
    MeasurementContract,
    RasterRole,
    load_stage_input,
    publish_checkpoint,
)
from seedvision.benchmarking.recipe_catalogue import (
    RecipeCatalogueError,
    build_recipe_catalogue,
)
from seedvision.benchmarking.recipes import (
    RecipeContractError,
    RecipeIdentity,
    RecipeProvenanceMode,
    RecipeSpec,
    SourceResolution,
    StageRunRecord,
    StageRunState,
    StaleRecipeIdentity,
    validate_checkpoint_identity,
    validate_stage_run,
)
from seedvision.benchmarking.reference_runner import (
    MAX_ARRAY_BYTES,
    MAX_ARRAY_ELEMENTS,
    MAX_PARAMETERS_BYTES,
    PINNED_PLANTCV_COMMIT,
    PINNED_PLANTCV_VERSION,
    PINNED_REQUIREMENTS_SHA256,
    PlantCVReferenceRunner,
    ReferenceCancelledError,
    ReferenceExecutionError,
    ReferenceIdentityError,
    ReferenceResult,
    ReferenceTimeoutError,
    plantcv_reference_status,
)
from seedvision.benchmarking.reference_stage import (
    MAX_REFERENCE_STAGE_TIMEOUT_SECONDS,
    _cancelled,
    _strict_metadata,
    _validate_reference_identity,
)


REFERENCE_MASK_STAGE_FORMAT = "seedfiddle.reference-mask-stage-output"
REFERENCE_MASK_STAGE_VERSION = 1
SUPPORTED_REFERENCE_MASK_OPERATIONS = frozenset({
    "filter.median_blur",
    "morphology.fill",
    "morphology.erode",
    "morphology.dilate",
})

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_REFERENCE_RECIPE_IDS = frozenset({
    "plantcv_seed_binary",
    "plantcv_seed_mean",
    "plantcv_seed_otsu",
})
_INPUT_PORT = {
    "filter.median_blur": "gray",
    "morphology.fill": "mask",
    "morphology.erode": "gray",
    "morphology.dilate": "gray",
}
_OUTPUT_ARRAY = {
    "filter.median_blur": "image",
    "morphology.fill": "mask",
    "morphology.erode": "mask",
    "morphology.dilate": "mask",
}
_CALLABLES = {
    "filter.median_blur": "plantcv.plantcv.median_blur",
    "morphology.fill": "plantcv.plantcv.fill",
    "morphology.erode": "plantcv.plantcv.erode",
    "morphology.dilate": "plantcv.plantcv.dilate",
}


class ReferenceMaskStageError(ValueError):
    """A stage, input binding, or output identity violates its exact contract."""


class ReferenceMaskStageStatus(StrEnum):
    EXECUTED = "executed"
    EMPTY = "empty"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNAVAILABLE = "unavailable"
    UNSUPPORTED = "unsupported"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True, slots=True)
class ReferenceMaskStageOutcome:
    """Stage outcome and its output artifact, when successful."""

    status: ReferenceMaskStageStatus
    run_record: StageRunRecord
    checkpoint_reference: CheckpointReference | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", ReferenceMaskStageStatus(self.status))
        if not isinstance(self.run_record, StageRunRecord):
            raise TypeError("run_record must be a StageRunRecord.")
        successful = self.status in {
            ReferenceMaskStageStatus.EXECUTED,
            ReferenceMaskStageStatus.EMPTY,
        }
        if successful != (self.checkpoint_reference is not None):
            raise ValueError("Only executed or empty stages carry a checkpoint reference.")
        expected_state = {
            ReferenceMaskStageStatus.EXECUTED: StageRunState.EXECUTED,
            ReferenceMaskStageStatus.EMPTY: StageRunState.EMPTY,
            ReferenceMaskStageStatus.FAILED: StageRunState.FAILED,
            ReferenceMaskStageStatus.CANCELLED: StageRunState.FAILED,
            ReferenceMaskStageStatus.UNAVAILABLE: StageRunState.UNAVAILABLE,
            ReferenceMaskStageStatus.UNSUPPORTED: StageRunState.UNSUPPORTED,
            ReferenceMaskStageStatus.UNRESOLVED: StageRunState.UNRESOLVED,
        }[self.status]
        if self.run_record.state is not expected_state:
            raise ValueError("Stage status and immutable run-record state differ.")
        if self.checkpoint_reference is not None and (
            self.run_record.checkpoint_sha256 != self.checkpoint_reference.checkpoint_sha256
        ):
            raise ValueError("Stage run and checkpoint reference hashes differ.")


@lru_cache(maxsize=1)
def _catalogue():
    try:
        return build_recipe_catalogue()
    except (RecipeCatalogueError, OSError, ValueError) as error:
        raise ReferenceMaskStageError(f"Committed PlantCV recipe pins are unavailable: {error}") from error


def _no_artifact(
    recipe: RecipeSpec,
    stage_id: str,
    upstream_digest: str,
    status: ReferenceMaskStageStatus,
    reason: str,
) -> ReferenceMaskStageOutcome:
    compact_reason = " ".join(str(reason).split())[:512] or status.value
    state = {
        ReferenceMaskStageStatus.FAILED: StageRunState.FAILED,
        ReferenceMaskStageStatus.CANCELLED: StageRunState.FAILED,
        ReferenceMaskStageStatus.UNAVAILABLE: StageRunState.UNAVAILABLE,
        ReferenceMaskStageStatus.UNSUPPORTED: StageRunState.UNSUPPORTED,
        ReferenceMaskStageStatus.UNRESOLVED: StageRunState.UNRESOLVED,
    }[status]
    record = StageRunRecord(
        stage_id=stage_id,
        recipe_identity=recipe.identity,
        state=state,
        upstream_checkpoint_sha256=upstream_digest,
        reason=compact_reason,
    )
    validate_stage_run(recipe, record, expected_upstream_checkpoint_sha256=upstream_digest)
    return ReferenceMaskStageOutcome(status, record, None)


def _require_digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ReferenceMaskStageError(f"{name} must be a lowercase SHA-256 digest.")
    return value


def _canonical_contract(recipe: RecipeSpec, stage_id: str):
    if recipe.recipe_id not in _REFERENCE_RECIPE_IDS:
        return None
    try:
        canonical = _catalogue().recipe(recipe.recipe_id)
        canonical_target = canonical.stage(stage_id)
        current_target = recipe.stage(stage_id)
    except (KeyError, RecipeCatalogueError) as error:
        raise ReferenceMaskStageError("Recipe/stage ID is absent from the pinned PlantCV seed catalogue.") from error
    if current_target.operation_id != canonical_target.operation_id:
        raise ReferenceMaskStageError("Stage operation differs from the pinned catalogue stage.")

    if any(parameter.parameter_id == "roi" for parameter in current_target.parameters):
        return canonical, canonical_target, current_target, "ROI execution is outside the bounded original-call domain."

    if len(recipe.stages) != len(canonical.stages):
        raise ReferenceMaskStageError("Recipe stage topology differs from the pinned catalogue.")
    normalized_stages = []
    for current, pinned in zip(recipe.stages, canonical.stages, strict=True):
        if current.stage_id != pinned.stage_id:
            raise ReferenceMaskStageError("Recipe stage order or IDs differ from the pinned catalogue.")
        if len(current.parameters) != len(pinned.parameters):
            raise ReferenceMaskStageError("Recipe parameter declarations differ from the pinned catalogue.")
        normalized_parameters = []
        for actual_parameter, pinned_parameter in zip(current.parameters, pinned.parameters, strict=True):
            if actual_parameter.parameter_id != pinned_parameter.parameter_id:
                raise ReferenceMaskStageError("Recipe parameter IDs differ from the pinned catalogue.")
            normalized_parameters.append(replace(actual_parameter, value=pinned_parameter.value))
        normalized_stages.append(replace(current, parameters=tuple(normalized_parameters)))
    normalized = replace(recipe, stages=tuple(normalized_stages), layout=canonical.layout)
    if normalized != canonical:
        raise ReferenceMaskStageError(
            "Recipe topology, source mapping, ports, provenance, or parameter declarations differ from the pinned catalogue."
        )
    return canonical, canonical_target, current_target, None


def _selected_parameters(recipe: RecipeSpec, stage_id: str, operation: str) -> dict[str, Any]:
    parameters = recipe.parameters_for_runner(stage_id)
    expected_keys = {
        "filter.median_blur": {"ksize"},
        "morphology.fill": {"size"},
        "morphology.erode": {"ksize", "iterations"},
        "morphology.dilate": {"ksize", "iterations"},
    }[operation]
    if set(parameters) != expected_keys or any(value is None for value in parameters.values()):
        raise ValueError("Required stage parameters are unselected or outside the operation contract.")
    if operation == "filter.median_blur":
        ksize = parameters["ksize"]
        if type(ksize) is not int or not 1 <= ksize <= 255 or ksize % 2 != 1:
            raise ValueError("Median blur supports only scalar odd ksize values from 1 through 255.")
    elif operation in {"morphology.erode", "morphology.dilate"}:
        ksize, iterations = parameters["ksize"], parameters["iterations"]
        if type(ksize) is not int or not 2 <= ksize <= 255:
            raise ValueError("Morphology supports integer ksize values from 2 through 255.")
        if type(iterations) is not int or not 1 <= iterations <= 64:
            raise ValueError("Morphology supports integer iteration counts from 1 through 64.")
    else:
        size = parameters["size"]
        if type(size) is not int or size < 1:
            raise ValueError("PlantCV fill supports positive integer component-area thresholds.")
    try:
        encoded = json.dumps(
            parameters, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError, RecursionError) as error:
        raise ValueError("Selected parameters do not fit strict finite JSON.") from error
    if len(encoded) > MAX_PARAMETERS_BYTES:
        raise ValueError("Selected parameters exceed the pinned runner's protocol limit.")
    return parameters


def _median_reflect_domain_defined(shape: tuple[int, ...], ksize: int) -> bool:
    """Return whether SciPy 1.15.3 reflect indexing stays inside each axis.

    PlantCV 4.11.3 delegates its median filter to SciPy. On non-singleton axes,
    SciPy's reflect mapping can index one element before the logical view when
    the footprint radius reaches four full axis lengths. That makes output
    depend on excluded backing-storage bytes, so this bridge declines that
    source-undefined shape/kernel combination instead of recording it as exact.
    Singleton axes are clamped by SciPy and do not have this failure mode.
    """

    radius = ksize // 2
    return all(length == 1 or radius < 4 * length for length in shape)


def _validate_gray_identity(identity: CheckpointIdentity) -> None:
    semantics = identity.image_semantics
    try:
        dtype = np.dtype(semantics.get("dtype")).str
    except (TypeError, ValueError) as error:
        raise ReferenceMaskStageError("Median-blur input identity does not describe a gray image.") from error
    value_range = semantics.get("value_range")
    if (
        str(semantics.get("color_order", "")).upper() not in {"GRAY", "GRAYSCALE", "Y"}
        or dtype != np.dtype(np.uint8).str
        or not isinstance(value_range, (tuple, list))
        or tuple(value_range) != (0, 255)
    ):
        raise ReferenceMaskStageError("Median-blur input must be a uint8 grayscale image with range [0,255].")


def _validate_binary_identity(identity: CheckpointIdentity) -> None:
    semantics = identity.mask_semantics
    if semantics.get("representation") != "uint8_0_255":
        raise ReferenceMaskStageError("Morphology input identity must declare the uint8 0/255 mask representation.")


def expected_reference_mask_stage_identity(
    input_identity: CheckpointIdentity,
    recipe: RecipeSpec,
    stage_id: str,
    upstream_checkpoint_sha256: str,
) -> CheckpointIdentity:
    """Derive the exact output identity for one supported original call."""

    if not isinstance(input_identity, CheckpointIdentity) or not isinstance(recipe, RecipeSpec):
        raise TypeError("input_identity and recipe must be immutable checkpoint/recipe contracts.")
    upstream = _require_digest(upstream_checkpoint_sha256, "upstream_checkpoint_sha256")
    stage = recipe.stage(stage_id)
    if recipe.provenance.mode is not RecipeProvenanceMode.ORIGINAL_REFERENCE:
        raise ReferenceMaskStageError("Only original-reference PlantCV stages have output identities here.")
    operation = stage.operation_id
    if operation not in SUPPORTED_REFERENCE_MASK_OPERATIONS:
        raise ReferenceMaskStageError(f"PlantCV operation {operation!r} is outside this mask-stage bridge.")
    canonical = _canonical_contract(recipe, stage_id)
    if canonical is None or canonical[3] is not None:
        raise ReferenceMaskStageError("Recipe is not an exact pinned PlantCV seed graph.")
    try:
        parameters = _selected_parameters(recipe, stage_id, operation)
    except ValueError as error:
        raise ReferenceMaskStageError(str(error)) from error
    if operation == "filter.median_blur" and not _median_reflect_domain_defined(
        input_identity.coordinate_frame.shape, parameters["ksize"]
    ):
        raise ReferenceMaskStageError(
            "SciPy reflect indexing is source-undefined for this frame and median kernel; "
            "the kernel radius must be less than four times every non-singleton axis length."
        )
    provenance = recipe.provenance
    if (
        provenance.backend_id != "plantcv.reference"
        or provenance.backend_version != PINNED_PLANTCV_VERSION
        or provenance.reference_id != "plantcv"
        or provenance.reference_version != PINNED_PLANTCV_VERSION
        or provenance.reference_commit != PINNED_PLANTCV_COMMIT
        or provenance.environment_lock_sha256 != PINNED_REQUIREMENTS_SHA256
        or provenance.evidence_status is not SourceResolution.PINNED
    ):
        raise ReferenceMaskStageError("Recipe provenance differs from the exact PlantCV source and environment pins.")

    image_semantics = dict(input_identity.image_semantics)
    mask_semantics = dict(input_identity.mask_semantics)
    if operation == "filter.median_blur":
        _validate_gray_identity(input_identity)
    else:
        _validate_binary_identity(input_identity)
        mask_semantics["border_policy"] = f"plantcv_4.11.3_{operation}_default_without_roi"
    identity = recipe.identity
    return CheckpointIdentity(
        source_sha256=input_identity.source_sha256,
        coordinate_frame=input_identity.coordinate_frame,
        stage_id=stage_id,
        recipe_id=identity.recipe_id,
        recipe_version=identity.recipe_version,
        recipe_sha256=identity.recipe_sha256,
        graph_sha256=identity.graph_sha256,
        settings_sha256=identity.settings_sha256,
        backend_id=identity.backend_id,
        backend_version=identity.backend_version,
        environment_versions=dict(recipe.provenance.environment_versions),
        image_semantics=image_semantics,
        mask_semantics=mask_semantics,
        instance_semantics=dict(input_identity.instance_semantics),
        upstream_checkpoint_sha256=upstream,
    )


def _validate_selected_input(
    checkpoint,
    *,
    operation: str,
    input_bindings: Mapping[str, str],
    input_roles: Mapping[str, RasterRole],
    frame: CoordinateFrame,
) -> np.ndarray:
    port = _INPUT_PORT[operation]
    if set(input_bindings) != {port} or set(input_roles) != {port}:
        raise ReferenceMaskStageError(f"Stage requires exactly one explicit {port!r} input binding and role.")
    array_name = input_bindings[port]
    if not isinstance(array_name, str) or not array_name or array_name not in checkpoint.arrays:
        raise ReferenceMaskStageError("Explicit input binding does not name an array in the loaded checkpoint.")
    role = input_roles[port]
    expected_kind = "image" if operation == "filter.median_blur" else "mask"
    expected_role = RasterRole(
        expected_kind,
        np.dtype(np.uint8).str,
        "HW",
        (0, 255) if expected_kind == "image" else None,
    )
    if role != expected_role or checkpoint.raster_roles.get(array_name) != role:
        raise ReferenceMaskStageError("Bound raster role differs from the exact pinned operation input contract.")
    array = checkpoint.arrays[array_name]
    if (
        not isinstance(array, np.ndarray)
        or array.dtype != np.dtype(np.uint8)
        or array.ndim != 2
        or tuple(array.shape) != frame.shape
        or not array.flags.c_contiguous
        or array.nbytes > MAX_ARRAY_BYTES
        or array.size > MAX_ARRAY_ELEMENTS
    ):
        raise ReferenceMaskStageError("Bound input must be a bounded contiguous uint8 HxW raster in the requested coordinate frame.")
    if expected_kind == "image":
        _validate_gray_identity(checkpoint.identity)
    else:
        _validate_binary_identity(checkpoint.identity)
        if array.size and not np.isin(array, (0, 255)).all():
            raise ReferenceMaskStageError("Bound BinaryMask contains values outside uint8 0/255.")
        is_empty = not bool(np.any(array))
        if is_empty != (checkpoint.execution.status is ExecutionStatus.EMPTY):
            raise ReferenceMaskStageError("Input mask pixels disagree with the recorded empty/available execution state.")
    return array


def _validate_runner_result(
    result: Any,
    *,
    operation: str,
    parameters: Mapping[str, Any],
    recipe: RecipeSpec,
    frame: CoordinateFrame,
) -> tuple[np.ndarray, str]:
    if type(result) is not ReferenceResult:
        raise ReferenceMaskStageError("Runner did not return its checked ReferenceResult type.")
    if result.operation != operation:
        raise ReferenceMaskStageError("Runner returned a different operation than the selected recipe stage.")
    output_name = _OUTPUT_ARRAY[operation]
    if set(result.arrays) != {output_name}:
        raise ReferenceMaskStageError(f"Runner must return exactly one {output_name!r} raster.")
    array = result.arrays[output_name]
    if (
        not isinstance(array, np.ndarray)
        or array.dtype != np.dtype(np.uint8)
        or array.ndim != 2
        or tuple(array.shape) != frame.shape
        or not array.flags.c_contiguous
        or array.nbytes > MAX_ARRAY_BYTES
        or array.size > MAX_ARRAY_ELEMENTS
    ):
        raise ReferenceMaskStageError("Runner raster changed dtype, layout, bounds, or coordinate frame.")
    if output_name == "mask" and array.size and not np.isin(array, (0, 255)).all():
        raise ReferenceMaskStageError("Original morphology result is not uint8 0/255 BinaryMask data.")

    expected_effective = dict(parameters)
    expected_effective["callable"] = _CALLABLES[operation]
    actual_effective = result.effective_parameters
    if not isinstance(actual_effective, Mapping) or set(actual_effective) != set(expected_effective):
        raise ReferenceMaskStageError("Runner effective parameters differ from the exact selected settings.")
    for name, expected in expected_effective.items():
        actual = actual_effective[name]
        if type(actual) is not type(expected) or actual != expected:
            raise ReferenceMaskStageError(f"Runner effective parameter {name!r} differs from the recipe value.")
    if not isinstance(result.observations, Mapping):
        raise ReferenceMaskStageError("Runner observations are not a primitive JSON object.")
    if (
        not isinstance(result.intermediates, tuple)
        or not isinstance(result.visualizations, Mapping)
        or not isinstance(result.channel_semantics, Mapping)
    ):
        raise ReferenceMaskStageError("Runner diagnostics do not match the checked result contract.")
    for value in (result.wall_runtime_ns, result.computation_runtime_ns):
        if type(value) is not int or value < 0:
            raise ReferenceMaskStageError("Runner runtime must be a non-negative measured integer.")
    _validate_reference_identity(result, recipe)
    return array, output_name


def run_reference_mask_stage(
    input_root: Path | str,
    input_checkpoint_sha256: str,
    recipe: RecipeSpec,
    stage_id: str,
    runner: PlantCVReferenceRunner,
    output_root: Path | str,
    *,
    expected_source_sha256: str,
    expected_coordinate_frame: CoordinateFrame,
    expected_recipe_identity: RecipeIdentity,
    expected_upstream_checkpoint_sha256: str,
    expected_input_identity: CheckpointIdentity,
    expected_input_measurement: MeasurementContract,
    expected_output_identity: CheckpointIdentity | None,
    input_bindings: Mapping[str, str],
    input_roles: Mapping[str, RasterRole],
    output_measurement: MeasurementContract,
    allow_cpu_transfer: bool,
    limits: CheckpointLimits = CheckpointLimits(),
    timeout_seconds: float | None = None,
    cancel: Any = None,
) -> ReferenceMaskStageOutcome:
    """Run one exact pinned PlantCV public image/mask operation and checkpoint it.

    The input artifact is resolved here by digest and is never accepted as a
    caller-provided array. One public callable runs in the isolated reference
    interpreter; ROI and other unbound options are reported unsupported.
    """

    if not isinstance(recipe, RecipeSpec) or not isinstance(expected_recipe_identity, RecipeIdentity):
        raise TypeError("recipe and expected_recipe_identity must be immutable recipe contracts.")
    if type(runner) is not PlantCVReferenceRunner:
        raise TypeError("runner must be an explicitly supplied PlantCVReferenceRunner instance.")
    if not isinstance(expected_coordinate_frame, CoordinateFrame):
        raise TypeError("expected_coordinate_frame must be a CoordinateFrame.")
    if not isinstance(expected_input_identity, CheckpointIdentity) or not isinstance(expected_input_measurement, MeasurementContract):
        raise TypeError("Expected input identity and measurement contracts are required.")
    if expected_output_identity is not None and not isinstance(expected_output_identity, CheckpointIdentity):
        raise TypeError("expected_output_identity must be a CheckpointIdentity or None.")
    if not isinstance(output_measurement, MeasurementContract):
        raise TypeError("output_measurement must be an explicit MeasurementContract.")
    if output_measurement.eligible:
        raise ReferenceMaskStageError("A PlantCV image/mask operation cannot mark science measurements eligible.")
    if type(allow_cpu_transfer) is not bool or not allow_cpu_transfer:
        raise ReferenceMaskStageError("Reference output serialization requires explicit allow_cpu_transfer=True.")
    if expected_recipe_identity != recipe.identity:
        raise ReferenceMaskStageError("Recipe identity differs from the caller's exact expected recipe identity.")
    source_digest = _require_digest(expected_source_sha256, "expected_source_sha256")
    input_digest = _require_digest(input_checkpoint_sha256, "input_checkpoint_sha256")
    upstream_digest = _require_digest(expected_upstream_checkpoint_sha256, "expected_upstream_checkpoint_sha256")
    if upstream_digest != input_digest:
        raise ReferenceMaskStageError("Expected upstream digest differs from the exact input checkpoint address.")
    if expected_input_identity.source_sha256 != source_digest:
        raise ReferenceMaskStageError("Expected input checkpoint source SHA differs from the requested source.")
    if expected_input_identity.coordinate_frame != expected_coordinate_frame:
        raise ReferenceMaskStageError("Expected input checkpoint frame differs from the requested coordinate frame.")
    expected_elements = math.prod(expected_coordinate_frame.shape)
    if expected_elements > MAX_ARRAY_ELEMENTS or expected_elements * np.dtype(np.uint8).itemsize > MAX_ARRAY_BYTES:
        raise ReferenceMaskStageError("Requested frame exceeds the original runner's bounded uint8 raster domain.")
    if isinstance(timeout_seconds, bool) or (timeout_seconds is not None and type(timeout_seconds) not in (int, float)):
        raise ReferenceMaskStageError("timeout_seconds must be finite and in (0,3600].")
    resolved_timeout = runner.timeout_seconds if timeout_seconds is None else timeout_seconds
    try:
        resolved_timeout = float(resolved_timeout)
    except (OverflowError, ValueError):
        raise ReferenceMaskStageError("timeout_seconds must be finite and in (0,3600].") from None
    if not math.isfinite(resolved_timeout) or not 0 < resolved_timeout <= MAX_REFERENCE_STAGE_TIMEOUT_SECONDS:
        raise ReferenceMaskStageError("timeout_seconds must be finite and in (0,3600].")
    if not isinstance(input_bindings, Mapping) or not isinstance(input_roles, Mapping):
        raise ReferenceMaskStageError("Input bindings and raster roles must be explicit mappings.")
    if any(not isinstance(key, str) or not isinstance(value, str) for key, value in input_bindings.items()):
        raise ReferenceMaskStageError("Input bindings must map recipe input ports to checkpoint array names.")
    if any(not isinstance(key, str) or not isinstance(value, RasterRole) for key, value in input_roles.items()):
        raise ReferenceMaskStageError("Input roles must map recipe input ports to RasterRole contracts.")
    try:
        stage = recipe.stage(stage_id)
    except KeyError as error:
        raise ReferenceMaskStageError(f"Recipe does not contain stage {stage_id!r}.") from error

    if _cancelled(cancel):
        return _no_artifact(recipe, stage_id, input_digest, ReferenceMaskStageStatus.CANCELLED, "Cancelled before stage validation.")
    if recipe.provenance.mode is not RecipeProvenanceMode.ORIGINAL_REFERENCE:
        return _no_artifact(recipe, stage_id, input_digest, ReferenceMaskStageStatus.UNSUPPORTED, "Only pinned original-reference PlantCV stages can execute here.")
    if recipe.provenance.evidence_status is SourceResolution.UNAVAILABLE or stage.resolution is SourceResolution.UNAVAILABLE:
        return _no_artifact(recipe, stage_id, input_digest, ReferenceMaskStageStatus.UNAVAILABLE, recipe.provenance.evidence_note or stage.resolution_note)
    if recipe.provenance.evidence_status is SourceResolution.UNRESOLVED or stage.resolution is SourceResolution.UNRESOLVED:
        return _no_artifact(recipe, stage_id, input_digest, ReferenceMaskStageStatus.UNRESOLVED, recipe.provenance.evidence_note or stage.resolution_note)

    operation = stage.operation_id
    if operation not in SUPPORTED_REFERENCE_MASK_OPERATIONS:
        return _no_artifact(
            recipe,
            stage_id,
            input_digest,
            ReferenceMaskStageStatus.UNSUPPORTED,
            f"Stage {stage_id!r} is outside the one-public-call image/mask bridge; watershed internals are not replayed separately.",
        )
    canonical = _canonical_contract(recipe, stage_id)
    if canonical is None:
        return _no_artifact(recipe, stage_id, input_digest, ReferenceMaskStageStatus.UNSUPPORTED, "Only the three pinned PlantCV seed recipe graphs are in this bridge scope.")
    if canonical[3] is not None:
        return _no_artifact(recipe, stage_id, input_digest, ReferenceMaskStageStatus.UNSUPPORTED, canonical[3])
    try:
        parameters = _selected_parameters(recipe, stage_id, operation)
    except ValueError as error:
        return _no_artifact(recipe, stage_id, input_digest, ReferenceMaskStageStatus.UNSUPPORTED, str(error))
    if operation == "filter.median_blur" and not _median_reflect_domain_defined(
        expected_coordinate_frame.shape, parameters["ksize"]
    ):
        return _no_artifact(
            recipe,
            stage_id,
            input_digest,
            ReferenceMaskStageStatus.UNSUPPORTED,
            "SciPy reflect indexing is source-undefined for this frame and median kernel; "
            "the kernel radius must be less than four times every non-singleton axis length.",
        )
    if recipe.provenance != canonical[0].provenance or stage.resolution is not SourceResolution.PINNED:
        raise ReferenceMaskStageError("Recipe backend, source audit, package pins, or stage resolution differs from the pinned catalogue.")

    expected_port = _INPUT_PORT[operation]
    required_inputs = {
        port.name for port in stage.ports
        if port.direction == "input" and port.required
    }
    if required_inputs != {expected_port}:
        raise ReferenceMaskStageError("Pinned stage input topology differs from the one-raster operation contract.")
    try:
        checkpoint = load_stage_input(
            input_root,
            input_digest,
            expected_identity=expected_input_identity,
            expected_measurement=expected_input_measurement,
            limits=limits,
        )
    except Exception as error:
        raise ReferenceMaskStageError(f"Exact stage input checkpoint could not be loaded: {error}") from error
    if checkpoint.identity.source_sha256 != source_digest:
        raise ReferenceMaskStageError("Loaded input checkpoint source SHA differs from the requested source.")
    if checkpoint.identity.coordinate_frame != expected_coordinate_frame:
        raise ReferenceMaskStageError("Loaded input checkpoint frame differs from the requested frame.")
    input_array = _validate_selected_input(
        checkpoint,
        operation=operation,
        input_bindings=input_bindings,
        input_roles=input_roles,
        frame=expected_coordinate_frame,
    )
    derived_identity = expected_reference_mask_stage_identity(
        checkpoint.identity, recipe, stage_id, input_digest
    )
    if expected_output_identity != derived_identity:
        raise ReferenceMaskStageError("Expected output identity differs from the exact recipe/source/frame/upstream contract.")
    try:
        validate_checkpoint_identity(
            recipe,
            stage_id,
            derived_identity,
            expected_identity=expected_output_identity,
            expected_upstream_checkpoint_sha256=input_digest,
        )
    except (RecipeContractError, StaleRecipeIdentity) as error:
        raise ReferenceMaskStageError(f"Derived output identity is not valid for this recipe stage: {error}") from error
    if _cancelled(cancel):
        return _no_artifact(recipe, stage_id, input_digest, ReferenceMaskStageStatus.CANCELLED, "Cancelled before PlantCV execution.")

    availability = plantcv_reference_status(
        python_executable=runner.python_executable,
        source_manifest_path=runner.source_manifest_path,
        source_root=runner.source_root,
        environment_manifest_path=runner.environment_manifest_path,
        requirements_lock_path=runner.requirements_lock_path,
    )
    if availability.get("available") is not True:
        return _no_artifact(
            recipe, stage_id, input_digest, ReferenceMaskStageStatus.UNAVAILABLE,
            str(availability.get("reason", "Pinned PlantCV source or environment is unavailable.")),
        )
    if _cancelled(cancel):
        return _no_artifact(recipe, stage_id, input_digest, ReferenceMaskStageStatus.CANCELLED, "Cancelled before PlantCV execution.")

    stage_started = time.perf_counter_ns()
    try:
        result = runner.run(
            operation,
            {expected_port: input_array},
            parameters,
            timeout_seconds=resolved_timeout,
            cancel_event=cancel,
        )
    except ReferenceCancelledError as error:
        return _no_artifact(recipe, stage_id, input_digest, ReferenceMaskStageStatus.CANCELLED, f"Cancelled during PlantCV execution: {error}")
    except ReferenceIdentityError as error:
        return _no_artifact(recipe, stage_id, input_digest, ReferenceMaskStageStatus.UNAVAILABLE, f"Pinned PlantCV source or environment became unavailable: {error}")
    except (ReferenceTimeoutError, ReferenceExecutionError) as error:
        status = ReferenceMaskStageStatus.CANCELLED if _cancelled(cancel) else ReferenceMaskStageStatus.FAILED
        return _no_artifact(recipe, stage_id, input_digest, status, f"PlantCV execution failed: {type(error).__name__}: {error}")
    except Exception as error:
        status = ReferenceMaskStageStatus.CANCELLED if _cancelled(cancel) else ReferenceMaskStageStatus.FAILED
        return _no_artifact(recipe, stage_id, input_digest, status, f"PlantCV execution failed: {type(error).__name__}: {error}")

    try:
        output, output_name = _validate_runner_result(
            result,
            operation=operation,
            parameters=parameters,
            recipe=recipe,
            frame=expected_coordinate_frame,
        )
    except (ReferenceMaskStageError, RecipeContractError, StaleRecipeIdentity, TypeError, ValueError) as error:
        return _no_artifact(recipe, stage_id, input_digest, ReferenceMaskStageStatus.FAILED, f"PlantCV result failed stage validation: {error}")
    if _cancelled(cancel):
        return _no_artifact(recipe, stage_id, input_digest, ReferenceMaskStageStatus.CANCELLED, "Cancelled before checkpoint publication.")

    output_role = (
        RasterRole("image", np.dtype(np.uint8).str, "HW", (0, 255))
        if output_name == "image"
        else RasterRole("mask", np.dtype(np.uint8).str, "HW")
    )
    if output_name == "mask":
        execution_status = ExecutionStatus.EMPTY if not np.any(output) else ExecutionStatus.AVAILABLE
        stage_status = ReferenceMaskStageStatus.EMPTY if execution_status is ExecutionStatus.EMPTY else ReferenceMaskStageStatus.EXECUTED
        run_state = StageRunState.EMPTY if execution_status is ExecutionStatus.EMPTY else StageRunState.EXECUTED
    else:
        execution_status = ExecutionStatus.AVAILABLE
        stage_status = ReferenceMaskStageStatus.EXECUTED
        run_state = StageRunState.EXECUTED
    execution = ExecutionRecord(
        status=execution_status,
        elapsed_ms=result.wall_runtime_ns / 1_000_000,
        transfer_ms=None,
    )
    source_mapping = stage.source_mappings[0]
    metadata = _strict_metadata({
        "format": REFERENCE_MASK_STAGE_FORMAT,
        "version": REFERENCE_MASK_STAGE_VERSION,
        "stage_id": stage_id,
        "operation_id": operation,
        "recipe_identity": recipe.identity.to_dict(),
        "source_mapping": source_mapping.to_dict(),
        "source_sha256": source_digest,
        "coordinate_frame": expected_coordinate_frame.to_dict(),
        "upstream_checkpoint_sha256": input_digest,
        "input_checkpoint_identity": checkpoint.identity.to_dict(),
        "input_bindings": dict(input_bindings),
        "input_raster_roles": {name: role.to_dict() for name, role in input_roles.items()},
        "output_identity": derived_identity.to_dict(),
        "output_raster_roles": {output_name: output_role.to_dict()},
        "runner_identity": dict(result.identity),
        "parameters": parameters,
        "effective_parameters": dict(result.effective_parameters),
        "observations": dict(result.observations),
        "intermediates": list(result.intermediates),
        "visualizations": dict(result.visualizations),
        "channel_semantics": dict(result.channel_semantics),
        "computation_runtime_ns": result.computation_runtime_ns,
        "wall_runtime_ns": result.wall_runtime_ns,
        "stage_runtime_ns": max(0, time.perf_counter_ns() - stage_started),
        "output_inventory": [{
            "name": output_name,
            "dtype": output.dtype.str,
            "shape": list(output.shape),
            "nbytes": int(output.nbytes),
        }],
        "measurement_contract": output_measurement.to_dict(),
        "scientific_validation_status": "not_certified",
    })
    try:
        reference = publish_checkpoint(
            output_root,
            derived_identity,
            {output_name: output},
            execution=execution,
            measurement=output_measurement,
            raster_roles={output_name: output_role},
            metadata=metadata,
            allow_cpu_transfer=True,
            limits=limits,
            cancel=cancel,
        )
    except CheckpointCancelled as error:
        return _no_artifact(recipe, stage_id, input_digest, ReferenceMaskStageStatus.CANCELLED, f"Cancelled during checkpoint publication: {error}")
    except Exception as error:
        return _no_artifact(recipe, stage_id, input_digest, ReferenceMaskStageStatus.FAILED, f"Checkpoint publication failed: {type(error).__name__}: {error}")

    run_record = StageRunRecord(
        stage_id=stage_id,
        recipe_identity=recipe.identity,
        state=run_state,
        checkpoint_sha256=reference.checkpoint_sha256,
        upstream_checkpoint_sha256=input_digest,
    )
    validate_stage_run(recipe, run_record, expected_upstream_checkpoint_sha256=input_digest)
    validate_checkpoint_identity(
        recipe,
        stage_id,
        reference.identity,
        expected_identity=derived_identity,
        expected_upstream_checkpoint_sha256=input_digest,
    )
    return ReferenceMaskStageOutcome(stage_status, run_record, reference)


__all__ = [
    "REFERENCE_MASK_STAGE_FORMAT",
    "REFERENCE_MASK_STAGE_VERSION",
    "ReferenceMaskStageError",
    "ReferenceMaskStageOutcome",
    "ReferenceMaskStageStatus",
    "SUPPORTED_REFERENCE_MASK_OPERATIONS",
    "expected_reference_mask_stage_identity",
    "run_reference_mask_stage",
]
