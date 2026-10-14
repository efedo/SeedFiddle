"""Strict one-stage bridge from a pinned PlantCV recipe to checkpoints.

This module binds one public threshold call at a time. It delegates execution to
the isolated :class:`PlantCVReferenceRunner`; it does not interpret or execute a
recipe graph and does not make a scientific-validity claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
import math
from pathlib import Path
import re
import time
from typing import Any, Mapping

import numpy as np

from seedvision.benchmarking.checkpoints import (
    CheckpointIdentity,
    CheckpointLimits,
    CheckpointReference,
    CheckpointCancelled,
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
    RecipeIdentity,
    RecipeProvenanceMode,
    RecipeSpec,
    SourceResolution,
    StageRunRecord,
    StageRunState,
    validate_checkpoint_identity,
    validate_stage_run,
)
from seedvision.benchmarking.reference_runner import (
    MAX_ARRAY_BYTES,
    MAX_ARRAY_ELEMENTS,
    MAX_PARAMETERS_BYTES,
    PINNED_PLANTCV_COMMIT,
    PINNED_PLANTCV_VERSION,
    PINNED_REFERENCE_PLATFORM,
    PINNED_REFERENCE_PYTHON,
    PINNED_REQUIREMENTS_SHA256,
    PINNED_SOURCE_AUDIT_SHA256,
    PlantCVReferenceRunner,
    ReferenceIdentityError,
    ReferenceResult,
    SUPPORTED_PLANTCV_OPERATIONS,
    plantcv_reference_status,
)


REFERENCE_STAGE_FORMAT = "seedfiddle.reference-stage-output"
REFERENCE_STAGE_VERSION = 1
MAX_REFERENCE_STAGE_METADATA_BYTES = 1024 * 1024
MAX_REFERENCE_STAGE_TIMEOUT_SECONDS = 3600.0
SUPPORTED_REFERENCE_STAGE_OPERATIONS = frozenset({
    "threshold.binary",
    "threshold.mean",
    "threshold.otsu",
})

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_NORMALIZE_DISTRIBUTION = re.compile(r"[-_.]+")
_PARAMETER_KEYS = {
    "threshold.binary": frozenset({"threshold", "object_type"}),
    "threshold.mean": frozenset({"ksize", "offset", "object_type"}),
    "threshold.otsu": frozenset({"object_type"}),
}
_CALLABLES = {
    "threshold.binary": "plantcv.plantcv.threshold.binary",
    "threshold.mean": "plantcv.plantcv.threshold.mean",
    "threshold.otsu": "plantcv.plantcv.threshold.otsu",
}
_CANONICAL_RECIPE_IDS = {
    "threshold.binary": "plantcv_seed_binary",
    "threshold.mean": "plantcv_seed_mean",
    "threshold.otsu": "plantcv_seed_otsu",
}


class ReferenceStageError(ValueError):
    """A recipe stage or checkpoint does not satisfy this bridge's contract."""


@dataclass(frozen=True, slots=True)
class ReferenceStageOutcome:
    """Run state plus the immutable artifact reference, when one was published."""

    run_record: StageRunRecord
    checkpoint_reference: CheckpointReference | None

    def __post_init__(self) -> None:
        if not isinstance(self.run_record, StageRunRecord):
            raise TypeError("run_record must be a StageRunRecord.")
        completed = self.run_record.state in {StageRunState.EXECUTED, StageRunState.EMPTY}
        if completed != (self.checkpoint_reference is not None):
            raise ValueError("Only executed or empty stage runs carry a checkpoint reference.")
        if self.checkpoint_reference is not None and (
            self.run_record.checkpoint_sha256 != self.checkpoint_reference.checkpoint_sha256
        ):
            raise ValueError("Stage run and checkpoint reference hashes differ.")


@lru_cache(maxsize=1)
def _pinned_catalogue():
    try:
        return build_recipe_catalogue()
    except (RecipeCatalogueError, OSError, ValueError) as error:
        raise ReferenceStageError(f"Committed PlantCV recipe pins are unavailable: {error}") from error


def _canonical_stage(operation_id: str):
    try:
        recipe_id = _CANONICAL_RECIPE_IDS[operation_id]
    except KeyError as error:
        raise ReferenceStageError(f"PlantCV operation {operation_id!r} has no stage bridge contract.") from error
    recipe = _pinned_catalogue().recipe(recipe_id)
    matches = tuple(stage for stage in recipe.stages if stage.operation_id == operation_id)
    if len(matches) != 1:
        raise ReferenceStageError("Committed PlantCV catalogue does not contain one exact threshold stage.")
    return recipe, matches[0]


def _source_fingerprint(mapping: Any) -> dict[str, Any]:
    data = mapping.to_dict()
    data.pop("note", None)
    return data


def _port_fingerprint(port: Any) -> tuple[Any, ...]:
    return (
        port.name,
        port.direction,
        port.data_type,
        port.required,
        port.external,
        dict(port.semantics),
    )


def _normalize_versions(values: Mapping[str, str]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for name, version in values.items():
        key = _NORMALIZE_DISTRIBUTION.sub("-", name).lower()
        if key in normalized:
            raise ReferenceStageError("Pinned recipe environment contains duplicate normalized package names.")
        normalized[key] = version
    return normalized


def _calculation_parameters(recipe: RecipeSpec, stage_id: str, operation_id: str) -> dict[str, Any]:
    parameters = recipe.parameters_for_runner(stage_id)
    expected_keys = _PARAMETER_KEYS[operation_id]
    if set(parameters) != expected_keys or any(value is None for value in parameters.values()):
        raise ReferenceStageError(
            f"Stage {stage_id!r} has unselected or unsupported calculation parameters for {operation_id}."
        )
    object_type = parameters.get("object_type")
    if not isinstance(object_type, str) or object_type not in {"light", "dark"}:
        raise ReferenceStageError("Threshold stage object_type must be selected as 'light' or 'dark'.")
    if operation_id == "threshold.binary":
        threshold = parameters["threshold"]
        if type(threshold) is not int or not 0 <= threshold <= 255:
            raise ReferenceStageError("Binary threshold must be an integer in [0,255].")
    elif operation_id == "threshold.mean":
        ksize = parameters["ksize"]
        offset = parameters["offset"]
        if type(ksize) is not int or not 3 <= ksize <= 255:
            raise ReferenceStageError("Mean-threshold requested ksize must be an integer in [3,255]; even sizes round up to odd.")
        if isinstance(offset, bool) or type(offset) not in (int, float):
            raise ReferenceStageError("Mean-threshold offset must be a finite number.")
        try:
            runner_offset = float(offset)
        except (OverflowError, ValueError):
            raise ReferenceStageError("Mean-threshold offset must be a finite, exactly representable number.") from None
        if not math.isfinite(runner_offset):
            raise ReferenceStageError("Mean-threshold offset must be a finite number.")
        # The isolated PlantCV worker consumes offsets as binary64. Reject
        # integers that would be rounded while crossing that JSON/float
        # boundary, rather than claiming the selected recipe value was run.
        if type(offset) is int and int(runner_offset) != offset:
            raise ReferenceStageError("Mean-threshold integer offset is not exactly representable by the runner.")
    try:
        encoded = json.dumps(parameters, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as error:
        raise ReferenceStageError("Threshold parameters are not strict JSON primitives.") from error
    if len(encoded) > MAX_PARAMETERS_BYTES:
        raise ReferenceStageError("Threshold parameters exceed the runner's bounded parameter limit.")
    return parameters


def _validate_pinned_stage(recipe: RecipeSpec, stage: Any) -> dict[str, Any]:
    operation_id = stage.operation_id
    canonical_recipe, canonical_stage = _canonical_stage(operation_id)
    if recipe.provenance != canonical_recipe.provenance:
        raise ReferenceStageError("Recipe provenance differs from the committed PlantCV source/environment pins.")
    provenance = recipe.provenance
    if (
        provenance.backend_id != "plantcv.reference"
        or provenance.backend_version != PINNED_PLANTCV_VERSION
        or provenance.reference_id != "plantcv"
        or provenance.reference_version != PINNED_PLANTCV_VERSION
        or provenance.reference_commit != PINNED_PLANTCV_COMMIT
        or provenance.environment_lock_sha256 != PINNED_REQUIREMENTS_SHA256
        or _normalize_versions(provenance.environment_versions).get("plantcv") != PINNED_PLANTCV_VERSION
        or provenance.evidence_status is not SourceResolution.PINNED
    ):
        raise ReferenceStageError("Recipe backend, PlantCV source, or dependency lock differs from the runner pins.")
    if stage.resolution is not SourceResolution.PINNED:
        raise ReferenceStageError("A runnable reference stage must have pinned source resolution.")
    if len(stage.source_mappings) != 1 or _source_fingerprint(stage.source_mappings[0]) != _source_fingerprint(canonical_stage.source_mappings[0]):
        raise ReferenceStageError("Stage source mapping differs from the exact public PlantCV callable pin.")
    if tuple(map(_port_fingerprint, stage.ports)) != tuple(map(_port_fingerprint, canonical_stage.ports)):
        raise ReferenceStageError("Stage ports or image/mask semantics differ from the pinned public operation contract.")
    return _calculation_parameters(recipe, stage.stage_id, operation_id)


def _mask_semantics(operation_id: str, parameters: Mapping[str, Any]) -> dict[str, Any]:
    """The threshold operation's explicit whole-frame output representation."""
    return {
        "foreground_polarity": parameters["object_type"],
        "representation": "uint8_0_255",
        "connectivity": "not_applicable_to_threshold",
        "validity_region": "full_input_frame",
        "border_policy": f"pinned_{operation_id}_operation",
    }


def expected_reference_stage_identity(
    input_identity: CheckpointIdentity,
    recipe: RecipeSpec,
    stage_id: str,
    upstream_checkpoint_sha256: str,
) -> CheckpointIdentity:
    """Build the exact output identity for one supported pinned threshold call."""
    if not isinstance(input_identity, CheckpointIdentity) or not isinstance(recipe, RecipeSpec):
        raise TypeError("input_identity and recipe must be their immutable contract types.")
    if not isinstance(upstream_checkpoint_sha256, str) or not _SHA256.fullmatch(upstream_checkpoint_sha256):
        raise ReferenceStageError("upstream_checkpoint_sha256 must be a lowercase SHA-256 digest.")
    try:
        stage = recipe.stage(stage_id)
    except KeyError as error:
        raise ReferenceStageError(f"Recipe does not contain stage {stage_id!r}.") from error
    if recipe.provenance.mode is not RecipeProvenanceMode.ORIGINAL_REFERENCE:
        raise ReferenceStageError("Only original-reference PlantCV stages have a checkpoint identity here.")
    if stage.operation_id not in SUPPORTED_REFERENCE_STAGE_OPERATIONS:
        raise ReferenceStageError(f"PlantCV operation {stage.operation_id!r} is not bound by this stage bridge.")
    parameters = _validate_pinned_stage(recipe, stage)
    identity = recipe.identity
    image_semantics = dict(input_identity.image_semantics)
    try:
        input_dtype = np.dtype(image_semantics.get("dtype")).str
    except (TypeError, ValueError) as error:
        raise ReferenceStageError("Threshold input identity must bind a uint8 gray image.") from error
    value_range = image_semantics.get("value_range")
    if (
        str(image_semantics.get("color_order", "")).upper() not in {"GRAY", "GRAYSCALE", "Y"}
        or input_dtype != np.dtype(np.uint8).str
        or not isinstance(value_range, (tuple, list))
        or tuple(value_range) != (0, 255)
    ):
        raise ReferenceStageError("Threshold input identity must bind a uint8 gray image with range [0,255].")
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
        mask_semantics=_mask_semantics(stage.operation_id, parameters),
        instance_semantics=dict(input_identity.instance_semantics),
        upstream_checkpoint_sha256=upstream_checkpoint_sha256,
    )


def _strict_metadata(payload: Mapping[str, Any]) -> dict[str, Any]:
    encoder = json.JSONEncoder(
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    encoded = bytearray()
    try:
        for part in encoder.iterencode(payload):
            if len(part) > MAX_REFERENCE_STAGE_METADATA_BYTES:
                raise ReferenceStageError("Reference stage metadata exceeds its byte limit.")
            chunk = part.encode("utf-8")
            if len(encoded) + len(chunk) > MAX_REFERENCE_STAGE_METADATA_BYTES:
                raise ReferenceStageError("Reference stage metadata exceeds its byte limit.")
            encoded.extend(chunk)
    except ReferenceStageError:
        raise
    except (TypeError, ValueError, UnicodeError, RecursionError) as error:
        raise ReferenceStageError("Reference stage output metadata must be strict finite JSON.") from error
    return json.loads(bytes(encoded).decode("utf-8"))


def _cancelled(cancel: Any) -> bool:
    if cancel is None:
        return False
    if callable(cancel):
        return bool(cancel())
    is_set = getattr(cancel, "is_set", None)
    if not callable(is_set):
        raise ReferenceStageError("cancel must be a callable or an event with is_set().")
    return bool(is_set())


def _failed_outcome(
    recipe: RecipeSpec,
    stage_id: str,
    upstream_checkpoint_sha256: str,
    reason: str,
    *,
    state: StageRunState = StageRunState.FAILED,
) -> ReferenceStageOutcome:
    compact_reason = " ".join(str(reason).split())[:512] or state.value
    record = StageRunRecord(
        stage_id=stage_id,
        recipe_identity=recipe.identity,
        state=state,
        upstream_checkpoint_sha256=upstream_checkpoint_sha256,
        reason=compact_reason,
    )
    validate_stage_run(recipe, record, expected_upstream_checkpoint_sha256=upstream_checkpoint_sha256)
    return ReferenceStageOutcome(record, None)


def _validate_reference_identity(result: ReferenceResult, recipe: RecipeSpec) -> None:
    identity = result.identity
    if not isinstance(identity, Mapping):
        raise ReferenceStageError("Runner omitted its verified reference identity.")
    expected_fields = {
        "reference": "plantcv",
        "version": PINNED_PLANTCV_VERSION,
        "commit": PINNED_PLANTCV_COMMIT,
        "python": PINNED_REFERENCE_PYTHON,
        "platform": PINNED_REFERENCE_PLATFORM,
        "source_audit_sha256": PINNED_SOURCE_AUDIT_SHA256,
        "requirements_lock_sha256": PINNED_REQUIREMENTS_SHA256,
    }
    for field, expected in expected_fields.items():
        if identity.get(field) != expected:
            raise ReferenceStageError(f"Runner reference identity field {field!r} differs from its pin.")
    identity_digest = identity.get("identity_digest")
    if not isinstance(identity_digest, str) or not _SHA256.fullmatch(identity_digest):
        raise ReferenceStageError("Runner reference identity digest is invalid.")
    expected_distributions = _normalize_versions(recipe.provenance.environment_versions)
    if identity.get("actual_distributions") != expected_distributions:
        raise ReferenceStageError("Runner installed dependency inventory differs from the pinned recipe.")


def validate_otsu_cutoff_adapter(value, object_type):
    """Validate explicitly adapter-derived scalar evidence, never a public return."""
    if not isinstance(value, Mapping):
        raise ReferenceStageError("Pinned Otsu scalar adapter metadata must be a mapping.")
    cutoff = value.get("reference_selected_cutoff")
    if type(cutoff) is not int or not 0 <= cutoff <= 255:
        raise ReferenceStageError("Pinned Otsu scalar adapter cutoff must be an integer in [0,255].")
    expected = {
        "reference_selected_cutoff": cutoff,
        "origin": "pinned_cv2.threshold_adapter",
        "public_return": "mask_only",
        "object_type": object_type,
        "method": "THRESH_OTSU|THRESH_BINARY" if object_type == "light" else "THRESH_OTSU|THRESH_BINARY_INV",
        "public_mask_match": True,
    }
    if object_type not in ("light", "dark") or set(value) != set(expected) or any(
            type(value[name]) is not type(actual) or value[name] != actual
            for name, actual in expected.items()):
        raise ReferenceStageError("Pinned Otsu scalar adapter metadata differs from its exact call.")
    return expected


def _validate_result(
    result: Any,
    *,
    recipe: RecipeSpec,
    stage_id: str,
    operation_id: str,
    parameters: Mapping[str, Any],
    frame: CoordinateFrame,
) -> ReferenceResult:
    if type(result) is not ReferenceResult:
        raise ReferenceStageError("Runner did not return its checked ReferenceResult type.")
    if result.operation != operation_id:
        raise ReferenceStageError("Runner result operation differs from the requested recipe stage.")
    if set(result.arrays) != {"mask"}:
        raise ReferenceStageError("Threshold stage must return exactly one output array named 'mask'.")
    mask = result.arrays["mask"]
    if not isinstance(mask, np.ndarray) or mask.dtype != np.dtype(np.uint8) or tuple(mask.shape) != frame.shape:
        raise ReferenceStageError("Threshold output mask changed dtype or coordinate frame.")
    if mask.size and not np.isin(mask, (0, 255)).all():
        raise ReferenceStageError("Threshold output mask is not the exact uint8 0/255 representation.")
    expected_effective = dict(parameters)
    if operation_id == "threshold.mean":
        expected_effective["ksize"] += int(parameters["ksize"] % 2 == 0)
    expected_effective["callable"] = _CALLABLES[operation_id]
    actual_effective = result.effective_parameters
    if not isinstance(actual_effective, Mapping) or set(actual_effective) != set(expected_effective):
        raise ReferenceStageError("Runner effective parameters do not match the exact recipe parameter set.")
    for key, expected in expected_effective.items():
        actual = actual_effective[key]
        if key == "offset":
            if isinstance(actual, bool) or type(actual) not in (int, float):
                raise ReferenceStageError("Runner's effective offset differs from the recipe setting.")
            if type(actual) is float and not math.isfinite(actual):
                raise ReferenceStageError("Runner's effective offset differs from the recipe setting.")
            # Python's mixed int/float equality compares the exact represented
            # values (unlike coercing both sides to float), so an altered
            # binary64 result cannot impersonate a larger selected integer.
            if actual != expected:
                raise ReferenceStageError("Runner's effective offset differs from the recipe setting.")
        elif type(actual) is not type(expected) or actual != expected:
            raise ReferenceStageError(f"Runner effective parameter {key!r} differs from the recipe setting.")
    if not isinstance(result.observations, Mapping) or not isinstance(result.channel_semantics, Mapping):
        raise ReferenceStageError("Runner observations or channel semantics are malformed.")
    if operation_id == "threshold.otsu":
        if set(result.observations) != {"otsu_cutoff_adapter"}:
            raise ReferenceStageError("Pinned Otsu result is missing its separate scalar adapter record.")
        validate_otsu_cutoff_adapter(result.observations["otsu_cutoff_adapter"], parameters["object_type"])
    if not isinstance(result.visualizations, Mapping) or not isinstance(result.intermediates, tuple):
        raise ReferenceStageError("Runner diagnostic products are malformed.")
    for field in ("wall_runtime_ns", "computation_runtime_ns"):
        value = getattr(result, field)
        if type(value) is not int or value < 0:
            raise ReferenceStageError(f"Runner {field} must be a non-negative integer.")
    _validate_reference_identity(result, recipe)
    return result


def run_reference_stage(
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
) -> ReferenceStageOutcome:
    """Execute one exact-input public threshold stage and publish its output.

    The content-addressed input is loaded and validated within this function.
    The only path that produces execution evidence invokes the supplied
    ``PlantCVReferenceRunner`` itself; callers cannot pass a prebuilt result.
    """

    if not isinstance(recipe, RecipeSpec) or not isinstance(expected_recipe_identity, RecipeIdentity):
        raise TypeError("recipe and expected_recipe_identity must be immutable recipe contracts.")
    if not isinstance(runner, PlantCVReferenceRunner) or type(runner) is not PlantCVReferenceRunner:
        raise TypeError("runner must be an explicitly supplied PlantCVReferenceRunner instance.")
    if not isinstance(expected_coordinate_frame, CoordinateFrame):
        raise TypeError("expected_coordinate_frame must be a CoordinateFrame.")
    if not isinstance(expected_input_identity, CheckpointIdentity) or not isinstance(expected_input_measurement, MeasurementContract):
        raise TypeError("Expected input identity and measurement contracts are required.")
    if not isinstance(output_measurement, MeasurementContract):
        raise TypeError("output_measurement must be an explicit MeasurementContract.")
    if output_measurement.eligible:
        raise ReferenceStageError("A threshold output cannot mark scientific measurement eligibility.")
    if type(allow_cpu_transfer) is not bool or not allow_cpu_transfer:
        raise ReferenceStageError("Reference output serialization requires allow_cpu_transfer=True.")
    if expected_recipe_identity != recipe.identity:
        raise ReferenceStageError("Recipe identity differs from the caller's exact expected recipe identity.")
    if not isinstance(expected_source_sha256, str) or not _SHA256.fullmatch(expected_source_sha256):
        raise ReferenceStageError("expected_source_sha256 must be a lowercase SHA-256 digest.")
    if not isinstance(input_checkpoint_sha256, str) or not _SHA256.fullmatch(input_checkpoint_sha256):
        raise ReferenceStageError("input_checkpoint_sha256 must be a lowercase SHA-256 digest.")
    if expected_upstream_checkpoint_sha256 != input_checkpoint_sha256:
        raise ReferenceStageError("Expected upstream digest differs from the exact input checkpoint content address.")
    if expected_input_identity.source_sha256 != expected_source_sha256:
        raise ReferenceStageError("Expected input checkpoint source SHA differs from the requested source identity.")
    if expected_input_identity.coordinate_frame != expected_coordinate_frame:
        raise ReferenceStageError("Expected input checkpoint frame differs from the requested coordinate frame.")
    resolved_timeout = runner.timeout_seconds if timeout_seconds is None else timeout_seconds
    if isinstance(resolved_timeout, bool) or type(resolved_timeout) not in (int, float):
        raise ReferenceStageError("timeout_seconds must be finite and in (0,3600].")
    try:
        resolved_timeout = float(resolved_timeout)
    except (OverflowError, ValueError):
        raise ReferenceStageError("timeout_seconds must be finite and in (0,3600].") from None
    if not math.isfinite(resolved_timeout) or not 0 < resolved_timeout <= MAX_REFERENCE_STAGE_TIMEOUT_SECONDS:
        raise ReferenceStageError("timeout_seconds must be finite and in (0,3600].")

    try:
        stage = recipe.stage(stage_id)
    except KeyError as error:
        raise ReferenceStageError(f"Recipe does not contain stage {stage_id!r}.") from error
    if not isinstance(input_bindings, Mapping) or not isinstance(input_roles, Mapping):
        raise ReferenceStageError("Input bindings and raster roles must be explicit mappings.")
    if any(not isinstance(key, str) or not isinstance(value, str) for key, value in input_bindings.items()):
        raise ReferenceStageError("Input bindings must map recipe port names to checkpoint array names.")
    if any(not isinstance(key, str) or not isinstance(value, RasterRole) for key, value in input_roles.items()):
        raise ReferenceStageError("Input roles must map recipe port names to RasterRole contracts.")

    checkpoint = load_stage_input(
        input_root,
        input_checkpoint_sha256,
        expected_identity=expected_input_identity,
        expected_measurement=expected_input_measurement,
        limits=limits,
    )
    if checkpoint.identity.source_sha256 != expected_source_sha256:
        raise ReferenceStageError("Loaded input checkpoint source SHA does not match the requested source.")
    if checkpoint.identity.coordinate_frame != expected_coordinate_frame:
        raise ReferenceStageError("Loaded input checkpoint frame does not exactly match the requested frame.")

    def state_without_execution(state: StageRunState, reason: str) -> ReferenceStageOutcome:
        return _failed_outcome(
            recipe,
            stage_id,
            input_checkpoint_sha256,
            reason,
            state=state,
        )

    if recipe.provenance.mode is not RecipeProvenanceMode.ORIGINAL_REFERENCE:
        return state_without_execution(
            StageRunState.UNSUPPORTED,
            "Only pinned original-reference PlantCV stages can run through this bridge.",
        )
    if recipe.provenance.evidence_status is SourceResolution.UNAVAILABLE:
        return state_without_execution(StageRunState.UNAVAILABLE, recipe.provenance.evidence_note)
    if stage.resolution is SourceResolution.UNAVAILABLE:
        return state_without_execution(StageRunState.UNAVAILABLE, stage.resolution_note)
    if stage.resolution is SourceResolution.UNRESOLVED:
        return state_without_execution(StageRunState.UNRESOLVED, stage.resolution_note)
    if stage.operation_id not in SUPPORTED_PLANTCV_OPERATIONS or stage.operation_id not in SUPPORTED_REFERENCE_STAGE_OPERATIONS:
        return state_without_execution(
            StageRunState.UNSUPPORTED,
            f"Public PlantCV operation {stage.operation_id!r} has no bound output semantics in this bridge.",
        )

    if recipe.provenance.evidence_status is not SourceResolution.PINNED:
        raise ReferenceStageError("Original-reference provenance must have pinned evidence status.")
    parameters = _validate_pinned_stage(recipe, stage)
    expected_inputs = {port.name for port in stage.ports if port.direction == "input" and port.required}
    if expected_inputs != {"gray"} or set(input_bindings) != expected_inputs or set(input_roles) != expected_inputs:
        raise ReferenceStageError("Threshold stage requires one explicitly bound gray raster input.")
    bound_name = input_bindings["gray"]
    if bound_name not in checkpoint.arrays or checkpoint.raster_roles.get(bound_name) != input_roles["gray"]:
        raise ReferenceStageError("Bound input raster is missing or its declared role differs from the checkpoint.")
    gray = checkpoint.arrays[bound_name]
    input_role = input_roles["gray"]
    if (
        input_role.kind != "image"
        or input_role.layout != "HW"
        or input_role.dtype != np.dtype(np.uint8).str
        or input_role.value_range != (0, 255)
        or gray.dtype != np.dtype(np.uint8)
        or tuple(gray.shape) != expected_coordinate_frame.shape
        or not gray.flags.c_contiguous
        or gray.nbytes > MAX_ARRAY_BYTES
        or gray.size > MAX_ARRAY_ELEMENTS
    ):
        raise ReferenceStageError(
            "The exact stage input must be a bounded contiguous uint8 HxW gray raster in the requested frame."
        )

    derived_identity = expected_reference_stage_identity(
        checkpoint.identity,
        recipe,
        stage_id,
        input_checkpoint_sha256,
    )
    if expected_output_identity != derived_identity:
        raise ReferenceStageError("Expected output checkpoint identity differs from the exact stage/source/frame/upstream chain.")
    validate_checkpoint_identity(
        recipe,
        stage_id,
        derived_identity,
        expected_identity=expected_output_identity,
        expected_upstream_checkpoint_sha256=input_checkpoint_sha256,
    )
    if _cancelled(cancel):
        return _failed_outcome(recipe, stage_id, input_checkpoint_sha256, "Cancelled before PlantCV execution.")

    availability = plantcv_reference_status(
        python_executable=runner.python_executable,
        source_manifest_path=runner.source_manifest_path,
        source_root=runner.source_root,
        environment_manifest_path=runner.environment_manifest_path,
        requirements_lock_path=runner.requirements_lock_path,
    )
    if availability.get("available") is not True:
        return _failed_outcome(
            recipe,
            stage_id,
            input_checkpoint_sha256,
            str(availability.get("reason", "Pinned PlantCV reference prerequisites are unavailable.")),
            state=StageRunState.UNAVAILABLE,
        )

    started_ns = time.perf_counter_ns()
    try:
        result = runner.run(
            stage.operation_id,
            {"gray": gray},
            parameters,
            timeout_seconds=resolved_timeout,
            cancel_event=cancel,
        )
    except ReferenceIdentityError as error:
        return _failed_outcome(
            recipe,
            stage_id,
            input_checkpoint_sha256,
            f"Pinned PlantCV source or environment became unavailable: {error}",
            state=StageRunState.UNAVAILABLE,
        )
    except Exception as error:
        cancelled = _cancelled(cancel)
        prefix = "Cancelled during PlantCV execution" if cancelled else "PlantCV execution failed"
        return _failed_outcome(recipe, stage_id, input_checkpoint_sha256, f"{prefix}: {type(error).__name__}: {error}")

    checked = _validate_result(
        result,
        recipe=recipe,
        stage_id=stage_id,
        operation_id=stage.operation_id,
        parameters=parameters,
        frame=expected_coordinate_frame,
    )
    if _cancelled(cancel):
        return _failed_outcome(recipe, stage_id, input_checkpoint_sha256, "Cancelled before checkpoint publication.")

    mask = checked.arrays["mask"]
    status = ExecutionStatus.EMPTY if not np.any(mask) else ExecutionStatus.AVAILABLE
    state = StageRunState.EMPTY if status is ExecutionStatus.EMPTY else StageRunState.EXECUTED
    execution = ExecutionRecord(
        status=status,
        elapsed_ms=checked.wall_runtime_ns / 1_000_000,
        transfer_ms=None,
    )
    mapping = stage.source_mappings[0]
    metadata = _strict_metadata({
        "format": REFERENCE_STAGE_FORMAT,
        "version": REFERENCE_STAGE_VERSION,
        "stage_id": stage_id,
        "operation_id": stage.operation_id,
        "recipe_identity": recipe.identity.to_dict(),
        "source_mapping": mapping.to_dict(),
        "source_sha256": expected_source_sha256,
        "coordinate_frame": expected_coordinate_frame.to_dict(),
        "upstream_checkpoint_sha256": input_checkpoint_sha256,
        "input_checkpoint_identity": checkpoint.identity.to_dict(),
        "input_bindings": dict(input_bindings),
        "input_raster_roles": {name: role.to_dict() for name, role in input_roles.items()},
        "output_raster_roles": {"mask": RasterRole("mask", np.dtype(np.uint8).str).to_dict()},
        "runner_identity": dict(checked.identity),
        "parameters": parameters,
        "effective_parameters": dict(checked.effective_parameters),
        "observations": dict(checked.observations),
        "intermediates": list(checked.intermediates),
        "visualizations": dict(checked.visualizations),
        "channel_semantics": dict(checked.channel_semantics),
        "computation_runtime_ns": checked.computation_runtime_ns,
        "wall_runtime_ns": checked.wall_runtime_ns,
        "stage_runtime_ns": max(0, time.perf_counter_ns() - started_ns),
        "output_inventory": [{
            "name": "mask",
            "dtype": mask.dtype.str,
            "shape": list(mask.shape),
            "nbytes": int(mask.nbytes),
        }],
        "measurement_contract": output_measurement.to_dict(),
        "scientific_validation_status": "not_certified",
    })

    try:
        reference = publish_checkpoint(
            output_root,
            derived_identity,
            {"mask": mask},
            execution=execution,
            measurement=output_measurement,
            raster_roles={"mask": RasterRole("mask", np.dtype(np.uint8).str)},
            metadata=metadata,
            allow_cpu_transfer=True,
            limits=limits,
            cancel=cancel,
        )
    except CheckpointCancelled as error:
        return _failed_outcome(recipe, stage_id, input_checkpoint_sha256, f"Cancelled during checkpoint publication: {error}")

    run_record = StageRunRecord(
        stage_id=stage_id,
        recipe_identity=recipe.identity,
        state=state,
        checkpoint_sha256=reference.checkpoint_sha256,
        upstream_checkpoint_sha256=input_checkpoint_sha256,
    )
    validate_stage_run(recipe, run_record, expected_upstream_checkpoint_sha256=input_checkpoint_sha256)
    validate_checkpoint_identity(
        recipe,
        stage_id,
        reference.identity,
        expected_identity=derived_identity,
        expected_upstream_checkpoint_sha256=input_checkpoint_sha256,
    )
    return ReferenceStageOutcome(run_record, reference)


__all__ = [
    "MAX_REFERENCE_STAGE_METADATA_BYTES",
    "REFERENCE_STAGE_FORMAT",
    "REFERENCE_STAGE_VERSION",
    "ReferenceStageError",
    "ReferenceStageOutcome",
    "SUPPORTED_REFERENCE_STAGE_OPERATIONS",
    "expected_reference_stage_identity",
    "run_reference_stage",
]
