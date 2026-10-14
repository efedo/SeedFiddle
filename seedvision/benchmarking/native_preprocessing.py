"""Benchmark-only native CUDA execution of the five seed preprocessing stages.

This coordinator connects the separately pinned median/morphology and threshold
stage bridges. Full-resolution calculations remain inside those CUDA bridges;
the only host transfers are the explicit immutable checkpoint archive boundary.
It does not execute the watershed or measurement portions of a recipe.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from seedvision.benchmarking.checkpoints import (
    CheckpointIdentity,
    CheckpointLimits,
    CheckpointReference,
    CoordinateFrame,
    ExecutionStatus,
    MeasurementContract,
    RasterRole,
    load_stage_input,
)
from seedvision.benchmarking.recipe_catalogue import RecipeCatalogueError, build_recipe_catalogue
from seedvision.benchmarking.reference_runner import MAX_ARRAY_BYTES, MAX_ARRAY_ELEMENTS
from seedvision.benchmarking.recipes import (
    RecipeIdentity,
    RecipeSpec,
    RecipeStage,
    StageRunRecord,
    StageRunState,
    validate_checkpoint_identity,
    validate_stage_run,
)
from seedvision.benchmarking.native_mask_stage import (
    NativeMaskIdentity,
    NativeMaskStageError,
    NativeMaskUnavailableError,
    NativeMaskStageOutcome,
    expected_native_mask_stage_identity,
    native_mask_identity,
    run_native_mask_stage,
)
from seedvision.benchmarking.native_threshold_stage import (
    NativeThresholdIdentity,
    NativeThresholdOutcome,
    NativeThresholdStageError,
    NativeThresholdUnavailableError,
    expected_native_threshold_stage_identity,
    native_threshold_identity,
    run_native_threshold_stage,
    _domain_reason as _native_threshold_domain_reason,
)


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_THRESHOLD_RECIPES = {
    "plantcv_seed_binary": "threshold.binary",
    "plantcv_seed_mean": "threshold.mean",
    "plantcv_seed_otsu": "threshold.otsu",
}
_INPUT_OUTPUT_CONTRACTS = {
    "filter.median_blur": ("gray", "GrayImage", "image", "GrayImage"),
    "threshold.binary": ("gray", "GrayImage", "mask", "BinaryMask"),
    "threshold.mean": ("gray", "GrayImage", "mask", "BinaryMask"),
    "threshold.otsu": ("gray", "GrayImage", "mask", "BinaryMask"),
    "morphology.fill": ("mask", "BinaryMask", "mask", "BinaryMask"),
    # The native erosion/dilation bridges deliberately expose their mask input
    # on the catalogue's historical ``gray`` port.
    "morphology.erode": ("gray", "BinaryMask", "mask", "BinaryMask"),
    "morphology.dilate": ("gray", "BinaryMask", "mask", "BinaryMask"),
}
_IMAGE_ROLE = RasterRole("image", np.dtype(np.uint8).str, "HW", (0, 255))
_MASK_ROLE = RasterRole("mask", np.dtype(np.uint8).str, "HW")
_OUTPUT_ROLE = {"GrayImage": _IMAGE_ROLE, "BinaryMask": _MASK_ROLE}


class NativePreprocessingError(ValueError):
    """The selected immutable recipe or prepared-image binding is invalid."""


class NativePreprocessingStatus(StrEnum):
    EXECUTED = "executed"
    EMPTY = "empty"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNAVAILABLE = "unavailable"
    UNSUPPORTED = "unsupported"
    UNRESOLVED = "unresolved"


class NativePreprocessingStageStatus(StrEnum):
    EXECUTED = "executed"
    EMPTY = "empty"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNAVAILABLE = "unavailable"
    UNSUPPORTED = "unsupported"
    UNRESOLVED = "unresolved"
    NOT_RUN = "not_run"


@dataclass(frozen=True, slots=True)
class NativePreprocessingStageResult:
    """One stage's projected recipe, exact bindings, status and checkpoint."""

    stage: RecipeStage
    selected_stage_id: str
    status: NativePreprocessingStageStatus
    run_record: StageRunRecord
    input_checkpoint_sha256: str | None
    input_bindings: Mapping[str, str]
    input_roles: Mapping[str, RasterRole]
    parameters: Mapping[str, Any]
    checkpoint_reference: CheckpointReference | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", NativePreprocessingStageStatus(self.status))
        if not isinstance(self.stage, RecipeStage) or not isinstance(self.run_record, StageRunRecord):
            raise TypeError("A native preprocessing result requires its immutable stage and run record.")
        if not isinstance(self.selected_stage_id, str) or not self.selected_stage_id.strip():
            raise ValueError("selected_stage_id must identify the stage in the full caller recipe.")
        if self.run_record.stage_id != self.stage.stage_id:
            raise ValueError("Preprocessing result stage and run record differ.")
        if self.status is NativePreprocessingStageStatus.NOT_RUN:
            if self.run_record.state is not StageRunState.NOT_RUN or self.checkpoint_reference is not None:
                raise ValueError("A not-run stage cannot carry execution evidence.")
        else:
            expected_state = {
                NativePreprocessingStageStatus.EXECUTED: StageRunState.EXECUTED,
                NativePreprocessingStageStatus.EMPTY: StageRunState.EMPTY,
                NativePreprocessingStageStatus.FAILED: StageRunState.FAILED,
                NativePreprocessingStageStatus.CANCELLED: StageRunState.FAILED,
                NativePreprocessingStageStatus.UNAVAILABLE: StageRunState.UNAVAILABLE,
                NativePreprocessingStageStatus.UNSUPPORTED: StageRunState.UNSUPPORTED,
                NativePreprocessingStageStatus.UNRESOLVED: StageRunState.UNRESOLVED,
            }[self.status]
            if self.run_record.state is not expected_state:
                raise ValueError("Preprocessing status and immutable stage run state differ.")
            completed = self.status in {
                NativePreprocessingStageStatus.EXECUTED,
                NativePreprocessingStageStatus.EMPTY,
            }
            if completed != (self.checkpoint_reference is not None):
                raise ValueError("Only executed or empty stages carry a checkpoint reference.")
            if self.checkpoint_reference is not None and (
                self.run_record.checkpoint_sha256 != self.checkpoint_reference.checkpoint_sha256
            ):
                raise ValueError("Stage run and checkpoint reference hashes differ.")
        if self.input_checkpoint_sha256 is not None and not _SHA256.fullmatch(self.input_checkpoint_sha256):
            raise ValueError("Input checkpoint address must be a lowercase SHA-256 digest.")
        if not isinstance(self.input_bindings, Mapping) or not isinstance(self.input_roles, Mapping):
            raise TypeError("Input bindings and raster roles must be mappings.")
        if not isinstance(self.parameters, Mapping):
            raise TypeError("Stage parameters must be a mapping.")
        object.__setattr__(self, "input_bindings", MappingProxyType(dict(self.input_bindings)))
        object.__setattr__(self, "input_roles", MappingProxyType(dict(self.input_roles)))
        object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))


@dataclass(frozen=True, slots=True)
class NativePreprocessingOutcome:
    """The complete five-stage result and both exact native environment pins."""

    status: NativePreprocessingStatus
    recipe_identity: RecipeIdentity
    native_mask_identity: NativeMaskIdentity
    native_threshold_identity: NativeThresholdIdentity
    stages: tuple[NativePreprocessingStageResult, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", NativePreprocessingStatus(self.status))
        if not isinstance(self.recipe_identity, RecipeIdentity):
            raise TypeError("recipe_identity must preserve the caller's exact selected recipe identity.")
        if not isinstance(self.native_mask_identity, NativeMaskIdentity):
            raise TypeError("native_mask_identity must be an exact native implementation pin.")
        if not isinstance(self.native_threshold_identity, NativeThresholdIdentity):
            raise TypeError("native_threshold_identity must be an exact native implementation pin.")
        stages = tuple(self.stages)
        if len(stages) != 5 or any(not isinstance(item, NativePreprocessingStageResult) for item in stages):
            raise ValueError("A native preprocessing outcome must retain all five stage records.")
        expected_threshold = _THRESHOLD_RECIPES.get(self.recipe_identity.recipe_id)
        expected_operations = (
            "filter.median_blur", expected_threshold, "morphology.fill",
            "morphology.erode", "morphology.dilate",
        )
        if expected_threshold is None or tuple(item.stage.operation_id for item in stages) != expected_operations:
            raise ValueError("Outcome stages do not match the authored five-stage preprocessing chain.")
        projected_recipe_ids = tuple(
            self.recipe_identity.recipe_id if item.stage.operation_id.startswith("threshold.")
            else "plantcv_seed_binary"
            for item in stages
        )
        if tuple(item.run_record.recipe_identity.recipe_id for item in stages) != projected_recipe_ids:
            raise ValueError("Stage records do not use the exact recipe identity required by their native bridge.")
        object.__setattr__(self, "stages", stages)

    @property
    def selected_recipe_identity(self) -> RecipeIdentity:
        """The caller's full recipe identity; stage records use bridge projections."""
        return self.recipe_identity

    @property
    def run_records(self) -> tuple[StageRunRecord, ...]:
        return tuple(item.run_record for item in self.stages)

    @property
    def final_checkpoint(self) -> CheckpointReference | None:
        """Expose a final artifact only when dilation completed successfully."""
        final = self.stages[-1]
        if (
            self.status in {NativePreprocessingStatus.EXECUTED, NativePreprocessingStatus.EMPTY}
            and all(item.status in {NativePreprocessingStageStatus.EXECUTED, NativePreprocessingStageStatus.EMPTY}
                    for item in self.stages)
            and final.status in {NativePreprocessingStageStatus.EXECUTED, NativePreprocessingStageStatus.EMPTY}
        ):
            return final.checkpoint_reference
        return None

    @property
    def latest_checkpoint(self) -> CheckpointReference | None:
        """Return the newest completed checkpoint, including a valid intermediate."""
        return next(
            (item.checkpoint_reference for item in reversed(self.stages) if item.checkpoint_reference is not None),
            None,
        )


def _catalogue_recipe(recipe_id: str) -> RecipeSpec:
    try:
        return build_recipe_catalogue().recipe(recipe_id)
    except (RecipeCatalogueError, OSError, ValueError, KeyError) as error:
        raise NativePreprocessingError("Committed PlantCV seed recipe pins are unavailable.") from error


def _chain_stages(recipe: RecipeSpec) -> tuple[RecipeStage, ...]:
    if recipe.recipe_id not in _THRESHOLD_RECIPES:
        raise NativePreprocessingError("Only the three authored pinned PlantCV seed recipes are supported.")
    operations = (
        "filter.median_blur", _THRESHOLD_RECIPES[recipe.recipe_id], "morphology.fill",
        "morphology.erode", "morphology.dilate",
    )
    selected: list[RecipeStage] = []
    for operation in operations:
        matches = tuple(stage for stage in recipe.stages if stage.operation_id == operation)
        if len(matches) != 1:
            raise NativePreprocessingError(f"Recipe must contain exactly one {operation!r} preprocessing stage.")
        selected.append(matches[0])
    return tuple(selected)


def _validate_selected_recipe(
    recipe: RecipeSpec,
    canonical: RecipeSpec,
    mask_identity: NativeMaskIdentity,
    threshold_identity: NativeThresholdIdentity,
) -> None:
    if not any(recipe.provenance == provenance for provenance in (
        canonical.provenance,
        mask_identity.recipe_provenance(),
        threshold_identity.recipe_provenance(),
    )):
        raise NativePreprocessingError("Recipe provenance is not the pinned PlantCV target or either exact native backend.")
    if len(recipe.stages) != len(canonical.stages):
        raise NativePreprocessingError("Recipe stage count differs from the pinned authored seed recipe.")
    normalized_stages = []
    for current, pinned in zip(recipe.stages, canonical.stages, strict=True):
        if current.stage_id != pinned.stage_id or len(current.parameters) != len(pinned.parameters):
            raise NativePreprocessingError("Recipe stage order, IDs, or parameter declarations differ from the pinned recipe.")
        parameters = []
        for actual, baseline in zip(current.parameters, pinned.parameters, strict=True):
            if actual.parameter_id != baseline.parameter_id:
                raise NativePreprocessingError("Parameter IDs differ from the pinned authored recipe.")
            if actual.affects_calculation:
                parameters.append(replace(actual, value=baseline.value))
            else:
                if actual != baseline:
                    raise NativePreprocessingError("Non-calculation controls differ from the pinned authored recipe.")
                parameters.append(actual)
        normalized_stages.append(replace(current, parameters=tuple(parameters)))
    normalized = replace(
        recipe,
        provenance=canonical.provenance,
        stages=tuple(normalized_stages),
        layout=canonical.layout,
    )
    if normalized != canonical:
        raise NativePreprocessingError(
            "Recipe source pins, ports, dependencies, or non-preprocessing contracts differ from the authored seed recipe."
        )


def _validate_actual_chain(recipe: RecipeSpec, stages: tuple[RecipeStage, ...]) -> None:
    stage_by_id = {stage.stage_id: stage for stage in recipe.stages}
    for index, stage in enumerate(stages):
        input_name, input_type, output_name, output_type = _INPUT_OUTPUT_CONTRACTS[stage.operation_id]
        inputs = [port for port in stage.ports if port.direction == "input"]
        outputs = [port for port in stage.ports if port.direction == "output"]
        if len(inputs) != 1 or len(outputs) != 1 or (
            inputs[0].name, inputs[0].data_type, outputs[0].name, outputs[0].data_type
        ) != (input_name, input_type, output_name, output_type):
            raise NativePreprocessingError(f"Pinned ports differ on preprocessing stage {stage.stage_id!r}.")
        incoming = tuple(edge for edge in recipe.dependencies if edge.target_stage == stage.stage_id)
        if len(incoming) != 1:
            raise NativePreprocessingError(f"Stage {stage.stage_id!r} must have exactly one graph input edge.")
        edge = incoming[0]
        if (edge.target_port, edge.data_type) != (input_name, input_type):
            raise NativePreprocessingError(f"Input edge does not match the declared port on {stage.stage_id!r}.")
        if index == 0:
            source = stage_by_id.get(edge.source_stage)
            if (
                source is None or source.operation_id != "external.image_inputs"
                or edge.source_port != "gray_image" or edge.data_type != "GrayImage"
            ):
                raise NativePreprocessingError("Median input is not connected to the external grayscale image port.")
        else:
            upstream = stages[index - 1]
            previous_output = _INPUT_OUTPUT_CONTRACTS[upstream.operation_id][2:]
            if (edge.source_stage, edge.source_port, edge.data_type) != (
                upstream.stage_id, previous_output[0], previous_output[1]
            ):
                raise NativePreprocessingError(
                    f"Graph edge into {stage.stage_id!r} does not come from the preceding preprocessing output."
                )


def _stage_recipe(
    selected_recipe: RecipeSpec,
    canonical: RecipeSpec,
    stage: RecipeStage,
    implementation: NativeMaskIdentity | NativeThresholdIdentity,
) -> tuple[RecipeSpec, RecipeStage]:
    """Project one selected operation into the exact one-stage bridge contract."""
    if stage.operation_id.startswith("threshold."):
        bridge_canonical = canonical
        bridge_stage = bridge_canonical.stage(stage.stage_id)
    else:
        # The current mask bridge publishes these four operations under its
        # binary-seed canonical stage IDs. Preserve the caller's full recipe
        # identity separately and transfer only this operation's selected
        # values into that exact bridge stage contract.
        bridge_canonical = _catalogue_recipe("plantcv_seed_binary")
        matches = tuple(item for item in bridge_canonical.stages if item.operation_id == stage.operation_id)
        if len(matches) != 1:
            raise NativePreprocessingError("Native mask bridge catalogue lacks one exact stage operation.")
        bridge_stage = matches[0]
    selected_parameters = selected_recipe.stage(stage.stage_id).parameters
    if len(selected_parameters) != len(bridge_stage.parameters) or any(
        selected.parameter_id != pinned.parameter_id
        for selected, pinned in zip(selected_parameters, bridge_stage.parameters, strict=True)
    ):
        raise NativePreprocessingError("Selected operation parameter inventory differs from its native bridge contract.")
    bridge_stage = replace(
        bridge_stage,
        parameters=tuple(
            replace(pinned, value=selected.value)
            for pinned, selected in zip(bridge_stage.parameters, selected_parameters, strict=True)
        ),
    )
    stages = tuple(
        bridge_stage if pinned.operation_id == stage.operation_id else pinned
        for pinned in bridge_canonical.stages
    )
    projection = replace(
        bridge_canonical,
        provenance=implementation.recipe_provenance(),
        stages=stages,
        layout=bridge_canonical.layout,
    )
    return projection, bridge_stage


def _expected_output_identity(
    input_identity: CheckpointIdentity,
    stage: RecipeStage,
    recipe: RecipeSpec,
    upstream_sha256: str,
    mask_identity: NativeMaskIdentity,
    threshold_identity: NativeThresholdIdentity,
) -> CheckpointIdentity:
    if stage.operation_id.startswith("threshold."):
        return expected_native_threshold_stage_identity(
            input_identity, recipe, stage.stage_id, upstream_sha256, threshold_identity,
        )
    return expected_native_mask_stage_identity(
        input_identity, recipe, stage.stage_id, upstream_sha256, mask_identity,
    )


def _cancelled(cancel: Any) -> bool:
    if cancel is None:
        return False
    if callable(cancel):
        return bool(cancel())
    return bool(cancel.is_set())


def _bindings_for(index: int, stages: tuple[RecipeStage, ...], initial_array: str) -> tuple[dict[str, str], dict[str, RasterRole]]:
    if index == 0:
        return {"gray": initial_array}, {"gray": _IMAGE_ROLE}
    previous = stages[index - 1]
    output_name = _INPUT_OUTPUT_CONTRACTS[previous.operation_id][2]
    output_type = _INPUT_OUTPUT_CONTRACTS[previous.operation_id][3]
    current_input = _INPUT_OUTPUT_CONTRACTS[stages[index].operation_id][0]
    role = _OUTPUT_ROLE[output_type]
    return {current_input: output_name}, {current_input: role}


def _stage_result(
    stage: RecipeStage,
    selected_stage_id: str,
    projected: RecipeSpec,
    status: NativePreprocessingStageStatus,
    reason: str,
    *,
    upstream_sha256: str | None,
    bindings: Mapping[str, str],
    roles: Mapping[str, RasterRole],
    reference: CheckpointReference | None = None,
    record: StageRunRecord | None = None,
) -> NativePreprocessingStageResult:
    if record is None:
        state = {
            NativePreprocessingStageStatus.NOT_RUN: StageRunState.NOT_RUN,
            NativePreprocessingStageStatus.FAILED: StageRunState.FAILED,
            NativePreprocessingStageStatus.CANCELLED: StageRunState.FAILED,
            NativePreprocessingStageStatus.UNAVAILABLE: StageRunState.UNAVAILABLE,
            NativePreprocessingStageStatus.UNSUPPORTED: StageRunState.UNSUPPORTED,
            NativePreprocessingStageStatus.UNRESOLVED: StageRunState.UNRESOLVED,
        }[status]
        record = StageRunRecord(
            stage_id=stage.stage_id,
            recipe_identity=projected.identity,
            state=state,
            upstream_checkpoint_sha256=upstream_sha256,
            reason=" ".join(str(reason).split())[:512],
        )
        validate_stage_run(projected, record)
    return NativePreprocessingStageResult(
        stage=stage,
        selected_stage_id=selected_stage_id,
        status=status,
        run_record=record,
        input_checkpoint_sha256=upstream_sha256,
        input_bindings=bindings,
        input_roles=roles,
        parameters=projected.parameters_for_runner(stage.stage_id),
        checkpoint_reference=reference,
    )


def _status_from_record(record: StageRunRecord, cancel: Any) -> NativePreprocessingStageStatus:
    mapping = {
        StageRunState.EXECUTED: NativePreprocessingStageStatus.EXECUTED,
        StageRunState.EMPTY: NativePreprocessingStageStatus.EMPTY,
        StageRunState.UNAVAILABLE: NativePreprocessingStageStatus.UNAVAILABLE,
        StageRunState.UNSUPPORTED: NativePreprocessingStageStatus.UNSUPPORTED,
        StageRunState.UNRESOLVED: NativePreprocessingStageStatus.UNRESOLVED,
    }
    if record.state in mapping:
        return mapping[record.state]
    if record.state is StageRunState.FAILED and _cancelled(cancel):
        return NativePreprocessingStageStatus.CANCELLED
    return NativePreprocessingStageStatus.FAILED


def _not_run_suffix(
    stages: tuple[RecipeStage, ...],
    selected_stage_ids: tuple[str, ...],
    projected: tuple[RecipeSpec, ...],
    results: list[NativePreprocessingStageResult],
    *,
    start: int,
    initial_array: str,
) -> None:
    for index in range(start, len(stages)):
        bindings, roles = _bindings_for(index, stages, initial_array)
        results.append(_stage_result(
            stages[index], selected_stage_ids[index], projected[index], NativePreprocessingStageStatus.NOT_RUN,
            f"Not run because preprocessing stopped before {stages[index].stage_id}.",
            upstream_sha256=None, bindings=bindings, roles=roles,
        ))


def _overall_status(status: NativePreprocessingStageStatus) -> NativePreprocessingStatus:
    if status is NativePreprocessingStageStatus.NOT_RUN:
        raise ValueError("NOT_RUN is not a terminal preprocessing status.")
    return NativePreprocessingStatus(status.value)


def _preflight_failure(
    stages: tuple[RecipeStage, ...],
    selected_stage_ids: tuple[str, ...],
    projected: tuple[RecipeSpec, ...],
    stage_index: int,
    status: NativePreprocessingStageStatus,
    reason: str,
    recipe_identity: RecipeIdentity,
    mask_identity: NativeMaskIdentity,
    threshold_identity: NativeThresholdIdentity,
    initial_array: str,
    initial_checkpoint_sha256: str,
) -> NativePreprocessingOutcome:
    results: list[NativePreprocessingStageResult] = []
    for index, stage in enumerate(stages):
        bindings, roles = _bindings_for(index, stages, initial_array)
        upstream = initial_checkpoint_sha256 if index == 0 else None
        if index == stage_index:
            results.append(_stage_result(
                stage, selected_stage_ids[index], projected[index], status, reason, upstream_sha256=upstream,
                bindings=bindings, roles=roles,
            ))
        else:
            results.append(_stage_result(
                stage, selected_stage_ids[index], projected[index], NativePreprocessingStageStatus.NOT_RUN,
                f"Not run because whole-chain preflight stopped at {stages[stage_index].stage_id}.",
                upstream_sha256=upstream, bindings=bindings, roles=roles,
            ))
    return NativePreprocessingOutcome(
        _overall_status(status), recipe_identity, mask_identity, threshold_identity, tuple(results),
    )


def run_native_preprocessing(
    input_root: Path | str,
    input_checkpoint_sha256: str,
    recipe: RecipeSpec,
    output_root: Path | str,
    *,
    expected_source_sha256: str,
    expected_coordinate_frame: CoordinateFrame,
    expected_recipe_identity: RecipeIdentity,
    expected_input_identity: CheckpointIdentity,
    expected_input_measurement: MeasurementContract,
    output_measurement: MeasurementContract,
    expected_native_mask_identity: NativeMaskIdentity,
    expected_native_threshold_identity: NativeThresholdIdentity,
    device: Any,
    input_bindings: Mapping[str, str],
    input_roles: Mapping[str, RasterRole],
    allow_cpu_transfer: bool,
    limits: CheckpointLimits = CheckpointLimits(),
    cancel: Any = None,
) -> NativePreprocessingOutcome:
    """Run median, one selected threshold, fill, erosion, and dilation.

    Each bridge receives a canonical-layout recipe projection with only its own
    selected calculation values. The outcome separately retains the caller's
    full recipe identity, including display-only layout and other selections.
    """
    import torch

    if not isinstance(recipe, RecipeSpec) or not isinstance(expected_recipe_identity, RecipeIdentity):
        raise TypeError("recipe and expected_recipe_identity must be immutable recipe contracts.")
    if recipe.identity != expected_recipe_identity:
        raise NativePreprocessingError("Recipe identity differs from the caller's exact expected recipe identity.")
    if not isinstance(expected_native_mask_identity, NativeMaskIdentity):
        raise TypeError("expected_native_mask_identity must be an exact NativeMaskIdentity.")
    if not isinstance(expected_native_threshold_identity, NativeThresholdIdentity):
        raise TypeError("expected_native_threshold_identity must be an exact NativeThresholdIdentity.")
    if type(device) is not torch.device or device.type != "cuda" or device.index is None:
        raise NativePreprocessingError("device must be an explicit torch.device such as cuda:0.")
    if device.index != expected_native_mask_identity.device_index or device.index != expected_native_threshold_identity.device_index:
        raise NativePreprocessingError("Device index differs from one of the exact native implementation identities.")
    if not isinstance(expected_input_identity, CheckpointIdentity):
        raise TypeError("expected_input_identity must be a CheckpointIdentity.")
    if not isinstance(expected_coordinate_frame, CoordinateFrame):
        raise TypeError("expected_coordinate_frame must be a CoordinateFrame.")
    if not isinstance(expected_input_measurement, MeasurementContract) or not isinstance(output_measurement, MeasurementContract):
        raise TypeError("Explicit input and output measurement contracts are required.")
    if not isinstance(limits, CheckpointLimits):
        raise TypeError("limits must be an explicit CheckpointLimits value.")
    if output_measurement.eligible:
        raise NativePreprocessingError("Native preprocessing cannot certify scientific measurement eligibility.")
    if type(allow_cpu_transfer) is not bool or not allow_cpu_transfer:
        raise NativePreprocessingError("Benchmark checkpoint serialization requires explicit allow_cpu_transfer=True.")
    if not isinstance(expected_source_sha256, str) or not _SHA256.fullmatch(expected_source_sha256):
        raise NativePreprocessingError("expected_source_sha256 must be a lowercase SHA-256 digest.")
    if not isinstance(input_checkpoint_sha256, str) or not _SHA256.fullmatch(input_checkpoint_sha256):
        raise NativePreprocessingError("input_checkpoint_sha256 must be a lowercase SHA-256 digest.")
    if expected_input_identity.source_sha256 != expected_source_sha256:
        raise NativePreprocessingError("Prepared checkpoint identity has a different source digest.")
    if expected_input_identity.coordinate_frame != expected_coordinate_frame:
        raise NativePreprocessingError("Prepared checkpoint identity has a different coordinate frame.")
    pixel_count = expected_coordinate_frame.shape[0] * expected_coordinate_frame.shape[1]
    if pixel_count > min(MAX_ARRAY_ELEMENTS, MAX_ARRAY_BYTES, limits.max_array_bytes, limits.max_total_array_bytes):
        raise NativePreprocessingError("Requested frame exceeds the bounded native uint8 raster domain.")
    if not isinstance(input_bindings, Mapping) or not isinstance(input_roles, Mapping):
        raise NativePreprocessingError("Prepared-image bindings and raster roles must be explicit mappings.")
    if any(not isinstance(key, str) or not isinstance(value, str) for key, value in input_bindings.items()):
        raise NativePreprocessingError("Input bindings must map ports to checkpoint array names.")
    if any(not isinstance(key, str) or not isinstance(value, RasterRole) for key, value in input_roles.items()):
        raise NativePreprocessingError("Input roles must map ports to RasterRole contracts.")
    if cancel is not None and not callable(cancel) and not callable(getattr(cancel, "is_set", None)):
        raise NativePreprocessingError("cancel must be a callable or an event with is_set().")

    canonical = _catalogue_recipe(recipe.recipe_id)
    selected_stages = _chain_stages(recipe)
    _validate_selected_recipe(recipe, canonical, expected_native_mask_identity, expected_native_threshold_identity)
    _validate_actual_chain(recipe, selected_stages)
    try:
        checkpoint = load_stage_input(
            input_root,
            input_checkpoint_sha256,
            expected_identity=expected_input_identity,
            expected_measurement=expected_input_measurement,
            limits=limits,
        )
    except Exception as error:
        raise NativePreprocessingError(f"Prepared grayscale checkpoint is invalid: {error}") from error
    if checkpoint.identity.source_sha256 != expected_source_sha256:
        raise NativePreprocessingError("Loaded checkpoint source digest differs from the requested source.")
    if checkpoint.identity.coordinate_frame != expected_coordinate_frame:
        raise NativePreprocessingError("Loaded checkpoint frame differs from the requested coordinate frame.")
    if set(input_bindings) != {"gray"} or set(input_roles) != {"gray"}:
        raise NativePreprocessingError("The median input edge requires one explicit 'gray' image binding and role.")
    initial_array = input_bindings["gray"]
    initial_role = input_roles["gray"]
    array = checkpoint.arrays.get(initial_array)
    if (
        initial_role != _IMAGE_ROLE
        or checkpoint.raster_roles.get(initial_array) != initial_role
        or not isinstance(array, np.ndarray)
        or array.dtype != np.dtype(np.uint8)
        or array.ndim != 2
        or tuple(array.shape) != expected_coordinate_frame.shape
        or not array.flags.c_contiguous
    ):
        raise NativePreprocessingError("Prepared grayscale binding must be the declared contiguous uint8 HxW image role.")

    stage_projections = tuple(
        _stage_recipe(
            recipe,
            canonical,
            stage,
            expected_native_threshold_identity if stage.operation_id.startswith("threshold.") else expected_native_mask_identity,
        )
        for stage in selected_stages
    )
    projected = tuple(item[0] for item in stage_projections)
    bridge_stages = tuple(item[1] for item in stage_projections)
    selected_stage_ids = tuple(stage.stage_id for stage in selected_stages)

    # Validate every selected native stage contract and every predicted output
    # identity before the first CUDA calculation. Layout is display-only, so all
    # bridge projections use the exact pinned canonical layout.
    preflight_identity = checkpoint.identity
    placeholder = "0" * 64
    for index, stage in enumerate(bridge_stages):
        try:
            preflight_identity = _expected_output_identity(
                preflight_identity, stage, projected[index], placeholder,
                expected_native_mask_identity, expected_native_threshold_identity,
            )
            if stage.operation_id.startswith("threshold."):
                domain_error = _native_threshold_domain_reason(
                    stage.operation_id,
                    projected[index].parameters_for_runner(stage.stage_id),
                )
                if domain_error is not None:
                    raise NativeThresholdStageError(domain_error)
        except (NativeMaskStageError, NativeThresholdStageError, ValueError, TypeError) as error:
            return _preflight_failure(
                bridge_stages, selected_stage_ids, projected, index, NativePreprocessingStageStatus.UNSUPPORTED,
                f"Whole-chain native preflight rejected {stage.stage_id}: {error}",
                recipe.identity, expected_native_mask_identity, expected_native_threshold_identity, initial_array,
                input_checkpoint_sha256,
            )

    # Check both environment pins before any stage starts. These calls inspect
    # code/runtime/device metadata but do not process the image.
    identity_failures: list[tuple[int, NativePreprocessingStageStatus, str]] = []
    try:
        if native_mask_identity(device) != expected_native_mask_identity:
            identity_failures.append((0, NativePreprocessingStageStatus.FAILED,
                                      "Current native mask implementation or CUDA environment differs from its caller pin."))
    except NativeMaskUnavailableError as error:
        identity_failures.append((0, NativePreprocessingStageStatus.UNAVAILABLE,
                                  f"Native mask implementation is unavailable: {type(error).__name__}: {error}"))
    except Exception as error:
        identity_failures.append((0, NativePreprocessingStageStatus.FAILED,
                                  f"Native mask identity validation failed: {type(error).__name__}: {error}"))
    try:
        if native_threshold_identity(device) != expected_native_threshold_identity:
            identity_failures.append((1, NativePreprocessingStageStatus.FAILED,
                                      "Current native threshold implementation or CUDA environment differs from its caller pin."))
    except NativeThresholdUnavailableError as error:
        identity_failures.append((1, NativePreprocessingStageStatus.UNAVAILABLE,
                                  f"Native threshold implementation is unavailable: {type(error).__name__}: {error}"))
    except Exception as error:
        identity_failures.append((1, NativePreprocessingStageStatus.FAILED,
                                  f"Native threshold identity validation failed: {type(error).__name__}: {error}"))
    if identity_failures:
        index, status, reason = min(identity_failures, key=lambda item: item[0])
        return _preflight_failure(
            bridge_stages, selected_stage_ids, projected, index, status, reason, recipe.identity,
            expected_native_mask_identity, expected_native_threshold_identity, initial_array,
            input_checkpoint_sha256,
        )
    if _cancelled(cancel):
        return _preflight_failure(
            bridge_stages, selected_stage_ids, projected, 0, NativePreprocessingStageStatus.CANCELLED,
            "Cancelled before native CUDA preprocessing began.", recipe.identity,
            expected_native_mask_identity, expected_native_threshold_identity, initial_array,
            input_checkpoint_sha256,
        )

    results: list[NativePreprocessingStageResult] = []
    current_root: Path | str = input_root
    current_digest = input_checkpoint_sha256
    current_identity = checkpoint.identity
    current_measurement = expected_input_measurement
    terminal = NativePreprocessingStageStatus.NOT_RUN

    for index, stage in enumerate(bridge_stages):
        bridge_recipe = projected[index]
        bindings, roles = _bindings_for(index, bridge_stages, initial_array)
        output_identity = _expected_output_identity(
            current_identity, stage, bridge_recipe, current_digest,
            expected_native_mask_identity, expected_native_threshold_identity,
        )
        common = {
            "input_root": current_root,
            "input_checkpoint_sha256": current_digest,
            "recipe": bridge_recipe,
            "stage_id": stage.stage_id,
            "output_root": output_root,
            "expected_source_sha256": expected_source_sha256,
            "expected_coordinate_frame": expected_coordinate_frame,
            "expected_recipe_identity": bridge_recipe.identity,
            "expected_upstream_checkpoint_sha256": current_digest,
            "expected_input_identity": current_identity,
            "expected_input_measurement": current_measurement,
            "expected_output_identity": output_identity,
            "device": device,
            "input_bindings": bindings,
            "input_roles": roles,
            "output_measurement": output_measurement,
            "allow_cpu_transfer": allow_cpu_transfer,
            "limits": limits,
            "cancel": cancel,
        }
        try:
            if stage.operation_id.startswith("threshold."):
                outcome: NativeMaskStageOutcome | NativeThresholdOutcome = run_native_threshold_stage(
                    **common, expected_implementation_identity=expected_native_threshold_identity,
                )
                stage_status = _status_from_record(outcome.run_record, cancel)
            else:
                outcome = run_native_mask_stage(
                    **common, expected_implementation_identity=expected_native_mask_identity,
                )
                stage_status = NativePreprocessingStageStatus(str(outcome.status))
        except Exception as error:
            if isinstance(error, (NativeMaskUnavailableError, NativeThresholdUnavailableError)):
                status = NativePreprocessingStageStatus.UNAVAILABLE
            else:
                status = (
                    NativePreprocessingStageStatus.CANCELLED if _cancelled(cancel)
                    else NativePreprocessingStageStatus.FAILED
                )
            results.append(_stage_result(
                stage, selected_stage_ids[index], bridge_recipe, status,
                f"Native preprocessing bridge failed: {type(error).__name__}: {error}",
                upstream_sha256=current_digest, bindings=bindings, roles=roles,
            ))
            terminal = status
            break

        record = outcome.run_record
        reference = outcome.checkpoint_reference
        try:
            validate_stage_run(
                bridge_recipe, record,
                expected_upstream_checkpoint_sha256=current_digest,
            )
            if record.stage_id != stage.stage_id or record.recipe_identity != bridge_recipe.identity:
                raise NativePreprocessingError("Bridge run record belongs to a different projected stage recipe.")
            if stage_status in {NativePreprocessingStageStatus.EXECUTED, NativePreprocessingStageStatus.EMPTY}:
                if reference is None:
                    raise NativePreprocessingError("Completed native stage did not return its immutable checkpoint.")
                if stage_status is NativePreprocessingStageStatus.EMPTY and stage.operation_id == "filter.median_blur":
                    raise NativePreprocessingError("A grayscale median image is never recorded as an empty mask.")
                expected_execution = (
                    ExecutionStatus.EMPTY if stage_status is NativePreprocessingStageStatus.EMPTY
                    else ExecutionStatus.AVAILABLE
                )
                if reference.execution.status is not expected_execution:
                    raise NativePreprocessingError("Checkpoint execution status disagrees with the stage's empty/output state.")
                validate_checkpoint_identity(
                    bridge_recipe,
                    stage.stage_id,
                    reference.identity,
                    expected_identity=output_identity,
                    expected_upstream_checkpoint_sha256=current_digest,
                )
                if record.checkpoint_sha256 != reference.checkpoint_sha256:
                    raise NativePreprocessingError("Bridge run record and checkpoint address differ.")
            elif reference is not None:
                raise NativePreprocessingError("Incomplete native stage returned a checkpoint reference.")
        except Exception as error:
            results.append(_stage_result(
                stage, selected_stage_ids[index], bridge_recipe, NativePreprocessingStageStatus.FAILED,
                f"Native stage evidence failed exact contract validation: {type(error).__name__}: {error}",
                upstream_sha256=current_digest, bindings=bindings, roles=roles,
            ))
            terminal = NativePreprocessingStageStatus.FAILED
            break

        results.append(_stage_result(
            stage, selected_stage_ids[index], bridge_recipe, stage_status, record.reason,
            upstream_sha256=current_digest, bindings=bindings, roles=roles,
            reference=reference, record=record,
        ))
        if stage_status not in {NativePreprocessingStageStatus.EXECUTED, NativePreprocessingStageStatus.EMPTY}:
            terminal = stage_status
            break
        assert reference is not None
        current_root = output_root
        current_digest = reference.checkpoint_sha256
        current_identity = reference.identity
        current_measurement = output_measurement
        terminal = stage_status

    if len(results) < len(bridge_stages):
        _not_run_suffix(
            bridge_stages, selected_stage_ids, projected, results, start=len(results), initial_array=initial_array,
        )
    return NativePreprocessingOutcome(
        _overall_status(terminal), recipe.identity,
        expected_native_mask_identity, expected_native_threshold_identity,
        tuple(results),
    )


__all__ = [
    "NativePreprocessingError",
    "NativePreprocessingOutcome",
    "NativePreprocessingStageResult",
    "NativePreprocessingStageStatus",
    "NativePreprocessingStatus",
    "run_native_preprocessing",
]
