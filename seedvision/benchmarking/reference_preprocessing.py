"""Execute the authored PlantCV seed preprocessing chain through pinned bridges.

This benchmark-only coordinator wires the existing one-call reference stages
into the five-stage median/threshold/fill/erode/dilate slice. It does not
interpret the later watershed or measurement stages as an execution graph.
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
    MeasurementContract,
    RasterRole,
    load_stage_input,
)
from seedvision.benchmarking.recipe_catalogue import RecipeCatalogueError, build_recipe_catalogue
from seedvision.benchmarking.recipes import (
    RecipeIdentity,
    RecipeSpec,
    RecipeStage,
    StageRunRecord,
    StageRunState,
    validate_stage_run,
)
from seedvision.benchmarking.reference_mask_stage import (
    ReferenceMaskStageError,
    ReferenceMaskStageOutcome,
    ReferenceMaskStageStatus,
    expected_reference_mask_stage_identity,
    run_reference_mask_stage,
)
from seedvision.benchmarking.reference_runner import PlantCVReferenceRunner, ReferenceCancelledError
from seedvision.benchmarking.reference_stage import (
    ReferenceStageError,
    ReferenceStageOutcome,
    expected_reference_stage_identity,
    run_reference_stage,
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
    "morphology.erode": ("gray", "BinaryMask", "mask", "BinaryMask"),
    "morphology.dilate": ("gray", "BinaryMask", "mask", "BinaryMask"),
}
_IMAGE_ROLE = RasterRole("image", np.dtype(np.uint8).str, "HW", (0, 255))
_MASK_ROLE = RasterRole("mask", np.dtype(np.uint8).str, "HW")
_OUTPUT_ROLE = {
    "GrayImage": _IMAGE_ROLE,
    "BinaryMask": _MASK_ROLE,
}


class ReferencePreprocessingError(ValueError):
    """The selected immutable recipe or prepared-image binding is not runnable."""


class ReferencePreprocessingStatus(StrEnum):
    EXECUTED = "executed"
    EMPTY = "empty"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNAVAILABLE = "unavailable"
    UNSUPPORTED = "unsupported"
    UNRESOLVED = "unresolved"


class ReferencePreprocessingStageStatus(StrEnum):
    EXECUTED = "executed"
    EMPTY = "empty"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNAVAILABLE = "unavailable"
    UNSUPPORTED = "unsupported"
    UNRESOLVED = "unresolved"
    NOT_RUN = "not_run"


@dataclass(frozen=True, slots=True)
class ReferencePreprocessingStageResult:
    """One stage's exact selected contract, bindings, run state, and artifact."""

    stage: RecipeStage
    status: ReferencePreprocessingStageStatus
    run_record: StageRunRecord
    input_checkpoint_sha256: str | None
    input_bindings: Mapping[str, str]
    input_roles: Mapping[str, RasterRole]
    parameters: Mapping[str, Any]
    checkpoint_reference: CheckpointReference | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", ReferencePreprocessingStageStatus(self.status))
        if not isinstance(self.stage, RecipeStage) or not isinstance(self.run_record, StageRunRecord):
            raise TypeError("A preprocessing result requires its immutable stage and run record.")
        if self.run_record.stage_id != self.stage.stage_id:
            raise ValueError("Preprocessing result stage and run record differ.")
        if self.status is ReferencePreprocessingStageStatus.NOT_RUN:
            if self.run_record.state is not StageRunState.NOT_RUN or self.checkpoint_reference is not None:
                raise ValueError("A not-run stage cannot carry execution evidence.")
        else:
            expected_state = {
                ReferencePreprocessingStageStatus.EXECUTED: StageRunState.EXECUTED,
                ReferencePreprocessingStageStatus.EMPTY: StageRunState.EMPTY,
                ReferencePreprocessingStageStatus.FAILED: StageRunState.FAILED,
                ReferencePreprocessingStageStatus.CANCELLED: StageRunState.FAILED,
                ReferencePreprocessingStageStatus.UNAVAILABLE: StageRunState.UNAVAILABLE,
                ReferencePreprocessingStageStatus.UNSUPPORTED: StageRunState.UNSUPPORTED,
                ReferencePreprocessingStageStatus.UNRESOLVED: StageRunState.UNRESOLVED,
            }[self.status]
            if self.run_record.state is not expected_state:
                raise ValueError("Preprocessing status and immutable stage run state differ.")
            successful = self.status in {
                ReferencePreprocessingStageStatus.EXECUTED,
                ReferencePreprocessingStageStatus.EMPTY,
            }
            if successful != (self.checkpoint_reference is not None):
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
class ReferencePreprocessingOutcome:
    """The five-stage chain result, including stages stopped before execution."""

    status: ReferencePreprocessingStatus
    recipe_identity: RecipeIdentity
    stages: tuple[ReferencePreprocessingStageResult, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", ReferencePreprocessingStatus(self.status))
        stages = tuple(self.stages)
        if len(stages) != 5 or any(not isinstance(item, ReferencePreprocessingStageResult) for item in stages):
            raise ValueError("A preprocessing outcome must retain all five chain-stage records.")
        if any(item.run_record.recipe_identity != self.recipe_identity for item in stages):
            raise ValueError("Every preprocessing stage must carry this exact recipe identity.")
        object.__setattr__(self, "stages", stages)

    @property
    def run_records(self) -> tuple[StageRunRecord, ...]:
        return tuple(item.run_record for item in self.stages)

    @property
    def final_checkpoint(self) -> CheckpointReference | None:
        final_stage = self.stages[-1]
        if final_stage.status not in {
            ReferencePreprocessingStageStatus.EXECUTED,
            ReferencePreprocessingStageStatus.EMPTY,
        }:
            return None
        return final_stage.checkpoint_reference

    @property
    def latest_checkpoint(self) -> CheckpointReference | None:
        """Return the last published intermediate, even when the chain stopped."""
        return next(
            (item.checkpoint_reference for item in reversed(self.stages) if item.checkpoint_reference is not None),
            None,
        )


def _catalogue_recipe(recipe_id: str) -> RecipeSpec:
    try:
        return build_recipe_catalogue().recipe(recipe_id)
    except (RecipeCatalogueError, OSError, ValueError, KeyError) as error:
        raise ReferencePreprocessingError("Committed PlantCV seed recipe pins are unavailable.") from error


def _chain_stages(recipe: RecipeSpec) -> tuple[RecipeStage, ...]:
    if recipe.recipe_id not in _THRESHOLD_RECIPES:
        raise ReferencePreprocessingError("Only the three authored pinned PlantCV seed recipes are supported.")
    by_operation: dict[str, list[RecipeStage]] = {}
    for stage in recipe.stages:
        by_operation.setdefault(stage.operation_id, []).append(stage)
    threshold_operation = _THRESHOLD_RECIPES[recipe.recipe_id]
    operations = (
        "filter.median_blur", threshold_operation, "morphology.fill",
        "morphology.erode", "morphology.dilate",
    )
    selected: list[RecipeStage] = []
    for operation in operations:
        matches = by_operation.get(operation, ())
        if len(matches) != 1:
            raise ReferencePreprocessingError(
                f"Recipe must contain exactly one {operation!r} preprocessing stage."
            )
        selected.append(matches[0])
    return tuple(selected)


def _validate_pinned_recipe(recipe: RecipeSpec) -> None:
    canonical = _catalogue_recipe(recipe.recipe_id)
    if len(recipe.stages) != len(canonical.stages):
        raise ReferencePreprocessingError("Recipe stage count differs from the pinned authored seed recipe.")
    normalized_stages = []
    for current, pinned in zip(recipe.stages, canonical.stages, strict=True):
        if current.stage_id != pinned.stage_id:
            raise ReferencePreprocessingError("Recipe stage order or IDs differ from the pinned seed recipe.")
        if len(current.parameters) != len(pinned.parameters):
            raise ReferencePreprocessingError("Parameter declarations differ from the pinned authored recipe.")
        parameters = []
        for actual, baseline in zip(current.parameters, pinned.parameters, strict=True):
            if actual.parameter_id != baseline.parameter_id:
                raise ReferencePreprocessingError("Parameter IDs differ from the pinned authored recipe.")
            if actual.affects_calculation:
                parameters.append(replace(actual, value=baseline.value))
            else:
                if actual != baseline:
                    raise ReferencePreprocessingError("Non-calculation controls differ from the pinned authored recipe.")
                parameters.append(actual)
        normalized_stages.append(replace(current, parameters=tuple(parameters)))
    # Canvas layout is display-only. Compare its canonical form while keeping
    # the caller's full layout identity in every result and checkpoint.
    normalized = replace(recipe, stages=tuple(normalized_stages), layout=canonical.layout)
    if normalized != canonical:
        raise ReferencePreprocessingError(
            "Recipe provenance, source pins, ports, dependencies, or non-preprocessing contracts differ from the pinned seed recipe."
        )


def _validate_actual_chain(recipe: RecipeSpec, stages: tuple[RecipeStage, ...]) -> None:
    stage_by_id = {stage.stage_id: stage for stage in recipe.stages}
    for index, stage in enumerate(stages):
        operation = stage.operation_id
        input_name, input_type, output_name, output_type = _INPUT_OUTPUT_CONTRACTS[operation]
        inputs = [port for port in stage.ports if port.direction == "input"]
        outputs = [port for port in stage.ports if port.direction == "output"]
        if len(inputs) != 1 or len(outputs) != 1:
            raise ReferencePreprocessingError("Every preprocessing operation must have exactly one input and output port.")
        if (inputs[0].name, inputs[0].data_type, outputs[0].name, outputs[0].data_type) != (
            input_name, input_type, output_name, output_type,
        ):
            raise ReferencePreprocessingError(f"Pinned ports differ on preprocessing stage {stage.stage_id!r}.")
        incoming = [edge for edge in recipe.dependencies if edge.target_stage == stage.stage_id]
        if len(incoming) != 1:
            raise ReferencePreprocessingError(f"Stage {stage.stage_id!r} must have one actual graph input edge.")
        edge = incoming[0]
        if edge.target_port != input_name or edge.data_type != input_type:
            raise ReferencePreprocessingError(f"Input edge does not match the declared port on {stage.stage_id!r}.")
        if index == 0:
            source = stage_by_id.get(edge.source_stage)
            if (
                source is None
                or source.operation_id != "external.image_inputs"
                or edge.source_port != "gray_image"
                or edge.data_type != "GrayImage"
            ):
                raise ReferencePreprocessingError("Median input is not connected to the recipe's external grayscale port.")
            continue
        upstream = stages[index - 1]
        previous_output = _INPUT_OUTPUT_CONTRACTS[upstream.operation_id][2:]
        if (
            edge.source_stage != upstream.stage_id
            or edge.source_port != previous_output[0]
            or edge.data_type != previous_output[1]
        ):
            raise ReferencePreprocessingError(
                f"Graph edge into {stage.stage_id!r} does not come from the preceding preprocessing output."
            )


def _cancelled(cancel: Any) -> bool:
    if cancel is None:
        return False
    if callable(cancel):
        return bool(cancel())
    is_set = getattr(cancel, "is_set", None)
    if not callable(is_set):
        raise ReferencePreprocessingError("cancel must be a callable or an event with is_set().")
    return bool(is_set())


def _failed_or_not_run(
    recipe: RecipeSpec,
    stage: RecipeStage,
    status: ReferencePreprocessingStageStatus,
    reason: str,
    *,
    upstream_sha256: str | None = None,
    input_bindings: Mapping[str, str] | None = None,
    input_roles: Mapping[str, RasterRole] | None = None,
) -> ReferencePreprocessingStageResult:
    state = {
        ReferencePreprocessingStageStatus.NOT_RUN: StageRunState.NOT_RUN,
        ReferencePreprocessingStageStatus.FAILED: StageRunState.FAILED,
        ReferencePreprocessingStageStatus.CANCELLED: StageRunState.FAILED,
        ReferencePreprocessingStageStatus.UNAVAILABLE: StageRunState.UNAVAILABLE,
        ReferencePreprocessingStageStatus.UNSUPPORTED: StageRunState.UNSUPPORTED,
        ReferencePreprocessingStageStatus.UNRESOLVED: StageRunState.UNRESOLVED,
    }[status]
    record = StageRunRecord(
        stage_id=stage.stage_id,
        recipe_identity=recipe.identity,
        state=state,
        upstream_checkpoint_sha256=upstream_sha256,
        reason=" ".join(str(reason).split())[:512],
    )
    validate_stage_run(recipe, record)
    return ReferencePreprocessingStageResult(
        stage=stage,
        status=status,
        run_record=record,
        input_checkpoint_sha256=upstream_sha256,
        input_bindings=input_bindings or {},
        input_roles=input_roles or {},
        parameters=recipe.parameters_for_runner(stage.stage_id),
    )


def _status_from_run_record(
    record: StageRunRecord,
    *,
    cancel: Any,
) -> ReferencePreprocessingStageStatus:
    if record.state is StageRunState.EXECUTED:
        return ReferencePreprocessingStageStatus.EXECUTED
    if record.state is StageRunState.EMPTY:
        return ReferencePreprocessingStageStatus.EMPTY
    if record.state is StageRunState.UNAVAILABLE:
        return ReferencePreprocessingStageStatus.UNAVAILABLE
    if record.state is StageRunState.UNRESOLVED:
        return ReferencePreprocessingStageStatus.UNRESOLVED
    if record.state is StageRunState.UNSUPPORTED:
        return ReferencePreprocessingStageStatus.UNSUPPORTED
    if record.state is StageRunState.FAILED and (
        _cancelled(cancel)
        or record.reason.lower().startswith("cancelled ")
        or "ReferenceCancelledError:" in record.reason
    ):
        return ReferencePreprocessingStageStatus.CANCELLED
    return ReferencePreprocessingStageStatus.FAILED


def _stage_output_identity(
    identity: CheckpointIdentity,
    recipe: RecipeSpec,
    stage: RecipeStage,
    upstream_sha256: str,
) -> CheckpointIdentity:
    if stage.operation_id.startswith("threshold."):
        return expected_reference_stage_identity(identity, recipe, stage.stage_id, upstream_sha256)
    return expected_reference_mask_stage_identity(identity, recipe, stage.stage_id, upstream_sha256)


def _preflight(
    recipe: RecipeSpec,
    stages: tuple[RecipeStage, ...],
    input_identity: CheckpointIdentity,
) -> tuple[RecipeStage, str] | None:
    identity = input_identity
    placeholder = "0" * 64
    for stage in stages:
        try:
            identity = _stage_output_identity(identity, recipe, stage, placeholder)
        except (ReferenceMaskStageError, ReferenceStageError, ValueError) as error:
            return stage, " ".join(str(error).split())[:512]
    return None


def run_reference_preprocessing(
    input_root: Path | str,
    input_checkpoint_sha256: str,
    recipe: RecipeSpec,
    runner: PlantCVReferenceRunner,
    output_root: Path | str,
    *,
    expected_source_sha256: str,
    expected_coordinate_frame: CoordinateFrame,
    expected_recipe_identity: RecipeIdentity,
    expected_input_identity: CheckpointIdentity,
    expected_input_measurement: MeasurementContract,
    output_measurement: MeasurementContract,
    input_bindings: Mapping[str, str],
    input_roles: Mapping[str, RasterRole],
    allow_cpu_transfer: bool,
    limits: CheckpointLimits = CheckpointLimits(),
    timeout_seconds: float | None = None,
    cancel: Any = None,
) -> ReferencePreprocessingOutcome:
    """Run the five authored preprocessing stages over exact checkpoint inputs.

    The selected parameters live in the immutable recipe. The prepared grayscale
    checkpoint's input port, array name, and role are supplied explicitly; each
    later binding follows the declared recipe edge and bridge output role.
    """

    if not isinstance(recipe, RecipeSpec) or not isinstance(expected_recipe_identity, RecipeIdentity):
        raise TypeError("recipe and expected_recipe_identity must be immutable recipe contracts.")
    if type(runner) is not PlantCVReferenceRunner:
        raise TypeError("runner must be an explicitly supplied PlantCVReferenceRunner instance.")
    if not isinstance(expected_input_identity, CheckpointIdentity):
        raise TypeError("expected_input_identity must be a CheckpointIdentity.")
    if not isinstance(expected_coordinate_frame, CoordinateFrame):
        raise TypeError("expected_coordinate_frame must be a CoordinateFrame.")
    if not isinstance(expected_input_measurement, MeasurementContract) or not isinstance(output_measurement, MeasurementContract):
        raise TypeError("Explicit input and output measurement contracts are required.")
    if output_measurement.eligible:
        raise ReferencePreprocessingError("Original PlantCV preprocessing cannot certify scientific measurement eligibility.")
    if type(allow_cpu_transfer) is not bool or not allow_cpu_transfer:
        raise ReferencePreprocessingError("Benchmark checkpoint serialization requires explicit allow_cpu_transfer=True.")
    if recipe.identity != expected_recipe_identity:
        raise ReferencePreprocessingError("Recipe identity differs from the caller's exact expected recipe identity.")
    if not isinstance(expected_source_sha256, str) or not _SHA256.fullmatch(expected_source_sha256):
        raise ReferencePreprocessingError("expected_source_sha256 must be a lowercase SHA-256 digest.")
    if not isinstance(input_checkpoint_sha256, str) or not _SHA256.fullmatch(input_checkpoint_sha256):
        raise ReferencePreprocessingError("input_checkpoint_sha256 must be a lowercase SHA-256 digest.")
    if not isinstance(input_bindings, Mapping) or not isinstance(input_roles, Mapping):
        raise ReferencePreprocessingError("Prepared-image bindings and raster roles must be explicit mappings.")
    if any(not isinstance(key, str) or not isinstance(value, str) for key, value in input_bindings.items()):
        raise ReferencePreprocessingError("Input bindings must map recipe ports to checkpoint array names.")
    if any(not isinstance(key, str) or not isinstance(value, RasterRole) for key, value in input_roles.items()):
        raise ReferencePreprocessingError("Input roles must map recipe ports to RasterRole contracts.")
    if not isinstance(limits, CheckpointLimits):
        raise TypeError("limits must be an explicit CheckpointLimits value.")

    stages = _chain_stages(recipe)
    _validate_pinned_recipe(recipe)
    _validate_actual_chain(recipe, stages)
    if expected_input_identity.source_sha256 != expected_source_sha256:
        raise ReferencePreprocessingError("Prepared checkpoint identity has a different source digest.")
    if expected_input_identity.coordinate_frame != expected_coordinate_frame:
        raise ReferencePreprocessingError("Prepared checkpoint identity has a different coordinate frame.")
    try:
        checkpoint = load_stage_input(
            input_root,
            input_checkpoint_sha256,
            expected_identity=expected_input_identity,
            expected_measurement=expected_input_measurement,
            limits=limits,
        )
    except Exception as error:
        raise ReferencePreprocessingError(f"Prepared grayscale checkpoint is invalid: {error}") from error
    if checkpoint.identity.source_sha256 != expected_source_sha256:
        raise ReferencePreprocessingError("Loaded checkpoint source digest differs from the requested source.")
    if checkpoint.identity.coordinate_frame != expected_coordinate_frame:
        raise ReferencePreprocessingError("Loaded checkpoint frame differs from the requested coordinate frame.")
    if set(input_bindings) != {"gray"} or set(input_roles) != {"gray"}:
        raise ReferencePreprocessingError("The actual median graph edge requires one explicit 'gray' image binding and role.")
    array_name = input_bindings["gray"]
    role = input_roles["gray"]
    array = checkpoint.arrays.get(array_name)
    if (
        role != _IMAGE_ROLE
        or checkpoint.raster_roles.get(array_name) != role
        or not isinstance(array, np.ndarray)
        or array.dtype != np.dtype(np.uint8)
        or array.ndim != 2
        or tuple(array.shape) != expected_coordinate_frame.shape
        or not array.flags.c_contiguous
    ):
        raise ReferencePreprocessingError("Prepared grayscale binding must be the declared contiguous uint8 HxW image role.")

    preflight_failure = _preflight(recipe, stages, checkpoint.identity)
    if preflight_failure is not None:
        unsupported_stage, reason = preflight_failure
        results = []
        for stage in stages:
            if stage.stage_id == unsupported_stage.stage_id:
                results.append(_failed_or_not_run(
                    recipe,
                    stage,
                    ReferencePreprocessingStageStatus.UNSUPPORTED,
                    reason,
                ))
                break
            results.append(_failed_or_not_run(
                recipe,
                stage,
                ReferencePreprocessingStageStatus.NOT_RUN,
                f"Not run because whole-chain preflight stopped at {unsupported_stage.stage_id}: {reason}",
            ))
        next_index = len(results)
        for stage in stages[next_index:]:
            results.append(_failed_or_not_run(
                recipe,
                stage,
                ReferencePreprocessingStageStatus.NOT_RUN,
                f"Not run because whole-chain preflight stopped at {unsupported_stage.stage_id}.",
            ))
        return ReferencePreprocessingOutcome(
            ReferencePreprocessingStatus.UNSUPPORTED,
            recipe.identity,
            tuple(results),
        )

    results: list[ReferencePreprocessingStageResult] = []
    current_root: Path | str = input_root
    current_digest = input_checkpoint_sha256
    current_identity = checkpoint.identity
    current_measurement = expected_input_measurement
    current_array_name = array_name
    current_role = role

    for index, stage in enumerate(stages):
        input_port = _INPUT_OUTPUT_CONTRACTS[stage.operation_id][0]
        bindings = {input_port: current_array_name}
        roles = {input_port: current_role}
        output_identity = _stage_output_identity(current_identity, recipe, stage, current_digest)
        common = {
            "input_root": current_root,
            "input_checkpoint_sha256": current_digest,
            "recipe": recipe,
            "stage_id": stage.stage_id,
            "runner": runner,
            "output_root": output_root,
            "expected_source_sha256": expected_source_sha256,
            "expected_coordinate_frame": expected_coordinate_frame,
            "expected_recipe_identity": expected_recipe_identity,
            "expected_upstream_checkpoint_sha256": current_digest,
            "expected_input_identity": current_identity,
            "expected_input_measurement": current_measurement,
            "expected_output_identity": output_identity,
            "input_bindings": bindings,
            "input_roles": roles,
            "output_measurement": output_measurement,
            "allow_cpu_transfer": allow_cpu_transfer,
            "limits": limits,
            "timeout_seconds": timeout_seconds,
            "cancel": cancel,
        }
        try:
            if stage.operation_id.startswith("threshold."):
                outcome: ReferenceMaskStageOutcome | ReferenceStageOutcome = run_reference_stage(**common)
            else:
                outcome = run_reference_mask_stage(**common)
        except Exception as error:
            cancelled = _cancelled(cancel)
            status = (
                ReferencePreprocessingStageStatus.CANCELLED
                if cancelled or isinstance(error, ReferenceCancelledError)
                else ReferencePreprocessingStageStatus.FAILED
            )
            failed = _failed_or_not_run(
                recipe,
                stage,
                status,
                f"Preprocessing bridge failed: {type(error).__name__}: {error}",
                upstream_sha256=current_digest,
                input_bindings=bindings,
                input_roles=roles,
            )
            results.append(failed)
            terminal = status
            break

        if isinstance(outcome, ReferenceMaskStageOutcome):
            status_map = {
                ReferenceMaskStageStatus.EXECUTED: ReferencePreprocessingStageStatus.EXECUTED,
                ReferenceMaskStageStatus.EMPTY: ReferencePreprocessingStageStatus.EMPTY,
                ReferenceMaskStageStatus.FAILED: ReferencePreprocessingStageStatus.FAILED,
                ReferenceMaskStageStatus.CANCELLED: ReferencePreprocessingStageStatus.CANCELLED,
                ReferenceMaskStageStatus.UNAVAILABLE: ReferencePreprocessingStageStatus.UNAVAILABLE,
                ReferenceMaskStageStatus.UNSUPPORTED: ReferencePreprocessingStageStatus.UNSUPPORTED,
                ReferenceMaskStageStatus.UNRESOLVED: ReferencePreprocessingStageStatus.UNRESOLVED,
            }
            stage_status = status_map[outcome.status]
            run_record = outcome.run_record
            reference = outcome.checkpoint_reference
        else:
            stage_status = _status_from_run_record(outcome.run_record, cancel=cancel)
            run_record = outcome.run_record
            reference = outcome.checkpoint_reference
        result = ReferencePreprocessingStageResult(
            stage=stage,
            status=stage_status,
            run_record=run_record,
            input_checkpoint_sha256=current_digest,
            input_bindings=bindings,
            input_roles=roles,
            parameters=recipe.parameters_for_runner(stage.stage_id),
            checkpoint_reference=reference,
        )
        results.append(result)
        if stage_status not in {
            ReferencePreprocessingStageStatus.EXECUTED,
            ReferencePreprocessingStageStatus.EMPTY,
        }:
            terminal = (
                ReferencePreprocessingStageStatus.CANCELLED
                if stage_status is ReferencePreprocessingStageStatus.CANCELLED
                else stage_status
            )
            break
        assert reference is not None
        if reference.identity != output_identity:
            failed = _failed_or_not_run(
                recipe,
                stage,
                ReferencePreprocessingStageStatus.FAILED,
                "Bridge returned a checkpoint with an unexpected stage identity.",
                upstream_sha256=current_digest,
                input_bindings=bindings,
                input_roles=roles,
            )
            results[-1] = failed
            terminal = ReferencePreprocessingStageStatus.FAILED
            break
        if index < len(stages) - 1:
            output_type = _INPUT_OUTPUT_CONTRACTS[stage.operation_id][3]
            current_root = output_root
            current_digest = reference.checkpoint_sha256
            current_identity = reference.identity
            current_measurement = output_measurement
            current_array_name = _INPUT_OUTPUT_CONTRACTS[stage.operation_id][2]
            current_role = _OUTPUT_ROLE[output_type]
        terminal = stage_status

    for stage in stages[len(results):]:
        results.append(_failed_or_not_run(
            recipe,
            stage,
            ReferencePreprocessingStageStatus.NOT_RUN,
            f"Not run because preprocessing stopped at {results[-1].stage.stage_id} ({results[-1].status.value}).",
        ))
    aggregate = {
        ReferencePreprocessingStageStatus.EXECUTED: ReferencePreprocessingStatus.EXECUTED,
        ReferencePreprocessingStageStatus.EMPTY: ReferencePreprocessingStatus.EMPTY,
        ReferencePreprocessingStageStatus.FAILED: ReferencePreprocessingStatus.FAILED,
        ReferencePreprocessingStageStatus.CANCELLED: ReferencePreprocessingStatus.CANCELLED,
        ReferencePreprocessingStageStatus.UNAVAILABLE: ReferencePreprocessingStatus.UNAVAILABLE,
        ReferencePreprocessingStageStatus.UNSUPPORTED: ReferencePreprocessingStatus.UNSUPPORTED,
        ReferencePreprocessingStageStatus.UNRESOLVED: ReferencePreprocessingStatus.UNRESOLVED,
    }[terminal]
    return ReferencePreprocessingOutcome(aggregate, recipe.identity, tuple(results))


__all__ = [
    "ReferencePreprocessingError",
    "ReferencePreprocessingOutcome",
    "ReferencePreprocessingStageResult",
    "ReferencePreprocessingStageStatus",
    "ReferencePreprocessingStatus",
    "run_reference_preprocessing",
]
