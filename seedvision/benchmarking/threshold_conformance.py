"""Execute and retain one exact PlantCV/CUDA threshold-stage pair.

This is a bounded comparison orchestrator for binary, adaptive-mean, and Otsu
threshold stages. It delegates each execution to the established one-stage
bridge, then compares only verified immutable output checkpoints. Agreement is
conformance evidence, not scientific validation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping

import numpy as np

from seedvision.benchmarking.checkpoints import (
    BenchmarkCheckpoint,
    CheckpointError,
    CheckpointIdentity,
    CheckpointLimits,
    CheckpointReference,
    CoordinateFrame,
    ExecutionStatus,
    MeasurementContract,
    RasterRole,
    load_checkpoint,
    load_stage_input,
)
from seedvision.benchmarking.conformance import (
    MAX_CONFORMANCE_REPORT_BYTES,
    CheckpointPairContract,
    ConformanceError,
    ThresholdBorderCorrespondence,
    ThresholdMaskSemanticCorrespondence,
    compare_masks,
)
from seedvision.benchmarking.native_threshold_stage import (
    NativeThresholdIdentity,
    NativeThresholdOutcome,
    NativeThresholdStageError,
    expected_native_threshold_stage_identity,
    native_threshold_identity,
    run_native_threshold_stage,
)
from seedvision.benchmarking.recipe_catalogue import build_recipe_catalogue
from seedvision.benchmarking.recipes import (
    RecipeIdentity,
    RecipeProvenance,
    RecipeProvenanceMode,
    RecipeSpec,
    StageRunRecord,
    StageRunState,
    validate_stage_run,
)
from seedvision.benchmarking.reference_runner import PlantCVReferenceRunner
from seedvision.benchmarking.reference_stage import (
    ReferenceStageOutcome,
    ReferenceStageError,
    expected_reference_stage_identity,
    run_reference_stage,
)


THRESHOLD_PAIR_REPORT_FORMAT = "seedfiddle.threshold-stage-pair-report"
THRESHOLD_PAIR_REPORT_VERSION = 1
MAX_THRESHOLD_PAIR_REPORT_BYTES = MAX_CONFORMANCE_REPORT_BYTES
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_THRESHOLD_OPERATIONS = frozenset({"threshold.binary", "threshold.mean", "threshold.otsu"})
_OUTPUT_MASK_ROLE = RasterRole("mask", np.dtype(np.uint8).str)
_GRAY_IMAGE_ROLE = RasterRole("image", np.dtype(np.uint8).str, "HW", (0, 255))


class ThresholdPairError(ValueError):
    """The requested two-backend stage pair is stale or outside its contract."""


@dataclass(frozen=True, slots=True)
class ThresholdConformanceOutcome:
    """Immutable report address plus the two actual stage outcomes."""

    report_path: Path
    report_sha256: str
    report_json: str
    reference_outcome: ReferenceStageOutcome
    native_outcome: NativeThresholdOutcome

    @property
    def report(self) -> dict[str, Any]:
        """Return a fresh decoded report so callers cannot mutate retained data."""
        return json.loads(self.report_json)


def _json_bytes(value: Any, *, maximum: int = MAX_THRESHOLD_PAIR_REPORT_BYTES) -> bytes:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError, RecursionError) as error:
        raise ThresholdPairError("Threshold pair report is not strict finite JSON.") from error
    if len(encoded) > maximum:
        raise ThresholdPairError(f"Threshold pair report exceeds {maximum} bytes.")
    return encoded


def _same_json_types(left: Any, right: Any) -> bool:
    if isinstance(left, Mapping):
        return isinstance(right, Mapping) and set(left) == set(right) and all(
            _same_json_types(left[key], right[key]) for key in left
        )
    if isinstance(left, (tuple, list)):
        return isinstance(right, (tuple, list)) and len(left) == len(right) and all(
            _same_json_types(a, b) for a, b in zip(left, right)
        )
    return type(left) is type(right) and left == right


def _cancel_contract(cancel: Any) -> None:
    if cancel is None or callable(cancel):
        return
    if not callable(getattr(cancel, "is_set", None)):
        raise ThresholdPairError("cancel must be a callable or an event with is_set().")


def _reference_timeout(runner: PlantCVReferenceRunner, timeout_seconds: float | None) -> float:
    value = runner.timeout_seconds if timeout_seconds is None else timeout_seconds
    if isinstance(value, bool) or type(value) not in (int, float):
        raise ThresholdPairError("timeout_seconds must be finite and in (0,3600].")
    try:
        value = float(value)
    except (OverflowError, ValueError):
        raise ThresholdPairError("timeout_seconds must be finite and in (0,3600].") from None
    if not math.isfinite(value) or not 0 < value <= 3600:
        raise ThresholdPairError("timeout_seconds must be finite and in (0,3600].")
    return value


def _stage_parameters(recipe: RecipeSpec, stage_id: str) -> dict[str, Any]:
    return recipe.parameters_for_runner(stage_id)


def _prepare_report_directory(artifact_store: Path | str) -> Path:
    try:
        root = Path(artifact_store)
        if root.exists() and (root.is_symlink() or not root.is_dir()):
            raise ThresholdPairError("Explicit artifact store must be a regular directory, not a file or symlink.")
        root.mkdir(parents=True, exist_ok=True)
        root = root.resolve(strict=True)
        report_dir = root / "threshold-conformance-v1"
        if report_dir.exists() and (report_dir.is_symlink() or not report_dir.is_dir()):
            raise ThresholdPairError("Threshold report directory must be a regular directory, not a symlink.")
        report_dir.mkdir(exist_ok=True)
        return report_dir.resolve(strict=True)
    except ThresholdPairError:
        raise
    except OSError as error:
        raise ThresholdPairError(f"Cannot prepare the explicit threshold report store: {error}") from error


def _publish_report(report_directory: Path, payload: Mapping[str, Any]) -> tuple[Path, str, str]:
    encoded = _json_bytes(payload)
    digest = hashlib.sha256(encoded).hexdigest()
    destination = report_directory / f"{digest}.json"
    if destination.is_symlink():
        raise ThresholdPairError("Content-addressed threshold report path cannot be a symlink.")
    if destination.exists():
        try:
            if not destination.is_file() or destination.stat().st_size > MAX_THRESHOLD_PAIR_REPORT_BYTES:
                raise ThresholdPairError("Existing report is not a bounded regular file.")
            with destination.open("rb") as stream:
                existing = stream.read(MAX_THRESHOLD_PAIR_REPORT_BYTES + 1)
            if destination.is_symlink() or existing != encoded:
                raise ThresholdPairError("Existing content-addressed report differs from its expected bytes.")
        except OSError as error:
            raise ThresholdPairError(f"Cannot verify an existing threshold report: {error}") from error
        return destination, digest, encoded.decode("utf-8")

    descriptor = -1
    temporary_path: Path | None = None
    try:
        descriptor, name = tempfile.mkstemp(prefix=".threshold-report-", suffix=".tmp", dir=report_directory)
        temporary_path = Path(name)
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary_path, destination)
        except FileExistsError:
            if destination.is_symlink() or not destination.is_file() or destination.stat().st_size > MAX_THRESHOLD_PAIR_REPORT_BYTES:
                raise ThresholdPairError("Concurrent report publication found an invalid existing destination.")
            with destination.open("rb") as stream:
                existing = stream.read(MAX_THRESHOLD_PAIR_REPORT_BYTES + 1)
            if destination.is_symlink() or existing != encoded:
                raise ThresholdPairError("Concurrent report publication found different content at the same address.")
    except ThresholdPairError:
        raise
    except OSError as error:
        raise ThresholdPairError(f"Atomic no-overwrite report publication failed: {error}") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
    return destination, digest, encoded.decode("utf-8")


def _exception_reason(error: BaseException) -> str:
    value = " ".join(f"{type(error).__name__}: {error}".split())
    return value[:512] or type(error).__name__


def _failed_reference(
    recipe: RecipeSpec,
    stage_id: str,
    upstream_sha256: str,
    error: BaseException,
) -> ReferenceStageOutcome:
    record = StageRunRecord(
        stage_id=stage_id,
        recipe_identity=recipe.identity,
        state=StageRunState.FAILED,
        upstream_checkpoint_sha256=upstream_sha256,
        reason=_exception_reason(error),
    )
    validate_stage_run(recipe, record, expected_upstream_checkpoint_sha256=upstream_sha256)
    return ReferenceStageOutcome(record, None)


def _failed_native(
    recipe: RecipeSpec,
    stage_id: str,
    upstream_sha256: str,
    error: BaseException,
) -> NativeThresholdOutcome:
    record = StageRunRecord(
        stage_id=stage_id,
        recipe_identity=recipe.identity,
        state=StageRunState.FAILED,
        upstream_checkpoint_sha256=upstream_sha256,
        reason=_exception_reason(error),
    )
    validate_stage_run(recipe, record, expected_upstream_checkpoint_sha256=upstream_sha256)
    return NativeThresholdOutcome(record, None)


def _stage_evidence(
    outcome: ReferenceStageOutcome | NativeThresholdOutcome,
    *,
    output_root: Path | str,
    recipe: RecipeSpec,
    stage_id: str,
    upstream_sha256: str,
    expected_identity: CheckpointIdentity,
    output_measurement: MeasurementContract,
    limits: CheckpointLimits,
) -> tuple[dict[str, Any], BenchmarkCheckpoint | None]:
    record = outcome.run_record
    evidence: dict[str, Any] = {
        "run_record": record.to_dict(),
        "expected_checkpoint_identity": expected_identity.to_dict(),
        "actual_checkpoint_sha256": record.checkpoint_sha256,
        "checkpoint_validation": "not_applicable",
        "checkpoint_validation_reason": None,
    }
    try:
        validate_stage_run(recipe, record, expected_upstream_checkpoint_sha256=upstream_sha256)
    except Exception as error:
        evidence["checkpoint_validation"] = "rejected"
        evidence["checkpoint_validation_reason"] = _exception_reason(error)
        return evidence, None
    reference = outcome.checkpoint_reference
    if record.state not in {StageRunState.EXECUTED, StageRunState.EMPTY}:
        if reference is not None:
            evidence["checkpoint_validation"] = "rejected"
            evidence["checkpoint_validation_reason"] = "Non-success stage state carries a checkpoint reference."
        return evidence, None
    if reference is None or record.checkpoint_sha256 is None:
        evidence["checkpoint_validation"] = "rejected"
        evidence["checkpoint_validation_reason"] = "Successful stage state omitted its checkpoint reference."
        return evidence, None
    try:
        if reference.checkpoint_sha256 != record.checkpoint_sha256 or reference.identity != expected_identity:
            raise CheckpointError("Run record or reference differs from the exact expected output identity.")
        checkpoint = load_checkpoint(
            output_root,
            record.checkpoint_sha256,
            expected_identity=expected_identity,
            expected_measurement=output_measurement,
            limits=limits,
        )
        expected_execution_status = ExecutionStatus.EMPTY if record.state is StageRunState.EMPTY else ExecutionStatus.AVAILABLE
        if checkpoint.execution.status is not expected_execution_status:
            raise CheckpointError("Checkpoint execution status differs from its exact StageRunRecord state.")
        if set(checkpoint.arrays) != {"mask"} or set(checkpoint.raster_roles) != {"mask"}:
            raise CheckpointError("Threshold checkpoint output inventory is not exactly one mask raster.")
        if checkpoint.raster_roles["mask"] != _OUTPUT_MASK_ROLE:
            raise CheckpointError("Threshold checkpoint mask role differs from the exact uint8 HW contract.")
        evidence["checkpoint_validation"] = "verified"
        evidence["actual_checkpoint_identity"] = checkpoint.identity.to_dict()
        evidence["execution_status"] = checkpoint.execution.status.value
        return evidence, checkpoint
    except Exception as error:
        evidence["checkpoint_validation"] = "rejected"
        evidence["checkpoint_validation_reason"] = _exception_reason(error)
        return evidence, None


def _report_stage(outcome: ReferenceStageOutcome | NativeThresholdOutcome, evidence: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "state": outcome.run_record.state.value,
        "evidence": dict(evidence),
    }


def run_threshold_conformance_pair(
    input_root: Path | str,
    input_checkpoint_sha256: str,
    output_root: Path | str,
    artifact_store: Path | str,
    reference_recipe: RecipeSpec,
    native_recipe: RecipeSpec,
    reference_stage_id: str,
    native_stage_id: str,
    runner: PlantCVReferenceRunner,
    *,
    expected_source_sha256: str,
    expected_coordinate_frame: CoordinateFrame,
    expected_input_identity: CheckpointIdentity,
    expected_input_measurement: MeasurementContract,
    expected_reference_recipe_identity: RecipeIdentity,
    expected_native_recipe_identity: RecipeIdentity,
    expected_reference_provenance: RecipeProvenance,
    expected_native_implementation_identity: NativeThresholdIdentity,
    expected_reference_output_identity: CheckpointIdentity,
    expected_native_output_identity: CheckpointIdentity,
    input_bindings: Mapping[str, str],
    input_roles: Mapping[str, RasterRole],
    output_measurement: MeasurementContract,
    device: Any,
    allow_cpu_transfer: bool,
    limits: CheckpointLimits = CheckpointLimits(),
    timeout_seconds: float | None = None,
    cancel: Any = None,
) -> ThresholdConformanceOutcome:
    """Run both exact stages from one input checkpoint and retain a bounded report.

    Contract validation and identity derivation complete before either backend
    executes. Each backend is invoked through its existing stage bridge; callers
    cannot supply prebuilt execution results.
    """
    import torch

    if not isinstance(reference_recipe, RecipeSpec) or not isinstance(native_recipe, RecipeSpec):
        raise TypeError("Both exact immutable RecipeSpec values are required.")
    if type(runner) is not PlantCVReferenceRunner:
        raise TypeError("runner must be an explicit PlantCVReferenceRunner instance.")
    if not isinstance(expected_coordinate_frame, CoordinateFrame):
        raise TypeError("expected_coordinate_frame must be a CoordinateFrame.")
    if not isinstance(expected_input_identity, CheckpointIdentity):
        raise TypeError("expected_input_identity must be a CheckpointIdentity.")
    if not isinstance(expected_input_measurement, MeasurementContract):
        raise TypeError("expected_input_measurement must be a MeasurementContract.")
    if not isinstance(expected_reference_recipe_identity, RecipeIdentity) or not isinstance(expected_native_recipe_identity, RecipeIdentity):
        raise TypeError("Exact expected recipe identities are required for both backends.")
    if not isinstance(expected_reference_provenance, RecipeProvenance):
        raise TypeError("expected_reference_provenance must bind the reference backend and environment.")
    if not isinstance(expected_native_implementation_identity, NativeThresholdIdentity):
        raise TypeError("expected_native_implementation_identity must bind native code and environment.")
    if not isinstance(expected_reference_output_identity, CheckpointIdentity) or not isinstance(expected_native_output_identity, CheckpointIdentity):
        raise TypeError("Exact expected output checkpoint identities are required for both backends.")
    if not isinstance(output_measurement, MeasurementContract) or output_measurement.eligible:
        raise ThresholdPairError("Threshold output measurement must be explicit and ineligible.")
    if not isinstance(limits, CheckpointLimits):
        raise TypeError("limits must be a CheckpointLimits contract.")
    if type(allow_cpu_transfer) is not bool or not allow_cpu_transfer:
        raise ThresholdPairError("Checkpoint output archival requires explicit allow_cpu_transfer=True.")
    if type(device) is not torch.device or device.type != "cuda" or device.index is None or device.index < 0:
        raise ThresholdPairError("device must be an explicit torch.device such as cuda:0.")
    if expected_native_implementation_identity.device_index != device.index:
        raise ThresholdPairError("Expected native implementation identity is pinned to a different CUDA device.")
    if type(expected_source_sha256) is not str or not _SHA256.fullmatch(expected_source_sha256):
        raise ThresholdPairError("expected_source_sha256 must be a lowercase SHA-256 digest.")
    if type(input_checkpoint_sha256) is not str or not _SHA256.fullmatch(input_checkpoint_sha256):
        raise ThresholdPairError("input_checkpoint_sha256 must be a lowercase SHA-256 digest.")
    if expected_input_identity.source_sha256 != expected_source_sha256:
        raise ThresholdPairError("Input identity source SHA differs from the caller's expected source.")
    if expected_input_identity.coordinate_frame != expected_coordinate_frame:
        raise ThresholdPairError("Input identity frame differs from the caller's expected coordinate frame.")
    if expected_reference_recipe_identity != reference_recipe.identity or expected_native_recipe_identity != native_recipe.identity:
        raise ThresholdPairError("A recipe differs from its exact caller-expected identity.")
    if reference_recipe.provenance != expected_reference_provenance:
        raise ThresholdPairError("Reference provenance/backend/environment differs from its caller pin.")
    if native_recipe.provenance != expected_native_implementation_identity.recipe_provenance():
        raise ThresholdPairError("Native provenance/backend/environment differs from its exact implementation pin.")
    if reference_recipe.provenance.mode is not RecipeProvenanceMode.ORIGINAL_REFERENCE or native_recipe.provenance.mode is not RecipeProvenanceMode.NATIVE_COMPATIBLE:
        raise ThresholdPairError("The pair requires distinct original-reference and native-compatible provenance.")
    if expected_reference_provenance.backend_id == expected_native_implementation_identity.backend_id:
        raise ThresholdPairError("Reference and native stages must retain distinct backend identities.")
    if expected_input_identity.upstream_checkpoint_sha256 is None:
        raise ThresholdPairError("Input replay requires an identity already bound to an upstream checkpoint.")
    if not isinstance(input_bindings, Mapping) or set(input_bindings) != {"gray"}:
        raise ThresholdPairError("Both threshold stages require exactly one explicit gray input binding.")
    if not isinstance(input_roles, Mapping) or set(input_roles) != {"gray"} or input_roles.get("gray") != _GRAY_IMAGE_ROLE:
        raise ThresholdPairError("Both threshold stages require the exact uint8 gray image role.")
    input_array_name = input_bindings["gray"]
    if type(input_array_name) is not str or not input_array_name:
        raise ThresholdPairError("Gray input binding must name one checkpoint array.")
    _cancel_contract(cancel)
    resolved_timeout = _reference_timeout(runner, timeout_seconds)

    try:
        reference_stage = reference_recipe.stage(reference_stage_id)
        native_stage = native_recipe.stage(native_stage_id)
    except KeyError as error:
        raise ThresholdPairError("A requested threshold stage ID is absent from its exact recipe.") from error
    operation_id = reference_stage.operation_id
    if operation_id not in _THRESHOLD_OPERATIONS or native_stage.operation_id != operation_id:
        raise ThresholdPairError("Reference and native recipes must name the same supported public threshold operation.")
    reference_parameters = _stage_parameters(reference_recipe, reference_stage_id)
    native_parameters = _stage_parameters(native_recipe, native_stage_id)
    if not _same_json_types(reference_parameters, native_parameters):
        raise ThresholdPairError("Selected reference/native calculation parameters differ, including numeric types.")

    # Both stage bridges perform canonical source-function and typed-port checks
    # in their exact identity constructors. Derive both before invoking either.
    try:
        expected_reference = expected_reference_stage_identity(
            expected_input_identity,
            reference_recipe,
            reference_stage_id,
            input_checkpoint_sha256,
        )
        expected_native = expected_native_threshold_stage_identity(
            expected_input_identity,
            native_recipe,
            native_stage_id,
            input_checkpoint_sha256,
            expected_native_implementation_identity,
        )
    except (ReferenceStageError, NativeThresholdStageError, TypeError, ValueError) as error:
        raise ThresholdPairError(f"Threshold operation/source/port preflight failed: {error}") from error
    if expected_reference != expected_reference_output_identity or expected_native != expected_native_output_identity:
        raise ThresholdPairError("Caller output identities differ from the exact bridge-derived recipe/backend/environment chain.")

    correspondence = ThresholdMaskSemanticCorrespondence(
        operation_id=operation_id,
        reference_recipe=reference_recipe,
        candidate_recipe=native_recipe,
        input_identity=expected_input_identity,
        input_checkpoint_sha256=input_checkpoint_sha256,
        reference_identity=expected_reference,
        candidate_identity=expected_native,
        native_implementation_identity=expected_native_implementation_identity,
        input_array_name=input_array_name,
        input_role=input_roles["gray"],
        output_measurement=output_measurement,
        border_correspondence=(
            ThresholdBorderCorrespondence.REPLICATED_INPUT
            if operation_id == "threshold.mean"
            else ThresholdBorderCorrespondence.NOT_APPLICABLE_POINTWISE
        ),
    )

    # Resolve and validate the same immutable input once up front, before either
    # backend can execute. The bridges repeat this load and bind their own call.
    try:
        input_checkpoint = load_stage_input(
            input_root,
            input_checkpoint_sha256,
            expected_identity=expected_input_identity,
            expected_measurement=expected_input_measurement,
            limits=limits,
        )
    except CheckpointError as error:
        raise ThresholdPairError(f"Exact stage-pair input checkpoint failed replay validation: {error}") from error
    if input_checkpoint.identity.source_sha256 != expected_source_sha256 or input_checkpoint.identity.coordinate_frame != expected_coordinate_frame:
        raise ThresholdPairError("Loaded input checkpoint differs from exact source/frame pins.")
    if input_array_name not in input_checkpoint.arrays or input_checkpoint.raster_roles.get(input_array_name) != _GRAY_IMAGE_ROLE:
        raise ThresholdPairError("Bound gray input array or checkpoint role is missing or mismatched.")
    input_gray = input_checkpoint.arrays[input_array_name]
    if input_gray.dtype != np.dtype(np.uint8) or tuple(input_gray.shape) != expected_coordinate_frame.shape or not input_gray.flags.c_contiguous:
        raise ThresholdPairError("Bound gray input raster must exactly match the uint8 coordinate frame without conversion.")

    # The native identity is checked before the reference execution whenever
    # the requested device exists. A missing device remains an explicit native
    # UNAVAILABLE stage outcome later; it does not invent a CPU fallback.
    try:
        native_available = bool(torch.cuda.is_available()) and device.index < torch.cuda.device_count()
    except Exception:
        native_available = False
    if native_available:
        try:
            actual_native_identity = native_threshold_identity(device)
        except Exception as error:
            raise ThresholdPairError(f"Cannot verify the loaded native implementation before execution: {error}") from error
        if actual_native_identity != expected_native_implementation_identity:
            raise ThresholdPairError("Loaded native implementation or device environment differs from the caller's exact pin.")

    report_directory = _prepare_report_directory(artifact_store)
    preflight_report = {
        "schema": THRESHOLD_PAIR_REPORT_FORMAT,
        "version": THRESHOLD_PAIR_REPORT_VERSION,
        "operation_id": operation_id,
        "source_sha256": expected_source_sha256,
        "coordinate_frame": expected_coordinate_frame.to_dict(),
        "input_checkpoint_sha256": input_checkpoint_sha256,
        "output_identities": {
            "reference": expected_reference.to_dict(),
            "native": expected_native.to_dict(),
        },
        "semantic_correspondence": correspondence.to_dict(),
    }
    _json_bytes(preflight_report)

    try:
        reference_outcome = run_reference_stage(
            input_root,
            input_checkpoint_sha256,
            reference_recipe,
            reference_stage_id,
            runner,
            output_root,
            expected_source_sha256=expected_source_sha256,
            expected_coordinate_frame=expected_coordinate_frame,
            expected_recipe_identity=expected_reference_recipe_identity,
            expected_upstream_checkpoint_sha256=input_checkpoint_sha256,
            expected_input_identity=expected_input_identity,
            expected_input_measurement=expected_input_measurement,
            expected_output_identity=expected_reference,
            input_bindings=input_bindings,
            input_roles=input_roles,
            output_measurement=output_measurement,
            allow_cpu_transfer=True,
            limits=limits,
            timeout_seconds=resolved_timeout,
            cancel=cancel,
        )
    except Exception as error:
        reference_outcome = _failed_reference(reference_recipe, reference_stage_id, input_checkpoint_sha256, error)

    try:
        native_outcome = run_native_threshold_stage(
            input_root,
            input_checkpoint_sha256,
            native_recipe,
            native_stage_id,
            output_root,
            expected_source_sha256=expected_source_sha256,
            expected_coordinate_frame=expected_coordinate_frame,
            expected_recipe_identity=expected_native_recipe_identity,
            expected_upstream_checkpoint_sha256=input_checkpoint_sha256,
            expected_input_identity=expected_input_identity,
            expected_input_measurement=expected_input_measurement,
            expected_output_identity=expected_native,
            expected_implementation_identity=expected_native_implementation_identity,
            device=device,
            input_bindings=input_bindings,
            input_roles=input_roles,
            output_measurement=output_measurement,
            allow_cpu_transfer=True,
            limits=limits,
            cancel=cancel,
        )
    except Exception as error:
        native_outcome = _failed_native(native_recipe, native_stage_id, input_checkpoint_sha256, error)

    reference_evidence, reference_checkpoint = _stage_evidence(
        reference_outcome,
        output_root=output_root,
        recipe=reference_recipe,
        stage_id=reference_stage_id,
        upstream_sha256=input_checkpoint_sha256,
        expected_identity=expected_reference,
        output_measurement=output_measurement,
        limits=limits,
    )
    native_evidence, native_checkpoint = _stage_evidence(
        native_outcome,
        output_root=output_root,
        recipe=native_recipe,
        stage_id=native_stage_id,
        upstream_sha256=input_checkpoint_sha256,
        expected_identity=expected_native,
        output_measurement=output_measurement,
        limits=limits,
    )

    comparison: dict[str, Any] | None = None
    comparison_reason: str | None = None
    if reference_checkpoint is not None and native_checkpoint is not None:
        pair = CheckpointPairContract(
            reference_checkpoint_sha256=reference_checkpoint.checkpoint_sha256,
            candidate_checkpoint_sha256=native_checkpoint.checkpoint_sha256,
            reference_identity=expected_reference,
            candidate_identity=expected_native,
            comparison_scope="stage",
        )
        try:
            comparison = compare_masks(
                output_root,
                pair,
                reference_array="mask",
                candidate_array="mask",
                semantic_correspondence=correspondence,
            )
        except (ConformanceError, CheckpointError, ValueError) as error:
            comparison_reason = _exception_reason(error)
    else:
        comparison_reason = "Both exact bridge outputs were not available as verified immutable checkpoints."

    if comparison is None:
        pair_status = "not_evaluated"
        conformance_record: dict[str, Any] = {
            "status": "not_evaluated",
            "passed": None,
            "reason": comparison_reason or "No pixel comparison was performed.",
        }
    else:
        passed = bool(comparison["conformance"]["passed"])
        pair_status = "evaluated_pass" if passed else "evaluated_mismatch"
        conformance_record = {
            "status": "evaluated",
            "passed": passed,
            "result": comparison,
        }

    report = {
        "schema": THRESHOLD_PAIR_REPORT_FORMAT,
        "version": THRESHOLD_PAIR_REPORT_VERSION,
        "pair_status": pair_status,
        "stage_execution_complete": (
            reference_outcome.run_record.state in {StageRunState.EXECUTED, StageRunState.EMPTY}
            and native_outcome.run_record.state in {StageRunState.EXECUTED, StageRunState.EMPTY}
        ),
        "operation_id": operation_id,
        "source_sha256": expected_source_sha256,
        "coordinate_frame": expected_coordinate_frame.to_dict(),
        "input_checkpoint_sha256": input_checkpoint_sha256,
        "input_checkpoint_identity": expected_input_identity.to_dict(),
        "input_measurement": expected_input_measurement.to_dict(),
        "selected_parameters": dict(correspondence.selected_parameters),
        "semantic_correspondence": correspondence.to_dict(),
        "stages": {
            "reference": _report_stage(reference_outcome, reference_evidence),
            "native": _report_stage(native_outcome, native_evidence),
        },
        "conformance": conformance_record,
        "scientific_validity": {
            "status": "not_evaluated",
            "certification": "not_certified",
            "independent_truth": False,
        },
    }
    report_path, report_sha256, report_json = _publish_report(report_directory, report)
    return ThresholdConformanceOutcome(
        report_path=report_path,
        report_sha256=report_sha256,
        report_json=report_json,
        reference_outcome=reference_outcome,
        native_outcome=native_outcome,
    )


__all__ = [
    "MAX_THRESHOLD_PAIR_REPORT_BYTES",
    "THRESHOLD_PAIR_REPORT_FORMAT",
    "THRESHOLD_PAIR_REPORT_VERSION",
    "ThresholdConformanceOutcome",
    "ThresholdPairError",
    "run_threshold_conformance_pair",
]
