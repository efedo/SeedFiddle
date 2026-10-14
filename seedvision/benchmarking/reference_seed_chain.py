"""Coordinate the authored seed chain using actual pinned public PlantCV calls.

Watershed internals are artifacts of one call, never separately executed recipe
stages. This benchmark-only path does not change native CUDA graph execution.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping
from uuid import uuid4

import numpy as np

from .checkpoints import (
    CheckpointCancelled, CheckpointIdentity, CheckpointLimits, CheckpointReference,
    CoordinateFrame, ExecutionRecord, ExecutionStatus, MeasurementContract,
    RasterRole, _canonical_json, _cancelled, _freeze_json, _thaw_json,
    load_stage_input, publish_checkpoint,
)
from .recipes import RecipeIdentity, RecipeSpec
from .reference_preprocessing import (
    ReferencePreprocessingOutcome, _validate_pinned_recipe, run_reference_preprocessing,
)
from .reference_runner import (
    PlantCVReferenceRunner, ReferenceCancelledError, ReferenceResult,
    _validate_watershed_capture,
)
from .reference_stage import _validate_reference_identity


class ReferenceSeedChainError(ValueError):
    """A prepared input or selected source-chain contract is inconsistent."""


@dataclass(frozen=True, slots=True)
class ReferenceRasterInput:
    root: Path | str
    checkpoint_sha256: str
    identity: CheckpointIdentity
    measurement: MeasurementContract
    array_name: str
    role: RasterRole

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_sha256": self.checkpoint_sha256, "array_name": self.array_name,
            "identity": self.identity.to_dict(), "measurement": self.measurement.to_dict(),
            "role": self.role.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class ReferenceSourceCall:
    """invoked records a runner request; failed requests need not reach PlantCV."""
    operation: str
    call_id: str
    status: str
    invoked: bool
    reason: str
    checkpoint_reference: CheckpointReference | None
    details: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.status not in {"executed", "empty", "failed", "unavailable", "cancelled", "not_run"}:
            raise ValueError("Invalid source-call status.")
        if self.status in {"executed", "empty"} and (not self.invoked or self.checkpoint_reference is None):
            raise ValueError("Successful source calls require invocation and a checkpoint.")
        if self.status in {"not_run", "unavailable"} and self.invoked:
            raise ValueError("Uninvoked source calls cannot carry invocation evidence.")
        object.__setattr__(self, "details", _freeze_json(self.details, name="source-call details"))

    def to_dict(self) -> dict[str, Any]:
        reference = self.checkpoint_reference
        return {
            "operation": self.operation, "call_id": self.call_id,
            "call_id_origin": "seedfiddle_coordinator_attempt", "status": self.status,
            "invoked": self.invoked, "reason": self.reason,
            "invoked_meaning": "reference_runner_requested_not_proof_of_public_completion",
            "checkpoint_sha256": None if reference is None else reference.checkpoint_sha256,
            "checkpoint_identity": None if reference is None else reference.identity.to_dict(),
            "details": _thaw_json(self.details),
        }


@dataclass(frozen=True, slots=True)
class ReferenceSeedChainOutcome:
    status: str
    preprocessing: ReferencePreprocessingOutcome
    calls: tuple[ReferenceSourceCall, ...]
    record: Mapping[str, Any]
    manifest_path: Path | None = None
    manifest_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.status not in {"executed", "empty", "partial", "cancelled"} or len(self.calls) != 3:
            raise ValueError("A seed-chain outcome requires three source-call records and a valid status.")
        if (self.manifest_path is None) != (self.manifest_sha256 is None):
            raise ValueError("Manifest path and digest must be supplied together.")
        if self.manifest_path is not None and self.status != "executed":
            raise ValueError("Only a complete executed chain publishes a manifest.")
        object.__setattr__(self, "record", _freeze_json(self.record, name="seed-chain record"))


def _digest(array: np.ndarray) -> str:
    return sha256(array.tobytes(order="C")).hexdigest()


def _input(binding: ReferenceRasterInput, *, source: str, frame: CoordinateFrame,
           channels: str, limits: CheckpointLimits) -> np.ndarray:
    if not isinstance(binding, ReferenceRasterInput):
        raise TypeError("Prepared inputs must be ReferenceRasterInput values.")
    if not isinstance(binding.identity, CheckpointIdentity) or not isinstance(binding.measurement, MeasurementContract):
        raise TypeError("Prepared inputs require explicit identity and measurement contracts.")
    if binding.identity.source_sha256 != source or binding.identity.coordinate_frame != frame:
        raise ReferenceSeedChainError("Prepared gray/RGB source or coordinate frame differs.")
    layout = "HW" if channels == "GRAY" else "HWC"
    role = RasterRole("image", np.dtype(np.uint8).str, layout, (0, 255))
    if binding.role != role or binding.identity.image_semantics.get("color_order") != channels:
        raise ReferenceSeedChainError("Prepared input has a different channel or raster-role contract.")
    checkpoint = load_stage_input(binding.root, binding.checkpoint_sha256,
                                  expected_identity=binding.identity,
                                  expected_measurement=binding.measurement, limits=limits)
    array = checkpoint.arrays.get(binding.array_name)
    shape = frame.shape if channels == "GRAY" else (*frame.shape, 3)
    if (checkpoint.raster_roles.get(binding.array_name) != role or not isinstance(array, np.ndarray)
            or array.dtype != np.uint8 or array.shape != shape or not array.flags.c_contiguous):
        raise ReferenceSeedChainError("Prepared input does not contain its exact typed image binding.")
    return array


def _parameters(recipe: RecipeSpec) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    def one(operation: str) -> dict[str, Any]:
        matches = [stage for stage in recipe.stages if stage.operation_id == operation]
        if len(matches) != 1:
            raise ReferenceSeedChainError(f"Expected one authored {operation} contract.")
        return recipe.parameters_for_runner(matches[0].stage_id)
    watershed = one("reference.internal.watershed.local_maxima") | one("reference.internal.watershed.marker_count_observation")
    size, color = one("analysis.size"), one("analysis.color")
    if set(watershed) != {"distance", "label"} or type(watershed["distance"]) is not int or not 1 <= watershed["distance"] <= 255:
        raise ReferenceSeedChainError("Watershed distance must be selected in [1,255].")
    for parameters, keys in ((watershed, {"distance", "label"}), (size, {"n_labels", "label"}),
                             (color, {"n_labels", "label", "colorspaces"})):
        if set(parameters) != keys:
            raise ReferenceSeedChainError("Unsupported source-call parameters.")
        label = parameters["label"]
        if not isinstance(label, str) or not label or len(label) > 128:
            raise ReferenceSeedChainError("Source label must be a nonempty string of at most 128 characters.")
        if "n_labels" in parameters and parameters["n_labels"] is not None and (
                type(parameters["n_labels"]) is not int or parameters["n_labels"] < 1):
            raise ReferenceSeedChainError("n_labels must be unselected or an explicit positive integer.")
    if color["colorspaces"] not in {"all", "rgb", "lab", "hsv"}:
        raise ReferenceSeedChainError("Unsupported source colorspaces.")
    return watershed, size, color


def _identity(rgb: ReferenceRasterInput, recipe: RecipeSpec, operation: str, upstream: str) -> CheckpointIdentity:
    identity = recipe.identity
    return replace(
        rgb.identity, stage_id=operation.replace(".", "_"),
        recipe_id=identity.recipe_id, recipe_version=identity.recipe_version,
        recipe_sha256=identity.recipe_sha256, graph_sha256=identity.graph_sha256,
        settings_sha256=identity.settings_sha256, backend_id=identity.backend_id,
        backend_version=identity.backend_version,
        environment_versions=dict(recipe.provenance.environment_versions),
        instance_semantics={
            "label_dtype": "int32", "background_label": 0,
            "label_id_mapping": "actual_public_watershed_ids_unchanged",
            "split_boundary_ownership": "actual_public_watershed",
            "visibility": "full_frame", "eligibility": "not_evaluated",
        }, upstream_checkpoint_sha256=upstream,
    )


def _checked_result(result: Any, operation: str, parameters: Mapping[str, Any], recipe: RecipeSpec) -> ReferenceResult:
    if type(result) is not ReferenceResult or result.operation != operation:
        raise ReferenceSeedChainError("Runner returned a different public-call result.")
    _validate_reference_identity(result, recipe)
    for key, value in parameters.items():
        actual = result.effective_parameters.get(key)
        if type(actual) is not type(value) or actual != value:
            raise ReferenceSeedChainError(f"Runner effective parameter differs: {key}.")
    expected_callable = {
        "segmentation.watershed.capture": "plantcv.plantcv.watershed.watershed_segmentation",
        "analysis.size": "plantcv.plantcv.analyze.size", "analysis.color": "plantcv.plantcv.analyze.color",
    }[operation]
    if result.effective_parameters.get("callable") != expected_callable:
        raise ReferenceSeedChainError("Runner public callable differs.")
    for value in (result.computation_runtime_ns, result.wall_runtime_ns):
        if type(value) is not int or value < 0:
            raise ReferenceSeedChainError("Runner omitted a valid actual runtime.")
    return result


def _manifest(root: Path, record: Mapping[str, Any], limits: CheckpointLimits, cancel: Any) -> tuple[Path, str]:
    raw = _canonical_json(record, name="source-chain manifest", maximum=limits.max_metadata_bytes)
    digest = sha256(raw).hexdigest()
    directory = root / "chains"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{digest}.json"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=directory, suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        if _cancelled(cancel):
            raise CheckpointCancelled("Source-chain manifest publication was cancelled.")
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path, digest


def run_reference_seed_chain(
    gray: ReferenceRasterInput, rgb: ReferenceRasterInput, recipe: RecipeSpec,
    runner: PlantCVReferenceRunner, output_root: Path | str, *,
    expected_source_sha256: str, expected_coordinate_frame: CoordinateFrame,
    expected_recipe_identity: RecipeIdentity, output_measurement: MeasurementContract,
    allow_cpu_transfer: bool, limits: CheckpointLimits = CheckpointLimits(),
    timeout_seconds: float | None = None, cancel: Any = None,
) -> ReferenceSeedChainOutcome:
    """Run five preprocessing calls, one captured watershed, and size/color.

Unselected measurement loop bounds use the authored int(labels.max()) rule.
Explicit positive selections are passed unchanged, including on an empty mask.
Successful individual checkpoints survive partial outcomes; only a fully
executed chain publishes the final manifest. Every output remains scientifically
ineligible under this original-reference execution contract.
"""
    if not isinstance(recipe, RecipeSpec) or recipe.identity != expected_recipe_identity:
        raise ReferenceSeedChainError("The exact selected recipe identity is required.")
    if type(runner) is not PlantCVReferenceRunner or not isinstance(limits, CheckpointLimits):
        raise TypeError("An explicit reference runner and checkpoint limits are required.")
    if not isinstance(expected_coordinate_frame, CoordinateFrame):
        raise TypeError("An explicit coordinate frame is required.")
    if not isinstance(output_measurement, MeasurementContract) or output_measurement.eligible:
        raise ReferenceSeedChainError("Reference outputs require a scientifically ineligible measurement contract.")
    if allow_cpu_transfer is not True:
        raise ReferenceSeedChainError("Reference checkpoint serialization requires allow_cpu_transfer=True.")
    _cancelled(cancel)  # Validate the cancellation interface before any public call.
    _validate_pinned_recipe(recipe)
    watershed_parameters, size_parameters, color_parameters = _parameters(recipe)
    _input(gray, source=expected_source_sha256, frame=expected_coordinate_frame, channels="GRAY", limits=limits)
    rgb_array = _input(rgb, source=expected_source_sha256, frame=expected_coordinate_frame, channels="RGB", limits=limits)
    rgb_digest = _digest(rgb_array)
    output_root = Path(output_root)
    preprocessing = run_reference_preprocessing(
        gray.root, gray.checkpoint_sha256, recipe, runner, output_root,
        expected_source_sha256=expected_source_sha256, expected_coordinate_frame=expected_coordinate_frame,
        expected_recipe_identity=expected_recipe_identity, expected_input_identity=gray.identity,
        expected_input_measurement=gray.measurement, output_measurement=output_measurement,
        input_bindings={"gray": gray.array_name}, input_roles={"gray": gray.role},
        allow_cpu_transfer=True, limits=limits, timeout_seconds=timeout_seconds, cancel=cancel,
    )
    if not isinstance(preprocessing, ReferencePreprocessingOutcome) or preprocessing.recipe_identity != recipe.identity:
        raise ReferenceSeedChainError("Preprocessing returned a different recipe identity.")
    operations = ("segmentation.watershed.capture", "analysis.size", "analysis.color")
    calls: list[ReferenceSourceCall] = []
    label_facts: dict[str, Any] = {}
    labels: np.ndarray | None = None
    watershed_reference: CheckpointReference | None = None
    capture_status = "not_run"

    def stopped(operation: str, status: str, reason: str, details: Mapping[str, Any] | None = None) -> None:
        calls.append(ReferenceSourceCall(operation, uuid4().hex, status, False, reason, None, details or {}))

    def publish(operation: str, parameters: Mapping[str, Any], inputs: Mapping[str, np.ndarray],
                upstream: str, *, requested: Mapping[str, Any]) -> tuple[ReferenceSourceCall, np.ndarray | None]:
        nonlocal capture_status, label_facts
        call_id = uuid4().hex
        details = {"requested_parameters": dict(requested), "parameters_passed": dict(parameters)}
        result_labels = None
        invoked = False
        input_digests: dict[str, str] = {}
        try:
            if _cancelled(cancel):
                raise ReferenceCancelledError("Cancelled before source call.")
            if _digest(rgb_array) != rgb_digest:
                raise ReferenceSeedChainError("Prepared analysis RGB changed before source invocation.")
            input_digests = {name: _digest(value) for name, value in inputs.items()}
            invoked = True
            result = _checked_result(runner.run(operation, inputs, parameters,
                                     timeout_seconds=timeout_seconds, cancel_event=cancel),
                                     operation, parameters, recipe)
            details["verified_public_result_received"] = True
            if _cancelled(cancel):
                raise ReferenceCancelledError("Cancelled after source call.")
            if _digest(rgb_array) != rgb_digest or any(_digest(inputs[name]) != digest for name, digest in input_digests.items()):
                raise ReferenceSeedChainError("Public-call inputs changed during execution.")
            arrays = dict(result.arrays)
            roles: dict[str, RasterRole] = {}
            origins: dict[str, str] = {}
            if operation == "segmentation.watershed.capture":
                capture = result.effective_parameters.get("capture")
                if not isinstance(capture, Mapping) or not isinstance(capture.get("call_id"), str) or not capture["call_id"]:
                    raise ReferenceSeedChainError("Watershed omitted its actual worker call identity.")
                # The runner has reconstructed empty coordinates. Validate the
                # same logical transport without modifying the returned arrays.
                checked_arrays = dict(arrays)
                if "empty_coordinates" in capture:
                    coordinates = checked_arrays.pop("coordinates", None)
                    if coordinates is None or coordinates.shape != (0, 2) or coordinates.dtype != np.dtype("<i8"):
                        raise ReferenceSeedChainError("Reconstructed empty coordinates differ.")
                _validate_watershed_capture(checked_arrays, [], dict(result.effective_parameters), capture["call_id"], inputs)
                if result.identity.get("capture_source_files") is None:
                    raise ReferenceSeedChainError("Watershed omitted capture source pins.")
                from .reference_watershed_capture import capture_source_records
                if result.identity["capture_source_files"] != capture_source_records():
                    raise ReferenceSeedChainError("Watershed capture source pins differ.")
                result_labels = arrays["labels"]
                capture_status = capture["status"]
                label_ids = np.unique(result_labels)
                positive_ids = label_ids[label_ids > 0].tolist()
                label_facts = {"positive_label_ids": positive_ids, "distinct_positive_label_count": len(positive_ids),
                               "maximum_label_id": int(result_labels.max()), "labels_sha256": _digest(result_labels),
                               "positive_marker_ids": capture.get("positive_marker_ids"),
                               "estimated_object_count": capture.get("estimated_object_count"),
                               "positive_marker_count": None if "positive_marker_ids" not in capture else len(capture["positive_marker_ids"])}
                details.update({"capture_status": capture_status, "worker_call_id": capture["call_id"],
                                "capture_reason": capture.get("reason", ""), "label_facts": label_facts})
                for name, array in arrays.items():
                    origins[name] = "actual_public_return" if name == "labels" else "actual_watershed_local"
                    if name in {"coordinates", "palette"}:
                        continue
                    if name in {"markers", "labels"}:
                        roles[name] = RasterRole("instances", array.dtype.str, "HW")
                    elif name == "diagnostic":
                        roles[name] = RasterRole("image", array.dtype.str, "HWC", (0, 255))
                        origins[name] = "BGR_to_RGB_adapter_of_actual_plt_img"
                    else:
                        roles[name] = RasterRole("raster", array.dtype.str, "HWC" if array.ndim == 3 else "HW")
            else:
                expected_arrays = {"analysis_image"} if operation == "analysis.size" else set()
                if set(arrays) != expected_arrays:
                    raise ReferenceSeedChainError("Source analysis output inventory differs.")
                if "analysis_image" in arrays:
                    image = arrays["analysis_image"]
                    if image.dtype != np.uint8 or image.shape != rgb_array.shape:
                        raise ReferenceSeedChainError("Size diagnostic differs from its RGB frame.")
                    roles["analysis_image"] = RasterRole("image", image.dtype.str, "HWC", (0, 255))
                    origins["analysis_image"] = "RGB_adapter_of_actual_public_return"
                # Color produces observations/chart only. This named raster is
                # the exact input context required by the checkpoint container.
                arrays["labels_context"] = inputs["labels"]
                roles["labels_context"] = RasterRole("instances", inputs["labels"].dtype.str, "HW")
                origins["labels_context"] = "input_context_not_computed_output"
                details.update({"n_labels_rule": "int(public_labels.max())" if requested["n_labels"] is None else "explicit_positive_selection",
                                "effective_n_labels": parameters["n_labels"], "labels_sha256": input_digests["labels"]})
            metadata = {
                "schema": "seedfiddle.public-seed-call.v1", "operation": operation,
                "call_id": call_id, "call_id_origin": "seedfiddle_coordinator_attempt",
                "recipe_identity": recipe.identity.to_dict(), "source_sha256": expected_source_sha256,
                "coordinate_frame": expected_coordinate_frame.to_dict(), "rgb_input": rgb.to_dict(),
                "input_array_sha256": input_digests, "upstream_checkpoint_sha256": upstream,
                "requested_parameters": dict(requested), "parameters_passed": dict(parameters),
                "effective_parameters": dict(result.effective_parameters), "runner_identity": dict(result.identity),
                "observations": dict(result.observations), "visualizations": dict(result.visualizations),
                "channel_semantics": dict(result.channel_semantics), "array_origins": origins,
                "computation_runtime_ns": result.computation_runtime_ns, "wall_runtime_ns": result.wall_runtime_ns,
                "scientifically_validated": False,
            }
            empty = operation == "segmentation.watershed.capture" and not np.any(result_labels)
            reference = publish_checkpoint(
                output_root, _identity(rgb, recipe, operation, upstream), arrays,
                execution=ExecutionRecord(ExecutionStatus.EMPTY if empty else ExecutionStatus.AVAILABLE,
                                          elapsed_ms=result.wall_runtime_ns / 1e6),
                measurement=output_measurement, raster_roles=roles, metadata=metadata,
                allow_cpu_transfer=True, limits=limits, cancel=cancel,
            )
            details["reference_identity_digest"] = result.identity["identity_digest"]
            return ReferenceSourceCall(operation, call_id, "empty" if empty else "executed", True, "", reference, details), result_labels
        except Exception as error:
            cancelled = isinstance(error, (ReferenceCancelledError, CheckpointCancelled)) or _cancelled(cancel)
            reason = " ".join(f"{type(error).__name__}: {error}".split())[:512]
            if invoked and (_digest(rgb_array) != rgb_digest or any(
                    _digest(inputs[name]) != digest for name, digest in input_digests.items())):
                reason = "ReferenceSeedChainError: Public-call inputs changed during execution; returned products withheld."
                details["input_contract_violated"] = True
            return ReferenceSourceCall(operation, call_id, "cancelled" if cancelled else "failed", invoked, reason, None, details), None

    final = preprocessing.final_checkpoint
    if final is None or _cancelled(cancel):
        for operation in operations:
            stopped(operation, "cancelled" if _cancelled(cancel) else "not_run", "Preprocessing did not provide a usable final mask.")
    else:
        mask_checkpoint = load_stage_input(output_root, final.checkpoint_sha256,
                                          expected_identity=final.identity, expected_measurement=output_measurement, limits=limits)
        watershed_call, labels = publish(operations[0], watershed_parameters,
                                        {"rgb": rgb_array, "mask": mask_checkpoint.arrays["mask"]},
                                        final.checkpoint_sha256, requested=watershed_parameters)
        calls.append(watershed_call)
        watershed_reference = watershed_call.checkpoint_reference
        for operation, requested in zip(operations[1:], (size_parameters, color_parameters), strict=True):
            if _cancelled(cancel) or calls[-1].status == "cancelled":
                stopped(operation, "cancelled", "Cancelled before downstream analysis.")
            elif labels is None or watershed_reference is None:
                stopped(operation, "not_run", "Watershed did not provide a published labels checkpoint.")
            else:
                # Each independent analysis consumes the content-verified
                # published labels, not a buffer left behind by a prior call.
                checkpoint = load_stage_input(output_root, watershed_reference.checkpoint_sha256,
                    expected_identity=watershed_reference.identity, expected_measurement=output_measurement, limits=limits)
                metadata = checkpoint.metadata
                if (metadata.get("operation") != operations[0] or metadata.get("call_id") != watershed_call.call_id
                        or _canonical_json(metadata.get("recipe_identity"), name="stored recipe") != _canonical_json(recipe.identity.to_dict(), name="expected recipe")
                        or _canonical_json(metadata.get("parameters_passed"), name="stored parameters") != _canonical_json(watershed_parameters, name="expected parameters")
                        or metadata.get("runner_identity", {}).get("identity_digest") != watershed_call.details.get("reference_identity_digest")):
                    raise ReferenceSeedChainError("Published watershed call bindings differ.")
                labels = checkpoint.arrays["labels"]
                if _digest(labels) != label_facts["labels_sha256"]:
                    raise ReferenceSeedChainError("Published watershed labels differ from their call digest.")
                effective = dict(requested)
                if effective["n_labels"] is None:
                    effective["n_labels"] = int(labels.max())
                if effective["n_labels"] == 0:
                    stopped(operation, "unavailable", "Unselected n_labels resolves to zero; public analysis requires a positive loop bound.",
                            {"requested_parameters": requested, "effective_n_labels": 0, "n_labels_rule": "int(public_labels.max())"})
                    continue
                input_name = "image" if operation == "analysis.size" else "rgb"
                call, _ = publish(operation, effective, {input_name: rgb_array, "labels": labels},
                                  watershed_reference.checkpoint_sha256, requested=requested)
                calls.append(call)
    cancelled = _cancelled(cancel) or preprocessing.status.value == "cancelled" or any(call.status == "cancelled" for call in calls)
    complete = not cancelled and capture_status == "available" and all(call.status in {"executed", "empty"} for call in calls)
    empty = (not cancelled and calls[0].status == "empty" and capture_status == "available"
             and all(call.status in {"executed", "unavailable"} for call in calls[1:]))
    status = "cancelled" if cancelled else "executed" if complete else "empty" if empty else "partial"
    record = {
        "schema": "seedfiddle.public-seed-chain.v1", "status": status, "complete": complete,
        "recipe_identity": recipe.identity.to_dict(), "source_sha256": expected_source_sha256,
        "coordinate_frame": expected_coordinate_frame.to_dict(), "inputs": {"gray": gray.to_dict(), "rgb": rgb.to_dict()},
        "preprocessing": [{"operation": stage.stage.operation_id, "status": stage.status.value,
                           "run_record": stage.run_record.to_dict()} for stage in preprocessing.stages],
        "source_calls": [call.to_dict() for call in calls], "label_facts": label_facts,
        "runner_request_count_after_preprocessing": sum(call.invoked for call in calls),
        "verified_public_result_count_after_preprocessing": sum(bool(call.details.get("verified_public_result_received")) for call in calls),
        "scientifically_validated": False,
    }
    manifest_path = manifest_sha256 = None
    if complete:
        try:
            manifest_path, manifest_sha256 = _manifest(output_root, record, limits, cancel)
        except CheckpointCancelled:
            status = "cancelled"
            record.update(status=status, complete=False)
    return ReferenceSeedChainOutcome(status, preprocessing, tuple(calls), record, manifest_path, manifest_sha256)


__all__ = ["ReferenceRasterInput", "ReferenceSourceCall", "ReferenceSeedChainError",
           "ReferenceSeedChainOutcome", "run_reference_seed_chain"]
