"""Read a caller-addressed complete public chain without executing analysis.

The supplied manifest digest is the trusted revision. Content hashes do not
independently prove numerical or scientific correctness of the original call.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from .checkpoints import (
    BenchmarkCheckpoint, CheckpointCancelled, CheckpointLimits, CoordinateFrame,
    ExecutionStatus, MeasurementContract, RasterRole, _canonical_json, _cancelled,
    _freeze_json, _thaw_json, load_checkpoint,
)
from .checkpoint_inspection import _verify_runner_identity
from .recipes import RecipeSpec, StageRunRecord, StageRunState, validate_stage_run
from .reference_preprocessing import _chain_stages, _stage_output_identity, _validate_pinned_recipe
from .reference_stage import REFERENCE_STAGE_FORMAT, _validate_result as _threshold_result
from .reference_mask_stage import REFERENCE_MASK_STAGE_FORMAT, _validate_runner_result as _mask_result
from .reference_runner import ReferenceResult, _strict_json_loads, _validate_watershed_capture
from .reference_seed_chain import ReferenceRasterInput, _checked_result, _digest, _identity, _parameters
from .reference_watershed_capture import capture_source_records


class ReferenceChainReadError(ValueError):
    """A manifest or one of its typed content-addressed bindings is inconsistent."""


@dataclass(frozen=True, slots=True)
class VerifiedReferenceSeedChain:
    manifest_sha256: str
    record: Mapping[str, Any]
    checkpoints: Mapping[str, BenchmarkCheckpoint]
    retained_array_bytes: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "record", _freeze_json(self.record, name="verified chain"))
        object.__setattr__(self, "checkpoints", MappingProxyType(dict(self.checkpoints)))


def _same(actual: Any, expected: Any, name: str) -> None:
    if _canonical_json(actual, name=name) != _canonical_json(expected, name=name):
        raise ReferenceChainReadError(f"Stored {name} differs from its exact binding.")


def _poll(cancel: Any) -> None:
    if _cancelled(cancel):
        raise CheckpointCancelled("Public chain verification cancelled; no verified result returned.")


def _result(checkpoint: BenchmarkCheckpoint, operation: str, *, arrays=None) -> ReferenceResult:
    meta = _thaw_json(checkpoint.metadata)
    return ReferenceResult(
        operation, dict(checkpoint.arrays) if arrays is None else arrays,
        meta["observations"], meta["effective_parameters"], tuple(meta.get("intermediates", ())),
        meta["visualizations"], meta["channel_semantics"], meta["runner_identity"],
        meta["computation_runtime_ns"], meta["wall_runtime_ns"],
    )


def read_reference_seed_chain(
    output_root: Path | str, manifest_sha256: str, *, gray: ReferenceRasterInput,
    rgb: ReferenceRasterInput, recipe: RecipeSpec, expected_source_sha256: str,
    expected_coordinate_frame: CoordinateFrame, output_measurement: MeasurementContract,
    limits: CheckpointLimits = CheckpointLimits(), cancel: Any = None,
) -> VerifiedReferenceSeedChain:
    """Verify every complete-chain binding before exposing any stored arrays.

The manifest never selects filesystem roots. The caller supplies its trusted
digest and prepared inputs, recipe and measurement contract. The numeric-array
cap covers retained checkpoint arrays across the entire read; metadata, loader
scratch and process peak/RSS are outside that bound.
"""
    if not isinstance(recipe, RecipeSpec) or not isinstance(limits, CheckpointLimits):
        raise TypeError("An immutable recipe and explicit checkpoint limits are required.")
    if not isinstance(gray, ReferenceRasterInput) or not isinstance(rgb, ReferenceRasterInput):
        raise TypeError("Explicit typed gray/RGB bindings are required.")
    if not isinstance(output_measurement, MeasurementContract) or output_measurement.eligible:
        raise ReferenceChainReadError("Public reference measurements must remain scientifically ineligible.")
    if not isinstance(expected_coordinate_frame, CoordinateFrame):
        raise TypeError("An exact coordinate frame is required.")
    if not isinstance(manifest_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", manifest_sha256):
        raise ReferenceChainReadError("A trusted lowercase manifest SHA-256 is required.")
    _poll(cancel)
    _validate_pinned_recipe(recipe)
    stages = _chain_stages(recipe)
    ws_parameters, size_parameters, color_parameters = _parameters(recipe)
    root = Path(output_root).resolve()
    manifest = root / "chains" / f"{manifest_sha256}.json"
    with manifest.open("rb") as stream:
        raw = stream.read(limits.max_metadata_bytes + 1)
    if len(raw) > limits.max_metadata_bytes or sha256(raw).hexdigest() != manifest_sha256:
        raise ReferenceChainReadError("Manifest bytes differ from the trusted revision or exceed the metadata cap.")
    record = _strict_json_loads(raw)
    required = {"schema", "status", "complete", "recipe_identity", "source_sha256", "coordinate_frame",
                "inputs", "preprocessing", "source_calls", "label_facts", "runner_request_count_after_preprocessing",
                "verified_public_result_count_after_preprocessing", "scientifically_validated"}
    if not isinstance(record, dict) or set(record) != required:
        raise ReferenceChainReadError("Public chain manifest schema differs.")
    for name, value in {"schema": "seedfiddle.public-seed-chain.v1", "status": "executed", "complete": True,
                        "scientifically_validated": False, "recipe_identity": recipe.identity.to_dict(),
                        "source_sha256": expected_source_sha256, "coordinate_frame": expected_coordinate_frame.to_dict(),
                        "inputs": {"gray": gray.to_dict(), "rgb": rgb.to_dict()},
                        "runner_request_count_after_preprocessing": 3,
                        "verified_public_result_count_after_preprocessing": 3}.items():
        _same(record[name], value, name)
    if not isinstance(record["preprocessing"], list) or len(record["preprocessing"]) != 5:
        raise ReferenceChainReadError("Exactly five preprocessing records are required.")
    if not isinstance(record["source_calls"], list) or len(record["source_calls"]) != 3:
        raise ReferenceChainReadError("Exactly three source-call records are required.")
    checkpoints: dict[str, BenchmarkCheckpoint] = {}
    retained = 0

    def load(name, store, digest, identity, measurement):
        nonlocal retained
        _poll(cancel)
        remaining = limits.max_total_array_bytes - retained
        if remaining < 1:
            raise ReferenceChainReadError("Retained chain arrays exceed their aggregate budget.")
        per_read = replace(limits, max_total_array_bytes=remaining,
                           max_array_bytes=min(limits.max_array_bytes, remaining))
        checkpoint = load_checkpoint(store, digest, expected_identity=identity,
                                     expected_measurement=measurement, limits=per_read)
        retained += sum(array.nbytes for array in checkpoint.arrays.values())
        if retained > limits.max_total_array_bytes:
            raise ReferenceChainReadError("Retained chain arrays exceed their aggregate budget.")
        checkpoints[name] = checkpoint
        _poll(cancel)
        return checkpoint

    def prepared(name, binding, channels):
        if not isinstance(binding, ReferenceRasterInput):
            raise TypeError("Explicit typed prepared inputs are required.")
        identity = binding.identity
        if identity.source_sha256 != expected_source_sha256 or identity.coordinate_frame != expected_coordinate_frame:
            raise ReferenceChainReadError("Prepared source/frame differs.")
        checkpoint = load(f"input.{name}", binding.root, binding.checkpoint_sha256, identity, binding.measurement)
        role = RasterRole("image", "|u1", "HW" if channels == "GRAY" else "HWC", (0, 255))
        array = checkpoint.arrays.get(binding.array_name)
        shape = expected_coordinate_frame.shape if channels == "GRAY" else (*expected_coordinate_frame.shape, 3)
        if (binding.role != role or checkpoint.raster_roles.get(binding.array_name) != role
                or identity.image_semantics.get("color_order") != channels or array is None
                or array.dtype != np.uint8 or array.shape != shape):
            raise ReferenceChainReadError("Prepared channel/raster binding differs.")
        return checkpoint, array

    gray_checkpoint, _ = prepared("gray", gray, "GRAY")
    _, rgb_array = prepared("rgb", rgb, "RGB")
    previous = gray_checkpoint
    previous_digest = gray.checkpoint_sha256
    previous_array_name = gray.array_name
    for index, (stage, item) in enumerate(zip(stages, record["preprocessing"], strict=True)):
        if not isinstance(item, dict) or set(item) != {"operation", "status", "run_record"} or item["operation"] != stage.operation_id:
            raise ReferenceChainReadError("Preprocessing order or operation differs.")
        run = StageRunRecord.from_dict(item["run_record"])
        validate_stage_run(recipe, run, expected_upstream_checkpoint_sha256=previous_digest)
        if run.stage_id != stage.stage_id or run.state not in {StageRunState.EXECUTED, StageRunState.EMPTY} or item["status"] != run.state.value:
            raise ReferenceChainReadError("Preprocessing run status differs.")
        identity = _stage_output_identity(previous.identity, recipe, stage, previous_digest)
        checkpoint = load(f"preprocessing.{index}", root, run.checkpoint_sha256, identity, output_measurement)
        metadata = checkpoint.metadata
        params = recipe.parameters_for_runner(stage.stage_id)
        output_name = "image" if index == 0 else "mask"
        output_role = RasterRole("image", "|u1", "HW", (0, 255)) if index == 0 else RasterRole("mask", "|u1", "HW")
        input_port = "mask" if stage.operation_id == "morphology.fill" else "gray"
        input_role = previous.raster_roles[previous_array_name]
        expected_meta = {
            "format": REFERENCE_STAGE_FORMAT if stage.operation_id.startswith("threshold.") else REFERENCE_MASK_STAGE_FORMAT,
            "version": 1, "stage_id": stage.stage_id, "operation_id": stage.operation_id,
            "recipe_identity": recipe.identity.to_dict(), "source_mapping": stage.source_mappings[0].to_dict(),
            "source_sha256": expected_source_sha256, "coordinate_frame": expected_coordinate_frame.to_dict(),
            "upstream_checkpoint_sha256": previous_digest, "input_checkpoint_identity": previous.identity.to_dict(),
            "input_bindings": {input_port: previous_array_name}, "input_raster_roles": {input_port: input_role.to_dict()},
            "output_raster_roles": {output_name: output_role.to_dict()}, "parameters": params,
            "measurement_contract": output_measurement.to_dict(), "scientific_validation_status": "not_certified",
        }
        for name, value in expected_meta.items():
            _same(metadata.get(name), value, f"preprocessing {name}")
        if not stage.operation_id.startswith("threshold."):
            _same(metadata.get("output_identity"), identity.to_dict(), "preprocessing output identity")
        if dict(checkpoint.raster_roles) != {output_name: output_role}:
            raise ReferenceChainReadError("Preprocessing artifact roles differ.")
        _verify_runner_identity(metadata, recipe)
        result = _result(checkpoint, stage.operation_id)
        if stage.operation_id.startswith("threshold."):
            _threshold_result(result, recipe=recipe, stage_id=stage.stage_id, operation_id=stage.operation_id,
                              parameters=params, frame=expected_coordinate_frame)
        else:
            _mask_result(result, operation=stage.operation_id, parameters=params, recipe=recipe, frame=expected_coordinate_frame)
        _same(checkpoint.execution.elapsed_ms, result.wall_runtime_ns / 1e6, "preprocessing measured runtime")
        array = checkpoint.arrays[output_name]
        expected_empty = index > 0 and not np.any(array)
        _same(checkpoint.execution.status.value, "empty" if expected_empty else "available", "preprocessing execution")
        _same(item["status"], "empty" if expected_empty else "executed", "preprocessing status")
        _same(metadata.get("output_inventory"), [{"name": output_name, "dtype": array.dtype.str,
              "shape": list(array.shape), "nbytes": array.nbytes}], "preprocessing inventory")
        previous, previous_digest, previous_array_name = checkpoint, checkpoint.checkpoint_sha256, output_name
    mask = previous.arrays["mask"]
    operations = ("segmentation.watershed.capture", "analysis.size", "analysis.color")
    source_keys = {"operation", "call_id", "call_id_origin", "status", "invoked", "reason", "invoked_meaning",
                   "checkpoint_sha256", "checkpoint_identity", "details"}
    ids = []
    labels = None
    label_facts = None
    watershed_digest = None
    for index, (operation, call, requested) in enumerate(zip(operations, record["source_calls"],
                                                           (ws_parameters, size_parameters, color_parameters), strict=True)):
        if not isinstance(call, dict) or set(call) != source_keys or call["operation"] != operation:
            raise ReferenceChainReadError("Source-call order or schema differs.")
        call_id = call["call_id"]
        if not isinstance(call_id, str) or not re.fullmatch(r"[0-9a-f]{32}", call_id) or call_id in ids:
            raise ReferenceChainReadError("Coordinator call identities must be distinct valid UUID hex values.")
        ids.append(call_id)
        _same(call["call_id_origin"], "seedfiddle_coordinator_attempt", "call identity origin")
        _same(call["invoked_meaning"], "reference_runner_requested_not_proof_of_public_completion", "call invocation meaning")
        _same(call["invoked"], True, "runner invocation")
        _same(call["reason"], "", "successful call reason")
        details = call["details"]
        if not isinstance(details, dict) or details.get("input_contract_violated"):
            raise ReferenceChainReadError("Successful source-call details are inconsistent.")
        _same(details.get("verified_public_result_received"), True, "verified public result")
        parameters = dict(requested)
        upstream = previous_digest if index == 0 else watershed_digest
        if index:
            if parameters["n_labels"] is None:
                parameters["n_labels"] = int(labels.max())
            if parameters["n_labels"] <= 0:
                raise ReferenceChainReadError("A default-empty chain cannot claim completed analyses.")
        identity = _identity(rgb, recipe, operation, upstream)
        _same(call["checkpoint_identity"], identity.to_dict(), "source-call checkpoint identity")
        name = ("watershed", "size", "color")[index]
        checkpoint = load(name, root, call["checkpoint_sha256"], identity, output_measurement)
        metadata = checkpoint.metadata
        expected_metadata = {"schema": "seedfiddle.public-seed-call.v1", "operation": operation, "call_id": call_id,
            "call_id_origin": "seedfiddle_coordinator_attempt", "recipe_identity": recipe.identity.to_dict(),
            "source_sha256": expected_source_sha256, "coordinate_frame": expected_coordinate_frame.to_dict(),
            "rgb_input": rgb.to_dict(), "upstream_checkpoint_sha256": upstream,
            "requested_parameters": requested, "parameters_passed": parameters, "scientifically_validated": False}
        for key, value in expected_metadata.items():
            _same(metadata.get(key), value, f"source {key}")
        _same(details.get("requested_parameters"), requested, "requested controls")
        _same(details.get("parameters_passed"), parameters, "passed controls")
        _verify_runner_identity(metadata, recipe)
        _same(details.get("reference_identity_digest"), metadata["runner_identity"]["identity_digest"], "call runtime identity")
        result_arrays = dict(checkpoint.arrays)
        if index:
            result_arrays.pop("labels_context", None)
        result = _checked_result(_result(checkpoint, operation, arrays=result_arrays), operation, parameters, recipe)
        _same(checkpoint.execution.elapsed_ms, result.wall_runtime_ns / 1e6, "source measured runtime")
        if index == 0:
            capture = result.effective_parameters["capture"]
            if not isinstance(capture.get("call_id"), str) or not re.fullmatch(r"[0-9a-f]{32}", capture["call_id"]):
                raise ReferenceChainReadError("Capture worker identity is invalid.")
            _same(capture.get("status"), "available", "required capture status")
            checked = dict(result.arrays)
            if "empty_coordinates" in capture:
                coordinates = checked.pop("coordinates", None)
                if coordinates is None or coordinates.dtype != np.dtype("<i8") or coordinates.shape != (0, 2):
                    raise ReferenceChainReadError("Empty-coordinate reconstruction differs.")
            _validate_watershed_capture(checked, [], dict(result.effective_parameters), capture["call_id"], {"mask": mask})
            _same(result.identity.get("capture_source_files"), capture_source_records(), "capture source pins")
            _same(details.get("worker_call_id"), capture["call_id"], "capture worker call identity")
            _same(details.get("capture_status"), "available", "call capture status")
            labels = checkpoint.arrays["labels"]
            label_ids = np.unique(labels)
            positive_ids = label_ids[label_ids > 0].tolist()
            label_facts = {"positive_label_ids": positive_ids, "distinct_positive_label_count": len(positive_ids),
                           "maximum_label_id": int(labels.max()), "labels_sha256": _digest(labels),
                           "positive_marker_ids": capture["positive_marker_ids"],
                           "estimated_object_count": capture["estimated_object_count"],
                           "positive_marker_count": len(capture["positive_marker_ids"])}
            _same(details.get("label_facts"), label_facts, "call label facts")
            _same(record["label_facts"], label_facts, "manifest label facts")
            expected_roles = {key: RasterRole("instances" if key in {"markers", "labels"} else "image" if key == "diagnostic" else "raster",
                                value.dtype.str, "HWC" if value.ndim == 3 else "HW", (0, 255) if key == "diagnostic" else None)
                              for key, value in checkpoint.arrays.items() if key not in {"coordinates", "palette"}}
            expected_origins = {key: "actual_public_return" if key == "labels" else "BGR_to_RGB_adapter_of_actual_plt_img" if key == "diagnostic" else "actual_watershed_local"
                                for key in checkpoint.arrays}
            inputs = {"rgb": _digest(rgb_array), "mask": _digest(mask)}
            empty = not np.any(labels)
            watershed_digest = checkpoint.checkpoint_sha256
        else:
            context = checkpoint.arrays.get("labels_context")
            if context is None or context.dtype != np.int32 or not np.array_equal(context, labels):
                raise ReferenceChainReadError("Analysis label context differs from the actual watershed output.")
            expected_roles = {"labels_context": RasterRole("instances", "<i4", "HW")}
            expected_origins = {"labels_context": "input_context_not_computed_output"}
            if index == 1:
                image = checkpoint.arrays.get("analysis_image")
                if set(checkpoint.arrays) != {"labels_context", "analysis_image"} or image.dtype != np.uint8 or image.shape != rgb_array.shape:
                    raise ReferenceChainReadError("Size artifact inventory/frame differs.")
                expected_roles["analysis_image"] = RasterRole("image", "|u1", "HWC", (0, 255))
                expected_origins["analysis_image"] = "RGB_adapter_of_actual_public_return"
            elif set(checkpoint.arrays) != {"labels_context"}:
                raise ReferenceChainReadError("Color observations-only artifact inventory differs.")
            inputs = {"image" if index == 1 else "rgb": _digest(rgb_array), "labels": _digest(labels)}
            _same(details.get("effective_n_labels"), parameters["n_labels"], "analysis loop bound")
            _same(details.get("n_labels_rule"), "int(public_labels.max())" if requested["n_labels"] is None else "explicit_positive_selection", "analysis loop-bound rule")
            _same(details.get("labels_sha256"), _digest(labels), "analysis labels digest")
            empty = False
        _same(checkpoint.metadata.get("input_array_sha256"), inputs, "source input pixels")
        if dict(checkpoint.raster_roles) != expected_roles:
            raise ReferenceChainReadError("Source artifact roles differ.")
        _same(metadata.get("array_origins"), expected_origins, "source artifact origins")
        _same(checkpoint.execution.status.value, "empty" if empty else "available", "source execution status")
        _same(call["status"], "empty" if empty else "executed", "source call status")
    _poll(cancel)
    return VerifiedReferenceSeedChain(manifest_sha256, record, checkpoints, retained)


__all__ = ["ReferenceChainReadError", "VerifiedReferenceSeedChain", "read_reference_seed_chain"]
