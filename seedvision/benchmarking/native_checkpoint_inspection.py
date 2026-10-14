"""Bounded inspection of native PlantCV-compatible stage checkpoints.

This reads archived pixels; it never reruns an operation or associates them
with the active desktop image. Verification requires the exact current native
code and recorded CUDA environment, keeping historical/stale artifacts visible
as unavailable for verification rather than silently substituting a backend.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from seedvision.benchmarking import native_mask_stage as masks
from seedvision.benchmarking import native_threshold_stage as thresholds
from seedvision.benchmarking import native_watershed_stage as watershed
from seedvision.benchmarking.checkpoint_inspection import (
    CheckpointInspection, InspectionStatus, _cancelled, _inventory, _json_same,
    _plain, _result, MAX_INSPECTED_METADATA_BYTES,
)
from seedvision.benchmarking.checkpoints import (
    CheckpointError, CheckpointIdentity, CheckpointLimits, ExecutionStatus,
    RasterRole, load_checkpoint,
)
from seedvision.benchmarking.recipes import (
    RecipeProvenanceMode, RecipeSpec, StageRunRecord, StageRunState,
    validate_stage_run,
)


# Each input/output load is bounded; the two retained pixel inventories together
# use at most twice this envelope. Only the bound arrays are exposed for display.
MAX_NATIVE_INSPECTED_ARRAY_BYTES = 32 * 1024 * 1024
MAX_NATIVE_INSPECTED_ARCHIVE_BYTES = 40 * 1024 * 1024


def _binding(operation: str):
    if operation in thresholds.SUPPORTED_NATIVE_THRESHOLD_OPERATIONS:
        return (
            thresholds, thresholds.native_threshold_identity,
            thresholds.expected_native_threshold_stage_identity,
            thresholds.NATIVE_THRESHOLD_STAGE_FORMAT,
            thresholds.NATIVE_THRESHOLD_STAGE_VERSION,
            "gray", "mask", RasterRole("image", np.dtype(np.uint8).str, "HW", (0, 255)),
            RasterRole("mask", np.dtype(np.uint8).str, "HW"),
        )
    if operation in masks.SUPPORTED_NATIVE_MASK_OPERATIONS:
        median = operation == "filter.median_blur"
        input_port = "mask" if operation == "morphology.fill" else "gray"
        role = RasterRole("image", np.dtype(np.uint8).str, "HW", (0, 255)) if median else RasterRole("mask", np.dtype(np.uint8).str, "HW")
        return (
            masks, masks.native_mask_identity, masks.expected_native_mask_stage_identity,
            masks.NATIVE_MASK_STAGE_FORMAT, masks.NATIVE_MASK_STAGE_VERSION,
            input_port, "image" if median else "mask", role, role,
        )
    if operation in watershed.SUPPORTED_NATIVE_WATERSHED_OPERATIONS:
        distance = operation == watershed.NATIVE_DISTANCE_OPERATION_ID
        return (
            watershed, watershed.native_watershed_identity,
            watershed.expected_native_watershed_stage_identity,
            watershed.NATIVE_WATERSHED_STAGE_FORMAT, watershed.NATIVE_WATERSHED_STAGE_VERSION,
            "mask" if distance else "peaks", "distance" if distance else "markers",
            RasterRole("mask", np.dtype(np.uint8).str, "HW") if distance else RasterRole("mask", np.dtype(np.bool_).str, "HW"),
            RasterRole("raster", np.dtype(np.float32).str, "HW") if distance else RasterRole("instances", np.dtype(np.int32).str, "HW"),
        )
    return None


def _effective_parameters(module, recipe, stage_id, operation):
    parameters = recipe.parameters_for_runner(stage_id)
    if module is thresholds:
        effective = dict(parameters)
        if operation == "threshold.mean":
            effective["offset"] = float(effective["offset"])
        # Original scalar Otsu cutoff equality is not promised by this bridge.
        return parameters, effective
    if module is masks:
        effective = dict(parameters)
        if operation == "filter.median_blur":
            effective["roi"] = None
        return parameters, effective
    if operation == watershed.NATIVE_DISTANCE_OPERATION_ID:
        return {}, {
            "distance_type": "DIST_L2", "requested_mask_size": 0,
            "effective_algorithm": "OpenCV 4.11.0 WithLabels 5x5 chamfer fixed-point sweep",
            "distance_unit": "pixel",
        }
    return {}, {"connectivity": 8, "background_label": 0}


def inspect_native_checkpoint(
    store: Path | str | None, recipe: RecipeSpec, record: StageRunRecord,
    *, cancel: Any = None,
) -> CheckpointInspection:
    """Verify one supported native stage and its exact archived input.

    No arrays are returned for an unbound/missing/stale/unsupported/cancelled
    chain. Current CUDA identity collection reads compact runtime/device facts;
    it does not calculate a raster. A different or absent runtime cannot verify
    historical execution as current. Scientific eligibility remains false.
    """
    if not isinstance(recipe, RecipeSpec) or not isinstance(record, StageRunRecord):
        raise TypeError("recipe and record must be immutable recipe contracts")
    if record.state is StageRunState.NOT_RUN:
        return _result(InspectionStatus.NOT_RUN, "This stage has not been recorded as executed.")
    if record.state not in {StageRunState.EXECUTED, StageRunState.EMPTY}:
        return _result(InspectionStatus.RECORDED, f"Recorded state is {record.state.value.upper()}; no output preview is available.")
    try:
        validate_stage_run(recipe, record)
        stage = recipe.stage(record.stage_id)
    except (TypeError, ValueError, KeyError) as error:
        return _result(InspectionStatus.MISMATCH, f"Stage run differs from the exact selected recipe: {error}")
    if recipe.provenance.mode is not RecipeProvenanceMode.NATIVE_COMPATIBLE or recipe.provenance.reference_id != "plantcv":
        return _result(InspectionStatus.UNSUPPORTED, "This verifier requires an explicit native-compatible PlantCV recipe.")
    binding = _binding(stage.operation_id)
    if binding is None:
        return _result(InspectionStatus.UNSUPPORTED, "No native artifact verifier is defined for this operation.")
    if not record.upstream_checkpoint_sha256 or not record.checkpoint_sha256:
        return _result(InspectionStatus.MISMATCH, "The historical record omits its exact input/output checkpoint address.")
    if store is None:
        return _result(InspectionStatus.STORE_UNBOUND, "Select a checkpoint store to inspect the native recorded digest.")
    root = Path(store).expanduser()
    addresses = (record.checkpoint_sha256, record.upstream_checkpoint_sha256)
    if not root.is_dir() or any((root/f"{digest}.checkpoint").is_symlink() or not (root/f"{digest}.checkpoint").is_dir() for digest in addresses):
        return _result(InspectionStatus.MISSING, "The selected store does not contain both native output and upstream artifacts.")
    limits = CheckpointLimits(
        max_array_bytes=MAX_NATIVE_INSPECTED_ARRAY_BYTES,
        max_total_array_bytes=MAX_NATIVE_INSPECTED_ARRAY_BYTES,
        max_npz_bytes=MAX_NATIVE_INSPECTED_ARCHIVE_BYTES,
        max_metadata_bytes=MAX_INSPECTED_METADATA_BYTES, max_arrays=8, max_dimensions=8,
    )
    module, identity_factory, output_identity_factory, artifact_format, version, input_port, output_name, input_role, output_role = binding
    try:
        if _cancelled(cancel):
            return _result(InspectionStatus.CANCELLED, "Native inspection was cancelled before loading.")
        output = load_checkpoint(root, record.checkpoint_sha256, limits=limits)
        metadata = output.metadata
        if _cancelled(cancel):
            return _result(InspectionStatus.CANCELLED, "Native inspection was cancelled after loading output.")
        payload = metadata.get("input_checkpoint_identity")
        if not isinstance(payload, Mapping):
            raise CheckpointError("Output omits its exact input checkpoint identity.")
        input_identity = CheckpointIdentity.from_dict(_plain(payload))
        upstream = load_checkpoint(root, record.upstream_checkpoint_sha256, expected_identity=input_identity, limits=limits)
        if _cancelled(cancel):
            return _result(InspectionStatus.CANCELLED, "Native inspection was cancelled after loading input.")
        implementation = metadata.get("implementation_identity")
        if not isinstance(implementation, Mapping) or type(implementation.get("device_index")) is not int or implementation["device_index"] < 0:
            raise CheckpointError("Output omits a valid native device/implementation identity.")
        try:
            native_identity = identity_factory(torch.device("cuda", implementation["device_index"]))
        except (thresholds.NativeThresholdUnavailableError, masks.NativeMaskUnavailableError, watershed.NativeWatershedUnavailableError) as error:
            return _result(InspectionStatus.UNSUPPORTED, f"The recorded native CUDA environment cannot be verified here: {error}")
        if not _json_same(implementation, native_identity.to_dict()):
            raise CheckpointError("Recorded native code or CUDA environment differs from the current implementation.")
        if _cancelled(cancel):
            return _result(InspectionStatus.CANCELLED, "Native inspection was cancelled during identity verification.")
        expected_identity = output_identity_factory(input_identity, recipe, stage.stage_id, record.upstream_checkpoint_sha256, native_identity)
        if output.identity != expected_identity:
            raise CheckpointError("Output identity differs from the exact native recipe/source/frame/settings/input contract.")
        expected_state = ExecutionStatus.EMPTY if record.state is StageRunState.EMPTY else ExecutionStatus.AVAILABLE
        if output.execution.status is not expected_state or upstream.execution.status not in {ExecutionStatus.AVAILABLE, ExecutionStatus.EMPTY}:
            raise CheckpointError("Input/output availability differs from the recorded stage state.")
        if len(stage.source_mappings) != 1:
            raise CheckpointError("The stage does not bind exactly one pinned source mapping.")
        parameters, effective = _effective_parameters(module, recipe, stage.stage_id, stage.operation_id)
        checks = {
            "format": artifact_format, "version": version, "stage_id": stage.stage_id,
            "operation_id": stage.operation_id, "execution_mode": RecipeProvenanceMode.NATIVE_COMPATIBLE.value,
            "recipe_identity": recipe.identity.to_dict(), "source_mapping": stage.source_mappings[0].to_dict(),
            "source_sha256": input_identity.source_sha256, "coordinate_frame": input_identity.coordinate_frame.to_dict(),
            "upstream_checkpoint_sha256": record.upstream_checkpoint_sha256,
            "input_checkpoint_identity": input_identity.to_dict(),
            "input_raster_roles": {input_port: input_role.to_dict()},
            "output_raster_roles": {output_name: output_role.to_dict()},
            "selected_parameters": parameters, "effective_parameters": effective,
            "native_supported_domain": module.native_supported_domain(stage.operation_id),
            "scientific_validation_status": "not_certified", "reference_comparison_status": "not_evaluated",
            "measurement_contract": output.measurement.to_dict(),
        }
        for key, expected in checks.items():
            if not _json_same(metadata.get(key), expected):
                raise CheckpointError(f"Native metadata field {key!r} differs from the exact stage contract.")
        if output.measurement.eligible:
            raise CheckpointError("This native stage cannot certify scientific measurement eligibility.")
        bindings = metadata.get("input_bindings")
        if not isinstance(bindings, Mapping) or set(bindings) != {input_port} or not isinstance(bindings[input_port], str):
            raise CheckpointError("Native output does not bind the exact selected input port.")
        array_name = bindings[input_port]
        if array_name not in upstream.arrays or upstream.raster_roles.get(array_name) != input_role:
            raise CheckpointError("The bound upstream pixels do not have the declared native input role.")
        if set(output.arrays) != {output_name} or output.raster_roles.get(output_name) != output_role:
            raise CheckpointError("Native output does not have the exact declared raster role.")
        image, result = upstream.arrays[array_name], output.arrays[output_name]
        shape = input_identity.coordinate_frame.shape
        if image.shape != shape or result.shape != shape or image.dtype.str != input_role.dtype or result.dtype.str != output_role.dtype:
            raise CheckpointError("Native pixel shapes/dtypes differ from the bound coordinate frame.")
        if input_role.kind == "mask" and image.dtype == np.uint8 and not np.isin(image, (0, 255)).all():
            raise CheckpointError("Native input mask is outside the uint8 0/255 domain.")
        if output_name == "mask":
            if not np.isin(result, (0, 255)).all() or (not np.any(result)) != (expected_state is ExecutionStatus.EMPTY):
                raise CheckpointError("Native mask pixels differ from their binary/state contract.")
        elif output_name == "image":
            if expected_state is not ExecutionStatus.AVAILABLE:
                raise CheckpointError("Median image output is available even when every pixel is zero.")
        elif output_name == "distance":
            if expected_state is not ExecutionStatus.AVAILABLE or not np.isfinite(result).all() or np.any(result < 0) or np.any(result[image == 0] != 0):
                raise CheckpointError("Native distance pixels violate the finite nonnegative background contract.")
        else:
            if np.any(result < 0) or not np.array_equal(result > 0, image) or (not np.any(result)) != (expected_state is ExecutionStatus.EMPTY):
                raise CheckpointError("Native marker pixels differ from the peak support/state contract.")
            labels = np.unique(result[result > 0])
            if not np.array_equal(labels, np.arange(1, len(labels)+1)) or type(metadata.get("marker_count")) is not int or metadata["marker_count"] != len(labels):
                raise CheckpointError("Native marker label inventory differs from the recorded component count.")
        inventory = [{"name": output_name, "dtype": result.dtype.str, "shape": list(result.shape), "nbytes": int(result.nbytes), "device": native_identity.device}]
        if not _json_same(metadata.get("output_inventory"), inventory):
            raise CheckpointError("Native output inventory differs from the verified raster.")
        for key in ("input_transfer_runtime_ns", "computation_runtime_ns"):
            if type(metadata.get(key)) is not int or metadata[key] < 0:
                raise CheckpointError("Native measured runtime is not a nonnegative integer.")
        if _cancelled(cancel):
            return _result(InspectionStatus.CANCELLED, "Native inspection was cancelled before preview publication.")
        compact = {key: _plain(metadata[key]) for key in ("operation_id", "selected_parameters", "effective_parameters", "native_supported_domain", "input_transfer_runtime_ns", "computation_runtime_ns", "scientific_validation_status", "reference_comparison_status", "marker_count") if key in metadata}
        compact["verified_native_implementation_identity"] = native_identity.to_dict()
        return CheckpointInspection(
            status=InspectionStatus.VERIFIED,
            detail="Archived native pixels, exact recipe/settings/source/frame/input identity, roles and current code/CUDA environment were verified. This does not rerun the calculation, prove reference equality, or independently match the active desktop image.",
            checkpoint_sha256=output.checkpoint_sha256, source_sha256=output.identity.source_sha256,
            coordinate_frame=output.identity.coordinate_frame, execution_status=output.execution.status,
            array_inventory=_inventory(upstream, "upstream")+_inventory(output, "output"),
            compact_metadata=compact, preview_arrays={f"input.{input_port}": image, f"output.{output_name}": result},
        )
    except (CheckpointError, TypeError, ValueError, KeyError, OSError, OverflowError) as error:
        return _result(InspectionStatus.MISMATCH, f"Native checkpoint contract failed verification: {error}")


__all__ = ["inspect_native_checkpoint", "MAX_NATIVE_INSPECTED_ARRAY_BYTES", "MAX_NATIVE_INSPECTED_ARCHIVE_BYTES"]
