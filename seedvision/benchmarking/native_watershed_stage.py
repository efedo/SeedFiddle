"""One-stage CUDA connected-marker bridge for PlantCV watershed recipes.

This adapter binds the pinned internal connected-marker recipe stage to the
existing CUDA 8-connected component implementation. The public PlantCV
watershed call does not publish this intermediate; native marker checkpoints
are explicitly distinct from original-reference outputs.
"""

from __future__ import annotations

from dataclasses import dataclass, fields as dataclass_fields, is_dataclass, replace
from enum import StrEnum
from functools import lru_cache
from hashlib import sha256
import json
import math
from pathlib import Path
import platform as host_platform
import re
import sys
import time
from types import CodeType
from typing import Any, Mapping

import numpy as np

from seedvision.benchmarking import checkpoints as _checkpoint_module
from seedvision.benchmarking import native_threshold_stage as _identity_tools
from seedvision.benchmarking.checkpoints import (
    CheckpointCancelled,
    CheckpointError,
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
from seedvision.benchmarking.recipe_catalogue import RecipeCatalogueError, build_recipe_catalogue
from seedvision.benchmarking.recipes import (
    RecipeContractError,
    RecipeIdentity,
    RecipeProvenance,
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
    PINNED_PLANTCV_COMMIT,
    PINNED_PLANTCV_VERSION,
    PINNED_REQUIREMENTS_SHA256,
    PINNED_SOURCE_AUDIT_SHA256,
)
from seedvision.cuda import ops as _ops_module
from seedvision.cuda import plantcv_distance as _distance_module
from seedvision.cuda.plantcv_distance import (
    DistanceCancelledError,
    DistanceResourceError,
    distance_map,
)
from seedvision.cuda.ops import connected_components


NATIVE_WATERSHED_STAGE_FORMAT = "seedfiddle.native-watershed-stage-output"
NATIVE_WATERSHED_STAGE_VERSION = 1
NATIVE_WATERSHED_BACKEND_ID = "seedvision.cuda.watershed_stages"
NATIVE_WATERSHED_BACKEND_VERSION = "1"
MAX_NATIVE_WATERSHED_METADATA_BYTES = 1024 * 1024
# Bound the compact per-component CPU statistics; this is not a GPU scratch-memory cap.
MAX_NATIVE_MARKER_PEAK_PIXELS = 2_000_000
NATIVE_MARKER_OPERATION_ID = "reference.internal.watershed.connected_markers"
NATIVE_MARKER_OUTPUT_NAME = "markers"
NATIVE_MARKER_INPUT_PORT = "peaks"
NATIVE_DISTANCE_OPERATION_ID = "reference.internal.watershed.distance_map"
NATIVE_DISTANCE_OUTPUT_NAME = "distance"
NATIVE_DISTANCE_INPUT_PORT = "mask"
SUPPORTED_NATIVE_WATERSHED_OPERATIONS = frozenset({
    NATIVE_DISTANCE_OPERATION_ID,
    NATIVE_MARKER_OPERATION_ID,
})
_INPUT_PORTS = {
    NATIVE_DISTANCE_OPERATION_ID: NATIVE_DISTANCE_INPUT_PORT,
    NATIVE_MARKER_OPERATION_ID: NATIVE_MARKER_INPUT_PORT,
}
_OUTPUT_NAMES = {
    NATIVE_DISTANCE_OPERATION_ID: NATIVE_DISTANCE_OUTPUT_NAME,
    NATIVE_MARKER_OPERATION_ID: NATIVE_MARKER_OUTPUT_NAME,
}
_CANONICAL_RECIPE_IDS = frozenset({
    "plantcv_seed_binary", "plantcv_seed_mean", "plantcv_seed_otsu",
})
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_CAPABILITY = re.compile(r"[0-9]+\.[0-9]+\Z")
_NATIVE_SOURCE_CONSTANTS = frozenset({
    "NATIVE_WATERSHED_STAGE_FORMAT", "NATIVE_WATERSHED_STAGE_VERSION",
    "NATIVE_WATERSHED_BACKEND_ID", "NATIVE_WATERSHED_BACKEND_VERSION",
    "MAX_NATIVE_WATERSHED_METADATA_BYTES", "MAX_NATIVE_MARKER_PEAK_PIXELS",
    "NATIVE_MARKER_OPERATION_ID", "NATIVE_MARKER_OUTPUT_NAME", "NATIVE_MARKER_INPUT_PORT",
    "NATIVE_DISTANCE_OPERATION_ID", "NATIVE_DISTANCE_OUTPUT_NAME", "NATIVE_DISTANCE_INPUT_PORT",
    "SUPPORTED_NATIVE_WATERSHED_OPERATIONS", "_INPUT_PORTS", "_OUTPUT_NAMES",
    "_CANONICAL_RECIPE_IDS", "_SHA256", "_CAPABILITY",
})
_DISTANCE_FUNCTIONS = {
    name: getattr(_distance_module, name)
    for name in (
        "_validate_mask", "_check_cancel", "_scan_from_left", "_forward_sweep",
        "_backward_sweep", "_distance_map_fixed_point", "distance_map",
    )
}
_DISTANCE_ERRORS = {
    "DistanceCancelledError": _distance_module.DistanceCancelledError,
    "DistanceResourceError": _distance_module.DistanceResourceError,
}
_DISTANCE_RUNTIME_DEPENDENCIES = {
    "torch": _distance_module.torch,
    "gcd": _distance_module.gcd,
}
_CHECKPOINT_ARCHIVE_ESTIMATOR = _checkpoint_module._estimate_npz_size
_CHECKPOINT_RUNTIME_DEPENDENCIES = {
    "np": _checkpoint_module.np,
    "BytesIO": _checkpoint_module.BytesIO,
    "CheckpointError": _checkpoint_module.CheckpointError,
}


class NativeWatershedStageError(ValueError):
    """A native marker stage or immutable binding is invalid."""


class NativeWatershedUnavailableError(RuntimeError):
    """The explicitly requested CUDA device/runtime is unavailable."""


class NativeWatershedStageStatus(StrEnum):
    """Stable stage states, keeping empty and cancellation explicit."""

    EXECUTED = "executed"
    EMPTY = "empty"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNAVAILABLE = "unavailable"
    UNSUPPORTED = "unsupported"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True, slots=True)
class NativeWatershedIdentity:
    """Exact loaded marker implementation and CUDA environment identity."""

    implementation_sha256: str
    reference_source_sha256: str
    python_version: str
    torch_version: str
    numpy_version: str
    cuda_runtime_version: str
    platform: str
    machine: str
    device_index: int
    device_name: str
    compute_capability: str

    def __post_init__(self) -> None:
        for name in ("implementation_sha256", "reference_source_sha256"):
            value = getattr(self, name)
            if not isinstance(value, str) or not _SHA256.fullmatch(value):
                raise NativeWatershedStageError(f"{name} must be a lowercase SHA-256 digest.")
        for name in (
            "python_version", "torch_version", "numpy_version", "cuda_runtime_version",
            "platform", "machine", "device_name",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or value != value.strip():
                raise NativeWatershedStageError(f"{name} must be non-empty trimmed text.")
        if type(self.device_index) is not int or self.device_index < 0:
            raise NativeWatershedStageError("device_index must be a non-negative integer.")
        if not isinstance(self.compute_capability, str) or not _CAPABILITY.fullmatch(self.compute_capability):
            raise NativeWatershedStageError("compute_capability must be a major.minor string.")

    @property
    def backend_id(self) -> str:
        return NATIVE_WATERSHED_BACKEND_ID

    @property
    def backend_version(self) -> str:
        return NATIVE_WATERSHED_BACKEND_VERSION

    @property
    def device(self) -> str:
        return f"cuda:{self.device_index}"

    @property
    def environment_versions(self) -> Mapping[str, str]:
        return {
            "python": self.python_version,
            "torch": self.torch_version,
            "numpy": self.numpy_version,
            "cuda_runtime": self.cuda_runtime_version,
            "platform": self.platform,
            "machine": self.machine,
            "cuda_device": self.device_name,
            "cuda_device_index": str(self.device_index),
            "cuda_compute_capability": self.compute_capability,
            "native_implementation_sha256": self.implementation_sha256,
            "plantcv_watershed_source_sha256": self.reference_source_sha256,
            "cuda_distance_primitive_source_sha256": _distance_module.PLANTCV_WATERSHED_SOURCE_SHA256,
            "opencv_reference": _distance_module.OPENCV_REFERENCE_VERSION,
            "opencv_reference_wheel": _distance_module.OPENCV_REFERENCE_WHEEL,
            "plantcv": PINNED_PLANTCV_VERSION,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend_id": self.backend_id,
            "backend_version": self.backend_version,
            "implementation_sha256": self.implementation_sha256,
            "reference_source_sha256": self.reference_source_sha256,
            "python_version": self.python_version,
            "torch_version": self.torch_version,
            "numpy_version": self.numpy_version,
            "cuda_runtime_version": self.cuda_runtime_version,
            "platform": self.platform,
            "machine": self.machine,
            "device_index": self.device_index,
            "device_name": self.device_name,
            "compute_capability": self.compute_capability,
            "environment_versions": dict(self.environment_versions),
        }

    def recipe_provenance(self) -> RecipeProvenance:
        return RecipeProvenance(
            mode=RecipeProvenanceMode.NATIVE_COMPATIBLE,
            backend_id=self.backend_id,
            backend_version=self.backend_version,
            reference_id="plantcv",
            reference_version=PINNED_PLANTCV_VERSION,
            reference_commit=PINNED_PLANTCV_COMMIT,
            source_manifest_sha256=PINNED_SOURCE_AUDIT_SHA256,
            environment_lock_sha256=PINNED_REQUIREMENTS_SHA256,
            environment_versions=dict(self.environment_versions),
            evidence_status=SourceResolution.PINNED,
        )


@dataclass(frozen=True, slots=True)
class NativeWatershedStageOutcome:
    """Marker execution state plus immutable output checkpoint, when present."""

    status: str
    run_record: StageRunRecord
    checkpoint_reference: CheckpointReference | None

    def __post_init__(self) -> None:
        if self.status not in {
            NativeWatershedStageStatus.EXECUTED, NativeWatershedStageStatus.EMPTY,
            NativeWatershedStageStatus.FAILED, NativeWatershedStageStatus.CANCELLED,
            NativeWatershedStageStatus.UNAVAILABLE, NativeWatershedStageStatus.UNSUPPORTED,
            NativeWatershedStageStatus.UNRESOLVED,
        }:
            raise ValueError("status is not a supported native watershed-stage outcome.")
        if not isinstance(self.run_record, StageRunRecord):
            raise TypeError("run_record must be a StageRunRecord.")
        completed = self.status in {NativeWatershedStageStatus.EXECUTED, NativeWatershedStageStatus.EMPTY}
        if completed != (self.checkpoint_reference is not None):
            raise ValueError("Only executed or empty native stages carry a checkpoint reference.")
        unresolved_states = {StageRunState.UNRESOLVED, StageRunState.UNAVAILABLE}
        if self.status is NativeWatershedStageStatus.UNRESOLVED:
            state_matches = self.run_record.state in unresolved_states
        else:
            expected_state = {
                NativeWatershedStageStatus.EXECUTED: StageRunState.EXECUTED,
                NativeWatershedStageStatus.EMPTY: StageRunState.EMPTY,
                NativeWatershedStageStatus.FAILED: StageRunState.FAILED,
                NativeWatershedStageStatus.CANCELLED: StageRunState.FAILED,
                NativeWatershedStageStatus.UNAVAILABLE: StageRunState.UNAVAILABLE,
                NativeWatershedStageStatus.UNSUPPORTED: StageRunState.UNSUPPORTED,
            }[self.status]
            state_matches = self.run_record.state is expected_state
        if not state_matches:
            raise ValueError("Outcome status and immutable stage-run state differ.")
        if self.checkpoint_reference is not None and (
            self.run_record.checkpoint_sha256 != self.checkpoint_reference.checkpoint_sha256
        ):
            raise ValueError("Native marker run and checkpoint reference hashes differ.")


def _module_source_record(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Reuse the checked source recorder without duplicating its AST logic."""
    try:
        return _identity_tools._module_source_record(*args, **kwargs)
    except _identity_tools.NativeThresholdStageError as error:
        raise NativeWatershedStageError(f"Native watershed implementation identity failed: {error}") from error


_IDENTITY_HELPERS = {
    name: getattr(_identity_tools, name)
    for name in (
        "_literal_node", "_normalize_identity_value", "_normalize_code_constant",
        "_code_payload", "_source_callable_nodes", "_verify_function_defaults",
        "_read_source_bounded", "_module_source_record",
    )
}


def native_supported_domain(operation_id: str) -> str:
    """Canonical archived support declaration, shared with inspection."""
    if operation_id == NATIVE_DISTANCE_OPERATION_ID:
        return (
            "uint8 0/255 HxW BinaryMask; CUDA float32 PlantCV/OpenCV-compatible 5x5 chamfer distance map; "
            f"full input frame; at most {_distance_module.MAX_RASTER_PIXELS} pixels"
        )
    if operation_id == NATIVE_MARKER_OPERATION_ID:
        return (
            "boolean HxW PeakMask; CUDA 8-connected components; int32 MarkerLabels; "
            f"full input frame; at most {MAX_NATIVE_MARKER_PEAK_PIXELS} true peak pixels; "
            "compact per-component statistics are the only compute-side host values"
        )
    raise KeyError(operation_id)


def _implementation_digest() -> str:
    """Hash bridge source and verify loaded CUDA callable against its source."""
    import torch

    for name, expected in _IDENTITY_HELPERS.items():
        if getattr(_identity_tools, name, None) is not expected:
            raise NativeWatershedStageError("Shared loaded-code identity helpers changed after import.")
    if connected_components is not _ops_module.connected_components:
        raise NativeWatershedStageError("Imported CUDA connected-components callable changed after import.")
    if _ops_module.np is not np:
        raise NativeWatershedStageError("CUDA ops NumPy dependency changed after import.")
    if (
        _distance_module.torch is not torch
        or _distance_module.torch is not _DISTANCE_RUNTIME_DEPENDENCIES["torch"]
    ):
        raise NativeWatershedStageError("CUDA distance PyTorch dependency changed after import.")
    if _distance_module.gcd is not _DISTANCE_RUNTIME_DEPENDENCIES["gcd"]:
        raise NativeWatershedStageError("CUDA distance gcd dependency changed after import.")
    if _checkpoint_module._estimate_npz_size is not _CHECKPOINT_ARCHIVE_ESTIMATOR:
        raise NativeWatershedStageError("Shared checkpoint archive estimator changed after import.")
    if any(
        getattr(_checkpoint_module, name, None) is not expected
        for name, expected in _CHECKPOINT_RUNTIME_DEPENDENCIES.items()
    ):
        raise NativeWatershedStageError("Shared checkpoint archive estimator dependency changed after import.")

    bridge_namespace = globals()
    for cls in (NativeWatershedIdentity, NativeWatershedStageOutcome):
        for name in (
            "__post_init__", "backend_id", "backend_version", "device",
            "environment_versions", "to_dict", "recipe_provenance",
        ):
            if hasattr(cls, name):
                bridge_namespace = dict(bridge_namespace)
                bridge_namespace[f"{cls.__name__}.{name}"] = getattr(cls, name)
    bridge_names = (
        "_module_source_record", "_implementation_digest", "native_watershed_identity",
        "_canonical_catalogue", "_canonical_recipe_and_stage", "_validate_recipe_stage",
        "_reference_source_sha256", "_validate_input_identity", "_strict_metadata",
        "_cancelled", "_outcome_without_checkpoint", "_validate_peak_mask",
        "_validate_binary_mask", "_validate_component_output", "_validate_distance_output",
        "_preflight_output_archive",
        "expected_native_watershed_stage_identity",
        "run_native_watershed_stage", "native_supported_domain",
        "NativeWatershedIdentity.__post_init__", "NativeWatershedIdentity.backend_id",
        "NativeWatershedIdentity.backend_version", "NativeWatershedIdentity.device",
        "NativeWatershedIdentity.environment_versions", "NativeWatershedIdentity.to_dict",
        "NativeWatershedIdentity.recipe_provenance", "NativeWatershedStageOutcome.__post_init__",
    )
    helper_names = (
        "_literal_node", "_normalize_identity_value", "_normalize_code_constant",
        "_code_payload", "_source_callable_nodes", "_verify_function_defaults",
        "_read_source_bounded", "_module_source_record",
    )
    helper_record = _module_source_record(
        "native_threshold_stage.py", Path(_identity_tools.__file__).resolve(), vars(_identity_tools),
        helper_names, frozenset({"MAX_IMPLEMENTATION_SOURCE_BYTES"}),
    )
    ops_record = _module_source_record(
        "ops.py", Path(_ops_module.__file__).resolve(), vars(_ops_module),
        ("connected_components",), frozenset(),
    )
    distance_names = (
        "_validate_mask", "_check_cancel", "_scan_from_left", "_forward_sweep",
        "_backward_sweep", "_distance_map_fixed_point", "distance_map",
    )
    distance_constants = frozenset({
        "PLANTCV_VERSION", "PLANTCV_COMMIT", "PLANTCV_WATERSHED_SOURCE_SHA256",
        "OPENCV_REFERENCE_VERSION", "OPENCV_REFERENCE_WHEEL", "MAX_RASTER_PIXELS",
        "SCAN_CHUNK_PIXELS", "_DIST_SHIFT", "_HV_COST", "_DIAGONAL_COST",
        "_LONG_COST", "_UINT32_MAX", "_DIST_MAX", "_COST_SIGN_BIT",
        "_INF_ENCODED", "_ZERO_ENCODED", "_DISTANCE_SCALE",
    })
    if any(getattr(_distance_module, name, None) is not expected for name, expected in _DISTANCE_FUNCTIONS.items()):
        raise NativeWatershedStageError("A loaded CUDA watershed distance helper changed after import.")
    if any(getattr(_distance_module, name, None) is not expected for name, expected in _DISTANCE_ERRORS.items()):
        raise NativeWatershedStageError("A loaded CUDA distance exception type changed after import.")
    if distance_map is not _distance_module.distance_map:
        raise NativeWatershedStageError("Imported CUDA watershed distance callable changed after import.")
    if DistanceCancelledError is not _distance_module.DistanceCancelledError:
        raise NativeWatershedStageError("Imported distance cancellation type changed after import.")
    if DistanceResourceError is not _distance_module.DistanceResourceError:
        raise NativeWatershedStageError("Imported distance resource error type changed after import.")
    distance_record = _module_source_record(
        "plantcv_distance.py", Path(_distance_module.__file__).resolve(), vars(_distance_module),
        distance_names, distance_constants,
    )
    checkpoint_record = _module_source_record(
        "checkpoints.py", Path(_checkpoint_module.__file__).resolve(), vars(_checkpoint_module),
        ("_estimate_npz_size",), frozenset(),
    )
    bridge_record = _module_source_record(
        "native_watershed_stage.py", Path(__file__).resolve(), bridge_namespace, bridge_names,
        _NATIVE_SOURCE_CONSTANTS, dataclass_names=("NativeWatershedIdentity", "NativeWatershedStageOutcome"),
    )
    payload = {
        "format": "seedfiddle.native-watershed-loaded-implementation",
        "version": 1,
        "status_values": [item.value for item in NativeWatershedStageStatus],
        "files": [helper_record, ops_record, distance_record, checkpoint_record, bridge_record],
    }
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


@lru_cache(maxsize=1)
def _canonical_catalogue():
    try:
        return build_recipe_catalogue()
    except (RecipeCatalogueError, OSError, ValueError) as error:
        raise NativeWatershedStageError(f"Committed PlantCV recipe contracts are unavailable: {error}") from error


def _canonical_recipe_and_stage(
    recipe_id: str,
    stage_id: str | None = None,
    operation_id: str = NATIVE_MARKER_OPERATION_ID,
):
    if recipe_id not in _CANONICAL_RECIPE_IDS:
        raise NativeWatershedStageError(f"Recipe {recipe_id!r} is outside the pinned PlantCV seed recipes.")
    recipe = _canonical_catalogue().recipe(recipe_id)
    if operation_id not in SUPPORTED_NATIVE_WATERSHED_OPERATIONS:
        raise NativeWatershedStageError(f"Operation {operation_id!r} is outside this native watershed bridge.")
    matches = tuple(stage for stage in recipe.stages if stage.operation_id == operation_id)
    if len(matches) != 1:
        raise NativeWatershedStageError(f"Pinned recipe does not contain exactly one {operation_id!r} stage.")
    stage = matches[0]
    if stage_id is not None and stage_id != stage.stage_id:
        raise NativeWatershedStageError("Requested stage ID does not match the canonical watershed stage.")
    return recipe, stage


def _reference_source_sha256() -> str:
    digests = {
        _canonical_recipe_and_stage(recipe_id, operation_id=operation)[1].source_mappings[0].source_sha256
        for recipe_id in sorted(_CANONICAL_RECIPE_IDS)
        for operation in sorted(SUPPORTED_NATIVE_WATERSHED_OPERATIONS)
    }
    if len(digests) != 1:
        raise NativeWatershedStageError("Pinned watershed recipe stages disagree on the PlantCV source digest.")
    digest = next(iter(digests))
    if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
        raise NativeWatershedStageError("Pinned PlantCV watershed source mapping has no SHA-256 digest.")
    return digest


def native_watershed_identity(device: Any) -> NativeWatershedIdentity:
    """Describe the exact native distance-map and marker implementation for a device."""
    import torch

    if type(device) is not torch.device or device.type != "cuda" or device.index is None:
        raise NativeWatershedStageError("device must be an explicit torch.device such as cuda:0.")
    if not torch.cuda.is_available():
        raise NativeWatershedUnavailableError("CUDA is not available in this PyTorch runtime.")
    if device.index < 0 or device.index >= torch.cuda.device_count():
        raise NativeWatershedUnavailableError(f"Requested CUDA device {device} is not available.")
    try:
        properties = torch.cuda.get_device_properties(device)
        runtime = torch.version.cuda or torch.version.hip
        if not runtime:
            raise NativeWatershedUnavailableError("PyTorch reports no CUDA or HIP runtime version.")
        return NativeWatershedIdentity(
            implementation_sha256=_implementation_digest(),
            reference_source_sha256=_reference_source_sha256(),
            python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            torch_version=str(torch.__version__),
            numpy_version=np.__version__,
            cuda_runtime_version=str(runtime),
            platform=sys.platform,
            machine=host_platform.machine() or "unknown",
            device_index=int(device.index),
            device_name=str(properties.name),
            compute_capability=f"{int(properties.major)}.{int(properties.minor)}",
        )
    except NativeWatershedUnavailableError:
        raise
    except NativeWatershedStageError:
        raise
    except Exception as error:
        raise NativeWatershedUnavailableError(f"Cannot inspect requested CUDA device {device}: {error}") from error


def _validate_recipe_stage(
    recipe: RecipeSpec,
    stage_id: str,
    implementation_identity: NativeWatershedIdentity,
) -> Any:
    try:
        selected_stage = recipe.stage(stage_id)
    except KeyError as error:
        raise NativeWatershedStageError(f"Recipe does not contain stage {stage_id!r}.") from error
    operation_id = selected_stage.operation_id
    canonical_recipe, canonical_stage = _canonical_recipe_and_stage(
        recipe.recipe_id, stage_id, operation_id,
    )
    if recipe.recipe_version != canonical_recipe.recipe_version:
        raise NativeWatershedStageError("Native recipe version differs from the pinned PlantCV seed recipe.")
    if recipe.provenance != implementation_identity.recipe_provenance():
        raise NativeWatershedStageError("Recipe provenance differs from the exact native marker identity.")
    if len(recipe.stages) != len(canonical_recipe.stages):
        raise NativeWatershedStageError("Recipe stage count differs from the pinned topology.")
    for current, canonical in zip(recipe.stages, canonical_recipe.stages, strict=True):
        if current.stage_id != canonical.stage_id:
            raise NativeWatershedStageError("Recipe stage order differs from the pinned topology.")
        if replace(current, parameters=canonical.parameters) != canonical:
            raise NativeWatershedStageError("Recipe stage contract or source mapping differs from the pinned catalogue.")
    if recipe.dependencies != canonical_recipe.dependencies:
        raise NativeWatershedStageError("Recipe dependencies differ from the pinned stage graph.")
    if canonical_stage.resolution is not SourceResolution.PINNED:
        raise NativeWatershedStageError("Watershed-stage source mapping is not pinned.")
    if canonical_stage.source_mappings[0].source_sha256 != implementation_identity.reference_source_sha256:
        raise NativeWatershedStageError("Recipe source mapping differs from the exact reference-source identity.")
    if recipe.parameters_for_runner(stage_id):
        raise NativeWatershedStageError("Connected-marker labeling has no calculation parameters.")
    return recipe.stage(stage_id)


def _validate_input_identity(
    identity: CheckpointIdentity,
    recipe: RecipeSpec,
    operation_id: str,
) -> None:
    canonical_recipe, _ = _canonical_recipe_and_stage(recipe.recipe_id, operation_id=operation_id)
    predecessor_operation = (
        "morphology.dilate"
        if operation_id == NATIVE_DISTANCE_OPERATION_ID
        else "reference.internal.watershed.local_maxima"
    )
    predecessor_stages = tuple(
        stage for stage in canonical_recipe.stages if stage.operation_id == predecessor_operation
    )
    if len(predecessor_stages) != 1 or identity.stage_id != predecessor_stages[0].stage_id:
        input_name = "BinaryMask" if operation_id == NATIVE_DISTANCE_OPERATION_ID else "PeakMask"
        raise NativeWatershedStageError(f"{input_name} input must be bound to the exact preceding recipe stage.")
    selected = recipe.identity
    if (
        identity.recipe_id != selected.recipe_id
        or identity.recipe_version != selected.recipe_version
        or identity.recipe_sha256 != selected.recipe_sha256
        or identity.graph_sha256 != selected.graph_sha256
        or identity.settings_sha256 != selected.settings_sha256
    ):
        raise NativeWatershedStageError(
            "Input checkpoint analytical recipe identity differs from the exact selected recipe and settings."
        )
    semantics = identity.mask_semantics
    if operation_id == NATIVE_DISTANCE_OPERATION_ID:
        if semantics.get("representation") != "uint8_0_255":
            raise NativeWatershedStageError("BinaryMask identity must declare uint8 0/255 pixels.")
    elif (
        semantics.get("representation") != "boolean"
        or semantics.get("foreground_polarity") != "true_peak_pixels"
    ):
        raise NativeWatershedStageError("PeakMask identity must declare boolean true-valued peak pixels.")


def expected_native_watershed_stage_identity(
    input_identity: CheckpointIdentity,
    recipe: RecipeSpec,
    stage_id: str,
    upstream_checkpoint_sha256: str,
    implementation_identity: NativeWatershedIdentity,
) -> CheckpointIdentity:
    """Derive the exact native intermediate identity from recipe, input and backend."""
    if not isinstance(input_identity, CheckpointIdentity) or not isinstance(recipe, RecipeSpec):
        raise TypeError("input_identity and recipe must be immutable checkpoint/recipe contracts.")
    if not isinstance(implementation_identity, NativeWatershedIdentity):
        raise TypeError("implementation_identity must be a NativeWatershedIdentity.")
    if not isinstance(upstream_checkpoint_sha256, str) or not _SHA256.fullmatch(upstream_checkpoint_sha256):
        raise NativeWatershedStageError("upstream_checkpoint_sha256 must be a lowercase SHA-256 digest.")
    stage = _validate_recipe_stage(recipe, stage_id, implementation_identity)
    operation_id = stage.operation_id
    _validate_input_identity(input_identity, recipe, operation_id)
    selected = recipe.identity
    output_identity = CheckpointIdentity(
        source_sha256=input_identity.source_sha256,
        coordinate_frame=input_identity.coordinate_frame,
        stage_id=stage_id,
        recipe_id=selected.recipe_id,
        recipe_version=selected.recipe_version,
        recipe_sha256=selected.recipe_sha256,
        graph_sha256=selected.graph_sha256,
        settings_sha256=selected.settings_sha256,
        backend_id=implementation_identity.backend_id,
        backend_version=implementation_identity.backend_version,
        environment_versions=dict(implementation_identity.environment_versions),
        image_semantics=dict(input_identity.image_semantics),
        mask_semantics=dict(input_identity.mask_semantics),
        instance_semantics=(
            {
                "label_dtype": np.dtype(np.int32).str,
                "background_label": 0,
                "label_id_mapping": "raster_order_8_connected_peak_components",
                "split_boundary_ownership": "not_applicable",
                "visibility": "peak_components_only",
                "eligibility": "not_scientifically_validated",
            }
            if operation_id == NATIVE_MARKER_OPERATION_ID
            else dict(input_identity.instance_semantics)
        ),
        upstream_checkpoint_sha256=upstream_checkpoint_sha256,
    )
    return output_identity


def _strict_metadata(payload: Mapping[str, Any], maximum: int) -> dict[str, Any]:
    try:
        encoded = json.dumps(
            payload, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError, RecursionError) as error:
        raise NativeWatershedStageError("Native watershed metadata must be strict finite JSON.") from error
    if len(encoded) > maximum:
        raise NativeWatershedStageError(f"Native watershed metadata exceeds its {maximum}-byte limit.")
    return json.loads(encoded.decode("utf-8"))


def _cancelled(cancel: Any) -> bool:
    if cancel is None:
        return False
    if callable(cancel):
        return bool(cancel())
    is_set = getattr(cancel, "is_set", None)
    if not callable(is_set):
        raise NativeWatershedStageError("cancel must be a callable or an event with is_set().")
    return bool(is_set())


def _outcome_without_checkpoint(
    recipe: RecipeSpec,
    stage_id: str,
    upstream_checkpoint_sha256: str,
    status: str,
    reason: str,
) -> NativeWatershedStageOutcome:
    state = {
        NativeWatershedStageStatus.FAILED: StageRunState.FAILED,
        NativeWatershedStageStatus.CANCELLED: StageRunState.FAILED,
        NativeWatershedStageStatus.UNAVAILABLE: StageRunState.UNAVAILABLE,
        NativeWatershedStageStatus.UNSUPPORTED: StageRunState.UNSUPPORTED,
        NativeWatershedStageStatus.UNRESOLVED: (
            StageRunState.UNRESOLVED
            if recipe.stage(stage_id).resolution is SourceResolution.UNRESOLVED
            else StageRunState.UNAVAILABLE
        ),
    }[status]
    compact_reason = " ".join(str(reason).split())[:512] or str(status)
    record = StageRunRecord(
        stage_id=stage_id,
        recipe_identity=recipe.identity,
        state=state,
        upstream_checkpoint_sha256=upstream_checkpoint_sha256,
        reason=compact_reason,
    )
    validate_stage_run(recipe, record, expected_upstream_checkpoint_sha256=upstream_checkpoint_sha256)
    return NativeWatershedStageOutcome(status, record, None)


def _validate_peak_mask(
    checkpoint: Any,
    input_bindings: Mapping[str, str],
    input_roles: Mapping[str, RasterRole],
    frame: CoordinateFrame,
) -> np.ndarray:
    if set(input_bindings) != {NATIVE_MARKER_INPUT_PORT} or set(input_roles) != {NATIVE_MARKER_INPUT_PORT}:
        raise NativeWatershedStageError("Stage requires one explicit PeakMask input binding and raster role.")
    array_name = input_bindings[NATIVE_MARKER_INPUT_PORT]
    if not isinstance(array_name, str) or not array_name or array_name not in checkpoint.arrays:
        raise NativeWatershedStageError("PeakMask binding does not name an input checkpoint array.")
    role = RasterRole("mask", np.dtype(np.bool_).str, "HW")
    if input_roles[NATIVE_MARKER_INPUT_PORT] != role or checkpoint.raster_roles.get(array_name) != role:
        raise NativeWatershedStageError("PeakMask must have the exact boolean HxW raster role.")
    array = checkpoint.arrays[array_name]
    if (
        not isinstance(array, np.ndarray) or array.dtype != np.dtype(np.bool_)
        or array.ndim != 2 or tuple(array.shape) != frame.shape or not array.flags.c_contiguous
        or array.nbytes > MAX_ARRAY_BYTES or array.size > MAX_ARRAY_ELEMENTS
    ):
        raise NativeWatershedStageError("PeakMask must be a bounded contiguous boolean HxW raster in the requested frame.")
    peak_count = int(np.count_nonzero(array))
    is_empty = peak_count == 0
    if checkpoint.execution.status not in {ExecutionStatus.EMPTY, ExecutionStatus.AVAILABLE}:
        raise NativeWatershedStageError("PeakMask input checkpoint must be available or empty.")
    if is_empty != (checkpoint.execution.status is ExecutionStatus.EMPTY):
        raise NativeWatershedStageError("PeakMask pixels disagree with the recorded empty/available state.")
    return array


def _validate_binary_mask(
    checkpoint: Any,
    input_bindings: Mapping[str, str],
    input_roles: Mapping[str, RasterRole],
    frame: CoordinateFrame,
) -> np.ndarray:
    if set(input_bindings) != {NATIVE_DISTANCE_INPUT_PORT} or set(input_roles) != {NATIVE_DISTANCE_INPUT_PORT}:
        raise NativeWatershedStageError("Distance-map stage requires one explicit BinaryMask input binding and role.")
    array_name = input_bindings[NATIVE_DISTANCE_INPUT_PORT]
    if not isinstance(array_name, str) or not array_name or array_name not in checkpoint.arrays:
        raise NativeWatershedStageError("BinaryMask binding does not name an input checkpoint array.")
    role = RasterRole("mask", np.dtype(np.uint8).str, "HW")
    if input_roles[NATIVE_DISTANCE_INPUT_PORT] != role or checkpoint.raster_roles.get(array_name) != role:
        raise NativeWatershedStageError("BinaryMask must have the exact uint8 HxW raster role.")
    array = checkpoint.arrays[array_name]
    if (
        not isinstance(array, np.ndarray) or array.dtype != np.dtype(np.uint8)
        or array.ndim != 2 or tuple(array.shape) != frame.shape or not array.flags.c_contiguous
        or array.nbytes > MAX_ARRAY_BYTES or array.size > MAX_ARRAY_ELEMENTS
    ):
        raise NativeWatershedStageError("BinaryMask must be a bounded contiguous uint8 HxW raster in the requested frame.")
    if array.size and not np.isin(array, (0, 255)).all():
        raise NativeWatershedStageError("BinaryMask contains values outside uint8 0/255.")
    is_empty = not bool(np.any(array))
    if checkpoint.execution.status not in {ExecutionStatus.EMPTY, ExecutionStatus.AVAILABLE}:
        raise NativeWatershedStageError("BinaryMask input checkpoint must be available or empty.")
    if is_empty != (checkpoint.execution.status is ExecutionStatus.EMPTY):
        raise NativeWatershedStageError("BinaryMask pixels disagree with the recorded empty/available state.")
    return array


def _validate_component_output(peaks: Any, labels: Any, marker_count: int, device: Any) -> bool:
    import torch

    if (
        not torch.is_tensor(labels) or labels.layout != torch.strided or labels.device != device
        or labels.dtype != torch.int32 or tuple(labels.shape) != (1, 1, *peaks.shape)
        or not labels.is_contiguous() or type(marker_count) is not int or marker_count < 0
    ):
        return False
    raster = labels[0, 0]
    foreground = peaks.bool()
    if bool(torch.any((raster > 0) != foreground).item()):
        return False
    if bool(torch.any(raster < 0).item()):
        return False
    maximum = int(raster.max().item()) if raster.numel() else 0
    if maximum != marker_count:
        return False
    height, width = foreground.shape
    for dy, dx in ((0, 1), (1, -1), (1, 0), (1, 1)):
        if dy == 0:
            y0, y1, y2, y3 = 0, height, 0, height
            if dx > 0:
                x0, x1, x2, x3 = 0, width - dx, dx, width
            else:
                x0, x1, x2, x3 = -dx, width, 0, width + dx
        else:
            y0, y1, y2, y3 = 0, height - dy, dy, height
            if dx > 0:
                x0, x1, x2, x3 = 0, width - dx, dx, width
            elif dx < 0:
                x0, x1, x2, x3 = -dx, width, 0, width + dx
            else:
                x0, x1, x2, x3 = 0, width, 0, width
        if y1 <= y0 or x1 <= x0:
            continue
        first_peaks = foreground[y0:y1, x0:x1]
        second_peaks = foreground[y2:y3, x2:x3]
        first_labels = raster[y0:y1, x0:x1]
        second_labels = raster[y2:y3, x2:x3]
        if bool(torch.any(first_peaks & second_peaks & (first_labels != second_labels)).item()):
            return False
    return True


def _validate_distance_output(mask: Any, distance: Any, device: Any) -> bool:
    import torch

    if (
        not torch.is_tensor(distance) or distance.layout != torch.strided or distance.device != device
        or distance.dtype != torch.float32 or tuple(distance.shape) != tuple(mask.shape)
        or not distance.is_contiguous()
    ):
        return False
    return bool(torch.isfinite(distance).all().item()) and not bool((distance < 0).any().item())


def _preflight_output_archive(
    output_name: str,
    shape: tuple[int, int],
    dtype: np.dtype,
    limits: CheckpointLimits,
) -> int:
    """Estimate the output NPZ size from its header and bytes before CUDA work."""
    nbytes = math.prod(shape) * dtype.itemsize
    try:
        estimated_bytes = _CHECKPOINT_ARCHIVE_ESTIMATOR({
            output_name: (shape, dtype, nbytes),
        })
    except CheckpointError as error:
        raise NativeWatershedStageError(f"Cannot preflight the watershed checkpoint archive: {error}") from error
    if type(estimated_bytes) is not int or estimated_bytes < 0:
        raise NativeWatershedStageError("Shared checkpoint archive estimator returned an invalid size.")
    if estimated_bytes > limits.max_npz_bytes:
        raise NativeWatershedStageError(
            f"Estimated watershed NPZ archive size {estimated_bytes} exceeds the "
            f"{limits.max_npz_bytes}-byte max_npz_bytes limit."
        )
    return estimated_bytes


def run_native_watershed_stage(
    input_root: Path | str,
    input_checkpoint_sha256: str,
    recipe: RecipeSpec,
    stage_id: str,
    output_root: Path | str,
    *,
    expected_source_sha256: str,
    expected_coordinate_frame: CoordinateFrame,
    expected_recipe_identity: RecipeIdentity,
    expected_upstream_checkpoint_sha256: str,
    expected_input_identity: CheckpointIdentity,
    expected_input_measurement: MeasurementContract,
    expected_output_identity: CheckpointIdentity | None,
    expected_implementation_identity: NativeWatershedIdentity,
    device: Any,
    input_bindings: Mapping[str, str],
    input_roles: Mapping[str, RasterRole],
    output_measurement: MeasurementContract,
    allow_cpu_transfer: bool,
    limits: CheckpointLimits = CheckpointLimits(),
    cancel: Any = None,
) -> NativeWatershedStageOutcome:
    """Execute one bound PlantCV watershed distance-map or marker stage on CUDA."""
    import torch

    if not isinstance(recipe, RecipeSpec) or not isinstance(expected_recipe_identity, RecipeIdentity):
        raise TypeError("recipe and expected_recipe_identity must be immutable recipe contracts.")
    if not isinstance(expected_implementation_identity, NativeWatershedIdentity):
        raise TypeError("expected_implementation_identity must be explicit.")
    if type(device) is not torch.device or device.type != "cuda" or device.index is None:
        raise NativeWatershedStageError("device must be an explicit torch.device such as cuda:0.")
    if not isinstance(expected_coordinate_frame, CoordinateFrame):
        raise TypeError("expected_coordinate_frame must be a CoordinateFrame.")
    if not isinstance(expected_input_identity, CheckpointIdentity) or not isinstance(expected_input_measurement, MeasurementContract):
        raise TypeError("Expected input identity and measurement contracts are required.")
    if expected_output_identity is not None and not isinstance(expected_output_identity, CheckpointIdentity):
        raise TypeError("expected_output_identity must be a CheckpointIdentity or None.")
    if not isinstance(output_measurement, MeasurementContract):
        raise TypeError("output_measurement must be an explicit MeasurementContract.")
    if output_measurement.eligible:
        raise NativeWatershedStageError("Native watershed checkpoints cannot mark scientific measurements eligible.")
    if not isinstance(limits, CheckpointLimits):
        raise TypeError("limits must be a CheckpointLimits contract.")
    if type(allow_cpu_transfer) is not bool or not allow_cpu_transfer:
        raise NativeWatershedStageError("Checkpoint serialization requires explicit allow_cpu_transfer=True.")
    if expected_recipe_identity != recipe.identity:
        raise NativeWatershedStageError("Recipe identity differs from the caller's exact expected identity.")
    if not isinstance(input_bindings, Mapping) or not isinstance(input_roles, Mapping):
        raise NativeWatershedStageError("Stage input binding and raster role must be explicit mappings.")
    if any(not isinstance(name, str) or not isinstance(value, str) for name, value in input_bindings.items()):
        raise NativeWatershedStageError("Input bindings must map port names to checkpoint array names.")
    if any(not isinstance(name, str) or not isinstance(role, RasterRole) for name, role in input_roles.items()):
        raise NativeWatershedStageError("Input roles must map port names to RasterRole contracts.")
    if cancel is not None and not callable(cancel) and not callable(getattr(cancel, "is_set", None)):
        raise NativeWatershedStageError("cancel must be a callable or an event with is_set().")
    for name, digest in (
        ("expected_source_sha256", expected_source_sha256),
        ("input_checkpoint_sha256", input_checkpoint_sha256),
        ("expected_upstream_checkpoint_sha256", expected_upstream_checkpoint_sha256),
    ):
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise NativeWatershedStageError(f"{name} must be a lowercase SHA-256 digest.")
    if expected_upstream_checkpoint_sha256 != input_checkpoint_sha256:
        raise NativeWatershedStageError("Expected upstream digest differs from the exact input checkpoint address.")
    if expected_input_identity.source_sha256 != expected_source_sha256:
        raise NativeWatershedStageError("Input checkpoint source SHA differs from the requested source.")
    if expected_input_identity.coordinate_frame != expected_coordinate_frame:
        raise NativeWatershedStageError("Input checkpoint frame differs from the requested coordinate frame.")
    if math.prod(expected_coordinate_frame.shape) > MAX_ARRAY_ELEMENTS:
        raise NativeWatershedStageError("Requested frame exceeds the bounded raster element domain.")
    try:
        stage = recipe.stage(stage_id)
    except KeyError as error:
        raise NativeWatershedStageError(f"Recipe does not contain stage {stage_id!r}.") from error
    operation_id = stage.operation_id
    output_dtype = (
        np.dtype(np.float32)
        if operation_id == NATIVE_DISTANCE_OPERATION_ID
        else np.dtype(np.int32)
    )
    output_bytes = math.prod(expected_coordinate_frame.shape) * output_dtype.itemsize
    if output_bytes > min(MAX_ARRAY_BYTES, limits.max_array_bytes, limits.max_total_array_bytes):
        output_label = "DistanceMap" if operation_id == NATIVE_DISTANCE_OPERATION_ID else "MarkerLabels"
        raise NativeWatershedStageError(f"{output_label} output exceeds the bounded checkpoint array domain.")

    if _cancelled(cancel):
        return _outcome_without_checkpoint(
            recipe, stage_id, input_checkpoint_sha256, NativeWatershedStageStatus.CANCELLED,
            "Cancelled before native watershed-stage validation.",
        )
    if recipe.provenance.mode is not RecipeProvenanceMode.NATIVE_COMPATIBLE:
        return _outcome_without_checkpoint(
            recipe, stage_id, input_checkpoint_sha256, NativeWatershedStageStatus.UNSUPPORTED,
            "Only explicitly native-compatible recipes can execute through this bridge.",
        )
    if recipe.provenance.evidence_status is SourceResolution.UNAVAILABLE or stage.resolution is SourceResolution.UNAVAILABLE:
        return _outcome_without_checkpoint(
            recipe, stage_id, input_checkpoint_sha256, NativeWatershedStageStatus.UNAVAILABLE,
            recipe.provenance.evidence_note or stage.resolution_note or "Pinned PlantCV source is unavailable.",
        )
    if recipe.provenance.evidence_status is SourceResolution.UNRESOLVED or stage.resolution is SourceResolution.UNRESOLVED:
        return _outcome_without_checkpoint(
            recipe, stage_id, input_checkpoint_sha256, NativeWatershedStageStatus.UNRESOLVED,
            recipe.provenance.evidence_note or stage.resolution_note or "PlantCV source resolution is unresolved.",
        )
    if operation_id not in SUPPORTED_NATIVE_WATERSHED_OPERATIONS:
        return _outcome_without_checkpoint(
            recipe, stage_id, input_checkpoint_sha256, NativeWatershedStageStatus.UNSUPPORTED,
            f"Operation {operation_id!r} is outside the native watershed bridge.",
        )
    try:
        _validate_recipe_stage(recipe, stage_id, expected_implementation_identity)
    except NativeWatershedStageError:
        raise
    try:
        _validate_input_identity(expected_input_identity, recipe, operation_id)
    except NativeWatershedStageError as error:
        input_kind = "BinaryMask" if operation_id == NATIVE_DISTANCE_OPERATION_ID else "PeakMask"
        raise NativeWatershedStageError(f"{input_kind} checkpoint identity is invalid: {error}") from error
    expected_input_port = _INPUT_PORTS[operation_id]
    if set(input_bindings) != {expected_input_port} or set(input_roles) != {expected_input_port}:
        raise NativeWatershedStageError("Stage input bindings differ from its exact recipe port topology.")
    output_name = _OUTPUT_NAMES[operation_id]
    _preflight_output_archive(output_name, expected_coordinate_frame.shape, output_dtype, limits)
    try:
        checkpoint = load_stage_input(
            input_root, input_checkpoint_sha256,
            expected_identity=expected_input_identity,
            expected_measurement=expected_input_measurement,
            limits=limits,
        )
    except Exception as error:
        raise NativeWatershedStageError(f"Exact watershed input checkpoint could not be loaded: {error}") from error
    if checkpoint.identity.source_sha256 != expected_source_sha256:
        raise NativeWatershedStageError("Loaded watershed input source SHA differs from the requested source.")
    if checkpoint.identity.coordinate_frame != expected_coordinate_frame:
        raise NativeWatershedStageError("Loaded watershed input frame differs from the requested coordinate frame.")
    if operation_id == NATIVE_DISTANCE_OPERATION_ID:
        input_array = _validate_binary_mask(checkpoint, input_bindings, input_roles, expected_coordinate_frame)
        peak_count = None
    else:
        input_array = _validate_peak_mask(checkpoint, input_bindings, input_roles, expected_coordinate_frame)
        peak_count = int(np.count_nonzero(input_array))
    if peak_count is not None and peak_count > MAX_NATIVE_MARKER_PEAK_PIXELS:
        return _outcome_without_checkpoint(
            recipe, stage_id, input_checkpoint_sha256, NativeWatershedStageStatus.UNSUPPORTED,
            f"PeakMask has {peak_count} true pixels; this bridge bounds compact marker statistics to {MAX_NATIVE_MARKER_PEAK_PIXELS} input peaks.",
        )
    try:
        derived_identity = expected_native_watershed_stage_identity(
            checkpoint.identity, recipe, stage_id, input_checkpoint_sha256,
            expected_implementation_identity,
        )
    except (RecipeContractError, StaleRecipeIdentity, ValueError, TypeError) as error:
        raise NativeWatershedStageError(f"Exact watershed output checkpoint identity cannot be derived: {error}") from error
    if expected_output_identity != derived_identity:
        raise NativeWatershedStageError("Expected output identity differs from the exact recipe/source/frame/upstream contract.")
    try:
        validate_checkpoint_identity(
            recipe, stage_id, derived_identity,
            expected_identity=expected_output_identity,
            expected_upstream_checkpoint_sha256=input_checkpoint_sha256,
        )
    except (RecipeContractError, StaleRecipeIdentity) as error:
        raise NativeWatershedStageError(f"Derived output checkpoint identity is stale: {error}") from error

    if not torch.cuda.is_available() or device.index < 0 or device.index >= torch.cuda.device_count():
        return _outcome_without_checkpoint(
            recipe, stage_id, input_checkpoint_sha256, NativeWatershedStageStatus.UNAVAILABLE,
            f"Requested CUDA device {device} is unavailable.",
        )
    try:
        actual_identity = native_watershed_identity(device)
    except NativeWatershedUnavailableError as error:
        return _outcome_without_checkpoint(
            recipe, stage_id, input_checkpoint_sha256, NativeWatershedStageStatus.UNAVAILABLE, str(error),
        )
    if actual_identity != expected_implementation_identity:
        raise NativeWatershedStageError("Current watershed implementation or CUDA environment differs from its caller pin.")
    if _cancelled(cancel):
        return _outcome_without_checkpoint(
            recipe, stage_id, input_checkpoint_sha256, NativeWatershedStageStatus.CANCELLED,
            "Cancelled before native CUDA execution.",
        )

    try:
        transfer_started_ns = time.perf_counter_ns()
        input_tensor = torch.from_numpy(np.array(input_array, copy=True, order="C")).to(device=device)
        torch.cuda.synchronize(device)
        input_transfer_ns = time.perf_counter_ns() - transfer_started_ns
        expected_input_dtype = torch.uint8 if operation_id == NATIVE_DISTANCE_OPERATION_ID else torch.bool
        if (
            input_tensor.device != device or input_tensor.dtype != expected_input_dtype
            or tuple(input_tensor.shape) != expected_coordinate_frame.shape
        ):
            raise NativeWatershedStageError("Watershed input tensor did not retain the exact CUDA device, dtype and frame.")
        compute_started_ns = time.perf_counter_ns()
        if operation_id == NATIVE_DISTANCE_OPERATION_ID:
            distance = distance_map(
                input_tensor,
                cancel=(lambda: _cancelled(cancel)) if cancel is not None else None,
            )
            marker_count = None
        else:
            if peak_count == 0:
                labels = torch.zeros((1, 1, *expected_coordinate_frame.shape), device=device, dtype=torch.int32)
                marker_count = 0
            else:
                labels, statistics = connected_components(input_tensor)
                if not isinstance(statistics, Mapping) or not isinstance(statistics.get("area"), np.ndarray):
                    raise RuntimeError("CUDA connected-components omitted compact component areas.")
                marker_count = int(statistics["area"].size) - 1
                if marker_count < 0 or marker_count > MAX_NATIVE_MARKER_PEAK_PIXELS:
                    raise RuntimeError("CUDA connected-components returned an invalid marker count.")
        torch.cuda.synchronize(device)
        computation_runtime_ns = time.perf_counter_ns() - compute_started_ns
        if operation_id == NATIVE_DISTANCE_OPERATION_ID:
            output = distance
            output_valid = _validate_distance_output(input_tensor, output, device)
        else:
            output = labels[0, 0]
            output_valid = _validate_component_output(input_tensor, labels, marker_count, device)
        if not output_valid:
            return _outcome_without_checkpoint(
                recipe, stage_id, input_checkpoint_sha256, NativeWatershedStageStatus.FAILED,
                "CUDA watershed primitive failed its output-domain or connected-label postcondition.",
            )
    except DistanceCancelledError as error:
        return _outcome_without_checkpoint(
            recipe, stage_id, input_checkpoint_sha256, NativeWatershedStageStatus.CANCELLED,
            f"Distance-map calculation was cancelled: {error}",
        )
    except DistanceResourceError as error:
        return _outcome_without_checkpoint(
            recipe, stage_id, input_checkpoint_sha256, NativeWatershedStageStatus.UNSUPPORTED,
            f"Pinned distance-map resource limit was exceeded: {error}",
        )
    except NativeWatershedStageError:
        raise
    except Exception as error:
        return _outcome_without_checkpoint(
            recipe, stage_id, input_checkpoint_sha256, NativeWatershedStageStatus.FAILED,
            f"Native CUDA watershed stage failed: {type(error).__name__}: {error}",
        )
    if _cancelled(cancel):
        return _outcome_without_checkpoint(
            recipe, stage_id, input_checkpoint_sha256, NativeWatershedStageStatus.CANCELLED,
            "Cancelled before native watershed checkpoint serialization.",
        )

    # A zero-valued distance map is still an executed numeric raster. Only a
    # PeakMask with no true pixels has the native marker stage's EMPTY state.
    is_empty = operation_id == NATIVE_MARKER_OPERATION_ID and peak_count == 0
    state = StageRunState.EMPTY if is_empty else StageRunState.EXECUTED
    status = NativeWatershedStageStatus.EMPTY if is_empty else NativeWatershedStageStatus.EXECUTED
    execution = ExecutionRecord(
        status=ExecutionStatus.EMPTY if is_empty else ExecutionStatus.AVAILABLE,
        elapsed_ms=computation_runtime_ns / 1_000_000,
        transfer_ms=input_transfer_ns / 1_000_000,
    )
    output_role = (
        RasterRole("raster", np.dtype(np.float32).str, "HW")
        if operation_id == NATIVE_DISTANCE_OPERATION_ID
        else RasterRole("instances", np.dtype(np.int32).str, "HW")
    )
    source_mapping = stage.source_mappings[0]
    metadata = _strict_metadata({
        "format": NATIVE_WATERSHED_STAGE_FORMAT,
        "version": NATIVE_WATERSHED_STAGE_VERSION,
        "stage_id": stage_id,
        "operation_id": operation_id,
        "execution_mode": RecipeProvenanceMode.NATIVE_COMPATIBLE.value,
        "recipe_identity": recipe.identity.to_dict(),
        "source_mapping": source_mapping.to_dict(),
        "source_sha256": expected_source_sha256,
        "coordinate_frame": expected_coordinate_frame.to_dict(),
        "upstream_checkpoint_sha256": input_checkpoint_sha256,
        "input_checkpoint_identity": checkpoint.identity.to_dict(),
        "input_bindings": dict(input_bindings),
        "input_raster_roles": {name: role.to_dict() for name, role in input_roles.items()},
        "output_raster_roles": {output_name: output_role.to_dict()},
        "implementation_identity": expected_implementation_identity.to_dict(),
        "selected_parameters": {},
        "effective_parameters": (
            {
                "distance_type": "DIST_L2",
                "requested_mask_size": 0,
                "effective_algorithm": "OpenCV 4.11.0 WithLabels 5x5 chamfer fixed-point sweep",
                "distance_unit": "pixel",
            }
            if operation_id == NATIVE_DISTANCE_OPERATION_ID
            else {"connectivity": 8, "background_label": 0}
        ),
        "native_supported_domain": native_supported_domain(operation_id),
        **({"marker_count": marker_count} if operation_id == NATIVE_MARKER_OPERATION_ID else {}),
        "output_inventory": [{
            "name": output_name,
            "dtype": output.dtype.str if isinstance(output, np.ndarray) else np.dtype(
                np.float32 if operation_id == NATIVE_DISTANCE_OPERATION_ID else np.int32
            ).str,
            "shape": list(expected_coordinate_frame.shape),
            "nbytes": int(output.numel() * output.element_size()),
            "device": str(output.device),
        }],
        "input_transfer_runtime_ns": input_transfer_ns,
        "computation_runtime_ns": computation_runtime_ns,
        "original_reference_intermediate": False,
        "original_reference_public_call_exposes_intermediate": False,
        "reference_comparison_status": "not_evaluated",
        "scientific_validation_status": "not_certified",
        "measurement_contract": output_measurement.to_dict(),
    }, min(MAX_NATIVE_WATERSHED_METADATA_BYTES, limits.max_metadata_bytes))
    try:
        reference = publish_checkpoint(
            output_root, derived_identity, {output_name: output},
            execution=execution,
            measurement=output_measurement,
            raster_roles={output_name: output_role},
            metadata=metadata,
            allow_cpu_transfer=True,
            limits=limits,
            cancel=cancel,
        )
    except CheckpointCancelled as error:
        return _outcome_without_checkpoint(
            recipe, stage_id, input_checkpoint_sha256, NativeWatershedStageStatus.CANCELLED,
            f"Cancelled during checkpoint serialization: {error}",
        )
    except CheckpointError as error:
        return _outcome_without_checkpoint(
            recipe, stage_id, input_checkpoint_sha256, NativeWatershedStageStatus.FAILED,
            f"Watershed output checkpoint serialization failed: {error}",
        )
    except Exception as error:
        return _outcome_without_checkpoint(
            recipe, stage_id, input_checkpoint_sha256, NativeWatershedStageStatus.FAILED,
            f"Watershed output checkpoint serialization failed: {type(error).__name__}: {error}",
        )

    record = StageRunRecord(
        stage_id=stage_id,
        recipe_identity=recipe.identity,
        state=state,
        checkpoint_sha256=reference.checkpoint_sha256,
        upstream_checkpoint_sha256=input_checkpoint_sha256,
    )
    validate_stage_run(recipe, record, expected_upstream_checkpoint_sha256=input_checkpoint_sha256)
    validate_checkpoint_identity(
        recipe, stage_id, reference.identity,
        expected_identity=derived_identity,
        expected_upstream_checkpoint_sha256=input_checkpoint_sha256,
    )
    return NativeWatershedStageOutcome(status, record, reference)


__all__ = [
    "MAX_NATIVE_MARKER_PEAK_PIXELS",
    "MAX_NATIVE_WATERSHED_METADATA_BYTES",
    "NATIVE_DISTANCE_INPUT_PORT",
    "NATIVE_DISTANCE_OPERATION_ID",
    "NATIVE_DISTANCE_OUTPUT_NAME",
    "NATIVE_MARKER_OPERATION_ID",
    "NATIVE_MARKER_OUTPUT_NAME",
    "NATIVE_WATERSHED_BACKEND_ID",
    "NATIVE_WATERSHED_BACKEND_VERSION",
    "NATIVE_WATERSHED_STAGE_FORMAT",
    "NATIVE_WATERSHED_STAGE_VERSION",
    "SUPPORTED_NATIVE_WATERSHED_OPERATIONS",
    "NativeWatershedIdentity",
    "NativeWatershedStageError",
    "NativeWatershedStageOutcome",
    "NativeWatershedStageStatus",
    "NativeWatershedUnavailableError",
    "expected_native_watershed_stage_identity",
    "native_watershed_identity",
    "run_native_watershed_stage",
]
