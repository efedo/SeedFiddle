"""One-stage CUDA median and morphology bridges for benchmark checkpoints.

The adapter binds one exact native-compatible recipe stage to its source,
implementation, CUDA environment, input checkpoint, parameters, and output
checkpoint. Full-resolution arrays stay on CUDA during calculation; checkpoint
publication is the explicit serialization boundary. This is not full recipe
execution or scientific validation.
"""

from __future__ import annotations

from dataclasses import (
    MISSING,
    dataclass,
    fields as dataclass_fields,
    is_dataclass,
    replace as dataclass_replace,
)
from enum import StrEnum
from functools import lru_cache
from hashlib import sha256
import ast
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
    MAX_PARAMETERS_BYTES,
    PINNED_PLANTCV_COMMIT,
    PINNED_PLANTCV_VERSION,
    PINNED_REQUIREMENTS_SHA256,
    PINNED_SOURCE_AUDIT_SHA256,
)
from seedvision.benchmarking import native_threshold_stage as _identity_tools
from seedvision.cuda import plantcv_median as _median_module
from seedvision.cuda import plantcv_morphology as _morphology_module
from seedvision.cuda import tensor_callback as _tensor_callback
from seedvision.cuda.plantcv_median import (
    MedianCancelledError,
    UnsupportedMedianGeometryError,
    median_blur,
)
from seedvision.cuda.plantcv_morphology import dilate, erode
from seedvision.cuda import plantcv_fill as _fill_module
from seedvision.cuda.plantcv_fill import FillCancelledError, FillConvergenceError, fill


NATIVE_MASK_STAGE_FORMAT = "seedfiddle.native-mask-stage-output"
NATIVE_MASK_STAGE_VERSION = 1
NATIVE_MASK_BACKEND_ID = "seedvision.cuda.median_morphology"
NATIVE_MASK_BACKEND_VERSION = "1"
MAX_NATIVE_MASK_METADATA_BYTES = 1024 * 1024
SUPPORTED_NATIVE_MASK_OPERATIONS = frozenset({
    "filter.median_blur",
    "morphology.fill",
    "morphology.erode",
    "morphology.dilate",
})
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_CAPABILITY = re.compile(r"[0-9]+\.[0-9]+\Z")
_PARAMETER_KEYS = {
    "filter.median_blur": frozenset({"ksize"}),
    "morphology.fill": frozenset({"size"}),
    "morphology.erode": frozenset({"ksize", "iterations"}),
    "morphology.dilate": frozenset({"ksize", "iterations"}),
}
_CANONICAL_RECIPE_IDS = {
    "filter.median_blur": "plantcv_seed_binary",
    "morphology.fill": "plantcv_seed_binary",
    "morphology.erode": "plantcv_seed_binary",
    "morphology.dilate": "plantcv_seed_binary",
}
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
_NATIVE_SOURCE_CONSTANTS = frozenset({
    "NATIVE_MASK_STAGE_FORMAT", "NATIVE_MASK_STAGE_VERSION",
    "NATIVE_MASK_BACKEND_ID", "NATIVE_MASK_BACKEND_VERSION",
    "MAX_NATIVE_MASK_METADATA_BYTES", "SUPPORTED_NATIVE_MASK_OPERATIONS",
    "_SHA256", "_CAPABILITY", "_PARAMETER_KEYS", "_CANONICAL_RECIPE_IDS",
    "_INPUT_PORT", "_OUTPUT_ARRAY",
})
_MEDIAN_SOURCE_CONSTANTS = frozenset({
    "PLANTCV_VERSION", "PLANTCV_COMMIT", "PLANTCV_MEDIAN_SOURCE_SHA256",
    "PLANTCV_HELPER_REFERENCE_ENV_SHA256", "MIN_KERNEL_AXIS", "MAX_KERNEL_AXIS",
    "MAX_RASTER_PIXELS", "MAX_TEMPORARY_WINDOW_VALUES", "MAX_OUTPUTS_PER_TILE",
})
_MORPHOLOGY_SOURCE_CONSTANTS = frozenset({
    "PLANTCV_VERSION", "PLANTCV_COMMIT", "PLANTCV_ERODE_SOURCE_SHA256",
    "PLANTCV_DILATE_SOURCE_SHA256", "PLANTCV_HELPER_REFERENCE_ENV_SHA256",
    "MIN_KERNEL_SIZE", "MAX_KERNEL_SIZE", "MIN_ITERATIONS", "MAX_ITERATIONS",
    "MAX_RASTER_PIXELS",
})
_FILL_SOURCE_CONSTANTS = frozenset({
    "PLANTCV_VERSION", "PLANTCV_COMMIT", "PLANTCV_FILL_SOURCE_SHA256",
    "MIN_OBJECT_AREA", "MAX_RASTER_PIXELS", "MAX_EDGE_CHUNK",
    "MAX_UNION_PASSES", "MAX_ROOT_JUMPS", "PIXEL_CHUNK",
})


class NativeMaskStageError(ValueError):
    """A native mask stage or its immutable bindings are invalid."""


class NativeMaskUnavailableError(RuntimeError):
    """The explicitly requested CUDA device/runtime is unavailable."""


class NativeMaskStageStatus(StrEnum):
    """Stable stage outcome labels, including cancellation separate from failure."""

    EXECUTED = "executed"
    EMPTY = "empty"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNAVAILABLE = "unavailable"
    UNSUPPORTED = "unsupported"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True, slots=True)
class NativeMaskIdentity:
    """Exact loaded implementation and CUDA environment for this backend."""

    implementation_sha256: str
    python_version: str
    torch_version: str
    numpy_version: str
    cuda_runtime_version: str
    platform: str
    machine: str
    device_index: int
    device_name: str
    compute_capability: str
    reference_source_sha256: str

    def __post_init__(self) -> None:
        for name in ("implementation_sha256", "reference_source_sha256"):
            value = getattr(self, name)
            if not isinstance(value, str) or not _SHA256.fullmatch(value):
                raise NativeMaskStageError(f"{name} must be a lowercase SHA-256 digest.")
        for name in (
            "python_version", "torch_version", "numpy_version", "cuda_runtime_version",
            "platform", "machine", "device_name",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or value != value.strip():
                raise NativeMaskStageError(f"{name} must be non-empty trimmed text.")
        if type(self.device_index) is not int or self.device_index < 0:
            raise NativeMaskStageError("device_index must be a non-negative integer.")
        if not isinstance(self.compute_capability, str) or not _CAPABILITY.fullmatch(self.compute_capability):
            raise NativeMaskStageError("compute_capability must be a major.minor string.")

    @property
    def backend_id(self) -> str:
        return NATIVE_MASK_BACKEND_ID

    @property
    def backend_version(self) -> str:
        return NATIVE_MASK_BACKEND_VERSION

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
            "plantcv_median_source_sha256": _median_module.PLANTCV_MEDIAN_SOURCE_SHA256,
            "plantcv_erode_source_sha256": _morphology_module.PLANTCV_ERODE_SOURCE_SHA256,
            "plantcv_dilate_source_sha256": _morphology_module.PLANTCV_DILATE_SOURCE_SHA256,
            "plantcv_fill_source_sha256": _fill_module.PLANTCV_FILL_SOURCE_SHA256,
            "plantcv_median_helper_environment_sha256": _median_module.PLANTCV_HELPER_REFERENCE_ENV_SHA256,
            "plantcv_morphology_helper_environment_sha256": _morphology_module.PLANTCV_HELPER_REFERENCE_ENV_SHA256,
            "plantcv_reference_source_sha256": self.reference_source_sha256,
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
class NativeMaskStageOutcome:
    """Native execution state and immutable output checkpoint, when present."""

    status: str
    run_record: StageRunRecord
    checkpoint_reference: CheckpointReference | None

    def __post_init__(self) -> None:
        if self.status not in {
            NativeMaskStageStatus.EXECUTED, NativeMaskStageStatus.EMPTY,
            NativeMaskStageStatus.FAILED, NativeMaskStageStatus.CANCELLED,
            NativeMaskStageStatus.UNAVAILABLE, NativeMaskStageStatus.UNSUPPORTED,
            NativeMaskStageStatus.UNRESOLVED,
        }:
            raise ValueError("status is not a supported native mask-stage outcome.")
        if not isinstance(self.run_record, StageRunRecord):
            raise TypeError("run_record must be a StageRunRecord.")
        completed = self.status in {NativeMaskStageStatus.EXECUTED, NativeMaskStageStatus.EMPTY}
        if completed != (self.checkpoint_reference is not None):
            raise ValueError("Only executed or empty native stages carry a checkpoint reference.")
        expected_state = {
            NativeMaskStageStatus.EXECUTED: StageRunState.EXECUTED,
            NativeMaskStageStatus.EMPTY: StageRunState.EMPTY,
            NativeMaskStageStatus.FAILED: StageRunState.FAILED,
            NativeMaskStageStatus.CANCELLED: StageRunState.FAILED,
            NativeMaskStageStatus.UNAVAILABLE: StageRunState.UNAVAILABLE,
            NativeMaskStageStatus.UNSUPPORTED: StageRunState.UNSUPPORTED,
            NativeMaskStageStatus.UNRESOLVED: StageRunState.UNRESOLVED,
        }[self.status]
        if self.run_record.state is not expected_state:
            raise ValueError("Outcome status and immutable stage-run state differ.")
        if self.checkpoint_reference is not None and (
            self.run_record.checkpoint_sha256 != self.checkpoint_reference.checkpoint_sha256
        ):
            raise ValueError("Native stage run and checkpoint reference hashes differ.")


def _module_source_record(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Call the shared checked source recorder without cloning its AST machinery."""
    try:
        return _identity_tools._module_source_record(*args, **kwargs)
    except _identity_tools.NativeThresholdStageError as error:
        raise NativeMaskStageError(f"Native mask implementation identity failed: {error}") from error


def _normalize_mask_code_constant(value: Any) -> Any:
    """Extend the shared code normalizer for morphology's Ellipsis slice."""
    if value is Ellipsis:
        return {"ellipsis": True}
    if isinstance(value, CodeType):
        return {"code": _mask_code_payload(value)}
    if isinstance(value, bytes):
        return {"bytes_hex": value.hex()}
    if isinstance(value, tuple):
        return [_normalize_mask_code_constant(item) for item in value]
    if isinstance(value, frozenset):
        items = [_normalize_mask_code_constant(item) for item in value]
        return sorted(items, key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")))
    return _identity_tools._normalize_code_constant(value)


def _mask_code_payload(code: CodeType) -> dict[str, Any]:
    return {
        "argcount": code.co_argcount,
        "posonlyargcount": code.co_posonlyargcount,
        "kwonlyargcount": code.co_kwonlyargcount,
        "nlocals": code.co_nlocals,
        "stacksize": code.co_stacksize,
        "flags": code.co_flags,
        "bytecode_hex": code.co_code.hex(),
        "constants": [_normalize_mask_code_constant(item) for item in code.co_consts],
        "names": list(code.co_names),
        "varnames": list(code.co_varnames),
        "freevars": list(code.co_freevars),
        "cellvars": list(code.co_cellvars),
        "filename": code.co_filename,
        "name": code.co_name,
        "qualname": code.co_qualname,
        "firstlineno": code.co_firstlineno,
        "linetable_hex": code.co_linetable.hex(),
        "exceptiontable_hex": code.co_exceptiontable.hex(),
    }


def _morphology_source_record(
    path: Path,
    namespace: Mapping[str, Any],
    code_names: tuple[str, ...],
    constant_names: frozenset[str],
) -> dict[str, Any]:
    """Verify and record morphology source, including its compiled Ellipsis slices."""
    try:
        source = _identity_tools._read_source_bounded(path)
    except OSError as error:
        raise NativeMaskStageError(f"Cannot read morphology implementation source: {error}") from error
    if len(source) > _identity_tools.MAX_IMPLEMENTATION_SOURCE_BYTES:
        raise NativeMaskStageError("Morphology implementation source exceeds its hash bound.")
    try:
        compiled = compile(source, str(path), "exec")
        tree = ast.parse(source, filename=str(path))
    except (SyntaxError, UnicodeError, ValueError) as error:
        raise NativeMaskStageError(f"Cannot verify loaded morphology source: {error}") from error
    compiled_codes: dict[str, CodeType] = {}

    def visit(code: CodeType) -> None:
        compiled_codes[code.co_qualname] = code
        for item in code.co_consts:
            if isinstance(item, CodeType):
                visit(item)

    visit(compiled)
    callable_nodes = _identity_tools._source_callable_nodes(tree)
    code_records: list[dict[str, str]] = []
    for name in code_names:
        value = namespace.get(name)
        value = getattr(value, "__wrapped__", value)
        actual = getattr(value, "__code__", None)
        candidate = compiled_codes.get(name)
        if actual is None or candidate is None or _mask_code_payload(actual) != _mask_code_payload(candidate):
            raise NativeMaskStageError(f"Loaded morphology implementation {name} differs from current source.")
        definition = callable_nodes.get(name)
        if definition is None:
            raise NativeMaskStageError(f"Morphology implementation source is missing callable {name}.")
        defaults = _identity_tools._verify_function_defaults(name, value, definition, namespace)
        code_bytes = json.dumps(_mask_code_payload(actual), sort_keys=True, separators=(",", ":")).encode("utf-8")
        defaults_bytes = json.dumps(defaults, sort_keys=True, separators=(",", ":")).encode("utf-8")
        code_records.append({
            "name": name,
            "code_sha256": sha256(code_bytes).hexdigest(),
            "defaults_sha256": sha256(defaults_bytes).hexdigest(),
        })
    source_constants: dict[str, Any] = {}
    for statement in tree.body:
        targets: tuple[ast.expr, ...] = ()
        expression: ast.expr | None = None
        if isinstance(statement, ast.Assign):
            targets = tuple(statement.targets)
            expression = statement.value
        elif isinstance(statement, ast.AnnAssign):
            targets = (statement.target,)
            expression = statement.value
        if expression is None:
            continue
        for target in targets:
            if isinstance(target, ast.Name) and target.id in constant_names:
                source_constants[target.id] = _identity_tools._normalize_identity_value(
                    _identity_tools._literal_node(expression, namespace)
                )
    if set(source_constants) != set(constant_names):
        raise NativeMaskStageError("Morphology implementation source has an unexpected constant inventory.")
    runtime_constants = {
        name: _identity_tools._normalize_identity_value(namespace[name])
        for name in constant_names
        if name in namespace
    }
    if runtime_constants != source_constants:
        raise NativeMaskStageError("Loaded morphology constants differ from their source declarations.")
    return {
        "name": "plantcv_morphology.py",
        "source_sha256": sha256(source).hexdigest(),
        "loaded_code": code_records,
        "runtime_constants": runtime_constants,
        "dataclasses": [],
    }


def native_supported_domain(operation_id: str) -> str:
    """Canonical archived support declaration, shared with inspection."""
    return {
        "filter.median_blur": "uint8 grayscale; scalar or rectangular kernel in [1,255]; pinned SciPy reflect boundary; full canvas",
        "morphology.fill": "uint8 0/255 mask; 4-connected components; removes area strictly below size; full canvas; no ROI",
        "morphology.erode": "uint8 0/255 mask; square kernel [2,255]; iterations [1,64]; full canvas",
        "morphology.dilate": "uint8 0/255 mask; square kernel [2,255]; iterations [1,64]; full canvas",
    }[operation_id]


def _implementation_digest() -> str:
    helper_names = (
        "_literal_node", "_normalize_identity_value", "_normalize_code_constant",
        "_code_payload", "_source_callable_nodes", "_verify_function_defaults",
        "_read_source_bounded", "_module_source_record",
    )
    for name in helper_names:
        if getattr(_identity_tools, name, None) is not _IDENTITY_HELPERS[name]:
            raise NativeMaskStageError("Shared loaded-code identity helpers were replaced after import.")
    if median_blur is not _median_module.median_blur:
        raise NativeMaskStageError("Imported CUDA median callable was replaced after module loading.")
    if erode is not _morphology_module.erode or dilate is not _morphology_module.dilate:
        raise NativeMaskStageError("Imported CUDA morphology callables were replaced after module loading.")
    if fill is not _fill_module.fill:
        raise NativeMaskStageError("Imported CUDA small-object callable was replaced after module loading.")
    if FillCancelledError is not _fill_module.FillCancelledError:
        raise NativeMaskStageError("Imported CUDA fill cancellation type was replaced after module loading.")
    if FillConvergenceError is not _fill_module.FillConvergenceError:
        raise NativeMaskStageError("Imported CUDA fill convergence type was replaced after module loading.")
    if (_fill_module._tensor_callback is not _tensor_callback
            or _morphology_module._tensor_callback is not _tensor_callback):
        raise NativeMaskStageError("Imported CUDA callback helper differs from the recorded module.")

    bridge_names = (
        "_module_source_record", "_implementation_digest", "native_mask_identity",
        "_normalize_mask_code_constant", "_mask_code_payload", "_morphology_source_record",
        "_canonical_catalogue", "_canonical_recipe_and_stage", "_validate_recipe_stage",
        "_parameters", "expected_native_mask_stage_identity", "_strict_metadata",
        "_cancelled", "_outcome_without_checkpoint", "_validate_image_identity",
        "_validate_mask_identity", "_validate_selected_input", "_device_array_matches",
        "run_native_mask_stage", "native_supported_domain",
        "NativeMaskIdentity.__post_init__", "NativeMaskIdentity.backend_id",
        "NativeMaskIdentity.backend_version", "NativeMaskIdentity.device",
        "NativeMaskIdentity.environment_versions", "NativeMaskIdentity.to_dict",
        "NativeMaskIdentity.recipe_provenance", "NativeMaskStageOutcome.__post_init__",
    )
    bridge_namespace = globals()
    helper_module = vars(_identity_tools)
    helper_record = _module_source_record(
        "native_threshold_stage.py", Path(_identity_tools.__file__).resolve(), helper_module,
        helper_names, frozenset({"MAX_IMPLEMENTATION_SOURCE_BYTES"}),
    )
    median_names = (
        "_validate_gray", "_kernel_shape", "_roi_bounds", "_validate_scipy_reflect_geometry",
        "_reflect_indices", "_median_validated", "median_blur",
    )
    median_record = _module_source_record(
        "plantcv_median.py", Path(_median_module.__file__).resolve(), vars(_median_module),
        median_names, _MEDIAN_SOURCE_CONSTANTS,
    )
    morphology_names = (
        "_validate_gray", "_positive_integer", "_roi_bounds", "_check_cancel", "_axis_max", "_morphology", "erode", "dilate",
    )
    morphology_record = _morphology_source_record(
        Path(_morphology_module.__file__).resolve(), vars(_morphology_module),
        morphology_names, _MORPHOLOGY_SOURCE_CONSTANTS,
    )
    callback_record = _module_source_record(
        "tensor_callback.py", Path(_tensor_callback.__file__).resolve(), vars(_tensor_callback),
        ("_signature", "checked_cancel"), frozenset(),
    )
    fill_names = (
        "_validate_mask", "_positive_area", "_roi_bounds", "_check_cancel", "_union_edge_batch",
        "_union_direction", "_compress_roots", "_component_areas", "_filter_crop", "fill",
    )
    fill_record = _module_source_record(
        "plantcv_fill.py", Path(_fill_module.__file__).resolve(), vars(_fill_module),
        fill_names, _FILL_SOURCE_CONSTANTS,
    )
    for cls in (NativeMaskIdentity, NativeMaskStageOutcome):
        for name in (
            "__post_init__", "backend_id", "backend_version", "device",
            "environment_versions", "to_dict", "recipe_provenance",
        ):
            if hasattr(cls, name):
                bridge_namespace[f"{cls.__name__}.{name}"] = getattr(cls, name)
    bridge_record = _module_source_record(
        "native_mask_stage.py", Path(__file__).resolve(), bridge_namespace, bridge_names,
        _NATIVE_SOURCE_CONSTANTS, dataclass_names=("NativeMaskIdentity", "NativeMaskStageOutcome"),
    )
    payload = {
        "format": "seedfiddle.native-mask-loaded-implementation",
        "version": 1,
        "files": [helper_record, bridge_record, median_record, morphology_record, callback_record, fill_record],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return sha256(encoded).hexdigest()


_IDENTITY_HELPERS = {
    name: getattr(_identity_tools, name)
    for name in (
        "_literal_node", "_normalize_identity_value", "_normalize_code_constant",
        "_code_payload", "_source_callable_nodes", "_verify_function_defaults",
        "_read_source_bounded", "_module_source_record",
    )
}


def native_mask_identity(device: Any) -> NativeMaskIdentity:
    """Describe the exact loaded CUDA mask implementation for an explicit device."""
    import torch

    if type(device) is not torch.device or device.type != "cuda" or device.index is None:
        raise NativeMaskStageError("device must be an explicit torch.device such as cuda:0.")
    if not torch.cuda.is_available():
        raise NativeMaskUnavailableError("CUDA is not available in this PyTorch runtime.")
    if device.index < 0 or device.index >= torch.cuda.device_count():
        raise NativeMaskUnavailableError(f"Requested CUDA device {device} is not available.")
    try:
        properties = torch.cuda.get_device_properties(device)
        runtime = torch.version.cuda or torch.version.hip
        if not runtime:
            raise NativeMaskUnavailableError("PyTorch reports no CUDA or HIP runtime version.")
        source_pins = {
            "median": _median_module.PLANTCV_MEDIAN_SOURCE_SHA256,
            "fill": _fill_module.PLANTCV_FILL_SOURCE_SHA256,
            "erode": _morphology_module.PLANTCV_ERODE_SOURCE_SHA256,
            "dilate": _morphology_module.PLANTCV_DILATE_SOURCE_SHA256,
        }
        source_digest = sha256(json.dumps(source_pins, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        return NativeMaskIdentity(
            implementation_sha256=_implementation_digest(),
            python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            torch_version=str(torch.__version__),
            numpy_version=np.__version__,
            cuda_runtime_version=str(runtime),
            platform=sys.platform,
            machine=host_platform.machine() or "unknown",
            device_index=int(device.index),
            device_name=str(properties.name),
            compute_capability=f"{int(properties.major)}.{int(properties.minor)}",
            reference_source_sha256=source_digest,
        )
    except NativeMaskUnavailableError:
        raise
    except NativeMaskStageError:
        raise
    except Exception as error:
        raise NativeMaskUnavailableError(f"Cannot inspect requested CUDA device {device}: {error}") from error


@lru_cache(maxsize=1)
def _canonical_catalogue():
    try:
        return build_recipe_catalogue()
    except (RecipeCatalogueError, OSError, ValueError) as error:
        raise NativeMaskStageError(f"Committed PlantCV operation contracts are unavailable: {error}") from error


def _canonical_recipe_and_stage(operation_id: str):
    try:
        recipe_id = _CANONICAL_RECIPE_IDS[operation_id]
    except KeyError as error:
        raise NativeMaskStageError(f"Operation {operation_id!r} is not supported by the native mask bridge.") from error
    recipe = _canonical_catalogue().recipe(recipe_id)
    matches = tuple(stage for stage in recipe.stages if stage.operation_id == operation_id)
    if len(matches) != 1:
        raise NativeMaskStageError("Committed operation catalogue does not contain one exact mask stage.")
    return recipe, matches[0]


def _validate_recipe_stage(recipe: RecipeSpec, stage_id: str, operation_id: str) -> Any:
    canonical_recipe, canonical_stage = _canonical_recipe_and_stage(operation_id)
    if recipe.recipe_id != canonical_recipe.recipe_id or recipe.recipe_version != canonical_recipe.recipe_version:
        raise NativeMaskStageError("Native recipe does not use the pinned PlantCV seed recipe identity.")
    if stage_id != canonical_stage.stage_id:
        raise NativeMaskStageError("Requested stage ID does not match the canonical public PlantCV operation.")
    if len(recipe.stages) != len(canonical_recipe.stages):
        raise NativeMaskStageError("Native recipe topology differs from the pinned recipe description.")
    for current, canonical in zip(recipe.stages, canonical_recipe.stages, strict=True):
        if current.stage_id != canonical.stage_id:
            raise NativeMaskStageError("Native recipe stage order differs from the pinned recipe.")
        if current.stage_id == stage_id:
            if len(current.parameters) != len(canonical.parameters):
                raise NativeMaskStageError("Native stage parameter inventory differs from its public operation.")
            for selected, authored in zip(current.parameters, canonical.parameters, strict=True):
                if dataclass_replace(selected, value=authored.value) != authored:
                    raise NativeMaskStageError("Native stage parameter contract differs from the public operation.")
            if dataclass_replace(current, parameters=canonical.parameters) != canonical:
                raise NativeMaskStageError("Native stage source mapping or port semantics differ from the pinned operation.")
        elif current != canonical:
            raise NativeMaskStageError("Another stage in the native recipe differs from the pinned recipe description.")
    normalized = dataclass_replace(
        recipe, provenance=canonical_recipe.provenance, stages=canonical_recipe.stages,
    )
    if normalized != canonical_recipe:
        raise NativeMaskStageError("Native recipe dependencies, layout, or description differ from the pinned recipe.")
    stage = recipe.stage(stage_id)
    if stage.resolution is not SourceResolution.PINNED:
        raise NativeMaskStageError("Runnable native mask stages require pinned target source resolution.")
    return stage


def _parameters(recipe: RecipeSpec, stage_id: str, operation_id: str) -> dict[str, Any]:
    values = recipe.parameters_for_runner(stage_id)
    if set(values) != _PARAMETER_KEYS[operation_id] or any(value is None for value in values.values()):
        raise NativeMaskStageError("Every calculation-affecting mask parameter must be selected explicitly.")
    if operation_id == "filter.median_blur":
        ksize = values["ksize"]
        if type(ksize) is int:
            dimensions = (ksize, ksize)
        elif type(ksize) is list and len(ksize) == 2 and all(type(value) is int for value in ksize):
            dimensions = tuple(ksize)
        else:
            raise NativeMaskStageError("Median ksize must be an integer or a two-integer [height,width] list.")
        if any(value < _median_module.MIN_KERNEL_AXIS or value > _median_module.MAX_KERNEL_AXIS for value in dimensions):
            raise NativeMaskStageError("Median kernel dimensions must be in [1,255].")
    elif operation_id == "morphology.fill":
        size = values["size"]
        if type(size) is not int or size < _fill_module.MIN_OBJECT_AREA:
            raise NativeMaskStageError("PlantCV fill size must be a positive integer pixel-area threshold.")
    else:
        ksize, iterations = values["ksize"], values["iterations"]
        if type(ksize) is not int or not _morphology_module.MIN_KERNEL_SIZE <= ksize <= _morphology_module.MAX_KERNEL_SIZE:
            raise NativeMaskStageError("Morphology ksize must be an integer in [2,255].")
        if type(iterations) is not int or not _morphology_module.MIN_ITERATIONS <= iterations <= _morphology_module.MAX_ITERATIONS:
            raise NativeMaskStageError("Morphology iterations must be an integer in [1,64].")
    try:
        encoded = json.dumps(values, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError, UnicodeError, RecursionError) as error:
        raise NativeMaskStageError("Native mask parameters must be strict finite JSON primitives.") from error
    if len(encoded) > MAX_PARAMETERS_BYTES:
        raise NativeMaskStageError("Native mask parameters exceed the bounded recipe runner limit.")
    return values


def expected_native_mask_stage_identity(
    input_identity: CheckpointIdentity,
    recipe: RecipeSpec,
    stage_id: str,
    upstream_checkpoint_sha256: str,
    implementation_identity: NativeMaskIdentity,
) -> CheckpointIdentity:
    """Derive output identity from the exact recipe, input, code, and upstream."""
    if not isinstance(input_identity, CheckpointIdentity) or not isinstance(recipe, RecipeSpec):
        raise TypeError("input_identity and recipe must be immutable checkpoint/recipe contracts.")
    if not isinstance(implementation_identity, NativeMaskIdentity):
        raise TypeError("implementation_identity must be a NativeMaskIdentity.")
    if not isinstance(upstream_checkpoint_sha256, str) or not _SHA256.fullmatch(upstream_checkpoint_sha256):
        raise NativeMaskStageError("upstream_checkpoint_sha256 must be a lowercase SHA-256 digest.")
    if recipe.provenance != implementation_identity.recipe_provenance():
        raise NativeMaskStageError("Recipe provenance differs from the exact native mask implementation identity.")
    try:
        stage = recipe.stage(stage_id)
    except KeyError as error:
        raise NativeMaskStageError(f"Recipe does not contain stage {stage_id!r}.") from error
    operation = stage.operation_id
    if operation not in SUPPORTED_NATIVE_MASK_OPERATIONS:
        raise NativeMaskStageError(f"Operation {operation!r} is outside this native mask bridge.")
    _validate_recipe_stage(recipe, stage_id, operation)
    _parameters(recipe, stage_id, operation)
    if operation == "filter.median_blur":
        _validate_image_identity(input_identity)
        image_semantics = dict(input_identity.image_semantics)
        mask_semantics = dict(input_identity.mask_semantics)
    else:
        _validate_mask_identity(input_identity)
        image_semantics = dict(input_identity.image_semantics)
        mask_semantics = dict(input_identity.mask_semantics)
        mask_semantics["border_policy"] = (
            "native_supported_morphology.fill_full_canvas_4_connected"
            if operation == "morphology.fill"
            else f"native_supported_{operation}"
        )
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
        backend_id=implementation_identity.backend_id,
        backend_version=implementation_identity.backend_version,
        environment_versions=dict(implementation_identity.environment_versions),
        image_semantics=image_semantics,
        mask_semantics=mask_semantics,
        instance_semantics=dict(input_identity.instance_semantics),
        upstream_checkpoint_sha256=upstream_checkpoint_sha256,
    )


def _strict_metadata(payload: Mapping[str, Any], maximum: int) -> dict[str, Any]:
    try:
        encoded = json.dumps(
            payload, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError, RecursionError) as error:
        raise NativeMaskStageError("Native mask metadata must be strict finite JSON.") from error
    if len(encoded) > maximum:
        raise NativeMaskStageError(f"Native mask metadata exceeds its {maximum}-byte limit.")
    return json.loads(encoded.decode("utf-8"))


def _cancelled(cancel: Any) -> bool:
    if cancel is None:
        return False
    if callable(cancel):
        return bool(cancel())
    is_set = getattr(cancel, "is_set", None)
    if not callable(is_set):
        raise NativeMaskStageError("cancel must be a callable or an event with is_set().")
    return bool(is_set())


def _outcome_without_checkpoint(
    recipe: RecipeSpec,
    stage_id: str,
    upstream_checkpoint_sha256: str,
    status: str,
    reason: str,
) -> NativeMaskStageOutcome:
    state = {
        NativeMaskStageStatus.FAILED: StageRunState.FAILED,
        NativeMaskStageStatus.CANCELLED: StageRunState.FAILED,
        NativeMaskStageStatus.UNAVAILABLE: StageRunState.UNAVAILABLE,
        NativeMaskStageStatus.UNSUPPORTED: StageRunState.UNSUPPORTED,
        NativeMaskStageStatus.UNRESOLVED: StageRunState.UNRESOLVED,
    }[status]
    compact_reason = " ".join(str(reason).split())[:512] or status
    record = StageRunRecord(
        stage_id=stage_id,
        recipe_identity=recipe.identity,
        state=state,
        upstream_checkpoint_sha256=upstream_checkpoint_sha256,
        reason=compact_reason,
    )
    validate_stage_run(recipe, record, expected_upstream_checkpoint_sha256=upstream_checkpoint_sha256)
    return NativeMaskStageOutcome(status, record, None)


def _validate_image_identity(identity: CheckpointIdentity) -> None:
    semantics = identity.image_semantics
    try:
        dtype = np.dtype(semantics.get("dtype")).str
    except (TypeError, ValueError) as error:
        raise NativeMaskStageError("Median input identity does not describe a gray image.") from error
    value_range = semantics.get("value_range")
    if (
        str(semantics.get("color_order", "")).upper() not in {"GRAY", "GRAYSCALE", "Y"}
        or dtype != np.dtype(np.uint8).str
        or not isinstance(value_range, (tuple, list))
        or tuple(value_range) != (0, 255)
    ):
        raise NativeMaskStageError("Median input must be uint8 grayscale with range [0,255].")


def _validate_mask_identity(identity: CheckpointIdentity) -> None:
    if identity.mask_semantics.get("representation") != "uint8_0_255":
        raise NativeMaskStageError("Morphology input identity must declare uint8 0/255 mask representation.")


def _validate_selected_input(
    checkpoint: Any,
    operation: str,
    input_bindings: Mapping[str, str],
    input_roles: Mapping[str, RasterRole],
    frame: CoordinateFrame,
) -> np.ndarray:
    port = _INPUT_PORT[operation]
    if set(input_bindings) != {port} or set(input_roles) != {port}:
        raise NativeMaskStageError(f"Stage requires one explicit {port!r} input binding and role.")
    array_name = input_bindings[port]
    if not isinstance(array_name, str) or not array_name or array_name not in checkpoint.arrays:
        raise NativeMaskStageError("Input binding does not name an array in the loaded checkpoint.")
    kind = "image" if operation == "filter.median_blur" else "mask"
    expected_role = RasterRole(kind, np.dtype(np.uint8).str, "HW", (0, 255) if kind == "image" else None)
    role = input_roles[port]
    if role != expected_role or checkpoint.raster_roles.get(array_name) != role:
        raise NativeMaskStageError("Bound raster role differs from the exact native input contract.")
    array = checkpoint.arrays[array_name]
    if (
        not isinstance(array, np.ndarray) or array.dtype != np.dtype(np.uint8)
        or array.ndim != 2 or tuple(array.shape) != frame.shape or not array.flags.c_contiguous
        or array.nbytes > MAX_ARRAY_BYTES or array.size > MAX_ARRAY_ELEMENTS
    ):
        raise NativeMaskStageError("Bound input must be a bounded contiguous uint8 HxW raster in the requested frame.")
    if kind == "image":
        _validate_image_identity(checkpoint.identity)
        if checkpoint.execution.status is not ExecutionStatus.AVAILABLE:
            raise NativeMaskStageError("Median input image must come from an available checkpoint execution.")
    else:
        _validate_mask_identity(checkpoint.identity)
        if checkpoint.execution.status not in {ExecutionStatus.AVAILABLE, ExecutionStatus.EMPTY}:
            raise NativeMaskStageError("Morphology input must come from an available or empty mask execution.")
        if array.size and not np.isin(array, (0, 255)).all():
            raise NativeMaskStageError("Bound BinaryMask contains values outside uint8 0/255.")
        is_empty = not bool(np.any(array))
        if is_empty != (checkpoint.execution.status is ExecutionStatus.EMPTY):
            raise NativeMaskStageError("Input mask pixels disagree with the recorded empty/available state.")
    return array


def _device_array_matches(value: Any, frame: CoordinateFrame, device: Any, *, binary: bool) -> bool:
    import torch

    if (
        not torch.is_tensor(value) or value.layout != torch.strided or value.device != device
        or value.dtype != torch.uint8 or tuple(value.shape) != frame.shape or not value.is_contiguous()
    ):
        return False
    return not binary or bool(torch.logical_or(value == 0, value == 255).all().item())


def run_native_mask_stage(
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
    expected_implementation_identity: NativeMaskIdentity,
    device: Any,
    input_bindings: Mapping[str, str],
    input_roles: Mapping[str, RasterRole],
    output_measurement: MeasurementContract,
    allow_cpu_transfer: bool,
    limits: CheckpointLimits = CheckpointLimits(),
    cancel: Any = None,
) -> NativeMaskStageOutcome:
    """Execute one exact native median/fill/erosion/dilation stage on CUDA."""
    import torch

    if not isinstance(recipe, RecipeSpec) or not isinstance(expected_recipe_identity, RecipeIdentity):
        raise TypeError("recipe and expected_recipe_identity must be immutable recipe contracts.")
    if not isinstance(expected_implementation_identity, NativeMaskIdentity):
        raise TypeError("expected_implementation_identity must be explicit.")
    if type(device) is not torch.device or device.type != "cuda" or device.index is None:
        raise NativeMaskStageError("device must be an explicit torch.device such as cuda:0.")
    if not isinstance(expected_coordinate_frame, CoordinateFrame):
        raise TypeError("expected_coordinate_frame must be a CoordinateFrame.")
    if not isinstance(expected_input_identity, CheckpointIdentity) or not isinstance(expected_input_measurement, MeasurementContract):
        raise TypeError("Expected input identity and measurement contracts are required.")
    if expected_output_identity is not None and not isinstance(expected_output_identity, CheckpointIdentity):
        raise TypeError("expected_output_identity must be a CheckpointIdentity or None.")
    if not isinstance(output_measurement, MeasurementContract):
        raise TypeError("output_measurement must be an explicit MeasurementContract.")
    if not isinstance(limits, CheckpointLimits):
        raise TypeError("limits must be a CheckpointLimits contract.")
    if output_measurement.eligible:
        raise NativeMaskStageError("Native median/morphology outputs cannot mark scientific measurements eligible.")
    if type(allow_cpu_transfer) is not bool or not allow_cpu_transfer:
        raise NativeMaskStageError("Checkpoint serialization requires explicit allow_cpu_transfer=True.")
    if expected_recipe_identity != recipe.identity:
        raise NativeMaskStageError("Recipe identity differs from the caller's exact expected identity.")
    if not isinstance(input_bindings, Mapping) or not isinstance(input_roles, Mapping):
        raise NativeMaskStageError("Input bindings and raster roles must be explicit mappings.")
    if any(not isinstance(name, str) or not isinstance(value, str) for name, value in input_bindings.items()):
        raise NativeMaskStageError("Input bindings must map port names to checkpoint array names.")
    if any(not isinstance(name, str) or not isinstance(role, RasterRole) for name, role in input_roles.items()):
        raise NativeMaskStageError("Input roles must map port names to RasterRole contracts.")
    if cancel is not None and not callable(cancel) and not callable(getattr(cancel, "is_set", None)):
        raise NativeMaskStageError("cancel must be a callable or an event with is_set().")
    for name, digest in (
        ("expected_source_sha256", expected_source_sha256),
        ("input_checkpoint_sha256", input_checkpoint_sha256),
        ("expected_upstream_checkpoint_sha256", expected_upstream_checkpoint_sha256),
    ):
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise NativeMaskStageError(f"{name} must be a lowercase SHA-256 digest.")
    if expected_upstream_checkpoint_sha256 != input_checkpoint_sha256:
        raise NativeMaskStageError("Expected upstream digest differs from the exact input checkpoint address.")
    if expected_input_identity.source_sha256 != expected_source_sha256:
        raise NativeMaskStageError("Expected input checkpoint source SHA differs from the requested source.")
    if expected_input_identity.coordinate_frame != expected_coordinate_frame:
        raise NativeMaskStageError("Expected input checkpoint frame differs from the requested coordinate frame.")
    elements = math.prod(expected_coordinate_frame.shape)
    if elements > MAX_ARRAY_ELEMENTS or elements > MAX_ARRAY_BYTES:
        raise NativeMaskStageError("Requested frame exceeds the bounded uint8 raster domain.")
    try:
        stage = recipe.stage(stage_id)
    except KeyError as error:
        raise NativeMaskStageError(f"Recipe does not contain stage {stage_id!r}.") from error

    if _cancelled(cancel):
        return _outcome_without_checkpoint(
            recipe, stage_id, input_checkpoint_sha256, NativeMaskStageStatus.CANCELLED,
            "Cancelled before native mask-stage validation.",
        )
    if recipe.provenance.mode is not RecipeProvenanceMode.NATIVE_COMPATIBLE:
        return _outcome_without_checkpoint(
            recipe, stage_id, input_checkpoint_sha256, NativeMaskStageStatus.UNSUPPORTED,
            "Only explicitly native-compatible recipes can execute through this bridge.",
        )
    if recipe.provenance.evidence_status is SourceResolution.UNAVAILABLE or stage.resolution is SourceResolution.UNAVAILABLE:
        return _outcome_without_checkpoint(
            recipe, stage_id, input_checkpoint_sha256, NativeMaskStageStatus.UNAVAILABLE,
            recipe.provenance.evidence_note or stage.resolution_note or "Pinned PlantCV source is unavailable.",
        )
    if recipe.provenance.evidence_status is SourceResolution.UNRESOLVED or stage.resolution is SourceResolution.UNRESOLVED:
        return _outcome_without_checkpoint(
            recipe, stage_id, input_checkpoint_sha256, NativeMaskStageStatus.UNRESOLVED,
            recipe.provenance.evidence_note or stage.resolution_note or "PlantCV source resolution is unresolved.",
        )
    if recipe.provenance != expected_implementation_identity.recipe_provenance():
        raise NativeMaskStageError("Recipe provenance differs from the exact caller-expected native identity.")
    operation = stage.operation_id
    if operation not in SUPPORTED_NATIVE_MASK_OPERATIONS:
        return _outcome_without_checkpoint(
            recipe, stage_id, input_checkpoint_sha256, NativeMaskStageStatus.UNSUPPORTED,
            f"Operation {operation!r} is outside the native median/fill/erosion/dilation bridge.",
        )
    stage = _validate_recipe_stage(recipe, stage_id, operation)
    try:
        parameters = _parameters(recipe, stage_id, operation)
    except NativeMaskStageError as error:
        return _outcome_without_checkpoint(
            recipe, stage_id, input_checkpoint_sha256, NativeMaskStageStatus.UNSUPPORTED, str(error),
        )

    if set(input_bindings) != {_INPUT_PORT[operation]} or set(input_roles) != {_INPUT_PORT[operation]}:
        raise NativeMaskStageError("Stage input bindings differ from its exact recipe port topology.")
    try:
        checkpoint = load_stage_input(
            input_root, input_checkpoint_sha256,
            expected_identity=expected_input_identity,
            expected_measurement=expected_input_measurement,
            limits=limits,
        )
    except Exception as error:
        raise NativeMaskStageError(f"Exact stage input checkpoint could not be loaded: {error}") from error
    if checkpoint.identity.source_sha256 != expected_source_sha256:
        raise NativeMaskStageError("Loaded input checkpoint source SHA differs from the requested source.")
    if checkpoint.identity.coordinate_frame != expected_coordinate_frame:
        raise NativeMaskStageError("Loaded input checkpoint frame differs from the requested coordinate frame.")
    input_array = _validate_selected_input(
        checkpoint, operation, input_bindings, input_roles, expected_coordinate_frame,
    )
    try:
        derived_identity = expected_native_mask_stage_identity(
            checkpoint.identity, recipe, stage_id, input_checkpoint_sha256,
            expected_implementation_identity,
        )
    except (RecipeContractError, StaleRecipeIdentity, ValueError, TypeError) as error:
        raise NativeMaskStageError(f"Exact output checkpoint identity cannot be derived: {error}") from error
    if expected_output_identity != derived_identity:
        raise NativeMaskStageError("Expected output identity differs from the exact native stage/source/frame/upstream contract.")
    try:
        validate_checkpoint_identity(
            recipe, stage_id, derived_identity,
            expected_identity=expected_output_identity,
            expected_upstream_checkpoint_sha256=input_checkpoint_sha256,
        )
    except (RecipeContractError, StaleRecipeIdentity) as error:
        raise NativeMaskStageError(f"Derived output identity is not valid for this recipe stage: {error}") from error

    if not torch.cuda.is_available() or device.index < 0 or device.index >= torch.cuda.device_count():
        return _outcome_without_checkpoint(
            recipe, stage_id, input_checkpoint_sha256, NativeMaskStageStatus.UNAVAILABLE,
            f"Requested CUDA device {device} is unavailable.",
        )
    try:
        actual_identity = native_mask_identity(device)
    except NativeMaskUnavailableError as error:
        return _outcome_without_checkpoint(
            recipe, stage_id, input_checkpoint_sha256, NativeMaskStageStatus.UNAVAILABLE, str(error),
        )
    if actual_identity != expected_implementation_identity:
        raise NativeMaskStageError("Current native implementation or CUDA environment differs from its caller pin.")
    if _cancelled(cancel):
        return _outcome_without_checkpoint(
            recipe, stage_id, input_checkpoint_sha256, NativeMaskStageStatus.CANCELLED,
            "Cancelled before native CUDA execution.",
        )

    try:
        transfer_started_ns = time.perf_counter_ns()
        tensor = torch.from_numpy(np.array(input_array, copy=True, order="C")).to(device=device)
        torch.cuda.synchronize(device)
        input_transfer_ns = time.perf_counter_ns() - transfer_started_ns
        if tensor.device != device or tensor.dtype != torch.uint8 or tuple(tensor.shape) != expected_coordinate_frame.shape:
            raise NativeMaskStageError("Input tensor did not retain the exact CUDA device, dtype, and frame.")
        compute_started_ns = time.perf_counter_ns()
        if operation == "filter.median_blur":
            selected_ksize = parameters["ksize"]
            primitive_ksize = tuple(selected_ksize) if isinstance(selected_ksize, list) else selected_ksize
            output = median_blur(tensor, primitive_ksize, cancel=(lambda: _cancelled(cancel)) if cancel is not None else None)
            effective_parameters = {"ksize": primitive_ksize, "roi": None}
        elif operation == "morphology.fill":
            output = fill(
                tensor,
                parameters["size"],
                cancel=(lambda: _cancelled(cancel)) if cancel is not None else None,
            )
            effective_parameters = dict(parameters)
        elif operation == "morphology.erode":
            output = erode(tensor, parameters["ksize"], parameters["iterations"])
            effective_parameters = dict(parameters)
        else:
            output = dilate(tensor, parameters["ksize"], parameters["iterations"])
            effective_parameters = dict(parameters)
        torch.cuda.synchronize(device)
        computation_runtime_ns = time.perf_counter_ns() - compute_started_ns
    except UnsupportedMedianGeometryError as error:
        return _outcome_without_checkpoint(
            recipe, stage_id, input_checkpoint_sha256, NativeMaskStageStatus.UNSUPPORTED,
            f"Pinned SciPy median geometry is source-undefined: {error}",
        )
    except MedianCancelledError as error:
        return _outcome_without_checkpoint(
            recipe, stage_id, input_checkpoint_sha256, NativeMaskStageStatus.CANCELLED,
            f"Median filtering was cancelled: {error}",
        )
    except FillCancelledError as error:
        return _outcome_without_checkpoint(
            recipe, stage_id, input_checkpoint_sha256, NativeMaskStageStatus.CANCELLED,
            f"Small-object filtering was cancelled: {error}",
        )
    except FillConvergenceError as error:
        return _outcome_without_checkpoint(
            recipe, stage_id, input_checkpoint_sha256, NativeMaskStageStatus.FAILED,
            f"Small-object component analysis did not converge: {error}",
        )
    except NativeMaskStageError:
        raise
    except Exception as error:
        return _outcome_without_checkpoint(
            recipe, stage_id, input_checkpoint_sha256,
            NativeMaskStageStatus.CANCELLED if _cancelled(cancel) else NativeMaskStageStatus.FAILED,
            f"Native CUDA mask-stage execution failed: {type(error).__name__}: {error}",
        )

    binary_output = operation != "filter.median_blur"
    if not _device_array_matches(output, expected_coordinate_frame, device, binary=binary_output):
        return _outcome_without_checkpoint(
            recipe, stage_id, input_checkpoint_sha256, NativeMaskStageStatus.FAILED,
            "Native CUDA primitive returned a malformed raster or changed device/frame semantics.",
        )
    if _cancelled(cancel):
        return _outcome_without_checkpoint(
            recipe, stage_id, input_checkpoint_sha256, NativeMaskStageStatus.CANCELLED,
            "Cancelled before native checkpoint serialization.",
        )

    output_name = _OUTPUT_ARRAY[operation]
    output_role = (
        RasterRole("image", np.dtype(np.uint8).str, "HW", (0, 255))
        if output_name == "image"
        else RasterRole("mask", np.dtype(np.uint8).str, "HW")
    )
    is_empty = output_name == "mask" and int(torch.count_nonzero(output).item()) == 0
    state = StageRunState.EMPTY if is_empty else StageRunState.EXECUTED
    status = NativeMaskStageStatus.EMPTY if is_empty else NativeMaskStageStatus.EXECUTED
    execution = ExecutionRecord(
        status=ExecutionStatus.EMPTY if is_empty else ExecutionStatus.AVAILABLE,
        elapsed_ms=computation_runtime_ns / 1_000_000,
        transfer_ms=input_transfer_ns / 1_000_000,
    )
    source_mapping = stage.source_mappings[0]
    metadata = _strict_metadata({
        "format": NATIVE_MASK_STAGE_FORMAT,
        "version": NATIVE_MASK_STAGE_VERSION,
        "stage_id": stage_id,
        "operation_id": operation,
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
        "selected_parameters": parameters,
        "effective_parameters": effective_parameters,
        "native_supported_domain": native_supported_domain(operation),
        "output_inventory": [{
            "name": output_name,
            "dtype": np.dtype(np.uint8).str,
            "shape": list(expected_coordinate_frame.shape),
            "nbytes": int(output.numel() * output.element_size()),
            "device": str(output.device),
        }],
        "input_transfer_runtime_ns": input_transfer_ns,
        "computation_runtime_ns": computation_runtime_ns,
        "scientific_validation_status": "not_certified",
        "reference_comparison_status": "not_evaluated",
        "measurement_contract": output_measurement.to_dict(),
    }, min(MAX_NATIVE_MASK_METADATA_BYTES, limits.max_metadata_bytes))
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
            recipe, stage_id, input_checkpoint_sha256, NativeMaskStageStatus.CANCELLED,
            f"Cancelled during checkpoint serialization: {error}",
        )
    except CheckpointError as error:
        return _outcome_without_checkpoint(
            recipe, stage_id, input_checkpoint_sha256, NativeMaskStageStatus.FAILED,
            f"Native output checkpoint serialization failed: {error}",
        )
    except Exception as error:
        return _outcome_without_checkpoint(
            recipe, stage_id, input_checkpoint_sha256,
            NativeMaskStageStatus.CANCELLED if _cancelled(cancel) else NativeMaskStageStatus.FAILED,
            f"Native output checkpoint serialization failed: {type(error).__name__}: {error}",
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
    return NativeMaskStageOutcome(status, record, reference)


__all__ = [
    "MAX_NATIVE_MASK_METADATA_BYTES",
    "NATIVE_MASK_BACKEND_ID",
    "NATIVE_MASK_BACKEND_VERSION",
    "NATIVE_MASK_STAGE_FORMAT",
    "NATIVE_MASK_STAGE_VERSION",
    "SUPPORTED_NATIVE_MASK_OPERATIONS",
    "NativeMaskIdentity",
    "NativeMaskStageError",
    "NativeMaskStageOutcome",
    "NativeMaskStageStatus",
    "NativeMaskUnavailableError",
    "expected_native_mask_stage_identity",
    "native_mask_identity",
    "run_native_mask_stage",
]
