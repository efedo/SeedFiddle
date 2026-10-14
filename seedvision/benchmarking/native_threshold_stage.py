"""One-stage CUDA threshold bridge for explicit benchmark checkpoints.

This adapter runs only the CUDA binary, adaptive-mean, and supported-domain
Otsu primitives. It is a native-compatible operation bridge, not a DAG engine,
an original PlantCV execution claim, or scientific validation. Full image and
mask rasters remain on the requested CUDA device; output transfer is delegated
to the bounded checkpoint serializer after explicit caller opt-in.
"""

from __future__ import annotations

from dataclasses import (
    MISSING,
    dataclass,
    fields as dataclass_fields,
    is_dataclass,
    replace as dataclass_replace,
)
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
    RecipeIdentity,
    RecipeProvenance,
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
    PINNED_PLANTCV_COMMIT,
    PINNED_PLANTCV_VERSION,
    PINNED_REQUIREMENTS_SHA256,
    PINNED_SOURCE_AUDIT_SHA256,
)
from seedvision.cuda import plantcv_thresholds as _threshold_module
from seedvision.cuda.plantcv_thresholds import (
    MAX_ABSOLUTE_OFFSET,
    MAX_ADAPTIVE_KERNEL,
    MIN_ADAPTIVE_KERNEL,
    OtsuAmbiguityError,
    AdaptiveMeanPrecisionError,
    PLANTCV_COMMIT,
    PLANTCV_THRESHOLD_SOURCE_SHA256,
    PLANTCV_VERSION,
    threshold_binary,
    threshold_mean,
    threshold_otsu,
    threshold_otsu_result,
)


NATIVE_THRESHOLD_STAGE_FORMAT = "seedfiddle.native-threshold-stage-output"
NATIVE_THRESHOLD_STAGE_VERSION = 1
NATIVE_THRESHOLD_BACKEND_ID = "seedvision.cuda.thresholds"
NATIVE_THRESHOLD_BACKEND_VERSION = "1"
MAX_NATIVE_THRESHOLD_METADATA_BYTES = 1024 * 1024
MAX_IMPLEMENTATION_SOURCE_BYTES = 2 * 1024 * 1024
SUPPORTED_NATIVE_THRESHOLD_OPERATIONS = frozenset({
    "threshold.binary",
    "threshold.mean",
    "threshold.otsu",
})

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_CAPABILITY = re.compile(r"[0-9]+\.[0-9]+\Z")
_PARAMETER_KEYS = {
    "threshold.binary": frozenset({"threshold", "object_type"}),
    "threshold.mean": frozenset({"ksize", "offset", "object_type"}),
    "threshold.otsu": frozenset({"object_type"}),
}
_CANONICAL_RECIPE_IDS = {
    "threshold.binary": "plantcv_seed_binary",
    "threshold.mean": "plantcv_seed_mean",
    "threshold.otsu": "plantcv_seed_otsu",
}
_NATIVE_SOURCE_CONSTANTS = frozenset({
    "NATIVE_THRESHOLD_STAGE_FORMAT", "NATIVE_THRESHOLD_STAGE_VERSION",
    "NATIVE_THRESHOLD_BACKEND_ID", "NATIVE_THRESHOLD_BACKEND_VERSION",
    "MAX_NATIVE_THRESHOLD_METADATA_BYTES", "MAX_IMPLEMENTATION_SOURCE_BYTES",
    "SUPPORTED_NATIVE_THRESHOLD_OPERATIONS", "_PARAMETER_KEYS", "_CANONICAL_RECIPE_IDS",
    "_SHA256", "_CAPABILITY",
})
_PRIMITIVE_SOURCE_CONSTANTS = frozenset({
    "PLANTCV_VERSION", "PLANTCV_COMMIT", "PLANTCV_THRESHOLD_SOURCE_SHA256",
    "MIN_ADAPTIVE_KERNEL", "MAX_ADAPTIVE_KERNEL", "MAX_ABSOLUTE_OFFSET",
    "_OTSU_RELATIVE_GAP_DENOMINATOR", "_OTSU_MIN_CLASS_WEIGHT_DENOMINATOR",
    "MIN_OTSU_RELATIVE_PARTITION_GAP", "OTSU_CUTOFF_SELECTION_PROFILE",
})


class NativeThresholdStageError(ValueError):
    """A native threshold stage or its immutable bindings are invalid."""


class NativeThresholdUnavailableError(RuntimeError):
    """The explicitly requested CUDA device/runtime is unavailable."""


@dataclass(frozen=True, slots=True)
class NativeThresholdIdentity:
    """Exact caller-pinned code and CUDA environment for one native stage."""

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
    reference_source_sha256: str = PLANTCV_THRESHOLD_SOURCE_SHA256

    def __post_init__(self) -> None:
        for name in ("implementation_sha256", "reference_source_sha256"):
            value = getattr(self, name)
            if not isinstance(value, str) or not _SHA256.fullmatch(value):
                raise NativeThresholdStageError(f"{name} must be a lowercase SHA-256 digest.")
        for name in (
            "python_version", "torch_version", "numpy_version", "cuda_runtime_version",
            "platform", "machine", "device_name",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or value != value.strip():
                raise NativeThresholdStageError(f"{name} must be non-empty trimmed text.")
        if type(self.device_index) is not int or self.device_index < 0:
            raise NativeThresholdStageError("device_index must be a non-negative integer.")
        if not isinstance(self.compute_capability, str) or not _CAPABILITY.fullmatch(self.compute_capability):
            raise NativeThresholdStageError("compute_capability must be a major.minor string.")

    @property
    def backend_id(self) -> str:
        return NATIVE_THRESHOLD_BACKEND_ID

    @property
    def backend_version(self) -> str:
        return NATIVE_THRESHOLD_BACKEND_VERSION

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
            "plantcv_threshold_source_sha256": self.reference_source_sha256,
            "plantcv": PLANTCV_VERSION,
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
        """Bind this native identity to a separately identified PlantCV target."""
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
class NativeThresholdOutcome:
    """Native execution state plus its immutable output checkpoint, if any."""

    run_record: StageRunRecord
    checkpoint_reference: CheckpointReference | None

    def __post_init__(self) -> None:
        if not isinstance(self.run_record, StageRunRecord):
            raise TypeError("run_record must be a StageRunRecord.")
        completed = self.run_record.state in {StageRunState.EXECUTED, StageRunState.EMPTY}
        if completed != (self.checkpoint_reference is not None):
            raise ValueError("Only executed or empty native stage runs carry a checkpoint reference.")
        if self.checkpoint_reference is not None and (
            self.run_record.checkpoint_sha256 != self.checkpoint_reference.checkpoint_sha256
        ):
            raise ValueError("Native stage run and checkpoint reference hashes differ.")


def _literal_node(node: ast.AST, namespace: Mapping[str, Any] | None = None) -> Any:
    """Evaluate the small, side-effect-free literal grammar used by pins."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name) and namespace is not None and node.id in namespace:
        value = namespace[node.id]
        if value is None or type(value) in {str, int, float, bool}:
            return value
        raise NativeThresholdStageError("Pinned module literal references a non-primitive constant.")
    if isinstance(node, ast.List):
        return [_literal_node(item, namespace) for item in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_literal_node(item, namespace) for item in node.elts)
    if isinstance(node, ast.Set):
        return set(_literal_node(item, namespace) for item in node.elts)
    if isinstance(node, ast.Dict):
        return {_literal_node(key, namespace): _literal_node(value, namespace) for key, value in zip(node.keys, node.values)}
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _literal_node(node.operand, namespace)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp):
        left, right = _literal_node(node.left, namespace), _literal_node(node.right, namespace)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.FloorDiv):
            return left // right
        if isinstance(node.op, ast.LShift):
            return left << right
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in {"frozenset", "set"} and len(node.args) == 1:
            values = _literal_node(node.args[0], namespace)
            return frozenset(values) if node.func.id == "frozenset" else set(values)
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "re"
            and node.func.attr == "compile"
            and node.args
        ):
            return re.compile(_literal_node(node.args[0], namespace))
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "CheckpointLimits"
            and not node.args
            and not node.keywords
            and namespace is not None
            and "CheckpointLimits" in namespace
        ):
            return namespace["CheckpointLimits"]()
    raise NativeThresholdStageError("Pinned module contains a non-literal execution constant.")


def _normalize_identity_value(value: Any) -> Any:
    if isinstance(value, re.Pattern):
        return {"regex": value.pattern, "flags": value.flags}
    if isinstance(value, Mapping):
        return {str(key): _normalize_identity_value(item) for key, item in sorted(value.items())}
    if isinstance(value, (set, frozenset)):
        normalized = [_normalize_identity_value(item) for item in value]
        return sorted(normalized, key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")))
    if isinstance(value, (tuple, list)):
        return [_normalize_identity_value(item) for item in value]
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "dataclass": type(value).__qualname__,
            "fields": {
                field.name: _normalize_identity_value(getattr(value, field.name))
                for field in dataclass_fields(value)
            },
        }
    if value is None or type(value) in {str, int, float, bool}:
        if isinstance(value, float) and not math.isfinite(value):
            raise NativeThresholdStageError("Implementation constants cannot contain non-finite values.")
        return value
    raise NativeThresholdStageError("A runtime implementation constant has an unsupported value type.")


def _normalize_code_constant(value: Any) -> Any:
    if isinstance(value, CodeType):
        return {"code": _code_payload(value)}
    if isinstance(value, bytes):
        return {"bytes_hex": value.hex()}
    if isinstance(value, tuple):
        return [_normalize_code_constant(item) for item in value]
    if isinstance(value, frozenset):
        items = [_normalize_code_constant(item) for item in value]
        return sorted(items, key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")))
    if value is None or type(value) in {str, int, float, bool}:
        return _normalize_identity_value(value)
    raise NativeThresholdStageError("Loaded implementation code contains an unsupported constant.")


def _code_payload(code: CodeType) -> dict[str, Any]:
    return {
        "argcount": code.co_argcount,
        "posonlyargcount": code.co_posonlyargcount,
        "kwonlyargcount": code.co_kwonlyargcount,
        "nlocals": code.co_nlocals,
        "stacksize": code.co_stacksize,
        "flags": code.co_flags,
        "bytecode_hex": code.co_code.hex(),
        "constants": [_normalize_code_constant(item) for item in code.co_consts],
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


def _source_callable_nodes(tree: ast.Module) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    result: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            result[node.name] = node
        elif isinstance(node, ast.ClassDef):
            for member in node.body:
                if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    result[f"{node.name}.{member.name}"] = member
    return result


def _verify_function_defaults(
    qualified_name: str,
    function: Any,
    definition: ast.FunctionDef | ast.AsyncFunctionDef,
    namespace: Mapping[str, Any],
) -> dict[str, Any]:
    argument_list = [*definition.args.posonlyargs, *definition.args.args]
    expected_defaults = tuple(_literal_node(item, namespace) for item in definition.args.defaults)
    actual_defaults = getattr(function, "__defaults__", None) or ()
    if len(actual_defaults) != len(expected_defaults):
        raise NativeThresholdStageError(f"Loaded defaults for {qualified_name} differ from current source.")
    if any(
        _normalize_identity_value(actual) != _normalize_identity_value(expected)
        for actual, expected in zip(actual_defaults, expected_defaults)
    ):
        raise NativeThresholdStageError(f"Loaded defaults for {qualified_name} differ from current source.")
    expected_keyword_defaults: dict[str, Any] = {}
    for argument, default in zip(definition.args.kwonlyargs, definition.args.kw_defaults):
        if default is not None:
            expected_keyword_defaults[argument.arg] = _literal_node(default, namespace)
    actual_keyword_defaults = getattr(function, "__kwdefaults__", None) or {}
    if set(actual_keyword_defaults) != set(expected_keyword_defaults) or any(
        _normalize_identity_value(actual_keyword_defaults[name]) != _normalize_identity_value(expected)
        for name, expected in expected_keyword_defaults.items()
    ):
        raise NativeThresholdStageError(f"Loaded keyword defaults for {qualified_name} differ from current source.")
    return {
        "positional": [_normalize_identity_value(item) for item in actual_defaults],
        "keyword": {
            name: _normalize_identity_value(actual_keyword_defaults[name])
            for name in sorted(actual_keyword_defaults)
        },
    }


def _read_source_bounded(path: Path) -> bytes:
    with path.open("rb") as stream:
        return stream.read(MAX_IMPLEMENTATION_SOURCE_BYTES + 1)


def _module_source_record(
    name: str,
    path: Path,
    namespace: Mapping[str, Any],
    code_names: tuple[str, ...],
    constant_names: frozenset[str],
    dataclass_names: tuple[str, ...] = (),
) -> dict[str, Any]:
    try:
        source = _read_source_bounded(path)
    except OSError as error:
        raise NativeThresholdStageError(f"Cannot read implementation source {name}: {error}") from error
    if len(source) > MAX_IMPLEMENTATION_SOURCE_BYTES:
        raise NativeThresholdStageError(f"Implementation source {name} exceeds its hash bound.")
    try:
        compiled = compile(source, str(path), "exec", dont_inherit=True)
        tree = ast.parse(source, filename=str(path))
    except (SyntaxError, UnicodeError, ValueError) as error:
        raise NativeThresholdStageError(f"Cannot verify loaded implementation source {name}: {error}") from error
    compiled_codes: dict[str, CodeType] = {}

    def visit(code: CodeType) -> None:
        compiled_codes[code.co_qualname] = code
        for child in code.co_consts:
            if isinstance(child, CodeType):
                visit(child)

    visit(compiled)
    callable_nodes = _source_callable_nodes(tree)
    code_records: list[dict[str, str]] = []
    for qualified_name in code_names:
        value = namespace.get(qualified_name)
        value = getattr(value, "__wrapped__", value)
        if isinstance(value, property):
            value = value.fget
        code = getattr(value, "__code__", None)
        candidate = compiled_codes.get(qualified_name)
        if code is None or candidate is None or _code_payload(code) != _code_payload(candidate):
            raise NativeThresholdStageError(
                f"Loaded implementation {name}:{qualified_name} differs from freshly compiled source; refusing stale code."
            )
        definition = callable_nodes.get(qualified_name)
        if definition is None:
            raise NativeThresholdStageError(f"Implementation source {name} is missing callable {qualified_name}.")
        defaults = _verify_function_defaults(qualified_name, value, definition, namespace)
        code_bytes = json.dumps(_code_payload(code), sort_keys=True, separators=(",", ":")).encode("utf-8")
        defaults_bytes = json.dumps(defaults, sort_keys=True, separators=(",", ":")).encode("utf-8")
        code_records.append({
            "name": qualified_name,
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
                source_constants[target.id] = _normalize_identity_value(_literal_node(expression, namespace))
    if set(source_constants) != set(constant_names):
        raise NativeThresholdStageError(f"Implementation source {name} does not define the expected constant inventory.")
    runtime_constants = {
        constant: _normalize_identity_value(namespace[constant])
        for constant in constant_names
        if constant in namespace
    }
    if runtime_constants != source_constants:
        raise NativeThresholdStageError(
            f"Loaded implementation {name} constants differ from source; refusing stale bytecode or globals."
        )
    class_nodes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    class_records: list[dict[str, Any]] = []
    for class_name in dataclass_names:
        node = class_nodes.get(class_name)
        runtime_class = namespace.get(class_name)
        if node is None or runtime_class is None:
            raise NativeThresholdStageError(f"Implementation source {name} is missing dataclass {class_name}.")
        expected_fields: list[tuple[str, Any]] = []
        for statement in node.body:
            if not isinstance(statement, ast.AnnAssign) or not isinstance(statement.target, ast.Name):
                continue
            default = MISSING
            if statement.value is not None:
                if isinstance(statement.value, ast.Name) and statement.value.id in namespace:
                    default = namespace[statement.value.id]
                else:
                    default = _literal_node(statement.value, namespace)
            expected_fields.append((statement.target.id, default))
        actual_fields = dataclass_fields(runtime_class)
        if tuple(field.name for field in actual_fields) != tuple(name for name, _default in expected_fields):
            raise NativeThresholdStageError(f"Loaded dataclass {class_name} does not match its current source fields.")
        normalized_fields: list[dict[str, Any]] = []
        for actual, (field_name, expected_default) in zip(actual_fields, expected_fields):
            actual_default = actual.default
            if expected_default is MISSING:
                matches_default = actual_default is MISSING
                normalized_default: Any = {"missing": True}
            else:
                matches_default = _normalize_identity_value(actual_default) == _normalize_identity_value(expected_default)
                normalized_default = _normalize_identity_value(actual_default)
            if not matches_default:
                raise NativeThresholdStageError(f"Loaded dataclass {class_name}.{field_name} differs from current source.")
            normalized_fields.append({
                "name": field_name,
                "default": normalized_default,
                "init": actual.init,
                "repr": actual.repr,
                "compare": actual.compare,
            })
        class_records.append({"name": class_name, "fields": normalized_fields})
    return {
        "name": name,
        "source_sha256": sha256(source).hexdigest(),
        "loaded_code": code_records,
        "runtime_constants": runtime_constants,
        "dataclasses": class_records,
    }


def native_supported_domain(operation_id: str) -> str:
    """Canonical archived support declaration, shared with inspection."""
    return {
        "threshold.binary": "uint8 cutoff in [0,255]; light uses > and dark uses <=",
        "threshold.mean": "integer requested ksize [3,255] (even rounds up to odd), finite offset [-255,255], replicated borders, certified scalar/SIMD decisions",
        "threshold.otsu": "unique separated eligible histogram partition; ambiguous histograms are unsupported",
    }[operation_id]


def _implementation_digest() -> str:
    module_path = Path(__file__).resolve()
    primitive_path = Path(_threshold_module.__file__).resolve()
    bridge_names = (
        "_literal_node", "_normalize_identity_value", "_normalize_code_constant", "_code_payload",
        "_read_source_bounded",
        "_module_source_record", "_implementation_digest", "native_threshold_identity", "_canonical_catalogue",
        "_canonical_recipe_and_stage", "_validate_recipe_stage", "_parameters", "_domain_reason",
        "expected_native_threshold_stage_identity", "_strict_metadata", "_cancelled",
        "_outcome_without_checkpoint", "_mask_semantics_match", "run_native_threshold_stage", "native_supported_domain",
    )
    bridge_namespace = globals() | {
        "NativeThresholdIdentity": NativeThresholdIdentity,
        "NativeThresholdOutcome": NativeThresholdOutcome,
    }
    for cls in (NativeThresholdIdentity, NativeThresholdOutcome):
        for name in ("__post_init__", "to_dict", "recipe_provenance", "environment_versions", "device", "backend_id", "backend_version"):
            if hasattr(cls, name):
                bridge_namespace[f"{cls.__name__}.{name}"] = getattr(cls, name)
    bridge_names = bridge_names + tuple(
        f"{cls.__name__}.{name}"
        for cls in (NativeThresholdIdentity, NativeThresholdOutcome)
        for name in ("__post_init__", "to_dict", "recipe_provenance", "environment_versions", "device", "backend_id", "backend_version")
        if hasattr(cls, name)
    )
    primitive_names = (
        "_validate_gray", "_object_type", "_mask_from_condition", "_otsu_cutoff_from_histogram",
        "threshold_binary", "threshold_otsu", "threshold_otsu_result",
        "otsu_calculation_facts", "validate_otsu_calculation_facts", "_validate_mean_parameters", "_replicated_box_sum",
        "_adaptive_mean_condition", "threshold_mean",
    )
    primitive_namespace = vars(_threshold_module)
    if any(globals().get(name) is not primitive_namespace.get(name) for name in (
        "threshold_binary", "threshold_mean", "threshold_otsu", "threshold_otsu_result",
    )):
        raise NativeThresholdStageError("Imported CUDA threshold callables were replaced after module loading.")
    primitive_records = _module_source_record(
        "plantcv_thresholds.py", primitive_path, primitive_namespace, primitive_names, _PRIMITIVE_SOURCE_CONSTANTS,
        dataclass_names=("OtsuResult",),
    )
    bridge_record = _module_source_record(
        "native_threshold_stage.py", module_path, bridge_namespace, bridge_names, _NATIVE_SOURCE_CONSTANTS,
        dataclass_names=("NativeThresholdIdentity", "NativeThresholdOutcome"),
    )
    encoded = json.dumps(
        {"format": "seedfiddle.native-threshold-loaded-implementation", "version": 1,
         "files": [bridge_record, primitive_records]},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def native_threshold_identity(device: Any) -> NativeThresholdIdentity:
    """Describe the exact CUDA implementation environment for an explicit device."""
    import torch

    if type(device) is not torch.device or device.type != "cuda" or device.index is None:
        raise NativeThresholdStageError("device must be an explicit torch.device such as cuda:0.")
    if not torch.cuda.is_available():
        raise NativeThresholdUnavailableError("CUDA is not available in this PyTorch runtime.")
    if device.index < 0 or device.index >= torch.cuda.device_count():
        raise NativeThresholdUnavailableError(f"Requested CUDA device {device} is not available.")
    try:
        properties = torch.cuda.get_device_properties(device)
        runtime = torch.version.cuda or torch.version.hip
        if not runtime:
            raise NativeThresholdUnavailableError("PyTorch reports no CUDA or HIP runtime version.")
        capability = f"{int(properties.major)}.{int(properties.minor)}"
        return NativeThresholdIdentity(
            implementation_sha256=_implementation_digest(),
            python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            torch_version=str(torch.__version__),
            numpy_version=np.__version__,
            cuda_runtime_version=str(runtime),
            platform=sys.platform,
            machine=host_platform.machine() or "unknown",
            device_index=int(device.index),
            device_name=str(properties.name),
            compute_capability=capability,
        )
    except NativeThresholdUnavailableError:
        raise
    except NativeThresholdStageError:
        raise
    except Exception as error:
        raise NativeThresholdUnavailableError(f"Cannot inspect requested CUDA device {device}: {error}") from error


@lru_cache(maxsize=1)
def _canonical_catalogue():
    try:
        return build_recipe_catalogue()
    except (RecipeCatalogueError, OSError, ValueError) as error:
        raise NativeThresholdStageError(f"Committed PlantCV operation contracts are unavailable: {error}") from error


def _canonical_recipe_and_stage(operation_id: str):
    try:
        recipe_id = _CANONICAL_RECIPE_IDS[operation_id]
    except KeyError as error:
        raise NativeThresholdStageError(f"Operation {operation_id!r} is not supported by the native threshold bridge.") from error
    recipe = _canonical_catalogue().recipe(recipe_id)
    matches = tuple(stage for stage in recipe.stages if stage.operation_id == operation_id)
    if len(matches) != 1:
        raise NativeThresholdStageError("Committed operation catalogue does not contain one exact threshold stage.")
    return recipe, matches[0]


def _validate_recipe_stage(recipe: RecipeSpec, stage_id: str, operation_id: str) -> Any:
    canonical_recipe, canonical_stage = _canonical_recipe_and_stage(operation_id)
    if recipe.recipe_id != canonical_recipe.recipe_id or recipe.recipe_version != canonical_recipe.recipe_version:
        raise NativeThresholdStageError("Native recipe does not use the pinned PlantCV threshold recipe identity.")
    if set(item.stage_id for item in recipe.stages) != set(item.stage_id for item in canonical_recipe.stages):
        raise NativeThresholdStageError("Native recipe topology differs from the pinned PlantCV recipe description.")
    if stage_id != canonical_stage.stage_id:
        raise NativeThresholdStageError("Requested stage ID does not match the canonical public PlantCV operation.")
    try:
        stage = recipe.stage(stage_id)
    except KeyError as error:
        raise NativeThresholdStageError(f"Recipe does not contain stage {stage_id!r}.") from error
    # Native recipes retain the exact authored PlantCV target graph and change
    # only the calculation-affecting values selected for this execution.
    for current, canonical in zip(recipe.stages, canonical_recipe.stages):
        if current.stage_id != canonical.stage_id:
            raise NativeThresholdStageError("Native recipe stage order differs from the pinned recipe.")
        if current.stage_id == stage_id:
            if len(current.parameters) != len(canonical.parameters):
                raise NativeThresholdStageError("Native stage parameter inventory differs from its public operation.")
            for selected, authored in zip(current.parameters, canonical.parameters):
                if dataclass_replace(selected, value=authored.value) != authored:
                    raise NativeThresholdStageError("Native stage parameter contract differs from the public operation.")
            if dataclass_replace(current, parameters=canonical.parameters) != canonical:
                raise NativeThresholdStageError("Native stage source mapping or port semantics differ from the pinned operation.")
        elif current != canonical:
            raise NativeThresholdStageError("Another stage in the native recipe differs from the pinned recipe description.")
    native_normalized = dataclass_replace(
        recipe,
        provenance=canonical_recipe.provenance,
        stages=canonical_recipe.stages,
    )
    if native_normalized != canonical_recipe:
        raise NativeThresholdStageError("Native recipe dependencies, layout, or description differ from the pinned recipe.")
    if stage.resolution is not SourceResolution.PINNED:
        raise NativeThresholdStageError("Runnable native threshold stages require pinned target source resolution.")
    return stage


def _parameters(recipe: RecipeSpec, stage_id: str, operation_id: str) -> dict[str, Any]:
    values = recipe.parameters_for_runner(stage_id)
    if set(values) != _PARAMETER_KEYS[operation_id] or any(value is None for value in values.values()):
        raise NativeThresholdStageError("Every calculation-affecting threshold parameter must be selected explicitly.")
    polarity = values.get("object_type")
    if type(polarity) is not str or polarity not in {"light", "dark"}:
        raise NativeThresholdStageError("Threshold object_type must be exactly 'light' or 'dark'.")
    if operation_id == "threshold.binary":
        cutoff = values["threshold"]
        if type(cutoff) is not int or not 0 <= cutoff <= 255:
            raise NativeThresholdStageError("Binary threshold must be an integer in [0,255].")
    elif operation_id == "threshold.mean":
        ksize, offset = values["ksize"], values["offset"]
        if type(ksize) is not int:
            raise NativeThresholdStageError("Mean-threshold ksize must be an integer.")
        if isinstance(offset, bool) or type(offset) not in (int, float):
            raise NativeThresholdStageError("Mean-threshold offset must be a finite number.")
        try:
            finite_offset = float(offset)
        except (OverflowError, ValueError):
            finite_offset = math.inf
        if not math.isfinite(finite_offset):
            raise NativeThresholdStageError("Mean-threshold offset must be finite.")
    try:
        encoded = json.dumps(values, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as error:
        raise NativeThresholdStageError("Native threshold parameters must be strict JSON primitives.") from error
    if len(encoded) > MAX_NATIVE_THRESHOLD_METADATA_BYTES:
        raise NativeThresholdStageError("Native threshold parameters exceed the bounded metadata limit.")
    return values


def _domain_reason(operation_id: str, parameters: Mapping[str, Any]) -> str | None:
    if operation_id == "threshold.mean":
        ksize = parameters["ksize"]
        effective_ksize = ksize + int(ksize % 2 == 0)
        if ksize < MIN_ADAPTIVE_KERNEL or effective_ksize > MAX_ADAPTIVE_KERNEL:
            return f"Native adaptive-mean support requires requested ksize >= {MIN_ADAPTIVE_KERNEL} and effective odd ksize <= {MAX_ADAPTIVE_KERNEL}."
        offset = float(parameters["offset"])
        if abs(offset) > MAX_ABSOLUTE_OFFSET:
            return f"Native adaptive-mean support requires |offset| <= {MAX_ABSOLUTE_OFFSET:g}."
    if operation_id == "threshold.otsu":
        # Otsu's data-dependent supported domain is decided by the primitive.
        # Ties/near-ties become an explicit UNSUPPORTED run, never a fake mask.
        return None
    return None


def expected_native_threshold_stage_identity(
    input_identity: CheckpointIdentity,
    recipe: RecipeSpec,
    stage_id: str,
    upstream_checkpoint_sha256: str,
    implementation_identity: NativeThresholdIdentity,
) -> CheckpointIdentity:
    """Derive the exact output checkpoint identity from stage and implementation pins."""
    if not isinstance(input_identity, CheckpointIdentity) or not isinstance(recipe, RecipeSpec):
        raise TypeError("input_identity and recipe must be immutable checkpoint/recipe contracts.")
    if not isinstance(implementation_identity, NativeThresholdIdentity):
        raise TypeError("implementation_identity must be a NativeThresholdIdentity.")
    if not isinstance(upstream_checkpoint_sha256, str) or not _SHA256.fullmatch(upstream_checkpoint_sha256):
        raise NativeThresholdStageError("upstream_checkpoint_sha256 must be a lowercase SHA-256 digest.")
    if recipe.provenance != implementation_identity.recipe_provenance():
        raise NativeThresholdStageError("Recipe provenance does not exactly match the expected native implementation identity.")
    try:
        stage = recipe.stage(stage_id)
    except KeyError as error:
        raise NativeThresholdStageError(f"Recipe does not contain stage {stage_id!r}.") from error
    operation_id = stage.operation_id
    if operation_id not in SUPPORTED_NATIVE_THRESHOLD_OPERATIONS:
        raise NativeThresholdStageError(f"Operation {operation_id!r} is not supported by the native threshold bridge.")
    _validate_recipe_stage(recipe, stage_id, operation_id)
    parameters = _parameters(recipe, stage_id, operation_id)
    image_semantics = dict(input_identity.image_semantics)
    try:
        input_dtype = np.dtype(image_semantics.get("dtype")).str
    except (TypeError, ValueError) as error:
        raise NativeThresholdStageError("Threshold input identity must declare a uint8 gray image.") from error
    value_range = image_semantics.get("value_range")
    if (
        str(image_semantics.get("color_order", "")).upper() not in {"GRAY", "GRAYSCALE", "Y"}
        or input_dtype != np.dtype(np.uint8).str
        or not isinstance(value_range, (tuple, list))
        or tuple(value_range) != (0, 255)
    ):
        raise NativeThresholdStageError("Threshold input identity must bind uint8 gray pixels in [0,255].")
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
        mask_semantics={
            "foreground_polarity": parameters["object_type"],
            "representation": "uint8_0_255",
            "connectivity": "not_applicable_to_threshold",
            "validity_region": "full_input_frame",
            "border_policy": f"native_supported_{operation_id}",
        },
        instance_semantics=dict(input_identity.instance_semantics),
        upstream_checkpoint_sha256=upstream_checkpoint_sha256,
    )


def _strict_metadata(payload: Mapping[str, Any], maximum: int) -> dict[str, Any]:
    try:
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError, RecursionError) as error:
        raise NativeThresholdStageError("Native threshold metadata must be strict finite JSON.") from error
    if len(encoded) > maximum:
        raise NativeThresholdStageError(f"Native threshold metadata exceeds its {maximum}-byte limit.")
    return json.loads(encoded.decode("utf-8"))


def _cancelled(cancel: Any) -> bool:
    if cancel is None:
        return False
    if callable(cancel):
        return bool(cancel())
    is_set = getattr(cancel, "is_set", None)
    if not callable(is_set):
        raise NativeThresholdStageError("cancel must be a callable or an event with is_set().")
    return bool(is_set())


def _outcome_without_checkpoint(
    recipe: RecipeSpec,
    stage_id: str,
    upstream_checkpoint_sha256: str,
    reason: str,
    state: StageRunState,
) -> NativeThresholdOutcome:
    compact_reason = " ".join(str(reason).split())[:512] or state.value
    run_record = StageRunRecord(
        stage_id=stage_id,
        recipe_identity=recipe.identity,
        state=state,
        upstream_checkpoint_sha256=upstream_checkpoint_sha256,
        reason=compact_reason,
    )
    validate_stage_run(recipe, run_record, expected_upstream_checkpoint_sha256=upstream_checkpoint_sha256)
    return NativeThresholdOutcome(run_record, None)


def _mask_semantics_match(mask: Any, frame: CoordinateFrame, device: Any) -> bool:
    import torch

    return (
        torch.is_tensor(mask)
        and mask.layout == torch.strided
        and mask.device == device
        and mask.dtype == torch.uint8
        and tuple(mask.shape) == frame.shape
        and mask.is_contiguous()
        and bool(torch.logical_or(mask == 0, mask == 255).all().item())
    )


def run_native_threshold_stage(
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
    expected_output_identity: CheckpointIdentity,
    expected_implementation_identity: NativeThresholdIdentity,
    device: Any,
    input_bindings: Mapping[str, str],
    input_roles: Mapping[str, RasterRole],
    output_measurement: MeasurementContract,
    allow_cpu_transfer: bool,
    limits: CheckpointLimits = CheckpointLimits(),
    cancel: Any = None,
) -> NativeThresholdOutcome:
    """Execute one exact-input native-compatible threshold stage on CUDA."""
    import torch

    if not isinstance(recipe, RecipeSpec) or not isinstance(expected_recipe_identity, RecipeIdentity):
        raise TypeError("recipe and expected_recipe_identity must be immutable recipe contracts.")
    if not isinstance(expected_implementation_identity, NativeThresholdIdentity):
        raise TypeError("expected_implementation_identity must be explicit.")
    if type(device) is not torch.device or device.type != "cuda" or device.index is None:
        raise NativeThresholdStageError("device must be an explicit torch.device such as cuda:0.")
    if not isinstance(expected_coordinate_frame, CoordinateFrame):
        raise TypeError("expected_coordinate_frame must be a CoordinateFrame.")
    if not isinstance(expected_input_identity, CheckpointIdentity) or not isinstance(expected_input_measurement, MeasurementContract):
        raise TypeError("Expected input identity and measurement contracts are required.")
    if not isinstance(expected_output_identity, CheckpointIdentity) or not isinstance(output_measurement, MeasurementContract):
        raise TypeError("Expected output identity and measurement contracts are required.")
    if not isinstance(limits, CheckpointLimits):
        raise TypeError("limits must be a CheckpointLimits contract.")
    if output_measurement.eligible:
        raise NativeThresholdStageError("Native threshold output cannot mark scientific measurement eligibility.")
    if type(allow_cpu_transfer) is not bool or not allow_cpu_transfer:
        raise NativeThresholdStageError("Checkpoint serialization requires allow_cpu_transfer=True.")
    if expected_recipe_identity != recipe.identity:
        raise NativeThresholdStageError("Recipe identity differs from the caller's exact expected identity.")
    if recipe.provenance != expected_implementation_identity.recipe_provenance():
        raise NativeThresholdStageError("Recipe provenance differs from the exact caller-expected native implementation identity.")
    for name, digest in (
        ("expected_source_sha256", expected_source_sha256),
        ("input_checkpoint_sha256", input_checkpoint_sha256),
        ("expected_upstream_checkpoint_sha256", expected_upstream_checkpoint_sha256),
    ):
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise NativeThresholdStageError(f"{name} must be a lowercase SHA-256 digest.")
    if expected_upstream_checkpoint_sha256 != input_checkpoint_sha256:
        raise NativeThresholdStageError("Expected upstream digest differs from the exact input checkpoint address.")
    if expected_input_identity.source_sha256 != expected_source_sha256:
        raise NativeThresholdStageError("Expected input checkpoint source SHA differs from the requested source.")
    if expected_input_identity.coordinate_frame != expected_coordinate_frame:
        raise NativeThresholdStageError("Expected input checkpoint frame differs from the requested coordinate frame.")
    if not isinstance(input_bindings, Mapping) or not isinstance(input_roles, Mapping):
        raise NativeThresholdStageError("Input bindings and raster roles must be explicit mappings.")
    if any(not isinstance(name, str) or not isinstance(value, str) for name, value in input_bindings.items()):
        raise NativeThresholdStageError("Input bindings must map port names to checkpoint array names.")
    if any(not isinstance(name, str) or not isinstance(role, RasterRole) for name, role in input_roles.items()):
        raise NativeThresholdStageError("Input roles must map port names to RasterRole contracts.")

    try:
        stage = recipe.stage(stage_id)
    except KeyError as error:
        raise NativeThresholdStageError(f"Recipe does not contain stage {stage_id!r}.") from error
    if stage.operation_id not in SUPPORTED_NATIVE_THRESHOLD_OPERATIONS:
        raise NativeThresholdStageError(
            f"Operation {stage.operation_id!r} is outside this native threshold bridge contract."
        )
    stage = _validate_recipe_stage(recipe, stage_id, stage.operation_id)
    parameters = _parameters(recipe, stage_id, stage.operation_id)
    if recipe.provenance.mode is not RecipeProvenanceMode.NATIVE_COMPATIBLE:
        raise NativeThresholdStageError("Only explicitly native-compatible recipes can execute through this bridge.")

    checkpoint = load_stage_input(
        input_root,
        input_checkpoint_sha256,
        expected_identity=expected_input_identity,
        expected_measurement=expected_input_measurement,
        limits=limits,
    )
    if checkpoint.identity.source_sha256 != expected_source_sha256:
        raise NativeThresholdStageError("Loaded input checkpoint source SHA differs from the requested source.")
    if checkpoint.identity.coordinate_frame != expected_coordinate_frame:
        raise NativeThresholdStageError("Loaded input checkpoint frame differs from the requested frame.")
    expected_inputs = {port.name for port in stage.ports if port.direction == "input" and port.required}
    if expected_inputs != {"gray"} or set(input_bindings) != expected_inputs or set(input_roles) != expected_inputs:
        raise NativeThresholdStageError("Threshold stage requires one explicit gray-raster input binding and role.")
    bound_name = input_bindings["gray"]
    if bound_name not in checkpoint.arrays or checkpoint.raster_roles.get(bound_name) != input_roles["gray"]:
        raise NativeThresholdStageError("Bound input raster is missing or its role differs from the checkpoint.")
    gray = checkpoint.arrays[bound_name]
    role = input_roles["gray"]
    if (
        role.kind != "image"
        or role.layout != "HW"
        or role.dtype != np.dtype(np.uint8).str
        or role.value_range != (0, 255)
        or gray.dtype != np.dtype(np.uint8)
        or tuple(gray.shape) != expected_coordinate_frame.shape
        or not gray.flags.c_contiguous
        or gray.nbytes > min(MAX_ARRAY_BYTES, limits.max_array_bytes, limits.max_total_array_bytes)
        or gray.size > MAX_ARRAY_ELEMENTS
    ):
        raise NativeThresholdStageError("Native input must be a bounded contiguous uint8 HxW gray raster in the exact frame.")

    derived_identity = expected_native_threshold_stage_identity(
        checkpoint.identity,
        recipe,
        stage_id,
        input_checkpoint_sha256,
        expected_implementation_identity,
    )
    if expected_output_identity != derived_identity:
        raise NativeThresholdStageError("Expected output identity differs from the exact native stage/source/frame/upstream chain.")
    validate_checkpoint_identity(
        recipe,
        stage_id,
        derived_identity,
        expected_identity=expected_output_identity,
        expected_upstream_checkpoint_sha256=input_checkpoint_sha256,
    )

    unsupported_domain = _domain_reason(stage.operation_id, parameters)
    if unsupported_domain is not None:
        return _outcome_without_checkpoint(
            recipe,
            stage_id,
            input_checkpoint_sha256,
            unsupported_domain,
            StageRunState.UNSUPPORTED,
        )
    if _cancelled(cancel):
        return _outcome_without_checkpoint(
            recipe, stage_id, input_checkpoint_sha256, "Cancelled before native CUDA execution.", StageRunState.FAILED,
        )
    if not torch.cuda.is_available() or device.index >= torch.cuda.device_count():
        return _outcome_without_checkpoint(
            recipe,
            stage_id,
            input_checkpoint_sha256,
            f"Requested CUDA device {device} is unavailable.",
            StageRunState.UNAVAILABLE,
        )
    actual_implementation_identity = native_threshold_identity(device)
    if actual_implementation_identity != expected_implementation_identity:
        raise NativeThresholdStageError("Current native implementation or CUDA environment differs from its caller pin.")

    try:
        transfer_started_ns = time.perf_counter_ns()
        # The checkpoint loader owns a read-only NumPy array; make one bounded
        # writable host staging copy before the explicit CPU-to-CUDA transfer.
        input_tensor = torch.from_numpy(np.array(gray, copy=True, order="C")).to(device=device)
        torch.cuda.synchronize(device)
        input_transfer_ns = time.perf_counter_ns() - transfer_started_ns
        if input_tensor.device != device or input_tensor.dtype != torch.uint8 or tuple(input_tensor.shape) != expected_coordinate_frame.shape:
            raise NativeThresholdStageError("Input tensor did not retain its exact CUDA device, dtype, and frame.")
        compute_started_ns = time.perf_counter_ns()
        calculation = None
        if stage.operation_id == "threshold.binary":
            mask = threshold_binary(input_tensor, parameters["threshold"], parameters["object_type"])
            effective_parameters = dict(parameters)
        elif stage.operation_id == "threshold.mean":
            mask = threshold_mean(
                input_tensor,
                parameters["ksize"],
                parameters["offset"],
                parameters["object_type"],
            )
            effective_parameters = dict(parameters)
            effective_parameters["ksize"] += int(parameters["ksize"] % 2 == 0)
            effective_parameters["offset"] = float(parameters["offset"])
        else:
            result = threshold_otsu_result(input_tensor, parameters["object_type"])
            mask = result.mask
            calculation = _threshold_module.otsu_calculation_facts(
                result.native_selected_cutoff, result.object_type)
            effective_parameters = dict(parameters)
        torch.cuda.synchronize(device)
        computation_runtime_ns = time.perf_counter_ns() - compute_started_ns
    except OtsuAmbiguityError as error:
        return _outcome_without_checkpoint(
            recipe,
            stage_id,
            input_checkpoint_sha256,
            f"Native Otsu is unsupported for this histogram: {error}",
            StageRunState.UNSUPPORTED,
        )
    except AdaptiveMeanPrecisionError as error:
        return _outcome_without_checkpoint(
            recipe, stage_id, input_checkpoint_sha256,
            f"Native adaptive mean is unsupported at this rounding boundary: {error}",
            StageRunState.UNSUPPORTED,
        )
    except NativeThresholdStageError:
        raise
    except Exception as error:
        cancelled = _cancelled(cancel)
        prefix = "Cancelled during native CUDA threshold execution" if cancelled else "Native CUDA threshold execution failed"
        return _outcome_without_checkpoint(
            recipe,
            stage_id,
            input_checkpoint_sha256,
            f"{prefix}: {type(error).__name__}: {error}",
            StageRunState.FAILED,
        )

    if not _mask_semantics_match(mask, expected_coordinate_frame, device):
        return _outcome_without_checkpoint(
            recipe,
            stage_id,
            input_checkpoint_sha256,
            "Native threshold primitive returned a malformed CUDA mask or changed frame/device semantics.",
            StageRunState.FAILED,
        )
    if _cancelled(cancel):
        return _outcome_without_checkpoint(
            recipe, stage_id, input_checkpoint_sha256, "Cancelled before checkpoint serialization.", StageRunState.FAILED,
        )

    status = ExecutionStatus.EMPTY if int(torch.count_nonzero(mask).item()) == 0 else ExecutionStatus.AVAILABLE
    state = StageRunState.EMPTY if status is ExecutionStatus.EMPTY else StageRunState.EXECUTED
    execution = ExecutionRecord(
        status=status,
        elapsed_ms=computation_runtime_ns / 1_000_000,
        transfer_ms=input_transfer_ns / 1_000_000,
    )
    source_mapping = stage.source_mappings[0]
    domain = native_supported_domain(stage.operation_id)
    metadata = _strict_metadata({
        "format": NATIVE_THRESHOLD_STAGE_FORMAT,
        "version": NATIVE_THRESHOLD_STAGE_VERSION,
        "stage_id": stage_id,
        "operation_id": stage.operation_id,
        "execution_mode": RecipeProvenanceMode.NATIVE_COMPATIBLE.value,
        "recipe_identity": recipe.identity.to_dict(),
        "source_mapping": source_mapping.to_dict(),
        "source_sha256": expected_source_sha256,
        "coordinate_frame": expected_coordinate_frame.to_dict(),
        "upstream_checkpoint_sha256": input_checkpoint_sha256,
        "input_checkpoint_identity": checkpoint.identity.to_dict(),
        "input_bindings": dict(input_bindings),
        "input_raster_roles": {name: value.to_dict() for name, value in input_roles.items()},
        "output_raster_roles": {"mask": RasterRole("mask", np.dtype(np.uint8).str).to_dict()},
        "implementation_identity": expected_implementation_identity.to_dict(),
        "selected_parameters": parameters,
        "effective_parameters": effective_parameters,
        "calculation": calculation,
        "native_supported_domain": domain,
        "output_inventory": [{
            "name": "mask",
            "dtype": np.dtype(np.uint8).str,
            "shape": list(expected_coordinate_frame.shape),
            "nbytes": int(mask.numel() * mask.element_size()),
            "device": str(mask.device),
        }],
        "input_transfer_runtime_ns": input_transfer_ns,
        "computation_runtime_ns": computation_runtime_ns,
        "scientific_validation_status": "not_certified",
        "reference_comparison_status": "not_evaluated",
        "measurement_contract": output_measurement.to_dict(),
    }, min(MAX_NATIVE_THRESHOLD_METADATA_BYTES, limits.max_metadata_bytes))

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
        return _outcome_without_checkpoint(
            recipe,
            stage_id,
            input_checkpoint_sha256,
            f"Cancelled during checkpoint serialization: {error}",
            StageRunState.FAILED,
        )
    except CheckpointError as error:
        return _outcome_without_checkpoint(
            recipe,
            stage_id,
            input_checkpoint_sha256,
            f"Native output checkpoint serialization failed: {error}",
            StageRunState.FAILED,
        )

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
    return NativeThresholdOutcome(run_record, reference)


__all__ = [
    "MAX_NATIVE_THRESHOLD_METADATA_BYTES",
    "NATIVE_THRESHOLD_BACKEND_ID",
    "NATIVE_THRESHOLD_BACKEND_VERSION",
    "NATIVE_THRESHOLD_STAGE_FORMAT",
    "NATIVE_THRESHOLD_STAGE_VERSION",
    "SUPPORTED_NATIVE_THRESHOLD_OPERATIONS",
    "NativeThresholdIdentity",
    "NativeThresholdOutcome",
    "NativeThresholdStageError",
    "NativeThresholdUnavailableError",
    "expected_native_threshold_stage_identity",
    "native_threshold_identity",
    "run_native_threshold_stage",
]
