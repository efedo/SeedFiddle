"""Strict, immutable descriptions of reference recipes and their graph view.

This module describes recipes; it does not execute them or assert that any
PlantCV/SMPTS recipe has been verified. Graph projection uses the application's
existing :class:`PipelineGraph` model and keeps its native execution disabled.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
import hashlib
import json
import math
import re
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from seedvision.benchmarking.checkpoints import CheckpointIdentity
from seedvision.pipeline.model import (
    NodeStatus,
    PipelineConnection,
    PipelineGraph,
    PipelineNode,
)


RECIPE_SCHEMA = "seedfiddle.reference-recipe"
RECIPE_SCHEMA_VERSION = 1
MAX_RECIPE_BYTES = 2 * 1024 * 1024
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER = re.compile(r"[a-z][a-z0-9_.:-]{0,127}\Z")
_PORT_ID = re.compile(r"[a-z][a-z0-9_.:-]{0,127}\Z")
_TYPE_ID = re.compile(r"[A-Za-z][A-Za-z0-9_.:-]{0,127}\Z")
_GIT_COMMIT = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_UNSPECIFIED_UPSTREAM = object()


class RecipeProvenanceMode(StrEnum):
    ORIGINAL_REFERENCE = "original_reference"
    NATIVE_COMPATIBLE = "native_compatible"
    PAPER_RECONSTRUCTION = "paper_reconstruction"
    ENHANCED = "enhanced"


class SourceResolution(StrEnum):
    PINNED = "pinned"
    UNRESOLVED = "unresolved"
    UNAVAILABLE = "unavailable"


class StageRunState(StrEnum):
    UNRESOLVED = "unresolved"
    NOT_RUN = "not_run"
    UNAVAILABLE = "unavailable"
    EXECUTED = "executed"
    BYPASSED = "bypassed"
    EMPTY = "empty"
    FAILED = "failed"
    UNSUPPORTED = "unsupported"


class RecipeContractError(ValueError):
    """A recipe or graph projection violates its declared contract."""


class RecipeGraphMismatch(RecipeContractError):
    """A PipelineGraph differs from the recipe's exact node/edge contract."""


class StaleRecipeIdentity(RecipeContractError):
    """A run or checkpoint belongs to a different recipe/configuration."""


def _text(value: Any, name: str, *, identifier: bool = False) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise RecipeContractError(f"{name} must be non-empty trimmed text.")
    if identifier and not _IDENTIFIER.fullmatch(value):
        raise RecipeContractError(f"{name} must be a stable lowercase identifier.")
    return value


def _digest(value: Any, name: str, *, optional: bool = False) -> str | None:
    if optional and value is None:
        return None
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise RecipeContractError(f"{name} must be a lowercase SHA-256 digest.")
    return value


def _freeze_json(value: Any, *, name: str) -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RecipeContractError(f"{name} cannot contain NaN or infinity.")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise RecipeContractError(f"{name} object keys must be strings.")
            frozen[key] = _freeze_json(item, name=name)
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item, name=name) for item in value)
    raise RecipeContractError(f"{name} must contain strict JSON values.")


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _canonical_bytes(value: Any, *, name: str, maximum: int = MAX_RECIPE_BYTES) -> bytes:
    try:
        encoded = json.dumps(
            _thaw_json(value), ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as error:
        raise RecipeContractError(f"{name} is not strict canonical JSON.") from error
    if len(encoded) > maximum:
        raise RecipeContractError(f"{name} exceeds the {maximum}-byte limit.")
    return encoded


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value, name="recipe identity payload")).hexdigest()


def _exact_object(value: Any, fields: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise RecipeContractError(f"{name} has missing or unknown fields.")
    return value


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RecipeContractError(f"Duplicate JSON field {key!r}.")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise RecipeContractError(f"Non-finite JSON number {value!r} is forbidden.")


@dataclass(frozen=True, slots=True)
class RecipeProvenance:
    mode: RecipeProvenanceMode
    backend_id: str
    backend_version: str
    reference_id: str | None = None
    reference_version: str | None = None
    reference_commit: str | None = None
    source_manifest_sha256: str | None = None
    environment_lock_sha256: str | None = None
    environment_versions: Mapping[str, str] = field(default_factory=dict)
    license_id: str | None = None
    evidence_status: SourceResolution = SourceResolution.UNRESOLVED
    evidence_note: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", RecipeProvenanceMode(self.mode))
        object.__setattr__(self, "evidence_status", SourceResolution(self.evidence_status))
        for field in ("backend_id",):
            object.__setattr__(self, field, _text(getattr(self, field), field, identifier=True))
        for field in ("backend_version",):
            object.__setattr__(self, field, _text(getattr(self, field), field))
        for field in ("reference_id", "reference_version", "reference_commit", "license_id"):
            value = getattr(self, field)
            if value is not None:
                object.__setattr__(self, field, _text(value, field))
        object.__setattr__(self, "source_manifest_sha256", _digest(
            self.source_manifest_sha256, "source manifest SHA-256", optional=True
        ))
        object.__setattr__(self, "environment_lock_sha256", _digest(
            self.environment_lock_sha256, "environment lock SHA-256", optional=True
        ))
        if not isinstance(self.environment_versions, Mapping):
            raise RecipeContractError("Environment versions must be an object.")
        versions: dict[str, str] = {}
        for name, version in self.environment_versions.items():
            versions[_text(name, "environment package name")] = _text(version, "environment package version")
        object.__setattr__(self, "environment_versions", MappingProxyType(versions))
        if not isinstance(self.evidence_note, str) or self.evidence_note != self.evidence_note.strip():
            raise RecipeContractError("Provenance evidence note must be trimmed text.")
        if self.evidence_status in {SourceResolution.UNRESOLVED, SourceResolution.UNAVAILABLE} and not self.evidence_note:
            raise RecipeContractError("Unresolved or unavailable provenance must explain its state.")
        if self.mode is RecipeProvenanceMode.ORIGINAL_REFERENCE and self.evidence_status is SourceResolution.PINNED:
            if not all((self.reference_id, self.reference_version, self.reference_commit,
                        self.source_manifest_sha256, self.environment_lock_sha256,
                        self.environment_versions)):
                raise RecipeContractError(
                    "Pinned original-reference provenance requires source ID/version/commit, "
                    "source manifest, environment lock, and package version pins."
                )
            if not _GIT_COMMIT.fullmatch(self.reference_commit):
                raise RecipeContractError("Pinned original-reference provenance requires a full Git commit hash.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "backend_id": self.backend_id,
            "backend_version": self.backend_version,
            "reference_id": self.reference_id,
            "reference_version": self.reference_version,
            "reference_commit": self.reference_commit,
            "source_manifest_sha256": self.source_manifest_sha256,
            "environment_lock_sha256": self.environment_lock_sha256,
            "environment_versions": dict(self.environment_versions),
            "license_id": self.license_id,
            "evidence_status": self.evidence_status.value,
            "evidence_note": self.evidence_note,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "RecipeProvenance":
        data = _exact_object(value, {
            "mode", "backend_id", "backend_version", "reference_id", "reference_version",
            "reference_commit", "source_manifest_sha256", "environment_lock_sha256",
            "environment_versions", "license_id", "evidence_status", "evidence_note",
        }, "recipe provenance")
        return cls(**data)


@dataclass(frozen=True, slots=True)
class SourceMapping:
    reference_id: str
    reference_version: str
    resolution: SourceResolution
    source_path: str = ""
    source_symbol: str = ""
    source_url: str = ""
    source_sha256: str | None = None
    license_id: str = ""
    note: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "reference_id", _text(self.reference_id, "source reference ID", identifier=True))
        object.__setattr__(self, "reference_version", _text(self.reference_version, "source reference version"))
        object.__setattr__(self, "resolution", SourceResolution(self.resolution))
        for field in ("source_path", "source_symbol", "source_url", "license_id", "note"):
            value = getattr(self, field)
            if not isinstance(value, str) or value != value.strip():
                raise RecipeContractError(f"Source mapping {field} must be trimmed text.")
        object.__setattr__(self, "source_sha256", _digest(self.source_sha256, "source SHA-256", optional=True))
        if self.resolution is SourceResolution.PINNED:
            if not self.source_path or not self.source_symbol or self.source_sha256 is None:
                raise RecipeContractError("Pinned source mappings require a path, symbol and SHA-256.")
        elif not self.note:
            raise RecipeContractError("Unresolved or unavailable source mappings require a note.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "reference_id": self.reference_id,
            "reference_version": self.reference_version,
            "resolution": self.resolution.value,
            "source_path": self.source_path,
            "source_symbol": self.source_symbol,
            "source_url": self.source_url,
            "source_sha256": self.source_sha256,
            "license_id": self.license_id,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "SourceMapping":
        data = _exact_object(value, {
            "reference_id", "reference_version", "resolution", "source_path", "source_symbol",
            "source_url", "source_sha256", "license_id", "note",
        }, "source mapping")
        return cls(**data)


@dataclass(frozen=True, slots=True)
class PortContract:
    name: str
    direction: str
    data_type: str
    label: str
    required: bool = True
    external: bool = False
    semantics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _text(self.name, "port name", identifier=True))
        if not _PORT_ID.fullmatch(self.name):
            raise RecipeContractError("Port name must be a stable lowercase identifier.")
        if self.direction not in {"input", "output"}:
            raise RecipeContractError("Port direction must be input or output.")
        if not isinstance(self.data_type, str) or not _TYPE_ID.fullmatch(self.data_type):
            raise RecipeContractError("Port data type must be a stable type identifier.")
        object.__setattr__(self, "label", _text(self.label, "port label"))
        if type(self.required) is not bool or type(self.external) is not bool:
            raise RecipeContractError("Port required/external flags must be booleans.")
        if self.external and (self.direction != "input" or not self.required):
            raise RecipeContractError("Only required input ports may be declared external.")
        semantics = _freeze_json(self.semantics, name=f"port {self.name} semantics")
        if not isinstance(semantics, Mapping):
            raise RecipeContractError("Port semantics must be a JSON object.")
        object.__setattr__(self, "semantics", semantics)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "direction": self.direction, "data_type": self.data_type,
                "label": self.label, "required": self.required, "external": self.external,
                "semantics": _thaw_json(self.semantics)}

    @classmethod
    def from_dict(cls, value: Any) -> "PortContract":
        return cls(**_exact_object(value, {
            "name", "direction", "data_type", "label", "required", "external", "semantics",
        }, "port"))


@dataclass(frozen=True, slots=True)
class DependencyContract:
    source_stage: str
    source_port: str
    target_stage: str
    target_port: str
    data_type: str

    def __post_init__(self) -> None:
        for field in ("source_stage", "target_stage"):
            object.__setattr__(self, field, _text(getattr(self, field), field, identifier=True))
        for field in ("source_port", "target_port"):
            object.__setattr__(self, field, _text(getattr(self, field), field, identifier=True))
        if not _PORT_ID.fullmatch(self.source_port) or not _PORT_ID.fullmatch(self.target_port):
            raise RecipeContractError("Dependency ports must be stable lowercase identifiers.")
        if not isinstance(self.data_type, str) or not _TYPE_ID.fullmatch(self.data_type):
            raise RecipeContractError("Dependency data type must be a stable type identifier.")
        if self.source_stage == self.target_stage:
            raise RecipeContractError("A recipe stage cannot depend on itself.")

    def to_dict(self) -> dict[str, str]:
        return {"source_stage": self.source_stage, "source_port": self.source_port,
                "target_stage": self.target_stage, "target_port": self.target_port,
                "data_type": self.data_type}

    @classmethod
    def from_dict(cls, value: Any) -> "DependencyContract":
        return cls(**_exact_object(value, {
            "source_stage", "source_port", "target_stage", "target_port", "data_type",
        }, "dependency"))


@dataclass(frozen=True, slots=True)
class ParameterContract:
    parameter_id: str
    value_type: str
    value: Any
    affects_calculation: bool
    editable: bool = False
    units: str = ""
    description: str = ""
    minimum: float | None = None
    maximum: float | None = None
    choices: tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameter_id", _text(self.parameter_id, "parameter ID", identifier=True))
        if self.value_type not in {"boolean", "integer", "number", "string", "json"}:
            raise RecipeContractError("Unsupported parameter value_type.")
        parameter_value = _freeze_json(self.value, name=f"parameter {self.parameter_id}")
        valid = {
            "boolean": type(parameter_value) is bool,
            "integer": type(parameter_value) is int,
            "number": type(parameter_value) in {int, float},
            "string": isinstance(parameter_value, str),
            "json": True,
        }[self.value_type]
        if not valid:
            raise RecipeContractError(f"Parameter {self.parameter_id!r} does not match {self.value_type}.")
        if self.value_type == "number" and isinstance(parameter_value, float) and not math.isfinite(parameter_value):
            raise RecipeContractError("Numeric parameter values must be finite.")
        object.__setattr__(self, "value", parameter_value)
        if type(self.affects_calculation) is not bool or type(self.editable) is not bool:
            raise RecipeContractError("Parameter calculation/editable flags must be booleans.")
        if self.editable and not self.affects_calculation:
            raise RecipeContractError("A display-only parameter cannot be editable.")
        for field in ("units", "description"):
            metadata_value = getattr(self, field)
            if not isinstance(metadata_value, str) or metadata_value != metadata_value.strip():
                raise RecipeContractError(f"Parameter {field} must be trimmed text.")
        for field in ("minimum", "maximum"):
            value = getattr(self, field)
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)):
                raise RecipeContractError(f"Parameter {field} must be a finite number or null.")
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise RecipeContractError("Parameter minimum cannot exceed maximum.")
        choices = tuple(_freeze_json(item, name=f"parameter {self.parameter_id} choices") for item in self.choices)
        if choices and parameter_value not in choices:
            raise RecipeContractError(f"Parameter {self.parameter_id!r} value is outside its choices.")
        if len(choices) != len(set(_canonical_bytes(item, name="parameter choice") for item in choices)):
            raise RecipeContractError("Parameter choices must be unique.")
        object.__setattr__(self, "choices", choices)
        if self.value_type in {"integer", "number"}:
            if self.minimum is not None and parameter_value < self.minimum:
                raise RecipeContractError("Parameter value is below its minimum.")
            if self.maximum is not None and parameter_value > self.maximum:
                raise RecipeContractError("Parameter value is above its maximum.")
        elif self.minimum is not None or self.maximum is not None:
            raise RecipeContractError("Only numeric parameters may have minimum/maximum limits.")

    def to_dict(self) -> dict[str, Any]:
        return {"parameter_id": self.parameter_id, "value_type": self.value_type,
                "value": _thaw_json(self.value), "affects_calculation": self.affects_calculation,
                "editable": self.editable, "units": self.units, "description": self.description,
                "minimum": self.minimum, "maximum": self.maximum,
                "choices": [_thaw_json(item) for item in self.choices]}

    @classmethod
    def from_dict(cls, value: Any) -> "ParameterContract":
        data = _exact_object(value, {
            "parameter_id", "value_type", "value", "affects_calculation", "editable", "units",
            "description", "minimum", "maximum", "choices",
        }, "parameter")
        if not isinstance(data["choices"], list):
            raise RecipeContractError("Parameter choices must be a JSON array.")
        data["choices"] = tuple(data["choices"])
        return cls(**data)


@dataclass(frozen=True, slots=True)
class BranchCondition:
    expression: str
    label: str
    outcome_when_true: str = "execute"

    def __post_init__(self) -> None:
        object.__setattr__(self, "expression", _text(self.expression, "branch condition"))
        object.__setattr__(self, "label", _text(self.label, "branch label"))
        if self.outcome_when_true not in {"execute", "bypass"}:
            raise RecipeContractError("Conditional outcome must be execute or bypass.")

    def to_dict(self) -> dict[str, str]:
        return {"expression": self.expression, "label": self.label, "outcome_when_true": self.outcome_when_true}

    @classmethod
    def from_dict(cls, value: Any) -> "BranchCondition":
        return cls(**_exact_object(value, {"expression", "label", "outcome_when_true"}, "branch condition"))


@dataclass(frozen=True, slots=True)
class RecipeStage:
    stage_id: str
    title: str
    operation_id: str
    description: str
    ports: tuple[PortContract, ...]
    parameters: tuple[ParameterContract, ...] = ()
    source_mappings: tuple[SourceMapping, ...] = ()
    resolution: SourceResolution = SourceResolution.PINNED
    resolution_note: str = ""
    condition: BranchCondition | None = None

    def __post_init__(self) -> None:
        for field in ("stage_id", "operation_id"):
            object.__setattr__(self, field, _text(getattr(self, field), field, identifier=True))
        for field in ("title", "description"):
            object.__setattr__(self, field, _text(getattr(self, field), field))
        ports = tuple(self.ports)
        if any(not isinstance(item, PortContract) for item in ports):
            raise RecipeContractError("Stage ports must be PortContract values.")
        if len({(item.direction, item.name) for item in ports}) != len(ports):
            raise RecipeContractError(f"Stage {self.stage_id!r} repeats a port.")
        if not any(item.direction == "output" for item in ports):
            raise RecipeContractError(f"Stage {self.stage_id!r} must declare an output port.")
        object.__setattr__(self, "ports", ports)
        parameters = tuple(self.parameters)
        if any(not isinstance(item, ParameterContract) for item in parameters):
            raise RecipeContractError("Stage parameters must be ParameterContract values.")
        if len({item.parameter_id for item in parameters}) != len(parameters):
            raise RecipeContractError(f"Stage {self.stage_id!r} repeats a parameter.")
        object.__setattr__(self, "parameters", parameters)
        sources = tuple(self.source_mappings)
        if any(not isinstance(item, SourceMapping) for item in sources):
            raise RecipeContractError("Stage source_mappings must contain SourceMapping values.")
        if len({(item.reference_id, item.source_path, item.source_symbol) for item in sources}) != len(sources):
            raise RecipeContractError(f"Stage {self.stage_id!r} repeats a source mapping.")
        object.__setattr__(self, "source_mappings", sources)
        object.__setattr__(self, "resolution", SourceResolution(self.resolution))
        if not isinstance(self.resolution_note, str) or self.resolution_note != self.resolution_note.strip():
            raise RecipeContractError("Stage resolution note must be trimmed text.")
        if self.resolution is SourceResolution.PINNED:
            if not sources or any(item.resolution is not SourceResolution.PINNED for item in sources):
                raise RecipeContractError("Pinned stages require pinned source mappings.")
            if self.resolution_note:
                raise RecipeContractError("Pinned stages cannot carry a resolution warning.")
        elif not self.resolution_note:
            raise RecipeContractError("Unresolved/unavailable stages must explain their source state.")
        if self.condition is not None and not isinstance(self.condition, BranchCondition):
            raise RecipeContractError("Stage condition must be a BranchCondition or null.")

    def to_dict(self) -> dict[str, Any]:
        return {"stage_id": self.stage_id, "title": self.title, "operation_id": self.operation_id,
                "description": self.description, "ports": [item.to_dict() for item in self.ports],
                "parameters": [item.to_dict() for item in self.parameters],
                "source_mappings": [item.to_dict() for item in self.source_mappings],
                "resolution": self.resolution.value, "resolution_note": self.resolution_note,
                "condition": None if self.condition is None else self.condition.to_dict()}

    @classmethod
    def from_dict(cls, value: Any) -> "RecipeStage":
        data = _exact_object(value, {
            "stage_id", "title", "operation_id", "description", "ports", "parameters",
            "source_mappings", "resolution", "resolution_note", "condition",
        }, "recipe stage")
        for key, decoder in (("ports", PortContract.from_dict), ("parameters", ParameterContract.from_dict),
                             ("source_mappings", SourceMapping.from_dict)):
            if not isinstance(data[key], list):
                raise RecipeContractError(f"Stage {key} must be a JSON array.")
            data[key] = tuple(decoder(item) for item in data[key])
        if data["condition"] is not None:
            data["condition"] = BranchCondition.from_dict(data["condition"])
        return cls(**data)


@dataclass(frozen=True, slots=True)
class LayoutPosition:
    stage_id: str
    x: float
    y: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "stage_id", _text(self.stage_id, "layout stage ID", identifier=True))
        for field in ("x", "y"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise RecipeContractError(f"Layout {field} must be a finite number.")
            object.__setattr__(self, field, float(value))

    def to_dict(self) -> dict[str, Any]:
        return {"stage_id": self.stage_id, "x": self.x, "y": self.y}

    @classmethod
    def from_dict(cls, value: Any) -> "LayoutPosition":
        return cls(**_exact_object(value, {"stage_id", "x", "y"}, "layout position"))


@dataclass(frozen=True, slots=True)
class RecipeIdentity:
    recipe_id: str
    recipe_version: str
    mode: RecipeProvenanceMode
    backend_id: str
    backend_version: str
    recipe_sha256: str
    graph_sha256: str
    settings_sha256: str
    layout_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "recipe_id", _text(self.recipe_id, "recipe ID", identifier=True))
        object.__setattr__(self, "recipe_version", _text(self.recipe_version, "recipe version"))
        object.__setattr__(self, "mode", RecipeProvenanceMode(self.mode))
        object.__setattr__(self, "backend_id", _text(self.backend_id, "backend ID", identifier=True))
        object.__setattr__(self, "backend_version", _text(self.backend_version, "backend version"))
        for field in ("recipe_sha256", "graph_sha256", "settings_sha256", "layout_sha256"):
            object.__setattr__(self, field, _digest(getattr(self, field), field))

    def analytical_dict(self) -> dict[str, str]:
        """Identity components that bind analysis; layout is intentionally absent."""
        return {"recipe_id": self.recipe_id, "recipe_version": self.recipe_version,
                "mode": self.mode.value, "backend_id": self.backend_id,
                "backend_version": self.backend_version, "recipe_sha256": self.recipe_sha256,
                "graph_sha256": self.graph_sha256, "settings_sha256": self.settings_sha256}

    def to_dict(self) -> dict[str, str]:
        return self.analytical_dict() | {"layout_sha256": self.layout_sha256}

    @classmethod
    def from_dict(cls, value: Any) -> "RecipeIdentity":
        data = _exact_object(value, {
            "recipe_id", "recipe_version", "mode", "backend_id", "backend_version",
            "recipe_sha256", "graph_sha256", "settings_sha256", "layout_sha256",
        }, "recipe identity")
        return cls(**data)


@dataclass(frozen=True, slots=True)
class RecipeSpec:
    recipe_id: str
    recipe_version: str
    title: str
    description: str
    provenance: RecipeProvenance
    stages: tuple[RecipeStage, ...]
    dependencies: tuple[DependencyContract, ...] = ()
    layout: tuple[LayoutPosition, ...] = ()
    schema_version: int = RECIPE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "recipe_id", _text(self.recipe_id, "recipe ID", identifier=True))
        object.__setattr__(self, "recipe_version", _text(self.recipe_version, "recipe version"))
        object.__setattr__(self, "title", _text(self.title, "recipe title"))
        object.__setattr__(self, "description", _text(self.description, "recipe description"))
        if type(self.schema_version) is not int or self.schema_version != RECIPE_SCHEMA_VERSION:
            raise RecipeContractError(f"Unsupported recipe schema version {self.schema_version!r}.")
        if not isinstance(self.provenance, RecipeProvenance):
            raise RecipeContractError("Recipe provenance must be a RecipeProvenance.")
        stages = tuple(self.stages)
        if not stages or any(not isinstance(item, RecipeStage) for item in stages):
            raise RecipeContractError("A recipe requires one or more RecipeStage values.")
        if len({item.stage_id for item in stages}) != len(stages):
            raise RecipeContractError("Recipe stage IDs must be unique.")
        object.__setattr__(self, "stages", stages)
        dependencies = tuple(self.dependencies)
        if any(not isinstance(item, DependencyContract) for item in dependencies):
            raise RecipeContractError("Recipe dependencies must be DependencyContract values.")
        if len(set(dependencies)) != len(dependencies):
            raise RecipeContractError("Recipe dependencies must be unique.")
        object.__setattr__(self, "dependencies", dependencies)
        layout = tuple(self.layout)
        if any(not isinstance(item, LayoutPosition) for item in layout):
            raise RecipeContractError("Recipe layout must contain LayoutPosition values.")
        if len({item.stage_id for item in layout}) != len(layout):
            raise RecipeContractError("Recipe layout cannot repeat a stage.")
        unknown_layout = {item.stage_id for item in layout} - {item.stage_id for item in stages}
        if unknown_layout:
            raise RecipeContractError(f"Recipe layout has unknown stage IDs: {sorted(unknown_layout)}.")
        object.__setattr__(self, "layout", layout)
        self._validate_topology()
        _canonical_bytes(self.to_dict(), name="recipe document")

    def _validate_topology(self) -> None:
        stages = {stage.stage_id: stage for stage in self.stages}
        ports = {
            stage.stage_id: {
                (port.direction, port.name): port for port in stage.ports
            }
            for stage in self.stages
        }
        used_inputs: set[tuple[str, str]] = set()
        incoming: set[tuple[str, str]] = set()
        outgoing: dict[str, list[str]] = {stage_id: [] for stage_id in stages}
        indegree = {stage_id: 0 for stage_id in stages}
        for edge in self.dependencies:
            if edge.source_stage not in stages or edge.target_stage not in stages:
                raise RecipeContractError("Dependency references a missing recipe stage.")
            source = ports[edge.source_stage].get(("output", edge.source_port))
            target = ports[edge.target_stage].get(("input", edge.target_port))
            if source is None or target is None:
                raise RecipeContractError("Dependency references a missing or misdirected port.")
            if source.data_type != edge.data_type or target.data_type != edge.data_type:
                raise RecipeContractError("Dependency data type does not match both endpoint ports.")
            if source.semantics != target.semantics:
                raise RecipeContractError("Dependency endpoint semantics differ; declare an explicit conversion stage.")
            target_key = (edge.target_stage, edge.target_port)
            if target_key in used_inputs:
                raise RecipeContractError("PipelineGraph inputs support one producer per port.")
            used_inputs.add(target_key)
            incoming.add(target_key)
            outgoing[edge.source_stage].append(edge.target_stage)
            indegree[edge.target_stage] += 1
        for stage in self.stages:
            for port in stage.ports:
                if port.direction == "input" and port.required and not port.external:
                    if (stage.stage_id, port.name) not in incoming:
                        raise RecipeContractError(f"Required input {stage.stage_id}.{port.name} has no producer.")
        ready = [stage.stage_id for stage in self.stages if indegree[stage.stage_id] == 0]
        visited = 0
        while ready:
            current = ready.pop(0)
            visited += 1
            for target in outgoing[current]:
                indegree[target] -= 1
                if indegree[target] == 0:
                    ready.append(target)
        if visited != len(stages):
            raise RecipeContractError("Recipe dependencies must form a directed acyclic graph.")

    def to_dict(self) -> dict[str, Any]:
        return {"schema": RECIPE_SCHEMA, "schema_version": self.schema_version,
                "recipe_id": self.recipe_id, "recipe_version": self.recipe_version,
                "title": self.title, "description": self.description,
                "provenance": self.provenance.to_dict(),
                "stages": [item.to_dict() for item in self.stages],
                "dependencies": [item.to_dict() for item in self.dependencies],
                "layout": [item.to_dict() for item in self.layout]}

    def to_json(self) -> str:
        return _canonical_bytes(self.to_dict(), name="recipe document").decode("utf-8")

    @classmethod
    def from_json(cls, value: str | bytes) -> "RecipeSpec":
        if isinstance(value, bytes):
            if len(value) > MAX_RECIPE_BYTES:
                raise RecipeContractError("Recipe JSON exceeds the size limit.")
            try:
                value = value.decode("utf-8")
            except UnicodeDecodeError as error:
                raise RecipeContractError("Recipe JSON must be UTF-8.") from error
        if not isinstance(value, str):
            raise RecipeContractError("Recipe JSON must be text or UTF-8 bytes.")
        try:
            encoded_size = len(value.encode("utf-8"))
        except UnicodeEncodeError as error:
            raise RecipeContractError("Recipe JSON must be valid UTF-8 text.") from error
        if encoded_size > MAX_RECIPE_BYTES:
            raise RecipeContractError("Recipe JSON exceeds the size limit.")
        try:
            data = json.loads(value, object_pairs_hook=_reject_duplicate_pairs, parse_constant=_reject_constant)
        except (json.JSONDecodeError, UnicodeError, RecursionError) as error:
            raise RecipeContractError("Recipe JSON is malformed.") from error
        data = _exact_object(data, {
            "schema", "schema_version", "recipe_id", "recipe_version", "title", "description",
            "provenance", "stages", "dependencies", "layout",
        }, "recipe")
        if data["schema"] != RECIPE_SCHEMA:
            raise RecipeContractError("Unknown recipe schema.")
        if not isinstance(data["stages"], list) or not isinstance(data["dependencies"], list) or not isinstance(data["layout"], list):
            raise RecipeContractError("Recipe stages, dependencies and layout must be JSON arrays.")
        return cls(
            recipe_id=data["recipe_id"], recipe_version=data["recipe_version"],
            title=data["title"], description=data["description"],
            provenance=RecipeProvenance.from_dict(data["provenance"]),
            stages=tuple(RecipeStage.from_dict(item) for item in data["stages"]),
            dependencies=tuple(DependencyContract.from_dict(item) for item in data["dependencies"]),
            layout=tuple(LayoutPosition.from_dict(item) for item in data["layout"]),
            schema_version=data["schema_version"],
        )

    @property
    def identity(self) -> RecipeIdentity:
        # Stage/recipe layout and presentation text are omitted from analytic
        # digests. Values for explicitly display-only parameters are omitted too.
        core = self.to_dict()
        core.pop("layout")
        core.pop("title")
        core.pop("description")
        for stage in core["stages"]:
            stage.pop("title")
            stage.pop("description")
            stage.pop("resolution_note")
            stage["parameters"] = [
                {key: value for key, value in parameter.items() if key != "value"}
                | ({"value": parameter["value"]} if parameter["affects_calculation"] else {})
                for parameter in stage["parameters"]
            ]
            for port in stage["ports"]:
                port.pop("label")
            for parameter in stage["parameters"]:
                parameter.pop("description")
        core["provenance"].pop("evidence_note")
        for stage in core["stages"]:
            for source in stage["source_mappings"]:
                source.pop("note")
            if stage["condition"] is not None:
                stage["condition"].pop("label")
        graph_payload = {
            "recipe_id": self.recipe_id, "recipe_version": self.recipe_version,
            "mode": self.provenance.mode.value, "backend_id": self.provenance.backend_id,
            "backend_version": self.provenance.backend_version,
            "reference_id": self.provenance.reference_id,
            "reference_version": self.provenance.reference_version,
            "reference_commit": self.provenance.reference_commit,
            "source_manifest_sha256": self.provenance.source_manifest_sha256,
            "environment_lock_sha256": self.provenance.environment_lock_sha256,
            "environment_versions": dict(self.provenance.environment_versions),
            "stages": [{"stage_id": stage.stage_id, "operation_id": stage.operation_id,
                        "ports": [{key: value for key, value in port.to_dict().items() if key != "label"}
                                  for port in stage.ports],
                        "source_mappings": [{key: value for key, value in source.to_dict().items() if key != "note"}
                                            for source in stage.source_mappings],
                        "resolution": stage.resolution.value,
                        "condition": None if stage.condition is None else {
                            "expression": stage.condition.expression,
                            "outcome_when_true": stage.condition.outcome_when_true,
                        }}
                       for stage in self.stages],
            "dependencies": [edge.to_dict() for edge in self.dependencies],
        }
        settings_payload = {
            stage.stage_id: {parameter.parameter_id: _thaw_json(parameter.value)
                             for parameter in stage.parameters if parameter.affects_calculation}
            for stage in self.stages
        }
        return RecipeIdentity(
            recipe_id=self.recipe_id, recipe_version=self.recipe_version,
            mode=self.provenance.mode, backend_id=self.provenance.backend_id,
            backend_version=self.provenance.backend_version,
            recipe_sha256=_sha(core), graph_sha256=_sha(graph_payload),
            settings_sha256=_sha(settings_payload), layout_sha256=_sha([item.to_dict() for item in self.layout]),
        )

    def parameters_for_runner(self, stage_id: str) -> dict[str, Any]:
        """Return the exact calculation-affecting values expected by a runner."""
        stage = self.stage(stage_id)
        return {item.parameter_id: _thaw_json(item.value) for item in stage.parameters if item.affects_calculation}

    def stage(self, stage_id: str) -> RecipeStage:
        for stage in self.stages:
            if stage.stage_id == stage_id:
                return stage
        raise KeyError(stage_id)


@dataclass(frozen=True, slots=True)
class StageRunRecord:
    stage_id: str
    recipe_identity: RecipeIdentity
    state: StageRunState
    checkpoint_sha256: str | None = None
    upstream_checkpoint_sha256: str | None = None
    reason: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "stage_id", _text(self.stage_id, "run stage ID", identifier=True))
        if not isinstance(self.recipe_identity, RecipeIdentity):
            raise RecipeContractError("Stage run requires a RecipeIdentity.")
        object.__setattr__(self, "state", StageRunState(self.state))
        object.__setattr__(self, "checkpoint_sha256", _digest(self.checkpoint_sha256, "checkpoint SHA-256", optional=True))
        object.__setattr__(self, "upstream_checkpoint_sha256", _digest(
            self.upstream_checkpoint_sha256, "upstream checkpoint SHA-256", optional=True
        ))
        if not isinstance(self.reason, str) or self.reason != self.reason.strip():
            raise RecipeContractError("Stage run reason must be trimmed text.")
        if self.state in {StageRunState.EXECUTED, StageRunState.EMPTY}:
            if self.checkpoint_sha256 is None or self.reason:
                raise RecipeContractError("Executed/empty stages require a checkpoint hash and no failure reason.")
        elif self.checkpoint_sha256 is not None:
            raise RecipeContractError("Only executed or empty stages may publish a checkpoint hash.")
        if self.state in {StageRunState.UNRESOLVED, StageRunState.UNAVAILABLE, StageRunState.FAILED, StageRunState.UNSUPPORTED, StageRunState.BYPASSED} and not self.reason:
            raise RecipeContractError(f"{self.state.value} stage state requires a reason.")

    def to_dict(self) -> dict[str, Any]:
        return {"stage_id": self.stage_id, "recipe_identity": self.recipe_identity.to_dict(),
                "state": self.state.value, "checkpoint_sha256": self.checkpoint_sha256,
                "upstream_checkpoint_sha256": self.upstream_checkpoint_sha256, "reason": self.reason}

    @classmethod
    def from_dict(cls, value: Any) -> "StageRunRecord":
        data = _exact_object(value, {
            "stage_id", "recipe_identity", "state", "checkpoint_sha256",
            "upstream_checkpoint_sha256", "reason",
        }, "stage run record")
        data["recipe_identity"] = RecipeIdentity.from_dict(data["recipe_identity"])
        return cls(**data)


def initial_stage_run(recipe: RecipeSpec, stage_id: str) -> StageRunRecord:
    stage = recipe.stage(stage_id)
    state = {
        SourceResolution.PINNED: StageRunState.NOT_RUN,
        SourceResolution.UNRESOLVED: StageRunState.UNRESOLVED,
        SourceResolution.UNAVAILABLE: StageRunState.UNAVAILABLE,
    }[stage.resolution]
    reason = stage.resolution_note if state in {StageRunState.UNRESOLVED, StageRunState.UNAVAILABLE} else ""
    return StageRunRecord(stage_id, recipe.identity, state, reason=reason)


def _upstream_expectations(
    expected: Mapping[str, str | None] | None,
    stage_ids: set[str],
) -> dict[str, str | None]:
    """Validate caller-supplied upstream bindings; absent keys mean unspecified."""
    if expected is None:
        return {}
    if not isinstance(expected, Mapping):
        raise RecipeContractError("Expected upstream checkpoint bindings must be an object.")
    result: dict[str, str | None] = {}
    for stage_id, digest in expected.items():
        if not isinstance(stage_id, str) or stage_id not in stage_ids:
            raise RecipeContractError(f"Expected upstream binding names unknown stage {stage_id!r}.")
        result[stage_id] = _digest(
            digest, f"expected upstream checkpoint SHA-256 for {stage_id}", optional=True,
        )
    return result


def validate_stage_run(
    recipe: RecipeSpec,
    record: StageRunRecord,
    *,
    expected_upstream_checkpoint_sha256: str | None | object = _UNSPECIFIED_UPSTREAM,
) -> None:
    stage = recipe.stage(record.stage_id)
    current = recipe.identity
    if record.recipe_identity.analytical_dict() != current.analytical_dict():
        raise StaleRecipeIdentity("Stage run belongs to a stale recipe, graph, settings, or backend identity.")
    if stage.resolution in {SourceResolution.UNRESOLVED, SourceResolution.UNAVAILABLE}:
        if record.state in {StageRunState.EXECUTED, StageRunState.EMPTY, StageRunState.BYPASSED}:
            raise RecipeContractError(
                f"A source-{stage.resolution.value} stage cannot claim completed or bypassed state "
                f"{record.state.value}."
            )
    if record.state is StageRunState.UNRESOLVED and stage.resolution is not SourceResolution.UNRESOLVED:
        raise RecipeContractError("Only a source-unresolved stage can have unresolved run state.")
    if record.state is StageRunState.BYPASSED and stage.condition is None:
        raise RecipeContractError("Only a conditional stage can be recorded as bypassed.")
    if expected_upstream_checkpoint_sha256 is not _UNSPECIFIED_UPSTREAM:
        expected = _digest(
            expected_upstream_checkpoint_sha256,
            "expected upstream checkpoint SHA-256",
            optional=True,
        )
        if record.upstream_checkpoint_sha256 != expected:
            raise StaleRecipeIdentity("Stage run was produced from a different upstream checkpoint.")


def validate_checkpoint_identity(
    recipe: RecipeSpec,
    stage_id: str,
    checkpoint: CheckpointIdentity,
    *,
    expected_identity: CheckpointIdentity,
    expected_upstream_checkpoint_sha256: str | None | object = _UNSPECIFIED_UPSTREAM,
) -> None:
    """Reject checkpoint reuse when any analytical recipe identity is stale."""
    recipe.stage(stage_id)
    if not isinstance(checkpoint, CheckpointIdentity):
        raise TypeError("checkpoint must be a CheckpointIdentity.")
    if not isinstance(expected_identity, CheckpointIdentity):
        raise TypeError("expected_identity must be a CheckpointIdentity.")
    if checkpoint != expected_identity:
        raise StaleRecipeIdentity("Checkpoint source, frame, semantics, environment, or upstream identity is stale.")
    current = recipe.identity
    if (
        checkpoint.stage_id != stage_id
        or checkpoint.recipe_id != current.recipe_id
        or checkpoint.recipe_version != current.recipe_version
        or checkpoint.recipe_sha256 != current.recipe_sha256
        or checkpoint.graph_sha256 != current.graph_sha256
        or checkpoint.settings_sha256 != current.settings_sha256
        or checkpoint.backend_id != current.backend_id
        or checkpoint.backend_version != current.backend_version
    ):
        raise StaleRecipeIdentity("Checkpoint recipe, stage, graph, settings, or backend identity is stale.")
    declared_environment = dict(recipe.provenance.environment_versions)
    if declared_environment and dict(checkpoint.environment_versions) != declared_environment:
        raise StaleRecipeIdentity("Checkpoint environment versions differ from the pinned recipe environment.")
    if expected_upstream_checkpoint_sha256 is not _UNSPECIFIED_UPSTREAM:
        expected = _digest(
            expected_upstream_checkpoint_sha256,
            "expected upstream checkpoint SHA-256",
            optional=True,
        )
        if checkpoint.upstream_checkpoint_sha256 != expected:
            raise StaleRecipeIdentity("Checkpoint belongs to a different upstream stage result.")


def _stage_details(recipe: RecipeSpec, stage: RecipeStage) -> str:
    payload = {
        "recipe_id": recipe.recipe_id,
        "recipe_version": recipe.recipe_version,
        "provenance": recipe.provenance.to_dict(),
        "stage_id": stage.stage_id,
        "operation_id": stage.operation_id,
        "resolution": stage.resolution.value,
        "resolution_note": stage.resolution_note,
        "condition": None if stage.condition is None else stage.condition.to_dict(),
        "ports": [port.to_dict() for port in stage.ports],
        "source_mappings": [source.to_dict() for source in stage.source_mappings],
        "parameters": [parameter.to_dict() for parameter in stage.parameters],
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)


def _pipeline_status(state: StageRunState) -> tuple[NodeStatus, str]:
    if state is StageRunState.UNRESOLVED:
        return NodeStatus.PLANNED, "Source operation is unresolved"
    if state is StageRunState.UNAVAILABLE:
        return NodeStatus.BLOCKED, "Reference source or execution is unavailable"
    if state is StageRunState.NOT_RUN:
        return NodeStatus.IDLE, "Not run"
    if state in {StageRunState.EXECUTED, StageRunState.EMPTY}:
        return NodeStatus.COMPLETE, state.value
    if state is StageRunState.BYPASSED:
        return NodeStatus.BYPASSED, "Condition evaluated: bypassed"
    if state is StageRunState.FAILED:
        return NodeStatus.FAILED, "Execution failed"
    return NodeStatus.BLOCKED, state.value


def to_pipeline_graph(
    recipe: RecipeSpec,
    run_records: Iterable[StageRunRecord] = (),
    *,
    expected_upstream_checkpoint_sha256s: Mapping[str, str | None] | None = None,
) -> PipelineGraph:
    """Project a recipe into existing native graph nodes and typed connections."""
    if not isinstance(recipe, RecipeSpec):
        raise TypeError("recipe must be a RecipeSpec.")
    records: dict[str, StageRunRecord] = {}
    stage_ids = {stage.stage_id for stage in recipe.stages}
    upstream_expectations = _upstream_expectations(expected_upstream_checkpoint_sha256s, stage_ids)
    for record in run_records:
        if not isinstance(record, StageRunRecord):
            raise RecipeContractError("run_records must contain StageRunRecord values.")
        if record.stage_id not in stage_ids:
            raise RecipeContractError(f"Run record refers to unknown stage {record.stage_id!r}.")
        if record.stage_id in records:
            raise RecipeContractError(f"Duplicate stage run record for {record.stage_id!r}.")
        validate_stage_run(
            recipe,
            record,
            expected_upstream_checkpoint_sha256=upstream_expectations.get(
                record.stage_id, _UNSPECIFIED_UPSTREAM,
            ),
        )
        records[record.stage_id] = record
    unexpected = set(records) - {stage.stage_id for stage in recipe.stages}
    if unexpected:
        raise RecipeContractError(f"Run records refer to extra stages: {sorted(unexpected)}.")
    layout = {item.stage_id: item for item in recipe.layout}
    order = {stage_id: index for index, stage_id in enumerate(_topological_ids(recipe))}
    nodes: list[PipelineNode] = []
    for stage in recipe.stages:
        position = layout.get(stage.stage_id)
        x, y = (position.x, position.y) if position is not None else ((order[stage.stage_id] % 4) * 260.0, (order[stage.stage_id] // 4) * 170.0)
        record = records.get(stage.stage_id) or initial_stage_run(recipe, stage.stage_id)
        status, detail = _pipeline_status(record.state)
        if record.reason:
            detail = f"{detail}: {record.reason}"
        nodes.append(PipelineNode(
            identifier=stage.stage_id,
            title=stage.title,
            category="reference",
            description=stage.description,
            details=_stage_details(recipe, stage),
            x=x, y=y, enabled=True, implemented=False,
            bypassable=stage.condition is not None,
            status=status, status_detail=detail,
            input_ports=tuple((port.name, port.label) for port in stage.ports if port.direction == "input"),
            output_ports=tuple((port.name, port.label) for port in stage.ports if port.direction == "output"),
        ))
    connections = tuple(PipelineConnection(
        source=edge.source_stage, target=edge.target_stage, data_type=edge.data_type,
        source_port=edge.source_port, target_port=edge.target_port,
    ) for edge in recipe.dependencies)
    graph = PipelineGraph(nodes, connections)
    # PipelineGraph derives concrete sockets from its declared connections.
    # Restore every authored contract port afterward so unconnected optional
    # and external inputs remain visible with the recipe's exact labels/types.
    for stage in recipe.stages:
        node = graph.nodes[stage.stage_id]
        node.input_ports = tuple((port.name, port.label) for port in stage.ports if port.direction == "input")
        node.output_ports = tuple((port.name, port.label) for port in stage.ports if port.direction == "output")
        node.input_port_types = {port.name: port.data_type for port in stage.ports if port.direction == "input"}
        node.output_port_types = {port.name: port.data_type for port in stage.ports if port.direction == "output"}
    # PipelineGraph remains the sole topology/rendering model. These immutable
    # sidecar identities make the projection auditable without changing native
    # graph persistence or cache behavior in this infrastructure-only chunk.
    graph.reference_recipe_identity = recipe.identity
    graph.reference_recipe_mode = recipe.provenance.mode.value
    graph.reference_recipe_sources = MappingProxyType({
        stage.stage_id: tuple(_freeze_json(source.to_dict(), name="source mapping") for source in stage.source_mappings)
        for stage in recipe.stages
    })
    graph.reference_recipe_runs = MappingProxyType({
        stage.stage_id: records.get(stage.stage_id) or initial_stage_run(recipe, stage.stage_id)
        for stage in recipe.stages
    })
    validate_pipeline_graph(
        recipe, graph,
        expected_upstream_checkpoint_sha256s=expected_upstream_checkpoint_sha256s,
    )
    return graph


def _topological_ids(recipe: RecipeSpec) -> tuple[str, ...]:
    indegree = {stage.stage_id: 0 for stage in recipe.stages}
    outgoing = {stage.stage_id: [] for stage in recipe.stages}
    for edge in recipe.dependencies:
        indegree[edge.target_stage] += 1
        outgoing[edge.source_stage].append(edge.target_stage)
    ready = [stage.stage_id for stage in recipe.stages if indegree[stage.stage_id] == 0]
    result: list[str] = []
    while ready:
        stage_id = ready.pop(0)
        result.append(stage_id)
        for target in outgoing[stage_id]:
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
    if len(result) != len(indegree):
        raise RecipeContractError("Recipe dependencies must form a directed acyclic graph.")
    return tuple(result)


def validate_pipeline_graph(
    recipe: RecipeSpec,
    graph: PipelineGraph,
    *,
    expected_upstream_checkpoint_sha256s: Mapping[str, str | None] | None = None,
) -> None:
    """Require exact declared nodes, source maps, typed ports and dependencies."""
    if not isinstance(graph, PipelineGraph):
        raise TypeError("graph must be the existing PipelineGraph model.")
    expected_ids = {stage.stage_id for stage in recipe.stages}
    upstream_expectations = _upstream_expectations(expected_upstream_checkpoint_sha256s, expected_ids)
    if graph.unused_nodes:
        unused_ids = sorted(str(identifier) for identifier in graph.unused_nodes)
        raise RecipeGraphMismatch(
            f"Reference projection cannot contain unused nodes: {unused_ids}."
        )
    actual_ids = set(graph.nodes)
    missing, extra = expected_ids - actual_ids, actual_ids - expected_ids
    if missing or extra:
        raise RecipeGraphMismatch(f"Recipe graph stages differ; missing={sorted(missing)}, extra={sorted(extra)}.")
    for stage in recipe.stages:
        node = graph.nodes[stage.stage_id]
        if (node.identifier != stage.stage_id or node.title != stage.title
                or node.description != stage.description or node.category != "reference"):
            raise RecipeGraphMismatch(f"Recipe stage metadata differs for {stage.stage_id!r}.")
        if node.implemented:
            raise RecipeGraphMismatch(f"Reference projection stage {stage.stage_id!r} must not claim native execution.")
        if not node.enabled or node.parameters or node.parameter_specs or node.parameter_sections:
            raise RecipeGraphMismatch(
                f"Reference projection stage {stage.stage_id!r} must remain read-only until execution is wired."
            )
        expected_inputs = tuple((port.name, port.label) for port in stage.ports if port.direction == "input")
        expected_outputs = tuple((port.name, port.label) for port in stage.ports if port.direction == "output")
        if node.input_ports != expected_inputs or node.output_ports != expected_outputs:
            raise RecipeGraphMismatch(f"Missing, extra, or mislabelled typed ports on {stage.stage_id!r}.")
        input_types = {port.name: port.data_type for port in stage.ports if port.direction == "input"}
        output_types = {port.name: port.data_type for port in stage.ports if port.direction == "output"}
        if node.input_port_types != input_types or node.output_port_types != output_types:
            raise RecipeGraphMismatch(f"Miswired or mistyped ports on {stage.stage_id!r}.")
        if node.details != _stage_details(recipe, stage):
            raise RecipeGraphMismatch(f"Source mapping, provenance, condition, or parameter description differs on {stage.stage_id!r}.")
        if node.bypassable != (stage.condition is not None):
            raise RecipeGraphMismatch(f"Conditional bypass declaration differs on {stage.stage_id!r}.")
    expected_edges = {
        (edge.source_stage, edge.source_port, edge.target_stage, edge.target_port, edge.data_type)
        for edge in recipe.dependencies
    }
    actual_edge_items = [
        (edge.source, edge.source_port, edge.target, edge.target_port, edge.data_type)
        for edge in graph.connections
    ]
    actual_edges = set(actual_edge_items)
    if len(actual_edge_items) != len(actual_edges):
        raise RecipeGraphMismatch("Recipe graph contains duplicate dependencies.")
    if actual_edges != expected_edges:
        missing_edges = sorted(expected_edges - actual_edges)
        extra_edges = sorted(actual_edges - expected_edges)
        raise RecipeGraphMismatch(f"Recipe graph dependencies are missing or miswired; missing={missing_edges}, extra={extra_edges}.")
    template_items = [
        (edge.source, edge.source_port, edge.target, edge.target_port, edge.data_type)
        for edge in graph.connection_templates
    ]
    if set(template_items) != expected_edges or len(template_items) != len(expected_edges):
        raise RecipeGraphMismatch("Recipe graph declares extra, missing, or duplicate reconnectable dependencies.")
    if graph.unused_connections:
        raise RecipeGraphMismatch("Recipe graph contains suspended dependencies outside the declared view.")
    try:
        graph.topological_order()
    except ValueError as error:
        raise RecipeGraphMismatch("Recipe graph is not a DAG.") from error
    if getattr(graph, "reference_recipe_identity", None) is None or graph.reference_recipe_identity.analytical_dict() != recipe.identity.analytical_dict():
        raise RecipeGraphMismatch("Recipe graph is missing or has stale recipe identity metadata.")
    if getattr(graph, "reference_recipe_mode", None) != recipe.provenance.mode.value:
        raise RecipeGraphMismatch("Recipe graph backend mode is missing or stale.")
    sources = getattr(graph, "reference_recipe_sources", None)
    expected_sources = {
        stage.stage_id: tuple(_freeze_json(source.to_dict(), name="source mapping") for source in stage.source_mappings)
        for stage in recipe.stages
    }
    if sources is None or dict(sources) != expected_sources:
        raise RecipeGraphMismatch("Recipe graph is missing, extra, or stale source mappings.")
    runs = getattr(graph, "reference_recipe_runs", None)
    if runs is None or set(runs) != expected_ids:
        raise RecipeGraphMismatch("Recipe graph is missing or has extra stage run states.")
    for stage in recipe.stages:
        record = runs[stage.stage_id]
        if not isinstance(record, StageRunRecord):
            raise RecipeGraphMismatch(f"Invalid run state for {stage.stage_id!r}.")
        if record.stage_id != stage.stage_id:
            raise RecipeGraphMismatch(
                f"Run record for {stage.stage_id!r} is bound to stage {record.stage_id!r}."
            )
        try:
            validate_stage_run(
                recipe,
                record,
                expected_upstream_checkpoint_sha256=upstream_expectations.get(
                    stage.stage_id, _UNSPECIFIED_UPSTREAM,
                ),
            )
        except RecipeContractError as error:
            raise RecipeGraphMismatch(f"Stale run state for {stage.stage_id!r}: {error}") from error
        expected_status, expected_detail = _pipeline_status(record.state)
        if record.reason:
            expected_detail = f"{expected_detail}: {record.reason}"
        node = graph.nodes[stage.stage_id]
        if node.status != expected_status or node.status_detail != expected_detail:
            raise RecipeGraphMismatch(f"Stage run state differs from the declared record for {stage.stage_id!r}.")
