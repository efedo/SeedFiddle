"""Immutable, source-bound checkpoints for explicit benchmark transfers.

This package is deliberately separate from the production analysis path. A
caller must opt in to CPU serialization, and the configured limits are applied
before any GPU tensor is copied to host memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from io import BytesIO
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from types import MappingProxyType
from typing import Any, Callable, Mapping
import zipfile

import numpy as np


CHECKPOINT_FORMAT = "seedfiddle-benchmark-checkpoint"
CHECKPOINT_VERSION = 1
MAX_CHECKPOINT_BYTES = 512 * 1024 * 1024
MAX_ARRAY_BYTES = 256 * 1024 * 1024
MAX_METADATA_BYTES = 2 * 1024 * 1024
MAX_ARRAYS = 64
MAX_ARRAY_DIMENSIONS = 8
MAX_ZIP_RATIO = 200.0
_DEFAULT_TOTAL_ARRAY_BYTES = MAX_CHECKPOINT_BYTES - MAX_ARRAYS * 65536

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER = re.compile(r"[a-z][a-z0-9_.:-]{0,127}\Z")
_ARRAY_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_.-]{0,127}\Z")
_RESERVED_ARRAY_NAMES = frozenset({"file", "allow_pickle"})


class CheckpointError(ValueError):
    """Base error for an invalid, incompatible, or unavailable checkpoint."""


class InvalidCheckpoint(CheckpointError):
    """The persisted checkpoint is malformed, corrupt, or over a hard limit."""


class CheckpointMismatch(CheckpointError):
    """A valid checkpoint does not have the exact requested replay identity."""


class CheckpointCancelled(CheckpointError):
    """Checkpoint staging was cancelled before its atomic publish completed."""


class ExecutionStatus(StrEnum):
    AVAILABLE = "available"
    EMPTY = "empty"
    FAILED = "failed"
    UNSUPPORTED = "unsupported"


def _freeze_json(value: Any, *, name: str) -> Any:
    """Validate strict JSON primitives and recursively make them immutable."""

    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{name} cannot contain NaN or infinity.")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{name} object keys must be strings.")
            frozen[key] = _freeze_json(item, name=name)
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item, name=name) for item in value)
    raise ValueError(f"{name} must contain only JSON primitive values.")


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _canonical_json(value: Any, *, name: str, maximum: int | None = None) -> bytes:
    try:
        encoded = json.dumps(
            _thaw_json(value),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as error:
        raise ValueError(f"{name} is not valid canonical JSON metadata.") from error
    if maximum is not None and len(encoded) > maximum:
        raise ValueError(f"{name} exceeds the {maximum}-byte metadata limit.")
    return encoded


def _require_text(value: Any, name: str, *, token: bool = False) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    if value != value.strip():
        raise ValueError(f"{name} cannot have leading or trailing whitespace.")
    if token and not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"{name} must be a stable lowercase identifier.")
    return value


def _require_sha256(value: Any, name: str, *, optional: bool = False) -> str | None:
    if optional and value is None:
        return None
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest.")
    return value


def _require_shape(value: Any, name: str) -> tuple[int, int]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"{name} must be a height/width pair.")
    if any(type(part) is not int or part <= 0 for part in value):
        raise ValueError(f"{name} dimensions must be positive integers.")
    return int(value[0]), int(value[1])


@dataclass(frozen=True, slots=True)
class CoordinateFrame:
    """Exact raster coordinate identity; frames are never resized or inferred."""

    frame_id: str
    shape: tuple[int, int]
    origin: str
    axes: str
    units: str
    transform_sha256: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "frame_id", _require_text(self.frame_id, "frame ID", token=True))
        object.__setattr__(self, "shape", _require_shape(self.shape, "coordinate frame shape"))
        for field_name in ("origin", "axes", "units"):
            object.__setattr__(self, field_name, _require_text(getattr(self, field_name), f"frame {field_name}"))
        object.__setattr__(
            self,
            "transform_sha256",
            _require_sha256(self.transform_sha256, "coordinate transform SHA-256", optional=True),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "shape": list(self.shape),
            "origin": self.origin,
            "axes": self.axes,
            "units": self.units,
            "transform_sha256": self.transform_sha256,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> "CoordinateFrame":
        expected = {"frame_id", "shape", "origin", "axes", "units", "transform_sha256"}
        if not isinstance(payload, dict) or set(payload) != expected:
            raise ValueError("Coordinate-frame metadata has an invalid schema.")
        return cls(**payload)


_IMAGE_SEMANTICS = frozenset(
    {"color_order", "dtype", "value_range", "channel_conversion", "geometric_transform", "interpolation"}
)
_MASK_SEMANTICS = frozenset(
    {"foreground_polarity", "representation", "connectivity", "validity_region", "border_policy"}
)
_INSTANCE_SEMANTICS = frozenset(
    {"label_dtype", "background_label", "label_id_mapping", "split_boundary_ownership", "visibility", "eligibility"}
)


def _validate_semantics(value: Mapping[str, Any], required: frozenset[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object.")
    frozen = _freeze_json(value, name=name)
    if not isinstance(frozen, Mapping) or not required.issubset(frozen):
        missing = sorted(required - set(frozen)) if isinstance(frozen, Mapping) else sorted(required)
        raise ValueError(f"{name} is missing required fields: {', '.join(missing)}.")
    return frozen


@dataclass(frozen=True, slots=True)
class CheckpointIdentity:
    """All inputs whose change makes a stored raster unsafe to replay."""

    source_sha256: str
    coordinate_frame: CoordinateFrame
    stage_id: str
    recipe_id: str
    recipe_version: str
    recipe_sha256: str
    graph_sha256: str
    settings_sha256: str
    backend_id: str
    backend_version: str
    environment_versions: Mapping[str, str]
    image_semantics: Mapping[str, Any]
    mask_semantics: Mapping[str, Any]
    instance_semantics: Mapping[str, Any]
    upstream_checkpoint_sha256: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_sha256", _require_sha256(self.source_sha256, "source SHA-256"))
        if not isinstance(self.coordinate_frame, CoordinateFrame):
            raise ValueError("coordinate_frame must be an exact CoordinateFrame.")
        for field_name in ("stage_id", "recipe_id", "backend_id"):
            object.__setattr__(self, field_name, _require_text(getattr(self, field_name), field_name, token=True))
        for field_name in ("recipe_version", "backend_version"):
            object.__setattr__(self, field_name, _require_text(getattr(self, field_name), field_name))
        for field_name in ("recipe_sha256", "graph_sha256", "settings_sha256", "upstream_checkpoint_sha256"):
            object.__setattr__(
                self,
                field_name,
                _require_sha256(
                    getattr(self, field_name),
                    field_name.replace("_", " "),
                    optional=field_name == "upstream_checkpoint_sha256",
                ),
            )
        versions = _freeze_json(self.environment_versions, name="environment versions")
        if not isinstance(versions, Mapping) or not versions:
            raise ValueError("environment_versions must be a non-empty JSON object.")
        if any(not isinstance(key, str) or not key or not isinstance(value, str) or not value for key, value in versions.items()):
            raise ValueError("Environment dependency versions must map names to non-empty strings.")
        object.__setattr__(self, "environment_versions", versions)
        object.__setattr__(self, "image_semantics", _validate_semantics(self.image_semantics, _IMAGE_SEMANTICS, "image semantics"))
        object.__setattr__(self, "mask_semantics", _validate_semantics(self.mask_semantics, _MASK_SEMANTICS, "mask semantics"))
        instance = _validate_semantics(self.instance_semantics, _INSTANCE_SEMANTICS, "instance semantics")
        if type(instance["background_label"]) is not int or instance["background_label"] != 0:
            raise ValueError("Instance semantics must declare integer label 0 as background.")
        object.__setattr__(self, "instance_semantics", instance)
        _canonical_json(self.to_dict(), name="checkpoint identity", maximum=MAX_METADATA_BYTES)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_sha256": self.source_sha256,
            "coordinate_frame": self.coordinate_frame.to_dict(),
            "stage_id": self.stage_id,
            "recipe_id": self.recipe_id,
            "recipe_version": self.recipe_version,
            "recipe_sha256": self.recipe_sha256,
            "graph_sha256": self.graph_sha256,
            "settings_sha256": self.settings_sha256,
            "backend_id": self.backend_id,
            "backend_version": self.backend_version,
            "environment_versions": _thaw_json(self.environment_versions),
            "upstream_checkpoint_sha256": self.upstream_checkpoint_sha256,
            "image_semantics": _thaw_json(self.image_semantics),
            "mask_semantics": _thaw_json(self.mask_semantics),
            "instance_semantics": _thaw_json(self.instance_semantics),
        }

    @classmethod
    def from_dict(cls, payload: Any) -> "CheckpointIdentity":
        expected = {
            "source_sha256", "coordinate_frame", "stage_id", "recipe_id", "recipe_version", "recipe_sha256",
            "graph_sha256", "settings_sha256", "backend_id", "backend_version", "environment_versions",
            "upstream_checkpoint_sha256", "image_semantics", "mask_semantics", "instance_semantics",
        }
        if not isinstance(payload, dict) or set(payload) != expected:
            raise ValueError("Checkpoint identity has an invalid schema.")
        values = dict(payload)
        values["coordinate_frame"] = CoordinateFrame.from_dict(values["coordinate_frame"])
        return cls(**values)


@dataclass(frozen=True, slots=True)
class MeasurementContract:
    """Definition and eligibility provenance for measurements from a stage."""

    definition_id: str
    definition_version: str
    mask_type: str
    units: str
    calibration_status: str
    uncertainty: str
    eligible: bool
    reason: str = ""

    def __post_init__(self) -> None:
        for name in ("definition_id", "definition_version", "mask_type", "units", "calibration_status", "uncertainty"):
            object.__setattr__(self, name, _require_text(getattr(self, name), f"measurement {name}"))
        if type(self.eligible) is not bool:
            raise ValueError("measurement eligible must be a boolean.")
        if not isinstance(self.reason, str) or (self.reason and self.reason != self.reason.strip()):
            raise ValueError("measurement reason must be trimmed text.")
        if self.eligible and self.reason:
            raise ValueError("Eligible measurements cannot carry an ineligibility reason.")
        if not self.eligible and not self.reason:
            raise ValueError("Ineligible measurements must state a reason.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "definition_id": self.definition_id,
            "definition_version": self.definition_version,
            "mask_type": self.mask_type,
            "units": self.units,
            "calibration_status": self.calibration_status,
            "uncertainty": self.uncertainty,
            "eligible": self.eligible,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> "MeasurementContract":
        expected = {"definition_id", "definition_version", "mask_type", "units", "calibration_status", "uncertainty", "eligible", "reason"}
        if not isinstance(payload, dict) or set(payload) != expected:
            raise ValueError("Measurement-contract metadata has an invalid schema.")
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class RasterRole:
    """Explicit raster binding to a frame; unlisted arrays are compact data."""

    kind: str
    dtype: str
    layout: str = "HW"
    value_range: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"image", "mask", "instances", "raster"}:
            raise ValueError("Raster role kind must be image, mask, instances, or raster.")
        try:
            dtype = np.dtype(self.dtype)
        except (TypeError, ValueError) as error:
            raise ValueError("Raster role dtype is invalid.") from error
        if dtype.hasobject or dtype.kind not in "biuf":
            raise ValueError("Raster roles require a primitive real numeric dtype.")
        object.__setattr__(self, "dtype", dtype.str)
        if self.layout not in {"HW", "HWC", "CHW"}:
            raise ValueError("Raster layout must be HW, HWC, or CHW.")
        if self.kind in {"mask", "instances"} and self.layout != "HW":
            raise ValueError(f"{self.kind} rasters must use HW layout.")
        if self.value_range is not None:
            if not isinstance(self.value_range, (tuple, list)) or len(self.value_range) != 2:
                raise ValueError("Raster value_range must be a finite minimum/maximum pair.")
            lower, upper = self.value_range
            if (
                isinstance(lower, bool) or isinstance(upper, bool)
                or not isinstance(lower, (int, float)) or not isinstance(upper, (int, float))
                or not math.isfinite(lower) or not math.isfinite(upper) or lower > upper
            ):
                raise ValueError("Raster value_range must have finite ordered bounds.")
            object.__setattr__(self, "value_range", (float(lower), float(upper)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "dtype": self.dtype,
            "layout": self.layout,
            "value_range": None if self.value_range is None else list(self.value_range),
        }

    @classmethod
    def from_dict(cls, payload: Any) -> "RasterRole":
        expected = {"kind", "dtype", "layout", "value_range"}
        if not isinstance(payload, dict) or set(payload) != expected:
            raise ValueError("Raster role metadata has an invalid schema.")
        values = dict(payload)
        if values["value_range"] is not None:
            if not isinstance(values["value_range"], list):
                raise ValueError("Raster role value_range must be a JSON array.")
            values["value_range"] = tuple(values["value_range"])
        return cls(**values)


@dataclass(frozen=True, slots=True)
class ExecutionRecord:
    status: ExecutionStatus
    reason: str = ""
    warnings: tuple[str, ...] = ()
    elapsed_ms: float | None = None
    peak_memory_bytes: int | None = None
    transfer_ms: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", ExecutionStatus(self.status))
        if not isinstance(self.reason, str) or (self.reason and self.reason != self.reason.strip()):
            raise ValueError("execution reason must be trimmed text.")
        warnings = tuple(self.warnings)
        if any(not isinstance(value, str) or not value or value != value.strip() for value in warnings):
            raise ValueError("execution warnings must be non-empty trimmed strings.")
        object.__setattr__(self, "warnings", warnings)
        for name in ("elapsed_ms", "transfer_ms"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0):
                raise ValueError(f"execution {name} must be a finite non-negative number.")
        if self.peak_memory_bytes is not None and (type(self.peak_memory_bytes) is not int or self.peak_memory_bytes < 0):
            raise ValueError("peak_memory_bytes must be a non-negative integer.")
        if self.status in {ExecutionStatus.FAILED, ExecutionStatus.UNSUPPORTED} and not self.reason:
            raise ValueError("Failed or unsupported execution must state its reason.")
        if self.status in {ExecutionStatus.AVAILABLE, ExecutionStatus.EMPTY} and self.reason:
            raise ValueError("Available or empty execution cannot carry a failure reason.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "reason": self.reason,
            "warnings": list(self.warnings),
            "elapsed_ms": self.elapsed_ms,
            "peak_memory_bytes": self.peak_memory_bytes,
            "transfer_ms": self.transfer_ms,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> "ExecutionRecord":
        expected = {"status", "reason", "warnings", "elapsed_ms", "peak_memory_bytes", "transfer_ms"}
        if not isinstance(payload, dict) or set(payload) != expected:
            raise ValueError("Execution metadata has an invalid schema.")
        values = dict(payload)
        if not isinstance(values["warnings"], list):
            raise ValueError("Execution warnings must be a JSON array.")
        values["warnings"] = tuple(values["warnings"])
        return cls(**values)


@dataclass(frozen=True, slots=True)
class CheckpointLimits:
    """Per-operation caps, which may be lowered but never raised past hard caps."""

    max_total_array_bytes: int = _DEFAULT_TOTAL_ARRAY_BYTES
    max_array_bytes: int = MAX_ARRAY_BYTES
    max_metadata_bytes: int = MAX_METADATA_BYTES
    max_npz_bytes: int = MAX_CHECKPOINT_BYTES
    max_arrays: int = MAX_ARRAYS
    max_dimensions: int = MAX_ARRAY_DIMENSIONS

    def __post_init__(self) -> None:
        caps = {
            "max_total_array_bytes": MAX_CHECKPOINT_BYTES,
            "max_array_bytes": MAX_ARRAY_BYTES,
            "max_metadata_bytes": MAX_METADATA_BYTES,
            "max_npz_bytes": MAX_CHECKPOINT_BYTES,
            "max_arrays": MAX_ARRAYS,
            "max_dimensions": MAX_ARRAY_DIMENSIONS,
        }
        for name, cap in caps.items():
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= cap:
                raise ValueError(f"{name} must be between 1 and its hard safety limit ({cap}).")
        if self.max_array_bytes > self.max_total_array_bytes:
            raise ValueError("max_array_bytes cannot exceed max_total_array_bytes.")


@dataclass(frozen=True, slots=True)
class CheckpointReference:
    checkpoint_sha256: str
    path: Path
    identity: CheckpointIdentity
    execution: ExecutionRecord


@dataclass(frozen=True, slots=True)
class BenchmarkCheckpoint:
    checkpoint_sha256: str
    identity: CheckpointIdentity
    execution: ExecutionRecord
    measurement: MeasurementContract
    metadata: Mapping[str, Any]
    arrays: Mapping[str, np.ndarray]
    raster_roles: Mapping[str, RasterRole]


def _cancelled(cancel: Any) -> bool:
    if cancel is None:
        return False
    if callable(cancel):
        return bool(cancel())
    is_set = getattr(cancel, "is_set", None)
    if not callable(is_set):
        raise TypeError("cancel must be a callable or an event with is_set().")
    return bool(is_set())


def _raise_if_cancelled(cancel: Any) -> None:
    if _cancelled(cancel):
        raise CheckpointCancelled("Benchmark checkpoint serialization was cancelled.")


def _array_descriptor(value: Any, *, limits: CheckpointLimits, name: str) -> tuple[tuple[int, ...], np.dtype, int]:
    """Inspect array size and dtype without copying a tensor to the CPU."""

    if isinstance(value, np.ndarray):
        shape, dtype, nbytes = tuple(value.shape), value.dtype, int(value.nbytes)
    elif hasattr(value, "detach") and hasattr(value, "device") and hasattr(value, "numel"):
        try:
            shape = tuple(int(size) for size in value.shape)
            nbytes = int(value.numel()) * int(value.element_size())
            dtype_name = str(value.dtype)
            if dtype_name.startswith("torch."):
                dtype_name = dtype_name[6:]
            dtype = np.dtype(dtype_name)
        except Exception as error:
            raise CheckpointError(f"Cannot inspect tensor {name} for bounded serialization.") from error
        if math.prod(shape) * dtype.itemsize != nbytes:
            raise CheckpointError(f"Tensor {name} has an unsupported or inconsistent storage layout.")
    else:
        raise CheckpointError(f"Array {name} must be a NumPy array or a PyTorch tensor.")
    if dtype.hasobject or dtype.kind not in "biufc":
        raise CheckpointError(f"Array {name} must have a primitive numeric dtype; object and structured arrays are forbidden.")
    if len(shape) > limits.max_dimensions or any(type(size) is not int or size < 0 for size in shape):
        raise CheckpointError(f"Array {name} exceeds the dimension limit or has invalid dimensions.")
    if nbytes > limits.max_array_bytes:
        raise CheckpointError(f"Array {name} exceeds the {limits.max_array_bytes}-byte limit.")
    return shape, dtype, nbytes


def _image_range(identity: CheckpointIdentity) -> tuple[float, float]:
    value = identity.image_semantics.get("value_range")
    if not isinstance(value, tuple) or len(value) != 2:
        raise CheckpointError("Image raster semantics must declare a numeric value_range pair.")
    lower, upper = value
    if (
        isinstance(lower, bool) or isinstance(upper, bool)
        or not isinstance(lower, (int, float)) or not isinstance(upper, (int, float))
        or not math.isfinite(lower) or not math.isfinite(upper) or lower > upper
    ):
        raise CheckpointError("Image raster value_range is invalid.")
    return float(lower), float(upper)


def _validate_raster_contract(
    name: str,
    role: RasterRole,
    shape: tuple[int, ...],
    dtype: np.dtype,
    identity: CheckpointIdentity,
) -> tuple[float, float] | None:
    if dtype.str != role.dtype:
        raise CheckpointError(f"Raster {name} dtype {dtype.str} differs from its declared role dtype {role.dtype}.")
    height, width = identity.coordinate_frame.shape
    if role.layout == "HW":
        spatial = shape == (height, width)
    elif role.layout == "HWC":
        spatial = len(shape) == 3 and shape[:2] == (height, width) and shape[2] > 0
    else:
        spatial = len(shape) == 3 and shape[1:] == (height, width) and shape[0] > 0
    if not spatial:
        raise CheckpointError(f"Raster {name} shape {shape} does not match {identity.coordinate_frame.frame_id} {identity.coordinate_frame.shape} with {role.layout} layout.")

    if role.kind == "image":
        try:
            expected_dtype = np.dtype(identity.image_semantics["dtype"])
        except (TypeError, ValueError) as error:
            raise CheckpointError("Image semantics must declare a valid raster dtype.") from error
        if expected_dtype.str != role.dtype:
            raise CheckpointError("Image raster dtype differs from the image semantics identity.")
        channels = {"RGB": 3, "BGR": 3, "RGBA": 4, "BGRA": 4, "GRAY": 1, "GRAYSCALE": 1, "Y": 1}.get(
            str(identity.image_semantics["color_order"]).upper()
        )
        if channels is None:
            raise CheckpointError("Image color_order must identify a supported channel layout.")
        actual_channels = 1 if role.layout == "HW" else shape[2] if role.layout == "HWC" else shape[0]
        if actual_channels != channels:
            raise CheckpointError("Image raster channels disagree with its color_order semantics.")
        expected_range = _image_range(identity)
        if role.value_range is not None and role.value_range != expected_range:
            raise CheckpointError("Image raster role range differs from the image semantics identity.")
        return expected_range

    if role.kind == "mask":
        representations = {
            "boolean": (np.dtype(np.bool_).str, {0, 1}),
            "uint8_0_1": (np.dtype(np.uint8).str, {0, 1}),
            "uint8_0_255": (np.dtype(np.uint8).str, {0, 255}),
        }
        representation = identity.mask_semantics["representation"]
        if representation not in representations:
            raise CheckpointError(f"Unsupported explicit mask representation: {representation!r}.")
        expected_dtype, _ = representations[representation]
        if role.dtype != expected_dtype or role.value_range is not None:
            raise CheckpointError("Mask dtype or range does not match its declared representation semantics.")
        return None

    if role.kind == "instances":
        try:
            expected_dtype = np.dtype(identity.instance_semantics["label_dtype"])
        except (TypeError, ValueError) as error:
            raise CheckpointError("Instance semantics must declare a valid label dtype.") from error
        if expected_dtype.kind not in "iu" or expected_dtype.str != role.dtype or role.value_range is not None:
            raise CheckpointError("Instance raster dtype does not match integer label semantics.")
        return None

    return role.value_range


def _validate_raster_values(name: str, array: np.ndarray, role: RasterRole, identity: CheckpointIdentity) -> None:
    value_range = _validate_raster_contract(name, role, tuple(array.shape), array.dtype, identity)
    if role.kind == "mask":
        representation = identity.mask_semantics["representation"]
        allowed = {"boolean": {0, 1}, "uint8_0_1": {0, 1}, "uint8_0_255": {0, 255}}[representation]
        if not set(np.unique(array).tolist()).issubset(allowed):
            raise CheckpointError(f"Mask raster {name} contains values outside its declared representation.")
    elif role.kind == "instances":
        if array.size and (np.any(array < 0) or int(array.max()) < 0):
            raise CheckpointError(f"Instance raster {name} contains negative labels.")
        if identity.instance_semantics["background_label"] != 0:
            raise CheckpointError("Instance raster background must be integer zero.")
    if value_range is not None:
        if array.size and (not np.isfinite(array).all() or float(array.min()) < value_range[0] or float(array.max()) > value_range[1]):
            raise CheckpointError(f"Raster {name} contains values outside its declared range.")


def _numpy_payload(value: Any, *, allow_cpu_transfer: bool, limits: CheckpointLimits, name: str) -> np.ndarray:
    if isinstance(value, np.ndarray):
        array = value
    elif hasattr(value, "detach") and hasattr(value, "device") and hasattr(value, "numel"):
        if not allow_cpu_transfer:
            raise CheckpointError("CPU checkpoint transfer requires allow_cpu_transfer=True.")
        try:
            element_count = int(value.numel())
            element_size = int(value.element_size())
        except Exception as error:
            raise CheckpointError(f"Cannot inspect tensor {name} for bounded serialization.") from error
        estimated = element_count * element_size
        if estimated > limits.max_array_bytes:
            raise CheckpointError(f"Array {name} exceeds the {limits.max_array_bytes}-byte limit.")
        # copy=True is explicit: a live CUDA tensor is moved to host only on
        # this benchmark API and never by the production pipeline.
        tensor = value.detach().contiguous().to(device="cpu", copy=True)
        array = tensor.numpy()
        if array.nbytes != estimated:
            raise CheckpointError(f"Tensor {name} changed size during CPU serialization.")
    else:
        raise CheckpointError(f"Array {name} must be a NumPy array or a PyTorch tensor.")

    result = np.array(array, copy=True, order="C")
    result.flags.writeable = False
    return result


def _array_data_sha256(array: np.ndarray) -> str:
    raw = b"" if array.nbytes == 0 else memoryview(np.ascontiguousarray(array)).cast("B")
    return sha256(raw).hexdigest()


def _estimate_npz_size(inventory: Mapping[str, tuple[tuple[int, ...], np.dtype, int]]) -> int:
    """Bound the stored NPZ size without materializing any tensor on the CPU.

    ``np.savez`` writes NPY v1 headers into uncompressed ZIP members. Its
    members use ``force_zip64=True``, adding a 20-byte local ZIP extra field;
    central-directory records and the end record are also included here.
    Array names and the NPY headers are derivable from the preflight inventory,
    so this estimate is exact for that format and runs before tensor transfer.
    The post-serialization size check remains as a guard against format drift.
    """

    total = 22  # ordinary ZIP end-of-central-directory record
    for name, (shape, dtype, nbytes) in inventory.items():
        try:
            header = BytesIO()
            np.lib.format.write_array_header_1_0(
                header,
                {
                    "descr": np.lib.format.dtype_to_descr(dtype),
                    "fortran_order": False,
                    "shape": shape,
                },
            )
        except (TypeError, ValueError, OverflowError) as error:
            raise CheckpointError(f"Cannot preflight the NPZ header for array {name}.") from error
        member_name_bytes = len(f"{name}.npy".encode("ascii"))
        member_bytes = header.tell() + nbytes
        # ZIP local header + filename + forced ZIP64 extra, and central
        # directory header + filename. The member payload is stored verbatim.
        total += member_bytes + 30 + member_name_bytes + 20 + 46 + member_name_bytes
    return total


def _array_spec(name: str, array: np.ndarray, role: RasterRole | None) -> dict[str, Any]:
    return {
        "name": name,
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "layout": "C-contiguous",
        "role": {"kind": "compact"} if role is None else role.to_dict(),
        "nbytes": int(array.nbytes),
        "data_sha256": _array_data_sha256(array),
    }


def _content_sha256(manifest: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in manifest.items() if key != "checkpoint_sha256"}
    return sha256(b"seedfiddle-benchmark-checkpoint-v1\0" + _canonical_json(payload, name="checkpoint manifest")).hexdigest()


def publish_checkpoint(
    root: Path | str,
    identity: CheckpointIdentity,
    arrays: Mapping[str, Any],
    *,
    execution: ExecutionRecord,
    measurement: MeasurementContract,
    raster_roles: Mapping[str, RasterRole] | None = None,
    metadata: Mapping[str, Any] | None = None,
    allow_cpu_transfer: bool = False,
    limits: CheckpointLimits = CheckpointLimits(),
    cancel: Any = None,
) -> CheckpointReference:
    """Stage a bounded benchmark checkpoint and publish it by content hash.

    ``allow_cpu_transfer`` is deliberately required for the operation. In
    particular, GPU tensors are copied to CPU only after their sizes are known
    to fit the caller's limits.
    """

    if not allow_cpu_transfer:
        raise CheckpointError("Benchmark checkpoint serialization requires explicit allow_cpu_transfer=True.")
    if not isinstance(identity, CheckpointIdentity):
        raise TypeError("identity must be a CheckpointIdentity.")
    if not isinstance(execution, ExecutionRecord):
        raise TypeError("execution must be an ExecutionRecord.")
    if not isinstance(measurement, MeasurementContract):
        raise TypeError("measurement must be a MeasurementContract.")
    if not isinstance(arrays, Mapping):
        raise TypeError("arrays must be a mapping of safe names to NumPy arrays or tensors.")
    if len(arrays) > limits.max_arrays:
        raise CheckpointError(f"Checkpoint exceeds the {limits.max_arrays}-array limit.")
    if execution.status in {ExecutionStatus.FAILED, ExecutionStatus.UNSUPPORTED} and arrays:
        raise CheckpointError("Failed or unsupported stages cannot publish output arrays.")
    if execution.status in {ExecutionStatus.FAILED, ExecutionStatus.UNSUPPORTED} and measurement.eligible:
        raise CheckpointError("Failed or unsupported stages cannot mark measurements eligible.")
    if execution.status in {ExecutionStatus.AVAILABLE, ExecutionStatus.EMPTY} and not arrays:
        raise CheckpointError("Available or empty stages must retain at least one output array.")

    roles = {} if raster_roles is None else dict(raster_roles)
    if any(not isinstance(name, str) or not isinstance(role, RasterRole) for name, role in roles.items()):
        raise CheckpointError("raster_roles must map array names to RasterRole contracts.")
    if set(roles) - set(arrays):
        raise CheckpointError("A raster role names an array that is not present in the checkpoint.")

    # Inventory every source before any tensor can be copied from CUDA. This
    # makes the aggregate cap a true preflight rather than a post-transfer test.
    inventory: dict[str, tuple[tuple[int, ...], np.dtype, int]] = {}
    total_bytes = 0
    for name, value in arrays.items():
        if (
            not isinstance(name, str)
            or not _ARRAY_NAME.fullmatch(name)
            or name.endswith(".npy")
            or name in _RESERVED_ARRAY_NAMES
        ):
            raise CheckpointError(f"Invalid checkpoint array name: {name!r}.")
        descriptor = _array_descriptor(value, limits=limits, name=name)
        shape, dtype, nbytes = descriptor
        total_bytes += nbytes
        if total_bytes > limits.max_total_array_bytes:
            raise CheckpointError(f"Checkpoint arrays exceed the {limits.max_total_array_bytes}-byte total limit.")
        role = roles.get(name)
        if role is not None:
            try:
                _validate_raster_contract(name, role, shape, dtype, identity)
            except CheckpointError:
                raise
            except Exception as error:
                raise CheckpointError(f"Raster {name} does not satisfy its explicit role contract.") from error
        inventory[name] = descriptor

    estimated_npz_bytes = _estimate_npz_size(inventory)
    if estimated_npz_bytes > limits.max_npz_bytes:
        raise CheckpointError(
            f"Checkpoint array archive requires approximately {estimated_npz_bytes} bytes, "
            f"exceeding the {limits.max_npz_bytes}-byte limit."
        )

    _raise_if_cancelled(cancel)
    normalized: dict[str, np.ndarray] = {}
    for name, value in arrays.items():
        array = _numpy_payload(value, allow_cpu_transfer=True, limits=limits, name=name)
        expected_shape, expected_dtype, expected_bytes = inventory[name]
        if tuple(array.shape) != expected_shape or array.dtype != expected_dtype or array.nbytes != expected_bytes:
            raise CheckpointError(f"Array {name} changed shape, dtype, or size during checkpoint staging.")
        role = roles.get(name)
        if role is not None:
            _validate_raster_values(name, array, role, identity)
        normalized[name] = array
        _raise_if_cancelled(cancel)

    try:
        frozen_metadata = _freeze_json({} if metadata is None else metadata, name="checkpoint metadata")
        if not isinstance(frozen_metadata, Mapping):
            raise ValueError("checkpoint metadata must be a JSON object.")
        _canonical_json(frozen_metadata, name="checkpoint metadata", maximum=limits.max_metadata_bytes)
    except ValueError as error:
        raise CheckpointError(str(error)) from error
    manifest: dict[str, Any] = {
        "format": CHECKPOINT_FORMAT,
        "version": CHECKPOINT_VERSION,
        "identity": identity.to_dict(),
        "execution": execution.to_dict(),
        "measurement": measurement.to_dict(),
        "metadata": _thaw_json(frozen_metadata),
        "arrays": [_array_spec(name, value, roles.get(name)) for name, value in sorted(normalized.items())],
    }
    manifest["checkpoint_sha256"] = _content_sha256(manifest)
    digest = manifest["checkpoint_sha256"]
    destination_root = Path(root).resolve()
    destination_root.mkdir(parents=True, exist_ok=True)
    destination = destination_root / f"{digest}.checkpoint"
    if destination.exists():
        checkpoint = load_checkpoint(destination_root, digest, limits=limits, expected_identity=identity, expected_measurement=measurement)
        if checkpoint.execution != execution or _thaw_json(checkpoint.metadata) != _thaw_json(frozen_metadata):
            raise CheckpointError("An immutable checkpoint with this content address has conflicting metadata.")
        return CheckpointReference(digest, destination, identity, execution)

    staging = Path(tempfile.mkdtemp(prefix=f".{digest}.", dir=destination_root))
    try:
        _raise_if_cancelled(cancel)
        with (staging / "arrays.npz").open("wb") as stream:
            # Stored NPZ members avoid turning legitimate all-zero masks into
            # high-ratio compressed entries that our bomb-ratio guard rejects.
            np.savez(stream, **normalized)
        archive_size = (staging / "arrays.npz").stat().st_size
        if archive_size > limits.max_npz_bytes:
            raise CheckpointError(f"Checkpoint array archive exceeds the {limits.max_npz_bytes}-byte limit.")
        _raise_if_cancelled(cancel)
        encoded_manifest = _canonical_json(manifest, name="checkpoint manifest", maximum=limits.max_metadata_bytes)
        with (staging / "manifest.json").open("wb") as stream:
            stream.write(encoded_manifest)
            stream.flush()
            os.fsync(stream.fileno())
        _raise_if_cancelled(cancel)
        try:
            os.replace(staging, destination)
        except OSError:
            # A concurrent writer may have won the same immutable address.
            if not destination.is_dir():
                raise
            loaded = load_checkpoint(destination_root, digest, limits=limits, expected_identity=identity, expected_measurement=measurement)
            if loaded.checkpoint_sha256 != digest:
                raise CheckpointError("A conflicting checkpoint occupied the requested content address.")
        loaded = load_checkpoint(destination_root, digest, limits=limits, expected_identity=identity, expected_measurement=measurement)
        return CheckpointReference(digest, destination, identity, loaded.execution)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}.")
        result[key] = value
    return result


def _read_manifest(path: Path, limits: CheckpointLimits) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise InvalidCheckpoint("Checkpoint manifest is missing or is a symbolic link.")
    if path.stat().st_size > limits.max_metadata_bytes:
        raise InvalidCheckpoint("Checkpoint manifest exceeds the metadata size limit.")
    try:
        with path.open("rb") as stream:
            encoded = stream.read(limits.max_metadata_bytes + 1)
        if len(encoded) > limits.max_metadata_bytes:
            raise InvalidCheckpoint("Checkpoint manifest exceeds the metadata size limit.")
        payload = json.loads(
            encoded.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=lambda item: (_ for _ in ()).throw(ValueError(f"Invalid JSON number {item}.")),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise InvalidCheckpoint("Checkpoint manifest is malformed JSON.") from error
    expected = {"format", "version", "checkpoint_sha256", "identity", "execution", "measurement", "metadata", "arrays"}
    if not isinstance(payload, dict) or set(payload) != expected:
        raise InvalidCheckpoint("Checkpoint manifest has an invalid schema.")
    if payload["format"] != CHECKPOINT_FORMAT or type(payload["version"]) is not int or payload["version"] != CHECKPOINT_VERSION:
        raise InvalidCheckpoint("Unsupported benchmark checkpoint format or version.")
    _require_sha256(payload["checkpoint_sha256"], "checkpoint SHA-256")
    if _content_sha256(payload) != payload["checkpoint_sha256"]:
        raise InvalidCheckpoint("Checkpoint metadata or content address has been tampered with.")
    return payload


def _decode_raster_role(payload: Any) -> RasterRole | None:
    if isinstance(payload, dict) and payload == {"kind": "compact"}:
        return None
    try:
        return RasterRole.from_dict(payload)
    except (TypeError, ValueError) as error:
        raise InvalidCheckpoint("Checkpoint array role is malformed.") from error


def _validate_array_specs(
    payload: Any,
    limits: CheckpointLimits,
    identity: CheckpointIdentity,
) -> tuple[dict[str, Any], ...]:
    if not isinstance(payload, list) or len(payload) > limits.max_arrays:
        raise InvalidCheckpoint("Checkpoint array inventory exceeds its limit or has the wrong type.")
    specs: list[dict[str, Any]] = []
    seen: set[str] = set()
    total = 0
    for item in payload:
        expected = {"name", "dtype", "shape", "layout", "role", "nbytes", "data_sha256"}
        if not isinstance(item, dict) or set(item) != expected:
            raise InvalidCheckpoint("Checkpoint array specification has an invalid schema.")
        name = item["name"]
        if (
            not isinstance(name, str)
            or not _ARRAY_NAME.fullmatch(name)
            or name.endswith(".npy")
            or name in _RESERVED_ARRAY_NAMES
            or name in seen
        ):
            raise InvalidCheckpoint("Checkpoint contains a duplicate or invalid array name.")
        seen.add(name)
        try:
            dtype = np.dtype(item["dtype"])
        except (TypeError, ValueError) as error:
            raise InvalidCheckpoint(f"Checkpoint array {name} has an invalid dtype.") from error
        if dtype.hasobject or dtype.kind not in "biufc" or dtype.str != item["dtype"] or item["layout"] != "C-contiguous":
            raise InvalidCheckpoint(f"Checkpoint array {name} uses a forbidden or non-canonical dtype.")
        shape = item["shape"]
        if not isinstance(shape, list) or len(shape) > limits.max_dimensions or any(type(part) is not int or part < 0 for part in shape):
            raise InvalidCheckpoint(f"Checkpoint array {name} has invalid dimensions.")
        elements = math.prod(shape) if shape else 1
        nbytes = elements * dtype.itemsize
        if type(item["nbytes"]) is not int or item["nbytes"] != nbytes or nbytes > limits.max_array_bytes:
            raise InvalidCheckpoint(f"Checkpoint array {name} has inconsistent or oversized byte length.")
        role = _decode_raster_role(item["role"])
        if role is not None:
            try:
                _validate_raster_contract(name, role, tuple(shape), dtype, identity)
            except (CheckpointError, TypeError, ValueError) as error:
                raise InvalidCheckpoint(f"Checkpoint raster {name} violates its frame or semantic contract.") from error
        total += nbytes
        if total > limits.max_total_array_bytes:
            raise InvalidCheckpoint("Checkpoint arrays exceed the total-byte limit.")
        _require_sha256(item["data_sha256"], f"array {name} SHA-256")
        specs.append(item)
    if specs != sorted(specs, key=lambda item: item["name"]):
        raise InvalidCheckpoint("Checkpoint array specifications are not in canonical order.")
    return tuple(specs)


def _validate_npz(
    path: Path,
    specs: tuple[dict[str, Any], ...],
    limits: CheckpointLimits,
    identity: CheckpointIdentity,
) -> dict[str, np.ndarray]:
    if path.is_symlink() or not path.is_file():
        raise InvalidCheckpoint("Checkpoint array archive is missing or linked.")
    try:
        with path.open("rb") as stream:
            snapshot = stream.read(limits.max_npz_bytes + 1)
    except OSError as error:
        raise InvalidCheckpoint("Checkpoint array archive could not be read.") from error
    if len(snapshot) > limits.max_npz_bytes:
        raise InvalidCheckpoint("Checkpoint array archive exceeds its byte limit.")
    expected_names = {f"{item['name']}.npy" for item in specs}
    try:
        with zipfile.ZipFile(BytesIO(snapshot), "r") as archive:
            infos = archive.infolist()
            if len(infos) != len(specs) or len(infos) > limits.max_arrays:
                raise InvalidCheckpoint("Checkpoint archive has an unexpected number of members.")
            names = [item.filename for item in infos]
            if len(names) != len(set(names)) or set(names) != expected_names:
                raise InvalidCheckpoint("Checkpoint archive has unexpected, duplicate, or unsafe members.")
            if any(item.is_dir() or item.file_size > limits.max_array_bytes + 65536 for item in infos):
                raise InvalidCheckpoint("Checkpoint archive contains an oversized member.")
            if sum(item.file_size for item in infos) > limits.max_total_array_bytes + limits.max_arrays * 65536:
                raise InvalidCheckpoint("Checkpoint archive exceeds the expanded size limit.")
            if any(item.file_size and (item.compress_size == 0 or item.file_size / item.compress_size > MAX_ZIP_RATIO) for item in infos):
                raise InvalidCheckpoint("Checkpoint archive has an unsafe compression ratio.")
            # The ZIP reader and np.load below both read this immutable bounded
            # byte snapshot, so replacing the path between checks cannot alter
            # the headers that NumPy will later deserialize.
            by_name = {item.filename: item for item in infos}
            for spec in specs:
                member = f"{spec['name']}.npy"
                with archive.open(member, "r") as stream:
                    try:
                        version = np.lib.format.read_magic(stream)
                        if version != (1, 0):
                            raise InvalidCheckpoint("Only bounded NPY v1.0 array members are supported.")
                        shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(
                            stream, max_header_size=65536
                        )
                    except InvalidCheckpoint:
                        raise
                    except Exception as error:
                        raise InvalidCheckpoint(f"Checkpoint array {spec['name']} has an invalid NPY header.") from error
                    if dtype.hasobject or dtype.kind not in "biufc":
                        raise InvalidCheckpoint(f"Checkpoint array {spec['name']} uses a forbidden dtype.")
                    if dtype.str != spec["dtype"] or list(shape) != spec["shape"]:
                        raise InvalidCheckpoint(f"Checkpoint array {spec['name']} header differs from its dtype/shape contract.")
                    if fortran_order:
                        raise InvalidCheckpoint(f"Checkpoint array {spec['name']} is not in the declared C-contiguous layout.")
                    if by_name[member].file_size - stream.tell() != spec["nbytes"]:
                        raise InvalidCheckpoint(f"Checkpoint array {spec['name']} payload length differs from its header.")
    except (OSError, zipfile.BadZipFile, RuntimeError) as error:
        raise InvalidCheckpoint("Checkpoint array archive is not a valid ZIP/NPZ file.") from error

    result: dict[str, np.ndarray] = {}
    try:
        with np.load(BytesIO(snapshot), allow_pickle=False, max_header_size=65536) as archive:
            if set(archive.files) != {item["name"] for item in specs}:
                raise InvalidCheckpoint("Checkpoint array names do not match the manifest.")
            for item in specs:
                name = item["name"]
                array = archive[name]
                if array.dtype.str != item["dtype"] or list(array.shape) != item["shape"] or int(array.nbytes) != item["nbytes"]:
                    raise InvalidCheckpoint(f"Checkpoint array {name} does not match its exact dtype/shape contract.")
                if _array_data_sha256(array) != item["data_sha256"]:
                    raise InvalidCheckpoint(f"Checkpoint array {name} content hash does not match.")
                role = _decode_raster_role(item["role"])
                if role is not None:
                    try:
                        _validate_raster_values(name, array, role, identity)
                    except (CheckpointError, TypeError, ValueError) as error:
                        raise InvalidCheckpoint(f"Checkpoint raster {name} violates its declared values.") from error
                immutable_bytes = array.tobytes(order="C")
                immutable = np.frombuffer(immutable_bytes, dtype=array.dtype).reshape(array.shape)
                result[name] = immutable
    except InvalidCheckpoint:
        raise
    except Exception as error:
        raise InvalidCheckpoint("Checkpoint arrays could not be loaded safely.") from error
    return result


def load_checkpoint(
    root: Path | str,
    checkpoint_sha256: str,
    *,
    expected_identity: CheckpointIdentity | None = None,
    expected_measurement: MeasurementContract | None = None,
    limits: CheckpointLimits = CheckpointLimits(),
) -> BenchmarkCheckpoint:
    """Load an immutable checkpoint and verify every identity and array byte."""

    try:
        digest = _require_sha256(checkpoint_sha256, "checkpoint SHA-256")
    except ValueError as error:
        raise InvalidCheckpoint(str(error)) from error
    base = Path(root).resolve()
    path = base / f"{digest}.checkpoint"
    if path.is_symlink() or not path.is_dir():
        raise InvalidCheckpoint("Checkpoint content address does not name a regular checkpoint directory.")
    manifest = _read_manifest(path / "manifest.json", limits)
    if manifest["checkpoint_sha256"] != digest:
        raise InvalidCheckpoint("Checkpoint directory name does not match its content address.")
    try:
        identity = CheckpointIdentity.from_dict(manifest["identity"])
        execution = ExecutionRecord.from_dict(manifest["execution"])
        measurement = MeasurementContract.from_dict(manifest["measurement"])
        metadata = _freeze_json(manifest["metadata"], name="checkpoint metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError("metadata must be an object")
        _canonical_json(metadata, name="checkpoint metadata", maximum=limits.max_metadata_bytes)
    except (TypeError, ValueError) as error:
        raise InvalidCheckpoint("Checkpoint contract metadata is invalid.") from error
    if expected_identity is not None and identity != expected_identity:
        raise CheckpointMismatch("Checkpoint source, frame, recipe, graph, settings, backend, or upstream identity differs from the replay request.")
    if expected_measurement is not None and measurement != expected_measurement:
        raise CheckpointMismatch("Checkpoint measurement definition or eligibility differs from the replay request.")
    specs = _validate_array_specs(manifest["arrays"], limits, identity)
    if execution.status in {ExecutionStatus.FAILED, ExecutionStatus.UNSUPPORTED} and specs:
        raise InvalidCheckpoint("Failed or unsupported checkpoint unexpectedly contains output arrays.")
    if execution.status in {ExecutionStatus.AVAILABLE, ExecutionStatus.EMPTY} and not specs:
        raise InvalidCheckpoint("Available or empty checkpoint has no output arrays.")
    if execution.status in {ExecutionStatus.FAILED, ExecutionStatus.UNSUPPORTED} and measurement.eligible:
        raise InvalidCheckpoint("Failed or unsupported checkpoint marks a measurement eligible.")
    arrays = _validate_npz(path / "arrays.npz", specs, limits, identity)
    roles = {
        item["name"]: role
        for item in specs
        if (role := _decode_raster_role(item["role"])) is not None
    }
    return BenchmarkCheckpoint(
        checkpoint_sha256=digest,
        identity=identity,
        execution=execution,
        measurement=measurement,
        metadata=metadata,
        arrays=MappingProxyType(arrays),
        raster_roles=MappingProxyType(roles),
    )


def load_stage_input(
    root: Path | str,
    checkpoint_sha256: str,
    *,
    expected_identity: CheckpointIdentity,
    expected_measurement: MeasurementContract | None = None,
    limits: CheckpointLimits = CheckpointLimits(),
) -> BenchmarkCheckpoint:
    """Resolve an exact stage input for replay; no frame or dtype coercion occurs."""

    if expected_identity.upstream_checkpoint_sha256 is None:
        raise CheckpointMismatch("Stage replay requires an identity bound to its exact upstream checkpoint.")
    checkpoint = load_checkpoint(
        root,
        checkpoint_sha256,
        expected_identity=expected_identity,
        expected_measurement=expected_measurement,
        limits=limits,
    )
    if checkpoint.execution.status not in {ExecutionStatus.AVAILABLE, ExecutionStatus.EMPTY}:
        raise CheckpointMismatch(f"Stage replay input is {checkpoint.execution.status.value}.")
    return checkpoint
