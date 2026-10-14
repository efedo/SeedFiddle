"""Private JSON/NPZ worker for the isolated PlantCV reference environment.

This is invoked only as ``<explicit-reference-python> -m
seedvision.benchmarking.plantcv_worker``. It is not a second application
launcher and must never be imported from the desktop analysis path.
"""

from __future__ import annotations

import argparse
import importlib.abc
import importlib.metadata
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import platform
import re
import sys
import tempfile
import time
from typing import Any, Mapping

import numpy as np

from . import reference_watershed_capture as watershed_capture

from .reference_runner import (
    MAX_ARRAY_BYTES,
    MAX_ARRAY_DIMENSIONS,
    MAX_ARRAY_ELEMENTS,
    MAX_INPUT_ARCHIVE_BYTES,
    MAX_JSON_BYTES,
    MAX_OUTPUT_ARCHIVE_BYTES,
    _RESERVED_NPZ_KEYS,
    PROTOCOL_VERSION,
    PINNED_PLANTCV_COMMIT,
    PINNED_PLANTCV_VERSION,
    PINNED_REFERENCE_PLATFORM,
    PINNED_REFERENCE_PYTHON,
    PINNED_REQUIREMENTS_SHA256,
    PINNED_SOURCE_AUDIT_SHA256,
    _canonical_name,
    _json_bytes,
    _read_file_bounded,
    _read_npz,
    _sha256,
    _strict_json_loads,
)


_OPERATION_INPUTS: dict[str, tuple[str, ...]] = {
    "threshold.binary": ("gray",),
    "threshold.mean": ("gray",),
    "threshold.otsu": ("gray",),
    "morphology.erode": ("gray",),
    "morphology.dilate": ("gray",),
    "morphology.fill": ("mask",),
    "filter.median_blur": ("gray",),
    "segmentation.watershed": ("rgb", "mask"),
    "segmentation.watershed.capture": ("rgb", "mask"),
    "analysis.size": ("image", "labels"),
    "analysis.color": ("rgb", "labels"),
    "transform.correct_color": ("target_rgb", "target_mask", "source_rgb", "source_mask"),
    "transform.auto_correct_color": ("rgb",),
}

# These limits define the runner's supported operational domain. The original
# API is called with these values unchanged; requests outside the bounded
# domain fail explicitly instead of being clamped.
MAX_KERNEL_SIDE = 255
MAX_MORPHOLOGY_ITERATIONS = 64
MAX_WATERSHED_DISTANCE = 255
MAX_COLOR_CARD_BLOCK_SIZE = 255
MAX_COLOR_CARD_RADIUS = 512

_CALLABLE_SOURCES: dict[str, tuple[str, str, str]] = {
    "threshold.binary": ("threshold.binary", "plantcv.plantcv.threshold.threshold_methods", "plantcv/plantcv/threshold/threshold_methods.py"),
    "threshold.mean": ("threshold.mean", "plantcv.plantcv.threshold.threshold_methods", "plantcv/plantcv/threshold/threshold_methods.py"),
    "threshold.otsu": ("threshold.otsu", "plantcv.plantcv.threshold.threshold_methods", "plantcv/plantcv/threshold/threshold_methods.py"),
    "morphology.erode": ("erode", "plantcv.plantcv.erode", "plantcv/plantcv/erode.py"),
    "morphology.dilate": ("dilate", "plantcv.plantcv.dilate", "plantcv/plantcv/dilate.py"),
    "morphology.fill": ("fill", "plantcv.plantcv.fill", "plantcv/plantcv/fill.py"),
    "filter.median_blur": ("median_blur", "plantcv.plantcv.median_blur", "plantcv/plantcv/median_blur.py"),
    "segmentation.watershed": ("watershed.watershed_segmentation", "plantcv.plantcv.watershed", "plantcv/plantcv/watershed.py"),
    "segmentation.watershed.capture": ("watershed.watershed_segmentation", "plantcv.plantcv.watershed", "plantcv/plantcv/watershed.py"),
    "analysis.size": ("analyze.size", "plantcv.plantcv.analyze.size", "plantcv/plantcv/analyze/size.py"),
    "analysis.color": ("analyze.color", "plantcv.plantcv.analyze.color", "plantcv/plantcv/analyze/color.py"),
    "transform.correct_color": ("transform.correct_color", "plantcv.plantcv.transform.color_correction", "plantcv/plantcv/transform/color_correction.py"),
    "transform.auto_correct_color": ("transform.auto_correct_color", "plantcv.plantcv.transform.auto_correct_color", "plantcv/plantcv/transform/auto_correct_color.py"),
}


class WorkerError(Exception):
    pass


def _read_request(path: Path) -> dict[str, Any]:
    try:
        raw = _read_file_bounded(path, MAX_JSON_BYTES, description="worker request JSON")
    except Exception as exc:
        raise WorkerError(f"Cannot read worker request: {exc}") from exc
    try:
        value = _strict_json_loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise WorkerError(f"Worker request is invalid JSON: {exc}") from exc
    if not isinstance(value, dict) or value.get("protocol_version") != PROTOCOL_VERSION:
        raise WorkerError("Unsupported reference worker protocol")
    if not isinstance(value.get("request_id"), str) or not re.fullmatch(r"[0-9a-f]{32}", value["request_id"]):
        raise WorkerError("Invalid reference worker request id")
    if not isinstance(value.get("operation"), str) or value["operation"] not in _OPERATION_INPUTS:
        raise WorkerError(f"Unsupported PlantCV reference operation: {value.get('operation')!r}")
    if not isinstance(value.get("parameters"), dict):
        raise WorkerError("Operation parameters must be a JSON object")
    if not isinstance(value.get("input_arrays"), dict):
        raise WorkerError("Input array manifest is missing")
    return value


def _platform_identity() -> str:
    machine = platform.machine().lower()
    if os.name == "nt" and machine in {"amd64", "x86_64", "x64"}:
        return "Windows-x86_64"
    return f"{platform.system()}-{machine}"


def _installed_distributions() -> dict[str, str]:
    result: dict[str, str] = {}
    # Packaging bootstrap tools are supplied by ``venv`` and are not part of
    # the PlantCV lock or imported by the reference operations.
    bootstrap_tools = {"pip", "setuptools", "wheel"}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        version = distribution.version
        if not name or not version:
            continue
        normalized = _canonical_name(name)
        if normalized in bootstrap_tools:
            continue
        if normalized in result and result[normalized] != version:
            raise WorkerError(f"Multiple installed versions found for {normalized}")
        result[normalized] = version
    return dict(sorted(result.items()))


def _verify_identity(expected: Any) -> dict[str, Any]:
    if not isinstance(expected, dict):
        raise WorkerError("Expected PlantCV identity is missing")
    if expected.get("reference") != "plantcv" or expected.get("version") != PINNED_PLANTCV_VERSION:
        raise WorkerError("Request does not target the pinned PlantCV release")
    if expected.get("commit") != PINNED_PLANTCV_COMMIT:
        raise WorkerError("Request does not target the pinned PlantCV source commit")
    if sys.version.split()[0] != PINNED_REFERENCE_PYTHON:
        raise WorkerError(f"Reference Python must be {PINNED_REFERENCE_PYTHON}; found {sys.version.split()[0]}")
    actual_platform = _platform_identity()
    if actual_platform != PINNED_REFERENCE_PLATFORM:
        raise WorkerError(f"Reference platform must be {PINNED_REFERENCE_PLATFORM}; found {actual_platform}")
    actual_version = importlib.metadata.version("plantcv")
    if actual_version != PINNED_PLANTCV_VERSION:
        raise WorkerError(f"Expected PlantCV {PINNED_PLANTCV_VERSION}; found {actual_version}")
    expected_distributions = expected.get("expected_distributions")
    if not isinstance(expected_distributions, dict) or not expected_distributions:
        raise WorkerError("Expected dependency inventory is missing")
    actual_distributions = _installed_distributions()
    normalized_expected = {
        _canonical_name(name): version for name, version in expected_distributions.items()
        if isinstance(name, str) and isinstance(version, str)
    }
    package_records = expected.get("packages")
    if not isinstance(package_records, list) or not package_records:
        raise WorkerError("Expected pinned package archive inventory is missing")
    versions_from_archives: dict[str, str] = {}
    for package in package_records:
        if not isinstance(package, dict):
            raise WorkerError("Expected pinned package archive record is invalid")
        name = package.get("name")
        version = package.get("version")
        archive_hash = package.get("archive_sha256")
        if (
            not isinstance(name, str)
            or not isinstance(version, str)
            or not isinstance(archive_hash, str)
            or not re.fullmatch(r"[0-9a-f]{64}", archive_hash)
        ):
            raise WorkerError("Expected pinned package archive record is invalid")
        key = _canonical_name(name)
        if key in versions_from_archives:
            raise WorkerError(f"Duplicate pinned package record for {key}")
        versions_from_archives[key] = version
    if normalized_expected != versions_from_archives:
        raise WorkerError("Expected dependency versions do not match the hash-pinned archive inventory")
    if normalized_expected != actual_distributions:
        missing = sorted(set(normalized_expected) - set(actual_distributions))
        extra = sorted(set(actual_distributions) - set(normalized_expected))
        different = sorted(
            name for name in set(normalized_expected) & set(actual_distributions)
            if normalized_expected[name] != actual_distributions[name]
        )
        raise WorkerError(
            "Installed reference dependencies differ from the frozen inventory "
            f"(missing={missing}, extra={extra}, different={different})"
        )
    identity_digest = expected.get("identity_digest")
    if not isinstance(identity_digest, str) or not re.fullmatch(r"[0-9a-f]{64}", identity_digest):
        raise WorkerError("Expected identity digest is invalid")
    if expected.get("source_audit_sha256") != PINNED_SOURCE_AUDIT_SHA256:
        raise WorkerError("Expected source audit manifest does not match the pin")
    if expected.get("requirements_lock_sha256") != PINNED_REQUIREMENTS_SHA256:
        raise WorkerError("Expected dependency lock does not match the pin")
    identity_fields = (
        "reference", "version", "commit", "python", "platform", "source_audit_sha256",
        "source_content_digest", "source_files", "environment_manifest_sha256",
        "requirements_lock_sha256", "packages",
    )
    identity_payload = {name: expected.get(name) for name in identity_fields}
    if _sha256(_json_bytes(identity_payload)) != identity_digest:
        raise WorkerError("Expected identity digest does not match its manifest fields")
    return {
        "reference": "plantcv",
        "version": actual_version,
        "commit": PINNED_PLANTCV_COMMIT,
        "python": PINNED_REFERENCE_PYTHON,
        "platform": actual_platform,
        "source_audit_sha256": expected.get("source_audit_sha256"),
        "source_content_digest": expected.get("source_content_digest"),
        "environment_manifest_sha256": expected.get("environment_manifest_sha256"),
        "requirements_lock_sha256": expected.get("requirements_lock_sha256"),
        "identity_digest": identity_digest,
        "actual_distributions": actual_distributions,
    }


def _verify_installed_source(expected: Any) -> None:
    if not isinstance(expected, dict):
        raise WorkerError("Expected source manifest is missing")
    # The source audit's file records are passed separately below; this guard
    # prevents accidentally treating archive package hashes as source hashes.
    files = expected.get("source_files")
    if not isinstance(files, list):
        return
    distribution = importlib.metadata.distribution("plantcv")
    distribution_root = Path(distribution.locate_file("")).resolve()
    for record in files:
        if not isinstance(record, dict):
            raise WorkerError("Expected installed PlantCV source record is invalid")
        relative = record.get("path")
        if not isinstance(relative, str) or not relative.startswith("plantcv/plantcv/"):
            continue
        candidate = (distribution_root / relative).resolve()
        try:
            candidate.relative_to(distribution_root)
            content = candidate.read_bytes()
        except (OSError, ValueError) as exc:
            raise WorkerError(f"Missing installed audited PlantCV source {relative}: {exc}") from exc
        if len(content) != record.get("bytes") or _sha256(content) != record.get("sha256"):
            raise WorkerError(f"Installed PlantCV source differs from the pinned audit: {relative}")


class _AuditedPlantCVSourceLoader(importlib.abc.Loader):
    """Compile an audited PlantCV module from checked source bytes, never a pyc."""

    def __init__(self, fullname: str, path: Path, record: Mapping[str, Any]) -> None:
        self.fullname = fullname
        self.path = path.resolve()
        self.record = dict(record)
        self.verified_functions: dict[str, Any] = {}
        self.verified_code_objects: dict[str, Any] = {}
        self.verified_classes: dict[str, Any] = {}
        self.verified_class_code_objects: dict[tuple[str, str], Any] = {}

    def create_module(self, spec: Any) -> None:
        return None

    def exec_module(self, module: Any) -> None:
        try:
            source = self.path.read_bytes()
        except OSError as exc:
            raise WorkerError(f"Cannot read audited PlantCV module {self.path}: {exc}") from exc
        if len(source) != self.record.get("bytes") or _sha256(source) != self.record.get("sha256"):
            raise WorkerError(f"PlantCV source changed before compilation: {self.record.get('path')}")
        # A custom Loader.exec_module does not call SourceFileLoader.get_code,
        # so Python never consults a stale or unchecked-hash .pyc for this
        # audited module. The verified bytes themselves become executable code.
        module.__file__ = str(self.path)
        module.__cached__ = None
        code = compile(source, str(self.path), "exec", dont_inherit=True)
        exec(code, module.__dict__)
        self.verified_functions = {
            name: value for name, value in module.__dict__.items()
            if inspect.isfunction(value) and getattr(value, "__module__", None) == self.fullname
        }
        self.verified_code_objects = {
            name: function.__code__ for name, function in self.verified_functions.items()
        }
        self.verified_classes = {
            name: value for name, value in module.__dict__.items()
            if inspect.isclass(value) and getattr(value, "__module__", None) == self.fullname
        }
        self.verified_class_code_objects = {
            (name, member_name): member.__code__
            for name, cls in self.verified_classes.items()
            for member_name, member in vars(cls).items() if inspect.isfunction(member)
        }


class _AuditedPlantCVSourceFinder(importlib.abc.MetaPathFinder):
    """Route pinned PlantCV source modules through the verified source loader."""

    def __init__(self, source_records: Any, distribution_root: Path) -> None:
        if not isinstance(source_records, list):
            raise WorkerError("Pinned PlantCV source list is invalid")
        self.distribution_root = distribution_root.resolve()
        self.modules: dict[str, tuple[Path, Mapping[str, Any]]] = {}
        for record in source_records:
            if not isinstance(record, dict):
                raise WorkerError("Pinned PlantCV source record is invalid")
            relative = record.get("path")
            if not isinstance(relative, str) or not relative.startswith("plantcv/plantcv/"):
                continue
            if "\\" in relative or ".." in relative.split("/") or not relative.endswith(".py"):
                raise WorkerError(f"Invalid audited PlantCV module path {relative!r}")
            parts = relative[:-3].split("/")
            if parts[-1] == "__init__":
                parts.pop()
            if not parts:
                raise WorkerError(f"Invalid audited PlantCV module path {relative!r}")
            fullname = ".".join(parts)
            path = (self.distribution_root / relative).resolve()
            try:
                path.relative_to(self.distribution_root)
            except ValueError as exc:
                raise WorkerError(f"Audited PlantCV module escapes its distribution: {relative}") from exc
            if fullname in self.modules:
                raise WorkerError(f"Duplicate audited PlantCV module record: {fullname}")
            self.modules[fullname] = (path, record)

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        source = self.modules.get(fullname)
        if source is None:
            return None
        module_path, record = source
        is_package = module_path.name == "__init__.py"
        loader = _AuditedPlantCVSourceLoader(fullname, module_path, record)
        return importlib.util.spec_from_file_location(
            fullname,
            module_path,
            loader=loader,
            submodule_search_locations=[str(module_path.parent)] if is_package else None,
        )


def _install_audited_source_finder(source_records: Any, distribution_root: Path) -> _AuditedPlantCVSourceFinder:
    finder = _AuditedPlantCVSourceFinder(source_records, distribution_root)
    already_loaded = sorted(set(finder.modules).intersection(sys.modules))
    if already_loaded:
        raise WorkerError(f"Audited PlantCV modules were imported before source verification: {already_loaded}")
    sys.meta_path.insert(0, finder)
    return finder


def _verify_package_import_location(distribution_root: Path | None = None) -> Path:
    """Reject a shadow PlantCV package before executing any package code."""
    if distribution_root is None:
        distribution = importlib.metadata.distribution("plantcv")
        distribution_root = Path(distribution.locate_file("")).resolve()
    else:
        distribution_root = distribution_root.resolve()
    expected_origin = (distribution_root / "plantcv" / "__init__.py").resolve()
    spec = importlib.util.find_spec("plantcv")
    if spec is None or spec.origin is None:
        raise WorkerError("Cannot locate the pinned PlantCV import package")
    if Path(spec.origin).resolve() != expected_origin:
        raise WorkerError(
            "PlantCV import would resolve outside the verified distribution: "
            f"{spec.origin} (expected {expected_origin})"
        )
    return distribution_root


def _pcv_callable(pcv: Any, attribute_path: str) -> Any:
    value = pcv
    for attribute in attribute_path.split("."):
        value = getattr(value, attribute)
    return value


def _verify_callable_origin(
    operation: str,
    pcv: Any,
    source_records: Any,
    distribution_root: Path,
    audited_finder: _AuditedPlantCVSourceFinder,
) -> None:
    """Bind the executed public callable to its exact audited source module."""
    record_map = {
        record.get("path"): record for record in source_records
        if isinstance(record, dict) and isinstance(record.get("path"), str)
    } if isinstance(source_records, list) else {}
    attribute_path, module_name, relative = _CALLABLE_SOURCES[operation]
    record = record_map.get(relative)
    if not isinstance(record, dict):
        raise WorkerError(f"Pinned source audit does not include callable module {relative}")
    expected_path = (distribution_root / relative).resolve()
    callable_object = _pcv_callable(pcv, attribute_path)
    actual_path = inspect.getsourcefile(callable_object)
    module = sys.modules.get(module_name)
    module_path = getattr(module, "__file__", None)
    loader = getattr(getattr(module, "__spec__", None), "loader", None)
    function_name = attribute_path.rsplit(".", 1)[-1]
    if (
        getattr(callable_object, "__module__", None) != module_name
        or actual_path is None
        or Path(actual_path).resolve() != expected_path
        or module_path is None
        or Path(module_path).resolve() != expected_path
        or not isinstance(loader, _AuditedPlantCVSourceLoader)
        or loader.path != expected_path
        or loader.verified_functions.get(function_name) is not callable_object
        or loader.verified_code_objects.get(function_name) is not getattr(callable_object, "__code__", None)
        or audited_finder.modules.get(module_name, (None,))[0] != expected_path
    ):
        raise WorkerError(f"PlantCV callable {operation} did not load from its audited module {relative}")
    try:
        source = expected_path.read_bytes()
    except OSError as exc:
        raise WorkerError(f"Cannot read callable source {relative}: {exc}") from exc
    if len(source) != record.get("bytes") or _sha256(source) != record.get("sha256"):
        raise WorkerError(f"PlantCV callable module differs from the pinned audit: {relative}")


def _rgb_to_plantcv_bgr(array: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(array[..., ::-1])


def _plantcv_bgr_to_rgb(array: np.ndarray) -> np.ndarray:
    if array.ndim != 3 or array.shape[2] != 3:
        return array
    return np.ascontiguousarray(array[..., ::-1])


def _capture_source_inventory(source_files: Any) -> list[dict[str, Any]]:
    if not isinstance(source_files, list):
        raise WorkerError("Capture source inventory is missing")
    records = [dict(record) for record in source_files]
    by_path = {record.get("path"): record for record in records}
    for record in watershed_capture.capture_source_records():
        previous = by_path.get(record["path"])
        if previous is not None:
            if any(previous.get(key) != record[key] for key in ("bytes", "sha256")):
                raise WorkerError("Capture source pin contradicts the canonical audit")
        else:
            records.append(record)
    return records


def _capture_bindings(pcv: Any) -> tuple[Any, ...]:
    """Verify public aliases and retain their compiled identities across the call."""
    target = pcv.watershed.watershed_segmentation
    bindings: list[Any] = [target, target.__code__]
    for alias, module_name in (
        ("color_palette", "plantcv.plantcv.color_palette"),
        ("_cv2_findcontours", "plantcv.plantcv._helpers"),
        ("_debug", "plantcv.plantcv._debug"),
    ):
        module = sys.modules.get(module_name)
        function = getattr(module, alias, None)
        loader = getattr(getattr(module, "__spec__", None), "loader", None)
        if (
            not isinstance(loader, _AuditedPlantCVSourceLoader)
            or loader.verified_functions.get(alias) is not function
            or loader.verified_code_objects.get(alias) is not getattr(function, "__code__", None)
            or target.__globals__.get(alias) is not function
        ):
            raise WorkerError(f"Capture helper alias differs from verified source: {alias}")
        bindings.extend((function, function.__code__))
    cls = type(pcv.outputs)
    module = sys.modules.get(cls.__module__)
    loader = getattr(getattr(module, "__spec__", None), "loader", None)
    method = getattr(cls, "add_observation", None)
    if (
        not isinstance(loader, _AuditedPlantCVSourceLoader)
        or loader.verified_classes.get(cls.__name__) is not cls
        or loader.verified_class_code_objects.get((cls.__name__, "add_observation")) is not getattr(method, "__code__", None)
        or target.__globals__.get("outputs") is not pcv.outputs
        or target.__globals__.get("params") is not pcv.params
    ):
        raise WorkerError("Capture observation/parameter binding differs from verified source")
    bindings.extend((cls, method, method.__code__, pcv.outputs, pcv.params))
    for function in (_rgb_to_plantcv_bgr, _plantcv_bgr_to_rgb, watershed_capture.capture_watershed,
                     watershed_capture._admit, watershed_capture._snapshot):
        bindings.extend((function, function.__code__))
    return tuple(bindings)


def _array_manifest_matches(manifest: Any, arrays: Mapping[str, np.ndarray]) -> None:
    if not isinstance(manifest, dict) or set(manifest) != set(arrays):
        raise WorkerError("Input NPZ keys differ from the declared input arrays")
    for name, array in arrays.items():
        record = manifest.get(name)
        if not isinstance(record, dict):
            raise WorkerError(f"Invalid input array declaration for {name}")
        if record.get("shape") != list(array.shape) or record.get("dtype") != array.dtype.str:
            raise WorkerError(f"Input array metadata mismatch for {name}")
        if record.get("nbytes") != array.nbytes:
            raise WorkerError(f"Input array byte count mismatch for {name}")


def _require_keys(operation: str, arrays: Mapping[str, np.ndarray]) -> None:
    required = set(_OPERATION_INPUTS[operation])
    if set(arrays) != required:
        raise WorkerError(
            f"Operation {operation} requires exactly {sorted(required)}; received {sorted(arrays)}"
        )


def _validate_image(array: np.ndarray, name: str, *, channels: int | None = None) -> None:
    if array.dtype != np.uint8:
        raise WorkerError(f"{name} must use uint8 pixels")
    if channels is None:
        if array.ndim != 2:
            raise WorkerError(f"{name} must be a two-dimensional image")
    elif array.ndim != 3 or array.shape[2] != channels:
        raise WorkerError(f"{name} must have shape (height, width, {channels})")


def _validate_mask(array: np.ndarray, name: str, shape: tuple[int, int] | None = None, *, binary: bool = False) -> None:
    if array.ndim != 2 or array.dtype not in (np.dtype("uint8"), np.dtype("int32"), np.dtype("uint16")):
        raise WorkerError(f"{name} must be a two-dimensional uint8/uint16/int32 mask")
    if shape is not None and array.shape != shape:
        raise WorkerError(f"{name} shape must match its paired image")
    if np.any(array < 0):
        raise WorkerError(f"{name} cannot contain negative labels")
    if binary and not np.isin(array, (0, 255)).all():
        raise WorkerError(f"{name} must be a 0/255 binary mask")


def _only_parameters(parameters: Mapping[str, Any], allowed: set[str]) -> None:
    extra = set(parameters) - allowed
    if extra:
        raise WorkerError(f"Unsupported parameters for this PlantCV operation: {sorted(extra)}")


def _integer(value: Any, name: str, *, minimum: int, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum or (maximum is not None and value > maximum):
        raise WorkerError(f"Parameter {name} must be an integer in the supported range")
    return value


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise WorkerError(f"Parameter {name} must be a finite number")
    return float(value)


def _object_type(parameters: Mapping[str, Any], effective: dict[str, Any]) -> str:
    value = parameters.get("object_type", "light")
    if value not in {"light", "dark"}:
        raise WorkerError("Parameter object_type must be 'light' or 'dark'")
    effective["object_type"] = value
    return value


def _to_primitive(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_to_primitive(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _to_primitive(value.item())
    if isinstance(value, Mapping):
        return {str(key): _to_primitive(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_to_primitive(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    # Altair chart internals can contain Pandas/Numpy scalars handled above;
    # unsupported objects are omitted by the caller rather than stringified.
    raise TypeError(f"Not a JSON primitive: {type(value).__name__}")


def _observations(pcv: Any) -> dict[str, Any]:
    outputs = pcv.outputs
    payload: dict[str, Any] = {}
    for name in ("observations", "measurements", "metadata"):
        value = getattr(outputs, name, None)
        if value:
            payload[name] = _to_primitive(value)
    return payload


def _run_operation(
    request: Mapping[str, Any], arrays: Mapping[str, np.ndarray], scratch: Path, pcv: Any
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    operation = str(request["operation"])
    parameters = request["parameters"]
    effective: dict[str, Any] = {"callable": ""}
    outputs: dict[str, np.ndarray] = {}
    visualizations: dict[str, Any] = {}
    observations: dict[str, Any] = {}
    channel_semantics: dict[str, Any] = {}

    def params_only(allowed: set[str]) -> None:
        _only_parameters(parameters, allowed)

    if operation.startswith("threshold."):
        _validate_image(arrays["gray"], "gray")
        object_type = _object_type(parameters, effective)
        if operation == "threshold.binary":
            params_only({"threshold", "object_type"})
            threshold = _integer(parameters.get("threshold"), "threshold", minimum=0, maximum=255)
            effective.update({"callable": "plantcv.plantcv.threshold.binary", "threshold": threshold})
            result = pcv.threshold.binary(arrays["gray"], threshold=threshold, object_type=object_type)
        elif operation == "threshold.mean":
            params_only({"ksize", "offset", "object_type"})
            ksize = _integer(parameters.get("ksize"), "ksize", minimum=3, maximum=MAX_KERNEL_SIDE)
            # Retain the request in the protocol/recipe identity. The public
            # method itself increments even sizes before its OpenCV call.
            effective_ksize = ksize + int(ksize % 2 == 0)
            offset = _number(parameters.get("offset"), "offset")
            effective.update({"callable": "plantcv.plantcv.threshold.mean", "ksize": effective_ksize, "offset": offset})
            result = pcv.threshold.mean(arrays["gray"], ksize=ksize, offset=offset, object_type=object_type)
        else:
            params_only({"object_type"})
            effective["callable"] = "plantcv.plantcv.threshold.otsu"
            result = pcv.threshold.otsu(arrays["gray"], object_type=object_type)
            # PlantCV returns the mask only. Capture the scalar separately from
            # the exact pinned OpenCV call, and verify it preserves that mask.
            import cv2
            binary_flag = cv2.THRESH_BINARY if object_type == "light" else cv2.THRESH_BINARY_INV
            selected_cutoff, adapter_mask = cv2.threshold(
                arrays["gray"], 0, 255, cv2.THRESH_OTSU | binary_flag)
            if (adapter_mask.dtype != np.uint8 or adapter_mask.shape != arrays["gray"].shape
                    or not np.array_equal(adapter_mask, result)
                    or not math.isfinite(selected_cutoff) or not selected_cutoff.is_integer()
                    or not 0 <= selected_cutoff <= 255):
                raise WorkerError("Pinned Otsu scalar adapter did not preserve the public mask")
            observations["otsu_cutoff_adapter"] = {
                "reference_selected_cutoff": int(selected_cutoff),
                "origin": "pinned_cv2.threshold_adapter",
                "public_return": "mask_only",
                "object_type": object_type,
                "method": "THRESH_OTSU|THRESH_BINARY" if object_type == "light" else "THRESH_OTSU|THRESH_BINARY_INV",
                "public_mask_match": True,
            }
        outputs["mask"] = np.asarray(result)

    elif operation in {"morphology.erode", "morphology.dilate"}:
        _validate_image(arrays["gray"], "gray")
        params_only({"ksize", "iterations"})
        ksize = _integer(parameters.get("ksize"), "ksize", minimum=2, maximum=MAX_KERNEL_SIDE)
        iterations = _integer(
            parameters.get("iterations"), "iterations", minimum=1, maximum=MAX_MORPHOLOGY_ITERATIONS
        )
        effective.update({"ksize": ksize, "iterations": iterations})
        if operation == "morphology.erode":
            effective["callable"] = "plantcv.plantcv.erode"
            result = pcv.erode(arrays["gray"], ksize=ksize, i=iterations)
        else:
            effective["callable"] = "plantcv.plantcv.dilate"
            result = pcv.dilate(arrays["gray"], ksize=ksize, i=iterations)
        outputs["mask"] = np.asarray(result)

    elif operation == "morphology.fill":
        _validate_image(arrays["mask"], "mask")
        params_only({"size"})
        size = _integer(parameters.get("size"), "size", minimum=1)
        unique = np.unique(arrays["mask"])
        if len(unique) > 2:
            raise WorkerError("PlantCV fill accepts a binary image with at most two values")
        effective.update({"callable": "plantcv.plantcv.fill", "size": size})
        outputs["mask"] = np.asarray(pcv.fill(arrays["mask"], size=size))

    elif operation == "filter.median_blur":
        _validate_image(arrays["gray"], "gray")
        params_only({"ksize"})
        ksize = _integer(parameters.get("ksize"), "ksize", minimum=1, maximum=MAX_KERNEL_SIDE)
        if ksize % 2 != 1:
            raise WorkerError("Parameter ksize must be odd for PlantCV median blur")
        effective.update({"callable": "plantcv.plantcv.median_blur", "ksize": ksize})
        outputs["image"] = np.asarray(pcv.median_blur(arrays["gray"], ksize=ksize))

    elif operation in {"segmentation.watershed", "segmentation.watershed.capture"}:
        _validate_image(arrays["rgb"], "rgb", channels=3)
        _validate_mask(arrays["mask"], "mask", arrays["rgb"].shape[:2], binary=True)
        params_only({"distance", "label"})
        distance = _integer(
            parameters.get("distance", 10), "distance", minimum=1, maximum=MAX_WATERSHED_DISTANCE
        )
        label = parameters.get("label", "SeedFiddle")
        if not isinstance(label, str) or not label or len(label) > 128:
            raise WorkerError("Parameter label must be a non-empty string up to 128 characters")
        effective.update({
            "callable": "plantcv.plantcv.watershed.watershed_segmentation", "distance": distance, "label": label,
            "algorithm": "PlantCV 4.11.3 public watershed: distanceTransformWithLabels, peak_local_max, connected markers, skimage.watershed(-distance)",
        })
        if operation == "segmentation.watershed.capture":
            watershed_capture._admit(
                arrays["mask"].shape,
                min(MAX_ARRAY_BYTES, MAX_OUTPUT_ARCHIVE_BYTES - 1024 * 1024), MAX_ARRAY_ELEMENTS,
            )
        kwargs = dict(rgb_img=_rgb_to_plantcv_bgr(arrays["rgb"]), mask=arrays["mask"], distance=distance, label=label)
        if operation == "segmentation.watershed.capture":
            captured = watershed_capture.capture_watershed(
                pcv.watershed.watershed_segmentation, mask=arrays["mask"], kwargs=kwargs,
                max_bytes=min(MAX_ARRAY_BYTES, MAX_OUTPUT_ARCHIVE_BYTES - 1024 * 1024),
                max_elements=MAX_ARRAY_ELEMENTS,
            )
            result = captured.labels
            effective["capture"] = dict(captured.metadata)
            effective["capture"]["call_id"] = request.get("request_id", "direct-worker-call")
            if captured.metadata["status"] == "available":
                outputs.update(captured.arrays)
                outputs["diagnostic"] = _plantcv_bgr_to_rgb(outputs["diagnostic_bgr"])
                effective["capture"]["artifacts"]["diagnostic"] = {
                    "source_local": "plt_img", "adapter": "BGR to RGB channel reversal",
                    "shape": list(outputs["diagnostic"].shape), "dtype": outputs["diagnostic"].dtype.str,
                    "nbytes": int(outputs["diagnostic"].nbytes), "sha256": _sha256(outputs["diagnostic"].tobytes()),
                }
                if outputs["coordinates"].shape == (0, 2):
                    del outputs["coordinates"]
                    effective["capture"]["empty_coordinates"] = dict(watershed_capture.EMPTY_COORDINATES)
        else:
            result = pcv.watershed.watershed_segmentation(**kwargs)
        channel_semantics = {
            "input_boundary": "RGB uint8 HWC",
            "plantcv_input": "BGR uint8 HWC (explicit RGB channel reversal)",
            "raster_output": "integer instance labels; no color channels",
        }
        if "labels" not in outputs:
            outputs["labels"] = np.asarray(result)
        if operation == "segmentation.watershed.capture":
            channel_semantics.update({
                "diagnostic_bgr": "actual public plt_img in BGR channel order",
                "diagnostic": "RGB adapter of that same captured BGR raster",
                "palette": "actual shuffled public RGB triples passed verbatim to OpenCV BGR drawing; indexed by label ID",
                "coordinates": "actual row,column intp64 coordinates in the input frame",
            })
        observations = _observations(pcv)

    elif operation == "analysis.size":
        image = arrays["image"]
        if image.dtype != np.uint8 or image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[2] != 3):
            raise WorkerError("image must be a uint8 grayscale or three-channel image")
        _validate_mask(arrays["labels"], "labels", image.shape[:2])
        params_only({"n_labels", "label", "calibration"})
        n_labels = _integer(parameters.get("n_labels", 1), "n_labels", minimum=1)
        label = parameters.get("label", "SeedFiddle")
        if not isinstance(label, str) or not label or len(label) > 128:
            raise WorkerError("Parameter label must be a non-empty string up to 128 characters")
        calibration = None
        if "calibration" in parameters:
            selected = parameters["calibration"]
            if not isinstance(selected, dict) or set(selected) != {"px_width", "px_height", "unit"}:
                raise WorkerError("calibration must contain exactly px_width, px_height and unit")
            px_width = _number(selected["px_width"], "calibration.px_width")
            px_height = _number(selected["px_height"], "calibration.px_height")
            unit = selected["unit"]
            if px_width <= 0 or px_height <= 0 or not isinstance(unit, str) or unit not in {"pixels", "mm"}:
                raise WorkerError("calibration requires positive scales and a pixels or mm unit")
            calibration = {"px_width": px_width, "px_height": px_height, "unit": unit}
        effective.update({"callable": "plantcv.plantcv.analyze.size", "n_labels": n_labels, "label": label, "unit": "pixels"})
        plantcv_image = _rgb_to_plantcv_bgr(image) if image.ndim == 3 else image
        if calibration is None:
            pcv.params.unit = "pixels"
            size_result = np.asarray(
                pcv.analyze.size(img=plantcv_image, labeled_mask=arrays["labels"], n_labels=n_labels, label=label)
            )
        else:
            previous = (pcv.params.px_width, pcv.params.px_height, pcv.params.unit)
            effective.update({"unit": calibration["unit"], "calibration": calibration,
                              "calibration_origin": "explicit_pcv.params_adapter"})
            try:
                pcv.params.px_width = calibration["px_width"]
                pcv.params.px_height = calibration["px_height"]
                pcv.params.unit = calibration["unit"]
                size_result = np.asarray(
                    pcv.analyze.size(img=plantcv_image, labeled_mask=arrays["labels"], n_labels=n_labels, label=label)
                )
            finally:
                pcv.params.px_width, pcv.params.px_height, pcv.params.unit = previous
                if (pcv.params.px_width, pcv.params.px_height, pcv.params.unit) != previous:
                    raise WorkerError("The public size calibration context did not restore its previous values")
            effective["calibration_context_restored"] = True
        outputs["analysis_image"] = _plantcv_bgr_to_rgb(size_result)
        if image.ndim == 3:
            channel_semantics = {
                "input_boundary": "RGB uint8 HWC",
                "plantcv_input": "BGR uint8 HWC (explicit RGB channel reversal)",
                "raster_output": "RGB uint8 HWC (PlantCV BGR visualization reversed to RGB)",
                "visualizations": "PlantCV size-analysis overlay colors are returned in RGB channel order",
            }
        else:
            channel_semantics = {
                "input_boundary": "grayscale uint8 HW",
                "plantcv_input": "grayscale uint8 HW, unchanged",
                "raster_output": "grayscale uint8 HW",
            }
        observations = _observations(pcv)

    elif operation == "analysis.color":
        _validate_image(arrays["rgb"], "rgb", channels=3)
        _validate_mask(arrays["labels"], "labels", arrays["rgb"].shape[:2])
        params_only({"n_labels", "colorspaces", "label"})
        n_labels = _integer(parameters.get("n_labels", 1), "n_labels", minimum=1)
        colorspaces = parameters.get("colorspaces", "hsv")
        if colorspaces not in {"all", "rgb", "lab", "hsv"}:
            raise WorkerError("Parameter colorspaces must be one of all, rgb, lab, or hsv")
        label = parameters.get("label", "SeedFiddle")
        if not isinstance(label, str) or not label or len(label) > 128:
            raise WorkerError("Parameter label must be a non-empty string up to 128 characters")
        effective.update({
            "callable": "plantcv.plantcv.analyze.color", "n_labels": n_labels,
            "colorspaces": colorspaces, "label": label,
        })
        plantcv_rgb = _rgb_to_plantcv_bgr(arrays["rgb"])
        chart = pcv.analyze.color(
            rgb_img=plantcv_rgb, labeled_mask=arrays["labels"], n_labels=n_labels,
            colorspaces=colorspaces, label=label,
        )
        channel_semantics = {
            "input_boundary": "RGB uint8 HWC",
            "plantcv_input": "BGR uint8 HWC (explicit RGB channel reversal)",
            "observations": "PlantCV red/green/blue channel labels refer to the original RGB boundary after conversion",
            "visualizations": "PlantCV color histogram chart uses channel labels for the original RGB boundary",
        }
        observations = _observations(pcv)
        if hasattr(chart, "to_dict"):
            visualizations["color_histogram"] = _to_primitive(chart.to_dict())

    elif operation == "transform.correct_color":
        target, source = arrays["target_rgb"], arrays["source_rgb"]
        _validate_image(target, "target_rgb", channels=3)
        _validate_image(source, "source_rgb", channels=3)
        _validate_mask(arrays["target_mask"], "target_mask", target.shape[:2])
        _validate_mask(arrays["source_mask"], "source_mask", source.shape[:2])
        params_only(set())
        effective.update({
            "callable": "plantcv.plantcv.transform.color_correction.correct_color",
            "target_mask": "target_mask", "source_mask": "source_mask",
            "output_directory": "isolated worker temporary directory",
        })
        output_directory = scratch / "color-correction"
        target_matrix, source_matrix, matrix, corrected = pcv.transform.correct_color(
            target_img=_rgb_to_plantcv_bgr(target),
            target_mask=arrays["target_mask"],
            source_img=_rgb_to_plantcv_bgr(source),
            source_mask=arrays["source_mask"],
            output_directory=str(output_directory),
        )
        outputs.update({
            "target_color_matrix": np.asarray(target_matrix),
            "source_color_matrix": np.asarray(source_matrix),
            "transformation_matrix": np.asarray(matrix),
            "corrected_image": _plantcv_bgr_to_rgb(np.asarray(corrected)),
        })
        channel_semantics = {
            "input_boundary": "target_rgb and source_rgb are RGB uint8 HWC",
            "plantcv_input": "target_img and source_img are BGR uint8 HWC (explicit RGB channel reversal)",
            "corrected_image": "RGB uint8 HWC (PlantCV BGR output reversed to RGB)",
            "color_matrices": {
                "target_color_matrix": {
                    "columns": ["chip_number", "mean_R", "mean_G", "mean_B"],
                    "channel_scale": "RGB means are normalized to [0, 1]",
                },
                "source_color_matrix": {
                    "columns": ["chip_number", "mean_R", "mean_G", "mean_B"],
                    "channel_scale": "RGB means are normalized to [0, 1]",
                },
                "transformation_matrix": {
                    "shape": [9, 9],
                    "rows_source_polynomial_basis": [
                        "R", "G", "B", "R^2", "G^2", "B^2", "R^3", "G^3", "B^3",
                    ],
                    "columns_target_polynomial_basis": [
                        "R", "G", "B", "R^2", "G^2", "B^2", "R^3", "G^3", "B^3",
                    ],
                    "basis_source": (
                        "PlantCV 4.11.3 get_matrix_m/calc_transformation_matrix: "
                        "rows are the source chip RGB powers [R,G,B,R^2,G^2,B^2,R^3,G^3,B^3]; "
                        "columns are target RGB powers [R,G,B,R^2,G^2,B^2,R^3,G^3,B^3]. "
                        "Chip means are normalized to [0,1]."
                    ),
                },
            },
        }

    elif operation == "transform.auto_correct_color":
        _validate_image(arrays["rgb"], "rgb", channels=3)
        allowed = {"color_chip_size", "adaptive_method", "block_size", "radius", "min_size", "aspect_ratio", "solidity", "roi_bounds_xyxy"}
        params_only(allowed)
        color_chip_size = parameters.get("color_chip_size")
        color_chip_size_call = color_chip_size
        if color_chip_size is not None:
            if isinstance(color_chip_size, str):
                if color_chip_size not in {"passport", "classic", "cameratrax"}:
                    raise WorkerError("color_chip_size string must be passport, classic, or cameratrax")
            elif isinstance(color_chip_size, list) and len(color_chip_size) == 2:
                color_chip_size = [
                    _number(color_chip_size[0], "color_chip_size[0]"),
                    _number(color_chip_size[1], "color_chip_size[1]"),
                ]
                if min(color_chip_size) <= 0:
                    raise WorkerError("color_chip_size dimensions must be positive")
                color_chip_size_call = tuple(color_chip_size)
            else:
                raise WorkerError("color_chip_size must be a supported card name or [width_mm, height_mm]")
        kwargs: dict[str, Any] = {}
        for name, default in (
            ("adaptive_method", 1), ("block_size", 51), ("radius", 20), ("min_size", 1000),
            ("aspect_ratio", 1.27), ("solidity", 0.8),
        ):
            if name in parameters:
                value = parameters[name]
                if name in {"adaptive_method", "block_size", "radius", "min_size"}:
                    if name == "block_size":
                        value = _integer(value, name, minimum=3, maximum=MAX_COLOR_CARD_BLOCK_SIZE)
                    elif name == "radius":
                        value = _integer(value, name, minimum=1, maximum=MAX_COLOR_CARD_RADIUS)
                    else:
                        value = _integer(value, name, minimum=0 if name == "adaptive_method" else 1)
                else:
                    value = _number(value, name)
                kwargs[name] = value
            else:
                kwargs[name] = default
        if kwargs["adaptive_method"] not in {0, 1}:
            raise WorkerError("adaptive_method must be 0 (mean) or 1 (Gaussian)")
        if kwargs["block_size"] % 2 != 1:
            raise WorkerError("block_size must be odd")
        if kwargs["aspect_ratio"] <= 0 or not (0 <= kwargs["solidity"] <= 1):
            raise WorkerError("aspect_ratio must be positive and solidity must be between 0 and 1")
        if "roi_bounds_xyxy" in parameters:
            bounds = parameters["roi_bounds_xyxy"]
            height, width = arrays["rgb"].shape[:2]
            if bounds is not None:
                if not isinstance(bounds, list) or len(bounds) != 4 or any(type(value) is not int for value in bounds):
                    raise WorkerError("roi_bounds_xyxy must be null or four integer half-open bounds")
                x0, y0, x1, y1 = bounds
                if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
                    raise WorkerError("roi_bounds_xyxy must be nonempty and inside the RGB frame")
                contour = np.array([[[x0, y0]], [[x0, y1]], [[x1, y1]], [[x1, y0]]], dtype=np.int32)
                kwargs["roi"] = pcv.Objects(
                    contours=[[contour]], hierarchy=[np.array([[[-1, -1, -1, -1]]], dtype=np.int32)]
                )
            else:
                kwargs["roi"] = None
            effective["roi_adapter"] = {
                "requested_bounds_xyxy": bounds,
                "effective_bounds_xyxy": bounds if bounds is not None else [0, 0, width, height],
                "bounds_semantics": "half-open source-frame x0,y0,x1,y1",
                "origin": "explicit_pcv.Objects_adapter",
                "public_helper": "plantcv.plantcv._helpers._rect_filter",
                "public_helper_sha256": "c05b0c944210657332ce307f2066caab55f2ebc8d8fea11988db3e476e7cdfba",
                "corrected_output_frame": "full input frame",
            }
        effective.update({
            "callable": "plantcv.plantcv.transform.auto_correct_color",
            "color_chip_size": color_chip_size,
            "adaptive_method": kwargs["adaptive_method"],
            "block_size": kwargs["block_size"],
            "radius": kwargs["radius"],
            "min_size": kwargs["min_size"],
            "aspect_ratio": kwargs["aspect_ratio"],
            "solidity": kwargs["solidity"],
        })
        corrected = np.asarray(
            pcv.transform.auto_correct_color(
                rgb_img=_rgb_to_plantcv_bgr(arrays["rgb"]), color_chip_size=color_chip_size_call, **kwargs
            )
        )
        outputs["corrected_image"] = _plantcv_bgr_to_rgb(corrected)
        observations = _observations(pcv)
        calibration_effects = {
            "unit": getattr(pcv.params, "unit", None),
            "px_width": getattr(pcv.params, "px_width", None),
            "px_height": getattr(pcv.params, "px_height", None),
            "chip_measurements": observations.get("metadata", {}),
        }
        effective["calibration_effects"] = _to_primitive(calibration_effects)
        channel_semantics = {
            "input_boundary": "RGB uint8 HWC",
            "plantcv_input": "BGR uint8 HWC (explicit RGB channel reversal)",
            "corrected_image": "RGB uint8 HWC (PlantCV BGR output reversed to RGB)",
            "observations": "Original PlantCV color-card chip-size metadata is retained",
        }

    return outputs, observations, effective, visualizations, channel_semantics


def _write_response(path: Path, response: dict[str, Any]) -> None:
    raw = _json_bytes(response)
    if len(raw) > MAX_JSON_BYTES:
        raise WorkerError("Worker response exceeds the JSON size limit")
    path.write_bytes(raw)


def _execute(request_path: Path, inputs_path: Path, response_path: Path, outputs_path: Path) -> None:
    request: dict[str, Any] | None = None
    audited_finder: _AuditedPlantCVSourceFinder | None = None
    try:
        request = _read_request(request_path)
        identity = _verify_identity(request.get("identity"))
        source_files = request["identity"].get("source_files")
        capture_mode = request["operation"] == "segmentation.watershed.capture"
        if capture_mode:
            source_files = _capture_source_inventory(source_files)
            identity["capture_source_files"] = watershed_capture.capture_source_records()
        _verify_installed_source({"source_files": source_files})
        distribution_root = _verify_package_import_location()
        arrays = _read_npz(inputs_path, max_archive_bytes=MAX_INPUT_ARCHIVE_BYTES)
        _array_manifest_matches(request["input_arrays"], arrays)
        _require_keys(request["operation"], arrays)
        # PlantCV remains unimported until the exact isolated runtime identity
        # and source checks have passed. Selected audited modules are compiled
        # directly from verified bytes; cached bytecode is never consulted.
        audited_finder = _install_audited_source_finder(source_files, distribution_root)
        from plantcv import plantcv as pcv

        outer_package = sys.modules.get("plantcv")
        expected_outer = (distribution_root / "plantcv" / "__init__.py").resolve()
        expected_inner = (distribution_root / "plantcv" / "plantcv" / "__init__.py").resolve()
        if (
            getattr(outer_package, "__file__", None) is None
            or Path(outer_package.__file__).resolve() != expected_outer
            or getattr(pcv, "__file__", None) is None
            or Path(pcv.__file__).resolve() != expected_inner
        ):
            raise WorkerError("Imported PlantCV package files differ from the pinned distribution location")
        _verify_callable_origin(request["operation"], pcv, source_files, distribution_root, audited_finder)

        pcv.params.debug = None
        pcv.params.debug_outdir = str(response_path.parent)
        pcv.params.device = 0
        pcv.params.verbose = False
        pcv.params.sample_label = "SeedFiddle"
        pcv.params.line_thickness = 1
        pcv.params.unit = "pixels"
        pcv.outputs.clear()
        capture_bindings = _capture_bindings(pcv) if capture_mode else ()
        with tempfile.TemporaryDirectory(prefix="plantcv-call-", dir=response_path.parent) as scratch_name:
            started = time.perf_counter_ns()
            output_arrays, observations, effective, visualizations, channel_semantics = _run_operation(
                request, arrays, Path(scratch_name), pcv
            )
            computation_runtime_ns = time.perf_counter_ns() - started
        if capture_mode:
            after_identity = _verify_identity(request.get("identity"))
            if after_identity != {key: value for key, value in identity.items() if key != "capture_source_files"}:
                raise WorkerError("Capture runtime identity changed during execution")
            _verify_installed_source({"source_files": source_files})
            _verify_callable_origin(request["operation"], pcv, source_files, distribution_root, audited_finder)
            after = _capture_bindings(pcv)
            if len(after) != len(capture_bindings) or any(a is not b for a, b in zip(after, capture_bindings)):
                raise WorkerError("Capture public/helper/adapter identity changed during execution")
        if not output_arrays:
            # PlantCV colour analysis returns its original distribution chart
            # and records observations, but has no raster output.
            if request["operation"] != "analysis.color":
                raise WorkerError("Selected PlantCV callable produced no checked output")
            outputs_path.write_bytes(_empty_npz_bytes())
        else:
            normalized: dict[str, np.ndarray] = {}
            total_bytes = 0
            total_elements = 0
            for name, raw_array in output_arrays.items():
                if (
                    not isinstance(name, str)
                    or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", name)
                    or name in _RESERVED_NPZ_KEYS
                ):
                    raise WorkerError(f"Invalid PlantCV output name {name!r}")
                array = np.asarray(raw_array)
                if array.dtype.kind not in "biuf":
                    raise WorkerError(f"PlantCV output {name} has unsupported dtype {array.dtype}")
                if array.ndim < 1 or array.ndim > MAX_ARRAY_DIMENSIONS or 0 in array.shape:
                    raise WorkerError(f"PlantCV output {name} has unsupported shape {array.shape}")
                if array.nbytes > MAX_ARRAY_BYTES or array.size > MAX_ARRAY_ELEMENTS:
                    raise WorkerError(f"PlantCV output {name} exceeds its size limit")
                if array.dtype.kind == "f" and not np.isfinite(array).all():
                    # Non-finite values do not form a stable numeric raster.
                    raise WorkerError(f"PlantCV output {name} contains a non-finite value")
                normalized[name] = np.ascontiguousarray(array)
                total_bytes += array.nbytes
                total_elements += int(array.size)
            if total_bytes > MAX_ARRAY_BYTES or total_elements > MAX_ARRAY_ELEMENTS:
                raise WorkerError("PlantCV outputs exceed the aggregate array limit")
            if total_bytes > MAX_OUTPUT_ARCHIVE_BYTES - (1024 * 1024):
                raise WorkerError("PlantCV outputs leave insufficient room for the bounded NPZ archive")
            np.savez_compressed(outputs_path, **normalized)
        if outputs_path.stat().st_size > MAX_OUTPUT_ARCHIVE_BYTES:
            outputs_path.unlink(missing_ok=True)
            raise WorkerError("PlantCV output archive exceeds the protocol size limit")
        declarations = [
            {"name": name, "shape": list(array.shape), "dtype": array.dtype.str, "nbytes": int(array.nbytes)}
            for name, array in normalized.items()
        ] if output_arrays else []
        response = {
            "protocol_version": PROTOCOL_VERSION,
            "request_id": request["request_id"],
            "ok": True,
            "operation": request["operation"],
            "identity": identity,
            "effective_parameters": effective,
            "intermediates": declarations,
                "observations": observations,
                "visualizations": visualizations,
                "channel_semantics": channel_semantics,
                "computation_runtime_ns": int(computation_runtime_ns),
        }
        _write_response(response_path, response)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        response = {
            "protocol_version": PROTOCOL_VERSION,
            "request_id": request.get("request_id", "") if isinstance(request, dict) else "",
            "ok": False,
            "operation": request.get("operation", "") if isinstance(request, dict) else "",
            "error": error,
        }
        try:
            _write_response(response_path, response)
        except Exception:
            print(error, file=sys.stderr)
    finally:
        if audited_finder is not None:
            try:
                sys.meta_path.remove(audited_finder)
            except ValueError:
                pass


def _empty_npz_bytes() -> bytes:
    # NumPy writes a valid, empty NPZ archive; keeping the array wire channel
    # present makes the request/response protocol shape uniform.
    import io

    stream = io.BytesIO()
    np.savez_compressed(stream)
    return stream.getvalue()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--request", required=True)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--response", required=True)
    parser.add_argument("--outputs", required=True)
    args = parser.parse_args(argv)
    _execute(Path(args.request), Path(args.inputs), Path(args.response), Path(args.outputs))
    try:
        result = _strict_json_loads(Path(args.response).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return 2
    return 0 if isinstance(result, dict) and result.get("ok") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
