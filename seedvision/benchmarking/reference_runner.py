"""Isolated, bounded execution of selected original PlantCV operations.

This module is safe to import in the desktop process. PlantCV and its optional
scientific dependencies are loaded only by :mod:`plantcv_worker` in the
explicitly supplied reference Python environment.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
import threading
import time
from typing import Any, Mapping
import uuid
import zipfile

import numpy as np


PINNED_PLANTCV_VERSION = "4.11.3"
PINNED_PLANTCV_COMMIT = "77e3322fd0cd5a9bef18bb99402f20c8f2db87a9"
PINNED_REFERENCE_PYTHON = "3.12.10"
PINNED_REFERENCE_PLATFORM = "Windows-x86_64"
PINNED_SOURCE_AUDIT_SHA256 = "a747fdfe701e22627b0861c914c12f23d88b934f1dfe3197587aecc910b92bd2"
PINNED_REQUIREMENTS_SHA256 = "80b8a6dcab4131a1c5d1c788ba8bbf12296f233fdf65f24c86bca14ea6682b5b"

PROTOCOL_VERSION = 1
MAX_JSON_BYTES = 2 * 1024 * 1024
MAX_INPUT_ARCHIVE_BYTES = 64 * 1024 * 1024
MAX_OUTPUT_ARCHIVE_BYTES = 128 * 1024 * 1024
MAX_ARCHIVE_UNCOMPRESSED_BYTES = 256 * 1024 * 1024
MAX_ARRAY_BYTES = 128 * 1024 * 1024
MAX_ARRAY_ELEMENTS = 64_000_000
MAX_ARRAY_DIMENSIONS = 3
MAX_ARRAYS = 64
MAX_CAPTURED_LOG_BYTES = 64 * 1024
MAX_PARAMETERS_BYTES = 256 * 1024
SUPPORTED_PLANTCV_OPERATIONS = frozenset({
    "threshold.binary",
    "threshold.mean",
    "threshold.otsu",
    "threshold.gaussian",
    "threshold.triangle",
    "threshold.texture",
    "threshold.dual_channels",
    "threshold.custom_range",
    "analysis.naive_bayes_classifier",
    "transform.get_color_matrix",
    "transform.std_color_matrix",
    "transform.affine_color_correction",
    "morphology.erode",
    "morphology.dilate",
    "morphology.fill",
    "filter.median_blur",
    "segmentation.watershed",
    "segmentation.watershed.capture",
    "analysis.size",
    "analysis.color",
    "transform.correct_color",
    "transform.auto_correct_color",
})

_ARRAY_KEY_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")
_ALLOWED_DTYPES = "biuf"  # bool, signed/unsigned integer, real floating point
_RESERVED_NPZ_KEYS = frozenset({"file", "allow_pickle"})


class ReferenceRunnerError(RuntimeError):
    """Base error for reference runner failures."""


class ReferenceIdentityError(ReferenceRunnerError):
    """Raised when the source or reference environment differs from its pin."""


class ReferenceProtocolError(ReferenceRunnerError):
    """Raised when a request or response violates the primitive protocol."""


class ReferenceTimeoutError(ReferenceRunnerError):
    """Raised when the isolated worker exceeds its wall-clock limit."""


class ReferenceCancelledError(ReferenceRunnerError):
    """Raised when an active reference request is cancelled."""


class ReferenceExecutionError(ReferenceRunnerError):
    """Raised when the selected original PlantCV callable fails."""


@dataclass(frozen=True)
class ReferenceResult:
    """Checked primitive metadata and raster outputs from an isolated run."""

    operation: str
    arrays: Mapping[str, np.ndarray]
    observations: Mapping[str, Any]
    effective_parameters: Mapping[str, Any]
    intermediates: tuple[Mapping[str, Any], ...]
    visualizations: Mapping[str, Any]
    channel_semantics: Mapping[str, Any]
    identity: Mapping[str, Any]
    computation_runtime_ns: int
    wall_runtime_ns: int


def _json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ReferenceProtocolError(f"Value is not strict JSON: {exc}") from exc


def _strict_json_loads(raw: str | bytes) -> Any:
    """Decode RFC-style JSON while rejecting NaN, infinities and duplicate keys."""
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value} is forbidden")

    def finite_float(value: str) -> float:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"JSON number {value!r} overflows the finite float range")
        return number

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON object key {key!r}")
            result[key] = value
        return result

    return json.loads(
        raw,
        parse_constant=reject_constant,
        parse_float=finite_float,
        object_pairs_hook=unique_object,
    )


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_json(path: Path, *, max_bytes: int = MAX_JSON_BYTES) -> tuple[Any, bytes]:
    try:
        with path.open("rb") as handle:
            raw = handle.read(max_bytes + 1)
    except OSError as exc:
        raise ReferenceIdentityError(f"Cannot read identity file {path}: {exc}") from exc
    if len(raw) > max_bytes:
        raise ReferenceIdentityError(f"Identity file is over the {max_bytes}-byte limit: {path}")
    try:
        return _strict_json_loads(raw.decode("utf-8")), raw
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ReferenceIdentityError(f"Invalid JSON identity file {path}: {exc}") from exc


def _inside(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ReferenceIdentityError("Source manifest paths must be non-empty POSIX paths")
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ReferenceIdentityError(f"Source manifest path escapes its root: {relative}") from exc
    return candidate


def _verify_source_manifest(manifest_path: Path, source_root: Path) -> tuple[dict[str, Any], str]:
    manifest, raw = _load_json(manifest_path)
    if _sha256(raw) != PINNED_SOURCE_AUDIT_SHA256:
        raise ReferenceIdentityError("Pinned PlantCV source audit manifest has changed")
    if not isinstance(manifest, dict):
        raise ReferenceIdentityError("PlantCV source audit must be a JSON object")
    if (
        manifest.get("reference") != "plantcv"
        or manifest.get("version") != PINNED_PLANTCV_VERSION
        or manifest.get("commit") != PINNED_PLANTCV_COMMIT
    ):
        raise ReferenceIdentityError("PlantCV source audit does not match the pinned release and commit")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ReferenceIdentityError("PlantCV source audit has no audited files")
    checked: list[dict[str, Any]] = []
    for record in files:
        if not isinstance(record, dict):
            raise ReferenceIdentityError("PlantCV source audit contains an invalid file record")
        relative = record.get("path")
        expected_hash = record.get("sha256")
        expected_bytes = record.get("bytes")
        if not isinstance(expected_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
            raise ReferenceIdentityError(f"Invalid SHA-256 for audited source {relative!r}")
        if isinstance(expected_bytes, bool) or not isinstance(expected_bytes, int) or expected_bytes < 0:
            raise ReferenceIdentityError(f"Invalid byte count for audited source {relative!r}")
        path = _inside(source_root, relative)
        try:
            with path.open("rb") as handle:
                content = handle.read(expected_bytes + 1)
        except OSError as exc:
            raise ReferenceIdentityError(f"Missing audited PlantCV source {relative}: {exc}") from exc
        actual_hash = _sha256(content)
        if len(content) != expected_bytes or actual_hash != expected_hash:
            raise ReferenceIdentityError(f"Audited PlantCV source changed: {relative}")
        checked.append({"path": relative, "sha256": actual_hash, "bytes": len(content)})
    source_content_digest = _sha256(_json_bytes({
        "version": PINNED_PLANTCV_VERSION,
        "commit": PINNED_PLANTCV_COMMIT,
        "files": sorted(checked, key=lambda item: item["path"]),
    }))
    return manifest, source_content_digest


def _parse_requirements_lock(path: Path) -> dict[str, tuple[str, str]]:
    try:
        with path.open("rb") as handle:
            raw = handle.read((1024 * 1024) + 1)
    except OSError as exc:
        raise ReferenceIdentityError(f"Cannot read PlantCV requirements lock {path}: {exc}") from exc
    if _sha256(raw) != PINNED_REQUIREMENTS_SHA256:
        raise ReferenceIdentityError("Pinned PlantCV requirements lock has changed")
    result: dict[str, tuple[str, str]] = {}
    for line in raw.decode("utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r"([^=\s]+)==([^\s]+)\s+--hash=sha256:([0-9a-f]{64})", line)
        if not match:
            raise ReferenceIdentityError(f"Invalid pinned package line in requirements lock: {line}")
        name, version, archive_hash = match.groups()
        key = _canonical_name(name)
        if key in result:
            raise ReferenceIdentityError(f"Duplicate package in requirements lock: {name}")
        result[key] = (version, archive_hash)
    if result.get("plantcv", (None, None))[0] != PINNED_PLANTCV_VERSION:
        raise ReferenceIdentityError("Requirements lock does not pin the audited PlantCV release")
    return result


def _build_expected_identity(
    *,
    source_manifest_path: Path,
    source_root: Path,
    environment_manifest_path: Path,
    requirements_lock_path: Path,
) -> dict[str, Any]:
    source_manifest, source_content_digest = _verify_source_manifest(source_manifest_path, source_root)
    environment_map, environment_raw = _load_json(environment_manifest_path)
    if not isinstance(environment_map, dict) or not environment_map:
        raise ReferenceIdentityError("Frozen PlantCV environment inventory must be a package-to-version map")
    expected_versions: dict[str, str] = {}
    for name, version in environment_map.items():
        if not isinstance(name, str) or not isinstance(version, str):
            raise ReferenceIdentityError("Frozen PlantCV environment entries must be string package versions")
        key = _canonical_name(name)
        if key in expected_versions:
            raise ReferenceIdentityError(f"Duplicate normalized package name in environment inventory: {name}")
        expected_versions[key] = version
    locked = _parse_requirements_lock(requirements_lock_path)
    locked_versions = {name: version_hash[0] for name, version_hash in locked.items()}
    if expected_versions != locked_versions:
        missing = sorted(set(locked_versions) - set(expected_versions))
        extra = sorted(set(expected_versions) - set(locked_versions))
        different = sorted(
            name for name in set(expected_versions) & set(locked_versions)
            if expected_versions[name] != locked_versions[name]
        )
        raise ReferenceIdentityError(
            "Frozen PlantCV environment inventory differs from the hash-pinned requirements "
            f"(missing={missing}, extra={extra}, different={different})"
        )
    identity_payload = {
        "reference": "plantcv",
        "version": PINNED_PLANTCV_VERSION,
        "commit": PINNED_PLANTCV_COMMIT,
        "python": PINNED_REFERENCE_PYTHON,
        "platform": PINNED_REFERENCE_PLATFORM,
        "source_audit_sha256": PINNED_SOURCE_AUDIT_SHA256,
        "source_content_digest": source_content_digest,
        "source_files": sorted(
            [
                {"path": item["path"], "sha256": item["sha256"], "bytes": item["bytes"]}
                for item in source_manifest["files"]
            ],
            key=lambda item: item["path"],
        ),
        "environment_manifest_sha256": _sha256(environment_raw),
        "requirements_lock_sha256": PINNED_REQUIREMENTS_SHA256,
        "packages": [
            {"name": name, "version": version_hash[0], "archive_sha256": version_hash[1]}
            for name, version_hash in sorted(locked.items())
        ],
    }
    return {
        **identity_payload,
        "expected_distributions": expected_versions,
        "identity_digest": _sha256(_json_bytes(identity_payload)),
    }


def validate_array_mapping(
    arrays: Mapping[str, Any],
    *,
    max_aggregate_bytes: int = MAX_ARRAY_BYTES,
    max_aggregate_elements: int = MAX_ARRAY_ELEMENTS,
) -> dict[str, np.ndarray]:
    """Preflight a complete NumPy-array inventory, then normalize its layout."""
    if not isinstance(arrays, Mapping) or not arrays:
        raise ReferenceProtocolError("At least one named input array is required")
    if len(arrays) > MAX_ARRAYS:
        raise ReferenceProtocolError(f"Array mapping exceeds the {MAX_ARRAYS}-array limit")
    preflight: list[tuple[str, np.ndarray]] = []
    total_bytes = 0
    total_elements = 0
    for name, value in arrays.items():
        if not isinstance(name, str) or not _ARRAY_KEY_RE.fullmatch(name) or name in _RESERVED_NPZ_KEYS:
            raise ReferenceProtocolError(f"Invalid input array name: {name!r}")
        if not isinstance(value, np.ndarray):
            raise ReferenceProtocolError(f"Input array {name!r} must be a NumPy array")
        array = value
        if array.dtype.kind not in _ALLOWED_DTYPES:
            raise ReferenceProtocolError(f"Input array {name!r} has unsupported dtype {array.dtype}")
        if array.ndim < 1 or array.ndim > MAX_ARRAY_DIMENSIONS or 0 in array.shape:
            raise ReferenceProtocolError(f"Input array {name!r} has unsupported shape {array.shape}")
        if array.size > MAX_ARRAY_ELEMENTS or array.nbytes > MAX_ARRAY_BYTES:
            raise ReferenceProtocolError(f"Input array {name!r} exceeds its size limit")
        preflight.append((name, array))
        total_bytes += array.nbytes
        total_elements += int(array.size)
    if total_bytes > max_aggregate_bytes or total_elements > max_aggregate_elements:
        raise ReferenceProtocolError("Input arrays exceed the aggregate size limit")
    for name, array in preflight:
        if array.dtype.kind == "f" and not np.isfinite(array).all():
            raise ReferenceProtocolError(f"Input array {name!r} contains a non-finite value")
    normalized = {name: np.ascontiguousarray(array) for name, array in preflight}
    return normalized


def _read_file_bounded(path: Path, maximum: int, *, description: str) -> bytes:
    try:
        with path.open("rb") as handle:
            content = handle.read(maximum + 1)
    except OSError as exc:
        raise ReferenceProtocolError(f"Cannot read {description}: {exc}") from exc
    if not content or len(content) > maximum:
        raise ReferenceProtocolError(f"{description} is missing or exceeds its {maximum}-byte limit")
    return content


def _validate_npz_bytes(
    content: bytes,
    *,
    max_archive_bytes: int,
    allow_empty: bool = False,
) -> dict[str, tuple[tuple[int, ...], np.dtype[Any], int]]:
    if not content or len(content) > max_archive_bytes:
        raise ReferenceProtocolError(f"NPZ archive size is outside the allowed range: {len(content)} bytes")
    headers: dict[str, tuple[tuple[int, ...], np.dtype[Any], int]] = {}
    aggregate_elements = 0
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            members = archive.infolist()
            if len(members) > MAX_ARRAYS:
                raise ReferenceProtocolError(f"NPZ archive exceeds the {MAX_ARRAYS}-array limit")
            if not members and not allow_empty:
                raise ReferenceProtocolError("NPZ archive is empty")
            total_uncompressed = 0
            for member in members:
                if member.is_dir() or not member.filename.endswith(".npy") or "/" in member.filename or "\\" in member.filename:
                    raise ReferenceProtocolError("NPZ archive contains an invalid member")
                key = member.filename[:-4]
                if not _ARRAY_KEY_RE.fullmatch(key) or key in _RESERVED_NPZ_KEYS or key in headers:
                    raise ReferenceProtocolError("NPZ archive contains an invalid or duplicate array name")
                total_uncompressed += member.file_size
                if member.file_size > MAX_ARRAY_BYTES:
                    raise ReferenceProtocolError(f"NPZ member {key!r} exceeds its size limit")
                if member.flag_bits & 0x1:
                    raise ReferenceProtocolError("Encrypted NPZ members are not supported")
                with archive.open(member, "r") as member_stream:
                    try:
                        version = np.lib.format.read_magic(member_stream)
                        if version == (1, 0):
                            shape, _fortran_order, dtype = np.lib.format.read_array_header_1_0(member_stream)
                        elif version in {(2, 0), (3, 0)}:
                            shape, _fortran_order, dtype = np.lib.format.read_array_header_2_0(member_stream)
                        else:
                            raise ValueError(f"unsupported NPY format version {version}")
                    except Exception as exc:
                        raise ReferenceProtocolError(f"Invalid NPY header for NPZ member {key!r}: {exc}") from exc
                    header_bytes = member_stream.tell()
                if dtype.kind not in _ALLOWED_DTYPES or dtype.hasobject:
                    raise ReferenceProtocolError(f"NPZ member {key!r} has unsupported dtype {dtype}")
                if len(shape) < 1 or len(shape) > MAX_ARRAY_DIMENSIONS or any(dim <= 0 for dim in shape):
                    raise ReferenceProtocolError(f"NPZ member {key!r} has unsupported shape {shape}")
                element_count = math.prod(shape)
                aggregate_elements += element_count
                if aggregate_elements > MAX_ARRAY_ELEMENTS:
                    raise ReferenceProtocolError("NPZ arrays exceed the aggregate element limit")
                expected_bytes = header_bytes + element_count * dtype.itemsize
                if element_count > MAX_ARRAY_ELEMENTS or expected_bytes > MAX_ARRAY_BYTES:
                    raise ReferenceProtocolError(f"NPZ member {key!r} exceeds its element or byte limit")
                if expected_bytes != member.file_size:
                    raise ReferenceProtocolError(f"NPZ member {key!r} length disagrees with its NPY header")
                headers[key] = (tuple(shape), dtype, int(expected_bytes - header_bytes))
            if total_uncompressed > MAX_ARCHIVE_UNCOMPRESSED_BYTES:
                raise ReferenceProtocolError("NPZ archive exceeds the aggregate uncompressed size limit")
    except (OSError, zipfile.BadZipFile) as exc:
        raise ReferenceProtocolError(f"Invalid NPZ archive: {exc}") from exc
    if sum(meta[2] for meta in headers.values()) > MAX_ARRAY_BYTES:
        raise ReferenceProtocolError("NPZ arrays exceed the aggregate array-byte limit")
    return headers


def _read_npz(
    path: Path, *, max_archive_bytes: int, allow_empty: bool = False
) -> dict[str, np.ndarray]:
    content = _read_file_bounded(path, max_archive_bytes, description="NPZ archive")
    headers = _validate_npz_bytes(content, max_archive_bytes=max_archive_bytes, allow_empty=allow_empty)
    try:
        with np.load(io.BytesIO(content), allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
    except Exception as exc:
        raise ReferenceProtocolError(f"Cannot decode NPZ arrays without pickle: {exc}") from exc
    for name, array in arrays.items():
        shape, dtype, nbytes = headers[name]
        if array.shape != shape or array.dtype != dtype or array.nbytes != nbytes:
            raise ReferenceProtocolError(f"Decoded NPZ array {name!r} differs from its validated NPY header")
    return validate_array_mapping(arrays) if arrays else {}


def _safe_parameters(parameters: Mapping[str, Any] | None) -> dict[str, Any]:
    if parameters is None:
        return {}
    if not isinstance(parameters, Mapping):
        raise ReferenceProtocolError("Operation parameters must be a JSON object")
    result = dict(parameters)
    raw = _json_bytes(result)
    if len(raw) > MAX_PARAMETERS_BYTES:
        raise ReferenceProtocolError("Operation parameters exceed the protocol size limit")
    return result


class _BoundedLogReader(threading.Thread):
    def __init__(self, stream: Any) -> None:
        super().__init__(daemon=True)
        self.stream = stream
        self.data = bytearray()

    def run(self) -> None:
        try:
            while True:
                block = self.stream.read(8192)
                if not block:
                    return
                remaining = MAX_CAPTURED_LOG_BYTES - len(self.data)
                if remaining > 0:
                    self.data.extend(block[:remaining])
        except OSError:
            # A stream can be closed by the supervising thread during cleanup.
            pass


def _cancelled(cancel: Any) -> bool:
    if cancel is None:
        return False
    if callable(cancel):
        return bool(cancel())
    is_set = getattr(cancel, "is_set", None)
    if callable(is_set):
        return bool(is_set())
    raise TypeError("cancel_event must be a callable or expose is_set()")


def _run_child(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    timeout_seconds: float,
    cancel_event: Any,
) -> tuple[int, bytes, bytes]:
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)):
        raise ValueError("timeout_seconds must be a positive finite number")
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be a positive finite number")
    if cancel_event is not None and not callable(cancel_event) and not callable(getattr(cancel_event, "is_set", None)):
        raise TypeError("cancel_event must be a callable or expose is_set()")
    try:
        child = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
        )
    except OSError as exc:
        raise ReferenceRunnerError(f"Could not start the isolated PlantCV worker: {exc}") from exc
    assert child.stdout is not None and child.stderr is not None
    stdout_reader = _BoundedLogReader(child.stdout)
    stderr_reader = _BoundedLogReader(child.stderr)
    started = time.monotonic()
    reason: str | None = None
    try:
        stdout_reader.start()
        stderr_reader.start()
        while child.poll() is None:
            if _cancelled(cancel_event):
                reason = "cancelled"
                try:
                    child.kill()
                except OSError:
                    if child.poll() is None:
                        raise
                break
            if time.monotonic() - started >= timeout_seconds:
                reason = "timeout"
                try:
                    child.kill()
                except OSError:
                    if child.poll() is None:
                        raise
                break
            time.sleep(min(0.025, timeout_seconds / 20.0))
    finally:
        # Cancellation callbacks are user code and may raise. Always stop and
        # reap the external interpreter and release all pipe handles on every
        # exit path, including BaseException from the callback.
        try:
            if child.poll() is None:
                try:
                    child.kill()
                except OSError:
                    if child.poll() is None:
                        raise
        finally:
            child.wait()
            if stdout_reader.ident is not None:
                stdout_reader.join()
            if stderr_reader.ident is not None:
                stderr_reader.join()
            for stream in (child.stdout, child.stderr):
                try:
                    stream.close()
                except OSError:
                    pass
    if reason == "cancelled":
        raise ReferenceCancelledError("PlantCV reference execution was cancelled")
    if reason == "timeout":
        raise ReferenceTimeoutError(f"PlantCV reference execution exceeded {timeout_seconds:g} seconds")
    return int(child.returncode or 0), bytes(stdout_reader.data), bytes(stderr_reader.data)


def _default_paths() -> tuple[Path, Path, Path, Path]:
    root = _repo_root()
    return (
        root / "artifacts" / "reference-sources" / "plantcv-source-audit.json",
        root / "artifacts" / "reference-sources" / "plantcv-v4.11.3",
        root / "artifacts" / "reference-sources" / "plantcv-environment.json",
        root / "config" / "plantcv-reference-requirements.txt",
    )


def plantcv_reference_status(
    *,
    python_executable: str | os.PathLike[str] | None,
    source_manifest_path: str | os.PathLike[str] | None = None,
    source_root: str | os.PathLike[str] | None = None,
    environment_manifest_path: str | os.PathLike[str] | None = None,
    requirements_lock_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Return a non-throwing availability record for original PlantCV."""
    defaults = _default_paths()
    executable = Path(python_executable).expanduser() if python_executable else None
    paths = (
        Path(source_manifest_path) if source_manifest_path else defaults[0],
        Path(source_root) if source_root else defaults[1],
        Path(environment_manifest_path) if environment_manifest_path else defaults[2],
        Path(requirements_lock_path) if requirements_lock_path else defaults[3],
    )
    if executable is None:
        return {
            "backend": "plantcv_reference", "available": False, "state": "unavailable",
            "reason": "No isolated reference Python was supplied.",
        }
    if not executable.is_file():
        return {
            "backend": "plantcv_reference", "available": False, "state": "unavailable",
            "reason": f"Reference Python does not exist: {executable}",
        }
    try:
        identity = _build_expected_identity(
            source_manifest_path=paths[0],
            source_root=paths[1],
            environment_manifest_path=paths[2],
            requirements_lock_path=paths[3],
        )
    except ReferenceRunnerError as exc:
        return {"backend": "plantcv_reference", "available": False, "state": "unavailable", "reason": str(exc)}
    probe = r"""
import hashlib, importlib.metadata as metadata, importlib.util, json, os, platform, re, sys
from pathlib import Path
normalize = lambda value: re.sub(r"[-_.]+", "-", value).lower()
payload = json.loads(sys.argv[1])
distributions = {}
for distribution in metadata.distributions():
    name = distribution.metadata.get("Name")
    if not name:
        continue
    normalized = normalize(name)
    if normalized in {"pip", "setuptools", "wheel"}:
        continue
    version = distribution.version
    if normalized in distributions and distributions[normalized] != version:
        raise RuntimeError("multiple versions of " + normalized)
    distributions[normalized] = version
root = Path(metadata.distribution("plantcv").locate_file("")).resolve()
spec = importlib.util.find_spec("plantcv")
outer = (root / "plantcv" / "__init__.py").resolve()
if spec is None or spec.origin is None or Path(spec.origin).resolve() != outer:
    raise RuntimeError("PlantCV import resolves outside its installed distribution")
from plantcv import plantcv as pcv
inner = (root / "plantcv" / "plantcv" / "__init__.py").resolve()
if Path(pcv.__file__).resolve() != inner:
    raise RuntimeError("PlantCV module resolves outside its installed distribution")
for record in payload["source_files"]:
    if not record["path"].startswith("plantcv/plantcv/"):
        continue
    source = (root / record["path"]).resolve()
    content = source.read_bytes()
    if len(content) != record["bytes"] or hashlib.sha256(content).hexdigest() != record["sha256"]:
        raise RuntimeError("audited source mismatch: " + record["path"])
machine = platform.machine().lower()
platform_name = "Windows-x86_64" if os.name == "nt" and machine in {"amd64", "x86_64", "x64"} else platform.system() + "-" + machine
print(json.dumps({"python": sys.version.split()[0], "platform": platform_name, "distributions": distributions}, allow_nan=False))
"""
    probe_payload = json.dumps({"source_files": identity["source_files"]}, separators=(",", ":"))
    try:
        with tempfile.TemporaryDirectory(prefix="seedfiddle-plantcv-probe-") as temp_name:
            env = os.environ.copy()
            env.pop("PYTHONHOME", None)
            env["PYTHONPATH"] = ""
            env["PYTHONNOUSERSITE"] = "1"
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            return_code, stdout, stderr = _run_child(
                [str(executable.resolve()), "-c", probe, probe_payload],
                cwd=Path(temp_name), env=env, timeout_seconds=10.0, cancel_event=None,
            )
        if return_code != 0:
            raise ReferenceIdentityError(
                "Reference interpreter identity probe failed: "
                + stderr.decode("utf-8", errors="replace")[:2048]
            )
        observed = _strict_json_loads(stdout.decode("utf-8"))
        if not isinstance(observed, dict) or not isinstance(observed.get("distributions"), dict):
            raise ReferenceIdentityError("Reference interpreter returned an invalid identity probe")
        versions = {
            _canonical_name(name): version
            for name, version in observed.get("distributions", {}).items()
            if isinstance(name, str) and isinstance(version, str)
        }
        if (
            observed.get("python") != PINNED_REFERENCE_PYTHON
            or observed.get("platform") != PINNED_REFERENCE_PLATFORM
            or versions != identity["expected_distributions"]
        ):
            raise ReferenceIdentityError("Reference interpreter does not match the frozen Python/platform/dependency inventory")
    except (ReferenceRunnerError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        return {
            "backend": "plantcv_reference", "available": False, "state": "unavailable",
            "python_executable": str(executable.resolve()), "reason": str(exc),
        }
    return {
        "backend": "plantcv_reference", "available": True, "state": "environment_and_source_verified",
        "version": PINNED_PLANTCV_VERSION, "commit": PINNED_PLANTCV_COMMIT,
        "python_executable": str(executable.resolve()), "identity_digest": identity["identity_digest"],
    }


def smpts_reference_status() -> dict[str, Any]:
    """Report original SMPTS as unavailable until its code is recovered."""
    return {
        "backend": "smpts_reference",
        "available": False,
        "state": "unavailable",
        "reason": "The original SMPTS implementation has not been recovered or executed; no original-code adapter is available.",
    }


def _validate_watershed_capture(
    arrays: dict[str, np.ndarray], declarations: list[Mapping[str, Any]],
    effective: Mapping[str, Any], request_id: str, inputs: Mapping[str, np.ndarray],
) -> None:
    """Validate this operation's logical artifacts without relaxing NPZ rules."""
    from .reference_watershed_capture import EMPTY_COORDINATES, SOURCE_LOCALS

    capture = effective.get("capture")
    if (
        not isinstance(capture, dict)
        or capture.get("schema") != "seedfiddle.public-watershed-capture.v1"
        or capture.get("call_id") != request_id
        or capture.get("scientifically_validated") is not False
    ):
        raise ReferenceProtocolError("Watershed capture metadata is invalid")
    mask = inputs.get("mask")
    labels = arrays.get("labels")
    if (
        mask is None or labels is None or labels.dtype != np.int32
        or labels.shape != mask.shape or np.any(labels < 0) or np.any(labels[mask == 0] != 0)
    ):
        raise ReferenceProtocolError("Watershed labels differ from their input frame/support")
    if capture.get("status") == "unavailable":
        if set(arrays) != {"labels"} or "artifacts" in capture or "empty_coordinates" in capture:
            raise ReferenceProtocolError("Unavailable capture must retain only public labels")
        if not isinstance(capture.get("reason"), str) or not capture["reason"]:
            raise ReferenceProtocolError("Unavailable capture omitted its reason")
        return
    if capture.get("status") != "available":
        raise ReferenceProtocolError("Unknown watershed capture status")
    expected = set(SOURCE_LOCALS) | {"diagnostic"}
    empty = capture.get("empty_coordinates")
    if empty is not None:
        if _json_bytes(empty) != _json_bytes(EMPTY_COORDINATES) or "coordinates" in arrays:
            raise ReferenceProtocolError("Invalid empty-coordinate capture transport")
        if set(arrays) != expected - {"coordinates"} or len(arrays) + 1 > MAX_ARRAYS:
            raise ReferenceProtocolError("Empty-coordinate logical artifact inventory differs")
        arrays["coordinates"] = np.empty((0, 2), dtype=np.dtype("<i8"))
        declarations.append({"name": "coordinates", "shape": [0, 2], "dtype": "<i8", "nbytes": 0})
    elif set(arrays) != expected:
        raise ReferenceProtocolError("Watershed capture artifact inventory differs")
    if len(arrays) > MAX_ARRAYS or sum(a.nbytes for a in arrays.values()) > MAX_ARRAY_BYTES or sum(a.size for a in arrays.values()) > MAX_ARRAY_ELEMENTS:
        raise ReferenceProtocolError("Watershed capture logical artifacts exceed their aggregate limit")
    shape = mask.shape
    specs = {
        "distance": (np.dtype(np.float32), shape), "negative_distance": (np.dtype(np.float32), shape),
        "peaks": (np.dtype(np.bool_), shape), "markers": (np.dtype(np.int32), shape),
        "diagnostic_bgr": (np.dtype(np.uint8), (*shape, 3)), "diagnostic": (np.dtype(np.uint8), (*shape, 3)),
    }
    for name, (dtype, expected_shape) in specs.items():
        if arrays[name].dtype != dtype or arrays[name].shape != expected_shape:
            raise ReferenceProtocolError(f"Watershed capture {name} has invalid dtype/frame")
    coords, palette = arrays["coordinates"], arrays["palette"]
    if coords.dtype != np.dtype("<i8") or coords.ndim != 2 or coords.shape[1] != 2 or coords.shape[0] > mask.size:
        raise ReferenceProtocolError("Watershed coordinates have invalid dtype/shape")
    if palette.dtype != np.uint8 or palette.ndim != 2 or palette.shape[1] != 3 or not 1 <= palette.shape[0] <= mask.size:
        raise ReferenceProtocolError("Watershed palette has invalid dtype/shape")
    artifacts = capture.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != expected:
        raise ReferenceProtocolError("Watershed capture omitted artifact provenance")
    for name, array in arrays.items():
        artifact = artifacts[name]
        if not isinstance(artifact, dict):
            raise ReferenceProtocolError("Watershed artifact provenance is invalid")
        actual = {"shape": list(array.shape), "dtype": array.dtype.str, "nbytes": int(array.nbytes), "sha256": _sha256(array.tobytes())}
        if any(_json_bytes(artifact.get(key)) != _json_bytes(value) for key, value in actual.items()):
            raise ReferenceProtocolError(f"Watershed capture artifact identity differs: {name}")
        if artifact.get("source_local") != SOURCE_LOCALS.get(name, "plt_img"):
            raise ReferenceProtocolError(f"Watershed capture source-local identity differs: {name}")
    if artifacts["diagnostic"].get("adapter") != "BGR to RGB channel reversal" or not np.array_equal(arrays["diagnostic"], arrays["diagnostic_bgr"][..., ::-1]):
        raise ReferenceProtocolError("Watershed diagnostic RGB adapter differs")
    if not np.isfinite(arrays["distance"]).all() or np.any(arrays["distance"] < 0) or not np.array_equal(arrays["negative_distance"], -arrays["distance"]):
        raise ReferenceProtocolError("Watershed distance relation differs")
    if coords.size and (np.any(coords < 0) or np.any(coords[:, 0] >= shape[0]) or np.any(coords[:, 1] >= shape[1]) or len(np.unique(coords, axis=0)) != len(coords)):
        raise ReferenceProtocolError("Watershed coordinates leave their frame or duplicate peaks")
    if np.count_nonzero(arrays["peaks"]) != len(coords) or (coords.size and not np.all(arrays["peaks"][tuple(coords.T)])):
        raise ReferenceProtocolError("Watershed peak support differs from coordinates")
    markers = arrays["markers"]
    marker_ids, label_ids = np.unique(markers), np.unique(labels)
    if np.any(markers < 0) or not np.array_equal(markers > 0, arrays["peaks"]) or not np.isin(label_ids[label_ids > 0], marker_ids).all():
        raise ReferenceProtocolError("Watershed marker/label relation differs")
    if (
        type(capture.get("estimated_object_count")) is not int
        or capture["estimated_object_count"] != len(marker_ids) - 1
        or _json_bytes(capture.get("positive_marker_ids")) != _json_bytes(marker_ids[marker_ids > 0].tolist())
        or _json_bytes(capture.get("positive_label_ids")) != _json_bytes(label_ids[label_ids > 0].tolist())
        or type(capture.get("label_max")) is not int or capture["label_max"] != int(labels.max())
        or len(palette) != len(label_ids) or int(labels.max()) >= len(palette)
    ):
        raise ReferenceProtocolError("Watershed capture count/ID/palette facts differ")


class PlantCVReferenceRunner:
    """Run an audited PlantCV operation in an explicit reference interpreter."""

    def __init__(
        self,
        python_executable: str | os.PathLike[str],
        *,
        source_manifest_path: str | os.PathLike[str] | None = None,
        source_root: str | os.PathLike[str] | None = None,
        environment_manifest_path: str | os.PathLike[str] | None = None,
        requirements_lock_path: str | os.PathLike[str] | None = None,
        timeout_seconds: float = 120.0,
        temporary_root: str | os.PathLike[str] | None = None,
    ) -> None:
        defaults = _default_paths()
        self.python_executable = Path(python_executable).expanduser().resolve()
        self.source_manifest_path = Path(source_manifest_path).resolve() if source_manifest_path else defaults[0].resolve()
        self.source_root = Path(source_root).resolve() if source_root else defaults[1].resolve()
        self.environment_manifest_path = (
            Path(environment_manifest_path).resolve() if environment_manifest_path else defaults[2].resolve()
        )
        self.requirements_lock_path = (
            Path(requirements_lock_path).resolve() if requirements_lock_path else defaults[3].resolve()
        )
        self.timeout_seconds = timeout_seconds
        self.temporary_root = Path(temporary_root).expanduser().resolve() if temporary_root else None

    def run(
        self,
        operation: str,
        arrays: Mapping[str, Any],
        parameters: Mapping[str, Any] | None = None,
        *,
        timeout_seconds: float | None = None,
        cancel_event: Any = None,
    ) -> ReferenceResult:
        """Execute one whitelisted callable and return checked raster/JSON results.

        ``arrays`` contains only bounded numeric NumPy arrays. The selected
        operation and parameter names are checked again in the isolated worker.
        """
        if not isinstance(operation, str) or not operation:
            raise ReferenceProtocolError("A PlantCV operation name is required")
        if operation not in SUPPORTED_PLANTCV_OPERATIONS:
            raise ReferenceProtocolError(f"PlantCV operation is not in the audited callable allowlist: {operation}")
        if not self.python_executable.is_file():
            raise ReferenceIdentityError(f"Reference Python does not exist: {self.python_executable}")
        timeout = self.timeout_seconds if timeout_seconds is None else timeout_seconds
        identity = _build_expected_identity(
            source_manifest_path=self.source_manifest_path,
            source_root=self.source_root,
            environment_manifest_path=self.environment_manifest_path,
            requirements_lock_path=self.requirements_lock_path,
        )
        input_arrays = validate_array_mapping(
            arrays,
            max_aggregate_bytes=MAX_INPUT_ARCHIVE_BYTES - (1024 * 1024),
        )
        params = _safe_parameters(parameters)
        if len(_json_bytes(params)) > MAX_PARAMETERS_BYTES:
            raise ReferenceProtocolError("Operation parameters exceed the protocol size limit")
        request_id = uuid.uuid4().hex
        request = {
            "protocol_version": PROTOCOL_VERSION,
            "request_id": request_id,
            "operation": operation,
            "parameters": params,
            "identity": identity,
            "input_arrays": {
                name: {"shape": list(array.shape), "dtype": array.dtype.str, "nbytes": int(array.nbytes)}
                for name, array in input_arrays.items()
            },
        }
        request_bytes = _json_bytes(request)
        if len(request_bytes) > MAX_JSON_BYTES:
            raise ReferenceProtocolError("Reference request exceeds the JSON size limit")
        repo_root = _repo_root()
        worker_module = "seedvision.benchmarking.plantcv_worker"
        if self.temporary_root is not None and not self.temporary_root.is_dir():
            raise ReferenceRunnerError(f"Reference temporary root is not an existing directory: {self.temporary_root}")
        with tempfile.TemporaryDirectory(prefix="seedfiddle-plantcv-", dir=self.temporary_root) as temp_name:
            work_dir = Path(temp_name)
            request_path = work_dir / "request.json"
            input_path = work_dir / "inputs.npz"
            response_path = work_dir / "response.json"
            output_path = work_dir / "outputs.npz"
            request_path.write_bytes(request_bytes)
            try:
                np.savez_compressed(input_path, **input_arrays)
            except Exception as exc:
                raise ReferenceProtocolError(f"Could not encode input arrays: {exc}") from exc
            if input_path.stat().st_size > MAX_INPUT_ARCHIVE_BYTES:
                raise ReferenceProtocolError("Input NPZ archive exceeds the protocol size limit")
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(repo_root)
            environment.pop("PYTHONHOME", None)
            environment["PYTHONNOUSERSITE"] = "1"
            environment["PYTHONDONTWRITEBYTECODE"] = "1"
            environment["MPLCONFIGDIR"] = str(work_dir / "matplotlib")
            environment["MPLBACKEND"] = "Agg"
            environment["TMP"] = str(work_dir)
            environment["TEMP"] = str(work_dir)
            (work_dir / "matplotlib").mkdir()
            command = [
                str(self.python_executable),
                "-m",
                worker_module,
                "--request",
                str(request_path),
                "--inputs",
                str(input_path),
                "--response",
                str(response_path),
                "--outputs",
                str(output_path),
            ]
            wall_started = time.perf_counter_ns()
            return_code, stdout, stderr = _run_child(
                command,
                cwd=work_dir,
                env=environment,
                timeout_seconds=timeout,
                cancel_event=cancel_event,
            )
            wall_runtime_ns = time.perf_counter_ns() - wall_started
            if not response_path.is_file():
                detail = stderr.decode("utf-8", errors="replace").strip()
                if not detail:
                    detail = stdout.decode("utf-8", errors="replace").strip()
                raise ReferenceExecutionError(
                    f"PlantCV worker exited {return_code} without a response" + (f": {detail}" if detail else "")
                )
            response_bytes = _read_file_bounded(
                response_path, MAX_JSON_BYTES, description="PlantCV worker response JSON"
            )
            try:
                response = _strict_json_loads(response_bytes.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                raise ReferenceProtocolError(f"PlantCV worker returned invalid JSON: {exc}") from exc
            if not isinstance(response, dict) or response.get("protocol_version") != PROTOCOL_VERSION:
                raise ReferenceProtocolError("PlantCV worker response has an unsupported protocol version")
            if response.get("request_id") != request_id:
                raise ReferenceProtocolError("PlantCV worker response request id does not match")
            if return_code != 0 or response.get("ok") is not True:
                message = response.get("error", "PlantCV worker failed")
                if not isinstance(message, str):
                    message = "PlantCV worker failed with a non-text error"
                raise ReferenceExecutionError(message)
            if response.get("operation") != operation:
                raise ReferenceProtocolError("PlantCV worker response operation does not match the request")
            response_identity = response.get("identity")
            if not isinstance(response_identity, dict) or response_identity.get("identity_digest") != identity["identity_digest"]:
                raise ReferenceIdentityError("Worker identity does not match the verified source and environment manifest")
            if response_identity.get("actual_distributions") != identity["expected_distributions"]:
                raise ReferenceIdentityError("Worker reports a different installed reference dependency inventory")
            if (
                response_identity.get("reference") != "plantcv"
                or response_identity.get("version") != PINNED_PLANTCV_VERSION
                or response_identity.get("commit") != PINNED_PLANTCV_COMMIT
                or response_identity.get("python") != PINNED_REFERENCE_PYTHON
                or response_identity.get("platform") != PINNED_REFERENCE_PLATFORM
            ):
                raise ReferenceIdentityError("Worker runtime identity does not match the pinned reference release")
            if operation == "segmentation.watershed.capture":
                from .reference_watershed_capture import capture_source_records
                if response_identity.get("capture_source_files") != capture_source_records():
                    raise ReferenceIdentityError("Worker capture source identities differ from their pins")
            declared_arrays = response.get("intermediates")
            if not isinstance(declared_arrays, list):
                raise ReferenceProtocolError("PlantCV worker returned invalid intermediate declarations")
            allow_observation_only = operation == "analysis.color"
            if not declared_arrays and not allow_observation_only:
                raise ReferenceProtocolError("PlantCV worker returned no declared intermediates")
            if not output_path.is_file():
                raise ReferenceProtocolError("PlantCV worker omitted its NPZ outputs")
            result_arrays = _read_npz(
                output_path,
                max_archive_bytes=MAX_OUTPUT_ARCHIVE_BYTES,
                allow_empty=allow_observation_only,
            )
            declared: dict[str, Mapping[str, Any]] = {}
            for entry in declared_arrays:
                if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
                    raise ReferenceProtocolError("PlantCV worker returned an invalid intermediate declaration")
                name = entry["name"]
                if name in declared:
                    raise ReferenceProtocolError(f"PlantCV worker declared duplicate intermediate {name!r}")
                declared[name] = entry
            if set(declared) != set(result_arrays):
                raise ReferenceProtocolError("PlantCV NPZ arrays differ from the declared intermediates")
            checked_intermediates: list[Mapping[str, Any]] = []
            for name, array in result_arrays.items():
                entry = declared[name]
                if entry.get("shape") != list(array.shape) or entry.get("dtype") != array.dtype.str:
                    raise ReferenceProtocolError(f"PlantCV output metadata does not match array {name!r}")
                if entry.get("nbytes") != array.nbytes:
                    raise ReferenceProtocolError(f"PlantCV output byte count does not match array {name!r}")
                if array.dtype.kind not in _ALLOWED_DTYPES or array.ndim > MAX_ARRAY_DIMENSIONS:
                    raise ReferenceProtocolError(f"PlantCV output array {name!r} has unsupported data")
                checked_intermediates.append(dict(entry))
            effective = response.get("effective_parameters")
            observations = response.get("observations")
            visualizations = response.get("visualizations", {})
            channel_semantics = response.get("channel_semantics", {})
            runtime_ns = response.get("computation_runtime_ns")
            if not isinstance(effective, dict) or not isinstance(observations, dict):
                raise ReferenceProtocolError("PlantCV worker omitted effective parameters or observations")
            if operation == "segmentation.watershed.capture":
                _validate_watershed_capture(result_arrays, checked_intermediates, effective, request_id, input_arrays)
            if not isinstance(visualizations, dict):
                raise ReferenceProtocolError("PlantCV worker returned invalid visualizations")
            if not isinstance(channel_semantics, dict):
                raise ReferenceProtocolError("PlantCV worker returned invalid channel semantics")
            required_channel_fields = {
                "segmentation.watershed": {"input_boundary", "plantcv_input", "raster_output"},
                "segmentation.watershed.capture": {"input_boundary", "plantcv_input", "raster_output", "diagnostic", "diagnostic_bgr", "palette", "coordinates"},
                "analysis.color": {"input_boundary", "plantcv_input", "observations", "visualizations"},
                "transform.correct_color": {"input_boundary", "plantcv_input", "corrected_image", "color_matrices"},
                "transform.auto_correct_color": {"input_boundary", "plantcv_input", "corrected_image", "observations"},
            }.get(operation, set())
            if not required_channel_fields.issubset(channel_semantics):
                raise ReferenceProtocolError("PlantCV worker omitted required input/output channel semantics")
            if operation == "analysis.size" and not {"input_boundary", "plantcv_input", "raster_output"}.issubset(channel_semantics):
                raise ReferenceProtocolError("PlantCV worker omitted required size-analysis channel semantics")
            if isinstance(runtime_ns, bool) or not isinstance(runtime_ns, int) or runtime_ns < 0:
                raise ReferenceProtocolError("PlantCV worker returned invalid runtime metadata")
            if operation == "segmentation.watershed.capture" and _cancelled(cancel_event):
                raise ReferenceCancelledError("PlantCV capture cancelled before publication")
            return ReferenceResult(
                operation=operation,
                arrays=result_arrays,
                observations=observations,
                effective_parameters=effective,
                intermediates=tuple(checked_intermediates),
                visualizations=visualizations,
                channel_semantics=channel_semantics,
                identity=response_identity,
                computation_runtime_ns=runtime_ns,
                wall_runtime_ns=wall_runtime_ns,
            )


__all__ = [
    "PlantCVReferenceRunner",
    "ReferenceCancelledError",
    "ReferenceExecutionError",
    "ReferenceIdentityError",
    "ReferenceProtocolError",
    "ReferenceResult",
    "ReferenceRunnerError",
    "ReferenceTimeoutError",
    "SUPPORTED_PLANTCV_OPERATIONS",
    "plantcv_reference_status",
    "smpts_reference_status",
    "validate_array_mapping",
]
