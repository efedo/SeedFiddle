"""Reconstruct the pinned PlantCV runner's ignored source metadata safely.

This module only reads the committed source registry and binary requirements
lock. It performs no network or environment activity at import time. Missing
source files are reported as unavailable unless a caller explicitly supplies a
fetch callback to :func:`materialize_reference_setup` with ``fetch_sources``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import tempfile
from typing import Any, Callable, Iterable, Mapping
from urllib.parse import urlsplit

from .reference_runner import (
    PINNED_PLANTCV_COMMIT,
    PINNED_PLANTCV_VERSION,
    PINNED_REFERENCE_PLATFORM,
    PINNED_REFERENCE_PYTHON,
    PINNED_REQUIREMENTS_SHA256,
    PINNED_SOURCE_AUDIT_SHA256,
    _canonical_name,
)


MAX_CONFIG_BYTES = 2 * 1024 * 1024
MAX_REQUIREMENTS_BYTES = 1024 * 1024
MAX_SOURCE_FILES = 15
MAX_SOURCE_BYTES = 8 * 1024 * 1024
MAX_TOTAL_SOURCE_BYTES = 32 * 1024 * 1024
MAX_PACKAGES = 67
MAX_FETCH_CHUNKS = 16_384
MAX_EMPTY_FETCH_CHUNKS = 32
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_LOCK_LINE_RE = re.compile(r"([^=\s]+)==([^\s]+)\s+--hash=sha256:([0-9a-f]{64})$")


class ReferenceSetupError(RuntimeError):
    """Raised when committed reference setup evidence fails its pins."""


@dataclass(frozen=True, slots=True)
class _SourceRecord:
    path: str
    url: str
    sha256: str
    byte_count: int


@dataclass(frozen=True, slots=True)
class ReferenceSetupStatus:
    """Read-only status for the source and wheel prerequisites."""

    target_dir: Path
    source_root: Path
    source_audit_path: Path
    environment_manifest_path: Path
    source_audit_sha256: str
    environment_manifest_sha256: str
    requirements_lock_sha256: str
    source_file_count: int
    package_count: int
    unavailable_sources: tuple[str, ...]
    wheel_path: Path
    wheel_status: str
    metadata_available: bool

    @property
    def sources_available(self) -> bool:
        return not self.unavailable_sources

    @property
    def retained_wheel_available(self) -> bool:
        return self.wheel_status == "verified"

    @property
    def provisioning_prerequisites_available(self) -> bool:
        """Whether all source records and the retained flyr wheel are present."""
        return self.sources_available and self.retained_wheel_available and self.metadata_available


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _strict_json(raw: bytes, label: str) -> Any:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=unique_object,
                          parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ReferenceSetupError(f"Invalid {label}: {exc}") from exc


def _read_bounded(path: Path, maximum: int, label: str) -> bytes:
    try:
        with path.open("rb") as handle:
            raw = handle.read(maximum + 1)
    except OSError as exc:
        raise ReferenceSetupError(f"Cannot read {label} {path}: {exc}") from exc
    if len(raw) > maximum:
        raise ReferenceSetupError(f"{label} exceeds the {maximum}-byte limit")
    return raw


def _pretty_crlf_json(value: Any) -> bytes:
    try:
        text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    except (TypeError, ValueError) as exc:
        raise ReferenceSetupError(f"Reference metadata is not strict JSON: {exc}") from exc
    return text.replace("\n", "\r\n").encode("utf-8")


def _safe_relative_path(value: Any, *, label: str) -> tuple[str, ...]:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise ReferenceSetupError(f"{label} must be a non-empty relative POSIX path")
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if posix.is_absolute() or windows.is_absolute() or windows.drive:
        raise ReferenceSetupError(f"{label} must be relative: {value!r}")
    parts = tuple(value.split("/"))
    if any(part in {"", ".", ".."} for part in parts):
        raise ReferenceSetupError(f"{label} contains an unsafe path component: {value!r}")
    return parts


def _safe_child(root: Path, relative: str, *, label: str) -> Path:
    parts = _safe_relative_path(relative, label=label)
    root_real = root.resolve(strict=False)
    candidate = root_real.joinpath(*parts)
    current = root_real
    for part in parts:
        current = current / part
        if current.is_symlink():
            raise ReferenceSetupError(f"{label} traverses a symbolic link: {relative!r}")
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(root_real)
    except ValueError as exc:
        raise ReferenceSetupError(f"{label} escapes its target directory: {relative!r}") from exc
    return candidate


def _load_source_registry(config_path: Path) -> tuple[dict[str, Any], bytes]:
    raw = _read_bounded(config_path, MAX_CONFIG_BYTES, "peer reference registry")
    config = _strict_json(raw, "peer reference registry")
    if not isinstance(config, dict):
        raise ReferenceSetupError("Peer reference registry must be a JSON object")
    return config, raw


def _validate_source_records(config: Mapping[str, Any]) -> tuple[_SourceRecord, ...]:
    if (config.get("reference"), config.get("version"), config.get("commit")) != (
        "plantcv", PINNED_PLANTCV_VERSION, PINNED_PLANTCV_COMMIT,
    ):
        raise ReferenceSetupError("Peer registry does not match the pinned PlantCV release and commit")
    raw_files = config.get("files")
    if not isinstance(raw_files, list) or len(raw_files) != MAX_SOURCE_FILES:
        raise ReferenceSetupError(f"Pinned PlantCV source registry must contain exactly {MAX_SOURCE_FILES} files")
    records: list[_SourceRecord] = []
    seen_paths: set[str] = set()
    seen_urls: set[str] = set()
    total_bytes = 0
    expected_prefix = f"https://raw.githubusercontent.com/danforthcenter/plantcv/{PINNED_PLANTCV_COMMIT}/"
    for raw in raw_files:
        if not isinstance(raw, dict):
            raise ReferenceSetupError("Pinned PlantCV source registry contains a non-object file record")
        path = raw.get("path")
        _safe_relative_path(path, label="source path")
        if path in seen_paths:
            raise ReferenceSetupError(f"Duplicate pinned source path: {path}")
        seen_paths.add(path)
        url = raw.get("url")
        if not isinstance(url, str) or not url.startswith(expected_prefix):
            raise ReferenceSetupError(f"Source URL is not pinned to the audited raw GitHub commit: {path}")
        parsed = urlsplit(url)
        if parsed.scheme != "https" or parsed.netloc != "raw.githubusercontent.com" or parsed.query or parsed.fragment:
            raise ReferenceSetupError(f"Source URL is not an exact pinned HTTPS raw URL: {path}")
        if url != expected_prefix + path:
            raise ReferenceSetupError(f"Source URL path does not match its pinned source path: {path}")
        if url in seen_urls:
            raise ReferenceSetupError(f"Duplicate pinned source URL: {url}")
        seen_urls.add(url)
        digest = raw.get("sha256")
        if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
            raise ReferenceSetupError(f"Invalid pinned SHA-256 for source {path}")
        byte_count = raw.get("bytes")
        if isinstance(byte_count, bool) or not isinstance(byte_count, int) or not 0 <= byte_count <= MAX_SOURCE_BYTES:
            raise ReferenceSetupError(f"Invalid or oversized byte count for source {path}")
        total_bytes += byte_count
        if total_bytes > MAX_TOTAL_SOURCE_BYTES:
            raise ReferenceSetupError("Pinned source inventory exceeds the aggregate byte limit")
        records.append(_SourceRecord(path, url, digest, byte_count))
    return tuple(records)


def _source_audit_projection(config: Mapping[str, Any], records: tuple[_SourceRecord, ...]) -> dict[str, Any]:
    return {
        "reference": "plantcv",
        "version": PINNED_PLANTCV_VERSION,
        "commit": PINNED_PLANTCV_COMMIT,
        "retrieved_date": config.get("retrieved_date"),
        "files": [
            {"path": item.path, "url": item.url, "sha256": item.sha256, "bytes": item.byte_count}
            for item in records
        ],
    }


def _parse_locked_packages(lock_raw: bytes) -> dict[str, tuple[str, str, str]]:
    if _sha256(lock_raw) != PINNED_REQUIREMENTS_SHA256:
        raise ReferenceSetupError("Pinned PlantCV requirements lock has changed")
    result: dict[str, tuple[str, str, str]] = {}
    try:
        lines = lock_raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ReferenceSetupError("Pinned PlantCV requirements lock is not UTF-8") from exc
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = _LOCK_LINE_RE.fullmatch(line)
        if not match:
            raise ReferenceSetupError(f"Invalid hash-locked package line: {line}")
        name, version, digest = match.groups()
        key = _canonical_name(name)
        if key in result:
            raise ReferenceSetupError(f"Duplicate normalized package name in requirements lock: {name}")
        result[key] = (name, version, digest)
    if len(result) != MAX_PACKAGES:
        raise ReferenceSetupError(f"PlantCV requirements lock must contain exactly {MAX_PACKAGES} packages")
    if result.get("plantcv", (None, None, None))[1] != PINNED_PLANTCV_VERSION:
        raise ReferenceSetupError("Requirements lock does not pin the audited PlantCV release")
    return result


def _environment_inventory(config: Mapping[str, Any], lock_raw: bytes) -> dict[str, str]:
    environment = config.get("environment")
    if not isinstance(environment, dict):
        raise ReferenceSetupError("Peer registry has no frozen environment record")
    if environment.get("python") != PINNED_REFERENCE_PYTHON or environment.get("platform") != PINNED_REFERENCE_PLATFORM:
        raise ReferenceSetupError("Peer registry environment does not match the pinned Python and platform")
    if environment.get("lockfile_sha256") != PINNED_REQUIREMENTS_SHA256:
        raise ReferenceSetupError("Peer registry requirements-lock hash differs from the runner pin")
    packages = environment.get("packages")
    if not isinstance(packages, list) or len(packages) != MAX_PACKAGES:
        raise ReferenceSetupError(f"Peer registry must inventory exactly {MAX_PACKAGES} packages")
    locked = _parse_locked_packages(lock_raw)
    inventory: dict[str, str] = {}
    seen: set[str] = set()
    for item in packages:
        if not isinstance(item, dict):
            raise ReferenceSetupError("Peer environment package inventory contains a non-object entry")
        name, version, digest = item.get("name"), item.get("version"), item.get("archive_sha256")
        if not isinstance(name, str) or not name or not isinstance(version, str) or not version:
            raise ReferenceSetupError("Peer environment package name and version must be non-empty strings")
        if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
            raise ReferenceSetupError(f"Invalid archive SHA-256 for environment package {name}")
        key = _canonical_name(name)
        if key in seen:
            raise ReferenceSetupError(f"Duplicate normalized package in peer environment inventory: {name}")
        seen.add(key)
        locked_record = locked.get(key)
        if locked_record is None or locked_record[1:] != (version, digest):
            raise ReferenceSetupError(f"Peer environment package differs from exact lock record: {name}")
        inventory[name] = version
    if seen != set(locked):
        missing = sorted(set(locked) - seen)
        extra = sorted(seen - set(locked))
        raise ReferenceSetupError(f"Peer environment package inventory differs from lock (missing={missing}, extra={extra})")
    if environment.get("lockfile") != "config/plantcv-reference-requirements.txt":
        raise ReferenceSetupError("Peer registry points to an unexpected requirements-lock path")
    return inventory


def _retained_flyr(config: Mapping[str, Any], locked: Mapping[str, tuple[str, str, str]]) -> tuple[str, int, str]:
    environment = config["environment"]
    packages = environment["packages"]
    flyr = [item for item in packages if isinstance(item, dict) and item.get("name") == "flyr"]
    if len(flyr) != 1 or not isinstance(flyr[0].get("retained_artifact"), dict):
        raise ReferenceSetupError("Peer registry must identify the retained binary flyr wheel")
    retained = flyr[0]["retained_artifact"]
    path = retained.get("path")
    parts = _safe_relative_path(path, label="retained flyr wheel path")
    if parts[:2] != ("artifacts", "reference-wheels"):
        raise ReferenceSetupError("Retained flyr wheel must remain under artifacts/reference-wheels")
    byte_count, digest = retained.get("bytes"), retained.get("sha256")
    locked_flyr = locked.get("flyr")
    if (isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count < 0
            or not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest)
            or locked_flyr is None or digest != locked_flyr[2] or digest != flyr[0].get("archive_sha256")):
        raise ReferenceSetupError("Retained flyr wheel does not match the binary-only lock hash")
    return path, byte_count, digest


def _load_plan(
    repository_root: Path,
    config_path: Path | None,
    requirements_lock_path: Path | None,
) -> tuple[dict[str, Any], tuple[_SourceRecord, ...], bytes, bytes, bytes, dict[str, str], dict[str, tuple[str, str, str]], tuple[str, int, str]]:
    root = repository_root.resolve(strict=False)
    config_file = config_path or root / "config" / "peer_reference_sources.json"
    config, _ = _load_source_registry(config_file)
    records = _validate_source_records(config)
    projection = _source_audit_projection(config, records)
    source_manifest_raw = _pretty_crlf_json(projection)
    source_manifest_hash = _sha256(source_manifest_raw)
    if source_manifest_hash != PINNED_SOURCE_AUDIT_SHA256:
        raise ReferenceSetupError("Reconstructed PlantCV source audit differs from the runner's pinned raw hash")
    if not isinstance(projection["retrieved_date"], str) or not projection["retrieved_date"]:
        raise ReferenceSetupError("Pinned source audit requires a retrieved date")

    environment = config.get("environment")
    if not isinstance(environment, dict):
        raise ReferenceSetupError("Peer registry has no frozen environment record")
    if environment.get("lockfile") != "config/plantcv-reference-requirements.txt":
        raise ReferenceSetupError("Peer registry points to an unexpected requirements-lock path")
    lock_file = requirements_lock_path or _safe_child(
        root, environment["lockfile"], label="requirements lock path",
    )
    lock_raw = _read_bounded(lock_file, MAX_REQUIREMENTS_BYTES, "PlantCV requirements lock")
    if _sha256(lock_raw) != PINNED_REQUIREMENTS_SHA256:
        raise ReferenceSetupError("Pinned PlantCV requirements lock has changed")
    locked = _parse_locked_packages(lock_raw)
    inventory = _environment_inventory(config, lock_raw)
    environment_raw = _pretty_crlf_json(inventory)
    retained = _retained_flyr(config, locked)
    return config, records, source_manifest_raw, environment_raw, lock_raw, inventory, locked, retained


def _verify_existing_or_missing(path: Path, expected_hash: str, expected_bytes: int, label: str) -> bool:
    if not path.exists():
        return False
    if path.is_symlink() or not path.is_file():
        raise ReferenceSetupError(f"Existing {label} is not a regular file: {path}")
    try:
        with path.open("rb") as handle:
            raw = handle.read(expected_bytes + 1)
    except OSError as exc:
        raise ReferenceSetupError(f"Cannot read existing {label} {path}: {exc}") from exc
    if len(raw) != expected_bytes or _sha256(raw) != expected_hash:
        raise ReferenceSetupError(f"Existing {label} does not match its pinned bytes: {path}")
    return True


def _source_locations(target_dir: Path, records: tuple[_SourceRecord, ...]) -> tuple[Path, dict[str, Path]]:
    target = target_dir.resolve(strict=False)
    if target.exists() and not target.is_dir():
        raise ReferenceSetupError(f"Reference setup target is not a directory: {target}")
    source_root = _safe_child(target, "plantcv-v4.11.3", label="source root")
    locations = {record.path: _safe_child(source_root, record.path, label="source path") for record in records}
    return source_root, locations


def inspect_reference_setup(
    target_dir: Path | None = None,
    *,
    repository_root: Path | None = None,
    config_path: Path | None = None,
    requirements_lock_path: Path | None = None,
) -> ReferenceSetupStatus:
    """Inspect setup prerequisites without writing or fetching any artifact."""
    root = (repository_root or Path(__file__).resolve().parents[2]).resolve(strict=False)
    target = (target_dir or root / "artifacts" / "reference-sources").resolve(strict=False)
    config, records, source_manifest_raw, environment_raw, lock_raw, inventory, locked, retained = _load_plan(
        root, config_path, requirements_lock_path,
    )
    source_root, locations = _source_locations(target, records)
    unavailable = tuple(
        path for path, location in locations.items()
        if not _verify_existing_or_missing(location, next(item.sha256 for item in records if item.path == path),
                                           next(item.byte_count for item in records if item.path == path),
                                           f"PlantCV source {path}")
    )
    source_manifest_path = _safe_child(target, "plantcv-source-audit.json", label="source audit output")
    environment_manifest_path = _safe_child(target, "plantcv-environment.json", label="environment output")
    source_hash, environment_hash = _sha256(source_manifest_raw), _sha256(environment_raw)
    source_meta = _verify_existing_or_missing(source_manifest_path, source_hash, len(source_manifest_raw), "source audit metadata")
    environment_meta = _verify_existing_or_missing(environment_manifest_path, environment_hash, len(environment_raw), "environment metadata")

    wheel_relative, wheel_bytes, wheel_hash = retained
    wheel_path = _safe_child(root, wheel_relative, label="retained flyr wheel")
    wheel_status = "verified" if _verify_existing_or_missing(wheel_path, wheel_hash, wheel_bytes, "retained flyr wheel") else "missing"
    return ReferenceSetupStatus(
        target_dir=target,
        source_root=source_root,
        source_audit_path=source_manifest_path,
        environment_manifest_path=environment_manifest_path,
        source_audit_sha256=source_hash,
        environment_manifest_sha256=environment_hash,
        requirements_lock_sha256=_sha256(lock_raw),
        source_file_count=len(records),
        package_count=len(inventory),
        unavailable_sources=unavailable,
        wheel_path=wheel_path,
        wheel_status=wheel_status,
        metadata_available=source_meta and environment_meta,
    )


def _iter_bounded_chunks(chunks: Iterable[bytes] | bytes, *, maximum: int, label: str) -> Iterable[bytes]:
    iterator: Iterable[Any] = (chunks,) if isinstance(chunks, bytes) else chunks
    total = 0
    empty_chunks = 0
    for chunk_index, chunk in enumerate(iterator, start=1):
        if chunk_index > MAX_FETCH_CHUNKS:
            raise ReferenceSetupError(f"Fetch callback exceeded the {MAX_FETCH_CHUNKS}-chunk limit for {label}")
        if not isinstance(chunk, (bytes, bytearray, memoryview)):
            raise ReferenceSetupError(f"Fetch callback yielded a non-byte chunk for {label}")
        chunk_size = chunk.nbytes if isinstance(chunk, memoryview) else len(chunk)
        if chunk_size > maximum - total:
            raise ReferenceSetupError(f"Fetched {label} exceeds its declared byte limit")
        if chunk_size == 0:
            empty_chunks += 1
            if empty_chunks > MAX_EMPTY_FETCH_CHUNKS:
                raise ReferenceSetupError(f"Fetch callback made no byte progress for {label}")
        else:
            total += chunk_size
        data = _copy_fetch_chunk(chunk)
        yield data


def _copy_fetch_chunk(chunk: bytes | bytearray | memoryview) -> bytes:
    """Copy one already size-checked chunk into an immutable write buffer."""
    return bytes(chunk)


def _fetch_source_atomically(
    target: Path,
    record: _SourceRecord,
    fetcher: Callable[[str, int], Iterable[bytes] | bytes],
) -> bool:
    """Fetch and verify one source with bounded streaming and atomic publication."""
    if target.exists():
        if _verify_existing_or_missing(target, record.sha256, record.byte_count, f"PlantCV source {record.path}"):
            return False
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() or target.parent.is_symlink():
        raise ReferenceSetupError(f"Source target traverses a symbolic link: {record.path}")
    temp_name: str | None = None
    digest = hashlib.sha256()
    total = 0
    try:
        fd, temp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
        with os.fdopen(fd, "wb") as handle:
            chunks = fetcher(record.url, record.byte_count)
            for chunk in _iter_bounded_chunks(chunks, maximum=min(record.byte_count, MAX_SOURCE_BYTES), label=record.path):
                total += len(chunk)
                digest.update(chunk)
                handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        if total != record.byte_count or digest.hexdigest() != record.sha256:
            raise ReferenceSetupError(f"Fetched source failed its pinned length or SHA-256: {record.path}")
        if target.exists():
            if _verify_existing_or_missing(target, record.sha256, record.byte_count, f"PlantCV source {record.path}"):
                return False
        # A hard-link publish is atomic and fails if another writer created the target.
        try:
            os.link(temp_name, target)
        except FileExistsError:
            if _verify_existing_or_missing(target, record.sha256, record.byte_count, f"PlantCV source {record.path}"):
                return False
            raise ReferenceSetupError(f"A different file appeared at the source target: {record.path}")
        return True
    finally:
        if temp_name is not None:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass


def _atomic_write_if_missing(path: Path, content: bytes, *, label: str) -> bool:
    digest = _sha256(content)
    if _verify_existing_or_missing(path, digest, len(content), label):
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or path.parent.is_symlink():
        raise ReferenceSetupError(f"Output path traverses a symbolic link: {path}")
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if _verify_existing_or_missing(path, digest, len(content), label):
                return False
            raise ReferenceSetupError(f"A different file appeared at output path: {path}")
        return True
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def materialize_reference_setup(
    target_dir: Path | None = None,
    *,
    repository_root: Path | None = None,
    config_path: Path | None = None,
    requirements_lock_path: Path | None = None,
    fetch_sources: bool = False,
    fetcher: Callable[[str, int], Iterable[bytes] | bytes] | None = None,
) -> ReferenceSetupStatus:
    """Recreate runner metadata and optionally fetch only audited source bytes.

    Network access is never implicit. When ``fetch_sources`` is true, the caller
    must provide a callback; it receives a pinned raw GitHub URL and the exact
    declared byte ceiling, and must yield bytes chunks. All chunks are hashed and
    bounded before a source is atomically published.
    """
    if not isinstance(fetch_sources, bool):
        raise TypeError("fetch_sources must be a bool")
    if fetch_sources and fetcher is None:
        raise ReferenceSetupError("Explicit source fetch requires a caller-supplied fetcher callback")
    if fetcher is not None and not fetch_sources:
        raise ReferenceSetupError("A fetch callback is accepted only with fetch_sources=True")

    root = (repository_root or Path(__file__).resolve().parents[2]).resolve(strict=False)
    target = (target_dir or root / "artifacts" / "reference-sources").resolve(strict=False)
    config, records, source_manifest_raw, environment_raw, lock_raw, inventory, locked, retained = _load_plan(
        root, config_path, requirements_lock_path,
    )
    source_root, locations = _source_locations(target, records)

    # Preflight every existing output before any writes; wrong bytes are never replaced.
    missing_sources: list[_SourceRecord] = []
    for record in records:
        location = locations[record.path]
        if not _verify_existing_or_missing(location, record.sha256, record.byte_count, f"PlantCV source {record.path}"):
            missing_sources.append(record)
    source_manifest_path = _safe_child(target, "plantcv-source-audit.json", label="source audit output")
    environment_manifest_path = _safe_child(target, "plantcv-environment.json", label="environment output")
    _verify_existing_or_missing(source_manifest_path, _sha256(source_manifest_raw), len(source_manifest_raw), "source audit metadata")
    _verify_existing_or_missing(environment_manifest_path, _sha256(environment_raw), len(environment_raw), "environment metadata")
    wheel_relative, wheel_bytes, wheel_hash = retained
    wheel_path = _safe_child(root, wheel_relative, label="retained flyr wheel")
    _verify_existing_or_missing(wheel_path, wheel_hash, wheel_bytes, "retained flyr wheel")

    created: list[tuple[Path, str, int]] = []
    try:
        if fetch_sources and missing_sources:
            assert fetcher is not None
            for record in missing_sources:
                path = locations[record.path]
                if _fetch_source_atomically(path, record, fetcher):
                    created.append((path, record.sha256, record.byte_count))
        if _atomic_write_if_missing(source_manifest_path, source_manifest_raw, label="source audit metadata"):
            created.append((source_manifest_path, _sha256(source_manifest_raw), len(source_manifest_raw)))
        if _atomic_write_if_missing(environment_manifest_path, environment_raw, label="environment metadata"):
            created.append((environment_manifest_path, _sha256(environment_raw), len(environment_raw)))
    except BaseException:
        # Remove only outputs created by this call; retain all pre-existing artifacts.
        for path, expected_hash, expected_bytes in reversed(created):
            try:
                if path.is_file() and not path.is_symlink():
                    with path.open("rb") as handle:
                        current = handle.read(expected_bytes + 1)
                    if len(current) == expected_bytes and _sha256(current) == expected_hash:
                        path.unlink()
            except OSError:
                pass
        raise
    return inspect_reference_setup(
        target,
        repository_root=root,
        config_path=config_path,
        requirements_lock_path=requirements_lock_path,
    )


__all__ = [
    "MAX_PACKAGES",
    "MAX_SOURCE_FILES",
    "ReferenceSetupError",
    "ReferenceSetupStatus",
    "inspect_reference_setup",
    "materialize_reference_setup",
]
