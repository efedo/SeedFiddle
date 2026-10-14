"""Portable, byte-preserving bundles of explicitly selected checkpoints.

This is an archival boundary. It loads existing bounded CPU archives to verify
them, never reruns an operation, and never changes eligibility or provenance.
The bundle contains the selected content addresses and their complete upstream
closure. Its index is published only after every copy has been verified.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Callable, Iterable

from seedvision.benchmarking.checkpoints import (
    CheckpointCancelled, CheckpointLimits, InvalidCheckpoint, load_checkpoint,
)


_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_FILES = ("manifest.json", "arrays.npz")
_INDEX = "checkpoint-bundle.json"
_FORMAT = "seedfiddle-checkpoint-bundle"
_ARRAY_LIMITS = CheckpointLimits(
    max_total_array_bytes=64 * 1024 * 1024,
    max_array_bytes=64 * 1024 * 1024,
    max_metadata_bytes=1024 * 1024,
    max_npz_bytes=80 * 1024 * 1024,
    max_arrays=8,
)


@dataclass(frozen=True, slots=True)
class CheckpointBundleLimits:
    max_checkpoints: int = 256
    max_encoded_bytes: int = 512 * 1024 * 1024
    max_index_bytes: int = 256 * 1024

    def __post_init__(self) -> None:
        for name, cap in (("max_checkpoints", 256),
                          ("max_encoded_bytes", 512 * 1024 * 1024),
                          ("max_index_bytes", 256 * 1024)):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= cap:
                raise ValueError(f"{name} must be between 1 and {cap}.")


@dataclass(frozen=True, slots=True)
class CheckpointBundle:
    path: Path
    roots: tuple[str, ...]
    checkpoints: tuple[str, ...]
    # Relative paths and raw-file SHA-256 values, including the bundle index.
    files: tuple[tuple[str, str], ...]


def _cancel(cancel: Callable[[], bool] | None) -> None:
    if cancel is not None and cancel():
        raise CheckpointCancelled("Checkpoint bundle creation was cancelled.")


def _digest(value: object) -> str:
    if not isinstance(value, str) or not _DIGEST.fullmatch(value):
        raise InvalidCheckpoint("Bundle checkpoint addresses must be lowercase SHA-256 values.")
    return value


def _raw_hash(path: Path, cancel=None, *, max_bytes=None) -> str:
    digest = sha256()
    maximum = path.stat().st_size if max_bytes is None else max_bytes
    total = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            _cancel(cancel)
            total += len(chunk)
            if total > maximum:
                raise InvalidCheckpoint("A checkpoint file grew beyond its preflight byte limit.")
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_files(store: Path, digest: str) -> tuple[Path, Path]:
    directory = store / f"{digest}.checkpoint"
    if directory.is_symlink() or not directory.is_dir():
        raise InvalidCheckpoint("A bundle dependency is not a regular checkpoint directory.")
    if {item.name for item in directory.iterdir()} != set(_FILES):
        raise InvalidCheckpoint("A bundled checkpoint must contain exactly its manifest and arrays.")
    paths = tuple(directory / name for name in _FILES)
    for path, maximum in zip(paths, (_ARRAY_LIMITS.max_metadata_bytes, _ARRAY_LIMITS.max_npz_bytes)):
        if path.is_symlink() or not path.is_file() or path.stat().st_size > maximum:
            raise InvalidCheckpoint("A checkpoint file is unavailable or exceeds the bundle limit.")
    return paths


def _addresses(values: Iterable[str], maximum: int) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise InvalidCheckpoint("Bundle roots must be an iterable of checkpoint addresses.")
    roots: set[str] = set()
    for value in values:
        roots.add(_digest(value))
        if len(roots) > maximum:
            raise InvalidCheckpoint("Too many selected checkpoint addresses.")
    if not roots:
        raise InvalidCheckpoint("Select at least one checkpoint to bundle.")
    return tuple(sorted(roots))


def export_checkpoint_bundle(
    source_store: Path | str,
    checkpoint_sha256s: Iterable[str],
    destination: Path | str,
    *,
    cancel: Callable[[], bool] | None = None,
    limits: CheckpointBundleLimits = CheckpointBundleLimits(),
) -> CheckpointBundle:
    """Copy an exact upstream closure into a new, atomically published directory."""
    if cancel is not None and not callable(cancel):
        raise TypeError("cancel must be a zero-argument callable or None")
    _cancel(cancel)
    roots = _addresses(checkpoint_sha256s, limits.max_checkpoints)
    store, target = Path(source_store).resolve(), Path(destination).absolute()
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    records: dict[str, dict[str, object]] = {}
    encoded_bytes = 0
    pending = list(roots)
    with tempfile.TemporaryDirectory(prefix=".checkpoint-bundle-", dir=target.parent) as temporary:
        staged = Path(temporary) / "bundle"
        staged.mkdir()
        while pending:
            _cancel(cancel)
            digest = pending.pop()
            if digest in records:
                continue
            if len(records) >= limits.max_checkpoints:
                raise InvalidCheckpoint("Checkpoint closure exceeds the bundle count limit.")
            paths = _checkpoint_files(store, digest)
            sizes = {path.name: path.stat().st_size for path in paths}
            encoded_bytes += sum(sizes.values())
            if encoded_bytes > limits.max_encoded_bytes:
                raise InvalidCheckpoint("Checkpoint closure exceeds the encoded byte limit.")
            raw = {path.name: _raw_hash(path, cancel, max_bytes=sizes[path.name]) for path in paths}
            checkpoint = load_checkpoint(store, digest, limits=_ARRAY_LIMITS)
            upstream = checkpoint.identity.upstream_checkpoint_sha256
            del checkpoint  # Retain no decoded raster between checkpoints.
            output = staged / f"{digest}.checkpoint"
            output.mkdir()
            for source in paths:
                copied = output / source.name
                copied_bytes = 0
                with source.open("rb") as reader, copied.open("xb") as writer:
                    while chunk := reader.read(1024 * 1024):
                        _cancel(cancel)
                        copied_bytes += len(chunk)
                        if copied_bytes > sizes[source.name]:
                            raise InvalidCheckpoint("A checkpoint changed while bundling its preflight bytes.")
                        writer.write(chunk)
                if (_raw_hash(copied, cancel, max_bytes=sizes[source.name]) != raw[source.name]
                        or _raw_hash(source, cancel, max_bytes=sizes[source.name]) != raw[source.name]):
                    raise InvalidCheckpoint("A checkpoint changed while bundling.")
            copied_checkpoint = load_checkpoint(staged, digest, limits=_ARRAY_LIMITS)
            del copied_checkpoint
            records[digest] = {"upstream": upstream, "files": raw}
            if upstream is not None:
                pending.append(upstream)
        payload = {"format": _FORMAT, "version": 1, "roots": list(roots), "checkpoints": records}
        data = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode("utf-8")
        if len(data) > limits.max_index_bytes:
            raise InvalidCheckpoint("Checkpoint bundle index exceeds its byte limit.")
        (staged / _INDEX).write_bytes(data)
        _cancel(cancel)
        if target.exists():
            raise FileExistsError(target)
        os.rename(staged, target)
    files = [(f"{digest}.checkpoint/{name}", str(raw))
             for digest, record in sorted(records.items())
             for name, raw in sorted(record["files"].items())]
    files.append((_INDEX, sha256(data).hexdigest()))
    return CheckpointBundle(target, roots, tuple(sorted(records)), tuple(files))


def verify_checkpoint_bundle(
    directory: Path | str,
    *,
    limits: CheckpointBundleLimits = CheckpointBundleLimits(),
) -> CheckpointBundle:
    """Verify raw copies, checkpoint contents and complete recorded dependencies."""
    path = Path(directory).absolute()
    index = path / _INDEX
    if path.is_symlink() or not path.is_dir() or index.is_symlink() or not index.is_file():
        raise InvalidCheckpoint("Checkpoint bundle is unavailable.")
    if index.stat().st_size > limits.max_index_bytes:
        raise InvalidCheckpoint("Checkpoint bundle index exceeds its byte limit.")
    try:
        with index.open('rb') as stream:
            data = stream.read(limits.max_index_bytes + 1)
        if len(data) > limits.max_index_bytes:
            raise InvalidCheckpoint("Checkpoint bundle index exceeds its byte limit.")
        payload = json.loads(data)
    except (ValueError, UnicodeError) as error:
        raise InvalidCheckpoint("Checkpoint bundle index is invalid.") from error
    if (not isinstance(payload, dict) or set(payload) != {"format", "version", "roots", "checkpoints"}
            or payload["format"] != _FORMAT or type(payload["version"]) is not int or payload["version"] != 1
            or not isinstance(payload["roots"], list) or not isinstance(payload["checkpoints"], dict)):
        raise InvalidCheckpoint("Unsupported checkpoint bundle schema.")
    roots = _addresses(payload["roots"], limits.max_checkpoints)
    if list(roots) != payload["roots"]:
        raise InvalidCheckpoint("Bundle roots must be unique and sorted.")
    records = payload["checkpoints"]
    if not 1 <= len(records) <= limits.max_checkpoints:
        raise InvalidCheckpoint("Checkpoint bundle count exceeds its limit.")
    expected_names = {_INDEX, *(f"{_digest(digest)}.checkpoint" for digest in records)}
    if {item.name for item in path.iterdir()} != expected_names:
        raise InvalidCheckpoint("Checkpoint bundle files differ from its inventory.")
    encoded_bytes = 0
    files: list[tuple[str, str]] = []
    for digest, record in sorted(records.items()):
        if (not isinstance(record, dict) or set(record) != {"upstream", "files"}
                or not isinstance(record["files"], dict) or set(record["files"]) != set(_FILES)):
            raise InvalidCheckpoint("Invalid checkpoint bundle record.")
        if record["upstream"] is not None:
            _digest(record["upstream"])
        sources = _checkpoint_files(path, digest)
        sizes = {source.name: source.stat().st_size for source in sources}
        encoded_bytes += sum(sizes.values())
        if encoded_bytes > limits.max_encoded_bytes:
            raise InvalidCheckpoint("Checkpoint bundle exceeds the encoded byte limit.")
        for source in sources:
            expected = _digest(record["files"][source.name])
            if _raw_hash(source, max_bytes=sizes[source.name]) != expected:
                raise InvalidCheckpoint("Checkpoint bundle raw-file verification failed.")
            files.append((f"{digest}.checkpoint/{source.name}", expected))
        checkpoint = load_checkpoint(path, digest, limits=_ARRAY_LIMITS)
        if checkpoint.identity.upstream_checkpoint_sha256 != record["upstream"]:
            raise InvalidCheckpoint("Checkpoint upstream differs from the bundle index.")
        del checkpoint
    reached: set[str] = set()
    for root in roots:
        current: str | None = root
        chain: set[str] = set()
        while current is not None:
            if current not in records or current in chain:
                raise InvalidCheckpoint("Checkpoint bundle has a missing or cyclic dependency.")
            chain.add(current)
            reached.add(current)
            current = records[current]["upstream"]
    if reached != set(records):
        raise InvalidCheckpoint("Checkpoint bundle includes unreferenced checkpoints.")
    files.append((_INDEX, sha256(data).hexdigest()))
    return CheckpointBundle(path, roots, tuple(sorted(records)), tuple(sorted(files)))
