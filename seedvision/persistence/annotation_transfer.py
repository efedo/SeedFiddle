"""Export and import per-image annotations between projects.

A bundle is a zip file holding ``manifest.json`` and one archive per image and
annotation kind. Entries are matched to images by the SHA-256 of the image
file, because every archive is bound to the exact image bytes it was drawn on.
Archives are validated against their image when exported and again when
imported.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import zipfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from seedvision.persistence.instance_masks import read_source_raster_shape
from seedvision.persistence.manual_seed_centres import ManualSeedCentreStore
from seedvision.persistence.project_analysis import (
    MANUAL_SEED_CENTRES_SIDECAR,
    PROJECT_ANALYSIS_MAX_IMAGES,
    REFERENCE_REGIONS_SIDECAR,
)
from seedvision.persistence.project_annotations import ANNOTATION_FOLDERS
from seedvision.persistence.reference_regions import (
    ReferenceRegionStore,
    file_sha256,
)


ANNOTATION_BUNDLE_FORMAT = "seedfiddle-annotations"
ANNOTATION_BUNDLE_VERSION = 1
ANNOTATION_BUNDLE_EXTENSION = ".seedfiddle-annotations.zip"
ANNOTATION_KIND_LABELS = {
    MANUAL_SEED_CENTRES_SIDECAR: "manual seed centres",
    REFERENCE_REGIONS_SIDECAR: "reference layers",
}
_MANIFEST = "manifest.json"
_MAX_MANIFEST_BYTES = 8 * 1024 * 1024
_MAX_MEMBER_BYTES = 512 * 1024 * 1024
_MAX_ENTRIES = 2 * PROJECT_ANALYSIS_MAX_IMAGES
_MEMBER_NAME = re.compile(r"[a-z-]+/[0-9]{1,6}-[A-Za-z0-9_.-]{1,64}\.npz")
_SHA256 = re.compile(r"[0-9a-f]{64}")


class AnnotationTransferError(ValueError):
    """An annotation bundle could not be written, read or matched."""


@dataclass(frozen=True, slots=True)
class AnnotationExportItem:
    """One image's archive of one annotation kind, ready to export."""

    image_path: Path
    kind: str
    archive_path: Path


@dataclass(frozen=True, slots=True)
class AnnotationBundleEntry:
    """One archive listed in a bundle manifest."""

    kind: str
    image_name: str
    image_sha256: str
    source_shape: tuple[int, int]
    member: str


def _validated_archive(
    workspace_root: Path, image: Path, kind: str, archive: Path
) -> tuple[int, int]:
    """Load an archive against its image, raising if either is unusable."""

    source_shape = read_source_raster_shape(image)
    if kind == REFERENCE_REGIONS_SIDECAR:
        loaded = ReferenceRegionStore(workspace_root).load_project_archive(
            image, archive, None
        )
    elif kind == MANUAL_SEED_CENTRES_SIDECAR:
        loaded = ManualSeedCentreStore(workspace_root).load_project_archive(
            image, archive, source_shape
        )
    else:
        raise AnnotationTransferError(f"Unknown annotation kind {kind!r}.")
    if loaded is None:
        raise AnnotationTransferError(f"Annotation archive does not exist: {archive}")
    return source_shape


def write_annotation_bundle(
    destination: Path | str,
    items: Iterable[AnnotationExportItem],
    *,
    workspace_root: Path | str,
) -> tuple[AnnotationBundleEntry, ...]:
    """Validate every archive against its image, then write one bundle atomically."""

    root = Path(workspace_root).resolve()
    destination = Path(destination)
    items = tuple(items)
    if not items:
        raise AnnotationTransferError("Choose at least one annotation to export.")
    if len(items) > _MAX_ENTRIES:
        raise AnnotationTransferError(
            f"A bundle holds at most {_MAX_ENTRIES:,} annotation archives."
        )
    seen: set[tuple[str, str]] = set()
    entries: list[tuple[AnnotationBundleEntry, Path]] = []
    for index, item in enumerate(items, start=1):
        image = Path(item.image_path).resolve()
        archive = Path(item.archive_path).resolve()
        source_shape = _validated_archive(root, image, item.kind, archive)
        digest = file_sha256(image)
        if (item.kind, digest) in seen:
            raise AnnotationTransferError(
                f"{image.name} is listed twice for {ANNOTATION_KIND_LABELS[item.kind]}."
            )
        seen.add((item.kind, digest))
        stem = re.sub(r"[^A-Za-z0-9_.-]+", "-", image.stem).strip(".-")[:48] or "image"
        member = f"{ANNOTATION_FOLDERS[item.kind]}/{index:04d}-{stem}.npz"
        entries.append(
            (
                AnnotationBundleEntry(
                    kind=item.kind,
                    image_name=image.name,
                    image_sha256=digest,
                    source_shape=source_shape,
                    member=member,
                ),
                archive,
            )
        )
    manifest = {
        "format": ANNOTATION_BUNDLE_FORMAT,
        "version": ANNOTATION_BUNDLE_VERSION,
        "entries": [
            {
                "kind": entry.kind,
                "image_name": entry.image_name,
                "image_sha256": entry.image_sha256,
                "source_shape": list(entry.source_shape),
                "member": entry.member,
            }
            for entry, _ in entries
        ],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        # Archives are already compressed NumPy files; storing them avoids a
        # second, pointless compression pass.
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_STORED) as bundle:
            bundle.writestr(
                _MANIFEST, json.dumps(manifest, indent=2, sort_keys=True) + "\n"
            )
            for entry, archive in entries:
                bundle.write(archive, entry.member)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return tuple(entry for entry, _ in entries)


def _entry_from_payload(
    payload: object, members: Mapping[str, zipfile.ZipInfo]
) -> AnnotationBundleEntry:
    if not isinstance(payload, dict) or set(payload) != {
        "kind",
        "image_name",
        "image_sha256",
        "source_shape",
        "member",
    }:
        raise AnnotationTransferError("A bundle entry has missing or unknown fields.")
    kind = payload["kind"]
    if kind not in ANNOTATION_FOLDERS:
        raise AnnotationTransferError(f"Unknown annotation kind {kind!r} in bundle.")
    name = payload["image_name"]
    if not isinstance(name, str) or not name or len(name) > 255 or "\x00" in name:
        raise AnnotationTransferError("A bundle entry has an invalid image name.")
    digest = payload["image_sha256"]
    if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
        raise AnnotationTransferError(f"Invalid image SHA-256 for {name} in bundle.")
    shape = payload["source_shape"]
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or not all(type(value) is int and value > 0 for value in shape)
    ):
        raise AnnotationTransferError(f"Invalid image dimensions for {name} in bundle.")
    member = payload["member"]
    if (
        not isinstance(member, str)
        or not _MEMBER_NAME.fullmatch(member)
        or not member.startswith(ANNOTATION_FOLDERS[kind] + "/")
    ):
        raise AnnotationTransferError(f"Invalid archive name for {name} in bundle.")
    info = members.get(member)
    if info is None:
        raise AnnotationTransferError(f"The bundle is missing {member}.")
    if info.file_size > _MAX_MEMBER_BYTES:
        raise AnnotationTransferError(f"{member} is larger than an annotation archive can be.")
    return AnnotationBundleEntry(kind, name, digest, (shape[0], shape[1]), member)


def read_annotation_bundle(source: Path | str) -> tuple[AnnotationBundleEntry, ...]:
    """Read and strictly validate a bundle manifest without extracting archives."""

    source = Path(source)
    try:
        with zipfile.ZipFile(source) as bundle:
            members = {info.filename: info for info in bundle.infolist()}
            manifest_info = members.get(_MANIFEST)
            if manifest_info is None:
                raise AnnotationTransferError(
                    f"{source.name} is not a Seed Fiddle annotation bundle."
                )
            if manifest_info.file_size > _MAX_MANIFEST_BYTES:
                raise AnnotationTransferError("The bundle manifest is too large.")
            payload = json.loads(bundle.read(_MANIFEST).decode("utf-8"))
    except (OSError, zipfile.BadZipFile, UnicodeError, json.JSONDecodeError) as error:
        raise AnnotationTransferError(
            f"Could not read annotation bundle {source}: {error}"
        ) from error
    if (
        not isinstance(payload, dict)
        or payload.get("format") != ANNOTATION_BUNDLE_FORMAT
        or set(payload) != {"format", "version", "entries"}
    ):
        raise AnnotationTransferError(
            f"{source.name} is not a Seed Fiddle annotation bundle."
        )
    if payload["version"] != ANNOTATION_BUNDLE_VERSION:
        raise AnnotationTransferError(
            f"Unsupported annotation bundle version {payload['version']!r}."
        )
    raw_entries = payload["entries"]
    if not isinstance(raw_entries, list) or len(raw_entries) > _MAX_ENTRIES:
        raise AnnotationTransferError("The bundle manifest has an invalid entry list.")
    entries = tuple(_entry_from_payload(item, members) for item in raw_entries)
    keys = [(entry.kind, entry.image_sha256) for entry in entries]
    if len(set(keys)) != len(keys):
        raise AnnotationTransferError(
            "The bundle lists one image twice for the same annotation kind."
        )
    if len({entry.member for entry in entries}) != len(entries):
        raise AnnotationTransferError("The bundle lists one archive twice.")
    return entries


def extract_annotation(
    source: Path | str, entry: AnnotationBundleEntry, directory: Path | str
) -> Path:
    """Copy one archive out of a bundle into ``directory``."""

    target = Path(directory) / Path(entry.member).name
    try:
        with zipfile.ZipFile(source) as bundle:
            info = bundle.getinfo(entry.member)
            if info.file_size > _MAX_MEMBER_BYTES:
                raise AnnotationTransferError(
                    f"{entry.member} is larger than an annotation archive can be."
                )
            with bundle.open(info) as stream, target.open("wb") as output:
                remaining = _MAX_MEMBER_BYTES
                while block := stream.read(1024 * 1024):
                    remaining -= len(block)
                    if remaining < 0:
                        raise AnnotationTransferError(
                            f"{entry.member} is larger than its manifest records."
                        )
                    output.write(block)
    except (OSError, KeyError, zipfile.BadZipFile) as error:
        raise AnnotationTransferError(
            f"Could not extract {entry.member} from {source}: {error}"
        ) from error
    return target
