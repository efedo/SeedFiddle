"""Durable, image-bound persistence for painted reference regions."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import os
from pathlib import Path
import re
import tempfile
from zipfile import BadZipFile

import numpy as np


REFERENCE_REGION_VERSION = 1
_MATERIAL_CLASSES = frozenset((0, 1, 2, 3))
_BOUNDARY_CLASSES = frozenset((0, 1, 2))


class ReferenceRegionError(ValueError):
    """Base error for an unusable reference-region archive."""


class ImageFingerprintMismatch(ReferenceRegionError):
    """The image bytes no longer match those used when references were saved."""

    def __init__(
        self,
        image_path: Path,
        archive_path: Path,
        expected_sha256: str,
        actual_sha256: str,
    ) -> None:
        super().__init__(
            f"Saved references for {image_path.name} require image SHA-256 "
            f"{expected_sha256}, but the current file is {actual_sha256}."
        )
        self.image_path = image_path
        self.archive_path = archive_path
        self.expected_sha256 = expected_sha256
        self.actual_sha256 = actual_sha256


class InvalidReferenceArchive(ReferenceRegionError):
    """A saved reference archive is corrupt or violates its schema."""


@dataclass(frozen=True, slots=True)
class ReferenceRegionBundle:
    """All six outputs of the Reference layers node in image coordinates."""

    shape: tuple[int, int]
    background: np.ndarray | None = None
    foreground: np.ndarray | None = None
    other: np.ndarray | None = None
    physical_edge: np.ndarray | None = None
    non_edge: np.ndarray | None = None
    annotated_seeds: np.ndarray | None = None
    annotation_origin: str = "manual"


def file_sha256(path: Path | str) -> str:
    """Return a byte-exact SHA-256 fingerprint without loading the file at once."""

    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ReferenceRegionStore:
    """Save and restore per-image reference layers beneath the project root."""

    def __init__(self, project_root: Path | str) -> None:
        self.project_root = Path(project_root).resolve()
        self.directory = self.project_root / "projects" / "reference-regions"

    def image_identity(self, image_path: Path | str) -> str:
        """Return an identity stable across moves of the complete project tree."""

        resolved = Path(image_path).resolve()
        try:
            relative = resolved.relative_to(self.project_root)
        except ValueError:
            identity = f"absolute:{resolved.as_posix()}"
        else:
            identity = f"project:{relative.as_posix()}"
        return identity.casefold() if os.name == "nt" else identity

    def path_for(self, image_path: Path | str) -> Path:
        image = Path(image_path)
        identity_digest = sha256(
            self.image_identity(image).encode("utf-8")
        ).hexdigest()[:20]
        readable_stem = re.sub(r"[^A-Za-z0-9_.-]+", "-", image.stem).strip(".-")
        readable_stem = readable_stem[:48] or "image"
        return self.directory / f"{readable_stem}-{identity_digest}.references.npz"

    def save(
        self, image_path: Path | str, bundle: ReferenceRegionBundle
    ) -> Path:
        """Atomically save an exclusive categorical representation of all layers."""

        image = Path(image_path).resolve()
        if not image.is_file():
            raise ReferenceRegionError(f"Source image does not exist: {image}")
        height, width = _validated_shape(bundle.shape)
        background = _binary_mask(bundle.background, (height, width), "background")
        foreground = _binary_mask(bundle.foreground, (height, width), "foreground")
        other = _binary_mask(bundle.other, (height, width), "other")
        if np.any(background & foreground) or np.any(background & other) or np.any(
            foreground & other
        ):
            raise ReferenceRegionError(
                "Background, foreground, and other reference regions must be exclusive."
            )
        physical = _binary_mask(
            bundle.physical_edge, (height, width), "physical edge"
        )
        non_edge = _binary_mask(bundle.non_edge, (height, width), "non-edge")
        if np.any(physical & non_edge):
            raise ReferenceRegionError(
                "Physical-edge and non-edge reference regions must be exclusive."
            )
        instances = _instance_labels(bundle.annotated_seeds, (height, width))

        material = np.zeros((height, width), dtype=np.uint8)
        material[background] = 1
        material[foreground] = 2
        material[other] = 3
        boundary = np.zeros((height, width), dtype=np.uint8)
        boundary[physical] = 1
        boundary[non_edge] = 2

        destination = self.path_for(image)
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            with temporary.open("wb") as stream:
                np.savez_compressed(
                    stream,
                    version=np.asarray(REFERENCE_REGION_VERSION, dtype=np.uint16),
                    image_identity=np.asarray(self.image_identity(image)),
                    image_sha256=np.asarray(file_sha256(image)),
                    height=np.asarray(height, dtype=np.int64),
                    width=np.asarray(width, dtype=np.int64),
                    material=material,
                    boundary=boundary,
                    annotated_seeds=instances,
                    annotation_origin=np.asarray(str(bundle.annotation_origin)),
                )
            temporary.replace(destination)
        finally:
            if temporary.exists():
                temporary.unlink()
        return destination

    def load_if_present(
        self,
        image_path: Path | str,
        expected_shape: tuple[int, int],
    ) -> ReferenceRegionBundle | None:
        """Load only after the current image fingerprint matches saved metadata."""

        image = Path(image_path).resolve()
        source = self.path_for(image)
        if not source.is_file():
            return None
        expected_height, expected_width = _validated_shape(expected_shape)
        try:
            with np.load(source, allow_pickle=False) as archive:
                for name in (
                    "version",
                    "image_identity",
                    "image_sha256",
                    "height",
                    "width",
                    "material",
                    "boundary",
                    "annotated_seeds",
                    "annotation_origin",
                ):
                    if name not in archive.files:
                        raise InvalidReferenceArchive(
                            f"Saved reference archive is missing {name!r}: {source}"
                        )
                version = _scalar_int(archive["version"], "version")
                if version != REFERENCE_REGION_VERSION:
                    raise InvalidReferenceArchive(
                        f"Unsupported reference-region version {version} in {source}."
                    )
                saved_identity = _scalar_text(
                    archive["image_identity"], "image identity"
                )
                if saved_identity != self.image_identity(image):
                    raise InvalidReferenceArchive(
                        f"Saved reference archive belongs to a different image: {source}"
                    )
                saved_digest = _scalar_text(archive["image_sha256"], "image SHA-256")
                if not re.fullmatch(r"[0-9a-f]{64}", saved_digest):
                    raise InvalidReferenceArchive(
                        f"Invalid image SHA-256 metadata in {source}."
                    )

                # Do not materialize any full-resolution annotation raster until
                # the byte-exact source-image fingerprint has been checked.
                actual_digest = file_sha256(image)
                if saved_digest != actual_digest:
                    raise ImageFingerprintMismatch(
                        image, source, saved_digest, actual_digest
                    )
                saved_shape = (
                    _scalar_int(archive["height"], "height"),
                    _scalar_int(archive["width"], "width"),
                )
                if saved_shape != (expected_height, expected_width):
                    raise InvalidReferenceArchive(
                        "Saved reference dimensions "
                        f"{saved_shape[1]}×{saved_shape[0]} do not match the image "
                        f"{expected_width}×{expected_height}: {source}"
                    )
                material = np.asarray(archive["material"])
                boundary = np.asarray(archive["boundary"])
                instances = np.asarray(archive["annotated_seeds"])
                annotation_origin = _scalar_text(
                    archive["annotation_origin"], "annotation origin"
                )
        except ImageFingerprintMismatch:
            raise
        except InvalidReferenceArchive:
            raise
        except (OSError, ValueError, TypeError, EOFError, BadZipFile) as error:
            raise InvalidReferenceArchive(
                f"Could not read saved reference regions from {source}: {error}"
            ) from error

        shape = (expected_height, expected_width)
        material = _categorical_raster(
            material, shape, _MATERIAL_CLASSES, "material", source
        )
        boundary = _categorical_raster(
            boundary, shape, _BOUNDARY_CLASSES, "boundary", source
        )
        try:
            instances = _instance_labels(instances, shape)
        except ReferenceRegionError as error:
            raise InvalidReferenceArchive(
                f"Invalid annotated-seed raster in {source}: {error}"
            ) from error
        return ReferenceRegionBundle(
            shape=shape,
            background=_nonempty(material == 1),
            foreground=_nonempty(material == 2),
            other=_nonempty(material == 3),
            physical_edge=_nonempty(boundary == 1),
            non_edge=_nonempty(boundary == 2),
            annotated_seeds=_nonempty(instances),
            annotation_origin=annotation_origin or "manual",
        )


def _validated_shape(shape: tuple[int, int]) -> tuple[int, int]:
    if len(shape) != 2:
        raise ReferenceRegionError("Reference-region shape must contain height and width.")
    height, width = (int(shape[0]), int(shape[1]))
    if height <= 0 or width <= 0:
        raise ReferenceRegionError("Reference-region dimensions must be positive.")
    return height, width


def _binary_mask(
    values: np.ndarray | None, shape: tuple[int, int], name: str
) -> np.ndarray:
    if values is None:
        return np.zeros(shape, dtype=bool)
    array = np.asarray(values)
    if array.shape != shape:
        raise ReferenceRegionError(
            f"{name.capitalize()} reference shape {array.shape} does not match {shape}."
        )
    return np.asarray(array, dtype=bool)


def _instance_labels(
    values: np.ndarray | None, shape: tuple[int, int]
) -> np.ndarray:
    if values is None:
        return np.zeros(shape, dtype=np.uint16)
    array = np.asarray(values)
    if array.ndim != 2 or array.shape != shape:
        raise ReferenceRegionError(
            f"Annotated seed shape {array.shape} does not match {shape}."
        )
    if not np.issubdtype(array.dtype, np.integer):
        raise ReferenceRegionError("Annotated seed IDs must be integers.")
    if np.any(array < 0) or np.any(array > np.iinfo(np.uint16).max):
        raise ReferenceRegionError("Annotated seed IDs must fit in unsigned 16 bits.")
    return array.astype(np.uint16, copy=False)


def _categorical_raster(
    values: np.ndarray,
    shape: tuple[int, int],
    allowed: frozenset[int],
    name: str,
    source: Path,
) -> np.ndarray:
    if values.ndim != 2 or values.shape != shape or not np.issubdtype(
        values.dtype, np.integer
    ):
        raise InvalidReferenceArchive(
            f"Invalid {name} raster shape or type in {source}."
        )
    classes = frozenset(int(value) for value in np.unique(values))
    if not classes <= allowed:
        raise InvalidReferenceArchive(
            f"Invalid {name} classes {sorted(classes - allowed)} in {source}."
        )
    return values.astype(np.uint8, copy=False)


def _scalar_int(values: np.ndarray, name: str) -> int:
    array = np.asarray(values)
    if array.shape != () or not np.issubdtype(array.dtype, np.integer):
        raise InvalidReferenceArchive(f"Invalid scalar {name} metadata.")
    return int(array)


def _scalar_text(values: np.ndarray, name: str) -> str:
    array = np.asarray(values)
    if array.shape != () or array.dtype.kind not in {"U", "S"}:
        raise InvalidReferenceArchive(f"Invalid scalar {name} metadata.")
    value = array.item()
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _nonempty(values: np.ndarray) -> np.ndarray | None:
    return values if np.any(values) else None
