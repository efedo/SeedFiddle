"""Compact image-bound persistence for manually edited seed centres."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import os
from pathlib import Path
import re
import tempfile
from zipfile import BadZipFile

import numpy as np

from seedvision.persistence.reference_regions import file_sha256


MANUAL_SEED_CENTRE_VERSION = 1
MANUAL_SEED_CENTRE_MODES = frozenset(("augment", "replace_automatic"))


class ManualSeedCentreStoreError(ValueError):
    """Base error for an unusable manual-centre sidecar."""


class ManualSeedCentreFingerprintMismatch(ManualSeedCentreStoreError):
    """The source image bytes differ from the sidecar's image fingerprint."""


class InvalidManualSeedCentreArchive(ManualSeedCentreStoreError):
    """A manual-centre sidecar is corrupt or violates its schema."""


@dataclass(frozen=True, slots=True)
class StoredManualSeedCentres:
    """Manual points in immutable source-image coordinates."""

    source_shape: tuple[int, int]
    centres_xy: np.ndarray
    mode: str = "augment"


class ManualSeedCentreStore:
    """Atomically save small per-image centre edits beneath the project root."""

    def __init__(self, project_root: Path | str) -> None:
        self.project_root = Path(project_root).resolve()
        self.directory = self.project_root / "projects" / "manual-seed-centres"

    def image_identity(self, image_path: Path | str) -> str:
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
        return self.directory / f"{readable_stem}-{identity_digest}.centres.npz"

    def save(
        self,
        image_path: Path | str,
        centres_xy: np.ndarray,
        *,
        mode: str,
        source_shape: tuple[int, int],
    ) -> Path:
        """Save exact source-coordinate floats, including meaningful empty replace."""

        image = Path(image_path).resolve()
        if not image.is_file():
            raise ManualSeedCentreStoreError(
                f"Source image does not exist: {image}"
            )
        height, width = _validated_shape(source_shape)
        centres = _validated_centres(
            centres_xy, (height, width), ManualSeedCentreStoreError
        )
        mode = _validated_mode(mode, ManualSeedCentreStoreError)
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
                    version=np.asarray(
                        MANUAL_SEED_CENTRE_VERSION, dtype=np.uint16
                    ),
                    image_identity=np.asarray(self.image_identity(image)),
                    image_sha256=np.asarray(file_sha256(image)),
                    source_height=np.asarray(height, dtype=np.int64),
                    source_width=np.asarray(width, dtype=np.int64),
                    coordinate_space=np.asarray("source-image"),
                    mode=np.asarray(mode),
                    centres_xy=centres,
                )
            temporary.replace(destination)
        finally:
            if temporary.exists():
                temporary.unlink()
        return destination

    def load_if_present(
        self,
        image_path: Path | str,
        expected_source_shape: tuple[int, int],
    ) -> StoredManualSeedCentres | None:
        """Load only points bound to the unchanged source photograph."""

        image = Path(image_path).resolve()
        return self._load_archive(
            image,
            self.path_for(image),
            expected_source_shape,
            require_current_identity=True,
        )

    def load_project_archive(
        self,
        image_path: Path | str,
        archive_path: Path | str,
        expected_source_shape: tuple[int, int],
    ) -> StoredManualSeedCentres | None:
        """Load the exact source-coordinate sidecar named by a project master.

        Source dimensions and bytes remain authoritative.  The stored path
        identity is not compared with this host's recomputation because its
        case normalization is intentionally part of loose-workspace discovery,
        not the explicit project-file binding.
        """

        return self._load_archive(
            Path(image_path).resolve(),
            Path(archive_path).resolve(),
            expected_source_shape,
            require_current_identity=False,
        )

    def _load_archive(
        self,
        image: Path,
        source: Path,
        expected_source_shape: tuple[int, int],
        *,
        require_current_identity: bool,
    ) -> StoredManualSeedCentres | None:
        if not source.is_file():
            return None
        expected_shape = _validated_shape(expected_source_shape)
        try:
            with np.load(source, allow_pickle=False) as archive:
                required = (
                    "version",
                    "image_identity",
                    "image_sha256",
                    "source_height",
                    "source_width",
                    "coordinate_space",
                    "mode",
                    "centres_xy",
                )
                missing = [name for name in required if name not in archive.files]
                if missing:
                    raise InvalidManualSeedCentreArchive(
                        f"Manual-centre archive is missing {missing[0]!r}: {source}"
                    )
                version = _scalar_int(archive["version"], "version")
                if version != MANUAL_SEED_CENTRE_VERSION:
                    raise InvalidManualSeedCentreArchive(
                        f"Unsupported manual-centre version {version} in {source}."
                    )
                identity = _scalar_text(archive["image_identity"], "image identity")
                if not identity or "\x00" in identity:
                    raise InvalidManualSeedCentreArchive(
                        f"Invalid image identity metadata in {source}."
                    )
                if require_current_identity and identity != self.image_identity(image):
                    raise InvalidManualSeedCentreArchive(
                        f"Manual-centre archive belongs to a different image: {source}"
                    )
                saved_digest = _scalar_text(
                    archive["image_sha256"], "image SHA-256"
                )
                if not re.fullmatch(r"[0-9a-f]{64}", saved_digest):
                    raise InvalidManualSeedCentreArchive(
                        f"Invalid image SHA-256 metadata in {source}."
                    )
                if saved_digest != file_sha256(image):
                    raise ManualSeedCentreFingerprintMismatch(
                        f"Manual centres for {image.name} were saved against different "
                        f"image bytes. The sidecar was left unchanged at {source}."
                    )
                saved_shape = (
                    _scalar_int(archive["source_height"], "source height"),
                    _scalar_int(archive["source_width"], "source width"),
                )
                if saved_shape != expected_shape:
                    raise InvalidManualSeedCentreArchive(
                        f"Saved source dimensions {saved_shape[1]}×{saved_shape[0]} "
                        f"do not match {expected_shape[1]}×{expected_shape[0]}: "
                        f"{source}"
                    )
                if _scalar_text(archive["coordinate_space"], "coordinate space") != (
                    "source-image"
                ):
                    raise InvalidManualSeedCentreArchive(
                        f"Unsupported manual-centre coordinate space in {source}."
                    )
                mode = _validated_mode(
                    _scalar_text(archive["mode"], "mode"),
                    InvalidManualSeedCentreArchive,
                )
                centres = _validated_centres(
                    np.asarray(archive["centres_xy"]),
                    saved_shape,
                    InvalidManualSeedCentreArchive,
                )
        except (ManualSeedCentreStoreError,):
            raise
        except (OSError, ValueError, TypeError, EOFError, BadZipFile) as error:
            raise InvalidManualSeedCentreArchive(
                f"Could not read manual seed centres from {source}: {error}"
            ) from error
        centres.flags.writeable = False
        return StoredManualSeedCentres(
            source_shape=saved_shape,
            centres_xy=centres,
            mode=mode,
        )


def _validated_shape(shape: tuple[int, int]) -> tuple[int, int]:
    if len(shape) != 2:
        raise ManualSeedCentreStoreError(
            "Source shape must contain height and width."
        )
    height, width = (int(shape[0]), int(shape[1]))
    if height <= 0 or width <= 0:
        raise ManualSeedCentreStoreError("Source dimensions must be positive.")
    return height, width


def _validated_centres(
    values: np.ndarray,
    shape: tuple[int, int],
    error_type: type[ManualSeedCentreStoreError],
) -> np.ndarray:
    centres = np.asarray(values)
    if centres.ndim != 2 or centres.shape[1:] != (2,) or centres.dtype.kind != "f":
        raise error_type("Manual seed centres must be an N×2 floating-point array.")
    centres = centres.astype(np.float64, copy=False)
    if not np.all(np.isfinite(centres)):
        raise error_type("Manual seed centres must contain finite coordinates.")
    height, width = shape
    if len(centres) and (
        np.any(centres[:, 0] < 0.0)
        or np.any(centres[:, 0] >= width)
        or np.any(centres[:, 1] < 0.0)
        or np.any(centres[:, 1] >= height)
    ):
        raise error_type("Manual seed centres must remain inside the source image.")
    return centres


def _validated_mode(
    mode: str, error_type: type[ManualSeedCentreStoreError]
) -> str:
    mode = str(mode)
    if mode not in MANUAL_SEED_CENTRE_MODES:
        raise error_type(
            f"Manual seed-centre mode must be one of "
            f"{sorted(MANUAL_SEED_CENTRE_MODES)}."
        )
    return mode


def _scalar_int(values: np.ndarray, name: str) -> int:
    array = np.asarray(values)
    if array.shape != () or not np.issubdtype(array.dtype, np.integer):
        raise InvalidManualSeedCentreArchive(f"Invalid scalar {name} metadata.")
    return int(array)


def _scalar_text(values: np.ndarray, name: str) -> str:
    array = np.asarray(values)
    if array.shape != () or array.dtype.kind not in {"U", "S"}:
        raise InvalidManualSeedCentreArchive(f"Invalid scalar {name} metadata.")
    value = array.item()
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)
