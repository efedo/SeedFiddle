"""Safe import of full-resolution categorical seed-instance masks.

Ad-hoc mask files are interpreted in the annotation editor's corrected-image
coordinates.  Repository-bundled references use an explicit manifest and
source-image coordinates so they can be fingerprinted against the immutable
photograph before the current calibration homography is applied.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from zipfile import BadZipFile

import cv2
import numpy as np

from seedvision.persistence.reference_regions import file_sha256


INSTANCE_REFERENCE_MANIFEST_VERSION = 1
INSTANCE_REFERENCE_DIRECTORY = "seed-instance-references"
INSTANCE_REFERENCE_MANIFEST = "manifest.json"
_LABEL_KEYS = ("labels", "annotated_seeds")


class InstanceMaskImportError(ValueError):
    """A selected or bundled seed-instance mask is unsafe to install."""


class InstanceMaskFingerprintMismatch(InstanceMaskImportError):
    """A bundled mask no longer belongs to the byte-exact current image."""


@dataclass(frozen=True, slots=True)
class ImportedInstanceMask:
    """A fully validated label map ready to become an editable draft."""

    labels: np.ndarray
    source_path: Path
    coordinate_space: str
    origin: str
    reviewed: bool | None = None
    notes: str = ""
    source_sha256: str | None = None
    mask_sha256: str | None = None
    external_reference_count: int | None = None

    @property
    def instance_count(self) -> int:
        values = np.unique(self.labels)
        return int(np.count_nonzero(values))


def read_source_raster_shape(path: Path | str) -> tuple[int, int]:
    """Return the decoder-space source shape used by the analysis pipeline."""

    source = Path(path).resolve()
    try:
        image = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)
    except cv2.error as error:
        raise InstanceMaskImportError(
            f"Could not decode source dimensions for {source}: {error}"
        ) from error
    if image is None or image.ndim != 2:
        raise InstanceMaskImportError(
            f"Could not decode source dimensions for {source}."
        )
    shape = (int(image.shape[0]), int(image.shape[1]))
    del image
    return shape


def load_corrected_instance_mask(
    path: Path | str,
    expected_shape: tuple[int, int],
) -> ImportedInstanceMask:
    """Read an ad-hoc PNG, TIFF, or NPZ in corrected-image coordinates.

    IDs are preserved exactly.  The importer deliberately does not resize,
    relabel, round, or otherwise reinterpret a categorical raster.
    """

    source = Path(path).resolve()
    expected = _validated_shape(expected_shape, "corrected image")
    labels, declared_space = _read_mask_file(source)
    if declared_space not in {None, "corrected"}:
        raise InstanceMaskImportError(
            f"{source.name} declares coordinate_space={declared_space!r}. "
            "Source-coordinate masks require the fingerprinted bundled-reference "
            "manifest so the current calibration can be applied safely."
        )
    labels = _validated_labels(labels, expected, source)
    return ImportedInstanceMask(
        labels=labels,
        source_path=source,
        coordinate_space="corrected",
        origin=f"import:{source}",
    )


def load_bundled_instance_mask(
    project_root: Path | str,
    image_path: Path | str,
    *,
    source_shape: tuple[int, int],
    corrected_shape: tuple[int, int],
    source_to_corrected: np.ndarray,
) -> ImportedInstanceMask | None:
    """Load the source-bound bundled mask for ``image_path``, if one exists.

    The manifest image path, current image SHA-256, encoded dimensions, mask
    checksum, categorical values, declared counts, and transform are all
    validated before a returned array can reach controller state.
    """

    root = Path(project_root).resolve()
    image = Path(image_path).resolve()
    reference_root = (root / INSTANCE_REFERENCE_DIRECTORY).resolve()
    manifest_path = reference_root / INSTANCE_REFERENCE_MANIFEST
    if not manifest_path.is_file():
        return None
    source_height, source_width = _validated_shape(source_shape, "source image")
    corrected_height, corrected_width = _validated_shape(
        corrected_shape, "corrected image"
    )
    try:
        image.relative_to(root)
    except ValueError:
        return None

    payload = _read_manifest(manifest_path)
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise InstanceMaskImportError(
            f"Bundled instance-reference manifest has no valid entries list: "
            f"{manifest_path}"
        )
    matches: list[dict[str, object]] = []
    for index, raw_entry in enumerate(entries):
        if not isinstance(raw_entry, dict):
            raise InstanceMaskImportError(
                f"Bundled instance-reference entry {index + 1} is not an object."
            )
        entry_image = _safe_manifest_path(
            root, raw_entry.get("image_path"), f"entry {index + 1} image_path"
        )
        if _same_path(entry_image, image):
            matches.append(raw_entry)
    if not matches:
        return None
    if len(matches) != 1:
        raise InstanceMaskImportError(
            f"The bundled manifest contains {len(matches)} entries for "
            f"{image.name}; the match is ambiguous."
        )
    entry = matches[0]

    coordinate_space = entry.get("coordinate_space")
    if coordinate_space != "source":
        raise InstanceMaskImportError(
            "Bundled seed-instance references must declare "
            'coordinate_space="source".'
        )
    saved_digest = _sha256_text(entry.get("image_sha256"), "image_sha256")
    actual_digest = _read_file_sha256(image, "source image")
    if saved_digest != actual_digest:
        raise InstanceMaskFingerprintMismatch(
            f"Bundled seed-instance references for {image.name} require image "
            f"SHA-256 {saved_digest}, but the current file is {actual_digest}. "
            "Nothing was loaded."
        )
    saved_width = _positive_int(entry.get("width"), "width")
    saved_height = _positive_int(entry.get("height"), "height")
    if (saved_height, saved_width) != (source_height, source_width):
        raise InstanceMaskImportError(
            f"Bundled source dimensions {saved_width}×{saved_height} do not "
            f"match the current source image {source_width}×{source_height}."
        )

    mask_path = _safe_manifest_path(
        reference_root, entry.get("mask_path"), "mask_path"
    )
    if mask_path.suffix.lower() not in {".png", ".tif", ".tiff"}:
        raise InstanceMaskImportError(
            "Bundled source masks must use lossless PNG or TIFF."
        )
    declared_mask_digest = _sha256_text(entry.get("mask_sha256"), "mask_sha256")
    actual_mask_digest = _read_file_sha256(mask_path, "bundled mask")
    if declared_mask_digest != actual_mask_digest:
        raise InstanceMaskFingerprintMismatch(
            f"Bundled mask {mask_path.name} has SHA-256 "
            f"{actual_mask_digest}, not the manifest value "
            f"{declared_mask_digest}. Nothing was loaded."
        )
    source_labels, declared_space = _read_mask_file(mask_path)
    if declared_space not in {None, "source"}:
        raise InstanceMaskImportError(
            f"Bundled mask declares incompatible coordinate space "
            f"{declared_space!r}."
        )
    source_labels = _validated_labels(
        source_labels, (source_height, source_width), mask_path
    )
    actual_count = _nonzero_identifier_count(source_labels)
    expected_count = _nonnegative_int(entry.get("instance_count"), "instance_count")
    if expected_count != actual_count:
        raise InstanceMaskImportError(
            f"Bundled mask contains {actual_count:,} nonzero seed IDs, not "
            f"the manifest value {expected_count:,}. Nothing was loaded."
        )
    dish_count = _nonnegative_int(
        entry.get("dish_instance_count"), "dish_instance_count"
    )
    external_count = _nonnegative_int(
        entry.get("external_reference_count"), "external_reference_count"
    )
    if dish_count + external_count != expected_count:
        raise InstanceMaskImportError(
            "dish_instance_count plus external_reference_count must equal "
            "instance_count."
        )
    reviewed = entry.get("reviewed")
    if not isinstance(reviewed, bool):
        raise InstanceMaskImportError(
            "Bundled instance-reference entries require a boolean reviewed field."
        )
    notes = entry.get("notes")
    if not isinstance(notes, str):
        raise InstanceMaskImportError(
            "Bundled instance-reference notes must be text."
        )

    transform = np.asarray(source_to_corrected, dtype=np.float64)
    if (
        transform.shape != (3, 3)
        or not np.all(np.isfinite(transform))
        or abs(float(np.linalg.det(transform))) < 1e-12
    ):
        raise InstanceMaskImportError(
            "The current source-to-corrected calibration transform is invalid."
        )
    try:
        corrected = cv2.warpPerspective(
            source_labels,
            transform,
            (corrected_width, corrected_height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    except cv2.error as error:
        raise InstanceMaskImportError(
            f"Could not transform bundled categorical labels: {error}"
        ) from error
    corrected = _validated_labels(
        corrected, (corrected_height, corrected_width), mask_path
    )
    source_ids = set(int(value) for value in np.unique(source_labels)) - {0}
    corrected_ids = set(int(value) for value in np.unique(corrected)) - {0}
    if corrected_ids != source_ids:
        raise InstanceMaskImportError(
            "Nearest-neighbour calibration did not preserve the complete set of "
            "nonzero seed IDs. Nothing was loaded."
        )
    relative_mask = mask_path.relative_to(root).as_posix()
    origin = (
        f"bundled:{relative_mask};source-sha256={actual_digest};"
        f"mask-sha256={actual_mask_digest};reviewed={str(reviewed).lower()}"
    )
    return ImportedInstanceMask(
        labels=corrected,
        source_path=mask_path,
        coordinate_space="source->corrected",
        origin=origin,
        reviewed=reviewed,
        notes=notes,
        source_sha256=actual_digest,
        mask_sha256=actual_mask_digest,
        external_reference_count=external_count,
    )


def _read_manifest(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise InstanceMaskImportError(
            f"Could not read bundled instance-reference manifest {path}: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise InstanceMaskImportError(
            f"Bundled instance-reference manifest must contain one object: {path}"
        )
    version = payload.get("version")
    if isinstance(version, bool) or not isinstance(version, int):
        raise InstanceMaskImportError(
            "Bundled instance-reference manifest version must be an integer."
        )
    if version != INSTANCE_REFERENCE_MANIFEST_VERSION:
        raise InstanceMaskImportError(
            f"Unsupported bundled instance-reference manifest version {version}."
        )
    return payload


def _read_mask_file(path: Path) -> tuple[np.ndarray, str | None]:
    if not path.is_file():
        raise InstanceMaskImportError(f"Seed-instance mask does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix in {".png", ".tif", ".tiff"}:
        try:
            values = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        except cv2.error as error:
            raise InstanceMaskImportError(
                f"Could not decode categorical mask {path}: {error}"
            ) from error
        if values is None:
            raise InstanceMaskImportError(
                f"Could not decode categorical mask {path}."
            )
        return np.asarray(values), None
    if suffix == ".npz":
        try:
            with np.load(path, allow_pickle=False) as archive:
                present = [key for key in _LABEL_KEYS if key in archive.files]
                if len(present) != 1:
                    raise InstanceMaskImportError(
                        f"NPZ mask must contain exactly one of {_LABEL_KEYS}; "
                        f"found {present or 'neither'}."
                    )
                values = np.array(archive[present[0]], copy=True)
                coordinate_space = (
                    _scalar_text(archive["coordinate_space"], "coordinate_space")
                    if "coordinate_space" in archive.files
                    else None
                )
        except InstanceMaskImportError:
            raise
        except (OSError, ValueError, TypeError, EOFError, BadZipFile) as error:
            raise InstanceMaskImportError(
                f"Could not read categorical NPZ mask {path}: {error}"
            ) from error
        return values, coordinate_space
    raise InstanceMaskImportError(
        "Seed-instance masks must be lossless PNG, TIFF, or NPZ files."
    )


def _validated_labels(
    values: np.ndarray,
    expected_shape: tuple[int, int],
    source: Path,
) -> np.ndarray:
    labels = np.asarray(values)
    if labels.ndim != 2:
        raise InstanceMaskImportError(
            f"Seed-instance mask must be a single-channel raster: {source}"
        )
    if not np.issubdtype(labels.dtype, np.integer):
        raise InstanceMaskImportError(
            f"Seed-instance mask IDs must be integers, not {labels.dtype}: {source}"
        )
    if labels.shape != expected_shape:
        raise InstanceMaskImportError(
            f"Mask is {labels.shape[1]}×{labels.shape[0]}, but the required "
            f"image coordinates are {expected_shape[1]}×{expected_shape[0]}. "
            "Nothing was resized or loaded."
        )
    if bool(np.any(labels < 0)) or bool(
        np.any(labels > np.iinfo(np.uint16).max)
    ):
        raise InstanceMaskImportError(
            "Seed-instance IDs must be nonnegative integers no greater than 65,535."
        )
    return np.ascontiguousarray(labels, dtype=np.uint16)


def _validated_shape(shape: tuple[int, int], name: str) -> tuple[int, int]:
    if len(shape) != 2:
        raise InstanceMaskImportError(f"{name.capitalize()} shape must be 2-D.")
    height, width = (int(shape[0]), int(shape[1]))
    if height <= 0 or width <= 0:
        raise InstanceMaskImportError(
            f"{name.capitalize()} dimensions must be positive."
        )
    return height, width


def _safe_manifest_path(root: Path, value: object, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise InstanceMaskImportError(f"Bundled {name} must be a non-empty path.")
    relative = Path(value)
    if relative.is_absolute():
        raise InstanceMaskImportError(f"Bundled {name} must be relative.")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise InstanceMaskImportError(
            f"Bundled {name} escapes its allowed directory."
        ) from error
    return resolved


def _same_path(first: Path, second: Path) -> bool:
    return str(first).casefold() == str(second).casefold()


def _sha256_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise InstanceMaskImportError(f"Bundled {name} must be a lowercase SHA-256.")
    return value


def _read_file_sha256(path: Path, name: str) -> str:
    try:
        return file_sha256(path)
    except OSError as error:
        raise InstanceMaskImportError(
            f"Could not fingerprint {name} {path}: {error}"
        ) from error


def _positive_int(value: object, name: str) -> int:
    parsed = _nonnegative_int(value, name)
    if parsed <= 0:
        raise InstanceMaskImportError(f"Bundled {name} must be positive.")
    return parsed


def _nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise InstanceMaskImportError(
            f"Bundled {name} must be a nonnegative integer."
        )
    return value


def _scalar_text(values: np.ndarray, name: str) -> str:
    array = np.asarray(values)
    if array.shape != () or array.dtype.kind not in {"U", "S"}:
        raise InstanceMaskImportError(f"NPZ {name} must be scalar text.")
    value = array.item()
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _nonzero_identifier_count(labels: np.ndarray) -> int:
    identifiers = np.unique(labels)
    return int(np.count_nonzero(identifiers))
