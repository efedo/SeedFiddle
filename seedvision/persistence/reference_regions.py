"""Durable, image-bound persistence for painted reference regions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from zipfile import BadZipFile

import numpy as np


REFERENCE_REGION_VERSION = 5
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
class SeedInstanceAnnotation:
    """Reviewed semantic labels attached to one annotated seed identity.

    Coat pattern is mutually exclusive. Conditions are non-exclusive, but an
    explicit ``conditions_reviewed`` bit distinguishes a reviewed sound seed
    from an instance whose condition labels have not yet been considered.
    """

    seed_id: int
    coat_pattern: str | None = None
    conditions: tuple[str, ...] = ()
    conditions_reviewed: bool = False
    shape_reviewed: bool = False
    outline_visibility: str = "unknown"
    pose: str = "unknown"
    hilum_point: tuple[float, float] | None = None
    hilum_direction: tuple[float, float] | None = None
    physical_seed_id: str | None = None
    shape_exclusion_reason: str | None = None
    full_length_visible: bool = False


@dataclass(frozen=True, slots=True)
class ReferenceRegionBundle:
    """The active reference layers in full image coordinates.

    ``physical_edge`` and ``non_edge`` are retained only as a source-compatible
    bridge for version-one callers.  Version-two archives do not persist them:
    physical and non-physical edge supervision is derived deterministically
    from ``annotated_seeds``.
    """

    shape: tuple[int, int]
    background: np.ndarray | None = None
    foreground: np.ndarray | None = None
    other: np.ndarray | None = None
    physical_edge: np.ndarray | None = None
    non_edge: np.ndarray | None = None
    annotated_seeds: np.ndarray | None = None
    annotation_origin: str = "manual"
    seed_annotations: tuple[SeedInstanceAnnotation, ...] = ()
    annotation_species: str = ""


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
        """Atomically save material references and annotated seed identities."""

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
        instances = _instance_labels(bundle.annotated_seeds, (height, width))
        seed_annotations = _validated_seed_annotations(
            bundle.seed_annotations,
            frozenset(int(value) for value in np.unique(instances) if value),
        )
        annotation_species = _validated_identifier_or_empty(
            bundle.annotation_species, "annotation species"
        )

        material = np.zeros((height, width), dtype=np.uint8)
        material[background] = 1
        material[foreground] = 2
        material[other] = 3
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
                    annotated_seeds=instances,
                    annotation_origin=np.asarray(str(bundle.annotation_origin)),
                    seed_annotations_json=np.asarray(
                        json.dumps(
                            [
                                {
                                    "seed_id": annotation.seed_id,
                                    "coat_pattern": annotation.coat_pattern,
                                    "conditions": list(annotation.conditions),
                                    "conditions_reviewed": (
                                        annotation.conditions_reviewed
                                    ),
                                    "shape_reviewed": annotation.shape_reviewed,
                                    "outline_visibility": annotation.outline_visibility,
                                    "pose": annotation.pose,
                                    "hilum_point": (
                                        None
                                        if annotation.hilum_point is None
                                        else list(annotation.hilum_point)
                                    ),
                                    "hilum_direction": (
                                        None
                                        if annotation.hilum_direction is None
                                        else list(annotation.hilum_direction)
                                    ),
                                    "physical_seed_id": annotation.physical_seed_id,
                                    "full_length_visible": annotation.full_length_visible,
                                    "shape_exclusion_reason": (
                                        annotation.shape_exclusion_reason
                                    ),
                                }
                                for annotation in seed_annotations
                            ],
                            separators=(",", ":"),
                            sort_keys=True,
                        )
                    ),
                    annotation_species=np.asarray(annotation_species),
                )
            temporary.replace(destination)
        finally:
            if temporary.exists():
                temporary.unlink()
        return destination

    def backup_and_replace_invalid(
        self,
        image_path: Path | str,
        replacement_shape: tuple[int, int],
    ) -> tuple[Path, Path]:
        """Preserve the canonical sidecar, then replace it with an empty one.

        The backup is completed before :meth:`save` performs its own atomic
        replacement. If creating the new archive fails, the original invalid
        sidecar remains at its canonical path and the completed backup remains
        available for recovery.
        """

        image = Path(image_path).resolve()
        source = self.path_for(image)
        if not source.is_file():
            raise ReferenceRegionError(
                f"Reference-region archive does not exist: {source}"
            )
        shape = _validated_shape(replacement_shape)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup = source.with_name(
            f"{source.name}.invalid-{timestamp}.bak"
        )
        counter = 2
        while backup.exists():
            backup = source.with_name(
                f"{source.name}.invalid-{timestamp}-{counter}.bak"
            )
            counter += 1

        descriptor, temporary_name = tempfile.mkstemp(
            dir=source.parent,
            prefix=f".{source.name}.backup.",
            suffix=".tmp",
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            shutil.copy2(source, temporary)
            temporary.replace(backup)
        finally:
            if temporary.exists():
                temporary.unlink()

        destination = self.save(
            image,
            ReferenceRegionBundle(shape=shape),
        )
        return backup, destination

    def load_if_present(
        self,
        image_path: Path | str,
        expected_shape: tuple[int, int] | None = None,
    ) -> ReferenceRegionBundle | None:
        """Load after validating the image fingerprint and every saved raster.

        ``expected_shape`` is normally the current corrected-image shape.  A
        project master knows only the immutable raw source dimensions, so it
        may pass ``None`` to validate the archive against its own positive,
        internally consistent corrected-coordinate shape instead.
        """

        image = Path(image_path).resolve()
        return self._load_archive(
            image,
            self.path_for(image),
            expected_shape,
            require_current_identity=True,
        )

    def load_project_archive(
        self,
        image_path: Path | str,
        archive_path: Path | str,
        expected_shape: tuple[int, int] | None = None,
    ) -> ReferenceRegionBundle | None:
        """Load the exact sidecar named by a validated project master.

        A project master binds this explicit path to both the source-image and
        sidecar SHA-256 digests.  The archive's historical ``image_identity``
        remains schema-validated but is deliberately not recomputed: that
        identity controls loose-workspace discovery and can differ after moving
        a complete workspace between case-insensitive and case-sensitive hosts.
        """

        return self._load_archive(
            Path(image_path).resolve(),
            Path(archive_path).resolve(),
            expected_shape,
            require_current_identity=False,
        )

    def _load_archive(
        self,
        image: Path,
        source: Path,
        expected_shape: tuple[int, int] | None,
        *,
        require_current_identity: bool,
    ) -> ReferenceRegionBundle | None:
        if not source.is_file():
            return None
        expected = (
            None if expected_shape is None else _validated_shape(expected_shape)
        )
        try:
            with np.load(source, allow_pickle=False) as archive:
                if "version" not in archive.files:
                    raise InvalidReferenceArchive(
                        f"Saved reference archive is missing 'version': {source}"
                    )
                version = _scalar_int(archive["version"], "version")
                if version not in {1, 2, 3, 4, REFERENCE_REGION_VERSION}:
                    raise InvalidReferenceArchive(
                        f"Unsupported reference-region version {version} in {source}."
                    )
                required = [
                    "version",
                    "image_identity",
                    "image_sha256",
                    "height",
                    "width",
                    "material",
                    "annotated_seeds",
                    "annotation_origin",
                ]
                if version == 1:
                    required.append("boundary")
                for name in required:
                    if name not in archive.files:
                        raise InvalidReferenceArchive(
                            f"Saved reference archive is missing {name!r}: {source}"
                        )
                saved_identity = _scalar_text(
                    archive["image_identity"], "image identity"
                )
                if not saved_identity or "\x00" in saved_identity:
                    raise InvalidReferenceArchive(
                        f"Invalid image identity metadata in {source}."
                    )
                if (
                    require_current_identity
                    and saved_identity != self.image_identity(image)
                ):
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
                saved_shape = _validated_shape(
                    (
                        _scalar_int(archive["height"], "height"),
                        _scalar_int(archive["width"], "width"),
                    )
                )
                if expected is not None and saved_shape != expected:
                    raise InvalidReferenceArchive(
                        "Saved reference dimensions "
                        f"{saved_shape[1]}×{saved_shape[0]} do not match the image "
                        f"{expected[1]}×{expected[0]}: {source}"
                    )
                material = np.asarray(archive["material"])
                legacy_boundary = (
                    np.asarray(archive["boundary"])
                    if version == 1
                    else None
                )
                instances = np.asarray(archive["annotated_seeds"])
                annotation_origin = _scalar_text(
                    archive["annotation_origin"], "annotation origin"
                )
                if version >= 3:
                    for name in ("seed_annotations_json", "annotation_species"):
                        if name not in archive.files:
                            raise InvalidReferenceArchive(
                                f"Saved reference archive is missing {name!r}: {source}"
                            )
                    seed_annotations_text = _scalar_text(
                        archive["seed_annotations_json"],
                        "seed annotations",
                    )
                    annotation_species = _scalar_text(
                        archive["annotation_species"],
                        "annotation species",
                    )
                else:
                    seed_annotations_text = "[]"
                    annotation_species = ""
        except ImageFingerprintMismatch:
            raise
        except InvalidReferenceArchive:
            raise
        except (OSError, ValueError, TypeError, EOFError, BadZipFile) as error:
            raise InvalidReferenceArchive(
                f"Could not read saved reference regions from {source}: {error}"
            ) from error

        shape = saved_shape
        material = _categorical_raster(
            material, shape, _MATERIAL_CLASSES, "material", source
        )
        if legacy_boundary is not None:
            # Validate old data so a corrupt version-one file is not silently
            # accepted, but intentionally discard the retired painted layer.
            _categorical_raster(
                legacy_boundary, shape, _BOUNDARY_CLASSES, "boundary", source
            )
        try:
            instances = _instance_labels(instances, shape)
        except ReferenceRegionError as error:
            raise InvalidReferenceArchive(
                f"Invalid annotated-seed raster in {source}: {error}"
            ) from error
        try:
            decoded_annotations = json.loads(seed_annotations_text)
        except (TypeError, ValueError) as error:
            raise InvalidReferenceArchive(
                f"Invalid seed-annotation metadata in {source}: {error}"
            ) from error
        try:
            seed_annotations = _seed_annotations_from_payload(
                decoded_annotations,
                frozenset(int(value) for value in np.unique(instances) if value),
                version=version,
            )
            annotation_species = _validated_identifier_or_empty(
                annotation_species, "annotation species"
            )
        except ReferenceRegionError as error:
            raise InvalidReferenceArchive(
                f"Invalid seed-annotation metadata in {source}: {error}"
            ) from error
        return ReferenceRegionBundle(
            shape=shape,
            background=_nonempty(material == 1),
            foreground=_nonempty(material == 2),
            other=_nonempty(material == 3),
            physical_edge=None,
            non_edge=None,
            annotated_seeds=_nonempty(instances),
            annotation_origin=annotation_origin or "manual",
            seed_annotations=seed_annotations,
            annotation_species=annotation_species,
        )


_TRAIT_IDENTIFIER = re.compile(r"[a-z][a-z0-9_]{0,63}")
_PHYSICAL_SEED_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}")
_OUTLINE_VISIBILITIES = frozenset(
    ("unknown", "complete", "partly_occluded", "image_cutoff", "uncertain")
)
_SEED_POSES = frozenset(("unknown", "flat", "oblique", "side", "uncertain"))


def _validated_identifier_or_empty(value: object, name: str) -> str:
    text = str(value or "")
    if text and _TRAIT_IDENTIFIER.fullmatch(text) is None:
        raise ReferenceRegionError(
            f"{name.capitalize()} must be an empty string or a stable lowercase identifier."
        )
    return text


def _validated_seed_annotations(
    values: object,
    extant_seed_ids: frozenset[int],
) -> tuple[SeedInstanceAnnotation, ...]:
    try:
        annotations = tuple(values or ())
    except TypeError as error:
        raise ReferenceRegionError("Seed annotations must be an iterable.") from error
    normalized: list[SeedInstanceAnnotation] = []
    seen: set[int] = set()
    for value in annotations:
        if not isinstance(value, SeedInstanceAnnotation):
            raise ReferenceRegionError(
                "Seed annotations must contain SeedInstanceAnnotation values."
            )
        if isinstance(value.seed_id, bool) or not isinstance(
            value.seed_id, (int, np.integer)
        ):
            raise ReferenceRegionError("Seed annotation IDs must be integers.")
        seed_id = int(value.seed_id)
        if seed_id <= 0 or seed_id > np.iinfo(np.uint16).max:
            raise ReferenceRegionError("Seed annotation IDs must fit in unsigned 16 bits.")
        if seed_id in seen:
            raise ReferenceRegionError(f"Duplicate semantic annotation for seed {seed_id}.")
        if seed_id not in extant_seed_ids:
            raise ReferenceRegionError(
                f"Semantic annotation refers to absent seed ID {seed_id}."
            )
        seen.add(seed_id)
        if value.coat_pattern is not None and not isinstance(
            value.coat_pattern, str
        ):
            raise ReferenceRegionError(
                "Seed coat pattern must be a string or null."
            )
        coat_pattern = (
            None
            if value.coat_pattern in {None, ""}
            else _validated_identifier_or_empty(value.coat_pattern, "coat pattern")
        )
        if not isinstance(value.conditions, (tuple, list)) or not all(
            isinstance(condition, str) for condition in value.conditions
        ):
            raise ReferenceRegionError(
                "Seed annotation conditions must be a sequence of strings."
            )
        validated_conditions = tuple(
            _validated_identifier_or_empty(condition, "condition")
            for condition in value.conditions
        )
        if len(set(validated_conditions)) != len(validated_conditions):
            raise ReferenceRegionError(
                f"Seed {seed_id} contains duplicate condition identifiers."
            )
        conditions = tuple(sorted(validated_conditions))
        if "" in conditions:
            raise ReferenceRegionError("Condition identifiers cannot be empty.")
        if not isinstance(value.conditions_reviewed, bool):
            raise ReferenceRegionError("conditions_reviewed must be Boolean.")
        if conditions and not value.conditions_reviewed:
            raise ReferenceRegionError(
                "Selected seed conditions require conditions_reviewed=true."
            )
        if not isinstance(value.shape_reviewed, bool):
            raise ReferenceRegionError("shape_reviewed must be Boolean.")
        if not isinstance(value.full_length_visible, bool):
            raise ReferenceRegionError("full_length_visible must be Boolean.")
        outline_visibility = str(value.outline_visibility)
        if outline_visibility not in _OUTLINE_VISIBILITIES:
            raise ReferenceRegionError("Seed outline visibility is unsupported.")
        pose = str(value.pose)
        if pose not in _SEED_POSES:
            raise ReferenceRegionError("Seed pose is unsupported.")
        hilum_point = _optional_finite_pair(value.hilum_point, "hilum point")
        hilum_direction = _optional_finite_pair(
            value.hilum_direction, "hilum direction"
        )
        if hilum_direction is not None and np.hypot(*hilum_direction) <= 1e-9:
            raise ReferenceRegionError("Hilum direction cannot be a zero vector.")
        physical_seed_id = None
        if value.physical_seed_id not in {None, ""}:
            physical_seed_id = str(value.physical_seed_id)
            if _PHYSICAL_SEED_IDENTIFIER.fullmatch(physical_seed_id) is None:
                raise ReferenceRegionError(
                    "Physical seed ID must be a stable 1--128 character identifier."
                )
        exclusion = None
        if value.shape_exclusion_reason not in {None, ""}:
            exclusion = str(value.shape_exclusion_reason).strip()
            if not exclusion or len(exclusion) > 160 or "\x00" in exclusion:
                raise ReferenceRegionError(
                    "Shape exclusion reason must contain at most 160 characters."
                )
        if value.shape_reviewed and (
            outline_visibility == "unknown" or pose == "unknown"
        ):
            missing = []
            if outline_visibility == "unknown":
                missing.append("Outline")
            if pose == "unknown":
                missing.append("Pose")
            raise ReferenceRegionError(
                f"Seed {seed_id} is marked for shape modelling, but "
                + " and ".join(missing)
                + (" is Unknown. " if len(missing) == 1 else " are Unknown. ")
                + "In Annotate seed instances, select this seed and "
                "choose explicit values, or clear ‘Use for shape modelling’."
            )
        normalized.append(
            SeedInstanceAnnotation(
                seed_id=seed_id,
                coat_pattern=coat_pattern,
                conditions=conditions,
                conditions_reviewed=value.conditions_reviewed,
                shape_reviewed=value.shape_reviewed,
                outline_visibility=outline_visibility,
                pose=pose,
                hilum_point=hilum_point,
                hilum_direction=hilum_direction,
                physical_seed_id=physical_seed_id,
                shape_exclusion_reason=exclusion,
                full_length_visible=value.full_length_visible,
            )
        )
    return tuple(sorted(normalized, key=lambda item: item.seed_id))


def _seed_annotations_from_payload(
    payload: object,
    extant_seed_ids: frozenset[int],
    *,
    version: int = REFERENCE_REGION_VERSION,
) -> tuple[SeedInstanceAnnotation, ...]:
    if not isinstance(payload, list):
        raise ReferenceRegionError("Seed annotations must be a JSON list.")
    values: list[SeedInstanceAnnotation] = []
    for item in payload:
        legacy_keys = {
            "seed_id",
            "coat_pattern",
            "conditions",
            "conditions_reviewed",
        }
        current_keys = legacy_keys | {
            "shape_reviewed",
            "outline_visibility",
            "pose",
            "hilum_point",
            "hilum_direction",
            "physical_seed_id",
            "shape_exclusion_reason",
        }
        expected_keys = current_keys if version >= 4 else legacy_keys
        if version >= 5:
            expected_keys = expected_keys | {"full_length_visible"}
        if not isinstance(item, dict) or set(item) != expected_keys:
            raise ReferenceRegionError("A seed annotation has an invalid schema.")
        if not isinstance(item["conditions"], list) or not all(
            isinstance(condition, str) for condition in item["conditions"]
        ):
            raise ReferenceRegionError("Seed annotation conditions must be strings.")
        if not isinstance(item["conditions_reviewed"], bool):
            raise ReferenceRegionError("conditions_reviewed must be Boolean.")
        coat = item["coat_pattern"]
        if coat is not None and not isinstance(coat, str):
            raise ReferenceRegionError("Seed coat pattern must be a string or null.")
        values.append(
            SeedInstanceAnnotation(
                seed_id=item["seed_id"],
                coat_pattern=coat,
                conditions=tuple(item["conditions"]),
                conditions_reviewed=item["conditions_reviewed"],
                shape_reviewed=(False if version < 4 else item["shape_reviewed"]),
                full_length_visible=(False if version < 5 else item["full_length_visible"]),
                outline_visibility=(
                    "unknown" if version < 4 else item["outline_visibility"]
                ),
                pose="unknown" if version < 4 else item["pose"],
                hilum_point=(
                    None
                    if version < 4 or item["hilum_point"] is None
                    else tuple(item["hilum_point"])
                ),
                hilum_direction=(
                    None
                    if version < 4 or item["hilum_direction"] is None
                    else tuple(item["hilum_direction"])
                ),
                physical_seed_id=(
                    None if version < 4 else item["physical_seed_id"]
                ),
                shape_exclusion_reason=(
                    None if version < 4 else item["shape_exclusion_reason"]
                ),
            )
        )
    return _validated_seed_annotations(values, extant_seed_ids)


def _optional_finite_pair(value: object, name: str) -> tuple[float, float] | None:
    if value is None:
        return None
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ReferenceRegionError(f"{name.capitalize()} must contain X and Y.")
    pair = (float(value[0]), float(value[1]))
    if not np.all(np.isfinite(pair)):
        raise ReferenceRegionError(f"{name.capitalize()} must be finite.")
    return pair


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
