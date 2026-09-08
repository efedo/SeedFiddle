"""Portable, versioned master files for Seed Fiddle project analyses.

The JSON master deliberately stores only compact configuration and file
references.  Full-resolution images, categorical reference rasters, manual
centres, GPU tensors, and node caches remain in their existing files/caches.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from hashlib import sha256
import json
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import tempfile
from types import MappingProxyType
from typing import Iterable, Mapping

from seedvision.persistence.analysis_settings import (
    AnalysisSettingsError,
    AnalysisSettingsProfile,
    analysis_settings_profile_from_payload,
    analysis_settings_profile_to_payload,
)
from seedvision.persistence.instance_masks import (
    InstanceMaskImportError,
    read_source_raster_shape,
)
from seedvision.persistence.manual_seed_centres import (
    ManualSeedCentreStore,
    ManualSeedCentreStoreError,
)
from seedvision.persistence.reference_regions import (
    ReferenceRegionError,
    ReferenceRegionStore,
    file_sha256,
)
from seedvision.reference_library.contracts import (
    BiologicalContext,
    SpeciesLibraryPin,
)


PROJECT_ANALYSIS_FORMAT = "seedfiddle-project"
PROJECT_ANALYSIS_VERSION = 2
PROJECT_ANALYSIS_EXTENSION = ".seedfiddle-project.json"
PROJECT_ANALYSIS_DEFAULT_NAME = "analysis.seedfiddle-project.json"
PROJECT_ANALYSIS_MAX_BYTES = 16 * 1024 * 1024
PROJECT_ANALYSIS_MAX_IMAGES = 10_000

PROJECT_RELATIVE = "project_relative"
EXTERNAL_ABSOLUTE = "external_absolute"
REFERENCE_REGIONS_SIDECAR = "reference_regions"
MANUAL_SEED_CENTRES_SIDECAR = "manual_seed_centres"
_SIDECAR_KINDS = (
    REFERENCE_REGIONS_SIDECAR,
    MANUAL_SEED_CENTRES_SIDECAR,
)
_SHA256_LENGTH = 64


class ProjectAnalysisError(ValueError):
    """Base error for project-master persistence."""


class InvalidProjectAnalysis(ProjectAnalysisError):
    """A project master is malformed, unsafe, or uses an unsupported schema."""


class ProjectAnalysisIOError(ProjectAnalysisError):
    """A valid project master could not be read or written."""


class ProjectFileStatus(str, Enum):
    """Availability of a file referenced by an otherwise valid master."""

    AVAILABLE = "available"
    MISSING = "missing"
    FINGERPRINT_MISMATCH = "fingerprint_mismatch"
    DIMENSION_MISMATCH = "dimension_mismatch"
    INVALID_SIDECAR = "invalid_sidecar"
    UNREADABLE = "unreadable"
    UNCHECKED = "unchecked"


@dataclass(frozen=True, slots=True)
class ProjectImageSpec:
    """Current image state from which a fresh master record is captured."""

    path: Path | str
    source_shape: tuple[int, int]
    include_reference_regions: bool = True
    include_manual_seed_centres: bool = True
    reference_regions_path: Path | str | None = None
    manual_seed_centres_path: Path | str | None = None
    biological_context: BiologicalContext | None = None
    capture_group_id: str | None = None


@dataclass(frozen=True, slots=True)
class ProjectFileReference:
    """A source image locator and its byte-exact capture fingerprint."""

    location: str
    path: str
    sha256: str


@dataclass(frozen=True, slots=True)
class ProjectSidecarReference:
    """A compact project-relative sidecar referenced by one image."""

    kind: str
    path: str
    sha256: str


@dataclass(frozen=True, slots=True)
class ProjectImageRecord:
    """One ordered image entry stored in the master document."""

    identifier: str
    source: ProjectFileReference
    source_shape: tuple[int, int]
    sidecars: tuple[ProjectSidecarReference, ...] = ()
    biological_context: BiologicalContext | None = None
    capture_group_id: str | None = None

    def sidecar(self, kind: str) -> ProjectSidecarReference | None:
        return next((item for item in self.sidecars if item.kind == kind), None)


@dataclass(frozen=True, slots=True)
class ProjectUiState:
    """Small presentation state intentionally separate from analysis settings."""

    node_positions: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    bundle_cables: bool = False
    route_around_nodes: bool = False
    selected_node: str | None = None
    selected_overlay: str | None = None

    def __post_init__(self) -> None:
        positions = {
            str(identifier): tuple(coordinates)
            for identifier, coordinates in dict(self.node_positions).items()
        }
        object.__setattr__(self, "node_positions", MappingProxyType(positions))


@dataclass(frozen=True, slots=True)
class ProjectAnalysisDocument:
    """Immutable compact content of one version-two project master."""

    analysis_settings: AnalysisSettingsProfile
    images: tuple[ProjectImageRecord, ...]
    species: str | None = None
    selected_image_id: str | None = None
    ui_state: ProjectUiState | None = None
    species_library: SpeciesLibraryPin | None = None
    biological_context: BiologicalContext | None = None
    capture_group_id: str | None = None


@dataclass(frozen=True, slots=True)
class ResolvedProjectSidecar:
    reference: ProjectSidecarReference
    path: Path
    status: ProjectFileStatus
    actual_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class ResolvedProjectImage:
    record: ProjectImageRecord
    path: Path
    status: ProjectFileStatus
    actual_sha256: str | None
    sidecars: tuple[ResolvedProjectSidecar, ...]

    def sidecar(self, kind: str) -> ResolvedProjectSidecar | None:
        return next(
            (item for item in self.sidecars if item.reference.kind == kind), None
        )


@dataclass(frozen=True, slots=True)
class ProjectFileIssue:
    image_id: str
    role: str
    path: Path
    status: ProjectFileStatus
    expected_sha256: str
    actual_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class ProjectAnalysisLoadResult:
    document: ProjectAnalysisDocument
    images: tuple[ResolvedProjectImage, ...]
    issues: tuple[ProjectFileIssue, ...]

    @property
    def selected_image(self) -> ResolvedProjectImage | None:
        selected = self.document.selected_image_id
        return next(
            (item for item in self.images if item.record.identifier == selected),
            None,
        )


class ProjectAnalysisStore:
    """Capture, atomically save, and safely resolve project master documents."""

    def __init__(self, project_root: Path | str) -> None:
        self.project_root = Path(project_root).resolve()
        self.default_path = (
            self.project_root / "projects" / PROJECT_ANALYSIS_DEFAULT_NAME
        )

    def capture(
        self,
        *,
        analysis_settings: AnalysisSettingsProfile,
        images: Iterable[ProjectImageSpec],
        species: str | None = None,
        selected_image: Path | str | None = None,
        ui_state: ProjectUiState | None = None,
        species_library: SpeciesLibraryPin | None = None,
        biological_context: BiologicalContext | None = None,
        capture_group_id: str | None = None,
    ) -> ProjectAnalysisDocument:
        """Fingerprint current images and discover their existing compact sidecars."""

        # Round-trip through the settings module's public seam so an invalid or
        # hand-built profile cannot enter a master document.
        try:
            settings = analysis_settings_profile_from_payload(
                analysis_settings_profile_to_payload(analysis_settings)
            )
        except AnalysisSettingsError as error:
            raise InvalidProjectAnalysis(
                f"Invalid embedded analysis-settings profile: {error}"
            ) from error
        records: list[ProjectImageRecord] = []
        resolved_sources: dict[str, str] = {}
        reference_store = ReferenceRegionStore(self.project_root)
        centre_store = ManualSeedCentreStore(self.project_root)
        image_specs = tuple(images)
        if len(image_specs) > PROJECT_ANALYSIS_MAX_IMAGES:
            raise InvalidProjectAnalysis(
                f"Project masters support at most {PROJECT_ANALYSIS_MAX_IMAGES:,} images."
            )
        for index, spec in enumerate(image_specs):
            if not isinstance(spec, ProjectImageSpec):
                raise InvalidProjectAnalysis(
                    f"Image entry {index} must be a ProjectImageSpec."
                )
            source_path = Path(spec.path).resolve()
            if not source_path.is_file():
                raise InvalidProjectAnalysis(
                    f"Project source image does not exist: {source_path}"
                )
            source_shape = _validated_shape(
                spec.source_shape, f"image entry {index} source shape"
            )
            try:
                decoded_shape = read_source_raster_shape(source_path)
            except (InstanceMaskImportError, OSError) as error:
                raise InvalidProjectAnalysis(
                    f"Could not validate source-image dimensions for {source_path}: "
                    f"{error}"
                ) from error
            if decoded_shape != source_shape:
                raise InvalidProjectAnalysis(
                    f"Image entry {index} declares {source_shape[1]}×{source_shape[0]}, "
                    f"but {source_path.name} decodes as "
                    f"{decoded_shape[1]}×{decoded_shape[0]}."
                )
            source_reference = self._capture_source(source_path)
            identifier = _image_identifier(
                source_reference.location, source_reference.path
            )
            path_key = _path_key(source_path)
            if path_key in resolved_sources:
                raise InvalidProjectAnalysis(
                    f"Project source image is listed more than once: {source_path}"
                )
            resolved_sources[path_key] = identifier
            sidecars: list[ProjectSidecarReference] = []
            candidates = (
                (
                    REFERENCE_REGIONS_SIDECAR,
                    spec.include_reference_regions,
                    spec.reference_regions_path,
                ),
                (
                    MANUAL_SEED_CENTRES_SIDECAR,
                    spec.include_manual_seed_centres,
                    spec.manual_seed_centres_path,
                ),
            )
            for kind, include, override in candidates:
                if not isinstance(include, bool):
                    raise InvalidProjectAnalysis(
                        f"Image entry {index} {kind} inclusion must be Boolean."
                    )
                if not include and override is not None:
                    raise InvalidProjectAnalysis(
                        f"Image entry {index} cannot supply a {kind} path while "
                        "excluding that sidecar."
                    )
                if override is None:
                    sidecar_path = (
                        reference_store.path_for(source_path)
                        if kind == REFERENCE_REGIONS_SIDECAR
                        else centre_store.path_for(source_path)
                    )
                else:
                    sidecar_path = self._explicit_sidecar_path(
                        override, kind, index
                    )
                if override is not None and not sidecar_path.is_file():
                    raise InvalidProjectAnalysis(
                        f"Explicit {kind} sidecar does not exist for "
                        f"{source_path.name}: {sidecar_path}"
                    )
                if include and sidecar_path.is_file():
                    try:
                        if kind == REFERENCE_REGIONS_SIDECAR:
                            restored = reference_store.load_project_archive(
                                source_path, sidecar_path, None
                            )
                        else:
                            restored = centre_store.load_project_archive(
                                source_path, sidecar_path, source_shape
                            )
                    except (
                        ReferenceRegionError,
                        ManualSeedCentreStoreError,
                        OSError,
                    ) as error:
                        raise InvalidProjectAnalysis(
                            f"Cannot include invalid {kind} sidecar for "
                            f"{source_path.name}: {error}"
                        ) from error
                    if restored is None:
                        raise InvalidProjectAnalysis(
                            f"The {kind} sidecar disappeared while capturing "
                            f"{source_path.name}."
                        )
                    sidecars.append(self._capture_sidecar(kind, sidecar_path))
            records.append(
                ProjectImageRecord(
                    identifier=identifier,
                    source=source_reference,
                    source_shape=source_shape,
                    sidecars=tuple(sidecars),
                    biological_context=_validated_biological_context(
                        spec.biological_context, f"image entry {index}"
                    ),
                    capture_group_id=_validated_context_identifier(
                        spec.capture_group_id, f"image entry {index} capture group"
                    ),
                )
            )
        selected_id = None
        if selected_image is not None:
            selected_path = Path(selected_image).resolve()
            selected_id = resolved_sources.get(_path_key(selected_path))
            if selected_id is None:
                raise InvalidProjectAnalysis(
                    "The selected image must be one of the captured project images."
                )
        document = ProjectAnalysisDocument(
            analysis_settings=settings,
            images=tuple(records),
            species=_validated_optional_text(species, "project species"),
            selected_image_id=selected_id,
            ui_state=ui_state,
            species_library=_validated_library_pin(species_library),
            biological_context=_validated_biological_context(
                biological_context, "project"
            ),
            capture_group_id=_validated_context_identifier(
                capture_group_id, "project capture group"
            ),
        )
        # The serializer is also the single strict document-level validator.
        _document_payload(document)
        return document

    def _explicit_sidecar_path(
        self, value: Path | str, kind: str, image_index: int
    ) -> Path:
        candidate = Path(value)
        resolved = (
            candidate.resolve()
            if candidate.is_absolute()
            else (self.project_root / candidate).resolve()
        )
        try:
            resolved.relative_to(self.project_root)
        except ValueError as error:
            raise InvalidProjectAnalysis(
                f"Image entry {image_index} {kind} sidecar escapes the workspace "
                f"root: {resolved}"
            ) from error
        return resolved

    def capture_and_save(
        self,
        *,
        analysis_settings: AnalysisSettingsProfile,
        images: Iterable[ProjectImageSpec],
        species: str | None = None,
        selected_image: Path | str | None = None,
        ui_state: ProjectUiState | None = None,
        species_library: SpeciesLibraryPin | None = None,
        biological_context: BiologicalContext | None = None,
        capture_group_id: str | None = None,
        destination: Path | str | None = None,
    ) -> Path:
        document = self.capture(
            analysis_settings=analysis_settings,
            images=images,
            species=species,
            selected_image=selected_image,
            ui_state=ui_state,
            species_library=species_library,
            biological_context=biological_context,
            capture_group_id=capture_group_id,
        )
        return self.save(document, destination)

    def save(
        self,
        document: ProjectAnalysisDocument,
        destination: Path | str | None = None,
    ) -> Path:
        """Atomically replace a master after fully validating and encoding it."""

        if not isinstance(document, ProjectAnalysisDocument):
            raise InvalidProjectAnalysis(
                "Project master save requires a ProjectAnalysisDocument."
            )
        payload = _document_payload(document)
        try:
            encoded = (
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8")
        except (TypeError, ValueError) as error:
            raise InvalidProjectAnalysis(
                f"Project master contains a non-JSON value: {error}"
            ) from error
        if len(encoded) > PROJECT_ANALYSIS_MAX_BYTES:
            raise InvalidProjectAnalysis(
                "Project master exceeds the compact 16 MiB format limit."
            )
        target = _master_path(destination or self.default_path)
        descriptor: int | None = None
        temporary: Path | None = None
        operation_error: OSError | None = None
        cleanup_error: OSError | None = None
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary_name = tempfile.mkstemp(
                dir=target.parent,
                prefix=f".{target.name}.",
                suffix=".tmp",
            )
            temporary = Path(temporary_name)
            with os.fdopen(descriptor, "wb") as stream:
                descriptor = None
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        except OSError as error:
            operation_error = error
        finally:
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError as error:
                    cleanup_error = error
            if temporary is not None and temporary.exists():
                try:
                    temporary.unlink()
                except OSError as error:
                    cleanup_error = cleanup_error or error
        if operation_error is not None:
            detail = (
                f"; temporary-file cleanup also failed: {cleanup_error}"
                if cleanup_error is not None
                else ""
            )
            raise ProjectAnalysisIOError(
                f"Could not atomically save project master {target}: "
                f"{operation_error}{detail}"
            ) from operation_error
        if cleanup_error is not None:
            raise ProjectAnalysisIOError(
                f"Project master was saved to {target}, but temporary-file cleanup "
                f"failed: {cleanup_error}"
            ) from cleanup_error
        return target

    def load(
        self,
        source: Path | str | None = None,
        *,
        verify_files: bool = True,
    ) -> ProjectAnalysisLoadResult:
        """Parse the master strictly, then report missing/changed files as issues."""

        path = _master_path(source or self.default_path)
        try:
            if path.stat().st_size > PROJECT_ANALYSIS_MAX_BYTES:
                raise InvalidProjectAnalysis(
                    "Project master exceeds the compact 16 MiB format limit."
                )
            text = path.read_text(encoding="utf-8")
        except InvalidProjectAnalysis:
            raise
        except (OSError, UnicodeError) as error:
            raise ProjectAnalysisIOError(
                f"Could not read project master {path}: {error}"
            ) from error
        try:
            payload = json.loads(
                text,
                object_pairs_hook=_unique_object,
                parse_constant=_invalid_json_constant,
            )
        except InvalidProjectAnalysis:
            raise
        except json.JSONDecodeError as error:
            raise InvalidProjectAnalysis(
                f"Project master is not valid JSON: {path}: {error}"
            ) from error
        except RecursionError as error:
            raise InvalidProjectAnalysis(
                f"Project master JSON is nested too deeply: {path}."
            ) from error
        document = _document_from_payload(payload)
        return self.resolve(document, verify_files=verify_files)

    def resolve(
        self,
        document: ProjectAnalysisDocument,
        *,
        verify_files: bool = True,
    ) -> ProjectAnalysisLoadResult:
        """Resolve safe relative locators against this store's current root."""

        # Revalidate documents supplied directly rather than trusting dataclass
        # construction to have enforced the on-disk schema.
        document = _document_from_payload(_document_payload(document))
        resolved_images: list[ResolvedProjectImage] = []
        issues: list[ProjectFileIssue] = []
        reference_store = ReferenceRegionStore(self.project_root)
        centre_store = ManualSeedCentreStore(self.project_root)
        for record in document.images:
            source_path = self._resolve_source(record.source)
            if verify_files and not _source_locator_is_native(record.source):
                status, actual = ProjectFileStatus.UNREADABLE, None
            else:
                status, actual = _file_status(
                    source_path, record.source.sha256, verify_files
                )
            if status is ProjectFileStatus.AVAILABLE:
                try:
                    decoded_shape = read_source_raster_shape(source_path)
                except (InstanceMaskImportError, OSError):
                    status = ProjectFileStatus.UNREADABLE
                else:
                    if decoded_shape != record.source_shape:
                        status = ProjectFileStatus.DIMENSION_MISMATCH
            if status not in {
                ProjectFileStatus.AVAILABLE,
                ProjectFileStatus.UNCHECKED,
            }:
                issues.append(
                    ProjectFileIssue(
                        image_id=record.identifier,
                        role="source_image",
                        path=source_path,
                        status=status,
                        expected_sha256=record.source.sha256,
                        actual_sha256=actual,
                    )
                )
            resolved_sidecars: list[ResolvedProjectSidecar] = []
            for sidecar in record.sidecars:
                sidecar_path = _resolve_project_relative(
                    self.project_root, sidecar.path, f"{sidecar.kind} path"
                )
                sidecar_status, sidecar_actual = _file_status(
                    sidecar_path, sidecar.sha256, verify_files
                )
                if (
                    verify_files
                    and sidecar_status is ProjectFileStatus.AVAILABLE
                ):
                    if status is not ProjectFileStatus.AVAILABLE:
                        sidecar_status = ProjectFileStatus.INVALID_SIDECAR
                    else:
                        sidecar_status = _validated_sidecar_status(
                            kind=sidecar.kind,
                            path=sidecar_path,
                            source_path=source_path,
                            source_shape=record.source_shape,
                            reference_store=reference_store,
                            centre_store=centre_store,
                        )
                resolved_sidecars.append(
                    ResolvedProjectSidecar(
                        reference=sidecar,
                        path=sidecar_path,
                        status=sidecar_status,
                        actual_sha256=sidecar_actual,
                    )
                )
                if sidecar_status not in {
                    ProjectFileStatus.AVAILABLE,
                    ProjectFileStatus.UNCHECKED,
                }:
                    issues.append(
                        ProjectFileIssue(
                            image_id=record.identifier,
                            role=sidecar.kind,
                            path=sidecar_path,
                            status=sidecar_status,
                            expected_sha256=sidecar.sha256,
                            actual_sha256=sidecar_actual,
                        )
                    )
            resolved_images.append(
                ResolvedProjectImage(
                    record=record,
                    path=source_path,
                    status=status,
                    actual_sha256=actual,
                    sidecars=tuple(resolved_sidecars),
                )
            )
        return ProjectAnalysisLoadResult(
            document=document,
            images=tuple(resolved_images),
            issues=tuple(issues),
        )

    def _capture_source(self, path: Path) -> ProjectFileReference:
        try:
            relative = path.relative_to(self.project_root)
        except ValueError:
            location = EXTERNAL_ABSOLUTE
            stored_path = path.as_posix()
        else:
            location = PROJECT_RELATIVE
            stored_path = _canonical_relative(relative, "source image path")
        try:
            digest = file_sha256(path)
        except OSError as error:
            raise InvalidProjectAnalysis(
                f"Could not fingerprint source image {path}: {error}"
            ) from error
        return ProjectFileReference(location, stored_path, digest)

    def _capture_sidecar(
        self, kind: str, path: Path
    ) -> ProjectSidecarReference:
        try:
            relative = path.resolve().relative_to(self.project_root)
        except ValueError as error:
            raise InvalidProjectAnalysis(
                f"Project sidecar escapes the workspace root: {path}"
            ) from error
        try:
            digest = file_sha256(path)
        except OSError as error:
            raise InvalidProjectAnalysis(
                f"Could not fingerprint {kind} sidecar {path}: {error}"
            ) from error
        return ProjectSidecarReference(
            kind=kind,
            path=_canonical_relative(relative, f"{kind} path"),
            sha256=digest,
        )

    def _resolve_source(self, reference: ProjectFileReference) -> Path:
        if reference.location == PROJECT_RELATIVE:
            return _resolve_project_relative(
                self.project_root, reference.path, "source image path"
            )
        if reference.location != EXTERNAL_ABSOLUTE:
            raise InvalidProjectAnalysis(
                f"Unknown source-image location {reference.location!r}."
            )
        return _external_path(reference.path)


def _document_payload(document: ProjectAnalysisDocument) -> dict[str, object]:
    try:
        settings_payload = analysis_settings_profile_to_payload(
            document.analysis_settings
        )
    except AnalysisSettingsError as error:
        raise InvalidProjectAnalysis(
            f"Invalid embedded analysis-settings profile: {error}"
        ) from error
    known_nodes = {node.identifier for node in document.analysis_settings.nodes}
    species = _validated_optional_text(document.species, "project species")
    if not isinstance(document.images, tuple):
        raise InvalidProjectAnalysis(
            "Project document images must be an immutable tuple."
        )
    if len(document.images) > PROJECT_ANALYSIS_MAX_IMAGES:
        raise InvalidProjectAnalysis(
            f"Project masters support at most {PROJECT_ANALYSIS_MAX_IMAGES:,} images."
        )
    images: list[dict[str, object]] = []
    image_ids: set[str] = set()
    source_keys: set[tuple[str, str]] = set()
    for index, record in enumerate(document.images):
        if not isinstance(record, ProjectImageRecord):
            raise InvalidProjectAnalysis(
                f"Project image {index} must be a ProjectImageRecord."
            )
        source = record.source
        if not isinstance(source, ProjectFileReference):
            raise InvalidProjectAnalysis(f"Project image {index} has no valid source.")
        _validated_digest(source.sha256, f"image {index} SHA-256")
        _validated_source_locator(source.location, source.path)
        expected_id = _image_identifier(source.location, source.path)
        if record.identifier != expected_id:
            raise InvalidProjectAnalysis(
                f"Project image {index} identifier does not match its locator."
            )
        if record.identifier in image_ids:
            raise InvalidProjectAnalysis("Project image identifiers must be unique.")
        image_ids.add(record.identifier)
        source_key = (source.location, source.path)
        if source_key in source_keys:
            raise InvalidProjectAnalysis("Project source locators must be unique.")
        source_keys.add(source_key)
        sidecars: dict[str, object] = {}
        for sidecar in record.sidecars:
            if not isinstance(sidecar, ProjectSidecarReference):
                raise InvalidProjectAnalysis(
                    f"Project image {index} has an invalid sidecar record."
                )
            if sidecar.kind not in _SIDECAR_KINDS or sidecar.kind in sidecars:
                raise InvalidProjectAnalysis(
                    f"Project image {index} has an invalid or duplicate sidecar kind."
                )
            _validated_relative(sidecar.path, f"{sidecar.kind} path")
            _validated_digest(sidecar.sha256, f"{sidecar.kind} SHA-256")
            sidecars[sidecar.kind] = {
                "path": sidecar.path,
                "sha256": sidecar.sha256,
            }
        height, width = _validated_shape(
            record.source_shape, f"image {index} source shape"
        )
        images.append(
            {
                "id": record.identifier,
                "source": {"location": source.location, "path": source.path},
                "image_sha256": source.sha256,
                "source_shape": [height, width],
                "sidecars": sidecars,
                "biological_context": _biological_context_payload(
                    _validated_biological_context(
                        record.biological_context, f"image {index}"
                    )
                ),
                "capture_group_id": _validated_context_identifier(
                    record.capture_group_id, f"image {index} capture group"
                ),
            }
        )
    selected = document.selected_image_id
    if selected is not None and selected not in image_ids:
        raise InvalidProjectAnalysis(
            "Selected image identifier is not present in the project image list."
        )
    payload: dict[str, object] = {
        "format": PROJECT_ANALYSIS_FORMAT,
        "version": PROJECT_ANALYSIS_VERSION,
        "analysis_settings": settings_payload,
        "project": {
            "species": species,
            "selected_image": selected,
            "species_library": _library_pin_payload(
                _validated_library_pin(document.species_library)
            ),
            "biological_context": _biological_context_payload(
                _validated_biological_context(
                    document.biological_context, "project"
                )
            ),
            "capture_group_id": _validated_context_identifier(
                document.capture_group_id, "project capture group"
            ),
        },
        "images": images,
    }
    if document.ui_state is not None:
        payload["ui"] = _ui_payload(document.ui_state, known_nodes)
    return payload


def _document_from_payload(payload: object) -> ProjectAnalysisDocument:
    root = _object(payload, "project master")
    _keys(
        root,
        required={"format", "version", "analysis_settings", "project", "images"},
        optional={"ui"},
        name="project master",
    )
    if root["format"] != PROJECT_ANALYSIS_FORMAT:
        raise InvalidProjectAnalysis(
            f"Unsupported project format {root['format']!r}."
        )
    if not _exact_int(root["version"]) or root["version"] not in {1, PROJECT_ANALYSIS_VERSION}:
        raise InvalidProjectAnalysis(
            f"Unsupported project version {root['version']!r}."
        )
    try:
        settings = analysis_settings_profile_from_payload(root["analysis_settings"])
    except AnalysisSettingsError as error:
        raise InvalidProjectAnalysis(
            f"Invalid embedded analysis-settings profile: {error}"
        ) from error
    known_nodes = {node.identifier for node in settings.nodes}
    project = _object(root["project"], "project metadata")
    project_version = int(root["version"])
    _keys(
        project,
        required=(
            {"species", "selected_image"}
            if project_version == 1
            else {
                "species", "selected_image", "species_library",
                "biological_context", "capture_group_id",
            }
        ),
        optional=set(),
        name="project metadata",
    )
    species = _validated_optional_text(project["species"], "project species")
    selected = project["selected_image"]
    if selected is not None:
        selected = _digest_text(selected, "selected image identifier")
    species_library = (
        None
        if project_version == 1
        else _library_pin_from_payload(project["species_library"])
    )
    biological_context = (
        None
        if project_version == 1
        else _biological_context_from_payload(
            project["biological_context"], "project"
        )
    )
    capture_group_id = (
        None
        if project_version == 1
        else _validated_context_identifier(
            project["capture_group_id"], "project capture group"
        )
    )
    raw_images = root["images"]
    if not isinstance(raw_images, list):
        raise InvalidProjectAnalysis("Project images must be a JSON array.")
    if len(raw_images) > PROJECT_ANALYSIS_MAX_IMAGES:
        raise InvalidProjectAnalysis(
            f"Project masters support at most {PROJECT_ANALYSIS_MAX_IMAGES:,} images."
        )
    images: list[ProjectImageRecord] = []
    ids: set[str] = set()
    locators: set[tuple[str, str]] = set()
    for index, raw_image in enumerate(raw_images):
        image = _object(raw_image, f"image {index}")
        _keys(
            image,
            required=(
                {"id", "source", "image_sha256", "source_shape", "sidecars"}
                if project_version == 1
                else {
                    "id", "source", "image_sha256", "source_shape", "sidecars",
                    "biological_context", "capture_group_id",
                }
            ),
            optional=set(),
            name=f"image {index}",
        )
        source = _object(image["source"], f"image {index} source")
        _keys(
            source,
            required={"location", "path"},
            optional=set(),
            name=f"image {index} source",
        )
        location, stored_path = _validated_source_locator(
            source["location"], source["path"]
        )
        digest = _digest_text(image["image_sha256"], f"image {index} SHA-256")
        identifier = _digest_text(image["id"], f"image {index} identifier")
        if identifier != _image_identifier(location, stored_path):
            raise InvalidProjectAnalysis(
                f"Image {index} identifier does not match its locator."
            )
        if identifier in ids:
            raise InvalidProjectAnalysis("Project image identifiers must be unique.")
        ids.add(identifier)
        locator = (location, stored_path)
        if locator in locators:
            raise InvalidProjectAnalysis("Project source locators must be unique.")
        locators.add(locator)
        shape = _validated_shape(image["source_shape"], f"image {index} source shape")
        raw_sidecars = _object(image["sidecars"], f"image {index} sidecars")
        unknown_sidecars = set(raw_sidecars) - set(_SIDECAR_KINDS)
        if unknown_sidecars:
            raise InvalidProjectAnalysis(
                f"Image {index} has unsupported sidecar(s): {sorted(unknown_sidecars)}."
            )
        sidecars: list[ProjectSidecarReference] = []
        for kind in _SIDECAR_KINDS:
            if kind not in raw_sidecars:
                continue
            sidecar = _object(raw_sidecars[kind], f"image {index} {kind}")
            _keys(
                sidecar,
                required={"path", "sha256"},
                optional=set(),
                name=f"image {index} {kind}",
            )
            sidecars.append(
                ProjectSidecarReference(
                    kind=kind,
                    path=_validated_relative(sidecar["path"], f"{kind} path"),
                    sha256=_digest_text(sidecar["sha256"], f"{kind} SHA-256"),
                )
            )
        images.append(
            ProjectImageRecord(
                identifier=identifier,
                source=ProjectFileReference(location, stored_path, digest),
                source_shape=shape,
                sidecars=tuple(sidecars),
                biological_context=(
                    None
                    if project_version == 1
                    else _biological_context_from_payload(
                        image["biological_context"], f"image {index}"
                    )
                ),
                capture_group_id=(
                    None
                    if project_version == 1
                    else _validated_context_identifier(
                        image["capture_group_id"], f"image {index} capture group"
                    )
                ),
            )
        )
    if selected is not None and selected not in ids:
        raise InvalidProjectAnalysis(
            "Selected image identifier is not present in the project image list."
        )
    ui_state = None
    if "ui" in root:
        ui_state = _ui_from_payload(root["ui"], known_nodes)
    return ProjectAnalysisDocument(
        analysis_settings=settings,
        images=tuple(images),
        species=species,
        selected_image_id=selected,
        ui_state=ui_state,
        species_library=species_library,
        biological_context=biological_context,
        capture_group_id=capture_group_id,
    )


def _ui_payload(state: ProjectUiState, known_nodes: set[str]) -> dict[str, object]:
    if not isinstance(state, ProjectUiState):
        raise InvalidProjectAnalysis("Project UI state must be a ProjectUiState.")
    positions: dict[str, list[float]] = {}
    for identifier, coordinates in state.node_positions.items():
        node_id = _known_node(identifier, known_nodes, "node position")
        positions[node_id] = list(_coordinates(coordinates, f"{node_id} position"))
    if not isinstance(state.bundle_cables, bool) or not isinstance(
        state.route_around_nodes, bool
    ):
        raise InvalidProjectAnalysis("Project cable-routing flags must be Boolean.")
    selected_node = (
        None
        if state.selected_node is None
        else _known_node(state.selected_node, known_nodes, "selected node")
    )
    selected_overlay = _validated_optional_text(
        state.selected_overlay, "selected overlay"
    )
    return {
        "node_positions": positions,
        "bundle_cables": state.bundle_cables,
        "route_around_nodes": state.route_around_nodes,
        "selected_node": selected_node,
        "selected_overlay": selected_overlay,
    }


def _ui_from_payload(payload: object, known_nodes: set[str]) -> ProjectUiState:
    ui = _object(payload, "project UI state")
    _keys(
        ui,
        required={
            "node_positions",
            "bundle_cables",
            "route_around_nodes",
            "selected_node",
            "selected_overlay",
        },
        optional=set(),
        name="project UI state",
    )
    raw_positions = _object(ui["node_positions"], "node positions")
    positions = {
        _known_node(identifier, known_nodes, "node position"): _coordinates(
            coordinates, f"{identifier} position"
        )
        for identifier, coordinates in raw_positions.items()
    }
    for field_name in ("bundle_cables", "route_around_nodes"):
        if not isinstance(ui[field_name], bool):
            raise InvalidProjectAnalysis(
                f"Project UI field {field_name!r} must be Boolean."
            )
    selected_node = ui["selected_node"]
    if selected_node is not None:
        selected_node = _known_node(selected_node, known_nodes, "selected node")
    return ProjectUiState(
        node_positions=positions,
        bundle_cables=ui["bundle_cables"],
        route_around_nodes=ui["route_around_nodes"],
        selected_node=selected_node,
        selected_overlay=_validated_optional_text(
            ui["selected_overlay"], "selected overlay"
        ),
    )


def _file_status(
    path: Path, expected_sha256: str, verify: bool
) -> tuple[ProjectFileStatus, str | None]:
    if not verify:
        return ProjectFileStatus.UNCHECKED, None
    if not path.is_file():
        return ProjectFileStatus.MISSING, None
    try:
        actual = file_sha256(path)
    except OSError:
        return ProjectFileStatus.UNREADABLE, None
    return (
        (ProjectFileStatus.AVAILABLE, actual)
        if actual == expected_sha256
        else (ProjectFileStatus.FINGERPRINT_MISMATCH, actual)
    )


def _validated_sidecar_status(
    *,
    kind: str,
    path: Path,
    source_path: Path,
    source_shape: tuple[int, int],
    reference_store: ReferenceRegionStore,
    centre_store: ManualSeedCentreStore,
) -> ProjectFileStatus:
    try:
        restored = (
            reference_store.load_project_archive(source_path, path, None)
            if kind == REFERENCE_REGIONS_SIDECAR
            else centre_store.load_project_archive(source_path, path, source_shape)
        )
    except (
        ReferenceRegionError,
        ManualSeedCentreStoreError,
        OSError,
    ):
        return ProjectFileStatus.INVALID_SIDECAR
    return (
        ProjectFileStatus.AVAILABLE
        if restored is not None
        else ProjectFileStatus.MISSING
    )


def _source_locator_is_native(reference: ProjectFileReference) -> bool:
    if reference.location == PROJECT_RELATIVE:
        return True
    # The schema accepts absolute paths from either major path flavour so a
    # moved project can still describe what is missing.  Only the host-native
    # flavour may be probed; otherwise Windows could reinterpret ``/var/x`` as
    # a current-drive path, or POSIX could reinterpret ``C:/x`` as relative.
    return Path(reference.path).is_absolute()


def _master_path(value: Path | str) -> Path:
    path = Path(value).resolve()
    if not path.name.casefold().endswith(PROJECT_ANALYSIS_EXTENSION):
        raise InvalidProjectAnalysis(
            f"Project masters must use the {PROJECT_ANALYSIS_EXTENSION} extension."
        )
    return path


def _validated_source_locator(location: object, value: object) -> tuple[str, str]:
    if location == PROJECT_RELATIVE:
        return PROJECT_RELATIVE, _validated_relative(value, "source image path")
    if location != EXTERNAL_ABSOLUTE:
        raise InvalidProjectAnalysis(
            f"Source image location must be {PROJECT_RELATIVE!r} or "
            f"{EXTERNAL_ABSOLUTE!r}."
        )
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise InvalidProjectAnalysis("External image path must be non-empty.")
    if not (
        Path(value).is_absolute()
        or PureWindowsPath(value).is_absolute()
        or PurePosixPath(value).is_absolute()
    ):
        raise InvalidProjectAnalysis("External image path must be absolute.")
    if "\x00" in value:
        raise InvalidProjectAnalysis("External image path contains a NUL byte.")
    return EXTERNAL_ABSOLUTE, value


def _external_path(value: str) -> Path:
    # A foreign-platform absolute locator remains an explicit unresolved path;
    # it is reported missing instead of being silently rebased under the root.
    path = Path(value)
    return path.resolve() if path.is_absolute() else path


def _canonical_relative(path: Path, name: str) -> str:
    return _validated_relative(path.as_posix(), name)


def _validated_relative(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise InvalidProjectAnalysis(f"{name.capitalize()} must be a non-empty path.")
    if "\\" in value or "\x00" in value:
        raise InvalidProjectAnalysis(
            f"{name.capitalize()} must use canonical POSIX separators."
        )
    relative = PurePosixPath(value)
    if relative.is_absolute() or value != relative.as_posix() or value == ".":
        raise InvalidProjectAnalysis(
            f"{name.capitalize()} must be a normalized project-relative path."
        )
    if any(part in {"", ".", ".."} for part in relative.parts):
        raise InvalidProjectAnalysis(f"{name.capitalize()} escapes the project root.")
    return value


def _resolve_project_relative(root: Path, value: str, name: str) -> Path:
    relative = PurePosixPath(_validated_relative(value, name))
    resolved = (root / Path(*relative.parts)).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise InvalidProjectAnalysis(f"{name.capitalize()} escapes the project root.") from error
    return resolved


def _image_identifier(location: str, path: str) -> str:
    return sha256(f"{location}\0{path}".encode("utf-8")).hexdigest()


def _validated_shape(value: object, name: str) -> tuple[int, int]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise InvalidProjectAnalysis(f"{name.capitalize()} must be [height, width].")
    if any(not _exact_int(item) or item <= 0 for item in value):
        raise InvalidProjectAnalysis(f"{name.capitalize()} values must be positive integers.")
    return int(value[0]), int(value[1])


def _validated_context_identifier(value: object, name: str) -> str | None:
    if value is None:
        return None
    if (
        not isinstance(value, str)
        or not re.fullmatch(r"[a-z][a-z0-9_.-]{0,127}", value)
    ):
        raise InvalidProjectAnalysis(
            f"{name.capitalize()} must be null or a stable lowercase identifier."
        )
    return value


def _validated_biological_context(
    value: object, name: str
) -> BiologicalContext | None:
    if value is None:
        return None
    if not isinstance(value, BiologicalContext):
        raise InvalidProjectAnalysis(
            f"{name.capitalize()} biological context is invalid."
        )
    try:
        return BiologicalContext(
            value.species_id,
            value.lineage_group_id,
            value.accession_id,
            value.seed_lot_id,
        )
    except ValueError as error:
        raise InvalidProjectAnalysis(
            f"{name.capitalize()} biological context is invalid: {error}"
        ) from error


def _biological_context_payload(
    value: BiologicalContext | None,
) -> dict[str, object] | None:
    if value is None:
        return None
    return {
        "species_id": value.species_id,
        "lineage_group_id": value.lineage_group_id,
        "accession_id": value.accession_id,
        "seed_lot_id": value.seed_lot_id,
    }


def _biological_context_from_payload(
    value: object, name: str
) -> BiologicalContext | None:
    if value is None:
        return None
    payload = _object(value, f"{name} biological context")
    _keys(
        payload,
        required={
            "species_id", "lineage_group_id", "accession_id", "seed_lot_id"
        },
        optional=set(),
        name=f"{name} biological context",
    )
    try:
        return BiologicalContext(
            species_id=payload["species_id"],
            lineage_group_id=payload["lineage_group_id"],
            accession_id=payload["accession_id"],
            seed_lot_id=payload["seed_lot_id"],
        )
    except ValueError as error:
        raise InvalidProjectAnalysis(
            f"Invalid {name} biological context: {error}"
        ) from error


def _validated_library_pin(value: object) -> SpeciesLibraryPin | None:
    if value is None:
        return None
    if not isinstance(value, SpeciesLibraryPin):
        raise InvalidProjectAnalysis("Project species-library pin is invalid.")
    try:
        return SpeciesLibraryPin(
            value.library_id,
            value.version,
            value.species_id,
            value.sha256,
            value.resolution,
        )
    except ValueError as error:
        raise InvalidProjectAnalysis(
            f"Project species-library pin is invalid: {error}"
        ) from error


def _library_pin_payload(
    value: SpeciesLibraryPin | None,
) -> dict[str, object] | None:
    if value is None:
        return None
    return {
        "library_id": value.library_id,
        "version": value.version,
        "species_id": value.species_id,
        "sha256": value.sha256,
        "resolution": value.resolution,
    }


def _library_pin_from_payload(value: object) -> SpeciesLibraryPin | None:
    if value is None:
        return None
    payload = _object(value, "species-library pin")
    _keys(
        payload,
        required={"library_id", "version", "species_id", "sha256", "resolution"},
        optional=set(),
        name="species-library pin",
    )
    try:
        return SpeciesLibraryPin(
            payload["library_id"], payload["version"], payload["species_id"],
            payload["sha256"], payload["resolution"],
        )
    except ValueError as error:
        raise InvalidProjectAnalysis(
            f"Invalid species-library pin: {error}"
        ) from error


def _validated_optional_text(value: object, name: str) -> str | None:
    if value is None:
        return None
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > 256
        or "\x00" in value
    ):
        raise InvalidProjectAnalysis(
            f"{name.capitalize()} must be null or trimmed text up to 256 characters."
        )
    return value


def _validated_digest(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != _SHA256_LENGTH or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise InvalidProjectAnalysis(f"{name.capitalize()} must be a lowercase SHA-256.")
    return value


def _digest_text(value: object, name: str) -> str:
    return _validated_digest(value, name)


def _coordinates(value: object, name: str) -> tuple[float, float]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise InvalidProjectAnalysis(f"{name.capitalize()} must contain x and y.")
    coordinates: list[float] = []
    for coordinate in value:
        if isinstance(coordinate, bool) or not isinstance(coordinate, (int, float)):
            raise InvalidProjectAnalysis(f"{name.capitalize()} must be numeric.")
        parsed = float(coordinate)
        if not math.isfinite(parsed):
            raise InvalidProjectAnalysis(f"{name.capitalize()} must be finite.")
        coordinates.append(parsed)
    return coordinates[0], coordinates[1]


def _known_node(value: object, known_nodes: set[str], name: str) -> str:
    if not isinstance(value, str) or value not in known_nodes:
        raise InvalidProjectAnalysis(f"Project {name} must name a known pipeline node.")
    return value


def _object(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise InvalidProjectAnalysis(f"{name.capitalize()} must be one JSON object.")
    if any(not isinstance(key, str) for key in value):
        raise InvalidProjectAnalysis(f"{name.capitalize()} keys must be text.")
    return value


def _keys(
    value: Mapping[str, object], *, required: set[str], optional: set[str], name: str
) -> None:
    missing = required - set(value)
    unknown = set(value) - required - optional
    if missing:
        raise InvalidProjectAnalysis(
            f"{name.capitalize()} is missing field(s): {sorted(missing)}."
        )
    if unknown:
        raise InvalidProjectAnalysis(
            f"{name.capitalize()} has unknown field(s): {sorted(unknown)}."
        )


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise InvalidProjectAnalysis(f"Duplicate JSON field {key!r}.")
        result[key] = value
    return result


def _invalid_json_constant(value: str) -> object:
    raise InvalidProjectAnalysis(f"Non-finite JSON constant {value!r} is invalid.")


def _exact_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _path_key(path: Path) -> str:
    text = str(path.resolve())
    return text.casefold() if os.name == "nt" else text


__all__ = [
    "EXTERNAL_ABSOLUTE",
    "MANUAL_SEED_CENTRES_SIDECAR",
    "PROJECT_ANALYSIS_DEFAULT_NAME",
    "PROJECT_ANALYSIS_EXTENSION",
    "PROJECT_ANALYSIS_FORMAT",
    "PROJECT_ANALYSIS_MAX_BYTES",
    "PROJECT_ANALYSIS_MAX_IMAGES",
    "PROJECT_ANALYSIS_VERSION",
    "PROJECT_RELATIVE",
    "REFERENCE_REGIONS_SIDECAR",
    "InvalidProjectAnalysis",
    "ProjectAnalysisDocument",
    "ProjectAnalysisError",
    "ProjectAnalysisIOError",
    "ProjectAnalysisLoadResult",
    "ProjectAnalysisStore",
    "ProjectFileIssue",
    "ProjectFileReference",
    "ProjectFileStatus",
    "ProjectImageRecord",
    "ProjectImageSpec",
    "ProjectSidecarReference",
    "ProjectUiState",
    "ResolvedProjectImage",
    "ResolvedProjectSidecar",
]
