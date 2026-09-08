"""Typed contracts for immutable, reusable species reference libraries.

The library node moves compact descriptors and statistical summaries only.
Full-resolution images and annotation rasters remain owned by their source
projects and can never be recovered from a published library.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
import re
from types import MappingProxyType
from typing import Mapping

import numpy as np


SPECIES_LIBRARY_FORMAT = "seedfiddle-species-library"
SPECIES_LIBRARY_SCHEMA_VERSION = 1
SPECIES_LIBRARY_EXTENSION = ".seedfiddle-species-library.zip"
SPECIES_LIBRARY_MAX_SOURCES = 10_000
SPECIES_LIBRARY_MAX_ARRAYS = 256
SPECIES_LIBRARY_MAX_ARRAY_BYTES = 256 * 1024 * 1024

FOREGROUND_COLOUR_SCHEMA = "foreground-colour-lab-frequency-v1"
FOREGROUND_NOISE_SCHEMA = "foreground-noise-oriented-multiscale-v2"
MATERIAL_PROTOTYPE_SCHEMA = "material-prototype-production-descriptor-v2"
EDGE_PROTOTYPE_SCHEMA = "edge-tangent-normal-production-strip-v2"
SEED_TRAIT_SCHEMA = "seed-trait-production-prototype-v2"
SHAPE_SUMMARY_SCHEMA = "species-shape-summary-v3"
DIMENSIONS_SHAPE_SCHEMA = "species-dimensions-shape-bank-v3"

_IDENTIFIER = re.compile(r"[a-z][a-z0-9_.-]{0,127}")
_LIBRARY_TOKEN = re.compile(r"[a-z0-9][a-z0-9_.-]{0,127}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_PHYSICAL_SEED_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}")


class LibraryProduct(StrEnum):
    FOREGROUND_COLOUR = "foreground_colour"
    FOREGROUND_NOISE = "foreground_noise"
    MATERIAL_PROTOTYPES = "material_prototypes"
    EDGE_PROTOTYPES = "edge_prototypes"
    SEED_TRAITS = "seed_traits"
    SHAPE_SUMMARY = "shape_summary"
    DIMENSIONS_SHAPE = "dimensions_shape"


class ValidationTier(StrEnum):
    UNAVAILABLE = "unavailable"
    PROVISIONAL = "provisional"
    VALIDATED = "validated"
    MULTI_CONTEXT_VALIDATED = "multi_context_validated"


class LibraryStatus(StrEnum):
    DRAFT = "draft"
    PUBLISHED = "published"
    RETIRED = "retired"


class ReferenceSourceMode(StrEnum):
    CURRENT_IMAGE_ONLY = "current_image_only"
    SPECIES_LIBRARY_ONLY = "species_library_only"
    SPECIES_LIBRARY_AND_CURRENT = "species_library_and_current"


class ShapeReferenceSourceMode(StrEnum):
    CURRENT_IMAGE_ONLY = "current_image_only"
    SPECIES_LIBRARY_ONLY = "species_library_only"
    SPECIES_LIBRARY_PRIOR_AND_CURRENT = "species_library_prior_and_current"


@dataclass(frozen=True, slots=True)
class BiologicalContext:
    species_id: str
    lineage_group_id: str | None = None
    accession_id: str | None = None
    seed_lot_id: str | None = None

    def __post_init__(self) -> None:
        _require_identifier(self.species_id, "species ID")
        for name in ("lineage_group_id", "accession_id", "seed_lot_id"):
            _optional_identifier(getattr(self, name), name.replace("_", " "))

    def fallback_paths(self) -> tuple[tuple[str, ...], ...]:
        """Return most-specific to species-only hierarchy keys."""

        values = self.hierarchy_path()
        return tuple(values[:stop] for stop in range(len(values), 0, -1))

    def hierarchy_path(self) -> tuple[str, ...]:
        """Return an unambiguous biological path with optional levels tagged.

        Missing lineage or accession levels must not cause an accession, lot,
        or lineage identifier to be interpreted as another level after tuple
        compaction.  The species remains the stable root for compatibility;
        optional descendants carry their semantic level in the key.
        """

        values = [self.species_id]
        for level, value in (
            ("lineage", self.lineage_group_id),
            ("accession", self.accession_id),
            ("lot", self.seed_lot_id),
        ):
            if value is not None:
                values.append(f"{level}:{value}")
        return tuple(values)


@dataclass(frozen=True, slots=True)
class SpeciesLibraryPin:
    library_id: str
    version: str
    species_id: str
    sha256: str
    resolution: str = "user_catalogue"

    def __post_init__(self) -> None:
        _require_library_token(self.library_id, "library ID")
        _require_library_token(self.version, "library version")
        _require_identifier(self.species_id, "species ID")
        _require_sha256(self.sha256, "library SHA-256")
        if self.resolution not in {"user_catalogue", "embedded_project"}:
            raise ValueError("Library pin resolution is unsupported.")


@dataclass(frozen=True, slots=True)
class LibrarySourceRecord:
    source_sha256: str
    annotation_sha256: str
    display_label: str
    source_shape: tuple[int, int]
    capture_group_id: str | None = None
    biological_context: BiologicalContext | None = None
    review_status: str = "reviewed"

    def __post_init__(self) -> None:
        _require_sha256(self.source_sha256, "source SHA-256")
        _require_sha256(self.annotation_sha256, "annotation SHA-256")
        if not str(self.display_label).strip() or len(self.display_label) > 256:
            raise ValueError("Library source display labels must contain 1--256 characters.")
        if (
            not isinstance(self.source_shape, tuple)
            or len(self.source_shape) != 2
            or any(isinstance(value, bool) or int(value) <= 0 for value in self.source_shape)
        ):
            raise ValueError("Library source dimensions must be two positive integers.")
        _optional_identifier(self.capture_group_id, "capture group ID")
        if self.review_status not in {"reviewed", "partial", "rejected"}:
            raise ValueError("Library source review status is unsupported.")


@dataclass(frozen=True, slots=True)
class ProductValidation:
    product: LibraryProduct
    tier: ValidationTier
    source_count: int
    seed_count: int = 0
    sample_count: int = 0
    prototype_count: int = 0
    effective_weight: float = 0.0
    metrics: Mapping[str, float] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("source_count", "seed_count", "sample_count", "prototype_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) < 0:
                raise ValueError(f"{name} must be a non-negative integer.")
        if not np.isfinite(self.effective_weight) or self.effective_weight < 0:
            raise ValueError("effective_weight must be finite and non-negative.")
        normalized = {}
        for name, value in dict(self.metrics).items():
            if not str(name).strip() or not np.isfinite(float(value)):
                raise ValueError("Validation metrics need non-empty names and finite values.")
            normalized[str(name)] = float(value)
        object.__setattr__(self, "metrics", MappingProxyType(normalized))
        object.__setattr__(self, "warnings", tuple(str(value) for value in self.warnings))


@dataclass(frozen=True, slots=True)
class LibraryArraySpec:
    name: str
    dtype: str
    shape: tuple[int, ...]
    byte_count: int
    sha256: str

    def __post_init__(self) -> None:
        _require_identifier(self.name, "array name")
        if np.dtype(self.dtype).hasobject:
            raise ValueError("Library arrays cannot use object dtype.")
        if any(isinstance(value, bool) or int(value) < 0 for value in self.shape):
            raise ValueError("Library array shapes must be non-negative integers.")
        if isinstance(self.byte_count, bool) or self.byte_count < 0:
            raise ValueError("Library array byte counts must be non-negative.")
        _require_sha256(self.sha256, "array SHA-256")


@dataclass(frozen=True, slots=True)
class SpeciesLibraryManifest:
    library_id: str
    version: str
    content_sha256: str
    species_id: str
    species_display_name: str
    trait_vocabulary_sha256: str
    seed_fiddle_version: str
    created_utc: str
    status: LibraryStatus
    sources: tuple[LibrarySourceRecord, ...]
    products: tuple[ProductValidation, ...]
    descriptor_schemas: Mapping[str, str]
    extraction_settings_sha256: str
    aggregation_settings_sha256: str
    corrected_colour_space_version: str
    arrays: tuple[LibraryArraySpec, ...]
    parent_content_sha256: str | None = None

    def __post_init__(self) -> None:
        _require_library_token(self.library_id, "library ID")
        _require_library_token(self.version, "library version")
        _require_sha256(self.content_sha256, "library content SHA-256", allow_empty=True)
        _require_identifier(self.species_id, "species ID")
        if not self.species_display_name.strip():
            raise ValueError("Species display name cannot be empty.")
        _require_sha256(self.trait_vocabulary_sha256, "trait vocabulary SHA-256")
        _require_sha256(self.extraction_settings_sha256, "extraction settings SHA-256")
        _require_sha256(self.aggregation_settings_sha256, "aggregation settings SHA-256")
        if self.parent_content_sha256 is not None:
            _require_sha256(self.parent_content_sha256, "parent content SHA-256")
        if len(self.sources) > SPECIES_LIBRARY_MAX_SOURCES:
            raise ValueError("Species library contains too many sources.")
        source_hashes = tuple(item.source_sha256 for item in self.sources)
        if len(source_hashes) != len(set(source_hashes)):
            raise ValueError("Species library source images must be unique.")
        product_ids = tuple(item.product for item in self.products)
        if len(product_ids) != len(set(product_ids)):
            raise ValueError("Species library product records must be unique.")
        if len(self.arrays) > SPECIES_LIBRARY_MAX_ARRAYS:
            raise ValueError("Species library contains too many arrays.")
        names = tuple(item.name for item in self.arrays)
        if len(names) != len(set(names)):
            raise ValueError("Species library array names must be unique.")
        schemas = {}
        for product, schema in dict(self.descriptor_schemas).items():
            _require_identifier(str(product), "descriptor product")
            _require_identifier(str(schema), "descriptor schema")
            schemas[str(product)] = str(schema)
        object.__setattr__(self, "descriptor_schemas", MappingProxyType(schemas))

    def validation_for(self, product: LibraryProduct) -> ProductValidation:
        return next(
            (value for value in self.products if value.product is product),
            ProductValidation(product, ValidationTier.UNAVAILABLE, 0),
        )


@dataclass(frozen=True, slots=True)
class SourceProfileBank:
    """Source-tagged positive-class profiles evaluated independently at runtime."""

    centres: np.ndarray
    scales: np.ndarray
    weights: np.ndarray
    source_indices: np.ndarray
    sample_counts: np.ndarray
    schema_id: str
    feature_names: tuple[str, ...]
    half_distances: np.ndarray | None = None

    def __post_init__(self) -> None:
        centres = _readonly_array(self.centres, np.float32, "profile centres", 2)
        scales = _readonly_array(self.scales, np.float32, "profile scales", 2)
        if centres.shape != scales.shape or centres.shape[1] != len(self.feature_names):
            raise ValueError("Profile centres/scales must match their feature schema.")
        rows = centres.shape[0]
        weights = _vector(self.weights, np.float32, rows, "profile weights")
        indices = _vector(self.source_indices, np.int32, rows, "profile sources")
        counts = _vector(self.sample_counts, np.int64, rows, "profile sample counts")
        if np.any(scales <= 0) or np.any(weights < 0) or np.any(indices < 0) or np.any(counts < 0):
            raise ValueError("Profile scales/weights/counts/source indices are invalid.")
        if rows and not np.any(weights > 0):
            raise ValueError("At least one profile weight must be positive.")
        half = (
            np.ones(rows, np.float32)
            if self.half_distances is None
            else _vector(self.half_distances, np.float32, rows, "profile half distances")
        )
        if np.any(half <= 0):
            raise ValueError("Profile half distances must be positive.")
        _require_identifier(self.schema_id, "profile schema")
        object.__setattr__(self, "centres", centres)
        object.__setattr__(self, "scales", scales)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "source_indices", indices)
        object.__setattr__(self, "sample_counts", counts)
        object.__setattr__(self, "half_distances", half)

    def excluding_sources(self, source_indices: set[int]) -> SourceProfileBank:
        keep = ~np.isin(self.source_indices, tuple(source_indices))
        retained_sources = self.source_indices[keep]
        retained_weights = _balanced_source_weights(
            self.weights[keep], retained_sources
        )
        return type(self)(
            self.centres[keep], self.scales[keep], retained_weights,
            retained_sources, self.sample_counts[keep], self.schema_id,
            self.feature_names, self.half_distances[keep],
        )


class SpeciesForegroundColourBank(SourceProfileBank):
    pass


class SpeciesForegroundNoiseBank(SourceProfileBank):
    pass


@dataclass(frozen=True, slots=True)
class SourcePrototypeBank:
    centres: np.ndarray
    scales: np.ndarray
    weights: np.ndarray
    source_indices: np.ndarray
    seed_ids: np.ndarray
    class_ids: np.ndarray
    sample_counts: np.ndarray
    schema_id: str
    feature_names: tuple[str, ...]
    class_names: tuple[str, ...]

    def __post_init__(self) -> None:
        centres = _readonly_array(self.centres, np.float32, "prototype centres", 2)
        scales = _readonly_array(self.scales, np.float32, "prototype scales", 2)
        if centres.shape != scales.shape or centres.shape[1] != len(self.feature_names):
            raise ValueError("Prototype centres/scales must match their feature schema.")
        rows = centres.shape[0]
        weights = _vector(self.weights, np.float32, rows, "prototype weights")
        sources = _vector(self.source_indices, np.int32, rows, "prototype sources")
        seeds = _vector(self.seed_ids, np.int32, rows, "prototype seed IDs")
        classes = _vector(self.class_ids, np.int16, rows, "prototype class IDs")
        counts = _vector(self.sample_counts, np.int64, rows, "prototype sample counts")
        if np.any(scales <= 0) or np.any(weights < 0) or np.any(sources < 0) or np.any(seeds < 0) or np.any(counts < 0):
            raise ValueError("Prototype metadata contains invalid negative values.")
        if np.any(classes < 0) or (rows and np.any(classes >= len(self.class_names))):
            raise ValueError("Prototype class IDs are outside the class-name table.")
        _require_identifier(self.schema_id, "prototype schema")
        object.__setattr__(self, "centres", centres)
        object.__setattr__(self, "scales", scales)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "source_indices", sources)
        object.__setattr__(self, "seed_ids", seeds)
        object.__setattr__(self, "class_ids", classes)
        object.__setattr__(self, "sample_counts", counts)

    def excluding_sources(self, source_indices: set[int]) -> SourcePrototypeBank:
        keep = ~np.isin(self.source_indices, tuple(source_indices))
        retained_sources = self.source_indices[keep]
        retained_classes = self.class_ids[keep]
        retained_weights = self.weights[keep].copy()
        for class_id in range(len(self.class_names)):
            selected = np.flatnonzero(retained_classes == class_id)
            if len(selected):
                retained_weights[selected] = _balanced_source_weights(
                    retained_weights[selected], retained_sources[selected]
                )
        return type(self)(
            self.centres[keep], self.scales[keep], retained_weights,
            retained_sources, self.seed_ids[keep], retained_classes,
            self.sample_counts[keep], self.schema_id, self.feature_names,
            self.class_names,
        )


class SpeciesMaterialPrototypeBank(SourcePrototypeBank):
    pass


class SpeciesEdgePrototypeBank(SourcePrototypeBank):
    pass


class SpeciesSeedTraitBank(SourcePrototypeBank):
    pass


@dataclass(frozen=True, slots=True)
class ShapeObservation:
    source_index: int
    seed_id: int
    physical_seed_id: str | None
    hierarchy_path: tuple[str, ...]
    pose: str
    visibility: str
    calibrated: bool
    measurement: tuple[float, ...]
    measurement_uncertainty: tuple[float, ...]
    contour_signature: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.source_index < 0 or self.seed_id <= 0:
            raise ValueError("Shape observations need non-negative source and positive seed IDs.")
        if (
            self.physical_seed_id is not None
            and _PHYSICAL_SEED_IDENTIFIER.fullmatch(str(self.physical_seed_id))
            is None
        ):
            raise ValueError(
                "Physical seed ID must be a stable 1--128 character identifier."
            )
        if self.pose not in {"flat", "oblique", "side", "uncertain"}:
            raise ValueError("Shape observation pose is unsupported.")
        if self.visibility != "complete":
            raise ValueError("Published shape observations must have complete visibility.")
        if len(self.measurement) != len(self.measurement_uncertainty):
            raise ValueError("Shape measurements and uncertainties must have equal dimensions.")
        values = np.asarray((*self.measurement, *self.measurement_uncertainty), np.float64)
        if not np.all(np.isfinite(values)) or np.any(np.asarray(self.measurement_uncertainty) < 0):
            raise ValueError("Shape observations require finite values and non-negative uncertainty.")
        if self.contour_signature and not np.all(
            np.isfinite(np.asarray(self.contour_signature, np.float64))
        ):
            raise ValueError("Shape contour signatures must be finite.")


SHAPE_MEASUREMENT_NAMES = (
    "maximum_span", "body_length", "body_width", "ovality", "projected_area",
    "ellipse_relative_area", "non_ellipticity", "asymmetry", "solidity",
    "concavity_fraction", "protrusion_fraction", "neck_width_fraction",
    "thin_extension_fraction", "local_feature_height", "local_feature_arc_fraction",
    "local_feature_axis_cosine", "local_feature_axis_sine",
    "local_feature_support", "local_feature_hilum_associated",
)


@dataclass(frozen=True, slots=True)
class SpeciesShapeSummary:
    observations: tuple[ShapeObservation, ...]
    measurement_names: tuple[str, ...] = SHAPE_MEASUREMENT_NAMES
    schema_id: str = SHAPE_SUMMARY_SCHEMA

    def __post_init__(self) -> None:
        _require_identifier(self.schema_id, "shape-summary schema")
        if any(len(item.measurement) != len(self.measurement_names) for item in self.observations):
            raise ValueError("Shape-summary observations do not match the measurement schema.")


@dataclass(frozen=True, slots=True)
class ShapePopulationComponent:
    hierarchy_path: tuple[str, ...]
    pose: str
    mean: tuple[float, ...]
    covariance: tuple[tuple[float, ...], ...]
    mean_uncertainty: tuple[float, ...]
    predictive_interval_95: tuple[tuple[float, float], ...]
    effective_physical_seed_count: float
    source_count: int
    contour_mean: tuple[float, ...] = ()
    contour_modes: tuple[tuple[float, ...], ...] = ()
    contour_variances: tuple[float, ...] = ()
    physical_dimensions_available: bool = True
    calibrated_physical_seed_count: float = 0.0
    dimensionless_physical_seed_count: float = 0.0

    def __post_init__(self) -> None:
        dimension = len(self.mean)
        if len(self.covariance) != dimension or any(len(row) != dimension for row in self.covariance):
            raise ValueError("Shape component covariance must be square.")
        if len(self.mean_uncertainty) != dimension or len(self.predictive_interval_95) != dimension:
            raise ValueError("Shape component uncertainty dimensions do not match its mean.")
        numeric = np.asarray((self.mean, self.mean_uncertainty, *self.covariance), np.float64)
        if not np.all(np.isfinite(numeric)):
            raise ValueError("Shape component statistics must be finite.")
        if self.effective_physical_seed_count < 0 or self.source_count < 0:
            raise ValueError("Shape component counts cannot be negative.")
        if not isinstance(self.physical_dimensions_available, bool):
            raise ValueError("Shape component physical-dimension availability must be Boolean.")
        if (
            self.calibrated_physical_seed_count < 0
            or self.dimensionless_physical_seed_count < 0
        ):
            raise ValueError("Shape component calibrated/dimensionless counts cannot be negative.")


@dataclass(frozen=True, slots=True)
class SpeciesDimensionsShapeBank:
    observations: tuple[ShapeObservation, ...]
    components: tuple[ShapePopulationComponent, ...]
    measurement_names: tuple[str, ...] = SHAPE_MEASUREMENT_NAMES
    schema_id: str = DIMENSIONS_SHAPE_SCHEMA
    minimum_component_seeds: int = 2
    shrinkage_seed_count: float = 5.0
    maximum_contour_modes: int = 4

    def __post_init__(self) -> None:
        _require_identifier(self.schema_id, "dimensions/shape schema")
        if any(len(item.measurement) != len(self.measurement_names) for item in self.observations):
            raise ValueError("Dimensions/shape observations do not match the schema.")
        if any(len(item.mean) != len(self.measurement_names) for item in self.components):
            raise ValueError("Dimensions/shape components do not match the schema.")
        if self.minimum_component_seeds < 1:
            raise ValueError("Shape-bank minimum component support must be positive.")
        if self.shrinkage_seed_count < 0 or self.maximum_contour_modes < 0:
            raise ValueError("Shape-bank shrinkage and mode limits cannot be negative.")

    def select_component(self, context: BiologicalContext, pose: str) -> tuple[ShapePopulationComponent | None, tuple[str, ...]]:
        for path in context.fallback_paths():
            candidate = next((item for item in self.components if item.hierarchy_path == path and item.pose == pose), None)
            if candidate is not None:
                return candidate, path
        candidate = next((item for item in self.components if item.hierarchy_path == (context.species_id,) and item.pose == "uncertain"), None)
        return candidate, (() if candidate is None else candidate.hierarchy_path)


@dataclass(frozen=True, slots=True)
class SpeciesLibraryArtifact:
    manifest: SpeciesLibraryManifest
    foreground_colour: SpeciesForegroundColourBank | None = None
    foreground_noise: SpeciesForegroundNoiseBank | None = None
    material_prototypes: SpeciesMaterialPrototypeBank | None = None
    edge_prototypes: SpeciesEdgePrototypeBank | None = None
    seed_traits: SpeciesSeedTraitBank | None = None
    shape_summary: SpeciesShapeSummary | None = None
    dimensions_shape: SpeciesDimensionsShapeBank | None = None

    def excluding_source_sha256(self, source_sha256: str) -> tuple[SpeciesLibraryArtifact, tuple[int, ...]]:
        excluded = tuple(index for index, source in enumerate(self.manifest.sources) if source.source_sha256 == source_sha256)
        if not excluded:
            return self, ()
        excluded_set = set(excluded)
        def filtered(bank):
            return None if bank is None else bank.excluding_sources(excluded_set)
        summary = None if self.shape_summary is None else SpeciesShapeSummary(
            tuple(item for item in self.shape_summary.observations if item.source_index not in excluded_set),
            self.shape_summary.measurement_names,
            self.shape_summary.schema_id,
        )
        if self.dimensions_shape is None:
            dimensions = None
        else:
            # A fitted hierarchy component contains sufficient statistics from
            # every contributing source. Retaining it after excluding the
            # current image would silently leak that image back into runtime.
            # Refit the compact bank from the retained reviewed observations.
            from seedvision.measurement.shape_model import (
                fit_species_dimensions_shape_bank,
            )
            dimensions = fit_species_dimensions_shape_bank(
                tuple(
                    item
                    for item in self.dimensions_shape.observations
                    if item.source_index not in excluded_set
                ),
                minimum_component_seeds=self.dimensions_shape.minimum_component_seeds,
                shrinkage_seed_count=self.dimensions_shape.shrinkage_seed_count,
                maximum_contour_modes=self.dimensions_shape.maximum_contour_modes,
            )
        return SpeciesLibraryArtifact(
            self.manifest, filtered(self.foreground_colour), filtered(self.foreground_noise),
            filtered(self.material_prototypes), filtered(self.edge_prototypes),
            filtered(self.seed_traits), summary, dimensions,
        ), excluded


@dataclass(frozen=True, slots=True)
class SpeciesLibraryProvenance:
    pin: SpeciesLibraryPin
    eligible_source_count: int
    excluded_source_indices: tuple[int, ...]
    selected_hierarchy_path: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    in_sample_local_update: bool = False


@dataclass(frozen=True, slots=True)
class ResolvedSpeciesLibrary:
    artifact: SpeciesLibraryArtifact | None
    provenance: SpeciesLibraryProvenance | None
    product_compatibility: Mapping[str, str]
    error: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "product_compatibility",
            MappingProxyType(
                {str(name): str(status) for name, status in dict(self.product_compatibility).items()}
            ),
        )


def _require_identifier(value: object, name: str) -> str:
    text = str(value)
    if _IDENTIFIER.fullmatch(text) is None:
        raise ValueError(f"{name.capitalize()} must be a stable lowercase identifier.")
    return text


def _optional_identifier(value: object, name: str) -> str | None:
    if value is None:
        return None
    return _require_identifier(value, name)


def _require_library_token(value: object, name: str) -> str:
    text = str(value)
    if _LIBRARY_TOKEN.fullmatch(text) is None or text in {".", ".."}:
        raise ValueError(f"{name.capitalize()} must be a safe stable token.")
    return text


def _require_sha256(value: object, name: str, *, allow_empty: bool = False) -> str:
    text = str(value)
    if allow_empty and not text:
        return text
    if _SHA256.fullmatch(text) is None:
        raise ValueError(f"{name.capitalize()} must be 64 lowercase hexadecimal characters.")
    return text


def _readonly_array(values, dtype, name: str, dimensions: int) -> np.ndarray:
    array = np.ascontiguousarray(values, dtype=dtype)
    if array.ndim != dimensions or not np.all(np.isfinite(array)):
        raise ValueError(f"{name.capitalize()} must be a finite {dimensions}-D array.")
    if array.nbytes > SPECIES_LIBRARY_MAX_ARRAY_BYTES:
        raise ValueError(f"{name.capitalize()} exceeds the library size limit.")
    array.flags.writeable = False
    return array


def _vector(values, dtype, length: int, name: str) -> np.ndarray:
    array = _readonly_array(values, dtype, name, 1)
    if len(array) != length:
        raise ValueError(f"{name.capitalize()} length does not match the bank.")
    return array


def _balanced_source_weights(weights: np.ndarray, sources: np.ndarray) -> np.ndarray:
    result = np.asarray(weights, np.float64).copy()
    unique = np.unique(sources)
    for source in unique:
        selected = sources == source
        total = max(float(result[selected].sum()), 1e-12)
        result[selected] /= total * max(len(unique), 1)
    return result.astype(np.float32)
