"""Procedural seed-instance separation from calibrated diagnostic rasters.

The algorithm deliberately separates *evidence construction* from the final
topological partition.  Seed-scale CUDA rasters can be supplied lazily; only a
bounded working copy is materialized for OpenCV's marker-controlled watershed,
which is the one topology operation not currently available in the tensor
pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum, StrEnum

import cv2
import numpy as np


MANUAL_CENTRE_SUPPRESSION_DIAMETER_FRACTION = 0.45


def _alternative_candidate_colour(score: float) -> np.ndarray:
    """Return the bounded red-to-green colour used for rejected candidates."""

    bounded_score = float(np.clip(score, 0.0, 1.0))
    channels = np.rint(
        (
            235.0 * (1.0 - bounded_score) + 35.0,
            220.0 * bounded_score + 30.0,
            55.0,
        )
    )
    return np.asarray(np.clip(channels, 0.0, 255.0), dtype=np.uint8)


class ManualSeedCentreMode(StrEnum):
    """How edited point markers interact with automatic centre proposals."""

    AUGMENT = "augment"
    REPLACE_AUTOMATIC = "replace_automatic"


class ManualSeedCentreSpace(StrEnum):
    """Coordinate frame carried by persisted or runtime centre edits."""

    SOURCE_IMAGE = "source_image"
    CORRECTED_IMAGE = "corrected_image"


class ProceduralMarkerSource(IntEnum):
    """Compact source codes aligned with result watershed-marker centres."""

    AUTOMATIC = 0
    MANUAL = 1
    ANNOTATED = 2


@dataclass(frozen=True, slots=True)
class ManualSeedCentres:
    """User-edited centre markers with an explicit image-coordinate frame.

    ``augment`` preserves automatic markers except for those within 0.45 seed
    diameter of a manual point. ``replace_automatic`` uses the supplied points
    as the complete editable marker set. Applied instance annotations remain a
    separate full-mask authority in both modes. Baseline analysis resolves
    ``source_image`` points through the current calibration; the low-level
    procedural API receives ``corrected_image`` points in its supplied raster.
    """

    centres_xy: tuple[tuple[float, float], ...] = ()
    mode: ManualSeedCentreMode = ManualSeedCentreMode.AUGMENT
    coordinate_space: ManualSeedCentreSpace = ManualSeedCentreSpace.CORRECTED_IMAGE

    def __post_init__(self) -> None:
        try:
            mode = ManualSeedCentreMode(self.mode)
        except ValueError as error:
            raise ValueError(
                "Manual centre mode must be 'augment' or 'replace_automatic'."
            ) from error
        try:
            coordinate_space = ManualSeedCentreSpace(self.coordinate_space)
        except ValueError as error:
            raise ValueError(
                "Manual centre coordinate space must be 'source_image' or "
                "'corrected_image'."
            ) from error
        normalized: list[tuple[float, float]] = []
        for centre in self.centres_xy:
            if len(centre) != 2:
                raise ValueError("Each manual seed centre must contain x and y.")
            x, y = float(centre[0]), float(centre[1])
            if not np.isfinite(x) or not np.isfinite(y):
                raise ValueError("Manual seed-centre coordinates must be finite.")
            normalized.append((x, y))
        object.__setattr__(self, "centres_xy", tuple(normalized))
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "coordinate_space", coordinate_space)

    def translated(self, dx: float, dy: float) -> ManualSeedCentres:
        """Return the same edits translated into another raster frame."""

        return ManualSeedCentres(
            tuple((x + float(dx), y + float(dy)) for x, y in self.centres_xy),
            self.mode,
            self.coordinate_space,
        )

    def in_coordinate_space(
        self,
        centres_xy: tuple[tuple[float, float], ...],
        coordinate_space: ManualSeedCentreSpace,
    ) -> ManualSeedCentres:
        """Return these edit semantics with transformed point coordinates."""

        return ManualSeedCentres(centres_xy, self.mode, coordinate_space)


@dataclass(frozen=True, slots=True)
class ProceduralInstanceSettings:
    """Controls for geometry-aware marker construction and watershed."""

    working_maximum_dimension: int = 1600
    foreground_threshold_scale: float = 0.82
    occupancy_closing_fraction: float = 0.12
    occupancy_hole_area_fraction: float = 1.25
    dish_margin_fraction: float = 0.12
    boundary_edge_weight: float = 0.35
    boundary_ridge_weight: float = 0.65
    boundary_semantic_floor: float = 0.08
    boundary_nonphysical_discount: float = 0.95
    boundary_physical_ridge_weight: float = 0.55
    boundary_trace_weight: float = 0.45
    boundary_surface_darkening_weight: float = 0.30
    trace_minimum_length_fraction: float = 0.22
    trace_convexity_weight: float = 0.60
    reference_texture_weight: float = 0.35
    centre_geometry_smoothing_fraction: float = 0.14
    centre_material_weight: float = 0.20
    centre_distance_weight: float = 0.20
    centre_flattened_grayscale_weight: float = 0.60
    centre_minimum_separation_fraction: float = 0.42
    sparse_centre_minimum_separation_fraction: float = 0.58
    sparse_seed_area_fraction: float = 0.47
    packed_seed_cell_fraction: float = 0.72
    marker_count_multiplier: float = 1.02
    minimum_marker_score: float = 0.12
    minimum_instance_area_fraction: float = 0.26
    soft_minimum_instance_area_fraction: float = 0.39
    maximum_instance_area_fraction: float = 1.45
    soft_maximum_instance_width_fraction: float = 1.15
    hard_maximum_instance_width_fraction: float = 1.35
    maximum_internal_concavity_fraction: float = 0.15
    maximum_protrusion_area_fraction: float = 0.10
    minimum_instance_solidity: float = 0.62
    maximum_instance_axis_ratio: float = 2.80
    candidate_hypotheses_per_marker: int = 5
    candidate_overlap_fraction: float = 0.02

    def __post_init__(self) -> None:
        if not 256 <= self.working_maximum_dimension <= 4096:
            raise ValueError("Working maximum dimension must be between 256 and 4096.")
        weights = (self.boundary_edge_weight, self.boundary_ridge_weight)
        if any(value < 0.0 for value in weights) or sum(weights) <= 0.0:
            raise ValueError("Boundary evidence weights must be non-negative and non-zero.")
        if not 0.0 <= self.boundary_semantic_floor <= 1.0:
            raise ValueError("Unclassified boundary floor must be between zero and one.")
        if not 0.0 <= self.boundary_nonphysical_discount <= 1.0:
            raise ValueError("Non-physical discount must be between zero and one.")
        if not 0.0 <= self.boundary_physical_ridge_weight <= 1.0:
            raise ValueError("Physical-ridge weight must be between zero and one.")
        if not 0.0 <= self.boundary_trace_weight <= 1.0:
            raise ValueError("Oriented-trace weight must be between zero and one.")
        if not 0.0 <= self.boundary_surface_darkening_weight <= 1.0:
            raise ValueError("Surface-darkening weight must be between zero and one.")
        if not 0.02 <= self.trace_minimum_length_fraction <= 2.0:
            raise ValueError("Minimum trace length must be between 0.02 and 2 seed diameters.")
        if not 0.0 <= self.trace_convexity_weight <= 1.0:
            raise ValueError("Trace convexity weight must be between zero and one.")
        if not 0.0 <= self.reference_texture_weight <= 1.0:
            raise ValueError("Reference texture weight must be between zero and one.")
        centre_weights = (
            self.centre_material_weight,
            self.centre_distance_weight,
            self.centre_flattened_grayscale_weight,
        )
        if any(value < 0.0 for value in centre_weights) or sum(centre_weights) <= 0.0:
            raise ValueError("Centre evidence weights must be non-negative and non-zero.")
        if not 0.0 < self.foreground_threshold_scale <= 2.0:
            raise ValueError("Foreground threshold scale must be positive.")
        if not 0.0 <= self.dish_margin_fraction <= 0.5:
            raise ValueError("Dish margin must be between 0 and 0.5 seed diameters.")
        if not 0.0 <= self.occupancy_hole_area_fraction <= 4.0:
            raise ValueError("Occupancy hole area must be between 0 and 4 seed areas.")
        if not 0.02 <= self.centre_geometry_smoothing_fraction <= 0.5:
            raise ValueError("Centre geometry smoothing must be between 0.02 and 0.5 diameters.")
        if not 0.1 <= self.centre_minimum_separation_fraction <= 1.0:
            raise ValueError("Centre separation must be between 0.1 and 1.0 seed diameters.")
        if not 0.1 <= self.sparse_centre_minimum_separation_fraction <= 1.0:
            raise ValueError("Sparse centre separation must be between 0.1 and 1.0 seed diameters.")
        if not 0.05 <= self.sparse_seed_area_fraction <= 2.0:
            raise ValueError(
                "Expected sparse seed area must be between 0.05 and 2 diameter squared."
            )
        if not 0.05 <= self.packed_seed_cell_fraction <= 2.0:
            raise ValueError(
                "Expected packed seed-cell area must be between 0.05 and 2 diameter squared."
            )
        if not 0.5 <= self.marker_count_multiplier <= 2.0:
            raise ValueError("Marker-count multiplier must be between 0.5 and 2.0.")
        if not (
            0.05
            <= self.minimum_instance_area_fraction
            < self.maximum_instance_area_fraction
            and 0.05
            <= self.soft_minimum_instance_area_fraction
            < self.maximum_instance_area_fraction
        ):
            raise ValueError("Instance area fractions are inconsistent.")
        if not (
            0.50
            <= self.soft_maximum_instance_width_fraction
            < self.hard_maximum_instance_width_fraction
            <= 4.0
        ):
            raise ValueError("Soft and hard instance-width limits are inconsistent.")
        if not 0.0 <= self.maximum_internal_concavity_fraction <= 1.0:
            raise ValueError("Maximum internal concavity must be between zero and one.")
        if not 0.0 <= self.maximum_protrusion_area_fraction <= 1.0:
            raise ValueError("Maximum protrusion area must be between zero and one.")
        if not 0.10 <= self.minimum_instance_solidity <= 1.0:
            raise ValueError("Minimum instance solidity must be between 0.1 and 1.")
        if not 1.0 <= self.maximum_instance_axis_ratio <= 8.0:
            raise ValueError("Maximum instance axis ratio must be between 1 and 8.")
        if not 1 <= self.candidate_hypotheses_per_marker <= 9:
            raise ValueError("Candidate hypotheses per marker must be between 1 and 9.")
        if not 0.0 <= self.candidate_overlap_fraction <= 0.25:
            raise ValueError("Candidate overlap fraction must be between zero and 0.25.")


@dataclass(frozen=True, slots=True)
class ProceduralInstanceResult:
    """Instance labels and the diagnostic evidence used to construct them.

    ``centres_xy`` are final region centroids. ``marker_centres_xy`` instead
    records the actual surviving watershed seed for each renumbered label, in
    label order, with its provenance in the aligned ``marker_sources`` array.
    """

    labels: np.ndarray
    centres_xy: np.ndarray
    marker_scores: np.ndarray
    instance_confidences: np.ndarray
    occupancy_likelihood: np.ndarray
    occupancy_mask: np.ndarray
    boundary_cost: np.ndarray
    centre_likelihood: np.ndarray
    source_shape: tuple[int, int]
    working_scale: float
    marker_centres_xy: np.ndarray = field(
        default_factory=lambda: np.empty((0, 2), np.float32)
    )
    marker_sources: np.ndarray = field(
        default_factory=lambda: np.empty(0, np.uint8)
    )
    rejected_manual_centres_xy: np.ndarray = field(
        default_factory=lambda: np.empty((0, 2), np.float32)
    )
    rejected_manual_centre_reasons: tuple[str, ...] = ()
    concavity: np.ndarray = field(
        default_factory=lambda: np.zeros((1, 1), np.uint8)
    )
    alternative_candidates_rgba: np.ndarray = field(
        default_factory=lambda: np.zeros((1, 1, 4), np.uint8)
    )
    candidate_scores: np.ndarray = field(
        default_factory=lambda: np.empty(0, np.float32)
    )
    candidate_selected: np.ndarray = field(
        default_factory=lambda: np.empty(0, bool)
    )
    instance_area_px2: np.ndarray = field(
        default_factory=lambda: np.empty(0, np.float32)
    )
    instance_maximum_width_px: np.ndarray = field(
        default_factory=lambda: np.empty(0, np.float32)
    )
    instance_width_fractions: np.ndarray = field(
        default_factory=lambda: np.empty(0, np.float32)
    )
    instance_concavity_fractions: np.ndarray = field(
        default_factory=lambda: np.empty(0, np.float32)
    )
    instance_protrusion_fractions: np.ndarray = field(
        default_factory=lambda: np.empty(0, np.float32)
    )
    instance_solidities: np.ndarray = field(
        default_factory=lambda: np.empty(0, np.float32)
    )
    instance_axis_ratios: np.ndarray = field(
        default_factory=lambda: np.empty(0, np.float32)
    )

    @property
    def count(self) -> int:
        return int(len(self.centres_xy))

    def marker_centres_for_source(
        self, source: ProceduralMarkerSource
    ) -> np.ndarray:
        """Return actual surviving watershed markers from one source class."""

        selected = np.asarray(self.marker_sources, dtype=np.uint8) == int(source)
        return np.asarray(self.marker_centres_xy, dtype=np.float32)[selected]

    @property
    def editable_marker_centres_xy(self) -> np.ndarray:
        """Return non-annotation markers suitable for initializing replace mode."""

        selected = (
            np.asarray(self.marker_sources, dtype=np.uint8)
            != int(ProceduralMarkerSource.ANNOTATED)
        )
        return np.asarray(self.marker_centres_xy, dtype=np.float32)[selected]

    def instance_rgba(self) -> np.ndarray:
        """Return deterministic colours for the integer instance identities."""

        identifiers = np.asarray(self.labels, dtype=np.uint32)
        red = np.uint8((identifiers * 73 + 41) % 239 + 16)
        green = np.uint8((identifiers * 151 + 17) % 239 + 16)
        blue = np.uint8((identifiers * 199 + 89) % 239 + 16)
        alpha = np.uint8(identifiers > 0) * 220
        return np.dstack((red, green, blue, alpha))

    def confidence_raster(self) -> np.ndarray:
        """Map compact per-instance confidence back onto each assigned pixel."""

        lookup = np.zeros(int(np.max(self.labels, initial=0)) + 1, np.float32)
        available = min(len(self.instance_confidences), max(0, len(lookup) - 1))
        if available:
            lookup[1 : available + 1] = self.instance_confidences[:available]
        return np.uint8(np.clip(np.rint(lookup[self.labels] * 255.0), 0, 255))

    def statistics_for_label(self, label: int) -> dict[str, float | int]:
        """Return compact, source-resolution geometry for one visible label."""

        index = int(label) - 1
        if index < 0 or index >= self.count:
            raise ValueError(f"Procedural label {label} is not present.")
        return {
            "label": int(label),
            "area_px2": float(self.instance_area_px2[index]),
            "maximum_width_px": float(self.instance_maximum_width_px[index]),
            "width_fraction": float(self.instance_width_fractions[index]),
            "concavity_fraction": float(self.instance_concavity_fractions[index]),
            "protrusion_fraction": float(self.instance_protrusion_fractions[index]),
            "solidity": float(self.instance_solidities[index]),
            "axis_ratio": float(self.instance_axis_ratios[index]),
            "confidence": float(self.instance_confidences[index]),
            "marker_score": float(self.marker_scores[index]),
        }


@dataclass(slots=True)
class _InstanceCandidate:
    """One compact marker-conditioned mask hypothesis at working resolution."""

    marker_index: int
    hypothesis_index: int
    x0: int
    y0: int
    mask: np.ndarray
    score: float
    area: float
    width_fraction: float
    concavity_fraction: float
    concavity_mask: np.ndarray
    protrusion_fraction: float
    solidity: float
    axis_ratio: float
    boundary_support: float
    selected: bool = False

    @property
    def x1(self) -> int:
        return self.x0 + int(self.mask.shape[1])

    @property
    def y1(self) -> int:
        return self.y0 + int(self.mask.shape[0])


@dataclass(frozen=True, slots=True)
class PreparedProceduralInstanceInputs:
    """One bounded host copy of the parameter-independent separator inputs.

    Preparing a full-resolution CUDA analysis for OpenCV topology requires one
    resize and device-to-host transfer per evidence raster.  A short parameter
    search can safely reuse this immutable working set because the separator
    only reads these arrays and constructs fresh parameter-dependent products.
    """

    source_shape: tuple[int, int]
    working_scale: float
    working_maximum_dimension: int
    seed_diameter_px: float
    valid_mask: np.ndarray
    material_probability: np.ndarray | None
    foreground_probability: np.ndarray
    foreground_noise_probability: np.ndarray
    background_probability: np.ndarray
    refined_background_probability: np.ndarray
    edge_magnitude: np.ndarray
    edge_ridges: np.ndarray
    physical_edge_probability: np.ndarray
    non_edge_probability: np.ndarray
    normalized_net_physical_edge_probability: np.ndarray
    thinned_reference_edge_ridges: np.ndarray
    oriented_edge_trace_labels: np.ndarray
    oriented_edge_trace_continuity: np.ndarray
    semantic_edge_evidence_available: bool
    reference_surface_probability: np.ndarray | None
    flattened_grayscale: np.ndarray | None
    surface_darkening_magnitude: np.ndarray | None

    @property
    def working_shape(self) -> tuple[int, int]:
        return tuple(int(value) for value in self.valid_mask.shape)

    def annotation_targets(
        self,
        annotations: np.ndarray,
        *,
        mask_to_valid: bool = True,
    ) -> np.ndarray:
        """Return nearest-resized labels aligned with this working set.

        Invalid pixels are excluded by default so dish-exterior annotation
        strokes cannot enter a fitting score as missed target area.
        """

        values = _resize(
            np.asarray(annotations, dtype=np.uint16),
            (self.working_shape[1], self.working_shape[0]),
            cv2.INTER_NEAREST,
        )
        values = np.asarray(values, dtype=np.uint16).copy()
        if mask_to_valid:
            values[~self.valid_mask] = 0
        return values


def _u8(values) -> np.ndarray:
    source = np.asarray(values)
    if source.dtype == np.uint8:
        return source
    finite = np.nan_to_num(source, nan=0.0, posinf=255.0, neginf=0.0)
    if finite.size and float(finite.max()) <= 1.0:
        finite = finite * 255.0
    return np.uint8(np.clip(np.rint(finite), 0, 255))


def _resize(values: np.ndarray, size: tuple[int, int], interpolation: int) -> np.ndarray:
    width, height = size
    if values.shape[:2] == (height, width):
        return values
    return cv2.resize(values, size, interpolation=interpolation)


def _working_u8(values, size: tuple[int, int], interpolation: int) -> np.ndarray:
    """Download only the bounded topology working raster from a GPU input."""

    if hasattr(values, "gpu_tensor"):
        import torch
        import torch.nn.functional as functional

        tensor = values.gpu_tensor(dtype=torch.float32)
        if tensor.ndim == 2:
            tensor = tensor[None, None]
        elif tensor.ndim == 3:
            tensor = tensor[None]
        width, height = size
        if tensor.shape[-2:] != (height, width):
            if interpolation == cv2.INTER_NEAREST:
                mode = "nearest"
            elif interpolation == cv2.INTER_AREA:
                mode = "area"
            else:
                mode = "bilinear"
            arguments = {} if mode in {"nearest", "area"} else {"align_corners": False}
            tensor = functional.interpolate(tensor, (height, width), mode=mode, **arguments)
        result = tensor[0, 0].detach().cpu().numpy()
        return np.uint8(np.clip(np.rint(result), 0, 255))
    return _resize(_u8(values), size, interpolation)


def _working_labels(values, size: tuple[int, int]) -> np.ndarray:
    """Download a bounded categorical raster without truncating identifiers."""

    if hasattr(values, "gpu_tensor"):
        import torch
        import torch.nn.functional as functional

        tensor = values.gpu_tensor(dtype=torch.float32)
        if tensor.ndim == 2:
            tensor = tensor[None, None]
        elif tensor.ndim == 3:
            tensor = tensor[None]
        width, height = size
        if tensor.shape[-2:] != (height, width):
            tensor = functional.interpolate(tensor, (height, width), mode="nearest")
        return tensor[0, 0].round().to(dtype=torch.int32).detach().cpu().numpy()
    source = np.asarray(values)
    return np.asarray(_resize(source, size, cv2.INTER_NEAREST), dtype=np.int32)


def _readonly(values: np.ndarray) -> np.ndarray:
    """Return a contiguous array protected from accidental trial mutation."""

    result = np.ascontiguousarray(values)
    result.setflags(write=False)
    return result


def _ellipse_kernel(radius: float) -> np.ndarray:
    integer_radius = max(1, int(round(radius)))
    size = integer_radius * 2 + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _supplement_packed_markers(
    centre_likelihood: np.ndarray,
    eligible_mask: np.ndarray,
    *,
    seed_diameter: float,
    minimum_separation_fraction: float,
    minimum_score: float,
    target_count: int,
    anchor_x: np.ndarray,
    anchor_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fill a dense-component marker deficit with a seed-scale packing prior.

    Regional maxima are high-quality marker proposals, but in a packed dish a
    slightly stronger neighbouring seed can suppress a true maximum over most
    of the configured NMS radius.  That formerly made the geometric count cap
    a cap in name only: the separator could return a third of the number of
    seeds implied by calibrated seed area.

    This fallback does *not* find peaks in edge texture.  It samples the already
    seed-scale-smoothed centre field on a coarse lattice and greedily adds the
    best well-separated location.  Priority combines interior quality with
    distance from every accepted/manual marker.  Consequently its population
    is bounded by the calibrated area target, and coat texture cannot create
    extra markers.  Callers retain authority over where it may operate through
    ``eligible_mask`` (semantic boundary barriers and annotated regions are
    excluded there).
    """

    target_count = max(0, int(target_count))
    anchors_x = np.asarray(anchor_x, dtype=np.int32).reshape(-1)
    anchors_y = np.asarray(anchor_y, dtype=np.int32).reshape(-1)
    if target_count <= len(anchors_x) or not np.any(eligible_mask):
        return (
            np.empty(0, np.int32),
            np.empty(0, np.int32),
            np.empty(0, np.float32),
        )

    height, width = centre_likelihood.shape
    # Six to ten samples across a typical seed retain positional precision
    # without turning a bounded 1600-pixel topology raster into a million-item
    # candidate set.  Each staggered row samples the gaps of its predecessor.
    lattice_step = max(2, int(round(seed_diameter * 0.10)))
    rows = np.arange(lattice_step // 2, height, lattice_step, dtype=np.int32)
    candidate_x_parts: list[np.ndarray] = []
    candidate_y_parts: list[np.ndarray] = []
    for row_index, y in enumerate(rows):
        offset = lattice_step // 2
        if row_index & 1:
            offset += lattice_step // 2
        xx = np.arange(offset, width, lattice_step, dtype=np.int32)
        if not len(xx):
            continue
        allowed = eligible_mask[int(y), xx]
        if np.any(allowed):
            candidate_x_parts.append(xx[allowed])
            candidate_y_parts.append(np.full(np.count_nonzero(allowed), y, np.int32))
    if not candidate_x_parts:
        return (
            np.empty(0, np.int32),
            np.empty(0, np.int32),
            np.empty(0, np.float32),
        )

    candidate_x = np.concatenate(candidate_x_parts)
    candidate_y = np.concatenate(candidate_y_parts)
    candidate_scores = np.asarray(
        centre_likelihood[candidate_y, candidate_x], dtype=np.float32
    )
    retained = candidate_scores >= float(minimum_score)
    candidate_x = candidate_x[retained]
    candidate_y = candidate_y[retained]
    candidate_scores = candidate_scores[retained]
    if not len(candidate_x):
        return (
            np.empty(0, np.int32),
            np.empty(0, np.int32),
            np.empty(0, np.float32),
        )

    minimum_distance = max(
        float(lattice_step),
        float(seed_diameter) * float(minimum_separation_fraction),
    )
    minimum_squared = minimum_distance * minimum_distance
    nearest_squared = np.full(len(candidate_x), np.inf, np.float32)
    for x, y in zip(anchors_x, anchors_y, strict=True):
        squared = (candidate_x - int(x)) ** 2 + (candidate_y - int(y)) ** 2
        nearest_squared = np.minimum(nearest_squared, squared.astype(np.float32))

    robust_scale = float(np.quantile(candidate_scores, 0.98))
    quality = np.clip(candidate_scores / max(1e-6, robust_scale), 0.0, 1.0)
    added_x: list[int] = []
    added_y: list[int] = []
    added_scores: list[float] = []
    required = target_count - len(anchors_x)
    spatial_saturation = max(minimum_distance, seed_diameter * 0.90)
    for _ in range(required):
        allowed = nearest_squared >= minimum_squared
        if not np.any(allowed):
            break
        # Once candidates satisfy the strict separation constraint, prefer
        # plausible interiors but still cover an otherwise marker-free basin.
        spatial = np.clip(
            np.sqrt(nearest_squared) / max(1e-6, spatial_saturation),
            0.0,
            1.0,
        )
        priority = np.where(allowed, 0.68 * quality + 0.32 * spatial, -1.0)
        selected = int(np.argmax(priority))
        x = int(candidate_x[selected])
        y = int(candidate_y[selected])
        added_x.append(x)
        added_y.append(y)
        added_scores.append(float(candidate_scores[selected]))
        squared = (candidate_x - x) ** 2 + (candidate_y - y) ** 2
        nearest_squared = np.minimum(nearest_squared, squared.astype(np.float32))

    return (
        np.asarray(added_x, np.int32),
        np.asarray(added_y, np.int32),
        np.asarray(added_scores, np.float32),
    )


def _renumber_labels(labels: np.ndarray) -> np.ndarray:
    identifiers = np.unique(labels)
    identifiers = identifiers[identifiers > 0]
    if not len(identifiers):
        return np.zeros(labels.shape, dtype=np.int32)
    lookup = np.zeros(int(identifiers[-1]) + 1, dtype=np.int32)
    lookup[identifiers] = np.arange(1, len(identifiers) + 1, dtype=np.int32)
    return lookup[np.asarray(labels, dtype=np.int32)]


def _material_occupancy(
    occupancy_u8: np.ndarray,
    threshold: float,
    interior_valid: np.ndarray,
    diameter: float,
    settings: ProceduralInstanceSettings,
) -> np.ndarray:
    """Build one material support hypothesis from an absolute threshold."""

    occupancy = (occupancy_u8 >= float(threshold)) & interior_valid
    occupancy = cv2.morphologyEx(
        np.uint8(occupancy) * 255,
        cv2.MORPH_CLOSE,
        _ellipse_kernel(diameter * settings.occupancy_closing_fraction),
    ) > 0
    occupancy &= interior_valid
    flood = np.uint8(occupancy) * 255
    cv2.floodFill(flood, None, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    _count, component_labels, component_stats, _centroids = (
        cv2.connectedComponentsWithStats(np.uint8(holes > 0), connectivity=8)
    )
    maximum_hole_area = (
        diameter * diameter * settings.occupancy_hole_area_fraction
    )
    fill_component = component_stats[:, cv2.CC_STAT_AREA] <= maximum_hole_area
    fill_component[0] = False
    occupancy |= fill_component[component_labels]
    return cv2.morphologyEx(
        np.uint8(occupancy) * 255,
        cv2.MORPH_OPEN,
        _ellipse_kernel(max(1.0, diameter * 0.025)),
    ) > 0


def _candidate_overlap_pixels(
    first: _InstanceCandidate, second: _InstanceCandidate
) -> int:
    left = max(first.x0, second.x0)
    top = max(first.y0, second.y0)
    right = min(first.x1, second.x1)
    bottom = min(first.y1, second.y1)
    if right <= left or bottom <= top:
        return 0
    first_roi = first.mask[
        top - first.y0 : bottom - first.y0,
        left - first.x0 : right - first.x0,
    ]
    second_roi = second.mask[
        top - second.y0 : bottom - second.y0,
        left - second.x0 : right - second.x0,
    ]
    return int(np.count_nonzero(first_roi & second_roi))


def _candidate_from_component(
    component: np.ndarray,
    *,
    marker_index: int,
    hypothesis_index: int,
    marker_score: float,
    marker_source: int,
    boundary: np.ndarray,
    diameter: float,
    expected_area_fraction: float,
    settings: ProceduralInstanceSettings,
    origin_xy: tuple[int, int] = (0, 0),
) -> tuple[_InstanceCandidate | None, str]:
    """Measure, hard-filter, and softly score one compact candidate mask."""

    rows, columns = np.nonzero(component)
    if not len(rows):
        return None, "instance_below_minimum_area"
    local_x0, local_x1 = int(columns.min()), int(columns.max()) + 1
    local_y0, local_y1 = int(rows.min()), int(rows.max()) + 1
    local = np.asarray(
        component[local_y0:local_y1, local_x0:local_x1], dtype=np.uint8
    )
    x0 = int(origin_xy[0]) + local_x0
    y0 = int(origin_xy[1]) + local_y0
    area = float(np.count_nonzero(local))
    hard_minimum_area = diameter * diameter * settings.minimum_instance_area_fraction
    soft_minimum_area = (
        diameter * diameter * settings.soft_minimum_instance_area_fraction
    )
    maximum_area = diameter * diameter * settings.maximum_instance_area_fraction
    annotated = marker_source == int(ProceduralMarkerSource.ANNOTATED)
    if not annotated and area < hard_minimum_area:
        return None, "instance_below_minimum_area"
    if area > maximum_area:
        return None, "instance_above_maximum_area"
    contours, _hierarchy = cv2.findContours(
        local, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None, "instance_missing_contour"
    contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    solidity = area / max(1.0, hull_area)
    concavity = max(0.0, hull_area - area) / max(1.0, area)
    # Visualize only exterior-connected pockets between the observed perimeter
    # and its convex envelope. Enclosed internal holes are intentionally not
    # called perimeter concavity.
    padded = np.pad(local > 0, 1, mode="constant", constant_values=False)
    hull_mask = np.zeros_like(padded, np.uint8)
    cv2.fillConvexPoly(hull_mask, np.int32(hull[:, 0, :] + 1), 1)
    background_components, background_labels = cv2.connectedComponents(
        np.uint8(~padded), connectivity=8
    )
    del background_components
    exterior_label = int(background_labels[0, 0])
    concavity_mask = (
        (hull_mask > 0) & (background_labels == exterior_label)
    )[1:-1, 1:-1]
    (_centre, (box_width, box_height), _angle) = cv2.minAreaRect(contour)
    width_fraction = max(float(box_width), float(box_height)) / max(1.0, diameter)
    axis_ratio = max(box_width, box_height) / max(
        1.0, min(box_width, box_height)
    )
    opened = cv2.morphologyEx(
        local,
        cv2.MORPH_OPEN,
        _ellipse_kernel(max(1.0, diameter * 0.10)),
    ) > 0
    protrusion = float(np.count_nonzero((local > 0) & ~opened)) / max(1.0, area)
    # Annotation-derived markers identify trusted centres, not permission for
    # the surrounding watershed to grow into an implausible Frankenstein
    # region. The hard upper-area/width/concavity/protrusion/solidity/axis
    # constraints therefore apply to every inferred candidate. Only the hard
    # minimum-area rule is waived for a deliberately tiny reviewed instance.
    if concavity > settings.maximum_internal_concavity_fraction:
        return None, "instance_above_maximum_concavity"
    if protrusion > settings.maximum_protrusion_area_fraction:
        return None, "instance_above_maximum_protrusion"
    if solidity < settings.minimum_instance_solidity:
        return None, "instance_below_minimum_solidity"
    if axis_ratio > settings.maximum_instance_axis_ratio:
        return None, "instance_above_maximum_axis_ratio"
    if width_fraction > settings.hard_maximum_instance_width_fraction:
        return None, "instance_above_hard_maximum_width"

    perimeter = cv2.morphologyEx(local, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    boundary_values = boundary[y0 : y0 + local.shape[0], x0 : x0 + local.shape[1]][
        perimeter
    ]
    boundary_support = (
        float(np.mean(boundary_values)) if boundary_values.size else 0.0
    )
    expected_area = max(1.0, diameter * diameter * expected_area_fraction)
    area_confidence = float(
        np.exp(-0.5 * (np.log(max(area / expected_area, 1e-4)) / 0.58) ** 2)
    )
    minimum_penalty = min(1.0, area / max(1.0, soft_minimum_area))
    if width_fraction <= settings.soft_maximum_instance_width_fraction:
        width_penalty = 1.0
    else:
        width_penalty = max(
            0.0,
            1.0
            - (
                width_fraction - settings.soft_maximum_instance_width_fraction
            )
            / max(
                1e-6,
                settings.hard_maximum_instance_width_fraction
                - settings.soft_maximum_instance_width_fraction,
            ),
        )
    shape_penalty = (
        max(0.0, 1.0 - concavity / max(1e-6, settings.maximum_internal_concavity_fraction))
        * max(0.0, 1.0 - protrusion / max(1e-6, settings.maximum_protrusion_area_fraction))
    ) ** 0.25
    score = float(
        np.clip(
            (
                0.42 * float(marker_score)
                + 0.36 * boundary_support
                + 0.22 * area_confidence
            )
            * minimum_penalty
            * width_penalty
            * shape_penalty,
            0.0,
            1.0,
        )
    )
    if annotated:
        score = 1.0
    return (
        _InstanceCandidate(
            marker_index=marker_index,
            hypothesis_index=hypothesis_index,
            x0=x0,
            y0=y0,
            mask=local > 0,
            score=score,
            area=area,
            width_fraction=width_fraction,
            concavity_fraction=concavity,
            concavity_mask=concavity_mask,
            protrusion_fraction=protrusion,
            solidity=solidity,
            axis_ratio=axis_ratio,
            boundary_support=boundary_support,
        ),
        "",
    )


def _manual_centres_at_working_scale(
    edits: ManualSeedCentres | None,
    prepared: PreparedProceduralInstanceInputs,
    occupancy: np.ndarray,
    annotations: np.ndarray | None,
    annotation_x: np.ndarray,
    annotation_y: np.ndarray,
    *,
    annotation_suppression_radius: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[str, ...],
]:
    """Validate corrected/source-raster points for bounded CPU topology.

    Points never expand the material mask. A click on valid dish background is
    therefore reported as ``outside_material`` instead of silently creating an
    image region that the evidence did not classify as seed material.
    """

    if edits is not None and not isinstance(edits, ManualSeedCentres):
        raise TypeError("manual_seed_centres must be a ManualSeedCentres value.")
    if edits is None or not edits.centres_xy:
        return (
            np.empty(0, np.int32),
            np.empty(0, np.int32),
            np.empty((0, 2), np.float32),
            np.empty((0, 2), np.float32),
            (),
        )
    source_height, source_width = prepared.source_shape
    height, width = prepared.working_shape
    scale = float(prepared.working_scale)
    accepted_x: list[int] = []
    accepted_y: list[int] = []
    accepted_source: list[tuple[float, float]] = []
    rejected: list[tuple[float, float]] = []
    reasons: list[str] = []
    occupied_working_pixels: set[tuple[int, int]] = set()
    annotation_radius_squared = float(annotation_suppression_radius) ** 2

    def reject(x: float, y: float, reason: str) -> None:
        rejected.append((float(x), float(y)))
        reasons.append(reason)

    for x, y in edits.centres_xy:
        if x < 0.0 or y < 0.0 or x >= source_width or y >= source_height:
            reject(x, y, "outside_source")
            continue
        working_x = int(round(float(x) * scale))
        working_y = int(round(float(y) * scale))
        working_x = int(np.clip(working_x, 0, width - 1))
        working_y = int(np.clip(working_y, 0, height - 1))
        if not prepared.valid_mask[working_y, working_x]:
            reject(x, y, "outside_valid_region")
            continue
        if not occupancy[working_y, working_x]:
            reject(x, y, "outside_material")
            continue
        if (
            annotations is not None
            and annotations[working_y, working_x] > 0
        ) or (
            len(annotation_x)
            and np.any(
                (annotation_x - working_x) ** 2
                + (annotation_y - working_y) ** 2
                < annotation_radius_squared
            )
        ):
            reject(x, y, "near_annotated_instance")
            continue
        key = (working_x, working_y)
        if key in occupied_working_pixels:
            reject(x, y, "duplicate_working_pixel")
            continue
        occupied_working_pixels.add(key)
        accepted_x.append(working_x)
        accepted_y.append(working_y)
        accepted_source.append((float(x), float(y)))

    return (
        np.asarray(accepted_x, np.int32),
        np.asarray(accepted_y, np.int32),
        np.asarray(accepted_source, np.float32).reshape(-1, 2),
        np.asarray(rejected, np.float32).reshape(-1, 2),
        tuple(reasons),
    )


def _trace_ellipse_geometry(
    points_xy: np.ndarray, diameter: float
) -> tuple[float, tuple[float, float] | None]:
    """Score one trace and return its fitted seed-centre hypothesis.

    A long straight coat stripe can have excellent pixel continuity, so length
    alone is not physical-boundary evidence.  An ellipse fitted to a bounded
    sample gives a cheap, ordering-independent check that the trace turns in
    one direction at a plausible seed scale.  Partial arcs are allowed; the
    score is evidence, not a requirement that the component close.
    """

    if len(points_xy) < 8:
        return 0.0, None
    if len(points_xy) > 2048:
        indices = np.linspace(0, len(points_xy) - 1, 2048, dtype=np.int32)
        points_xy = points_xy[indices]
    try:
        (centre_x, centre_y), (axis_a, axis_b), angle_degrees = cv2.fitEllipse(
            np.asarray(points_xy, dtype=np.float32).reshape(-1, 1, 2)
        )
    except cv2.error:
        return 0.0, None
    semi_a = 0.5 * float(axis_a)
    semi_b = 0.5 * float(axis_b)
    smaller = min(semi_a, semi_b)
    larger = max(semi_a, semi_b)
    if (
        smaller < diameter * 0.08
        or larger > diameter * 1.35
        or larger / max(smaller, 1e-6) > 4.0
    ):
        return 0.0, None

    angle = np.deg2rad(float(angle_degrees))
    cosine = float(np.cos(angle))
    sine = float(np.sin(angle))
    relative_x = points_xy[:, 0].astype(np.float32) - float(centre_x)
    relative_y = points_xy[:, 1].astype(np.float32) - float(centre_y)
    rotated_x = cosine * relative_x + sine * relative_y
    rotated_y = -sine * relative_x + cosine * relative_y
    radius = np.sqrt(
        (rotated_x / max(semi_a, 1e-6)) ** 2
        + (rotated_y / max(semi_b, 1e-6)) ** 2
    )
    residual = float(np.median(np.abs(radius - 1.0)))
    fit_score = float(np.exp(-0.5 * (residual / 0.16) ** 2))

    angles = np.mod(
        np.arctan2(
            rotated_y / max(semi_b, 1e-6),
            rotated_x / max(semi_a, 1e-6),
        ),
        2.0 * np.pi,
    )
    occupied_bins = np.unique(np.floor(angles * (36.0 / (2.0 * np.pi))).astype(np.int32))
    arc_coverage = len(occupied_bins) / 36.0
    arc_score = float(np.clip(arc_coverage / 0.16, 0.0, 1.0))
    scale_score = float(
        np.clip(smaller / max(diameter * 0.18, 1e-6), 0.0, 1.0)
        * np.clip((diameter * 1.35 - larger) / max(diameter * 0.35, 1e-6), 0.0, 1.0)
    )
    score = float(np.clip(fit_score * arc_score * scale_score, 0.0, 1.0))
    return score, (float(centre_x), float(centre_y))


def _oriented_trace_support(
    trace_labels: np.ndarray,
    trace_continuity: np.ndarray,
    diameter: float,
    settings: ProceduralInstanceSettings,
) -> np.ndarray:
    """Convert categorical traces into length/continuity/convexity evidence."""

    labels = np.asarray(trace_labels, dtype=np.int32)
    selected_y, selected_x = np.nonzero(labels > 0)
    if not len(selected_y):
        return np.zeros(labels.shape, np.float32)
    identifiers = labels[selected_y, selected_x]
    order = np.argsort(identifiers, kind="stable")
    identifiers = identifiers[order]
    selected_y = selected_y[order]
    selected_x = selected_x[order]
    unique_ids, starts, counts = np.unique(
        identifiers, return_index=True, return_counts=True
    )
    continuity = np.asarray(trace_continuity, dtype=np.float32)
    use_continuity = bool(np.any(continuity > 1e-6))
    selected_scores = np.zeros(len(selected_y), np.float32)
    minimum_length = diameter * settings.trace_minimum_length_fraction
    saturation_span = max(diameter * 0.55, 1.0)
    for _identifier, start, count in zip(unique_ids, starts, counts, strict=True):
        if count < minimum_length:
            continue
        stop = int(start + count)
        yy = selected_y[start:stop]
        xx = selected_x[start:stop]
        excess_length = max(0.0, float(count) - minimum_length)
        length_score = 1.0 - float(np.exp(-excess_length / saturation_span))
        length_score = max(0.12, length_score)
        continuity_score = (
            float(np.mean(continuity[yy, xx])) if use_continuity else 1.0
        )
        convexity, _fitted_centre = _trace_ellipse_geometry(
            np.column_stack((xx, yy)), diameter
        )
        convexity_factor = (
            1.0
            - settings.trace_convexity_weight
            + settings.trace_convexity_weight * convexity
        )
        component_score = float(
            np.clip(
                length_score * (0.30 + 0.70 * continuity_score) * convexity_factor,
                0.0,
                1.0,
            )
        )
        selected_scores[start:stop] = component_score
    if not np.any(selected_scores > 0.0):
        return np.zeros(labels.shape, np.float32)
    support = np.zeros(labels.shape, np.float32)
    support[selected_y, selected_x] = selected_scores
    support = cv2.dilate(support, np.ones((3, 3), np.uint8))
    return cv2.GaussianBlur(support, (0, 0), sigmaX=0.65)


def prepare_procedural_instance_inputs(
    valid_mask,
    seed_diameter_px: float,
    *,
    material_probability=None,
    foreground_probability,
    foreground_noise_probability,
    background_probability,
    refined_background_probability,
    edge_magnitude,
    edge_ridges,
    physical_edge_probability=None,
    non_edge_probability=None,
    normalized_net_physical_edge_probability=None,
    thinned_reference_edge_ridges=None,
    oriented_edge_trace_labels=None,
    oriented_edge_trace_continuity=None,
    reference_surface_probability=None,
    flattened_grayscale=None,
    surface_darkening_magnitude=None,
    working_maximum_dimension: int = 1600,
) -> PreparedProceduralInstanceInputs:
    """Materialize the separator's bounded, parameter-independent inputs once."""

    if not 256 <= int(working_maximum_dimension) <= 4096:
        raise ValueError("Working maximum dimension must be between 256 and 4096.")
    source_height, source_width = tuple(int(value) for value in valid_mask.shape[-2:])
    scale = min(
        1.0,
        float(working_maximum_dimension) / max(source_height, source_width),
    )
    width = max(16, int(round(source_width * scale)))
    height = max(16, int(round(source_height * scale)))
    size = (width, height)

    valid = _readonly(_working_u8(valid_mask, size, cv2.INTER_NEAREST) > 0)
    material = (
        None
        if material_probability is None
        else _readonly(
            _working_u8(material_probability, size, cv2.INTER_AREA) / 255.0
        )
    )
    foreground = _readonly(
        _working_u8(foreground_probability, size, cv2.INTER_AREA) / 255.0
    )
    foreground_noise = _readonly(
        _working_u8(foreground_noise_probability, size, cv2.INTER_AREA) / 255.0
    )
    background = _readonly(
        _working_u8(background_probability, size, cv2.INTER_AREA) / 255.0
    )
    refined_background = _readonly(
        _working_u8(refined_background_probability, size, cv2.INTER_AREA) / 255.0
    )
    edge = _readonly(_working_u8(edge_magnitude, size, cv2.INTER_AREA) / 255.0)
    ridges = _readonly(_working_u8(edge_ridges, size, cv2.INTER_AREA) / 255.0)
    physical_probability = (
        _readonly(np.zeros_like(edge))
        if physical_edge_probability is None
        else _readonly(
            _working_u8(physical_edge_probability, size, cv2.INTER_AREA) / 255.0
        )
    )
    nonphysical_probability = (
        _readonly(np.zeros_like(edge))
        if non_edge_probability is None
        else _readonly(
            _working_u8(non_edge_probability, size, cv2.INTER_AREA) / 255.0
        )
    )
    normalized_net_probability = (
        _readonly(np.zeros_like(edge))
        if normalized_net_physical_edge_probability is None
        else _readonly(
            _working_u8(
                normalized_net_physical_edge_probability,
                size,
                cv2.INTER_AREA,
            )
            / 255.0
        )
    )
    reference_ridges = (
        _readonly(np.zeros_like(edge))
        if thinned_reference_edge_ridges is None
        else _readonly(
            _working_u8(
                thinned_reference_edge_ridges, size, cv2.INTER_AREA
            ) / 255.0
        )
    )
    trace_labels = (
        _readonly(np.zeros(edge.shape, np.int32))
        if oriented_edge_trace_labels is None
        else _readonly(_working_labels(oriented_edge_trace_labels, size))
    )
    trace_continuity = (
        _readonly(np.zeros_like(edge))
        if oriented_edge_trace_continuity is None
        else _readonly(
            _working_u8(
                oriented_edge_trace_continuity, size, cv2.INTER_AREA
            ) / 255.0
        )
    )
    semantic_available = bool(
        np.any(physical_probability > (1.0 / 255.0))
        or np.any(nonphysical_probability > (1.0 / 255.0))
        or np.any(normalized_net_probability > (1.0 / 255.0))
    )
    reference_surface = (
        None
        if reference_surface_probability is None
        else _readonly(
            _working_u8(reference_surface_probability, size, cv2.INTER_AREA) / 255.0
        )
    )
    flattened = (
        None
        if flattened_grayscale is None
        else _readonly(
            _working_u8(flattened_grayscale, size, cv2.INTER_AREA) / 255.0
        )
    )
    surface_darkening = (
        None
        if surface_darkening_magnitude is None
        else _readonly(
            _working_u8(surface_darkening_magnitude, size, cv2.INTER_AREA) / 255.0
        )
    )
    return PreparedProceduralInstanceInputs(
        source_shape=(source_height, source_width),
        working_scale=float(scale),
        working_maximum_dimension=int(working_maximum_dimension),
        seed_diameter_px=max(6.0, float(seed_diameter_px) * scale),
        valid_mask=valid,
        material_probability=material,
        foreground_probability=foreground,
        foreground_noise_probability=foreground_noise,
        background_probability=background,
        refined_background_probability=refined_background,
        edge_magnitude=edge,
        edge_ridges=ridges,
        physical_edge_probability=physical_probability,
        non_edge_probability=nonphysical_probability,
        normalized_net_physical_edge_probability=normalized_net_probability,
        thinned_reference_edge_ridges=reference_ridges,
        oriented_edge_trace_labels=trace_labels,
        oriented_edge_trace_continuity=trace_continuity,
        semantic_edge_evidence_available=semantic_available,
        reference_surface_probability=reference_surface,
        flattened_grayscale=flattened,
        surface_darkening_magnitude=surface_darkening,
    )


def procedural_seed_instances_from_prepared(
    prepared: PreparedProceduralInstanceInputs,
    *,
    seed_instance_annotations=None,
    manual_seed_centres: ManualSeedCentres | None = None,
    settings: ProceduralInstanceSettings = ProceduralInstanceSettings(),
) -> ProceduralInstanceResult:
    """Run only parameter-dependent CPU topology on a prepared working set."""

    if settings.working_maximum_dimension != prepared.working_maximum_dimension:
        raise ValueError(
            "Prepared inputs use a different working maximum dimension; "
            "prepare them again for these settings."
        )
    if manual_seed_centres is not None and not isinstance(
        manual_seed_centres, ManualSeedCentres
    ):
        raise TypeError("manual_seed_centres must be a ManualSeedCentres value.")
    if (
        manual_seed_centres is not None
        and manual_seed_centres.coordinate_space
        is not ManualSeedCentreSpace.CORRECTED_IMAGE
    ):
        raise ValueError(
            "Prepared procedural inputs require centres in their source raster's "
            "corrected-image coordinate frame."
        )
    source_height, source_width = prepared.source_shape
    scale = prepared.working_scale
    height, width = prepared.working_shape
    size = (width, height)
    valid = prepared.valid_mask
    diameter = prepared.seed_diameter_px

    foreground = prepared.foreground_probability
    foreground_noise = prepared.foreground_noise_probability
    background = prepared.background_probability
    refined_background = prepared.refined_background_probability
    edge = prepared.edge_magnitude
    ridges = prepared.edge_ridges
    physical_probability = prepared.physical_edge_probability
    nonphysical_probability = prepared.non_edge_probability
    normalized_net_probability = (
        prepared.normalized_net_physical_edge_probability
    )
    reference_ridges = prepared.thinned_reference_edge_ridges
    trace_labels = prepared.oriented_edge_trace_labels
    trace_continuity = prepared.oriented_edge_trace_continuity
    reference_surface = prepared.reference_surface_probability
    flattened_grayscale = prepared.flattened_grayscale
    surface_darkening = prepared.surface_darkening_magnitude

    if prepared.material_probability is not None:
        # The hierarchical material node has already combined colour, texture,
        # prototypes, Background, Other, contradiction, and unknown evidence.
        # Recombining raw channels here would recreate the permissive max/min
        # failure mode and make the procedural overlay disagree with its graph
        # input.
        occupancy_likelihood = prepared.material_probability.copy()
    else:
        # Compatibility for direct callers and historical fitting fixtures.
        inverse_background = 1.0 - np.minimum(background, refined_background)
        occupancy_likelihood = np.maximum(foreground, foreground_noise)
        if reference_surface is not None:
            occupancy_likelihood = (
                occupancy_likelihood * (1.0 - settings.reference_texture_weight)
                + reference_surface * settings.reference_texture_weight
            )
        occupancy_likelihood *= 0.30 + 0.70 * inverse_background
    occupancy_likelihood *= valid
    occupancy_u8 = np.uint8(np.clip(np.rint(occupancy_likelihood * 255.0), 0, 255))
    valid_values = occupancy_u8[valid]
    otsu_input = valid_values.reshape(-1, 1) if valid_values.size else np.zeros((1, 1), np.uint8)
    otsu_threshold, _ = cv2.threshold(
        otsu_input, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )
    threshold = float(np.clip(
        otsu_threshold * settings.foreground_threshold_scale,
        18.0,
        180.0,
    ))
    interior_valid = cv2.erode(
        np.uint8(valid) * 255,
        _ellipse_kernel(diameter * settings.dish_margin_fraction),
    ) > 0
    # Suppress the bright dish rim before closing. Colour/noise likelihood can
    # respond mainly to a patterned perimeter, so each hypothesis also fills
    # only enclosed seed-sized holes.
    occupancy = _material_occupancy(
        occupancy_u8, threshold, interior_valid, diameter, settings
    )

    # Broad Lab edges merely nominate boundary locations.  Precise ridges and
    # long, coherent convex-oriented traces improve localization, but none of
    # these generic signals can by itself declare a physical boundary: coat
    # pattern edges can be equally strong.  Where annotation-trained semantics
    # exist, the physical-minus-nonphysical margin gates the *entire* candidate
    # cost.  This is deliberately unlike the former additive reference branch,
    # whose discount could not suppress the dominant generic contribution.
    boundary_weights = np.asarray(
        (
            settings.boundary_edge_weight,
            settings.boundary_ridge_weight,
            (
                settings.boundary_surface_darkening_weight
                if surface_darkening is not None
                else 0.0
            ),
        ),
        dtype=np.float32,
    )
    boundary_weights /= boundary_weights.sum()
    candidate_boundary = (
        boundary_weights[0] * edge + boundary_weights[1] * ridges
    )
    if surface_darkening is not None:
        candidate_boundary += boundary_weights[2] * surface_darkening
    candidate_boundary += (
        settings.boundary_physical_ridge_weight
        * reference_ridges
        * (1.0 - candidate_boundary)
    )
    trace_support = _oriented_trace_support(
        trace_labels,
        trace_continuity,
        diameter,
        settings,
    )
    candidate_boundary += (
        settings.boundary_trace_weight
        * trace_support
        * (1.0 - candidate_boundary)
    )
    candidate_boundary = cv2.GaussianBlur(
        candidate_boundary, (0, 0), sigmaX=max(0.35, diameter * 0.008)
    )
    candidate_scale = (
        float(np.quantile(candidate_boundary[valid], 0.995))
        if np.any(valid)
        else 1.0
    )
    candidate_boundary = np.clip(
        candidate_boundary / max(1e-5, candidate_scale), 0.0, 1.0
    )
    if prepared.semantic_edge_evidence_available:
        semantic_margin = (
            normalized_net_probability
            if np.any(normalized_net_probability > (1.0 / 255.0))
            else np.clip(
                physical_probability - nonphysical_probability, 0.0, 1.0
            )
        )
        classified_gate = (
            settings.boundary_semantic_floor
            + (1.0 - settings.boundary_semantic_floor) * semantic_margin
        )
        # A sparse set of annotated instances cannot classify every edge in
        # the dish.  Where both learned class memberships are near zero, keep
        # the generic candidate neutral instead of crushing it to the floor.
        # The classifier only gains authority in proportion to the stronger
        # of its physical/non-physical memberships.
        classification_strength = np.maximum(
            physical_probability, nonphysical_probability
        )
        classification_strength = np.maximum(
            classification_strength, normalized_net_probability
        )
        semantic_gate = (
            1.0 - classification_strength
            + classification_strength * classified_gate
        )
        nonedge_gate = np.clip(
            1.0
            - settings.boundary_nonphysical_discount
            * nonphysical_probability,
            0.0,
            1.0,
        )
        candidate_boundary *= semantic_gate * nonedge_gate
    boundary = cv2.GaussianBlur(
        candidate_boundary, (0, 0), sigmaX=max(0.35, diameter * 0.006)
    )
    boundary = np.clip(boundary, 0.0, 1.0) * valid

    # Centre depth is measured against the *redesigned* boundary field after
    # non-physical suppression and trace/convexity support.  Using only the
    # exterior of a packed material component produces one dish-sized plateau;
    # using the former raw edge/noise mixture produces a marker on every coat
    # stripe.  The semantic/geometric barrier gives one regional basin per
    # plausible seed while retaining the material exterior as a fallback.
    occupied_boundary = boundary[occupancy]
    strong_threshold = (
        float(np.quantile(occupied_boundary, 0.67))
        if occupied_boundary.size
        else 0.4
    )
    strong_threshold = float(np.clip(strong_threshold, 0.18, 0.62))
    traversable = occupancy & (boundary < strong_threshold)
    raw_boundary_depth = cv2.distanceTransform(
        np.uint8(traversable), cv2.DIST_L2, 5
    )
    exterior_depth = cv2.distanceTransform(np.uint8(occupancy), cv2.DIST_L2, 5)
    geometry_sigma = max(0.8, diameter * settings.centre_geometry_smoothing_fraction)
    boundary_depth = cv2.GaussianBlur(
        raw_boundary_depth, (0, 0), sigmaX=max(0.6, geometry_sigma * 0.32)
    )
    exterior_depth = cv2.GaussianBlur(
        exterior_depth, (0, 0), sigmaX=max(0.6, geometry_sigma * 0.40)
    )
    depth = np.maximum(boundary_depth, 0.18 * exterior_depth)
    distance_score = np.tanh(depth / max(1.0, diameter * 0.30))
    material_core = cv2.GaussianBlur(
        occupancy_likelihood * occupancy,
        (0, 0),
        sigmaX=geometry_sigma,
    )
    material_scale = (
        float(np.quantile(material_core[interior_valid], 0.995))
        if np.any(interior_valid)
        else 1.0
    )
    material_core = material_core / (material_core + max(1e-5, material_scale))
    if flattened_grayscale is None:
        flattened_core = np.zeros_like(material_core)
        flattened_weight = 0.0
    else:
        # The flattened image is centred on locally expected brightness.  A
        # seed-scale normalized convolution suppresses coat stripes while the
        # absolute residual retains either dark- or light-coated seed centres.
        occupancy_float = occupancy.astype(np.float32)
        normalization = cv2.GaussianBlur(
            occupancy_float, (0, 0), sigmaX=geometry_sigma
        )
        blurred_flattened = cv2.GaussianBlur(
            flattened_grayscale * occupancy_float,
            (0, 0),
            sigmaX=geometry_sigma,
        ) / np.maximum(normalization, 1e-4)
        flattened_core = np.clip(
            np.abs(blurred_flattened - 0.5) * 2.0, 0.0, 1.0
        )
        flattened_core *= np.clip(normalization, 0.0, 1.0)
        flattened_weight = settings.centre_flattened_grayscale_weight
    centre_weights = np.asarray(
        (
            settings.centre_material_weight,
            settings.centre_distance_weight,
            flattened_weight,
        ),
        dtype=np.float32,
    )
    centre_weights /= max(1e-6, float(centre_weights.sum()))
    centre = (
        centre_weights[0] * material_core
        + centre_weights[1] * distance_score
        + centre_weights[2] * flattened_core
    )
    centre *= cv2.GaussianBlur(
        occupancy.astype(np.float32), (0, 0), sigmaX=max(0.7, diameter * 0.05)
    )
    centre *= valid

    coverage = float(np.count_nonzero(occupancy)) / max(1, np.count_nonzero(valid))
    separation_fraction = (
        settings.sparse_centre_minimum_separation_fraction
        if coverage < 0.35
        else settings.centre_minimum_separation_fraction
    )
    nms_radius = diameter * separation_fraction
    local_maximum = cv2.dilate(centre, _ellipse_kernel(nms_radius))
    peak_mask = (centre >= local_maximum - 1e-6) & occupancy
    # Flat-topped maxima are common after bounded resizing. Treat each
    # connected plateau as one candidate rather than consuming the marker
    # budget with hundreds of adjacent, identical pixels.
    component_count, peak_components = cv2.connectedComponents(
        np.uint8(peak_mask), connectivity=8
    )
    peak_x_values: list[int] = []
    peak_y_values: list[int] = []
    peak_score_values: list[float] = []
    peak_rows, peak_columns = np.nonzero(peak_components)
    if len(peak_rows):
        component_ids = peak_components[peak_rows, peak_columns]
        order_by_component = np.argsort(component_ids, kind="stable")
        component_ids = component_ids[order_by_component]
        peak_rows = peak_rows[order_by_component]
        peak_columns = peak_columns[order_by_component]
        starts = np.flatnonzero(
            np.r_[True, component_ids[1:] != component_ids[:-1]]
        )
        stops = np.r_[starts[1:], len(component_ids)]
        for start, stop in zip(starts, stops, strict=True):
            rows = peak_rows[start:stop]
            columns = peak_columns[start:stop]
            component_scores = centre[rows, columns]
            best = int(np.argmax(component_scores))
            peak_x_values.append(int(columns[best]))
            peak_y_values.append(int(rows[best]))
            peak_score_values.append(float(component_scores[best]))
    peak_x = np.asarray(peak_x_values, np.int32)
    peak_y = np.asarray(peak_y_values, np.int32)
    peak_scores = np.asarray(peak_score_values, np.float32)
    retained = peak_scores >= settings.minimum_marker_score
    peak_x, peak_y, peak_scores = (
        peak_x[retained],
        peak_y[retained],
        peak_scores[retained],
    )
    order = np.argsort(peak_scores)[::-1]
    peak_x, peak_y, peak_scores = peak_x[order], peak_y[order], peak_scores[order]

    annotation_x = np.empty(0, np.int32)
    annotation_y = np.empty(0, np.int32)
    annotations = None
    annotation_ids = np.empty(0, np.uint16)
    if seed_instance_annotations is not None:
        annotations = prepared.annotation_targets(
            seed_instance_annotations,
            mask_to_valid=True,
        )
        annotation_ids = np.unique(annotations)
        annotation_ids = annotation_ids[annotation_ids > 0]
        annotation_centres: list[tuple[int, int]] = []
        for annotation_id in annotation_ids:
            rows, columns = np.nonzero(annotations == annotation_id)
            if len(rows):
                annotation_centres.append(
                    (int(round(float(columns.mean()))), int(round(float(rows.mean()))))
                )
        if annotation_centres:
            annotation_x = np.asarray(
                [item[0] for item in annotation_centres], np.int32
            )
            annotation_y = np.asarray(
                [item[1] for item in annotation_centres], np.int32
            )
            occupancy[annotations > 0] = True
            minimum_squared = (diameter * separation_fraction * 0.85) ** 2
            keep_automatic = np.ones(len(peak_x), dtype=bool)
            for x, y in zip(annotation_x, annotation_y, strict=True):
                keep_automatic &= (peak_x - x) ** 2 + (peak_y - y) ** 2 >= minimum_squared
            peak_x = peak_x[keep_automatic]
            peak_y = peak_y[keep_automatic]
            peak_scores = peak_scores[keep_automatic]

    (
        manual_x,
        manual_y,
        accepted_manual_source_xy,
        rejected_manual_xy,
        rejected_manual_reasons,
    ) = _manual_centres_at_working_scale(
        manual_seed_centres,
        prepared,
        occupancy,
        annotations,
        annotation_x,
        annotation_y,
        annotation_suppression_radius=(
            diameter * separation_fraction * 0.85
        ),
    )

    area_fraction = (
        settings.packed_seed_cell_fraction
        if coverage >= 0.55
        else settings.sparse_seed_area_fraction
    )
    expected_count = max(
        1,
        int(round(np.count_nonzero(occupancy) / max(1.0, area_fraction * diameter * diameter))),
    )
    maximum_markers = max(
        len(annotation_x),
        1,
        int(round(expected_count * settings.marker_count_multiplier)),
    )
    automatic_count = max(0, maximum_markers - len(annotation_x))
    automatic_x = peak_x[:automatic_count]
    automatic_y = peak_y[:automatic_count]
    automatic_scores = peak_scores[:automatic_count]
    replace_automatic = (
        manual_seed_centres is not None
        and manual_seed_centres.mode is ManualSeedCentreMode.REPLACE_AUTOMATIC
    )

    # In a dense dish, regional-max NMS alone can discard several adjacent
    # genuine centres whenever one maximum is only slightly stronger.  Use the
    # calibrated area estimate as a target (never an unbounded texture-driven
    # count) and fill its deficit with well-separated points from the smoothed
    # interior field.  Sparse scenes retain the stricter component maxima so
    # background material cannot be tiled with speculative seeds.
    if (
        not replace_automatic
        and coverage >= 0.55
        and len(annotation_x) + len(automatic_x) < maximum_markers
    ):
        marker_eligible = (
            traversable
            & (raw_boundary_depth >= max(1.0, diameter * 0.080))
            & (centre >= settings.minimum_marker_score)
        )
        if annotations is not None:
            marker_eligible &= annotations == 0
        supplemental_x, supplemental_y, supplemental_scores = (
            _supplement_packed_markers(
                centre,
                marker_eligible,
                seed_diameter=diameter,
                minimum_separation_fraction=separation_fraction,
                minimum_score=settings.minimum_marker_score,
                # A manual point is an additional authority, not a consumer
                # of the calibrated automatic count. Including it in both the
                # target and anchor set preserves that automatic population
                # while preventing a supplemental point from landing beside it.
                target_count=maximum_markers + len(manual_x),
                anchor_x=np.concatenate(
                    (annotation_x, manual_x, automatic_x)
                ),
                anchor_y=np.concatenate(
                    (annotation_y, manual_y, automatic_y)
                ),
            )
        )
        automatic_x = np.concatenate((automatic_x, supplemental_x))
        automatic_y = np.concatenate((automatic_y, supplemental_y))
        automatic_scores = np.concatenate((automatic_scores, supplemental_scores))

    if replace_automatic:
        automatic_x = np.empty(0, np.int32)
        automatic_y = np.empty(0, np.int32)
        automatic_scores = np.empty(0, np.float32)
    elif len(manual_x) and len(automatic_x):
        suppression_squared = (
            diameter * MANUAL_CENTRE_SUPPRESSION_DIAMETER_FRACTION
        ) ** 2
        keep_automatic = np.ones(len(automatic_x), dtype=bool)
        for x, y in zip(manual_x, manual_y, strict=True):
            keep_automatic &= (
                (automatic_x - x) ** 2 + (automatic_y - y) ** 2
                > suppression_squared
            )
        automatic_x = automatic_x[keep_automatic]
        automatic_y = automatic_y[keep_automatic]
        automatic_scores = automatic_scores[keep_automatic]

    peak_x = np.concatenate((annotation_x, manual_x, automatic_x))
    peak_y = np.concatenate((annotation_y, manual_y, automatic_y))
    peak_scores = np.concatenate(
        (
            np.ones(len(annotation_x) + len(manual_x), np.float32),
            automatic_scores,
        )
    )
    peak_sources = np.concatenate(
        (
            np.full(
                len(annotation_x),
                int(ProceduralMarkerSource.ANNOTATED),
                np.uint8,
            ),
            np.full(
                len(manual_x), int(ProceduralMarkerSource.MANUAL), np.uint8
            ),
            np.full(
                len(automatic_x),
                int(ProceduralMarkerSource.AUTOMATIC),
                np.uint8,
            ),
        )
    )

    topography = np.uint8(np.clip(np.rint(boundary * 255.0), 0, 255))
    annotation_lookup = None
    if annotations is not None and len(annotation_ids):
        annotation_lookup = np.zeros(int(annotation_ids[-1]) + 1, np.int32)
        annotation_lookup[annotation_ids] = np.arange(
            2, len(annotation_ids) + 2, dtype=np.int32
        )

    hypothesis_count = int(settings.candidate_hypotheses_per_marker)
    threshold_scales = np.linspace(0.65, 1.35, hypothesis_count, dtype=np.float32)
    # An odd default naturally contains 1.0; explicitly replace the nearest
    # entry so every configured count retains the exact current material mask.
    threshold_scales[int(np.argmin(np.abs(threshold_scales - 1.0)))] = 1.0
    candidates: list[_InstanceCandidate] = []
    candidates_by_marker: list[list[_InstanceCandidate]] = [
        [] for _ in range(len(peak_x))
    ]
    rejection_reasons = ["instance_below_minimum_area"] * len(peak_x)
    marker_radius = max(1, int(round(diameter * 0.035)))
    for hypothesis_index, threshold_scale in enumerate(threshold_scales):
        candidate_occupancy = _material_occupancy(
            occupancy_u8,
            threshold * float(threshold_scale),
            interior_valid,
            diameter,
            settings,
        )
        if annotations is not None:
            candidate_occupancy[annotations > 0] = True
        markers = np.zeros((height, width), dtype=np.int32)
        markers[~candidate_occupancy] = 1
        for identifier, (x, y) in enumerate(
            zip(peak_x, peak_y, strict=True), start=2
        ):
            cv2.circle(
                markers,
                (int(x), int(y)),
                marker_radius,
                identifier,
                -1,
            )
        # Reviewed masks remain authoritative marker regions under every
        # hypothesis, so a nearby proposal cannot punch through them.
        if annotation_lookup is not None and annotations is not None:
            selected_annotations = annotations > 0
            markers[selected_annotations] = annotation_lookup[
                annotations[selected_annotations]
            ]
        watershed = cv2.watershed(
            cv2.cvtColor(topography, cv2.COLOR_GRAY2BGR), markers
        )
        hypothesis_labels = np.where(
            (watershed >= 2) & candidate_occupancy,
            watershed - 1,
            0,
        ).astype(np.int32)
        nonzero_y, nonzero_x = np.nonzero(hypothesis_labels)
        nonzero_ids = hypothesis_labels[nonzero_y, nonzero_x]
        bbox_min_x = np.full(len(peak_x) + 1, width, np.int32)
        bbox_min_y = np.full(len(peak_x) + 1, height, np.int32)
        bbox_max_x = np.full(len(peak_x) + 1, -1, np.int32)
        bbox_max_y = np.full(len(peak_x) + 1, -1, np.int32)
        if len(nonzero_ids):
            np.minimum.at(bbox_min_x, nonzero_ids, nonzero_x)
            np.minimum.at(bbox_min_y, nonzero_ids, nonzero_y)
            np.maximum.at(bbox_max_x, nonzero_ids, nonzero_x)
            np.maximum.at(bbox_max_y, nonzero_ids, nonzero_y)
        for marker_index in range(len(peak_x)):
            label_id = marker_index + 1
            if bbox_max_x[label_id] < bbox_min_x[label_id]:
                rejection_reasons[marker_index] = "instance_below_minimum_area"
                continue
            component_x0 = int(bbox_min_x[label_id])
            component_y0 = int(bbox_min_y[label_id])
            component_x1 = int(bbox_max_x[label_id]) + 1
            component_y1 = int(bbox_max_y[label_id]) + 1
            component_mask = (
                hypothesis_labels[
                    component_y0:component_y1,
                    component_x0:component_x1,
                ]
                == label_id
            )
            if (
                annotations is not None
                and marker_index < len(annotation_ids)
            ):
                reviewed_mask = (
                    annotations[
                        component_y0:component_y1,
                        component_x0:component_x1,
                    ]
                    == int(annotation_ids[marker_index])
                )
                # A full reviewed instance is authoritative geometry, not
                # merely a centre from which watershed may absorb neighbours.
                # Very small legacy marker blobs are still treated as centres
                # and allowed to grow, preserving that older workflow.
                if np.count_nonzero(reviewed_mask) >= (
                    diameter
                    * diameter
                    * settings.minimum_instance_area_fraction
                ):
                    component_mask = reviewed_mask
            candidate, rejection = _candidate_from_component(
                component_mask,
                marker_index=marker_index,
                hypothesis_index=hypothesis_index,
                marker_score=float(peak_scores[marker_index]),
                marker_source=int(peak_sources[marker_index]),
                boundary=boundary,
                diameter=diameter,
                expected_area_fraction=area_fraction,
                settings=settings,
                origin_xy=(component_x0, component_y0),
            )
            if candidate is None:
                rejection_reasons[marker_index] = rejection
                continue
            duplicate = False
            for existing in candidates_by_marker[marker_index]:
                overlap = _candidate_overlap_pixels(candidate, existing)
                union = candidate.area + existing.area - overlap
                if overlap / max(1.0, union) >= 0.985:
                    duplicate = True
                    if candidate.score > existing.score:
                        existing.score = candidate.score
                    break
            if duplicate:
                continue
            candidates.append(candidate)
            candidates_by_marker[marker_index].append(candidate)

    # Approximate maximum-weight independent-set selection. Annotation-derived
    # candidates are fixed first; remaining masks are ranked by their complete
    # evidence/geometry score and may not substantially overlap an already
    # retained mask or provide a second shape for the same marker.
    selected_candidates: list[_InstanceCandidate] = []
    selected_markers: set[int] = set()
    ordered_candidates = sorted(
        candidates,
        key=lambda candidate: (
            peak_sources[candidate.marker_index]
            == int(ProceduralMarkerSource.ANNOTATED),
            candidate.score,
            candidate.area,
        ),
        reverse=True,
    )
    for candidate in ordered_candidates:
        if candidate.marker_index in selected_markers:
            continue
        conflicts = False
        for selected_candidate in selected_candidates:
            overlap = _candidate_overlap_pixels(candidate, selected_candidate)
            if overlap > (
                min(candidate.area, selected_candidate.area)
                * settings.candidate_overlap_fraction
            ):
                conflicts = True
                break
        if conflicts:
            continue
        candidate.selected = True
        selected_candidates.append(candidate)
        selected_markers.add(candidate.marker_index)

    # Two bounded local-improvement passes repair the common greedy failure in
    # which one early mediocre mask blocks a better later combination (or vice
    # versa). This is a deterministic weighted set-packing heuristic rather
    # than an exponential exact solver over thousands of masks.
    for _pass in range(2):
        improved = False
        for candidate in ordered_candidates:
            if candidate.selected:
                continue
            conflicts = [
                selected_candidate
                for selected_candidate in selected_candidates
                if selected_candidate.marker_index == candidate.marker_index
                or _candidate_overlap_pixels(candidate, selected_candidate)
                > (
                    min(candidate.area, selected_candidate.area)
                    * settings.candidate_overlap_fraction
                )
            ]
            if not conflicts:
                continue
            if any(
                peak_sources[item.marker_index]
                == int(ProceduralMarkerSource.ANNOTATED)
                for item in conflicts
            ):
                continue
            if candidate.score <= sum(item.score for item in conflicts) + 1e-6:
                continue
            for item in conflicts:
                item.selected = False
                selected_candidates.remove(item)
                selected_markers.discard(item.marker_index)
            candidate.selected = True
            selected_candidates.append(candidate)
            selected_markers.add(candidate.marker_index)
            improved = True
        if not improved:
            break
    for candidate in ordered_candidates:
        if candidate.selected or candidate.marker_index in selected_markers:
            continue
        if any(
            _candidate_overlap_pixels(candidate, item)
            > min(candidate.area, item.area) * settings.candidate_overlap_fraction
            for item in selected_candidates
        ):
            continue
        candidate.selected = True
        selected_candidates.append(candidate)
        selected_markers.add(candidate.marker_index)

    selected_candidates.sort(key=lambda candidate: candidate.marker_index)
    labels = np.zeros((height, width), np.int32)
    concavity_raster = np.zeros((height, width), np.uint8)
    surviving_markers = np.zeros(len(peak_x), dtype=bool)
    assigned_candidates: list[_InstanceCandidate] = []
    for label_id, candidate in enumerate(selected_candidates, start=1):
        label_roi = labels[candidate.y0 : candidate.y1, candidate.x0 : candidate.x1]
        assign = candidate.mask & (label_roi == 0)
        if not np.any(assign):
            candidate.selected = False
            continue
        label_roi[assign] = label_id
        concavity_value = np.uint8(
            np.clip(
                np.rint(
                    candidate.concavity_fraction
                    / max(1e-6, settings.maximum_internal_concavity_fraction)
                    * 255.0
                ),
                0,
                255,
            )
        )
        concavity_roi = concavity_raster[
            candidate.y0 : candidate.y1, candidate.x0 : candidate.x1
        ]
        concavity_roi[candidate.concavity_mask] = np.maximum(
            concavity_roi[candidate.concavity_mask], concavity_value
        )
        surviving_markers[candidate.marker_index] = True
        assigned_candidates.append(candidate)
    labels = _renumber_labels(labels)

    manual_start = len(annotation_x)
    manual_stop = manual_start + len(manual_x)
    culled_manual = ~surviving_markers[manual_start:manual_stop]
    if np.any(culled_manual):
        rejected_manual_xy = np.concatenate(
            (rejected_manual_xy, accepted_manual_source_xy[culled_manual]),
            axis=0,
        )
        manual_reasons = tuple(
            rejection_reasons[index]
            for index in range(manual_start, manual_stop)
            if culled_manual[index - manual_start]
        )
        rejected_manual_reasons = (*rejected_manual_reasons, *manual_reasons)
    marker_centres = np.column_stack(
        (
            peak_x[surviving_markers] / scale,
            peak_y[surviving_markers] / scale,
        )
    ).astype(np.float32)
    marker_sources = np.asarray(
        peak_sources[surviving_markers], dtype=np.uint8
    )

    label_count = int(labels.max())
    if label_count:
        flat_labels = labels.reshape(-1)
        area_values = np.bincount(flat_labels, minlength=label_count + 1).astype(np.float32)
        yy, xx = np.indices(labels.shape, dtype=np.float32)
        sum_x = np.bincount(flat_labels, weights=xx.reshape(-1), minlength=label_count + 1)
        sum_y = np.bincount(flat_labels, weights=yy.reshape(-1), minlength=label_count + 1)
        centre_x = sum_x[1:] / np.maximum(area_values[1:], 1.0)
        centre_y = sum_y[1:] / np.maximum(area_values[1:], 1.0)

        marker_scores = np.asarray(
            [peak_scores[item.marker_index] for item in assigned_candidates],
            np.float32,
        )
        confidences = np.asarray(
            [item.score for item in assigned_candidates], np.float32
        )
        centres = np.column_stack((centre_x / scale, centre_y / scale)).astype(np.float32)
    else:
        centres = np.empty((0, 2), np.float32)
        marker_scores = np.empty(0, np.float32)
        confidences = np.empty(0, np.float32)

    alternative_rgba = np.zeros((height, width, 4), np.uint8)
    for candidate in sorted(candidates, key=lambda item: item.score):
        if candidate.selected:
            continue
        score = float(np.clip(candidate.score, 0.0, 1.0))
        colour = _alternative_candidate_colour(score)
        roi = alternative_rgba[
            candidate.y0 : candidate.y1, candidate.x0 : candidate.x1
        ]
        roi[candidate.mask, :3] = colour
        roi[candidate.mask, 3] = np.maximum(
            roi[candidate.mask, 3], np.uint8(round(20 + score * 45))
        )
        contours, _hierarchy = cv2.findContours(
            np.uint8(candidate.mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(
            roi,
            contours,
            -1,
            (*[int(value) for value in colour], round(130 + score * 125)),
            1,
        )

    maximum_label = int(np.max(labels, initial=0))
    label_dtype = np.uint16 if maximum_label <= np.iinfo(np.uint16).max else np.uint32
    return ProceduralInstanceResult(
        labels=np.asarray(labels, dtype=label_dtype),
        centres_xy=np.asarray(centres, dtype=np.float32).reshape(-1, 2),
        marker_scores=np.asarray(marker_scores, dtype=np.float32),
        instance_confidences=np.asarray(confidences, dtype=np.float32),
        occupancy_likelihood=np.uint8(np.clip(np.rint(occupancy_likelihood * 255.0), 0, 255)),
        occupancy_mask=np.uint8(occupancy) * 255,
        boundary_cost=np.uint8(np.clip(np.rint(boundary * 255.0), 0, 255)),
        centre_likelihood=np.uint8(np.clip(np.rint(centre * 255.0), 0, 255)),
        source_shape=(source_height, source_width),
        working_scale=float(scale),
        marker_centres_xy=np.asarray(marker_centres, np.float32).reshape(-1, 2),
        marker_sources=marker_sources,
        rejected_manual_centres_xy=np.asarray(
            rejected_manual_xy, np.float32
        ).reshape(-1, 2),
        rejected_manual_centre_reasons=tuple(rejected_manual_reasons),
        concavity=concavity_raster,
        alternative_candidates_rgba=alternative_rgba,
        candidate_scores=np.asarray(
            [candidate.score for candidate in candidates], np.float32
        ),
        candidate_selected=np.asarray(
            [candidate.selected for candidate in candidates], bool
        ),
        instance_area_px2=np.asarray(
            [candidate.area / max(scale * scale, 1e-12) for candidate in assigned_candidates],
            np.float32,
        ),
        instance_maximum_width_px=np.asarray(
            [candidate.width_fraction * diameter / max(scale, 1e-12) for candidate in assigned_candidates],
            np.float32,
        ),
        instance_width_fractions=np.asarray(
            [candidate.width_fraction for candidate in assigned_candidates], np.float32
        ),
        instance_concavity_fractions=np.asarray(
            [candidate.concavity_fraction for candidate in assigned_candidates], np.float32
        ),
        instance_protrusion_fractions=np.asarray(
            [candidate.protrusion_fraction for candidate in assigned_candidates], np.float32
        ),
        instance_solidities=np.asarray(
            [candidate.solidity for candidate in assigned_candidates], np.float32
        ),
        instance_axis_ratios=np.asarray(
            [candidate.axis_ratio for candidate in assigned_candidates], np.float32
        ),
    )


def procedural_seed_instances(
    valid_mask,
    seed_diameter_px: float,
    *,
    material_probability=None,
    foreground_probability,
    foreground_noise_probability,
    background_probability,
    refined_background_probability,
    edge_magnitude,
    edge_ridges,
    physical_edge_probability=None,
    non_edge_probability=None,
    normalized_net_physical_edge_probability=None,
    thinned_reference_edge_ridges=None,
    oriented_edge_trace_labels=None,
    oriented_edge_trace_continuity=None,
    reference_surface_probability=None,
    flattened_grayscale=None,
    surface_darkening_magnitude=None,
    seed_instance_annotations=None,
    manual_seed_centres: ManualSeedCentres | None = None,
    settings: ProceduralInstanceSettings = ProceduralInstanceSettings(),
) -> ProceduralInstanceResult:
    """Separate visible seeds, preparing one bounded copy of every input."""

    prepared = prepare_procedural_instance_inputs(
        valid_mask,
        seed_diameter_px,
        material_probability=material_probability,
        foreground_probability=foreground_probability,
        foreground_noise_probability=foreground_noise_probability,
        background_probability=background_probability,
        refined_background_probability=refined_background_probability,
        edge_magnitude=edge_magnitude,
        edge_ridges=edge_ridges,
        physical_edge_probability=physical_edge_probability,
        non_edge_probability=non_edge_probability,
        normalized_net_physical_edge_probability=(
            normalized_net_physical_edge_probability
        ),
        thinned_reference_edge_ridges=thinned_reference_edge_ridges,
        oriented_edge_trace_labels=oriented_edge_trace_labels,
        oriented_edge_trace_continuity=oriented_edge_trace_continuity,
        reference_surface_probability=reference_surface_probability,
        flattened_grayscale=flattened_grayscale,
        surface_darkening_magnitude=surface_darkening_magnitude,
        working_maximum_dimension=settings.working_maximum_dimension,
    )
    return procedural_seed_instances_from_prepared(
        prepared,
        seed_instance_annotations=seed_instance_annotations,
        manual_seed_centres=manual_seed_centres,
        settings=settings,
    )
