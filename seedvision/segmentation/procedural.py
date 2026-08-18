"""Procedural seed-instance separation from calibrated diagnostic rasters.

The algorithm deliberately separates *evidence construction* from the final
topological partition.  Seed-scale CUDA rasters can be supplied lazily; only a
bounded working copy is materialized for OpenCV's marker-controlled watershed,
which is the one topology operation not currently available in the tensor
pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


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
    trace_minimum_length_fraction: float = 0.22
    trace_convexity_weight: float = 0.60
    reference_texture_weight: float = 0.35
    centre_geometry_smoothing_fraction: float = 0.14
    centre_material_weight: float = 0.55
    centre_distance_weight: float = 0.45
    centre_minimum_separation_fraction: float = 0.42
    sparse_centre_minimum_separation_fraction: float = 0.58
    sparse_seed_area_fraction: float = 0.47
    packed_seed_cell_fraction: float = 0.72
    marker_count_multiplier: float = 1.02
    minimum_marker_score: float = 0.12
    minimum_instance_area_fraction: float = 0.18
    maximum_instance_area_fraction: float = 1.45

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
        if not 0.02 <= self.trace_minimum_length_fraction <= 2.0:
            raise ValueError("Minimum trace length must be between 0.02 and 2 seed diameters.")
        if not 0.0 <= self.trace_convexity_weight <= 1.0:
            raise ValueError("Trace convexity weight must be between zero and one.")
        if not 0.0 <= self.reference_texture_weight <= 1.0:
            raise ValueError("Reference texture weight must be between zero and one.")
        centre_weights = (self.centre_material_weight, self.centre_distance_weight)
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
        if not 0.5 <= self.marker_count_multiplier <= 2.0:
            raise ValueError("Marker-count multiplier must be between 0.5 and 2.0.")
        if not 0.05 <= self.minimum_instance_area_fraction < self.maximum_instance_area_fraction:
            raise ValueError("Instance area fractions are inconsistent.")


@dataclass(frozen=True, slots=True)
class ProceduralInstanceResult:
    """Instance labels and the diagnostic evidence used to construct them."""

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

    @property
    def count(self) -> int:
        return int(len(self.centres_xy))

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
    foreground_probability: np.ndarray
    foreground_noise_probability: np.ndarray
    background_probability: np.ndarray
    refined_background_probability: np.ndarray
    edge_magnitude: np.ndarray
    edge_ridges: np.ndarray
    physical_edge_probability: np.ndarray
    non_edge_probability: np.ndarray
    thinned_reference_edge_ridges: np.ndarray
    oriented_edge_trace_labels: np.ndarray
    oriented_edge_trace_continuity: np.ndarray
    semantic_edge_evidence_available: bool
    reference_surface_probability: np.ndarray | None

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
    foreground_probability,
    foreground_noise_probability,
    background_probability,
    refined_background_probability,
    edge_magnitude,
    edge_ridges,
    physical_edge_probability=None,
    non_edge_probability=None,
    thinned_reference_edge_ridges=None,
    oriented_edge_trace_labels=None,
    oriented_edge_trace_continuity=None,
    reference_surface_probability=None,
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
    )
    reference_surface = (
        None
        if reference_surface_probability is None
        else _readonly(
            _working_u8(reference_surface_probability, size, cv2.INTER_AREA) / 255.0
        )
    )
    return PreparedProceduralInstanceInputs(
        source_shape=(source_height, source_width),
        working_scale=float(scale),
        working_maximum_dimension=int(working_maximum_dimension),
        seed_diameter_px=max(6.0, float(seed_diameter_px) * scale),
        valid_mask=valid,
        foreground_probability=foreground,
        foreground_noise_probability=foreground_noise,
        background_probability=background,
        refined_background_probability=refined_background,
        edge_magnitude=edge,
        edge_ridges=ridges,
        physical_edge_probability=physical_probability,
        non_edge_probability=nonphysical_probability,
        thinned_reference_edge_ridges=reference_ridges,
        oriented_edge_trace_labels=trace_labels,
        oriented_edge_trace_continuity=trace_continuity,
        semantic_edge_evidence_available=semantic_available,
        reference_surface_probability=reference_surface,
    )


def procedural_seed_instances_from_prepared(
    prepared: PreparedProceduralInstanceInputs,
    *,
    seed_instance_annotations=None,
    settings: ProceduralInstanceSettings = ProceduralInstanceSettings(),
) -> ProceduralInstanceResult:
    """Run only parameter-dependent CPU topology on a prepared working set."""

    if settings.working_maximum_dimension != prepared.working_maximum_dimension:
        raise ValueError(
            "Prepared inputs use a different working maximum dimension; "
            "prepare them again for these settings."
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
    reference_ridges = prepared.thinned_reference_edge_ridges
    trace_labels = prepared.oriented_edge_trace_labels
    trace_continuity = prepared.oriented_edge_trace_continuity
    reference_surface = prepared.reference_surface_probability

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
    # Suppress the bright dish rim before closing.  Otherwise a nearly closed
    # rim can be mistaken for one enormous foreground object.
    occupancy = (occupancy_u8 >= threshold) & interior_valid
    close_kernel = _ellipse_kernel(diameter * settings.occupancy_closing_fraction)
    occupancy = cv2.morphologyEx(
        np.uint8(occupancy) * 255,
        cv2.MORPH_CLOSE,
        close_kernel,
    ) > 0
    occupancy &= interior_valid
    # Colour/noise likelihood often responds to a seed's patterned perimeter
    # while assigning its pale or dark centre a low score.  Fill only enclosed,
    # seed-sized holes; open background and large gaps remain background.
    flood = np.uint8(occupancy) * 255
    cv2.floodFill(flood, None, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    component_count, component_labels, component_stats, _ = cv2.connectedComponentsWithStats(
        np.uint8(holes > 0), connectivity=8
    )
    maximum_hole_area = diameter * diameter * settings.occupancy_hole_area_fraction
    fill_component = component_stats[:, cv2.CC_STAT_AREA] <= maximum_hole_area
    fill_component[0] = False
    occupancy |= fill_component[component_labels]
    occupancy = cv2.morphologyEx(
        np.uint8(occupancy) * 255,
        cv2.MORPH_OPEN,
        _ellipse_kernel(max(1.0, diameter * 0.025)),
    ) > 0

    # Broad Lab edges merely nominate boundary locations.  Precise ridges and
    # long, coherent convex-oriented traces improve localization, but none of
    # these generic signals can by itself declare a physical boundary: coat
    # pattern edges can be equally strong.  Where annotation-trained semantics
    # exist, the physical-minus-nonphysical margin gates the *entire* candidate
    # cost.  This is deliberately unlike the former additive reference branch,
    # whose discount could not suppress the dominant generic contribution.
    boundary_weights = np.asarray(
        (settings.boundary_edge_weight, settings.boundary_ridge_weight),
        dtype=np.float32,
    )
    boundary_weights /= boundary_weights.sum()
    candidate_boundary = (
        boundary_weights[0] * edge + boundary_weights[1] * ridges
    )
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
        semantic_margin = np.clip(
            physical_probability - nonphysical_probability, 0.0, 1.0
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
    centre_weights = np.asarray(
        (settings.centre_material_weight, settings.centre_distance_weight),
        dtype=np.float32,
    )
    centre_weights /= max(1e-6, float(centre_weights.sum()))
    centre = (
        centre_weights[0] * material_core
        + centre_weights[1] * distance_score
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

    manual_x = np.empty(0, np.int32)
    manual_y = np.empty(0, np.int32)
    annotations = None
    annotation_ids = np.empty(0, np.uint16)
    if seed_instance_annotations is not None:
        annotations = prepared.annotation_targets(
            seed_instance_annotations,
            mask_to_valid=True,
        )
        annotation_ids = np.unique(annotations)
        annotation_ids = annotation_ids[annotation_ids > 0]
        manual_centres: list[tuple[int, int]] = []
        for annotation_id in annotation_ids:
            rows, columns = np.nonzero(annotations == annotation_id)
            if len(rows):
                manual_centres.append(
                    (int(round(float(columns.mean()))), int(round(float(rows.mean()))))
                )
        if manual_centres:
            manual_x = np.asarray([item[0] for item in manual_centres], np.int32)
            manual_y = np.asarray([item[1] for item in manual_centres], np.int32)
            occupancy[annotations > 0] = True
            minimum_squared = (diameter * separation_fraction * 0.85) ** 2
            keep_automatic = np.ones(len(peak_x), dtype=bool)
            for x, y in zip(manual_x, manual_y, strict=True):
                keep_automatic &= (peak_x - x) ** 2 + (peak_y - y) ** 2 >= minimum_squared
            peak_x = peak_x[keep_automatic]
            peak_y = peak_y[keep_automatic]
            peak_scores = peak_scores[keep_automatic]

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
        len(manual_x),
        1,
        int(round(expected_count * settings.marker_count_multiplier)),
    )
    automatic_count = max(0, maximum_markers - len(manual_x))
    automatic_x = peak_x[:automatic_count]
    automatic_y = peak_y[:automatic_count]
    automatic_scores = peak_scores[:automatic_count]

    # In a dense dish, regional-max NMS alone can discard several adjacent
    # genuine centres whenever one maximum is only slightly stronger.  Use the
    # calibrated area estimate as a target (never an unbounded texture-driven
    # count) and fill its deficit with well-separated points from the smoothed
    # interior field.  Sparse scenes retain the stricter component maxima so
    # background material cannot be tiled with speculative seeds.
    if coverage >= 0.55 and len(manual_x) + len(automatic_x) < maximum_markers:
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
                target_count=maximum_markers,
                anchor_x=np.concatenate((manual_x, automatic_x)),
                anchor_y=np.concatenate((manual_y, automatic_y)),
            )
        )
        automatic_x = np.concatenate((automatic_x, supplemental_x))
        automatic_y = np.concatenate((automatic_y, supplemental_y))
        automatic_scores = np.concatenate((automatic_scores, supplemental_scores))

    peak_x = np.concatenate((manual_x, automatic_x))
    peak_y = np.concatenate((manual_y, automatic_y))
    peak_scores = np.concatenate(
        (np.ones(len(manual_x), np.float32), automatic_scores)
    )

    markers = np.zeros((height, width), dtype=np.int32)
    markers[~occupancy] = 1
    for identifier, (x, y) in enumerate(zip(peak_x, peak_y, strict=True), start=2):
        cv2.circle(markers, (int(x), int(y)), max(1, int(round(diameter * 0.035))), identifier, -1)
    # Whole annotated masks are authoritative, not just centre hints.  Apply
    # them after automatic marker discs so a nearby proposal can never punch a
    # different identifier through a reviewed seed.
    if annotations is not None and len(annotation_ids):
        annotation_lookup = np.zeros(int(annotation_ids[-1]) + 1, np.int32)
        annotation_lookup[annotation_ids] = np.arange(
            2, len(annotation_ids) + 2, dtype=np.int32
        )
        selected_annotations = annotations > 0
        markers[selected_annotations] = annotation_lookup[annotations[selected_annotations]]
    topography = np.uint8(np.clip(np.rint(boundary * 255.0), 0, 255))
    watershed = cv2.watershed(cv2.cvtColor(topography, cv2.COLOR_GRAY2BGR), markers)
    labels = np.where((watershed >= 2) & occupancy, watershed - 1, 0).astype(np.int32)

    minimum_area = diameter * diameter * settings.minimum_instance_area_fraction
    maximum_area = diameter * diameter * settings.maximum_instance_area_fraction
    if labels.max() > 0:
        areas = np.bincount(labels.reshape(-1))
        retained_labels = areas >= minimum_area
        retained_labels[0] = False
        labels = np.where(retained_labels[labels], labels, 0).astype(np.int32)
    labels = _renumber_labels(labels)

    label_count = int(labels.max())
    if label_count:
        flat_labels = labels.reshape(-1)
        area_values = np.bincount(flat_labels, minlength=label_count + 1).astype(np.float32)
        yy, xx = np.indices(labels.shape, dtype=np.float32)
        sum_x = np.bincount(flat_labels, weights=xx.reshape(-1), minlength=label_count + 1)
        sum_y = np.bincount(flat_labels, weights=yy.reshape(-1), minlength=label_count + 1)
        centre_x = sum_x[1:] / np.maximum(area_values[1:], 1.0)
        centre_y = sum_y[1:] / np.maximum(area_values[1:], 1.0)

        perimeter = np.zeros(labels.shape, dtype=bool)
        difference = labels[:, 1:] != labels[:, :-1]
        perimeter[:, 1:] |= difference
        perimeter[:, :-1] |= difference
        difference = labels[1:, :] != labels[:-1, :]
        perimeter[1:, :] |= difference
        perimeter[:-1, :] |= difference
        perimeter &= labels > 0
        perimeter_labels = labels[perimeter]
        boundary_totals = np.bincount(
            perimeter_labels,
            weights=boundary[perimeter],
            minlength=label_count + 1,
        )
        boundary_counts = np.bincount(perimeter_labels, minlength=label_count + 1)
        boundary_support = boundary_totals[1:] / np.maximum(boundary_counts[1:], 1)

        area_ratio = area_values[1:] / max(1.0, diameter * diameter * area_fraction)
        area_confidence = np.exp(
            -0.5 * (np.log(np.maximum(area_ratio, 1e-4)) / 0.58) ** 2
        )
        if len(peak_x):
            squared_distances = (
                (centre_x[:, None] - peak_x[None, :]) ** 2
                + (centre_y[:, None] - peak_y[None, :]) ** 2
            )
            marker_scores = peak_scores[np.argmin(squared_distances, axis=1)]
        else:
            marker_scores = np.zeros(label_count, np.float32)
        oversized_penalty = np.minimum(1.0, maximum_area / np.maximum(area_values[1:], 1.0))
        confidences = np.clip(
            (
                0.42 * marker_scores
                + 0.36 * boundary_support
                + 0.22 * area_confidence
            )
            * oversized_penalty,
            0.0,
            1.0,
        ).astype(np.float32)
        centres = np.column_stack((centre_x / scale, centre_y / scale)).astype(np.float32)
    else:
        centres = np.empty((0, 2), np.float32)
        marker_scores = np.empty(0, np.float32)
        confidences = np.empty(0, np.float32)

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
    )


def procedural_seed_instances(
    valid_mask,
    seed_diameter_px: float,
    *,
    foreground_probability,
    foreground_noise_probability,
    background_probability,
    refined_background_probability,
    edge_magnitude,
    edge_ridges,
    physical_edge_probability=None,
    non_edge_probability=None,
    thinned_reference_edge_ridges=None,
    oriented_edge_trace_labels=None,
    oriented_edge_trace_continuity=None,
    reference_surface_probability=None,
    seed_instance_annotations=None,
    settings: ProceduralInstanceSettings = ProceduralInstanceSettings(),
) -> ProceduralInstanceResult:
    """Separate visible seeds, preparing one bounded copy of every input."""

    prepared = prepare_procedural_instance_inputs(
        valid_mask,
        seed_diameter_px,
        foreground_probability=foreground_probability,
        foreground_noise_probability=foreground_noise_probability,
        background_probability=background_probability,
        refined_background_probability=refined_background_probability,
        edge_magnitude=edge_magnitude,
        edge_ridges=edge_ridges,
        physical_edge_probability=physical_edge_probability,
        non_edge_probability=non_edge_probability,
        thinned_reference_edge_ridges=thinned_reference_edge_ridges,
        oriented_edge_trace_labels=oriented_edge_trace_labels,
        oriented_edge_trace_continuity=oriented_edge_trace_continuity,
        reference_surface_probability=reference_surface_probability,
        working_maximum_dimension=settings.working_maximum_dimension,
    )
    return procedural_seed_instances_from_prepared(
        prepared,
        seed_instance_annotations=seed_instance_annotations,
        settings=settings,
    )
