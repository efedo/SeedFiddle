"""Shape-guided, edge-refined seed-instance filling.

The approximate ellipse found here is deliberately only a geometric prior.  A
successful result always checks a separately optimized closed contour through
the supplied edge raster. The ellipse itself is only an approximate maximum:
canonical Smart-fill growth is unpenalized deep inside, gradually penalized
from a configurable distance inside the boundary, and stopped at a small hard
outward cutoff. Insufficient or spatially unrelated evidence returns an
explicit refusal instead of silently painting the nominal ellipse.

All expensive work is bounded by small candidate/sample banks and one local
region around the clicked seed.  The module has no Qt dependency so live-view
code can calculate a preview once and commit that exact preview on click.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, pi

import cv2
import numpy as np

from .tools import SmartFillOptions, smart_fill_region


# Eight-connected foreground is the standard digital-topology choice for a
# diagonally sampled biological object: it avoids artificial one-pixel cracks
# without adding a user control that would change annotation semantics.
SHAPE_FILL_CONNECTIVITY = 8


@dataclass(frozen=True, slots=True)
class ShapeGuidedFillOptions:
    """Controls for fitting a rotated prior and refining it to real edges."""

    shape: str = "ellipse"
    preferred_scale: float = 1.0
    auto_rotation: bool = True
    initial_rotation_degrees: float = 0.0
    rotation_step_degrees: float = 15.0
    centre_search_fraction: float = 0.10
    minimum_scale: float = 0.75
    maximum_scale: float = 1.30
    maximum_axis_ratio: float = 2.30
    coarse_angular_samples: int = 64
    boundary_angular_samples: int = 128
    penalty_start_inside_fraction: float = 0.05
    outward_penalty_half_life_fraction: float = 0.05
    outward_hard_cutoff_fraction: float = 0.10
    boundary_smoothness: float = 0.45
    lab_contrast_weight: float = 0.12
    colour_tolerance_lab: float = 18.0
    edge_barrier_threshold: float = 0.58
    edge_gap_sealing_fraction: float = 0.02
    minimum_boundary_strength: float = 0.25
    minimum_edge_coverage: float = 0.34
    minimum_sector_coverage: float = 0.62
    maximum_unsupported_arc_fraction: float = 0.20
    maximum_added_pixels: int = 150_000
    maximum_roi_size_px: int = 768
    maximum_other_label_overlap: float = 0.12

    def __post_init__(self) -> None:
        if self.shape not in {"circle", "ellipse"}:
            raise ValueError("Shape must be circle or ellipse.")
        if not 0.40 <= self.preferred_scale <= 2.0:
            raise ValueError("Preferred shape scale must be between 0.4 and 2.")
        if not -180.0 <= self.initial_rotation_degrees <= 180.0:
            raise ValueError("Initial rotation must be between -180 and 180 degrees.")
        if not 3.0 <= self.rotation_step_degrees <= 90.0:
            raise ValueError("Rotation step must be between 3 and 90 degrees.")
        if not 0.0 <= self.centre_search_fraction <= 0.50:
            raise ValueError("Centre search fraction must be between zero and 0.5.")
        if not 0.40 <= self.minimum_scale <= 1.50:
            raise ValueError("Minimum scale must be between 0.4 and 1.5.")
        if not self.minimum_scale <= self.maximum_scale <= 2.0:
            raise ValueError("Maximum scale must be at least the minimum and at most 2.")
        if not 1.0 <= self.maximum_axis_ratio <= 4.0:
            raise ValueError("Maximum axis ratio must be between one and four.")
        if not 32 <= self.coarse_angular_samples <= 128:
            raise ValueError("Coarse angular samples must be between 32 and 128.")
        if self.coarse_angular_samples % 8:
            raise ValueError("Coarse angular samples must be divisible by eight.")
        if not 64 <= self.boundary_angular_samples <= 256:
            raise ValueError("Boundary angular samples must be between 64 and 256.")
        if self.boundary_angular_samples % 8:
            raise ValueError("Boundary angular samples must be divisible by eight.")
        if not 0.0 <= self.penalty_start_inside_fraction <= 0.50:
            raise ValueError(
                "Shape penalty start must be between zero and 0.5 seed diameter "
                "inside the fitted oval."
            )
        if not 0.005 <= self.outward_penalty_half_life_fraction <= 0.50:
            raise ValueError(
                "Outward penalty half-life must be between 0.005 and 0.5 seed "
                "diameter."
            )
        if not 0.01 <= self.outward_hard_cutoff_fraction <= 0.50:
            raise ValueError(
                "Outward hard cutoff must be between 0.01 and 0.5 seed diameter."
            )
        if (
            self.outward_penalty_half_life_fraction
            > self.outward_hard_cutoff_fraction
        ):
            raise ValueError(
                "Outward penalty half-life cannot exceed the hard cutoff."
            )
        for name, value in (
            ("boundary smoothness", self.boundary_smoothness),
            ("Lab contrast weight", self.lab_contrast_weight),
            ("edge barrier threshold", self.edge_barrier_threshold),
            ("edge-gap sealing fraction", self.edge_gap_sealing_fraction),
            ("minimum boundary strength", self.minimum_boundary_strength),
            ("minimum edge coverage", self.minimum_edge_coverage),
            ("minimum sector coverage", self.minimum_sector_coverage),
            ("maximum unsupported arc fraction", self.maximum_unsupported_arc_fraction),
            ("maximum other-label overlap", self.maximum_other_label_overlap),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name.capitalize()} must be between zero and one.")
        if not 1.0 <= self.colour_tolerance_lab <= 100.0:
            raise ValueError("Lab tolerance must be between 1 and 100.")
        if not 1 <= self.maximum_added_pixels <= 10_000_000:
            raise ValueError("Maximum added pixels must be positive.")
        if not 128 <= self.maximum_roi_size_px <= 2048:
            raise ValueError("Maximum ROI size must be between 128 and 2048 pixels.")


@dataclass(frozen=True, slots=True)
class EllipseHypothesis:
    """One compact auto-rotated ellipse prior in full-image coordinates."""

    centre_xy: tuple[float, float]
    axes_xy: tuple[float, float]
    angle_degrees: float
    polygon: np.ndarray
    score: float
    edge_coverage: float


@dataclass(frozen=True, slots=True)
class ShapeGuidedFillRegion:
    """A compact safe-to-commit mask and diagnostic preview geometry."""

    x: int
    y: int
    mask: np.ndarray
    prior_polygon: np.ndarray | None
    boundary_polygon: np.ndarray | None
    supported_boundary: np.ndarray | None
    added_count: int
    edge_coverage: float
    sector_coverage: float
    maximum_gap_fraction: float
    confidence: float
    accepted: bool
    reason: str


def _normalized_edge(values: np.ndarray) -> np.ndarray:
    edge = np.asarray(values)
    if edge.ndim != 2:
        raise ValueError("Edge strength must be a two-dimensional raster.")
    result = edge.astype(np.float32, copy=False)
    if result.size and float(np.nanmax(result)) > 1.0:
        result = result / 255.0
    return np.nan_to_num(result, nan=0.0, posinf=1.0, neginf=0.0).clip(0.0, 1.0)


def shape_outward_extension_pressure(
    signed_outward_distance_px: np.ndarray | float,
    preferred_seed_diameter_px: float,
    options: ShapeGuidedFillOptions = ShapeGuidedFillOptions(),
) -> np.ndarray:
    """Return Shape fill's one-sided geometric extension pressure.

    Pressure is one deep inside the ellipse, then begins decaying at the
    configured inward offset.  It halves every configured fraction of the
    preferred seed diameter and is exactly zero strictly beyond the outward
    hard cutoff.
    """

    diameter = float(preferred_seed_diameter_px)
    if not np.isfinite(diameter) or diameter <= 0.0:
        raise ValueError("Preferred seed diameter must be positive and finite.")
    signed_distance = np.asarray(signed_outward_distance_px, dtype=np.float32)
    if not np.all(np.isfinite(signed_distance)):
        raise ValueError("Signed outward distances must be finite.")
    start = -diameter * float(options.penalty_start_inside_fraction)
    distance_from_start = np.maximum(signed_distance - start, 0.0)
    half_life = max(
        np.finfo(np.float32).eps,
        diameter * float(options.outward_penalty_half_life_fraction),
    )
    cutoff = diameter * float(options.outward_hard_cutoff_fraction)
    pressure = np.exp2(-distance_from_start / half_life).astype(
        np.float32, copy=False
    )
    return np.where(signed_distance > cutoff, 0.0, pressure).astype(
        np.float32, copy=False
    )


def _ellipse_geometry(
    centres_x: np.ndarray,
    centres_y: np.ndarray,
    axes_a: np.ndarray,
    axes_b: np.ndarray,
    rotations: np.ndarray,
    angles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    cosine = np.cos(angles)[None, :]
    sine = np.sin(angles)[None, :]
    cos_r = np.cos(rotations)[:, None]
    sin_r = np.sin(rotations)[:, None]
    local_x = axes_a[:, None] * cosine
    local_y = axes_b[:, None] * sine
    xs = centres_x[:, None] + local_x * cos_r - local_y * sin_r
    ys = centres_y[:, None] + local_x * sin_r + local_y * cos_r
    return xs.astype(np.float32), ys.astype(np.float32)


def _candidate_scores(
    edge: np.ndarray,
    centres_x: np.ndarray,
    centres_y: np.ndarray,
    axes_a: np.ndarray,
    axes_b: np.ndarray,
    rotations: np.ndarray,
    options: ShapeGuidedFillOptions,
    *,
    reference_centre: tuple[float, float],
    reference_radius: float,
    angular_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    angles = np.linspace(0.0, 2.0 * pi, angular_samples, endpoint=False, dtype=np.float32)
    xs, ys = _ellipse_geometry(
        centres_x, centres_y, axes_a, axes_b, rotations, angles
    )
    sampled = cv2.remap(
        edge,
        xs,
        ys,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(np.float32)
    height, width = edge.shape
    valid = (xs >= 0.0) & (xs <= width - 1) & (ys >= 0.0) & (ys <= height - 1)
    sampled *= valid
    sector_count = 8
    sector_width = angular_samples // sector_count
    sectors = sampled.reshape(len(sampled), sector_count, sector_width)
    sector_peaks = np.max(sectors, axis=2)
    sector_coverage = np.mean(
        sector_peaks >= options.minimum_boundary_strength * 0.70, axis=1
    )
    point_coverage = np.mean(
        sampled >= options.minimum_boundary_strength * 0.70, axis=1
    )
    evidence_score = (
        0.48 * np.mean(sampled, axis=1)
        + 0.27 * np.mean(sector_peaks, axis=1)
        + 0.25 * sector_coverage
    )
    centre_distance = np.hypot(
        centres_x - float(reference_centre[0]),
        centres_y - float(reference_centre[1]),
    )
    centre_scale = max(1.0, reference_radius * max(options.centre_search_fraction, 0.04))
    centre_penalty = 0.045 * centre_distance / centre_scale
    radius_penalty = 0.025 * np.abs(axes_a / max(1.0, reference_radius) - 1.0)
    invalid_penalty = 0.30 * (1.0 - np.mean(valid, axis=1))
    return (
        evidence_score - centre_penalty - radius_penalty - invalid_penalty,
        point_coverage.astype(np.float32),
    )


def _ellipse_polygon(
    centre_xy: tuple[float, float],
    axes_xy: tuple[float, float],
    angle_radians: float,
    samples: int,
) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * pi, samples, endpoint=False, dtype=np.float32)
    xs, ys = _ellipse_geometry(
        np.asarray((centre_xy[0],), np.float32),
        np.asarray((centre_xy[1],), np.float32),
        np.asarray((axes_xy[0],), np.float32),
        np.asarray((axes_xy[1],), np.float32),
        np.asarray((angle_radians,), np.float32),
        angles,
    )
    return np.rint(np.column_stack((xs[0], ys[0]))).astype(np.int32)


def fit_rotated_edge_ellipse(
    edge_strength: np.ndarray,
    centre_xy: tuple[float, float],
    expected_seed_diameter_px: float,
    options: ShapeGuidedFillOptions = ShapeGuidedFillOptions(),
) -> EllipseHypothesis | None:
    """Find a bounded, automatically rotated ellipse prior near ``centre_xy``.

    A returned hypothesis is not sufficient authorization to paint.  Call
    :func:`shape_guided_fill_region` to refine and confidence-check a real
    boundary before committing any pixels.
    """

    diameter = float(expected_seed_diameter_px)
    if not np.isfinite(diameter) or diameter < 8.0:
        raise ValueError("Expected seed diameter must be at least eight pixels.")
    raw_edge = np.asarray(edge_strength)
    if raw_edge.ndim != 2:
        raise ValueError("Edge strength must be a two-dimensional raster.")
    # Normalize and dilate only the local candidate window.  A full-resolution
    # confidence conversion would make hover cost depend on photograph size.
    source_height, source_width = raw_edge.shape
    cx, cy = float(centre_xy[0]), float(centre_xy[1])
    preferred_diameter = diameter * float(options.preferred_scale)
    radius = preferred_diameter * 0.5
    candidate_extent = int(
        ceil(
            radius * options.maximum_scale
            + preferred_diameter * options.centre_search_fraction
            + 8.0
        )
    )
    x0 = max(0, int(np.floor(cx - candidate_extent)))
    x1 = min(source_width, int(np.ceil(cx + candidate_extent + 1)))
    y0 = max(0, int(np.floor(cy - candidate_extent)))
    y1 = min(source_height, int(np.ceil(cy + candidate_extent + 1)))
    if x1 - x0 > options.maximum_roi_size_px or y1 - y0 > options.maximum_roi_size_px:
        return None
    edge = _normalized_edge(raw_edge[y0:y1, x0:x1])
    if edge.size == 0 or float(np.max(edge)) <= 0.0:
        return None
    local_cx, local_cy = cx - x0, cy - y0

    # Fit against a narrowly dilated copy so a one-pixel ridge is not missed by
    # coarse angular samples.  Exact, undilated evidence is used for refinement.
    dilated_three = cv2.dilate(edge, np.ones((3, 3), np.uint8))
    dilated_five = cv2.dilate(edge, np.ones((5, 5), np.uint8))
    fit_edge = np.maximum(edge, np.maximum(dilated_three * 0.86, dilated_five * 0.62))
    centre_offset = preferred_diameter * options.centre_search_fraction
    offsets = np.asarray((-centre_offset, 0.0, centre_offset), np.float32)
    if centre_offset <= 0.0:
        offsets = np.asarray((0.0,), np.float32)
    scales = np.unique(
        np.concatenate(
            (
                np.linspace(options.minimum_scale, options.maximum_scale, 6),
                np.asarray((1.0,)),
            )
        )
    ).astype(np.float32)
    ratios = (
        np.asarray((1.0,), np.float32)
        if options.shape == "circle"
        else np.linspace(1.0, options.maximum_axis_ratio, 6, dtype=np.float32)
    )
    if options.shape == "circle" or not options.auto_rotation:
        rotations_degrees = np.asarray((options.initial_rotation_degrees,), np.float32)
    else:
        rotations_degrees = np.arange(
            0.0, 180.0, options.rotation_step_degrees, dtype=np.float32
        )
    centre_dx, centre_dy, scale_grid, ratio_grid, rotation_grid = np.meshgrid(
        offsets, offsets, scales, ratios, rotations_degrees, indexing="ij"
    )
    coarse_cx = (local_cx + centre_dx.ravel()).astype(np.float32)
    coarse_cy = (local_cy + centre_dy.ravel()).astype(np.float32)
    coarse_a = (radius * scale_grid.ravel()).astype(np.float32)
    coarse_b = (coarse_a / ratio_grid.ravel()).astype(np.float32)
    coarse_rotation = np.deg2rad(rotation_grid.ravel()).astype(np.float32)
    scores, coverage = _candidate_scores(
        fit_edge,
        coarse_cx,
        coarse_cy,
        coarse_a,
        coarse_b,
        coarse_rotation,
        options,
        reference_centre=(local_cx, local_cy),
        reference_radius=radius,
        angular_samples=options.coarse_angular_samples,
    )
    top_count = min(4, len(scores))
    if top_count == 0:
        return None
    top_indices = np.argpartition(scores, -top_count)[-top_count:]

    # A small full-resolution coordinate refinement keeps the bank bounded
    # while recovering angles and centres between the coarse grid positions.
    fine_cx: list[float] = []
    fine_cy: list[float] = []
    fine_a: list[float] = []
    fine_b: list[float] = []
    fine_rotation: list[float] = []
    centre_steps = (-2.0, 0.0, 2.0)
    scale_steps = (0.96, 1.0, 1.04)
    ratio_steps = (0.96, 1.0, 1.04) if options.shape == "ellipse" else (1.0,)
    angle_steps = (
        (-5.0, 0.0, 5.0)
        if options.shape == "ellipse" and options.auto_rotation
        else (0.0,)
    )
    for index in top_indices:
        base_ratio = float(coarse_a[index] / max(3.0, coarse_b[index]))
        for dy in centre_steps:
            for dx in centre_steps:
                for scale_step in scale_steps:
                    for ratio_step in ratio_steps:
                        refined_ratio = np.clip(
                            base_ratio * ratio_step, 1.0, options.maximum_axis_ratio
                        )
                        refined_a = float(coarse_a[index]) * scale_step
                        refined_b = refined_a / float(refined_ratio)
                        for angle_step in angle_steps:
                            fine_cx.append(float(coarse_cx[index]) + dx)
                            fine_cy.append(float(coarse_cy[index]) + dy)
                            fine_a.append(refined_a)
                            fine_b.append(refined_b)
                            fine_rotation.append(
                                float(coarse_rotation[index]) + np.deg2rad(angle_step)
                            )
    fine_cx_array = np.asarray(fine_cx, np.float32)
    fine_cy_array = np.asarray(fine_cy, np.float32)
    fine_a_array = np.asarray(fine_a, np.float32)
    fine_b_array = np.asarray(fine_b, np.float32)
    fine_rotation_array = np.asarray(fine_rotation, np.float32)
    fine_scores, fine_coverage = _candidate_scores(
        fit_edge,
        fine_cx_array,
        fine_cy_array,
        fine_a_array,
        fine_b_array,
        fine_rotation_array,
        options,
        reference_centre=(local_cx, local_cy),
        reference_radius=radius,
        angular_samples=max(96, options.coarse_angular_samples),
    )
    best = int(np.argmax(fine_scores))
    best_rotation = float(fine_rotation_array[best] % pi)
    local_best_centre = (float(fine_cx_array[best]), float(fine_cy_array[best]))
    polygon = _ellipse_polygon(
        local_best_centre,
        (float(fine_a_array[best]), float(fine_b_array[best])),
        best_rotation,
        options.boundary_angular_samples,
    )
    polygon += np.asarray((x0, y0), np.int32)
    return EllipseHypothesis(
        centre_xy=(local_best_centre[0] + x0, local_best_centre[1] + y0),
        axes_xy=(float(fine_a_array[best]), float(fine_b_array[best])),
        angle_degrees=float(np.rad2deg(best_rotation)),
        polygon=polygon,
        score=float(fine_scores[best]),
        edge_coverage=float(fine_coverage[best]),
    )


def _smart_fill_options(
    options: ShapeGuidedFillOptions,
    maximum_distance_px: int,
    preferred_diameter_px: float,
) -> SmartFillOptions:
    """Map Shape fill's retained controls to canonical Smart fill.

    Shape authors its own ellipse-relative pressure field, so Smart fill's
    cursor-relative radius only bounds the compact working window and its
    cursor fall-off is unused. Shape semantics deliberately fix tunnelling off
    and use eight-connected foreground topology.
    """

    return SmartFillOptions(
        colour_tolerance_lab=options.colour_tolerance_lab,
        # Click-origin colour range is a Smart-fill-only control. Shape fill
        # validates its own fitted interior evidence and retains its established
        # neighbour-step semantics.
        click_colour_tolerance_lab=360.0,
        edge_stop_threshold=options.edge_barrier_threshold,
        tunnel_strength=0.0,
        edge_gap_sealing_px=max(
            0,
            min(
                64,
                int(
                    round(
                        preferred_diameter_px * options.edge_gap_sealing_fraction
                    )
                ),
            ),
        ),
        maximum_distance_from_cursor_px=int(
            np.clip(maximum_distance_px, 4, 4096)
        ),
        falloff_half_life_px=4096.0,
        maximum_added_pixels=options.maximum_added_pixels,
        connectivity=SHAPE_FILL_CONNECTIVITY,
    )


def _compact_mask_in_region(
    source_mask: np.ndarray,
    source_origin_xy: tuple[int, int],
    target_shape: tuple[int, int],
    target_origin_xy: tuple[int, int],
) -> np.ndarray:
    """Place one compact mask into another full-image-coordinate region."""

    result = np.zeros(target_shape, dtype=bool)
    source_x0, source_y0 = map(int, source_origin_xy)
    target_x0, target_y0 = map(int, target_origin_xy)
    source_x1 = source_x0 + source_mask.shape[1]
    source_y1 = source_y0 + source_mask.shape[0]
    target_x1 = target_x0 + target_shape[1]
    target_y1 = target_y0 + target_shape[0]
    overlap_x0 = max(source_x0, target_x0)
    overlap_y0 = max(source_y0, target_y0)
    overlap_x1 = min(source_x1, target_x1)
    overlap_y1 = min(source_y1, target_y1)
    if overlap_x0 >= overlap_x1 or overlap_y0 >= overlap_y1:
        return result
    result[
        overlap_y0 - target_y0 : overlap_y1 - target_y0,
        overlap_x0 - target_x0 : overlap_x1 - target_x0,
    ] = source_mask[
        overlap_y0 - source_y0 : overlap_y1 - source_y0,
        overlap_x0 - source_x0 : overlap_x1 - source_x0,
    ]
    return result


def _maximum_circular_false_run(values: np.ndarray) -> int:
    supported = np.asarray(values, dtype=bool).ravel()
    if not len(supported):
        return 0
    if not np.any(supported):
        return len(supported)
    doubled = np.concatenate((~supported, ~supported)).astype(np.int16)
    changes = np.diff(np.pad(doubled, (1, 1)))
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    return min(len(supported), int(np.max(stops - starts, initial=0)))


def _cyclic_boundary_path(
    unary: np.ndarray,
    smoothness: float,
    maximum_shift: int,
) -> np.ndarray:
    angle_count, state_count = unary.shape
    # Rotate to the best-supported angular row so the small cyclic start bank
    # is not chosen from an arbitrary evidence gap.
    rotation = int(np.argmax(np.max(unary, axis=1)))
    rolled = np.roll(unary, -rotation, axis=0)
    centre_state = state_count // 2
    top = np.argsort(rolled[0])[-min(10, state_count) :]
    start_states = np.unique(np.concatenate((top, (centre_state,))))
    states = np.arange(state_count, dtype=np.int32)
    best_score = -np.inf
    best_path: np.ndarray | None = None
    shifts = np.arange(-maximum_shift, maximum_shift + 1, dtype=np.int32)
    for start_state in start_states:
        previous = np.full(state_count, -np.inf, np.float32)
        previous[int(start_state)] = rolled[0, int(start_state)]
        predecessors = np.full((angle_count, state_count), -1, np.int16)
        for angle_index in range(1, angle_count):
            candidates = np.full((len(shifts), state_count), -np.inf, np.float32)
            for shift_index, shift in enumerate(shifts):
                sources = states - shift
                valid = (sources >= 0) & (sources < state_count)
                candidates[shift_index, valid] = (
                    previous[sources[valid]] - smoothness * abs(int(shift))
                )
            chosen = np.argmax(candidates, axis=0)
            predecessors[angle_index] = (states - shifts[chosen]).astype(np.int16)
            previous = candidates[chosen, states] + rolled[angle_index]
        closure = smoothness * 1.5 * np.abs(states - int(start_state))
        closure[np.abs(states - int(start_state)) > maximum_shift] = np.inf
        end_state = int(np.argmax(previous - closure))
        total = float(previous[end_state] - closure[end_state])
        if not np.isfinite(total) or total <= best_score:
            continue
        path = np.empty(angle_count, np.int32)
        path[-1] = end_state
        for angle_index in range(angle_count - 1, 0, -1):
            source = int(predecessors[angle_index, path[angle_index]])
            path[angle_index - 1] = int(start_state) if source < 0 else source
        best_score = total
        best_path = path
    if best_path is None:
        return np.full(angle_count, centre_state, np.int32)
    return np.roll(best_path, rotation)


def _refusal(
    reason: str,
    *,
    x: int,
    y: int,
    shape: tuple[int, int],
    prior: np.ndarray | None = None,
    boundary: np.ndarray | None = None,
    supported: np.ndarray | None = None,
    edge_coverage: float = 0.0,
    sector_coverage: float = 0.0,
    gap_fraction: float = 1.0,
    confidence: float = 0.0,
) -> ShapeGuidedFillRegion:
    return ShapeGuidedFillRegion(
        x=x,
        y=y,
        mask=np.zeros(shape, dtype=bool),
        prior_polygon=prior,
        boundary_polygon=boundary,
        supported_boundary=supported,
        added_count=0,
        edge_coverage=edge_coverage,
        sector_coverage=sector_coverage,
        maximum_gap_fraction=gap_fraction,
        confidence=confidence,
        accepted=False,
        reason=reason,
    )


def shape_guided_fill_region(
    labels: np.ndarray,
    corrected_bgr: np.ndarray,
    edge_strength: np.ndarray,
    click_xy: tuple[float, float],
    instance_id: int,
    expected_seed_diameter_px: float,
    options: ShapeGuidedFillOptions = ShapeGuidedFillOptions(),
    *,
    corrected_lab: np.ndarray | None = None,
) -> ShapeGuidedFillRegion:
    """Return an edge-refined local mask or an explicit confidence refusal."""

    source_labels = np.asarray(labels)
    image = np.asarray(corrected_bgr)
    raw_edge = np.asarray(edge_strength)
    if source_labels.ndim != 2 or image.shape[:2] != source_labels.shape or raw_edge.shape != source_labels.shape:
        raise ValueError("Labels, image, and edge strength must share one image shape.")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Shape-guided fill requires a three-channel corrected BGR image.")
    if not 1 <= int(instance_id) <= np.iinfo(np.uint16).max:
        raise ValueError("Instance ID must fit in an unsigned 16-bit label map.")
    height, width = source_labels.shape
    click_x = int(np.clip(round(click_xy[0]), 0, width - 1))
    click_y = int(np.clip(round(click_xy[1]), 0, height - 1))
    occupied = int(source_labels[click_y, click_x])
    if occupied not in {0, int(instance_id)}:
        return _refusal(
            "The cursor is inside a different annotated seed.",
            x=click_x,
            y=click_y,
            shape=(1, 1),
        )
    diameter = float(expected_seed_diameter_px)
    if not np.isfinite(diameter) or diameter < 8.0:
        raise ValueError("Expected seed diameter must be at least eight pixels.")
    preferred_diameter = diameter * float(options.preferred_scale)
    outward_cutoff_px = (
        preferred_diameter * options.outward_hard_cutoff_fraction
    )
    estimated_extent = int(
        ceil(
            preferred_diameter * 0.5 * options.maximum_scale
            + preferred_diameter * options.centre_search_fraction
            + outward_cutoff_px
            + 12.0
        )
    )
    if estimated_extent * 2 + 1 > options.maximum_roi_size_px:
        return _refusal(
            "The requested seed scale exceeds the bounded shape-guided working region.",
            x=click_x,
            y=click_y,
            shape=(1, 1),
        )
    hypothesis = fit_rotated_edge_ellipse(
        raw_edge,
        (click_x, click_y),
        expected_seed_diameter_px,
        options,
    )
    if hypothesis is None:
        return _refusal(
            "No edge evidence was available near the cursor; no ellipse was painted.",
            x=click_x,
            y=click_y,
            shape=(1, 1),
        )

    centre_x, centre_y = hypothesis.centre_xy
    axis_a, axis_b = hypothesis.axes_xy
    extent = ceil(max(axis_a, axis_b) + outward_cutoff_px + 4)
    x0, x1 = max(0, int(np.floor(centre_x - extent))), min(width, int(np.ceil(centre_x + extent + 1)))
    y0, y1 = max(0, int(np.floor(centre_y - extent))), min(height, int(np.ceil(centre_y + extent + 1)))
    roi_height, roi_width = y1 - y0, x1 - x0
    if roi_width > options.maximum_roi_size_px or roi_height > options.maximum_roi_size_px:
        return _refusal(
            "The fitted seed exceeds the bounded shape-guided working region.",
            x=x0,
            y=y0,
            shape=(max(1, roi_height), max(1, roi_width)),
            prior=hypothesis.polygon,
        )
    roi_labels = source_labels[y0:y1, x0:x1]
    roi_edge = _normalized_edge(raw_edge[y0:y1, x0:x1])
    if corrected_lab is None:
        roi_lab = cv2.cvtColor(
            np.ascontiguousarray(image[y0:y1, x0:x1], dtype=np.uint8),
            cv2.COLOR_BGR2LAB,
        )
    else:
        lab = np.asarray(corrected_lab)
        if lab.shape != image.shape:
            raise ValueError("Precomputed Lab image must match corrected BGR.")
        roi_lab = np.ascontiguousarray(lab[y0:y1, x0:x1], dtype=np.uint8)

    prior_local = hypothesis.polygon - np.asarray((x0, y0), np.int32)
    prior_mask = np.zeros((roi_height, roi_width), np.uint8)
    cv2.fillPoly(
        prior_mask, [prior_local.reshape(-1, 1, 2)], 1, lineType=cv2.LINE_8
    )
    # Form a signed distance: positive outside, negative inside.  This lets the
    # soft search penalty begin before the nominal oval while retaining a hard
    # cutoff only outside it.
    outside_distance = cv2.distanceTransform(
        (prior_mask == 0).astype(np.uint8),
        cv2.DIST_L2,
        cv2.DIST_MASK_PRECISE,
    )
    inside_distance = cv2.distanceTransform(
        prior_mask.astype(np.uint8),
        cv2.DIST_L2,
        cv2.DIST_MASK_PRECISE,
    )
    signed_distance = outside_distance - inside_distance
    extension_pressure = shape_outward_extension_pressure(
        signed_distance,
        preferred_diameter,
        options,
    )
    # One byte of effective growth pressure is the minimum meaningful support.
    # This makes an intentionally tiny half-life visibly restrictive even in a
    # perfectly uniform, edgeless area instead of letting it flood to the hard
    # cutoff unchanged.
    allowed = extension_pressure >= (1.0 / 255.0)
    maximum_work_distance = int(
        ceil(
            np.hypot(
                max(click_x - x0, x1 - 1 - click_x),
                max(click_y - y0, y1 - 1 - click_y),
            )
        )
    ) + 1
    smart_options = _smart_fill_options(
        options, maximum_work_distance, preferred_diameter
    )
    smart_region = smart_fill_region(
        source_labels,
        image,
        raw_edge,
        (click_x, click_y),
        instance_id,
        smart_options,
        corrected_lab=corrected_lab,
        allowed_mask=allowed,
        allowed_origin_xy=(x0, y0),
        extension_pressure_mask=extension_pressure,
        extension_pressure_origin_xy=(x0, y0),
    )
    core = _compact_mask_in_region(
        smart_region.mask,
        (smart_region.x, smart_region.y),
        (roi_height, roi_width),
        (x0, y0),
    )

    angles = np.linspace(
        0.0,
        2.0 * pi,
        options.boundary_angular_samples,
        endpoint=False,
        dtype=np.float32,
    )
    rotation = np.deg2rad(hypothesis.angle_degrees)
    cos_t, sin_t = np.cos(angles), np.sin(angles)
    cos_r, sin_r = np.cos(rotation), np.sin(rotation)
    base_x = centre_x + axis_a * cos_t * cos_r - axis_b * sin_t * sin_r
    base_y = centre_y + axis_a * cos_t * sin_r + axis_b * sin_t * cos_r
    normal_local_x = cos_t / max(3.0, axis_a)
    normal_local_y = sin_t / max(3.0, axis_b)
    normal_x = normal_local_x * cos_r - normal_local_y * sin_r
    normal_y = normal_local_x * sin_r + normal_local_y * cos_r
    normal_length = np.maximum(1e-6, np.hypot(normal_x, normal_y))
    normal_x /= normal_length
    normal_y /= normal_length
    # Every inward normal is searchable almost to its centre projection. The
    # fitted ellipse contributes no inward penalty or inward band. Positive
    # offsets alone are limited by the explicit outward cutoff.
    inward_limits = np.maximum(
        3.0,
        (base_x - centre_x) * normal_x
        + (base_y - centre_y) * normal_y
        - 1.0,
    )
    inward_extent = float(np.max(inward_limits))
    state_count = min(
        129,
        max(65, int(np.ceil(inward_extent + outward_cutoff_px)) + 1),
    )
    offsets = np.linspace(
        -inward_extent,
        outward_cutoff_px,
        state_count,
        dtype=np.float32,
    )
    offset_step = float(abs(offsets[1] - offsets[0]))
    valid_offset = offsets[None, :] >= -inward_limits[:, None]
    map_x = (base_x[:, None] + normal_x[:, None] * offsets[None, :]).astype(np.float32)
    map_y = (base_y[:, None] + normal_y[:, None] * offsets[None, :]).astype(np.float32)
    edge_samples = cv2.remap(
        roi_edge,
        map_x - x0,
        map_y - y0,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(np.float32)
    edge_samples[~valid_offset] = 0.0
    left = np.pad(edge_samples[:, :-1], ((0, 0), (1, 0)), constant_values=0.0)
    right = np.pad(edge_samples[:, 1:], ((0, 0), (0, 1)), constant_values=0.0)
    local_peak = (edge_samples >= left) & (edge_samples >= right)
    evidence = np.where(local_peak, edge_samples, edge_samples * 0.25)
    boundary_pressure = shape_outward_extension_pressure(
        offsets,
        preferred_diameter,
        options,
    )

    if options.lab_contrast_weight > 0.0:
        inner_x = (map_x - 2.0 * normal_x[:, None]).astype(np.float32)
        inner_y = (map_y - 2.0 * normal_y[:, None]).astype(np.float32)
        outer_x = (map_x + 2.0 * normal_x[:, None]).astype(np.float32)
        outer_y = (map_y + 2.0 * normal_y[:, None]).astype(np.float32)
        contrasts = []
        for channel in range(3):
            plane = roi_lab[:, :, channel].astype(np.float32)
            inside = cv2.remap(
                plane,
                inner_x - x0,
                inner_y - y0,
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            outside = cv2.remap(
                plane,
                outer_x - x0,
                outer_y - y0,
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            contrasts.append(outside - inside)
        contrast = np.sqrt(sum(component * component for component in contrasts))
        contrast = np.clip(contrast / 60.0, 0.0, 1.0)
        evidence += options.lab_contrast_weight * contrast

    core_u8 = core.astype(np.uint8)
    core_inner_x = (map_x - x0 - 2.0 * normal_x[:, None]).astype(np.float32)
    core_inner_y = (map_y - y0 - 2.0 * normal_y[:, None]).astype(np.float32)
    core_outer_x = (map_x - x0 + 2.0 * normal_x[:, None]).astype(np.float32)
    core_outer_y = (map_y - y0 + 2.0 * normal_y[:, None]).astype(np.float32)
    inside_core = cv2.remap(
        core_u8,
        core_inner_x,
        core_inner_y,
        cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    outside_core = cv2.remap(
        core_u8,
        core_outer_x,
        core_outer_y,
        cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    evidence += 0.10 * ((inside_core > 0) & (outside_core == 0))
    # Apply the one-sided prior after every evidence term so neither Lab
    # contrast nor interior-core support can bypass the outward penalty.
    evidence *= boundary_pressure[None, :]
    unary = evidence
    unary[~valid_offset] = -1.0e6
    maximum_physical_shift = max(1.0, preferred_diameter * 0.03)
    maximum_shift = max(
        1, min(5, int(np.ceil(maximum_physical_shift / offset_step)))
    )
    smoothness = (
        0.025 + 0.075 * options.boundary_smoothness
    ) * offset_step
    path = _cyclic_boundary_path(unary, smoothness, maximum_shift)
    rows = np.arange(len(angles), dtype=np.int32)
    chosen_offsets = offsets[path]
    boundary_x = base_x + normal_x * chosen_offsets
    boundary_y = base_y + normal_y * chosen_offsets
    boundary = np.rint(np.column_stack((boundary_x, boundary_y))).astype(np.int32)
    boundary[:, 0] = np.clip(boundary[:, 0], 0, width - 1)
    boundary[:, 1] = np.clip(boundary[:, 1], 0, height - 1)
    chosen_edge = edge_samples[rows, path]
    supported = chosen_edge >= options.minimum_boundary_strength
    edge_coverage = float(np.mean(supported))
    sector_count = 16
    sector_width = len(supported) // sector_count
    sector_support = supported.reshape(sector_count, sector_width)
    sector_coverage = float(np.mean(np.any(sector_support, axis=1)))
    maximum_gap = _maximum_circular_false_run(supported)
    maximum_gap_fraction = float(maximum_gap / max(1, len(supported)))
    mean_edge = float(np.mean(chosen_edge))
    confidence = float(
        np.clip(
            0.55 * edge_coverage
            + 0.25 * mean_edge
            + 0.20 * sector_coverage,
            0.0,
            1.0,
        )
    )
    boundary_local = boundary - np.asarray((x0, y0), np.int32)
    polygon_mask = np.zeros((roi_height, roi_width), np.uint8)
    cv2.fillPoly(
        polygon_mask,
        [boundary_local.reshape(-1, 1, 2)],
        1,
        lineType=cv2.LINE_8,
    )
    other = (roi_labels != 0) & (roi_labels != int(instance_id))
    polygon_pixels = int(np.count_nonzero(polygon_mask))
    other_overlap = (
        float(np.count_nonzero((polygon_mask > 0) & other)) / polygon_pixels
        if polygon_pixels
        else 1.0
    )
    refusal_reason = ""
    if edge_coverage < options.minimum_edge_coverage:
        refusal_reason = (
            f"Only {edge_coverage:.0%} of the refined boundary has credible edge "
            "support; no ellipse was painted."
        )
    elif sector_coverage < options.minimum_sector_coverage:
        refusal_reason = (
            f"Edge evidence reaches only {sector_coverage:.0%} of boundary sectors; "
            "no ellipse was painted."
        )
    elif maximum_gap_fraction > options.maximum_unsupported_arc_fraction:
        refusal_reason = (
            f"The largest unsupported boundary arc is {maximum_gap_fraction:.0%}; "
            "no ellipse was painted."
        )
    elif other_overlap > options.maximum_other_label_overlap:
        refusal_reason = (
            f"The candidate overlaps {other_overlap:.0%} of another annotated seed; "
            "nothing was painted."
        )
    if refusal_reason:
        return _refusal(
            refusal_reason,
            x=x0,
            y=y0,
            shape=(roi_height, roi_width),
            prior=hypothesis.polygon,
            boundary=boundary,
            supported=supported,
            edge_coverage=edge_coverage,
            sector_coverage=sector_coverage,
            gap_fraction=maximum_gap_fraction,
            confidence=confidence,
        )
    writable = (roi_labels == 0) | (roi_labels == int(instance_id))
    # The refined contour supplies the confidence/refusal check, not a stamped
    # polygon. The already-calculated canonical Smart-fill core is the final
    # selection under the one-sided prior: completely unrestricted throughout
    # the fitted ellipse, gradually discouraged just outside it, and impossible
    # only beyond the hard cutoff.
    mask = core & writable
    added = int(np.count_nonzero(mask & (roi_labels != int(instance_id))))
    if added > options.maximum_added_pixels:
        return _refusal(
            f"The refined boundary would add {added:,} pixels, above the configured "
            f"{options.maximum_added_pixels:,}-pixel limit.",
            x=x0,
            y=y0,
            shape=(roi_height, roi_width),
            prior=hypothesis.polygon,
            boundary=boundary,
            supported=supported,
            edge_coverage=edge_coverage,
            sector_coverage=sector_coverage,
            gap_fraction=maximum_gap_fraction,
            confidence=confidence,
        )
    return ShapeGuidedFillRegion(
        x=x0,
        y=y0,
        mask=mask,
        prior_polygon=hypothesis.polygon,
        boundary_polygon=boundary,
        supported_boundary=supported,
        added_count=added,
        edge_coverage=edge_coverage,
        sector_coverage=sector_coverage,
        maximum_gap_fraction=maximum_gap_fraction,
        confidence=confidence,
        accepted=True,
        reason="",
    )


def shape_guided_fill_instance(
    labels: np.ndarray,
    corrected_bgr: np.ndarray,
    edge_strength: np.ndarray,
    click_xy: tuple[float, float],
    instance_id: int,
    expected_seed_diameter_px: float,
    options: ShapeGuidedFillOptions = ShapeGuidedFillOptions(),
    *,
    corrected_lab: np.ndarray | None = None,
) -> tuple[np.ndarray, int]:
    """Apply one accepted region to a copied label map, preserving other IDs."""

    source = np.asarray(labels)
    region = shape_guided_fill_region(
        source,
        corrected_bgr,
        edge_strength,
        click_xy,
        instance_id,
        expected_seed_diameter_px,
        options,
        corrected_lab=corrected_lab,
    )
    result = source.astype(np.uint16, copy=True)
    if not region.accepted or not region.added_count:
        return result, 0
    roi = result[
        region.y : region.y + region.mask.shape[0],
        region.x : region.x + region.mask.shape[1],
    ]
    writable = region.mask & ((roi == 0) | (roi == int(instance_id)))
    roi[writable] = np.uint16(instance_id)
    return result, region.added_count


__all__ = (
    "EllipseHypothesis",
    "SHAPE_FILL_CONNECTIVITY",
    "ShapeGuidedFillOptions",
    "ShapeGuidedFillRegion",
    "fit_rotated_edge_ellipse",
    "shape_guided_fill_instance",
    "shape_guided_fill_region",
    "shape_outward_extension_pressure",
)
