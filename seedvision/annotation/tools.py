"""Image-aware tools used while annotating individual seeds.

These functions deliberately have no Qt dependency.  The desktop view supplies
the already-calculated edge rasters and corrected BGR image, then applies the
returned path, polygon, or label map as one undoable annotation draft change.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, hypot, pi

import cv2
import numpy as np


ANNOTATION_EDGE_SOURCES = frozenset(
    {
        "adaptive",
        "ridges",
        "reference_ridges",
        "conservative_reference_ridges",
        "normalized_reference_ridges",
        "traces",
        "magnitude",
        "net_physical",
        "conservative_net_physical",
        "normalized_net_physical",
    }
)
EDGE_TRACE_EDGE_SOURCES = ANNOTATION_EDGE_SOURCES - {
    "net_physical", "conservative_net_physical"
}
SMART_FILL_EDGE_SOURCES = ANNOTATION_EDGE_SOURCES


def _edge_source_error(allowed: frozenset[str]) -> str:
    return "Edge source must be one of: " + ", ".join(sorted(allowed)) + "."


@dataclass(frozen=True, slots=True)
class EdgeTraceOptions:
    """Controls for a magnetic edge path between two user points."""

    search_radius_px: int = 18
    edge_attraction: float = 0.85
    tangent_mode: str = "undirected"
    tangent_weight: float = 0.45
    smoothing: int = 2
    edge_source: str = "magnitude"
    fill_closed_loops: bool = True

    def __post_init__(self) -> None:
        if not 2 <= self.search_radius_px <= 200:
            raise ValueError("Edge search radius must be between 2 and 200 pixels.")
        if not 0.0 <= self.edge_attraction <= 1.0:
            raise ValueError("Edge attraction must be between zero and one.")
        if self.tangent_mode not in {"off", "undirected", "directed"}:
            raise ValueError("Tangent mode must be off, undirected, or directed.")
        if not 0.0 <= self.tangent_weight <= 1.0:
            raise ValueError("Tangent weight must be between zero and one.")
        if not 0 <= self.smoothing <= 8:
            raise ValueError("Path smoothing must be between zero and eight.")
        if self.edge_source not in EDGE_TRACE_EDGE_SOURCES:
            raise ValueError(_edge_source_error(EDGE_TRACE_EDGE_SOURCES))
        if not isinstance(self.fill_closed_loops, bool):
            raise ValueError("Fill closed loops must be enabled or disabled.")


@dataclass(frozen=True, slots=True)
class SmartFillOptions:
    """Controls for locally adaptive, edge-stopped instance growth."""

    colour_tolerance_lab: float = 18.0
    edge_stop_threshold: float = 0.58
    tunnel_strength: float = 0.0
    maximum_distance_from_cursor_px: int = 60
    falloff_half_life_px: float = 40.0
    edge_gap_sealing_px: int = 2
    maximum_added_pixels: int = 150_000
    connectivity: int = 8
    edge_source: str = "magnitude"
    click_colour_tolerance_lab: float = 72.0

    def __post_init__(self) -> None:
        if not 1.0 <= self.colour_tolerance_lab <= 100.0:
            raise ValueError("Lab tolerance must be between 1 and 100.")
        if not 1.0 <= self.click_colour_tolerance_lab <= 360.0:
            raise ValueError(
                "Click-origin Lab tolerance must be between 1 and 360."
            )
        if not 0.0 <= self.edge_stop_threshold <= 1.0:
            raise ValueError("Edge stop threshold must be between zero and one.")
        if not 0.0 <= self.tunnel_strength <= 1.0:
            raise ValueError("Tunnel strength must be between zero and one.")
        if not 4 <= self.maximum_distance_from_cursor_px <= 4096:
            raise ValueError(
                "Maximum distance from cursor must be between 4 and 4096 pixels."
            )
        if not 1.0 <= self.falloff_half_life_px <= 4096.0:
            raise ValueError(
                "Fall-off half-life must be between 1 and 4096 pixels."
            )
        if not 0 <= int(self.edge_gap_sealing_px) <= 64:
            raise ValueError("Edge-gap sealing must be between 0 and 64 pixels.")
        if not 1 <= self.maximum_added_pixels <= 10_000_000:
            raise ValueError("Maximum added pixels must be positive.")
        if self.connectivity not in {4, 8}:
            raise ValueError("Connectivity must be four or eight.")
        if self.edge_source not in SMART_FILL_EDGE_SOURCES:
            raise ValueError(_edge_source_error(SMART_FILL_EDGE_SOURCES))


@dataclass(frozen=True, slots=True)
class SmartFillRegion:
    """Compact smart-fill preview mask and its full-image origin."""

    x: int
    y: int
    mask: np.ndarray
    added_count: int


def _normalized_strength(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32)
    if result.ndim != 2:
        raise ValueError("Edge strength must be a two-dimensional raster.")
    if result.size and float(np.nanmax(result)) > 1.0:
        result = result / 255.0
    return np.nan_to_num(result, nan=0.0, posinf=1.0, neginf=0.0).clip(0.0, 1.0)


def _smart_fill_extension_pressure(
    distance_px: np.ndarray | float,
    half_life_px: float,
) -> np.ndarray:
    """Return the exact radial Smart Fill pressure ``2**(-d / h)``."""

    return np.exp2(
        -np.asarray(distance_px, dtype=np.float32) / float(half_life_px)
    )


def _edge_evidence(values: np.ndarray, source: str) -> np.ndarray:
    """Return comparable edge confidence without treating trace IDs as strength.

    Oriented traces are categorical component labels.  Normalizing their raw
    integer IDs would make early components artificially weak and late
    components artificially strong, so every positive trace is structural
    support.  The remaining sources are genuine scalar confidence rasters.
    """

    raster = np.asarray(values)
    if source == "traces":
        if raster.ndim != 2:
            raise ValueError("Edge strength must be a two-dimensional raster.")
        return (raster > 0).astype(np.float32)
    if source == "adaptive" and np.issubdtype(raster.dtype, np.integer):
        maximum = int(np.max(raster, initial=0))
        # Values outside an 8-bit confidence range can only be categorical
        # labels in the current pipeline contract.
        if maximum > 255:
            return (raster > 0).astype(np.float32)
    return _normalized_strength(raster)


def _anchored_support_component(
    edge_strip: np.ndarray,
    centre_row: int,
    source: str,
) -> np.ndarray | None:
    """Find thin support continuously joining both clicked anchors."""

    positive = edge_strip[edge_strip > 0.0]
    if not positive.size:
        return None
    if source == "traces":
        threshold = 0.5
    elif source in {"ridges", "reference_ridges", "conservative_reference_ridges"}:
        threshold = max(0.04, min(0.35, float(np.quantile(positive, 0.20))))
    else:
        threshold = max(0.08, min(0.55, float(np.quantile(positive, 0.55))))
    support = np.ascontiguousarray(edge_strip >= threshold, dtype=np.uint8)
    _count, labels = cv2.connectedComponents(support, connectivity=8)

    def endpoint_label(column: int) -> int:
        candidates = labels[:, column]
        distances = np.abs(np.arange(len(candidates)) - centre_row)
        valid = candidates > 0
        if not np.any(valid):
            return 0
        distances = np.where(valid, distances, len(candidates) + 1)
        return int(candidates[int(np.argmin(distances))])

    start_label = endpoint_label(0)
    end_label = endpoint_label(edge_strip.shape[1] - 1)
    if start_label <= 0 or start_label != end_label:
        return None
    return labels == start_label


def snap_edge_point(
    point_xy: tuple[float, float],
    edge_strength: np.ndarray,
    search_radius_px: int,
) -> tuple[int, int]:
    """Snap a cursor point to the strongest nearby edge with a distance bias."""

    edge_values = np.asarray(edge_strength)
    if edge_values.ndim != 2:
        raise ValueError("Edge strength must be a two-dimensional raster.")
    height, width = edge_values.shape
    x = int(np.clip(round(point_xy[0]), 0, width - 1))
    y = int(np.clip(round(point_xy[1]), 0, height - 1))
    radius = max(1, int(search_radius_px))
    x0, x1 = max(0, x - radius), min(width, x + radius + 1)
    y0, y1 = max(0, y - radius), min(height, y + radius + 1)
    roi = _normalized_strength(edge_values[y0:y1, x0:x1])
    yy, xx = np.mgrid[y0:y1, x0:x1]
    distance = np.hypot(xx - x, yy - y)
    inside = distance <= radius
    peak = float(np.max(roi[inside])) if np.any(inside) else 0.0
    if peak <= 0.0:
        return x, y
    # Prefer the nearest credible edge.  Maximising strength with only a weak
    # distance bias made a cursor placed directly on one seed edge jump to a
    # slightly stronger parallel edge several pixels away.
    credible = inside & (roi >= max(0.04, peak * 0.30))
    score = -distance + 0.35 * roi
    score[~credible] = -np.inf
    snapped_y, snapped_x = np.unravel_index(int(np.argmax(score)), score.shape)
    return int(snapped_x + x0), int(snapped_y + y0)


def trace_edge_path(
    start_xy: tuple[float, float],
    end_xy: tuple[float, float],
    edge_strength: np.ndarray,
    options: EdgeTraceOptions = EdgeTraceOptions(),
    tangent_hue: np.ndarray | None = None,
) -> np.ndarray:
    """Return a fast continuity-aware magnetic path between two anchors.

    Evidence is sampled in a bounded perpendicular corridor, then a banded
    dynamic program finds one continuous seam.  A ridge component that reaches
    both clicked anchors is locked preferentially, preventing a stronger nearby
    parallel ridge from stealing the middle of the path.  Tangent evidence is
    evaluated against each local transition and is gated by edge support.
    """

    edge_values = np.asarray(edge_strength)
    if edge_values.ndim != 2:
        raise ValueError("Edge strength must be a two-dimensional raster.")
    height, width = edge_values.shape
    hue_values = None if tangent_hue is None else np.asarray(tangent_hue)
    if hue_values is not None and hue_values.shape != edge_values.shape:
        raise ValueError("Tangent hue and edge strength must have the same shape.")
    start_x = int(np.clip(round(start_xy[0]), 0, width - 1))
    start_y = int(np.clip(round(start_xy[1]), 0, height - 1))
    end_x = int(np.clip(round(end_xy[0]), 0, width - 1))
    end_y = int(np.clip(round(end_xy[1]), 0, height - 1))
    if (start_x, start_y) == (end_x, end_y):
        return np.asarray(((start_x, start_y),), dtype=np.int32)

    delta_x = float(end_x - start_x)
    delta_y = float(end_y - start_y)
    length = hypot(delta_x, delta_y)
    unit_x, unit_y = delta_x / length, delta_y / length
    normal_x, normal_y = -unit_y, unit_x
    sample_count = min(1024, max(2, int(np.ceil(length)) + 1))
    along = np.linspace(0.0, length, sample_count, dtype=np.float32)
    padding = float(options.search_radius_px)
    offset_count = min(129, int(options.search_radius_px) * 2 + 1)
    offsets = np.linspace(-padding, padding, offset_count, dtype=np.float32)
    map_x = (
        float(start_x)
        + along[None, :] * unit_x
        + offsets[:, None] * normal_x
    ).astype(np.float32, copy=False)
    map_y = (
        float(start_y)
        + along[None, :] * unit_y
        + offsets[:, None] * normal_y
    ).astype(np.float32, copy=False)
    interpolation = (
        cv2.INTER_NEAREST
        if options.edge_source == "traces"
        else cv2.INTER_LINEAR
    )
    edge_strip = cv2.remap(
        edge_values,
        map_x,
        map_y,
        interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    edge_strip = _edge_evidence(edge_strip, options.edge_source)
    attraction = float(options.edge_attraction)
    row_count, column_count = edge_strip.shape
    centre_row = int(np.argmin(np.abs(offsets)))
    locked_component = _anchored_support_component(
        edge_strip, centre_row, options.edge_source
    )
    straightness = np.abs(offsets[:, None]) / max(1.0, padding)
    score = attraction * edge_strip - (1.0 - attraction) * 0.32 * straightness
    if locked_component is not None:
        # A component touching both anchors is stronger topological evidence
        # than an unrelated parallel ridge, even if that ridge is brighter.
        score += np.where(locked_component, 0.72, -0.72).astype(np.float32)

    hue_strip = None
    if hue_values is not None and options.tangent_mode != "off":
        hue_strip = cv2.remap(
            hue_values,
            map_x,
            map_y,
            cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        ).astype(np.float32)
    # Banded Viterbi seam.  Keeping the transition band narrow makes long live
    # previews deterministic and interactive while prohibiting teleports.
    step_length = length / max(1, sample_count - 1)
    maximum_jump = max(1.0, step_length * 1.8)
    offset_step = max(1e-6, float(abs(offsets[1] - offsets[0])))
    maximum_shift = max(1, int(np.ceil(maximum_jump / offset_step)))
    previous_score = np.full(row_count, -np.inf, dtype=np.float32)
    previous_score[centre_row] = score[centre_row, 0]
    predecessors = np.full((column_count, row_count), -1, dtype=np.int16)
    smoothness = 0.07 + 0.045 * float(options.smoothing)
    rows = np.arange(row_count)
    for column in range(1, column_count):
        best = np.full(row_count, -np.inf, dtype=np.float32)
        best_predecessor = np.full(row_count, -1, dtype=np.int16)
        for shift in range(-maximum_shift, maximum_shift + 1):
            current_start = max(0, shift)
            current_stop = min(row_count, row_count + shift)
            if current_start >= current_stop:
                continue
            current_rows = rows[current_start:current_stop]
            prior_rows = current_rows - shift
            candidate = previous_score[prior_rows] - smoothness * abs(shift)
            if hue_strip is not None:
                lateral = float(shift) * offset_step
                move_x = unit_x * step_length + normal_x * lateral
                move_y = unit_y * step_length + normal_y * lateral
                move_angle = atan2(move_y, move_x)
                encoded = hue_strip[current_rows, column]
                if options.tangent_mode == "undirected":
                    tangent_angle = encoded * pi / 180.0
                    alignment = np.abs(np.cos(move_angle - tangent_angle))
                else:
                    tangent_angle = encoded * 2.0 * pi / 180.0
                    alignment = 0.5 + 0.5 * np.cos(move_angle - tangent_angle)
                candidate = candidate + (
                    float(options.tangent_weight)
                    * alignment
                    * edge_strip[current_rows, column]
                )
            improved = candidate > best[current_rows]
            if np.any(improved):
                selected_rows = current_rows[improved]
                best[selected_rows] = candidate[improved]
                best_predecessor[selected_rows] = prior_rows[improved]
        previous_score = best + score[:, column]
        predecessors[column] = best_predecessor

    selected_rows = np.empty(column_count, dtype=np.int32)
    selected_rows[-1] = centre_row
    if not np.isfinite(previous_score[centre_row]):
        selected_rows[-1] = int(np.argmax(previous_score))
    for column in range(column_count - 1, 0, -1):
        prior = int(predecessors[column, selected_rows[column]])
        selected_rows[column - 1] = centre_row if prior < 0 else prior
    selected_rows[0] = centre_row
    selected_rows[-1] = centre_row
    selected_offsets = offsets[selected_rows]

    path_x = float(start_x) + along * unit_x + selected_offsets * normal_x
    path_y = float(start_y) + along * unit_y + selected_offsets * normal_y
    path = np.column_stack((path_x, path_y))
    path[:, 0] = np.clip(path[:, 0], 0, width - 1)
    path[:, 1] = np.clip(path[:, 1], 0, height - 1)
    path[0] = (start_x, start_y)
    path[-1] = (end_x, end_y)
    return np.rint(path).astype(np.int32)


def _star_convex_edge_fill(
    edge_strength: np.ndarray,
    lab_image: np.ndarray,
    anchor_xy: tuple[int, int],
    radius: int,
    options: SmartFillOptions,
) -> np.ndarray:
    """Recover one closed, locally edge-supported seed around an interior click.

    Flood fill is ideal when a calculated barrier is closed.  Touching seeds
    often leave one- or two-pixel gaps, however, and neighbour-relative colour
    comparison can then legitimately walk into another seed of the same coat
    colour.  This bounded fallback follows a smooth star-convex edge contour;
    it is deliberately used only after the ordinary flood reaches the outer
    cursor-distance limit.
    """

    edge = _normalized_strength(edge_strength)
    lab = np.asarray(lab_image)
    height, width = edge.shape
    anchor_x, anchor_y = int(anchor_xy[0]), int(anchor_xy[1])
    maximum_radius = max(4, int(radius))
    expected_radius = max(4.0, maximum_radius / 2.4)
    angular_samples = int(
        np.clip(round(2.0 * pi * expected_radius * 0.65), 72, 160)
    )
    angles = np.linspace(
        0.0, 2.0 * pi, angular_samples, endpoint=False, dtype=np.float32
    )
    radial_values = np.arange(
        3, max(4, int(round(maximum_radius * 0.82))) + 1, dtype=np.float32
    )
    cosine = np.cos(angles)
    sine = np.sin(angles)
    map_x = anchor_x + cosine[:, None] * radial_values[None, :]
    map_y = anchor_y + sine[:, None] * radial_values[None, :]
    edge_samples = cv2.remap(
        edge,
        map_x,
        map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(np.float32)

    luminance = np.asarray(lab[:, :, 0], dtype=np.uint8)
    gradient_x = cv2.Sobel(luminance, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(luminance, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gradient_x, gradient_y)
    nonzero_gradient = gradient[gradient > 0.0]
    gradient_scale = (
        float(np.quantile(nonzero_gradient, 0.98))
        if nonzero_gradient.size
        else 1.0
    )
    gradient = np.clip(gradient / max(1.0, gradient_scale), 0.0, 1.0)
    gradient_samples = cv2.remap(
        gradient,
        map_x,
        map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(np.float32)

    # The radial pressure prior makes weak structural evidence increasingly
    # able to stop outward growth. At one half-life the effective edge barrier
    # threshold is exactly half its value at the cursor.
    radial_pressure = _smart_fill_extension_pressure(
        radial_values[None, :], options.falloff_half_life_px
    )
    threshold_floor = (
        float(options.edge_stop_threshold) * radial_pressure * 0.45
    )
    structural = np.clip(
        (edge_samples - threshold_floor)
        / np.maximum(0.05, 1.0 - threshold_floor),
        0.0,
        1.0,
    )
    boundary_evidence = np.maximum(structural, gradient_samples)
    angular_coverage = float(
        np.mean(np.max(boundary_evidence, axis=1) >= 0.12)
    )
    if angular_coverage < 0.25:
        # With no plausible enclosing edge, returning a shape-prior circle
        # would be less honest than retaining the visibly radius-limited flood.
        return np.zeros((height, width), dtype=bool)
    edge_weight = 1.35 * (1.0 - 0.35 * float(options.tunnel_strength))
    contrast_weight = 0.65 * max(
        0.20, 1.0 - float(options.colour_tolerance_lab) / 120.0
    )
    deviation = np.abs(radial_values[None, :] - expected_radius) / expected_radius
    unary = edge_weight * structural + contrast_weight * gradient_samples
    unary -= 0.18 * deviation
    unary -= 0.35 * np.exp(-radial_values[None, :] / 6.0)

    state_count = len(radial_values)
    state_indices = np.arange(state_count, dtype=np.int32)
    transition_steps = np.arange(-4, 5, dtype=np.int32)
    transition_cost = 0.14 * (1.0 - 0.35 * float(options.tunnel_strength))
    best_score = -np.inf
    best_path: np.ndarray | None = None
    # Trying a handful of plausible first-angle states makes the seam cyclic:
    # the final state must return close to the same radius.
    start_candidates = np.argsort(unary[0])[-10:]
    for start_state in start_candidates:
        previous = np.full(state_count, -np.inf, dtype=np.float32)
        previous[int(start_state)] = unary[0, int(start_state)]
        predecessors = np.full(
            (angular_samples, state_count), -1, dtype=np.int16
        )
        for angle_index in range(1, angular_samples):
            candidates = np.full(
                (len(transition_steps), state_count),
                -np.inf,
                dtype=np.float32,
            )
            for transition_index, step in enumerate(transition_steps):
                source_indices = state_indices - step
                valid = (source_indices >= 0) & (source_indices < state_count)
                targets = state_indices[valid]
                candidates[transition_index, targets] = (
                    previous[source_indices[valid]]
                    - transition_cost * abs(int(step))
                )
            chosen = np.argmax(candidates, axis=0)
            predecessors[angle_index] = (
                state_indices - transition_steps[chosen]
            ).astype(np.int16)
            previous = candidates[chosen, state_indices] + unary[angle_index]
        close_to_start = np.abs(state_indices - int(start_state)) <= 4
        end_state = int(
            np.argmax(np.where(close_to_start, previous, -np.inf))
        )
        total = float(
            previous[end_state]
            - transition_cost * abs(end_state - int(start_state))
        )
        if total <= best_score:
            continue
        path = np.empty(angular_samples, dtype=np.int32)
        path[-1] = end_state
        for angle_index in range(angular_samples - 1, 0, -1):
            predecessor = int(predecessors[angle_index, path[angle_index]])
            path[angle_index - 1] = (
                int(start_state) if predecessor < 0 else predecessor
            )
        best_score = total
        best_path = path

    mask = np.zeros((height, width), dtype=np.uint8)
    if best_path is None:
        return mask.astype(bool)
    selected_radii = radial_values[best_path]
    polygon = np.rint(
        np.column_stack(
            (
                anchor_x + cosine * selected_radii,
                anchor_y + sine * selected_radii,
            )
        )
    ).astype(np.int32)
    polygon[:, 0] = np.clip(polygon[:, 0], 0, width - 1)
    polygon[:, 1] = np.clip(polygon[:, 1], 0, height - 1)
    cv2.fillPoly(mask, [polygon], 1)
    return mask.astype(bool)


def smart_fill_region(
    labels: np.ndarray,
    corrected_bgr: np.ndarray,
    edge_strength: np.ndarray,
    click_xy: tuple[float, float],
    instance_id: int,
    options: SmartFillOptions = SmartFillOptions(),
    *,
    corrected_lab: np.ndarray | None = None,
    allowed_mask: np.ndarray | None = None,
    allowed_origin_xy: tuple[int, int] = (0, 0),
    extension_pressure_mask: np.ndarray | None = None,
    extension_pressure_origin_xy: tuple[int, int] = (0, 0),
) -> SmartFillRegion:
    """Return a fast locally adaptive fill region suitable for live preview.

    OpenCV's floating-range flood fill compares each candidate with an already
    accepted neighbour. A separate hard click-origin range prevents those
    individually acceptable steps from accumulating into unlimited colour
    drift. Existing marks are optional: an empty instance starts directly at
    the clicked pixel. ``allowed_mask`` optionally
    supplies a compact full-image-coordinate envelope. It is used by Shape fill
    to reuse this exact growth method without allocating a full-resolution mask.
    ``extension_pressure_mask`` can replace the ordinary cursor-radial pressure
    with a compact caller-authored pressure field; Shape fill uses this for its
    signed ellipse-relative decay, including the configured inward start.
    """

    source_labels = np.asarray(labels)
    image = np.asarray(corrected_bgr)
    edge_values = np.asarray(edge_strength)
    if source_labels.ndim != 2 or image.shape[:2] != source_labels.shape or edge_values.shape != source_labels.shape:
        raise ValueError("Labels, image, and edge strength must share one image shape.")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Smart fill requires a three-channel corrected BGR image.")
    if not 1 <= int(instance_id) <= np.iinfo(np.uint16).max:
        raise ValueError("Instance ID must fit in an unsigned 16-bit label map.")
    compact_allowed = None
    if allowed_mask is not None:
        compact_allowed = np.asarray(allowed_mask, dtype=bool)
        if compact_allowed.ndim != 2:
            raise ValueError("Smart-fill allowed mask must be two-dimensional.")
    compact_pressure = None
    if extension_pressure_mask is not None:
        compact_pressure = np.asarray(
            extension_pressure_mask, dtype=np.float32
        )
        if compact_pressure.ndim != 2:
            raise ValueError(
                "Smart-fill extension pressure mask must be two-dimensional."
            )
        if not np.all(np.isfinite(compact_pressure)):
            raise ValueError("Smart-fill extension pressure must be finite.")
        if np.any(compact_pressure < 0.0) or np.any(compact_pressure > 1.0):
            raise ValueError(
                "Smart-fill extension pressure must lie between zero and one."
            )
    height, width = source_labels.shape
    click_x = int(np.clip(round(click_xy[0]), 0, width - 1))
    click_y = int(np.clip(round(click_xy[1]), 0, height - 1))
    # This is deliberately a distance from the click, not an instance width.
    # The circular guard therefore permits at most twice this value from one
    # side of a newly filled region to the other when the cursor is central.
    radius = int(options.maximum_distance_from_cursor_px)
    anchor_x, anchor_y = click_x, click_y

    x0, x1 = max(0, anchor_x - radius), min(width, anchor_x + radius + 1)
    y0, y1 = max(0, anchor_y - radius), min(height, anchor_y + radius + 1)
    roi_labels = source_labels[y0:y1, x0:x1]
    roi_edge = _normalized_strength(edge_values[y0:y1, x0:x1])
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
    local_anchor = (anchor_x - x0, anchor_y - y0)
    allowed_roi = np.ones(roi_labels.shape, dtype=bool)
    if compact_allowed is not None:
        allowed_roi.fill(False)
        allowed_x0, allowed_y0 = map(int, allowed_origin_xy)
        allowed_x1 = allowed_x0 + compact_allowed.shape[1]
        allowed_y1 = allowed_y0 + compact_allowed.shape[0]
        overlap_x0 = max(x0, allowed_x0)
        overlap_y0 = max(y0, allowed_y0)
        overlap_x1 = min(x1, allowed_x1)
        overlap_y1 = min(y1, allowed_y1)
        if overlap_x0 < overlap_x1 and overlap_y0 < overlap_y1:
            allowed_roi[
                overlap_y0 - y0 : overlap_y1 - y0,
                overlap_x0 - x0 : overlap_x1 - x0,
            ] = compact_allowed[
                overlap_y0 - allowed_y0 : overlap_y1 - allowed_y0,
                overlap_x0 - allowed_x0 : overlap_x1 - allowed_x0,
            ]
        if not allowed_roi[local_anchor[1], local_anchor[0]]:
            return SmartFillRegion(
                x0, y0, np.zeros(roi_labels.shape, dtype=bool), 0
            )
    yy, xx = np.ogrid[: roi_labels.shape[0], : roi_labels.shape[1]]
    radial_distance_squared = (
        (xx - local_anchor[0]) ** 2 + (yy - local_anchor[1]) ** 2
    )
    radial_distance = np.sqrt(radial_distance_squared.astype(np.float32))
    outside_radius = radial_distance_squared > radius * radius
    other_instance = (roi_labels != 0) & (roi_labels != int(instance_id))
    if compact_pressure is None:
        radial_pressure = _smart_fill_extension_pressure(
            radial_distance, options.falloff_half_life_px
        )
    else:
        radial_pressure = np.zeros(roi_labels.shape, dtype=np.float32)
        pressure_x0, pressure_y0 = map(int, extension_pressure_origin_xy)
        pressure_x1 = pressure_x0 + compact_pressure.shape[1]
        pressure_y1 = pressure_y0 + compact_pressure.shape[0]
        overlap_x0 = max(x0, pressure_x0)
        overlap_y0 = max(y0, pressure_y0)
        overlap_x1 = min(x1, pressure_x1)
        overlap_y1 = min(y1, pressure_y1)
        if overlap_x0 < overlap_x1 and overlap_y0 < overlap_y1:
            radial_pressure[
                overlap_y0 - y0 : overlap_y1 - y0,
                overlap_x0 - x0 : overlap_x1 - x0,
            ] = compact_pressure[
                overlap_y0 - pressure_y0 : overlap_y1 - pressure_y0,
                overlap_x0 - pressure_x0 : overlap_x1 - pressure_x0,
            ]
    edge_limit = float(options.edge_stop_threshold) * radial_pressure
    # Empty edge evidence remains passable even at very small pressure; for
    # continuous evidence the decaying threshold makes weaker edges sufficient
    # to stop growth farther from the cursor. Binary ridges remain hard edges.
    edge_barrier = (roi_edge > 0.0) & (roi_edge >= edge_limit)
    edge_barrier &= roi_labels != int(instance_id)
    gap_radius = int(options.edge_gap_sealing_px)
    if gap_radius > 0 and np.any(edge_barrier):
        # Treat short holes in an otherwise coherent ridge as pressure leaks in
        # a thin wall. Closing seals only gaps bounded by nearby barrier support;
        # genuinely open arcs wider than the configured radius stay traversable.
        kernel_size = gap_radius * 2 + 1
        # A rectangular closing element is intentional here. An ellipse erodes
        # a one-pixel oriented trace back open after dilation, so it fails at
        # exactly the nearly-complete thin ridges this control is meant to seal.
        gap_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (kernel_size, kernel_size)
        )
        edge_barrier = cv2.morphologyEx(
            edge_barrier.astype(np.uint8), cv2.MORPH_CLOSE, gap_kernel
        ).astype(bool)
        edge_barrier &= roi_labels != int(instance_id)
    # OpenCV's fast floating-range fill uses one fixed neighbour tolerance.
    # Add an exact, vectorized radial prior over the selected connectivity. For
    # each candidate, find the smallest normalized Lab step to any touching
    # neighbour that is strictly closer to the cursor. A candidate is blocked
    # when even that easiest inward step exceeds 2**(-distance / half_life).
    # The ordinary floating-range check below remains the connected-growth
    # engine and handles local continuity in every direction.
    lab_int = roi_lab.astype(np.int16)
    anchor_lab = lab_int[local_anchor[1], local_anchor[0]]
    click_colour_distance = np.abs(
        lab_int[:, :, 0] - anchor_lab[0]
    ).astype(np.float32)
    np.maximum(
        click_colour_distance,
        np.abs(lab_int[:, :, 1] - anchor_lab[1]) / 0.72,
        out=click_colour_distance,
    )
    np.maximum(
        click_colour_distance,
        np.abs(lab_int[:, :, 2] - anchor_lab[2]) / 0.72,
        out=click_colour_distance,
    )
    click_colour_barrier = (
        click_colour_distance > float(options.click_colour_tolerance_lab)
    )
    click_colour_barrier[local_anchor[1], local_anchor[0]] = False
    # Existing parts of the active instance are never erased by a fill. They
    # may differ from the click colour, but newly added pixels must still pass
    # the click-origin range below.
    click_colour_barrier &= roi_labels != int(instance_id)
    minimum_inward_step = np.full(roi_labels.shape, np.inf, dtype=np.float32)
    neighbour_offsets = [(-1, 0), (0, -1), (0, 1), (1, 0)]
    if options.connectivity == 8:
        neighbour_offsets.extend(((-1, -1), (-1, 1), (1, -1), (1, 1)))
    roi_height, roi_width = roi_labels.shape
    base_tolerance = float(options.colour_tolerance_lab)
    for delta_y, delta_x in neighbour_offsets:
        candidate_y0 = max(0, -delta_y)
        candidate_y1 = min(roi_height, roi_height - delta_y)
        candidate_x0 = max(0, -delta_x)
        candidate_x1 = min(roi_width, roi_width - delta_x)
        candidate_slice = (
            slice(candidate_y0, candidate_y1),
            slice(candidate_x0, candidate_x1),
        )
        neighbour_slice = (
            slice(candidate_y0 + delta_y, candidate_y1 + delta_y),
            slice(candidate_x0 + delta_x, candidate_x1 + delta_x),
        )
        candidate_lab = lab_int[candidate_slice]
        neighbour_lab = lab_int[neighbour_slice]
        normalized_step = np.abs(
            candidate_lab[:, :, 0] - neighbour_lab[:, :, 0]
        ).astype(np.float32)
        np.maximum(
            normalized_step,
            np.abs(candidate_lab[:, :, 1] - neighbour_lab[:, :, 1]) / 0.72,
            out=normalized_step,
        )
        np.maximum(
            normalized_step,
            np.abs(candidate_lab[:, :, 2] - neighbour_lab[:, :, 2]) / 0.72,
            out=normalized_step,
        )
        normalized_step /= base_tolerance
        inward = (
            radial_distance_squared[neighbour_slice]
            < radial_distance_squared[candidate_slice]
        )
        normalized_step[~inward] = np.inf
        np.minimum(
            minimum_inward_step[candidate_slice],
            normalized_step,
            out=minimum_inward_step[candidate_slice],
        )
    radial_colour_barrier = minimum_inward_step > radial_pressure
    radial_colour_barrier[local_anchor[1], local_anchor[0]] = False
    radial_colour_barrier &= roi_labels != int(instance_id)
    # Tunnelling opens one bounded passage through the nearest weak barrier.
    # The former implementation raised the threshold everywhere, weakening
    # every seed boundary in the ROI and also changing the colour tolerance.
    if options.tunnel_strength > 0.0 and np.any(edge_barrier):
        weak_limit = np.minimum(
            1.0, edge_limit + 0.32 * float(options.tunnel_strength)
        )
        weak = edge_barrier & (roi_edge <= weak_limit)
        weak_y, weak_x = np.nonzero(weak)
        if len(weak_y):
            weak_distance = (
                (weak_x - local_anchor[0]) ** 2
                + (weak_y - local_anchor[1]) ** 2
            )
            nearest = int(np.argmin(weak_distance))
            passage_radius = max(
                1, int(round(1.0 + 3.0 * float(options.tunnel_strength)))
            )
            passage = (
                (xx - int(weak_x[nearest])) ** 2
                + (yy - int(weak_y[nearest])) ** 2
                <= passage_radius * passage_radius
            )
            edge_barrier &= ~(weak & passage)
    blocked = (
        outside_radius
        | ~allowed_roi
        | other_instance
        | edge_barrier
        | radial_colour_barrier
        | click_colour_barrier
    )
    blocked[local_anchor[1], local_anchor[0]] = False
    flood_mask = np.zeros(
        (roi_labels.shape[0] + 2, roi_labels.shape[1] + 2), dtype=np.uint8
    )
    flood_mask[1:-1, 1:-1][blocked] = 1
    tolerance = float(options.colour_tolerance_lab)
    differences = (tolerance, tolerance * 0.72, tolerance * 0.72)
    flags = (
        int(options.connectivity)
        | cv2.FLOODFILL_MASK_ONLY
        | (255 << 8)
    )
    cv2.floodFill(
        roi_lab,
        flood_mask,
        local_anchor,
        (0, 0, 0),
        loDiff=differences,
        upDiff=differences,
        flags=flags,
    )
    accepted = flood_mask[1:-1, 1:-1] == 255
    # Include the first barrier pixel touching the accepted interior.  With a
    # thinned-ridge or trace source the visible annotation now meets that exact
    # one-pixel centreline, while the flood itself still cannot cross it.
    frontier = cv2.dilate(
        accepted.astype(np.uint8),
        np.ones((3, 3), dtype=np.uint8),
        iterations=1,
    ).astype(bool)
    accepted |= (
        frontier
        & edge_barrier
        & ~other_instance
        & ~click_colour_barrier
    )
    roi_is_unclipped = (
        x0 > 0 and y0 > 0 and x1 < width and y1 < height
    )
    outer_ring = (
        radial_distance_squared >= (0.92 * radius) ** 2
    ) & ~outside_radius
    outer_reached = (
        roi_is_unclipped
        and np.any(outer_ring)
        and float(np.mean(accepted[outer_ring])) > 0.12
    )
    if outer_reached and compact_pressure is None:
        recovered = _star_convex_edge_fill(
            roi_edge,
            roi_lab,
            local_anchor,
            radius,
            options,
        )
        if np.any(recovered):
            recovered &= ~click_colour_barrier
            accepted = recovered
            accepted |= roi_labels == int(instance_id)
            accepted &= ~other_instance
    accepted &= allowed_roi
    accepted |= roi_labels == int(instance_id)
    accepted &= ~click_colour_barrier | (roi_labels == int(instance_id))
    if compact_allowed is not None:
        accepted &= allowed_roi

    existing_active = roi_labels == int(instance_id)

    def retain_click_connected_additions(values: np.ndarray) -> np.ndarray:
        """Preserve old marks but reject detached recovery/limit fragments."""

        _count, components = cv2.connectedComponents(
            np.uint8(values), connectivity=int(options.connectivity)
        )
        click_component = int(components[local_anchor[1], local_anchor[0]])
        connected = (
            components == click_component
            if click_component > 0
            else np.zeros_like(values, dtype=bool)
        )
        return existing_active | (connected & ~existing_active)

    # Star-convex recovery and click-origin clipping are not themselves growth
    # operations. Reassert the defining Smart-fill invariant afterwards so a
    # narrow detached trail or island cannot survive as if tunnelling were on.
    accepted = retain_click_connected_additions(accepted)
    added_mask = accepted & (roi_labels != int(instance_id))
    added = int(np.count_nonzero(added_mask))
    if added > options.maximum_added_pixels:
        added_y, added_x = np.nonzero(added_mask)
        squared_distance = (
            (added_x - local_anchor[0]) ** 2
            + (added_y - local_anchor[1]) ** 2
        )
        keep = np.argpartition(
            squared_distance, options.maximum_added_pixels - 1
        )[: options.maximum_added_pixels]
        limited = roi_labels == int(instance_id)
        limited[added_y[keep], added_x[keep]] = True
        accepted = retain_click_connected_additions(limited)
        added = int(np.count_nonzero(accepted & ~existing_active))
    return SmartFillRegion(x0, y0, accepted, added)


def smart_fill_instance(
    labels: np.ndarray,
    corrected_bgr: np.ndarray,
    edge_strength: np.ndarray,
    click_xy: tuple[float, float],
    instance_id: int,
    options: SmartFillOptions = SmartFillOptions(),
    *,
    corrected_lab: np.ndarray | None = None,
) -> tuple[np.ndarray, int]:
    """Apply the locally adaptive fill preview to a copied label map."""

    source_labels = np.asarray(labels)
    region = smart_fill_region(
        source_labels,
        corrected_bgr,
        edge_strength,
        click_xy,
        instance_id,
        options,
        corrected_lab=corrected_lab,
    )
    result = source_labels.astype(np.uint16, copy=True)
    height, width = region.mask.shape
    roi = result[region.y : region.y + height, region.x : region.x + width]
    writable = region.mask & ((roi == 0) | (roi == int(instance_id)))
    roi[writable] = np.uint16(instance_id)
    return result, region.added_count
