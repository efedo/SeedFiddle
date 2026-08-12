"""Image-aware tools used while annotating individual seeds.

These functions deliberately have no Qt dependency.  The desktop view supplies
the already-calculated edge rasters and corrected BGR image, then applies the
returned path, polygon, or label map as one undoable annotation draft change.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, hypot, pi

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class EdgeTraceOptions:
    """Controls for a magnetic edge path between two user points."""

    search_radius_px: int = 18
    edge_attraction: float = 0.85
    tangent_mode: str = "undirected"
    tangent_weight: float = 0.45
    smoothing: int = 2

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


@dataclass(frozen=True, slots=True)
class ShapeSnapOptions:
    """Controls for fitting an edge-supported circle or ellipse."""

    shape: str = "ellipse"
    edge_search_radius_px: int = 14
    centre_search_radius_px: int = 5
    angular_samples: int = 96
    rotation_degrees: float = 0.0
    tangent_mode: str = "undirected"
    tangent_weight: float = 0.35

    def __post_init__(self) -> None:
        if self.shape not in {"circle", "ellipse"}:
            raise ValueError("Shape must be circle or ellipse.")
        if not 1 <= self.edge_search_radius_px <= 200:
            raise ValueError("Shape edge search must be between 1 and 200 pixels.")
        if not 0 <= self.centre_search_radius_px <= 100:
            raise ValueError("Shape centre search must be between 0 and 100 pixels.")
        if not 24 <= self.angular_samples <= 360:
            raise ValueError("Shape samples must be between 24 and 360.")
        if not -180.0 <= self.rotation_degrees <= 180.0:
            raise ValueError("Shape rotation must be between -180 and 180 degrees.")
        if self.tangent_mode not in {"off", "undirected", "directed"}:
            raise ValueError("Tangent mode must be off, undirected, or directed.")
        if not 0.0 <= self.tangent_weight <= 1.0:
            raise ValueError("Tangent weight must be between zero and one.")


@dataclass(frozen=True, slots=True)
class SmartFillOptions:
    """Controls for locally adaptive, edge-stopped instance growth."""

    colour_tolerance_lab: float = 18.0
    edge_stop_threshold: float = 0.58
    tunnel_strength: float = 0.0
    maximum_radius_px: int = 160
    maximum_added_pixels: int = 150_000
    connectivity: int = 8

    def __post_init__(self) -> None:
        if not 1.0 <= self.colour_tolerance_lab <= 100.0:
            raise ValueError("Lab tolerance must be between 1 and 100.")
        if not 0.0 <= self.edge_stop_threshold <= 1.0:
            raise ValueError("Edge stop threshold must be between zero and one.")
        if not 0.0 <= self.tunnel_strength <= 1.0:
            raise ValueError("Tunnel strength must be between zero and one.")
        if not 4 <= self.maximum_radius_px <= 4096:
            raise ValueError("Maximum fill radius must be between 4 and 4096 pixels.")
        if not 1 <= self.maximum_added_pixels <= 10_000_000:
            raise ValueError("Maximum added pixels must be positive.")
        if self.connectivity not in {4, 8}:
            raise ValueError("Connectivity must be four or eight.")


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
    score = roi - 0.18 * distance / max(1.0, float(radius))
    score[~inside] = -np.inf
    snapped_y, snapped_x = np.unravel_index(int(np.argmax(score)), score.shape)
    if float(roi[snapped_y, snapped_x]) <= 0.0:
        return x, y
    return int(snapped_x + x0), int(snapped_y + y0)


def trace_edge_path(
    start_xy: tuple[float, float],
    end_xy: tuple[float, float],
    edge_strength: np.ndarray,
    options: EdgeTraceOptions = EdgeTraceOptions(),
    tangent_hue: np.ndarray | None = None,
) -> np.ndarray:
    """Return a fast magnetic path sampled across a bounded line corridor.

    OpenCV samples the complete perpendicular search strip in native code. A
    smoothed maximum-support track then supplies the preview path. This avoids
    the former Python heap search whose latency grew to seconds for long paths
    or wide search radii, while retaining edge and tangent evidence.
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
    edge_strip = cv2.remap(
        edge_values,
        map_x,
        map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    edge_strip = _normalized_strength(edge_strip)
    attraction = float(options.edge_attraction)
    score = attraction * edge_strip

    if hue_values is not None and options.tangent_mode != "off":
        hue_strip = cv2.remap(
            hue_values,
            map_x,
            map_y,
            cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        ).astype(np.float32)
        path_angle = atan2(delta_y, delta_x)
        if options.tangent_mode == "undirected":
            tangent_angle = hue_strip * pi / 180.0
            alignment = np.abs(np.cos(path_angle - tangent_angle))
        else:
            tangent_angle = hue_strip * 2.0 * pi / 180.0
            alignment = 0.5 + 0.5 * np.cos(path_angle - tangent_angle)
        score += float(options.tangent_weight) * alignment

    score -= 0.10 * np.abs(offsets[:, None]) / max(1.0, padding)
    selected_offsets = offsets[np.argmax(score, axis=0)]
    sigma = 0.65 + 0.55 * float(options.smoothing)
    selected_offsets = cv2.GaussianBlur(
        selected_offsets[None, :],
        (0, 0),
        sigmaX=sigma,
        borderType=cv2.BORDER_REPLICATE,
    )[0]

    # Anchor both clicks and limit lateral jumps so adjacent strong coat edges
    # cannot make the preview teleport between unrelated boundaries.
    step_length = length / max(1, sample_count - 1)
    maximum_jump = max(1.0, step_length * 1.8)
    selected_offsets[0] = 0.0
    for index in range(1, sample_count):
        selected_offsets[index] = np.clip(
            selected_offsets[index],
            selected_offsets[index - 1] - maximum_jump,
            selected_offsets[index - 1] + maximum_jump,
        )
    selected_offsets[-1] = 0.0
    for index in range(sample_count - 2, -1, -1):
        selected_offsets[index] = np.clip(
            selected_offsets[index],
            selected_offsets[index + 1] - maximum_jump,
            selected_offsets[index + 1] + maximum_jump,
        )

    path_x = float(start_x) + along * unit_x + selected_offsets * normal_x
    path_y = float(start_y) + along * unit_y + selected_offsets * normal_y
    path = np.column_stack((path_x, path_y))
    path[:, 0] = np.clip(path[:, 0], 0, width - 1)
    path[:, 1] = np.clip(path[:, 1], 0, height - 1)
    path[0] = (start_x, start_y)
    path[-1] = (end_x, end_y)
    return np.rint(path).astype(np.int32)


def snap_shape_polygon(
    centre_xy: tuple[float, float],
    edge_xy: tuple[float, float],
    edge_strength: np.ndarray,
    options: ShapeSnapOptions = ShapeSnapOptions(),
    tangent_hue: np.ndarray | None = None,
    *,
    fallback_radius_px: float = 30.0,
) -> np.ndarray:
    """Fit the closest locally edge-supported circle or ellipse."""

    edge = np.asarray(edge_strength)
    if edge.ndim != 2:
        raise ValueError("Edge strength must be a two-dimensional raster.")
    edge_scale = 255.0 if np.issubdtype(edge.dtype, np.integer) else 1.0
    height, width = edge.shape
    hue = None if tangent_hue is None else np.asarray(tangent_hue)
    if hue is not None and hue.shape != edge.shape:
        raise ValueError("Tangent hue and edge strength must have the same shape.")
    cx, cy = float(centre_xy[0]), float(centre_xy[1])
    drag_x, drag_y = float(edge_xy[0]) - cx, float(edge_xy[1]) - cy
    using_fallback = hypot(drag_x, drag_y) < 3.0
    if using_fallback:
        drag_x = max(3.0, float(fallback_radius_px))
        drag_y = (
            drag_x
            if options.shape == "circle"
            else max(3.0, drag_x * 0.72)
        )
    if options.shape == "circle":
        radius_x = radius_y = max(3.0, hypot(drag_x, drag_y))
    else:
        radius_x = max(3.0, abs(drag_x))
        radius_y = max(3.0, abs(drag_y))
        if radius_x <= 3.0 or radius_y <= 3.0:
            radius_x = max(radius_x, float(fallback_radius_px))
            radius_y = max(radius_y, float(fallback_radius_px) * 0.72)
    rotation = float(options.rotation_degrees) * pi / 180.0
    angles = np.linspace(0.0, 2.0 * pi, options.angular_samples, endpoint=False)
    cosine, sine = np.cos(angles), np.sin(angles)
    cos_r, sin_r = cos(rotation), np.sin(rotation)
    centre_radius = int(options.centre_search_radius_px)
    centre_offsets = sorted({-centre_radius, 0, centre_radius})
    edge_radius = float(options.edge_search_radius_px)
    radial_offsets = np.linspace(-edge_radius, edge_radius, 7)
    best_score = -np.inf
    best_polygon: np.ndarray | None = None
    for offset_y in centre_offsets:
        for offset_x in centre_offsets:
            candidate_cx, candidate_cy = cx + offset_x, cy + offset_y
            for radial_offset in radial_offsets:
                candidate_rx = max(3.0, radius_x + radial_offset)
                candidate_ry = max(3.0, radius_y + radial_offset)
                local_x = candidate_rx * cosine
                local_y = candidate_ry * sine
                xs = candidate_cx + local_x * cos_r - local_y * sin_r
                ys = candidate_cy + local_x * sin_r + local_y * cos_r
                ix = np.rint(xs).astype(np.int32)
                iy = np.rint(ys).astype(np.int32)
                valid = (ix >= 0) & (ix < width) & (iy >= 0) & (iy < height)
                if np.count_nonzero(valid) < options.angular_samples * 0.75:
                    continue
                support = float(edge[iy[valid], ix[valid]].mean()) / edge_scale
                if hue is not None and options.tangent_mode != "off":
                    dx_dt = -candidate_rx * sine
                    dy_dt = candidate_ry * cosine
                    tangent_angles = np.arctan2(
                        dx_dt * sin_r + dy_dt * cos_r,
                        dx_dt * cos_r - dy_dt * sin_r,
                    )
                    encoded = hue[iy[valid], ix[valid]].astype(np.float32)
                    if options.tangent_mode == "undirected":
                        encoded_angles = encoded * pi / 180.0
                        alignment = np.abs(np.cos(tangent_angles[valid] - encoded_angles))
                    else:
                        encoded_angles = encoded * 2.0 * pi / 180.0
                        alignment = 0.5 + 0.5 * np.cos(
                            tangent_angles[valid] - encoded_angles
                        )
                    support = (
                        support * (1.0 - options.tangent_weight)
                        + float(np.mean(alignment)) * options.tangent_weight
                    )
                displacement_penalty = (
                    hypot(offset_x, offset_y) / max(1.0, centre_radius) * 0.04
                    if centre_radius
                    else 0.0
                )
                radius_penalty = abs(radial_offset) / max(1.0, edge_radius) * 0.04
                score = support - displacement_penalty - radius_penalty
                if score > best_score:
                    best_score = score
                    best_polygon = np.column_stack((xs, ys))
    if best_polygon is None:
        local_x = radius_x * cosine
        local_y = radius_y * sine
        best_polygon = np.column_stack(
            (
                cx + local_x * cos_r - local_y * sin_r,
                cy + local_x * sin_r + local_y * cos_r,
            )
        )
    return np.rint(best_polygon).astype(np.int32)


def smart_fill_region(
    labels: np.ndarray,
    corrected_bgr: np.ndarray,
    edge_strength: np.ndarray,
    click_xy: tuple[float, float],
    instance_id: int,
    options: SmartFillOptions = SmartFillOptions(),
    *,
    corrected_lab: np.ndarray | None = None,
) -> SmartFillRegion:
    """Return a fast locally adaptive fill region suitable for live preview.

    OpenCV's floating-range flood fill compares each candidate with an already
    accepted neighbour rather than the original click colour. That preserves
    gradual traversal across patterned coats while running in native code fast
    enough for a debounced hover preview. Existing marks are optional: an empty
    instance starts directly at the clicked pixel.
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
    height, width = source_labels.shape
    click_x = int(np.clip(round(click_xy[0]), 0, width - 1))
    click_y = int(np.clip(round(click_xy[1]), 0, height - 1))
    radius = int(options.maximum_radius_px)
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
    yy, xx = np.ogrid[: roi_labels.shape[0], : roi_labels.shape[1]]
    outside_radius = (
        (xx - local_anchor[0]) ** 2 + (yy - local_anchor[1]) ** 2
        > radius * radius
    )
    other_instance = (roi_labels != 0) & (roi_labels != int(instance_id))
    edge_limit = min(
        1.0,
        float(options.edge_stop_threshold) + 0.32 * options.tunnel_strength,
    )
    edge_barrier = roi_edge > edge_limit
    edge_barrier &= roi_labels != int(instance_id)
    blocked = outside_radius | other_instance | edge_barrier
    blocked[local_anchor[1], local_anchor[0]] = False
    flood_mask = np.zeros(
        (roi_labels.shape[0] + 2, roi_labels.shape[1] + 2), dtype=np.uint8
    )
    flood_mask[1:-1, 1:-1][blocked] = 1
    tolerance = float(options.colour_tolerance_lab) * (
        1.0 + 0.85 * options.tunnel_strength
    )
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
    accepted |= roi_labels == int(instance_id)
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
        accepted = limited
        added = int(options.maximum_added_pixels)
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
