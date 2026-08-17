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
    boundary_edge_weight: float = 0.56
    boundary_sensor_weight: float = 0.24
    boundary_ridge_weight: float = 0.12
    boundary_shadow_weight: float = 0.08
    boundary_reference_weight: float = 0.35
    boundary_nonedge_discount: float = 0.85
    reference_texture_weight: float = 0.35
    strong_boundary_quantile: float = 0.67
    centre_ring_inner_fraction: float = 0.27
    centre_ring_outer_fraction: float = 0.58
    centre_material_weight: float = 0.30
    centre_distance_weight: float = 0.40
    centre_ring_weight: float = 0.30
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
        weights = (
            self.boundary_edge_weight,
            self.boundary_sensor_weight,
            self.boundary_ridge_weight,
            self.boundary_shadow_weight,
        )
        if any(value < 0.0 for value in weights) or sum(weights) <= 0.0:
            raise ValueError("Boundary evidence weights must be non-negative and non-zero.")
        if not 0.0 <= self.boundary_reference_weight <= 1.0:
            raise ValueError("Reference-edge weight must be between zero and one.")
        if not 0.0 <= self.boundary_nonedge_discount <= 1.0:
            raise ValueError("Non-edge discount must be between zero and one.")
        if not 0.0 <= self.reference_texture_weight <= 1.0:
            raise ValueError("Reference texture weight must be between zero and one.")
        centre_weights = (
            self.centre_material_weight,
            self.centre_distance_weight,
            self.centre_ring_weight,
        )
        if any(value < 0.0 for value in centre_weights) or sum(centre_weights) <= 0.0:
            raise ValueError("Centre evidence weights must be non-negative and non-zero.")
        if not 0.0 < self.foreground_threshold_scale <= 2.0:
            raise ValueError("Foreground threshold scale must be positive.")
        if not 0.0 <= self.dish_margin_fraction <= 0.5:
            raise ValueError("Dish margin must be between 0 and 0.5 seed diameters.")
        if not 0.0 <= self.occupancy_hole_area_fraction <= 4.0:
            raise ValueError("Occupancy hole area must be between 0 and 4 seed areas.")
        if not 0.05 <= self.strong_boundary_quantile <= 0.95:
            raise ValueError("Strong-boundary quantile must be between 0.05 and 0.95.")
        if not 0.1 <= self.centre_minimum_separation_fraction <= 1.0:
            raise ValueError("Centre separation must be between 0.1 and 1.0 seed diameters.")
        if not 0.1 <= self.sparse_centre_minimum_separation_fraction <= 1.0:
            raise ValueError("Sparse centre separation must be between 0.1 and 1.0 seed diameters.")
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


def _ellipse_kernel(radius: float) -> np.ndarray:
    integer_radius = max(1, int(round(radius)))
    size = integer_radius * 2 + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _disk_average_kernel(inner_radius: float, outer_radius: float) -> np.ndarray:
    radius = max(2, int(round(outer_radius)))
    yy, xx = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    distance = np.hypot(xx, yy)
    selected = (distance >= max(0.0, inner_radius)) & (distance <= outer_radius)
    kernel = selected.astype(np.float32)
    return kernel / max(1.0, float(kernel.sum()))


def _renumber_labels(labels: np.ndarray) -> np.ndarray:
    identifiers = np.unique(labels)
    identifiers = identifiers[identifiers > 0]
    if not len(identifiers):
        return np.zeros(labels.shape, dtype=np.int32)
    lookup = np.zeros(int(identifiers[-1]) + 1, dtype=np.int32)
    lookup[identifiers] = np.arange(1, len(identifiers) + 1, dtype=np.int32)
    return lookup[np.asarray(labels, dtype=np.int32)]


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
    reference_surface_probability=None,
    sensor_noise,
    shadow_likelihood,
    seed_instance_annotations=None,
    settings: ProceduralInstanceSettings = ProceduralInstanceSettings(),
) -> ProceduralInstanceResult:
    """Separate visible seed instances with scale-aware markers and watershed."""

    source_height, source_width = tuple(int(value) for value in valid_mask.shape[-2:])
    scale = min(
        1.0,
        float(settings.working_maximum_dimension) / max(source_height, source_width),
    )
    width = max(16, int(round(source_width * scale)))
    height = max(16, int(round(source_height * scale)))
    size = (width, height)
    valid = _working_u8(valid_mask, size, cv2.INTER_NEAREST) > 0
    diameter = max(6.0, float(seed_diameter_px) * scale)

    foreground = _working_u8(foreground_probability, size, cv2.INTER_AREA) / 255.0
    foreground_noise = _working_u8(foreground_noise_probability, size, cv2.INTER_AREA) / 255.0
    background = _working_u8(background_probability, size, cv2.INTER_AREA) / 255.0
    refined_background = _working_u8(refined_background_probability, size, cv2.INTER_AREA) / 255.0
    edge = _working_u8(edge_magnitude, size, cv2.INTER_AREA) / 255.0
    ridges = _working_u8(edge_ridges, size, cv2.INTER_AREA) / 255.0
    sensor = _working_u8(sensor_noise, size, cv2.INTER_AREA) / 255.0
    shadow = _working_u8(shadow_likelihood, size, cv2.INTER_AREA) / 255.0
    physical_reference = (
        edge
        if physical_edge_probability is None
        else _working_u8(physical_edge_probability, size, cv2.INTER_AREA) / 255.0
    )
    non_edge_reference = (
        np.zeros_like(edge)
        if non_edge_probability is None
        else _working_u8(non_edge_probability, size, cv2.INTER_AREA) / 255.0
    )
    reference_surface = (
        None
        if reference_surface_probability is None
        else _working_u8(
            reference_surface_probability, size, cv2.INTER_AREA
        )
        / 255.0
    )

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

    boundary_weights = np.asarray(
        (
            settings.boundary_edge_weight,
            settings.boundary_sensor_weight,
            settings.boundary_ridge_weight,
            settings.boundary_shadow_weight,
        ),
        dtype=np.float32,
    )
    boundary_weights /= boundary_weights.sum()
    boundary = (
        boundary_weights[0] * edge
        + boundary_weights[1] * sensor
        + boundary_weights[2] * ridges
        + boundary_weights[3] * shadow
    )
    reference_boundary = physical_reference * (
        1.0 - settings.boundary_nonedge_discount * non_edge_reference
    )
    boundary = (
        (1.0 - settings.boundary_reference_weight) * boundary
        + settings.boundary_reference_weight * reference_boundary
    )
    boundary = cv2.GaussianBlur(boundary, (0, 0), sigmaX=max(0.45, diameter * 0.012))
    boundary *= valid
    boundary_scale = float(np.quantile(boundary[valid], 0.995)) if np.any(valid) else 1.0
    boundary = np.clip(boundary / max(1e-5, boundary_scale), 0.0, 1.0)

    occupied_boundary = boundary[occupancy]
    strong_threshold = (
        float(np.quantile(occupied_boundary, settings.strong_boundary_quantile))
        if occupied_boundary.size
        else 0.4
    )
    strong_threshold = float(np.clip(strong_threshold, 0.18, 0.62))
    traversable = occupancy & (boundary < strong_threshold)
    distance = cv2.distanceTransform(np.uint8(traversable), cv2.DIST_L2, 5)
    distance = cv2.GaussianBlur(distance, (0, 0), sigmaX=max(0.6, diameter * 0.045))
    distance_score = np.clip(distance / max(1.0, diameter * 0.38), 0.0, 1.0)

    ring_kernel = _disk_average_kernel(
        diameter * settings.centre_ring_inner_fraction,
        diameter * settings.centre_ring_outer_fraction,
    )
    inner_kernel = _disk_average_kernel(0.0, diameter * 0.22)
    ring_support = cv2.filter2D(boundary, cv2.CV_32F, ring_kernel)
    inner_edge = cv2.filter2D(boundary, cv2.CV_32F, inner_kernel)
    ring_score = np.clip((ring_support - 0.32 * inner_edge) / 0.35, 0.0, 1.0)
    material_core = cv2.GaussianBlur(
        occupancy_likelihood, (0, 0), sigmaX=max(0.8, diameter * 0.16)
    )
    material_scale = (
        float(np.quantile(material_core[interior_valid], 0.995))
        if np.any(interior_valid)
        else 1.0
    )
    material_core = np.clip(material_core / max(1e-5, material_scale), 0.0, 1.0)
    centre_weights = np.asarray(
        (
            settings.centre_material_weight,
            settings.centre_distance_weight,
            settings.centre_ring_weight,
        ),
        dtype=np.float32,
    )
    centre_weights /= max(1e-6, float(centre_weights.sum()))
    centre = (
        centre_weights[0] * material_core
        + centre_weights[1] * distance_score
        + centre_weights[2] * ring_score
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
    peak_y, peak_x = np.nonzero(peak_mask)
    peak_scores = centre[peak_y, peak_x]
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
        annotations = _resize(
            np.asarray(seed_instance_annotations, dtype=np.uint16),
            size,
            cv2.INTER_NEAREST,
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
    peak_x = np.concatenate((manual_x, peak_x[:automatic_count]))
    peak_y = np.concatenate((manual_y, peak_y[:automatic_count]))
    peak_scores = np.concatenate(
        (np.ones(len(manual_x), np.float32), peak_scores[:automatic_count])
    )

    markers = np.zeros((height, width), dtype=np.int32)
    markers[~occupancy] = 1
    if annotations is not None and len(annotation_ids):
        annotation_lookup = np.zeros(int(annotation_ids[-1]) + 1, np.int32)
        annotation_lookup[annotation_ids] = np.arange(
            2, len(annotation_ids) + 2, dtype=np.int32
        )
        selected_annotations = annotations > 0
        markers[selected_annotations] = annotation_lookup[annotations[selected_annotations]]
    for identifier, (x, y) in enumerate(zip(peak_x, peak_y, strict=True), start=2):
        cv2.circle(markers, (int(x), int(y)), max(1, int(round(diameter * 0.035))), identifier, -1)
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
