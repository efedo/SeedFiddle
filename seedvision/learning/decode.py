"""Topology decoders for multi-head U-Net and StarDist predictions."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from seedvision.learning.contracts import LearnedInstanceResult, ModelFamily
from seedvision.learning.targets import relabel_consecutive


def _single_numpy(values, *, sigmoid: bool = False) -> np.ndarray:
    try:
        import torch

        if torch.is_tensor(values):
            tensor = values.detach()
            if sigmoid:
                tensor = tensor.sigmoid()
            while tensor.ndim > 2:
                tensor = tensor[0]
            return tensor.float().cpu().numpy()
    except ImportError:
        pass
    array = np.asarray(values, dtype=np.float32)
    while array.ndim > 2:
        array = array[0]
    if sigmoid:
        array = 1.0 / (1.0 + np.exp(-np.clip(array, -30.0, 30.0)))
    return array


def _channels_numpy(values, *, sigmoid: bool = False) -> np.ndarray:
    try:
        import torch

        if torch.is_tensor(values):
            tensor = values.detach()
            if tensor.ndim == 4:
                tensor = tensor[0]
            if sigmoid:
                tensor = tensor.sigmoid()
            return tensor.float().cpu().numpy()
    except ImportError:
        pass
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 4:
        array = array[0]
    if sigmoid:
        array = 1.0 / (1.0 + np.exp(-np.clip(array, -30.0, 30.0)))
    return array


def _u8_probability(values: np.ndarray) -> np.ndarray:
    return np.uint8(np.rint(np.clip(values, 0.0, 1.0) * 255.0))


def _centres_and_confidence(labels: np.ndarray, score: np.ndarray):
    count = int(labels.max(initial=0))
    centres = np.zeros((count, 2), dtype=np.float32)
    confidences = np.zeros(count, dtype=np.float32)
    for identifier in range(1, count + 1):
        rows, columns = np.nonzero(labels == identifier)
        if not len(rows):
            continue
        centres[identifier - 1] = (float(columns.mean()), float(rows.mean()))
        confidences[identifier - 1] = float(np.mean(score[rows, columns]))
    return centres, np.clip(confidences, 0.0, 1.0)


@dataclass(frozen=True, slots=True)
class UNetWatershedSettings:
    interior_threshold: float = 0.50
    centre_threshold: float = 0.30
    centre_minimum_separation_px: float = 12.0
    minimum_instance_area_px: int = 24
    physical_boundary_weight: float = 0.72
    distance_topography_weight: float = 0.28
    pattern_boundary_discount: float = 0.80
    uncertainty_penalty: float = 0.20
    closing_radius_px: int = 1
    foreground_erosion_px: int = 0

    def __post_init__(self) -> None:
        for name in (
            "interior_threshold",
            "centre_threshold",
            "physical_boundary_weight",
            "distance_topography_weight",
            "pattern_boundary_discount",
            "uncertainty_penalty",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between zero and one.")
        if self.centre_minimum_separation_px < 1.0:
            raise ValueError("centre_minimum_separation_px must be at least one.")
        if self.minimum_instance_area_px < 1:
            raise ValueError("minimum_instance_area_px must be positive.")
        if self.closing_radius_px < 0 or self.foreground_erosion_px < 0:
            raise ValueError("Foreground morphology radii cannot be negative.")


def _automatic_markers(
    centre_score: np.ndarray,
    foreground: np.ndarray,
    *,
    threshold: float,
    separation_px: float,
) -> tuple[np.ndarray, np.ndarray]:
    radius = max(1, int(round(separation_px)))
    size = 2 * radius + 1
    local_maximum = cv2.dilate(
        centre_score.astype(np.float32),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)),
    )
    candidates = (
        (centre_score >= local_maximum - 1e-7)
        & (centre_score >= float(threshold))
        & foreground
    )
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        np.uint8(candidates), connectivity=8
    )
    points: list[tuple[float, float, float]] = []
    for identifier in range(1, count):
        x0 = stats[identifier, cv2.CC_STAT_LEFT]
        y0 = stats[identifier, cv2.CC_STAT_TOP]
        width = stats[identifier, cv2.CC_STAT_WIDTH]
        height = stats[identifier, cv2.CC_STAT_HEIGHT]
        region = labels[y0 : y0 + height, x0 : x0 + width] == identifier
        local = centre_score[y0 : y0 + height, x0 : x0 + width]
        index = int(np.argmax(np.where(region, local, -1.0)))
        local_y, local_x = np.unravel_index(index, local.shape)
        x, y = x0 + local_x, y0 + local_y
        points.append((float(x), float(y), float(centre_score[y, x])))
    points.sort(key=lambda item: item[2], reverse=True)
    retained: list[tuple[float, float, float]] = []
    minimum_squared = float(separation_px) ** 2
    for point in points:
        if all(
            (point[0] - other[0]) ** 2 + (point[1] - other[1]) ** 2
            >= minimum_squared
            for other in retained
        ):
            retained.append(point)
    xy = np.asarray([(item[0], item[1]) for item in retained], dtype=np.float32).reshape(-1, 2)
    scores = np.asarray([item[2] for item in retained], dtype=np.float32)
    return xy, scores


def decode_unet_watershed(
    outputs,
    *,
    valid_mask: np.ndarray | None = None,
    painted_instances: np.ndarray | None = None,
    settings: UNetWatershedSettings = UNetWatershedSettings(),
    checkpoint_id: str = "",
) -> LearnedInstanceResult:
    """Decode dense U-Net heads using a pattern-aware marker watershed."""

    interior = _single_numpy(outputs["interior_logits"], sigmoid=True)
    physical = _single_numpy(outputs["physical_boundary_logits"], sigmoid=True)
    pattern = _single_numpy(outputs["pattern_boundary_logits"], sigmoid=True)
    centre = _single_numpy(outputs["centre_logits"], sigmoid=True)
    distance = _single_numpy(outputs["distance"], sigmoid=False)
    uncertainty_logits = _channels_numpy(outputs["uncertainty_logits"])
    uncertainty = 1.0 / (
        1.0 + np.exp(-np.clip(np.mean(uncertainty_logits, axis=0), -30.0, 30.0))
    )
    shape = interior.shape
    valid = np.ones(shape, dtype=bool) if valid_mask is None else np.asarray(valid_mask) > 0
    if valid.shape != shape:
        raise ValueError("The valid mask must match the prediction dimensions.")
    foreground = (interior >= settings.interior_threshold) & valid
    if settings.closing_radius_px > 0:
        radius = int(settings.closing_radius_px)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)
        foreground = cv2.morphologyEx(
            np.uint8(foreground), cv2.MORPH_CLOSE, kernel
        ) > 0
        foreground &= valid
    if settings.foreground_erosion_px > 0:
        radius = int(settings.foreground_erosion_px)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)
        foreground = cv2.erode(np.uint8(foreground), kernel) > 0
        foreground &= valid

    centre_score = centre * np.sqrt(np.clip(distance, 0.0, 1.0)) * interior
    automatic_xy, automatic_scores = _automatic_markers(
        centre_score,
        foreground,
        threshold=settings.centre_threshold,
        separation_px=settings.centre_minimum_separation_px,
    )
    markers = np.zeros(shape, dtype=np.int32)
    markers[~foreground] = 1
    next_marker = 2
    manual_centres = np.empty((0, 2), dtype=np.float32)
    if painted_instances is not None:
        painted = relabel_consecutive(np.asarray(painted_instances))
        if painted.shape != shape:
            raise ValueError("Painted instances must match the prediction dimensions.")
        manual: list[tuple[float, float]] = []
        for identifier in range(1, int(painted.max(initial=0)) + 1):
            selected = painted == identifier
            rows, columns = np.nonzero(selected)
            if not len(rows):
                continue
            markers[selected] = next_marker
            foreground[selected] = True
            manual.append((float(columns.mean()), float(rows.mean())))
            next_marker += 1
        manual_centres = np.asarray(manual, dtype=np.float32).reshape(-1, 2)
    if len(manual_centres):
        minimum_squared = (settings.centre_minimum_separation_px * 0.75) ** 2
        keep = np.ones(len(automatic_xy), dtype=bool)
        for point in manual_centres:
            keep &= np.sum((automatic_xy - point) ** 2, axis=1) >= minimum_squared
        automatic_xy = automatic_xy[keep]
        automatic_scores = automatic_scores[keep]
    for x, y in automatic_xy:
        cv2.circle(markers, (int(round(x)), int(round(y))), 1, next_marker, -1)
        next_marker += 1
    if next_marker == 2:
        labels = np.zeros(shape, dtype=np.int32)
    else:
        effective_physical = physical * (
            1.0 - settings.pattern_boundary_discount * pattern
        )
        topography = (
            settings.physical_boundary_weight * effective_physical
            + settings.distance_topography_weight * (1.0 - np.clip(distance, 0.0, 1.0))
            + settings.uncertainty_penalty * uncertainty
        )
        topography /= max(
            1e-6,
            settings.physical_boundary_weight
            + settings.distance_topography_weight
            + settings.uncertainty_penalty,
        )
        topography_u8 = _u8_probability(topography)
        watershed = cv2.watershed(
            cv2.cvtColor(topography_u8, cv2.COLOR_GRAY2BGR), markers
        )
        labels = np.where((watershed >= 2) & foreground, watershed - 1, 0).astype(np.int32)
        labels = _remove_small(labels, settings.minimum_instance_area_px)
    confidence_score = np.clip(
        interior * (1.0 - uncertainty) * (1.0 - 0.5 * physical), 0.0, 1.0
    )
    centres, confidences = _centres_and_confidence(labels, confidence_score)
    return LearnedInstanceResult(
        family=ModelFamily.UNET_WATERSHED,
        labels=labels,
        centres_xy=centres,
        instance_confidences=confidences,
        rasters={
            "interior": _u8_probability(interior),
            "physical_boundary": _u8_probability(physical),
            "pattern_boundary": _u8_probability(pattern),
            "centre": _u8_probability(centre_score),
            "distance": _u8_probability(distance),
            "uncertainty": _u8_probability(uncertainty),
        },
        method="multi-head U-Net with pattern-aware marker-controlled watershed",
        checkpoint_id=checkpoint_id,
    )


def _remove_small(labels: np.ndarray, minimum_area: int) -> np.ndarray:
    labels = relabel_consecutive(labels)
    if not labels.max(initial=0):
        return labels
    areas = np.bincount(labels.reshape(-1))
    keep = areas >= int(minimum_area)
    keep[0] = False
    return relabel_consecutive(np.where(keep[labels], labels, 0))


@dataclass(frozen=True, slots=True)
class StarDistDecodeSettings:
    object_threshold: float = 0.45
    nms_iou_threshold: float = 0.35
    local_maximum_radius_px: int = 3
    minimum_instance_area_px: int = 24
    maximum_candidates: int = 4096

    def __post_init__(self) -> None:
        if not 0.0 <= self.object_threshold <= 1.0:
            raise ValueError("object_threshold must be between zero and one.")
        if not 0.0 <= self.nms_iou_threshold <= 1.0:
            raise ValueError("nms_iou_threshold must be between zero and one.")
        if self.local_maximum_radius_px < 1:
            raise ValueError("local_maximum_radius_px must be positive.")
        if self.minimum_instance_area_px < 1 or self.maximum_candidates < 1:
            raise ValueError("Area and candidate limits must be positive.")


@dataclass(frozen=True, slots=True)
class HybridDecodeSettings:
    """U-Net topology with StarDist proposals admitted only inside seed material."""

    unet: UNetWatershedSettings = UNetWatershedSettings()
    stardist: StarDistDecodeSettings = StarDistDecodeSettings()
    stardist_gate_interior_threshold: float = 0.50
    stardist_gate_distance_threshold: float = 0.35

    def __post_init__(self) -> None:
        if not 0.0 <= self.stardist_gate_interior_threshold <= 1.0:
            raise ValueError("Hybrid StarDist gate threshold must be between zero and one.")
        if not 0.0 <= self.stardist_gate_distance_threshold <= 1.0:
            raise ValueError("Hybrid distance-core threshold must be between zero and one.")


def _polygon(center_x: float, center_y: float, distances: np.ndarray) -> np.ndarray:
    angles = np.arange(len(distances), dtype=np.float32) * (
        2.0 * np.pi / max(1, len(distances))
    )
    return np.column_stack(
        (
            center_x + distances * np.cos(angles),
            center_y + distances * np.sin(angles),
        )
    ).astype(np.float32)


def _polygon_bounds(polygon: np.ndarray, shape: tuple[int, int]):
    height, width = shape
    x0 = max(0, int(np.floor(polygon[:, 0].min())))
    y0 = max(0, int(np.floor(polygon[:, 1].min())))
    x1 = min(width, int(np.ceil(polygon[:, 0].max())) + 1)
    y1 = min(height, int(np.ceil(polygon[:, 1].max())) + 1)
    return x0, y0, x1, y1


def _polygon_iou(first: np.ndarray, second: np.ndarray, shape: tuple[int, int]) -> float:
    first_bounds = _polygon_bounds(first, shape)
    second_bounds = _polygon_bounds(second, shape)
    x0 = min(first_bounds[0], second_bounds[0])
    y0 = min(first_bounds[1], second_bounds[1])
    x1 = max(first_bounds[2], second_bounds[2])
    y1 = max(first_bounds[3], second_bounds[3])
    if (
        first_bounds[2] <= second_bounds[0]
        or second_bounds[2] <= first_bounds[0]
        or first_bounds[3] <= second_bounds[1]
        or second_bounds[3] <= first_bounds[1]
    ):
        return 0.0
    width, height = x1 - x0, y1 - y0
    if width <= 0 or height <= 0:
        return 0.0
    first_mask = np.zeros((height, width), dtype=np.uint8)
    second_mask = np.zeros_like(first_mask)
    offset = np.asarray((x0, y0), dtype=np.float32)
    cv2.fillPoly(first_mask, [np.rint(first - offset).astype(np.int32)], 1)
    cv2.fillPoly(second_mask, [np.rint(second - offset).astype(np.int32)], 1)
    intersection = int(np.count_nonzero(first_mask & second_mask))
    union = int(np.count_nonzero(first_mask | second_mask))
    return intersection / max(1, union)


def decode_stardist(
    outputs,
    *,
    valid_mask: np.ndarray | None = None,
    settings: StarDistDecodeSettings = StarDistDecodeSettings(),
    checkpoint_id: str = "",
) -> LearnedInstanceResult:
    """Decode dense radial predictions into non-overlapping instance labels."""

    object_probability = _single_numpy(outputs["object_logits"], sigmoid=True)
    distances = _channels_numpy(outputs["radial_distances"])
    uncertainty = _single_numpy(outputs["radial_uncertainty_logits"], sigmoid=True)
    shape = object_probability.shape
    if distances.ndim != 3 or distances.shape[1:] != shape:
        raise ValueError("StarDist radial predictions must be ray x height x width.")
    valid = np.ones(shape, dtype=bool) if valid_mask is None else np.asarray(valid_mask) > 0
    if valid.shape != shape:
        raise ValueError("The valid mask must match the prediction dimensions.")
    radius = int(settings.local_maximum_radius_px)
    maximum = cv2.dilate(
        object_probability.astype(np.float32),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2),
    )
    candidate_mask = (
        (object_probability >= settings.object_threshold)
        & (object_probability >= maximum - 1e-7)
        & valid
    )
    rows, columns = np.nonzero(candidate_mask)
    scores = object_probability[rows, columns] * (1.0 - uncertainty[rows, columns])
    order = np.argsort(scores)[::-1][: settings.maximum_candidates]
    retained_polygons: list[np.ndarray] = []
    retained_scores: list[float] = []
    retained_centres: list[tuple[float, float]] = []
    if len(order):
        proposal_diameters = [
            2.0 * float(np.median(distances[:, int(rows[index]), int(columns[index])]))
            for index in order[: min(256, len(order))]
        ]
        grid_size = max(8, int(round(np.median(proposal_diameters))))
    else:
        grid_size = 32
    spatial_index: dict[tuple[int, int], list[int]] = {}

    def polygon_cells(polygon):
        x0, y0, x1, y1 = _polygon_bounds(polygon, shape)
        return tuple(
            (cell_x, cell_y)
            for cell_y in range(y0 // grid_size, max(y0 // grid_size, (y1 - 1) // grid_size) + 1)
            for cell_x in range(x0 // grid_size, max(x0 // grid_size, (x1 - 1) // grid_size) + 1)
        )

    for index in order:
        row, column = int(rows[index]), int(columns[index])
        radial = np.maximum(0.5, distances[:, row, column])
        polygon = _polygon(float(column), float(row), radial)
        cells = polygon_cells(polygon)
        neighbours = {
            retained_index
            for cell in cells
            for retained_index in spatial_index.get(cell, ())
        }
        if any(
            _polygon_iou(polygon, retained_polygons[retained_index], shape)
            > settings.nms_iou_threshold
            for retained_index in neighbours
        ):
            continue
        retained_index = len(retained_polygons)
        retained_polygons.append(polygon)
        retained_scores.append(float(scores[index]))
        retained_centres.append((float(column), float(row)))
        for cell in cells:
            spatial_index.setdefault(cell, []).append(retained_index)

    labels = np.zeros(shape, dtype=np.int32)
    final_scores: list[float] = []
    final_centres: list[tuple[float, float]] = []
    for polygon, score, centre in zip(
        retained_polygons, retained_scores, retained_centres, strict=True
    ):
        x0, y0, x1, y1 = _polygon_bounds(polygon, shape)
        if x1 <= x0 or y1 <= y0:
            continue
        local_mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
        offset = np.asarray((x0, y0), dtype=np.float32)
        cv2.fillPoly(
            local_mask,
            [np.rint(polygon - offset).astype(np.int32)],
            1,
        )
        local_labels = labels[y0:y1, x0:x1]
        selected = (
            (local_mask > 0)
            & valid[y0:y1, x0:x1]
            & (local_labels == 0)
        )
        if int(np.count_nonzero(selected)) < settings.minimum_instance_area_px:
            continue
        identifier = len(final_scores) + 1
        local_labels[selected] = identifier
        final_scores.append(score)
        final_centres.append(centre)
    labels = relabel_consecutive(labels)
    # Cropping by higher-scored polygons can shift visible centroids; report
    # decoded label centroids while preserving the proposal score.
    centres, _ = _centres_and_confidence(labels, object_probability)
    confidences = np.asarray(final_scores[: len(centres)], dtype=np.float32)
    return LearnedInstanceResult(
        family=ModelFamily.STARDIST,
        labels=labels,
        centres_xy=centres,
        instance_confidences=np.clip(confidences, 0.0, 1.0),
        rasters={
            "object_probability": _u8_probability(object_probability),
            "radial_uncertainty": _u8_probability(uncertainty),
        },
        method=f"StarDist with {distances.shape[0]} rays and polygon NMS",
        checkpoint_id=checkpoint_id,
    )


def decode_unet_stardist_hybrid(
    unet_outputs,
    stardist_outputs,
    *,
    valid_mask: np.ndarray | None = None,
    settings: HybridDecodeSettings = HybridDecodeSettings(),
    checkpoint_id: str = "",
) -> LearnedInstanceResult:
    """Use gated StarDist proposals as optional U-Net watershed markers.

    U-Net remains responsible for material support and final boundaries. This
    prevents standalone StarDist objectness from proposing dish-rim/glare
    polygons while allowing its shape-aware centres to recover seeds missed by
    the U-Net centre head.
    """

    interior = _single_numpy(unet_outputs["interior_logits"], sigmoid=True)
    distance = _single_numpy(unet_outputs["distance"])
    gate = (
        (interior >= settings.stardist_gate_interior_threshold)
        & (distance >= settings.stardist_gate_distance_threshold)
    )
    if valid_mask is not None:
        valid = np.asarray(valid_mask) > 0
        if valid.shape != gate.shape:
            raise ValueError("Hybrid valid mask must match prediction dimensions.")
        gate &= valid
    else:
        valid = None
    star = decode_stardist(
        stardist_outputs,
        valid_mask=gate,
        settings=settings.stardist,
        checkpoint_id=checkpoint_id,
    )
    markers = np.zeros(gate.shape, dtype=np.int32)
    next_identifier = 1
    for x, y in star.centres_xy:
        column, row = int(round(float(x))), int(round(float(y)))
        if 0 <= row < gate.shape[0] and 0 <= column < gate.shape[1] and gate[row, column]:
            cv2.circle(markers, (column, row), 1, next_identifier, -1)
            next_identifier += 1
    result = decode_unet_watershed(
        unet_outputs,
        valid_mask=valid,
        painted_instances=markers if next_identifier > 1 else None,
        settings=settings.unet,
        checkpoint_id=checkpoint_id,
    )
    rasters = dict(result.rasters)
    rasters["gated_stardist_object"] = star.rasters["object_probability"]
    rasters["stardist_markers"] = _u8_probability(markers > 0)
    result.rasters = rasters
    result.method = (
        "U-Net pattern-aware watershed with interior-gated StarDist markers"
    )
    return result
