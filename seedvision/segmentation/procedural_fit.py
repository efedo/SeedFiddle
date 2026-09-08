"""Small deterministic fitter for the procedural instance separator.

The fitter deliberately knows nothing about Qt or node state.  Its caller
provides a function that evaluates an immutable
:class:`ProceduralInstanceSettings` proposal, so an optimization run cannot
silently change the active pipeline.  This also lets the pipeline prepare and
reuse bounded topology inputs instead of repeatedly materializing full-size
host rasters.

Annotated instances are treated as supervision, not as watershed markers,
while fitting.  Predictions that overlap an annotation are matched one-to-one to
the labelled seeds.  Pixels belonging to the matched prediction but not to its
annotation count as false positives, including leaked background and pixels
from an adjacent seed.  Their loss grows exponentially with distance from the
matched annotation.  Disjoint predictions are ignored by default because a
user may have annotated only a subset of an image; callers can declare the
annotations complete to score every predicted object.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from math import isfinite

import cv2
import numpy as np

from seedvision.segmentation.procedural import (
    ProceduralInstanceResult,
    ProceduralInstanceSettings,
)


@dataclass(frozen=True, slots=True)
class ProceduralFitParameter:
    """One bounded coordinate searched by the lightweight fitter."""

    name: str
    minimum: float
    maximum: float
    initial_step: float


# Keep the search intentionally small.  These controls have direct effects on
# foreground support, boundary traversal, marker density, or final region
# filtering.  Display-only controls and the topology working dimension are not
# eligible for fitting.
DEFAULT_PROCEDURAL_FIT_PARAMETERS = (
    ProceduralFitParameter("foreground_threshold_scale", 0.55, 1.25, 0.16),
    ProceduralFitParameter("occupancy_closing_fraction", 0.03, 0.24, 0.04),
    ProceduralFitParameter("boundary_semantic_floor", 0.0, 0.40, 0.08),
    ProceduralFitParameter("boundary_nonphysical_discount", 0.20, 1.0, 0.16),
    ProceduralFitParameter("boundary_physical_ridge_weight", 0.0, 1.0, 0.18),
    ProceduralFitParameter("boundary_trace_weight", 0.0, 1.0, 0.18),
    ProceduralFitParameter("centre_minimum_separation_fraction", 0.24, 0.82, 0.10),
    ProceduralFitParameter(
        "sparse_centre_minimum_separation_fraction", 0.30, 0.90, 0.10
    ),
    ProceduralFitParameter("marker_count_multiplier", 0.70, 1.40, 0.12),
    ProceduralFitParameter("minimum_marker_score", 0.04, 0.36, 0.06),
    ProceduralFitParameter("minimum_instance_area_fraction", 0.12, 0.30, 0.04),
    ProceduralFitParameter("soft_minimum_instance_area_fraction", 0.32, 0.60, 0.06),
    ProceduralFitParameter("soft_maximum_instance_width_fraction", 0.95, 1.25, 0.06),
    ProceduralFitParameter("hard_maximum_instance_width_fraction", 1.28, 1.60, 0.06),
    ProceduralFitParameter("maximum_internal_concavity_fraction", 0.08, 0.30, 0.04),
    ProceduralFitParameter("maximum_protrusion_area_fraction", 0.05, 0.22, 0.03),
)

_FITTABLE_PARAMETER_NAMES = {
    # Occupancy support.
    "foreground_threshold_scale",
    "occupancy_closing_fraction",
    "occupancy_hole_area_fraction",
    "dish_margin_fraction",
    # Watershed boundary cost.
    "boundary_edge_weight",
    "boundary_ridge_weight",
    "boundary_semantic_floor",
    "boundary_nonphysical_discount",
    "boundary_physical_ridge_weight",
    "boundary_trace_weight",
    "trace_minimum_length_fraction",
    "trace_convexity_weight",
    # Marker construction and label filtering.
    "centre_geometry_smoothing_fraction",
    "centre_material_weight",
    "centre_distance_weight",
    "centre_minimum_separation_fraction",
    "sparse_centre_minimum_separation_fraction",
    "marker_count_multiplier",
    "minimum_marker_score",
    "minimum_instance_area_fraction",
    "soft_minimum_instance_area_fraction",
    "maximum_instance_area_fraction",
    "soft_maximum_instance_width_fraction",
    "hard_maximum_instance_width_fraction",
    "maximum_internal_concavity_fraction",
    "maximum_protrusion_area_fraction",
    "minimum_instance_solidity",
    "maximum_instance_axis_ratio",
}


@dataclass(frozen=True, slots=True)
class ProceduralFitOptions:
    """Bounds and loss weights for a short coordinate-search fit."""

    maximum_evaluations: int = 35
    passes: int = 2
    step_decay: float = 0.50
    false_positive_weight: float = 2.0
    false_negative_weight: float = 1.0
    overreach_distance_scale_fraction: float = 0.50
    instance_penalty_weight: float = 0.10
    minimum_match_iou: float = 0.20
    missed_seed_weight: float = 0.50
    incorrect_concavity_weight: float = 2.0
    annotations_are_complete: bool = False
    parameters: tuple[ProceduralFitParameter, ...] = DEFAULT_PROCEDURAL_FIT_PARAMETERS

    def __post_init__(self) -> None:
        if self.maximum_evaluations < 1:
            raise ValueError("At least one procedural fit evaluation is required.")
        if self.passes < 1:
            raise ValueError("At least one procedural fit pass is required.")
        if not 0.0 < self.step_decay <= 1.0:
            raise ValueError("Procedural fit step decay must be in (0, 1].")
        if not isfinite(self.false_negative_weight) or self.false_negative_weight <= 0.0:
            raise ValueError("False-negative weight must be positive.")
        if not isfinite(self.false_positive_weight) or self.false_positive_weight <= 0.0:
            raise ValueError("False-positive weight must be positive.")
        if not isfinite(self.overreach_distance_scale_fraction) or (
            self.overreach_distance_scale_fraction <= 0.0
        ):
            raise ValueError(
                "Overreach distance scale fraction must be positive."
            )
        if not isfinite(self.instance_penalty_weight) or self.instance_penalty_weight < 0.0:
            raise ValueError("Instance penalty weight must be non-negative.")
        _validate_reference_costs(
            self.minimum_match_iou, self.missed_seed_weight,
            self.incorrect_concavity_weight,
        )
        names: set[str] = set()
        for parameter in self.parameters:
            if parameter.name not in _FITTABLE_PARAMETER_NAMES:
                raise ValueError(
                    f"Procedural parameter {parameter.name!r} is not eligible "
                    "for instance-label fitting."
                )
            if parameter.name in names:
                raise ValueError(
                    f"Duplicate procedural fit parameter {parameter.name!r}."
                )
            if not parameter.minimum < parameter.maximum:
                raise ValueError(
                    f"Fit bounds for {parameter.name!r} are inconsistent."
                )
            if parameter.initial_step <= 0.0:
                raise ValueError(
                    f"Fit step for {parameter.name!r} must be positive."
                )
            names.add(parameter.name)


@dataclass(frozen=True, slots=True)
class ProceduralFitScore:
    """Instance-aware asymmetric error against the annotated subset."""

    loss: float
    true_positive_pixels: int
    false_positive_pixels: int
    false_negative_pixels: int
    matched_instances: int
    annotated_instances: int
    evaluated_predictions: int
    false_positive_instances: int
    false_negative_instances: int
    pixel_precision: float
    pixel_recall: float
    distance_weighted_false_positive_pixels: float = 0.0
    overreach_distance_scale_px: float = 1.0
    missed_reference_pixels: int = 0
    incorrect_concavity_pixels: int = 0
    incorrect_concavity_cost: float = 0.0
    total_pixel_cost: float = 0.0

    @property
    def perfect_fit(self) -> bool:
        return self.loss <= 1e-12


@dataclass(frozen=True, slots=True)
class ProceduralReferenceErrorMap:
    """Post-inference pixel costs, shared exactly with the fitting objective.

    Blue: underreach; red: overreach; amber: unmatched reference; magenta:
    incorrect perimeter-concavity pocket. Alpha represents total cost at that
    pixel, capped at ``maximum_display_cost``. Unreviewed objects are ignored.
    """

    rgba: np.ndarray
    matched_instances: int
    underreach_pixels: int
    overreach_pixels: int
    maximum_display_cost: float
    missed_reference_instances: int
    missed_reference_pixels: int
    incorrect_concavity_pixels: int
    pixel_costs: np.ndarray
    matched_pairs: tuple[tuple[int, int], ...]


def _validate_reference_costs(
    minimum_match_iou: float, missed_seed_weight: float,
    incorrect_concavity_weight: float,
) -> None:
    if not isfinite(minimum_match_iou) or not 0.0 <= minimum_match_iou < 1.0:
        raise ValueError("Minimum match IoU must be in [0, 1).")
    if not isfinite(missed_seed_weight) or not 0.0 < missed_seed_weight <= 1.0:
        raise ValueError("Missed-seed cost must be in (0, 1].")
    if not isfinite(incorrect_concavity_weight) or incorrect_concavity_weight < 0.0:
        raise ValueError("Incorrect-concavity cost must be finite and non-negative.")


@dataclass(frozen=True, slots=True)
class ProceduralFitTrial:
    """One actual separator evaluation in deterministic search order."""

    evaluation_number: int
    pass_number: int
    parameter_name: str
    parameter_value: float
    score: ProceduralFitScore


@dataclass(frozen=True, slots=True)
class ProceduralFitResult:
    """A non-mutating settings proposal and its audit trail."""

    initial_settings: ProceduralInstanceSettings
    proposed_settings: ProceduralInstanceSettings
    initial_score: ProceduralFitScore
    proposed_score: ProceduralFitScore
    trials: tuple[ProceduralFitTrial, ...]
    cancelled: bool = False

    @property
    def evaluations(self) -> int:
        return len(self.trials)

    @property
    def improved(self) -> bool:
        return self.proposed_score.loss < self.initial_score.loss - 1e-12


@dataclass(frozen=True, slots=True)
class _ProceduralFitScoringContext:
    """Fit-local immutable annotation geometry reused by every trial."""

    target: np.ndarray
    target_count: int
    target_areas: np.ndarray
    target_bounds: np.ndarray
    distance_to_annotation: np.ndarray
    overreach_distance_scale_px: float


def _consecutive_labels(values: np.ndarray, *, label: str) -> np.ndarray:
    source = np.asarray(values)
    if source.ndim != 2:
        raise ValueError(f"{label} must be a two-dimensional label raster.")
    if not (
        np.issubdtype(source.dtype, np.integer)
        or np.issubdtype(source.dtype, np.bool_)
    ):
        raise ValueError(f"{label} must contain integer instance identifiers.")
    if np.any(source < 0):
        raise ValueError(f"{label} cannot contain negative instance identifiers.")
    identifiers, inverse = np.unique(source, return_inverse=True)
    mapped = np.zeros(len(identifiers), dtype=np.int32)
    positive = identifiers > 0
    mapped[positive] = np.arange(1, int(np.count_nonzero(positive)) + 1)
    return mapped[inverse].reshape(source.shape)


def _label_bounds(labels: np.ndarray, count: int) -> np.ndarray:
    """Return inclusive-exclusive ``x0, y0, x1, y1`` bounds per label."""

    height, width = labels.shape
    bounds = np.empty((count + 1, 4), dtype=np.int32)
    bounds[:, 0] = width
    bounds[:, 1] = height
    bounds[:, 2:] = 0
    if count == 0:
        return bounds
    y, x = np.nonzero(labels)
    identifiers = labels[y, x]
    np.minimum.at(bounds[:, 0], identifiers, x)
    np.minimum.at(bounds[:, 1], identifiers, y)
    np.maximum.at(bounds[:, 2], identifiers, x + 1)
    np.maximum.at(bounds[:, 3], identifiers, y + 1)
    return bounds


def _overreach_multiplier(
    distances: np.ndarray,
    distance_scale_px: float,
) -> np.ndarray:
    """Map distance outside a target to a bounded exponential pixel cost.

    The subtraction by one makes the cost zero on the annotated region and
    small immediately outside it.  One distance scale carries one equivalent
    false-positive pixel of cost, two scales carry three, and so on.  Capping
    the exponent keeps a grossly erroneous region finite without changing the
    ordering of realistic proposals.
    """

    normalized = np.minimum(
        np.asarray(distances, dtype=np.float32) / max(1e-6, distance_scale_px),
        8.0,
    )
    return np.exp2(normalized) - 1.0


def _prepare_scoring_context(
    annotations: np.ndarray,
    prediction_shape: tuple[int, int],
    *,
    overreach_distance_scale_fraction: float,
    seed_diameter_px: float | None,
) -> _ProceduralFitScoringContext:
    """Prepare immutable target geometry once for a multi-trial fit."""

    if len(prediction_shape) != 2:
        raise ValueError("Prediction must be a two-dimensional label raster.")
    target_source = np.asarray(annotations)
    if target_source.ndim != 2:
        raise ValueError("Annotations must be a two-dimensional label raster.")
    if target_source.shape != prediction_shape:
        target_source = cv2.resize(
            target_source,
            (prediction_shape[1], prediction_shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    target = _consecutive_labels(target_source, label="Annotations")
    target_count = int(target.max(initial=0))
    if target_count == 0:
        raise ValueError("At least one annotated seed instance is required.")
    target_areas = np.bincount(target.reshape(-1), minlength=target_count + 1)
    if seed_diameter_px is None:
        equivalent_diameters = 2.0 * np.sqrt(
            np.asarray(
                target_areas[1 : target_count + 1], dtype=np.float64
            )
            / np.pi
        )
        reference_seed_diameter_px = float(np.median(equivalent_diameters))
    else:
        reference_seed_diameter_px = float(seed_diameter_px)
        if (
            not isfinite(reference_seed_diameter_px)
            or reference_seed_diameter_px <= 0
        ):
            raise ValueError("Seed diameter for overreach scoring must be positive.")
    distance_scale_px = max(
        1.0,
        float(overreach_distance_scale_fraction) * reference_seed_diameter_px,
    )
    target_bounds = _label_bounds(target, target_count)
    distance_to_annotation = cv2.distanceTransform(
        np.asarray(target == 0, dtype=np.uint8),
        cv2.DIST_L2,
        cv2.DIST_MASK_PRECISE,
    )
    for array in (
        target,
        target_areas,
        target_bounds,
        distance_to_annotation,
    ):
        array.setflags(write=False)
    return _ProceduralFitScoringContext(
        target=target,
        target_count=target_count,
        target_areas=target_areas,
        target_bounds=target_bounds,
        distance_to_annotation=distance_to_annotation,
        overreach_distance_scale_px=distance_scale_px,
    )


def _maximum_weight_assignment(weights: np.ndarray) -> list[tuple[int, int]]:
    """Rectangular Hungarian assignment; private zero-weight skip columns.

    Only compact metadata is handled here, never an image tensor. Rows are
    references, columns are predictions plus one dummy per reference.
    """
    rows, columns = weights.shape
    costs = np.concatenate((-weights, np.zeros((rows, rows))), axis=1)
    column_count = costs.shape[1]
    u = np.zeros(rows + 1)
    v = np.zeros(column_count + 1)
    owner = np.zeros(column_count + 1, np.int32)
    way = np.zeros(column_count + 1, np.int32)
    for row in range(1, rows + 1):
        owner[0] = row
        minimum = np.full(column_count + 1, np.inf)
        used = np.zeros(column_count + 1, bool)
        column = 0
        while True:
            used[column] = True
            current = int(owner[column])
            available = np.flatnonzero(~used[1:]) + 1
            reduced = costs[current - 1, available - 1] - u[current] - v[available]
            improved = reduced < minimum[available]
            changed = available[improved]
            minimum[changed] = reduced[improved]
            way[changed] = column
            next_column = int(available[np.argmin(minimum[available])])
            delta = minimum[next_column]
            u[owner[used]] += delta
            v[used] -= delta
            minimum[~used] -= delta
            column = next_column
            if owner[column] == 0:
                break
        while column:
            previous = int(way[column])
            owner[column] = owner[previous]
            column = previous
    return [
        (int(owner[column]) - 1, column - 1)
        for column in range(1, columns + 1)
        if owner[column] and weights[int(owner[column]) - 1, column - 1] > 0.0
    ]


def _match_instances(
    target: np.ndarray,
    predicted: np.ndarray,
    target_areas: np.ndarray,
    *,
    minimum_match_iou: float = 0.20,
) -> tuple[
    list[tuple[int, int, int]], set[int], set[int], set[int], np.ndarray,
]:
    """Globally maximize normalized overlap, with explicit unmatched choices.

    A slight touch cannot pair two neighbouring seeds. The reward is IoU minus
    the minimum acceptable IoU, so marginal pairs cannot force a stronger pair
    apart merely to increase match count. Independent overlap components are
    solved separately to avoid a dense whole-dish assignment matrix.
    """
    prediction_count = int(predicted.max(initial=0))
    prediction_areas = np.bincount(
        predicted.reshape(-1), minlength=prediction_count + 1
    )
    overlap = (target > 0) & (predicted > 0)
    pairs: dict[tuple[int, int], tuple[int, float]] = {}
    by_target: dict[int, set[int]] = {}
    by_prediction: dict[int, set[int]] = {}
    if np.any(overlap):
        stride = prediction_count + 1
        codes, intersections = np.unique(
            target[overlap].astype(np.int64) * stride + predicted[overlap],
            return_counts=True,
        )
        for code, intersection in zip(codes, intersections, strict=True):
            target_id, prediction_id = int(code // stride), int(code % stride)
            union = int(target_areas[target_id]) + int(
                prediction_areas[prediction_id]
            ) - int(intersection)
            iou = float(intersection) / max(1, union)
            if iou <= minimum_match_iou:
                continue
            pairs[target_id, prediction_id] = (int(intersection), iou)
            by_target.setdefault(target_id, set()).add(prediction_id)
            by_prediction.setdefault(prediction_id, set()).add(target_id)

    accepted: list[tuple[int, int, int]] = []
    remaining = set(by_target)
    while remaining:
        targets: set[int] = set()
        predictions: set[int] = set()
        pending = {min(remaining)}
        while pending:
            target_id = pending.pop()
            targets.add(target_id)
            for prediction_id in by_target[target_id] - predictions:
                predictions.add(prediction_id)
                pending.update(by_prediction[prediction_id] - targets)
        remaining.difference_update(targets)
        target_ids, prediction_ids = sorted(targets), sorted(predictions)
        weights = np.zeros((len(target_ids), len(prediction_ids)))
        for row, target_id in enumerate(target_ids):
            for column, prediction_id in enumerate(prediction_ids):
                pair = pairs.get((target_id, prediction_id))
                if pair is not None:
                    weights[row, column] = pair[1] - minimum_match_iou
        for row, column in _maximum_weight_assignment(weights):
            target_id, prediction_id = target_ids[row], prediction_ids[column]
            accepted.append((target_id, prediction_id, pairs[target_id, prediction_id][0]))
    accepted.sort()
    return (
        accepted, {item[0] for item in accepted}, {item[1] for item in accepted},
        set(by_prediction), prediction_areas,
    )


def _perimeter_concavity_pockets(mask: np.ndarray) -> np.ndarray:
    """Exterior-connected hull deficits, separately per connected component.

    Holes and spaces between disconnected components are not concavities.
    This is the same perimeter-pocket definition as the procedural overlay.
    """
    padded = np.pad(np.asarray(mask, np.uint8), 1)
    hulls = np.zeros_like(padded)
    contours, _ = cv2.findContours(
        padded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    for contour in contours:
        if len(contour) >= 3:
            cv2.fillConvexPoly(hulls, cv2.convexHull(contour), 1)
    _, exterior = cv2.connectedComponents(np.uint8(padded == 0), connectivity=8)
    return ((hulls > 0) & (exterior == exterior[0, 0]))[1:-1, 1:-1]


def _compare_instances(
    predicted: np.ndarray,
    context: _ProceduralFitScoringContext,
    *,
    false_positive_weight: float,
    false_negative_weight: float,
    instance_penalty_weight: float,
    annotations_are_complete: bool,
    minimum_match_iou: float,
    missed_seed_weight: float,
    incorrect_concavity_weight: float,
    render: bool = False,
) -> tuple[ProceduralFitScore, ProceduralReferenceErrorMap | None]:
    """Single source of truth for both fitting costs and the diagnostic map."""
    target = context.target
    count = int(predicted.max(initial=0))
    accepted, matched_targets, matched_predictions, eligible, prediction_areas = (
        _match_instances(target, predicted, context.target_areas,
                         minimum_match_iou=minimum_match_iou)
    )
    evaluated = set(range(1, count + 1)) if annotations_are_complete else eligible
    extra_predictions = evaluated - matched_predictions
    prediction_bounds = _label_bounds(predicted, count)
    tp = fp = fn = missed_pixels = concavity_pixels = 0
    weighted_overreach = 0.0
    pixel_cost = 0.0
    costs = np.zeros(predicted.shape, np.float32) if render else None
    colours = np.zeros(predicted.shape, np.uint8) if render else None
    strongest = np.zeros(predicted.shape, np.float32) if render else None
    concavity_display = np.zeros(predicted.shape, bool) if render else None

    def add(kind: int, region, mask: np.ndarray, values) -> None:
        nonlocal pixel_cost
        value = np.broadcast_to(np.asarray(values, np.float32), (np.count_nonzero(mask),))
        pixel_cost += float(np.sum(value, dtype=np.float64))
        if costs is not None:
            costs[region][mask] += value
            # Strongest term wins; equal costs use a fixed category priority.
            # Sum all contributions instead of overwriting crossed-pair costs.
            win = mask.copy()
            win[mask] = (value > strongest[region][mask]) | (
                (value == strongest[region][mask]) & (kind > colours[region][mask])
            )
            strongest[region][win] = np.broadcast_to(
                np.asarray(values, np.float32), (np.count_nonzero(mask),)
            )[win[mask]]
            colours[region][win] = kind

    for target_id, prediction_id, intersection in accepted:
        tx0, ty0, tx1, ty1 = context.target_bounds[target_id]
        px0, py0, px1, py1 = prediction_bounds[prediction_id]
        region = np.s_[min(ty0, py0):max(ty1, py1), min(tx0, px0):max(tx1, px1)]
        target_roi = target[region] == target_id
        predicted_roi = predicted[region] == prediction_id
        underreach = target_roi & ~predicted_roi
        overreach = predicted_roi & ~target_roi
        tp += intersection
        fp += int(prediction_areas[prediction_id]) - intersection
        fn += int(context.target_areas[target_id]) - intersection
        add(1, region, underreach, false_negative_weight)
        if np.any(overreach):
            distance = cv2.distanceTransform(
                np.uint8(~target_roi), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
            )
            multiplier = _overreach_multiplier(
                distance[overreach], context.overreach_distance_scale_px
            )
            weighted_overreach += float(np.sum(multiplier, dtype=np.float64))
            add(3, region, overreach, false_positive_weight * multiplier)
        # A correctly reproduced hilum or notch has no error pixels here.
        # Do not call enclosed holes or an entire candidate's interior concave.
        concave_error = underreach & _perimeter_concavity_pockets(predicted_roi)
        concavity_pixels += int(np.count_nonzero(concave_error))
        add(4, region, concave_error, incorrect_concavity_weight)
        if concavity_display is not None and incorrect_concavity_weight > 0.0:
            concavity_display[region] |= concave_error

    for target_id in range(1, context.target_count + 1):
        if target_id in matched_targets:
            continue
        x0, y0, x1, y1 = context.target_bounds[target_id]
        region = np.s_[y0:y1, x0:x1]
        mask = target[region] == target_id
        area = int(context.target_areas[target_id])
        fn += area
        missed_pixels += area
        add(2, region, mask, false_negative_weight * missed_seed_weight)

    for prediction_id in sorted(extra_predictions):
        x0, y0, x1, y1 = prediction_bounds[prediction_id]
        region = np.s_[y0:y1, x0:x1]
        mask = predicted[region] == prediction_id
        fp += int(prediction_areas[prediction_id])
        multiplier = _overreach_multiplier(
            context.distance_to_annotation[region][mask],
            context.overreach_distance_scale_px,
        )
        weighted_overreach += float(np.sum(multiplier, dtype=np.float64))
        add(3, region, mask, false_positive_weight * multiplier)

    missed_count = context.target_count - len(matched_targets)
    # Fixed reviewed-area denominator. The former TP+error normalization made
    # every empty prediction cost 1 regardless of its missed-seed weight.
    area = max(1, int(np.sum(context.target_areas[1:])))
    count_cost = instance_penalty_weight * (
        false_positive_weight * len(extra_predictions)
        + false_negative_weight * missed_seed_weight * missed_count
    ) / max(1, context.target_count)
    score = ProceduralFitScore(
        loss=pixel_cost / area + count_cost,
        true_positive_pixels=tp, false_positive_pixels=fp,
        false_negative_pixels=fn, matched_instances=len(accepted),
        annotated_instances=context.target_count,
        evaluated_predictions=len(evaluated),
        false_positive_instances=len(extra_predictions),
        false_negative_instances=missed_count,
        pixel_precision=tp / max(1, tp + fp), pixel_recall=tp / max(1, tp + fn),
        distance_weighted_false_positive_pixels=weighted_overreach,
        overreach_distance_scale_px=context.overreach_distance_scale_px,
        missed_reference_pixels=missed_pixels,
        incorrect_concavity_pixels=concavity_pixels,
        incorrect_concavity_cost=incorrect_concavity_weight * concavity_pixels,
        total_pixel_cost=pixel_cost,
    )
    if costs is None:
        return score, None
    # Fixed units, never normalized to this image's maximum: changing a weight
    # changes brightness honestly. Exact unsaturated costs remain available.
    maximum_display_cost = 4.0
    colours[concavity_display] = 4
    palette = np.asarray(
        ((0, 0, 0), (40, 120, 255), (255, 180, 25), (255, 55, 35), (235, 45, 235)),
        np.uint8,
    )
    rgba = np.zeros((*predicted.shape, 4), np.uint8)
    rgba[..., :3] = palette[colours]
    rgba[..., 3] = np.uint8(np.clip(np.rint(255 * costs / maximum_display_cost), 0, 255))
    return score, ProceduralReferenceErrorMap(
        rgba=rgba, matched_instances=len(accepted), underreach_pixels=fn - missed_pixels,
        overreach_pixels=fp, maximum_display_cost=maximum_display_cost,
        missed_reference_instances=missed_count, missed_reference_pixels=missed_pixels,
        incorrect_concavity_pixels=concavity_pixels, pixel_costs=costs,
        matched_pairs=tuple((item[0], item[1]) for item in accepted),
    )


def procedural_reference_error_map(
    annotations: np.ndarray,
    prediction: np.ndarray,
    *,
    overreach_weight: float = 2.0,
    overreach_distance_scale_fraction: float = 0.50,
    seed_diameter_px: float | None = None,
    minimum_match_iou: float = 0.20,
    missed_seed_weight: float = 0.50,
    incorrect_concavity_weight: float = 2.0,
    annotations_are_complete: bool = False,
) -> ProceduralReferenceErrorMap:
    """Render the exact fitting pixel costs, without altering any prediction."""
    options = ProceduralFitOptions(
        false_positive_weight=overreach_weight,
        overreach_distance_scale_fraction=overreach_distance_scale_fraction,
        minimum_match_iou=minimum_match_iou, missed_seed_weight=missed_seed_weight,
        incorrect_concavity_weight=incorrect_concavity_weight,
        annotations_are_complete=annotations_are_complete,
    )
    predicted = _consecutive_labels(prediction, label="Prediction")
    context = _prepare_scoring_context(
        annotations, predicted.shape,
        overreach_distance_scale_fraction=options.overreach_distance_scale_fraction,
        seed_diameter_px=seed_diameter_px,
    )
    _, comparison = _compare_instances(
        predicted, context, false_positive_weight=overreach_weight,
        false_negative_weight=1.0, instance_penalty_weight=0.10,
        annotations_are_complete=annotations_are_complete,
        minimum_match_iou=minimum_match_iou, missed_seed_weight=missed_seed_weight,
        incorrect_concavity_weight=incorrect_concavity_weight, render=True,
    )
    assert comparison is not None
    # Compact IDs in the solver are an implementation detail, not annotation IDs.
    target_ids = np.unique(annotations)
    target_ids = target_ids[target_ids > 0]
    prediction_ids = np.unique(prediction)
    prediction_ids = prediction_ids[prediction_ids > 0]
    # Resampling can erase small IDs; use the exact same nearest-neighbour grid.
    resized = np.asarray(annotations)
    if resized.shape != predicted.shape:
        resized = cv2.resize(resized, (predicted.shape[1], predicted.shape[0]),
                             interpolation=cv2.INTER_NEAREST)
        target_ids = np.unique(resized[resized > 0])
    return replace(comparison, matched_pairs=tuple(
        (int(target_ids[a - 1]), int(prediction_ids[b - 1]))
        for a, b in comparison.matched_pairs
    ))


def score_procedural_instances(
    annotations: np.ndarray,
    prediction: np.ndarray,
    *,
    false_positive_weight: float = 2.0,
    false_negative_weight: float = 1.0,
    overreach_distance_scale_fraction: float = 0.50,
    seed_diameter_px: float | None = None,
    instance_penalty_weight: float = 0.10,
    annotations_are_complete: bool = False,
    minimum_match_iou: float = 0.20,
    missed_seed_weight: float = 0.50,
    incorrect_concavity_weight: float = 2.0,
    _scoring_context: _ProceduralFitScoringContext | None = None,
) -> ProceduralFitScore:
    """Global overlap assignment and explicit, reviewed-area-normalized costs.

    Weak incidental contacts are ignored in partial-review mode, not converted
    into large false-positive objects. Substantial unmatched fragments are
    scored; whole-dish review additionally scores every disjoint prediction.
    """
    ProceduralFitOptions(
        false_positive_weight=false_positive_weight,
        false_negative_weight=false_negative_weight,
        overreach_distance_scale_fraction=overreach_distance_scale_fraction,
        instance_penalty_weight=instance_penalty_weight,
        minimum_match_iou=minimum_match_iou, missed_seed_weight=missed_seed_weight,
        incorrect_concavity_weight=incorrect_concavity_weight,
    )
    predicted = _consecutive_labels(prediction, label="Prediction")
    context = _scoring_context or _prepare_scoring_context(
        annotations, predicted.shape,
        overreach_distance_scale_fraction=overreach_distance_scale_fraction,
        seed_diameter_px=seed_diameter_px,
    )
    if context.target.shape != predicted.shape:
        raise ValueError("Procedural fit predictions changed shape between trials.")
    score, _ = _compare_instances(
        predicted, context, false_positive_weight=false_positive_weight,
        false_negative_weight=false_negative_weight,
        instance_penalty_weight=instance_penalty_weight,
        annotations_are_complete=annotations_are_complete,
        minimum_match_iou=minimum_match_iou, missed_seed_weight=missed_seed_weight,
        incorrect_concavity_weight=incorrect_concavity_weight,
    )
    return score


def fit_procedural_settings(
    annotations: np.ndarray,
    evaluate: Callable[
        [ProceduralInstanceSettings], ProceduralInstanceResult | np.ndarray
    ],
    *,
    initial_settings: ProceduralInstanceSettings = ProceduralInstanceSettings(),
    options: ProceduralFitOptions = ProceduralFitOptions(),
    seed_diameter_px: float | None = None,
    progress: Callable[[ProceduralFitTrial, int], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> ProceduralFitResult:
    """Return the best bounded settings proposal without applying it.

    ``evaluate`` should run the procedural separator *without* passing the
    annotated instances as authoritative markers.  It may close over cached,
    already bounded evidence arrays.  The returned settings remain a proposal;
    applying them and invalidating dependent node caches is the caller's
    explicit responsibility.
    """

    annotation_array = np.asarray(annotations)
    cache: dict[ProceduralInstanceSettings, ProceduralFitScore] = {}
    trials: list[ProceduralFitTrial] = []
    scoring_context: _ProceduralFitScoringContext | None = None

    def run(
        candidate: ProceduralInstanceSettings,
        *,
        pass_number: int,
        parameter_name: str,
        parameter_value: float,
    ) -> ProceduralFitScore | None:
        nonlocal scoring_context
        if candidate in cache:
            return cache[candidate]
        if len(trials) >= options.maximum_evaluations:
            return None
        output = evaluate(candidate)
        prediction = output.labels if isinstance(output, ProceduralInstanceResult) else output
        prediction_array = np.asarray(prediction)
        if scoring_context is None:
            scoring_context = _prepare_scoring_context(
                annotation_array,
                prediction_array.shape,
                overreach_distance_scale_fraction=(
                    options.overreach_distance_scale_fraction
                ),
                seed_diameter_px=seed_diameter_px,
            )
        score = score_procedural_instances(
            annotation_array,
            prediction_array,
            false_positive_weight=options.false_positive_weight,
            false_negative_weight=options.false_negative_weight,
            overreach_distance_scale_fraction=(
                options.overreach_distance_scale_fraction
            ),
            seed_diameter_px=seed_diameter_px,
            instance_penalty_weight=options.instance_penalty_weight,
            annotations_are_complete=options.annotations_are_complete,
            minimum_match_iou=options.minimum_match_iou,
            missed_seed_weight=options.missed_seed_weight,
            incorrect_concavity_weight=options.incorrect_concavity_weight,
            _scoring_context=scoring_context,
        )
        cache[candidate] = score
        trial = ProceduralFitTrial(
            evaluation_number=len(trials) + 1,
            pass_number=pass_number,
            parameter_name=parameter_name,
            parameter_value=float(parameter_value),
            score=score,
        )
        trials.append(trial)
        if progress is not None:
            progress(trial, options.maximum_evaluations)
        return score

    initial_score = run(
        initial_settings,
        pass_number=0,
        parameter_name="initial",
        parameter_value=0.0,
    )
    assert initial_score is not None
    current_settings = initial_settings
    current_score = initial_score
    was_cancelled = False
    if current_score.perfect_fit:
        return ProceduralFitResult(
            initial_settings=initial_settings,
            proposed_settings=current_settings,
            initial_score=initial_score,
            proposed_score=current_score,
            trials=tuple(trials),
        )

    for pass_index in range(options.passes):
        for parameter in options.parameters:
            if cancelled is not None and cancelled():
                was_cancelled = True
                break
            if len(trials) >= options.maximum_evaluations:
                break
            current_value = float(getattr(current_settings, parameter.name))
            step = parameter.initial_step * options.step_decay**pass_index
            best_settings = current_settings
            best_score = current_score
            for direction in (-1.0, 1.0):
                candidate_value = float(
                    np.clip(
                        current_value + direction * step,
                        parameter.minimum,
                        parameter.maximum,
                    )
                )
                if abs(candidate_value - current_value) <= 1e-12:
                    continue
                candidate = replace(
                    current_settings,
                    **{parameter.name: candidate_value},
                )
                candidate_score = run(
                    candidate,
                    pass_number=pass_index + 1,
                    parameter_name=parameter.name,
                    parameter_value=candidate_value,
                )
                if candidate_score is None:
                    break
                if candidate_score.loss < best_score.loss - 1e-12:
                    best_settings = candidate
                    best_score = candidate_score
                if best_score.perfect_fit:
                    break
            current_settings = best_settings
            current_score = best_score
            if current_score.perfect_fit:
                break
        if was_cancelled or len(trials) >= options.maximum_evaluations:
            break
        if current_score.perfect_fit:
            break

    return ProceduralFitResult(
        initial_settings=initial_settings,
        proposed_settings=current_settings,
        initial_score=initial_score,
        proposed_score=current_score,
        trials=tuple(trials),
        cancelled=was_cancelled,
    )


__all__ = [
    "DEFAULT_PROCEDURAL_FIT_PARAMETERS",
    "ProceduralFitOptions",
    "ProceduralFitParameter",
    "ProceduralFitResult",
    "ProceduralFitScore",
    "ProceduralFitTrial",
    "fit_procedural_settings",
    "score_procedural_instances",
]
