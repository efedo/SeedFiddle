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
    ProceduralFitParameter("minimum_instance_area_fraction", 0.06, 0.48, 0.07),
)

_FITTABLE_PARAMETER_NAMES = {
    # Occupancy support.
    "foreground_threshold_scale",
    "occupancy_closing_fraction",
    "occupancy_hole_area_fraction",
    "dish_margin_fraction",
    "reference_texture_weight",
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
}


@dataclass(frozen=True, slots=True)
class ProceduralFitOptions:
    """Bounds and loss weights for a short coordinate-search fit."""

    maximum_evaluations: int = 33
    passes: int = 2
    step_decay: float = 0.50
    false_positive_weight: float = 2.0
    false_negative_weight: float = 1.0
    overreach_distance_scale_fraction: float = 0.50
    instance_penalty_weight: float = 0.10
    annotations_are_complete: bool = False
    parameters: tuple[ProceduralFitParameter, ...] = DEFAULT_PROCEDURAL_FIT_PARAMETERS

    def __post_init__(self) -> None:
        if self.maximum_evaluations < 1:
            raise ValueError("At least one procedural fit evaluation is required.")
        if self.passes < 1:
            raise ValueError("At least one procedural fit pass is required.")
        if not 0.0 < self.step_decay <= 1.0:
            raise ValueError("Procedural fit step decay must be in (0, 1].")
        if self.false_negative_weight <= 0.0:
            raise ValueError("False-negative weight must be positive.")
        if self.false_positive_weight <= self.false_negative_weight:
            raise ValueError(
                "False-positive weight must exceed the false-negative weight."
            )
        if not isfinite(self.overreach_distance_scale_fraction) or (
            self.overreach_distance_scale_fraction <= 0.0
        ):
            raise ValueError(
                "Overreach distance scale fraction must be positive."
            )
        if self.instance_penalty_weight < 0.0:
            raise ValueError("Instance penalty weight must be non-negative.")
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

    @property
    def perfect_fit(self) -> bool:
        return self.loss <= 1e-12


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


def _distance_weighted_overreach(
    target: np.ndarray,
    predicted: np.ndarray,
    *,
    accepted: list[tuple[int, int, int]],
    false_positive_prediction_ids: set[int],
    target_bounds: np.ndarray,
    distance_to_annotation: np.ndarray,
    prediction_count: int,
    distance_scale_px: float,
) -> tuple[float, float]:
    """Return exponential FP-pixel equivalents and their scale in pixels.

    Leakage from a matched component is measured from *its own* annotated
    seed, rather than from the union of annotations.  This ensures that a
    component merging into an adjacent annotated seed remains costly.  An
    unmatched evaluated component is measured from the nearest annotation.
    """

    if not accepted and not false_positive_prediction_ids:
        return 0.0, distance_scale_px

    prediction_bounds = _label_bounds(predicted, prediction_count)
    weighted_pixels = 0.0

    for target_id, prediction_id, _intersection in accepted:
        tx0, ty0, tx1, ty1 = target_bounds[target_id]
        px0, py0, px1, py1 = prediction_bounds[prediction_id]
        x0, y0 = min(tx0, px0), min(ty0, py0)
        x1, y1 = max(tx1, px1), max(ty1, py1)
        target_roi = target[y0:y1, x0:x1]
        predicted_roi = predicted[y0:y1, x0:x1]
        false_positive = (predicted_roi == prediction_id) & (
            target_roi != target_id
        )
        if not np.any(false_positive):
            continue
        distance = cv2.distanceTransform(
            np.asarray(target_roi != target_id, dtype=np.uint8),
            cv2.DIST_L2,
            cv2.DIST_MASK_PRECISE,
        )
        weighted_pixels += float(
            np.sum(
                _overreach_multiplier(
                    distance[false_positive], distance_scale_px
                ),
                dtype=np.float64,
            )
        )

    if false_positive_prediction_ids:
        for prediction_id in false_positive_prediction_ids:
            px0, py0, px1, py1 = prediction_bounds[prediction_id]
            prediction_roi = predicted[py0:py1, px0:px1] == prediction_id
            distances = distance_to_annotation[py0:py1, px0:px1][prediction_roi]
            weighted_pixels += float(
                np.sum(
                    _overreach_multiplier(distances, distance_scale_px),
                    dtype=np.float64,
                )
            )
    return weighted_pixels, distance_scale_px


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
    _scoring_context: _ProceduralFitScoringContext | None = None,
) -> ProceduralFitScore:
    """Score a prediction with exponentially distance-weighted overreach.

    Instance identifiers need not correspond.  A deterministic greedy
    one-to-one overlap match is used.  An unmatched predicted fragment that
    overlaps an annotated seed is still scored as a false-positive instance;
    a wholly disjoint prediction is evaluated only when annotations are marked
    complete.
    """

    if false_negative_weight <= 0.0:
        raise ValueError("False-negative weight must be positive.")
    if false_positive_weight <= false_negative_weight:
        raise ValueError(
            "False-positive weight must exceed the false-negative weight."
        )
    if not isfinite(overreach_distance_scale_fraction) or (
        overreach_distance_scale_fraction <= 0.0
    ):
        raise ValueError("Overreach distance scale fraction must be positive.")
    if instance_penalty_weight < 0.0:
        raise ValueError("Instance penalty weight must be non-negative.")

    predicted = _consecutive_labels(prediction, label="Prediction")
    context = _scoring_context or _prepare_scoring_context(
        annotations,
        predicted.shape,
        overreach_distance_scale_fraction=overreach_distance_scale_fraction,
        seed_diameter_px=seed_diameter_px,
    )
    if context.target.shape != predicted.shape:
        raise ValueError("Procedural fit predictions changed shape between trials.")
    target = context.target
    target_count = context.target_count
    prediction_count = int(predicted.max(initial=0))
    target_areas = context.target_areas
    prediction_areas = np.bincount(
        predicted.reshape(-1), minlength=prediction_count + 1
    )

    overlap = (target > 0) & (predicted > 0)
    if np.any(overlap):
        stride = prediction_count + 1
        codes, intersections = np.unique(
            target[overlap].astype(np.int64) * stride + predicted[overlap],
            return_counts=True,
        )
        target_ids = (codes // stride).astype(np.int32)
        prediction_ids = (codes % stride).astype(np.int32)
        pairs = []
        for target_id, prediction_id, intersection in zip(
            target_ids, prediction_ids, intersections, strict=True
        ):
            union = (
                int(target_areas[target_id])
                + int(prediction_areas[prediction_id])
                - int(intersection)
            )
            pairs.append(
                (
                    int(intersection),
                    float(intersection) / max(1, union),
                    int(target_id),
                    int(prediction_id),
                )
            )
    else:
        pairs = []

    matched_target: set[int] = set()
    matched_prediction: set[int] = set()
    accepted: list[tuple[int, int, int]] = []
    for intersection, iou, target_id, prediction_id in sorted(
        pairs,
        key=lambda item: (-item[0], -item[1], item[2], item[3]),
    ):
        if target_id in matched_target or prediction_id in matched_prediction:
            continue
        matched_target.add(target_id)
        matched_prediction.add(prediction_id)
        accepted.append((target_id, prediction_id, intersection))

    true_positive_pixels = 0
    false_positive_pixels = 0
    false_negative_pixels = 0
    for target_id, prediction_id, intersection in accepted:
        true_positive_pixels += intersection
        false_positive_pixels += int(prediction_areas[prediction_id]) - intersection
        false_negative_pixels += int(target_areas[target_id]) - intersection
    for target_id in range(1, target_count + 1):
        if target_id not in matched_target:
            false_negative_pixels += int(target_areas[target_id])

    overlapping_predictions = {pair[3] for pair in pairs}
    evaluated_predictions = (
        set(range(1, prediction_count + 1))
        if annotations_are_complete
        else overlapping_predictions
    )
    false_positive_prediction_ids = evaluated_predictions - matched_prediction
    for prediction_id in false_positive_prediction_ids:
        false_positive_pixels += int(prediction_areas[prediction_id])

    false_positive_instances = len(false_positive_prediction_ids)
    false_negative_instances = target_count - len(matched_target)
    (
        distance_weighted_false_positive_pixels,
        overreach_distance_scale_px,
    ) = _distance_weighted_overreach(
        target,
        predicted,
        accepted=accepted,
        false_positive_prediction_ids=false_positive_prediction_ids,
        target_bounds=context.target_bounds,
        distance_to_annotation=context.distance_to_annotation,
        prediction_count=prediction_count,
        distance_scale_px=context.overreach_distance_scale_px,
    )
    weighted_pixel_error = (
        false_positive_weight * distance_weighted_false_positive_pixels
        + false_negative_weight * false_negative_pixels
    )
    # Tversky-style error with target-distance-weighted overreach; exact
    # one-to-one agreement remains the unique zero.
    pixel_loss = weighted_pixel_error / max(
        1.0, true_positive_pixels + weighted_pixel_error
    )
    weighted_instance_error = instance_penalty_weight * (
        false_positive_weight * false_positive_instances
        + false_negative_weight * false_negative_instances
    ) / max(1, target_count)
    pixel_precision = true_positive_pixels / max(
        1, true_positive_pixels + false_positive_pixels
    )
    pixel_recall = true_positive_pixels / max(
        1, true_positive_pixels + false_negative_pixels
    )
    return ProceduralFitScore(
        loss=float(pixel_loss + weighted_instance_error),
        true_positive_pixels=int(true_positive_pixels),
        false_positive_pixels=int(false_positive_pixels),
        false_negative_pixels=int(false_negative_pixels),
        matched_instances=len(matched_target),
        annotated_instances=target_count,
        evaluated_predictions=len(evaluated_predictions),
        false_positive_instances=false_positive_instances,
        false_negative_instances=false_negative_instances,
        pixel_precision=float(pixel_precision),
        pixel_recall=float(pixel_recall),
        distance_weighted_false_positive_pixels=float(
            distance_weighted_false_positive_pixels
        ),
        overreach_distance_scale_px=float(overreach_distance_scale_px),
    )


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
