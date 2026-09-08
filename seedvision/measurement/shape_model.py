"""Reviewed-seed measurements and hierarchical 2-D dimensions/shape models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np

from seedvision.measurement.geometry import SeedShapeMeasurement, measure_shape_mask
from seedvision.persistence.reference_regions import SeedInstanceAnnotation
from seedvision.reference_library.contracts import (
    BiologicalContext,
    SHAPE_MEASUREMENT_NAMES,
    ShapeObservation,
    ShapePopulationComponent,
    SpeciesDimensionsShapeBank,
    SpeciesShapeSummary,
)


@dataclass(frozen=True, slots=True)
class ReviewedSeedShape:
    seed_id: int
    annotation: SeedInstanceAnnotation
    measurement: SeedShapeMeasurement | None
    eligible: bool
    exclusion_reason: str | None = None
    length_eligible: bool = False


@dataclass(frozen=True, slots=True)
class SeedMeasurementSummary:
    observations: tuple[ReviewedSeedShape, ...]
    eligible_count: int
    calibrated_count: int
    length_unit: str
    measurement_names: tuple[str, ...] = SHAPE_MEASUREMENT_NAMES
    boundary_turn_degrees: tuple[float, ...] = ()
    boundary_curvature_times_diameter: tuple[float, ...] = ()

    @property
    def size_observations(self):
        return tuple(item for item in self.observations
                     if (item.length_eligible or item.eligible) and item.measurement is not None)

    @property
    def size_statistics(self) -> tuple[float, float, float] | None:
        """Mean, between-seed SD and uncertainty of mean (shared scale retained)."""
        items = self.size_observations
        if not items:
            return None
        widths = np.asarray([item.measurement.maximum_span for item in items])
        sd = float(np.std(widths, ddof=1)) if len(items) > 1 else 0.0
        # Do not divide shared calibration or annotation systematic uncertainty
        # by sqrt(n). Sampling variability alone decreases with sample count.
        measurement_error = float(np.mean([
            item.measurement.measurement_uncertainty[0] ** 2 for item in items]))
        return float(widths.mean()), sd, float(np.sqrt(sd ** 2 / len(items) + measurement_error))

    @property
    def mean_internal_concavity(self) -> float | None:
        values = [item.measurement.concavity_fraction for item in self.observations
                  if item.eligible and item.measurement is not None]
        return float(np.mean(values)) if values else None

    @property
    def vectors(self) -> np.ndarray:
        values = [
            item.measurement.vector()
            for item in self.observations
            if item.eligible and item.measurement is not None
        ]
        return np.asarray(values, np.float64).reshape(
            len(values), len(self.measurement_names)
        )


@dataclass(frozen=True, slots=True)
class SeedPoseShapeFamily:
    pose: str
    component: ShapePopulationComponent
    local_observation_count: int
    source: str
    prior_probability: float = 0.0
    out_of_family_distance: float = 0.0


@dataclass(frozen=True, slots=True)
class ShapeCompatibilityAssessment:
    pose: str
    compatibility: float
    standardized_distance: float
    out_of_family: bool


@dataclass(frozen=True, slots=True)
class SeedDimensionsShapeModel:
    families: tuple[SeedPoseShapeFamily, ...]
    measurement_summary: SeedMeasurementSummary
    source_mode: str
    selected_hierarchy_path: tuple[str, ...]
    legacy_processing_diameter_px: float
    compatibility_seed_diameter_px: float
    in_sample_local_update: bool = False
    warnings: tuple[str, ...] = ()

    def family(self, pose: str) -> SeedPoseShapeFamily | None:
        return next((item for item in self.families if item.pose == pose), None)

    def assess_pose_hypotheses(
        self, measurement: Iterable[float]
    ) -> tuple[ShapeCompatibilityAssessment, ...]:
        return tuple(
            assess_candidate_shape(measurement, family.component)
            for family in self.families
        )


def measure_reviewed_seed_instances(
    labels: np.ndarray | None,
    annotations: Iterable[SeedInstanceAnnotation],
    *,
    pixels_per_mm: float | None = None,
    calibration_relative_uncertainty: float = 0.0,
    contour_samples: int = 128,
    boundary_perturbation_radius_px: int = 2,
) -> SeedMeasurementSummary:
    """Full outlines inform shape; explicitly visible full spans also inform size.

    Partial masks never enter ovality, contour or concavity population fitting.
    """

    if labels is None:
        return SeedMeasurementSummary((), 0, 0, "px")
    if not 1 <= boundary_perturbation_radius_px <= 8:
        raise ValueError("Boundary perturbation radius must be between 1 and 8 pixels.")
    values = np.asarray(labels)
    if values.ndim != 2 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError("Seed instance labels must be a two-dimensional integer raster.")
    by_id = {int(item.seed_id): item for item in annotations}
    results = []
    boundary_turns, boundary_curvatures = [], []
    for seed_id in (int(value) for value in np.unique(values) if value > 0):
        annotation = by_id.get(seed_id, SeedInstanceAnnotation(seed_id))
        reason = _shape_ineligibility_reason(annotation)
        length_eligible = (
            (annotation.outline_visibility == "complete" or annotation.full_length_visible)
            and not annotation.shape_exclusion_reason
        )
        measurement = None
        if reason is None or length_eligible:
            mask = values == seed_id
            if _touches_image_edge(mask) and not annotation.full_length_visible:
                reason = "mask touches image edge"
                length_eligible = False
            elif _meaningfully_disconnected(mask):
                reason = "mask is disconnected"
                length_eligible = False
            else:
                try:
                    measurement = measure_shape_mask(
                        mask,
                        pixels_per_mm=pixels_per_mm,
                        calibration_relative_uncertainty=(
                            calibration_relative_uncertainty
                        ),
                        hilum_point=annotation.hilum_point,
                        contour_samples=contour_samples,
                        perturbation_radii=tuple(
                            range(1, boundary_perturbation_radius_px + 1)
                        ),
                    )
                except ValueError as error:
                    reason = str(error)
                    length_eligible = False
                if measurement is not None and reason is None:
                    turns, curvatures = _boundary_distribution(mask, contour_samples)
                    boundary_turns.extend(turns)
                    boundary_curvatures.extend(curvatures)
        results.append(
            ReviewedSeedShape(
                seed_id,
                annotation,
                measurement,
                measurement is not None and reason is None,
                reason,
                length_eligible and measurement is not None,
            )
        )
    eligible = sum(item.eligible for item in results)
    return SeedMeasurementSummary(
        tuple(results),
        eligible,
        eligible if pixels_per_mm is not None else 0,
        "mm" if pixels_per_mm is not None else "px",
        boundary_turn_degrees=tuple(boundary_turns),
        boundary_curvature_times_diameter=tuple(boundary_curvatures),
    )


def _boundary_distribution(mask, sample_count):
    """Equal arc-length samples, smoothed before turning; seed-balanced samples.

    Positive curvature denotes convex turning regardless of contour winding.
    k*maximum-span is scale independent. Raw image orientation is not a prior.
    """
    import cv2
    rows, columns = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
    mask = mask[rows[0]:rows[-1]+1, columns[0]:columns[-1]+1]
    contours, _ = cv2.findContours(np.asarray(mask, np.uint8), cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_NONE)
    points = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(float)
    closed = np.vstack((points, points[:1]))
    distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(closed, axis=0), axis=1))]
    positions = np.linspace(0., distance[-1], sample_count, endpoint=False)
    sampled = np.column_stack([np.interp(positions, distance, closed[:, i]) for i in (0, 1)])
    sampled = (np.roll(sampled, 2, 0) + 4*np.roll(sampled, 1, 0) + 6*sampled
               + 4*np.roll(sampled, -1, 0) + np.roll(sampled, -2, 0)) / 16.
    incoming = sampled - np.roll(sampled, 1, 0)
    outgoing = np.roll(sampled, -1, 0) - sampled
    cross = incoming[:, 0]*outgoing[:, 1] - incoming[:, 1]*outgoing[:, 0]
    turns = np.arctan2(cross, np.sum(incoming*outgoing, axis=1))
    turns *= 1. if turns.sum() >= 0 else -1.
    span = np.max(np.linalg.norm(sampled[:, None] - sampled[None], axis=2))
    step = (np.linalg.norm(incoming, axis=1) + np.linalg.norm(outgoing, axis=1)) / 2
    return np.degrees(turns).tolist(), (turns*span/np.maximum(step, 1e-6)).tolist()


def shape_observations_from_summary(
    summary: SeedMeasurementSummary,
    *,
    source_index: int,
    biological_context: BiologicalContext,
) -> tuple[ShapeObservation, ...]:
    path = biological_context.hierarchy_path()
    values = []
    for item in summary.observations:
        if not item.eligible or item.measurement is None:
            continue
        measurement = item.measurement
        values.append(
            ShapeObservation(
                source_index=source_index,
                seed_id=item.seed_id,
                physical_seed_id=item.annotation.physical_seed_id,
                hierarchy_path=path,
                pose=item.annotation.pose,
                visibility=item.annotation.outline_visibility,
                calibrated=measurement.length_unit == "mm",
                measurement=measurement.vector(),
                measurement_uncertainty=measurement.measurement_uncertainty,
                contour_signature=measurement.contour_signature,
            )
        )
    return tuple(values)


def build_species_shape_summary(
    observations: Iterable[ShapeObservation],
) -> SpeciesShapeSummary:
    """Create the diagnostic-only summary without fitting a procedural prior."""

    return SpeciesShapeSummary(tuple(observations))


def fit_species_dimensions_shape_bank(
    observations: Iterable[ShapeObservation],
    *,
    minimum_component_seeds: int = 2,
    shrinkage_seed_count: float = 5.0,
    maximum_contour_modes: int = 4,
) -> SpeciesDimensionsShapeBank:
    """Fit robust pose-conditioned hierarchy components with partial pooling.

    Physical dimensions are fitted only from calibrated observations. Repeated
    views sharing ``physical_seed_id`` contribute one biological group mean per
    pose and hierarchy path. Uncalibrated observations can still strengthen the
    dimensionless ovality/contour family; they never enter millimetre fields.
    """

    values = tuple(observations)
    if minimum_component_seeds < 1 or shrinkage_seed_count < 0:
        raise ValueError("Shape component support and shrinkage must be non-negative.")
    groups: dict[tuple[tuple[str, ...], str], list[ShapeObservation]] = {}
    for item in values:
        for stop in range(1, len(item.hierarchy_path) + 1):
            groups.setdefault((item.hierarchy_path[:stop], item.pose), []).append(item)
    components: dict[tuple[tuple[str, ...], str], ShapePopulationComponent] = {}
    for path_length in sorted({len(path) for path, _pose in groups}):
        for (path, pose), records in sorted(groups.items()):
            if len(path) != path_length:
                continue
            calibrated = _physical_seed_groups(
                records, calibrated_only=True
            )
            all_shapes = _physical_seed_groups(records)
            if len(all_shapes[0]) < minimum_component_seeds:
                continue
            parent = components.get((path[:-1], pose)) if len(path) > 1 else None
            if len(calibrated[0]) >= minimum_component_seeds:
                physical_parent = (
                    parent
                    if parent is not None and parent.physical_dimensions_available
                    else None
                )
                component = _fit_component(
                    path,
                    pose,
                    calibrated[0],
                    calibrated[1],
                    calibrated[2],
                    calibrated[3],
                    parent=physical_parent,
                    shrinkage_seed_count=shrinkage_seed_count,
                    maximum_contour_modes=maximum_contour_modes,
                    physical_dimensions_available=True,
                    calibrated_seed_count=len(calibrated[0]),
                    dimensionless_seed_count=len(all_shapes[0]),
                )
                component = _augment_dimensionless_shape(
                    component,
                    all_shapes[0],
                    all_shapes[1],
                    all_shapes[2],
                    maximum_contour_modes=maximum_contour_modes,
                    parent=parent,
                    shrinkage_seed_count=shrinkage_seed_count,
                )
            else:
                normalized_vectors, normalized_uncertainties = (
                    _dimensionless_vectors(all_shapes[0], all_shapes[2])
                )
                component = _fit_component(
                    path,
                    pose,
                    normalized_vectors,
                    all_shapes[1],
                    normalized_uncertainties,
                    all_shapes[3],
                    parent=parent,
                    shrinkage_seed_count=shrinkage_seed_count,
                    maximum_contour_modes=maximum_contour_modes,
                    physical_dimensions_available=False,
                    calibrated_seed_count=len(calibrated[0]),
                    dimensionless_seed_count=len(all_shapes[0]),
                )
            components[(path, pose)] = component
    return SpeciesDimensionsShapeBank(
        values,
        tuple(components.values()),
        minimum_component_seeds=minimum_component_seeds,
        shrinkage_seed_count=shrinkage_seed_count,
        maximum_contour_modes=maximum_contour_modes,
    )


def assemble_seed_dimensions_shape_model(
    *,
    local_summary: SeedMeasurementSummary,
    legacy_processing_diameter_px: float,
    source_mode: str,
    biological_context: BiologicalContext | None = None,
    library_bank: SpeciesDimensionsShapeBank | None = None,
    minimum_component_seeds: int = 2,
    shrinkage_seed_count: float = 5.0,
    maximum_contour_modes: int = 4,
) -> SeedDimensionsShapeModel:
    """Resolve current/library/prior+local model without double-counting sources."""

    if legacy_processing_diameter_px <= 0:
        raise ValueError("Legacy processing diameter must be positive.")
    if minimum_component_seeds < 1:
        raise ValueError("Minimum component seeds must be positive.")
    if shrinkage_seed_count < 0 or maximum_contour_modes < 0:
        raise ValueError("Shape shrinkage and contour-mode limits must be non-negative.")
    local_observations = tuple(
        item.measurement
        for item in local_summary.observations
        if item.eligible and item.measurement is not None
    )
    poses = ("flat", "oblique", "side", "uncertain")
    families = []
    selected_path: tuple[str, ...] = ()
    warnings = []
    local_can_update_physical = local_summary.length_unit == "mm"
    in_sample = False
    for pose in poses:
        component = None
        path = ()
        component_source = "library"
        if library_bank is not None and biological_context is not None and source_mode != "current_image_only":
            component, path = library_bank.select_component(biological_context, pose)
            if path and len(path) > len(selected_path):
                selected_path = path
        local_pose = [
            item.measurement
            for item in local_summary.observations
            if item.eligible
            and item.measurement is not None
            and item.annotation.pose == pose
        ]
        if source_mode == "current_image_only":
            component = _local_component(
                local_pose,
                pose,
                minimum_component_seeds=minimum_component_seeds,
                maximum_contour_modes=maximum_contour_modes,
            )
            component_source = "local"
        elif (
            source_mode == "species_library_prior_and_current"
            and len(local_pose) >= minimum_component_seeds
            and (component is None or local_can_update_physical)
        ):
            component = _update_component(
                component,
                local_pose,
                pose,
                shrinkage_seed_count=shrinkage_seed_count,
                maximum_contour_modes=maximum_contour_modes,
            )
            in_sample = True
            component_source = "library+local"
        if component is not None:
            families.append(
                SeedPoseShapeFamily(
                    pose,
                    component,
                    len(local_pose),
                    component_source,
                )
            )
    total_support = sum(
        item.component.effective_physical_seed_count for item in families
    )
    families = [
        SeedPoseShapeFamily(
            item.pose,
            item.component,
            item.local_observation_count,
            item.source,
            (
                item.component.effective_physical_seed_count / total_support
                if total_support > 0
                else 0.0
            ),
            _out_of_family_distance(len(item.component.mean)),
        )
        for item in families
    ]
    # Migration contract: the scalar compatibility port remains exactly the
    # historical processing diameter until each consumer opts into a specific
    # distribution quantity. The richer model travels through separate ports.
    compatibility = legacy_processing_diameter_px
    if source_mode != "current_image_only" and not families:
        warnings.append("No compatible library shape component is available.")
    if (
        source_mode == "species_library_prior_and_current"
        and local_observations
        and not local_can_update_physical
    ):
        warnings.append(
            "Current reviewed shapes are uncalibrated; they remain diagnostic "
            "and do not update the library's physical-dimension prior."
        )
    return SeedDimensionsShapeModel(
        tuple(families), local_summary, source_mode, selected_path,
        float(legacy_processing_diameter_px), float(compatibility),
        in_sample_local_update=in_sample, warnings=tuple(warnings),
    )


def candidate_shape_compatibility(
    measurement: Iterable[float], component: ShapePopulationComponent
) -> float:
    """Robust joint compatibility score; not advertised as a posterior."""

    values = np.asarray(tuple(measurement), np.float64)
    if not component.physical_dimensions_available:
        values, _uncertainty = _dimensionless_vectors(
            values[None], np.zeros_like(values)[None]
        )
        values = values[0]
    mean = np.asarray(component.mean, np.float64)
    covariance = np.asarray(component.covariance, np.float64)
    if values.shape != mean.shape:
        raise ValueError("Candidate measurement does not match the shape model.")
    indices = (
        np.arange(len(mean))
        if component.physical_dimensions_available
        else _DIMENSIONLESS_MEASUREMENT_INDICES
    )
    selected_covariance = covariance[np.ix_(indices, indices)]
    regularized = selected_covariance + np.eye(len(indices)) * max(
        1e-8, np.trace(selected_covariance) * 1e-6
    )
    delta = values[indices] - mean[indices]
    distance = float(delta @ np.linalg.pinv(regularized) @ delta)
    return float(np.exp(-0.5 * min(distance, 80.0) / max(len(indices), 1)))


def assess_candidate_shape(
    measurement: Iterable[float], component: ShapePopulationComponent
) -> ShapeCompatibilityAssessment:
    """Return a pose-specific compatibility and explicit abstention decision."""

    values = np.asarray(tuple(measurement), np.float64)
    if not component.physical_dimensions_available:
        values, _uncertainty = _dimensionless_vectors(
            values[None], np.zeros_like(values)[None]
        )
        values = values[0]
    mean = np.asarray(component.mean, np.float64)
    covariance = np.asarray(component.covariance, np.float64)
    if values.shape != mean.shape:
        raise ValueError("Candidate measurement does not match the shape model.")
    indices = (
        np.arange(len(mean))
        if component.physical_dimensions_available
        else _DIMENSIONLESS_MEASUREMENT_INDICES
    )
    selected_covariance = covariance[np.ix_(indices, indices)]
    regularized = selected_covariance + np.eye(len(indices)) * max(
        1e-8, np.trace(selected_covariance) * 1e-6
    )
    delta = values[indices] - mean[indices]
    distance = float(
        delta @ np.linalg.pinv(regularized) @ delta
    )
    threshold = _out_of_family_distance(len(indices))
    return ShapeCompatibilityAssessment(
        component.pose,
        float(np.exp(-0.5 * min(distance, 80.0) / max(len(indices), 1))),
        distance,
        distance > threshold,
    )


def _shape_ineligibility_reason(annotation: SeedInstanceAnnotation) -> str | None:
    if not annotation.shape_reviewed:
        return "shape reference not reviewed"
    if annotation.outline_visibility != "complete":
        return f"outline visibility is {annotation.outline_visibility}"
    if annotation.pose not in {"flat", "oblique", "side", "uncertain"}:
        return "pose is not explicitly reviewed"
    if annotation.shape_exclusion_reason:
        return annotation.shape_exclusion_reason
    return None


def _touches_image_edge(mask: np.ndarray) -> bool:
    return bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())


def _meaningfully_disconnected(mask: np.ndarray) -> bool:
    import cv2

    count, _labels, statistics, _centroids = cv2.connectedComponentsWithStats(
        np.asarray(mask, np.uint8), connectivity=8
    )
    areas = sorted((int(value) for value in statistics[1:, cv2.CC_STAT_AREA]), reverse=True)
    return count > 2 and len(areas) > 1 and areas[1] > max(4, areas[0] * 0.01)


def _physical_seed_groups(records, *, calibrated_only=False):
    grouped: dict[str, list[ShapeObservation]] = {}
    selected_sources: set[int] = set()
    for item in records:
        if calibrated_only and not item.calibrated:
            continue
        selected_sources.add(item.source_index)
        key = item.physical_seed_id or f"source:{item.source_index}:seed:{item.seed_id}"
        grouped.setdefault(key, []).append(item)
    vectors = []
    contours = []
    uncertainties = []
    for values in grouped.values():
        vectors.append(np.mean([item.measurement for item in values], axis=0))
        uncertainties.append(
            np.sqrt(
                np.mean(
                    np.square(
                        [item.measurement_uncertainty for item in values]
                    ),
                    axis=0,
                )
            )
        )
        signatures = [item.contour_signature for item in values if item.contour_signature]
        if signatures:
            contours.append(np.mean(signatures, axis=0))
    return (
        np.asarray(vectors, np.float64),
        np.asarray(contours, np.float64),
        np.asarray(uncertainties, np.float64),
        len(selected_sources),
    )


_PHYSICAL_MEASUREMENT_INDICES = np.asarray((0, 1, 2, 4), np.int64)
_DIMENSIONLESS_MEASUREMENT_INDICES = np.asarray(
    tuple(
        index
        for index in range(len(SHAPE_MEASUREMENT_NAMES))
        if index not in {0, 1, 2, 4}
    ),
    np.int64,
)


def _dimensionless_vectors(vectors, uncertainties):
    """Normalize size-bearing fields while preserving true shape quantities.

    The normalized size fields are internal placeholders required by the fixed
    joint-vector schema. ``physical_dimensions_available=False`` prevents them
    from being presented or consumed as millimetres.
    """

    values = np.asarray(vectors, np.float64).copy()
    errors = np.asarray(uncertainties, np.float64).copy()
    if (
        values.ndim != 2
        or values.shape[1] != len(SHAPE_MEASUREMENT_NAMES)
        or errors.shape != values.shape
    ):
        raise ValueError(
            "Dimensionless shape vectors must use the current measurement schema."
        )
    span = np.maximum(values[:, 0], 1e-9)
    errors[:, 1] /= span
    errors[:, 2] /= span
    errors[:, 4] /= np.square(span)
    values[:, 1] /= span
    values[:, 2] /= span
    values[:, 4] /= np.square(span)
    values[:, 0] = 1.0
    errors[:, 0] = 0.0
    return values, errors


def _augment_dimensionless_shape(
    component,
    vectors,
    contours,
    uncertainties,
    *,
    maximum_contour_modes,
    parent,
    shrinkage_seed_count,
):
    """Use every physical seed for dimensionless fields of a calibrated fit."""

    normalized, normalized_uncertainty = _dimensionless_vectors(
        vectors, uncertainties
    )
    auxiliary = _fit_component(
        component.hierarchy_path,
        component.pose,
        normalized,
        contours,
        normalized_uncertainty,
        component.source_count,
        parent=parent,
        shrinkage_seed_count=shrinkage_seed_count,
        maximum_contour_modes=maximum_contour_modes,
        physical_dimensions_available=False,
        calibrated_seed_count=component.calibrated_physical_seed_count,
        dimensionless_seed_count=len(normalized),
    )
    indices = _DIMENSIONLESS_MEASUREMENT_INDICES
    mean = np.asarray(component.mean, np.float64).copy()
    uncertainty = np.asarray(component.mean_uncertainty, np.float64).copy()
    predictive = np.asarray(component.predictive_interval_95, np.float64).copy()
    covariance = np.asarray(component.covariance, np.float64).copy()
    auxiliary_covariance = np.asarray(auxiliary.covariance, np.float64)
    mean[indices] = np.asarray(auxiliary.mean)[indices]
    uncertainty[indices] = np.asarray(auxiliary.mean_uncertainty)[indices]
    predictive[indices] = np.asarray(auxiliary.predictive_interval_95)[indices]
    covariance[np.ix_(indices, indices)] = auxiliary_covariance[np.ix_(indices, indices)]
    # Retain calibrated cross-covariances, then project the assembled symmetric
    # matrix back to the positive-semidefinite cone for stable Mahalanobis use.
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    covariance = (eigenvectors * np.maximum(eigenvalues, 1e-10)) @ eigenvectors.T
    return ShapePopulationComponent(
        component.hierarchy_path,
        component.pose,
        tuple(mean),
        tuple(tuple(row) for row in covariance),
        tuple(uncertainty),
        tuple(tuple(row) for row in predictive),
        float(len(normalized)),
        max(component.source_count, auxiliary.source_count),
        auxiliary.contour_mean,
        auxiliary.contour_modes,
        auxiliary.contour_variances,
        True,
        component.calibrated_physical_seed_count,
        float(len(normalized)),
    )


def _fit_component(
    path,
    pose,
    vectors,
    contours,
    uncertainties,
    source_count,
    *,
    parent,
    shrinkage_seed_count,
    maximum_contour_modes,
    physical_dimensions_available=True,
    calibrated_seed_count=None,
    dimensionless_seed_count=None,
):
    centre = np.median(vectors, axis=0)
    deviations = np.abs(vectors - centre)
    scale = np.maximum(1e-8, 1.4826 * np.median(deviations, axis=0))
    standardized = np.clip((vectors - centre) / scale, -4.0, 4.0)
    robust_vectors = centre + standardized * scale
    covariance = (
        np.cov(robust_vectors, rowvar=False)
        if len(vectors) > 1
        else np.diag(np.square(scale))
    )
    covariance = np.atleast_2d(covariance) + np.diag(np.square(scale) * 0.05)
    measurement_variance = (
        np.mean(np.square(uncertainties), axis=0)
        if np.asarray(uncertainties).shape == vectors.shape
        else np.zeros(len(centre), np.float64)
    )
    if parent is not None and shrinkage_seed_count > 0:
        local_weight = len(vectors) / (len(vectors) + shrinkage_seed_count)
        centre = local_weight * centre + (1.0 - local_weight) * np.asarray(parent.mean)
        covariance = local_weight * covariance + (1.0 - local_weight) * np.asarray(parent.covariance)
    standard_error = np.sqrt(
        np.maximum(np.diag(covariance), 0.0) / max(len(vectors), 1)
        + measurement_variance / max(source_count, 1)
    )
    predictive_scale = np.sqrt(
        np.maximum(np.diag(covariance), 0.0) + measurement_variance
    )
    predictive = np.column_stack(
        (centre - 1.96 * predictive_scale, centre + 1.96 * predictive_scale)
    )
    contour_mean, contour_modes, contour_variances = _fit_contour_modes(contours, maximum_contour_modes)
    return ShapePopulationComponent(
        tuple(path), pose, tuple(centre), tuple(tuple(row) for row in covariance),
        tuple(standard_error), tuple(tuple(row) for row in predictive),
        float(len(vectors)), int(source_count), tuple(contour_mean),
        tuple(tuple(row) for row in contour_modes), tuple(contour_variances),
        bool(physical_dimensions_available),
        float(len(vectors) if calibrated_seed_count is None else calibrated_seed_count),
        float(len(vectors) if dimensionless_seed_count is None else dimensionless_seed_count),
    )


def _fit_contour_modes(contours, maximum_modes):
    if contours.ndim != 2 or not len(contours):
        return (), (), ()
    contours = np.asarray([_regularized_contour(value) for value in contours])
    contours = np.asarray(
        [
            min(
                (value, np.roll(value, len(value) // 2)),
                key=lambda row: tuple(np.round(row, 8)),
            )
            for value in contours
        ]
    )
    mean = np.mean(contours, axis=0)
    if len(contours) < 2:
        return tuple(mean), (), ()
    _u, singular, vt = np.linalg.svd(contours - mean, full_matrices=False)
    variances = np.square(singular) / max(len(contours) - 1, 1)
    keep = min(len(variances), _held_out_contour_mode_count(contours, maximum_modes))
    return tuple(mean), tuple(vt[:keep]), tuple(variances[:keep])


def _regularized_contour(signature):
    values = np.asarray(signature, np.float64)
    if not len(values):
        return values
    spectrum = np.fft.rfft(values)
    maximum_harmonic = min(12, max(3, len(values) // 16))
    spectrum[maximum_harmonic + 1 :] = 0
    return np.fft.irfft(spectrum, n=len(values))


def _held_out_contour_mode_count(contours, maximum_modes):
    maximum = min(int(maximum_modes), max(0, len(contours) - 1))
    if maximum <= 0 or len(contours) < 3:
        return maximum
    errors = np.zeros(maximum + 1, np.float64)
    for held_index in range(len(contours)):
        training = np.delete(contours, held_index, axis=0)
        centre = np.mean(training, axis=0)
        _u, _singular, modes = np.linalg.svd(
            training - centre, full_matrices=False
        )
        delta = contours[held_index] - centre
        for count in range(maximum + 1):
            reconstruction = centre
            if count:
                selected = modes[:count]
                reconstruction = centre + (delta @ selected.T) @ selected
            errors[count] += np.mean(
                np.square(contours[held_index] - reconstruction)
            )
    errors /= len(contours)
    best = 0
    for count in range(1, maximum + 1):
        if errors[count] <= errors[best] * 0.97:
            best = count
    return best


def _local_component(
    measurements,
    pose,
    *,
    minimum_component_seeds=1,
    maximum_contour_modes=0,
):
    if len(measurements) < minimum_component_seeds:
        return None
    array = np.asarray([item.vector() for item in measurements], np.float64)
    contours = np.asarray(
        [item.contour_signature for item in measurements], np.float64
    )
    uncertainties = np.asarray(
        [item.measurement_uncertainty for item in measurements], np.float64
    )
    return _fit_component(
        (), pose, array, contours, uncertainties, 1, parent=None,
        shrinkage_seed_count=0.0,
        maximum_contour_modes=maximum_contour_modes,
    )


def _update_component(
    prior,
    measurements,
    pose,
    *,
    shrinkage_seed_count,
    maximum_contour_modes,
):
    local = _local_component(
        measurements,
        pose,
        maximum_contour_modes=maximum_contour_modes,
    )
    if prior is None:
        return local
    assert local is not None
    prior_precision = np.linalg.pinv(np.asarray(prior.covariance))
    local_covariance = np.asarray(local.covariance)
    effective_local_count = max(len(measurements), 1) / (
        1.0 + float(shrinkage_seed_count)
    )
    local_precision = np.linalg.pinv(local_covariance) * effective_local_count
    covariance = np.linalg.pinv(prior_precision + local_precision)
    mean = covariance @ (
        prior_precision @ np.asarray(prior.mean)
        + local_precision @ np.asarray(local.mean)
    )
    standard_error = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    predictive_covariance = covariance + np.asarray(prior.covariance)
    predictive = np.column_stack((mean - 1.96 * np.sqrt(np.maximum(np.diag(predictive_covariance), 0.0)), mean + 1.96 * np.sqrt(np.maximum(np.diag(predictive_covariance), 0.0))))
    return ShapePopulationComponent(
        prior.hierarchy_path, pose, tuple(mean), tuple(tuple(row) for row in covariance),
        tuple(standard_error), tuple(tuple(row) for row in predictive),
        prior.effective_physical_seed_count + effective_local_count, prior.source_count,
        prior.contour_mean, prior.contour_modes, prior.contour_variances,
        prior.physical_dimensions_available,
        prior.calibrated_physical_seed_count + effective_local_count,
        prior.dimensionless_physical_seed_count + effective_local_count,
    )


def _out_of_family_distance(dimension):
    dimension = max(1, int(dimension))
    return float(dimension + 3.0 * np.sqrt(2.0 * dimension) + 2.0)
