"""Reviewed-seed measurements and hierarchical 2-D dimensions/shape models."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, replace
from typing import Iterable, Mapping

import numpy as np

from seedvision.measurement.geometry import (
    SeedShapeMeasurement, measure_shape_mask, _validated_pixels_per_mm,
    _scale_shape_measurement,
)
from seedvision.persistence.reference_regions import SeedInstanceAnnotation
from seedvision.measurement.lineage import (
    ReviewedMeasurementContext, ReviewedMeasurementLineage, prepare_reviewed_inputs,
)
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
    pixel_measurement: SeedShapeMeasurement | None = None
    calibrated_measurement: SeedShapeMeasurement | None = None
    calibration_state: str = "unavailable"
    calibration_reason: str = ""


@dataclass(frozen=True, slots=True)
class SeedMeasurementSummary:
    observations: tuple[ReviewedSeedShape, ...]
    eligible_count: int
    calibrated_count: int
    length_unit: str
    measurement_names: tuple[str, ...] = SHAPE_MEASUREMENT_NAMES
    boundary_turn_degrees: tuple[float, ...] = ()
    boundary_curvature_times_diameter: tuple[float, ...] = ()
    calibration_state: str = "unavailable"
    calibration_reason: str = ""
    lineage: ReviewedMeasurementLineage | None = None

    @property
    def size_observations(self):
        return tuple(item for item in self.observations
                     if (item.length_eligible or item.eligible) and item.measurement is not None)

    @property
    def size_statistics(self) -> tuple[float, float, float] | None:
        """Mean, between-seed SD and uncertainty of mean (shared scale retained)."""
        return self._size_statistics_result()[0]

    @property
    def size_statistics_unavailable_reason(self) -> str:
        return self._size_statistics_result()[1]

    def _size_statistics_result(self):
        items = self.size_observations
        if not items:
            return None, "No eligible size observations."
        if any(item.measurement.length_unit != self.length_unit for item in items):
            return None, "Size observations have inconsistent units."
        widths = np.asarray([item.measurement.maximum_span for item in items])
        errors = [item.measurement.measurement_uncertainty[0] for item in items]
        # Do not divide shared calibration or annotation systematic uncertainty
        # by sqrt(n). Sampling variability alone decreases with sample count.
        try:
            with np.errstate(over='ignore', invalid='ignore', under='ignore'):
                sd = float(np.std(widths, ddof=1)) if len(items) > 1 else 0.0
                measurement_error = float(np.mean([value ** 2 for value in errors]))
                statistics = (float(widths.mean()), sd,
                    float(np.sqrt(sd ** 2 / len(items) + measurement_error)))
        except OverflowError:
            return None, "Size statistics exceed the finite numeric range."
        if (not np.isfinite(statistics).all()
                or (measurement_error == 0 and any(value > 0 for value in errors))
                or (sd == 0 and np.any(widths != widths[0]))):
            return None, "Size statistics are not representable in the current unit."
        return statistics, ""

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
    update_state: str | None = None
    update_unavailability_reason: str | None = None
    pooling_method: str | None = None
    conditioning_scales: tuple[float, ...] = ()


@dataclass(frozen=True, slots=True)
class ShapeComponentUpdate:
    component: ShapePopulationComponent | None
    state: str
    source: str
    unavailability_reason: str | None = None
    pooling_method: str | None = None
    conditioning_scales: tuple[float, ...] = ()

    def __post_init__(self):
        if self.state not in {'not_attempted', 'local_only', 'applied', 'unavailable'}:
            raise ValueError('Shape component update state is unsupported.')
        if self.source not in {'local', 'library', 'library+local'}:
            raise ValueError('Shape component update source is unsupported.')
        if self.state == 'unavailable':
            if self.component is None or self.unavailability_reason not in {
                    'invalid_covariance', 'singular_precision', 'numeric_range'}:
                raise ValueError('Unavailable updates require a retained component and explicit reason.')
            if self.source == 'library+local':
                raise ValueError('Unavailable updates cannot claim pooled evidence.')
        elif self.unavailability_reason is not None:
            raise ValueError('Available updates cannot carry an unavailability reason.')
        if self.state in {'applied', 'local_only'}:
            expected_source = 'library+local' if self.state == 'applied' else 'local'
            if self.component is None or self.source != expected_source:
                raise ValueError('Applied and local-only updates require their actual component source.')
        elif self.state == 'not_attempted' and self.source == 'library+local':
            raise ValueError('Unattempted updates cannot claim pooled evidence.')
        if self.pooling_method is not None and (self.state != 'applied' or
                self.pooling_method not in {'legacy_precision', 'mixed_shape_precision',
                    'coordinate_conditioned_precision'}):
            raise ValueError('A pooling method describes a successfully applied update.')
        scales = tuple(self.conditioning_scales)
        if self.pooling_method == 'coordinate_conditioned_precision':
            if (len(scales) != 19 or any(isinstance(v, (bool, np.bool_)) for v in scales)
                    or not np.isfinite(scales).all() or any(v <= 0 for v in scales)):
                raise ValueError('Conditioned pooling requires positive finite coordinate scales.')
        elif scales:
            raise ValueError('Only conditioned pooling carries coordinate scales.')
        object.__setattr__(self, 'conditioning_scales', scales)


@dataclass(frozen=True, slots=True)
class ShapeCompatibilityAssessment:
    pose: str
    compatibility: float | None
    standardized_distance: float | None
    out_of_family: bool | None
    scoring_unavailability_reason: str | None = None

    def __post_init__(self):
        if self.scoring_unavailability_reason is None:
            if (self.compatibility is None or self.standardized_distance is None
                    or not np.isfinite((self.compatibility, self.standardized_distance)).all()
                    or not 0 <= self.compatibility <= 1 or self.standardized_distance < 0
                    or not isinstance(self.out_of_family, bool)):
                raise ValueError("Available shape scores require finite values and a verdict.")
        elif (self.scoring_unavailability_reason not in {
                'invalid_measurement', 'invalid_covariance', 'numeric_range'}
                or any(value is not None for value in (
                    self.compatibility, self.standardized_distance, self.out_of_family))):
            raise ValueError("Unavailable shape scores require a reason and no score or verdict.")

    @property
    def scoring_available(self) -> bool:
        return self.scoring_unavailability_reason is None


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
        values = tuple(measurement)
        return tuple(
            assess_candidate_shape(values, family.component)
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
    input_context: ReviewedMeasurementContext | None = None,
) -> SeedMeasurementSummary:
    prepared = prepare_reviewed_inputs(labels, annotations, context=input_context)
    return _measure_prepared_seed_instances(prepared, pixels_per_mm=pixels_per_mm,
        calibration_relative_uncertainty=calibration_relative_uncertainty,
        contour_samples=contour_samples,
        boundary_perturbation_radius_px=boundary_perturbation_radius_px)


def _measure_prepared_seed_instances(
    prepared, *, pixels_per_mm=None, calibration_relative_uncertainty=0.0,
    contour_samples=128, boundary_perturbation_radius_px=2,
) -> SeedMeasurementSummary:
    """Full outlines inform shape; explicitly visible full spans also inform size.

    Partial masks never enter ovality, contour or concavity population fitting.
    """

    labels, annotations = prepared.labels, prepared.annotations
    try:
        scale = _validated_pixels_per_mm(pixels_per_mm)
        calibration_reason = "No pixels_per_mm scale was supplied." if scale is None else ""
    except ValueError as error:
        scale, calibration_reason = None, str(error)
    if labels is None:
        return SeedMeasurementSummary((), 0, 0, "px", calibration_reason=(
            calibration_reason or "No seed measurement bundles are available."))
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
        calibrated = None
        instance_calibration_reason = calibration_reason
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
                        pixels_per_mm=None,
                        calibration_relative_uncertainty=(
                            calibration_relative_uncertainty
                        ),
                        hilum_point=annotation.hilum_point,
                        contour_samples=contour_samples,
                        perturbation_radii=tuple(
                            range(1, boundary_perturbation_radius_px + 1)
                        ),
                    )
                    if measurement.provenance is not None:
                        measurement = replace(measurement, provenance=replace(
                            measurement.provenance,
                            visibility=annotation.outline_visibility,
                            source_image_frame=('corrected_image' if prepared.lineage
                                and prepared.lineage.source_image_frame == 'full_corrected'
                                else 'unknown')))
                except ValueError as error:
                    reason = str(error)
                    length_eligible = False
                if measurement is not None and scale is not None:
                    try:
                        calibrated = _scale_shape_measurement(measurement, scale)
                    except ValueError as error:
                        instance_calibration_reason = str(error)
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
                pixel_measurement=measurement,
                calibrated_measurement=calibrated,
                calibration_state="available" if calibrated is not None else "unavailable",
                calibration_reason=(instance_calibration_reason if measurement is not None
                    else "No eligible measurement bundle is available."),
            )
        )
    eligible = sum(item.eligible for item in results)
    measured = [item for item in results if item.measurement is not None]
    available = sum(item.calibrated_measurement is not None for item in measured)
    all_calibrated = bool(measured) and available == len(measured)
    if all_calibrated:
        results = [replace(item, measurement=item.calibrated_measurement)
                   if item.calibrated_measurement is not None else item for item in results]
    state = "available" if all_calibrated else "partial" if available else "unavailable"
    if all_calibrated:
        summary_reason = ""
    elif measured:
        summary_reason = calibration_reason or (
            "One or more seed bundles cannot be calibrated; summary measurements remain in pixels.")
    else:
        summary_reason = calibration_reason or "No seed measurement bundles are available."
    return SeedMeasurementSummary(
        tuple(results),
        eligible,
        sum(item.eligible and item.calibrated_measurement is not None for item in results),
        "mm" if all_calibrated else "px",
        boundary_turn_degrees=tuple(boundary_turns),
        boundary_curvature_times_diameter=tuple(boundary_curvatures),
        calibration_state=state,
        calibration_reason=summary_reason,
        lineage=prepared.lineage,
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
        measurement = item.calibrated_measurement or item.measurement
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
            all_shapes = _physical_seed_groups(records, normalize_per_view=True)
            calibrated_seed_count = len({
                _seed_group_key(item) for item in records if item.calibrated
            })
            if len(all_shapes[0]) < minimum_component_seeds:
                continue
            parent = components.get((path[:-1], pose)) if len(path) > 1 else None
            unavailability_reason = None
            component = None
            if calibrated_seed_count >= minimum_component_seeds:
                try:
                    calibrated = _physical_seed_groups(records, calibrated_only=True)
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
                        calibrated_seed_count=calibrated_seed_count,
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
                except _PhysicalFitNumericUnavailable:
                    component = None
                    unavailability_reason = "numeric_range"
            if component is None:
                component = _fit_component(
                    path,
                    pose,
                    all_shapes[0],
                    all_shapes[1],
                    all_shapes[2],
                    all_shapes[3],
                    parent=(parent if parent is not None and not parent.physical_dimensions_available else None),
                    shrinkage_seed_count=shrinkage_seed_count,
                    maximum_contour_modes=maximum_contour_modes,
                    physical_dimensions_available=False,
                    calibrated_seed_count=calibrated_seed_count,
                    dimensionless_seed_count=len(all_shapes[0]),
                )
                component = replace(component, physical_fit_unavailability_reason=unavailability_reason)
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
        update_state = 'not_attempted'
        update_reason = None
        pooling_method = None
        conditioning_scales = ()
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
            update = _update_component_result(
                component,
                local_pose,
                pose,
                shrinkage_seed_count=shrinkage_seed_count,
                maximum_contour_modes=maximum_contour_modes,
            )
            component = update.component
            update_state, update_reason = update.state, update.unavailability_reason
            pooling_method, conditioning_scales = update.pooling_method, update.conditioning_scales
            in_sample = in_sample or update.state in {'applied', 'local_only'}
            component_source = update.source
            if update_reason is not None:
                warnings.append(f'{pose}: library/current shape pooling unavailable '
                    f'({update_reason}); retained {component_source} component.')
        if component is not None:
            families.append(
                SeedPoseShapeFamily(
                    pose,
                    component,
                    len(local_pose),
                    component_source,
                    update_state=update_state,
                    update_unavailability_reason=update_reason,
                    pooling_method=pooling_method,
                    conditioning_scales=conditioning_scales,
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
            item.update_state,
            item.update_unavailability_reason,
            item.pooling_method,
            item.conditioning_scales,
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
) -> float | None:
    """Joint compatibility, or None when the comparison is unavailable."""
    return assess_candidate_shape(measurement, component).compatibility


def _candidate_shape_distance(measurement, component):
    values = np.asarray(tuple(measurement), np.float64)
    mean = np.asarray(component.mean, np.float64)
    if mean.shape != (len(SHAPE_MEASUREMENT_NAMES),):
        raise ValueError("Shape model must use the current measurement schema.")
    if values.shape != mean.shape:
        raise ValueError("Candidate measurement does not match the shape model.")
    indices = (np.arange(len(mean)) if component.physical_dimensions_available
               else _DIMENSIONLESS_MEASUREMENT_INDICES)
    dimension = len(indices)
    if (not np.isfinite(values).all() or np.any(values[[0, 1, 2, 4]] < 0)
            or values[0] <= 0):
        return None, dimension, 'invalid_measurement'
    if not component.physical_dimensions_available:
        try:
            values, _uncertainty = _dimensionless_vectors(
                values[None], np.zeros_like(values)[None])
            values = values[0]
        except _ShapeNormalizationNumericUnavailable:
            return None, dimension, 'numeric_range'
    covariance = np.asarray(component.covariance, np.float64)
    selected_covariance = covariance[np.ix_(indices, indices)]
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise', under='ignore'):
            # Validate in a bounded scale so the guard itself cannot overflow.
            # These are the same absolute/relative tolerances as shape pooling.
            magnitude = float(np.max(np.abs(selected_covariance)))
            scale = max(1., magnitude)
            scaled = selected_covariance / scale
            if (not np.allclose(scaled, scaled.T, rtol=1e-12, atol=1e-12 / scale)
                    or np.linalg.eigvalsh(scaled).min()
                    < -max(1e-12 / scale, (magnitude / scale) * 1e-12)):
                return None, dimension, 'invalid_covariance'
            regularized = selected_covariance + np.eye(dimension) * max(
                1e-8, np.trace(selected_covariance) * 1e-6)
            delta = values[indices] - mean[indices]
            distance = float(delta @ np.linalg.pinv(regularized) @ delta)
            if (not np.isfinite(distance) or distance < 0
                    or (distance == 0 and np.any(delta != 0))):
                return None, dimension, 'numeric_range'
    except (FloatingPointError, np.linalg.LinAlgError):
        return None, dimension, 'numeric_range'
    return distance, dimension, None


def assess_candidate_shape(
    measurement: Iterable[float], component: ShapePopulationComponent
) -> ShapeCompatibilityAssessment:
    """Keep a computational abstention separate from an out-of-family verdict."""
    distance, dimension, reason = _candidate_shape_distance(measurement, component)
    if reason is not None:
        return ShapeCompatibilityAssessment(component.pose, None, None, None, reason)
    threshold = _out_of_family_distance(dimension)
    return ShapeCompatibilityAssessment(
        component.pose,
        float(np.exp(-0.5 * min(distance, 80.0) / max(dimension, 1))),
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


def _seed_group_key(item):
    return item.physical_seed_id or f"source:{item.source_index}:seed:{item.seed_id}"


def _physical_seed_groups(records, *, calibrated_only=False, normalize_per_view=False):
    """Aggregate biological seeds, normalizing each view before mixing units."""
    grouped: dict[str, list[ShapeObservation]] = {}
    selected_sources: set[int] = set()
    for item in records:
        if calibrated_only and not item.calibrated:
            continue
        selected_sources.add(item.source_index)
        key = _seed_group_key(item)
        grouped.setdefault(key, []).append(item)
    vectors = []
    contours = []
    uncertainties = []
    for values in grouped.values():
        view_vectors = np.asarray([item.measurement for item in values], np.float64)
        view_errors = np.asarray([item.measurement_uncertainty for item in values], np.float64)
        if normalize_per_view:
            view_vectors, view_errors = _dimensionless_vectors(view_vectors, view_errors)
        with np.errstate(over='ignore', under='ignore', invalid='ignore'):
            mean = np.mean(view_vectors, axis=0)
            error = np.sqrt(np.mean(np.square(view_errors), axis=0))
        # Preserve ordinary arithmetic unless an intermediate loses the
        # finite mean/RMS. This is still the existing shared-error model,
        # not propagation of uncertainty in the normalizing span.
        unsafe_mean = ~np.isfinite(mean)
        if np.any(unsafe_mean):
            scale = np.max(np.abs(view_vectors[:, unsafe_mean]), axis=0)
            mean[unsafe_mean] = np.mean(view_vectors[:, unsafe_mean] / scale, axis=0) * scale
        unsafe_error = ~np.isfinite(error) | ((error == 0) & np.any(view_errors > 0, axis=0))
        if np.any(unsafe_error):
            scale = np.max(view_errors[:, unsafe_error], axis=0)
            error[unsafe_error] = np.sqrt(np.mean(np.square(
                view_errors[:, unsafe_error] / scale), axis=0)) * scale
        if (not np.isfinite(mean).all() or not np.isfinite(error).all()
                or np.any((error == 0) & np.any(view_errors > 0, axis=0))):
            if not normalize_per_view:
                raise _PhysicalFitNumericUnavailable("Physical seed groups are not representable.")
            raise ValueError("Normalized seed groups are not representable.")
        vectors.append(mean)
        uncertainties.append(error)
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


class _ShapeNormalizationNumericUnavailable(ValueError):
    """A finite input cannot be represented in the dimensionless fit schema."""


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
    if (not np.isfinite(values).all() or not np.isfinite(errors).all()
            or np.any(errors < 0) or np.any(values[:, (0, 1, 2, 4)] < 0)
            or np.any(values[:, 0] <= 0)):
        raise ValueError("Dimensionless shape normalization requires finite values and positive spans.")
    span = values[:, 0].copy()
    # A floor in a physical unit changes shape when the same view is rescaled.
    # Preserve ordinary area arithmetic, with sequential division only when
    # squaring the span cannot be represented.
    with np.errstate(over='ignore', under='ignore', divide='ignore', invalid='ignore'):
        squared_span = np.square(span)
        safe_square = np.isfinite(squared_span) & (squared_span > 0)
        for array in (values, errors):
            original = array[:, (1, 2, 4)].copy()
            array[:, 1] /= span
            array[:, 2] /= span
            array[safe_square, 4] /= squared_span[safe_square]
            array[~safe_square, 4] = (array[~safe_square, 4] / span[~safe_square]) / span[~safe_square]
            normalized = array[:, (1, 2, 4)]
            if (not np.isfinite(normalized).all()
                    or np.any((original > 0) & (normalized == 0))):
                raise _ShapeNormalizationNumericUnavailable("Dimensionless shape values or uncertainties are not representable.")
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
    """Use already per-view-normalized seed groups in a calibrated fit."""

    normalized, normalized_uncertainty = vectors, uncertainties
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
    with _physical_fit_numeric_domain():
        # Retain calibrated cross-covariances, then project the assembled symmetric
        # matrix back to the positive-semidefinite cone for stable Mahalanobis use.
        covariance = 0.5 * (covariance + covariance.T)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        covariance = (eigenvectors * np.maximum(eigenvalues, 1e-10)) @ eigenvectors.T
        _require_physical_fit_numbers(mean, uncertainty, predictive, covariance)
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


class _PhysicalFitNumericUnavailable(ArithmeticError):
    """A finite calibrated bundle cannot support this floating-point fit."""


@contextmanager
def _physical_fit_numeric_domain():
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise', under='ignore'):
            yield
    except FloatingPointError as error:
        raise _PhysicalFitNumericUnavailable("Physical shape fit exceeds numeric range.") from error


def _require_physical_fit_numbers(*arrays):
    if any(not np.isfinite(array).all() for array in arrays):
        raise _PhysicalFitNumericUnavailable("Physical shape fit exceeds numeric range.")


def _check_physical_square_range(values):
    dimensions = np.asarray(values)[:, _PHYSICAL_MEASUREMENT_INDICES]
    squared = np.square(dimensions)
    if np.any((dimensions != 0) & (squared == 0)):
        raise _PhysicalFitNumericUnavailable("Physical shape fit variance is not representable.")


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
    if parent is not None and parent.physical_dimensions_available != physical_dimensions_available:
        parent = None
    with (_physical_fit_numeric_domain() if physical_dimensions_available else nullcontext()):
        centre = np.median(vectors, axis=0)
        deviations = np.abs(vectors - centre)
        scale = np.maximum(1e-8, 1.4826 * np.median(deviations, axis=0))
        standardized = np.clip((vectors - centre) / scale, -4.0, 4.0)
        robust_vectors = centre + standardized * scale
        if physical_dimensions_available:
            _check_physical_square_range(robust_vectors - np.mean(robust_vectors, axis=0))
            if np.asarray(uncertainties).shape == vectors.shape:
                _check_physical_square_range(uncertainties)
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
        if physical_dimensions_available:
            _require_physical_fit_numbers(centre, covariance, standard_error, predictive)
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
    units = {(item.length_unit, item.area_unit) for item in measurements}
    if len(units) != 1 or not units <= {('px', 'px²'), ('mm', 'mm²')}:
        raise ValueError('Local shape measurements must use one supported dimensional unit.')
    physical = units == {('mm', 'mm²')}
    array = np.asarray([item.vector() for item in measurements], np.float64)
    contours = np.asarray(
        [item.contour_signature for item in measurements], np.float64
    )
    uncertainties = np.asarray(
        [item.measurement_uncertainty for item in measurements], np.float64
    )
    if physical:
        try:
            return _fit_component(
                (), pose, array, contours, uncertainties, 1, parent=None,
                shrinkage_seed_count=0.0, maximum_contour_modes=maximum_contour_modes,
                physical_dimensions_available=True, calibrated_seed_count=len(measurements),
            )
        except _PhysicalFitNumericUnavailable:
            reason = "numeric_range"
    else:
        reason = None
    array, uncertainties = _dimensionless_vectors(array, uncertainties)
    component = _fit_component(
        (), pose, array, contours, uncertainties, 1, parent=None,
        shrinkage_seed_count=0.0, maximum_contour_modes=maximum_contour_modes,
        physical_dimensions_available=False,
        calibrated_seed_count=len(measurements) if physical else 0,
    )
    return replace(component, physical_fit_unavailability_reason=reason)



class ShapeUpdateUnavailable(ValueError):
    """A component is valid evidence but the requested pooling cannot be applied."""

    def __init__(self, reason):
        self.reason = reason
        message = ('Shape covariance must be symmetric and positive semidefinite.'
            if reason == 'invalid_covariance' else f'Shape pooling unavailable: {reason}.')
        super().__init__(message)


def _update_covariance_valid(covariance):
    magnitude = float(np.max(np.abs(covariance)))
    scale = max(1., magnitude)
    scaled = covariance / scale
    if (not np.allclose(scaled, scaled.T, rtol=1e-12, atol=1e-12 / scale)
            or np.linalg.eigvalsh(scaled).min()
            < -max(1e-12 / scale, (magnitude / scale) * 1e-12)):
        raise ShapeUpdateUnavailable('invalid_covariance')


def _update_pinv(matrix, *, hermitian=False):
    # Match NumPy's existing default pinv cutoff. Discarded directions cannot
    # be interpreted as a measured zero mean or zero uncertainty.
    _update_required_directions(matrix)
    result = np.linalg.pinv(matrix, hermitian=hermitian)
    _require_update_numbers(result)
    return result


def _update_required_directions(matrix):
    singular = np.linalg.svd(matrix, compute_uv=False)
    _require_update_numbers(singular)
    if singular[0] == 0 or singular[-1] / singular[0] <= 1e-15:
        raise ShapeUpdateUnavailable('singular_precision')


def _require_update_numbers(*arrays):
    if any(not np.isfinite(array).all() for array in arrays):
        raise ShapeUpdateUnavailable('numeric_range')


def _update_matvec(matrix, vector):
    products = matrix * vector[np.newaxis, :]
    if np.any((matrix != 0) & (vector[np.newaxis, :] != 0) & (products == 0)):
        raise ShapeUpdateUnavailable('numeric_range')
    result = matrix @ vector
    _require_update_numbers(result)
    return result


def _weighted_update_precision(precision, count):
    weighted = precision * count
    if np.any((precision != 0) & (weighted == 0)):
        raise ShapeUpdateUnavailable('numeric_range')
    _require_update_numbers(weighted)
    return weighted


def _guarded_shape_pool(operation, *args):
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise', under='ignore'):
            return operation(*args)
    except (FloatingPointError, np.linalg.LinAlgError):
        raise ShapeUpdateUnavailable('numeric_range') from None


def _update_component_result(
    prior,
    measurements,
    pose,
    *,
    shrinkage_seed_count,
    maximum_contour_modes,
):
    if (isinstance(shrinkage_seed_count, (bool, np.bool_))
            or not np.isfinite(shrinkage_seed_count) or shrinkage_seed_count < 0):
        raise ValueError('Shape shrinkage must be finite and nonnegative.')
    local = _local_component(
        measurements,
        pose,
        maximum_contour_modes=maximum_contour_modes,
    )
    if prior is None:
        return ShapeComponentUpdate(local, 'local_only' if local is not None else 'not_attempted', 'local')
    assert local is not None
    effective_local_count = max(len(measurements), 1) / (
        1.0 + float(shrinkage_seed_count)
    )
    mixed = prior.physical_dimensions_available != local.physical_dimensions_available
    baseline = local if mixed and local.physical_dimensions_available else prior
    try:
        if effective_local_count <= 0 or not np.isfinite(effective_local_count):
            raise ShapeUpdateUnavailable('numeric_range')
        conditioning_scales = ()
        if mixed:
            component = _guarded_shape_pool(_pool_mixed_dimension_components,
                prior, local, pose, effective_local_count)
            pooling_method = 'mixed_shape_precision'
        else:
            try:
                component = _guarded_shape_pool(_pool_same_dimension_components,
                    prior, local, pose, effective_local_count)
                pooling_method = 'legacy_precision'
            except ShapeUpdateUnavailable as error:
                if error.reason not in {'singular_precision', 'numeric_range'}:
                    raise
                component, conditioning_scales = _guarded_shape_pool(
                    _pool_conditioned_same_dimension_components,
                    prior, local, pose, effective_local_count)
                pooling_method = 'coordinate_conditioned_precision'
    except ShapeUpdateUnavailable as error:
        return ShapeComponentUpdate(baseline, 'unavailable',
            'local' if baseline is local else 'library', error.reason)
    return ShapeComponentUpdate(component, 'applied', 'library+local',
        pooling_method=pooling_method, conditioning_scales=conditioning_scales)


def _update_component(prior, measurements, pose, *, shrinkage_seed_count, maximum_contour_modes):
    """Legacy component-only helper; failed updates raise instead of losing diagnostics."""
    update = _update_component_result(prior, measurements, pose,
        shrinkage_seed_count=shrinkage_seed_count, maximum_contour_modes=maximum_contour_modes)
    if update.unavailability_reason is not None:
        raise ShapeUpdateUnavailable(update.unavailability_reason)
    return update.component


def _pool_same_dimension_components(prior, local, pose, effective_local_count):
    prior_covariance = np.asarray(prior.covariance)
    local_covariance = np.asarray(local.covariance)
    _update_covariance_valid(prior_covariance)
    _update_covariance_valid(local_covariance)
    prior_precision = _update_pinv(prior_covariance)
    local_precision = _weighted_update_precision(_update_pinv(local_covariance), effective_local_count)
    covariance = _update_pinv(prior_precision + local_precision)
    mean = _update_matvec(covariance, (
        _update_matvec(prior_precision, np.asarray(prior.mean))
        + _update_matvec(local_precision, np.asarray(local.mean)))
    )
    return _same_dimension_pooled_component(prior, local, pose,
        effective_local_count, mean, covariance)


def _conditioned_update_inverse(matrix):
    if np.any(np.diag(matrix) <= 0):
        raise ShapeUpdateUnavailable('singular_precision')
    # Equilibrate each covariance/precision solve as well. A covariance
    # contrast between cohorts must not reintroduce the raw unit cutoff.
    scales = np.sqrt(np.diag(matrix))
    smaller = np.minimum(scales[:, None], scales[None, :])
    larger = np.maximum(scales[:, None], scales[None, :])
    normalized = _conditioned_covariance(matrix, smaller, larger)
    _update_covariance_valid(normalized)
    _update_required_directions(normalized)
    factor = np.linalg.cholesky(normalized)
    inverse_normalized = np.linalg.solve(factor.T,
        np.linalg.solve(factor, np.eye(len(matrix))))
    inverse_normalized = (inverse_normalized + inverse_normalized.T) * .5
    inverse = (inverse_normalized / smaller) / larger
    _require_update_numbers(inverse)
    if np.any((inverse_normalized != 0) & (inverse == 0)):
        raise ShapeUpdateUnavailable('numeric_range')
    return inverse


def _conditioned_covariance(covariance, smaller_scale, larger_scale):
    # Dividing by the smaller factor first avoids a preventable intermediate
    # underflow when covariance coordinates have very different units.
    result = (covariance / smaller_scale) / larger_scale
    _require_update_numbers(result)
    if np.any((covariance != 0) & (result == 0)):
        raise ShapeUpdateUnavailable('numeric_range')
    return (result + result.T) * .5


def _pool_conditioned_same_dimension_components(prior, local, pose, effective_local_count):
    """Solve the same Gaussian pooling formula in dimensionless coordinates.

    No ridge/jitter is added. Each scale carries its coordinate's unit; neither
    this numerical conditioning nor its successful result validates physical
    calibration or the underlying statistical assumptions.
    """
    prior_covariance = np.asarray(prior.covariance)
    local_covariance = np.asarray(local.covariance)
    if np.any(np.diag(prior_covariance) <= 0) or np.any(np.diag(local_covariance) <= 0):
        raise ShapeUpdateUnavailable('singular_precision')
    scales = np.sqrt(np.maximum(np.diag(prior_covariance), np.diag(local_covariance)))
    smaller = np.minimum(scales[:, None], scales[None, :])
    larger = np.maximum(scales[:, None], scales[None, :])
    prior_normalized = _conditioned_covariance(prior_covariance, smaller, larger)
    local_normalized = _conditioned_covariance(local_covariance, smaller, larger)
    prior_precision = _conditioned_update_inverse(prior_normalized)
    local_precision = _weighted_update_precision(
        _conditioned_update_inverse(local_normalized), effective_local_count)
    covariance_normalized = _conditioned_update_inverse(prior_precision + local_precision)
    # Centering at the prior keeps an identical large common offset exact.
    # Opposite large offsets may overflow subtraction even when their
    # dimensionless separation and final posterior are representable.
    with np.errstate(over='ignore'):
        delta = np.asarray(local.mean) - np.asarray(prior.mean)
        normalized_delta = delta / scales
    overflowed = ~np.isfinite(delta)
    if np.any(overflowed):
        normalized_delta[overflowed] = (
            np.asarray(local.mean)[overflowed] / scales[overflowed]
            - np.asarray(prior.mean)[overflowed] / scales[overflowed])
    _require_update_numbers(normalized_delta)
    if np.any((delta != 0) & (normalized_delta == 0)):
        raise ShapeUpdateUnavailable('numeric_range')
    correction_normalized = _update_matvec(covariance_normalized,
        _update_matvec(local_precision, normalized_delta))
    correction = correction_normalized * scales
    covariance = (covariance_normalized * larger) * smaller
    _require_update_numbers(correction, covariance)
    if (np.any((correction_normalized != 0) & (correction == 0))
            or np.any((covariance_normalized != 0) & (covariance == 0))):
        raise ShapeUpdateUnavailable('numeric_range')
    _update_covariance_valid(covariance)
    mean = np.asarray(prior.mean) + correction
    component = _same_dimension_pooled_component(prior, local, pose,
        effective_local_count, mean, covariance)
    return component, tuple(float(value) for value in scales)


def _same_dimension_pooled_component(prior, local, pose, effective_local_count, mean, covariance):
    if np.any(np.diag(covariance) <= 0):
        raise ShapeUpdateUnavailable('numeric_range')
    standard_error = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    # Preserve successful legacy arithmetic. Only an overflowing variance
    # sum needs the equivalent scaled Euclidean norm of standard deviations.
    with np.errstate(over='ignore'):
        predictive_covariance = covariance + np.asarray(prior.covariance)
    predictive_scale = np.sqrt(np.maximum(np.diag(predictive_covariance), 0.0))
    overflowed = ~np.isfinite(predictive_scale)
    if np.any(overflowed):
        predictive_scale[overflowed] = np.hypot(
            np.sqrt(np.diag(covariance)[overflowed]),
            np.sqrt(np.diag(np.asarray(prior.covariance))[overflowed]))
    predictive = np.column_stack((mean - 1.96 * predictive_scale, mean + 1.96 * predictive_scale))
    effective_count = _updated_support_count(prior.effective_physical_seed_count, effective_local_count)
    calibrated_count = _updated_support_count(prior.calibrated_physical_seed_count,
        effective_local_count if local.calibrated_physical_seed_count > 0 else 0)
    dimensionless_count = _updated_support_count(prior.dimensionless_physical_seed_count, effective_local_count)
    _require_update_numbers(mean, covariance, standard_error, predictive,
        effective_count, calibrated_count, dimensionless_count)
    return ShapePopulationComponent(
        prior.hierarchy_path, pose, tuple(mean), tuple(tuple(row) for row in covariance),
        tuple(standard_error), tuple(tuple(row) for row in predictive),
        effective_count, prior.source_count,
        prior.contour_mean, prior.contour_modes, prior.contour_variances,
        prior.physical_dimensions_available,
        calibrated_count, dimensionless_count,
        prior.physical_fit_unavailability_reason or local.physical_fit_unavailability_reason,
    )


def _updated_support_count(existing, increment):
    result = existing + increment
    if not np.isfinite(result) or (increment > 0 and result <= existing):
        raise ShapeUpdateUnavailable('numeric_range')
    return result


def _update_mixed_dimension_components(prior, local, pose, effective_local_count):
    if not np.isfinite(effective_local_count) or effective_local_count <= 0:
        raise ValueError('Effective local update count must be finite and positive.')
    return _guarded_shape_pool(_pool_mixed_dimension_components, prior, local, pose, effective_local_count)


def _pool_mixed_dimension_components(prior, local, pose, effective_local_count):
    """Pool shape only; retain the physical component's dimensional block.

    Cross-block covariance is explicitly zero because a dimensionless prior
    cannot provide a calibrated relationship between shape and physical size.
    """
    physical = local if local.physical_dimensions_available else prior
    indices = _DIMENSIONLESS_MEASUREMENT_INDICES
    prior_covariance = np.asarray(prior.covariance)[np.ix_(indices, indices)]
    local_covariance = np.asarray(local.covariance)[np.ix_(indices, indices)]
    def shape_precision(covariance):
        # The same dimensionless ridge convention used by candidate scoring
        # gives finite nonzero precision to identical or singular cohorts.
        _update_covariance_valid(covariance)
        ridge = max(1e-8, np.trace(covariance) * 1e-6)
        _require_update_numbers(ridge)
        return _update_pinv(covariance + np.eye(len(indices)) * ridge, hermitian=True)
    prior_precision = shape_precision(prior_covariance)
    local_precision = _weighted_update_precision(shape_precision(local_covariance), effective_local_count)
    pooled_covariance = _update_pinv(prior_precision + local_precision, hermitian=True)
    pooled_mean = _update_matvec(pooled_covariance, (
        _update_matvec(prior_precision, np.asarray(prior.mean)[indices])
        + _update_matvec(local_precision, np.asarray(local.mean)[indices])))
    if np.any(np.diag(pooled_covariance) <= 0):
        raise ShapeUpdateUnavailable('numeric_range')
    mean = np.asarray(physical.mean).copy()
    mean[indices] = pooled_mean
    covariance = np.asarray(physical.covariance).copy()
    covariance[np.ix_(indices, indices)] = pooled_covariance
    dimensions = _PHYSICAL_MEASUREMENT_INDICES
    covariance[np.ix_(dimensions, indices)] = 0
    covariance[np.ix_(indices, dimensions)] = 0
    uncertainty = np.asarray(physical.mean_uncertainty).copy()
    uncertainty[indices] = np.sqrt(np.maximum(np.diag(pooled_covariance), 0))
    predictive = np.asarray(physical.predictive_interval_95).copy()
    predictive_scale = np.sqrt(np.maximum(np.diag(pooled_covariance + prior_covariance), 0))
    predictive[indices] = np.column_stack((pooled_mean - 1.96 * predictive_scale,
                                          pooled_mean + 1.96 * predictive_scale))
    effective_count = _updated_support_count(prior.effective_physical_seed_count, effective_local_count)
    calibrated_count = _updated_support_count(physical.calibrated_physical_seed_count,
            prior.calibrated_physical_seed_count if not prior.physical_dimensions_available
            else effective_local_count if local.calibrated_physical_seed_count > 0 else 0)
    dimensionless_count = _updated_support_count(prior.dimensionless_physical_seed_count, effective_local_count)
    _require_update_numbers(mean, covariance, uncertainty, predictive,
        effective_count, calibrated_count, dimensionless_count)
    return ShapePopulationComponent(
        prior.hierarchy_path, pose, tuple(mean), tuple(tuple(row) for row in covariance),
        tuple(uncertainty), tuple(tuple(row) for row in predictive),
        effective_count,
        max(prior.source_count, local.source_count), prior.contour_mean,
        prior.contour_modes, prior.contour_variances, True,
        calibrated_count, dimensionless_count,
    )


def _out_of_family_distance(dimension):
    dimension = max(1, int(dimension))
    return float(dimension + 3.0 * np.sqrt(2.0 * dimension) + 2.0)
