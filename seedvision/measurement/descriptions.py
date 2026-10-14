"""Compact value/definition records for observed-mask geometry.

These records describe calculations and sensitivity, not scientific validity.
Legacy bundles have no provenance and retain explicitly unknown definitions.
"""
from dataclasses import dataclass
import math
from numbers import Integral, Real


SHAPE_DEFINITION_REVISION = 'seedfiddle.observed-mask-geometry-v1'


@dataclass(frozen=True, slots=True)
class ShapeMeasurementProvenance:
    definition_revision: str
    perturbation_radii_px: tuple[int, ...]
    boundary_alternative_count: int
    calibration_relative_uncertainty: float
    pixels_per_mm: float | None = None
    source_image_frame: str = 'unknown'
    visibility: str = 'unknown'

    def __post_init__(self):
        self.validate()

    def validate(self):
        if not isinstance(self.definition_revision, str) or not self.definition_revision:
            raise ValueError('Measurement definition revision must be explicit.')
        if (not isinstance(self.perturbation_radii_px, tuple)
                or any(isinstance(r, bool) or not isinstance(r, Integral) or r <= 0
                       for r in self.perturbation_radii_px)):
            raise ValueError('Measurement perturbation radii must be a tuple of positive integers.')
        if (isinstance(self.boundary_alternative_count, bool)
                or not isinstance(self.boundary_alternative_count, Integral)
                or not 0 <= self.boundary_alternative_count <= 4 * len(self.perturbation_radii_px)):
            raise ValueError('Measurement boundary alternative count is inconsistent with its radii.')
        for value, name in ((self.calibration_relative_uncertainty, 'relative uncertainty'),
                            (self.pixels_per_mm, 'pixels_per_mm')):
            if name == 'pixels_per_mm' and value is None:
                continue
            try:
                valid = not isinstance(value, bool) and isinstance(value, Real) and math.isfinite(value)
            except OverflowError:
                valid = False
            if not valid or (name == 'relative uncertainty' and not 0 <= value <= .5) or (name == 'pixels_per_mm' and value <= 0):
                raise ValueError('Measurement provenance requires valid finite calibration scalars.')
        if self.source_image_frame not in {'unknown', 'raw_image', 'corrected_image'}:
            raise ValueError('Measurement source image frame is unsupported.')
        if self.visibility not in {'unknown', 'complete', 'partly_occluded', 'image_cutoff', 'uncertain'}:
            raise ValueError('Measurement visibility is unsupported.')


@dataclass(frozen=True, slots=True)
class DescribedUncertainty:
    value: float | None
    unit: str
    semantics: str


@dataclass(frozen=True, slots=True)
class DescribedShapeValue:
    key: str
    value: float | tuple
    definition_revision: str
    definition: str
    unit: str
    coordinate_frame: str
    source_image_frame: str
    visibility: str
    measurement_basis: str
    provenance: ShapeMeasurementProvenance | None
    uncertainty: DescribedUncertainty


# The order is the existing 19-component vector contract, including the local
# feature's doubled angle (not the whole ellipse's image orientation).
_VECTOR_DEFINITIONS = (
    ('maximum_span', 'Greatest pairwise distance between convex-hull contour points.'),
    ('body_length', 'Major diameter of the body ellipse after iterative radial-residual clipping.'),
    ('body_width', 'Minor diameter of the body ellipse after iterative radial-residual clipping.'),
    ('ovality', 'Pixel-derived body length divided by body width, with the pixel denominator floor of 1e-9.'),
    ('projected_area', 'Nonzero supplied-mask pixel count, optionally converted by squared pixels_per_mm.'),
    ('ellipse_relative_area', 'Mask pixel count divided by pi times fitted body diameters divided by four, with a 1e-9 denominator floor.'),
    ('non_ellipticity', 'RMS contour radial deviation relative to the expected body-ellipse radius.'),
    ('asymmetry', 'Mean absolute difference between opposite halves of the angular radial-residual signature.'),
    ('solidity', 'Mask pixel count divided by the rasterized convex-hull pixel count.'),
    ('concavity_fraction', 'Rasterized convex-hull pixels outside the mask divided by mask pixel count.'),
    ('protrusion_fraction', 'Mean positive angular radial-residual signature value.'),
    ('neck_width_fraction', 'Tenth percentile of twice extension distance-transform values divided by body width; one when no extension exists.'),
    ('thin_extension_fraction', 'Extension pixels with twice their distance-transform value below 0.18 times body width, divided by mask pixel count.'),
    ('local_feature_height', 'Selected angular feature signed radial-residual peak; zero when absent.'),
    ('local_feature_arc_fraction', 'Selected feature support fraction of angular bins; zero when absent.'),
    ('local_feature_axis_cosine', 'Cosine of twice the local-feature angle relative to the body major axis; zero when absent.'),
    ('local_feature_axis_sine', 'Sine of twice the local-feature angle relative to the body major axis; zero when absent.'),
    ('local_feature_support', 'Selected feature robust-residual support heuristic, clipped to one; zero when absent.'),
    ('local_feature_hilum_associated', 'One when a supplied hilum landmark meets the feature angular-support criterion, otherwise zero.'),
)


def describe_shape_measurement(measurement):
    """Return immutable compact records; never infer raw/corrected image lineage."""
    provenance = measurement.provenance
    if provenance is not None:
        if not isinstance(provenance, ShapeMeasurementProvenance):
            raise ValueError('Measurement provenance must be a typed record.')
        provenance.validate()
    current = provenance is not None and provenance.definition_revision == SHAPE_DEFINITION_REVISION
    revision = SHAPE_DEFINITION_REVISION if current else 'unknown'
    image_frame = provenance.source_image_frame if provenance is not None else 'unknown'
    visibility = provenance.visibility if provenance is not None else 'unknown'
    vector = measurement.vector()
    if len(measurement.measurement_uncertainty) != len(vector):
        raise ValueError('Descriptions require the complete measurement uncertainty vector.')
    records = []

    def record(key, value, definition, unit, frame, uncertainty=None, semantics='unavailable'):
        return DescribedShapeValue(key, value, revision, definition if current else 'unknown',
            unit, frame if current else 'unknown', image_frame, visibility,
            'observed_mask' if current else 'unknown', provenance,
            DescribedUncertainty(uncertainty, unit, semantics if current else 'unknown'))

    for index, ((key, definition), value, error) in enumerate(zip(
            _VECTOR_DEFINITIONS, vector, measurement.measurement_uncertainty)):
        unit = measurement.length_unit if index in (0, 1, 2) else measurement.area_unit if index == 4 else 'dimensionless'
        semantics = 'morphology_rms_combined_in_quadrature_with_shared_scale_sensitivity'
        if current and provenance.boundary_alternative_count == 0:
            if index in (0, 1, 2, 4) and provenance.calibration_relative_uncertainty > 0:
                semantics = 'shared_scale_sensitivity_only_boundary_unavailable'
            else:
                error, semantics = None, 'unavailable_boundary_sensitivity'
        records.append(record(key, value, definition, unit, 'input_mask', error, semantics))
    records.extend((
        record('maximum_span_endpoints', measurement.maximum_span_endpoints,
               'Pixel xy endpoints of the maximum contour-hull span in the supplied mask.', 'px', 'input_mask_pixel_xy'),
        record('ellipse.center_xy', measurement.ellipse.center_xy,
               'Body-ellipse center in supplied-mask pixel xy coordinates.', 'px', 'input_mask_pixel_xy'),
        record('ellipse.orientation_degrees', measurement.ellipse.orientation_degrees,
               'Major-axis direction clockwise from image +x towards +y, modulo 180 degrees.',
               'degrees', 'input_mask_pixel_axes', measurement.ellipse.orientation_uncertainty_degrees,
               'heuristic_axis_ratio_dependent_degrees'),
        record('ellipse.support_fraction', measurement.ellipse.support_fraction,
               'Fraction of contour points retained by the final radial-residual support test.',
               'dimensionless', 'input_mask'),
        record('contour_signature', measurement.contour_signature,
               'Angular bins of median normalized radial residuals in body-ellipse axes.',
               'dimensionless', 'body_ellipse_axes'),
    ))
    if measurement.local_feature is not None:
        records.append(record('local_feature.angle_degrees', measurement.local_feature.angle_degrees,
            'Selected signature bin-center angle clockwise from the body major axis, modulo 360 degrees.',
            'degrees', 'body_ellipse_axes'))
    return tuple(records)
