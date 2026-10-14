"""Value descriptions preserve calculation facts without inventing lineage."""
from dataclasses import FrozenInstanceError, asdict, replace
import json
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from seedvision.measurement import geometry as g
from seedvision.measurement.descriptions import SHAPE_DEFINITION_REVISION, ShapeMeasurementProvenance
from seedvision.measurement.shape_model import measure_reviewed_seed_instances, shape_observations_from_summary
from seedvision.persistence.reference_regions import SeedInstanceAnnotation
from seedvision.reference_library.contracts import BiologicalContext, SHAPE_MEASUREMENT_NAMES


def mask():
    value = np.zeros((96, 128), np.uint8)
    cv2.ellipse(value, (64, 48), (24, 14), 25, 0, 360, 1, -1)
    return value


class MeasurementDescriptionTests(unittest.TestCase):
    def test_vector_descriptions_bind_current_values_definitions_and_uncertainties(self):
        item = g.measure_shape_mask(mask(), perturbation_radii=(0, 1, -2, 2),
                                    calibration_relative_uncertainty=.02)
        records = item.described_values
        self.assertEqual(tuple(r.key for r in records[:19]), SHAPE_MEASUREMENT_NAMES)
        self.assertEqual(tuple(r.value for r in records[:19]), item.vector())
        self.assertEqual(tuple(r.uncertainty.value for r in records[:19]), item.measurement_uncertainty)
        self.assertEqual(item.provenance.perturbation_radii_px, (1, 2))
        self.assertEqual(item.provenance.boundary_alternative_count, 8)
        self.assertEqual(item.provenance.calibration_relative_uncertainty, .02)
        self.assertTrue(all(r.definition_revision == SHAPE_DEFINITION_REVISION for r in records))
        self.assertTrue(all(r.definition != 'unknown' for r in records))
        self.assertTrue(all(r.source_image_frame == 'unknown' and r.visibility == 'unknown' for r in records))
        json.dumps([asdict(r) for r in records], allow_nan=False)

    def test_scaling_preserves_pixel_coordinates_and_dimensionless_units(self):
        pixels = g.measure_shape_mask(mask())
        scaled = g._scale_shape_measurement(pixels, 2.5)
        self.assertEqual(scaled.provenance.pixels_per_mm, 2.5)
        self.assertEqual(scaled.provenance.calibration_relative_uncertainty,
                         pixels.provenance.calibration_relative_uncertainty)
        self.assertEqual(pixels.provenance.pixels_per_mm, None)
        px = {r.key: r for r in pixels.described_values}
        mm = {r.key: r for r in scaled.described_values}
        for index, name in enumerate(SHAPE_MEASUREMENT_NAMES):
            unit = 'mm' if index in (0, 1, 2) else 'mm²' if index == 4 else 'dimensionless'
            self.assertEqual(mm[name].unit, unit)
            self.assertEqual(mm[name].uncertainty.unit, unit)
            if index not in (0, 1, 2, 4):
                self.assertEqual(mm[name].value, px[name].value)
                self.assertEqual(mm[name].uncertainty, px[name].uncertainty)
        for name in ('maximum_span_endpoints', 'ellipse.center_xy'):
            self.assertEqual(mm[name].value, px[name].value)
            self.assertEqual(mm[name].uncertainty, px[name].uncertainty)
            self.assertEqual(mm[name].coordinate_frame, 'input_mask_pixel_xy')
            self.assertEqual(mm[name].unit, 'px')

    def test_legacy_bundle_has_unknown_definitions_and_keeps_supplied_errors(self):
        current = g.measure_shape_mask(mask())
        legacy = replace(current, provenance=None)
        self.assertEqual(legacy.vector(), current.vector())
        for record in legacy.described_values:
            self.assertEqual(record.definition_revision, 'unknown')
            self.assertEqual(record.definition, 'unknown')
            self.assertEqual(record.coordinate_frame, 'unknown')
            self.assertEqual(record.measurement_basis, 'unknown')
            self.assertEqual(record.uncertainty.semantics, 'unknown')
        self.assertEqual(tuple(r.uncertainty.value for r in legacy.described_values[:19]),
                         legacy.measurement_uncertainty)
        self.assertIsNone(g._scale_shape_measurement(legacy, 2).provenance)

    def test_no_boundary_alternatives_withhold_sensitivity_instead_of_claiming_zero(self):
        item = g.measure_shape_mask(mask(), perturbation_radii=())
        self.assertEqual(item.provenance.boundary_alternative_count, 0)
        self.assertEqual(item.provenance.perturbation_radii_px, ())
        self.assertTrue(all(r.uncertainty.value is None for r in item.described_values[:19]))
        records = {r.key: r for r in item.described_values}
        self.assertIsNone(records['ellipse.center_xy'].uncertainty.value)
        self.assertEqual(records['ellipse.center_xy'].uncertainty.semantics, 'unavailable')
        self.assertEqual(records['ellipse.orientation_degrees'].uncertainty.value,
                         item.ellipse.orientation_uncertainty_degrees)
        self.assertEqual(records['ellipse.orientation_degrees'].uncertainty.semantics,
                         'heuristic_axis_ratio_dependent_degrees')

    def test_no_successful_alternative_is_recorded_without_inventing_a_bound(self):
        original = g._shape_from_contour
        calls = 0
        def fail_alternatives(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls > 1:
                raise ValueError('declared contour fit failure')
            return original(*args, **kwargs)
        with patch.object(g, '_shape_from_contour', side_effect=fail_alternatives):
            item = g.measure_shape_mask(mask(), perturbation_radii=(1,))
        self.assertEqual(item.provenance.perturbation_radii_px, (1,))
        self.assertEqual(item.provenance.boundary_alternative_count, 0)
        self.assertTrue(all(r.uncertainty.value is None for r in item.described_values[:19]))

    def test_reviewed_visibility_reaches_pixel_and_mm_bundles_without_image_lineage(self):
        for visibility in ('complete', 'partly_occluded'):
            annotation = SeedInstanceAnnotation(1, shape_reviewed=True,
                outline_visibility=visibility, pose='flat', full_length_visible=True)
            summary = measure_reviewed_seed_instances(mask().astype(np.uint16), (annotation,), pixels_per_mm=2)
            item = summary.observations[0]
            self.assertEqual(item.eligible, visibility == 'complete')
            for measurement in (item.pixel_measurement, item.calibrated_measurement):
                self.assertTrue(all(r.visibility == visibility for r in measurement.described_values))
                self.assertTrue(all(r.source_image_frame == 'unknown' for r in measurement.described_values))
            observations = shape_observations_from_summary(summary, source_index=0,
                biological_context=BiologicalContext(species_id='soybean'))
            self.assertEqual(len(observations), 1 if visibility == 'complete' else 0)
            if observations:
                self.assertNotIn('provenance', asdict(observations[0]))
                self.assertEqual(observations[0].measurement, item.calibrated_measurement.vector())

    def test_records_and_provenance_are_immutable_compact_metadata(self):
        item = g.measure_shape_mask(mask())
        with self.assertRaises(FrozenInstanceError):
            item.provenance.visibility = 'complete'
        record = item.described_values[0]
        with self.assertRaises(FrozenInstanceError):
            record.unit = 'mm'
        with self.assertRaises(FrozenInstanceError):
            record.uncertainty.value = 0
        with patch.object(g, 'image_to_tensor', side_effect=AssertionError('unexpected upload')):
            self.assertEqual(item.described_values[0], record)

    def test_scale_sensitivity_remains_available_when_boundary_sensitivity_is_missing(self):
        item = g.measure_shape_mask(mask(), perturbation_radii=(),
                                    calibration_relative_uncertainty=.02, pixels_per_mm=2)
        for index, record in enumerate(item.described_values[:19]):
            if index in (0, 1, 2, 4):
                self.assertEqual(record.uncertainty.value, item.measurement_uncertainty[index])
                self.assertEqual(record.uncertainty.semantics,
                                 'shared_scale_sensitivity_only_boundary_unavailable')
            else:
                self.assertIsNone(record.uncertainty.value)

    def test_invalid_provenance_cannot_emit_current_calculation_claims(self):
        valid = ShapeMeasurementProvenance(SHAPE_DEFINITION_REVISION, (1, 2), 8, .02)
        for changes in ({'perturbation_radii_px': [1]}, {'perturbation_radii_px': (True,)},
                        {'boundary_alternative_count': 9}, {'boundary_alternative_count': -1},
                        {'calibration_relative_uncertainty': np.inf}, {'pixels_per_mm': 0},
                        {'pixels_per_mm': True}, {'visibility': 'complete_inferred'}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(valid, **changes)
        item = g.measure_shape_mask(mask())
        object.__setattr__(item.provenance, 'boundary_alternative_count', -1)
        with self.assertRaisesRegex(ValueError, 'alternative count'):
            _ = item.described_values

    def test_unknown_revision_does_not_inherit_current_definitions(self):
        item = g.measure_shape_mask(mask())
        foreign = replace(item, provenance=replace(item.provenance,
                           definition_revision='legacy-or-unknown-revision'))
        self.assertTrue(all(r.definition == 'unknown' and r.uncertainty.semantics == 'unknown'
                            for r in foreign.described_values))
