"""Repeated biological views must be normalized before their units are mixed."""

from tests.functional_units import functional_units
from dataclasses import replace
import unittest

import numpy as np

from seedvision.measurement import shape_model as model
from seedvision.reference_library.contracts import ShapeObservation


def observation(source, physical_id, dimensions, *, calibrated=False):
    span, length, width, area = dimensions
    vector = (span, length, width, length / width, area, *([0.2] * 14))
    errors = (0.5, length * 0.1, width * 0.1, 0.02, area * 0.1, *([0.01] * 14))
    return ShapeObservation(source, source + 1, physical_id, ('soybean',), 'flat',
                            'complete', calibrated, vector, errors, ())


def rescale(item, factor):
    def scaled(values):
        return tuple(value * factor if index in (0, 1, 2)
                     else value * factor * factor if index == 4 else value
                     for index, value in enumerate(values))
    return replace(item, measurement=scaled(item.measurement),
                   measurement_uncertainty=scaled(item.measurement_uncertainty))


@functional_units
class ShapeViewNormalizationTests(unittest.TestCase):
    def check_test_differing_views_are_normalized_before_biological_mean_and_rms(self):
        a = observation(0, 'one', (10, 5, 2, 20))
        b = observation(1, 'one', (20, 16, 6, 160))
        vectors, _, errors, sources = model._physical_seed_groups(
            (a, b), normalize_per_view=True)
        np.testing.assert_allclose(vectors[0, (0, 1, 2, 4)], (1, .65, .25, .3))
        self.assertAlmostEqual(errors[0, 1], np.sqrt((.05**2 + .08**2) / 2))
        self.assertEqual(errors[0, 0], 0)
        self.assertEqual(sources, 2)

    def check_test_unit_changes_preserve_groups_and_one_row_per_physical_seed(self):
        a = observation(0, 'one', (10, 5, 2, 20))
        b = observation(1, 'one', (20, 16, 6, 160))
        c = observation(2, 'two', (14, 8, 4, 50))
        expected = model._physical_seed_groups((a, b, c), normalize_per_view=True)
        for factor in (1e-12, .01, 1000):
            actual = model._physical_seed_groups((a, rescale(b, factor), c),
                                                 normalize_per_view=True)
            np.testing.assert_allclose(actual[0], expected[0], rtol=2e-15)
            np.testing.assert_allclose(actual[2], expected[2], rtol=2e-15)
            self.assertEqual(actual[0].shape, (2, 19))
            self.assertEqual(actual[3], 3)

    def check_test_calibrated_group_filters_uncalibrated_views_without_normalizing(self):
        a = observation(0, 'one', (10, 5, 2, 20), calibrated=True)
        b = observation(1, 'one', (20, 16, 6, 160))
        c = observation(2, 'one', (14, 8, 4, 50), calibrated=True)
        vectors, _, errors, sources = model._physical_seed_groups(
            (a, b, c), calibrated_only=True)
        np.testing.assert_array_equal(vectors[0], np.mean((a.measurement, c.measurement), axis=0))
        np.testing.assert_array_equal(errors[0], np.sqrt(np.mean(
            np.square((a.measurement_uncertainty, c.measurement_uncertainty)), axis=0)))
        self.assertEqual(sources, 2)

    def check_test_population_fallback_and_calibrated_augmentation_are_unit_invariant(self):
        for calibrated in (False, True):
            a = observation(0, 'one', (10, 5, 2, 20), calibrated=calibrated)
            b = observation(1, 'one', (20, 16, 6, 160))
            c = observation(2, 'two', (14, 8, 4, 50), calibrated=calibrated)
            expected = model.fit_species_dimensions_shape_bank((a, b, c)).components[0]
            actual = model.fit_species_dimensions_shape_bank((a, rescale(b, 1e-12), c)).components[0]
            np.testing.assert_allclose(actual.mean, expected.mean, rtol=2e-14, atol=2e-14)
            np.testing.assert_allclose(actual.covariance, expected.covariance, rtol=2e-13, atol=2e-13)
            np.testing.assert_allclose(actual.mean_uncertainty, expected.mean_uncertainty, rtol=2e-14)
            self.assertEqual(actual.physical_dimensions_available, calibrated)
            self.assertEqual(actual.calibrated_physical_seed_count, 2 if calibrated else 0)
            self.assertEqual(actual.dimensionless_physical_seed_count, 2)
            self.assertEqual(actual.source_count, 2 if calibrated else 3)

    def check_test_extreme_squared_span_intermediate_can_normalize_finite_area(self):
        for span, area, expected in ((1e160, 1e308, 1e-12), (1e-200, 1e-320, 1e80)):
            values = np.ones((1, 19)); values[0, (0, 1, 2, 4)] = (span, span, span, area)
            actual, errors = model._dimensionless_vectors(values, np.zeros_like(values))
            self.assertAlmostEqual(actual[0, 4] / expected, 1, delta=2e-5)
            np.testing.assert_array_equal(actual[0, :3], (1, 1, 1))
            self.assertTrue(np.isfinite(errors).all())

    def check_test_invalid_span_or_unrepresentable_normalized_value_is_withheld(self):
        for span in (0, -1, np.inf, np.nan):
            values = np.ones((1, 19)); values[0, 0] = span
            with self.assertRaisesRegex(ValueError, 'positive spans'):
                model._dimensionless_vectors(values, np.zeros_like(values))
        for span, length in ((1e300, 1e-300), (1e-300, 1e300)):
            values = np.ones((1, 19)); values[0, 0] = span; values[0, 1] = length
            with self.assertRaisesRegex(ValueError, 'not representable'):
                model._dimensionless_vectors(values, np.zeros_like(values))

    def check_test_group_rms_retains_finite_errors_when_their_squares_are_unrepresentable(self):
        for error in (1e200, 1e-200):
            a = observation(0, 'one', (10, 5, 2, 20))
            errors = list(a.measurement_uncertainty)
            errors[3] = error
            a = replace(a, measurement_uncertainty=tuple(errors))
            b = replace(a, source_index=1, seed_id=2)
            vectors, _, grouped_errors, _ = model._physical_seed_groups(
                (a, b), normalize_per_view=True)
            self.assertEqual(grouped_errors[0, 3], error)
            self.assertTrue(np.isfinite(vectors).all())
    def check_test_group_mean_retains_finite_values_when_the_sum_overflows(self):
        a = observation(0, 'one', (10, 5, 2, 20))
        values = list(a.measurement); values[3] = 1e308
        a = replace(a, measurement=tuple(values))
        b = replace(a, source_index=1, seed_id=2)
        vectors, _, _, _ = model._physical_seed_groups((a, b), normalize_per_view=True)
        self.assertEqual(vectors[0, 3], 1e308)

    def test_workflow_lifecycle(self):
        """7 related ShapeViewNormalizationTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_differing_views_are_normalized_before_biological_mean_and_rms)
        self.scenario(self.check_test_unit_changes_preserve_groups_and_one_row_per_physical_seed)
        self.scenario(self.check_test_calibrated_group_filters_uncalibrated_views_without_normalizing)
        self.scenario(self.check_test_population_fallback_and_calibrated_augmentation_are_unit_invariant)
        self.scenario(self.check_test_extreme_squared_span_intermediate_can_normalize_finite_area)
        self.scenario(self.check_test_group_rms_retains_finite_errors_when_their_squares_are_unrepresentable)
        self.scenario(self.check_test_group_mean_retains_finite_values_when_the_sum_overflows)

    def test_validation_and_recovery(self):
        """1 related ShapeViewNormalizationTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_invalid_span_or_unrepresentable_normalized_value_is_withheld)
