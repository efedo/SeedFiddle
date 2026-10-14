"""Candidate scoring distinguishes computational abstention from incompatibility."""
from dataclasses import asdict, replace
import json
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from seedvision.measurement import shape_model as model
from seedvision.measurement.geometry import measure_shape_mask


def fixture(physical=True):
    mask = np.zeros((96, 128), np.uint8)
    cv2.ellipse(mask, (64, 48), (24, 14), 17, 0, 360, 1, -1)
    measurement = measure_shape_mask(mask, pixels_per_mm=10 if physical else None)
    component = model._local_component((measurement, measurement), 'flat')
    return np.asarray(measurement.vector()), component


def diagonal_component(component, diagonal=1.):
    return replace(component, covariance=tuple(tuple(row) for row in np.eye(19) * diagonal))


class ShapeScoringAvailabilityTests(unittest.TestCase):
    def assert_unavailable(self, vector, component, reason='numeric_range'):
        assessment = model.assess_candidate_shape(vector, component)
        self.assertFalse(assessment.scoring_available)
        self.assertIsNone(assessment.compatibility)
        self.assertIsNone(assessment.standardized_distance)
        self.assertIsNone(assessment.out_of_family)
        self.assertEqual(assessment.scoring_unavailability_reason, reason)
        self.assertIsNone(model.candidate_shape_compatibility(vector, component))
        self.assertEqual(json.loads(json.dumps(asdict(assessment), allow_nan=False))['compatibility'], None)
        return assessment

    def test_ordinary_diagonal_distance_has_known_physical_and_dimensionless_values(self):
        for physical in (True, False):
            vector, component = fixture(physical)
            component = diagonal_component(component)
            vector[3] += 2
            assessment = model.assess_candidate_shape(vector, component)
            dimension = 19 if physical else 15
            expected = 4 / (1 + dimension * 1e-6)
            self.assertTrue(assessment.scoring_available)
            self.assertAlmostEqual(assessment.standardized_distance, expected, places=14)
            self.assertAlmostEqual(assessment.compatibility, np.exp(-.5 * expected / dimension), places=14)
            self.assertFalse(assessment.out_of_family)
            self.assertEqual(model.candidate_shape_compatibility(vector, component), assessment.compatibility)

    def test_identical_mask_is_available_and_distinct_outlier_is_not_numeric_abstention(self):
        vector, component = fixture()
        perfect = model.assess_candidate_shape(vector, component)
        self.assertTrue(perfect.scoring_available)
        self.assertEqual(perfect.compatibility, 1)
        self.assertEqual(perfect.standardized_distance, 0)
        self.assertFalse(perfect.out_of_family)
        vector[3] += 100
        outlier = model.assess_candidate_shape(vector, diagonal_component(component))
        self.assertTrue(outlier.scoring_available)
        self.assertTrue(outlier.out_of_family)
        self.assertGreater(outlier.compatibility, 0)
        self.assertLess(outlier.compatibility, .2)

    def test_finite_psd_covariance_trace_overflow_is_unavailable(self):
        vector, component = fixture()
        self.assert_unavailable(vector, diagonal_component(component, 1e308))

    def test_finite_delta_and_quadratic_overflow_do_not_become_a_capped_score(self):
        vector, component = fixture()
        component = diagonal_component(component)
        vector[0] = 1e308
        self.assert_unavailable(vector, component)
        mean = list(component.mean); mean[16] = -1e308
        component = replace(component, mean=tuple(mean))
        vector[16] = 1e308
        self.assert_unavailable(vector, component)

    def test_positive_quadratic_underflow_cannot_claim_perfect_match(self):
        vector, component = fixture()
        component = diagonal_component(component)
        vector[18] = 1e-200
        self.assert_unavailable(vector, component)

    def test_nonfinite_invalid_dimensions_and_wrong_length_have_distinct_contracts(self):
        vector, component = fixture()
        for index, value in ((3, np.nan), (3, np.inf), (0, 0), (1, -1)):
            changed = vector.copy(); changed[index] = value
            self.assert_unavailable(changed, component, 'invalid_measurement')
        for changed in (vector[:-1], vector.reshape(1, -1)):
            with self.assertRaisesRegex(ValueError, 'does not match'):
                model.assess_candidate_shape(changed, component)

    def test_nonrepresentable_dimensionless_normalization_is_numeric_unavailability(self):
        vector, component = fixture(False)
        vector[0] = 1e300; vector[1] = 1e-300
        self.assert_unavailable(vector, component)

    def test_asymmetric_and_indefinite_selected_covariance_are_explicitly_invalid(self):
        vector, component = fixture()
        for symmetric in (True, False):
            covariance = np.eye(19)
            if symmetric:
                covariance[3, 3] = -1
            else:
                covariance[3, 5] = 2
            self.assert_unavailable(vector, replace(component,
                covariance=tuple(tuple(row) for row in covariance)), 'invalid_covariance')

    def test_excluded_covariance_block_cannot_disable_dimensionless_scoring(self):
        vector, component = fixture(False)
        component = diagonal_component(component)
        expected = model.assess_candidate_shape(vector, component)
        covariance = np.asarray(component.covariance)
        covariance[0, 0] = -1e308; covariance[0, 1] = 1e308
        covariance[4, 4] = 1e308
        changed = replace(component, covariance=tuple(tuple(row) for row in covariance))
        self.assertEqual(model.assess_candidate_shape(vector, changed), expected)

    def test_linear_algebra_failure_abstains_but_unrelated_error_propagates(self):
        vector, component = fixture()
        with patch.object(np.linalg, 'pinv', side_effect=np.linalg.LinAlgError('did not converge')):
            self.assert_unavailable(vector, component)
        with patch.object(np.linalg, 'pinv', side_effect=ValueError('unrelated defect')):
            with self.assertRaisesRegex(ValueError, 'unrelated defect'):
                model.assess_candidate_shape(vector, component)

    def test_pose_hypotheses_snapshot_generator_once_and_preserve_all_abstentions(self):
        vector, component = fixture()
        families = tuple(model.SeedPoseShapeFamily(pose, replace(component, pose=pose), 2, 'current')
                         for pose in ('flat', 'side', 'oblique'))
        summary = model.measure_reviewed_seed_instances(None, ())
        population = model.SeedDimensionsShapeModel(families, summary, 'current_image_only', (), 50, 50)
        results = population.assess_pose_hypotheses(iter(vector))
        self.assertEqual(len(results), 3)
        self.assertTrue(all(r.compatibility == 1 and r.scoring_available for r in results))
        vector[3] = np.nan
        unavailable = population.assess_pose_hypotheses(iter(vector))
        self.assertEqual(len(unavailable), 3)
        self.assertTrue(all(r.scoring_unavailability_reason == 'invalid_measurement'
                            and r.out_of_family is None for r in unavailable))

    def test_legacy_assessment_constructor_and_invalid_atomic_states(self):
        legacy = model.ShapeCompatibilityAssessment('flat', .5, 2., False)
        self.assertTrue(legacy.scoring_available)
        for values in ((None, None, None, None), (np.nan, 2., False, None),
                       (.5, -1., False, None), (.5, 2., None, None),
                       (None, None, False, 'numeric_range'),
                       (None, None, None, 'invented')):
            with self.assertRaises(ValueError):
                model.ShapeCompatibilityAssessment('flat', *values)
