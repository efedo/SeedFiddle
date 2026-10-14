"""Conditioned pooling uses the same Gaussian model without dropping directions."""
from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np

from seedvision.measurement import shape_model as model
from tests.test_shape_update_availability import fixture, update
from tests.test_local_shape_units import measurements


def covariance_component(component, covariance, mean=None):
    return replace(component, covariance=tuple(tuple(row) for row in covariance),
                   mean=component.mean if mean is None else tuple(mean))


class ConditionedShapePoolingTests(unittest.TestCase):
    def test_anisotropic_diagonal_pooling_matches_independent_scalar_gaussian_oracle(self):
        values, component = fixture()
        vp = np.geomspace(1e-24, 1e24, 19)
        vl = vp * np.linspace(.5, 3, 19)
        mp = np.linspace(1, 2, 19); ml = np.linspace(3, 5, 19)
        prior = covariance_component(component, np.diag(vp), mp)
        local = covariance_component(component, np.diag(vl), ml)
        result = update(prior, local, values, shrinkage=3.)
        n = len(values) / 4
        gain = vp * n / (vl + vp * n)
        expected_variance = vp * vl / (vl + vp * n)
        self.assertEqual(result.state, 'applied')
        self.assertEqual(result.pooling_method, 'coordinate_conditioned_precision')
        np.testing.assert_allclose(result.component.mean, mp + gain * (ml - mp), rtol=2e-14)
        np.testing.assert_allclose(np.diag(result.component.covariance), expected_variance, rtol=2e-14)
        self.assertEqual(result.component.effective_physical_seed_count, 2 + n)
        self.assertEqual(result.component.calibrated_physical_seed_count, 2 + n)

    def test_correlated_anisotropic_block_matches_small_matrix_oracle_and_stays_psd(self):
        values, component = fixture()
        scales = np.ones(19); scales[:2] = [1e-12, 1e12]
        cp = np.eye(19); cl = np.eye(19) * 2
        cp[:2, :2] = [[1, .3], [.3, 1]]
        cl[:2, :2] = [[2, -.2], [-.2, 3]]
        mp = np.linspace(1, 2, 19); ml = np.linspace(3, 4, 19)
        prior = covariance_component(component, cp * scales[:, None] * scales[None, :], mp * scales)
        local = covariance_component(component, cl * scales[:, None] * scales[None, :], ml * scales)
        result = update(prior, local, values)
        self.assertEqual(result.state, 'applied')
        expected = np.linalg.inv(np.linalg.inv(cp[:2, :2]) + np.linalg.inv(cl[:2, :2]))
        expected_mean = expected @ (np.linalg.inv(cp[:2, :2]) @ mp[:2] +
                                    np.linalg.inv(cl[:2, :2]) @ ml[:2])
        normalized = (np.asarray(result.component.covariance) / scales[:, None]) / scales[None, :]
        np.testing.assert_allclose(normalized[:2, :2], expected, rtol=2e-13, atol=1e-15)
        np.testing.assert_allclose(np.asarray(result.component.mean)[:2] / scales[:2], expected_mean, rtol=2e-13)
        np.testing.assert_allclose(normalized, normalized.T, rtol=0, atol=1e-15)
        self.assertGreater(np.linalg.eigvalsh(normalized).min(), 0)

    def test_coordinate_unit_congruence_transforms_mean_covariance_and_conditioning_scales(self):
        values, component = fixture()
        covariance = np.diag(np.geomspace(1e-18, 1e4, 19))
        prior = covariance_component(component, covariance)
        local = covariance_component(component, covariance * 2, np.asarray(component.mean) * 1.1)
        original = update(prior, local, values)
        units = np.ones(19); units[[0, 1, 2]] = .01; units[4] = .01**2
        def convert(c):
            return covariance_component(c, np.asarray(c.covariance) * units[:, None] * units[None, :],
                                        np.asarray(c.mean) * units)
        converted = update(convert(prior), convert(local), values)
        self.assertEqual(original.state, 'applied')
        self.assertEqual(converted.state, 'applied')
        self.assertEqual(converted.pooling_method, 'coordinate_conditioned_precision')
        np.testing.assert_allclose(converted.component.mean, np.asarray(original.component.mean) * units, rtol=2e-13)
        np.testing.assert_allclose(converted.component.covariance,
            np.asarray(original.component.covariance) * units[:, None] * units[None, :], rtol=2e-13, atol=0)
        np.testing.assert_allclose(converted.conditioning_scales,
                                   np.asarray(original.conditioning_scales) * units, rtol=2e-14)

    def test_actual_mask_fits_apply_with_shrinkage_and_keep_all_nonzero_shape_coordinates(self):
        for scale in (None, 2., 20., 1e-79):
            current = measurements(scale=scale)
            prior = model._local_component(current, 'flat')
            for shrinkage in (0., 3., 50.):
                result = model._update_component_result(prior, current, 'flat',
                    shrinkage_seed_count=shrinkage, maximum_contour_modes=0)
                self.assertEqual(result.state, 'applied')
                self.assertEqual(result.pooling_method, 'coordinate_conditioned_precision')
                np.testing.assert_array_equal(result.component.mean, prior.mean)
                np.testing.assert_allclose(result.component.covariance,
                    np.asarray(prior.covariance) / (1 + len(current)/(1+shrinkage)), rtol=3e-12, atol=1e-30)
                self.assertGreater(result.component.mean[3], 1)
                self.assertEqual(result.component.physical_fit_unavailability_reason,
                                 prior.physical_fit_unavailability_reason)

    def test_actual_different_mask_cohorts_use_shrinkage_in_conditioned_mean(self):
        prior = model._local_component(measurements((28, 10), 2.), 'flat')
        current = measurements((20, 15), 2.)
        low = model._update_component_result(prior, current, 'flat', shrinkage_seed_count=0, maximum_contour_modes=0)
        high = model._update_component_result(prior, current, 'flat', shrinkage_seed_count=50, maximum_contour_modes=0)
        self.assertEqual(low.state, 'applied'); self.assertEqual(high.state, 'applied')
        self.assertLess(abs(high.component.mean[3] - prior.mean[3]), abs(low.component.mean[3] - prior.mean[3]))
        self.assertGreater(low.component.covariance[18][18], 0)

    def test_large_identical_offset_remains_exact_without_scaling_the_offset(self):
        values, component = fixture()
        covariance = np.eye(19); covariance[0, 0] = 1e-200
        prior = covariance_component(component, covariance, (1e300,)*19)
        result = update(prior, prior, values)
        self.assertEqual(result.state, 'applied')
        np.testing.assert_array_equal(result.component.mean, prior.mean)
        self.assertGreater(result.component.covariance[0][0], 0)

    def test_conditioned_signed_mean_cancellation_is_valid(self):
        values, component = fixture(False)
        covariance = np.eye(19); covariance[15, 15] = 1e-200
        mp = np.asarray(component.mean).copy(); mp[15] = 1e-100
        ml = mp.copy(); ml[15] = -1e-100
        result = update(covariance_component(component, covariance, mp),
                        covariance_component(component, covariance, ml), values)
        self.assertEqual(result.state, 'applied')
        self.assertLessEqual(abs(result.component.mean[15]),
                             4 * np.finfo(np.float64).eps * abs(mp[15]))
        self.assertGreater(result.component.covariance[15][15], 0)

    def test_nearly_singular_correlation_and_zero_variance_disagreement_still_abstain(self):
        values, component = fixture()
        for correlation in (1., 1-1e-16):
            covariance = np.eye(19); covariance[0, 1] = covariance[1, 0] = correlation
            covariance[18, 18] = 1e-30
            prior = covariance_component(component, covariance)
            result = update(prior, prior, values)
            self.assertEqual(result.state, 'unavailable')
            self.assertEqual(result.unavailability_reason, 'singular_precision')
            self.assertIs(result.component, prior)
        covariance = np.eye(19); covariance[3, 3] = 0
        prior = covariance_component(component, covariance)
        local = replace(prior, mean=tuple(np.asarray(prior.mean) + 1))
        result = update(prior, local, values)
        self.assertIs(result.component, prior)
        self.assertEqual(result.state, 'unavailable')

    def test_conditioned_solver_failure_preserves_exact_support_and_unrelated_error_propagates(self):
        values, component = fixture()
        covariance = np.eye(19); covariance[18, 18] = 1e-30
        prior = covariance_component(component, covariance)
        with patch.object(np.linalg, 'cholesky', side_effect=np.linalg.LinAlgError('solver failed')):
            result = update(prior, prior, values)
        self.assertEqual(result.unavailability_reason, 'numeric_range')
        self.assertIs(result.component, prior)
        self.assertEqual(result.component.effective_physical_seed_count, 2)
        with patch.object(np.linalg, 'cholesky', side_effect=ValueError('unrelated solver defect')):
            with self.assertRaisesRegex(ValueError, 'unrelated solver defect'):
                update(prior, prior, values)

    def test_conditioning_metadata_is_detached_and_cannot_claim_unapplied_conditioning(self):
        _, component = fixture(); scales = [1.]*19
        outcome = model.ShapeComponentUpdate(component, 'applied', 'library+local',
            pooling_method='coordinate_conditioned_precision', conditioning_scales=scales)
        scales[0] = 5
        self.assertEqual(outcome.conditioning_scales[0], 1)
        for state, method, values in [('unavailable', 'coordinate_conditioned_precision', [1.]*19),
                ('applied', 'unknown', []), ('applied', 'legacy_precision', [1.]*19),
                ('applied', 'coordinate_conditioned_precision', [1.]*18),
                ('applied', 'coordinate_conditioned_precision', [float('inf')]*19)]:
            with self.subTest(state=state, method=method), self.assertRaises(ValueError):
                model.ShapeComponentUpdate(component, state, 'library+local',
                    'numeric_range' if state == 'unavailable' else None,
                    pooling_method=method, conditioning_scales=values)
