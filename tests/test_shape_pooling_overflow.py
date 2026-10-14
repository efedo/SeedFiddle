"""High-precision Gaussian oracles for representable intermediate-overflow cases."""
from dataclasses import replace
from decimal import Decimal, localcontext
import unittest

import numpy as np

from seedvision.measurement import shape_model as model
from tests.test_conditioned_shape_pooling import covariance_component
from tests.test_shape_update_availability import fixture, update


def oracle(prior_variance, local_variance, prior_mean, local_mean, weight):
    with localcontext() as context:
        context.prec = 700
        p, r, a, b, w = map(Decimal.from_float,
            map(float, (prior_variance, local_variance, prior_mean, local_mean, weight)))
        denominator = r + w * p
        variance = p * r / denominator
        mean = (r * a + w * p * b) / denominator
        radius = Decimal.from_float(1.96) * (variance + p).sqrt()
        return float(mean), float(variance), (float(mean - radius), float(mean + radius))


class ShapePoolingOverflowTests(unittest.TestCase):
    def test_preventable_overflow_matches_decimal_gaussian_outputs_and_support(self):
        values, component = fixture()
        for p, r, a, b in ((1e-308, 1e-308, 1., 1.), (.5, .5, 1e308, 1e308),
                           (1.6e308, 1.6e308, 1., 1.), (1e-306, 2e-306, 1e100, 1e100)):
            for shrinkage in (0., 1., 3.):
                prior = covariance_component(component, np.eye(19) * p, (a,) * 19)
                local = covariance_component(component, np.eye(19) * r, (b,) * 19)
                result = update(prior, local, values, shrinkage)
                expected_mean, expected_var, expected_interval = oracle(p, r, a, b, len(values)/(1+shrinkage))
                with self.subTest(p=p, r=r, shrinkage=shrinkage):
                    self.assertEqual(result.state, 'applied')
                    np.testing.assert_allclose(result.component.mean, expected_mean, rtol=3e-14, atol=0)
                    np.testing.assert_allclose(np.diag(result.component.covariance), expected_var, rtol=3e-14, atol=0)
                    np.testing.assert_allclose(result.component.predictive_interval_95,
                        np.tile(expected_interval, (19, 1)), rtol=3e-14, atol=1e-300)
                    self.assertEqual(result.component.effective_physical_seed_count,
                        component.effective_physical_seed_count + len(values)/(1+shrinkage))
                    self.assertEqual(result.component.physical_fit_unavailability_reason,
                        component.physical_fit_unavailability_reason)

    def test_opposite_huge_means_are_scaled_before_overflowing_subtraction(self):
        values, component = fixture()
        for shrinkage in (0., 1., 3.):
            prior = covariance_component(component, np.eye(19) * 1.6e308, (-1e308,) * 19)
            local = covariance_component(component, np.eye(19) * 1.6e308, (1e308,) * 19)
            result = update(prior, local, values, shrinkage)
            mean, variance, _ = oracle(1.6e308, 1.6e308, -1e308, 1e308, len(values)/(1+shrinkage))
            self.assertEqual(result.state, 'applied')
            np.testing.assert_allclose(result.component.mean, mean, rtol=1e-13, atol=4e292)
            np.testing.assert_allclose(np.diag(result.component.covariance), variance, rtol=3e-14)
            conditioned, scales = model._guarded_shape_pool(model._pool_conditioned_same_dimension_components,
                prior, local, 'flat', len(values)/(1+shrinkage))
            np.testing.assert_allclose(conditioned.mean, mean, rtol=1e-13, atol=4e292)
            self.assertEqual(len(scales), 19)

    def test_variance_boundary_continuity_and_positive_subnormal_output(self):
        values, component = fixture()
        for variance in (1e-307, 2e-308, 1e-308, 1e-310, 1e-320):
            prior = covariance_component(component, np.eye(19) * variance, (1.,) * 19)
            result = update(prior, prior, values)
            self.assertEqual(result.state, 'applied')
            np.testing.assert_allclose(np.diag(result.component.covariance), variance / 2, rtol=1e-12, atol=np.nextafter(0., 1.))
            np.testing.assert_allclose(result.component.mean, prior.mean, rtol=4*np.finfo(float).eps, atol=0)
            self.assertGreater(result.component.covariance[0][0], 0)

    def test_unit_congruence_preserves_rescued_gaussian(self):
        values, component = fixture()
        prior = covariance_component(component, np.eye(19) * 1e-308, (1.,) * 19)
        local = covariance_component(component, np.eye(19) * 2e-308, (2.,) * 19)
        original = update(prior, local, values)
        units = np.ones(19); units[[0, 1, 2]] = 1e50; units[4] = 1e100
        def convert(c):
            return covariance_component(c, np.asarray(c.covariance) * units[:, None] * units[None, :],
                                        np.asarray(c.mean) * units)
        converted = update(convert(prior), convert(local), values)
        self.assertEqual(original.state, 'applied')
        self.assertEqual(converted.state, 'applied')
        np.testing.assert_allclose(converted.component.mean, np.asarray(original.component.mean) * units, rtol=1e-12)
        np.testing.assert_allclose(converted.component.covariance,
            np.asarray(original.component.covariance) * units[:, None] * units[None, :], rtol=1e-12, atol=0)

    def test_unrepresentable_restored_variance_or_support_retains_exact_prior(self):
        values, component = fixture()
        smallest = np.nextafter(0., 1.)
        prior = covariance_component(component, np.eye(19) * smallest, (1.,) * 19)
        result = update(prior, prior, values, shrinkage=0)
        self.assertEqual(result.state, 'unavailable')
        self.assertEqual(result.unavailability_reason, 'numeric_range')
        self.assertIs(result.component, prior)
        prior = replace(component, effective_physical_seed_count=1e308)
        result = update(prior, component, values)
        self.assertEqual(result.state, 'unavailable')
        self.assertIs(result.component, prior)

    def test_native_result_serializes_rescued_update_without_refitting(self):
        import json
        from unittest.mock import patch
        from seedvision.export.results import result_report
        from tests.test_shape_model_export import analysis_with_outcome
        values, component = fixture()
        prior = covariance_component(component, np.eye(19) * 1e-308, (1.,) * 19)
        outcome = update(prior, prior, values)
        with patch.object(model, '_update_component_result', side_effect=AssertionError('Export refit')):
            report = result_report(analysis_with_outcome(outcome))
        encoded = json.dumps(report, allow_nan=False)
        restored = json.loads(encoded)['shape_model']['families'][0]
        self.assertEqual(restored['update_state'], 'applied')
        self.assertEqual(restored['pooling_method'], 'coordinate_conditioned_precision')
        self.assertIsNone(restored['update_unavailability_reason'])
        self.assertEqual(restored['component']['effective_physical_seed_count'],
            outcome.component.effective_physical_seed_count)

    def test_mixed_unrepresentable_support_retains_exact_physical_shell(self):
        for prior_physical in (False, True):
            for field in ('effective_physical_seed_count', 'dimensionless_physical_seed_count',
                          'calibrated_physical_seed_count'):
                values, prior = fixture(prior_physical)
                _, local = fixture(not prior_physical)
                if field == 'calibrated_physical_seed_count':
                    # Both directions contain a nonzero calibrated increment.
                    if prior_physical:
                        prior = replace(prior, **{field: 1e308})
                        local = replace(local, **{field: 7.})
                    else:
                        prior = replace(prior, **{field: 7.})
                        local = replace(local, **{field: 1e308})
                else:
                    prior = replace(prior, **{field: 1e308})
                physical = prior if prior_physical else local
                with self.subTest(prior_physical=prior_physical, field=field):
                    result = update(prior, local, values)
                    self.assertEqual(result.state, 'unavailable')
                    self.assertEqual(result.unavailability_reason, 'numeric_range')
                    self.assertIs(result.component, physical)
                    self.assertEqual(result.source, 'library' if prior_physical else 'local')


if __name__ == '__main__':
    unittest.main()
