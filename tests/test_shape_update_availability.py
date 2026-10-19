"""Precision pooling availability is separate from fit/scoring availability."""

from tests.functional_units import functional_units
from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np

from seedvision.measurement import shape_model as model
from seedvision.reference_library.contracts import BiologicalContext, SpeciesDimensionsShapeBank
from tests.test_local_shape_units import measurements
from tests.test_reviewed_measurement_lineage import labels_fixture, annotation


def fixture(physical=True, covariance=1.):
    values = measurements(scale=4. if physical else None)
    component = model._local_component(values, 'flat')
    component = replace(component,
        covariance=tuple(tuple(row) for row in np.eye(19) * covariance))
    return values, component


def update(prior, local, values, shrinkage=1.):
    with patch.object(model, '_local_component', return_value=local):
        return model._update_component_result(prior, values, 'flat',
            shrinkage_seed_count=shrinkage, maximum_contour_modes=0)


@functional_units
class ShapeUpdateAvailabilityTests(unittest.TestCase):
    def assert_unavailable(self, result, baseline, reason):
        self.assertEqual(result.state, 'unavailable')
        self.assertEqual(result.unavailability_reason, reason)
        self.assertIs(result.component, baseline)
        self.assertNotEqual(result.source, 'library+local')
        self.assertEqual(result.component.effective_physical_seed_count,
            baseline.effective_physical_seed_count)

    def check_test_ordinary_same_representation_pooling_preserves_exact_existing_arithmetic(self):
        for physical in (False, True):
            values, prior = fixture(physical, covariance=2.)
            _, local = fixture(physical, covariance=3.)
            local = replace(local, mean=tuple(np.asarray(local.mean) + .1))
            result = update(prior, local, values, shrinkage=5.)
            n = len(values)/6.
            pp = np.linalg.pinv(np.asarray(prior.covariance))
            lp = np.linalg.pinv(np.asarray(local.covariance)) * n
            covariance = np.linalg.pinv(pp + lp)
            mean = covariance @ (pp @ prior.mean + lp @ local.mean)
            self.assertEqual(result.state, 'applied')
            self.assertIsNone(result.unavailability_reason)
            self.assertEqual(result.pooling_method, 'legacy_precision')
            self.assertEqual(result.conditioning_scales, ())
            np.testing.assert_array_equal(result.component.mean, mean)
            np.testing.assert_array_equal(result.component.covariance, covariance)
            self.assertEqual(result.component.effective_physical_seed_count,
                prior.effective_physical_seed_count + n)

    def check_test_same_representation_singular_or_cutoff_direction_cannot_become_zero_mean_certainty(self):
        values, prior = fixture(); _, local = fixture()
        for variance in (0., 1e-16):
            covariance = np.eye(19); covariance[3, 3] = variance
            invalid = replace(prior, covariance=tuple(tuple(row) for row in covariance))
            result = update(invalid, local, values)
            self.assertGreater(result.component.mean[3], 1.)
            if variance == 0:
                self.assert_unavailable(result, invalid, 'singular_precision')
                with patch.object(model, '_local_component', return_value=local), self.assertRaises(model.ShapeUpdateUnavailable):
                    model._update_component(invalid, values, 'flat',
                        shrinkage_seed_count=1, maximum_contour_modes=0)
            else:
                self.assertEqual(result.state, 'applied')
                self.assertEqual(result.pooling_method, 'coordinate_conditioned_precision')
                self.assertGreater(result.component.covariance[3][3], 0)

    def check_test_precision_weighting_underflow_retains_prior_without_counting_unused_local_fit(self):
        values, prior = fixture(); _, local = fixture(covariance=1e308)
        result = update(prior, local, values, shrinkage=1e308)
        self.assert_unavailable(result, prior, 'numeric_range')
        self.assertEqual(result.component.calibrated_physical_seed_count,
            prior.calibrated_physical_seed_count)

    def check_test_finite_precision_sum_mean_product_and_predictive_overflow_are_rescued(self):
        for covariance, mean in ((1e-308, 1.), (.5, 1e308), (1.6e308, 1.)):
            with self.subTest(covariance=covariance, mean=mean):
                values, prior = fixture(covariance=covariance)
                prior = replace(prior, mean=(mean,)*19)
                local = replace(prior, mean=(mean,)*19)
                result = update(prior, local, values)
                self.assertEqual(result.state, 'applied')
                np.testing.assert_allclose(np.diag(result.component.covariance), covariance / 2, rtol=2e-14, atol=0)
                np.testing.assert_array_equal(result.component.mean, prior.mean)
                self.assertTrue(np.isfinite(result.component.predictive_interval_95).all())

    def check_test_nonzero_product_underflow_cannot_silently_replace_evidence_with_zero(self):
        values, prior = fixture(covariance=1e308)
        prior = replace(prior, mean=(1e-200,)*19)
        result = update(prior, prior, values)
        self.assertEqual(result.state, 'applied')
        np.testing.assert_array_equal(result.component.mean, prior.mean)
        # A genuine lost nonzero separation must still retain the prior.
        local = replace(prior, mean=(0.,)*19)
        self.assert_unavailable(update(local, prior, values), local, 'numeric_range')

    def check_test_solver_failure_abstains_but_unrelated_programming_error_propagates(self):
        values, prior = fixture(); _, local = fixture()
        with patch.object(np.linalg, 'pinv', side_effect=np.linalg.LinAlgError('solver did not converge')):
            result = update(prior, local, values)
        self.assertEqual(result.state, 'applied')
        self.assertEqual(result.pooling_method, 'coordinate_conditioned_precision')
        with patch.object(np.linalg, 'pinv', side_effect=np.linalg.LinAlgError('legacy solver failure')), \
             patch.object(np.linalg, 'solve', side_effect=np.linalg.LinAlgError('conditioned solver failure')):
            result = update(prior, local, values)
        self.assert_unavailable(result, prior, 'numeric_range')
        with patch.object(np.linalg, 'pinv', side_effect=ValueError('unrelated implementation defect')):
            with self.assertRaisesRegex(ValueError, 'unrelated implementation defect'):
                update(prior, local, values)

    def check_test_legitimate_signed_shape_mean_cancellation_remains_available(self):
        values, prior = fixture(False)
        mean = np.asarray(prior.mean).copy(); mean[15] = 1.
        prior = replace(prior, mean=tuple(mean))
        mean[15] = -1.
        local = replace(prior, mean=tuple(mean))
        result = update(prior, local, values)
        self.assertEqual(result.state, 'applied')
        self.assertEqual(result.component.mean[15], 0.)
        self.assertGreater(result.component.covariance[15][15], 0)

    def check_test_invalid_covariance_has_distinct_reason_in_both_representations(self):
        for physical in (False, True):
            values, prior = fixture(physical); _, local = fixture(physical)
            for symmetric in (False, True):
                covariance = np.eye(19)
                covariance[3, 3] = -1 if symmetric else 1
                if not symmetric: covariance[3, 5] = 1.
                invalid = replace(prior, covariance=tuple(tuple(row) for row in covariance))
                self.assert_unavailable(update(invalid, local, values), invalid, 'invalid_covariance')

    def check_test_mixed_failure_retains_exact_physical_shell_and_existing_fit_reason(self):
        for prior_physical in (False, True):
            values, prior = fixture(prior_physical)
            _, local = fixture(not prior_physical)
            normalized = local if prior_physical else prior
            covariance = np.asarray(normalized.covariance).copy(); covariance[3, 3] = -1
            normalized = replace(normalized,
                covariance=tuple(tuple(row) for row in covariance),
                physical_fit_unavailability_reason='numeric_range', calibrated_physical_seed_count=7.)
            if prior_physical: local = normalized
            else: prior = normalized
            physical = prior if prior_physical else local
            result = update(prior, local, values)
            self.assert_unavailable(result, physical, 'invalid_covariance')
            self.assertEqual(result.source, 'library' if prior_physical else 'local')
            self.assertTrue(result.component.physical_dimensions_available)
            self.assertIsNone(result.component.physical_fit_unavailability_reason)

    def check_test_mixed_existing_ridge_preserves_identical_nonzero_singular_shape_means(self):
        values, prior = fixture(False, covariance=0.)
        _, local = fixture(True, covariance=0.)
        result = update(prior, local, values)
        self.assertEqual(result.state, 'applied')
        indices = model._DIMENSIONLESS_MEASUREMENT_INDICES
        np.testing.assert_allclose(np.asarray(result.component.mean)[indices],
            np.asarray(local.mean)[indices], atol=1e-15)
        self.assertGreater(result.component.mean[3], 1)
        dimensions = model._PHYSICAL_MEASUREMENT_INDICES
        np.testing.assert_array_equal(np.asarray(result.component.mean)[dimensions],
            np.asarray(local.mean)[dimensions])

    def check_test_local_only_fit_is_distinct_from_applied_pooling_and_invalid_shrinkage_rejects(self):
        values, local = fixture()
        result = update(None, local, values)
        self.assertEqual(result.state, 'local_only')
        self.assertEqual(result.source, 'local')
        self.assertIs(result.component, local)
        for invalid in (float('nan'), float('inf'), True, -1):
            with self.subTest(shrinkage=invalid), self.assertRaises(ValueError):
                update(local, local, values, shrinkage=invalid)

    def check_test_actual_assembly_reports_unavailable_pooling_and_preserves_processing_scalar(self):
        values, prior = fixture()
        prior = replace(prior, hierarchy_path=('lupinus_mutabilis',),
            covariance=tuple((0.,)*19 for _ in range(19)))
        summary = model.measure_reviewed_seed_instances(labels_fixture(), (annotation(),), pixels_per_mm=4.)
        bank = SpeciesDimensionsShapeBank((), (prior,))
        result = model.assemble_seed_dimensions_shape_model(local_summary=summary,
            legacy_processing_diameter_px=87, source_mode='species_library_prior_and_current',
            biological_context=BiologicalContext('lupinus_mutabilis'),
            library_bank=bank, minimum_component_seeds=1)
        family = result.family('flat')
        self.assertIs(family.component, prior)
        self.assertEqual(family.source, 'library')
        self.assertEqual(family.update_state, 'unavailable')
        self.assertEqual(family.update_unavailability_reason, 'singular_precision')
        self.assertFalse(result.in_sample_local_update)
        self.assertTrue(any('pooling unavailable' in warning for warning in result.warnings))
        self.assertEqual(result.compatibility_seed_diameter_px, 87)

    def check_test_update_outcome_atomic_states_reject_misleading_reason_combinations(self):
        _, prior = fixture()
        for state, component, source, reason in (('unavailable', None, 'library', 'numeric_range'),
                ('applied', prior, 'library+local', 'numeric_range'), ('unavailable', prior, 'library', None),
                ('unavailable', prior, 'library', 'fit_failed'),
                ('unavailable', prior, 'library+local', 'numeric_range'),
                ('applied', None, 'library+local', None), ('applied', prior, 'library', None),
                ('local_only', None, 'local', None), ('local_only', prior, 'library', None),
                ('not_attempted', prior, 'library+local', None)):
            with self.subTest(state=state, reason=reason), self.assertRaises(ValueError):
                model.ShapeComponentUpdate(component, state, source, reason)

    def test_workflow_lifecycle(self):
        """6 related ShapeUpdateAvailabilityTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_ordinary_same_representation_pooling_preserves_exact_existing_arithmetic)
        self.scenario(self.check_test_same_representation_singular_or_cutoff_direction_cannot_become_zero_mean_certainty)
        self.scenario(self.check_test_precision_weighting_underflow_retains_prior_without_counting_unused_local_fit)
        self.scenario(self.check_test_finite_precision_sum_mean_product_and_predictive_overflow_are_rescued)
        self.scenario(self.check_test_nonzero_product_underflow_cannot_silently_replace_evidence_with_zero)
        self.scenario(self.check_test_mixed_existing_ridge_preserves_identical_nonzero_singular_shape_means)

    def test_validation_and_recovery(self):
        """7 related ShapeUpdateAvailabilityTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_solver_failure_abstains_but_unrelated_programming_error_propagates)
        self.scenario(self.check_test_legitimate_signed_shape_mean_cancellation_remains_available)
        self.scenario(self.check_test_invalid_covariance_has_distinct_reason_in_both_representations)
        self.scenario(self.check_test_mixed_failure_retains_exact_physical_shell_and_existing_fit_reason)
        self.scenario(self.check_test_local_only_fit_is_distinct_from_applied_pooling_and_invalid_shrinkage_rejects)
        self.scenario(self.check_test_actual_assembly_reports_unavailable_pooling_and_preserves_processing_scalar)
        self.scenario(self.check_test_update_outcome_atomic_states_reject_misleading_reason_combinations)
