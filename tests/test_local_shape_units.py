"""Current-image shape fits must keep calibrated and normalized blocks apart."""

from tests.functional_units import functional_units
from dataclasses import replace
import unittest

import cv2
import numpy as np

from seedvision.measurement import geometry as g, shape_model as model
from seedvision.persistence.reference_regions import SeedInstanceAnnotation


def measurements(axes=(24, 14), scale=None):
    values = []
    for angle in (0, 27):
        mask = np.zeros((96, 128), np.uint8)
        cv2.ellipse(mask, (64, 48), axes, angle, 0, 360, 1, -1)
        values.append(g.measure_shape_mask(mask, pixels_per_mm=scale, perturbation_radii=()))
    return values


@functional_units
class LocalShapeUnitTests(unittest.TestCase):
    def check_test_local_pixels_are_dimensionless_and_mm_are_physical(self):
        for scale in (None, 2.5):
            component = model._local_component(measurements(scale=scale), 'flat')
            self.assertEqual(component.physical_dimensions_available, scale is not None)
            self.assertEqual(component.calibrated_physical_seed_count, 2 if scale else 0)
            self.assertEqual(component.dimensionless_physical_seed_count, 2)
            if scale is None:
                self.assertEqual(component.mean[0], 1)

    def check_test_local_fit_refuses_mixed_or_unsupported_units(self):
        pixels = measurements()
        for values in ((pixels[0], g._scale_shape_measurement(pixels[1], 2.5)),
                       (replace(pixels[0], length_unit='cm', area_unit='cm²'),)):
            with self.assertRaisesRegex(ValueError, 'one supported dimensional unit'):
                model._local_component(values, 'flat')

    def check_test_pixel_fit_is_invariant_to_rescaling_dimensional_values(self):
        original = measurements()
        # A raster unit change alters only dimensional values/errors, while
        # the pixel-frame endpoint facts and true shape quantities stay fixed.
        changed = [replace(g._scale_shape_measurement(m, 1e12), length_unit='px', area_unit='px²')
                   for m in original]
        first = model._local_component(original, 'flat')
        second = model._local_component(changed, 'flat')
        np.testing.assert_allclose(second.mean, first.mean, rtol=2e-14)
        np.testing.assert_allclose(second.covariance, first.covariance, rtol=1e-12, atol=1e-15)

    def check_test_mixed_updates_keep_exact_physical_blocks_in_both_directions(self):
        dimensions = model._PHYSICAL_MEASUREMENT_INDICES
        shapes = model._DIMENSIONLESS_MEASUREMENT_INDICES
        for prior_physical in (False, True):
            prior = model._local_component(measurements((28, 10), 2 if prior_physical else None), 'flat')
            current = measurements((20, 15), None if prior_physical else 3)
            local = model._local_component(current, 'flat')
            shell = prior if prior_physical else local
            updated = model._update_component(prior, current, 'flat',
                shrinkage_seed_count=2, maximum_contour_modes=0)
            self.assertTrue(updated.physical_dimensions_available)
            for name in ('mean', 'mean_uncertainty', 'predictive_interval_95'):
                np.testing.assert_array_equal(np.asarray(getattr(updated, name))[dimensions],
                                               np.asarray(getattr(shell, name))[dimensions])
            covariance = np.asarray(updated.covariance)
            np.testing.assert_array_equal(covariance[np.ix_(dimensions, dimensions)],
                np.asarray(shell.covariance)[np.ix_(dimensions, dimensions)])
            np.testing.assert_array_equal(covariance[np.ix_(dimensions, shapes)], 0)
            np.testing.assert_array_equal(covariance[np.ix_(shapes, dimensions)], 0)
            self.assertGreaterEqual(np.linalg.eigvalsh(covariance).min(), -1e-12)
            self.assertEqual(updated.calibrated_physical_seed_count, shell.calibrated_physical_seed_count)
            self.assertNotEqual(updated.mean[3], local.mean[3])
            self.assertNotEqual(updated.mean[3], prior.mean[3])

    def check_test_mixed_shape_update_uses_shrinkage_without_changing_physical_dimensions(self):
        prior = model._local_component(measurements((28, 10)), 'flat')
        current = measurements((20, 15), 3)
        low = model._update_component(prior, current, 'flat', shrinkage_seed_count=0, maximum_contour_modes=0)
        high = model._update_component(prior, current, 'flat', shrinkage_seed_count=50, maximum_contour_modes=0)
        self.assertLess(abs(high.mean[3] - prior.mean[3]), abs(low.mean[3] - prior.mean[3]))
        np.testing.assert_array_equal(np.asarray(high.mean)[(0, 1, 2, 4),],
                                       np.asarray(low.mean)[(0, 1, 2, 4),])

    def check_test_singular_shape_covariance_retains_nonzero_identical_means(self):
        px = model._local_component(measurements(), 'flat')
        mm = model._local_component(measurements(scale=2), 'flat')
        zeros = tuple((0.,) * 19 for _ in range(19))
        px = replace(px, covariance=zeros)
        mm = replace(mm, covariance=zeros)
        result = model._update_mixed_dimension_components(px, mm, 'flat', 1)
        indices = model._DIMENSIONLESS_MEASUREMENT_INDICES
        np.testing.assert_allclose(np.asarray(result.mean)[indices], np.asarray(mm.mean)[indices], atol=1e-15)
        self.assertGreater(result.mean[3], 1)
        self.assertTrue(np.isfinite(result.covariance).all())

    def check_test_matching_dimensionless_updates_do_not_invent_calibrated_support(self):
        current = measurements()
        prior = model._local_component(current, 'flat')
        singular = np.linalg.svd(prior.covariance, compute_uv=False)
        self.assertLess(singular[-1] / singular[0], 1e-17)
        outcome = model._update_component_result(prior, current, 'flat', shrinkage_seed_count=2, maximum_contour_modes=0)
        # Coordinate conditioning now retains all these anisotropic directions.
        self.assertEqual(outcome.state, 'applied')
        self.assertIsNone(outcome.unavailability_reason)
        self.assertEqual(outcome.pooling_method, 'coordinate_conditioned_precision')
        result = outcome.component
        self.assertFalse(result.physical_dimensions_available)
        self.assertEqual(result.calibrated_physical_seed_count, 0)
        self.assertGreater(result.mean[3], 1)
        np.testing.assert_array_equal(result.mean, prior.mean)
        np.testing.assert_allclose(result.covariance, np.asarray(prior.covariance) / (1 + 2/3),
                                   rtol=2e-12, atol=1e-30)

    def check_test_mixed_shape_pooling_is_independent_of_physical_unit_scale(self):
        current = measurements((20, 15))
        first = model._local_component(measurements((28, 10), 2), 'flat')
        second = model._local_component(measurements((28, 10), 20), 'flat')
        updates = [model._update_component(prior, current, 'flat',
            shrinkage_seed_count=2, maximum_contour_modes=0) for prior in (first, second)]
        indices = model._DIMENSIONLESS_MEASUREMENT_INDICES
        np.testing.assert_array_equal(np.asarray(updates[0].mean)[indices], np.asarray(updates[1].mean)[indices])
        np.testing.assert_array_equal(np.asarray(updates[0].covariance)[np.ix_(indices, indices)],
                                       np.asarray(updates[1].covariance)[np.ix_(indices, indices)])

    def check_test_invalid_mixed_shape_covariance_is_rejected_explicitly(self):
        prior = model._local_component(measurements(), 'flat')
        local = model._local_component(measurements(scale=2), 'flat')
        for symmetric in (True, False):
            covariance = np.asarray(prior.covariance).copy()
            covariance[3, 3] = -1 if symmetric else covariance[3, 3]
            if not symmetric:
                covariance[3, 5] = 1
            invalid = replace(prior, covariance=tuple(tuple(r) for r in covariance))
            with self.assertRaisesRegex(ValueError, 'symmetric and positive semidefinite'):
                model._update_mixed_dimension_components(invalid, local, 'flat', 1)

    def check_test_current_only_model_preserves_processing_diameter_scalar(self):
        labels = np.zeros((96, 128), np.uint16)
        cv2.ellipse(labels, (64, 48), (24, 14), 0, 0, 360, 1, -1)
        annotation = SeedInstanceAnnotation(1, shape_reviewed=True, outline_visibility='complete', pose='flat')
        for scale in (None, 2):
            summary = model.measure_reviewed_seed_instances(labels, (annotation,), pixels_per_mm=scale)
            result = model.assemble_seed_dimensions_shape_model(local_summary=summary,
                source_mode='current_image_only', minimum_component_seeds=1,
                legacy_processing_diameter_px=87)
            self.assertEqual(result.compatibility_seed_diameter_px, 87)
            self.assertEqual(result.family('flat').component.physical_dimensions_available, scale is not None)

    def test_workflow_lifecycle(self):
        """9 related LocalShapeUnitTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_local_pixels_are_dimensionless_and_mm_are_physical)
        self.scenario(self.check_test_local_fit_refuses_mixed_or_unsupported_units)
        self.scenario(self.check_test_pixel_fit_is_invariant_to_rescaling_dimensional_values)
        self.scenario(self.check_test_mixed_updates_keep_exact_physical_blocks_in_both_directions)
        self.scenario(self.check_test_mixed_shape_update_uses_shrinkage_without_changing_physical_dimensions)
        self.scenario(self.check_test_singular_shape_covariance_retains_nonzero_identical_means)
        self.scenario(self.check_test_matching_dimensionless_updates_do_not_invent_calibrated_support)
        self.scenario(self.check_test_mixed_shape_pooling_is_independent_of_physical_unit_scale)
        self.scenario(self.check_test_current_only_model_preserves_processing_diameter_scalar)

    def test_validation_and_recovery(self):
        """1 related LocalShapeUnitTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_invalid_mixed_shape_covariance_is_rejected_explicitly)
