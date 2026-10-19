"""Finite calibrated evidence may exceed the physical fitter's numeric domain."""

from tests.functional_units import functional_units
from dataclasses import replace
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from seedvision.measurement import geometry as g, shape_model as model
from seedvision.reference_library.contracts import (
    BiologicalContext, LibrarySourceRecord, SpeciesLibraryArtifact, SpeciesLibraryPin,
)
from seedvision.reference_library import persistence as storage


def measurements(scale=10):
    values = []
    for axes in ((24, 14), (29, 16)):
        mask = np.zeros((96, 128), np.uint8)
        cv2.ellipse(mask, (64, 48), axes, 17, 0, 360, 1, -1)
        values.append(g.measure_shape_mask(mask, pixels_per_mm=scale))
    return values


def observations(values, *, hierarchy=('soybean',)):
    return tuple(model.ShapeObservation(i, i + 1, f'seed:{i}', hierarchy, 'flat',
        'complete', True, m.vector(), m.measurement_uncertainty, m.contour_signature)
        for i, m in enumerate(values))


def bank(values):
    return model.fit_species_dimensions_shape_bank(observations(values), maximum_contour_modes=0)


def manifest(count):
    return storage.create_manifest(library_id='numeric-fit', version='1', species_id='soybean',
        species_display_name='Soybean', trait_vocabulary_sha256='a' * 64,
        seed_fiddle_version='test', sources=tuple(LibrarySourceRecord(
            f'{i + 1:064x}', f'{i + 101:064x}', f'source-{i}', (96, 128))
            for i in range(count)), products=(), descriptor_schemas={},
        extraction_settings_sha256='b' * 64, aggregation_settings_sha256='c' * 64,
        corrected_colour_space_version='opencv-lab-u8-v1')


@functional_units
class PhysicalShapeFitAvailabilityTests(unittest.TestCase):
    def assert_fallback(self, component, count=2):
        self.assertFalse(component.physical_dimensions_available)
        self.assertEqual(component.physical_fit_unavailability_reason, 'numeric_range')
        self.assertEqual(component.calibrated_physical_seed_count, count)
        self.assertEqual(component.dimensionless_physical_seed_count, count)
        self.assertEqual(component.effective_physical_seed_count, count)
        self.assertEqual(component.mean[0], 1)
        for name in ('mean', 'covariance', 'mean_uncertainty', 'predictive_interval_95'):
            self.assertTrue(np.isfinite(getattr(component, name)).all(), name)

    def check_test_real_calibrated_area_covariance_overflow_retains_all_dimensionless_groups(self):
        values = measurements(1e-79)
        self.assertTrue(all(np.isfinite(m.vector()).all() for m in values))
        fitted = bank(values)
        self.assertEqual(fitted.observations, observations(values))
        self.assert_fallback(fitted.components[0])
        self.assertEqual(fitted.components[0].source_count, 2)
        reference = bank(measurements()).components[0]
        indices = model._DIMENSIONLESS_MEASUREMENT_INDICES
        np.testing.assert_allclose(np.asarray(fitted.components[0].mean)[indices],
            np.asarray(reference.mean)[indices], rtol=2e-13, atol=2e-13)

    def check_test_positive_physical_variance_underflow_retains_normalized_error(self):
        values = measurements(1e82)
        self.assertTrue(all(m.measurement_uncertainty[4] > 0 for m in values))
        fitted = bank(values)
        self.assert_fallback(fitted.components[0])
        vectors, _, errors, _ = model._physical_seed_groups(observations(values), normalize_per_view=True)
        self.assertTrue(np.all(errors[:, 4] > 0))
        self.assertTrue(np.all(np.square(errors[:, 4]) > 0))
        self.assertGreater(fitted.components[0].mean_uncertainty[4], 0)

    def check_test_physical_covariance_underflow_is_detected_without_measurement_errors(self):
        values = [replace(m, measurement_uncertainty=(0.,) * 19) for m in measurements(1e84)]
        self.assert_fallback(bank(values).components[0])

    def check_test_parent_predictive_sum_overflow_has_dedicated_numeric_failure(self):
        measurement = measurements()[0]
        errors = list(measurement.measurement_uncertainty); errors[4] = 1e154
        vectors = np.asarray([measurement.vector()])
        uncertainties = np.asarray([errors])
        parent = model._local_component((measurement,), 'flat')
        covariance = np.asarray(parent.covariance).copy(); covariance[4, 4] = 1e308
        parent = replace(parent, covariance=tuple(tuple(row) for row in covariance))
        with self.assertRaises(model._PhysicalFitNumericUnavailable):
            model._fit_component(('soybean',), 'flat', vectors, np.empty((0, 0)),
                uncertainties, 2, parent=parent, shrinkage_seed_count=9, maximum_contour_modes=0)

    def check_test_repeated_views_preserve_biological_source_and_calibrated_counts_on_fallback(self):
        records = observations(measurements(1e-79))
        repeated = replace(records[0], source_index=2, seed_id=3)
        fitted = model.fit_species_dimensions_shape_bank((*records, repeated), maximum_contour_modes=0)
        self.assert_fallback(fitted.components[0])
        self.assertEqual(fitted.components[0].source_count, 3)
        self.assertEqual(len(fitted.observations), 3)

    def check_test_local_mm_numeric_fallback_preserves_calibrated_support(self):
        for scale in (1e-79, 1e82):
            self.assert_fallback(model._local_component(measurements(scale), 'flat'))

    def check_test_physical_grouping_rescues_mean_and_rms_without_dropping_sources(self):
        for error in (1e200, 1e-200):
            item = observations(measurements())[0]
            vector = list(item.measurement); vector[4] = 1e308
            errors = list(item.measurement_uncertainty); errors[4] = error
            item = replace(item, measurement=tuple(vector), measurement_uncertainty=tuple(errors))
            second = replace(item, source_index=1, seed_id=2)
            grouped, _, uncertainty, sources = model._physical_seed_groups((item, second), calibrated_only=True)
            self.assertEqual(grouped[0, 4], 1e308)
            self.assertEqual(uncertainty[0, 4], error)
            self.assertEqual(sources, 2)

    def check_test_scaled_rms_cannot_silently_drop_smallest_positive_uncertainty(self):
        item = observations(measurements(1e100))[0]
        records = []
        for source in range(5):
            errors = [0.] * 19
            if source == 0:
                errors[4] = np.nextafter(0., 1.)
            records.append(replace(item, source_index=source, seed_id=source + 1,
                                   measurement_uncertainty=tuple(errors)))
        with self.assertRaises(model._PhysicalFitNumericUnavailable):
            model._physical_seed_groups(records, calibrated_only=True)
        fitted = model.fit_species_dimensions_shape_bank(records,
            minimum_component_seeds=1, maximum_contour_modes=0)
        self.assert_fallback(fitted.components[0], count=1)
        self.assertEqual(fitted.components[0].source_count, 5)
        self.assertEqual(fitted.observations, tuple(records))
        # An already normalized shape error this small is outside the rescue
        # domain. Explicit rejection must replace the tempting false zero.
        normalized = []
        for record in records:
            vector = list(record.measurement); vector[0] = 1
            normalized.append(replace(record, measurement=tuple(vector)))
        with self.assertRaisesRegex(ValueError, 'Normalized seed groups are not representable'):
            model._physical_seed_groups(normalized, normalize_per_view=True)

    def check_test_ordinary_physical_fit_remains_available_with_compatible_parent(self):
        values = measurements()
        root = bank(values).components[0]
        self.assertTrue(root.physical_dimensions_available)
        self.assertIsNone(root.physical_fit_unavailability_reason)
        grouped = model._physical_seed_groups(observations(values), calibrated_only=True)
        local = model._fit_component(('soybean',), 'flat', *grouped,
            parent=None, shrinkage_seed_count=5, maximum_contour_modes=0)
        same = model._fit_component(('soybean',), 'flat', *grouped,
            parent=local, shrinkage_seed_count=5, maximum_contour_modes=0)
        np.testing.assert_allclose(same.mean, local.mean, rtol=2e-15, atol=1e-15)
        np.testing.assert_allclose(same.covariance, local.covariance, rtol=2e-15, atol=1e-15)

    def check_test_incompatible_parent_cannot_mix_physical_and_normalized_blocks(self):
        values = measurements()
        physical = model._local_component(values, 'flat')
        grouped = model._physical_seed_groups(observations(values), normalize_per_view=True)
        expected = model._fit_component(('soybean',), 'flat', *grouped, parent=None,
            shrinkage_seed_count=5, maximum_contour_modes=0, physical_dimensions_available=False)
        actual = model._fit_component(('soybean',), 'flat', *grouped, parent=physical,
            shrinkage_seed_count=5, maximum_contour_modes=0, physical_dimensions_available=False)
        self.assertEqual(actual, expected)

    def check_test_same_representation_update_retains_reason_and_calibrated_evidence(self):
        values = measurements(1e-79)
        prior = model._local_component(values, 'flat')
        singular = np.linalg.svd(prior.covariance, compute_uv=False)
        self.assertLess(singular[-1] / singular[0], 1e-17)
        outcome = model._update_component_result(prior, values, 'flat', shrinkage_seed_count=1, maximum_contour_modes=0)
        self.assertEqual(outcome.state, 'applied')
        self.assertIsNone(outcome.unavailability_reason)
        self.assertEqual(outcome.pooling_method, 'coordinate_conditioned_precision')
        result = outcome.component
        self.assertFalse(result.physical_dimensions_available)
        self.assertEqual(result.physical_fit_unavailability_reason, 'numeric_range')
        self.assertEqual(result.calibrated_physical_seed_count, 3)

    def check_test_mixed_updates_clear_reason_only_when_successful_physical_shell_exists(self):
        for prior_unavailable in (False, True):
            prior = model._local_component(measurements(1e-79 if prior_unavailable else 10), 'flat')
            current = measurements(10 if prior_unavailable else 1e-79)
            result = model._update_component(prior, current, 'flat', shrinkage_seed_count=1, maximum_contour_modes=0)
            self.assertTrue(result.physical_dimensions_available)
            self.assertIsNone(result.physical_fit_unavailability_reason)
            self.assertEqual(result.calibrated_physical_seed_count, 4 if prior_unavailable else 3)
            physical = model._local_component(current, 'flat') if prior_unavailable else prior
            indices = model._PHYSICAL_MEASUREMENT_INDICES
            np.testing.assert_array_equal(np.asarray(result.mean)[indices], np.asarray(physical.mean)[indices])

    def check_test_unrelated_fitting_error_and_unrepresentable_fallback_propagate(self):
        with patch.object(model, '_fit_component', side_effect=ValueError('unrelated fitter defect')):
            with self.assertRaisesRegex(ValueError, 'unrelated fitter defect'):
                bank(measurements())
        item = observations(measurements())[0]
        vector = list(item.measurement); vector[0] = 1e300; vector[1] = 1e-300
        with self.assertRaisesRegex(ValueError, 'not representable'):
            model.fit_species_dimensions_shape_bank((replace(item, measurement=tuple(vector)),))

    def check_test_component_payload_roundtrip_legacy_and_invalid_reasons(self):
        component = bank(measurements(1e-79)).components[0]
        payload = storage._shape_component_payload(component)
        self.assertEqual(storage._shape_component_from_payload(json.loads(json.dumps(payload, allow_nan=False))), component)
        legacy = dict(payload); legacy.pop('physical_fit_unavailability_reason')
        self.assertIsNone(storage._shape_component_from_payload(legacy).physical_fit_unavailability_reason)
        for changes in ({'physical_dimensions_available': True}, {'physical_fit_unavailability_reason': 'invented'}):
            with self.assertRaisesRegex(ValueError, 'unavailability reason'):
                storage._shape_component_from_payload(dict(payload, **changes))

    def check_test_real_library_publish_load_preserves_reason_counts_and_float64_evidence(self):
        fitted = bank(measurements(1e-79))
        with tempfile.TemporaryDirectory() as temporary:
            store = storage.SpeciesLibraryStore(Path(temporary) / 'libraries')
            published = store.publish(SpeciesLibraryArtifact(manifest(2), dimensions_shape=fitted))
            m = published.manifest
            loaded = store.load(SpeciesLibraryPin(m.library_id, m.version, m.species_id, m.content_sha256))
            self.assertEqual(loaded.dimensions_shape.components, fitted.components)
            for actual, original in zip(loaded.dimensions_shape.observations, fitted.observations):
                self.assertEqual(actual.measurement, original.measurement)
                self.assertEqual(actual.measurement_uncertainty, original.measurement_uncertainty)
                self.assertEqual(replace(actual, contour_signature=()),
                                 replace(original, contour_signature=()))
                np.testing.assert_array_equal(actual.contour_signature,
                    np.asarray(original.contour_signature, np.float32).astype(np.float64))
            arrays, _ = storage._artifact_arrays(loaded)
            self.assertEqual(arrays['dimensions_shape.measurements'].dtype, np.float64)
            self.assertEqual(loaded.dimensions_shape.components[0].source_count, 2)

    def check_test_legacy_library_digest_is_verified_before_absent_reason_defaults(self):
        fitted = bank(measurements())
        with tempfile.TemporaryDirectory() as temporary:
            store = storage.SpeciesLibraryStore(Path(temporary) / 'libraries')
            published = store.publish(SpeciesLibraryArtifact(manifest(2), dimensions_shape=fitted))
            m = published.manifest
            directory = store.version_path(m.species_id, m.library_id, m.version)
            path = directory / storage.MANIFEST_NAME
            payload = json.loads(path.read_text(encoding='utf-8'))
            for item in payload['bank_metadata']['banks']['dimensions_shape']['components']:
                item.pop('physical_fit_unavailability_reason')
            arrays, _ = storage._artifact_arrays(published)
            payload['content_sha256'] = storage._content_digest(payload, arrays)
            path.write_text(json.dumps(payload, allow_nan=False), encoding='utf-8')
            stored_bytes = path.read_bytes()
            pin = SpeciesLibraryPin(m.library_id, m.version, m.species_id, payload['content_sha256'])
            loaded = store.load(pin)
            self.assertIsNone(loaded.dimensions_shape.components[0].physical_fit_unavailability_reason)
            self.assertEqual(path.read_bytes(), stored_bytes)
            payload['bank_metadata']['banks']['dimensions_shape']['components'][0]['mean'][0] += 1
            path.write_text(json.dumps(payload, allow_nan=False), encoding='utf-8')
            with self.assertRaisesRegex(storage.InvalidSpeciesLibrary, 'content hash'):
                store.load(pin)

    def check_test_unknown_component_fields_and_nonfinite_predictive_intervals_reject(self):
        component = bank(measurements()).components[0]
        payload = storage._shape_component_payload(component)
        with self.assertRaisesRegex(storage.InvalidSpeciesLibrary, 'invalid schema'):
            storage._shape_component_from_payload(dict(payload, invented_field=1))
        intervals = list(component.predictive_interval_95)
        intervals[0] = (0., np.inf)
        with self.assertRaisesRegex(ValueError, 'predictive intervals must be finite'):
            replace(component, predictive_interval_95=tuple(intervals))

    def test_workflow_lifecycle(self):
        """14 related PhysicalShapeFitAvailabilityTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_real_calibrated_area_covariance_overflow_retains_all_dimensionless_groups)
        self.scenario(self.check_test_positive_physical_variance_underflow_retains_normalized_error)
        self.scenario(self.check_test_physical_covariance_underflow_is_detected_without_measurement_errors)
        self.scenario(self.check_test_repeated_views_preserve_biological_source_and_calibrated_counts_on_fallback)
        self.scenario(self.check_test_local_mm_numeric_fallback_preserves_calibrated_support)
        self.scenario(self.check_test_physical_grouping_rescues_mean_and_rms_without_dropping_sources)
        self.scenario(self.check_test_scaled_rms_cannot_silently_drop_smallest_positive_uncertainty)
        self.scenario(self.check_test_ordinary_physical_fit_remains_available_with_compatible_parent)
        self.scenario(self.check_test_incompatible_parent_cannot_mix_physical_and_normalized_blocks)
        self.scenario(self.check_test_same_representation_update_retains_reason_and_calibrated_evidence)
        self.scenario(self.check_test_mixed_updates_clear_reason_only_when_successful_physical_shell_exists)
        self.scenario(self.check_test_unrelated_fitting_error_and_unrepresentable_fallback_propagate)
        self.scenario(self.check_test_real_library_publish_load_preserves_reason_counts_and_float64_evidence)
        self.scenario(self.check_test_legacy_library_digest_is_verified_before_absent_reason_defaults)

    def test_validation_and_recovery(self):
        """3 related PhysicalShapeFitAvailabilityTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_parent_predictive_sum_overflow_has_dedicated_numeric_failure)
        self.scenario(self.check_test_component_payload_roundtrip_legacy_and_invalid_reasons)
        self.scenario(self.check_test_unknown_component_fields_and_nonfinite_predictive_intervals_reject)
