"""Actual review snapshots, producer cache boundaries and export identity."""
from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from seedvision.export.calibration_facts import calibration_facts
from seedvision.export.results import export_report, result_report
from seedvision.measurement.lineage import (
    facts_sha256, prepare_reviewed_inputs, reviewed_measurement_context,
)
from seedvision.measurement.shape_model import measure_reviewed_seed_instances
from seedvision.persistence.reference_regions import ReferenceRegionBundle, SeedInstanceAnnotation
from tests.test_result_prediction_revision import _analysis


def annotation(seed_id=1, **kwargs):
    return SeedInstanceAnnotation(seed_id, shape_reviewed=True,
        outline_visibility='complete', pose='flat', **kwargs)


def labels_fixture(shape=(64, 96)):
    labels = np.zeros(shape, np.uint16)
    cv2.ellipse(labels, (shape[1]//2, shape[0]//2), (12, 8), 17, 0, 360, 1, -1)
    return labels


def analysis_fixture(*, scale=4., source=None, proven=True):
    current = _analysis(source)
    current.calibration.corrected_bgr = np.zeros((64, 96, 3), np.uint8)
    current.calibration.pixels_per_mm = scale
    labels, annotations = labels_fixture(), (annotation(physical_seed_id='seed-A'),)
    if proven:
        from seedvision.annotation.coordinates import reproject_bundle
        transported = reproject_bundle(ReferenceRegionBundle(labels.shape,
            annotated_seeds=labels, seed_annotations=annotations, source_to_corrected=np.eye(3)),
            current.calibration.affine_matrix, labels.shape)
        labels, annotations = transported.annotated_seeds, transported.seed_annotations
    context = reviewed_measurement_context(
        sha256(Path(source).read_bytes()).hexdigest() if source else None,
        current.calibration.corrected_bgr, current.calibration.affine_matrix,
        np.eye(3) if proven else None, calibration_facts=calibration_facts(current.calibration),
        transport_applied=proven)
    current.seed_measurement_summary = measure_reviewed_seed_instances(
        labels, annotations, pixels_per_mm=scale,
        calibration_relative_uncertainty=.02, input_context=context)
    return current


class ReviewedMeasurementLineageTests(unittest.TestCase):
    def test_prepared_inputs_detach_normalize_and_hash_consumed_last_wins_defaults(self):
        labels = labels_fixture(); labels[5:8, 5:8] = 2
        conditions = ['split']; point = [45., 32.]
        last = annotation(conditions=conditions, hilum_point=point)
        unused = annotation(999)
        prepared = prepare_reviewed_inputs(labels, (annotation(), unused, last))
        self.assertEqual([item.seed_id for item in prepared.annotations], [1, 2])
        self.assertEqual(prepared.annotations[1], SeedInstanceAnnotation(2))
        self.assertEqual(prepared.lineage.mask_sha256, sha256(labels.tobytes()).hexdigest())
        self.assertEqual(prepared.lineage.effective_annotations_sha256,
            facts_sha256([asdict(item) for item in prepared.annotations]))
        same = prepare_reviewed_inputs(labels.astype(np.int32), (last, unused))
        self.assertEqual(prepared.cache_key, same.cache_key)
        labels[:] = 0; conditions.append('stained'); point[0] = 1
        self.assertTrue(np.any(prepared.labels))
        self.assertFalse(prepared.labels.flags.writeable)
        self.assertEqual(prepared.annotations[0].conditions, ('split',))
        self.assertEqual(prepared.annotations[0].hilum_point, (45., 32.))

    def test_direct_measurement_records_mask_facts_without_inventing_image_frame(self):
        summary = measure_reviewed_seed_instances(labels_fixture(), (annotation(),))
        self.assertEqual(summary.lineage.source_image_frame, 'unknown')
        self.assertEqual(summary.lineage.transform_status, 'not_recorded')
        self.assertIsNone(summary.lineage.corrected_image_sha256)
        self.assertEqual(summary.observations[0].pixel_measurement.provenance.source_image_frame, 'unknown')
        for bad in (np.zeros((3, 4), float), -np.ones((3, 4), np.int16), np.full((3, 4), 65536)):
            with self.subTest(labels=bad.dtype), self.assertRaises(ValueError):
                prepare_reviewed_inputs(bad, ())

    def test_malformed_annotation_ids_cannot_silently_bind_to_another_label(self):
        labels = labels_fixture()
        for bad in (1.5, 1., True, np.bool_(True), '1', 0, -1, 65536):
            with self.subTest(seed_id=bad), self.assertRaises(ValueError):
                measure_reviewed_seed_instances(labels, (annotation(bad),))
        for valid in (1, np.int64(1)):
            summary = measure_reviewed_seed_instances(labels, (annotation(valid),))
            self.assertEqual(summary.observations[0].seed_id, 1)
            self.assertIs(type(summary.observations[0].annotation.seed_id), int)

    def test_declared_transport_preserves_cutoff_decision_and_full_canvas_mapping(self):
        from seedvision.annotation.coordinates import reproject_bundle
        labels = labels_fixture(); current = np.eye(3); current[0, 2] = -45
        transported = reproject_bundle(ReferenceRegionBundle(labels.shape,
            annotated_seeds=labels, seed_annotations=(annotation(),), source_to_corrected=np.eye(3)),
            current, labels.shape)
        context = reviewed_measurement_context(None, np.zeros((*labels.shape, 3), np.uint8),
            current, np.eye(3), transport_applied=True)
        summary = measure_reviewed_seed_instances(transported.annotated_seeds,
            transported.seed_annotations, input_context=context)
        self.assertEqual(summary.lineage.source_image_frame, 'full_corrected')
        self.assertEqual(summary.lineage.transform_status, 'proven_reprojection')
        np.testing.assert_array_equal(summary.lineage.applied_label_mapping, current)
        item = summary.observations[0]
        self.assertEqual(item.annotation.outline_visibility, 'image_cutoff')
        self.assertFalse(item.eligible)
        self.assertIsNone(item.measurement)
        self.assertEqual(summary.lineage.mask_shape, labels.shape)
        with self.assertRaises(ValueError):
            measure_reviewed_seed_instances(labels[:10], (), input_context=context)

    def test_declared_transform_without_applied_transport_does_not_prove_alignment(self):
        labels = labels_fixture()
        context = reviewed_measurement_context(None, np.zeros((*labels.shape, 3), np.uint8),
            np.eye(3), np.eye(3))
        summary = measure_reviewed_seed_instances(labels, (annotation(),), input_context=context)
        self.assertEqual(summary.lineage.source_image_frame, 'unknown')
        self.assertIsNone(summary.lineage.applied_label_mapping)
        self.assertEqual(summary.lineage.reference_source_identity_state, 'not_recorded')
        with self.assertRaises(ValueError):
            reviewed_measurement_context(None, np.zeros((*labels.shape, 3), np.uint8),
                np.eye(3), transport_applied=True)

    def test_producer_recomputes_changed_review_facts_without_unrelated_upstream_nodes(self):
        from seedvision.calibration import image as c
        from seedvision.calibration.geometry import DishDetectionSettings
        from seedvision.segmentation.baseline import analyze_image, PipelineAnalysisCache
        from tests.test_neutral_calibration import fixture, ruler
        image, card = fixture(); labels = labels_fixture(image.shape[:2])
        cache = PipelineAnalysisCache()
        options = dict(node_cache=cache, enabled_nodes=set(),
            calibration_settings=c.CalibrationSettings(apply_perspective_correction=False),
            dish_settings=DishDetectionSettings(expected_outer_diameter_mm=0),
            seed_instance_annotations=labels, seed_instance_traits=(annotation(),))
        with patch.object(c, 'detect_colour_card', return_value=card), patch.object(c, 'detect_ruler', side_effect=ruler):
            first = analyze_image(image, **options)
            same = analyze_image(image, **options)
            self.assertIs(first.seed_measurement_summary, same.seed_measurement_summary)
            labels[32, 59] = 0
            changed = analyze_image(image, **options)
            self.assertIsNot(changed.seed_measurement_summary, first.seed_measurement_summary)
            self.assertIs(changed.calibration.gpu_corrected_bgr, first.calibration.gpu_corrected_bgr)
            self.assertIn('seed_scale_estimation', cache.last_computed_nodes)
            self.assertIn('layout_detection', cache.last_reused_nodes)
            self.assertIn('deskew_colour', cache.last_reused_nodes)
            options['seed_instance_traits'] = (replace(annotation(), shape_reviewed=False),)
            changed_annotation = analyze_image(image, **options)
            self.assertFalse(changed_annotation.seed_measurement_summary.observations[0].eligible)
            options['reference_transform'] = np.eye(3)
            proven = analyze_image(image, **options)
            self.assertIsNot(proven.seed_measurement_summary, changed_annotation.seed_measurement_summary)
            self.assertEqual(proven.seed_measurement_summary.lineage.source_image_frame, 'full_corrected')
            self.assertEqual(proven.seed_measurement_summary.lineage.mask_shape, image.shape[:2])
            self.assertEqual(proven.seed_measurement_summary.lineage.calibration_facts_sha256,
                facts_sha256(calibration_facts(proven.calibration)))
            result_report(proven)


class ReviewedMeasurementExportTests(unittest.TestCase):
    def test_actual_bundles_units_definitions_and_lineage_export_without_physical_prediction_claims(self):
        current = analysis_fixture(); report = result_report(current)
        reviewed = report['reviewed_measurements']; row = reviewed['observations'][0]
        self.assertEqual(report['schema_version'], 3)
        self.assertEqual(report['identity']['revision_contract'], 'seedfiddle.result-revision-v7')
        self.assertEqual(reviewed['schema_version'], 1)
        self.assertEqual(row['pixel_bundle']['length_unit'], 'px')
        self.assertEqual(row['scaled_bundle']['length_unit'], 'mm')
        self.assertAlmostEqual(row['pixel_bundle']['maximum_span']/4, row['scaled_bundle']['maximum_span'])
        self.assertEqual(row['scaled_bundle']['provenance']['pixels_per_mm'], 4)
        self.assertEqual(row['scaled_bundle']['described_values'][0]['source_image_frame'], 'corrected_image')
        self.assertEqual(row['scaled_bundle']['described_values'][19]['unit'], 'px')
        self.assertEqual(row['annotation']['physical_seed_id'], 'seed-A')
        self.assertEqual(reviewed, report['identity']['reviewed_measurements'])
        self.assertFalse(reviewed['physical_measurements_eligible'])
        self.assertFalse(report['scientifically_validated'])
        self.assertIsNone(report['instances'][0]['physical_length_mm'])
        self.assertIsNone(report['instances'][0]['physical_area_mm2'])
        json.dumps(report, allow_nan=False)

    def test_every_exported_review_fact_stales_approval_with_unchanged_prediction_labels(self):
        mutations = (
            lambda s: replace(s, boundary_turn_degrees=(1., 2.)),
            lambda s: replace(s, calibration_reason='Updated conversion reason.'),
            lambda s: replace(s, observations=(replace(s.observations[0], exclusion_reason='updated'),)),
            lambda s: replace(s, observations=(replace(s.observations[0], length_eligible=False),)),
            lambda s: replace(s, observations=(replace(s.observations[0],
                pixel_measurement=replace(s.observations[0].pixel_measurement, maximum_span=30.)),)),
            lambda s: replace(s, observations=(replace(s.observations[0],
                calibrated_measurement=replace(s.observations[0].calibrated_measurement,
                    measurement_uncertainty=(.4,)*19)),)),
            lambda s: replace(s, lineage=replace(s.lineage, mask_sha256='a'*64)),
        )
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                current = analysis_fixture(); before = result_report(current)
                review = {'result_revision': before['result_revision'], 'decisions': {'1': 'accepted'}}
                self.assertEqual(result_report(current, review=review)['accepted_count'], 1)
                current.seed_measurement_summary = mutate(current.seed_measurement_summary)
                after = result_report(current, review=review)
                self.assertNotEqual(before['result_revision'], after['result_revision'])
                self.assertEqual(after['accepted_count'], 0)

    def test_stored_known_source_image_transform_calibration_or_annotation_mismatch_rejects(self):
        mutations = (
            lambda a: a.calibration.corrected_bgr.__setitem__((0, 0, 0), 1),
            lambda a: setattr(a.calibration, 'corrected_bgr', np.zeros((65, 96, 3), np.uint8)),
            lambda a: a.calibration.affine_matrix.__setitem__((0, 2), 1.),
            lambda a: setattr(a.calibration, 'pixels_per_mm', 5.),
            lambda a: setattr(a, 'seed_measurement_summary', replace(a.seed_measurement_summary,
                observations=(replace(a.seed_measurement_summary.observations[0],
                    annotation=annotation(physical_seed_id='different')),))),
        )
        for mutate in mutations:
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                current = analysis_fixture(); mutate(current); result_report(current)
        with TemporaryDirectory() as directory:
            source = Path(directory)/'image.png'; source.write_bytes(b'original')
            current = analysis_fixture(source=source)
            source.write_bytes(b'changed')
            with self.assertRaises(ValueError): result_report(current)

    def test_numeric_scaled_failure_retains_pixel_bundle_and_explicit_reason(self):
        current = analysis_fixture(scale=1e-200); report = result_report(current)
        row = report['reviewed_measurements']['observations'][0]
        self.assertIsNotNone(row['pixel_bundle'])
        self.assertIsNone(row['scaled_bundle'])
        self.assertEqual(row['calibration_state'], 'unavailable')
        self.assertTrue(row['calibration_reason'])
        self.assertTrue(row['shape_eligible'])
        json.dumps(report, allow_nan=False)

    def test_legacy_absence_unknown_frame_and_unknown_descriptions_remain_explicit(self):
        current = _analysis(); report = result_report(current)
        self.assertNotIn('reviewed_measurements', report)
        self.assertIsNone(report['identity']['reviewed_measurements'])
        current = analysis_fixture(proven=False)
        unknown = result_report(current)['reviewed_measurements']['lineage']
        self.assertEqual(unknown['source_image_frame'], 'unknown')
        self.assertEqual(unknown['transform_status'], 'legacy_unknown')
        summary = current.seed_measurement_summary; item = summary.observations[0]
        legacy = replace(item.pixel_measurement, provenance=None)
        current.seed_measurement_summary = replace(summary, lineage=None,
            observations=(replace(item, measurement=legacy, pixel_measurement=legacy,
                calibrated_measurement=None),), length_unit='px')
        reviewed = result_report(current)['reviewed_measurements']
        self.assertIsNone(reviewed['lineage'])
        self.assertEqual(reviewed['lineage_state'], 'legacy_not_recorded')
        self.assertEqual(reviewed['observations'][0]['pixel_bundle']['described_values'][0]['definition'], 'unknown')

    def test_export_roundtrip_is_detached_and_never_reads_current_editor_annotations(self):
        current = analysis_fixture(); report = result_report(current)
        stored = json.dumps(report, sort_keys=True)
        current.seed_instance_traits = (annotation(physical_seed_id='new-editor-state'),)
        self.assertEqual(result_report(current)['result_revision'], report['result_revision'])
        with TemporaryDirectory() as directory:
            output = export_report(directory, current, report)
            saved = json.loads((output/'results.json').read_text(encoding='utf-8'))
        self.assertEqual(saved['reviewed_measurements'], report['reviewed_measurements'])
        current.seed_measurement_summary = replace(current.seed_measurement_summary, lineage=None)
        self.assertEqual(json.dumps(report, sort_keys=True), stored)

    def test_nonfinite_measurement_payload_cannot_be_reviewed_or_exported(self):
        current = analysis_fixture(); summary = current.seed_measurement_summary
        item = summary.observations[0]
        current.seed_measurement_summary = replace(summary, observations=(replace(item,
            pixel_measurement=replace(item.pixel_measurement, contour_signature=(float('nan'),))),))
        with self.assertRaises(ValueError): result_report(current)
