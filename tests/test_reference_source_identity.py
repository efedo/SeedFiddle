"""Source association is preserved separately from annotation generation."""
from dataclasses import replace
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from seedvision.annotation.coordinates import reproject_bundle
from seedvision.measurement.lineage import prepare_reviewed_inputs, reviewed_measurement_context
from seedvision.persistence.reference_regions import (
    ImageFingerprintMismatch, InvalidReferenceArchive, ReferenceRegionBundle,
    ReferenceRegionError, ReferenceRegionStore, file_sha256,
)
from tests.test_reviewed_measurement_lineage import analysis_fixture, annotation, labels_fixture


class ReferenceSourceArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.image = self.root / 'capture.png'
        self.image.write_bytes(b'annotation source snapshot')
        self.digest = file_sha256(self.image)
        self.store = ReferenceRegionStore(self.root)
        self.labels = labels_fixture()
        self.bundle = ReferenceRegionBundle(self.labels.shape, annotated_seeds=self.labels,
            seed_annotations=(annotation(),), source_to_corrected=np.eye(3))

    def payload(self):
        destination = self.store.save(self.image, self.bundle)
        with np.load(destination, allow_pickle=False) as archive:
            return destination, {name: np.array(archive[name]) for name in archive.files}

    @staticmethod
    def write_payload(destination, payload):
        with destination.open('wb') as stream:
            np.savez_compressed(stream, **payload)

    def test_v7_round_trip_preserves_known_generation_and_unknown_history(self):
        for known in (None, self.digest):
            with self.subTest(snapshot=known):
                path = self.store.save(self.image, replace(self.bundle,
                    annotation_snapshot_source_sha256=known))
                loaded = self.store.load_if_present(self.image)
                self.assertEqual(loaded.archive_source_sha256, self.digest)
                self.assertEqual(loaded.annotation_snapshot_source_sha256, known)
                with np.load(path, allow_pickle=False) as archive:
                    self.assertEqual(int(archive['version']), 7)
                    self.assertEqual('annotation_snapshot_source_sha256' in archive.files, known is not None)

    def test_every_legacy_version_preserves_binding_without_inventing_generation(self):
        destination, payload = self.payload()
        payload['seed_annotations_json'] = np.asarray('[]')
        payload['annotation_snapshot_source_sha256'] = np.asarray(self.digest)
        for version in range(1, 7):
            with self.subTest(version=version):
                payload['version'] = np.asarray(version, np.uint16)
                payload['boundary'] = np.zeros(self.labels.shape, np.uint8)
                self.write_payload(destination, payload)
                loaded = self.store.load_if_present(self.image)
                self.assertEqual(loaded.archive_source_sha256, self.digest)
                self.assertIsNone(loaded.annotation_snapshot_source_sha256)
                self.store.save(self.image, loaded)
                restored = self.store.load_if_present(self.image)
                self.assertIsNone(restored.annotation_snapshot_source_sha256)

    def test_reprojection_preserves_both_source_identities(self):
        bundle = replace(self.bundle, archive_source_sha256=self.digest,
            annotation_snapshot_source_sha256=self.digest)
        current = np.eye(3); current[0, 2] = 2
        shifted = reproject_bundle(bundle, current, self.labels.shape)
        self.assertEqual(shifted.archive_source_sha256, self.digest)
        self.assertEqual(shifted.annotation_snapshot_source_sha256, self.digest)
        self.assertFalse(np.array_equal(shifted.annotated_seeds, self.labels))

    def test_save_rejects_bad_identity_before_rasters_or_destination_creation(self):
        for name in ('archive_source_sha256', 'annotation_snapshot_source_sha256'):
            for bad in ('', 'A'*64, 'g'*64, 'a'*63, True, '0'*64):
                with self.subTest(field=name, value=bad), patch(
                    'seedvision.persistence.reference_regions._binary_mask', side_effect=AssertionError('raster accessed')
                ), self.assertRaises(ReferenceRegionError):
                    self.store.save(self.image, replace(self.bundle, **{name: bad}))
        self.assertFalse(self.store.directory.exists())

    def test_replaced_source_cannot_rebind_loaded_snapshot_or_overwrite_archive(self):
        destination = self.store.save(self.image, self.bundle)
        loaded = self.store.load_if_present(self.image)
        before = destination.read_bytes()
        self.image.write_bytes(b'different capture at the same path')
        with self.assertRaises(ImageFingerprintMismatch):
            self.store.save(self.image, loaded)
        self.assertEqual(destination.read_bytes(), before)

    def test_generation_metadata_rejects_malformed_scalar_and_sequence_values(self):
        destination, payload = self.payload()
        for bad in ('', 'A'*64, 'g'*64, 'a'*63, 123, [self.digest, self.digest]):
            with self.subTest(value=bad):
                payload['annotation_snapshot_source_sha256'] = np.asarray(bad)
                self.write_payload(destination, payload)
                with self.assertRaises(InvalidReferenceArchive):
                    self.store.load_if_present(self.image)

    def test_both_identity_mismatches_reject_before_raster_materialization(self):
        destination, payload = self.payload()
        real_load = np.load

        class MetadataOnly:
            def __init__(self, archive): self.archive = archive; self.files = archive.files
            def __enter__(self): return self
            def __exit__(self, *args): self.archive.close()
            def __getitem__(self, key):
                if key in {'material', 'annotated_seeds', 'boundary'}:
                    raise AssertionError('raster accessed before source validation')
                return self.archive[key]

        for field in ('image_sha256', 'annotation_snapshot_source_sha256'):
            with self.subTest(field=field):
                changed = dict(payload, **{field: np.asarray('0'*64)})
                self.write_payload(destination, changed)
                with patch('seedvision.persistence.reference_regions.np.load',
                    side_effect=lambda *args, **kwargs: MetadataOnly(real_load(*args, **kwargs))
                ), self.assertRaises(ImageFingerprintMismatch):
                    self.store.load_if_present(self.image)


class ReferenceSourceAnalysisTests(unittest.TestCase):
    def test_saved_project_batch_forwards_archive_and_generation_bindings(self):
        from seedvision.export.project_runner import run_project
        from seedvision.persistence import ProjectAnalysisStore, ProjectImageSpec, analysis_settings_profile_from_graph
        from seedvision.pipeline import build_default_pipeline
        with TemporaryDirectory() as temporary:
            root = Path(temporary); source = root/'capture.png'
            cv2.imwrite(str(source), np.zeros((64, 96, 3), np.uint8))
            digest = file_sha256(source); labels = labels_fixture()
            archive = ReferenceRegionStore(root).save(source, ReferenceRegionBundle(labels.shape,
                annotated_seeds=labels, seed_annotations=(annotation(),), source_to_corrected=np.eye(3),
                annotation_snapshot_source_sha256=digest))
            (root/'config').mkdir()
            shutil.copyfile(Path(__file__).parents[1]/'config'/'traits.json', root/'config'/'traits.json')
            master = ProjectAnalysisStore(root).capture_and_save(
                analysis_settings=analysis_settings_profile_from_graph(build_default_pipeline()),
                images=(ProjectImageSpec(source, labels.shape, reference_regions_path=archive),),
                destination=root/'test.seedfiddle-project.json')
            before = master.read_bytes()
            with patch('seedvision.export.project_runner.analyze_path',
                side_effect=lambda *args, **kwargs: analysis_fixture(source=source)
            ) as analyzer, patch('seedvision.export.project_runner.export_project_native_graphs'):
                self.assertEqual(len(run_project(root, master, root/'output')), 1)
            self.assertEqual(analyzer.call_args.kwargs['reference_archive_source_sha256'], digest)
            self.assertEqual(analyzer.call_args.kwargs['reference_annotation_snapshot_source_sha256'], digest)
            self.assertEqual(master.read_bytes(), before)

    def test_source_states_are_independent_of_transport_and_each_other(self):
        labels = labels_fixture(); image = np.zeros((*labels.shape, 3), np.uint8)
        known = 'a'*64
        for source, archive, snapshot, states in (
            (known, None, None, ('not_recorded', 'not_recorded')),
            (known, known, None, ('verified', 'not_recorded')),
            (known, None, known, ('not_recorded', 'verified')),
            (known, known, known, ('verified', 'verified')),
            (None, known, known, ('recorded_unverified', 'recorded_unverified')),
        ):
            with self.subTest(source=source, archive=archive, snapshot=snapshot):
                context = reviewed_measurement_context(source, image, np.eye(3), np.eye(3),
                    transport_applied=True, reference_archive_source_sha256=archive,
                    reference_annotation_snapshot_source_sha256=snapshot)
                lineage = prepare_reviewed_inputs(labels, (annotation(),), context=context).lineage
                self.assertEqual((lineage.reference_archive_source_identity_state,
                    lineage.reference_annotation_snapshot_source_identity_state), states)
                self.assertEqual(lineage.source_image_frame, 'full_corrected')
                self.assertEqual(lineage.reference_annotation_snapshot_source_sha256, snapshot)
        for name in ('reference_archive_source_sha256', 'reference_annotation_snapshot_source_sha256'):
            with self.subTest(field=name), self.assertRaises(ValueError):
                reviewed_measurement_context(known, image, np.eye(3), **{name: 'b'*64})
        context = reviewed_measurement_context(known, image, np.eye(3),
            reference_archive_source_sha256=known)
        with self.assertRaises(ValueError):
            prepare_reviewed_inputs(labels, (), context=replace(context,
                reference_archive_source_identity_state='recorded_unverified'))

    def test_analyze_path_rejects_mismatch_before_decoding_and_forwards_verified_facts(self):
        from seedvision.segmentation.baseline import analyze_path
        with TemporaryDirectory() as temporary:
            path = Path(temporary)/'capture.png'
            cv2.imwrite(str(path), np.zeros((24, 32, 3), np.uint8))
            digest = file_sha256(path)
            for name in ('reference_archive_source_sha256', 'reference_annotation_snapshot_source_sha256'):
                with self.subTest(field=name), patch('seedvision.segmentation.baseline.cv2.imread',
                    side_effect=AssertionError('decoded mismatched source')
                ), self.assertRaises(ValueError):
                    analyze_path(path, **{name: '0'*64})
            with patch('seedvision.segmentation.baseline.analyze_image', return_value='done') as analyzer:
                self.assertEqual(analyze_path(path, reference_archive_source_sha256=digest,
                    reference_annotation_snapshot_source_sha256=digest), 'done')
                self.assertEqual(analyzer.call_args.kwargs['_source_sha256'], digest)
                self.assertEqual(analyzer.call_args.kwargs['reference_archive_source_sha256'], digest)
                self.assertEqual(analyzer.call_args.kwargs['reference_annotation_snapshot_source_sha256'], digest)

    def test_actual_producer_identity_changes_recompute_only_measurement_dependents(self):
        from seedvision.calibration import image as calibration
        from seedvision.calibration.geometry import DishDetectionSettings
        from seedvision.segmentation.baseline import analyze_image, PipelineAnalysisCache
        from tests.test_neutral_calibration import fixture, ruler
        image, card = fixture(); labels = labels_fixture(image.shape[:2]); digest = 'a'*64
        cache = PipelineAnalysisCache()
        options = dict(node_cache=cache, enabled_nodes=set(), _source_sha256=digest,
            calibration_settings=calibration.CalibrationSettings(apply_perspective_correction=False),
            dish_settings=DishDetectionSettings(expected_outer_diameter_mm=0),
            seed_instance_annotations=labels, seed_instance_traits=(annotation(),),
            reference_transform=np.eye(3), reference_archive_source_sha256=digest)
        with patch.object(calibration, 'detect_colour_card', return_value=card), \
             patch.object(calibration, 'detect_ruler', side_effect=ruler):
            first = analyze_image(image, **options)
            same = analyze_image(image, **options)
            self.assertIs(first.seed_measurement_summary, same.seed_measurement_summary)
            self.assertEqual(first.seed_measurement_summary.lineage.reference_archive_source_identity_state, 'verified')
            self.assertEqual(first.seed_measurement_summary.lineage.reference_annotation_snapshot_source_identity_state, 'not_recorded')
            options['reference_annotation_snapshot_source_sha256'] = digest
            changed = analyze_image(image, **options)
            self.assertIsNot(changed.seed_measurement_summary, first.seed_measurement_summary)
            self.assertIs(changed.calibration.gpu_corrected_bgr, first.calibration.gpu_corrected_bgr)
            self.assertIn('seed_scale_estimation', cache.last_computed_nodes)
            self.assertIn('deskew_colour', cache.last_reused_nodes)
            self.assertEqual(changed.seed_measurement_summary.lineage.reference_annotation_snapshot_source_identity_state, 'verified')
            options['reference_annotation_snapshot_source_sha256'] = 'b'*64
            with patch('seedvision.segmentation.baseline.CudaContext.resolve',
                side_effect=AssertionError('CUDA accessed before identity validation')
            ), self.assertRaises(ValueError):
                analyze_image(image, **options)

    def test_export_records_both_facts_and_each_change_stales_review(self):
        from seedvision.export.results import result_report
        with TemporaryDirectory() as temporary:
            source = Path(temporary)/'capture.png'; source.write_bytes(b'current source')
            digest = file_sha256(source); current = analysis_fixture(source=source)
            before = result_report(current)
            for name in ('reference_archive_source_sha256', 'reference_annotation_snapshot_source_sha256'):
                with self.subTest(field=name):
                    summary = current.seed_measurement_summary
                    state = name.removesuffix('_sha256') + '_identity_state'
                    lineage = replace(summary.lineage, **{name: digest, state: 'verified',
                        'reference_source_identity_state': 'verified'})
                    current.seed_measurement_summary = replace(summary, lineage=lineage)
                    report = result_report(current, review={'result_revision': before['result_revision'],
                        'decisions': {'1': 'accepted'}})
                    self.assertNotEqual(before['result_revision'], report['result_revision'])
                    self.assertEqual(report['instances'][0]['review'], 'unreviewed')
                    self.assertEqual(report['reviewed_measurements']['lineage'][name], digest)
                    current.seed_measurement_summary = replace(summary, lineage=replace(lineage, **{name: '0'*64}))
                    with self.assertRaises(ValueError): result_report(current)
                    current.seed_measurement_summary = summary

    def test_recorded_unverified_history_is_not_upgraded_by_export(self):
        from seedvision.export.results import result_report
        with TemporaryDirectory() as temporary:
            source = Path(temporary)/'capture.png'; source.write_bytes(b'current source')
            digest = file_sha256(source); current = analysis_fixture(source=source)
            summary = current.seed_measurement_summary
            lineage = replace(summary.lineage, source_sha256=None,
                reference_archive_source_sha256=digest, reference_archive_source_identity_state='recorded_unverified',
                reference_source_identity_state='recorded_unverified')
            current.seed_measurement_summary = replace(summary, lineage=lineage)
            report = result_report(current)
            self.assertEqual(report['reviewed_measurements']['lineage']['reference_archive_source_identity_state'], 'recorded_unverified')
            self.assertFalse(report['reviewed_measurements']['scientifically_validated'])


class ReferenceSourceDesktopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.application = QApplication.instance() or QApplication([])

    def test_native_controller_preserves_loaded_facts_through_save_and_clear(self):
        from seedvision.ui.main_window import _path_identity
        from tests.test_project_analysis_ui import ProjectAnalysisUiTests as ui
        from tests.review_fixtures import dispose_window
        with TemporaryDirectory() as temporary:
            root = Path(temporary); source = ui._write_image(root/'images'/'capture.png')
            window = ui._window(root)
            self.addCleanup(dispose_window, window)
            key = _path_identity(source); labels = np.zeros((36, 48), np.uint16); labels[10:20, 10:20] = 1
            digest = file_sha256(source)
            bundle = ReferenceRegionBundle(labels.shape, annotated_seeds=labels,
                seed_annotations=(annotation(),), source_to_corrected=np.eye(3),
                archive_source_sha256=digest, annotation_snapshot_source_sha256=digest)
            window._install_reference_region_bundle(key, bundle, sync_view=False)
            recovered = window._reference_bundle_for_project_key(key, source)
            self.assertEqual(recovered.archive_source_sha256, digest)
            self.assertEqual(recovered.annotation_snapshot_source_sha256, digest)
            window._persist_applied_reference_regions(automatic=True)
            loaded = window._reference_region_store.load_if_present(source)
            self.assertEqual(loaded.annotation_snapshot_source_sha256, digest)
            result = SimpleNamespace(image_path=source, source_sha256='0'*64,
                calibration=SimpleNamespace(affine_matrix=np.eye(3), corrected_bgr=np.zeros((36, 48, 3), np.uint8)))
            self.assertFalse(window._prepare_reference_frame(key, result))
            self.assertEqual(window._reference_source_identities[key], (digest, digest))
            window._clear_project_session_state()
            self.assertEqual(window._reference_source_identities, {})

    def test_worker_detaches_and_forwards_reference_identity_metadata(self):
        import inspect
        from seedvision.ui.main_window import _AnalysisTask
        arguments = {name: None for name, parameter in inspect.signature(_AnalysisTask).parameters.items()
            if parameter.default is inspect.Parameter.empty}
        arguments.update(path=Path('capture.png'), learning_root=Path('.'), pipeline_revision=1,
            species='unknown', seed_trait_species='', background_colour_enabled=True,
            background_reference_points=(), foreground_reference_points=(), seed_instance_traits=(),
            seed_trait_coat_patterns=(), seed_trait_conditions=(), enabled_nodes=frozenset(),
            dirty_nodes=frozenset(), reference_archive_source_sha256='a'*64,
            reference_annotation_snapshot_source_sha256='b'*64)
        task = _AnalysisTask(**arguments)
        with patch('seedvision.segmentation.baseline.analyze_path', return_value=SimpleNamespace()) as analyzer:
            task.run()
        self.assertEqual(analyzer.call_args.kwargs['reference_archive_source_sha256'], 'a'*64)
        self.assertEqual(analyzer.call_args.kwargs['reference_annotation_snapshot_source_sha256'], 'b'*64)


if __name__ == '__main__':
    unittest.main()
