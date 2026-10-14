"""Synthetic native evidence, CUDA extraction, stop and publication witnesses."""
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch

from seedvision.learning.trait_classifier import (
    TraitTrainingCancelled, predict_traits, train_trait_model,
)
from seedvision.learning.trait_features import TraitFeatureBatch
from seedvision.learning.trait_training import (
    TraitTrainingConfiguration, TraitTrainingRequest, activate_trait_model,
    load_frozen_trait_model, materialize_trait_evidence, read_trait_manifest,
    save_frozen_trait_model, train_trait_manifest,
)
from seedvision.persistence.reference_regions import (
    ReferenceRegionBundle, ReferenceRegionStore, SeedInstanceAnnotation, file_sha256,
)
from seedvision.pipeline import build_default_pipeline


def compact_manifest():
    examples = []
    for split, values in (('train', (.1, .2, .8, .9)), ('calibration', (.15, .85))):
        for index, value in enumerate(values):
            examples.append(dict(example_id=f'{split}-{index}', physical_id=f'{split}-physical-{index}',
                group_id=f'{split}-group', trait_id='dark' if value < .5 else 'light',
                truth_provenance='synthetic software fixture; no biological truth', permitted_use=True,
                split=split, input=dict(kind='compact_features', values=[value] + [0.] * 19,
                                       provenance='synthetic declared features; no verified extractor')))
    return dict(format='seedfiddle-trait-training', version=1, feature_schema='seed-classifier-pixel-v1',
        vocabulary=dict(revision='synthetic-v1', definition_source='synthetic fixture', definition_source_sha256='a' * 64,
            traits=[dict(identifier='dark', definition='Synthetic low first feature', evidence_reference='synthetic'),
                    dict(identifier='light', definition='Synthetic high first feature', evidence_reference='synthetic')]), examples=examples)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required for native trait training')
class TraitTrainingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.path = self.root / 'evidence.json'
        self.output = self.root / 'model.json'
        self.payload = compact_manifest()
        self.config = TraitTrainingConfiguration(iterations=80, calibration_iterations=11)

    def manifest(self):
        self.path.write_text(json.dumps(self.payload), encoding='utf-8')
        return read_trait_manifest(self.path)

    def request(self, **changes):
        manifest = self.manifest()
        return replace(TraitTrainingRequest(self.path, self.output, self.config, False, manifest.content_sha256), **changes)

    def reference(self, item, *, transform=np.eye(3), generation=True, boundary=False):
        identifier = item['example_id']
        image_path = self.root / f'{identifier}.png'
        image = np.zeros((12, 12, 3), np.uint8)
        image[4:8, 4:8, 2] = 30 if item['trait_id'] == 'dark' else 220
        image[0, 0, 0] = len(identifier) + len(list(self.root.glob('*.png')))
        self.assertTrue(cv2.imwrite(str(image_path), image))
        labels = np.zeros((12, 12), np.int32)
        labels[4:8, 4:8] = 9
        if boundary:
            labels[0, 0] = 9
        archive_path = ReferenceRegionStore(self.root).save(image_path, ReferenceRegionBundle(
            labels.shape, annotated_seeds=labels, source_to_corrected=transform,
            seed_annotations=(SeedInstanceAnnotation(9, shape_reviewed=True, outline_visibility='complete', pose='flat',
                physical_seed_id=item['physical_id']),),
            annotation_snapshot_source_sha256=file_sha256(image_path) if generation else None))
        item['input'] = dict(kind='reference_seed', image_path=str(image_path), image_sha256=file_sha256(image_path),
            archive_path=str(archive_path), archive_sha256=file_sha256(archive_path), seed_id=9)
        return image_path, archive_path

    def test_compact_training_atomic_reload_and_actual_prediction(self):
        progress = []
        report = train_trait_manifest(self.request(), progress_callback=lambda *values: progress.append(values))
        model = load_frozen_trait_model(self.output)
        self.assertEqual(model.model_sha256, report['model_sha256'])
        self.assertTrue(report['saved'])
        self.assertFalse(report['scientifically_validated'])
        self.assertEqual({stage for stage, _, _ in progress}, {'evidence', 'training', 'calibration'})
        training, calibration, _, audit = materialize_trait_evidence(read_trait_manifest(self.path))
        self.assertTrue(training.values.is_cuda)
        result = predict_traits(model, TraitFeatureBatch(calibration.values, (1, 2), (), '{}'), abstention_threshold=.7)
        self.assertEqual([row['predicted_trait_id'] for row in result['predictions']], ['dark', 'light'])
        self.assertEqual(model.payload['native_training_evidence']['input_audit'], audit)
        self.assertIn('not verified', audit[0]['representation'])
        self.assertEqual(list(self.root.glob('.trait-model-*')), [])

    def test_real_reference_cuda_extraction_matches_independent_pixel_oracle(self):
        for item in self.payload['examples']:
            self.reference(item, generation=item['split'] == 'train')
        with patch.object(torch.Tensor, 'cpu', side_effect=AssertionError('CUDA raster downloaded')):
            training, calibration, vocabulary, audit = materialize_trait_evidence(self.manifest())
        expected = torch.tensor([30 / 255, 30 / 255, 220 / 255, 220 / 255], device='cuda', dtype=torch.float64)
        torch.testing.assert_close(training.values[:, 0], expected)
        torch.testing.assert_close(training.values[:, 15], torch.full((4,), np.log(16), device='cuda', dtype=torch.float64))
        torch.testing.assert_close(training.values[:, 19], torch.full((4,), 4., device='cuda', dtype=torch.float64))
        self.assertEqual({row['generation_state'] for row in audit}, {'unknown', 'recorded_matching'})
        report = train_trait_manifest(TraitTrainingRequest(self.path, self.output, self.config))
        self.assertTrue(report['saved'])
        self.assertFalse(load_frozen_trait_model(self.output).payload['scientifically_validated'])
        from types import SimpleNamespace
        from seedvision.learning.trait_pipeline import TraitClassifierSettings, run_trait_classifier
        source = self.payload['examples'][-1]['input']
        image = cv2.imread(source['image_path'])
        labels = np.zeros((12, 12), np.int32)
        labels[4:8, 4:8] = 9
        with patch.object(torch.Tensor, 'cpu', side_effect=AssertionError('CUDA raster downloaded')):
            prediction, _ = run_trait_classifier('procedural_instances', SimpleNamespace(labels=labels, source_shape=(12, 12)),
                corrected_bgr=image, crop_offset=(0, 0), affine=np.eye(3),
                gpu_source=torch.tensor(image, device='cuda').permute(2, 0, 1)[None].float(),
                gpu_valid=torch.ones((1, 1, 12, 12), device='cuda', dtype=torch.bool),
                settings=TraitClassifierSettings(str(self.output), .7), root=self.root, cache={})
        self.assertEqual(prediction.report['predictions'][0]['predicted_trait_id'], 'light')
        self.assertEqual(prediction.report['model_sha256'], report['model_sha256'])

    def test_denied_unusable_input_is_not_resolved_or_opened(self):
        denied = dict(self.payload['examples'][0], example_id='denied', physical_id='denied-physical',
                      group_id='denied-group', permitted_use=False, input={'kind': 'nonsense', 'image_path': None,
                      'values': {'unusable': 'intentionally'}})
        self.payload['examples'].append(denied)
        with patch('seedvision.learning.trait_training._reference_features', side_effect=AssertionError('Denied input accessed')):
            report = train_trait_manifest(self.request())
        model = load_frozen_trait_model(self.output)
        self.assertFalse(model.payload['native_training_evidence']['input_audit'][-1]['included'])
        self.assertEqual(model.payload['split_audit']['training']['excluded'][0]['example_id'], 'denied')

    def test_manifest_and_controls_reject_before_any_raster_or_optimizer(self):
        for config in ({'iterations': True}, {'learning_rate': float('nan')}, {'l2': -1},
                       {'temperature_bounds': (2, 1)}, {'calibration_iterations': 1}, {'abstention_threshold': True}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                TraitTrainingConfiguration(**config)
        for key in ('group_id', 'physical_id', 'example_id'):
            payload = compact_manifest()
            payload['examples'][-1][key] = payload['examples'][0][key]
            self.path.write_text(json.dumps(payload))
            with self.subTest(key=key), patch('cv2.imread', side_effect=AssertionError('raster ran')), self.assertRaisesRegex(ValueError, 'overlap'):
                read_trait_manifest(self.path)
        self.path.write_text('{"format":1,"format":2}')
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            read_trait_manifest(self.path)

    def test_file_hash_and_missing_transform_reject_before_cuda_extraction(self):
        item = self.payload['examples'][0]
        _, archive = self.reference(item, transform=None)
        with patch('seedvision.learning.trait_training.extract_trait_features', side_effect=AssertionError('extraction ran')):
            for field in ('image_sha256', 'archive_sha256'):
                saved = item['input'][field]
                item['input'][field] = 'b' * 64
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
                    materialize_trait_evidence(self.manifest())
                item['input'][field] = saved
            with self.assertRaisesRegex(ValueError, 'recorded.*transform'):
                materialize_trait_evidence(self.manifest())

    def test_actual_duplicate_mask_aliases_cannot_cross_splits(self):
        first, last = self.payload['examples'][0], self.payload['examples'][-1]
        image, archive = self.reference(first)
        # Unknown physical annotation is preserved so declared aliases cannot conceal identical pixels.
        bundle = ReferenceRegionStore(self.root).load_project_archive(image, archive)
        alias = ReferenceRegionStore(self.root).save(image, replace(bundle,
            seed_annotations=(replace(bundle.seed_annotations[0], physical_seed_id=None),)))
        first['input']['archive_sha256'] = file_sha256(alias)
        last['input'] = dict(first['input'])
        with patch('torch.optim.Adam', side_effect=AssertionError('fitting ran')), self.assertRaisesRegex(ValueError, 'Actual source/selected-mask.*overlaps'):
            train_trait_manifest(self.request())

    def test_known_generation_mismatch_rejects_before_raster_decode(self):
        item = self.payload['examples'][0]
        _, archive_path = self.reference(item)
        with np.load(archive_path, allow_pickle=False) as archive:
            fields = {key: archive[key] for key in archive.files}
        fields['annotation_snapshot_source_sha256'] = np.asarray('b' * 64)
        np.savez_compressed(archive_path, **fields)
        item['input']['archive_sha256'] = file_sha256(archive_path)
        with patch('cv2.imdecode', side_effect=AssertionError('Source decoded')), self.assertRaisesRegex(ValueError, 'require image SHA'):
            materialize_trait_evidence(self.manifest())

    def test_consumed_snapshots_survive_path_replacement_and_enforce_bounded_reads(self):
        item = self.payload['examples'][0]
        image, archive = self.reference(item)
        original_image, original_archive = image.read_bytes(), archive.read_bytes()
        load_snapshot = ReferenceRegionStore.load_project_archive_snapshot
        def replaced_during_archive_load(store, image_path, archive_path, **contents):
            image.write_bytes(b'changed source path')
            archive.write_bytes(b'changed archive path')
            try:
                return load_snapshot(store, image_path, archive_path, **contents)
            finally:
                image.write_bytes(original_image)
                archive.write_bytes(original_archive)
        decode = cv2.imdecode
        def replaced_during_decode(content, flags):
            # Replacing both paths after materialization cannot change consumed evidence.
            image.write_bytes(b'changed path')
            archive.write_bytes(b'changed archive path')
            result = decode(content, flags)
            image.write_bytes(original_image)
            archive.write_bytes(original_archive)
            return result
        with patch.object(ReferenceRegionStore, 'load_project_archive_snapshot', replaced_during_archive_load), \
             patch('cv2.imdecode', side_effect=replaced_during_decode):
            training, _, _, audit = materialize_trait_evidence(self.manifest())
        self.assertAlmostEqual(training.values[0, 0].item(), 30 / 255)
        self.assertEqual(audit[0]['consumed_bytes'], 'immutable hash-matching image/archive snapshots')
        with patch('seedvision.learning.trait_training._MAX_SOURCE_BYTES', 4), \
             patch('cv2.imdecode', side_effect=AssertionError('Oversized input decoded')), \
             self.assertRaisesRegex(ValueError, 'exceeds.*byte limit'):
            materialize_trait_evidence(self.manifest())

    def test_review_boundary_and_source_support_are_actual_eligibility_guards(self):
        self.reference(self.payload['examples'][0], boundary=True)
        with self.assertRaisesRegex(ValueError, 'canvas boundary'):
            materialize_trait_evidence(self.manifest())
        self.reference(self.payload['examples'][0], transform=np.array(((1, 0, 20), (0, 1, 0), (0, 0, 1))))
        with self.assertRaisesRegex(ValueError, 'outside decoded source support'):
            materialize_trait_evidence(self.manifest())

    def test_manifest_snapshot_and_evidence_output_paths_preserve_inputs(self):
        request = self.request()
        self.path.write_text(self.path.read_text() + ' ')
        with self.assertRaisesRegex(ValueError, 'changed after'):
            train_trait_manifest(request)
        with self.assertRaisesRegex(ValueError, 'differ.*manifest'):
            TraitTrainingRequest(self.path, self.path)
        image, _ = self.reference(self.payload['examples'][0])
        before = image.read_bytes()
        with self.assertRaisesRegex(ValueError, 'differ.*input paths'):
            train_trait_manifest(self.request(output_path=image))
        self.assertEqual(before, image.read_bytes())

    def test_each_training_control_changes_actual_fit_or_decision_diagnostics(self):
        training, calibration, vocabulary, _ = materialize_trait_evidence(self.manifest())
        baseline = train_trait_model(training, calibration, vocabulary, iterations=30, calibration_iterations=9).payload
        for controls in ({'iterations': 2}, {'learning_rate': .001}, {'l2': 2.}):
            arguments = dict(iterations=30, calibration_iterations=9)
            arguments.update(controls)
            changed = train_trait_model(training, calibration, vocabulary, **arguments).payload
            self.assertNotEqual(baseline['weights'], changed['weights'])
        cold = train_trait_model(training, calibration, vocabulary, iterations=30, temperature_bounds=(.1, .2), calibration_iterations=3).payload
        hot = train_trait_model(training, calibration, vocabulary, iterations=30, temperature_bounds=(2., 4.), calibration_iterations=7).payload
        self.assertNotEqual(cold['temperature_calibration']['temperature'], hot['temperature_calibration']['temperature'])
        self.assertNotEqual(cold['development_diagnostics']['nll'], hot['development_diagnostics']['nll'])
        strict = train_trait_model(training, calibration, vocabulary, iterations=30, calibration_iterations=9, abstention_threshold=1).payload
        self.assertEqual(strict['weights'], baseline['weights'])
        self.assertEqual(strict['development_diagnostics']['coverage'], 0)

    def test_cancellation_during_evidence_training_calibration_preserves_old_output(self):
        for stop_stage in ('evidence', 'training', 'calibration'):
            stopped = [False]
            self.output.write_bytes(b'previous model bytes')
            def progress(stage, current, maximum):
                if stage == stop_stage:
                    stopped[0] = True
            with self.subTest(stage=stop_stage), self.assertRaises(TraitTrainingCancelled):
                train_trait_manifest(self.request(), cancellation_requested=lambda: stopped[0], progress_callback=progress)
            self.assertEqual(self.output.read_bytes(), b'previous model bytes')
            self.assertEqual(list(self.root.glob('.trait-model-*')), [])

    def test_atomic_precommit_stop_write_failure_and_late_stop(self):
        report = train_trait_manifest(self.request())
        model = load_frozen_trait_model(self.output)
        self.output.write_bytes(b'old')
        checks = [0]
        def stopped():
            checks[0] += 1
            return checks[0] == 2
        with self.assertRaises(TraitTrainingCancelled):
            save_frozen_trait_model(model, self.output, cancellation_requested=stopped)
        self.assertEqual(self.output.read_bytes(), b'old')
        with patch('seedvision.learning.trait_training.os.replace', side_effect=OSError('write failed')), self.assertRaises(OSError):
            save_frozen_trait_model(model, self.output)
        self.assertEqual(self.output.read_bytes(), b'old')
        self.assertEqual(list(self.root.glob('.trait-model-*')), [])
        stopped_after = [False]
        original_replace = __import__('os').replace
        def replaced(source, destination):
            original_replace(source, destination)
            stopped_after[0] = True
        with patch('seedvision.learning.trait_training.os.replace', side_effect=replaced):
            report = train_trait_manifest(self.request(), cancellation_requested=lambda: stopped_after[0])
        self.assertTrue(report['saved'])
        self.assertTrue(report['cancellation_after_save'])
        self.assertEqual(load_frozen_trait_model(self.output).model_sha256, report['model_sha256'])

    def test_activation_is_transactional_and_invalidates_only_classifier_dependents(self):
        train_trait_manifest(self.request())
        graph = build_default_pipeline()
        affected, model = activate_trait_model(graph, self.output, abstention_threshold=.67)
        self.assertEqual(set(affected), {'trait_classifier', *graph.downstream('trait_classifier', recursive=True)})
        self.assertTrue(graph.node('trait_classifier').enabled)
        self.assertEqual(graph.node('trait_classifier').parameters['trait_abstention_threshold'], .67)
        before = graph.to_dict()
        self.output.write_text('{}')
        with self.assertRaises(ValueError):
            activate_trait_model(graph, self.output)
        self.assertEqual(graph.to_dict(), before)
        self.output.write_text(model.to_json())
        graph.set_enabled('trait_classifier', False)
        before = graph.to_dict()
        original = type(graph).set_enabled
        def fail_actual(instance, *args):
            if instance is graph:
                raise ValueError('injected activation failure')
            return original(instance, *args)
        with patch.object(type(graph), 'set_enabled', fail_actual), self.assertRaisesRegex(ValueError, 'injected'):
            activate_trait_model(graph, self.output, abstention_threshold=.5)
        self.assertEqual(graph.to_dict(), before)


class TraitTrainingUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.application = QApplication.instance() or QApplication([])

    def setUp(self):
        import shutil
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / 'config').mkdir()
        shutil.copyfile(Path(__file__).parents[1] / 'config' / 'traits.json', self.root / 'config' / 'traits.json')
        self.path = self.root / 'evidence.json'
        self.path.write_text(json.dumps(compact_manifest()))

    def window(self):
        from seedvision.ui.main_window import MainWindow
        from tests.review_fixtures import dispose_window
        window = MainWindow(self.root)
        self.addCleanup(lambda: dispose_window(window))
        return window

    def test_dialog_snapshots_manifest_controls_and_activation_choice(self):
        from seedvision.ui.trait_training_workflow import TraitTrainingDialog
        dialog = TraitTrainingDialog(project_root=self.root)
        dialog.manifest_row.edit.setText(str(self.path))
        dialog.iterations_spin.setValue(17)
        dialog.learning_rate_spin.setValue(.04)
        dialog.l2_spin.setValue(.2)
        dialog.lower_temperature_spin.setValue(.2)
        dialog.upper_temperature_spin.setValue(5)
        dialog.calibration_spin.setValue(13)
        dialog.threshold_spin.setValue(.7)
        dialog.activate_checkbox.setChecked(True)
        dialog.accept()
        request = dialog.request()
        dialog.iterations_spin.setValue(99)
        self.assertEqual(request.configuration, TraitTrainingConfiguration(17, .04, .2, (.2, 5.), 13, .7))
        self.assertEqual(dialog.request(), request)
        self.assertTrue(request.activate_after_training)
        self.assertEqual(request.manifest_sha256, file_sha256(self.path))
        dialog.close()

    def test_trait_worker_forwards_real_controls_task_identity_and_terminal_outcomes(self):
        from seedvision.ui.trait_training_workflow import TraitTrainingTask
        request = TraitTrainingRequest(self.path, self.root / 'model.json', TraitTrainingConfiguration(iterations=3))
        task = TraitTrainingTask(request)
        progress, reports, cancelled, failed = [], [], [], []
        task.signals.progress.connect(lambda *args: progress.append(args))
        task.signals.completed.connect(lambda *args: reports.append(args))
        task.signals.cancelled.connect(cancelled.append)
        task.signals.failed.connect(lambda *args: failed.append(args))
        def run(received, **callbacks):
            self.assertIs(received, request)
            callbacks['progress_callback']('training', 1, 3)
            return {'saved': True}
        with patch('seedvision.ui.trait_training_workflow.train_trait_manifest', side_effect=run):
            task.run()
        self.assertEqual(progress, [(task, 'training', 1, 3)])
        self.assertEqual(reports, [(task, {'saved': True})])
        with patch('seedvision.ui.trait_training_workflow.train_trait_manifest', side_effect=TraitTrainingCancelled()):
            task.run()
        self.assertEqual(cancelled, [task])
        with patch('seedvision.ui.trait_training_workflow.train_trait_manifest', side_effect=ValueError('bad evidence')):
            task.run()
        self.assertEqual(failed, [(task, 'bad evidence')])

    def test_serial_busy_stop_and_stale_terminal_signals_preserve_current_job(self):
        window = self.window()
        request = TraitTrainingRequest(self.path, self.root / 'model.json')
        with patch.object(window._thread_pool, 'start') as started:
            task = window._begin_trait_training(request)
        started.assert_called_once_with(task)
        self.assertEqual(window._thread_pool.maxThreadCount(), 1)
        self.assertFalse(window.train_trait_model_action.isEnabled())
        with self.assertRaisesRegex(ValueError, 'busy'):
            window._begin_trait_training(request)
        task.cancel()
        self.assertIs(window._learning_training_task, task)
        window._trait_training_cancelled(object())
        self.assertIs(window._learning_training_task, task)
        window._trait_training_progressed(task, 'training', 2, 4)
        self.assertEqual(window._learning_training_progress.value(), 2)
        window._trait_training_cancelled(task)
        self.assertIsNone(window._learning_training_task)
        self.assertTrue(window.train_trait_model_action.isEnabled())

    def test_saved_only_late_stop_and_changed_graph_never_activate(self):
        window = self.window()
        report = dict(output_path='saved.json', cancellation_after_save=False)
        for activate, late_stop, changed in ((False, False, False), (True, True, False), (True, False, True)):
            request = TraitTrainingRequest(self.path, self.root / 'model.json', activate_after_training=activate)
            with patch.object(window._thread_pool, 'start'):
                task = window._begin_trait_training(request)
            if late_stop:
                task.cancel()
            if changed:
                window.pipeline.revision += 1
            with patch.object(window, '_activate_trait_model') as activation:
                window._trait_training_completed(task, report)
            activation.assert_not_called()
            self.assertIsNone(window._learning_training_task)

    def test_completion_activation_failure_keeps_saved_result_and_clears_busy(self):
        window = self.window()
        request = TraitTrainingRequest(self.path, self.root / 'model.json', activate_after_training=True)
        with patch.object(window._thread_pool, 'start'):
            task = window._begin_trait_training(request)
        with patch.object(window, '_activate_trait_model', side_effect=ValueError('upstream disabled')) as activation:
            window._trait_training_completed(task, dict(output_path='saved.json', cancellation_after_save=False))
        activation.assert_called_once_with(request.output_path, abstention_threshold=.8)
        self.assertIn('Model saved; activation failed', window.statusBar().currentMessage())
        self.assertIsNone(window._learning_training_task)

    def test_serial_worker_terminal_slot_runs_on_gui_thread(self):
        from PySide6.QtCore import QThread
        from threading import Event
        from time import monotonic, sleep
        window = self.window()
        request = TraitTrainingRequest(self.path, self.root / 'model.json')
        worker_threads, gui_threads = [], []
        finished = Event()
        def trained(received, **callbacks):
            worker_threads.append(QThread.currentThread())
            callbacks['progress_callback']('training', 1, 1)
            return dict(output_path='saved.json', cancellation_after_save=False)
        # A signal to a real bound MainWindow slot invokes its UI/status work on the GUI thread.
        original_finish = window._finish_learning_training_ui
        def finish():
            gui_threads.append(QThread.currentThread())
            original_finish()
            finished.set()
        with patch('seedvision.ui.trait_training_workflow.train_trait_manifest', side_effect=trained), \
             patch.object(window, '_finish_learning_training_ui', side_effect=finish):
            window._begin_trait_training(request)
            deadline = monotonic() + 5
            while not finished.is_set() and monotonic() < deadline:
                self.application.processEvents()
                sleep(.005)
            self.assertTrue(finished.is_set())
            self.assertTrue(window._thread_pool.waitForDone(5000))
        self.assertEqual(gui_threads, [self.application.thread()])
        self.assertNotEqual(worker_threads[0], self.application.thread())

    def test_active_model_output_is_preserved_and_close_cancels_current_task(self):
        from tests.review_fixtures import dispose_window
        window = self.window()
        current = self.root / 'active.json'
        window.pipeline.set_parameter('trait_classifier', 'trait_model_path', str(current))
        window.pipeline.set_enabled('trait_classifier', True)
        with self.assertRaisesRegex(ValueError, 'new file'):
            window._begin_trait_training(TraitTrainingRequest(self.path, current))
        request = TraitTrainingRequest(self.path, self.root / 'new.json')
        with patch.object(window._thread_pool, 'start'):
            task = window._begin_trait_training(request)
        dispose_window(window)
        self.assertTrue(task.cancellation_requested())
        self.assertIsNone(window._learning_training_task)
        window._trait_training_completed(task, dict(output_path='new.json', cancellation_after_save=False))
        self.assertIsNone(window._learning_training_task)


if __name__ == '__main__':
    unittest.main()
