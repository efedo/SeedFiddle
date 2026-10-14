"""Real directory training, CUDA row caching and asynchronous desktop lifecycle."""
from dataclasses import replace
import os
from pathlib import Path
import random
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtWidgets import QApplication, QDialog

from seedvision.benchmarking import native_kmeans_training as native
from seedvision.benchmarking import native_kmeans_recipe as factory
from seedvision.benchmarking import native_recipe as runner
from seedvision.benchmarking import kmeans_model_artifact as artifacts
from seedvision.cuda import plantcv_kmeans_model as models
from seedvision.ui import native_kmeans_training_dialog as ui
from seedvision.ui import native_kmeans_dialog as picker


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class NativeKMeansTrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.folder = self.root/'training'
        self.folder.mkdir()
        for name in ('a.png', 'b.png'):
            self.assertTrue(cv2.imwrite(str(self.folder/name), np.zeros((17, 18, 3), np.uint8)))
        self.request = native.KMeansTrainingRequest(str(self.folder), patch_size=2, sampling=5, n_init=1)

    def test_real_training_export_and_fit_only_edits_reuse_rows_and_random_selection(self):
        request = replace(self.request, num_imgs=7)
        first = native.train_kmeans_directory(request)
        state = random.getstate()
        with patch.object(native._directory.random, 'choices', side_effect=AssertionError('fit edit resampled files')), \
                patch.object(torch, 'from_numpy', side_effect=AssertionError('fit edit uploaded images')):
            second = native.train_kmeans_directory(replace(request, seed=0xFFFFFFFF, n_init=2, clusters=1),
                cached_rows=first.cache)
        self.assertIs(second.cache.rows, first.cache.rows)
        self.assertEqual(random.getstate(), state)
        self.assertEqual(second.summary.rows, 35)
        self.assertIsNone(second.summary.source_n_threads)
        self.assertEqual(second.summary.source_reduction_profile, 'all-valid-openmp-reductions-v1')
        self.assertEqual(len(second.export.model.centres), 1)
        self.assertEqual(models.parse_kmeans_model(second.export.payload), second.export.model)
        path = self.root/'trained.json'
        path.write_bytes(second.export.payload)
        selected = factory.model_source_for_path(path)
        self.assertEqual(factory.load_selected_model(selected), second.export.model)
        artifact = artifacts.parse_kmeans_artifact(second.artifact_payload)
        provenance = artifact.provenance
        self.assertEqual(provenance['request']['seed'], 0xFFFFFFFF)
        self.assertEqual(provenance['request']['sampling_kind'], 'count')
        self.assertEqual(provenance['manifest']['selected_names'], list(first.cache.manifest.selected_names))
        self.assertEqual(provenance['fit_plan']['rows'], 35)
        self.assertEqual(provenance['fit_parameters']['n_init'], 2)
        self.assertIsNone(provenance['fit_result']['source_n_threads'])
        self.assertEqual(provenance['fit_result']['random_identity']['seed'], 0xFFFFFFFF)
        self.assertEqual(artifact.model.centres, second.export.model.centres)
        self.assertNotEqual(artifact.model.source_sha256, second.export.model.source_sha256)

    def test_extraction_edits_preserve_selection_and_rebuild_only_rows(self):
        first = native.train_kmeans_directory(replace(self.request, num_imgs=6))
        selection = first.cache.without_rows()
        self.assertIsNone(selection.rows)
        with patch.object(native._directory.random, 'choices', side_effect=AssertionError('patch edit resampled files')):
            second = native.train_kmeans_directory(replace(self.request, num_imgs=6, patch_size=3, sampling=7),
                cached_rows=selection)
        self.assertIs(second.cache.manifest, first.cache.manifest)
        self.assertEqual(second.cache.rows.plan.features, 27)
        self.assertEqual(second.summary.rows, 42)
        self.assertNotEqual(second.cache.rows.features.data_ptr(), first.cache.rows.features.data_ptr())
        third = native.train_kmeans_directory(replace(self.request, sampling=.75), cached_rows=second.cache.without_rows())
        self.assertEqual(third.summary.rows, 2*int(.75*16*17))
        self.assertIs(type(third.cache.key[3]), float)

    def test_changed_assets_cache_rows_and_callback_mutations_withhold_models(self):
        first = native.train_kmeans_directory(self.request)
        path = self.folder/'a.png'
        raw = path.read_bytes()
        path.write_bytes(raw+b'changed')
        with patch.object(torch, 'from_numpy', side_effect=AssertionError('upload before asset verification')):
            with self.assertRaisesRegex(ValueError, 'bytes changed'):
                native.train_kmeans_directory(self.request, cached_rows=first.cache)
        path.write_bytes(raw)
        first.cache.rows.features.zero_()
        with self.assertRaisesRegex(ValueError, 'rows changed'):
            native.train_kmeans_directory(self.request, cached_rows=first.cache)
        first = native.train_kmeans_directory(self.request)
        def mutate():
            first.cache.rows.features.zero_()
            return False
        with self.assertRaisesRegex(ValueError, 'rows changed'):
            native.train_kmeans_directory(self.request, cached_rows=first.cache, cancel=mutate)
        with self.assertRaises(native._directory.KMeansDirectoryCancelled):
            native.train_kmeans_directory(self.request, cancel=lambda: True)
        with self.assertRaisesRegex(ValueError, 'bool'):
            native.train_kmeans_directory(self.request, cancel=lambda: 1)

    def test_bad_parameters_and_retained_rows_obey_effective_bounds(self):
        for changes in (dict(patch_size=True), dict(sampling=True), dict(sampling=0),
                dict(sampling=float('nan')), dict(sampling=1.), dict(sampling=1.1), dict(seed=-1), dict(n_init=0)):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                native.train_kmeans_directory(replace(self.request, **changes))
        first = native.train_kmeans_directory(self.request)
        images = native._directory.materialize_kmeans_training_directory(first.cache.manifest)
        changed = replace(self.request, patch_size=3)
        plan = native._training.preflight_kmeans_image_fit(images.images, changed.clusters, source_n_threads=None,
            patch_size=changed.patch_size, sampling=changed.sampling, n_init=changed.n_init)
        identity = native._row_calculation_identity()
        with patch.object(native, '_row_calculation_identity', return_value=identity), \
                patch.object(native._training, 'MAX_CUDA_LIVE_BYTES', plan.peak_live_bytes):
            with self.assertRaisesRegex(ValueError, 'retained row cache'):
                native.train_kmeans_directory(changed, cached_rows=first.cache)

    def test_calculation_identity_rejects_changed_loaded_producer_contract(self):
        identity = native._row_calculation_identity()
        self.assertEqual(len(identity), 64)
        with patch.object(native._training, 'PATCH_SEED', 7):
            with self.assertRaises(ValueError):
                native._row_calculation_identity()

    def test_provenance_projection_never_inspects_or_transfers_dense_fit_products(self):
        completed = native.train_kmeans_directory(self.request)
        provenance = artifacts.parse_kmeans_artifact(completed.artifact_payload).provenance
        class DenseSentinel:
            def __getattribute__(self, name):
                raise AssertionError('provenance traversed dense CUDA data')
        scalar = provenance['fit_result']
        fitted = SimpleNamespace(centres=DenseSentinel(), counts=DenseSentinel(),
            labels=DenseSentinel(), row_distances=DenseSentinel(),
            plan=native._fit.KMeansFitPlan(**provenance['fit_plan']),
            inertia=native._fit._intervals.F64Interval(**scalar['inertia']),
            scaled_tolerance=native._fit._intervals.F64Interval(**scalar['scaled_tolerance']),
            n_steps=scalar['n_steps'], n_iter=scalar['n_iter'], source_n_threads=None,
            source_reduction_profile=scalar['source_reduction_profile'],
            random_identity=scalar['random_identity'])
        result = native._training_provenance(self.request, completed.cache.manifest,
            completed.cache.key[-1], fitted, completed.export.model, completed.cache.rows.features.device)
        self.assertEqual(result['fit_result'], scalar)
        self.assertEqual(artifacts.parse_kmeans_artifact(
            artifacts.encode_kmeans_artifact(completed.export.model, result)).provenance, provenance)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class NativeKMeansTrainingDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.folder = self.root/'training'
        self.folder.mkdir()
        self.assertTrue(cv2.imwrite(str(self.folder/'a.png'), np.zeros((17, 18, 3), np.uint8)))
        self.completion = native.train_kmeans_directory(native.KMeansTrainingRequest(str(self.folder),
            patch_size=2, sampling=5, clusters=1, n_init=1))
        self.dialog = ui.NativeKMeansTrainingDialog()
        self.dialog.directory_edit.setText(str(self.folder))

    def tearDown(self):
        self.dialog.close()
        self.wait()
        self.application.processEvents()
        self.temp.cleanup()

    def wait(self):
        deadline = time.monotonic()+15
        while self.dialog._worker is not None and time.monotonic() < deadline:
            self.application.processEvents()
            time.sleep(.005)
        self.assertIsNone(self.dialog._worker, 'training worker did not drain')

    def configure(self):
        self.dialog.patch_size.setValue(2)
        self.dialog.sampling_mode.setCurrentIndex(1)
        self.dialog.sampling_count.setValue(5)
        self.dialog.clusters.setValue(1)
        self.dialog.n_init.setValue(1)

    def test_source_defaults_uint32_seed_and_sampling_type_reach_request(self):
        request = self.dialog._make_request()
        self.assertEqual((request.patch_size, request.n_init, request.num_imgs, request.sampling), (10, 10, 0, None))
        self.dialog.seed_edit.setText('4294967295')
        self.dialog.prefix_edit.setText('middle')
        self.dialog.image_count.setValue(7)
        self.dialog.sampling_mode.setCurrentIndex(2)
        self.dialog.sampling_fraction.setValue(.999999)
        request = self.dialog._make_request()
        self.assertEqual((request.seed, request.prefix, request.num_imgs), (0xFFFFFFFF, 'middle', 7))
        self.assertIs(type(request.sampling), float)
        self.assertEqual(request.sampling, .999999)
        for text in ('4294967296', '-1', '', 'abc', '١'):
            self.dialog.seed_edit.setText(text)
            with self.assertRaises(ValueError):
                self.dialog._make_request()

    def test_real_async_training_and_parameter_changes_invalidate_only_dependents(self):
        self.configure()
        self.dialog._start_training()
        self.wait()
        self.assertTrue(self.dialog.save_button.isEnabled(), self.dialog.status_label.text())
        cache = self.dialog._cached_rows
        self.dialog.seed_edit.setText('31')
        self.assertIs(self.dialog._cached_rows, cache)
        self.assertFalse(self.dialog.save_button.isEnabled())
        self.dialog._start_training()
        self.wait()
        self.assertIs(self.dialog._cached_rows.rows, cache.rows)
        self.dialog.patch_size.setValue(3)
        self.assertIsNone(self.dialog._cached_rows.rows)
        self.assertIs(self.dialog._cached_rows.manifest, cache.manifest)
        self.dialog.prefix_edit.setText('new')
        self.assertIsNone(self.dialog._cached_rows)

    def test_worker_disables_every_input_and_reject_waits_for_worker_drain(self):
        entered, release = threading.Event(), threading.Event()
        def delayed(request, *, cached_rows, cancel):
            entered.set()
            release.wait(5)
            return self.completion
        self.dialog._operation = delayed
        self.dialog._completion = self.completion
        self.dialog._start_training()
        self.assertTrue(entered.wait(5))
        for widget in self.dialog._selection_controls+self.dialog._row_controls+self.dialog._fit_controls:
            self.assertFalse(widget.isEnabled())
        self.assertFalse(self.dialog.save_button.isEnabled())
        finished = []
        self.dialog.finished.connect(finished.append)
        self.dialog.reject()
        self.assertEqual(finished, [])
        self.assertTrue(self.dialog._cancel.is_set())
        release.set()
        self.wait()
        self.assertEqual(finished, [QDialog.DialogCode.Rejected])
        self.assertIsNone(self.dialog._completion)
        self.assertFalse(self.dialog.save_button.isEnabled())

    def test_stop_and_stale_settings_discard_even_successful_late_worker_results(self):
        for action in ('stop', 'change'):
            entered, release = threading.Event(), threading.Event()
            def delayed(request, *, cached_rows, cancel):
                entered.set()
                release.wait(5)
                return self.completion
            self.dialog._operation = delayed
            self.dialog._start_training()
            self.assertTrue(entered.wait(5))
            if action == 'stop':
                self.dialog._stop_training()
            else:
                self.dialog.seed_edit.setText('19')
            release.set()
            self.wait()
            self.assertIsNone(self.dialog._completion)
            self.assertFalse(self.dialog.save_button.isEnabled())

    def test_atomic_save_and_commit_failure_preserve_existing_file_and_result_state(self):
        path = self.root/'centres.json'
        path.write_bytes(b'existing')
        self.dialog._completion = self.completion
        class FailedSave:
            def __init__(self, path): pass
            def open(self, mode): return True
            def write(self, data): return len(data)
            def commit(self): return False
            def errorString(self): return 'commit failed'
            def cancelWriting(self): pass
        with patch.object(ui.QFileDialog, 'getSaveFileName', return_value=(str(path), '')):
            with patch.object(ui, 'QSaveFile', FailedSave), patch.object(ui.QMessageBox, 'warning') as warning:
                self.dialog._save_model()
            warning.assert_called_once()
            self.assertEqual(path.read_bytes(), b'existing')
            self.assertIsNone(self.dialog.saved_model_path)
            self.dialog._save_model()
        self.assertEqual(path.read_bytes(), self.completion.artifact_payload)
        self.assertEqual(self.dialog.saved_model_path, str(path.resolve()))
        artifact = artifacts.load_kmeans_artifact(path)
        self.assertEqual(artifact.model.centres, self.completion.export.model.centres)
        self.assertEqual(artifact.provenance['manifest']['manifest_sha256'], self.completion.summary.directory_sha256)

    def test_saved_training_model_enters_existing_picker_and_reuses_image_cache(self):
        from tests.test_native_kmeans_recipe import _image, _recipe, _write_model
        initial, trained, image_path = self.root/'old.json', self.root/'trained.json', self.root/'image.png'
        _write_model(initial)
        trained.write_bytes(self.completion.artifact_payload)
        self.assertTrue(cv2.imwrite(str(image_path), _image()))
        dialog = picker.NativeKMeansDialog(_recipe(initial))
        self.addCleanup(dialog.close)
        first = runner.run_native_recipe_path(dialog.recipe, image_path, cache=dialog.cache)
        self.assertEqual(first.status, 'executed')
        class CompletedDialog:
            saved_model_path = str(trained)
            def __init__(self, parent): pass
            def exec(self): return QDialog.DialogCode.Accepted
        with patch.object(picker, 'NativeKMeansTrainingDialog', CompletedDialog):
            dialog._train_model()
        self.assertEqual(dialog._selected_model_source()['path'], str(trained.resolve()))
        result = runner.run_native_recipe_path(dialog.recipe, image_path, cache=dialog.cache)
        self.assertEqual(result.status, 'executed')
        self.assertEqual(dialog.cache.last_reused_nodes, (factory.IMAGE_STAGE_ID,))
        self.assertEqual(dialog.cache.last_computed_nodes, (factory.MODEL_STAGE_ID, factory.PREDICT_STAGE_ID, factory.MASK_STAGE_ID))
        self.assertIs(first.node(factory.IMAGE_STAGE_ID).outputs['rgb_image'], result.node(factory.IMAGE_STAGE_ID).outputs['rgb_image'])
