"""Actual recipe terminal boundaries, historical transport and Qt ownership."""
from hashlib import sha256
from pathlib import Path
import shutil
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch

from seedvision.benchmarking import native_recipe as runner
from seedvision.benchmarking.checkpoint_inspection import CheckpointInspection, InspectionStatus
from seedvision.benchmarking.recipe_catalogue import build_recipe_catalogue
from seedvision.benchmarking.recipes import initial_stage_run
from seedvision.export import method_qc as qc, recipe_attempt_qc as attempts
from seedvision.persistence import (ProjectAnalysisStore, ProjectImageSpec,
    analysis_settings_profile_from_graph, method_qc_history as history, snapshot as snapshots)
from seedvision.pipeline import build_default_pipeline
from seedvision.segmentation.baseline import PipelineAnalysisCache
from seedvision.ui import native_recipe_dialog as native_ui
from tests.test_native_recipe import fixture, selected_recipe, PREFIX


class RecipeAttemptFixture:
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.recipe = runner.build_native_seed_recipe(selected_recipe())
        self.path = self.root/'seed.png'
        gray, _ = fixture()
        self.assertTrue(cv2.imwrite(str(self.path), np.repeat(gray[..., None], 3, axis=2)))

    def attempt(self, request=None, **options):
        request = request or attempts.capture_recipe_attempt_request(self.recipe, {'source_image': self.path})
        error = RuntimeError('Actual selected attempt failed.')
        return attempts.recipe_attempt_qc(request, state='failed', phase='native_path',
            detail=str(error), exception=error, invoked=True, **options)


class RecipeAttemptQCTests(RecipeAttemptFixture, unittest.TestCase):
    def test_actual_corrupt_image_preparation_records_no_completed_outputs(self):
        self.path.write_bytes(b'This selected file is not an encoded image.')
        worker = native_ui._RunWorker(self.recipe, self.path, PipelineAnalysisCache(), threading.Event())
        delivered = []
        worker.signals.completed.connect(delivered.append)
        worker.run()
        self.assertEqual(len(delivered), 1)
        self.assertIsInstance(delivered[0], runner.NativeRecipeError)
        self.assertNotIn('WinError 5', str(delivered[0]), 'A sandbox path error is not the intended decode witness.')
        self.assertIsNotNone(worker.quality, worker.quality_error)
        value = worker.quality.payload['identity']
        record = value['attempt_record']
        self.assertEqual((record['state'], record['phase'], record['result_returned']), ('failed', 'native_path', False))
        self.assertEqual(record['invoked_executor'], 'seedvision.benchmarking.native_recipe.run_native_recipe_path')
        self.assertEqual(record['inputs_before'][0]['observation']['sha256'], sha256(self.path.read_bytes()).hexdigest())
        for key in ('native_artifact', 'node_outcomes', 'recorded_inspection', 'measurement_eligibility', 'scientific_validity'):
            self.assertIsNone(value[key])
        self.assertIsNone(record['returned_inputs'])
        self.assertFalse(list(self.root.glob('**/native-result.json')))

    def test_actual_selected_model_mutation_fails_before_result_without_format_claim(self):
        from tests.test_native_naive_bayes_recipe import _recipe_with_model, MODEL_STAGE_ID
        from tests.test_plantcv_naive_bayes import _pdf
        model = self.root/'model.tsv'
        model.write_bytes(_pdf())
        recipe, _ = _recipe_with_model(model)
        model.write_bytes(b'Corrupt replacement of the selected model.')
        worker = native_ui._RunWorker(recipe, self.path, PipelineAnalysisCache(), threading.Event(),
                                      {MODEL_STAGE_ID: model})
        delivered = []
        worker.signals.completed.connect(delivered.append)
        worker.run()
        self.assertIsInstance(delivered[0], Exception)
        self.assertNotIn('WinError 5', str(delivered[0]), 'The selected model must reach its actual binding guard.')
        self.assertIn('model', str(delivered[0]).lower())
        record = worker.quality.payload['identity']['attempt_record']
        self.assertFalse(record['result_returned'])
        self.assertIsNone(record['returned_inputs'])
        selected = next(row for row in record['inputs_before'] if row['name'] == 'model.'+MODEL_STAGE_ID)
        self.assertEqual(selected['observation']['sha256'], sha256(model.read_bytes()).hexdigest())
        self.assertIsNone(worker.quality.payload['identity']['native_artifact'])

    def test_process_control_propagates_and_secondary_recording_failure_preserves_original_exception(self):
        worker = native_ui._RunWorker(self.recipe, self.path, PipelineAnalysisCache(), threading.Event())
        delivered = []
        worker.signals.completed.connect(delivered.append)
        with patch.object(native_ui, 'run_native_recipe_path', side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                worker.run()
        self.assertEqual(delivered, [])
        self.assertIsNone(worker.quality)
        original = RuntimeError('Original execution error.')
        with patch.object(native_ui, 'run_native_recipe_path', side_effect=original), \
             patch.object(native_ui, 'recipe_attempt_qc', side_effect=OSError('QC disk read failed.')):
            worker.run()
        self.assertIs(delivered[0], original)
        self.assertIn('QC disk read failed', worker.quality_error)
        self.assertIn('QC disk read failed', original.__notes__[0])

    def test_before_after_changes_are_observations_and_historical_loading_never_reopens_files(self):
        request = attempts.capture_recipe_attempt_request(self.recipe, {'source_image': self.path,
            'missing_selection': self.root/'absent.png'})
        before = request.payload['inputs'][0]['observation']['sha256']
        self.path.write_bytes(b'Different bytes at terminal capture.')
        quality = self.attempt(request)
        row = quality.payload['identity']['attempt_record']
        self.assertNotEqual(before, row['inputs_after'][0]['observation']['sha256'])
        self.assertEqual(row['inputs_before'][1]['observation']['state'], 'missing')
        path = qc.export_method_qc(quality, self.root/'historical.json')
        self.path.unlink()
        with patch.object(attempts, '_observe', side_effect=AssertionError('Historical source probing')):
            self.assertEqual(qc.load_method_qc(path), quality)
        detached = request.payload
        detached['inputs'].clear()
        self.assertEqual(len(request.payload['inputs']), 2)

    def test_actual_file_and_aggregate_capture_limits_leave_oversize_bytes_unknown(self):
        selections = {}
        for name, size in (('first', 12), ('second', 12), ('third', 12), ('large', 17)):
            path = self.root/(name+'.bin')
            path.write_bytes(b'a'*size)
            selections[name] = path
        with patch.object(attempts, 'MAX_FILE_BYTES', 16), patch.object(attempts, 'MAX_TOTAL_BYTES', 24):
            request = attempts.capture_recipe_attempt_request(self.recipe, selections)
            rows = request.payload['inputs']
            self.assertEqual([row['observation']['state'] for row in rows],
                             ['hashed', 'hashed', 'budget_exhausted', 'over_limit'])
            self.assertTrue(all(row['observation']['sha256'] is None for row in rows[2:]))
            quality = self.attempt(request)
            self.assertEqual(quality.payload['identity']['attempt_record']['inputs_before'], rows)

    def test_uncaptured_request_and_unavailable_original_executor_do_not_invent_native_execution(self):
        unknown = attempts.capture_recipe_attempt_request(None, {'selected_request': self.root/'absent.recipe.json'})
        rejected = attempts.recipe_attempt_qc(unknown, state='rejected', phase='request_capture',
                                             detail='Selected recipe could not be parsed.')
        self.assertIsNone(rejected.payload['recipe'])
        self.assertIsNone(rejected.payload['identity']['recipe_identity'])
        reference = build_recipe_catalogue().recipe('plantcv_seed_otsu')
        request = attempts.capture_recipe_attempt_request(reference, {'source_image': self.path})
        unavailable = attempts.recipe_attempt_qc(request, state='unavailable', phase='batch_selection',
                                                 detail='Declared original executor unavailable.')
        self.assertEqual(unavailable.payload['identity']['recipe_mode'], reference.provenance.mode.value)
        self.assertIsNone(unavailable.payload['identity']['attempt_record']['invoked_executor'])
        self.assertIsNone(unavailable.payload['identity']['node_outcomes'])

    def test_strict_import_rejects_forged_route_output_return_and_nonterminal_claims(self):
        quality = self.attempt()
        for mutate in (
            lambda p: p.update(version=True),
            lambda p: p['identity'].update(native_artifact={}),
            lambda p: p['identity'].update(node_outcomes=[]),
            lambda p: p['identity'].update(scientific_validity=True),
            lambda p: p['identity']['attempt_record'].update(state='executed'),
            lambda p: p['identity']['attempt_record'].update(invoked_executor='plantcv'),
            lambda p: p['identity']['attempt_record'].update(phase='batch_selection'),
            lambda p: p['identity']['attempt_record'].update(result_returned=True, invoked_executor=None),
            lambda p: p['identity']['attempt_record'].update(returned_inputs={'source_sha256':'a'*64, 'coordinate_frame': {}}),
            lambda p: p['identity']['attempt_record']['inputs_after'][0].update(locator='another.png'),
        ):
            payload = quality.payload
            mutate(payload)
            payload['qc_revision'] = sha256(qc._bytes(payload['identity'])).hexdigest()
            with self.subTest(payload=payload['identity']['attempt_record']):
                with self.assertRaises(ValueError):
                    qc.MethodQC(qc._bytes(payload).decode('utf-8'))
        with self.assertRaises(ValueError):
            qc.MethodQC(quality.payload_json[:-1]+',"version":2}')

    def test_distinct_attempts_and_changed_facts_cannot_reuse_acknowledgment(self):
        first, second = self.attempt(), self.attempt()
        self.assertNotEqual(first.revision, second.revision)
        ack = self.root/'ack.json'
        qc.acknowledge_method_qc(first, ack, 'recorded operator', reviewed_at_utc='2026-10-12T12:00:00+00:00')
        with self.assertRaisesRegex(ValueError, 'stale'):
            qc.load_method_qc_acknowledgment(ack, second)
        changed = first.payload
        changed['identity']['attempt_record']['detail'] = 'Additional captured failure detail.'
        changed['qc_revision'] = sha256(qc._bytes(changed['identity'])).hexdigest()
        with self.assertRaisesRegex(ValueError, 'stale'):
            qc.load_method_qc_acknowledgment(ack, qc.MethodQC(qc._bytes(changed).decode('utf-8')))

    def test_mixed_v1_v2_exact_history_and_acknowledgments_survive_restore_and_resnapshot(self):
        master = ProjectAnalysisStore(self.root).capture_and_save(
            analysis_settings=analysis_settings_profile_from_graph(build_default_pipeline()),
            images=(ProjectImageSpec(self.path, fixture()[0].shape),),
            destination=self.root/'portable.seedfiddle-project.json')
        reference = build_recipe_catalogue().recipe('plantcv_seed_otsu')
        stage = next(row for row in reference.stages if row.operation_id == 'threshold.otsu')
        legacy = qc.qc_from_stage_inspection(reference, initial_stage_run(reference, stage.stage_id),
            CheckpointInspection(InspectionStatus.NOT_RUN, 'Historical selected stage not run.'))
        terminal = self.attempt()
        for index, quality in enumerate((legacy, terminal)):
            ack = self.root/(str(index)+'-ack.json')
            qc.acknowledge_method_qc(quality, ack, 'operator '+str(index), reviewed_at_utc='2026-10-12T12:00:00+00:00')
            history.retain_method_qc(self.root, quality, project_path=master,
                acknowledgment=qc.load_method_qc_acknowledgment(ack, quality))
        before = history.load_project_method_qc(self.root, master)
        package = snapshots.create_snapshot(self.root, master, self.root.parent/(self.root.name+'-snapshots'))
        self.addCleanup(shutil.rmtree, package.parent, True)
        shutil.rmtree(self.root)
        restored = self.root.parent/(self.root.name+'-restored')
        self.addCleanup(shutil.rmtree, restored, True)
        with patch.object(attempts, '_observe', side_effect=AssertionError('Historical input probing')):
            working = snapshots.restore_snapshot(package, restored)
            after = history.load_project_method_qc(restored, working)
            second = snapshots.create_snapshot(restored, working, package.parent)
            repeated = snapshots.restore_snapshot(second, restored)
            again = history.load_project_method_qc(restored, repeated)
        self.assertEqual((before.records, before.acknowledgments), (after.records, after.acknowledgments))
        self.assertEqual((before.records, before.acknowledgments), (again.records, again.acknowledgments))
        self.assertEqual({history._quality(raw).payload['version'] for raw in again.records}, {1, 2})
        self.assertFalse(again.diagnostics)


class RecipeAttemptQtTests(RecipeAttemptFixture, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.application = QApplication.instance() or QApplication([])

    def dispose(self, dialog):
        dialog._worker = None
        dialog._cancel = None
        dialog.reject()
        dialog.deleteLater()
        self.application.processEvents()

    def test_actual_async_failure_viewer_reopens_acknowledges_and_retains_without_outputs(self):
        from seedvision.ui.method_qc_dialog import MethodQCDialog
        self.path.write_bytes(b'Actual corrupt image selected for Qt execution.')
        dialog = native_ui.NativeRecipeDialog(self.recipe)
        self.addCleanup(self.dispose, dialog)
        dialog.set_source_path(self.path)
        sentinel = object()
        dialog.cache.values['unrelated-retained'] = sentinel
        dialog.start_run()
        deadline = time.monotonic()+20
        while dialog._worker is not None and time.monotonic() < deadline:
            self.application.processEvents()
            time.sleep(.01)
        self.assertIsNone(dialog._worker)
        self.assertIsNone(dialog.result)
        self.assertFalse(dialog.export_button.isEnabled())
        self.assertFalse(dialog.histogram.isVisible())
        self.assertIs(dialog.cache.values['unrelated-retained'], sentinel)
        quality = dialog._attempt_quality
        self.assertIsNotNone(quality, dialog.status_label.text())
        self.assertNotIn('WinError 5', quality.payload_json)
        with patch.object(MethodQCDialog, 'exec', return_value=0), \
             patch.object(MethodQCDialog, 'set_quality', autospec=True) as installed:
            dialog._review_method_qc()
        self.assertEqual(installed.call_args.args[1], quality)
        view = MethodQCDialog(quality, project_root=self.root, project_path=self.root/'owned.seedfiddle-project.json')
        self.addCleanup(self.dispose, view)
        self.assertIn('Recorded recipe attempt: failed', view.summary.text())
        exported, ack = self.root/'view.json', self.root/'view-ack.json'
        with patch('seedvision.ui.method_qc_dialog.QFileDialog.getSaveFileName', return_value=(str(exported), '')):
            view._export()
        with patch('seedvision.ui.method_qc_dialog.QFileDialog.getOpenFileName', return_value=(str(exported), '')):
            view._load()
        view.operator.setText('Qt attempt reviewer')
        with patch('seedvision.ui.method_qc_dialog.QFileDialog.getSaveFileName', return_value=(str(ack), '')):
            view._acknowledge()
        view._retain()
        retained = history.load_project_method_qc(self.root, self.root/'owned.seedfiddle-project.json')
        self.assertEqual([history._quality(raw) for raw in retained.records], [quality])
        self.assertEqual(len(retained.acknowledgments), 1)
        self.assertIn('Qt attempt reviewer', view.acknowledgment['operator_account'])
        self.assertIsNone(dialog.result)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_stop_after_actual_runner_return_records_captured_frame_without_installation(self):
        worker = native_ui._RunWorker(self.recipe, self.path, PipelineAnalysisCache(), threading.Event())
        delivered, actual = [], []
        worker.signals.completed.connect(delivered.append)
        original = native_ui.run_native_recipe_path
        def run_then_stop(*args, **kwargs):
            result = original(*args, **kwargs)
            actual.append(result)
            worker.cancel.set()
            return result
        with patch.object(native_ui, 'run_native_recipe_path', side_effect=run_then_stop):
            worker.run()
        self.assertIsInstance(delivered[0], runner.NativeRecipeCancelled)
        self.assertIsNotNone(worker.quality, worker.quality_error)
        row = worker.quality.payload['identity']['attempt_record']
        self.assertTrue(row['result_returned'])
        self.assertEqual(row['returned_inputs'], {'source_sha256':actual[0].source_sha256,
            'coordinate_frame':actual[0].coordinate_frame.to_dict()})
        self.assertEqual((row['state'], row['phase']), ('cancelled', 'delivery'))
        self.assertIsNone(worker.quality.payload['identity']['node_outcomes'])
        tensors = [value for node in actual[0].nodes for value in node.outputs.values() if isinstance(value, torch.Tensor)]
        self.assertTrue(tensors)
        self.assertTrue(all(value.is_cuda for value in tensors))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_late_queued_stop_retains_unknown_after_inputs_and_clears_only_owned_outputs(self):
        dialog = native_ui.NativeRecipeDialog(self.recipe)
        self.addCleanup(self.dispose, dialog)
        dialog.set_source_path(self.path)
        worker = native_ui._RunWorker(self.recipe, self.path, dialog.cache, threading.Event())
        delivered = []
        worker.signals.completed.connect(delivered.append)
        worker.run()
        self.assertNotIsInstance(delivered[0], Exception)
        dialog._worker, dialog._cancel = worker, worker.cancel
        dialog.stop()
        with patch.object(attempts, '_observe', side_effect=AssertionError('No UI-thread file probing')):
            dialog._run_completed(worker, delivered[0])
        self.assertIsNone(dialog.result)
        self.assertIsNone(dialog._worker)
        row = dialog._attempt_quality.payload['identity']['attempt_record']
        self.assertTrue(row['result_returned'])
        self.assertIsNone(row['inputs_after'])
        self.assertEqual(row['returned_inputs']['coordinate_frame'], delivered[0].coordinate_frame.to_dict())
        self.assertFalse(dialog.export_button.isEnabled())

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_stale_same_recipe_delivery_preserves_replacement_worker_chart_cache_and_selection(self):
        self.recipe = runner.build_native_seed_recipe(selected_recipe(**{'analysis.color': {'colorspaces': 'hsv'}}))
        dialog = native_ui.NativeRecipeDialog(self.recipe)
        self.addCleanup(self.dispose, dialog)
        dialog.set_source_path(self.path)
        old = native_ui._RunWorker(self.recipe, self.path, dialog.cache, threading.Event())
        delivered = []
        old.signals.completed.connect(delivered.append)
        old.run()
        self.assertNotIsInstance(delivered[0], Exception)
        dialog._completed(delivered[0])
        stage = PREFIX+'analyze_color'
        dialog._select_stage(stage)
        dialog.show()
        self.application.processEvents()
        chart = dialog.histogram.chart
        self.assertIsNotNone(chart)
        newer = native_ui._RunWorker(self.recipe, self.path, dialog.cache, threading.Event())
        dialog._worker, dialog._cancel = newer, newer.cancel
        dialog.status_label.setText('Replacement calculation is active.')
        before = dict(dialog.cache.values)
        old.cancel.set()
        dialog._run_completed(old, delivered[0])
        self.assertIs(dialog._worker, newer)
        self.assertIs(dialog._cancel, newer.cancel)
        self.assertIs(dialog.result, delivered[0])
        self.assertEqual(dialog._selected_stage, stage)
        self.assertIs(dialog.histogram.chart, chart)
        self.assertEqual(dialog.status_label.text(), 'Replacement calculation is active.')
        self.assertEqual(set(dialog.cache.values), set(before))
        self.assertTrue(all(dialog.cache.values[key] is value for key, value in before.items()))
        self.assertIsNone(dialog._attempt_quality)
        self.assertEqual(len(dialog._attempt_history), 1)
        self.assertEqual(dialog._attempt_history[0].payload['identity']['attempt_record']['state'], 'cancelled')
