"""Actual correction producers and their recorded terminal QC boundaries."""
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
import shutil
import tempfile
import threading
import time
from types import MappingProxyType, SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch

from seedvision.benchmarking.native_correction_recipe import (
    build_native_correction_recipe, NativeCorrectionCancelled, NativeCorrectionError, STAGE_ID,
)
from seedvision.benchmarking.native_recipe import NativeRecipeCancelled
from seedvision.export import method_qc as qc, recipe_attempt_qc as attempts
from seedvision.persistence import method_qc_history as history, snapshot as snapshots
from seedvision.persistence.correction_session import CorrectionProjectContext
from seedvision.segmentation.baseline import PipelineAnalysisCache
from seedvision.ui import native_correction_dialog as ui, native_recipe_dialog as generic_ui
from tests.test_native_correction_dialog import _write_inputs
from tests import test_project_correction_inputs as project_fixtures


class _Fixture:
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.paths = _write_inputs(self.root)
        self.recipe = build_native_correction_recipe()

    def worker(self, *, cache=None, generation=1):
        return ui._CorrectionRunWorker(self.recipe, self.paths, cache or PipelineAnalysisCache(),
                                      None, threading.Event(), generation=generation)

    def run_worker(self, worker):
        delivered = []
        worker.signals.completed.connect(delivered.append)
        worker.run()
        self.assertEqual(len(delivered), 1)
        return delivered[0]

    def project_worker(self):
        fixture = project_fixtures.ProjectCorrectionInputTests()
        fixture.setUp()
        self.addCleanup(fixture.tearDown)
        master, document = fixture.project()
        binding = document.correction_bindings[0]
        paths = {asset.role: fixture.workspace/asset.path for asset in binding.inputs}
        context = CorrectionProjectContext(fixture.workspace, document.images[0], fixture.source)
        worker = ui._CorrectionRunWorker(fixture.recipe, paths, PipelineAnalysisCache(), None,
            threading.Event(), generation=1, context=context, binding=binding)
        return fixture, master, worker


class CorrectionAttemptQCTests(_Fixture, unittest.TestCase):
    def test_actual_wrong_shape_and_type_retain_four_roles_without_returned_products(self):
        for value in (np.zeros((2, 2), np.uint8), np.zeros((17, 25, 3), np.uint8)):
            with self.subTest(shape=value.shape):
                self.assertTrue(cv2.imwrite(str(self.paths['target_mask']), value))
                worker = self.worker()
                completion = self.run_worker(worker)
                self.assertIsInstance(completion.result, NativeCorrectionError)
                self.assertNotIn('WinError 5', str(completion.result))
                self.assertIsNone(completion.prepared)
                self.assertIsNotNone(worker.quality, worker.quality_error)
                identity = worker.quality.payload['identity']
                row = identity['attempt_record']
                self.assertEqual((row['state'], row['phase']), ('failed', 'correction_path'))
                self.assertEqual(row['invoked_executor'], attempts.CORRECTION_FILE_EXECUTOR)
                self.assertFalse(row['result_returned'])
                self.assertIsNone(row['returned_inputs'])
                self.assertEqual([item['name'] for item in row['inputs_before']], list(self.paths))
                for item in row['inputs_before']:
                    self.assertEqual(item['observation']['sha256'], sha256(self.paths[item['name']].read_bytes()).hexdigest())
                for key in ('native_artifact', 'node_outcomes', 'measurement_eligibility', 'scientific_validity'):
                    self.assertIsNone(identity[key])
                self.assertFalse(list(self.root.glob('**/native-result.json')))

    def test_actual_project_pin_failure_uses_project_wrapper_without_original_fallback(self):
        fixture, master, worker = self.project_worker()
        selected = worker.paths['target_mask']
        selected.write_bytes(b'Changed selected transport bytes.')
        for original in fixture.paths.values():
            original.unlink()
        completion = self.run_worker(worker)
        self.assertIsInstance(completion.result, ValueError)
        self.assertNotIn('WinError 5', str(completion.result))
        row = worker.quality.payload['identity']['attempt_record']
        self.assertEqual(row['invoked_executor'], attempts.PROJECT_CORRECTION_EXECUTOR)
        self.assertEqual((row['state'], row['phase'], row['result_returned']), ('failed', 'correction_path', False))
        observed = next(item for item in row['inputs_before'] if item['name'] == 'target_mask')
        self.assertEqual(observed['locator'], str(selected))
        self.assertEqual(observed['observation']['sha256'], sha256(selected.read_bytes()).hexdigest())
        self.assertIsNone(row['returned_inputs'])
        self.assertTrue(master.exists())

    def test_process_control_and_secondary_record_failure_preserve_original_exception(self):
        worker = self.worker()
        delivered = []
        worker.signals.completed.connect(delivered.append)
        with patch.object(ui._inputs, 'run_native_correction_files', side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                worker.run()
        self.assertEqual(delivered, [])
        original = NativeCorrectionError('Original correction failure.')
        with patch.object(ui._inputs, 'run_native_correction_files', side_effect=original), \
             patch.object(generic_ui, 'recipe_attempt_qc', side_effect=OSError('Secondary capture failure.')):
            worker.run()
        self.assertIs(delivered[0].result, original)
        self.assertIn('Secondary capture failure', worker.quality_error)
        self.assertIn('Secondary capture failure', original.__notes__[0])

    def test_context_argument_failure_does_not_claim_project_wrapper_invocation(self):
        worker = self.worker()
        worker.binding = object()
        completion = self.run_worker(worker)
        self.assertIsInstance(completion.result, AttributeError)
        row = worker.quality.payload['identity']['attempt_record']
        self.assertEqual(row['phase'], 'request_capture')
        self.assertIsNone(row['invoked_executor'])
        self.assertFalse(row['result_returned'])

    def test_strict_route_and_conditional_return_schema_reject_inconsistent_claims(self):
        self.assertTrue(cv2.imwrite(str(self.paths['target_mask']), np.zeros((2, 2), np.uint8)))
        worker = self.worker()
        self.run_worker(worker)
        for changes in (
            {'invoked_executor': attempts.NATIVE_PATH_EXECUTOR},
            {'phase': 'native_path'},
            {'phase': 'result_validation'},
            {'invoked_executor': 'seedvision.benchmarking.native_correction_recipe'},
            {'result_returned': True, 'returned_inputs': {'source_sha256': 'a'*64,
                'coordinate_frame': {'name': 'source', 'shape': [17, 25]}}},
        ):
            value = worker.quality.payload
            value['identity']['attempt_record'].update(changes)
            value['qc_revision'] = sha256(qc._bytes(value['identity'])).hexdigest()
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                qc.MethodQC(qc._bytes(value).decode('utf-8'))
        from tests.test_native_recipe import selected_recipe
        from seedvision.benchmarking.native_recipe import build_native_seed_recipe
        request = attempts.capture_recipe_attempt_request(build_native_seed_recipe(selected_recipe()), {})
        with self.assertRaises(ValueError):
            attempts.recipe_attempt_qc(request, state='failed', phase='correction_path',
                detail='Wrong route.', exception=ValueError(), invoked=True,
                invoked_executor=attempts.PROJECT_CORRECTION_EXECUTOR)

    def test_actual_correction_qc_ack_survives_source_removal_snapshot_and_resnapshot(self):
        fixture, master, worker = self.project_worker()
        worker.cancel.set()
        completion = self.run_worker(worker)
        self.assertIsInstance(completion.result, NativeRecipeCancelled)
        quality = worker.quality
        self.assertEqual(quality.payload['identity']['attempt_record']['invoked_executor'], attempts.PROJECT_CORRECTION_EXECUTOR)
        ack_path = fixture.workspace/'reviewed.json'
        qc.acknowledge_method_qc(quality, ack_path, 'correction reviewer',
                                reviewed_at_utc='2026-10-12T12:00:00+00:00')
        history.retain_method_qc(fixture.workspace, quality, project_path=master,
            acknowledgment=qc.load_method_qc_acknowledgment(ack_path, quality))
        before = history.load_project_method_qc(fixture.workspace, master)
        package = snapshots.create_snapshot(fixture.workspace, master, fixture.root/'packages')
        shutil.rmtree(fixture.workspace)
        shutil.rmtree(fixture.root/'origin')
        restored = fixture.root/'restored'
        with patch.object(attempts, '_observe', side_effect=AssertionError('Historical source probe')):
            working = snapshots.restore_snapshot(package, restored)
            after = history.load_project_method_qc(restored, working)
            second = snapshots.create_snapshot(restored, working, package.parent)
            repeated = snapshots.restore_snapshot(second, restored)
            again = history.load_project_method_qc(restored, repeated)
        self.assertEqual((before.records, before.acknowledgments), (after.records, after.acknowledgments))
        self.assertEqual((before.records, before.acknowledgments), (again.records, again.acknowledgments))
        request = attempts.capture_recipe_attempt_request(self.recipe, {})
        other = attempts.recipe_attempt_qc(request, state='cancelled', phase='correction_path', detail='Different attempt.',
            invoked=True, invoked_executor=attempts.PROJECT_CORRECTION_EXECUTOR)
        self.assertNotEqual(other.revision, quality.revision)
        self.assertFalse(again.diagnostics)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required for actual correction completion')
class CorrectionAttemptQtTests(_Fixture, unittest.TestCase):
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

    def dialog(self):
        dialog = ui.NativeCorrectionDialog(self.recipe)
        self.addCleanup(self.dispose, dialog)
        for name, path in self.paths.items():
            dialog.set_input_path(name, path)
        return dialog

    def wait(self, dialog):
        deadline = time.monotonic()+25
        while dialog._worker is not None and time.monotonic() < deadline:
            self.application.processEvents()
            time.sleep(.01)
        self.assertIsNone(dialog._worker, dialog.status_label.text())

    def test_async_failure_clears_previous_success_and_opens_actual_attempt_viewer(self):
        from seedvision.ui.method_qc_dialog import MethodQCDialog
        dialog = self.dialog()
        dialog.start_run()
        self.wait(dialog)
        self.assertIsNotNone(dialog.result, dialog.status_label.text())
        sentinel = object()
        dialog.cache.values['unrelated'] = sentinel
        self.assertTrue(cv2.imwrite(str(self.paths['target_mask']), np.zeros((2, 2), np.uint8)))
        dialog.start_run()
        self.wait(dialog)
        self.assertIsNone(dialog.result)
        self.assertIsNone(dialog._prepared)
        self.assertIsNone(dialog._result_generation)
        self.assertFalse(dialog.export_button.isEnabled())
        self.assertIs(dialog.cache.values['unrelated'], sentinel)
        quality = dialog._attempt_quality
        self.assertIsNotNone(quality, dialog.status_label.text())
        self.assertEqual(quality.payload['identity']['attempt_record']['invoked_executor'], attempts.CORRECTION_FILE_EXECUTOR)
        with patch.object(MethodQCDialog, 'exec', return_value=0), \
             patch.object(MethodQCDialog, 'set_quality', autospec=True) as installed:
            dialog._review_method_qc()
        self.assertEqual(installed.call_args.args[1], quality)

    def test_stop_after_actual_file_and_project_return_records_distinct_source_target_facts(self):
        target = cv2.imread(str(self.paths['target_rgb']), cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(str(self.paths['target_mask']), cv2.IMREAD_UNCHANGED)
        self.assertTrue(cv2.imwrite(str(self.paths['target_rgb']), np.pad(target, ((0, 0), (0, 4), (0, 0)), mode='edge')))
        self.assertTrue(cv2.imwrite(str(self.paths['target_mask']), np.pad(mask, ((0, 0), (0, 4)))))
        _, _, project = self.project_worker()
        for worker, owner, name, executor in (
            (self.worker(), ui._inputs, 'run_native_correction_files', attempts.CORRECTION_FILE_EXECUTOR),
            (project, ui, 'run_project_correction', attempts.PROJECT_CORRECTION_EXECUTOR),
        ):
            actual = []
            original = getattr(owner, name)
            def run_then_stop(*args, **kwargs):
                result, prepared = original(*args, **kwargs)
                actual.append(result)
                worker.cancel.set()
                return result, prepared
            # Instrument the UI binding, preserving the verified preparation
            # module's actual callable closure during the real runner call.
            instrument = (patch.object(ui, '_inputs', SimpleNamespace(run_native_correction_files=run_then_stop))
                          if executor == attempts.CORRECTION_FILE_EXECUTOR else
                          patch.object(owner, name, side_effect=run_then_stop))
            with self.subTest(executor=executor), instrument:
                completion = self.run_worker(worker)
                self.assertIsInstance(completion.result, NativeCorrectionCancelled)
                self.assertIsNone(completion.prepared)
                row = worker.quality.payload['identity']['attempt_record']
                self.assertEqual((row['state'], row['phase'], row['result_returned']), ('cancelled', 'delivery', True))
                self.assertEqual(row['invoked_executor'], executor)
                self.assertEqual(row['returned_inputs'], {'source_sha256': actual[0].source_sha256,
                    'coordinate_frame': actual[0].coordinate_frame.to_dict(),
                    'target_source_sha256': actual[0].target_source_sha256,
                    'target_coordinate_frame': actual[0].target_coordinate_frame.to_dict()})
                self.assertNotEqual(actual[0].source_sha256, actual[0].target_source_sha256)
                if executor == attempts.CORRECTION_FILE_EXECUTOR:
                    self.assertNotEqual(actual[0].coordinate_frame.shape, actual[0].target_coordinate_frame.shape)
                self.assertTrue(actual[0].nodes[0].outputs['corrected_image'].is_cuda)
                self.assertIsNone(worker.quality.payload['identity']['native_artifact'])

    def test_queued_stop_deduplicates_history_without_ui_file_reads_or_installation(self):
        dialog = self.dialog()
        worker = self.worker(cache=dialog.cache, generation=dialog._generation)
        completion = self.run_worker(worker)
        self.assertNotIsInstance(completion.result, Exception)
        dialog._worker, dialog._cancel = worker, worker.cancel
        dialog.stop()
        with patch.object(attempts, '_observe', side_effect=AssertionError('No Qt source reads')):
            dialog._correction_completed(worker, completion)
            dialog._correction_completed(worker, completion)
        self.assertIsNone(dialog._worker)
        self.assertIsNone(dialog.result)
        self.assertIsNone(dialog._prepared)
        self.assertIsNone(dialog._result_generation)
        self.assertEqual(len(dialog._attempt_history), 1)
        row = dialog._attempt_quality.payload['identity']['attempt_record']
        self.assertIsNone(row['inputs_after'])
        self.assertEqual(row['returned_inputs']['target_source_sha256'], completion.result.target_source_sha256)
        self.assertFalse(dialog.export_button.isEnabled())

    def test_stale_generation_preserves_replacement_worker_outputs_cache_and_controls(self):
        dialog = self.dialog()
        old = self.worker(cache=dialog.cache, generation=dialog._generation)
        completion = self.run_worker(old)
        dialog._completed(completion)
        self.assertIs(dialog.result, completion.result)
        dialog._generation += 1
        newer = self.worker(cache=dialog.cache, generation=dialog._generation)
        dialog._worker, dialog._cancel = newer, newer.cancel
        dialog.status_label.setText('Replacement correction is active.')
        before, selected, generation = dict(dialog.cache.values), dialog._selected_stage, dialog._generation
        prepared, result_generation = dialog._prepared, dialog._result_generation
        with patch.object(attempts, '_observe', side_effect=AssertionError('No stale Qt source reads')):
            dialog._correction_completed(old, completion)
        self.assertIs(dialog._worker, newer)
        self.assertIs(dialog._cancel, newer.cancel)
        self.assertIs(dialog.result, completion.result)
        self.assertIs(dialog._prepared, prepared)
        self.assertEqual((dialog._selected_stage, dialog._generation, dialog._result_generation),
                         (selected, generation, result_generation))
        self.assertEqual(dialog.status_label.text(), 'Replacement correction is active.')
        self.assertEqual(set(dialog.cache.values), set(before))
        self.assertTrue(all(dialog.cache.values[key] is value for key, value in before.items()))
        self.assertIsNone(dialog._attempt_quality)
        self.assertEqual(len(dialog._attempt_history), 1)
        self.assertEqual(dialog._attempt_history[0].payload['identity']['attempt_record']['state'], 'cancelled')

    def test_actual_return_with_invalid_ports_records_validation_failure_without_outputs(self):
        dialog = self.dialog()
        worker = self.worker(cache=dialog.cache, generation=dialog._generation)
        completion = self.run_worker(worker)
        bad_node = replace(completion.result.nodes[0], outputs=MappingProxyType({}))
        bad = replace(completion.result, nodes=(bad_node,))
        dialog._worker, dialog._cancel = worker, worker.cancel
        with patch.object(attempts, '_observe', side_effect=AssertionError('No validation Qt source reads')):
            dialog._correction_completed(worker, ui._Completion(bad, completion.prepared, completion.generation))
        self.assertIsNone(dialog.result)
        self.assertIsNone(dialog._prepared)
        self.assertFalse(dialog.export_button.isEnabled())
        row = dialog._attempt_quality.payload['identity']['attempt_record']
        self.assertEqual((row['state'], row['phase'], row['result_returned']), ('failed', 'result_validation', True))
        self.assertEqual(row['invoked_executor'], attempts.CORRECTION_FILE_EXECUTOR)
        self.assertEqual(row['returned_inputs']['target_source_sha256'], completion.result.target_source_sha256)
        self.assertIsNone(dialog._attempt_quality.payload['identity']['node_outcomes'])
