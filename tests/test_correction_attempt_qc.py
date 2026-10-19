"""Actual correction producers and their recorded terminal QC boundaries."""

from tests.functional_units import functional_units
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


@functional_units
class CorrectionAttemptQCTests(_Fixture, unittest.TestCase):


    def check_test_process_control_and_secondary_record_failure_preserve_original_exception(self):
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

    def check_test_context_argument_failure_does_not_claim_project_wrapper_invocation(self):
        worker = self.worker()
        worker.binding = object()
        completion = self.run_worker(worker)
        self.assertIsInstance(completion.result, AttributeError)
        row = worker.quality.payload['identity']['attempt_record']
        self.assertEqual(row['phase'], 'request_capture')
        self.assertIsNone(row['invoked_executor'])
        self.assertFalse(row['result_returned'])

    def check_test_strict_route_and_conditional_return_schema_reject_inconsistent_claims(self):
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


    def test_validation_and_recovery(self):
        """3 related CorrectionAttemptQCTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_process_control_and_secondary_record_failure_preserve_original_exception)
        self.scenario(self.check_test_context_argument_failure_does_not_claim_project_wrapper_invocation)
        self.scenario(self.check_test_strict_route_and_conditional_return_schema_reject_inconsistent_claims)


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
