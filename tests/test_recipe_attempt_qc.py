"""Actual recipe terminal boundaries, historical transport and Qt ownership."""

from tests.functional_units import functional_units
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


@functional_units
class RecipeAttemptQCTests(RecipeAttemptFixture, unittest.TestCase):

    def check_test_actual_selected_model_mutation_fails_before_result_without_format_claim(self):
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

    def check_test_process_control_propagates_and_secondary_recording_failure_preserves_original_exception(self):
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

    def check_test_before_after_changes_are_observations_and_historical_loading_never_reopens_files(self):
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

    def check_test_actual_file_and_aggregate_capture_limits_leave_oversize_bytes_unknown(self):
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



    def check_test_distinct_attempts_and_changed_facts_cannot_reuse_acknowledgment(self):
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


    def test_validation_and_recovery(self):
        """3 related RecipeAttemptQCTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_actual_selected_model_mutation_fails_before_result_without_format_claim)
        self.scenario(self.check_test_process_control_propagates_and_secondary_recording_failure_preserves_original_exception)
        self.scenario(self.check_test_actual_file_and_aggregate_capture_limits_leave_oversize_bytes_unknown)

    def test_workflow_lifecycle(self):
        """2 related RecipeAttemptQCTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_before_after_changes_are_observations_and_historical_loading_never_reopens_files)
        self.scenario(self.check_test_distinct_attempts_and_changed_facts_cannot_reuse_acknowledgment)


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
