"""Whole-run terminal evidence, actual Qt/batch parity and publication guards."""
from copy import deepcopy
from hashlib import sha256
import inspect
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from seedvision.export import run_outcomes as runs
from seedvision.timing import AnalysisCancelled
import tests.test_authoritative_parity as parity_fixture


class RunOutcomeTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / 'source.png'
        self.source.write_bytes(b'actual source byte fixture; runner controlled')

    def failed_outcome(self, diagnostic='calibration failed', **kwargs):
        def runner(path, **settings):
            settings['progress_callback']('deskew_colour', 'started')
            raise ValueError(diagnostic)
        with self.assertLogs('seedvision.export.run_outcomes', level='ERROR'):
            return runs.run_native_analysis(self.source, analysis_runner=runner, learning_root=self.root, **kwargs)

    def test_failed_request_is_detached_unknown_outputs_and_diagnostic_paths_do_not_hash(self):
        masks = np.ones((4, 5), bool)
        first = self.failed_outcome('A:/calibration failed', foreground_reference_mask=masks)
        raw = first.payload_json
        masks.fill(False)
        second = self.failed_outcome('B:/calibration failed', foreground_reference_mask=np.ones((4, 5), bool))
        self.assertEqual(first.payload['identity'], second.payload['identity'])
        self.assertEqual(first.payload['outcome_revision'], second.payload['outcome_revision'])
        self.assertNotEqual(first.payload['diagnostic'], second.payload['diagnostic'])
        self.assertEqual(first.payload_json, raw)
        identity = first.payload['identity']
        self.assertEqual(identity['state'], 'failed')
        self.assertEqual(identity['last_started_node'], 'deskew_colour')
        self.assertEqual(identity['error_type'], 'ValueError')
        for field in ('calibration_facts', 'prediction_revision', 'backend_outcomes'):
            self.assertIsNone(identity[field])
        self.assertEqual(identity['request']['supplied_masks']['foreground_reference_mask']['sha256'],
                         sha256(np.ones((4, 5), bool).tobytes()).hexdigest())

    def test_invalid_bindings_reject_before_runner_and_source_replacement_preserves_both_facts(self):
        called = []
        result = runs.run_native_analysis(self.source, analysis_runner=lambda *a, **k: called.append(1),
            learning_root=self.root, reference_archive_source_sha256='0' * 64)
        self.assertEqual(called, [])
        self.assertEqual(result.payload['identity']['state'], 'rejected')
        initial = sha256(self.source.read_bytes()).hexdigest()
        def changed(path, **kwargs):
            path.write_bytes(b'changed during failed calculation')
            raise ValueError('failure after replacement')
        with self.assertLogs('seedvision.export.run_outcomes', level='ERROR'):
            result = runs.run_native_analysis(self.source, learning_root=self.root, analysis_runner=changed)
        identity = result.payload['identity']
        self.assertEqual(identity['state'], 'rejected')
        self.assertEqual(identity['reason_code'], 'source_changed')
        self.assertEqual(identity['source_at_request']['sha256'], initial)
        self.assertNotEqual(identity['source_at_completion'], initial)

    def test_cancellation_and_device_failure_are_distinct_and_process_control_is_not_swallowed(self):
        def runner(error):
            def call(*args, **kwargs):
                raise error
            return call
        cancelled = runs.run_native_analysis(self.source, learning_root=self.root,
            analysis_runner=runner(AnalysisCancelled('stop')))
        self.assertEqual(cancelled.payload['identity']['state'], 'cancelled')
        with self.assertLogs('seedvision.export.run_outcomes', level='ERROR'):
            fatal = runs.run_native_analysis(self.source, learning_root=self.root,
                analysis_runner=runner(torch.cuda.OutOfMemoryError('OOM')))
        self.assertEqual(fatal.payload['identity']['reason_code'], 'device_failure')
        for error in (KeyboardInterrupt(), SystemExit(3)):
            with self.subTest(error=error), self.assertRaises(type(error)):
                runs.run_native_analysis(self.source, learning_root=self.root, analysis_runner=runner(error))
        cancelled_after = runs.run_native_analysis(self.source, learning_root=self.root,
            analysis_runner=lambda *a, **k: object(), cancellation_requested=lambda: True)
        self.assertEqual(cancelled_after.payload['identity']['state'], 'cancelled')

    def test_missing_sources_keep_opaque_selection_identity_and_cannot_collide(self):
        results = []
        for name in ('missing-a.png', 'missing-b.png'):
            with self.assertLogs('seedvision.export.run_outcomes', level='ERROR'):
                results.append(runs.run_native_analysis(self.root / name, learning_root=self.root))
        self.assertNotEqual(results[0].payload['outcome_revision'], results[1].payload['outcome_revision'])
        for value in results:
            self.assertEqual(value.payload['identity']['state'], 'unavailable')
            self.assertIsNone(value.payload['identity']['source_at_request']['sha256'])
            self.assertIsNotNone(value.payload['identity']['source_at_request']['opaque_locator'])

    def test_strict_round_trip_rejects_fabricated_outputs_revision_fields_and_unsafe_shapes(self):
        outcome = self.failed_outcome()
        self.assertEqual(runs.load_run_outcome(json.dumps(outcome.payload)), outcome.payload)
        mutations = [
            lambda p: p.update(version=True),
            lambda p: p.update(outcome_revision='0' * 64),
            lambda p: p['identity'].update(prediction_revision='f' * 64),
            lambda p: p['identity'].update(calibration_facts={}),
            lambda p: p['identity'].update(state=[]),
            lambda p: p['identity'].update(error_type=None),
            lambda p: p['identity']['source_at_request'].update(sha256=None, identity_state='observed'),
            lambda p: p['identity']['request'].update(fabricated_alignment=True),
            lambda p: p['identity']['request'].update(biological_context=[]),
            lambda p: p['identity']['request'].update(supplied_annotations={}),
            lambda p: p['identity']['request'].update(trait_vocabulary=[]),
            lambda p: p['identity']['request'].update(supplied_reference_points=[]),
            lambda p: p['identity']['request'].update(background_colour_enabled=1),
            lambda p: p['identity']['request'].update(enabled_nodes=['duplicate', 'duplicate']),
            lambda p: p['identity']['request']['supplied_masks'].update(foreground_reference_mask={'shape':[True,3], 'dtype':'|b1', 'sha256':'f'*64}),
            lambda p: p['identity']['implementation']['files'].update(**{'export/run_outcomes.py':'bad'}),
        ]
        for mutate in mutations:
            payload = deepcopy(outcome.payload)
            mutate(payload)
            # Recomputing a checksum cannot make structurally false facts valid.
            if payload['outcome_revision'] != '0' * 64:
                payload['outcome_revision'] = sha256(runs._json(payload['identity']).encode()).hexdigest()
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                runs.load_run_outcome(json.dumps(payload))
        for raw in ('{"format":1,"format":2}', 'NaN', 'x' * (runs.LIMIT + 1)):
            with self.assertRaises(ValueError):
                runs.load_run_outcome(raw)
        with self.assertRaisesRegex(ValueError, 'locator'):
            runs.NativeRunOutcome(self.root / 'different.png', outcome.payload_json)

    def test_atomic_publication_and_acknowledgment_never_create_instance_approvals(self):
        outcome = self.failed_outcome()
        output = self.root / 'outputs'
        first = runs.export_run_outcome(output, outcome)
        old = (first / 'run-outcome.json').read_bytes()
        second = runs.export_run_outcome(output, outcome)
        self.assertNotEqual(first, second)
        self.assertEqual((second / 'run-outcome.json').read_bytes(), old)
        self.assertFalse((first / 'results.json').exists())
        ack = runs.acknowledge_run_outcome(self.root / 'ack', outcome, 'reviewer')
        record = json.loads(ack.read_bytes())
        self.assertEqual(record['outcome_revision'], outcome.payload['outcome_revision'])
        self.assertEqual(record['decision'], 'acknowledged')
        self.assertNotIn('decisions', record)
        self.assertFalse((self.root / 'result-reviews').exists())
        with self.assertRaises(ValueError):
            runs.acknowledge_run_outcome(self.root / 'ack', outcome, '')
        with patch('seedvision.export.run_outcomes.os.fsync', side_effect=OSError('publication failed')):
            with self.assertRaises(OSError):
                runs.export_run_outcome(output, outcome)
        self.assertEqual(sorted(output.iterdir()), sorted([first, second]))
        self.assertEqual((first / 'run-outcome.json').read_bytes(), old)

    def test_cancelled_record_keeps_observed_source_and_dependency_changes(self):
        checkpoint = self.root / 'selected.bin'
        checkpoint.write_bytes(b'initial selected bytes')
        from seedvision.learning.pipeline import UNetPipelineSettings
        def runner(path, **kwargs):
            path.write_bytes(b'changed source')
            checkpoint.write_bytes(b'changed selection')
            raise AnalysisCancelled('stop after inputs changed')
        result = runs.run_native_analysis(self.source, learning_root=self.root, analysis_runner=runner,
            unet_settings=UNetPipelineSettings(checkpoint_path=str(checkpoint)))
        facts = result.payload['identity']
        self.assertEqual(facts['state'], 'cancelled')
        self.assertNotEqual(facts['source_at_request']['sha256'], facts['source_at_completion'])
        self.assertEqual(facts['dependency_verification_at_completion'], 'changed')


@unittest.skipUnless(torch.cuda.is_available(), 'Actual terminal run parity requires CUDA')
class NativeRunOutcomeTests(unittest.TestCase):
    setUpClass = classmethod(parity_fixture.AuthoritativeParityTests.setUpClass.__func__)
    setUp = parity_fixture.AuthoritativeParityTests.setUp
    save_project = parity_fixture.AuthoritativeParityTests.save_project
    install_profile = parity_fixture.AuthoritativeParityTests.install_profile
    decode = staticmethod(parity_fixture.AuthoritativeParityTests.decode)

    def failure_routes(self, *, error=ValueError('calibration fixture failure'), cancel=False):
        from seedvision.export.project_runner import run_project
        from seedvision.segmentation.baseline import analyze_path, PipelineAnalysisCache
        from seedvision.ui.main_window import _AnalysisTask
        from PySide6.QtCore import QThreadPool
        captured = []
        def actual(path, **kwargs):
            captured.append((path, kwargs))
            return analyze_path(path, **kwargs)
        with patch('seedvision.export.project_runner.analyze_path', side_effect=actual), \
             patch('seedvision.segmentation.baseline.calibrate_image', side_effect=error), \
             patch('seedvision.export.project_runner.export_project_native_graphs') as graphs, \
             self.assertLogs('seedvision.export.run_outcomes', level='ERROR'):
            destination = run_project(self.root, self.master, self.root / 'reports')[0]
        batch = runs.load_run_outcome((destination / 'run-outcome.json').read_text(encoding='utf-8'))
        self.assertTrue(graphs.called)
        path, kwargs = captured[-1]
        parameters = inspect.signature(_AnalysisTask).parameters
        task_kwargs = {name: kwargs[name] for name in parameters if name in kwargs}
        task_kwargs.update(path=path, background_reference_points=(), foreground_reference_points=(),
            physical_edge_reference_mask=None, non_edge_reference_mask=None, manual_seed_centres=None,
            pipeline_revision=self.window.pipeline.revision, node_cache=PipelineAnalysisCache(), dirty_nodes=frozenset())
        task = _AnalysisTask(**task_kwargs)
        task.setAutoDelete(False)
        completed, terminal, failures, cancelled = [], [], [], []
        task.signals.completed.connect(lambda *args: completed.append(args))
        task.signals.terminal_outcome.connect(lambda outcome, revision: terminal.append((outcome, revision)))
        task.signals.terminal_outcome.connect(self.window._analysis_terminal_outcome)
        task.signals.failed.connect(lambda *args: failures.append(args))
        task.signals.failed.connect(self.window._analysis_failed)
        task.signals.cancelled.connect(lambda *args: cancelled.append(args))
        self.window._active_tasks[self.key] = task
        if cancel:
            task.cancel()
        pool = QThreadPool()
        with patch('seedvision.segmentation.baseline.calibrate_image', side_effect=error), \
             patch('seedvision.ui.main_window.QMessageBox.warning'), \
             self.assertLogs('seedvision.export.run_outcomes', level='ERROR') if not cancel else self.subTest(cancel=True):
            pool.start(task)
            self.assertTrue(pool.waitForDone(30000))
            self.app.processEvents()
        self.assertEqual(completed, [])
        self.assertEqual(len(terminal), 1)
        self.outcome, self.revision = terminal[0]
        self.task = task
        if not cancel:
            self.assertEqual(self.outcome.payload['identity'], batch['identity'])
            self.assertEqual(self.outcome.payload['outcome_revision'], batch['outcome_revision'])
            self.assertEqual(len(failures), 1)
        else:
            self.assertEqual(len(cancelled), 1)
        return batch, destination

    def test_actual_decode_cuda_request_and_qt_batch_failure_parity_with_unknown_outputs(self):
        batch, destination = self.failure_routes()
        identity = batch['identity']
        self.assertEqual(identity['state'], 'failed')
        self.assertEqual(identity['source_at_request']['sha256'], sha256(self.source.read_bytes()).hexdigest())
        self.assertEqual(identity['request']['supplied_reference_transform']['sha256'],
                         sha256(self.reference_transform.tobytes()).hexdigest())
        self.assertIsNone(identity['calibration_facts'])
        self.assertIsNone(identity['prediction_revision'])
        self.assertFalse((destination / 'results.json').exists())
        self.assertIn(self.key, self.window._image_run_outcomes)
        self.assertNotIn(self.key, self.window._active_tasks)
        saved = list((self.root / 'artifacts' / 'run-outcomes').glob('*/run-outcome.json'))
        self.assertEqual(len(saved), 1)
        self.assertEqual(runs.load_run_outcome(saved[0].read_text(encoding='utf-8'))['outcome_revision'], batch['outcome_revision'])

    def test_qt_failure_acknowledgment_cannot_save_prediction_review(self):
        self.failure_routes()
        from PySide6.QtWidgets import QDialog, QPushButton, QLabel
        observed = []
        def exercise(dialog):
            observed.extend(label.text() for label in dialog.findChildren(QLabel))
            buttons = dialog.findChildren(QPushButton)
            self.assertFalse(any(button.text().startswith('Save review decisions') for button in buttons))
            next(button for button in buttons if button.text().startswith('Acknowledge')).click()
            next(button for button in buttons if button.text().startswith('Export terminal')).click()
            return QDialog.DialogCode.Rejected
        with patch.object(QDialog, 'exec', new=exercise), \
             patch('seedvision.ui.results.QFileDialog.getExistingDirectory', return_value=str(self.root / 'ui-export')):
            self.window._review_results()
        self.assertTrue(any('cannot approve seeds' in text for text in observed))
        records = list((self.root / 'artifacts' / 'run-outcome-acknowledgments').glob('*.json'))
        self.assertEqual(len(records), 1)
        self.assertEqual(json.loads(records[0].read_bytes())['decision'], 'acknowledged')
        self.assertFalse((self.root / 'result-reviews').exists())
        exported = list((self.root / 'ui-export').glob('*/run-outcome.json'))
        self.assertEqual(len(exported), 1)
        self.assertEqual(runs.load_run_outcome(exported[0].read_text(encoding='utf-8')), self.outcome.payload)
        self.assertFalse((exported[0].parent / 'results.json').exists())

    def test_batch_rejected_cancelled_and_device_failure_publish_terminal_without_companions(self):
        from contextlib import nullcontext
        from seedvision.export.project_runner import run_project
        errors = [(runs.AnalysisInputIntegrityError('stale reference'), 'rejected', runs.AnalysisInputIntegrityError),
                  (AnalysisCancelled('stop'), 'cancelled', AnalysisCancelled),
                  (torch.cuda.OutOfMemoryError('out of memory'), 'failed', RuntimeError)]
        for index, (error, state, raised) in enumerate(errors):
            output = self.root / f'blocked-output-{index}'
            logs = self.assertLogs('seedvision.export.run_outcomes', level='ERROR') if state == 'failed' else nullcontext()
            with self.subTest(state=state), logs, \
                 patch('seedvision.segmentation.baseline.calibrate_image', side_effect=error), \
                 patch('seedvision.export.project_runner.export_project_native_graphs') as companions:
                with self.assertRaises(raised):
                    run_project(self.root, self.master, output)
            companions.assert_not_called()
            records = list(output.glob('run-outcomes/*/run-outcome.json'))
            self.assertEqual(len(records), 1)
            facts = runs.load_run_outcome(records[0].read_text(encoding='utf-8'))['identity']
            self.assertEqual(facts['state'], state)
            self.assertIsNone(facts['prediction_revision'])
            self.assertEqual(list(output.glob('**/results.json')), [])

    def test_stale_outcome_history_survives_without_removing_replacement_task_or_current_ui(self):
        self.failure_routes()
        old = self.outcome
        self.window._image_run_outcomes.clear()
        self.window.pipeline.set_parameter('trait_classifier', 'trait_abstention_threshold', .91)
        replacement = SimpleNamespace(pipeline_revision=self.window.pipeline.revision, cancellation_requested=False)
        self.window._active_tasks[self.key] = replacement
        self.window._analysis_terminal_outcome(old, self.revision)
        with patch('seedvision.ui.main_window.QMessageBox.warning') as warning:
            self.window._analysis_failed(str(self.source), 'obsolete failure', self.revision)
            self.window._analysis_cancelled(str(self.source), self.revision)
        self.assertIs(self.window._active_tasks[self.key], replacement)
        self.assertNotIn(self.key, self.window._image_run_outcomes)
        self.assertFalse(warning.called)
        self.assertIn(old.payload['outcome_revision'], self.window._run_outcome_history)
        self.window._active_tasks.clear()

    def test_queued_source_replacement_does_not_install_old_terminal_as_current(self):
        self.failure_routes()
        self.window._image_run_outcomes.clear()
        self.source.write_bytes(self.source.read_bytes() + b'changed after terminal signal')
        self.window._analysis_terminal_outcome(self.outcome, self.revision)
        self.assertNotIn(self.key, self.window._image_run_outcomes)
        self.assertIn(self.outcome.payload['outcome_revision'], self.window._run_outcome_history)

    def test_cancelled_worker_emits_terminal_record_and_no_completed_result(self):
        self.failure_routes(cancel=True)
        self.assertEqual(self.outcome.payload['identity']['state'], 'cancelled')
        self.assertIsNone(self.outcome.payload['identity']['prediction_revision'])
        self.window._active_tasks.clear()

    def new_task(self):
        from seedvision.ui.main_window import _AnalysisTask
        from seedvision.pipeline.settings import production_settings
        from seedvision.segmentation.baseline import PipelineAnalysisCache
        settings = production_settings(self.graph)
        mask = (self.labels > 0).copy()
        settings.update(path=self.source, learning_root=self.root, species='soybean',
            background_reference_points=(), foreground_reference_points=(), background_reference_mask=None,
            foreground_reference_mask=mask, background_exclusion_mask=None, foreground_exclusion_mask=None,
            physical_edge_reference_mask=None, non_edge_reference_mask=None, seed_instance_annotations=self.labels.copy(),
            seed_instance_traits=(), seed_trait_species='', seed_trait_coat_patterns=(), seed_trait_conditions=(),
            manual_seed_centres=None, background_colour_enabled=True, enabled_nodes=frozenset({'procedural_instances'}),
            pipeline_revision=self.window.pipeline.revision, node_cache=PipelineAnalysisCache(), dirty_nodes=frozenset(),
            reference_transform=self.reference_transform)
        task = _AnalysisTask(**settings)
        return task, mask

    def test_worker_constructor_freezes_supplied_annotation_arrays_before_dispatch(self):
        task, mask = self.new_task()
        original = sha256(mask.tobytes()).hexdigest()
        mask.fill(False)
        captured = []
        task.signals.terminal_outcome.connect(lambda outcome, revision: captured.append(outcome))
        with patch('seedvision.segmentation.baseline.calibrate_image', side_effect=ValueError('failed after dispatch')), \
             self.assertLogs('seedvision.export.run_outcomes', level='ERROR'):
            task.run()
        self.assertEqual(captured[0].payload['identity']['request']['supplied_masks']['foreground_reference_mask']['sha256'], original)

    def test_cancellation_after_successful_wrapper_return_preserves_exact_request_and_no_completed_result(self):
        from PySide6.QtCore import QThreadPool
        task, mask = self.new_task()
        task.setAutoDelete(False)
        completed, terminal, cancelled = [], [], []
        task.signals.completed.connect(lambda *args: completed.append(args))
        task.signals.terminal_outcome.connect(lambda outcome, revision: terminal.append(outcome))
        task.signals.terminal_outcome.connect(self.window._analysis_terminal_outcome)
        task.signals.cancelled.connect(lambda *args: cancelled.append(args))
        task.signals.cancelled.connect(self.window._analysis_cancelled)
        self.window._active_tasks[self.key] = task
        actual = runs.run_native_analysis
        def cancel_after_return(*args, **kwargs):
            result = actual(*args, **kwargs)
            self.assertNotIsInstance(result, runs.NativeRunOutcome)
            task.cancel()
            return result
        pool = QThreadPool()
        with patch('seedvision.export.run_outcomes.run_native_analysis', side_effect=cancel_after_return):
            pool.start(task)
            self.assertTrue(pool.waitForDone(30000))
            self.app.processEvents()
        self.assertEqual(completed, [])
        self.assertEqual(len(cancelled), 1)
        self.assertEqual(len(terminal), 1)
        facts = terminal[0].payload['identity']
        self.assertEqual(facts['state'], 'cancelled')
        self.assertIsNone(facts['prediction_revision'])
        self.assertEqual(facts['request']['supplied_masks']['foreground_reference_mask']['sha256'],
                         sha256(mask.tobytes()).hexdigest())
        self.assertIn(terminal[0].payload['outcome_revision'], self.window._run_outcome_history)
        self.assertNotIn(self.key, self.window._active_tasks)
        self.assertEqual(len(list((self.root / 'artifacts' / 'run-outcomes').glob('*/run-outcome.json'))), 1)

    def test_queued_same_revision_signals_cannot_interfere_with_a_replacement_task(self):
        from PySide6.QtCore import QObject
        self.failure_routes()
        self.window._image_run_outcomes.clear()
        replacement = SimpleNamespace(pipeline_revision=self.revision,
                                      cancellation_requested=False, signals=QObject())
        self.window._active_tasks[self.key] = replacement
        self.task.signals.cancelled.connect(self.window._analysis_cancelled)
        with patch('seedvision.ui.main_window.QMessageBox.warning') as warning:
            self.task.signals.terminal_outcome.emit(self.outcome, self.revision)
            self.task.signals.failed.emit(str(self.source), 'obsolete same-revision failure', self.revision)
            self.task.signals.cancelled.emit(str(self.source), self.revision)
            self.app.processEvents()
        self.assertFalse(warning.called)
        self.assertIs(self.window._active_tasks[self.key], replacement)
        self.assertNotIn(self.key, self.window._image_run_outcomes)
        self.assertIn(self.outcome.payload['outcome_revision'], self.window._run_outcome_history)
        self.window._active_tasks.clear()
