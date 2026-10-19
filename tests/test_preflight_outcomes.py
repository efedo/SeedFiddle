"""Incomplete request evidence through real batch and Qt preparation boundaries."""

from tests.functional_units import functional_units
from copy import deepcopy
from hashlib import sha256
import json
import traceback
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from seedvision.export import preflight_outcomes as preflight
from seedvision.export import run_outcomes as runs
from seedvision.timing import AnalysisCancelled
import tests.test_run_outcomes as native_fixture


@functional_units
class PreflightOutcomeTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / 'selected.png'
        self.source.write_bytes(b'actual selected file bytes')

    def outcome(self, *, path=None, kind='image', phase='request_capture', error=None):
        return preflight.begin_preflight(path or self.source, kind, phase).outcome(
            error or ValueError('invalid supplied request'))

    def check_test_invalid_settings_and_duplicate_positional_arguments_never_execute_analysis(self):
        for args, kwargs in [((), {'settings': object()}), ((), {'settings': 0}),
                             ((object(),), {'settings': object()})]:
            runner = Mock()
            outcome = runs.run_native_analysis(self.source, *args, analysis_runner=runner,
                learning_root=self.root, **kwargs)
            runner.assert_not_called()
            self.assertIsInstance(outcome, preflight.PreflightRunOutcome)
            self.assertEqual(outcome.payload['identity']['phase'], 'request_capture')
            self.assertEqual(outcome.payload['identity']['state'], 'rejected')
            self.assertEqual(outcome.payload['identity']['subject_at_request']['sha256'],
                             sha256(self.source.read_bytes()).hexdigest())
            for name in preflight.UNKNOWN:
                self.assertIsNone(outcome.payload['identity'][name])

    def check_test_malformed_raster_and_reference_facts_are_rejected_before_runner(self):
        for kwargs in ({'foreground_reference_mask': np.ones(7, bool)},
                       {'reference_transform': np.ones(3)},
                       {'reference_archive_source_sha256': 'invalid'},
                       {'enabled_nodes': ['duplicate', 'duplicate']}):
            runner = Mock()
            outcome = runs.run_native_analysis(self.source, analysis_runner=runner,
                learning_root=self.root, **kwargs)
            runner.assert_not_called()
            self.assertIsInstance(outcome, preflight.PreflightRunOutcome)
            self.assertEqual(outcome.payload['identity']['state'], 'rejected')

    def check_test_capture_changes_preserve_actual_before_and_after_without_inventing_request(self):
        initial = sha256(self.source.read_bytes()).hexdigest()
        def changed(*args):
            self.source.write_bytes(b'changed before request could be captured')
            raise ValueError('invalid capture')
        with patch.object(runs, '_request', side_effect=changed):
            outcome = runs.run_native_analysis(self.source, analysis_runner=Mock())
        facts = outcome.payload['identity']
        self.assertEqual(facts['subject_at_request']['sha256'], initial)
        self.assertEqual(facts['subject_at_completion']['sha256'], sha256(self.source.read_bytes()).hexdigest())
        self.assertEqual(facts['subject_verification_at_completion'], 'changed')
        self.assertIsNone(facts['request_facts'])

    def check_test_unavailable_subject_observations_never_become_verified_even_when_both_unknown(self):
        missing = self.root / 'missing.png'
        outcome = self.outcome(path=missing)
        self.assertEqual(outcome.payload['identity']['subject_verification_at_completion'], 'unavailable')
        self.assertIsNone(outcome.payload['identity']['subject_at_request']['sha256'])
        attempt = preflight.begin_preflight(self.source, 'image', 'desktop_prepare')
        self.source.unlink()
        facts = attempt.outcome(FileNotFoundError('selected file disappeared')).payload['identity']
        self.assertEqual(facts['state'], 'unavailable')
        self.assertIsNone(facts['subject_at_completion']['sha256'])
        self.assertEqual(facts['subject_verification_at_completion'], 'unavailable')

    def check_test_exception_categories_cancellation_and_process_control_remain_distinct(self):
        for error, state in ((AnalysisCancelled('stop'), 'cancelled'), (RuntimeError('unexpected'), 'failed'),
                             (OSError('cannot read'), 'unavailable'), (TypeError('invalid'), 'rejected')):
            with self.subTest(state=state), patch.object(runs, '_request', side_effect=error):
                outcome = runs.run_native_analysis(self.source, analysis_runner=Mock())
                self.assertEqual(outcome.payload['identity']['state'], state)
                self.assertEqual(outcome.payload['identity']['error_type'], type(error).__name__)
        for error in (KeyboardInterrupt(), SystemExit(2)):
            with patch.object(runs, '_request', side_effect=error), self.assertRaises(type(error)):
                runs.run_native_analysis(self.source)
        wrapped = ValueError('public load exception')
        wrapped.__cause__ = OSError('actual I/O cause')
        self.assertEqual(self.outcome(error=wrapped).payload['identity']['state'], 'unavailable')

    def check_test_strict_loader_rejects_forged_captured_outputs_subjects_and_attempts(self):
        outcome = self.outcome()
        self.assertEqual(runs.load_run_outcome(outcome.payload_json), outcome.payload)
        mutations = [lambda p: p.update(version=True),
            lambda p: p['identity'].update(attempt_id='bad'),
            lambda p: p['identity'].update(attempt_id=[]),
            lambda p: p['identity'].update(subject_kind=[]),
            lambda p: p['identity'].update(phase='project_load'),
            lambda p: p['identity'].update(state=[]),
            lambda p: p['identity'].update(reason_code='analysis_exception'),
            lambda p: p['identity'].update(error_type=None),
            lambda p: p['identity'].update(subject_verification_at_completion='changed'),
            lambda p: p['identity']['subject_at_request'].update(identity_state='unavailable'),
            lambda p: p['identity']['subject_at_completion'].update(sha256=None),
            lambda p: p['execution'].update(subject_path=[]),
            lambda p: p['identity']['implementation']['files'].update(**{'ui/main_window.py':'bad'}),
            *[lambda p, name=name: p['identity'].update({name: {}}) for name in preflight.UNKNOWN]]
        for mutate in mutations:
            payload = deepcopy(outcome.payload)
            mutate(payload)
            payload['outcome_revision'] = sha256(runs._json(payload['identity']).encode()).hexdigest()
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                runs.load_run_outcome(json.dumps(payload))
        with self.assertRaises(ValueError):
            preflight.PreflightRunOutcome(self.root / 'other.png', outcome.payload_json)
        with self.assertRaises(ValueError):
            runs.NativeRunOutcome(self.source, outcome.payload_json)
        for raw in ('{"format":1,"format":2}', 'NaN', 'x' * (runs.LIMIT + 1)):
            with self.assertRaises(ValueError):
                runs.load_run_outcome(raw)

    def check_test_distinct_incomplete_attempts_and_project_subjects_cannot_transfer_acknowledgment(self):
        first, second = self.outcome(), self.outcome()
        self.assertNotEqual(first.payload['outcome_revision'], second.payload['outcome_revision'])
        saved = runs.acknowledge_run_outcome(self.root / 'ack', first, 'operator')
        self.assertNotEqual(json.loads(saved.read_bytes())['outcome_revision'], second.payload['outcome_revision'])
        project = self.outcome(kind='project', phase='project_load')
        self.assertIsNone(project.image_path)
        self.assertIsNone(project.payload['identity']['image_facts'])
        self.assertEqual(runs.load_run_outcome(project.payload_json), project.payload)

    def check_test_atomic_preflight_publication_failure_preserves_previous_records_and_acknowledgments(self):
        outcome = self.outcome()
        directory = self.root / 'records'
        first = runs.export_run_outcome(directory, outcome)
        old = (first / 'run-outcome.json').read_bytes()
        ack = runs.acknowledge_run_outcome(self.root / 'ack', outcome, 'operator')
        old_ack = ack.read_bytes()
        with patch.object(Path, 'rename', side_effect=OSError('publish failure')), self.assertRaises(OSError):
            runs.export_run_outcome(directory, self.outcome())
        self.assertEqual((first / 'run-outcome.json').read_bytes(), old)
        self.assertEqual(ack.read_bytes(), old_ack)
        self.assertEqual(list(directory.glob('.run-outcome-*.incomplete')), [])
        self.assertEqual(list(directory.glob('*/results.json')), [])

    def check_test_recording_failure_keeps_original_exception_identity_traceback_and_secondary_note(self):
        error = ValueError('original preflight failure')
        directory = self.root / 'records'
        prior = runs.export_run_outcome(directory, self.outcome())
        old = (prior / 'run-outcome.json').read_bytes()
        def fail_preparation():
            raise error
        with patch.object(Path, 'rename', side_effect=OSError('disk publication failed')), \
             self.assertLogs('seedvision.export.preflight_outcomes', level='ERROR'):
            try:
                with preflight.record_preflight(self.source, 'image', 'image_references', directory):
                    fail_preparation()
            except ValueError as caught:
                self.assertIs(caught, error)
                self.assertEqual(traceback.extract_tb(caught.__traceback__)[-1].name, 'fail_preparation')
            else:
                self.fail('The original preflight failure was swallowed.')
        self.assertTrue(any('OSError: disk publication failed' in note for note in error.__notes__))
        self.assertEqual((prior / 'run-outcome.json').read_bytes(), old)
        self.assertEqual(list(directory.glob('.run-outcome-*.incomplete')), [])

    def check_test_real_malformed_project_and_invalid_arguments_publish_project_only_evidence(self):
        from seedvision.export.project_runner import run_project
        project = self.root / 'malformed.seedfiddle-project.json'
        project.write_text('{broken JSON', encoding='utf-8')
        for index, padding, phase in ((0, None, 'project_load'), (1, -1, 'project_arguments')):
            output = self.root / f'batch-{index}'
            with patch('seedvision.export.project_runner.analyze_path') as runner, \
                 patch('seedvision.export.project_runner.export_project_native_graphs') as companion, \
                 self.assertRaises(ValueError) as caught:
                run_project(self.root, project, output, crop_padding_px=padding)
            runner.assert_not_called()
            companion.assert_not_called()
            path, = output.glob('run-outcomes/*/run-outcome.json')
            payload = runs.load_run_outcome(path.read_text(encoding='utf-8'))
            self.assertEqual(payload['identity']['subject_kind'], 'project')
            self.assertEqual(payload['identity']['phase'], phase)
            self.assertEqual(payload['identity']['subject_at_request']['sha256'], sha256(project.read_bytes()).hexdigest())
            self.assertIsNone(payload['identity']['image_facts'])
            self.assertTrue(any('Terminal record:' in note for note in caught.exception.__notes__))

    def test_validation_and_recovery(self):
        """8 related PreflightOutcomeTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_invalid_settings_and_duplicate_positional_arguments_never_execute_analysis)
        self.scenario(self.check_test_malformed_raster_and_reference_facts_are_rejected_before_runner)
        self.scenario(self.check_test_unavailable_subject_observations_never_become_verified_even_when_both_unknown)
        self.scenario(self.check_test_exception_categories_cancellation_and_process_control_remain_distinct)
        self.scenario(self.check_test_strict_loader_rejects_forged_captured_outputs_subjects_and_attempts)
        self.scenario(self.check_test_atomic_preflight_publication_failure_preserves_previous_records_and_acknowledgments)
        self.scenario(self.check_test_recording_failure_keeps_original_exception_identity_traceback_and_secondary_note)
        self.scenario(self.check_test_real_malformed_project_and_invalid_arguments_publish_project_only_evidence)

    def test_workflow_lifecycle(self):
        """2 related PreflightOutcomeTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_capture_changes_preserve_actual_before_and_after_without_inventing_request)
        self.scenario(self.check_test_distinct_incomplete_attempts_and_project_subjects_cannot_transfer_acknowledgment)


@unittest.skipUnless(torch.cuda.is_available(), 'Native boundary fixtures require CUDA')
class NativePreflightOutcomeTests(unittest.TestCase):
    setUpClass = classmethod(native_fixture.NativeRunOutcomeTests.setUpClass.__func__)
    setUp = native_fixture.NativeRunOutcomeTests.setUp
    save_project = native_fixture.NativeRunOutcomeTests.save_project
    install_profile = native_fixture.NativeRunOutcomeTests.install_profile
    decode = staticmethod(native_fixture.NativeRunOutcomeTests.decode)
    new_task = native_fixture.NativeRunOutcomeTests.new_task



    def test_project_subject_and_stale_pretask_attempt_preserve_history_without_image_cleanup(self):
        project = preflight.begin_preflight(self.master, 'project', 'project_load').outcome(ValueError('invalid project'))
        outcome = preflight.begin_preflight(self.source, 'image', 'desktop_prepare').outcome(ValueError('stale preparation'))
        replacement = SimpleNamespace(pipeline_revision=self.window.pipeline.revision)
        self.window._active_tasks[self.key] = replacement
        with patch.object(self.window, '_finish_analysis_failure') as cleanup:
            self.window._analysis_terminal_outcome(project, self.window.pipeline.revision)
            self.window._analysis_preflight_failed(project, self.window.pipeline.revision)
            self.window._analysis_preflight_failed(outcome, self.window.pipeline.revision)
        cleanup.assert_not_called()
        self.assertIs(self.window._active_tasks[self.key], replacement)
        self.assertNotIn(self.key, self.window._image_run_outcomes)
        self.assertIn(project.payload['outcome_revision'], self.window._run_outcome_history)
        self.assertIn(outcome.payload['outcome_revision'], self.window._run_outcome_history)
        self.window._active_tasks.clear()
