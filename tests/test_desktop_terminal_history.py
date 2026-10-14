"""Earlier desktop rejection boundaries and bounded evidence-only history review."""
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch

from seedvision.export import preflight_outcomes as preflight
from seedvision.export import run_outcomes as runs
from seedvision.export import run_outcome_history as history
import tests.test_preflight_outcomes as native_fixture


class TerminalHistoryTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / 'selected.png'
        self.source.write_bytes(b'actual historical source bytes')
        self.directory = self.root / 'history'

    def outcome(self):
        return preflight.begin_preflight(self.source, 'image', 'desktop_source').outcome(ValueError('source rejected'))

    def test_loading_old_native_and_preflight_records_does_not_probe_deleted_source_or_implementation(self):
        first = self.outcome()
        def failed(*args, **kwargs):
            raise ValueError('computation failure')
        with self.assertLogs('seedvision.export.run_outcomes', level='ERROR'):
            second = runs.run_native_analysis(self.source, learning_root=self.root, analysis_runner=failed)
        paths = [runs.export_run_outcome(self.directory, value) / 'run-outcome.json' for value in (first, second)]
        self.source.unlink()
        with patch.object(runs, '_source_digest', side_effect=AssertionError('historical paths must not be probed')):
            loaded = history.load_run_outcome_history(self.directory)
            detached = [history.load_terminal_record(path) for path in paths]
        self.assertEqual(len(loaded.outcomes), 2)
        self.assertEqual(loaded.diagnostics, ())
        self.assertEqual([value.payload for value in detached], [first.payload, second.payload])
        self.assertEqual({value.payload['outcome_revision'] for value in loaded.outcomes},
                         {first.payload['outcome_revision'], second.payload['outcome_revision']})

    def test_identical_copies_deduplicate_but_conflicting_diagnostics_withhold_all_copies(self):
        outcome = self.outcome()
        paths = [runs.export_run_outcome(self.directory, outcome) / 'run-outcome.json' for _ in range(2)]
        self.assertEqual(len(history.load_run_outcome_history(self.directory).outcomes), 1)
        payload = outcome.payload
        payload['diagnostic'] = 'different diagnostic with same semantic revision'
        paths[1].write_text(json.dumps(payload), encoding='utf-8')
        loaded = history.load_run_outcome_history(self.directory)
        self.assertEqual(loaded.outcomes, ())
        self.assertTrue(any('Conflicting terminal records' in note for note in loaded.diagnostics))

    def test_malformed_duplicate_key_non_utf8_and_oversized_records_are_diagnostic(self):
        payloads = (b'{bad JSON', b'{"format":1,"format":2}', b'\xff', b'x' * (runs.LIMIT + 1))
        for index, raw in enumerate(payloads):
            directory = self.directory / f'invalid-{index}'
            directory.mkdir(parents=True)
            (directory / 'run-outcome.json').write_bytes(raw)
        loaded = history.load_run_outcome_history(self.directory)
        self.assertEqual(loaded.outcomes, ())
        self.assertEqual(len(loaded.diagnostics), 4)
        self.assertFalse(loaded.incomplete)
        for directory in self.directory.iterdir():
            with self.assertRaises(ValueError):
                history.load_terminal_record(directory / 'run-outcome.json')

    def test_entry_record_and_byte_budgets_withhold_incomplete_inventory_and_bound_reads(self):
        for _ in range(3):
            runs.export_run_outcome(self.directory, self.outcome())
        for kwargs in ({'entry_limit': 1}, {'record_limit': 2}, {'byte_limit': 10}):
            with self.subTest(kwargs=kwargs):
                loaded = history.load_run_outcome_history(self.directory, **kwargs)
                self.assertTrue(loaded.incomplete)
                self.assertEqual(loaded.outcomes, ())
                if 'byte_limit' in kwargs:
                    self.assertLessEqual(loaded.bytes_read, 11)
                if 'entry_limit' in kwargs:
                    self.assertEqual(loaded.entries_examined, 2)
        for value in (0, -1, True):
            with self.assertRaises(ValueError):
                history.load_run_outcome_history(self.directory, entry_limit=value)

    def test_incomplete_staging_and_missing_or_unreadable_record_never_fabricate_a_terminal(self):
        self.directory.mkdir()
        (self.directory / '.run-outcome-live.incomplete').mkdir()
        (self.directory / 'missing-record').mkdir()
        loaded = history.load_run_outcome_history(self.directory)
        self.assertEqual(loaded.outcomes, ())
        self.assertEqual(len(loaded.diagnostics), 1)
        with patch.object(Path, 'open', side_effect=PermissionError('record cannot be read')):
            loaded = history.load_run_outcome_history(self.directory)
        self.assertEqual(loaded.outcomes, ())
        self.assertTrue(loaded.diagnostics)
        self.assertEqual(history.load_run_outcome_history(self.root / 'absent-history').outcomes, ())

    def test_new_nonimage_subjects_and_phase_inventory_round_trip_without_image_authority(self):
        for kind, phase in (('settings_profile', 'desktop_profile_load'), ('annotation', 'desktop_mask_import')):
            with self.subTest(kind=kind):
                outcome = preflight.begin_preflight(self.source, kind, phase).outcome(ValueError('invalid selected artifact'))
                saved = runs.export_run_outcome(self.directory, outcome) / 'run-outcome.json'
                self.assertIsNone(history.load_terminal_record(saved).image_path)
                self.assertEqual(history.load_terminal_record(saved).payload, outcome.payload)
                for name in preflight.UNKNOWN:
                    self.assertIsNone(outcome.payload['identity'][name])

    def test_nonregular_and_reparse_records_reject_before_opening_content(self):
        self.directory.mkdir()
        with self.assertRaises(ValueError):
            history.load_terminal_record(self.directory)
        info = SimpleNamespace(st_mode=self.source.stat().st_mode, st_file_attributes=0x400)
        with patch.object(Path, 'lstat', return_value=info), patch.object(Path, 'open') as opened, \
             self.assertRaises(ValueError):
            history.load_terminal_record(self.source)
        opened.assert_not_called()


@unittest.skipUnless(torch.cuda.is_available(), 'Native desktop boundary fixtures require CUDA')
class DesktopTerminalHistoryTests(unittest.TestCase):
    setUpClass = classmethod(native_fixture.NativePreflightOutcomeTests.setUpClass.__func__)
    setUp = native_fixture.NativePreflightOutcomeTests.setUp
    save_project = native_fixture.NativePreflightOutcomeTests.save_project
    install_profile = native_fixture.NativePreflightOutcomeTests.install_profile
    decode = staticmethod(native_fixture.NativePreflightOutcomeTests.decode)

    def records(self):
        return [runs.load_run_outcome(path.read_text(encoding='utf-8'))
                for path in (self.root / 'artifacts' / 'run-outcomes').glob('*/run-outcome.json')]

    def test_actual_changed_source_rejection_preserves_drafts_warns_once_and_blocks_runner(self):
        self.window.image_view._image_path = self.source
        self.window._source_fingerprints[self.key] = sha256(self.source.read_bytes()).hexdigest()
        draft = self.labels.copy()
        self.window._draft_instance_annotations[self.key] = draft
        self.window._analyses[self.key] = object()
        self.source.write_bytes(self.source.read_bytes() + b'changed externally')
        with patch('seedvision.ui.main_window.QMessageBox.warning') as warning, \
             patch('seedvision.ui.main_window._AnalysisTask') as task:
            self.window._analyze_current_image()
        warning.assert_called_once()
        task.assert_not_called()
        self.assertIs(self.window._draft_instance_annotations[self.key], draft)
        self.assertNotIn(self.key, self.window._analyses)
        payload, = self.records()
        self.assertEqual(payload['identity']['phase'], 'desktop_source')
        self.assertEqual(payload['identity']['state'], 'rejected')
        self.assertEqual(payload['identity']['subject_at_request']['sha256'], sha256(self.source.read_bytes()).hexdigest())
        self.assertIn(self.key, self.window._image_run_outcomes)

    def test_actual_missing_source_preserves_drafts_and_replacement_task_with_unknown_observation(self):
        draft = self.labels.copy()
        self.window._draft_instance_annotations[self.key] = draft
        replacement = SimpleNamespace(pipeline_revision=self.window.pipeline.revision)
        self.window._active_tasks[self.key] = replacement
        self.source.unlink()
        with patch('seedvision.ui.reference_frames.QMessageBox.warning') as warning:
            self.assertFalse(self.window._check_source_revision(self.source))
        warning.assert_called_once()
        self.assertIs(self.window._active_tasks[self.key], replacement)
        self.assertIs(self.window._draft_instance_annotations[self.key], draft)
        self.assertNotIn(self.key, self.window._image_run_outcomes)
        payload, = self.records()
        self.assertEqual(payload['identity']['state'], 'unavailable')
        self.assertIsNone(payload['identity']['subject_at_request']['sha256'])
        self.assertEqual(payload['identity']['subject_verification_at_completion'], 'unavailable')
        self.window._active_tasks.clear()

    def test_real_invalid_project_and_profile_keep_current_image_graph_and_drafts(self):
        draft = self.labels.copy()
        self.window._draft_instance_annotations[self.key] = draft
        self.window.image_view._image_path = self.source
        original_project = self.window._current_project_path
        original_revision = self.window.pipeline.revision
        for kind, name, method, phase in (('project', 'invalid.seedfiddle-project.json', self.window._open_project, 'desktop_project_load'),
                ('settings_profile', 'invalid.seedfiddle-settings.json', self.window._load_analysis_settings_profile, 'desktop_profile_load')):
            path = self.root / name
            path.write_text('{malformed JSON', encoding='utf-8')
            with patch('seedvision.ui.main_window.QMessageBox.critical') as warning, \
                 patch.object(self.window, '_confirm_project_replacement') as confirm, \
                 patch.object(self.window, '_analyze_current_image') as analyze:
                self.assertFalse(method(path))
            warning.assert_called_once()
            confirm.assert_not_called()
            analyze.assert_not_called()
            self.assertIs(self.window._draft_instance_annotations[self.key], draft)
            self.assertEqual(self.window.image_view.image_path, self.source)
            self.assertEqual(self.window.pipeline.revision, original_revision)
            self.assertEqual(self.window._current_project_path, original_project)
            payload = next(row for row in self.records() if row['identity']['subject_kind'] == kind)
            self.assertEqual(payload['identity']['phase'], phase)
            self.assertEqual(payload['identity']['subject_at_request']['sha256'], sha256(path.read_bytes()).hexdigest())
            self.assertIsNone(payload['identity']['image_facts'])

    def test_project_apply_failure_keeps_existing_image_and_drafts_before_session_clear(self):
        from seedvision.persistence.analysis_settings import AnalysisSettingsError
        from seedvision.ui.main_window import apply_analysis_settings_profile
        draft = self.labels.copy()
        self.window._draft_instance_annotations[self.key] = draft
        self.window.image_view._image_path = self.source
        def apply(graph, profile):
            if graph is self.window.pipeline:
                raise AnalysisSettingsError('current graph application rejected')
            return apply_analysis_settings_profile(graph, profile)
        with patch('seedvision.ui.main_window.apply_analysis_settings_profile', side_effect=apply), \
             patch.object(self.window, '_confirm_project_replacement', return_value=True), \
             patch('seedvision.ui.main_window.QMessageBox.critical') as warning:
            self.assertFalse(self.window._open_project(self.master))
        warning.assert_called_once()
        self.assertIs(self.window._draft_instance_annotations[self.key], draft)
        self.assertEqual(self.window.image_view.image_path, self.source)
        self.assertFalse(self.window._installing_project)
        payload, = self.records()
        self.assertEqual(payload['identity']['phase'], 'desktop_project_install')

    def test_actual_project_apply_after_mutation_rolls_back_graph_and_keeps_current_session(self):
        original = self.window.pipeline.node('ruler_detection').parameters['ruler_length_mm']
        proposed = original + 7.
        self.graph.set_parameter('ruler_detection', 'ruler_length_mm', proposed)
        self.save_project()
        draft = self.labels.copy()
        self.window._draft_instance_annotations[self.key] = draft
        self.window.image_view._image_path = self.source
        revision = self.window.pipeline.revision
        observed = []
        def reject_mutation():
            observed.append(self.window.pipeline.node('ruler_detection').parameters['ruler_length_mm'])
            raise RuntimeError('invariant rejects installed parameters')
        with patch.object(self.window.pipeline, '_validate_connections', side_effect=reject_mutation), \
             patch.object(self.window, '_confirm_project_replacement', return_value=True), \
             patch('seedvision.ui.main_window.QMessageBox.critical'):
            self.assertFalse(self.window._open_project(self.master))
        self.assertEqual(observed, [proposed])
        self.assertEqual(self.window.pipeline.node('ruler_detection').parameters['ruler_length_mm'], original)
        self.assertEqual(self.window.pipeline.revision, revision)
        self.assertIs(self.window._draft_instance_annotations[self.key], draft)
        self.assertEqual(self.window.image_view.image_path, self.source)
        payload, = self.records()
        self.assertEqual(payload['identity']['phase'], 'desktop_project_install')
        self.assertEqual(payload['identity']['state'], 'failed')

    def test_actual_profile_apply_after_mutation_rolls_back_graph_without_recompute(self):
        from seedvision.persistence.analysis_settings import save_analysis_settings_profile
        from PySide6.QtWidgets import QMessageBox
        original = self.window.pipeline.node('ruler_detection').parameters['ruler_length_mm']
        proposed = original + 9.
        self.graph.set_parameter('ruler_detection', 'ruler_length_mm', proposed)
        selected = save_analysis_settings_profile(self.root / 'valid.seedfiddle-settings.json', self.graph)
        revision = self.window.pipeline.revision
        observed = []
        def reject_mutation():
            observed.append(self.window.pipeline.node('ruler_detection').parameters['ruler_length_mm'])
            raise RuntimeError('profile invariant failure after mutation')
        with patch.object(self.window.pipeline, '_validate_connections', side_effect=reject_mutation), \
             patch('seedvision.ui.main_window.QMessageBox.question', return_value=QMessageBox.StandardButton.Yes), \
             patch('seedvision.ui.main_window.QMessageBox.critical'), \
             patch.object(self.window, '_analyze_current_image') as analyze:
            self.assertFalse(self.window._load_analysis_settings_profile(selected))
        analyze.assert_not_called()
        self.assertEqual(observed, [proposed])
        self.assertEqual(self.window.pipeline.node('ruler_detection').parameters['ruler_length_mm'], original)
        self.assertEqual(self.window.pipeline.revision, revision)
        payload, = self.records()
        self.assertEqual(payload['identity']['phase'], 'desktop_profile_install')
        self.assertEqual(payload['identity']['subject_at_request']['sha256'], sha256(selected.read_bytes()).hexdigest())

    def test_actual_invalid_explicit_mask_preserves_labels_generation_and_existing_analysis(self):
        self.window.image_view._image_path = self.source
        original = SimpleNamespace(calibration=SimpleNamespace(corrected_bgr=self.image))
        self.window._analyses[self.key] = original
        self.window._draft_instance_annotations[self.key] = self.labels.copy()
        self.window._draft_instance_generation_sources[self.key] = 'a' * 64
        selected = self.root / 'wrong-sized-labels.png'
        self.assertTrue(cv2.imwrite(str(selected), np.zeros((3, 4), np.uint16)))
        with patch('seedvision.ui.main_window.QFileDialog.getOpenFileName', return_value=(str(selected), '')), \
             patch('seedvision.ui.main_window.QMessageBox.critical') as warning:
            self.window._choose_instance_mask_file()
        warning.assert_called_once()
        self.assertIs(self.window._analyses[self.key], original)
        np.testing.assert_array_equal(self.window._draft_instance_annotations[self.key], self.labels)
        self.assertEqual(self.window._draft_instance_generation_sources[self.key], 'a' * 64)
        payload, = self.records()
        self.assertEqual(payload['identity']['subject_kind'], 'annotation')
        self.assertEqual(payload['identity']['phase'], 'desktop_mask_import')
        self.assertEqual(payload['identity']['subject_at_request']['sha256'], sha256(selected.read_bytes()).hexdigest())
        self.assertNotIn(self.key, self.window._image_run_outcomes)

    def test_reopened_disk_history_ack_export_keeps_current_analysis_drafts_and_reviews_unchanged(self):
        from PySide6.QtWidgets import QDialog, QTableWidget, QPushButton, QLabel
        outcome = preflight.begin_preflight(self.master, 'project', 'desktop_project_load').outcome(ValueError('historical rejected project'))
        saved = runs.export_run_outcome(self.root / 'artifacts' / 'run-outcomes', outcome) / 'run-outcome.json'
        old = saved.read_bytes()
        self.window._run_outcome_history.clear()
        current = object()
        self.window._analyses[self.key] = current
        draft = self.labels.copy()
        self.window._draft_instance_annotations[self.key] = draft
        observed = []
        def exercise(dialog):
            observed.extend(label.text() for label in dialog.findChildren(QLabel))
            if dialog.windowTitle() == 'Terminal run history':
                table = dialog.findChild(QTableWidget)
                self.assertEqual(table.rowCount(), 1)
                self.assertEqual(table.item(0, 0).text(), 'project')
                table.selectRow(0)
                next(b for b in dialog.findChildren(QPushButton) if b.text().startswith('Review selected')).click()
            else:
                next(b for b in dialog.findChildren(QPushButton) if b.text().startswith('Acknowledge')).click()
                next(b for b in dialog.findChildren(QPushButton) if b.text().startswith('Export terminal')).click()
            return QDialog.DialogCode.Rejected
        with patch.object(QDialog, 'exec', new=exercise), \
             patch('seedvision.ui.results.QFileDialog.getExistingDirectory', return_value=str(self.root / 'history-export')):
            self.window.terminal_history_action.trigger()
        self.assertIs(self.window._analyses[self.key], current)
        self.assertIs(self.window._draft_instance_annotations[self.key], draft)
        self.assertEqual(self.window._image_run_outcomes, {})
        self.assertEqual(self.window._run_outcome_history, {})
        self.assertFalse((self.root / 'result-reviews').exists())
        self.assertEqual(saved.read_bytes(), old)
        self.assertEqual(self.window._selected_history_outcome.payload, outcome.payload)
        self.assertTrue(any('observations made during earlier attempts' in note for note in observed))
        acknowledgment, = (self.root / 'artifacts' / 'run-outcome-acknowledgments').glob('*.json')
        self.assertEqual(json.loads(acknowledgment.read_bytes())['outcome_revision'], outcome.payload['outcome_revision'])
        exported, = (self.root / 'history-export').glob('*/run-outcome.json')
        self.assertEqual(history.load_terminal_record(exported).payload, outcome.payload)

    def test_explicit_historical_file_load_does_not_install_missing_source_as_current(self):
        outcome = preflight.begin_preflight(self.source, 'image', 'desktop_source').outcome(ValueError('historic source rejection'))
        saved = runs.export_run_outcome(self.root / 'explicit-history', outcome) / 'run-outcome.json'
        self.source.unlink()
        with patch('seedvision.ui.results.QFileDialog.getOpenFileName', return_value=(str(saved), '')), \
             patch.object(self.window, '_review_run_outcome') as review:
            self.window._open_terminal_record()
        review.assert_called_once()
        self.assertEqual(review.call_args.args[0].payload, outcome.payload)
        self.assertEqual(self.window._image_run_outcomes, {})
        self.assertEqual(self.window._run_outcome_history, {})
