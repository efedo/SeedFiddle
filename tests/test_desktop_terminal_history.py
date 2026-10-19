"""Earlier desktop rejection boundaries and bounded evidence-only history review."""

from tests.functional_units import functional_units
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


@functional_units
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

    def check_test_loading_old_native_and_preflight_records_does_not_probe_deleted_source_or_implementation(self):
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

    def check_test_identical_copies_deduplicate_but_conflicting_diagnostics_withhold_all_copies(self):
        outcome = self.outcome()
        paths = [runs.export_run_outcome(self.directory, outcome) / 'run-outcome.json' for _ in range(2)]
        self.assertEqual(len(history.load_run_outcome_history(self.directory).outcomes), 1)
        payload = outcome.payload
        payload['diagnostic'] = 'different diagnostic with same semantic revision'
        paths[1].write_text(json.dumps(payload), encoding='utf-8')
        loaded = history.load_run_outcome_history(self.directory)
        self.assertEqual(loaded.outcomes, ())
        self.assertTrue(any('Conflicting terminal records' in note for note in loaded.diagnostics))

    def check_test_malformed_duplicate_key_non_utf8_and_oversized_records_are_diagnostic(self):
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

    def check_test_entry_record_and_byte_budgets_withhold_incomplete_inventory_and_bound_reads(self):
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

    def check_test_incomplete_staging_and_missing_or_unreadable_record_never_fabricate_a_terminal(self):
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

    def check_test_new_nonimage_subjects_and_phase_inventory_round_trip_without_image_authority(self):
        for kind, phase in (('settings_profile', 'desktop_profile_load'), ('annotation', 'desktop_mask_import')):
            with self.subTest(kind=kind):
                outcome = preflight.begin_preflight(self.source, kind, phase).outcome(ValueError('invalid selected artifact'))
                saved = runs.export_run_outcome(self.directory, outcome) / 'run-outcome.json'
                self.assertIsNone(history.load_terminal_record(saved).image_path)
                self.assertEqual(history.load_terminal_record(saved).payload, outcome.payload)
                for name in preflight.UNKNOWN:
                    self.assertIsNone(outcome.payload['identity'][name])

    def check_test_nonregular_and_reparse_records_reject_before_opening_content(self):
        self.directory.mkdir()
        with self.assertRaises(ValueError):
            history.load_terminal_record(self.directory)
        info = SimpleNamespace(st_mode=self.source.stat().st_mode, st_file_attributes=0x400)
        with patch.object(Path, 'lstat', return_value=info), patch.object(Path, 'open') as opened, \
             self.assertRaises(ValueError):
            history.load_terminal_record(self.source)
        opened.assert_not_called()

    def test_workflow_lifecycle(self):
        """3 related TerminalHistoryTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_loading_old_native_and_preflight_records_does_not_probe_deleted_source_or_implementation)
        self.scenario(self.check_test_malformed_duplicate_key_non_utf8_and_oversized_records_are_diagnostic)
        self.scenario(self.check_test_new_nonimage_subjects_and_phase_inventory_round_trip_without_image_authority)

    def test_validation_and_recovery(self):
        """4 related TerminalHistoryTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_identical_copies_deduplicate_but_conflicting_diagnostics_withhold_all_copies)
        self.scenario(self.check_test_entry_record_and_byte_budgets_withhold_incomplete_inventory_and_bound_reads)
        self.scenario(self.check_test_incomplete_staging_and_missing_or_unreadable_record_never_fabricate_a_terminal)
        self.scenario(self.check_test_nonregular_and_reparse_records_reject_before_opening_content)


@unittest.skipUnless(torch.cuda.is_available(), 'Native desktop boundary fixtures require CUDA')
@functional_units
class DesktopTerminalHistoryTests(unittest.TestCase):
    setUpClass = classmethod(native_fixture.NativePreflightOutcomeTests.setUpClass.__func__)
    setUp = native_fixture.NativePreflightOutcomeTests.setUp
    save_project = native_fixture.NativePreflightOutcomeTests.save_project
    install_profile = native_fixture.NativePreflightOutcomeTests.install_profile
    decode = staticmethod(native_fixture.NativePreflightOutcomeTests.decode)

    def records(self):
        return [runs.load_run_outcome(path.read_text(encoding='utf-8'))
                for path in (self.root / 'artifacts' / 'run-outcomes').glob('*/run-outcome.json')]




    def check_test_project_apply_failure_keeps_existing_image_and_drafts_before_session_clear(self):
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


    def check_test_actual_profile_apply_after_mutation_rolls_back_graph_without_recompute(self):
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


    def check_test_reopened_disk_history_ack_export_keeps_current_analysis_drafts_and_reviews_unchanged(self):
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


    def test_validation_and_recovery(self):
        """2 related DesktopTerminalHistoryTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_project_apply_failure_keeps_existing_image_and_drafts_before_session_clear)
        self.scenario(self.check_test_actual_profile_apply_after_mutation_rolls_back_graph_without_recompute)

    def test_workflow_lifecycle(self):
        """1 related DesktopTerminalHistoryTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_reopened_disk_history_ack_export_keeps_current_analysis_drafts_and_reviews_unchanged)
