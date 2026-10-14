"""Real terminal/acknowledgment transport and private Qt historical consumers."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch

from seedvision.export import preflight_outcomes as preflight, run_outcomes as runs
from seedvision.persistence import (
    ProjectAnalysisStore, ProjectImageSpec, analysis_settings_profile_from_graph,
)
from seedvision.persistence import snapshot as snapshots, terminal_history as portable
from seedvision.pipeline import build_default_pipeline
import tests.test_preflight_outcomes as desktop_fixture


class TerminalSnapshotTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.root = self.base / 'original'
        self.root.mkdir()
        source = self.root / 'live.png'
        self.assertTrue(cv2.imwrite(str(source), np.zeros((16, 16, 3), np.uint8)))
        self.master = ProjectAnalysisStore(self.root).capture_and_save(
            analysis_settings=analysis_settings_profile_from_graph(build_default_pipeline()),
            images=(ProjectImageSpec(source, (16, 16)),), destination=self.root / 'test.seedfiddle-project.json')
        self.historical = self.root / 'historical.png'
        self.historical.write_bytes(b'earlier observed bytes')
        self.outcome = preflight.begin_preflight(self.historical, 'image', 'desktop_source').outcome(ValueError('earlier rejection'))
        self.saved = runs.export_run_outcome(self.root / 'artifacts' / 'run-outcomes', self.outcome) / 'run-outcome.json'
        self.ack_directory = self.root / 'artifacts' / 'run-outcome-acknowledgments'

    def create(self):
        return snapshots.create_snapshot(self.root, self.master, self.base / 'snapshots')

    def index(self, package):
        return json.loads((package / 'snapshot-index.json').read_text(encoding='utf-8'))

    def test_actual_native_preflight_and_multiple_acknowledgments_survive_original_root_deletion(self):
        def failed(*args, **kwargs):
            raise RuntimeError('actual native execution failed')
        with self.assertLogs('seedvision.export.run_outcomes', level='ERROR'):
            native = runs.run_native_analysis(self.historical, runner=failed)
        native_file = runs.export_run_outcome(self.root / 'artifacts' / 'run-outcomes', native) / 'run-outcome.json'
        ack_paths = [runs.acknowledge_run_outcome(self.ack_directory, outcome, account)
                     for outcome, account in ((self.outcome, 'first reviewer'), (self.outcome, 'second reviewer'), (native, 'native reviewer'))]
        expected = {self.saved.read_bytes(), native_file.read_bytes()}
        ack_bytes = {path.read_bytes() for path in ack_paths}
        before_master = self.master.read_bytes()
        self.historical.unlink()
        with patch.object(runs, '_source_digest', side_effect=AssertionError('historical sources must remain inert')):
            package = self.create()
        self.assertEqual(self.master.read_bytes(), before_master)
        index = self.index(package)
        self.assertEqual(index['version'], 1)
        rows = portable.verify_terminal_evidence(package, index)
        self.assertEqual({raw for name, _, _, _, raw in rows if name.startswith('terminal-history/')}, expected)
        self.assertEqual({raw for name, _, _, _, raw in rows if name.startswith('terminal-acknowledgments/')}, ack_bytes)
        self.assertTrue(self.root.resolve().is_relative_to(self.base.resolve()))
        shutil.rmtree(self.root)
        restored_root = self.base / 'restored'
        working = snapshots.restore_snapshot(package, restored_root)
        history = portable.load_project_terminal_history(restored_root, working)
        self.assertEqual({value.payload_json.encode('utf-8') for value in history.outcomes}, expected)
        self.assertEqual(set(portable.load_project_acknowledgments(restored_root, working).records), ack_bytes)
        self.assertFalse(history.diagnostics)
        self.assertFalse((restored_root / 'artifacts').exists())
        self.assertFalse((restored_root / 'result-reviews').exists())
        second = snapshots.create_snapshot(restored_root, working, self.base / 'second-snapshots')
        again = snapshots.restore_snapshot(second, restored_root)
        self.assertEqual({value.payload_json.encode('utf-8') for value in portable.load_project_terminal_history(restored_root, again).outcomes}, expected)
        self.assertEqual(set(portable.load_project_acknowledgments(restored_root, again).records), ack_bytes)

    def test_identical_workspace_private_copies_coalesce_and_only_selected_private_project_is_read(self):
        private = self.master.parent / 'terminal-history'
        runs.export_run_outcome(private, self.outcome)
        other = self.root / 'other-project' / 'other.seedfiddle-project.json'
        another = preflight.begin_preflight(other, 'project', 'desktop_project_load').outcome(ValueError('different private project'))
        runs.export_run_outcome(other.parent / 'terminal-history', another)
        loaded = portable.load_project_terminal_history(self.root, self.master)
        self.assertEqual([value.payload for value in loaded.outcomes], [self.outcome.payload])
        self.assertEqual(len(portable.load_project_terminal_history(self.root, other).outcomes), 2)
        package = self.create()
        self.assertEqual(len([name for name in self.index(package)['supplementary_evidence'] if name.startswith('terminal-history/')]), 1)

    def test_conflicting_copies_reject_snapshot_without_selecting_an_arbitrary_diagnostic(self):
        copied = runs.export_run_outcome(self.master.parent / 'terminal-history', self.outcome) / 'run-outcome.json'
        payload = self.outcome.payload
        payload['diagnostic'] = 'contradictory historical diagnostic'
        copied.write_text(json.dumps(payload), encoding='utf-8')
        loaded = portable.load_project_terminal_history(self.root, self.master)
        self.assertEqual(loaded.outcomes, ())
        self.assertTrue(loaded.diagnostics)
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            self.create()
        self.assertFalse((self.base / 'snapshots').exists())

    def test_relative_selected_project_uses_its_workspace_private_namespace(self):
        relative = Path('relative-project') / 'working.seedfiddle-project.json'
        outcome = preflight.begin_preflight(self.root / relative, 'project', 'desktop_project_load').outcome(ValueError('relative private history'))
        runs.export_run_outcome(self.root / relative.parent / 'terminal-history', outcome)
        self.assertEqual({item.payload['outcome_revision'] for item in portable.load_project_terminal_history(self.root, relative).outcomes},
                         {self.outcome.payload['outcome_revision'], outcome.payload['outcome_revision']})

    def test_restore_aggregate_ack_limit_bounds_object_read_before_publication(self):
        ack = runs.acknowledge_run_outcome(self.ack_directory, self.outcome, 'reviewer')
        package = self.create()
        cap = len(ack.read_bytes()) - 1
        original = portable._read
        observed = []
        def counted(path, limit):
            if str(path).endswith('.terminal-ack.json'):
                observed.append(limit)
            return original(path, limit)
        with patch.object(portable, 'ACK_BYTES', cap), patch.object(portable, '_read', side_effect=counted), self.assertRaises(ValueError):
            snapshots.restore_snapshot(package, self.base / 'rejected')
        self.assertEqual(observed, [cap])
        self.assertFalse((self.base / 'rejected').exists())

    def test_reparse_acknowledgment_is_diagnostic_before_opening_content(self):
        from types import SimpleNamespace
        ack = runs.acknowledge_run_outcome(self.ack_directory, self.outcome, 'reviewer')
        original = Path.lstat
        def linked(path):
            info = original(path)
            return SimpleNamespace(st_mode=info.st_mode, st_file_attributes=0x400) if path == ack else info
        with patch.object(Path, 'lstat', new=linked), patch.object(Path, 'open', side_effect=AssertionError('linked file content opened')):
            loaded = portable.load_project_acknowledgments(self.root, self.master)
        self.assertEqual(loaded.records, ())
        self.assertTrue(loaded.diagnostics)

    def test_linked_terminal_inventory_is_withheld_before_scanning_records(self):
        from types import SimpleNamespace
        directory = self.root / 'artifacts' / 'run-outcomes'
        original = Path.lstat
        def linked(path):
            info = original(path)
            return SimpleNamespace(st_mode=info.st_mode, st_file_attributes=0x400) if path == directory else info
        with patch.object(Path, 'lstat', new=linked), patch.object(portable, 'load_run_outcome_history') as reader:
            loaded = portable.load_project_terminal_history(self.root, self.master)
        reader.assert_not_called()
        self.assertTrue(loaded.incomplete)
        self.assertEqual(loaded.outcomes, ())

    def test_omitted_terminal_mapping_does_not_silently_discard_verified_history(self):
        package = self.create()
        index = self.index(package)
        name = next(name for name in index['supplementary_evidence'] if name.startswith('terminal-history/'))
        index['supplementary_evidence'].pop(name)
        (package / 'snapshot-index.json').write_text(json.dumps(index), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'omitted or extra'):
            snapshots.restore_snapshot(package, self.base / 'rejected')
        self.assertFalse((self.base / 'rejected').exists())

    def test_malformed_orphan_and_oversized_acknowledgments_cannot_publish_a_complete_snapshot(self):
        ack = runs.acknowledge_run_outcome(self.ack_directory, self.outcome, 'reviewer')
        original = ack.read_bytes()
        record = json.loads(original)
        orphan = {**record, 'outcome_revision': 'a' * 64}
        for raw in (b'{', b'{"contract":1,"contract":2}', b'x' * (portable.ACK_LIMIT + 1), json.dumps(orphan).encode()):
            with self.subTest(raw_length=len(raw)):
                ack.write_bytes(raw)
                with self.assertRaises(ValueError):
                    self.create()
                self.assertFalse((self.base / 'snapshots').exists())
        ack.write_bytes(original)
        self.assertTrue(self.create().is_dir())

    def test_acknowledgment_contract_rejects_extra_seed_decisions_blank_accounts_and_non_utc_time(self):
        path = runs.acknowledge_run_outcome(self.ack_directory, self.outcome, 'reviewer')
        original = json.loads(path.read_bytes())
        self.assertEqual(portable.load_acknowledgment(path.read_bytes()), original)
        for changes in ({'decisions': {'1': 'accepted'}}, {'operator_account': '  '}, {'decision': 'accepted'},
                        {'reviewed_at_utc': '2026-10-12T12:00:00'}, {'reviewed_at_utc': '2026-10-12T12:00:00+01:00'}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                portable.load_acknowledgment(json.dumps({**original, **changes}).encode())

    def test_combined_history_and_acknowledgment_budgets_withhold_incomplete_capture(self):
        runs.acknowledge_run_outcome(self.ack_directory, self.outcome, 'first')
        runs.acknowledge_run_outcome(self.ack_directory, self.outcome, 'second')
        with patch.object(portable, 'ACK_RECORDS', 1):
            loaded = portable.load_project_acknowledgments(self.root, self.master)
            self.assertTrue(loaded.incomplete)
            self.assertEqual(loaded.records, ())
            with self.assertRaises(ValueError):
                self.create()
        with patch.object(portable, 'HISTORY_BYTES', len(self.saved.read_bytes()) - 1):
            self.assertTrue(portable.load_project_terminal_history(self.root, self.master).incomplete)
            with self.assertRaises(ValueError):
                self.create()
        with patch.object(portable, 'HISTORY_ENTRIES', 1):
            runs.export_run_outcome(self.master.parent / 'terminal-history', self.outcome)
            self.assertTrue(portable.load_project_terminal_history(self.root, self.master).incomplete)

    def test_unsafe_unverified_or_revision_mismatched_inventory_rejects_before_restore_publication(self):
        runs.acknowledge_run_outcome(self.ack_directory, self.outcome, 'reviewer')
        package = self.create()
        original = self.index(package)
        name = next(name for name in original['supplementary_evidence'] if name.startswith('terminal-history/'))
        object_path = original['supplementary_evidence'][name]
        for replacement in ('terminal-history/../run-outcome.json', name.upper(),
                            'terminal-history/' + 'a' * 64 + '/run-outcome.json', name + '.',
                            'terminal-history/CON/run-outcome.json'):
            index = deepcopy(original)
            index['supplementary_evidence'][replacement] = index['supplementary_evidence'].pop(name)
            (package / 'snapshot-index.json').write_text(json.dumps(index), encoding='utf-8')
            with self.subTest(name=replacement), self.assertRaises(ValueError):
                snapshots.restore_snapshot(package, self.base / 'rejected')
            self.assertFalse((self.base / 'rejected').exists())
        index = deepcopy(original)
        index['files'].pop(object_path)
        (package / 'snapshot-index.json').write_text(json.dumps(index), encoding='utf-8')
        with self.assertRaises(ValueError):
            snapshots.restore_snapshot(package, self.base / 'rejected')
        self.assertFalse((self.base / 'rejected').exists())

    def test_changed_object_or_orphan_mapping_rejects_before_restore_publication(self):
        runs.acknowledge_run_outcome(self.ack_directory, self.outcome, 'reviewer')
        package = self.create()
        index = self.index(package)
        terminal = next(name for name in index['supplementary_evidence'] if name.startswith('terminal-history/'))
        object_path = index['supplementary_evidence'][terminal]
        raw = (package / object_path).read_bytes()
        (package / object_path).write_bytes(raw + b'changed')
        with self.assertRaises(ValueError):
            snapshots.restore_snapshot(package, self.base / 'rejected')
        self.assertFalse((self.base / 'rejected').exists())
        (package / object_path).write_bytes(raw)
        index['supplementary_evidence'].pop(terminal)
        (package / 'snapshot-index.json').write_text(json.dumps(index), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'no included'):
            snapshots.restore_snapshot(package, self.base / 'rejected')
        self.assertFalse((self.base / 'rejected').exists())

    def test_restore_terminal_copy_failure_never_publishes_working_master_or_completion_marker(self):
        package = self.create()
        old = (package / 'snapshot-index.json').read_bytes()
        original = snapshots.evidence_bytes
        def reject_restored(path, digest, limit):
            if Path(path).is_relative_to(self.base / 'partial-restore'):
                raise OSError('interrupted terminal copy')
            return original(path, digest, limit)
        with patch.object(snapshots, 'evidence_bytes', side_effect=reject_restored), self.assertRaises(OSError):
            snapshots.restore_snapshot(package, self.base / 'partial-restore')
        self.assertFalse(list((self.base / 'partial-restore').rglob('working.seedfiddle-project.json')))
        self.assertFalse(list((self.base / 'partial-restore').rglob('snapshot-index.json')))
        self.assertEqual((package / 'snapshot-index.json').read_bytes(), old)

    def test_create_terminal_copy_failure_never_publishes_completion_marker(self):
        with patch.object(snapshots, 'evidence_bytes', side_effect=OSError('interrupted evidence copy')), self.assertRaises(OSError):
            self.create()
        self.assertFalse(list((self.base / 'snapshots').rglob('snapshot-index.json')))

    def test_legacy_snapshot_without_terminal_namespace_still_restores(self):
        self.saved.unlink()
        self.saved.parent.rmdir()
        package = self.create()
        self.assertFalse(any(name.startswith('terminal-') for name in self.index(package)['supplementary_evidence']))
        working = snapshots.restore_snapshot(package, self.base / 'restored')
        self.assertFalse(portable.load_project_terminal_history(self.base / 'restored', working).outcomes)
        self.assertFalse(ProjectAnalysisStore(self.base / 'restored').load(working).issues)


@unittest.skipUnless(torch.cuda.is_available(), 'Actual native Qt fixture requires CUDA')
class TerminalSnapshotUiTests(unittest.TestCase):
    setUpClass = classmethod(desktop_fixture.NativePreflightOutcomeTests.setUpClass.__func__)
    setUp = desktop_fixture.NativePreflightOutcomeTests.setUp
    save_project = desktop_fixture.NativePreflightOutcomeTests.save_project
    install_profile = desktop_fixture.NativePreflightOutcomeTests.install_profile
    decode = staticmethod(desktop_fixture.NativePreflightOutcomeTests.decode)

    def test_actual_private_history_and_recorded_acknowledgments_preserve_current_analysis_and_drafts(self):
        from PySide6.QtWidgets import QDialog, QTableWidget, QPushButton, QPlainTextEdit
        private_master = self.root / 'selected-private' / 'working.seedfiddle-project.json'
        private_master.parent.mkdir()
        self.window._current_project_path = private_master
        outcome = preflight.begin_preflight(self.source, 'image', 'desktop_source').outcome(ValueError('private historical rejection'))
        saved = runs.export_run_outcome(private_master.parent / 'terminal-history', outcome) / 'run-outcome.json'
        ack = runs.acknowledge_run_outcome(private_master.parent / 'terminal-acknowledgments', outcome, 'recorded reviewer')
        raw, ackraw = saved.read_bytes(), ack.read_bytes()
        current, draft = object(), self.labels.copy()
        self.window._analyses[self.key] = current
        self.window._draft_instance_annotations[self.key] = draft
        notes = []
        def exercise(dialog):
            if dialog.windowTitle() == 'Terminal run history':
                table = dialog.findChild(QTableWidget)
                self.assertEqual(table.rowCount(), 1)
                table.selectRow(0)
                next(button for button in dialog.findChildren(QPushButton) if button.text().startswith('Review selected')).click()
            else:
                notes.extend(widget.toPlainText() for widget in dialog.findChildren(QPlainTextEdit))
                next(button for button in dialog.findChildren(QPushButton) if button.text().startswith('Acknowledge')).click()
            return QDialog.DialogCode.Rejected
        with patch.object(QDialog, 'exec', new=exercise):
            self.window.terminal_history_action.trigger()
        self.assertTrue(any('Recorded acknowledgment: recorded reviewer at ' in note for note in notes))
        self.assertIs(self.window._analyses[self.key], current)
        self.assertIs(self.window._draft_instance_annotations[self.key], draft)
        self.assertEqual(self.window._image_run_outcomes, {})
        self.assertEqual(self.window._run_outcome_history, {})
        self.assertFalse((self.root / 'result-reviews').exists())
        self.assertEqual(saved.read_bytes(), raw)
        self.assertEqual(ack.read_bytes(), ackraw)
        newack, = (self.root / 'artifacts' / 'run-outcome-acknowledgments').glob('*.json')
        self.assertEqual(portable.load_acknowledgment(newack.read_bytes())['outcome_revision'], outcome.payload['outcome_revision'])

    def test_actual_restored_project_open_exposes_transported_history_and_acknowledgment(self):
        from PySide6.QtWidgets import QDialog, QTableWidget, QPushButton, QPlainTextEdit
        outcome = preflight.begin_preflight(self.source, 'image', 'desktop_source').outcome(ValueError('transported historical rejection'))
        saved = runs.export_run_outcome(self.root / 'artifacts' / 'run-outcomes', outcome) / 'run-outcome.json'
        ack = runs.acknowledge_run_outcome(self.root / 'artifacts' / 'run-outcome-acknowledgments', outcome, 'transported reviewer')
        package = snapshots.create_snapshot(self.root, self.master, self.root / 'packages')
        working = snapshots.restore_snapshot(package, self.root)
        saved.unlink()
        saved.parent.rmdir()
        ack.unlink()
        with patch.object(self.window, '_confirm_project_replacement', return_value=True), \
             patch.object(self.window, '_analyze_current_image'), \
             patch('seedvision.ui.main_window.QMessageBox.critical') as critical:
            self.assertTrue(self.window._open_project(working))
        critical.assert_not_called()
        self.assertEqual(self.window._current_project_path, working.resolve())
        self.assertEqual(self.window._image_run_outcomes, {})
        notes = []
        def exercise(dialog):
            if dialog.windowTitle() == 'Terminal run history':
                table = dialog.findChild(QTableWidget)
                self.assertEqual(table.rowCount(), 1)
                table.selectRow(0)
                next(button for button in dialog.findChildren(QPushButton) if button.text().startswith('Review selected')).click()
            else:
                notes.extend(widget.toPlainText() for widget in dialog.findChildren(QPlainTextEdit))
            return QDialog.DialogCode.Rejected
        with patch.object(QDialog, 'exec', new=exercise):
            self.window.terminal_history_action.trigger()
        self.assertTrue(any('Recorded acknowledgment: transported reviewer at ' in note for note in notes))
        self.assertEqual(self.window._selected_history_outcome.payload, outcome.payload)
        self.assertEqual(self.window._image_run_outcomes, {})
        self.assertEqual(self.window._run_outcome_history, {})
        self.assertFalse((self.root / 'result-reviews').exists())
