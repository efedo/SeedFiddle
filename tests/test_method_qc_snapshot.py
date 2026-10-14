"""Explicit project ownership and actual portable QC consumers."""
from copy import deepcopy
from dataclasses import replace
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
import tests.test_method_qc as qc_fixture
import tests.test_project_analysis_ui as project_ui_fixture

from seedvision.benchmarking.checkpoint_inspection import CheckpointInspection, InspectionStatus
from seedvision.benchmarking.recipe_catalogue import build_recipe_catalogue
from seedvision.benchmarking.recipes import StageRunRecord, StageRunState
from seedvision.export import method_qc as qc
from seedvision.persistence import ProjectAnalysisStore, ProjectImageSpec, analysis_settings_profile_from_graph
from seedvision.persistence import snapshot as snapshots, method_qc_history as history
from seedvision.pipeline import build_default_pipeline


class MethodQCSnapshotTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.root = self.base/'original'
        self.root.mkdir()
        source = self.root/'live.png'
        self.assertTrue(cv2.imwrite(str(source), np.zeros((16, 16, 3), np.uint8)))
        self.master = ProjectAnalysisStore(self.root).capture_and_save(
            analysis_settings=analysis_settings_profile_from_graph(build_default_pipeline()),
            images=(ProjectImageSpec(source, (16, 16)),), destination=self.root/'first.seedfiddle-project.json')
        self.recipe = build_recipe_catalogue().recipe('plantcv_seed_otsu')
        self.stage = next(stage for stage in self.recipe.stages if stage.operation_id == 'threshold.otsu')
        self.quality = self.record()

    def record(self, recipe=None, failed=False):
        recipe = recipe or self.recipe
        record = StageRunRecord(self.stage.stage_id, recipe.identity,
            StageRunState.FAILED if failed else StageRunState.NOT_RUN,
            reason='Original stage executor rejected input.' if failed else '')
        return qc.qc_from_stage_inspection(recipe, record, CheckpointInspection(
            InspectionStatus.RECORDED if failed else InspectionStatus.NOT_RUN, 'Historical inspection.'))

    def acknowledge(self, quality=None, account='reviewer', when='2026-10-12T12:00:00+00:00'):
        quality = quality or self.quality
        path = self.base/(account.replace(' ', '_')+'-ack.json')
        qc.acknowledge_method_qc(quality, path, account, reviewed_at_utc=when)
        return qc.load_method_qc_acknowledgment(path, quality)

    def retain(self, quality=None, acknowledgment=None, project=None):
        return history.retain_method_qc(self.root, quality or self.quality,
            project_path=project or self.master, acknowledgment=acknowledgment)

    def create(self):
        return snapshots.create_snapshot(self.root, self.master, self.base/'snapshots')

    def index(self, package):
        return json.loads((package/'snapshot-index.json').read_text(encoding='utf-8'))

    def write_index(self, package, value):
        (package/'snapshot-index.json').write_text(json.dumps(value), encoding='utf-8')

    def assert_no_restore(self, package):
        destination = self.base/'rejected'
        with self.assertRaises((ValueError, OSError)):
            snapshots.restore_snapshot(package, destination)
        self.assertFalse(destination.exists(), 'Rejected evidence must fail before any restore publication.')

    def test_explicit_ownership_isolates_masters_sharing_a_directory_and_unassigned_workspace(self):
        second = self.master.with_name('second.seedfiddle-project.json')
        alternate = self.record(failed=True)
        self.retain()
        history.retain_method_qc(self.root, alternate, project_path=second)
        history.retain_method_qc(self.root, alternate)
        first = history.load_project_method_qc(self.root, self.master)
        other = history.load_project_method_qc(self.root, second)
        unassigned = history.load_project_method_qc(self.root)
        self.assertEqual([history._quality(raw) for raw in first.records], [self.quality])
        self.assertEqual([history._quality(raw) for raw in other.records], [alternate])
        self.assertEqual([history._quality(raw) for raw in unassigned.records], [alternate])
        relative = self.master.relative_to(self.root)
        self.assertEqual(history.load_project_method_qc(self.root, relative), first)
        package = self.create()
        transported = history.verify_method_qc_evidence(package, self.index(package))
        self.assertEqual([history._quality(raw) for name, _, _, _, raw in transported], [self.quality])

    def test_layout_views_multiple_accounts_and_exact_bytes_survive_deletion_and_resnapshot(self):
        moved = replace(self.recipe, layout=tuple(replace(position, x=position.x+40) for position in self.recipe.layout))
        alternate = self.record(recipe=moved)
        self.assertEqual(self.quality.revision, alternate.revision)
        self.assertNotEqual(self.quality.payload_json, alternate.payload_json)
        self.retain(acknowledgment=self.acknowledge(account='first reviewer'))
        self.retain(alternate, self.acknowledge(account='second reviewer'))
        self.retain(acknowledgment=self.acknowledge(account='third reviewer', when='2026-10-12T13:00:00+00:00'))
        # Preserve an explicitly retained, strict but noncanonical representation too.
        unusual = json.dumps(self.quality.payload, ensure_ascii=False, indent=2).encode('utf-8')
        directory = history.history_root(self.root, self.master)/'method-qc'
        (directory/(sha256(unusual).hexdigest()+'.json')).write_bytes(unusual)
        before = history.load_project_method_qc(self.root, self.master)
        self.assertEqual(len(before.records), 3)
        self.assertEqual(len(before.acknowledgments), 3)
        master_bytes = self.master.read_bytes()
        with patch('seedvision.benchmarking.checkpoint_inspection.load_checkpoint', side_effect=AssertionError('No old artifact probes')):
            package = self.create()
        self.assertEqual(self.master.read_bytes(), master_bytes)
        shutil.rmtree(self.root)
        restored = self.base/'restored'
        working = snapshots.restore_snapshot(package, restored)
        after = history.load_project_method_qc(restored, working)
        self.assertEqual(after.records, before.records)
        self.assertEqual(after.acknowledgments, before.acknowledgments)
        self.assertFalse(after.diagnostics)
        self.assertFalse((restored/'artifacts').exists())
        second = snapshots.create_snapshot(restored, working, self.base/'second-snapshots')
        again = snapshots.restore_snapshot(second, restored)
        repeated = history.load_project_method_qc(restored, again)
        self.assertEqual((repeated.records, repeated.acknowledgments), (before.records, before.acknowledgments))

    def test_exported_sidecar_is_not_retained_without_explicit_action_and_legacy_snapshot_restores(self):
        qc.export_method_qc(self.quality, self.root/'arbitrary-output'/'method-qc.json')
        self.assertFalse(history.load_project_method_qc(self.root, self.master).records)
        package = self.create()
        self.assertEqual(history.verify_method_qc_evidence(package, self.index(package)), ())
        working = snapshots.restore_snapshot(package, self.base/'restored')
        self.assertTrue(working.exists())
        self.assertFalse(history.load_project_method_qc(self.base/'restored', working).records)

    def test_orphan_stale_and_invalid_acknowledgments_do_not_publish_complete_snapshot(self):
        self.retain()
        directory = history.history_root(self.root, self.master)/'method-qc-acknowledgments'
        directory.mkdir()
        original = self.acknowledge()
        for changes in ({'qc_revision': 'f'*64}, {'decision': 'approved'}, {'operator_account': ' '},
                        {'reviewed_at_utc': '2026-10-12T12:00:00'}, {'seed_approval': True}):
            with self.subTest(changes=changes):
                (directory/'bad.json').write_text(json.dumps(original | changes), encoding='utf-8')
                captured = history.load_project_method_qc(self.root, self.master)
                self.assertTrue(captured.diagnostics)
                self.assertFalse(captured.acknowledgments)
                with self.assertRaisesRegex(ValueError, 'incomplete'):
                    self.create()
                self.assertFalse((self.base/'snapshots').exists())

    def test_malformed_duplicate_nonfinite_non_utf8_and_oversized_qc_are_diagnostic(self):
        directory = history.history_root(self.root, self.master)/'method-qc'
        directory.mkdir(parents=True)
        for raw in (b'{"format":1,"format":2}', b'{"x":NaN}', b'\xff', b' '* (qc.LIMIT+1)):
            with self.subTest(size=len(raw)):
                (directory/'bad.json').write_bytes(raw)
                captured = history.load_project_method_qc(self.root, self.master)
                self.assertTrue(captured.diagnostics)
                self.assertFalse(captured.records)
                with self.assertRaises(ValueError):
                    self.create()

    def test_entry_record_and_aggregate_budgets_withhold_incomplete_inventory(self):
        self.retain()
        self.retain(self.record(failed=True))
        for name, limit in (('ENTRIES', 1), ('RECORDS', 1), ('TOTAL_BYTES', len(self.quality.payload_json)-1)):
            with self.subTest(name=name), patch.object(history, name, limit):
                captured = history.load_project_method_qc(self.root, self.master)
                self.assertTrue(captured.incomplete)
                self.assertFalse(captured.records)
                self.assertFalse(captured.acknowledgments)
                with self.assertRaises(ValueError):
                    self.create()

    def test_ack_record_aggregate_and_single_file_bounds_reject_capture(self):
        self.retain(acknowledgment=self.acknowledge(account='one'))
        self.retain(acknowledgment=self.acknowledge(account='two'))
        for name, limit in (('ACK_RECORDS', 1), ('ACK_BYTES', 1)):
            with self.subTest(name=name), patch.object(history, name, limit):
                captured = history.load_project_method_qc(self.root, self.master)
                self.assertTrue(captured.incomplete)
                self.assertFalse(captured.records)
                with self.assertRaises(ValueError):
                    self.create()
        directory = history.history_root(self.root, self.master)/'method-qc-acknowledgments'
        (directory/'bad.json').write_bytes(b' '* (qc.ACK_LIMIT+1))
        with self.assertRaises(ValueError):
            self.create()

    def test_linked_inventory_and_record_are_withheld_before_implicit_content_read(self):
        paths = self.retain()
        original_lstat = Path.lstat
        for target in (history.history_root(self.root, self.master), paths[0]):
            def reparse(path, *args, **kwargs):
                info = original_lstat(path, *args, **kwargs)
                if path == target:
                    from types import SimpleNamespace
                    return SimpleNamespace(st_file_attributes=0x400, st_mode=info.st_mode)
                return info
            with self.subTest(target=target.name), patch.object(Path, 'lstat', reparse):
                captured = history.load_project_method_qc(self.root, self.master)
                self.assertFalse(captured.records)
                self.assertTrue(captured.diagnostics)

    def test_missing_extra_swapped_and_unsafe_reserved_mappings_reject_before_restore(self):
        self.retain(acknowledgment=self.acknowledge())
        package = self.create()
        original = self.index(package)
        names = [name for name in original['supplementary_evidence'] if name.startswith('method-qc')]
        self.assertEqual(len(names), 2)
        variants = []
        omitted = deepcopy(original)
        del omitted['supplementary_evidence'][names[0]]
        variants.append(omitted)
        swapped = deepcopy(original)
        a, b = names
        swapped['supplementary_evidence'][a], swapped['supplementary_evidence'][b] = (
            swapped['supplementary_evidence'][b], swapped['supplementary_evidence'][a])
        variants.append(swapped)
        unsafe = deepcopy(original)
        unsafe['supplementary_evidence']['method-qc/../escape.json'] = unsafe['supplementary_evidence'].pop(a)
        variants.append(unsafe)
        extra = deepcopy(original)
        digest = 'f'*64
        extra['files']['objects/'+digest+'.method-qc.json'] = digest
        variants.append(extra)
        duplicate = deepcopy(original)
        duplicate['supplementary_evidence'][a.upper()] = duplicate['supplementary_evidence'][a]
        variants.append(duplicate)
        for index in variants:
            with self.subTest(mappings=list(index['supplementary_evidence'])):
                self.write_index(package, index)
                self.assert_no_restore(package)

    def test_changed_objects_and_restore_byte_limits_reject_before_publication(self):
        self.retain(acknowledgment=self.acknowledge())
        package = self.create()
        index = self.index(package)
        target = package/next(value for name, value in index['supplementary_evidence'].items()
                            if name.startswith('method-qc/'))
        original = target.read_bytes()
        target.write_bytes(original+b' ')
        self.assert_no_restore(package)
        target.write_bytes(original)
        for name in ('TOTAL_BYTES', 'ACK_BYTES'):
            with self.subTest(name=name), patch.object(history, name, 1):
                self.assert_no_restore(package)

    def test_capture_and_restore_copy_failures_preserve_prior_complete_snapshots(self):
        self.retain()
        package = self.create()
        before = (package/'snapshot-index.json').read_bytes()
        original_write = Path.write_bytes
        def broken_qc(path, raw):
            if path.name.endswith('.method-qc.json'):
                raise OSError('QC copy failed')
            return original_write(path, raw)
        with patch.object(Path, 'write_bytes', broken_qc):
            with self.assertRaises(OSError):
                self.create()
            with self.assertRaises(OSError):
                snapshots.restore_snapshot(package, self.base/'restored')
        self.assertEqual((package/'snapshot-index.json').read_bytes(), before)
        for directory in (self.base/'snapshots').iterdir():
            if directory != package:
                self.assertFalse((directory/'snapshot-index.json').exists())
        for directory in (self.base/'restored'/'restored-projects').iterdir():
            self.assertFalse((directory/'working.seedfiddle-project.json').exists())
            self.assertFalse((directory/'snapshot-index.json').exists())

    def test_retention_validates_ack_before_writing_and_rejects_conflicting_existing_bytes(self):
        with self.assertRaises(ValueError):
            self.retain(acknowledgment=self.acknowledge() | {'qc_revision': 'f'*64})
        self.assertFalse(history.history_root(self.root, self.master).exists())
        path = self.retain()[0]
        self.assertEqual(self.retain(), (path,))
        path.write_bytes(b'conflicting bytes')
        with self.assertRaises(FileExistsError):
            self.retain()
        self.assertEqual(path.read_bytes(), b'conflicting bytes')


class MethodQCSnapshotQtTests(unittest.TestCase):
    setUp = MethodQCSnapshotTests.setUp
    record = MethodQCSnapshotTests.record
    acknowledge = MethodQCSnapshotTests.acknowledge
    retain = MethodQCSnapshotTests.retain
    create = MethodQCSnapshotTests.create

    def test_actual_retain_action_restored_graph_history_and_multiple_acknowledgments_are_inert(self):
        from PySide6.QtWidgets import QApplication, QInputDialog
        from seedvision.ui.method_qc_dialog import MethodQCDialog
        from seedvision.ui.reference_graph_dialog import ReferenceGraphDialog
        from seedvision.persistence.reference_graphs import ReferenceGraphAttachment
        from seedvision.benchmarking.recipes import initial_stage_run
        app = QApplication.instance() or QApplication([])
        viewer = MethodQCDialog(self.quality, project_root=self.root, project_path=self.master)
        self.addCleanup(viewer.deleteLater)
        viewer.acknowledgment = self.acknowledge(account='first reviewer')
        viewer.retain_button.click()
        self.assertIn('Retained 2', viewer.status.text())
        self.retain(acknowledgment=self.acknowledge(account='second reviewer'))
        package = self.create()
        shutil.rmtree(self.root)
        restored = self.base/'restored'
        working = snapshots.restore_snapshot(package, restored)
        attachment = ReferenceGraphAttachment(self.recipe,
            tuple(initial_stage_run(self.recipe, stage.stage_id) for stage in self.recipe.stages), self.stage.stage_id)
        dialog = ReferenceGraphDialog(reference_graphs=(attachment,), project_root=restored, project_path=working)
        self.addCleanup(dialog.deleteLater)
        before = dialog.reference_graphs
        master_before = working.read_bytes()
        def choose(parent, title, label, values, *args):
            return values[0], True
        with patch.object(QInputDialog, 'getItem', side_effect=choose), patch.object(MethodQCDialog, 'exec', return_value=0):
            dialog._qc_history_button.click()
        child = dialog.findChildren(MethodQCDialog)[-1]
        self.assertEqual(child.quality, self.quality)
        self.assertIn('first reviewer', child.status.text())
        self.assertIn('second reviewer', child.status.text())
        self.assertIn('do not approve seeds', child.status.text())
        self.assertIsNone(child.acknowledgment, 'Opening several historical events must not arbitrarily select an actor.')
        self.assertEqual(dialog.reference_graphs, before)
        self.assertEqual(working.read_bytes(), master_before)
        child.retain_button.click()
        self.assertEqual(len(history.load_project_method_qc(restored, working).acknowledgments), 2)
        app.processEvents()

    def test_actual_stage_and_native_review_pass_explicit_project_ownership_to_retention(self):
        from PySide6.QtWidgets import QApplication
        from seedvision.ui.method_qc_dialog import MethodQCDialog
        from seedvision.ui.reference_graph_dialog import ReferenceGraphDialog
        from seedvision.ui.native_recipe_dialog import NativeRecipeDialog
        app = QApplication.instance() or QApplication([])
        parent = ReferenceGraphDialog(project_root=self.root, project_path=self.master)
        self.addCleanup(parent.deleteLater)
        with patch.object(MethodQCDialog, 'exec', return_value=0):
            parent._review_stage_qc()
        child = parent.findChildren(MethodQCDialog)[-1]
        self.assertEqual(child._project_path, self.master)
        child.retain_button.click()
        from seedvision.benchmarking.native_recipe import build_native_seed_recipe
        native = NativeRecipeDialog(build_native_seed_recipe(self.recipe), parent)
        self.addCleanup(native.deleteLater)
        native._exported_quality = self.quality
        with patch.object(MethodQCDialog, 'exec', return_value=0):
            native.qc_button.click()
        saved = native.findChildren(MethodQCDialog)[-1]
        self.assertEqual(saved._project_path, self.master)
        saved.retain_button.click()
        self.assertFalse(history.load_project_method_qc(self.root).records)
        self.assertTrue(history.load_project_method_qc(self.root, self.master).records)
        app.processEvents()


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class ActualNativeMethodQCSnapshotTests(unittest.TestCase):
    setUp = MethodQCSnapshotTests.setUp
    record = MethodQCSnapshotTests.record
    create = MethodQCSnapshotTests.create
    acknowledge = MethodQCSnapshotTests.acknowledge

    def test_actual_cuda_archive_qc_survives_archive_deletion_and_private_restore(self):
        from tests.test_native_recipe import selected_recipe, fixture, FRAME, SOURCE, PREPARATION
        from seedvision.benchmarking import native_recipe
        from seedvision.export.native_results import export_native_recipe_result
        from seedvision.segmentation.baseline import PipelineAnalysisCache
        recipe = native_recipe.build_native_seed_recipe(selected_recipe())
        gray, rgb = (torch.as_tensor(value, device='cuda') for value in fixture())
        result = native_recipe.run_native_recipe(recipe, gray, rgb, source_sha256=SOURCE,
            coordinate_frame=FRAME, preparation=PREPARATION, cache=PipelineAnalysisCache())
        self.assertEqual(result.status, 'executed')
        report = export_native_recipe_result(self.root/'actual-archive', recipe, result)
        quality = qc.load_method_qc(report.parent/qc.FILENAME, native_archive=report.parent)
        self.assertFalse(history.load_project_method_qc(self.root, self.master).records)
        history.retain_method_qc(self.root, quality, project_path=self.master, acknowledgment=self.acknowledge(quality))
        before = history.load_project_method_qc(self.root, self.master)
        shutil.rmtree(report.parent)
        with patch('seedvision.export.native_results.load_native_recipe_result', side_effect=AssertionError('Historical archive must remain inert')):
            package = self.create()
            restored = self.base/'restored'
            working = snapshots.restore_snapshot(package, restored)
            after = history.load_project_method_qc(restored, working)
        self.assertEqual((before.records, before.acknowledgments), (after.records, after.acknowledgments))
        artifact = history._quality(after.records[0]).payload['identity']['native_artifact']
        self.assertEqual(artifact['source_sha256'], SOURCE)
        self.assertEqual(artifact['coordinate_frame'], FRAME.to_dict())
        self.assertEqual(artifact['status'], 'executed')


class ActualReferenceMethodQCSnapshotTests(unittest.TestCase):
    setUpClass = classmethod(qc_fixture.RealReferenceMethodQCTests.setUpClass.__func__)
    setUp = MethodQCSnapshotTests.setUp
    record = MethodQCSnapshotTests.record
    create = MethodQCSnapshotTests.create
    acknowledge = MethodQCSnapshotTests.acknowledge

    def test_actual_original_checkpoint_inspection_survives_removed_store_as_historical_qc(self):
        from seedvision.benchmarking.checkpoint_inspection import inspect_reference_checkpoint
        cls = type(self)
        recipe, record = cls.recipe, cls.outcome.run_record
        observed = inspect_reference_checkpoint(cls.store, recipe, record)
        self.assertEqual(observed.status, InspectionStatus.VERIFIED, observed.detail)
        quality = qc.qc_from_stage_inspection(recipe, record, observed)
        history.retain_method_qc(self.root, quality, project_path=self.master, acknowledgment=self.acknowledge(quality))
        before = history.load_project_method_qc(self.root, self.master)
        shutil.rmtree(cls.store)
        with patch('seedvision.benchmarking.checkpoint_inspection.load_checkpoint', side_effect=AssertionError('Removed checkpoint store must remain inert')):
            package = self.create()
            restored = self.base/'restored'
            working = snapshots.restore_snapshot(package, restored)
            after = history.load_project_method_qc(restored, working)
        self.assertEqual((before.records, before.acknowledgments), (after.records, after.acknowledgments))
        facts = history._quality(after.records[0]).payload['identity']['recorded_inspection']
        self.assertEqual(facts['status'], 'verified')
        self.assertEqual(facts['checkpoint_sha256'], record.checkpoint_sha256)
        self.assertEqual(facts['source_sha256'], cls.outcome.checkpoint_reference.identity.source_sha256)


@unittest.skipUnless(torch.cuda.is_available(), 'Actual desktop fixture requires CUDA')
class MainWindowMethodQCHistoryTests(unittest.TestCase):
    setUpClass = classmethod(project_ui_fixture.ProjectAnalysisUiTests.setUpClass.__func__)
    setUp = MethodQCSnapshotTests.setUp
    record = MethodQCSnapshotTests.record

    def test_actual_restored_project_main_window_opens_owned_qc_without_replacing_analysis_or_drafts(self):
        from PySide6.QtWidgets import QInputDialog
        from seedvision.ui.method_qc_dialog import MethodQCDialog
        from seedvision.ui.reference_graph_dialog import ReferenceGraphDialog
        from seedvision.persistence.reference_graphs import ReferenceGraphAttachment
        from seedvision.benchmarking.recipes import initial_stage_run
        recipe = build_recipe_catalogue().recipe('plantcv_seed_otsu')
        stage = recipe.stages[0]
        quality = qc.qc_from_stage_inspection(recipe, initial_stage_run(recipe, stage.stage_id),
            CheckpointInspection(InspectionStatus.NOT_RUN, 'Recorded original stage has not run.'))
        history.retain_method_qc(self.root, quality, project_path=self.master)
        package = snapshots.create_snapshot(self.root, self.master, self.root/'packages')
        working = snapshots.restore_snapshot(package, self.root)
        window = project_ui_fixture.ProjectAnalysisUiTests._window(self.root)
        self.addCleanup(project_ui_fixture.ProjectAnalysisUiTests._dispose.__func__, type(self), window)
        with patch.object(window, '_confirm_project_replacement', return_value=True), \
             patch.object(window, '_analyze_current_image'), patch('seedvision.ui.main_window.QMessageBox.critical') as critical:
            self.assertTrue(window._open_project(working))
        critical.assert_not_called()
        attachment = ReferenceGraphAttachment(recipe, tuple(initial_stage_run(recipe, item.stage_id) for item in recipe.stages), stage.stage_id)
        window._set_project_reference_graphs((attachment,))
        before = window._project_reference_graphs
        key = window._current_image_key()
        self.assertIsNotNone(key)
        current, draft = object(), np.zeros((16, 16), np.uint16)
        window._analyses[key] = current
        window._draft_instance_annotations[key] = draft
        observed = []
        def show_graph(dialog):
            self.assertEqual(dialog._project_path, working.resolve())
            dialog._qc_history_button.click()
            return 0
        def show_record(dialog):
            observed.append(dialog.quality)
            self.assertEqual(dialog._project_path, working.resolve())
            return 0
        def choose(parent, title, label, values, *args):
            return values[0], True
        with patch.object(ReferenceGraphDialog, 'exec', new=show_graph), \
             patch.object(MethodQCDialog, 'exec', new=show_record), patch.object(QInputDialog, 'getItem', side_effect=choose):
            window._show_reference_graphs()
        self.assertEqual(observed, [quality])
        self.assertIs(window._analyses[key], current)
        self.assertIs(window._draft_instance_annotations[key], draft)
        self.assertEqual(window._project_reference_graphs, before)
        self.assertFalse(window._image_run_outcomes)
