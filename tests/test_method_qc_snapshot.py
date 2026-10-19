"""Explicit project ownership and actual portable QC consumers."""

from tests.functional_units import functional_units
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


@functional_units
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



    def check_test_exported_sidecar_is_not_retained_without_explicit_action_and_legacy_snapshot_restores(self):
        qc.export_method_qc(self.quality, self.root/'arbitrary-output'/'method-qc.json')
        self.assertFalse(history.load_project_method_qc(self.root, self.master).records)
        package = self.create()
        self.assertEqual(history.verify_method_qc_evidence(package, self.index(package)), ())
        working = snapshots.restore_snapshot(package, self.base/'restored')
        self.assertTrue(working.exists())
        self.assertFalse(history.load_project_method_qc(self.base/'restored', working).records)


    def check_test_malformed_duplicate_nonfinite_non_utf8_and_oversized_qc_are_diagnostic(self):
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



    def check_test_linked_inventory_and_record_are_withheld_before_implicit_content_read(self):
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


    def check_test_changed_objects_and_restore_byte_limits_reject_before_publication(self):
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


    def check_test_retention_validates_ack_before_writing_and_rejects_conflicting_existing_bytes(self):
        with self.assertRaises(ValueError):
            self.retain(acknowledgment=self.acknowledge() | {'qc_revision': 'f'*64})
        self.assertFalse(history.history_root(self.root, self.master).exists())
        path = self.retain()[0]
        self.assertEqual(self.retain(), (path,))
        path.write_bytes(b'conflicting bytes')
        with self.assertRaises(FileExistsError):
            self.retain()
        self.assertEqual(path.read_bytes(), b'conflicting bytes')

    def test_workflow_lifecycle(self):
        """3 related MethodQCSnapshotTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_exported_sidecar_is_not_retained_without_explicit_action_and_legacy_snapshot_restores)
        self.scenario(self.check_test_malformed_duplicate_nonfinite_non_utf8_and_oversized_qc_are_diagnostic)
        self.scenario(self.check_test_linked_inventory_and_record_are_withheld_before_implicit_content_read)

    def test_validation_and_recovery(self):
        """2 related MethodQCSnapshotTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_changed_objects_and_restore_byte_limits_reject_before_publication)
        self.scenario(self.check_test_retention_validates_ack_before_writing_and_rejects_conflicting_existing_bytes)


class MethodQCSnapshotQtTests(unittest.TestCase):
    setUp = MethodQCSnapshotTests.setUp
    record = MethodQCSnapshotTests.record
    acknowledge = MethodQCSnapshotTests.acknowledge
    retain = MethodQCSnapshotTests.retain
    create = MethodQCSnapshotTests.create

    def test_actual_retain_action_restored_graph_history_and_multiple_acknowledgments_are_inert(self):
        from PySide6.QtWidgets import QApplication
        from seedvision.ui.method_qc_dialog import MethodQCDialog
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
        master_before = working.read_bytes()
        retained = history.load_project_method_qc(restored, working)
        qualities = history._identities(retained.records)
        events = tuple(history._ack(raw, qualities) for raw in retained.acknowledgments)
        child = MethodQCDialog(history._quality(retained.records[0]), project_root=restored, project_path=working)
        self.addCleanup(child.deleteLater)
        self.assertEqual(child.quality, self.quality)
        self.assertEqual({event['operator_account'] for event in events}, {'first reviewer', 'second reviewer'})
        self.assertTrue(all(event['qc_revision'] == child.quality.revision for event in events))
        self.assertIn('scientific eligibility remain unknown', child.status.text())
        self.assertIsNone(child.acknowledgment, 'Opening several historical events must not arbitrarily select an actor.')
        self.assertEqual(working.read_bytes(), master_before)
        child.retain_button.click()
        self.assertEqual(len(history.load_project_method_qc(restored, working).acknowledgments), 2)
        app.processEvents()



@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class ActualNativeMethodQCSnapshotTests(unittest.TestCase):
    setUp = MethodQCSnapshotTests.setUp
    record = MethodQCSnapshotTests.record
    create = MethodQCSnapshotTests.create
    acknowledge = MethodQCSnapshotTests.acknowledge



class ActualReferenceMethodQCSnapshotTests(unittest.TestCase):
    setUpClass = classmethod(qc_fixture.RealReferenceMethodQCTests.setUpClass.__func__)
    setUp = MethodQCSnapshotTests.setUp
    record = MethodQCSnapshotTests.record
    create = MethodQCSnapshotTests.create
    acknowledge = MethodQCSnapshotTests.acknowledge



@unittest.skipUnless(torch.cuda.is_available(), 'Actual desktop fixture requires CUDA')
class MainWindowMethodQCHistoryTests(unittest.TestCase):
    setUpClass = classmethod(project_ui_fixture.ProjectAnalysisUiTests.setUpClass.__func__)
    setUp = MethodQCSnapshotTests.setUp
    record = MethodQCSnapshotTests.record
