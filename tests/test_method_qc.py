
from tests.functional_units import functional_units
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import shutil
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
import torch

from seedvision.benchmarking.checkpoint_inspection import (CheckpointInspection, InspectionStatus,
    inspect_reference_checkpoint)
from seedvision.benchmarking.recipes import StageRunRecord, StageRunState, initial_stage_run
from seedvision.benchmarking import native_recipe as runner
from seedvision.benchmarking.checkpoints import CoordinateFrame
from seedvision.benchmarking.recipe_catalogue import build_recipe_catalogue
from seedvision.export import method_qc as qc
from seedvision.export import native_results as archives
from seedvision.segmentation.baseline import PipelineAnalysisCache
from tests.test_native_recipe import selected_recipe, fixture, FRAME, SOURCE, PREPARATION
from tests.test_smpts_paper import contact_fixture


def recalculate(payload):
    payload['qc_revision'] = sha256(qc._bytes(payload['identity'])).hexdigest()
    return qc.MethodQC(qc._bytes(payload).decode('utf-8'))


@functional_units
class RecordedMethodQCTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.recipe = build_recipe_catalogue().recipe('plantcv_seed_otsu')
        self.stage = next(s for s in self.recipe.stages if s.operation_id == 'threshold.otsu')

    def quality(self, state=StageRunState.NOT_RUN, status=InspectionStatus.NOT_RUN, reason=''):
        checkpoint = 'b'*64 if state in (StageRunState.EXECUTED, StageRunState.EMPTY) else None
        record = StageRunRecord(self.stage.stage_id, self.recipe.identity, state,
                                checkpoint_sha256=checkpoint, reason=reason)
        return qc.qc_from_stage_inspection(self.recipe, record, CheckpointInspection(status, 'Recorded inspection fixture.'))


    def check_test_strict_schema_revision_nonfinite_duplicates_and_bounds_reject(self):
        quality = self.quality()
        for modify in (
            lambda p: p.update(version=True),
            lambda p: p['identity'].update(recipe_mode='paper_reconstruction'),
            lambda p: p['identity'].update(scientific_validity=True),
            lambda p: p['identity']['recorded_inspection'].update(source_sha256='a'*64),
            lambda p: p['identity'].update(node_outcomes=[None]),
        ):
            payload = quality.payload
            modify(payload)
            with self.assertRaises(ValueError):
                recalculate(payload)
        path = self.root/'bad.json'
        payload = quality.payload | {'qc_revision': 'f'*64}
        with self.assertRaisesRegex(ValueError, 'revision'):
            qc.MethodQC(qc._bytes(payload).decode('utf-8'))
        for raw in ('{"format":1,"format":2}', '{"x":NaN}', '{"x":Infinity}'):
            path.write_text(raw, encoding='utf-8')
            with self.assertRaises(ValueError):
                qc.load_method_qc(path)
        path.write_bytes(b' '* (qc.LIMIT+1))
        with self.assertRaisesRegex(ValueError, 'byte bound'):
            qc.load_method_qc(path)
        with self.assertRaises(ValueError):
            qc.MethodQC('{"format":1,"format":2}')
        with self.assertRaisesRegex(ValueError, 'terminal'):
            qc.qc_from_stage_inspection(self.recipe, initial_stage_run(self.recipe, self.stage.stage_id),
                CheckpointInspection(InspectionStatus.LOADING, 'Loading.'))

    def check_test_acknowledgment_exact_revision_conflicts_and_approval_claims_reject(self):
        quality = self.quality()
        path = self.root/'ack.json'
        instant = '2026-10-12T12:00:00+00:00'
        qc.acknowledge_method_qc(quality, path, 'operator', reviewed_at_utc=instant)
        original = path.read_bytes()
        self.assertEqual(qc.acknowledge_method_qc(quality, path, 'operator', reviewed_at_utc=instant), path)
        with self.assertRaises(FileExistsError):
            qc.acknowledge_method_qc(quality, path, 'someone else', reviewed_at_utc=instant)
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(qc.load_method_qc_acknowledgment(path, quality)['decision'], 'acknowledged')
        changed = self.quality(StageRunState.FAILED, InspectionStatus.RECORDED, 'Failed.')
        with self.assertRaisesRegex(ValueError, 'stale'):
            qc.load_method_qc_acknowledgment(path, changed)
        payload = json.loads(original)
        for key, value in (('decision', 'approved'), ('reviewed_at_utc', '2026-10-12T12:00:00'),
                           ('operator_account', ' '), ('seed_approval', True)):
            path.write_text(json.dumps(payload | {key: value}), encoding='utf-8')
            with self.assertRaises(ValueError):
                qc.load_method_qc_acknowledgment(path, quality)
        path.write_text('{"contract":1,"contract":2}', encoding='utf-8')
        with self.assertRaises(ValueError):
            qc.load_method_qc_acknowledgment(path, quality)

    def check_test_layout_only_reference_changes_preserve_semantic_qc_identity(self):
        original = self.quality()
        moved = replace(self.recipe, layout=tuple(replace(p, x=p.x+20) for p in self.recipe.layout))
        record = initial_stage_run(moved, self.stage.stage_id)
        actual = qc.qc_from_stage_inspection(moved, record,
            CheckpointInspection(InspectionStatus.NOT_RUN, 'Recorded inspection fixture.'))
        self.assertEqual(original.revision, actual.revision)
        self.assertNotEqual(original.payload['recipe'], actual.payload['recipe'])

    def test_validation_and_recovery(self):
        """2 related RecordedMethodQCTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_strict_schema_revision_nonfinite_duplicates_and_bounds_reject)
        self.scenario(self.check_test_acknowledgment_exact_revision_conflicts_and_approval_claims_reject)

    def test_workflow_lifecycle(self):
        """1 related RecordedMethodQCTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_layout_only_reference_changes_preserve_semantic_qc_identity)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class NativeMethodQCTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.recipe = runner.build_native_seed_recipe(selected_recipe())
        self.gray, self.rgb = (torch.as_tensor(v, device='cuda') for v in fixture())

    def run_recipe(self, gray=None, recipe=None, rgb=None, frame=FRAME):
        return runner.run_native_recipe(recipe or self.recipe,
            self.gray if gray is None else gray, self.rgb if rgb is None else rgb,
            source_sha256=SOURCE, coordinate_frame=frame, preparation=PREPARATION,
            cache=PipelineAnalysisCache())

    def export(self, result, name='run', recipe=None):
        path = archives.export_native_recipe_result(self.root/name, recipe or self.recipe, result)
        quality = qc.load_method_qc(path.parent/qc.FILENAME, native_archive=path.parent)
        return path, quality



    def test_legacy_archive_without_sidecar_derives_only_verified_stored_facts(self):
        path, quality = self.export(self.run_recipe())
        (path.parent/qc.FILENAME).unlink()
        with self.assertRaises(FileNotFoundError):
            qc.load_method_qc(path.parent/qc.FILENAME)
        derived = qc.qc_from_native_archive(path.parent)
        self.assertEqual(derived, quality)
        self.assertIsNone(derived.payload['identity']['scientific_validity'])





class MethodQCQtTests(unittest.TestCase):
    setUp = RecordedMethodQCTests.setUp
    quality = RecordedMethodQCTests.quality
    def test_real_qt_export_ack_load_and_stale_record_preserve_unknown_eligibility(self):
        from PySide6.QtWidgets import QApplication
        from seedvision.ui.method_qc_dialog import MethodQCDialog
        self.app = QApplication.instance() or QApplication([])
        quality = self.quality(StageRunState.EMPTY, InspectionStatus.MISSING)
        dialog = MethodQCDialog(quality)
        self.addCleanup(dialog.deleteLater)
        path, ack = self.root/'qc.json', self.root/'ack.json'
        with patch('seedvision.ui.method_qc_dialog.QFileDialog.getSaveFileName', return_value=(str(path),'JSON')):
            dialog.export_button.click()
        dialog.operator.setText('local reviewer')
        with patch('seedvision.ui.method_qc_dialog.QFileDialog.getSaveFileName', return_value=(str(ack),'JSON')):
            dialog.ack_button.click()
        self.assertEqual(dialog.acknowledgment['qc_revision'], quality.revision)
        dialog.set_quality(None)
        with patch('seedvision.ui.method_qc_dialog.QFileDialog.getOpenFileName', return_value=(str(path),'JSON')):
            dialog.load_button.click()
        with patch('seedvision.ui.method_qc_dialog.QFileDialog.getOpenFileName', return_value=(str(ack),'JSON')):
            dialog.load_ack_button.click()
        self.assertIn('Recorded acknowledgment: local reviewer', dialog.status.text())
        self.assertIn('Recorded checkpoint inspection: missing', dialog.summary.text())
        changed = self.quality(StageRunState.FAILED, InspectionStatus.RECORDED, 'Failed.')
        dialog.set_quality(changed)
        self.assertIsNone(dialog.acknowledgment)
        with patch('seedvision.ui.method_qc_dialog.QFileDialog.getOpenFileName', return_value=(str(ack),'JSON')):
            dialog.load_ack_button.click()
        self.assertIsNone(dialog.acknowledgment)
        self.assertIn('stale', dialog.status.text())
        self.assertIsNone(dialog.quality.payload['identity']['scientific_validity'])

    def test_actual_reference_graph_button_captures_selected_stage_only(self):
        from PySide6.QtWidgets import QApplication
        from seedvision.ui.method_qc_dialog import MethodQCDialog
        self.app = QApplication.instance() or QApplication([])
        from seedvision.persistence.reference_graphs import ReferenceGraphAttachment
        attachment = ReferenceGraphAttachment(self.recipe,
            tuple(initial_stage_run(self.recipe, s.stage_id) for s in self.recipe.stages), self.stage.stage_id)
        record = next(record for record in attachment.stage_runs if record.stage_id == attachment.selected_stage_id)
        quality = qc.qc_from_stage_inspection(attachment.recipe, record,
            CheckpointInspection(InspectionStatus.NOT_RUN, 'Recorded stage has not run.'))
        dialog = MethodQCDialog(quality)
        self.addCleanup(dialog.deleteLater)
        self.assertEqual(dialog.quality.payload['identity']['recipe_identity'], attachment.recipe.identity.analytical_dict())
        self.assertEqual(dialog.quality.payload['identity']['node_outcomes'][0]['stage_id'], self.stage.stage_id)
        self.assertEqual(len(dialog.quality.payload['identity']['node_outcomes']), 1)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class NativeMethodQCQtTests(unittest.TestCase):
    setUp = NativeMethodQCTests.setUp
    run_recipe = NativeMethodQCTests.run_recipe



class RealReferenceMethodQCTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tests.test_reference_stage import HAS_REFERENCE_ENV, _fixture
        from seedvision.benchmarking.reference_stage import run_reference_stage
        if not HAS_REFERENCE_ENV:
            raise unittest.SkipTest('Ignored pinned PlantCV reference environment unavailable')
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name)
        args, cls.recipe, cls.stage, input_ref, _ = _fixture(str(cls.root/'bridge'), operation_id='threshold.otsu')
        cls.outcome = run_reference_stage(**args)
        if cls.outcome.checkpoint_reference is None:
            raise AssertionError(cls.outcome.run_record)
        cls.store = cls.root/'store'
        cls.store.mkdir()
        for directory, digest in ((args['input_root'], input_ref.checkpoint_sha256),
                (args['output_root'], cls.outcome.checkpoint_reference.checkpoint_sha256)):
            shutil.copytree(directory/(digest+'.checkpoint'), cls.store/(digest+'.checkpoint'))

    def test_actual_pinned_stage_inspection_roundtrip_and_missing_transition_invalidate_ack(self):
        record = self.outcome.run_record
        inspection = inspect_reference_checkpoint(self.store, self.recipe, record)
        self.assertEqual(inspection.status, InspectionStatus.VERIFIED, inspection.detail)
        quality = qc.qc_from_stage_inspection(self.recipe, record, inspection)
        path = qc.export_method_qc(quality, self.root/'verified-qc.json')
        observed = quality.payload['identity']['recorded_inspection']
        self.assertEqual(observed['checkpoint_sha256'], record.checkpoint_sha256)
        self.assertEqual(observed['source_sha256'], self.outcome.checkpoint_reference.identity.source_sha256)
        self.assertEqual(observed['array_inventory'], qc._plain(inspection.array_inventory))
        ack = qc.acknowledge_method_qc(quality, self.root/'verified-ack.json', 'reviewer')
        missing = inspect_reference_checkpoint(self.root/'missing-store', self.recipe, record)
        self.assertEqual(missing.status, InspectionStatus.MISSING)
        changed = qc.qc_from_stage_inspection(self.recipe, record, missing)
        with self.assertRaisesRegex(ValueError, 'stale'):
            qc.load_method_qc_acknowledgment(ack, changed)
        with patch('seedvision.benchmarking.checkpoint_inspection.load_checkpoint', side_effect=AssertionError('No historical probe')):
            self.assertEqual(qc.load_method_qc(path), quality)
        payload = quality.payload
        payload['identity']['recorded_inspection']['checkpoint_sha256'] = 'f'*64
        with self.assertRaisesRegex(ValueError, 'contradicts'):
            recalculate(payload)
