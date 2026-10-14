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

    def test_recorded_reference_outcomes_keep_missing_facts_unknown_and_never_probe_sources(self):
        for state, status, reason in (
            (StageRunState.NOT_RUN, InspectionStatus.NOT_RUN, ''),
            (StageRunState.FAILED, InspectionStatus.RECORDED, 'Reference execution failed.'),
            (StageRunState.UNAVAILABLE, InspectionStatus.UNSUPPORTED, 'Executor unavailable.'),
            (StageRunState.EXECUTED, InspectionStatus.MISSING, ''),
            (StageRunState.EMPTY, InspectionStatus.MISMATCH, ''),
        ):
            with self.subTest(state=state):
                quality = self.quality(state, status, reason)
                identity = quality.payload['identity']
                self.assertEqual(identity['stage_record']['state'], state.value)
                self.assertEqual(identity['recorded_inspection']['status'], status.value)
                self.assertIsNone(identity['node_outcomes'][0]['output_ports'])
                for key in ('source_sha256', 'coordinate_frame', 'array_inventory', 'execution_status'):
                    self.assertIsNone(identity['recorded_inspection'][key])
                self.assertIsNone(identity['measurement_eligibility'])
                self.assertIsNone(identity['scientific_validity'])
                path = qc.export_method_qc(quality, self.root/(state.value+'.json'))
                with patch('seedvision.benchmarking.checkpoint_inspection.load_checkpoint', side_effect=AssertionError('Historical QC must not probe artifacts')):
                    self.assertEqual(qc.load_method_qc(path), quality)

    def test_strict_schema_revision_nonfinite_duplicates_and_bounds_reject(self):
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

    def test_acknowledgment_exact_revision_conflicts_and_approval_claims_reject(self):
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

    def test_layout_only_reference_changes_preserve_semantic_qc_identity(self):
        original = self.quality()
        moved = replace(self.recipe, layout=tuple(replace(p, x=p.x+20) for p in self.recipe.layout))
        record = initial_stage_run(moved, self.stage.stage_id)
        actual = qc.qc_from_stage_inspection(moved, record,
            CheckpointInspection(InspectionStatus.NOT_RUN, 'Recorded inspection fixture.'))
        self.assertEqual(original.revision, actual.revision)
        self.assertNotEqual(original.payload['recipe'], actual.payload['recipe'])


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

    def test_actual_native_nonempty_empty_and_failed_descendants_keep_states_and_ports(self):
        results = [self.run_recipe(), self.run_recipe(gray=torch.zeros_like(self.gray))]
        incomplete = runner.build_native_seed_recipe(selected_recipe(**{'filter.median_blur': {'ksize': None}}))
        results.append(self.run_recipe(recipe=incomplete))
        self.assertEqual(results[0].status, 'executed', [(n.stage_id,n.reason) for n in results[0].nodes])
        threshold = next(n for n in results[1].nodes if n.operation_id == 'threshold.binary')
        self.assertEqual(int(torch.count_nonzero(threshold.outputs['mask']).item()), 0)
        # The producer records executed zero masks and downstream count failures;
        # QC must preserve those facts rather than relabel the run as complete.
        self.assertTrue(any(n.status == 'failed' for n in results[2].nodes))
        self.assertTrue(any(n.status == 'not_run' for n in results[2].nodes))
        for index, result in enumerate(results):
            path, quality = self.export(result, str(index), incomplete if index == 2 else None)
            report, arrays = archives.load_native_recipe_result(path.parent, array_names=())
            self.assertEqual(arrays, {})
            identity = quality.payload['identity']
            self.assertEqual(identity['native_artifact']['result_revision'], report['result_revision'])
            self.assertEqual(identity['native_artifact']['execution_mode'], 'native_compatible')
            self.assertEqual(identity['native_artifact']['execution_backend'], 'seedvision.benchmarking.native_recipe')
            self.assertEqual(identity['node_outcomes'], [{'stage_id':n.stage_id, 'operation_id':n.operation_id,
                'state':n.status, 'reason':n.reason, 'output_ports':list(n.outputs)} for n in result.nodes])

    def test_swapped_forged_qc_and_unrequested_corrupt_arrays_reject_full_load(self):
        path, quality = self.export(self.run_recipe())
        other, alternate = self.export(self.run_recipe(gray=torch.zeros_like(self.gray)), 'empty')
        quality_path = path.parent/qc.FILENAME
        original = quality_path.read_bytes()
        quality_path.write_bytes((other.parent/qc.FILENAME).read_bytes())
        with self.assertRaisesRegex(ValueError, 'QC differs'):
            archives.load_native_recipe_result(path.parent, array_names=())
        payload = quality.payload
        payload['identity']['native_artifact']['result_revision'] = 'c'*64
        forged = recalculate(payload)
        quality_path.write_text(forged.payload_json, encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'QC differs'):
            qc.qc_from_native_archive(path.parent)
        quality_path.write_bytes(original)
        report, _ = archives.read_native_recipe_inventory(path.parent)
        archive = path.parent/'checkpoints'/(report['groups'][-1]['checkpoint_sha256']+'.checkpoint')/'arrays.npz'
        with np.load(archive, allow_pickle=False) as loaded:
            arrays = {name: loaded[name].copy() for name in loaded.files}
        value = next(value for value in arrays.values() if value.size)
        value.flat[0] = not value.flat[0] if value.dtype.kind == 'b' else value.flat[0]+1
        np.savez(archive, **arrays)
        with self.assertRaises(ValueError):
            qc.qc_from_native_archive(path.parent)

    def test_legacy_archive_without_sidecar_derives_only_verified_stored_facts(self):
        path, quality = self.export(self.run_recipe())
        (path.parent/qc.FILENAME).unlink()
        with self.assertRaises(FileNotFoundError):
            qc.load_method_qc(path.parent/qc.FILENAME)
        derived = qc.qc_from_native_archive(path.parent)
        self.assertEqual(derived, quality)
        self.assertIsNone(derived.payload['identity']['scientific_validity'])

    def test_qc_write_failure_and_cancellation_do_not_publish_incomplete_run(self):
        result = self.run_recipe()
        prior, _ = self.export(result)
        raw = prior.read_bytes()
        with patch.object(qc, 'export_method_qc', side_effect=OSError('QC disk failure')):
            with self.assertRaisesRegex(OSError, 'QC disk failure'):
                archives.export_native_recipe_result(self.root/'failed', self.recipe, result)
        self.assertFalse((self.root/'failed').exists())
        self.assertEqual(list(self.root.glob('.native-result-*')), [])
        with self.assertRaises(runner.NativeRecipeCancelled):
            archives.export_native_recipe_result(self.root/'cancelled', self.recipe, result, cancel=lambda:True)
        self.assertFalse((self.root/'cancelled').exists())
        self.assertEqual(prior.read_bytes(), raw)

    def test_actual_paper_crops_keep_paper_mode_and_source_frame(self):
        from tests.test_native_smpts_recipe import selected_recipe as paper_recipe
        from seedvision.benchmarking import native_smpts_recipe as factory
        recipe = paper_recipe()
        mask, rgb = contact_fixture()
        gray = np.where(mask, 220, 10).astype(np.uint8)
        gray_gpu, rgb_gpu = (torch.as_tensor(v, device='cuda') for v in (gray, rgb))
        frame = CoordinateFrame('source.decoded', mask.shape, 'top-left', 'x-right,y-down', 'pixel')
        result = self.run_recipe(gray=gray_gpu, recipe=recipe, rgb=rgb_gpu, frame=frame)
        path, quality = self.export(result, 'paper', recipe)
        identity = quality.payload['identity']
        self.assertEqual(identity['recipe_mode'], 'paper_reconstruction')
        self.assertEqual(identity['native_artifact']['execution_mode'], 'paper_reconstruction')
        self.assertEqual(identity['native_artifact']['execution_backend'], 'seedvision.benchmarking.native_recipe')
        self.assertEqual(identity['native_artifact']['coordinate_frame'], frame.to_dict())
        self.assertEqual(identity['node_outcomes'][-1]['stage_id'], factory.CROPS)
        report, _ = archives.load_native_recipe_result(path.parent, array_names=())
        self.assertEqual(report['reference_conformance'], 'not_evaluated_by_export')
        self.assertFalse(report['scientifically_validated'])
        self.assertIn('crops', identity['node_outcomes'][-1]['output_ports'])

    def test_actual_correction_archive_names_its_executor_and_rejects_generic_backend(self):
        from tests.test_native_correction_dialog import _write_inputs
        from seedvision.benchmarking.native_correction_recipe import build_native_correction_recipe
        from seedvision.benchmarking.native_correction_inputs import run_native_correction_files
        recipe = build_native_correction_recipe()
        paths = _write_inputs(self.root)
        result, prepared = run_native_correction_files(recipe, paths, cache=PipelineAnalysisCache())
        path, quality = self.export(result, 'correction', recipe)
        artifact = quality.payload['identity']['native_artifact']
        self.assertEqual(artifact['execution_backend'], 'seedvision.benchmarking.native_correction_recipe')
        self.assertEqual(artifact['source_sha256'], sha256(paths['source_rgb'].read_bytes()).hexdigest())
        self.assertEqual(artifact['coordinate_frame'], prepared.coordinate_frames['source_rgb'].to_dict())
        self.assertEqual(set(artifact['preparation']), {'target_rgb','target_mask','source_rgb','source_mask'})
        self.assertEqual(qc.qc_from_native_archive(path.parent), quality)
        payload = quality.payload
        payload['identity']['native_artifact']['execution_backend'] = 'seedvision.benchmarking.native_recipe'
        with self.assertRaisesRegex(ValueError, 'route'):
            recalculate(payload)


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
        from PySide6.QtWidgets import QApplication, QPushButton
        from seedvision.ui.reference_graph_dialog import ReferenceGraphDialog
        self.app = QApplication.instance() or QApplication([])
        from seedvision.persistence.reference_graphs import ReferenceGraphAttachment
        attachment = ReferenceGraphAttachment(self.recipe,
            tuple(initial_stage_run(self.recipe, s.stage_id) for s in self.recipe.stages), self.stage.stage_id)
        dialog = ReferenceGraphDialog(reference_graphs=(attachment,))
        self.addCleanup(dialog.deleteLater)
        dialog._show_stage(self.stage.stage_id)
        expected = dialog.active_attachment.recipe
        with patch('seedvision.ui.method_qc_dialog.MethodQCDialog.exec', return_value=0) as execute:
            dialog.findChild(QPushButton, 'referenceStageMethodQC').click()
            execute.assert_called_once()
        child = dialog.findChildren(__import__('seedvision.ui.method_qc_dialog', fromlist=['MethodQCDialog']).MethodQCDialog)[-1]
        self.assertEqual(child.quality.payload['identity']['recipe_identity'], expected.identity.analytical_dict())
        self.assertEqual(child.quality.payload['identity']['node_outcomes'][0]['stage_id'], self.stage.stage_id)
        self.assertEqual(len(child.quality.payload['identity']['node_outcomes']), 1)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class NativeMethodQCQtTests(unittest.TestCase):
    setUp = NativeMethodQCTests.setUp
    run_recipe = NativeMethodQCTests.run_recipe

    def test_actual_native_export_worker_and_qc_button_review_exact_stored_record(self):
        from PySide6.QtWidgets import QApplication
        from seedvision.ui.native_recipe_dialog import NativeRecipeDialog
        from seedvision.ui.method_qc_dialog import MethodQCDialog
        self.app = QApplication.instance() or QApplication([])
        dialog = NativeRecipeDialog(self.recipe)
        self.addCleanup(dialog.deleteLater)
        result = self.run_recipe()
        dialog._completed(result)
        with patch('seedvision.ui.native_recipe_dialog.QFileDialog.getExistingDirectory', return_value=str(self.root)):
            dialog.export_button.click()
        deadline = time.monotonic()+45
        while dialog._worker is not None and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(.01)
        self.assertIsNone(dialog._worker, dialog.status_label.text())
        self.assertIn('exported:', dialog.status_label.text())
        self.assertIs(dialog.result, result)
        self.assertIsNotNone(dialog._exported_quality)
        with patch.object(MethodQCDialog, 'exec', return_value=0) as execute:
            dialog.qc_button.click()
            execute.assert_called_once()
        child = dialog.findChildren(MethodQCDialog)[-1]
        self.assertEqual(child.quality, dialog._exported_quality)
        self.assertIsNone(child.acknowledgment)


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
