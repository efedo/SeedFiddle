"""Actual graph-ledger, interrupted publication and portable batch QC witnesses."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import shutil
import tempfile
import threading
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch

from seedvision.benchmarking import native_recipe as runner
from seedvision.benchmarking.native_correction_recipe import build_native_correction_recipe
from seedvision.export import method_qc as qc, recipe_attempt_qc as attempts, recipe_batch_qc as batches
from seedvision.export import project_native_graphs as graphs
from seedvision.persistence import (ProjectAnalysisStore, ProjectImageSpec,
    analysis_settings_profile_from_graph, method_qc_history as history, snapshot as snapshots)
from seedvision.pipeline import build_default_pipeline
from tests.test_native_recipe import fixture, selected_recipe
from tests.test_native_result_export import _attachment
from tests import test_project_correction_inputs as correction_fixtures


class _Fixture:
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root/'seeds.png'
        gray, _ = fixture()
        self.assertTrue(cv2.imwrite(str(self.path), np.repeat(gray[..., None], 3, axis=2)))
        self.source_sha = sha256(self.path.read_bytes()).hexdigest()
        self.recipe = runner.build_native_seed_recipe(selected_recipe())

    def export(self, name, attachments=None, **options):
        return graphs.export_project_native_graphs((_attachment(self.recipe),) if attachments is None else attachments,
            self.path, self.root/name, expected_source_sha256=self.source_sha, **options)

    def qualities(self, destination):
        captured = batches.load_project_native_graph_history(destination)
        self.assertFalse(captured.incomplete, captured.diagnostics)
        self.assertFalse(captured.diagnostics)
        return [qc.MethodQC(raw.decode('utf-8')) for raw in captured.records]

    def terminal(self, destination):
        qualities = self.qualities(destination)
        selected = [value for value in qualities if value.payload['identity']['scope'] == 'recipe_batch_attempt']
        self.assertEqual(len(selected), 1)
        return selected[0], qualities

    def cancel_after_export(self, destination, attachments=None, **options):
        stopped = threading.Event()
        original = graphs.export_native_recipe_result
        captured = []
        def export_then_stop(*args, **kwargs):
            report = original(*args, **kwargs)
            captured.append((report, (report.parent/qc.FILENAME).read_bytes()))
            stopped.set()
            return report
        with patch.object(graphs, 'export_native_recipe_result', side_effect=export_then_stop):
            with self.assertRaises(runner.NativeRecipeCancelled) as raised:
                self.export(destination, attachments, cancel=stopped.is_set, **options)
        return raised.exception, captured


class RecipeBatchQCTests(_Fixture, unittest.TestCase):
    def test_unavailable_original_and_missing_correction_binding_preserve_history_without_fallback(self):
        attachments = (_attachment(selected_recipe()), _attachment(build_native_correction_recipe()))
        before = [value.to_dict() for value in attachments]
        with patch.object(graphs, 'run_native_recipe_path', side_effect=AssertionError('No substitute executor')):
            ledger = self.export('unavailable', attachments)
        payload = json.loads(ledger.read_bytes())
        self.assertEqual([row['status'] for row in payload['graphs']], ['unavailable', 'unavailable'])
        self.assertEqual([value.to_dict() for value in attachments], before)
        self.assertEqual(list(ledger.parent.iterdir()), [ledger])
        for index, quality in enumerate(batches.read_project_graph_qc(ledger)):
            row = quality.payload['identity']['attempt_record']
            self.assertEqual((row['state'],row['phase']), ('unavailable','batch_selection'))
            self.assertIsNone(row['invoked_executor'])
            self.assertIsNone(quality.payload['identity']['native_artifact'])
            self.assertEqual(payload['graphs'][index]['historical_stage_runs'], before[index]['stage_runs'])

    def test_actual_corrupt_selected_image_records_failed_row_without_archive(self):
        self.path.write_bytes(b'Actual corrupt selected image bytes.')
        self.source_sha = sha256(self.path.read_bytes()).hexdigest()
        ledger = self.export('corrupt')
        row = json.loads(ledger.read_bytes())['graphs'][0]
        self.assertEqual(row['status'], 'failed')
        self.assertNotIn('WinError 5', row['reason'])
        quality = batches.load_project_graph_qc(ledger, 0)
        recorded = quality.payload['identity']['attempt_record']
        self.assertEqual((recorded['state'],recorded['phase'],recorded['result_returned']), ('failed','native_path',False))
        self.assertEqual(recorded['invoked_executor'], attempts.NATIVE_PATH_EXECUTOR)
        self.assertEqual(recorded['inputs_before'][0]['observation']['sha256'], self.source_sha)
        self.assertEqual(list(ledger.parent.iterdir()), [ledger])
        self.assertIsNone(quality.payload['identity']['node_outcomes'])

    def test_admission_rejection_preserves_prior_destination_and_separate_batch_association(self):
        destination = self.root/'prior'
        destination.mkdir()
        sentinel = destination/'prior.bin'
        sentinel.write_bytes(b'Already published unrelated data.')
        with patch.object(graphs, 'run_native_recipe_path', side_effect=AssertionError('No admission execution')):
            with self.assertRaises(FileExistsError) as raised:
                self.export('prior')
        quality, _ = self.terminal(destination)
        record = quality.payload['identity']['batch_record']
        self.assertEqual((record['phase'], record['archives_state'], record['completed_rows']), ('admission','not_created',[]))
        self.assertEqual(record['destination'], str(destination))
        self.assertIsNone(quality.payload['recipe'])
        self.assertEqual(raised.exception.batch_quality, quality)
        self.assertEqual(sentinel.read_bytes(), b'Already published unrelated data.')
        with self.assertRaises(ValueError):
            graphs.export_project_native_graphs((_attachment(self.recipe),), self.path, self.root/'wrong-source',
                                               expected_source_sha256='0'*64)
        other, _ = self.terminal(self.root/'wrong-source')
        self.assertEqual(other.payload['identity']['batch_record']['expected_source_sha256'], '0'*64)
        self.assertNotEqual(other.revision, quality.revision)

    def test_process_control_and_secondary_retention_failure_preserve_original_exception(self):
        with patch.object(graphs, 'run_native_recipe_path', side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.export('interrupt')
        self.assertFalse((self.root/'interrupt').exists())
        self.assertFalse((self.root/'interrupt.qc-history').exists())
        self.assertFalse(list(self.root.glob('.native-graphs-*')))
        with patch.object(graphs, 'retain_method_qc', side_effect=OSError('History publication failed.')):
            with self.assertRaises(runner.NativeRecipeCancelled) as raised:
                self.export('secondary', cancel=lambda: True)
        self.assertIn('History publication failed', raised.exception.__notes__[0])
        self.assertEqual(raised.exception.batch_quality.payload['identity']['batch_record']['phase'], 'between_rows')
        self.assertFalse((self.root/'secondary').exists())

    def test_legacy_rows_and_forged_embedded_links_never_invent_recorded_qc(self):
        ledger = self.export('legacy', (_attachment(selected_recipe()),))
        original = json.loads(ledger.read_bytes())
        row = original['graphs'][0]
        for mutate in (
            lambda value:value['method_qc'].update(raw_sha256='0'*64),
            lambda value:value['method_qc'].update(qc_revision='0'*64),
            lambda value:value['method_qc'].update(archive_path='../escaped/native-results.json'),
            lambda value:value.update(recipe_identity=self.recipe.identity.to_dict()),
            lambda value:value.update(status='executed'),
            lambda value:value.update(nodes=[{'stage_id':'invented output'}]),
        ):
            forged = deepcopy(original)
            mutate(forged['graphs'][0])
            path = self.root/'forged.json'
            path.write_text(json.dumps(forged), encoding='utf-8')
            with self.assertRaises(ValueError):
                batches.load_project_graph_qc(path, 0)
        legacy = deepcopy(original)
        legacy['graphs'][0].pop('method_qc')
        path = self.root/'legacy.json'
        path.write_text(json.dumps(legacy), encoding='utf-8')
        self.assertIsNone(batches.load_project_graph_qc(path, 0))
        for index in (-1, True, 1):
            with self.assertRaises(ValueError):
                batches.load_project_graph_qc(path, index)
        malformed = deepcopy(original)
        malformed['graphs'][0] = 1
        path.write_text(json.dumps(malformed), encoding='utf-8')
        with self.assertRaises(ValueError):
            batches.read_project_graph_qc(path)

    def test_actual_batch_row_viewer_loads_exact_unavailable_record_and_legacy_unknown(self):
        from PySide6.QtWidgets import QApplication
        from seedvision.ui.method_qc_dialog import MethodQCDialog
        app = QApplication.instance() or QApplication([])
        ledger = self.export('qt-unavailable', (_attachment(selected_recipe()),))
        view = MethodQCDialog()
        try:
            with patch('seedvision.ui.method_qc_dialog.QFileDialog.getOpenFileName', return_value=(str(ledger),'')), \
                 patch('seedvision.ui.method_qc_dialog.QInputDialog.getInt', return_value=(0,True)):
                view._load_batch_row()
            self.assertEqual(view.quality, batches.load_project_graph_qc(ledger, 0))
            self.assertIn('unavailable', view.summary.text())
            legacy = json.loads(ledger.read_bytes())
            legacy['graphs'][0].pop('method_qc')
            old = self.root/'old-ledger.json'
            old.write_text(json.dumps(legacy), encoding='utf-8')
            with patch('seedvision.ui.method_qc_dialog.QFileDialog.getOpenFileName', return_value=(str(old),'')), \
                 patch('seedvision.ui.method_qc_dialog.QInputDialog.getInt', return_value=(0,True)):
                view._load_batch_row()
            self.assertIsNone(view.quality)
            self.assertIn('remain unknown', view.status.text())
        finally:
            view.reject()
            view.deleteLater()
            app.processEvents()


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required for actual stored graph outcomes')
class ActualRecipeBatchQCTests(_Fixture, unittest.TestCase):
    def test_actual_success_links_exact_verified_sidecar_bytes_and_retains_project_owner(self):
        master = ProjectAnalysisStore(self.root).capture_and_save(
            analysis_settings=analysis_settings_profile_from_graph(build_default_pipeline()),
            images=(ProjectImageSpec(self.path, fixture()[0].shape),), native_recipes=(self.recipe,),
            destination=self.root/'owned.seedfiddle-project.json')
        ledger = self.export('success', project_root=self.root, project_path=master)
        payload = json.loads(ledger.read_bytes())
        self.assertEqual(payload['version'], 3)
        row = payload['graphs'][0]
        raw = (ledger.parent/Path(row['report']).parent/qc.FILENAME).read_bytes()
        self.assertEqual(row['method_qc']['payload_json'].encode('utf-8'), raw)
        self.assertEqual(row['method_qc']['raw_sha256'], sha256(raw).hexdigest())
        quality = batches.load_project_graph_qc(ledger, 0)
        self.assertEqual(quality.revision, row['method_qc']['qc_revision'])
        self.assertEqual(quality.payload['identity']['native_artifact']['status'], row['status'])
        retained = history.load_project_method_qc(self.root, master)
        self.assertIn(raw, retained.records)
        self.assertFalse(retained.diagnostics)

    def test_actual_selected_model_failure_records_binding_phase_without_native_archive(self):
        from tests.test_native_naive_bayes_recipe import _recipe_with_model
        from tests.test_plantcv_naive_bayes import _pdf
        model = self.root/'model.tsv'
        model.write_bytes(_pdf())
        recipe, _ = _recipe_with_model(model)
        model.write_bytes(b'Changed selected model bytes.')
        ledger = self.export('model-failure', (_attachment(recipe),))
        quality = batches.load_project_graph_qc(ledger, 0)
        row = quality.payload['identity']['attempt_record']
        self.assertEqual(row['state'], 'failed')
        self.assertEqual(row['invoked_executor'], attempts.NATIVE_PATH_EXECUTOR)
        self.assertFalse(row['result_returned'])
        observed = next(item for item in row['inputs_before'] if item['name'].startswith('model.'))
        self.assertEqual(observed['observation']['sha256'],sha256(model.read_bytes()).hexdigest())
        self.assertIsNone(quality.payload['identity']['native_artifact'])
        self.assertEqual(list(ledger.parent.iterdir()), [ledger])

    def test_cancellation_inside_generic_and_project_correction_rows_preserves_actual_attempt_route(self):
        calls = 0
        def cancel_inside():
            nonlocal calls
            calls += 1
            return calls >= 2
        with self.assertRaises(runner.NativeRecipeCancelled):
            self.export('inside-generic', cancel=cancel_inside)
        terminal, qualities = self.terminal(self.root/'inside-generic')
        record = terminal.payload['identity']['batch_record']
        self.assertEqual((record['phase'],record['current_row_index']), ('recipe_row',0))
        attempt = next(value for value in qualities if value.payload['identity']['scope'] == 'recipe_attempt')
        self.assertEqual(attempt.payload['identity']['attempt_record']['invoked_executor'], attempts.NATIVE_PATH_EXECUTOR)
        self.assertEqual(attempt.payload['identity']['attempt_record']['state'], 'cancelled')
        f = correction_fixtures.ProjectCorrectionInputTests()
        f.setUp()
        self.addCleanup(f.tearDown)
        _, document = f.project()
        calls = 0
        destination = f.root/'inside-correction'
        with self.assertRaises(runner.NativeRecipeCancelled):
            graphs.export_project_native_graphs((), f.source, destination,
                expected_source_sha256=document.images[0].source.sha256, project_root=f.workspace,
                native_recipes=(f.recipe,), correction_bindings=document.correction_bindings,
                project_image_record=document.images[0], cancel=cancel_inside)
        terminal, qualities = self.terminal(destination)
        attempt = next(value for value in qualities if value.payload['identity']['scope'] == 'recipe_attempt')
        row = attempt.payload['identity']['attempt_record']
        self.assertEqual((row['state'],row['phase'],row['invoked_executor']),
                         ('cancelled','correction_path',attempts.PROJECT_CORRECTION_EXECUTOR))
        self.assertEqual(len(row['inputs_before']), 4)
        self.assertFalse(row['result_returned'])
        self.assertFalse(list(f.root.glob('.native-graphs-*')))

    def test_between_rows_cancellation_preserves_success_qc_and_unexecuted_remaining_selection(self):
        error, captured = self.cancel_after_export('between', (_attachment(self.recipe), _attachment(selected_recipe())))
        quality, retained = self.terminal(self.root/'between')
        record = quality.payload['identity']['batch_record']
        self.assertEqual((record['state'],record['phase'],record['current_row_index']), ('cancelled','between_rows',None))
        self.assertEqual((len(record['requested_recipes']),len(record['completed_rows'])), (2,1))
        self.assertEqual(record['archives_state'], 'staging_removed')
        linked = record['completed_rows'][0]
        self.assertEqual(linked['qc_sha256'], sha256(captured[0][1]).hexdigest())
        original = qc.MethodQC(captured[0][1].decode('utf-8'))
        self.assertEqual(linked['status'], original.payload['identity']['native_artifact']['status'])
        self.assertIn(original, retained)
        self.assertEqual(error.batch_quality, quality)
        self.assertFalse(captured[0][0].exists())
        self.assertFalse((self.root/'between').exists())
        self.assertFalse(list(self.root.glob('.native-graphs-*')))

    def test_final_publication_cancellation_preserves_success_and_prior_concurrent_destination(self):
        stopped = threading.Event()
        destination = self.root/'final-stop'
        original = graphs.export_native_recipe_result
        def save_prior_and_stop(*args, **kwargs):
            report = original(*args, **kwargs)
            destination.mkdir()
            (destination/'prior.bin').write_bytes(b'Concurrent prior publication.')
            stopped.set()
            return report
        with patch.object(graphs,'export_native_recipe_result',side_effect=save_prior_and_stop):
            with self.assertRaises(runner.NativeRecipeCancelled):
                self.export('final-stop', cancel=stopped.is_set)
        quality, _ = self.terminal(destination)
        row = quality.payload['identity']['batch_record']
        self.assertEqual((row['phase'],row['archives_state'],row['publication_completed']),
                         ('batch_publication','staging_removed',False))
        self.assertEqual(len(row['completed_rows']), 1)
        self.assertEqual((destination/'prior.bin').read_bytes(), b'Concurrent prior publication.')
        self.assertFalse((destination/'native-graphs.json').exists())

    def test_actual_final_rename_failure_keeps_completed_recipe_and_prior_bytes(self):
        destination = self.root/'rename-failure'
        original = graphs.export_native_recipe_result
        def create_destination(*args, **kwargs):
            report = original(*args, **kwargs)
            destination.mkdir()
            (destination/'prior.bin').write_bytes(b'Prior destination is preserved.')
            return report
        with patch.object(graphs,'export_native_recipe_result',side_effect=create_destination):
            with self.assertRaises(OSError):
                self.export('rename-failure')
        quality, _ = self.terminal(destination)
        row = quality.payload['identity']['batch_record']
        self.assertEqual((row['state'],row['phase'],row['archives_state']), ('failed','batch_publication','staging_removed'))
        self.assertEqual(len(row['completed_rows']), 1)
        self.assertEqual((destination/'prior.bin').read_bytes(), b'Prior destination is preserved.')
        self.assertFalse(list(self.root.glob('.native-graphs-*')))

    def test_final_source_mutation_keeps_success_historical_and_new_bytes_observed(self):
        original = graphs.export_native_recipe_result
        def mutate_after_export(*args, **kwargs):
            report = original(*args, **kwargs)
            self.path.write_bytes(self.path.read_bytes()+b'changed after verified row')
            return report
        with patch.object(graphs,'export_native_recipe_result',side_effect=mutate_after_export):
            with self.assertRaises(ValueError):
                self.export('changed-source')
        quality, _ = self.terminal(self.root/'changed-source')
        row = quality.payload['identity']['batch_record']
        self.assertEqual(row['phase'], 'batch_publication')
        self.assertNotEqual(row['inputs_before'][0]['observation']['sha256'],row['inputs_after'][0]['observation']['sha256'])
        self.assertEqual(len(row['completed_rows']),1)
        self.assertFalse((self.root/'changed-source').exists())

    def test_exact_batch_ack_and_linked_rows_survive_private_snapshot_reopen_and_resnapshot(self):
        from PySide6.QtWidgets import QApplication
        from seedvision.ui.method_qc_dialog import MethodQCDialog
        app = QApplication.instance() or QApplication([])
        master = ProjectAnalysisStore(self.root).capture_and_save(
            analysis_settings=analysis_settings_profile_from_graph(build_default_pipeline()),
            images=(ProjectImageSpec(self.path, fixture()[0].shape),), native_recipes=(self.recipe,),
            destination=self.root/'portable.seedfiddle-project.json')
        self.cancel_after_export('portable-batch', project_root=self.root, project_path=master)
        quality, _ = self.terminal(self.root/'portable-batch')
        ack = self.root/'ack.json'
        qc.acknowledge_method_qc(quality,ack,'batch reviewer',reviewed_at_utc='2026-10-12T12:00:00+00:00')
        history.retain_method_qc(self.root,quality,project_path=master,
                                acknowledgment=qc.load_method_qc_acknowledgment(ack,quality))
        before = history.load_project_method_qc(self.root,master)
        package = snapshots.create_snapshot(self.root,master,self.root.parent/(self.root.name+'-snapshots'))
        self.addCleanup(shutil.rmtree,package.parent,True)
        shutil.rmtree(self.root)
        restored = self.root.parent/(self.root.name+'-restored')
        self.addCleanup(shutil.rmtree,restored,True)
        with patch.object(attempts,'_observe',side_effect=AssertionError('Historical source probe')):
            working = snapshots.restore_snapshot(package,restored)
            after = history.load_project_method_qc(restored,working)
            second = snapshots.create_snapshot(restored,working,package.parent)
            repeated = snapshots.restore_snapshot(second,restored)
            again = history.load_project_method_qc(restored,repeated)
            self.assertEqual((before.records,before.acknowledgments),(after.records,after.acknowledgments))
            self.assertEqual((before.records,before.acknowledgments),(again.records,again.acknowledgments))
            reopened = next(qc.MethodQC(raw.decode('utf-8')) for raw in again.records
                            if qc.MethodQC(raw.decode('utf-8')).revision == quality.revision)
            view = MethodQCDialog(reopened,project_root=restored,project_path=repeated)
            try:
                ack_path = restored/'historical-ack.json'
                ack_path.write_bytes(again.acknowledgments[0])
                view._install_acknowledgment(ack_path)
                self.assertIn('batch reviewer',view.status.text())
                self.assertIn('Recorded batch attempt: cancelled',view.summary.text())
                self.assertIn('not verified',view.summary.text())
            finally:
                view.reject(); view.deleteLater(); app.processEvents()

    def test_forged_batch_prefix_and_missing_exact_history_dependency_withhold_snapshot(self):
        master = ProjectAnalysisStore(self.root).capture_and_save(
            analysis_settings=analysis_settings_profile_from_graph(build_default_pipeline()),
            images=(ProjectImageSpec(self.path, fixture()[0].shape),), native_recipes=(self.recipe,),
            destination=self.root/'strict.seedfiddle-project.json')
        self.cancel_after_export('strict-batch', project_root=self.root,project_path=master)
        quality, _ = self.terminal(self.root/'strict-batch')
        for mutate in (
            lambda row:row['completed_rows'][0].update(index=1),
            lambda row:row.update(current_row_index=0),
            lambda row:row.update(publication_completed=True),
            lambda row:row['completed_rows'][0].update(archive_path='../outside/native-results.json'),
            lambda row:row.update(requested_recipes=[]),
        ):
            value = quality.payload
            mutate(value['identity']['batch_record'])
            value['qc_revision'] = sha256(qc._bytes(value['identity'])).hexdigest()
            with self.assertRaises(ValueError):
                qc.MethodQC(qc._bytes(value).decode('utf-8'))
        link = quality.payload['identity']['batch_record']['completed_rows'][0]
        linked_path = history.history_root(self.root,master)/'method-qc'/(link['qc_sha256']+'.json')
        linked_path.unlink()
        incomplete = history.load_project_method_qc(self.root,master)
        self.assertTrue(incomplete.incomplete)
        self.assertEqual(incomplete.records,())
        self.assertTrue(any('missing' in detail for detail in incomplete.diagnostics))
        with self.assertRaises(ValueError):
            snapshots.create_snapshot(self.root,master,self.root/'bad-package')
