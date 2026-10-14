"""Actual Bayes training UI, atomic TSV saves and project-held provenance."""
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import cv2
import torch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtCore import QSaveFile, QTimer
from PySide6.QtWidgets import QApplication, QDialog

from seedvision.benchmarking import native_naive_bayes_recipe as factory
from seedvision.benchmarking import native_naive_bayes_training as training
from seedvision.benchmarking import native_recipe as runner
from seedvision.export import native_results as outputs
from seedvision.export.project_native_graphs import export_project_native_graphs
from seedvision.model_io import naive_bayes_provenance as provenance
from seedvision.persistence import reference_models as assets
from seedvision.persistence.analysis_settings import analysis_settings_profile_from_graph
from seedvision.persistence.project_analysis import (ProjectAnalysisStore, ProjectImageSpec,
    InvalidProjectAnalysis, ProjectAnalysisIOError)
from seedvision.pipeline import build_default_pipeline
from seedvision.segmentation.baseline import PipelineAnalysisCache
from seedvision.ui import native_naive_bayes_dialog as picker
from seedvision.ui import native_naive_bayes_training_dialog as ui
from tests.test_native_kmeans_recipe import _parameter
from tests.test_native_naive_bayes_training import sample_bytes, constant_value_samples
from tests.test_plantcv_naive_bayes import _image, _pdf


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def bind_recipe(selected):
    return _parameter(factory.build_native_naive_bayes_recipe(), factory.MODEL_STAGE_ID,
        factory.MODEL_PARAMETER_ID, selected)


def assert_constant_project_archive_after_removal(test, selected, source_paths):
    recipe = bind_recipe(selected)
    workspace,store,master = test.make_project(recipe=recipe)
    for path in source_paths:
        test.assertTrue(path.resolve().is_relative_to(test.root.resolve()))
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    moved = test.root/'moved-constant-workspace'
    test.assertTrue(workspace.resolve().is_relative_to(test.root.resolve()))
    test.assertTrue(moved.resolve().is_relative_to(test.root.resolve()))
    workspace.rename(moved)
    loaded = ProjectAnalysisStore(moved).load(moved/master.name)
    test.assertFalse(loaded.issues)
    saved = loaded.document.native_recipes[0]
    test.assertEqual(saved,recipe)
    paths = assets.model_asset_paths(moved,saved,loaded.document.reference_model_assets)
    result = runner.run_native_recipe_path(saved,moved/'seeds.png',model_asset_paths=paths)
    test.assertEqual(result.status,'executed')
    ledger = export_project_native_graphs((),moved/'seeds.png',test.root/'constant-batch',
        expected_source_sha256=loaded.document.images[0].source.sha256,project_root=moved,
        reference_model_assets=loaded.document.reference_model_assets,native_recipes=(saved,))
    row = json.loads(ledger.read_bytes())['graphs'][0]
    test.assertEqual(row['status'],'executed',row.get('reason'))
    report,arrays = outputs.load_native_recipe_result(ledger.parent/Path(row['report']).parent)
    archived = report['recipe']['stages'][1]['parameters'][0]['value']
    test.assertEqual(archived[factory.TRAINING_FIELD],selected[factory.TRAINING_FIELD])
    test.assertTrue(arrays)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class NativeBayesTrainingWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.samples = self.root/'sampled-rgb.tsv'
        self.samples.write_bytes(sample_bytes(7))
        self.completion = training.train_bayes_sample_file(self.samples)
        self.model_path = self.root/'trained.tsv'
        self.model_path.write_bytes(self.completion.tsv_bytes)
        self.selected = factory.model_source_for_trained_model(self.model_path,
            self.completion.tsv_bytes, self.completion.provenance_json)
        self.recipe = bind_recipe(self.selected)
        self.image_path = self.root/'seeds.png'
        self.assertTrue(cv2.imwrite(str(self.image_path), _image()))
        self.dialogs = []

    def tearDown(self):
        for dialog in self.dialogs:
            dialog.close()
            if hasattr(dialog, '_worker'):
                self.wait(dialog)
        self.application.processEvents()

    def dialog(self, **kwargs):
        dialog = ui.NativeNaiveBayesTrainingDialog(**kwargs)
        dialog.samples_edit.setText(str(self.samples))
        self.dialogs.append(dialog)
        return dialog

    def wait(self, dialog):
        deadline = time.monotonic()+20
        while dialog._worker is not None and time.monotonic() < deadline:
            self.application.processEvents()
            time.sleep(.005)
        self.assertIsNone(dialog._worker, 'training worker did not drain')

    def make_project(self, recipe=None):
        workspace = self.root/'workspace'
        workspace.mkdir()
        (workspace/'config').mkdir()
        shutil.copyfile(Path(__file__).parents[1]/'config/traits.json', workspace/'config/traits.json')
        shutil.copyfile(self.image_path, workspace/'seeds.png')
        store = ProjectAnalysisStore(workspace)
        master = store.capture_and_save(
            analysis_settings=analysis_settings_profile_from_graph(build_default_pipeline()),
            images=(ProjectImageSpec(workspace/'seeds.png', (23,29)),),
            native_recipes=(recipe or self.recipe,), destination=workspace/'test.seedfiddle-project.json')
        return workspace, store, master

    def test_legacy_and_trained_selections_roundtrip_exact_tsv_and_independent_metadata(self):
        plain = factory.model_source_for_path(self.model_path)
        self.assertEqual(set(plain), set(factory.MODEL_FIELDS))
        self.assertEqual(set(self.selected), set(factory.MODEL_FIELDS)|{factory.TRAINING_FIELD})
        self.assertEqual(factory.load_selected_model(plain), self.completion.model)
        self.assertEqual(factory.load_selected_model(self.selected), self.completion.model)
        record = provenance.validate_training_binding(self.selected[factory.TRAINING_FIELD], self.completion.model)
        independent = record.metadata
        independent['sample_counts'][0] += 100
        self.assertEqual(record.metadata['sample_counts'], [14,7])
        restored = type(self.recipe).from_json(self.recipe.to_json())
        self.assertEqual(restored, self.recipe)
        self.assertEqual(restored.identity.graph_sha256, factory.build_native_naive_bayes_recipe().identity.graph_sha256)
        with self.assertRaises(ValueError):
            factory.model_source_for_trained_model(self.model_path, _pdf(), self.completion.provenance_json)

    def test_typed_provenance_rejects_mismatched_model_counts_profiles_and_calculation_facts(self):
        original = json.loads(self.completion.provenance_json)
        bad = []
        for field, value in (('model_sha256', 'a'*64), ('tables_sha256', 'b'*64),
                ('class_names', ['background','seed']), ('sample_counts', [15,7]),
                ('scientific_validation', True), ('parser_id', 'different')):
            changed = deepcopy(original)
            changed[field] = value
            bad.append(changed)
        changed = deepcopy(original)
        changed['channel_fits'][0][0]['numerical_profile'] = 'certified exact'
        bad.append(changed)
        changed = deepcopy(original)
        changed['channel_fits'][0][0]['numerical_profile'] = provenance.NUMERICAL_PROFILE
        bad.append(changed)
        changed = deepcopy(original)
        changed['source_pins']['scipy'] = 'other'
        bad.append(changed)
        changed = deepcopy(original)
        changed['fit_plan']['live_bytes'] = 1
        bad.append(changed)
        changed = deepcopy(original)
        changed['calculation']['seedvision.cuda._nvrtc']['source_sha256'] = 'bad'
        bad.append(changed)
        changed = deepcopy(original)
        del changed['calculation']['seedvision.cuda.plantcv_range']['computed_constants']
        bad.append(changed)
        changed = deepcopy(original)
        changed['calculation']['seedvision.cuda.plantcv_range']['computed_constants'] = 'bad'
        bad.append(changed)
        changed = deepcopy(original)
        changed['calculation']['seedvision.cuda._nvrtc']['computed_constants'] = 'a'*64
        bad.append(changed)
        for value in bad:
            with self.subTest(keys=list(value)), self.assertRaises(ValueError):
                provenance.parse_training_record(canonical(value), self.completion.model)
        invalid = deepcopy(self.selected)
        invalid[factory.TRAINING_FIELD]['sha256'] = 'f'*64
        with patch.object(factory, 'load_plantcv_pdf', side_effect=AssertionError('read before metadata check')):
            with self.assertRaises(ValueError):
                factory.load_selected_model(invalid)

    def test_old_and_buffered_profiles_preserve_record_bytes_and_prediction_cache(self):
        current = json.loads(self.completion.provenance_json)
        self.assertEqual({channel['numerical_profile'] for channels in current['channel_fits']
            for channel in channels}, {provenance.BUFFERED_NUMERICAL_PROFILE})
        old = deepcopy(current)
        for channels in old['channel_fits']:
            for channel in channels:
                channel['numerical_profile'] = provenance.NUMERICAL_PROFILE
        old_bytes = canonical(old)
        old_record = provenance.parse_training_record(old_bytes, self.completion.model)
        self.assertEqual(old_record.json_text.encode('utf-8'), old_bytes)
        self.assertEqual(provenance.parse_training_record(
            old_record.json_text, self.completion.model), old_record)
        previous = factory.model_source_for_trained_model(self.model_path,
            self.completion.tsv_bytes, old_bytes)
        self.assertEqual(factory.load_selected_model(previous), self.completion.model)
        self.assertEqual(factory.load_selected_model(self.selected), self.completion.model)
        cache = PipelineAnalysisCache()
        first = runner.run_native_recipe_path(bind_recipe(previous), self.image_path, cache=cache)
        second = runner.run_native_recipe_path(self.recipe, self.image_path, cache=cache)
        self.assertTrue(all(node.reused for node in second.nodes))
        self.assertIs(first.node(factory.CLASSIFIER_STAGE_ID).outputs['class_masks'],
            second.node(factory.CLASSIFIER_STAGE_ID).outputs['class_masks'])
        refreshed = second.node(factory.MODEL_STAGE_ID).parameters['model_source'][factory.TRAINING_FIELD]
        self.assertEqual(refreshed, self.selected[factory.TRAINING_FIELD])

    def test_constant_record_exact_facts_shapes_profiles_and_mean_overshoot(self):
        self.samples.write_bytes(constant_value_samples())
        result = training.train_bayes_sample_file(self.samples)
        original = json.loads(result.provenance_json)
        channel = original['channel_fits'][0][2]
        self.assertEqual(channel['constant_value'],255)
        self.assertGreater(channel['mean'],255)
        self.assertEqual(channel['numerical_profile'],provenance.CONSTANT_NUMERICAL_PROFILE)
        self.assertEqual(provenance.parse_training_record(result.provenance_json,result.model).json_text.encode(),
            result.provenance_json)
        variants = []
        for key,value in (('constant_value',False),('constant_value',255.0),('constant_value',-1),
                ('constant_value',256),('constant_value',0),('mean',255.0),('mean',256.0),
                ('covariance',0.0),('covariance',channel['covariance']*2),
                ('covariance_absolute_budget',0.0),('pdf_absolute_budget',channel['pdf_absolute_budget']*2),
                ('bandwidth',channel['bandwidth']*2),('effective_samples',1.0),
                ('numerical_profile',provenance.BUFFERED_NUMERICAL_PROFILE),
                ('numerical_profile',provenance.NUMERICAL_PROFILE)):
            changed = deepcopy(original)
            changed['channel_fits'][0][2][key] = value
            variants.append(changed)
        changed = deepcopy(original)
        del changed['channel_fits'][0][2]['constant_value']
        variants.append(changed)
        changed = deepcopy(original)
        changed['channel_fits'][0][0]['constant_value'] = 255
        variants.append(changed)
        changed = deepcopy(original)
        changed['channel_fits'][0][0]['numerical_profile'] = provenance.NUMERICAL_PROFILE
        variants.append(changed)
        for changed in variants:
            with self.subTest(channel=changed['channel_fits'][0]),self.assertRaises(ValueError):
                provenance.parse_training_record(canonical(changed),result.model)

    def test_actual_constant_training_save_move_and_companion_archive_without_sources(self):
        self.samples.write_bytes(constant_value_samples())
        dialog = self.dialog()
        dialog._start_training()
        self.wait(dialog)
        self.assertIsNotNone(dialog._completion)
        self.assertGreater(dialog._completion.fit.channels[0][2].mean,255)
        destination = self.root/'constant-trained.tsv'
        with patch.object(ui.QFileDialog,'getSaveFileName',return_value=(str(destination),'tsv')):
            dialog._save_model()
        self.assertEqual(dialog.result(),QDialog.DialogCode.Accepted)
        self.assertEqual(destination.read_bytes(),dialog._completion.tsv_bytes)
        selected = dialog.saved_model_source
        self.assertEqual(factory.load_selected_model(selected),dialog._completion.model)
        assert_constant_project_archive_after_removal(self,selected,(self.samples,destination))

    def test_canonical_provenance_byte_item_depth_and_numeric_bounds(self):
        raw = self.completion.provenance_json
        invalid = [b'x'*(provenance.MAX_PROVENANCE_BYTES+1), b'\xff',
            json.dumps(json.loads(raw), indent=2).encode(),
            raw.replace(b'"scientific_validation":false', b'"scientific_validation":false,"scientific_validation":false'),
            b'{"nested":'+b'['*30+b'0'+b']'*30+b'}',
            b'{"number":NaN}', b'{"number":'+str(2**100).encode()+b'}']
        for source in invalid:
            with self.subTest(source=source[:35]), self.assertRaises(ValueError):
                provenance.parse_training_record(source, self.completion.model)
        with patch.object(provenance, 'MAX_METADATA_ITEMS', 3), self.assertRaises(ValueError):
            provenance.parse_training_record(raw, self.completion.model)

    def test_metadata_only_origin_change_reuses_predictions_and_refreshes_node_parameters(self):
        other_samples = self.root/'other-sampled-rgb.tsv'
        other_samples.write_bytes(self.samples.read_bytes())
        second = training.train_bayes_sample_file(other_samples)
        self.assertEqual(second.tsv_bytes, self.completion.tsv_bytes)
        selected = factory.model_source_for_trained_model(self.model_path, second.tsv_bytes, second.provenance_json)
        changed = bind_recipe(selected)
        self.assertNotEqual(self.recipe.identity.settings_sha256, changed.identity.settings_sha256)
        cache = PipelineAnalysisCache()
        first = runner.run_native_recipe_path(self.recipe, self.image_path, cache=cache)
        result = runner.run_native_recipe_path(changed, self.image_path, cache=cache)
        self.assertTrue(all(node.reused for node in result.nodes))
        self.assertIs(first.node(factory.CLASSIFIER_STAGE_ID).outputs['class_masks'],
            result.node(factory.CLASSIFIER_STAGE_ID).outputs['class_masks'])
        current = result.node(factory.MODEL_STAGE_ID).parameters['model_source'][factory.TRAINING_FIELD]['json']
        self.assertEqual(json.loads(current)['sample_origin_path'], str(other_samples.resolve()))
        self.assertEqual(json.loads(first.node(factory.MODEL_STAGE_ID).parameters['model_source'][factory.TRAINING_FIELD]['json'])
            ['sample_origin_path'], str(self.samples.resolve()))
        outputs.export_native_recipe_result(self.root/'updated-history-run', changed, result)
        report, _ = outputs.load_native_recipe_result(self.root/'updated-history-run')
        archived = report['recipe']['stages'][1]['parameters'][0]['value'][factory.TRAINING_FIELD]
        self.assertEqual(json.loads(archived['json'])['sample_origin_path'], str(other_samples.resolve()))
        broken = deepcopy(selected)
        broken[factory.TRAINING_FIELD]['sha256'] = 'a'*64
        with patch.object(runner, 'prepare_native_recipe_inputs', side_effect=AssertionError('prepared before metadata check')):
            with self.assertRaises(ValueError):
                runner.run_native_recipe_path(bind_recipe(broken), self.image_path, cache=cache)
        self.assertIn('reference.recipe:'+self.recipe.recipe_id+':'+factory.IMAGE_STAGE_ID, cache.values)
        self.assertNotIn('reference.recipe:'+self.recipe.recipe_id+':'+factory.CLASSIFIER_STAGE_ID, cache.values)

    def test_loaded_provenance_codec_body_is_verified_before_numerical_cache_reuse(self):
        legacy = bind_recipe(factory.model_source_for_path(self.model_path))
        cache = PipelineAnalysisCache()
        runner.run_native_recipe_path(legacy, self.image_path, cache=cache)
        with patch.object(provenance, '_bounded_metadata', lambda value: None):
            with self.assertRaisesRegex(runner.NativeRecipeError, 'Loaded calculation'):
                runner.run_native_recipe_path(legacy, self.image_path, cache=cache)

    def test_real_async_training_ordered_facts_atomic_save_and_exact_reload(self):
        dialog = self.dialog()
        dialog._start_training()
        for widget in (dialog.samples_edit, dialog.choose_samples_button, dialog.train_button, dialog.save_button):
            self.assertFalse(widget.isEnabled())
        self.wait(dialog)
        self.assertTrue(dialog.save_button.isEnabled(), dialog.status_label.text())
        self.assertIn('seed: 14, background: 7', dialog.status_label.text())
        self.assertIn('empirical', dialog.status_label.text())
        self.assertIn('Keep', dialog.status_label.text())
        path = self.root/'saved-by-qt.tsv'
        with patch.object(ui.QFileDialog, 'getSaveFileName', return_value=(str(path), '')):
            dialog._save_model()
        self.assertEqual(path.read_bytes(), self.completion.tsv_bytes)
        self.assertEqual(factory.load_selected_model(dialog.saved_model_source), self.completion.model)
        self.assertEqual(dialog.saved_model_path, str(path.resolve()))
        self.assertEqual(dialog.result(), QDialog.DialogCode.Accepted)
        self.assertEqual(dialog.saved_model_source[factory.TRAINING_FIELD]['json'],
            dialog._completion.provenance_json.decode())

    def test_ui_completion_projects_compact_metadata_without_reading_dense_products(self):
        class DenseSentinel:
            def __getattribute__(self, name):
                raise AssertionError('UI inspected dense training products')
        dialog = self.dialog()
        dialog._run_path = str(self.samples.resolve())
        result = replace(self.completion, samples=DenseSentinel(), fit=DenseSentinel())
        self.assertEqual(dialog._validated_completion(result)['sample_counts'], [14,7])

    def test_stop_close_and_stale_selection_discard_successful_late_results(self):
        for action in ('stop', 'reject', 'accept', 'close', 'change'):
            with self.subTest(action=action):
                entered, release = threading.Event(), threading.Event()
                def delayed(path, *, cancel):
                    entered.set()
                    release.wait(5)
                    return self.completion
                dialog = self.dialog(operation=delayed)
                dialog._start_training()
                self.assertTrue(entered.wait(5))
                finished = []
                dialog.finished.connect(finished.append)
                if action == 'stop':
                    dialog._stop_training()
                elif action == 'change':
                    dialog.samples_edit.setText(str(self.root/'new.tsv'))
                else:
                    getattr(dialog, action)()
                    self.assertEqual(finished, [])
                release.set()
                self.wait(dialog)
                self.assertIsNone(dialog._completion)
                self.assertIsNone(dialog.saved_model_source)
                self.assertFalse(dialog.save_button.isEnabled())
                if action in ('reject','accept','close'):
                    self.assertEqual(finished, [QDialog.DialogCode.Rejected])

    def test_same_path_reselection_and_failed_training_clear_stale_save(self):
        dialog = self.dialog()
        dialog._completion = self.completion
        dialog.save_button.setEnabled(True)
        with patch.object(ui.QFileDialog, 'getOpenFileName', return_value=(str(self.samples), '')):
            dialog._choose_samples()
        self.assertIsNone(dialog._completion)
        self.assertFalse(dialog.save_button.isEnabled())
        self.samples.write_bytes(b'constant\n0,0,0\n0,0,0\n')
        dialog._start_training()
        self.wait(dialog)
        self.assertIn('constant HSV', dialog.status_label.text())
        self.assertIsNone(dialog._completion)
        self.assertFalse(dialog.save_button.isEnabled())
        dialog._operation = lambda path, cancel: replace(self.completion, provenance_json=b'{}')
        dialog._start_training()
        self.wait(dialog)
        self.assertIsNone(dialog._completion)
        self.assertIn('failed', dialog.status_label.text())

    def test_real_temporary_save_commit_failure_and_short_write_preserve_destination(self):
        dialog = self.dialog()
        dialog._run_path = str(self.samples.resolve())
        dialog._completion = self.completion
        destination = self.root/'existing.tsv'
        destination.write_bytes(b'existing model bytes')
        for mode in ('commit', 'short'):
            class FailedSave:
                def __init__(self, path):
                    self.target = QSaveFile(path)
                def open(self, flags): return self.target.open(flags)
                def write(self, payload):
                    written = self.target.write(payload)
                    return written-1 if mode == 'short' else written
                def commit(self): return False
                def errorString(self): return 'injected save failure'
                def cancelWriting(self): self.target.cancelWriting()
            with patch.object(ui, 'QSaveFile', FailedSave), \
                    patch.object(ui.QFileDialog, 'getSaveFileName', return_value=(str(destination), '')), \
                    patch.object(ui.QMessageBox, 'warning') as warning:
                dialog._save_model()
            warning.assert_called_once()
            self.assertEqual(destination.read_bytes(), b'existing model bytes')
            self.assertIsNone(dialog.saved_model_source)
            self.assertEqual(list(self.root.glob('*.incomplete')), [])

    def test_post_commit_reload_failure_emits_no_selection(self):
        dialog = self.dialog()
        dialog._run_path = str(self.samples.resolve())
        dialog._completion = self.completion
        destination = self.root/'committed.tsv'
        with patch.object(ui.QFileDialog, 'getSaveFileName', return_value=(str(destination), '')), \
                patch.object(ui, 'model_source_for_trained_model', side_effect=ValueError('changed after commit')), \
                patch.object(ui.QMessageBox, 'warning'):
            dialog._save_model()
        self.assertEqual(destination.read_bytes(), self.completion.tsv_bytes)
        self.assertIsNone(dialog.saved_model_source)
        self.assertIsNone(dialog.saved_model_path)

    def test_parent_picker_keeps_training_record_and_reuses_independent_image_cache(self):
        old_path = self.root/'old.tsv'
        old_path.write_bytes(_pdf())
        kept = []
        dialog = picker.NativeNaiveBayesDialog(bind_recipe(factory.model_source_for_path(old_path)), keep_in_project=kept.append)
        self.dialogs.append(dialog)
        first = runner.run_native_recipe_path(dialog.recipe, self.image_path, cache=dialog.cache)
        selected = deepcopy(self.selected)
        class CompletedDialog:
            saved_model_source = selected
            def __init__(self, parent): pass
            def exec(self): return QDialog.DialogCode.Accepted
            def deleteLater(self): pass
        with patch.object(picker, 'NativeNaiveBayesTrainingDialog', CompletedDialog):
            dialog._train_model()
        self.assertIn('training provenance', dialog.model_label.text())
        self.assertIn(dialog.train_model_button, dialog._export_conflicting_buttons())
        dialog._keep_recipe_in_project()
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0].parameters_for_runner(factory.MODEL_STAGE_ID)['model_source'], self.selected)
        after = runner.run_native_recipe_path(dialog.recipe, self.image_path, cache=dialog.cache)
        self.assertEqual(dialog.cache.last_reused_nodes, (factory.IMAGE_STAGE_ID,))
        self.assertEqual(dialog.cache.last_computed_nodes, (factory.MODEL_STAGE_ID, factory.CLASSIFIER_STAGE_ID))
        self.assertIs(first.node(factory.IMAGE_STAGE_ID).outputs['rgb_image'], after.node(factory.IMAGE_STAGE_ID).outputs['rgb_image'])

    def test_parent_close_waits_for_modal_training_worker_and_never_selects_late_model(self):
        parent = picker.NativeNaiveBayesDialog(self.recipe)
        self.dialogs.append(parent)
        release = threading.Event()
        def delayed(path, *, cancel):
            release.wait(5)
            return self.completion
        child = ui.NativeNaiveBayesTrainingDialog(parent, operation=delayed)
        child.samples_edit.setText(str(self.samples))
        finished = []
        parent.finished.connect(lambda code: finished.append((code, child._worker)))
        def start_and_close():
            child._start_training()
            parent.reject()
            self.assertEqual(finished, [])
            release.set()
        QTimer.singleShot(0, start_and_close)
        with patch.object(picker, 'NativeNaiveBayesTrainingDialog', return_value=child):
            parent._train_model()
        self.assertEqual(finished, [(QDialog.DialogCode.Rejected, None)])
        self.assertIsNone(child.saved_model_source)
        self.assertEqual(parent.recipe, self.recipe)
        self.assertIsNone(child._completion)

    def test_project_move_reopen_and_actual_batch_archive_retain_record_without_originals(self):
        workspace, store, master = self.make_project()
        self.samples.unlink()
        self.model_path.unlink()
        moved = self.root/'moved-workspace'
        self.assertTrue(workspace.resolve().is_relative_to(self.root.resolve()))
        self.assertTrue(moved.resolve().is_relative_to(self.root.resolve()))
        workspace.rename(moved)
        master = moved/master.name
        loaded = ProjectAnalysisStore(moved).load(master)
        self.assertFalse(loaded.issues)
        recipe = loaded.document.native_recipes[0]
        self.assertEqual(recipe, self.recipe)
        paths = assets.model_asset_paths(moved, recipe, loaded.document.reference_model_assets)
        result = runner.run_native_recipe_path(recipe, moved/'seeds.png', model_asset_paths=paths)
        self.assertEqual(result.status, 'executed')
        ledger = export_project_native_graphs((), moved/'seeds.png', self.root/'batch',
            expected_source_sha256=loaded.document.images[0].source.sha256, project_root=moved,
            reference_model_assets=loaded.document.reference_model_assets, native_recipes=(recipe,))
        row = json.loads(ledger.read_bytes())['graphs'][0]
        self.assertEqual(row['status'], 'executed', row.get('reason'))
        self.assertEqual(row['historical_stage_runs'], [])
        report, arrays = outputs.load_native_recipe_result(ledger.parent/Path(row['report']).parent)
        selected = report['recipe']['stages'][1]['parameters'][0]['value']
        self.assertEqual(selected[factory.TRAINING_FIELD], self.selected[factory.TRAINING_FIELD])
        self.assertTrue(arrays)
        self.assertFalse(report['scientifically_validated'])

    def test_invalid_training_metadata_and_project_publication_failure_preserve_previous_master(self):
        workspace, store, master = self.make_project()
        before = master.read_bytes()
        document = store.load(master).document
        malformed = deepcopy(self.selected)
        malformed[factory.TRAINING_FIELD]['sha256'] = 'a'*64
        with patch.object(assets, 'verified_asset_bytes', side_effect=AssertionError('read before metadata check')):
            with self.assertRaises(InvalidProjectAnalysis):
                store.capture_and_save(analysis_settings=document.analysis_settings,
                    images=(ProjectImageSpec(workspace/'seeds.png',(23,29)),),
                    native_recipes=(bind_recipe(malformed),), destination=master)
        self.assertEqual(master.read_bytes(), before)
        changed = json.loads(self.completion.provenance_json)
        changed['sample_origin_path'] = str(self.root/'another-origin.tsv')
        selected = factory.model_source_for_trained_model(self.model_path,
            self.completion.tsv_bytes, canonical(changed))
        updated = replace(document, native_recipes=(bind_recipe(selected),), reference_model_assets=())
        with patch('seedvision.persistence.project_analysis.os.replace', side_effect=OSError('project commit failed')) as publish:
            with self.assertRaises(ProjectAnalysisIOError):
                store.save(updated, master)
        publish.assert_called_once()
        self.assertEqual(master.read_bytes(), before)
        self.assertEqual(store.load(master).document.native_recipes, document.native_recipes)


if __name__ == '__main__':
    unittest.main()
