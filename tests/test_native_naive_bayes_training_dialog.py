"""Actual Bayes training UI, atomic TSV saves and project-held provenance."""

from tests.catalogue_fixtures import CatalogueTestCase

from tests.functional_units import functional_units
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
@functional_units
class NativeBayesTrainingWorkflowTests(CatalogueTestCase):
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

    def check_test_legacy_and_trained_selections_roundtrip_exact_tsv_and_independent_metadata(self):
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

    def check_test_typed_provenance_rejects_mismatched_model_counts_profiles_and_calculation_facts(self):
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


    def check_test_constant_record_exact_facts_shapes_profiles_and_mean_overshoot(self):
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


    def check_test_canonical_provenance_byte_item_depth_and_numeric_bounds(self):
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




    def check_test_ui_completion_projects_compact_metadata_without_reading_dense_products(self):
        class DenseSentinel:
            def __getattribute__(self, name):
                raise AssertionError('UI inspected dense training products')
        dialog = self.dialog()
        dialog._run_path = str(self.samples.resolve())
        result = replace(self.completion, samples=DenseSentinel(), fit=DenseSentinel())
        self.assertEqual(dialog._validated_completion(result)['sample_counts'], [14,7])



    def check_test_real_temporary_save_commit_failure_and_short_write_preserve_destination(self):
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

    def check_test_post_commit_reload_failure_emits_no_selection(self):
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


    def check_test_parent_close_waits_for_modal_training_worker_and_never_selects_late_model(self):
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



    def test_workflow_lifecycle(self):
        """5 related NativeBayesTrainingWorkflowTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_legacy_and_trained_selections_roundtrip_exact_tsv_and_independent_metadata)
        self.scenario(self.check_test_constant_record_exact_facts_shapes_profiles_and_mean_overshoot)
        self.scenario(self.check_test_canonical_provenance_byte_item_depth_and_numeric_bounds)
        self.scenario(self.check_test_ui_completion_projects_compact_metadata_without_reading_dense_products)
        self.scenario(self.check_test_parent_close_waits_for_modal_training_worker_and_never_selects_late_model)

    def test_validation_and_recovery(self):
        """3 related NativeBayesTrainingWorkflowTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_typed_provenance_rejects_mismatched_model_counts_profiles_and_calculation_facts)
        self.scenario(self.check_test_real_temporary_save_commit_failure_and_short_write_preserve_destination)
        self.scenario(self.check_test_post_commit_reload_failure_emits_no_selection)


if __name__ == '__main__':
    unittest.main()
