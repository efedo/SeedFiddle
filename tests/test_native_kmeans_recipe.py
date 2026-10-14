"""Frozen-centre bindings, public predictions, native caches and Qt execution."""
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtCore import QEventLoop, QTimer, Qt
from PySide6.QtWidgets import QApplication

from seedvision.benchmarking import native_kmeans_recipe as factory
from seedvision.benchmarking import native_recipe as runner
from seedvision.benchmarking.checkpoints import CoordinateFrame
from seedvision.benchmarking.recipe_catalogue import build_recipe_catalogue
from seedvision.benchmarking.recipes import RecipeSpec
from seedvision.cuda import plantcv_kmeans_model as models
from seedvision.segmentation.baseline import PipelineAnalysisCache
from seedvision.ui.native_kmeans_dialog import NativeKMeansDialog
from tests.test_plantcv_kmeans_predict import REFERENCE, ROOT, SCRIPT


def _image():
    result = np.empty((23, 29, 3), np.uint8)
    result[:, :10] = (220, 30, 20)
    result[:, 10:20] = (20, 30, 220)
    result[:, 20:] = (210, 210, 210)
    return result


def _centres(channels=3):
    if channels == 1:
        return [[float(value)]*9 for value in (25, 100, 200)]
    return [list(colour)*9 for colour in ((220., 30., 20.), (20., 30., 220.), (210., 210., 210.))]


def _write_model(path, centres=None, channels=3, patch_size=3):
    path.write_text(json.dumps({'schema': models.SCHEMA, 'preprocessing': models.PREPROCESSING,
        'dtype': 'float64', 'patch_size': patch_size, 'channels': channels,
        'centres': _centres(channels) if centres is None else centres}), encoding='utf-8')


def _parameter(recipe, stage_id, name, value):
    stage = recipe.stage(stage_id)
    changed = replace(stage, parameters=tuple(replace(item, value=value)
        if item.parameter_id == name else item for item in stage.parameters))
    return replace(recipe, stages=tuple(changed if item.stage_id == stage_id else item
        for item in recipe.stages))


def _recipe(path):
    return _parameter(factory.build_native_kmeans_recipe(), factory.MODEL_STAGE_ID,
        factory.MODEL_PARAMETER_ID, factory.model_source_for_path(path))


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class NativeKMeansRecipeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        image = _image()
        requests = [dict(image=image.tolist(), patch_size=3, centres=_centres()),
            dict(image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).tolist(),
                patch_size=3, centres=_centres(1))]
        reference = subprocess.run([str(REFERENCE), '-c', SCRIPT], cwd=ROOT,
            input=json.dumps(requests), text=True, capture_output=True, timeout=90)
        if reference.returncode:
            raise AssertionError(reference.stderr)
        cls.labels = tuple(np.asarray(item['labels'], np.uint8) for item in json.loads(reference.stdout))

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.path = self.root/'centres.json'
        _write_model(self.path)
        self.recipe = _recipe(self.path)
        self.image_path = self.root/'seeds.png'
        self.assertTrue(cv2.imwrite(str(self.image_path), _image()))
        self.cache = PipelineAnalysisCache()
        self.prefix = f'reference.recipe:{self.recipe.recipe_id}:'

    def tearDown(self):
        self.temp.cleanup()

    def run_recipe(self, recipe=None, **kwargs):
        return runner.run_native_recipe_path(recipe or self.recipe, self.image_path,
            cache=self.cache, **kwargs)

    def test_roundtrip_identity_and_both_image_ports_preserve_historical_catalogue(self):
        restored = RecipeSpec.from_json(self.recipe.to_json())
        self.assertEqual(restored.identity, self.recipe.identity)
        self.assertNotIn(self.recipe.recipe_id, {item.recipe_id for item in build_recipe_catalogue().recipes})
        self.assertEqual(len(self.recipe.stages), 4)
        edges = {(edge.source_port, edge.target_port) for edge in self.recipe.dependencies
                 if edge.source_stage == factory.IMAGE_STAGE_ID}
        self.assertEqual(edges, {('gray_image', 'gray_image'), ('rgb_image', 'rgb_image')})
        source = self.recipe.parameters_for_runner(factory.MODEL_STAGE_ID)['model_source']
        model = factory.load_selected_model(source)
        identity = factory.selected_model_identity(source, model)
        self.assertEqual(identity['centres_sha256'], model.centres_sha256)
        self.assertEqual(identity['channels'], 3)
        with self.assertRaises(TypeError):
            identity['patch_size'] = 1
        with self.assertRaises(ValueError):
            factory.selected_model_identity({**source, 'source_bytes': 1}, model)
        original = runner._output_signatures({'model': model})
        forged = replace(model, centres=tuple(reversed(model.centres)))
        self.assertNotEqual(original, runner._output_signatures({'model': forged}))

    def test_public_bgr_and_gray_labels_masks_and_prepared_cache_reuse(self):
        for channels, expected in zip((3, 1), self.labels):
            with self.subTest(channels=channels):
                _write_model(self.path, channels=channels)
                recipe = _recipe(self.path)
                result = self.run_recipe(recipe)
                self.assertEqual(result.status, 'executed', [(n.stage_id, n.reason) for n in result.nodes])
                actual = result.node(factory.PREDICT_STAGE_ID).outputs['labels']
                np.testing.assert_array_equal(actual.cpu().numpy(), expected)
                masks = result.node(factory.MASK_STAGE_ID).outputs
                self.assertEqual(masks['class_names'], ('0', '1', '2'))
                for index in range(3):
                    np.testing.assert_array_equal(masks['class_masks'][str(index)].cpu().numpy(),
                        np.where(expected == index, 255, 0).astype(np.uint8))
                again = self.run_recipe(recipe)
                self.assertTrue(all(node.reused for node in again.nodes))
        np.testing.assert_array_equal(self.cache.values['reference.prepared'][1].gray.cpu().numpy(),
            cv2.cvtColor(_image(), cv2.COLOR_BGR2GRAY))
        self.assertGreater(len(np.unique(self.labels[0])), 1)

    def test_category_edit_and_nested_mutation_recompute_only_mask_node(self):
        initial = self.run_recipe()
        categories = [0, 0, 2, -1, 500]
        edited = _parameter(self.recipe, factory.MASK_STAGE_ID, 'cat_list', categories)
        selected = self.run_recipe(edited)
        self.assertEqual(self.cache.last_computed_nodes, (factory.MASK_STAGE_ID,))
        self.assertEqual(selected.node(factory.MASK_STAGE_ID).outputs['class_names'], ('selected',))
        expected = np.where(np.isin(self.labels[0], categories), 255, 0).astype(np.uint8)
        output = selected.node(factory.MASK_STAGE_ID).outputs['class_masks']['selected']
        np.testing.assert_array_equal(output.cpu().numpy(), expected)
        output.bitwise_not_()
        repaired = self.run_recipe(edited)
        self.assertEqual(self.cache.last_computed_nodes, (factory.MASK_STAGE_ID,))
        np.testing.assert_array_equal(
            repaired.node(factory.MASK_STAGE_ID).outputs['class_masks']['selected'].cpu().numpy(), expected)
        self.assertIs(initial.node(factory.PREDICT_STAGE_ID).outputs['labels'],
                      repaired.node(factory.PREDICT_STAGE_ID).outputs['labels'])
        for invalid in ([], [True], '0', list(range(257))):
            with self.subTest(categories=invalid), self.assertRaises(runner.NativeRecipeError):
                self.run_recipe(_parameter(self.recipe, factory.MASK_STAGE_ID, 'cat_list', invalid))

    def test_changed_model_clears_only_dependents_before_preparation_and_reselection(self):
        self.run_recipe()
        prepared = self.cache.values['reference.prepared']
        image_entry = self.cache.values[self.prefix+factory.IMAGE_STAGE_ID]
        _write_model(self.path, centres=list(reversed(_centres())))
        with patch.object(runner, 'prepare_native_recipe_inputs',
                side_effect=AssertionError('binding must precede preparation')):
            with self.assertRaises(runner.NativeRecipeError):
                self.run_recipe()
        self.assertIs(self.cache.values['reference.prepared'], prepared)
        self.assertIs(self.cache.values[self.prefix+factory.IMAGE_STAGE_ID], image_entry)
        for stage in (factory.MODEL_STAGE_ID, factory.PREDICT_STAGE_ID, factory.MASK_STAGE_ID):
            self.assertNotIn(self.prefix+stage, self.cache.values)
        rebound = _recipe(self.path)
        result = self.run_recipe(rebound)
        self.assertEqual(result.status, 'executed')
        self.assertEqual(self.cache.last_reused_nodes, (factory.IMAGE_STAGE_ID,))
        self.assertIs(self.cache.values['reference.prepared'], prepared)
        np.testing.assert_array_equal(result.node(factory.PREDICT_STAGE_ID).outputs['labels'].cpu().numpy(),
            2-self.labels[0])

    def test_direct_api_forged_models_and_missing_assets_reject_before_calculation(self):
        source = self.recipe.parameters_for_runner(factory.MODEL_STAGE_ID)['model_source']
        model = factory.load_selected_model(source)
        identity = factory.selected_model_identity(source, model)
        variants = ({factory.MODEL_STAGE_ID: (replace(model, centres_sha256='a'*64), identity)},
            {factory.MODEL_STAGE_ID: (model, {**identity, 'channels': 1})},
            {'unknown': (model, identity)})
        for external in variants:
            with self.assertRaises(runner.NativeRecipeError):
                runner.run_native_recipe(self.recipe, None, None, source_sha256='a'*64,
                    coordinate_frame=CoordinateFrame('test', (23, 29), 'top-left', 'x-right,y-down', 'pixel'),
                    preparation={'gray': 'test', 'rgb': 'test'}, external_inputs=external)
        self.path.unlink()
        with patch.object(runner, 'prepare_native_recipe_inputs', side_effect=AssertionError('no preparation')):
            with self.assertRaises(runner.NativeRecipeError):
                self.run_recipe()
            with self.assertRaises(runner.NativeRecipeError):
                self.run_recipe(factory.build_native_kmeans_recipe())

    def test_label_mutation_recomputes_prediction_and_releases_stale_masks(self):
        result = self.run_recipe()
        result.node(factory.PREDICT_STAGE_ID).outputs['labels'].bitwise_not_()
        del result
        observed = False
        def observe_eviction():
            nonlocal observed
            if (self.prefix+factory.PREDICT_STAGE_ID not in self.cache.values
                    and self.prefix+factory.MASK_STAGE_ID not in self.cache.values):
                observed = True
            return False
        repaired = self.run_recipe(cancel=observe_eviction)
        self.assertTrue(observed)
        self.assertEqual(self.cache.last_computed_nodes, (factory.PREDICT_STAGE_ID, factory.MASK_STAGE_ID))
        np.testing.assert_array_equal(repaired.node(factory.PREDICT_STAGE_ID).outputs['labels'].cpu().numpy(),
                                      self.labels[0])

    def test_cancellation_and_ambiguous_decision_publish_no_old_descendant(self):
        self.run_recipe()
        with self.assertRaises(runner.NativeRecipeCancelled):
            self.run_recipe(cancel=lambda: True)
        _write_model(self.path, centres=[[127.5], [128.5]], channels=1, patch_size=1)
        self.assertTrue(cv2.imwrite(str(self.image_path), np.full((23, 29, 3), 128, np.uint8)))
        failed = self.run_recipe(_recipe(self.path))
        self.assertEqual(failed.node(factory.PREDICT_STAGE_ID).status, 'failed')
        self.assertIn('KMeansDecisionAmbiguous', failed.node(factory.PREDICT_STAGE_ID).reason)
        self.assertEqual(failed.node(factory.MASK_STAGE_ID).status, 'not_run')
        self.assertFalse(failed.node(factory.MASK_STAGE_ID).outputs)
        self.assertNotIn(self.prefix+factory.MASK_STAGE_ID, self.cache.values)


class NativeKMeansDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.path = self.root/'centres.json'
        _write_model(self.path)
        self.image_path = self.root/'seeds.png'
        self.assertTrue(cv2.imwrite(str(self.image_path), _image()))
        self.dialog = NativeKMeansDialog(factory.build_native_kmeans_recipe())

    def tearDown(self):
        self.dialog.close()
        self.application.processEvents()
        self.temp.cleanup()

    def pick(self):
        with patch('seedvision.ui.native_naive_bayes_dialog.QFileDialog.getOpenFileName',
                return_value=(str(self.path), '')) as picker:
            self.dialog._choose_model()
        self.assertIn('*.json', picker.call_args.args[-1])

    def wait(self):
        loop, timer = QEventLoop(), QTimer()
        timer.timeout.connect(lambda: loop.quit() if self.dialog._worker is None else None)
        timer.start(10)
        QTimer.singleShot(15000, loop.quit)
        loop.exec()
        timer.stop()
        self.assertIsNone(self.dialog._worker, 'native worker did not finish')

    def test_picker_save_and_real_async_cluster_pixels_reset_and_union_edit(self):
        self.dialog.set_source_path(self.image_path)
        self.assertFalse(self.dialog.run_button.isEnabled())
        self.pick()
        self.assertTrue(self.dialog.run_button.isEnabled())
        self.assertIn('3 clusters', self.dialog.model_label.text())
        self.assertIn('3×3 patches', self.dialog.model_label.text())
        self.assertIn('BGR features', self.dialog.model_label.text())
        self.assertEqual(self.dialog.model_label.textFormat(), Qt.TextFormat.PlainText)
        self.assertFalse(self.dialog.graph.nodes[factory.MODEL_STAGE_ID].parameter_specs)
        self.dialog._select_stage(factory.MASK_STAGE_ID)
        self.dialog.start_run()
        self.wait()
        self.assertIsNotNone(self.dialog.result, self.dialog.status_label.text())
        self.assertEqual(self.dialog.result.status, 'executed', self.dialog.status_label.text())
        outputs = self.dialog.result.node(factory.MASK_STAGE_ID).outputs
        for index, name in enumerate(outputs['class_names']):
            self.dialog.class_selector.setCurrentIndex(index)
            np.testing.assert_array_equal(self.dialog._last_class_preview,
                outputs['class_masks'][name].cpu().numpy())
        self.dialog._select_stage(factory.PREDICT_STAGE_ID)
        self.assertTrue(self.dialog.class_selector.isHidden())
        self.dialog._parameter_changed(factory.MASK_STAGE_ID, 'cat_list', '[0, 2]')
        self.dialog._select_stage(factory.MASK_STAGE_ID)
        self.dialog.start_run()
        self.wait()
        self.assertEqual(self.dialog.class_selector.currentText(), 'selected')
        self.assertEqual(self.dialog.cache.last_computed_nodes, (factory.MASK_STAGE_ID,))
        saved_path = self.root/'saved.seedfiddle-recipe.json'
        with patch('seedvision.ui.native_recipe_dialog.QFileDialog.getSaveFileName',
                return_value=(str(saved_path), '')):
            self.dialog._save_recipe()
        saved = RecipeSpec.from_json(saved_path.read_text(encoding='utf-8'))
        self.assertEqual(saved.identity, self.dialog.recipe.identity)
        self.dialog._parameters_reset(factory.MODEL_STAGE_ID)
        self.assertIsNone(self.dialog.result)
        self.assertIsNone(self.dialog._selected_model_source())
        self.assertIsNone(self.dialog._last_class_preview)
        self.assertFalse(self.dialog.run_button.isEnabled())

    def test_async_cancellation_and_bad_picker_clear_previous_pixels(self):
        self.dialog.set_source_path(self.image_path)
        self.pick()
        self.dialog._select_stage(factory.MASK_STAGE_ID)
        self.dialog.start_run()
        self.wait()
        self.assertIsNotNone(self.dialog._last_class_preview)
        entered = threading.Event()
        def wait_for_stop(_recipe, _path, *, cache, cancel):
            entered.set()
            deadline = time.monotonic()+5
            while not cancel() and time.monotonic() < deadline:
                time.sleep(0.005)
            if cancel():
                raise runner.NativeRecipeCancelled('test cancellation')
            raise AssertionError('cancellation not delivered')
        with patch('seedvision.ui.native_recipe_dialog.run_native_recipe_path', side_effect=wait_for_stop):
            self.dialog.start_run()
            self.assertTrue(entered.wait(2))
            self.assertIsNone(self.dialog._last_class_preview)
            self.assertFalse(self.dialog.choose_model_button.isEnabled())
            self.dialog.stop()
            self.wait()
        self.assertIn('cancelled', self.dialog.status_label.text())
        self.assertTrue(self.dialog.choose_model_button.isEnabled())
        self.path.write_bytes(b'bad centres')
        self.pick()
        self.assertIsNone(self.dialog._selected_model_source())
        self.assertTrue(self.dialog.class_selector.isHidden())
        self.assertIsNone(self.dialog._last_class_preview)

    def test_launcher_preserves_historical_attachments_and_selection(self):
        from seedvision.ui.reference_graph_dialog import ReferenceGraphDialog
        dialog = ReferenceGraphDialog()
        try:
            before, selected = dialog.reference_graphs, dialog.recipe_selector.currentData()
            with patch('seedvision.ui.native_kmeans_dialog.NativeKMeansDialog') as view:
                view.return_value.exec.return_value = 0
                dialog._open_native_kmeans()
                view.assert_called_once()
            self.assertEqual(dialog.reference_graphs, before)
            self.assertEqual(dialog.recipe_selector.currentData(), selected)
        finally:
            dialog.close()
