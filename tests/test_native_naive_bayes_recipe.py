"""Pinned class masks, asset bindings, cache invalidation and desktop execution."""
from dataclasses import replace
from types import MappingProxyType, SimpleNamespace
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtCore import Qt, QEventLoop, QTimer
from PySide6.QtWidgets import QApplication

from seedvision.benchmarking.checkpoints import CoordinateFrame
from seedvision.benchmarking.native_naive_bayes_recipe import (
    CLASSIFIER_STAGE_ID, IMAGE_STAGE_ID, MODEL_PARAMETER_ID, MODEL_STAGE_ID,
    build_native_naive_bayes_recipe, load_selected_model, model_source_for_path,
    selected_model_identity,
)
from seedvision.benchmarking.native_recipe import (
    NativeRecipeCancelled, NativeRecipeError, _nested_cuda_storage, _output_signatures,
    run_native_recipe, run_native_recipe_path,
)
from seedvision.benchmarking.recipe_catalogue import build_recipe_catalogue
from seedvision.benchmarking.recipes import RecipeSpec
from seedvision.segmentation.baseline import PipelineAnalysisCache
from seedvision.ui.native_naive_bayes_dialog import NativeNaiveBayesDialog
from tests.test_plantcv_naive_bayes import _constant_pdf, _image, _pdf, _reference

FRAME = CoordinateFrame('naive-bayes.test', (23, 29), 'top-left', 'x-right,y-down', 'pixel')
PREPARATION = {'gray': 'Synthetic uint8 fixture; classifier consumes RGB boundary only.',
    'rgb': 'OpenCV IMREAD_COLOR BGR channels reversed on CUDA to RGB.'}


def _recipe_with_model(path):
    source = model_source_for_path(path)
    recipe = build_native_naive_bayes_recipe()
    stage = recipe.stage(MODEL_STAGE_ID)
    changed = replace(stage, parameters=tuple(
        replace(item, value=source) if item.parameter_id == MODEL_PARAMETER_ID else item
        for item in stage.parameters))
    return replace(recipe, stages=tuple(changed if item.stage_id == MODEL_STAGE_ID else item
        for item in recipe.stages)), source


class NativeNaiveBayesRecipeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.model_path = self.root/'plantcv-model.tsv'
        self.model_bytes = _pdf()
        self.model_path.write_bytes(self.model_bytes)
        self.recipe, self.source = _recipe_with_model(self.model_path)
        self.image = _image()
        self.image_path = self.root/'seed.png'
        self.assertTrue(cv2.imwrite(str(self.image_path), self.image))

    def tearDown(self):
        self.temp.cleanup()

    def test_recipe_roundtrip_and_immutable_identity_are_separate_from_historical_catalogue(self):
        restored = RecipeSpec.from_json(self.recipe.to_json())
        self.assertEqual(restored.identity, self.recipe.identity)
        self.assertNotIn(self.recipe.recipe_id, {item.recipe_id for item in build_recipe_catalogue().recipes})
        model = load_selected_model(self.source)
        identity = selected_model_identity(self.source, model)
        self.assertEqual(identity['class_names'], model.class_names)
        with self.assertRaises(TypeError):
            identity['class_names'][0] = 'changed'
        bad_selection = {**self.source, 'source_bytes': self.source['source_bytes']+1}
        with self.assertRaises(ValueError):
            selected_model_identity(bad_selection, model)
        other_path = self.root/'other.tsv'
        other_path.write_bytes(_constant_pdf({'leaf': 0.001, 'soil': 0.1, 'shadow': 0.9}))
        other, _ = _recipe_with_model(other_path)
        self.assertNotEqual(other.identity.settings_sha256, self.recipe.identity.settings_sha256)
        self.assertEqual(other.identity.graph_sha256, self.recipe.identity.graph_sha256)

    def test_actual_class_masks_match_public_call_and_nested_mutation_recomputes_only_classifier(self):
        cache = PipelineAnalysisCache()
        first = run_native_recipe_path(self.recipe, self.image_path, cache=cache)
        self.assertEqual(first.status, 'executed', [(n.stage_id, n.reason) for n in first.nodes])
        reference = _reference(self.image, self.model_bytes)
        actual = first.node(CLASSIFIER_STAGE_ID).outputs
        self.assertEqual(tuple(actual['class_names']), tuple(reference))
        for name in actual['class_names']:
            np.testing.assert_array_equal(actual['class_masks'][name].cpu().numpy(), reference[name])
        second = run_native_recipe_path(self.recipe, self.image_path, cache=cache)
        self.assertTrue(all(node.reused for node in second.nodes))
        second.node(CLASSIFIER_STAGE_ID).outputs['class_masks']['leaf'].bitwise_not_()
        repaired = run_native_recipe_path(self.recipe, self.image_path, cache=cache)
        self.assertEqual(cache.last_computed_nodes, (CLASSIFIER_STAGE_ID,))
        self.assertEqual(cache.last_reused_nodes, (IMAGE_STAGE_ID, MODEL_STAGE_ID))
        for name in reference:
            np.testing.assert_array_equal(
                repaired.node(CLASSIFIER_STAGE_ID).outputs['class_masks'][name].cpu().numpy(), reference[name])

    def test_reselecting_changed_model_preserves_prepared_image_and_image_node(self):
        cache = PipelineAnalysisCache()
        run_native_recipe_path(self.recipe, self.image_path, cache=cache)
        prepared = cache.values['reference.prepared']
        replacement = _constant_pdf({'leaf': 0.001, 'soil': 0.1, 'shadow': 0.9})
        self.model_path.write_bytes(replacement)
        rebound, source = _recipe_with_model(self.model_path)
        self.assertNotEqual(source['source_sha256'], self.source['source_sha256'])
        result = run_native_recipe_path(rebound, self.image_path, cache=cache)
        self.assertIs(cache.values['reference.prepared'], prepared)
        self.assertEqual(cache.last_reused_nodes, (IMAGE_STAGE_ID,))
        self.assertEqual(cache.last_computed_nodes, (MODEL_STAGE_ID, CLASSIFIER_STAGE_ID))
        expected = _reference(self.image, replacement)
        for name in expected:
            np.testing.assert_array_equal(
                result.node(CLASSIFIER_STAGE_ID).outputs['class_masks'][name].cpu().numpy(), expected[name])

    def test_changed_file_rejected_before_image_cache_hit_and_retains_unaffected_image_node(self):
        cache = PipelineAnalysisCache()
        run_native_recipe_path(self.recipe, self.image_path, cache=cache)
        prefix = f'reference.recipe:{self.recipe.recipe_id}:'
        image_entry = cache.values[prefix+IMAGE_STAGE_ID]
        self.model_path.write_bytes(_constant_pdf({'leaf': 0.001, 'soil': 0.1, 'shadow': 0.9}))
        with patch('seedvision.benchmarking.native_recipe.prepare_native_recipe_inputs',
                side_effect=AssertionError('image preparation must not occur')):
            with self.assertRaises(NativeRecipeError):
                run_native_recipe_path(self.recipe, self.image_path, cache=cache)
        self.assertIn('reference.prepared', cache.values)
        self.assertIs(cache.values[prefix+IMAGE_STAGE_ID], image_entry)
        self.assertNotIn(prefix+MODEL_STAGE_ID, cache.values)
        self.assertNotIn(prefix+CLASSIFIER_STAGE_ID, cache.values)
        rebound, _ = _recipe_with_model(self.model_path)
        result = run_native_recipe_path(rebound, self.image_path, cache=cache)
        self.assertEqual(result.status, 'executed')
        self.assertEqual(cache.last_reused_nodes, (IMAGE_STAGE_ID,))

    def test_missing_invalid_and_cancelled_model_fail_before_image_preparation(self):
        empty = build_native_naive_bayes_recipe()
        with patch('seedvision.benchmarking.native_recipe.prepare_native_recipe_inputs',
                side_effect=AssertionError('must reject before image preparation')):
            with self.assertRaises(NativeRecipeError):
                run_native_recipe_path(empty, self.image_path)
            with self.assertRaises(NativeRecipeCancelled):
                run_native_recipe_path(self.recipe, self.image_path, cancel=lambda: True)
            self.model_path.write_bytes(b'not a probability table')
            with self.assertRaises(NativeRecipeError):
                run_native_recipe_path(self.recipe, self.image_path)
            self.model_path.unlink()
            with self.assertRaises(NativeRecipeError):
                run_native_recipe_path(self.recipe, self.image_path)

    def test_direct_api_rejects_forged_tables_identity_and_unknown_external_stages(self):
        model = load_selected_model(self.source)
        identity = selected_model_identity(self.source, model)
        forged = replace(model, tables_sha256='a'*64)
        cases = (
            {MODEL_STAGE_ID: (forged, identity)},
            {MODEL_STAGE_ID: (model, {**identity, 'source_bytes': 1})},
            {'unknown': (model, identity)},
        )
        for inputs in cases:
            with self.subTest(inputs=tuple(inputs)):
                with self.assertRaises(NativeRecipeError):
                    run_native_recipe(self.recipe, None, None, source_sha256='a'*64,
                        coordinate_frame=FRAME, preparation=PREPARATION,
                        external_inputs=inputs, cache=PipelineAnalysisCache())

    def test_nested_mask_signatures_and_alias_aware_storage(self):
        base = torch.zeros((2, 23, 29), dtype=torch.uint8, device='cuda')
        outputs = {'class_masks': MappingProxyType({'leaf': base[0], 'soil': base[1]}),
            'names': ('leaf', 'soil')}
        before = _output_signatures(outputs)
        storage = _nested_cuda_storage(outputs)
        self.assertEqual(len(storage), 1)
        self.assertEqual(sum(storage.values()), base.untyped_storage().nbytes())
        base[0, 4, 5] = 255
        self.assertNotEqual(before, _output_signatures(outputs))


class NativeNaiveBayesDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.model_path = self.root/'model.tsv'
        self.model_path.write_bytes(_pdf())
        self.image_path = self.root/'seed.png'
        self.image = _image()
        self.assertTrue(cv2.imwrite(str(self.image_path), self.image))
        self.dialog = NativeNaiveBayesDialog(build_native_naive_bayes_recipe())

    def tearDown(self):
        self.dialog.close()
        self.application.processEvents()
        self.temp.cleanup()

    def _pick_model(self):
        with patch('seedvision.ui.native_naive_bayes_dialog.QFileDialog.getOpenFileName',
                return_value=(str(self.model_path), '')):
            self.dialog._choose_model()

    def _wait(self):
        loop = QEventLoop()
        timer = QTimer()
        timer.timeout.connect(lambda: loop.quit() if self.dialog._worker is None else None)
        timer.start(10)
        QTimer.singleShot(15000, loop.quit)
        loop.exec()
        timer.stop()
        self.assertIsNone(self.dialog._worker, 'native worker did not finish')

    def test_picker_and_save_keep_one_effective_model_selection(self):
        self.dialog.set_source_path(self.image_path)
        self.assertFalse(self.dialog.run_button.isEnabled())
        self._pick_model()
        self.assertTrue(self.dialog.run_button.isEnabled())
        self.assertEqual(self.dialog.model_label.textFormat(), Qt.TextFormat.PlainText)
        self.assertFalse(self.dialog.graph.nodes[MODEL_STAGE_ID].parameter_specs)
        path = self.root/'saved.seedfiddle-recipe.json'
        with patch('seedvision.ui.native_recipe_dialog.QFileDialog.getSaveFileName',
                return_value=(str(path), '')):
            self.dialog._save_recipe()
        saved = RecipeSpec.from_json(path.read_text(encoding='utf-8'))
        self.assertEqual(saved.identity, self.dialog.recipe.identity)
        self.assertEqual(tuple(saved.parameters_for_runner(MODEL_STAGE_ID)), (MODEL_PARAMETER_ID,))

    def test_real_async_run_shows_actual_class_pixels_and_reset_clears_model(self):
        self.dialog.set_source_path(self.image_path)
        self._pick_model()
        self.dialog._select_stage(CLASSIFIER_STAGE_ID)
        self.dialog.start_run()
        self._wait()
        self.assertIsNotNone(self.dialog.result, self.dialog.status_label.text())
        self.assertEqual(self.dialog.result.status, 'executed', self.dialog.status_label.text())
        expected = _reference(self.image, _pdf())
        self.assertEqual([self.dialog.class_selector.itemText(i)
            for i in range(self.dialog.class_selector.count())], list(expected))
        for index, name in enumerate(expected):
            self.dialog.class_selector.setCurrentIndex(index)
            np.testing.assert_array_equal(self.dialog._last_class_preview, expected[name])
        self.dialog._parameters_reset(MODEL_STAGE_ID)
        self.assertIsNone(self.dialog.result)
        self.assertIsNone(self.dialog._selected_model_source())
        self.assertIsNone(self.dialog._last_class_preview)
        self.assertTrue(self.dialog.class_selector.isHidden())
        self.assertFalse(self.dialog.run_button.isEnabled())

    def test_async_cancellation_and_invalid_picker_clear_previous_preview(self):
        self.dialog.set_source_path(self.image_path)
        self._pick_model()
        mask = torch.full((23, 29), 255, dtype=torch.uint8, device='cuda')
        node = SimpleNamespace(operation_id='analysis.naive_bayes_classifier', status='executed',
            reason='', parameters={}, outputs={'class_masks': MappingProxyType({'leaf': mask}),
                'class_names': ('leaf',), 'model_identity': {}})
        self.dialog.result = SimpleNamespace(node=lambda _stage: node, source_sha256='a'*64,
            coordinate_frame=FRAME)
        self.dialog._select_stage(CLASSIFIER_STAGE_ID)
        entered = threading.Event()

        def wait_for_stop(_recipe, _path, *, cache, cancel):
            entered.set()
            deadline = time.monotonic()+5
            while not cancel() and time.monotonic() < deadline:
                time.sleep(0.005)
            if cancel():
                raise NativeRecipeCancelled('test cancellation')
            raise AssertionError('worker cancellation was not delivered')

        with patch('seedvision.ui.native_recipe_dialog.run_native_recipe_path', side_effect=wait_for_stop):
            self.dialog.start_run()
            self.assertTrue(entered.wait(2))
            self.assertIsNone(self.dialog.result)
            self.assertIsNone(self.dialog._last_class_preview)
            self.assertTrue(self.dialog.class_selector.isHidden())
            self.dialog.stop()
            self._wait()
        self.assertIn('cancelled', self.dialog.status_label.text())
        self.assertTrue(self.dialog.choose_model_button.isEnabled())
        self.model_path.write_bytes(b'bad probability table')
        self._pick_model()
        self.assertIsNone(self.dialog._selected_model_source())
        self.assertFalse(self.dialog.run_button.isEnabled())
        self.assertTrue(self.dialog.class_selector.isHidden())

    def test_launcher_preserves_historical_attachments_and_selection(self):
        from seedvision.ui.reference_graph_dialog import ReferenceGraphDialog
        dialog = ReferenceGraphDialog()
        try:
            before = dialog.reference_graphs
            previous_id = dialog.recipe_selector.currentData()
            with patch('seedvision.ui.native_naive_bayes_dialog.NativeNaiveBayesDialog') as view:
                view.return_value.exec.return_value = 0
                dialog._open_native_naive_bayes()
                view.assert_called_once()
            self.assertEqual(dialog.reference_graphs, before)
            self.assertEqual(dialog.recipe_selector.currentData(), previous_id)
        finally:
            dialog.close()
