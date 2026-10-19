"""Pinned class masks, asset bindings, cache invalidation and desktop execution."""

from tests.catalogue_fixtures import CatalogueTestCase

from tests.functional_units import functional_units
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


@functional_units
class NativeNaiveBayesRecipeTests(CatalogueTestCase):
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


    def check_test_actual_class_masks_match_public_call_and_nested_mutation_recomputes_only_classifier(self):
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

    def check_test_reselecting_changed_model_preserves_prepared_image_and_image_node(self):
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



    def check_test_direct_api_rejects_forged_tables_identity_and_unknown_external_stages(self):
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

    def check_test_nested_mask_signatures_and_alias_aware_storage(self):
        base = torch.zeros((2, 23, 29), dtype=torch.uint8, device='cuda')
        outputs = {'class_masks': MappingProxyType({'leaf': base[0], 'soil': base[1]}),
            'names': ('leaf', 'soil')}
        before = _output_signatures(outputs)
        storage = _nested_cuda_storage(outputs)
        self.assertEqual(len(storage), 1)
        self.assertEqual(sum(storage.values()), base.untyped_storage().nbytes())
        base[0, 4, 5] = 255
        self.assertNotEqual(before, _output_signatures(outputs))

    def test_validation_and_recovery(self):
        """2 related NativeNaiveBayesRecipeTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_actual_class_masks_match_public_call_and_nested_mutation_recomputes_only_classifier)
        self.scenario(self.check_test_direct_api_rejects_forged_tables_identity_and_unknown_external_stages)

    def test_workflow_lifecycle(self):
        """2 related NativeNaiveBayesRecipeTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_reselecting_changed_model_preserves_prepared_image_and_image_node)
        self.scenario(self.check_test_nested_mask_signatures_and_alias_aware_storage)


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
