"""Frozen-centre bindings, public predictions, native caches and Qt execution."""

from tests.catalogue_fixtures import CatalogueTestCase
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
class NativeKMeansRecipeTests(CatalogueTestCase):
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
