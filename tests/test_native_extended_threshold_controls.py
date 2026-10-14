"""Pinned control witnesses and desktop routing for the four working thresholds."""
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch

from seedvision.benchmarking.native_extended_threshold_recipes import build_native_extended_threshold_recipes
from seedvision.benchmarking.recipe_catalogue import build_recipe_catalogue
from seedvision.benchmarking.recipes import RecipeSpec
from seedvision.benchmarking.native_recipe import run_native_recipe
from seedvision.benchmarking.checkpoints import CoordinateFrame
from tests.test_native_extended_threshold_recipe import _fixture, _reference_mask, _with_parameters, SOURCE, PREPARATION


def _fixtures():
    rng = np.random.default_rng(20261008)
    noise = rng.integers(0, 256, (31, 37), dtype=np.uint8)
    y, x = np.indices((31, 37))
    checker = (((x//3+y//4) % 2)*220+17).astype(np.uint8)
    edge = np.where(x < 18, 38, 206).astype(np.uint8)
    return noise, checker, edge


def _native_mask(operation, image, case):
    recipe = _with_parameters(build_native_extended_threshold_recipes()[
        'plantcv_threshold_'+operation], **case)
    is_rgb = operation == 'dual_channels'
    gray = np.zeros(image.shape[:2], dtype=np.uint8) if is_rgb else image
    rgb = np.ascontiguousarray(image[..., ::-1]) if is_rgb else np.repeat(image[..., None], 3, axis=-1)
    result = run_native_recipe(recipe,
        torch.as_tensor(np.ascontiguousarray(gray), device='cuda'),
        torch.as_tensor(np.ascontiguousarray(rgb), device='cuda'),
        source_sha256=SOURCE, coordinate_frame=CoordinateFrame(
            'control.witness', gray.shape, 'top-left', 'x-right,y-down', 'pixel'),
        preparation=PREPARATION)
    node = result.node(recipe.stages[1].stage_id)
    if node.status != 'executed':
        raise AssertionError(node.reason)
    return node.outputs['mask'].cpu().numpy()


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA is required for threshold control witnesses')
class ExtendedThresholdControlTests(unittest.TestCase):
    def witness(self, operation, base, variant, fixtures):
        for image in fixtures:
            native = [_native_mask(operation, image, case) for case in (base, variant)]
            if np.array_equal(native[0], native[1]):
                continue
            for actual, case in zip(native, (base, variant)):
                expected = _reference_mask(image, operation, case)
                np.testing.assert_array_equal(actual, expected,
                    err_msg=f'{operation} control {base!r} -> {variant!r}')
            return
        self.fail(f'No fixture makes {operation} control live: {base!r} -> {variant!r}')

    def test_every_gaussian_control_changes_a_pinned_graph_mask(self):
        base = {'ksize': 3, 'offset': 0, 'object_type': 'light'}
        for parameter, value in (('ksize', 9), ('offset', 24), ('object_type', 'dark')):
            with self.subTest(parameter=parameter):
                self.witness('gaussian', base, {**base, parameter:value}, _fixtures())

    def test_every_dual_channel_control_changes_a_pinned_graph_mask(self):
        rng = np.random.default_rng(502)
        image = rng.integers(0, 256, (31, 37, 3), dtype=np.uint8)
        base = {'x_channel':'B', 'y_channel':'G',
            'points':[[25,225],[225,25]], 'above':True}
        for parameter, value in (('x_channel','R'), ('y_channel','R'),
                ('points',[[45,190],[210,70]]), ('above',False)):
            with self.subTest(parameter=parameter):
                self.witness('dual_channels', base, {**base,parameter:value}, (image,))

    def test_every_texture_control_changes_a_pinned_graph_mask(self):
        # Two-level fixtures keep declared precision withholding separate from
        # whether an exposed parameter reaches the actual graph calculation.
        base = {'ksize':3, 'offset':1, 'threshold':90,
            'texture_method':'dissimilarity', 'borders':'nearest'}
        for parameter, value in (('ksize',7), ('offset',2), ('threshold',190),
                ('texture_method','homogeneity'), ('borders','wrap')):
            with self.subTest(parameter=parameter):
                selected = {**base, 'threshold':40} if parameter == 'borders' else base
                self.witness('texture', selected, {**selected,parameter:value}, _fixtures()[1:])

    def test_triangle_polarity_changes_a_pinned_graph_mask(self):
        self.witness('triangle', {'object_type':'light'}, {'object_type':'dark'}, (_fixture()[2],))


class ExtendedThresholdDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
        from PySide6.QtWidgets import QApplication
        cls.application = QApplication.instance() or QApplication([])

    def wait_until(self, predicate, timeout=30):
        deadline = time.monotonic()+timeout
        while not predicate() and time.monotonic() < deadline:
            self.application.processEvents()
            time.sleep(.01)
        self.application.processEvents()
        self.assertTrue(predicate(), 'The native threshold worker did not drain.')

    def test_selector_routes_each_operation_and_cancel_preserves_historical_attachments(self):
        from PySide6.QtWidgets import QPushButton
        from seedvision.ui.reference_graph_dialog import ReferenceGraphDialog
        before = build_recipe_catalogue().to_json()
        recipes = build_native_extended_threshold_recipes()
        dialog = ReferenceGraphDialog()
        try:
            button = dialog.findChild(QPushButton, 'runNativeExtendedThreshold')
            self.assertIsNotNone(button)
            self.assertTrue(button.isEnabled())
            attached = dialog.reference_graphs
            for recipe in recipes.values():
                with self.subTest(recipe=recipe.recipe_id), patch(
                        'seedvision.ui.reference_graph_dialog.QInputDialog.getItem',
                        return_value=(recipe.title, True)), patch(
                        'seedvision.ui.native_recipe_dialog.NativeRecipeDialog') as native_dialog:
                    native_dialog.return_value.exec.return_value = 0
                    button.click()
                    native_dialog.assert_called_once_with(recipe, dialog, model_asset_paths={})
                    native_dialog.return_value.exec.assert_called_once()
                    native_dialog.return_value.deleteLater.assert_called_once()
                self.assertEqual(dialog.reference_graphs, attached)
            with patch('seedvision.ui.reference_graph_dialog.QInputDialog.getItem',
                    return_value=('', False)), patch(
                    'seedvision.ui.native_recipe_dialog.NativeRecipeDialog') as native_dialog:
                button.click()
                native_dialog.assert_not_called()
            self.assertEqual(dialog.reference_graphs, attached)
            self.assertEqual(build_recipe_catalogue().to_json(), before)
        finally:
            dialog.reject()
            dialog.deleteLater()

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is required for the async threshold dialog')
    def test_async_run_changed_control_reset_and_save(self):
        from seedvision.ui.native_recipe_dialog import NativeRecipeDialog
        recipe = build_native_extended_threshold_recipes()['plantcv_threshold_gaussian']
        stage = recipe.stages[1]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'threshold-input.png'
            output_path = Path(directory)/'selected-recipe.json'
            gray = _fixtures()[0]
            gray = np.full(gray.shape,37,dtype=np.uint8)
            gray[:,gray.shape[1]//2:] = 219
            bgr = np.stack((gray, gray, gray), axis=-1)
            self.assertTrue(cv2.imwrite(str(path), bgr))
            dialog = NativeRecipeDialog(recipe)
            dialog.show()
            try:
                dialog.set_source_path(path)
                dialog.start_run()
                self.wait_until(lambda: dialog._worker is None)
                self.assertIsNotNone(dialog.result, dialog.status_label.text())
                first = dialog.result.node(stage.stage_id).outputs['mask'].clone()
                dialog._select_stage(stage.stage_id)
                self.assertFalse(dialog.preview.pixmap().isNull())
                dialog._parameter_changed(stage.stage_id, 'ksize', '15')
                self.assertIsNone(dialog.result)
                self.assertTrue(dialog.preview.pixmap().isNull())
                dialog.start_run()
                self.wait_until(lambda: dialog._worker is None)
                self.assertIsNotNone(dialog.result, dialog.status_label.text())
                self.assertEqual(dialog.cache.last_computed_nodes, (stage.stage_id,))
                self.assertIn(recipe.stages[0].stage_id, dialog.cache.last_reused_nodes)
                self.assertFalse(torch.equal(first, dialog.result.node(stage.stage_id).outputs['mask']))
                dialog.inspector.parameters_reset.emit(stage.stage_id)
                self.assertEqual(dialog.recipe, recipe)
                with patch('seedvision.ui.native_recipe_dialog.QFileDialog.getSaveFileName',
                        return_value=(str(output_path), '')):
                    dialog._save_recipe()
                self.assertEqual(RecipeSpec.from_json(output_path.read_bytes()), dialog.recipe)
            finally:
                dialog.reject()
                self.wait_until(lambda: dialog._worker is None)
                dialog.deleteLater()
