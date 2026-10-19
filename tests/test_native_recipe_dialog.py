import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from tests.catalogue_fixtures import CatalogueTestCase
from unittest.mock import patch

import cv2
import numpy as np
import torch

from tests.test_native_recipe import selected_recipe, fixture, PREFIX


class NativeRecipeDialogTests(CatalogueTestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
        from PySide6.QtWidgets import QApplication
        cls.application = QApplication.instance() or QApplication([])

    def wait_until(self, predicate, timeout=20):
        deadline = time.monotonic()+timeout
        while not predicate() and time.monotonic() < deadline:
            self.application.processEvents()
            time.sleep(.01)
        self.application.processEvents()
        self.assertTrue(predicate(), 'Qt worker did not reach the expected state.')

    def test_selected_parameters_round_trip_without_changing_historical_graph(self):
        from seedvision.ui.native_recipe_dialog import NativeRecipeDialog
        from seedvision.benchmarking.recipes import RecipeSpec
        recipe = selected_recipe()
        dialog = NativeRecipeDialog(recipe)
        try:
            stage = PREFIX+'ws_local_maxima'
            dialog.inspector.parameter_changed.emit(stage, 'distance', '12')
            self.assertEqual(dialog.recipe.parameters_for_runner(stage)['distance'], 12)
            self.assertNotEqual(dialog.recipe.identity.settings_sha256, recipe.identity.settings_sha256)
            with tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary)/'selected.seedfiddle-recipe.json'
                with patch('seedvision.ui.native_recipe_dialog.QFileDialog.getSaveFileName', return_value=(str(output), '')):
                    dialog._save_recipe()
                restored = RecipeSpec.from_json(output.read_bytes())
                self.assertEqual(restored, dialog.recipe)
            dialog.inspector.parameters_reset.emit(stage)
            self.assertEqual(dialog.recipe, recipe)
        finally:
            dialog.reject()
            dialog.deleteLater()



    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is required for size diagnostics')
    def test_size_diagnostic_cap_retains_visible_measurements_and_clears_preview(self):
        from seedvision.benchmarking import native_recipe as implementation
        from seedvision.ui.native_recipe_dialog import NativeRecipeDialog
        runtime = implementation._runtime_records(torch.device('cuda', torch.cuda.current_device()))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'seed.png'
            gray, _ = fixture()
            self.assertTrue(cv2.imwrite(str(path), np.repeat(gray[...,None], 3, axis=2)))
            dialog = NativeRecipeDialog(selected_recipe())
            dialog.show()
            try:
                dialog.set_source_path(path)
                with patch.object(implementation, '_runtime_records', return_value=runtime), \
                        patch.object(implementation._size, 'render_plantcv_size_diagnostic_rgb',
                            side_effect=implementation._size.RasterSpanLimitError('Diagnostic cap')):
                    dialog.start_run()
                    self.wait_until(lambda: dialog._worker is None)
                self.assertIsNotNone(dialog.result, dialog.status_label.text())
                dialog._select_stage(PREFIX+'analyze_size')
                self.assertEqual(dialog.result.node(PREFIX+'analyze_size').status, 'partial')
                self.assertIn('area:', dialog.facts.toPlainText())
                self.assertIn('Unavailable outputs: diagnostic', dialog.facts.toPlainText())
                self.assertIn('No image output', dialog.preview.text())
                self.assertTrue(dialog.preview.pixmap().isNull())
            finally:
                dialog.reject()
                self.wait_until(lambda: dialog._worker is None)
                dialog.deleteLater()


    def test_close_requests_cancellation_and_waits_for_worker_to_drain(self):
        from seedvision.ui.native_recipe_dialog import NativeRecipeDialog
        from seedvision.benchmarking.native_recipe import NativeRecipeCancelled
        entered, release = threading.Event(), threading.Event()
        def controlled_run(*args, **kwargs):
            entered.set()
            self.assertTrue(release.wait(10))
            if kwargs['cancel']():
                raise NativeRecipeCancelled('Stopped')
            raise AssertionError('Close did not request cancellation.')
        dialog = NativeRecipeDialog(selected_recipe())
        dialog.show()
        try:
            with patch('seedvision.ui.native_recipe_dialog.run_native_recipe_path', side_effect=controlled_run):
                dialog.set_source_path(Path('images/IMG_9632c.JPG'))
                dialog.start_run()
                self.wait_until(entered.is_set)
                dialog.close()
                self.assertTrue(dialog.isVisible())
                self.assertTrue(dialog._cancel.is_set())
                release.set()
                self.wait_until(lambda: dialog._worker is None)
                self.assertFalse(dialog.isVisible())
                self.assertIsNone(dialog.result)
        finally:
            release.set()
            self.wait_until(lambda: dialog._worker is None)
            dialog.reject()
            dialog.deleteLater()


if __name__ == '__main__':
    unittest.main()
