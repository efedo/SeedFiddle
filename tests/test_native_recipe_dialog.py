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

from tests.test_native_recipe import selected_recipe, fixture, PREFIX


class NativeRecipeDialogTests(unittest.TestCase):
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
        from seedvision.ui.reference_graph_dialog import ReferenceGraphDialog
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
        reference = ReferenceGraphDialog()
        try:
            index = reference.recipe_selector.findData('plantcv_seed_binary')
            reference.recipe_selector.setCurrentIndex(index)
            before = reference.reference_graphs
            with patch('seedvision.ui.native_recipe_dialog.NativeRecipeDialog.exec', return_value=0):
                reference._open_native_recipe()
            self.assertEqual(reference.reference_graphs, before)
            self.assertTrue(reference._native_run_button.isEnabled())
        finally:
            reference.reject()
            reference.deleteLater()

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is required for the size thickness control')
    def test_working_size_thickness_async_preview_reset_and_save(self):
        from seedvision.ui.native_recipe_dialog import NativeRecipeDialog
        from seedvision.benchmarking.native_recipe import build_native_seed_recipe
        from seedvision.benchmarking.recipes import RecipeSpec
        recipe = build_native_seed_recipe(selected_recipe())
        stage = PREFIX+'analyze_size'
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'seed.png'
            gray,_ = fixture()
            self.assertTrue(cv2.imwrite(str(path),np.repeat(gray[...,None],3,axis=2)))
            dialog = NativeRecipeDialog(recipe)
            dialog.show()
            try:
                dialog.set_source_path(path)
                dialog.start_run()
                self.wait_until(lambda: dialog._worker is None)
                self.assertIsNotNone(dialog.result,dialog.status_label.text())
                first = dialog.result.node(stage).outputs['diagnostic']
                dialog._select_stage(stage)
                self.assertIn('line_thickness',dialog.graph.nodes[stage].parameters)
                dialog.inspector.parameter_changed.emit(stage,'line_thickness','1')
                self.assertIsNone(dialog.result)
                dialog.start_run()
                self.wait_until(lambda: dialog._worker is None)
                self.assertIsNotNone(dialog.result,dialog.status_label.text())
                self.assertEqual(dialog.result.node(stage).parameters['line_thickness'],1)
                self.assertEqual(dialog.cache.last_computed_nodes,(stage,))
                self.assertFalse(dialog.preview.pixmap().isNull())
                self.assertFalse(torch.equal(first,dialog.result.node(stage).outputs['diagnostic']))
                output = Path(temporary)/'working.json'
                with patch('seedvision.ui.native_recipe_dialog.QFileDialog.getSaveFileName',
                        return_value=(str(output),'')):
                    dialog._save_recipe()
                self.assertEqual(RecipeSpec.from_json(output.read_bytes()),dialog.recipe)
                dialog.inspector.parameters_reset.emit(stage)
                self.assertEqual(dialog.recipe,recipe)
            finally:
                dialog.reject()
                self.wait_until(lambda: dialog._worker is None)
                dialog.deleteLater()

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is required for real Qt recipe execution')
    def test_actual_async_execution_preview_and_changed_parameter_reuse(self):
        from seedvision.ui.native_recipe_dialog import NativeRecipeDialog
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'seed.png'
            gray, _ = fixture()
            self.assertTrue(cv2.imwrite(str(path), np.repeat(gray[..., None], 3, axis=2)))
            dialog = NativeRecipeDialog(selected_recipe(**{'analysis.color': {'colorspaces': 'hsv'}}))
            dialog.show()
            try:
                dialog.set_source_path(path)
                dialog.start_run()
                self.assertFalse(dialog.inspector.isEnabled())
                self.assertTrue(dialog.stop_button.isEnabled())
                self.wait_until(lambda: dialog._worker is None)
                self.assertIsNotNone(dialog.result, dialog.status_label.text())
                self.assertEqual(dialog.result.status, 'executed')
                labels = dialog.result.labels
                dialog._select_stage(PREFIX+'ws_masked_flood')
                self.assertFalse(dialog.preview.pixmap().isNull())
                dialog._select_stage(PREFIX+'analyze_color')
                self.assertTrue(dialog.histogram.isVisible())
                self.assertEqual(dialog.histogram.chart.status, 'available')
                self.assertFalse(dialog.preview.isVisible())
                cache_objects = {key: id(value) for key, value in dialog.cache.values.items()}
                if dialog.histogram.samples.count() > 1:
                    dialog.histogram.samples.setCurrentIndex(1)
                self.application.processEvents()
                self.assertEqual({key: id(value) for key, value in dialog.cache.values.items()}, cache_objects)
                self.assertIs(dialog.result.labels, labels)
                dialog._parameter_changed(PREFIX+'analyze_color', 'colorspaces', 'lab')
                self.assertFalse(dialog.histogram.isVisible(), 'Changed observations must not leave a stale chart.')
                dialog.start_run()
                self.wait_until(lambda: dialog._worker is None)
                self.assertIsNotNone(dialog.result, dialog.status_label.text())
                self.assertIs(dialog.result.labels, labels)
                self.assertEqual(dialog.cache.last_computed_nodes, (PREFIX+'analyze_color',))
                dialog._select_stage(PREFIX+'analyze_color')
                self.assertEqual(dialog.histogram.chart.status, 'empty')
                self.assertFalse(dialog.histogram.samples.isEnabled())
                dialog._select_stage(PREFIX+'ws_masked_flood')
                self.assertFalse(dialog.histogram.isVisible())
                self.assertTrue(dialog.preview.isVisible())
                dialog._select_stage(PREFIX+'analyze_size')
                self.assertTrue(dialog.graph.nodes[PREFIX+'analyze_size'].implemented)
                self.assertIn('graph_1', dialog.facts.toPlainText())
                self.assertIn('area:', dialog.facts.toPlainText())
                self.assertIn('ellipse_major_axis:', dialog.facts.toPlainText())
                self.assertIn('diagnostic', dialog.facts.toPlainText())
                self.assertFalse(dialog.preview.pixmap().isNull())
                dialog._parameter_changed(PREFIX+'analyze_size', 'label', 'measured')
                dialog.start_run()
                self.wait_until(lambda: dialog._worker is None)
                self.assertIsNotNone(dialog.result, dialog.status_label.text())
                self.assertEqual(dialog.cache.last_computed_nodes, (PREFIX+'analyze_size',))
                self.assertIn('measured_1', dialog.facts.toPlainText())
                self.assertNotIn('graph_1', dialog.facts.toPlainText())
                self.assertIs(dialog.result.labels, labels)
            finally:
                dialog.reject()
                self.wait_until(lambda: dialog._worker is None)
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

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is required for a populated chart before rerun')
    def test_rerun_failure_and_cancellation_clear_previous_chart_and_observations(self):
        from seedvision.ui.native_recipe_dialog import NativeRecipeDialog
        from seedvision.benchmarking.native_recipe import NativeRecipeCancelled
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'seed.png'
            gray, _ = fixture()
            self.assertTrue(cv2.imwrite(str(path), np.repeat(gray[..., None], 3, axis=2)))
            dialog = NativeRecipeDialog(selected_recipe(**{'analysis.color': {'colorspaces': 'hsv'}}))
            dialog.show()
            release = threading.Event()
            try:
                dialog.set_source_path(path)
                dialog.start_run()
                self.wait_until(lambda: dialog._worker is None)
                successful_result = dialog.result
                self.assertIsNotNone(successful_result, dialog.status_label.text())
                for error in (RuntimeError('controlled failure'), NativeRecipeCancelled('controlled stop')):
                    with self.subTest(error=type(error).__name__):
                        dialog.result = successful_result
                        dialog._select_stage(PREFIX+'analyze_color')
                        self.assertTrue(dialog.histogram.isVisible())
                        self.assertIn('Available outputs:', dialog.facts.toPlainText())
                        entered = threading.Event()
                        release.clear()
                        def controlled_run(*args, **kwargs):
                            entered.set()
                            if not release.wait(10):
                                raise AssertionError('Test failed to release rerun worker')
                            raise error
                        with patch('seedvision.ui.native_recipe_dialog.run_native_recipe_path', side_effect=controlled_run):
                            dialog.start_run()
                            self.wait_until(entered.is_set)
                            self.assertIsNone(dialog.result)
                            self.assertFalse(dialog.histogram.isVisible())
                            self.assertIsNone(dialog.histogram.chart)
                            self.assertNotIn('Available outputs:', dialog.facts.toPlainText())
                            release.set()
                            self.wait_until(lambda: dialog._worker is None)
                            self.assertIsNone(dialog.result)
                            self.assertFalse(dialog.histogram.isVisible())
                            self.assertIsNone(dialog.histogram.chart)
                            self.assertNotIn('Available outputs:', dialog.facts.toPlainText())
                            self.assertIn('cancelled' if isinstance(error, NativeRecipeCancelled) else 'failed',
                                dialog.status_label.text())
            finally:
                release.set()
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
