from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


class SelectedInstanceImageViewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        try:
            from PySide6.QtWidgets import QApplication
        except ImportError as error:
            raise unittest.SkipTest(f"PySide6 unavailable: {error}")
        cls.application = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        from PySide6.QtGui import QColor, QImage

        from seedvision.ui.image_view import ImageView

        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.image_path = Path(self.temporary.name) / "selected-seeds.png"
        image = QImage(1200, 800, QImage.Format.Format_RGB32)
        image.fill(QColor("white"))
        self.assertTrue(image.save(str(self.image_path)))

        self.view = ImageView()
        self.addCleanup(self.view.close)
        self.view.resize(360, 300)
        succeeded, error = self.view.load_image(self.image_path)
        self.assertTrue(succeeded, error)
        self.view.show()
        self.application.processEvents()
        self.view.actual_size()

    def _two_seed_labels(self) -> np.ndarray:
        labels = np.zeros((800, 1200), dtype=np.uint16)
        labels[280:320, 260:310] = 1
        labels[500:550, 860:920] = 2
        return labels

    def _viewport_centre(self) -> tuple[float, float]:
        centre = self.view.mapToScene(self.view.viewport().rect().center())
        return float(centre.x()), float(centre.y())

    def test_selected_only_overlay_is_cropped_and_excludes_other_ids(self) -> None:
        labels = self._two_seed_labels()
        # A hidden neighbouring-ID pixel lies inside the one-pixel crop margin.
        labels[279, 259] = 2
        self.view.set_instance_annotations(labels, render=False)
        self.view.set_active_instance_id(1)

        self.view.set_show_selected_instance_only(True)

        item = self.view._instance_annotation_overlay_item
        self.assertIsNotNone(item)
        assert item is not None
        bounds = item.sceneBoundingRect()
        self.assertAlmostEqual(bounds.x(), 259.0)
        self.assertAlmostEqual(bounds.y(), 279.0)
        self.assertAlmostEqual(bounds.width(), 52.0)
        self.assertAlmostEqual(bounds.height(), 42.0)
        image = item.render_tile(0, 0)
        # Tile pixels retain global full-image coordinates; the one-pixel crop
        # controls exposure without resampling the authoritative label map.
        self.assertEqual(image.pixelColor(259, 279).alpha(), 0)
        self.assertGreater(image.pixelColor(260, 280).alpha(), 0)

    def test_id_change_centres_selected_seed_without_changing_zoom(self) -> None:
        self.view.set_instance_annotations(self._two_seed_labels(), render=False)
        self.view.set_active_instance_id(1)
        self.view.set_show_selected_instance_only(True)
        first_zoom = float(self.view.transform().m11())
        first_centre = self._viewport_centre()
        self.assertAlmostEqual(first_centre[0], 284.5, delta=2.0)
        self.assertAlmostEqual(first_centre[1], 299.5, delta=2.0)

        self.view.set_active_instance_id(2)
        self.application.processEvents()

        second_centre = self._viewport_centre()
        self.assertAlmostEqual(second_centre[0], 889.5, delta=2.0)
        self.assertAlmostEqual(second_centre[1], 524.5, delta=2.0)
        self.assertAlmostEqual(float(self.view.transform().m11()), first_zoom)
        item = self.view._instance_annotation_overlay_item
        self.assertIsNotNone(item)
        assert item is not None
        self.assertGreater(item.sceneBoundingRect().x(), 850.0)

    def test_absent_id_hides_overlay_without_moving_view(self) -> None:
        self.view.set_instance_annotations(self._two_seed_labels(), render=False)
        self.view.set_active_instance_id(1)
        self.view.set_show_selected_instance_only(True)
        before = self._viewport_centre()
        zoom = float(self.view.transform().m11())

        self.view.set_active_instance_id(99)
        self.application.processEvents()

        after = self._viewport_centre()
        self.assertAlmostEqual(after[0], before[0], delta=0.01)
        self.assertAlmostEqual(after[1], before[1], delta=0.01)
        self.assertAlmostEqual(float(self.view.transform().m11()), zoom)
        self.assertIsNone(self.view._instance_annotation_overlay_item)

    def test_label_replacement_and_committed_edit_invalidate_bounds_cache(self) -> None:
        labels = np.zeros((800, 1200), dtype=np.uint16)
        labels[100:110, 120:130] = 1
        self.view.set_instance_annotations(labels, render=False)
        self.assertEqual(self.view._instance_bounds(1), (120, 100, 130, 110))
        self.assertIn(1, self.view._instance_bounds_cache)

        replacement = np.zeros_like(labels)
        replacement[200:215, 240:260] = 1
        self.view.set_instance_annotations(replacement, render=False)
        self.assertNotIn(1, self.view._instance_bounds_cache)
        self.assertEqual(self.view._instance_bounds(1), (240, 200, 260, 215))

        self.view.set_active_instance_id(1)
        self.view.set_show_selected_instance_only(True)
        self.view.set_instance_annotation_editing(True)
        self.view.set_reference_brush_radius(4.0)
        self.view._detach_active_reference_buffers()
        from PySide6.QtCore import QPointF

        self.view._emit_reference_brush_dab(QPointF(320.0, 260.0), erase=False)
        self.view._refresh_instance_annotation_overlay()

        self.assertEqual(self.view._instance_bounds(1), (240, 200, 325, 265))

    def test_selected_only_paint_and_erase_preserve_hidden_neighbour(self) -> None:
        from PySide6.QtCore import QPointF

        labels = np.zeros((800, 1200), dtype=np.uint16)
        labels[90:105, 90:105] = 1
        labels[100:115, 105:120] = 2
        original_neighbour = labels == 2
        self.view.set_instance_annotations(labels, render=False)
        self.view.set_active_instance_id(1)
        self.view.set_show_selected_instance_only(True)
        self.view.set_instance_annotation_editing(True)
        self.view.set_reference_brush_radius(18.0)

        point = QPointF(105.0, 105.0)
        self.view._detach_active_reference_buffers()
        self.view._emit_reference_brush_dab(point, erase=False)
        painted = self.view._instance_annotations
        self.assertIsNotNone(painted)
        assert painted is not None
        self.assertTrue(np.all(painted[original_neighbour] == 2))
        self.assertEqual(int(painted[105, 100]), 1)

        self.view._detach_active_reference_buffers()
        self.view._emit_reference_brush_dab(point, erase=True)
        erased = self.view._instance_annotations
        self.assertIsNotNone(erased)
        assert erased is not None
        self.assertTrue(np.all(erased[original_neighbour] == 2))
        self.assertEqual(int(erased[100, 100]), 0)


class SelectedInstanceMainWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        try:
            from PySide6.QtWidgets import QApplication
        except ImportError as error:
            raise unittest.SkipTest(f"PySide6 unavailable: {error}")
        cls.application = QApplication.instance() or QApplication([])

    def test_checkbox_plumbs_to_view_and_tracks_analysis_availability(self) -> None:
        from PySide6.QtGui import QColor, QImage

        from seedvision.ui.main_window import MainWindow

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image_path = root / "images" / "seed.png"
            image_path.parent.mkdir(parents=True)
            image = QImage(96, 72, QImage.Format.Format_RGB32)
            image.fill(QColor("white"))
            self.assertTrue(image.save(str(image_path)))
            window = MainWindow(root)
            self.addCleanup(window.close)

            checkbox = window.show_selected_instance_checkbox
            self.assertEqual(checkbox.text(), "Show selected seed only")
            self.assertFalse(checkbox.isChecked())
            self.assertFalse(checkbox.isEnabled())
            self.assertFalse(window.image_view._show_selected_instance_only)

            window.image_view._analysis_result = SimpleNamespace()
            window._sync_background_controls()
            self.assertTrue(checkbox.isEnabled())
            checkbox.setChecked(True)
            self.assertTrue(window.image_view._show_selected_instance_only)

            key = window._current_image_key()
            self.assertIsNotNone(key)
            assert key is not None
            window._active_tasks[key] = object()
            try:
                window._sync_background_controls()
                self.assertFalse(checkbox.isEnabled())
            finally:
                window._active_tasks.pop(key, None)

            window.image_view._analysis_result = None
            window._sync_background_controls()
            self.assertFalse(checkbox.isEnabled())


if __name__ == "__main__":
    unittest.main()
