"""Annotation centroids and radial hilum metadata must never become predictions."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from seedvision.annotation.geometry import annotation_centres, outward_hilum_direction


class AnnotationGeometryTests(unittest.TestCase):
    def test_all_ids_use_area_centroids_including_disconnected_and_partial_masks(self):
        labels = np.zeros((260, 35), np.uint16)
        labels[120:140, 3:7] = 1  # Spans a row chunk.
        labels[1:4, 20:24] = 2
        labels[12, 34] = 2  # A disconnected fragment contributes area, not bbox midpoint.
        labels[250, 19] = 65535
        before = labels.copy()
        centres = annotation_centres(labels)
        self.assertEqual(set(centres), {1, 2, 65535})
        for i in centres:
            y, x = np.nonzero(labels == i)
            np.testing.assert_allclose(centres[i], (x.mean(), y.mean()))
        np.testing.assert_array_equal(labels, before)
        self.assertEqual(annotation_centres(labels.astype(np.uint64)), centres)

    def test_empty_invalid_and_degenerate_geometry(self):
        for labels in (None, np.zeros((0, 3), np.uint16), np.zeros((5, 7), np.uint16)):
            self.assertEqual(annotation_centres(labels), {})
        for labels in (np.zeros((2, 3, 1), np.uint16), np.zeros((3, 3)),
                       np.array([[-1]]), np.array([[65536]])):
            with self.assertRaises(ValueError):
                annotation_centres(labels)
        for centre, point in ((None, (2, 3)), ((2, 3), None), ((2, 3), (2, 3)),
                              ((2, 3), (np.nan, 3))):
            self.assertIsNone(outward_hilum_direction(centre, point))

    def test_hilum_direction_is_outward_radial_unit_vector(self):
        for delta in ((3, 4), (-3, 4), (-3, -4), (3, -4)):
            point = (20 + delta[0], 30 + delta[1])
            np.testing.assert_allclose(outward_hilum_direction((20, 30), point), np.asarray(delta) / 5)


class AnnotationCentresUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def make_window(self):
        from PySide6.QtGui import QImage
        from seedvision.ui.main_window import MainWindow
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        (root / "images").mkdir()
        path = root / "images" / "example.png"
        image = QImage(160, 120, QImage.Format.Format_RGB888)
        image.fill(0xffffff)
        self.assertTrue(image.save(str(path)))
        window = MainWindow(root)
        self.addCleanup(window.close)
        return window, path

    def test_compact_seed_selector_and_no_routine_explanatory_labels(self):
        from PySide6.QtWidgets import QCheckBox, QLabel, QSizePolicy
        window, _ = self.make_window()
        spin = window.instance_id_spin
        self.assertEqual(spin.prefix(), "")
        self.assertEqual(spin.maximum(), 9999)
        self.assertEqual(spin.sizePolicy().horizontalPolicy(), QSizePolicy.Policy.Fixed)
        self.assertEqual(window.new_instance_button.text(), "Next empty")
        row = spin.parentWidget()
        self.assertEqual(row.layout().itemAt(0).widget().text(), "Seed:")
        self.assertIs(row.layout().itemAt(1).widget(), window.instance_colour_swatch)
        self.assertEqual(row.findChildren(QCheckBox), [])
        self.assertEqual(window.show_selected_instance_checkbox.text(), "Show selected only")
        for name in ("existing_instance_combo", "instance_boundary_supervision_label",
                     "seed_trait_status_label", "seed_hilum_direction_checkbox",
                     "seed_hilum_direction_angle_spin"):
            self.assertFalse(hasattr(window, name), name)
        self.assertTrue(window.seed_shape_status_label.isHidden())
        for label in window.instance_annotation_controls.findChildren(QLabel):
            if not label.isHidden():
                self.assertNotIn("No coat-pattern vocabulary", label.text())
                self.assertNotIn("Applied complete seed masks", label.text())

    def test_crosshairs_follow_masks_selection_visibility_opacity_and_image_switch(self):
        window, _ = self.make_window()
        view = window.image_view
        labels = np.zeros((120, 160), np.uint16)
        labels[10:31, 20:61] = 1
        labels[70:91, 90:111] = 2
        view.set_instance_annotations(labels)
        view.render_reference_annotations_without_analysis()
        def markers():
            return [i for i in view._overlay_items if i.data(0) == "annotation-centres"]
        self.assertEqual(len(markers()), 1)
        self.assertEqual(markers()[0].centres, {1: (40., 20.), 2: (100., 80.)})
        cached = view.instance_centres()
        with patch("seedvision.ui.image_view.annotation_centres", side_effect=AssertionError("Display-only change recalculated geometry")):
            view.set_annotation_opacity(.23)
            view.set_show_selected_instance_only(True)
            view.set_active_instance_id(2)
            self.assertEqual(markers()[0].centres, {2: (100., 80.)})
            self.assertAlmostEqual(markers()[0].opacity(), .23)
            view.set_instance_annotations_visible(False)
            self.assertEqual(markers(), [])
            view.set_instance_annotations_visible(True)
            self.assertIs(view.instance_centres(), cached)
        view.set_show_selected_instance_only(False)
        view._instance_annotations[30:41, 60:71] = 1
        view._refresh_instance_annotation_overlay()
        self.assertNotEqual(view.instance_centre(1), (40., 20.))
        self.assertEqual(markers()[0].centres, annotation_centres(view._instance_annotations))
        # Crosshairs do not write manual centres or trigger annotation edits.
        self.assertIsNone(view._analysis_result)
        self.assertEqual(view._manual_seed_centres.shape[0], 0)
        view.clear_image()
        self.assertEqual(markers(), [])
        self.assertEqual(view.instance_centres(), {})

    def test_hilum_tracks_outline_edits_undo_and_persistence(self):
        from seedvision.persistence.reference_regions import ReferenceRegionStore
        window, image_path = self.make_window()
        key = window._current_image_key()
        labels = np.zeros((120, 160), np.uint16)
        labels[20:41, 30:71] = 1  # Centre (50, 30).
        window.image_view.set_instance_annotations(labels)
        window._instance_annotations_edited(labels)
        window.instance_id_spin.setValue(1)
        window._hilum_landmark_edited((70., 45.), (-1., 0.))  # Ignore arbitrary supplied direction.
        expected = outward_hilum_direction((50, 30), (70, 45))
        np.testing.assert_allclose(window._draft_seed_annotations[key][1].hilum_direction, expected)
        np.testing.assert_allclose(window.image_view._hilum_direction, expected)
        self.assertIn("(auto)", window.seed_hilum_direction_label.text())
        changed = labels.copy()
        changed[20:41, 71:92] = 1
        window.image_view.set_instance_annotations(changed)
        window._instance_annotations_edited(changed)
        adjusted = outward_hilum_direction(annotation_centres(changed)[1], (70, 45))
        np.testing.assert_allclose(window._draft_seed_annotations[key][1].hilum_direction, adjusted)
        self.assertFalse(np.allclose(expected, adjusted))
        window._undo_instance_reference_edit()
        np.testing.assert_array_equal(window.image_view._instance_annotations, labels)
        np.testing.assert_allclose(window._draft_seed_annotations[key][1].hilum_direction, expected)
        with patch.object(window, "_analyze_current_image"):
            window._apply_instance_annotations()
        restored = ReferenceRegionStore(image_path.parent.parent).load_if_present(image_path, labels.shape)
        self.assertIsNotNone(restored)
        np.testing.assert_allclose(restored.seed_annotations[0].hilum_direction, expected)
        np.testing.assert_array_equal(restored.annotated_seeds, labels)
        window.instance_id_spin.setValue(2)
        self.assertEqual(window.seed_hilum_direction_label.text(), "—")
        self.assertIsNone(window.image_view._hilum_direction)

    def test_direction_is_unknown_at_centroid_and_removed_without_landmark(self):
        from seedvision.persistence.reference_regions import SeedInstanceAnnotation
        from seedvision.ui.main_window import MainWindow
        labels = np.ones((3, 3), np.uint16)
        for point in ((1., 1.), None):
            derived = MainWindow._derive_seed_hilum_directions(labels, {
                1: SeedInstanceAnnotation(1, hilum_point=point, hilum_direction=(1., 0.))})
            self.assertIsNone(derived[1].hilum_direction)

    def test_brush_publishes_fresh_centroids_before_edit_listeners_run(self):
        from PySide6.QtCore import Qt
        from PySide6.QtTest import QTest
        from seedvision.ui.image_view import ImageView
        _, image_path = self.make_window()
        view = ImageView()
        self.addCleanup(view.close)
        view.resize(500, 400)
        view.show()
        view.load_image(image_path)
        labels = np.zeros((120, 160), np.uint16)
        labels[20:41, 30:71] = 1
        view.set_instance_annotations(labels)
        view.set_instance_annotation_editing(True)
        view.set_reference_brush_radius(8)
        view.set_instance_annotation_tool("brush")
        self.app.processEvents()
        self.assertEqual(view.instance_centre(1), (50., 30.))  # Prime cached geometry.
        received = []
        view.instance_annotations_edited.connect(
            lambda mask: received.append((dict(view.instance_centres()), annotation_centres(mask))))
        QTest.mouseClick(view.viewport(), Qt.MouseButton.LeftButton, pos=view.mapFromScene(90., 65.))
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0][0], received[0][1])
        self.assertNotEqual(received[0][0][1], (50., 30.))

    def test_next_empty_does_not_reuse_an_occupied_seed_at_capacity(self):
        window, _ = self.make_window()
        key = window._current_image_key()
        window._draft_instance_annotations[key] = np.arange(10000, dtype=np.uint16).reshape(100, 100)
        window.instance_id_spin.setValue(5)
        window._new_instance_annotation()
        self.assertEqual(window.instance_id_spin.value(), 5)
        self.assertIn("9,999", window.statusBar().currentMessage())
