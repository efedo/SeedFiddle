"""Annotation centroids and radial hilum metadata must never become predictions."""

from tests.review_fixtures import dispose_window, workspace_window
from tests.functional_units import functional_units
import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from seedvision.annotation.geometry import annotation_centres, annotation_median_length, outward_hilum_direction


@functional_units
class AnnotationGeometryTests(unittest.TestCase):
    def check_test_all_ids_use_area_centroids_including_disconnected_and_partial_masks(self):
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

    def check_test_empty_invalid_and_degenerate_geometry(self):
        for labels in (None, np.zeros((0, 3), np.uint16), np.zeros((5, 7), np.uint16)):
            self.assertEqual(annotation_centres(labels), {})
        for labels in (np.zeros((2, 3, 1), np.uint16), np.zeros((3, 3)),
                       np.array([[-1]]), np.array([[65536]])):
            with self.assertRaises(ValueError):
                annotation_centres(labels)
        for centre, point in ((None, (2, 3)), ((2, 3), None), ((2, 3), (2, 3)),
                              ((2, 3), (np.nan, 3))):
            self.assertIsNone(outward_hilum_direction(centre, point))

    def check_test_hilum_direction_is_outward_radial_unit_vector(self):
        for delta in ((3, 4), (-3, 4), (-3, -4), (3, -4)):
            point = (20 + delta[0], 30 + delta[1])
            np.testing.assert_allclose(outward_hilum_direction((20, 30), point), np.asarray(delta) / 5)

    def check_test_median_length_is_the_long_axis_at_any_rotation(self):
        labels = np.zeros((300, 900), np.uint16)
        y, x = np.indices(labels.shape)
        for identifier, (centre_x, angle) in enumerate(((100, 0.0), (300, 0.6), (500, np.pi / 2)), 1):
            along = (x - centre_x) * np.cos(angle) + (y - 150) * np.sin(angle)
            across = (y - 150) * np.cos(angle) - (x - centre_x) * np.sin(angle)
            labels[(along / 40.0) ** 2 + (across / 12.0) ** 2 <= 1.0] = identifier
        self.assertAlmostEqual(annotation_median_length(labels), 80.0, delta=1.0)
        # A merged blob and a stray pixel do not move the median.
        labels[50:250, 650:850] = 4
        labels[5, 5] = 5
        self.assertAlmostEqual(annotation_median_length(labels), 80.0, delta=1.0)
        for empty in (None, np.zeros((0, 3), np.uint16), np.zeros((5, 7), np.uint16)):
            self.assertEqual(annotation_median_length(empty), 0.0)
        with self.assertRaises(ValueError):
            annotation_median_length(np.zeros((3, 3)))

    def test_workflow_lifecycle(self):
        """3 related AnnotationGeometryTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_median_length_is_the_long_axis_at_any_rotation)
        self.scenario(self.check_test_all_ids_use_area_centroids_including_disconnected_and_partial_masks)
        self.scenario(self.check_test_hilum_direction_is_outward_radial_unit_vector)

    def test_validation_and_recovery(self):
        """1 related AnnotationGeometryTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_empty_invalid_and_degenerate_geometry)


@functional_units
class AnnotationCentresUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def make_window(self):
        from PySide6.QtGui import QImage
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        (root / "images").mkdir()
        path = root / "images" / "example.png"
        image = QImage(160, 120, QImage.Format.Format_RGB888)
        image.fill(0xffffff)
        self.assertTrue(image.save(str(path)))
        window = workspace_window(root)
        self.addCleanup(dispose_window, window)
        return window, path

    def check_test_compact_seed_selector_and_no_routine_explanatory_labels(self):
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
        self.assertEqual(window.instance_visibility_combo.itemText(0), "all")
        self.assertEqual(window.instance_visibility_combo.itemText(1), "selected")
        for name in ("existing_instance_combo", "instance_boundary_supervision_label",
                     "seed_trait_status_label", "seed_hilum_direction_checkbox",
                     "seed_hilum_direction_angle_spin"):
            self.assertFalse(hasattr(window, name), name)
        self.assertTrue(window.seed_shape_status_label.isHidden())
        for label in window.instance_annotation_controls.findChildren(QLabel):
            if not label.isHidden():
                self.assertNotIn("No coat-pattern vocabulary", label.text())
                self.assertNotIn("Applied complete seed masks", label.text())

    def check_test_crosshairs_follow_masks_selection_visibility_opacity_and_image_switch(self):
        window, _ = self.make_window()
        view = window.image_view
        labels = np.zeros((120, 160), np.uint16)
        labels[10:31, 20:61] = 1
        labels[70:91, 90:111] = 2
        view.set_instance_annotations(labels)
        view.render_reference_annotations_without_analysis()
        def markers():
            return [i for i in view._overlay_items if i.data(0) == "annotation-centres"]
        self.assertEqual(markers(), [])
        view._set_bgr_base_image(np.zeros((120, 160, 3), np.uint8))
        view._refresh_instance_annotation_overlay()
        self.assertEqual(len(markers()), 1)
        self.assertEqual(markers()[0].centres, {1: (40., 20.), 2: (100., 80.)})
        self.assertAlmostEqual(markers()[0].arm, annotation_median_length(labels) / 6.0)
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

    def check_test_crosshairs_span_a_third_of_the_median_seed_at_every_zoom(self):
        from PySide6.QtCore import QRectF
        from PySide6.QtGui import QImage, QPainter
        from PySide6.QtWidgets import QGraphicsScene
        from seedvision.ui.image_view import _InstanceCentreItem

        scene = QGraphicsScene()
        self.addCleanup(scene.clear)
        scene.addItem(_InstanceCentreItem({1: (200.0, 150.0)}, (300, 400), 120.0))
        for zoom in (0.25, 1.0, 3.0):
            width, height = int(400 * zoom), int(300 * zoom)
            image = QImage(width, height, QImage.Format.Format_RGB32)
            image.fill(0xff7f7f7f)
            painter = QPainter(image)
            scene.render(painter, QRectF(0, 0, width, height), QRectF(0, 0, 400, 300))
            painter.end()
            pixels = np.frombuffer(image.constBits(), np.uint8).reshape(height, image.bytesPerLine() // 4, 4)
            columns = np.flatnonzero((pixels[:, :, :3] != 127).any(axis=2).any(axis=0))
            # 120 px seeds give a 40 px marker; square stroke caps add up to 4 screen pixels.
            self.assertAlmostEqual(columns[-1] - columns[0] + 1, 40.0 * zoom, delta=5.0)

    def check_test_centre_markers_follow_the_image_except_in_the_editor(self):
        from PySide6.QtCore import QPointF, QRectF
        from PySide6.QtWidgets import QGraphicsItem

        window, _ = self.make_window()
        view = window.image_view
        constant = QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations
        record = {"point": QPointF(20.0, 20.0), "source": "automatic"}

        def marker(**options):
            first = len(view._overlay_items)
            view._render_manual_seed_centre_marker(record, **options)
            return view._overlay_items[first]

        display = marker(editing=False, seed_radius=12.0)
        self.assertFalse(display.flags() & constant)
        self.assertEqual(display.rect(), QRectF(-12.0, -12.0, 24.0, 24.0))
        editor = marker(editing=True, seed_radius=12.0)
        self.assertTrue(editor.flags() & constant)
        self.assertEqual(editor.rect(), QRectF(-7.0, -7.0, 14.0, 14.0))
        unknown_size = marker(editing=False)
        self.assertTrue(unknown_size.flags() & constant)
        self.assertEqual(unknown_size.rect(), QRectF(-5.0, -5.0, 10.0, 10.0))

        # 60 px estimated seeds give a 20 px marker through the real overlay path.
        first = len(view._overlay_items)
        with patch.object(view, "_procedural_marker_records", return_value=[record]):
            view._render_procedural_centres(SimpleNamespace(procedural_instances=object(), estimated_seed_diameter_px=60.0))
        self.assertEqual(view._overlay_items[first].rect(), QRectF(-10.0, -10.0, 20.0, 20.0))

    def check_test_rejected_proposals_show_only_on_the_centre_overlay_with_their_reason(self):
        from PySide6.QtCore import QPointF, QRectF
        from PySide6.QtGui import QColor

        window, _ = self.make_window()
        view = window.image_view
        used = {"point": QPointF(20.0, 20.0), "source": "automatic"}
        result = SimpleNamespace(
            crop_offset=(100, 200),
            estimated_seed_diameter_px=60.0,
            procedural_instances=SimpleNamespace(
                rejected_marker_centres_xy=np.asarray([[10.0, 30.0]], np.float32),
                rejected_marker_reasons=("instance_above_maximum_concavity",),
            ),
        )
        with patch.object(view, "_procedural_marker_records", return_value=[used]):
            # The instance overlay shows only markers in use.
            first = len(view._overlay_items)
            view._render_procedural_centres(result)
            self.assertEqual(len(view._overlay_items) - first, 1)

            first = len(view._overlay_items)
            view._render_procedural_centres(result, show_rejected=True)
            ring, *cross, marker = view._overlay_items[first:]
            self.assertEqual(len(cross), 2)
            self.assertEqual(ring.pos(), QPointF(110.0, 230.0))
            self.assertEqual(ring.rect(), QRectF(-10.0, -10.0, 20.0, 20.0))
            self.assertEqual(ring.pen().color(), QColor("#ff9f40"))
            self.assertEqual(ring.toolTip(), "Rejected proposal: above maximum concavity.")
            self.assertEqual(ring.zValue(), 34)
            self.assertEqual(marker.pos(), QPointF(20.0, 20.0))
            self.assertEqual(marker.pen().color(), QColor("#43ddff"))

            # They are not editable, so the centre editor leaves them out.
            first = len(view._overlay_items)
            with patch.object(view, "_manual_seed_centre_editing", True):
                view._render_procedural_centres(result, show_rejected=True)
            self.assertEqual(len(view._overlay_items), first)

        key = window._current_image_key()
        window._analyses[key] = SimpleNamespace(procedural_instances=SimpleNamespace(
            marker_sources=np.zeros(3, np.uint8),
            rejected_marker_reasons=(
                "instance_above_maximum_concavity",
                "instance_below_minimum_area",
                "instance_above_maximum_concavity",
            ),
        ))
        self.assertIn(
            "orange × = rejected proposal (3: 2 above maximum concavity, 1 below minimum area)",
            window._procedural_marker_key(include_rejected=True),
        )
        self.assertNotIn("rejected proposal", window._procedural_marker_key())
        window._analyses.pop(key)

    def check_test_annotation_palette_can_extend_beyond_image_view(self):
        from PySide6.QtCore import QPoint

        window, _ = self.make_window()
        window.resize(1200, 700)
        window.show()
        window.annotate_instances_action.setEnabled(True)
        window.annotate_instances_action.setChecked(True)
        self.app.processEvents()
        panel = window.reference_panel
        self.assertIs(panel.parentWidget(), window.annotation_workspace)
        self.assertTrue(panel.isVisible())
        handle_centre = window.reference_panel_drag_handle.mapTo(
            window.annotation_workspace,
            window.reference_panel_drag_handle.rect().center(),
        )
        hit_widget = window.annotation_workspace.childAt(handle_centre)
        self.assertIsNotNone(hit_widget)
        self.assertTrue(
            hit_widget is panel or panel.isAncestorOf(hit_widget)
        )
        bounds = window.annotation_workspace.rect()
        window.image_view._move_context_panel_to(
            QPoint(bounds.right() - panel.width() + 1, bounds.top()),
            remember=True,
        )
        image_right = window.image_view.mapTo(
            window.annotation_workspace, QPoint(window.image_view.width(), 0)
        ).x()
        self.assertGreater(panel.geometry().right(), image_right)
        self.assertFalse(panel.isHidden())
        window._show_pipeline_workspace()
        self.assertTrue(panel.isHidden())
        window._show_image_workspace()
        self.assertFalse(panel.isHidden())

    def check_test_annotations_only_appear_on_deskewed_image(self):
        from PySide6.QtCore import QPointF, Qt
        from PySide6.QtTest import QTest

        window, _ = self.make_window()
        view = window.image_view
        view.set_overlay_mode("none")
        material = np.zeros((120, 160), bool)
        material[30:40, 20:30] = True
        labels = np.zeros((120, 160), np.uint16)
        labels[65:75, 85:95] = 1
        view.set_reference_masks(material, None, render=False)
        view.set_instance_annotations(labels, render=False)
        result = SimpleNamespace(calibration=SimpleNamespace(
            corrected_bgr=np.zeros((120, 160, 3), np.uint8),
            colour_card=None,
        ))
        view.show_analysis(result)
        self.assertEqual(view._displayed_base, "corrected")
        self.assertEqual(len(view._overlay_items), 3)
        view.set_hilum_editing(True)
        view.set_hilum_landmark((90., 70.), None)
        self.assertEqual(len(view._hilum_items), 2)

        view.set_overlay_mode("raw_image")
        self.assertEqual(view._displayed_base, "source")
        self.assertEqual(view._overlay_items, [])
        self.assertEqual(view._hilum_items, [])
        view.set_instance_annotations_visible(False)
        view.set_instance_annotations_visible(True)
        view.set_material_reference_annotations_visible(False)
        view.set_material_reference_annotations_visible(True)
        self.assertEqual(view._overlay_items, [])
        view.set_hilum_editing(False)
        view.set_instance_annotation_editing(True)
        edits = []
        view.instance_annotations_edited.connect(edits.append)
        QTest.mouseClick(
            view.viewport(), Qt.MouseButton.LeftButton,
            pos=view.mapFromScene(110., 90.),
        )
        self.assertEqual(edits, [])
        np.testing.assert_array_equal(view.instance_annotations(), labels)
        view.set_instance_annotation_tool("smart_fill")
        view._update_instance_assisted_preview(QPointF(110., 90.))
        self.assertEqual(view._instance_preview_items, [])

        view.set_overlay_mode("colour_reference")
        self.assertEqual(view._overlay_items, [])
        view.set_overlay_mode("none")
        self.assertEqual(len(view._overlay_items), 3)
        view.clear_analysis()
        view.render_reference_annotations_without_analysis()
        self.assertEqual(view._overlay_items, [])

    def check_test_next_unannotated_cycles_painted_seeds_missing_condition_or_shape(self):
        from seedvision.persistence.reference_regions import SeedInstanceAnnotation
        from tests.review_fixtures import dispose_window

        window, _ = self.make_window()
        self.addCleanup(dispose_window, window)
        labels = np.zeros((120, 160), np.uint16)
        labels[10, 10] = 1
        labels[20, 20] = 2
        labels[30, 30] = 3
        window._instance_annotations_edited(labels)
        key = window._current_image_key()
        self.assertIsNotNone(key)
        complete = dict(
            conditions_reviewed=True,
            shape_reviewed=True,
            outline_visibility="complete",
            pose="flat",
        )
        window._draft_seed_annotations[key] = {
            1: SeedInstanceAnnotation(1, **complete),
            2: SeedInstanceAnnotation(2, **{**complete, "conditions_reviewed": False}),
            3: SeedInstanceAnnotation(3, **{**complete, "shape_reviewed": False}),
        }
        self.assertEqual(window.next_unannotated_button.text(), "Next unannotated")
        selector = window.new_instance_button.parentWidget().layout()
        self.assertIs(selector.itemAt(4).widget(), window.next_unannotated_button)
        window.instance_id_spin.setValue(1)
        window._next_unannotated_instance()
        self.assertEqual(window.instance_id_spin.value(), 2)
        window._next_unannotated_instance()
        self.assertEqual(window.instance_id_spin.value(), 3)
        window._next_unannotated_instance()
        self.assertEqual(window.instance_id_spin.value(), 2)
        window._draft_seed_annotations[key][2] = SeedInstanceAnnotation(2, **complete)
        window._draft_seed_annotations[key][3] = SeedInstanceAnnotation(3, **complete)
        window._next_unannotated_instance()
        self.assertEqual(window.instance_id_spin.value(), 2)
        self.assertIn("Every painted seed", window.statusBar().currentMessage())

    def check_test_hilum_tracks_outline_edits_undo_and_persistence(self):
        from seedvision.persistence.reference_regions import ReferenceRegionStore
        window, image_path = self.make_window()
        key = window._current_image_key()
        labels = np.zeros((120, 160), np.uint16)
        labels[20:41, 30:71] = 1  # Centre (50, 30).
        window.image_view.set_instance_annotations(labels)
        window._instance_annotations_edited(labels)
        window.instance_id_spin.setValue(1)
        self.assertIn("861 interior pixels", window._project_summary_text())
        self.assertFalse(hasattr(window, "instance_annotation_status_label"))
        window._hilum_landmark_edited((70., 45.), (-1., 0.))  # Ignore arbitrary supplied direction.
        expected = outward_hilum_direction((50, 30), (70, 45))
        np.testing.assert_allclose(window._draft_seed_annotations[key][1].hilum_direction, expected)
        np.testing.assert_allclose(window.image_view._hilum_direction, expected)
        self.assertFalse(hasattr(window, "seed_hilum_direction_label"))
        self.assertTrue(window.seed_hilum_clear_button.isEnabled())
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
        window._reference_transforms[key] = np.eye(3)
        with patch.object(window, "_analyze_current_image"):
            window._apply_instance_annotations()
        restored = ReferenceRegionStore(image_path.parent.parent).load_if_present(image_path, labels.shape)
        self.assertIsNotNone(restored)
        np.testing.assert_allclose(restored.seed_annotations[0].hilum_direction, expected)
        np.testing.assert_array_equal(restored.annotated_seeds, labels)
        window.instance_id_spin.setValue(2)
        self.assertFalse(window.seed_hilum_clear_button.isEnabled())
        self.assertIsNone(window.image_view._hilum_direction)
        window.instance_id_spin.setValue(1)
        window.seed_hilum_clear_button.click()
        try:
            cleared = window._draft_seed_annotations.get(key, {}).get(1)
            self.assertTrue(cleared is None or cleared.hilum_point is None)
            self.assertTrue(cleared is None or cleared.hilum_direction is None)
        finally:
            window._revert_instance_annotations()

    def check_test_direction_is_unknown_at_centroid_and_removed_without_landmark(self):
        from seedvision.persistence.reference_regions import SeedInstanceAnnotation
        from seedvision.ui.main_window import MainWindow
        labels = np.ones((3, 3), np.uint16)
        for point in ((1., 1.), None):
            derived = MainWindow._derive_seed_hilum_directions(labels, {
                1: SeedInstanceAnnotation(1, hilum_point=point, hilum_direction=(1., 0.))})
            self.assertIsNone(derived[1].hilum_direction)

    def check_test_brush_publishes_fresh_centroids_before_edit_listeners_run(self):
        from PySide6.QtCore import Qt
        from PySide6.QtTest import QTest
        from seedvision.ui.image_view import ImageView
        _, image_path = self.make_window()
        view = ImageView()
        self.addCleanup(view.close)
        view.resize(500, 400)
        view.show()
        view.load_image(image_path)
        view._set_bgr_base_image(np.zeros((120, 160, 3), np.uint8))
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

    def check_test_next_empty_does_not_reuse_an_occupied_seed_at_capacity(self):
        window, _ = self.make_window()
        key = window._current_image_key()
        window._draft_instance_annotations[key] = np.arange(10000, dtype=np.uint16).reshape(100, 100)
        window.instance_id_spin.setValue(5)
        window._new_instance_annotation()
        self.assertEqual(window.instance_id_spin.value(), 5)
        self.assertIn("9,999", window.statusBar().currentMessage())

    def test_workflow_lifecycle(self):
        """11 related AnnotationCentresUiTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_crosshairs_span_a_third_of_the_median_seed_at_every_zoom)
        self.scenario(self.check_test_centre_markers_follow_the_image_except_in_the_editor)
        self.scenario(self.check_test_rejected_proposals_show_only_on_the_centre_overlay_with_their_reason)
        self.scenario(self.check_test_compact_seed_selector_and_no_routine_explanatory_labels)
        self.scenario(self.check_test_crosshairs_follow_masks_selection_visibility_opacity_and_image_switch)
        self.scenario(self.check_test_annotation_palette_can_extend_beyond_image_view)
        self.scenario(self.check_test_annotations_only_appear_on_deskewed_image)
        self.scenario(self.check_test_hilum_tracks_outline_edits_undo_and_persistence)
        self.scenario(self.check_test_direction_is_unknown_at_centroid_and_removed_without_landmark)
        self.scenario(self.check_test_brush_publishes_fresh_centroids_before_edit_listeners_run)
        self.scenario(self.check_test_next_empty_does_not_reuse_an_occupied_seed_at_capacity)

    def test_validation_and_recovery(self):
        """1 related AnnotationCentresUiTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_next_unannotated_cycles_painted_seeds_missing_condition_or_shape)
