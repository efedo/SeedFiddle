from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class ReferenceEdgeToolbarRevisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtGui import QFont, QFontDatabase
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])
        cls.old_font = cls.app.font()
        # Offscreen Windows Qt has no system font database unless loaded.
        font = Path("C:/Windows/Fonts/segoeui.ttf")
        if font.exists():
            QFontDatabase.addApplicationFont(str(font))
            cls.app.setFont(QFont("Segoe UI", 9))

    @classmethod
    def tearDownClass(cls):
        cls.app.setFont(cls.old_font)

    def test_toolbar_toggles_and_opacity_are_display_only(self):
        from seedvision.ui.main_window import MainWindow
        from seedvision.persistence import analysis_settings_profile_from_graph

        window = MainWindow(ROOT)
        self.addCleanup(window.close)
        window.resize(1360, 920)
        window.show()
        self.app.processEvents()
        profile = analysis_settings_profile_from_graph(window.pipeline)
        image_transform = window.image_view.transform()
        graph_transform = window.pipeline_canvas.transform()
        self.assertEqual(window.analyze_action.text(), "Run pipeline")
        actions = window.workflow_toolbar.actions()
        for removed in (window.fit_action, window.actual_size_action, window.split_workspace_action):
            self.assertNotIn(removed, actions)
        for image, pipeline in ((False, True), (True, False), (False, False), (True, True)):
            window.image_workspace_action.setChecked(image)
            window.pipeline_workspace_action.setChecked(pipeline)
            self.app.processEvents()
            self.assertEqual(not window.image_view.isHidden(), image)
            self.assertEqual(not window.pipeline_canvas.isHidden(), pipeline)
        self.assertEqual(window.image_view.transform(), image_transform)
        self.assertEqual(window.pipeline_canvas.transform(), graph_transform)
        self.assertIn("inset", window.workflow_toolbar.styleSheet())
        self.assertEqual(window.overlay_opacity_slider.maximumWidth(), 102)
        self.assertTrue(window.reference_visibility_controls.isVisible())
        self.assertTrue(window.annotation_opacity_slider.isVisible())
        # Annotation opacity stays next to its checkboxes on a wide toolbar.
        window.resize(1920, 1080)
        self.app.processEvents()
        gap = (window.annotation_opacity_slider.x()
               - window.reference_visibility_controls.geometry().right())
        self.assertLess(gap, 80)
        self.assertFalse(window.reference_panel.isAncestorOf(window.reference_visibility_controls))
        window.annotation_opacity_slider.setValue(23)
        window.show_material_references_checkbox.setChecked(False)
        window.show_instance_annotations_checkbox.setChecked(False)
        self.assertAlmostEqual(window.image_view._annotation_opacity, 0.23)
        self.assertAlmostEqual(window.image_view._overlay_opacity, 0.68)
        self.assertEqual(analysis_settings_profile_from_graph(window.pipeline), profile)
        # Existing commands that reveal the image must also check its toggle.
        from unittest.mock import patch
        window.image_workspace_action.setChecked(False)
        with patch.object(window, "_open_path"):
            window._open_list_item(window.image_list.currentItem())
        self.assertTrue(window.image_workspace_action.isChecked())
        self.assertFalse(window.image_view.isHidden())

    def test_pipeline_zoom_controls_fit_narrow_and_wide_panes(self):
        from seedvision.pipeline import build_default_pipeline
        from seedvision.ui.pipeline_canvas import PipelineCanvas

        canvas = PipelineCanvas(build_default_pipeline())
        self.addCleanup(canvas.close)
        canvas.show()
        for width in (420, 900, 1400, 420):
            canvas.resize(width, 400)
            self.app.processEvents()
            for button in (canvas.fit_button, canvas.actual_size_button):
                self.assertTrue(button.isVisible())
                self.assertTrue(canvas.control_bar.rect().contains(button.geometry()))
            self.assertLessEqual(canvas.control_bar.geometry().right(), canvas.viewport().geometry().right())
            if width == 420:
                self.assertTrue(canvas.tools_button.isVisible())
                canvas._populate_tools_menu()
                self.assertEqual(len(canvas.tools_menu.actions()), 5)
        canvas.scale(0.4, 0.4)
        canvas.actual_size_button.click()
        self.assertEqual(canvas.transform().m11(), 1.0)
        self.assertEqual(canvas.zoom_label.text(), "100%")

    def test_annotation_opacity_does_not_touch_analytical_items_or_masks(self):
        from seedvision.ui.image_view import ImageView

        view = ImageView()
        self.addCleanup(view.close)
        mask = np.zeros((24, 24), bool)
        mask[3:8, 3:8] = True
        labels = np.zeros((24, 24), np.uint16)
        labels[12:20, 12:20] = 1
        view.set_reference_masks(mask, None, render=False)
        view.set_instance_annotations(labels, render=False)
        view._render_background_reference_points()
        view._render_instance_annotations()
        analytic = view._scene.addRect(0, 0, 24, 24)
        view._overlay_items.append(analytic)
        fixed = view._scene.addRect(0, 0, 1, 1)
        fixed.setData(0, "fixed-opacity-overlay-annotation")
        view._overlay_items.append(fixed)
        view.set_overlay_opacity(0.41)
        view.set_annotation_opacity(0.19)
        authored = [item for item in view._overlay_items if item.data(1) == "reference-annotation"]
        self.assertEqual(len(authored), 3)  # Material mask, instance mask, annotation centres.
        for item in authored:
            self.assertAlmostEqual(item.opacity(), 0.19)
        self.assertAlmostEqual(analytic.opacity(), 0.41)
        self.assertEqual(fixed.opacity(), 1.0)
        view.set_overlay_opacity(0.9)
        for item in authored:
            self.assertAlmostEqual(item.opacity(), 0.19)
        view._render_instance_annotations()
        self.assertAlmostEqual(view._instance_annotation_overlay_item.opacity(), 0.19)
        np.testing.assert_array_equal(view.reference_mask("background"), mask)
        np.testing.assert_array_equal(view._instance_annotations, labels)

    def test_selected_node_has_visible_cosmetic_outline_at_overview_zoom(self):
        from PySide6.QtGui import QImage, QPainter
        from PySide6.QtWidgets import QStyleOptionGraphicsItem
        from seedvision.pipeline import build_default_pipeline
        from seedvision.ui.pipeline_canvas import PipelineCanvas

        canvas = PipelineCanvas(build_default_pipeline())
        self.addCleanup(canvas.close)
        item = canvas.node_items["reference_edge_probability"]
        counts = []
        for selected in (False, True):
            item.setSelected(selected)
            image = QImage(200, 300, QImage.Format.Format_RGBA8888)
            image.fill(0)
            painter = QPainter(image)
            painter.translate(8, 8)
            painter.scale(0.12, 0.12)
            item.paint(painter, QStyleOptionGraphicsItem())
            painter.end()
            pixels = np.frombuffer(image.constBits(), np.uint8).reshape(300, 200, 4)
            cyan = (pixels[:, :, 0] == 69) & (pixels[:, :, 1] == 221) & (pixels[:, :, 2] == 255)
            counts.append(int(cyan.sum()))
        self.assertGreater(counts[1], counts[0] + 100)
        self.assertEqual(item.zValue(), 4.0)

    def test_colour_overlay_order_and_other_node_socket(self):
        from seedvision.ui.main_window import MainWindow, _overlay_node_owner, _overlay_output_port_id

        window = MainWindow(ROOT)
        self.addCleanup(window.close)
        entries = [(label, mode) for label, mode in window._overlay_entries
                   if _overlay_node_owner(mode) == "background_likelihood"]
        self.assertEqual([mode for _, mode in entries], [
            "foreground_mask", "foreground_colour_gamut", "background_likelihood",
            "background_colour_gamut", "other_colour_probability", "other_colour_gamut",
            "foreground_colour_excess",
        ])
        ports = dict(window.pipeline.node("background_likelihood").output_ports)
        self.assertEqual(ports[_overlay_output_port_id("other_colour_gamut")], "Accepted other colours (HSV)")
        window.overlay_combo.setCurrentIndex(window.overlay_combo.findData("other_colour_gamut"))
        self.app.processEvents()
        self.assertEqual(window.image_view._overlay_mode, "other_colour_gamut")
        self.assertTrue(window.hsv_value_slider_action.isVisible())
        self.assertFalse(window.overlay_opacity_slider_action.isVisible())

    def test_other_hsv_matches_production_frequency_model_without_refitting(self):
        import cv2
        import torch
        from seedvision.cuda.ops import lab_colour_frequency_distribution
        from seedvision.ui.pipeline_inspector import BackgroundColourGamut
        from seedvision.visualization import BackgroundColourProfile

        # The exact HSV sampling grid used by the plot at 640 x 440.
        hsv = np.empty((246, 506, 3), np.uint8)
        hsv[:, :, 0] = np.rint(np.linspace(0, 179, 506)).astype(np.uint8)[None, :]
        hsv[:, :, 1] = np.rint(np.linspace(0, 255, 246)).astype(np.uint8)[:, None]
        hsv[:, :, 2] = 204
        lab = cv2.cvtColor(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB), cv2.COLOR_RGB2LAB).astype(np.float32)
        samples = torch.tensor(np.repeat(lab[[110, 180], [135, 290]], [20, 3], axis=0))
        probability, centres, scales, weights, *_ = lab_colour_frequency_distribution(
            torch.tensor(lab), samples, torch.ones(lab.shape[:2], dtype=torch.bool),
            refinement_iterations=0, scale_multiplier=1.7, frequency_weight_power=0.0,
        )
        tuples = lambda tensor: tuple(tuple(map(float, row)) for row in tensor.tolist())
        profile = BackgroundColourProfile(
            centre_lab=(220., 128., 128.), scale_lab=(8., 4., 4.),
            bgr_low=(200, 200, 200), bgr_high=(255, 255, 255), sample_count=500,
            sample_fraction=0.2, excluded_component_centres_lab=tuples(centres),
            excluded_component_scales_lab=tuples(scales),
            excluded_component_weights=tuple(map(float, weights.tolist())),
        )
        other = BackgroundColourGamut.profile_for_class(SimpleNamespace(background_colour_profile=profile), "other")
        parameters = BackgroundColourGamut.parameters_for_class({
            "background_distribution_scale_multiplier": 1.7,
            "background_chroma_weight": 7.0, "background_frequency_weight_power": 1.0,
        }, "other")
        image, displayed, _ = BackgroundColourGamut.render_hsv_value_slice(
            other, parameters, class_name="other", value=0.8, width=640, height=440,
        )
        # The first few achromatic rows are intentionally isolated to the
        # neutral swatch, not repeated across all hue coordinates.
        np.testing.assert_allclose(displayed[12:], probability.numpy()[12:], atol=2e-6)
        self.assertEqual(other.component_centres_lab, profile.excluded_component_centres_lab)
        self.assertEqual(image.width(), 640)
        self.assertEqual(profile.centre_lab, (220., 128., 128.))
        empty, values, empty_centres = BackgroundColourGamut.render_hsv_value_slice(None, {}, class_name="other", value=0.8)
        self.assertFalse(values.any())
        self.assertFalse(empty.isNull())
        self.assertEqual(empty_centres.shape, (0, 2))
