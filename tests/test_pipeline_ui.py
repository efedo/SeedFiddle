from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]


class PipelineCanvasTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        try:
            from PySide6.QtWidgets import QApplication
        except ImportError as error:
            raise unittest.SkipTest(f"PySide6 unavailable: {error}")
        cls.application = QApplication.instance() or QApplication([])

    def test_canvas_contains_default_nodes_and_live_edges(self) -> None:
        from seedvision.pipeline import build_default_pipeline
        from seedvision.ui.pipeline_canvas import PipelineCanvas

        canvas = PipelineCanvas(build_default_pipeline())
        canvas.resize(900, 360)
        canvas.show()
        self.application.processEvents()
        canvas.fit_graph()
        self.assertGreater(canvas.horizontalScrollBar().maximum(), 0)
        self.assertEqual(len(canvas.node_items), 41)
        self.assertEqual(len(canvas.edge_items), 89)
        for calibration_node_id in (
            "colour_reference",
            "ruler_detection",
            "deskew_colour",
            "scale_calibration",
        ):
            self.assertIn(calibration_node_id, canvas.node_items)
        for overlay_node_id in (
            "instance_masks",
            "background_likelihood",
            "refined_background_likelihood",
            "edge_gradients",
            "undirected_edges",
            "directed_edges",
            "edge_ridges",
            "edge_traces",
            "seed_edge_curves",
        ):
            self.assertIn(overlay_node_id, canvas.node_items)
        for stage_node_id in (
            "seed_scale_estimation",
            "foreground_segmentation",
            "distance_candidates",
            "circle_candidates",
            "identification",
        ):
            self.assertIn(stage_node_id, canvas.node_items)
        for advanced_node_id in (
            "seed_interior",
            "boundary_normals",
            "touching_split",
            "ellipse_likelihood",
            "proposal_disagreement",
            "assignment_confidence",
            "contact_graph",
            "illumination_decomposition",
            "image_quality",
            "radial_profile",
            "wrinkling",
            "coat_damage",
            "pattern_decomposition",
            "colour_probabilities",
            "calibration_residuals",
        ):
            self.assertIn(advanced_node_id, canvas.node_items)
        edge = canvas.edge_items[0]
        old_end = edge.path().pointAtPercent(1.0)
        target = edge.target
        target.setPos(target.pos().x() + 25, target.pos().y() + 15)
        new_end = edge.path().pointAtPercent(1.0)
        self.assertNotEqual(old_end, new_end)
        shared = canvas.node_items["edge_gradients"]
        self.assertNotEqual(
            shared.output_anchor("undirected"),
            shared.output_anchor("directed"),
        )
        canvas.close()

    def test_node_cards_support_typed_blueprint_controls_and_time_footers(self) -> None:
        from PySide6.QtWidgets import QCheckBox, QComboBox

        from seedvision.pipeline import build_default_pipeline
        from seedvision.ui.pipeline_canvas import (
            PipelineCanvas,
            _format_calculation_time,
        )

        graph = build_default_pipeline()
        graph.node("edge_gradients").calculation_seconds = 0.01234
        canvas = PipelineCanvas(graph)
        self.assertIsInstance(
            canvas.node_items["colour_reference"]._inline_editors[
                "apply_colour_balance"
            ],
            QCheckBox,
        )
        self.assertIsInstance(
            canvas.node_items["refined_background_likelihood"]._inline_editors[
                "noise_direction_integration"
            ],
            QComboBox,
        )
        edge_item = canvas.node_items["edge_gradients"]
        controls_bottom = (
            edge_item._controls_y
            + (len(edge_item.node.inline_parameters) - 1) * edge_item.ROW_HEIGHT
            + 22.0
        )
        self.assertGreaterEqual(edge_item._footer_y, controls_bottom + 4.0)
        self.assertEqual(edge_item.height, edge_item._footer_y + 24.0)
        self.assertLess(canvas.node_items["colour_reference"].height, 150.0)
        self.assertEqual(canvas.node_items["undirected_edges"].height, 108.0)
        self.assertEqual(_format_calculation_time(0.01234), "Last calc: 12.3 ms")
        self.assertEqual(_format_calculation_time(1.25), "Last calc: 1.25 s")
        canvas.close()

    def test_graph_layout_and_node_drop_prevent_overlaps(self) -> None:
        from seedvision.pipeline import build_default_pipeline
        from seedvision.ui.pipeline_canvas import PipelineCanvas

        canvas = PipelineCanvas(build_default_pipeline())
        items = tuple(canvas.node_items.values())
        for index, first in enumerate(items):
            for second in items[index + 1:]:
                self.assertFalse(
                    canvas._items_overlap(first, float(first.pos().y()), second),
                    (first.node.identifier, second.node.identifier),
                )

        moved = canvas.node_items["colour_reference"]
        obstacle = canvas.node_items["deskew_colour"]
        moved.setPos(obstacle.pos())
        canvas._node_released(moved.node.identifier)
        self.assertFalse(
            canvas._items_overlap(moved, float(moved.pos().y()), obstacle)
        )
        canvas.close()

    def test_worker_progress_turns_only_the_current_revision_node_green(self) -> None:
        from seedvision.pipeline import NodeStatus
        from seedvision.ui.main_window import MainWindow

        path = ROOT / "images" / "IMG_9670c.JPG"
        if not path.exists():
            self.skipTest("Pilot image is not present")
        window = MainWindow(ROOT)
        succeeded, error = window.image_view.load_image(path)
        self.assertTrue(succeeded, error)
        node = window.pipeline.node("layout_detection")
        node.status = NodeStatus.RUNNING
        window._analysis_node_progress(
            str(path), "layout_detection", "completed", window.pipeline.revision
        )
        self.assertEqual(node.status, NodeStatus.COMPLETE)
        self.assertEqual(node.status_detail, "Calculated")

        node.status = NodeStatus.RUNNING
        window._analysis_node_progress(
            str(path),
            "layout_detection",
            "completed",
            window.pipeline.revision + 1,
        )
        self.assertEqual(node.status, NodeStatus.RUNNING)
        window.close()

    def test_identification_stages_expose_methods_and_relevant_parameters(self) -> None:
        from seedvision.pipeline import build_default_pipeline
        from seedvision.ui.pipeline_inspector import PipelineInspector

        graph = build_default_pipeline()
        inspector = PipelineInspector()
        expected_parameter_counts = {
            "seed_scale_estimation": 9,
            "foreground_segmentation": 15,
            "distance_candidates": 4,
            "circle_candidates": 8,
            "identification": 3,
        }
        for node_id, parameter_count in expected_parameter_counts.items():
            inspector.set_node(graph.node(node_id))
            self.assertEqual(len(inspector._parameter_widgets), parameter_count)
            self.assertTrue(inspector.method_heading.isVisibleTo(inspector))
            self.assertFalse(inspector.method_heading.isChecked())
            self.assertFalse(inspector.details_label.isVisibleTo(inspector))
            inspector.method_heading.click()
            self.assertTrue(inspector.details_label.isVisibleTo(inspector))
            self.assertGreater(len(inspector.details_label.text()), 80)
        inspector.set_node(graph.node("foreground_segmentation"))
        self.assertFalse(inspector.details_label.isVisibleTo(inspector))
        self.assertEqual(inspector.title_label.text(), "Dish foreground mask")
        self.assertEqual(inspector.parameter_form.rowCount(), 15)
        self.assertTrue(
            all(widget.toolTip() for widget in inspector._parameter_widgets)
        )
        for node_id, minimum_count in {
            "layout_detection": 8,
            "background_likelihood": 11,
            "refined_background_likelihood": 9,
            "edge_gradients": 4,
            "undirected_edges": 0,
            "directed_edges": 0,
            "instance_masks": 3,
            "edge_ridges": 4,
            "edge_traces": 6,
            "seed_edge_curves": 19,
        }.items():
            inspector.set_node(graph.node(node_id))
            self.assertGreaterEqual(
                len(inspector._parameter_widgets), minimum_count, node_id
            )
        inspector.close()

    def test_background_node_shows_probability_contours_over_colour_gamut(self) -> None:
        from seedvision.pipeline import build_default_pipeline
        from seedvision.ui.pipeline_inspector import PipelineInspector
        from seedvision.visualization import BackgroundColourProfile

        profile = BackgroundColourProfile(
            centre_lab=(224.0, 127.0, 132.0),
            scale_lab=(9.0, 5.0, 7.0),
            bgr_low=(205, 208, 198),
            bgr_high=(241, 244, 238),
            sample_count=2400,
            sample_fraction=0.12,
            component_centres_lab=((224.0, 127.0, 132.0), (210.0, 138.0, 118.0)),
            component_scales_lab=((9.0, 5.0, 7.0), (7.0, 6.0, 5.0)),
            component_weights=(0.75, 0.25),
            refinement_iterations=2,
        )
        inspector = PipelineInspector()
        inspector.set_analysis_result(
            SimpleNamespace(
                layers=SimpleNamespace(
                    background_colour_profile=profile,
                    background_mode="automatic",
                )
            )
        )
        inspector.set_node(build_default_pipeline().node("background_likelihood"))

        self.assertTrue(inspector.gamut_heading.isVisibleTo(inspector))
        self.assertFalse(inspector.gamut_widget._image.isNull())
        self.assertGreater(float(inspector.gamut_widget._probability.max()), 0.95)
        self.assertLess(float(inspector.gamut_widget._probability.min()), 0.25)
        self.assertIn("25/50/75/90%", inspector.gamut_caption.text())
        inspector.close()

    def test_foreground_node_shows_its_probability_colour_gamut(self) -> None:
        from seedvision.pipeline import build_default_pipeline
        from seedvision.ui.pipeline_inspector import PipelineInspector
        from seedvision.visualization import ForegroundColourProfile

        profile = ForegroundColourProfile(
            centre_lab=(152.0, 141.0, 118.0),
            scale_lab=(18.0, 9.0, 8.0),
            bgr_low=(36, 62, 80),
            bgr_high=(211, 218, 224),
            sample_count=1800,
            sample_fraction=0.34,
            component_centres_lab=((152.0, 141.0, 118.0), (205.0, 128.0, 130.0)),
            component_scales_lab=((18.0, 9.0, 8.0), (11.0, 5.0, 6.0)),
            component_weights=(0.62, 0.38),
            refinement_iterations=2,
        )
        inspector = PipelineInspector()
        inspector.set_analysis_result(
            SimpleNamespace(
                foreground_reference_count=420,
                layers=SimpleNamespace(foreground_colour_profile=profile),
            )
        )
        inspector.set_node(build_default_pipeline().node("foreground_segmentation"))

        self.assertTrue(inspector.gamut_heading.isVisibleTo(inspector))
        self.assertEqual(inspector.gamut_heading.text(), "Accepted foreground colours")
        self.assertEqual(inspector.gamut_widget._class_name, "foreground")
        self.assertFalse(inspector.gamut_widget._image.isNull())
        self.assertIn("painted-reference", inspector.gamut_caption.text())
        inspector.close()

    def test_overlay_node_selection_updates_the_viewer_layer(self) -> None:
        from seedvision.ui.main_window import MainWindow

        window = MainWindow(ROOT)
        window.pipeline_canvas.select_node("directed_edges")
        self.application.processEvents()
        self.assertEqual(window.overlay_combo.currentData(), "directed_edges")
        self.assertEqual(window.image_view._overlay_mode, "directed_edges")
        self.assertEqual(window.pipeline_inspector.title_label.text(), "Directed edge tangents")
        self.assertEqual(len(window.pipeline_inspector._parameter_widgets), 0)
        window.pipeline_canvas.select_node("edge_gradients")
        self.application.processEvents()
        self.assertEqual(window.overlay_combo.currentData(), "edge_gradients")
        self.assertEqual(window.image_view._overlay_mode, "edge_gradients")
        self.assertEqual(len(window.pipeline_inspector._parameter_widgets), 4)
        self.assertEqual(
            len(window.pipeline_canvas.node_items["edge_gradients"]._inline_editors),
            4,
        )
        window.pipeline_canvas.select_node("refined_background_likelihood")
        self.application.processEvents()
        self.assertEqual(
            window.overlay_combo.currentData(), "refined_background_likelihood"
        )
        self.assertEqual(
            window.pipeline_inspector.title_label.text(),
            "Background noise profile",
        )
        self.assertEqual(len(window.pipeline_inspector._parameter_widgets), 10)
        window.pipeline_canvas.select_node("seed_edge_curves")
        self.application.processEvents()
        self.assertEqual(window.overlay_combo.currentData(), "seed_edge_curves")
        self.assertEqual(
            window.pipeline_inspector.title_label.text(),
            "Seed-boundary confirmation",
        )
        self.assertEqual(len(window.pipeline_inspector._parameter_widgets), 19)
        window.pipeline_canvas.select_node("boundary_normals")
        self.application.processEvents()
        self.assertEqual(window.overlay_combo.currentData(), "boundary_confidence")
        self.assertEqual(window.image_view._overlay_mode, "boundary_confidence")
        self.assertEqual(len(window.pipeline_inspector._parameter_widgets), 1)
        window.pipeline_canvas.select_node("ruler_detection")
        self.application.processEvents()
        self.assertEqual(window.overlay_combo.currentData(), "ruler_detection")
        window.pipeline_canvas.select_node("deskew_colour")
        self.application.processEvents()
        self.assertEqual(window.overlay_combo.currentData(), "deskew_colour")
        window.pipeline_canvas.select_node("scale_calibration")
        self.application.processEvents()
        self.assertEqual(window.overlay_combo.currentData(), "calibrated_image")
        window.close()

    def test_viewer_layer_selection_selects_its_owning_graph_node(self) -> None:
        from seedvision.ui.main_window import MainWindow

        window = MainWindow(ROOT)
        index = window.overlay_combo.findData("boundary_magnitude")
        self.assertGreaterEqual(index, 0)
        window.overlay_combo.setCurrentIndex(index)
        self.application.processEvents()
        self.assertEqual(window._selected_pipeline_node, "boundary_normals")
        self.assertEqual(
            window.overlay_owner_label.text(), "Boundary confidence & normals"
        )
        highlighted = [
            edge
            for edge in window.pipeline_canvas.edge_items
            if edge.pen().widthF() >= 4.0
        ]
        self.assertTrue(highlighted)
        self.assertTrue(
            all(
                "boundary_normals"
                in (edge.connection.source, edge.connection.target)
                for edge in highlighted
            )
        )
        window.close()

    def test_node_selection_shows_stage_overlay_sections_and_connections(self) -> None:
        from seedvision.ui.main_window import MainWindow

        window = MainWindow(ROOT)
        window.show()
        self.application.processEvents()
        self.assertGreaterEqual(window.metadata_scroll.verticalScrollBar().width(), 16)

        window.pipeline_canvas.select_node("foreground_segmentation")
        self.application.processEvents()
        self.assertEqual(window.overlay_combo.currentData(), "foreground_mask")
        self.assertFalse(window.calibration_section.isVisible())
        self.assertFalse(window.baseline_section.isVisible())
        highlighted = [
            edge
            for edge in window.pipeline_canvas.edge_items
            if edge.pen().widthF() >= 4.0
        ]
        self.assertTrue(highlighted)
        self.assertTrue(
            all(
                "foreground_segmentation"
                in (edge.connection.source, edge.connection.target)
                for edge in highlighted
            )
        )

        window.pipeline_canvas.select_node("deskew_colour")
        self.application.processEvents()
        self.assertTrue(window.calibration_section.isVisible())
        self.assertFalse(window.baseline_section.isVisible())
        window.pipeline_canvas.select_node("identification")
        self.application.processEvents()
        self.assertFalse(window.calibration_section.isVisible())
        self.assertTrue(window.baseline_section.isVisible())
        self.assertFalse(window.analyze_button.isVisible())
        self.assertFalse(window.warning_label.isVisible())
        window.close()

    def test_image_and_pipeline_share_a_split_workspace(self) -> None:
        from seedvision.ui.main_window import MainWindow

        window = MainWindow(ROOT)
        self.assertEqual(window.workspace_splitter.count(), 2)
        self.assertIs(window.workspace_splitter.widget(0), window.image_view)
        self.assertIs(window.workspace_splitter.widget(1), window.pipeline_canvas)
        self.assertFalse(window.image_view.isHidden())
        self.assertFalse(window.pipeline_canvas.isHidden())
        self.assertRegex(
            window.pipeline_inspector.details_label.styleSheet(),
            r"#(?:d2d9e1|303b46)",
        )
        window.close()

    def test_background_node_can_be_disabled_without_disabling_identification(self) -> None:
        from seedvision.pipeline import NodeStatus
        from seedvision.ui.main_window import MainWindow

        window = MainWindow(ROOT)
        window._pipeline_enabled_changed("background_likelihood", False)
        self.assertFalse(window.pipeline.node("background_likelihood").enabled)
        self.assertTrue(window.pipeline.node("identification").enabled)
        self.assertEqual(
            window.pipeline.node("background_likelihood").status,
            NodeStatus.BYPASSED,
        )
        self.assertEqual(
            window.pipeline.node("refined_background_likelihood").status,
            NodeStatus.BLOCKED,
        )
        self.assertFalse(window.background_enabled_checkbox.isChecked())
        window.close()

    def test_expanded_node_settings_feed_runtime_settings_objects(self) -> None:
        from seedvision.ui.main_window import MainWindow

        window = MainWindow(ROOT)
        window.pipeline.set_parameter(
            "layout_detection", "hough_accumulator_threshold", 41
        )
        window.pipeline.set_parameter(
            "distance_candidates", "distance_min_depth_fraction", 0.20
        )
        window.pipeline.set_parameter(
            "seed_edge_curves", "boundary_arc_span_degrees", 84.0
        )
        window.pipeline.set_parameter("edge_gradients", "edge_chroma_weight", 1.8)
        window.pipeline.set_parameter(
            "background_likelihood", "background_colour_components", 3
        )
        window.pipeline.set_parameter(
            "foreground_segmentation", "foreground_refinement_iterations", 3
        )

        self.assertEqual(window._dish_settings().hough_accumulator_threshold, 41)
        self.assertAlmostEqual(
            window._baseline_settings().distance_min_depth_fraction, 0.20
        )
        self.assertAlmostEqual(
            window._layer_settings().boundary_arc_span_degrees, 84.0
        )
        self.assertAlmostEqual(
            window._layer_settings().edge_chroma_weight,
            1.8,
        )
        self.assertEqual(window._layer_settings().background_colour_components, 3)
        self.assertEqual(
            window._baseline_settings().foreground_refinement_iterations, 3
        )
        self.assertEqual(
            window.pipeline.node("edge_gradients").parameters["edge_chroma_weight"],
            1.8,
        )
        inline_editor = window.pipeline_canvas.node_items[
            "edge_gradients"
        ]._inline_editors["edge_chroma_weight"]
        inline_editor.setValue(1.9)
        inline_editor.editingFinished.emit()
        self.application.processEvents()
        self.assertAlmostEqual(window._layer_settings().edge_chroma_weight, 1.9)
        window.pipeline.set_parameter(
            "seed_interior", "maximum_dimension", 768
        )
        window.pipeline.set_parameter(
            "wrinkling", "wrinkle_scale_fraction", 0.05
        )
        self.assertEqual(window._advanced_settings().maximum_dimension, 768)
        self.assertAlmostEqual(
            window._advanced_settings().wrinkle_scale_fraction, 0.05
        )
        with self.assertRaises(ValueError):
            window._validate_settings_override(
                "seed_edge_curves", "boundary_radius_min_fraction", 1.0
            )
        window.close()

    def test_setting_change_invalidates_only_node_and_graph_dependents(self) -> None:
        from seedvision.pipeline import NodeStatus
        from seedvision.ui.main_window import MainWindow

        window = MainWindow(ROOT)
        for node_id in (
            "edge_gradients",
            "undirected_edges",
            "directed_edges",
            "seed_edge_curves",
        ):
            window.pipeline.set_status(node_id, NodeStatus.COMPLETE, "Ready")
        window._pipeline_parameter_changed(
            "edge_gradients", "edge_strength_gamma", 0.72
        )
        self.assertEqual(
            window.pipeline.node("edge_gradients").status, NodeStatus.IDLE
        )
        self.assertEqual(
            window.pipeline.node("directed_edges").status, NodeStatus.IDLE
        )
        self.assertEqual(
            window.pipeline.node("seed_edge_curves").status, NodeStatus.IDLE
        )
        self.assertEqual(
            window.pipeline.node("undirected_edges").status, NodeStatus.IDLE
        )
        window.close()

    def test_image_view_switches_between_analysis_layers(self) -> None:
        import numpy as np

        from seedvision.ui.image_view import ImageView
        from seedvision.visualization import AnalysisLayers, NoiseFrequencyProfile

        labels = np.zeros((12, 12), dtype=np.uint16)
        labels[3:9, 3:9] = 1
        layers = AnalysisLayers(
            offset_x=4,
            offset_y=5,
            instance_labels=labels,
            instance_colours=np.asarray(((0, 0, 0), (240, 50, 80)), np.uint8),
            background_likelihood=np.full((12, 12), 180, np.uint8),
            refined_background_likelihood=np.full((12, 12), 160, np.uint8),
            noise_frequency_profile=NoiseFrequencyProfile(
                band_scales_px=(0.8, 1.8, 4.0),
                background_log_rms=(0.1, 0.2, 0.3),
                nonbackground_log_rms=(0.3, 0.2, 0.1),
                background_sample_count=100,
                nonbackground_sample_count=100,
                separation=1.0,
            ),
            edge_likelihood=np.full((12, 12), 120, np.uint8),
            directed_edge_hue=np.full((12, 12), 45, np.uint8),
            undirected_edge_hue=np.full((12, 12), 90, np.uint8),
            seed_edge_curve_likelihood=np.full((12, 12), 100, np.uint8),
            seed_edge_curve_radius_px=np.full((12, 12), 4.0, np.float32),
            valid_mask=np.full((12, 12), 255, np.uint8),
        )
        result = SimpleNamespace(
            dish=SimpleNamespace(center_x=10, center_y=10, radius=8),
            proposals=(
                SimpleNamespace(center_x=10, center_y=10, radius=4),
            ),
            layers=layers,
            crop_offset=(4, 5),
            foreground_probability=np.arange(144, dtype=np.uint8).reshape(12, 12),
            foreground_mask=np.uint8(labels > 0) * 255,
        )
        view = ImageView()
        view.show_analysis(result)
        self.assertEqual(len(view._overlay_items), 3)
        for mode in (
            "instance_masks",
            "background_likelihood",
            "refined_background_likelihood",
            "undirected_edges",
            "directed_edges",
            "seed_edge_curves",
            "foreground_mask",
            "foreground_binary_mask",
        ):
            view.set_overlay_mode(mode)
            self.assertEqual(len(view._overlay_items), 1)
        view.set_overlay_opacity(0.42)
        self.assertAlmostEqual(view._overlay_items[0].opacity(), 0.42)
        view.set_overlay_mode("raw_image")
        self.assertEqual(len(view._overlay_items), 0)
        view.set_overlay_mode("none")
        self.assertEqual(len(view._overlay_items), 0)
        view.close()

    def test_image_view_edits_full_resolution_binary_reference_masks(self) -> None:
        from PySide6.QtCore import QPointF, Qt
        from PySide6.QtGui import QColor, QImage
        from PySide6.QtTest import QTest

        from seedvision.ui.image_view import ImageView

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "background-points.png"
            image = QImage(100, 100, QImage.Format.Format_RGB32)
            image.fill(QColor("white"))
            self.assertTrue(image.save(str(path)))

            view = ImageView()
            view.resize(320, 260)
            succeeded, error = view.load_image(path)
            self.assertTrue(succeeded, error)
            view.show()
            self.application.processEvents()
            view.fit_image()
            edits: list[tuple[str, object]] = []
            finished: list[bool] = []
            view.reference_mask_edited.connect(
                lambda class_name, mask: edits.append((class_name, mask))
            )
            view.reference_paint_finished.connect(lambda: finished.append(True))
            view.set_background_point_editing(True)
            position = view.mapFromScene(QPointF(50.0, 50.0))
            view.set_reference_brush_radius(16.0)
            QTest.mouseMove(view.viewport(), position)
            self.application.processEvents()
            self.assertIsNotNone(view._reference_brush_outline_item)
            self.assertTrue(view._reference_brush_outline_item.isVisible())
            self.assertAlmostEqual(
                view._reference_brush_outline_item.rect().width(), 32.0
            )
            QTest.mouseClick(
                view.viewport(), Qt.MouseButton.LeftButton, pos=position
            )
            self.assertEqual(len(edits), 1)
            self.assertEqual(edits[-1][0], "background")
            first_mask = edits[-1][1]
            self.assertEqual(first_mask.shape, (100, 100))
            self.assertEqual(first_mask.dtype.kind, "b")
            self.assertTrue(first_mask[50, 50])
            first_area = int(first_mask.sum())
            self.assertGreater(first_area, 300)
            self.assertEqual(len(finished), 1)
            stroke_start = view.mapFromScene(QPointF(20.0, 25.0))
            stroke_end = view.mapFromScene(QPointF(80.0, 25.0))
            QTest.mousePress(
                view.viewport(), Qt.MouseButton.LeftButton, pos=stroke_start
            )
            QTest.mouseMove(view.viewport(), stroke_end, delay=10)
            QTest.mouseRelease(
                view.viewport(), Qt.MouseButton.LeftButton, pos=stroke_end
            )
            self.assertEqual(len(edits), 2)
            stroke_mask = edits[-1][1]
            self.assertTrue(stroke_mask[25, 20])
            self.assertTrue(stroke_mask[25, 50])
            self.assertTrue(stroke_mask[25, 80])
            self.assertGreater(int(stroke_mask.sum()), first_area)
            self.assertEqual(len(finished), 2)
            view._render_background_reference_points()
            painted_item = view._overlay_items[-1]
            self.assertEqual(painted_item.pixmap().size().width(), 100)
            self.assertEqual(painted_item.pixmap().size().height(), 100)
            QTest.mouseClick(
                view.viewport(), Qt.MouseButton.RightButton, pos=position
            )
            self.assertEqual(len(edits), 3)
            self.assertFalse(edits[-1][1][50, 50])
            self.assertLess(int(edits[-1][1].sum()), int(stroke_mask.sum()))
            QTest.mouseClick(
                view.viewport(), Qt.MouseButton.LeftButton, pos=position
            )
            self.assertTrue(edits[-1][1][50, 50])
            view.set_reference_erase_mode(True)
            self.assertEqual(
                view._reference_brush_outline_item.pen().style(),
                Qt.PenStyle.DashLine,
            )
            QTest.mouseClick(
                view.viewport(), Qt.MouseButton.LeftButton, pos=position
            )
            self.assertFalse(edits[-1][1][50, 50])
            view.set_reference_erase_mode(False)
            view.set_foreground_point_editing(True)
            QTest.mouseClick(
                view.viewport(), Qt.MouseButton.LeftButton, pos=position
            )
            self.assertEqual(edits[-1][0], "foreground")
            self.assertTrue(edits[-1][1][50, 50])
            QTest.mouseClick(
                view.viewport(), Qt.MouseButton.RightButton, pos=position
            )
            self.assertEqual(edits[-1][0], "foreground")
            self.assertFalse(edits[-1][1][50, 50])
            view.close()

    def test_reference_masks_run_analysis_only_after_explicit_apply(self) -> None:
        import numpy as np

        from seedvision.ui.main_window import MainWindow

        window = MainWindow(ROOT)
        key = window._current_image_key()
        if key is None or window.image_view.image_size is None:
            window.close()
            self.skipTest("No workspace image is present")
        width, height = window.image_view.image_size
        mask = np.zeros((height, width), dtype=bool)
        mask[20:35, 30:50] = True
        analyses: list[bool] = []
        window._analyze_current_image = lambda **_kwargs: analyses.append(True)

        window._reference_mask_edited("foreground", mask)
        self.assertEqual(analyses, [])
        self.assertIn(key, window._reference_masks_dirty)
        self.assertNotIn(key, window._applied_foreground_reference_masks)

        window._apply_reference_masks()
        self.assertEqual(analyses, [True])
        self.assertNotIn(key, window._reference_masks_dirty)
        self.assertTrue(
            np.array_equal(window._applied_foreground_reference_masks[key], mask)
        )

        window._clear_foreground_points()
        self.assertEqual(analyses, [True])
        self.assertIn(key, window._reference_masks_dirty)
        self.assertIn(key, window._applied_foreground_reference_masks)
        window._apply_reference_masks()
        self.assertEqual(analyses, [True, True])
        self.assertNotIn(key, window._applied_foreground_reference_masks)
        window.close()

    def test_calibration_nodes_render_references_and_metric_scale(self) -> None:
        from PySide6.QtWidgets import (
            QGraphicsEllipseItem,
            QGraphicsPathItem,
            QGraphicsPixmapItem,
            QGraphicsTextItem,
        )

        from seedvision.pipeline import NodeStatus
        from seedvision.segmentation import analyze_path
        from seedvision.ui.image_view import ImageView
        from seedvision.ui.main_window import MainWindow

        path = ROOT / "images" / "IMG_9670c.JPG"
        if not path.exists():
            self.skipTest("Pilot image is not present")
        result = analyze_path(path)
        view = ImageView()
        succeeded, error = view.load_image(path)
        self.assertTrue(succeeded, error)
        raw_size = view.image_size
        view.show_analysis(result)
        view.set_overlay_mode("colour_reference")
        self.assertEqual(view.image_size, raw_size)
        self.assertGreaterEqual(len(view._overlay_items), 25)
        self.assertGreaterEqual(view._overlay_items[0].pen().widthF(), 9.0)
        self.assertFalse(
            any(
                isinstance(item, QGraphicsTextItem)
                and item.toPlainText() == "5 cm"
                for item in view._overlay_items
            )
        )
        view.set_overlay_mode("deskew_colour")
        self.assertEqual(
            view.image_size,
            (
                result.calibration.corrected_bgr.shape[1],
                result.calibration.corrected_bgr.shape[0],
            ),
        )
        self.assertGreaterEqual(len(view._overlay_items), 25)
        self.assertFalse(
            any(
                isinstance(item, QGraphicsTextItem)
                and item.toPlainText() == "5 cm"
                for item in view._overlay_items
            )
        )
        view.set_overlay_mode("ruler_detection")
        self.assertGreaterEqual(len(view._overlay_items), 3)
        self.assertGreaterEqual(view._overlay_items[0].pen().widthF(), 11.0)
        self.assertFalse(
            any(
                isinstance(item, QGraphicsTextItem)
                and item.toPlainText() == "5 cm"
                for item in view._overlay_items
            )
        )
        view.set_overlay_mode("layout_detection")
        dish_edges = [
            item
            for item in view._overlay_items
            if isinstance(item, QGraphicsEllipseItem)
        ]
        self.assertEqual(len(dish_edges), 2)
        self.assertNotEqual(
            dish_edges[0].pen().color().name(),
            dish_edges[1].pen().color().name(),
        )
        layout_diameters = {round(item.rect().width()) for item in dish_edges}
        self.assertIn(result.dish.inner_radius * 2, layout_diameters)
        self.assertIn(result.dish.outer_radius * 2, layout_diameters)
        view.set_overlay_mode("proposals")
        proposal_diameters = {
            round(item.rect().width())
            for item in view._overlay_items
            if isinstance(item, QGraphicsEllipseItem)
        }
        self.assertIn(result.dish.outer_radius * 2, proposal_diameters)
        self.assertNotIn(result.dish.inner_radius * 2, proposal_diameters)
        view.set_overlay_mode("background_likelihood")
        sampling_bands = [
            item
            for item in view._overlay_items
            if isinstance(item, QGraphicsPathItem)
        ]
        self.assertEqual(len(sampling_bands), 1)
        self.assertIn("Initial background sampling band", sampling_bands[0].toolTip())
        view.set_overlay_mode("refined_background_likelihood")
        noise_rasters = [
            item
            for item in view._overlay_items
            if isinstance(item, QGraphicsPixmapItem)
        ]
        self.assertEqual(len(noise_rasters), 2)
        view.set_overlay_mode("calibrated_image")
        self.assertGreaterEqual(len(view._overlay_items), 5)
        self.assertEqual(
            sum(
                isinstance(item, QGraphicsTextItem)
                and item.toPlainText() == "5 cm"
                for item in view._overlay_items
            ),
            2,
        )
        for mode in (
            "seed_interior_probability",
            "boundary_confidence",
            "touching_split_likelihood",
            "ellipse_likelihood",
            "proposal_disagreement",
            "instance_assignment_confidence",
            "contact_graph",
            "illumination_field",
            "image_quality_risk",
            "radial_profile_residual",
            "wrinkling_likelihood",
            "coat_damage_likelihood",
            "pattern_classes",
            "colour_classes",
            "calibration_residual_risk",
            "colour_probability:0",
            "pattern_probability:0",
        ):
            view.set_overlay_mode(mode)
            self.assertGreaterEqual(len(view._overlay_items), 1, mode)
        view.clear_analysis()
        self.assertEqual(len(view._overlay_items), 0)
        view.close()

        window = MainWindow(ROOT)
        window._mark_analysis_complete(result)
        for node_id in (
            "colour_reference",
            "ruler_detection",
            "deskew_colour",
            "scale_calibration",
        ):
            self.assertEqual(window.pipeline.node(node_id).status, NodeStatus.COMPLETE)
        self.assertAlmostEqual(
            window.pipeline.node("seed_edge_curves").calculation_seconds,
            result.node_timings_seconds["seed_edge_curves"],
        )
        for node_id in (
            "seed_interior",
            "boundary_normals",
            "touching_split",
            "ellipse_likelihood",
            "proposal_disagreement",
            "assignment_confidence",
            "contact_graph",
            "illumination_decomposition",
            "image_quality",
            "radial_profile",
            "wrinkling",
            "coat_damage",
            "pattern_decomposition",
            "colour_probabilities",
            "calibration_residuals",
        ):
            self.assertEqual(window.pipeline.node(node_id).status, NodeStatus.COMPLETE)
        for node_id in (
            "foreground_segmentation",
            "distance_candidates",
            "circle_candidates",
            "identification",
        ):
            self.assertEqual(window.pipeline.node(node_id).status, NodeStatus.COMPLETE)
        self.assertIn(
            "Threshold",
            window.pipeline.node("foreground_segmentation").status_detail,
        )
        self.assertIn(
            "Hough circle",
            window.pipeline.node("circle_candidates").status_detail,
        )
        window.close()


if __name__ == "__main__":
    unittest.main()
