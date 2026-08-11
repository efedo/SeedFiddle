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
        self.assertEqual(len(canvas.node_items), 28)
        self.assertEqual(len(canvas.edge_items), 60)
        self.assertNotIn("circle_candidates", canvas.node_items)
        self.assertEqual(canvas.unused_nodes_button.text(), "Unused nodes (20)")
        unused_actions = [
            action.text() for action in canvas.unused_nodes_menu.actions()
        ]
        self.assertIn("Add Circle candidates to graph", unused_actions)
        self.assertIn("Add Distance-peak candidates to graph", unused_actions)
        self.assertIn(
            "Add Lightening derivative upper cutoff to graph", unused_actions
        )
        self.assertIn(
            "Add Darkening derivative upper cutoff to graph", unused_actions
        )
        self.assertIn(
            "Add Directional surface darkness gradients to graph", unused_actions
        )
        for calibration_node_id in (
            "colour_reference",
            "ruler_detection",
            "deskew_colour",
            "scale_calibration",
        ):
            self.assertIn(calibration_node_id, canvas.node_items)
        for overlay_node_id in (
            "background_likelihood",
            "refined_background_likelihood",
            "foreground_noise_likelihood",
            "edge_gradients",
            "frequency_noise_masks",
            "undirected_edges",
            "directed_edges",
            "edge_ridges",
            "edge_traces",
            "procedural_instances",
        ):
            self.assertIn(overlay_node_id, canvas.node_items)
        for stage_node_id in (
            "seed_scale_estimation",
            "perimeter_background_reference",
            "foreground_segmentation",
        ):
            self.assertIn(stage_node_id, canvas.node_items)
        for advanced_node_id in (
            "seed_interior",
            "boundary_normals",
            "ellipse_likelihood",
            "illumination_decomposition",
            "image_quality",
            "pattern_decomposition",
            "colour_probabilities",
            "calibration_residuals",
        ):
            self.assertIn(advanced_node_id, canvas.node_items)
        for dormant_node_id in (
            "distance_candidates",
            "surface_darkness_gradients",
            "identification",
            "instance_masks",
            "seed_edge_curves",
            "touching_split",
            "review",
            "output",
        ):
            self.assertNotIn(dormant_node_id, canvas.node_items)
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

    def test_manual_overlap_is_allowed_and_auto_arrange_removes_it(self) -> None:
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
        self.assertTrue(
            canvas._items_overlap(moved, float(moved.pos().y()), obstacle)
        )
        canvas.auto_arrange()
        items = tuple(canvas.node_items.values())
        for index, first in enumerate(items):
            for second in items[index + 1:]:
                self.assertFalse(
                    canvas._items_overlap(first, float(first.pos().y()), second),
                    (first.node.identifier, second.node.identifier),
                )
        canvas.close()

    def test_unused_node_toolbox_can_restore_circle_candidates(self) -> None:
        from seedvision.pipeline import build_default_pipeline
        from seedvision.ui.pipeline_canvas import PipelineCanvas

        graph = build_default_pipeline()
        canvas = PipelineCanvas(graph)
        restored: list[str] = []
        canvas.unused_node_restored.connect(restored.append)
        canvas.unused_nodes_menu.actions()[0].trigger()
        self.application.processEvents()
        self.assertEqual(restored, ["circle_candidates"])
        self.assertIn("circle_candidates", canvas.node_items)
        self.assertEqual(len(canvas.node_items), 29)
        self.assertEqual(len(canvas.edge_items), 67)
        self.assertEqual(canvas.unused_nodes_button.text(), "Unused nodes (19)")
        self.assertTrue(canvas.unused_nodes_button.isEnabled())
        self.assertTrue(canvas.node_items["circle_candidates"].isSelected())
        self.assertTrue(canvas.move_to_unused_button.isEnabled())
        shelved: list[str] = []
        canvas.unused_node_shelved.connect(shelved.append)
        canvas.move_to_unused_button.click()
        self.application.processEvents()
        self.assertEqual(shelved, ["circle_candidates"])
        self.assertNotIn("circle_candidates", canvas.node_items)
        self.assertIn("circle_candidates", graph.unused_nodes)
        self.assertFalse(graph.node("circle_candidates").enabled)
        self.assertEqual(len(canvas.edge_items), 60)
        self.assertEqual(canvas.unused_nodes_button.text(), "Unused nodes (20)")
        restore_action = next(
            action
            for action in canvas.unused_nodes_menu.actions()
            if action.text() == "Add Circle candidates to graph"
        )
        restore_action.trigger()
        self.application.processEvents()
        self.assertIn("circle_candidates", canvas.node_items)
        canvas.close()

    def test_main_window_exposes_circle_overlay_only_after_toolbox_restore(self) -> None:
        from seedvision.ui.main_window import MainWindow

        window = MainWindow(ROOT)
        self.assertEqual(window.overlay_combo.findData("circle_candidates"), -1)
        window.pipeline_canvas.unused_nodes_menu.actions()[0].trigger()
        self.application.processEvents()
        self.assertGreaterEqual(
            window.overlay_combo.findData("circle_candidates"), 0
        )
        self.assertIn("circle_candidates", window.pipeline_canvas.node_items)
        self.assertEqual(
            window.pipeline_inspector.title_label.text(),
            "Node: Circle candidates",
        )
        self.assertFalse(window.pipeline.node("circle_candidates").enabled)
        window.pipeline_canvas.move_to_unused_button.click()
        self.application.processEvents()
        self.assertEqual(window.overlay_combo.findData("circle_candidates"), -1)
        self.assertIn("circle_candidates", window.pipeline.unused_nodes)
        self.assertEqual(
            window.pipeline_inspector.title_label.text(),
            "Node: Grayscale & local lighting",
        )
        window.close()

    def test_selected_node_marks_direct_neighbours_purple(self) -> None:
        from seedvision.pipeline import build_default_pipeline
        from seedvision.ui.pipeline_canvas import PipelineCanvas

        graph = build_default_pipeline()
        canvas = PipelineCanvas(graph)
        canvas.select_node("edge_gradients")
        self.application.processEvents()
        adjacent = set(graph.upstream("edge_gradients")) | set(
            graph.downstream("edge_gradients")
        )
        self.assertTrue(canvas.node_items["edge_gradients"].isSelected())
        self.assertTrue(all(canvas.node_items[node_id]._adjacent for node_id in adjacent))
        self.assertFalse(canvas.node_items["layout_detection"]._adjacent)
        canvas.close()

    def test_image_and_graph_zoom_controls_report_scale(self) -> None:
        from PySide6.QtGui import QPalette

        from seedvision.pipeline import build_default_pipeline
        from seedvision.ui.image_view import ImageView
        from seedvision.ui.pipeline_canvas import PipelineCanvas

        view = ImageView()
        succeeded, error = view.load_image(ROOT / "images" / "IMG_9670c.JPG")
        self.assertTrue(succeeded, error)
        view.actual_size()
        view.zoom_in()
        self.assertEqual(view.zoom_label.text(), "120%")
        canvas = PipelineCanvas(build_default_pipeline())
        canvas.resetTransform()
        canvas.zoom_in()
        self.assertEqual(canvas.zoom_label.text(), "118%")
        self.assertEqual(canvas.auto_arrange_button.text(), "Auto arrange")
        self.assertEqual(canvas.unused_nodes_button.text(), "Unused nodes (20)")
        for button in (
            view.zoom_out_button,
            view.zoom_in_button,
            view.fit_button,
            view.actual_size_button,
            canvas.auto_arrange_button,
            canvas.unused_nodes_button,
            canvas.zoom_out_button,
            canvas.zoom_in_button,
            canvas.fit_button,
        ):
            self.assertEqual(
                button.palette()
                .color(QPalette.ColorGroup.Active, QPalette.ColorRole.ButtonText)
                .name(),
                "#f2f6fa",
            )
        self.assertEqual(
            canvas.move_to_unused_button.palette()
            .color(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText)
            .name(),
            "#aeb9c4",
        )
        self.assertIn("QToolButton:disabled", canvas.control_bar.styleSheet())
        view.close()
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
            "circle_candidates": 15,
            "identification": 1,
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
        self.assertEqual(
            inspector.title_label.text(), "Node: Foreground colour probability"
        )
        self.assertEqual(inspector.parameter_form.rowCount(), 15)
        self.assertTrue(
            all(widget.toolTip() for widget in inspector._parameter_widgets)
        )
        for node_id, minimum_count in {
            "layout_detection": 8,
            "background_likelihood": 11,
            "refined_background_likelihood": 9,
            "foreground_noise_likelihood": 9,
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
        self.assertIn("HSV hue/tint/shade", inspector.gamut_caption.text())
        projection = inspector.gamut_widget._image
        top = projection.pixelColor(projection.width() // 2, 0)
        bottom = projection.pixelColor(
            projection.width() // 2, projection.height() - 1
        )
        self.assertGreater(min(top.red(), top.green(), top.blue()), 245)
        self.assertLess(max(bottom.red(), bottom.green(), bottom.blue()), 10)
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
        self.assertEqual(
            window.pipeline_inspector.title_label.text(),
            "Node: Directed edge tangents",
        )
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
        restore_surface = next(
            action
            for action in window.pipeline_canvas.unused_nodes_menu.actions()
            if action.text()
            == "Add Directional surface darkness gradients to graph"
        )
        restore_surface.trigger()
        self.application.processEvents()
        window.pipeline_canvas.select_node("surface_darkness_gradients")
        self.application.processEvents()
        self.assertEqual(
            window.overlay_combo.currentData(), "surface_lightening_gradient"
        )
        self.assertEqual(len(window.pipeline_inspector._parameter_widgets), 7)
        restore_lightening = next(
            action
            for action in window.pipeline_canvas.unused_nodes_menu.actions()
            if action.text()
            == "Add Lightening derivative upper cutoff to graph"
        )
        restore_lightening.trigger()
        self.application.processEvents()
        window.pipeline_canvas.select_node("lightening_gradient_ceiling")
        self.application.processEvents()
        self.assertEqual(
            window.overlay_combo.currentData(), "weak_lightening_gradient"
        )
        self.assertEqual(len(window.pipeline_inspector._parameter_widgets), 1)
        window.pipeline_canvas.select_node("frequency_noise_masks")
        self.application.processEvents()
        self.assertEqual(window.overlay_combo.currentData(), "darkness_noise_fine")
        self.assertEqual(len(window.pipeline_inspector._parameter_widgets), 7)
        window.pipeline_canvas.select_node("perimeter_background_reference")
        self.application.processEvents()
        self.assertEqual(
            window.overlay_combo.currentData(), "perimeter_background_reference"
        )
        self.assertEqual(
            window.pipeline_inspector.title_label.text(),
            "Node: Perimeter background reference",
        )
        self.assertEqual(len(window.pipeline_inspector._parameter_widgets), 2)
        window.pipeline_canvas.select_node("refined_background_likelihood")
        self.application.processEvents()
        self.assertEqual(
            window.overlay_combo.currentData(), "refined_background_likelihood"
        )
        self.assertEqual(
            window.pipeline_inspector.title_label.text(),
            "Node: Background noise probability",
        )
        self.assertEqual(len(window.pipeline_inspector._parameter_widgets), 10)
        window.pipeline_canvas.select_node("foreground_noise_likelihood")
        self.application.processEvents()
        self.assertEqual(
            window.overlay_combo.currentData(), "foreground_noise_likelihood"
        )
        self.assertEqual(
            window.pipeline_inspector.title_label.text(),
            "Node: Foreground noise probability",
        )
        self.assertEqual(len(window.pipeline_inspector._parameter_widgets), 10)
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
        restore_action = next(
            action
            for action in window.pipeline_canvas.unused_nodes_menu.actions()
            if action.text() == "Add Seed identification to graph"
        )
        restore_action.trigger()
        self.application.processEvents()
        self.assertFalse(window.calibration_section.isVisible())
        self.assertTrue(window.baseline_section.isVisible())
        self.assertFalse(window.analyze_button.isVisible())
        self.assertFalse(window.warning_label.isVisible())
        window.close()

    def test_image_and_pipeline_share_a_split_workspace(self) -> None:
        from seedvision.ui.main_window import MainWindow

        window = MainWindow(ROOT)
        self.assertEqual(window.windowTitle(), "Seed Fiddle")
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

    def test_overlay_selectors_group_layers_and_stay_synchronised(self) -> None:
        from seedvision.ui.main_window import MainWindow

        window = MainWindow(ROOT)
        self.assertEqual(
            window.overlay_combo.findData("surface_lightening_gradient"), -1
        )
        restore_surface = next(
            action
            for action in window.pipeline_canvas.unused_nodes_menu.actions()
            if action.text()
            == "Add Directional surface darkness gradients to graph"
        )
        restore_surface.trigger()
        self.application.processEvents()
        lightening_index = window.overlay_combo.findData(
            "surface_lightening_gradient"
        )
        self.assertGreater(lightening_index, 0)
        self.assertTrue(window.overlay_combo.itemText(lightening_index).startswith("    "))
        heading_item = window.overlay_combo.model().item(lightening_index - 1)
        self.assertEqual(
            heading_item.text(), "Directional surface darkness gradients"
        )
        self.assertFalse(heading_item.isEnabled())
        self.assertTrue(heading_item.font().bold())

        window.pipeline_canvas.select_node("frequency_noise_masks")
        self.application.processEvents()
        node_selector = window.pipeline_inspector.overlay_combo
        self.assertEqual(node_selector.count(), 6)
        self.assertEqual(node_selector.currentData(), "darkness_noise_fine")
        coarse_colour = node_selector.findData("colour_noise_coarse")
        self.assertGreaterEqual(coarse_colour, 0)
        node_selector.setCurrentIndex(coarse_colour)
        self.application.processEvents()
        self.assertEqual(window.overlay_combo.currentData(), "colour_noise_coarse")
        self.assertEqual(window.image_view._overlay_mode, "colour_noise_coarse")

        window.pipeline_canvas.select_node("procedural_instances")
        self.application.processEvents()
        self.assertEqual(node_selector.count(), 6)
        self.assertEqual(node_selector.currentData(), "procedural_instances")
        self.assertGreaterEqual(node_selector.findData("procedural_confidence"), 0)

        window.pipeline_canvas.select_node("metadata")
        self.application.processEvents()
        self.assertEqual(node_selector.itemText(0), "No image overlays")
        self.assertFalse(node_selector.isEnabled())
        window.close()

    def test_compact_toolbar_and_contextual_paint_panel(self) -> None:
        from PySide6.QtWidgets import QLabel

        from seedvision.ui.main_window import MainWindow

        window = MainWindow(ROOT)
        window.show()
        self.application.processEvents()

        toolbar_labels = {
            label.text()
            for label in window.workflow_toolbar.findChildren(QLabel)
        }
        self.assertIn("Overlay:", toolbar_labels)
        self.assertIn("Opacity:", toolbar_labels)
        self.assertEqual(window.paint_background_action.text(), "Paint background")
        self.assertEqual(
            window.paint_foreground_action.text(), "Paint foreground references"
        )
        self.assertEqual(
            window.annotate_instances_action.text(), "Annotate seed instances"
        )

        detail_labels = {
            label.text()
            for label in window.metadata_scroll.widget().findChildren(QLabel)
        }
        self.assertNotIn("Selected pipeline node", detail_labels)
        self.assertNotIn("Viewer overlay", detail_labels)
        self.assertNotIn("Painted background reference", detail_labels)
        self.assertTrue(window.overlay_owner_label.isHidden())
        self.assertIs(window.reference_panel.parentWidget(), window.image_view)
        self.assertTrue(window.reference_panel.isHidden())

        window.paint_background_action.setEnabled(True)
        window.paint_foreground_action.setEnabled(True)
        window.annotate_instances_action.setEnabled(True)
        window.background_point_button.setEnabled(True)
        window.foreground_point_button.setEnabled(True)
        window.background_exclusion_button.setEnabled(True)
        window.foreground_exclusion_button.setEnabled(True)
        window.erase_background_points_button.setEnabled(True)
        window.erase_foreground_points_button.setEnabled(True)
        window.erase_background_exclusion_button.setEnabled(True)
        window.erase_foreground_exclusion_button.setEnabled(True)
        window.paint_background_action.setChecked(True)
        self.application.processEvents()
        self.assertTrue(window.background_point_button.isChecked())
        self.assertFalse(window.reference_panel.isHidden())
        self.assertEqual(window.image_view._reference_point_mode, "background")

        window.paint_foreground_action.setChecked(True)
        self.application.processEvents()
        self.assertFalse(window.paint_background_action.isChecked())
        self.assertFalse(window.background_point_button.isChecked())
        self.assertTrue(window.foreground_point_button.isChecked())
        self.assertEqual(window.image_view._reference_point_mode, "foreground")

        window.foreground_exclusion_button.setChecked(True)
        self.application.processEvents()
        self.assertFalse(window.foreground_point_button.isChecked())
        self.assertEqual(
            window.image_view._reference_point_mode, "foreground_exclusion"
        )
        window.show_reference_areas_checkbox.setChecked(False)
        self.assertFalse(window.image_view._reference_annotations_visible)

        for erase_button, expected_mode in (
            (window.erase_background_points_button, "background"),
            (window.erase_foreground_points_button, "foreground"),
            (window.erase_background_exclusion_button, "background_exclusion"),
            (window.erase_foreground_exclusion_button, "foreground_exclusion"),
        ):
            erase_button.click()
            self.application.processEvents()
            self.assertTrue(window.reference_eraser_button.isChecked())
            self.assertEqual(window.image_view._reference_point_mode, expected_mode)

        window.annotate_instances_action.setChecked(True)
        self.application.processEvents()
        self.assertFalse(window.paint_background_action.isChecked())
        self.assertFalse(window.paint_foreground_action.isChecked())
        self.assertEqual(window.image_view._reference_point_mode, "instance")
        self.assertTrue(window.reference_controls.isHidden())
        self.assertFalse(window.instance_annotation_controls.isHidden())
        window.instance_eraser_button.setChecked(True)
        self.assertEqual(window.image_view._instance_annotation_tool, "eraser")

        window.annotate_instances_action.setChecked(False)
        self.application.processEvents()
        self.assertTrue(window.reference_panel.isHidden())
        self.assertIsNone(window.image_view._reference_point_mode)
        window.close()

    def test_background_node_can_be_disabled_without_disabling_foreground_noise(self) -> None:
        from seedvision.pipeline import NodeStatus
        from seedvision.ui.main_window import MainWindow

        window = MainWindow(ROOT)
        window._pipeline_enabled_changed("background_likelihood", False)
        self.assertFalse(window.pipeline.node("background_likelihood").enabled)
        self.assertTrue(window.pipeline.node("foreground_noise_likelihood").enabled)
        self.assertEqual(
            window.pipeline.node("background_likelihood").status,
            NodeStatus.BYPASSED,
        )
        self.assertEqual(
            window.pipeline.node("refined_background_likelihood").status,
            NodeStatus.BYPASSED,
        )
        self.assertFalse(window.background_enabled_checkbox.isChecked())
        window.close()

    def test_settings_reset_button_restores_node_defaults(self) -> None:
        from seedvision.ui.main_window import MainWindow

        window = MainWindow(ROOT)
        node = window.pipeline.node("foreground_segmentation")
        default = node.default_parameters["foreground_reference_weight"]
        window._pipeline_parameter_changed(
            "foreground_segmentation", "foreground_reference_weight", 0.25
        )
        window.pipeline_canvas.select_node("foreground_segmentation")
        self.application.processEvents()
        window.pipeline_inspector.reset_parameters_button.click()
        self.application.processEvents()
        self.assertEqual(node.parameters["foreground_reference_weight"], default)
        window.close()

    def test_expanded_node_settings_feed_runtime_settings_objects(self) -> None:
        from seedvision.ui.main_window import MainWindow

        window = MainWindow(ROOT)
        window.pipeline.set_parameter(
            "layout_detection", "hough_accumulator_threshold", 41
        )
        window.pipeline.set_parameter(
            "foreground_noise_likelihood",
            "foreground_noise_vector_length_fraction",
            0.75,
        )
        window.pipeline.set_parameter("edge_gradients", "edge_chroma_weight", 1.8)
        window.pipeline.set_parameter(
            "background_likelihood", "background_colour_components", 3
        )
        window.pipeline.set_parameter(
            "perimeter_background_reference",
            "perimeter_background_buffer_cm",
            0.45,
        )
        window.pipeline.set_parameter(
            "perimeter_background_reference",
            "perimeter_background_band_thickness_cm",
            0.70,
        )
        window.pipeline.set_parameter(
            "foreground_segmentation", "foreground_refinement_iterations", 3
        )

        self.assertEqual(window._dish_settings().hough_accumulator_threshold, 41)
        self.assertAlmostEqual(
            window._layer_settings().foreground_noise_vector_length_fraction,
            0.75,
        )
        self.assertAlmostEqual(
            window._layer_settings().edge_chroma_weight,
            1.8,
        )
        self.assertEqual(window._layer_settings().background_colour_components, 3)
        self.assertAlmostEqual(
            window._layer_settings().perimeter_background_buffer_cm,
            0.45,
        )
        self.assertAlmostEqual(
            window._layer_settings().perimeter_background_band_thickness_cm,
            0.70,
        )
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
            "seed_interior", "interior_foreground_noise_weight", 0.60
        )
        window.pipeline.set_parameter(
            "image_quality", "quality_noise_scale_fraction", 0.05
        )
        self.assertEqual(window._advanced_settings().maximum_dimension, 768)
        self.assertAlmostEqual(
            window._advanced_settings().interior_foreground_noise_weight,
            0.60,
        )
        self.assertAlmostEqual(
            window._advanced_settings().quality_noise_scale_fraction, 0.05
        )
        with self.assertRaises(ValueError):
            window._validate_settings_override(
                "edge_ridges", "ridge_low_threshold", 0.30
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
            "edge_ridges",
            "edge_traces",
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
            window.pipeline.node("edge_ridges").status, NodeStatus.IDLE
        )
        self.assertEqual(
            window.pipeline.node("edge_traces").status, NodeStatus.IDLE
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
            foreground_noise_likelihood=np.full((12, 12), 145, np.uint8),
            foreground_noise_frequency_profile=NoiseFrequencyProfile(
                band_scales_px=(0.8, 1.8, 4.0),
                background_log_rms=(0.3, 0.2, 0.1),
                nonbackground_log_rms=(0.1, 0.2, 0.3),
                background_sample_count=100,
                nonbackground_sample_count=100,
                separation=1.2,
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
            "foreground_noise_likelihood",
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
        reference = np.zeros((12, 12), dtype=bool)
        reference[4:8, 4:8] = True
        view.set_reference_masks(reference, reference, reference, reference)
        self.assertEqual(len(view._overlay_items), 4)
        view.set_reference_annotations_visible(False)
        self.assertEqual(len(view._overlay_items), 0)
        view.set_reference_annotations_visible(True)
        self.assertEqual(len(view._overlay_items), 4)
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
            # QTest does not emit a move when the global cursor already happens
            # to be at the requested point after an earlier GUI test.
            QTest.mouseMove(view.viewport(), view.viewport().rect().topLeft())
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
            view.set_background_exclusion_editing(True)
            QTest.mouseClick(
                view.viewport(), Qt.MouseButton.LeftButton, pos=position
            )
            self.assertEqual(edits[-1][0], "background_exclusion")
            self.assertTrue(edits[-1][1][50, 50])
            view.set_foreground_exclusion_editing(True)
            QTest.mouseClick(
                view.viewport(), Qt.MouseButton.LeftButton, pos=position
            )
            self.assertEqual(edits[-1][0], "foreground_exclusion")
            self.assertTrue(edits[-1][1][50, 50])
            view.close()

    def test_image_view_edits_distinct_seed_instance_ids(self) -> None:
        import numpy as np

        from PySide6.QtCore import QPointF, Qt
        from PySide6.QtGui import QColor, QImage
        from PySide6.QtTest import QTest

        from seedvision.ui.image_view import ImageView

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "seed-instances.png"
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
            edits: list[np.ndarray] = []
            view.instance_annotations_edited.connect(
                lambda labels: edits.append(labels)
            )
            view.set_instance_annotation_editing(True)
            first = view.mapFromScene(QPointF(30.0, 50.0))
            second = view.mapFromScene(QPointF(70.0, 50.0))
            view.set_reference_brush_radius(8.0)
            view.set_active_instance_id(1)
            QTest.mouseClick(view.viewport(), Qt.MouseButton.LeftButton, pos=first)
            self.assertEqual(int(edits[-1][50, 30]), 1)
            view.set_active_instance_id(2)
            QTest.mouseClick(view.viewport(), Qt.MouseButton.LeftButton, pos=second)
            self.assertEqual(int(edits[-1][50, 30]), 1)
            self.assertEqual(int(edits[-1][50, 70]), 2)
            self.assertEqual(set(np.unique(edits[-1])), {0, 1, 2})
            self.assertNotEqual(
                view.instance_colour(1).name(), view.instance_colour(2).name()
            )
            view._render_instance_annotations()
            self.assertGreater(len(view._overlay_items), 0)
            view.set_reference_erase_mode(True)
            QTest.mouseClick(view.viewport(), Qt.MouseButton.LeftButton, pos=second)
            self.assertEqual(int(edits[-1][50, 70]), 0)
            self.assertEqual(int(edits[-1][50, 30]), 1)
            view.close()

    def test_instance_brush_uses_live_stroke_without_rebuilding_analysis(self) -> None:
        import numpy as np

        from PySide6.QtCore import QPointF, Qt
        from PySide6.QtGui import QColor, QImage
        from PySide6.QtTest import QTest

        from seedvision.ui.image_view import ImageView

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "responsive-seed-brush.png"
            image = QImage(160, 120, QImage.Format.Format_RGB32)
            image.fill(QColor("white"))
            self.assertTrue(image.save(str(path)))

            view = ImageView()
            view.resize(480, 360)
            succeeded, error = view.load_image(path)
            self.assertTrue(succeeded, error)
            view.show()
            self.application.processEvents()
            view.fit_image()
            view.set_reference_brush_radius(7.0)
            view.set_instance_annotation_editing(True)
            renders: list[bool] = []
            view._render_analysis = lambda: renders.append(True)
            start = view.mapFromScene(QPointF(35.0, 55.0))
            end = view.mapFromScene(QPointF(105.0, 55.0))

            QTest.mousePress(view.viewport(), Qt.MouseButton.LeftButton, pos=start)
            self.assertIsNotNone(view._instance_live_stroke_item)
            QTest.mouseMove(view.viewport(), end, delay=1)
            self.assertIsNotNone(view._instance_live_stroke_item)
            self.assertEqual(renders, [])
            QTest.mouseRelease(view.viewport(), Qt.MouseButton.LeftButton, pos=end)

            labels = view.instance_annotations()
            self.assertIsNotNone(labels)
            self.assertGreater(np.count_nonzero(labels == 1), 500)
            self.assertIsNone(view._instance_live_stroke_item)
            self.assertEqual(renders, [])
            view.close()

    def test_image_view_applies_assisted_instance_annotation_tools(self) -> None:
        import numpy as np

        from PySide6.QtCore import QPointF
        from PySide6.QtGui import QColor, QImage

        from seedvision.annotation import (
            EdgeTraceOptions,
            ShapeSnapOptions,
            SmartFillOptions,
        )
        from seedvision.ui.image_view import ImageView

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "assisted-instances.png"
            image = QImage(120, 100, QImage.Format.Format_RGB32)
            image.fill(QColor("#787878"))
            self.assertTrue(image.save(str(path)))
            edge = np.zeros((100, 120), dtype=np.uint8)
            edge[30, 10:61] = 255
            import cv2

            cv2.ellipse(edge, (88, 52), (18, 12), 0, 0, 360, 255, 2)
            result = SimpleNamespace(
                calibration=SimpleNamespace(
                    corrected_bgr=np.full((100, 120, 3), 120, dtype=np.uint8)
                ),
                layers=SimpleNamespace(
                    edge_likelihood=edge,
                    undirected_edge_hue=np.zeros_like(edge),
                    directed_edge_hue=np.zeros_like(edge),
                    offset_x=0,
                    offset_y=0,
                ),
                crop_offset=(0, 0),
                estimated_seed_diameter_px=30.0,
            )
            view = ImageView()
            succeeded, error = view.load_image(path)
            self.assertTrue(succeeded, error)
            view.set_overlay_mode("none")
            view.show_analysis(result)
            view.set_instance_annotation_editing(True)
            statuses: list[str] = []
            view.instance_tool_status.connect(statuses.append)

            view.set_active_instance_id(1)
            view.set_instance_annotation_tool("edge_trace")
            view.set_edge_trace_options(
                EdgeTraceOptions(tangent_mode="off", smoothing=0)
            )
            view._instance_tool_points = [QPointF(10, 30)]
            self.assertTrue(view._apply_instance_assisted_tool(QPointF(60, 30)))
            self.assertGreater(np.count_nonzero(view._instance_annotations == 1), 40)

            view.set_active_instance_id(2)
            view.set_instance_annotation_tool("shape_snap")
            view.set_shape_snap_options(
                ShapeSnapOptions(
                    shape="ellipse",
                    edge_search_radius_px=5,
                    centre_search_radius_px=2,
                    tangent_mode="off",
                )
            )
            view._instance_tool_points = [QPointF(88, 52)]
            self.assertTrue(view._apply_instance_assisted_tool(QPointF(106, 64)))
            self.assertEqual(int(view._instance_annotations[52, 88]), 2)

            view.set_active_instance_id(3)
            labels = view.instance_annotations()
            labels[76:80, 24:28] = 3
            view.set_instance_annotations(labels)
            view.set_instance_annotation_tool("smart_fill")
            view.set_smart_fill_options(
                SmartFillOptions(
                    maximum_radius_px=14,
                    maximum_added_pixels=2_000,
                )
            )
            self.assertTrue(view._apply_instance_assisted_tool(QPointF(26, 78)))
            self.assertGreater(np.count_nonzero(view._instance_annotations == 3), 200)
            self.assertTrue(any("Edge trace" in status for status in statuses))
            self.assertTrue(any("Shape snap" in status for status in statuses))
            self.assertTrue(any("Smart fill" in status for status in statuses))
            view.close()

    def test_assisted_instance_tools_preview_on_hover_and_apply_on_click(self) -> None:
        import cv2
        import numpy as np

        from PySide6.QtCore import QPointF, Qt
        from PySide6.QtGui import QColor, QImage
        from PySide6.QtTest import QTest

        from seedvision.annotation import (
            EdgeTraceOptions,
            ShapeSnapOptions,
            SmartFillOptions,
        )
        from seedvision.ui.image_view import ImageView

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "preview-instances.png"
            image = QImage(160, 120, QImage.Format.Format_RGB32)
            image.fill(QColor("#787878"))
            self.assertTrue(image.save(str(path)))
            edge = np.zeros((120, 160), dtype=np.uint8)
            edge[28, 12:72] = 255
            cv2.ellipse(edge, (112, 58), (20, 14), 0, 0, 360, 255, 2)
            result = SimpleNamespace(
                calibration=SimpleNamespace(
                    corrected_bgr=np.full((120, 160, 3), 120, dtype=np.uint8)
                ),
                layers=SimpleNamespace(
                    edge_likelihood=edge,
                    undirected_edge_hue=np.zeros_like(edge),
                    directed_edge_hue=np.zeros_like(edge),
                    offset_x=0,
                    offset_y=0,
                ),
                crop_offset=(0, 0),
                estimated_seed_diameter_px=38.0,
            )
            view = ImageView()
            view.resize(480, 360)
            succeeded, error = view.load_image(path)
            self.assertTrue(succeeded, error)
            view.set_overlay_mode("none")
            view.show_analysis(result)
            view.set_instance_annotation_editing(True)
            view.show()
            self.application.processEvents()
            view.fit_image()
            edits: list[np.ndarray] = []
            view.instance_annotations_edited.connect(edits.append)

            view.set_active_instance_id(1)
            view.set_instance_annotation_tool("edge_trace")
            view.set_edge_trace_options(
                EdgeTraceOptions(tangent_mode="off", smoothing=0)
            )
            first = view.mapFromScene(QPointF(14, 28))
            second = view.mapFromScene(QPointF(68, 28))
            # Force distinct global cursor positions so QTest emits moves even
            # when a previous GUI test ended at one of these coordinates.
            QTest.mouseMove(view.viewport(), view.viewport().rect().topLeft())
            QTest.mouseMove(view.viewport(), first)
            QTest.qWait(55)
            self.assertIsNotNone(view._instance_preview_endpoint)
            QTest.mouseClick(view.viewport(), Qt.MouseButton.LeftButton, pos=first)
            self.assertIsNotNone(view._instance_trace_anchor)
            QTest.mouseMove(view.viewport(), view.viewport().rect().bottomRight())
            QTest.mouseMove(view.viewport(), second)
            QTest.qWait(55)
            self.assertIsNotNone(view._instance_preview_geometry)
            self.assertGreater(len(view._instance_preview_geometry), 20)
            QTest.mouseClick(view.viewport(), Qt.MouseButton.LeftButton, pos=second)
            self.assertTrue(edits)
            self.assertGreater(np.count_nonzero(edits[-1] == 1), 40)

            view.set_active_instance_id(2)
            view.set_instance_annotation_tool("shape_snap")
            view.set_shape_snap_options(
                ShapeSnapOptions(
                    shape="ellipse",
                    edge_search_radius_px=5,
                    centre_search_radius_px=2,
                    tangent_mode="off",
                )
            )
            shape_centre = view.mapFromScene(QPointF(112, 58))
            QTest.mouseMove(view.viewport(), shape_centre)
            QTest.qWait(55)
            self.assertIsNotNone(view._instance_preview_geometry)
            self.assertIsNotNone(view._instance_shape_reference_geometry)
            self.assertGreater(len(view._instance_preview_items), 0)
            QTest.mouseClick(
                view.viewport(), Qt.MouseButton.LeftButton, pos=shape_centre
            )
            self.assertEqual(int(edits[-1][58, 112]), 2)

            view.set_active_instance_id(3)
            view.set_instance_annotation_tool("smart_fill")
            view.set_smart_fill_options(
                SmartFillOptions(
                    maximum_radius_px=14,
                    maximum_added_pixels=2_000,
                )
            )
            fill_centre = view.mapFromScene(QPointF(35, 88))
            QTest.mouseMove(view.viewport(), fill_centre)
            QTest.qWait(55)
            self.assertIsNotNone(view._instance_preview_region)
            self.assertGreater(view._instance_preview_region.added_count, 100)
            QTest.mouseClick(
                view.viewport(), Qt.MouseButton.LeftButton, pos=fill_centre
            )
            self.assertEqual(int(edits[-1][88, 35]), 3)
            view.close()

    def test_instance_annotation_panel_exposes_advanced_tool_options(self) -> None:
        from seedvision.ui.main_window import MainWindow

        window = MainWindow(ROOT)
        self.assertEqual(window.instance_tool_options_stack.count(), 5)
        window.instance_edge_trace_button.setChecked(True)
        self.application.processEvents()
        self.assertEqual(window.image_view._instance_annotation_tool, "edge_trace")
        self.assertIs(
            window.instance_tool_options_stack.currentWidget(),
            window.instance_tool_pages["edge_trace"],
        )
        window.edge_trace_tangent_combo.setCurrentIndex(2)
        window.edge_trace_attraction_spin.setValue(72)
        self.assertEqual(window.image_view._edge_trace_options.tangent_mode, "directed")
        self.assertAlmostEqual(window.image_view._edge_trace_options.edge_attraction, 0.72)

        window.instance_shape_snap_button.setChecked(True)
        window.shape_snap_shape_combo.setCurrentIndex(1)
        window.shape_snap_rotation_spin.setValue(22.5)
        self.assertEqual(window.image_view._shape_snap_options.shape, "circle")
        self.assertAlmostEqual(window.image_view._shape_snap_options.rotation_degrees, 22.5)

        window.instance_smart_fill_button.setChecked(True)
        window.smart_fill_tunnel_combo.setCurrentIndex(3)
        window.smart_fill_connectivity_combo.setCurrentIndex(1)
        self.assertAlmostEqual(window.image_view._smart_fill_options.tunnel_strength, 0.75)
        self.assertEqual(window.image_view._smart_fill_options.connectivity, 4)
        window.close()

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
        background_exclusion = np.zeros((height, width), dtype=bool)
        foreground_exclusion = np.zeros((height, width), dtype=bool)
        background_exclusion[5:12, 7:14] = True
        foreground_exclusion[40:47, 50:57] = True
        window._reference_mask_edited(
            "background_exclusion", background_exclusion
        )
        window._reference_mask_edited(
            "foreground_exclusion", foreground_exclusion
        )
        window._apply_reference_masks()
        self.assertEqual(analyses, [True, True, True])
        self.assertTrue(
            np.array_equal(
                window._applied_background_exclusion_masks[key],
                background_exclusion,
            )
        )
        self.assertTrue(
            np.array_equal(
                window._applied_foreground_exclusion_masks[key],
                foreground_exclusion,
            )
        )
        window.close()

    def test_instance_annotations_apply_separately_from_reference_masks(self) -> None:
        import numpy as np

        from seedvision.ui.main_window import MainWindow

        window = MainWindow(ROOT)
        key = window._current_image_key()
        if key is None or window.image_view.image_size is None:
            window.close()
            self.skipTest("No workspace image is present")
        width, height = window.image_view.image_size
        labels = np.zeros((height, width), dtype=np.uint16)
        labels[20:28, 30:38] = 1
        labels[60:68, 70:78] = 2
        analyses: list[bool] = []
        window._analyze_current_image = lambda **_kwargs: analyses.append(True)

        window._instance_annotations_edited(labels)
        self.assertIn(key, window._instance_annotations_dirty)
        self.assertNotIn(key, window._applied_instance_annotations)
        self.assertNotIn(key, window._draft_foreground_reference_masks)

        window._apply_instance_annotations()
        self.assertEqual(analyses, [True])
        self.assertNotIn(key, window._instance_annotations_dirty)
        self.assertTrue(
            np.array_equal(window._applied_instance_annotations[key], labels)
        )
        self.assertIn(
            "instance_masks", window._cache_dirty_nodes.get(key, set())
        )
        self.assertIn(
            "procedural_instances", window._cache_dirty_nodes.get(key, set())
        )
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
        view.set_overlay_mode("perimeter_background_reference")
        perimeter_bands = [
            item
            for item in view._overlay_items
            if isinstance(item, QGraphicsPathItem)
        ]
        self.assertEqual(len(perimeter_bands), 1)
        self.assertIn("0.35 cm buffer", perimeter_bands[0].toolTip())
        self.assertIn("0.50 cm thickness", perimeter_bands[0].toolTip())
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
            window.pipeline.node("edge_gradients").calculation_seconds,
            result.node_timings_seconds["edge_gradients"],
        )
        for node_id in (
            "illumination_decomposition",
            "image_quality",
            "calibration_residuals",
        ):
            self.assertEqual(window.pipeline.node(node_id).status, NodeStatus.COMPLETE)
        for node_id in (
            "seed_interior",
            "boundary_normals",
            "ellipse_likelihood",
            "pattern_decomposition",
            "colour_probabilities",
            "surface_darkness_gradients",
            "lightening_gradient_ceiling",
            "darkening_gradient_ceiling",
        ):
            self.assertEqual(window.pipeline.node(node_id).status, NodeStatus.BYPASSED)
        for node_id in (
            "perimeter_background_reference",
            "foreground_segmentation",
            "foreground_noise_likelihood",
        ):
            self.assertEqual(window.pipeline.node(node_id).status, NodeStatus.COMPLETE)
        self.assertIn(
            "0.35 cm buffer",
            window.pipeline.node("perimeter_background_reference").status_detail,
        )
        self.assertIn(
            "Threshold",
            window.pipeline.node("foreground_segmentation").status_detail,
        )
        self.assertIn("circle_candidates", window.pipeline.unused_nodes)
        self.assertEqual(
            window.pipeline.node("circle_candidates").status_detail,
            "Disabled by default while this branch is under review",
        )
        window.close()


if __name__ == "__main__":
    unittest.main()
