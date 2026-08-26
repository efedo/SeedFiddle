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
        self.assertEqual(len(canvas.edge_items), 154)
        self.assertNotIn("circle_candidates", canvas.node_items)
        self.assertEqual(canvas.unused_nodes_button.text(), "Unused nodes (19)")
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
            "Add Calibration residual risk to graph", unused_actions
        )
        for calibration_node_id in (
            "ruler_detection",
            "deskew_colour",
        ):
            self.assertIn(calibration_node_id, canvas.node_items)
        for overlay_node_id in (
            "background_likelihood",
            "refined_background_likelihood",
            "edge_gradients",
            "surface_darkness_gradients",
            "frequency_noise_masks",
            "edge_traces",
            "seed_edge_curves",
            "procedural_instances",
        ):
            self.assertIn(overlay_node_id, canvas.node_items)
        for stage_node_id in (
            "seed_scale_estimation",
            "perimeter_background_reference",
            "background_likelihood",
        ):
            self.assertIn(stage_node_id, canvas.node_items)
        for advanced_node_id in (
            "boundary_normals",
            "illumination_decomposition",
            "image_quality",
            "wrinkling",
            "pattern_decomposition",
            "colour_probabilities",
        ):
            self.assertIn(advanced_node_id, canvas.node_items)
        for dormant_node_id in (
            "distance_candidates",
            "calibration_residuals",
            "identification",
            "instance_masks",
            "ellipse_likelihood",
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

    def test_current_image_name_is_in_pane_and_list_selection_tracks_it(self) -> None:
        from seedvision.ui.main_window import MainWindow

        window = MainWindow(ROOT)
        path = window.image_view.image_path
        self.assertIsNotNone(path)
        self.assertEqual(window.image_view.current_file_label.text(), path.name)
        current = window.image_list.currentItem()
        self.assertIsNotNone(current)
        self.assertEqual(Path(current.data(256)).resolve(), path.resolve())
        self.assertIn("Add one or more images", window.open_action.toolTip())
        self.assertNotIn("Add images below", window.open_action.toolTip())
        self.assertFalse(hasattr(window, "add_images_button"))
        window.close()

    def test_run_to_node_scopes_execution_to_enabled_ancestors(self) -> None:
        from unittest.mock import patch

        from seedvision.ui.main_window import MainWindow

        window = MainWindow(ROOT)
        with patch.object(window, "_analyze_current_image") as analyze:
            window._run_pipeline_to_node("procedural_instances")
        arguments = analyze.call_args.kwargs
        self.assertEqual(arguments["dirty_nodes"], {"procedural_instances"})
        scope = arguments["enabled_node_scope"]
        self.assertIn("procedural_instances", scope)
        self.assertIn("layout_detection", scope)
        self.assertIn("reference_edge_probability", scope)
        self.assertNotIn("review", scope)
        self.assertNotIn("measurements", scope)
        window.close()

    def test_analysis_cache_is_lru_bounded_and_gpu_jobs_are_serialized(self) -> None:
        import numpy as np
        import torch

        from seedvision.cuda import GpuRaster
        from seedvision.segmentation import PipelineAnalysisCache
        from seedvision.ui.main_window import MainWindow

        window = MainWindow(ROOT)
        self.assertEqual(window._thread_pool.maxThreadCount(), 1)
        window._analysis_caches.clear()
        window._analyses.clear()
        old_raster = GpuRaster(
            torch.ones((1, 1, 8, 8), dtype=torch.uint8),
            numpy_dtype=np.uint8,
            name="old cache",
        )
        old_raster.numpy()
        old_cache = PipelineAnalysisCache(values={"raster": old_raster})
        new_cache = PipelineAnalysisCache(
            values={
                "raster": GpuRaster(
                    torch.ones((1, 1, 8, 8), dtype=torch.uint8),
                    numpy_dtype=np.uint8,
                    name="new cache",
                )
            }
        )
        window._analysis_caches["old"] = old_cache
        window._analysis_caches["new"] = new_cache
        window.ANALYSIS_CACHE_MAX_IMAGES = 1
        evicted = window._trim_analysis_caches(protected={"new"})

        self.assertEqual(evicted, ("old",))
        self.assertNotIn("old", window._analysis_caches)
        self.assertIn("new", window._analysis_caches)
        self.assertFalse(old_raster.is_materialized)
        self.assertEqual(old_cache.values, {})
        window.close()

    def test_failed_analysis_purges_its_partial_cache(self) -> None:
        from unittest.mock import patch

        import numpy as np
        import torch

        from seedvision.cuda import GpuRaster
        from seedvision.segmentation import PipelineAnalysisCache
        from seedvision.ui.main_window import MainWindow

        window = MainWindow(ROOT)
        key = window._current_image_key()
        path = window.image_view.image_path
        if key is None or path is None:
            window.close()
            self.skipTest("No workspace image is present")
        raster = GpuRaster(
            torch.ones((1, 1, 16, 16), dtype=torch.uint8),
            numpy_dtype=np.uint8,
            name="partial failure",
        )
        raster.numpy()
        cache = PipelineAnalysisCache(values={"partial": raster})
        window._analysis_caches[key] = cache
        window._active_tasks[key] = object()
        window._pending_analysis_key = key
        events = []
        with (
            patch.object(
                window,
                "_analyze_current_image",
                side_effect=lambda **_kwargs: events.append("resume"),
            ),
            patch(
                "seedvision.ui.main_window.QMessageBox.warning",
                side_effect=lambda *args: events.append("warning"),
            ),
        ):
            window._analysis_failed(str(path), "synthetic failure", window.pipeline.revision)

        self.assertNotIn(key, window._analysis_caches)
        self.assertFalse(raster.is_materialized)
        self.assertEqual(cache.values, {})
        self.assertEqual(events, ["resume", "warning"])
        window.close()

    def test_superseded_analysis_is_cancelled_and_remains_visible(self) -> None:
        from time import monotonic

        from seedvision.pipeline import NodeStatus
        from seedvision.ui.main_window import MainWindow, _AnalysisActivity

        class FakeTask:
            def __init__(self) -> None:
                self.cancelled = False

            def cancel(self) -> None:
                self.cancelled = True

        window = MainWindow(ROOT)
        key = window._current_image_key()
        path = window.image_view.image_path
        if key is None or path is None:
            window.close()
            self.skipTest("No workspace image is present")
        task = FakeTask()
        started = monotonic() - 5.0
        old_revision = window.pipeline.revision
        window._active_tasks[key] = task
        window._analysis_activities[key] = _AnalysisActivity(
            path=path,
            pipeline_revision=old_revision,
            started_at=started,
            last_progress_at=started,
            current_node="reference_texture_prototypes",
            current_node_started_at=started,
        )
        # Simulate a setting mutation that has already advanced the graph.
        window.pipeline.revision += 1
        window._analyze_current_image(
            dirty_nodes={"reference_texture_prototypes"}
        )

        self.assertTrue(task.cancelled)
        self.assertEqual(window._pending_analysis_key, key)
        self.assertIsNotNone(
            window._analysis_activities[key].cancellation_requested_at
        )
        self.assertEqual(
            window.pipeline.node("reference_texture_prototypes").status,
            NodeStatus.RUNNING,
        )
        self.assertIn("Cancellation requested", window.analysis_activity_label.text())
        self.assertIn("updated rerun is queued", window.analysis_activity_label.text())

        window._active_tasks.clear()
        window._analysis_activities.clear()
        window._pending_analysis_key = None
        window.close()

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
            canvas.node_items["deskew_colour"]._inline_editors[
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
        noise_item = canvas.node_items["refined_background_likelihood"]
        foreground_integration = noise_item._inline_editors[
            "foreground_noise_direction_integration"
        ]
        background_integration = noise_item._inline_editors[
            "noise_direction_integration"
        ]
        self.assertIsInstance(foreground_integration, QComboBox)
        self.assertEqual(foreground_integration.currentText(), "1st tertile")
        self.assertGreaterEqual(foreground_integration.findText("1st tertile"), 0)
        self.assertEqual(background_integration.findText("1st tertile"), -1)
        edge_item = canvas.node_items["edge_gradients"]
        controls_bottom = (
            edge_item._controls_y
            + (len(edge_item.node.inline_parameters) - 1) * edge_item.ROW_HEIGHT
            + 22.0
        )
        self.assertGreaterEqual(edge_item._footer_y, controls_bottom + 4.0)
        self.assertEqual(edge_item.height, edge_item._footer_y + 24.0)
        self.assertLess(canvas.node_items["deskew_colour"].height, 300.0)
        self.assertGreaterEqual(canvas.node_items["edge_gradients"].height, 108.0)
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

        moved = canvas.node_items["ruler_detection"]
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
        self.assertEqual(len(canvas.edge_items), 162)
        self.assertEqual(canvas.unused_nodes_button.text(), "Unused nodes (18)")
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
        self.assertEqual(len(canvas.edge_items), 154)
        self.assertEqual(canvas.unused_nodes_button.text(), "Unused nodes (19)")
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
            "Node: Grayscale and local lighting",
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
        self.assertFalse(canvas.node_items["ruler_detection"]._adjacent)
        canvas.close()

    def test_canvas_disconnects_and_reconnects_authored_ports(self) -> None:
        from seedvision.pipeline import build_default_pipeline
        from seedvision.ui.pipeline_canvas import PipelineCanvas

        graph = build_default_pipeline()
        canvas = PipelineCanvas(graph)
        changes: list[tuple[str, ...]] = []
        canvas.connections_changed.connect(changes.append)
        connection = next(
            edge
            for edge in graph.connections
            if edge.source == "deskew_colour"
            and edge.target == "refined_background_likelihood"
            and edge.source_port == "corrected_image"
        )
        canvas._disconnect_connection(connection)
        self.assertEqual(len(canvas.edge_items), 153)
        self.assertFalse(graph.node(connection.target).enabled)
        self.assertIsNone(
            graph.connection_for_input(connection.target, connection.target_port)
        )

        canvas._connection_drag = (
            connection.target,
            "input",
            connection.target_port,
        )
        canvas._connection_preview.show()
        canvas._finish_connection_drag(
            canvas.node_items[connection.source].output_anchor(
                connection.source_port
            )
        )
        self.assertEqual(len(canvas.edge_items), 154)
        self.assertTrue(graph.node(connection.target).enabled)
        self.assertIsNotNone(
            graph.connection_for_input(connection.target, connection.target_port)
        )
        self.assertEqual(len(changes), 2)
        canvas.close()

    def test_connection_paths_do_not_block_plain_left_drag_panning(self) -> None:
        from PySide6.QtCore import QPoint, Qt
        from PySide6.QtTest import QTest

        from seedvision.pipeline import build_default_pipeline
        from seedvision.ui.pipeline_canvas import PipelineCanvas, PipelineEdgeItem

        canvas = PipelineCanvas(build_default_pipeline())
        canvas.resize(900, 500)
        canvas.show()
        canvas.resetTransform()
        horizontal = canvas.horizontalScrollBar()
        vertical = canvas.verticalScrollBar()
        horizontal.setValue((horizontal.minimum() + horizontal.maximum()) // 2)
        vertical.setValue((vertical.minimum() + vertical.maximum()) // 2)
        self.application.processEvents()

        def visible_edge_point():
            for edge in canvas.edge_items:
                self.assertEqual(
                    edge.acceptedMouseButtons(), Qt.MouseButton.RightButton
                )
                for fraction in (0.2, 0.35, 0.5, 0.65, 0.8):
                    point = canvas.mapFromScene(
                        edge.path().pointAtPercent(fraction)
                    )
                    if (
                        canvas.viewport()
                        .rect()
                        .adjusted(20, 20, -20, -20)
                        .contains(point)
                        and isinstance(canvas.itemAt(point), PipelineEdgeItem)
                        and canvas._port_at(canvas.mapToScene(point)) is None
                    ):
                        return point
            return None

        hit_point = visible_edge_point()
        self.assertIsNotNone(hit_point)

        canvas.scene().clearSelection()
        start_scroll = (horizontal.value(), vertical.value())
        QTest.mousePress(
            canvas.viewport(), Qt.MouseButton.LeftButton, pos=hit_point
        )
        QTest.mouseMove(canvas.viewport(), hit_point + QPoint(70, 35), delay=10)
        QTest.mouseRelease(
            canvas.viewport(),
            Qt.MouseButton.LeftButton,
            pos=hit_point + QPoint(70, 35),
        )
        self.application.processEvents()
        self.assertNotEqual(
            (horizontal.value(), vertical.value()), start_scroll
        )
        self.assertFalse(
            any(
                isinstance(item, PipelineEdgeItem)
                for item in canvas.scene().selectedItems()
            )
        )

        # Selection remains available as an explicit gesture for keyboard
        # disconnection without making ordinary navigation fragile.
        hit_point = visible_edge_point()
        self.assertIsNotNone(hit_point)
        QTest.mouseClick(
            canvas.viewport(),
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.ControlModifier,
            pos=hit_point,
        )
        self.application.processEvents()
        self.assertTrue(
            any(
                isinstance(item, PipelineEdgeItem)
                for item in canvas.scene().selectedItems()
            )
        )
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
        self.assertEqual(canvas.unused_nodes_button.text(), "Unused nodes (19)")
        self.assertEqual(canvas.bundle_cables_button.text(), "Bundle cables")
        self.assertEqual(
            canvas.route_around_nodes_button.text(), "Route around nodes"
        )
        self.assertTrue(canvas.bundle_cables_button.isCheckable())
        self.assertTrue(canvas.route_around_nodes_button.isCheckable())
        self.assertFalse(canvas.cable_bundling_enabled)
        self.assertFalse(canvas.obstacle_routing_enabled)
        self.assertFalse(canvas.bundle_cables_button.isChecked())
        self.assertFalse(canvas.route_around_nodes_button.isChecked())
        canvas.bundle_cables_button.click()
        canvas.route_around_nodes_button.click()
        self.assertTrue(canvas.cable_bundling_enabled)
        self.assertTrue(canvas.obstacle_routing_enabled)
        self.assertIn("display only", canvas.bundle_cables_button.toolTip())
        self.assertIn("display only", canvas.route_around_nodes_button.toolTip())
        for button in (
            view.zoom_out_button,
            view.zoom_in_button,
            view.fit_button,
            view.actual_size_button,
            canvas.auto_arrange_button,
            canvas.unused_nodes_button,
            canvas.bundle_cables_button,
            canvas.route_around_nodes_button,
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
        self.assertIn("QToolButton:checked", canvas.control_bar.styleSheet())
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

    def test_selected_overlay_shows_large_red_calculating_banner(self) -> None:
        from seedvision.ui.main_window import MainWindow

        window = MainWindow(ROOT)
        path = window.image_view.image_path
        self.assertIsNotNone(path)
        window._select_overlay_mode("background_likelihood")
        window._set_analysis_running(
            path, frozenset({"background_likelihood"})
        )
        banner = window.image_view.calculating_banner
        self.assertFalse(banner.isHidden())
        self.assertIn("Calculating Background colour probability", banner.text())
        self.assertIn("#ff3434", banner.styleSheet())
        self.assertIn("font-size: 28px", banner.styleSheet())

        window._analysis_node_progress(
            str(path),
            "background_likelihood",
            "completed",
            window.pipeline.revision,
        )
        self.assertTrue(banner.isHidden())
        window.close()

    def test_project_node_displays_live_project_image_and_annotation_information(self) -> None:
        from seedvision.ui.main_window import MainWindow

        window = MainWindow(ROOT)
        window.pipeline_canvas.select_node("project")
        self.application.processEvents()

        self.assertEqual(window.pipeline_inspector.title_label.text(), "Node: Project")
        self.assertFalse(window.pipeline_inspector.project_info_container.isHidden())
        summary = window.pipeline_inspector.project_info_label.text()
        self.assertIn("Master:", summary)
        self.assertIn("State:", summary)
        self.assertIn("Images:", summary)
        self.assertIn("Current image:", summary)
        self.assertIn("Species:", summary)
        self.assertIn("Applied material annotations:", summary)
        self.assertIn("Applied seed annotations:", summary)
        self.assertEqual(
            window.pipeline.node("project").output_port_types,
            {"raw_image": "RawImage", "annotations": "ImageAnnotations"},
        )
        window.close()

    def test_identification_stages_expose_methods_and_relevant_parameters(self) -> None:
        from seedvision.pipeline import build_default_pipeline
        from seedvision.ui.pipeline_inspector import PipelineInspector

        graph = build_default_pipeline()
        inspector = PipelineInspector()
        expected_parameter_counts = {
            "seed_scale_estimation": 10,
            "background_likelihood": 22,
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
        inspector.set_node(graph.node("background_likelihood"))
        self.assertFalse(inspector.details_label.isVisibleTo(inspector))
        self.assertEqual(
            inspector.title_label.text(), "Node: Material colour probabilities"
        )
        background_node = graph.node("background_likelihood")
        self.assertEqual(
            inspector.parameter_form.rowCount(),
            len(background_node.parameter_specs)
            + len(background_node.parameter_sections),
        )
        self.assertEqual(
            [label.text() for label in inspector._parameter_section_labels],
            [section.title for section in background_node.parameter_sections],
        )
        self.assertTrue(
            all(
                label.objectName() == "parameterSectionHeading"
                for label in inspector._parameter_section_labels
            )
        )
        self.assertTrue(
            all(widget.toolTip() for widget in inspector._parameter_widgets)
        )
        for node_id, minimum_count in {
            "layout_detection": 8,
            "background_likelihood": 11,
            "refined_background_likelihood": 18,
            "edge_gradients": 17,
            "instance_masks": 3,
            "reference_edge_probability": 20,
            "edge_traces": 10,
            "seed_edge_curves": 19,
        }.items():
            inspector.set_node(graph.node(node_id))
            self.assertGreaterEqual(
                len(inspector._parameter_widgets), minimum_count, node_id
            )
        inspector.close()

    def test_background_colour_gamut_renders_as_full_hsv_value_slice(self) -> None:
        import numpy as np

        from seedvision.pipeline import build_default_pipeline
        from seedvision.ui.pipeline_inspector import (
            BackgroundColourGamut,
            PipelineInspector,
        )
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

        self.assertFalse(hasattr(inspector, "gamut_widget"))
        image, probability, centres = BackgroundColourGamut.render_hsv_value_slice(
            profile,
            build_default_pipeline().node("background_likelihood").parameters,
            class_name="background",
            value=0.90,
        )
        self.assertEqual((image.width(), image.height()), (960, 640))
        self.assertGreater(float(probability.max()), 0.50)
        self.assertLess(float(probability.min()), 0.25)
        self.assertTrue(np.all(probability[0] == 0.0))
        self.assertEqual(centres.shape, (2, 2))
        self.assertGreater(BackgroundColourGamut.dominant_hsv_value(profile), 0.0)
        inspector.close()

    def test_real_exterior_annulus_profile_retains_all_gamut_contours(self) -> None:
        import numpy as np

        from seedvision.pipeline import build_default_pipeline
        from seedvision.ui.pipeline_inspector import BackgroundColourGamut
        from seedvision.visualization import BackgroundColourProfile

        # Regression values from IMG_0002c's independently sampled exterior
        # annulus. Its pale tinted dominant mode does not lie on one exact HSV
        # slice and previously produced no 50/75/90% contours.
        profile = BackgroundColourProfile(
            centre_lab=(229.44585, 125.82103, 133.49913),
            scale_lab=(8.0, 3.0, 3.0),
            bgr_low=(196, 209, 210),
            bgr_high=(204, 217, 219),
            sample_count=32768,
            sample_fraction=0.10,
            component_centres_lab=(
                (229.44585, 125.82103, 133.49913),
                (47.87748, 127.86430, 124.57809),
                (182.71518, 127.11466, 133.07942),
                (54.76571, 127.78262, 124.78435),
            ),
            component_scales_lab=(
                (8.0, 3.0, 3.0),
                (8.0, 3.0, 3.0),
                (19.27484, 3.0, 3.0),
                (8.0, 3.0, 3.0),
            ),
            component_weights=(0.8633118, 0.0293884, 0.0072937, 0.1000061),
            refinement_iterations=0,
        )
        _image, probability, centre_points = (
            BackgroundColourGamut.render_hsv_value_slice(
                profile,
                build_default_pipeline().node("background_likelihood").parameters,
                class_name="background",
                value=0.89,
            )
        )
        self.assertGreater(float(probability.max()), 0.90)
        for level, _colour in BackgroundColourGamut.CONTOURS:
            self.assertGreater(
                int(
                    np.count_nonzero(
                        BackgroundColourGamut._contour_edge(probability, level)
                    )
                ),
                0,
                level,
            )
        self.assertEqual(centre_points.shape, (4, 2))

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
            source="painted",
            source_sample_count=420,
        )
        inspector = PipelineInspector()
        inspector.set_analysis_result(
            SimpleNamespace(
                foreground_reference_count=420,
                layers=SimpleNamespace(foreground_colour_profile=profile),
            )
        )
        inspector.set_node(build_default_pipeline().node("background_likelihood"))

        self.assertFalse(hasattr(inspector, "gamut_widget"))
        self.assertTrue(inspector.foreground_start_heading.isVisibleTo(inspector))
        self.assertIn("Bypassed", inspector.foreground_start_label.text())

        automatic_profile = ForegroundColourProfile(
            centre_lab=profile.centre_lab,
            scale_lab=profile.scale_lab,
            bgr_low=profile.bgr_low,
            bgr_high=profile.bgr_high,
            sample_count=profile.sample_count,
            sample_fraction=profile.sample_fraction,
            component_centres_lab=profile.component_centres_lab,
            component_scales_lab=profile.component_scales_lab,
            component_weights=profile.component_weights,
            refinement_iterations=0,
            source="isolated_reference_seeds",
            source_sample_count=1800,
        )
        inspector.set_analysis_result(
            SimpleNamespace(
                foreground_reference_count=0,
                layers=SimpleNamespace(
                    foreground_colour_profile=automatic_profile
                ),
            )
        )
        inspector.set_node(build_default_pipeline().node("background_likelihood"))
        self.assertIn("Isolated reference seeds", inspector.foreground_start_label.text())
        self.assertIn("#", inspector.foreground_start_label.text())
        self.assertIn("1,800", inspector.foreground_start_label.text())
        inspector.close()

    def test_nearly_neutral_dark_mode_does_not_span_every_hsv_hue(self) -> None:
        from seedvision.pipeline import build_default_pipeline
        from seedvision.ui.pipeline_inspector import BackgroundColourGamut
        from seedvision.visualization import ForegroundColourProfile

        profile = ForegroundColourProfile(
            centre_lab=(50.0, 134.0, 124.0),
            scale_lab=(10.0, 6.0, 6.0),
            bgr_low=(40, 42, 46),
            bgr_high=(50, 52, 56),
            sample_count=800,
            sample_fraction=0.10,
            component_centres_lab=((50.0, 134.0, 124.0),),
            component_scales_lab=((10.0, 6.0, 6.0),),
            component_weights=(1.0,),
            source="painted",
            source_sample_count=800,
        )
        _image, probability, centre_points = (
            BackgroundColourGamut.render_hsv_value_slice(
                profile,
                build_default_pipeline().node("background_likelihood").parameters,
                class_name="foreground",
                value=0.20,
            )
        )
        self.assertLess(float(probability.max()), 0.25)
        self.assertEqual(centre_points.shape, (1, 2))

    def test_colour_gamut_does_not_project_neutral_membership_across_hues(self) -> None:
        from seedvision.ui.pipeline_inspector import BackgroundColourGamut
        from seedvision.visualization import ForegroundColourProfile

        neutral_profile = ForegroundColourProfile(
            centre_lab=(128.0, 128.0, 128.0),
            scale_lab=(8.0, 4.0, 4.0),
            bgr_low=(118, 118, 118),
            bgr_high=(138, 138, 138),
            sample_count=500,
            sample_fraction=0.10,
            component_centres_lab=((128.0, 128.0, 128.0),),
            component_scales_lab=((8.0, 4.0, 4.0),),
            component_weights=(1.0,),
            refinement_iterations=0,
        )
        _image, probability, _centre_points = (
            BackgroundColourGamut.render_hsv_value_slice(
                neutral_profile,
                {
                    "foreground_chroma_weight": 1.8,
                    "foreground_distribution_scale_multiplier": 1.0,
                    "foreground_frequency_weight_power": 0.0,
                },
                class_name="foreground",
                value=0.50,
            )
        )
        self.assertTrue((probability == 0.0).all())

    def test_overlay_node_selection_updates_the_viewer_layer(self) -> None:
        from seedvision.ui.main_window import MainWindow

        window = MainWindow(ROOT)
        window.pipeline_canvas.select_node("edge_gradients")
        self.application.processEvents()
        directed_index = window.pipeline_inspector.overlay_combo.findData(
            "directed_edges"
        )
        self.assertGreaterEqual(directed_index, 0)
        window.pipeline_inspector.overlay_combo.setCurrentIndex(directed_index)
        self.application.processEvents()
        self.assertEqual(window.overlay_combo.currentData(), "directed_edges")
        self.assertEqual(window.image_view._overlay_mode, "directed_edges")
        self.assertEqual(
            window.pipeline_inspector.title_label.text(),
            "Node: Edge gradients",
        )
        self.assertEqual(len(window.pipeline_inspector._parameter_widgets), 17)
        self.assertEqual(
            len(window.pipeline_canvas.node_items["edge_gradients"]._inline_editors),
            4,
        )
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
            "Node: Material noise probabilities",
        )
        self.assertEqual(len(window.pipeline_inspector._parameter_widgets), 18)
        node_overlays = {
            window.pipeline_inspector.overlay_combo.itemData(index)
            for index in range(window.pipeline_inspector.overlay_combo.count())
        }
        self.assertTrue(
            {
                "refined_background_likelihood",
                "foreground_noise_likelihood",
                "other_noise_probability",
            }.issubset(node_overlays)
        )
        window.pipeline_canvas.select_node("boundary_normals")
        self.application.processEvents()
        self.assertEqual(window.overlay_combo.currentData(), "boundary_confidence")
        self.assertEqual(window.image_view._overlay_mode, "boundary_confidence")
        self.assertEqual(len(window.pipeline_inspector._parameter_widgets), 1)
        window.pipeline_canvas.select_node("ruler_detection")
        self.application.processEvents()
        self.assertEqual(window.overlay_combo.currentData(), "ruler_detection")
        calibrated_index = window.pipeline_inspector.overlay_combo.findData(
            "calibrated_image"
        )
        self.assertGreaterEqual(calibrated_index, 0)
        window.pipeline_inspector.overlay_combo.setCurrentIndex(calibrated_index)
        self.application.processEvents()
        self.assertEqual(window.overlay_combo.currentData(), "calibrated_image")
        window.pipeline_canvas.select_node("deskew_colour")
        self.application.processEvents()
        self.assertEqual(window.overlay_combo.currentData(), "deskew_colour")
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
            window.overlay_owner_label.text(), "Boundary confidence and normals"
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

        window.pipeline_canvas.select_node("background_likelihood")
        self.application.processEvents()
        self.assertEqual(window.overlay_combo.currentData(), "background_likelihood")
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
                "background_likelihood"
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
        self.assertTrue(window.overlay_combo.isHidden())
        self.assertIs(window.overlay_button.menu(), window.overlay_menu)
        self.assertEqual(
            window.overlay_combo.findData("automatic_foreground_colour_evidence"),
            -1,
        )
        self.assertEqual(window.overlay_combo.findData("foreground_feature"), -1)
        node_menus = {
            action.text(): tuple(child.text() for child in action.menu().actions())
            for action in window.overlay_menu.actions()
            if action.menu() is not None
        }
        self.assertNotIn("Viewer", node_menus)
        self.assertNotIn(
            "None",
            {label for labels in node_menus.values() for label in labels},
        )
        self.assertIn(
            "Estimated illumination field",
            node_menus["Grayscale and local lighting"],
        )
        nodes_by_title = {
            node.title: node for node in window.pipeline.nodes.values()
        }
        menu_contracts = tuple(
            (
                menu_action.text(),
                tuple(
                    (overlay_action.text(), overlay_action.isCheckable())
                    for overlay_action in menu_action.menu().actions()
                ),
            )
            for menu_action in window.overlay_menu.actions()
            if menu_action.menu() is not None
        )
        self.assertTrue(all("&" not in title for title, _labels in menu_contracts))
        for menu_title, overlay_contracts in menu_contracts:
            self.assertIn(menu_title, nodes_by_title)
            node = nodes_by_title[menu_title]
            self.assertIn(node.identifier, window.pipeline_canvas.node_items)
            window.pipeline_canvas.select_node(node.identifier)
            self.application.processEvents()
            self.assertEqual(
                window.pipeline_inspector.title_label.text(),
                f"Node: {menu_title}",
            )
            output_labels = set(dict(node.output_ports).values())
            for overlay_label, is_checkable in overlay_contracts:
                self.assertFalse(is_checkable)
                self.assertIn(overlay_label, output_labels)
        self.assertIn("Directional surface darkness gradients", node_menus)
        self.assertIn("Edge gradients", node_menus)
        self.assertIn("Reference texture prototypes", node_menus)
        edge_actions = set(node_menus["Edge gradients"])
        self.assertIn("Edge tangent (undirected)", edge_actions)
        self.assertIn("Edge tangent (directed)", edge_actions)

        window.pipeline_canvas.select_node("hue_only")
        self.application.processEvents()
        node_selector = window.pipeline_inspector.overlay_combo
        self.assertEqual(node_selector.count(), 1)
        self.assertEqual(node_selector.currentData(), "hue_only")

        window.pipeline_canvas.select_node("wavelet_decomposition")
        self.application.processEvents()
        self.assertEqual(node_selector.count(), 5)
        self.assertEqual(node_selector.currentData(), "wavelet_detail_1")
        self.assertGreaterEqual(node_selector.findData("wavelet_residual"), 0)

        window.pipeline_canvas.select_node("frequency_noise_masks")
        self.application.processEvents()
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
        self.assertEqual(node_selector.count(), 7)
        self.assertEqual(node_selector.currentData(), "procedural_instances")
        self.assertGreaterEqual(node_selector.findData("procedural_confidence"), 0)
        self.assertGreaterEqual(node_selector.findData("procedural_concavity"), 0)
        self.assertGreaterEqual(
            node_selector.findData("procedural_alternative_candidates"), 0
        )

        window.pipeline_canvas.select_node("reference_texture_prototypes")
        self.application.processEvents()
        self.assertEqual(node_selector.count(), 7)
        self.assertGreaterEqual(
            node_selector.findData("reference_texture_prototypes"), 0
        )
        self.assertGreaterEqual(
            node_selector.findData("reference_prototype_footprints"), 0
        )
        self.assertGreaterEqual(
            node_selector.findData("reference_seed_surface_probability"), 0
        )
        self.assertGreaterEqual(
            node_selector.findData("reference_background_texture_probability"), 0
        )
        self.assertGreaterEqual(
            node_selector.findData("reference_other_texture_probability"), 0
        )
        self.assertGreaterEqual(
            node_selector.findData("physical_edge_probability"), 0
        )
        self.assertGreaterEqual(
            node_selector.findData("non_edge_probability"), 0
        )

        window.pipeline_canvas.select_node("reference_edge_probability")
        self.application.processEvents()
        self.assertEqual(node_selector.count(), 6)
        self.assertEqual(node_selector.currentData(), "reference_edge_comparison")
        self.assertEqual(
            window.pipeline.node("reference_edge_probability").parameters[
                "net_physical_edge_internal_scale"
            ],
            0.5,
        )
        self.assertEqual(len(window.pipeline_inspector._parameter_widgets), 20)
        self.assertEqual(
            window.pipeline.node("reference_edge_probability").parameter_specs[0].label,
            "Non-physical subtraction weight",
        )
        comparison_index = node_selector.findData("reference_edge_comparison")
        net_index = node_selector.findData("net_physical_edge_probability")
        self.assertGreaterEqual(comparison_index, 0)
        self.assertGreaterEqual(net_index, 0)
        self.assertEqual(
            node_selector.itemText(comparison_index),
            "Physical blue / non-physical red",
        )
        node_selector.setCurrentIndex(comparison_index)
        self.application.processEvents()
        self.assertEqual(
            window.overlay_combo.currentData(), "reference_edge_comparison"
        )
        self.assertIn("Magenta", window.overlay_legend_label.text())
        node_selector.setCurrentIndex(net_index)
        self.application.processEvents()
        self.assertEqual(
            window.image_view._overlay_mode, "net_physical_edge_probability"
        )
        self.assertIn("black floor", window.overlay_legend_label.text())
        self.assertIn("0.5 × non-physical edge", window.overlay_legend_label.text())

        self.assertGreaterEqual(
            node_selector.findData("net_reference_edge_ridges"), 0
        )
        self.assertGreaterEqual(
            node_selector.findData("locally_normalized_net_physical_edge"), 0
        )
        self.assertGreaterEqual(
            node_selector.findData("normalized_net_reference_edge_ridges"), 0
        )

        window.pipeline_canvas.select_node("seed_edge_curves")
        self.application.processEvents()
        self.assertEqual(node_selector.count(), 11)
        self.assertGreaterEqual(
            node_selector.findData("edge_oval_hypotheses"), 0
        )
        self.assertGreaterEqual(
            node_selector.findData("oval_centre_probability"), 0
        )
        curve_node = window.pipeline.node("seed_edge_curves")
        self.assertEqual(
            curve_node.output_port_types["oval_hypotheses"],
            "EdgeOvalHypotheses",
        )
        self.assertEqual(
            curve_node.output_port_types["oval_centre_probability"],
            "CentreProbability",
        )
        reference_index = window.overlay_combo.findData("reference_edge_ridges")
        self.assertGreater(reference_index, 0)
        self.assertEqual(
            window.overlay_combo.itemText(reference_index).strip(),
            "Thinned reference edge ridge",
        )

        window.pipeline_canvas.select_node("metadata")
        self.application.processEvents()
        self.assertEqual(node_selector.itemText(0), "No image overlays")
        self.assertFalse(node_selector.isEnabled())
        window.close()

    def test_edge_oval_overlay_draws_only_retained_edge_hypotheses(self) -> None:
        from types import SimpleNamespace

        import numpy as np

        from seedvision.ui.image_view import ImageView

        geometry = SimpleNamespace(
            materialize=lambda: {
                "oval_centres_xy": np.asarray(((30.0, 40.0), (80.0, 90.0))),
                "oval_axes_xy": np.asarray(((18.0, 12.0), (9.0, 7.0))),
                "oval_angle_radians": np.deg2rad((25.0, 0.0)),
                "oval_confidence": np.asarray((0.8, 0.0)),
            }
        )
        result = SimpleNamespace(
            layers=SimpleNamespace(edge_fit_geometry=geometry),
            crop_offset=(5, 7),
        )
        view = ImageView()
        view._render_edge_oval_hypotheses(result)
        self.assertEqual(len(view._overlay_items), 1)
        item = view._overlay_items[0]
        self.assertAlmostEqual(item.pos().x(), 35.0)
        self.assertAlmostEqual(item.pos().y(), 47.0)
        self.assertAlmostEqual(item.rotation(), 25.0)
        self.assertGreater(item.pen().widthF(), 2.0)
        view.close()

    def test_other_probability_overlays_are_registered_owned_and_synchronised(self) -> None:
        from seedvision.ui.main_window import MainWindow, _overlay_node_owner

        window = MainWindow(ROOT)
        expectations = (
            (
                "other_colour_probability",
                "Other colour probability",
                "background_likelihood",
            ),
            (
                "other_noise_probability",
                "Other noise probability",
                "refined_background_likelihood",
            ),
        )
        for mode, label, owner in expectations:
            self.assertEqual(_overlay_node_owner(mode), owner)
            global_index = window.overlay_combo.findData(mode)
            self.assertGreaterEqual(global_index, 0)
            self.assertEqual(
                window.overlay_combo.itemText(global_index).strip(), label
            )

            window.pipeline_canvas.select_node(owner)
            self.application.processEvents()
            local_selector = window.pipeline_inspector.overlay_combo
            local_index = local_selector.findData(mode)
            self.assertGreaterEqual(local_index, 0)
            self.assertEqual(local_selector.itemText(local_index), label)
            local_selector.setCurrentIndex(local_index)
            self.application.processEvents()

            self.assertEqual(window.overlay_combo.currentData(), mode)
            self.assertEqual(window.image_view._overlay_mode, mode)
            self.assertEqual(window._selected_pipeline_node, owner)
            legend = window.overlay_legend_label.text()
            self.assertIn("painted Other", legend)
            self.assertIn("Reference Other-material probability", legend)
            if mode == "other_colour_probability":
                self.assertIn("not forced output", legend)
            else:
                self.assertIn("non-Other", legend)
                self.assertIn("Directional", legend)
                self.assertIn("integration", legend)
        window.close()

    def test_probability_legends_state_low_and_high_display_values(self) -> None:
        from seedvision.ui.main_window import MainWindow

        window = MainWindow(ROOT)
        window._update_overlay_legend("background_likelihood")
        self.assertIn(
            "white = low displayed probability/evidence; black = high",
            window.overlay_legend_label.text(),
        )
        window._update_overlay_legend("foreground_mask")
        foreground_legend = window.overlay_legend_label.text()
        self.assertIn("black = low displayed probability/evidence", foreground_legend)
        self.assertIn("need not sum to one", foreground_legend)
        window._update_overlay_legend("reference_background_texture_probability")
        reference_legend = window.overlay_legend_label.text()
        self.assertIn("black = low displayed probability/evidence", reference_legend)
        self.assertIn("no-match", reference_legend)
        window.close()

    def test_compact_toolbar_and_contextual_paint_panel(self) -> None:
        from PySide6.QtCore import QPoint, Qt
        from PySide6.QtTest import QTest
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
        self.assertEqual(window.paint_background_action.text(), "Material references")
        self.assertEqual(
            window.annotate_instances_action.text(), "Annotate seed instances"
        )
        self.assertFalse(hasattr(window, "paint_foreground_action"))
        self.assertFalse(hasattr(window, "show_boundary_references_checkbox"))
        self.assertFalse(
            hasattr(window, "use_annotated_instance_boundaries_checkbox")
        )
        foreground_gamut = window.overlay_combo.findData(
            "foreground_colour_gamut"
        )
        background_gamut = window.overlay_combo.findData(
            "background_colour_gamut"
        )
        self.assertGreaterEqual(foreground_gamut, 0)
        self.assertGreaterEqual(background_gamut, 0)
        window.overlay_combo.setCurrentIndex(foreground_gamut)
        self.application.processEvents()
        self.assertTrue(window.hsv_value_slider_action.isVisible())
        self.assertTrue(window.hsv_peak_button_action.isVisible())
        self.assertFalse(window.overlay_opacity_slider_action.isVisible())
        self.assertEqual(
            window.pipeline_inspector.overlay_combo.findData(
                "foreground_colour_gamut"
            ),
            window.pipeline_inspector.overlay_combo.currentIndex(),
        )
        window.hsv_value_slider.setValue(42)
        self.assertEqual(window.hsv_value_label.text(), "42%")
        self.assertAlmostEqual(window.image_view._hsv_gamut_value, 0.42)
        raw_overlay = window.overlay_combo.findData("raw_image")
        window.overlay_combo.setCurrentIndex(raw_overlay)
        self.application.processEvents()
        self.assertFalse(window.hsv_value_slider_action.isVisible())
        self.assertFalse(window.hsv_peak_button_action.isVisible())
        self.assertTrue(window.overlay_opacity_slider_action.isVisible())
        prototype_collage = window.overlay_combo.findData(
            "reference_texture_prototypes"
        )
        self.assertGreaterEqual(prototype_collage, 0)
        window.overlay_combo.setCurrentIndex(prototype_collage)
        self.application.processEvents()
        self.assertFalse(window.overlay_opacity_slider_action.isVisible())

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
        window.annotate_instances_action.setEnabled(True)
        window.background_point_button.setEnabled(True)
        window.foreground_point_button.setEnabled(True)
        window.background_exclusion_button.setEnabled(True)
        window.erase_background_points_button.setEnabled(True)
        window.erase_foreground_points_button.setEnabled(True)
        window.erase_background_exclusion_button.setEnabled(True)
        window.erase_foreground_exclusion_button.setEnabled(True)
        window.paint_background_action.setChecked(True)
        self.application.processEvents()
        self.assertTrue(window.background_point_button.isChecked())
        self.assertFalse(window.reference_panel.isHidden())
        self.assertEqual(window.image_view._reference_point_mode, "background")
        window.resize(1200, 650)
        window._show_split_workspace()
        self.application.processEvents()
        self.assertGreater(
            window.reference_panel_scroll.verticalScrollBar().maximum(), 0
        )
        self.assertGreaterEqual(
            window.foreground_point_button.height(),
            window.foreground_point_button.sizeHint().height(),
        )
        self.assertLessEqual(
            window.reference_panel_contents.width(),
            window.reference_panel_scroll.viewport().width(),
        )
        initial_panel_position = window.reference_panel.pos()
        drag_center = window.reference_panel_drag_handle.rect().center()
        QTest.mousePress(
            window.reference_panel_drag_handle,
            Qt.MouseButton.LeftButton,
            pos=drag_center,
        )
        QTest.mouseMove(
            window.reference_panel_drag_handle,
            drag_center + QPoint(60, 30),
            delay=10,
        )
        QTest.mouseRelease(
            window.reference_panel_drag_handle,
            Qt.MouseButton.LeftButton,
            pos=drag_center + QPoint(60, 30),
        )
        self.application.processEvents()
        self.assertNotEqual(window.reference_panel.pos(), initial_panel_position)
        window.image_view._move_context_panel_to(QPoint(180, 90), remember=True)
        self.application.processEvents()
        moved_panel_position = window.reference_panel.pos()
        window.image_view._layout_context_panel()
        self.assertEqual(window.reference_panel.pos(), moved_panel_position)
        self.assertEqual(
            window.reference_panel_drag_handle.toolTip(),
            "Drag this bar to reposition the painting controls over the image.",
        )

        window.background_exclusion_button.setChecked(True)
        self.application.processEvents()
        self.assertEqual(window.image_view._reference_point_mode, "other")
        self.assertFalse(window.reference_visibility_controls.isHidden())
        window.show_material_references_checkbox.setChecked(False)
        self.assertFalse(
            window.image_view._material_reference_annotations_visible
        )
        self.assertFalse(window.image_view._boundary_reference_annotations_visible)
        self.assertFalse(window.image_view._reference_annotations_visible)
        window.show_material_references_checkbox.setChecked(True)
        self.assertTrue(
            window.image_view._material_reference_annotations_visible
        )

        window.reference_eraser_button.setChecked(True)
        self.assertTrue(window.reference_eraser_button.isChecked())
        self.assertEqual(window.image_view._reference_point_mode, "other")

        window.annotate_instances_action.setChecked(True)
        self.application.processEvents()
        self.assertFalse(window.paint_background_action.isChecked())
        self.assertEqual(window.image_view._reference_point_mode, "instance")
        self.assertTrue(window.reference_controls.isHidden())
        self.assertFalse(window.instance_annotation_controls.isHidden())
        self.assertFalse(window.reference_visibility_controls.isHidden())
        self.assertIn(
            "automatically supply physical contours",
            window.instance_boundary_supervision_label.text(),
        )
        window.show_instance_annotations_checkbox.setChecked(False)
        self.assertFalse(window.image_view._instance_annotations_visible)
        self.assertEqual(
            window.instance_proposal_combo.itemText(0),
            "No instance result available",
        )
        self.assertFalse(window.use_instance_proposal_button.isEnabled())
        window._sync_annotation_proposal_choices(
            SimpleNamespace(
                procedural_instances=SimpleNamespace(count=17),
                unet_instances=None,
                stardist_instances=SimpleNamespace(count=16),
            ),
            enabled=True,
        )
        self.assertEqual(window.instance_proposal_combo.count(), 2)
        self.assertEqual(
            window.instance_proposal_combo.itemData(0), "procedural_instances"
        )
        self.assertEqual(
            window.instance_proposal_combo.itemData(1), "stardist_instances"
        )
        self.assertTrue(window.use_instance_proposal_button.isEnabled())
        window.instance_eraser_button.setChecked(True)
        self.assertEqual(window.image_view._instance_annotation_tool, "eraser")

        window.annotate_instances_action.setChecked(False)
        self.application.processEvents()
        self.assertTrue(window.reference_panel.isHidden())
        self.assertIsNone(window.image_view._reference_point_mode)
        window.close()

    def test_cleared_reference_layer_returns_to_paint_and_accepts_dabs(self) -> None:
        import numpy as np
        from PySide6.QtCore import QPointF

        from seedvision.ui.main_window import MainWindow

        window = MainWindow(ROOT)
        key = window._current_image_key()
        if key is None or window.image_view.image_size is None:
            window.close()
            self.skipTest("No workspace image is present")
        width, height = window.image_view.image_size
        window.foreground_point_button.setEnabled(True)
        window.foreground_point_button.setChecked(True)
        window.reference_eraser_button.setEnabled(True)
        window.reference_eraser_button.setChecked(True)
        self.assertTrue(window.image_view._reference_erase_enabled)

        window._clear_foreground_points()

        self.assertTrue(window.reference_paint_mode_button.isChecked())
        self.assertFalse(window.reference_eraser_button.isChecked())
        self.assertFalse(window.image_view._reference_erase_enabled)
        self.assertFalse(
            np.any(window.image_view.reference_mask("foreground", copy=False))
        )
        point = QPointF(float(min(40, width - 1)), float(min(40, height - 1)))
        window.image_view._emit_reference_brush_dab(
            point, erase=window.image_view._reference_erase_enabled
        )
        self.assertTrue(
            np.any(window.image_view.reference_mask("foreground", copy=False))
        )
        window.close()

    def test_pipeline_instance_result_becomes_editable_draft_with_provenance(self) -> None:
        import numpy as np

        from seedvision.ui.main_window import MainWindow

        window = MainWindow(ROOT)
        key = window._current_image_key()
        image_size = window.image_view.image_size
        if key is None or image_size is None:
            window.close()
            self.skipTest("No workspace image is present")
        width, height = image_size
        proposal = SimpleNamespace(
            count=2,
            labels=np.asarray(
                ((0, 4, 4), (9, 9, 0), (9, 0, 0)), dtype=np.int32
            ),
        )
        result = SimpleNamespace(
            calibration=SimpleNamespace(
                corrected_bgr=SimpleNamespace(shape=(height, width, 3))
            ),
            layers=SimpleNamespace(valid_mask=np.ones((6, 6), dtype=np.uint8)),
            crop_offset=(2, 3),
            procedural_instances=proposal,
            unet_instances=None,
            stardist_instances=None,
        )
        window._analyses[key] = result
        window._sync_annotation_proposal_choices(result, enabled=True)

        window._use_instance_proposal_as_draft()

        draft = window._draft_instance_annotations[key]
        self.assertEqual(draft.shape, (height, width))
        self.assertEqual(set(np.unique(draft)), {0, 1, 2})
        self.assertIn(key, window._instance_annotations_dirty)
        self.assertEqual(
            window._draft_instance_annotation_origins[key],
            "pipeline:procedural_instances",
        )
        window.close()

    def test_background_node_can_be_disabled_without_disabling_foreground_noise(self) -> None:
        from seedvision.ui.main_window import MainWindow

        window = MainWindow(ROOT)
        window._pipeline_parameter_changed(
            "background_likelihood", "background_colour_enabled", False
        )
        self.assertFalse(
            window.pipeline.node("background_likelihood").parameters[
                "background_colour_enabled"
            ]
        )
        self.assertTrue(window.pipeline.node("background_likelihood").enabled)
        self.assertTrue(window.pipeline.node("refined_background_likelihood").enabled)
        self.assertFalse(window.background_enabled_checkbox.isChecked())
        window.close()

    def test_combined_noise_node_preserves_independent_enable_switches(self) -> None:
        from seedvision.ui.main_window import MainWindow

        window = MainWindow(ROOT)
        window._pipeline_parameter_changed(
            "refined_background_likelihood", "background_noise_enabled", False
        )
        node = window.pipeline.node("refined_background_likelihood")
        self.assertFalse(node.parameters["background_noise_enabled"])
        self.assertTrue(node.parameters["foreground_noise_enabled"])
        self.assertTrue(node.enabled)
        settings = window._layer_settings()
        self.assertFalse(settings.background_noise_enabled)
        self.assertTrue(settings.foreground_noise_enabled)
        window.close()

    def test_settings_reset_button_restores_node_defaults(self) -> None:
        from seedvision.ui.main_window import MainWindow

        window = MainWindow(ROOT)
        node = window.pipeline.node("background_likelihood")
        default = node.default_parameters["foreground_reference_weight"]
        window._pipeline_parameter_changed(
            "background_likelihood", "foreground_reference_weight", 0.25
        )
        window.pipeline_canvas.select_node("background_likelihood")
        self.application.processEvents()
        window.pipeline_inspector.reset_parameters_button.click()
        self.application.processEvents()
        self.assertEqual(node.parameters["foreground_reference_weight"], default)
        window.close()

    def test_expanded_node_settings_feed_runtime_settings_objects(self) -> None:
        from seedvision.ui.main_window import MainWindow

        window = MainWindow(ROOT)
        self.assertTrue(
            window.pipeline.node("background_likelihood").parameters[
                "background_keep_perimeter_reference"
            ]
        )
        self.assertTrue(
            window.keep_perimeter_background_reference_checkbox.isChecked()
        )
        self.assertIn(
            "in addition to any painted Background marks",
            window.keep_perimeter_background_reference_checkbox.toolTip(),
        )
        window.pipeline.set_parameter(
            "layout_detection", "hough_accumulator_threshold", 41
        )
        window.pipeline.set_parameter(
            "refined_background_likelihood",
            "foreground_noise_vector_length_fraction",
            0.75,
        )
        window.pipeline.set_parameter("edge_gradients", "edge_chroma_weight", 1.8)
        window.pipeline.set_parameter(
            "edge_traces", "trace_curvature_policy", "require"
        )
        window.pipeline.set_parameter(
            "edge_traces", "trace_curvature_tolerance_degrees", 4.0
        )
        window.pipeline.set_parameter(
            "background_likelihood", "background_colour_components", 64
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
            "background_likelihood",
            "foreground_distribution_scale_multiplier",
            1.7,
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
        self.assertEqual(
            window._layer_settings().trace_curvature_policy,
            "require",
        )
        self.assertAlmostEqual(
            window._layer_settings().trace_curvature_tolerance_degrees,
            4.0,
        )
        self.assertEqual(window._layer_settings().background_colour_components, 64)
        self.assertAlmostEqual(
            window._layer_settings().perimeter_background_buffer_cm,
            0.45,
        )
        self.assertAlmostEqual(
            window._layer_settings().perimeter_background_band_thickness_cm,
            0.70,
        )
        self.assertAlmostEqual(
            window._baseline_settings().foreground_distribution_scale_multiplier,
            1.7,
        )
        self.assertEqual(
            window.pipeline.node("edge_gradients").parameters["edge_chroma_weight"],
            1.8,
        )
        inline_editor = window.pipeline_canvas.node_items[
            "edge_gradients"
        ]._inline_editors["edge_blur_sigma"]
        inline_editor.setValue(1.9)
        inline_editor.editingFinished.emit()
        self.application.processEvents()
        self.assertAlmostEqual(window._layer_settings().edge_blur_sigma, 1.9)
        window.pipeline.set_parameter(
            "image_quality", "quality_noise_scale_fraction", 0.05
        )
        self.assertAlmostEqual(
            window._advanced_settings().quality_noise_scale_fraction, 0.05
        )
        with self.assertRaises(ValueError):
            window._validate_settings_override(
                "edge_gradients", "ridge_low_threshold", 0.30
            )
        window.close()

    def test_setting_change_invalidates_only_node_and_graph_dependents(self) -> None:
        from seedvision.pipeline import NodeStatus
        from seedvision.ui.main_window import MainWindow

        window = MainWindow(ROOT)
        for node_id in (
            "edge_gradients",
            "reference_texture_prototypes",
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
            window.pipeline.node("edge_traces").status, NodeStatus.IDLE
        )
        window.close()

    def test_perimeter_source_checkbox_invalidates_every_background_consumer(self) -> None:
        from unittest.mock import patch

        from seedvision.segmentation import PipelineAnalysisCache
        from seedvision.ui.main_window import MainWindow

        with patch.object(
            MainWindow, "_auto_load_reference_regions", return_value=False
        ):
            window = MainWindow(ROOT)
        key = window._current_image_key()
        if key is None:
            window.close()
            self.skipTest("No workspace image is present")
        window._analysis_caches[key] = PipelineAnalysisCache()
        window._analyses[key] = object()
        window._cache_dirty_nodes.pop(key, None)
        scheduled: list[set[str]] = []
        window._analyze_current_image = lambda **kwargs: scheduled.append(
            set(kwargs.get("dirty_nodes", ()))
        )

        self.assertTrue(
            window.keep_perimeter_background_reference_checkbox.isChecked()
        )
        window.keep_perimeter_background_reference_checkbox.setChecked(False)

        self.assertFalse(
            window.pipeline.node("background_likelihood").parameters[
                "background_keep_perimeter_reference"
            ]
        )
        self.assertFalse(
            window._layer_settings().background_keep_perimeter_reference
        )
        self.assertFalse(
            window.image_view._automatic_background_reference_visible
        )
        dirty = window._cache_dirty_nodes[key]
        for node_id in (
            "background_likelihood",
            "refined_background_likelihood",
            "reference_texture_prototypes",
        ):
            self.assertIn(node_id, dirty)
        self.assertEqual(len(scheduled), 1)
        self.assertTrue(dirty.issubset(scheduled[0]))
        window.close()

    def test_annotated_seed_foreground_checkbox_defaults_on_and_invalidates_consumers(self) -> None:
        from unittest.mock import patch

        from seedvision.segmentation import PipelineAnalysisCache
        from seedvision.ui.main_window import MainWindow

        with patch.object(
            MainWindow, "_auto_load_reference_regions", return_value=False
        ):
            window = MainWindow(ROOT)
        key = window._current_image_key()
        if key is None:
            window.close()
            self.skipTest("No workspace image is present")
        window._analysis_caches[key] = PipelineAnalysisCache()
        window._analyses[key] = object()
        window.image_view._analysis_result = object()
        window._cache_dirty_nodes.pop(key, None)
        scheduled: list[set[str]] = []
        window._analyze_current_image = lambda **kwargs: scheduled.append(
            set(kwargs.get("dirty_nodes", ()))
        )
        window._sync_background_controls()

        checkbox = window.include_seed_instances_as_foreground_checkbox
        self.assertTrue(checkbox.isChecked())
        self.assertTrue(checkbox.isEnabled())
        self.assertIn("safely inset interiors", checkbox.toolTip())
        self.assertIn("Apply + save", checkbox.toolTip())
        checkbox.setChecked(False)

        self.assertFalse(
            window.pipeline.node("background_likelihood").parameters[
                "foreground_include_annotated_seed_instances"
            ]
        )
        self.assertFalse(
            window._baseline_settings().foreground_include_annotated_seed_instances
        )
        dirty = window._cache_dirty_nodes[key]
        for node_id in (
            "background_likelihood",
            "refined_background_likelihood",
            "reference_texture_prototypes",
        ):
            self.assertIn(node_id, dirty)
        self.assertEqual(len(scheduled), 1)
        self.assertTrue(dirty.issubset(scheduled[0]))

        # Programmatic inspector/reset-style edits mirror the contextual control.
        window._pipeline_parameter_changed(
            "background_likelihood",
            "foreground_include_annotated_seed_instances",
            False,
        )
        self.assertFalse(checkbox.isChecked())
        window.close()

    def test_net_edge_subtraction_recomputes_thinned_net_ridge_dependents(self) -> None:
        from unittest.mock import patch

        from PySide6.QtWidgets import QMessageBox

        from seedvision.pipeline import NodeStatus
        from seedvision.segmentation import PipelineAnalysisCache
        from seedvision.ui.main_window import MainWindow
        warning_patcher = patch.object(
            QMessageBox,
            "warning",
            return_value=QMessageBox.StandardButton.No,
        )
        warning_patcher.start()
        self.addCleanup(warning_patcher.stop)
        window = MainWindow(ROOT)
        path = ROOT / "images" / "IMG_0002c.JPG"
        key = str(path.resolve()).casefold()
        window.image_view._image_path = path
        window._analyses[key] = object()
        sentinel = object()
        window._analysis_caches[key] = PipelineAnalysisCache(
            values={"sentinel": sentinel}
        )
        window._cache_dirty_nodes[key] = set()
        net_index = window.overlay_combo.findData("net_physical_edge_probability")
        self.assertGreaterEqual(net_index, 0)
        window.overlay_combo.setCurrentIndex(net_index)
        node = window.pipeline.node("reference_edge_probability")
        node.status = NodeStatus.COMPLETE
        window.pipeline.node("procedural_instances").status = NodeStatus.COMPLETE
        revision = window.pipeline.revision

        with patch.object(window, "_analyze_current_image") as analyze:
            window._pipeline_parameter_changed(
                "reference_edge_probability",
                "net_physical_edge_internal_scale",
                1.25,
            )
            affected = {
                "reference_edge_probability",
                "edge_traces",
                "unet_instances",
                "stardist_instances",
                "seed_edge_curves",
                "procedural_instances",
            }
            analyze.assert_called_once_with(dirty_nodes=affected)
            self.assertNotIn(key, window._analyses)
            self.assertEqual(window.pipeline.revision, revision + 1)
            self.assertEqual(window._cache_dirty_nodes[key], affected)
            self.assertIs(window._analysis_caches[key].values["sentinel"], sentinel)
            self.assertEqual(node.status, NodeStatus.IDLE)
            self.assertEqual(
                window.pipeline.node("procedural_instances").status,
                NodeStatus.IDLE,
            )

            window._pipeline_parameters_reset("reference_edge_probability")
            self.assertEqual(analyze.call_count, 2)
            self.assertEqual(window.pipeline.revision, revision + 2)
            self.assertEqual(window._cache_dirty_nodes[key], affected)

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
            reference_seed_surface_probability=np.full(
                (12, 12), 135, np.uint8
            ),
            reference_background_texture_probability=np.full(
                (12, 12), 115, np.uint8
            ),
            reference_other_texture_probability=np.full(
                (12, 12), 95, np.uint8
            ),
            other_colour_probability=np.full((12, 12), 73, np.uint8),
            other_noise_probability=np.full((12, 12), 203, np.uint8),
            physical_edge_probability=np.full((12, 12), 105, np.uint8),
            non_edge_probability=np.full((12, 12), 85, np.uint8),
            reference_edge_ridges=np.full((12, 12), 125, np.uint8),
            locally_normalized_net_physical_edge=np.full(
                (12, 12), 155, np.uint8
            ),
            normalized_net_reference_edge_ridges=np.full(
                (12, 12), 145, np.uint8
            ),
            hue_only_rgb=np.full((12, 12, 3), (158, 0, 0), np.uint8),
            wavelet_details=tuple(
                np.full((12, 12, 3), float(level), np.float32)
                for level in (2, 4, 6, 8)
            ),
            wavelet_residual=np.full((12, 12, 3), 128.0, np.float32),
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
            "other_colour_probability",
            "other_noise_probability",
            "reference_seed_surface_probability",
            "reference_background_texture_probability",
            "reference_other_texture_probability",
            "physical_edge_probability",
            "non_edge_probability",
            "reference_edge_comparison",
            "net_physical_edge_probability",
            "locally_normalized_net_physical_edge",
            "reference_edge_ridges",
            "normalized_net_reference_edge_ridges",
            "hue_only",
            "wavelet_detail_1",
            "wavelet_detail_2",
            "wavelet_detail_3",
            "wavelet_detail_4",
            "wavelet_residual",
            "undirected_edges",
            "directed_edges",
            "seed_edge_curves",
            "foreground_mask",
            "foreground_binary_mask",
        ):
            view.set_overlay_mode(mode)
            self.assertEqual(len(view._overlay_items), 1)
        from PySide6.QtWidgets import QGraphicsPixmapItem

        for mode, expected in (
            ("other_colour_probability", 73),
            ("other_noise_probability", 203),
        ):
            view.set_overlay_mode(mode)
            raster_items = [
                item
                for item in view._overlay_items
                if isinstance(item, QGraphicsPixmapItem)
            ]
            self.assertEqual(len(raster_items), 1)
            image = raster_items[0].pixmap().toImage()
            colour = image.pixelColor(0, 0)
            self.assertEqual(
                (colour.red(), colour.green(), colour.blue()),
                (expected, expected, expected),
            )
            self.assertEqual(raster_items[0].pos().x(), layers.offset_x)
            self.assertEqual(raster_items[0].pos().y(), layers.offset_y)
        view.set_overlay_opacity(0.42)
        self.assertAlmostEqual(view._overlay_items[0].opacity(), 0.42)
        view.set_overlay_mode("raw_image")
        self.assertEqual(len(view._overlay_items), 0)
        view.set_overlay_mode("none")
        self.assertEqual(len(view._overlay_items), 0)
        background = np.zeros((12, 12), dtype=bool)
        foreground = np.zeros_like(background)
        other = np.zeros_like(background)
        physical_edge = np.zeros_like(background)
        non_edge = np.zeros_like(background)
        instances = np.zeros((12, 12), dtype=np.uint16)
        background[1:3, 1:3] = True
        foreground[4:6, 1:3] = True
        other[7:9, 1:3] = True
        physical_edge[1:3, 7:9] = True
        non_edge[4:6, 7:9] = True
        instances[8:10, 8:10] = 4
        view.set_reference_masks(
            background,
            foreground,
            other,
            other,
            physical_edge_mask=physical_edge,
            non_edge_mask=non_edge,
            normalize_material=False,
            render=False,
        )
        view.set_instance_annotations(instances, render=False)
        view.refresh_analysis()
        # Legacy edge/non-edge arrays can be loaded but are never presented.
        self.assertEqual(len(view._overlay_items), 4)
        self.assertIsNotNone(view._instance_annotation_overlay_item)

        view.set_material_reference_annotations_visible(False)
        self.assertEqual(len(view._overlay_items), 1)
        view.set_boundary_reference_annotations_visible(False)
        self.assertEqual(len(view._overlay_items), 1)
        view.set_instance_annotations_visible(False)
        self.assertEqual(len(view._overlay_items), 0)
        # A draft refresh must not resurrect a hidden instance overlay.
        view.set_instance_annotations(instances)
        self.assertEqual(len(view._overlay_items), 0)
        view.set_material_reference_annotations_visible(True)
        self.assertEqual(len(view._overlay_items), 3)
        view.set_instance_annotations_visible(True)
        self.assertEqual(len(view._overlay_items), 4)
        view.set_boundary_reference_annotations_visible(True)
        self.assertEqual(len(view._overlay_items), 4)
        self.assertFalse(view._boundary_reference_annotations_visible)
        self.assertTrue(np.array_equal(view.reference_mask("other"), other))
        self.assertTrue(
            np.array_equal(view.reference_mask("physical_edge"), physical_edge)
        )
        self.assertTrue(np.array_equal(view.reference_mask("non_edge"), non_edge))
        self.assertTrue(np.array_equal(view.instance_annotations(), instances))

        # Compatibility API changes material display only and leaves labelled
        # seed instances alone; legacy boundaries remain hidden.
        view.set_reference_annotations_visible(False)
        self.assertEqual(len(view._overlay_items), 1)
        view.set_reference_annotations_visible(True)
        self.assertEqual(len(view._overlay_items), 4)
        view.close()

    def test_selected_procedural_instance_reports_geometry_in_node_panel(self) -> None:
        import numpy as np

        from seedvision.segmentation.procedural import ProceduralInstanceResult
        from seedvision.ui.main_window import MainWindow

        labels = np.ones((6, 7), np.uint16)
        procedural = ProceduralInstanceResult(
            labels=labels,
            centres_xy=np.asarray(((3.0, 2.5),), np.float32),
            marker_scores=np.asarray((0.91,), np.float32),
            instance_confidences=np.asarray((0.84,), np.float32),
            occupancy_likelihood=np.full(labels.shape, 255, np.uint8),
            occupancy_mask=np.full(labels.shape, 255, np.uint8),
            boundary_cost=np.zeros(labels.shape, np.uint8),
            centre_likelihood=np.zeros(labels.shape, np.uint8),
            source_shape=labels.shape,
            working_scale=1.0,
            instance_area_px2=np.asarray((42.0,), np.float32),
            instance_maximum_width_px=np.asarray((7.0,), np.float32),
            instance_width_fractions=np.asarray((1.17,), np.float32),
            instance_concavity_fractions=np.asarray((0.12,), np.float32),
            instance_protrusion_fractions=np.asarray((0.04,), np.float32),
            instance_solidities=np.asarray((0.89,), np.float32),
            instance_axis_ratios=np.asarray((1.44,), np.float32),
        )
        window = MainWindow(ROOT)
        window.image_view._analysis_result = SimpleNamespace(
            procedural_instances=procedural,
            crop_offset=(0, 0),
        )
        window._procedural_instance_selected(1)
        self.application.processEvents()
        text = window.pipeline_inspector.procedural_instance_statistics_label.text()
        self.assertEqual(window._selected_pipeline_node, "procedural_instances")
        self.assertIn("maximum width 7.0 px", text)
        self.assertIn("concavity 12.0%", text)
        self.assertIn("solidity 0.890", text)
        self.assertIn("axis ratio 1.440", text)
        window.close()

    def test_image_view_displays_full_size_gamut_without_changing_mask_size(self) -> None:
        import numpy as np

        from PySide6.QtGui import QColor, QImage
        from PySide6.QtTest import QTest

        from seedvision.pipeline import build_default_pipeline
        from seedvision.ui.image_view import ImageView
        from seedvision.visualization import ForegroundColourProfile

        profile = ForegroundColourProfile(
            centre_lab=(152.0, 141.0, 118.0),
            scale_lab=(18.0, 9.0, 8.0),
            bgr_low=(36, 62, 80),
            bgr_high=(211, 218, 224),
            sample_count=1800,
            sample_fraction=0.34,
            component_centres_lab=((152.0, 141.0, 118.0),),
            component_scales_lab=((18.0, 9.0, 8.0),),
            component_weights=(1.0,),
            refinement_iterations=0,
            source="painted",
            source_sample_count=1800,
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "gamut-source.png"
            source = QImage(320, 200, QImage.Format.Format_RGB32)
            source.fill(QColor("white"))
            self.assertTrue(source.save(str(path)))
            corrected = np.full((180, 300, 3), 210, np.uint8)
            result = SimpleNamespace(
                calibration=SimpleNamespace(corrected_bgr=corrected),
                layers=SimpleNamespace(
                    foreground_colour_profile=profile,
                    background_colour_profile=None,
                ),
            )
            view = ImageView()
            succeeded, error = view.load_image(path)
            self.assertTrue(succeeded, error)
            view.set_colour_gamut_parameters(
                "foreground",
                build_default_pipeline().node(
                "background_likelihood"
                ).parameters,
            )
            view.set_overlay_mode("foreground_colour_gamut")
            view.show_analysis(result)

            self.assertEqual(view._displayed_base, "gamut")
            self.assertEqual(
                (view._image_item.pixmap().width(), view._image_item.pixmap().height()),
                (960, 640),
            )
            self.assertEqual(view.image_size, (300, 180))
            self.assertGreater(
                view.dominant_hsv_gamut_value("foreground"), 0.0
            )
            original_key = view._gamut_base_key
            view.set_hsv_gamut_value(0.35)
            QTest.qWait(130)
            self.application.processEvents()
            self.assertNotEqual(view._gamut_base_key, original_key)
            self.assertEqual(view.image_size, (300, 180))
            view.set_overlay_mode("raw_image")
            self.application.processEvents()
            self.assertEqual(view.image_size, (320, 200))
            view.close()

    def test_image_view_displays_every_reference_prototype_in_a_full_pane_collage(self) -> None:
        import numpy as np

        from PySide6.QtGui import QColor, QImage

        from seedvision.ui.image_view import ImageView
        from seedvision.visualization import (
            ReferenceTextureProfile,
            ReferenceTexturePrototype,
        )

        class_names = (
            "background",
            "foreground",
            "other",
            "physical_edge",
            "non_edge",
        )
        prototypes = []
        for class_index, class_name in enumerate(class_names):
            for prototype_index in range(17):
                patch = np.full(
                    (18, 18, 3),
                    (
                        25 + class_index * 35,
                        30 + prototype_index * 7,
                        190 - class_index * 20,
                    ),
                    np.uint8,
                )
                patch.flags.writeable = False
                prototypes.append(
                    ReferenceTexturePrototype(
                        class_name=class_name,
                        patch_bgr=patch,
                        weight=1.0 / 17.0,
                        sample_count=24,
                        centre_xy=(20.0 + prototype_index, 30.0 + class_index),
                        tangent_degrees=(
                            25.0
                            if class_name in {"physical_edge", "non_edge"}
                            else None
                        ),
                    )
                )
        profile = ReferenceTextureProfile(
            prototypes=tuple(prototypes),
            class_sample_counts=tuple((name, 408) for name in class_names),
            material_feature_names=("Lab lightness", "fine darkness noise"),
            edge_feature_names=("edge magnitude", "tangent coherence"),
            working_scale=0.5,
            patch_size_px=18,
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "prototype-source.png"
            source = QImage(320, 200, QImage.Format.Format_RGB32)
            source.fill(QColor("white"))
            self.assertTrue(source.save(str(path)))
            result = SimpleNamespace(
                calibration=SimpleNamespace(
                    corrected_bgr=np.full((180, 300, 3), 210, np.uint8)
                ),
                layers=SimpleNamespace(reference_texture_profile=profile),
            )
            view = ImageView()
            succeeded, error = view.load_image(path)
            self.assertTrue(succeeded, error)
            view.set_overlay_mode("reference_texture_prototypes")
            view.show_analysis(result)

            self.assertEqual(view._displayed_base, "prototype_collage")
            self.assertEqual(view.image_size, (300, 180))
            self.assertGreaterEqual(view._image_item.pixmap().width(), 960)
            self.assertGreater(view._image_item.pixmap().height(), 1200)
            self.assertEqual(len(profile.prototypes), 85)
            view.set_overlay_mode("raw_image")
            self.application.processEvents()
            self.assertEqual(view.image_size, (320, 200))
            view.close()

    def test_image_view_edits_full_resolution_binary_reference_masks(self) -> None:
        import numpy as np

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
            view._update_reference_brush_outline(QPointF(50.0, 50.0))
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
            self.assertFalse(view.reference_mask("background")[50, 50])
            QTest.mouseClick(
                view.viewport(), Qt.MouseButton.RightButton, pos=position
            )
            self.assertEqual(edits[-1][0], "foreground")
            self.assertFalse(edits[-1][1][50, 50])
            view.set_background_exclusion_editing(True)
            QTest.mouseClick(
                view.viewport(), Qt.MouseButton.LeftButton, pos=position
            )
            self.assertEqual(edits[-1][0], "other")
            self.assertTrue(edits[-1][1][50, 50])
            self.assertFalse(view.reference_mask("foreground")[50, 50])
            self.assertFalse(view.reference_mask("background")[50, 50])
            view.set_physical_edge_reference_editing(True)
            view.set_edge_reference_snap(False)
            QTest.mouseClick(
                view.viewport(), Qt.MouseButton.LeftButton, pos=position
            )
            self.assertEqual(edits[-1][0], "physical_edge")
            self.assertTrue(edits[-1][1][50, 50])
            view.set_non_edge_reference_editing(True)
            QTest.mouseClick(
                view.viewport(), Qt.MouseButton.LeftButton, pos=position
            )
            self.assertEqual(edits[-1][0], "non_edge")
            self.assertTrue(edits[-1][1][50, 50])
            self.assertFalse(view.reference_mask("physical_edge")[50, 50])
            edge_strength = np.zeros((100, 100), dtype=np.float32)
            edge_strength[50, 60] = 1.0
            view._full_annotation_evidence = lambda _name: edge_strength
            view.set_physical_edge_reference_editing(True)
            view.set_edge_reference_snap(True)
            view.set_edge_reference_snap_strength(0.5)
            view._update_reference_brush_outline(QPointF(50.0, 50.0))
            self.assertAlmostEqual(
                view._reference_brush_outline_item.rect().center().x(), 55.0
            )
            view.set_edge_reference_snap_strength(1.0)
            view._update_reference_brush_outline(QPointF(50.0, 50.0))
            self.assertAlmostEqual(
                view._reference_brush_outline_item.rect().center().x(), 60.0
            )
            QTest.mouseClick(
                view.viewport(), Qt.MouseButton.LeftButton, pos=position
            )
            self.assertEqual(edits[-1][0], "physical_edge")
            self.assertTrue(edits[-1][1][50, 60])
            self.assertFalse(view.reference_mask("non_edge")[50, 60])
            applied_background = np.zeros((100, 100), dtype=bool)
            applied_background[50, 50] = True
            applied_foreground = np.zeros((100, 100), dtype=bool)
            applied_other = np.zeros((100, 100), dtype=bool)
            for applied in (
                applied_background,
                applied_foreground,
                applied_other,
            ):
                applied.flags.writeable = False
            view.set_reference_masks(
                applied_background,
                applied_foreground,
                applied_other,
                applied_other,
                copy=False,
                normalize_material=False,
            )
            view.set_foreground_point_editing(True)
            QTest.mouseClick(
                view.viewport(), Qt.MouseButton.LeftButton, pos=position
            )
            self.assertTrue(view.reference_mask("foreground")[50, 50])
            self.assertFalse(view.reference_mask("background")[50, 50])
            view.close()

    def test_background_painting_shows_only_exact_perimeter_ring_source(self) -> None:
        import numpy as np

        from PySide6.QtCore import QPointF, Qt
        from PySide6.QtGui import QColor, QImage
        from PySide6.QtWidgets import QGraphicsPathItem, QGraphicsPixmapItem

        from seedvision.ui.image_view import ImageView

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "automatic-background-source.png"
            image = QImage(96, 80, QImage.Format.Format_RGB32)
            image.fill(QColor("white"))
            self.assertTrue(image.save(str(path)))

            source_mask = np.zeros((20, 30), dtype=np.uint8)
            source_mask[2:6, 4:11] = 255
            source_mask[13, 24] = 255
            accepted_ring = np.zeros((80, 96), dtype=np.uint8)
            accepted_ring[4, 48] = 255
            accepted_ring[40, 91] = 255
            painted_mask = np.zeros((80, 96), dtype=bool)
            painted_mask[55:61, 67:73] = True
            result = SimpleNamespace(
                calibration=SimpleNamespace(
                    corrected_bgr=np.full((80, 96, 3), 235, np.uint8)
                ),
                layers=SimpleNamespace(
                    background_reference_source_mask=source_mask
                ),
                crop_offset=(7, 11),
                dish=SimpleNamespace(center_x=48.0, center_y=40.0),
                perimeter_background_band=SimpleNamespace(
                    inner_radius_px=40.0,
                    outer_radius_px=45.0,
                    outside_vessel=True,
                    sample_count=1337,
                    buffer_cm=0.35,
                    thickness_cm=0.50,
                    accepted_sample_mask=accepted_ring,
                ),
            )
            view = ImageView()
            succeeded, error = view.load_image(path)
            self.assertTrue(succeeded, error)
            view.set_overlay_mode("none")
            view.set_reference_masks(painted_mask, None, render=False)
            view.show_analysis(result)

            # The automatic source is a painting aid, not a general overlay.
            self.assertFalse(
                any(
                    item.toolTip().startswith(
                        "Automatic perimeter-matched Background source"
                    )
                    for item in view._overlay_items
                )
            )
            view.set_background_point_editing(True)
            source_items = [
                item
                for item in view._overlay_items
                if isinstance(item, QGraphicsPixmapItem)
                and item.toolTip().startswith(
                    "Automatic perimeter-matched Background source"
                )
            ]
            # A legacy/runtime crop source must not be painted inside the dish.
            self.assertEqual(source_items, [])

            ring_source_items = [
                item
                for item in view._overlay_items
                if isinstance(item, QGraphicsPixmapItem)
                and item.toolTip().startswith(
                    "Colour-filtered pixels retained from the perimeter"
                )
            ]
            self.assertEqual(len(ring_source_items), 1)
            self.assertEqual(ring_source_items[0].pos(), QPointF(0.0, 0.0))
            ring_source = ring_source_items[0].pixmap().toImage()
            self.assertEqual(ring_source.pixelColor(48, 4).alpha(), 92)
            self.assertEqual(ring_source.pixelColor(47, 4).alpha(), 0)
            included = ring_source.pixelColor(48, 4)
            for actual, expected in zip(
                (included.red(), included.green(), included.blue()),
                (65, 217, 255),
            ):
                # QPixmap stores translucent pixels premultiplied; converting
                # back to QColor can round a channel down by one.
                self.assertAlmostEqual(actual, expected, delta=1)

            rings = [
                item
                for item in view._overlay_items
                if isinstance(item, QGraphicsPathItem)
                and item.toolTip().startswith("Initial background sampling band")
            ]
            self.assertEqual(len(rings), 1)
            ring = rings[0]
            self.assertEqual(ring.brush().style(), Qt.BrushStyle.NoBrush)
            self.assertAlmostEqual(ring.path().boundingRect().width(), 90.0)
            self.assertFalse(ring.path().contains(QPointF(48.0, 40.0)))
            self.assertTrue(ring.path().contains(QPointF(90.5, 40.0)))

            view.set_automatic_background_reference_visible(False)
            self.assertFalse(
                any(
                    item.toolTip().startswith((
                        "Automatic perimeter-matched Background source",
                        "Initial background sampling band",
                    ))
                    for item in view._overlay_items
                )
            )
            # The manually painted Background raster stays visible.
            self.assertTrue(
                any(
                    isinstance(item, QGraphicsPixmapItem)
                    for item in view._overlay_items
                )
            )
            view.set_automatic_background_reference_visible(True)
            view.set_foreground_point_editing(True)
            self.assertFalse(
                any(
                    item.toolTip().startswith((
                        "Automatic perimeter-matched Background source",
                        "Initial background sampling band",
                    ))
                    for item in view._overlay_items
                )
            )
            view.close()

    def test_sparse_automatic_background_source_survives_large_view_downsampling(self) -> None:
        import numpy as np

        from PySide6.QtGui import QColor, QImage
        from PySide6.QtWidgets import QGraphicsPixmapItem

        from seedvision.ui.image_view import ImageView

        with tempfile.TemporaryDirectory() as temporary_directory:
            width, height = 4097, 9
            path = Path(temporary_directory) / "wide-background-source.png"
            image = QImage(width, height, QImage.Format.Format_RGB32)
            image.fill(QColor("white"))
            self.assertTrue(image.save(str(path)))
            source_mask = np.zeros((height, width), np.uint8)
            # The former point sampler chose columns 0, 2, 4, ... and silently
            # dropped this valid one-pixel source at odd x=1.
            source_mask[4, 1] = 255
            result = SimpleNamespace(
                calibration=SimpleNamespace(
                    corrected_bgr=np.full((height, width, 3), 235, np.uint8)
                ),
                layers=SimpleNamespace(
                    background_reference_source_mask=None
                ),
                crop_offset=(0, 0),
                dish=SimpleNamespace(center_x=width / 2.0, center_y=height / 2.0),
                perimeter_background_band=SimpleNamespace(
                    inner_radius_px=1.0,
                    outer_radius_px=3.0,
                    outside_vessel=True,
                    sample_count=1,
                    buffer_cm=0.35,
                    thickness_cm=0.50,
                    accepted_sample_mask=source_mask,
                ),
            )
            view = ImageView()
            succeeded, error = view.load_image(path)
            self.assertTrue(succeeded, error)
            view.set_overlay_mode("none")
            view.show_analysis(result)
            view.set_background_point_editing(True)

            source_items = [
                item
                for item in view._overlay_items
                if isinstance(item, QGraphicsPixmapItem)
                and item.toolTip().startswith(
                    "Colour-filtered pixels retained from the perimeter"
                )
            ]
            self.assertEqual(len(source_items), 1)
            rendered = source_items[0].pixmap().toImage()
            self.assertEqual(rendered.width(), 2048)
            self.assertTrue(
                any(
                    rendered.pixelColor(x, y).alpha() > 0
                    for y in range(rendered.height())
                    for x in range(rendered.width())
                )
            )
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

    def test_reference_brush_uses_live_stroke_and_redraws_once_on_release(self) -> None:
        import numpy as np

        from PySide6.QtCore import QPointF, Qt
        from PySide6.QtGui import QColor, QImage
        from PySide6.QtTest import QTest

        from seedvision.ui.image_view import ImageView

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "responsive-reference-brush.png"
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
            view.set_background_point_editing(True)
            renders: list[bool] = []
            view._render_analysis = lambda: renders.append(True)
            start = view.mapFromScene(QPointF(35.0, 55.0))
            end = view.mapFromScene(QPointF(105.0, 55.0))

            QTest.mousePress(view.viewport(), Qt.MouseButton.LeftButton, pos=start)
            self.assertIsNotNone(view._reference_live_stroke_item)
            QTest.mouseMove(view.viewport(), end, delay=1)
            self.assertIsNotNone(view._reference_live_stroke_item)
            self.assertEqual(renders, [])
            QTest.mouseRelease(view.viewport(), Qt.MouseButton.LeftButton, pos=end)

            mask = view.reference_mask("background")
            self.assertIsNotNone(mask)
            self.assertGreater(np.count_nonzero(mask), 500)
            self.assertIsNone(view._reference_live_stroke_item)
            self.assertEqual(renders, [True])
            view.close()

    def test_image_view_applies_assisted_instance_annotation_tools(self) -> None:
        import numpy as np

        from PySide6.QtCore import QPointF
        from PySide6.QtGui import QColor, QImage

        from seedvision.annotation import (
            EdgeTraceOptions,
            ShapeGuidedFillOptions,
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
            view.set_instance_annotation_tool("shape_guided_fill")
            view.set_shape_guided_fill_options(
                ShapeGuidedFillOptions(shape="ellipse"),
                edge_source="magnitude",
            )
            self.assertTrue(view._apply_instance_assisted_tool(QPointF(88, 52)))
            self.assertEqual(int(view._instance_annotations[52, 88]), 2)

            view.set_active_instance_id(3)
            labels = view.instance_annotations()
            labels[76:80, 24:28] = 3
            view.set_instance_annotations(labels)
            view.set_instance_annotation_tool("smart_fill")
            view.set_smart_fill_options(
                SmartFillOptions(
                    maximum_distance_from_cursor_px=14,
                    maximum_added_pixels=2_000,
                )
            )
            self.assertTrue(view._apply_instance_assisted_tool(QPointF(26, 78)))
            self.assertGreater(np.count_nonzero(view._instance_annotations == 3), 200)
            self.assertTrue(any("Edge trace" in status for status in statuses))
            self.assertTrue(any("Shape-guided fill" in status for status in statuses))
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
            ShapeGuidedFillOptions,
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
            view.set_instance_annotation_tool("shape_guided_fill")
            view.set_shape_guided_fill_options(
                ShapeGuidedFillOptions(shape="ellipse"),
                edge_source="magnitude",
            )
            shape_centre = view.mapFromScene(QPointF(112, 58))
            QTest.mouseMove(view.viewport(), view.viewport().rect().topLeft())
            QTest.mouseMove(view.viewport(), shape_centre)
            # Keep the debounced hover target deterministic when a prior Qt
            # view left a platform-level mouse move queued at another scene
            # coordinate. The public event path is still exercised above.
            view._update_reference_brush_outline(QPointF(112, 58))
            QTest.qWait(55)
            view._update_instance_assisted_preview(QPointF(112, 58))
            self.assertIsNotNone(view._instance_shape_guided_region)
            assert view._instance_shape_guided_region is not None
            self.assertIsNotNone(
                view._instance_shape_guided_region.prior_polygon
            )
            self.assertIsNotNone(
                view._instance_shape_guided_region.boundary_polygon
            )
            self.assertGreater(len(view._instance_preview_items), 0)
            QTest.mouseClick(
                view.viewport(), Qt.MouseButton.LeftButton, pos=shape_centre
            )
            self.assertEqual(int(edits[-1][58, 112]), 2)

            view.set_active_instance_id(3)
            view.set_instance_annotation_tool("smart_fill")
            view.set_smart_fill_options(
                SmartFillOptions(
                    maximum_distance_from_cursor_px=14,
                    maximum_added_pixels=2_000,
                )
            )
            fill_centre = view.mapFromScene(QPointF(35, 88))
            QTest.mouseMove(view.viewport(), view.viewport().rect().bottomRight())
            QTest.mouseMove(view.viewport(), fill_centre)
            # Offscreen Qt can suppress a synthetic mouse move when a prior
            # test left the platform cursor at the same global coordinate.
            # Exercise the exact hover handler explicitly as well so the
            # debounce test is independent of process-wide cursor history.
            view._update_reference_brush_outline(QPointF(35, 88))
            QTest.qWait(55)
            # Drain any stale platform move delivered after the explicit
            # handler call, then verify the smart-fill algorithm at the
            # intended scene coordinate. Trace and shape hover behaviour above
            # already cover the event-to-preview path.
            view._update_instance_assisted_preview(QPointF(35, 88))
            self.assertIsNotNone(view._instance_preview_region)
            self.assertGreater(
                view._instance_preview_region.added_count,
                100,
                str(view._instance_preview_point),
            )
            QTest.mouseClick(
                view.viewport(), Qt.MouseButton.LeftButton, pos=fill_centre
            )
            self.assertEqual(int(edits[-1][88, 35]), 3)
            view.close()

    def test_instance_annotation_panel_exposes_advanced_tool_options(self) -> None:
        import numpy as np
        from PySide6.QtCore import QPoint, QPointF, Qt
        from PySide6.QtGui import QColor, QImage, QWheelEvent

        from seedvision.ui.main_window import MainWindow

        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        project_root = Path(temporary.name)
        image_path = project_root / "images" / "seed.png"
        image_path.parent.mkdir(parents=True)
        image = QImage(80, 60, QImage.Format.Format_RGB32)
        image.fill(QColor("white"))
        self.assertTrue(image.save(str(image_path)))
        window = MainWindow(project_root)
        self.assertEqual(window.instance_tool_options_stack.count(), 5)
        self.assertNotIn("shape_snap", window.instance_tool_pages)
        self.assertFalse(hasattr(window, "instance_shape_snap_button"))
        with self.assertRaises(ValueError):
            window.image_view.set_instance_annotation_tool("shape_snap")
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

        window.instance_shape_guided_fill_button.setChecked(True)
        self.application.processEvents()
        self.assertEqual(
            window.image_view._instance_annotation_tool, "shape_guided_fill"
        )
        self.assertIs(
            window.instance_tool_options_stack.currentWidget(),
            window.instance_tool_pages["shape_guided_fill"],
        )
        self.assertAlmostEqual(
            window.shape_fill_outward_half_life_spin.value(), 0.05
        )
        self.assertAlmostEqual(
            window.shape_fill_outward_cutoff_spin.value(), 0.10
        )
        self.assertAlmostEqual(
            window.shape_fill_outward_cutoff_spin.maximum(), 0.50
        )
        window.shape_fill_outward_half_life_spin.setValue(0.14)
        self.assertAlmostEqual(
            window.shape_fill_outward_cutoff_spin.value(), 0.14
        )
        self.assertAlmostEqual(
            window.image_view._shape_guided_fill_options.outward_hard_cutoff_fraction,
            0.14,
        )
        window.shape_fill_outward_cutoff_spin.setValue(0.08)
        self.assertAlmostEqual(
            window.shape_fill_outward_half_life_spin.value(), 0.08
        )
        self.assertAlmostEqual(
            window.image_view._shape_guided_fill_options.outward_penalty_half_life_fraction,
            0.08,
        )
        window.shape_fill_axis_ratio_spin.setValue(2.6)
        window.shape_fill_rotation_spin.setValue(22.5)
        window.shape_fill_preferred_scale_spin.setValue(1.25)
        window.shape_fill_colour_step_spin.setValue(21.0)
        window.shape_fill_barrier_spin.setValue(46)
        window.shape_fill_outward_half_life_spin.setValue(0.06)
        window.shape_fill_outward_cutoff_spin.setValue(0.12)
        self.assertAlmostEqual(
            window.image_view._shape_guided_fill_options.maximum_axis_ratio,
            2.6,
        )
        shape_options = window.image_view._shape_guided_fill_options
        self.assertAlmostEqual(shape_options.preferred_scale, 1.25)
        self.assertAlmostEqual(shape_options.initial_rotation_degrees, 22.5)
        self.assertAlmostEqual(shape_options.colour_tolerance_lab, 21.0)
        self.assertAlmostEqual(shape_options.edge_barrier_threshold, 0.46)
        self.assertAlmostEqual(
            shape_options.outward_penalty_half_life_fraction,
            0.06,
        )
        self.assertAlmostEqual(shape_options.outward_hard_cutoff_fraction, 0.12)
        for removed_control in (
            "shape_fill_tangent_combo",
            "shape_fill_tangent_weight_spin",
            "shape_fill_tunnel_combo",
            "shape_fill_radius_spin",
            "shape_fill_falloff_spin",
            "shape_fill_connectivity_combo",
            "shape_fill_boundary_search_spin",
            "shape_fill_shape_adherence_spin",
        ):
            self.assertFalse(hasattr(window, removed_control), removed_control)
        shape_form = window.instance_tool_pages["shape_guided_fill"].layout()
        self.assertEqual(
            shape_form.labelForField(window.shape_fill_preferred_scale_spin).text(),
            "Oval size preference",
        )
        self.assertEqual(
            shape_form.labelForField(
                window.shape_fill_outward_half_life_spin
            ).text(),
            "Outward soft half-life",
        )
        self.assertEqual(
            shape_form.labelForField(window.shape_fill_outward_cutoff_spin).text(),
            "Outward hard cutoff",
        )
        self.assertEqual(window.image_view._shape_guided_edge_source, "adaptive")
        window.image_view.set_instance_annotation_editing(True)
        before_zoom = window.image_view.transform().m11()
        window.image_view.wheelEvent(
            QWheelEvent(
                QPointF(100.0, 100.0),
                QPointF(100.0, 100.0),
                QPoint(),
                QPoint(0, 120),
                Qt.MouseButton.NoButton,
                Qt.KeyboardModifier.NoModifier,
                Qt.ScrollPhase.NoScrollPhase,
                False,
            )
        )
        self.assertAlmostEqual(window.shape_fill_preferred_scale_spin.value(), 1.30)
        self.assertAlmostEqual(
            window.image_view._shape_guided_fill_options.preferred_scale, 1.30
        )
        self.assertAlmostEqual(window.image_view.transform().m11(), before_zoom)

        window.instance_smart_fill_button.setChecked(True)
        fill_form = window.instance_tool_pages["smart_fill"].layout()
        self.assertEqual(
            fill_form.labelForField(
                window.smart_fill_colour_tolerance_spin
            ).text(),
            "Neighbour colour step",
        )
        self.assertEqual(
            fill_form.labelForField(
                window.smart_fill_click_colour_tolerance_spin
            ).text(),
            "Click-origin colour range",
        )
        self.assertAlmostEqual(
            window.smart_fill_click_colour_tolerance_spin.value(), 72.0
        )
        self.assertIn(
            "initially clicked",
            window.smart_fill_click_colour_tolerance_spin.toolTip(),
        )
        self.assertEqual(
            fill_form.labelForField(window.smart_fill_edge_stop_spin).text(),
            "Edge barrier threshold",
        )
        self.assertEqual(
            fill_form.labelForField(window.smart_fill_radius_spin).text(),
            "Maximum distance from cursor",
        )
        self.assertAlmostEqual(window.smart_fill_radius_spin.value(), 0.60)
        self.assertIn(
            "not the seed's side-to-side width",
            window.smart_fill_radius_spin.toolTip(),
        )
        self.assertEqual(
            fill_form.labelForField(window.smart_fill_falloff_spin).text(),
            "Fall-off half-life",
        )
        self.assertAlmostEqual(window.smart_fill_falloff_spin.value(), 0.40)
        self.assertIn(
            "p(d) = 2^(-d / h)",
            window.smart_fill_falloff_spin.toolTip(),
        )
        diameter = float(
            getattr(window.image_view._analysis_result, "estimated_seed_diameter_px", 100.0)
        )
        window.smart_fill_falloff_spin.setValue(0.55)
        self.assertAlmostEqual(
            window.image_view._smart_fill_options.falloff_half_life_px,
            np.clip(diameter * 0.55, 1.0, 4096.0),
        )
        self.assertIn(
            "touching candidate pixel",
            window.smart_fill_colour_tolerance_spin.toolTip(),
        )
        window.smart_fill_click_colour_tolerance_spin.setValue(37.5)
        self.assertAlmostEqual(
            window.image_view._smart_fill_options.click_colour_tolerance_lab,
            37.5,
        )
        self.assertIn(
            "Lower values stop at weaker edges",
            window.smart_fill_edge_stop_spin.toolTip(),
        )
        self.assertFalse(hasattr(window, "smart_fill_tunnel_combo"))
        self.assertFalse(hasattr(window, "smart_fill_connectivity_combo"))
        window.smart_fill_gap_sealing_spin.setValue(0.035)
        self.assertGreater(window.image_view._smart_fill_options.edge_gap_sealing_px, 0)
        self.assertAlmostEqual(window.image_view._smart_fill_options.tunnel_strength, 0.0)
        self.assertEqual(window.image_view._smart_fill_options.connectivity, 4)
        window.close()

    def test_new_seed_preserves_the_selected_instance_tool(self) -> None:
        from PySide6.QtGui import QColor, QImage

        from seedvision.ui.main_window import MainWindow

        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        project_root = Path(temporary.name)
        image_path = project_root / "images" / "seed.png"
        image_path.parent.mkdir(parents=True)
        image = QImage(80, 60, QImage.Format.Format_RGB32)
        image.fill(QColor("white"))
        self.assertTrue(image.save(str(image_path)))
        window = MainWindow(project_root)
        if window._current_image_key() is None:
            window.close()
            self.skipTest("No workspace image is present")
        window.new_instance_button.setEnabled(True)
        window._instance_ids = lambda _annotations: (7,)
        window.smart_fill_colour_tolerance_spin.setValue(17.25)
        smart_tolerance = window.smart_fill_colour_tolerance_spin.value()
        window.smart_fill_click_colour_tolerance_spin.setValue(41.5)
        smart_click_tolerance = (
            window.smart_fill_click_colour_tolerance_spin.value()
        )
        for button, expected_tool in (
            (window.instance_paint_mode_button, "brush"),
            (window.instance_edge_trace_button, "edge_trace"),
            (window.instance_shape_guided_fill_button, "shape_guided_fill"),
            (window.instance_smart_fill_button, "smart_fill"),
            (window.instance_eraser_button, "eraser"),
        ):
            with self.subTest(tool=expected_tool):
                button.setChecked(True)
                window.instance_id_spin.setValue(1)
                self.application.processEvents()
                selected_page = window.instance_tool_options_stack.currentWidget()
                window.new_instance_button.setEnabled(True)

                window.new_instance_button.click()
                self.application.processEvents()

                self.assertTrue(button.isChecked())
                self.assertEqual(window.instance_id_spin.value(), 8)
                self.assertEqual(window._current_instance_tool(), expected_tool)
                self.assertEqual(
                    window.image_view._instance_annotation_tool, expected_tool
                )
                self.assertIs(
                    window.instance_tool_options_stack.currentWidget(), selected_page
                )
                if expected_tool == "smart_fill":
                    self.assertAlmostEqual(
                        window.smart_fill_colour_tolerance_spin.value(), smart_tolerance
                    )
                    self.assertAlmostEqual(
                        window.smart_fill_click_colour_tolerance_spin.value(),
                        smart_click_tolerance,
                    )
        window.close()

    def test_reference_masks_run_analysis_only_after_explicit_apply(self) -> None:
        import numpy as np

        from seedvision.pipeline import NodeStatus
        from seedvision.ui.main_window import MainWindow

        window = MainWindow(ROOT)
        # This controller test uses the real repository fixtures. Persistence is
        # covered against an isolated project root in test_reference_persistence.
        window._persist_applied_reference_regions = (
            lambda **_kwargs: ROOT / "test-reference-autosave.npz"
        )
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
        self.assertNotIn(key, window._draft_foreground_reference_masks)
        self.assertFalse(
            window._applied_foreground_reference_masks[key].flags.writeable
        )
        self.assertTrue(
            np.array_equal(
                window.image_view.reference_mask("foreground", copy=False),
                window._applied_foreground_reference_masks[key],
            )
        )
        self.assertIs(
            window.pipeline.node("project").status,
            NodeStatus.COMPLETE,
        )
        self.assertIn(
            "refined_background_likelihood", window._cache_dirty_nodes[key]
        )

        window._ensure_reference_draft("foreground")
        self.assertTrue(
            window._draft_foreground_reference_masks[key].flags.writeable
        )
        window._clear_foreground_points()
        self.assertEqual(analyses, [True])
        self.assertIn(key, window._reference_masks_dirty)
        self.assertIn(key, window._applied_foreground_reference_masks)
        window._apply_reference_masks()
        self.assertEqual(analyses, [True, True])
        self.assertNotIn(key, window._applied_foreground_reference_masks)
        other = np.zeros((height, width), dtype=bool)
        other[5:12, 7:14] = True
        window._reference_mask_edited("other", other)
        window._apply_reference_masks()
        self.assertEqual(analyses, [True, True, True])
        self.assertTrue(
            np.array_equal(
                window._applied_background_exclusion_masks[key],
                other,
            )
        )
        self.assertTrue(
            np.array_equal(
                window._applied_foreground_exclusion_masks[key],
                other,
            )
        )
        self.assertFalse(
            np.any(
                window._applied_background_exclusion_masks[key]
                & window._applied_foreground_reference_masks.get(
                    key, np.zeros_like(other)
                )
            )
        )

        physical = np.zeros((height, width), dtype=bool)
        non_edge = np.zeros((height, width), dtype=bool)
        physical[70:74, 80:95] = True
        non_edge[71:73, 85:90] = True
        window._reference_mask_edited("physical_edge", physical)
        window._reference_mask_edited("non_edge", non_edge)
        window._apply_reference_masks()
        # Retired boundary-class edit signals are ignored. Boundary learning is
        # driven solely by applied seed-instance labels.
        self.assertEqual(analyses, [True, True, True])
        self.assertNotIn(key, window._draft_physical_edge_reference_masks)
        self.assertNotIn(key, window._draft_non_edge_reference_masks)
        self.assertNotIn(key, window._applied_physical_edge_reference_masks)
        self.assertNotIn(key, window._applied_non_edge_reference_masks)
        self.assertIs(
            window.pipeline.node("project").status,
            NodeStatus.COMPLETE,
        )
        window.close()

    def test_instance_annotations_apply_separately_from_reference_masks(self) -> None:
        import numpy as np

        from seedvision.ui.main_window import MainWindow

        window = MainWindow(ROOT)
        # Do not let this controller test overwrite a developer's local archive.
        window._persist_applied_reference_regions = (
            lambda **_kwargs: ROOT / "test-reference-autosave.npz"
        )
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
        # Instance-derived reference textures/edges must refresh even when no
        # instance decoder itself is enabled.
        window.pipeline.node("instance_masks").enabled = False
        window.pipeline.node("procedural_instances").enabled = False
        window.pipeline.node("unet_instances").enabled = False

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
        self.assertNotIn(key, window._draft_instance_annotations)
        self.assertFalse(window._applied_instance_annotations[key].flags.writeable)
        self.assertIs(
            window.image_view._instance_annotations,
            window._applied_instance_annotations[key],
        )
        self.assertIn(
            "instance_masks", window._cache_dirty_nodes.get(key, set())
        )
        self.assertIn(
            "procedural_instances", window._cache_dirty_nodes.get(key, set())
        )
        self.assertIn(
            "reference_texture_prototypes",
            window._cache_dirty_nodes.get(key, set()),
        )
        self.assertIn(
            "reference_edge_probability",
            window._cache_dirty_nodes.get(key, set()),
        )
        window.close()

    def test_reference_seed_scale_overlay_boldly_labels_actual_seed_above(self) -> None:
        from PySide6.QtWidgets import (
            QGraphicsEllipseItem,
            QGraphicsRectItem,
            QGraphicsTextItem,
        )

        from seedvision.ui.image_view import ImageView

        view = ImageView()
        view.set_overlay_opacity(0.43)
        view._render_seed_scale(
            SimpleNamespace(
                reference_roi=(100, 150, 500, 450),
                estimated_seed_diameter_px=80.0,
                reference_seed_bounds=((210.0, 205.0, 288.0, 276.0),),
            )
        )

        circles = [
            item
            for item in view._overlay_items
            if isinstance(item, QGraphicsEllipseItem)
        ]
        labels = [
            item
            for item in view._overlay_items
            if isinstance(item, QGraphicsTextItem)
            and item.toPlainText() == "80 px"
        ]
        self.assertEqual(len(circles), 2)
        self.assertEqual(len(labels), 1)
        circle = next(
            item for item in circles if item.pen().color().name() == "#ffd84a"
        )
        halo = next(
            item for item in circles if item.pen().color().name() == "#17202a"
        )
        label = labels[0]
        self.assertTrue(circle.pen().isCosmetic())
        self.assertGreaterEqual(circle.pen().widthF(), 3.0)
        self.assertAlmostEqual(circle.opacity(), 0.43)
        self.assertAlmostEqual(label.opacity(), 0.43)
        self.assertGreater(halo.pen().widthF(), circle.pen().widthF())
        label_backgrounds = [
            item
            for item in view._overlay_items
            if isinstance(item, QGraphicsRectItem)
            and item.toolTip().startswith("Contrast background")
        ]
        self.assertEqual(len(label_backgrounds), 1)
        self.assertEqual(
            label_backgrounds[0].brush().color().name(), "#17202a"
        )
        self.assertLessEqual(
            label.sceneBoundingRect().bottom(), circle.sceneBoundingRect().top()
        )
        self.assertTrue(label.font().bold())
        self.assertAlmostEqual(circle.rect().left(), 210.0)
        self.assertAlmostEqual(circle.rect().top(), 205.0)
        self.assertAlmostEqual(
            label.sceneBoundingRect().center().x(),
            circle.sceneBoundingRect().center().x(),
            delta=1.0,
        )
        self.assertIn("80.0 pixels", label.toolTip())
        view.close()

    def test_calibration_nodes_render_references_and_metric_scale(self) -> None:
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import (
            QGraphicsEllipseItem,
            QGraphicsLineItem,
            QGraphicsPathItem,
            QGraphicsPixmapItem,
            QGraphicsPolygonItem,
            QGraphicsRectItem,
            QGraphicsTextItem,
        )
        import cv2
        import numpy as np

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
        supported_ticks = [
            item
            for item in view._overlay_items
            if isinstance(item, QGraphicsLineItem)
            and item.pen().color().name() == "#42dff5"
        ]
        self.assertEqual(
            len(supported_ticks), result.calibration.ruler.matched_tick_count
        )
        self.assertFalse(
            any(
                isinstance(item, QGraphicsTextItem)
                and item.toPlainText() == "5 cm"
                for item in view._overlay_items
            )
        )
        view.set_overlay_mode("ruler_evidence")
        evidence_colours = {
            item.pen().color().name()
            for item in view._overlay_items
            if isinstance(item, (QGraphicsLineItem, QGraphicsPolygonItem))
        }
        self.assertIn("#28d65f", evidence_colours)
        self.assertIn("#ff3038", evidence_colours)
        self.assertEqual(
            sum(
                isinstance(item, QGraphicsLineItem)
                and item.pen().color().name() == "#ff3038"
                for item in view._overlay_items
            ),
            len(result.calibration.ruler.metric_tick_points),
        )
        self.assertTrue(
            any(
                isinstance(item, QGraphicsTextItem)
                and "Cross-scale error:" in item.toPlainText()
                for item in view._overlay_items
            )
        )
        summary_plates = [
            item
            for item in view._overlay_items
            if isinstance(item, QGraphicsRectItem)
            and item.toolTip().startswith("Contrast plate for the ruler evidence")
        ]
        self.assertEqual(len(summary_plates), 1)
        self.assertLess(summary_plates[0].brush().color().value(), 40)
        imperial_lines = [
            item
            for item in view._overlay_items
            if isinstance(item, QGraphicsLineItem)
            and item.pen().color().name() == "#ff961f"
        ]
        for item, (outer, inner) in zip(
            imperial_lines,
            result.calibration.ruler.imperial_tick_segments,
            strict=True,
        ):
            displayed_length = item.line().length()
            measured_length = float(np.hypot(
                float(inner[0]) - float(outer[0]),
                float(inner[1]) - float(outer[1]),
            ))
            self.assertGreaterEqual(displayed_length, measured_length + 1.9)
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
        layout_widths = sorted(item.rect().width() for item in dish_edges)
        expected_widths = sorted(
            (result.dish.inner_axes[0] * 2.0, result.dish.outer_axes[0] * 2.0)
        )
        for actual, expected in zip(layout_widths, expected_widths, strict=True):
            self.assertAlmostEqual(actual, expected, delta=1.0)
        view.set_overlay_mode("seed_scale_estimation")
        scale_circles = [
            item
            for item in view._overlay_items
            if isinstance(item, QGraphicsEllipseItem)
            and item.pen().color().name() == "#ffd84a"
        ]
        expected_seed_count = max(1, len(result.reference_seed_bounds))
        self.assertEqual(len(scale_circles), expected_seed_count)
        self.assertTrue(
            all(circle.pen().widthF() >= 3.0 for circle in scale_circles)
        )
        expected_diameter_texts = {
            f"{value:.1f}".rstrip("0").rstrip(".") + " px"
            for value in result.reference_seed_diameters_px
        } or {
            f"{result.estimated_seed_diameter_px:.1f}".rstrip("0").rstrip(".")
            + " px"
        }
        scale_labels = [
            item
            for item in view._overlay_items
            if isinstance(item, QGraphicsTextItem)
            and item.toPlainText() in expected_diameter_texts
        ]
        self.assertEqual(len(scale_labels), expected_seed_count)
        self.assertLessEqual(
            scale_labels[0].sceneBoundingRect().bottom(),
            scale_circles[0].sceneBoundingRect().top(),
        )
        self.assertTrue(
            any(
                isinstance(item, QGraphicsTextItem)
                and "Final seed diameter:" in item.toPlainText()
                for item in view._overlay_items
            )
        )
        view.set_overlay_mode("perimeter_background_reference")
        perimeter_bands = [
            item
            for item in view._overlay_items
            if isinstance(item, QGraphicsPathItem)
        ]
        self.assertEqual(len(perimeter_bands), 1)
        self.assertNotEqual(
            perimeter_bands[0].brush().style(), Qt.BrushStyle.NoBrush
        )
        self.assertIn("0.35 cm buffer", perimeter_bands[0].toolTip())
        self.assertIn("0.50 cm thickness", perimeter_bands[0].toolTip())
        lab = np.clip(
            np.rint(np.asarray(result.perimeter_background_lab, dtype=np.float32)),
            0,
            255,
        ).astype(np.uint8)[None, None]
        expected_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)[0, 0]
        expected_hex = "#{:02X}{:02X}{:02X}".format(
            int(expected_bgr[2]), int(expected_bgr[1]), int(expected_bgr[0])
        )
        starting_colour_items = [
            item
            for item in view._overlay_items
            if item.toolTip().startswith(
                "Selected median starting background colour"
            )
        ]
        self.assertEqual(len(starting_colour_items), 3)
        swatches = [
            item
            for item in starting_colour_items
            if isinstance(item, QGraphicsRectItem)
            and item.brush().color().name().upper() == expected_hex
        ]
        self.assertEqual(len(swatches), 1)
        self.assertTrue(
            any(
                isinstance(item, QGraphicsTextItem)
                and "Median starting background" in item.toPlainText()
                and expected_hex in item.toPlainText()
                for item in starting_colour_items
            )
        )
        view.set_overlay_opacity(0.0)
        self.assertEqual(swatches[0].opacity(), 1.0)
        self.assertEqual(perimeter_bands[0].opacity(), 0.0)
        view.set_overlay_opacity(0.68)
        view.set_overlay_mode("proposals")
        proposal_diameters = {
            round(item.rect().width())
            for item in view._overlay_items
            if isinstance(item, QGraphicsEllipseItem)
        }
        self.assertIn(round(result.dish.outer_axes[0] * 2.0), proposal_diameters)
        self.assertNotIn(round(result.dish.inner_axes[0] * 2.0), proposal_diameters)
        view.set_overlay_mode("background_likelihood")
        sampling_bands = [
            item
            for item in view._overlay_items
            if isinstance(item, QGraphicsPathItem)
        ]
        self.assertEqual(len(sampling_bands), 1)
        self.assertIn("Initial background sampling band", sampling_bands[0].toolTip())
        self.assertEqual(
            sampling_bands[0].brush().style(), Qt.BrushStyle.NoBrush
        )
        background_rasters = [
            item
            for item in view._overlay_items
            if isinstance(item, QGraphicsPixmapItem)
        ]
        self.assertEqual(len(background_rasters), 2)
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
            "ruler_detection",
            "deskew_colour",
        ):
            self.assertEqual(window.pipeline.node(node_id).status, NodeStatus.COMPLETE)
        self.assertAlmostEqual(
            window.pipeline.node("edge_gradients").calculation_seconds,
            result.node_timings_seconds["edge_gradients"]
            + result.node_timings_seconds["edge_ridges"],
        )
        for node_id in (
            "illumination_decomposition",
            "image_quality",
            "surface_darkness_gradients",
            "pattern_decomposition",
            "colour_probabilities",
        ):
            self.assertEqual(window.pipeline.node(node_id).status, NodeStatus.COMPLETE)
        for node_id in (
            "boundary_normals",
            "ellipse_likelihood",
            "lightening_gradient_ceiling",
            "darkening_gradient_ceiling",
            "calibration_residuals",
        ):
            self.assertEqual(window.pipeline.node(node_id).status, NodeStatus.BYPASSED)
        for node_id in (
            "perimeter_background_reference",
        ):
            self.assertEqual(window.pipeline.node(node_id).status, NodeStatus.COMPLETE)
        self.assertEqual(
            window.pipeline.node("background_likelihood").status,
            NodeStatus.WARNING,
        )
        self.assertEqual(
            window.pipeline.node("refined_background_likelihood").status,
            NodeStatus.WARNING,
        )
        self.assertIn(
            "0.35 cm buffer",
            window.pipeline.node("perimeter_background_reference").status_detail,
        )
        self.assertIn(
            "intentionally zero",
            window.pipeline.node("background_likelihood").status_detail,
        )
        self.assertIn("circle_candidates", window.pipeline.unused_nodes)
        self.assertEqual(
            window.pipeline.node("circle_candidates").status_detail,
            "Disabled by default while this branch is under review",
        )
        window.close()


if __name__ == "__main__":
    unittest.main()
