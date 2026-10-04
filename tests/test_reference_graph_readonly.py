from __future__ import annotations

import os
import unittest


class ReferenceGraphReadOnlyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        try:
            from PySide6.QtWidgets import QApplication
        except ImportError as error:
            raise unittest.SkipTest(f"PySide6 unavailable: {error}")
        cls.application = QApplication.instance() or QApplication([])

    @staticmethod
    def _analytic_snapshot(graph):
        return (
            graph.revision,
            tuple(graph.connections),
            tuple(
                (
                    node_id,
                    node.enabled,
                    tuple(sorted(node.parameters.items())),
                    node.status,
                    node.status_detail,
                )
                for node_id, node in graph.nodes.items()
            ),
            tuple(graph.unused_nodes),
        )

    @staticmethod
    def _connected_pair_graph():
        from copy import deepcopy

        from seedvision.pipeline import PipelineGraph, build_default_pipeline

        full_graph = build_default_pipeline()
        connection = full_graph.connections[0]
        graph = PipelineGraph(
            [
                deepcopy(full_graph.node(connection.source)),
                deepcopy(full_graph.node(connection.target)),
            ],
            [connection],
        )
        return graph, graph.connections[0]

    def _drag_input_to_empty_space(self, canvas, connection) -> None:
        from PySide6.QtCore import QPoint, Qt
        from PySide6.QtTest import QTest

        canvas.resize(700, 450)
        canvas.show()
        self.application.processEvents()
        canvas.fit_graph()
        self.application.processEvents()
        target = canvas.node_items[connection.target]
        start = canvas.mapFromScene(target.input_anchor(connection.target_port))
        end = canvas.viewport().rect().bottomRight() - QPoint(12, 12)
        self.assertTrue(canvas.viewport().rect().contains(start), start)
        self.assertTrue(canvas.viewport().rect().contains(end), end)
        self.assertIsNotNone(canvas._port_at(canvas.mapToScene(start)))
        self.assertIsNone(canvas._port_at(canvas.mapToScene(end)))
        QTest.mousePress(canvas.viewport(), Qt.MouseButton.LeftButton, pos=start)
        QTest.mouseMove(canvas.viewport(), end, 25)
        QTest.mouseRelease(canvas.viewport(), Qt.MouseButton.LeftButton, pos=end)
        self.application.processEvents()

    def test_canvas_blocks_direct_and_real_qt_edit_paths(self) -> None:
        from PySide6.QtCore import QPoint, Qt
        from PySide6.QtGui import QContextMenuEvent
        from PySide6.QtTest import QTest
        from PySide6.QtWidgets import QLabel

        from seedvision.pipeline import build_default_pipeline
        from seedvision.ui.pipeline_canvas import PipelineCanvas

        graph = build_default_pipeline()
        graph.restore_unused_node("circle_candidates")
        canvas = PipelineCanvas(graph, read_only=True)
        canvas.resize(840, 620)
        canvas.show()
        self.application.processEvents()
        self.assertTrue(canvas.read_only)
        with self.assertRaises(AttributeError):
            canvas.read_only = False

        for button in (canvas.move_to_unused_button, canvas.unused_nodes_button):
            self.assertTrue(button.isHidden())
            self.assertEqual(button.toolTip(), "")
        canvas._populate_tools_menu()
        tool_actions = {action.text() for action in canvas.tools_menu.actions()}
        self.assertIn("Auto arrange", tool_actions)
        self.assertIn("Bundle cables", tool_actions)
        self.assertIn("Route around nodes", tool_actions)
        self.assertNotIn("Move to unused", tool_actions)

        node = canvas.node_items["ruler_detection"]
        self.assertIsInstance(node._inline_editors["ruler_length_mm"], QLabel)
        self.assertEqual(
            node._inline_editors["ruler_length_mm"].text(),
            str(graph.node("ruler_detection").parameters["ruler_length_mm"]),
        )
        self.assertIn("read-only", node.toolTip())
        self.assertNotIn("Drag between sockets", node.toolTip())

        graph_facts = self._analytic_snapshot(graph)
        parameter_events: list[tuple[str, str, object]] = []
        connection_events: list[object] = []
        run_events: list[str] = []
        canvas.parameter_changed.connect(
            lambda *args: parameter_events.append(args)
        )
        canvas.connections_changed.connect(connection_events.append)
        canvas.run_to_node_requested.connect(run_events.append)

        connection = graph.connections[0]
        canvas._disconnect_connection(connection)
        canvas._finish_connection_drag(QPoint(10, 10))
        canvas._inline_parameter_changed("ruler_detection", "ruler_length_mm", 99.0)
        canvas._restore_unused_node("calibration_residuals")
        canvas.select_node("circle_candidates")
        canvas._shelve_selected_optional_node()
        canvas._request_run_to_node("ruler_detection")
        edge = canvas.edge_items[0]
        node_context = QContextMenuEvent(
            QContextMenuEvent.Reason.Mouse, QPoint(5, 5), QPoint(5, 5)
        )
        canvas.node_items["ruler_detection"].contextMenuEvent(node_context)
        edge_context = QContextMenuEvent(
            QContextMenuEvent.Reason.Mouse, QPoint(5, 5), QPoint(5, 5)
        )
        edge.contextMenuEvent(edge_context)

        edge.setSelected(True)
        canvas.setFocus()
        QTest.keyClick(canvas, Qt.Key.Key_Delete)
        QTest.keyClick(canvas, Qt.Key.Key_Backspace)

        self.assertEqual(self._analytic_snapshot(graph), graph_facts)
        self.assertEqual(parameter_events, [])
        self.assertEqual(connection_events, [])
        self.assertEqual(run_events, [])
        self.assertIsNone(canvas._connection_drag)
        self.assertTrue(node_context.isAccepted())
        self.assertTrue(edge_context.isAccepted())
        self.assertTrue(edge.toolTip().endswith("Reference graph is read-only."))
        canvas.close()

    def test_visible_input_to_empty_drop_disconnects_editable_but_not_reference(self) -> None:
        from seedvision.segmentation.baseline import PipelineAnalysisCache
        from seedvision.ui.pipeline_canvas import PipelineCanvas

        editable_graph, editable_connection = self._connected_pair_graph()
        editable_canvas = PipelineCanvas(editable_graph)
        editable_events: list[object] = []
        editable_canvas.connections_changed.connect(editable_events.append)
        editable_cache = PipelineAnalysisCache(values={"raw.image": object()})
        editable_cache_values = editable_cache.values
        editable_canvas.connections_changed.connect(
            lambda *_: editable_cache.values.clear()
        )
        editable_revision = editable_graph.revision
        self._drag_input_to_empty_space(editable_canvas, editable_connection)
        self.assertNotIn(editable_connection, editable_graph.connections)
        self.assertEqual(editable_graph.revision, editable_revision + 1)
        self.assertEqual(len(editable_events), 1)
        self.assertIs(editable_cache.values, editable_cache_values)
        self.assertEqual(editable_cache.values, {})
        editable_canvas.close()

        reference_graph, reference_connection = self._connected_pair_graph()
        reference_canvas = PipelineCanvas(reference_graph, read_only=True)
        reference_cache = PipelineAnalysisCache(values={"raw.image": object()})
        reference_cache_values = reference_cache.values
        reference_marker = reference_cache.values["raw.image"]
        reference_events: list[object] = []
        reference_canvas.connections_changed.connect(reference_events.append)
        reference_canvas.connections_changed.connect(
            lambda *_: reference_cache.values.clear()
        )
        before = self._analytic_snapshot(reference_graph)
        self._drag_input_to_empty_space(reference_canvas, reference_connection)
        self.assertEqual(self._analytic_snapshot(reference_graph), before)
        self.assertIs(reference_cache.values, reference_cache_values)
        self.assertIs(reference_cache.values["raw.image"], reference_marker)
        self.assertEqual(reference_events, [])
        self.assertIsNone(reference_canvas._connection_drag)
        reference_canvas.close()

    def test_layout_and_routing_remain_display_only(self) -> None:
        from seedvision.pipeline import build_default_pipeline
        from seedvision.ui.pipeline_canvas import PipelineCanvas

        graph = build_default_pipeline()
        canvas = PipelineCanvas(graph, read_only=True)
        initial_revision = graph.revision
        initial_parameters = {
            node_id: tuple(sorted(node.parameters.items()))
            for node_id, node in graph.nodes.items()
        }
        initial_status = {
            node_id: (node.status, node.status_detail)
            for node_id, node in graph.nodes.items()
        }
        node_id = next(iter(canvas.node_items))
        item = canvas.node_items[node_id]
        item.setPos(item.pos().x() + 311, item.pos().y() + 207)
        moved_position = item.pos()
        layouts: list[bool] = []
        presentations: list[bool] = []
        canvas.layout_changed.connect(lambda: layouts.append(True))
        canvas.presentation_changed.connect(lambda: presentations.append(True))
        canvas.auto_arrange()
        canvas.set_cable_bundling_enabled(True)
        canvas.set_obstacle_routing_enabled(True)

        self.assertNotEqual(canvas.node_items[node_id].pos(), moved_position)
        self.assertTrue(layouts)
        self.assertEqual(len(presentations), 2)
        self.assertEqual(graph.revision, initial_revision)
        self.assertEqual(
            {
                key: tuple(sorted(node.parameters.items()))
                for key, node in graph.nodes.items()
            },
            initial_parameters,
        )
        self.assertEqual(
            {
                key: (node.status, node.status_detail)
                for key, node in graph.nodes.items()
            },
            initial_status,
        )
        canvas.close()

    def test_inspector_presents_facts_and_suppresses_all_native_actions(self) -> None:
        from PySide6.QtWidgets import QLabel

        from seedvision.pipeline import build_default_pipeline
        from seedvision.ui.pipeline_inspector import PipelineInspector

        graph = build_default_pipeline()
        inspector = PipelineInspector(read_only=True)
        self.assertTrue(inspector.read_only)
        with self.assertRaises(AttributeError):
            inspector.read_only = False
        node = graph.node("procedural_instances")
        inspector.set_node(node)
        inspector.enable_unified_optimization()
        inspector.set_optimization_context(ready=True, busy=False)
        graph_facts = self._analytic_snapshot(graph)

        self.assertTrue(inspector.enabled_checkbox.isHidden())
        self.assertFalse(inspector.enabled_fact_label.isHidden())
        self.assertIn("enabled", inspector.enabled_fact_label.text())
        self.assertTrue(inspector.reset_parameters_button.isHidden())
        self.assertTrue(inspector.procedural_fit_container.isHidden())
        self.assertTrue(inspector.procedural_centres_container.isHidden())
        self.assertTrue(inspector.optimization_container.isHidden())
        self.assertTrue(inspector.reference_edge_fit_container.isHidden())
        self.assertTrue(inspector._parameter_widgets)
        self.assertTrue(
            all(isinstance(widget, QLabel) for widget in inspector._parameter_widgets)
        )
        # Hidden native-specific widgets are not exposed as editable controls.
        self.assertFalse(inspector.procedural_fit_button.isVisible())
        self.assertFalse(inspector.procedural_centres_edit_button.isVisible())

        inspector.set_node(graph.node("background_likelihood"))
        inspector.set_analysis_result(object())
        self.assertTrue(inspector.foreground_start_heading.isHidden())
        self.assertTrue(inspector.foreground_start_label.isHidden())
        self.assertEqual(inspector.foreground_start_heading.text(), "")
        self.assertEqual(inspector.foreground_start_label.text(), "")
        self.assertEqual(inspector.foreground_start_heading.toolTip(), "")
        self.assertEqual(inspector.foreground_start_label.toolTip(), "")
        inspector.set_overlay_options((("Native overlay", "native"),), "native")
        self.assertTrue(inspector.overlay_combo.isHidden())
        self.assertEqual(inspector.overlay_combo.count(), 0)
        inspector.set_node(graph.node("project"))
        inspector.set_project_summary("Native project summary")
        self.assertTrue(inspector.project_info_container.isHidden())
        self.assertNotIn("Native project summary", inspector.project_info_label.text())
        inspector.set_node(node)

        parameter_events: list[tuple[str, str, object]] = []
        enabled_events: list[tuple[str, bool]] = []
        reset_events: list[str] = []
        action_events: list[tuple[str, str, object]] = []
        inspector.parameter_changed.connect(lambda *args: parameter_events.append(args))
        inspector.enabled_changed.connect(lambda *args: enabled_events.append(args))
        inspector.parameters_reset.connect(reset_events.append)
        inspector.node_action_requested.connect(lambda *args: action_events.append(args))

        inspector._enabled_toggled(False)
        inspector._parameter_edited("reference_texture_weight", 0.0)
        inspector._reset_parameters()
        inspector._request_optimization()
        inspector._procedural_centres_mode_changed(0)
        inspector._reference_edge_fit_requested()
        inspector._procedural_centres_edit_toggled(True)
        inspector._procedural_centres_undo_requested()
        inspector._procedural_centres_reset_requested()
        inspector._procedural_fit_coverage_changed(True)
        inspector._procedural_fit_requested()
        inspector.procedural_fit_button.click()
        inspector.procedural_centres_edit_button.click()
        inspector.procedural_centres_undo_button.click()
        inspector.procedural_centres_reset_button.click()
        self.assertEqual(parameter_events, [])
        self.assertEqual(enabled_events, [])
        self.assertEqual(reset_events, [])
        self.assertEqual(action_events, [])

        # Identifier collisions retain readable facts while hiding every native
        # action block in the reference view.
        inspector.set_node(graph.node("reference_edge_probability"))
        self.assertTrue(inspector.reference_edge_fit_container.isHidden())
        inspector._reference_edge_fit_requested()
        inspector.reference_edge_fit_button.click()
        self.assertEqual(action_events, [])
        inspector.set_node(graph.node("project"))
        self.assertTrue(inspector.procedural_centres_container.isHidden())
        self.assertEqual(self._analytic_snapshot(graph), graph_facts)
        inspector.close()

    def test_native_canvas_and_inspector_default_to_editable(self) -> None:
        from PySide6.QtWidgets import QDoubleSpinBox

        from seedvision.pipeline import build_default_pipeline
        from seedvision.ui.pipeline_canvas import PipelineCanvas
        from seedvision.ui.pipeline_inspector import PipelineInspector

        graph = build_default_pipeline()
        canvas = PipelineCanvas(graph)
        self.assertFalse(canvas.read_only)
        run_events: list[str] = []
        canvas.run_to_node_requested.connect(run_events.append)
        canvas._request_run_to_node("ruler_detection")
        self.assertEqual(run_events, ["ruler_detection"])
        canvas.close()

        inspector = PipelineInspector()
        self.assertFalse(inspector.read_only)
        inspector.set_node(graph.node("background_likelihood"))
        self.assertFalse(inspector.foreground_start_heading.isHidden())
        self.assertIn("Run the foreground node", inspector.foreground_start_label.text())
        inspector.set_node(graph.node("surface_darkness_gradients"))
        self.assertTrue(
            any(
                isinstance(widget, QDoubleSpinBox)
                for widget in inspector._parameter_widgets
            )
        )
        parameter_events: list[tuple[str, str, object]] = []
        inspector.parameter_changed.connect(lambda *args: parameter_events.append(args))
        inspector._parameter_edited("surface_gradient_blur_sigma", 3.0)
        self.assertEqual(
            parameter_events,
            [("surface_darkness_gradients", "surface_gradient_blur_sigma", 3.0)],
        )
        inspector.close()


if __name__ == "__main__":
    unittest.main()
