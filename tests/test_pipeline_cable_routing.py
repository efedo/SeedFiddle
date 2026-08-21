from __future__ import annotations

import os
from time import perf_counter
import unittest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _node(identifier: str, x: float, y: float):
    from seedvision.pipeline import PipelineNode

    return PipelineNode(
        identifier=identifier,
        title=identifier.replace("_", " ").title(),
        category="Routing test",
        description=f"Synthetic {identifier} node",
        x=x,
        y=y,
    )


def _bundle_graph(*, with_obstacle: bool = False):
    from seedvision.pipeline import PipelineConnection, PipelineGraph

    nodes = [
        _node("source", 0.0, 100.0),
        _node("upper_target", 680.0, 40.0),
        _node("lower_target", 680.0, 180.0),
        _node("reverse_target", -400.0, 100.0),
    ]
    if with_obstacle:
        nodes.append(_node("obstacle", 360.0, 100.0))
    return PipelineGraph(
        nodes,
        (
            PipelineConnection(
                "source",
                "upper_target",
                "SyntheticData",
                source_port="out",
                target_port="in",
            ),
            PipelineConnection(
                "source",
                "lower_target",
                "SyntheticData",
                source_port="out",
                target_port="in",
            ),
            PipelineConnection(
                "source",
                "reverse_target",
                "SyntheticData",
                source_port="out",
                target_port="in",
            ),
        ),
    )


def _routing_graph(*, obstacle_count: int = 2):
    from seedvision.pipeline import PipelineConnection, PipelineGraph

    nodes = [
        _node("source", 0.0, 100.0),
        _node("target", 900.0, 100.0),
    ]
    obstacle_positions = ((300.0, 100.0), (575.0, 100.0))
    nodes.extend(
        _node(f"obstacle_{index}", x, y)
        for index, (x, y) in enumerate(
            obstacle_positions[:obstacle_count], start=1
        )
    )
    # This card is deliberately away from the shortest path. It ensures the
    # route is safe against every unrelated card, not just known blockers.
    if obstacle_count > 1:
        nodes.append(_node("off_path_card", 440.0, 410.0))
    return PipelineGraph(
        nodes,
        (
            PipelineConnection(
                "source",
                "target",
                "SyntheticData",
                source_port="out",
                target_port="in",
            ),
        ),
    )


def _reverse_bundle_graph():
    from seedvision.pipeline import PipelineConnection, PipelineGraph

    return PipelineGraph(
        (
            _node("source", 500.0, 100.0),
            _node("upper_target", 0.0, 40.0),
            _node("lower_target", 0.0, 180.0),
            _node("obstacle", 260.0, 100.0),
        ),
        (
            PipelineConnection(
                "source",
                "upper_target",
                "SyntheticData",
                source_port="out",
                target_port="in",
            ),
            PipelineConnection(
                "source",
                "lower_target",
                "SyntheticData",
                source_port="out",
                target_port="in",
            ),
        ),
    )


def _points(points) -> tuple[tuple[float, float], ...]:
    return tuple(
        (round(float(point.x()), 4), round(float(point.y()), 4))
        for point in points
    )


def _contains_contiguous_points(
    route: tuple[tuple[float, float], ...],
    segment: tuple[tuple[float, float], ...],
) -> bool:
    return any(
        route[index : index + len(segment)] == segment
        for index in range(len(route) - len(segment) + 1)
    )


class PipelineCableRoutingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            from PySide6.QtWidgets import QApplication
        except ImportError as error:
            raise unittest.SkipTest(f"PySide6 unavailable: {error}")
        cls.application = QApplication.instance() or QApplication([])

    def assert_edges_clear_unrelated_cards(self, canvas) -> None:
        from PySide6.QtCore import QRectF
        from PySide6.QtGui import QPainterPath, QPainterPathStroker

        stroker = QPainterPathStroker()
        stroker.setWidth(2.0)
        card_paths = {}
        for node_id, item in canvas.node_items.items():
            card = QPainterPath()
            card.addRect(
                item.mapRectToScene(QRectF(0.0, 0.0, item.width, item.height))
            )
            card_paths[node_id] = card
        collisions = []
        for edge in canvas.edge_items:
            stroke = stroker.createStroke(edge.path())
            for node_id, card in card_paths.items():
                if node_id in (edge.connection.source, edge.connection.target):
                    continue
                if stroke.intersects(card):
                    collisions.append(
                        (edge.connection.source, edge.connection.target, node_id)
                    )
        self.assertFalse(
            collisions,
            f"{len(collisions)} edge/card collisions: {collisions[:20]}",
        )

    def assert_cables_clear_unrelated_cards(self, canvas) -> None:
        from PySide6.QtCore import QRectF
        from PySide6.QtGui import QPainterPath, QPainterPathStroker

        stroker = QPainterPathStroker()
        stroker.setWidth(9.0)
        card_paths = {}
        for node_id, item in canvas.node_items.items():
            card = QPainterPath()
            card.addRect(
                item.mapRectToScene(QRectF(0.0, 0.0, item.width, item.height))
            )
            card_paths[node_id] = card
        collisions = []
        for cable in canvas.cable_items:
            stroke = stroker.createStroke(cable.path())
            for node_id, card in card_paths.items():
                if node_id == cable.bundle_key[0]:
                    continue
                if stroke.intersects(card):
                    collisions.append((cable.bundle_key, node_id))
        self.assertFalse(
            collisions,
            f"{len(collisions)} cable/card collisions: {collisions[:20]}",
        )

    def test_default_mode_preserves_the_established_cubic_paths(self) -> None:
        from seedvision.ui.pipeline_canvas import PipelineCanvas

        canvas = PipelineCanvas(_bundle_graph())
        self.assertFalse(canvas.cable_bundling_enabled)
        self.assertFalse(canvas.obstacle_routing_enabled)
        self.assertEqual(canvas.cable_items, [])
        for edge in canvas.edge_items:
            start = edge.source.output_anchor(edge.connection.source_port)
            end = edge.target.input_anchor(edge.connection.target_port)
            horizontal = max(60.0, abs(float(end.x()) - float(start.x())) * 0.48)
            path = edge.path()

            self.assertEqual(_points(edge.route_points), _points((start, end)))
            self.assertFalse(edge.is_bundled)
            self.assertIsNone(edge.bundle_key)
            self.assertEqual(path.elementCount(), 4)
            expected = (
                (float(start.x()), float(start.y())),
                (float(start.x()) + horizontal, float(start.y())),
                (float(end.x()) - horizontal, float(end.y())),
                (float(end.x()), float(end.y())),
            )
            actual = tuple(
                (path.elementAt(index).x, path.elementAt(index).y)
                for index in range(path.elementCount())
            )
            for actual_point, expected_point in zip(actual, expected, strict=True):
                self.assertAlmostEqual(actual_point[0], expected_point[0], places=6)
                self.assertAlmostEqual(actual_point[1], expected_point[1], places=6)
        canvas.close()

    def test_same_source_and_direction_form_one_deterministic_cable(self) -> None:
        from seedvision.ui.pipeline_canvas import PipelineCanvas

        canvas = PipelineCanvas(_bundle_graph())
        graph_revision = canvas.graph.revision
        canvas.set_cable_bundling_enabled(True)
        self.assertEqual(canvas.graph.revision, graph_revision)

        forward = sorted(
            (
                edge
                for edge in canvas.edge_items
                if edge.connection.target in {"upper_target", "lower_target"}
            ),
            key=lambda edge: edge.connection.target,
        )
        reverse = next(
            edge
            for edge in canvas.edge_items
            if edge.connection.target == "reverse_target"
        )
        self.assertEqual(len(canvas.cable_items), 1)
        cable = canvas.cable_items[0]
        self.assertTrue(all(edge.is_bundled for edge in forward))
        self.assertEqual({edge.bundle_key for edge in forward}, {cable.bundle_key})
        self.assertFalse(reverse.is_bundled)
        self.assertIsNone(reverse.bundle_key)

        trunk = _points(cable.route_points)
        routes = tuple(_points(edge.route_points) for edge in forward)
        self.assertGreaterEqual(len(trunk), 2)
        self.assertTrue(all(_contains_contiguous_points(route, trunk) for route in routes))
        split = trunk[-1]
        self.assertTrue(all(route.index(split) < len(route) - 1 for route in routes))
        self.assertNotEqual(routes[0][-1], routes[1][-1])

        first_layout = (cable.bundle_key, trunk, routes)
        canvas.set_cable_bundling_enabled(False)
        canvas.set_cable_bundling_enabled(True)
        second_cable = canvas.cable_items[0]
        second_forward = sorted(
            (
                edge
                for edge in canvas.edge_items
                if edge.connection.target in {"upper_target", "lower_target"}
            ),
            key=lambda edge: edge.connection.target,
        )
        self.assertEqual(
            first_layout,
            (
                second_cable.bundle_key,
                _points(second_cable.route_points),
                tuple(_points(edge.route_points) for edge in second_forward),
            ),
        )
        canvas.close()

    def test_reverse_bundle_exits_the_output_side_before_routing_left(self) -> None:
        from seedvision.ui.pipeline_canvas import PipelineCanvas

        canvas = PipelineCanvas(_reverse_bundle_graph())
        canvas.set_cable_bundling_enabled(True)
        canvas.set_obstacle_routing_enabled(True)

        self.assertEqual(len(canvas.cable_items), 1)
        cable = canvas.cable_items[0]
        self.assertEqual(cable.bundle_key[1], -1)
        source_x = float(
            canvas.edge_items[0]
            .source.output_anchor(canvas.edge_items[0].connection.source_port)
            .x()
        )
        target_xs = [
            float(edge.target.input_anchor(edge.connection.target_port).x())
            for edge in canvas.edge_items
        ]
        self.assertGreater(float(cable.route_points[0].x()), source_x)
        self.assertLess(float(cable.route_points[-1].x()), min(target_xs))
        self.assertTrue(all(edge.is_bundled for edge in canvas.edge_items))
        self.assert_edges_clear_unrelated_cards(canvas)
        self.assert_cables_clear_unrelated_cards(canvas)
        canvas.close()

    def test_obstacle_routes_clear_every_non_endpoint_node_and_round_bends(self) -> None:
        from PySide6.QtGui import QPainterPath, QPainterPathStroker

        from seedvision.ui.pipeline_canvas import PipelineCanvas

        canvas = PipelineCanvas(_routing_graph())
        graph_revision = canvas.graph.revision
        canvas.set_obstacle_routing_enabled(True)
        self.assertEqual(canvas.graph.revision, graph_revision)
        edge = canvas.edge_items[0]
        self.assertGreater(len(edge.route_points), 2)
        self.assertEqual(
            _points((edge.route_points[0], edge.route_points[-1])),
            _points(
                (
                    edge.source.output_anchor(edge.connection.source_port),
                    edge.target.input_anchor(edge.connection.target_port),
                )
            ),
        )
        for first, second in zip(edge.route_points, edge.route_points[1:]):
            self.assertTrue(
                abs(float(first.x()) - float(second.x())) < 0.01
                or abs(float(first.y()) - float(second.y())) < 0.01,
                _points((first, second)),
            )

        stroker = QPainterPathStroker()
        stroker.setWidth(2.0)
        stroke = stroker.createStroke(edge.path())
        for node_id, item in canvas.node_items.items():
            if node_id in {edge.connection.source, edge.connection.target}:
                continue
            visible_card = item.mapRectToScene(
                item.boundingRect().adjusted(8.0, 8.0, -8.0, -8.0)
            )
            obstacle_path = QPainterPath()
            obstacle_path.addRect(visible_card)
            self.assertFalse(stroke.intersects(obstacle_path), node_id)

        element_types = {
            edge.path().elementAt(index).type
            for index in range(edge.path().elementCount())
        }
        self.assertIn(QPainterPath.ElementType.CurveToElement, element_types)
        canvas.close()

    def test_moving_an_obstacle_recomputes_the_route(self) -> None:
        from seedvision.ui.pipeline_canvas import PipelineCanvas

        canvas = PipelineCanvas(_routing_graph(obstacle_count=1))
        canvas.set_obstacle_routing_enabled(True)
        edge = canvas.edge_items[0]
        before = _points(edge.route_points)
        self.assertGreater(len(before), 2)

        obstacle = canvas.node_items["obstacle_1"]
        obstacle.setPos(obstacle.pos().x(), obstacle.pos().y() + 400.0)
        canvas._node_released(obstacle.node.identifier)
        self.application.processEvents()
        after = _points(edge.route_points)

        self.assertNotEqual(after, before)
        self.assertEqual(len(after), 2)
        self.assertEqual(after[0], before[0])
        self.assertEqual(after[-1], before[-1])
        canvas.close()

    def test_combined_bundling_and_obstacle_routing_preserve_clearance(self) -> None:
        from PySide6.QtGui import QPainterPath, QPainterPathStroker

        from seedvision.ui.pipeline_canvas import PipelineCanvas

        canvas = PipelineCanvas(_bundle_graph(with_obstacle=True))
        canvas.set_cable_bundling_enabled(True)
        canvas.set_obstacle_routing_enabled(True)
        forward = [
            edge
            for edge in canvas.edge_items
            if edge.connection.target in {"upper_target", "lower_target"}
        ]
        self.assertEqual(len(canvas.cable_items), 1)
        self.assertTrue(all(edge.is_bundled for edge in forward))

        obstacle = canvas.node_items["obstacle"]
        obstacle_rect = obstacle.mapRectToScene(
            obstacle.boundingRect().adjusted(8.0, 8.0, -8.0, -8.0)
        )
        obstacle_path = QPainterPath()
        obstacle_path.addRect(obstacle_rect)
        stroker = QPainterPathStroker()
        stroker.setWidth(9.0)
        for routed_item in (*forward, *canvas.cable_items):
            self.assertFalse(
                stroker.createStroke(routed_item.path()).intersects(obstacle_path)
            )
            element_types = {
                routed_item.path().elementAt(index).type
                for index in range(routed_item.path().elementCount())
            }
            self.assertIn(QPainterPath.ElementType.CurveToElement, element_types)
        canvas.close()

    def test_authored_graph_routes_clear_all_cards_with_bounded_refresh_time(self) -> None:
        from seedvision.pipeline import build_default_pipeline
        from seedvision.ui.pipeline_canvas import PipelineCanvas

        canvas = PipelineCanvas(build_default_pipeline())
        canvas.set_obstacle_routing_enabled(True)
        self.assert_edges_clear_unrelated_cards(canvas)

        started = perf_counter()
        canvas.set_cable_bundling_enabled(True)
        combined_seconds = perf_counter() - started
        self.assertLess(
            combined_seconds,
            2.0,
            f"Combined cable/obstacle refresh took {combined_seconds:.3f}s",
        )
        self.assert_edges_clear_unrelated_cards(canvas)
        self.assert_cables_clear_unrelated_cards(canvas)
        canvas.close()

    def test_bundled_branches_remain_individually_interactive(self) -> None:
        from PySide6.QtCore import Qt

        from seedvision.ui.pipeline_canvas import PipelineCanvas

        graph = _bundle_graph()
        canvas = PipelineCanvas(graph)
        canvas.set_cable_bundling_enabled(True)
        self.assertEqual(len(canvas.cable_items), 1)
        cable = canvas.cable_items[0]
        self.assertEqual(cable.acceptedMouseButtons(), Qt.MouseButton.NoButton)
        self.assertFalse(
            bool(cable.flags() & cable.GraphicsItemFlag.ItemIsSelectable)
        )

        for edge in canvas.edge_items:
            self.assertEqual(edge.acceptedMouseButtons(), Qt.MouseButton.RightButton)
            self.assertTrue(
                bool(edge.flags() & edge.GraphicsItemFlag.ItemIsSelectable)
            )
        chosen = next(
            edge
            for edge in canvas.edge_items
            if edge.connection.target == "upper_target"
        )
        chosen.setSelected(True)
        self.assertTrue(chosen.isSelected())
        canvas._disconnect_connection(chosen.connection)

        targets = {connection.target for connection in graph.connections}
        self.assertNotIn("upper_target", targets)
        self.assertIn("lower_target", targets)
        self.assertIn("reverse_target", targets)
        self.assertEqual(len(canvas.edge_items), 2)
        self.assertEqual(canvas.cable_items, [])
        canvas.close()


if __name__ == "__main__":
    unittest.main()
