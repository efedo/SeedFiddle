from __future__ import annotations

import json
import unittest
import ast
from pathlib import Path

from seedvision.pipeline import NodeStatus, build_default_pipeline


class PipelineModelTests(unittest.TestCase):
    def test_default_pipeline_is_left_to_right_and_acyclic(self) -> None:
        graph = build_default_pipeline()
        order = graph.topological_order()
        self.assertEqual(order[0], "raw_images")
        self.assertEqual(order[-1], "output")
        self.assertEqual(len(graph.nodes), 41)
        self.assertEqual(len(graph.connections), 89)
        for connection in graph.connections:
            self.assertLess(graph.node(connection.source).x, graph.node(connection.target).x)

    def test_identification_change_invalidates_only_downstream_nodes(self) -> None:
        graph = build_default_pipeline()
        graph.set_status("raw_images", NodeStatus.COMPLETE, "image.jpg")
        graph.set_status("layout_detection", NodeStatus.COMPLETE, "Dish found")
        graph.set_status("identification", NodeStatus.COMPLETE, "16 proposals")
        affected = graph.set_parameter(
            "circle_candidates", "circle_accumulator_threshold", 24
        )
        self.assertEqual(
            affected,
            (
                "circle_candidates",
                "identification",
                "instance_masks",
                "seed_edge_curves",
                "radial_profile",
                "assignment_confidence",
                "wrinkling",
                "coat_damage",
                "proposal_disagreement",
                "contact_graph",
                "review",
                "measurements",
                "classification",
                "aggregation",
                "output",
            ),
        )
        self.assertEqual(graph.node("identification").status, NodeStatus.IDLE)
        self.assertEqual(graph.node("layout_detection").status, NodeStatus.COMPLETE)
        self.assertEqual(graph.node("raw_images").status, NodeStatus.COMPLETE)

    def test_overlay_nodes_expose_the_analysis_branches(self) -> None:
        graph = build_default_pipeline()
        self.assertTrue(graph.node("background_likelihood").bypassable)
        self.assertEqual(
            graph.upstream("instance_masks"),
            (
                "identification",
                "background_likelihood",
                "refined_background_likelihood",
                "seed_scale_estimation",
            ),
        )
        self.assertEqual(
            graph.upstream("refined_background_likelihood"),
            ("background_likelihood", "seed_scale_estimation"),
        )
        self.assertEqual(
            graph.upstream("background_likelihood"),
            ("deskew_colour", "seed_scale_estimation"),
        )
        self.assertEqual(graph.upstream("edge_gradients"), ("deskew_colour",))
        self.assertEqual(graph.upstream("undirected_edges"), ("edge_gradients",))
        self.assertEqual(graph.upstream("directed_edges"), ("edge_gradients",))
        edge_outputs = {
            connection.target: connection.source_port
            for connection in graph.connections
            if connection.source == "edge_gradients"
        }
        self.assertEqual(edge_outputs["undirected_edges"], "undirected")
        self.assertEqual(edge_outputs["directed_edges"], "directed")
        self.assertIn("edge_ridges", edge_outputs)
        self.assertIn("edge_traces", edge_outputs)
        self.assertEqual(
            graph.upstream("seed_edge_curves"),
            (
                "edge_traces",
                "edge_gradients",
                "seed_scale_estimation",
                "background_likelihood",
                "foreground_segmentation",
                "instance_masks",
            ),
        )
        self.assertEqual(graph.upstream("edge_ridges"), ("edge_gradients",))
        self.assertEqual(
            graph.upstream("edge_traces"), ("edge_ridges", "edge_gradients")
        )
        self.assertEqual(
            graph.upstream("review"),
            (
                "touching_split",
                "proposal_disagreement",
                "assignment_confidence",
                "contact_graph",
                "instance_masks",
            ),
        )
        self.assertEqual(
            graph.upstream("seed_interior"),
            (
                "foreground_segmentation",
                "background_likelihood",
                "refined_background_likelihood",
            ),
        )
        self.assertEqual(
            graph.upstream("boundary_normals"),
            ("seed_interior", "directed_edges"),
        )
        self.assertEqual(
            graph.upstream("classification"),
            (
                "review",
                "wrinkling",
                "coat_damage",
                "pattern_decomposition",
                "colour_probabilities",
                "calibration_residuals",
            ),
        )

    def test_calibration_nodes_feed_the_corrected_analysis_path(self) -> None:
        graph = build_default_pipeline()
        self.assertEqual(graph.upstream("colour_reference"), ("metadata",))
        self.assertEqual(graph.upstream("deskew_colour"), ("colour_reference",))
        self.assertEqual(graph.upstream("ruler_detection"), ("deskew_colour",))
        self.assertEqual(
            graph.upstream("scale_calibration"),
            ("deskew_colour", "ruler_detection"),
        )
        self.assertEqual(
            graph.upstream("identification"),
            ("distance_candidates", "circle_candidates"),
        )
        self.assertEqual(
            graph.upstream("seed_scale_estimation"),
            ("deskew_colour", "layout_detection"),
        )
        self.assertEqual(
            graph.upstream("foreground_segmentation"),
            ("deskew_colour", "layout_detection", "seed_scale_estimation"),
        )
        affected = graph.set_parameter(
            "ruler_detection", "ruler_length_mm", 200.0
        )
        self.assertNotIn("deskew_colour", affected)
        self.assertIn("scale_calibration", affected)
        self.assertNotIn("identification", affected)
        self.assertNotIn("foreground_segmentation", affected)
        self.assertNotIn("circle_candidates", affected)
        self.assertIn("output", affected)

    def test_pipeline_configuration_is_json_serializable(self) -> None:
        payload = build_default_pipeline().to_dict()
        encoded = json.dumps(payload)
        self.assertIn("Seed identification", encoded)
        self.assertIn("InstanceProposals", encoded)
        self.assertIn("calculation_seconds", payload["nodes"][0])

    def test_every_configurable_node_has_compact_graph_controls(self) -> None:
        graph = build_default_pipeline()
        for node in graph.nodes.values():
            if not node.parameter_specs:
                continue
            self.assertGreaterEqual(len(node.inline_parameters), 1, node.identifier)
            self.assertLessEqual(len(node.inline_parameters), 4, node.identifier)
            spec_keys = {spec.key for spec in node.parameter_specs}
            self.assertTrue(
                all(key in spec_keys for key, _ in node.inline_parameters),
                node.identifier,
            )

    def test_every_exposed_parameter_has_a_computational_consumer(self) -> None:
        graph = build_default_pipeline()
        controls = {
            (node.identifier, spec.key)
            for node in graph.nodes.values()
            for spec in node.parameter_specs
        }
        consumed_keys: set[str] = set()

        class CalculationReadVisitor(ast.NodeVisitor):
            def __init__(self) -> None:
                self.function_names: list[str] = []

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                self.function_names.append(node.name)
                self.generic_visit(node)
                self.function_names.pop()

            visit_AsyncFunctionDef = visit_FunctionDef

            def visit_Attribute(self, node: ast.Attribute) -> None:
                function_name = (
                    self.function_names[-1] if self.function_names else ""
                )
                if isinstance(node.ctx, ast.Load) and function_name != "__post_init__":
                    consumed_keys.add(node.attr)
                self.generic_visit(node)

        source_root = Path(__file__).resolve().parents[1] / "seedvision"
        for path in source_root.rglob("*.py"):
            relative = path.relative_to(source_root).as_posix()
            if relative == "pipeline/model.py" or relative.startswith("ui/"):
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            CalculationReadVisitor().visit(tree)

        unused = sorted(
            f"{node_id}.{key}"
            for node_id, key in controls
            if key not in consumed_keys
        )
        self.assertEqual(unused, [])

    def test_invalid_parameter_value_is_rejected(self) -> None:
        graph = build_default_pipeline()
        with self.assertRaises(ValueError):
            graph.set_parameter(
                "circle_candidates", "circle_accumulator_threshold", 100
            )


class NodeTimingRecorderTests(unittest.TestCase):
    def test_cpu_and_explicit_spans_are_aggregated_by_node(self) -> None:
        from seedvision.timing import NodeTimingRecorder

        events = []
        recorder = NodeTimingRecorder(
            progress_callback=lambda node_id, state: events.append((node_id, state))
        )
        recorder.add_seconds("decode", 0.125)
        with recorder.measure("analysis"):
            sum(range(100))
        with recorder.measure("analysis"):
            sum(range(100))
        silent = recorder.start("preparation", report_progress=False)
        recorder.stop(silent)
        result = recorder.finalize()
        self.assertEqual(result["decode"], 0.125)
        self.assertGreater(result["analysis"], 0.0)
        self.assertEqual(
            events,
            [
                ("analysis", "started"),
                ("analysis", "completed"),
                ("analysis", "started"),
                ("analysis", "completed"),
            ],
        )


class BaselineSettingsTests(unittest.TestCase):
    def test_out_of_range_setting_is_rejected(self) -> None:
        try:
            from seedvision.segmentation import BaselineSettings
        except ImportError as error:
            self.skipTest(f"Baseline dependencies unavailable: {error}")
        with self.assertRaises(ValueError):
            BaselineSettings(inner_radius_fraction=0.5)
        with self.assertRaises(ValueError):
            BaselineSettings(reference_roi_x_min=0.8, reference_roi_x_max=0.6)
        with self.assertRaises(ValueError):
            BaselineSettings(
                circle_min_radius_fraction=0.7,
                circle_max_radius_fraction=0.6,
            )

    def test_composite_dish_and_layer_settings_are_validated(self) -> None:
        from seedvision.segmentation import (
            AdvancedAnalysisSettings,
            AnalysisLayerSettings,
            DishDetectionSettings,
        )

        with self.assertRaises(ValueError):
            DishDetectionSettings(min_radius_fraction=0.3, max_radius_fraction=0.2)
        with self.assertRaises(ValueError):
            AnalysisLayerSettings(
                curve_radius_low_fraction=0.6,
                curve_radius_nominal_fraction=0.5,
            )
        with self.assertRaises(ValueError):
            AdvancedAnalysisSettings(compute_device="metal")


if __name__ == "__main__":
    unittest.main()
