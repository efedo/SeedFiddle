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
        self.assertEqual(set(order), set(graph.nodes))
        self.assertEqual(len(graph.nodes), 27)
        self.assertEqual(len(graph.connections), 50)
        self.assertEqual(
            graph.upstream("perimeter_background_reference"),
            ("deskew_colour", "layout_detection", "scale_calibration"),
        )
        self.assertEqual(
            graph.downstream("perimeter_background_reference"),
            ("background_likelihood",),
        )
        for connection in graph.connections:
            self.assertLess(graph.node(connection.source).x, graph.node(connection.target).x)

    def test_identification_change_invalidates_only_downstream_nodes(self) -> None:
        graph = build_default_pipeline()
        for node_id in tuple(graph.unused_nodes):
            graph.restore_unused_node(node_id)
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
                "proposal_disagreement",
                "wrinkling",
                "coat_damage",
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
        for node_id in tuple(graph.unused_nodes):
            graph.restore_unused_node(node_id)
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
            graph.upstream("foreground_noise_likelihood"),
            ("foreground_segmentation", "seed_scale_estimation"),
        )
        self.assertEqual(
            graph.upstream("background_likelihood"),
            (
                "deskew_colour",
                "seed_scale_estimation",
                "perimeter_background_reference",
            ),
        )
        self.assertEqual(graph.upstream("edge_gradients"), ("deskew_colour",))
        self.assertEqual(graph.upstream("undirected_edges"), ("edge_gradients",))
        self.assertEqual(graph.upstream("directed_edges"), ("edge_gradients",))
        self.assertEqual(
            graph.upstream("surface_darkness_gradients"),
            ("deskew_colour", "seed_scale_estimation"),
        )
        self.assertEqual(
            graph.upstream("lightening_gradient_ceiling"),
            ("surface_darkness_gradients",),
        )
        self.assertEqual(
            graph.upstream("darkening_gradient_ceiling"),
            ("surface_darkness_gradients",),
        )
        self.assertEqual(
            graph.upstream("frequency_noise_masks"),
            ("deskew_colour", "seed_scale_estimation"),
        )
        surface_outputs = {
            (connection.target, connection.target_port): connection.source_port
            for connection in graph.connections
            if connection.source == "surface_darkness_gradients"
        }
        self.assertEqual(
            surface_outputs,
            {
                ("lightening_gradient_ceiling", "magnitude"): "lightening_magnitude",
                ("lightening_gradient_ceiling", "direction"): "lightening_direction",
                ("darkening_gradient_ceiling", "magnitude"): "darkening_magnitude",
                ("darkening_gradient_ceiling", "direction"): "darkening_direction",
            },
        )
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
            set(graph.upstream("seed_edge_curves")),
            {
                "edge_traces",
                "edge_gradients",
                "seed_scale_estimation",
                "background_likelihood",
                "foreground_segmentation",
                "instance_masks",
            },
        )
        self.assertEqual(graph.upstream("edge_ridges"), ("edge_gradients",))
        self.assertEqual(
            graph.upstream("edge_traces"), ("edge_ridges", "edge_gradients")
        )
        self.assertEqual(
            set(graph.upstream("review")),
            {
                "touching_split",
                "proposal_disagreement",
                "assignment_confidence",
                "contact_graph",
                "instance_masks",
            },
        )
        self.assertEqual(
            graph.upstream("seed_interior"),
            (
                "foreground_segmentation",
                "background_likelihood",
                "refined_background_likelihood",
                "foreground_noise_likelihood",
            ),
        )
        affected = graph.set_parameter(
            "foreground_noise_likelihood",
            "foreground_noise_vector_length_fraction",
            0.70,
        )
        self.assertIn("seed_interior", affected)
        self.assertIn("boundary_normals", affected)
        self.assertEqual(
            graph.upstream("boundary_normals"),
            ("seed_interior", "directed_edges"),
        )
        self.assertEqual(
            set(graph.upstream("classification")),
            {
                "review",
                "wrinkling",
                "coat_damage",
                "pattern_decomposition",
                "colour_probabilities",
                "calibration_residuals",
            },
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
            graph.upstream("layout_detection"),
            ("deskew_colour", "scale_calibration"),
        )
        self.assertNotIn("identification", graph.nodes)
        self.assertIn("identification", graph.unused_nodes)
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
        self.assertIn("layout_detection", affected)
        self.assertNotIn("identification", affected)
        self.assertIn("foreground_segmentation", affected)
        self.assertIn("foreground_noise_likelihood", affected)
        self.assertNotIn("circle_candidates", affected)
        self.assertNotIn("output", affected)

    def test_circle_candidates_are_preserved_in_the_unused_node_toolbox(self) -> None:
        graph = build_default_pipeline()
        self.assertNotIn("circle_candidates", graph.nodes)
        self.assertIn("circle_candidates", graph.unused_nodes)
        self.assertEqual(graph.node("circle_candidates").title, "Circle candidates")
        self.assertEqual(len(graph.unused_nodes), 20)
        self.assertEqual(len(graph.unused_connections), 61)
        restored = graph.restore_unused_node("circle_candidates")
        self.assertEqual(len(restored), 7)
        self.assertIn("circle_candidates", graph.nodes)
        self.assertNotIn("circle_candidates", graph.unused_nodes)
        self.assertEqual(graph.upstream("circle_candidates"), (
            "foreground_segmentation",
            "seed_scale_estimation",
            "edge_gradients",
            "image_quality",
            "illumination_decomposition",
        ))

    def test_surface_darkness_branch_is_disabled_in_the_unused_toolbox(self) -> None:
        graph = build_default_pipeline()
        branch = {
            "surface_darkness_gradients",
            "lightening_gradient_ceiling",
            "darkening_gradient_ceiling",
        }
        self.assertTrue(branch.issubset(graph.unused_nodes))
        self.assertTrue(branch.isdisjoint(graph.nodes))
        self.assertTrue(all(not graph.node(node_id).enabled for node_id in branch))
        self.assertTrue(
            all(graph.node(node_id).status is NodeStatus.BYPASSED for node_id in branch)
        )
        preserved = {
            connection
            for connection in graph.unused_connections
            if connection.source in branch or connection.target in branch
        }
        self.assertEqual(len(preserved), 6)

        restored = graph.restore_unused_node("surface_darkness_gradients")
        self.assertEqual(len(restored), 2)
        self.assertEqual(
            graph.upstream("surface_darkness_gradients"),
            ("deskew_colour", "seed_scale_estimation"),
        )
        self.assertFalse(graph.node("surface_darkness_gradients").enabled)

    def test_pipeline_configuration_is_json_serializable(self) -> None:
        payload = build_default_pipeline().to_dict()
        encoded = json.dumps(payload)
        self.assertIn("Seed identification", encoded)
        self.assertIn("Circle candidates", encoded)
        self.assertIn("InstanceProposals", encoded)
        self.assertIn("calculation_seconds", payload["nodes"][0])
        self.assertEqual(len(payload["unused_nodes"]), 20)

    def test_unfinished_candidate_and_mask_branches_default_to_disabled(self) -> None:
        graph = build_default_pipeline()
        roots = {"seed_interior"}
        expected = set(roots)
        for root in roots:
            expected.update(graph.downstream(root, recursive=True))
        self.assertEqual(
            {node.identifier for node in graph.nodes.values() if not node.enabled},
            expected,
        )
        self.assertTrue(all(graph.node(node_id).bypassable for node_id in expected))
        self.assertTrue(
            all(graph.node(node_id).status is NodeStatus.BYPASSED for node_id in expected)
        )
        self.assertFalse(graph.node("circle_candidates").enabled)
        self.assertFalse(graph.node("distance_candidates").enabled)
        self.assertFalse(graph.node("instance_masks").enabled)
        self.assertFalse(graph.node("surface_darkness_gradients").enabled)
        self.assertIn("surface_darkness_gradients", graph.unused_nodes)

    def test_disabling_a_node_disables_all_dependents(self) -> None:
        graph = build_default_pipeline()
        affected = graph.set_enabled("background_likelihood", False)
        self.assertFalse(graph.node("background_likelihood").enabled)
        self.assertTrue(graph.node("identification").enabled is False)
        self.assertTrue(graph.node("foreground_noise_likelihood").enabled)
        self.assertTrue(
            all(not graph.node(node_id).enabled for node_id in affected)
        )

    def test_reset_parameters_restores_authored_defaults(self) -> None:
        graph = build_default_pipeline()
        original = graph.node("foreground_segmentation").parameters[
            "foreground_reference_weight"
        ]
        graph.set_parameter(
            "foreground_segmentation", "foreground_reference_weight", 0.25
        )
        affected = graph.reset_parameters("foreground_segmentation")
        self.assertEqual(
            graph.node("foreground_segmentation").parameters[
                "foreground_reference_weight"
            ],
            original,
        )
        self.assertIn("foreground_segmentation", affected)
        self.assertEqual(graph.reset_parameters("foreground_segmentation"), ())

    def test_every_configurable_node_has_compact_graph_controls(self) -> None:
        graph = build_default_pipeline()
        for node in (*graph.nodes.values(), *graph.unused_nodes.values()):
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
            for node in (*graph.nodes.values(), *graph.unused_nodes.values())
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
        graph.restore_unused_node("circle_candidates")
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
        with self.assertRaises(ValueError):
            BaselineSettings(
                circle_edge_magnitude_weight=0.0,
                circle_sensor_noise_weight=0.0,
                circle_flattened_grayscale_weight=0.0,
                circle_shadow_weight=0.0,
                circle_highlight_weight=0.0,
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
