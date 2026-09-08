from __future__ import annotations

import json
import unittest
import ast
from pathlib import Path
from unittest.mock import Mock

from seedvision.pipeline import (
    INTENTIONALLY_UNEXPOSED_SETTINGS,
    NodeStatus,
    build_default_pipeline,
)


class PipelineModelTests(unittest.TestCase):
    def test_foreground_requires_authored_evidence_and_has_no_refinement_controls(self) -> None:
        graph = build_default_pipeline()
        colour_keys = set(graph.node("background_likelihood").parameters)
        foreground_keys = {key for key in colour_keys if key.startswith("foreground_")}
        background_keys = {key for key in colour_keys if key.startswith("background_")}
        self.assertFalse(any("automatic" in key for key in foreground_keys))
        self.assertFalse(any("refinement" in key for key in foreground_keys))
        self.assertFalse(any("refinement" in key for key in background_keys))
        self.assertIn("foreground_reference_weight", foreground_keys)
        self.assertIn(
            "foreground_include_annotated_seed_instances", foreground_keys
        )

    def test_reference_seed_traits_are_terminal_and_do_not_replace_material_classes(
        self,
    ) -> None:
        graph = build_default_pipeline()
        node = graph.node("reference_seed_traits")
        self.assertEqual(node.title, "Reference seed traits")
        self.assertEqual(
            tuple(port for port, _label in node.output_ports),
            (
                "coat_white",
                "coat_banded_light",
                "coat_banded_dark",
                "coat_other",
                "condition_immature",
                "condition_split",
                "condition_wrinkled",
                "condition_stained",
            ),
        )
        self.assertEqual(graph.downstream("reference_seed_traits"), ())
        self.assertNotIn(
            "reference_seed_traits",
            graph.upstream("reference_texture_prototypes"),
        )
        self.assertNotIn(
            "reference_seed_traits",
            graph.upstream("material_evidence_decision"),
        )
        affected = graph.set_parameter(
            "reference_seed_traits",
            "reference_seed_trait_class_contrast",
            4.25,
        )
        self.assertEqual(affected, ("reference_seed_traits",))

    def test_default_pipeline_is_left_to_right_and_acyclic(self) -> None:
        graph = build_default_pipeline()
        order = graph.topological_order()
        self.assertEqual(order[0], "project")
        self.assertEqual(set(order), set(graph.nodes))
        self.assertEqual(len(graph.nodes), 29)
        self.assertEqual(len(graph.connections), 179)
        self.assertNotIn("perimeter_background_reference", graph.nodes)
        self.assertEqual(
            graph.connection_for_input(
                "background_likelihood", "perimeter_colour_samples"
            ).source,
            "layout_detection",
        )
        automatic_source = graph.connection_for_input(
            "reference_texture_prototypes", "automatic_background_reference"
        )
        self.assertIsNotNone(automatic_source)
        self.assertEqual(
            (automatic_source.source, automatic_source.source_port),
            ("background_likelihood", "background_reference_source"),
        )
        annotation_exclusion = graph.connection_for_input(
            "background_likelihood", "annotations"
        )
        self.assertIsNotNone(annotation_exclusion)
        self.assertEqual(
            (annotation_exclusion.source, annotation_exclusion.source_port),
            ("project", "annotations"),
        )
        for consumer in (
            "refined_background_likelihood",
            "reference_texture_prototypes",
        ):
            annotated_foreground = graph.connection_for_input(
                consumer, "automatic_foreground_reference"
            )
            self.assertIsNotNone(annotated_foreground)
            self.assertEqual(
                (
                    annotated_foreground.source,
                    annotated_foreground.source_port,
                ),
                (
                    "background_likelihood",
                    "annotated_foreground_reference_source",
                ),
            )
        centre_edits = graph.connection_for_input(
            "procedural_instances", "manual_centres"
        )
        self.assertIsNotNone(centre_edits)
        self.assertEqual(
            (centre_edits.source, centre_edits.source_port),
            ("project", "annotations"),
        )
        validated_oval_centres = graph.connection_for_input(
            "procedural_instances", "validated_oval_centres"
        )
        self.assertIsNotNone(validated_oval_centres)
        self.assertEqual(
            (
                validated_oval_centres.source,
                validated_oval_centres.source_port,
            ),
            ("seed_edge_curves", "oval_centre_probability"),
        )
        true_edge_support = graph.connection_for_input(
            "reference_edge_probability", "magnitude"
        )
        self.assertIsNotNone(true_edge_support)
        self.assertEqual(
            (true_edge_support.source, true_edge_support.source_port),
            ("edge_gradients", "magnitude"),
        )
        authoritative_reference_edge = graph.connection_for_input(
            "procedural_instances", "reference_probability"
        )
        self.assertIsNotNone(authoritative_reference_edge)
        self.assertEqual(
            (
                authoritative_reference_edge.source,
                authoritative_reference_edge.source_port,
            ),
            ("reference_edge_probability", "edge_probability"),
        )
        for connection in graph.connections:
            self.assertLess(graph.node(connection.source).x, graph.node(connection.target).x)

    def test_every_authored_dependency_has_labelled_typed_ports(self) -> None:
        graph = build_default_pipeline()
        all_nodes = {**graph.nodes, **graph.unused_nodes}
        all_connections = (*graph.connections, *graph.unused_connections)
        self.assertEqual(len(graph.connection_templates), 257)
        self.assertTrue(all(node.output_ports for node in all_nodes.values()))
        self.assertEqual(
            len(all_connections),
            len({(edge.target, edge.target_port) for edge in all_connections}),
        )
        for connection in all_connections:
            self.assertTrue(connection.source_port)
            self.assertTrue(connection.target_port)
            source = all_nodes[connection.source]
            target = all_nodes[connection.target]
            self.assertIn(connection.source_port, dict(source.output_ports))
            self.assertIn(connection.target_port, dict(target.input_ports))
            self.assertEqual(
                source.output_port_types[connection.source_port],
                connection.data_type,
            )
            self.assertEqual(
                target.input_port_types[connection.target_port],
                connection.data_type,
            )

    def test_restored_toolbox_connections_are_also_left_to_right(self) -> None:
        graph = build_default_pipeline()
        for node_id in tuple(graph.unused_nodes):
            graph.restore_unused_node(node_id)
        for connection in graph.connections:
            self.assertLess(
                graph.node(connection.source).x,
                graph.node(connection.target).x,
                (connection.source, connection.target),
            )

    def test_every_node_matches_the_audited_direct_input_contract(self) -> None:
        """Lock runtime inputs independently of the wires that create ports."""

        graph = build_default_pipeline()
        for node_id in tuple(graph.unused_nodes):
            graph.restore_unused_node(node_id)
        expected = {
            "project": set(),
            "metadata": {"project"},
            "species_reference_library": {"project", "metadata"},
            "deskew_colour": {"project"},
            "hue_only": {"deskew_colour", "layout_detection"},
            "wavelet_decomposition": {"deskew_colour", "layout_detection"},
            "ruler_detection": {"deskew_colour"},
            "layout_detection": {"deskew_colour", "ruler_detection"},
            "seed_scale_estimation": {
                "project", "deskew_colour", "layout_detection",
                "species_reference_library",
            },
            "background_likelihood": {
                "deskew_colour", "layout_detection", "seed_scale_estimation",
                "ruler_detection", "project", "species_reference_library",
            },
            "refined_background_likelihood": {
                "background_likelihood", "deskew_colour", "layout_detection",
                "seed_scale_estimation", "project", "species_reference_library",
            },
            "edge_gradients": {
                "wavelet_decomposition", "deskew_colour", "layout_detection",
                "illumination_decomposition",
            },
            "surface_darkness_gradients": {
                "deskew_colour", "layout_detection", "seed_scale_estimation",
            },
            "lightening_gradient_ceiling": {"surface_darkness_gradients"},
            "darkening_gradient_ceiling": {"surface_darkness_gradients"},
            "frequency_noise_masks": {
                "deskew_colour", "layout_detection", "seed_scale_estimation",
            },
            "reference_texture_prototypes": {
                "background_likelihood",
                "deskew_colour", "layout_detection", "seed_scale_estimation",
                "project", "edge_gradients",
                "frequency_noise_masks", "species_reference_library",
            },
            "reference_seed_traits": {
                "deskew_colour", "project", "metadata",
                "seed_scale_estimation", "edge_gradients",
                "frequency_noise_masks", "material_evidence_decision",
                "species_reference_library",
            },
            "material_evidence_decision": {
                "background_likelihood", "refined_background_likelihood",
                "reference_texture_prototypes", "seed_scale_estimation",
            },
            "reference_edge_probability": {
                "reference_texture_prototypes", "edge_gradients",
                "seed_scale_estimation",
            },
            "edge_traces": {
                "reference_edge_probability", "edge_gradients",
                "seed_scale_estimation",
            },
            "distance_candidates": {
                "material_evidence_decision", "seed_scale_estimation",
            },
            "circle_candidates": {
                "deskew_colour", "layout_detection", "seed_scale_estimation",
                "edge_gradients", "image_quality",
                "illumination_decomposition", "distance_candidates",
            },
            "identification": {
                "circle_candidates", "distance_candidates",
                "seed_scale_estimation",
            },
            "instance_masks": {
                "identification", "layout_detection",
                "material_evidence_decision", "seed_scale_estimation",
            },
            "seed_edge_curves": {
                "edge_traces", "edge_gradients", "seed_scale_estimation",
                "material_evidence_decision",
                "reference_edge_probability",
            },
            "boundary_normals": {
                "material_evidence_decision", "deskew_colour", "layout_detection",
                "seed_scale_estimation",
            },
            "touching_split": {
                "distance_candidates", "boundary_normals",
                "material_evidence_decision", "layout_detection",
                "seed_scale_estimation",
            },
            "ellipse_likelihood": {
                "identification", "instance_masks", "boundary_normals",
                "seed_scale_estimation", "deskew_colour", "layout_detection",
            },
            "proposal_disagreement": {
                "circle_candidates", "distance_candidates", "ellipse_likelihood",
                "material_evidence_decision", "seed_scale_estimation", "layout_detection",
            },
            "assignment_confidence": {
                "identification", "instance_masks", "boundary_normals",
                "material_evidence_decision", "layout_detection",
            },
            "contact_graph": {
                "identification", "assignment_confidence", "boundary_normals",
            },
            "illumination_decomposition": {
                "deskew_colour", "seed_scale_estimation", "layout_detection",
            },
            "image_quality": {
                "deskew_colour", "layout_detection", "seed_scale_estimation",
            },
            "procedural_instances": {
                "layout_detection", "seed_scale_estimation",
                "material_evidence_decision",
                "edge_gradients", "reference_edge_probability", "edge_traces",
                "surface_darkness_gradients", "illumination_decomposition",
                "seed_edge_curves", "project",
            },
            "unet_instances": {
                "metadata", "deskew_colour", "layout_detection",
                "seed_scale_estimation",
                "background_likelihood", "refined_background_likelihood",
                "edge_gradients",
                "reference_edge_probability", "illumination_decomposition",
                "image_quality",
            },
            "stardist_instances": {
                "metadata", "deskew_colour", "layout_detection",
                "seed_scale_estimation",
                "background_likelihood", "refined_background_likelihood",
                "edge_gradients",
                "reference_edge_probability", "illumination_decomposition",
                "image_quality",
            },
            "radial_profile": {
                "identification", "instance_masks", "deskew_colour",
                "layout_detection",
            },
            "wrinkling": {
                "material_evidence_decision", "boundary_normals", "deskew_colour",
                "layout_detection", "seed_scale_estimation",
            },
            "coat_damage": {
                "radial_profile", "boundary_normals", "material_evidence_decision",
                "deskew_colour", "layout_detection", "seed_scale_estimation",
            },
            "pattern_decomposition": {
                "material_evidence_decision", "deskew_colour", "layout_detection",
                "seed_scale_estimation",
            },
            "colour_probabilities": {
                "material_evidence_decision", "deskew_colour", "layout_detection",
            },
            "calibration_residuals": {
                "ruler_detection", "deskew_colour", "layout_detection",
            },
            "review": {
                "instance_masks", "touching_split", "assignment_confidence",
                "proposal_disagreement", "contact_graph",
            },
            "measurements": {"review", "ruler_detection"},
            "classification": {
                "coat_damage", "review", "wrinkling", "pattern_decomposition",
                "colour_probabilities", "calibration_residuals",
            },
            "aggregation": {"measurements", "classification"},
            "output": {"aggregation"},
        }
        self.assertEqual(set(expected), set(graph.nodes))
        self.assertEqual(
            {
                node_id: set(graph.upstream(node_id))
                for node_id in graph.nodes
            },
            expected,
        )

    def test_each_generated_datum_has_one_succinct_output_socket(self) -> None:
        graph = build_default_pipeline()
        for node in (*graph.nodes.values(), *graph.unused_nodes.values()):
            types = tuple(node.output_port_types.values())
            self.assertEqual(len(types), len(set(types)), node.identifier)

    def test_geometry_and_scale_changes_reach_their_hidden_consumers(self) -> None:
        graph = build_default_pipeline()
        for node_id in tuple(graph.unused_nodes):
            graph.restore_unused_node(node_id)
        layout_affected = graph.set_parameter(
            "layout_detection", "rim_pair_expected_separation_fraction", 0.05
        )
        self.assertIn("edge_gradients", layout_affected)
        self.assertIn("frequency_noise_masks", layout_affected)
        self.assertIn("image_quality", layout_affected)

        scale_affected = graph.set_parameter(
            "seed_scale_estimation", "reference_scale_factor", 0.73
        )
        self.assertIn("edge_traces", scale_affected)
        self.assertIn("reference_edge_probability", scale_affected)
        self.assertIn("seed_edge_curves", scale_affected)

    def test_disconnection_bypasses_and_reconnection_restores_only_dependents(self) -> None:
        graph = build_default_pipeline()
        connection = next(
            edge
            for edge in graph.connections
            if edge.source == "background_likelihood"
            and edge.target == "refined_background_likelihood"
            and edge.source_port == "annotated_foreground_reference_source"
        )
        graph.set_enabled("procedural_instances", False)
        affected = graph.disconnect(connection)
        self.assertIn("refined_background_likelihood", affected)
        self.assertFalse(graph.node("refined_background_likelihood").enabled)
        self.assertFalse(graph.node("procedural_instances").enabled)
        self.assertEqual(
            graph.missing_input_ports("refined_background_likelihood"),
            (connection.target_port,),
        )
        with self.assertRaisesRegex(ValueError, "Reconnect required"):
            graph.set_enabled("refined_background_likelihood", True)

        reconnected = graph.connect(
            connection.source,
            connection.source_port,
            connection.target,
            connection.target_port,
        )
        self.assertEqual(set(reconnected), set(affected))
        self.assertTrue(graph.node("refined_background_likelihood").enabled)
        # This node was deliberately disabled before the wire edit.
        self.assertFalse(graph.node("procedural_instances").enabled)
        self.assertFalse(graph.node("unet_instances").enabled)

    def test_reconnection_rejects_unimplemented_typed_substitutions(self) -> None:
        graph = build_default_pipeline()
        connection = next(
            edge
            for edge in graph.connections
            if edge.source == "background_likelihood"
            and edge.target == "refined_background_likelihood"
            and edge.source_port == "annotated_foreground_reference_source"
        )
        graph.disconnect(connection)
        other_source = graph.node("background_likelihood")
        other_port = next(
            port_id
            for port_id, data_type in other_source.output_port_types.items()
            if data_type == "BackgroundProbability"
        )
        with self.assertRaisesRegex(
            ValueError,
            "different data types|requires|not an authored calculation",
        ):
            graph.connect(
                "background_likelihood",
                other_port,
                connection.target,
                connection.target_port,
            )

    def test_identification_change_invalidates_only_downstream_nodes(self) -> None:
        graph = build_default_pipeline()
        for node_id in tuple(graph.unused_nodes):
            graph.restore_unused_node(node_id)
        graph.set_status("project", NodeStatus.COMPLETE, "image.jpg")
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
                "ellipse_likelihood",
                "assignment_confidence",
                "radial_profile",
                "proposal_disagreement",
                "contact_graph",
                "coat_damage",
                "review",
                "measurements",
                "classification",
                "aggregation",
                "output",
            ),
        )
        self.assertEqual(graph.node("identification").status, NodeStatus.IDLE)
        self.assertEqual(graph.node("layout_detection").status, NodeStatus.COMPLETE)
        self.assertEqual(graph.node("project").status, NodeStatus.COMPLETE)

    def test_overlay_nodes_expose_the_analysis_branches(self) -> None:
        graph = build_default_pipeline()
        for node_id in tuple(graph.unused_nodes):
            graph.restore_unused_node(node_id)
        self.assertFalse(graph.node("background_likelihood").bypassable)
        self.assertEqual(
            graph.upstream("instance_masks"),
            (
                "identification",
                "layout_detection",
                "material_evidence_decision",
                "seed_scale_estimation",
            ),
        )
        self.assertEqual(
            graph.upstream("refined_background_likelihood"),
            (
                "deskew_colour",
                "layout_detection",
                "species_reference_library",
                "seed_scale_estimation",
                "project",
                "background_likelihood",
            ),
        )
        self.assertNotIn("foreground_noise_likelihood", graph.nodes)
        self.assertEqual(
            graph.node("refined_background_likelihood").title,
            "Material noise probabilities",
        )
        self.assertFalse(
            {
                "background_probability",
                "foreground_probability",
            }
            & {
                connection.target_port
                for connection in graph.connections
                if connection.target == "refined_background_likelihood"
            }
        )
        self.assertTrue(
            graph.node("refined_background_likelihood").parameters[
                "background_noise_enabled"
            ]
        )
        self.assertTrue(
            graph.node("refined_background_likelihood").parameters[
                "foreground_noise_enabled"
            ]
        )
        self.assertEqual(
            graph.upstream("background_likelihood"),
            (
                "deskew_colour",
                "species_reference_library",
                "layout_detection",
                "seed_scale_estimation",
                "ruler_detection",
                "project",
            ),
        )
        self.assertEqual(
            graph.upstream("edge_gradients"),
            (
                "wavelet_decomposition",
                "deskew_colour",
                "illumination_decomposition",
                "layout_detection",
            ),
        )
        self.assertIn("undirected", dict(graph.node("edge_gradients").output_ports))
        self.assertIn("directed", dict(graph.node("edge_gradients").output_ports))
        self.assertEqual(
            graph.upstream("surface_darkness_gradients"),
            ("deskew_colour", "layout_detection", "seed_scale_estimation"),
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
            ("deskew_colour", "layout_detection", "seed_scale_estimation"),
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
                ("procedural_instances", "surface_darkening"): "darkening_magnitude",
            },
        )
        edge_outputs = {
            connection.target: connection.source_port
            for connection in graph.connections
            if connection.source == "edge_gradients"
        }
        self.assertIn("ridges", dict(graph.node("edge_gradients").output_ports))
        self.assertEqual(edge_outputs["reference_texture_prototypes"], "ridges")
        self.assertIn("edge_traces", edge_outputs)
        self.assertEqual(
            set(graph.upstream("seed_edge_curves")),
            {
                "edge_traces",
                "edge_gradients",
                "seed_scale_estimation",
                "material_evidence_decision",
                "reference_edge_probability",
            },
        )
        self.assertEqual(
            set(graph.upstream("reference_edge_probability")),
            {
                "reference_texture_prototypes",
                "edge_gradients",
                "seed_scale_estimation",
            },
        )
        self.assertEqual(
            graph.set_parameter(
                "reference_edge_probability",
                "reference_ridge_high_threshold",
                0.30,
            ),
            (
                "reference_edge_probability",
                "edge_traces",
                "unet_instances",
                "stardist_instances",
                "seed_edge_curves",
                "procedural_instances",
            ),
        )
        self.assertEqual(
            graph.upstream("edge_traces"),
            (
                "edge_gradients",
                "reference_edge_probability",
                "seed_scale_estimation",
            ),
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
        affected = graph.set_parameter(
            "refined_background_likelihood",
            "foreground_noise_vector_length_fraction",
            0.70,
        )
        self.assertIn("boundary_normals", affected)
        self.assertEqual(
            graph.upstream("boundary_normals"),
            (
                "material_evidence_decision",
                "deskew_colour",
                "layout_detection",
                "seed_scale_estimation",
            ),
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

    def test_procedural_instances_have_explicit_evidence_dependencies(self) -> None:
        graph = build_default_pipeline()
        self.assertEqual(
            graph.upstream("procedural_instances"),
            (
                "layout_detection",
                "seed_scale_estimation",
                "material_evidence_decision",
                "edge_gradients",
                "reference_edge_probability",
                "edge_traces",
                "surface_darkness_gradients",
                "illumination_decomposition",
                "seed_edge_curves",
                "project",
            ),
        )
        graph.set_status("procedural_instances", NodeStatus.COMPLETE, "Calculated")
        affected = graph.set_parameter(
            "procedural_instances", "minimum_marker_score", 0.13
        )
        self.assertEqual(affected, ("procedural_instances",))
        self.assertEqual(graph.node("procedural_instances").status, NodeStatus.IDLE)

    def test_parameter_batch_is_one_revision_and_one_invalidation(self) -> None:
        graph = build_default_pipeline()
        graph.set_status("procedural_instances", NodeStatus.COMPLETE, "Calculated")
        starting_revision = graph.revision
        original_invalidate = graph.invalidate
        graph.invalidate = Mock(wraps=original_invalidate)

        affected = graph.set_parameters(
            "procedural_instances",
            {
                "marker_count_multiplier": 1.17,
                "minimum_marker_score": 0.19,
            },
        )

        self.assertEqual(affected, ("procedural_instances",))
        self.assertEqual(graph.revision, starting_revision + 1)
        self.assertEqual(
            graph.node("procedural_instances").parameters["marker_count_multiplier"],
            1.17,
        )
        self.assertEqual(
            graph.node("procedural_instances").parameters["minimum_marker_score"],
            0.19,
        )
        graph.invalidate.assert_called_once_with(affected)
        self.assertEqual(graph.node("procedural_instances").status, NodeStatus.IDLE)

        self.assertEqual(
            graph.set_parameters(
                "procedural_instances",
                {
                    "marker_count_multiplier": 1.17,
                    "minimum_marker_score": 0.19,
                },
            ),
            (),
        )
        self.assertEqual(graph.revision, starting_revision + 1)
        self.assertEqual(graph.invalidate.call_count, 1)

    def test_parameter_batch_validation_is_atomic(self) -> None:
        graph = build_default_pipeline()
        graph.set_status("procedural_instances", NodeStatus.COMPLETE, "Calculated")
        node = graph.node("procedural_instances")
        original_parameters = dict(node.parameters)
        starting_revision = graph.revision
        original_invalidate = graph.invalidate
        graph.invalidate = Mock(wraps=original_invalidate)

        with self.assertRaisesRegex(ValueError, "at most"):
            graph.set_parameters(
                "procedural_instances",
                {
                    "marker_count_multiplier": 1.17,
                    "minimum_marker_score": 2.0,
                },
            )

        self.assertEqual(node.parameters, original_parameters)
        self.assertEqual(node.status, NodeStatus.COMPLETE)
        self.assertEqual(graph.revision, starting_revision)
        graph.invalidate.assert_not_called()

    def test_procedural_marker_count_multiplier_is_exposed(self) -> None:
        from seedvision.segmentation.procedural import ProceduralInstanceSettings

        node = build_default_pipeline().node("procedural_instances")
        spec = next(
            item
            for item in node.parameter_specs
            if item.key == "marker_count_multiplier"
        )
        self.assertEqual(node.parameters["marker_count_multiplier"], 1.02)
        self.assertEqual(spec.kind, "float")
        self.assertLessEqual(spec.minimum, 0.70)
        self.assertGreaterEqual(spec.maximum, 1.40)
        self.assertEqual(
            ProceduralInstanceSettings(**node.parameters).marker_count_multiplier,
            1.02,
        )

    def test_calibration_nodes_feed_the_corrected_analysis_path(self) -> None:
        graph = build_default_pipeline()
        self.assertEqual(graph.upstream("metadata"), ("project",))
        self.assertEqual(
            graph.upstream("deskew_colour"),
            ("project",),
        )
        self.assertEqual(graph.upstream("ruler_detection"), ("deskew_colour",))
        self.assertEqual(
            graph.upstream("layout_detection"),
            ("deskew_colour", "ruler_detection"),
        )
        self.assertNotIn("identification", graph.nodes)
        self.assertIn("identification", graph.unused_nodes)
        self.assertEqual(
            graph.upstream("seed_scale_estimation"),
            (
                "deskew_colour", "layout_detection", "project",
                "species_reference_library",
            ),
        )
        self.assertEqual(
            graph.upstream("background_likelihood"),
            (
                "deskew_colour",
                "species_reference_library",
                "layout_detection",
                "seed_scale_estimation",
                "ruler_detection",
                "project",
            ),
        )
        affected = graph.set_parameter(
            "ruler_detection", "ruler_length_mm", 200.0
        )
        self.assertNotIn("deskew_colour", affected)
        self.assertIn("ruler_detection", affected)
        self.assertIn("layout_detection", affected)
        self.assertNotIn("identification", affected)
        self.assertIn("background_likelihood", affected)
        self.assertIn("refined_background_likelihood", affected)
        self.assertNotIn("circle_candidates", affected)
        self.assertNotIn("output", affected)

    def test_gradient_method_change_invalidates_only_gradient_descendants(self) -> None:
        graph = build_default_pipeline()
        for node_id in graph.nodes:
            graph.node(node_id).status = NodeStatus.COMPLETE

        affected = graph.set_parameter(
            "edge_gradients", "edge_gradient_method", "prewitt"
        )

        self.assertEqual(affected[0], "edge_gradients")
        self.assertIn("reference_edge_probability", affected)
        self.assertIn("edge_traces", affected)
        self.assertIn("procedural_instances", affected)
        for upstream_id in (
            "project",
            "metadata",
            "deskew_colour",
            "ruler_detection",
            "layout_detection",
            "wavelet_decomposition",
            "background_likelihood",
            "refined_background_likelihood",
        ):
            self.assertNotIn(upstream_id, affected)
            self.assertIs(graph.node(upstream_id).status, NodeStatus.COMPLETE)

    def test_circle_candidates_are_preserved_in_the_unused_node_toolbox(self) -> None:
        graph = build_default_pipeline()
        self.assertNotIn("circle_candidates", graph.nodes)
        self.assertIn("circle_candidates", graph.unused_nodes)
        self.assertEqual(graph.node("circle_candidates").title, "Circle candidates")
        self.assertEqual(len(graph.unused_nodes), 19)
        self.assertEqual(len(graph.unused_connections), 78)
        restored = graph.restore_unused_node("circle_candidates")
        self.assertEqual(len(restored), 8)
        self.assertIn("circle_candidates", graph.nodes)
        self.assertNotIn("circle_candidates", graph.unused_nodes)
        self.assertEqual(graph.upstream("circle_candidates"), (
            "deskew_colour",
            "layout_detection",
            "seed_scale_estimation",
            "edge_gradients",
            "image_quality",
            "illumination_decomposition",
        ))

    def test_surface_darkness_is_active_while_threshold_ceiling_diagnostics_are_optional(self) -> None:
        graph = build_default_pipeline()
        ceilings = {
            "lightening_gradient_ceiling",
            "darkening_gradient_ceiling",
        }
        self.assertIn("surface_darkness_gradients", graph.nodes)
        self.assertTrue(graph.node("surface_darkness_gradients").enabled)
        self.assertTrue(ceilings.issubset(graph.unused_nodes))
        self.assertTrue(ceilings.isdisjoint(graph.nodes))
        self.assertTrue(all(not graph.node(node_id).enabled for node_id in ceilings))
        self.assertTrue(
            all(graph.node(node_id).status is NodeStatus.BYPASSED for node_id in ceilings)
        )
        preserved = {
            connection
            for connection in graph.unused_connections
            if connection.source in ceilings or connection.target in ceilings
        }
        self.assertEqual(len(preserved), 4)

        restored = graph.restore_unused_node("lightening_gradient_ceiling")
        self.assertEqual(len(restored), 2)
        self.assertFalse(graph.node("lightening_gradient_ceiling").enabled)

    def test_calibration_residual_risk_is_preserved_in_unused_toolbox(self) -> None:
        graph = build_default_pipeline()
        self.assertNotIn("calibration_residuals", graph.nodes)
        self.assertIn("calibration_residuals", graph.unused_nodes)
        node = graph.node("calibration_residuals")
        self.assertEqual(node.title, "Calibration residual risk")
        self.assertFalse(node.enabled)
        self.assertIs(node.status, NodeStatus.BYPASSED)

        restored = graph.restore_unused_node("calibration_residuals")
        self.assertEqual(len(restored), 4)
        self.assertEqual(
            graph.upstream("calibration_residuals"),
            (
                "deskew_colour",
                "ruler_detection",
                "layout_detection",
            ),
        )
        self.assertFalse(node.enabled)

    def test_pipeline_configuration_is_json_serializable(self) -> None:
        payload = build_default_pipeline().to_dict()
        encoded = json.dumps(payload)
        self.assertIn("Seed identification", encoded)
        self.assertIn("Circle candidates", encoded)
        self.assertIn("InstanceProposals", encoded)
        self.assertIn("calculation_seconds", payload["nodes"][0])
        self.assertEqual(len(payload["unused_nodes"]), 19)

    def test_project_is_the_only_root_and_supplies_image_and_annotations(self) -> None:
        graph = build_default_pipeline()
        node = graph.node("project")
        self.assertEqual(node.category, "Project")
        self.assertEqual(node.title, "Project")
        self.assertNotIn("raw_images", graph.nodes)
        self.assertNotIn("reference_layers", graph.nodes)
        self.assertNotIn("manual_seed_centres", graph.nodes)
        self.assertNotIn("foreground_segmentation", graph.nodes)
        self.assertEqual(
            tuple(port for port, _label in node.output_ports),
            ("raw_image", "annotations", "species_library"),
        )
        self.assertEqual(
            node.output_port_types,
            {
                "raw_image": "RawImage",
                "annotations": "ImageAnnotations",
                "species_library": "SpeciesLibraryPin",
            },
        )
        self.assertEqual(
            [node_id for node_id in graph.nodes if not graph.upstream(node_id)],
            ["project"],
        )
        self.assertEqual(
            set(graph.downstream_from_port("project", "annotations")),
            {
                "background_likelihood",
                "refined_background_likelihood",
                "seed_scale_estimation",
                "reference_texture_prototypes",
                "reference_seed_traits",
                "procedural_instances",
            },
        )
        self.assertEqual(
            {
                connection.source_port
                for connection in graph.connections
                if connection.source == "project"
            },
            {"raw_image", "annotations", "species_library"},
        )
        self.assertEqual(
            set(graph.downstream_from_port("project", "raw_image")),
            {"metadata", "deskew_colour"},
        )
        for node_id in tuple(graph.unused_nodes):
            graph.restore_unused_node(node_id)
        self.assertEqual(
            set(graph.downstream("project", recursive=True)),
            set(graph.nodes) - {"project"},
        )
        affected = {
            "project",
            *graph.downstream("project", recursive=True),
        }
        self.assertTrue(
            {
                "background_likelihood",
                "refined_background_likelihood",
                "reference_edge_probability",
                "procedural_instances",
            }.issubset(affected)
        )

    def test_unfinished_candidate_and_mask_branches_default_to_disabled(self) -> None:
        graph = build_default_pipeline()
        roots = {"boundary_normals"}
        expected = set(roots)
        for root in roots:
            expected.update(graph.downstream(root, recursive=True))
        expected.update({"unet_instances", "stardist_instances"})
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
        self.assertTrue(graph.node("surface_darkness_gradients").enabled)
        self.assertFalse(graph.node("unet_instances").enabled)
        self.assertFalse(graph.node("stardist_instances").enabled)
        self.assertIn("surface_darkness_gradients", graph.nodes)

    def test_disabling_a_node_disables_all_dependents(self) -> None:
        graph = build_default_pipeline()
        affected = graph.set_parameter(
            "background_likelihood", "background_colour_enabled", False
        )
        self.assertFalse(
            graph.node("background_likelihood").parameters[
                "background_colour_enabled"
            ]
        )
        self.assertTrue(graph.node("background_likelihood").enabled)
        self.assertTrue(graph.node("refined_background_likelihood").enabled)
        self.assertIn("refined_background_likelihood", affected)

    def test_combined_noise_node_keeps_independent_enable_controls(self) -> None:
        graph = build_default_pipeline()
        affected = graph.set_parameter(
            "refined_background_likelihood", "foreground_noise_enabled", False
        )
        node = graph.node("refined_background_likelihood")
        self.assertTrue(node.parameters["background_noise_enabled"])
        self.assertFalse(node.parameters["foreground_noise_enabled"])
        self.assertTrue(node.enabled)
        self.assertIn("material_evidence_decision", affected)

    def test_reset_parameters_restores_authored_defaults(self) -> None:
        graph = build_default_pipeline()
        original = graph.node("background_likelihood").parameters[
            "foreground_reference_weight"
        ]
        graph.set_parameter(
            "background_likelihood", "foreground_reference_weight", 0.25
        )
        affected = graph.reset_parameters("background_likelihood")
        self.assertEqual(
            graph.node("background_likelihood").parameters[
                "foreground_reference_weight"
            ],
            original,
        )
        self.assertIn("background_likelihood", affected)
        self.assertEqual(graph.reset_parameters("background_likelihood"), ())

    def test_reference_colour_capacity_controls_share_honest_naming(self) -> None:
        graph = build_default_pipeline()
        background = graph.node("background_likelihood")
        foreground_spec = next(
            spec
            for spec in background.parameter_specs
            if spec.key == "foreground_reference_components"
        )
        background_spec = next(
            spec
            for spec in background.parameter_specs
            if spec.key == "background_colour_components"
        )

        self.assertEqual(
            foreground_spec.label,
            "Maximum reference colour modes",
        )
        self.assertEqual(background_spec.label, foreground_spec.label)
        self.assertEqual(background.parameters[background_spec.key], 32)
        self.assertEqual(background.default_parameters[background_spec.key], 32)
        self.assertEqual(background_spec.minimum, 1)
        self.assertEqual(background_spec.maximum, 256)
        self.assertEqual(background_spec.step, 4)

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

    def test_every_configurable_node_has_complete_operation_ordered_sections(self) -> None:
        graph = build_default_pipeline()
        for node in (*graph.nodes.values(), *graph.unused_nodes.values()):
            keys = tuple(spec.key for spec in node.parameter_specs)
            section_keys = tuple(
                key for section in node.parameter_sections for key in section.keys
            )
            if not keys:
                self.assertEqual(node.parameter_sections, (), node.identifier)
                continue
            self.assertTrue(node.parameter_sections, node.identifier)
            self.assertTrue(
                all(section.title.strip() for section in node.parameter_sections),
                node.identifier,
            )
            self.assertEqual(len(section_keys), len(set(section_keys)), node.identifier)
            self.assertEqual(set(section_keys), set(keys), node.identifier)
            self.assertTrue(
                all(spec.label.strip() and spec.description.strip() for spec in node.parameter_specs),
                node.identifier,
            )

        self.assertEqual(
            tuple(
                section.title
                for section in graph.node("reference_edge_probability").parameter_sections
            ),
            (
                "Reference sources",
                "Optional conservative net evidence",
                "Training-example selection",
                "Strip descriptor geometry and resolution",
                "Edge prototype fitting and matching",
                "Supported ridge extraction",
                "Reference-edge support normalization",
            ),
        )
        self.assertEqual(
            tuple(
                section.title
                for section in graph.node("procedural_instances").parameter_sections
            ),
            (
                "Working raster and dish region",
                "Seed-material occupancy",
                "Boundary evidence",
                "Centre likelihood",
                "Automatic marker selection",
                "Candidate geometry limits",
                "Candidate search and combination",
                "Reference matching and fitting costs",
            ),
        )

    def test_every_computational_setting_is_exposed_or_explicitly_private(self) -> None:
        from dataclasses import fields

        from seedvision.calibration.geometry import DishDetectionSettings
        from seedvision.calibration.image import CalibrationSettings
        from seedvision.learning.pipeline import (
            StarDistPipelineSettings,
            UNetPipelineSettings,
        )
        from seedvision.segmentation.baseline import BaselineSettings
        from seedvision.segmentation.procedural import ProceduralInstanceSettings
        from seedvision.visualization.advanced import AdvancedAnalysisSettings
        from seedvision.visualization.layers import AnalysisLayerSettings

        graph = build_default_pipeline()
        exposed = {
            spec.key
            for node in (*graph.nodes.values(), *graph.unused_nodes.values())
            for spec in node.parameter_specs
        }
        settings_types = (
            BaselineSettings,
            AnalysisLayerSettings,
            AdvancedAnalysisSettings,
            ProceduralInstanceSettings,
            UNetPipelineSettings,
            StarDistPipelineSettings,
            CalibrationSettings,
            DishDetectionSettings,
        )
        computational = {
            item.name for settings_type in settings_types for item in fields(settings_type)
        }
        hidden = computational - exposed
        self.assertEqual(hidden, set(INTENTIONALLY_UNEXPOSED_SETTINGS))
        self.assertTrue(all(INTENTIONALLY_UNEXPOSED_SETTINGS.values()))

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

    def test_oriented_trace_convexity_controls_are_explicit_and_consumed(self) -> None:
        graph = build_default_pipeline()
        node = graph.node("edge_traces")
        specs = {spec.key: spec for spec in node.parameter_specs}

        self.assertEqual(node.parameters["trace_curvature_policy"], "prefer")
        self.assertEqual(
            specs["trace_curvature_policy"].choices,
            ("off", "prefer", "require"),
        )
        self.assertEqual(
            node.parameters["trace_curvature_tolerance_degrees"], 6.0
        )
        affected = graph.set_parameter(
            "edge_traces", "trace_curvature_policy", "require"
        )
        self.assertIn("edge_traces", affected)
        self.assertTrue(
            any(
                edge.source == "edge_traces"
                and edge.target == "seed_edge_curves"
                for edge in graph.connection_templates
            )
        )
        with self.assertRaises(ValueError):
            graph.set_parameter(
                "edge_traces", "trace_curvature_policy", "globally convex"
            )

    def test_foreground_noise_first_tertile_is_default_and_exposed(self) -> None:
        graph = build_default_pipeline()
        noise = graph.node("refined_background_likelihood")
        spec = next(
            item
            for item in noise.parameter_specs
            if item.key == "foreground_noise_direction_integration"
        )

        self.assertEqual(
            noise.parameters["foreground_noise_direction_integration"],
            "1st tertile",
        )
        self.assertEqual(
            noise.default_parameters[
                "foreground_noise_direction_integration"
            ],
            "1st tertile",
        )
        self.assertIn("1st tertile", spec.choices)
        background_spec = next(
            item
            for item in noise.parameter_specs
            if item.key == "noise_direction_integration"
        )
        self.assertNotIn("1st tertile", background_spec.choices)
        self.assertEqual(
            noise.parameters["noise_direction_integration"], "maximum"
        )
        affected = graph.set_parameter(
            "refined_background_likelihood",
            "foreground_noise_direction_integration",
            "median",
        )
        self.assertEqual(affected[0], "refined_background_likelihood")
        self.assertIn("boundary_normals", affected)
        self.assertIn("procedural_instances", affected)
        with self.assertRaises(ValueError):
            graph.set_parameter(
                "refined_background_likelihood",
                "foreground_noise_direction_integration",
                "lower-ish",
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

    def test_cancellation_stops_at_the_next_timed_node_boundary(self) -> None:
        from seedvision.timing import AnalysisCancelled, NodeTimingRecorder

        cancelled = False
        recorder = NodeTimingRecorder(
            cancellation_requested=lambda: cancelled
        )
        span = recorder.start("first")
        cancelled = True
        with self.assertRaises(AnalysisCancelled):
            recorder.stop(span)
        with self.assertRaises(AnalysisCancelled):
            recorder.start("second")


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
            BaselineSettings,
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
            AnalysisLayerSettings(
                reference_ridge_low_threshold=0.50,
                reference_ridge_high_threshold=0.40,
            )
        with self.assertRaises(ValueError):
            AnalysisLayerSettings(trace_curvature_policy="sometimes")
        with self.assertRaises(ValueError):
            AnalysisLayerSettings(trace_curvature_tolerance_degrees=46.0)
        self.assertEqual(
            AnalysisLayerSettings().foreground_noise_direction_integration,
            "1st tertile",
        )
        self.assertTrue(
            AnalysisLayerSettings().background_keep_perimeter_reference
        )
        self.assertTrue(
            BaselineSettings().foreground_include_annotated_seed_instances
        )
        self.assertTrue(
            build_default_pipeline().node("background_likelihood").parameters[
                "foreground_include_annotated_seed_instances"
            ]
        )
        self.assertEqual(
            AnalysisLayerSettings(
                foreground_noise_direction_integration="1st tertile"
            ).foreground_noise_direction_integration,
            "1st tertile",
        )
        with self.assertRaises(ValueError):
            AnalysisLayerSettings(
                foreground_noise_direction_integration="lower-ish"
            )
        with self.assertRaises(ValueError):
            AdvancedAnalysisSettings(compute_device="metal")

    def test_background_reference_colour_capacity_default_and_range(self) -> None:
        from seedvision.segmentation import AnalysisLayerSettings

        self.assertEqual(
            AnalysisLayerSettings().background_colour_components,
            32,
        )
        self.assertEqual(
            AnalysisLayerSettings(
                background_colour_components=256
            ).background_colour_components,
            256,
        )
        with self.assertRaises(ValueError):
            AnalysisLayerSettings(background_colour_components=257)

    def test_reference_texture_material_and_edge_capacities_are_independent(self) -> None:
        from seedvision.segmentation import AnalysisLayerSettings

        defaults = AnalysisLayerSettings()
        self.assertEqual(
            defaults.reference_texture_material_prototypes_per_class,
            64,
        )
        self.assertEqual(
            defaults.reference_texture_edge_prototypes_per_class,
            256,
        )
        self.assertEqual(defaults.reference_texture_class_contrast, 4.0)
        self.assertEqual(defaults.reference_edge_class_contrast, 4.0)
        graph = build_default_pipeline()
        material_node = graph.node("reference_texture_prototypes")
        edge_node = graph.node("reference_edge_probability")
        self.assertEqual(
            dict(edge_node.output_ports)["interior_direction"],
            "Physical-edge predicted interior direction",
        )
        self.assertEqual(
            dict(edge_node.output_ports)["excess_comparison"],
            "Physical vs non-physical prototype excess",
        )
        self.assertEqual(
            edge_node.output_port_types["interior_direction"],
            "PhysicalEdgeInteriorDirection",
        )
        self.assertEqual(
            edge_node.output_port_types["excess_comparison"],
            "ReferenceEdgeExcessComparison",
        )
        material_specs = {spec.key: spec for spec in material_node.parameter_specs}
        edge_specs = {spec.key: spec for spec in edge_node.parameter_specs}
        self.assertEqual(
            material_node.parameters["reference_texture_material_prototypes_per_class"],
            64,
        )
        self.assertEqual(
            edge_node.parameters["reference_texture_edge_prototypes_per_class"],
            256,
        )
        self.assertEqual(
            material_specs["reference_texture_material_prototypes_per_class"].maximum,
            256,
        )
        self.assertEqual(
            edge_specs["reference_texture_edge_prototypes_per_class"].maximum,
            1024,
        )
        self.assertEqual(
            material_specs["reference_texture_class_contrast"].label,
            "Material class contrast",
        )
        self.assertEqual(
            edge_specs["reference_edge_class_contrast"].label,
            "Edge class contrast",
        )
        graph.set_parameter(
            "reference_texture_prototypes",
            "reference_texture_material_prototypes_per_class",
            72,
        )
        graph.set_parameter(
            "reference_edge_probability",
            "reference_texture_edge_prototypes_per_class",
            320,
        )
        graph.reset_parameters("reference_texture_prototypes")
        graph.reset_parameters("reference_edge_probability")
        self.assertEqual(
            material_node.parameters["reference_texture_material_prototypes_per_class"],
            64,
        )
        self.assertEqual(
            edge_node.parameters["reference_texture_edge_prototypes_per_class"],
            256,
        )
        with self.assertRaises(ValueError):
            AnalysisLayerSettings(
                reference_texture_material_prototypes_per_class=264
            )
        with self.assertRaises(ValueError):
            AnalysisLayerSettings(
                reference_texture_edge_prototypes_per_class=1040
            )
        with self.assertRaises(ValueError):
            AnalysisLayerSettings(reference_texture_class_contrast=0.5)
        with self.assertRaises(ValueError):
            AnalysisLayerSettings(reference_edge_class_contrast=12.5)

    def test_net_physical_edge_internal_subtraction_control(self) -> None:
        from seedvision.segmentation import AnalysisLayerSettings

        defaults = AnalysisLayerSettings()
        self.assertEqual(defaults.net_physical_edge_internal_scale, 0.5)
        with self.assertRaises(ValueError):
            AnalysisLayerSettings(net_physical_edge_internal_scale=-0.01)
        with self.assertRaises(ValueError):
            AnalysisLayerSettings(net_physical_edge_internal_scale=2.01)

        graph = build_default_pipeline()
        node = graph.node("reference_edge_probability")
        specs = {spec.key: spec for spec in node.parameter_specs}
        self.assertEqual(node.parameters["net_physical_edge_internal_scale"], 0.5)
        self.assertEqual(specs["net_physical_edge_internal_scale"].minimum, 0.0)
        self.assertEqual(specs["net_physical_edge_internal_scale"].maximum, 2.0)
        self.assertFalse(specs["net_physical_edge_internal_scale"].display_only)
        graph.node("reference_edge_probability").status = NodeStatus.COMPLETE
        graph.node("procedural_instances").status = NodeStatus.COMPLETE
        revision = graph.revision
        affected = graph.set_parameter(
            "reference_edge_probability",
            "net_physical_edge_internal_scale",
            1.25,
        )
        self.assertEqual(
            affected,
            (
                "reference_edge_probability",
                "edge_traces",
                "unet_instances",
                "stardist_instances",
                "seed_edge_curves",
                "procedural_instances",
            ),
        )
        self.assertEqual(graph.revision, revision + 1)
        self.assertEqual(node.status, NodeStatus.IDLE)
        self.assertEqual(graph.node("procedural_instances").status, NodeStatus.IDLE)
        self.assertEqual(
            graph.reset_parameters("reference_edge_probability"),
            affected,
        )
        self.assertEqual(graph.revision, revision + 2)
        self.assertEqual(node.status, NodeStatus.IDLE)
        self.assertEqual(node.parameters["net_physical_edge_internal_scale"], 0.5)

    def test_local_edge_normalization_controls_are_postclassification(self) -> None:
        from seedvision.segmentation import AnalysisLayerSettings

        defaults = AnalysisLayerSettings()
        self.assertEqual(
            defaults.reference_edge_normalization_radius_fraction, 0.30
        )
        self.assertEqual(
            defaults.reference_edge_normalization_target_support, 0.35
        )
        self.assertEqual(
            defaults.reference_edge_normalization_maximum_gain, 2.50
        )
        self.assertEqual(
            defaults.reference_edge_normalization_absolute_floor, 0.04
        )
        with self.assertRaises(ValueError):
            AnalysisLayerSettings(
                reference_edge_normalization_radius_fraction=0.01
            )
        with self.assertRaises(ValueError):
            AnalysisLayerSettings(reference_edge_normalization_maximum_gain=0.9)

        graph = build_default_pipeline()
        ridge_node = graph.node("reference_edge_probability")
        self.assertEqual(
            ridge_node.parameters["reference_edge_normalization_maximum_gain"],
            2.50,
        )
        affected = graph.set_parameter(
            "reference_edge_probability",
            "reference_edge_normalization_maximum_gain",
            3.0,
        )
        self.assertEqual(
            affected,
            (
                "reference_edge_probability",
                "edge_traces",
                "unet_instances",
                "stardist_instances",
                "seed_edge_curves",
                "procedural_instances",
            ),
        )

    def test_oriented_trace_ridge_source_is_explicit_and_computational(self) -> None:
        from seedvision.segmentation import AnalysisLayerSettings

        self.assertEqual(AnalysisLayerSettings().trace_edge_source, "generic_ridges")
        with self.assertRaises(ValueError):
            AnalysisLayerSettings(trace_edge_source="mystery_edges")
        graph = build_default_pipeline()
        node = graph.node("edge_traces")
        self.assertEqual(node.parameters["trace_edge_source"], "generic_ridges")
        spec = next(
            spec for spec in node.parameter_specs if spec.key == "trace_edge_source"
        )
        self.assertEqual(
            spec.choices,
            (
                "generic_ridges",
                "reference_ridges",
                "net_reference_ridges",
                "normalized_net_reference_ridges",
            ),
        )
        self.assertEqual(
            graph.set_parameter(
                "edge_traces", "trace_edge_source", "reference_ridges"
            ),
            ("edge_traces", "seed_edge_curves", "procedural_instances"),
        )


if __name__ == "__main__":
    unittest.main()
