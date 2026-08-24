from __future__ import annotations

from copy import deepcopy
import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Callable
from unittest.mock import Mock, patch

from seedvision.persistence import (
    ANALYSIS_SETTINGS_FORMAT,
    ANALYSIS_SETTINGS_VERSION,
    AnalysisSettingsIOError,
    IncompatibleAnalysisSettingsProfile,
    InvalidAnalysisSettingsProfile,
    analysis_settings_profile_from_graph,
    analysis_settings_profile_from_payload,
    analysis_settings_profile_to_payload,
    apply_analysis_settings_profile,
    load_analysis_settings_profile,
    load_and_apply_analysis_settings_profile,
    save_analysis_settings_profile,
)
from seedvision.pipeline import NodeStatus, build_default_pipeline


def _node_payload(payload: dict[str, object], node_id: str) -> dict[str, object]:
    nodes = payload["nodes"]
    assert isinstance(nodes, list)
    return next(item for item in nodes if item["id"] == node_id)


class AnalysisSettingsPersistenceTests(unittest.TestCase):
    def test_json_round_trip_is_human_readable_and_analysis_only(self) -> None:
        graph = build_default_pipeline()
        graph.set_parameter("ruler_detection", "ruler_length_mm", 175.0)
        node = graph.node("ruler_detection")
        node.x = 876.5
        node.y = -42.25
        node.status = NodeStatus.COMPLETE
        node.status_detail = "runtime detail that must not be persisted"
        node.calculation_seconds = 1.25

        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "laboratory.seedfiddle-settings.json"
            destination.write_text("obsolete contents", encoding="utf-8")
            saved = save_analysis_settings_profile(destination, graph)
            loaded = load_analysis_settings_profile(saved)

            self.assertEqual(
                analysis_settings_profile_to_payload(loaded),
                analysis_settings_profile_to_payload(
                    analysis_settings_profile_from_graph(graph)
                ),
            )
            text = destination.read_text(encoding="utf-8")
            self.assertTrue(text.endswith("\n"))
            self.assertIn(f'"format": "{ANALYSIS_SETTINGS_FORMAT}"', text)
            self.assertIn(f'"version": {ANALYSIS_SETTINGS_VERSION}', text)
            self.assertNotIn("calculation_seconds", text)
            self.assertNotIn("runtime detail", text)
            self.assertNotIn('"position"', text)
            self.assertFalse(
                tuple(destination.parent.glob(f".{destination.name}.*.tmp"))
            )

    def test_apply_restores_parameters_topology_enabled_and_suspended_once(self) -> None:
        source = build_default_pipeline()
        source.restore_unused_node("circle_candidates")
        source.set_parameter("ruler_detection", "ruler_length_mm", 183.0)
        source.set_parameter(
            "reference_edge_probability",
            "net_physical_edge_internal_scale",
            0.8,
        )
        disconnected = next(
            connection
            for connection in source.connections
            if connection.source == "reference_layers"
            and connection.source_port == "centres"
            and connection.target == "procedural_instances"
        )
        source.disconnect(disconnected)
        profile = analysis_settings_profile_from_graph(source)

        target = build_default_pipeline()
        target.node("ruler_detection").x = 901.0
        target.node("ruler_detection").y = 902.0
        target.node("circle_candidates").x = 777.0
        target.node("circle_candidates").y = 778.0
        for node in target.nodes.values():
            node.status = NodeStatus.COMPLETE
            node.status_detail = "cached"
        starting_revision = target.revision
        original_invalidate = target.invalidate
        target.invalidate = Mock(wraps=original_invalidate)

        result = apply_analysis_settings_profile(target, profile)

        self.assertTrue(result.changed)
        self.assertTrue(result.structure_changed)
        self.assertTrue(result.analytical_changed)
        self.assertEqual(target.revision, starting_revision + 1)
        target.invalidate.assert_called_once()
        self.assertIn("circle_candidates", target.nodes)
        self.assertNotIn(disconnected, target.connections)
        self.assertNotIn(disconnected, target.unused_connections)
        self.assertIn("procedural_instances", target._connection_suspended)
        self.assertEqual(target.node("ruler_detection").x, 901.0)
        self.assertEqual(target.node("ruler_detection").y, 902.0)
        self.assertEqual(target.node("circle_candidates").x, 777.0)
        self.assertEqual(target.node("circle_candidates").y, 778.0)
        self.assertEqual(
            analysis_settings_profile_to_payload(
                analysis_settings_profile_from_graph(target)
            ),
            analysis_settings_profile_to_payload(profile),
        )

    def test_no_op_is_neutral_and_net_ridge_control_is_analytical(self) -> None:
        graph = build_default_pipeline()
        profile = analysis_settings_profile_from_graph(graph)
        graph.node("reference_edge_probability").status = NodeStatus.COMPLETE
        graph.node("procedural_instances").status = NodeStatus.COMPLETE
        graph.node("reference_edge_probability").x = 321.0
        revision = graph.revision

        no_op = apply_analysis_settings_profile(graph, profile)
        self.assertFalse(no_op.changed)
        self.assertEqual(graph.revision, revision)
        self.assertEqual(
            graph.node("reference_edge_probability").status, NodeStatus.COMPLETE
        )

        payload = analysis_settings_profile_to_payload(profile)
        edge_node = _node_payload(payload, "reference_edge_probability")
        parameters = edge_node["parameters"]
        assert isinstance(parameters, dict)
        parameters["net_physical_edge_internal_scale"] = 1.25
        changed_profile = analysis_settings_profile_from_payload(payload)
        result = apply_analysis_settings_profile(graph, changed_profile)

        self.assertTrue(result.changed)
        self.assertFalse(result.structure_changed)
        self.assertTrue(result.analytical_changed)
        self.assertEqual(
            result.affected_node_ids,
            (
                "reference_edge_probability",
                "reference_edge_ridges",
                "edge_traces",
                "procedural_instances",
                "unet_instances",
                "stardist_instances",
            ),
        )
        self.assertEqual(graph.revision, revision + 1)
        self.assertEqual(
            graph.node("reference_edge_probability").status, NodeStatus.IDLE
        )
        self.assertEqual(graph.node("procedural_instances").status, NodeStatus.IDLE)
        self.assertEqual(graph.node("reference_edge_probability").x, 321.0)

    def test_graph_compatibility_is_fully_validated_before_mutation(self) -> None:
        graph = build_default_pipeline()
        graph.node("ruler_detection").status = NodeStatus.COMPLETE
        original_profile = analysis_settings_profile_from_graph(graph)
        original_payload = analysis_settings_profile_to_payload(original_profile)
        payload = deepcopy(original_payload)
        ruler = _node_payload(payload, "ruler_detection")
        ruler_parameters = ruler["parameters"]
        assert isinstance(ruler_parameters, dict)
        ruler_parameters["ruler_length_mm"] = 200.0
        procedural = _node_payload(payload, "procedural_instances")
        procedural_parameters = procedural["parameters"]
        assert isinstance(procedural_parameters, dict)
        procedural_parameters["minimum_marker_score"] = 3.0
        profile = analysis_settings_profile_from_payload(payload)
        revision = graph.revision
        suspended = set(graph._connection_suspended)

        with self.assertRaisesRegex(
            IncompatibleAnalysisSettingsProfile, "minimum_marker_score"
        ):
            apply_analysis_settings_profile(graph, profile)

        self.assertEqual(graph.revision, revision)
        self.assertEqual(graph._connection_suspended, suspended)
        self.assertEqual(graph.node("ruler_detection").status, NodeStatus.COMPLETE)
        self.assertEqual(
            analysis_settings_profile_to_payload(
                analysis_settings_profile_from_graph(graph)
            ),
            original_payload,
        )

    def test_version_two_profile_adopts_new_candidate_geometry_defaults(self) -> None:
        graph = build_default_pipeline()
        payload = analysis_settings_profile_to_payload(
            analysis_settings_profile_from_graph(graph)
        )
        payload["version"] = 2
        procedural = _node_payload(payload, "procedural_instances")
        parameters = procedural["parameters"]
        assert isinstance(parameters, dict)
        new_keys = {
            "soft_minimum_instance_area_fraction",
            "soft_maximum_instance_width_fraction",
            "hard_maximum_instance_width_fraction",
            "maximum_internal_concavity_fraction",
            "maximum_protrusion_area_fraction",
            "candidate_hypotheses_per_marker",
            "candidate_overlap_fraction",
        }
        for key in new_keys:
            parameters.pop(key)

        target = build_default_pipeline()
        apply_analysis_settings_profile(
            target, analysis_settings_profile_from_payload(payload)
        )

        for key in new_keys:
            self.assertEqual(
                target.node("procedural_instances").parameters[key],
                graph.node("procedural_instances").parameters[key],
            )

    def test_version_four_profile_migrates_combined_annotation_and_colour_nodes(self) -> None:
        source = build_default_pipeline()
        payload = analysis_settings_profile_to_payload(
            analysis_settings_profile_from_graph(source)
        )
        payload["version"] = 4
        background = _node_payload(payload, "background_likelihood")
        combined_parameters = dict(background["parameters"])
        foreground_parameters = {
            key: value
            for key, value in combined_parameters.items()
            if key.startswith("foreground_") or key == "inner_radius_fraction"
        }
        background["parameters"] = {
            key: value
            for key, value in combined_parameters.items()
            if key not in foreground_parameters
            and key != "background_colour_enabled"
        }
        background["enabled"] = False
        payload["nodes"].extend(
            (
                {
                    "id": "foreground_segmentation",
                    "active": True,
                    "enabled": True,
                    "connection_suspended": False,
                    "parameters": foreground_parameters,
                },
                {
                    "id": "manual_seed_centres",
                    "active": True,
                    "enabled": True,
                    "connection_suspended": False,
                    "parameters": {},
                },
            )
        )
        foreground_ports = {
            "foreground_probability",
            "annotated_foreground_reference_source",
            "proposal_region",
        }
        for connection in payload["connections"]:
            if (
                connection["source"] == "background_likelihood"
                and connection["source_port"] in foreground_ports
            ):
                connection["source"] = "foreground_segmentation"
            elif (
                connection["source"] == "reference_layers"
                and connection["source_port"] == "centres"
            ):
                connection["source"] = "manual_seed_centres"

        target = build_default_pipeline()
        apply_analysis_settings_profile(
            target, analysis_settings_profile_from_payload(payload)
        )

        combined = target.node("background_likelihood")
        self.assertTrue(combined.enabled)
        self.assertFalse(combined.parameters["background_colour_enabled"])
        self.assertEqual(
            combined.parameters["foreground_reference_weight"],
            source.node("background_likelihood").parameters[
                "foreground_reference_weight"
            ],
        )
        centre_connection = target.connection_for_input(
            "procedural_instances", "manual_centres"
        )
        self.assertIsNotNone(centre_connection)
        self.assertEqual(
            (centre_connection.source, centre_connection.source_port),
            ("reference_layers", "centres"),
        )

    def test_missing_and_unknown_catalogue_members_fail_clearly(self) -> None:
        cases: list[tuple[str, Callable[[dict[str, object]], None]]] = []

        def missing_node(payload: dict[str, object]) -> None:
            nodes = payload["nodes"]
            assert isinstance(nodes, list)
            nodes.pop()

        cases.append(("missing nodes", missing_node))

        def unknown_node(payload: dict[str, object]) -> None:
            nodes = payload["nodes"]
            assert isinstance(nodes, list)
            extra = deepcopy(nodes[0])
            extra["id"] = "future_unknown_node"
            nodes.append(extra)

        cases.append(("unknown nodes", unknown_node))

        def missing_parameter(payload: dict[str, object]) -> None:
            parameters = _node_payload(payload, "ruler_detection")["parameters"]
            assert isinstance(parameters, dict)
            parameters.pop("ruler_length_mm")

        cases.append(("missing parameters", missing_parameter))

        def unknown_parameter(payload: dict[str, object]) -> None:
            parameters = _node_payload(payload, "ruler_detection")["parameters"]
            assert isinstance(parameters, dict)
            parameters["invented_parameter"] = 1

        cases.append(("unknown parameters", unknown_parameter))

        def missing_connection(payload: dict[str, object]) -> None:
            connections = payload["connections"]
            assert isinstance(connections, list)
            connections.pop()

        cases.append(("missing connections", missing_connection))

        def unknown_connection(payload: dict[str, object]) -> None:
            connections = payload["connections"]
            assert isinstance(connections, list)
            extra = deepcopy(connections[0])
            extra["target_port"] = "future_unknown_port"
            connections.append(extra)

        cases.append(("unknown connections", unknown_connection))

        for message, mutate in cases:
            with self.subTest(message=message):
                graph = build_default_pipeline()
                before = analysis_settings_profile_to_payload(
                    analysis_settings_profile_from_graph(graph)
                )
                payload = deepcopy(before)
                mutate(payload)
                profile = analysis_settings_profile_from_payload(payload)
                revision = graph.revision
                with self.assertRaisesRegex(
                    IncompatibleAnalysisSettingsProfile, message
                ):
                    apply_analysis_settings_profile(graph, profile)
                self.assertEqual(graph.revision, revision)
                self.assertEqual(
                    analysis_settings_profile_to_payload(
                        analysis_settings_profile_from_graph(graph)
                    ),
                    before,
                )

    def test_inconsistent_enabled_and_suspended_states_are_rejected(self) -> None:
        graph = build_default_pipeline()
        base = analysis_settings_profile_to_payload(
            analysis_settings_profile_from_graph(graph)
        )

        inactive_enabled = deepcopy(base)
        circle = _node_payload(inactive_enabled, "circle_candidates")
        self.assertFalse(circle["active"])
        circle["enabled"] = True
        with self.assertRaisesRegex(
            IncompatibleAnalysisSettingsProfile, "Toolbox node.*cannot be enabled"
        ):
            apply_analysis_settings_profile(
                graph, analysis_settings_profile_from_payload(inactive_enabled)
            )

        false_suspension = deepcopy(base)
        procedural = _node_payload(false_suspension, "procedural_instances")
        procedural["enabled"] = False
        procedural["connection_suspended"] = True
        with self.assertRaisesRegex(
            IncompatibleAnalysisSettingsProfile, "disconnected required input"
        ):
            apply_analysis_settings_profile(
                graph, analysis_settings_profile_from_payload(false_suspension)
            )

    def test_disconnection_behind_toolbox_node_is_not_mistaken_for_shelving(self) -> None:
        source = build_default_pipeline()
        source.restore_unused_node("circle_candidates")
        disconnected = next(
            connection
            for connection in source.connections
            if connection.target == "circle_candidates"
        )
        still_connected = next(
            connection
            for connection in source.connections
            if connection.target == "circle_candidates"
            and connection != disconnected
        )
        source.disconnect(disconnected)
        source.shelve_node("circle_candidates")
        profile = analysis_settings_profile_from_graph(source)

        target = build_default_pipeline()
        result = apply_analysis_settings_profile(target, profile)
        self.assertTrue(result.changed)
        self.assertNotIn(disconnected, target.unused_connections)
        self.assertIn(still_connected, target.unused_connections)
        target.restore_unused_node("circle_candidates")
        self.assertNotIn(disconnected, target.connections)
        self.assertIn(still_connected, target.connections)

    def test_invalid_json_and_failed_replace_preserve_existing_file_and_graph(self) -> None:
        graph = build_default_pipeline()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            duplicate = root / "duplicate.json"
            duplicate.write_text('{"format":"a","format":"b"}', encoding="utf-8")
            with self.assertRaisesRegex(
                InvalidAnalysisSettingsProfile, "Duplicate JSON object key"
            ):
                load_analysis_settings_profile(duplicate)

            non_finite = root / "non-finite.json"
            non_finite.write_text('{"value": NaN}', encoding="utf-8")
            with self.assertRaisesRegex(
                InvalidAnalysisSettingsProfile, "Non-finite JSON number"
            ):
                load_analysis_settings_profile(non_finite)

            destination = root / "existing.seedfiddle-settings.json"
            destination.write_text("keep this exact file", encoding="utf-8")
            with patch.object(Path, "replace", side_effect=OSError("synthetic")):
                with self.assertRaises(AnalysisSettingsIOError):
                    save_analysis_settings_profile(destination, graph)
            self.assertEqual(
                destination.read_text(encoding="utf-8"), "keep this exact file"
            )

    def test_load_and_apply_failure_does_not_mutate_graph(self) -> None:
        graph = build_default_pipeline()
        payload = analysis_settings_profile_to_payload(
            analysis_settings_profile_from_graph(graph)
        )
        parameters = _node_payload(payload, "ruler_detection")["parameters"]
        assert isinstance(parameters, dict)
        parameters["ruler_length_mm"] = -1.0
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "invalid.seedfiddle-settings.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            revision = graph.revision
            with self.assertRaises(IncompatibleAnalysisSettingsProfile):
                load_and_apply_analysis_settings_profile(path, graph)
            self.assertEqual(graph.revision, revision)
            self.assertEqual(
                graph.node("ruler_detection").parameters["ruler_length_mm"],
                150.0,
            )

    def test_replace_and_cleanup_failures_preserve_the_primary_error(self) -> None:
        graph = build_default_pipeline()
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "existing.seedfiddle-settings.json"
            destination.write_text("keep this exact profile", encoding="utf-8")
            with patch.object(
                Path, "replace", side_effect=OSError("replace failed")
            ), patch.object(
                Path, "unlink", side_effect=OSError("unlink failed")
            ):
                with self.assertRaisesRegex(
                    AnalysisSettingsIOError,
                    "replace failed.*cleanup also failed",
                ):
                    save_analysis_settings_profile(destination, graph)

            self.assertEqual(
                destination.read_text(encoding="utf-8"),
                "keep this exact profile",
            )
            for leftover in destination.parent.glob(f".{destination.name}.*.tmp"):
                leftover.unlink()

    def test_fdopen_failure_closes_owned_descriptor_and_preserves_destination(self) -> None:
        graph = build_default_pipeline()
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "existing.seedfiddle-settings.json"
            destination.write_text("keep this exact profile", encoding="utf-8")
            real_close = os.close
            with patch(
                "seedvision.persistence.analysis_settings.os.fdopen",
                side_effect=OSError("fdopen failed"),
            ), patch(
                "seedvision.persistence.analysis_settings.os.close",
                wraps=real_close,
            ) as close_descriptor:
                with self.assertRaisesRegex(
                    AnalysisSettingsIOError, "fdopen failed"
                ):
                    save_analysis_settings_profile(destination, graph)

            close_descriptor.assert_called_once()
            self.assertEqual(
                destination.read_text(encoding="utf-8"),
                "keep this exact profile",
            )
            self.assertEqual(
                tuple(destination.parent.glob(f".{destination.name}.*.tmp")),
                (),
            )


if __name__ == "__main__":
    unittest.main()
