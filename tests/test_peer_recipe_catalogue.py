"""Focused integrity checks for the audited, non-executed peer recipes."""

from __future__ import annotations

from tests.functional_units import functional_units

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import unittest

from seedvision.benchmarking.recipe_catalogue import (
    CATALOGUE_VERSION,
    RecipeCatalogue,
    RecipeCatalogueError,
    build_recipe_catalogue,
    source_manifest_projection,
    source_manifest_sha256,
    validate_recipe_catalogue,
)
from seedvision.benchmarking.recipes import (
    RecipeGraphMismatch,
    RecipeProvenanceMode,
    SourceResolution,
    StageRunState,
    to_pipeline_graph,
    validate_pipeline_graph,
)
from seedvision.pipeline.model import NodeStatus, PipelineNode


ROOT = Path(__file__).resolve().parents[1]


def _read_source_config() -> dict[str, object]:
    return json.loads((ROOT / "config" / "peer_reference_sources.json").read_text(encoding="utf-8"))


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@functional_units
class PeerRecipeCatalogueTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalogue = build_recipe_catalogue()
        cls.config = _read_source_config()

    def check_test_catalogue_json_and_recipe_identities_are_deterministic(self) -> None:
        first = self.catalogue.to_json()
        restored = RecipeCatalogue.from_json(first)

        self.assertEqual(restored.to_json(), first)
        self.assertEqual(restored.sha256, self.catalogue.sha256)
        self.assertEqual(self.catalogue.version, CATALOGUE_VERSION)
        self.assertEqual(
            [recipe.recipe_id for recipe in self.catalogue.recipes],
            [
                "plantcv_color_card_correct_color",
                "plantcv_seed_binary",
                "plantcv_seed_mean",
                "plantcv_seed_otsu",
                "smpts_paper_reconstruction",
            ],
        )

    def check_test_catalogue_json_rejects_duplicate_keys_and_execution_claims(self) -> None:
        document = self.catalogue.to_json()
        duplicated = document.replace(
            "{\"execution_claims\":",
            "{\"version\":\"0.1.0\",\"execution_claims\":",
            1,
        )
        with self.assertRaises(RecipeCatalogueError):
            RecipeCatalogue.from_json(duplicated)

        altered = json.loads(document)
        altered["execution_claims"]["runnable"] = True
        with self.assertRaises(RecipeCatalogueError):
            RecipeCatalogue.from_json(json.dumps(altered))

    def check_test_scope_keeps_every_accepted_family_and_extended_operation(self) -> None:
        scope = self.catalogue.to_dict()["feature_scope"]
        self.assertEqual(scope["accepted_feature_families"], [f"F{i:02d}" for i in range(1, 23)])
        self.assertTrue(scope["remaining_scope_is_still_accepted"])
        extended = " ".join(scope["extended_seed_operations_remain_in_scope"])
        self.assertIn("Gaussian, triangle, dual-channel and texture", extended)
        self.assertIn("naive-Bayes and k-means", extended)
        self.assertIn("F18 reviewed per-seed semantic/hilum annotation", extended)
        pending = " ".join(scope["accepted_core_work_not_closed_by_this_slice"])
        self.assertIn("F04.4 channel/range", pending)
        self.assertIn("F22 native desktop views", pending)
        self.assertFalse(self.catalogue.to_dict()["execution_claims"]["runnable"])
        self.assertEqual(self.catalogue.to_dict()["execution_claims"]["run_evidence"], [])

    def check_test_committed_source_projection_and_environment_lock_match_their_pins(self) -> None:
        for reference in ("plantcv", "smpts_paper"):
            projection = source_manifest_projection(self.config, reference)
            canonical = json.dumps(
                projection,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            self.assertEqual(source_manifest_sha256(self.config, reference), _sha256(canonical))

        environment = self.config["environment"]
        lock_path = ROOT / environment["lockfile"]
        self.assertEqual(_sha256(lock_path.read_bytes()), environment["lockfile_sha256"])
        self.assertEqual(self.config["version"], "4.11.3")
        self.assertEqual(self.config["commit"], "77e3322fd0cd5a9bef18bb99402f20c8f2db87a9")

        expected_files = {item["path"]: item for item in self.config["files"]}
        for recipe in self.catalogue.recipes:
            if recipe.recipe_id == "smpts_paper_reconstruction":
                self.assertEqual(recipe.provenance.mode, RecipeProvenanceMode.PAPER_RECONSTRUCTION)
                self.assertEqual(
                    recipe.provenance.source_manifest_sha256,
                    source_manifest_sha256(self.config, "smpts_paper"),
                )
                continue
            self.assertEqual(recipe.provenance.mode, RecipeProvenanceMode.ORIGINAL_REFERENCE)
            self.assertEqual(recipe.provenance.source_manifest_sha256, source_manifest_sha256(self.config, "plantcv"))
            self.assertEqual(recipe.provenance.environment_lock_sha256, environment["lockfile_sha256"])
            self.assertIn("Python 3.12.10 on Windows-x86_64", recipe.provenance.evidence_note)
            for stage in recipe.stages:
                for mapping in stage.source_mappings:
                    if mapping.reference_id != "plantcv":
                        continue
                    pin = expected_files[mapping.source_path]
                    self.assertEqual(mapping.resolution, SourceResolution.PINNED)
                    self.assertEqual(mapping.source_sha256, pin["sha256"])
                    self.assertEqual(mapping.source_url, pin["url"])
                    self.assertEqual(mapping.reference_version, self.config["version"])
                    self.assertEqual(mapping.license_id, self.config["license"]["plantcv"])

        for recipe_id, operation_id, symbol in (
            ("plantcv_seed_binary", "threshold.binary", "plantcv.plantcv.threshold.binary"),
            ("plantcv_seed_otsu", "threshold.otsu", "plantcv.plantcv.threshold.otsu"),
            ("plantcv_seed_mean", "threshold.mean", "plantcv.plantcv.threshold.mean"),
        ):
            recipe = self.catalogue.recipe(recipe_id)
            stage = next(item for item in recipe.stages if item.operation_id == operation_id)
            self.assertEqual(stage.source_mappings[0].source_symbol, symbol)

    def check_test_seed_recipe_order_ports_and_unselected_required_values_are_explicit(self) -> None:
        expected_thresholds = {
            "plantcv_seed_binary": "threshold.binary",
            "plantcv_seed_otsu": "threshold.otsu",
            "plantcv_seed_mean": "threshold.mean",
        }
        for recipe_id, threshold_operation in expected_thresholds.items():
            recipe = self.catalogue.recipe(recipe_id)
            operations = [stage.operation_id for stage in recipe.stages]
            self.assertEqual(
                operations,
                [
                    "external.image_inputs", "filter.median_blur", threshold_operation,
                    "morphology.fill", "morphology.erode", "morphology.dilate",
                    "reference.internal.watershed.distance_map",
                    "reference.internal.watershed.local_maxima",
                    "reference.internal.watershed.connected_markers",
                    "reference.internal.watershed.negate_distance",
                    "reference.internal.watershed.masked_flood",
                    "reference.internal.watershed.marker_count_observation",
                    "analysis.size", "analysis.color",
                ],
            )
            self.assertTrue(all(stage.stage_id.startswith("pcv_seed_") for stage in recipe.stages))
            self.assertIn("SeedFiddle-authored", recipe.description)
            self.assertIn("PlantCV did not publish this exact end-to-end seed workflow", recipe.description)

            stages = {stage.operation_id: stage for stage in recipe.stages}
            self.assertEqual(stages["external.image_inputs"].resolution, SourceResolution.UNRESOLVED)
            external_outputs = {port.name: port for port in stages["external.image_inputs"].ports}
            self.assertEqual(external_outputs["gray_image"].data_type, "GrayImage")
            self.assertEqual(external_outputs["rgb_image"].data_type, "RGBImage")
            self.assertIn("does not decode, convert channels, deskew, rectify, or apply", stages["external.image_inputs"].description)

            median = stages["filter.median_blur"]
            self.assertIsNone(median.parameters[0].value)
            self.assertIn("UNSELECTED", median.parameters[0].description)
            threshold = stages[threshold_operation]
            if threshold_operation == "threshold.binary":
                values = {item.parameter_id: item.value for item in threshold.parameters}
                self.assertIsNone(values["threshold"])
                self.assertEqual(values["object_type"], "light")
            elif threshold_operation == "threshold.mean":
                values = {item.parameter_id: item.value for item in threshold.parameters}
                self.assertIsNone(values["ksize"])
                self.assertIsNone(values["offset"])
                self.assertEqual(values["object_type"], "light")
            else:
                self.assertEqual(threshold.parameters[0].value, "light")

            fill = stages["morphology.fill"]
            self.assertIsNone(fill.parameters[0].value)
            self.assertIn("not hole filling", fill.description)
            for operation in ("morphology.erode", "morphology.dilate"):
                self.assertTrue(all(item.value is None for item in stages[operation].parameters))
            self.assertIsNone(stages["analysis.size"].parameters[0].value)
            self.assertIsNone(stages["analysis.color"].parameters[0].value)
            self.assertEqual(stages["analysis.color"].parameters[1].value, "all")
            self.assertTrue(all(item.affects_calculation for stage in recipe.stages for item in stage.parameters))

    def check_test_watershed_source_substages_describe_one_public_runner_call(self) -> None:
        recipe = self.catalogue.recipe("plantcv_seed_otsu")
        internal = [stage for stage in recipe.stages if stage.operation_id.startswith("reference.internal.watershed.")]
        self.assertEqual(len(internal), 6)
        self.assertEqual(
            [stage.operation_id.rsplit(".", 1)[-1] for stage in internal],
            ["distance_map", "local_maxima", "connected_markers", "negate_distance", "masked_flood", "marker_count_observation"],
        )
        self.assertTrue(all("single PlantCV 4.11.3 public call" in stage.description for stage in internal))
        self.assertTrue(all("does not independently replay or checkpoint" in stage.description for stage in internal))
        self.assertTrue(all("plantcv.plantcv.watershed.watershed_segmentation" in mapping.source_symbol
                            for stage in internal for mapping in stage.source_mappings))
        self.assertTrue(all("plantcv/plantcv/watershed.py" == mapping.source_path
                            for stage in internal for mapping in stage.source_mappings))

        by_operation = {stage.operation_id: stage for stage in internal}
        marker_count_stage = by_operation["reference.internal.watershed.marker_count_observation"]
        marker_count_inputs = {port.name: port for port in marker_count_stage.ports if port.direction == "input"}
        marker_count = next(port for port in marker_count_stage.ports if port.name == "marker_count")
        self.assertEqual(marker_count.semantics["formula"], "unique(marker_labels) - 1")
        self.assertEqual(marker_count.semantics["derived_from"], "connected marker labels")
        self.assertEqual(marker_count.semantics["not_derived_from"], "watershed labels")
        self.assertIn("after flood", marker_count.semantics["timing"])
        self.assertEqual(marker_count_inputs["markers"].data_type, "MarkerLabels")
        self.assertEqual(marker_count_inputs["flood_complete"].data_type, "OperationCompletion")
        self.assertNotIn("labels", marker_count_inputs)
        self.assertEqual(marker_count_inputs["flood_complete"].semantics["role"], "sequencing only")
        self.assertIn("estimated_object_count", marker_count_stage.description)
        self.assertIn("8-connected", by_operation["reference.internal.watershed.connected_markers"].description)
        self.assertEqual(by_operation["reference.internal.watershed.local_maxima"].parameters[0].value, 10)
        self.assertNotIn("distance", {parameter.parameter_id for parameter in by_operation["reference.internal.watershed.masked_flood"].parameters})
        self.assertEqual(marker_count_stage.parameters[0].value, "SeedFiddle")
        self.assertEqual(sum(parameter.parameter_id == "distance" for stage in internal for parameter in stage.parameters), 1)
        self.assertLess(recipe.stages.index(by_operation["reference.internal.watershed.masked_flood"]), recipe.stages.index(marker_count_stage))
        self.assertIn("dist_transform", by_operation["reference.internal.watershed.local_maxima"].description)
        self.assertIn("distanceTransformWithLabels", by_operation["reference.internal.watershed.distance_map"].description)
        self.assertIn("skimage.segmentation.watershed", by_operation["reference.internal.watershed.masked_flood"].description)
        line_evidence = {
            "reference.internal.watershed.distance_map": "line 47",
            "reference.internal.watershed.local_maxima": "lines 49-51",
            "reference.internal.watershed.connected_markers": "line 53",
            "reference.internal.watershed.negate_distance": "line 54",
            "reference.internal.watershed.masked_flood": "line 55",
            "reference.internal.watershed.marker_count_observation": "line 56",
        }
        for operation_id, line in line_evidence.items():
            self.assertTrue(any(line in mapping.note for mapping in by_operation[operation_id].source_mappings))
        count_evidence = marker_count_stage.source_mappings[0]
        self.assertIn("line 56 calculates the count after the line-55 flood", count_evidence.note)
        self.assertIn("lines 76-78 emit it later", count_evidence.note)
        self.assertEqual(count_evidence.source_path, "plantcv/plantcv/watershed.py")

    def check_test_measurement_and_colour_outputs_name_pinned_observations(self) -> None:
        recipe = self.catalogue.recipe("plantcv_seed_otsu")
        stages = {stage.operation_id: stage for stage in recipe.stages}
        size_outputs = {port.name: port for port in stages["analysis.size"].ports if port.direction == "output"}
        size_variables = size_outputs["observations"].semantics["variables"]
        self.assertEqual(len(size_variables), 16)
        self.assertIn("ellipse_major_axis", size_variables)
        self.assertIn("longest_path", size_variables)
        self.assertEqual(size_outputs["observations"].semantics["physical_calibration"], "not supplied")

        colour_outputs = {port.name: port for port in stages["analysis.color"].ports if port.direction == "output"}
        colour_variables = colour_outputs["observations"].semantics["variables"]
        self.assertIn("hue_circular_mean", colour_variables)
        self.assertIn("lightness_frequencies", colour_variables)
        self.assertEqual(colour_outputs["observations"].semantics["hue_frequency_bins"], 180)
        self.assertEqual(colour_outputs["histogram"].data_type, "Visualization")
        self.assertIn("does not claim biological colour validity", stages["analysis.color"].description)

    def check_test_color_correction_is_an_explicit_external_input_operation(self) -> None:
        recipe = self.catalogue.recipe("plantcv_color_card_correct_color")
        self.assertEqual(len(recipe.stages), 1)
        stage = recipe.stages[0]
        self.assertEqual(stage.operation_id, "transform.correct_color")
        self.assertEqual(
            {port.name for port in stage.ports if port.direction == "input"},
            {"target_rgb", "target_mask", "source_rgb", "source_mask"},
        )
        self.assertTrue(all(port.external for port in stage.ports if port.direction == "input"))
        ports = {port.name: port for port in stage.ports}
        self.assertEqual(tuple(ports["target_matrix"].semantics["columns"]), ("chip_number", "mean_R", "mean_G", "mean_B"))
        self.assertEqual(ports["transformation_matrix"].semantics["shape"], "9x9")
        self.assertIn("No image decoder, card detector, matrix source", stage.description)
        self.assertIn("temporary output directory", stage.source_mappings[0].note)

    def check_test_smpts_paper_graph_preserves_ambiguity_and_original_unavailability(self) -> None:
        recipe = self.catalogue.recipe("smpts_paper_reconstruction")
        self.assertEqual(recipe.provenance.mode, RecipeProvenanceMode.PAPER_RECONSTRUCTION)
        self.assertEqual(recipe.provenance.evidence_status, SourceResolution.PINNED)
        self.assertEqual(len(recipe.stages), 9)
        self.assertTrue(all(stage.stage_id.startswith("smpts_paper_") for stage in recipe.stages))
        self.assertTrue(all(stage.resolution is SourceResolution.UNRESOLVED for stage in recipe.stages))

        operation_ids = {stage.operation_id for stage in recipe.stages}
        for operation in (
            "paper_reconstruction.threshold_branch_a", "paper_reconstruction.threshold_branch_b",
            "paper_reconstruction.filter_branch_a", "paper_reconstruction.filter_branch_b",
            "paper_reconstruction.mask_combine", "paper_reconstruction.contact_screen",
            "paper_reconstruction.targeted_erosion", "paper_reconstruction.extract_seed_crops",
        ):
            self.assertIn(operation, operation_ids)
        erosion = next(stage for stage in recipe.stages if stage.operation_id.endswith("targeted_erosion"))
        self.assertIsNotNone(erosion.condition)
        extraction = next(stage for stage in recipe.stages if stage.operation_id.endswith("extract_seed_crops"))
        self.assertEqual(
            next(port for port in extraction.ports if port.name == "crops").semantics["mask_output"],
            "not inferred",
        )
        for stage in recipe.stages:
            self.assertEqual(len(stage.source_mappings), 2)
            mapping_by_id = {item.reference_id: item for item in stage.source_mappings}
            self.assertEqual(mapping_by_id["smpts_paper"].resolution, SourceResolution.UNRESOLVED)
            self.assertEqual(mapping_by_id["smpts_paper"].source_url, self.config["smpts"]["paper_url"])
            self.assertEqual(mapping_by_id["smpts_original"].resolution, SourceResolution.UNAVAILABLE)
            self.assertEqual(mapping_by_id["smpts_original"].source_url, self.config["smpts"]["original_capsule"])
            self.assertIn("HTTP 403", mapping_by_id["smpts_original"].note)
            self.assertTrue(all(parameter.value is None for parameter in stage.parameters))
            self.assertTrue(all("UNSELECTED" in parameter.description for parameter in stage.parameters))

        for stage in recipe.stages:
            for port in stage.ports:
                if port.data_type == "BinaryMask":
                    self.assertEqual(port.semantics["dtype"], "unresolved")
                    self.assertEqual(port.semantics["foreground_value"], "unresolved")
                    self.assertEqual(port.semantics["background_value"], "unresolved")
                    self.assertEqual(port.semantics["polarity"], "unresolved")
                    self.assertEqual(port.semantics["channel_conversion"], "unresolved")
                    self.assertNotIn("range", port.semantics)
                if port.data_type == "ImageRaster":
                    self.assertEqual(port.semantics["dtype"], "unresolved")
                    self.assertEqual(port.semantics["channel_conversion"], "unresolved")
                    self.assertEqual(port.semantics["intensity_range"], "unresolved")
        threshold_inputs = [
            port for stage in recipe.stages if "threshold_branch" in stage.operation_id
            for port in stage.ports if port.direction == "input"
        ]
        self.assertEqual(len(threshold_inputs), 2)
        self.assertTrue(all(port.external and port.data_type == "ImageRaster" for port in threshold_inputs))
        self.assertEqual(
            next(port for port in extraction.ports if port.name == "image").semantics["channel_order"],
            "unresolved",
        )
        self.assertEqual(
            next(port for port in extraction.ports if port.name == "rectangles").semantics["coordinate_frame"],
            "unresolved",
        )

    def check_test_recipe_layout_changes_do_not_change_analytical_identity(self) -> None:
        recipe = self.catalogue.recipe("plantcv_seed_binary")
        first = recipe.layout[0]
        moved = replace(recipe, layout=(replace(first, x=first.x + 100), *recipe.layout[1:]))
        self.assertEqual(recipe.identity.analytical_dict(), moved.identity.analytical_dict())
        self.assertNotEqual(recipe.identity.layout_sha256, moved.identity.layout_sha256)

    def check_test_graph_projection_rejects_miswire_omission_extra_stale_and_unused_nodes(self) -> None:
        recipe = self.catalogue.recipe("plantcv_seed_binary")
        graph = to_pipeline_graph(recipe)
        first_edge = graph.connections[0]
        graph.connections = (replace(first_edge, target_port="wrong_port"), *graph.connections[1:])
        with self.assertRaises(RecipeGraphMismatch):
            validate_pipeline_graph(recipe, graph)

        graph = to_pipeline_graph(recipe)
        graph.nodes.pop(recipe.stages[0].stage_id)
        with self.assertRaises(RecipeGraphMismatch):
            validate_pipeline_graph(recipe, graph)

        graph = to_pipeline_graph(recipe)
        graph.nodes["undeclared_extra"] = replace(graph.nodes[recipe.stages[0].stage_id], identifier="undeclared_extra")
        with self.assertRaises(RecipeGraphMismatch):
            validate_pipeline_graph(recipe, graph)

        graph = to_pipeline_graph(recipe)
        graph.reference_recipe_identity = self.catalogue.recipe("plantcv_seed_otsu").identity
        with self.assertRaises(RecipeGraphMismatch):
            validate_pipeline_graph(recipe, graph)

        graph = to_pipeline_graph(recipe)
        active_id = recipe.stages[0].stage_id
        graph.unused_nodes[active_id] = replace(graph.nodes[active_id], implemented=True)
        with self.assertRaises(RecipeGraphMismatch):
            validate_pipeline_graph(recipe, graph)

        graph = to_pipeline_graph(recipe)
        graph.unused_nodes["distinct_unused_extra"] = replace(
            graph.nodes[active_id], identifier="distinct_unused_extra", implemented=True,
        )
        with self.assertRaises(RecipeGraphMismatch):
            validate_pipeline_graph(recipe, graph)

    def check_test_projected_graphs_keep_execution_states_unrun_or_unresolved(self) -> None:
        validate_recipe_catalogue(self.catalogue)
        for recipe in self.catalogue.recipes:
            graph = to_pipeline_graph(recipe)
            self.assertTrue(all(not node.implemented for node in graph.nodes.values()))
            self.assertFalse(any(node.status is NodeStatus.COMPLETE for node in graph.nodes.values()))
            for record in graph.reference_recipe_runs.values():
                self.assertIn(record.state, {StageRunState.NOT_RUN, StageRunState.UNRESOLVED})

    def test_workflow_lifecycle(self):
        """10 related PeerRecipeCatalogueTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_catalogue_json_and_recipe_identities_are_deterministic)
        self.scenario(self.check_test_scope_keeps_every_accepted_family_and_extended_operation)
        self.scenario(self.check_test_committed_source_projection_and_environment_lock_match_their_pins)
        self.scenario(self.check_test_seed_recipe_order_ports_and_unselected_required_values_are_explicit)
        self.scenario(self.check_test_watershed_source_substages_describe_one_public_runner_call)
        self.scenario(self.check_test_measurement_and_colour_outputs_name_pinned_observations)
        self.scenario(self.check_test_color_correction_is_an_explicit_external_input_operation)
        self.scenario(self.check_test_smpts_paper_graph_preserves_ambiguity_and_original_unavailability)
        self.scenario(self.check_test_recipe_layout_changes_do_not_change_analytical_identity)
        self.scenario(self.check_test_projected_graphs_keep_execution_states_unrun_or_unresolved)

    def test_validation_and_recovery(self):
        """2 related PeerRecipeCatalogueTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_catalogue_json_rejects_duplicate_keys_and_execution_claims)
        self.scenario(self.check_test_graph_projection_rejects_miswire_omission_extra_stale_and_unused_nodes)


if __name__ == "__main__":
    unittest.main()
