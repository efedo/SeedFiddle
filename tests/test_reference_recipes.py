"""Contract tests for reference recipes; fixtures here are not real recipes."""

from __future__ import annotations

from tests.functional_units import functional_units

import copy
from dataclasses import replace
import json
import unittest

from seedvision.benchmarking.checkpoints import CheckpointIdentity, CoordinateFrame
from seedvision.benchmarking.recipes import (
    BranchCondition,
    DependencyContract,
    LayoutPosition,
    ParameterContract,
    PortContract,
    RecipeContractError,
    RecipeProvenance,
    RecipeProvenanceMode,
    RecipeSpec,
    RecipeStage,
    RecipeGraphMismatch,
    SourceMapping,
    SourceResolution,
    StageRunRecord,
    StageRunState,
    StaleRecipeIdentity,
    initial_stage_run,
    to_pipeline_graph,
    validate_checkpoint_identity,
    validate_pipeline_graph,
    validate_stage_run,
)
from seedvision.pipeline.model import NodeStatus, PipelineConnection, PipelineNode


_SHA = "a" * 64


def _contract_recipe() -> RecipeSpec:
    """Small synthetic topology used only to test data/graph contracts."""
    source = SourceMapping(
        reference_id="contract_demo",
        reference_version="0",
        resolution=SourceResolution.UNRESOLVED,
        note="Test-only mapping; this is not a scientific source or verified recipe.",
    )
    provenance = RecipeProvenance(
        mode=RecipeProvenanceMode.ENHANCED,
        backend_id="contract_demo",
        backend_version="test-only",
        reference_id="contract_demo",
        reference_version="0",
        environment_lock_sha256="b" * 64,
        environment_versions={"test_runner": "0"},
        evidence_status=SourceResolution.UNRESOLVED,
        evidence_note="Synthetic test fixture only; no reference execution or science claim.",
    )
    source_stage = RecipeStage(
        stage_id="source",
        title="Synthetic source",
        operation_id="test.source",
        description="Test-only external image source.",
        ports=(PortContract("image", "output", "Image", "Image", semantics={"color_order": "RGB", "dtype": "uint8"}),),
        source_mappings=(source,),
        resolution=SourceResolution.UNRESOLVED,
        resolution_note="Test fixture has no executable reference source.",
    )
    threshold = RecipeStage(
        stage_id="threshold",
        title="Synthetic threshold",
        operation_id="test.threshold",
        description="Test-only threshold contract.",
        ports=(
            PortContract("image", "input", "Image", "Input image", semantics={"color_order": "RGB", "dtype": "uint8"}),
            PortContract("mask", "output", "Mask", "Output mask", semantics={"foreground": "white", "dtype": "uint8"}),
        ),
        parameters=(
            ParameterContract("cutoff", "integer", 127, True, True, "intensity", "Test-only value.", 0, 255),
            ParameterContract("caption", "string", "fixture", False, False, description="Display-only fixture text."),
        ),
        source_mappings=(source,),
        resolution=SourceResolution.UNRESOLVED,
        resolution_note="Test fixture has no executable reference source.",
    )
    split = RecipeStage(
        stage_id="split",
        title="Synthetic conditional split",
        operation_id="test.split",
        description="Test-only conditional stage.",
        ports=(
            PortContract("mask", "input", "Mask", "Foreground mask", semantics={"foreground": "white", "dtype": "uint8"}),
            PortContract("instances", "output", "Instances", "Instance labels", semantics={"background": 0, "dtype": "uint32"}),
        ),
        source_mappings=(source,),
        resolution=SourceResolution.UNRESOLVED,
        resolution_note="Test fixture has no executable reference source.",
        condition=BranchCondition("contact_count > 0", "Contact handling"),
    )
    return RecipeSpec(
        recipe_id="contract_demo",
        recipe_version="1",
        title="Synthetic contract test",
        description="Only exercises serialization and topology contracts.",
        provenance=provenance,
        stages=(source_stage, threshold, split),
        dependencies=(
            DependencyContract("source", "image", "threshold", "image", "Image"),
            DependencyContract("threshold", "mask", "split", "mask", "Mask"),
        ),
        layout=(LayoutPosition("source", 0, 0), LayoutPosition("threshold", 240, 0), LayoutPosition("split", 480, 0)),
    )


def _pinned_contract_recipe() -> RecipeSpec:
    recipe = _contract_recipe()
    source = SourceMapping(
        reference_id="contract_demo",
        reference_version="0",
        resolution=SourceResolution.PINNED,
        source_path="test_only.py",
        source_symbol="test_operation",
        source_sha256=_SHA,
        note="Test-only source mapping; no scientific claim.",
    )
    stages = tuple(replace(
        stage,
        source_mappings=(source,),
        resolution=SourceResolution.PINNED,
        resolution_note="",
    ) for stage in recipe.stages)
    return replace(recipe, stages=stages)


def _unavailable_contract_recipe() -> RecipeSpec:
    recipe = _contract_recipe()
    source = SourceMapping(
        reference_id="contract_demo",
        reference_version="0",
        resolution=SourceResolution.UNAVAILABLE,
        note="Test-only unavailable source mapping.",
    )
    stages = tuple(replace(
        stage,
        source_mappings=(source,),
        resolution=SourceResolution.UNAVAILABLE,
        resolution_note="Test-only unavailable operation.",
    ) for stage in recipe.stages)
    return replace(recipe, stages=stages)


def _checkpoint_identity(recipe: RecipeSpec, stage_id: str, *, upstream: str | None = None) -> CheckpointIdentity:
    identity = recipe.identity
    return CheckpointIdentity(
        source_sha256=_SHA,
        coordinate_frame=CoordinateFrame("full", (10, 20), "top-left", "x-right/y-down", "pixel"),
        stage_id=stage_id,
        recipe_id=identity.recipe_id,
        recipe_version=identity.recipe_version,
        recipe_sha256=identity.recipe_sha256,
        graph_sha256=identity.graph_sha256,
        settings_sha256=identity.settings_sha256,
        backend_id=identity.backend_id,
        backend_version=identity.backend_version,
        environment_versions=dict(recipe.provenance.environment_versions),
        upstream_checkpoint_sha256=upstream,
        image_semantics={
            "color_order": "RGB", "dtype": "uint8", "value_range": [0, 255],
            "channel_conversion": "none", "geometric_transform": "none", "interpolation": "none",
        },
        mask_semantics={
            "foreground_polarity": "white-foreground", "representation": "binary",
            "connectivity": 8, "validity_region": "full", "border_policy": "retain",
        },
        instance_semantics={
            "label_dtype": "uint32", "background_label": 0, "label_id_mapping": "identity",
            "split_boundary_ownership": "declared", "visibility": "complete", "eligibility": "all",
        },
    )


@functional_units
class ReferenceRecipeTests(unittest.TestCase):
    def check_test_strict_json_round_trip_and_deep_immutability(self) -> None:
        recipe = _contract_recipe()
        encoded = recipe.to_json()
        restored = RecipeSpec.from_json(encoded)
        self.assertEqual(RecipeSpec.from_json(encoded), recipe)
        self.assertEqual(restored, recipe)
        parameter = recipe.stage("threshold").parameters[1]
        with self.assertRaises(TypeError):
            parameter.value["unsafe"] = True

        document = recipe.to_json()
        with self.assertRaises(RecipeContractError):
            RecipeSpec.from_json(document[:-1] + ', "unknown": 1}')
        with self.assertRaises(RecipeContractError):
            RecipeSpec.from_json('{"schema":"x","schema":"y"}')
        with self.assertRaises(RecipeContractError):
            RecipeSpec.from_json('{"value":NaN}')
        with self.assertRaises(RecipeContractError):
            RecipeSpec.from_json(b"\xff")

        invalid_mode = recipe.to_dict()
        invalid_mode["provenance"]["mode"] = "unknown_mode"
        with self.assertRaises(RecipeContractError):
            RecipeSpec.from_json(json.dumps(invalid_mode))
        provenance_payload = recipe.provenance.to_dict()
        provenance_payload["mode"] = "unknown_mode"
        with self.assertRaises(RecipeContractError):
            RecipeProvenance.from_dict(provenance_payload)

    def check_test_parameter_choices_and_one_sided_bounds_validate_the_parameter_value(self) -> None:
        valid_choice = ParameterContract(
            "threshold_method", "string", "otsu", True, True,
            choices=("otsu", "mean"),
        )
        self.assertEqual(valid_choice.value, "otsu")
        with self.assertRaisesRegex(RecipeContractError, "choices"):
            ParameterContract(
                "threshold_method", "string", "triangle", True, True,
                description="Display metadata must not shadow this value.",
                choices=("otsu", "mean"),
            )

        ParameterContract("lower_only", "number", 1, True, minimum=0)
        ParameterContract("upper_only", "integer", 2, True, maximum=2)
        with self.assertRaisesRegex(RecipeContractError, "below its minimum"):
            ParameterContract("lower_only", "number", -1, True, minimum=0)
        with self.assertRaisesRegex(RecipeContractError, "above its maximum"):
            ParameterContract("upper_only", "integer", 3, True, maximum=2)

        recipe = _contract_recipe()
        document = recipe.to_dict()
        document["stages"][1]["parameters"][0]["value"] = -10
        with self.assertRaisesRegex(RecipeContractError, "below its minimum"):
            RecipeSpec.from_json(json.dumps(document))
        document["stages"][1]["parameters"][0]["value"] = 999
        with self.assertRaisesRegex(RecipeContractError, "above its maximum"):
            RecipeSpec.from_json(json.dumps(document))

    def check_test_recipe_rejects_duplicate_missing_mistyped_and_cyclic_dependencies(self) -> None:
        recipe = _contract_recipe()
        with self.assertRaisesRegex(RecipeContractError, "unique"):
            replace(recipe, dependencies=recipe.dependencies + (recipe.dependencies[0],))
        with self.assertRaisesRegex(RecipeContractError, "no producer"):
            replace(recipe, dependencies=recipe.dependencies[:1])
        with self.assertRaisesRegex(RecipeContractError, "data type"):
            replace(recipe, dependencies=(replace(recipe.dependencies[0], data_type="Mask"), *recipe.dependencies[1:]))
        mismatched_semantics = replace(
            recipe.stages[1],
            ports=(
                replace(recipe.stages[1].ports[0], semantics={"color_order": "BGR", "dtype": "uint8"}),
                recipe.stages[1].ports[1],
            ),
        )
        with self.assertRaisesRegex(RecipeContractError, "semantics"):
            replace(recipe, stages=(recipe.stages[0], mismatched_semantics, recipe.stages[2]))
        cycle_source = replace(
            recipe.stages[0],
            ports=(PortContract(
                       "in_mask", "input", "Mask", "Mask", external=True,
                       semantics={"foreground": "white", "dtype": "uint8"},
                   ),
                   PortContract(
                       "image", "output", "Image", "Image",
                       semantics={"color_order": "RGB", "dtype": "uint8"},
                   )),
        )
        with self.assertRaisesRegex(RecipeContractError, "acyclic"):
            replace(
                recipe,
                stages=(cycle_source, recipe.stages[1], recipe.stages[2]),
                dependencies=(
                    DependencyContract("source", "image", "threshold", "image", "Image"),
                    DependencyContract("threshold", "mask", "source", "in_mask", "Mask"),
                    DependencyContract("threshold", "mask", "split", "mask", "Mask"),
                ),
            )

    def check_test_editing_layout_or_display_only_values_does_not_change_analytic_identity(self) -> None:
        recipe = _contract_recipe()
        original = recipe.identity
        moved = replace(recipe, layout=(LayoutPosition("source", 1000, -40),))
        self.assertEqual(moved.identity.recipe_sha256, original.recipe_sha256)
        self.assertEqual(moved.identity.graph_sha256, original.graph_sha256)
        self.assertEqual(moved.identity.settings_sha256, original.settings_sha256)
        self.assertNotEqual(moved.identity.layout_sha256, original.layout_sha256)
        relabelled = replace(recipe, title="Different title", description="Presentation-only edit.")
        self.assertEqual(relabelled.identity.analytical_dict(), original.analytical_dict())

        threshold = recipe.stage("threshold")
        display_parameter = replace(threshold.parameters[1], value="different display text")
        display_stage = replace(threshold, parameters=(threshold.parameters[0], display_parameter))
        display_recipe = replace(recipe, stages=(recipe.stages[0], display_stage, recipe.stages[2]))
        self.assertEqual(display_recipe.identity.analytical_dict(), original.analytical_dict())

        calc_parameter = replace(threshold.parameters[0], value=128)
        calc_stage = replace(threshold, parameters=(calc_parameter, threshold.parameters[1]))
        calc_recipe = replace(recipe, stages=(recipe.stages[0], calc_stage, recipe.stages[2]))
        self.assertNotEqual(calc_recipe.identity.settings_sha256, original.settings_sha256)
        self.assertNotEqual(calc_recipe.identity.recipe_sha256, original.recipe_sha256)
        self.assertEqual(calc_recipe.identity.graph_sha256, original.graph_sha256)
        self.assertEqual(recipe.parameters_for_runner("threshold"), {"cutoff": 127})

    def check_test_existing_pipeline_graph_projection_is_read_only_exact_and_auditable(self) -> None:
        recipe = _contract_recipe()
        graph = to_pipeline_graph(recipe)
        self.assertEqual(graph.topological_order(), ("source", "threshold", "split"))
        self.assertTrue(all(not node.implemented for node in graph.nodes.values()))
        self.assertEqual(graph.node("source").status, NodeStatus.PLANNED)
        self.assertTrue(graph.node("split").bypassable)
        validate_pipeline_graph(recipe, graph)
        with self.assertRaises(TypeError):
            graph.reference_recipe_sources["threshold"][0]["note"] = "mutate"
        moved_graph = to_pipeline_graph(replace(recipe, layout=(LayoutPosition("source", 1000, -40),)))
        validate_pipeline_graph(recipe, moved_graph)

        graph.nodes["extra"] = PipelineNode("extra", "Extra", "reference", "extra", 0, 0, implemented=False)
        with self.assertRaisesRegex(RecipeGraphMismatch, "extra"):
            validate_pipeline_graph(recipe, graph)
        del graph.nodes["extra"]
        del graph.nodes["split"]
        with self.assertRaisesRegex(RecipeGraphMismatch, "missing"):
            validate_pipeline_graph(recipe, graph)
        graph.nodes["split"] = to_pipeline_graph(recipe).node("split")

        graph.connections = (PipelineConnection("source", "split", "Image", "image", "mask"),)
        with self.assertRaisesRegex(RecipeGraphMismatch, "dependencies"):
            validate_pipeline_graph(recipe, graph)
        graph = to_pipeline_graph(recipe)
        graph.nodes["threshold"].input_port_types["image"] = "Mask"
        with self.assertRaisesRegex(RecipeGraphMismatch, "mistyped"):
            validate_pipeline_graph(recipe, graph)
        graph = to_pipeline_graph(recipe)
        graph.nodes["split"].details += "\nsource mapping edited"
        with self.assertRaisesRegex(RecipeGraphMismatch, "Source mapping"):
            validate_pipeline_graph(recipe, graph)
        graph = to_pipeline_graph(recipe)
        graph.reference_recipe_sources = {}
        with self.assertRaisesRegex(RecipeGraphMismatch, "source mappings"):
            validate_pipeline_graph(recipe, graph)
        graph = to_pipeline_graph(recipe)
        graph.node("threshold").parameters["fake_control"] = 1
        with self.assertRaisesRegex(RecipeGraphMismatch, "read-only"):
            validate_pipeline_graph(recipe, graph)

    def check_test_projection_consumes_record_iterator_before_snapshotting_identity(self) -> None:
        recipe = _pinned_contract_recipe()

        def records():
            # An external iterator can run code before yielding any records.
            object.__setattr__(recipe, "recipe_version", "2")
            yield from ()

        graph = to_pipeline_graph(recipe, records())
        self.assertEqual(graph.reference_recipe_identity, recipe.identity)
        for record in graph.reference_recipe_runs.values():
            self.assertEqual(record.recipe_identity, recipe.identity)
        validate_pipeline_graph(recipe, graph)

    def check_test_run_mapping_callback_cannot_leave_a_changed_recipe_validated(self) -> None:
        recipe = _pinned_contract_recipe()
        graph = to_pipeline_graph(recipe)
        last_stage = recipe.stages[-1].stage_id

        class ChangingRuns(dict):
            def __getitem__(self, key):
                record = super().__getitem__(key)
                if key == last_stage:
                    object.__setattr__(recipe, "recipe_version", "2")
                return record

        graph.reference_recipe_runs = ChangingRuns(graph.reference_recipe_runs)
        with self.assertRaisesRegex(RecipeGraphMismatch, "Stale run state"):
            validate_pipeline_graph(recipe, graph)

    def check_test_graph_run_mapping_must_match_each_stage_id(self) -> None:
        recipe = _contract_recipe()
        graph = to_pipeline_graph(recipe)
        runs = dict(graph.reference_recipe_runs)
        runs["source"], runs["threshold"] = runs["threshold"], runs["source"]
        graph.reference_recipe_runs = runs
        with self.assertRaisesRegex(RecipeGraphMismatch, "bound to stage"):
            validate_pipeline_graph(recipe, graph)

    def check_test_reference_projection_rejects_all_unused_nodes_even_with_id_collisions(self) -> None:
        recipe = _contract_recipe()
        cases = (
            # An inactive duplicate uses an active ID and used to disappear in
            # the union of active and unused ID sets.
            ("threshold", "threshold"),
            # A stale distinct node can also be stored under an active key;
            # the key sets then still appear to match the recipe exactly.
            ("threshold", "unused_variant"),
            # Ordinary distinct toolbox nodes must also be rejected.
            ("obsolete", "obsolete"),
        )
        for unused_key, inactive_id in cases:
            with self.subTest(unused_key=unused_key, inactive_id=inactive_id):
                graph = to_pipeline_graph(recipe)
                graph.unused_nodes[unused_key] = PipelineNode(
                    inactive_id,
                    "Inactive altered stage",
                    "reference",
                    "Unapproved inactive source and parameters.",
                    0,
                    0,
                    details="changed source mapping",
                    implemented=True,
                    parameters={"unapproved": 1},
                )
                with self.assertRaisesRegex(RecipeGraphMismatch, "unused nodes"):
                    validate_pipeline_graph(recipe, graph)

    def check_test_downstream_run_upstream_bindings_are_optional_or_explicitly_checked(self) -> None:
        recipe = _pinned_contract_recipe()
        upstream_sha = "c" * 64
        output_sha = "d" * 64
        source = StageRunRecord(
            "source", recipe.identity, StageRunState.EXECUTED,
            checkpoint_sha256=upstream_sha,
        )
        threshold = StageRunRecord(
            "threshold", recipe.identity, StageRunState.EXECUTED,
            checkpoint_sha256=output_sha,
            upstream_checkpoint_sha256=upstream_sha,
        )
        records = (source, threshold)
        validate_stage_run(recipe, threshold)
        default_graph = to_pipeline_graph(recipe, records)
        validate_pipeline_graph(recipe, default_graph)
        expected = {"source": None, "threshold": upstream_sha}
        graph = to_pipeline_graph(
            recipe, records, expected_upstream_checkpoint_sha256s=expected,
        )
        validate_pipeline_graph(
            recipe, graph, expected_upstream_checkpoint_sha256s=expected,
        )
        self.assertEqual(graph.node("threshold").status, NodeStatus.COMPLETE)

        with self.assertRaises(StaleRecipeIdentity):
            to_pipeline_graph(
                recipe, records,
                expected_upstream_checkpoint_sha256s={"threshold": "e" * 64},
            )
        with self.assertRaises(RecipeGraphMismatch):
            validate_pipeline_graph(
                recipe, graph,
                expected_upstream_checkpoint_sha256s={"threshold": None},
            )

        downstream_checkpoint = _checkpoint_identity(recipe, "threshold", upstream=upstream_sha)
        validate_checkpoint_identity(
            recipe, "threshold", downstream_checkpoint, expected_identity=downstream_checkpoint,
        )
        validate_checkpoint_identity(
            recipe, "threshold", downstream_checkpoint, expected_identity=downstream_checkpoint,
            expected_upstream_checkpoint_sha256=upstream_sha,
        )

    def check_test_stage_execution_states_and_stale_identities_are_explicit(self) -> None:
        recipe = _contract_recipe()
        unresolved = initial_stage_run(recipe, "threshold")
        self.assertEqual(unresolved.state, StageRunState.UNRESOLVED)
        self.assertIn("no executable", unresolved.reason)
        run_payload = unresolved.to_dict()
        original_payload = copy.deepcopy(run_payload)
        self.assertEqual(StageRunRecord.from_dict(run_payload), unresolved)
        self.assertEqual(StageRunRecord.from_dict(run_payload), unresolved)
        self.assertEqual(run_payload, original_payload)
        not_run = StageRunRecord("threshold", recipe.identity, StageRunState.NOT_RUN)
        validate_stage_run(recipe, not_run)
        pinned_recipe = _pinned_contract_recipe()
        conditional_bypass = StageRunRecord(
            "split", pinned_recipe.identity, StageRunState.BYPASSED, reason="No contact candidate.",
        )
        validate_stage_run(pinned_recipe, conditional_bypass)
        with self.assertRaisesRegex(RecipeContractError, "conditional"):
            validate_stage_run(pinned_recipe, replace(conditional_bypass, stage_id="threshold"))

        old_record = StageRunRecord("threshold", recipe.identity, StageRunState.NOT_RUN)
        changed = replace(
            recipe,
            stages=(
                recipe.stages[0],
                replace(recipe.stages[1], parameters=(
                    replace(recipe.stages[1].parameters[0], value=128),
                    recipe.stages[1].parameters[1],
                )),
                recipe.stages[2],
            ),
        )
        with self.assertRaises(StaleRecipeIdentity):
            validate_stage_run(changed, old_record)

        checkpoint = _checkpoint_identity(recipe, "threshold")
        validate_checkpoint_identity(recipe, "threshold", checkpoint, expected_identity=checkpoint)
        stale_environment = replace(checkpoint, environment_versions={"test_runner": "changed"})
        with self.assertRaises(StaleRecipeIdentity):
            validate_checkpoint_identity(
                recipe, "threshold", stale_environment, expected_identity=stale_environment,
            )
        validate_checkpoint_identity(recipe, "threshold", checkpoint, expected_identity=checkpoint)
        stale_source = replace(checkpoint, source_sha256="b" * 64)
        with self.assertRaises(StaleRecipeIdentity):
            validate_checkpoint_identity(recipe, "threshold", checkpoint, expected_identity=stale_source)
        with self.assertRaises(StaleRecipeIdentity):
            validate_checkpoint_identity(changed, "threshold", checkpoint, expected_identity=checkpoint)
        upstream = _SHA.replace("a", "b")
        with self.assertRaises(StaleRecipeIdentity):
            validate_checkpoint_identity(
                recipe, "threshold", checkpoint, expected_identity=checkpoint,
                expected_upstream_checkpoint_sha256=upstream,
            )
        record = StageRunRecord(
            "threshold", pinned_recipe.identity, StageRunState.EXECUTED,
            checkpoint_sha256=_SHA, upstream_checkpoint_sha256=upstream,
        )
        with self.assertRaises(StaleRecipeIdentity):
            validate_stage_run(pinned_recipe, record, expected_upstream_checkpoint_sha256=_SHA)

    def check_test_unresolved_and_unavailable_sources_cannot_be_recorded_as_completed_or_bypassed(self) -> None:
        unresolved_recipe = _contract_recipe()
        unavailable_recipe = _unavailable_contract_recipe()
        for recipe, resolution, expected_status in (
            (unresolved_recipe, SourceResolution.UNRESOLVED, NodeStatus.PLANNED),
            (unavailable_recipe, SourceResolution.UNAVAILABLE, NodeStatus.BLOCKED),
        ):
            graph = to_pipeline_graph(recipe)
            self.assertEqual(graph.node("threshold").status, expected_status)
            for stage_id, state in (
                ("threshold", StageRunState.EXECUTED),
                ("threshold", StageRunState.EMPTY),
                ("split", StageRunState.BYPASSED),
            ):
                record = StageRunRecord(
                    stage_id,
                    recipe.identity,
                    state,
                    checkpoint_sha256=_SHA if state in {StageRunState.EXECUTED, StageRunState.EMPTY} else None,
                    reason="Conditional stage would be skipped." if state is StageRunState.BYPASSED else "",
                )
                with self.subTest(resolution=resolution, stage=stage_id, state=state):
                    with self.assertRaisesRegex(RecipeContractError, "source-.*completed or bypassed"):
                        validate_stage_run(recipe, record)
                    decoded = StageRunRecord.from_dict(record.to_dict())
                    with self.assertRaisesRegex(RecipeContractError, "source-.*completed or bypassed"):
                        to_pipeline_graph(recipe, (decoded,))

    def check_test_original_reference_pinned_provenance_requires_complete_source_and_environment_pins(self) -> None:
        with self.assertRaisesRegex(RecipeContractError, "requires source ID/version/commit"):
            RecipeProvenance(
                mode=RecipeProvenanceMode.ORIGINAL_REFERENCE,
                backend_id="smpts",
                backend_version="1",
                evidence_status=SourceResolution.PINNED,
            )

        pinned = RecipeProvenance(
            mode=RecipeProvenanceMode.ORIGINAL_REFERENCE,
            backend_id="smpts",
            backend_version="1",
            reference_id="smpts_original",
            reference_version="1.0",
            reference_commit="f" * 40,
            source_manifest_sha256="a" * 64,
            environment_lock_sha256="b" * 64,
            environment_versions={"python": "3.12.10", "runner": "1.0"},
            evidence_status=SourceResolution.PINNED,
        )
        recipe = replace(_contract_recipe(), provenance=pinned)
        document = recipe.to_dict()
        document["provenance"]["environment_lock_sha256"] = None
        with self.assertRaisesRegex(RecipeContractError, "requires source ID/version/commit"):
            RecipeSpec.from_json(json.dumps(document))

        with self.assertRaisesRegex(RecipeContractError, "full Git commit"):
            replace(pinned, reference_commit="main")

    def check_test_pipeline_view_carries_explicit_not_run_checkpoint_identity(self) -> None:
        recipe = _contract_recipe()
        record = StageRunRecord("threshold", recipe.identity, StageRunState.NOT_RUN)
        graph = to_pipeline_graph(recipe, (record,))
        self.assertEqual(graph.node("threshold").status, NodeStatus.IDLE)
        self.assertEqual(graph.node("threshold").status_detail, "Not run")
        self.assertEqual(graph.reference_recipe_runs["threshold"], record)
        validate_pipeline_graph(recipe, graph)

    def test_workflow_lifecycle(self):
        """10 related ReferenceRecipeTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_strict_json_round_trip_and_deep_immutability)
        self.scenario(self.check_test_parameter_choices_and_one_sided_bounds_validate_the_parameter_value)
        self.scenario(self.check_test_editing_layout_or_display_only_values_does_not_change_analytic_identity)
        self.scenario(self.check_test_existing_pipeline_graph_projection_is_read_only_exact_and_auditable)
        self.scenario(self.check_test_projection_consumes_record_iterator_before_snapshotting_identity)
        self.scenario(self.check_test_run_mapping_callback_cannot_leave_a_changed_recipe_validated)
        self.scenario(self.check_test_graph_run_mapping_must_match_each_stage_id)
        self.scenario(self.check_test_downstream_run_upstream_bindings_are_optional_or_explicitly_checked)
        self.scenario(self.check_test_original_reference_pinned_provenance_requires_complete_source_and_environment_pins)
        self.scenario(self.check_test_pipeline_view_carries_explicit_not_run_checkpoint_identity)

    def test_validation_and_recovery(self):
        """4 related ReferenceRecipeTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_recipe_rejects_duplicate_missing_mistyped_and_cyclic_dependencies)
        self.scenario(self.check_test_reference_projection_rejects_all_unused_nodes_even_with_id_collisions)
        self.scenario(self.check_test_stage_execution_states_and_stale_identities_are_explicit)
        self.scenario(self.check_test_unresolved_and_unavailable_sources_cannot_be_recorded_as_completed_or_bypassed)


if __name__ == "__main__":
    unittest.main()
