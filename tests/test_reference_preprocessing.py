from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np

from seedvision.benchmarking.checkpoints import (
    CheckpointIdentity,
    CheckpointReference,
    CoordinateFrame,
    ExecutionRecord,
    ExecutionStatus,
    MeasurementContract,
    RasterRole,
    load_checkpoint,
    publish_checkpoint,
)
from seedvision.benchmarking.recipe_catalogue import build_recipe_catalogue
from seedvision.benchmarking.recipes import DependencyContract, RecipeSpec, StageRunRecord, StageRunState
from seedvision.benchmarking.reference_runner import PlantCVReferenceRunner
from seedvision.benchmarking.reference_mask_stage import ReferenceMaskStageOutcome, ReferenceMaskStageStatus
from seedvision.benchmarking.reference_stage import ReferenceStageOutcome
from seedvision.benchmarking.reference_preprocessing import (
    ReferencePreprocessingError,
    ReferencePreprocessingStageStatus,
    ReferencePreprocessingStatus,
    run_reference_preprocessing,
)


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PYTHON = ROOT / "artifacts" / "plantcv-reference-env" / "Scripts" / "python.exe"
REFERENCE_FILES = (
    REFERENCE_PYTHON,
    ROOT / "artifacts" / "reference-sources" / "plantcv-v4.11.3",
    ROOT / "artifacts" / "reference-sources" / "plantcv-source-audit.json",
    ROOT / "artifacts" / "reference-sources" / "plantcv-environment.json",
    ROOT / "config" / "plantcv-reference-requirements.txt",
)
HAS_REFERENCE_ENV = all(path.exists() for path in REFERENCE_FILES)
_GRAY_ROLE = RasterRole("image", np.dtype(np.uint8).str, "HW", (0, 255))


def _measurement(name: str) -> MeasurementContract:
    return MeasurementContract(
        definition_id=name,
        definition_version="1",
        mask_type="reference_preprocessing_raster",
        units="pixel",
        calibration_status="not_applicable",
        uncertainty="not_evaluated",
        eligible=False,
        reason="Original preprocessing output is not a validated biological measurement.",
    )


def _selected_recipe(recipe_id: str, *, median_ksize: int = 3, binary_cutoff: int = 128) -> RecipeSpec:
    recipe = build_recipe_catalogue().recipe(recipe_id)
    values = {
        "filter.median_blur": {"ksize": median_ksize},
        "threshold.binary": {"threshold": binary_cutoff, "object_type": "light"},
        "threshold.mean": {"ksize": 5, "offset": 0.0, "object_type": "light"},
        "threshold.otsu": {"object_type": "light"},
        "morphology.fill": {"size": 5},
        "morphology.erode": {"ksize": 3, "iterations": 1},
        "morphology.dilate": {"ksize": 3, "iterations": 1},
    }
    stages = []
    for stage in recipe.stages:
        selected = values.get(stage.operation_id)
        if selected is None:
            stages.append(stage)
            continue
        parameters = tuple(
            replace(parameter, value=selected[parameter.parameter_id])
            if parameter.parameter_id in selected else parameter
            for parameter in stage.parameters
        )
        stages.append(replace(stage, parameters=parameters))
    return replace(recipe, stages=tuple(stages))


def _gray_fixture(shape: tuple[int, int], *, empty: bool = False) -> np.ndarray:
    gray = np.zeros(shape, dtype=np.uint8)
    if empty:
        return gray
    gray.fill(32)
    height, width = shape
    gray[max(2, height // 5):max(3, 4 * height // 5), max(2, width // 5):max(3, 4 * width // 5)] = 205
    gray[height // 2, width // 2] = 255
    return np.ascontiguousarray(gray)


class _ReferencePreprocessingFixtures:
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="reference-preprocessing-contract-")
        self.root = Path(self.temporary.name)
        self.runner = PlantCVReferenceRunner(REFERENCE_PYTHON)
        self.case_counter = 0

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _arguments(
        self,
        recipe: RecipeSpec | None = None,
        *,
        shape: tuple[int, int] = (48, 72),
        empty: bool = False,
    ) -> dict[str, object]:
        self.case_counter += 1
        recipe = _selected_recipe("plantcv_seed_binary") if recipe is None else recipe
        base = self.root / f"case-{self.case_counter}"
        source_sha256 = sha256(f"reference-preprocessing-source-{self.case_counter}".encode()).hexdigest()
        frame = CoordinateFrame(
            frame_id="reference_preprocessing_gray_v1",
            shape=shape,
            origin="top_left_pixel_centres",
            axes="x_right_y_down",
            units="pixel",
            transform_sha256=sha256(b"identity grayscale fixture transform").hexdigest(),
        )
        input_identity = CheckpointIdentity(
            source_sha256=source_sha256,
            coordinate_frame=frame,
            stage_id="prepared_gray_input",
            recipe_id="seedfiddle_prepared_gray_fixture",
            recipe_version="1",
            recipe_sha256="a" * 64,
            graph_sha256="b" * 64,
            settings_sha256="c" * 64,
            backend_id="seedfiddle.fixture",
            backend_version="1",
            environment_versions={"numpy": np.__version__},
            image_semantics={
                "color_order": "GRAY",
                "dtype": "uint8",
                "value_range": [0, 255],
                "channel_conversion": "caller_supplied_gray",
                "geometric_transform": "fixture_identity_frame",
                "interpolation": "not_applicable",
            },
            mask_semantics={
                "foreground_polarity": "light",
                "representation": "uint8_0_255",
                "connectivity": "not_applicable",
                "validity_region": "full_frame",
                "border_policy": "not_applicable",
            },
            instance_semantics={
                "label_dtype": "int32",
                "background_label": 0,
                "label_id_mapping": "not_created",
                "split_boundary_ownership": "not_applicable",
                "visibility": "not_applicable",
                "eligibility": "not_evaluated",
            },
            upstream_checkpoint_sha256="d" * 64,
        )
        input_measurement = _measurement("prepared_gray")
        reference = publish_checkpoint(
            base / "input",
            input_identity,
            {"gray": _gray_fixture(shape, empty=empty)},
            execution=ExecutionRecord(ExecutionStatus.AVAILABLE),
            measurement=input_measurement,
            raster_roles={"gray": _GRAY_ROLE},
            metadata={"fixture_only": True},
            allow_cpu_transfer=True,
        )
        return {
            "input_root": base / "input",
            "input_checkpoint_sha256": reference.checkpoint_sha256,
            "recipe": recipe,
            "runner": self.runner,
            "output_root": base / "output",
            "expected_source_sha256": source_sha256,
            "expected_coordinate_frame": frame,
            "expected_recipe_identity": recipe.identity,
            "expected_input_identity": input_identity,
            "expected_input_measurement": input_measurement,
            "output_measurement": _measurement("plantcv_preprocessing_stage"),
            "input_bindings": {"gray": "gray"},
            "input_roles": {"gray": _GRAY_ROLE},
            "allow_cpu_transfer": True,
        }

class ReferencePreprocessingContractTests(_ReferencePreprocessingFixtures, unittest.TestCase):
    def test_invalid_preprocessing_edge_is_rejected_before_runner_execution(self) -> None:
        recipe = _selected_recipe("plantcv_seed_binary")
        median = next(stage for stage in recipe.stages if stage.operation_id == "filter.median_blur")
        threshold = next(stage for stage in recipe.stages if stage.operation_id == "threshold.binary")
        external = next(stage for stage in recipe.stages if stage.operation_id == "external.image_inputs")
        dependencies = tuple(
            edge for edge in recipe.dependencies
            if not (edge.source_stage == median.stage_id and edge.target_stage == threshold.stage_id)
        ) + (DependencyContract(external.stage_id, "gray_image", threshold.stage_id, "gray", "GrayImage"),)
        malformed = replace(recipe, dependencies=dependencies)
        arguments = self._arguments(malformed)
        with patch.object(self.runner, "run", side_effect=AssertionError("invalid topology must not run")) as run:
            with self.assertRaisesRegex(ReferencePreprocessingError, "dependencies"):
                run_reference_preprocessing(**arguments)
            run.assert_not_called()

    def test_full_recipe_identity_can_include_selected_downstream_values(self) -> None:
        recipe = _selected_recipe("plantcv_seed_binary")
        maxima = next(stage for stage in recipe.stages if stage.operation_id == "reference.internal.watershed.local_maxima")
        changed = replace(
            maxima,
            parameters=tuple(
                replace(parameter, value=4) if parameter.parameter_id == "distance" else parameter
                for parameter in maxima.parameters
            ),
        )
        recipe = replace(
            recipe,
            stages=tuple(changed if stage.stage_id == maxima.stage_id else stage for stage in recipe.stages),
        )
        arguments = self._arguments(recipe)
        with patch(
            "seedvision.benchmarking.reference_mask_stage.plantcv_reference_status",
            return_value={"available": False, "reason": "test source unavailable"},
        ), patch.object(self.runner, "run", side_effect=AssertionError("unavailable source must not run")) as run:
            outcome = run_reference_preprocessing(**arguments)
        self.assertEqual(outcome.recipe_identity, recipe.identity)
        self.assertEqual(outcome.status, ReferencePreprocessingStatus.UNAVAILABLE)
        self.assertEqual(outcome.stages[0].status, ReferencePreprocessingStageStatus.UNAVAILABLE)
        self.assertTrue(all(stage.status is ReferencePreprocessingStageStatus.NOT_RUN for stage in outcome.stages[1:]))
        self.assertEqual(outcome.stages[0].parameters, {"ksize": 3})
        self.assertEqual(outcome.stages[0].input_bindings, {"gray": "gray"})
        self.assertEqual(outcome.stages[0].input_roles, {"gray": _GRAY_ROLE})
        run.assert_not_called()

    def test_display_layout_changes_do_not_change_pinned_analysis_identity(self) -> None:
        recipe = _selected_recipe("plantcv_seed_binary")
        moved = replace(recipe.layout[0], x=recipe.layout[0].x + 120.0, y=recipe.layout[0].y + 40.0)
        changed_layout_recipe = replace(
            recipe,
            layout=(moved, *recipe.layout[1:]),
        )
        self.assertEqual(recipe.identity.analytical_dict(), changed_layout_recipe.identity.analytical_dict())
        self.assertNotEqual(recipe.identity.layout_sha256, changed_layout_recipe.identity.layout_sha256)
        arguments = self._arguments(changed_layout_recipe)
        with patch(
            "seedvision.benchmarking.reference_mask_stage.plantcv_reference_status",
            return_value={"available": False, "reason": "test source unavailable"},
        ):
            outcome = run_reference_preprocessing(**arguments)
        self.assertEqual(outcome.status, ReferencePreprocessingStatus.UNAVAILABLE)
        self.assertEqual(outcome.recipe_identity, changed_layout_recipe.identity)
        self.assertEqual(outcome.stages[0].run_record.recipe_identity, changed_layout_recipe.identity)

    def test_median_boundary_rejection_stops_before_any_plantcv_call(self) -> None:
        recipe = _selected_recipe("plantcv_seed_binary", median_ksize=129)
        arguments = self._arguments(recipe, shape=(16, 16))
        with patch.object(self.runner, "run", side_effect=AssertionError("undefined median boundary must not run")) as run:
            outcome = run_reference_preprocessing(**arguments)
        self.assertEqual(outcome.status, ReferencePreprocessingStatus.UNSUPPORTED)
        self.assertEqual(outcome.stages[0].status, ReferencePreprocessingStageStatus.UNSUPPORTED)
        self.assertIn("source-undefined", outcome.stages[0].run_record.reason)
        self.assertTrue(all(stage.status is ReferencePreprocessingStageStatus.NOT_RUN for stage in outcome.stages[1:]))
        run.assert_not_called()

    def test_pre_cancelled_chain_records_cancellation_and_downstream_not_run(self) -> None:
        arguments = self._arguments()
        cancel = threading.Event()
        cancel.set()
        arguments["cancel"] = cancel
        with patch.object(self.runner, "run", side_effect=AssertionError("cancelled chain must not run")) as run:
            outcome = run_reference_preprocessing(**arguments)
        self.assertEqual(outcome.status, ReferencePreprocessingStatus.CANCELLED)
        self.assertEqual(outcome.stages[0].status, ReferencePreprocessingStageStatus.CANCELLED)
        self.assertEqual(outcome.stages[0].run_record.state, StageRunState.FAILED)
        self.assertTrue(all(stage.status is ReferencePreprocessingStageStatus.NOT_RUN for stage in outcome.stages[1:]))
        run.assert_not_called()

    def test_failed_unavailable_or_cancelled_dilation_has_no_partial_final_checkpoint(self) -> None:
        recipe = _selected_recipe("plantcv_seed_binary")
        final_stage = next(stage for stage in recipe.stages if stage.operation_id == "morphology.dilate")
        cases = (
            (ReferenceMaskStageStatus.FAILED, StageRunState.FAILED, "PlantCV execution failed", ReferencePreprocessingStatus.FAILED),
            (ReferenceMaskStageStatus.UNAVAILABLE, StageRunState.UNAVAILABLE, "Pinned source unavailable", ReferencePreprocessingStatus.UNAVAILABLE),
            (ReferenceMaskStageStatus.CANCELLED, StageRunState.FAILED, "Cancelled before PlantCV execution", ReferencePreprocessingStatus.CANCELLED),
        )

        for mask_status, run_state, reason, overall_status in cases:
            with self.subTest(status=mask_status.value):
                arguments = self._arguments(recipe)

                def successful_reference(kwargs: dict[str, object]):
                    stage_id = kwargs["stage_id"]
                    source_sha = f"{stage_id}:{kwargs['expected_upstream_checkpoint_sha256']}".encode()
                    digest = sha256(source_sha).hexdigest()
                    record = StageRunRecord(
                        stage_id=stage_id,
                        recipe_identity=recipe.identity,
                        state=StageRunState.EXECUTED,
                        checkpoint_sha256=digest,
                        upstream_checkpoint_sha256=kwargs["expected_upstream_checkpoint_sha256"],
                    )
                    reference = CheckpointReference(
                        digest,
                        Path("synthetic-reference-checkpoint"),
                        kwargs["expected_output_identity"],
                        ExecutionRecord(ExecutionStatus.AVAILABLE),
                    )
                    if recipe.stage(stage_id).operation_id.startswith("threshold."):
                        return ReferenceStageOutcome(record, reference)
                    return ReferenceMaskStageOutcome(ReferenceMaskStageStatus.EXECUTED, record, reference)

                def mask_result(**kwargs: object):
                    stage_id = kwargs["stage_id"]
                    if stage_id != final_stage.stage_id:
                        return successful_reference(kwargs)
                    record = StageRunRecord(
                        stage_id=stage_id,
                        recipe_identity=recipe.identity,
                        state=run_state,
                        upstream_checkpoint_sha256=kwargs["expected_upstream_checkpoint_sha256"],
                        reason=reason,
                    )
                    return ReferenceMaskStageOutcome(mask_status, record, None)

                with patch(
                    "seedvision.benchmarking.reference_preprocessing.run_reference_mask_stage",
                    side_effect=mask_result,
                ) as mask_bridge, patch(
                    "seedvision.benchmarking.reference_preprocessing.run_reference_stage",
                    side_effect=lambda **kwargs: successful_reference(kwargs),
                ) as threshold_bridge:
                    outcome = run_reference_preprocessing(**arguments)

                self.assertEqual(outcome.status, overall_status)
                self.assertEqual(outcome.stages[-1].stage.stage_id, final_stage.stage_id)
                self.assertEqual(outcome.stages[-1].status.value, mask_status.value)
                self.assertIsNone(outcome.stages[-1].checkpoint_reference)
                self.assertIsNone(outcome.final_checkpoint)
                self.assertEqual(outcome.latest_checkpoint, outcome.stages[-2].checkpoint_reference)
                self.assertIsNotNone(outcome.latest_checkpoint)
                self.assertEqual(mask_bridge.call_count, 4)
                threshold_bridge.assert_called_once()


@unittest.skipUnless(HAS_REFERENCE_ENV, "the ignored pinned PlantCV reference environment is unavailable")
class ReferencePreprocessingPlantCVTests(_ReferencePreprocessingFixtures, unittest.TestCase):
    def test_all_three_authored_chains_run_original_plantcv_and_preserve_stage_provenance(self) -> None:
        recipes = (
            "plantcv_seed_binary",
            "plantcv_seed_mean",
            "plantcv_seed_otsu",
        )
        for recipe_id in recipes:
            with self.subTest(recipe=recipe_id):
                recipe = _selected_recipe(recipe_id)
                arguments = self._arguments(recipe)
                outcome = run_reference_preprocessing(**arguments)
                self.assertIn(outcome.status, {
                    ReferencePreprocessingStatus.EXECUTED,
                    ReferencePreprocessingStatus.EMPTY,
                })
                self.assertEqual(
                    [stage.stage.operation_id for stage in outcome.stages],
                    [
                        "filter.median_blur",
                        _threshold_operation(recipe_id),
                        "morphology.fill",
                        "morphology.erode",
                        "morphology.dilate",
                    ],
                )
                self.assertEqual(len(outcome.run_records), 5)
                self.assertEqual(outcome.stages[0].status, ReferencePreprocessingStageStatus.EXECUTED)
                self.assertTrue(all(stage.status in {
                    ReferencePreprocessingStageStatus.EXECUTED,
                    ReferencePreprocessingStageStatus.EMPTY,
                } for stage in outcome.stages))
                for index, stage_result in enumerate(outcome.stages):
                    reference = stage_result.checkpoint_reference
                    self.assertIsNotNone(reference)
                    self.assertEqual(stage_result.run_record.checkpoint_sha256, reference.checkpoint_sha256)
                    self.assertEqual(stage_result.run_record.recipe_identity, recipe.identity)
                    self.assertEqual(stage_result.input_checkpoint_sha256, (
                        arguments["input_checkpoint_sha256"] if index == 0
                        else outcome.stages[index - 1].checkpoint_reference.checkpoint_sha256
                    ))
                    checkpoint = load_checkpoint(arguments["output_root"], reference.checkpoint_sha256)
                    self.assertEqual(checkpoint.metadata["operation_id"], stage_result.stage.operation_id)
                    self.assertEqual(checkpoint.metadata["parameters"], recipe.parameters_for_runner(stage_result.stage.stage_id))
                    self.assertFalse(checkpoint.measurement.eligible)
                    self.assertIn("source_mapping", checkpoint.metadata)
                final = outcome.final_checkpoint
                self.assertIsNotNone(final)
                final_checkpoint = load_checkpoint(arguments["output_root"], final.checkpoint_sha256)
                self.assertEqual(set(final_checkpoint.arrays), {"mask"})
                self.assertEqual(final_checkpoint.raster_roles["mask"], RasterRole("mask", np.dtype(np.uint8).str))

    def test_changed_preprocessing_parameter_changes_checkpoint_and_upstream_chain(self) -> None:
        baseline_args = self._arguments(_selected_recipe("plantcv_seed_binary", median_ksize=3))
        changed_args = self._arguments(_selected_recipe("plantcv_seed_binary", median_ksize=5))
        baseline = run_reference_preprocessing(**baseline_args)
        changed = run_reference_preprocessing(**changed_args)
        self.assertIn(baseline.status, {ReferencePreprocessingStatus.EXECUTED, ReferencePreprocessingStatus.EMPTY})
        self.assertIn(changed.status, {ReferencePreprocessingStatus.EXECUTED, ReferencePreprocessingStatus.EMPTY})
        self.assertNotEqual(
            baseline.stages[0].checkpoint_reference.checkpoint_sha256,
            changed.stages[0].checkpoint_reference.checkpoint_sha256,
        )
        self.assertNotEqual(
            baseline.stages[1].input_checkpoint_sha256,
            changed.stages[1].input_checkpoint_sha256,
        )
        self.assertNotEqual(
            baseline.stages[1].checkpoint_reference.identity.settings_sha256,
            changed.stages[1].checkpoint_reference.identity.settings_sha256,
        )

    def test_empty_threshold_masks_continue_through_fill_erosion_and_dilation(self) -> None:
        recipe = _selected_recipe("plantcv_seed_binary", binary_cutoff=255)
        arguments = self._arguments(recipe, empty=True)
        outcome = run_reference_preprocessing(**arguments)
        self.assertEqual(outcome.status, ReferencePreprocessingStatus.EMPTY)
        self.assertEqual(outcome.stages[0].status, ReferencePreprocessingStageStatus.EXECUTED)
        self.assertTrue(all(
            stage.status is ReferencePreprocessingStageStatus.EMPTY
            for stage in outcome.stages[1:]
        ))
        self.assertEqual(len(outcome.run_records), 5)
        self.assertIsNotNone(outcome.final_checkpoint)
        final = load_checkpoint(arguments["output_root"], outcome.final_checkpoint.checkpoint_sha256)
        self.assertFalse(np.any(final.arrays["mask"]))


def _threshold_operation(recipe_id: str) -> str:
    return {
        "plantcv_seed_binary": "threshold.binary",
        "plantcv_seed_mean": "threshold.mean",
        "plantcv_seed_otsu": "threshold.otsu",
    }[recipe_id]


if __name__ == "__main__":
    unittest.main()

