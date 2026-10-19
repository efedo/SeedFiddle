from __future__ import annotations

from tests.catalogue_fixtures import CatalogueTestCase

from tests.functional_units import functional_units

from dataclasses import replace
from hashlib import sha256
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np

import seedvision.benchmarking.checkpoints as checkpoint_module
from seedvision.benchmarking.checkpoints import (
    CheckpointIdentity,
    CheckpointMismatch,
    CoordinateFrame,
    ExecutionRecord,
    ExecutionStatus,
    MeasurementContract,
    RasterRole,
    load_checkpoint,
    publish_checkpoint,
)
from seedvision.benchmarking.recipe_catalogue import build_recipe_catalogue
from seedvision.benchmarking.recipes import (
    RecipeProvenanceMode,
    SourceResolution,
    StageRunState,
)
from seedvision.benchmarking.reference_runner import (
    PlantCVReferenceRunner,
    ReferenceExecutionError,
)
import seedvision.benchmarking.reference_stage as reference_stage_module
from seedvision.benchmarking.reference_stage import (
    ReferenceStageError,
    expected_reference_stage_identity,
    run_reference_stage,
)


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PYTHON = ROOT / "artifacts" / "plantcv-reference-env" / "Scripts" / "python.exe"
SOURCE_ROOT = ROOT / "artifacts" / "reference-sources" / "plantcv-v4.11.3"
SOURCE_MANIFEST = ROOT / "artifacts" / "reference-sources" / "plantcv-source-audit.json"
ENVIRONMENT_MANIFEST = ROOT / "artifacts" / "reference-sources" / "plantcv-environment.json"
REQUIREMENTS_LOCK = ROOT / "config" / "plantcv-reference-requirements.txt"
HAS_REFERENCE_ENV = all(path.exists() for path in (
    REFERENCE_PYTHON, SOURCE_ROOT, SOURCE_MANIFEST, ENVIRONMENT_MANIFEST, REQUIREMENTS_LOCK,
))


def _recipe(operation_id: str = "threshold.otsu"):
    catalogue = build_recipe_catalogue()
    recipe_id = {
        "threshold.binary": "plantcv_seed_binary",
        "threshold.mean": "plantcv_seed_mean",
        "threshold.otsu": "plantcv_seed_otsu",
    }[operation_id]
    recipe = catalogue.recipe(recipe_id)
    stage = next(item for item in recipe.stages if item.operation_id == operation_id)
    return recipe, stage


def _measurement(name: str) -> MeasurementContract:
    return MeasurementContract(
        definition_id=name,
        definition_version="1",
        mask_type="binary_threshold_output",
        units="pixel",
        calibration_status="not_applicable",
        uncertainty="not_evaluated",
        eligible=False,
        reason="This benchmark stage does not certify a biological measurement.",
    )


def _image_semantics() -> dict[str, object]:
    return {
        "color_order": "GRAY",
        "dtype": "uint8",
        "value_range": [0, 255],
        "channel_conversion": "caller_supplied_gray",
        "geometric_transform": "fixture_identity_frame",
        "interpolation": "not_applicable",
    }


def _input_identity(frame: CoordinateFrame, source_sha256: str) -> CheckpointIdentity:
    return CheckpointIdentity(
        source_sha256=source_sha256,
        coordinate_frame=frame,
        stage_id="prepared_gray",
        recipe_id="prepared_image_fixture",
        recipe_version="1",
        recipe_sha256="a" * 64,
        graph_sha256="b" * 64,
        settings_sha256="c" * 64,
        backend_id="seedfiddle.fixture",
        backend_version="1",
        environment_versions={"python": "3.12.10"},
        image_semantics=_image_semantics(),
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


def _gray_image(*, empty: bool = False) -> np.ndarray:
    image = np.zeros((48, 72), dtype=np.uint8)
    if empty:
        return image
    image.fill(32)
    yy, xx = np.ogrid[:48, :72]
    image[(xx - 36) ** 2 + (yy - 24) ** 2 < 14**2] = 210
    return image


def _fixture(
    temporary: str,
    *,
    operation_id: str = "threshold.otsu",
    empty: bool = False,
    recipe_override=None,
):
    recipe, stage = _recipe(operation_id) if recipe_override is None else (recipe_override, None)
    if stage is None:
        stage = next(item for item in recipe.stages if item.operation_id == operation_id)
    source_sha256 = sha256(b"synthetic engineering-only source").hexdigest()
    frame = CoordinateFrame(
        frame_id="synthetic_gray_v1",
        shape=(48, 72),
        origin="top_left_pixel_centres",
        axes="x_right_y_down",
        units="pixel",
        transform_sha256=sha256(b"identity transform").hexdigest(),
    )
    input_identity = _input_identity(frame, source_sha256)
    input_root = Path(temporary) / "input"
    output_root = Path(temporary) / "output"
    input_measurement = _measurement("prepared_gray_input")
    input_reference = publish_checkpoint(
        input_root,
        input_identity,
        {"gray": _gray_image(empty=empty)},
        execution=ExecutionRecord(ExecutionStatus.AVAILABLE),
        measurement=input_measurement,
        raster_roles={"gray": RasterRole("image", np.dtype(np.uint8).str, "HW", (0, 255))},
        metadata={"fixture_only": True},
        allow_cpu_transfer=True,
    )
    output_measurement = _measurement("plantcv_threshold_output")
    output_identity = expected_reference_stage_identity(
        input_identity,
        recipe,
        stage.stage_id,
        input_reference.checkpoint_sha256,
    )
    args = {
        "input_root": input_root,
        "input_checkpoint_sha256": input_reference.checkpoint_sha256,
        "recipe": recipe,
        "stage_id": stage.stage_id,
        "runner": PlantCVReferenceRunner(REFERENCE_PYTHON),
        "output_root": output_root,
        "expected_source_sha256": source_sha256,
        "expected_coordinate_frame": frame,
        "expected_recipe_identity": recipe.identity,
        "expected_upstream_checkpoint_sha256": input_reference.checkpoint_sha256,
        "expected_input_identity": input_identity,
        "expected_input_measurement": input_measurement,
        "expected_output_identity": output_identity,
        "input_bindings": {"gray": "gray"},
        "input_roles": {"gray": RasterRole("image", np.dtype(np.uint8).str, "HW", (0, 255))},
        "output_measurement": output_measurement,
        "allow_cpu_transfer": True,
    }
    return args, recipe, stage, input_reference, output_measurement


@functional_units
class ReferenceStageContractTests(CatalogueTestCase):
    def check_test_exact_replay_bindings_are_required_before_runner_call(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args, recipe, _stage, _input, _measurement_contract = _fixture(temporary)
            runner = args["runner"]
            with patch.object(runner, "run", side_effect=AssertionError("must not run")) as run:
                bad = dict(args, expected_source_sha256="f" * 64)
                with self.assertRaisesRegex(ReferenceStageError, "source SHA"):
                    run_reference_stage(**bad)
                bad = dict(args, expected_upstream_checkpoint_sha256="e" * 64)
                with self.assertRaisesRegex(ReferenceStageError, "upstream digest"):
                    run_reference_stage(**bad)
                bad = dict(
                    args,
                    expected_coordinate_frame=replace(args["expected_coordinate_frame"], frame_id="stale_frame"),
                )
                with self.assertRaisesRegex(ReferenceStageError, "frame"):
                    run_reference_stage(**bad)
                bad = dict(
                    args,
                    expected_input_measurement=_measurement("different_input_definition"),
                )
                with self.assertRaisesRegex(CheckpointMismatch, "measurement"):
                    run_reference_stage(**bad)
                bad = dict(args, input_bindings={"gray": "wrong_name"})
                with self.assertRaisesRegex(ReferenceStageError, "input raster is missing"):
                    run_reference_stage(**bad)
                bad = dict(args, input_roles={"gray": RasterRole("raster", np.dtype(np.uint8).str, "HW", (0, 255))})
                with self.assertRaisesRegex(ReferenceStageError, "role differs"):
                    run_reference_stage(**bad)
                bad = dict(args, output_measurement=replace(args["output_measurement"], eligible=True, reason=""))
                with self.assertRaisesRegex(ReferenceStageError, "cannot mark scientific"):
                    run_reference_stage(**bad)
                self.assertEqual(run.call_count, 0)

    def check_test_stale_recipe_parameter_source_and_environment_pins_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args, recipe, stage, _input, _measurement_contract = _fixture(temporary)
            runner = args["runner"]
            with patch.object(runner, "run", side_effect=AssertionError("must not run")) as run:
                changed_parameters = tuple(
                    replace(parameter, value="dark") if parameter.parameter_id == "object_type" else parameter
                    for parameter in stage.parameters
                )
                changed_stage = replace(stage, parameters=changed_parameters)
                changed_recipe = replace(
                    recipe,
                    stages=tuple(changed_stage if item.stage_id == stage.stage_id else item for item in recipe.stages),
                )
                with self.assertRaisesRegex(ReferenceStageError, "expected recipe identity"):
                    run_reference_stage(**dict(args, recipe=changed_recipe))

                changed_mapping = replace(stage.source_mappings[0], source_sha256="0" * 64)
                changed_stage = replace(stage, source_mappings=(changed_mapping,))
                changed_recipe = replace(
                    recipe,
                    stages=tuple(changed_stage if item.stage_id == stage.stage_id else item for item in recipe.stages),
                )
                with self.assertRaisesRegex(ReferenceStageError, "source mapping differs"):
                    run_reference_stage(**dict(
                        args,
                        recipe=changed_recipe,
                        expected_recipe_identity=changed_recipe.identity,
                    ))

                changed_provenance = replace(
                    recipe.provenance,
                    environment_versions=dict(recipe.provenance.environment_versions) | {"plantcv": "9.9.9"},
                )
                changed_recipe = replace(recipe, provenance=changed_provenance)
                with self.assertRaisesRegex(ReferenceStageError, "provenance differs"):
                    run_reference_stage(**dict(
                        args,
                        recipe=changed_recipe,
                        expected_recipe_identity=changed_recipe.identity,
                    ))
                self.assertEqual(run.call_count, 0)

    def check_test_unselected_required_cutoff_is_rejected(self) -> None:
        with self.assertRaisesRegex(ReferenceStageError, "unselected"):
            recipe, stage = _recipe("threshold.binary")
            # The catalogue intentionally leaves the binary cutoff unset.
            expected_reference_stage_identity(
                _input_identity(
                    CoordinateFrame("frame", (48, 72), "top_left", "x_right_y_down", "pixel"),
                    "a" * 64,
                ),
                recipe,
                stage.stage_id,
                "b" * 64,
            )

    def check_test_mean_offset_that_binary64_would_round_is_rejected_before_execution(self) -> None:
        recipe, stage = _recipe("threshold.mean")
        changed_stage = replace(
            stage,
            parameters=tuple(
                replace(parameter, value=9007199254740993)
                if parameter.parameter_id == "offset"
                else replace(parameter, value=3)
                if parameter.parameter_id == "ksize"
                else parameter
                for parameter in stage.parameters
            ),
        )
        selected_recipe = replace(
            recipe,
            stages=tuple(changed_stage if item.stage_id == stage.stage_id else item for item in recipe.stages),
        )
        frame = CoordinateFrame("frame", (48, 72), "top_left", "x_right_y_down", "pixel")
        with self.assertRaisesRegex(ReferenceStageError, "not exactly representable"):
            expected_reference_stage_identity(
                _input_identity(frame, "a" * 64),
                selected_recipe,
                stage.stage_id,
                "b" * 64,
            )

    def check_test_explicit_and_inherited_timeout_values_are_bounded_before_runner_call(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args, _recipe_value, _stage, _input, _measurement_contract = _fixture(temporary)
            runner = args["runner"]
            invalid_explicit = (0, -1, 3600.01, True, float("nan"), float("inf"))
            with patch.object(runner, "run", side_effect=AssertionError("invalid timeout must not run")) as run:
                for value in invalid_explicit:
                    with self.subTest(explicit=value), self.assertRaisesRegex(ReferenceStageError, "timeout_seconds"):
                        run_reference_stage(**dict(args, timeout_seconds=value))
                self.assertEqual(run.call_count, 0)

                for value in (3601, float("nan"), True):
                    runner.timeout_seconds = value
                    with self.subTest(inherited=value), self.assertRaisesRegex(ReferenceStageError, "timeout_seconds"):
                        run_reference_stage(**args)
                self.assertEqual(run.call_count, 0)

    def check_test_valid_explicit_and_inherited_timeout_boundaries_reach_runner(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args, _recipe_value, _stage, _input, _measurement_contract = _fixture(temporary)
            runner = args["runner"]
            with (
                patch.object(reference_stage_module, "plantcv_reference_status", return_value={"available": True}),
                patch.object(runner, "run", side_effect=ReferenceExecutionError("stop after timeout check")) as run,
            ):
                for value in (0.001, 3600):
                    with self.subTest(explicit=value):
                        outcome = run_reference_stage(**dict(args, timeout_seconds=value))
                        self.assertEqual(outcome.run_record.state, StageRunState.FAILED)
                        self.assertEqual(run.call_args.kwargs["timeout_seconds"], float(value))
                runner.timeout_seconds = 3600
                outcome = run_reference_stage(**args)
                self.assertEqual(outcome.run_record.state, StageRunState.FAILED)
                self.assertEqual(run.call_args.kwargs["timeout_seconds"], 3600.0)
                self.assertEqual(run.call_count, 3)

    def check_test_internal_or_wrong_provenance_stages_are_marked_unsupported_without_running(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args, recipe, stage, _input, _measurement_contract = _fixture(temporary)
            runner = args["runner"]
            with patch.object(runner, "run", side_effect=AssertionError("must not run")) as run:
                private_stage = replace(stage, operation_id="segmentation.watershed.markers")
                private_recipe = replace(
                    recipe,
                    stages=tuple(private_stage if item.stage_id == stage.stage_id else item for item in recipe.stages),
                )
                outcome = run_reference_stage(**dict(
                    args,
                    recipe=private_recipe,
                    expected_recipe_identity=private_recipe.identity,
                    expected_output_identity=None,
                ))
                self.assertEqual(outcome.run_record.state, StageRunState.UNSUPPORTED)
                self.assertIsNone(outcome.checkpoint_reference)

                for mode in (
                    RecipeProvenanceMode.NATIVE_COMPATIBLE,
                    RecipeProvenanceMode.ENHANCED,
                    RecipeProvenanceMode.PAPER_RECONSTRUCTION,
                ):
                    wrong_mode_recipe = replace(
                        recipe,
                        provenance=replace(recipe.provenance, mode=mode),
                    )
                    outcome = run_reference_stage(**dict(
                        args,
                        recipe=wrong_mode_recipe,
                        expected_recipe_identity=wrong_mode_recipe.identity,
                        expected_output_identity=None,
                    ))
                    self.assertEqual(outcome.run_record.state, StageRunState.UNSUPPORTED)
                    self.assertIsNone(outcome.checkpoint_reference)
                self.assertEqual(run.call_count, 0)

    def check_test_unavailable_stage_stays_explicitly_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args, recipe, stage, _input, _measurement_contract = _fixture(temporary)
            unavailable_stage = replace(
                stage,
                resolution=SourceResolution.UNAVAILABLE,
                resolution_note="The original operation source is unavailable.",
            )
            unavailable_recipe = replace(
                recipe,
                stages=tuple(unavailable_stage if item.stage_id == stage.stage_id else item for item in recipe.stages),
            )
            with patch.object(args["runner"], "run", side_effect=AssertionError("must not run")) as run:
                outcome = run_reference_stage(**dict(
                    args,
                    recipe=unavailable_recipe,
                    expected_recipe_identity=unavailable_recipe.identity,
                    expected_output_identity=None,
                ))
            self.assertEqual(outcome.run_record.state, StageRunState.UNAVAILABLE)
            self.assertIsNone(outcome.checkpoint_reference)
            run.assert_not_called()

    def check_test_runner_exception_and_pre_execution_cancellation_are_failed_records(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args, recipe, _stage, _input, _measurement_contract = _fixture(temporary)
            runner = args["runner"]
            with (
                patch.object(reference_stage_module, "plantcv_reference_status", return_value={"available": True}),
                patch.object(runner, "run", side_effect=ReferenceExecutionError("synthetic runner failure")) as run,
            ):
                outcome = run_reference_stage(**args)
            self.assertEqual(outcome.run_record.state, StageRunState.FAILED)
            self.assertIn("synthetic runner failure", outcome.run_record.reason)
            self.assertIsNone(outcome.checkpoint_reference)
            run.assert_called_once()

            with patch.object(runner, "run", side_effect=AssertionError("cancelled input must not run")) as run:
                cancelled = threading.Event()
                cancelled.set()
                outcome = run_reference_stage(**dict(args, cancel=cancelled))
            self.assertEqual(outcome.run_record.state, StageRunState.FAILED)
            self.assertIn("Cancelled", outcome.run_record.reason)
            self.assertIsNone(outcome.checkpoint_reference)
            run.assert_not_called()

    def check_test_missing_reference_interpreter_is_unavailable_not_failed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args, _recipe_value, _stage, _input, _measurement_contract = _fixture(temporary)
            missing_runner = PlantCVReferenceRunner(Path(temporary) / "missing-reference-python.exe")
            args["runner"] = missing_runner
            with patch.object(missing_runner, "run", side_effect=AssertionError("must not run")) as run:
                outcome = run_reference_stage(**args)
            self.assertEqual(outcome.run_record.state, StageRunState.UNAVAILABLE)
            self.assertIn("does not exist", outcome.run_record.reason)
            self.assertIsNone(outcome.checkpoint_reference)
            run.assert_not_called()

    def test_workflow_lifecycle(self):
        """3 related ReferenceStageContractTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_exact_replay_bindings_are_required_before_runner_call)
        self.scenario(self.check_test_explicit_and_inherited_timeout_values_are_bounded_before_runner_call)
        self.scenario(self.check_test_valid_explicit_and_inherited_timeout_boundaries_reach_runner)

    def test_validation_and_recovery(self):
        """7 related ReferenceStageContractTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_stale_recipe_parameter_source_and_environment_pins_are_rejected)
        self.scenario(self.check_test_unselected_required_cutoff_is_rejected)
        self.scenario(self.check_test_mean_offset_that_binary64_would_round_is_rejected_before_execution)
        self.scenario(self.check_test_internal_or_wrong_provenance_stages_are_marked_unsupported_without_running)
        self.scenario(self.check_test_unavailable_stage_stays_explicitly_unavailable)
        self.scenario(self.check_test_runner_exception_and_pre_execution_cancellation_are_failed_records)
        self.scenario(self.check_test_missing_reference_interpreter_is_unavailable_not_failed)


@unittest.skipUnless(HAS_REFERENCE_ENV, "pinned PlantCV reference source/environment is not provisioned")
class IsolatedReferenceStageTests(unittest.TestCase):


    pass


if __name__ == "__main__":
    unittest.main()
