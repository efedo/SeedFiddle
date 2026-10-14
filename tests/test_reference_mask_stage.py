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
    ParameterContract,
    RecipeSpec,
    StageRunState,
)
from seedvision.benchmarking.reference_runner import (
    PlantCVReferenceRunner,
    ReferenceExecutionError,
    ReferenceResult,
)
from seedvision.benchmarking.reference_mask_stage import (
    ReferenceMaskStageError,
    ReferenceMaskStageStatus,
    expected_reference_mask_stage_identity,
    run_reference_mask_stage,
)


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PYTHON = ROOT / "artifacts" / "plantcv-reference-env" / "Scripts" / "python.exe"
HAS_REFERENCE_ENV = all(path.exists() for path in (
    REFERENCE_PYTHON,
    ROOT / "artifacts" / "reference-sources" / "plantcv-v4.11.3",
    ROOT / "artifacts" / "reference-sources" / "plantcv-source-audit.json",
    ROOT / "artifacts" / "reference-sources" / "plantcv-environment.json",
    ROOT / "config" / "plantcv-reference-requirements.txt",
))


def _measurement(name: str) -> MeasurementContract:
    return MeasurementContract(
        definition_id=name,
        definition_version="1",
        mask_type="reference_image_or_binary_mask",
        units="pixel",
        calibration_status="not_applicable",
        uncertainty="not_evaluated",
        eligible=False,
        reason="A one-call PlantCV artifact does not validate scientific measurements.",
    )


def _fixture_image() -> np.ndarray:
    image = np.full((48, 72), 32, dtype=np.uint8)
    image[8:38, 14:58] = 190
    image[20, 36] = 255
    image[42, 8] = 255
    return image


def _fixture_mask() -> np.ndarray:
    mask = np.zeros((48, 72), dtype=np.uint8)
    mask[8:37, 15:55] = 255
    mask[20:25, 28:43] = 0
    mask[2:4, 2:4] = 255
    return mask


@unittest.skipUnless(HAS_REFERENCE_ENV, "the ignored pinned PlantCV reference environment is unavailable")
class ReferenceMaskStageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runner = PlantCVReferenceRunner(REFERENCE_PYTHON)
        cls.catalogue = build_recipe_catalogue()
        cls.temporary = tempfile.TemporaryDirectory(prefix="reference-mask-stage-")
        cls.root = Path(cls.temporary.name)
        cls.case_counter = 0
        cls.frame = CoordinateFrame(
            frame_id="mask_stage_gray_fixture_v1",
            shape=(48, 72),
            origin="top_left_pixel_centres",
            axes="x_right_y_down",
            units="pixel",
            transform_sha256=sha256(b"mask stage identity frame").hexdigest(),
        )
        cls.source_sha256 = sha256(b"mask stage synthetic engineering fixture").hexdigest()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def _recipe(self, operation: str, values: dict[str, object] | None = None) -> tuple[RecipeSpec, str]:
        recipe = self.catalogue.recipe("plantcv_seed_binary")
        stage = next(item for item in recipe.stages if item.operation_id == operation)
        values = {} if values is None else values
        updated = replace(
            stage,
            parameters=tuple(
                replace(parameter, value=values[parameter.parameter_id])
                if parameter.parameter_id in values else parameter
                for parameter in stage.parameters
            ),
        )
        recipe = replace(
            recipe,
            stages=tuple(updated if item.stage_id == stage.stage_id else item for item in recipe.stages),
        )
        return recipe, stage.stage_id

    def _case(
        self,
        operation: str,
        values: dict[str, object] | None = None,
        *,
        array: np.ndarray | None = None,
        case_name: str | None = None,
        recipe: RecipeSpec | None = None,
        stage_id: str | None = None,
    ) -> dict[str, object]:
        if recipe is None:
            recipe, stage_id = self._recipe(operation, values)
        assert stage_id is not None
        port = "gray" if operation in {"filter.median_blur", "morphology.erode", "morphology.dilate"} else "mask"
        array_name = f"{port}_source"
        is_image = operation == "filter.median_blur"
        source_array = array if array is not None else (_fixture_image() if is_image else _fixture_mask())
        frame = replace(self.frame, shape=tuple(source_array.shape))
        role = (
            RasterRole("image", np.dtype(np.uint8).str, "HW", (0, 255))
            if is_image else RasterRole("mask", np.dtype(np.uint8).str, "HW")
        )
        image_semantics = {
            "color_order": "GRAY",
            "dtype": "uint8",
            "value_range": [0, 255],
            "channel_conversion": "fixture_grayscale_boundary",
            "geometric_transform": "identity_frame",
            "interpolation": "not_applicable",
        }
        mask_semantics = {
            "foreground_polarity": "light",
            "representation": "uint8_0_255",
            "connectivity": "source_defined_binary_foreground",
            "validity_region": "full_input_frame",
            "border_policy": "prepared_input",
        }
        input_identity = CheckpointIdentity(
            source_sha256=self.source_sha256,
            coordinate_frame=frame,
            stage_id="prepared_mask_stage_input",
            recipe_id="seedfiddle_mask_stage_fixture",
            recipe_version="1",
            recipe_sha256="a" * 64,
            graph_sha256="b" * 64,
            settings_sha256="c" * 64,
            backend_id="seedfiddle.fixture",
            backend_version="1",
            environment_versions={"numpy": np.__version__},
            image_semantics=image_semantics,
            mask_semantics=mask_semantics,
            instance_semantics={
                "label_dtype": "<i4",
                "background_label": 0,
                "label_id_mapping": "not_created",
                "split_boundary_ownership": "not_applicable",
                "visibility": "not_applicable",
                "eligibility": "not_evaluated",
            },
            upstream_checkpoint_sha256="d" * 64,
        )
        self.case_counter += 1
        base = self.root / (case_name or f"{operation.replace('.', '-')}-{self._testMethodName}-{self.case_counter}")
        input_root = base / "input"
        output_root = base / "output"
        input_execution = (
            ExecutionStatus.EMPTY if not is_image and not np.any(source_array)
            else ExecutionStatus.AVAILABLE
        )
        input_reference = publish_checkpoint(
            input_root,
            input_identity,
            {array_name: source_array},
            execution=ExecutionRecord(input_execution),
            measurement=_measurement("prepared_input"),
            raster_roles={array_name: role},
            metadata={"fixture_only": True},
            allow_cpu_transfer=True,
        )
        output_identity = None
        if operation in {"filter.median_blur", "morphology.fill", "morphology.erode", "morphology.dilate"}:
            try:
                output_identity = expected_reference_mask_stage_identity(
                    input_identity, recipe, stage_id, input_reference.checkpoint_sha256
                )
            except ReferenceMaskStageError:
                # Unsupported parameter domains still produce an honest stage record.
                output_identity = None
        output_measurement = _measurement("plantcv_original_mask_stage_output")
        return {
            "input_root": input_root,
            "input_checkpoint_sha256": input_reference.checkpoint_sha256,
            "recipe": recipe,
            "stage_id": stage_id,
            "runner": self.runner,
            "output_root": output_root,
            "expected_source_sha256": self.source_sha256,
            "expected_coordinate_frame": frame,
            "expected_recipe_identity": recipe.identity,
            "expected_upstream_checkpoint_sha256": input_reference.checkpoint_sha256,
            "expected_input_identity": input_identity,
            "expected_input_measurement": _measurement("prepared_input"),
            "expected_output_identity": output_identity,
            "input_bindings": {port: array_name},
            "input_roles": {port: role},
            "output_measurement": output_measurement,
            "allow_cpu_transfer": True,
            "input_reference": input_reference,
        }

    def _run(self, case: dict[str, object]):
        args = {key: value for key, value in case.items() if key != "input_reference"}
        return run_reference_mask_stage(**args)

    def test_four_public_calls_execute_through_pinned_runner_and_publish_typed_outputs(self) -> None:
        cases = (
            ("filter.median_blur", {"ksize": 3}),
            ("morphology.fill", {"size": 10}),
            ("morphology.erode", {"ksize": 3, "iterations": 2}),
            ("morphology.dilate", {"ksize": 3, "iterations": 2}),
        )
        for operation, values in cases:
            with self.subTest(operation=operation):
                case = self._case(operation, values)
                outcome = self._run(case)
                self.assertIn(outcome.status, {ReferenceMaskStageStatus.EXECUTED, ReferenceMaskStageStatus.EMPTY})
                self.assertIsNotNone(outcome.checkpoint_reference)
                checkpoint = load_checkpoint(case["output_root"], outcome.run_record.checkpoint_sha256)
                output_name = "image" if operation == "filter.median_blur" else "mask"
                self.assertEqual(set(checkpoint.arrays), {output_name})
                output = checkpoint.arrays[output_name]
                self.assertEqual(output.dtype, np.dtype(np.uint8))
                self.assertEqual(output.shape, self.frame.shape)
                self.assertFalse(checkpoint.measurement.eligible)
                metadata = checkpoint.metadata
                self.assertEqual(metadata["operation_id"], operation)
                self.assertEqual(metadata["parameters"], values)
                self.assertEqual(metadata["effective_parameters"], values | {"callable": {
                    "filter.median_blur": "plantcv.plantcv.median_blur",
                    "morphology.fill": "plantcv.plantcv.fill",
                    "morphology.erode": "plantcv.plantcv.erode",
                    "morphology.dilate": "plantcv.plantcv.dilate",
                }[operation]})
                self.assertIn("observations", metadata)
                self.assertIs(type(metadata["computation_runtime_ns"]), int)
                self.assertEqual(metadata["runner_identity"]["version"], "4.11.3")
                self.assertEqual(metadata["runner_identity"]["commit"], "77e3322fd0cd5a9bef18bb99402f20c8f2db87a9")
                self.assertEqual(len(metadata["runner_identity"]["actual_distributions"]), 67)
                self.assertEqual(metadata["runner_identity"]["actual_distributions"]["plantcv"], "4.11.3")
                self.assertEqual(metadata["scientific_validation_status"], "not_certified")
                self.assertEqual(
                    checkpoint.raster_roles[output_name].kind,
                    "image" if output_name == "image" else "mask",
                )
                if operation == "filter.median_blur":
                    self.assertFalse(np.array_equal(output, _fixture_image()))
                elif operation == "morphology.fill":
                    self.assertEqual(int(output[22, 35]), 0, "PlantCV fill must preserve the hole in a large object.")
                    self.assertEqual(int(output[2:4, 2:4].sum()), 0, "PlantCV fill removes undersized foreground components.")
                elif operation == "morphology.erode":
                    self.assertLess(int(np.count_nonzero(output)), int(np.count_nonzero(_fixture_mask())))
                    self.assertEqual(set(np.unique(output).tolist()) - {0, 255}, set())
                else:
                    self.assertGreater(int(np.count_nonzero(output)), int(np.count_nonzero(_fixture_mask())))
                    self.assertEqual(set(np.unique(output).tolist()) - {0, 255}, set())

    def test_empty_mask_and_gray_zero_image_have_distinct_execution_states(self) -> None:
        with tempfile.TemporaryDirectory(dir=self.root) as temporary:
            case = self._case(
                "morphology.erode", {"ksize": 3, "iterations": 1},
                array=np.zeros(self.frame.shape, dtype=np.uint8),
            )
            outcome = self._run(case)
            self.assertEqual(outcome.status, ReferenceMaskStageStatus.EMPTY)
            self.assertEqual(outcome.run_record.state, StageRunState.EMPTY)
            output = load_checkpoint(case["output_root"], outcome.run_record.checkpoint_sha256)
            self.assertEqual(output.execution.status, ExecutionStatus.EMPTY)

        with tempfile.TemporaryDirectory(dir=self.root) as temporary:
            case = self._case(
                "filter.median_blur", {"ksize": 3},
                array=np.zeros(self.frame.shape, dtype=np.uint8),
            )
            outcome = self._run(case)
            self.assertEqual(outcome.status, ReferenceMaskStageStatus.EXECUTED)
            output = load_checkpoint(case["output_root"], outcome.run_record.checkpoint_sha256)
            self.assertEqual(output.execution.status, ExecutionStatus.AVAILABLE)

    def test_fill_accepts_original_positive_integer_domain_without_numeric_coercion(self) -> None:
        with tempfile.TemporaryDirectory(dir=self.root) as temporary:
            large_size = 10**1000
            case = self._case("morphology.fill", {"size": large_size})
            outcome = self._run(case)
            self.assertEqual(outcome.status, ReferenceMaskStageStatus.EMPTY)
            output = load_checkpoint(case["output_root"], outcome.run_record.checkpoint_sha256)
            recorded = output.metadata["effective_parameters"]["size"]
            self.assertEqual(recorded, large_size)
            self.assertIs(type(recorded), int)

    def test_unsupported_even_tuple_roi_and_over_budget_parameters_do_not_call_runner(self) -> None:
        unsupported = (
            ("filter.median_blur", {"ksize": 4}),
            ("filter.median_blur", {"ksize": (3, 3)}),
            ("morphology.erode", {"ksize": 256, "iterations": 1}),
            ("morphology.dilate", {"ksize": 3, "iterations": 65}),
            ("morphology.erode", {"ksize": 3, "iterations": 1.0}),
            ("morphology.fill", {"size": 0}),
        )
        for operation, values in unsupported:
            with self.subTest(operation=operation, values=values if isinstance(values, dict) else None), tempfile.TemporaryDirectory(dir=self.root) as temporary:
                case = self._case(operation, values)
                with patch.object(case["runner"], "run", side_effect=AssertionError("unsupported domain must not run")):
                    outcome = self._run(case)
                self.assertEqual(outcome.status, ReferenceMaskStageStatus.UNSUPPORTED)
                self.assertIsNone(outcome.checkpoint_reference)

        with tempfile.TemporaryDirectory(dir=self.root) as temporary:
            recipe, stage_id = self._recipe("filter.median_blur", {"ksize": 3})
            stage = recipe.stage(stage_id)
            roi = ParameterContract("roi", "json", {"x": 1}, True)
            changed = replace(stage, parameters=stage.parameters + (roi,))
            recipe = replace(recipe, stages=tuple(changed if item.stage_id == stage_id else item for item in recipe.stages))
            case = self._case("filter.median_blur", recipe=recipe, stage_id=stage_id)
            case["expected_output_identity"] = None
            with patch.object(case["runner"], "run", side_effect=AssertionError("ROI is unsupported")):
                outcome = self._run(case)
            self.assertEqual(outcome.status, ReferenceMaskStageStatus.UNSUPPORTED)
            self.assertIn("ROI", outcome.run_record.reason)

    def test_median_reflect_boundary_rejects_undefined_thin_rasters_and_executes_safe_singleton_axis(self) -> None:
        for ksize in (33, 57, 255):
            with self.subTest(shape=(4, 7), ksize=ksize):
                case = self._case(
                    "filter.median_blur",
                    {"ksize": ksize},
                    array=np.arange(28, dtype=np.uint8).reshape(4, 7),
                )
                with patch.object(case["runner"], "run", side_effect=AssertionError("undefined reflect domain must not execute")):
                    outcome = self._run(case)
                self.assertEqual(outcome.status, ReferenceMaskStageStatus.UNSUPPORTED)
                self.assertIn("source-undefined", outcome.run_record.reason)
                self.assertIsNone(outcome.checkpoint_reference)
                self.assertFalse(Path(case["output_root"]).exists())

        safe_case = self._case(
            "filter.median_blur",
            {"ksize": 255},
            array=np.arange(33, dtype=np.uint8).reshape(1, 33),
        )
        safe_outcome = self._run(safe_case)
        self.assertEqual(safe_outcome.status, ReferenceMaskStageStatus.EXECUTED)
        self.assertIsNotNone(safe_outcome.checkpoint_reference)
        safe_output = load_checkpoint(safe_case["output_root"], safe_outcome.run_record.checkpoint_sha256)
        self.assertEqual(safe_output.arrays["image"].shape, (1, 33))
        self.assertEqual(safe_output.raster_roles["image"].kind, "image")

    def test_large_positive_morphology_kernel_and_iteration_bounds_are_rejected_pre_execution(self) -> None:
        for values in (
            {"ksize": 10**1000, "iterations": 1},
            {"ksize": 3, "iterations": 10**1000},
            {"ksize": True, "iterations": 1},
        ):
            with self.subTest(values=values), tempfile.TemporaryDirectory(dir=self.root) as temporary:
                case = self._case("morphology.erode", values)
                with patch.object(case["runner"], "run", side_effect=AssertionError("unbounded input must not run")):
                    outcome = self._run(case)
                self.assertEqual(outcome.status, ReferenceMaskStageStatus.UNSUPPORTED)

    def test_exact_source_frame_upstream_measurement_binding_and_roles_are_required(self) -> None:
        with tempfile.TemporaryDirectory(dir=self.root) as temporary:
            case = self._case("morphology.erode", {"ksize": 3, "iterations": 1})
            bad_cases = (
                ("source", {**case, "expected_source_sha256": "f" * 64}, "source SHA"),
                ("upstream", {**case, "expected_upstream_checkpoint_sha256": "e" * 64}, "upstream digest"),
                ("frame", {**case, "expected_coordinate_frame": replace(self.frame, frame_id="wrong_frame")}, "frame"),
                ("measurement", {**case, "expected_input_measurement": _measurement("different")}, "measurement"),
                ("role", {**case, "input_roles": {"gray": RasterRole("image", np.dtype(np.uint8).str, "HW", (0, 255))}}, "role"),
                ("recipe_identity", {**case, "expected_recipe_identity": replace(case["recipe"].identity, settings_sha256="f" * 64)}, "recipe identity"),
                ("output_identity", {**case, "expected_output_identity": replace(case["expected_output_identity"], settings_sha256="f" * 64)}, "output identity"),
            )
            for name, bad, message in bad_cases:
                with self.subTest(name=name):
                    with patch.object(case["runner"], "run", side_effect=AssertionError("contract mismatch must not run")):
                        with self.assertRaisesRegex(ReferenceMaskStageError, message):
                            self._run(bad)

            huge_frame = replace(self.frame, shape=(10_000, 10_000))
            huge = {
                **case,
                "expected_coordinate_frame": huge_frame,
                "expected_input_identity": replace(
                    case["expected_input_identity"], coordinate_frame=huge_frame
                ),
            }
            with patch.object(case["runner"], "run", side_effect=AssertionError("oversized frame must not run")):
                with self.assertRaisesRegex(ReferenceMaskStageError, "bounded uint8 raster"):
                    self._run(huge)

            recipe, stage_id = self._recipe("morphology.fill", {"size": 10})
            stage = recipe.stage(stage_id)
            altered_mapping = replace(stage.source_mappings[0], source_sha256="f" * 64)
            altered_stage = replace(stage, source_mappings=(altered_mapping,))
            altered_recipe = replace(
                recipe,
                stages=tuple(altered_stage if item.stage_id == stage_id else item for item in recipe.stages),
            )
            altered = self._case("morphology.fill", recipe=altered_recipe, stage_id=stage_id)
            with self.assertRaisesRegex(ReferenceMaskStageError, "source mapping"):
                self._run(altered)

    def test_invalid_mask_values_and_frame_shape_are_rejected_before_runner(self) -> None:
        import seedvision.benchmarking.reference_mask_stage as module

        with tempfile.TemporaryDirectory(dir=self.root) as temporary:
            case = self._case("morphology.fill", {"size": 5})
            checkpoint = load_checkpoint(case["input_root"], case["input_checkpoint_sha256"])
            malformed = replace(
                checkpoint,
                arrays={"mask_source": np.full(self.frame.shape, 127, dtype=np.uint8)},
            )
            with patch.object(module, "load_stage_input", return_value=malformed):
                with patch.object(case["runner"], "run", side_effect=AssertionError("invalid binary pixels must not run")):
                    with self.assertRaisesRegex(ReferenceMaskStageError, "outside uint8 0/255"):
                        self._run(case)

            wrong_shape = replace(
                checkpoint,
                arrays={"mask_source": np.zeros((47, 72), dtype=np.uint8)},
            )
            with patch.object(module, "load_stage_input", return_value=wrong_shape):
                with patch.object(case["runner"], "run", side_effect=AssertionError("wrong frame must not run")):
                    with self.assertRaisesRegex(ReferenceMaskStageError, "coordinate frame"):
                        self._run(case)

    def test_availability_cancellation_and_original_failure_states_publish_nothing(self) -> None:
        import seedvision.benchmarking.reference_mask_stage as module

        with tempfile.TemporaryDirectory(dir=self.root) as temporary:
            case = self._case("morphology.dilate", {"ksize": 3, "iterations": 1})
            with patch.object(module, "plantcv_reference_status", return_value={"available": False, "reason": "source audit unavailable"}):
                with patch.object(case["runner"], "run", side_effect=AssertionError("unavailable source must not execute")):
                    unavailable = self._run(case)
            self.assertEqual(unavailable.status, ReferenceMaskStageStatus.UNAVAILABLE)
            self.assertIsNone(unavailable.checkpoint_reference)

            cancel = threading.Event()
            cancel.set()
            cancelled = self._run({**case, "cancel": cancel})
            self.assertEqual(cancelled.status, ReferenceMaskStageStatus.CANCELLED)
            self.assertIsNone(cancelled.checkpoint_reference)

            with patch.object(case["runner"], "run", side_effect=ReferenceExecutionError("simulated original-call failure")):
                failed = self._run(case)
            self.assertEqual(failed.status, ReferenceMaskStageStatus.FAILED)
            self.assertIsNone(failed.checkpoint_reference)
            self.assertFalse(Path(case["output_root"]).exists())

    def test_malformed_original_result_fails_without_publishing_artifacts(self) -> None:
        with tempfile.TemporaryDirectory(dir=self.root) as temporary:
            case = self._case("morphology.erode", {"ksize": 3, "iterations": 1})
            real_run = case["runner"].run

            def non_binary(*args, **kwargs):
                result = real_run(*args, **kwargs)
                return replace(result, arrays={"mask": np.full(self.frame.shape, 12, dtype=np.uint8)})

            with patch.object(case["runner"], "run", side_effect=non_binary):
                outcome = self._run(case)
            self.assertEqual(outcome.status, ReferenceMaskStageStatus.FAILED)
            self.assertIn("0/255", outcome.run_record.reason)
            self.assertIsNone(outcome.checkpoint_reference)
            self.assertFalse(Path(case["output_root"]).exists())

        with tempfile.TemporaryDirectory(dir=self.root) as temporary:
            case = self._case("morphology.dilate", {"ksize": 3, "iterations": 1})
            real_run = case["runner"].run

            def wrong_environment(*args, **kwargs):
                result = real_run(*args, **kwargs)
                identity = dict(result.identity)
                identity["actual_distributions"] = dict(identity["actual_distributions"])
                identity["actual_distributions"]["plantcv"] = "0.0.0"
                return replace(result, identity=identity)

            with patch.object(case["runner"], "run", side_effect=wrong_environment):
                outcome = self._run(case)
            self.assertEqual(outcome.status, ReferenceMaskStageStatus.FAILED)
            self.assertIn("dependency inventory", outcome.run_record.reason)
            self.assertIsNone(outcome.checkpoint_reference)
            self.assertFalse(Path(case["output_root"]).exists())

    def test_internal_watershed_substages_are_explicitly_unsupported(self) -> None:
        recipe = self.catalogue.recipe("plantcv_seed_binary")
        stage = next(item for item in recipe.stages if item.operation_id == "reference.internal.watershed.local_maxima")
        with tempfile.TemporaryDirectory(dir=self.root) as temporary:
            case = self._case(
                "morphology.fill",
                {"size": 1},
                recipe=recipe,
                stage_id=stage.stage_id,
            )
            case["expected_output_identity"] = None
            with patch.object(case["runner"], "run", side_effect=AssertionError("internal watershed is not replayed")):
                outcome = self._run(case)
            self.assertEqual(outcome.status, ReferenceMaskStageStatus.UNSUPPORTED)
            self.assertIn("watershed internals", outcome.run_record.reason)


if __name__ == "__main__":
    unittest.main()
