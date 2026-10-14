from __future__ import annotations

from contextlib import ExitStack
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
from seedvision.benchmarking.native_mask_stage import (
    NativeMaskIdentity,
    NativeMaskStageOutcome,
    NativeMaskStageStatus,
    native_mask_identity,
)
from seedvision.benchmarking.native_preprocessing import (
    NativePreprocessingStageStatus,
    NativePreprocessingStatus,
    run_native_preprocessing,
)
from seedvision.benchmarking.native_threshold_stage import (
    NativeThresholdIdentity,
    NativeThresholdOutcome,
    native_threshold_identity,
)
from seedvision.benchmarking.recipe_catalogue import build_recipe_catalogue
from seedvision.benchmarking.recipes import RecipeSpec, StageRunRecord, StageRunState


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
_IMAGE_ROLE = RasterRole("image", np.dtype(np.uint8).str, "HW", (0, 255))
_MASK_ROLE = RasterRole("mask", np.dtype(np.uint8).str, "HW")


def _measurement(name: str) -> MeasurementContract:
    return MeasurementContract(
        definition_id=name,
        definition_version="1",
        mask_type="native_preprocessing_raster",
        units="pixel",
        calibration_status="not_applicable",
        uncertainty="not_evaluated",
        eligible=False,
        reason="Preprocessing output is not a validated biological measurement.",
    )


def _selected_recipe(recipe_id: str = "plantcv_seed_binary", *, median_ksize: int = 3,
                    binary_cutoff: int = 128, fill_size: int = 5, mean_ksize: int = 5) -> RecipeSpec:
    recipe = build_recipe_catalogue().recipe(recipe_id)
    selected = {
        "filter.median_blur": {"ksize": median_ksize},
        "threshold.binary": {"threshold": binary_cutoff, "object_type": "light"},
        "threshold.mean": {"ksize": mean_ksize, "offset": 0.0, "object_type": "light"},
        "threshold.otsu": {"object_type": "light"},
        "morphology.fill": {"size": fill_size},
        "morphology.erode": {"ksize": 3, "iterations": 1},
        "morphology.dilate": {"ksize": 3, "iterations": 1},
    }
    stages = []
    for stage in recipe.stages:
        values = selected.get(stage.operation_id)
        if values is None:
            stages.append(stage)
            continue
        parameters = tuple(
            replace(parameter, value=values[parameter.parameter_id])
            if parameter.parameter_id in values else parameter
            for parameter in stage.parameters
        )
        stages.append(replace(stage, parameters=parameters))
    return replace(recipe, stages=tuple(stages))


def _mask_identity() -> NativeMaskIdentity:
    return NativeMaskIdentity(
        implementation_sha256=sha256(b"test native mask implementation").hexdigest(),
        python_version="3.12.10",
        torch_version="2.13.0+cu128",
        numpy_version=np.__version__,
        cuda_runtime_version="12.8",
        platform="win32",
        machine="AMD64",
        device_index=0,
        device_name="Test CUDA device",
        compute_capability="8.6",
        reference_source_sha256=sha256(b"test pinned PlantCV mask source").hexdigest(),
    )


def _threshold_identity() -> NativeThresholdIdentity:
    return NativeThresholdIdentity(
        implementation_sha256=sha256(b"test native threshold implementation").hexdigest(),
        python_version="3.12.10",
        torch_version="2.13.0+cu128",
        numpy_version=np.__version__,
        cuda_runtime_version="12.8",
        platform="win32",
        machine="AMD64",
        device_index=0,
        device_name="Test CUDA device",
        compute_capability="8.6",
    )


def _gray_fixture(shape: tuple[int, int], *, empty: bool = False, two_level: bool = False) -> np.ndarray:
    gray = np.zeros(shape, dtype=np.uint8)
    if empty:
        return gray
    gray.fill(32)
    height, width = shape
    gray[max(2, height // 5):max(3, 4 * height // 5), max(2, width // 5):max(3, 4 * width // 5)] = 205
    if not two_level:
        gray[height // 2, width // 2] = 255
    return np.ascontiguousarray(gray)


class _NativePreprocessingFixtures:
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="native-preprocessing-contract-")
        self.root = Path(self.temporary.name)
        self.mask_identity = _mask_identity()
        self.threshold_identity = _threshold_identity()
        self.counter = 0

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _arguments(self, recipe: RecipeSpec | None = None, *, shape: tuple[int, int] = (48, 72),
                   empty: bool = False, two_level: bool = False) -> dict[str, object]:
        self.counter += 1
        recipe = _selected_recipe() if recipe is None else recipe
        case = self.root / f"case-{self.counter}"
        source_sha256 = sha256(f"native-preprocessing-source-{self.counter}".encode()).hexdigest()
        frame = CoordinateFrame(
            frame_id="native_preprocessing_gray_v1",
            shape=shape,
            origin="top_left_pixel_centres",
            axes="x_right_y_down",
            units="pixel",
            transform_sha256=sha256(b"native preprocessing fixture frame").hexdigest(),
        )
        identity = CheckpointIdentity(
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
                "color_order": "GRAY", "dtype": "uint8", "value_range": [0, 255],
                "channel_conversion": "caller_supplied_gray",
                "geometric_transform": "fixture_identity_frame", "interpolation": "not_applicable",
            },
            mask_semantics={
                "foreground_polarity": "light", "representation": "uint8_0_255",
                "connectivity": "not_applicable", "validity_region": "full_frame",
                "border_policy": "not_applicable",
            },
            instance_semantics={
                "label_dtype": "int32", "background_label": 0,
                "label_id_mapping": "not_created", "split_boundary_ownership": "not_applicable",
                "visibility": "not_applicable", "eligibility": "not_evaluated",
            },
            upstream_checkpoint_sha256="d" * 64,
        )
        input_measurement = _measurement("prepared_gray")
        reference = publish_checkpoint(
            case / "input", identity, {"gray": _gray_fixture(shape, empty=empty, two_level=two_level)},
            execution=ExecutionRecord(ExecutionStatus.AVAILABLE), measurement=input_measurement,
            raster_roles={"gray": _IMAGE_ROLE}, metadata={"fixture_only": True}, allow_cpu_transfer=True,
        )
        return {
            "input_root": case / "input",
            "input_checkpoint_sha256": reference.checkpoint_sha256,
            "recipe": recipe,
            "output_root": case / "native-output",
            "expected_source_sha256": source_sha256,
            "expected_coordinate_frame": frame,
            "expected_recipe_identity": recipe.identity,
            "expected_input_identity": identity,
            "expected_input_measurement": input_measurement,
            "output_measurement": _measurement("native_preprocessing_stage"),
            "expected_native_mask_identity": self.mask_identity,
            "expected_native_threshold_identity": self.threshold_identity,
            "device": __import__("torch").device("cuda:0"),
            "input_bindings": {"gray": "gray"},
            "input_roles": {"gray": _IMAGE_ROLE},
            "allow_cpu_transfer": True,
        }

    def _patched_identities(self):
        stack = ExitStack()
        stack.enter_context(patch(
            "seedvision.benchmarking.native_preprocessing.native_mask_identity",
            return_value=self.mask_identity,
        ))
        stack.enter_context(patch(
            "seedvision.benchmarking.native_preprocessing.native_threshold_identity",
            return_value=self.threshold_identity,
        ))
        return stack

    def _fake_stage(self, *, state: StageRunState = StageRunState.EXECUTED,
                    only_operation: str | None = None, only_status: object | None = None,
                    state_by_operation: dict[str, StageRunState] | None = None,
                    failure_reason: str = "synthetic unsupported stage",
                    calls: list[dict[str, object]] | None = None):
        def bridge(**kwargs: object):
            recipe = kwargs["recipe"]
            stage = recipe.stage(kwargs["stage_id"])
            if calls is not None:
                calls.append(dict(kwargs))
            if only_operation is not None and stage.operation_id == only_operation and only_status is not None:
                local_state = {
                    "executed": StageRunState.EXECUTED,
                    "empty": StageRunState.EMPTY,
                    "failed": StageRunState.FAILED,
                    "cancelled": StageRunState.FAILED,
                    "unavailable": StageRunState.UNAVAILABLE,
                    "unsupported": StageRunState.UNSUPPORTED,
                    "unresolved": StageRunState.UNRESOLVED,
                }[str(only_status)]
            else:
                local_state = (state_by_operation or {}).get(
                    stage.operation_id,
                    state if only_operation is None or stage.operation_id != only_operation else StageRunState.FAILED,
                )
            digest = sha256(f"{stage.stage_id}:{kwargs['expected_upstream_checkpoint_sha256']}".encode()).hexdigest()
            if local_state in {StageRunState.EXECUTED, StageRunState.EMPTY}:
                record = StageRunRecord(
                    stage_id=stage.stage_id, recipe_identity=recipe.identity, state=local_state,
                    checkpoint_sha256=digest, upstream_checkpoint_sha256=kwargs["expected_upstream_checkpoint_sha256"],
                )
                execution = ExecutionRecord(
                    ExecutionStatus.EMPTY if local_state is StageRunState.EMPTY else ExecutionStatus.AVAILABLE,
                )
                reference = CheckpointReference(
                    digest, Path(f"synthetic-{stage.stage_id}"), kwargs["expected_output_identity"], execution,
                )
                if stage.operation_id.startswith("threshold."):
                    return NativeThresholdOutcome(record, reference)
                status = NativeMaskStageStatus.EMPTY if local_state is StageRunState.EMPTY else NativeMaskStageStatus.EXECUTED
                return NativeMaskStageOutcome(status, record, reference)
            record = StageRunRecord(
                stage_id=stage.stage_id, recipe_identity=recipe.identity, state=local_state,
                upstream_checkpoint_sha256=kwargs["expected_upstream_checkpoint_sha256"],
                reason=failure_reason,
            )
            if stage.operation_id.startswith("threshold."):
                return NativeThresholdOutcome(record, None)
            return NativeMaskStageOutcome(only_status or NativeMaskStageStatus.FAILED, record, None)

        return bridge


class NativePreprocessingContractTests(_NativePreprocessingFixtures, unittest.TestCase):
    def test_layout_is_display_only_and_stage_projections_are_exact(self) -> None:
        recipe = _selected_recipe()
        moved = replace(recipe.layout[0], x=recipe.layout[0].x + 90.0, y=recipe.layout[0].y + 20.0)
        selected = replace(recipe, layout=(moved, *recipe.layout[1:]))
        arguments = self._arguments(selected)
        calls: list[dict[str, object]] = []
        fake_mask = self._fake_stage(calls=calls)
        fake_threshold = self._fake_stage(calls=calls)
        with self._patched_identities(), patch(
            "seedvision.benchmarking.native_preprocessing.run_native_mask_stage", side_effect=fake_mask,
        ), patch(
            "seedvision.benchmarking.native_preprocessing.run_native_threshold_stage", side_effect=fake_threshold,
        ):
            outcome = run_native_preprocessing(**arguments)
        self.assertEqual(outcome.status, NativePreprocessingStatus.EXECUTED)
        self.assertEqual(outcome.recipe_identity, selected.identity)
        self.assertNotEqual(outcome.stages[0].run_record.recipe_identity, selected.identity)
        self.assertEqual(len(outcome.run_records), 5)
        self.assertIsNotNone(outcome.final_checkpoint)
        self.assertEqual(len(calls), 5)
        self.assertEqual(
            [call["recipe"].stage(call["stage_id"]).operation_id for call in calls],
            ["filter.median_blur", "threshold.binary", "morphology.fill", "morphology.erode", "morphology.dilate"],
        )
        self.assertEqual(calls[0]["input_bindings"], {"gray": "gray"})
        self.assertEqual(calls[1]["input_bindings"], {"gray": "image"})
        self.assertEqual(calls[2]["input_bindings"], {"mask": "mask"})
        self.assertEqual(calls[3]["input_bindings"], {"gray": "mask"})
        self.assertEqual(calls[4]["input_bindings"], {"gray": "mask"})
        self.assertEqual(calls[0]["input_checkpoint_sha256"], arguments["input_checkpoint_sha256"])
        for index in range(1, len(calls)):
            previous = outcome.stages[index - 1].checkpoint_reference
            self.assertEqual(calls[index]["input_checkpoint_sha256"], previous.checkpoint_sha256)
            self.assertEqual(calls[index]["expected_input_identity"], previous.identity)
            self.assertEqual(calls[index]["expected_upstream_checkpoint_sha256"], previous.checkpoint_sha256)
        self.assertEqual(calls[1]["recipe"].parameters_for_runner(calls[1]["stage_id"]), {
            "threshold": 128, "object_type": "light",
        })
        self.assertTrue(all(stage.run_record.recipe_identity.layout_sha256 ==
                            build_recipe_catalogue().recipe("plantcv_seed_binary").identity.layout_sha256
                            for stage in outcome.stages))

    def test_empty_masks_are_valid_and_chain_continues_through_all_stages(self) -> None:
        arguments = self._arguments(empty=True)
        calls: list[dict[str, object]] = []
        fake = self._fake_stage(
            state_by_operation={
                "threshold.binary": StageRunState.EMPTY,
                "morphology.fill": StageRunState.EMPTY,
                "morphology.erode": StageRunState.EMPTY,
                "morphology.dilate": StageRunState.EMPTY,
            },
            calls=calls,
        )
        with self._patched_identities(), patch(
            "seedvision.benchmarking.native_preprocessing.run_native_mask_stage", side_effect=fake,
        ), patch(
            "seedvision.benchmarking.native_preprocessing.run_native_threshold_stage", side_effect=fake,
        ):
            outcome = run_native_preprocessing(**arguments)
        self.assertEqual(outcome.status, NativePreprocessingStatus.EMPTY)
        self.assertEqual(outcome.stages[0].status, NativePreprocessingStageStatus.EXECUTED)
        self.assertTrue(all(item.status is NativePreprocessingStageStatus.EMPTY for item in outcome.stages[1:]))
        self.assertEqual(len(calls), 5)
        self.assertIsNotNone(outcome.final_checkpoint)

    def test_mean_and_otsu_use_their_exact_threshold_contracts_and_mask_bridge_projections(self) -> None:
        for recipe_id, operation_id in (
            ("plantcv_seed_mean", "threshold.mean"),
            ("plantcv_seed_otsu", "threshold.otsu"),
        ):
            with self.subTest(recipe=recipe_id):
                recipe = _selected_recipe(recipe_id)
                arguments = self._arguments(recipe)
                calls: list[dict[str, object]] = []
                fake = self._fake_stage(calls=calls)
                with self._patched_identities(), patch(
                    "seedvision.benchmarking.native_preprocessing.run_native_mask_stage", side_effect=fake,
                ), patch(
                    "seedvision.benchmarking.native_preprocessing.run_native_threshold_stage", side_effect=fake,
                ):
                    outcome = run_native_preprocessing(**arguments)
                self.assertEqual(outcome.status, NativePreprocessingStatus.EXECUTED)
                self.assertEqual(outcome.recipe_identity, recipe.identity)
                self.assertEqual(outcome.stages[0].stage.operation_id, "filter.median_blur")
                self.assertNotEqual(outcome.stages[0].stage.stage_id, outcome.stages[0].selected_stage_id)
                self.assertEqual(outcome.stages[1].stage.operation_id, operation_id)
                self.assertEqual(outcome.stages[1].stage.stage_id, outcome.stages[1].selected_stage_id)
                self.assertEqual(outcome.stages[2].stage.operation_id, "morphology.fill")
                self.assertEqual(outcome.stages[2].stage.stage_id.split("_")[2], "binary")
                self.assertEqual(outcome.stages[1].run_record.recipe_identity.recipe_id, recipe_id)
                self.assertTrue(all(
                    stage.run_record.recipe_identity.recipe_id == "plantcv_seed_binary"
                    for index, stage in enumerate(outcome.stages) if index != 1
                ))
                self.assertEqual(len(calls), 5)

    def test_unsupported_selected_stage_is_found_before_any_native_bridge_runs(self) -> None:
        recipe = _selected_recipe("plantcv_seed_mean", mean_ksize=256)
        arguments = self._arguments(recipe)
        with self._patched_identities(), patch(
            "seedvision.benchmarking.native_preprocessing.run_native_mask_stage",
        ) as mask_bridge, patch(
            "seedvision.benchmarking.native_preprocessing.run_native_threshold_stage",
        ) as threshold_bridge:
            outcome = run_native_preprocessing(**arguments)
        self.assertEqual(outcome.status, NativePreprocessingStatus.UNSUPPORTED)
        self.assertIn('effective odd ksize <= 255', outcome.stages[1].run_record.reason)
        self.assertEqual(
            outcome.stages[1].status,
            NativePreprocessingStageStatus.UNSUPPORTED,
            [(item.status.value, item.run_record.reason) for item in outcome.stages],
        )
        self.assertTrue(all(stage.status is NativePreprocessingStageStatus.NOT_RUN
                            for index, stage in enumerate(outcome.stages) if index != 1))
        self.assertEqual(len(outcome.run_records), 5)
        self.assertIsNone(outcome.final_checkpoint)
        mask_bridge.assert_not_called()
        threshold_bridge.assert_not_called()

    def test_precancelled_chain_records_cancellation_without_calling_a_bridge(self) -> None:
        arguments = self._arguments()
        cancel = threading.Event()
        cancel.set()
        arguments["cancel"] = cancel
        with self._patched_identities(), patch(
            "seedvision.benchmarking.native_preprocessing.run_native_mask_stage",
        ) as mask_bridge, patch(
            "seedvision.benchmarking.native_preprocessing.run_native_threshold_stage",
        ) as threshold_bridge:
            outcome = run_native_preprocessing(**arguments)
        self.assertEqual(outcome.status, NativePreprocessingStatus.CANCELLED)
        self.assertEqual(outcome.stages[0].status, NativePreprocessingStageStatus.CANCELLED)
        self.assertTrue(all(stage.status is NativePreprocessingStageStatus.NOT_RUN for stage in outcome.stages[1:]))
        self.assertIsNone(outcome.final_checkpoint)
        mask_bridge.assert_not_called()
        threshold_bridge.assert_not_called()

    def test_failure_text_does_not_turn_an_unrelated_threshold_failure_into_cancellation(self) -> None:
        arguments = self._arguments()
        calls: list[dict[str, object]] = []
        mask_bridge = self._fake_stage(calls=calls)
        threshold_bridge = self._fake_stage(
            only_operation="threshold.binary",
            only_status=NativeMaskStageStatus.FAILED,
            failure_reason="An unrelated processing failure mentions cancelled input data.",
            calls=calls,
        )
        with self._patched_identities(), patch(
            "seedvision.benchmarking.native_preprocessing.run_native_mask_stage", side_effect=mask_bridge,
        ), patch(
            "seedvision.benchmarking.native_preprocessing.run_native_threshold_stage", side_effect=threshold_bridge,
        ):
            outcome = run_native_preprocessing(**arguments)
        self.assertEqual(outcome.status, NativePreprocessingStatus.FAILED)
        self.assertEqual(outcome.stages[1].status, NativePreprocessingStageStatus.FAILED)
        self.assertTrue(all(stage.status is NativePreprocessingStageStatus.NOT_RUN for stage in outcome.stages[2:]))
        self.assertIsNone(outcome.final_checkpoint)

    def test_failed_threshold_with_cancellation_event_set_during_call_is_cancelled(self) -> None:
        arguments = self._arguments()
        cancel = threading.Event()
        arguments["cancel"] = cancel
        mask_bridge = self._fake_stage()

        def cancelled_threshold(**kwargs: object):
            cancel.set()
            recipe = kwargs["recipe"]
            stage = recipe.stage(kwargs["stage_id"])
            record = StageRunRecord(
                stage_id=stage.stage_id,
                recipe_identity=recipe.identity,
                state=StageRunState.FAILED,
                upstream_checkpoint_sha256=kwargs["expected_upstream_checkpoint_sha256"],
                reason="Threshold primitive stopped.",
            )
            return NativeThresholdOutcome(record, None)

        with self._patched_identities(), patch(
            "seedvision.benchmarking.native_preprocessing.run_native_mask_stage", side_effect=mask_bridge,
        ), patch(
            "seedvision.benchmarking.native_preprocessing.run_native_threshold_stage",
            side_effect=cancelled_threshold,
        ):
            outcome = run_native_preprocessing(**arguments)
        self.assertEqual(outcome.status, NativePreprocessingStatus.CANCELLED)
        self.assertEqual(outcome.stages[1].status, NativePreprocessingStageStatus.CANCELLED)
        self.assertTrue(all(stage.status is NativePreprocessingStageStatus.NOT_RUN for stage in outcome.stages[2:]))

    def test_unavailable_stage_stops_chain_and_never_exposes_intermediate_as_final(self) -> None:
        arguments = self._arguments()
        mask_calls: list[dict[str, object]] = []
        threshold_calls: list[dict[str, object]] = []
        mask_bridge = self._fake_stage(only_operation="morphology.fill", only_status=NativeMaskStageStatus.UNAVAILABLE,
                                       calls=mask_calls)
        threshold_bridge = self._fake_stage(calls=threshold_calls)
        with self._patched_identities(), patch(
            "seedvision.benchmarking.native_preprocessing.run_native_mask_stage", side_effect=mask_bridge,
        ), patch(
            "seedvision.benchmarking.native_preprocessing.run_native_threshold_stage", side_effect=threshold_bridge,
        ):
            outcome = run_native_preprocessing(**arguments)
        self.assertEqual(outcome.status, NativePreprocessingStatus.UNAVAILABLE)
        self.assertEqual(outcome.stages[2].status, NativePreprocessingStageStatus.UNAVAILABLE)
        self.assertEqual([item.status for item in outcome.stages[3:]], [
            NativePreprocessingStageStatus.NOT_RUN, NativePreprocessingStageStatus.NOT_RUN,
        ])
        self.assertIsNone(outcome.final_checkpoint)
        self.assertIsNotNone(outcome.latest_checkpoint)
        self.assertEqual(len(mask_calls), 2)
        self.assertEqual(len(threshold_calls), 1)

    def test_runtime_identity_mismatch_stops_before_the_first_stage(self) -> None:
        arguments = self._arguments()
        changed = replace(self.mask_identity, implementation_sha256=sha256(b"different implementation").hexdigest())
        with patch(
            "seedvision.benchmarking.native_preprocessing.native_mask_identity", return_value=changed,
        ), patch(
            "seedvision.benchmarking.native_preprocessing.native_threshold_identity",
            return_value=self.threshold_identity,
        ), patch(
            "seedvision.benchmarking.native_preprocessing.run_native_mask_stage",
        ) as mask_bridge:
            outcome = run_native_preprocessing(**arguments)
        self.assertEqual(outcome.status, NativePreprocessingStatus.FAILED)
        self.assertEqual(outcome.stages[0].status, NativePreprocessingStageStatus.FAILED)
        self.assertTrue(all(stage.status is NativePreprocessingStageStatus.NOT_RUN for stage in outcome.stages[1:]))
        mask_bridge.assert_not_called()

    def test_unavailable_native_environment_is_recorded_before_computation(self) -> None:
        from seedvision.benchmarking.native_mask_stage import NativeMaskUnavailableError

        arguments = self._arguments()
        with patch(
            "seedvision.benchmarking.native_preprocessing.native_mask_identity",
            side_effect=NativeMaskUnavailableError("device unavailable"),
        ), patch(
            "seedvision.benchmarking.native_preprocessing.native_threshold_identity",
            return_value=self.threshold_identity,
        ), patch(
            "seedvision.benchmarking.native_preprocessing.run_native_mask_stage",
        ) as mask_bridge:
            outcome = run_native_preprocessing(**arguments)
        self.assertEqual(outcome.status, NativePreprocessingStatus.UNAVAILABLE)
        self.assertEqual(outcome.stages[0].status, NativePreprocessingStageStatus.UNAVAILABLE)
        self.assertTrue(all(stage.status is NativePreprocessingStageStatus.NOT_RUN for stage in outcome.stages[1:]))
        mask_bridge.assert_not_called()

    def test_stage_checkpoint_identity_mismatch_fails_and_stops_downstream(self) -> None:
        arguments = self._arguments()
        calls: list[dict[str, object]] = []
        generic = self._fake_stage(calls=calls)

        def corrupt_fill(**kwargs: object):
            outcome = generic(**kwargs)
            if kwargs["recipe"].stage(kwargs["stage_id"]).operation_id != "morphology.fill":
                return outcome
            reference = outcome.checkpoint_reference
            bad_identity = replace(reference.identity, backend_version="unexpected")
            bad_reference = CheckpointReference(
                reference.checkpoint_sha256, reference.path, bad_identity, reference.execution,
            )
            return NativeMaskStageOutcome(outcome.status, outcome.run_record, bad_reference)

        with self._patched_identities(), patch(
            "seedvision.benchmarking.native_preprocessing.run_native_mask_stage", side_effect=corrupt_fill,
        ), patch(
            "seedvision.benchmarking.native_preprocessing.run_native_threshold_stage", side_effect=generic,
        ):
            outcome = run_native_preprocessing(**arguments)
        self.assertEqual(outcome.status, NativePreprocessingStatus.FAILED)
        self.assertEqual(outcome.stages[2].status, NativePreprocessingStageStatus.FAILED)
        self.assertTrue(all(stage.status is NativePreprocessingStageStatus.NOT_RUN for stage in outcome.stages[3:]))
        self.assertIsNone(outcome.stages[2].checkpoint_reference)
        self.assertIsNone(outcome.final_checkpoint)
        self.assertEqual(len(calls), 3)

    def test_preflight_binds_exact_source_frame_and_input_array_role(self) -> None:
        arguments = self._arguments()
        arguments["input_bindings"] = {"gray": "missing"}
        with self.assertRaisesRegex(ValueError, "grayscale binding"):
            run_native_preprocessing(**arguments)
        wrong_source = self._arguments()
        wrong_source["expected_source_sha256"] = sha256(b"another source").hexdigest()
        with self.assertRaisesRegex(ValueError, "different source digest"):
            run_native_preprocessing(**wrong_source)
        wrong_frame = self._arguments()
        wrong_frame["expected_coordinate_frame"] = replace(
            wrong_frame["expected_coordinate_frame"], frame_id="another-frame",
        )
        with self.assertRaisesRegex(ValueError, "different coordinate frame"):
            run_native_preprocessing(**wrong_frame)


@unittest.skipUnless(HAS_REFERENCE_ENV, "the ignored pinned PlantCV reference environment is unavailable")
class NativePreprocessingPlantCVParityTests(_NativePreprocessingFixtures, unittest.TestCase):
    def test_native_binary_mean_and_restricted_otsu_chains_match_original_stage_pixels(self) -> None:
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available for native preprocessing conformance.")
        from seedvision.benchmarking.reference_preprocessing import run_reference_preprocessing
        from seedvision.benchmarking.reference_runner import PlantCVReferenceRunner

        native_identities = {
            "expected_native_mask_identity": native_mask_identity(torch.device("cuda:0")),
            "expected_native_threshold_identity": native_threshold_identity(torch.device("cuda:0")),
        }
        for recipe_id in ("plantcv_seed_binary", "plantcv_seed_mean", "plantcv_seed_otsu"):
            with self.subTest(recipe=recipe_id):
                recipe = _selected_recipe(recipe_id)
                arguments = self._arguments(
                    recipe,
                    shape=(48, 72),
                    two_level=recipe_id == "plantcv_seed_otsu",
                )
                arguments.update(native_identities)
                arguments["device"] = torch.device("cuda:0")
                native = run_native_preprocessing(**arguments)
                self.assertIn(native.status, {NativePreprocessingStatus.EXECUTED, NativePreprocessingStatus.EMPTY})
                self.assertIsNotNone(native.final_checkpoint)

                reference_arguments = {
                    key: value for key, value in arguments.items()
                    if key not in {"expected_native_mask_identity", "expected_native_threshold_identity", "device"}
                }
                reference_arguments["runner"] = PlantCVReferenceRunner(REFERENCE_PYTHON)
                reference_arguments["output_root"] = Path(arguments["output_root"]).parent / "reference-output"
                original = run_reference_preprocessing(**reference_arguments)
                self.assertIn(original.status, {"executed", "empty"})
                self.assertEqual(len(original.stages), len(native.stages))
                self.assertEqual(native.stages[1].stage.operation_id, {
                    "plantcv_seed_binary": "threshold.binary",
                    "plantcv_seed_mean": "threshold.mean",
                    "plantcv_seed_otsu": "threshold.otsu",
                }[recipe_id])
                for native_stage, original_stage in zip(native.stages, original.stages, strict=True):
                    self.assertIn(native_stage.status, {NativePreprocessingStageStatus.EXECUTED,
                                                        NativePreprocessingStageStatus.EMPTY})
                    self.assertIn(original_stage.status.value, {"executed", "empty"})
                    native_checkpoint = load_checkpoint(
                        arguments["output_root"], native_stage.checkpoint_reference.checkpoint_sha256,
                    )
                    original_checkpoint = load_checkpoint(
                        reference_arguments["output_root"], original_stage.checkpoint_reference.checkpoint_sha256,
                    )
                    output_name = "image" if native_stage.stage.operation_id == "filter.median_blur" else "mask"
                    np.testing.assert_array_equal(
                        native_checkpoint.arrays[output_name], original_checkpoint.arrays[output_name],
                    )


if __name__ == "__main__":
    unittest.main()
