from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np
import torch

import seedvision.benchmarking.checkpoints as checkpoint_module
import seedvision.benchmarking.threshold_conformance as pair_module
import seedvision.benchmarking.native_threshold_stage as native_stage_module
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
from seedvision.benchmarking.conformance import ThresholdBorderCorrespondence
from seedvision.benchmarking.native_threshold_stage import (
    NativeThresholdIdentity,
    expected_native_threshold_stage_identity,
)
from seedvision.benchmarking.recipe_catalogue import build_recipe_catalogue
from seedvision.benchmarking.recipes import (
    RecipeProvenance,
    RecipeProvenanceMode,
    SourceResolution,
    StageRunState,
)
from seedvision.benchmarking.reference_runner import PlantCVReferenceRunner
from seedvision.benchmarking.reference_stage import expected_reference_stage_identity
from seedvision.benchmarking.threshold_conformance import (
    ThresholdPairError,
    run_threshold_conformance_pair,
)


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PYTHON = ROOT / "artifacts" / "plantcv-reference-env" / "Scripts" / "python.exe"
REFERENCE_SOURCE_ROOT = ROOT / "artifacts" / "reference-sources" / "plantcv-v4.11.3"
REFERENCE_SOURCE_MANIFEST = ROOT / "artifacts" / "reference-sources" / "plantcv-source-audit.json"
REFERENCE_ENVIRONMENT_MANIFEST = ROOT / "artifacts" / "reference-sources" / "plantcv-environment.json"
REFERENCE_REQUIREMENTS = ROOT / "config" / "plantcv-reference-requirements.txt"
REFERENCE_ENVIRONMENT_READY = all(path.exists() for path in (
    REFERENCE_PYTHON,
    REFERENCE_SOURCE_ROOT,
    REFERENCE_SOURCE_MANIFEST,
    REFERENCE_ENVIRONMENT_MANIFEST,
    REFERENCE_REQUIREMENTS,
))


def _digest(value: bytes | str) -> str:
    if isinstance(value, str):
        value = value.encode("utf-8")
    return sha256(value).hexdigest()


def _measurement(name: str) -> MeasurementContract:
    return MeasurementContract(
        definition_id=name,
        definition_version="1",
        mask_type="binary_threshold_output",
        units="pixel",
        calibration_status="not_applicable",
        uncertainty="not_evaluated",
        eligible=False,
        reason="This stage comparison is conformance evidence, not scientific validation.",
    )


def _implementation(device_index: int) -> NativeThresholdIdentity:
    if torch.cuda.is_available() and device_index < torch.cuda.device_count():
        return native_stage_module.native_threshold_identity(torch.device("cuda", device_index))
    return NativeThresholdIdentity(
        implementation_sha256="a" * 64,
        python_version="3.12.10",
        torch_version=torch.__version__,
        numpy_version=np.__version__,
        cuda_runtime_version="unavailable",
        platform="test-platform",
        machine="unavailable-test-device",
        device_index=device_index,
        device_name="unavailable-test-device",
        compute_capability="8.6",
    )


def _selected(operation_id: str) -> dict[str, object]:
    return {
        "threshold.binary": {"threshold": 127, "object_type": "light"},
        "threshold.mean": {"ksize": 3, "offset": 1.25, "object_type": "light"},
        "threshold.otsu": {"object_type": "light"},
    }[operation_id]


def _image(operation_id: str, *, ambiguous_otsu: bool = False, empty: bool = False) -> np.ndarray:
    if ambiguous_otsu:
        return np.array([[0, 0, 1, 2, 2]], dtype=np.uint8)
    if empty:
        return np.full((13, 19), 255, dtype=np.uint8)
    if operation_id == "threshold.otsu":
        image = np.full((24, 32), 16, dtype=np.uint8)
        image[:, 15:] = 220
        image[5:17, 5:13] = 44
        return image
    rng = np.random.default_rng(0xF221)
    return rng.integers(0, 256, size=(24, 32), dtype=np.uint8)


def _fixture(
    temporary: str,
    operation_id: str = "threshold.binary",
    *,
    selected: dict[str, object] | None = None,
    values: np.ndarray | None = None,
    implementation: NativeThresholdIdentity | None = None,
) -> dict[str, object]:
    if selected is None:
        selected = _selected(operation_id)
    if values is None:
        values = _image(operation_id)
    device_index = torch.cuda.current_device() if torch.cuda.is_available() else 0
    device = torch.device("cuda", device_index)
    if implementation is None:
        implementation = _implementation(device_index)
    recipe_id = {
        "threshold.binary": "plantcv_seed_binary",
        "threshold.mean": "plantcv_seed_mean",
        "threshold.otsu": "plantcv_seed_otsu",
    }[operation_id]
    reference_recipe = build_recipe_catalogue().recipe(recipe_id)
    reference_stage = next(stage for stage in reference_recipe.stages if stage.operation_id == operation_id)
    selected_stage = replace(
        reference_stage,
        parameters=tuple(
            replace(parameter, value=selected[parameter.parameter_id])
            if parameter.parameter_id in selected else parameter
            for parameter in reference_stage.parameters
        ),
    )
    reference_recipe = replace(
        reference_recipe,
        stages=tuple(selected_stage if stage.stage_id == reference_stage.stage_id else stage for stage in reference_recipe.stages),
    )
    native_recipe = replace(reference_recipe, provenance=implementation.recipe_provenance())
    frame = CoordinateFrame(
        frame_id="pair_gray_frame_v1",
        shape=tuple(values.shape),
        origin="top_left_pixel_centres",
        axes="x_right_y_down",
        units="pixel",
        transform_sha256=_digest("identity frame for threshold pair"),
    )
    source_sha = _digest(b"engineering-only threshold pair source")
    input_identity = CheckpointIdentity(
        source_sha256=source_sha,
        coordinate_frame=frame,
        stage_id="prepared_gray",
        recipe_id="prepared_gray_fixture",
        recipe_version="1",
        recipe_sha256=_digest("input recipe"),
        graph_sha256=_digest("input graph"),
        settings_sha256=_digest("input settings"),
        backend_id="seedfiddle.fixture",
        backend_version="1",
        environment_versions={"python": "3.12.10", "numpy": np.__version__},
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
        upstream_checkpoint_sha256=_digest("prepared gray's prior checkpoint"),
    )
    input_measurement = _measurement("prepared_gray_input")
    output_measurement = _measurement("threshold_pair_output")
    input_root = Path(temporary) / "input"
    output_root = Path(temporary) / "outputs"
    report_store = Path(temporary) / "reports"
    input_role = RasterRole("image", np.dtype(np.uint8).str, "HW", (0, 255))
    input_reference = publish_checkpoint(
        input_root,
        input_identity,
        {"gray": values},
        execution=ExecutionRecord(ExecutionStatus.AVAILABLE),
        measurement=input_measurement,
        raster_roles={"gray": input_role},
        metadata={"engineering_fixture_only": True},
        allow_cpu_transfer=True,
    )
    reference_stage = reference_recipe.stage(reference_stage.stage_id)
    native_stage = next(stage for stage in native_recipe.stages if stage.operation_id == operation_id)
    reference_output_identity = expected_reference_stage_identity(
        input_identity,
        reference_recipe,
        reference_stage.stage_id,
        input_reference.checkpoint_sha256,
    )
    native_output_identity = expected_native_threshold_stage_identity(
        input_identity,
        native_recipe,
        native_stage.stage_id,
        input_reference.checkpoint_sha256,
        implementation,
    )
    return {
        "input_root": input_root,
        "input_checkpoint_sha256": input_reference.checkpoint_sha256,
        "output_root": output_root,
        "artifact_store": report_store,
        "reference_recipe": reference_recipe,
        "native_recipe": native_recipe,
        "reference_stage_id": reference_stage.stage_id,
        "native_stage_id": native_stage.stage_id,
        "runner": PlantCVReferenceRunner(REFERENCE_PYTHON),
        "expected_source_sha256": source_sha,
        "expected_coordinate_frame": frame,
        "expected_input_identity": input_identity,
        "expected_input_measurement": input_measurement,
        "expected_reference_recipe_identity": reference_recipe.identity,
        "expected_native_recipe_identity": native_recipe.identity,
        "expected_reference_provenance": reference_recipe.provenance,
        "expected_native_implementation_identity": implementation,
        "expected_reference_output_identity": reference_output_identity,
        "expected_native_output_identity": native_output_identity,
        "input_bindings": {"gray": "gray"},
        "input_roles": {"gray": input_role},
        "output_measurement": output_measurement,
        "device": device,
        "allow_cpu_transfer": True,
    }


class ThresholdConformanceContractTests(unittest.TestCase):
    @unittest.skipUnless(REFERENCE_ENVIRONMENT_READY, "Pinned isolated PlantCV reference environment is not provisioned")
    def test_input_parameter_provenance_and_implementation_mismatches_precede_both_runs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args = _fixture(temporary, "threshold.mean")
            runner = args["runner"]
            with patch.object(runner, "run", side_effect=AssertionError("reference must not run")) as run:
                changed_native = dict(args)
                changed_native_recipe = args["native_recipe"]
                changed_stage = changed_native_recipe.stage(args["native_stage_id"])
                changed_stage = replace(
                    changed_stage,
                    parameters=tuple(
                        replace(parameter, value=1) if parameter.parameter_id == "offset" else parameter
                        for parameter in changed_stage.parameters
                    ),
                )
                changed_native_recipe = replace(
                    changed_native_recipe,
                    stages=tuple(changed_stage if stage.stage_id == changed_stage.stage_id else stage for stage in changed_native_recipe.stages),
                )
                changed_native.update(
                    native_recipe=changed_native_recipe,
                    expected_native_recipe_identity=changed_native_recipe.identity,
                )
                with self.assertRaisesRegex(ThresholdPairError, "parameters differ"):
                    run_threshold_conformance_pair(**changed_native)

                wrong_source = dict(args, expected_source_sha256="f" * 64)
                with self.assertRaisesRegex(ThresholdPairError, "source"):
                    run_threshold_conformance_pair(**wrong_source)

                with self.assertRaisesRegex(ThresholdPairError, "allow_cpu_transfer"):
                    run_threshold_conformance_pair(**dict(args, allow_cpu_transfer=False))

                wrong_measurement = _measurement("wrong_input_measurement")
                with self.assertRaisesRegex(ThresholdPairError, "input checkpoint failed replay validation"):
                    run_threshold_conformance_pair(**dict(args, expected_input_measurement=wrong_measurement))

                wrong_frame = replace(args["expected_coordinate_frame"], frame_id="stale_frame")
                with self.assertRaisesRegex(ThresholdPairError, "coordinate frame"):
                    run_threshold_conformance_pair(**dict(args, expected_coordinate_frame=wrong_frame))

                wrong_provenance = replace(args["expected_reference_provenance"], backend_id="forged.reference")
                with self.assertRaisesRegex(ThresholdPairError, "provenance/backend/environment"):
                    run_threshold_conformance_pair(**dict(args, expected_reference_provenance=wrong_provenance))

                if torch.cuda.is_available():
                    fake_implementation = replace(args["expected_native_implementation_identity"], implementation_sha256="0" * 64)
                    fake_recipe = replace(args["native_recipe"], provenance=fake_implementation.recipe_provenance())
                    fake_stage = fake_recipe.stage(args["native_stage_id"])
                    fake_output_identity = expected_native_threshold_stage_identity(
                        args["expected_input_identity"],
                        fake_recipe,
                        fake_stage.stage_id,
                        args["input_checkpoint_sha256"],
                        fake_implementation,
                    )
                    with self.assertRaisesRegex(ThresholdPairError, "implementation or device environment"):
                        run_threshold_conformance_pair(**dict(
                            args,
                            native_recipe=fake_recipe,
                            expected_native_recipe_identity=fake_recipe.identity,
                            expected_native_implementation_identity=fake_implementation,
                            expected_native_output_identity=fake_output_identity,
                        ))
                self.assertEqual(run.call_count, 0)

    @unittest.skipUnless(REFERENCE_ENVIRONMENT_READY, "Pinned isolated PlantCV reference environment is not provisioned")
    def test_cancelled_pair_retains_distinct_failed_states_and_cannot_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args = _fixture(temporary)
            cancelled = threading.Event()
            cancelled.set()
            with patch.object(args["runner"], "run", side_effect=AssertionError("cancellation must precede runner")) as run:
                outcome = run_threshold_conformance_pair(**dict(args, cancel=cancelled))
            report = outcome.report
            self.assertEqual(run.call_count, 0)
            self.assertEqual(report["pair_status"], "not_evaluated")
            self.assertIsNone(report["conformance"]["passed"])
            self.assertEqual(outcome.reference_outcome.run_record.state, StageRunState.FAILED)
            self.assertEqual(outcome.native_outcome.run_record.state, StageRunState.FAILED)
            self.assertIsNone(outcome.reference_outcome.checkpoint_reference)
            self.assertIsNone(outcome.native_outcome.checkpoint_reference)
            self.assertEqual(report["scientific_validity"]["certification"], "not_certified")

    @unittest.skipUnless(REFERENCE_ENVIRONMENT_READY, "Pinned isolated PlantCV reference environment is not provisioned")
    def test_native_failure_retains_valid_reference_artifact_without_false_pass(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("Injected CUDA execution failure requires an available CUDA device.")
        with tempfile.TemporaryDirectory() as temporary:
            args = _fixture(temporary)
            with patch.object(torch.cuda, "synchronize", side_effect=RuntimeError("injected pair failure")):
                outcome = run_threshold_conformance_pair(**args)
            report = outcome.report
            self.assertEqual(outcome.reference_outcome.run_record.state, StageRunState.EXECUTED)
            self.assertIsNotNone(outcome.reference_outcome.checkpoint_reference)
            self.assertEqual(outcome.native_outcome.run_record.state, StageRunState.FAILED)
            self.assertIn("injected pair failure", outcome.native_outcome.run_record.reason)
            self.assertTrue(report["stages"]["reference"]["evidence"]["checkpoint_validation"] == "verified")
            self.assertEqual(report["pair_status"], "not_evaluated")
            self.assertFalse(report["stage_execution_complete"])
            self.assertIsNone(report["conformance"]["passed"])

    @unittest.skipUnless(REFERENCE_ENVIRONMENT_READY, "Pinned isolated PlantCV reference environment is not provisioned")
    def test_reference_and_native_ambiguity_or_device_unavailability_never_claim_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            ambiguous = _fixture(temporary, "threshold.otsu", values=_image("threshold.otsu", ambiguous_otsu=True))
            outcome = run_threshold_conformance_pair(**ambiguous)
            report = outcome.report
            self.assertEqual(outcome.native_outcome.run_record.state, StageRunState.UNSUPPORTED)
            self.assertIsNone(outcome.native_outcome.checkpoint_reference)
            self.assertEqual(report["pair_status"], "not_evaluated")
            self.assertIsNone(report["conformance"]["passed"])

        with tempfile.TemporaryDirectory() as temporary:
            args = _fixture(temporary)
            with patch.object(torch.cuda, "is_available", return_value=False):
                outcome = run_threshold_conformance_pair(**args)
            self.assertEqual(outcome.reference_outcome.run_record.state, StageRunState.EXECUTED)
            self.assertEqual(outcome.native_outcome.run_record.state, StageRunState.UNAVAILABLE)
            self.assertIsNone(outcome.native_outcome.checkpoint_reference)
            self.assertEqual(outcome.report["pair_status"], "not_evaluated")

    @unittest.skipUnless(REFERENCE_ENVIRONMENT_READY and torch.cuda.is_available(), "Actual pair execution requires pinned PlantCV and CUDA")
    def test_actual_binary_mean_and_otsu_pairs_run_from_one_input_and_compare_masks(self) -> None:
        cases = (
            ("threshold.binary", _selected("threshold.binary"), _image("threshold.binary")),
            ("threshold.mean", _selected("threshold.mean"), _image("threshold.mean")),
            ("threshold.mean", {**_selected("threshold.mean"), "ksize": 4}, _image("threshold.mean")),
            ("threshold.otsu", _selected("threshold.otsu"), _image("threshold.otsu")),
        )
        for operation_id, selected, values in cases:
            with self.subTest(operation_id=operation_id), tempfile.TemporaryDirectory() as temporary:
                args = _fixture(temporary, operation_id, selected=selected, values=values)
                outcome = run_threshold_conformance_pair(**args)
                report = outcome.report
                self.assertEqual(report["pair_status"], "evaluated_pass")
                self.assertTrue(report["conformance"]["passed"])
                self.assertEqual(report["conformance"]["result"]["conformance"]["mismatch_pixel_count"], 0)
                self.assertEqual(outcome.reference_outcome.run_record.upstream_checkpoint_sha256, args["input_checkpoint_sha256"])
                self.assertEqual(outcome.native_outcome.run_record.upstream_checkpoint_sha256, args["input_checkpoint_sha256"])
                self.assertEqual(report["semantic_correspondence"]["operation_id"], operation_id)
                self.assertEqual(report["scientific_validity"]["status"], "not_evaluated")
                self.assertEqual(report["scientific_validity"]["certification"], "not_certified")
                self.assertTrue(outcome.report_path.is_file())
                contents = outcome.report_path.read_bytes()
                self.assertEqual(sha256(contents).hexdigest(), outcome.report_sha256)
                self.assertEqual(contents.decode("utf-8"), json.dumps(report, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")))
                self.assertEqual(list(outcome.report_path.parent.glob("*.tmp")), [])

    @unittest.skipUnless(REFERENCE_ENVIRONMENT_READY and torch.cuda.is_available(), "Empty pair execution requires pinned PlantCV and CUDA")
    def test_valid_empty_pair_is_evaluated_as_empty_not_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            selected = {"threshold": 255, "object_type": "light"}
            args = _fixture(temporary, "threshold.binary", selected=selected, values=_image("threshold.binary", empty=True))
            outcome = run_threshold_conformance_pair(**args)
            self.assertEqual(outcome.reference_outcome.run_record.state, StageRunState.EMPTY)
            self.assertEqual(outcome.native_outcome.run_record.state, StageRunState.EMPTY)
            report = outcome.report
            self.assertEqual(report["pair_status"], "evaluated_pass")
            self.assertEqual(report["conformance"]["result"]["conformance"]["reference_execution_status"], "empty")
            self.assertEqual(report["conformance"]["result"]["conformance"]["candidate_execution_status"], "empty")

    def test_report_publication_is_bounded_content_addressed_and_no_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = pair_module._prepare_report_directory(temporary)
            payload = {"stable": True, "status": "not_evaluated"}
            first = pair_module._publish_report(directory, payload)
            second = pair_module._publish_report(directory, payload)
            self.assertEqual(first, second)
            self.assertEqual(first[0].read_bytes(), json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))
            with self.assertRaisesRegex(ThresholdPairError, "exceeds"):
                pair_module._publish_report(directory, {"oversized": "x" * pair_module.MAX_THRESHOLD_PAIR_REPORT_BYTES})
            self.assertEqual(len(list(directory.glob("*.json"))), 1)
            self.assertEqual(list(directory.glob(".threshold-report-*.tmp")), [])

    def test_missing_reference_prerequisites_and_cuda_are_reported_as_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args = _fixture(temporary)
            args["runner"] = PlantCVReferenceRunner(Path(temporary) / "missing-reference-python.exe")
            with patch.object(torch.cuda, "is_available", return_value=False):
                outcome = run_threshold_conformance_pair(**args)
            self.assertEqual(outcome.reference_outcome.run_record.state, StageRunState.UNAVAILABLE)
            self.assertEqual(outcome.native_outcome.run_record.state, StageRunState.UNAVAILABLE)
            self.assertEqual(outcome.report["pair_status"], "not_evaluated")
            self.assertFalse(outcome.report["stage_execution_complete"])
            self.assertIsNone(outcome.report["conformance"]["passed"])


if __name__ == "__main__":
    unittest.main()
