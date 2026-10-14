from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from io import BytesIO
import threading
from pathlib import Path
import tempfile
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch

import seedvision.benchmarking.checkpoints as checkpoint_module
import seedvision.benchmarking.native_threshold_stage as native_stage_module
from seedvision.benchmarking.checkpoints import (
    CheckpointError,
    CheckpointIdentity,
    CheckpointLimits,
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
from seedvision.benchmarking.reference_runner import PlantCVReferenceRunner
from seedvision.benchmarking.native_threshold_stage import (
    NativeThresholdIdentity,
    NativeThresholdStageError,
    expected_native_threshold_stage_identity,
    native_threshold_identity,
    run_native_threshold_stage,
)


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PYTHON = ROOT / "artifacts" / "plantcv-reference-env" / "Scripts" / "python.exe"
REFERENCE_SOURCE_ROOT = ROOT / "artifacts" / "reference-sources" / "plantcv-v4.11.3"
REFERENCE_ENVIRONMENT_READY = all(path.exists() for path in (
    REFERENCE_PYTHON,
    REFERENCE_SOURCE_ROOT,
    ROOT / "artifacts" / "reference-sources" / "plantcv-source-audit.json",
    ROOT / "artifacts" / "reference-sources" / "plantcv-environment.json",
    ROOT / "config" / "plantcv-reference-requirements.txt",
))


def _identity(device: torch.device) -> NativeThresholdIdentity:
    if torch.cuda.is_available() and device.index is not None and device.index < torch.cuda.device_count():
        return native_threshold_identity(device)
    return NativeThresholdIdentity(
        implementation_sha256="a" * 64,
        python_version="3.12.10",
        torch_version="2.13.0+cu126",
        numpy_version=np.__version__,
        cuda_runtime_version="12.6",
        platform=sys.platform,
        machine="Unavailable test host",
        device_index=device.index if device.index is not None else 0,
        device_name="Unavailable test device",
        compute_capability="8.6",
    )


def _recipe(operation_id: str, identity: NativeThresholdIdentity, selected: dict | None = None):
    recipe_id = {
        "threshold.binary": "plantcv_seed_binary",
        "threshold.mean": "plantcv_seed_mean",
        "threshold.otsu": "plantcv_seed_otsu",
    }[operation_id]
    recipe = build_recipe_catalogue().recipe(recipe_id)
    stage = next(item for item in recipe.stages if item.operation_id == operation_id)
    selected = selected or {
        "threshold.binary": {"threshold": 127},
        "threshold.mean": {"ksize": 3, "offset": 1.25},
        "threshold.otsu": {},
    }[operation_id]
    changed_stage = replace(
        stage,
        parameters=tuple(
            replace(parameter, value=selected[parameter.parameter_id])
            if parameter.parameter_id in selected
            else parameter
            for parameter in stage.parameters
        ),
    )
    return replace(
        recipe,
        provenance=identity.recipe_provenance(),
        stages=tuple(changed_stage if item.stage_id == stage.stage_id else item for item in recipe.stages),
    ), stage.stage_id


def _measurement(name: str) -> MeasurementContract:
    return MeasurementContract(
        definition_id=name,
        definition_version="1",
        mask_type="binary_threshold_output",
        units="pixel",
        calibration_status="not_applicable",
        uncertainty="not_evaluated",
        eligible=False,
        reason="Native threshold output is not scientific validation.",
    )


def _image(values: np.ndarray) -> dict[str, object]:
    return {
        "color_order": "GRAY",
        "dtype": values.dtype.name,
        "value_range": [0, 255],
        "channel_conversion": "caller_supplied_gray",
        "geometric_transform": "fixture_identity_frame",
        "interpolation": "not_applicable",
    }


def _input_identity(frame: CoordinateFrame, source_sha256: str, values: np.ndarray) -> CheckpointIdentity:
    return CheckpointIdentity(
        source_sha256=source_sha256,
        coordinate_frame=frame,
        stage_id="prepared_gray",
        recipe_id="synthetic_input_fixture",
        recipe_version="1",
        recipe_sha256="1" * 64,
        graph_sha256="2" * 64,
        settings_sha256="3" * 64,
        backend_id="seedfiddle.fixture",
        backend_version="1",
        environment_versions={"python": "3.12.10"},
        image_semantics=_image(values),
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
        upstream_checkpoint_sha256="4" * 64,
    )


def _sample(operation_id: str) -> np.ndarray:
    rng = np.random.default_rng(0x51EED)
    if operation_id == "threshold.otsu":
        data = np.full((13, 19), 10, dtype=np.uint8)
        data[:, 9:] = 200
        return data
    return rng.integers(0, 256, size=(13, 19), dtype=np.uint8)


def _fixture(
    temporary: str,
    operation_id: str = "threshold.binary",
    *,
    values: np.ndarray | None = None,
    selected=None,
    implementation_identity: NativeThresholdIdentity | None = None,
):
    device = torch.device("cuda", torch.cuda.current_device() if torch.cuda.is_available() else 0)
    identity = _identity(device) if implementation_identity is None else implementation_identity
    recipe, stage_id = _recipe(operation_id, identity, selected)
    values = _sample(operation_id) if values is None else values
    source_sha256 = sha256(b"synthetic engineering-only grayscale frame").hexdigest()
    frame = CoordinateFrame(
        frame_id="synthetic_gray_test_v1",
        shape=tuple(values.shape),
        origin="top_left_pixel_centres",
        axes="x_right_y_down",
        units="pixel",
        transform_sha256=sha256(b"test identity transform").hexdigest(),
    )
    input_identity = _input_identity(frame, source_sha256, values)
    input_measurement = _measurement("native_input_fixture")
    input_role = RasterRole("image", np.dtype(np.uint8).str, "HW", (0, 255))
    input_root = Path(temporary) / "input"
    output_root = Path(temporary) / "output"
    input_reference = publish_checkpoint(
        input_root,
        input_identity,
        {"gray": values},
        execution=ExecutionRecord(ExecutionStatus.AVAILABLE),
        measurement=input_measurement,
        raster_roles={"gray": input_role},
        metadata={"engineering_fixture": True},
        allow_cpu_transfer=True,
    )
    output_measurement = _measurement("native_threshold_output")
    output_identity = expected_native_threshold_stage_identity(
        input_identity, recipe, stage_id, input_reference.checkpoint_sha256, identity,
    )
    parameters = recipe.parameters_for_runner(stage_id)
    args = {
        "input_root": input_root,
        "input_checkpoint_sha256": input_reference.checkpoint_sha256,
        "recipe": recipe,
        "stage_id": stage_id,
        "output_root": output_root,
        "expected_source_sha256": source_sha256,
        "expected_coordinate_frame": frame,
        "expected_recipe_identity": recipe.identity,
        "expected_upstream_checkpoint_sha256": input_reference.checkpoint_sha256,
        "expected_input_identity": input_identity,
        "expected_input_measurement": input_measurement,
        "expected_output_identity": output_identity,
        "expected_implementation_identity": identity,
        "device": device,
        "input_bindings": {"gray": "gray"},
        "input_roles": {"gray": input_role},
        "output_measurement": output_measurement,
        "allow_cpu_transfer": True,
    }
    return args, values, parameters


class NativeThresholdContractTests(unittest.TestCase):
    def test_loaded_code_hash_refuses_same_size_source_edit_after_import(self) -> None:
        module_path = Path(native_stage_module.__file__).resolve()
        baseline = native_stage_module._implementation_digest()
        self.assertRegex(baseline, r"^[0-9a-f]{64}$")
        original_open = Path.open
        original = native_stage_module._read_source_bounded(module_path)
        needle = b'return f"cuda:{self.device_index}"'
        replacement = b'return f"cu:{self.device_index+0}"'
        self.assertEqual(len(needle), len(replacement))
        self.assertIn(needle, original)
        changed = original.replace(needle, replacement, 1)
        self.assertEqual(len(changed), len(original))

        def open_source(path, *args, **kwargs):
            if path.resolve() == module_path:
                return BytesIO(changed)
            return original_open(path, *args, **kwargs)

        with patch.object(Path, "open", open_source):
            with self.assertRaisesRegex(
                NativeThresholdStageError,
                r"Loaded implementation .*:NativeThresholdIdentity\.device differs from freshly compiled source",
            ):
                native_stage_module._implementation_digest()

    def test_native_identity_has_distinct_exact_provenance(self) -> None:
        device = torch.device("cuda", torch.cuda.current_device() if torch.cuda.is_available() else 0)
        identity = _identity(device)
        provenance = identity.recipe_provenance()
        self.assertEqual(provenance.mode, RecipeProvenanceMode.NATIVE_COMPATIBLE)
        self.assertEqual(provenance.backend_id, identity.backend_id)
        self.assertEqual(provenance.environment_versions, identity.environment_versions)
        self.assertEqual(identity.to_dict()["implementation_sha256"], provenance.environment_versions["native_implementation_sha256"])

    def test_original_reference_provenance_cannot_be_used_for_native_execution(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args, _values, _parameters = _fixture(temporary)
            original = build_recipe_catalogue().recipe("plantcv_seed_binary")
            wrong_recipe = replace(args["recipe"], provenance=original.provenance)
            with self.assertRaisesRegex(NativeThresholdStageError, "provenance differs"):
                run_native_threshold_stage(**dict(
                    args,
                    recipe=wrong_recipe,
                    expected_recipe_identity=wrong_recipe.identity,
                ))

    def test_stale_source_frame_upstream_recipe_and_role_bindings_reject_before_execution(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args, _values, _parameters = _fixture(temporary)
            with patch.object(native_stage_module, "threshold_binary", side_effect=AssertionError("must not execute")):
                with self.assertRaisesRegex(NativeThresholdStageError, "source SHA"):
                    run_native_threshold_stage(**dict(args, expected_source_sha256="f" * 64))
                with self.assertRaisesRegex(NativeThresholdStageError, "upstream digest"):
                    run_native_threshold_stage(**dict(args, expected_upstream_checkpoint_sha256="e" * 64))
                changed_frame = replace(args["expected_coordinate_frame"], frame_id="stale_frame")
                with self.assertRaisesRegex(NativeThresholdStageError, "frame"):
                    run_native_threshold_stage(**dict(args, expected_coordinate_frame=changed_frame))
                changed_role = RasterRole("raster", np.dtype(np.uint8).str, "HW", (0, 255))
                with self.assertRaisesRegex(NativeThresholdStageError, "role differs"):
                    run_native_threshold_stage(**dict(args, input_roles={"gray": changed_role}))
                changed_recipe = replace(args["recipe"], provenance=replace(
                    args["recipe"].provenance,
                    environment_versions=dict(args["recipe"].provenance.environment_versions) | {"torch": "0.0"},
                ))
                with self.assertRaisesRegex(NativeThresholdStageError, "Recipe identity"):
                    run_native_threshold_stage(**dict(args, recipe=changed_recipe))

    def test_input_size_limit_rejects_before_cuda_upload_or_calculation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args, _values, _parameters = _fixture(temporary)
            small_limits = CheckpointLimits(max_total_array_bytes=100, max_array_bytes=100)
            with patch.object(native_stage_module, "threshold_binary", side_effect=AssertionError("must not execute")):
                with self.assertRaises(CheckpointError):
                    run_native_threshold_stage(**dict(args, limits=small_limits))

    def test_output_metadata_limit_is_checked_before_checkpoint_host_transfer(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("Output tensor transfer preflight requires a CUDA device.")
        with tempfile.TemporaryDirectory() as temporary:
            args, _values, _parameters = _fixture(temporary)
            # The exact input manifest is 2,282 bytes in this fixture, while
            # the native output contract needs more than this lowered cap.
            limits = CheckpointLimits(max_metadata_bytes=3000)
            with patch.object(
                checkpoint_module,
                "_numpy_payload",
                side_effect=AssertionError("metadata rejection must precede CUDA-to-CPU tensor transfer"),
            ):
                with self.assertRaisesRegex(NativeThresholdStageError, "metadata exceeds"):
                    run_native_threshold_stage(**dict(args, limits=limits))
            self.assertEqual(list(args["output_root"].glob("*.checkpoint")), [])

    def test_cancellation_before_execution_is_failed_without_an_output_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args, _values, _parameters = _fixture(temporary)
            cancelled = threading.Event()
            cancelled.set()
            with patch.object(native_stage_module, "threshold_binary", side_effect=AssertionError("cancelled must not execute")):
                outcome = run_native_threshold_stage(**dict(args, cancel=cancelled))
            self.assertEqual(outcome.run_record.state, StageRunState.FAILED)
            self.assertIn("Cancelled", outcome.run_record.reason)
            self.assertIsNone(outcome.checkpoint_reference)

    def test_cuda_runtime_failure_is_recorded_without_an_output_checkpoint(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA failure state requires the CUDA bridge path.")
        with tempfile.TemporaryDirectory() as temporary:
            args, _values, _parameters = _fixture(temporary)
            with patch.object(torch.cuda, "synchronize", side_effect=RuntimeError("injected CUDA sync failure")):
                outcome = run_native_threshold_stage(**args)
            self.assertEqual(outcome.run_record.state, StageRunState.FAILED)
            self.assertIn("injected CUDA sync failure", outcome.run_record.reason)
            self.assertIsNone(outcome.checkpoint_reference)
            self.assertEqual(list(args["output_root"].glob("*.checkpoint")), [])

    def test_unavailable_cuda_is_recorded_without_cpu_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args, _values, _parameters = _fixture(temporary)
            with (
                patch.object(torch.cuda, "is_available", return_value=False),
                patch.object(native_stage_module, "threshold_binary", side_effect=AssertionError("no CPU fallback")),
            ):
                outcome = run_native_threshold_stage(**args)
            self.assertEqual(outcome.run_record.state, StageRunState.UNAVAILABLE)
            self.assertIn("unavailable", outcome.run_record.reason)
            self.assertIsNone(outcome.checkpoint_reference)

    def test_mean_outside_native_domain_is_unsupported_without_execution(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args, _values, _parameters = _fixture(
                temporary,
                "threshold.mean",
                selected={"ksize": 257, "offset": 1.25},
            )
            with patch.object(native_stage_module, "threshold_mean", side_effect=AssertionError("unsupported domain must not execute")):
                outcome = run_native_threshold_stage(**args)
            self.assertEqual(outcome.run_record.state, StageRunState.UNSUPPORTED)
            self.assertIn("ksize", outcome.run_record.reason)
            self.assertIsNone(outcome.checkpoint_reference)

    def test_fresh_runtime_pin_rejects_a_dishonest_implementation_digest(self) -> None:
        actual = native_threshold_identity(torch.device("cuda", torch.cuda.current_device()))
        dishonest = replace(actual, implementation_sha256="0" * 64)
        with tempfile.TemporaryDirectory() as temporary:
            args, _values, _parameters = _fixture(temporary, implementation_identity=dishonest)
            with self.assertRaisesRegex(NativeThresholdStageError, "differs from its caller pin"):
                run_native_threshold_stage(**args)
            self.assertEqual(list(args["output_root"].glob("*.checkpoint")), [])


@unittest.skipUnless(torch.cuda.is_available(), "Native threshold stage requires CUDA")
class NativeThresholdCudaStageTests(unittest.TestCase):
    @unittest.skipUnless(REFERENCE_ENVIRONMENT_READY, "Pinned isolated PlantCV reference environment is not provisioned")
    def test_actual_native_checkpoints_match_pinned_runner_for_binary_mean_and_otsu(self) -> None:
        cases = (
            ("threshold.binary", _sample("threshold.binary"), {"threshold": 127, "object_type": "light"}),
            ("threshold.binary", _sample("threshold.binary"), {"threshold": 127, "object_type": "dark"}),
            ("threshold.mean", _sample("threshold.mean"), {"ksize": 3, "offset": 1.25, "object_type": "light"}),
            ("threshold.mean", _sample("threshold.mean"), {"ksize": 3, "offset": 1.25, "object_type": "dark"}),
            ("threshold.otsu", _sample("threshold.otsu"), {"object_type": "light"}),
            ("threshold.otsu", _sample("threshold.otsu"), {"object_type": "dark"}),
        )
        runner = PlantCVReferenceRunner(REFERENCE_PYTHON)
        for operation, image, parameters_override in cases:
            with self.subTest(operation=operation), tempfile.TemporaryDirectory() as temporary:
                args, values, parameters = _fixture(
                    temporary,
                    operation,
                    values=image,
                    selected=parameters_override,
                )
                original_payload = checkpoint_module._numpy_payload
                observed_transfer = []

                def require_cuda_until_serializer(value, **keywords):
                    if keywords.get("name") == "mask":
                        self.assertTrue(torch.is_tensor(value))
                        self.assertEqual(value.device, args["device"])
                        self.assertEqual(value.dtype, torch.uint8)
                        observed_transfer.append(str(value.device))
                    return original_payload(value, **keywords)

                with patch.object(checkpoint_module, "_numpy_payload", side_effect=require_cuda_until_serializer):
                    outcome = run_native_threshold_stage(**args)
                self.assertIn(outcome.run_record.state, {StageRunState.EXECUTED, StageRunState.EMPTY})
                self.assertEqual(observed_transfer, [str(args["device"])])
                reference = runner.run(operation, {"gray": values}, parameters)
                loaded = load_checkpoint(
                    args["output_root"],
                    outcome.checkpoint_reference.checkpoint_sha256,
                    expected_identity=args["expected_output_identity"],
                    expected_measurement=args["output_measurement"],
                )
                np.testing.assert_array_equal(loaded.arrays["mask"], reference.arrays["mask"])
                self.assertEqual(outcome.run_record.upstream_checkpoint_sha256, args["input_checkpoint_sha256"])
                self.assertEqual(loaded.metadata["selected_parameters"], parameters)
                self.assertEqual(loaded.metadata["effective_parameters"], parameters)
                self.assertEqual(loaded.metadata["execution_mode"], RecipeProvenanceMode.NATIVE_COMPATIBLE.value)
                self.assertEqual(loaded.metadata["output_inventory"][0]["device"], str(args["device"]))
                self.assertEqual(loaded.metadata["output_raster_roles"]["mask"]["kind"], "mask")
                self.assertGreaterEqual(loaded.metadata["computation_runtime_ns"], 0)
                self.assertGreaterEqual(loaded.metadata["input_transfer_runtime_ns"], 0)
                self.assertEqual(loaded.metadata["implementation_identity"], args["expected_implementation_identity"].to_dict())
                self.assertFalse(loaded.measurement.eligible)
                self.assertEqual(loaded.metadata["scientific_validation_status"], "not_certified")
                self.assertEqual(loaded.metadata["reference_comparison_status"], "not_evaluated")

    def test_otsu_ambiguity_is_unsupported_and_never_publishes_a_fake_mask(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            image = np.array([[0, 0, 1, 2, 2]], dtype=np.uint8)
            args, _values, _parameters = _fixture(temporary, "threshold.otsu", values=image)
            outcome = run_native_threshold_stage(**args)
            self.assertEqual(outcome.run_record.state, StageRunState.UNSUPPORTED)
            self.assertIn("Otsu", outcome.run_record.reason)
            self.assertIsNone(outcome.checkpoint_reference)
            self.assertEqual(list(args["output_root"].glob("*.checkpoint")), [])

    def test_mean_scaling_boundary_is_unsupported_without_publishing_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            image = np.full((255,3),255,dtype=np.uint8)
            image[0,0],image[1,0],image[127,2] = 0,254,253
            args,_,_ = _fixture(temporary,'threshold.mean',values=image,
                selected={'ksize':255,'offset':2,'object_type':'dark'})
            outcome = run_native_threshold_stage(**args)
            self.assertEqual(outcome.run_record.state,StageRunState.UNSUPPORTED)
            self.assertIn('adaptive mean',outcome.run_record.reason)
            self.assertIsNone(outcome.checkpoint_reference)
            self.assertEqual(list(args['output_root'].glob('*.checkpoint')),[])

    def test_empty_output_is_published_and_serialization_cancellation_cleans_staging(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            image = np.full((13, 19), 255, dtype=np.uint8)
            args, _values, _parameters = _fixture(
                temporary,
                "threshold.binary",
                values=image,
                selected={"threshold": 255},
            )
            outcome = run_native_threshold_stage(**args)
            self.assertEqual(outcome.run_record.state, StageRunState.EMPTY)
            loaded = load_checkpoint(
                args["output_root"],
                outcome.checkpoint_reference.checkpoint_sha256,
                expected_identity=args["expected_output_identity"],
                expected_measurement=args["output_measurement"],
            )
            self.assertEqual(loaded.execution.status, ExecutionStatus.EMPTY)
            self.assertFalse(np.any(loaded.arrays["mask"]))

        with tempfile.TemporaryDirectory() as temporary:
            args, _values, _parameters = _fixture(temporary)
            cancelled = threading.Event()
            original_savez = checkpoint_module.np.savez

            def save_then_cancel(*positional, **keywords):
                original_savez(*positional, **keywords)
                cancelled.set()

            with patch.object(checkpoint_module.np, "savez", side_effect=save_then_cancel):
                outcome = run_native_threshold_stage(**dict(args, cancel=cancelled))
            self.assertEqual(outcome.run_record.state, StageRunState.FAILED)
            self.assertIn("Cancelled", outcome.run_record.reason)
            self.assertIsNone(outcome.checkpoint_reference)
            self.assertEqual(list(args["output_root"].glob("*.checkpoint")), [])
            self.assertEqual(list(args["output_root"].glob(".*")), [])


if __name__ == "__main__":
    unittest.main()
