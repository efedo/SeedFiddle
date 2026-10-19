from __future__ import annotations

from tests.catalogue_fixtures import CatalogueTestCase

from tests.functional_units import functional_units

from dataclasses import replace
from hashlib import sha256
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import seedvision.benchmarking.native_watershed_stage as native_stage_module
from seedvision.benchmarking.checkpoints import (
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
from seedvision.benchmarking.native_watershed_stage import (
    MAX_NATIVE_MARKER_PEAK_PIXELS,
    NATIVE_DISTANCE_INPUT_PORT,
    NATIVE_DISTANCE_OPERATION_ID,
    NATIVE_DISTANCE_OUTPUT_NAME,
    NATIVE_MARKER_INPUT_PORT,
    NATIVE_MARKER_OPERATION_ID,
    NATIVE_MARKER_OUTPUT_NAME,
    NativeWatershedIdentity,
    NativeWatershedStageError,
    NativeWatershedStageStatus,
    expected_native_watershed_stage_identity,
    native_watershed_identity,
    run_native_watershed_stage,
)


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PYTHON = ROOT / "artifacts" / "plantcv-reference-env" / "Scripts" / "python.exe"
REFERENCE_ENVIRONMENT_READY = all(path.exists() for path in (
    REFERENCE_PYTHON,
    ROOT / "artifacts" / "reference-sources" / "plantcv-v4.11.3",
    ROOT / "artifacts" / "reference-sources" / "plantcv-source-audit.json",
    ROOT / "artifacts" / "reference-sources" / "plantcv-environment.json",
    ROOT / "config" / "plantcv-reference-requirements.txt",
))
HAS_CUDA = torch.cuda.is_available() and torch.cuda.device_count() > 0


def _measurement(name: str) -> MeasurementContract:
    return MeasurementContract(
        definition_id=name,
        definition_version="1",
        mask_type="native_distance_or_marker_raster",
        units="pixel",
        calibration_status="not_applicable",
        uncertainty="not_evaluated",
        eligible=False,
        reason="Native watershed intermediates do not validate scientific measurements.",
    )


def _mask_semantics(representation: str) -> dict[str, object]:
    return {
        "foreground_polarity": "true_peak_pixels" if representation == "boolean" else "light",
        "representation": representation,
        "connectivity": "source_defined_binary_foreground",
        "validity_region": "full_input_frame",
        "border_policy": "prepared_input",
    }


def _instance_semantics() -> dict[str, object]:
    return {
        "label_dtype": "<i4",
        "background_label": 0,
        "label_id_mapping": "not_created",
        "split_boundary_ownership": "not_applicable",
        "visibility": "not_applicable",
        "eligibility": "not_evaluated",
    }


def _synthetic_identity() -> NativeWatershedIdentity:
    return NativeWatershedIdentity(
        implementation_sha256=native_stage_module._implementation_digest(),
        reference_source_sha256=native_stage_module._reference_source_sha256(),
        python_version="3.12.10",
        torch_version=str(torch.__version__),
        numpy_version=np.__version__,
        cuda_runtime_version="12.6",
        platform="test-platform",
        machine="test-machine",
        device_index=0,
        device_name="Test CUDA device",
        compute_capability="8.6",
    )


def _recipe(identity: NativeWatershedIdentity, recipe_id: str, operation_id: str):
    recipe = build_recipe_catalogue().recipe(recipe_id)
    stage = next(item for item in recipe.stages if item.operation_id == operation_id)
    return replace(recipe, provenance=identity.recipe_provenance()), stage.stage_id


@functional_units
class NativeWatershedIdentityTests(CatalogueTestCase):
    def check_test_loaded_distance_and_marker_implementation_identity_is_stable(self) -> None:
        first = native_stage_module._implementation_digest()
        self.assertRegex(first, r"^[0-9a-f]{64}$")
        self.assertEqual(first, native_stage_module._implementation_digest())
        with patch.object(
            native_stage_module._distance_module,
            "DistanceResourceError",
            type("ReplacedDistanceResourceError", (ValueError,), {}),
        ):
            with self.assertRaisesRegex(NativeWatershedStageError, "exception type changed"):
                native_stage_module._implementation_digest()

    def check_test_stage_identity_binds_recipe_source_backend_and_upstream(self) -> None:
        identity = _synthetic_identity()
        for operation_id in (NATIVE_DISTANCE_OPERATION_ID, NATIVE_MARKER_OPERATION_ID):
            recipe, stage_id = _recipe(identity, "plantcv_seed_binary", operation_id)
            representation = "uint8_0_255" if operation_id == NATIVE_DISTANCE_OPERATION_ID else "boolean"
            predecessor = (
                "pcv_seed_binary_dilate"
                if operation_id == NATIVE_DISTANCE_OPERATION_ID
                else "pcv_seed_binary_ws_local_maxima"
            )
            input_identity = CheckpointIdentity(
                source_sha256="1" * 64,
                coordinate_frame=CoordinateFrame(
                    frame_id="watershed_identity_frame",
                    shape=(8, 11),
                    origin="top_left_pixel_centres",
                    axes="x_right_y_down",
                    units="pixel",
                    transform_sha256="2" * 64,
                ),
                stage_id=predecessor,
                recipe_id=recipe.recipe_id,
                recipe_version=recipe.recipe_version,
                recipe_sha256=recipe.identity.recipe_sha256,
                graph_sha256=recipe.identity.graph_sha256,
                settings_sha256=recipe.identity.settings_sha256,
                backend_id="seedfiddle.fixture",
                backend_version="1",
                environment_versions={"fixture": "1"},
                image_semantics={
                    "color_order": "GRAY", "dtype": "uint8", "value_range": [0, 255],
                    "channel_conversion": "fixture", "geometric_transform": "identity",
                    "interpolation": "not_applicable",
                },
                mask_semantics=_mask_semantics(representation),
                instance_semantics=_instance_semantics(),
                upstream_checkpoint_sha256="3" * 64,
            )
            first = expected_native_watershed_stage_identity(
                input_identity, recipe, stage_id, "4" * 64, identity,
            )
            changed_upstream = expected_native_watershed_stage_identity(
                input_identity, recipe, stage_id, "5" * 64, identity,
            )
            self.assertEqual(first.stage_id, stage_id)
            self.assertEqual(first.backend_id, identity.backend_id)
            self.assertEqual(first.upstream_checkpoint_sha256, "4" * 64)
            self.assertNotEqual(first, changed_upstream)
            stale_settings = replace(input_identity, settings_sha256="5" * 64)
            with self.assertRaisesRegex(NativeWatershedStageError, "exact selected recipe and settings"):
                expected_native_watershed_stage_identity(
                    stale_settings, recipe, stage_id, "4" * 64, identity,
                )
            if operation_id == NATIVE_MARKER_OPERATION_ID:
                self.assertEqual(first.instance_semantics["visibility"], "peak_components_only")
            else:
                self.assertEqual(first.instance_semantics, input_identity.instance_semantics)

        mean_recipe, mean_stage_id = _recipe(
            identity, "plantcv_seed_mean", NATIVE_DISTANCE_OPERATION_ID,
        )
        binary_input = replace(
            input_identity,
            stage_id=next(
                stage.stage_id for stage in mean_recipe.stages
                if stage.operation_id == "morphology.dilate"
            ),
        )
        with self.assertRaisesRegex(NativeWatershedStageError, "exact selected recipe and settings"):
            expected_native_watershed_stage_identity(
                binary_input, mean_recipe, mean_stage_id, "4" * 64, identity,
            )

    def check_test_archive_preflight_rejects_before_cuda_probe_or_primitive(self) -> None:
        identity = _synthetic_identity()
        recipe, stage_id = _recipe(identity, "plantcv_seed_binary", NATIVE_DISTANCE_OPERATION_ID)
        mask = np.zeros((8, 11), dtype=np.uint8)
        mask[1:7, 2:9] = 255
        frame = CoordinateFrame(
            frame_id="native_watershed_archive_frame",
            shape=tuple(mask.shape),
            origin="top_left_pixel_centres",
            axes="x_right_y_down",
            units="pixel",
            transform_sha256=sha256(b"archive-preflight-frame").hexdigest(),
        )
        source = sha256(b"archive-preflight-source").hexdigest()
        input_identity = CheckpointIdentity(
            source_sha256=source,
            coordinate_frame=frame,
            stage_id="pcv_seed_binary_dilate",
            recipe_id=recipe.recipe_id,
            recipe_version=recipe.recipe_version,
            recipe_sha256=recipe.identity.recipe_sha256,
            graph_sha256=recipe.identity.graph_sha256,
            settings_sha256=recipe.identity.settings_sha256,
            backend_id="seedfiddle.fixture",
            backend_version="1",
            environment_versions={"numpy": np.__version__},
            image_semantics={
                "color_order": "GRAY", "dtype": "uint8", "value_range": [0, 255],
                "channel_conversion": "fixture", "geometric_transform": "identity",
                "interpolation": "not_applicable",
            },
            mask_semantics=_mask_semantics("uint8_0_255"),
            instance_semantics=_instance_semantics(),
            upstream_checkpoint_sha256="6" * 64,
        )
        role = RasterRole("mask", np.dtype(np.uint8).str, "HW")
        input_measurement = _measurement("archive_preflight_input")
        with tempfile.TemporaryDirectory(prefix="watershed-archive-preflight-") as directory:
            input_root = Path(directory) / "input"
            input_reference = publish_checkpoint(
                input_root,
                input_identity,
                {NATIVE_DISTANCE_INPUT_PORT: mask},
                execution=ExecutionRecord(ExecutionStatus.AVAILABLE),
                measurement=input_measurement,
                raster_roles={NATIVE_DISTANCE_INPUT_PORT: role},
                metadata={"engineering_fixture": True},
                allow_cpu_transfer=True,
            )
            output_identity = expected_native_watershed_stage_identity(
                input_identity, recipe, stage_id, input_reference.checkpoint_sha256, identity,
            )
            exact_size = native_stage_module._preflight_output_archive(
                NATIVE_DISTANCE_OUTPUT_NAME, frame.shape, np.dtype(np.float32), CheckpointLimits(),
            )
            arguments = {
                "input_root": input_root,
                "input_checkpoint_sha256": input_reference.checkpoint_sha256,
                "recipe": recipe,
                "stage_id": stage_id,
                "output_root": Path(directory) / "output",
                "expected_source_sha256": source,
                "expected_coordinate_frame": frame,
                "expected_recipe_identity": recipe.identity,
                "expected_upstream_checkpoint_sha256": input_reference.checkpoint_sha256,
                "expected_input_identity": input_identity,
                "expected_input_measurement": input_measurement,
                "expected_output_identity": output_identity,
                "expected_implementation_identity": identity,
                "device": torch.device("cuda", 0),
                "input_bindings": {NATIVE_DISTANCE_INPUT_PORT: NATIVE_DISTANCE_INPUT_PORT},
                "input_roles": {NATIVE_DISTANCE_INPUT_PORT: role},
                "output_measurement": _measurement("archive_preflight_output"),
                "allow_cpu_transfer": True,
                "limits": CheckpointLimits(max_npz_bytes=exact_size - 1),
            }
            with patch.object(torch.cuda, "is_available", side_effect=AssertionError("archive preflight was late")):
                with patch.object(native_stage_module, "distance_map") as primitive:
                    with self.assertRaisesRegex(NativeWatershedStageError, "max_npz_bytes"):
                        run_native_watershed_stage(**arguments)
                    primitive.assert_not_called()

    def test_workflow_lifecycle(self):
        """2 related NativeWatershedIdentityTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_loaded_distance_and_marker_implementation_identity_is_stable)
        self.scenario(self.check_test_stage_identity_binds_recipe_source_backend_and_upstream)

    def test_validation_and_recovery(self):
        """1 related NativeWatershedIdentityTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_archive_preflight_rejects_before_cuda_probe_or_primitive)


@unittest.skipUnless(
    HAS_CUDA and REFERENCE_ENVIRONMENT_READY,
    "the native watershed stage bridge requires CUDA and the pinned local PlantCV reference environment",
)
class NativeWatershedCudaStageTests(CatalogueTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.device = torch.device("cuda", torch.cuda.current_device())
        cls.implementation = native_watershed_identity(cls.device)
        cls.reference_runner = PlantCVReferenceRunner(REFERENCE_PYTHON)

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="native-watershed-stage-")
        self.root = Path(self.temporary.name)
        self.source_sha256 = sha256(b"native watershed synthetic engineering fixture").hexdigest()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _case(self, operation_id: str, values: np.ndarray) -> dict[str, object]:
        recipe_id = "plantcv_seed_binary"
        recipe, stage_id = _recipe(self.implementation, recipe_id, operation_id)
        is_distance = operation_id == NATIVE_DISTANCE_OPERATION_ID
        if values.ndim != 2 or values.size == 0:
            raise AssertionError("Test stage inputs must be non-empty two-dimensional arrays.")
        frame = CoordinateFrame(
            frame_id="native_watershed_test_frame",
            shape=tuple(values.shape),
            origin="top_left_pixel_centres",
            axes="x_right_y_down",
            units="pixel",
            transform_sha256=sha256(b"native watershed test identity frame").hexdigest(),
        )
        input_port = NATIVE_DISTANCE_INPUT_PORT if is_distance else NATIVE_MARKER_INPUT_PORT
        input_array_name = "binary_mask" if is_distance else "peak_mask"
        output_name = NATIVE_DISTANCE_OUTPUT_NAME if is_distance else NATIVE_MARKER_OUTPUT_NAME
        predecessor = "pcv_seed_binary_dilate" if is_distance else "pcv_seed_binary_ws_local_maxima"
        representation = "uint8_0_255" if is_distance else "boolean"
        input_identity = CheckpointIdentity(
            source_sha256=self.source_sha256,
            coordinate_frame=frame,
            stage_id=predecessor,
            recipe_id=recipe_id,
            recipe_version=recipe.recipe_version,
            recipe_sha256=recipe.identity.recipe_sha256,
            graph_sha256=recipe.identity.graph_sha256,
            settings_sha256=recipe.identity.settings_sha256,
            backend_id="seedfiddle.fixture",
            backend_version="1",
            environment_versions={"numpy": np.__version__},
            image_semantics={
                "color_order": "GRAY", "dtype": "uint8", "value_range": [0, 255],
                "channel_conversion": "fixture", "geometric_transform": "identity",
                "interpolation": "not_applicable",
            },
            mask_semantics=_mask_semantics(representation),
            instance_semantics=_instance_semantics(),
            upstream_checkpoint_sha256="6" * 64,
        )
        input_role = RasterRole("mask", np.dtype(values.dtype).str, "HW")
        input_status = ExecutionStatus.EMPTY if not np.any(values) else ExecutionStatus.AVAILABLE
        base = self.root / operation_id.rsplit(".", 1)[-1]
        input_root = base / "input"
        output_root = base / "output"
        input_measurement = _measurement("native_watershed_input")
        input_reference = publish_checkpoint(
            input_root,
            input_identity,
            {input_array_name: values},
            execution=ExecutionRecord(input_status),
            measurement=input_measurement,
            raster_roles={input_array_name: input_role},
            metadata={"engineering_fixture": True},
            allow_cpu_transfer=True,
        )
        output_identity = expected_native_watershed_stage_identity(
            input_identity, recipe, stage_id, input_reference.checkpoint_sha256, self.implementation,
        )
        return {
            "input_root": input_root,
            "input_checkpoint_sha256": input_reference.checkpoint_sha256,
            "recipe": recipe,
            "stage_id": stage_id,
            "output_root": output_root,
            "expected_source_sha256": self.source_sha256,
            "expected_coordinate_frame": frame,
            "expected_recipe_identity": recipe.identity,
            "expected_upstream_checkpoint_sha256": input_reference.checkpoint_sha256,
            "expected_input_identity": input_identity,
            "expected_input_measurement": input_measurement,
            "expected_output_identity": output_identity,
            "expected_implementation_identity": self.implementation,
            "device": self.device,
            "input_bindings": {input_port: input_array_name},
            "input_roles": {input_port: input_role},
            "output_measurement": _measurement("native_watershed_output"),
            "allow_cpu_transfer": True,
            "input_values": values,
            "output_name": output_name,
            "is_distance": is_distance,
        }

    @staticmethod
    def _run(case: dict[str, object]):
        args = {key: value for key, value in case.items() if key not in {"input_values", "output_name", "is_distance"}}
        return run_native_watershed_stage(**args)






    def test_original_reference_recipe_is_never_treated_as_a_native_internal_replay(self) -> None:
        case = self._case(NATIVE_MARKER_OPERATION_ID, np.array([[True, False]], dtype=np.bool_))
        recipe = replace(
            case["recipe"],
            provenance=replace(case["recipe"].provenance, mode=RecipeProvenanceMode.ORIGINAL_REFERENCE),
        )
        outcome = self._run(case | {"recipe": recipe, "expected_recipe_identity": recipe.identity, "expected_output_identity": None})
        self.assertEqual(outcome.status, NativeWatershedStageStatus.UNSUPPORTED)
        self.assertEqual(outcome.run_record.state, StageRunState.UNSUPPORTED)
        self.assertIn("native-compatible", outcome.run_record.reason)
        self.assertIsNone(outcome.checkpoint_reference)


if __name__ == "__main__":
    unittest.main()
