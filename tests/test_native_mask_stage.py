from __future__ import annotations

from tests.catalogue_fixtures import CatalogueTestCase

from dataclasses import replace
from hashlib import sha256
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import seedvision.benchmarking.native_mask_stage as native_stage_module
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
from seedvision.benchmarking.recipes import RecipeSpec, StageRunState
from seedvision.benchmarking.reference_runner import PlantCVReferenceRunner
from seedvision.benchmarking.native_mask_stage import (
    NativeMaskIdentity,
    NativeMaskStageError,
    NativeMaskStageStatus,
    expected_native_mask_stage_identity,
    native_mask_identity,
    run_native_mask_stage,
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
HAS_CUDA = torch.cuda.is_available() and torch.cuda.device_count() > 0


def _measurement(name: str) -> MeasurementContract:
    return MeasurementContract(
        definition_id=name,
        definition_version="1",
        mask_type="native_image_or_binary_mask",
        units="pixel",
        calibration_status="not_applicable",
        uncertainty="not_evaluated",
        eligible=False,
        reason="A native image/mask checkpoint does not validate scientific measurements.",
    )


def _image_semantics() -> dict[str, object]:
    return {
        "color_order": "GRAY",
        "dtype": "uint8",
        "value_range": [0, 255],
        "channel_conversion": "fixture_grayscale_boundary",
        "geometric_transform": "identity_frame",
        "interpolation": "not_applicable",
    }


def _mask_semantics() -> dict[str, object]:
    return {
        "foreground_polarity": "light",
        "representation": "uint8_0_255",
        "connectivity": "source_defined_binary_foreground",
        "validity_region": "full_input_frame",
        "border_policy": "prepared_input",
    }


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


def _fill_semantics_mask() -> np.ndarray:
    mask = np.zeros((10, 14), dtype=np.uint8)
    mask[0, 1:5] = 255
    mask[2, 8] = 255
    mask[3, 9] = 255
    mask[5, 1:3] = 255
    mask[6:8, 6] = 255
    mask[7, 7] = 255
    return mask


def _native_recipe(
    operation: str,
    identity: NativeMaskIdentity,
    values: dict[str, object],
) -> tuple[RecipeSpec, str]:
    recipe = build_recipe_catalogue().recipe("plantcv_seed_binary")
    stage = next(item for item in recipe.stages if item.operation_id == operation)
    updated_stage = replace(
        stage,
        parameters=tuple(
            replace(parameter, value=values[parameter.parameter_id])
            if parameter.parameter_id in values else parameter
            for parameter in stage.parameters
        ),
    )
    recipe = replace(
        recipe,
        provenance=identity.recipe_provenance(),
        stages=tuple(updated_stage if item.stage_id == stage.stage_id else item for item in recipe.stages),
    )
    return recipe, stage.stage_id


class NativeMaskIdentityTests(CatalogueTestCase):
    def test_loaded_implementation_identity_is_stable_and_rejects_replaced_code(self) -> None:
        first = native_stage_module._implementation_digest()
        self.assertRegex(first, r"^[0-9a-f]{64}$")
        self.assertEqual(first, native_stage_module._implementation_digest())
        with patch.object(native_stage_module, "_parameters", side_effect=lambda *args: {}):
            with self.assertRaisesRegex(NativeMaskStageError, "differs from freshly compiled source"):
                native_stage_module._implementation_digest()


@unittest.skipUnless(HAS_CUDA, "the native mask checkpoint bridge requires an available CUDA device")
class NativeMaskCudaStageTests(CatalogueTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.device = torch.device("cuda", torch.cuda.current_device())
        cls.implementation = native_mask_identity(cls.device)
        cls.catalogue = build_recipe_catalogue()
        cls.reference_runner = PlantCVReferenceRunner(REFERENCE_PYTHON) if HAS_REFERENCE_ENV else None

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="native-mask-stage-")
        self.root = Path(self.temporary.name)
        self.frame = CoordinateFrame(
            frame_id="native_mask_gray_fixture_v1",
            shape=(48, 72),
            origin="top_left_pixel_centres",
            axes="x_right_y_down",
            units="pixel",
            transform_sha256=sha256(b"native mask identity frame").hexdigest(),
        )
        self.source_sha256 = sha256(b"native mask synthetic engineering fixture").hexdigest()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _case(
        self,
        operation: str,
        parameters: dict[str, object],
        array: np.ndarray | None = None,
    ) -> dict[str, object]:
        recipe, stage_id = _native_recipe(operation, self.implementation, parameters)
        is_image = operation == "filter.median_blur"
        values = array if array is not None else (_fixture_image() if is_image else _fixture_mask())
        frame = replace(self.frame, shape=tuple(values.shape))
        port = "mask" if operation == "morphology.fill" else "gray"
        array_name = port
        role = (
            RasterRole("image", np.dtype(np.uint8).str, "HW", (0, 255))
            if is_image else RasterRole("mask", np.dtype(np.uint8).str, "HW")
        )
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
            image_semantics=_image_semantics(),
            mask_semantics=_mask_semantics(),
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
        base = self.root / operation.replace(".", "-")
        input_root = base / "input"
        output_root = base / "output"
        input_status = ExecutionStatus.EMPTY if not is_image and not np.any(values) else ExecutionStatus.AVAILABLE
        input_reference = publish_checkpoint(
            input_root,
            input_identity,
            {array_name: values},
            execution=ExecutionRecord(input_status),
            measurement=_measurement("prepared_input"),
            raster_roles={array_name: role},
            metadata={"engineering_fixture": True},
            allow_cpu_transfer=True,
        )
        output_identity = expected_native_mask_stage_identity(
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
            "expected_input_measurement": _measurement("prepared_input"),
            "expected_output_identity": output_identity,
            "expected_implementation_identity": self.implementation,
            "device": self.device,
            "input_bindings": {port: array_name},
            "input_roles": {port: role},
            "output_measurement": _measurement("native_mask_stage_output"),
            "allow_cpu_transfer": True,
            "input_array": values,
            "port": port,
        }

    @staticmethod
    def _run(case: dict[str, object]):
        args = {key: value for key, value in case.items() if key not in {"input_array", "port"}}
        return run_native_mask_stage(**args)





    def test_original_reference_fill_recipe_is_not_mislabelled_as_native(self) -> None:
        recipe = self.catalogue.recipe("plantcv_seed_binary")
        fill_stage = next(item for item in recipe.stages if item.operation_id == "morphology.fill")
        base = self._case("morphology.erode", {"ksize": 3, "iterations": 1})
        outcome = run_native_mask_stage(
            base["input_root"], base["input_checkpoint_sha256"], recipe, fill_stage.stage_id,
            base["output_root"], expected_source_sha256=base["expected_source_sha256"],
            expected_coordinate_frame=base["expected_coordinate_frame"],
            expected_recipe_identity=recipe.identity,
            expected_upstream_checkpoint_sha256=base["input_checkpoint_sha256"],
            expected_input_identity=base["expected_input_identity"],
            expected_input_measurement=base["expected_input_measurement"],
            expected_output_identity=None,
            expected_implementation_identity=self.implementation,
            device=self.device,
            input_bindings=base["input_bindings"], input_roles=base["input_roles"],
            output_measurement=base["output_measurement"], allow_cpu_transfer=True,
        )
        self.assertEqual(outcome.status, NativeMaskStageStatus.UNSUPPORTED)
        self.assertEqual(outcome.run_record.state, StageRunState.UNSUPPORTED)
        self.assertIn("native-compatible", outcome.run_record.reason)
        self.assertIsNone(outcome.checkpoint_reference)


if __name__ == "__main__":
    unittest.main()
