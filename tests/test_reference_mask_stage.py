from __future__ import annotations

from tests.catalogue_fixtures import CatalogueTestCase

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
class ReferenceMaskStageTests(CatalogueTestCase):
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
