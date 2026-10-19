from __future__ import annotations

from tests.functional_units import functional_units

from dataclasses import replace
import hashlib
from pathlib import Path
import re
import tempfile
import unittest

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
from seedvision.benchmarking.conformance import (
    CheckpointPairContract,
    ConformanceError,
    PrimitiveArrayContract,
    ThresholdBorderCorrespondence,
    ThresholdMaskSemanticCorrespondence,
    compare_instance_partitions,
    compare_masks,
    compare_primitive_outputs,
)
from seedvision.benchmarking.native_threshold_stage import (
    NativeThresholdIdentity,
    expected_native_threshold_stage_identity,
)
from seedvision.benchmarking.recipe_catalogue import build_recipe_catalogue
from seedvision.benchmarking.recipes import RecipeProvenanceMode
from seedvision.benchmarking.reference_runner import (
    PINNED_PLANTCV_COMMIT,
    PINNED_PLANTCV_VERSION,
    PINNED_REFERENCE_PLATFORM,
    PINNED_REFERENCE_PYTHON,
    PINNED_REQUIREMENTS_SHA256,
    PINNED_SOURCE_AUDIT_SHA256,
)
from seedvision.benchmarking.reference_stage import expected_reference_stage_identity


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _identity(
    backend: str,
    *,
    stage: str = "partition",
    source: str = "a" * 64,
    frame: CoordinateFrame | None = None,
    upstream: str | None = "b" * 64,
    mask_representation: str = "boolean",
) -> CheckpointIdentity:
    if frame is None:
        frame = CoordinateFrame(
            frame_id="source_pixels_v1",
            shape=(8, 12),
            origin="top_left_pixel_centres",
            axes="x_right_y_down",
            units="pixel",
            transform_sha256="c" * 64,
        )
    return CheckpointIdentity(
        source_sha256=source,
        coordinate_frame=frame,
        stage_id=stage,
        recipe_id=f"{backend}_recipe",
        recipe_version="1",
        recipe_sha256=_digest(f"recipe:{backend}"),
        graph_sha256=_digest(f"graph:{backend}"),
        settings_sha256=_digest(f"settings:{backend}"),
        backend_id=backend,
        backend_version="1.0",
        environment_versions={"python": "3.12.4", "numpy": "2.5.0", backend: "1.0"},
        image_semantics={
            "color_order": "BGR",
            "dtype": "uint8",
            "value_range": [0, 255],
            "channel_conversion": "none",
            "geometric_transform": "none",
            "interpolation": "none",
        },
        mask_semantics={
            "foreground_polarity": "nonzero_is_foreground",
            "representation": mask_representation,
            "connectivity": 8,
            "validity_region": "all_pixels",
            "border_policy": "constant_background",
        },
        instance_semantics={
            "label_dtype": "int32",
            "background_label": 0,
            "label_id_mapping": "arbitrary_positive_ids_with_zero_background",
            "split_boundary_ownership": "background",
            "visibility": "all_detected_instances",
            "eligibility": "all_detected_instances",
        },
        upstream_checkpoint_sha256=upstream,
    )


def _measurement(*, eligible: bool = True) -> MeasurementContract:
    return MeasurementContract(
        definition_id="seedfiddle.instance_area",
        definition_version="1",
        mask_type="instance_partition",
        units="pixel^2",
        calibration_status="uncalibrated",
        uncertainty="not_estimated",
        eligible=eligible,
        reason="" if eligible else "Reference output is unavailable.",
    )


@functional_units
class BenchmarkConformanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def publish(
        self,
        identity: CheckpointIdentity,
        array: np.ndarray | None,
        *,
        kind: str | None = None,
        name: str = "output",
        status: ExecutionStatus = ExecutionStatus.AVAILABLE,
    ):
        arrays = {} if array is None else {name: array}
        roles = {} if kind is None or array is None else {name: RasterRole(kind, array.dtype.str)}
        eligible = status not in {ExecutionStatus.FAILED, ExecutionStatus.UNSUPPORTED}
        return publish_checkpoint(
            self.root,
            identity,
            arrays,
            execution=ExecutionRecord(
                status,
                reason="reference stage failed" if status == ExecutionStatus.FAILED else (
                    "operation is unsupported" if status == ExecutionStatus.UNSUPPORTED else ""
                ),
            ),
            measurement=_measurement(eligible=eligible),
            raster_roles=roles,
            allow_cpu_transfer=True,
        )

    def pair(self, reference, candidate, reference_identity, candidate_identity, *, scope="stage"):
        return CheckpointPairContract(
            reference_checkpoint_sha256=reference.checkpoint_sha256,
            candidate_checkpoint_sha256=candidate.checkpoint_sha256,
            reference_identity=reference_identity,
            candidate_identity=candidate_identity,
            comparison_scope=scope,
        )

    def _threshold_pair(self, operation_id: str = "threshold.binary"):
        recipe_id = {
            "threshold.binary": "plantcv_seed_binary",
            "threshold.mean": "plantcv_seed_mean",
            "threshold.otsu": "plantcv_seed_otsu",
        }[operation_id]
        reference_recipe = build_recipe_catalogue().recipe(recipe_id)
        reference_stage = next(stage for stage in reference_recipe.stages if stage.operation_id == operation_id)
        selected = {
            "threshold.binary": {"threshold": 127, "object_type": "light"},
            "threshold.mean": {"ksize": 3, "offset": 1.25, "object_type": "light"},
            "threshold.otsu": {"object_type": "light"},
        }[operation_id]
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
        implementation = NativeThresholdIdentity(
            implementation_sha256=_digest("test native code"),
            python_version="3.12.4",
            torch_version="2.6.0+cu126",
            numpy_version=np.__version__,
            cuda_runtime_version="12.6",
            platform="win32",
            machine="engineering-test-device",
            device_index=0,
            device_name="engineering-test-device",
            compute_capability="8.6",
        )
        candidate_recipe = replace(reference_recipe, provenance=implementation.recipe_provenance())
        frame = CoordinateFrame(
            frame_id="threshold_input_frame",
            shape=(8, 12),
            origin="top_left_pixel_centres",
            axes="x_right_y_down",
            units="pixel",
            transform_sha256=_digest("threshold identity frame"),
        )
        input_identity = replace(
            _identity("prepared_gray", stage="prepared_gray"),
            source_sha256=_digest("threshold source"),
            coordinate_frame=frame,
            image_semantics={
                "color_order": "GRAY", "dtype": "uint8", "value_range": [0, 255],
                "channel_conversion": "caller_supplied_gray", "geometric_transform": "identity",
                "interpolation": "not_applicable",
            },
            upstream_checkpoint_sha256=_digest("threshold upstream"),
        )
        gray = np.arange(96, dtype=np.uint8).reshape(8, 12)
        input_measurement = _measurement(eligible=False)
        input_reference = publish_checkpoint(
            self.root,
            input_identity,
            {"gray": gray},
            execution=ExecutionRecord(ExecutionStatus.AVAILABLE),
            measurement=input_measurement,
            raster_roles={"gray": RasterRole("image", gray.dtype.str, "HW", (0, 255))},
            allow_cpu_transfer=True,
        )
        output_measurement = _measurement(eligible=False)
        reference_identity = expected_reference_stage_identity(
            input_identity, reference_recipe, reference_stage.stage_id, input_reference.checkpoint_sha256,
        )
        candidate_stage = next(stage for stage in candidate_recipe.stages if stage.operation_id == operation_id)
        candidate_identity = expected_native_threshold_stage_identity(
            input_identity, candidate_recipe, candidate_stage.stage_id,
            input_reference.checkpoint_sha256, implementation,
        )
        mask = np.zeros((8, 12), dtype=np.uint8)
        mask[2:6, 3:8] = 255
        role_record = {"mask": RasterRole("mask", mask.dtype.str).to_dict()}
        input_role_record = {"gray": RasterRole("image", gray.dtype.str, "HW", (0, 255)).to_dict()}
        shared_metadata = {
            "stage_id": reference_stage.stage_id,
            "operation_id": operation_id,
            "source_sha256": input_identity.source_sha256,
            "coordinate_frame": frame.to_dict(),
            "upstream_checkpoint_sha256": input_reference.checkpoint_sha256,
            "input_checkpoint_identity": input_identity.to_dict(),
            "input_bindings": {"gray": "gray"},
            "input_raster_roles": input_role_record,
            "output_raster_roles": role_record,
            "measurement_contract": output_measurement.to_dict(),
            "output_inventory": [{"name": "mask", "dtype": mask.dtype.str, "shape": [8, 12], "nbytes": mask.nbytes}],
        }
        reference_metadata = {
            **shared_metadata,
            "format": "seedfiddle.reference-stage-output",
            "version": 1,
            "recipe_identity": reference_recipe.identity.to_dict(),
            "source_mapping": reference_stage.source_mappings[0].to_dict(),
            "parameters": selected,
            "runner_identity": {
                "reference": "plantcv",
                "version": PINNED_PLANTCV_VERSION,
                "commit": PINNED_PLANTCV_COMMIT,
                "python": PINNED_REFERENCE_PYTHON,
                "platform": PINNED_REFERENCE_PLATFORM,
                "source_audit_sha256": PINNED_SOURCE_AUDIT_SHA256,
                "requirements_lock_sha256": PINNED_REQUIREMENTS_SHA256,
                "identity_digest": _digest("verified test runner identity"),
                "actual_distributions": {
                    re.sub(r"[-_.]+", "-", name).lower(): version
                    for name, version in reference_recipe.provenance.environment_versions.items()
                },
            },
            "scientific_validation_status": "not_certified",
        }
        candidate_metadata = {
            **shared_metadata,
            "format": "seedfiddle.native-threshold-stage-output",
            "version": 1,
            "execution_mode": RecipeProvenanceMode.NATIVE_COMPATIBLE.value,
            "recipe_identity": candidate_recipe.identity.to_dict(),
            "source_mapping": candidate_stage.source_mappings[0].to_dict(),
            "selected_parameters": selected,
            "implementation_identity": implementation.to_dict(),
            "output_inventory": [{
                "name": "mask", "dtype": mask.dtype.str, "shape": [8, 12],
                "nbytes": mask.nbytes, "device": implementation.device,
            }],
            "native_supported_domain": {
                "threshold.binary": "uint8 cutoff in [0,255]; light uses > and dark uses <=",
                "threshold.mean": "odd ksize [3,255], finite offset [-255,255], replicated borders, pinned float32 rounding",
                "threshold.otsu": "unique separated eligible histogram partition; ambiguous histograms are unsupported",
            }[operation_id],
            "reference_comparison_status": "not_evaluated",
            "scientific_validation_status": "not_certified",
        }
        reference_output = publish_checkpoint(
            self.root,
            reference_identity,
            {"mask": mask},
            execution=ExecutionRecord(ExecutionStatus.AVAILABLE),
            measurement=output_measurement,
            raster_roles={"mask": RasterRole("mask", mask.dtype.str)},
            metadata=reference_metadata,
            allow_cpu_transfer=True,
        )
        candidate_output = publish_checkpoint(
            self.root,
            candidate_identity,
            {"mask": mask.copy()},
            execution=ExecutionRecord(ExecutionStatus.AVAILABLE),
            measurement=output_measurement,
            raster_roles={"mask": RasterRole("mask", mask.dtype.str)},
            metadata=candidate_metadata,
            allow_cpu_transfer=True,
        )
        border = (
            ThresholdBorderCorrespondence.REPLICATED_INPUT
            if operation_id == "threshold.mean"
            else ThresholdBorderCorrespondence.NOT_APPLICABLE_POINTWISE
        )
        adapter = ThresholdMaskSemanticCorrespondence(
            operation_id=operation_id,
            reference_recipe=reference_recipe,
            candidate_recipe=candidate_recipe,
            input_identity=input_identity,
            input_checkpoint_sha256=input_reference.checkpoint_sha256,
            reference_identity=reference_identity,
            candidate_identity=candidate_identity,
            native_implementation_identity=implementation,
            input_array_name="gray",
            input_role=RasterRole("image", gray.dtype.str, "HW", (0, 255)),
            output_measurement=output_measurement,
            border_correspondence=border,
        )
        pair = self.pair(reference_output, candidate_output, reference_identity, candidate_identity)
        return pair, adapter, reference_identity, candidate_identity, reference_output, candidate_output

    def check_test_threshold_border_correspondence_is_explicit_and_exact(self) -> None:
        pair, adapter, reference_identity, candidate_identity, _reference, _candidate = self._threshold_pair()
        with self.assertRaisesRegex(ConformanceError, "Mask representation semantics differ"):
            compare_masks(self.root, pair, reference_array="mask", candidate_array="mask")
        report = compare_masks(
            self.root,
            pair,
            reference_array="mask",
            candidate_array="mask",
            semantic_correspondence=adapter,
        )
        self.assertTrue(report["conformance"]["passed"])
        self.assertEqual(
            report["mask_semantics_correspondence"]["border_policy_correspondence"]["common_contract"],
            "not_applicable_pointwise",
        )
        self.assertEqual(report["scientific_metrics"]["status"], "not_evaluated")

        with self.assertRaisesRegex(ValueError, "supports binary, mean, or Otsu"):
            replace(adapter, operation_id="threshold.watershed")
        with self.assertRaisesRegex(ValueError, "Border correspondence"):
            replace(adapter, border_correspondence=ThresholdBorderCorrespondence.REPLICATED_INPUT)
        forged_identity = replace(candidate_identity, mask_semantics={**dict(candidate_identity.mask_semantics), "border_policy": "anything"})
        malformed_adapter = object.__new__(ThresholdMaskSemanticCorrespondence)
        for name in adapter.__dataclass_fields__:
            object.__setattr__(malformed_adapter, name, getattr(adapter, name))
        object.__setattr__(malformed_adapter, "candidate_identity", forged_identity)
        with self.assertRaisesRegex(ConformanceError, "exact expected and loaded identities"):
            compare_masks(
                self.root,
                pair,
                reference_array="mask",
                candidate_array="mask",
                semantic_correspondence=malformed_adapter,
            )





    def check_test_primitive_path_rejects_raster_representation_semantic_mismatch(self) -> None:
        reference_identity = _identity("plantcv_reference", stage="foreground_mask")
        candidate_identity = replace(
            _identity("seedfiddle_native", stage="foreground_mask"),
            mask_semantics={
                **dict(_identity("seedfiddle_native", stage="foreground_mask").mask_semantics),
                "connectivity": 4,
            },
        )
        mask = np.zeros((8, 12), dtype=np.bool_)
        reference = self.publish(reference_identity, mask, kind="mask")
        candidate = self.publish(candidate_identity, mask.copy(), kind="mask")
        with self.assertRaisesRegex(ConformanceError, "Mask raster representation semantics"):
            compare_primitive_outputs(
                self.root,
                self.pair(reference, candidate, reference_identity, candidate_identity),
                reference_array="output",
                candidate_array="output",
                contract=PrimitiveArrayContract(mask.dtype.str, mask.shape),
            )



    def check_test_instance_label_representation_mismatch_is_rejected(self) -> None:
        reference_identity = _identity("plantcv_reference", stage="instance_partition")
        candidate_identity = replace(
            _identity("seedfiddle_native", stage="instance_partition"),
            instance_semantics={
                **dict(_identity("seedfiddle_native", stage="instance_partition").instance_semantics),
                "label_dtype": "uint16",
            },
        )
        labels = np.zeros((8, 12), dtype=np.int32)
        labels[2:5, 3:7] = 5
        candidate_labels = labels.astype(np.uint16)
        reference = self.publish(reference_identity, labels, kind="instances")
        candidate = self.publish(candidate_identity, candidate_labels, kind="instances")
        with self.assertRaisesRegex(ConformanceError, "role dtype"):
            compare_instance_partitions(
                self.root,
                self.pair(reference, candidate, reference_identity, candidate_identity),
                reference_array="output",
                candidate_array="output",
            )







    def test_workflow_lifecycle(self):
        """1 related BenchmarkConformanceTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_threshold_border_correspondence_is_explicit_and_exact)

    def test_validation_and_recovery(self):
        """2 related BenchmarkConformanceTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_primitive_path_rejects_raster_representation_semantic_mismatch)
        self.scenario(self.check_test_instance_label_representation_mismatch_is_rejected)


if __name__ == "__main__":
    unittest.main()
