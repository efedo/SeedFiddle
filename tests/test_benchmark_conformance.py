from __future__ import annotations

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

    def test_threshold_border_correspondence_is_explicit_and_exact(self) -> None:
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

    def test_threshold_correspondence_rejects_parameter_type_and_nonborder_semantic_changes(self) -> None:
        pair, adapter, reference_identity, candidate_identity, reference, candidate = self._threshold_pair("threshold.mean")
        changed_stage = next(stage for stage in adapter.candidate_recipe.stages if stage.operation_id == "threshold.mean")
        changed_stage = replace(
            changed_stage,
            parameters=tuple(
                replace(parameter, value=1) if parameter.parameter_id == "offset" else parameter
                for parameter in changed_stage.parameters
            ),
        )
        changed_recipe = replace(
            adapter.candidate_recipe,
            stages=tuple(changed_stage if stage.stage_id == changed_stage.stage_id else stage for stage in adapter.candidate_recipe.stages),
        )
        with self.assertRaisesRegex(ValueError, "selected parameters must match exactly"):
            replace(adapter, candidate_recipe=changed_recipe)

        changed_semantics = dict(candidate_identity.mask_semantics)
        changed_semantics["foreground_polarity"] = "dark"
        corrupted_identity = replace(candidate_identity, mask_semantics=changed_semantics)
        candidate_checkpoint = load_checkpoint(self.root, candidate.checkpoint_sha256)
        corrupted_output = publish_checkpoint(
            self.root,
            corrupted_identity,
            {"mask": candidate_checkpoint.arrays["mask"]},
            execution=ExecutionRecord(ExecutionStatus.AVAILABLE),
            measurement=candidate_checkpoint.measurement,
            raster_roles={"mask": candidate_checkpoint.raster_roles["mask"]},
            metadata=dict(candidate_checkpoint.metadata),
            allow_cpu_transfer=True,
        )
        corrupted_pair = self.pair(reference, corrupted_output, reference_identity, corrupted_identity)
        with self.assertRaisesRegex(ValueError, "exact bridge-derived identities"):
            replace(adapter, candidate_identity=corrupted_identity)
        with self.assertRaisesRegex(ConformanceError, "exact expected and loaded identities"):
            compare_masks(
                self.root,
                corrupted_pair,
                reference_array="mask",
                candidate_array="mask",
                semantic_correspondence=adapter,
            )

    def test_primitive_arrays_obey_explicit_shape_dtype_range_and_tolerance(self) -> None:
        reference_identity = _identity("plantcv_reference", stage="distance_map")
        candidate_identity = _identity("seedfiddle_native", stage="distance_map")
        reference = self.publish(reference_identity, np.array([0.25, 0.5, 0.0], dtype=np.float32))
        candidate = self.publish(candidate_identity, np.array([0.25, 0.509, 0.0], dtype=np.float32))
        pair = self.pair(reference, candidate, reference_identity, candidate_identity)
        contract = PrimitiveArrayContract(np.dtype(np.float32).str, (3,), (0.0, 1.0))

        report = compare_primitive_outputs(
            self.root,
            pair,
            reference_array="output",
            candidate_array="output",
            contract=contract,
            atol=0.01,
        )

        self.assertTrue(report["conformance"]["passed"])
        self.assertEqual(report["conformance"]["mismatch_count"], 0)
        self.assertEqual(report["checkpoint_identities"]["reference"]["backend"]["id"], "plantcv_reference")
        self.assertEqual(report["checkpoint_identities"]["candidate"]["environment_versions"]["numpy"], "2.5.0")
        self.assertEqual(report["scientific_metrics"]["status"], "not_evaluated")
        self.assertFalse(report["scientific_metrics"]["independent_truth"])

        outside_tolerance = self.publish(
            candidate_identity,
            np.array([0.25, 0.7, 0.0], dtype=np.float32),
        )
        failed = compare_primitive_outputs(
            self.root,
            self.pair(reference, outside_tolerance, reference_identity, candidate_identity),
            reference_array="output",
            candidate_array="output",
            contract=contract,
            atol=0.01,
        )
        self.assertFalse(failed["conformance"]["passed"])
        self.assertEqual(failed["conformance"]["mismatch_count"], 1)

        wrong_shape = self.publish(candidate_identity, np.array([0.25, 0.5], dtype=np.float32))
        with self.assertRaisesRegex(ConformanceError, "shape"):
            compare_primitive_outputs(
                self.root,
                self.pair(reference, wrong_shape, reference_identity, candidate_identity),
                reference_array="output",
                candidate_array="output",
                contract=contract,
            )

        large_integer_reference = self.publish(reference_identity, np.array([2**53], dtype=np.int64))
        large_integer_candidate = self.publish(candidate_identity, np.array([2**53 + 1], dtype=np.int64))
        integer_contract = PrimitiveArrayContract(np.dtype(np.int64).str, (1,))
        integer_report = compare_primitive_outputs(
            self.root,
            self.pair(large_integer_reference, large_integer_candidate, reference_identity, candidate_identity),
            reference_array="output",
            candidate_array="output",
            contract=integer_contract,
        )
        self.assertFalse(integer_report["conformance"]["passed"])
        self.assertEqual(integer_report["conformance"]["max_absolute_error"], 1)
        with self.assertRaisesRegex(ValueError, "only to floating-point"):
            compare_primitive_outputs(
                self.root,
                self.pair(large_integer_reference, large_integer_candidate, reference_identity, candidate_identity),
                reference_array="output",
                candidate_array="output",
                contract=integer_contract,
                atol=1.0,
            )

    def test_primitive_integer_chunks_and_ranges_preserve_exact_values(self) -> None:
        reference_identity = _identity("plantcv_reference", stage="primitive_values")
        candidate_identity = _identity("seedfiddle_native", stage="primitive_values")

        for dtype in (np.bool_, np.int32, np.uint64):
            with self.subTest(dtype=np.dtype(dtype).name):
                if dtype is np.bool_:
                    values = np.array([[True, False], [True, True]], dtype=dtype)
                else:
                    values = np.array([[1, 0], [7, 11]], dtype=dtype)
                reference = self.publish(reference_identity, values)
                candidate = self.publish(candidate_identity, values.copy())
                report = compare_primitive_outputs(
                    self.root,
                    self.pair(reference, candidate, reference_identity, candidate_identity),
                    reference_array="output",
                    candidate_array="output",
                    contract=PrimitiveArrayContract(values.dtype.str, values.shape),
                )
                self.assertTrue(report["conformance"]["passed"])

        # Cross the public comparator's one-million-element chunk boundary so
        # integer diagnostics cannot accidentally index the original array.
        values = np.ones((1_000_001, 1), dtype=np.uint64)
        reference = self.publish(reference_identity, values)
        candidate = self.publish(candidate_identity, values.copy())
        chunked = compare_primitive_outputs(
            self.root,
            self.pair(reference, candidate, reference_identity, candidate_identity),
            reference_array="output",
            candidate_array="output",
            contract=PrimitiveArrayContract(values.dtype.str, values.shape),
        )
        self.assertTrue(chunked["conformance"]["passed"])
        self.assertEqual(chunked["conformance"]["element_count"], values.size)

        maximum = 2**53
        exact_contract = PrimitiveArrayContract(np.dtype(np.int64).str, (1,), (maximum, maximum))
        self.assertEqual(exact_contract.value_range, (maximum, maximum))
        valid = self.publish(reference_identity, np.array([maximum], dtype=np.int64))
        outside = self.publish(candidate_identity, np.array([maximum + 1], dtype=np.int64))
        with self.assertRaisesRegex(ConformanceError, "declared value range"):
            compare_primitive_outputs(
                self.root,
                self.pair(valid, outside, reference_identity, candidate_identity),
                reference_array="output",
                candidate_array="output",
                contract=exact_contract,
            )

        integer_extremes = (
            (
                np.dtype(np.int64),
                np.iinfo(np.int64).min,
                np.iinfo(np.int64).max,
                2**64 - 1,
            ),
            (
                np.dtype(np.uint64),
                0,
                np.iinfo(np.uint64).max,
                2**64 - 1,
            ),
        )
        for dtype, reference_value, candidate_value, exact_error in integer_extremes:
            with self.subTest(dtype=dtype.name):
                reference = self.publish(reference_identity, np.array([reference_value], dtype=dtype))
                candidate = self.publish(candidate_identity, np.array([candidate_value], dtype=dtype))
                report = compare_primitive_outputs(
                    self.root,
                    self.pair(reference, candidate, reference_identity, candidate_identity),
                    reference_array="output",
                    candidate_array="output",
                    contract=PrimitiveArrayContract(dtype.str, (1,)),
                )
                self.assertFalse(report["conformance"]["passed"])
                self.assertEqual(report["conformance"]["max_absolute_error"], exact_error)

    def test_float_tolerance_is_reference_relative_and_overflow_safe(self) -> None:
        reference_identity = _identity("plantcv_reference", stage="float_values")
        candidate_identity = _identity("seedfiddle_native", stage="float_values")
        contract = PrimitiveArrayContract(np.dtype(np.float64).str, (1,))

        reference = self.publish(reference_identity, np.array([100.0], dtype=np.float64))
        candidate = self.publish(candidate_identity, np.array([111.0], dtype=np.float64))
        report = compare_primitive_outputs(
            self.root,
            self.pair(reference, candidate, reference_identity, candidate_identity),
            reference_array="output",
            candidate_array="output",
            contract=contract,
            rtol=0.1,
        )
        self.assertFalse(report["conformance"]["passed"])
        self.assertEqual(report["conformance"]["mismatch_count"], 1)

        reference = self.publish(reference_identity, np.array([100.0], dtype=np.float64))
        candidate = self.publish(candidate_identity, np.array([110.0], dtype=np.float64))
        boundary = compare_primitive_outputs(
            self.root,
            self.pair(reference, candidate, reference_identity, candidate_identity),
            reference_array="output",
            candidate_array="output",
            contract=contract,
            rtol=0.1,
        )
        self.assertTrue(boundary["conformance"]["passed"])
        self.assertEqual(boundary["conformance"]["max_relative_error_for_nonzero_reference"], 0.1)

        just_outside = self.publish(
            candidate_identity,
            np.array([np.nextafter(110.0, np.inf)], dtype=np.float64),
        )
        outside_boundary = compare_primitive_outputs(
            self.root,
            self.pair(reference, just_outside, reference_identity, candidate_identity),
            reference_array="output",
            candidate_array="output",
            contract=contract,
            rtol=0.1,
        )
        self.assertFalse(outside_boundary["conformance"]["passed"])

        reference = self.publish(reference_identity, np.array([111.0], dtype=np.float64))
        candidate = self.publish(candidate_identity, np.array([100.0], dtype=np.float64))
        reverse = compare_primitive_outputs(
            self.root,
            self.pair(reference, candidate, reference_identity, candidate_identity),
            reference_array="output",
            candidate_array="output",
            contract=contract,
            rtol=0.1,
        )
        self.assertTrue(reverse["conformance"]["passed"])

        for dtype, magnitude in ((np.float32, 3e38), (np.float64, 1.7e308)):
            with self.subTest(dtype=np.dtype(dtype).name):
                reference = self.publish(reference_identity, np.array([-magnitude], dtype=dtype))
                candidate = self.publish(candidate_identity, np.array([magnitude], dtype=dtype))
                extreme = compare_primitive_outputs(
                    self.root,
                    self.pair(reference, candidate, reference_identity, candidate_identity),
                    reference_array="output",
                    candidate_array="output",
                    contract=PrimitiveArrayContract(np.dtype(dtype).str, (1,)),
                    rtol=1.2,
                )
                self.assertFalse(extreme["conformance"]["passed"])
                self.assertEqual(extreme["conformance"]["mismatch_count"], 1)

    def test_primitive_path_rejects_raster_representation_semantic_mismatch(self) -> None:
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

    def test_masks_are_exact_and_status_is_part_of_conformance(self) -> None:
        reference_identity = _identity("plantcv_reference", stage="foreground_mask")
        candidate_identity = _identity("seedfiddle_native", stage="foreground_mask")
        reference_mask = np.zeros((8, 12), dtype=np.bool_)
        reference_mask[2:6, 3:8] = True
        candidate_mask = reference_mask.copy()
        candidate_mask[5, 7] = False
        reference = self.publish(reference_identity, reference_mask, kind="mask")
        candidate = self.publish(candidate_identity, candidate_mask, kind="mask")
        report = compare_masks(
            self.root,
            self.pair(reference, candidate, reference_identity, candidate_identity),
            reference_array="output",
            candidate_array="output",
        )
        self.assertFalse(report["conformance"]["passed"])
        self.assertEqual(report["conformance"]["mismatch_pixel_count"], 1)

        blank = np.zeros((8, 12), dtype=np.bool_)
        empty_candidate = self.publish(
            candidate_identity,
            blank,
            kind="mask",
            status=ExecutionStatus.EMPTY,
        )
        empty_reference = self.publish(
            reference_identity,
            blank,
            kind="mask",
            status=ExecutionStatus.EMPTY,
        )
        empty_report = compare_masks(
            self.root,
            self.pair(empty_reference, empty_candidate, reference_identity, candidate_identity),
            reference_array="output",
            candidate_array="output",
        )
        self.assertTrue(empty_report["conformance"]["passed"])
        self.assertEqual(empty_report["conformance"]["reference_execution_status"], "empty")

        malformed_empty = self.publish(
            candidate_identity,
            reference_mask,
            kind="mask",
            status=ExecutionStatus.EMPTY,
        )
        with self.assertRaisesRegex(ConformanceError, "marked empty"):
            compare_masks(
                self.root,
                self.pair(empty_reference, malformed_empty, reference_identity, candidate_identity),
                reference_array="output",
                candidate_array="output",
            )

        available_blank = self.publish(candidate_identity, blank, kind="mask")
        status_mismatch = compare_masks(
            self.root,
            self.pair(empty_reference, available_blank, reference_identity, candidate_identity),
            reference_array="output",
            candidate_array="output",
        )
        self.assertFalse(status_mismatch["conformance"]["passed"])
        self.assertFalse(status_mismatch["conformance"]["execution_status_match"])

    def test_mask_representation_mismatch_is_rejected(self) -> None:
        reference_identity = _identity("plantcv_reference", stage="foreground_mask")
        candidate_identity = _identity(
            "seedfiddle_native",
            stage="foreground_mask",
            mask_representation="uint8_0_1",
        )
        reference = self.publish(reference_identity, np.zeros((8, 12), dtype=np.bool_), kind="mask")
        candidate = self.publish(candidate_identity, np.zeros((8, 12), dtype=np.uint8), kind="mask")
        with self.assertRaisesRegex(ConformanceError, "representation"):
            compare_masks(
                self.root,
                self.pair(reference, candidate, reference_identity, candidate_identity),
                reference_array="output",
                candidate_array="output",
            )

    def test_instance_label_representation_mismatch_is_rejected(self) -> None:
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

    def test_instance_label_permutation_is_equivalent_and_auditable(self) -> None:
        reference_identity = _identity("plantcv_reference", stage="instance_partition")
        candidate_identity = _identity("seedfiddle_native", stage="instance_partition")
        truth = np.zeros((8, 12), dtype=np.int32)
        truth[1:4, 1:4] = 17
        truth[4:7, 7:10] = 91
        candidate = np.zeros_like(truth)
        candidate[1:4, 1:4] = 900
        candidate[4:7, 7:10] = 2
        reference_checkpoint = self.publish(reference_identity, truth, kind="instances")
        candidate_checkpoint = self.publish(candidate_identity, candidate, kind="instances")

        report = compare_instance_partitions(
            self.root,
            self.pair(reference_checkpoint, candidate_checkpoint, reference_identity, candidate_identity),
            reference_array="output",
            candidate_array="output",
            max_matched_id_records=1,
        )

        comparison = report["conformance"]
        self.assertTrue(comparison["passed"])
        self.assertTrue(comparison["partition_equivalent"])
        self.assertTrue(comparison["label_ids_ignored"])
        metrics = comparison["partition_agreement_metrics"]
        self.assertEqual(metrics["true_instances"], 2)
        self.assertEqual(metrics["predicted_instances"], 2)
        self.assertEqual(metrics["true_positives"], 2)
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertEqual(metrics["f1"], 1.0)
        self.assertEqual(metrics["panoptic_quality"], 1.0)
        self.assertEqual(metrics["mean_matched_iou"], 1.0)
        self.assertEqual(metrics["boundary_f1"], 1.0)
        audit = comparison["matched_id_audit"]
        self.assertEqual(audit["matched_count"], 2)
        self.assertEqual(len(audit["records"]), 1)
        self.assertTrue(audit["records_truncated"])
        self.assertEqual(
            audit["records"][0],
            {"reference_instance_id": 17, "candidate_instance_id": 900, "iou": 1.0},
        )

    def test_empty_instance_pair_is_valid_and_failed_or_unsupported_is_rejected(self) -> None:
        reference_identity = _identity("plantcv_reference", stage="instance_partition")
        candidate_identity = _identity("seedfiddle_native", stage="instance_partition")
        empty = np.zeros((8, 12), dtype=np.int32)
        reference = self.publish(reference_identity, empty, kind="instances", status=ExecutionStatus.EMPTY)
        candidate = self.publish(candidate_identity, empty, kind="instances", status=ExecutionStatus.EMPTY)

        report = compare_instance_partitions(
            self.root,
            self.pair(reference, candidate, reference_identity, candidate_identity),
            reference_array="output",
            candidate_array="output",
        )
        self.assertTrue(report["conformance"]["passed"])
        self.assertEqual(report["conformance"]["reference_execution_status"], "empty")
        self.assertEqual(report["conformance"]["partition_agreement_metrics"]["true_instances"], 0)
        self.assertEqual(report["conformance"]["partition_agreement_metrics"]["predicted_instances"], 0)
        self.assertEqual(report["conformance"]["partition_agreement_metrics"]["f1"], 1.0)
        self.assertIsNone(report["conformance"]["partition_agreement_metrics"]["mean_matched_iou"])

        for unavailable in (ExecutionStatus.FAILED, ExecutionStatus.UNSUPPORTED):
            with self.subTest(status=unavailable):
                unavailable_checkpoint = self.publish(
                    candidate_identity,
                    None,
                    kind="instances",
                    status=unavailable,
                )
                pair = self.pair(reference, unavailable_checkpoint, reference_identity, candidate_identity)
                with self.assertRaisesRegex(ConformanceError, "cannot be compared"):
                    compare_instance_partitions(
                        self.root,
                        pair,
                        reference_array="output",
                        candidate_array="output",
                    )

    def test_split_and_merge_statistics_reuse_the_existing_evaluator(self) -> None:
        reference_identity = _identity("plantcv_reference", stage="instance_partition")
        candidate_identity = _identity("seedfiddle_native", stage="instance_partition")

        two_objects = np.zeros((8, 12), dtype=np.int32)
        two_objects[2:4, 1:3] = 10
        two_objects[2:4, 4:6] = 20
        merged = np.zeros_like(two_objects)
        merged[2:4, 1:3] = 99
        merged[2:4, 4:6] = 99
        ref_merge = self.publish(reference_identity, two_objects, kind="instances")
        cand_merge = self.publish(candidate_identity, merged, kind="instances")
        merge_report = compare_instance_partitions(
            self.root,
            self.pair(ref_merge, cand_merge, reference_identity, candidate_identity),
            reference_array="output",
            candidate_array="output",
        )
        merge = merge_report["conformance"]["partition_agreement_metrics"]
        self.assertFalse(merge_report["conformance"]["partition_equivalent"])
        self.assertEqual((merge["true_instances"], merge["predicted_instances"]), (2, 1))
        self.assertEqual((merge["true_positives"], merge["false_positives"], merge["false_negatives"]), (1, 0, 1))
        self.assertEqual(merge["precision"], 1.0)
        self.assertEqual(merge["recall"], 0.5)
        self.assertAlmostEqual(merge["f1"], 2 / 3)
        self.assertEqual(merge["merged_instances"], 1)

        one_object = np.zeros((8, 12), dtype=np.int32)
        one_object[2:4, 1:5] = 10
        split = np.zeros_like(one_object)
        split[2:4, 1:3] = 8
        split[2:4, 3:5] = 9
        ref_split = self.publish(reference_identity, one_object, kind="instances")
        cand_split = self.publish(candidate_identity, split, kind="instances")
        split_report = compare_instance_partitions(
            self.root,
            self.pair(ref_split, cand_split, reference_identity, candidate_identity),
            reference_array="output",
            candidate_array="output",
        )
        split_metrics = split_report["conformance"]["partition_agreement_metrics"]
        self.assertFalse(split_report["conformance"]["partition_equivalent"])
        self.assertEqual((split_metrics["true_instances"], split_metrics["predicted_instances"]), (1, 2))
        self.assertEqual((split_metrics["true_positives"], split_metrics["false_positives"], split_metrics["false_negatives"]), (1, 1, 0))
        self.assertEqual(split_metrics["precision"], 0.5)
        self.assertEqual(split_metrics["recall"], 1.0)
        self.assertAlmostEqual(split_metrics["f1"], 2 / 3)
        self.assertEqual(split_metrics["split_instances"], 1)

    def test_exact_identity_rejects_stale_recipe_environment_source_frame_and_upstream(self) -> None:
        reference_identity = _identity("plantcv_reference", stage="distance_map")
        candidate_identity = _identity("seedfiddle_native", stage="distance_map")
        reference = self.publish(reference_identity, np.ones((2,), dtype=np.float32))
        candidate = self.publish(candidate_identity, np.ones((2,), dtype=np.float32))

        stale_variants = (
            replace(candidate_identity, recipe_sha256=_digest("stale recipe")),
            replace(candidate_identity, environment_versions={"python": "3.12.4", "numpy": "different"}),
            replace(candidate_identity, backend_version="9.9"),
            replace(candidate_identity, graph_sha256=_digest("stale graph")),
            replace(candidate_identity, settings_sha256=_digest("stale settings")),
            replace(candidate_identity, stage_id="other_stage"),
        )
        for stale in stale_variants:
            with self.subTest(identity=stale.to_dict()["recipe_sha256"]):
                pair = self.pair(reference, candidate, reference_identity, stale)
                with self.assertRaisesRegex(ValueError, "exact stage adapter"):
                    compare_primitive_outputs(
                        self.root,
                        pair,
                        reference_array="output",
                        candidate_array="output",
                        contract=PrimitiveArrayContract(np.dtype(np.float32).str, (2,)),
                    )

        changed_source = replace(candidate_identity, source_sha256="d" * 64)
        with self.assertRaisesRegex(ValueError, "same source"):
            self.pair(reference, candidate, reference_identity, changed_source)
        changed_frame = replace(
            candidate_identity,
            coordinate_frame=replace(candidate_identity.coordinate_frame, frame_id="other_frame"),
        )
        with self.assertRaisesRegex(ValueError, "same coordinate frame"):
            self.pair(reference, candidate, reference_identity, changed_frame)
        changed_upstream = replace(candidate_identity, upstream_checkpoint_sha256="e" * 64)
        with self.assertRaisesRegex(ValueError, "same upstream"):
            self.pair(reference, candidate, reference_identity, changed_upstream)

    def test_pipeline_scope_binds_but_does_not_require_equal_upstream_checkpoints(self) -> None:
        reference_identity = _identity("plantcv_reference", stage="final_mask", upstream="d" * 64)
        candidate_identity = _identity("seedfiddle_native", stage="final_mask", upstream="e" * 64)
        mask = np.zeros((8, 12), dtype=np.bool_)
        reference = self.publish(reference_identity, mask, kind="mask", status=ExecutionStatus.EMPTY)
        candidate = self.publish(candidate_identity, mask, kind="mask", status=ExecutionStatus.EMPTY)
        with self.assertRaisesRegex(ValueError, "same upstream"):
            self.pair(reference, candidate, reference_identity, candidate_identity)

        report = compare_masks(
            self.root,
            self.pair(reference, candidate, reference_identity, candidate_identity, scope="pipeline"),
            reference_array="output",
            candidate_array="output",
        )
        self.assertTrue(report["conformance"]["passed"])
        self.assertIsNone(report["input_binding"]["upstream_checkpoint_sha256"])
        self.assertEqual(report["input_binding"]["reference_upstream_checkpoint_sha256"], "d" * 64)
        self.assertEqual(report["input_binding"]["candidate_upstream_checkpoint_sha256"], "e" * 64)

    def test_corrupt_checkpoint_content_is_rejected_before_comparison(self) -> None:
        reference_identity = _identity("plantcv_reference", stage="foreground_mask")
        candidate_identity = _identity("seedfiddle_native", stage="foreground_mask")
        blank = np.zeros((8, 12), dtype=np.bool_)
        reference = self.publish(reference_identity, blank, kind="mask")
        candidate = self.publish(candidate_identity, blank, kind="mask")
        (candidate.path / "arrays.npz").write_bytes(b"tampered checkpoint content")

        with self.assertRaisesRegex(ConformanceError, "could not be verified"):
            compare_masks(
                self.root,
                self.pair(reference, candidate, reference_identity, candidate_identity),
                reference_array="output",
                candidate_array="output",
            )


if __name__ == "__main__":
    unittest.main()
