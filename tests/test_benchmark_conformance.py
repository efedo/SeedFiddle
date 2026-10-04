from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
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
    publish_checkpoint,
)
from seedvision.benchmarking.conformance import (
    CheckpointPairContract,
    ConformanceError,
    PrimitiveArrayContract,
    compare_instance_partitions,
    compare_masks,
    compare_primitive_outputs,
)


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
