"""Focused tests for explicit CUDA per-seed crop artifacts."""

from __future__ import annotations

from dataclasses import fields, replace
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from seedvision.calibration.image import ImageCalibration
from seedvision.export.crops import extract_seed_crops
from seedvision.learning.contracts import LearnedInstanceResult, ModelFamily
from seedvision.segmentation.baseline import BaselineAnalysis
from seedvision.segmentation.procedural import ProceduralInstanceResult


def _revision(character):
    return character * 64


def _analysis(labels, source_shape, *, crop_offset=(2, 1), frame_shape=(8, 12),
              method="unet", device="cuda"):
    labels = np.asarray(labels, dtype=np.int32).copy()
    height, width = frame_shape
    rows = torch.arange(height, dtype=torch.float32, device=device)[None, None, :, None]
    cols = torch.arange(width, dtype=torch.float32, device=device)[None, None, None, :]
    tensor = torch.empty((1, 3, height, width), dtype=torch.float32, device=device)
    tensor[:, 0] = cols
    tensor[:, 1] = rows
    tensor[:, 2] = 100.0 + cols + rows
    corrected = np.zeros((height, width, 3), np.uint8)
    affine = np.asarray(((1.0, 0.0, 0.5), (0.0, 1.0, -0.25), (0.0, 0.0, 1.0)))
    calibration = ImageCalibration(
        corrected_bgr=corrected,
        colour_card=None,
        ruler=None,
        affine_matrix=affine,
        deskew_degrees=0.0,
        perspective_corrected=False,
        perspective_strength=0.0,
        channel_gains_bgr=(1.0, 1.0, 1.0),
        pixels_per_mm=None,
        scale_confidence=0.0,
        ruler_tick_spacing_px=None,
        scale_bar_mm=0.0,
        warnings=(),
        gpu_corrected_bgr=tensor,
    )
    layers = SimpleNamespace(
        valid_mask=np.ones(source_shape, dtype=bool),
        offset_x=int(crop_offset[0]),
        offset_y=int(crop_offset[1]),
    )
    if method == "procedural":
        count = int(labels.max(initial=0))
        centers = np.zeros((count, 2), np.float32)
        zeros = np.zeros(labels.shape, np.float32)
        result = ProceduralInstanceResult(
            labels=labels,
            centres_xy=centers,
            marker_scores=np.zeros(count, np.float32),
            instance_confidences=np.ones(count, np.float32),
            occupancy_likelihood=zeros.copy(),
            occupancy_mask=zeros.astype(bool),
            boundary_cost=zeros.copy(),
            centre_likelihood=zeros.copy(),
            source_shape=tuple(source_shape),
            working_scale=min(source_shape[0] / labels.shape[0], source_shape[1] / labels.shape[1]),
        )
        result_name = "procedural_instances"
    else:
        count = int(labels.max(initial=0))
        family = ModelFamily.UNET_WATERSHED if method == "unet" else ModelFamily.STARDIST
        result = LearnedInstanceResult(
            family=family,
            labels=labels,
            centres_xy=np.zeros((count, 2), np.float32),
            instance_confidences=np.ones(count, np.float32),
            method=method,
            checkpoint_id="model.ckpt#sha256=checkpoint-digest",
        )
        result_name = "unet_instances" if method == "unet" else "stardist_instances"

    kwargs = {
        "image_path": None,
        "dish": None,
        "proposals": (),
        "estimated_seed_diameter_px": 0.0,
        "reference_seed_count": 0,
        "foreground_threshold": 0.0,
        "foreground_pixel_count": 0,
        "analysis_region_pixel_count": 0,
        "distance_candidate_count": 0,
        "circle_candidate_count": 0,
        "crowding": "",
        "warnings": (),
        "calibration": calibration,
        "layers": layers,
        "advanced": None,
        "crop_offset": tuple(crop_offset),
        "foreground_feature": None,
        "foreground_colour_probability": None,
        "foreground_probability": None,
        "foreground_mask": None,
        "distance_transform": None,
        "distance_candidate_geometry": np.empty((0, 4), np.float32),
        "circle_candidate_geometry": np.empty((0, 4), np.float32),
        "reference_roi": (0, 0, 0, 0),
        "perimeter_background_lab": None,
        "perimeter_background_band": None,
        "background_prior_deviation": 0.0,
        "foreground_reference_count": 0,
        "source_sha256": "source-digest",
        result_name: result,
    }
    return BaselineAnalysis(**kwargs), result, tensor


class SeedCropTests(unittest.TestCase):
    def test_actual_cuda_crop_is_aligned_isolated_and_transformable(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required to verify resident crop tensors.")
        labels = np.zeros((5, 6), np.int32)
        labels[1, 1:3] = 1
        labels[1, 3] = 2
        analysis, result, source = _analysis(labels, labels.shape)
        source_before = source.clone()
        labels_before = result.labels.copy()

        batch = extract_seed_crops(
            analysis,
            method="unet_instances",
            seed_ids=(1,),
            padding_px=1,
            result_revision=_revision("a"),
        )

        self.assertEqual(batch.state, "complete")
        crop = batch.records[0]
        self.assertEqual(crop.status, "available")
        self.assertEqual(crop.corrected_bbox, (2, 1, 6, 4))
        self.assertEqual(tuple(crop.image_crop.shape), (1, 3, 3, 4))
        self.assertEqual(tuple(crop.mask_crop.shape), (3, 4))
        self.assertEqual(crop.image_crop.device.type, "cuda")
        self.assertEqual(crop.mask_crop.device.type, "cuda")
        self.assertEqual(crop.mask_crop.dtype, torch.bool)
        self.assertTrue(torch.equal(crop.image_crop, source[:, :, 1:4, 2:6]))
        expected_mask = torch.zeros((3, 4), dtype=torch.bool, device="cuda")
        expected_mask[1, 1:3] = True
        self.assertTrue(torch.equal(crop.mask_crop, expected_mask))
        self.assertEqual(crop.foreign_seed_ids, (2,))
        self.assertEqual(crop.foreign_pixel_count, 1)
        self.assertEqual(crop.source_validity, "unknown")
        self.assertEqual(crop.method, "unet_instances")
        self.assertEqual(crop.label_family, "unet_watershed")
        self.assertEqual(crop.checkpoint_id, "model.ckpt#sha256=checkpoint-digest")
        self.assertEqual(crop.result_revision, _revision("a"))
        self.assertFalse(crop.touches_analysis_crop)
        self.assertFalse(crop.clipped_by_corrected_frame)

        source_point = np.asarray((3.0, 4.0, 1.0))
        mapped = np.asarray(crop.source_to_crop) @ source_point
        expected = np.asarray(crop.source_to_corrected) @ source_point
        expected[:2] -= np.asarray((2.0, 1.0))
        np.testing.assert_allclose(mapped, expected)
        self.assertTrue(torch.equal(source, source_before))
        np.testing.assert_array_equal(result.labels, labels_before)

    def test_procedural_scale_uses_actual_per_axis_rounding(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required to verify resident crop tensors.")
        labels = np.zeros((3, 5), np.int32)
        labels[1, 3] = 1
        analysis, _, _ = _analysis(
            labels,
            (7, 11),
            crop_offset=(2, 1),
            frame_shape=(10, 16),
            method="procedural",
        )

        batch = extract_seed_crops(
            analysis,
            method="procedural_instances",
            seed_ids=(1,),
            padding_px=0,
            result_revision=_revision("b"),
        )

        crop = batch.records[0]
        self.assertEqual(crop.status, "available")
        self.assertEqual(crop.corrected_bbox, (9, 3, 11, 6))
        self.assertEqual(crop.label_shape, (3, 5))
        self.assertEqual(crop.source_shape, (7, 11))
        self.assertEqual(crop.analysis_crop_offset, (2, 1))
        self.assertEqual(crop.label_to_source_scale_xy, (11 / 5, 7 / 3))
        self.assertEqual(int(crop.mask_crop.sum().item()), 6)
        self.assertTrue(torch.all(crop.mask_crop))

    def test_padding_identity_clipping_and_missing_ids_are_explicit(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required to verify resident crop tensors.")
        labels = np.zeros((5, 6), np.int32)
        labels[0, 0] = 1
        analysis, result, source = _analysis(
            labels, labels.shape, crop_offset=(0, 0), frame_shape=(8, 12)
        )
        original_labels = result.labels.copy()
        original_source = source.clone()

        first = extract_seed_crops(
            analysis,
            method="unet_instances",
            seed_ids=(1, 99),
            padding_px=1,
            result_revision=_revision("c"),
        )
        second = extract_seed_crops(
            analysis,
            method="unet_instances",
            seed_ids=(1,),
            padding_px=2,
            result_revision=_revision("c"),
        )

        clipped, missing = first.records
        wider = second.records[0]
        self.assertEqual(first.state, "complete")
        self.assertEqual(clipped.status, "available")
        self.assertEqual(clipped.requested_bbox, (-1, -1, 2, 2))
        self.assertEqual(clipped.corrected_bbox, (0, 0, 2, 2))
        self.assertTrue(clipped.clipped_by_corrected_frame)
        self.assertTrue(clipped.touches_corrected_frame)
        self.assertTrue(clipped.touches_analysis_crop)
        self.assertEqual(clipped.source_validity, "unknown")
        self.assertEqual(missing.status, "missing")
        self.assertIsNone(missing.image_crop)
        self.assertIsNone(missing.mask_crop)
        self.assertNotEqual(clipped.artifact_identity, wider.artifact_identity)
        self.assertEqual(wider.padding_px, 2)
        self.assertTrue(torch.equal(source, original_source))
        np.testing.assert_array_equal(result.labels, original_labels)

    def test_cuda_absence_is_unavailable_without_cpu_fallback(self):
        labels = np.zeros((5, 6), np.int32)
        labels[1, 1] = 1
        analysis, _, _ = _analysis(labels, labels.shape, device="cpu")

        batch = extract_seed_crops(
            analysis,
            method="unet_instances",
            seed_ids=(1, 7),
            padding_px=0,
            result_revision=_revision("d"),
        )

        self.assertEqual(batch.state, "unavailable")
        self.assertEqual([record.status for record in batch.records], ["unavailable", "unavailable"])
        self.assertTrue(all(record.image_crop is None and record.mask_crop is None for record in batch.records))
        self.assertIn("CUDA", batch.records[0].reason)

    def test_cuda_tensor_must_match_corrected_cpu_frame_dimensions(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required to verify corrected-frame consistency.")
        labels = np.zeros((5, 6), np.int32)
        labels[2, 2] = 1
        analysis, _, source = _analysis(labels, labels.shape)
        mismatched_calibration = replace(
            analysis.calibration,
            corrected_bgr=np.zeros((7, 12, 3), dtype=np.uint8),
        )
        mismatched_analysis = replace(analysis, calibration=mismatched_calibration)
        original_source = source.clone()

        batch = extract_seed_crops(
            mismatched_analysis,
            method="unet_instances",
            seed_ids=(1,),
            padding_px=0,
            result_revision=_revision("0"),
        )

        self.assertEqual(batch.state, "unavailable")
        self.assertEqual(batch.records[0].status, "unavailable")
        self.assertIsNone(batch.records[0].image_crop)
        self.assertIsNone(batch.records[0].mask_crop)
        self.assertIn("do not match", batch.records[0].reason)
        self.assertTrue(torch.equal(source, original_source))

    def test_cancellation_marks_unfinished_ids_and_never_mutates_inputs(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required to verify resident crop tensors.")
        labels = np.zeros((5, 6), np.int32)
        labels[1, 1] = 1
        labels[2, 2] = 2
        analysis, result, source = _analysis(labels, labels.shape)
        source_before = source.clone()
        labels_before = result.labels.copy()
        calls = 0

        def cancel_after_first_record():
            nonlocal calls
            calls += 1
            return calls >= 6

        batch = extract_seed_crops(
            analysis,
            method="unet_instances",
            seed_ids=(1, 2, 3),
            padding_px=0,
            result_revision=_revision("e"),
            cancel=cancel_after_first_record,
        )

        self.assertEqual(batch.state, "cancelled")
        self.assertEqual([record.status for record in batch.records], ["available", "cancelled", "cancelled"])
        self.assertIsNone(batch.records[1].image_crop)
        self.assertTrue(torch.equal(source, source_before))
        np.testing.assert_array_equal(result.labels, labels_before)

    def test_large_padding_is_rejected_before_any_cuda_crop_allocation(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required to verify crop allocation budgets.")
        height = width = 2048
        labels = np.zeros((height, width), np.int32)
        labels[8, 8] = 1
        labels[20, 20] = 2
        labels[30, 30] = 3
        labels[40, 40] = 4
        analysis, result, source = _analysis(
            labels,
            labels.shape,
            crop_offset=(0, 0),
            frame_shape=(height, width),
        )
        original_labels = result.labels.copy()
        original_source = source.clone()
        allocated_before = torch.cuda.memory_allocated(source.device)

        batch = extract_seed_crops(
            analysis,
            method="unet_instances",
            seed_ids=(1, 2, 3, 4),
            padding_px=2048,
            result_revision=_revision("f"),
        )

        allocated_after = torch.cuda.memory_allocated(source.device)
        self.assertEqual(batch.state, "failed")
        self.assertEqual([record.status for record in batch.records], ["failed"] * 4)
        self.assertTrue(all("budget" in record.reason.lower() for record in batch.records))
        self.assertTrue(all(record.image_crop is None and record.mask_crop is None
                            for record in batch.records))
        self.assertEqual(allocated_after, allocated_before)
        self.assertTrue(torch.equal(source, original_source))
        np.testing.assert_array_equal(result.labels, original_labels)

    def test_result_revision_and_integer_geometry_inputs_are_strict(self):
        labels = np.zeros((5, 6), np.int32)
        labels[1, 1] = 1
        analysis, _, _ = _analysis(
            labels, labels.shape, crop_offset=(2.5, 1), device="cpu"
        )
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            extract_seed_crops(
                analysis, method="unet_instances", seed_ids=(1,), padding_px=0,
                result_revision="not-a-digest",
            )
        batch = extract_seed_crops(
            analysis,
            method="unet_instances",
            seed_ids=(1,),
            padding_px=0,
            result_revision=_revision("7"),
        )
        self.assertEqual(batch.state, "unavailable")
        self.assertIn("integer x and y", batch.records[0].reason)
        with self.assertRaisesRegex(ValueError, "positive integers"):
            extract_seed_crops(
                analysis, method="unet_instances", seed_ids=(1.5,), padding_px=0,
                result_revision=_revision("8"),
            )
        with self.assertRaisesRegex(ValueError, "non-negative integer"):
            extract_seed_crops(
                analysis, method="unet_instances", seed_ids=(1,), padding_px=0.5,
                result_revision=_revision("9"),
            )

    def test_procedural_source_shape_rejects_fractional_and_boolean_dimensions(self):
        labels = np.zeros((3, 5), np.int32)
        labels[1, 2] = 1
        analysis, result, _ = _analysis(
            labels, (7, 11), method="procedural", device="cpu"
        )
        for invalid_shape in ((7.5, 11), (True, 11)):
            with self.subTest(source_shape=invalid_shape):
                invalid_result = replace(result, source_shape=invalid_shape)
                invalid_analysis = replace(analysis, procedural_instances=invalid_result)
                batch = extract_seed_crops(
                    invalid_analysis,
                    method="procedural_instances",
                    seed_ids=(1,),
                    padding_px=0,
                    result_revision=_revision("1"),
                )
                self.assertEqual(batch.state, "unavailable")
                self.assertEqual(batch.records[0].status, "unavailable")
                self.assertIn("positive integers", batch.records[0].reason)

    def test_record_contract_exposes_serializable_metadata_and_crop_tensors(self):
        from seedvision.export.crops import SeedCropRecord

        names = {field.name for field in fields(SeedCropRecord)}
        self.assertIn("image_crop", names)
        self.assertIn("mask_crop", names)
        self.assertIn("artifact_identity", names)
        self.assertIn("label_family", names)
        self.assertIn("source_to_crop", names)
        self.assertIn("source_validity", names)


if __name__ == "__main__":
    unittest.main()
