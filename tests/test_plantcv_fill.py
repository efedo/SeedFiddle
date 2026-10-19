from __future__ import annotations

from tests.functional_units import functional_units

import hashlib
import json
import os
from pathlib import Path
import time
import unittest

import cv2
import numpy as np
import torch

from seedvision.cuda.plantcv_fill import (
    MAX_RASTER_PIXELS,
    FillCancelledError,
    FillResourceError,
    PLANTCV_COMMIT,
    PLANTCV_FILL_SOURCE_SHA256,
    PLANTCV_VERSION,
    fill,
)


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PYTHON = ROOT / "artifacts" / "plantcv-reference-env" / "Scripts" / "python.exe"
SOURCE_ROOT = ROOT / "artifacts" / "reference-sources" / "plantcv-v4.11.3"
FILL_SOURCE = SOURCE_ROOT / "plantcv" / "plantcv" / "fill.py"
REFERENCE_READY = all(
    path.exists()
    for path in (
        REFERENCE_PYTHON,
        ROOT / "artifacts" / "reference-sources" / "plantcv-source-audit.json",
        ROOT / "artifacts" / "reference-sources" / "plantcv-environment.json",
        FILL_SOURCE,
        ROOT / "config" / "plantcv-reference-requirements.txt",
    )
)


def _reference_runner():
    from seedvision.benchmarking.reference_runner import PlantCVReferenceRunner

    return PlantCVReferenceRunner(REFERENCE_PYTHON)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA small-object filtering requires an available CUDA device")
@functional_units
class PlantCVFillTests(unittest.TestCase):
    def _assert_matches_reference(self, mask: np.ndarray, size: int) -> torch.Tensor:
        reference = _reference_runner().run(
            "morphology.fill",
            {"mask": mask},
            {"size": size},
        )
        self.assertEqual(reference.operation, "morphology.fill")
        self.assertEqual(reference.effective_parameters["callable"], "plantcv.plantcv.fill")
        self.assertEqual(reference.effective_parameters["size"], size)
        self.assertEqual(reference.identity["version"], PLANTCV_VERSION)
        self.assertEqual(reference.identity["commit"], PLANTCV_COMMIT)

        source = torch.as_tensor(mask.copy(), dtype=torch.uint8, device="cuda")
        before = source.clone()
        actual = fill(source, size)
        expected = torch.as_tensor(reference.arrays["mask"], dtype=torch.uint8, device="cuda")
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(source, before, rtol=0, atol=0)
        self.assertEqual(actual.device, source.device)
        self.assertEqual(actual.dtype, torch.uint8)
        self.assertEqual(tuple(actual.shape), tuple(source.shape))
        self.assertTrue(bool(torch.all((actual == 0) | (actual == 255)).item()))
        return actual

    def check_test_source_pin_and_full_runner_pixel_cap(self) -> None:
        self.assertEqual((PLANTCV_VERSION, PLANTCV_COMMIT), (
            "4.11.3",
            "77e3322fd0cd5a9bef18bb99402f20c8f2db87a9",
        ))
        self.assertEqual(MAX_RASTER_PIXELS, 64_000_000)
        if FILL_SOURCE.is_file():
            self.assertEqual(
                hashlib.sha256(FILL_SOURCE.read_bytes()).hexdigest(),
                PLANTCV_FILL_SOURCE_SHA256,
            )
        config = json.loads(
            (ROOT / "config" / "peer_reference_sources.json").read_text(encoding="utf-8")
        )
        source_record = next(item for item in config["files"] if item["path"] == "plantcv/plantcv/fill.py")
        self.assertEqual(source_record["sha256"], PLANTCV_FILL_SOURCE_SHA256)

    def check_test_area_threshold_equality_four_connectivity_and_border_objects(self) -> None:
        mask = np.zeros((10, 14), dtype=np.uint8)
        mask[0, 1:5] = 255  # Four pixels touching the top border.
        mask[2, 8] = 255
        mask[3, 9] = 255  # Diagonal contact remains two 4-connected singletons.
        mask[5, 1:3] = 255  # Area two, removed at size three.
        mask[6:8, 6] = 255
        mask[7, 7] = 255  # Area three, retained exactly at size three.
        actual = self._assert_matches_reference(mask, 3)

        expected = np.zeros_like(mask)
        expected[0, 1:5] = 255
        expected[6:8, 6] = 255
        expected[7, 7] = 255
        torch.testing.assert_close(actual.cpu(), torch.from_numpy(expected), rtol=0, atol=0)

    def check_test_diagonal_contact_does_not_join_foreground_components(self) -> None:
        mask = np.array(((255, 0), (0, 255)), dtype=np.uint8)
        actual = self._assert_matches_reference(mask, 2)
        self.assertFalse(bool(torch.any(actual).item()))

    def check_test_empty_and_full_masks_keep_source_area_boundary_semantics(self) -> None:
        empty = np.zeros((7, 11), dtype=np.uint8)
        self._assert_matches_reference(empty, 4)

        full = np.full((7, 11), 255, dtype=np.uint8)
        retained = self._assert_matches_reference(full, int(full.size))
        self.assertTrue(bool(torch.all(retained == 255).item()))
        removed = self._assert_matches_reference(full, int(full.size) + 1)
        self.assertFalse(bool(torch.any(removed).item()))

    def check_test_binary_domain_parameters_cancellation_and_resource_limit_are_strict(self) -> None:
        mask = torch.zeros((4, 5), dtype=torch.uint8, device="cuda")
        mask[1, 1:3] = 255
        before = mask.clone()
        with self.assertRaisesRegex(ValueError, "CUDA"):
            fill(mask.cpu(), 2)
        with self.assertRaises(TypeError):
            fill(mask.float(), 2)
        with self.assertRaisesRegex(ValueError, "0 and 255"):
            fill(torch.tensor([[0, 1]], dtype=torch.uint8, device="cuda"), 2)
        with self.assertRaises(TypeError):
            fill(mask, True)
        with self.assertRaises(TypeError):
            fill(mask, 2.0)
        with self.assertRaises(ValueError):
            fill(mask, 0)
        with self.assertRaises(ValueError):
            fill(mask, -1)
        with self.assertRaises(FillCancelledError):
            fill(mask, 2, cancel=lambda: True)
        torch.testing.assert_close(mask, before, rtol=0, atol=0)

        oversized = mask[:1, :1].expand(8000, 8001)
        with self.assertRaises(FillResourceError):
            fill(oversized, 2)

        huge_area = fill(mask, 10**100)
        self.assertFalse(bool(torch.any(huge_area).item()))
        torch.testing.assert_close(mask, before, rtol=0, atol=0)

    def check_test_noncontiguous_input_retains_shape_device_and_original_values(self) -> None:
        base = torch.zeros((8, 14), dtype=torch.uint8, device="cuda")
        base[2, 2:6] = 255
        base[4:7, 10] = 255
        view = base[:, ::2]
        self.assertFalse(view.is_contiguous())
        before = view.clone()
        actual = fill(view, 3)
        expected = fill(view.contiguous(), 3)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(view, before, rtol=0, atol=0)
        self.assertEqual(actual.device, view.device)
        self.assertEqual(actual.dtype, torch.uint8)
        self.assertEqual(tuple(actual.shape), tuple(view.shape))

    def check_test_cancellation_during_component_work_returns_no_partial_mask(self) -> None:
        mask = torch.full((64, 64), 255, dtype=torch.uint8, device="cuda")
        before = mask.clone()
        checks = 0

        def cancel_during_work() -> bool:
            nonlocal checks
            checks += 1
            return checks >= 8

        with self.assertRaises(FillCancelledError):
            fill(mask, 2, cancel=cancel_during_work)
        self.assertGreaterEqual(checks, 8)
        torch.testing.assert_close(mask, before, rtol=0, atol=0)

    def check_test_committed_real_image_mask_performance_and_memory_diagnostic(self) -> None:
        image_path = ROOT / "images" / "IMG_9632c.JPG"
        gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        self.assertIsNotNone(gray, "committed image fixture must decode")
        mask = np.where(gray >= 128, 255, 0).astype(np.uint8)
        reference = _reference_runner().run(
            "morphology.fill",
            {"mask": mask},
            {"size": 32},
        )
        gpu_mask = torch.as_tensor(mask, dtype=torch.uint8, device="cuda")
        before = gpu_mask.clone()
        baseline_bytes = torch.cuda.memory_allocated(gpu_mask.device)
        torch.cuda.reset_peak_memory_stats(gpu_mask.device)

        started = time.perf_counter()
        actual = fill(gpu_mask, 32)
        torch.cuda.synchronize(gpu_mask.device)
        elapsed_seconds = time.perf_counter() - started
        peak_increment = torch.cuda.max_memory_allocated(gpu_mask.device) - baseline_bytes

        expected = torch.as_tensor(reference.arrays["mask"], dtype=torch.uint8, device="cuda")
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(gpu_mask, before, rtol=0, atol=0)
        self.assertLess(elapsed_seconds, 45.0, f"fixture diagnostic took {elapsed_seconds:.2f}s")
        self.assertLess(peak_increment, 512 * 1024 * 1024, f"peak CUDA allocation was {peak_increment:,} bytes")
        self.assertEqual(tuple(actual.shape), tuple(mask.shape))

    @unittest.skipUnless(REFERENCE_READY, "the pinned isolated PlantCV reference runtime is not present")
    def test_area_threshold_equality_four_connectivity_and_border_objects(self):
        self.scenario(self.check_test_area_threshold_equality_four_connectivity_and_border_objects, labelled=False)

    @unittest.skipUnless(REFERENCE_READY, "the pinned isolated PlantCV reference runtime is not present")
    def test_diagonal_contact_does_not_join_foreground_components(self):
        self.scenario(self.check_test_diagonal_contact_does_not_join_foreground_components, labelled=False)

    @unittest.skipUnless(REFERENCE_READY, "the pinned isolated PlantCV reference runtime is not present")
    def test_empty_and_full_masks_keep_source_area_boundary_semantics(self):
        self.scenario(self.check_test_empty_and_full_masks_keep_source_area_boundary_semantics, labelled=False)

    @unittest.skipUnless(REFERENCE_READY, "the pinned isolated PlantCV reference runtime is not present")
    @unittest.skipUnless(
        os.environ.get("SEEDFIDDLE_RUN_FILL_IMAGE_DIAGNOSTIC") == "1",
        "run explicitly with SEEDFIDDLE_RUN_FILL_IMAGE_DIAGNOSTIC=1",
    )
    def test_committed_real_image_mask_performance_and_memory_diagnostic(self):
        self.scenario(self.check_test_committed_real_image_mask_performance_and_memory_diagnostic, labelled=False)

    def test_workflow_lifecycle(self):
        """2 related PlantCVFillTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_source_pin_and_full_runner_pixel_cap)
        self.scenario(self.check_test_noncontiguous_input_retains_shape_device_and_original_values)

    def test_validation_and_recovery(self):
        """2 related PlantCVFillTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_binary_domain_parameters_cancellation_and_resource_limit_are_strict)
        self.scenario(self.check_test_cancellation_during_component_work_returns_no_partial_mask)


if __name__ == "__main__":
    unittest.main()
