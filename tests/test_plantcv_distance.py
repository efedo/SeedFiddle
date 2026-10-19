from __future__ import annotations

from tests.functional_units import functional_units

import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import textwrap
import time
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch

from seedvision.cuda.plantcv_distance import (
    MAX_RASTER_PIXELS,
    DistanceCancelledError,
    DistanceResourceError,
    OPENCV_DISTANCE_SOURCE_URL,
    OPENCV_REFERENCE_VERSION,
    OPENCV_REFERENCE_WHEEL,
    PLANTCV_COMMIT,
    PLANTCV_VERSION,
    PLANTCV_WATERSHED_SOURCE_SHA256,
    distance_map,
)


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PYTHON = ROOT / "artifacts" / "plantcv-reference-env" / "Scripts" / "python.exe"
SOURCE_CONFIG = ROOT / "config" / "peer_reference_sources.json"
SOURCE_ROOT = ROOT / "artifacts" / "reference-sources" / "plantcv-v4.11.3"
WATERSHED_SOURCE = SOURCE_ROOT / "plantcv" / "plantcv" / "watershed.py"
REFERENCE_READY = all(
    path.exists()
    for path in (
        REFERENCE_PYTHON,
        SOURCE_CONFIG,
        ROOT / "config" / "plantcv-reference-requirements.txt",
        WATERSHED_SOURCE,
        ROOT / "artifacts" / "reference-sources" / "plantcv-environment.json",
    )
)


def _pinned_opencv_distance(masks: list[np.ndarray]) -> tuple[list[np.ndarray], dict[str, str]]:
    """Call the installed OpenCV function from the pinned PlantCV environment."""
    child_source = textwrap.dedent(
        r"""
        import importlib.metadata, json, sys
        from pathlib import Path
        import cv2
        import numpy as np

        input_path, output_path, metadata_path = map(Path, sys.argv[1:4])
        results = {}
        with np.load(input_path, allow_pickle=False) as inputs:
            for name in sorted(inputs.files):
                mask = inputs[name]
                distance, _ = cv2.distanceTransformWithLabels(
                    mask, cv2.DIST_L2, maskSize=0
                )
                results[name] = distance
        np.savez_compressed(output_path, **results)
        metadata_path.write_text(json.dumps({
            "opencv": cv2.__version__,
            "opencv_distribution": importlib.metadata.version("opencv-python"),
            "call": "cv2.distanceTransformWithLabels(mask, cv2.DIST_L2, maskSize=0)[0]",
        }, sort_keys=True), encoding="utf-8")
        """
    )
    with tempfile.TemporaryDirectory(prefix="seedfiddle-distance-reference-") as temporary:
        directory = Path(temporary)
        inputs_path = directory / "inputs.npz"
        outputs_path = directory / "outputs.npz"
        metadata_path = directory / "reference.json"
        arrays = {
            f"mask_{index:03d}": np.ascontiguousarray(mask, dtype=np.uint8)
            for index, mask in enumerate(masks)
        }
        np.savez_compressed(inputs_path, **arrays)
        completed = subprocess.run(
            [
                str(REFERENCE_PYTHON),
                "-c",
                child_source,
                str(inputs_path),
                str(outputs_path),
                str(metadata_path),
            ],
            cwd=ROOT,
            capture_output=True,
            check=False,
            text=True,
            timeout=300,
        )
        if completed.returncode != 0:
            raise AssertionError(
                "Pinned OpenCV distanceTransformWithLabels call failed: "
                f"{completed.stderr[-6000:]}"
            )
        with np.load(outputs_path, allow_pickle=False) as outputs:
            results = [outputs[f"mask_{index:03d}"].copy() for index in range(len(masks))]
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return results, metadata


def _varied_masks() -> list[np.ndarray]:
    rng = np.random.default_rng(0xF09_411)
    masks = [
        np.zeros((5, 9), dtype=np.uint8),
        np.full((5, 9), 255, dtype=np.uint8),
        np.array([[255, 255, 0, 255, 255, 255, 255]], dtype=np.uint8),
        np.array([[0], [255], [255], [255], [255]], dtype=np.uint8),
        np.pad(np.array([[0]], dtype=np.uint8), ((0, 4), (0, 7)), constant_values=255),
    ]
    ring = np.full((9, 13), 255, dtype=np.uint8)
    ring[4, 6] = 0
    masks.append(ring)
    two_holes = np.full((11, 17), 255, dtype=np.uint8)
    two_holes[3, 4] = 0
    two_holes[7, 12] = 0
    masks.append(two_holes)
    for shape, density in (((13, 21), 0.08), ((23, 11), 0.43), ((31, 37), 0.91)):
        masks.append((rng.random(shape) < density).astype(np.uint8) * 255)
    return masks


class PlantCVDistanceReferenceTests(unittest.TestCase):
    def test_plantcv_source_pin_and_reference_wheel_are_explicit(self) -> None:
        config = json.loads(SOURCE_CONFIG.read_text(encoding="utf-8"))
        self.assertEqual((config["version"], config["commit"]), (PLANTCV_VERSION, PLANTCV_COMMIT))
        self.assertEqual(PLANTCV_VERSION, "4.11.3")
        self.assertEqual(PLANTCV_COMMIT, "77e3322fd0cd5a9bef18bb99402f20c8f2db87a9")
        record = next(item for item in config["files"] if item["path"] == "plantcv/plantcv/watershed.py")
        self.assertEqual(record["sha256"], PLANTCV_WATERSHED_SOURCE_SHA256)
        self.assertEqual(record["bytes"], 3222)
        if WATERSHED_SOURCE.is_file():
            raw = WATERSHED_SOURCE.read_bytes()
            self.assertEqual(len(raw), record["bytes"])
            self.assertEqual(hashlib.sha256(raw).hexdigest(), PLANTCV_WATERSHED_SOURCE_SHA256)

        requirements = (ROOT / "config" / "plantcv-reference-requirements.txt").read_text(encoding="utf-8")
        self.assertIn(f"{OPENCV_REFERENCE_WHEEL} --hash=sha256:", requirements)
        self.assertEqual(OPENCV_REFERENCE_VERSION, "4.11.0")
        self.assertEqual(
            OPENCV_DISTANCE_SOURCE_URL,
            "https://github.com/opencv/opencv/blob/4.11.0/modules/imgproc/src/distransform.cpp",
        )
        self.assertEqual(MAX_RASTER_PIXELS, 64_000_000)


@unittest.skipUnless(
    torch.cuda.is_available() and REFERENCE_READY,
    "CUDA and the pinned PlantCV reference environment are required",
)
@functional_units
class PlantCVDistancePrimitiveTests(unittest.TestCase):
    def check_test_pinned_fixed_point_transform_matches_varied_masks_exactly(self) -> None:
        masks = _varied_masks()
        expected, metadata = _pinned_opencv_distance(masks)
        self.assertEqual(metadata["opencv"], OPENCV_REFERENCE_VERSION)
        self.assertEqual(metadata["opencv_distribution"], "4.11.0.86")
        self.assertEqual(
            metadata["call"],
            "cv2.distanceTransformWithLabels(mask, cv2.DIST_L2, maskSize=0)[0]",
        )
        self.assertEqual(expected[0].dtype, np.float32)
        self.assertEqual(float(expected[0].max()), 0.0)
        self.assertTrue(np.all(expected[1] == np.float32(65533.8046875)))
        # The labeled 5x5 transform's (1, 2) step is its pinned 2.1969 cost,
        # not sqrt(5), which distinguishes it from precise Euclidean distance.
        self.assertEqual(float(expected[4][1, 2]), np.float32(2.1968994140625))

        for index, (mask, reference) in enumerate(zip(masks, expected)):
            with self.subTest(index=index, shape=mask.shape):
                source = torch.as_tensor(mask.copy(), dtype=torch.uint8, device="cuda")
                before = source.clone()
                actual = distance_map(source)
                self.assertEqual(actual.dtype, torch.float32)
                self.assertEqual(actual.device, source.device)
                self.assertEqual(tuple(actual.shape), tuple(source.shape))
                np.testing.assert_array_equal(actual.cpu().numpy(), reference)
                torch.testing.assert_close(source, before, rtol=0, atol=0)

    def check_test_committed_fixture_mask_matches_reference_with_bounded_cuda_storage(self) -> None:
        fixture = ROOT / "images" / "IMG_9632c.JPG"
        gray = cv2.imread(str(fixture), cv2.IMREAD_GRAYSCALE)
        self.assertIsNotNone(gray, "committed fixture must decode")
        mask = np.where(gray >= 128, 255, 0).astype(np.uint8)
        reference, metadata = _pinned_opencv_distance([mask])
        self.assertEqual(metadata["opencv"], OPENCV_REFERENCE_VERSION)

        source = torch.as_tensor(mask, dtype=torch.uint8, device="cuda")
        before = source.clone()
        torch.cuda.synchronize(source.device)
        baseline = torch.cuda.memory_allocated(source.device)
        torch.cuda.reset_peak_memory_stats(source.device)
        started = time.perf_counter()
        actual = distance_map(source)
        torch.cuda.synchronize(source.device)
        elapsed = time.perf_counter() - started
        peak_delta = torch.cuda.max_memory_allocated(source.device) - baseline

        np.testing.assert_array_equal(actual.cpu().numpy(), reference[0])
        torch.testing.assert_close(source, before, rtol=0, atol=0)
        self.assertEqual(tuple(actual.shape), mask.shape)
        self.assertLess(elapsed, 45.0, f"fixture transform took {elapsed:.2f}s")
        self.assertLess(peak_delta, 512 * 1024 * 1024, f"CUDA workspace was {peak_delta:,} bytes")

    def check_test_scan_chunks_preserve_both_sweep_carries(self) -> None:
        masks = _varied_masks()[-3:]
        expected, _ = _pinned_opencv_distance(masks)
        with patch("seedvision.cuda.plantcv_distance.SCAN_CHUNK_PIXELS", 11):
            for index, (mask, reference) in enumerate(zip(masks, expected)):
                with self.subTest(index=index):
                    actual = distance_map(torch.as_tensor(mask.copy(), dtype=torch.uint8, device="cuda"))
                    np.testing.assert_array_equal(actual.cpu().numpy(), reference)

    def check_test_strided_input_validation_cancellation_and_resource_cap(self) -> None:
        base = torch.zeros((10, 18), dtype=torch.uint8, device="cuda")
        base[2:8, 4:14] = 255
        view = base[:, ::2]
        self.assertFalse(view.is_contiguous())
        before = view.clone()
        actual = distance_map(view)
        expected = distance_map(view.contiguous())
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(view, before, rtol=0, atol=0)

        with self.assertRaisesRegex(ValueError, "CUDA"):
            distance_map(view.cpu())
        with self.assertRaises(TypeError):
            distance_map(view.float())
        with self.assertRaisesRegex(ValueError, "0 and 255"):
            distance_map(torch.tensor([[0, 1]], dtype=torch.uint8, device="cuda"))
        with self.assertRaises(TypeError):
            distance_map(view, cancel=True)
        with self.assertRaises(DistanceCancelledError):
            distance_map(view, cancel=lambda: True)

        checks = 0

        def cancel_during_sweep() -> bool:
            nonlocal checks
            checks += 1
            return checks >= 7

        with self.assertRaises(DistanceCancelledError):
            distance_map(view, cancel=cancel_during_sweep)
        self.assertGreaterEqual(checks, 7)
        torch.testing.assert_close(view, before, rtol=0, atol=0)

        oversized = view[:1, :1].expand(8000, 8001)
        with self.assertRaises(DistanceResourceError):
            distance_map(oversized)
        self.assertEqual(MAX_RASTER_PIXELS, 64_000_000)

    def test_workflow_lifecycle(self):
        """3 related PlantCVDistancePrimitiveTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_pinned_fixed_point_transform_matches_varied_masks_exactly)
        self.scenario(self.check_test_committed_fixture_mask_matches_reference_with_bounded_cuda_storage)
        self.scenario(self.check_test_scan_chunks_preserve_both_sweep_carries)

    def test_validation_and_recovery(self):
        """1 related PlantCVDistancePrimitiveTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_strided_input_validation_cancellation_and_resource_cap)


if __name__ == "__main__":
    unittest.main()
