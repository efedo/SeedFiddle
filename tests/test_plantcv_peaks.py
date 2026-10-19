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

from seedvision.cuda.plantcv_peaks import (
    MAX_CANDIDATES,
    MAX_DISTANCE,
    MAX_ESTIMATED_TENSOR_WORKSPACE_BYTES,
    MAX_NMS_ROUNDS,
    MAX_RASTER_PIXELS,
    NUMPY_REFERENCE_VERSION,
    PeakCancelledError,
    PeakInputError,
    PeakResourceError,
    PLANTCV_COMMIT,
    PLANTCV_VERSION,
    PLANTCV_WATERSHED_SOURCE_SHA256,
    SCIKIT_IMAGE_COORD_SOURCE_SHA256,
    SCIKIT_IMAGE_PEAK_SOURCE_SHA256,
    SCIPY_FILTERS_SOURCE_SHA256,
    SCIPY_MEASUREMENTS_SOURCE_SHA256,
    SCIPY_MORPHOLOGY_SOURCE_SHA256,
    SCIPY_REFERENCE_VERSION,
    SCIKIT_IMAGE_REFERENCE_VERSION,
    _binary_opening_cross,
    _greedy_spacing,
    local_maxima,
)


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PYTHON = ROOT / "artifacts" / "plantcv-reference-env" / "Scripts" / "python.exe"
SOURCE_ROOT = ROOT / "artifacts" / "reference-sources" / "plantcv-v4.11.3"
WATERSHED_SOURCE = SOURCE_ROOT / "plantcv" / "plantcv" / "watershed.py"
REFERENCE_READY = all(
    path.exists()
    for path in (
        REFERENCE_PYTHON,
        ROOT / "config" / "peer_reference_sources.json",
        ROOT / "config" / "plantcv-reference-requirements.txt",
        ROOT / "artifacts" / "reference-sources" / "plantcv-environment.json",
        WATERSHED_SOURCE,
    )
)


def _pinned_peak_reference(
    cases: list[tuple[str, np.ndarray, np.ndarray, int]],
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, str]]:
    """Run peak_local_max in the repository's pinned CPU reference runtime."""
    child_source = textwrap.dedent(
        r"""
        import importlib.metadata, json, sys
        from pathlib import Path
        import numpy as np
        import scipy
        import skimage
        from skimage.feature import peak_local_max

        input_path, output_path, metadata_path = map(Path, sys.argv[1:4])
        results = {}
        with np.load(input_path, allow_pickle=False) as inputs:
            for case in sorted(name[:-6] for name in inputs.files if name.endswith('_image')):
                image = inputs[case + '_image']
                mask = inputs[case + '_mask']
                distance = int(inputs[case + '_distance'])
                coordinates = peak_local_max(
                    image, min_distance=distance, labels=mask
                )
                peaks = np.zeros(image.shape, dtype=bool)
                if len(coordinates):
                    peaks[tuple(coordinates.T)] = True
                results[case + '_coordinates'] = coordinates
                results[case + '_peaks'] = peaks
        np.savez_compressed(output_path, **results)
        metadata_path.write_text(json.dumps({
            'scikit_image': skimage.__version__,
            'scipy': scipy.__version__,
            'numpy': np.__version__,
            'scikit_image_distribution': importlib.metadata.version('scikit-image'),
            'call': 'peak_local_max(image, min_distance=distance, labels=mask)',
        }, sort_keys=True), encoding='utf-8')
        """
    )
    with tempfile.TemporaryDirectory(prefix="seedfiddle-peak-reference-") as temporary:
        directory = Path(temporary)
        input_path = directory / "inputs.npz"
        output_path = directory / "outputs.npz"
        metadata_path = directory / "reference.json"
        arrays: dict[str, np.ndarray] = {}
        for name, image, mask, distance in cases:
            arrays[f"{name}_image"] = np.ascontiguousarray(image, dtype=np.float32)
            arrays[f"{name}_mask"] = np.ascontiguousarray(mask, dtype=np.uint8)
            arrays[f"{name}_distance"] = np.asarray(distance, dtype=np.int64)
        np.savez_compressed(input_path, **arrays)
        completed = subprocess.run(
            [
                str(REFERENCE_PYTHON),
                "-c",
                child_source,
                str(input_path),
                str(output_path),
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
                "Pinned scikit-image peak_local_max call failed: "
                f"{completed.stderr[-6000:]}"
            )
        with np.load(output_path, allow_pickle=False) as outputs:
            result = {
                name: (
                    outputs[f"{name}_coordinates"].copy(),
                    outputs[f"{name}_peaks"].copy(),
                )
                for name, *_ in cases
            }
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return result, metadata


def _small_cases() -> list[tuple[str, np.ndarray, np.ndarray, int]]:
    rng = np.random.default_rng(0xF10_411)
    cases: list[tuple[str, np.ndarray, np.ndarray, int]] = []

    all_background = np.arange(15 * 19, dtype=np.float32).reshape(15, 19)
    cases.append(("all_background", all_background, np.zeros((15, 19), np.uint8), 2))

    all_foreground_constant = np.full((13, 17), 7.0, dtype=np.float32)
    cases.append(
        (
            "all_foreground_constant",
            all_foreground_constant,
            np.full((13, 17), 255, np.uint8),
            2,
        )
    )

    all_foreground_varied = rng.integers(0, 40, (17, 21)).astype(np.float32)
    cases.append(
        (
            "all_foreground_varied",
            all_foreground_varied,
            np.full((17, 21), 255, np.uint8),
            3,
        )
    )

    isolated = np.zeros((13, 15), dtype=np.float32)
    isolated[6, 7] = 4.0
    isolated_mask = np.zeros((13, 15), dtype=np.uint8)
    isolated_mask[6, 7] = 255
    cases.append(("single_pixel_roi", isolated, isolated_mask, 1))

    border = np.zeros((15, 19), dtype=np.float32)
    border_mask = np.zeros((15, 19), dtype=np.uint8)
    for y, x, value in ((1, 7, 30), (2, 7, 8), (3, 7, 20), (7, 11, 15)):
        border[y, x] = value
        border_mask[y, x] = 255
    cases.append(("border_distance_two", border, border_mask, 2))

    disconnected = np.zeros((21, 27), dtype=np.float32)
    disconnected_mask = np.zeros((21, 27), dtype=np.uint8)
    disconnected_mask[3:12, 3:11] = 255
    disconnected_mask[6:9, 6:8] = 0  # a hole in the first island
    disconnected_mask[13:18, 17:24] = 255  # same label value, separate island
    disconnected[disconnected_mask == 255] = rng.integers(
        1, 50, int(np.count_nonzero(disconnected_mask))
    )
    cases.append(("disconnected_hole_roi", disconnected, disconnected_mask, 2))

    square = np.zeros((17, 19), dtype=np.float32)
    square_mask = np.zeros((17, 19), dtype=np.uint8)
    square[6:9, 7:10] = 8.0
    square_mask[6:9, 7:10] = 255
    cases.append(("trivial_opening_fallback", square, square_mask, 1))

    flat_line = np.zeros((17, 19), dtype=np.float32)
    flat_line_mask = np.zeros((17, 19), dtype=np.uint8)
    flat_line[8, 4:13] = 10.0
    flat_line_mask[8, 4:13] = 255
    cases.append(("greedy_spacing_chain", flat_line, flat_line_mask, 2))

    no_spacing = np.zeros((17, 19), dtype=np.float32)
    no_spacing_mask = np.zeros((17, 19), dtype=np.uint8)
    no_spacing[8, 4:13] = 10.0
    no_spacing_mask[8, 4:13] = 255
    cases.append(("distance_one_keeps_plateau", no_spacing, no_spacing_mask, 1))

    strict_threshold = np.zeros((13, 17), dtype=np.float32)
    strict_threshold[6, 8] = 0.0  # global minimum cannot be a peak
    strict_threshold[6, 12] = 2.0
    strict_mask = np.zeros((13, 17), dtype=np.uint8)
    strict_mask[6, 8] = 255
    strict_mask[6, 12] = 255
    cases.append(("strict_global_minimum", strict_threshold, strict_mask, 1))

    return cases


class NmsVisitEstimateTests(unittest.TestCase):
    def test_separable_maxima_equal_square_pool_at_edges_short_dimensions_and_ties(self):
        from seedvision.cuda import plantcv_peaks as peaks
        for shape in ((1, 8), (8, 1), (3, 8), (8, 3)):
            for distance in (2, 3):
                for dtype in (torch.float32, torch.float64):
                    with self.subTest(shape=shape, distance=distance, dtype=dtype):
                        values = (torch.arange(shape[0] * shape[1]).reshape(shape) % 3).to(dtype)
                        if dtype == torch.float64:
                            values = -values
                        radius = distance - 1
                        kernel = 2 * radius + 1
                        expected = torch.nn.functional.max_pool2d(values[None, None], kernel,
                            stride=1, padding=radius)[0, 0]
                        self.assertTrue(torch.equal(peaks._spacing_maximum(values, kernel, radius), expected))

    def test_round_estimate_counts_four_axis_windows_dense_and_candidate_work(self):
        from seedvision.cuda import plantcv_peaks as peaks
        self.assertEqual(peaks.MAX_NMS_SAMPLE_VISITS, 64_000_000_000)
        for shape, distance, candidates in (((3, 8), 2, 6), ((8, 3), 3, 7)):
            with self.subTest(shape=shape, distance=distance):
                kernel = 2 * (distance - 1) + 1
                self.assertEqual(peaks._nms_round_visits(shape, distance, candidates),
                    shape[0] * shape[1] * (24 + 4 * kernel) + 64 * candidates)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
@functional_units
class NmsVisitBudgetTests(unittest.TestCase):
    def check_test_round_visit_floor_boundary_and_incomplete_output(self):
        from seedvision.cuda import plantcv_peaks as peaks
        coords = torch.tensor([[0, 0], [0, 2], [0, 4]], dtype=torch.int64, device='cuda')
        one_round = peaks._nms_round_visits((1, 5), 3, 3)
        # The middle point is suppressed in round one; the last needs round two.
        with patch.object(peaks, 'MAX_NMS_SAMPLE_VISITS', 2 * one_round - 1):
            with self.assertRaisesRegex(peaks.PeakResourceError, 'round/sample-visit limit'):
                peaks._greedy_spacing(coords, 3, (1, 5), None)
        with patch.object(peaks, 'MAX_NMS_SAMPLE_VISITS', 2 * one_round):
            selected = peaks._greedy_spacing(coords, 3, (1, 5), None)
        self.assertEqual(selected.tolist(), [[0, 0], [0, 4]])

    def check_test_zero_affordable_rounds_reject_before_rank_or_canvas_allocation(self):
        from seedvision.cuda import plantcv_peaks as peaks
        coords = torch.tensor([[0, 0], [0, 2], [0, 4]], dtype=torch.int64, device='cuda')
        visits = peaks._nms_round_visits((1, 5), 3, 3)
        def forbidden(*args, **kwargs):
            raise AssertionError('NMS allocation preceded work admission')
        with patch.object(peaks, 'MAX_NMS_SAMPLE_VISITS', visits - 1), \
             patch.object(torch, 'arange', side_effect=forbidden), \
             patch.object(torch, 'ones', side_effect=forbidden), \
             patch.object(torch, 'empty', side_effect=forbidden), \
             patch.object(torch, 'zeros', side_effect=forbidden):
            with self.assertRaisesRegex(peaks.PeakResourceError, 'before allocation'):
                peaks._greedy_spacing(coords, 3, (1, 5), None)

    def check_test_distance_one_and_empty_shortcuts_bypass_nms_budget(self):
        from seedvision.cuda import plantcv_peaks as peaks
        coords = torch.tensor([[0, 0], [2, 7]], dtype=torch.int64, device='cuda')
        empty = torch.empty((0, 2), dtype=torch.int64, device='cuda')
        with patch.object(peaks, 'MAX_NMS_SAMPLE_VISITS', 0):
            self.assertIs(peaks._greedy_spacing(coords, 1, (3, 8), None), coords)
            self.assertIs(peaks._greedy_spacing(empty, 3, (3, 8), None), empty)

    def check_test_rectangular_roi_d2_and_d3_match_priority_greedy_oracle(self):
        from seedvision.cuda import plantcv_peaks as peaks
        cases = (
            ((3, 8), 2, [[0, 0], [0, 1], [0, 2], [1, 5], [1, 6], [2, 7]]),
            ((3, 8), 3, [[0, 0], [0, 2], [0, 4], [1, 6], [2, 7], [2, 4]]),
        )
        for shape, distance, points in cases:
            with self.subTest(distance=distance):
                coords = torch.tensor(points, dtype=torch.int64, device='cuda')
                selected = peaks._greedy_spacing(coords, distance, shape, None).tolist()
                expected = []
                for point in points:
                    if all(max(abs(point[0] - kept[0]), abs(point[1] - kept[1])) >= distance
                           for kept in expected):
                        expected.append(point)
                self.assertEqual(selected, expected)

    def test_validation_and_recovery(self):
        """3 related NmsVisitBudgetTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_round_visit_floor_boundary_and_incomplete_output)
        self.scenario(self.check_test_zero_affordable_rounds_reject_before_rank_or_canvas_allocation)
        self.scenario(self.check_test_distance_one_and_empty_shortcuts_bypass_nms_budget)

    def test_workflow_lifecycle(self):
        """1 related NmsVisitBudgetTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_rectangular_roi_d2_and_d3_match_priority_greedy_oracle)


class PlantCVPeakContractTests(unittest.TestCase):
    def test_pinned_source_versions_and_contract_are_explicit(self) -> None:
        self.assertEqual((PLANTCV_VERSION, PLANTCV_COMMIT), (
            "4.11.3",
            "77e3322fd0cd5a9bef18bb99402f20c8f2db87a9",
        ))
        self.assertEqual(PLANTCV_WATERSHED_SOURCE_SHA256,
                         "d5fd90bf4e064117ff47ee333d2b83e62134901fda36f6966ae1b0be4a82a712")
        self.assertEqual(SCIKIT_IMAGE_REFERENCE_VERSION, "0.25.2")
        self.assertEqual(SCIPY_REFERENCE_VERSION, "1.15.3")
        self.assertEqual(NUMPY_REFERENCE_VERSION, "2.2.6")
        self.assertEqual(MAX_DISTANCE, 128)
        self.assertEqual(MAX_CANDIDATES, 1_000_000)
        self.assertEqual(MAX_NMS_ROUNDS, 4_096)
        self.assertEqual(MAX_RASTER_PIXELS, 64_000_000)
        self.assertEqual(MAX_ESTIMATED_TENSOR_WORKSPACE_BYTES, 768 * 1024 * 1024)
        if WATERSHED_SOURCE.is_file():
            self.assertEqual(
                hashlib.sha256(WATERSHED_SOURCE.read_bytes()).hexdigest(),
                PLANTCV_WATERSHED_SOURCE_SHA256,
            )
        self.assertEqual(
            SCIKIT_IMAGE_PEAK_SOURCE_SHA256,
            "193b2cf2d30805cb1250c0427e4a43b2a221cd7efb4ba7e827e85cc3ebb79fcd",
        )
        self.assertEqual(
            SCIKIT_IMAGE_COORD_SOURCE_SHA256,
            "cfc12d50f0cc30f06096ca2701fc89b69592843f0037a4c8184058f8ea50ebe6",
        )
        self.assertEqual(
            SCIPY_FILTERS_SOURCE_SHA256,
            "919dcbfd3cccf9c2b61210214d27c828a49529db75a6432f2d9c99442946ec65",
        )
        self.assertEqual(
            SCIPY_MORPHOLOGY_SOURCE_SHA256,
            "0f6ee388080105895b9d31b96b86912515c37e954f8b153a0fd1eeefeef462d2af",
        )
        self.assertEqual(
            SCIPY_MEASUREMENTS_SOURCE_SHA256,
            "1c7a3878776ba69c6811d12d7a1fd12288e365a0cec74d6285ffebcae9330ba",
        )

    def test_cross_opening_and_priority_rounds_preserve_pinned_edge_cases(self) -> None:
        square = torch.ones((3, 3), dtype=torch.bool)
        opened = _binary_opening_cross(square)
        expected_opening = torch.tensor(
            ((False, True, False), (True, True, True), (False, True, False)),
            dtype=torch.bool,
        )
        torch.testing.assert_close(opened, expected_opening)

        line = torch.tensor(
            [[0, x] for x in range(9)],
            dtype=torch.int64,
        )
        selected = _greedy_spacing(line, 2, (1, 9), None)
        torch.testing.assert_close(
            selected,
            torch.tensor([[0, 0], [0, 2], [0, 4], [0, 6], [0, 8]], dtype=torch.int64),
        )


@unittest.skipUnless(
    torch.cuda.is_available() and REFERENCE_READY,
    "CUDA and the pinned scikit-image reference environment are required",
)
@functional_units
class PlantCVPeakCudaTests(unittest.TestCase):
    def _assert_matches_reference(
        self,
        cases: list[tuple[str, np.ndarray, np.ndarray, int]],
    ) -> None:
        expected, metadata = _pinned_peak_reference(cases)
        self.assertEqual(metadata["scikit_image"], SCIKIT_IMAGE_REFERENCE_VERSION)
        self.assertEqual(metadata["scipy"], SCIPY_REFERENCE_VERSION)
        self.assertEqual(metadata["numpy"], NUMPY_REFERENCE_VERSION)
        self.assertEqual(
            metadata["call"],
            "peak_local_max(image, min_distance=distance, labels=mask)",
        )
        for name, image, mask, distance in cases:
            with self.subTest(name=name, shape=image.shape, distance=distance):
                device_image = torch.as_tensor(image.copy(), dtype=torch.float32, device="cuda")
                device_mask = torch.as_tensor(mask.copy(), dtype=torch.uint8, device="cuda")
                image_before = device_image.clone()
                mask_before = device_mask.clone()
                coordinates, peaks = local_maxima(
                    device_image, device_mask, distance
                )
                expected_coordinates, expected_peaks = expected[name]
                np.testing.assert_array_equal(coordinates.cpu().numpy(), expected_coordinates)
                np.testing.assert_array_equal(peaks.cpu().numpy(), expected_peaks)
                self.assertEqual(coordinates.dtype, torch.int64)
                self.assertEqual(peaks.dtype, torch.bool)
                self.assertEqual(coordinates.device, device_image.device)
                self.assertEqual(peaks.device, device_image.device)
                self.assertTrue(coordinates.is_contiguous())
                self.assertEqual(tuple(peaks.shape), tuple(image.shape))
                torch.testing.assert_close(device_image, image_before, rtol=0, atol=0)
                torch.testing.assert_close(device_mask, mask_before, rtol=0, atol=0)

    def check_test_varied_shapes_match_pinned_ordered_coordinates_and_every_mask_pixel(self) -> None:
        self._assert_matches_reference(_small_cases())

    def check_test_validation_cancellation_and_explicit_resource_limits(self) -> None:
        image = torch.zeros((15, 19), dtype=torch.float32, device="cuda")
        mask = torch.zeros((15, 19), dtype=torch.uint8, device="cuda")
        mask[7, 4:13] = 255
        image[7, 4:13] = 10.0
        before_image, before_mask = image.clone(), mask.clone()

        with self.assertRaises(PeakCancelledError):
            local_maxima(image, mask, 2, cancel=lambda: True)
        with self.assertRaises(TypeError):
            local_maxima(image, mask, 2, cancel=True)
        with self.assertRaises(PeakInputError):
            local_maxima(image, mask, 0)
        with self.assertRaises(PeakResourceError):
            local_maxima(image, mask, MAX_DISTANCE + 1)
        with self.assertRaises(PeakInputError):
            local_maxima(image[:, ::2], mask[:, ::2], 2)
        with self.assertRaises(PeakInputError):
            local_maxima(image, mask.cpu(), 2)
        with self.assertRaises(TypeError):
            local_maxima(image.double(), mask, 2)

        bad_mask = mask.clone()
        bad_mask[0, 0] = 1
        with self.assertRaisesRegex(PeakInputError, "0 and 255"):
            local_maxima(image, bad_mask, 2)
        bad_image = image.clone()
        bad_image[0, 0] = float("nan")
        with self.assertRaisesRegex(PeakInputError, "finite"):
            local_maxima(bad_image, mask, 2)

        with patch("seedvision.cuda.plantcv_peaks.MAX_CANDIDATES", 2):
            with self.assertRaises(PeakResourceError):
                local_maxima(image, mask, 1)
        with patch("seedvision.cuda.plantcv_peaks.MAX_NMS_ROUNDS", 1):
            with self.assertRaises(PeakResourceError):
                local_maxima(image, mask, 2)
        with patch(
            "seedvision.cuda.plantcv_peaks.MAX_ESTIMATED_TENSOR_WORKSPACE_BYTES", 1
        ):
            with patch("seedvision.cuda.plantcv_peaks.F.max_pool2d") as max_pool:
                with self.assertRaises(PeakResourceError):
                    local_maxima(image, mask, 2)
                max_pool.assert_not_called()
        torch.testing.assert_close(image, before_image, rtol=0, atol=0)
        torch.testing.assert_close(mask, before_mask, rtol=0, atol=0)

    def check_test_real_full_resolution_fixture_matches_pinned_peak_local_max(self) -> None:
        fixture = ROOT / "images" / "IMG_9632c.JPG"
        gray = cv2.imread(str(fixture), cv2.IMREAD_GRAYSCALE)
        self.assertIsNotNone(gray, "committed fixture must decode")
        mask_array = np.where(gray >= 128, 255, 0).astype(np.uint8)
        mask = torch.as_tensor(mask_array, dtype=torch.uint8, device="cuda")

        from seedvision.cuda.plantcv_distance import distance_map

        torch.cuda.synchronize(mask.device)
        memory_before = torch.cuda.memory_allocated(mask.device)
        torch.cuda.reset_peak_memory_stats(mask.device)
        started = time.perf_counter()
        distances = distance_map(mask)
        coordinates, peaks = local_maxima(distances, mask, 10)
        torch.cuda.synchronize(mask.device)
        elapsed = time.perf_counter() - started
        peak_bytes = torch.cuda.max_memory_allocated(mask.device) - memory_before

        expected, metadata = _pinned_peak_reference(
            [("fixture", distances.cpu().numpy(), mask_array, 10)]
        )
        self.assertEqual(metadata["scikit_image"], SCIKIT_IMAGE_REFERENCE_VERSION)
        expected_coordinates, expected_peaks = expected["fixture"]
        np.testing.assert_array_equal(coordinates.cpu().numpy(), expected_coordinates)
        np.testing.assert_array_equal(peaks.cpu().numpy(), expected_peaks)
        self.assertEqual(tuple(distances.shape), gray.shape)
        self.assertEqual(tuple(peaks.shape), gray.shape)
        self.assertLess(elapsed, 120.0, f"full fixture stages took {elapsed:.2f}s")
        self.assertLess(
            peak_bytes,
            1_500 * 1024 * 1024,
            f"fixture CUDA allocation delta was {peak_bytes:,} bytes",
        )

    def test_workflow_lifecycle(self):
        """2 related PlantCVPeakCudaTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_varied_shapes_match_pinned_ordered_coordinates_and_every_mask_pixel)
        self.scenario(self.check_test_real_full_resolution_fixture_matches_pinned_peak_local_max)

    def test_validation_and_recovery(self):
        """1 related PlantCVPeakCudaTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_validation_cancellation_and_explicit_resource_limits)


if __name__ == "__main__":
    unittest.main()
