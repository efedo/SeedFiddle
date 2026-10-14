from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import textwrap
import unittest
from unittest.mock import patch

import numpy as np
import torch

from seedvision.cuda.plantcv_median import (
    MAX_KERNEL_AXIS,
    MAX_RASTER_PIXELS,
    MAX_TEMPORARY_WINDOW_VALUES,
    MedianResourceError,
    MedianCancelledError,
    PLANTCV_COMMIT,
    PLANTCV_HELPER_REFERENCE_ENV_SHA256,
    PLANTCV_MEDIAN_SOURCE_SHA256,
    PLANTCV_VERSION,
    UnsupportedMedianGeometryError,
    median_blur,
)


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PYTHON = ROOT / "artifacts" / "plantcv-reference-env" / "Scripts" / "python.exe"
SOURCE_ROOT = ROOT / "artifacts" / "reference-sources" / "plantcv-v4.11.3"
SOURCE_CONFIG = ROOT / "config" / "peer_reference_sources.json"
HELPER_PATH = ROOT / "artifacts" / "plantcv-reference-env" / "Lib" / "site-packages" / "plantcv" / "plantcv" / "_helpers.py"
REFERENCE_ENV_READY = all(
    path.exists()
    for path in (
        REFERENCE_PYTHON,
        ROOT / "artifacts" / "reference-sources" / "plantcv-source-audit.json",
        ROOT / "artifacts" / "reference-sources" / "plantcv-environment.json",
        SOURCE_ROOT / "plantcv" / "plantcv" / "median_blur.py",
        ROOT / "config" / "plantcv-reference-requirements.txt",
        HELPER_PATH,
    )
)


def _direct_public_median(cases: list[tuple[np.ndarray, int | tuple[int, int], tuple[int, int, int, int] | None]]):
    """Call PlantCV's original public API in the installed pinned interpreter.

    The arrays/specification cross the process boundary only inside a temporary
    test directory. No production bridge or source implementation is involved.
    """
    child_source = textwrap.dedent(
        r"""
        import hashlib, importlib.metadata, json, sys
        from pathlib import Path
        import numpy as np
        import scipy
        import plantcv.plantcv as pcv
        from plantcv.plantcv.classes import Objects
        from plantcv.plantcv import _helpers

        request_path, input_path, output_path, meta_path = map(Path, sys.argv[1:5])
        request = json.loads(request_path.read_text(encoding='utf-8'))
        outputs = {}
        with np.load(input_path, allow_pickle=False) as inputs:
            for case in request:
                index = case['index']
                gray = inputs[f'gray_{index}']
                ksize = case['ksize']
                if isinstance(ksize, list):
                    ksize = tuple(ksize)
                roi = None
                if case['roi'] is not None:
                    x0, y0, x1, y1 = case['roi']
                    contour = [np.array([
                        [[x0, y0]], [[x0, y1]], [[x1, y1]], [[x1, y0]]
                    ], dtype=np.int32)]
                    hierarchy = np.array([[[-1, -1, -1, -1]]], dtype=np.int32)
                    roi = Objects(contours=[contour], hierarchy=[hierarchy])
                outputs[f'image_{index}'] = pcv.median_blur(gray, ksize=ksize, roi=roi)
        np.savez_compressed(output_path, **outputs)
        helper_bytes = Path(_helpers.__file__).read_bytes()
        meta = {
            'plantcv': importlib.metadata.version('plantcv'),
            'scipy': scipy.__version__,
            'python': sys.version.split()[0],
            'helper_sha256': hashlib.sha256(helper_bytes).hexdigest(),
            'callable': pcv.median_blur.__module__ + '.median_blur',
        }
        meta_path.write_text(json.dumps(meta, sort_keys=True), encoding='utf-8')
        """
    )
    request = []
    arrays = {}
    for index, (image, ksize, roi) in enumerate(cases):
        arrays[f"gray_{index}"] = np.ascontiguousarray(image, dtype=np.uint8)
        request.append({
            "index": index,
            "ksize": list(ksize) if type(ksize) is tuple else ksize,
            "roi": None if roi is None else list(roi),
        })
    with tempfile.TemporaryDirectory(prefix="seedfiddle-median-reference-") as temporary:
        directory = Path(temporary)
        request_path = directory / "request.json"
        input_path = directory / "inputs.npz"
        output_path = directory / "outputs.npz"
        meta_path = directory / "reference.json"
        request_path.write_text(json.dumps(request, separators=(",", ":")), encoding="utf-8")
        np.savez_compressed(input_path, **arrays)
        completed = subprocess.run(
            [str(REFERENCE_PYTHON), "-c", child_source, str(request_path), str(input_path), str(output_path), str(meta_path)],
            cwd=ROOT,
            capture_output=True,
            check=False,
            text=True,
            timeout=300,
        )
        if completed.returncode != 0:
            raise AssertionError(
                "Pinned PlantCV public median call failed in its isolated interpreter: "
                f"{completed.stderr[-6000:]}"
            )
        with np.load(output_path, allow_pickle=False) as outputs:
            results = [outputs[f"image_{index}"].copy() for index in range(len(cases))]
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    return results, metadata


class PlantCVMedianSourceTests(unittest.TestCase):
    def test_public_source_and_transitive_helper_evidence_are_distinguished(self) -> None:
        config = json.loads(SOURCE_CONFIG.read_text(encoding="utf-8"))
        self.assertEqual((config["version"], config["commit"]), (PLANTCV_VERSION, PLANTCV_COMMIT))
        self.assertEqual(PLANTCV_VERSION, "4.11.3")
        self.assertEqual(PLANTCV_COMMIT, "77e3322fd0cd5a9bef18bb99402f20c8f2db87a9")
        source = next(record for record in config["files"] if record["path"] == "plantcv/plantcv/median_blur.py")
        self.assertEqual(source["sha256"], PLANTCV_MEDIAN_SOURCE_SHA256)
        self.assertEqual(source["bytes"], 1355)
        local_source = SOURCE_ROOT / "plantcv" / "plantcv" / "median_blur.py"
        if local_source.is_file():
            raw = local_source.read_bytes()
            self.assertEqual(len(raw), source["bytes"])
            self.assertEqual(hashlib.sha256(raw).hexdigest(), PLANTCV_MEDIAN_SOURCE_SHA256)

        # _helpers.py is installed and hash-observed, but intentionally absent
        # from the direct-file source manifest.
        paths = {record["path"] for record in config["files"]}
        self.assertNotIn("plantcv/plantcv/_helpers.py", paths)
        if HELPER_PATH.is_file():
            helper_hash = hashlib.sha256(HELPER_PATH.read_bytes()).hexdigest()
            self.assertEqual(helper_hash, PLANTCV_HELPER_REFERENCE_ENV_SHA256)
            helper_source = HELPER_PATH.read_text(encoding="utf-8")
            self.assertIn("def _rect_filter", helper_source)
            self.assertIn("def _rect_replace", helper_source)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA median primitive requires an available CUDA device")
class PlantCVMedianPrimitiveTests(unittest.TestCase):
    def test_cancellation_before_during_and_after_tiles_preserves_input(self) -> None:
        gray = torch.arange(64, dtype=torch.uint8, device="cuda").reshape(8, 8)
        original = gray.clone()
        with patch("seedvision.cuda.plantcv_median._median_validated") as worker:
            with self.assertRaises(MedianCancelledError):
                median_blur(gray, 3, cancel=lambda: True)
            worker.assert_not_called()
            with self.assertRaises(TypeError):
                median_blur(gray, 3, cancel=True)
            worker.assert_not_called()
        # Four tiles for this ROI. Include cancellation after the final tile.
        for stop_at in (3, 6):
            calls = 0

            def cancelled() -> bool:
                nonlocal calls
                calls += 1
                return calls == stop_at

            with self.subTest(stop_at=stop_at), patch(
                "seedvision.cuda.plantcv_median.MAX_OUTPUTS_PER_TILE", 4
            ), self.assertRaises(MedianCancelledError):
                median_blur(gray, 3, roi=(2, 2, 6, 6), cancel=cancelled)
            self.assertEqual(calls, stop_at)
            torch.testing.assert_close(gray, original)
        torch.testing.assert_close(
            median_blur(gray, 3, cancel=lambda: False), median_blur(gray, 3)
        )

    def test_scalar_and_tuple_kernels_preserve_shape_dtype_device_and_input(self) -> None:
        gray = torch.tensor(
            [[0, 255, 20, 180, 1, 120],
             [255, 10, 160, 40, 240, 70],
             [80, 200, 30, 250, 90, 0],
             [15, 110, 220, 50, 140, 255]],
            dtype=torch.uint8,
            device="cuda",
        )
        before = gray.clone()
        for kernel in (1, 3, 4, (2, 4), (5, 1), (1, 5)):
            with self.subTest(kernel=kernel):
                result = median_blur(gray, kernel)
                self.assertEqual(result.dtype, torch.uint8)
                self.assertEqual(result.device, gray.device)
                self.assertEqual(tuple(result.shape), tuple(gray.shape))
                self.assertTrue(bool(torch.all(result <= 255)))
        torch.testing.assert_close(gray, before)

    def test_binary_inputs_remain_binary_and_parameters_change_selected_results(self) -> None:
        binary_np = np.random.default_rng(52).integers(0, 2, size=(15, 15), dtype=np.uint8) * 255
        binary = torch.from_numpy(binary_np).cuda()
        for kernel in (1, 2, 3, (3, 5), (4, 2)):
            result = median_blur(binary, kernel)
            self.assertTrue(set(result.unique().tolist()) <= {0, 255})
        self.assertFalse(torch.equal(median_blur(binary, 1), median_blur(binary, 3)))
        self.assertFalse(torch.equal(median_blur(binary, (3, 5)), median_blur(binary, (5, 3))))

    def test_noncontiguous_nonoverlapping_views_are_supported(self) -> None:
        base = torch.arange(0, 256, dtype=torch.uint8, device="cuda").reshape(8, 32)
        view = base[:, ::3]
        self.assertFalse(view.is_contiguous())
        actual = median_blur(view, (4, 3))
        expected = median_blur(view.contiguous(), (4, 3))
        torch.testing.assert_close(actual, expected)

    def test_roi_is_half_open_reflects_at_crop_and_preserves_exterior(self) -> None:
        image = torch.arange(0, 15 * 17, dtype=torch.int32, device="cuda").remainder(256).to(torch.uint8).reshape(15, 17)
        bounds = (3, 2, 14, 12)
        result = median_blur(image, (4, 3), roi=bounds)
        x0, y0, x1, y1 = bounds
        exterior = torch.ones_like(image, dtype=torch.bool)
        exterior[y0:y1, x0:x1] = False
        torch.testing.assert_close(result[exterior], image[exterior])
        self.assertEqual(tuple(result.shape), tuple(image.shape))
        self.assertEqual(result.device, image.device)
        self.assertEqual(result.dtype, image.dtype)
        self.assertFalse(torch.equal(result, median_blur(image, (4, 3))))

    def test_argument_checks_precede_work_and_reject_invalid_domains(self) -> None:
        valid = torch.arange(20, dtype=torch.uint8, device="cuda").reshape(4, 5)
        with self.assertRaisesRegex(ValueError, "CUDA"):
            median_blur(valid.cpu(), 3)
        with self.assertRaisesRegex(TypeError, "uint8"):
            median_blur(valid.float(), 3)
        with self.assertRaises(TypeError):
            median_blur([[1, 2], [3, 4]], 3)
        for malformed in (
            valid[0],
            valid.unsqueeze(0),
            torch.empty((0, 4), dtype=torch.uint8, device="cuda"),
            torch.empty((4, 0), dtype=torch.uint8, device="cuda"),
        ):
            with self.subTest(shape=tuple(malformed.shape)), self.assertRaises(ValueError):
                median_blur(malformed, 3)
        with self.assertRaises(TypeError):
            median_blur(torch.sparse_coo_tensor([[0], [0]], [1], (4, 4), device="cuda", check_invariants=True), 3)
        with self.assertRaisesRegex(ValueError, "overlapping"):
            median_blur(valid.as_strided((3, 3), (1, 1)), 3)

        for kernel in (True, 3.0, "3", [], (), (3,), (3, 3, 3), (3, False), (0, 3), (3, 256)):
            with self.subTest(kernel=kernel), self.assertRaises((TypeError, ValueError)):
                median_blur(valid, kernel)
        for bad_roi in (
            [],
            (0, 0, 2),
            (0, 0, 2, 2.0),
            (False, 0, 2, 2),
            (-1, 0, 2, 2),
            (0, 0, 6, 2),
            (1, 1, 1, 2),
            (0, 3, 2, 2),
        ):
            with self.subTest(roi=bad_roi), self.assertRaises((TypeError, ValueError)):
                median_blur(valid, 3, roi=bad_roi)
        oversized = valid[:1, :1].expand(8000, 8001)
        with self.assertRaises(MedianResourceError):
            median_blur(oversized, 3)
        self.assertEqual(MAX_RASTER_PIXELS, 64_000_000)
        self.assertEqual(MAX_KERNEL_AXIS, 255)


@unittest.skipUnless(torch.cuda.is_available() and REFERENCE_ENV_READY, "CUDA and pinned PlantCV environment are required")
class PlantCVMedianReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from seedvision.benchmarking.reference_runner import PlantCVReferenceRunner

        cls.runner = PlantCVReferenceRunner(REFERENCE_PYTHON)

    def test_odd_scalar_no_roi_matches_the_actual_reference_runner(self) -> None:
        rng = np.random.default_rng(0x5EED)
        image = rng.integers(0, 256, size=(17, 23), dtype=np.uint8)
        cases = ((image, 1), (image, 3), (image, 5))
        for gray, kernel in cases:
            with self.subTest(kernel=kernel):
                reference = self.runner.run("filter.median_blur", {"gray": gray}, {"ksize": kernel})
                actual = median_blur(torch.from_numpy(gray).cuda(), kernel)
                np.testing.assert_array_equal(actual.cpu().numpy(), reference.arrays["image"])

    def test_even_tuple_narrow_axis_roi_and_maximum_kernels_match_public_plantcv(self) -> None:
        rng = np.random.default_rng(0xC0FFEE)
        small = rng.integers(0, 256, size=(4, 7), dtype=np.uint8)
        roi_image = rng.integers(0, 256, size=(11, 14), dtype=np.uint8)
        cases = [
            (small, 4, None),
            (small, (2, 4), None),
            (small, (1, 5), None),
            (small, (5, 1), None),
            (small, 31, None),  # just below the source-undefined height bound
            (roi_image, (4, 2), (2, 1, 12, 9)),
        ]
        expected, metadata = _direct_public_median(cases)
        self.assertEqual(metadata["plantcv"], PLANTCV_VERSION)
        self.assertEqual(metadata["scipy"], "1.15.3")
        self.assertEqual(metadata["python"], "3.12.10")
        self.assertEqual(metadata["helper_sha256"], PLANTCV_HELPER_REFERENCE_ENV_SHA256)
        self.assertEqual(metadata["callable"], "plantcv.plantcv.median_blur.median_blur")
        for (image, kernel, roi), target in zip(cases, expected):
            with self.subTest(kernel=kernel, roi=roi):
                actual = median_blur(torch.from_numpy(image).cuda(), kernel, roi=roi)
                np.testing.assert_array_equal(actual.cpu().numpy(), target)

        safe_maximum = rng.integers(0, 256, size=(1, 33), dtype=np.uint8)
        max_expected, max_metadata = _direct_public_median(
            [(safe_maximum, 255, None), (safe_maximum, (255, 17), None)]
        )
        self.assertEqual(max_metadata["scipy"], "1.15.3")
        for kernel, target in zip((255, (255, 17)), max_expected):
            with self.subTest(kernel=kernel, geometry=safe_maximum.shape):
                actual = median_blur(torch.from_numpy(safe_maximum).cuda(), kernel)
                np.testing.assert_array_equal(actual.cpu().numpy(), target)

        max_roi_image = rng.integers(0, 256, size=(5, 37), dtype=np.uint8)
        max_roi = (2, 1, 35, 2)  # one-pixel-high, 33-pixel-wide crop
        max_roi_expected, _ = _direct_public_median(
            [(max_roi_image, (255, 255), max_roi)]
        )
        max_roi_actual = median_blur(
            torch.from_numpy(max_roi_image).cuda(), (255, 255), roi=max_roi,
        )
        np.testing.assert_array_equal(max_roi_actual.cpu().numpy(), max_roi_expected[0])

    def test_source_undefined_large_reflect_geometry_is_rejected_before_work(self) -> None:
        from unittest import mock

        small = torch.arange(28, dtype=torch.uint8, device="cuda").reshape(4, 7)
        with mock.patch(
            "seedvision.cuda.plantcv_median.torch.empty",
            side_effect=AssertionError("allocation occurred before geometry rejection"),
        ):
            for kernel in (32, 255, (32, 3), (3, 255)):
                with self.subTest(kernel=kernel), self.assertRaises(UnsupportedMedianGeometryError):
                    median_blur(small, kernel)

            larger = torch.arange(120, dtype=torch.uint8, device="cuda").reshape(10, 12)
            with self.assertRaises(UnsupportedMedianGeometryError):
                median_blur(larger, 32, roi=(2, 2, 6, 6))

        # A one-pixel axis is explicitly clamped by pinned SciPy's source.
        safe_thin = torch.arange(33, dtype=torch.uint8, device="cuda").reshape(1, 33)
        self.assertEqual(tuple(median_blur(safe_thin, 255).shape), (1, 33))

    def test_full_fixture_and_max_kernel_memory_are_bounded(self) -> None:
        import cv2

        fixture = ROOT / "images" / "IMG_9632c.JPG"
        image = cv2.imread(str(fixture), cv2.IMREAD_GRAYSCALE)
        self.assertIsNotNone(image)
        gray = torch.from_numpy(image).cuda()
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated(gray.device)
        torch.cuda.reset_peak_memory_stats(gray.device)
        actual = median_blur(gray, (3, 5))
        torch.cuda.synchronize()
        peak_delta = torch.cuda.max_memory_allocated(gray.device) - baseline
        self.assertLess(peak_delta, 256 * 1024 * 1024, f"unexpected CUDA workspace: {peak_delta:,} bytes")
        self.assertEqual(tuple(actual.shape), image.shape)

        reference = self.runner.run(
            "filter.median_blur", {"gray": image}, {"ksize": 5},
        )
        runner_actual = median_blur(gray, 5)
        np.testing.assert_array_equal(runner_actual.cpu().numpy(), reference.arrays["image"])

        # The two-axis maximum-kernel case is checked against the original
        # public PlantCV call using the largest source-defined narrow shape.
        tiny = np.arange(33, dtype=np.uint8).reshape(1, 33)
        expected, _ = _direct_public_median([(tiny, (255, 255), None)])
        tiny_gpu = torch.from_numpy(tiny).cuda()
        torch.cuda.synchronize()
        maximum_baseline = torch.cuda.memory_allocated(tiny_gpu.device)
        torch.cuda.reset_peak_memory_stats(tiny_gpu.device)
        maximum = median_blur(tiny_gpu, (255, 255))
        torch.cuda.synchronize()
        maximum_peak = torch.cuda.max_memory_allocated(tiny_gpu.device) - maximum_baseline
        self.assertLess(maximum_peak, 64 * 1024 * 1024, f"maximum-kernel workspace: {maximum_peak:,} bytes")
        np.testing.assert_array_equal(maximum.cpu().numpy(), expected[0])
        self.assertGreaterEqual(MAX_TEMPORARY_WINDOW_VALUES, 255 * 255)


if __name__ == "__main__":
    unittest.main()
