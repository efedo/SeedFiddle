from __future__ import annotations

from tests.functional_units import functional_units

import hashlib
import json
from pathlib import Path
import unittest

import numpy as np
import torch

from seedvision.cuda.plantcv_morphology import (
    MAX_ITERATIONS,
    MAX_KERNEL_SIZE,
    MAX_RASTER_PIXELS,
    MIN_ITERATIONS,
    MIN_KERNEL_SIZE,
    MorphologyResourceError,
    PLANTCV_COMMIT,
    PLANTCV_DILATE_SOURCE_SHA256,
    PLANTCV_ERODE_SOURCE_SHA256,
    PLANTCV_HELPER_REFERENCE_ENV_SHA256,
    PLANTCV_VERSION,
    dilate,
    erode,
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
        SOURCE_ROOT / "plantcv" / "plantcv" / "erode.py",
        SOURCE_ROOT / "plantcv" / "plantcv" / "dilate.py",
        ROOT / "config" / "plantcv-reference-requirements.txt",
        HELPER_PATH,
    )
)


def _reference_runner():
    from seedvision.benchmarking.reference_runner import PlantCVReferenceRunner

    return PlantCVReferenceRunner(REFERENCE_PYTHON)


class PlantCVMorphologySourceTests(unittest.TestCase):
    def test_public_source_hashes_match_the_committed_release_registry(self) -> None:
        config = json.loads(SOURCE_CONFIG.read_text(encoding="utf-8"))
        self.assertEqual(config["reference"], "plantcv")
        self.assertEqual(config["version"], PLANTCV_VERSION)
        self.assertEqual(config["commit"], PLANTCV_COMMIT)
        self.assertEqual(PLANTCV_VERSION, "4.11.3")
        self.assertEqual(PLANTCV_COMMIT, "77e3322fd0cd5a9bef18bb99402f20c8f2db87a9")
        records = {record["path"]: record for record in config["files"]}
        expected = {
            "plantcv/plantcv/erode.py": (PLANTCV_ERODE_SOURCE_SHA256, 1200),
            "plantcv/plantcv/dilate.py": (PLANTCV_DILATE_SOURCE_SHA256, 1098),
        }
        for relative_path, (digest, byte_count) in expected.items():
            with self.subTest(path=relative_path):
                record = records[relative_path]
                self.assertEqual(record["sha256"], digest)
                self.assertEqual(record["bytes"], byte_count)
                local_path = SOURCE_ROOT.joinpath(*relative_path.split("/"))
                if local_path.is_file():
                    raw = local_path.read_bytes()
                    self.assertEqual(len(raw), byte_count)
                    self.assertEqual(hashlib.sha256(raw).hexdigest(), digest)

    @unittest.skipUnless(HELPER_PATH.is_file(), "isolated reference helper source is not present")
    def test_transitive_helper_hash_is_recorded_as_environment_evidence_only(self) -> None:
        raw = HELPER_PATH.read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), PLANTCV_HELPER_REFERENCE_ENV_SHA256)
        # The helper source is a transitive installed file omitted from the
        # committed 15-file source audit, so this observed digest is not a
        # substitute for a direct upstream source pin.
        config = json.loads(SOURCE_CONFIG.read_text(encoding="utf-8"))
        source_paths = {record["path"] for record in config["files"]}
        self.assertNotIn("plantcv/plantcv/_helpers.py", source_paths)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA morphology primitives require an available CUDA device")
@functional_units
class PlantCVMorphologyPrimitiveTests(unittest.TestCase):
    def check_test_kernel_and_iteration_domain_matches_reference_runner_limits(self) -> None:
        self.assertEqual((MIN_KERNEL_SIZE, MAX_KERNEL_SIZE), (2, 255))
        self.assertEqual((MIN_ITERATIONS, MAX_ITERATIONS), (1, 64))
        self.assertEqual(MAX_RASTER_PIXELS, 64_000_000)

    def check_test_erode_and_dilate_preserve_gray_values_and_cuda_residency(self) -> None:
        gray = torch.tensor(
            [[0, 1, 16, 127, 128, 254, 255],
             [255, 220, 180, 128, 90, 10, 0],
             [12, 60, 100, 140, 180, 240, 255]],
            dtype=torch.uint8,
            device="cuda",
        )
        before = gray.clone()
        for primitive in (erode, dilate):
            result = primitive(gray, 2, 1)
            self.assertEqual(result.device, gray.device)
            self.assertEqual(result.dtype, torch.uint8)
            self.assertEqual(tuple(result.shape), tuple(gray.shape))
            self.assertTrue(torch.all(result <= 255))
        torch.testing.assert_close(gray, before)

    def check_test_noncontiguous_view_matches_contiguous_input_and_retains_shape(self) -> None:
        base = torch.arange(0, 120, dtype=torch.uint8, device="cuda").reshape(5, 24)
        view = base[:, ::3]
        self.assertFalse(view.is_contiguous())
        for primitive in (erode, dilate):
            actual = primitive(view, 4, 2)
            expected = primitive(view.contiguous(), 4, 2)
            torch.testing.assert_close(actual, expected)
            self.assertEqual(tuple(actual.shape), tuple(view.shape))
            self.assertEqual(actual.device, view.device)

    def check_test_constants_retain_extremes_at_neutral_full_canvas_borders(self) -> None:
        for value in (0, 255):
            gray = torch.full((3, 5), value, dtype=torch.uint8, device="cuda")
            for primitive in (erode, dilate):
                torch.testing.assert_close(primitive(gray, 5, 3), gray)

    def check_test_input_parameters_and_resource_domain_are_strict(self) -> None:
        valid = torch.arange(16, dtype=torch.uint8, device="cuda").reshape(4, 4)
        cpu = valid.cpu()
        with self.assertRaisesRegex(ValueError, "CUDA"):
            erode(cpu, 3)
        with self.assertRaisesRegex(TypeError, "uint8"):
            dilate(valid.float(), 3)
        with self.assertRaises(TypeError):
            erode([[1, 2], [3, 4]], 3)
        for malformed in (
            valid[0],
            valid.unsqueeze(0),
            torch.empty((0, 4), dtype=torch.uint8, device="cuda"),
            torch.empty((4, 0), dtype=torch.uint8, device="cuda"),
        ):
            with self.subTest(shape=tuple(malformed.shape)), self.assertRaises(ValueError):
                erode(malformed, 3)
        with self.assertRaises(TypeError):
            erode(
                torch.sparse_coo_tensor(
                    [[0], [0]], [1], (4, 4), device="cuda", check_invariants=True,
                ),
                3,
            )
        overlapping = valid.as_strided((3, 3), (1, 1))
        with self.assertRaisesRegex(ValueError, "overlapping"):
            erode(overlapping, 3)
        for value in (True, 3.0, "3"):
            with self.subTest(ksize=value), self.assertRaises(TypeError):
                erode(valid, value)
        for value in (1, 256):
            with self.subTest(ksize=value), self.assertRaises(ValueError):
                erode(valid, value)
        for value in (True, 1.0, 0, 65):
            with self.subTest(iterations=value), self.assertRaises((TypeError, ValueError)):
                dilate(valid, 3, value)

        # An expanded zero-stride view makes the 64-million-pixel boundary
        # test cheap; validation must reject before any image operation runs.
        oversized = valid[:1, :1].expand(8000, 8001)
        with self.assertRaises(MorphologyResourceError):
            erode(oversized, 3)

    def check_test_accepted_kernel_and_iteration_values_affect_calculation(self) -> None:
        generator = torch.Generator(device="cuda").manual_seed(0xE20DE)
        gray = torch.randint(0, 256, (11, 13), dtype=torch.uint8, device="cuda", generator=generator)
        self.assertFalse(torch.equal(erode(gray, 2, 1), erode(gray, 3, 1)))
        self.assertFalse(torch.equal(dilate(gray, 2, 1), dilate(gray, 3, 1)))
        light_spot = torch.zeros((15, 15), dtype=torch.uint8, device="cuda")
        light_spot[7, 7] = 255
        dark_spot = torch.full((15, 15), 255, dtype=torch.uint8, device="cuda")
        dark_spot[7, 7] = 0
        self.assertFalse(torch.equal(dilate(light_spot, 3, 1), dilate(light_spot, 3, 2)))
        self.assertFalse(torch.equal(erode(dark_spot, 3, 1), erode(dark_spot, 3, 2)))

    def test_validation_and_recovery(self):
        """1 related PlantCVMorphologyPrimitiveTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_kernel_and_iteration_domain_matches_reference_runner_limits)

    def test_workflow_lifecycle(self):
        """5 related PlantCVMorphologyPrimitiveTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_erode_and_dilate_preserve_gray_values_and_cuda_residency)
        self.scenario(self.check_test_noncontiguous_view_matches_contiguous_input_and_retains_shape)
        self.scenario(self.check_test_constants_retain_extremes_at_neutral_full_canvas_borders)
        self.scenario(self.check_test_input_parameters_and_resource_domain_are_strict)
        self.scenario(self.check_test_accepted_kernel_and_iteration_values_affect_calculation)


@unittest.skipUnless(torch.cuda.is_available() and REFERENCE_ENV_READY, "CUDA and the isolated pinned PlantCV environment are required")
@functional_units
class PlantCVMorphologyRunnerComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runner = _reference_runner()

    def _compare(self, operation: str, image: np.ndarray, ksize: int, iterations: int) -> np.ndarray:
        result = self.runner.run(
            f"morphology.{operation}",
            {"gray": image},
            {"ksize": ksize, "iterations": iterations},
        )
        primitive = erode if operation == "erode" else dilate
        actual = primitive(torch.from_numpy(np.ascontiguousarray(image)).cuda(), ksize, iterations)
        self.assertEqual(actual.device.type, "cuda")
        self.assertEqual(actual.dtype, torch.uint8)
        self.assertEqual(tuple(actual.shape), image.shape)
        output = actual.detach().cpu().numpy()
        np.testing.assert_array_equal(output, result.arrays["mask"])
        return output

    def check_test_even_anchor_odd_anchor_small_arrays_and_repeated_passes_match(self) -> None:
        edge = np.array(
            [[0, 1, 16, 127, 128],
             [255, 220, 180, 10, 0],
             [12, 60, 100, 240, 255],
             [80, 40, 200, 160, 20]],
            dtype=np.uint8,
        )
        narrow = np.array([[0, 255, 32], [220, 1, 180]], dtype=np.uint8)
        one_axis = np.array([[0, 255, 1, 254]], dtype=np.uint8)
        cases = (
            ("erode", edge, 2, 1),       # even-kernel lower/right anchor and image border
            ("dilate", edge, 2, 1),
            ("erode", edge, 4, 2),       # even kernel, repeated passes
            ("dilate", edge, 4, 2),
            ("erode", edge, 3, 3),       # odd kernel, repeated passes
            ("dilate", edge, 3, 3),
            ("erode", narrow, 5, 2),     # kernel wider than the input
            ("dilate", narrow, 5, 2),
            ("erode", one_axis, 255, 1), # one-pixel axis and maximum kernel
            ("dilate", one_axis, 255, 1),
        )
        for operation, image, kernel, iterations in cases:
            with self.subTest(operation=operation, ksize=kernel, iterations=iterations, shape=image.shape):
                self._compare(operation, image, kernel, iterations)

    def check_test_largest_public_kernel_has_bounded_cuda_memory_and_matches_runner(self) -> None:
        rng = np.random.default_rng(0xE20DE)
        image = rng.integers(0, 256, size=(768, 1024), dtype=np.uint8)
        gray = torch.from_numpy(image).cuda()
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated(gray.device)
        torch.cuda.reset_peak_memory_stats(gray.device)
        for operation, primitive in (("erode", erode), ("dilate", dilate)):
            with self.subTest(operation=operation):
                result = self.runner.run(
                    f"morphology.{operation}",
                    {"gray": image},
                    {"ksize": 255, "iterations": 2},
                )
                actual = primitive(gray, 255, 2)
                torch.cuda.synchronize()
                np.testing.assert_array_equal(actual.cpu().numpy(), result.arrays["mask"])
        peak_delta = torch.cuda.max_memory_allocated(gray.device) - baseline
        self.assertLess(peak_delta, 256 * 1024 * 1024, f"unexpected CUDA workspace: {peak_delta:,} bytes")

    def check_test_real_fixture_raster_matches_the_pinned_runner(self) -> None:
        import cv2

        fixture = ROOT / "images" / "IMG_9632c.JPG"
        image = cv2.imread(str(fixture), cv2.IMREAD_GRAYSCALE)
        self.assertIsNotNone(image)
        self.assertEqual(image.dtype, np.uint8)
        # This compares morphology over the exact same fixture-derived raster;
        # it does not assert grayscale decoder parity across OpenCV versions.
        self._compare("erode", image, 3, 2)
        self._compare("dilate", image, 4, 2)

    def test_workflow_lifecycle(self):
        """3 related PlantCVMorphologyRunnerComparisonTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_even_anchor_odd_anchor_small_arrays_and_repeated_passes_match)
        self.scenario(self.check_test_largest_public_kernel_has_bounded_cuda_memory_and_matches_runner)
        self.scenario(self.check_test_real_fixture_raster_matches_the_pinned_runner)


if __name__ == "__main__":
    unittest.main()
