from __future__ import annotations

from tests.functional_units import functional_units

import hashlib
import json
from pathlib import Path
import unittest

import numpy as np
import torch

from seedvision.cuda.plantcv_thresholds import (
    PLANTCV_COMMIT,
    PLANTCV_THRESHOLD_SOURCE_SHA256,
    PLANTCV_VERSION,
    MIN_OTSU_RELATIVE_PARTITION_GAP,
    OtsuAmbiguityError,
    AdaptiveMeanPrecisionError,
    _otsu_cutoff_from_histogram,
    _adaptive_mean_condition,
    threshold_binary,
    threshold_mean,
    threshold_otsu,
)


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PYTHON = ROOT / "artifacts" / "plantcv-reference-env" / "Scripts" / "python.exe"
SOURCE_ROOT = ROOT / "artifacts" / "reference-sources" / "plantcv-v4.11.3"
SOURCE_PATH = SOURCE_ROOT / "plantcv" / "plantcv" / "threshold" / "threshold_methods.py"
SOURCE_CONFIG = ROOT / "config" / "peer_reference_sources.json"
REFERENCE_ENV_READY = all(
    path.exists()
    for path in (
        REFERENCE_PYTHON,
        ROOT / "artifacts" / "reference-sources" / "plantcv-source-audit.json",
        ROOT / "artifacts" / "reference-sources" / "plantcv-environment.json",
        SOURCE_PATH,
        ROOT / "config" / "plantcv-reference-requirements.txt",
    )
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA threshold primitives require an available CUDA device")
@functional_units
class PlantCVThresholdPrimitiveTests(unittest.TestCase):
    def check_test_mean_certificate_covers_both_polarities_fractional_offsets_and_safe_disagreement(self):
        # One exact sum has f32 mean216 and rational/double mean217.
        sums = torch.tensor([[14_077_913]],dtype=torch.int64,device='cuda')
        for polarity,offset,source_value in (('light',1.5,215),('dark',1.5,216),
                ('light',-0.5,217),('dark',-0.5,218)):
            delta = int(np.ceil(offset) if polarity=='light' else np.floor(offset))
            source = torch.tensor([[source_value]],dtype=torch.uint8,device='cuda')
            with self.assertRaises(AdaptiveMeanPrecisionError):
                _adaptive_mean_condition(sums,source,255,delta,polarity,frame_shape=(255,255))
        for polarity in ('light','dark'):
            source = torch.tensor([[0]],dtype=torch.uint8,device='cuda')
            actual = _adaptive_mean_condition(sums,source,255,2,polarity,frame_shape=(255,255))
            self.assertEqual(bool(actual.item()),polarity=='dark')

    def check_test_source_pin_matches_committed_registry_and_available_source_bytes(self) -> None:
        config = json.loads(SOURCE_CONFIG.read_text(encoding="utf-8"))
        source_record = next(
            item for item in config["files"]
            if item["path"] == "plantcv/plantcv/threshold/threshold_methods.py"
        )
        self.assertEqual(PLANTCV_VERSION, "4.11.3")
        self.assertEqual(PLANTCV_COMMIT, "77e3322fd0cd5a9bef18bb99402f20c8f2db87a9")
        self.assertEqual(config["version"], PLANTCV_VERSION)
        self.assertEqual(config["commit"], PLANTCV_COMMIT)
        self.assertEqual(source_record["sha256"], PLANTCV_THRESHOLD_SOURCE_SHA256)
        if SOURCE_PATH.is_file():
            self.assertEqual(hashlib.sha256(SOURCE_PATH.read_bytes()).hexdigest(), PLANTCV_THRESHOLD_SOURCE_SHA256)

    def check_test_binary_threshold_is_strict_at_the_cutoff_and_preserves_cuda_output(self) -> None:
        gray = torch.tensor([[0, 127, 128, 255]], dtype=torch.uint8, device="cuda")
        light = threshold_binary(gray, 127, "light")
        dark = threshold_binary(gray, 127, "dark")
        torch.testing.assert_close(light, torch.tensor([[0, 0, 255, 255]], dtype=torch.uint8, device="cuda"))
        torch.testing.assert_close(dark, torch.tensor([[255, 255, 0, 0]], dtype=torch.uint8, device="cuda"))
        self.assertEqual(light.device, gray.device)
        self.assertEqual(dark.device, gray.device)
        self.assertEqual(light.dtype, torch.uint8)
        self.assertEqual(set(light.unique().cpu().tolist()), {0, 255})
        torch.testing.assert_close(threshold_binary(gray, 0), torch.tensor([[0, 255, 255, 255]], dtype=torch.uint8, device="cuda"))
        torch.testing.assert_close(threshold_binary(gray, 255), torch.zeros_like(gray))

    def check_test_otsu_accepts_plateaus_and_rejects_tied_or_near_partition_scores(self) -> None:
        tied = [0] * 256
        tied[10] = tied[200] = 9
        self.assertEqual(_otsu_cutoff_from_histogram(tied), 10)
        # Distinct occupied-bin partitions can be mathematically tied while
        # the pinned default IPP and generic OpenCV paths choose differently.
        for counts in ((1, 1, 1), (1, 2, 1), (1, 4, 1), (2, 2, 2), (2, 5, 2)):
            histogram = list(counts) + [0] * (256 - len(counts))
            with self.subTest(counts=counts), self.assertRaises(OtsuAmbiguityError):
                _otsu_cutoff_from_histogram(histogram)

        self.assertEqual(MIN_OTSU_RELATIVE_PARTITION_GAP, 1e-6)
        large_unique_near_tie = [10_000_000, 10_000_000, 10_000_001] + [0] * 253
        with self.assertRaisesRegex(OtsuAmbiguityError, "1e-6 relative score gap"):
            _otsu_cutoff_from_histogram(large_unique_near_tie)
        large_separated = [100_000, 100_000, 100_100] + [0] * 253
        self.assertEqual(_otsu_cutoff_from_histogram(large_separated), 1)

        # OpenCV skips a candidate partition when either class probability is
        # below FLT_EPSILON. These compact count arrays avoid expanding tens of
        # millions of pixels in the unit-level scorer check.
        rare_class_cases = (
            ((127, 128), (20_000_000, 1)),
            ((127, 128), (1, 20_000_000)),
            ((0, 127, 255), (20_000_000, 1, 1)),
            ((0, 127, 255), (1, 20_000_000, 1)),
            ((0, 127, 255), (1, 1, 20_000_000)),
        )
        for bins, counts in rare_class_cases:
            histogram = [0] * 256
            for bin_value, count in zip(bins, counts):
                histogram[bin_value] = count
            with self.subTest(bins=bins, counts=counts):
                self.assertEqual(_otsu_cutoff_from_histogram(histogram), 0)
        epsilon_boundary = [0] * 256
        epsilon_boundary[127] = 8_388_607
        epsilon_boundary[128] = 1
        self.assertEqual(_otsu_cutoff_from_histogram(epsilon_boundary), 127)
        below_epsilon = [0] * 256
        below_epsilon[127] = 8_388_608
        below_epsilon[128] = 1
        self.assertEqual(_otsu_cutoff_from_histogram(below_epsilon), 0)

        constant = [0] * 256
        constant[173] = 5
        self.assertEqual(_otsu_cutoff_from_histogram(constant), 0)

        gray = torch.full((3, 5), 173, dtype=torch.uint8, device="cuda")
        torch.testing.assert_close(threshold_otsu(gray, "light"), torch.full_like(gray, 255))
        torch.testing.assert_close(threshold_otsu(gray, "dark"), torch.zeros_like(gray))
        zero = torch.zeros((2, 4), dtype=torch.uint8, device="cuda")
        torch.testing.assert_close(threshold_otsu(zero, "light"), torch.zeros_like(zero))
        torch.testing.assert_close(threshold_otsu(zero, "dark"), torch.full_like(zero, 255))

    def check_test_adaptive_mean_polarity_rounding_offset_and_kernel_affect_results(self) -> None:
        constant = torch.full((5, 7), 100, dtype=torch.uint8, device="cuda")
        torch.testing.assert_close(threshold_mean(constant, 3, 0.1, "light"), torch.full_like(constant, 255))
        torch.testing.assert_close(threshold_mean(constant, 3, 0.1, "dark"), torch.full_like(constant, 255))
        torch.testing.assert_close(threshold_mean(constant, 3, -0.1, "light"), torch.zeros_like(constant))
        torch.testing.assert_close(threshold_mean(constant, 3, 1.5, "dark"), torch.zeros_like(constant))

        ramp = torch.arange(0, 81, dtype=torch.uint8, device="cuda").reshape(9, 9) * 3
        small_kernel = threshold_mean(ramp, 3, 1.25, "light")
        large_kernel = threshold_mean(ramp, 5, 1.25, "light")
        shifted_offset = threshold_mean(ramp, 3, -2.0, "light")
        self.assertFalse(torch.equal(small_kernel, large_kernel))
        self.assertFalse(torch.equal(small_kernel, shifted_offset))
        self.assertEqual(small_kernel.device, ramp.device)
        self.assertEqual(small_kernel.dtype, torch.uint8)
        self.assertEqual(set(small_kernel.unique().cpu().tolist()), {0, 255})

    def check_test_noncontiguous_and_small_edge_inputs_remain_supported(self) -> None:
        base = torch.tensor(
            [[0, 12, 40, 80, 120, 160, 200, 255],
             [255, 200, 160, 120, 80, 40, 12, 0],
             [20, 60, 100, 140, 180, 220, 240, 250],
             [250, 240, 220, 180, 140, 100, 60, 20]],
            dtype=torch.uint8,
            device="cuda",
        )
        view = base[:, ::2]
        self.assertFalse(view.is_contiguous())
        for operation in (
            lambda image: threshold_binary(image, 100, "dark"),
            lambda image: threshold_mean(image, 5, -0.5, "light"),
            lambda image: threshold_otsu(image, "light"),
        ):
            actual = operation(view)
            expected = operation(view.contiguous())
            torch.testing.assert_close(actual, expected)
            self.assertEqual(actual.shape, view.shape)
            self.assertEqual(actual.device, view.device)

    def check_test_invalid_inputs_and_unsupported_parameters_fail_explicitly(self) -> None:
        cpu = torch.zeros((3, 4), dtype=torch.uint8)
        valid = cpu.cuda()
        with self.assertRaisesRegex(ValueError, "CUDA"):
            threshold_binary(cpu, 100)
        with self.assertRaisesRegex(TypeError, "uint8"):
            threshold_binary(valid.float(), 100)
        for malformed in (valid[0], valid.unsqueeze(0), torch.empty((0, 4), dtype=torch.uint8, device="cuda")):
            with self.subTest(shape=tuple(malformed.shape)), self.assertRaises(ValueError):
                threshold_binary(malformed, 100)
        for bad_threshold in (True, 1.5, -1, 256):
            with self.subTest(threshold=bad_threshold), self.assertRaises((TypeError, ValueError)):
                threshold_binary(valid, bad_threshold)
        for bad_object_type in ("medium", None):
            with self.subTest(object_type=bad_object_type), self.assertRaises((TypeError, ValueError)):
                threshold_binary(valid, 100, bad_object_type)
        for bad_kernel in (True, 2, 256, 257):
            with self.subTest(ksize=bad_kernel), self.assertRaises((TypeError, ValueError)):
                threshold_mean(valid, bad_kernel, 0)
        for bad_offset in (True, float("nan"), float("inf"), 256.0):
            with self.subTest(offset=bad_offset), self.assertRaises((TypeError, ValueError)):
                threshold_mean(valid, 3, bad_offset)
        with self.assertRaisesRegex(ValueError, "non-empty"):
            threshold_otsu(torch.empty((0, 4), dtype=torch.uint8, device="cuda"))

    def test_workflow_lifecycle(self):
        """5 related PlantCVThresholdPrimitiveTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_mean_certificate_covers_both_polarities_fractional_offsets_and_safe_disagreement)
        self.scenario(self.check_test_source_pin_matches_committed_registry_and_available_source_bytes)
        self.scenario(self.check_test_binary_threshold_is_strict_at_the_cutoff_and_preserves_cuda_output)
        self.scenario(self.check_test_adaptive_mean_polarity_rounding_offset_and_kernel_affect_results)
        self.scenario(self.check_test_noncontiguous_and_small_edge_inputs_remain_supported)

    def test_validation_and_recovery(self):
        """2 related PlantCVThresholdPrimitiveTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_otsu_accepts_plateaus_and_rejects_tied_or_near_partition_scores)
        self.scenario(self.check_test_invalid_inputs_and_unsupported_parameters_fail_explicitly)


@unittest.skipUnless(torch.cuda.is_available() and REFERENCE_ENV_READY, "CUDA and isolated pinned PlantCV environment are required")
@functional_units
class PlantCVOriginalRunnerComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from seedvision.benchmarking.reference_runner import PlantCVReferenceRunner

        cls.runner = PlantCVReferenceRunner(REFERENCE_PYTHON)

    def _compare(self, operation: str, image: np.ndarray, parameters: dict, gpu_input: torch.Tensor, primitive):
        result = self.runner.run(operation, {"gray": image}, parameters)
        actual = primitive(gpu_input)
        self.assertEqual(actual.device.type, "cuda")
        self.assertEqual(actual.dtype, torch.uint8)
        np.testing.assert_array_equal(actual.detach().cpu().numpy(), result.arrays["mask"])
        return result

    def check_test_binary_supported_otsu_and_mean_match_the_pinned_isolated_runner(self) -> None:
        rng = np.random.default_rng(0x51EED)
        random_image = rng.integers(0, 256, size=(13, 19), dtype=np.uint8)
        edge_image = np.array(
            [[0, 0, 1, 127, 128, 254, 255],
             [255, 220, 180, 128, 90, 10, 0],
             [12, 12, 60, 100, 180, 240, 255],
             [200, 160, 120, 80, 40, 1, 0],
             [25, 75, 125, 175, 225, 250, 255]],
            dtype=np.uint8,
        )
        two_level = np.full((9, 11), 10, dtype=np.uint8)
        two_level[:, 5:] = 200
        constant = np.full((7, 9), 117, dtype=np.uint8)

        for cutoff in (0, 127, 255):
            for polarity in ("light", "dark"):
                self._compare(
                    "threshold.binary", random_image,
                    {"threshold": cutoff, "object_type": polarity},
                    torch.from_numpy(random_image).cuda(),
                    lambda tensor, cutoff=cutoff, polarity=polarity: threshold_binary(tensor, cutoff, polarity),
                )

        for image in (random_image, two_level, constant):
            for polarity in ("light", "dark"):
                self._compare(
                    "threshold.otsu", image, {"object_type": polarity},
                    torch.from_numpy(image).cuda(),
                    lambda tensor, polarity=polarity: threshold_otsu(tensor, polarity),
                )

        # Regression from the pinned PlantCV/OpenCV 4.11 runner. Default IPP
        # returns this mask, while the exact score has tied occupied partitions;
        # the CUDA API reports that ambiguity instead of claiming a cutoff.
        near_tie = np.array([[0, 0, 1, 2, 2]], dtype=np.uint8)
        near_tie_result = self.runner.run(
            "threshold.otsu", {"gray": near_tie}, {"object_type": "light"},
        )
        np.testing.assert_array_equal(
            near_tie_result.arrays["mask"],
            np.array([[0, 0, 0, 255, 255]], dtype=np.uint8),
        )
        with self.assertRaises(OtsuAmbiguityError):
            threshold_otsu(torch.from_numpy(near_tie).cuda(), "light")

        mean_cases = (
            (edge_image, 3, 0.5, "light"),
            (edge_image, 3, 0.5, "dark"),
            (edge_image, 3, -0.5, "light"),
            (edge_image, 3, -0.5, "dark"),
            (random_image, 5, 1.5, "light"),
            (random_image, 5, 1.5, "dark"),
            (random_image, 5, -1.5, "light"),
            (random_image, 5, -1.5, "dark"),
        )
        for image, ksize, offset, polarity in mean_cases:
            gpu_base = torch.from_numpy(image).cuda()
            gpu_view = gpu_base[:, ::1]
            self._compare(
                "threshold.mean", image,
                {"ksize": ksize, "offset": offset, "object_type": polarity},
                gpu_view,
                lambda tensor, ksize=ksize, offset=offset, polarity=polarity: threshold_mean(
                    tensor, ksize, offset, polarity,
                ),
            )

    def check_test_original_runner_comparison_includes_noncontiguous_cuda_view(self) -> None:
        image = np.array(
            [[0, 22, 44, 66, 88, 110, 132, 154, 176, 198, 220, 242],
             [255, 233, 211, 189, 167, 145, 123, 101, 79, 57, 35, 13],
             [4, 24, 64, 84, 124, 144, 184, 204, 244, 254, 254, 254],
             [254, 254, 244, 204, 184, 144, 124, 84, 64, 24, 4, 0]],
            dtype=np.uint8,
        )
        gpu_view = torch.from_numpy(image).cuda()[:, ::2]
        noncontiguous_image = image[:, ::2].copy()
        self.assertFalse(gpu_view.is_contiguous())
        self._compare(
            "threshold.mean", noncontiguous_image,
            {"ksize": 5, "offset": 0.25, "object_type": "dark"},
            gpu_view,
            lambda tensor: threshold_mean(tensor, 5, 0.25, "dark"),
        )

    def check_test_maximum_mean_kernel_certifies_edges_and_withholds_float32_half_case(self) -> None:
        rng = np.random.default_rng(0x255)
        image = rng.integers(0, 256, size=(768, 1024), dtype=np.uint8)
        # Place one exact 255x255 window whose integer mean is 216.500007689...
        # OpenCV's float32 scale produces 216.5f and ties-to-even rounds to 216.
        # This distinguishes the pinned boxFilter result from exact rational
        # half-up rounding while retaining random neighborhoods and replicated
        # outer-image borders in the same runner comparison.
        near_half_window = np.full((255, 255), 216, dtype=np.uint8)
        near_half_window.reshape(-1)[:32514] = 217
        near_half_window[127, 127] = 216
        image[128:383, 128:383] = near_half_window
        tensor = torch.from_numpy(image).cuda()
        # Stable small random frames still compare against the actual runner.
        edge = rng.integers(0,256,size=(19,23),dtype=np.uint8)
        for polarity in ("light", "dark"):
            self._compare(
                "threshold.mean", edge,
                {"ksize": 255, "offset": 0.5, "object_type": polarity},
                torch.as_tensor(edge,device='cuda'),
                lambda value, polarity=polarity: threshold_mean(value, 255, 0.5, polarity),
            )
        original = self.runner.run('threshold.mean',{'gray':image},
            {'ksize':255,'offset':0.5,'object_type':'light'})
        self.assertEqual(int(original.arrays['mask'][255,255]),255)
        # SIMD happens to choose the float32 byte here. The portable native
        # contract now withholds decisions that differ from the scalar tail.
        with self.assertRaises(AdaptiveMeanPrecisionError):
            threshold_mean(tensor,255,0.5,'light')

    def check_test_scalar_mean_tail_counterexample_is_not_silently_miscomputed(self) -> None:
        image = np.full((255,3),255,dtype=np.uint8)
        image[0,0],image[1,0],image[127,2] = 0,254,253
        original = self.runner.run('threshold.mean',{'gray':image},
            {'ksize':255,'offset':2,'object_type':'dark'})
        self.assertEqual(int(original.arrays['mask'][127,2]),255)
        tensor = torch.as_tensor(image,device='cuda')
        with self.assertRaisesRegex(AdaptiveMeanPrecisionError,'scalar/SIMD'):
            threshold_mean(tensor,255,2,'dark')
        np.testing.assert_array_equal(tensor.cpu().numpy(),image)

    def check_test_otsu_extreme_class_imbalance_matches_default_ipp_runner(self) -> None:
        bins = (127, 128)
        counts = (20_000_000, 1)
        image = np.concatenate(
            [np.full(count, value, dtype=np.uint8) for value, count in zip(bins, counts)]
        ).reshape(1, -1)
        self.assertEqual(image.nbytes, 20_000_001)
        result = self._compare(
            "threshold.otsu", image, {"object_type": "light"},
            torch.from_numpy(image).cuda(), lambda tensor: threshold_otsu(tensor, "light"),
        )
        self.assertEqual(int(np.count_nonzero(result.arrays["mask"] == 255)), image.size)

    def test_workflow_lifecycle(self):
        """4 related PlantCVOriginalRunnerComparisonTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_binary_supported_otsu_and_mean_match_the_pinned_isolated_runner)
        self.scenario(self.check_test_original_runner_comparison_includes_noncontiguous_cuda_view)
        self.scenario(self.check_test_scalar_mean_tail_counterexample_is_not_silently_miscomputed)
        self.scenario(self.check_test_otsu_extreme_class_imbalance_matches_default_ipp_runner)

    def test_validation_and_recovery(self):
        """1 related PlantCVOriginalRunnerComparisonTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_maximum_mean_kernel_certifies_edges_and_withholds_float32_half_case)


if __name__ == "__main__":
    unittest.main()
