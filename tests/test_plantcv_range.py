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

from seedvision.cuda.plantcv_range import (
    PLANTCV_COMMIT,
    PLANTCV_HELPER_REFERENCE_ENV_SHA256,
    PLANTCV_THRESHOLD_SOURCE_SHA256,
    PLANTCV_VERSION,
    RangeCancelledError,
    RangeResourceError,
    UnsupportedRangeModeError,
    _channels,
    threshold_custom_range,
)


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PYTHON = ROOT / "artifacts" / "plantcv-reference-env" / "Scripts" / "python.exe"
SOURCE_PATH = ROOT / "artifacts" / "reference-sources" / "plantcv-v4.11.3" / "plantcv" / "plantcv" / "threshold" / "threshold_methods.py"
HELPER_PATH = ROOT / "artifacts" / "plantcv-reference-env" / "Lib" / "site-packages" / "plantcv" / "plantcv" / "_helpers.py"
REFERENCE_ENV_READY = REFERENCE_PYTHON.is_file() and SOURCE_PATH.is_file() and HELPER_PATH.is_file()


def _reference_custom_range(image: np.ndarray, cases: list[dict]) -> dict[str, np.ndarray]:
    """Run the pinned public PlantCV function in its isolated interpreter."""
    child = textwrap.dedent(
        r"""
        import json, sys
        from pathlib import Path
        import numpy as np
        import plantcv.plantcv as pcv
        request_path, source_path, output_path = map(Path, sys.argv[1:4])
        requests = json.loads(request_path.read_text(encoding='utf-8'))
        with np.load(source_path, allow_pickle=False) as archive:
            image = archive['bgr']
            outputs = {}
            for item in requests:
                mask, masked = pcv.threshold.custom_range(
                    image, item['lower'], item['upper'], channel=item['channel'])
                outputs[item['name'] + '_mask'] = mask
                outputs[item['name'] + '_masked'] = masked
        np.savez_compressed(output_path, **outputs)
        """
    )
    with tempfile.TemporaryDirectory(prefix="seedfiddle-range-reference-") as directory:
        folder = Path(directory)
        input_path = folder / "input.npz"
        request_path = folder / "request.json"
        output_path = folder / "output.npz"
        np.savez_compressed(input_path, bgr=np.ascontiguousarray(image))
        request_path.write_text(json.dumps(cases), encoding="utf-8")
        subprocess.run(
            [str(REFERENCE_PYTHON), "-c", child, str(request_path), str(input_path), str(output_path)],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        with np.load(output_path, allow_pickle=False) as archive:
            return {key: archive[key].copy() for key in archive.files}


def _reference_channel_bytes(image: np.ndarray) -> dict[str, np.ndarray]:
    """Read the exact pinned byte conversions used within custom_range."""
    child = textwrap.dedent(
        r"""
        import sys
        from pathlib import Path
        import cv2
        import numpy as np
        from plantcv.plantcv import _helpers
        source_path, output_path = map(Path, sys.argv[1:3])
        with np.load(source_path, allow_pickle=False) as archive:
            image = archive['bgr']
            outputs = {
                'rgb': image.reshape(-1, 3),
                'hsv': cv2.cvtColor(image, cv2.COLOR_BGR2HSV).reshape(-1, 3),
                'lab': cv2.cvtColor(image, cv2.COLOR_BGR2LAB).reshape(-1, 3),
                'gray': cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).reshape(-1, 1),
                'cmyk': np.stack([
                    _helpers._rgb2cmyk(image, channel=channel).reshape(-1)
                    for channel in ('c', 'm', 'y', 'k')
                ], axis=1),
            }
        np.savez_compressed(output_path, **outputs)
        """
    )
    with tempfile.TemporaryDirectory(prefix="seedfiddle-range-bytes-") as directory:
        folder = Path(directory)
        input_path, output_path = folder / "input.npz", folder / "output.npz"
        np.savez_compressed(input_path, bgr=np.ascontiguousarray(image))
        subprocess.run(
            [str(REFERENCE_PYTHON), "-c", child, str(input_path), str(output_path)],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        with np.load(output_path, allow_pickle=False) as archive:
            return {key: archive[key].copy() for key in archive.files}


class PlantCVRangeContractTests(unittest.TestCase):
    def test_source_and_transitive_helper_pins_match_available_bytes(self) -> None:
        self.assertEqual(PLANTCV_VERSION, "4.11.3")
        self.assertEqual(PLANTCV_COMMIT, "77e3322fd0cd5a9bef18bb99402f20c8f2db87a9")
        self.assertEqual(
            PLANTCV_THRESHOLD_SOURCE_SHA256,
            "04a671632e7e4decf5a662779cd88473b5aff750554afeee98eabe02eaa49907",
        )
        self.assertEqual(
            PLANTCV_HELPER_REFERENCE_ENV_SHA256,
            "c05b0c944210657332ce307f2066caab55f2ebc8d8fea11988db3e476e7cdfba",
        )
        if SOURCE_PATH.is_file():
            self.assertEqual(hashlib.sha256(SOURCE_PATH.read_bytes()).hexdigest(), PLANTCV_THRESHOLD_SOURCE_SHA256)
        if HELPER_PATH.is_file():
            self.assertEqual(hashlib.sha256(HELPER_PATH.read_bytes()).hexdigest(), PLANTCV_HELPER_REFERENCE_ENV_SHA256)


@unittest.skipUnless(REFERENCE_ENV_READY, "isolated pinned PlantCV environment is unavailable")
class PlantCVRangeConversionTests(unittest.TestCase):
    def test_channel_byte_converters_match_pinned_opencv_and_plantcv_helpers(self) -> None:
        rng = np.random.default_rng(0xC04A)
        image = rng.integers(0, 256, size=(137, 193, 3), dtype=np.uint8)
        # Cover the complete neutral axis and OpenCV's common exact hue axes.
        neutral = np.arange(256, dtype=np.uint8)
        image.reshape(-1, 3)[:256] = np.stack((neutral, neutral, neutral), axis=1)
        image.reshape(-1, 3)[256:259] = np.array(
            [[0, 0, 255], [0, 255, 0], [255, 0, 0]], dtype=np.uint8
        )
        reference = _reference_channel_bytes(image)
        pixels = torch.from_numpy(image.reshape(-1, 3))
        for mode, key in (("RGB", "rgb"), ("HSV", "hsv"), ("LAB", "lab"), ("GRAY", "gray"), ("CMYK", "cmyk")):
            with self.subTest(mode=mode):
                actual = _channels(pixels, mode).numpy()
                np.testing.assert_array_equal(actual, reference[key])


@unittest.skipUnless(torch.cuda.is_available(), "custom-range CUDA primitive requires an available CUDA device")
class PlantCVRangeCudaTests(unittest.TestCase):
    def test_cmyk_cuda_true_division_matches_every_pinned_channel_byte(self) -> None:
        # The original CPU conversion test cannot detect CUDA's scalar-divisor
        # reciprocal path. This pattern also reproduces a dual-channel mask
        # disagreement at a truncating byte boundary.
        yy, xx = np.indices((275, 715))
        rgb = np.stack(((7*xx+3*yy) % 256, (11*yy+xx) % 256,
                        (13*xx+5*yy) % 256), axis=-1).astype(np.uint8)
        image = np.ascontiguousarray(rgb[..., ::-1])
        neutral = np.arange(256, dtype=np.uint8)
        image.reshape(-1, 3)[:256] = np.stack((neutral, neutral, neutral), axis=1)
        expected = _reference_channel_bytes(image)['cmyk']
        source = torch.tensor(image.reshape(-1, 3), device='cuda')
        actual = _channels(source, 'CMYK')
        self.assertEqual(actual.device, source.device)
        np.testing.assert_array_equal(actual.cpu().numpy(), expected)
        np.testing.assert_array_equal(source.cpu().numpy(), image.reshape(-1, 3))

    def test_raw_bgr_slots_inclusive_bounds_gray_alias_and_masked_pixels_match_reference(self) -> None:
        image = np.array(
            [
                [[0, 0, 0], [10, 20, 30], [11, 20, 30], [255, 255, 255]],
                [[255, 0, 0], [0, 255, 0], [0, 0, 255], [128, 128, 128]],
            ],
            dtype=np.uint8,
        )
        cases = [
            {"name": "raw", "channel": "rGb", "lower": [10, 20, 30], "upper": [10, 20, 30]},
            {"name": "gray", "channel": "GRAY", "lower": [29], "upper": [150]},
            {"name": "grey", "channel": "GREY", "lower": [0], "upper": [255]},
        ]
        expected = _reference_custom_range(image, cases)
        gpu = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).cuda()
        for case in cases:
            with self.subTest(channel=case["channel"]):
                mask, masked = threshold_custom_range(gpu, case["channel"], case["lower"], case["upper"])
                np.testing.assert_array_equal(mask.cpu().numpy(), expected[case["name"] + "_mask"])
                np.testing.assert_array_equal(masked.cpu().numpy(), expected[case["name"] + "_masked"].transpose(2, 0, 1)[None])
                self.assertEqual(mask.device.type, "cuda")
                self.assertEqual(masked.device.type, "cuda")
                self.assertEqual(mask.dtype, torch.uint8)

    @unittest.skipUnless(REFERENCE_ENV_READY, "isolated pinned PlantCV environment is unavailable")
    def test_rgb_hsv_cmyk_and_gray_match_actual_pinned_public_chain(self) -> None:
        rng = np.random.default_rng(0xF044)
        image = rng.integers(0, 256, size=(31, 43, 3), dtype=np.uint8)
        # Include the color axes, achromatics, byte extrema and gray formula ties.
        special = np.array(
            [
                [0, 0, 0], [255, 255, 255], [0, 0, 255], [0, 255, 0], [255, 0, 0],
                [1, 2, 3], [10, 100, 200], [128, 127, 129], [254, 1, 127],
            ], dtype=np.uint8,
        )
        image.reshape(-1, 3)[: len(special)] = special
        cases = [
            {"name": "rgb", "channel": "RGB", "lower": [25, 40, 55], "upper": [220, 200, 180]},
            {"name": "hsv", "channel": "HSV", "lower": [7, 35, 40], "upper": [160, 235, 245]},
            {"name": "cmyk", "channel": "CMYK", "lower": [5, 10, 0, 0], "upper": [250, 245, 240, 220]},
            {"name": "gray", "channel": "gray", "lower": [65], "upper": [191]},
        ]
        expected = _reference_custom_range(image, cases)
        gpu = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).cuda()
        original = gpu.clone()
        for case in cases:
            with self.subTest(channel=case["channel"]):
                mask, masked = threshold_custom_range(gpu, case["channel"], case["lower"], case["upper"])
                np.testing.assert_array_equal(mask.cpu().numpy(), expected[case["name"] + "_mask"])
                np.testing.assert_array_equal(masked.cpu().numpy(), expected[case["name"] + "_masked"].transpose(2, 0, 1)[None])
                self.assertEqual(mask.device, gpu.device)
                self.assertEqual(masked.device, gpu.device)
        torch.testing.assert_close(gpu, original)

    @unittest.skipUnless(REFERENCE_ENV_READY, "isolated pinned PlantCV environment is unavailable")
    def test_lab_custom_range_and_masked_output_match_pinned_public_chain(self) -> None:
        rng = np.random.default_rng(0x1AB4)
        image = rng.integers(0, 256, size=(19, 23, 3), dtype=np.uint8)
        special = np.array(
            [
                [0, 0, 0], [255, 255, 255], [0, 0, 255], [0, 255, 0], [255, 0, 0],
                [1, 1, 1], [15, 16, 17], [127, 128, 129], [253, 254, 255],
            ], dtype=np.uint8,
        )
        image.reshape(-1, 3)[: len(special)] = special
        case = {"name": "lab", "channel": "LAB", "lower": [80, 115, 120], "upper": [210, 150, 165]}
        expected = _reference_custom_range(image, [case])
        gpu = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).cuda()
        mask, masked = threshold_custom_range(gpu, "LAB", case["lower"], case["upper"])
        np.testing.assert_array_equal(mask.cpu().numpy(), expected["lab_mask"])
        np.testing.assert_array_equal(masked.cpu().numpy(), expected["lab_masked"].transpose(2, 0, 1)[None])
        self.assertGreater(int((mask != 0).sum().item()), 0)
        self.assertLess(int((mask != 0).sum().item()), image.shape[0] * image.shape[1])

    def test_bounds_are_exact_integer_byte_sequences_and_reversed_bounds_are_valid(self) -> None:
        valid = torch.zeros((1, 3, 1, 1), dtype=torch.uint8, device="cuda")
        for bad in ((1.5, 255, 255), (True, 255, 255), (-1, 0, 0), (0, 0, 256)):
            with self.subTest(bound=bad), self.assertRaises((TypeError, ValueError)):
                threshold_custom_range(valid, "RGB", bad, (255, 255, 255))
        one = torch.tensor([[[[10]], [[20]], [[30]]]], dtype=torch.uint8, device="cuda")
        mask, masked = threshold_custom_range(one, "RGB", (21, 0, 0), (255, 255, 255))
        torch.testing.assert_close(mask, torch.zeros((1, 1), dtype=torch.uint8, device="cuda"))
        torch.testing.assert_close(masked, torch.zeros_like(one))

    def test_float32_byte_preparation_noncontiguous_input_and_cancellation(self) -> None:
        values = torch.tensor(
            [[[[10.49, 10.5, 11.5, 255.4]], [[20.0, 20.0, 20.0, 20.0]], [[30.0, 30.0, 30.0, 30.0]]]],
            dtype=torch.float32,
            device="cuda",
        )
        mask, masked = threshold_custom_range(values, "RGB", (10, 20, 30), (12, 20, 30))
        torch.testing.assert_close(mask, torch.tensor([[255, 255, 255, 0]], dtype=torch.uint8, device="cuda"))
        torch.testing.assert_close(masked[0, 0], torch.tensor([[10, 10, 12, 0]], dtype=torch.uint8, device="cuda"))
        with self.assertRaises(RangeCancelledError):
            threshold_custom_range(values, "RGB", (0, 0, 0), (255, 255, 255), cancel=lambda: True)
        values_before_cancel = values.clone()
        cancellation_checks = 0

        def cancel_after_first_tile() -> bool:
            nonlocal cancellation_checks
            cancellation_checks += 1
            return cancellation_checks >= 3

        # The entry check and first tile pass; cancellation then stops the
        # second tile after partial internal writes, without returning them.
        with patch("seedvision.cuda.plantcv_range.MAX_TILE_PIXELS", 2):
            with self.assertRaises(RangeCancelledError):
                threshold_custom_range(
                    values,
                    "RGB",
                    (0, 0, 0),
                    (255, 255, 255),
                    cancel=cancel_after_first_tile,
                )
        self.assertEqual(cancellation_checks, 3)
        torch.testing.assert_close(values, values_before_cancel)
        sliced_mask, sliced_masked = threshold_custom_range(
            values[..., ::2], "RGB", (0, 0, 0), (255, 255, 255)
        )
        self.assertEqual(tuple(sliced_mask.shape), (1, 2))
        self.assertEqual(tuple(sliced_masked.shape), (1, 3, 1, 2))
        interleaved_storage = torch.arange(18, dtype=torch.uint8, device="cuda").reshape(1, 3, 6)
        interleaved_view = torch.as_strided(
            interleaved_storage,
            size=(1, 3, 2, 2),
            stride=(18, 6, 2, 3),
        )
        with self.assertRaisesRegex(ValueError, "sorted-stride extent"):
            threshold_custom_range(
                interleaved_view, "RGB", (0, 0, 0), (255, 255, 255)
            )
        with self.assertRaisesRegex(ValueError, "finite"):
            bad = values.clone()
            bad[0, 0, 0, 0] = float("nan")
            threshold_custom_range(bad, "RGB", (0, 0, 0), (255, 255, 255))

    def test_invalid_shape_mode_and_resource_requests_fail_before_outputs(self) -> None:
        valid = torch.zeros((1, 3, 2, 2), dtype=torch.uint8, device="cuda")
        with self.assertRaisesRegex(ValueError, "1x3xHxW"):
            threshold_custom_range(valid[0], "RGB", (0, 0, 0), (255, 255, 255))
        with self.assertRaises(UnsupportedRangeModeError):
            threshold_custom_range(valid, "YUV", (0, 0, 0), (255, 255, 255))
        with self.assertRaises((TypeError, ValueError)):
            threshold_custom_range(valid, "CMYK", (0, 0, 0), (255, 255, 255))
        with self.assertRaisesRegex(ValueError, "exactly 3"):
            threshold_custom_range(valid, "LAB", (0, 0), (255, 255, 255))
        image = torch.empty((1, 3, 1, 1), dtype=torch.uint8, device="cuda")
        with patch("seedvision.cuda.plantcv_range.MAX_RASTER_PIXELS", 0):
            with patch("seedvision.cuda.plantcv_range.torch.empty") as allocate_output:
                with self.assertRaises(RangeResourceError):
                    threshold_custom_range(image, "RGB", (0, 0, 0), (255, 255, 255))
                allocate_output.assert_not_called()
