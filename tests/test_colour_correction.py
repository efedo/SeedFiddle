
from tests.functional_units import functional_units
import unittest
from unittest.mock import patch

import numpy as np
import torch

from seedvision.cuda.colour_correction import (
    CorrectionCancelled, affine_colour_correction_cuda, colour_matrix_cuda, correct_colour_cuda,
)


def _chip_means(image, mask):
    """Independent NumPy chip means: [chip ID, R, G, B] normalised to [0, 1]."""
    rows = []
    for chip in np.unique(mask):
        if chip == 0:
            continue
        bgr = image[mask == chip].astype(np.float64).mean(axis=0)
        rows.append((chip, *(bgr[::-1] / 255.0)))
    return np.asarray(rows, np.float64).reshape(-1, 4)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA is required for colour correction')
@functional_units
class ColourCorrectionTests(unittest.TestCase):
    def check_test_chip_rgb_means_and_headers_match_numpy(self):
        rng = np.random.default_rng(307)
        image = rng.integers(0, 256, (17, 23, 3), dtype=np.uint8)
        mask = rng.choice(np.array([0, 10, 70, 255], dtype=np.uint8), (17, 23))
        headers, matrix = colour_matrix_cuda(torch.as_tensor(image, device='cuda'),
                                             torch.as_tensor(mask, device='cuda'))
        self.assertEqual(headers, ('chip_number', 'r_avg', 'g_avg', 'b_avg'))
        np.testing.assert_allclose(matrix, _chip_means(image, mask), rtol=0, atol=1e-12)
        self.assertEqual(matrix[:, 0].tolist(), [10, 70, 255])

    def check_test_affine_fit_recovers_a_known_transform(self):
        rng = np.random.default_rng(308)
        image = rng.integers(0, 256, (19, 27, 3), dtype=np.uint8)
        image[0, :2] = [(0, 0, 0), (255, 255, 255)]
        source = np.column_stack((np.arange(24) * 10 + 10, rng.uniform(.15, .85, (24, 3))))
        gain, offset = np.array([1.2, .8, 1.05]), np.array([-.08, .05, .02])
        target = source.copy()
        target[:, 1:] = source[:, 1:] * gain + offset
        result = affine_colour_correction_cuda(torch.as_tensor(image, device='cuda'), source, target)
        self.assertEqual(result.corrected_bgr.device.type, 'cuda')
        self.assertEqual(result.corrected_bgr.dtype, torch.uint8)
        np.testing.assert_allclose(result.transformation_matrix, np.vstack((np.diag(gain), offset)),
                                   atol=1e-10)
        rgb = image[..., ::-1].astype(np.float64) / 255.0
        expected = (255.0 * np.clip(rgb * gain + offset, 0, 1)).astype(np.uint8)[..., ::-1]
        difference = np.abs(result.corrected_bgr.cpu().numpy().astype(np.int16) - expected.astype(np.int16))
        self.assertLessEqual(int(difference.max()), 1)
        np.testing.assert_array_equal(image[0, :2], [(0, 0, 0), (255, 255, 255)])

    def check_test_polynomial_fit_maps_source_chips_onto_target_chips(self):
        rng = np.random.default_rng(309)
        source = rng.integers(20, 235, (17, 25, 3), dtype=np.uint8)
        target = np.clip(source.astype(np.int16) + [9, -7, 5], 0, 255).astype(np.uint8)
        mask = np.zeros((17, 25), dtype=np.uint8)
        # Each small patch has a distinct colour; background remains zero.
        for index in range(24):
            row, column = divmod(index, 6)
            ys, xs = slice(1 + row * 4, 4 + row * 4), slice(1 + column * 4, 4 + column * 4)
            source[ys, xs] = rng.integers(20, 235, 3, dtype=np.uint8)
            target[ys, xs] = np.clip(source[ys, xs].astype(np.int16) + [9, -7, 5], 0, 255)
            mask[ys, xs] = (index + 1) * 10
        result = correct_colour_cuda(torch.as_tensor(target, device='cuda'),
                                     torch.as_tensor(mask, device='cuda'), torch.as_tensor(source, device='cuda'),
                                     torch.as_tensor(mask, device='cuda'))
        np.testing.assert_allclose(result.source_matrix, _chip_means(source, mask), atol=1e-12)
        np.testing.assert_allclose(result.target_matrix, _chip_means(target, mask), atol=1e-12)
        source_rgb = result.source_matrix[:, 1:]
        basis = np.concatenate((source_rgb, source_rgb**2, source_rgb**3), axis=1)
        # A constant per-channel offset is close to the nine-term fit's span.
        np.testing.assert_allclose(basis @ result.transformation_matrix[:, :3], result.target_matrix[:, 1:],
                                   atol=0.01)
        corrected = result.corrected_bgr.cpu().numpy()
        self.assertLessEqual(int(np.abs(corrected[mask > 0].astype(np.int16)
                                        - target[mask > 0].astype(np.int16)).max()), 4)

    def check_test_tiled_chip_reduction_and_affine_cancellation_preserve_input(self):
        from seedvision.cuda import colour_correction as module
        image = torch.arange(300, device='cuda').remainder(256).to(torch.uint8).reshape(10, 10, 3)
        mask = torch.zeros((10, 10), dtype=torch.uint8, device='cuda')
        mask[1:9, 1:9] = 10
        original = image.clone()
        expected = colour_matrix_cuda(image, mask)[1]
        with patch.object(module, 'TILE_PIXELS', 7):
            np.testing.assert_array_equal(colour_matrix_cuda(image, mask)[1], expected)
            calls = 0

            def cancel():
                nonlocal calls
                calls += 1
                return calls >= 5
            matrix = np.array([[10, 0, 0, 0], [20, 1, 0, 0], [30, 0, 1, 0], [40, 0, 0, 1]], dtype=float)
            with self.assertRaises(CorrectionCancelled):
                affine_colour_correction_cuda(image, matrix, matrix, cancel=cancel)
        torch.testing.assert_close(image, original)

    def check_test_empty_and_missing_background_contracts_are_explicit(self):
        image = torch.zeros((5, 7, 3), dtype=torch.uint8, device='cuda')
        _, matrix = colour_matrix_cuda(image, torch.zeros((5, 7), dtype=torch.uint8, device='cuda'))
        self.assertEqual(matrix.shape, (0, 4))
        with self.assertRaisesRegex(ValueError, 'background zero'):
            colour_matrix_cuda(image, torch.ones((5, 7), dtype=torch.uint8, device='cuda'))
        with self.assertRaises(ValueError):
            colour_matrix_cuda(image.cpu(), torch.zeros((5, 7), dtype=torch.uint8, device='cuda'))

    def check_test_invalid_matrices_and_preallocation_cancellation_return_no_output(self):
        from seedvision.cuda import colour_correction as module
        image = torch.zeros((5, 7, 3), dtype=torch.uint8, device='cuda')
        matrix = np.array([[10, .1, .2, .3], [20, .4, .5, .6]], dtype=np.float64)
        invalid = matrix.copy()
        invalid[0, 1] = np.nan
        with patch.object(module.torch, 'empty_like') as allocate:
            with self.assertRaises(ValueError):
                affine_colour_correction_cuda(image, invalid, matrix)
            with self.assertRaises(ValueError):
                affine_colour_correction_cuda(image, matrix[:1], matrix)
            with self.assertRaises(CorrectionCancelled):
                affine_colour_correction_cuda(image, matrix, matrix, cancel=lambda: True)
            allocate.assert_not_called()

    def test_workflow_lifecycle(self):
        """3 related ColourCorrectionTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_chip_rgb_means_and_headers_match_numpy)
        self.scenario(self.check_test_affine_fit_recovers_a_known_transform)
        self.scenario(self.check_test_polynomial_fit_maps_source_chips_onto_target_chips)

    def test_validation_and_recovery(self):
        """3 related ColourCorrectionTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_tiled_chip_reduction_and_affine_cancellation_preserve_input)
        self.scenario(self.check_test_empty_and_missing_background_contracts_are_explicit)
        self.scenario(self.check_test_invalid_matrices_and_preallocation_cancellation_return_no_output)


if __name__ == '__main__':
    unittest.main()
