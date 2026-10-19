
from tests.functional_units import functional_units
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from seedvision.cuda.plantcv_correction import (
    CorrectionCancelled, affine_color_correction_cuda, correct_color_cuda,
    get_color_matrix_cuda,
)


_ROOT = Path(__file__).resolve().parents[1]
_REFERENCE = Path(os.environ.get('SEEDVISION_PLANTCV_REFERENCE_PYTHON',
    str(_ROOT/'artifacts'/'plantcv-reference-env'/'Scripts'/'python.exe')))
_SCRIPT = r'''
import sys, tempfile, importlib, hashlib
from importlib.metadata import version
from pathlib import Path
import numpy as np
from plantcv import plantcv as pcv
assert version('plantcv') == '4.11.3'
assert version('numpy') == '2.2.6'
assert version('opencv-python') == '4.11.0.86'
module = importlib.import_module('plantcv.plantcv.transform.color_correction')
assert hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() == 'c0c3cf636db7a881aef820743501614fc3dccd0c1b732ac7df5d54a7607cce68'
pcv.params.debug = None
values = np.load(sys.argv[2], allow_pickle=False)
operation = sys.argv[1]
if operation == 'matrix':
    headers, matrix = pcv.transform.get_color_matrix(values['image'], values['mask'])
    result = {'matrix': matrix, 'headers': np.asarray(headers)}
elif operation == 'affine':
    result = {'image': pcv.transform.affine_color_correction(values['image'], values['source'], values['target'])}
elif operation == 'polynomial':
    with tempfile.TemporaryDirectory() as output:
        target, source, matrix, image = pcv.transform.correct_color(values['target_image'], values['target_mask'],
            values['source_image'], values['source_mask'], output)
    result = {'target': target, 'source': source, 'matrix': matrix, 'image': image}
else:
    raise AssertionError(operation)
np.savez(sys.argv[3], **result)
'''


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA is required for native colour correction')
@functional_units
class PlantCVCorrectionTests(unittest.TestCase):
    def reference(self, operation, **arrays):
        if not _REFERENCE.is_file():
            self.fail('Pinned PlantCV reference interpreter is required for correction conformance.')
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            np.savez(path/'input.npz', **arrays)
            completed = subprocess.run([str(_REFERENCE), '-c', _SCRIPT, operation,
                str(path/'input.npz'), str(path/'output.npz')], capture_output=True, text=True, timeout=90)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            with np.load(path/'output.npz', allow_pickle=False) as result:
                return {name: result[name].copy() for name in result.files}

    def check_test_chip_rgb_means_and_headers_match_pinned_public_call(self):
        rng = np.random.default_rng(307)
        image = rng.integers(0, 256, (17, 23, 3), dtype=np.uint8)
        mask = rng.choice(np.array([0, 10, 70, 255], dtype=np.uint8), (17, 23))
        expected = self.reference('matrix', image=image, mask=mask)
        headers, matrix = get_color_matrix_cuda(torch.as_tensor(image, device='cuda'),
            torch.as_tensor(mask, device='cuda'))
        self.assertEqual(headers, tuple(expected['headers']))
        np.testing.assert_allclose(matrix, expected['matrix'], rtol=0, atol=2e-15)
        self.assertEqual(matrix[:, 0].tolist(), [10, 70, 255])

    def check_test_affine_pixels_match_pinned_public_call_with_one_byte_tolerance(self):
        rng = np.random.default_rng(308)
        image = rng.integers(0, 256, (19, 27, 3), dtype=np.uint8)
        image[0, :2] = [(0, 0, 0), (255, 255, 255)]
        source = np.column_stack((np.arange(24)*10+10, rng.uniform(.15, .85, (24, 3))))
        target = source.copy()
        target[:, 1:] = source[:, 1:] * [1.2, .8, 1.05] + [-.08, .05, .02]
        expected = self.reference('affine', image=image, source=source, target=target)
        result = affine_color_correction_cuda(torch.as_tensor(image, device='cuda'), source, target)
        self.assertEqual(result.corrected_bgr.device.type, 'cuda')
        self.assertEqual(result.corrected_bgr.dtype, torch.uint8)
        actual = result.corrected_bgr.cpu().numpy()
        difference = np.abs(actual.astype(np.int16) - expected['image'].astype(np.int16))
        self.assertLessEqual(int(difference.max()), 1)
        np.testing.assert_array_equal(image[0, :2], [(0, 0, 0), (255, 255, 255)])

    def check_test_polynomial_outputs_match_pinned_public_call(self):
        rng = np.random.default_rng(309)
        source = rng.integers(20, 235, (17, 25, 3), dtype=np.uint8)
        target = np.clip(source.astype(np.int16)+[9, -7, 5], 0, 255).astype(np.uint8)
        mask = np.zeros((17, 25), dtype=np.uint8)
        # Each small patch has a distinct colour; background remains zero.
        for index in range(24):
            row, column = divmod(index, 6)
            ys, xs = slice(1+row*4, 4+row*4), slice(1+column*4, 4+column*4)
            source[ys, xs] = rng.integers(20, 235, 3, dtype=np.uint8)
            target[ys, xs] = np.clip(source[ys, xs].astype(np.int16)+[9, -7, 5], 0, 255)
            mask[ys, xs] = (index+1)*10
        expected = self.reference('polynomial', source_image=source, target_image=target,
            source_mask=mask, target_mask=mask)
        result = correct_color_cuda(torch.as_tensor(target, device='cuda'),
            torch.as_tensor(mask, device='cuda'), torch.as_tensor(source, device='cuda'),
            torch.as_tensor(mask, device='cuda'))
        np.testing.assert_allclose(result.source_matrix, expected['source'], rtol=0, atol=2e-15)
        np.testing.assert_allclose(result.target_matrix, expected['target'], rtol=0, atol=2e-15)
        np.testing.assert_allclose(result.transformation_matrix, expected['matrix'], rtol=1e-8, atol=1e-8)
        difference = np.abs(result.corrected_bgr.cpu().numpy().astype(np.int16)
            - expected['image'].astype(np.int16))
        self.assertLessEqual(int(difference.max()), 1)

    def check_test_tiled_chip_reduction_and_affine_cancellation_preserve_input(self):
        from seedvision.cuda import plantcv_correction as module
        image = torch.arange(300, device='cuda').remainder(256).to(torch.uint8).reshape(10, 10, 3)
        mask = torch.zeros((10, 10), dtype=torch.uint8, device='cuda')
        mask[1:9, 1:9] = 10
        original = image.clone()
        expected = get_color_matrix_cuda(image, mask)[1]
        with patch.object(module, 'TILE_PIXELS', 7):
            np.testing.assert_array_equal(get_color_matrix_cuda(image, mask)[1], expected)
            calls = 0
            def cancel():
                nonlocal calls
                calls += 1
                return calls >= 5
            matrix = np.array([[10, 0, 0, 0], [20, 1, 0, 0], [30, 0, 1, 0], [40, 0, 0, 1]], dtype=float)
            with self.assertRaises(CorrectionCancelled):
                affine_color_correction_cuda(image, matrix, matrix, cancel=cancel)
        torch.testing.assert_close(image, original)

    def check_test_empty_and_missing_background_contracts_are_explicit(self):
        image = torch.zeros((5, 7, 3), dtype=torch.uint8, device='cuda')
        _, matrix = get_color_matrix_cuda(image, torch.zeros((5, 7), dtype=torch.uint8, device='cuda'))
        self.assertEqual(matrix.shape, (0, 4))
        with self.assertRaisesRegex(ValueError, 'background zero'):
            get_color_matrix_cuda(image, torch.ones((5, 7), dtype=torch.uint8, device='cuda'))
        with self.assertRaises(ValueError):
            get_color_matrix_cuda(image.cpu(), torch.zeros((5, 7), dtype=torch.uint8, device='cuda'))

    def check_test_invalid_matrices_and_preallocation_cancellation_return_no_output(self):
        from seedvision.cuda import plantcv_correction as module
        image = torch.zeros((5, 7, 3), dtype=torch.uint8, device='cuda')
        matrix = np.array([[10, .1, .2, .3], [20, .4, .5, .6]], dtype=np.float64)
        invalid = matrix.copy()
        invalid[0, 1] = np.nan
        with patch.object(module.torch, 'empty_like') as allocate:
            with self.assertRaises(ValueError):
                affine_color_correction_cuda(image, invalid, matrix)
            with self.assertRaises(ValueError):
                affine_color_correction_cuda(image, matrix[:1], matrix)
            with self.assertRaises(CorrectionCancelled):
                affine_color_correction_cuda(image, matrix, matrix, cancel=lambda: True)
            allocate.assert_not_called()

    def test_workflow_lifecycle(self):
        """3 related PlantCVCorrectionTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_chip_rgb_means_and_headers_match_pinned_public_call)
        self.scenario(self.check_test_affine_pixels_match_pinned_public_call_with_one_byte_tolerance)
        self.scenario(self.check_test_polynomial_outputs_match_pinned_public_call)

    def test_validation_and_recovery(self):
        """3 related PlantCVCorrectionTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_tiled_chip_reduction_and_affine_cancellation_preserve_input)
        self.scenario(self.check_test_empty_and_missing_background_contracts_are_explicit)
        self.scenario(self.check_test_invalid_matrices_and_preallocation_cancellation_return_no_output)


if __name__ == '__main__':
    unittest.main()
