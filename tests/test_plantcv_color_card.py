from __future__ import annotations

from tests.functional_units import functional_units

import json
import os
from pathlib import Path
import subprocess
import unittest

import numpy as np
import torch

from seedvision.cuda.plantcv_color_card import (
    ColorCardCancelled, ColorCardError, ColorCardResourceError,
    create_color_card_mask, std_color_matrix,
)


_ROOT = Path(__file__).resolve().parents[1]
_REFERENCE = Path(os.environ.get(
    'SEEDVISION_PLANTCV_REFERENCE_PYTHON',
    str(_ROOT/'artifacts'/'plantcv-reference-env'/'Scripts'/'python.exe'),
))
_EXPECTED_COLOR_SOURCE = 'c0c3cf636db7a881aef820743501614fc3dccd0c1b732ac7df5d54a7607cce68'
_EXPECTED_ROI_SOURCE = '60b71913da5dc807938b035d6655fec99cbb1ed8df8b2d4dba1ae51abeaa553b'
_REFERENCE_SCRIPT = r'''
import cv2, hashlib, importlib, importlib.metadata, json, sys
from pathlib import Path
import numpy as np
from plantcv import plantcv as pcv
assert importlib.metadata.version('numpy') == '2.2.6'
assert importlib.metadata.version('opencv-python') == '4.11.0.86'
pcv.params.debug = None
p = json.loads(sys.stdin.read())
if p['operation'] == 'matrix':
    result = pcv.transform.std_color_matrix(pos=p['pos']).tolist()
else:
    image = np.zeros((p['height'], p['width'], 3), dtype=np.uint8)
    try:
        result = pcv.transform.create_color_card_mask(
            rgb_img=image, radius=p['radius'], start_coord=tuple(p['start_coord']),
            spacing=tuple(p['spacing']), nrows=p['nrows'], ncols=p['ncols'],
            exclude=p['exclude']).tolist()
    except Exception as error:
        if not p.get('allow_error'):
            raise
        result = {'error_type': type(error).__name__}
color_source = Path(pcv.transform.std_color_matrix.__code__.co_filename)
roi_module = importlib.import_module('plantcv.plantcv.roi.roi_methods')
roi_source = Path(roi_module.circle.__code__.co_filename)
def sha256(path): return hashlib.sha256(path.read_bytes()).hexdigest()
print(json.dumps({'result': result,
                  'plantcv_version': importlib.metadata.version('plantcv'),
                  'opencv_version': cv2.__version__,
                  'color_source_sha256': sha256(color_source),
                  'roi_source_sha256': sha256(roi_source)}))
'''


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA is required for color-card operations')
@functional_units
class PlantCVColorCardParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not _REFERENCE.is_file():
            raise AssertionError('Pinned PlantCV reference interpreter is required for card conformance.')

    def _reference(self, payload):
        completed = subprocess.run(
            [str(_REFERENCE), '-c', _REFERENCE_SCRIPT],
            input=json.dumps(payload), text=True, capture_output=True, timeout=90,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        result = json.loads(completed.stdout.strip().splitlines()[-1])
        self.assertEqual(result['plantcv_version'], '4.11.3')
        self.assertEqual(result['opencv_version'], '4.11.0')
        self.assertEqual(result['color_source_sha256'], _EXPECTED_COLOR_SOURCE)
        self.assertEqual(result['roi_source_sha256'], _EXPECTED_ROI_SOURCE)
        return result

    def check_test_standard_matrix_matches_all_four_public_orientations_and_flooring(self):
        for position in (0, 1, 2, 3, 1.9):
            reference = self._reference({'operation': 'matrix', 'pos': position})
            expected = np.asarray(reference['result'], dtype=np.float64)
            actual = std_color_matrix(position).cpu().numpy()
            np.testing.assert_array_equal(actual, expected)
        with self.assertRaises(ColorCardError):
            std_color_matrix(4)
        with self.assertRaises(ColorCardError):
            std_color_matrix(-0.1)

    def _compare_mask(self, height, width, radius, start, spacing, rows, cols, exclude):
        reference = self._reference({
            'operation': 'mask', 'height': height, 'width': width,
            'radius': radius, 'start_coord': list(start), 'spacing': list(spacing),
            'nrows': rows, 'ncols': cols, 'exclude': exclude,
        })
        actual = create_color_card_mask(
            torch.zeros((height, width, 3), dtype=torch.uint8, device='cuda'),
            radius, start, spacing, rows, cols, exclude=exclude,
        )
        np.testing.assert_array_equal(actual.cpu().numpy(), np.asarray(reference['result'], dtype=np.uint8))

    def check_test_filled_circles_match_public_mask_with_overlap_and_exclusions(self):
        self._compare_mask(18, 24, 2, (2, 2), (3, 3), 4, 6, [0, 5, 18])

    def check_test_radius_zero_one_and_larger_radii_match_exact_circle_bytes(self):
        for radius in (0, 1, 2, 3, 5, 8, 16, 31):
            edge = max(2, 2*radius+1)
            self._compare_mask(edge, edge, radius, (radius, radius), (1, 1), 2, 2, [])

    def check_test_empty_clipped_roi_preserves_public_failure(self):
        reference = self._reference({'operation': 'mask', 'height': 1, 'width': 1,
            'radius': 0, 'start_coord': [1, 0], 'spacing': [1, 1],
            'nrows': 1, 'ncols': 1, 'exclude': [0], 'allow_error': True})
        self.assertEqual(reference['result']['error_type'], 'IndexError')
        with self.assertRaisesRegex(ColorCardError, 'contains no pixels'):
            create_color_card_mask(torch.zeros((1, 1, 3), dtype=torch.uint8, device='cuda'),
                0, (1, 0), (1, 1), 1, 1, exclude=[0])

    def check_test_edge_clipping_negative_duplicate_exclusions_and_relabeling(self):
        self._compare_mask(8, 8, 2, (2, 2), (4, 4), 2, 2, [])
        self._compare_mask(12, 16, 1, (2, 2), (3, 3), 3, 4, [-1, 0, 0])

    def check_test_bounds_cancellation_and_resource_limits(self):
        image = torch.zeros((8, 8, 3), dtype=torch.uint8, device='cuda')
        with self.assertRaises(ColorCardError):
            create_color_card_mask(image, 2, (1, 1), (2, 2), 1, 1, exclude=[0])
        with self.assertRaises(ColorCardError):
            create_color_card_mask(image, 1, (2, 2), (2, 2), 1, 1, exclude=[4])
        with self.assertRaises(ColorCardCancelled):
            create_color_card_mask(image, 1, (2, 2), (3, 3), 2, 2, cancel=lambda: True)
        with self.assertRaises(ColorCardResourceError):
            create_color_card_mask(image, 1, (2, 2), (3, 3), 5, 6)
        with self.assertRaises(ColorCardResourceError):
            create_color_card_mask(image, 4097, (4097, 4097), (1, 1), 1, 1)

    def check_test_mid_raster_cancellation_preserves_image_and_exclusion_input(self):
        image = torch.arange(12*20*3, dtype=torch.int32, device='cuda').remainder(256)
        image = image.to(torch.uint8).reshape(12, 20, 3)
        before, exclusions, calls = image.clone(), [0, 2], 0

        def cancel_after_some_rows():
            nonlocal calls
            calls += 1
            return calls > 6

        with self.assertRaises(ColorCardCancelled):
            create_color_card_mask(image, 3, (4, 4), (5, 0), 1, 3,
                exclude=exclusions, cancel=cancel_after_some_rows)
        self.assertGreater(calls, 6)
        self.assertTrue(torch.equal(image, before))
        self.assertEqual(exclusions, [0, 2])

    def test_workflow_lifecycle(self):
        """4 related PlantCVColorCardParityTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_standard_matrix_matches_all_four_public_orientations_and_flooring)
        self.scenario(self.check_test_filled_circles_match_public_mask_with_overlap_and_exclusions)
        self.scenario(self.check_test_radius_zero_one_and_larger_radii_match_exact_circle_bytes)
        self.scenario(self.check_test_edge_clipping_negative_duplicate_exclusions_and_relabeling)

    def test_validation_and_recovery(self):
        """3 related PlantCVColorCardParityTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_empty_clipped_roi_preserves_public_failure)
        self.scenario(self.check_test_bounds_cancellation_and_resource_limits)
        self.scenario(self.check_test_mid_raster_cancellation_preserves_image_and_exclusion_input)


if __name__ == '__main__':
    unittest.main()
