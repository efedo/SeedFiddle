
from tests.functional_units import functional_units
import json
import math
import os
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

import numpy as np
import torch

from seedvision.cuda.plantcv_color import (
    ColorAnalysisCancelled, ColorAnalysisError, ColorAnalysisResourceError,
    _hue_statistics, _positive_byte_statistics, analyze_color_cuda,
)


_ROOT = Path(__file__).resolve().parents[1]
_REFERENCE = Path(os.environ.get('SEEDVISION_PLANTCV_REFERENCE_PYTHON',
    str(_ROOT/'artifacts'/'plantcv-reference-env'/'Scripts'/'python.exe')))
_SCRIPT = r'''
import importlib, json, sys, hashlib
from importlib.metadata import version
from pathlib import Path
import numpy as np
from plantcv import plantcv as pcv
assert version('plantcv') == '4.11.3'
assert version('numpy') == '2.2.6'
assert version('scipy') == '1.15.3'
assert version('opencv-python') == '4.11.0.86'
pcv.params.debug = None
payload = json.loads(sys.stdin.read())
module = importlib.import_module('plantcv.plantcv.analyze.color')
assert hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() == '1a28a86a82af612f65ffd55a142df8d23673cd5ba80ec9a2fbfda10606f7b6b8'
helper = importlib.import_module('plantcv.plantcv._helpers')
assert hashlib.sha256(Path(helper.__file__).read_bytes()).hexdigest() == 'c05b0c944210657332ce307f2066caab55f2ebc8d8fea11988db3e476e7cdfba'
captured = []
original = module.outputs.add_observation
def normalize(value):
    if isinstance(value, type): return str(value)
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, (tuple, list)): return [normalize(item) for item in value]
    return value
def capture(*args, **kwargs):
    captured.append({key: normalize(value) for key, value in kwargs.items()})
    return original(*args, **kwargs)
module.outputs.add_observation = capture
error_type = None
try:
    pcv.analyze.color(np.asarray(payload['image'], dtype=np.uint8),
        np.asarray(payload['mask'], dtype=np.int32), n_labels=payload['n_labels'],
        colorspaces=payload['colorspaces'], label=payload['label'])
except Exception as error:
    error_type = type(error).__name__
print(json.dumps({'error_type': error_type, 'observations': captured}, allow_nan=True))
'''


class ColorStatisticsTests(unittest.TestCase):
    def test_hue_median_excludes_zero_hue_samples(self):
        mean, deviation, median = _hue_statistics([100, 1]+[0]*254)
        self.assertAlmostEqual(mean, 2.0)
        self.assertAlmostEqual(deviation, 0.0)
        self.assertEqual(median, 2.0)

    def test_positive_byte_median_excludes_zero_samples(self):
        mean, median = _positive_byte_statistics([100, 0, 1]+[0]*253)
        self.assertAlmostEqual(mean, 2/255*100)
        self.assertAlmostEqual(median, 2/255*100)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA is required for colour analysis')
@functional_units
class PlantCVColorParityTests(unittest.TestCase):
    def reference(self, image, mask, n_labels, colorspaces, label='sample'):
        self.assertTrue(_REFERENCE.is_file(), 'Pinned PlantCV reference is required.')
        completed = subprocess.run([str(_REFERENCE), '-c', _SCRIPT],
            input=json.dumps(dict(image=image.tolist(), mask=mask.tolist(), n_labels=n_labels,
                colorspaces=colorspaces, label=label)), capture_output=True, text=True, timeout=90)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        return json.loads(completed.stdout)

    def compare(self, result, reference):
        self.assertEqual(result.status, 'available')
        self.assertIsNone(reference['error_type'])
        original = {(item['sample'], item['variable']): item for item in reference['observations']}
        actual = {(label.sample, item.variable): item for label in result.labels
            for item in (*label.histograms, *label.statistics)}
        self.assertEqual(set(actual), set(original))
        for key, item in actual.items():
            observed = original[key]
            for field in ('trait', 'method', 'scale', 'datatype'):
                self.assertEqual(getattr(item, field), observed[field], (key, field))
            if hasattr(item, 'values'):
                self.assertEqual(item.values, tuple(observed['value']), key)
                self.assertEqual(item.bin_labels, tuple(observed['label']), key)
            else:
                self.assertEqual(item.units, observed['label'])
                if math.isnan(item.value):
                    self.assertTrue(math.isnan(observed['value']), key)
                    self.assertFalse(item.available)
                elif math.isinf(item.value):
                    self.assertEqual(item.value, observed['value'], key)
                else:
                    # Circular std near zero amplifies last-bit trig reductions.
                    tolerance = 1e-5 if item.variable == 'hue_circular_std' else 1e-10
                    self.assertAlmostEqual(item.value, observed['value'], delta=tolerance, msg=str(key))

    def check_test_all_histograms_statistics_bins_and_labels_match_public_call(self):
        rng = np.random.default_rng(1707)
        image = rng.integers(0, 256, (9, 11, 3), dtype=np.uint8)
        image[2, 2], image[3, 3] = (51, 51, 51), (0, 0, 0)
        mask = np.ones((9, 11), dtype=np.uint8)
        mask[0, :], mask[:, 0], mask[-1, :] = 2, 2, 0
        reference = self.reference(image, mask, 2, 'all', ['first', 'second'])
        result = analyze_color_cuda(torch.as_tensor(image, device='cuda'), torch.as_tensor(mask, device='cuda'),
            n_labels=2, colorspaces='ALL', label=['first', 'second'])
        self.compare(result, reference)
        self.assertEqual([item.pixel_count for item in result.labels], [70, 18])
        indices = np.arange(13*12, dtype=np.int32).reshape(13,12)
        image = np.stack((indices%31+20, indices%37+80, indices%53+180), axis=-1).astype(np.uint8)
        labels = np.maximum(indices-11, 0).astype(np.int32)
        expanded = analyze_color_cuda(torch.as_tensor(image, device='cuda'),
            torch.as_tensor(labels, device='cuda'), n_labels=144, colorspaces='all', label='expanded')
        self.compare(expanded, self.reference(image, labels, 144, 'all', 'expanded'))
        self.assertEqual([item.pixel_count for item in expanded.labels], [1]*144)

    def check_test_binary_remap_and_empty_rgb_lab_hsv_states_match_public_call(self):
        image = np.zeros((3, 4, 3), dtype=np.uint8)
        image[0, 0] = (4, 80, 190)
        mask = np.zeros((3, 4), dtype=np.uint8)
        mask[0, :2] = 255
        result = analyze_color_cuda(torch.as_tensor(image, device='cuda'), torch.as_tensor(mask, device='cuda'),
            colorspaces='all', label='binary')
        self.compare(result, self.reference(image, mask, 1, 'all', 'binary'))
        self.assertEqual(result.labels[0].pixel_count, 2)
        empty = np.zeros_like(mask)
        for space in ('rgb', 'lab'):
            result = analyze_color_cuda(torch.as_tensor(image, device='cuda'), torch.as_tensor(empty, device='cuda'),
                colorspaces=space, label='empty')
            self.compare(result, self.reference(image, empty, 1, space, 'empty'))
        for space in ('hsv', 'all'):
            result = analyze_color_cuda(torch.as_tensor(image, device='cuda'), torch.as_tensor(empty, device='cuda'),
                colorspaces=space, label='empty')
            reference = self.reference(image, empty, 1, space, 'empty')
            self.assertEqual(result.status, 'reference_error')
            self.assertEqual(result.error_type, reference['error_type'])
            self.assertEqual(result.failed_sample, 'empty_1')
            self.assertEqual(result.labels, ())

    def check_test_missing_label_failure_returns_no_partial_complete_bundle(self):
        image = np.full((3, 4, 3), (12, 90, 160), dtype=np.uint8)
        mask = np.ones((3, 4), dtype=np.uint8)
        reference = self.reference(image, mask, 2, 'all')
        result = analyze_color_cuda(torch.as_tensor(image, device='cuda'), torch.as_tensor(mask, device='cuda'),
            n_labels=2, colorspaces='all', label='sample')
        self.assertEqual(reference['error_type'], 'UnboundLocalError')
        self.assertEqual(result.status, 'reference_error')
        self.assertEqual(result.failed_sample, 'sample_2')
        self.assertEqual(result.labels, ())

    def check_test_two_unique_nonzero_remap_discards_other_value_and_singleton_255_does_not_remap(self):
        image = np.full((3, 4, 3), (80, 120, 160), dtype=np.uint8)
        for values in ((2, 255), (255, 255)):
            mask = np.full((3, 4), values[0], dtype=np.uint8)
            mask[0, 0] = values[1]
            reference = self.reference(image, mask, 2, 'rgb')
            result = analyze_color_cuda(torch.as_tensor(image, device='cuda'), torch.as_tensor(mask, device='cuda'),
                n_labels=2, colorspaces='rgb', label='sample')
            self.compare(result, reference)
            self.assertEqual([item.pixel_count for item in result.labels], [1, 0] if values[0] == 2 else [0, 0])

    def check_test_tiling_cancellation_and_input_resource_validation(self):
        from seedvision.cuda import plantcv_color as module
        image = torch.arange(300, device='cuda').remainder(256).to(torch.uint8).reshape(10, 10, 3)
        mask = torch.ones((10, 10), dtype=torch.uint8, device='cuda')
        original = image.clone()
        expected = analyze_color_cuda(image, mask, colorspaces='all')
        with patch.object(module, 'TILE_PIXELS', 7):
            self.assertEqual(analyze_color_cuda(image, mask, colorspaces='all'), expected)
            calls = 0
            def cancel():
                nonlocal calls
                calls += 1
                return calls >= 4
            with self.assertRaises(ColorAnalysisCancelled):
                analyze_color_cuda(image, mask, cancel=cancel)
        torch.testing.assert_close(image, original)
        with self.assertRaises(ColorAnalysisError):
            analyze_color_cuda(image.cpu(), mask)
        with self.assertRaises(ColorAnalysisResourceError):
            analyze_color_cuda(image, mask, n_labels=module.MAX_LABELS+1)
        with self.assertRaises(ColorAnalysisError):
            analyze_color_cuda(image, mask, n_labels=2, label=['one'])

    def check_test_int32_instance_ids_and_negative_binary_remap_match_public_call(self):
        image = np.full((3, 4, 3), (80, 120, 160), dtype=np.uint8)
        mask = np.ones((3, 4), dtype=np.int32)
        mask[0, :3] = [2, 65537, -7]
        result = analyze_color_cuda(torch.as_tensor(image, device='cuda'), torch.as_tensor(mask, device='cuda'),
            n_labels=2, colorspaces='all', label='sample')
        self.compare(result, self.reference(image, mask, 2, 'all'))
        self.assertEqual([item.pixel_count for item in result.labels], [9, 1])
        mask.fill(-7)
        mask[0, 0] = 255
        result = analyze_color_cuda(torch.as_tensor(image, device='cuda'), torch.as_tensor(mask, device='cuda'),
            n_labels=1, colorspaces='all', label='sample')
        self.compare(result, self.reference(image, mask, 1, 'all'))
        self.assertEqual(result.labels[0].pixel_count, 1)

    def test_workflow_lifecycle(self):
        """4 related PlantCVColorParityTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_all_histograms_statistics_bins_and_labels_match_public_call)
        self.scenario(self.check_test_binary_remap_and_empty_rgb_lab_hsv_states_match_public_call)
        self.scenario(self.check_test_two_unique_nonzero_remap_discards_other_value_and_singleton_255_does_not_remap)
        self.scenario(self.check_test_int32_instance_ids_and_negative_binary_remap_match_public_call)

    def test_validation_and_recovery(self):
        """2 related PlantCVColorParityTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_missing_label_failure_returns_no_partial_complete_bundle)
        self.scenario(self.check_test_tiling_cancellation_and_input_resource_validation)


if __name__ == '__main__':
    unittest.main()
