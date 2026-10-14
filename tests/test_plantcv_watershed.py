import os
from pathlib import Path
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
import torch
import cv2

from seedvision.cuda.plantcv_watershed import (
    WatershedCancelled, WatershedResourceError, masked_watershed,
)


_ROOT = Path(__file__).resolve().parents[1]
_REFERENCE = Path(os.environ.get('SEEDVISION_PLANTCV_REFERENCE_PYTHON',
    str(_ROOT/'artifacts'/'plantcv-reference-env'/'Scripts'/'python.exe')))
_PINS_SCRIPT = r'''
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
import skimage
assert version('scikit-image') == '0.25.2'
assert version('numpy') == '2.2.6'
assert version('scipy') == '1.15.3'
wrapper = Path(skimage.__file__).parent/'segmentation'/'_watershed.py'
assert sha256(wrapper.read_bytes()).hexdigest() == 'efde33654577d7c1ce8dd8beb9d4cc8e78001d7d2594b1556fce900f7c488963'
'''
_SCRIPT = _PINS_SCRIPT + r'''
import sys
import numpy as np
from skimage.segmentation import watershed
values = np.load(sys.argv[1], allow_pickle=False)
results = {}
for i in range(int(sys.argv[3])):
    results[f'labels{i}'] = watershed(values[f'e{i}'], values[f'm{i}'], mask=values[f'b{i}'])
np.savez(sys.argv[2], **results)
'''

_CHAIN_SCRIPT = _PINS_SCRIPT + r'''
import sys
import numpy as np
import cv2
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
assert version('plantcv') == '4.11.3'
assert version('opencv-python') == '4.11.0.86'
assert version('scikit-image') == '0.25.2'
assert version('numpy') == '2.2.6'
assert version('scipy') == '1.15.3'
mask = np.load(sys.argv[1], allow_pickle=False)
distance, _ = cv2.distanceTransformWithLabels(mask, cv2.DIST_L2, 5)
coordinates = peak_local_max(distance, min_distance=10, labels=mask)
peaks = np.zeros(mask.shape, dtype=bool)
peaks[tuple(coordinates.T)] = True
markers, _ = ndimage.label(peaks, structure=np.ones((3, 3)))
labels = watershed(-distance, markers, mask=mask)
np.savez(sys.argv[2], distance=distance, coordinates=coordinates, peaks=peaks,
    markers=markers, labels=labels)
'''


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA is required for native watershed')
class PlantCVWatershedTests(unittest.TestCase):
    def reference(self, cases):
        self.assertTrue(_REFERENCE.is_file(), 'Pinned reference interpreter is required.')
        arrays = {f'{prefix}{i}': array for i, case in enumerate(cases)
            for prefix, array in zip(('e', 'm', 'b'), case)}
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            np.savez(directory/'input.npz', **arrays)
            completed = subprocess.run([str(_REFERENCE), '-c', _SCRIPT, str(directory/'input.npz'),
                str(directory/'output.npz'), str(len(cases))], capture_output=True, text=True, timeout=120)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            with np.load(directory/'output.npz', allow_pickle=False) as loaded:
                return [loaded[f'labels{i}'].copy() for i in range(len(cases))]

    def test_exact_pinned_heap_ties_plateaus_borders_and_random_partitions(self):
        rng = np.random.default_rng(111)
        cases = []
        for shape in ((1, 1), (1, 13), (11, 1), (9, 13), (15, 17)):
            for mode in range(5):
                elevation = rng.integers(-5, 6, shape).astype(np.float32)
                if mode == 0:
                    elevation.fill(0)
                mask = (rng.random(shape) > .2).astype(np.uint8)*255
                if mode == 1:
                    mask.fill(255)
                if mode == 2:
                    mask.fill(0)
                markers = np.zeros(shape, dtype=np.int32)
                if mode != 3:
                    indices = rng.choice(elevation.size, size=min(5, elevation.size), replace=False)
                    markers.reshape(-1)[indices] = [3, -7, 12, 999, 4][:len(indices)]
                cases.append((elevation, markers, mask))
        expected = self.reference(cases)
        for case, oracle in zip(cases, expected):
            inputs = tuple(torch.as_tensor(array, device='cuda') for array in case)
            originals = tuple(tensor.clone() for tensor in inputs)
            actual = masked_watershed(*inputs)
            self.assertEqual(actual.device, inputs[0].device)
            self.assertEqual(actual.dtype, torch.int32)
            np.testing.assert_array_equal(actual.cpu().numpy(), oracle)
            for tensor, original in zip(inputs, originals):
                torch.testing.assert_close(tensor, original)

    def test_kernel_batches_and_custom_pytorch_stream_preserve_exact_labels(self):
        from seedvision.cuda import plantcv_watershed as module
        elevation = np.zeros((17, 23), dtype=np.float32)
        markers = np.zeros((17, 23), dtype=np.int32)
        markers[1, 1], markers[15, 21], markers[1, 21] = 8, 19, 31
        mask = np.ones((17, 23), dtype=np.bool_)
        expected = self.reference([(elevation, markers, mask)])[0]
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream), patch.object(module, 'KERNEL_BATCH_PIXELS', 7):
            actual = masked_watershed(*(torch.as_tensor(value, device='cuda') for value in (elevation, markers, mask)))
        stream.synchronize()
        np.testing.assert_array_equal(actual.cpu().numpy(), expected)

    def test_full_fixture_distance_peaks_markers_and_partition_match_pinned_chain(self):
        from seedvision.cuda.plantcv_distance import distance_map
        from seedvision.cuda.plantcv_peaks import local_maxima
        from seedvision.cuda.ops import connected_components

        gray = cv2.imread(str(_ROOT/'images'/'IMG_9632c.JPG'), cv2.IMREAD_GRAYSCALE)
        self.assertIsNotNone(gray, 'Committed fixture must decode.')
        mask_array = np.where(gray >= 128, 255, 0).astype(np.uint8)
        self.assertEqual(mask_array.shape, (2080, 3120))
        self.assertTrue(_REFERENCE.is_file(), 'Pinned reference interpreter is required.')
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            np.save(directory/'mask.npy', mask_array, allow_pickle=False)
            completed = subprocess.run([str(_REFERENCE), '-c', _CHAIN_SCRIPT,
                str(directory/'mask.npy'), str(directory/'output.npz')],
                capture_output=True, text=True, timeout=180)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            with np.load(directory/'output.npz', allow_pickle=False) as loaded:
                expected = {name: loaded[name].copy() for name in loaded.files}

        mask = torch.as_tensor(mask_array, device='cuda')
        distance = distance_map(mask)
        coordinates, peaks = local_maxima(distance, mask, 10)
        markers, statistics = connected_components(peaks)
        markers = markers.reshape(mask.shape).contiguous()
        np.testing.assert_array_equal(distance.cpu().numpy(), expected['distance'])
        np.testing.assert_array_equal(coordinates.cpu().numpy(), expected['coordinates'])
        np.testing.assert_array_equal(peaks.cpu().numpy(), expected['peaks'])
        np.testing.assert_array_equal(markers.cpu().numpy(), expected['markers'])
        self.assertEqual(len(statistics['area'])-1, int(expected['markers'].max()))

        torch.cuda.synchronize(mask.device)
        before = torch.cuda.memory_allocated(mask.device)
        torch.cuda.reset_peak_memory_stats(mask.device)
        started = time.perf_counter()
        labels = masked_watershed(-distance, markers, mask)
        torch.cuda.synchronize(mask.device)
        elapsed = time.perf_counter()-started
        peak_bytes = torch.cuda.max_memory_allocated(mask.device)-before
        print(f'\nFull fixture watershed: {elapsed:.3f}s; additional CUDA allocation '
            f'{peak_bytes:,} bytes; foreground {np.count_nonzero(mask_array):,}.', flush=True)
        self.assertLess(elapsed, 240, 'Bounded full-fixture watershed exceeded four minutes.')
        self.assertLess(peak_bytes, 160*1024*1024)
        self.assertEqual(labels.device, mask.device)
        np.testing.assert_array_equal(labels.cpu().numpy(), expected['labels'])

    def test_cancellation_and_resource_preflight_do_not_return_or_modify_inputs(self):
        from seedvision.cuda import plantcv_watershed as module
        elevation = torch.zeros((9, 11), dtype=torch.float32, device='cuda')
        markers = torch.zeros((9, 11), dtype=torch.int32, device='cuda')
        markers[4, 5] = 1
        mask = torch.ones((9, 11), dtype=torch.uint8, device='cuda')
        originals = tuple(value.clone() for value in (elevation, markers, mask))
        # Cancel during initialization and again after flooding has begun.
        for stop_at in (7, 21):
            calls = 0
            def cancel():
                nonlocal calls
                calls += 1
                return calls >= stop_at
            with self.subTest(stop_at=stop_at), patch.object(module, 'KERNEL_BATCH_PIXELS', 7):
                with self.assertRaises(WatershedCancelled):
                    masked_watershed(elevation, markers, mask, cancel=cancel)
        with patch.object(module, 'MAX_WORKSPACE_BYTES', 1), patch.object(module, 'fixed_kernel_program') as compile:
            with self.assertRaises(WatershedResourceError):
                masked_watershed(elevation, markers, mask)
            compile.assert_not_called()
        for value, original in zip((elevation, markers, mask), originals):
            torch.testing.assert_close(value, original)

    def test_invalid_cpu_nonfinite_and_mismatched_inputs_are_rejected(self):
        elevation = torch.zeros((7, 9), dtype=torch.float32, device='cuda')
        markers = torch.zeros((7, 9), dtype=torch.int32, device='cuda')
        mask = torch.ones((7, 9), dtype=torch.bool, device='cuda')
        with self.assertRaises(ValueError):
            masked_watershed(elevation.cpu(), markers, mask)
        with self.assertRaises(ValueError):
            masked_watershed(elevation, markers[:, :7], mask)
        elevation[0, 0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'finite'):
            masked_watershed(elevation, markers, mask)


if __name__ == '__main__':
    unittest.main()
