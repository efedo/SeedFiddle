from __future__ import annotations

import json
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from seedvision.cuda import plantcv_kmeans_preprocess as native

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / 'artifacts' / 'plantcv-reference-env' / 'Scripts' / 'python.exe'
SCRIPT = r'''
import hashlib,importlib,importlib.metadata,json,sys
from pathlib import Path
import numpy as np
from scipy.ndimage._filters import _gaussian_kernel1d
from skimage.util import dtype as skdtype
from skimage._shared import filters as skfilters
train_kmeans=importlib.import_module('plantcv.learn.train_kmeans')
assert importlib.metadata.version('plantcv')=='4.11.3'
assert {name:importlib.metadata.version(name) for name in ('scipy','scikit-image','scikit-learn','numpy')} == {
    'scipy':'1.15.3','scikit-image':'0.25.2','scikit-learn':'1.9.1','numpy':'2.2.6'}
assert hashlib.sha256(Path(skfilters.__file__).read_bytes()).hexdigest()=='0e1ef6097a65b07575c6088b12b89011e3509fcd9711f890b5a9d86dc4b31b53'
assert hashlib.sha256(Path(skdtype.__file__).read_bytes()).hexdigest()=='4239bbec08aefd9078ec7e92b1cae19e344b13dbc74673bb701df8c294c1d0fd'
bits=[format(int(value),'016x') for value in _gaussian_kernel1d(5,0,20).view(np.uint64)]
assert bits==json.loads(sys.argv[1])
results=[]
for item in json.load(sys.stdin):
    image=np.asarray(item['pixels'],dtype=np.uint8).reshape(item['shape'])
    p=item['patch_size']
    before,after=(p-1)//2,p//2
    borders=((before,after),(before,after)) if image.ndim==2 else ((before,after),(before,after),(0,0))
    padded=np.pad(image,borders,mode='edge')
    blurred=train_kmeans.patch_extract(padded,patch_size=1).reshape(padded.shape)
    results.append(blurred.tolist())
print(json.dumps(results))
'''


def cases():
    rng = np.random.default_rng(5012026)
    gray = np.zeros((9, 13), dtype=np.uint8)
    gray[:, 7:] = 255
    gray[3:6, 4:10] = 91
    rgb = rng.integers(0, 256, size=(11, 17, 3), dtype=np.uint8)
    rgb[0, :, 0] = 255
    rgb[:, 0, 1] = 0
    constant = np.empty((5, 8, 3), dtype=np.uint8)
    constant[:] = (33, 127, 254)
    return [
        (np.full((5, 8), 173, np.uint8), 1, 4),
        (gray, 2, 7), (gray, 3, 5), (gray, 10, 8), (gray, 11, 4),
        (rgb, 2, 6), (rgb, 3, 5), (rgb, 10, 8), (rgb, 11, 4),
        (constant, 10, 8),
        (np.array([[0], [31], [255], [120], [4]], dtype=np.uint8), 4, 1),
        (np.array([[0, 31, 255, 120, 4]], dtype=np.uint8), 5, 1),
    ]


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class KMeansGaussianPreprocessTests(unittest.TestCase):
    def reference(self, items):
        requests = [dict(shape=list(image.shape), pixels=image.reshape(-1).tolist(), patch_size=p)
                    for image, p, _ in items]
        result = subprocess.run([str(REFERENCE), '-c', SCRIPT,
                                 json.dumps(list(native.SIGMA5_WEIGHT_BITS))], cwd=ROOT,
                                input=json.dumps(requests), text=True,
                                capture_output=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stderr)
        return [np.asarray(value, dtype=np.uint16) for value in json.loads(result.stdout)]

    def test_odd_even_gray_rgb_and_degenerate_tile_halos_match_public_patch_extract(self):
        items = cases()
        references = self.reference(items)
        for (image, p, tile), expected in zip(items, references):
            with self.subTest(shape=image.shape, patch_size=p, tile_side=tile):
                source = torch.as_tensor(image.copy(), device='cuda')
                version = source._version
                actual = native.gaussian_patch_raster(source, p, tile_side=tile)
                np.testing.assert_array_equal(actual.cpu().numpy(), expected)
                self.assertEqual(source._version, version)
                np.testing.assert_array_equal(source.cpu().numpy(), image)
                self.assertEqual(actual.dtype, torch.uint16)
                self.assertEqual(actual.device.type, 'cuda')

    def test_custom_stream_tile_repartition_and_half_boundary_withholding(self):
        image = np.empty((17, 23, 3), dtype=np.uint8)
        image[:] = (33, 127, 254)
        source = torch.as_tensor(image, device='cuda')
        expected = self.reference([(image, 10, 8)])[0]
        stream = torch.cuda.Stream(device=source.device)
        with torch.cuda.stream(stream):
            tiled = native.gaussian_patch_raster(source, 10, tile_side=8)
        stream.synchronize()
        np.testing.assert_array_equal(tiled.cpu().numpy(), expected)
        large_tiles = native.gaussian_patch_raster(source, 10, tile_side=128)
        np.testing.assert_array_equal(tiled.cpu().numpy(), large_tiles.cpu().numpy())
        self.assertLess(native._byte_error_bound(), 1e-9)
        # Enlarge only the derived bound to exercise all-raster withholding.
        with patch.object(native, '_byte_error_bound', return_value=0.5):
            with self.assertRaises(native.KMeansPrecisionError):
                native.gaussian_patch_raster(torch.full((4, 5), 91, dtype=torch.uint8,
                                                        device='cuda'), 1)

    def test_cancel_before_allocation_mid_raster_and_final_poll_returns_no_output(self):
        source = torch.arange(12 * 13, dtype=torch.uint8, device='cuda').reshape(12, 13)
        before, version = source.clone(), source._version
        with patch.object(native.torch, 'empty', side_effect=AssertionError('allocated before cancel')):
            with self.assertRaises(native.KMeansPreprocessCancelled):
                native.gaussian_patch_raster(source, 4, cancel=lambda: True)
        calls = 0
        def cancel_after_start():
            nonlocal calls
            calls += 1
            return calls >= 7
        with self.assertRaises(native.KMeansPreprocessCancelled):
            native.gaussian_patch_raster(source, 4, cancel=cancel_after_start, tile_side=4)
        self.assertEqual(calls, 7)
        calls = 0
        def count_polls():
            nonlocal calls
            calls += 1
            return False
        native.gaussian_patch_raster(source, 1, cancel=count_polls)
        final_poll = calls
        calls = 0
        def cancel_final():
            nonlocal calls
            calls += 1
            return calls >= final_poll
        with self.assertRaises(native.KMeansPreprocessCancelled):
            native.gaussian_patch_raster(source, 1, cancel=cancel_final)
        self.assertEqual(source._version, version)
        torch.testing.assert_close(source, before)

    def test_image_padded_output_work_and_scratch_caps_precede_allocation(self):
        source = torch.zeros((2, 3, 3), dtype=torch.uint8, device='cuda')
        for name in ('MAX_IMAGE_PIXELS', 'MAX_PADDED_PIXELS', 'MAX_OUTPUT_BYTES',
                     'MAX_FILTER_WORK', 'MAX_SCRATCH_BYTES'):
            with self.subTest(cap=name), patch.object(native, name, 1), patch.object(
                    native.torch, 'empty', side_effect=AssertionError('allocated before cap')):
                with self.assertRaises(native.KMeansPreprocessError):
                    native.gaussian_patch_raster(source, 3)

    def test_actual_maximum_tile_scratch_stays_bounded_on_a_multitile_raster(self):
        image = torch.arange(520 * 517 * 3, dtype=torch.int64, device='cuda').remainder(256).to(
            torch.uint8).reshape(520, 517, 3)
        base = torch.cuda.memory_allocated(image.device)
        torch.cuda.reset_peak_memory_stats(image.device)
        result = native.gaussian_patch_raster(image, 10, tile_side=native.MAX_TILE_SIDE)
        torch.cuda.synchronize(image.device)
        peak_delta = torch.cuda.max_memory_allocated(image.device) - base
        persistent = result.numel() * result.element_size()
        self.assertLessEqual(peak_delta - persistent, native.MAX_SCRATCH_BYTES)
        self.assertLessEqual(native._preflight_scratch_bytes(3, native.MAX_TILE_SIDE),
                             native.MAX_SCRATCH_BYTES)

    def test_tiny_tiles_charge_repeated_vertical_halo_work_before_allocation(self):
        image = torch.zeros((2, 3), dtype=torch.uint8, device='cuda')
        with patch.object(native, 'MAX_FILTER_WORK', 5000):
            native.gaussian_patch_raster(image, 1, tile_side=256)
            with patch.object(native.torch, 'empty', side_effect=AssertionError('allocated before halo cap')):
                with self.assertRaises(native.KMeansPreprocessError):
                    native.gaussian_patch_raster(image, 1, tile_side=1)

    def test_invalid_types_layout_and_parameters_fail_before_work(self):
        source = torch.zeros((4, 5), dtype=torch.uint8, device='cuda')
        with self.assertRaises(TypeError):
            native.gaussian_patch_raster(source, 3, cancel=object())
        invalid = [source[:, ::2], source.cpu(), source.float(), source.reshape(4, 5, 1),
                   torch.zeros((0, 5), dtype=torch.uint8, device='cuda'),
                   torch.sparse_coo_tensor(torch.tensor([[0], [0]], device='cuda'),
                                          torch.tensor([1], dtype=torch.uint8, device='cuda'),
                                          (4, 5), check_invariants=True)]
        for image in invalid:
            with self.assertRaises(native.KMeansPreprocessError):
                native.gaussian_patch_raster(image, 3)
        for p in (0, 32, True, 1.5):
            with self.assertRaises(native.KMeansPreprocessError):
                native.gaussian_patch_raster(source, p)
        for tile in (0, native.MAX_TILE_SIDE + 1, True, 1.5):
            with self.assertRaises(native.KMeansPreprocessError):
                native.gaussian_patch_raster(source, 3, tile_side=tile)


if __name__ == '__main__':
    unittest.main()
