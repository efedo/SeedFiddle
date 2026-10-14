import hashlib
import json
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

import numpy as np
import torch

from seedvision.cuda import plantcv_kmeans_features as native

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / 'artifacts/plantcv-reference-env/Scripts/python.exe'
SCRIPT = r'''
import hashlib,importlib,importlib.metadata,json,sys
from pathlib import Path
import numpy as np
from sklearn.feature_extraction import image as extraction
train=importlib.import_module('plantcv.learn.train_kmeans')
assert importlib.metadata.version('plantcv')=='4.11.3'
assert importlib.metadata.version('numpy')=='2.2.6'
assert importlib.metadata.version('scikit-image')=='0.25.2'
assert importlib.metadata.version('scikit-learn')=='1.9.1'
assert hashlib.sha256(Path(train.__file__).read_bytes()).hexdigest()=='dbfb8a528e92984a20e73c7f265a41efa9fa088b42ed4f65addbe6ab40ee8b4d'
assert hashlib.sha256(Path(extraction.__file__).read_bytes()).hexdigest()=='55a0eee1228f4fc6ab96698dea6e5c0e8b4d5fa5c663ea40efa7f1d7dbf56eb5'
results=[]
for item in json.load(sys.stdin):
    p=item['patch_size']
    if item['kind']=='public':
        image=np.asarray(item['values'],dtype=np.uint8).reshape(item['shape'])
        before,after=(p-1)//2,p//2
        pad=((before,after),(before,after))+(((0,0),) if image.ndim==3 else ())
        padded=np.pad(image,pad,mode='edge')
        raster=train.patch_extract(padded,patch_size=1).reshape(padded.shape)
        features=train.patch_extract(padded,patch_size=p)
    else:
        if 'values' in item:
            raster=np.asarray(item['values'],dtype=np.uint16).reshape(item['shape'])
        else:
            raster=((np.arange(np.prod(item['shape']),dtype=np.uint32)*37+251)%4096).astype(np.uint16).reshape(item['shape'])
        extracted=extraction.extract_patches_2d(raster,(p,p))
        features=extracted.reshape(extracted.shape[0],-1)
    result=dict(shape=list(features.shape),sha256=hashlib.sha256(features.tobytes()).hexdigest())
    if item['kind']=='public':
        result.update(raster=raster.tolist(),features=features.tolist())
    results.append(result)
print(json.dumps(results))
'''


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class PlantCVKMeansFeatureTests(unittest.TestCase):
    def public(self, requests):
        result = subprocess.run([str(REFERENCE), '-c', SCRIPT], input=json.dumps(requests),
                                text=True, capture_output=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def collect(self, raster, p, tile):
        arrays, starts, expected_start = [], [], 0
        for start, features in native.iter_kmeans_patch_tiles(raster, p, tile_pixels=tile):
            self.assertEqual(start, expected_start)
            self.assertEqual(features.device, raster.device)
            self.assertEqual(features.dtype, torch.uint16)
            self.assertTrue(features.is_contiguous())
            starts.append(start)
            expected_start += features.shape[0]
            arrays.append(features.cpu().numpy())
            del features
        return starts, np.vstack(arrays)

    def test_gray_rgb_odd_even_and_split_row_tiles_match_actual_public_patches(self):
        rng = np.random.default_rng(5041)
        requests = []
        for channels in (1, 3):
            image = rng.integers(0, 256, (7, 9) if channels == 1 else (7, 9, 3), dtype=np.uint8)
            for p in (1, 2, 3, 4, 10):
                requests.append(dict(kind='public', patch_size=p, shape=list(image.shape), values=image.tolist()))
        for request, reference in zip(requests, self.public(requests)):
            with self.subTest(channels=len(request['shape']), p=request['patch_size']):
                raster = torch.tensor(reference['raster'], dtype=torch.uint16, device='cuda')
                before, version = raster.cpu().numpy().copy(), raster._version
                _, actual = self.collect(raster, request['patch_size'], 5)
                np.testing.assert_array_equal(actual, reference['features'])
                _, wider = self.collect(raster, request['patch_size'], 31)
                np.testing.assert_array_equal(actual, wider)
                self.assertEqual(raster._version, version)
                np.testing.assert_array_equal(raster.cpu().numpy(), before)

    def test_uint16_values_above_byte_range_and_independent_p1_tiles(self):
        image = np.asarray([[[0, 256, 1023], [4095, 511, 300]],
                            [[700, 1200, 999], [655, 2048, 3500]]], dtype=np.uint16)
        reference = self.public([dict(kind='raw', values=image.tolist(), shape=list(image.shape), patch_size=1)])[0]
        raster = torch.tensor(image, device='cuda')
        starts, actual = self.collect(raster, 1, 2)
        self.assertEqual(starts, [0, 2])
        self.assertGreater(int(actual.max()), 255)
        self.assertEqual(hashlib.sha256(actual.tobytes()).hexdigest(), reference['sha256'])
        iterator = native.iter_kmeans_patch_tiles(raster, 1, tile_pixels=2)
        _, tile = next(iterator)
        tile.zero_()
        np.testing.assert_array_equal(raster.cpu().numpy(), image)

    def test_actual_maximum_patch_rgb_multitile_allocator_and_public_feature_hash(self):
        shape, p = (101, 100, 3), 31
        reference = self.public([dict(kind='raw', shape=list(shape), patch_size=p)])[0]
        raster = (torch.arange(np.prod(shape), dtype=torch.int32, device='cuda')
                  .mul_(37).add_(251).remainder_(4096).to(torch.uint16).reshape(shape))
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        digest, starts, expected_start = hashlib.sha256(), [], 0
        for start, tile in native.iter_kmeans_patch_tiles(raster, p):
            self.assertEqual(start, expected_start)
            expected_start += tile.shape[0]
            starts.append(start)
            digest.update(tile.cpu().numpy().tobytes())
            del tile
        torch.cuda.synchronize()
        self.assertEqual(starts, [0, 4060])
        self.assertEqual(expected_start, 4970)
        self.assertEqual(digest.hexdigest(), reference['sha256'])
        self.assertLessEqual(torch.cuda.max_memory_allocated() - base, native.MAX_FEATURE_TILE_BYTES)

    def test_custom_stream_and_cancellation_preserve_source(self):
        source = torch.arange(11 * 13 * 3, dtype=torch.int32, device='cuda').to(torch.uint16).reshape(11, 13, 3)
        before, version = source.cpu().numpy().copy(), source._version
        expected_starts, expected = self.collect(source, 4, 9)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            starts, actual = self.collect(source, 4, 9)
        stream.synchronize()
        self.assertEqual(starts, expected_starts)
        np.testing.assert_array_equal(actual, expected)
        polls = 0
        def cancel():
            nonlocal polls
            polls += 1
            return polls >= 4
        with self.assertRaises(native.KMeansFeatureCancelled):
            list(native.iter_kmeans_patch_tiles(source, 4, cancel=cancel, tile_pixels=2))
        self.assertEqual(source._version, version)
        np.testing.assert_array_equal(source.cpu().numpy(), before)

    def test_cancel_and_caps_precede_tile_allocation(self):
        source = torch.zeros((4, 5, 3), dtype=torch.uint16, device='cuda')
        with patch.object(native.torch, 'empty', side_effect=AssertionError('allocated before cancel')):
            with self.assertRaises(native.KMeansFeatureCancelled):
                next(native.iter_kmeans_patch_tiles(source, 2, cancel=lambda: True))
        for cap in ('MAX_PATCH_PIXELS', 'MAX_PADDED_PIXELS', 'MAX_PATCH_FEATURES', 'MAX_FEATURE_TILE_BYTES'):
            with self.subTest(cap=cap), patch.object(native, cap, 1), patch.object(
                    native.torch, 'empty', side_effect=AssertionError('allocated before cap')):
                with self.assertRaises(native.KMeansFeatureError):
                    next(native.iter_kmeans_patch_tiles(source, 2))

    def test_invalid_rasters_and_tile_parameters_are_explicit(self):
        source = torch.zeros((4, 5), dtype=torch.uint16, device='cuda')
        for raster in (source.cpu(), source.to(torch.uint8), source[:, ::2], source.reshape(4, 5, 1)):
            with self.assertRaises(native.KMeansFeatureError):
                next(native.iter_kmeans_patch_tiles(raster, 2))
        for p in (0, 6, 32, True, 1.5):
            with self.assertRaises(native.KMeansFeatureError):
                next(native.iter_kmeans_patch_tiles(source, p))
        for tile in (0, 4097, True, 1.5):
            with self.assertRaises(native.KMeansFeatureError):
                next(native.iter_kmeans_patch_tiles(source, 2, tile_pixels=tile))
        with self.assertRaises(TypeError):
            next(native.iter_kmeans_patch_tiles(source, 2, cancel=object()))
