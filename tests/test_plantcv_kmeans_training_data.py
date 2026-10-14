"""Actual pinned public training patches and bounded native CUDA allocation."""
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

import numpy as np
import torch

from seedvision.cuda import plantcv_kmeans_features as features
from seedvision.cuda import plantcv_kmeans_preprocess as preprocess
from seedvision.cuda import plantcv_kmeans_training_data as native

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT/'artifacts/plantcv-reference-env/Scripts/python.exe'
SCRIPT = r'''
import hashlib,importlib,importlib.metadata,json,sys
from pathlib import Path
import numpy as np
import sklearn.feature_extraction.image as feature_source
train=importlib.import_module('plantcv.learn.train_kmeans')
assert {name:importlib.metadata.version(name) for name in ('plantcv','numpy','scipy','scikit-image','scikit-learn','opencv-python')}=={
    'plantcv':'4.11.3','numpy':'2.2.6','scipy':'1.15.3','scikit-image':'0.25.2','scikit-learn':'1.9.1','opencv-python':'4.11.0.86'}
assert hashlib.sha256(Path(train.__file__).read_bytes()).hexdigest()=='dbfb8a528e92984a20e73c7f265a41efa9fa088b42ed4f65addbe6ab40ee8b4d'
assert hashlib.sha256(Path(feature_source.__file__).read_bytes()).hexdigest()=='55a0eee1228f4fc6ab96698dea6e5c0e8b4d5fa5c663ea40efa7f1d7dbf56eb5'
rows=[]
for item in json.load(sys.stdin):
    image=np.asarray(item['image'],dtype=np.uint8)
    patches=np.ascontiguousarray(train.patch_extract(image,patch_size=item['patch_size'],sigma=5,sampling=None))
    rows.append(dict(shape=list(patches.shape),dtype=str(patches.dtype),sha256=hashlib.sha256(patches.tobytes()).hexdigest()))
print(json.dumps(rows))
'''


def _reference(requests):
    result = subprocess.run([str(REFERENCE), '-c', SCRIPT], cwd=ROOT,
        input=json.dumps(requests), text=True, capture_output=True, timeout=90)
    if result.returncode:
        raise AssertionError(result.stderr)
    return json.loads(result.stdout)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class KMeansTrainingDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(417)
        cls.fixtures = []
        for shape in ((35, 37), (35, 37, 3)):
            image = rng.integers(0, 256, size=shape, dtype=np.uint8)
            image[0, 0], image[-1, -1] = 255, 0
            cls.fixtures.extend((image, size) for size in (1, 2, 3, 4, 31))
        cls.fixtures.extend((np.asarray(image, np.uint8), size) for image, size in
            (([[0], [255], [17], [90]], 1), ([[0, 255, 17, 90]], 1), ([[128]], 1)))
        cls.reference = _reference([dict(image=image.tolist(), patch_size=size)
            for image, size in cls.fixtures])

    def test_unpadded_gray_bgr_odd_even_and_degenerate_patches_match_actual_public(self):
        for (image, size), expected in zip(self.fixtures, self.reference):
            with self.subTest(shape=image.shape, patch_size=size):
                source = torch.tensor(image, dtype=torch.uint8, device='cuda')
                version = source._version
                result = native.extract_full_kmeans_training_patches(source, size, tile_pixels=17,
                    gaussian_tile_side=7)
                self.assertEqual(result.dtype, torch.uint16)
                self.assertEqual(result.device, source.device)
                self.assertTrue(result.is_contiguous())
                self.assertEqual(list(result.shape), expected['shape'])
                self.assertEqual(str(result.cpu().numpy().dtype), expected['dtype'])
                self.assertEqual(sha256(result.cpu().numpy().tobytes()).hexdigest(), expected['sha256'])
                self.assertEqual(result.shape[0], (image.shape[0]-size+1)*(image.shape[1]-size+1))
                self.assertEqual(source._version, version)
                np.testing.assert_array_equal(source.cpu().numpy(), image)

    def test_actual_p31_bgr_multitile_matrix_peak_and_public_checksum(self):
        y = np.arange(101, dtype=np.uint16)[:, None, None]
        x = np.arange(100, dtype=np.uint16)[None, :, None]
        channel = np.arange(3, dtype=np.uint16)[None, None, :]
        image = ((y*29+x*17+channel*71+(x*y)%251)%256).astype(np.uint8)
        expected = _reference([dict(image=image.tolist(), patch_size=31)])[0]
        source = torch.tensor(image, dtype=torch.uint8, device='cuda')
        plan = native.preflight_kmeans_training_data(source, 31)
        self.assertEqual((plan.patch_count, plan.feature_count), (4970, 2883))
        self.assertGreater(plan.patch_count, plan.tile_pixels)
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        result = native.extract_full_kmeans_training_patches(source, 31)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()-baseline
        self.assertEqual(list(result.shape), expected['shape'])
        self.assertEqual(sha256(result.cpu().numpy().tobytes()).hexdigest(), expected['sha256'])
        self.assertLessEqual(peak, plan.peak_live_bytes-source.numel()*source.element_size())
        self.assertEqual(result.numel()*result.element_size(), plan.dataset_bytes)

    def test_custom_stream_and_tile_repartition_preserve_rows(self):
        source = torch.arange(13*17*3, dtype=torch.int32, device='cuda').reshape(13, 17, 3)
        source = source.remainder_(256).to(torch.uint8)
        expected = native.extract_full_kmeans_training_patches(source, 4)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            result = native.extract_full_kmeans_training_patches(source, 4, tile_pixels=7,
                gaussian_tile_side=5)
        stream.synchronize()
        self.assertTrue(torch.equal(result, expected))

    def test_early_gaussian_and_final_cancellation_return_no_matrix(self):
        source = torch.arange(15*17*3, dtype=torch.int32, device='cuda').reshape(15, 17, 3)
        source = source.remainder_(256).to(torch.uint8)
        version = source._version
        calls = 0
        def count_polls():
            nonlocal calls
            calls += 1
            return False
        native.extract_full_kmeans_training_patches(source, 3, tile_pixels=16,
            gaussian_tile_side=7, cancel=count_polls)
        total = calls
        self.assertGreater(total, 10)
        for target in (1, 4, total):
            calls = 0
            def cancel_at_target():
                nonlocal calls
                calls += 1
                return calls == target
            with self.subTest(poll=target), self.assertRaises(native.KMeansTrainingDataCancelled):
                native.extract_full_kmeans_training_patches(source, 3, tile_pixels=16,
                    gaussian_tile_side=7, cancel=cancel_at_target)
            self.assertEqual(calls, target)
            self.assertEqual(source._version, version)
        calls = 0
        def mutate_at_final_poll():
            nonlocal calls
            calls += 1
            if calls == total:
                source[0, 0, 0].fill_(42)
            return False
        with self.assertRaisesRegex(native.KMeansTrainingDataError, 'changed'):
            native.extract_full_kmeans_training_patches(source, 3, tile_pixels=16,
                gaussian_tile_side=7, cancel=mutate_at_final_poll)

    def test_cancellation_and_bad_coverage_inside_feature_iteration_withhold_matrix(self):
        source = torch.zeros((15, 17, 3), dtype=torch.uint8, device='cuda')
        original = features.iter_kmeans_patch_tiles
        cancelled = False
        def cancel_at_first_tile(*args, **kwargs):
            nonlocal cancelled
            for start, tile in original(*args, **kwargs):
                cancelled = True
                yield start, tile
        with patch.object(features, 'iter_kmeans_patch_tiles', cancel_at_first_tile):
            with self.assertRaises(native.KMeansTrainingDataCancelled):
                native.extract_full_kmeans_training_patches(source, 3, tile_pixels=16,
                    cancel=lambda: cancelled)
        def broken_coverage(*args, **kwargs):
            for start, tile in original(*args, **kwargs):
                yield start+1, tile
        with patch.object(features, 'iter_kmeans_patch_tiles', broken_coverage):
            with self.assertRaisesRegex(native.KMeansTrainingDataError, 'cover'):
                native.extract_full_kmeans_training_patches(source, 3, tile_pixels=16)

    def test_global_output_work_live_and_geometry_caps_precede_gaussian_allocation(self):
        source = torch.zeros((1500, 2000, 3), dtype=torch.uint8, device='cuda')
        with patch.object(preprocess, 'gaussian_patch_raster', side_effect=AssertionError('allocated before cap')):
            with self.assertRaisesRegex(native.KMeansTrainingDataError, 'output byte cap'):
                native.extract_full_kmeans_training_patches(source, 3)
        small = source[:7, :9].contiguous()
        plan = native.preflight_kmeans_training_data(small, 3)
        cases = [('MAX_WORK', plan.work-1), ('MAX_CUDA_LIVE_BYTES', plan.peak_live_bytes-1),
                 ('MAX_PATCHES', plan.patch_count-1)]
        for name, limit in cases:
            with self.subTest(cap=name), patch.object(native, name, limit), \
                    patch.object(preprocess, 'gaussian_patch_raster', side_effect=AssertionError('allocated before cap')):
                with self.assertRaises(native.KMeansTrainingDataError):
                    native.extract_full_kmeans_training_patches(small, 3)
        with patch.object(preprocess, 'gaussian_patch_raster', side_effect=AssertionError('allocated before geometry')):
            with self.assertRaises(native.KMeansTrainingDataError):
                native.extract_full_kmeans_training_patches(small, 10)

    def test_invalid_rasters_callbacks_and_tile_parameters_are_explicit(self):
        source = torch.zeros((7, 9, 3), dtype=torch.uint8, device='cuda')
        for invalid in (source.cpu(), source.to(torch.float32), source.transpose(0, 1),
                        source[:, :, :1].contiguous(), source[:0], None):
            with self.subTest(raster=type(invalid).__name__), self.assertRaises(ValueError):
                native.preflight_kmeans_training_data(invalid, 3)
        for name, values in (('tile_pixels', (0, True, 4097, 1.5)),
                             ('gaussian_tile_side', (0, True, 257, 1.5))):
            for value in values:
                with self.subTest(parameter=name, value=value), self.assertRaises(ValueError):
                    native.preflight_kmeans_training_data(source, 3, **{name: value})
        for value in (0, True, 32, 1.5):
            with self.subTest(patch_size=value), self.assertRaises(ValueError):
                native.preflight_kmeans_training_data(source, value)
        with self.assertRaises(TypeError):
            native.preflight_kmeans_training_data(source, 3, cancel=42)

    def test_mutation_during_preflight_is_rejected_before_gaussian_allocation(self):
        source = torch.zeros((7, 9, 3), dtype=torch.uint8, device='cuda')
        calls = 0
        def mutate_after_value_checks():
            nonlocal calls
            calls += 1
            if calls == 2:
                source[0, 0, 0].fill_(255)
            return False
        with patch.object(preprocess, 'gaussian_patch_raster', side_effect=AssertionError('allocated changed input')):
            with self.assertRaisesRegex(native.KMeansTrainingDataError, 'during preflight'):
                native.extract_full_kmeans_training_patches(source, 3, cancel=mutate_after_value_checks)
