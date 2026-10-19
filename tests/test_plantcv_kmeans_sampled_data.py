"""Actual pinned public sampled patches, CUDA storage and source RNG order."""

from tests.functional_units import functional_units
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

import numpy as np
import torch

from seedvision.cuda import plantcv_kmeans_preprocess as preprocess
from seedvision.cuda import plantcv_kmeans_sampled_data as native

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
assert np.dtype('l').itemsize==4 and np.dtype(int).itemsize==8
results=[]
for item in json.load(sys.stdin):
    image=np.asarray(item['image'],dtype=np.uint8)
    size,sampling,seed=item['patch_size'],item['sampling'],item['seed']
    total=(image.shape[0]-size+1)*(image.shape[1]-size+1)
    count=min(sampling,total) if type(sampling) is int else int(sampling*total)
    rng=np.random.RandomState(seed)
    rows=rng.randint(image.shape[0]-size+1,size=count)
    columns=rng.randint(image.shape[1]-size+1,size=count)
    assert rows.dtype==np.dtype('l') and columns.dtype==np.dtype('l')
    try:
        patches=np.ascontiguousarray(train.patch_extract(image,patch_size=size,sigma=5,sampling=sampling,seed=seed))
        results.append(dict(shape=list(patches.shape),sha256=hashlib.sha256(patches.tobytes()).hexdigest(),
            index_dtype=str(rows.dtype),duplicates=len(set(zip(rows.tolist(),columns.tolist())))<count))
    except Exception as error:
        results.append(dict(error=type(error).__name__,message=str(error)))
print(json.dumps(results))
'''


def _reference(cases):
    payload = [dict(image=image.tolist(), patch_size=size, sampling=sampling, seed=seed)
        for image, size, sampling, seed in cases]
    result = subprocess.run([str(REFERENCE), '-c', SCRIPT], cwd=ROOT,
        input=json.dumps(payload), text=True, capture_output=True, timeout=90)
    if result.returncode:
        raise AssertionError(result.stderr)
    return json.loads(result.stdout)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
@functional_units
class KMeansSampledDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(20261008)
        cls.cases = []
        for shape in ((36, 39), (36, 39, 3)):
            image = rng.integers(0, 256, size=shape, dtype=np.uint8)
            cls.cases.extend((image, size, sampling, seed) for size, sampling, seed in
                ((1, 7, 1), (2, 11, 7), (3, .37, 1), (4, 19, 7), (31, 5, 1)))
        small = rng.integers(0, 256, size=(7, 8), dtype=np.uint8)
        cls.cases.extend(((small, 1, 56, 1), (small, 1, 1000, 1),
            (np.asarray([[128]], dtype=np.uint8), 1, 1, 2**32-1)))
        cls.expected = _reference(cls.cases)

    def check_test_gray_bgr_odd_even_degenerate_and_replacement_modes_match_actual_public(self):
        for case, expected in zip(self.cases, self.expected):
            image, size, sampling, seed = case
            with self.subTest(shape=image.shape, patch=size, sampling=sampling, seed=seed):
                source = torch.tensor(image, device='cuda')
                version = source._version
                result = native.extract_sampled_kmeans_training_patches(source, size,
                    sampling, seed=seed, tile_pixels=13, gaussian_tile_side=7)
                self.assertEqual(result.device, source.device)
                self.assertEqual(result.dtype, torch.uint16)
                self.assertTrue(result.is_contiguous())
                self.assertEqual(list(result.shape), expected['shape'])
                self.assertEqual(sha256(result.cpu().numpy().tobytes()).hexdigest(), expected['sha256'])
                self.assertEqual(expected['index_dtype'], 'int32')
                self.assertEqual(source._version, version)
                np.testing.assert_array_equal(source.cpu().numpy(), image)
        self.assertTrue(self.expected[-3]['duplicates'])
        self.assertTrue(self.expected[-2]['duplicates'])

    def check_test_actual_p31_bgr_multitile_output_peak_and_public_checksum(self):
        y = np.arange(101, dtype=np.uint16)[:, None, None]
        x = np.arange(100, dtype=np.uint16)[None, :, None]
        c = np.arange(3, dtype=np.uint16)[None, None, :]
        image = ((y*29+x*17+c*71+(x*y)%251)%256).astype(np.uint8)
        expected = _reference([(image, 31, 4970, 7)])[0]
        source = torch.tensor(image, device='cuda')
        plan = native.preflight_sampled_kmeans_training_patches(source, 31, 4970, seed=7)
        self.assertEqual((plan.patch_count, plan.feature_count), (4970, 2883))
        self.assertGreater(plan.patch_count, plan.tile_pixels)
        self.assertEqual(plan.cpu_index_bytes, 4970*2*np.dtype('l').itemsize)
        native.extract_sampled_kmeans_training_patches(source[:3, :3].contiguous(), 1, 1)
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        result = native.extract_sampled_kmeans_training_patches(source, 31, 4970, seed=7)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()-baseline
        self.assertLessEqual(peak, plan.peak_live_bytes-source.numel()*source.element_size())
        self.assertEqual(sha256(result.cpu().numpy().tobytes()).hexdigest(), expected['sha256'])
        self.assertEqual(result.numel()*result.element_size(), plan.dataset_bytes)

    def check_test_zero_fraction_floor_matches_actual_public_failure_before_gaussian(self):
        image = np.zeros((34, 35, 3), dtype=np.uint8)
        expected = _reference([(image, 31, .01, 1)])[0]
        self.assertEqual(expected['error'], 'ValueError')
        self.assertIn('reshape', expected['message'])
        with patch.object(preprocess, 'gaussian_patch_raster', side_effect=AssertionError('allocated zero rows')):
            with self.assertRaisesRegex(native.KMeansSampledDataError, 'zero sampled rows'):
                native.extract_sampled_kmeans_training_patches(torch.tensor(image, device='cuda'), 31, .01)

    def check_test_custom_stream_and_tile_repartition_preserve_draw_order(self):
        image, size, sampling, seed = self.cases[7]
        source = torch.tensor(image, device='cuda')
        expected = native.extract_sampled_kmeans_training_patches(source, size, sampling, seed=seed)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            result = native.extract_sampled_kmeans_training_patches(source, size, sampling,
                seed=seed, tile_pixels=7, gaussian_tile_side=5)
        stream.synchronize()
        self.assertTrue(torch.equal(result, expected))

    def check_test_early_gaussian_late_tile_and_final_cancellation_withhold_matrix(self):
        source = torch.zeros((36, 39, 3), dtype=torch.uint8, device='cuda')
        version, calls = source._version, 0
        def count_polls():
            nonlocal calls
            calls += 1
            return False
        native.extract_sampled_kmeans_training_patches(source, 3, 500,
            tile_pixels=13, gaussian_tile_side=7, cancel=count_polls)
        total = calls
        for target in (1, 4, total-2, total):
            calls = 0
            def cancel_at_target():
                nonlocal calls
                calls += 1
                return calls == target
            with self.subTest(poll=target), self.assertRaises(native.KMeansSampledDataCancelled):
                native.extract_sampled_kmeans_training_patches(source, 3, 500,
                    tile_pixels=13, gaussian_tile_side=7, cancel=cancel_at_target)
            self.assertEqual(calls, target)
            self.assertEqual(source._version, version)

    def check_test_callback_mutations_in_preflight_gaussian_tiles_and_final_poll_withhold_matrix(self):
        source = torch.zeros((36, 39, 3), dtype=torch.uint8, device='cuda')
        calls = 0
        def count_polls():
            nonlocal calls
            calls += 1
            return False
        native.extract_sampled_kmeans_training_patches(source, 3, 500,
            tile_pixels=13, gaussian_tile_side=7, cancel=count_polls)
        total = calls
        for target in (1, 2, 4, total-2, total):
            calls = 0
            def mutate_at_target():
                nonlocal calls
                calls += 1
                if calls == target:
                    source[0, 0, 0].fill_(source[0, 0, 0].item()+1)
                return False
            with self.subTest(poll=target), self.assertRaisesRegex(native.KMeansSampledDataError, 'changed'):
                native.extract_sampled_kmeans_training_patches(source, 3, 500,
                    tile_pixels=13, gaussian_tile_side=7, cancel=mutate_at_target)
            self.assertEqual(calls, target)

    def check_test_unchanged_callback_exceptions_preserve_identity_across_gaussian_boundary(self):
        source = torch.zeros((36, 39, 3), dtype=torch.uint8, device='cuda')
        calls = 0
        def count_polls():
            nonlocal calls
            calls += 1
            return False
        native.extract_sampled_kmeans_training_patches(source, 3, 500,
            tile_pixels=13, gaussian_tile_side=7, cancel=count_polls)
        total = calls
        version = source._version
        for target in (1, 4, total):
            calls = 0
            failure = RuntimeError('unchanged callback failed')
            def fail_at_target():
                nonlocal calls
                calls += 1
                if calls == target:
                    raise failure
                return False
            with self.subTest(poll=target), self.assertRaises(RuntimeError) as caught:
                native.extract_sampled_kmeans_training_patches(source, 3, 500,
                    tile_pixels=13, gaussian_tile_side=7, cancel=fail_at_target)
            self.assertIs(caught.exception, failure)
            self.assertEqual(calls, target)
            self.assertEqual(source._version, version)

    def check_test_output_index_work_live_and_geometry_caps_precede_gaussian_allocation(self):
        output_limited = torch.zeros((600, 600, 3), dtype=torch.uint8, device='cuda')
        index_limited = torch.zeros((1025, 2048), dtype=torch.uint8, device='cuda')
        with patch.object(preprocess, 'gaussian_patch_raster', side_effect=AssertionError('allocated before cap')):
            with self.assertRaisesRegex(native.KMeansSampledDataError, 'output byte cap'):
                native.extract_sampled_kmeans_training_patches(output_limited, 31, 25000)
            with self.assertRaisesRegex(native.KMeansSampledDataError, 'CPU metadata cap'):
                native.extract_sampled_kmeans_training_patches(index_limited, 1, 1025*2048)
        small = torch.zeros((36, 39, 3), dtype=torch.uint8, device='cuda')
        plan = native.preflight_sampled_kmeans_training_patches(small, 3, 50)
        for name, limit in (('MAX_WORK', plan.work-1), ('MAX_CUDA_LIVE_BYTES', plan.peak_live_bytes-1)):
            with self.subTest(cap=name), patch.object(native, name, limit), \
                    patch.object(preprocess, 'gaussian_patch_raster', side_effect=AssertionError('allocated before cap')):
                with self.assertRaises(native.KMeansSampledDataError):
                    native.extract_sampled_kmeans_training_patches(small, 3, 50)
        with self.assertRaises(native.KMeansSampledDataError):
            native.preflight_sampled_kmeans_training_patches(small, 40, 5)

    def check_test_invalid_rasters_sampling_seeds_callbacks_and_tiles_are_explicit(self):
        source = torch.zeros((7, 9, 3), dtype=torch.uint8, device='cuda')
        for value in (source.cpu(), source.to(torch.float32), source.transpose(0, 1), source[:0], None):
            with self.subTest(raster=type(value).__name__), self.assertRaises(ValueError):
                native.extract_sampled_kmeans_training_patches(value, 3, 5)
        for value in (None, 0, -1, True, 0., 1., float('nan'), float('inf'), '5', .0001):
            with self.subTest(sampling=value), self.assertRaises(ValueError):
                native.preflight_sampled_kmeans_training_patches(source, 3, value)
        for value in (None, True, -1, 2**32, 1.5):
            with self.subTest(seed=value), self.assertRaises(ValueError):
                native.preflight_sampled_kmeans_training_patches(source, 3, 5, seed=value)
        for name, values in (('tile_pixels', (0, True, 4097, 1.5)),
                             ('gaussian_tile_side', (0, True, 257, 1.5))):
            for value in values:
                with self.subTest(parameter=name, value=value), self.assertRaises(ValueError):
                    native.preflight_sampled_kmeans_training_patches(source, 3, 5, **{name: value})
        with self.assertRaises(TypeError):
            native.extract_sampled_kmeans_training_patches(source, 3, 5, cancel=42)

    def check_test_loaded_source_binds_index_layout_and_storage_constants(self):
        from seedvision.benchmarking.native_recipe import _module_record
        self.assertTrue(_module_record(native)['loaded_code'])
        for name, value in (('RANDOM_INDEX_DTYPE', 'q'), ('MAX_SAMPLE_INDEX_BYTES', 1)):
            with self.subTest(constant=name), patch.object(native, name, value):
                with self.assertRaises(ValueError):
                    _module_record(native)

    def test_workflow_lifecycle(self):
        """5 related KMeansSampledDataTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_gray_bgr_odd_even_degenerate_and_replacement_modes_match_actual_public)
        self.scenario(self.check_test_actual_p31_bgr_multitile_output_peak_and_public_checksum)
        self.scenario(self.check_test_custom_stream_and_tile_repartition_preserve_draw_order)
        self.scenario(self.check_test_output_index_work_live_and_geometry_caps_precede_gaussian_allocation)
        self.scenario(self.check_test_loaded_source_binds_index_layout_and_storage_constants)

    def test_validation_and_recovery(self):
        """5 related KMeansSampledDataTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_zero_fraction_floor_matches_actual_public_failure_before_gaussian)
        self.scenario(self.check_test_early_gaussian_late_tile_and_final_cancellation_withhold_matrix)
        self.scenario(self.check_test_callback_mutations_in_preflight_gaussian_tiles_and_final_poll_withhold_matrix)
        self.scenario(self.check_test_unchanged_callback_exceptions_preserve_identity_across_gaussian_boundary)
        self.scenario(self.check_test_invalid_rasters_sampling_seeds_callbacks_and_tiles_are_explicit)
