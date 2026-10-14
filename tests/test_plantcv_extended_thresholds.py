import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

import numpy as np
import torch

from seedvision.cuda import plantcv_extended_thresholds as extended
from seedvision.cuda.plantcv_thresholds import OtsuAmbiguityError


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT/'artifacts'/'plantcv-reference-env'/'Scripts'/'python.exe'
SCRIPT = r'''
import hashlib, importlib, json, sys
from pathlib import Path
from importlib.metadata import version
import numpy as np
from plantcv import plantcv as pcv
assert version('plantcv') == '4.11.3'
assert version('numpy') == '2.2.6'
assert version('opencv-python') == '4.11.0.86'
assert version('scipy') == '1.15.3'
assert version('scikit-image') == '0.25.2'
filters = importlib.import_module('scipy.ndimage._filters')
assert hashlib.sha256(Path(filters.__file__).read_bytes()).hexdigest() == '919dcbfd3cccf9c2b61210214d27c828a49529db75a6432f2d9c99442946ec65'
texture = importlib.import_module('skimage.feature.texture')
assert hashlib.sha256(Path(texture.__file__).read_bytes()).hexdigest() == '0bc7c6ea025fa7d69cd23c5c66e34252eb0eda8ec503117406d542a4c93e50d7'
module = importlib.import_module('plantcv.plantcv.threshold.threshold_methods')
assert hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() == '04a671632e7e4decf5a662779cd88473b5aff750554afeee98eabe02eaa49907'
helper = importlib.import_module('plantcv.plantcv._helpers')
assert hashlib.sha256(Path(helper.__file__).read_bytes()).hexdigest() == 'c05b0c944210657332ce307f2066caab55f2ebc8d8fea11988db3e476e7cdfba'
pcv.params.debug = None
payload = json.loads(sys.stdin.read())
image = np.asarray(payload['image'], dtype=np.uint8)
result = []
for params in payload['cases']:
    actual = getattr(pcv.threshold, payload['operation'])(image, **params)
    result.append(actual.tolist())
print(json.dumps(result))
'''

TRIANGLE_OUTCOME_SCRIPT = r'''
import hashlib, importlib, json, sys
from pathlib import Path
from importlib.metadata import version
import cv2, numpy as np
from plantcv import plantcv as pcv
assert version('plantcv') == '4.11.3'
assert version('numpy') == '2.2.6'
assert version('opencv-python') == '4.11.0.86'
assert cv2.__version__ == '4.11.0'
module = importlib.import_module('plantcv.plantcv.threshold.threshold_methods')
assert hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() == '04a671632e7e4decf5a662779cd88473b5aff750554afeee98eabe02eaa49907'
pcv.params.debug = None
payload = json.load(sys.stdin)
image = np.asarray(payload['image'], dtype=np.uint8)
outcomes = []
for case in payload['cases']:
    try:
        pcv.threshold.triangle(image, **case)
    except Exception as error:
        outcomes.append(type(error).__name__)
    else:
        outcomes.append(None)
print(json.dumps(outcomes))
'''


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required for extended thresholds')
class ExtendedThresholdTests(unittest.TestCase):
    def reference(self, image, operation, cases):
        self.assertTrue(REFERENCE.is_file(), 'Pinned original reference environment is required')
        completed = subprocess.run([str(REFERENCE), '-c', SCRIPT],
            input=json.dumps(dict(image=image.tolist(), operation=operation, cases=cases)),
            capture_output=True, text=True, timeout=90)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        return [np.asarray(value, dtype=np.uint8) for value in json.loads(completed.stdout)]

    def test_large_gaussian_kernel_bits_match_pinned_opencv(self):
        script = '''
import json,cv2,numpy as np
from importlib.metadata import version
assert version('opencv-python')=='4.11.0.86' and cv2.__version__=='4.11.0'
print(json.dumps({str(k):cv2.getGaussianKernel(k,0,cv2.CV_32F).reshape(-1).view(np.uint32).tolist()
    for k in (11,13,15)}))
'''
        result = subprocess.run([str(REFERENCE), '-c', script], capture_output=True,
            text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        expected = json.loads(result.stdout)
        for kernel, bits in extended.GAUSSIAN_FLOAT32_KERNEL_BITS.items():
            self.assertEqual(bits, tuple(expected[str(kernel)]))

    def test_gaussian_11_13_15_safe_decisions_match_pinned_public(self):
        image = np.full((31,47), 37, dtype=np.uint8)
        image[:,19:] = 219
        cases = [dict(ksize=k, offset=offset, object_type=polarity)
            for k in (11,13,15) for offset in (-255,-1.25,0,0.5,12,255)
            for polarity in ('light','dark')]
        expected = self.reference(image, 'gaussian', cases)
        source = torch.as_tensor(image, device='cuda')
        masks = {}
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream), patch.object(extended, 'TILE_PIXELS', 101):
            for parameters, original in zip(cases, expected):
                actual = extended.threshold_gaussian(source, **parameters)
                stream.synchronize()
                np.testing.assert_array_equal(actual.cpu().numpy(), original, err_msg=str(parameters))
                masks[(parameters['ksize'],parameters['offset'],parameters['object_type'])] = original
        self.assertTrue(any(not np.array_equal(masks[(a,12,'light')], masks[(b,12,'light')])
            for a,b in ((11,13),(13,15),(11,15))))
        np.testing.assert_array_equal(source.cpu().numpy(), image)

    def test_gaussian_float32_ambiguous_byte_rounding_withholds_complete_mask(self):
        image = np.random.default_rng(411015).integers(0,256,(128,128),dtype=np.uint8)
        self.reference(image, 'gaussian', [dict(ksize=15,offset=0,object_type='light')])
        source = torch.as_tensor(image, device='cuda')
        version = source._version
        with self.assertRaises(extended.ExtendedThresholdPrecisionError):
            extended.threshold_gaussian(source, 15, 0)
        self.assertEqual(source._version, version)
        np.testing.assert_array_equal(source.cpu().numpy(), image)

    def test_gaussian_large_work_scratch_preflight_and_cancellation(self):
        source = torch.full((257,257),80,dtype=torch.uint8,device='cuda')
        for name, limit in (('MAX_GAUSSIAN_FLOAT_WORK',1), ('MAX_GAUSSIAN_FLOAT_SCRATCH_BYTES',1)):
            with patch.object(extended,name,limit), patch.object(torch,'empty_like') as allocate:
                with self.assertRaises(extended.ExtendedThresholdError):
                    extended.threshold_gaussian(source,15,1)
                allocate.assert_not_called()
        before = source.clone()
        version = source._version
        calls = 0
        def cancel():
            nonlocal calls
            calls += 1
            return calls >= 3
        with patch.object(extended,'TILE_PIXELS',101), self.assertRaises(extended.ExtendedThresholdCancelled):
            extended.threshold_gaussian(source,11,1,cancel=cancel)
        self.assertEqual(source._version,version)
        torch.testing.assert_close(source,before)

    def test_gaussian_large_multitile_actual_allocator_stays_within_scratch_cap(self):
        kernel = 15
        tile = min(extended.TILE_PIXELS, extended.MAX_GAUSSIAN_FLOAT_SCRATCH_BYTES //
            (25*kernel*kernel+64*kernel+128))
        source = torch.full((1,3*tile+17),80,dtype=torch.uint8,device='cuda')
        extended.threshold_gaussian(source[:,:1].contiguous(),kernel,1)
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        actual = extended.threshold_gaussian(source,kernel,1)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()-baseline-actual.numel()
        self.assertLessEqual(peak,extended.MAX_GAUSSIAN_FLOAT_SCRATCH_BYTES)
        self.assertEqual(int(torch.count_nonzero(actual != 255)),0)

    def test_gaussian_all_kernels_offsets_polarities_and_replicated_borders_match_public_calls(self):
        rng = np.random.default_rng(20261007)
        cases = [dict(ksize=kernel, offset=offset, object_type=polarity)
            for kernel in (3, 5, 7, 9) for offset in (-255, -1.25, -0.5, 0, 0.5, 1.25, 255)
            for polarity in ('light', 'dark')]
        for shape in ((1, 1), (1, 19), (17, 1), (2, 3), (19, 23)):
            image = rng.integers(0, 256, shape, dtype=np.uint8)
            expected = self.reference(image, 'gaussian', cases)
            source = torch.as_tensor(image, device='cuda')
            for parameters, original in zip(cases, expected):
                actual = extended.threshold_gaussian(source, **parameters)
                np.testing.assert_array_equal(actual.cpu().numpy(), original, err_msg=str((shape, parameters)))
                self.assertEqual(actual.device, source.device)
            np.testing.assert_array_equal(source.cpu().numpy(), image)

    def test_gaussian_tiled_halo_stream_and_rounding_match_original(self):
        rng = np.random.default_rng(309)
        image = rng.integers(0, 256, (21, 25), dtype=np.uint8)
        image[::2, ::2] = 0
        image[1::2, 1::2] = 255
        cases = [dict(ksize=kernel, offset=0, object_type='light') for kernel in (3, 5, 7, 9)]
        expected = self.reference(image, 'gaussian', cases)
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream), patch.object(extended, 'TILE_PIXELS', 17):
            source = torch.as_tensor(image, device='cuda')
            results = [extended.threshold_gaussian(source, **parameters) for parameters in cases]
        stream.synchronize()
        for actual, original in zip(results, expected):
            np.testing.assert_array_equal(actual.cpu().numpy(), original)

    def test_dual_all_channels_and_strict_polarities_match_public_calls(self):
        image = np.random.default_rng(512).integers(0, 256, (9, 13, 3), dtype=np.uint8)
        image[0, :4] = ((0, 0, 0), (255, 255, 255), (0, 0, 255), (255, 0, 0))
        names = list(extended.CHANNELS)
        cases = [dict(x_channel=x, y_channel=names[(index+5)%len(names)],
            points=[[0, 15], [255, 200]], above=above)
            for index, x in enumerate(names) for above in (True, False)]
        expected = self.reference(image, 'dual_channels', cases)
        source = torch.as_tensor(image, device='cuda')
        with patch.object(extended, 'TILE_PIXELS', 11):
            for parameters, original in zip(cases, expected):
                actual = extended.threshold_dual_channels(source, **parameters)
                np.testing.assert_array_equal(actual.cpu().numpy(), original, err_msg=str(parameters))
        np.testing.assert_array_equal(source.cpu().numpy(), image)

    def test_dual_vertical_coincident_extra_points_and_global_indices_match_original(self):
        image = np.arange(120, dtype=np.uint8).reshape(5, 8, 3)
        cases = [dict(x_channel=x, y_channel=y, points=points, above=above)
            for x, y in (('R', 'R'), ('index', 'G'), ('gray', 'index'))
            for points in ([[8, 3], [8, 20]], [[0, 0], [0, 0]], [[0, 5], [40, 5]])
            for above in (True, False)]
        expected = self.reference(image, 'dual_channels', cases)
        source = torch.as_tensor(image, device='cuda')
        with patch.object(extended, 'TILE_PIXELS', 7):
            for parameters, original in zip(cases, expected):
                np.testing.assert_array_equal(extended.threshold_dual_channels(source, **parameters).cpu().numpy(), original)
        with self.assertWarns(UserWarning):
            actual = extended.threshold_dual_channels(source, 'R', 'R', [[0, 0], [1, 1], ['ignored']])
        expected = extended.threshold_dual_channels(source, 'R', 'R', [[0, 0], [1, 1]])
        torch.testing.assert_close(actual, expected)
        equal = extended.threshold_dual_channels(source, 'R', 'R', [[0, 0], [0, 0]], above=False)
        self.assertEqual(int(torch.count_nonzero(equal)), 0)

    def test_invalid_inputs_resource_preflight_and_mid_calculation_cancellation(self):
        gray = torch.arange(30, dtype=torch.uint8, device='cuda').reshape(5, 6)
        image = gray.unsqueeze(-1).expand(-1, -1, 3).contiguous()
        for kernel in (1, 2, 17, True):
            with self.assertRaises((TypeError, ValueError)):
                extended.threshold_gaussian(gray, kernel, 0)
        for offset in (float('inf'), float('nan'), 256, True):
            with self.assertRaises((TypeError, ValueError)):
                extended.threshold_gaussian(gray, 3, offset)
        with self.assertRaises(ValueError):
            extended.threshold_gaussian(gray.cpu(), 3, 0)
        with self.assertRaises(ValueError):
            extended.threshold_gaussian(gray.T, 3, 0)
        for parameters in (('r', 'G', [[0, 0], [1, 1]], True),
                           ('R', 'G', [[0, 0]], True),
                           ('R', 'G', [[0, 0], [float('nan'), 1]], True),
                           ('R', 'G', [[0, 0], [1, 1]], 1)):
            with self.assertRaises(ValueError):
                extended.threshold_dual_channels(image, *parameters)
        with patch.object(extended, 'MAX_RASTER_PIXELS', 29), patch.object(torch, 'empty_like') as allocation:
            with self.assertRaises(ValueError):
                extended.threshold_gaussian(gray, 3, 0)
            allocation.assert_not_called()
        for operation in (lambda cancel: extended.threshold_gaussian(gray, 9, 0, cancel=cancel),
                          lambda cancel: extended.threshold_dual_channels(image, 'R', 'G', [[0, 0], [1, 1]], cancel=cancel)):
            calls = 0
            def cancel():
                nonlocal calls
                calls += 1
                return calls >= 3
            with patch.object(extended, 'TILE_PIXELS', 7), self.assertRaises(extended.ExtendedThresholdCancelled):
                operation(cancel)
        np.testing.assert_array_equal(gray.cpu().numpy(), np.arange(30, dtype=np.uint8).reshape(5, 6))


    def test_triangle_matches_public_masks_and_ignored_xstep_is_not_a_control(self):
        two_level = np.full((12, 12), 30, dtype=np.uint8)
        two_level.flat[::3] = 180
        last_bin = np.full((7, 9), 254, dtype=np.uint8)
        last_bin.flat[::3] = 255
        constant = np.full((5, 7), 42, dtype=np.uint8)
        cases = [dict(object_type=polarity, xstep=step)
            for polarity in ('light', 'dark') for step in (1, 2, 7)]
        for image in (two_level, last_bin, constant):
            with self.subTest(levels=np.unique(image).tolist()):
                expected = self.reference(image, 'triangle', cases)
                source = torch.as_tensor(image, device='cuda')
                for group, polarity in ((0, 'light'), (3, 'dark')):
                    for result in expected[group+1:group+3]:
                        np.testing.assert_array_equal(result, expected[group])
                    actual = extended.threshold_triangle(source, polarity)
                    self.assertEqual(actual.dtype, torch.uint8)
                    self.assertEqual(actual.device, source.device)
                    np.testing.assert_array_equal(actual.cpu().numpy(), expected[group])
                np.testing.assert_array_equal(source.cpu().numpy(), image)
        self.assertNotIn('xstep', inspect.signature(extended.threshold_triangle).parameters)

    def test_triangle_preserves_public_no_peak_preamble_failures(self):
        for image in (np.zeros((5, 7), dtype=np.uint8), np.full((5, 7), 255, dtype=np.uint8)):
            completed = subprocess.run([str(REFERENCE), '-c', TRIANGLE_OUTCOME_SCRIPT],
                input=json.dumps(dict(image=image.tolist(), cases=[dict(object_type='light', xstep=1)])),
                capture_output=True, text=True, timeout=90)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(json.loads(completed.stdout), ['ValueError'])
            with self.assertRaises(extended.TrianglePreambleError):
                extended.threshold_triangle(torch.as_tensor(image, device='cuda'))

    def test_triangle_withholds_ambiguous_otsu_partitions(self):
        image = np.repeat(np.asarray((20, 120, 220), dtype=np.uint8), 36).reshape(9, 12)
        public = self.reference(image, 'triangle', [dict(object_type='light', xstep=1)])[0]
        self.assertEqual(public.shape, image.shape)
        with self.assertRaises(OtsuAmbiguityError):
            extended.threshold_triangle(torch.as_tensor(image, device='cuda'))

    def test_triangle_cancel_after_histogram_preserves_input_and_returns_no_mask(self):
        source = torch.full((12, 12), 30, dtype=torch.uint8, device='cuda')
        source.reshape(-1)[::3] = 180
        before = source.clone()
        calls = 0
        def cancel():
            nonlocal calls
            calls += 1
            return calls >= 2
        with patch.object(extended, 'threshold_otsu') as otsu:
            with self.assertRaises(extended.ExtendedThresholdCancelled):
                extended.threshold_triangle(source, cancel=cancel)
            otsu.assert_not_called()
        torch.testing.assert_close(source, before)

    def test_triangle_pixel_cap_and_initial_cancel_preflight_before_histogram(self):
        from threading import Event
        source = torch.full((5, 7), 30, dtype=torch.uint8, device='cuda')
        with patch.object(extended, 'MAX_TRIANGLE_PIXELS', 1), patch.object(torch, 'bincount') as histogram:
            with self.assertRaises(extended.ExtendedThresholdError):
                extended.threshold_triangle(source)
            histogram.assert_not_called()
        event = Event()
        event.set()
        with patch.object(torch, 'bincount') as histogram:
            with self.assertRaises(extended.ExtendedThresholdCancelled):
                extended.threshold_triangle(source, cancel=event.is_set)
            histogram.assert_not_called()
        for invalid in (source.cpu(), source.T):
            with self.assertRaises(ValueError):
                extended.threshold_triangle(invalid)

    def test_texture_constant_cells_and_byte_cutoff_equality_match_public(self):
        methods = extended.TEXTURE_PROPERTIES
        zero = np.zeros((3, 4), dtype=np.uint8)
        cases = [dict(ksize=3, offset=1, texture_method=method, borders=border,
            threshold=0 if method in ('contrast', 'dissimilarity') else 1)
            for border in extended.TEXTURE_BORDERS for method in methods]
        source = torch.as_tensor(zero, device='cuda')
        for parameters, expected in zip(cases, self.reference(zero, 'texture', cases), strict=True):
            with self.subTest(parameters=parameters):
                actual = extended.threshold_texture(source, **parameters)
                self.assertEqual(actual.dtype, torch.uint8)
                self.assertEqual(actual.device, source.device)
                np.testing.assert_array_equal(actual.cpu().numpy(), expected)
                self.assertFalse(bool(actual.any()))
        image = np.full((4, 5), 42, dtype=np.uint8)
        cases = [dict(ksize=3, offset=1, texture_method=method, borders='nearest', threshold=0)
            for method in methods]
        source = torch.as_tensor(image, device='cuda')
        for parameters, expected in zip(cases, self.reference(image, 'texture', cases), strict=True):
            np.testing.assert_array_equal(extended.threshold_texture(source, **parameters).cpu().numpy(), expected)
        np.testing.assert_array_equal(source.cpu().numpy(), image)

    def test_texture_nonconstant_properties_borders_and_tiled_halos_match_public(self):
        rows, columns = np.indices((11, 19))
        image = (((rows//3+columns//4) % 2)*255).astype(np.uint8)
        cases = [dict(ksize=5, offset=2, texture_method='dissimilarity', borders=border, threshold=0)
            for border in extended.TEXTURE_BORDERS]
        cases.append(dict(ksize=5, offset=2, texture_method='dissimilarity', borders='nearest', threshold=127))
        source = torch.as_tensor(image, device='cuda')
        expected = self.reference(image, 'texture', cases)
        with patch.object(extended, 'TILE_PIXELS', 17):
            for parameters, original in zip(cases, expected, strict=True):
                np.testing.assert_array_equal(extended.threshold_texture(source, **parameters).cpu().numpy(), original)
        image = ((np.indices((7, 11)).sum(axis=0) % 2)*8).astype(np.uint8)
        cases = [dict(ksize=3, offset=1, texture_method=method, borders='nearest', threshold=0)
            for method in extended.TEXTURE_PROPERTIES if method != 'correlation']
        source = torch.as_tensor(image, device='cuda')
        with patch.object(extended, 'TILE_PIXELS', 13):
            for parameters, original in zip(cases, self.reference(image, 'texture', cases), strict=True):
                np.testing.assert_array_equal(extended.threshold_texture(source, **parameters).cpu().numpy(), original)

    def test_texture_k7_k9_all_documented_properties_match_public(self):
        image = np.random.default_rng(92017).integers(0, 16, size=(13, 17), dtype=np.uint8)
        cases = [dict(ksize=kernel, offset=2, texture_method=method, borders='reflect', threshold=0)
            for kernel in (7, 9) for method in extended.TEXTURE_PROPERTIES]
        source = torch.as_tensor(image, device='cuda')
        for parameters, original in zip(cases, self.reference(image, 'texture', cases), strict=True):
            with self.subTest(parameters=parameters):
                np.testing.assert_array_equal(extended.threshold_texture(source, **parameters).cpu().numpy(), original)

    def test_texture_custom_stream_and_nonconstant_correlation_match_public(self):
        image = np.random.default_rng(119).integers(0, 64, size=(11, 15), dtype=np.uint8)
        cases = [dict(ksize=5, offset=2, texture_method=method, borders='mirror', threshold=0)
            for method in ('dissimilarity', 'ASM', 'energy')]
        expected = self.reference(image, 'texture', cases)
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream), patch.object(extended, 'TILE_PIXELS', 13):
            source = torch.as_tensor(image, device='cuda')
            actual = [extended.threshold_texture(source, **parameters) for parameters in cases]
        stream.synchronize()
        for result, original in zip(actual, expected, strict=True):
            np.testing.assert_array_equal(result.cpu().numpy(), original)
        image = np.random.default_rng(884).integers(0, 256, size=(13, 17), dtype=np.uint8)
        cases = [dict(ksize=5, offset=2, texture_method='correlation', borders='reflect', threshold=0)]
        expected = self.reference(image, 'texture', cases)[0]
        np.testing.assert_array_equal(extended.threshold_texture(torch.as_tensor(image, device='cuda'),
            **cases[0]).cpu().numpy(), expected)

    def test_texture_uncertain_byte_boundaries_and_conversion_range_are_withheld(self):
        ramp = np.tile(np.arange(3, dtype=np.uint8), (3, 1))
        for method in ('contrast', 'dissimilarity'):
            with self.subTest(method=method), self.assertRaises(extended.ExtendedThresholdPrecisionError):
                extended.threshold_texture(torch.as_tensor(ramp, device='cuda'), 3, 0, offset=1,
                    texture_method=method)
        checker = ((np.indices((5, 7)).sum(axis=0) % 2)*255).astype(np.uint8)
        cases = [dict(ksize=3, offset=1, texture_method='correlation', borders='nearest', threshold=0)]
        public = self.reference(checker, 'texture', cases)[0]
        self.assertNotEqual(int(public[2, 3]), 0)
        with self.assertRaises(extended.ExtendedThresholdPrecisionError):
            extended.threshold_texture(torch.as_tensor(checker, device='cuda'), **cases[0])
        cases[0]['texture_method'] = 'contrast'
        self.assertTrue(self.reference(checker, 'texture', cases)[0].any())
        with self.assertRaises(extended.ExtendedThresholdPrecisionError):
            extended.threshold_texture(torch.as_tensor(checker, device='cuda'), **cases[0])
        diagonal = np.repeat(np.arange(3, dtype=np.uint8)[:, None], 5, axis=1)
        for method in ('homogeneity', 'correlation'):
            cases = [dict(ksize=3, offset=1, texture_method=method, borders='nearest', threshold=0)]
            self.assertEqual(self.reference(diagonal, 'texture', cases)[0].shape, diagonal.shape)
            with self.assertRaises(extended.ExtendedThresholdPrecisionError):
                extended.threshold_texture(torch.as_tensor(diagonal, device='cuda'), **cases[0])

    def test_texture_largest_kernel_multiple_tiles_stay_within_scratch_budget(self):
        source = torch.randint(0, 16, (80, 100), dtype=torch.uint8, device='cuda')
        before = source.clone()
        torch.cuda.synchronize(source.device)
        torch.cuda.reset_peak_memory_stats(source.device)
        baseline = torch.cuda.memory_allocated(source.device)
        result = extended.threshold_texture(source, 9, 0, offset=1, texture_method='ASM')
        torch.cuda.synchronize(source.device)
        peak = torch.cuda.max_memory_allocated(source.device)-baseline
        self.assertLessEqual(peak-result.numel(), extended.MAX_TEXTURE_SCRATCH_BYTES)
        self.assertEqual(result.device, source.device)
        self.assertEqual(result.dtype, torch.uint8)
        torch.testing.assert_close(source, before)

    def test_texture_invalid_domains_budgets_and_cancellation_preserve_input(self):
        from threading import Event
        image = np.arange(30, dtype=np.uint8).reshape(5, 6)
        source = torch.as_tensor(image, device='cuda')
        parameters = dict(ksize=3, offset=1, texture_method='dissimilarity', borders='nearest', threshold=0)
        for updates in (dict(ksize=1), dict(ksize=2), dict(ksize=11), dict(ksize=True),
                dict(offset=0), dict(offset=3), dict(offset=True), dict(threshold=True), dict(threshold=256),
                dict(texture_method='entropy'), dict(borders='grid-mirror')):
            with self.subTest(updates=updates), self.assertRaises((TypeError, ValueError)):
                extended.threshold_texture(source, **(parameters | updates))
        for invalid in (source.cpu(), source.T):
            with self.assertRaises(ValueError):
                extended.threshold_texture(invalid, **parameters)
        for cap, limit in (('MAX_TEXTURE_PIXELS', 29), ('MAX_TEXTURE_SCRATCH_BYTES', 1)):
            with patch.object(extended, cap, limit), patch.object(torch, 'empty_like') as allocation:
                with self.assertRaises(extended.ExtendedThresholdError):
                    extended.threshold_texture(source, **parameters)
                allocation.assert_not_called()
        cancelled = Event()
        cancelled.set()
        with patch.object(torch, 'empty_like') as allocation:
            with self.assertRaises(extended.ExtendedThresholdCancelled):
                extended.threshold_texture(source, **parameters, cancel=cancelled.is_set)
            allocation.assert_not_called()
        calls = 0
        def cancel_mid_tile():
            nonlocal calls
            calls += 1
            return calls >= 6
        with patch.object(extended, 'TILE_PIXELS', 3), self.assertRaises(extended.ExtendedThresholdCancelled):
            extended.threshold_texture(source, **parameters, cancel=cancel_mid_tile)
        np.testing.assert_array_equal(source.cpu().numpy(), image)


if __name__ == '__main__':
    unittest.main()
