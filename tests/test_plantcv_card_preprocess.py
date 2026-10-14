import hashlib
import json
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

import numpy as np
import torch

from seedvision.cuda import plantcv_card_preprocess as card

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / 'artifacts/plantcv-reference-env/Scripts/python.exe'
DETECTOR_SHA256 = 'e3ad4028e8a831e523815e33ee057f08ba73f5a3f60b3933f11504dd71b7bb49'
SCRIPT = r'''
import hashlib, importlib, json, sys
from pathlib import Path
from importlib.metadata import version
import cv2, numpy as np
from plantcv import plantcv as pcv
assert version('plantcv') == '4.11.3'
assert version('numpy') == '2.2.6'
assert version('opencv-python') == '4.11.0.86' and cv2.__version__ == '4.11.0'
module = importlib.import_module('plantcv.plantcv.transform.detect_color_card')
assert hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() == 'e3ad4028e8a831e523815e33ee057f08ba73f5a3f60b3933f11504dd71b7bb49'
pcv.params.debug = None
payload = json.load(sys.stdin)
operation = payload['operation']
if operation == 'kernels':
    result = {str(k):hashlib.sha256(cv2.getGaussianKernel(k,0,cv2.CV_32F).astype('<f4').tobytes()).hexdigest()
              for k in range(3,256,2)}
elif operation == 'byte_weights':
    taps = cv2.getGaussianKernel(11,0,cv2.CV_64F).ravel()
    error, left = 0.0, []
    for value in taps[:5]:
        adjusted = float(value)*256 + error
        integer = int(np.rint(adjusted))
        error = adjusted-integer
        left.append(integer)
    result = left+[256-2*sum(left)]+left[::-1]
else:
    image = np.asarray(payload['image'],dtype=np.uint8)
    if operation == 'blur':
        result = cv2.GaussianBlur(image,(11,11),0).tolist()
    elif operation == 'adaptive':
        result = [cv2.adaptiveThreshold(image,255,method,cv2.THRESH_BINARY_INV,k,2).tolist()
                  for method,k in payload['cases']]
    elif operation == 'public':
        # Execute the actual public detector through its preprocessing boundary.
        # Contour interception intentionally stops geometry, which this gate
        # does not claim to implement or validate.
        captured = []
        original = cv2.findContours
        class StopAfterPreprocessing(Exception): pass
        def capture(mask,*args,**kwargs):
            captured.append(mask.copy())
            raise StopAfterPreprocessing()
        cv2.findContours = capture
        try:
            pcv.transform.detect_color_card(image, adaptive_method=payload['method'],
                block_size=payload['size'])
        except StopAfterPreprocessing:
            pass
        finally:
            cv2.findContours = original
        assert len(captured) == 1
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        result = [gray.tolist(),cv2.GaussianBlur(gray,(11,11),0).tolist(),captured[0].tolist()]
    else:
        raise AssertionError(operation)
print(json.dumps(result))
'''


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required for card preprocessing')
class ColorCardPreprocessTests(unittest.TestCase):
    def oracle(self, operation, **payload):
        self.assertTrue(REFERENCE.is_file(), 'Pinned original reference environment required')
        result = subprocess.run([str(REFERENCE), '-c', SCRIPT],
            input=json.dumps(dict(operation=operation, **payload)), capture_output=True,
            text=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def test_all_kernel_identities_and_integer_byte_weights_match_pinned_original(self):
        expected = self.oracle('kernels')
        self.assertEqual(set(card.GAUSSIAN_KERNEL_SHA256), set(range(3,256,2)))
        for size, digest in card.GAUSSIAN_KERNEL_SHA256.items():
            self.assertEqual(digest, expected[str(size)])
            self.assertEqual(len(card._weights(size)), size)
        self.assertEqual(card.BYTE11_WEIGHTS, tuple(self.oracle('byte_weights')))
        self.assertEqual(sum(card.BYTE11_WEIGHTS), 256)
        self.assertLess(255*256**2+32768, 1 << 24)

    def test_byte_blur_reflect101_random_impulses_singleton_and_tile_seams(self):
        rng = np.random.default_rng(411311)
        images = [rng.integers(0,256,shape,dtype=np.uint8)
            for shape in ((1,1),(1,2),(2,1),(2,3),(1,39),(37,1),(31,47))]
        impulse = np.zeros((19,23),dtype=np.uint8)
        impulse[0,0] = impulse[-1,-1] = impulse[9,11] = 255
        images.append(impulse)
        for image in images:
            source = torch.as_tensor(image, device='cuda')
            version = source._version
            expected = np.asarray(self.oracle('blur',image=image.tolist()), dtype=np.uint8)
            for tile in (7, card.MAX_TILE_SIDE):
                actual = card.gaussian_blur11_cuda(source,tile_side=tile)
                np.testing.assert_array_equal(actual.cpu().numpy(),expected)
            self.assertEqual(source._version, version)
            np.testing.assert_array_equal(source.cpu().numpy(),image)

    def test_mean_and_exact_dyadic_gaussian_borders_match_pinned_adaptive(self):
        rng = np.random.default_rng(411303)
        cases = [(method,k) for method,kernels in ((0,(3,51,255)),(1,(3,5,7,9))) for k in kernels]
        for shape in ((1,1),(1,29),(23,1),(2,3),(27,31)):
            image = rng.integers(0,256,shape,dtype=np.uint8)
            expected = self.oracle('adaptive',image=image.tolist(),cases=cases)
            source = torch.as_tensor(image,device='cuda')
            for (method,size), original in zip(cases,expected):
                actual = card.adaptive_card_threshold_cuda(source,method,size,tile_side=13)
                np.testing.assert_array_equal(actual.cpu().numpy(),np.asarray(original,dtype=np.uint8))

    def test_large_gaussian_certified_decisions_match_pinned_default_and_maximum(self):
        image = np.full((37,43),37,dtype=np.uint8)
        image[:,21:] = 219
        cases = [(1,k) for k in (11,13,15,51,127,255)]
        expected = self.oracle('adaptive',image=image.tolist(),cases=cases)
        source = torch.as_tensor(image,device='cuda')
        for (method,size), original in zip(cases,expected):
            actual = card.adaptive_card_threshold_cuda(source,method,size,tile_side=17)
            np.testing.assert_array_equal(actual.cpu().numpy(),np.asarray(original,dtype=np.uint8))
        self.assertTrue(any(not np.array_equal(expected[0],other) for other in expected[1:]))

    def test_actual_public_detector_preprocessing_bgr_and_both_methods(self):
        image = np.full((41,53,3),(220,200,180),dtype=np.uint8)
        image[9:31,17:39] = (20,30,40)
        source = torch.as_tensor(image,device='cuda')
        for method in (0,1):
            expected = self.oracle('public',image=image.tolist(),method=method,size=51)
            result = card.preprocess_color_card_cuda(source,method,51,tile_side=19)
            self.assertEqual((result.adaptive_method,result.block_size),(method,51))
            for actual,original in zip((result.gray,result.blurred,result.threshold),expected):
                self.assertEqual(actual.device,source.device)
                self.assertEqual(actual.dtype,torch.uint8)
                np.testing.assert_array_equal(actual.cpu().numpy(),np.asarray(original,dtype=np.uint8))
        np.testing.assert_array_equal(source.cpu().numpy(),image)

    def test_interval_certifies_binary_predicate_even_when_mean_bytes_differ(self):
        means = torch.tensor([[100.5,101.5,253.5,254.5]],dtype=torch.float64,device='cuda')
        # Both endpoints give false, despite distinct rounded means.
        high = torch.tensor([[101,102,254,255]],dtype=torch.uint8,device='cuda')
        self.assertFalse(bool(card._certified_decision(means,high,1e-5).any()))
        low = torch.tensor([[97,98,250,251]],dtype=torch.uint8,device='cuda')
        self.assertTrue(bool(card._certified_decision(means,low,1e-5).all()))
        ambiguous = torch.tensor([[99,100,252,253]],dtype=torch.uint8,device='cuda')
        with self.assertRaises(card.ColorCardPreprocessPrecisionError):
            card._certified_decision(means,ambiguous,1e-5)

    def test_exact_half_ties_equality_and_saturation_use_source_byte_rule(self):
        means = torch.tensor([[100.5,101.5,253.5,254.5,-1,256]],dtype=torch.float64,device='cuda')
        source = torch.tensor([[98,100,252,252,0,253]],dtype=torch.uint8,device='cuda')
        expected = torch.tensor([[True,True,True,True,False,True]],device='cuda')
        torch.testing.assert_close(card._certified_decision(means,source,0),expected)

    def test_binary_uncertainty_withholds_complete_mask_and_preserves_input(self):
        # This actual public preprocessing fixture reaches an undecidable binary
        # boundary after the byte blur; the native path must withhold its result.
        image = np.full((41,53,3),(230,190,170),dtype=np.uint8)
        image[9:31,17:39] = (15,45,75)
        self.oracle('public',image=image.tolist(),method=1,size=51)
        source = torch.as_tensor(image,device='cuda')
        version = source._version
        with self.assertRaises(card.ColorCardPreprocessPrecisionError):
            card.preprocess_color_card_cuda(source,1,51,tile_side=19)
        self.assertEqual(source._version,version)
        np.testing.assert_array_equal(source.cpu().numpy(),image)

    def test_random_binary_certificate_matches_reference_without_requiring_exact_mean(self):
        image = np.random.default_rng(411051).integers(0,256,(128,128),dtype=np.uint8)
        expected, = self.oracle('adaptive',image=image.tolist(),cases=[(1,51)])
        source = torch.as_tensor(image,device='cuda')
        actual = card.adaptive_card_threshold_cuda(source,1,51,tile_side=37)
        np.testing.assert_array_equal(actual.cpu().numpy(),np.asarray(expected,dtype=np.uint8))

    def test_mean_scalar_tail_counterexample_withholds_incorrect_mask(self):
        image = np.full((255,3),255,dtype=np.uint8)
        image[0,0],image[1,0],image[127,2] = 0,254,253
        expected, = self.oracle('adaptive',image=image.tolist(),cases=[(0,255)])
        self.assertEqual(expected[127][2],255)
        source = torch.as_tensor(image,device='cuda')
        before,version = source.clone(),source._version
        with self.assertRaisesRegex(card.ColorCardPreprocessPrecisionError,'scalar/SIMD'):
            card.adaptive_card_threshold_cuda(source,0,255,tile_side=19)
        self.assertEqual(source._version,version)
        torch.testing.assert_close(source,before)

    def test_mean_singleton_full_frame_uses_reduced_kernel_across_tiles(self):
        for shape in ((1,1),(1,255),(255,1)):
            image = np.full(shape,255,dtype=np.uint8)
            image.reshape(-1)[::2] = 254
            image.reshape(-1)[0] = 0
            image.reshape(-1)[-1] = 253
            cases = [(0,k) for k in (3,51,255)]
            expected = self.oracle('adaptive',image=image.tolist(),cases=cases)
            source = torch.as_tensor(image,device='cuda')
            for (_,size),original in zip(cases,expected):
                for tile in (17,256):
                    actual = card.adaptive_card_threshold_cuda(source,0,size,tile_side=tile)
                    np.testing.assert_array_equal(actual.cpu().numpy(),np.asarray(original,dtype=np.uint8))

    def test_work_scratch_retained_storage_preflight_before_output_allocation(self):
        source = torch.full((17,19,3),80,dtype=torch.uint8,device='cuda')
        for name in ('MAX_FILTER_WORK','MAX_SCRATCH_BYTES','MAX_RETAINED_BYTES','MAX_IMAGE_PIXELS'):
            with patch.object(card,name,1), patch.object(torch,'empty') as allocate:
                with self.assertRaises(card.ColorCardPreprocessError):
                    card.preprocess_color_card_cuda(source)
                allocate.assert_not_called()
        height,width,size,tile = 19,23,51,7
        expected = sum((min(tile,height-y)+size-1)*(min(tile,width-x)+size-1)
            +size*((min(tile,height-y)+size-1)*min(tile,width-x)
                   +min(tile,height-y)*min(tile,width-x))
            for y in range(0,height,tile) for x in range(0,width,tile))
        self.assertEqual(card._filter_work(height,width,size,tile), expected)

    def test_cancellation_and_callback_mutation_before_allocation_and_mid_pass(self):
        source = torch.full((17,19,3),80,dtype=torch.uint8,device='cuda')
        with patch.object(torch,'empty') as allocate:
            with self.assertRaises(card.ColorCardPreprocessCancelled):
                card.preprocess_color_card_cuda(source,cancel=lambda:True)
            allocate.assert_not_called()
        def mutate():
            source[0,0,0] = 81
            return False
        with patch.object(torch,'empty') as allocate:
            with self.assertRaisesRegex(card.ColorCardPreprocessError,'changed'):
                card.preprocess_color_card_cuda(source,cancel=mutate)
            allocate.assert_not_called()
        gray = torch.full((37,43),80,dtype=torch.uint8,device='cuda')
        before,version = gray.clone(),gray._version
        calls = 0
        def cancel():
            nonlocal calls
            calls += 1
            return calls == 20
        with self.assertRaises(card.ColorCardPreprocessCancelled):
            card.adaptive_card_threshold_cuda(gray,1,51,cancel=cancel,tile_side=17)
        self.assertEqual(gray._version,version)
        torch.testing.assert_close(gray,before)
        calls = 0
        def change():
            nonlocal calls
            calls += 1
            if calls == 20:
                gray[0,0] = 81
            return False
        with self.assertRaisesRegex(card.ColorCardPreprocessError,'changed'):
            card.adaptive_card_threshold_cuda(gray,1,51,cancel=change,tile_side=17)

    def test_custom_stream_and_multitile_actual_allocator_bound(self):
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            for shape,size,method in (((1,519),255,1),((259,263),255,1),((259,263),255,0)):
                source = torch.full(shape,80,dtype=torch.uint8,device='cuda')
                card.adaptive_card_threshold_cuda(source[:1,:1].contiguous(),method,size)
                stream.synchronize()
                baseline = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                result = card.adaptive_card_threshold_cuda(source,method,size)
                stream.synchronize()
                peak = torch.cuda.max_memory_allocated()-baseline-result.numel()
                self.assertLessEqual(peak,card._scratch_bytes(*shape,size,card.MAX_TILE_SIDE))
                self.assertLessEqual(peak,card.MAX_SCRATCH_BYTES)
                self.assertEqual(int(torch.count_nonzero(result)),0)
                del result,source

    def test_validation_and_changed_runtime_kernel_are_explicit_failures(self):
        gray = torch.full((5,7),80,dtype=torch.uint8,device='cuda')
        for method,size,tile in ((True,51,256),(2,51,256),(1,2,256),(1,257,256),(1,51,0)):
            with self.assertRaises(card.ColorCardPreprocessError):
                card.adaptive_card_threshold_cuda(gray,method,size,tile_side=tile)
        for invalid in (gray.cpu(),gray.to(torch.float32),gray[:,::2],gray[None],torch.empty((0,1),device='cuda',dtype=torch.uint8)):
            with self.assertRaises(card.ColorCardPreprocessError):
                card.adaptive_card_threshold_cuda(invalid)
        with patch.object(card.cv2,'getGaussianKernel',return_value=np.zeros((51,1),dtype=np.float32)):
            with patch.object(torch,'empty_like') as allocate:
                with self.assertRaisesRegex(card.ColorCardPreprocessError,'taps differ'):
                    card.adaptive_card_threshold_cuda(gray)
                allocate.assert_not_called()
        with self.assertRaises(TypeError):
            card.adaptive_card_threshold_cuda(gray,cancel=False)


if __name__ == '__main__':
    unittest.main()
