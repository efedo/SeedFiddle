"""Pinned NumPy/sklearn tolerance values and bounded CUDA producers."""
import json
import math
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

import numpy as np
import torch

from seedvision.cuda import _nvrtc
from seedvision.cuda import plantcv_kmeans_tolerance as native

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT/'artifacts/plantcv-reference-env/Scripts/python.exe'
SCRIPT = r'''
import hashlib,json,sys
from pathlib import Path
import numpy as np
import numpy._core._methods as methods
import sklearn.cluster._kmeans as source
assert np.__version__=='2.2.6'
assert hashlib.sha256(Path(methods.__file__).read_bytes()).hexdigest()=='41068be3404b05b582867995683f73619b41a9fc72b9f0cc7983f55f53131146'
assert hashlib.sha256(Path(source.__file__).read_bytes()).hexdigest()=='0cf335b8b9b4abe67259ccdcbaf07f02bd7feee31e63a06a1cce32a87fcd375d'
request=json.load(sys.stdin)
result=dict(tolerance=[],shift=[])
for item in request['tolerance']:
    x=np.ascontiguousarray(item['features'],dtype=np.float64)
    result['tolerance'].append(float(source._tolerance(x,item['tol'])))
for item in request['shift']:
    old=np.ascontiguousarray(item['old'],dtype=np.float64)
    new=np.ascontiguousarray(item['new'],dtype=np.float64)
    result['shift'].append(float(np.sum((new-old)**2)))
print(json.dumps(result))
'''


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class KMeansToleranceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cases = [dict(features=[[17, 42]]*7, tol=.001),
            dict(features=[[17, 42]], tol=.1), dict(features=[[0], [255]], tol=1.),
            dict(features=((np.arange(4097)[:, None]*7+np.arange(31))%256).tolist(), tol=.0001),
            dict(features=((np.arange(17)[:, None]*7+np.arange(4096))%256).tolist(), tol=.1)]
        old = (((np.arange(256)[:, None]+np.arange(4096))%255)+.125)
        shifts = [dict(old=[[17., 42.]], new=[[17., 42.]]),
            dict(old=[[0., 255., 17.125, 42.75]], new=[[255., 0., 17.25, 42.875]]),
            dict(old=[[0.]], new=[[math.ulp(0.)]]),
            dict(old=old.tolist(), new=(old+.03125).tolist())]
        completed = subprocess.run([str(REFERENCE), '-c', SCRIPT], cwd=ROOT,
            input=json.dumps(dict(tolerance=cases, shift=shifts)), text=True, capture_output=True, timeout=90)
        if completed.returncode:
            raise AssertionError(completed.stderr)
        cls.cases, cls.shifts, cls.reference = cases, shifts, json.loads(completed.stdout)

    def test_actual_pinned_two_pass_variance_and_shift_values_are_enclosed(self):
        for case, expected in zip(self.cases, self.reference['tolerance']):
            x = torch.tensor(case['features'], dtype=torch.uint16, device='cuda')
            result = native.scaled_kmeans_tolerance(x, case['tol'])
            self.assertLessEqual(result.lo, expected)
            self.assertGreaterEqual(result.hi, expected)
            self.assertEqual(result.is_point, expected == 0.)
        for case, expected in zip(self.shifts, self.reference['shift']):
            old = torch.tensor(case['old'], dtype=torch.float64, device='cuda')
            new = torch.tensor(case['new'], dtype=torch.float64, device='cuda')
            result = native.centre_shift_interval(old, new)
            self.assertLessEqual(result.lo, expected)
            self.assertGreaterEqual(result.hi, expected)
        self.assertEqual(native.centre_shift_interval(old, old).lo, 0.)

    def test_zero_tolerance_bypasses_content_scan_compiler_and_output_allocation(self):
        x = torch.tensor([[65535]], dtype=torch.uint16, device='cuda')
        with patch.object(_nvrtc, 'fixed_kernel_program', side_effect=AssertionError('compiled tol0')):
            with patch.object(torch, 'empty', side_effect=AssertionError('allocated tol0')):
                result = native.scaled_kmeans_tolerance(x, 0.)
        self.assertTrue(result.is_point)
        self.assertEqual(result.lo, 0.)

    def test_custom_stream_row_tiles_and_actual_scratch_peak(self):
        case = self.cases[3]
        x = torch.tensor(case['features'], dtype=torch.uint16, device='cuda')
        native.scaled_kmeans_tolerance(x, case['tol'])
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        with torch.cuda.stream(stream), patch.object(native, 'MOMENT_TILE_ROWS', 17):
            result = native.scaled_kmeans_tolerance(x, case['tol'])
        stream.synchronize()
        self.assertLessEqual(torch.cuda.max_memory_allocated()-baseline, native.MAX_SCRATCH_BYTES)
        self.assertLessEqual(result.lo, self.reference['tolerance'][3])
        self.assertGreaterEqual(result.hi, self.reference['tolerance'][3])

    def test_caps_invalid_values_and_overflow_are_explicit(self):
        x = torch.tensor([[0], [255]], dtype=torch.uint16, device='cuda')
        old, new = x.double(), (255-x.view(torch.int16)).double()
        for constant in ('MAX_WORK', 'MAX_SCRATCH_BYTES', 'MAX_CUDA_LIVE_BYTES'):
            with patch.object(native, constant, 1):
                with patch.object(_nvrtc, 'fixed_kernel_program', side_effect=AssertionError('compiled over cap')):
                    with self.assertRaises(native.KMeansToleranceError):
                        native.scaled_kmeans_tolerance(x, .1)
                    with self.assertRaises(native.KMeansToleranceError):
                        native.centre_shift_interval(old, new)
        for tol in (-1., float('nan'), float('inf'), True):
            with self.assertRaises(native.KMeansToleranceError):
                native.scaled_kmeans_tolerance(x, tol)
        with self.assertRaisesRegex(native.KMeansToleranceError, 'finite binary64'):
            native.scaled_kmeans_tolerance(x, 1e308)
        with self.assertRaisesRegex(native.KMeansToleranceError, 'rounded bytes'):
            native.scaled_kmeans_tolerance(torch.tensor([[256]], dtype=torch.uint16, device='cuda'), .1)
        for value in (float('nan'), -1., 256.):
            with self.assertRaises(native.KMeansToleranceError):
                native.centre_shift_interval(old, torch.full_like(new, value))

    def test_cancellation_and_every_callback_mutation_precede_subsequent_work(self):
        x = torch.tensor([[0], [255]], dtype=torch.uint16, device='cuda')
        old, new = x.double(), (255-x.view(torch.int16)).double()
        for operation, inputs in ((lambda values, cancel: native.scaled_kmeans_tolerance(values[0], .1, cancel=cancel), (x,)),
                (lambda values, cancel: native.centre_shift_interval(*values, cancel=cancel), (old, new))):
            total = 0
            def count():
                nonlocal total
                total += 1
                return False
            operation(inputs, count)
            for target in (1, total):
                calls = 0
                def cancel():
                    nonlocal calls
                    calls += 1
                    return calls == target
                with self.assertRaises(native.KMeansToleranceCancelled):
                    operation(inputs, cancel)
            for which in range(len(inputs)):
                for target in range(1, total+1):
                    values = tuple(value.clone() for value in inputs)
                    calls = 0
                    def mutate():
                        nonlocal calls
                        calls += 1
                        if calls == target:
                            values[which].resize_(0)
                        return False
                    with self.assertRaisesRegex(native.KMeansToleranceError, 'changed'):
                        operation(values, mutate)
                    self.assertEqual(calls, target)

    def test_loaded_source_binds_integer_tiles_and_numeric_limits(self):
        from seedvision.benchmarking.native_recipe import _module_record
        self.assertTrue(_module_record(native)['loaded_code'])
        for name, value in (('MOMENT_TILE_ROWS', 1), ('SKLEARN_KMEANS_SHA256', 'changed'),
                ('NUMPY_VERSION', 'changed'), ('NUMPY_METHODS_SHA256', 'changed')):
            with patch.object(native, name, value):
                with self.assertRaises(ValueError):
                    _module_record(native)
