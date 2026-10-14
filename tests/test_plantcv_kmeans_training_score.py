"""Actual pinned Lloyd/inertia calls and CUDA training-score contracts."""
from hashlib import sha256
import json
import math
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

import numpy as np
import torch

from seedvision.cuda import _nvrtc
from seedvision.cuda import plantcv_kmeans_training_score as native
from seedvision.cuda.plantcv_kmeans_classifier import _decision_margin_bound

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT/'artifacts/plantcv-reference-env/Scripts/python.exe'
SCRIPT = r'''
import hashlib,importlib.metadata,json,sys
from pathlib import Path
import numpy as np
import sklearn.cluster._kmeans as source
from sklearn.cluster._k_means_common import _inertia_dense
assert importlib.metadata.version('scikit-learn')=='1.9.1'
assert np.__version__=='2.2.6'
for name,digest in (('_kmeans.py','0cf335b8b9b4abe67259ccdcbaf07f02bd7feee31e63a06a1cce32a87fcd375d'),
        ('_k_means_common.pyx','9a05c0184b411fd1d68c9033e265d4b41824f01f1a66d6c494cfb11f1780b489'),
        ('_k_means_lloyd.pyx','66ba7cad80c8000e350c0dd4014e790b424e7c0157b479c98fc7610071aa2a92')):
    assert hashlib.sha256(Path(source.__file__).with_name(name).read_bytes()).hexdigest()==digest
results=[]
for item in json.load(sys.stdin):
    x=np.ascontiguousarray(item['features'],dtype=np.float64)
    centres=np.ascontiguousarray(item['centres'],dtype=np.float64)
    weights=np.ones(len(x),dtype=np.float64)
    totals=[]
    first=None
    for threads in (1,2,4):
        labels,inertia=source._labels_inertia(x,weights,centres,n_threads=threads)
        if first is None:
            first=labels
        else:
            assert np.array_equal(first,labels)
        totals.append(float(inertia))
    rows=np.asarray([_inertia_dense(x[i:i+1],weights[i:i+1],centres,first[i:i+1],1) for i in range(len(x))])
    results.append(dict(labels=first.tolist(),inertia=totals,rows=rows.tolist(),
        rows_sha=hashlib.sha256(rows.tobytes()).hexdigest()))
print(json.dumps(results))
'''


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class KMeansTrainingScoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cases = []
        for width in (1, 2, 3, 4, 5, 7, 8, 9, 31):
            rows = ((np.arange(17)[:, None]*11+np.arange(width)*3)%40).tolist()
            cases.append(dict(features=rows, centres=[[17]*width, [200]*width], integral=True))
            cases.append(dict(features=rows, centres=[[17.125]*width, [200.375]*width], integral=False))
        cases.extend([
            dict(features=[[220], [200], [180]], centres=[[190], [210]], integral=True),
            dict(features=[[17], [200], [255]], centres=[[17.25], [17.25], [200.75]], integral=False),
            dict(features=[[0], [1], [255]], centres=[[.5], [.5]], integral=False),
            dict(features=[[255], [254], [253]], centres=[[i] for i in range(256)], integral=True),
            dict(features=((np.arange(4097)[:, None]*7+np.arange(31))%40).tolist(),
                centres=[[17.125]*31, [200.375]*31], integral=False),
            dict(features=((np.arange(129)[:, None]*7+np.arange(4096))%256).tolist(),
                centres=[[(i+j)%256 for j in range(4096)] for i in range(256)], integral=True),
        ])
        completed = subprocess.run([str(REFERENCE), '-c', SCRIPT], cwd=ROOT,
            input=json.dumps(cases), text=True, capture_output=True, timeout=90)
        if completed.returncode:
            raise AssertionError(completed.stderr)
        cls.cases, cls.reference = cases, json.loads(completed.stdout)

    def tensors(self, index=0):
        item = self.cases[index]
        return (torch.tensor(item['features'], dtype=torch.uint16, device='cuda'),
            torch.tensor(item['centres'], dtype=torch.float64, device='cuda'))

    def assert_reference(self, result, index, threads):
        item, expected = self.cases[index], self.reference[index]
        self.assertEqual(result.labels.dtype, torch.uint8)
        self.assertEqual(result.labels.device.type, 'cuda')
        self.assertEqual(result.row_distances.dtype, torch.float64)
        self.assertEqual(result.row_distances.device.type, 'cuda')
        self.assertEqual(result.labels.cpu().tolist(), expected['labels'])
        self.assertEqual(result.source_n_threads, threads)
        self.assertLessEqual(result.inertia.lo, expected['inertia'][(1, 2, 4).index(threads)])
        self.assertGreaterEqual(result.inertia.hi, expected['inertia'][(1, 2, 4).index(threads)])
        if item['integral']:
            self.assertTrue(result.inertia.is_point)
            self.assertEqual(sha256(result.row_distances.cpu().numpy().tobytes()).hexdigest(), expected['rows_sha'])
        else:
            self.assertFalse(result.inertia.is_point)
            errors = np.abs(result.row_distances.cpu().numpy()-np.asarray(expected['rows']))
            self.assertLessEqual(float(errors.max()), _decision_margin_bound(len(item['centres'][0])))

    def test_actual_pinned_labels_rows_and_inertia_threads_match_certified_result(self):
        for index in range(len(self.cases)-2):
            for threads in (1, 2, 4):
                with self.subTest(case=index, threads=threads):
                    features, centres = self.tensors(index)
                    versions = (features._version, centres._version)
                    result = native.score_kmeans_training_rows(features, centres, source_n_threads=threads)
                    self.assert_reference(result, index, threads)
                    self.assertEqual((features._version, centres._version), versions)
        # Both winning scores here are negative; clamping would invent label0.
        self.assertEqual(self.reference[18]['labels'][0], 1)
        self.assertEqual(self.reference[19]['labels'][0], 0)
        self.assertEqual(self.reference[21]['labels'][0], 255)

    def test_multitile_actual_peak_stays_bounded_and_matches_pinned_threaded_inertia(self):
        native.score_kmeans_training_rows(*self.tensors(), source_n_threads=1)
        for index in (len(self.cases)-2, len(self.cases)-1):
            values = self.tensors(index)
            plan = native.preflight_kmeans_training_score(*values, source_n_threads=4)
            torch.cuda.synchronize()
            baseline = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            result = native.score_kmeans_training_rows(*values, source_n_threads=4)
            torch.cuda.synchronize()
            self.assertLessEqual(torch.cuda.max_memory_allocated()-baseline, plan.workspace_bytes)
            self.assert_reference(result, index, 4)
            self.assertLessEqual(plan.peak_live_bytes, native.MAX_CUDA_LIVE_BYTES)

    def test_custom_stream_partition_and_mutable_centre_norm_recompute(self):
        index = len(self.cases)-2
        values = self.tensors(index)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            result = native.score_kmeans_training_rows(*values, source_n_threads=2, tile_rows=17)
        stream.synchronize()
        self.assert_reference(result, index, 2)
        features = torch.tensor([[0], [255]], dtype=torch.uint16, device='cuda')
        centres = torch.tensor([[0.], [255.]], dtype=torch.float64, device='cuda')
        first = native.score_kmeans_training_rows(features, centres, source_n_threads=1)
        centres.copy_(torch.tensor([[255.], [0.]], dtype=torch.float64, device='cuda'))
        second = native.score_kmeans_training_rows(features, centres, source_n_threads=1)
        self.assertEqual(first.labels.cpu().tolist(), [0, 1])
        self.assertEqual(second.labels.cpu().tolist(), [1, 0])

    def test_fractional_bisector_in_late_tile_withholds_all_labels(self):
        features = torch.tensor([[0], [200], [1]], dtype=torch.uint16, device='cuda')
        centres = torch.tensor([[.5], [1.5]], dtype=torch.float64, device='cuda')
        versions = (features._version, centres._version)
        with self.assertRaises(native.KMeansTrainingScorePrecisionError):
            native.score_kmeans_training_rows(features, centres, source_n_threads=4, tile_rows=2)
        self.assertEqual((features._version, centres._version), versions)

    def test_all_reduction_profile_encloses_all_actual_public_threaded_inertias(self):
        for index in (0, 1, 19, len(self.cases)-2):
            with self.subTest(case=index):
                result = native.score_kmeans_training_rows(*self.tensors(index), source_n_threads=None)
                self.assertIsNone(result.source_n_threads)
                self.assertEqual(result.source_reduction_profile, 'all-valid-openmp-reductions-v1')
                self.assertEqual(result.labels.cpu().tolist(), self.reference[index]['labels'])
                for inertia in self.reference[index]['inertia']:
                    self.assertLessEqual(result.inertia.lo, inertia)
                    self.assertGreaterEqual(result.inertia.hi, inertia)
                recorded = native.score_kmeans_training_rows(*self.tensors(index), source_n_threads=2)
                self.assertEqual(recorded.source_reduction_profile, 'recorded-openmp-threads-v1')
                self.assertEqual(result.inertia, recorded.inertia)
        with self.assertRaises(TypeError):
            native.score_kmeans_training_rows(*self.tensors())
        features = torch.tensor([[0], [200], [1]], dtype=torch.uint16, device='cuda')
        centres = torch.tensor([[.5], [1.5]], dtype=torch.float64, device='cuda')
        with self.assertRaises(native.KMeansTrainingScorePrecisionError):
            native.score_kmeans_training_rows(features, centres, source_n_threads=None)

    def test_every_callback_cancellation_and_storage_mutation_precedes_further_work(self):
        total = 0
        def count():
            nonlocal total
            total += 1
            return False
        native.score_kmeans_training_rows(*self.tensors(), source_n_threads=1, cancel=count)
        for target in (1, total//2, total):
            calls = 0
            def cancel():
                nonlocal calls
                calls += 1
                return calls == target
            values = self.tensors()
            versions = tuple(value._version for value in values)
            with self.assertRaises(native.KMeansTrainingScoreCancelled):
                native.score_kmeans_training_rows(*values, source_n_threads=1, cancel=cancel)
            self.assertEqual(calls, target)
            self.assertEqual(tuple(value._version for value in values), versions)
        original_empty = torch.empty
        for input_index in (0, 1):
            for target in range(1, total+1):
                values = self.tensors()
                calls, mutated = 0, False
                def mutate():
                    nonlocal calls, mutated
                    calls += 1
                    if calls == target:
                        values[input_index].resize_(0)
                        mutated = True
                    return False
                def allocation(*args, **kwargs):
                    self.assertFalse(mutated, 'allocation after callback mutation')
                    return original_empty(*args, **kwargs)
                with self.subTest(input=input_index, poll=target), patch.object(torch, 'empty', side_effect=allocation):
                    with self.assertRaisesRegex(native.KMeansTrainingScoreError, 'changed'):
                        native.score_kmeans_training_rows(*values, source_n_threads=1, cancel=mutate)
                self.assertEqual(calls, target)

    def test_global_caps_invalid_shapes_values_and_callbacks_reject_before_kernel(self):
        values = self.tensors()
        for constant in ('MAX_DATASET_BYTES', 'MAX_WORK', 'MAX_SCRATCH_BYTES', 'MAX_CUDA_LIVE_BYTES'):
            with self.subTest(cap=constant), patch.object(native, constant, 1):
                with patch.object(_nvrtc, 'fixed_kernel_program', side_effect=AssertionError('kernel before cap')):
                    with self.assertRaises(native.KMeansTrainingScoreError):
                        native.score_kmeans_training_rows(*values, source_n_threads=1)
        for threads in (0, -1, True, 1.5):
            with self.assertRaises(native.KMeansTrainingScoreError):
                native.score_kmeans_training_rows(*values, source_n_threads=threads)
        for tile in (0, 4097, True):
            with self.assertRaises(native.KMeansTrainingScoreError):
                native.score_kmeans_training_rows(*values, source_n_threads=1, tile_rows=tile)
        with self.assertRaisesRegex(native.KMeansTrainingScoreError, 'bool'):
            native.score_kmeans_training_rows(*values, source_n_threads=1, cancel=lambda: 1)
        for feature in (256, 65535):
            bad = torch.tensor([[feature]], dtype=torch.uint16, device='cuda')
            with patch.object(_nvrtc, 'fixed_kernel_program', side_effect=AssertionError('kernel before value validation')):
                with self.assertRaisesRegex(native.KMeansTrainingScoreError, 'bytes'):
                    native.score_kmeans_training_rows(bad, values[1][:, :1].contiguous(), source_n_threads=1)
        for centre in (float('nan'), float('inf'), -1., 256.):
            bad = torch.tensor([[centre]], dtype=torch.float64, device='cuda')
            with self.assertRaisesRegex(native.KMeansTrainingScoreError, 'finite'):
                native.score_kmeans_training_rows(values[0][:, :1].contiguous(), bad, source_n_threads=1)
        for inputs in ((values[0].cpu(), values[1]), (values[0], values[1].float()),
                (values[0][:, :0], values[1][:, :0]), (values[0].T, values[1]),
                (values[0], values[1][:, :0])):
            with self.assertRaises(native.KMeansTrainingScoreError):
                native.score_kmeans_training_rows(*inputs, source_n_threads=1)

    def test_loaded_source_records_bind_kernel_limits_and_pinned_cython_identity(self):
        from seedvision.benchmarking.native_recipe import _module_record
        self.assertTrue(_module_record(native)['loaded_code'])
        for name, value in (('MAX_WORK', 1), ('SKLEARN_COMMON_SHA256', 'changed')):
            with patch.object(native, name, value):
                with self.assertRaises(ValueError):
                    _module_record(native)
