"""Exact pinned sklearn dense updates and CUDA row-order/resource contracts."""

from tests.functional_units import functional_units
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

import numpy as np
import torch

from seedvision.cuda import _nvrtc
from seedvision.cuda import plantcv_kmeans_minibatch as native

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT/'artifacts/plantcv-reference-env/Scripts/python.exe'
SCRIPT = r'''
import hashlib,importlib.metadata,json,sys
from pathlib import Path
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import sklearn.cluster._kmeans as source
from sklearn.cluster._k_means_minibatch import _minibatch_update_dense
assert importlib.metadata.version('scikit-learn')=='1.9.1'
assert importlib.metadata.version('numpy')=='2.2.6'
assert hashlib.sha256(Path(source.__file__).read_bytes()).hexdigest()=='0cf335b8b9b4abe67259ccdcbaf07f02bd7feee31e63a06a1cce32a87fcd375d'
assert hashlib.sha256(Path(source.__file__).with_name('_k_means_minibatch.pyx').read_bytes()).hexdigest()=='b6ffd63a5a9c6f9162d77e1568911fd3540c4ceaaa8d5243cb341470e758fff5'
results=[]
for item in json.load(sys.stdin):
    x=np.ascontiguousarray(item['features'],dtype=np.float64)
    centres=np.ascontiguousarray(item['centres'],dtype=np.float64)
    counts=np.asarray(item['counts'],dtype=np.float64)
    if item.get('public'):
        model=MiniBatchKMeans(n_clusters=len(centres),init=centres.copy(),n_init=1,
            reassignment_ratio=0,random_state=17).partial_fit(centres)
        model.cluster_centers_=centres.copy()
        model._counts=counts.copy()
        model._n_threads=1
        labels=model.predict(x)
        model.partial_fit(x)
        updated,newcounts=model.cluster_centers_,model._counts
    else:
        labels=np.asarray(item['labels'],dtype=np.int32)
        updated=np.empty_like(centres)
        newcounts=counts.copy()
        _minibatch_update_dense(x,np.ones(len(x)),centres,updated,newcounts,labels,1)
    results.append(dict(labels=labels.tolist(),
        centres_sha256=hashlib.sha256(updated.tobytes()).hexdigest(),
        counts_sha256=hashlib.sha256(newcounts.tobytes()).hexdigest()))
print(json.dumps(results))
'''


def _reference(requests):
    result = subprocess.run([str(REFERENCE), '-c', SCRIPT], cwd=ROOT,
        input=json.dumps(requests), text=True, capture_output=True, timeout=90)
    if result.returncode:
        raise AssertionError(result.stderr)
    return json.loads(result.stdout)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
@functional_units
class KMeansMiniBatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(719)
        features = np.concatenate((rng.integers(0, 40, (517, 5)), rng.integers(90, 145, (514, 5))))
        public = dict(features=features.tolist(), centres=[[20.25]*5, [120.125]*5, [240.5]*5],
            counts=[0, 2**48+17, 123], public=True)
        fractional = dict(features=(rng.random((257, 7))*255).tolist(),
            centres=(rng.random((8, 7))*255).tolist(), counts=[0, 1, 2, 3, 2**44, 0, 7, 19],
            labels=(np.arange(257)%7).tolist())
        signed_zero = dict(features=[[255., 0., 17.]]*17,
            centres=[[255., 0., 17.], [-0.0]*3], counts=[2**53-17, 9], labels=[0]*17)
        rows, width, clusters = 129, 4096, 256
        stress = dict(features=((np.arange(rows)[:, None]*17+np.arange(width))%256).astype(float).tolist(),
            centres=(((np.arange(clusters)[:, None]+np.arange(width))%250)+.125).tolist(),
            counts=(np.arange(clusters)*17).tolist(), labels=((np.arange(rows)*3)%clusters).tolist())
        signed_count = dict(features=[[17., 42.]]*3,
            centres=[[17., 42.], [-0.0, 13.]], counts=[0, -0.0], labels=[0]*3)
        cls.requests = [public, fractional, signed_zero, stress, signed_count]
        cls.reference = _reference(cls.requests)

    def tensors(self, index=0):
        case, expected = self.requests[index], self.reference[index]
        return (torch.tensor(case['features'], dtype=torch.float64, device='cuda'),
            torch.tensor(expected['labels'], dtype=torch.uint8, device='cuda'),
            torch.tensor(case['centres'], dtype=torch.float64, device='cuda'),
            torch.tensor(case['counts'], dtype=torch.float64, device='cuda'))

    def assert_reference(self, result, index):
        expected = self.reference[index]
        self.assertEqual(result.centres.dtype, torch.float64)
        self.assertEqual(result.centres.device.type, 'cuda')
        self.assertEqual(sha256(result.centres.cpu().numpy().tobytes()).hexdigest(), expected['centres_sha256'])
        self.assertEqual(sha256(result.counts.cpu().numpy().tobytes()).hexdigest(), expected['counts_sha256'])

    def check_test_actual_public_partial_fit_and_cython_updates_match_binary64_bytes(self):
        for index in range(3):
            with self.subTest(case=index):
                values = self.tensors(index)
                saved = tuple(value.clone() for value in values)
                versions = tuple(value._version for value in values)
                result = native.update_unit_weight_centres(*values)
                self.assert_reference(result, index)
                self.assertEqual(tuple(value._version for value in values), versions)
                for value, expected in zip(values, saved):
                    self.assertTrue(torch.equal(value, expected))
                self.assertNotEqual(result.centres.data_ptr(), values[2].data_ptr())
                self.assertNotEqual(result.counts.data_ptr(), values[3].data_ptr())
        values = self.tensors(2)
        result = native.update_unit_weight_centres(*values)
        self.assertTrue(bool(torch.signbit(result.centres[1]).all().item()))

    def check_test_empty_cluster_preserves_negative_zero_count_bytes_from_actual_cython(self):
        result = native.update_unit_weight_centres(*self.tensors(4))
        self.assert_reference(result, 4)
        self.assertTrue(bool(torch.signbit(result.counts[1]).item()))

    def check_test_maximum_feature_and_cluster_tables_match_public_cython_and_peak_bound(self):
        values = self.tensors(3)
        # Warm compiler/context separately so the measured phase is the update.
        probe = self.tensors(2)
        native.update_unit_weight_centres(*probe)
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        result = native.update_unit_weight_centres(*values)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()-baseline
        self.assertLessEqual(peak, native.MAX_SCRATCH_BYTES)
        self.assert_reference(result, 3)

    def check_test_custom_stream_and_row_tile_repartition_preserve_source_order(self):
        values = self.tensors()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream), patch.object(native, 'UPDATE_TILE_ROWS', 17):
            result = native.update_unit_weight_centres(*values)
        stream.synchronize()
        self.assert_reference(result, 0)

    def check_test_initial_mid_tile_and_final_cancel_withhold_outputs_and_preserve_inputs(self):
        values = self.tensors()
        versions = tuple(value._version for value in values)
        calls = 0
        def count_polls():
            nonlocal calls
            calls += 1
            return False
        native.update_unit_weight_centres(*values, cancel=count_polls)
        total = calls
        for target in (1, total-2, total):
            calls = 0
            def cancel_at_target():
                nonlocal calls
                calls += 1
                return calls == target
            with self.subTest(poll=target), self.assertRaises(native.KMeansMiniBatchCancelled):
                native.update_unit_weight_centres(*values, cancel=cancel_at_target)
            self.assertEqual(calls, target)
            self.assertEqual(tuple(value._version for value in values), versions)
        calls = 0
        def mutate_at_final_poll():
            nonlocal calls
            calls += 1
            if calls == total:
                values[2][0, 0].fill_(42)
            return False
        with self.assertRaisesRegex(native.KMeansMiniBatchError, 'changed'):
            native.update_unit_weight_centres(*values, cancel=mutate_at_final_poll)

    def check_test_preflight_mutation_is_rejected_before_kernel_or_output_allocation(self):
        values = self.tensors(2)
        calls = 0
        def mutate_during_validation():
            nonlocal calls
            calls += 1
            if calls == 2:
                values[0][0, 0].fill_(0)
            return False
        with patch.object(_nvrtc, 'fixed_kernel_program', side_effect=AssertionError('created changed-input kernel')):
            with self.assertRaisesRegex(native.KMeansMiniBatchError, 'during preflight'):
                native.update_unit_weight_centres(*values, cancel=mutate_during_validation)

    def check_test_callback_storage_mutations_stop_before_any_subsequent_allocation_or_launch(self):
        expected_launches = (
            [], ['init_accum'], ['init_accum', 'count_rows', 'add_rows'],
            ['init_accum', 'count_rows', 'add_rows', 'count_rows', 'add_rows'],
            ['init_accum', 'count_rows', 'add_rows', 'count_rows', 'add_rows', 'finish_update'],
        )
        original_empty, original_zeros = torch.empty_like, torch.zeros_like
        for input_index in range(4):
            for target, expected in enumerate(expected_launches, 1):
                with self.subTest(input=input_index, poll=target):
                    values = self.tensors()
                    dimensions = native._preflight(*values, None)
                    calls, mutated, launches = 0, False, []
                    def mutate_storage():
                        nonlocal calls, mutated
                        calls += 1
                        if calls == target:
                            values[input_index].resize_(0)
                            mutated = True
                        return False
                    def checked_allocation(original, *args, **kwargs):
                        self.assertFalse(mutated, 'allocation followed a storage mutation')
                        return original(*args, **kwargs)
                    class RecordingProgram:
                        def launch_1d(program, name, *args):
                            self.assertFalse(mutated, 'raw CUDA launch followed a storage mutation')
                            launches.append(name)
                    with patch.object(native, '_preflight', return_value=dimensions), \
                            patch.object(_nvrtc, 'fixed_kernel_program', return_value=RecordingProgram()), \
                            patch.object(torch, 'empty_like', side_effect=lambda *a, **kw:
                                checked_allocation(original_empty, *a, **kw)), \
                            patch.object(torch, 'zeros_like', side_effect=lambda *a, **kw:
                                checked_allocation(original_zeros, *a, **kw)):
                        with self.assertRaisesRegex(native.KMeansMiniBatchError, 'changed'):
                            native.update_unit_weight_centres(*values, cancel=mutate_storage)
                    self.assertEqual(calls, target)
                    self.assertEqual(launches, expected)

    def check_test_shapes_types_values_counts_labels_and_work_limits_reject_before_kernel(self):
        values = self.tensors(2)
        variants = []
        for index, replacement in ((0, values[0].cpu()), (0, values[0].to(torch.float32)),
                (0, values[0][:0]), (0, values[0].transpose(0, 1)), (1, values[1].to(torch.int64)),
                (1, torch.full_like(values[1], 2)), (2, values[2][:, :2].contiguous()),
                (3, values[3][:1]), (3, torch.tensor([0.5, 9.], device='cuda', dtype=torch.float64)),
                (3, torch.tensor([2**53, 9.], device='cuda', dtype=torch.float64))):
            changed = list(values)
            changed[index] = replacement
            variants.append(changed)
        for bad in (float('nan'), float('inf'), -1., 256.):
            changed = list(values)
            changed[0] = values[0].clone()
            changed[0][0, 0] = bad
            variants.append(changed)
        with patch.object(_nvrtc, 'fixed_kernel_program', side_effect=AssertionError('kernel created before validation')):
            for index, case in enumerate(variants):
                with self.subTest(case=index), self.assertRaises(native.KMeansMiniBatchError):
                    native.update_unit_weight_centres(*case)
            for name in ('MAX_WORK', 'MAX_SCRATCH_BYTES', 'MAX_CUDA_LIVE_BYTES'):
                with self.subTest(cap=name), patch.object(native, name, 1):
                    with self.assertRaises(native.KMeansMiniBatchError):
                        native.update_unit_weight_centres(*values)
            # The accumulator alone is insufficient: count-row and init/final
            # visits are charged before constructing a program as well.
            only_accumulator_work = values[2].shape[0]*values[0].numel()
            with patch.object(native, 'MAX_WORK', only_accumulator_work):
                with self.assertRaisesRegex(native.KMeansMiniBatchError, 'work'):
                    native.update_unit_weight_centres(*values)
            with self.assertRaises(TypeError):
                native.update_unit_weight_centres(*values, cancel=1)

    def check_test_parallel_launcher_covers_multiple_blocks_and_bounds_signed32_arguments(self):
        source = r'''extern "C" __global__ void write_items(int* out, int count) {
            int index = blockIdx.x*blockDim.x+threadIdx.x;
            if (index<count) out[index]=index;
        }'''
        result = torch.full((260,), -1, dtype=torch.int32, device='cuda')
        program = _nvrtc.fixed_kernel_program(source, result.device)
        program.launch_1d('write_items', (result,), (257,), 257)
        self.assertTrue(torch.equal(result[:257], torch.arange(257, device='cuda', dtype=torch.int32)))
        self.assertTrue(bool((result[257:] == -1).all().item()))
        for work in (0, -1, True, 2**31, 1.5):
            with self.subTest(work=work), self.assertRaises(ValueError):
                program.launch_1d('write_items', (result,), (257,), work)
        for argument in (True, 2**31, -2**31-1, 1.5):
            with self.subTest(argument=argument), self.assertRaises(ValueError):
                program.launch_1d('write_items', (result,), (argument,), 1)

    def check_test_loaded_source_records_cover_kernel_count_limit_and_parallel_launcher(self):
        from seedvision.benchmarking.native_recipe import _module_record
        from seedvision.cuda import plantcv_kmeans_training_data as data
        for module in (native, data, _nvrtc):
            self.assertTrue(_module_record(module)['loaded_code'])
        for name, value in (('EXACT_COUNT_LIMIT', 10), ('_CUDA_SOURCE', native._CUDA_SOURCE+'\n'),
                            ('UPDATE_TILE_ROWS', 17)):
            with self.subTest(constant=name), patch.object(native, name, value):
                with self.assertRaises(ValueError):
                    _module_record(native)

    def test_workflow_lifecycle(self):
        """5 related KMeansMiniBatchTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_actual_public_partial_fit_and_cython_updates_match_binary64_bytes)
        self.scenario(self.check_test_empty_cluster_preserves_negative_zero_count_bytes_from_actual_cython)
        self.scenario(self.check_test_maximum_feature_and_cluster_tables_match_public_cython_and_peak_bound)
        self.scenario(self.check_test_custom_stream_and_row_tile_repartition_preserve_source_order)
        self.scenario(self.check_test_parallel_launcher_covers_multiple_blocks_and_bounds_signed32_arguments)

    def test_validation_and_recovery(self):
        """5 related KMeansMiniBatchTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_initial_mid_tile_and_final_cancel_withhold_outputs_and_preserve_inputs)
        self.scenario(self.check_test_preflight_mutation_is_rejected_before_kernel_or_output_allocation)
        self.scenario(self.check_test_callback_storage_mutations_stop_before_any_subsequent_allocation_or_launch)
        self.scenario(self.check_test_shapes_types_values_counts_labels_and_work_limits_reject_before_kernel)
        self.scenario(self.check_test_loaded_source_records_cover_kernel_count_limit_and_parallel_launcher)
