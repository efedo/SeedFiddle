"""Actual public/private pinned k-means++ IDs, centres and RNG state on CUDA."""

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
from seedvision.cuda import plantcv_kmeans_plusplus as native
from seedvision.cuda import plantcv_kmeans_random as random_source

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT/'artifacts/plantcv-reference-env/Scripts/python.exe'
SCRIPT = r'''
import hashlib,importlib.metadata,json,struct,sys
from pathlib import Path
import numpy as np
from sklearn.cluster import MiniBatchKMeans,kmeans_plusplus
import sklearn.cluster._kmeans as source
from sklearn.utils.extmath import row_norms
assert importlib.metadata.version('scikit-learn')=='1.9.1' and np.__version__=='2.2.6'
assert hashlib.sha256(Path(source.__file__).read_bytes()).hexdigest()=='0cf335b8b9b4abe67259ccdcbaf07f02bd7feee31e63a06a1cce32a87fcd375d'
def state_sha(rng):
    algorithm,keys,position,has_gauss,cached=rng.get_state()
    h=hashlib.sha256(algorithm.encode('ascii')+b'\0')
    h.update(keys.astype('<u4',copy=False).tobytes())
    h.update(struct.pack('<IId',int(position),int(has_gauss),float(cached)))
    return h.hexdigest()
results=[]
for item in json.load(sys.stdin):
    x=np.asarray(item['features'],dtype=np.float64)
    k,seed=item['k'],item['seed']
    norms=row_norms(x,squared=True)
    weights=np.ones(len(x),dtype=np.float64)
    rng=np.random.RandomState(seed)
    centres,indices=kmeans_plusplus(x,k,x_squared_norms=norms,sample_weight=weights,random_state=rng)
    private_rng=np.random.RandomState(seed)
    private_centres,private_indices=source._kmeans_plusplus(x,k,norms,weights,private_rng)
    init_rng=np.random.RandomState(seed)
    initialized=MiniBatchKMeans(n_clusters=k)._init_centroids(x,norms,'k-means++',init_rng,weights)
    assert np.array_equal(centres,private_centres) and np.array_equal(indices,private_indices)
    assert np.array_equal(centres,initialized)
    assert state_sha(rng)==state_sha(private_rng)==state_sha(init_rng)
    distances=((x[:,None,:]-centres[None,:,:])**2).sum(axis=2)
    potential=float(distances.min(axis=1).sum())
    results.append(dict(indices=indices.tolist(),centres_sha256=hashlib.sha256(centres.tobytes()).hexdigest(),
        state_sha256=state_sha(rng),potential=potential))
print(json.dumps(results))
'''


def _reference(cases):
    payload = [dict(features=features.tolist(), k=k, seed=seed) for features, k, seed in cases]
    result = subprocess.run([str(REFERENCE), '-c', SCRIPT], cwd=ROOT,
        input=json.dumps(payload), text=True, capture_output=True, timeout=90)
    if result.returncode:
        raise AssertionError(result.stderr)
    return json.loads(result.stdout)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
@functional_units
class KMeansPlusPlusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(2119)
        cls.cases = [(rng.integers(0, 256, (17, 1), dtype=np.uint16), 1, 0),
            (rng.integers(0, 256, (97, 31), dtype=np.uint16), 8, 7),
            (np.full((21, 3), 17, dtype=np.uint16), 5, 1),
            (np.asarray([[0, 0], [0, 2], [2, 0], [2, 2]], dtype=np.uint16), 3, 19),
            (rng.integers(0, 256, (129, 4096), dtype=np.uint16), 3, 2**32-1)]
        cls.reference = _reference(cls.cases)

    def assert_reference(self, result, schedule, expected):
        self.assertEqual(list(result.centre_indices), expected['indices'])
        self.assertEqual(result.centres.dtype, torch.float64)
        self.assertEqual(result.centres.device.type, 'cuda')
        self.assertTrue(result.centres.is_contiguous())
        self.assertEqual(sha256(result.centres.cpu().numpy().tobytes()).hexdigest(), expected['centres_sha256'])
        self.assertEqual(schedule.identity()['state_sha256'], expected['state_sha256'])
        self.assertEqual(result.potential, expected['potential'])

    def check_test_actual_public_private_and_estimator_init_match_ids_centres_and_state(self):
        for (features, k, seed), expected in zip(self.cases, self.reference):
            with self.subTest(shape=features.shape, clusters=k, seed=seed):
                source = torch.tensor(features, device='cuda')
                version = source._version
                schedule = random_source.MiniBatchRandomSchedule(seed)
                result = native.initialize_kmeans_plusplus(source, k, schedule, tile_rows=17)
                self.assert_reference(result, schedule, expected)
                self.assertEqual(source._version, version)
                np.testing.assert_array_equal(source.cpu().numpy(), features)
        self.assertEqual(self.reference[2]['indices'][1:], [0]*4)

    def check_test_actual_multitile_peak_and_public_checksum_fit_global_plan(self):
        features = ((np.arange(4097)[:, None]*17+np.arange(31)*29)%256).astype(np.uint16)
        expected = _reference([(features, 3, 17)])[0]
        source = torch.tensor(features, device='cuda')
        schedule = random_source.MiniBatchRandomSchedule(17)
        plan = native.preflight_kmeans_plusplus(source, 3)
        self.assertGreater(plan.rows, plan.tile_rows)
        native.initialize_kmeans_plusplus(source[:2].contiguous(), 1,
            random_source.MiniBatchRandomSchedule(0))
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        result = native.initialize_kmeans_plusplus(source, 3, schedule)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()-baseline
        self.assertLessEqual(peak, plan.workspace_bytes)
        self.assert_reference(result, schedule, expected)

    def check_test_custom_stream_and_distance_tile_repartition_preserve_public_state(self):
        features, k, seed = self.cases[1]
        source = torch.tensor(features, device='cuda')
        schedule = random_source.MiniBatchRandomSchedule(seed)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            result = native.initialize_kmeans_plusplus(source, k, schedule, tile_rows=7)
        stream.synchronize()
        self.assert_reference(result, schedule, self.reference[1])

    def check_test_early_distance_candidate_and_final_cancellation_withhold_centres(self):
        features, k, seed = self.cases[1]
        source = torch.tensor(features, device='cuda')
        version, calls = source._version, 0
        def count_polls():
            nonlocal calls
            calls += 1
            return False
        native.initialize_kmeans_plusplus(source, k, random_source.MiniBatchRandomSchedule(seed),
            tile_rows=17, cancel=count_polls)
        total = calls
        for target in (1, 4, total//2, total):
            calls = 0
            def cancel_at_target():
                nonlocal calls
                calls += 1
                return calls == target
            with self.subTest(poll=target), self.assertRaises(native.KMeansPlusPlusCancelled):
                native.initialize_kmeans_plusplus(source, k, random_source.MiniBatchRandomSchedule(seed),
                    tile_rows=17, cancel=cancel_at_target)
            self.assertEqual(calls, target)
            self.assertEqual(source._version, version)

    def check_test_rng_and_poll_storage_mutations_stop_before_following_raw_launches(self):
        features, k, seed = self.cases[1]
        original_program = _nvrtc.fixed_kernel_program(native._CUDA_SOURCE, torch.device('cuda:0'))
        for mode in ('first_rng', 'local_rng', 'callback'):
            with self.subTest(mode=mode):
                source = torch.tensor(features, device='cuda')
                schedule = random_source.MiniBatchRandomSchedule(seed)
                mutated, calls = False, 0
                original_draw = schedule.first_center if mode == 'first_rng' else schedule.local_trials
                def mutate_after_draw(*args):
                    nonlocal mutated
                    result = original_draw(*args)
                    source.resize_(0)
                    mutated = True
                    return result
                def mutate_callback():
                    nonlocal calls, mutated
                    calls += 1
                    if mode == 'callback' and calls == 4:
                        source.resize_(0)
                        mutated = True
                    return False
                class CheckedProgram:
                    def launch_1d(program, *args):
                        self.assertFalse(mutated, 'raw launch followed a storage mutation')
                        return original_program.launch_1d(*args)
                name = 'first_center' if mode == 'first_rng' else 'local_trials'
                with patch.object(_nvrtc, 'fixed_kernel_program', return_value=CheckedProgram()), \
                        patch.object(schedule, name, side_effect=mutate_after_draw):
                    with self.assertRaisesRegex(native.KMeansPlusPlusError, 'changed'):
                        native.initialize_kmeans_plusplus(source, k, schedule,
                            tile_rows=17, cancel=mutate_callback)

    def check_test_invalid_values_geometry_and_global_caps_precede_rng_and_kernel(self):
        features, k, seed = self.cases[1]
        source = torch.tensor(features, device='cuda')
        plan = native.preflight_kmeans_plusplus(source, k)
        schedule = random_source.MiniBatchRandomSchedule(seed)
        before = schedule.identity()
        with patch.object(_nvrtc, 'fixed_kernel_program', side_effect=AssertionError('kernel before validation')):
            for value in (256, 32767, 32768, 65535):
                changed = source.clone()
                changed[0, 0] = value
                with self.subTest(value=value), self.assertRaisesRegex(native.KMeansPlusPlusError, 'byte values'):
                    native.initialize_kmeans_plusplus(changed, k, schedule)
            for name, limit in (('MAX_WORK', plan.work-1), ('MAX_CUDA_LIVE_BYTES', plan.peak_live_bytes-1),
                    ('MAX_DATASET_BYTES', source.numel()*2-1), ('EXACT_INTEGER_LIMIT', source.numel()*65025)):
                with self.subTest(cap=name), patch.object(native, name, limit):
                    with self.assertRaises(native.KMeansPlusPlusError):
                        native.initialize_kmeans_plusplus(source, k, schedule)
            for value in (None, source.cpu(), source.to(torch.float64), source.transpose(0, 1), source[:0]):
                with self.subTest(raster=type(value).__name__), self.assertRaises(ValueError):
                    native.initialize_kmeans_plusplus(value, k, schedule)
            for value in (0, True, 257, source.shape[0]+1, 1.5):
                with self.subTest(clusters=value), self.assertRaises(ValueError):
                    native.initialize_kmeans_plusplus(source, value, schedule)
            for value in (0, True, 4097, 1.5):
                with self.subTest(tile=value), self.assertRaises(ValueError):
                    native.initialize_kmeans_plusplus(source, k, schedule, tile_rows=value)
            with self.assertRaises(TypeError):
                native.initialize_kmeans_plusplus(source, k, schedule, cancel=42)
        self.assertEqual(schedule.identity(), before)

    def check_test_loaded_source_binds_kernel_integer_proof_and_tile_limits(self):
        from seedvision.benchmarking.native_recipe import _module_record
        self.assertTrue(_module_record(native)['loaded_code'])
        for name, value in (('_CUDA_SOURCE', native._CUDA_SOURCE+'\n'), ('EXACT_INTEGER_LIMIT', 1),
                            ('MAX_TILE_ROWS', 1)):
            with self.subTest(constant=name), patch.object(native, name, value):
                with self.assertRaises(ValueError):
                    _module_record(native)

    def test_workflow_lifecycle(self):
        """3 related KMeansPlusPlusTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_actual_public_private_and_estimator_init_match_ids_centres_and_state)
        self.scenario(self.check_test_actual_multitile_peak_and_public_checksum_fit_global_plan)
        self.scenario(self.check_test_custom_stream_and_distance_tile_repartition_preserve_public_state)

    def test_validation_and_recovery(self):
        """4 related KMeansPlusPlusTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_early_distance_candidate_and_final_cancellation_withhold_centres)
        self.scenario(self.check_test_rng_and_poll_storage_mutations_stop_before_following_raw_launches)
        self.scenario(self.check_test_invalid_values_geometry_and_global_caps_precede_rng_and_kernel)
        self.scenario(self.check_test_loaded_source_binds_kernel_integer_proof_and_tile_limits)
