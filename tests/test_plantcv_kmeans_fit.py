"""Complete native fitting compared to traced actual public MiniBatchKMeans.fit."""

from tests.functional_units import functional_units
from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import unittest
import warnings
from unittest.mock import patch

import numpy as np
import torch

from seedvision.cuda import plantcv_kmeans_fit as native

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT/'artifacts/plantcv-reference-env/Scripts/python.exe'
SCRIPT = r'''
import hashlib,importlib.metadata,json,struct,sys,warnings
from pathlib import Path
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import sklearn.cluster._kmeans as source
assert importlib.metadata.version('scikit-learn')=='1.9.1'
assert np.__version__=='2.2.6'
assert hashlib.sha256(Path(source.__file__).read_bytes()).hexdigest()=='0cf335b8b9b4abe67259ccdcbaf07f02bd7feee31e63a06a1cce32a87fcd375d'
def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def state_sha(rng):
    algorithm,keys,position,has_gauss,cached=rng.get_state()
    h=hashlib.sha256(algorithm.encode('ascii')+b'\0')
    h.update(keys.astype('<u4',copy=False).tobytes())
    h.update(struct.pack('<IId',int(position),int(has_gauss),float(cached)))
    return h.hexdigest()
class TracedRandomState(np.random.RandomState):
    def __init__(self,seed):
        super().__init__(seed); self.calls=[]
    def record(self,kind,value):
        a=np.asarray(value)
        self.calls.append(dict(kind=kind,shape=list(a.shape),dtype=a.dtype.str,sha=digest(a),state=state_sha(self)))
        return value
    def randint(self,low,high=None,size=None,dtype=int):
        return self.record('randint',super().randint(low,high,size))
    def choice(self,a,size=None,replace=True,p=None):
        return self.record('choice',super().choice(a,size=size,replace=replace,p=p))
    def uniform(self,low=0.,high=1.,size=None):
        return self.record('uniform',super().uniform(low,high,size=size))
class TracedModel(MiniBatchKMeans):
    def _init_centroids(self,*args,**kwargs):
        result=super()._init_centroids(*args,**kwargs)
        self.initial.append(digest(result)); return result
    def _random_reassign(self):
        before=self._n_since_last_reassign
        counts=digest(self._counts)
        due=super()._random_reassign()
        self.timers.append([before,counts,bool(due),self._n_since_last_reassign]); return due
    def _mini_batch_convergence(self,step,n_steps,n_samples,centers_squared_diff,batch_inertia):
        result=super()._mini_batch_convergence(step,n_steps,n_samples,centers_squared_diff,batch_inertia)
        self.convergence.append(dict(inertia=float(batch_inertia),difference=float(centers_squared_diff),stop=bool(result),
            ewa=None if self._ewa_inertia is None else float(self._ewa_inertia),
            minimum=None if self._ewa_inertia_min is None else float(self._ewa_inertia_min),counter=self._no_improvement))
        return result
original_step=source._mini_batch_step
results=[]
for item in json.load(sys.stdin):
    rng=TracedRandomState(item['seed'])
    params=item['parameters']
    model=TracedModel(n_clusters=item['k'],random_state=rng,**params)
    model.initial=[]; model.timers=[]; model.convergence=[]
    updates=[]
    def step(*args,**kwargs):
        record=dict(old_centres=digest(kwargs['centers']),old_counts=digest(kwargs['weight_sums']))
        try:
            return original_step(*args,**kwargs)
        finally:
            record.update(centres=digest(kwargs['centers_new']),counts=digest(kwargs['weight_sums']))
            updates.append(record)
    source._mini_batch_step=step
    error=None
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('always')
        try:
            model.fit(np.ascontiguousarray(item['features'],dtype=np.uint16))
        except Exception as exception:
            error=type(exception).__name__
    record=dict(error=error,threads=int(model._n_threads),batch=model._batch_size,init_size=model._init_size,
        n_init=model._n_init,tol=float(model._tol),calls=rng.calls,state=state_sha(rng),initial=model.initial,
        timers=model.timers,convergence=model.convergence,updates=updates,
        warnings=[type(notice.message).__name__ for notice in notices])
    if error is None:
        record.update(centres=digest(model.cluster_centers_),counts=digest(model._counts),
            labels=model.labels_.tolist() if params.get('compute_labels',True) else None,
            inertia=float(model.inertia_),steps=model.n_steps_,iterations=model.n_iter_)
    results.append(record)
source._mini_batch_step=original_step
print(json.dumps(results))
'''

THREAD_SCRIPT = r'''
import hashlib,importlib.metadata,json,sys
from pathlib import Path
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import sklearn.cluster._kmeans as source
from threadpoolctl import threadpool_limits
assert importlib.metadata.version('scikit-learn')=='1.9.1'
assert np.__version__=='2.2.6'
assert hashlib.sha256(Path(source.__file__).read_bytes()).hexdigest()=='0cf335b8b9b4abe67259ccdcbaf07f02bd7feee31e63a06a1cce32a87fcd375d'
def digest(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
results=[]
for item in json.load(sys.stdin):
    outputs=[]
    for threads in (1,2,4):
        with threadpool_limits(limits=threads,user_api='openmp'):
            model=MiniBatchKMeans(n_clusters=item['k'],random_state=item['seed'],**item['parameters'])
            model.fit(np.ascontiguousarray(item['features'],dtype=np.uint16))
        assert model._n_threads==threads
        outputs.append(dict(threads=int(model._n_threads),centres=digest(model.cluster_centers_),
            counts=digest(model._counts),labels=model.labels_.tolist(),inertia=float(model.inertia_),
            steps=model.n_steps_,iterations=model.n_iter_))
    results.append(outputs)
print(json.dumps(results))
'''


def _tensor_sha(value):
    return sha256(value.cpu().numpy().tobytes()).hexdigest()


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
@functional_units
class KMeansFitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data = [[(10, 100, 220)[i % 3]+(i*7+j*3)%17 for j in range(3)] for i in range(79)]
        cases = [dict(features=data, k=3, seed=19, parameters=dict(max_iter=3, batch_size=7, n_init=2, init_size=21, max_no_improvement=None)),
            dict(features=data, k=3, seed=3, parameters=dict(max_iter=3, batch_size=7, init_size=79)),
            dict(features=[[0, 0]]*35, k=5, seed=2**32-1, parameters=dict(max_iter=2, batch_size=7, n_init=2, max_no_improvement=None)),
            dict(features=data[:17], k=3, seed=31, parameters=dict(max_iter=2, batch_size=100, init_size=1, n_init=2, max_no_improvement=0)),
            dict(features=data, k=3, seed=17, parameters=dict(max_iter=5, batch_size=7, tol=.001, reassignment_ratio=0., max_no_improvement=None)),
            dict(features=data, k=3, seed=19, parameters=dict(max_iter=2, batch_size=7, compute_labels=False, max_no_improvement=None)),
            dict(features=[[0, 0]]*9, k=2, seed=19, parameters=dict(max_iter=1, batch_size=9, reassignment_ratio=11.)),
            dict(features=[[0, 0]]*9, k=1, seed=19, parameters=dict(max_iter=1, batch_size=99, compute_labels=False)),
            dict(features=[[0, 0]]*17, k=1, seed=31, parameters=dict(max_iter=3, batch_size=7, tol=1., max_no_improvement=0)),
            dict(features=[[(10, 70, 140, 220)[i % 4]+(i*7+j*3)%8 for j in range(243)] for i in range(4097)],
                k=4, seed=7, parameters=dict(max_iter=1, batch_size=1024, n_init=2, init_size=1537, max_no_improvement=None))]
        completed = subprocess.run([str(REFERENCE), '-c', SCRIPT], cwd=ROOT,
            input=json.dumps(cases), text=True, capture_output=True, timeout=90)
        if completed.returncode:
            raise AssertionError(completed.stderr)
        cls.cases, cls.reference = cases, json.loads(completed.stdout)

    def run_traced(self, index):
        item, expected = self.cases[index], self.reference[index]
        x = torch.tensor(item['features'], dtype=torch.uint16, device='cuda')
        original_schedule = native._random.MiniBatchRandomSchedule
        original_init = native._plusplus.initialize_kmeans_plusplus
        original_update = native._update.update_unit_weight_centres
        original_timer = native._control.MiniBatchControl.random_reassign_due
        original_convergence = native._intervals.IntervalMiniBatchControl.convergence
        observed = dict(calls=[],initial=[],timers=[],updates=[],convergence=[])
        class Schedule(original_schedule):
            def _record(self, name, value):
                result = super()._record(name, value)
                array = np.asarray(value)
                kind = 'randint' if name.endswith('randint') else ('uniform' if name.endswith('uniform') else 'choice')
                observed['calls'].append(dict(kind=kind,shape=list(array.shape),dtype=array.dtype.str,
                    sha=sha256(array.tobytes()).hexdigest(),state=self.identity()['state_sha256']))
                observed['schedule'] = self
                return result
        def initialize(*args, **kwargs):
            result = original_init(*args, **kwargs)
            observed['initial'].append(_tensor_sha(result.centres))
            return result
        def update(*args, **kwargs):
            result = original_update(*args, **kwargs)
            observed['updates'].append(dict(old_centres=_tensor_sha(args[2]), old_counts=_tensor_sha(args[3]), result=result))
            return result
        def timer(control, counts):
            before, digest = control.n_since_last_reassign, sha256(counts.tobytes()).hexdigest()
            due = original_timer(control, counts)
            observed['timers'].append([before, digest, due, control.n_since_last_reassign])
            return due
        def convergence(control, **kwargs):
            stop = original_convergence(control, **kwargs)
            observed['convergence'].append(dict(inertia=kwargs['batch_inertia'], difference=kwargs['centres_squared_diff'],
                stop=stop, ewa=control.ewa_inertia, minimum=control.ewa_inertia_min, counter=control.no_improvement))
            return stop
        error, result = None, None
        with patch.object(native._random, 'MiniBatchRandomSchedule', Schedule), \
                patch.object(native._plusplus, 'initialize_kmeans_plusplus', side_effect=initialize), \
                patch.object(native._update, 'update_unit_weight_centres', side_effect=update), \
                patch.object(native._control.MiniBatchControl, 'random_reassign_due', timer), \
                patch.object(native._intervals.IntervalMiniBatchControl, 'convergence', convergence), \
                warnings.catch_warnings(record=True) as notices:
            warnings.simplefilter('always')
            try:
                result = native.fit_kmeans_training_rows(x, item['k'], source_n_threads=expected['threads'],
                    seed=item['seed'], **item['parameters'])
            except Exception as exception:
                error = type(exception).__name__
        observed['warnings'] = [type(notice.message).__name__ for notice in notices]
        return result, error, observed

    def assert_contains(self, interval, value):
        if value is None:
            self.assertIsNone(interval)
        else:
            self.assertLessEqual(interval.lo, value)
            self.assertGreaterEqual(interval.hi, value)

    def check_test_actual_public_fit_matches_all_rng_draws_initializations_steps_and_final_outputs(self):
        for index, expected in enumerate(self.reference):
            with self.subTest(case=index):
                result, error, observed = self.run_traced(index)
                self.assertEqual(error, expected['error'])
                self.assertEqual(observed['warnings'], expected['warnings'])
                self.assertEqual(observed['calls'], expected['calls'])
                self.assertEqual(observed['schedule'].identity()['state_sha256'], expected['state'])
                self.assertEqual(observed['initial'], expected['initial'])
                self.assertEqual(observed['timers'], expected['timers'])
                updates = [dict(old_centres=entry['old_centres'],old_counts=entry['old_counts'],
                    centres=_tensor_sha(entry['result'].centres),counts=_tensor_sha(entry['result'].counts)) for entry in observed['updates']]
                self.assertEqual(updates, expected['updates'])
                self.assertEqual(len(observed['convergence']), len(expected['convergence']))
                for current, source in zip(observed['convergence'], expected['convergence']):
                    self.assertEqual(current['stop'], source['stop'])
                    self.assertEqual(current['counter'], source['counter'])
                    for name in ('inertia', 'ewa', 'minimum'):
                        self.assert_contains(current[name], source[name])
                    if current['difference'] is not None:
                        self.assert_contains(current['difference'], source['difference'])
                if error is None:
                    self.assertEqual(_tensor_sha(result.centres), expected['centres'])
                    self.assertEqual(_tensor_sha(result.counts), expected['counts'])
                    self.assertEqual(None if result.labels is None else result.labels.cpu().tolist(), expected['labels'])
                    self.assert_contains(result.inertia, expected['inertia'])
                    self.assert_contains(result.scaled_tolerance, expected['tol'])
                    self.assertEqual((result.n_steps, result.n_iter), (expected['steps'], expected['iterations']))
                    self.assertEqual((result.plan.batch_size, result.plan.init_size, result.plan.n_init),
                        (expected['batch'], expected['init_size'], expected['n_init']))
                    self.assertEqual(result.source_n_threads, expected['threads'])

    def check_test_zero_and_constant_tolerance_skip_centre_shift_calculation(self):
        for index in (0, 8):
            item, expected = self.cases[index], self.reference[index]
            x = torch.tensor(item['features'], dtype=torch.uint16, device='cuda')
            with patch.object(native._tolerance, 'centre_shift_interval', side_effect=AssertionError('shift at source tol0')):
                result = native.fit_kmeans_training_rows(x, item['k'], source_n_threads=expected['threads'],
                    seed=item['seed'], **item['parameters'])
            self.assertEqual(result.scaled_tolerance.lo, 0.)
            self.assertTrue(result.scaled_tolerance.is_point)

    def check_test_all_reduction_profile_matches_actual_public_fits_at_one_two_and_four_threads(self):
        cases = self.cases[:2]
        completed = subprocess.run([str(REFERENCE), '-c', THREAD_SCRIPT], cwd=ROOT,
            input=json.dumps(cases), text=True, capture_output=True, timeout=90)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        for case, references in zip(cases, json.loads(completed.stdout)):
            x = torch.tensor(case['features'], dtype=torch.uint16, device='cuda')
            with patch.object(os, 'cpu_count', side_effect=AssertionError('inferred CPU count')), \
                    patch.object(torch, 'get_num_threads', side_effect=AssertionError('inferred Torch threads')):
                result = native.fit_kmeans_training_rows(x, case['k'], source_n_threads=None,
                    seed=case['seed'], **case['parameters'])
            self.assertIsNone(result.source_n_threads)
            self.assertEqual(result.source_reduction_profile, 'all-valid-openmp-reductions-v1')
            for expected in references:
                self.assertEqual(_tensor_sha(result.centres), expected['centres'])
                self.assertEqual(_tensor_sha(result.counts), expected['counts'])
                self.assertEqual(result.labels.cpu().tolist(), expected['labels'])
                self.assert_contains(result.inertia, expected['inertia'])
                self.assertEqual((result.n_steps, result.n_iter), (expected['steps'], expected['iterations']))
        with self.assertRaises(TypeError):
            native.fit_kmeans_training_rows(x, case['k'])
        recorded = native.fit_kmeans_training_rows(x, case['k'], source_n_threads=2,
            seed=case['seed'], **case['parameters'])
        self.assertEqual(recorded.source_reduction_profile, 'recorded-openmp-threads-v1')

    def check_test_all_reduction_profile_withholds_fit_when_initialization_comparison_is_ambiguous(self):
        case = self.cases[0]
        x = torch.tensor(case['features'], dtype=torch.uint16, device='cuda')
        original = native._score.score_kmeans_training_rows
        def ambiguous(*args, **kwargs):
            result = original(*args, **kwargs)
            self.assertIsNone(result.source_n_threads)
            return replace(result, inertia=native._intervals.F64Interval(1., 2.))
        with patch.object(native._score, 'score_kmeans_training_rows', side_effect=ambiguous), \
                patch.object(native._update, 'update_unit_weight_centres', side_effect=AssertionError('update after ambiguity')):
            with self.assertRaises(native._intervals.KMeansNumericalAmbiguity):
                native.fit_kmeans_training_rows(x, case['k'], source_n_threads=None,
                    seed=case['seed'], **case['parameters'])

    def check_test_whole_job_caps_and_parameters_reject_before_rng_or_output_allocation(self):
        x = torch.zeros((17, 3), dtype=torch.uint16, device='cuda')
        for constant in ('MAX_TOTAL_WORK', 'MAX_CUDA_LIVE_BYTES', 'MAX_CPU_METADATA_BYTES', 'MAX_STEPS'):
            with patch.object(native, constant, 1):
                with patch.object(native._random, 'MiniBatchRandomSchedule', side_effect=AssertionError('RNG before cap')):
                    with patch.object(torch, 'empty', side_effect=AssertionError('allocation before cap')):
                        with self.assertRaises(native.KMeansFitError):
                            native.fit_kmeans_training_rows(x, 3, source_n_threads=1)
        for kwargs in (dict(n_init=True), dict(n_init='random'), dict(init_size=0), dict(tol=float('nan')),
                dict(batch_size=0), dict(seed=None), dict(max_iter=0), dict(max_no_improvement=True),
                dict(reassignment_ratio=-1), dict(compute_labels=1), dict(source_n_threads=0)):
            parameters = dict(source_n_threads=1)
            parameters.update(kwargs)
            with self.subTest(parameters=kwargs), self.assertRaises(native.KMeansFitError):
                native.fit_kmeans_training_rows(x, 3, **parameters)
        bad = torch.tensor([[256]], dtype=torch.uint16, device='cuda')
        with patch.object(native._random, 'MiniBatchRandomSchedule', side_effect=AssertionError('RNG before values')):
            with self.assertRaisesRegex(native.KMeansFitError, 'rounded bytes'):
                native.fit_kmeans_training_rows(bad, 1, source_n_threads=1)

    def check_test_custom_stream_input_preservation_actual_live_peak_and_cancel_withholding(self):
        item = self.cases[9]
        x = torch.tensor(item['features'], dtype=torch.uint16, device='cuda')
        parameters = dict(item['parameters'], source_n_threads=self.reference[9]['threads'], seed=item['seed'])
        plan = native.preflight_kmeans_fit(x, item['k'], **parameters)
        native.fit_kmeans_training_rows(x, item['k'], **parameters)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        version = x._version
        total = 0
        def count():
            nonlocal total
            total += 1
            return False
        with torch.cuda.stream(stream):
            result = native.fit_kmeans_training_rows(x, item['k'], **parameters, cancel=count)
        stream.synchronize()
        self.assertEqual(_tensor_sha(result.centres), self.reference[9]['centres'])
        self.assertEqual(x._version, version)
        self.assertLessEqual(torch.cuda.max_memory_allocated()-baseline, plan.peak_live_bytes-x.numel()*2)
        for target in (1, total//2, total):
            calls = 0
            def cancel():
                nonlocal calls
                calls += 1
                return calls == target
            with self.assertRaises(native.KMeansFitCancelled):
                native.fit_kmeans_training_rows(x, item['k'], **parameters, cancel=cancel)
            self.assertEqual(calls, target)
            self.assertEqual(x._version, version)

    def check_test_input_mutation_during_rng_and_callbacks_stops_before_following_stage(self):
        item = self.cases[0]
        parameters = dict(item['parameters'], source_n_threads=self.reference[0]['threads'], seed=item['seed'])
        for target in (1, 5, 25):
            x = torch.tensor(item['features'], dtype=torch.uint16, device='cuda')
            calls = 0
            def mutate():
                nonlocal calls
                calls += 1
                if calls == target:
                    x.resize_(0)
                return False
            with self.assertRaisesRegex(native.KMeansFitError, 'changed'):
                native.fit_kmeans_training_rows(x, item['k'], **parameters, cancel=mutate)
            self.assertEqual(calls, target)
        x = torch.tensor(item['features'], dtype=torch.uint16, device='cuda')
        def mutated_draw(schedule, n, size):
            x.resize_(0)
            return np.zeros(size, dtype=np.int32)
        with patch.object(native._random.MiniBatchRandomSchedule, 'validation_indices', mutated_draw):
            with self.assertRaisesRegex(native.KMeansFitError, 'changed'):
                native.fit_kmeans_training_rows(x, item['k'], **parameters)

    def check_test_loaded_source_binds_complete_job_limits_and_pinned_source(self):
        from seedvision.benchmarking.native_recipe import _module_record
        self.assertTrue(_module_record(native)['loaded_code'])
        for name, value in (('MAX_TOTAL_WORK', 1), ('SKLEARN_KMEANS_SHA256', 'changed')):
            with patch.object(native, name, value):
                with self.assertRaises(ValueError):
                    _module_record(native)

    def check_test_metadata_only_preflight_matches_tensor_plan_without_allocation_or_rng(self):
        for index, case in enumerate(self.cases):
            x = torch.tensor(case['features'], dtype=torch.uint16, device='cuda')
            parameters = dict(case['parameters'], source_n_threads=self.reference[index]['threads'], seed=case['seed'])
            expected = native.preflight_kmeans_fit(x, case['k'], **parameters)
            with patch.object(torch, 'empty', side_effect=AssertionError('allocated during shape preflight')), \
                    patch.object(native._random, 'MiniBatchRandomSchedule', side_effect=AssertionError('RNG during preflight')):
                self.assertEqual(native.preflight_kmeans_fit_shape(*map(int, x.shape), case['k'], **parameters), expected)
        for rows, width in ((True, 3), (3, True), (0, 3), (3, 0), (3., 3), (1_000_001, 1), (1, 4097)):
            with self.assertRaises(native.KMeansFitError):
                native.preflight_kmeans_fit_shape(rows, width, 1, source_n_threads=1)

    def test_workflow_lifecycle(self):
        """4 related KMeansFitTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_actual_public_fit_matches_all_rng_draws_initializations_steps_and_final_outputs)
        self.scenario(self.check_test_zero_and_constant_tolerance_skip_centre_shift_calculation)
        self.scenario(self.check_test_all_reduction_profile_matches_actual_public_fits_at_one_two_and_four_threads)
        self.scenario(self.check_test_metadata_only_preflight_matches_tensor_plan_without_allocation_or_rng)

    def test_validation_and_recovery(self):
        """5 related KMeansFitTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_all_reduction_profile_withholds_fit_when_initialization_comparison_is_ambiguous)
        self.scenario(self.check_test_whole_job_caps_and_parameters_reject_before_rng_or_output_allocation)
        self.scenario(self.check_test_custom_stream_input_preservation_actual_live_peak_and_cancel_withholding)
        self.scenario(self.check_test_input_mutation_during_rng_and_callbacks_stops_before_following_stage)
        self.scenario(self.check_test_loaded_source_binds_complete_job_limits_and_pinned_source)
