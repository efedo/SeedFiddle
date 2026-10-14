"""Compact control decisions checked against actual pinned sklearn methods."""
from dataclasses import asdict, replace
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path
import struct
import subprocess
import unittest
from unittest.mock import patch

import numpy as np
import torch

from seedvision.cuda import plantcv_kmeans_control as native
from seedvision.cuda import plantcv_kmeans_minibatch as update
from seedvision.cuda.plantcv_kmeans_random import MiniBatchRandomSchedule

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT/'artifacts/plantcv-reference-env/Scripts/python.exe'
SCRIPT = r'''
import hashlib,importlib.metadata,json,struct,sys
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
def bits(value):
    return None if value is None else struct.pack('<d',value).hex()
class TracedRandomState(np.random.RandomState):
    def __init__(self,seed):
        super().__init__(seed)
        self.rows=None
    def choice(self,a,size=None,replace=True,p=None):
        assert not replace and p is None
        value=super().choice(a,size=size,replace=replace,p=p)
        self.rows=value.tolist()
        return value
request=json.load(sys.stdin)
result=dict(timers=[],convergence=[],steps=[],orders=[])
for item in request['timers']:
    model=MiniBatchKMeans(n_clusters=item['k'])
    model._batch_size=item['batch']
    model._n_since_last_reassign=item['since']
    trace=[]
    for counts in item['counts']:
        model._counts=np.asarray(counts,dtype=np.float64)
        due=model._random_reassign()
        trace.append([bool(due),model._n_since_last_reassign])
    result['timers'].append(trace)
for item in request['convergence']:
    model=MiniBatchKMeans(n_clusters=2,max_no_improvement=item['limit'])
    model._batch_size=item['batch']
    model._tol=item['tol']
    model._ewa_inertia=item.get('ewa')
    model._ewa_inertia_min=item.get('minimum')
    model._no_improvement=item.get('counter',0)
    trace=[]
    for step,(inertia,diff) in enumerate(item['values']):
        stop=model._mini_batch_convergence(step,len(item['values']),item['samples'],diff,inertia)
        trace.append([bool(stop),bits(model._ewa_inertia),bits(model._ewa_inertia_min),model._no_improvement])
    result['convergence'].append(trace)
for item in request['steps']:
    x=np.ascontiguousarray(item['features'],dtype=np.float64)
    centres=np.ascontiguousarray(item['centres'],dtype=np.float64)
    weights=np.ones(len(x),dtype=np.float64)
    labels,_=source._labels_inertia(x,weights,centres,n_threads=1)
    disabled=np.empty_like(centres)
    after=np.asarray(item['counts'],dtype=np.float64)
    source._mini_batch_step(x,weights,centres,disabled,after,np.random.RandomState(item['seed']),random_reassign=False,n_threads=1)
    enabled=np.empty_like(centres)
    counts=np.asarray(item['counts'],dtype=np.float64)
    rng=TracedRandomState(item['seed'])
    error=None
    try:
        source._mini_batch_step(x,weights,centres,enabled,counts,rng,random_reassign=True,reassignment_ratio=item['ratio'],n_threads=1)
    except ValueError as exception:
        error=type(exception).__name__
    result['steps'].append(dict(labels=labels.tolist(),post_counts=after.tolist(),
        disabled_sha=digest(disabled),centres_sha=digest(enabled),counts_sha=digest(counts),
        rows=rng.rows,state_sha=state_sha(rng),error=error))
for values in request['orders']:
    result['orders'].append(np.argsort(np.asarray(values,dtype=np.float64)).tolist())
print(json.dumps(result))
'''


@lru_cache(maxsize=1)
def _fixtures():
    timers = [dict(k=2, batch=3, since=0, counts=[[1, 1]]*7+[[0, 1], [1, 1]]),
        dict(k=3, batch=7, since=23, counts=[[1, 1, 1], [1, 0, 1], [1, 1, 1]]),
        dict(k=256, batch=4096, since=0, counts=[[1]*256])]
    convergence = []
    for limit in (None, 0, 1, 10):
        convergence.append(dict(limit=limit, batch=7, samples=101, tol=0.,
            values=[[1e15, 0.]]+[[147., 0.]]*13+[[100., 0.], [70., 0.], [1000., 0.]]))
    convergence.extend([
        dict(limit=10, batch=7, samples=7, tol=0., values=[[1000., 0.], [35., 0.], [28., 0.], [28., 0.]]),
        dict(limit=10, batch=7, samples=101, tol=.25, ewa=8., minimum=7., counter=9,
            values=[[100., 0.], [35., .25], [35., .25000000000000006], [35., 0.]]),
        dict(limit=None, batch=3, samples=17, tol=0., ewa=10.125, minimum=9.75, counter=3,
            values=[[300., 0.], [31.25, 0.], [27.125, 0.], [28.75, 0.]])])
    steps = [
        dict(features=[[0., 0.]]*9, centres=[[0., 0.], [255., 255.], [120., 120.]],
            counts=[91, 1, 0], ratio=.01, seed=19),
        dict(features=[[0.]]*7, centres=[[0.], [255.]], counts=[0, 0], ratio=.01, seed=7),
        dict(features=[[0.]]*7, centres=[[0.], [255.]], counts=[0, 100], ratio=.01, seed=31),
        dict(features=[[0.]]*7, centres=[[0.]]*256, counts=[0]*255+[1_000_000], ratio=.01, seed=19),
        dict(features=[[0.]]*1, centres=[[0.]]*17, counts=[0]*16+[100], ratio=.01, seed=19),
        dict(features=[[17., 42.]]*9, centres=[[17., 42.], [200., 200.]], counts=[1, 1], ratio=11., seed=19),
    ]
    rng = np.random.default_rng(72)
    orders = []
    for k in (1, 2, 4, 8, 16, 31, 64, 128, 256):
        orders.extend([[0]*k, [1]*k, [i % 3 for i in range(k)],
            [int(i) for i in rng.integers(0, 7, k)]])
    request = dict(timers=timers, convergence=convergence, steps=steps, orders=orders)
    completed = subprocess.run([str(REFERENCE), '-c', SCRIPT], cwd=ROOT,
        input=json.dumps(request), text=True, capture_output=True, timeout=90)
    if completed.returncode:
        raise AssertionError(completed.stderr)
    return request, json.loads(completed.stdout)


def _bits(value):
    return None if value is None else struct.pack('<d', value).hex()


class KMeansControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.request, cls.reference = _fixtures()

    def test_preupdate_reassignment_timer_matches_actual_source_at_zero_and_threshold(self):
        for item, expected in zip(self.request['timers'], self.reference['timers']):
            control = native.MiniBatchControl(item['batch'], item['k'], n_since_last_reassign=item['since'])
            trace = []
            for counts in item['counts']:
                trace.append([control.random_reassign_due(np.asarray(counts, dtype=np.float64)),
                    control.n_since_last_reassign])
            self.assertEqual(trace, expected)

    def test_every_convergence_branch_matches_source_binary64_state_and_return(self):
        for item, expected in zip(self.request['convergence'], self.reference['convergence']):
            control = native.MiniBatchControl(item['batch'], 2, tol=item['tol'],
                max_no_improvement=item['limit'], ewa_inertia=item.get('ewa'),
                ewa_inertia_min=item.get('minimum'), no_improvement=item.get('counter', 0))
            for step, ((inertia, diff), state) in enumerate(zip(item['values'], expected)):
                with self.subTest(limit=item['limit'], tol=item['tol'], step=step):
                    stop = control.convergence(step=step, n_steps=len(item['values']),
                        n_samples=item['samples'], centres_squared_diff=diff, batch_inertia=inertia)
                    self.assertEqual([stop, _bits(control.ewa_inertia), _bits(control.ewa_inertia_min),
                        control.no_improvement], state)

    def test_default_numpy_tied_count_order_matches_pinned_source(self):
        for values, expected in zip(self.request['orders'], self.reference['orders']):
            with self.subTest(k=len(values), pattern=values[:8]):
                self.assertEqual(np.argsort(np.asarray(values, dtype=np.float64)).tolist(), expected)

    def test_strict_threshold_empty_selection_and_rng_state_match_actual_source_step(self):
        for item, expected in zip(self.request['steps'], self.reference['steps']):
            counts = np.asarray(expected['post_counts'], dtype=np.float64)
            schedule = MiniBatchRandomSchedule(item['seed'])
            plan = native.select_low_count_reassignments(counts, batch_rows=len(item['features']),
                reassignment_ratio=item['ratio'], schedule=schedule)
            self.assertEqual(None if plan.new_rows is None else list(plan.new_rows), expected['rows'])
            self.assertEqual(schedule.identity()['state_sha256'], expected['state_sha'])
            self.assertEqual(plan.counts_sha256, sha256(counts.tobytes()).hexdigest())
            if item is self.request['steps'][0]:
                self.assertEqual(plan.to_reassign, (False, False, True))
            if expected['error'] is None:
                reset = native.low_count_reset_value(counts, plan)
                counts[np.asarray(plan.to_reassign)] = reset
                self.assertEqual(sha256(counts.tobytes()).hexdigest(), expected['counts_sha'])
            else:
                self.assertTrue(all(plan.to_reassign))
                with self.assertRaises(ValueError):
                    native.low_count_reset_value(counts, plan)

    def test_empty_selection_still_evaluates_minimum_without_random_draw(self):
        counts = np.array([10., 10.])
        schedule = MiniBatchRandomSchedule(1)
        identity = schedule.identity()
        plan = native.select_low_count_reassignments(counts, batch_rows=7,
            reassignment_ratio=.01, schedule=schedule)
        self.assertEqual(plan.to_reassign, (False, False))
        self.assertIsNone(plan.new_rows)
        self.assertEqual(schedule.identity(), identity)
        with patch.object(native.np, 'min', wraps=np.min) as minimum:
            self.assertEqual(native.low_count_reset_value(counts, plan), 10.)
            self.assertEqual(minimum.call_count, 1)

    def test_invalid_control_state_and_calls_fail_before_state_changes(self):
        for kwargs in (dict(batch_size=True), dict(batch_size=4097), dict(n_clusters=257),
                dict(tol=float('nan')), dict(max_no_improvement=True),
                dict(n_since_last_reassign=-1), dict(ewa_inertia=float('inf'))):
            parameters = dict(batch_size=7, n_clusters=2)
            parameters.update(kwargs)
            with self.subTest(parameters=parameters), self.assertRaises(native.KMeansControlError):
                native.MiniBatchControl(**parameters)
        control = native.MiniBatchControl(7, 2)
        for kwargs in (dict(step=True), dict(step=2), dict(n_samples=6),
                dict(centres_squared_diff=float('nan')), dict(batch_inertia=-1),
                dict(batch_inertia=10**1000)):
            arguments = dict(step=1, n_steps=2, n_samples=17, centres_squared_diff=0., batch_inertia=1.)
            arguments.update(kwargs)
            before = asdict(control)
            with self.subTest(parameters=kwargs), self.assertRaises(native.KMeansControlError):
                control.convergence(**arguments)
            self.assertEqual(asdict(control), before)
        control.no_improvement = -1
        before = asdict(control)
        with self.assertRaises(native.KMeansControlError):
            control.random_reassign_due(np.array([0., 0.]))
        self.assertEqual(asdict(control), before)

    def test_bad_snapshots_and_reassignment_parameters_do_not_advance_rng(self):
        schedule = MiniBatchRandomSchedule(1)
        identity = schedule.identity()
        snapshots = [np.array([], dtype=np.float64), np.zeros(257), np.array([1.5]),
            np.array([-1.]), np.array([float('nan')]), np.array([2.**53+2]),
            np.zeros(2, dtype=np.float32), np.zeros(4)[::2], [0., 1.]]
        for counts in snapshots:
            with self.subTest(counts=repr(counts)), self.assertRaises(native.KMeansControlError):
                native.select_low_count_reassignments(counts, batch_rows=7,
                    reassignment_ratio=.01, schedule=schedule)
            self.assertEqual(schedule.identity(), identity)
        for batch, ratio in ((0, .01), (4097, .01), (True, .01), (7, 0),
                (7, float('inf')), (7, -.01), (7, True)):
            with self.subTest(batch=batch, ratio=ratio), self.assertRaises(native.KMeansControlError):
                native.select_low_count_reassignments(np.array([0., 10.]),
                    batch_rows=batch, reassignment_ratio=ratio, schedule=schedule)
            self.assertEqual(schedule.identity(), identity)

    def test_count_snapshot_binding_rejects_changed_metadata_and_callback_mutation(self):
        counts = np.array([0., 10.])
        schedule = MiniBatchRandomSchedule(1)
        plan = native.select_low_count_reassignments(counts, batch_rows=7,
            reassignment_ratio=.01, schedule=schedule)
        for changed in (replace(plan, counts_sha256='changed'),
                replace(plan, to_reassign=(0, 1)), replace(plan, to_reassign=(False,))):
            with self.assertRaises(native.KMeansControlError):
                native.low_count_reset_value(counts, changed)
        counts[1] += 1
        with self.assertRaises(native.KMeansControlError):
            native.low_count_reset_value(counts, plan)
        def mutated_draw(batch, count):
            counts[1] += 1
            return np.arange(count)
        with patch.object(schedule, 'reassignment_indices', side_effect=mutated_draw):
            with self.assertRaisesRegex(native.KMeansControlError, 'changed'):
                native.select_low_count_reassignments(counts, batch_rows=7,
                    reassignment_ratio=.01, schedule=schedule)

    def test_loaded_source_binds_control_limits_and_pinned_source_identity(self):
        from seedvision.benchmarking.native_recipe import _module_record
        self.assertTrue(_module_record(native)['loaded_code'])
        for name, value in (('MAX_CLUSTERS', 1), ('SKLEARN_KMEANS_SHA256', 'changed')):
            with self.subTest(constant=name), patch.object(native, name, value):
                with self.assertRaises(ValueError):
                    _module_record(native)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class KMeansReassignmentCudaTests(unittest.TestCase):
    def test_cuda_update_then_reassignment_matches_actual_source_including_failure_order(self):
        request, reference = _fixtures()
        for item, expected in zip(request['steps'], reference['steps']):
            with self.subTest(k=len(item['centres']), rows=len(item['features']), ratio=item['ratio']):
                features = torch.tensor(item['features'], dtype=torch.float64, device='cuda')
                labels = torch.tensor(expected['labels'], dtype=torch.uint8, device='cuda')
                centres = torch.tensor(item['centres'], dtype=torch.float64, device='cuda')
                counts = torch.tensor(item['counts'], dtype=torch.float64, device='cuda')
                result = update.update_unit_weight_centres(features, labels, centres, counts)
                self.assertEqual(sha256(result.centres.cpu().numpy().tobytes()).hexdigest(), expected['disabled_sha'])
                compact_counts = result.counts.cpu().numpy()
                schedule = MiniBatchRandomSchedule(item['seed'])
                plan = native.select_low_count_reassignments(compact_counts, batch_rows=len(features),
                    reassignment_ratio=item['ratio'], schedule=schedule)
                mask = torch.tensor(plan.to_reassign, dtype=torch.bool, device='cuda')
                if plan.n_reassigns:
                    row_ids = torch.tensor(plan.new_rows, dtype=torch.int64, device='cuda')
                    result.centres[mask] = features[row_ids]
                if expected['error'] is None:
                    result.counts[mask] = native.low_count_reset_value(compact_counts, plan)
                else:
                    # The actual source writes centres and consumes RNG before
                    # evaluating min(empty); preserve that observable order.
                    with self.assertRaises(ValueError):
                        native.low_count_reset_value(compact_counts, plan)
                self.assertEqual(sha256(result.centres.cpu().numpy().tobytes()).hexdigest(), expected['centres_sha'])
                self.assertEqual(sha256(result.counts.cpu().numpy().tobytes()).hexdigest(), expected['counts_sha'])
                self.assertEqual(schedule.identity()['state_sha256'], expected['state_sha'])
