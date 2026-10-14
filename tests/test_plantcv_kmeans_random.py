"""Exact random-call outputs/state from actual pinned MiniBatchKMeans.fit."""
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import struct
import unittest
from unittest.mock import patch

import numpy as np

from seedvision.cuda import plantcv_kmeans_random as native

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
def state_sha(rng):
    algorithm,keys,position,has_gauss,cached=rng.get_state()
    h=hashlib.sha256(algorithm.encode('ascii')+b'\0')
    h.update(keys.astype('<u4',copy=False).tobytes())
    h.update(struct.pack('<IId',int(position),int(has_gauss),float(cached)))
    return h.hexdigest()
class TracedRandomState(np.random.RandomState):
    def __init__(self,seed):
        super().__init__(seed)
        self.calls=[]
    def record(self,kind,value,**details):
        a=np.asarray(value)
        self.calls.append(dict(kind=kind,shape=list(a.shape),dtype=a.dtype.str,
            sha256=hashlib.sha256(a.tobytes()).hexdigest(),state_sha256=state_sha(self),**details))
        return value
    def randint(self,low,high=None,size=None,dtype=int):
        # Preserve default legacy C-long behavior; do not forward Python's
        # NumPy2 default int64 as an explicit dtype when source omitted it.
        value=super().randint(low,high,size)
        return self.record('randint',value,n=int(high if high is not None else low),size=int(size))
    def choice(self,a,size=None,replace=True,p=None):
        if p is not None:
            assert np.all(p==p[0])
        value=super().choice(a,size=size,replace=replace,p=p)
        return self.record('choice',value,n=int(a),size=None if size is None else int(size),
            replace=bool(replace),weighted=p is not None)
    def uniform(self,low=0.,high=1.,size=None):
        assert low==0. and high==1.
        value=super().uniform(low,high,size=size)
        return self.record('uniform',value,size=int(size))
results=[]
for item in json.load(sys.stdin):
    rng=TracedRandomState(item['seed'])
    x=np.asarray(item['features'],dtype=np.uint16)
    model=MiniBatchKMeans(n_clusters=item['k'],n_init=2,batch_size=7,max_iter=2,
        random_state=rng,max_no_improvement=None,reassignment_ratio=.01).fit(x)
    results.append(dict(calls=rng.calls,state_sha256=state_sha(rng),steps=int(model.n_steps_)))
print(json.dumps(results))
'''


def _reference(cases):
    result = subprocess.run([str(REFERENCE), '-c', SCRIPT], cwd=ROOT,
        input=json.dumps(cases), text=True, capture_output=True, timeout=90)
    if result.returncode:
        raise AssertionError(result.stderr)
    return json.loads(result.stdout)


class KMeansRandomTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data = ((np.arange(79)[:, None]*17+np.arange(5)*31)%256).tolist()
        cls.cases = [dict(seed=19, features=data, k=3),
            dict(seed=2**32-1, features=[[17, 42]]*35, k=5)]
        cls.reference = _reference(cls.cases)

    def test_replay_actual_public_fit_call_sequence_matches_outputs_and_every_state(self):
        observed = set()
        for case, expected in zip(self.cases, self.reference):
            schedule = native.MiniBatchRandomSchedule(case['seed'])
            for index, call in enumerate(expected['calls']):
                with self.subTest(seed=case['seed'], call=index, kind=call['kind']):
                    if call['kind'] == 'randint':
                        method = schedule.validation_indices if index == 0 else schedule.init_subset
                        result = method(call['n'], call['size'])
                        observed.add('validation' if index == 0 else 'subset')
                    elif call['kind'] == 'uniform':
                        result = schedule.local_trials(call['size'])
                        observed.add('uniform')
                    elif not call['replace']:
                        self.assertFalse(call['weighted'])
                        result = schedule.reassignment_indices(call['n'], call['size'])
                        observed.add('reassignment')
                    elif call['size'] is None:
                        self.assertTrue(call['weighted'])
                        result = schedule.first_center(call['n'])
                        observed.add('first')
                    else:
                        self.assertTrue(call['weighted'])
                        result = schedule.minibatch_indices(call['n'], call['size'])
                        observed.add('batch')
                    result = np.asarray(result)
                    self.assertEqual(list(result.shape), call['shape'])
                    self.assertEqual(result.dtype.str, call['dtype'])
                    self.assertEqual(sha256(result.tobytes()).hexdigest(), call['sha256'])
                    identity = schedule.identity()
                    self.assertEqual(identity['state_sha256'], call['state_sha256'])
                    self.assertEqual(identity['calls'], index+1)
            self.assertEqual(schedule.identity()['state_sha256'], expected['state_sha256'])
        self.assertEqual(observed, {'validation', 'subset', 'uniform', 'first', 'batch', 'reassignment'})

    def test_full_init_size_skips_draw_but_validation_always_consumes_vector(self):
        schedule = native.MiniBatchRandomSchedule(31)
        before = schedule.identity()
        self.assertIsNone(schedule.init_subset(17, 17))
        self.assertEqual(schedule.identity(), before)
        expected = np.random.RandomState(31).randint(0, 17, 17)
        np.testing.assert_array_equal(schedule.validation_indices(17, 17), expected)
        self.assertEqual(schedule.identity()['calls'], 1)
        self.assertNotEqual(schedule.identity()['state_sha256'], before['state_sha256'])

    def test_weighted_choice_is_distinct_from_uniform_integer_draws(self):
        schedule = native.MiniBatchRandomSchedule(19)
        values = schedule.minibatch_indices(101, 31)
        self.assertFalse(np.array_equal(values, np.random.RandomState(19).randint(101, size=31)))
        same = native.MiniBatchRandomSchedule(19)
        np.testing.assert_array_equal(same.minibatch_indices(101, 31), values)
        self.assertEqual(same.identity(), schedule.identity())

    def test_maximum_weighted_probability_and_selection_metadata_are_bounded(self):
        schedule = native.MiniBatchRandomSchedule(3)
        result = schedule.minibatch_indices(native.MAX_SAMPLES, native.MAX_BATCH_SIZE)
        self.assertEqual(result.shape, (native.MAX_BATCH_SIZE,))
        self.assertGreaterEqual(schedule.peak_cpu_bytes,
            2*native.MAX_SAMPLES*8+native.MAX_BATCH_SIZE*8)
        self.assertLessEqual(schedule.peak_cpu_bytes, native.MAX_CPU_BYTES)
        before = schedule.identity()
        with patch.object(native, 'MAX_CPU_BYTES', 1):
            for draw in (lambda: schedule.validation_indices(17, 7),
                         lambda: schedule.init_subset(17, 7),
                         lambda: schedule.first_center(17),
                         lambda: schedule.local_trials(3),
                         lambda: schedule.minibatch_indices(17, 7),
                         lambda: schedule.reassignment_indices(17, 7)):
                with self.assertRaisesRegex(native.KMeansRandomError, 'CPU byte cap'):
                    draw()
                self.assertEqual(schedule.identity(), before)

    def test_invalid_seed_sizes_and_domains_fail_without_advancing_state(self):
        for seed in (None, True, -1, 2**32, 1.5, '1'):
            with self.subTest(seed=seed), self.assertRaises(native.KMeansRandomError):
                native.MiniBatchRandomSchedule(seed)
        schedule = native.MiniBatchRandomSchedule(1)
        before = schedule.identity()
        calls = [lambda: schedule.validation_indices(0, 1),
            lambda: schedule.validation_indices(17, 18), lambda: schedule.init_subset(17, True),
            lambda: schedule.first_center(native.MAX_SAMPLES+1),
            lambda: schedule.local_trials(0), lambda: schedule.local_trials(65),
            lambda: schedule.minibatch_indices(17, 18),
            lambda: schedule.minibatch_indices(10000, 4097),
            lambda: schedule.reassignment_indices(4097, 1),
            lambda: schedule.reassignment_indices(17, 18)]
        for index, draw in enumerate(calls):
            with self.subTest(case=index), self.assertRaises(native.KMeansRandomError):
                draw()
            self.assertEqual(schedule.identity(), before)

    def test_recorded_draw_digest_changes_with_method_and_caller_output_is_independent(self):
        validation = native.MiniBatchRandomSchedule(7)
        subset = native.MiniBatchRandomSchedule(7)
        rows = validation.validation_indices(17, 7)
        np.testing.assert_array_equal(rows, subset.init_subset(17, 7))
        self.assertEqual(validation.identity()['state_sha256'], subset.identity()['state_sha256'])
        self.assertNotEqual(validation.identity()['draw_sha256'], subset.identity()['draw_sha256'])
        identity = validation.identity()
        rows.fill(0)
        self.assertEqual(validation.identity(), identity)

    def test_loaded_source_binds_random_limits_and_pinned_source_identity(self):
        from seedvision.benchmarking.native_recipe import _module_record
        self.assertTrue(_module_record(native)['loaded_code'])
        for name, value in (('MAX_BATCH_SIZE', 1), ('SKLEARN_KMEANS_SHA256', 'changed')):
            with self.subTest(constant=name), patch.object(native, name, value):
                with self.assertRaises(ValueError):
                    _module_record(native)
