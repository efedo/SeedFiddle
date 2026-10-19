"""Real decoded directories against the pinned public binary trainer."""

from tests.functional_units import functional_units
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch

from seedvision.benchmarking import naive_bayes_training_directory as native
from seedvision.cuda import plantcv_naive_bayes_sampling as sampling
from seedvision.cuda.plantcv_naive_bayes import parse_plantcv_pdf, naive_bayes_masks
from tests.test_plantcv_naive_bayes_sampling import coordinate_pair, pair, state_record

REFERENCE = Path(__file__).resolve().parents[1]/'artifacts/plantcv-reference-env/Scripts/python.exe'
SCRIPT = r'''
from contextlib import redirect_stdout
from hashlib import sha256
import importlib,io,json,struct,sys,tempfile
from importlib.metadata import version
from pathlib import Path
from unittest.mock import patch
import numpy as np
m=importlib.import_module('plantcv.learn.naive_bayes')
assert {n:version(n) for n in ('plantcv','numpy','scipy','opencv-python')}=={
    'plantcv':'4.11.3','numpy':'2.2.6','scipy':'1.15.3','opencv-python':'4.11.0.86'}
assert sha256(Path(m.__file__).read_bytes()).hexdigest()=='f59c95c493d189b6110d47a60a97c86e306ab87583a2f128eff8c389aa6b56eb'
def state():
    a,k,p,g,c=np.random.get_state()
    return dict(algorithm=a,keys=k.astype('<u4').tobytes().hex(),position=p,has_gauss=g,cached_gaussian=struct.pack('<d',c).hex())
case=json.load(sys.stdin)
np.random.seed(case['seed'])
if case.get('cached_gaussian'): np.random.normal()
before=state(); captures=[]; steps=[]
walk=[(str(Path(case['images'])/d),dirs,files) for d,dirs,files in case['walk']]
actual_randint=np.random.randint
def draw(*args,**kwargs):
    value=actual_randint(*args,**kwargs); steps.append(state()); return value
actual_kde=m.stats.gaussian_kde
def kde(values):
    captures.append(values.tolist()); return actual_kde(values)
answer=dict(before=before)
with tempfile.TemporaryDirectory() as folder:
    output=Path(folder)/'model.tsv'
    try:
        with patch.object(m.os,'walk',side_effect=lambda *a,**k:iter(walk)),patch.object(m.stats,'gaussian_kde',side_effect=kde),patch.object(np.random,'randint',side_effect=draw),redirect_stdout(io.StringIO()):
            m.naive_bayes(case['images'],case['masks'],str(output))
        answer['tsv']=output.read_text()
    except Exception as error:
        answer['error']=type(error).__name__
answer.update(after=state(),steps=steps,columns=captures)
print(json.dumps(answer))
'''


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
@functional_units
class BayesDirectoryTrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.images, self.masks = self.root/'images', self.root/'masks'
        self.images.mkdir()
        self.masks.mkdir()
        self.walk = [('.', [], ['first.png'])]
        self.put('first.png', *pair())

    def put(self, relative, image, mask, *, encoded='.png'):
        destination = self.images/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        for path, value in ((destination, image), (self.masks/destination.name, mask)):
            ok, data = cv2.imencode(encoded, value)
            self.assertTrue(ok)
            path.write_bytes(data.tobytes())

    def oracle(self, *, seed=17, cached_gaussian=False):
        result = subprocess.run([str(REFERENCE), '-c', SCRIPT], input=json.dumps(dict(
            images=str(self.images), masks=str(self.masks), walk=self.walk,
            seed=seed, cached_gaussian=cached_gaussian)), text=True, capture_output=True, timeout=120)
        if result.returncode:
            raise AssertionError(result.stderr)
        return json.loads(result.stdout)

    def run_native(self, *, rng=None, **kwargs):
        walk = [(str(self.images/directory), dirs, files) for directory, dirs, files in self.walk]
        with patch.object(native.os, 'walk', side_effect=lambda *a, **k:iter(walk)):
            return native.train_bayes_directories(self.images, self.masks,
                rng=np.random.RandomState(17) if rng is None else rng, **kwargs)

    def assert_oracle(self, expected, result):
        self.assertEqual(state_record(sampling._state(self.rng)), expected['after'])
        for category, values in enumerate(result.samples):
            self.assertEqual(values.device.type, 'cuda')
            for channel in range(3):
                self.assertEqual(values[:,channel].cpu().tolist(), expected['columns'][channel*2+category])
        source = parse_plantcv_pdf(expected['tsv'])
        self.assertEqual(source.class_names, result.model.class_names)
        for category, channels in enumerate(result.fit.channels):
            for channel, fit in enumerate(channels):
                np.testing.assert_allclose(result.model.probabilities[category][channel],
                    source.probabilities[category][channel], rtol=0, atol=fit.pdf_absolute_budget)










    def check_test_caps_precede_decode_and_upload_and_invalid_arguments_consume_no_rng(self):
        for name in ('MAX_DIRECTORY_ENTRIES','MAX_OCCURRENCES','MAX_FILE_BYTES','MAX_SOURCE_BYTES','MAX_DECODED_BYTES','MAX_MANIFEST_BYTES'):
            rng = np.random.RandomState(17)
            before = sampling._state(rng)
            with (self.subTest(cap=name), patch.object(native, name, 0),
                    patch.object(native, '_calculation_records', return_value={})):
                with patch.object(native, '_decode', side_effect=AssertionError('decoded before bound')):
                    with self.assertRaises(native.BayesDirectoryError):
                        self.run_native(rng=rng)
            self.assertEqual(before, sampling._state(rng))
        for kwargs in (dict(device='cpu'), dict(rng=np.random.default_rng()), dict(cancel=False)):
            with self.subTest(kwargs=kwargs), self.assertRaises(native.BayesDirectoryError):
                self.run_native(**kwargs)


    def check_test_retained_samples_rng_and_last_pdf_mutations_even_raised_callbacks(self):
        self.put('second.png', *pair(seed=91))
        self.walk[0][2].append('second.png')
        for kind in ('retained','rng_raise','pdf_storage'):
            calls = 0
            original = sampling.sample_naive_bayes_image
            captured = []
            fitted = []
            fit = native._fit.fit_naive_bayes_hsv
            rng = np.random.RandomState(17)
            def sample(*args, **kwargs):
                result = original(*args, **kwargs)
                captured.append(result)
                return result
            def train(*args, **kwargs):
                result = fit(*args, **kwargs)
                fitted.append(result)
                return result
            changed = False
            def mutate():
                nonlocal changed
                if not changed and captured and kind != 'pdf_storage':
                    changed = True
                    if kind == 'retained':
                        captured[0].samples[0].add_(1)
                    else:
                        rng.randint(0,100,3)
                        raise RuntimeError('changed RNG before raising')
                if not changed and fitted and kind == 'pdf_storage':
                    changed = True
                    fitted[0].probabilities.data = fitted[0].probabilities.clone()
                return False
            with (self.subTest(kind=kind),
                    patch.object(sampling, 'sample_naive_bayes_image', side_effect=sample),
                    patch.object(native._fit, 'fit_naive_bayes_hsv', side_effect=train),
                    patch.object(native, '_calculation_records', return_value={})):
                with self.assertRaisesRegex((native.BayesDirectoryError,sampling.BayesSamplingError), 'changed'):
                    self.run_native(rng=rng, cancel=mutate)
            self.assertTrue(changed)



    def check_test_exact_walk_record_bound_is_independent_of_aggregate_entry_count(self):
        walk = [(str(self.images/str(i)), [], []) for i in range(native.MAX_DIRECTORY_ENTRIES+1)]
        with patch.object(native.os, 'walk', side_effect=lambda *a, **k:iter(walk[:-1])):
            selected = native.resolve_bayes_training_directories(self.images, self.masks)
        self.assertEqual(len(selected.walk), native.MAX_DIRECTORY_ENTRIES)
        with patch.object(native.os, 'walk', side_effect=lambda *a, **k:iter(walk)):
            with self.assertRaisesRegex(native.BayesDirectoryError, 'entry-count'):
                native.resolve_bayes_training_directories(self.images, self.masks)

    def test_validation_and_recovery(self):
        """2 related BayesDirectoryTrainingTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_caps_precede_decode_and_upload_and_invalid_arguments_consume_no_rng)
        self.scenario(self.check_test_retained_samples_rng_and_last_pdf_mutations_even_raised_callbacks)

    def test_workflow_lifecycle(self):
        """1 related BayesDirectoryTrainingTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_exact_walk_record_bound_is_independent_of_aggregate_entry_count)


if __name__ == '__main__':
    unittest.main()
