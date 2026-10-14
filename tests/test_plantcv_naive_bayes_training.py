"""Ordered CUDA KDE versus the actual pinned SciPy training dependency."""
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

import numpy as np
import torch

from seedvision.cuda import plantcv_naive_bayes_training as native
from seedvision.model_io import naive_bayes_provenance as records

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT/'artifacts/plantcv-reference-env/Scripts/python.exe'
SCRIPT = r'''
import hashlib, json, sys
from importlib.metadata import version
from pathlib import Path
import numpy as np
from scipy.stats import gaussian_kde
import scipy.stats._kde as kde_module
assert version('numpy') == '2.2.6'
assert version('scipy') == '1.15.3'
assert hashlib.sha256(Path(kde_module.__file__).read_bytes()).hexdigest() == 'e08717ede16febe4d374f21348036fae6a4739f0de88fa32e9c0e8a097ad6839'
out=[]
for values in json.loads(sys.stdin.read()):
    channels=[]
    for column in np.asarray(values,dtype=np.uint8).T:
        try:
            kde=gaussian_kde(column)
            channels.append(dict(pdf=kde(range(256)).tolist(), mean=float(np.average(column,weights=kde.weights)),
                covariance=float(kde._data_covariance[0,0]), factor=float(kde.factor), neff=float(kde.neff)))
        except Exception as e:
            channels.append(dict(error=type(e).__name__))
    out.append(channels)
print(json.dumps(out))
'''

# Descriptor-based fixtures keep million-row oracle inputs out of JSON. Only
# independently calculated compact moments/PDFs return from the pinned process.
MOMENT_SCRIPT = r'''
import hashlib, json, sys
from importlib.metadata import version
from pathlib import Path
import numpy as np
from scipy.stats import gaussian_kde
import scipy.stats._kde as kde_module
assert version('numpy') == '2.2.6'
assert version('scipy') == '1.15.3'
assert np.getbufsize() == 8192
assert hashlib.sha256(Path(kde_module.__file__).read_bytes()).hexdigest() == 'e08717ede16febe4d374f21348036fae6a4739f0de88fa32e9c0e8a097ad6839'
def fixture(n, kind, value):
    if kind == 'constant':
        return np.full((n,3), value, dtype=np.uint8)
    i = np.arange(n, dtype=np.uint64)
    return np.stack(((i*73+19+value)%256, (i*151+7+value)%256,
        (i*37+251+value)%256), axis=1).astype(np.uint8)
out=[]
for case in json.loads(sys.stdin.read()):
    x=fixture(case['n'], case['kind'], case['value'])
    w=np.ones(len(x), dtype=np.float64)/len(x)
    sw=float(w.sum())
    sw2=float((w*w).sum())
    channels=[]
    for column in x.T:
        weighted=column*w
        numerator=float(weighted.sum())
        assert numerator == float(weighted.reshape(1,-1).sum(axis=1)[0])
        mean=float(np.average(column, weights=w))
        result=dict(sums=[v.hex() for v in (sw,sw2,numerator)],
            initial=[v.hex() for v in (float(w[0]),sw,sw2,mean,sw-sw2/sw)])
        if case.get('pdf'):
            try:
                kde=gaussian_kde(column)
                result.update(pdf=kde(range(256)).tolist(), covariance=float(kde._data_covariance[0,0]),
                    factor=float(kde.factor), neff=float(kde.neff))
            except Exception as error:
                result['error']=type(error).__name__
        channels.append(result)
    out.append(channels)
print(json.dumps(out))
'''


def moment_fixture(n, kind, value):
    if kind == 'constant':
        return np.full((n, 3), value, dtype=np.uint8)
    i = np.arange(n, dtype=np.uint64)
    return np.stack(((i*73+19+value)%256, (i*151+7+value)%256,
        (i*37+251+value)%256), axis=1).astype(np.uint8)


def moment_oracle(cases):
    if not REFERENCE.is_file():
        raise AssertionError('the pinned NumPy/SciPy reference environment is required')
    run = subprocess.run([str(REFERENCE), '-c', MOMENT_SCRIPT], input=json.dumps(cases),
        capture_output=True, text=True, timeout=120)
    if run.returncode:
        raise AssertionError(run.stderr)
    return json.loads(run.stdout)


def oracle(arrays):
    if not REFERENCE.is_file():
        raise AssertionError('the pinned SciPy/PlantCV reference environment is required')
    run = subprocess.run([str(REFERENCE), '-c', SCRIPT], input=json.dumps([a.tolist() for a in arrays]),
        capture_output=True, text=True, timeout=120)
    if run.returncode:
        raise AssertionError(run.stderr)
    return json.loads(run.stdout)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA is required')
class BayesKdeTrainingTests(unittest.TestCase):
    def test_buffered_weighted_moment_bits_at_outer_boundaries_and_constant_outcomes(self):
        boundaries = (8191,8192,8193,16383,16384,16385,32767,32768,32769,
            65535,65536,65537,999999,1000000)
        original = (2,3,4,6,7,8,15,16,127,128,129,257,4095,4096,4097,
            65535,65536,65537,999999,1000000)
        cases = [dict(n=n, kind='constant', value=value)
            for n in sorted(set(boundaries+original)) for value in (0,1,10,127,179,200,255)]
        cases += [dict(n=n, kind='pattern', value=value) for n in boundaries for value in (0,117)]
        expected = moment_oracle(cases)
        # Observe the real production helper directly in addition to the actual
        # initial_stats entry point; no host replica of its reduction is used.
        audit = r'''
extern "C" __global__ void audit_sums(const unsigned char* x, double* out, int n) {
    int c=blockIdx.x*blockDim.x+threadIdx.x;
    if (c>=3) return;
    for (int kind=0; kind<3; ++kind)
        out[c*3+kind]=buffered_sum(x,n,c,1.0/(double)n,kind);
}
'''
        program = native._nvrtc.fixed_kernel_program(native._CUDA_SOURCE+audit, torch.device('cuda'))
        stats = torch.empty((3,8), dtype=torch.float64, device='cuda')
        sums = torch.empty((3,3), dtype=torch.float64, device='cuda')
        for case, channels in zip(cases, expected):
            with self.subTest(**case):
                x = torch.tensor(moment_fixture(**case), device='cuda')
                program.launch_1d('initial_stats', (x,stats), (case['n'],), 3)
                program.launch_1d('audit_sums', (x,sums), (case['n'],), 3)
                actual_stats, actual_sums = stats.cpu().tolist(), sums.cpu().tolist()
                for c, source in enumerate(channels):
                    self.assertEqual([v.hex() for v in actual_stats[c][:5]], source['initial'])
                    self.assertEqual([v.hex() for v in actual_sums[c]], source['sums'])
                    if case['kind'] == 'constant':
                        self.assertEqual([v.hex() for v in records._constant_moments(case['n'],case['value'])],
                            source['initial'])
        # These two former disagreements have opposite source singularity
        # outcomes. Exact means now recover both without relaxing the fit gate.
        by_case = {(case['n'],case['kind'],case['value']): channels[0]
            for case,channels in zip(cases,expected)}
        self.assertEqual(float.fromhex(by_case[65537,'constant',179]['initial'][3]), 179)
        self.assertNotEqual(float.fromhex(by_case[1000000,'constant',200]['initial'][3]), 200)

    def test_large_buffered_fits_match_actual_pinned_kde_all_pdf_values(self):
        cases = [dict(n=n,kind='pattern',value=117,pdf=True) for n in (8193,65537,1000000)]
        expected = moment_oracle(cases)
        for case, channels in zip(cases, expected):
            with self.subTest(n=case['n']):
                x = torch.tensor(moment_fixture(case['n'],case['kind'],case['value']), device='cuda')
                result = native.fit_naive_bayes_hsv((x,), ('one',))
                actual = result.probabilities.cpu().numpy()[0]
                for c, source in enumerate(channels):
                    fit = result.channels[0][c]
                    self.assertEqual(fit.mean.hex(), source['initial'][3])
                    self.assertEqual(fit.effective_samples, source['neff'])
                    self.assertEqual(fit.scott_factor, source['factor'])
                    self.assertEqual(fit.numerical_profile, native.NUMERICAL_PROFILE)
                    self.assertLessEqual(abs(fit.covariance-source['covariance']), fit.covariance_absolute_budget)
                    np.testing.assert_allclose(actual[c], source['pdf'], rtol=0, atol=fit.pdf_absolute_budget)
                    np.testing.assert_allclose(actual[c], source['pdf'], rtol=3e-12, atol=2e-15)

    def test_fixed_reference_profile_ignores_installed_numpy_buffer_setting(self):
        x = torch.tensor(moment_fixture(8193, 'pattern', 0), device='cuda')
        first = native.fit_naive_bayes_hsv((x,), ('one',))
        saved = np.getbufsize()
        try:
            np.setbufsize(1024)
            self.assertEqual(np.getbufsize(), 1024)
            second = native.fit_naive_bayes_hsv((x,), ('one',))
        finally:
            np.setbufsize(saved)
        self.assertEqual(first.channels, second.channels)
        torch.testing.assert_close(first.probabilities, second.probabilities, rtol=0, atol=0)

    def test_all_pdf_values_across_pairwise_transitions_source_order_and_tails(self):
        rng = np.random.default_rng(171)
        arrays = [rng.integers(0, 256, (n, 3), dtype=np.uint8)
            for n in (2, 3, 4, 7, 8, 15, 127, 128, 129, 257, 4095, 4096, 4097, 4117)]
        arrays += [np.array([[0, 1, 254], [255, 2, 255]], dtype=np.uint8),
            np.array([[200, 100, 1]]*128+[[201, 101, 2]], dtype=np.uint8)]
        arrays += [arrays[-1][::-1].copy()]
        expected = oracle(arrays)
        xs = tuple(torch.tensor(a, device='cuda') for a in arrays)
        before = tuple(x.clone() for x in xs)
        result = native.fit_naive_bayes_hsv(xs, tuple(f'c{i}' for i in range(len(xs))))
        self.assertEqual(result.probabilities.dtype, torch.float64)
        self.assertEqual(result.probabilities.device.type, 'cuda')
        actual = result.probabilities.cpu().numpy()
        for index, channels in enumerate(expected):
            for c, source in enumerate(channels):
                fit = result.channels[index][c]
                self.assertNotIn('error', source)
                self.assertEqual(fit.mean, source['mean'])
                self.assertLessEqual(abs(fit.covariance-source['covariance']), fit.covariance_absolute_budget)
                self.assertEqual(fit.effective_samples, source['neff'])
                self.assertEqual(fit.scott_factor, source['factor'])
                np.testing.assert_allclose(actual[index, c], source['pdf'], rtol=0,
                    atol=fit.pdf_absolute_budget, err_msg=f'class{index}/channel{c}')
                # This independent tighter check detects mistakes that a deliberately
                # conservative reduction budget could otherwise hide.
                np.testing.assert_allclose(actual[index, c], source['pdf'], rtol=3e-12, atol=2e-15)
        for x, saved in zip(xs, before):
            torch.testing.assert_close(x, saved, rtol=0, atol=0)

    def test_ordered_density_chunks_are_bit_identical_and_one_class_is_valid(self):
        array = np.random.default_rng(21).integers(0, 256, (521, 3), dtype=np.uint8)
        x = torch.tensor(array, device='cuda')
        first = native.fit_naive_bayes_hsv((x,), ('one',))
        with patch.object(native, 'SAMPLE_TILE', 13):
            second = native.fit_naive_bayes_hsv((x,), ('one',))
        torch.testing.assert_close(first.probabilities, second.probabilities, rtol=0, atol=0)
        self.assertEqual(first.channels, second.channels)
        self.assertEqual(first.class_names, ('one',))

    def test_constant_source_success_and_singular_failures_all_pdf_bins_without_variance_floor(self):
        cases = [dict(n=n,kind='constant',value=value,pdf=True)
            for n in (3,6,7) for value in range(256)]
        cases += [dict(n=n,kind='constant',value=value,pdf=True) for n,value in
            ((2,1),(129,10),(8191,179),(8192,200),(8193,255),
             (65535,179),(65536,255),(65537,179),(999999,200),(1000000,200))]
        expected = moment_oracle(cases)
        outcomes = {}
        for case, channels in zip(cases,expected):
            with self.subTest(n=case['n'],value=case['value']):
                x = torch.full((case['n'],3),case['value'],dtype=torch.uint8,device='cuda')
                if any('error' in source for source in channels):
                    self.assertTrue(all('error' in source for source in channels))
                    self.assertTrue(all(float.fromhex(source['initial'][3]) == case['value'] for source in channels))
                    with self.assertRaisesRegex(native.BayesTrainingError, 'constant HSV.*singular'):
                        native.fit_naive_bayes_hsv((x,),('constant',))
                    outcomes[case['n'],case['value']] = 'singular'
                    continue
                result = native.fit_naive_bayes_hsv((x,),('constant',))
                actual = result.probabilities.cpu().numpy()[0]
                for c,source in enumerate(channels):
                    fit = result.channels[0][c]
                    self.assertIs(type(fit),native.BayesConstantChannelFit)
                    self.assertEqual(fit.constant_value,case['value'])
                    self.assertEqual(fit.numerical_profile,native.CONSTANT_NUMERICAL_PROFILE)
                    self.assertEqual(fit.mean.hex(),source['initial'][3])
                    self.assertNotEqual(fit.mean,case['value'])
                    self.assertGreater(fit.covariance,0)
                    self.assertLess(fit.covariance,1e-20)
                    self.assertLessEqual(abs(fit.covariance-source['covariance']),fit.covariance_absolute_budget)
                    self.assertEqual(fit.effective_samples,source['neff'])
                    self.assertEqual(fit.scott_factor,source['factor'])
                    np.testing.assert_allclose(actual[c],source['pdf'],rtol=0,atol=fit.pdf_absolute_budget)
                    np.testing.assert_allclose(actual[c],source['pdf'],rtol=3e-10,atol=0)
                    self.assertGreater(actual[c,case['value']],0)
                    np.testing.assert_array_equal(np.delete(actual[c],case['value']),np.zeros(255))
                    np.testing.assert_array_equal(np.delete(source['pdf'],case['value']),np.zeros(255))
                outcomes[case['n'],case['value']] = 'positive'
        for case in ((3,200),(6,255),(7,10),(1000000,200)):
            self.assertEqual(outcomes[case],'positive')
        self.assertEqual(outcomes[65537,179],'singular')

    def test_all_constant_byte_mean_bits_around_default_buffer_boundaries(self):
        cases = [dict(n=n,kind='constant',value=value) for n in (8191,8192,8193) for value in range(256)]
        expected = moment_oracle(cases)
        program = native._nvrtc.fixed_kernel_program(native._CUDA_SOURCE,torch.device('cuda'))
        stats = torch.empty((3,8),dtype=torch.float64,device='cuda')
        for case,channels in zip(cases,expected):
            with self.subTest(n=case['n'],value=case['value']):
                x = torch.full((case['n'],3),case['value'],dtype=torch.uint8,device='cuda')
                program.launch_1d('initial_stats',(x,stats),(case['n'],),3)
                actual = stats.cpu().tolist()
                compact = records._constant_moments(case['n'],case['value'])
                for c,source in enumerate(channels):
                    self.assertEqual([v.hex() for v in actual[c][:5]],source['initial'])
                    self.assertEqual([v.hex() for v in compact],source['initial'])

    def test_mixed_constant_and_varied_channels_and_late_singular_class_withhold_completion(self):
        varied = np.stack((np.arange(6),np.arange(10,16),np.full(6,255)),axis=1).astype(np.uint8)
        expected = oracle([varied])[0]
        x = torch.tensor(varied,device='cuda')
        result = native.fit_naive_bayes_hsv((x,),('mixed',))
        self.assertEqual([type(f) for f in result.channels[0]],
            [native.BayesChannelFit,native.BayesChannelFit,native.BayesConstantChannelFit])
        for c,source in enumerate(expected):
            np.testing.assert_allclose(result.probabilities.cpu().numpy()[0,c],source['pdf'],
                rtol=0,atol=result.channels[0][c].pdf_absolute_budget)
        singular = x.clone()
        singular[:,2] = 0
        with self.assertRaisesRegex(native.BayesTrainingError,"class 'late'.*value.*singular"):
            native.fit_naive_bayes_hsv((x,singular),('positive','late'))

    def test_invalid_inputs_and_all_caps_fail_before_kernel_compilation(self):
        x = torch.arange(9, dtype=torch.uint8, device='cuda').reshape(3, 3)
        cases = [((x.cpu(),), ('a',)), ((x.float(),), ('a',)), ((x[:, :2],), ('a',)),
            ((x[:1],), ('a',)), ((x,), ('',)), ((x, x), ('a', 'a')), ([x], ('a',)),
            ((x,), ('a\tb',)), ((x.T,), ('a',)), ((), ())]
        with patch.object(native._nvrtc, 'fixed_kernel_program', side_effect=AssertionError('allocated before validation')):
            for samples, names in cases:
                with self.subTest(names=names), self.assertRaises(native.BayesTrainingError):
                    native.fit_naive_bayes_hsv(samples, names)
            for cap in ('MAX_SAMPLES', 'MAX_WORK', 'MAX_SCRATCH_BYTES', 'MAX_CUDA_LIVE_BYTES'):
                with self.subTest(cap=cap), patch.object(native, cap, 1):
                    with self.assertRaises(native.BayesTrainingError):
                        native.fit_naive_bayes_hsv((x,), ('a',))

    def test_cancellation_and_mutating_callbacks_publish_no_partial_fit(self):
        x = torch.arange(27, dtype=torch.uint8, device='cuda').reshape(9, 3)
        calls = 0
        def cancel():
            nonlocal calls
            calls += 1
            return calls == 4
        with self.assertRaises(native.BayesTrainingCancelled):
            native.fit_naive_bayes_hsv((x,), ('a',), cancel=cancel)
        for value in (None, 1, 'yes'):
            with self.assertRaises(native.BayesTrainingError):
                native.fit_naive_bayes_hsv((x,), ('a',), cancel=lambda: value)
        def mutate():
            x.add_(1)
            return False
        with self.assertRaisesRegex(native.BayesTrainingError, 'changed'):
            native.fit_naive_bayes_hsv((x,), ('a',), cancel=mutate)
        def mutate_and_raise():
            x.add_(1)
            raise RuntimeError('callback failure')
        with self.assertRaisesRegex(native.BayesTrainingError, 'changed'):
            native.fit_naive_bayes_hsv((x,), ('a',), cancel=mutate_and_raise)
        calls = 0
        def mutate_later():
            nonlocal calls
            calls += 1
            if calls == 5:
                x.add_(1)
            return False
        with self.assertRaisesRegex(native.BayesTrainingError, 'changed'):
            native.fit_naive_bayes_hsv((x,), ('a',), cancel=mutate_later)

    def test_custom_stream_actual_peak_and_full_backing_storage_charge(self):
        x = torch.tensor(np.random.default_rng(88).integers(0, 256, (8203, 3), dtype=np.uint8), device='cuda')
        native.fit_naive_bayes_hsv((x[:3].contiguous(),), ('warmup',))
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            result = native.fit_naive_bayes_hsv((x,), ('a',))
        stream.synchronize()
        self.assertLessEqual(torch.cuda.max_memory_allocated()-baseline, result.plan.scratch_bytes)
        self.assertEqual(result.plan.live_bytes, result.plan.scratch_bytes+x.untyped_storage().nbytes())
        with patch.object(native, 'MAX_CUDA_LIVE_BYTES', result.plan.scratch_bytes+30):
            with self.assertRaises(native.BayesTrainingError):
                native.fit_naive_bayes_hsv((x[:3],), ('a',))

    def test_partial_pdf_is_watched_during_core_density_callbacks(self):
        x = torch.arange(27, dtype=torch.uint8, device='cuda').reshape(9, 3)
        original = torch.zeros
        for kind in ('value', 'storage'):
            with self.subTest(kind=kind):
                pdf = None
                calls = 0
                def capture(*args, **kwargs):
                    nonlocal pdf
                    tensor = original(*args, **kwargs)
                    if tuple(tensor.shape) == (1,3,256):
                        pdf = tensor
                    return tensor
                def mutate_during_density():
                    nonlocal calls
                    calls += 1
                    if calls == 8:  # After the first density chunk.
                        self.assertIsNotNone(pdf)
                        if kind == 'value':
                            pdf.add_(1)
                        else:
                            pdf.data = pdf.clone()
                    return False
                with patch.object(native, 'SAMPLE_TILE', 3), patch.object(torch, 'zeros', side_effect=capture):
                    with self.assertRaisesRegex(native.BayesTrainingError, 'changed'):
                        native.fit_naive_bayes_hsv((x,), ('a',), cancel=mutate_during_density)


if __name__ == '__main__':
    unittest.main()
