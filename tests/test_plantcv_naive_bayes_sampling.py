"""Actual public binary Bayes sampler observations, RNG and CUDA boundaries."""

from tests.functional_units import functional_units
import json
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

import numpy as np
import torch

from seedvision.cuda import plantcv_naive_bayes_sampling as native
from seedvision.cuda import plantcv_naive_bayes_training as training
from seedvision.cuda.plantcv_naive_bayes import parse_plantcv_pdf

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
answers=[]
for case in json.load(sys.stdin):
    np.random.seed(case['seed'])
    if case.get('cached_gaussian'): np.random.normal()
    before=state(); captures=[]; steps=[]
    pairs=[(np.asarray(i,dtype=np.uint8),np.asarray(mask,dtype=np.uint8)) for i,mask in case['pairs']]
    names=[str(i)+'.png' for i in range(len(pairs))]
    actual_randint=np.random.randint
    def draw(*args,**kwargs):
        value=actual_randint(*args,**kwargs)
        steps.append(state())
        return value
    def read(path,flags):
        index=int(Path(path).stem)
        return pairs[index][0 if flags==1 else 1]
    actual_kde=m.stats.gaussian_kde
    def kde(values):
        captures.append(values.tolist())
        if case.get('real_fit'): return actual_kde(values)
        return lambda queries: np.full(len(queries),1/256)
    with tempfile.TemporaryDirectory() as folder:
        output=Path(folder)/'model.tsv'
        answer=dict(before=before)
        try:
            with patch.object(m.os,'walk',return_value=[('images',[],names)]),patch.object(m.os.path,'exists',return_value=True),patch.object(m.cv2,'imread',side_effect=read),patch.object(m.stats,'gaussian_kde',side_effect=kde),patch.object(np.random,'randint',side_effect=draw),redirect_stdout(io.StringIO()):
                m.naive_bayes('images','masks',str(output))
            answer['columns']=captures
            if case.get('real_fit'): answer['tsv']=output.read_text()
        except Exception as error:
            answer['error']=type(error).__name__
        answer.update(after=state(),steps=steps)
        answers.append(answer)
print(json.dumps(answers))
'''


def state_record(value):
    return dict(algorithm=value.algorithm, keys=value.keys.hex(), position=value.position,
        has_gauss=value.has_gauss, cached_gaussian=value.cached_gaussian.hex())


def public(cases):
    payload = [dict(case, pairs=[(image.tolist(), mask.tolist()) for image, mask in case['pairs']])
        for case in cases]
    result = subprocess.run([str(REFERENCE), '-c', SCRIPT], input=json.dumps(payload),
        text=True, capture_output=True, timeout=120)
    if result.returncode:
        raise AssertionError(result.stderr)
    return json.loads(result.stdout)


def pair(height=31, width=37, foreground=555, background=None, seed=991):
    image = np.random.default_rng(seed).integers(0, 256, (height, width, 3), dtype=np.uint8)
    mask = np.full(height*width, 128, dtype=np.uint8)
    mask[:foreground] = 255
    end = height*width if background is None else foreground+background
    mask[foreground:end] = 0
    return image, mask.reshape(height, width)


def coordinate_pair(image_shape=(31,37), mask_shape=(20,19), *, foreground=None, seed=991):
    """Categorical values in the overlap, ignored128 elsewhere in the mask."""
    image = np.random.default_rng(seed).integers(0,256,(*image_shape,3),dtype=np.uint8)
    height,width = min(image_shape[0],mask_shape[0]),min(image_shape[1],mask_shape[1])
    valid = np.zeros((height,width),dtype=np.uint8)
    valid.reshape(-1)[:height*width//2 if foreground is None else foreground] = 255
    mask = np.full(mask_shape,128,dtype=np.uint8)
    mask[:height,:width] = valid
    return image,mask


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
@functional_units
class BayesImageSamplingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = [dict(seed=17, pairs=[pair()], cached_gaussian=True),
            dict(seed=31, pairs=[pair(9, 11, 23, 37)]),
            dict(seed=19, pairs=[pair(1, 12, 10, 2)]),
            dict(seed=11, pairs=[pair(1, 9, 4, 5)]),
            dict(seed=11, pairs=[pair(1, 2, 1, 1)]),
            dict(seed=11, pairs=[pair(1, 2, 0, 0)]),
            dict(seed=11, pairs=[pair(1, 12, 10, 1)]),
            dict(seed=11, pairs=[pair(1, 10, 10, 0)]),
            dict(seed=41, pairs=[pair(), pair(39, 41, 901, seed=992)]),
            dict(seed=43, pairs=[pair(41, 43, 860)], real_fit=True)]
        cls.expected = public(cls.cases)

    def check_test_actual_public_independent_columns_default_dtype_and_complete_rng_state(self):
        for case, expected in zip(self.cases[:6], self.expected[:6]):
            image, mask = case['pairs'][0]
            rng = np.random.RandomState(case['seed'])
            if case.get('cached_gaussian'):
                rng.normal()
            source = torch.tensor(image, device='cuda')
            selection = torch.tensor(mask, device='cuda')
            signatures = native._tensor_signature(source), native._tensor_signature(selection)
            result = native.sample_naive_bayes_image(source, selection, rng=rng,
                conversion_tile=13, gather_tile=7)
            with self.subTest(shape=image.shape, foreground=result.plan.foreground_pixels):
                self.assertEqual(state_record(result.rng_before), expected['before'])
                self.assertEqual(state_record(result.rng_after), expected['after'])
                self.assertEqual(result.rng_after, native._state(rng))
                self.assertEqual(result.random_calls, 6)
                self.assertEqual(result.random_index_dtype, np.dtype('l').str)
                self.assertEqual(result.class_names, ('plant', 'background'))
                for category, values in enumerate(result.samples):
                    self.assertEqual(values.device, source.device)
                    self.assertEqual(values.dtype, torch.uint8)
                    self.assertTrue(values.is_contiguous())
                    for channel in range(3):
                        self.assertEqual(values[:, channel].cpu().tolist(), expected['columns'][channel*2+category])
                self.assertEqual(signatures, (native._tensor_signature(source), native._tensor_signature(selection)))
                np.testing.assert_array_equal(source.cpu().numpy(), image)
                np.testing.assert_array_equal(selection.cpu().numpy(), mask)

    def check_test_source_invalid_background_follows_consumed_foreground_draw(self):
        for case, expected in zip(self.cases[6:8], self.expected[6:8]):
            image, mask = case['pairs'][0]
            rng = np.random.RandomState(case['seed'])
            self.assertEqual(expected['error'], 'ValueError')
            self.assertEqual(len(expected['steps']), 1)
            self.assertNotEqual(expected['before'], expected['after'])
            with self.assertRaisesRegex(native.BayesSamplingSourceError, 'channel0/class1'):
                native.sample_naive_bayes_image(torch.tensor(image, device='cuda'),
                    torch.tensor(mask, device='cuda'), rng=rng)
            self.assertEqual(state_record(native._state(rng)), expected['after'])

    def check_test_actual_public_coordinate_indexing_smaller_larger_mixed_and_ignored_masks(self):
        shapes = (((31,37),(20,19)),((20,19),(31,37)),((31,37),(20,43)),
            ((20,43),(31,37)),((31,37),(41,19)),((31,37),(99,111)),((1,23),(3,27)))
        cases = [dict(seed=17,pairs=[coordinate_pair(a,b)]) for a,b in shapes]
        for case,expected in zip(cases,public(cases)):
            self.assertNotIn('error',expected)
            image,mask = case['pairs'][0]
            rng = np.random.RandomState(17)
            result = native.sample_naive_bayes_image(torch.tensor(image,device='cuda'),
                torch.tensor(mask,device='cuda'),rng=rng,conversion_tile=31,gather_tile=7)
            with self.subTest(image=image.shape,mask=mask.shape):
                self.assertEqual((result.plan.height,result.plan.width),image.shape[:2])
                self.assertEqual((result.plan.mask_height,result.plan.mask_width),mask.shape)
                self.assertEqual(result.plan.mask_tiles,(mask.size+4095)//4096)
                self.assertEqual(state_record(result.rng_before),expected['before'])
                self.assertEqual(state_record(native._state(rng)),expected['after'])
                for category,values in enumerate(result.samples):
                    for channel in range(3):
                        self.assertEqual(values[:,channel].cpu().tolist(),expected['columns'][channel*2+category])

    def check_test_categorical_out_of_bounds_priority_and_zero_rows_fail_before_any_draw(self):
        cases,priorities = [],[]
        for foreground,kind in ((555,'foreground'),(555,'background'),(555,'both'),
                (0,'foreground'),(0,'background')):
            image,mask = coordinate_pair((31,37),(33,39),foreground=foreground)
            if kind in ('foreground','both'): mask[32,0] = 255
            if kind in ('background','both'): mask[0,38] = 0
            cases.append(dict(seed=17,pairs=[(image,mask)]))
            priorities.append('foreground' if kind in ('foreground','both') else 'background')
        saved = np.random.get_state()
        try:
            for case,priority,expected in zip(cases,priorities,public(cases)):
                self.assertEqual(expected['error'],'IndexError')
                self.assertEqual(expected['steps'],[])
                self.assertEqual(expected['before'],expected['after'])
                image,mask = case['pairs'][0]
                np.random.seed(17)
                with (self.subTest(priority=priority,foreground=np.count_nonzero(mask==255)),
                        patch.object(np.random,'randint',side_effect=AssertionError('draw before failed indexing')),
                        self.assertRaisesRegex(native.BayesSamplingSourceError,'hue '+priority)):
                    native.sample_naive_bayes_image(torch.tensor(image,device='cuda'),
                        torch.tensor(mask,device='cuda'))
                self.assertEqual(state_record(native._state(np.random)),expected['after'])
        finally:
            np.random.set_state(saved)

    def check_test_multiple_occurrences_append_in_source_order_and_fit_actual_public_pdfs(self):
        case, expected = self.cases[8], self.expected[8]
        rng = np.random.RandomState(case['seed'])
        parts = [native.sample_naive_bayes_image(torch.tensor(image, device='cuda'),
            torch.tensor(mask, device='cuda'), rng=rng) for image, mask in case['pairs']]
        for category in range(2):
            joined = torch.cat([p.samples[category] for p in parts])
            for channel in range(3):
                self.assertEqual(joined[:, channel].cpu().tolist(), expected['columns'][channel*2+category])
        self.assertEqual(state_record(native._state(rng)), expected['after'])
        case, expected = self.cases[9], self.expected[9]
        image, mask = case['pairs'][0]
        samples = native.sample_naive_bayes_image(torch.tensor(image, device='cuda'),
            torch.tensor(mask, device='cuda'), rng=np.random.RandomState(case['seed']))
        fitted = training.fit_naive_bayes_hsv(samples.samples, samples.class_names)
        model = parse_plantcv_pdf(expected['tsv'])
        self.assertEqual(model.class_names, samples.class_names)
        for category, channels in enumerate(fitted.channels):
            for channel, fit in enumerate(channels):
                np.testing.assert_allclose(fitted.probabilities[category, channel].cpu().numpy(),
                    model.probabilities[category][channel], rtol=0, atol=fit.pdf_absolute_budget)

    def check_test_implicit_global_state_uses_existing_stream_and_retains_gaussian_cache(self):
        original = np.random.get_state()
        try:
            case, expected = self.cases[0], self.expected[0]
            np.random.seed(case['seed'])
            np.random.normal()
            image, mask = case['pairs'][0]
            result = native.sample_naive_bayes_image(torch.tensor(image, device='cuda'),
                torch.tensor(mask, device='cuda'))
            self.assertEqual(state_record(result.rng_before), expected['before'])
            self.assertEqual(state_record(native._state(np.random)), expected['after'])
        finally:
            np.random.set_state(original)

    def check_test_cap_and_layout_failures_precede_compilation_or_random_draws(self):
        image, mask = pair()
        source, selection = torch.tensor(image, device='cuda'), torch.tensor(mask, device='cuda')
        rng = np.random.RandomState(17)
        before = native._state(rng)
        with patch.object(native._nvrtc, 'fixed_kernel_program', side_effect=AssertionError('compiled before static bound')):
            for name in ('MAX_IMAGE_PIXELS', 'MAX_WORK', 'MAX_CUDA_LIVE_BYTES', 'MAX_CPU_METADATA_BYTES'):
                with self.subTest(cap=name), patch.object(native, name, 1), self.assertRaises(native.BayesSamplingError):
                    native.sample_naive_bayes_image(source, selection, rng=rng)
            for bgr, binary in ((source.cpu(), selection), (source.float(), selection),
                    (source.transpose(0, 1), selection.T), (source, selection[:0])):
                with self.assertRaises(native.BayesSamplingError):
                    native.sample_naive_bayes_image(bgr, binary, rng=rng)
            for kwargs in (dict(rng=np.random.default_rng(1)), dict(conversion_tile=True),
                    dict(gather_tile=0), dict(cancel=False)):
                with self.assertRaises(native.BayesSamplingError):
                    native.sample_naive_bayes_image(source, selection, **kwargs)
            with patch.object(native, 'MAX_WORK', 200_000), self.assertRaisesRegex(native.BayesSamplingError, 'work'):
                native.sample_naive_bayes_image(source, selection, rng=rng, conversion_tile=1)
        with patch.object(native, 'MAX_SAMPLES', 1), self.assertRaises(native.BayesSamplingError):
            native.sample_naive_bayes_image(source, selection, rng=rng)
        self.assertEqual(before, native._state(rng))
        # Sparse foreground can fit16M pixels; dense foreground hits the separate
        # two-class sample cap. Complete backing storage is charged independently.
        domain = (4000, 4000, 4000, 4000, 3907, 64_000_000)
        self.assertLess(native._plan(domain, 1000, 15_999_000, 16384, 16384).peak_live_bytes,
            native.MAX_CUDA_LIVE_BYTES)
        with self.assertRaisesRegex(native.BayesSamplingError, 'sample cap'):
            native._plan(domain, 16_000_000, 0, 16384, 16384)
        backing = torch.zeros(native.MAX_CUDA_LIVE_BYTES, dtype=torch.uint8, device='cuda')
        with self.assertRaisesRegex(native.BayesSamplingError, 'input/count'):
            native.sample_naive_bayes_image(backing[:3].reshape(1,1,3), selection[:1,:1].contiguous(), rng=rng)

    def check_test_cancellation_after_each_actual_public_draw_preserves_partial_rng_state(self):
        case, expected = self.cases[0], self.expected[0]
        image, mask = case['pairs'][0]
        bgr, binary = torch.tensor(image, device='cuda'), torch.tensor(mask, device='cuda')
        for target in [expected['before'], *expected['steps']]:
            rng = np.random.RandomState(case['seed'])
            rng.normal()
            with self.subTest(position=target['position']), self.assertRaises(native.BayesSamplingCancelled):
                native.sample_naive_bayes_image(bgr, binary, rng=rng,
                    cancel=lambda: state_record(native._state(rng)) == target)
            self.assertEqual(state_record(native._state(rng)), target)

    def check_test_callback_input_partial_output_and_rng_mutations_are_guarded_even_on_raise(self):
        image, mask = pair()
        for kind in ('image', 'mask_storage', 'output', 'rng', 'image_and_raise'):
            source, selection = torch.tensor(image, device='cuda'), torch.tensor(mask, device='cuda')
            rng, initial = np.random.RandomState(17), native._state(np.random.RandomState(17))
            allocated = []
            empty = torch.empty
            def tracked(*args, **kwargs):
                x = empty(*args, **kwargs)
                if x.dtype == torch.uint8 and tuple(x.shape) == (555//10, 3):
                    allocated.append(x)
                return x
            changed = False
            def mutate():
                nonlocal changed
                if not changed and native._state(rng) != initial:
                    changed = True
                    if kind.startswith('image'):
                        source.add_(1)
                    elif kind == 'mask_storage':
                        selection.data = selection.clone()
                    elif kind == 'output':
                        allocated[0].transpose_(0,1)
                    else:
                        rng.randint(0, 100, 3)
                    if kind.endswith('raise'):
                        raise RuntimeError('callback raised after changing input')
                return False
            with self.subTest(kind=kind), patch.object(native.torch, 'empty', side_effect=tracked):
                with self.assertRaisesRegex(native.BayesSamplingError, 'changed'):
                    native.sample_naive_bayes_image(source, selection, rng=rng, cancel=mutate)
            self.assertTrue(changed)
        for answer in (None, 0, [], '', 1, [False]):
            with self.subTest(callback_return=answer), self.assertRaisesRegex(native.BayesSamplingError, 'return bool'):
                native.sample_naive_bayes_image(source, selection, cancel=lambda: answer)

    def check_test_custom_stream_actual_peak_and_tile_repartition_preserve_selected_bytes(self):
        image, mask = pair(1000,1000,510_007)
        source, selection = torch.tensor(image, device='cuda'), torch.tensor(mask, device='cuda')
        native.sample_naive_bayes_image(source[:2,:2].contiguous(), selection[:2,:2].contiguous(),
            rng=np.random.RandomState(1))
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            first = native.sample_naive_bayes_image(source, selection, rng=np.random.RandomState(17),
                conversion_tile=65536, gather_tile=13001)
        stream.synchronize()
        actual = torch.cuda.max_memory_allocated()-baseline
        backing = source.untyped_storage().nbytes()+selection.untyped_storage().nbytes()
        self.assertLessEqual(actual, first.plan.peak_live_bytes-backing)
        second = native.sample_naive_bayes_image(source, selection, rng=np.random.RandomState(17),
            conversion_tile=3211, gather_tile=997)
        self.assertEqual(first.rng_after, second.rng_after)
        self.assertEqual(first.draws_sha256, second.draws_sha256)
        self.assertLess(first.plan.work, second.plan.work)
        self.assertGreater(first.plan.mask_tiles, 1)
        for a,b in zip(first.samples, second.samples):
            self.assertTrue(torch.equal(a,b))
        self.assertEqual(first.plan.output_bytes, sum(x.numel() for x in first.samples))

    def check_test_different_dimensions_actual_peak_stream_repartition_and_compact_only_transfer(self):
        image,mask = coordinate_pair((700,900),(800,1000))
        source,selection = torch.tensor(image,device='cuda'),torch.tensor(mask,device='cuda')
        native.sample_naive_bayes_image(source[:1,:1].contiguous(),selection[:1,:1].contiguous(),
            rng=np.random.RandomState(17))
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        cpu = torch.Tensor.cpu
        def compact_only(tensor,*args,**kwargs):
            self.assertEqual(tuple(tensor.shape),((mask.size+4095)//4096,4))
            return cpu(tensor,*args,**kwargs)
        with torch.cuda.stream(stream),patch.object(torch.Tensor,'cpu',compact_only):
            first = native.sample_naive_bayes_image(source,selection,rng=np.random.RandomState(17),
                conversion_tile=65536,gather_tile=13001)
        stream.synchronize()
        actual = torch.cuda.max_memory_allocated()-baseline
        backing = source.untyped_storage().nbytes()+selection.untyped_storage().nbytes()
        self.assertLessEqual(actual,first.plan.peak_live_bytes-backing)
        second = native.sample_naive_bayes_image(source,selection,rng=np.random.RandomState(17),
            conversion_tile=3211,gather_tile=997)
        self.assertEqual(first.rng_after,second.rng_after)
        self.assertEqual(first.draws_sha256,second.draws_sha256)
        self.assertGreater(second.plan.work,first.plan.work)
        for a,b in zip(first.samples,second.samples): self.assertTrue(torch.equal(a,b))

    def check_test_independent_mask_cap_and_maximum_ignored_mask_preflight(self):
        source = torch.zeros((1,1,3),dtype=torch.uint8,device='cuda')
        selection = torch.full((4000,4000),128,dtype=torch.uint8,device='cuda')
        rng = np.random.RandomState(17)
        initial = native._state(rng)
        with (patch.object(native,'MAX_IMAGE_PIXELS',15_999_999),
                patch.object(native._nvrtc,'fixed_kernel_program',side_effect=AssertionError('compiled before cap')),
                self.assertRaisesRegex(native.BayesSamplingError,'pixel cap')):
            native.sample_naive_bayes_image(source,selection,rng=rng)
        self.assertEqual(native._state(rng),initial)
        result = native.sample_naive_bayes_image(source,selection,rng=rng)
        self.assertEqual(result.plan.mask_tiles,3907)
        self.assertEqual(result.plan.samples_per_class,0)
        self.assertEqual(result.rng_after,initial)
        self.assertTrue(all(x.shape==(0,3) for x in result.samples))

    def check_test_final_callback_storage_swap_or_raised_rng_mutation_withholds_samples(self):
        image, mask = pair()
        source, selection = torch.tensor(image, device='cuda'), torch.tensor(mask, device='cuda')
        calls = 0
        def count():
            nonlocal calls
            calls += 1
            return False
        native.sample_naive_bayes_image(source, selection, rng=np.random.RandomState(17), cancel=count)
        final_call = calls
        empty = torch.empty
        for kind in ('output_storage', 'rng_and_raise'):
            calls, allocated = 0, []
            rng = np.random.RandomState(17)
            def tracked(*args, **kwargs):
                x = empty(*args, **kwargs)
                if x.dtype == torch.uint8 and tuple(x.shape) == (555//10, 3):
                    allocated.append(x)
                return x
            def mutate():
                nonlocal calls
                calls += 1
                if calls == final_call:
                    if kind == 'output_storage':
                        allocated[1].data = allocated[1].clone()
                    else:
                        rng.randint(0, 100, 3)
                        raise RuntimeError('final callback modified RNG and raised')
                return False
            with self.subTest(kind=kind), patch.object(native.torch, 'empty', side_effect=tracked):
                with self.assertRaisesRegex(native.BayesSamplingError, 'changed'):
                    native.sample_naive_bayes_image(source, selection, rng=rng, cancel=mutate)
            self.assertEqual(calls, final_call)

    def check_test_rng_change_inside_draw_is_detected_by_predicted_stream(self):
        image, mask = pair()
        source, selection = torch.tensor(image, device='cuda'), torch.tensor(mask, device='cuda')
        original = np.random.get_state()
        actual_randint = np.random.randint
        try:
            np.random.seed(17)
            def race(*args, **kwargs):
                result = actual_randint(*args, **kwargs)
                actual_randint(0, 100, 3)
                return result
            with patch.object(np.random, 'randint', side_effect=race):
                with self.assertRaisesRegex(native.BayesSamplingError, 'changed during an ordered draw'):
                    native.sample_naive_bayes_image(source, selection)
        finally:
            np.random.set_state(original)

    def check_test_directory_reservations_fail_before_current_draw_and_charge_full_backing(self):
        image, mask = pair()
        source, selection = torch.tensor(image, device='cuda'), torch.tensor(mask, device='cuda')
        first = native.sample_naive_bayes_image(source, selection, rng=np.random.RandomState(17))
        rng = np.random.RandomState(31)
        before = native._state(rng)
        budget = native._JobBudget(first.plan.output_bytes, 2*first.plan.samples_per_class, first.plan.work)
        domains = (
            ('MAX_SAMPLES', 2*first.plan.samples_per_class+1),
            ('MAX_WORK', 2*first.plan.work-1),
            ('MAX_CUDA_LIVE_BYTES', first.plan.peak_live_bytes+first.plan.output_bytes-1),
        )
        for cap, maximum in domains:
            with self.subTest(cap=cap), patch.object(native, cap, maximum):
                with self.assertRaises(native.BayesSamplingError):
                    native.sample_naive_bayes_image(source, selection, rng=rng, _budget=budget)
            self.assertEqual(before, native._state(rng))
        for bad in (False, object(), native._JobBudget(retained_rows=True),
                native._JobBudget(consumed_work=-1)):
            with self.subTest(budget=bad), self.assertRaises(native.BayesSamplingError):
                native.sample_naive_bayes_image(source, selection, rng=rng, _budget=bad)
        reserved = native._JobBudget(retained_backing_bytes=native.MAX_CUDA_LIVE_BYTES)
        with patch.object(native._nvrtc, 'fixed_kernel_program', side_effect=AssertionError('compiled')):
            with self.assertRaisesRegex(native.BayesSamplingError, 'input/count'):
                native.sample_naive_bayes_image(source, selection, rng=rng, _budget=reserved)
        self.assertEqual(before, native._state(rng))

    def test_workflow_lifecycle(self):
        """7 related BayesImageSamplingTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_actual_public_independent_columns_default_dtype_and_complete_rng_state)
        self.scenario(self.check_test_actual_public_coordinate_indexing_smaller_larger_mixed_and_ignored_masks)
        self.scenario(self.check_test_multiple_occurrences_append_in_source_order_and_fit_actual_public_pdfs)
        self.scenario(self.check_test_implicit_global_state_uses_existing_stream_and_retains_gaussian_cache)
        self.scenario(self.check_test_custom_stream_actual_peak_and_tile_repartition_preserve_selected_bytes)
        self.scenario(self.check_test_different_dimensions_actual_peak_stream_repartition_and_compact_only_transfer)
        self.scenario(self.check_test_rng_change_inside_draw_is_detected_by_predicted_stream)

    def test_validation_and_recovery(self):
        """8 related BayesImageSamplingTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_source_invalid_background_follows_consumed_foreground_draw)
        self.scenario(self.check_test_categorical_out_of_bounds_priority_and_zero_rows_fail_before_any_draw)
        self.scenario(self.check_test_cap_and_layout_failures_precede_compilation_or_random_draws)
        self.scenario(self.check_test_cancellation_after_each_actual_public_draw_preserves_partial_rng_state)
        self.scenario(self.check_test_callback_input_partial_output_and_rng_mutations_are_guarded_even_on_raise)
        self.scenario(self.check_test_independent_mask_cap_and_maximum_ignored_mask_preflight)
        self.scenario(self.check_test_final_callback_storage_swap_or_raised_rng_mutation_withholds_samples)
        self.scenario(self.check_test_directory_reservations_fail_before_current_draw_and_charge_full_backing)


if __name__ == '__main__':
    unittest.main()
