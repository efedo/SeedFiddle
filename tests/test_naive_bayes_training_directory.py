"""Real decoded directories against the pinned public binary trainer."""
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

    def test_actual_public_directory_constant_value_profiles_pdfs_and_rng(self):
        from seedvision.model_io import naive_bayes_provenance as records
        image = np.random.default_rng(24).integers(0,255,(10,12,3),dtype=np.uint8)
        image[:,:,0] = 255
        mask = np.zeros((10,12),dtype=np.uint8)
        mask.reshape(-1)[:60] = 255
        self.put('first.png',image,mask)
        expected = self.oracle()
        self.assertNotIn('error',expected)
        self.rng = np.random.RandomState(17)
        result = self.run_native(rng=self.rng)
        self.assert_oracle(expected,result)
        metadata = records.parse_training_record(result.provenance_json,result.model).metadata
        self.assertEqual(metadata['sample_counts'],[6,6])
        for channels in metadata['channel_fits']:
            self.assertEqual(channels[2]['constant_value'],255)
            self.assertGreater(channels[2]['mean'],255)
            self.assertEqual(channels[2]['numerical_profile'],records.CONSTANT_NUMERICAL_PROFILE)

    def test_actual_public_nested_unsorted_duplicate_names_flat_masks_and_provenance(self):
        self.put('a/same.png', *pair(seed=31))
        self.put('z/same.png', *pair(seed=32))
        self.walk = [('.', ['z','a'], ['first.png']), ('z', [], ['same.png']), ('a', [], ['same.png'])]
        expected = self.oracle(cached_gaussian=True)
        self.rng = np.random.RandomState(17)
        self.rng.normal()
        result = self.run_native(rng=self.rng, conversion_tile=31, gather_tile=13)
        self.assert_oracle(expected, result)
        self.assertEqual([p.image_relative_path for p in result.selection.pairs],
            ['first.png','z/same.png','a/same.png'])
        record = json.loads(result.provenance_json)
        self.assertEqual(record['kind'], 'native_bayes_directory_training')
        self.assertFalse(record['scientific_validation'])
        self.assertEqual(record['rng_before']['keys'], expected['before']['keys'])
        self.assertEqual(record['rng_after']['sha256'], sampling._state(self.rng).sha256)
        self.assertEqual(record['model_sha256'], sha256(result.tsv_bytes).hexdigest())
        self.assertEqual(record['tables_sha256'], result.model.tables_sha256)
        self.assertEqual(record['sample_counts'], [165,165])
        self.assertEqual(len(record['occurrences']), 3)
        self.assertEqual(record['occurrences'][1]['mask'], record['occurrences'][2]['mask'])
        self.assertEqual(record['sampling_work'], sum(o['sampling_plan']['work'] for o in record['occurrences']))
        self.assertIn('seedvision.cuda.plantcv_naive_bayes_sampling', record['calculation'])
        self.assertIn(native.__name__, record['calculation'])
        # Generated TSV remains an inference input after the source roots vanish.
        self.temp.cleanup()
        model = parse_plantcv_pdf(result.tsv_bytes)
        pixels = torch.tensor([[[5,10,15],[100,150,200]]], dtype=torch.uint8, device='cuda')
        classified = naive_bayes_masks(pixels, model)
        self.assertIsNotNone(classified)

    def test_actual_suffix_missing_masks_byte_flags_and_16bit_source_conversion(self):
        image, mask = pair(seed=55)
        self.put('oddnamepng', image, mask)
        self.put('second.jpg', *pair(seed=56), encoded='.jpg')
        self.put('excluded.jpeg', image, mask)
        self.put('excluded.PNG', image, mask)
        self.put('missing.png', image, mask)
        (self.masks/'missing.png').unlink()
        self.put('highbit.png', image.astype(np.uint16)*257, mask.astype(np.uint16)*257)
        self.walk = [('.', [], ['excluded.jpeg','missing.png','oddnamepng','excluded.PNG','highbit.png','second.jpg','first.png'])]
        expected = self.oracle()
        self.rng = np.random.RandomState(17)
        result = self.run_native(rng=self.rng)
        self.assert_oracle(expected, result)
        self.assertEqual([p.image_relative_path for p in result.selection.pairs],
            ['oddnamepng','highbit.png','second.jpg','first.png'])
        self.assertEqual(result.selection.skipped_missing_masks, ('missing.png',))

    def test_late_undecodable_source_preserves_earlier_actual_draws(self):
        (self.images/'bad.png').write_bytes(b'undecodable')
        (self.masks/'bad.png').write_bytes((self.masks/'first.png').read_bytes())
        self.walk[0][2].append('bad.png')
        expected = self.oracle()
        self.assertIn('error', expected)
        self.assertEqual(len(expected['steps']), 6)
        rng = np.random.RandomState(17)
        with self.assertRaises(native.BayesDirectorySourceError):
            self.run_native(rng=rng)
        self.assertEqual(state_record(sampling._state(rng)), expected['after'])

    def test_zero_rows_keep_source_failure_and_smaller_masks_fit_actual_public_pdfs(self):
        self.put('first.png', *pair(3,3,4,5))
        expected = self.oracle()
        self.assertIn('error', expected)
        self.assertEqual(len(expected['steps']), 6)
        rng = np.random.RandomState(17)
        with self.assertRaisesRegex(native._fit.BayesTrainingError, 'at least two'):
            self.run_native(rng=rng)
        self.assertEqual(state_record(sampling._state(rng)), expected['after'])
        image, mask = pair(31,37,555)
        self.put('first.png', image, mask[:20,:20])
        expected = self.oracle()
        self.assertNotIn('error', expected)
        self.rng = np.random.RandomState(17)
        result = self.run_native(rng=self.rng)
        self.assert_oracle(expected,result)

    def test_actual_decoded_different_dimensions_masks_match_public_pdfs_and_typed_facts(self):
        from seedvision.model_io import naive_bayes_provenance as records
        for image_shape,mask_shape in (((31,37),(20,19)),((20,19),(31,37)),
                ((31,37),(20,43)),((20,43),(31,37))):
            self.put('first.png',*coordinate_pair(image_shape,mask_shape))
            expected = self.oracle()
            self.assertNotIn('error',expected)
            self.rng = np.random.RandomState(17)
            result = self.run_native(rng=self.rng,conversion_tile=31,gather_tile=7)
            with self.subTest(image=image_shape,mask=mask_shape):
                self.assert_oracle(expected,result)
                metadata = records.parse_training_record(result.provenance_json,result.model).metadata
                self.assertEqual(metadata['parser_id'],records.COORDINATE_DIRECTORY_PARSER_ID)
                occurrence = metadata['occurrences'][0]
                self.assertEqual(occurrence['image_shape'],[*image_shape,3])
                self.assertEqual(occurrence['mask_shape'],list(mask_shape))
                self.assertEqual(occurrence['sampling_plan']['mask_width'],mask_shape[1])

    def test_first_and_late_categorical_out_of_bounds_preserve_actual_draw_order(self):
        image,mask = coordinate_pair((31,37),(33,39))
        for kind in ('foreground','background'):
            bad = mask.copy()
            bad[32,0] = 255 if kind=='foreground' else 0
            self.put('bad.png',image,bad)
            for names in (['bad.png'],['first.png','bad.png']):
                self.walk = [('.',[],names)]
                expected = self.oracle()
                self.assertEqual(expected['error'],'IndexError')
                self.assertEqual(len(expected['steps']),6*(len(names)-1))
                rng = np.random.RandomState(17)
                with (self.subTest(kind=kind,names=names),
                        self.assertRaisesRegex(sampling.BayesSamplingSourceError,'hue '+kind)):
                    self.run_native(rng=rng)
                self.assertEqual(state_record(sampling._state(rng)),expected['after'])

    def test_aggregate_work_rows_and_retained_backing_fail_before_second_draw(self):
        self.put('second.png', *pair(seed=71))
        self.walk[0][2].append('second.png')
        expected = self.oracle()
        self.assertEqual(len(expected['steps']), 12)
        image, mask = pair()
        sample = sampling.sample_naive_bayes_image(torch.tensor(image, device='cuda'),
            torch.tensor(mask, device='cuda'), rng=np.random.RandomState(17))
        for name, maximum in (('MAX_WORK', 2*sample.plan.work-1),
                ('MAX_SAMPLES', 4*sample.plan.samples_per_class-1),
                ('MAX_CUDA_LIVE_BYTES', sample.plan.peak_live_bytes+sample.plan.output_bytes-1)):
            rng = np.random.RandomState(17)
            # Isolate resource accounting from the separate loaded-code guard,
            # which correctly rejects patched production constants otherwise.
            with (self.subTest(cap=name), patch.object(sampling, name, maximum),
                    patch.object(native, '_calculation_records', return_value={})):
                with self.assertRaises(sampling.BayesSamplingError):
                    self.run_native(rng=rng)
            self.assertEqual(state_record(sampling._state(rng)), expected['steps'][5])

    def test_fit_cap_failure_precedes_concatenation_and_preserves_sampling(self):
        expected = self.oracle()
        rng = np.random.RandomState(17)
        with (patch.object(native._fit, 'MAX_CUDA_LIVE_BYTES', 1),
                patch.object(native.torch, 'cat', side_effect=AssertionError('concatenated before fit preflight')),
                patch.object(native, '_calculation_records', return_value={})):
            with self.assertRaisesRegex(native._fit.BayesTrainingError, 'concatenation'):
                self.run_native(rng=rng)
        self.assertEqual(state_record(sampling._state(rng)), expected['after'])

    def test_caps_precede_decode_and_upload_and_invalid_arguments_consume_no_rng(self):
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

    def test_final_callback_source_and_new_mask_changes_withhold_model(self):
        calls = 0
        def count():
            nonlocal calls
            calls += 1
            return False
        self.run_native(cancel=count)
        final = calls
        for kind in ('image','mask','added_mask','walk_order'):
            if kind == 'added_mask':
                self.put('missing.png', *pair(seed=81))
                (self.masks/'missing.png').unlink()
                self.walk[0][2].append('missing.png')
            calls = 0
            saved_image = (self.images/'first.png').read_bytes()
            saved_mask = (self.masks/'first.png').read_bytes()
            def mutate():
                nonlocal calls
                calls += 1
                if calls == final:
                    if kind == 'image':
                        (self.images/'first.png').write_bytes(saved_image+b'changed')
                    elif kind == 'mask':
                        (self.masks/'first.png').write_bytes(saved_mask+b'changed')
                    elif kind == 'added_mask':
                        (self.masks/'missing.png').write_bytes(saved_mask)
                    else:
                        self.walk[0][2].append('unselected.txt')
                return False
            with self.subTest(kind=kind), self.assertRaisesRegex(native.BayesDirectoryError, 'changed'):
                self.run_native(cancel=mutate)
            (self.images/'first.png').write_bytes(saved_image)
            (self.masks/'first.png').write_bytes(saved_mask)
            if kind == 'added_mask':
                (self.masks/'missing.png').unlink()

    def test_retained_samples_rng_and_last_pdf_mutations_even_raised_callbacks(self):
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

    def test_implicit_global_rng_and_cancel_after_source_draw_preserve_actual_state(self):
        expected = self.oracle(cached_gaussian=True)
        original = np.random.get_state()
        try:
            np.random.seed(17)
            np.random.normal()
            walk = [(str(self.images/directory), dirs, files) for directory, dirs, files in self.walk]
            with patch.object(native.os, 'walk', side_effect=lambda *a, **k:iter(walk)):
                result = native.train_bayes_directories(self.images, self.masks)
            self.assertEqual(state_record(sampling._state(np.random)), expected['after'])
            self.assertEqual(json.loads(result.provenance_json)['rng_before']['keys'], expected['before']['keys'])
        finally:
            np.random.set_state(original)
        rng = np.random.RandomState(17)
        rng.normal()
        target = expected['steps'][2]
        with self.assertRaises(sampling.BayesSamplingCancelled):
            self.run_native(rng=rng, cancel=lambda:state_record(sampling._state(rng)) == target)
        self.assertEqual(state_record(sampling._state(rng)), target)
        for answer in (None,0,[],1):
            with self.subTest(answer=answer), self.assertRaisesRegex(sampling.BayesSamplingError, 'return bool'):
                self.run_native(cancel=lambda:answer)

    def test_custom_stream_actual_peak_includes_retained_chunks_and_fit_concatenation(self):
        warm = self.run_native()
        del warm
        self.put('first.png', *pair(500,500,125_000))
        self.put('second.png', *pair(500,500,125_000,seed=99))
        self.walk[0][2].append('second.png')
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            result = self.run_native(conversion_tile=65_536, gather_tile=1021)
        stream.synchronize()
        record = json.loads(result.provenance_json)
        retained, peak = 0, record['fit_preflight']['peak_live_bytes']
        for occurrence in record['occurrences']:
            plan = occurrence['sampling_plan']
            peak = max(peak, retained+plan['peak_live_bytes'])
            retained += plan['output_bytes']
        self.assertLessEqual(torch.cuda.max_memory_allocated()-baseline, peak)
        self.assertEqual(record['sample_counts'], [25_000,25_000])

    def test_exact_walk_record_bound_is_independent_of_aggregate_entry_count(self):
        walk = [(str(self.images/str(i)), [], []) for i in range(native.MAX_DIRECTORY_ENTRIES+1)]
        with patch.object(native.os, 'walk', side_effect=lambda *a, **k:iter(walk[:-1])):
            selected = native.resolve_bayes_training_directories(self.images, self.masks)
        self.assertEqual(len(selected.walk), native.MAX_DIRECTORY_ENTRIES)
        with patch.object(native.os, 'walk', side_effect=lambda *a, **k:iter(walk)):
            with self.assertRaisesRegex(native.BayesDirectoryError, 'entry-count'):
                native.resolve_bayes_training_directories(self.images, self.masks)


if __name__ == '__main__':
    unittest.main()
