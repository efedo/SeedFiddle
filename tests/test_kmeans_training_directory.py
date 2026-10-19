"""Directory selection, bounded decoding and CUDA fits against public PlantCV."""

from tests.functional_units import functional_units
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import random
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch

from seedvision.benchmarking import kmeans_training_directory as native

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT/'artifacts/plantcv-reference-env/Scripts/python.exe'
SCRIPT = r'''
import hashlib,importlib,importlib.metadata,json,random,sys,tempfile
from pathlib import Path
import numpy as np
train=importlib.import_module('plantcv.learn.train_kmeans')
assert importlib.metadata.version('plantcv')=='4.11.3'
assert importlib.metadata.version('scikit-learn')=='1.9.1'
assert np.__version__=='2.2.6'
assert hashlib.sha256(Path(train.__file__).read_bytes()).hexdigest()=='dbfb8a528e92984a20e73c7f265a41efa9fa088b42ed4f65addbe6ab40ee8b4d'
def digest(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
answers=[]
original_extract,original_read,original_choices=train.patch_extract,train.cv2.imread,random.choices
for case in json.load(sys.stdin):
    parts,reads,selected=[],[],[]
    def extract(*args,**kwargs):
        result=original_extract(*args,**kwargs)
        parts.append(result)
        return result
    def read(path,*args,**kwargs):
        result=original_read(path,*args,**kwargs)
        reads.append(dict(name=Path(path).name,shape=None if result is None else list(result.shape),
                          pixels=None if result is None else digest(result)))
        return result
    def choices(population,**kwargs):
        result=original_choices(population,**kwargs)
        selected.extend(result)
        return result
    train.patch_extract,train.cv2.imread,random.choices=extract,read,choices
    random.seed(case['random_seed'])
    answer={}
    try:
        with tempfile.TemporaryDirectory(dir='artifacts') as output:
            fit=train.train_kmeans(case['directory'],2,out_path=str(Path(output)/'model.fit'),
                prefix=case['prefix'],num_imgs=case['num_imgs'],patch_size=2,sampling=5,seed=19)
        answer.update(features=digest(np.vstack(parts)),centres=digest(fit.cluster_centers_),
            counts=digest(fit._counts),labels=fit.labels_.tolist(),inertia=float(fit.inertia_),
            steps=fit.n_steps_,iterations=fit.n_iter_,threads=int(fit._n_threads))
    except Exception as error:
        answer['error']=type(error).__name__
    answer.update(selected=selected or sorted(p.name for p in Path(case['directory']).iterdir()),
        reads=reads,random_state=hashlib.sha256(repr(random.getstate()).encode()).hexdigest())
    answers.append(answer)
    train.patch_extract,train.cv2.imread,random.choices=original_extract,original_read,original_choices
print(json.dumps(answers))
'''


def _tensor_sha(tensor):
    return sha256(tensor.cpu().numpy().tobytes()).hexdigest()


@functional_units
class KMeansTrainingDirectoryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT/'artifacts')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def image(self, name='a.png', shape=(17, 18), value=0, dtype=np.uint8):
        image = np.full(shape, value, dtype=dtype)
        self.assertTrue(cv2.imwrite(str(self.root/name), image))
        return image

    def manifest(self):
        self.image()
        (self.root/'z-invalid.txt').write_bytes(b'not an image')
        return native.resolve_kmeans_training_directory(self.root)


    def check_test_regular_decode_skips_are_hashed_and_nonfiles_are_preserved(self):
        self.image()
        (self.root/'b-empty').write_bytes(b'')
        (self.root/'c-invalid').write_bytes(b'invalid')
        (self.root/'d-directory').mkdir()
        manifest = native.resolve_kmeans_training_directory(self.root)
        self.assertEqual([item.status for item in manifest.occurrences],
            ['accepted', 'decode_skipped', 'decode_skipped', 'non_file'])
        self.assertEqual(manifest.occurrences[1].source_sha256, sha256(b'').hexdigest())
        self.assertEqual(manifest.occurrences[2].source_sha256, sha256(b'invalid').hexdigest())
        self.assertEqual(manifest.source_bytes, sum((self.root/item.name).stat().st_size
            for item in manifest.occurrences if item.status != 'non_file'))
        self.assertEqual(manifest.decoded_bytes, 4*17*18)
        self.assertIs(native.validate_kmeans_training_directory(manifest), manifest)

    def check_test_prefix_is_substring_and_skipped_files_are_never_read(self):
        self.image('a-middle-keep.png')
        (self.root/'b-drop').write_bytes(b'anything')
        original = native._read_source
        def read(path):
            self.assertIn('keep', path.name)
            return original(path)
        with patch.object(native, '_read_source', side_effect=read):
            manifest = native.resolve_kmeans_training_directory(self.root, prefix='keep')
            native.validate_kmeans_training_directory(manifest)
        self.assertEqual(manifest.occurrences[1].status, 'prefix_skipped')

    def check_test_qt_header_only_and_opencv_unchanged_pixels_for_supported_rasters(self):
        original = np.arange(9*11*3, dtype=np.uint8).reshape(9, 11, 3)
        for extension in ('.png', '.jpg', '.bmp', '.tif', '.webp'):
            with self.subTest(extension=extension):
                encoded, raw = cv2.imencode(extension, original)
                self.assertTrue(encoded)
                data = raw.tobytes()
                with patch.object(native.QImageReader, 'read', side_effect=AssertionError('Qt must not decode pixels')):
                    header = native._header(data)
                self.assertEqual(header, (9, 11, 4*9*11))
                self.assertTrue(np.array_equal(native._decode(data, header),
                    cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)))

    def check_test_depth_alpha_and_header_pixel_limits_reject_before_unsupported_analysis(self):
        self.image(shape=(17, 18), dtype=np.uint16, value=1024)
        with patch.object(native.cv2, 'imdecode', side_effect=AssertionError('uint16 pixels allocated')):
            with self.assertRaisesRegex(native.KMeansDirectoryError, 'eight-bit'):
                native.resolve_kmeans_training_directory(self.root)
        self.image(shape=(17, 18, 4), value=50)
        with self.assertRaisesRegex(native.KMeansDirectoryError, 'gray or BGR'):
            native.resolve_kmeans_training_directory(self.root)
        self.image()
        with patch.object(native, 'MAX_IMAGE_PIXELS', 17*18-1), \
                patch.object(native.cv2, 'imdecode', side_effect=AssertionError('pixels before header cap')):
            with self.assertRaisesRegex(native.KMeansDirectoryError, 'pixel bound'):
                native.resolve_kmeans_training_directory(self.root)

    def check_test_source_decoded_and_selection_caps_precede_relevant_reads_decodes_and_rng(self):
        self.image()
        self.image('b.png')
        size = (self.root/'a.png').stat().st_size
        for constant, limit, message in (('MAX_FILE_BYTES', size-1, 'per-file'),
                ('MAX_SOURCE_BYTES', size-1, 'aggregate'), ('MAX_DECODED_BYTES', 4*17*18-1, 'header bound')):
            with patch.object(native, constant, limit), \
                    patch.object(native.cv2, 'imdecode', side_effect=AssertionError('decode before cap')):
                with self.assertRaisesRegex(native.KMeansDirectoryError, message):
                    native.resolve_kmeans_training_directory(self.root)
        for constant in ('MAX_DIRECTORY_ENTRIES', 'MAX_OCCURRENCES'):
            with patch.object(native, constant, 1), \
                    patch.object(native.random, 'choices', side_effect=AssertionError('RNG before count cap')):
                with self.assertRaises(native.KMeansDirectoryError):
                    native.resolve_kmeans_training_directory(self.root)
        with patch.object(native.random, 'choices', side_effect=AssertionError('invalid count consumed RNG')):
            for count in (-1, 257, True, 1.5):
                with self.assertRaises(native.KMeansDirectoryError):
                    native.resolve_kmeans_training_directory(self.root, num_imgs=count)

    def check_test_unbounded_source_decoder_format_is_explicitly_unavailable(self):
        self.image()
        with patch.object(native, '_header', return_value=None), patch.object(native.cv2, 'haveImageReader', return_value=True):
            with self.assertRaisesRegex(native.KMeansDirectoryError, 'bounded Qt raster header'):
                native.resolve_kmeans_training_directory(self.root)

    def check_test_changes_to_accepted_skipped_nonfile_or_listing_prevent_all_uploads(self):
        manifest = self.manifest()
        original = (self.root/'a.png').read_bytes()
        for name in ('a.png', 'z-invalid.txt'):
            with self.subTest(name=name):
                before = (self.root/name).read_bytes()
                (self.root/name).write_bytes(before+b'changed')
                with patch.object(torch, 'from_numpy', side_effect=AssertionError('upload before full verification')):
                    with self.assertRaisesRegex(native.KMeansDirectoryError, 'bytes changed'):
                        native.materialize_kmeans_training_directory(manifest)
                (self.root/name).write_bytes(before)
        self.assertEqual((self.root/'a.png').read_bytes(), original)
        (self.root/'new-entry').write_bytes(b'')
        with self.assertRaisesRegex(native.KMeansDirectoryError, 'listing changed'):
            native.materialize_kmeans_training_directory(manifest)
        (self.root/'new-entry').unlink()
        (self.root/'zz-dir').mkdir()
        manifest = native.resolve_kmeans_training_directory(self.root)
        (self.root/'zz-dir').rmdir()
        (self.root/'zz-dir').write_bytes(b'file now')
        with self.assertRaisesRegex(native.KMeansDirectoryError, 'non-file'):
            native.materialize_kmeans_training_directory(manifest)

    def check_test_forged_manifest_fields_and_aggregates_are_rejected_before_file_reads(self):
        manifest = self.manifest()
        variants = [replace(manifest, manifest_sha256='changed'), replace(manifest, source_bytes=0),
            replace(manifest, decoded_bytes=1), replace(manifest, num_imgs=True),
            replace(manifest, occurrences=(replace(manifest.occurrences[0], shape=(17, 18, 4)),)+manifest.occurrences[1:]),
            replace(manifest, occurrences=(replace(manifest.occurrences[0], source_sha256='x'*64),)+manifest.occurrences[1:]),
            replace(manifest, selected_names=('../escape',)+manifest.selected_names[1:])]
        for value in variants:
            with patch.object(native, '_read_source', side_effect=AssertionError('read before manifest check')):
                with self.assertRaises(native.KMeansDirectoryError):
                    native.validate_kmeans_training_directory(value)

    def check_test_cancellation_bool_contract_and_final_callback_mutation_withhold_manifest(self):
        self.image()
        with self.assertRaises(native.KMeansDirectoryCancelled):
            native.resolve_kmeans_training_directory(self.root, cancel=lambda: True)
        with self.assertRaisesRegex(native.KMeansDirectoryError, 'bool'):
            native.resolve_kmeans_training_directory(self.root, cancel=lambda: 1)
        manifest = native.resolve_kmeans_training_directory(self.root)
        calls = 0
        def count():
            nonlocal calls
            calls += 1
            return False
        native.validate_kmeans_training_directory(manifest, cancel=count)
        last, calls = calls, 0
        def mutate():
            nonlocal calls
            calls += 1
            if calls == last:
                (self.root/'a.png').write_bytes(b'changed by final callback')
            return False
        with self.assertRaisesRegex(native.KMeansDirectoryError, 'bytes changed'):
            native.validate_kmeans_training_directory(manifest, cancel=mutate)

    def check_test_loaded_source_binds_source_pin_and_decoder_resource_contract(self):
        from seedvision.benchmarking.native_recipe import _module_record
        self.assertTrue(_module_record(native)['loaded_code'])
        for name, value in (('MAX_IMAGE_PIXELS', 1), ('DECODED_BYTES_PER_PIXEL', 8),
                ('PLANTCV_TRAIN_SHA256', 'changed'), ('BYTE_FORMATS', ())):
            with patch.object(native, name, value), self.assertRaises(ValueError):
                _module_record(native)

    def check_test_duplicate_occurrences_upload_once_and_preserve_bgr_channel_order(self):
        original = self.image(shape=(17, 18, 3), value=(10, 20, 30))
        with patch.object(native.random, 'choices', return_value=['a.png']*4):
            manifest = native.resolve_kmeans_training_directory(self.root, num_imgs=4)
        with patch.object(torch, 'from_numpy', wraps=torch.from_numpy) as upload:
            materialized = native.materialize_kmeans_training_directory(manifest)
        self.assertEqual(upload.call_count, 1)
        self.assertTrue(all(image is materialized.images[0] for image in materialized.images))
        self.assertTrue(np.array_equal(materialized.images[0].cpu().numpy(), original))
        rows = native._training.assemble_kmeans_training_images(materialized.images, 2, sampling=5)
        self.assertEqual(rows.plan.input_bytes, original.nbytes)
        self.assertEqual(rows.plan.rows, 20)
        self.assertTrue(torch.equal(rows.features[:5].view(torch.int16), rows.features[15:].view(torch.int16)))

    def check_test_custom_stream_fit_is_bounded_and_changes_during_fit_withhold_result(self):
        self.image(shape=(17, 18, 3))
        manifest = native.resolve_kmeans_training_directory(self.root)
        parameters = dict(source_n_threads=1, patch_size=2, sampling=5, n_init=1,
            max_iter=1, max_no_improvement=None)
        native.fit_kmeans_training_directory(manifest, 1, **parameters)
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            fitted = native.fit_kmeans_training_directory(manifest, 1, **parameters)
        stream.synchronize()
        self.assertEqual(fitted.training.training_rows.features.device.type, 'cuda')
        self.assertLessEqual(torch.cuda.max_memory_allocated()-baseline, fitted.training.plan.peak_live_bytes)
        original = native._training.fit_kmeans_training_images
        def mutate(*args, **kwargs):
            result = original(*args, **kwargs)
            (self.root/'a.png').write_bytes(b'changed during fit')
            return result
        with patch.object(native._training, 'fit_kmeans_training_images', side_effect=mutate):
            with self.assertRaisesRegex(native.KMeansDirectoryError, 'bytes changed'):
                native.fit_kmeans_training_directory(manifest, 1, **parameters)


    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_duplicate_occurrences_upload_once_and_preserve_bgr_channel_order(self):
        self.scenario(self.check_test_duplicate_occurrences_upload_once_and_preserve_bgr_channel_order, labelled=False)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_custom_stream_fit_is_bounded_and_changes_during_fit_withhold_result(self):
        self.scenario(self.check_test_custom_stream_fit_is_bounded_and_changes_during_fit_withhold_result, labelled=False)

    def test_workflow_lifecycle(self):
        """6 related KMeansTrainingDirectoryTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_regular_decode_skips_are_hashed_and_nonfiles_are_preserved)
        self.scenario(self.check_test_prefix_is_substring_and_skipped_files_are_never_read)
        self.scenario(self.check_test_qt_header_only_and_opencv_unchanged_pixels_for_supported_rasters)
        self.scenario(self.check_test_source_decoded_and_selection_caps_precede_relevant_reads_decodes_and_rng)
        self.scenario(self.check_test_changes_to_accepted_skipped_nonfile_or_listing_prevent_all_uploads)
        self.scenario(self.check_test_loaded_source_binds_source_pin_and_decoder_resource_contract)

    def test_validation_and_recovery(self):
        """4 related KMeansTrainingDirectoryTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_depth_alpha_and_header_pixel_limits_reject_before_unsupported_analysis)
        self.scenario(self.check_test_unbounded_source_decoder_format_is_explicitly_unavailable)
        self.scenario(self.check_test_forged_manifest_fields_and_aggregates_are_rejected_before_file_reads)
        self.scenario(self.check_test_cancellation_bool_contract_and_final_callback_mutation_withhold_manifest)
