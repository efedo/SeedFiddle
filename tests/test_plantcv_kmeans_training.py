"""Ordered CUDA image training compared with pinned public PlantCV and sklearn."""

from tests.functional_units import functional_units
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

import numpy as np
import torch

from seedvision.cuda import plantcv_kmeans_training as native

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT/'artifacts/plantcv-reference-env/Scripts/python.exe'
SCRIPT = r'''
import hashlib,importlib,importlib.metadata,json,tempfile,sys
from pathlib import Path
import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from joblib import dump
from plantcv.plantcv import kmeans_classifier as classifier
train=importlib.import_module('plantcv.learn.train_kmeans')
assert importlib.metadata.version('plantcv')=='4.11.3'
assert importlib.metadata.version('scikit-learn')=='1.9.1'
assert np.__version__=='2.2.6'
assert hashlib.sha256(Path(train.__file__).read_bytes()).hexdigest()=='dbfb8a528e92984a20e73c7f265a41efa9fa088b42ed4f65addbe6ab40ee8b4d'
def digest(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
results=[]
for case in json.load(sys.stdin):
    images=[np.array(image,dtype=np.uint8) for image in case['images']]
    parts=[train.patch_extract(image,patch_size=case['patch_size'],sampling=case['sampling']) for image in images]
    features=np.vstack(parts)
    if case.get('public_train'):
        with tempfile.TemporaryDirectory(dir='artifacts') as directory:
            for i,image in enumerate(images): assert cv2.imwrite(str(Path(directory)/f'{i:03}.png'),image)
            fitted=train.train_kmeans(directory,case['k'],out_path=str(Path(directory)/'output.fit'),
                patch_size=case['patch_size'],sampling=case['sampling'],seed=case['seed'])
    else:
        fitted=MiniBatchKMeans(n_clusters=case['k'],random_state=case['seed'],**case['parameters']).fit(features)
    with tempfile.TemporaryDirectory(dir='artifacts') as directory:
        model_path=str(Path(directory)/'prediction.fit')
        dump(fitted,model_path)
        predicted=classifier.predict_kmeans(images[0],model_path,patch_size=case['patch_size'])
        masks=classifier.mask_kmeans(predicted,len(fitted.cluster_centers_))
    results.append(dict(features=digest(features),shape=list(features.shape),centres=digest(fitted.cluster_centers_),
        counts=digest(fitted._counts),labels=fitted.labels_.tolist(),inertia=float(fitted.inertia_),
        steps=fitted.n_steps_,iterations=fitted.n_iter_,threads=int(fitted._n_threads),
        predicted=digest(predicted),masks={key:digest(value) for key,value in masks.items()}))
print(json.dumps(results))
'''


def _sha(tensor):
    return sha256(tensor.cpu().numpy().tobytes()).hexdigest()


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
@functional_units
class KMeansImageTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(719)
        gray = [rng.integers(0, 17, shape, dtype=np.uint8)+base
            for shape, base in (((7, 8), 10), ((8, 9), 100), ((9, 7), 220))]
        bgr = [rng.integers(0, 11, shape, dtype=np.uint8)+np.array(base, dtype=np.uint8)
            for shape, base in (((8, 9, 3), (10, 20, 30)), ((9, 10, 3), (100, 120, 140)), ((10, 8, 3), (210, 220, 230)))]
        standard = dict(max_iter=1, batch_size=1024, n_init=2, max_no_improvement=None)
        cls.cases = [dict(images=[image.tolist() for image in gray], patch_size=1, sampling=None, k=3, seed=7, parameters=standard),
            dict(images=[image.tolist() for image in bgr], patch_size=2, sampling=7, k=3, seed=19, parameters=standard),
            dict(images=[gray[0].tolist(), gray[0].tolist()], patch_size=3, sampling=.37, k=1, seed=31, parameters=standard),
            dict(images=[image.tolist() for image in bgr], patch_size=4, sampling=None, k=3, seed=3, parameters=standard),
            dict(images=[np.zeros((70, 71, 3), np.uint8).tolist()]*2, patch_size=31, sampling=None, k=1, seed=7,
                parameters=dict(max_iter=1, batch_size=1024, n_init=1, max_no_improvement=None)),
            dict(images=[np.zeros((17, 18), np.uint8).tolist(), np.zeros((19, 20), np.uint8).tolist()],
                patch_size=10, sampling=None, k=2, seed=19, parameters={}, public_train=True),
            dict(images=[np.zeros((17, 18, 3), np.uint8).tolist()]*2,
                patch_size=10, sampling=7, k=2, seed=31, parameters={}, public_train=True)]
        completed = subprocess.run([str(REFERENCE), '-c', SCRIPT], cwd=ROOT,
            input=json.dumps(cls.cases), text=True, capture_output=True, timeout=120)
        if completed.returncode:
            raise AssertionError(completed.stderr)
        cls.reference = json.loads(completed.stdout)

    def inputs(self, index):
        return tuple(torch.tensor(image, dtype=torch.uint8, device='cuda') for image in self.cases[index]['images'])

    def parameters(self, index):
        case = self.cases[index]
        return dict(case['parameters'], patch_size=case['patch_size'], sampling=case['sampling'],
            seed=case['seed'], source_n_threads=self.reference[index]['threads'])

    def check_test_actual_public_patches_and_complete_fit_match_gray_bgr_and_source_defaults(self):
        for index, expected in enumerate(self.reference):
            with self.subTest(case=index):
                images = self.inputs(index)
                versions = tuple(image._version for image in images)
                result = native.fit_kmeans_training_images(images, self.cases[index]['k'], **self.parameters(index))
                self.assertEqual(_sha(result.training_rows.features), expected['features'])
                self.assertEqual(list(result.training_rows.features.shape), expected['shape'])
                self.assertEqual(_sha(result.fit.centres), expected['centres'])
                self.assertEqual(_sha(result.fit.counts), expected['counts'])
                self.assertEqual(result.fit.labels.cpu().tolist(), expected['labels'])
                self.assertLessEqual(result.fit.inertia.lo, expected['inertia'])
                self.assertGreaterEqual(result.fit.inertia.hi, expected['inertia'])
                self.assertEqual((result.fit.n_steps, result.fit.n_iter), (expected['steps'], expected['iterations']))
                self.assertEqual(versions, tuple(image._version for image in images))
                self.assertEqual(result.fit.source_n_threads, expected['threads'])

    def check_test_sampling_restarts_seed_one_per_image_and_repartition_preserves_cached_rows(self):
        images = self.inputs(2)
        first = native.assemble_kmeans_training_images(images, 3, sampling=.37)
        second = native.assemble_kmeans_training_images(images, 3, sampling=.37, tile_pixels=3, gaussian_tile_side=7)
        self.assertEqual(_sha(first.features), self.reference[2]['features'])
        self.assertEqual(_sha(second.features), self.reference[2]['features'])
        half = first.plan.rows//2
        self.assertTrue(torch.equal(first.features[:half].view(torch.int16), first.features[half:].view(torch.int16)))
        self.assertNotEqual(first.features.data_ptr(), second.features.data_ptr())
        for seed in (7, 31):
            result = native.fit_kmeans_training_images(images, 1, **dict(self.parameters(2), seed=seed))
            self.assertEqual(_sha(result.training_rows.features), self.reference[2]['features'])

    def check_test_trained_exported_numeric_models_predict_and_mask_like_public_joblib_models(self):
        from seedvision.cuda.plantcv_kmeans_export import export_kmeans_fit
        from seedvision.cuda.plantcv_kmeans_predict import predict_kmeans
        from seedvision.cuda.plantcv_kmeans import mask_kmeans
        for index, expected in enumerate(self.reference):
            with self.subTest(case=index):
                images = self.inputs(index)
                case = self.cases[index]
                result = native.fit_kmeans_training_images(images, case['k'], **self.parameters(index))
                exported = export_kmeans_fit(result.fit, case['patch_size'], result.plan.data.channels)
                labels = predict_kmeans(images[0], exported.model)
                self.assertEqual(_sha(labels), expected['predicted'])
                masks = mask_kmeans(labels, case['k'])
                self.assertEqual({key:_sha(value) for key,value in masks.items()}, expected['masks'])

    def check_test_actual_multimage_p31_custom_stream_peak_fits_combined_job_plan(self):
        images = self.inputs(4)
        parameters = self.parameters(4)
        plan = native.preflight_kmeans_image_fit(images, 1, **parameters)
        native.fit_kmeans_training_images(images, 1, **parameters)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        with torch.cuda.stream(stream):
            result = native.fit_kmeans_training_images(images, 1, **parameters)
        stream.synchronize()
        self.assertEqual(_sha(result.training_rows.features), self.reference[4]['features'])
        self.assertLessEqual(torch.cuda.max_memory_allocated()-baseline, plan.peak_live_bytes-plan.data.input_bytes)
        self.assertEqual(result.plan, plan)

    def check_test_combined_caps_and_bad_fit_parameters_precede_gaussian_rng_and_allocation(self):
        images = self.inputs(0)
        for name in ('MAX_IMAGES', 'MAX_DATASET_BYTES', 'MAX_COMBINED_WORK', 'MAX_CUDA_LIVE_BYTES'):
            with patch.object(native, name, 1), patch.object(torch, 'empty', side_effect=AssertionError('allocation before cap')), \
                    patch.object(native._full, 'extract_full_kmeans_training_patches', side_effect=AssertionError('Gaussian before cap')):
                with self.assertRaises(native.KMeansImageTrainingError):
                    native.fit_kmeans_training_images(images, 3, **self.parameters(0))
        with patch.object(native._fit, 'MAX_TOTAL_WORK', 1), patch.object(torch, 'empty', side_effect=AssertionError('allocation before fit cap')):
            with self.assertRaises(native._fit.KMeansFitError):
                native.fit_kmeans_training_images(images, 3, **self.parameters(0))
        for kwargs in (dict(n_init=0), dict(seed=None), dict(source_n_threads=0), dict(max_iter=0)):
            with self.assertRaises(native._fit.KMeansFitError), patch.object(torch, 'empty', side_effect=AssertionError('allocation before fit validation')):
                native.fit_kmeans_training_images(images, 3, **dict(self.parameters(0), **kwargs))
        for invalid in ([], [None], list(images)+[torch.zeros((7, 8, 3), dtype=torch.uint8, device='cuda')],
                [images[0].cpu()]):
            with self.assertRaises(ValueError):
                native.assemble_kmeans_training_images(invalid, 1)

    def check_test_cancellation_and_every_assembly_callback_mutation_withhold_all_rows(self):
        sources = [torch.zeros((3, 4), dtype=torch.uint8, device='cuda') for _ in range(2)]
        total = 0
        def count():
            nonlocal total
            total += 1
            return False
        native.assemble_kmeans_training_images(sources, 1, cancel=count)
        for target in range(1, total+1):
            images = [image.clone() for image in sources]
            calls = 0
            def mutate():
                nonlocal calls
                calls += 1
                if calls == target:
                    images[1].resize_(0)
                return False
            with self.assertRaisesRegex(native.KMeansImageTrainingError, 'changed'):
                native.assemble_kmeans_training_images(images, 1, cancel=mutate)
            self.assertEqual(calls, target)
        images = self.inputs(0)
        total = 0
        native.fit_kmeans_training_images(images, 3, **self.parameters(0), cancel=count)
        for target in (1, total//2, total):
            calls = 0
            def cancel():
                nonlocal calls
                calls += 1
                return calls == target
            with self.assertRaises(native.KMeansImageTrainingCancelled):
                native.fit_kmeans_training_images(images, 3, **self.parameters(0), cancel=cancel)
            self.assertEqual(calls, target)
        with self.assertRaisesRegex(native.KMeansImageTrainingError, 'bool'):
            native.assemble_kmeans_training_images(images, 1, cancel=lambda: 1)

    def check_test_wrong_patch_coverage_and_other_image_mutation_stop_before_copy(self):
        images = self.inputs(0)
        original = native._full.extract_full_kmeans_training_patches
        def mutated(*args, **kwargs):
            value = original(*args, **kwargs)
            images[1].resize_(0)
            return value
        with patch.object(native._full, 'extract_full_kmeans_training_patches', side_effect=mutated):
            with self.assertRaisesRegex(native.KMeansImageTrainingError, 'changed'):
                native.assemble_kmeans_training_images(images, 1)
        images = self.inputs(0)
        with patch.object(native._full, 'extract_full_kmeans_training_patches', return_value=torch.zeros((1, 1), dtype=torch.uint16, device='cuda')):
            with self.assertRaisesRegex(native.KMeansImageTrainingError, 'complete matrix'):
                native.assemble_kmeans_training_images(images, 1)

    def check_test_early_callback_mutation_precedes_stop_and_caller_exception(self):
        for action in ('continue', 'stop', 'raise'):
            for fit in (False, True):
                with self.subTest(action=action, fit=fit):
                    images = [torch.zeros((3, 4), dtype=torch.uint8, device='cuda') for _ in range(2)]
                    sentinel = RuntimeError('caller failure after mutation')
                    def callback():
                        images[1].add_(1)
                        if action == 'raise':
                            raise sentinel
                        return action == 'stop'
                    with self.assertRaisesRegex(native.KMeansImageTrainingError, 'changed'):
                        if fit:
                            native.fit_kmeans_training_images(images, 1, source_n_threads=1, cancel=callback)
                        else:
                            native.assemble_kmeans_training_images(images, 1, cancel=callback)
        for sentinel in (RuntimeError('unchanged caller failure'), KeyboardInterrupt('caller interruption')):
            def unchanged():
                raise sentinel
            with self.assertRaises(type(sentinel)) as caught:
                native.assemble_kmeans_training_images(images, 1, cancel=unchanged)
            self.assertIs(caught.exception, sentinel)
        with self.assertRaises(native.KMeansImageTrainingCancelled):
            native.assemble_kmeans_training_images(images, 1, cancel=lambda: True)

    def check_test_real_nested_patch_callback_preserves_typed_mutation_and_unrelated_exception(self):
        from seedvision.cuda.tensor_callback import TensorCallbackMutation
        for sampling in (None, .5):
            producer = native._full if sampling is None else native._sampled
            original = producer._preprocess.gaussian_patch_raster
            for action in ('continue', 'stop', 'raise', 'unchanged_raise', 'unchanged_stop'):
                with self.subTest(sampling=sampling, action=action):
                    images = [torch.zeros((3, 4), dtype=torch.uint8, device='cuda') for _ in range(2)]
                    extracting = False
                    called = False
                    sentinel = RuntimeError('nested caller failure')
                    def extraction(*args, **kwargs):
                        nonlocal extracting
                        extracting = True
                        try:
                            return original(*args, **kwargs)
                        finally:
                            extracting = False
                    def callback():
                        nonlocal called
                        if not extracting or called:
                            return False
                        called = True
                        if not action.startswith('unchanged'):
                            images[0].add_(1)
                        if action.endswith('raise'):
                            raise sentinel
                        return action.endswith('stop')
                    expected = (RuntimeError if action == 'unchanged_raise' else
                                native.KMeansImageTrainingCancelled if action == 'unchanged_stop' else
                                native.KMeansImageTrainingError)
                    with patch.object(producer._preprocess, 'gaussian_patch_raster', side_effect=extraction), self.assertRaises(expected) as caught:
                        native.assemble_kmeans_training_images(images, 1, sampling=sampling, cancel=callback)
                    self.assertTrue(called, 'Must enter the actual patch producer before triggering the callback')
                    if action == 'unchanged_raise':
                        self.assertIs(caught.exception, sentinel)
                    elif not action.startswith('unchanged'):
                        self.assertIn('changed', str(caught.exception))
                        self.assertIsInstance(caught.exception.__cause__, TensorCallbackMutation)

    def check_test_plain_sampled_errors_preserve_identity_and_unrelated_cause(self):
        for cause in (None, RuntimeError('ordinary sampler callback failure')):
            with self.subTest(cause=type(cause).__name__):
                images = [torch.zeros((3, 4), dtype=torch.uint8, device='cuda') for _ in range(2)]
                failure = native._sampled.KMeansSampledDataError('ordinary sampled request failure')
                def extraction(*args, **kwargs):
                    raise failure from cause
                with patch.object(native._sampled, 'extract_sampled_kmeans_training_patches', side_effect=extraction):
                    with self.assertRaises(native._sampled.KMeansSampledDataError) as caught:
                        native.assemble_kmeans_training_images(images, 1, sampling=.5)
                self.assertIs(caught.exception, failure)
                self.assertIs(caught.exception.__cause__, cause)

    def check_test_loaded_source_binds_patch_seed_and_combined_limits(self):
        from seedvision.benchmarking.native_recipe import _module_record
        self.assertTrue(_module_record(native)['loaded_code'])
        for name, value in (('PATCH_SEED', 7), ('MAX_COMBINED_WORK', 1), ('PLANTCV_TRAIN_SHA256', 'changed')):
            with patch.object(native, name, value), self.assertRaises(ValueError):
                _module_record(native)

    def check_test_duplicate_occurrences_share_input_storage_but_repeat_all_patch_rows(self):
        image = self.inputs(2)[0]
        result = native.assemble_kmeans_training_images((image, image), 3, sampling=.37)
        self.assertEqual(result.plan.input_bytes, image.untyped_storage().nbytes())
        self.assertEqual(_sha(result.features), self.reference[2]['features'])
        self.assertEqual(result.plan.rows, 2*result.plan.image_plans[0].patch_count)

    def test_workflow_lifecycle(self):
        """7 related KMeansImageTrainingTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_actual_public_patches_and_complete_fit_match_gray_bgr_and_source_defaults)
        self.scenario(self.check_test_sampling_restarts_seed_one_per_image_and_repartition_preserves_cached_rows)
        self.scenario(self.check_test_trained_exported_numeric_models_predict_and_mask_like_public_joblib_models)
        self.scenario(self.check_test_actual_multimage_p31_custom_stream_peak_fits_combined_job_plan)
        self.scenario(self.check_test_combined_caps_and_bad_fit_parameters_precede_gaussian_rng_and_allocation)
        self.scenario(self.check_test_plain_sampled_errors_preserve_identity_and_unrelated_cause)
        self.scenario(self.check_test_duplicate_occurrences_share_input_storage_but_repeat_all_patch_rows)

    def test_validation_and_recovery(self):
        """5 related KMeansImageTrainingTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_cancellation_and_every_assembly_callback_mutation_withhold_all_rows)
        self.scenario(self.check_test_wrong_patch_coverage_and_other_image_mutation_stop_before_copy)
        self.scenario(self.check_test_early_callback_mutation_precedes_stop_and_caller_exception)
        self.scenario(self.check_test_real_nested_patch_callback_preserves_typed_mutation_and_unrelated_exception)
        self.scenario(self.check_test_loaded_source_binds_patch_seed_and_combined_limits)
