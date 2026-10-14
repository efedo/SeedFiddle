from dataclasses import replace
import json
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

import numpy as np
import torch

from seedvision.cuda import plantcv_kmeans_classifier as scoring
from seedvision.cuda import plantcv_kmeans_features as features
from seedvision.cuda import plantcv_kmeans_model as models
from seedvision.cuda import plantcv_kmeans_predict as native
from seedvision.cuda import plantcv_kmeans_preprocess as preprocess

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / 'artifacts/plantcv-reference-env/Scripts/python.exe'
SCRIPT = r'''
import hashlib,importlib,importlib.metadata,json,sys,tempfile
from pathlib import Path
import cv2,numpy as np,joblib
from sklearn.cluster import MiniBatchKMeans
import sklearn.cluster._kmeans as sklearn_source
from plantcv.plantcv import kmeans_classifier,params
train=importlib.import_module('plantcv.learn.train_kmeans')
assert {name:importlib.metadata.version(name) for name in ('plantcv','numpy','scipy','scikit-image','scikit-learn','opencv-python')}=={
    'plantcv':'4.11.3','numpy':'2.2.6','scipy':'1.15.3','scikit-image':'0.25.2','scikit-learn':'1.9.1','opencv-python':'4.11.0.86'}
assert hashlib.sha256(Path(train.__file__).read_bytes()).hexdigest()=='dbfb8a528e92984a20e73c7f265a41efa9fa088b42ed4f65addbe6ab40ee8b4d'
assert hashlib.sha256(Path(kmeans_classifier.__file__).read_bytes()).hexdigest()=='09019da00e0713c51cff31ffd240f7bf8aa0b4e228f6d88f2bed9c1d40430392'
assert hashlib.sha256(Path(sklearn_source.__file__).read_bytes()).hexdigest()=='0cf335b8b9b4abe67259ccdcbaf07f02bd7feee31e63a06a1cce32a87fcd375d'
params.debug=None
results=[]
with tempfile.TemporaryDirectory(dir='artifacts') as temporary:
    directory=Path(temporary)
    for index,item in enumerate(json.load(sys.stdin)):
        image=np.asarray(item['image'],dtype=np.uint8)
        p=item['patch_size']
        channels=1 if image.ndim==2 else image.shape[2]
        feature_count=p*p*channels
        path=directory/f'model-{index}.fit'
        if item.get('fitted'):
            training=directory/f'training-{index}'
            training.mkdir()
            rng=np.random.default_rng(731)
            for j in range(2):
                # Independent synthetic training pixels; this is engineering
                # conformance evidence, not held-out scientific validation.
                array=rng.integers(0,256,size=(25,29,3),dtype=np.uint8)
                array[:,:14]=array[:,:14]//4
                array[:,14:]=192+array[:,14:]//4
                assert cv2.imwrite(str(training/f'train-{j}.png'),array)
            fitted=train.train_kmeans(str(training),k=3,out_path=str(path),patch_size=p,
                                      sigma=5,sampling=None,n_init=1,seed=17)
        else:
            centres=np.asarray(item.get('centres',[[20.0]*feature_count,[128.0]*feature_count,[236.0]*feature_count]),dtype=np.float64)
            fitted=MiniBatchKMeans(n_clusters=len(centres),n_init=1,random_state=17).fit(centres)
            fitted.cluster_centers_=centres
            joblib.dump(fitted,path)
        assert fitted.cluster_centers_.dtype==np.float64
        labels=kmeans_classifier.predict_kmeans(image,model_path=str(path),patch_size=p)
        results.append(dict(centres=fitted.cluster_centers_.tolist(),labels=labels.tolist()))
print(json.dumps(results))
'''


def cases():
    rng = np.random.default_rng(771)
    gray = np.where(np.indices((17, 19))[1] < 9, 30, 225)
    gray = np.clip(gray + rng.integers(-7, 8, gray.shape), 0, 255).astype(np.uint8)
    rgb = np.stack((gray, np.flipud(gray), (255 - gray) // 2), axis=2)
    result = [dict(image=image.tolist(), patch_size=p)
              for image in (gray, rgb) for p in (1, 2, 3, 10, 31)]
    result += [dict(image=[[0], [31], [255], [120], [4]], patch_size=4),
               dict(image=[[0, 31, 255, 120, 4]], patch_size=5),
               dict(image=rgb.tolist(), patch_size=3, fitted=True),
               dict(image=np.full((3, 4), 128, np.uint8).tolist(), patch_size=1,
                    centres=[[127.0], [129.0]]),
               dict(image=np.full((3, 4), 128, np.uint8).tolist(), patch_size=1,
                    centres=[[127.5], [128.5]])]
    return result


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class KMeansPredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.requests = cases()
        result = subprocess.run([str(REFERENCE), '-c', SCRIPT], cwd=ROOT,
                                input=json.dumps(cls.requests), text=True,
                                capture_output=True, timeout=90)
        if result.returncode:
            raise AssertionError(result.stderr)
        cls.references = json.loads(result.stdout)

    def model(self, request, reference):
        channels = 1 if np.asarray(request['image']).ndim == 2 else 3
        return models.model_from_centres(reference['centres'], request['patch_size'], channels)

    def test_full_images_match_public_predict_joblib_path_gray_rgb_odd_even_and_fitted_model(self):
        for request, reference in zip(self.requests[:-1], self.references[:-1]):
            with self.subTest(p=request['patch_size'], fitted=request.get('fitted', False),
                              shape=np.asarray(request['image']).shape):
                source = torch.tensor(request['image'], dtype=torch.uint8, device='cuda')
                version = source._version
                actual = native.predict_kmeans(source, self.model(request, reference),
                                               tile_pixels=7, gaussian_tile_side=7)
                np.testing.assert_array_equal(actual.cpu().numpy(), reference['labels'])
                self.assertEqual(actual.shape, source.shape[:2])
                self.assertEqual(actual.dtype, torch.uint8)
                self.assertEqual(actual.device, source.device)
                self.assertEqual(source._version, version)
                np.testing.assert_array_equal(source.cpu().numpy(), request['image'])
        self.assertGreater(len(np.unique(self.references[0]['labels'])), 1)

    def test_fractional_public_tie_withholds_complete_prediction(self):
        request, reference = self.requests[-1], self.references[-1]
        self.assertTrue(np.all(np.asarray(reference['labels']) == 0))
        source = torch.tensor(request['image'], dtype=torch.uint8, device='cuda')
        with self.assertRaises(scoring.KMeansDecisionAmbiguous):
            native.predict_kmeans(source, self.model(request, reference), tile_pixels=2)

    def test_custom_stream_and_repartition_preserve_public_labels(self):
        request, reference = self.requests[12], self.references[12]
        source = torch.tensor(request['image'], dtype=torch.uint8, device='cuda')
        model = self.model(request, reference)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            actual = native.predict_kmeans(source, model, tile_pixels=31, gaussian_tile_side=128)
        stream.synchronize()
        np.testing.assert_array_equal(actual.cpu().numpy(), reference['labels'])

    def test_global_caps_model_geometry_and_forged_identity_fail_before_gaussian(self):
        source = torch.zeros((7, 9, 3), dtype=torch.uint8, device='cuda')
        model = models.model_from_centres(((0.0,) * 27, (255.0,) * 27), 3, 3)
        for cap in ('MAX_WORK', 'MAX_SCRATCH_BYTES', 'MAX_CUDA_LIVE_BYTES'):
            with self.subTest(cap=cap), patch.object(native, cap, 1), patch.object(
                    preprocess, 'gaussian_patch_raster', side_effect=AssertionError('ran Gaussian before global cap')):
                with self.assertRaises(native.KMeansPredictionError):
                    native.predict_kmeans(source, model)
        with patch.object(preprocess, 'gaussian_patch_raster', side_effect=AssertionError('ran Gaussian before model check')):
            with self.assertRaises(models.KMeansModelError):
                native.predict_kmeans(source, replace(model, centres_sha256='0' * 64))
            with self.assertRaises(native.KMeansPredictionError):
                native.predict_kmeans(source[:, :, 0].contiguous(), model)
            with patch.object(scoring, 'MAX_WORK', 1):
                with self.assertRaises(scoring.KMeansResourceLimit):
                    native.predict_kmeans(source, model)

    def test_actual_maximum_patch_multitile_full_prediction_peak_matches_global_plan(self):
        source = torch.zeros((100, 101, 3), dtype=torch.uint8, device='cuda')
        model = models.model_from_centres(((0.0,) * 2883, (255.0,) * 2883), 31, 3)
        plan = native.preflight_kmeans_prediction(source, model)
        self.assertLessEqual(plan.scratch_bytes, native.MAX_SCRATCH_BYTES)
        self.assertLessEqual(plan.live_bytes, native.MAX_CUDA_LIVE_BYTES)
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        prepare_count = classify_count = 0
        original_prepare, original_classify = scoring._prepare_cuda_centres, scoring._classify_prepared_features
        def count_prepare(*args, **kwargs):
            nonlocal prepare_count
            prepare_count += 1
            return original_prepare(*args, **kwargs)
        def count_classify(*args, **kwargs):
            nonlocal classify_count
            classify_count += 1
            return original_classify(*args, **kwargs)
        # Plain counters release tensor arguments. MagicMock call histories
        # would retain previous feature tiles and distort the live-memory gate.
        with patch.object(scoring, '_prepare_cuda_centres', count_prepare), patch.object(
                scoring, '_classify_prepared_features', count_classify):
            labels = native.predict_kmeans(source, model)
        self.assertEqual(prepare_count, 1)
        self.assertGreater(classify_count, 1)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() - base
        self.assertLessEqual(peak, plan.gaussian_output_bytes + plan.label_output_bytes + plan.scratch_bytes)
        self.assertTrue(bool((labels == 0).all().item()))

    def test_early_mid_final_cancellation_and_mutated_input_publish_no_prediction(self):
        source = torch.zeros((4, 5), dtype=torch.uint8, device='cuda')
        model = models.model_from_centres(((0.0,), (255.0,)), 1, 1)
        cancel_errors = (native.KMeansPredictionCancelled, preprocess.KMeansPreprocessCancelled,
                         features.KMeansFeatureCancelled, scoring.KMeansClassificationCancelled)
        with patch.object(preprocess, 'gaussian_patch_raster', side_effect=AssertionError('ran Gaussian after cancellation')):
            with self.assertRaises(cancel_errors):
                native.predict_kmeans(source, model, cancel=lambda: True)
        calls = 0
        def count():
            nonlocal calls
            calls += 1
            return False
        native.predict_kmeans(source, model, tile_pixels=2, cancel=count)
        final = calls
        for limit in (10, final - 10, final):
            calls = 0
            def cancel():
                nonlocal calls
                calls += 1
                return calls >= limit
            with self.assertRaises(cancel_errors):
                native.predict_kmeans(source, model, tile_pixels=2, cancel=cancel)
        calls = 0
        def mutate_last():
            nonlocal calls
            calls += 1
            if calls == final:
                source[0, 0] = 1
            return False
        with self.assertRaises(native.KMeansPredictionError):
            native.predict_kmeans(source, model, tile_pixels=2, cancel=mutate_last)
