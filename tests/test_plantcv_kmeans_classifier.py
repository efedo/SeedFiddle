import json
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

import numpy as np
import torch

from seedvision.cuda import plantcv_kmeans_classifier as native

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / 'artifacts/plantcv-reference-env/Scripts/python.exe'
SCRIPT = r'''
import hashlib,json,sys
from pathlib import Path
import numpy as np,sklearn
from sklearn.cluster import KMeans,MiniBatchKMeans
import sklearn.cluster._kmeans as source
assert sklearn.__version__=='1.9.1' and np.__version__=='2.2.6'
assert hashlib.sha256(Path(source.__file__).read_bytes()).hexdigest()=='0cf335b8b9b4abe67259ccdcbaf07f02bd7feee31e63a06a1cce32a87fcd375d'
results=[]
for item in json.load(sys.stdin):
    centres=np.asarray(item['centres'],dtype=np.float64)
    features=np.asarray(item['features'],dtype=np.uint16)
    count,width=centres.shape
    train=np.zeros((max(count,2),width),dtype=np.float64)
    train[:,0]=np.arange(max(count,2))%256
    labels=[]
    for cls in (KMeans,MiniBatchKMeans):
        model=cls(n_clusters=count,n_init=1,random_state=0).fit(train)
        model.cluster_centers_=centres
        labels.append(model.predict(features).tolist())
    results.append(labels)
print(json.dumps(results))
'''


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class KMeansClassifierTests(unittest.TestCase):
    def public(self, cases):
        result = subprocess.run([str(REFERENCE), '-c', SCRIPT], input=json.dumps(cases),
                                text=True, capture_output=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def test_integer_ties_duplicates_fractional_separated_and_256th_class_match_public(self):
        cases = [dict(centres=[[0], [2]], features=[[1], [0], [2], [255], [1]]),
                 dict(centres=[[0, 0], [0, 0], [2, 0]], features=[[0, 0], [1, 0], [2, 0]]),
                 dict(centres=[[0.25, 1.5], [100.5, 200.25], [254.5, 8.25]],
                      features=[[0, 2], [100, 201], [255, 7], [1, 0]]),
                 dict(centres=[[float(i)] for i in range(256)], features=[[0], [127], [255]])]
        for case, references in zip(cases, self.public(cases)):
            source = torch.tensor(case['features'], dtype=torch.uint8, device='cuda')
            before, version = source.clone(), source._version
            centres = tuple(tuple(float(x) for x in row) for row in case['centres'])
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                actual = native.classify_kmeans_features(source, centres, tile_rows=2)
            stream.synchronize()
            for expected in references:
                self.assertEqual(actual.cpu().tolist(), expected)
            self.assertEqual(actual.dtype, torch.uint8)
            self.assertEqual(actual.device, source.device)
            self.assertEqual(source._version, version)
            torch.testing.assert_close(source, before)
        self.assertEqual(references[0], [0, 127, 255])

    def test_fractional_tie_near_tie_and_late_tile_withhold_complete_vector(self):
        cases = [dict(centres=[[0.5], [1.5]], features=[[0], [1]]),
                 dict(centres=[[0.5], [1.5 + 1e-12]], features=[[1]])]
        for case, references in zip(cases, self.public(cases)):
            self.assertEqual(references[0], references[1])
            centres = tuple(tuple(row) for row in case['centres'])
            source = torch.tensor(case['features'], dtype=torch.uint8, device='cuda')
            with self.assertRaises(native.KMeansDecisionAmbiguous):
                native.classify_kmeans_features(source, centres, tile_rows=1)
        # Identical fractional rows have no distinct competing decision.
        labels = native.classify_kmeans_features(torch.tensor([[1]], dtype=torch.uint8, device='cuda'),
                                                ((0.5,), (0.5,)))
        self.assertEqual(labels.cpu().tolist(), [0])

    def test_actual_maximum_feature_multitile_scratch_stays_within_full_budget(self):
        width = 4096
        centres = tuple((float(i),) * width for i in (0, 85, 170, 255))
        rows = native._kmeans_tile_rows(width, 4 * width) + 5
        source = torch.zeros((rows, width), dtype=torch.uint8, device='cuda')
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        labels = native.classify_kmeans_features(source, centres)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() - base - labels.numel()
        self.assertLessEqual(peak, native.MAX_SCRATCH_BYTES)
        self.assertTrue(bool((labels == 0).all().item()))

    def test_work_centre_and_scratch_caps_precede_gpu_allocations(self):
        source = torch.zeros((4, 2), dtype=torch.uint8, device='cuda')
        for cap in ('MAX_WORK', 'MAX_CENTRE_VALUES', 'MAX_SCRATCH_BYTES'):
            with self.subTest(cap=cap), patch.object(native, cap, 1), patch.object(
                    native.torch, 'tensor', side_effect=AssertionError('centres copied before cap')):
                with self.assertRaises(native.KMeansResourceLimit):
                    native.classify_kmeans_features(source, ((0.0, 0.0), (1.5, 2.5)))

    def test_early_mid_and_final_cancellation_never_return_partial_labels(self):
        source = torch.zeros((4, 2), dtype=torch.uint8, device='cuda')
        centres = ((0.0, 0.0), (255.0, 255.0))
        with patch.object(native.torch, 'tensor', side_effect=AssertionError('allocated before cancel')):
            with self.assertRaises(native.KMeansClassificationCancelled):
                native.classify_kmeans_features(source, centres, cancel=lambda: True)
        for limit in (5, 12):
            calls = 0
            def cancel():
                nonlocal calls
                calls += 1
                return calls >= limit
            with self.assertRaises(native.KMeansClassificationCancelled):
                native.classify_kmeans_features(source, centres, cancel=cancel, tile_rows=1)
        for callback in (object(), lambda: 1):
            with self.assertRaises(native.KMeansClassificationError):
                native.classify_kmeans_features(source, centres, cancel=callback)

    def test_zero_rows_and_invalid_tensor_and_metadata_domains(self):
        empty = torch.empty((0, 2), dtype=torch.uint8, device='cuda')
        result = native.classify_kmeans_features(empty, ((0.0, 0.0),))
        self.assertEqual((result.shape, result.dtype, result.device), (torch.Size([0]), torch.uint8, empty.device))
        source = torch.zeros((2, 2), dtype=torch.uint8, device='cuda')
        for image in (source.cpu(), source.to(torch.uint16), source[:, ::2], source.unsqueeze(2)):
            with self.assertRaises(native.KMeansClassificationError):
                native.classify_kmeans_features(image, ((0.0, 0.0),))
        for centres in (((True, 0.0),), (('0', 0.0),), ((0.0,),), ((256.0, 0.0),),
                        ((float('nan'), 0.0),), [[0.0, 0.0]], ()):
            with self.assertRaises(native.KMeansClassificationError):
                native.classify_kmeans_features(source, centres)
        with self.assertRaises(native.KMeansResourceLimit):
            native.classify_kmeans_features(source, ((0.0, 0.0),), tile_rows=True)
