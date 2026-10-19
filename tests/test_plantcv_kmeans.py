
from tests.functional_units import functional_units
import json
from pathlib import Path
import subprocess
import threading
import unittest
from unittest.mock import patch

import numpy as np
import torch

from seedvision.cuda import plantcv_kmeans as native

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT/'artifacts'/'plantcv-reference-env'/'Scripts'/'python.exe'
SCRIPT = r'''
import hashlib,importlib.metadata,json,sys
from pathlib import Path
import cv2,numpy as np
from plantcv.plantcv import kmeans_classifier,params
assert importlib.metadata.version('plantcv')=='4.11.3'
assert importlib.metadata.version('opencv-python')=='4.11.0.86'
assert np.__version__=='2.2.6' and cv2.__version__=='4.11.0'
assert hashlib.sha256(Path(kmeans_classifier.__file__).read_bytes()).hexdigest()=='09019da00e0713c51cff31ffd240f7bf8aa0b4e228f6d88f2bed9c1d40430392'
params.debug=None
results=[]
for request in json.load(sys.stdin):
    labels=np.asarray(request['labels'],dtype=np.uint8)
    try:
        value=kmeans_classifier.mask_kmeans(labels,request['k'],cat_list=request['categories'])
    except Exception as error:
        results.append({'error':type(error).__name__})
    else:
        results.append({'mapping':{key:mask.tolist() for key,mask in value.items()}}
            if isinstance(value,dict) else {'array':value.tolist()})
print(json.dumps(results))
'''


@unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
@functional_units
class PlantCVKMeansMaskTests(unittest.TestCase):
    def public(self,labels,k,categories):
        result = subprocess.run([str(REFERENCE),'-c',SCRIPT],input=json.dumps([
            dict(labels=labels.tolist(),k=k,categories=ids) for ids in categories]),
            capture_output=True,text=True,timeout=45)
        self.assertEqual(result.returncode,0,result.stderr)
        return json.loads(result.stdout)

    def check_test_separate_mask_order_and_bytes_match_pinned_public_call(self):
        labels = np.random.default_rng(20261008).choice(
            np.asarray([0,1,2,5,127,255],dtype=np.uint8),size=(23,29))
        expected = self.public(labels,7,[None])[0]['mapping']
        source = torch.as_tensor(labels.copy(),device='cuda')
        before,version = source.clone(),source._version
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            actual = native.mask_kmeans(source,7,tile_pixels=19)
        stream.synchronize()
        self.assertEqual(list(actual),list(expected))
        for class_id,mask in actual.items():
            self.assertEqual(mask.device,source.device)
            self.assertEqual(mask.dtype,torch.uint8)
            np.testing.assert_array_equal(mask.cpu().numpy(),expected[class_id])
        self.assertEqual(source._version,version)
        torch.testing.assert_close(source,before)

    def check_test_unions_duplicates_absent_and_out_of_byte_ids_match_public(self):
        labels = np.asarray([[0,1,2,3,4,5,255],[5,5,3,2,1,0,7]],dtype=np.uint8)
        categories = [[5,1,5],[0],[255,99],[256,-1],[4,3,2,1,0]]
        expected = self.public(labels,6,categories)
        source = torch.as_tensor(labels.copy(),device='cuda')
        for ids,original in zip(categories,expected):
            with self.subTest(ids=ids):
                np.testing.assert_array_equal(native.mask_kmeans(source,6,ids,
                    tile_pixels=3).cpu().numpy(),original['array'])
        np.testing.assert_array_equal(source.cpu().numpy(),labels)

    def check_test_k_zero_and_empty_category_failure_match_public(self):
        labels = np.asarray([[0,1],[1,0]],dtype=np.uint8)
        self.assertEqual(self.public(labels,0,[None])[0],{'mapping':{}})
        self.assertEqual(self.public(labels,2,[[]])[0],{'error':'UnboundLocalError'})
        source = torch.as_tensor(labels,device='cuda')
        self.assertEqual(native.mask_kmeans(source,0),{})
        with self.assertRaises(native.NativeKMeansError):
            native.mask_kmeans(source,2,[])

    def check_test_output_cap_precedes_allocation_and_exact_small_cap_succeeds(self):
        labels = torch.tensor([[0,1,2],[1,0,2]],dtype=torch.uint8,device='cuda')
        with patch.object(native,'MAX_OUTPUT_BYTES',11),patch.object(torch,'empty') as allocate:
            with self.assertRaises(native.NativeKMeansError):
                native.mask_kmeans(labels,2)
            allocate.assert_not_called()
        with patch.object(native,'MAX_OUTPUT_BYTES',5),patch.object(torch,'empty_like') as allocate:
            with self.assertRaises(native.NativeKMeansError):
                native.mask_kmeans(labels,2,[1])
            allocate.assert_not_called()
        expected = self.public(labels.cpu().numpy(),2,[None])[0]['mapping']
        with patch.object(native,'MAX_OUTPUT_BYTES',12):
            actual = native.mask_kmeans(labels,2)
        self.assertEqual(sum(mask.numel() for mask in actual.values()),12)
        for class_id in actual:
            np.testing.assert_array_equal(actual[class_id].cpu().numpy(),expected[class_id])

    def check_test_mid_category_and_tile_cancellation_preserve_source(self):
        source = torch.arange(120,dtype=torch.uint8,device='cuda').reshape(10,12)
        before,version = source.clone(),source._version
        for categories,stop_at,tile in (([0,1,2,3],7,120),(None,5,7)):
            calls = 0
            def cancel():
                nonlocal calls
                calls += 1
                return calls >= stop_at
            with self.assertRaises(native.NativeKMeansCancelled):
                native.mask_kmeans(source,8,categories,cancel=cancel,tile_pixels=tile)
            self.assertEqual(calls,stop_at)
        with self.assertRaises(TypeError):
            native.mask_kmeans(source,1,cancel=threading.Event())
        with patch.object(torch,'empty') as allocate:
            with self.assertRaises(native.NativeKMeansCancelled):
                native.mask_kmeans(source,1,cancel=lambda:True)
            allocate.assert_not_called()
        self.assertEqual(source._version,version)
        torch.testing.assert_close(source,before)

    def check_test_layout_and_resource_controls_reject_before_work(self):
        source = torch.zeros((2,6),dtype=torch.uint8,device='cuda')
        for image in (source.cpu(),source.to_sparse(),source[:,::2],source.float(),source[:0]):
            with self.subTest(image=str(image.layout)),self.assertRaises(native.NativeKMeansError):
                native.mask_kmeans(image,1)
        for count in (True,-1,257,1.5):
            with self.assertRaises(native.NativeKMeansError):
                native.mask_kmeans(source,count)
        for ids in ([],[0]*(native.MAX_CAT_LIST+1),[True],[1.5],'1',(x for x in range(2))):
            with self.assertRaises(native.NativeKMeansError):
                native.mask_kmeans(source,1,ids)
        for tile in (0,True,native.MAX_TILE_PIXELS+1):
            with self.assertRaises(native.NativeKMeansError):
                native.mask_kmeans(source,1,tile_pixels=tile)

    def check_test_actual_largest_output_allocation_stays_within_tiled_scratch(self):
        pixels = native.MAX_OUTPUT_BYTES//256
        self.assertLessEqual(pixels,native.MAX_PIXELS)
        source = torch.full((1024,pixels//1024),255,dtype=torch.uint8,device='cuda')
        native.mask_kmeans(source[:1,:1].contiguous(),1)
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        masks = native.mask_kmeans(source,256)
        torch.cuda.synchronize()
        allocated_output = sum(mask.numel() for mask in masks.values())
        self.assertEqual(allocated_output,native.MAX_OUTPUT_BYTES)
        peak_scratch = torch.cuda.max_memory_allocated()-baseline-allocated_output
        self.assertLessEqual(peak_scratch,native.MAX_SCRATCH_BYTES)
        self.assertEqual(int(torch.count_nonzero(masks['0'])),0)
        self.assertEqual(int(torch.count_nonzero(masks['255'])),pixels)

    def test_workflow_lifecycle(self):
        """3 related PlantCVKMeansMaskTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_separate_mask_order_and_bytes_match_pinned_public_call)
        self.scenario(self.check_test_unions_duplicates_absent_and_out_of_byte_ids_match_public)
        self.scenario(self.check_test_actual_largest_output_allocation_stays_within_tiled_scratch)

    def test_validation_and_recovery(self):
        """4 related PlantCVKMeansMaskTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_k_zero_and_empty_category_failure_match_public)
        self.scenario(self.check_test_output_cap_precedes_allocation_and_exact_small_cap_succeeds)
        self.scenario(self.check_test_mid_category_and_tile_cancellation_preserve_source)
        self.scenario(self.check_test_layout_and_resource_controls_reject_before_work)
