
from tests.functional_units import functional_units
import json
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

import numpy as np
import torch

from seedvision.cuda import plantcv_contours as contours

REFERENCE = Path(__file__).resolve().parents[1] / 'artifacts/plantcv-reference-env/Scripts/python.exe'
SCRIPT = r'''
import json, sys
from importlib.metadata import version
import cv2, numpy as np
assert version('opencv-python') == '4.11.0.86' and cv2.__version__ == '4.11.0'
assert version('numpy') == '2.2.6'
payload = json.load(sys.stdin)
output = []
for values in payload['masks']:
    image = np.asarray(values,dtype=np.uint8)
    c,h = cv2.findContours(image,cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE if payload['approximation']=='simple' else cv2.CHAIN_APPROX_NONE)
    output.append({'contours':[x.tolist() for x in c],
        'hierarchy':np.empty((1,0,4),dtype=np.int32).tolist() if h is None else h.tolist()})
print(json.dumps(output))
'''


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
@functional_units
class BinaryContourTests(unittest.TestCase):
    def oracle(self, masks, approximation='simple'):
        result = subprocess.run([str(REFERENCE), '-c', SCRIPT],
            input=json.dumps({'masks':[m.tolist() for m in masks], 'approximation':approximation}),
            text=True, capture_output=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def compare(self, masks, *, approximation='simple', tile_pixels=7):
        expected = self.oracle(masks, approximation)
        for mask, original in zip(masks, expected):
            source = torch.as_tensor(mask,device='cuda')
            version = source._version
            result = contours.trace_binary_contours_cuda(source,
                approximation=approximation,tile_pixels=tile_pixels)
            self.assertEqual([c.tolist() for c in result.contours],original['contours'])
            self.assertEqual(result.hierarchy.tolist(), original['hierarchy'])
            self.assertEqual(result.frame_shape,mask.shape)
            self.assertEqual(result.approximation,approximation)
            for array in (*result.contours, result.hierarchy):
                self.assertEqual(array.dtype,np.int32)
                self.assertFalse(array.flags.writeable)
                with self.assertRaises(ValueError):
                    array.setflags(write=True)
            self.assertEqual(source._version,version)
            np.testing.assert_array_equal(source.cpu().numpy(),mask)

    def check_test_all_512_three_by_three_masks_exact_simple_order_and_hierarchy(self):
        masks = [(((value >> np.arange(9)) & 1).reshape(3,3)*255).astype(np.uint8)
                 for value in range(512)]
        self.compare(masks)

    def check_test_nested_seam_diagonal_lines_self_touch_and_seeded_masks(self):
        ring = np.zeros((31,47),dtype=np.uint8)
        ring[1:-1,1:-1]=255
        ring[5:-5,5:-5]=0
        ring[9:-9,9:-9]=255
        diagonal=np.eye(27,dtype=np.uint8)*255
        horizontal=np.zeros((3,31),dtype=np.uint8);horizontal[1,:]=255
        vertical=horizontal.T.copy()
        rng=np.random.default_rng(411003)
        masks=[ring,diagonal,horizontal,vertical,np.full((1,27),255,dtype=np.uint8),
            np.full((27,1),255,dtype=np.uint8),np.full((5,9),255,dtype=np.uint8)]
        masks += [(rng.integers(0,2,(17,19),dtype=np.uint8)*255) for _ in range(8)]
        self.compare(masks,tile_pixels=31)
        self.compare(masks,approximation='none',tile_pixels=23)

    def check_test_noncontiguous_nonoverlapping_input_preserves_view_and_source(self):
        source=torch.zeros((29,37),dtype=torch.uint8,device='cuda')
        source[4:25,5:32]=255
        source[9:21,11:25]=0
        view=source[::2,::2].T
        expected=self.oracle([view.cpu().numpy()])[0]
        result=contours.trace_binary_contours_cuda(view,tile_pixels=19)
        self.assertEqual([c.tolist() for c in result.contours],expected['contours'])
        self.assertEqual(result.hierarchy.tolist(),expected['hierarchy'])

    def check_test_empty_is_valid_and_unsupported_and_resources_never_become_empty(self):
        source=torch.zeros((5,7),dtype=torch.uint8,device='cuda')
        self.assertEqual(contours.trace_binary_contours_cuda(source).topology_status,'empty_mask')
        for status, error in [('unsupported_test',contours.BinaryContourUnsupportedError),
                ('unsupported_resource_bound:test',contours.BinaryContourResourceError)]:
            with patch.object(contours.size,'_trace_sparse_contours',return_value=(None,None,status)):
                with self.assertRaises(error):
                    contours.trace_binary_contours_cuda(source)
        source.fill_(255)
        with patch.object(contours,'MAX_BOUNDARY_POINTS',3):
            with patch.object(contours.size,'_trace_sparse_contours') as tracer:
                with self.assertRaises(contours.BinaryContourResourceError):
                    contours.trace_binary_contours_cuda(source,tile_pixels=5)
                tracer.assert_not_called()
        with patch.object(contours.size,'MAX_CONTOUR_WORK_UNITS',1):
            with self.assertRaises(contours.BinaryContourResourceError):
                contours.trace_binary_contours_cuda(source)

    def check_test_validation_scratch_cancel_and_first_or_final_callback_mutation(self):
        source=torch.zeros((13,17),dtype=torch.uint8,device='cuda')
        source[2:11,2:15]=255
        with patch.object(contours,'MAX_SCRATCH_BYTES',1):
            with patch.object(contours.size,'_validate_mask') as validate:
                with self.assertRaises(contours.BinaryContourResourceError):
                    contours.trace_binary_contours_cuda(source)
                validate.assert_not_called()
        with self.assertRaises(contours.BinaryContourCancelled):
            contours.trace_binary_contours_cuda(source,cancel=lambda:True)
        calls=0
        def count():
            nonlocal calls
            calls+=1
            return False
        contours.trace_binary_contours_cuda(source,cancel=count)
        final=calls
        for target in (1,3,final):
            calls=0
            def mutate():
                nonlocal calls
                calls+=1
                if calls==target: source[0,0]=255
                return False
            with self.assertRaisesRegex(contours.BinaryContourError,'changed'):
                contours.trace_binary_contours_cuda(source,cancel=mutate)
            source[0,0]=0
        source[0,0]=1
        with self.assertRaisesRegex(ValueError,'binary'):
            contours.trace_binary_contours_cuda(source)
        with self.assertRaises(contours.BinaryContourError):
            contours.trace_binary_contours_cuda(source,approximation='list')
        with self.assertRaises(contours.BinaryContourResourceError):
            contours.trace_binary_contours_cuda(source,tile_pixels=True)

    def check_test_actual_custom_stream_allocation_is_within_declared_scratch(self):
        source=torch.zeros((513,769),dtype=torch.uint8,device='cuda')
        source[40:-40,40:-40]=255;source[140:-140,140:-140]=0
        stream=torch.cuda.Stream(device=source.device)
        torch.cuda.synchronize()
        baseline=torch.cuda.memory_allocated(source.device)
        torch.cuda.reset_peak_memory_stats(source.device)
        with torch.cuda.stream(stream):
            result=contours.trace_binary_contours_cuda(source,tile_pixels=65536)
        stream.synchronize()
        self.assertEqual(len(result.contours),2)
        self.assertLessEqual(torch.cuda.max_memory_allocated(source.device)-baseline,
                            contours.MAX_SCRATCH_BYTES)

    def test_workflow_lifecycle(self):
        """5 related BinaryContourTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_all_512_three_by_three_masks_exact_simple_order_and_hierarchy)
        self.scenario(self.check_test_nested_seam_diagonal_lines_self_touch_and_seeded_masks)
        self.scenario(self.check_test_noncontiguous_nonoverlapping_input_preserves_view_and_source)
        self.scenario(self.check_test_empty_is_valid_and_unsupported_and_resources_never_become_empty)
        self.scenario(self.check_test_actual_custom_stream_allocation_is_within_declared_scratch)

    def test_validation_and_recovery(self):
        """1 related BinaryContourTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_validation_scratch_cancel_and_first_or_final_callback_mutation)


if __name__=='__main__':
    unittest.main()
