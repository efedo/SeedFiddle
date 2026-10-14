import json
import hashlib
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from seedvision.cuda import plantcv_minrect as rect
from seedvision.cuda.plantcv_card_detection import GRID_BOX_411

REFERENCE=Path(__file__).resolve().parents[1]/'artifacts/plantcv-reference-env/Scripts/python.exe'
SCRIPT=r'''
import cv2,json,sys,numpy as np
from importlib.metadata import version
assert version('opencv-python')=='4.11.0.86' and cv2.__version__=='4.11.0'
assert version('numpy')=='2.2.6'
result=[]
for item in json.load(sys.stdin):
    points=np.asarray(item['points'],dtype=item['dtype']).reshape(-1,2)
    box=cv2.minAreaRect(points)
    hull=cv2.convexHull(points,clockwise=False,returnPoints=True)
    result.append({'rect':box,'box':cv2.boxPoints(box).tolist(),
        'hull':None if hull is None else hull.tolist()})
print(json.dumps(result))
'''


class PinnedMinRectTests(unittest.TestCase):
    def oracle(self,fixtures):
        result=subprocess.run([str(REFERENCE),'-c',SCRIPT],
            input=json.dumps([{'points':p.tolist(),'dtype':str(p.dtype)} for p in fixtures]),
            text=True,capture_output=True,timeout=90)
        self.assertEqual(result.returncode,0,result.stderr)
        return json.loads(result.stdout)

    def test_exact_pinned_rectangles_hulls_boxes_ties_degeneracy_and_point_clouds(self):
        fixtures=[np.empty((0,2),dtype=np.int32),np.array([[7,11]],dtype=np.int32),
            np.array([[2,4],[29,13]],dtype=np.int32),
            np.array([[4,2],[4,9],[4,16]],dtype=np.int32),
            np.array([(i*100,j*100) for j in range(6) for i in range(4)],np.int32)]
        for angle in (0,1,13,30,45,60,89,90,127,179):
            points=cv2.boxPoints(((300,400),(50,80),angle))
            fixtures.extend([points,points.astype(np.int32),
                (points+np.array([2**24,2**24],np.float32)),np.vstack((points,points[0]))])
        rng=np.random.default_rng(411000)
        fixtures += [rng.integers(-10000,10000,(n,2),dtype=np.int32)
                     for n in (3,4,5,8,13,31,100,511) for _ in range(8)]
        expected=self.oracle(fixtures)
        for points,original in zip(fixtures,expected):
            actual=rect.min_area_rect_411(points)
            np.testing.assert_array_equal(np.r_[actual[0],actual[1],actual[2]],
                                         np.r_[original['rect'][0],original['rect'][1],original['rect'][2]])
            np.testing.assert_array_equal(rect.box_points_411(actual),np.asarray(original['box'],np.float32))
            hull=cv2.convexHull(points,clockwise=False,returnPoints=True)
            self.assertEqual(None if hull is None else hull.tolist(),original['hull'])

    def test_fixed_grid_box_bits_are_pinned_and_cannot_be_mutated(self):
        grid=np.array([(i*100,j*100) for j in range(6) for i in range(4)],np.int32)
        original=self.oracle([grid])[0]
        expected=np.asarray(original['box'],dtype=np.float32)
        self.assertEqual(GRID_BOX_411.tobytes(),expected.tobytes())
        with self.assertRaises(ValueError): GRID_BOX_411.setflags(write=True)

    def test_retained_primary_mathematical_source_pins(self):
        root=Path(__file__).resolve().parents[1]/'artifacts/reference-sources/opencv-4.11.0-card'
        for name,pin in (('rotcalipers.cpp',rect.ROTCALIPERS_SOURCE_SHA256),('types.cpp',rect.TYPES_SOURCE_SHA256)):
            self.assertEqual(hashlib.sha256((root/name).read_bytes()).hexdigest(),pin)

    def test_bounds_before_hull_and_cancel_before_after_hull_and_long_loops(self):
        for points in (np.zeros((2,3),np.int32),np.array([[float('nan'),0]],np.float32),
                       np.array([[65_000_000,0]],np.int32)):
            with patch.object(rect.cv2,'convexHull') as hull:
                with self.assertRaises(ValueError):rect.min_area_rect_411(points)
                hull.assert_not_called()
        points=np.array([[0,0],[0,2],[2,2],[2,0]],np.int32)
        with patch.object(rect,'MAX_POINTS',3),patch.object(rect.cv2,'convexHull') as hull:
            with self.assertRaises(ValueError):rect.min_area_rect_411(points)
            hull.assert_not_called()
        calls=0
        def count():
            nonlocal calls
            calls+=1
        rect.min_area_rect_411(points,guard=count)
        final=calls
        for target in (1,2,final):
            calls=0
            def cancel():
                nonlocal calls
                calls+=1
                if calls==target: raise RuntimeError('cancelled')
            with self.assertRaisesRegex(RuntimeError,'cancelled'):
                rect.min_area_rect_411(points,guard=cancel)


if __name__=='__main__':unittest.main()
