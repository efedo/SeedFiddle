from __future__ import annotations

import json
import subprocess
import unittest
from unittest.mock import patch
from pathlib import Path
from seedvision.cuda.plantcv_raster_spans import (
    RasterSpanCancelled, RasterSpanLimitError, fill_convex_poly_spans,
    fill_polygon_spans, filled_circle_spans, intersect_scanline_spans,
    line2_points, line8_points, normalize_spans, thick_line_spans,
    _insert_active_edge_bounded,
    iter_adjacent_closed_polyline_spans,
)

_REFERENCE_PYTHON = (Path(__file__).resolve().parents[1] / "artifacts" /
                     "plantcv-reference-env" / "Scripts" / "python.exe")


def _pinned_pixels(cases):
    if not _REFERENCE_PYTHON.is_file():
        raise FileNotFoundError(f"Pinned OpenCV oracle missing: {_REFERENCE_PYTHON}")
    source = r'''
import importlib.metadata
import json, sys
import cv2
import numpy as np
assert cv2.__version__ == "4.11.0", cv2.__version__
assert importlib.metadata.version("opencv-python") == "4.11.0.86"
assert np.__version__ == "2.2.6"
out = []
for case in json.load(sys.stdin):
    image = np.zeros((case["height"], case["width"]), dtype=np.uint8)
    kind = case["kind"]
    if kind == "line":
        cv2.line(image, tuple(case["start"]), tuple(case["end"]), 255,
                 case["thickness"], cv2.LINE_8, shift=case.get("shift", 0))
    elif kind == "convex":
        cv2.fillConvexPoly(image, np.asarray(case["points"], dtype=np.int32), 255,
                           cv2.LINE_8, shift=case["shift"])
    elif kind == "polygon":
        cv2.fillPoly(image, [np.asarray(case["points"], dtype=np.int32)], 255,
                     cv2.LINE_8, shift=0)
    elif kind == "circle":
        cv2.circle(image, tuple(case["center"]), case["radius"], 255, -1, cv2.LINE_8)
    elif kind == "contour":
        cv2.drawContours(image, [np.asarray(case['points'],np.int32).reshape(-1,1,2)],
                         -1, 255, case['thickness'], cv2.LINE_8)
    else:
        raise ValueError(kind)
    y, x = np.nonzero(image)
    out.append(sorted([[int(px), int(py)] for px, py in zip(x, y)]))
print(json.dumps(out, separators=(",", ":")))
'''
    result = subprocess.run([str(_REFERENCE_PYTHON), "-c", source],
        input=json.dumps(cases), text=True, capture_output=True, check=True, timeout=30)
    return json.loads(result.stdout)


class OpenCvRasterSpanTests(unittest.TestCase):
    def test_adjacent_stroke_batches_match_pinned_pixels_closure_and_single_large_edge(self):
        contour = [(x,2) for x in range(2,17)] + [(16,y) for y in range(3,15)]
        contour += [(x,14) for x in range(15,1,-1)] + [(2,y) for y in range(13,2,-1)]
        cases = [dict(kind='contour',width=19,height=17,points=contour,thickness=t)
                 for t in (1,2,5,13)]
        cases += [dict(kind='contour',width=3,height=h,points=[(1,h//2)],thickness=32767)
                  for h in (8191,8192,8193)]
        expected = _pinned_pixels(cases)
        for case, pixels in zip(cases,expected):
            for edges in (1,3,256):
                batches = iter_adjacent_closed_polyline_spans(case['points'],case['width'],
                    case['height'],thickness=case['thickness'],batch_edges=edges)
                actual = set()
                for spans in batches:
                    actual.update(self._span_pixels(spans,case['width'],case['height']))
                self.assertEqual(actual,{tuple(p) for p in pixels})

    def test_adjacent_stroke_work_and_raw_limits_span_batches_and_poll_before_resume(self):
        points = [(1,1),(2,1),(2,2),(1,2)]
        with patch('seedvision.cuda.plantcv_raster_spans.MAX_RASTER_SPAN_SEGMENTS',3):
            batches = iter_adjacent_closed_polyline_spans(points,4,4,thickness=1,batch_edges=1)
            self.assertTrue(next(batches))
            with self.assertRaises(RasterSpanLimitError):
                next(batches)
        stopped = [False]
        budget = [0]
        batches = iter_adjacent_closed_polyline_spans(points,4,4,thickness=5,batch_edges=1,
            cancel=lambda:stopped[0],work_budget=budget)
        self.assertTrue(next(batches))
        self.assertGreater(budget[0],0)
        stopped[0] = True
        with self.assertRaises(RasterSpanCancelled):
            next(batches)
        with self.assertRaisesRegex(ValueError,'adjacent'):
            list(iter_adjacent_closed_polyline_spans([(0,0),(3,3)],4,4))
        with patch('seedvision.cuda.plantcv_raster_spans.MAX_RASTER_SPAN_WORK',1):
            with self.assertRaises(RasterSpanLimitError):
                list(iter_adjacent_closed_polyline_spans(points,4,4,work_budget=[0]))

    @staticmethod
    def _span_pixels(spans, width, height):
        pixels = set()
        for y, x0, x1 in spans:
            if not (0 <= y < height and 0 <= x0 <= x1 < width):
                raise AssertionError(f"out-of-frame or invalid span: {(y,x0,x1)}")
            pixels.update((x,y) for x in range(x0, x1 + 1))
        return pixels

    def _assert_oracle(self, cases, builder):
        expected = _pinned_pixels(cases)
        for case, expected_xy in zip(cases, expected):
            with self.subTest(case=case):
                self.assertEqual(builder(case), {tuple(point) for point in expected_xy})

    def test_line8_points_match_pinned_cv2_with_global_clipping_and_tile_seams(self):
        cases = [
            {"kind":"line","width":9,"height":7,"start":[-7,1],"end":[14,6],"thickness":1},
            {"kind":"line","width":9,"height":7,"start":[-7,6],"end":[14,1],"thickness":1},
            {"kind":"line","width":9,"height":7,"start":[1,-7],"end":[6,14],"thickness":1},
            {"kind":"line","width":9,"height":7,"start":[6,-7],"end":[1,14],"thickness":1},
            {"kind":"line","width":513,"height":37,"start":[-19,5],"end":[530,31],"thickness":1},
            {"kind":"line","width":37,"height":513,"start":[5,-19],"end":[31,530],"thickness":1},
        ]
        self._assert_oracle(cases, lambda c: set(line8_points(c["width"],c["height"],tuple(c["start"]),tuple(c["end"]))))

    def test_line2_matches_pinned_cv2_for_fractional_fixedpoint_clip_phase(self):
        q = 1 << 16
        cases = [
            {"kind":"line","width":13,"height":11,"start":[-3*q+q//4,2*q+q//3],"end":[15*q+q//5,9*q+q//7],"shift":16,"thickness":1},
            {"kind":"line","width":13,"height":11,"start":[2*q+q//7,12*q+q//3],"end":[10*q+q//2,-2*q+q//5],"shift":16,"thickness":1},
            {"kind":"line","width":513,"height":41,"start":[-9*q+q//3,4*q+q//5],"end":[519*q+q//7,32*q+q//4],"shift":16,"thickness":1},
        ]
        self._assert_oracle(cases, lambda c: set(line2_points(c["width"],c["height"],tuple(c["start"]),tuple(c["end"]))))

    def test_fill_convex_poly_spans_match_pinned_cv2(self):
        q = 1 << 16
        cases = [
            {"kind":"convex","width":17,"height":13,"shift":0,"points":[[2,2],[14,3],[11,10],[4,9]]},
            {"kind":"convex","width":17,"height":13,"shift":16,"points":[[-q//3,2*q+q//4],[13*q+q//5,q+q//2],[12*q+q//7,11*q+q//3],[3*q+q//2,10*q+q//5]]},
            {"kind":"convex","width":513,"height":37,"shift":16,"points":[[480*q+q//3,3*q+q//5],[520*q+q//2,5*q+q//7],[518*q+q//4,31*q+q//3],[482*q+q//6,29*q+q//2]]},
        ]
        self._assert_oracle(cases, lambda c: self._span_pixels(fill_convex_poly_spans(tuple(map(tuple,c["points"])),c["width"],c["height"],shift=c["shift"]),c["width"],c["height"]))

    def test_collect_poly_edges_fill_spans_match_pinned_cv2(self):
        cases = [
            {"kind":"polygon","width":19,"height":13,"points":[[2,2],[15,3],[12,10],[5,9]]},
            {"kind":"polygon","width":19,"height":13,"points":[[-4,2],[14,1],[10,11],[3,8]]},
            {"kind":"polygon","width":513,"height":43,"points":[[480,2],[520,4],[518,38],[482,40]]},
        ]
        self._assert_oracle(cases, lambda c: self._span_pixels(fill_polygon_spans(tuple(map(tuple,c["points"])),c["width"],c["height"]),c["width"],c["height"]))

    def test_filled_circle_spans_match_pinned_cv2_clipped_at_edges(self):
        cases = [
            {"kind":"circle","width":13,"height":11,"center":[6,5],"radius":3},
            {"kind":"circle","width":13,"height":11,"center":[0,0],"radius":3},
            {"kind":"circle","width":13,"height":11,"center":[12,10],"radius":4},
            {"kind":"circle","width":13,"height":11,"center":[-2,5],"radius":3},
        ]
        self._assert_oracle(cases, lambda c: self._span_pixels(filled_circle_spans(c["width"],c["height"],tuple(c["center"]),c["radius"]),c["width"],c["height"]))

    def test_thick_line_spans_match_pinned_cv2_thickness_five(self):
        cases = [
            {"kind":"line","width":513,"height":47,"start":[-17,4],"end":[530,39],"thickness":5},
            {"kind":"line","width":31,"height":25,"start":[2,3],"end":[28,20],"thickness":5},
            {"kind":"line","width":31,"height":25,"start":[15,-8],"end":[15,32],"thickness":5},
            {"kind":"line","width":31,"height":25,"start":[-8,12],"end":[38,12],"thickness":5},
            {"kind":"line","width":13,"height":11,"start":[6,5],"end":[6,5],"thickness":5},
        ]
        self._assert_oracle(cases, lambda c: self._span_pixels(thick_line_spans(c["width"],c["height"],tuple(c["start"]),tuple(c["end"]),thickness=5),c["width"],c["height"]))

    def test_span_normalization_intersection_and_bounds(self):
        spans = normalize_spans([(1,4,4),(1,1,2),(1,3,3),(1,-4,0)],width=10,height=5)
        self.assertEqual(spans, ((1,0,4),))
        endpoints = intersect_scanline_spans(((1,1,4),(2,2,5)),((1,3,6),(2,0,3)))
        self.assertEqual(endpoints, ((2,2),(4,1)))
        with self.assertRaises(RasterSpanLimitError):
            normalize_spans(iter([(0,0,0)]*3),width=2,height=2,max_work=2)
        with self.assertRaises(RasterSpanLimitError):
            normalize_spans(iter([(99,0,0)]*3),width=2,height=2,max_work=2)

    def test_primitive_work_honors_cancellation_and_point_caps(self):
        cancelled = lambda: True
        with self.assertRaises(RasterSpanCancelled):
            line8_points(8,8,(0,0),(7,5),cancel=cancelled)
        with self.assertRaises(RasterSpanCancelled):
            line2_points(8,8,(0,0),(7<<16,5<<16),cancel=cancelled)
        with self.assertRaises(RasterSpanCancelled):
            fill_polygon_spans(((1,1),(6,1),(6,6),(1,6)),8,8,cancel=cancelled)
        with self.assertRaises(RasterSpanLimitError):
            line8_points(1_000_002,1,(0,0),(1_000_001,0))

    def test_active_edge_search_and_shift_cost_are_bounded_before_insert(self):
        active = [[i, i] for i in range(64)]
        before = [entry.copy() for entry in active]
        with patch('seedvision.cuda.plantcv_raster_spans.MAX_RASTER_SPAN_WORK', 5):
            with self.assertRaises(RasterSpanLimitError):
                _insert_active_edge_bounded(active, [100, 100], work=0, cancel=None)
            # Exercise the public path too: many edges enter the same scan row.
            with self.assertRaises(RasterSpanLimitError):
                fill_polygon_spans(tuple((i//2, i%2) for i in range(128)), 64, 2)
        self.assertEqual(active, before)
        with patch('seedvision.cuda.plantcv_raster_spans.MAX_RASTER_SPAN_WORK', 100):
            with self.assertRaisesRegex(RasterSpanLimitError, 'insertion work'):
                _insert_active_edge_bounded(active, [99, 32], work=60, cancel=None)
        self.assertEqual(active, before)

    def test_active_edge_insertion_polls_cancellation_inside_one_row(self):
        active = [[i, i] for i in range(5000)]
        before = [entry.copy() for entry in active]
        with self.assertRaises(RasterSpanCancelled):
            _insert_active_edge_bounded(active, [6000, 6000], work=4095, cancel=lambda: True)
        self.assertEqual(active, before)
        # Equal-x insertion retains the source's order, and suffix movement
        # itself polls when it crosses a work interval.
        with self.assertRaises(RasterSpanCancelled):
            _insert_active_edge_bounded(active, [6000, 0], work=0, cancel=lambda: True)
        self.assertEqual(active, before)
        equal = [[0, 1], [1, 1]]
        _insert_active_edge_bounded(equal, [2, 1], work=0, cancel=None)
        self.assertEqual(equal, [[2, 1], [0, 1], [1, 1]])


if __name__ == "__main__":
    unittest.main()
