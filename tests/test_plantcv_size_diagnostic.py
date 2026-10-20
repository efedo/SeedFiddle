from __future__ import annotations

from tests.functional_units import functional_units

from dataclasses import replace
import json
import subprocess
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch

from seedvision.cuda import plantcv_size as size
from seedvision.cuda import plantcv_raster_spans as raster
from tests.test_plantcv_size import _REFERENCE_PYTHON


_DRAWING_SCRIPT = r"""
from importlib.metadata import version
import cv2,json,sys
import numpy as np
assert version('opencv-python')=='4.11.0.86' and cv2.__version__=='4.11.0'
results=[]
for case in json.load(sys.stdin):
    h,w=case['shape']
    image=np.zeros((h,w),dtype=np.uint8)
    if case['kind']=='circle':
        cv2.circle(image,tuple(case['center']),10,255,case['thickness'],lineType=cv2.LINE_8)
    else:
        contour=np.asarray(case['points'],dtype=np.int32).reshape(-1,1,2)
        cv2.drawContours(image,[contour],-1,255,case['thickness'],lineType=cv2.LINE_8)
    results.append(image.tolist())
print(json.dumps(results))
"""

_SIZE_IMAGE_SCRIPT = r"""
from importlib.metadata import version
from hashlib import sha256
from pathlib import Path
import inspect,json,sys
import cv2,numpy as np
from plantcv import plantcv as pcv
assert version('plantcv')=='4.11.3'
assert version('opencv-python')=='4.11.0.86' and cv2.__version__=='4.11.0'
assert version('numpy')=='2.2.6'
assert sha256(Path(inspect.getsourcefile(pcv.analyze.size)).read_bytes()).hexdigest()=='db5942d8cda707952e5b43777f3641a195383f62f79ab33a81b4f053bc1eb482'
payload=json.load(sys.stdin)
mask=np.asarray(payload['mask'],dtype=np.int32)
image=np.asarray(payload['image'],dtype=np.uint8)
pcv.params.debug=None; pcv.params.verbose=False; pcv.params.unit='pixels'; pcv.params.line_thickness=payload.get('line_thickness',5)
pcv.outputs.clear()
out=pcv.analyze.size(img=image,labeled_mask=mask,n_labels=int(payload['n_labels']),label='oracle')
print(json.dumps(np.asarray(out,dtype=np.uint8).tolist()))
"""


def reference(script, payload):
    completed = subprocess.run([str(_REFERENCE_PYTHON), '-c', script],
        input=json.dumps(payload), text=True, capture_output=True, timeout=90)
    if completed.returncode:
        raise AssertionError(completed.stderr)
    return np.asarray(json.loads(completed.stdout), dtype=np.uint8) if script == _SIZE_IMAGE_SCRIPT else json.loads(completed.stdout)


def span_pixels(spans, shape):
    image = np.zeros(shape, dtype=np.uint8)
    for row, left, right in spans:
        image[row, left:right+1] = 255
    return image


def fixtures():
    shapes = []
    for name, shape, points in (
        ('polygon', (72, 88), [[14,12],[58,9],[72,31],[61,57],[28,62],[10,38]]),
        ('frame', (72, 88), [[0,5],[27,8],[24,39],[7,46],[0,31]]),
        ('hole', (80, 96), [[12,8],[74,12],[83,39],[67,70],[27,74],[8,42]]),
    ):
        mask = np.zeros(shape, dtype=np.uint8)
        cv2.fillPoly(mask, [np.asarray(points, np.int32)], 1)
        if name == 'hole':
            cv2.fillPoly(mask, [np.asarray([[31,28],[57,26],[62,43],[48,56],[29,48]], np.int32)], 0)
        shapes.append((name, mask, 1))
    labels = np.zeros((96, 128), dtype=np.uint8)
    cv2.fillPoly(labels, [np.asarray([[8,12],[36,9],[48,28],[40,52],[15,57],[5,34]], np.int32)], 1)
    cv2.fillPoly(labels, [np.asarray([[52,18],[83,13],[91,37],[76,62],[54,51]], np.int32)], 2)
    shapes.append(('two_labels', labels, 2))
    shapes.append(('empty', np.zeros((32, 40), dtype=np.uint8), 1))
    return shapes


@functional_units
class SizeDiagnosticRasterTests(unittest.TestCase):
    def check_test_caliper_fallback_thickness_is_separate_from_requested_drawing(self):
        cases = (
            ([[15,5],[25,35],[24,35],[14,5]], (20,20), 1),
            ([[4,15],[35,25],[35,26],[4,16]], (20,20), 5),
            ([[0,10],[30,11],[0,12]], (10,11), 1),
        )
        for points, center, expected_thickness in cases:
            with self.subTest(points=points), patch.object(size, 'thick_line_spans',
                    wraps=size.thick_line_spans) as line:
                size._plantcv_longest_path_details(hull=np.asarray(points, np.int32).reshape(-1,1,2),
                    width=40, height=40, cmx=center[0], cmy=center[1], line_thickness=1)
                self.assertEqual(line.call_args.kwargs['thickness'], expected_thickness)

    def check_test_circle_and_closed_contours_match_isolated_411_pixels(self):
        cases = [{'kind':'circle', 'shape':[41,41], 'center':list(center)}
            for center in ((20,20),(0,0),(0,20),(20,0),(40,20),(20,40))]
        cases += [{'kind':'contour', 'shape':[36,38], 'points':points} for points in (
            [(5,4),(30,7),(24,29),(8,31)],
            [(4,4),(29,4),(29,13),(13,13),(13,29),(4,29)],
            [(0,0),(37,0),(37,35),(0,35)],
            [(5,5),(20,20),(5,20),(20,5)],
            [(12,12)], [(5,5),(22,17)])]
        cases = [dict(case, thickness=thickness) for case in cases for thickness in (1,2,3,5,8)]
        for case, expected in zip(cases, reference(_DRAWING_SCRIPT, cases)):
            with self.subTest(case=case):
                h, w = case['shape']
                spans = (size._plantcv_circle_outline_spans(w, h, tuple(case['center']),
                    thickness=case['thickness'])
                    if case['kind'] == 'circle' else
                    raster.closed_polyline_spans(case['points'], w, h, thickness=case['thickness']))
                np.testing.assert_array_equal(span_pixels(spans, (h,w)), expected)
        # Repeated edges in every direction exercise the interior templates;
        # the translated copy at the frame edge exercises the clipping path.
        loop = [(0,0),(1,0),(2,1),(2,2),(1,3),(0,3),(-1,2),(-1,1)]
        paths = [loop*3, [(0,0)]*3]
        for thickness in (1,2,3,5,8,13,31):
            for path in paths:
                for offset in ((48,48),(1,1),(93,1),(1,92),(93,92),
                        (thickness+3,thickness+3)):
                    points = [(x+offset[0],y+offset[1]) for x,y in path]
                    batches = tuple(raster.iter_adjacent_closed_polyline_spans(
                        points, 96, 96, thickness=thickness, batch_edges=2, cache_directions=True))
                    actual = span_pixels((span for batch in batches for span in batch), (96,96))
                    expected = span_pixels(raster.closed_polyline_spans(
                        points, 96, 96, thickness=thickness), (96,96))
                    np.testing.assert_array_equal(actual, expected)

    def check_test_shared_work_cap_rejects_before_polygon_allocation(self):
        with patch.object(raster, 'MAX_RASTER_SPAN_WORK', 1), \
                patch.object(raster, 'fill_convex_poly_spans') as fill:
            with self.assertRaises(raster.RasterSpanLimitError):
                raster.closed_polyline_spans(((2,2),(30,30),(2,30)), 40, 40, work_budget=[0])
            fill.assert_not_called()
        with self.assertRaises(raster.RasterSpanCancelled):
            raster.closed_polyline_spans(((2,2),(30,30)), 40, 40, cancel=lambda: True)

    def check_test_thin_geometry_caps_and_cancellation_precede_rasterization(self):
        with patch.object(raster, 'MAX_RASTER_SPAN_WORK', 1), \
                patch.object(raster, 'line8_points') as line:
            with self.assertRaises(raster.RasterSpanLimitError):
                raster.closed_polyline_spans(((2,2),(30,30)), 40, 40, thickness=1)
            line.assert_not_called()
        with patch.object(raster, 'MAX_RASTER_SPAN_WORK', 1):
            with self.assertRaises(raster.RasterSpanLimitError):
                raster.circle_outline_spans(40, 40, (20,20), 10)
        with self.assertRaises(raster.RasterSpanCancelled):
            raster.circle_outline_spans(40, 40, (20,20), 10, cancel=lambda: True)
        points = tuple((x+48,y+48) for x,y in
            ([(0,0),(1,0),(2,1),(2,2),(1,3),(0,3),(-1,2),(-1,1)]*3))
        with patch.object(raster, 'thick_line_fixed_spans', wraps=raster.thick_line_fixed_spans) as line:
            tuple(raster.iter_adjacent_closed_polyline_spans(points, 96, 96, batch_edges=16))
            self.assertEqual(line.call_count, len(points), 'Default execution retains every primitive call.')
        with self.assertRaisesRegex(ValueError, 'must be a bool'):
            tuple(raster.iter_adjacent_closed_polyline_spans(points, 96, 96, cache_directions=1))
        work, stopped = [0], [False]
        with patch.object(raster, 'thick_line_fixed_spans', wraps=raster.thick_line_fixed_spans) as line:
            batches = raster.iter_adjacent_closed_polyline_spans(points, 96, 96,
                batch_edges=16, work_budget=work, cancel=lambda:stopped[0], cache_directions=True)
            next(batches)
            self.assertEqual(line.call_count, 8, 'The second loop reuses each directed edge.')
            stopped[0] = True
            with self.assertRaises(raster.RasterSpanCancelled):
                next(batches)
        batches = raster.iter_adjacent_closed_polyline_spans(points, 96, 96,
            batch_edges=16, work_budget=work, cache_directions=True)
        next(batches)
        with patch.object(raster, 'MAX_RASTER_SPAN_WORK', work[0]+1):
            with self.assertRaises(raster.RasterSpanLimitError):
                next(batches)

    def test_workflow_lifecycle(self):
        """2 related SizeDiagnosticRasterTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_caliper_fallback_thickness_is_separate_from_requested_drawing)
        self.scenario(self.check_test_circle_and_closed_contours_match_isolated_411_pixels)

    def test_validation_and_recovery(self):
        """2 related SizeDiagnosticRasterTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_shared_work_cap_rejects_before_polygon_allocation)
        self.scenario(self.check_test_thin_geometry_caps_and_cancellation_precede_rasterization)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA is required')
@functional_units
class SizeDiagnosticCudaTests(unittest.TestCase):
    def measurements(self, labels, count, *, line_thickness=5):
        results = tuple(size.premeasure_plantcv_binary_mask(torch.as_tensor(
            (labels == value).astype(np.uint8)*255, device='cuda'), line_thickness=line_thickness)
            for value in range(1, count+1))
        self.assertTrue(all(item.plantcv_bundle_available for item in results),
            [item.plantcv_bundle_unavailable_reason for item in results])
        return results

    def check_test_actual_rgb_diagnostics_match_public_images_and_preserve_input(self):
        for name, labels, count in fixtures():
            with self.subTest(kind=name):
                yy, xx = np.indices(labels.shape)
                bgr = np.stack(((xx*5+3)%256, (yy*7+11)%256,
                    ((xx+yy)*3+19)%256), axis=-1).astype(np.uint8)
                image = torch.as_tensor(bgr[..., ::-1].copy(), device='cuda')
                before = image.clone()
                results = self.measurements(labels, count)
                actual = size.render_plantcv_size_diagnostic_rgb(image, results)
                expected = reference(_SIZE_IMAGE_SCRIPT,
                    {'mask':labels.tolist(), 'image':bgr.tolist(), 'n_labels':count})
                np.testing.assert_array_equal(actual.cpu().numpy(), expected[..., ::-1].copy())
                torch.testing.assert_close(image, before)
                self.assertEqual(actual.device, image.device)
                self.assertEqual(actual.dtype, torch.uint8)
                self.assertNotEqual(actual.data_ptr(), image.data_ptr())
                iterator = raster.iter_adjacent_closed_polyline_spans
                def small_batches(*args, **kwargs):
                    return iterator(*args, **kwargs, batch_edges=2)
                with patch.object(size, 'iter_adjacent_closed_polyline_spans', side_effect=small_batches):
                    batched = size.render_plantcv_size_diagnostic_rgb(image, results)
                torch.testing.assert_close(batched, actual, rtol=0, atol=0)

    def check_test_thickness_one_actual_images_match_pinned_including_frame_and_holes(self):
        for name, labels, count in fixtures():
            with self.subTest(kind=name):
                bgr = np.full((*labels.shape, 3), [19,73,121], dtype=np.uint8)
                image = torch.as_tensor(bgr[..., ::-1].copy(), device='cuda')
                results = self.measurements(labels, count, line_thickness=1)
                actual = size.render_plantcv_size_diagnostic_rgb(image, results, line_thickness=1)
                expected = reference(_SIZE_IMAGE_SCRIPT, dict(mask=labels.tolist(),
                    image=bgr.tolist(), n_labels=count, line_thickness=1))
                np.testing.assert_array_equal(actual.cpu().numpy(), expected[..., ::-1].copy())
                np.testing.assert_array_equal(image.cpu().numpy(), bgr[..., ::-1])

    def check_test_thickness_validation_and_mismatched_geometry_reject_before_clone(self):
        _, labels, count = fixtures()[0]
        image = torch.zeros((*labels.shape, 3), dtype=torch.uint8, device='cuda')
        results = self.measurements(labels, count, line_thickness=1)
        for thickness in (True, 0, -1, 1.5, 32768):
            with self.subTest(thickness=thickness), self.assertRaises(ValueError):
                size.premeasure_plantcv_binary_mask(None, line_thickness=thickness)
            with self.subTest(thickness=thickness), self.assertRaises(ValueError):
                size.render_plantcv_size_diagnostic_rgb(image, results, line_thickness=thickness)
        with patch.object(torch.Tensor, 'clone') as clone:
            with self.assertRaisesRegex(ValueError, 'must match'):
                size.render_plantcv_size_diagnostic_rgb(image, results)
            clone.assert_not_called()

    def check_test_cancellation_and_write_cap_publish_no_partial_image(self):
        _, labels, count = fixtures()[0]
        results = self.measurements(labels, count)
        image = torch.zeros((*labels.shape, 3), dtype=torch.uint8, device='cuda')
        before = image.clone()
        original = size._paint_rgb_spans_
        state = {'cancel':False, 'painted':0}
        def first_stroke(*args, **kwargs):
            value = original(*args, **kwargs)
            state['painted'] += 1
            state['cancel'] = True
            return value
        with patch.object(size, '_paint_rgb_spans_', side_effect=first_stroke):
            with self.assertRaises(size.PlantCVSizeCancelledError):
                size.render_plantcv_size_diagnostic_rgb(image, results,
                    cancel=lambda: state['cancel'])
        self.assertEqual(state['painted'], 1)
        torch.testing.assert_close(image, before)
        iterator = raster.iter_adjacent_closed_polyline_spans
        state['cancel'] = False
        state['batches'] = 0
        def cancel_between_batches(*args, **kwargs):
            for spans in iterator(*args, **kwargs, batch_edges=2):
                yield spans
                state['batches'] += 1
                state['cancel'] = True
        with patch.object(size, 'iter_adjacent_closed_polyline_spans', side_effect=cancel_between_batches):
            with self.assertRaises(size.PlantCVSizeCancelledError):
                size.render_plantcv_size_diagnostic_rgb(image, results,
                    cancel=lambda: state['cancel'])
        self.assertEqual(state['batches'], 1)
        torch.testing.assert_close(image, before)
        with patch.object(size, 'MAX_SIZE_DIAGNOSTIC_PIXELS', 0):
            with self.assertRaises(raster.RasterSpanLimitError):
                size.render_plantcv_size_diagnostic_rgb(image, results)
        torch.testing.assert_close(image, before)
        budgets = []
        charge = size._charge_size_geometry_work
        def record_budget(budget, amount, cancel=None):
            if budget is not None and not any(value is budget for value in budgets):
                budgets.append(budget)
            return charge(budget, amount, cancel)
        with patch.object(size, '_charge_size_geometry_work', side_effect=record_budget):
            expected = size.render_plantcv_size_diagnostic_rgb(image, results)
        self.assertEqual(len(budgets), 2, 'Preparation and painting retain separate bounded work counters.')
        limit = max(value[0] for value in budgets)+1
        self.assertGreater(sum(value[0] for value in budgets), limit)
        with patch.object(size, 'MAX_RASTER_SPAN_WORK', limit), \
                patch.object(raster, 'MAX_RASTER_SPAN_WORK', limit):
            actual = size.render_plantcv_size_diagnostic_rgb(image, results)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def check_test_aggregate_points_and_hierarchy_shape_preflight_before_cpu_transfer(self):
        _, labels, count = fixtures()[0]
        result = self.measurements(labels, count)[0]
        image = torch.zeros((*labels.shape, 3), dtype=torch.uint8, device='cuda')
        points = sum(len(c) for c in result.boundary_topology.contours_xy)
        with patch.object(size, 'MAX_SIZE_DIAGNOSTIC_POINTS', points), \
                patch.object(torch.Tensor, 'cpu') as cpu:
            with self.assertRaises(raster.RasterSpanLimitError):
                size.render_plantcv_size_diagnostic_rgb(image, (result,result))
            cpu.assert_not_called()
        bad_topology = replace(result.boundary_topology,
            hierarchy=torch.zeros((2,1,4), dtype=torch.int32, device='cuda'))
        with patch.object(torch.Tensor, 'cpu') as cpu:
            with self.assertRaises(NotImplementedError):
                size.render_plantcv_size_diagnostic_rgb(image, (replace(result,
                    boundary_topology=bad_topology),))
            cpu.assert_not_called()
        with self.assertRaises(TypeError):
            size.render_plantcv_size_diagnostic_rgb(image, iter((result,)))

    def check_test_actual_multibatch_allocator_stays_within_declared_scratch(self):
        device = torch.device('cuda')
        warm = torch.zeros((8,8,3), dtype=torch.uint8, device=device)
        size._paint_rgb_spans_(warm, ((0,0,7),), (255,0,0), [0,0])
        del warm
        dense = tuple((row,0,1199) for row in range(1200))
        # Exactly1M records: one two-pixel span plus999,999 singletons.
        # The extra pixel forces a second flush without exceeding the shared cap.
        fragmented = ((0,0,1),) + tuple((index//1200,index%1200,index%1200)
            for index in range(2, 1_000_001))
        for name, spans, pixels in (('dense',dense,1_440_000),
                                    ('fragmented',fragmented,1_000_001)):
            with self.subTest(kind=name):
                canvas = torch.zeros((1200,1200,3), dtype=torch.uint8, device=device)
                torch.cuda.synchronize()
                baseline = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                budget = [0,0]
                size._paint_rgb_spans_(canvas, spans, (255,0,0), budget)
                torch.cuda.synchronize()
                extra = torch.cuda.max_memory_allocated()-baseline
                self.assertGreater(extra, 0)
                self.assertLessEqual(extra, size.MAX_SIZE_DIAGNOSTIC_SCRATCH_BYTES)
                self.assertEqual(budget, [pixels,len(spans)])
                flat = canvas.view(-1,3)
                self.assertTrue(bool(torch.all(flat[:pixels,0:2] == 0)))
                self.assertTrue(bool(torch.all(flat[:pixels,2] == 255)))
                self.assertTrue(bool(torch.all(flat[pixels:] == 0)))

    def test_workflow_lifecycle(self):
        """4 related SizeDiagnosticCudaTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_actual_rgb_diagnostics_match_public_images_and_preserve_input)
        self.scenario(self.check_test_thickness_one_actual_images_match_pinned_including_frame_and_holes)
        self.scenario(self.check_test_aggregate_points_and_hierarchy_shape_preflight_before_cpu_transfer)
        self.scenario(self.check_test_actual_multibatch_allocator_stays_within_declared_scratch)

    def test_validation_and_recovery(self):
        """2 related SizeDiagnosticCudaTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_thickness_validation_and_mismatched_geometry_reject_before_clone)
        self.scenario(self.check_test_cancellation_and_write_cap_publish_no_partial_image)


if __name__ == '__main__':
    unittest.main()
