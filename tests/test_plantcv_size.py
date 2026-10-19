from __future__ import annotations

from tests.functional_units import functional_units

import json
import os
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

import numpy as np
import cv2

import torch

from seedvision.cuda.plantcv_size import (
    MAX_RASTER_PIXELS,
    MAX_TILE_PIXELS,
    MAX_CONTOUR_SCAN_ROWS,
    CompactBoundaryTopology,
    PlantCVSizeResourceError,
    PlantCVSizeCancelledError,
    OPENCV_RUNTIME_VERSION,
    PLANTCV_OPENCV_PYTHON_VERSION,
    PLANTCV_OPENCV_VERSION,
    _fitellipse_uses_411_perturbation,
    _fitellipse_no_direct_411,
    _fitellipse_no_direct_411_details,
    _fitellipse_orientation_ambiguous,
    _NEIGHBOR_OFFSETS,
    _span_components,
    _trace_sparse_contours,
    _valid_retr_hierarchy,
    premeasure_plantcv_binary_mask,
)



def _pinned_public_size(mask, *, line_thickness=5):
    script = r'''
from importlib.metadata import version
from hashlib import sha256
from pathlib import Path
import inspect, json, sys
import cv2, numpy as np
from plantcv import plantcv as pcv
assert version("plantcv") == "4.11.3"
assert version("opencv-python") == "4.11.0.86"
assert version("numpy") == "2.2.6"
assert cv2.__version__ == "4.11.0"
assert sha256(Path(inspect.getsourcefile(pcv.analyze.size)).read_bytes()).hexdigest() == "db5942d8cda707952e5b43777f3641a195383f62f79ab33a81b4f053bc1eb482"
assert sha256(Path(inspect.getsourcefile(pcv.within_frame)).read_bytes()).hexdigest() == "d5c4bf3457ba143c7eb6acda6feffee9284e2762a36d60c5f7378d13be94b5e7"
payload = json.load(sys.stdin)
mask = np.asarray(payload['mask'], dtype=np.int32)
pcv.params.debug = None
pcv.params.verbose = False
pcv.params.unit = "pixels"
pcv.params.line_thickness = payload['line_thickness']
pcv.outputs.clear()
pcv.analyze.size(img=np.zeros((*mask.shape,3),dtype=np.uint8),
    labeled_mask=mask, n_labels=1, label="oracle")
sample = next(iter(pcv.outputs.observations.values()))
def clean(value):
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, (tuple,list)): return [clean(x) for x in value]
    if isinstance(value, type): return value.__name__
    return value
print(json.dumps({name: {key:clean(item[key]) for key in
    ("value","trait","method","scale","datatype","label")}
    for name,item in sample.items()}))
'''
    result = subprocess.run([str(_REFERENCE_PYTHON), "-c", script],
        input=json.dumps({'mask':mask.tolist(), 'line_thickness':line_thickness}),
        text=True, capture_output=True, timeout=90)
    if result.returncode:
        raise AssertionError(result.stderr)
    return json.loads(result.stdout)

_ROOT = Path(__file__).resolve().parents[1]
_REFERENCE_PYTHON = Path(os.environ.get('SEEDVISION_PLANTCV_REFERENCE_PYTHON',
    str(_ROOT/'artifacts'/'plantcv-reference-env'/'Scripts'/'python.exe')))
_PINNED_CONTOUR_SCRIPT = r"""
import json, sys
from importlib.metadata import version
import cv2
import numpy as np
assert version('opencv-python') == '4.11.0.86'
assert version('numpy') == '2.2.6'
payload = json.load(sys.stdin)
def trace(payload):
    mask = np.asarray(payload['data'], dtype=np.uint8).reshape(payload['shape'])
    contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    return {'version': cv2.__version__,
        'contours': [c[:, 0, :].tolist() for c in contours],
        'hierarchy': None if hierarchy is None else hierarchy.tolist()}
print(json.dumps([trace(item) for item in payload['masks']]
    if 'masks' in payload else trace(payload)))
"""


def _compact_boundary_records(mask):
    height, width = mask.shape
    rows = []
    for y, x in np.argwhere(mask != 0):
        bits = 0
        for index, (dy, dx) in enumerate(_NEIGHBOR_OFFSETS):
            ny, nx = int(y+dy), int(x+dx)
            if 0 <= ny < height and 0 <= nx < width and mask[ny, nx]:
                bits |= 1 << index
        if bits != 255:
            rows.append((int(x), int(y), bits))
    return np.asarray(rows, dtype=np.int32).reshape(-1, 3)


def _pinned_contours(mask):
    if not _REFERENCE_PYTHON.is_file():
        raise AssertionError('Pinned OpenCV interpreter is required for contour conformance.')
    completed = subprocess.run([str(_REFERENCE_PYTHON), '-c', _PINNED_CONTOUR_SCRIPT],
        input=json.dumps({'shape': list(mask.shape), 'data': mask.reshape(-1).tolist()}),
        text=True, capture_output=True, timeout=90, check=True)
    return json.loads(completed.stdout)



def _pinned_contours_batch(masks):
    if not _REFERENCE_PYTHON.is_file():
        raise AssertionError('Pinned OpenCV interpreter is required for contour conformance.')
    payload = {'masks': [
        {'shape': list(mask.shape), 'data': mask.reshape(-1).tolist()} for mask in masks]}
    completed = subprocess.run([str(_REFERENCE_PYTHON), '-c', _PINNED_CONTOUR_SCRIPT],
        input=json.dumps(payload), text=True, capture_output=True, timeout=90, check=True)
    return json.loads(completed.stdout)


def _nested_contour_fixtures():
    ring = np.zeros((21, 29), dtype=np.uint8)
    ring[2:19, 2:20] = 255
    ring[6:15, 6:16] = 0
    siblings = np.zeros((31, 43), dtype=np.uint8)
    siblings[2:29, 2:41] = 255
    siblings[6:13, 6:13] = 0
    siblings[17:25, 25:34] = 0
    deep = np.zeros((41, 41), dtype=np.uint8)
    deep[2:39, 2:39] = 255
    deep[6:35, 6:35] = 0
    deep[10:31, 10:31] = 255
    deep[14:27, 14:27] = 0
    deep[18:23, 18:23] = 255
    nested_separate = ring.copy()
    nested_separate[9:12, 9:13] = 255
    nested_separate[3:9, 23:27] = 255
    diagonal = np.full((5, 5), 255, dtype=np.uint8)
    diagonal[2, 2] = diagonal[3, 3] = diagonal[4, 4] = 0
    full_frame = np.full((7, 7), 255, dtype=np.uint8)
    full_frame[2:5, 2:5] = 0
    separated = np.full((7, 7), 255, dtype=np.uint8)
    separated[2, 3] = separated[4, 3] = 0
    return (('ring', ring), ('sibling_holes', siblings), ('deep_nesting', deep),
        ('nested_and_separate', nested_separate), ('diagonal_background', diagonal),
        ('frame_touching_ring', full_frame), ('separated_holes', separated))



_PINNED_FITELLIPSE_SCRIPT = r"""
import json, sys
from importlib.metadata import version
import cv2
import numpy as np
assert version('opencv-python') == '4.11.0.86'
assert version('numpy') == '2.2.6'
assert cv2.__version__ == '4.11.0'
result = []
for points in json.load(sys.stdin):
    center, axes, angle = cv2.fitEllipse(np.asarray(points, dtype=np.int32).reshape(-1, 1, 2))
    result.append([list(center), list(axes), float(angle)])
print(json.dumps(result))
"""


def _pinned_fitellipse_batch(contours):
    payload = [np.asarray(points, dtype=np.int32).reshape(-1, 2).tolist() for points in contours]
    result = subprocess.run([str(_REFERENCE_PYTHON), '-c', _PINNED_FITELLIPSE_SCRIPT],
        input=json.dumps(payload), text=True, capture_output=True, timeout=90, check=True)
    return json.loads(result.stdout)


def _fitellipse_oracle_masks():
    vertical = np.zeros((64, 80), dtype=np.uint8)
    vertical[8:23, 42] = 255
    translated = np.zeros((64, 80), dtype=np.uint8)
    translated[24:39, 51] = 255
    horizontal = np.zeros((64, 80), dtype=np.uint8)
    horizontal[31, 20:35] = 255
    diagonal = np.zeros((64, 80), dtype=np.uint8)
    antidiagonal = np.zeros((64, 80), dtype=np.uint8)
    for point in range(10, 25):
        diagonal[point, point+8] = 255
        antidiagonal[point, 50-point] = 255
    saddle = np.asarray([[255, 0, 255], [0, 255, 0], [255, 0, 255]], dtype=np.uint8)
    translated_saddle = np.zeros((20, 25), dtype=np.uint8)
    translated_saddle[8:11, 13:16] = saddle
    return [('vertical', vertical), ('translated_vertical', translated),
        ('horizontal', horizontal), ('diagonal', diagonal), ('antidiagonal', antidiagonal),
        ('saddle', saddle), ('translated_saddle', translated_saddle)]


@functional_units
class PlantCVSizeSparseContourTests(unittest.TestCase):
    def check_test_supported_single_contours_match_pinned_order_hierarchy_and_revisits(self):
        rectangle = np.zeros((13, 17), dtype=np.uint8)
        rectangle[2:11, 3:14] = 255
        concave = np.zeros((15, 15), dtype=np.uint8)
        concave[2:13, 2:5] = 255
        concave[10:13, 2:12] = 255
        clipped = np.zeros((12, 14), dtype=np.uint8)
        clipped[:9, :8] = 255
        saddle = np.asarray([[255, 0, 255], [0, 255, 0], [255, 0, 255]], dtype=np.uint8)
        singleton = np.zeros((3, 3), dtype=np.uint8)
        singleton[1, 1] = 255
        for mask in (rectangle, concave, clipped, saddle, singleton):
            with self.subTest(shape=mask.shape, foreground=int(np.count_nonzero(mask))):
                expected = _pinned_contours(mask)
                contours, hierarchy, status = _trace_sparse_contours(
                    _compact_boundary_records(mask), width=mask.shape[1], height=mask.shape[0])
                self.assertEqual(status, 'single_external_contour')
                self.assertEqual(expected['version'], '4.11.0')
                self.assertEqual([c.tolist() for c in contours], expected['contours'])
                self.assertEqual(hierarchy.tolist(), expected['hierarchy'])

    def check_test_two_and_three_external_contours_match_pinned_order_and_hierarchy(self):
        two = np.zeros((14, 31), dtype=np.uint8)
        two[2:11, 2:12] = 255
        two[3:12, 20:29] = 255
        three = np.zeros((31, 37), dtype=np.uint8)
        three[2:10, 3:11] = 255
        three[3:9, 20:26] = 255
        three[18:27, 13:22] = 255
        for mask in (two, three):
            with self.subTest(components=2 if mask is two else 3):
                expected = _pinned_contours(mask)
                contours, hierarchy, status = _trace_sparse_contours(
                    _compact_boundary_records(mask), width=mask.shape[1], height=mask.shape[0])
                self.assertEqual(status, 'multiple_external_contours')
                self.assertEqual(expected['version'], '4.11.0')
                self.assertEqual([c.tolist() for c in contours], expected['contours'])
                self.assertEqual(hierarchy.tolist(), expected['hierarchy'])

    def check_test_holes_nested_islands_and_siblings_match_pinned_order_and_hierarchy(self):
        fixtures = _nested_contour_fixtures()
        for (name, mask), expected in zip(fixtures, _pinned_contours_batch([m for _, m in fixtures])):
            with self.subTest(kind=name):
                contours, hierarchy, status = _trace_sparse_contours(
                    _compact_boundary_records(mask), width=mask.shape[1], height=mask.shape[0])
                self.assertEqual(status, 'nested_contours')
                self.assertEqual(expected['version'], '4.11.0')
                self.assertEqual([c.tolist() for c in contours], expected['contours'])
                self.assertEqual(hierarchy.tolist(), expected['hierarchy'])

    def check_test_exhaustive_small_masks_and_seeded_random_masks_match_pinned_tree(self):
        masks = [np.asarray([(bits >> bit) & 1 for bit in range(9)],
            dtype=np.uint8).reshape(3, 3)*255 for bits in range(512)]
        rng = np.random.default_rng(418631)
        for index in range(256):
            shape = tuple(int(value) for value in rng.integers(1, 13, size=2))
            probability = (0.15, 0.35, 0.5, 0.7, 0.85)[index % 5]
            masks.append((rng.random(shape) < probability).astype(np.uint8)*255)
        for index, (mask, expected) in enumerate(zip(masks, _pinned_contours_batch(masks))):
            with self.subTest(case=index, shape=mask.shape):
                contours, hierarchy, status = _trace_sparse_contours(
                    _compact_boundary_records(mask), width=mask.shape[1], height=mask.shape[0])
                expected_links = expected['hierarchy']
                expected_status = ('empty_mask' if not expected['contours'] else
                    'nested_contours' if any(row[3] >= 0 for row in expected_links[0]) else
                    'single_external_contour' if len(expected['contours']) == 1 else
                    'multiple_external_contours')
                self.assertEqual(status, expected_status, mask.tolist())
                self.assertEqual([c.tolist() for c in contours], expected['contours'])
                if expected_links is None:
                    self.assertEqual(hierarchy.shape, (1, 0, 4))
                else:
                    self.assertEqual(hierarchy.tolist(), expected_links)

    def check_test_hierarchy_validation_rejects_malformed_and_detached_forests_and_cancels(self):
        valid = np.asarray([[
            [4, -1, 1, -1], [3, -1, 2, 0], [-1, -1, -1, 1],
            [-1, 1, -1, 0], [-1, 0, -1, -1]]], dtype=np.int32)
        self.assertTrue(_valid_retr_hierarchy(valid))
        self.assertTrue(_valid_retr_hierarchy(np.empty((1, 0, 4), dtype=np.int32)))
        malformed = [
            [[-1, -1, -1, -1], [-1, -1, 2, 2], [-1, -1, 1, 1]],
            [[1, -1, -1, -1], [-1, -1, -1, -1]],
            [[0, -1, -1, -1]], [[2, -1, -1, -1]],
            [[-1, -1, -1, -1], [-1, -1, -1, 0]],
            [[1, 1, -1, -1], [0, 0, -1, -1]]]
        for links in malformed:
            with self.subTest(links=links):
                self.assertFalse(_valid_retr_hierarchy(np.asarray([links], dtype=np.int32)))
        for invalid in (np.zeros((2, 2, 4), dtype=np.int32),
                np.zeros((2, 3), dtype=np.int32), np.zeros((1, 4), dtype=np.float64)):
            self.assertFalse(_valid_retr_hierarchy(invalid))
        with self.assertRaises(PlantCVSizeCancelledError):
            _valid_retr_hierarchy(valid, cancel=lambda: True)
        count = 5000
        chain = np.full((1, count, 4), -1, dtype=np.int32)
        chain[0, :-1, 0] = np.arange(1, count)
        chain[0, 1:, 1] = np.arange(count-1)
        calls = 0
        def cancel():
            nonlocal calls
            calls += 1
            return calls >= 3
        with self.assertRaises(PlantCVSizeCancelledError):
            _valid_retr_hierarchy(chain, cancel=cancel)
        self.assertEqual(calls, 3)

    def check_test_missing_background_rows_are_foreground_barriers(self):
        ring = np.full((7, 7), 255, dtype=np.uint8)
        ring[2:5, 2:5] = 0
        separated_holes = np.full((7, 7), 255, dtype=np.uint8)
        separated_holes[2, 3] = separated_holes[4, 3] = 0
        for mask in (ring, separated_holes):
            contours, hierarchy, status = _trace_sparse_contours(
                _compact_boundary_records(mask), width=7, height=7)
            expected = _pinned_contours(mask)
            self.assertEqual(status, 'nested_contours')
            self.assertEqual([c.tolist() for c in contours], expected['contours'])
            self.assertEqual(hierarchy.tolist(), expected['hierarchy'])
        ids, exterior = _span_components(((2, 3, 3), (4, 3, 3)), width=7,
            height=7, diagonal_connectivity=False, background=True)
        self.assertEqual(len(set(ids)), 2)
        self.assertFalse(exterior)

    def check_test_cancel_point_limit_and_oversized_scan_extent_are_explicit(self):
        mask = np.zeros((7, 9), dtype=np.uint8)
        mask[1:6, 2:7] = 255
        records = _compact_boundary_records(mask)
        with self.assertRaises(PlantCVSizeCancelledError):
            _trace_sparse_contours(records, width=9, height=7, cancel=lambda: True)
        from seedvision.cuda import plantcv_size as module
        with patch.object(module, 'MAX_TOPOLOGY_BOUNDARY_POINTS', 3):
            contours, hierarchy, status = _trace_sparse_contours(records, width=9, height=7)
        self.assertTrue(status.startswith('unsupported_resource_bound:'))
        self.assertIsNone(contours)
        self.assertIsNone(hierarchy)
        contours, hierarchy, status = _trace_sparse_contours(
            records, width=9, height=MAX_CONTOUR_SCAN_ROWS+1)
        self.assertEqual(status, 'unsupported_scan_extent')
        self.assertIsNone(contours)
        self.assertIsNone(hierarchy)

    def check_test_component_work_aggregate_step_and_point_caps_and_mid_trace_cancel(self):
        from seedvision.cuda import plantcv_size as module
        mask = np.zeros((11, 17), dtype=np.uint8)
        mask[2:5, 2:5] = 255
        mask[2:5, 10:13] = 255
        records = _compact_boundary_records(mask)
        for name, limit, reason in (
                ('MAX_CONTOUR_COMPONENTS', 1, 'contour component limit'),
                ('MAX_CONTOUR_WORK_UNITS', 1, 'contour work-unit limit'),
                ('MAX_CONTOUR_TRACE_STEPS', 12, 'aggregate contour trace-step'),
                ('MAX_CONTOUR_AGGREGATE_POINTS', 12, 'aggregate contour-point')):
            with self.subTest(limit=name), patch.object(module, name, limit):
                contours, hierarchy, status = _trace_sparse_contours(records, width=17, height=11)
                self.assertTrue(status.startswith('unsupported_resource_bound:'+reason), status)
                self.assertIsNone(contours)
                self.assertIsNone(hierarchy)
        completed = 0
        actual_trace = module._moore_trace
        def tracked_trace(*args, **kwargs):
            nonlocal completed
            result = actual_trace(*args, **kwargs)
            completed += 1
            return result
        with patch.object(module, '_moore_trace', side_effect=tracked_trace) as traced:
            with self.assertRaises(PlantCVSizeCancelledError):
                _trace_sparse_contours(records, width=17, height=11, cancel=lambda: completed >= 1)
        self.assertEqual(completed, 1)
        self.assertEqual(traced.call_count, 2)

    def check_test_holes_share_component_point_step_caps_and_cancel_without_partial_output(self):
        from seedvision.cuda import plantcv_size as module
        mask = _nested_contour_fixtures()[0][1]
        records = _compact_boundary_records(mask)
        expected = _pinned_contours(mask)
        points = sum(map(len, expected['contours']))
        for name, limit in (('MAX_CONTOUR_COMPONENTS', 1),
                ('MAX_CONTOUR_AGGREGATE_POINTS', points-1),
                ('MAX_CONTOUR_TRACE_STEPS', points+len(expected['contours'])-1)):
            with self.subTest(cap=name), patch.object(module, name, limit):
                contours, hierarchy, status = _trace_sparse_contours(
                    records, width=mask.shape[1], height=mask.shape[0])
                self.assertTrue(status.startswith('unsupported_resource_bound:'), status)
                self.assertIsNone(contours)
                self.assertIsNone(hierarchy)
        with patch.object(module, 'MAX_CONTOUR_AGGREGATE_POINTS', points):
            contours, hierarchy, status = _trace_sparse_contours(
                records, width=mask.shape[1], height=mask.shape[0])
        self.assertEqual(status, 'nested_contours')
        self.assertEqual([c.tolist() for c in contours], expected['contours'])
        self.assertEqual(hierarchy.tolist(), expected['hierarchy'])
        completed = 0
        actual_trace = module._moore_trace
        def tracked_trace(*args, **kwargs):
            nonlocal completed
            result = actual_trace(*args, **kwargs)
            completed += 1
            return result
        with patch.object(module, '_moore_trace', side_effect=tracked_trace) as traced:
            with self.assertRaises(PlantCVSizeCancelledError):
                _trace_sparse_contours(records, width=mask.shape[1], height=mask.shape[0],
                    cancel=lambda: completed >= 1)
        self.assertEqual(completed, 1)
        self.assertEqual(traced.call_count, 2)

    def check_test_aggregate_point_limit_counts_returned_points_without_closure_duplicate(self):
        from seedvision.cuda import plantcv_size as module
        mask = np.zeros((11, 17), dtype=np.uint8)
        mask[2:5, 2:5] = 255
        mask[2:5, 10:13] = 255
        records = _compact_boundary_records(mask)
        expected = _pinned_contours(mask)
        with patch.object(module, 'MAX_CONTOUR_AGGREGATE_POINTS', 16):
            contours, hierarchy, status = _trace_sparse_contours(records, width=17, height=11)
        self.assertEqual(status, 'multiple_external_contours')
        self.assertEqual(sum(map(len, contours)), 16)
        self.assertEqual([c.tolist() for c in contours], expected['contours'])
        self.assertEqual(hierarchy.tolist(), expected['hierarchy'])

    def check_test_ellipse_conditioning_detects_pinned_perturbation_and_polls_cancellation(self):
        mask = np.asarray([[255, 0, 255], [0, 255, 0], [255, 0, 255]], dtype=np.uint8)
        contour = np.asarray(_pinned_contours(mask)['contours'][0], dtype=np.int32)
        self.assertTrue(_fitellipse_uses_411_perturbation(contour))
        calls = 0
        def cancel():
            nonlocal calls
            calls += 1
            return calls >= 3
        with self.assertRaises(PlantCVSizeCancelledError):
            _fitellipse_uses_411_perturbation(contour, cancel=cancel)


    def check_test_orientation_ambiguity_guard_and_batched_transformed_contour_oracle(self):
        for cross, delta, expected in ((0.0, -1.0, True), (1e-12, 1e-9, True),
                (0.0, 1.0, False), (1e-5, -1.0, False), (np.nan, 1.0, True),
                (0.0, np.inf, True)):
            with self.subTest(cross=cross, delta=delta):
                self.assertEqual(_fitellipse_orientation_ambiguous(cross, delta), expected)
        fixtures = _fitellipse_oracle_masks()
        topology = _pinned_contours_batch([mask for _, mask in fixtures])
        contours = []
        for (name, _), tree in zip(fixtures, topology):
            points = np.asarray(tree['contours'][0], dtype=np.int32).reshape(-1, 2)
            contours.extend(((name, points), (name+'_reversed', points[::-1].copy())))
        expected = _pinned_fitellipse_batch([points for _, points in contours])
        stable = ambiguous = 0
        for (name, points), reference in zip(contours, expected):
            with self.subTest(contour=name):
                self.assertTrue(_fitellipse_uses_411_perturbation(points))
                ellipse, cross, delta = _fitellipse_no_direct_411_details(points)
                center, axes, angle = ellipse
                ref_center, ref_axes, ref_angle = reference
                np.testing.assert_allclose(center, ref_center, rtol=0, atol=1e-6)
                np.testing.assert_allclose(sorted(axes), sorted(ref_axes), rtol=0, atol=1e-6)
                if _fitellipse_orientation_ambiguous(cross, delta):
                    ambiguous += 1
                else:
                    stable += 1
                    np.testing.assert_allclose(angle, ref_angle, rtol=0, atol=1e-6)
        self.assertGreater(ambiguous, 0, 'corpus must exercise the pinned atan2 cut')
        self.assertGreater(stable, 0, 'corpus must retain stable orientation controls')

    def check_test_411_ellipse_port_polls_after_svd_and_each_solve(self):
        from seedvision.cuda import plantcv_size as module
        mask = np.asarray([[255, 0, 255], [0, 255, 0], [255, 0, 255]], dtype=np.uint8)
        contour = np.asarray(_pinned_contours(mask)['contours'][0], dtype=np.int32)
        for operation, target_call in (('SVDecomp', 1), ('solve', 1), ('solve', 2)):
            with self.subTest(operation=operation, target_call=target_call):
                original = getattr(module.cv2, operation)
                state = {'calls': 0, 'cancel': False}
                def cancel():
                    return state['cancel']
                def after_solver(*args, **kwargs):
                    state['calls'] += 1
                    result = original(*args, **kwargs)
                    if state['calls'] == target_call:
                        state['cancel'] = True
                    return result
                with patch.object(module.cv2, operation, side_effect=after_solver):
                    with self.assertRaises(PlantCVSizeCancelledError):
                        _fitellipse_no_direct_411_details(contour, cancel=cancel)
                self.assertTrue(state['cancel'])

    def check_test_411_ellipse_point_cap_precedes_design_allocation(self):
        from seedvision.cuda import plantcv_size as module
        mask = np.asarray([[255, 0, 255], [0, 255, 0], [255, 0, 255]], dtype=np.uint8)
        contour = np.asarray(_pinned_contours(mask)['contours'][0], dtype=np.int32)
        with patch.object(module, 'MAX_CONTOUR_AGGREGATE_POINTS', 5):
            with patch.object(module.np, 'empty', side_effect=AssertionError('allocated before cap')):
                with self.assertRaises(PlantCVSizeResourceError):
                    _fitellipse_no_direct_411_details(contour)

    def check_test_deterministic_411_ellipse_port_matches_public_saddle(self):
        mask = np.asarray([[255, 0, 255], [0, 255, 0], [255, 0, 255]], dtype=np.uint8)
        contour = np.asarray(_pinned_contours(mask)['contours'][0], dtype=np.int32)
        self.assertTrue(_fitellipse_uses_411_perturbation(contour))
        expected = _pinned_public_size(mask//255)
        center, axes, angle = _fitellipse_no_direct_411(contour)
        np.testing.assert_allclose(center, expected['ellipse_center']['value'], rtol=0, atol=1e-6)
        np.testing.assert_allclose((max(axes), min(axes)),
            (expected['ellipse_major_axis']['value'], expected['ellipse_minor_axis']['value']),
            rtol=0, atol=1e-6)
        np.testing.assert_allclose(angle, expected['ellipse_angle']['value'], rtol=0, atol=1e-6)
        eccentricity = float(np.sqrt(1-(min(axes)/max(axes))**2))
        np.testing.assert_allclose(eccentricity, expected['ellipse_eccentricity']['value'], rtol=0, atol=1e-6)

    def check_test_deterministic_411_ellipse_port_is_bounded_and_cancellable(self):
        from seedvision.cuda import plantcv_size as module
        mask = np.asarray([[255, 0, 255], [0, 255, 0], [255, 0, 255]], dtype=np.uint8)
        contour = np.asarray(_pinned_contours(mask)['contours'][0], dtype=np.int32)
        with patch.object(module, 'MAX_CONTOUR_AGGREGATE_POINTS', 5):
            with self.assertRaises(PlantCVSizeResourceError):
                _fitellipse_no_direct_411(contour)
        calls = 0
        def cancel():
            nonlocal calls
            calls += 1
            return calls >= 3
        with self.assertRaises(PlantCVSizeCancelledError):
            _fitellipse_no_direct_411(contour, cancel=cancel)

    def test_workflow_lifecycle(self):
        """7 related PlantCVSizeSparseContourTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_supported_single_contours_match_pinned_order_hierarchy_and_revisits)
        self.scenario(self.check_test_two_and_three_external_contours_match_pinned_order_and_hierarchy)
        self.scenario(self.check_test_holes_nested_islands_and_siblings_match_pinned_order_and_hierarchy)
        self.scenario(self.check_test_exhaustive_small_masks_and_seeded_random_masks_match_pinned_tree)
        self.scenario(self.check_test_orientation_ambiguity_guard_and_batched_transformed_contour_oracle)
        self.scenario(self.check_test_411_ellipse_port_polls_after_svd_and_each_solve)
        self.scenario(self.check_test_deterministic_411_ellipse_port_matches_public_saddle)

    def test_validation_and_recovery(self):
        """9 related PlantCVSizeSparseContourTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_hierarchy_validation_rejects_malformed_and_detached_forests_and_cancels)
        self.scenario(self.check_test_missing_background_rows_are_foreground_barriers)
        self.scenario(self.check_test_cancel_point_limit_and_oversized_scan_extent_are_explicit)
        self.scenario(self.check_test_component_work_aggregate_step_and_point_caps_and_mid_trace_cancel)
        self.scenario(self.check_test_holes_share_component_point_step_caps_and_cancel_without_partial_output)
        self.scenario(self.check_test_aggregate_point_limit_counts_returned_points_without_closure_duplicate)
        self.scenario(self.check_test_ellipse_conditioning_detects_pinned_perturbation_and_polls_cancellation)
        self.scenario(self.check_test_411_ellipse_point_cap_precedes_design_allocation)
        self.scenario(self.check_test_deterministic_411_ellipse_port_is_bounded_and_cancellable)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA size premeasurements require an available CUDA device")
@functional_units
class PlantCVSizePremeasurementTests(unittest.TestCase):
    def check_test_explicit_line_thickness_matches_all_pinned_observations(self):
        mask = np.zeros((64, 80), dtype=np.uint8)
        cv2.fillPoly(mask, [np.asarray([[9,45],[49,9],[57,18],[22,55]], np.int32)], 255)
        source = torch.as_tensor(mask, device='cuda')
        results = {}
        for thickness in (1, 5):
            expected = _pinned_public_size(mask//255, line_thickness=thickness)
            actual = premeasure_plantcv_binary_mask(source, line_thickness=thickness)
            self.assertTrue(actual.plantcv_bundle_available, actual.plantcv_bundle_unavailable_reason)
            self.assertEqual(actual.line_thickness, thickness)
            self.assertEqual(len(actual.output_observations), 17)
            for item in actual.output_observations:
                np.testing.assert_allclose(item.value, expected[item.variable]['value'],
                    rtol=1e-6, atol=1e-6, err_msg=str((thickness, item.variable)))
            results[thickness] = actual
        default = premeasure_plantcv_binary_mask(source)
        self.assertEqual(default.plantcv_observations, results[5].plantcv_observations)
        self.assertNotEqual(results[1].plantcv_observations.longest_path,
            results[5].plantcv_observations.longest_path)

    def check_test_actual_cuda_boundary_exports_trace_exactly_and_cancel_without_mutation(self):
        mask = torch.zeros((13, 17), dtype=torch.uint8, device='cuda')
        mask[2:11, 3:14] = 255
        before = mask.clone()
        expected = _pinned_contours(mask.cpu().numpy())
        result = premeasure_plantcv_binary_mask(mask)
        topology = result.boundary_topology
        self.assertEqual(topology.contour_status, 'single_external_contour')
        self.assertEqual([c.tolist() for c in topology.contours_xy], expected['contours'])
        self.assertEqual(topology.hierarchy.tolist(), expected['hierarchy'])
        calls = 0
        def cancel():
            nonlocal calls
            calls += 1
            return calls >= 4
        with self.assertRaises(PlantCVSizeCancelledError):
            premeasure_plantcv_binary_mask(mask, cancel=cancel)
        torch.testing.assert_close(mask, before)

    def check_test_single_external_contour_matches_pinned_public_size_and_metadata(self):
        for kind in ("polygon", "clipped", "concave"):
            with self.subTest(kind=kind):
                mask = np.zeros((96, 112), dtype=np.uint8)
                points = np.asarray([[14,17],[70,11],[86,35],[73,71],[31,79],[10,49]], dtype=np.int32)
                if kind == "clipped":
                    points[:, 0] -= 28
                elif kind == "concave":
                    points = np.asarray([[14,17],[70,11],[60,35],[86,70],[31,79],[10,49]], dtype=np.int32)
                cv2.fillPoly(mask, [points], 255)
                result = premeasure_plantcv_binary_mask(torch.as_tensor(mask, device="cuda"))
                self.assertTrue(result.plantcv_bundle_available, result.plantcv_bundle_unavailable_reason)
                expected = _pinned_public_size(mask//255)
                self.assertEqual(set(expected), {item.variable for item in result.output_observations})
                self.assertEqual(OPENCV_RUNTIME_VERSION, cv2.__version__)
                self.assertEqual(PLANTCV_OPENCV_PYTHON_VERSION, "4.11.0.86")
                for item in result.output_observations:
                    record = expected[item.variable]
                    for key in ("trait", "method", "scale"):
                        self.assertEqual(getattr(item, key), record[key], item.variable)
                    self.assertEqual(str(item.datatype), record["datatype"], item.variable)
                    self.assertEqual(list(item.label) if isinstance(item.label, tuple) else item.label,
                        record["label"], item.variable)
                    np.testing.assert_allclose(item.value, record["value"], rtol=1e-6, atol=1e-6,
                        err_msg=f"{kind}:{item.variable}")
                self.assertAlmostEqual(result.plantcv_observations.longest_path,
                    expected["longest_path"]["value"], places=10)

    def check_test_small_empty_and_frame_touching_objects_match_public_gate(self):
        for point in (None, (0,0), (3,4)):
            with self.subTest(point=point):
                mask = np.zeros((12,15), dtype=np.uint8)
                if point is not None:
                    mask[point] = 255
                result = premeasure_plantcv_binary_mask(torch.as_tensor(mask, device="cuda"))
                expected = _pinned_public_size(mask//255)
                self.assertTrue(result.plantcv_bundle_available)
                for item in result.output_observations:
                    np.testing.assert_array_equal(item.value, expected[item.variable]["value"],
                        err_msg=item.variable)
                self.assertEqual(result.plantcv_observations.object_in_frame, point != (0,0))

    def check_test_holes_nesting_and_separate_regions_match_pinned_public_size_and_metadata(self):
        for name, mask in _nested_contour_fixtures():
            with self.subTest(kind=name):
                before = torch.as_tensor(mask, device='cuda')
                original = before.clone()
                result = premeasure_plantcv_binary_mask(before)
                self.assertEqual(result.boundary_topology.contour_status, 'nested_contours')
                self.assertTrue(result.plantcv_bundle_available, result.plantcv_bundle_unavailable_reason)
                expected = _pinned_public_size(mask//255)
                self.assertEqual(set(expected), {item.variable for item in result.output_observations})
                for item in result.output_observations:
                    record = expected[item.variable]
                    for key in ('trait', 'method', 'scale'):
                        self.assertEqual(getattr(item, key), record[key], item.variable)
                    self.assertEqual(str(item.datatype), record['datatype'], item.variable)
                    self.assertEqual(list(item.label) if isinstance(item.label, tuple) else item.label,
                        record['label'], item.variable)
                    np.testing.assert_allclose(item.value, record['value'], rtol=1e-6, atol=1e-6,
                        err_msg=f'{name}:{item.variable}')
                self.assertAlmostEqual(result.plantcv_observations.longest_path,
                    expected['longest_path']['value'], places=10)
                touches = bool(np.any(mask[0]) or np.any(mask[-1])
                    or np.any(mask[:, 0]) or np.any(mask[:, -1]))
                self.assertEqual(result.plantcv_observations.object_in_frame, not touches)
                torch.testing.assert_close(before, original)

    def check_test_disjoint_components_frame_contact_and_diagonal_saddle_match_pinned_public_size(self):
        two = np.zeros((24, 31), dtype=np.uint8)
        two[2:10, 2:10] = 255
        two[13:21, 20:28] = 255
        three = np.zeros((31, 37), dtype=np.uint8)
        three[2:10, 3:11] = 255
        three[3:9, 20:26] = 255
        three[18:27, 13:22] = 255
        touching = np.zeros((28, 34), dtype=np.uint8)
        touching[2:11, :8] = 255
        touching[16:25, 22:31] = 255
        saddle = np.asarray([[255, 0, 255], [0, 255, 0], [255, 0, 255]], dtype=np.uint8)
        for name, mask, status, in_frame in (
                ('two', two, 'multiple_external_contours', True),
                ('three', three, 'multiple_external_contours', True),
                ('touching', touching, 'multiple_external_contours', False),
                ('saddle', saddle, 'single_external_contour', False)):
            with self.subTest(kind=name):
                expected_topology = _pinned_contours(mask)
                expected = _pinned_public_size(mask//255)
                result = premeasure_plantcv_binary_mask(torch.as_tensor(mask, device='cuda'))
                self.assertEqual(result.boundary_topology.contour_status, status)
                self.assertEqual([c.tolist() for c in result.boundary_topology.contours_xy],
                    expected_topology['contours'])
                self.assertEqual(result.boundary_topology.hierarchy.tolist(), expected_topology['hierarchy'])
                self.assertEqual(OPENCV_RUNTIME_VERSION, cv2.__version__)
                self.assertEqual(PLANTCV_OPENCV_PYTHON_VERSION, '4.11.0.86')
                self.assertTrue(result.plantcv_bundle_available, result.plantcv_bundle_unavailable_reason)
                self.assertEqual(set(expected), {item.variable for item in result.output_observations})
                for item in result.output_observations:
                    record = expected[item.variable]
                    for key in ('trait', 'method', 'scale'):
                        self.assertEqual(getattr(item, key), record[key], item.variable)
                    self.assertEqual(str(item.datatype), record['datatype'], item.variable)
                    self.assertEqual(list(item.label) if isinstance(item.label, tuple) else item.label,
                        record['label'], item.variable)
                    np.testing.assert_allclose(item.value, record['value'], rtol=1e-6, atol=1e-6,
                        err_msg=f'{name}:{item.variable}')
                self.assertEqual(result.plantcv_observations.object_in_frame, in_frame)
                self.assertAlmostEqual(result.plantcv_observations.longest_path,
                    expected['longest_path']['value'], places=10)

    def check_test_ambiguous_orientation_withholds_whole_public_bundle_but_stable_fits_remain(self):
        fixtures = _fitellipse_oracle_masks()
        topology = _pinned_contours_batch([mask for _, mask in fixtures])
        unavailable = available = 0
        for (name, mask), tree in zip(fixtures, topology):
            with self.subTest(mask=name):
                points = np.asarray(tree['contours'][0], dtype=np.int32)
                ellipse, cross, delta = _fitellipse_no_direct_411_details(points)
                ambiguous = _fitellipse_orientation_ambiguous(cross, delta)
                expected = _pinned_public_size(mask//255)
                self.assertEqual(len(expected), 17)
                result = premeasure_plantcv_binary_mask(torch.as_tensor(mask, device='cuda'))
                if ambiguous and OPENCV_RUNTIME_VERSION == '5.0.0':
                    unavailable += 1
                    self.assertFalse(result.plantcv_bundle_available)
                    self.assertIsNone(result.plantcv_observations)
                    self.assertIsNone(result.output_observations)
                    self.assertIn('orientation numerically unstable', result.plantcv_bundle_unavailable_reason)
                else:
                    available += 1
                    self.assertTrue(result.plantcv_bundle_available, result.plantcv_bundle_unavailable_reason)
                    self.assertEqual(set(expected), {item.variable for item in result.output_observations})
                    for item in result.output_observations:
                        np.testing.assert_allclose(item.value, expected[item.variable]['value'],
                            rtol=1e-6, atol=1e-6, err_msg=f'{name}:{item.variable}')
        if OPENCV_RUNTIME_VERSION == '5.0.0':
            self.assertGreater(unavailable, 0)
        self.assertGreater(available, 0)

    def check_test_raw_mask_moments_bounds_and_sparse_neighbor_bits(self) -> None:
        mask = torch.zeros((5, 7), dtype=torch.uint8, device="cuda")
        mask[1:4, 2:6] = 255
        before = mask.clone()

        result = premeasure_plantcv_binary_mask(mask)

        self.assertEqual(result.pixel_count, 12)
        self.assertEqual(result.moment_m10_px, 42)
        self.assertEqual(result.moment_m01_px, 24)
        self.assertEqual(result.centroid_xy_px, (3.5, 2.0))
        self.assertEqual(result.mask_bbox_xywh_px, (2, 1, 4, 3))
        self.assertFalse(result.touches_frame_candidate)
        self.assertTrue(result.in_bounds_candidate)
        self.assertTrue(result.plantcv_bundle_available)
        self.assertEqual(result.plantcv_observations.area, 12.0)
        self.assertEqual(result.plantcv_observations.center_of_mass, (3.5, 2.0))
        self.assertTrue(result.plantcv_observations.object_in_frame)
        self.assertEqual(len(result.output_observations), 17)
        self.assertIsInstance(result.boundary_topology, CompactBoundaryTopology)
        self.assertEqual(result.boundary_topology.status, "sparse_boundary_row_major_not_contour_order")
        records = result.boundary_topology.xy_neighbor_bits
        self.assertEqual(records.device.type, "cpu")
        self.assertEqual(records.dtype, torch.int32)
        self.assertEqual(tuple(records.shape), (10, 3))
        self.assertEqual(records[0].tolist(), [2, 1, 28])
        self.assertNotIn([3, 2], records[:, :2].tolist())
        self.assertTrue(torch.equal(mask, before))

    def check_test_empty_mask_has_raw_empty_state_without_claiming_observation_bundle(self) -> None:
        mask = torch.zeros((4, 6), dtype=torch.uint8, device="cuda")

        result = premeasure_plantcv_binary_mask(mask)

        self.assertEqual(result.pixel_count, 0)
        self.assertEqual((result.moment_m10_px, result.moment_m01_px), (0, 0))
        self.assertIsNone(result.centroid_xy_px)
        self.assertIsNone(result.mask_bbox_xywh_px)
        self.assertFalse(result.touches_frame_candidate)
        self.assertTrue(result.in_bounds_candidate)
        self.assertEqual(tuple(result.boundary_topology.xy_neighbor_bits.shape), (0, 3))
        self.assertIsNotNone(result.plantcv_observations)
        self.assertTrue(result.plantcv_bundle_available)
        self.assertTrue(result.plantcv_observations.object_in_frame)
        self.assertEqual(result.plantcv_observations.area, 0)

    def check_test_frame_contact_matches_one_pixel_border_rule_as_a_candidate(self) -> None:
        mask = torch.zeros((5, 7), dtype=torch.uint8, device="cuda")
        mask[0, 3] = 255
        mask[2, 2] = 255

        result = premeasure_plantcv_binary_mask(mask)

        self.assertEqual(result.pixel_count, 2)
        self.assertTrue(result.touches_frame_candidate)
        self.assertFalse(result.in_bounds_candidate)
        # The composed two-singleton contour has <=5 points, so public size
        # emits its zero bundle while preserving the frame-contact observation.
        self.assertTrue(result.plantcv_bundle_available)
        self.assertEqual(result.plantcv_observations.area, 0)
        self.assertFalse(result.plantcv_observations.object_in_frame)

    def check_test_neighbor_table_reads_across_internal_tile_seam(self) -> None:
        width = MAX_TILE_PIXELS + 5
        mask = torch.zeros((3, width), dtype=torch.uint8, device="cuda")
        mask[1, MAX_TILE_PIXELS - 1 : MAX_TILE_PIXELS + 1] = 255

        result = premeasure_plantcv_binary_mask(mask)

        records = result.boundary_topology.xy_neighbor_bits
        self.assertEqual(records[:, :2].tolist(), [[MAX_TILE_PIXELS - 1, 1], [MAX_TILE_PIXELS, 1]])
        bit_by_xy = {(int(x), int(y)): int(bits) for x, y, bits in records.tolist()}
        # E is bit 2 and W is bit 6 in N,NE,E,SE,S,SW,W,NW order.
        self.assertEqual(bit_by_xy[(MAX_TILE_PIXELS - 1, 1)] & (1 << 2), 1 << 2)
        self.assertEqual(bit_by_xy[(MAX_TILE_PIXELS, 1)] & (1 << 6), 1 << 6)

    def check_test_validation_rejects_cpu_nonbinary_and_oversized_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "CUDA"):
            premeasure_plantcv_binary_mask(torch.zeros((3, 4), dtype=torch.uint8))
        with self.assertRaisesRegex(ValueError, "binary"):
            premeasure_plantcv_binary_mask(
                torch.tensor([[0, 127], [255, 0]], dtype=torch.uint8, device="cuda")
            )
        one = torch.zeros((1, 1), dtype=torch.uint8, device="cuda")
        oversized = one.expand(8000, 8001)
        self.assertGreater(oversized.numel(), MAX_RASTER_PIXELS)
        with self.assertRaises(PlantCVSizeResourceError):
            premeasure_plantcv_binary_mask(oversized)

    def test_workflow_lifecycle(self):
        """9 related PlantCVSizePremeasurementTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_explicit_line_thickness_matches_all_pinned_observations)
        self.scenario(self.check_test_single_external_contour_matches_pinned_public_size_and_metadata)
        self.scenario(self.check_test_small_empty_and_frame_touching_objects_match_public_gate)
        self.scenario(self.check_test_holes_nesting_and_separate_regions_match_pinned_public_size_and_metadata)
        self.scenario(self.check_test_disjoint_components_frame_contact_and_diagonal_saddle_match_pinned_public_size)
        self.scenario(self.check_test_raw_mask_moments_bounds_and_sparse_neighbor_bits)
        self.scenario(self.check_test_empty_mask_has_raw_empty_state_without_claiming_observation_bundle)
        self.scenario(self.check_test_frame_contact_matches_one_pixel_border_rule_as_a_candidate)
        self.scenario(self.check_test_neighbor_table_reads_across_internal_tile_seam)

    def test_validation_and_recovery(self):
        """3 related PlantCVSizePremeasurementTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_actual_cuda_boundary_exports_trace_exactly_and_cancel_without_mutation)
        self.scenario(self.check_test_ambiguous_orientation_withholds_whole_public_bundle_but_stable_fits_remain)
        self.scenario(self.check_test_validation_rejects_cpu_nonbinary_and_oversized_inputs)


if __name__ == "__main__":
    unittest.main()
