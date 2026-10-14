from dataclasses import replace
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from seedvision.benchmarking.checkpoints import CoordinateFrame
from seedvision.benchmarking.native_recipe import (NativeRecipeCancelled,
    NativeRecipeError, prepare_native_recipe_inputs, run_native_recipe)
from seedvision.benchmarking.recipe_catalogue import build_recipe_catalogue
from seedvision.segmentation.baseline import PipelineAnalysisCache


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT/'artifacts'/'plantcv-reference-env'/'Scripts'/'python.exe'
PREFIX = 'pcv_seed_binary_'
SOURCE = 'a'*64
FRAME = CoordinateFrame('test.prepared', (96, 128), 'top-left', 'x-right,y-down', 'pixel')
PREPARATION = {'gray': 'Explicit synthetic uint8 intensity fixture; no RGB derivation.',
    'rgb': 'Explicit synthetic uint8 RGB fixture in the same frame.'}


def selected_recipe(kind='binary', **updates):
    recipe = build_recipe_catalogue().recipe('plantcv_seed_'+kind)
    selected = {
        'filter.median_blur': {'ksize': 3},
        'threshold.binary': {'threshold': 100, 'object_type': 'light'},
        'threshold.mean': {'ksize': 9, 'offset': 2, 'object_type': 'light'},
        'threshold.otsu': {'object_type': 'light'},
        'morphology.fill': {'size': 20},
        'morphology.erode': {'ksize': 3, 'iterations': 1},
        'morphology.dilate': {'ksize': 3, 'iterations': 1},
        'reference.internal.watershed.local_maxima': {'distance': 8},
        'analysis.color': {'colorspaces': 'rgb', 'label': 'graph'},
        'analysis.size': {'label': 'graph'},
    }
    for operation, values in updates.items():
        selected.setdefault(operation, {}).update(values)
    return replace(recipe, stages=tuple(replace(stage, parameters=tuple(
        replace(parameter, value=selected.get(stage.operation_id, {}).get(parameter.parameter_id,
            parameter.value)) for parameter in stage.parameters)) for stage in recipe.stages))


def fixture():
    y, x = np.ogrid[:96, :128]
    gray = np.full((96, 128), 35, dtype=np.uint8)
    selected = ((x-45)**2/20**2+(y-40)**2/14**2 < 1) | ((x-77)**2/19**2+(y-41)**2/13**2 < 1)
    gray[selected] = 210
    gray[10, 10] = 255
    rgb = np.empty((96, 128, 3), dtype=np.uint8)
    rgb[..., 0] = 120+x%70
    rgb[..., 1] = 60+y%60
    rgb[..., 2] = 35
    return gray, rgb


REFERENCE_SCRIPT = r'''
from importlib.metadata import version
import hashlib, importlib
import json, sys
import numpy as np
import cv2
from scipy import ndimage
from skimage.feature import peak_local_max
from plantcv import plantcv as pcv
assert version('plantcv') == '4.11.3'
assert version('opencv-python') == '4.11.0.86'
assert version('scikit-image') == '0.25.2'
assert version('numpy') == '2.2.6'
assert version('scipy') == '1.15.3'
size_module = importlib.import_module('plantcv.plantcv.analyze.size')
frame_module = importlib.import_module('plantcv.plantcv.within_frame')
from pathlib import Path
assert hashlib.sha256(Path(size_module.__file__).read_bytes()).hexdigest() == 'db5942d8cda707952e5b43777f3641a195383f62f79ab33a81b4f053bc1eb482'
assert hashlib.sha256(Path(frame_module.__file__).read_bytes()).hexdigest() == 'd5c4bf3457ba143c7eb6acda6feffee9284e2762a36d60c5f7378d13be94b5e7'
values = np.load(sys.argv[1], allow_pickle=False)
rgb = values['rgb']
bgr = np.ascontiguousarray(rgb[..., ::-1])
outputs = {}
outputs['median_blur'] = pcv.median_blur(gray_img=values['gray'], ksize=3)
kind = sys.argv[4] if len(sys.argv) > 4 else 'binary'
if kind == 'binary':
    outputs['binary_threshold'] = pcv.threshold.binary(gray_img=outputs['median_blur'], threshold=100,
        object_type='light')
elif kind == 'mean':
    outputs['binary_threshold'] = pcv.threshold.mean(gray_img=outputs['median_blur'], ksize=9, offset=2,
        object_type='light')
elif kind == 'otsu':
    outputs['binary_threshold'] = pcv.threshold.otsu(gray_img=outputs['median_blur'], object_type='light')
else:
    raise AssertionError(kind)
outputs['small_object_filter'] = pcv.fill(bin_img=outputs['binary_threshold'], size=20)
outputs['erode'] = pcv.erode(gray_img=outputs['small_object_filter'], ksize=3, i=1)
outputs['dilate'] = pcv.dilate(gray_img=outputs['erode'], ksize=3, i=1)
mask = outputs['dilate']
outputs['distance'], _ = cv2.distanceTransformWithLabels(mask, cv2.DIST_L2, 0)
outputs['coordinates'] = peak_local_max(outputs['distance'], min_distance=8, labels=mask)
outputs['peaks'] = np.zeros(mask.shape, dtype=bool)
outputs['peaks'][tuple(outputs['coordinates'].T)] = True
outputs['markers'], _ = ndimage.label(outputs['peaks'], structure=np.ones((3, 3)))
outputs['negative_distance'] = -outputs['distance']
pcv.params.debug = None
pcv.params.verbose = False
pcv.params.line_thickness = 5
pcv.params.unit = 'pixels'
outputs['labels'] = pcv.watershed_segmentation(rgb_img=bgr, mask=mask, distance=8, label='graph')
pcv.outputs.clear()
space = sys.argv[5] if len(sys.argv) > 5 else 'rgb'
chart = pcv.analyze.color(rgb_img=bgr, labeled_mask=outputs['labels'], n_labels=int(outputs['labels'].max()),
    colorspaces=space, label='graph')
observations = dict(pcv.outputs.observations)
observations['__returned_hue_chart__'] = chart.data.to_dict('records')
pcv.outputs.clear()
size_image = pcv.analyze.size(img=bgr, labeled_mask=outputs['labels'], n_labels=int(outputs['labels'].max()), label='graph')
outputs['size_diagnostic_rgb'] = size_image[..., ::-1].copy()
observations['__size_observations__'] = dict(pcv.outputs.observations)
with open(sys.argv[3], 'w', encoding='utf-8') as stream:
    json.dump(observations, stream)
np.savez(sys.argv[2], **outputs)
'''


class NativeRecipeIdentityTests(unittest.TestCase):
    def test_imported_calculation_alias_and_computed_lab_constants_reject_changes(self):
        from seedvision.benchmarking import native_recipe as implementation
        implementation._module_record(implementation._color)
        implementation._module_record(implementation._range)
        cases = (
            (implementation._color, '_channels', lambda pixels, mode: pixels),
            (implementation._range, '_LAB_GAMMA_VALUES', tuple(0 for _ in implementation._range._LAB_GAMMA_VALUES)),
            (implementation._range, '_LAB_CBRT_VALUES', tuple(0 for _ in implementation._range._LAB_CBRT_VALUES)),
            (implementation._range, '_OPENCV_CBRT_COEFFICIENTS', tuple(0.0 for _ in implementation._range._OPENCV_CBRT_COEFFICIENTS)),
        )
        for module, name, changed in cases:
            with self.subTest(name=name), patch.object(module, name, changed):
                with self.assertRaises(NativeRecipeError):
                    implementation._module_record(module)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA is required for native recipes')
class NativeRecipeTests(unittest.TestCase):
    def setUp(self):
        gray, rgb = fixture()
        self.gray, self.rgb = (torch.as_tensor(value, device='cuda') for value in (gray, rgb))
        self.cache = PipelineAnalysisCache()

    def run_recipe(self, recipe=None, **kwargs):
        return run_native_recipe(recipe or selected_recipe(), self.gray, self.rgb,
            source_sha256=SOURCE, coordinate_frame=FRAME, preparation=PREPARATION,
            cache=self.cache, **kwargs)

    def test_cached_lab_table_mutation_is_rejected_before_node_execution(self):
        from seedvision.benchmarking import native_recipe as implementation
        tables = implementation._range._lab_tables(self.gray.device)
        for tensor in tables:
            original = tensor.clone()
            try:
                tensor[0] ^= 1
                with self.assertRaisesRegex(NativeRecipeError, 'Cached LAB conversion table'):
                    self.run_recipe()
                self.assertFalse(any(key.startswith('reference.recipe:') for key in self.cache.values))
            finally:
                tensor.copy_(original)
        implementation._verify_cached_lab_tables(self.gray.device)

    def test_primitive_cancellation_is_preserved_when_callback_is_not_latched(self):
        from seedvision.benchmarking import native_recipe as implementation
        error_types = (NativeRecipeCancelled, implementation._median.MedianCancelledError,
            implementation._fill.FillCancelledError, implementation._distance.DistanceCancelledError,
            implementation._peaks.PeakCancelledError, implementation._watershed.WatershedCancelled,
            implementation._color.ColorAnalysisCancelled, implementation._range.RangeCancelledError,
            implementation._size.PlantCVSizeCancelledError)
        runtime_records = implementation._runtime_records(self.gray.device)
        for error_type in error_types:
            with self.subTest(error=error_type.__name__):
                state = {'armed': False, 'used': False}
                def cancel():
                    if state['armed'] and not state['used']:
                        state['used'] = True
                        return True
                    return False
                def controlled_dispatch(*args, **kwargs):
                    state['armed'] = True
                    self.assertTrue(cancel())
                    raise error_type('One-shot cancellation')
                # Freeze genuine runtime records before replacing dispatch, so
                # this regression isolates cancellation classification rather
                # than intentionally triggering loaded-code verification.
                with patch.object(implementation, '_runtime_records', return_value=runtime_records), \
                        patch.object(implementation, '_dispatch', side_effect=controlled_dispatch):
                    with self.assertRaises(NativeRecipeCancelled):
                        self.run_recipe(cancel=cancel)
                self.assertTrue(state['used'])
                self.assertFalse(cancel())

    def reference(self, kind='binary', space='rgb'):
        self.assertTrue(REFERENCE.is_file(), 'Pinned original reference interpreter is required.')
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            gray, rgb = fixture()
            np.savez(directory/'inputs.npz', gray=gray, rgb=rgb)
            completed = subprocess.run([str(REFERENCE), '-c', REFERENCE_SCRIPT,
                str(directory/'inputs.npz'), str(directory/'outputs.npz'), str(directory/'observations.json'), kind, space],
                capture_output=True, text=True, timeout=120)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            with np.load(directory/'outputs.npz', allow_pickle=False) as loaded:
                expected = {name: loaded[name].copy() for name in loaded.files}
            observations = json.loads((directory/'observations.json').read_text(encoding='utf-8'))
        return expected, observations

    def assert_size_observations(self, node, expected):
        self.assertEqual(node.status, 'executed', node.reason)
        self.assertEqual(set(node.outputs), {'observations', 'diagnostic'})
        self.assertEqual(node.outputs['diagnostic'].device, self.rgb.device)
        actual = node.outputs['observations']
        self.assertEqual(set(actual), set(expected))
        for sample, variables in expected.items():
            self.assertEqual(set(actual[sample]), set(variables))
            self.assertEqual(len(variables), 17)
            for variable, original in variables.items():
                observed = actual[sample][variable]
                self.assertEqual(set(observed), set(original))
                for field in ('trait', 'method', 'scale', 'datatype'):
                    self.assertEqual(observed[field], original[field], (sample, variable, field))
                self.assertEqual(tuple(observed['label']) if isinstance(observed['label'], tuple)
                    else observed['label'], tuple(original['label']) if isinstance(original['label'], list)
                    else original['label'])
                np.testing.assert_allclose(observed['value'], original['value'], rtol=1e-6, atol=1e-6,
                    err_msg=f'{sample}: {variable}')
                if variable == 'longest_path':
                    self.assertAlmostEqual(observed['value'], original['value'], places=10)
        with self.assertRaises(TypeError):
            actual[next(iter(actual))]['area']['value'] = 0

    def test_real_canonical_graph_matches_pinned_public_preprocessing_watershed_and_color(self):
        expected, observations = self.reference()
        result = self.run_recipe()
        self.assertEqual(result.recipe_identity, selected_recipe().identity)
        self.assertEqual(result.coordinate_frame, FRAME)
        self.assertEqual(result.source_sha256, SOURCE)
        self.assertRegex(result.runtime['nvrtc'], r'^\d+\.\d+$')
        self.assertRegex(result.runtime['cuda_driver_api'], r'^\d+$')
        self.assertEqual(result.status, 'executed')
        for suffix, port in (('median_blur', 'image'), ('binary_threshold', 'mask'),
                ('small_object_filter', 'mask'), ('erode', 'mask'), ('dilate', 'mask')):
            np.testing.assert_array_equal(result.node(PREFIX+suffix).outputs[port].cpu().numpy(), expected[suffix])
        for suffix, port, reference in (('ws_distance_map', 'distance', 'distance'),
                ('ws_local_maxima', 'coordinates', 'coordinates'), ('ws_local_maxima', 'peaks', 'peaks'),
                ('ws_markers', 'markers', 'markers'), ('ws_invert_distance', 'negative_distance', 'negative_distance'),
                ('ws_masked_flood', 'labels', 'labels')):
            node = result.node(PREFIX+suffix)
            self.assertEqual(node.status, 'executed', node.reason)
            np.testing.assert_array_equal(node.outputs[port].cpu().numpy(), expected[reference])
            self.assertEqual(node.outputs[port].device, self.gray.device)
        count = result.node(PREFIX+'ws_marker_count_observation').outputs['marker_count']
        self.assertEqual(count['value'], len(np.unique(expected['markers']))-1)
        color = result.node(PREFIX+'analyze_color')
        self.assertEqual(color.status, 'executed', color.reason)
        self.assertEqual(color.outputs['histogram'].status, 'empty')
        self.assertEqual(color.outputs['histogram'].data_records(), observations['__returned_hue_chart__'])
        for label in color.outputs['observations'].labels:
            reference = observations[label.sample]
            for histogram in label.histograms:
                np.testing.assert_array_equal(histogram.values, reference[histogram.variable]['value'])
        size = result.node(PREFIX+'analyze_size')
        self.assert_size_observations(size, observations['__size_observations__'])
        np.testing.assert_array_equal(size.outputs['diagnostic'].cpu().numpy(),
            expected['size_diagnostic_rgb'])

    def test_mean_and_supported_otsu_complete_calculation_chains_match_public_calls(self):
        for kind in ('mean', 'otsu'):
            with self.subTest(kind=kind):
                expected, observations = self.reference(kind)
                result = self.run_recipe(selected_recipe(kind))
                prefix = 'pcv_seed_'+kind+'_'
                for suffix, port, reference in (('median_blur', 'image', 'median_blur'),
                        (kind+'_threshold', 'mask', 'binary_threshold'),
                        ('small_object_filter', 'mask', 'small_object_filter'),
                        ('erode', 'mask', 'erode'), ('dilate', 'mask', 'dilate'),
                        ('ws_distance_map', 'distance', 'distance'),
                        ('ws_local_maxima', 'coordinates', 'coordinates'),
                        ('ws_local_maxima', 'peaks', 'peaks'), ('ws_markers', 'markers', 'markers'),
                        ('ws_invert_distance', 'negative_distance', 'negative_distance'),
                        ('ws_masked_flood', 'labels', 'labels')):
                    node = result.node(prefix+suffix)
                    self.assertEqual(node.status, 'executed', node.reason)
                    np.testing.assert_array_equal(node.outputs[port].cpu().numpy(), expected[reference])
                color = result.node(prefix+'analyze_color')
                self.assertEqual(color.status, 'executed', color.reason)
                self.assertEqual(color.outputs['histogram'].data_records(), observations['__returned_hue_chart__'])
                for label in color.outputs['observations'].labels:
                    for histogram in label.histograms:
                        np.testing.assert_array_equal(histogram.values,
                            observations[label.sample][histogram.variable]['value'])
                size = result.node(prefix+'analyze_size')
                if kind == 'mean':
                    self.assertEqual(size.status, 'unavailable')
                    self.assertIn('orientation numerically unstable', size.reason)
                    self.assertNotIn('observations', size.outputs)
                else:
                    self.assert_size_observations(size, observations['__size_observations__'])
                    self.assertEqual(result.status, 'executed')
                    np.testing.assert_array_equal(size.outputs['diagnostic'].cpu().numpy(),
                        expected['size_diagnostic_rgb'])

    def test_actual_hue_chart_is_a_complete_typed_recipe_output(self):
        _, observations = self.reference(space='all')
        result = self.run_recipe(selected_recipe(**{'analysis.color': {'colorspaces': 'all'}}))
        color = result.node(PREFIX+'analyze_color')
        self.assertEqual(color.status, 'executed', color.reason)
        self.assertEqual(color.outputs['histogram'].status, 'available')
        self.assertEqual(color.outputs['histogram'].data_records(), observations['__returned_hue_chart__'])
        self.assertEqual(len(color.outputs['histogram'].series), int(result.labels.max().item()))
        self.assertEqual(result.status, 'executed')

    def test_diagnostic_only_cap_keeps_observations_and_mutation_recomputes_size_only(self):
        from seedvision.benchmarking import native_recipe as implementation
        first = self.run_recipe()
        stage = PREFIX+'analyze_size'
        original = first.node(stage).outputs['diagnostic'].clone()
        first.node(stage).outputs['diagnostic'][0,0,0] ^= 255
        restored = self.run_recipe()
        self.assertEqual(self.cache.last_computed_nodes, (stage,))
        self.assertEqual(len(self.cache.last_reused_nodes), len(selected_recipe().stages)-1)
        torch.testing.assert_close(restored.node(stage).outputs['diagnostic'], original)
        runtime = implementation._runtime_records(self.gray.device)
        self.cache = PipelineAnalysisCache()
        with patch.object(implementation, '_runtime_records', return_value=runtime), \
                patch.object(implementation._size, 'render_plantcv_size_diagnostic_rgb',
                    side_effect=implementation._size.RasterSpanLimitError('Diagnostic cap')):
            partial = self.run_recipe()
        node = partial.node(stage)
        self.assertEqual(node.status, 'partial')
        self.assertEqual(set(node.outputs), {'observations'})
        self.assertEqual(node.outputs['observations'], restored.node(stage).outputs['observations'])
        self.assertEqual(node.reason, 'Unavailable outputs: diagnostic')
        self.assertEqual(partial.status, 'partial')

    def test_incremental_diagnostic_geometry_cap_releases_collection_but_keeps_all_observations(self):
        from seedvision.benchmarking import native_recipe as implementation
        recipe = selected_recipe(**{'analysis.size': {'n_labels': 3}})
        runtime = implementation._runtime_records(self.gray.device)
        measured = []
        original = implementation._size.premeasure_plantcv_binary_mask
        def record(*args, **kwargs):
            value = original(*args, **kwargs)
            measured.append(value)
            return value
        with patch.object(implementation, '_runtime_records', return_value=runtime), \
                patch.object(implementation._size, 'premeasure_plantcv_binary_mask', side_effect=record):
            normal = self.run_recipe(recipe)
        self.assertEqual(len(measured), 3)
        counts = [sum(len(c) for c in item.boundary_topology.contours_xy or ())
            for item in measured]
        boundary_counts = [int(item.boundary_topology.xy_neighbor_bits.shape[0])
            for item in measured]
        cap = max(counts[:2] + boundary_counts[:2])
        self.assertGreater(max(sum(counts[:2]), sum(boundary_counts[:2])), cap)
        self.cache = PipelineAnalysisCache()
        preflight = implementation._size._preflight_size_diagnostic_points
        with patch.object(implementation, '_runtime_records', return_value=runtime), \
                patch.object(implementation._size, 'MAX_SIZE_DIAGNOSTIC_POINTS', cap), \
                patch.object(implementation._size, '_preflight_size_diagnostic_points', wraps=preflight) as checked, \
                patch.object(implementation._size, 'render_plantcv_size_diagnostic_rgb') as rendered:
            limited = self.run_recipe(recipe)
        # The second label crosses the shared cap; the third still contributes
        # scalar observations but is no longer collected for rendering.
        self.assertEqual(checked.call_count, 2)
        rendered.assert_not_called()
        node = limited.node(PREFIX+'analyze_size')
        self.assertEqual(node.status, 'partial')
        self.assertEqual(set(node.outputs), {'observations'})
        self.assertEqual(len(node.outputs['observations']), 3)
        self.assertEqual(node.outputs['observations'],
            normal.node(PREFIX+'analyze_size').outputs['observations'])

    def test_repeat_layout_and_changed_controls_recompute_only_their_dependents(self):
        recipe = selected_recipe()
        first = self.run_recipe(recipe)
        repeat = self.run_recipe(recipe)
        self.assertEqual(self.cache.last_computed_nodes, ())
        self.assertEqual(len(self.cache.last_reused_nodes), len(recipe.stages))
        self.assertIs(first.labels, repeat.labels)
        moved = replace(recipe, layout=tuple(replace(item, x=item.x+42) for item in recipe.layout))
        self.run_recipe(moved)
        self.assertEqual(self.cache.last_computed_nodes, ())
        color_changed = self.run_recipe(selected_recipe(**{'analysis.color': {'colorspaces': 'lab'}}))
        self.assertEqual(self.cache.last_computed_nodes, (PREFIX+'analyze_color',))
        self.assertIs(first.labels, color_changed.labels)
        peak_changed = self.run_recipe(selected_recipe(**{
            'analysis.color': {'colorspaces': 'lab'},
            'reference.internal.watershed.local_maxima': {'distance': 14}}))
        expected = {PREFIX+suffix for suffix in ('ws_local_maxima', 'ws_markers', 'ws_masked_flood',
            'ws_marker_count_observation', 'analyze_color', 'analyze_size')}
        self.assertEqual(set(self.cache.last_computed_nodes), expected)
        self.assertTrue(peak_changed.node(PREFIX+'ws_distance_map').reused)
        self.assertTrue(peak_changed.node(PREFIX+'ws_invert_distance').reused)

    def test_size_controls_change_only_size_observations_and_cancel_reaches_primitive(self):
        from seedvision.benchmarking import native_recipe as implementation
        first = self.run_recipe()
        changed = self.run_recipe(selected_recipe(**{'analysis.size': {'n_labels': 1, 'label': 'selected'}}))
        self.assertEqual(self.cache.last_computed_nodes, (PREFIX+'analyze_size',))
        self.assertIs(changed.labels, first.labels)
        values = changed.node(PREFIX+'analyze_size').outputs['observations']
        self.assertEqual(set(values), {'selected_1'})
        self.assertEqual(values['selected_1']['area']['value'], int(torch.count_nonzero(first.labels == 1)))
        callbacks = []
        def callback():
            callbacks.append(True)
            return False
        def cancelled(mask, *, cancel, line_thickness):
            count = len(callbacks)
            self.assertFalse(cancel())
            self.assertEqual(len(callbacks), count + 1)
            self.assertEqual(line_thickness, 5)
            raise implementation._size.PlantCVSizeCancelledError('Stopped inside size')
        with patch.object(implementation._size, 'premeasure_plantcv_binary_mask', side_effect=cancelled):
            with self.assertRaises(implementation._size.PlantCVSizeCancelledError):
                implementation._dispatch('analysis.size', {'labels': first.labels, 'image': self.rgb},
                    {'n_labels': 1, 'label': 'selected'}, callback)

    def test_mutated_cached_output_and_failed_node_never_reuse_stale_descendants(self):
        original = self.run_recipe()
        distance = original.node(PREFIX+'ws_distance_map').outputs['distance']
        distance.fill_(99)
        recovered = self.run_recipe()
        self.assertFalse(recovered.node(PREFIX+'ws_distance_map').reused)
        self.assertFalse(recovered.node(PREFIX+'ws_masked_flood').reused)
        self.assertTrue(recovered.node(PREFIX+'dilate').reused)
        self.assertFalse(bool((recovered.node(PREFIX+'ws_distance_map').outputs['distance'] == 99).all().item()))
        failed = self.run_recipe(selected_recipe(**{'filter.median_blur': {'ksize': None}}))
        self.assertEqual(failed.node(PREFIX+'median_blur').status, 'failed')
        self.assertEqual(failed.node(PREFIX+'ws_masked_flood').status, 'not_run')
        self.assertIsNone(failed.labels)
        self.assertFalse(failed.node(PREFIX+'analyze_color').outputs)

    def test_mean_precision_failure_never_reuses_previous_mask_or_descendants(self):
        recipe = selected_recipe('mean',**{
            'filter.median_blur':{'ksize':1},
            'threshold.mean':{'ksize':255,'offset':2,'object_type':'dark'},
        })
        frame = replace(FRAME,shape=(255,3))
        gray = torch.full(frame.shape,255,dtype=torch.uint8,device='cuda')
        rgb = gray[:,:,None].expand(255,3,3).contiguous()
        def execute():
            return run_native_recipe(recipe,gray,rgb,source_sha256=SOURCE,
                coordinate_frame=frame,preparation=PREPARATION,cache=self.cache)
        first = execute()
        stage = next(s.stage_id for s in recipe.stages if s.operation_id=='threshold.mean')
        self.assertEqual(first.node(stage).status,'executed')
        gray[0,0],gray[1,0],gray[127,2] = 0,254,253
        failed = execute()
        self.assertEqual(failed.node(stage).status,'failed')
        self.assertIn('AdaptiveMeanPrecisionError',failed.node(stage).reason)
        self.assertFalse(failed.node(stage).outputs)
        self.assertIsNone(failed.labels)
        after = [s.stage_id for s in recipe.stages][[s.stage_id for s in recipe.stages].index(stage)+1:]
        self.assertTrue(all(failed.node(name).status=='not_run' and not failed.node(name).outputs
            for name in after))
        gray.fill_(255)
        recovered = execute()
        self.assertEqual(recovered.node(stage).status,'executed')
        self.assertFalse(recovered.node(stage).reused)
        self.assertTrue(recovered.node(stage).outputs)

    def test_source_frame_input_mutation_and_preparation_are_part_of_reuse_identity(self):
        first = self.run_recipe()
        self.gray[40, 40] = 0
        changed = self.run_recipe()
        self.assertFalse(changed.node(PREFIX+'median_blur').reused)
        self.assertFalse(changed.node(PREFIX+'ws_masked_flood').reused)
        frame = replace(FRAME, transform_sha256='b'*64)
        moved = run_native_recipe(selected_recipe(), self.gray, self.rgb, source_sha256=SOURCE,
            coordinate_frame=frame, preparation=PREPARATION, cache=self.cache)
        self.assertFalse(moved.node(PREFIX+'median_blur').reused)
        self.assertNotEqual(first.coordinate_frame, moved.coordinate_frame)

    def test_preflight_and_real_mid_recipe_cancellation_return_no_complete_result(self):
        from seedvision.benchmarking import native_recipe as module
        with patch.object(module, 'MAX_RETAINED_CUDA_BYTES', 1), patch.object(module, '_runtime_records') as identity:
            with self.assertRaisesRegex(NativeRecipeError, 'byte budget'):
                self.run_recipe()
            identity.assert_not_called()
        with self.assertRaisesRegex(NativeRecipeError, 'canonical'):
            self.run_recipe(replace(selected_recipe(), dependencies=tuple(reversed(selected_recipe().dependencies))))
        with self.assertRaisesRegex(NativeRecipeError, 'uint8'):
            run_native_recipe(selected_recipe(), self.gray.cpu(), self.rgb, source_sha256=SOURCE,
                coordinate_frame=FRAME, preparation=PREPARATION)
        checks = 0
        def cancel():
            nonlocal checks
            checks += 1
            return checks >= 16
        with self.assertRaises(NativeRecipeCancelled):
            self.run_recipe(cancel=cancel)
        self.assertTrue(self.cache.values, 'Completed predecessors may remain cached after cancellation.')
        resumed = self.run_recipe()
        self.assertIsNotNone(resumed.labels)

    def test_file_preparation_reuses_actual_pixels_and_rejects_mutated_prepared_cache(self):
        import cv2
        from seedvision.persistence.reference_regions import file_sha256
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'source.png'
            gray, rgb = fixture()
            self.assertTrue(cv2.imwrite(str(path), rgb[..., ::-1]))
            first = prepare_native_recipe_inputs(path, cache=self.cache)
            np.testing.assert_array_equal(first.rgb.cpu().numpy(), rgb)
            np.testing.assert_array_equal(first.gray.cpu().numpy(), cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
            self.assertEqual(first.source_sha256, file_sha256(path))
            self.assertIn(str(cv2.__version__), first.preparation['rgb'])
            repeated = prepare_native_recipe_inputs(path, cache=self.cache)
            self.assertIs(first.gray, repeated.gray)
            first.gray.fill_(0)
            repaired = prepare_native_recipe_inputs(path, cache=self.cache)
            self.assertIsNot(first.gray, repaired.gray)
            np.testing.assert_array_equal(repaired.gray.cpu().numpy(), cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
            rgb[0, 0] = 255
            self.assertTrue(cv2.imwrite(str(path), rgb[..., ::-1]))
            changed = prepare_native_recipe_inputs(path, cache=self.cache)
            self.assertNotEqual(changed.source_sha256, first.source_sha256)


if __name__ == '__main__':
    unittest.main()
