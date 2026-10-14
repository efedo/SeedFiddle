"""Colour-card compact archive contracts and allocation-free replay admission."""
from collections.abc import Mapping
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import MappingProxyType
import unittest
from unittest.mock import patch

import cv2
import torch

from seedvision.benchmarking import native_card_recipe as factory, native_card_replay as replay
from seedvision.benchmarking import native_recipe as runner
from seedvision.benchmarking.checkpoints import CheckpointLimits, CoordinateFrame
from seedvision.export import native_results as archives
from tests.test_native_card_recipe import _parameter, fixture


SHAPE = (620, 460)
ROI = (40, 40, 420, 580)


def tagged(value):
    collector = archives._Collector(CoordinateFrame('source.decoded', SHAPE, 'top-left', 'x-right,y-down', 'pixel'),
                                    CheckpointLimits(), None)
    return collector.encode(value)


def geometry(count=24):
    """A structurally valid saved product; no detector result is asserted."""
    return {
        'detected_chips': count, 'median_color_chip_size': 2500.0,
        'median_color_chip_width': 50.0, 'median_color_chip_height': 50.0,
        'selected_contour_indices': tuple(range(count)), 'bounding_rect': (10, 10, 200, 300),
        'centers_xy': tuple((30 + i * 10, 30 + i * 10) for i in range(24)),
        'corners_xy': ((20, 20), (250, 20), (250, 400), (20, 400)),
        'transformation': ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        'areas': (2500.0,) * count, 'widths': (50.0,) * count, 'heights': (50.0,) * count,
        'opencv_version': factory.card.cv2.__version__, 'numpy_version': factory.np.__version__,
        'grid_box_sha256': factory.card.GRID_BOX_SHA256, 'reference_sources': factory.card.REFERENCE_SOURCES,
    }


class CardCompactContractTests(unittest.TestCase):
    def test_actual_archive_tags_preserve_mapping_sequence_types_and_source_order(self):
        value = MappingProxyType({'geometry': geometry(23), 'empty': (), 'list': [None, True, 'é\n"\\']})
        decoded = replay.decode(tagged(value))
        self.assertEqual(decoded, replay.canonical(value))
        self.assertIsInstance(decoded, Mapping)
        self.assertIs(type(decoded['empty']), tuple)
        self.assertIs(type(decoded['list']), list)
        self.assertEqual(decoded['geometry']['selected_contour_indices'], tuple(range(23)))
        with self.assertRaises(TypeError):
            decoded['new'] = 1

    def test_complete_escaped_json_byte_boundary_including_empty_container_envelopes(self):
        for value in (None, False, 2**255, -0.0, 'é😀\x00\n"\\', (), [], {}, {'a\n': [True, ()]}):
            binding = tagged(value)
            size = len(json.dumps(binding, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf-8'))
            with self.subTest(value=value), patch.object(replay, 'MAX_METADATA_BYTES', size):
                self.assertEqual(replay.decode(binding), replay.canonical(value))
            with self.subTest(value=value), patch.object(replay, 'MAX_METADATA_BYTES', size - 1):
                with self.assertRaises(replay.NativeCardReplayError):
                    replay.decode(binding)
                with self.assertRaises(replay.NativeCardReplayError):
                    replay.canonical(value)

    def test_malformed_tags_duplicates_nonfinite_and_unsupported_types_reject(self):
        bad = (
            {'kind': 'array', 'name': 'array_0000'}, {'kind': 'record', 'type': 'unknown', 'fields': {}},
            {'kind': 'tuple', 'items': [], 'extra': 1}, {'kind': ['tuple'], 'items': []},
            {'kind': 'list', 'items': ()}, {'kind': 'mapping', 'items': [['key', 1], ['key', 2]]},
            {'kind': 'mapping', 'items': [[True, 1]]}, {'kind': 'mapping', 'items': [['key']]},
            float('nan'), float('inf'), 2**256, '\ud800',
        )
        for value in bad:
            with self.subTest(value=value), self.assertRaises(replay.NativeCardReplayError):
                replay.decode(value)
        for value in (object(), {1: 'not a string key'}, {1, 2}):
            with self.subTest(value=value), self.assertRaises(replay.NativeCardReplayError):
                replay.canonical(value)

    def test_depth_items_cycles_and_pretraversal_container_limits(self):
        cyclic = []; cyclic.append(cyclic)
        with self.assertRaisesRegex(replay.NativeCardReplayError, 'cycle'):
            replay.canonical(cyclic)
        nested = 0
        for _ in range(replay.MAX_METADATA_DEPTH + 1):
            nested = (nested,)
        with self.assertRaisesRegex(replay.NativeCardReplayError, 'depth'):
            replay.decode(tagged(nested))
        with patch.object(replay, 'MAX_METADATA_ITEMS', 1):
            self.assertEqual(replay.decode(tagged(())), ())
            with self.assertRaises(replay.NativeCardReplayError):
                replay.decode(tagged((1,)))
            class LargeMapping(Mapping):
                def __len__(self): return 2
                def __iter__(self): raise AssertionError('must reject before traversal')
                def __getitem__(self, key): raise AssertionError('must not access values')
            with self.assertRaises(replay.NativeCardReplayError):
                replay.canonical(LargeMapping())
        with patch.object(replay, 'MAX_METADATA_BYTES', 5):
            with self.assertRaises(replay.NativeCardReplayError):
                replay.decode('x' * 6)

    def test_long_text_traversal_can_cancel_before_returning_compact_result(self):
        calls = []
        def poll():
            calls.append(True)
            raise RuntimeError('cancelled')
        with self.assertRaisesRegex(RuntimeError, 'cancelled'):
            replay.decode('é' * 4096, poll)
        self.assertEqual(len(calls), 1)

    def test_geometry_accepts_non24_count_outside_centers_and_roi_edge_bbox(self):
        for count in (1, 23, 24, 25):
            value = geometry(count)
            value['centers_xy'] = ((-100, 900), *value['centers_xy'][1:])
            value['bounding_rect'] = (0, 0, ROI[2]-ROI[0], ROI[3]-ROI[1])
            actual = replay.validate(replay.decode(tagged(value)), 'geometry', SHAPE, ROI)
            self.assertEqual(actual['detected_chips'], count)
            self.assertEqual(actual['centers_xy'][0], (-100, 900))

    def test_geometry_shapes_integer_domains_local_frame_and_provenance_reject(self):
        mutations = (
            lambda v: v.update(detected_chips=True), lambda v: v.update(detected_chips=0),
            lambda v: v.update(median_color_chip_width=0), lambda v: v.update(areas=(1.0,)),
            lambda v: v.update(selected_contour_indices=(0,) * 24),
            lambda v: v.update(selected_contour_indices=tuple(reversed(range(24)))),
            lambda v: v.update(bounding_rect=(0, 0, 381, 540)),
            lambda v: v.update(corners_xy=((380, 20), *v['corners_xy'][1:])),
            lambda v: v.update(centers_xy=((2**31, 20), *v['centers_xy'][1:])),
            lambda v: v.update(centers_xy=((1.0, 20), *v['centers_xy'][1:])),
            lambda v: v.update(transformation=((1.0,),) * 3), lambda v: v.update(grid_box_sha256='0' * 64),
            lambda v: v.update(reference_sources=tuple(reversed(factory.card.REFERENCE_SOURCES))),
            lambda v: v.update(opencv_version='other'), lambda v: v.update(extra=0),
        )
        for mutate in mutations:
            value = geometry(); mutate(value)
            with self.subTest(mutate=mutate), self.assertRaises(replay.NativeCardReplayError):
                replay.validate(value, 'geometry', SHAPE, ROI)

    def test_changed_welltyped_numbers_remain_mismatches_and_sequence_shapes_are_honest(self):
        value = geometry(); expected = deepcopy(value); expected['median_color_chip_width'] = 51.0
        replay.validate(expected, 'geometry', SHAPE, ROI)
        compared = replay.compare(value, expected)
        self.assertEqual(compared['mismatched_elements'], 1)
        self.assertGreater(compared['compared_elements'], 150)
        self.assertTrue(compared['shape_equal'])
        for changed in ([1, 2], (1,), {'a': 1}):
            self.assertFalse(replay.compare((1, 2), changed)['shape_equal'])
        self.assertEqual(replay.compare({'empty': ()}, {'empty': ()})['mismatched_elements'], 0)

    def test_scale_absent_and_anisotropic_and_matrix_source_domains(self):
        for value in ({'px_width': None, 'px_height': None, 'unit': None},
                      {'px_width': .14, 'px_height': .22, 'unit': 'mm'}):
            self.assertEqual(replay.validate(value, 'scale', SHAPE, ROI), value)
        for value in ({'px_width': 0, 'px_height': .2, 'unit': 'mm'},
                      {'px_width': None, 'px_height': None, 'unit': 'mm'}):
            with self.assertRaises(replay.NativeCardReplayError):
                replay.validate(value, 'scale', SHAPE, ROI)
        self.assertEqual(replay.validate((), 'chip_matrix', SHAPE), ())
        rows = ((-1000.0, 0.0, .5, 1.0),) * 2
        self.assertEqual(replay.validate(rows, 'fit_matrix', SHAPE), rows)
        factory.correction._matrices(rows, rows)  # IDs are ignored; duplicate RGB rows permit a singular affine fit.
        coefficients = ((-5.0, 8.0, 0.0),) * 4
        self.assertEqual(replay.validate(coefficients, 'affine_matrix', SHAPE), coefficients)
        for value, role in (((), 'fit_matrix'), (rows, 'standard_matrix'),
                            (((10.0, -1.0, 0.0, 0.0),), 'chip_matrix'), (coefficients[:3], 'affine_matrix')):
            with self.subTest(role=role), self.assertRaises(replay.NativeCardReplayError):
                replay.validate(value, role, SHAPE)

    def test_controls_reuse_source_domains_and_keep_every_key_exact(self):
        valid = {
            'color_card.bgr': {}, 'color_card.preprocess': {'roi_bounds_xyxy': None, 'adaptive_method': 0, 'block_size': 3},
            'color_card.geometry': {'min_size': 0, 'aspect_ratio': .5, 'solidity': 0}, 'color_card.mask': {'radius': 0},
            'color_card.scale': {'color_chip_size': [True, 2]}, 'color_card.means': {},
            'color_card.standard': {'pos': 0}, 'color_card.affine': {},
        }
        for operation, parameters in valid.items():
            self.assertIs(replay.controls(operation, parameters), parameters)
            with self.assertRaises(replay.NativeCardReplayError):
                replay.controls(operation, {**parameters, 'extra': 1})
        for name in ('classic', 'PASSPORT', 'cameratrax', 'nano', 'mini', None, [7, 11]):
            replay.controls('color_card.scale', {'color_chip_size': name})
        invalid = (
            ('color_card.preprocess', {**valid['color_card.preprocess'], 'block_size': 4}),
            ('color_card.preprocess', {**valid['color_card.preprocess'], 'adaptive_method': True}),
            ('color_card.geometry', {**valid['color_card.geometry'], 'solidity': 1.1}),
            ('color_card.geometry', {**valid['color_card.geometry'], 'aspect_ratio': 0}),
            ('color_card.mask', {'radius': True}), ('color_card.mask', {'radius': 4097}),
            ('color_card.scale', {'color_chip_size': [0, 1]}), ('color_card.standard', {'pos': 1.0}),
        )
        for operation, parameters in invalid:
            with self.subTest(operation=operation), self.assertRaises(replay.NativeCardReplayError):
                replay.controls(operation, parameters)


class CardReplayAdmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        recipe = factory.build_native_card_recipe()
        cls.parameters = {stage.operation_id: recipe.parameters_for_runner(stage.stage_id)
                          for stage in recipe.stages if stage.operation_id in replay.PORTS}

    def test_full_and_local_raster_specs_retain_float64_labels(self):
        self.assertEqual(replay.raster_spec('roi_bgr', SHAPE, ROI), ((540, 380, 3), '|u1'))
        self.assertEqual(replay.raster_spec('source_rgb', SHAPE, ROI), ((620, 460, 3), '|u1'))
        self.assertEqual(replay.raster_spec('chip_labels', SHAPE, ROI), (SHAPE, '<f8'))
        other = (0, 0, 380, 540)
        self.assertEqual(replay.raster_spec('roi_binary', SHAPE, ROI), replay.raster_spec('roi_binary', SHAPE, other))
        self.assertNotEqual(replay.frames(SHAPE, ROI)[2], replay.frames(SHAPE, other)[2])
        for shape, bounds in (((0, 10), None), ((True, 10), None), (SHAPE, (0, 0, 461, 620))):
            with self.assertRaises(replay.NativeCardReplayError):
                replay.frames(shape, bounds)

    def test_all_eight_plans_admit_without_cuda_or_numpy_allocation(self):
        with patch.object(torch, 'as_tensor', side_effect=AssertionError('no upload')), \
                patch.object(torch, 'empty', side_effect=AssertionError('no allocation')), \
                patch.object(torch, 'zeros', side_effect=AssertionError('no allocation')), \
                patch.object(factory.np, 'asarray', side_effect=AssertionError('no CPU arrays')):
            for operation, parameters in self.parameters.items():
                with self.subTest(operation=operation):
                    plan = replay.plan(operation, SHAPE, parameters, bounds=ROI)
                    self.assertGreaterEqual(plan.cuda_bytes, replay.AUDIT_CUDA_BYTES)
                    self.assertLessEqual(plan.selected_bytes, replay.MAX_SELECTED_BYTES)
        self.assertEqual(replay.plan('color_card.scale', SHAPE, self.parameters['color_card.scale']).selected_bytes, 0)

    def test_tight_limits_reject_before_runtime_or_allocation_and_small_allocations_round(self):
        for operation, parameters in self.parameters.items():
            planned = replay.plan(operation, SHAPE, parameters, bounds=ROI)
            self.assertEqual(replay.plan(operation, SHAPE, parameters, bounds=ROI,
                selected_limit=planned.selected_bytes, cuda_limit=planned.cuda_bytes), planned)
            with self.subTest(operation=operation), self.assertRaises(replay.NativeCardReplayUnavailable):
                replay.plan(operation, SHAPE, parameters, bounds=ROI, cuda_limit=planned.cuda_bytes-1)
            if planned.selected_bytes:
                with self.assertRaises(replay.NativeCardReplayUnavailable):
                    replay.plan(operation, SHAPE, parameters, bounds=ROI, selected_limit=planned.selected_bytes-1)
        # Three separately allocated one-pixel images and three comparison buffers.
        tiny = replay.plan('color_card.bgr', (1, 1), {})
        self.assertEqual(tiny.cuda_bytes, 6 * 512 + replay.AUDIT_CUDA_BYTES)
        for caps in ({'selected_limit': True}, {'cuda_limit': -1}):
            with self.assertRaises(replay.NativeCardReplayError):
                replay.plan('color_card.bgr', (1, 1), {}, **caps)

    def test_preprocess_uses_actual_roi_and_both_filter_source_admission_paths(self):
        for method in (0, 1):
            for bounds in (None, ROI):
                parameters = {'roi_bounds_xyxy': bounds, 'adaptive_method': method, 'block_size': 51}
                planned = replay.plan('color_card.preprocess', SHAPE, parameters)
                _, local, _ = replay.frames(SHAPE, bounds)
                work, scratch = factory.preprocess._preflight(*local, 51, factory.preprocess.MAX_TILE_SIDE,
                                                             method=method, complete=True)
                self.assertEqual(planned.work, work)
                self.assertEqual(planned.primitive_scratch_bytes, scratch)
                self.assertEqual(planned.selected_bytes, 3 * 620 * 460 + 6 * local[0] * local[1])
        with patch.object(factory.card, 'MAX_RETAINED_BYTES', 1):
            with self.assertRaises(replay.NativeCardReplayUnavailable):
                replay.plan('color_card.preprocess', SHAPE, self.parameters['color_card.preprocess'])

    def test_source_contour_row_circle_and_pixel_caps_are_separate_from_archive_caps(self):
        with self.assertRaisesRegex(replay.NativeCardReplayUnavailable, 'row/scratch'):
            replay.plan('color_card.geometry', (1_000_001, 1), self.parameters['color_card.geometry'])
        with self.assertRaises(replay.NativeCardReplayUnavailable):
            replay.plan('color_card.mask', (3000, 3000), {'radius': 4096})
        with self.assertRaisesRegex(replay.NativeCardReplayUnavailable, 'pixel'):
            replay.plan('color_card.scale', (64_000_001, 1), {'color_chip_size': None})
        # The float64 label alone crosses the checkpoint cap while the primitive
        # remains below64M pixels. Raising a caller's selected limit cannot admit it.
        with self.assertRaisesRegex(replay.NativeCardReplayUnavailable, 'checkpoint'):
            replay.plan('color_card.mask', (1, 16_777_217), {'radius': 0}, selected_limit=1024**3)
        maximum = replay.MAX_SELECTED_BYTES // 12
        replay.plan('color_card.mask', (1, maximum), {'radius': 0})
        with self.assertRaisesRegex(replay.NativeCardReplayUnavailable, 'selected-array'):
            replay.plan('color_card.mask', (1, maximum + 1), {'radius': 0})


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class ActualCardProductContractTests(unittest.TestCase):
    def test_all_actual_compact_outputs_roundtrip_for_full_frame_and_roi(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'card.png'
            self.assertTrue(cv2.imwrite(str(path), fixture(varied=True)))
            for bounds in (None, ROI):
                recipe = _parameter(factory.build_native_card_recipe(), factory.PREPROCESS, 'roi_bounds_xyxy', bounds)
                result = runner.run_native_recipe_path(recipe, path)
                self.assertEqual(result.status, 'executed', [(n.stage_id, n.reason) for n in result.nodes])
                for stage in recipe.stages:
                    if stage.operation_id not in replay.PORTS:
                        continue
                    _, outputs, _ = replay.PORTS[stage.operation_id]
                    for port, role in outputs.items():
                        value = result.node(stage.stage_id).outputs[port]
                        with self.subTest(bounds=bounds, operation=stage.operation_id, port=port):
                            if role in replay.RASTER_ROLES:
                                shape, dtype = replay.raster_spec(role, SHAPE, bounds)
                                self.assertEqual(tuple(value.shape), shape)
                                self.assertEqual(value.dtype, torch.float64 if dtype == '<f8' else torch.uint8)
                                self.assertEqual(value.device.type, 'cuda')
                            else:
                                decoded = replay.validate(replay.decode(tagged(value)), role, SHAPE, bounds)
                                compared = replay.compare(value, decoded)
                                self.assertEqual(compared['mismatched_elements'], 0)
                                self.assertTrue(compared['shape_equal'])


if __name__ == '__main__':
    unittest.main()
