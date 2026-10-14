"""Exact saved watershed products and typed source/local archive contracts."""
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import torch

from seedvision.benchmarking import native_recipe as runner
from seedvision.benchmarking import native_stage_replay as replay
from seedvision.export import native_results as archives
from tests.test_native_recipe import FRAME, PREPARATION, SOURCE, fixture, selected_recipe


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class NativeWatershedStageReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.recipe = runner.build_native_seed_recipe(selected_recipe())
        gray, rgb = fixture()
        cls.saved_run = runner.run_native_recipe(cls.recipe,
            torch.tensor(gray, device='cuda'), torch.tensor(rgb, device='cuda'),
            source_sha256=SOURCE, coordinate_frame=FRAME, preparation=PREPARATION)
        cls.directory = cls.root / 'source'
        archives.export_native_recipe_result(cls.directory, cls.recipe, cls.saved_run)
        from tests.test_native_calibrated_seed_recipe import fixture as card_fixture, selected_recipe as card_recipe
        cls.local_recipe = card_recipe()
        path = cls.root / 'calibrated.png'
        if not cv2.imwrite(str(path), card_fixture()):
            raise RuntimeError('Calibrated fixture could not be saved.')
        cls.local_run = runner.run_native_recipe_path(cls.local_recipe, path)
        cls.local_directory = cls.root / 'local'
        archives.export_native_recipe_result(cls.local_directory, cls.local_recipe, cls.local_run)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def stage(self, suffix, recipe=None):
        return next(stage.stage_id for stage in (recipe or self.recipe).stages
                    if stage.operation_id == 'reference.internal.watershed.' + suffix)

    def hashes(self, directory):
        return {str(path.relative_to(directory)): sha256(path.read_bytes()).hexdigest()
                for path in directory.rglob('*') if path.is_file()}

    def changed(self, stage, *, local=False, **changes):
        recipe, saved = (self.local_recipe, self.local_run) if local else (self.recipe, self.saved_run)
        node = replace(saved.node(stage), **changes)
        changed_run = replace(saved, nodes=tuple(node if item.stage_id == stage else item for item in saved.nodes))
        directory = self.root / f'changed-{len(tuple(self.root.iterdir()))}'
        archives.export_native_recipe_result(directory, recipe, changed_run)
        return directory

    def recorded_attempt(self, directory, stage):
        calls = []
        def profile(frame, event, arg):
            if event == 'call' and frame.f_code is runner._dispatch.__code__:
                calls.append((frame.f_locals['operation'], set(frame.f_locals['inputs'])))
        previous = sys.getprofile()
        sys.setprofile(profile)
        try:
            result = replay.replay_native_stage(directory, stage)
        finally:
            sys.setprofile(previous)
        return result, calls

    def assert_invalid(self, directory, stage, *, before_load=False):
        before = self.hashes(directory)
        if before_load:
            with patch.object(archives, 'load_native_recipe_result') as load:
                result = replay.replay_native_stage(directory, stage)
            load.assert_not_called()
        else:
            result, calls = self.recorded_attempt(directory, stage)
            self.assertEqual(calls, [])
        self.assertEqual(result.status, 'invalid_archive', result.reason)
        self.assertFalse(result.outputs)
        self.assertEqual(self.hashes(directory), before)

    def test_all_five_source_and_calibrated_stages_compare_every_output_dispatch_only_target_and_preserve_archive(self):
        for recipe, saved, directory, scope in ((self.recipe, self.saved_run, self.directory, 'source'),
                (self.local_recipe, self.local_run, self.local_directory, 'calibrated_seed_roi')):
            before = self.hashes(directory)
            for operation in replay.WATERSHED_TYPED_OPERATIONS:
                stage = next(item.stage_id for item in recipe.stages if item.operation_id == operation)
                with self.subTest(scope=scope, operation=operation):
                    result, calls = self.recorded_attempt(directory, stage)
                    self.assertEqual(result.status, 'matched', result.reason)
                    self.assertEqual(calls, [(operation, set(replay.SUPPORTED_REPLAY_OPERATIONS[operation][0]))])
                    self.assertEqual(result.metadata['frame_scope'], scope)
                    self.assertEqual(tuple(result.metadata['coordinate_frame']['shape']), (96, 128))
                    self.assertFalse(result.scientifically_validated)
                    self.assertEqual(set(result.outputs), set(saved.node(stage).outputs))
                    self.assertEqual(set(result.metadata['comparison']['by_port']), set(result.outputs))
                    for name, expected in saved.node(stage).outputs.items():
                        actual = result.outputs[name]
                        if torch.is_tensor(expected):
                            self.assertEqual(actual.device.type, 'cuda')
                            self.assertTrue(torch.equal(actual, expected))
                        else:
                            self.assertEqual(actual, expected)
                        self.assertEqual(result.metadata['comparison']['by_port'][name]['mismatched_elements'], 0)
                    if scope != 'source':
                        self.assertEqual(result.metadata['frame_lineage']['local_to_parent_xy'], (600, 260))
            self.assertEqual(self.hashes(directory), before)

    def test_every_maxima_flood_and_count_branch_key_is_verified_before_loading(self):
        for local, recipe, saved in ((False, self.recipe, self.saved_run),
                                     (True, self.local_recipe, self.local_run)):
            for suffix in ('local_maxima', 'masked_flood', 'marker_count_observation'):
                stage = self.stage(suffix, recipe)
                node = saved.node(stage)
                for port in node.input_keys:
                    with self.subTest(local=local, suffix=suffix, port=port):
                        directory = self.changed(stage, local=local,
                            input_keys={**dict(node.input_keys), port: '0' * 64})
                        self.assert_invalid(directory, stage, before_load=True)
            # Count's completion edge traverses the entire flood branch, including RGB.
            flood = self.stage('masked_flood', recipe)
            node = saved.node(flood)
            self.assert_invalid(self.changed(flood, local=local,
                input_keys={**dict(node.input_keys), 'rgb': '0' * 64}),
                self.stage('marker_count_observation', recipe), before_load=True)

    def test_wrong_coordinate_float_bool_integer_and_rgb_headers_reject_before_loading(self):
        for local, recipe, saved in ((False, self.recipe, self.saved_run),
                                     (True, self.local_recipe, self.local_run)):
            cases = (
                ('local_maxima', 'coordinates', torch.zeros((2, 3), dtype=torch.int64, device='cuda'), 'local_maxima'),
                ('local_maxima', 'coordinates', torch.zeros((2, 2), dtype=torch.int32, device='cuda'), 'local_maxima'),
                ('local_maxima', 'peaks', torch.zeros((96, 128), dtype=torch.uint8, device='cuda'), 'connected_markers'),
                ('connected_markers', 'markers', torch.zeros((96, 128), dtype=torch.int64, device='cuda'), 'masked_flood'),
                ('negate_distance', 'negative_distance', torch.zeros((96, 128), dtype=torch.float64, device='cuda'), 'masked_flood'),
            )
            for suffix, port, value, selected in cases:
                stage = self.stage(suffix, recipe)
                with self.subTest(local=local, suffix=suffix, port=port):
                    self.assert_invalid(self.changed(stage, local=local,
                        outputs={**dict(saved.node(stage).outputs), port: value}),
                        self.stage(selected, recipe), before_load=True)
            flood = recipe.stage(self.stage('masked_flood', recipe))
            edge = next(item for item in recipe.dependencies if item.target_stage == flood.stage_id and item.target_port == 'rgb')
            source = saved.node(edge.source_stage)
            self.assert_invalid(self.changed(source.stage_id, local=local,
                outputs={**dict(source.outputs), edge.source_port: torch.zeros((95, 128, 3), dtype=torch.uint8, device='cuda')}),
                flood.stage_id, before_load=True)

    def test_peak_coordinates_require_in_frame_unique_mask_membership_and_stable_distance_order(self):
        stage = self.stage('local_maxima')
        original = self.saved_run.node(stage).outputs
        coords = original['coordinates']
        self.assertGreaterEqual(coords.shape[0], 2)
        variants = []
        outside = coords.clone(); outside[0, 0] = 96
        variants.append(('coordinates', outside))
        duplicate = coords.clone(); duplicate[1] = duplicate[0]
        variants.append(('coordinates', duplicate))
        variants.append(('coordinates', coords.flip(0).contiguous()))
        mask = original['peaks'].clone(); mask[coords[0, 0], coords[0, 1]] = False
        variants.append(('peaks', mask))
        for port, value in variants:
            with self.subTest(port=port, shape=tuple(value.shape)):
                self.assert_invalid(self.changed(stage, outputs={**dict(original), port: value}), stage)

    def test_completion_token_and_count_schema_are_exact_before_loading(self):
        flood = self.stage('masked_flood')
        count = self.stage('marker_count_observation')
        original = self.saved_run.node(flood).outputs
        for token in (False, 1):
            with self.subTest(token=token):
                directory = self.changed(flood, outputs={**dict(original), 'flood_complete': token})
                self.assert_invalid(directory, count, before_load=True)
                self.assert_invalid(directory, flood, before_load=True)
        record = dict(self.saved_run.node(count).outputs['marker_count'])
        for field, value in (('sample', 'other'), ('variable', 'count'), ('trait', 'other'),
                ('method', 'other'), ('scale', 'px'), ('datatype', 'int'), ('value', True), ('value', -1), ('value', 96 * 128)):
            with self.subTest(field=field, value=value):
                self.assert_invalid(self.changed(count, outputs={'marker_count': {**record, field: value}}),
                                    count, before_load=True)
        result = replay.replay_native_stage(self.changed(count,
            outputs={'marker_count': {**record, 'value': record['value'] + 1}}), count)
        self.assertEqual(result.status, 'mismatch', result.reason)
        self.assertEqual(result.outputs['marker_count'], record)
        self.assertEqual(result.metadata['comparison']['by_port']['marker_count']['mismatched_elements'], 1)

    def test_local_instance_labels_reject_negative_out_of_range_unknown_seed_background_and_changed_seeds(self):
        stage = self.stage('masked_flood', self.local_recipe)
        original = self.local_run.node(stage).outputs
        markers = self.local_run.node(self.stage('connected_markers', self.local_recipe)).outputs['markers']
        labels = original['labels']
        point = torch.nonzero((labels > 0) & (markers == 0))[0]
        seed = torch.nonzero(markers > 0)[0]
        background = torch.nonzero(labels == 0)[0]
        for location, value in ((point, -1), (point, 65_537), (point, 65_536),
                                (background, 1), (seed, 0)):
            with self.subTest(value=value, location=location.tolist()):
                changed = labels.clone(); changed[location[0], location[1]] = value
                self.assert_invalid(self.changed(stage, local=True,
                    outputs={**dict(original), 'labels': changed}), stage)

    def test_valid_alternate_seed_id_is_a_numeric_mismatch_and_returns_all_fresh_products(self):
        stage = self.stage('masked_flood')
        original = self.saved_run.node(stage).outputs
        markers = self.saved_run.node(self.stage('connected_markers')).outputs['markers']
        labels = original['labels'].clone()
        ids = torch.unique(markers); ids = ids[ids > 0]
        self.assertGreaterEqual(ids.numel(), 2)
        point = torch.nonzero((labels > 0) & (markers == 0))[0]
        current = labels[point[0], point[1]]
        labels[point[0], point[1]] = ids[0] if current != ids[0] else ids[1]
        result = replay.replay_native_stage(self.changed(stage,
            outputs={**dict(original), 'labels': labels}), stage)
        self.assertEqual(result.status, 'mismatch', result.reason)
        self.assertTrue(torch.equal(result.outputs['labels'], original['labels']))
        self.assertIs(result.outputs['flood_complete'], True)
        self.assertEqual(result.metadata['comparison']['mismatched_elements'], 1)

    def test_empty_actual_chain_replays_zero_coordinates_masks_markers_labels_and_count(self):
        gray = torch.zeros(FRAME.shape, dtype=torch.uint8, device='cuda')
        saved = runner.run_native_recipe(self.recipe, gray, gray[..., None].repeat(1, 1, 3),
            source_sha256=SOURCE, coordinate_frame=FRAME, preparation=PREPARATION)
        directory = self.root / 'empty'
        archives.export_native_recipe_result(directory, self.recipe, saved)
        for operation in replay.WATERSHED_TYPED_OPERATIONS:
            stage = next(item.stage_id for item in self.recipe.stages if item.operation_id == operation)
            result = replay.replay_native_stage(directory, stage)
            self.assertEqual(result.status, 'matched', result.reason)
            for port, value in result.outputs.items():
                if torch.is_tensor(value):
                    self.assertEqual(int(torch.count_nonzero(value)), 0)
                    if port == 'coordinates':
                        self.assertEqual(tuple(value.shape), (0, 2))
                elif port == 'marker_count':
                    self.assertEqual(value['value'], 0)
                else:
                    self.assertIs(value, True)
