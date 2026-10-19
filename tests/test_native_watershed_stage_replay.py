"""Exact saved watershed products and typed source/local archive contracts."""
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
import sys
import tempfile
import unittest
from tests.catalogue_fixtures import CatalogueTestCase
from unittest.mock import patch

import cv2
import torch

from seedvision.benchmarking import native_recipe as runner
from seedvision.benchmarking import native_stage_replay as replay
from seedvision.export import native_results as archives
from tests.test_native_recipe import FRAME, PREPARATION, SOURCE, fixture, selected_recipe


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class NativeWatershedStageReplayTests(CatalogueTestCase):
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
        from tests.observation import observe_functions
        with observe_functions(profile, runner._dispatch):
            result = replay.replay_native_stage(directory, stage)
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
