"""Saved calibrated-ROI replay: exact frames, ancestors and bounded transfers."""
from dataclasses import replace
from hashlib import sha256
import inspect
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from tests.catalogue_fixtures import CatalogueTestCase
from unittest.mock import patch

import cv2
import torch

from seedvision.benchmarking import native_calibrated_seed_recipe as factory
from seedvision.benchmarking import native_card_recipe as card
from seedvision.benchmarking import native_recipe as runner
from seedvision.benchmarking import native_stage_replay as replay
from seedvision.benchmarking.checkpoints import CoordinateFrame
from seedvision.export import native_results as archives
from tests.test_native_calibrated_seed_recipe import fixture, selected_recipe
from tests.test_native_card_recipe import _parameter

os.environ.setdefault('QT_QPA_PLATFORM','offscreen')


@unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
class NativeLocalStageReplayTests(CatalogueTestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(); cls.root = Path(cls.temp.name)
        cls.path = cls.root/'card-seeds.png'
        if not cv2.imwrite(str(cls.path),fixture()):
            raise RuntimeError('Fixture could not be saved.')
        cls.recipe = selected_recipe()
        cls.saved_run = runner.run_native_recipe_path(cls.recipe,cls.path)
        cls.directory = cls.root/'run'
        archives.export_native_recipe_result(cls.directory,cls.recipe,cls.saved_run)

    @classmethod
    def tearDownClass(cls): cls.temp.cleanup()

    def stage(self, operation, recipe=None):
        return next(item.stage_id for item in (recipe or self.recipe).stages if item.operation_id==operation)

    def hashes(self, directory=None):
        directory = directory or self.directory
        return {str(path.relative_to(directory)):sha256(path.read_bytes()).hexdigest()
                for path in directory.rglob('*') if path.is_file()}

    def changed(self, stage_id, **changes):
        node = replace(self.saved_run.node(stage_id),**changes)
        run = replace(self.saved_run,status='partial' if node.status=='failed' else self.saved_run.status,
            nodes=tuple(node if item.stage_id==stage_id else item for item in self.saved_run.nodes))
        directory = self.root/f'changed-{len(tuple(self.root.iterdir()))}'
        archives.export_native_recipe_result(directory,self.recipe,run)
        return directory

    def assert_invalid_before_loading(self, directory, stage):
        hashes = self.hashes(directory)
        with patch.object(archives,'load_native_recipe_result') as load:
            result = replay.replay_native_stage(directory,stage)
        self.assertEqual(result.status,'invalid_archive',result.reason)
        self.assertFalse(result.outputs); load.assert_not_called()
        self.assertEqual(self.hashes(directory),hashes)
