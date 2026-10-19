"""Typed saved-stage resource, cancellation and actual asynchronous Qt checks."""

from tests.catalogue_fixtures import CatalogueTestCase
from hashlib import sha256
import inspect
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest

import cv2
import torch

from seedvision.benchmarking import native_recipe as runner
from seedvision.benchmarking import native_stage_replay as replay
from seedvision.benchmarking.checkpoints import CoordinateFrame
from seedvision.export import native_results as archives
from tests.test_native_recipe import fixture, selected_recipe, FRAME, SOURCE, PREPARATION

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')


class TypedReplayPlanTests(CatalogueTestCase):
    def test_all_five_selected_byte_and_cuda_boundaries_preserve_full_fixture_admission(self):
        maximum = (replay.MAX_REPLAY_CUDA_BYTES - 128 * 1024 * 1024) // 128
        for operation in replay.WATERSHED_TYPED_OPERATIONS:
            parameters = {'distance': 8} if operation.endswith('local_maxima') else {}
            with self.subTest(operation=operation):
                plan = replay._plan((2080, 3120), operation, parameters, replay.MAX_TYPED_SELECTED_BYTES)
                self.assertEqual(plan['selected_limit_bytes'], 128 * 1024 * 1024)
                self.assertIsNone(plan['work_limit'])
                self.assertIn('native operation-specific', plan['work_details']['model'])
                with self.assertRaises(runner.NativeRecipeError):
                    replay._plan((2080, 3120), operation, parameters, replay.MAX_TYPED_SELECTED_BYTES + 1)
                self.assertEqual(replay._plan((1, maximum), operation, parameters, 1)['cuda_bytes'],
                                 replay.MAX_REPLAY_CUDA_BYTES)
                for shape in ((1, maximum + 1), (2160, 3840)):
                    with self.assertRaises(runner.NativeRecipeError):
                        replay._plan(shape, operation, parameters, 1)
        self.assertEqual(replay._plan((96, 128), 'threshold.binary', {}, 2 * 96 * 128)
                         ['selected_limit_bytes'], 48 * 1024 * 1024)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class TypedReplayIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.recipe = runner.build_native_seed_recipe(selected_recipe())
        gray, rgb = fixture()
        cls.saved_run = runner.run_native_recipe(cls.recipe, torch.tensor(gray, device='cuda'),
            torch.tensor(rgb, device='cuda'), source_sha256=SOURCE,
            coordinate_frame=FRAME, preparation=PREPARATION)
        cls.directory = cls.root / 'run'
        archives.export_native_recipe_result(cls.directory, cls.recipe, cls.saved_run)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def stage(self, operation, recipe=None):
        return next(stage.stage_id for stage in (recipe or self.recipe).stages
                    if stage.operation_id == operation)
