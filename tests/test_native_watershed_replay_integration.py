"""Typed saved-stage resource, cancellation and actual asynchronous Qt checks."""
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


class TypedReplayPlanTests(unittest.TestCase):
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

    def test_every_typed_stage_can_cancel_after_upload_before_dispatch(self):
        for operation in replay.WATERSHED_TYPED_OPERATIONS:
            calls, stopped = [], []
            def profile(frame, event, arg):
                if event == 'call' and frame.f_code is runner._dispatch.__code__:
                    calls.append(frame.f_locals['operation'])
            def cancel():
                frame = inspect.currentframe()
                while frame is not None:
                    if frame.f_code is replay.replay_native_stage.__code__ and 'inputs' in frame.f_locals:
                        stopped.append(True)
                        return True
                    frame = frame.f_back
                return False
            with self.subTest(operation=operation):
                sys.setprofile(profile)
                try:
                    result = replay.replay_native_stage(self.directory, self.stage(operation), cancel=cancel)
                finally:
                    sys.setprofile(None)
                self.assertTrue(stopped)
                self.assertEqual(result.status, 'cancelled', result.reason)
                self.assertFalse(result.outputs)
                self.assertEqual(calls, [])

    def test_second_inputs_expected_and_fresh_output_mutations_withhold_all_outputs(self):
        cases = (
            ('local_maxima', 'inputs', 'mask'),
            ('local_maxima', 'expected_products', 'peaks'),
            ('masked_flood', 'inputs', 'rgb'),
            ('masked_flood', 'outputs', 'labels'),
        )
        for suffix, group, port in cases:
            changed = []
            def mutate():
                frame = inspect.currentframe()
                while frame is not None:
                    if (frame.f_code is replay.replay_native_stage.__code__
                            and group in frame.f_locals and not changed):
                        value = frame.f_locals[group][port]
                        if value.dtype == torch.bool:
                            value.logical_not_()
                        else:
                            value.add_(1)
                        changed.append(True)
                    frame = frame.f_back
                return False
            with self.subTest(suffix=suffix, group=group, port=port):
                result = replay.replay_native_stage(self.directory,
                    self.stage('reference.internal.watershed.' + suffix), cancel=mutate)
                self.assertTrue(changed, result.reason)
                self.assertEqual(result.status, 'failed', result.reason)
                self.assertFalse(result.outputs)

    def test_cancellation_inside_peak_and_flood_calculations_withholds_partial_products(self):
        for suffix, primitive in (('local_maxima', runner._peaks._greedy_spacing),
                                  ('masked_flood', runner._watershed.masked_watershed)):
            reached = []
            def cancel():
                frame = inspect.currentframe()
                while frame is not None:
                    if frame.f_code is primitive.__code__:
                        reached.append(True)
                        return True
                    frame = frame.f_back
                return False
            with self.subTest(operation=suffix):
                result = replay.replay_native_stage(self.directory,
                    self.stage('reference.internal.watershed.' + suffix), cancel=cancel)
                self.assertTrue(reached, result.reason)
                self.assertEqual(result.status, 'cancelled', result.reason)
                self.assertFalse(result.outputs)

    def test_qt_actual_maxima_empty_maxima_and_count_use_declared_preview_and_keep_active_run(self):
        from PySide6.QtGui import QFont
        from PySide6.QtWidgets import QApplication
        from seedvision.ui.native_recipe_dialog import NativeRecipeDialog
        application = QApplication.instance() or QApplication([])
        application.setFont(QFont('Segoe UI', 9))
        empty_recipe = runner.build_native_seed_recipe(selected_recipe(**{'threshold.binary': {'threshold': 255}}))
        gray, rgb = fixture()
        empty_run = runner.run_native_recipe(empty_recipe, torch.tensor(gray, device='cuda'),
            torch.tensor(rgb, device='cuda'), source_sha256=SOURCE,
            coordinate_frame=FRAME, preparation=PREPARATION)
        empty_directory = self.root / 'empty-ui'
        archives.export_native_recipe_result(empty_directory, empty_recipe, empty_run)
        for recipe, saved_run, directory, suffix in (
            (self.recipe, self.saved_run, self.directory, 'local_maxima'),
            (empty_recipe, empty_run, empty_directory, 'local_maxima'),
            (self.recipe, self.saved_run, self.directory, 'marker_count_observation'),
        ):
            with self.subTest(suffix=suffix, empty=recipe is empty_recipe):
                before = {path.name: sha256(path.read_bytes()).hexdigest()
                          for path in directory.iterdir() if path.is_file()}
                dialog = NativeRecipeDialog(recipe)
                dialog.resize(1050, 650)
                dialog.show()
                dialog.result = saved_run
                stage = self.stage('reference.internal.watershed.' + suffix, recipe)
                dialog._select_stage(stage)
                try:
                    dialog.start_stage_replay(directory)
                    deadline = time.monotonic() + 45
                    while dialog._worker is not None and time.monotonic() < deadline:
                        application.processEvents()
                        time.sleep(.01)
                    self.assertIsNone(dialog._worker)
                    self.assertEqual(dialog.replay_result.status, 'matched', dialog.status_label.text())
                    self.assertIs(dialog.result, saved_run)
                    self.assertTrue(dialog.canvas.isEnabled())
                    if suffix == 'local_maxima':
                        self.assertEqual(dialog.replay_result.metadata['preview_port'], 'peaks')
                        self.assertFalse(dialog.preview.pixmap().isNull())
                        image_size = dialog.preview.pixmap().size()
                        self.assertAlmostEqual(image_size.width() / image_size.height(), 128 / 96, delta=.02)
                        if recipe is empty_recipe:
                            self.assertEqual(tuple(dialog.replay_result.outputs['coordinates'].shape), (0, 2))
                    else:
                        self.assertIsNone(dialog.replay_result.metadata['preview_port'])
                        self.assertIn('No image output', dialog.preview.text())
                        self.assertIn('estimated_object_count', dialog.facts.toPlainText())
                        self.assertLess(dialog.facts.toPlainText().index('Estimated object count'),
                                        dialog.facts.toPlainText().index('Replay details:'))
                        self.assertIn(str(saved_run.node(stage).outputs['marker_count']['value']),
                                      dialog.facts.toPlainText())
                    self.assertEqual({path.name: sha256(path.read_bytes()).hexdigest()
                        for path in directory.iterdir() if path.is_file()}, before)
                finally:
                    if dialog._worker is not None:
                        dialog.stop()
                        while dialog._worker is not None:
                            application.processEvents()
                            time.sleep(.01)
                    dialog.reject()
                    dialog.deleteLater()
                    application.processEvents()

    def test_actual_full_resolution_typed_stages_match_and_stay_in_cuda_envelope(self):
        source = Path(__file__).parents[1] / 'images' / 'IMG_9632c.JPG'
        gray = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)
        self.assertEqual(gray.shape, (2080, 3120))
        gray_gpu = torch.tensor(gray, device='cuda')
        rgb_gpu = gray_gpu[..., None].repeat(1, 1, 3)
        recipe = runner.build_native_seed_recipe(selected_recipe(**{'morphology.fill': {'size': 1}}))
        saved_run = runner.run_native_recipe(recipe, gray_gpu, rgb_gpu,
            source_sha256=sha256(source.read_bytes()).hexdigest(),
            coordinate_frame=CoordinateFrame('source.decoded', gray.shape, 'top-left', 'x-right,y-down', 'pixel'),
            preparation={'gray': 'committed decoded grayscale', 'rgb': 'explicit gray repeat'})
        directory = self.root / 'full'
        archives.export_native_recipe_result(directory, recipe, saved_run)
        for operation in replay.WATERSHED_TYPED_OPERATIONS:
            stage = self.stage(operation, recipe)
            self.assertEqual(saved_run.node(stage).status, 'executed', saved_run.node(stage).reason)
            rounds = []
            def profile(frame, event, arg):
                if (event == 'call' and frame.f_code is runner._peaks._check_cancel.__code__
                        and frame.f_back.f_code is runner._peaks._greedy_spacing.__code__):
                    rounds.append(True)
            torch.cuda.synchronize()
            before = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            sys.setprofile(profile)
            try:
                result = replay.replay_native_stage(directory, stage)
                torch.cuda.synchronize()
            finally:
                sys.setprofile(None)
            peak = torch.cuda.max_memory_allocated() - before
            with self.subTest(operation=operation):
                self.assertEqual(result.status, 'matched', result.reason)
                self.assertLessEqual(peak, result.metadata['plan']['cuda_bytes'])
                self.assertEqual(result.metadata['plan']['pixels'], gray.size)
                self.assertEqual(result.metadata['comparison']['mismatched_elements'], 0)
                for port, value in result.outputs.items():
                    expected = saved_run.node(stage).outputs[port]
                    if torch.is_tensor(value):
                        self.assertEqual(value.device.type, 'cuda')
                        self.assertTrue(torch.equal(value, expected))
                    else:
                        self.assertEqual(value, expected)
                print(f'\nFull committed {operation}: {time.perf_counter() - start:.3f}s; '
                      f'additional CUDA {peak:,} <= {result.metadata["plan"]["cuda_bytes"]:,}; '
                      f'NMS rounds {len(rounds)}.', flush=True)
            del result
