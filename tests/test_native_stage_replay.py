"""Actual saved-input CUDA stage replay, lineage and unavailable outcomes."""

from tests.functional_units import functional_units
from dataclasses import replace
from hashlib import sha256
import inspect
import os
from pathlib import Path
import sys
import tempfile
import shutil
import threading
import time
import tracemalloc
import unittest
from tests.catalogue_fixtures import CatalogueTestCase
from unittest.mock import patch

import torch
import cv2

from seedvision.benchmarking import native_recipe as runner
from seedvision.benchmarking import native_stage_replay as replay
from seedvision.export import native_results as archives
from tests.test_native_recipe import fixture, selected_recipe, FRAME, SOURCE, PREPARATION

os.environ.setdefault('QT_QPA_PLATFORM','offscreen')


@functional_units
class NativeStageReplayPlanTests(CatalogueTestCase):
    def check_test_cuda_envelope_boundary_full_fixture_and_uhd_scope(self):
        maximum = (replay.MAX_REPLAY_CUDA_BYTES-128*1024*1024)//128
        self.assertEqual(maximum,7_340_032)
        plan = replay._plan((1,maximum),'threshold.binary',{},2*maximum)
        self.assertEqual(plan['cuda_bytes'],replay.MAX_REPLAY_CUDA_BYTES)
        for shape in ((1,maximum+1),(2160,3840)):
            with self.subTest(shape=shape), self.assertRaises(runner.NativeRecipeError):
                replay._plan(shape,'threshold.binary',{},2*shape[0]*shape[1])
        self.assertLess(replay._plan((2080,3120),'threshold.binary',{},2*2080*3120)['cuda_bytes'],
                        replay.MAX_REPLAY_CUDA_BYTES)

    def check_test_fill_compression_work_boundary_and_area_shortcuts(self):
        # For a one-row raster below the1M edge chunk there is one batch:
        # 128P +104P +64*(32*(P-1)+104P).
        maximum = (replay.MAX_REPLAY_WORK+2048)//8936
        plan = replay._plan((1,maximum),'morphology.fill',{'size':2},2*maximum)
        self.assertEqual(plan['work'],8936*maximum-2048)
        self.assertEqual(plan['work_details']['full_parent_compressions'],65)
        with self.assertRaises(runner.NativeRecipeError):
            replay._plan((1,maximum+1),'morphology.fill',{'size':2},2*(maximum+1))
        with self.assertRaises(runner.NativeRecipeError):
            replay._plan((2080,3120),'morphology.fill',{'size':2},2*2080*3120)
        for size in (1,2080*3120+1):
            self.assertTrue(replay._plan((2080,3120),'morphology.fill',{'size':size},2*2080*3120)
                            ['work_details']['area_shortcut'])

    def check_test_distance_two_sweep_line_budget_and_median_maximum_kernel(self):
        limit = replay.MAX_DISTANCE_SCAN_CHUNKS//2
        plan = replay._plan((limit,1),'reference.internal.watershed.distance_map',{},5*limit)
        self.assertEqual(plan['work_details']['scan_chunks'],2*limit)
        self.assertEqual(plan['work'],128*limit+512*2*limit+4*runner._distance.SCAN_CHUNK_PIXELS)
        with self.assertRaises(runner.NativeRecipeError):
            replay._plan((limit+1,1),'reference.internal.watershed.distance_map',{},5*(limit+1))
        self.assertEqual(replay._plan((1,1_000_000),'reference.internal.watershed.distance_map',{},5_000_000)
                         ['work_details']['scan_chunks'],8)
        self.assertEqual(replay._plan((64,64),'filter.median_blur',{'ksize':255},2*64*64)['work'],64*64*255*255)
        with self.assertRaises(runner.NativeRecipeError):
            replay._plan((512,512),'filter.median_blur',{'ksize':255},2*512*512)

    def test_validation_and_recovery(self):
        """3 related NativeStageReplayPlanTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_cuda_envelope_boundary_full_fixture_and_uhd_scope)
        self.scenario(self.check_test_fill_compression_work_boundary_and_area_shortcuts)
        self.scenario(self.check_test_distance_two_sweep_line_budget_and_median_maximum_kernel)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
@functional_units
class NativeStageReplayTests(CatalogueTestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture_temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.fixture_temp.cleanup)
        cls.fixture_root = Path(cls.fixture_temp.name)
        gray,rgb = fixture()
        cls.recipe = runner.build_native_seed_recipe(selected_recipe())
        cls.saved_run = runner.run_native_recipe(cls.recipe, torch.tensor(gray,device='cuda'),
            torch.tensor(rgb,device='cuda'),source_sha256=SOURCE,coordinate_frame=FRAME,preparation=PREPARATION)
        cls.fixture_directory = cls.fixture_root/'run'
        archives.export_native_recipe_result(cls.fixture_directory,cls.recipe,cls.saved_run)
        from tests.replay_fixtures import archive_hashes
        cls.fixture_hashes = archive_hashes(cls.fixture_directory)
        cls.fixture_signature = runner._nested_signature((), tuple(n.outputs for n in cls.saved_run.nodes))

    @classmethod
    def tearDownClass(cls):
        from tests.replay_fixtures import archive_hashes
        if (archive_hashes(cls.fixture_directory) != cls.fixture_hashes or
                runner._nested_signature((), tuple(n.outputs for n in cls.saved_run.nodes)) != cls.fixture_signature):
            raise AssertionError('Replay test modified its shared source fixture')

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        self.run = self.saved_run
        self.directory = self.root/'run'
        shutil.copytree(self.fixture_directory,self.directory)

    def tearDown(self): self.temp.cleanup()

    def stage(self, operation):
        return next(stage.stage_id for stage in self.recipe.stages if stage.operation_id==operation)

    def save_changed(self, **changes):
        run = replace(self.run, **changes)
        directory = self.root/f'changed-{len(tuple(self.root.iterdir()))}'
        archives.export_native_recipe_result(directory,self.recipe,run)
        return directory

    def archive_hashes(self):
        return {str(path.relative_to(self.directory)):sha256(path.read_bytes()).hexdigest()
                for path in self.directory.rglob('*') if path.is_file()}

    def check_test_actual_raster_stage_replays_dispatch_only_target_preserve_archives_and_cuda(self):
        before = self.archive_hashes()
        for operation in ('filter.median_blur','threshold.binary','morphology.fill',
                          'morphology.erode','morphology.dilate','reference.internal.watershed.distance_map'):
            with self.subTest(operation=operation):
                calls = []
                def profile(frame,event,arg):
                    if event=='call' and frame.f_code is runner._dispatch.__code__:
                        calls.append(frame.f_locals['operation'])
                from tests.observation import observe_functions
                with observe_functions(profile, runner._dispatch):
                    result = replay.replay_native_stage(self.directory,self.stage(operation))
                self.assertEqual(result.status,'matched',result.reason)
                self.assertEqual(calls,[operation])
                self.assertFalse(result.scientifically_validated)
                self.assertEqual(result.metadata['comparison']['mismatched_elements'],0)
                self.assertTrue(result.metadata['comparison']['archive_payloads_verified'])
                self.assertTrue(all(value.device.type=='cuda' for value in result.outputs.values()))
                for port,value in result.outputs.items():
                    self.assertTrue(torch.equal(value,self.run.node(self.stage(operation)).outputs[port]))
        self.assertEqual(self.archive_hashes(),before)

    def check_test_mismatched_saved_output_is_reported_without_replacing_history(self):
        stage = self.stage('threshold.binary'); node = self.run.node(stage)
        changed = node.outputs['mask'].clone(); changed[0,0] = 255-changed[0,0]
        altered = replace(node,outputs={'mask':changed})
        directory = self.save_changed(nodes=tuple(altered if item.stage_id==stage else item for item in self.run.nodes))
        result = replay.replay_native_stage(directory,stage)
        self.assertEqual(result.status,'mismatch',result.reason)
        self.assertEqual(result.metadata['comparison']['mismatched_elements'],1)
        self.assertTrue(torch.equal(result.outputs['mask'],node.outputs['mask']))






    def check_test_admission_cancellation_and_callback_tensor_mutation_withhold_outputs(self):
        stage = self.stage('threshold.binary')
        with patch.object(replay,'MAX_SELECTED_BYTES',1), patch.object(archives,'load_native_recipe_result') as load:
            result = replay.replay_native_stage(self.directory,stage)
            self.assertEqual(result.status,'failed',result.reason); load.assert_not_called()
        self.assertEqual(replay.replay_native_stage(self.directory,stage,cancel=lambda:True).status,'cancelled')
        mutated = False
        def mutate():
            nonlocal mutated
            frame = inspect.currentframe()
            while frame is not None:
                if frame.f_code is replay.replay_native_stage.__code__ and 'source' in frame.f_locals and not mutated:
                    frame.f_locals['source'].add_(1); mutated = True
                frame = frame.f_back
            return False
        result = replay.replay_native_stage(self.directory,stage,cancel=mutate)
        self.assertTrue(mutated,result.reason)
        self.assertEqual(result.status,'failed',result.reason); self.assertFalse(result.outputs)


    def check_test_committed_full_resolution_saved_input_replay_cuda_and_host_peaks(self):
        from seedvision.benchmarking.checkpoints import CoordinateFrame
        from tests.test_native_core_recovery_recipe import selected_recipe as recovery_recipe
        source = Path(__file__).parents[1]/'images'/'IMG_9632c.JPG'
        gray = cv2.imread(str(source),cv2.IMREAD_GRAYSCALE)
        self.assertEqual(gray.shape,(2080,3120))
        gray_gpu = torch.tensor(gray,device='cuda'); rgb_gpu = gray_gpu[...,None].repeat(1,1,3)
        recipe = recovery_recipe()
        stage = next(item.stage_id for item in recipe.stages if item.operation_id=='threshold.binary')
        run = runner.run_native_recipe(recipe,gray_gpu,rgb_gpu,source_sha256=sha256(source.read_bytes()).hexdigest(),
            coordinate_frame=CoordinateFrame('source.decoded',gray.shape,'top-left','x-right,y-down','pixel'),
            preparation={'gray':'committed decoded grayscale','rgb':'explicit gray repeat'},target_node_id=stage)
        self.assertEqual(run.node(stage).status,'executed')
        self.assertTrue(all(node.status=='not_run' for node in run.nodes
            if node.operation_id not in ('external.image_inputs','threshold.binary')))
        directory = self.root/'full'; archives.export_native_recipe_result(directory,recipe,run)
        torch.cuda.synchronize(); before = torch.cuda.memory_allocated(); torch.cuda.reset_peak_memory_stats()
        tracemalloc.start(); start = time.perf_counter()
        try:
            result = replay.replay_native_stage(directory,stage)
            torch.cuda.synchronize(); _,host_peak = tracemalloc.get_traced_memory()
        finally: tracemalloc.stop()
        elapsed = time.perf_counter()-start; cuda_peak = torch.cuda.max_memory_allocated()-before
        self.assertEqual(result.status,'matched',result.reason)
        self.assertTrue(torch.equal(result.outputs['mask'],run.node(stage).outputs['mask']))
        self.assertLessEqual(cuda_peak,result.metadata['plan']['cuda_bytes'])
        # This observed peak includes report decoding and every verified group;
        # the compressed snapshot limit is not a total process memory claim.
        self.assertLessEqual(host_peak,384*1024*1024)
        print(f'\nFull committed saved-stage replay: {elapsed:.3f}s; CUDA peak {cuda_peak:,} '
              f'<= {result.metadata["plan"]["cuda_bytes"]:,}; traced host peak {host_peak:,}.',flush=True)



    def check_test_qt_close_cancels_and_drains_saved_stage_worker(self):
        from PySide6.QtWidgets import QApplication
        import seedvision.ui.native_recipe_dialog as ui
        application = QApplication.instance() or QApplication([])
        dialog = ui.NativeRecipeDialog(self.recipe); dialog.show()
        dialog._select_stage(self.stage('threshold.binary'))
        started = threading.Event()
        def blocked(directory,stage_id,*,cancel,**kwargs):
            started.set()
            while not cancel(): time.sleep(.01)
            return replay._result('cancelled',stage_id,'Stopped')
        with patch.object(ui,'replay_native_stage',side_effect=blocked):
            try:
                dialog.start_stage_replay(self.directory)
                deadline = time.monotonic()+10
                while not started.is_set() and time.monotonic()<deadline:
                    application.processEvents(); time.sleep(.01)
                self.assertTrue(started.is_set())
                dialog.reject()
                self.assertIsNotNone(dialog._worker)
                while dialog._worker is not None and time.monotonic()<deadline:
                    application.processEvents(); time.sleep(.01)
                self.assertIsNone(dialog._worker)
                self.assertFalse(dialog.isVisible()); self.assertIsNone(dialog.replay_result)
            finally:
                if dialog._worker is not None:
                    dialog.stop()
                    while dialog._worker is not None: application.processEvents(); time.sleep(.01)
                dialog.reject(); dialog.deleteLater(); application.processEvents()


    def test_workflow_lifecycle(self):
        """3 related NativeStageReplayTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_actual_raster_stage_replays_dispatch_only_target_preserve_archives_and_cuda)
        self.scenario(self.check_test_mismatched_saved_output_is_reported_without_replacing_history)
        self.scenario(self.check_test_committed_full_resolution_saved_input_replay_cuda_and_host_peaks)

    def test_validation_and_recovery(self):
        """2 related NativeStageReplayTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_admission_cancellation_and_callback_tensor_mutation_withhold_outputs)
        self.scenario(self.check_test_qt_close_cancels_and_drains_saved_stage_worker)
