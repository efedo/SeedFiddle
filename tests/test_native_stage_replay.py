"""Actual saved-input CUDA stage replay, lineage and unavailable outcomes."""
from dataclasses import replace
from hashlib import sha256
import inspect
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import tracemalloc
import unittest
from unittest.mock import patch

import torch
import cv2

from seedvision.benchmarking import native_recipe as runner
from seedvision.benchmarking import native_stage_replay as replay
from seedvision.export import native_results as archives
from tests.test_native_recipe import fixture, selected_recipe, FRAME, SOURCE, PREPARATION

os.environ.setdefault('QT_QPA_PLATFORM','offscreen')


class NativeStageReplayPlanTests(unittest.TestCase):
    def test_cuda_envelope_boundary_full_fixture_and_uhd_scope(self):
        maximum = (replay.MAX_REPLAY_CUDA_BYTES-128*1024*1024)//128
        self.assertEqual(maximum,7_340_032)
        plan = replay._plan((1,maximum),'threshold.binary',{},2*maximum)
        self.assertEqual(plan['cuda_bytes'],replay.MAX_REPLAY_CUDA_BYTES)
        for shape in ((1,maximum+1),(2160,3840)):
            with self.subTest(shape=shape), self.assertRaises(runner.NativeRecipeError):
                replay._plan(shape,'threshold.binary',{},2*shape[0]*shape[1])
        self.assertLess(replay._plan((2080,3120),'threshold.binary',{},2*2080*3120)['cuda_bytes'],
                        replay.MAX_REPLAY_CUDA_BYTES)

    def test_fill_compression_work_boundary_and_area_shortcuts(self):
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

    def test_distance_two_sweep_line_budget_and_median_maximum_kernel(self):
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


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class NativeStageReplayTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        gray,rgb = fixture()
        self.recipe = runner.build_native_seed_recipe(selected_recipe())
        self.run = runner.run_native_recipe(self.recipe, torch.tensor(gray,device='cuda'),
            torch.tensor(rgb,device='cuda'),source_sha256=SOURCE,coordinate_frame=FRAME,preparation=PREPARATION)
        self.directory = self.root/'run'
        archives.export_native_recipe_result(self.directory,self.recipe,self.run)

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

    def test_actual_raster_stage_replays_dispatch_only_target_preserve_archives_and_cuda(self):
        before = self.archive_hashes()
        for operation in ('filter.median_blur','threshold.binary','morphology.fill',
                          'morphology.erode','morphology.dilate','reference.internal.watershed.distance_map'):
            with self.subTest(operation=operation):
                calls = []
                def profile(frame,event,arg):
                    if event=='call' and frame.f_code is runner._dispatch.__code__:
                        calls.append(frame.f_locals['operation'])
                sys.setprofile(profile)
                try: result = replay.replay_native_stage(self.directory,self.stage(operation))
                finally: sys.setprofile(None)
                self.assertEqual(result.status,'matched',result.reason)
                self.assertEqual(calls,[operation])
                self.assertFalse(result.scientifically_validated)
                self.assertEqual(result.metadata['comparison']['mismatched_elements'],0)
                self.assertTrue(result.metadata['comparison']['archive_payloads_verified'])
                self.assertTrue(all(value.device.type=='cuda' for value in result.outputs.values()))
                for port,value in result.outputs.items():
                    self.assertTrue(torch.equal(value,self.run.node(self.stage(operation)).outputs[port]))
        self.assertEqual(self.archive_hashes(),before)

    def test_mismatched_saved_output_is_reported_without_replacing_history(self):
        stage = self.stage('threshold.binary'); node = self.run.node(stage)
        changed = node.outputs['mask'].clone(); changed[0,0] = 255-changed[0,0]
        altered = replace(node,outputs={'mask':changed})
        directory = self.save_changed(nodes=tuple(altered if item.stage_id==stage else item for item in self.run.nodes))
        result = replay.replay_native_stage(directory,stage)
        self.assertEqual(result.status,'mismatch',result.reason)
        self.assertEqual(result.metadata['comparison']['mismatched_elements'],1)
        self.assertTrue(torch.equal(result.outputs['mask'],node.outputs['mask']))

    def test_missing_runtime_binding_and_changed_runtime_withhold_before_array_loading(self):
        for missing in (False,True):
            runtime = dict(self.run.runtime)
            if missing: del runtime['calculation_modules_sha256']
            else: runtime['calculation_modules_sha256'] = '0'*64
            directory = self.save_changed(runtime=runtime)
            with patch.object(archives,'load_native_recipe_result') as load:
                result = replay.replay_native_stage(directory,self.stage('threshold.binary'))
                self.assertEqual(result.status,'unavailable',result.reason)
                self.assertFalse(result.outputs); load.assert_not_called()

    def test_wrong_edge_input_key_is_invalid_before_array_loading(self):
        stage = self.stage('threshold.binary'); node = self.run.node(stage)
        changed = replace(node,input_keys={'gray':'0'*64})
        directory = self.save_changed(nodes=tuple(changed if item.stage_id==stage else item for item in self.run.nodes))
        with patch.object(archives,'load_native_recipe_result') as load:
            result = replay.replay_native_stage(directory,stage)
            self.assertEqual(result.status,'invalid_archive',result.reason)
            self.assertFalse(result.outputs); load.assert_not_called()

    def test_expected_archive_revision_withholds_before_loading(self):
        with patch.object(archives,'load_native_recipe_result') as load:
            result = replay.replay_native_stage(self.directory,self.stage('threshold.binary'),expected_revision='0'*64)
            self.assertEqual(result.status,'invalid_archive',result.reason)
            self.assertFalse(result.outputs); load.assert_not_called()

    def test_valid_cropped_frame_archive_replays_in_its_declared_local_frame(self):
        from tests.test_native_calibrated_seed_recipe import fixture as card_fixture, selected_recipe as card_recipe
        recipe = card_recipe(); path = self.root/'card.png'
        self.assertTrue(cv2.imwrite(str(path),card_fixture()))
        run = runner.run_native_recipe_path(recipe,path)
        stage = next(item.stage_id for item in recipe.stages if item.operation_id=='threshold.binary')
        self.assertEqual(run.node(stage).status,'executed',run.node(stage).reason)
        directory = self.root/'cropped'; archives.export_native_recipe_result(directory,recipe,run)
        report, arrays = archives.load_native_recipe_result(directory)
        self.assertTrue(arrays); self.assertEqual(report['result_revision'],
            archives.read_native_recipe_inventory(directory)[0]['result_revision'])
        result = replay.replay_native_stage(directory,stage)
        self.assertEqual(result.status,'matched',result.reason)
        self.assertEqual(result.metadata['frame_scope'],'calibrated_seed_roi')
        self.assertEqual(tuple(result.metadata['coordinate_frame']['shape']),(96,128))
        self.assertEqual(result.metadata['frame_lineage']['local_to_parent_xy'],(600,260))
        self.assertTrue(torch.equal(result.outputs['mask'],run.node(stage).outputs['mask']))

    def test_unavailable_stage_unsupported_operation_and_invalid_archive_stay_explicit(self):
        stage = self.stage('threshold.binary'); node = self.run.node(stage)
        changed = replace(node,status='failed',outputs={},product_key=None,reason='Fixture unavailable')
        directory = self.save_changed(status='partial',nodes=tuple(changed if item.stage_id==stage else item for item in self.run.nodes))
        self.assertEqual(replay.replay_native_stage(directory,stage).status,'unavailable')
        self.assertEqual(replay.replay_native_stage(self.directory,self.stage('external.image_inputs')).status,'unsupported')
        path = self.directory/'native-results.json'; path.write_bytes(path.read_bytes()+b'!')
        result = replay.replay_native_stage(self.directory,stage)
        self.assertEqual(result.status,'invalid_archive',result.reason); self.assertFalse(result.outputs)

    def test_admission_cancellation_and_callback_tensor_mutation_withhold_outputs(self):
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

    def test_loaded_checkpoint_helper_change_withholds_replayed_output(self):
        import seedvision.benchmarking.checkpoints as checkpoints
        original = checkpoints._selected_array_names
        changed = False
        def mutate():
            nonlocal changed
            frame = inspect.currentframe()
            while frame is not None:
                if frame.f_code is replay.replay_native_stage.__code__ and 'source' in frame.f_locals and not changed:
                    checkpoints._selected_array_names = lambda names,available: original(names,available)
                    changed = True
                frame = frame.f_back
            return False
        try: result = replay.replay_native_stage(self.directory,self.stage('threshold.binary'),cancel=mutate)
        finally: checkpoints._selected_array_names = original
        self.assertTrue(changed,result.reason)
        self.assertEqual(result.status,'failed',result.reason); self.assertFalse(result.outputs)

    def test_committed_full_resolution_saved_input_replay_cuda_and_host_peaks(self):
        from seedvision.benchmarking.checkpoints import CoordinateFrame
        from tests.test_native_core_recovery_recipe import selected_recipe as recovery_recipe
        source = Path(__file__).parents[1]/'images'/'IMG_9632c.JPG'
        gray = cv2.imread(str(source),cv2.IMREAD_GRAYSCALE)
        self.assertEqual(gray.shape,(2080,3120))
        gray_gpu = torch.tensor(gray,device='cuda'); rgb_gpu = gray_gpu[...,None].repeat(1,1,3)
        recipe = recovery_recipe()
        run = runner.run_native_recipe(recipe,gray_gpu,rgb_gpu,source_sha256=sha256(source.read_bytes()).hexdigest(),
            coordinate_frame=CoordinateFrame('source.decoded',gray.shape,'top-left','x-right,y-down','pixel'),
            preparation={'gray':'committed decoded grayscale','rgb':'explicit gray repeat'})
        self.assertEqual(run.status,'executed')
        stage = next(item.stage_id for item in recipe.stages if item.operation_id=='threshold.binary')
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

    def test_zero_output_is_comparable_and_mean_otsu_use_their_actual_saved_inputs(self):
        for kind in ('mean','otsu'):
            recipe = runner.build_native_seed_recipe(selected_recipe(kind))
            gray,rgb = fixture()
            run = runner.run_native_recipe(recipe,torch.tensor(gray,device='cuda'),torch.tensor(rgb,device='cuda'),
                source_sha256=SOURCE,coordinate_frame=FRAME,preparation=PREPARATION)
            directory = self.root/kind; archives.export_native_recipe_result(directory,recipe,run)
            stage = next(item.stage_id for item in recipe.stages if item.operation_id=='threshold.'+kind)
            result = replay.replay_native_stage(directory,stage)
            self.assertEqual(result.status,'matched',result.reason)
        recipe = selected_recipe(**{'threshold.binary':{'threshold':255}})
        recipe = runner.build_native_seed_recipe(recipe)
        gray,rgb = fixture()
        run = runner.run_native_recipe(recipe,torch.tensor(gray,device='cuda'),torch.tensor(rgb,device='cuda'),
            source_sha256=SOURCE,coordinate_frame=FRAME,preparation=PREPARATION)
        directory = self.root/'empty'; archives.export_native_recipe_result(directory,recipe,run)
        stage = next(item.stage_id for item in recipe.stages if item.operation_id=='threshold.binary')
        result = replay.replay_native_stage(directory,stage)
        self.assertEqual(result.status,'matched',result.reason)
        self.assertFalse(bool(result.outputs['mask'].any().item()))

    def test_actual_async_qt_replay_action_preview_and_active_run_preservation(self):
        from PySide6.QtWidgets import QApplication, QFileDialog
        from seedvision.ui.native_recipe_dialog import NativeRecipeDialog
        application = QApplication.instance() or QApplication([])
        dialog = NativeRecipeDialog(self.recipe); dialog.show()
        dialog.result = self.run
        before = dialog.recipe.to_json(); hashes = self.archive_hashes()
        dialog._select_stage(self.stage('threshold.binary'))
        try:
            self.assertTrue(dialog.replay_button.isEnabled())
            with patch.object(QFileDialog,'getExistingDirectory',return_value=str(self.directory)):
                dialog.replay_button.click()
            self.assertIsNotNone(dialog._worker)
            self.assertFalse(dialog.canvas.isEnabled()); self.assertFalse(dialog.inspector.isEnabled())
            deadline = time.monotonic()+30
            while dialog._worker is not None and time.monotonic()<deadline:
                application.processEvents(); time.sleep(.01)
            self.assertIsNone(dialog._worker)
            self.assertEqual(dialog.replay_result.status,'matched',dialog.status_label.text())
            self.assertIs(dialog.result,self.run)
            self.assertEqual(dialog.recipe.to_json(),before)
            self.assertEqual(self.archive_hashes(),hashes)
            self.assertFalse(dialog.preview.pixmap().isNull())
            self.assertTrue(dialog.canvas.isEnabled()); self.assertTrue(dialog.inspector.isEnabled())
            self.assertIn('Saved-input replay',dialog.facts.toPlainText())
            dialog._select_stage(self.stage('analysis.size'))
            self.assertTrue(dialog.replay_button.isEnabled())
            dialog._select_stage(self.stage('external.image_inputs'))
            self.assertFalse(dialog.replay_button.isEnabled())
        finally:
            if dialog._worker is not None:
                dialog.stop()
                while dialog._worker is not None: application.processEvents(); time.sleep(.01)
            dialog.reject(); dialog.deleteLater(); application.processEvents()

    def test_qt_close_cancels_and_drains_saved_stage_worker(self):
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

    def test_qt_rejects_changed_selected_parameters_and_restores_controls_without_stale_facts(self):
        from PySide6.QtWidgets import QApplication
        import seedvision.ui.native_recipe_dialog as ui
        application = QApplication.instance() or QApplication([])
        selected = runner.build_native_seed_recipe(selected_recipe(**{'threshold.binary':{'threshold':37}}))
        self.assertNotEqual(selected.identity.settings_sha256,self.recipe.identity.settings_sha256)
        dialog = ui.NativeRecipeDialog(selected); dialog.show(); dialog.result = self.run
        dialog._select_stage(self.stage('threshold.binary'))
        button_states = [(button,button.isEnabled()) for button in dialog._export_conflicting_buttons()]
        try:
            with patch.object(ui,'replay_native_stage') as execute:
                dialog.start_stage_replay(self.directory)
                self.assertIn('replay is in progress',dialog.facts.toPlainText())
                deadline = time.monotonic()+20
                while dialog._worker is not None and time.monotonic()<deadline:
                    application.processEvents(); time.sleep(.01)
                self.assertIsNone(dialog._worker); execute.assert_not_called()
            self.assertIsNone(dialog.replay_result); self.assertIs(dialog.result,self.run)
            self.assertIn('different recipe settings',dialog.status_label.text())
            self.assertIn('active run is retained',dialog.facts.toPlainText())
            self.assertTrue(dialog.canvas.isEnabled()); self.assertTrue(dialog.inspector.isEnabled())
            self.assertTrue(dialog.replay_button.isEnabled()); self.assertFalse(dialog.stop_button.isEnabled())
            self.assertTrue(all(button.isEnabled()==enabled for button,enabled in button_states))
        finally:
            if dialog._worker is not None:
                dialog.stop()
                while dialog._worker is not None: application.processEvents(); time.sleep(.01)
            dialog.reject(); dialog.deleteLater(); application.processEvents()
