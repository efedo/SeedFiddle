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
class NativeLocalStageReplayTests(unittest.TestCase):
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

    def test_all_eight_operations_use_only_saved_local_rasters_and_local_admission(self):
        hashes = self.hashes()
        for kind in ('binary','mean','otsu'):
            recipe,run,directory = self.recipe,self.saved_run,self.directory
            operations = ['filter.median_blur','threshold.binary','morphology.fill',
                'morphology.erode','morphology.dilate','reference.internal.watershed.distance_map']
            if kind!='binary':
                recipe = selected_recipe(kind); run = runner.run_native_recipe_path(recipe,self.path)
                directory = self.root/kind; archives.export_native_recipe_result(directory,recipe,run)
                operations = ['threshold.'+kind]
            for operation in operations:
                with self.subTest(kind=kind,operation=operation):
                    calls,uploads = [],[]
                    def profile(frame,event,arg):
                        if event=='call' and frame.f_code is runner._dispatch.__code__:
                            calls.append(frame.f_locals['operation'])
                        if event=='c_call' and arg is torch.tensor:
                            caller = frame.f_locals
                            if frame.f_code is replay.replay_native_stage.__code__:
                                # Both CPU arrays remain selected/local; no parent array is retained.
                                uploads.extend(tuple(value.shape) for value in caller['arrays'].values())
                    torch.cuda.synchronize(); before = torch.cuda.memory_allocated()
                    torch.cuda.reset_peak_memory_stats(); sys.setprofile(profile)
                    try:
                        result = replay.replay_native_stage(directory,self.stage(operation,recipe))
                        torch.cuda.synchronize()
                    finally: sys.setprofile(None)
                    self.assertEqual(result.status,'matched',result.reason)
                    self.assertEqual(calls,[operation]); self.assertTrue(uploads)
                    self.assertTrue(all(shape==(96,128) for shape in uploads),uploads)
                    self.assertEqual(result.metadata['plan']['pixels'],96*128)
                    byte_factor = 5 if operation.endswith('distance_map') else 2
                    self.assertEqual(result.metadata['plan']['selected_bytes'],byte_factor*96*128)
                    self.assertLessEqual(torch.cuda.max_memory_allocated()-before,result.metadata['plan']['cuda_bytes'])
                    self.assertEqual(result.metadata['frame_scope'],'calibrated_seed_roi')
                    self.assertEqual(tuple(result.metadata['parent_coordinate_frame']['shape']),(620,900))
                    frame = CoordinateFrame.from_dict(dict(result.metadata['coordinate_frame']))
                    expected = CoordinateFrame.from_dict(dict(run.node(factory.CROP).outputs['metadata']['coordinate_frame']))
                    self.assertEqual(frame,expected)
                    self.assertEqual(result.metadata['frame_lineage']['local_to_parent_xy'],(600,260))
                    self.assertEqual(result.metadata['frame_lineage']['card_bounding_overlap_pixels'],0)
                    self.assertFalse(result.scientifically_validated)
                    self.assertTrue(result.metadata['comparison']['archive_payloads_verified'])
                    for port,value in result.outputs.items():
                        self.assertEqual(value.device.type,'cuda')
                        self.assertTrue(torch.equal(value,run.node(self.stage(operation,recipe)).outputs[port]))
                    del result
        self.assertEqual(self.hashes(),hashes)

    def test_full_bounds_crop_keeps_distinct_local_frame_and_verified_positive_overlap(self):
        recipe = _parameter(self.recipe,factory.CROP,'seed_roi_xyxy',[0,0,900,620])
        run = runner.run_native_recipe_path(recipe,self.path)
        directory = self.root/'full-bounds'; archives.export_native_recipe_result(directory,recipe,run)
        hashes = self.hashes(directory)
        for operation in ('filter.median_blur','threshold.binary'):
            result = replay.replay_native_stage(directory,self.stage(operation,recipe))
            self.assertEqual(result.status,'matched',result.reason)
            local = result.metadata['coordinate_frame']; parent = result.metadata['parent_coordinate_frame']
            self.assertEqual(local['shape'],parent['shape'])
            self.assertEqual(local['frame_id'],parent['frame_id']+'.seed_roi')
            self.assertNotEqual(local['transform_sha256'],parent['transform_sha256'])
            self.assertEqual(result.metadata['frame_lineage']['local_to_parent_xy'],(0,0))
            self.assertGreater(result.metadata['frame_lineage']['card_bounding_overlap_pixels'],0)
            self.assertEqual(result.metadata['plan']['pixels'],620*900)
        self.assertEqual(self.hashes(directory),hashes)

    def test_rehashed_crop_metadata_contradictions_are_rejected_before_pixels_load(self):
        stage = self.stage('threshold.binary')
        original = dict(self.saved_run.node(factory.CROP).outputs['metadata'])
        variants = []
        for field,value in (('bounds_xyxy',[599,260,727,356]),('local_to_parent_xy',[601,260]),
                ('source_sha256','0'*64),('card_bounding_overlap_pixels',1),
                ('gray_conversion','different conversion'),('labels_and_coordinates','parent pixels')):
            variants.append((field,{**original,field:value}))
        for field,value in (('frame_id','other.seed_roi'),('transform_sha256','0'*64),
                ('origin','center'),('shape',[128,96])):
            variants.append(('local '+field,{**original,'coordinate_frame':{**dict(original['coordinate_frame']),field:value}}))
        variants.append(('parent frame',{**original,'parent_frame':{**dict(original['parent_frame']),'frame_id':'other'}}))
        variants.append(('unbounded metadata',{**original,'extra':list(range(replay.MAX_FRAME_METADATA_ITEMS+1))}))
        variants.append(('array metadata',{**original,'extra':torch.zeros(1,dtype=torch.uint8,device='cuda')}))
        for name,metadata in variants:
            with self.subTest(name=name):
                outputs = {**dict(self.saved_run.node(factory.CROP).outputs),'metadata':metadata}
                self.assert_invalid_before_loading(self.changed(factory.CROP,outputs=outputs),stage)

    def test_source_record_card_bounds_and_geometry_contradictions_are_rejected(self):
        stage = self.stage('threshold.binary')
        source_edge = next(edge for edge in self.recipe.dependencies
            if edge.target_stage==factory.CROP and edge.target_port=='source_frame')
        source = self.saved_run.node(source_edge.source_stage)
        for field,value in (('source_sha256','f'*64),('coordinate_frame',{**dict(self.saved_run.coordinate_frame.to_dict()),'frame_id':'other'})):
            with self.subTest(field=field):
                record = {**dict(source.outputs['source_frame']),field:value}
                self.assert_invalid_before_loading(self.changed(source.stage_id,
                    outputs={**dict(source.outputs),'source_frame':record}),stage)
        self.assert_invalid_before_loading(self.changed(card.PREPROCESS,
            outputs={**dict(self.saved_run.node(card.PREPROCESS).outputs),'bounds':[0,0,459,620]}),stage)
        for rect in ([0,0,461,620],[-1,0,2,2],[0,0,0,2],[0.,0,2,2]):
            with self.subTest(rect=rect):
                geometry = {**dict(self.saved_run.node(card.GEOMETRY).outputs['geometry']),'bounding_rect':rect}
                self.assert_invalid_before_loading(self.changed(card.GEOMETRY,
                    outputs={**dict(self.saved_run.node(card.GEOMETRY).outputs),'geometry':geometry}),stage)

    def test_every_ancestor_key_and_local_parent_raster_contract_is_checked(self):
        stage = self.stage('morphology.dilate')
        for owner in (factory.CROP,self.stage('filter.median_blur'),self.stage('threshold.binary')):
            node = self.saved_run.node(owner); keys = dict(node.input_keys)
            for port in keys:
                with self.subTest(owner=owner,port=port):
                    self.assert_invalid_before_loading(self.changed(owner,input_keys={**keys,port:'0'*64}),stage)
        median = self.stage('filter.median_blur')
        bad_rasters = ((median,'image',torch.zeros((620,900),dtype=torch.uint8,device='cuda')),
            (median,'image',torch.zeros((96,128),dtype=torch.float32,device='cuda')),
            (card.AFFINE,'corrected_rgb',torch.zeros((619,900,3),dtype=torch.uint8,device='cuda')),
            (factory.CROP,'rgb_image',torch.zeros((96,127,3),dtype=torch.uint8,device='cuda')))
        for owner,port,value in bad_rasters:
            with self.subTest(owner=owner,port=port,shape=tuple(value.shape),dtype=value.dtype):
                self.assert_invalid_before_loading(self.changed(owner,
                    outputs={**dict(self.saved_run.node(owner).outputs),port:value}),stage)
        self.assert_invalid_before_loading(self.changed(median,status='failed',product_key=None,outputs={},
            reason='Fixture ancestor unavailable'),stage)

    def test_local_admission_and_callback_cancellation_withhold_before_dispatch(self):
        stage = self.stage('threshold.binary'); calls = []
        def profile(frame,event,arg):
            if event=='call' and frame.f_code is runner._dispatch.__code__:
                calls.append(frame.f_locals['operation'])
        with patch.object(replay,'MAX_SELECTED_BYTES',2*96*128-1),patch.object(archives,'load_native_recipe_result') as load:
            result = replay.replay_native_stage(self.directory,stage)
        self.assertEqual(result.status,'failed',result.reason); load.assert_not_called()
        reached = False
        def cancel():
            nonlocal reached
            frame = inspect.currentframe()
            while frame is not None:
                if frame.f_code is replay.replay_native_stage.__code__ and 'source' in frame.f_locals:
                    reached = True
                    return True
                frame = frame.f_back
            return False
        sys.setprofile(profile)
        try: result = replay.replay_native_stage(self.directory,stage,cancel=cancel)
        finally: sys.setprofile(None)
        self.assertTrue(reached); self.assertEqual(result.status,'cancelled',result.reason)
        self.assertFalse(result.outputs); self.assertFalse(calls)

    def test_qt_replays_calibrated_local_frame_and_retains_current_run(self):
        from PySide6.QtWidgets import QApplication
        from seedvision.ui.native_recipe_dialog import NativeRecipeDialog
        application = QApplication.instance() or QApplication([])
        dialog = NativeRecipeDialog(self.recipe); dialog.show(); dialog.result = self.saved_run
        stage = self.stage('threshold.binary'); dialog._select_stage(stage)
        hashes = self.hashes(); recipe_bytes = dialog.recipe.to_json()
        try:
            dialog.start_stage_replay(self.directory)
            deadline = time.monotonic()+30
            while dialog._worker is not None and time.monotonic()<deadline:
                application.processEvents(); time.sleep(.01)
            self.assertIsNone(dialog._worker)
            self.assertEqual(dialog.replay_result.status,'matched',dialog.status_label.text())
            self.assertEqual(dialog.replay_result.metadata['frame_scope'],'calibrated_seed_roi')
            self.assertIs(dialog.result,self.saved_run); self.assertEqual(dialog.recipe.to_json(),recipe_bytes)
            self.assertFalse(dialog.preview.pixmap().isNull())
            self.assertIn('calibrated_seed_roi',dialog.facts.toPlainText())
            self.assertEqual(self.hashes(),hashes)
            self.assertTrue(dialog.canvas.isEnabled()); self.assertTrue(dialog.inspector.isEnabled())
        finally:
            if dialog._worker is not None:
                dialog.stop()
                while dialog._worker is not None: application.processEvents(); time.sleep(.01)
            dialog.reject(); dialog.deleteLater(); application.processEvents()
