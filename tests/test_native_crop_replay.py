"""Actual dual-frame calibrated crop replay, admission, callback and Qt checks."""
from dataclasses import replace
from hashlib import sha256
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch

os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from PySide6.QtWidgets import QApplication
from seedvision.benchmarking import native_crop_replay as contract
from seedvision.benchmarking import native_calibrated_seed_recipe as factory
from seedvision.benchmarking import native_card_recipe as card
from seedvision.benchmarking import native_recipe as runner
from seedvision.benchmarking import native_stage_replay as replay
from seedvision.benchmarking.checkpoints import CoordinateFrame,_freeze_json
from seedvision.export import native_results as archives
from seedvision.ui.native_recipe_dialog import NativeRecipeDialog
from tests.test_native_calibrated_seed_recipe import fixture,selected_recipe
from tests.test_native_card_recipe import _parameter


class CropReplayContractTests(unittest.TestCase):
    def context(self,bounds=(2,3,8,7)):
        parent=CoordinateFrame('explicit.parent',(10,12),'top-left','x-right,y-down','pixel','a'*64)
        return contract.frames(parent,{'seed_roi_xyxy':bounds},'b'*64,(0,0,12,10))

    def test_strict_roi_distinct_frames_and_shifted_equal_shapes(self):
        context=self.context();shifted=self.context((3,3,9,7))
        self.assertEqual(context.local.shape,shifted.local.shape)
        self.assertNotEqual(context.local.transform_sha256,shifted.local.transform_sha256)
        self.assertEqual(context.parent.transform_sha256,'a'*64)
        self.assertEqual(context.local.frame_id,'explicit.parent.seed_roi')
        for bounds in ((True,3,8,7),(2.,3,8,7),(2,3,2,7),(-1,3,8,7),(2,3,13,7)):
            with self.subTest(bounds=bounds),self.assertRaises(contract.NativeCropReplayError):self.context(bounds)

    def test_metadata_fixed_binding_types_and_numeric_overlap_comparison(self):
        context=self.context()
        metadata=_freeze_json({**contract.fixed_metadata(context),'card_bounding_overlap_pixels':0},name='test crop metadata')
        self.assertEqual(contract.validate(metadata,'crop_metadata',context),metadata)
        changed={**metadata,'card_bounding_overlap_pixels':1}
        self.assertEqual(contract.compare(metadata,changed)['mismatched_elements'],1)
        contract.validate(changed,'crop_metadata',context)
        for value in (True,1.,-1,25):
            with self.assertRaises(contract.NativeCropReplayError):
                contract.validate({**metadata,'card_bounding_overlap_pixels':value},'crop_metadata',context)
        for field,value in (('bounds_xyxy',(2.,3,8,7)),('local_to_parent_xy',(True,3)),
                ('gray_conversion','other'),('source_sha256','c'*64)):
            with self.subTest(field=field),self.assertRaises(contract.NativeCropReplayError):
                contract.validate({**metadata,field:value},'crop_metadata',context)

    def test_allocation_free_selected_and_cuda_exact_boundaries(self):
        context=self.context((0,0,12,10))
        plan=contract.plan(context,selected_allocations=(360,360,120))
        self.assertEqual(plan['selected_bytes'],840)
        self.assertEqual(plan['selected_upload_bytes'],1536)
        self.assertEqual(plan['fresh_output_bytes'],1024)
        self.assertGreaterEqual(plan['cuda_bytes'],contract.AUDIT_CUDA_BYTES)
        boundary=contract.plan(context,selected_allocations=(360,360,120),
            selected_limit=840,cuda_limit=plan['cuda_bytes'])
        self.assertEqual(boundary,{**plan,'selected_limit_bytes':840,'cuda_limit_bytes':plan['cuda_bytes']})
        for changes in ({'selected_limit':839},{'cuda_limit':plan['cuda_bytes']-1}):
            with self.assertRaises(contract.NativeCropReplayUnavailable):
                contract.plan(context,selected_allocations=(360,360,120),**changes)


@unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
class NativeCropStageReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.root=Path(cls.temp.name)
        cls.path=cls.root/'card-seeds.png'
        assert cv2.imwrite(str(cls.path),fixture())
        cls.recipe=selected_recipe()
        cls.saved=runner.run_native_recipe_path(cls.recipe,cls.path)
        assert cls.saved.node(factory.CROP).status=='executed'
        cls.directory=cls.root/'base'
        archives.export_native_recipe_result(cls.directory,cls.recipe,cls.saved)

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def hashes(self,directory):
        return {str(p.relative_to(directory)):sha256(p.read_bytes()).hexdigest()
            for p in directory.rglob('*') if p.is_file()}

    def export(self,recipe,run,prefix='changed'):
        directory=self.root/(prefix+'-'+str(len(tuple(self.root.iterdir()))))
        archives.export_native_recipe_result(directory,recipe,run)
        return directory

    def changed(self,owner,**changes):
        node=replace(self.saved.node(owner),**changes)
        run=replace(self.saved,nodes=tuple(node if n.stage_id==owner else n for n in self.saved.nodes))
        return self.export(self.recipe,run)

    def invalid(self,directory):
        before=self.hashes(directory)
        with patch.object(archives,'load_native_recipe_result') as load:
            result=replay.replay_native_stage(directory,factory.CROP)
        self.assertEqual(result.status,'invalid_archive',result.reason)
        self.assertFalse(result.outputs);load.assert_not_called()
        self.assertEqual(self.hashes(directory),before)

    def attempt(self,directory=None,cancel=None):
        calls=[]
        def profile(frame,event,arg):
            if event=='call' and frame.f_code is runner._dispatch.__code__:
                calls.append(frame.f_locals['operation'])
        before=self.hashes(directory or self.directory)
        old=sys.getprofile();sys.setprofile(profile)
        try:result=replay.replay_native_stage(directory or self.directory,factory.CROP,cancel=cancel)
        finally:sys.setprofile(old)
        self.assertEqual(self.hashes(directory or self.directory),before)
        return result,calls

    def test_actual_crop_replays_only_target_with_both_frames_and_all_metadata_leaves(self):
        torch.cuda.synchronize();baseline=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
        result,calls=self.attempt();torch.cuda.synchronize()
        self.assertEqual(result.status,'matched',result.reason);self.assertEqual(calls,[contract.OPERATION])
        expected=self.saved.node(factory.CROP).outputs
        for name in ('rgb_image','gray_image'):
            self.assertEqual(result.outputs[name].device.type,'cuda')
            self.assertTrue(torch.equal(result.outputs[name],expected[name]))
        self.assertEqual(result.outputs['metadata'],expected['metadata'])
        self.assertEqual(result.metadata['plan']['pixels'],620*900)
        self.assertEqual(result.metadata['plan']['roi_pixels'],96*128)
        self.assertEqual(result.metadata['plan']['selected_bytes'],3*620*900+4*96*128)
        self.assertEqual(result.metadata['coordinate_frame']['shape'],(96,128))
        self.assertEqual(result.metadata['parent_coordinate_frame']['shape'],(620,900))
        self.assertEqual(result.metadata['frame_lineage']['local_to_parent_xy'],(600,260))
        self.assertGreater(result.metadata['comparison']['by_port']['metadata']['compared_elements'],20)
        self.assertLessEqual(torch.cuda.max_memory_allocated()-baseline,result.metadata['plan']['cuda_bytes'])
        self.assertFalse(result.scientifically_validated)

    def test_full_frame_full_width_partial_single_pixel_and_shifted_rois_preserve_aliases(self):
        for bounds in ((0,0,900,620),(0,100,900,103),(600,260,728,356),(599,260,727,356),(899,619,900,620)):
            with self.subTest(bounds=bounds):
                recipe=_parameter(self.recipe,factory.CROP,'seed_roi_xyxy',list(bounds))
                run=runner.run_native_recipe_path(recipe,self.path)
                source=run.node(card.AFFINE).outputs['corrected_rgb'];products=run.node(factory.CROP).outputs
                x0,y0,x1,y1=bounds
                self.assertTrue(torch.equal(products['rgb_image'],source[y0:y1,x0:x1]))
                alias=products['rgb_image'].untyped_storage().data_ptr()==source.untyped_storage().data_ptr()
                self.assertEqual(alias,y1-y0==1 or x0==0 and x1==900)
                expected_gray=cv2.cvtColor(products['rgb_image'].cpu().numpy()[...,::-1].copy(),cv2.COLOR_BGR2GRAY)
                np.testing.assert_array_equal(products['gray_image'].cpu().numpy(),expected_gray)
                directory=self.export(recipe,run,'roi');result,calls=self.attempt(directory)
                self.assertEqual(result.status,'matched',result.reason);self.assertEqual(calls,[contract.OPERATION])
                self.assertEqual(result.outputs['metadata']['local_to_parent_xy'],(x0,y0))
                self.assertEqual(result.metadata['coordinate_frame']['shape'],(y1-y0,x1-x0))
                del result,run

    def test_all_four_calibrated_and_four_diagnostic_graphs_support_crop_replay(self):
        from tests.test_native_calibrated_range_recipe import selected_recipe as range_recipe
        for kind in ('binary','mean','otsu','range'):
            base=range_recipe() if kind=='range' else selected_recipe(kind)
            for diagnostic in (False,True):
                with self.subTest(kind=kind,diagnostic=diagnostic):
                    recipe=runner._diagnostic_recipe.build_native_watershed_diagnostic_recipe(base) if diagnostic else base
                    run=runner.run_native_recipe_path(recipe,self.path)
                    directory=self.export(recipe,run,'variant');result,calls=self.attempt(directory)
                    self.assertEqual(result.status,'matched',result.reason);self.assertEqual(calls,[contract.OPERATION])
                    self.assertEqual(result.metadata['coordinate_frame']['frame_id'],'source.decoded.seed_roi')
                    del result,run

    def test_explicit_parent_identity_and_transform_survive_the_derived_local_frame(self):
        bgr=fixture();rgb=torch.tensor(bgr[...,::-1].copy(),device='cuda')
        gray=torch.tensor(cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY),device='cuda')
        parent=CoordinateFrame('arbitrary.valid',(620,900),'top-left','x-right,y-down','pixel','d'*64)
        run=runner.run_native_recipe(self.recipe,gray,rgb,source_sha256='e'*64,
            coordinate_frame=parent,preparation={'rgb':'declared RGB','gray':'declared gray'})
        result,calls=self.attempt(self.export(self.recipe,run,'explicit'))
        self.assertEqual(result.status,'matched',result.reason);self.assertEqual(calls,[contract.OPERATION])
        self.assertEqual(CoordinateFrame.from_dict(dict(result.metadata['parent_coordinate_frame'])),parent)
        self.assertEqual(result.metadata['coordinate_frame']['frame_id'],'arbitrary.valid.seed_roi')
        self.assertNotEqual(result.metadata['coordinate_frame']['transform_sha256'],parent.transform_sha256)

    def test_fixed_metadata_headers_controls_geometry_and_all_direct_producer_keys_fail_before_loading(self):
        outputs=dict(self.saved.node(factory.CROP).outputs);metadata=dict(outputs['metadata'])
        for field,value in (('bounds_xyxy',(599,260,727,356)),('local_to_parent_xy',(True,260)),
                ('gray_conversion','other'),('source_sha256','0'*64),('card_bounding_overlap_pixels',False)):
            with self.subTest(field=field):self.invalid(self.changed(factory.CROP,outputs={**outputs,'metadata':{**metadata,field:value}}))
        for name,raster in (('rgb_image',torch.zeros((96,127,3),device='cuda',dtype=torch.uint8)),
                ('gray_image',torch.zeros((96,128),device='cuda',dtype=torch.float32))):
            self.invalid(self.changed(factory.CROP,outputs={**outputs,name:raster}))
        crop=self.saved.node(factory.CROP)
        for port in crop.input_keys:
            self.invalid(self.changed(factory.CROP,input_keys={**crop.input_keys,port:'0'*64}))
        for owner in (card.AFFINE,card.GEOMETRY,card.PREPROCESS):
            row=self.saved.node(owner)
            key=next(iter(row.input_keys))
            self.invalid(self.changed(owner,input_keys={**row.input_keys,key:'0'*64}))
        geometry=dict(self.saved.node(card.GEOMETRY).outputs['geometry'])
        self.invalid(self.changed(card.GEOMETRY,outputs={**self.saved.node(card.GEOMETRY).outputs,
            'geometry':{**geometry,'bounding_rect':(True,*geometry['bounding_rect'][1:])}}))
        self.invalid(self.changed(card.GEOMETRY,outputs={**self.saved.node(card.GEOMETRY).outputs,
            'geometry':{**geometry,'extra':1}}))
        bad_recipe=_parameter(self.recipe,factory.CROP,'seed_roi_xyxy',[True,260,728,356])
        bad_node=replace(self.saved.node(factory.CROP),parameters=bad_recipe.parameters_for_runner(factory.CROP))
        bad_run=replace(self.saved,recipe_identity=bad_recipe.identity,
            nodes=tuple(bad_node if row.stage_id==factory.CROP else row for row in self.saved.nodes))
        self.invalid(self.export(bad_recipe,bad_run,'bad-selected-control'))

    def test_well_typed_wrong_rgb_gray_and_overlap_are_numeric_mismatches(self):
        outputs=self.saved.node(factory.CROP).outputs
        for name in ('rgb_image','gray_image'):
            changed=outputs[name].clone();changed.reshape(-1)[0]+=1
            result,calls=self.attempt(self.changed(factory.CROP,outputs={**outputs,name:changed}))
            self.assertEqual(result.status,'mismatch',result.reason);self.assertEqual(calls,[contract.OPERATION])
            self.assertEqual(result.metadata['comparison']['by_port'][name]['mismatched_elements'],1)
        metadata={**outputs['metadata'],'card_bounding_overlap_pixels':1}
        result,calls=self.attempt(self.changed(factory.CROP,outputs={**outputs,'metadata':metadata}))
        self.assertEqual(result.status,'mismatch',result.reason);self.assertEqual(calls,[contract.OPERATION])
        self.assertEqual(result.metadata['comparison']['by_port']['metadata']['mismatched_elements'],1)

    def test_admission_deleted_origin_strict_cancellation_and_private_buffer_mutation(self):
        with patch.object(replay,'MAX_TYPED_SELECTED_BYTES',1),patch.object(archives,'load_native_recipe_result') as load:
            result=replay.replay_native_stage(self.directory,factory.CROP)
        self.assertEqual(result.status,'unavailable',result.reason);self.assertFalse(result.outputs);load.assert_not_called()
        for cancel,status in ((lambda:True,'cancelled'),(lambda:1,'failed')):
            result,calls=self.attempt(cancel=cancel)
            self.assertEqual(result.status,status,result.reason);self.assertFalse(result.outputs);self.assertFalse(calls)
        raw=self.path.read_bytes();self.path.unlink()
        try:
            result,calls=self.attempt()
            self.assertEqual(result.status,'matched',result.reason);self.assertEqual(calls,[contract.OPERATION])
        finally:self.path.write_bytes(raw)
        for name in ('rgb','gray'):
            changed=[]
            def cancel():
                frame=sys._getframe()
                while frame:
                    if frame.f_code is factory._crop.__code__:
                        raster=frame.f_locals.get(name)
                        if raster is not None and not changed:
                            raster.add_(1);changed.append(True)
                            raise ValueError('caller exception after private mutation')
                        break
                    frame=frame.f_back
                return False
            result,calls=self.attempt(cancel=cancel)
            self.assertTrue(changed);self.assertEqual(result.status,'failed',result.reason);self.assertFalse(result.outputs)
            self.assertIn('changed',result.reason)

    def test_selected_expected_fresh_buffers_context_and_contract_mutations_withhold_outputs(self):
        for phase,port in (('inputs','corrected_rgb'),('expected_products','rgb_image'),
                ('expected_products','gray_image'),('outputs','rgb_image'),('outputs','gray_image')):
            changed=[]
            def cancel():
                frame=sys._getframe()
                while frame:
                    if frame.f_code is replay.replay_native_stage.__code__:
                        products=frame.f_locals.get(phase)
                        if products and not changed:
                            products[port].add_(1);changed.append(True)
                            raise ValueError('caller exception after selected buffer mutation')
                        break
                    frame=frame.f_back
                return False
            result,_=self.attempt(cancel=cancel)
            self.assertTrue(changed);self.assertEqual(result.status,'failed',result.reason);self.assertFalse(result.outputs)
        for raise_after in (False,True):
            changed=[]
            def change_context():
                frame=sys._getframe()
                while frame:
                    if frame.f_code is replay._crop_frame.__code__:
                        context=frame.f_locals.get('context')
                        if context is not None and not changed:
                            object.__setattr__(context.local,'transform_sha256','f'*64);changed.append(True)
                            if raise_after:raise ValueError('caller exception after context mutation')
                        break
                    frame=frame.f_back
                return False
            result,_=self.attempt(cancel=change_context)
            self.assertTrue(changed);self.assertEqual(result.status,'failed',result.reason);self.assertFalse(result.outputs)
        changed=[];original=contract.COMPARE_ELEMENTS
        def change_contract():
            frame=sys._getframe()
            while frame:
                if frame.f_code is replay.replay_native_stage.__code__:
                    if frame.f_locals.get('outputs') and not changed:
                        contract.COMPARE_ELEMENTS=17;changed.append(True)
                    break
                frame=frame.f_back
            return False
        try:result,_=self.attempt(cancel=change_contract)
        finally:contract.COMPARE_ELEMENTS=original
        self.assertTrue(changed);self.assertEqual(result.status,'failed',result.reason);self.assertFalse(result.outputs)

    def test_runtime_incompatibility_precedes_current_geometry_version_checks_and_array_loading(self):
        node=self.saved.node(card.GEOMETRY)
        geometry={**node.outputs['geometry'],'numpy_version':'previous runtime'}
        changed=replace(node,outputs={**node.outputs,'geometry':geometry})
        run=replace(self.saved,runtime={**self.saved.runtime,'numpy':'previous runtime'},
            nodes=tuple(changed if row.stage_id==node.stage_id else row for row in self.saved.nodes))
        directory=self.export(self.recipe,run,'older-runtime')
        with patch.object(archives,'load_native_recipe_result') as load:
            result=replay.replay_native_stage(directory,factory.CROP)
        self.assertEqual(result.status,'unavailable',result.reason);self.assertFalse(result.outputs);load.assert_not_called()

    def test_public_crop_cancellation_family_caller_identity_and_mutation_priority(self):
        saved=self.saved
        inputs={'corrected_rgb':saved.node(card.AFFINE).outputs['corrected_rgb'],
            'source_frame':saved.nodes[0].outputs['source_frame'],
            'geometry':saved.node(card.GEOMETRY).outputs['geometry'],
            'bounds':saved.node(card.PREPROCESS).outputs['bounds']}
        parameters={'seed_roi_xyxy':[600,260,728,356]}
        calls=[]
        def observe():
            calls.append(None)
            return False
        factory.execute_stage(contract.OPERATION,inputs,parameters,observe)
        self.assertGreaterEqual(len(calls),3)
        for stop in (1,max(2,len(calls)//2),len(calls)):
            count=[]
            def cancel():
                count.append(None)
                return len(count)==stop
            with self.subTest(stop=stop),self.assertRaises(card.card.AutomaticCardCancelled) as caught:
                factory.execute_stage(contract.OPERATION,inputs,parameters,cancel)
            self.assertIsInstance(caught.exception.__cause__,card.correction.CorrectionCancelled)
        caller_error=ValueError('unchanged crop caller exception')
        def caller():
            raise caller_error
        with self.assertRaises(ValueError) as caught:
            factory.execute_stage(contract.OPERATION,inputs,parameters,caller)
        self.assertIs(caught.exception,caller_error)
        for raises in (False,True):
            mutated=[]
            def corrupt():
                frame=sys._getframe()
                while frame:
                    if frame.f_code is factory._crop.__code__:
                        gray=frame.f_locals.get('gray')
                        if gray is not None and not mutated:
                            gray.fill_(17);mutated.append(True)
                            if raises:raise caller_error
                            return True
                    frame=frame.f_back
                return False
            with self.subTest(raises=raises),self.assertRaises(card.correction.CorrectionMutation):
                factory.execute_stage(contract.OPERATION,inputs,parameters,corrupt)
            self.assertTrue(mutated)

    def test_actual_async_qt_normal_crop_replay_summary_preview_and_active_result(self):
        app=QApplication.instance() or QApplication([])
        dialog=NativeRecipeDialog(self.recipe);dialog.set_source_path(self.path)
        dialog.resize(1050,650);dialog.show();dialog.start_run()
        def drain():
            deadline=time.monotonic()+120
            while dialog._worker is not None and time.monotonic()<deadline:
                app.processEvents();time.sleep(.01)
            self.assertIsNone(dialog._worker)
        try:
            drain();saved=dialog.result;self.assertIsNotNone(saved)
            dialog._select_stage(factory.CROP);dialog.start_stage_replay(self.directory);drain()
            self.assertEqual(dialog.replay_result.status,'matched',dialog.replay_result.reason)
            self.assertIs(dialog.result,saved)
            self.assertIn('Calibrated seed crop (half-open parent xyxy)',dialog.facts.toPlainText())
            self.assertIn('card detection and correction are not recomputed',dialog.facts.toPlainText())
            self.assertFalse(dialog.preview.pixmap().isNull())
        finally:dialog.close();app.processEvents()


if __name__=='__main__':unittest.main()
