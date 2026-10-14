"""Public small-object ROI equality and complete native working paths."""
from dataclasses import replace
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch

from seedvision.benchmarking import native_recipe as runner
from seedvision.benchmarking import native_fill_roi_recipe as factory
from seedvision.benchmarking import native_morphology_roi_recipe as morphology
from seedvision.benchmarking import native_stage_replay as replay
from seedvision.benchmarking.recipes import RecipeSpec
from seedvision.cuda import plantcv_fill as primitive
from seedvision.cuda.tensor_callback import TensorCallbackMutation
from seedvision.export import native_results as archives
from seedvision.segmentation.baseline import PipelineAnalysisCache
from tests.test_native_recipe import fixture, selected_recipe, FRAME, SOURCE, PREPARATION, REFERENCE
from tests.test_morphology_roi_recipe import _change

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')


def _recipe():
    return factory.build_native_fill_roi_recipe(runner.build_native_seed_recipe(selected_recipe()))


REFERENCE_SCRIPT = r'''
import hashlib, importlib, inspect, json, sys
from importlib.metadata import version
from pathlib import Path
import numpy as np
from plantcv import plantcv as pcv
from plantcv.plantcv.classes import Objects
from skimage.morphology import remove_small_objects
assert version('plantcv') == '4.11.3'
assert version('scikit-image') == '0.25.2'
assert inspect.signature(remove_small_objects).parameters['connectivity'].default == 1
assert 'max_size' not in inspect.signature(remove_small_objects).parameters
for name,digest in [('_helpers','c05b0c944210657332ce307f2066caab55f2ebc8d8fea11988db3e476e7cdfba'),('fill','73c003887b771504d9e89a8b63bab6211f0f2f22474d1ad9d429aea167d05803')]:
 module=importlib.import_module('plantcv.plantcv.'+name)
 assert hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()==digest
pcv.params.debug=None
source=np.load(sys.argv[1],allow_pickle=False)['mask']
cases=json.loads(Path(sys.argv[2]).read_text())
results={}
for index,case in enumerate(cases):
 bounds=case['roi'];roi=None
 if bounds is not None:
  x0,y0,x1,y1=bounds
  contour=[np.array([[[x0,y0]],[[x0,y1]],[[x1,y1]],[[x1,y0]]],np.int32)]
  hierarchy=np.array([[[-1,-1,-1,-1]]],np.int32)
  roi=Objects(contours=[contour],hierarchy=[hierarchy])
 output=pcv.fill(bin_img=source,size=case['size'],roi=roi)
 assert output.dtype == (np.dtype(np.uint8) if roi is None else np.dtype(np.int_))
 results[str(index)]=output
 if bounds is not None and x1<source.shape[1] and y1<source.shape[0]:
  rectangle=pcv.roi.rectangle(source,x=x0,y=y0,w=x1-x0+1,h=y1-y0+1)
  assert np.array_equal(output,pcv.fill(bin_img=source,size=case['size'],roi=rectangle))
np.savez(sys.argv[3],**results)
'''


class FillROIContractTests(unittest.TestCase):
    def test_all_twenty_eight_variants_and_reverse_order_preserve_base_contracts(self):
        self.assertEqual(len(factory.BASE_IDS), 28)
        for base_id in factory.BASE_IDS:
            with self.subTest(base=base_id):
                base = factory.canonical_base(base_id)
                before = base.to_dict()
                recipe = factory.build_native_fill_roi_recipe(base)
                runner._validate_recipe(recipe)
                self.assertEqual(RecipeSpec.from_json(recipe.to_json()).identity, recipe.identity)
                self.assertEqual(base.to_dict(), before)
                self.assertEqual(runner._diagnostic_recipe.base_recipe_id(recipe.recipe_id),
                                 runner._diagnostic_recipe.base_recipe_id(base_id))
                for stage, prior in zip(recipe.stages, base.stages, strict=True):
                    if stage.operation_id == factory.OPERATION:
                        self.assertEqual(stage.parameters[:-1], prior.parameters)
                        self.assertIsNone(stage.parameters[-1].value)
                    else:
                        self.assertEqual(stage, prior)
                if base_id in morphology.BASE_IDS:
                    selected = _change(recipe, factory.OPERATION, size=5, roi_bounds_xyxy=[1,2,3,4])
                    composed = factory.add_morphology_roi(selected)
                    expected = _change(factory.build_native_fill_roi_recipe(
                        morphology.build_native_morphology_roi_recipe(base)),
                        factory.OPERATION, size=5, roi_bounds_xyxy=[1,2,3,4])
                    self.assertEqual(composed.identity, expected.identity)
                    self.assertEqual(composed.layout, selected.layout)
                    runner._validate_recipe(composed)

    def test_bounds_work_and_full_canvas_memory_admission(self):
        controls = {'size': 2, factory.PARAMETER: [2,3,12,13]}
        local = replay._plan((20,30), factory.OPERATION, controls, 1200)
        full = replay._plan((20,30), factory.OPERATION, {'size':2}, 1200)
        self.assertEqual(local['work_details']['roi_pixels'], 100)
        self.assertEqual(local['work_details']['connectivity'], 4)
        self.assertLess(local['work'], full['work'])
        self.assertEqual(local['cuda_bytes'], full['cuda_bytes'])
        for value in ([0,0,31,20], [0,0,0,1], [True,0,1,1], [0.,0,1,1], '0,0,1,1'):
            with self.subTest(value=value), self.assertRaises((TypeError,ValueError)):
                replay._plan((20,30), factory.OPERATION, dict(controls, roi_bounds_xyxy=value), 1200)
        for size in (0, True, 1.5):
            with self.subTest(size=size), self.assertRaises(runner.NativeRecipeError):
                replay._plan((20,30), factory.OPERATION, dict(controls,size=size), 1200)
        with self.assertRaises(runner.NativeRecipeError):
            replay._plan((1,7_340_033), factory.OPERATION,
                         {'size':1,'roi_bounds_xyxy':[0,0,1,1]}, 14_680_066)

    def test_historical_mask_source_closure_tracks_new_fill_helpers(self):
        from seedvision.benchmarking import native_mask_stage as bridge
        from seedvision.cuda import plantcv_morphology
        baseline=bridge._implementation_digest()
        self.assertEqual(baseline,bridge._implementation_digest())
        for name in ('_roi_bounds', '_filter_crop'):
            with self.subTest(name=name), patch.object(primitive, name, lambda *args, **kwargs: None):
                with self.assertRaises(bridge.NativeMaskStageError):
                    bridge._implementation_digest()
        for module in (primitive,plantcv_morphology):
            with self.subTest(module=module.__name__), patch.object(module,'_tensor_callback',
                    SimpleNamespace(checked_cancel=lambda *args,**kwargs:False)):
                with self.assertRaises(bridge.NativeMaskStageError):
                    bridge._implementation_digest()
        self.assertEqual(baseline,bridge._implementation_digest())


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class FillROIIntegrationTests(unittest.TestCase):
    @unittest.skipUnless(REFERENCE.is_file(), 'Pinned PlantCV environment required')
    def test_104_exact_public_outputs_on_thin_border_shifted_full_and_strided_rois(self):
        rng = np.random.default_rng(702)
        mask = np.where(rng.random((17,23))>.53,255,0).astype(np.uint8)
        mask[0,1:8]=255;mask[6:11,7:13]=255;mask[8,9]=0
        rois = (None,[0,0,23,17],[0,0,1,1],[22,16,23,17],
                [3,5,18,6],[8,2,9,15],[0,2,8,12],[15,2,23,12],
                [3,0,15,7],[3,10,15,17],[2,3,12,13],[3,3,13,13],[10,8,11,9])
        cases = [{'size':size,'roi':roi} for size in (1,2,3,4,7,16,391,392) for roi in rois]
        self.assertEqual(len(cases),104)
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            np.savez(root/'input.npz',mask=mask)
            (root/'cases.json').write_text(json.dumps(cases))
            completed=subprocess.run([str(REFERENCE),'-c',REFERENCE_SCRIPT,str(root/'input.npz'),
                str(root/'cases.json'),str(root/'output.npz')],capture_output=True,text=True,timeout=90)
            self.assertEqual(completed.returncode,0,completed.stdout+completed.stderr)
            source=torch.empty((17,46),dtype=torch.uint8,device='cuda')[:,::2]
            source.copy_(torch.tensor(mask,device='cuda'));before=source.clone()
            with np.load(root/'output.npz',allow_pickle=False) as expected:
                for index,case in enumerate(cases):
                    bounds=None if case['roi'] is None else tuple(case['roi'])
                    actual=primitive.fill(source,case['size'],roi=bounds)
                    np.testing.assert_array_equal(actual.cpu().numpy(),expected[str(index)],err_msg=str(case))
                    self.assertEqual(actual.device,source.device)
                    self.assertEqual(actual.dtype,torch.uint8)
                    self.assertEqual(actual.shape,source.shape)
                    self.assertNotEqual(actual.data_ptr(),source.data_ptr())
            torch.testing.assert_close(source,before)

    @unittest.skipUnless(REFERENCE.is_file(), 'Pinned PlantCV environment required')
    def test_21_public_default_area_boundaries_empty_and_full_masks(self):
        structured=np.zeros((20,30),dtype=np.uint8)
        structured[:7,:9]=255       # 63 pixels
        structured[:8,12:20]=255    # 64 pixels
        structured[12:17,:13]=255   # 65 pixels
        masks=(structured,np.zeros_like(structured),np.full_like(structured,255))
        compared=0
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            for index,mask in enumerate(masks):
                rois=(None,[0,0,30,20],[4,0,30,20]) if index==0 else (None,[0,0,30,20])
                sizes=(63,64,65) if index==0 else (1,600,601)
                cases=[{'size':size,'roi':roi} for size in sizes for roi in rois]
                np.savez(root/'input.npz',mask=mask);(root/'cases.json').write_text(json.dumps(cases))
                completed=subprocess.run([str(REFERENCE),'-c',REFERENCE_SCRIPT,str(root/'input.npz'),
                    str(root/'cases.json'),str(root/'output.npz')],capture_output=True,text=True,timeout=90)
                self.assertEqual(completed.returncode,0,completed.stdout+completed.stderr)
                source=torch.tensor(mask,device='cuda');before=source.clone()
                with np.load(root/'output.npz',allow_pickle=False) as expected:
                    for case_id,case in enumerate(cases):
                        bounds=None if case['roi'] is None else tuple(case['roi'])
                        actual=primitive.fill(source,case['size'],roi=bounds)
                        np.testing.assert_array_equal(actual.cpu().numpy(),expected[str(case_id)],err_msg=str(case))
                        if index==0 and bounds is None:
                            for area,region in ((63,(slice(0,7),slice(0,9))),
                                                (64,(slice(0,8),slice(12,20))),
                                                (65,(slice(12,17),slice(0,13)))):
                                self.assertEqual(bool(actual[region].any()),area>=case['size'])
                        compared+=1
                torch.testing.assert_close(source,before)
        self.assertEqual(compared,21)

    def test_crop_local_connectivity_area_equality_holes_exterior_and_shortcuts(self):
        source=torch.zeros((10,14),dtype=torch.uint8,device='cuda')
        source[0,1:7]=255
        source[2,3]=255;source[3,4]=255
        source[5,1:3]=255
        source[6:8,6]=255;source[7,7]=255
        source[5:9,10:14]=255;source[6:8,11:13]=0
        before=source.clone();bounds=(4,0,14,10)
        actual=primitive.fill(source,4,roi=bounds)
        expected=before.clone()
        expected[0,4:7]=0
        expected[3,4]=0;expected[6:8,6]=0;expected[7,7]=0
        torch.testing.assert_close(actual,expected)
        full=primitive.fill(source,4)
        self.assertTrue(bool((full[0,4:7]==255).all()))
        self.assertTrue(bool((actual[6:8,11:13]==0).all()))
        equal=primitive.fill(source,3,roi=bounds)
        self.assertTrue(bool((equal[0,4:7]==255).all()))
        self.assertTrue(bool((equal[6:8,6]==255).all()))
        torch.testing.assert_close(primitive.fill(source,1,roi=bounds),source)
        removed=primitive.fill(source,10**100,roi=bounds)
        self.assertFalse(bool(removed[:,4:].any()))
        torch.testing.assert_close(removed[:,:4],source[:,:4])
        torch.testing.assert_close(source,before)

    def test_invalid_bounds_strict_cancel_and_input_preservation(self):
        source=torch.full((12,14),255,dtype=torch.uint8,device='cuda');before=source.clone()
        for bounds in ((0,0,15,12),(0,0,0,1),(True,0,1,1),[0,0,1,1]):
            with self.subTest(bounds=bounds), patch.object(primitive,'_filter_crop',side_effect=AssertionError('allocated')):
                with self.assertRaises((TypeError,ValueError)):
                    primitive.fill(source,2,roi=bounds)
        with self.assertRaises(TypeError):
            primitive.fill(source,2,cancel=lambda:1)
        polls=0
        def cancel():
            nonlocal polls
            polls+=1
            return polls>=8
        with self.assertRaises(primitive.FillCancelledError):
            primitive.fill(source,2,roi=(2,2,12,10),cancel=cancel)
        torch.testing.assert_close(source,before)

    def test_callback_mutation_of_union_area_crop_and_canvas_buffers_withholds_output(self):
        targets=(('fill','mask'),('fill','crop'),('fill','filtered'),('fill','output'),
                 ('_filter_crop','foreground'),('_filter_crop','parent'),('_filter_crop','areas'),
                 ('_filter_crop','retained_roots'),('_filter_crop','output'),
                 ('_union_direction','adjacent_flat'),('_union_edge_batch','edge_a'),
                 ('_union_edge_batch','root_a'),('_compress_roots','parent'),('_component_areas','areas'))
        for function,target in targets:
            for outcome in ('false','true','raise'):
                with self.subTest(function=function,target=target,outcome=outcome):
                    source=torch.full((12,14),255,dtype=torch.uint8,device='cuda');mutated=False
                    def cancel():
                        nonlocal mutated
                        frame=inspect.currentframe().f_back
                        while frame is not None:
                            if frame.f_code is getattr(primitive,function).__code__ and target in frame.f_locals:
                                value=frame.f_locals[target]
                                if torch.is_tensor(value):
                                    value.fill_(0);mutated=True
                                    if outcome=='raise':raise LookupError('callback changed buffer')
                                    return outcome=='true'
                            frame=frame.f_back
                        return False
                    with self.assertRaises(TensorCallbackMutation):
                        primitive.fill(source,2,roi=(2,2,12,10),cancel=cancel)
                    self.assertTrue(mutated)
        source=torch.full((12,14),255,dtype=torch.uint8,device='cuda')
        def storage_mutation():
            storage=source.untyped_storage();storage.resize_(storage.nbytes()+1)
            raise LookupError('resized')
        with self.assertRaises(TensorCallbackMutation):
            primitive.fill(source,2,cancel=storage_mutation)
        def unchanged_error():raise LookupError('unchanged caller error')
        with self.assertRaisesRegex(LookupError,'unchanged caller error'):
            primitive.fill(source,2,cancel=unchanged_error)

    def test_fill_roi_edits_reuse_ancestors_and_recompute_only_descendants(self):
        gray,rgb=fixture();gray,rgb=torch.tensor(gray,device='cuda'),torch.tensor(rgb,device='cuda')
        recipe=_change(_recipe(),factory.OPERATION,size=200,roi_bounds_xyxy=[20,20,70,70])
        cache=PipelineAnalysisCache()
        def run(selected):
            return runner.run_native_recipe(selected,gray,rgb,source_sha256=SOURCE,
                coordinate_frame=FRAME,preparation=PREPARATION,cache=cache)
        first=run(recipe);second=run(recipe)
        self.assertEqual(len(cache.last_reused_nodes),len(recipe.stages))
        stage=next(s.stage_id for s in recipe.stages if s.operation_id==factory.OPERATION)
        self.assertIs(second.node(stage).outputs['mask'],first.node(stage).outputs['mask'])
        shifted=_change(recipe,factory.OPERATION,roi_bounds_xyxy=[21,20,71,70])
        changed=run(shifted)
        self.assertEqual(changed.node(stage).status,'executed',changed.node(stage).reason)
        ancestors={s.stage_id for s in recipe.stages if s.operation_id in
                   ('external.image_inputs','filter.median_blur','threshold.binary')}
        self.assertEqual(set(cache.last_reused_nodes),ancestors)
        expected=primitive.fill(first.node(next(s.stage_id for s in recipe.stages
            if s.operation_id=='threshold.binary')).outputs['mask'],200,roi=(21,20,71,70))
        torch.testing.assert_close(changed.node(stage).outputs['mask'],expected)

    def test_null_and_roi_saved_replay_execute_only_fill_and_preserve_archive(self):
        gray,rgb=fixture();gray,rgb=torch.tensor(gray,device='cuda'),torch.tensor(rgb,device='cuda')
        for bounds in (None,[20,20,70,70]):
            recipe=_change(_recipe(),factory.OPERATION,roi_bounds_xyxy=bounds)
            result=runner.run_native_recipe(recipe,gray,rgb,source_sha256=SOURCE,
                coordinate_frame=FRAME,preparation=PREPARATION)
            stage=next(s.stage_id for s in recipe.stages if s.operation_id==factory.OPERATION)
            with tempfile.TemporaryDirectory() as temporary:
                saved=Path(temporary)/'saved';archives.export_native_recipe_result(saved,recipe,result)
                before={str(p.relative_to(saved)):p.read_bytes() for p in saved.rglob('*') if p.is_file()}
                calls=[]
                def profile(frame,event,arg):
                    if event=='call' and frame.f_code is runner._dispatch.__code__:calls.append(frame.f_locals['operation'])
                sys.setprofile(profile)
                try:replayed=replay.replay_native_stage(saved,stage)
                finally:sys.setprofile(None)
                self.assertEqual(replayed.status,'matched',replayed.reason)
                self.assertEqual(calls,[factory.OPERATION])
                torch.testing.assert_close(replayed.outputs['mask'],result.node(stage).outputs['mask'])
                self.assertEqual(before,{str(p.relative_to(saved)):p.read_bytes() for p in saved.rglob('*') if p.is_file()})

    def test_async_desktop_both_action_orders_preserve_edits_layout_keep_and_saved_replay(self):
        from PySide6.QtWidgets import QApplication
        from seedvision.ui.native_recipe_dialog import NativeRecipeDialog
        app=QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);source=root/'input.png';gray,_=fixture()
            self.assertTrue(cv2.imwrite(str(source),np.repeat(gray[...,None],3,axis=2)))
            kept=[];dialog=NativeRecipeDialog(runner.build_native_seed_recipe(selected_recipe()),keep_in_project=kept.append)
            dialog.show()
            try:
                moved=dialog.recipe.stages[0].stage_id;dialog.canvas.node_items[moved].setPos(123,456)
                dialog.fill_roi_button.click()
                stage=next(s.stage_id for s in dialog.recipe.stages if s.operation_id==factory.OPERATION)
                dialog._parameter_changed(stage,factory.PARAMETER,'[20,20,70,70]')
                dialog.morphology_roi_button.click()
                self.assertIn(dialog.recipe.recipe_id,factory.RECIPE_IDS)
                self.assertIn(factory.base_recipe_id(dialog.recipe.recipe_id),morphology.RECIPE_IDS)
                self.assertEqual(tuple(dialog.recipe.parameters_for_runner(stage)[factory.PARAMETER]),(20,20,70,70))
                self.assertEqual((dialog.canvas.node_items[moved].pos().x(),dialog.canvas.node_items[moved].pos().y()),(123,456))
                # The forward action order resolves to the same canonical identity.
                forward=NativeRecipeDialog(runner.build_native_seed_recipe(selected_recipe()))
                forward.morphology_roi_button.click();forward.fill_roi_button.click()
                forward._parameter_changed(stage,factory.PARAMETER,'[20,20,70,70]')
                self.assertEqual(replace(forward.recipe,layout=dialog.recipe.layout).identity,dialog.recipe.identity)
                forward.close();forward.deleteLater()
                dialog.set_source_path(source);dialog.start_run()
                self.assertFalse(dialog.fill_roi_button.isEnabled())
                deadline=time.monotonic()+45
                while dialog._worker is not None and time.monotonic()<deadline:app.processEvents();time.sleep(.01)
                app.processEvents();self.assertIsNone(dialog._worker)
                self.assertIsNotNone(dialog.result,dialog.status_label.text());node=dialog.result.node(stage)
                self.assertEqual(node.status,'executed',node.reason)
                dialog._select_stage(stage);self.assertFalse(dialog.preview.pixmap().isNull())
                selected=dialog._selected_recipe_with_layout();dialog.project_button.click()
                self.assertEqual(kept,[selected]);reopened=NativeRecipeDialog(kept[0])
                self.assertEqual(tuple(reopened.recipe.parameters_for_runner(stage)[factory.PARAMETER]),(20,20,70,70))
                reopened.close();reopened.deleteLater()
                saved=root/'saved';archives.export_native_recipe_result(saved,selected,dialog.result)
                source.unlink();dialog.start_stage_replay(saved)
                self.assertFalse(dialog.fill_roi_button.isEnabled())
                deadline=time.monotonic()+45
                while dialog._worker is not None and time.monotonic()<deadline:app.processEvents();time.sleep(.01)
                app.processEvents();self.assertIsNone(dialog._worker)
                self.assertEqual(dialog.replay_result.status,'matched',dialog.status_label.text())
                self.assertIs(dialog.result.node(stage),node)
            finally:dialog.close();app.processEvents()

    def test_calibrated_dual_roi_diagnostic_project_batch_and_local_replay(self):
        from seedvision.benchmarking import native_calibrated_seed_recipe as calibrated
        from seedvision.benchmarking import native_card_recipe as card
        from seedvision.benchmarking import native_watershed_diagnostic_recipe as diagnostic
        from seedvision.export.project_native_graphs import export_project_native_graphs
        from seedvision.persistence.project_analysis import ProjectAnalysisStore,ProjectImageSpec
        from seedvision.persistence.analysis_settings import analysis_settings_profile_from_graph
        from seedvision.pipeline import build_default_pipeline
        from tests import test_native_calibrated_seed_recipe as calibrated_tests
        from tests.test_native_result_export import _output
        recipe=factory.build_native_fill_roi_recipe(morphology.build_native_morphology_roi_recipe(
            diagnostic.build_native_watershed_diagnostic_recipe(calibrated_tests.selected_recipe())))
        recipe=_change(recipe,factory.OPERATION,size=80,roi_bounds_xyxy=[20,20,70,70])
        recipe=_change(recipe,'morphology.erode',roi_bounds_xyxy=[25,25,75,75])
        stage=next(s.stage_id for s in recipe.stages if s.operation_id==factory.OPERATION)
        erode=next(s.stage_id for s in recipe.stages if s.operation_id=='morphology.erode')
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);workspace=root/'workspace';workspace.mkdir();(workspace/'config').mkdir()
            shutil.copyfile(Path(__file__).parents[1]/'config/traits.json',workspace/'config/traits.json')
            source=workspace/'seeds.png';image=calibrated_tests.fixture();self.assertTrue(cv2.imwrite(str(source),image))
            actual=runner.run_native_recipe_path(recipe,source)
            self.assertEqual(actual.node(stage).status,'executed',actual.node(stage).reason)
            incoming=next(e for e in recipe.dependencies if e.target_stage==stage and e.target_port=='mask')
            input_mask=actual.node(incoming.source_stage).outputs[incoming.source_port]
            torch.testing.assert_close(actual.node(stage).outputs['mask'],primitive.fill(input_mask,80,roi=(20,20,70,70)))
            store=ProjectAnalysisStore(workspace)
            master=store.capture_and_save(analysis_settings=analysis_settings_profile_from_graph(build_default_pipeline()),
                images=(ProjectImageSpec(source,image.shape[:2]),),native_recipes=(recipe,),
                destination=workspace/'seeds.seedfiddle-project.json')
            document=store.load(master).document;self.assertEqual(document.native_recipes,(recipe,))
            ledger=export_project_native_graphs((),source,root/'runs',expected_source_sha256=document.images[0].source.sha256,
                project_root=workspace,native_recipes=document.native_recipes)
            row=json.loads(ledger.read_bytes())['graphs'][0];self.assertEqual(row['status'],'executed',row['reason'])
            saved=ledger.parent/Path(row['report']).parent;report,arrays=archives.load_native_recipe_result(saved)
            self.assertFalse(report['scientifically_validated'])
            targets=((stage,'mask'),(erode,'mask'),(calibrated.CROP,'rgb_image'),
                     (card.AFFINE,'corrected_rgb'),(diagnostic.STAGE,'diagnostic'))
            for stage_id,port in targets:
                np.testing.assert_array_equal(arrays[_output(report,stage_id,port)['name']],
                                              actual.node(stage_id).outputs[port].cpu().numpy())
            source.unlink()
            for stage_id,port in targets:
                calls=[]
                def profile(frame,event,arg):
                    if event=='call' and frame.f_code is runner._dispatch.__code__:calls.append(frame.f_locals['operation'])
                sys.setprofile(profile)
                try:result=replay.replay_native_stage(saved,stage_id)
                finally:sys.setprofile(None)
                self.assertEqual(result.status,'matched',result.reason)
                self.assertEqual(calls,[recipe.stage(stage_id).operation_id])
                torch.testing.assert_close(result.outputs[port],actual.node(stage_id).outputs[port])
                if stage_id in (stage,erode,diagnostic.STAGE):
                    self.assertEqual(result.metadata['frame_scope'],'calibrated_seed_roi')
                    self.assertEqual(tuple(result.metadata['coordinate_frame']['shape']),(96,128))
                    self.assertEqual(result.metadata['frame_lineage']['local_to_parent_xy'],(600,260))


if __name__=='__main__':unittest.main()
