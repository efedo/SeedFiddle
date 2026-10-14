"""Version compatibility, public median ROI values and native working paths."""
from dataclasses import replace
import hashlib
import itertools
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch

from seedvision.benchmarking import native_recipe as runner
from seedvision.benchmarking import native_median_roi_recipe as factory
from seedvision.benchmarking import native_fill_roi_recipe as fill
from seedvision.benchmarking import native_morphology_roi_recipe as morphology
from seedvision.benchmarking import native_stage_replay as replay
from seedvision.benchmarking.recipes import RecipeSpec
from seedvision.cuda import plantcv_median as primitive
from seedvision.export import native_results as archives
from seedvision.persistence.native_recipes import (native_recipes_from_payload,
    native_recipes_to_payload, select_native_recipe)
from seedvision.segmentation.baseline import PipelineAnalysisCache
from tests.test_native_recipe import fixture, selected_recipe, FRAME, SOURCE, PREPARATION
from tests.test_morphology_roi_recipe import _change
from tests.test_plantcv_median import _direct_public_median, REFERENCE_ENV_READY

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')


def _base():
    return runner.build_native_seed_recipe(selected_recipe())


def _add(recipe, control):
    if control == 'median':
        recipe = factory.build_native_median_roi_recipe(recipe)
        return _change(recipe, factory.OPERATION, ksize=[2,3], roi_bounds_xyxy=[20,20,70,70])
    if control == 'fill':
        recipe = (factory.add_fill_roi(recipe) if factory.is_median_roi_recipe(recipe)
                  else fill.build_native_fill_roi_recipe(recipe))
        return _change(recipe, fill.OPERATION, size=80, roi_bounds_xyxy=[21,21,71,71])
    recipe = (factory.add_morphology_roi(recipe) if factory.is_median_roi_recipe(recipe)
              else fill.add_morphology_roi(recipe) if recipe.recipe_id in fill.RECIPE_IDS
              else morphology.build_native_morphology_roi_recipe(recipe))
    recipe = _change(recipe, 'morphology.erode', ksize=2, roi_bounds_xyxy=[22,22,72,72])
    return _change(recipe, 'morphology.dilate', iterations=2, roi_bounds_xyxy=[23,23,73,73])


class MedianROIContractTests(unittest.TestCase):
    def test_versions_coexist_and_forged_contracts_reject(self):
        self.assertEqual(len(set(factory.BASE_IDS)),56)
        old = _base(); before = old.to_dict()
        new = factory.build_native_median_roi_recipe(old)
        self.assertEqual(old.to_dict(), before)
        self.assertEqual(old.recipe_id,new.recipe_id)
        self.assertNotEqual(old.recipe_version,new.recipe_version)
        self.assertEqual(old.dependencies,new.dependencies)
        self.assertEqual(old.layout,new.layout)
        for stage,prior in zip(new.stages,old.stages,strict=True):
            if stage.operation_id == factory.OPERATION:
                self.assertEqual(stage.parameters[:-1],prior.parameters)
                self.assertIsNone(stage.parameters[-1].value)
            else:
                self.assertEqual(stage,prior)
        runner._validate_recipe(old); runner._validate_recipe(new)
        both = select_native_recipe((old,),new)
        self.assertEqual(both,(old,new))
        self.assertEqual(native_recipes_from_payload(native_recipes_to_payload(both)),both)
        changed = _change(new,factory.OPERATION,roi_bounds_xyxy=[1,2,10,12])
        self.assertEqual(select_native_recipe(both,changed),(old,changed))
        self.assertEqual(RecipeSpec.from_json(changed.to_json()).identity,changed.identity)
        for bad in (replace(new,recipe_version=old.recipe_version),
                    replace(old,recipe_version=factory.RECIPE_VERSION),
                    replace(new,provenance=old.provenance),
                    replace(new,stages=old.stages)):
            with self.subTest(version=bad.recipe_version),self.assertRaises(runner.NativeRecipeError):
                runner._validate_recipe(bad)
        with self.assertRaises(ValueError): factory.build_native_median_roi_recipe(new)

    def test_all_six_action_orders_preserve_nondefault_independent_controls(self):
        from seedvision.benchmarking import native_watershed_diagnostic_recipe as diagnostic
        expected = None
        for order in itertools.permutations(('median','fill','morphology')):
            with self.subTest(order=order):
                recipe = replace(_base(), layout=tuple(replace(p,x=p.x+123,y=p.y+456) for p in _base().layout))
                for control in order: recipe = _add(recipe,control)
                runner._validate_recipe(recipe)
                self.assertTrue(factory.is_median_roi_recipe(recipe))
                self.assertEqual(RecipeSpec.from_json(recipe.to_json()).identity,recipe.identity)
                if expected is None: expected = recipe.identity
                self.assertEqual(recipe.identity,expected)
                self.assertEqual(diagnostic.base_recipe_id(recipe.recipe_id),_base().recipe_id)

    def test_crop_work_and_geometry_keep_full_canvas_admission(self):
        controls = {'ksize':[9,9],factory.PARAMETER:[2,3,12,13]}
        local = replay._plan((20,30),factory.OPERATION,controls,1200)
        full = replay._plan((20,30),factory.OPERATION,{'ksize':[9,9]},1200)
        self.assertEqual(local['work_details']['window_values'],8100)
        self.assertEqual(local['work_details']['roi_pixels'],100)
        self.assertEqual(local['work'],8100+16*600)
        self.assertLess(local['work'],full['work'])
        self.assertEqual(local['cuda_bytes'],full['cuda_bytes'])
        for bounds in ([0,0,31,20],[0,0,0,1],[True,0,1,1],[0.,0,1,1],'0,0,1,1'):
            with self.subTest(bounds=bounds),self.assertRaises((TypeError,ValueError)):
                replay._plan((20,30),factory.OPERATION,dict(controls,roi_bounds_xyxy=bounds),1200)
        # Exercise the established rejection contract; do not call the source on it.
        with self.assertRaises(primitive.UnsupportedMedianGeometryError):
            replay._plan((20,30),factory.OPERATION,{'ksize':17,'roi_bounds_xyxy':[2,3,4,13]},1200)
        with self.assertRaises(runner.NativeRecipeError):
            replay._plan((1,7_340_033),factory.OPERATION,
                         {'ksize':1,'roi_bounds_xyxy':[0,0,1,1]},14_680_066)


@unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
class MedianROIIntegrationTests(unittest.TestCase):
    @unittest.skipUnless(REFERENCE_ENV_READY,'Pinned PlantCV environment required')
    def test_116_public_gray_and_binary_outputs_through_graph_dispatch(self):
        rng = np.random.default_rng(701)
        gray = rng.integers(0,256,(17,23),dtype=np.uint8)
        binary = np.where(gray>127,255,0).astype(np.uint8)
        rois = (None,(0,0,23,17),(0,0,1,1),(22,16,23,17),
                (3,5,18,6),(8,2,9,15),(0,2,8,12),(15,2,23,12),
                (3,0,15,7),(3,10,15,17),(2,3,12,13),(3,3,13,13),(10,8,11,9))
        cases = [(gray,kernel,roi) for kernel in (1,2,3,4,5,9,(2,3),(3,4)) for roi in rois]
        cases += [(binary,kernel,roi) for kernel in (1,2,3,(2,3))
                  for roi in (None,(0,0,23,17),(2,3,12,13))]
        self.assertEqual(len(cases),116)
        expected,meta = _direct_public_median(cases)
        self.assertEqual(meta['plantcv'],'4.11.3'); self.assertEqual(meta['scipy'],'1.15.3')
        self.assertEqual(meta['helper_sha256'],primitive.PLANTCV_HELPER_REFERENCE_ENV_SHA256)
        installed = Path(__file__).parents[1]/'artifacts/plantcv-reference-env/Lib/site-packages/plantcv/plantcv/median_blur.py'
        self.assertEqual(hashlib.sha256(installed.read_bytes()).hexdigest(),primitive.PLANTCV_MEDIAN_SOURCE_SHA256)
        inputs = {}
        for name,image in (('gray',gray),('binary',binary)):
            value = torch.empty((17,46),dtype=torch.uint8,device='cuda')[:,::2]
            value.copy_(torch.tensor(image,device='cuda'));inputs[name]=value
        for (image,kernel,bounds),output in zip(cases,expected,strict=True):
            source = inputs['gray' if image is gray else 'binary']; before=source.clone()
            actual=runner._dispatch(factory.OPERATION,{'gray':source},
                {'ksize':list(kernel) if type(kernel) is tuple else kernel,
                 factory.PARAMETER:None if bounds is None else list(bounds)},None)['image']
            np.testing.assert_array_equal(actual.cpu().numpy(),output,err_msg=str((kernel,bounds)))
            self.assertEqual(actual.device,source.device);self.assertEqual(actual.dtype,torch.uint8)
            self.assertNotEqual(actual.data_ptr(),source.data_ptr());torch.testing.assert_close(source,before)
            if bounds is not None:
                outside=np.ones(gray.shape,dtype=bool);x0,y0,x1,y1=bounds;outside[y0:y1,x0:x1]=False
                np.testing.assert_array_equal(output[outside],image[outside])

    def test_invalid_bounds_reject_before_filter_allocation_and_cancel_preserves_input(self):
        source=torch.arange(600,device='cuda').to(torch.uint8).reshape(20,30);before=source.clone()
        with patch.object(primitive,'_median_validated') as worker:
            for bounds in ([0,0,0,1],[0,0,31,20],[False,0,1,1]):
                with self.subTest(bounds=bounds),self.assertRaises((TypeError,ValueError)):
                    runner._dispatch(factory.OPERATION,{'gray':source},{'ksize':3,factory.PARAMETER:bounds},None)
            worker.assert_not_called()
            with self.assertRaises(primitive.MedianCancelledError):
                runner._dispatch(factory.OPERATION,{'gray':source},{'ksize':3,factory.PARAMETER:[2,3,12,13]},lambda:True)
            worker.assert_not_called()
        calls=0
        def cancel():
            nonlocal calls
            calls+=1
            return calls==3
        with patch.object(primitive,'MAX_TEMPORARY_WINDOW_VALUES',27):
            with self.assertRaises(primitive.MedianCancelledError):
                runner._dispatch(factory.OPERATION,{'gray':source},{'ksize':3,factory.PARAMETER:[2,3,12,13]},cancel)
        torch.testing.assert_close(source,before)

    def test_shifted_bounds_and_kernel_edits_invalidate_only_median_descendants(self):
        gray,rgb=fixture();gray[20:35,20:35]=np.random.default_rng(7).integers(0,256,(15,15),dtype=np.uint8)
        gray,rgb=torch.tensor(gray,device='cuda'),torch.tensor(rgb,device='cuda')
        recipe=_change(factory.build_native_median_roi_recipe(_base()),factory.OPERATION,
                       roi_bounds_xyxy=[20,20,70,70])
        cache=PipelineAnalysisCache()
        def run(selected): return runner.run_native_recipe(selected,gray,rgb,source_sha256=SOURCE,
            coordinate_frame=FRAME,preparation=PREPARATION,cache=cache)
        stage=next(s.stage_id for s in recipe.stages if s.operation_id==factory.OPERATION)
        first=run(recipe);second=run(recipe)
        self.assertEqual(len(cache.last_reused_nodes),len(recipe.stages))
        self.assertIs(first.node(stage).outputs['image'],second.node(stage).outputs['image'])
        shifted=_change(recipe,factory.OPERATION,roi_bounds_xyxy=[21,20,71,70])
        changed=run(shifted)
        self.assertEqual(changed.node(stage).status,'executed',changed.node(stage).reason)
        source_stage=next(s.stage_id for s in recipe.stages if s.operation_id=='external.image_inputs')
        self.assertEqual(cache.last_reused_nodes,(source_stage,))
        self.assertFalse(torch.equal(first.node(stage).outputs['image'],changed.node(stage).outputs['image']))
        torch.testing.assert_close(changed.node(stage).outputs['image'],primitive.median_blur(gray,3,roi=(21,20,71,70)))
        kernel=run(_change(shifted,factory.OPERATION,ksize=[2,3]))
        self.assertEqual(cache.last_reused_nodes,(source_stage,))
        torch.testing.assert_close(kernel.node(stage).outputs['image'],primitive.median_blur(gray,(2,3),roi=(21,20,71,70)))

    def test_historical_null_and_explicit_roi_archives_replay_only_target(self):
        gray,rgb=fixture();gray,rgb=torch.tensor(gray,device='cuda'),torch.tensor(rgb,device='cuda')
        old=_base();new=factory.build_native_median_roi_recipe(old)
        for recipe in (old,new,_change(new,factory.OPERATION,ksize=[2,3],roi_bounds_xyxy=[20,20,70,70])):
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
                try:actual=replay.replay_native_stage(saved,stage)
                finally:sys.setprofile(None)
                self.assertEqual(actual.status,'matched',actual.reason);self.assertEqual(calls,[factory.OPERATION])
                torch.testing.assert_close(actual.outputs['image'],result.node(stage).outputs['image'])
                self.assertEqual(before,{str(p.relative_to(saved)):p.read_bytes() for p in saved.rglob('*') if p.is_file()})

    def test_desktop_six_orders_keep_reopen_worker_locks_and_source_less_replay(self):
        from PySide6.QtWidgets import QApplication
        from seedvision.ui.native_recipe_dialog import NativeRecipeDialog
        app=QApplication.instance() or QApplication([]);identities=[]
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);source=root/'input.png';gray,_=fixture()
            self.assertTrue(cv2.imwrite(str(source),np.repeat(gray[...,None],3,axis=2)))
            kept=[];dialog=None
            try:
                for order in itertools.permutations(('median','fill','morphology')):
                    candidate=NativeRecipeDialog(_base(),keep_in_project=kept.append);candidate.show()
                    moved=candidate.recipe.stages[0].stage_id;candidate.canvas.node_items[moved].setPos(123,456)
                    for control in order:
                        getattr(candidate,control+'_roi_button').click()
                        operations={'median':(factory.OPERATION,), 'fill':(fill.OPERATION,), 'morphology':morphology.OPERATIONS}[control]
                        for operation in operations:
                            stage=next(s.stage_id for s in candidate.recipe.stages if s.operation_id==operation)
                            index={factory.OPERATION:20,fill.OPERATION:21,'morphology.erode':22,'morphology.dilate':23}[operation]
                            candidate._parameter_changed(stage,factory.PARAMETER,f'[{index},{index},{index+50},{index+50}]')
                    identities.append(candidate.recipe.identity)
                    self.assertEqual((candidate.canvas.node_items[moved].pos().x(),candidate.canvas.node_items[moved].pos().y()),(123,456))
                    if dialog is not None:dialog.close();dialog.deleteLater()
                    dialog=candidate
                self.assertTrue(all(i==identities[0] for i in identities))
                stage=next(s.stage_id for s in dialog.recipe.stages if s.operation_id==factory.OPERATION)
                dialog.set_source_path(source);dialog.start_run()
                self.assertFalse(dialog.median_roi_button.isEnabled())
                deadline=time.monotonic()+60
                while dialog._worker is not None and time.monotonic()<deadline:app.processEvents();time.sleep(.01)
                app.processEvents();self.assertIsNone(dialog._worker)
                self.assertIsNotNone(dialog.result,dialog.status_label.text());node=dialog.result.node(stage)
                self.assertEqual(node.status,'executed',node.reason)
                dialog._select_stage(stage);self.assertFalse(dialog.preview.pixmap().isNull())
                selected=dialog._selected_recipe_with_layout();dialog.project_button.click();self.assertEqual(kept,[selected])
                reopened=NativeRecipeDialog(kept[0]);self.assertTrue(factory.is_median_roi_recipe(reopened.recipe))
                self.assertTrue(reopened.median_roi_button.isHidden());reopened.close();reopened.deleteLater()
                saved=root/'saved';archives.export_native_recipe_result(saved,selected,dialog.result)
                source.unlink();dialog.start_stage_replay(saved);self.assertFalse(dialog.median_roi_button.isEnabled())
                deadline=time.monotonic()+60
                while dialog._worker is not None and time.monotonic()<deadline:app.processEvents();time.sleep(.01)
                app.processEvents();self.assertIsNone(dialog._worker)
                self.assertEqual(dialog.replay_result.status,'matched',dialog.status_label.text())
                self.assertIs(dialog.result.node(stage),node)
            finally:
                if dialog is not None:dialog.close()
                app.processEvents()

    def test_calibrated_range_binary_median_project_versions_and_local_replay(self):
        from seedvision.benchmarking import native_watershed_diagnostic_recipe as diagnostic
        from seedvision.export.project_native_graphs import export_project_native_graphs
        from seedvision.persistence.project_analysis import ProjectAnalysisStore,ProjectImageSpec
        from seedvision.persistence.analysis_settings import analysis_settings_profile_from_graph
        from seedvision.pipeline import build_default_pipeline
        from tests.test_native_calibrated_range_recipe import selected_recipe as range_recipe
        from tests.test_native_calibrated_seed_recipe import fixture as calibrated_fixture
        old=diagnostic.build_native_watershed_diagnostic_recipe(range_recipe())
        for control in ('fill','morphology'):old=_add(old,control)
        recipe=_add(old,'median')
        self.assertEqual(old.recipe_id,recipe.recipe_id)
        self.assertNotEqual(old.recipe_version,recipe.recipe_version)
        stage=next(s.stage_id for s in recipe.stages if s.operation_id==factory.OPERATION)
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);workspace=root/'workspace';workspace.mkdir();(workspace/'config').mkdir()
            shutil.copyfile(Path(__file__).parents[1]/'config/traits.json',workspace/'config/traits.json')
            source=workspace/'seeds.png';image=calibrated_fixture();self.assertTrue(cv2.imwrite(str(source),image))
            actual=runner.run_native_recipe_path(recipe,source)
            self.assertEqual(actual.node(stage).status,'executed',actual.node(stage).reason)
            incoming=next(e for e in recipe.dependencies if e.target_stage==stage and e.target_port=='gray')
            input_mask=actual.node(incoming.source_stage).outputs[incoming.source_port]
            torch.testing.assert_close(actual.node(stage).outputs['image'],primitive.median_blur(input_mask,(2,3),roi=(20,20,70,70)))
            store=ProjectAnalysisStore(workspace)
            master=store.capture_and_save(analysis_settings=analysis_settings_profile_from_graph(build_default_pipeline()),
                images=(ProjectImageSpec(source,image.shape[:2]),),native_recipes=(old,recipe),
                destination=workspace/'seeds.seedfiddle-project.json')
            document=store.load(master).document;self.assertEqual(document.native_recipes,(old,recipe))
            ledger=export_project_native_graphs((),source,root/'runs',expected_source_sha256=document.images[0].source.sha256,
                project_root=workspace,native_recipes=document.native_recipes)
            import json
            rows=json.loads(ledger.read_bytes())['graphs'];self.assertEqual(len(rows),2)
            for row in rows:self.assertEqual(row['status'],'executed',row['reason'])
            source.unlink()
            for row in rows:
                saved=ledger.parent/Path(row['report']).parent
                replayed=replay.replay_native_stage(saved,stage)
                self.assertEqual(replayed.status,'matched',replayed.reason)
                self.assertEqual(replayed.metadata['frame_scope'],'calibrated_seed_roi')
                self.assertEqual(tuple(replayed.metadata['coordinate_frame']['shape']),(96,128))
                self.assertEqual(replayed.metadata['frame_lineage']['local_to_parent_xy'],(600,260))


if __name__=='__main__':unittest.main()
