"""Public small-object ROI equality and complete native working paths."""

from tests.functional_units import functional_units
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
from tests.catalogue_fixtures import CatalogueTestCase
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


@functional_units
class FillROIContractTests(CatalogueTestCase):
    def check_test_all_twenty_eight_variants_and_reverse_order_preserve_base_contracts(self):
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

    def check_test_bounds_work_and_full_canvas_memory_admission(self):
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

    def check_test_historical_mask_source_closure_tracks_new_fill_helpers(self):
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

    def test_workflow_lifecycle(self):
        """3 related FillROIContractTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_all_twenty_eight_variants_and_reverse_order_preserve_base_contracts)
        self.scenario(self.check_test_bounds_work_and_full_canvas_memory_admission)
        self.scenario(self.check_test_historical_mask_source_closure_tracks_new_fill_helpers)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
@functional_units
class FillROIIntegrationTests(CatalogueTestCase):


    def check_test_crop_local_connectivity_area_equality_holes_exterior_and_shortcuts(self):
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

    def check_test_invalid_bounds_strict_cancel_and_input_preservation(self):
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




    def check_test_async_desktop_both_action_orders_preserve_edits_layout_keep_and_saved_replay(self):
        """One real lifecycle; the contract matrix owns forward/reverse composition."""
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


    def test_workflow_lifecycle(self):
        """2 related FillROIIntegrationTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_crop_local_connectivity_area_equality_holes_exterior_and_shortcuts)
        self.scenario(self.check_test_async_desktop_both_action_orders_preserve_edits_layout_keep_and_saved_replay)

    def test_validation_and_recovery(self):
        """1 related FillROIIntegrationTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_invalid_bounds_strict_cancel_and_input_preservation)


if __name__=='__main__':unittest.main()
