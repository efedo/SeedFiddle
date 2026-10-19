"""Version compatibility, public median ROI values and native working paths."""

from tests.functional_units import functional_units
from dataclasses import replace
import hashlib
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import unittest
from tests.catalogue_fixtures import CatalogueTestCase
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
from tests.test_morphology_roi_recipe import _change, _composition_orders
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


@functional_units
class MedianROIContractTests(CatalogueTestCase):
    def check_test_versions_coexist_and_forged_contracts_reject(self):
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

    def check_test_all_six_action_orders_preserve_nondefault_independent_controls(self):
        from seedvision.benchmarking import native_watershed_diagnostic_recipe as diagnostic
        expected = None
        base = _base()
        base = replace(base, layout=tuple(replace(p,x=p.x+123,y=p.y+456) for p in base.layout))
        for order,recipe in _composition_orders(base,('median','fill','morphology'),_add):
            with self.subTest(order=order):
                runner._validate_recipe(recipe)
                self.assertTrue(factory.is_median_roi_recipe(recipe))
                self.assertEqual(RecipeSpec.from_json(recipe.to_json()).identity,recipe.identity)
                self.assertEqual(recipe.layout,base.layout)
                self.assertEqual(recipe.recipe_version,factory.RECIPE_VERSION)
                if expected is None: expected = recipe.identity
                self.assertEqual(recipe.identity,expected)
                self.assertEqual(diagnostic.base_recipe_id(recipe.recipe_id),base.recipe_id)

    def check_test_crop_work_and_geometry_keep_full_canvas_admission(self):
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

    def test_validation_and_recovery(self):
        """1 related MedianROIContractTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_versions_coexist_and_forged_contracts_reject)

    def test_workflow_lifecycle(self):
        """2 related MedianROIContractTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_all_six_action_orders_preserve_nondefault_independent_controls)
        self.scenario(self.check_test_crop_work_and_geometry_keep_full_canvas_admission)


@unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
class MedianROIIntegrationTests(CatalogueTestCase):

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






if __name__=='__main__':unittest.main()
