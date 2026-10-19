"""Actual dual-frame calibrated crop replay, admission, callback and Qt checks."""

from tests.catalogue_fixtures import CatalogueTestCase

from tests.functional_units import functional_units
from dataclasses import replace
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
from seedvision.segmentation.baseline import PipelineAnalysisCache
from tests.test_native_calibrated_seed_recipe import fixture,selected_recipe
from tests.test_native_card_recipe import _parameter
from tests.replay_fixtures import (archive_hashes, changed_archive,
    export_private_archive, replay_attempt, assert_stage_inputs_equal)


@functional_units
class CropReplayContractTests(CatalogueTestCase):
    def context(self,bounds=(2,3,8,7)):
        parent=CoordinateFrame('explicit.parent',(10,12),'top-left','x-right,y-down','pixel','a'*64)
        return contract.frames(parent,{'seed_roi_xyxy':bounds},'b'*64,(0,0,12,10))

    def check_test_strict_roi_distinct_frames_and_shifted_equal_shapes(self):
        context=self.context();shifted=self.context((3,3,9,7))
        self.assertEqual(context.local.shape,shifted.local.shape)
        self.assertNotEqual(context.local.transform_sha256,shifted.local.transform_sha256)
        self.assertEqual(context.parent.transform_sha256,'a'*64)
        self.assertEqual(context.local.frame_id,'explicit.parent.seed_roi')
        for bounds in ((True,3,8,7),(2.,3,8,7),(2,3,2,7),(-1,3,8,7),(2,3,13,7)):
            with self.subTest(bounds=bounds),self.assertRaises(contract.NativeCropReplayError):self.context(bounds)

    def check_test_metadata_fixed_binding_types_and_numeric_overlap_comparison(self):
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

    def check_test_allocation_free_selected_and_cuda_exact_boundaries(self):
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

    def test_workflow_lifecycle(self):
        """3 related CropReplayContractTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_strict_roi_distinct_frames_and_shifted_equal_shapes)
        self.scenario(self.check_test_metadata_fixed_binding_types_and_numeric_overlap_comparison)
        self.scenario(self.check_test_allocation_free_selected_and_cuda_exact_boundaries)


@unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
class NativeCropStageReplayTests(CatalogueTestCase):
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
        return archive_hashes(directory)

    def export(self,recipe,run,prefix='changed'):
        return export_private_archive(self.root,recipe,run,prefix=prefix)

    def changed(self,owner,**changes):
        node=replace(self.saved.node(owner),**changes)
        return changed_archive(self.root,self.recipe,self.saved,{owner:node})

    def invalid(self,directory):
        before=self.hashes(directory)
        with patch.object(archives,'load_native_recipe_result') as load:
            result=replay.replay_native_stage(directory,factory.CROP)
        self.assertEqual(result.status,'invalid_archive',result.reason)
        self.assertFalse(result.outputs);load.assert_not_called()
        self.assertEqual(self.hashes(directory),before)

    def attempt(self,directory=None,cancel=None):
        directory=directory or self.directory;before=self.hashes(directory)
        result,calls,_=replay_attempt(directory,factory.CROP,cancel=cancel)
        self.assertEqual(self.hashes(directory),before)
        return result,calls



    def test_all_four_calibrated_and_four_diagnostic_graphs_support_crop_replay(self):
        from tests.test_native_calibrated_range_recipe import selected_recipe as range_recipe
        prefix=tuple(stage.stage_id for stage in card.build_native_card_recipe().stages[1:])+(factory.CROP,)
        for kind in ('binary','mean','otsu','range'):
            base=range_recipe() if kind=='range' else selected_recipe(kind)
            for diagnostic in (False,True):
                with self.subTest(kind=kind,diagnostic=diagnostic):
                    recipe=runner._diagnostic_recipe.build_native_watershed_diagnostic_recipe(base) if diagnostic else base
                    runner._validate_recipe(recipe)
                    assert_stage_inputs_equal(self,recipe,self.recipe,prefix)
                    self.assertEqual(recipe.stage(base.stages[0].stage_id),base.stages[0])
                    self.assertTrue(any(edge.source_stage==factory.CROP and edge.source_port=='rgb_image'
                        and edge.target_stage not in prefix for edge in recipe.dependencies))
                    if diagnostic:
                        self.assertEqual(recipe.stages[:-1],base.stages)
                        self.assertEqual(recipe.dependencies[:-2],base.dependencies)
        # The base archive is the real selected binary run built once by this
        # class. One additional diagnostic graph proves the common prefix reaches
        # actual analysis/drawing consumers; all eight full numerical workflows
        # remain exercised by the frozen acceptance matrix.
        self.assertEqual(self.saved.node('pcv_seed_binary_analyze_color').status,'executed')
        self.assertEqual(self.saved.node('pcv_seed_binary_analyze_size').status,'executed')
        recipe=runner._diagnostic_recipe.build_native_watershed_diagnostic_recipe(self.recipe)
        run=runner.run_native_recipe_path(recipe,self.path)
        self.assertEqual(run.node(runner._diagnostic_recipe.STAGE).status,'executed')
        directory=self.export(recipe,run,'variant');result,calls=self.attempt(directory)
        self.assertEqual(result.status,'matched',result.reason);self.assertEqual(calls,[contract.OPERATION])
        self.assertEqual(result.metadata['coordinate_frame']['frame_id'],'source.decoded.seed_roi')
        for port in ('rgb_image','gray_image'):
            self.assertTrue(torch.equal(result.outputs[port],self.saved.node(factory.CROP).outputs[port]))







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



if __name__=='__main__':unittest.main()
