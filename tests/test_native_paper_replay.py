"""Actual paper saved-input replay, primitive schema/admission and Qt workflow."""

from tests.functional_units import functional_units
from dataclasses import replace
from hashlib import sha256
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from tests.catalogue_fixtures import CatalogueTestCase
from unittest.mock import patch
import cv2
import numpy as np
import torch

os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from PySide6.QtWidgets import QApplication
from seedvision.benchmarking import native_smpts_recipe as paper, native_paper_replay as contracts
from seedvision.benchmarking import native_recipe as runner, native_stage_replay as replay
from seedvision.benchmarking.checkpoints import CoordinateFrame
from seedvision.export import native_results as archives
from seedvision.ui.native_recipe_dialog import NativeRecipeDialog
from tests.test_native_smpts_recipe import parameter, selected_recipe
from tests.test_smpts_paper import contact_fixture, canonical_components
from tests.replay_fixtures import archive_hashes, changed_archive, replay_attempt

STAGES=(paper.MEAN,paper.MEDIAN,paper.GLOBAL,paper.INTERSECTION,paper.COMPONENTS,
    paper.CONTACTS,paper.EROSION,paper.RELOCALIZE,paper.CROPS)
COUNTS=(2,2,2,3,2,1,5,2,5)


@functional_units
class PaperReplayContractTests(CatalogueTestCase):
    def check_test_schema_preserves_tuple_types_ids_bounds_and_strict_integer_fields(self):
        rows=({'label_id':1,'area_pixels':2,'bounds_xyxy':(0,0,2,1)},)
        value=contracts.canonical(rows)
        self.assertEqual(contracts.inventory(value,(2,3)),rows)
        for changed in (dict(rows[0],label_id=True),dict(rows[0],label_id=2),
                dict(rows[0],area_pixels=0),dict(rows[0],bounds_xyxy=(0,0,4,1)),
                dict(rows[0],extra=1)):
            with self.assertRaises(contracts.NativePaperReplayError):
                contracts.inventory(contracts.canonical((changed,)),(2,3))
        with self.assertRaises(contracts.NativePaperReplayError):contracts.inventory(list(value),(2,3))
        for invalid in ((True,),(2,),(1,1),(1,0)):
            with self.assertRaises(contracts.NativePaperReplayError):contracts.ids(invalid,1,'Selected')
        difference=contracts.compare(value,contracts.canonical((dict(rows[0],area_pixels=1),)))
        self.assertEqual(difference['mismatched_elements'],1)

    def check_test_metadata_decoder_has_byte_item_depth_and_primitive_bounds(self):
        binding={'kind':'tuple','items':[{'kind':'mapping','items':[['label_id',1],
            ['area_pixels',2],['bounds_xyxy',{'kind':'tuple','items':[0,0,2,1]}]]}]}
        self.assertEqual(contracts.decode(binding)[0]['area_pixels'],2)
        for bad in ({'kind':'object','items':[]}, {'kind':'mapping','items':[['a',1],['a',2]]},
                {'kind':'array','name':'x'},float('nan')):
            with self.assertRaises(contracts.NativePaperReplayError):contracts.decode(bad)
        with patch.object(contracts,'MAX_METADATA_BYTES',32):
            with self.assertRaises(contracts.NativePaperReplayError):contracts.canonical('\\n'*32)
        with patch.object(contracts,'MAX_METADATA_DEPTH',1):
            with self.assertRaises(contracts.NativePaperReplayError):contracts.decode(binding)
        with patch.object(contracts,'MAX_METADATA_ITEMS',1):
            with self.assertRaises(contracts.NativePaperReplayError):contracts.decode(binding)

    def check_test_crop_offsets_parents_shape_and_overlapping_boxes(self):
        # Two disjoint components can have overlapping boxes; T may exceed P.
        rows=contracts.canonical(tuple({'label_id':i,'area_pixels':2,'bounds_xyxy':(0,0,3,3)} for i in (1,2)))
        context={'source_components':rows,'final_components':rows}
        crops=contracts.canonical(tuple(dict(row,parent_id=1,offset_pixels=(i-1)*9,crop_shape=(3,3))
            for i,row in enumerate(rows,1)))
        contracts.validate(crops,'crops',(3,3),context)
        self.assertEqual(contracts.crop_total(rows),18)
        self.assertEqual(contracts.raster_spec('packed_rgb',(3,3),context),((54,),'|u1'))
        for key,bad in (('offset_pixels',1),('parent_id',3),('crop_shape',(3,2)),
                ('area_pixels',True),('bounds_xyxy',(False,0,3,3)),('bounds_xyxy',(0.,0,3,3))):
            changed=contracts.canonical((dict(crops[0],**{key:bad}),crops[1]))
            with self.assertRaises(contracts.NativePaperReplayError):
                contracts.validate(changed,'crops',(3,3),context)
        value=contracts.assumptions(2);contracts.validate(value,'metadata',(3,3),context)
        for key,bad in (('original_execution',True),('crop_count',True),('median_border','reflect')):
            with self.assertRaises(contracts.NativePaperReplayError):
                contracts.validate(dict(value,**{key:bad}),'metadata',(3,3),context)

    def check_test_allocation_free_operation_limits_alias_facts_and_exact_cap(self):
        recipe=selected_recipe()
        for stage in STAGES:
            operation=recipe.stage(stage).operation_id
            opts={'selected_allocations':(12,48),'input_allocations':(12,)}
            result=contracts.plan(operation,(3,4),recipe.parameters_for_runner(stage),**opts)
            self.assertGreaterEqual(result['cuda_bytes'],60+contracts.AUDIT_CUDA_BYTES)
            contracts.plan(operation,(3,4),recipe.parameters_for_runner(stage),
                cuda_limit=result['cuda_bytes'],selected_limit=60,**opts)
            for name,limit in (('cuda_limit',result['cuda_bytes']-1),('selected_limit',59)):
                with self.assertRaises(contracts.NativePaperReplayUnavailable):
                    contracts.plan(operation,(3,4),recipe.parameters_for_runner(stage),**{name:limit},**opts)
        with self.assertRaises(contracts.NativePaperReplayUnavailable):
            contracts.plan('smpts_paper.median',(1100,1000),{'ksize':31},**opts)
        old_worst_case_shape = (1,7_272_728)
        admitted = contracts.plan('smpts_paper.components',old_worst_case_shape,{},**opts)
        self.assertEqual(admitted['component_work'], 48*old_worst_case_shape[1])
        with patch.object(paper.paper, 'MAX_COMPONENT_WORK', 576):
            admitted = contracts.plan('smpts_paper.components',(3,4),{},**opts)
            self.assertEqual(admitted['component_work'], 576)
        with patch.object(paper.paper, 'MAX_COMPONENT_WORK', 575):
            with self.assertRaises(contracts.NativePaperReplayUnavailable):
                contracts.plan('smpts_paper.components',(3,4),{},**opts)
        with self.assertRaises(contracts.NativePaperReplayError):
            contracts.plan('smpts_paper.erosion',(3,4),{'ksize':2},**opts)
        mean=contracts.plan('smpts_paper.mean',(2080,3120),{'ksize':25,'offset':2.,'object_type':'light'},
            selected_allocations=(2080*3120,2080*3120),input_allocations=(2080*3120,))
        self.assertLessEqual(mean['cuda_bytes'],contracts.MAX_CUDA_BYTES)
        result=contracts.plan('smpts_paper.crops',(3,3),{},crop_pixels=18,**opts)
        self.assertEqual(result['crop_pixels'],18)

    def test_workflow_lifecycle(self):
        """3 related PaperReplayContractTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_schema_preserves_tuple_types_ids_bounds_and_strict_integer_fields)
        self.scenario(self.check_test_metadata_decoder_has_byte_item_depth_and_primitive_bounds)
        self.scenario(self.check_test_crop_offsets_parents_shape_and_overlapping_boxes)

    def test_validation_and_recovery(self):
        """1 related PaperReplayContractTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_allocation_free_operation_limits_alias_facts_and_exact_cap)


@unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
class PaperStageReplayTests(CatalogueTestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary=tempfile.TemporaryDirectory(dir='artifacts');cls.root=Path(cls.temporary.name)
        mask,rgb=contact_fixture();gray=np.where(mask,220,10).astype(np.uint8)
        cls.cases={};cls.images={}
        for name,gray,rgb in (('full',gray,rgb),('singleton',np.full((1,1),220,np.uint8),np.full((1,1,3),220,np.uint8)),
                ('empty',np.full((2,3),10,np.uint8),np.full((2,3,3),10,np.uint8))):
            image=cls.root/(name+'.png');assert cv2.imwrite(str(image),rgb[:,:,::-1])
            recipe=selected_recipe();frame=CoordinateFrame('source.decoded',gray.shape,'top-left','x-right,y-down','pixel')
            saved=runner.run_native_recipe(recipe,torch.tensor(gray,device='cuda'),torch.tensor(rgb,device='cuda'),
                source_sha256=sha256(image.read_bytes()).hexdigest(),coordinate_frame=frame,
                preparation={'gray':'explicit test gray','rgb':'original test RGB'})
            assert saved.status=='executed',[(n.stage_id,n.status,n.reason) for n in saved.nodes]
            directory=cls.root/name;archives.export_native_recipe_result(directory,recipe,saved)
            cls.cases[name]=(recipe,saved,directory);cls.images[name]=image
        cls.full_mask,cls.full_rgb=contact_fixture()
        cls.full_gray=np.where(cls.full_mask,220,10).astype(np.uint8)

    @classmethod
    def tearDownClass(cls):cls.temporary.cleanup()

    def attempt(self,stage,case='full',directory=None,cancel=None,change=None):
        return replay_attempt(directory or self.cases[case][2],stage,
            cancel=cancel,inventory_change=change)

    def changed(self,**nodes):
        recipe,saved,_=self.cases['full']
        return changed_archive(self.root,recipe,saved,nodes)

    def test_all_nine_stage_instances_replay_only_selected_operation_and_preserve_archive(self):
        from tests.replay_fixtures import hidden_origins
        with hidden_origins(*self.images.values()):
            for case,(_,saved,directory) in self.cases.items():
                before=archive_hashes(directory)
                for stage,count in zip(STAGES,COUNTS):
                    with self.subTest(case=case,stage=stage):
                        result,calls,selected=self.attempt(stage,case)
                        self.assertEqual(result.status,'matched',result.reason)
                        self.assertEqual(calls,[saved.node(stage).operation_id])
                        self.assertEqual(len(selected),1);self.assertEqual(len(selected[0]),count)
                        self.assertEqual(set(result.outputs),set(saved.node(stage).outputs))
                        self.assertFalse(result.scientifically_validated)
                        self.assertEqual(result.metadata['frame_scope'],'source')
                        if stage==paper.CROPS:self.assertFalse(result.outputs['metadata']['original_execution'])
                after=archive_hashes(directory)
                self.assertEqual(before,after)

        self.check_working_foreground_intersection_and_downstream_replay()

    def check_working_foreground_intersection_and_downstream_replay(self):
        # Working intersection exclusions remain full-frame and bind downstream ancestry.
        from seedvision.pipeline.workflows import create_workflow, with_foreground_region
        recipe = selected_recipe()
        workflow = with_foreground_region(create_workflow(recipe), paper.INTERSECTION,
            [0, 0, self.full_gray.shape[1] // 2, self.full_gray.shape[0]])
        frame = CoordinateFrame('source.decoded', self.full_gray.shape, 'top-left', 'x-right,y-down', 'pixel')
        run = runner.run_native_recipe(workflow.recipe, torch.as_tensor(self.full_gray, device='cuda'),
            torch.as_tensor(self.full_rgb, device='cuda'), source_sha256=sha256(self.images['full'].read_bytes()).hexdigest(),
            coordinate_frame=frame, preparation={'gray':'explicit test gray','rgb':'original test RGB'},
            working_workflow=workflow, target_node_id=paper.COMPONENTS)
        directory = self.root / 'foreground-working'
        archives.export_native_recipe_result(directory, workflow.recipe, run)
        before = archive_hashes(directory)
        for stage in (paper.INTERSECTION, paper.COMPONENTS):
            result = replay.replay_native_stage(directory, stage)
            self.assertEqual(result.status, 'matched', result.reason)
            self.assertTrue(torch.equal(result.outputs['mask' if stage == paper.INTERSECTION else 'labels'],
                run.node(stage).outputs['mask' if stage == paper.INTERSECTION else 'labels']))
            self.assertEqual(result.metadata['comparison']['mismatched_elements'], 0)
        self.assertEqual(before, archive_hashes(directory))











if __name__=='__main__':unittest.main()
