"""Pinned public masks, actual archives, and bounded extended-threshold replay."""

from tests.functional_units import functional_units
from dataclasses import replace
from hashlib import sha256
import inspect
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from tests.catalogue_fixtures import CatalogueTestCase
from unittest.mock import patch

import numpy as np
import torch

from seedvision.benchmarking import native_recipe as runner
from seedvision.benchmarking import native_stage_replay as replay
from seedvision.benchmarking.checkpoints import CoordinateFrame
from seedvision.cuda import plantcv_extended_thresholds as extended
from seedvision.export import native_results as archives
from tests.test_native_extended_threshold_recipe import (
    _fixture, _reference_mask, _with_parameters, build_native_extended_threshold_recipes,
    SOURCE, PREPARATION,
)


@functional_units
class ExtendedThresholdReplayPlanTests(CatalogueTestCase):
    def check_test_existing_fourteen_admission_plans_remain_unchanged(self):
        expected = {
            'filter.median_blur':({'ksize':3},110592),
            'threshold.binary':({},3145728), 'threshold.mean':({'ksize':3,'offset':1,'object_type':'light'},3145728),
            'threshold.otsu':({},3145728), 'morphology.fill':({'size':1},196608),
            'morphology.erode':({'ksize':3,'iterations':1},73728),
            'morphology.dilate':({'ksize':3,'iterations':1},73728),
            'reference.internal.watershed.distance_map':({},3407872),
            **{op:({'distance':8} if op.endswith('local_maxima') else {},None)
               for op in replay.WATERSHED_TYPED_OPERATIONS},
            'native.watershed_debug_diagnostic':({},None),
        }
        self.assertEqual(len(expected),14)
        self.assertEqual(len(replay.SUPPORTED_REPLAY_OPERATIONS),48)
        for op,(parameters,work) in expected.items():
            plan = replay._plan((96,128),op,parameters,122880)
            self.assertEqual(plan['work'],work)
            self.assertEqual(plan['cuda_bytes'],272527360 if op.startswith('native.') else 135790592)
            typed = op in replay.WATERSHED_TYPED_OPERATIONS or op.startswith('native.')
            self.assertEqual(plan['selected_limit_bytes'],(128 if typed else 48)*1024*1024)

    def check_test_gaussian_float_sample_boundaries_and_shared_tile_formula(self):
        for kernel in (3,5,7,9,11,13,15):
            admission = extended._gaussian_admission(100_000,kernel)
            self.assertEqual(admission['window_sample_visits'],100_000*kernel*kernel)
            if kernel>=11:
                spp = 25*kernel*kernel+64*kernel+128
                self.assertEqual(admission['tile_pixels'],min(extended.TILE_PIXELS,
                    extended.MAX_GAUSSIAN_FLOAT_SCRATCH_BYTES//spp))
                maximum = extended.MAX_GAUSSIAN_FLOAT_WORK//(kernel*kernel)
                extended._gaussian_admission(maximum,kernel)
                with self.assertRaises(extended.ExtendedThresholdError):
                    extended._gaussian_admission(maximum+1,kernel)
                if kernel>=13:
                    params = dict(ksize=kernel,offset=1,object_type='light')
                    replay._plan((1,maximum),'threshold.gaussian',params,2*maximum)
                    with self.assertRaises(replay.NativeStageReplayUnavailable):
                        replay._plan((1,maximum+1),'threshold.gaussian',params,2*(maximum+1))
            else:
                self.assertEqual(admission['mode'],'exact_dyadic')
                self.assertEqual(admission['tile_pixels'],extended.TILE_PIXELS)

    def check_test_texture_pixel_pair_scratch_boundaries_and_triangle_source_ceiling(self):
        params = dict(ksize=9,offset=1,threshold=0,texture_method='energy',borders='mirror')
        plan = replay._plan((1,4_000_000),'threshold.texture',params,8_000_000)
        facts = plan['work_details']
        self.assertEqual(facts['oriented_pairs'],144)
        self.assertEqual(facts['pixel_pair_visits'],4_000_000*72)
        self.assertEqual(facts['scratch_per_pixel'],96*144+144*144+1024)
        self.assertLessEqual(facts['tile_scratch_bytes'],extended.MAX_TEXTURE_SCRATCH_BYTES)
        with self.assertRaises(replay.NativeStageReplayUnavailable):
            replay._plan((1,4_000_001),'threshold.texture',params,8_000_002)
        triangle = dict(object_type='light')
        source = replay._extended_threshold_admission(1<<24,'threshold.triangle',triangle)
        self.assertEqual(source['compact_cpu_histogram_bins'],256)
        with self.assertRaises(replay.NativeStageReplayUnavailable):
            replay._extended_threshold_admission((1<<24)+1,'threshold.triangle',triangle)
        maximum = 7_340_032
        replay._plan((1,maximum),'threshold.triangle',triangle,2*maximum)
        with self.assertRaises(replay.NativeStageReplayUnavailable):
            replay._plan((1,maximum+1),'threshold.triangle',triangle,2*(maximum+1))

    def check_test_dual_distinct_modes_selected_bytes_and_cuda_boundary(self):
        same = extended._dual_channel_admission(100_000,'R','G')
        distinct = extended._dual_channel_admission(100_000,'l','c')
        self.assertEqual(same['conversion_modes'],('RGB',))
        self.assertEqual(distinct['conversion_modes'],('LAB','CMYK'))
        self.assertEqual(distinct['scratch_per_pixel'],512)
        self.assertGreater(distinct['tile_scratch_bytes'],same['tile_scratch_bytes'])
        params = dict(x_channel='l',y_channel='c',points=[[0,0],[255,255]],above=True)
        replay._plan((1,7_340_032),'threshold.dual_channels',params,4*7_340_032)
        with self.assertRaises(replay.NativeStageReplayUnavailable):
            replay._plan((1,7_340_033),'threshold.dual_channels',params,4*7_340_033)
        with self.assertRaises(replay.NativeStageReplayUnavailable):
            replay._plan((1,1),'threshold.dual_channels',params,replay.MAX_SELECTED_BYTES+1)

    def check_test_malformed_controls_are_archive_errors_before_resource_checks(self):
        cases = (
            ('threshold.gaussian',dict(ksize=2,offset=1,object_type='light')),
            ('threshold.texture',dict(ksize=3,offset=3,threshold=0,texture_method='ASM',borders='nearest')),
            ('threshold.dual_channels',dict(x_channel='R',y_channel='G',points=[[0,0],[0,1]],above=1)),
            ('threshold.dual_channels',dict(x_channel='R',y_channel='G',points=[[0,0],['bad',1]],above=True)),
        )
        for op,params in cases:
            with self.subTest(operation=op),self.assertRaises(archives.NativeResultExportError):
                replay._plan((31,37),op,params,1)

    def test_workflow_lifecycle(self):
        """4 related ExtendedThresholdReplayPlanTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_existing_fourteen_admission_plans_remain_unchanged)
        self.scenario(self.check_test_gaussian_float_sample_boundaries_and_shared_tile_formula)
        self.scenario(self.check_test_texture_pixel_pair_scratch_boundaries_and_triangle_source_ceiling)
        self.scenario(self.check_test_malformed_controls_are_archive_errors_before_resource_checks)

    def test_validation_and_recovery(self):
        """1 related ExtendedThresholdReplayPlanTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_dual_distinct_modes_selected_bytes_and_cuda_boundary)


@unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
class ExtendedThresholdReplayTests(CatalogueTestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(dir='artifacts')
        cls.root = Path(cls.temporary.name)
        gray,rgb,triangle = _fixture()
        safe = np.full(gray.shape,37,dtype=np.uint8);safe[:,19:]=219
        cls.rgb = rgb
        cls.cases = (
            ('gaussian',gray,dict(ksize=5,offset=1.25,object_type='light')),
            ('triangle',triangle,dict(object_type='dark')),
            ('texture',gray,dict(ksize=3,offset=1,texture_method='dissimilarity',borders='nearest',threshold=30)),
            ('dual_channels',gray,dict(x_channel='R',y_channel='G',points=[[40,40],[180,150]],above=True)),
            ('gaussian',safe,dict(ksize=15,offset=12,object_type='light')),
            ('dual_channels',gray,dict(x_channel='l',y_channel='c',points=[[40,40],[180,150]],above=False)),
            ('texture',gray,dict(ksize=9,offset=1,texture_method='energy',borders='mirror',threshold=0)),
        )
        cls.runs = []
        recipes = build_native_extended_threshold_recipes()
        for index,(kind,image,parameters) in enumerate(cls.cases):
            recipe = _with_parameters(recipes['plantcv_threshold_'+kind],**parameters)
            frame = CoordinateFrame('test.extended-replay',image.shape,'top-left','x-right,y-down','pixel')
            run = runner.run_native_recipe(recipe,torch.tensor(image,device='cuda'),
                torch.tensor(rgb,device='cuda'),source_sha256=SOURCE,coordinate_frame=frame,preparation=PREPARATION)
            stage = recipe.stages[1].stage_id
            if run.node(stage).status!='executed':
                raise AssertionError(run.node(stage).reason)
            directory = cls.root/str(index)
            archives.export_native_recipe_result(directory,recipe,run)
            cls.runs.append((recipe,run,directory))

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def changed(self,index=0,*,input_image=None,mask=None,parameters=None,runtime=None,recorded_parameters=None):
        recipe,saved,_ = self.runs[index]
        if parameters is not None:
            recipe = _with_parameters(recipe,**parameters)
        source,target = recipe.stages
        nodes = []
        for node in saved.nodes:
            outputs = dict(node.outputs)
            if node.stage_id==source.stage_id and input_image is not None:
                outputs['gray_image' if self.cases[index][0]!='dual_channels' else 'rgb_image'] = input_image
            if node.stage_id==target.stage_id:
                if mask is not None:outputs['mask']=mask
                node = replace(node,parameters=recipe.parameters_for_runner(target.stage_id))
                if recorded_parameters is not None:
                    node = replace(node,parameters={**node.parameters,**recorded_parameters})
            nodes.append(replace(node,outputs=outputs))
        saved = replace(saved,nodes=tuple(nodes),recipe_identity=recipe.identity,
            runtime=saved.runtime if runtime is None else runtime)
        directory = self.root/('changed-'+str(len(tuple(self.root.iterdir()))))
        archives.export_native_recipe_result(directory,recipe,saved)
        return recipe,saved,directory

    def attempt(self,item,cancel=None):
        recipe,_,directory = item
        calls,loads = [],[]
        def profile(frame,event,arg):
            if event=='call' and frame.f_code is runner._dispatch.__code__:
                calls.append(frame.f_locals['operation'])
            if event=='call' and frame.f_code is archives.load_native_recipe_result.__code__:
                loads.append(tuple(frame.f_locals['array_names']))
        from tests.observation import observe_functions
        with observe_functions(profile, runner._dispatch, archives.load_native_recipe_result):
            result = replay.replay_native_stage(directory,recipe.stages[1].stage_id,cancel=cancel)
        return result,calls,loads

    def test_seven_actual_archives_match_pinned_public_masks_and_select_exact_saved_arrays(self):
        for case,item in zip(self.cases,self.runs):
            kind,image,parameters = case;recipe,saved,directory = item
            public = {**parameters,**({'xstep':1} if kind=='triangle' else {})}
            source = np.ascontiguousarray(self.rgb[...,::-1]) if kind=='dual_channels' else image
            expected = _reference_mask(source,kind,public)
            np.testing.assert_array_equal(saved.node(recipe.stages[1].stage_id).outputs['mask'].cpu().numpy(),expected)
            hashes = {p.relative_to(directory).as_posix():sha256(p.read_bytes()).hexdigest()
                for p in directory.rglob('*') if p.is_file()}
            inventory,_ = archives.read_native_recipe_inventory(directory)
            nodes = {n['stage_id']:dict(n['outputs']) for n in inventory['nodes']}
            edge = recipe.dependencies[0]
            names = {nodes[edge.source_stage][edge.source_port]['name'],nodes[edge.target_stage]['mask']['name']}
            result,calls,loads = self.attempt(item)
            self.assertEqual(result.status,'matched',result.reason)
            self.assertEqual(calls,['threshold.'+kind]);self.assertEqual(len(loads),1)
            self.assertEqual(set(loads[0]),names);self.assertEqual(len(loads[0]),2)
            self.assertEqual(result.metadata['plan']['selected_bytes'],image.size*(4 if kind=='dual_channels' else 2))
            self.assertEqual(result.metadata['frame_scope'],'source')
            self.assertEqual(result.metadata['comparison']['compared_elements'],image.size)
            self.assertFalse(result.scientifically_validated)
            self.assertEqual(result.outputs['mask'].device.type,'cuda')
            np.testing.assert_array_equal(result.outputs['mask'].cpu().numpy(),expected)
            self.assertEqual(hashes,{p.relative_to(directory).as_posix():sha256(p.read_bytes()).hexdigest()
                for p in directory.rglob('*') if p.is_file()})





    def test_cancel_and_input_expected_output_mutation_withhold_every_product(self):
        for group,port in (('inputs','gray'),('expected_products','mask'),('outputs','mask')):
            changed = []
            def mutate():
                frame = inspect.currentframe()
                while frame is not None:
                    if frame.f_code is replay.replay_native_stage.__code__ and group in frame.f_locals and not changed:
                        frame.f_locals[group][port].add_(1);changed.append(True)
                    frame = frame.f_back
                return False
            result,_,_ = self.attempt(self.runs[0],cancel=mutate)
            self.assertTrue(changed);self.assertEqual(result.status,'failed',result.reason)
            self.assertFalse(result.outputs)
        for index in range(4):
            def inside():
                frame = inspect.currentframe()
                while frame is not None:
                    if frame.f_code is runner._dispatch.__code__:return True
                    frame = frame.f_back
                return False
            result,_,_ = self.attempt(self.runs[index],cancel=inside)
            self.assertEqual(result.status,'cancelled',result.reason);self.assertFalse(result.outputs)
