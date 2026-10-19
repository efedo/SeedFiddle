"""Public range outputs through real native, archive, Qt and batch paths."""

from tests.functional_units import functional_units
from dataclasses import replace
from hashlib import sha256
import inspect
import json
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

from seedvision.benchmarking import native_range_recipe as factory
from seedvision.benchmarking import native_recipe as runner
from seedvision.benchmarking import native_stage_replay as replay
from seedvision.benchmarking.checkpoints import CoordinateFrame
from seedvision.benchmarking.native_extended_threshold_recipes import build_native_extended_threshold_recipes
from seedvision.benchmarking.recipe_catalogue import build_recipe_catalogue
from seedvision.benchmarking.recipes import RecipeSpec
from seedvision.cuda import plantcv_range as primitive
from seedvision.export import native_results as archives
from seedvision.export.project_native_graphs import export_project_native_graphs
from seedvision.persistence import native_recipes as persistence
from seedvision.persistence.reference_regions import file_sha256
from seedvision.segmentation.baseline import PipelineAnalysisCache
from tests.test_native_extended_threshold_recipe import _fixture, _with_parameters
from tests.test_plantcv_range import _reference_custom_range, _reference_channel_bytes


def execute(recipe, gray, rgb, cache=None, cancel=None):
    return runner.run_native_recipe(recipe, gray, rgb, source_sha256='a'*64,
        coordinate_frame=CoordinateFrame('test.native-range', tuple(gray.shape),
            'top-left', 'x-right,y-down', 'pixel'),
        preparation={'gray':'explicit grayscale', 'rgb':'explicit original RGB'},
        cache=cache, cancel=cancel)


class RangeContractTests(CatalogueTestCase):
    def test_separate_canonical_recipe_roundtrip_preserves_reference_and_extended_graphs(self):
        history = build_recipe_catalogue().to_json()
        extended = {key:value.to_json() for key,value in build_native_extended_threshold_recipes().items()}
        recipe = factory.build_native_range_recipe()
        runner._validate_recipe(recipe)
        self.assertEqual(RecipeSpec.from_json(recipe.to_json()), recipe)
        self.assertEqual(persistence.native_recipes_from_payload(
            persistence.native_recipes_to_payload((recipe,))), (recipe,))
        stage = recipe.stage(factory.STAGE_ID)
        self.assertEqual([p.name for p in stage.ports], ['rgb','mask','masked_rgb'])
        self.assertTrue(all(p.affects_calculation and p.editable for p in stage.parameters))
        self.assertEqual(stage.source_mappings[0].resolution.value, 'pinned')
        self.assertEqual(build_recipe_catalogue().to_json(), history)
        self.assertEqual({key:value.to_json() for key,value in build_native_extended_threshold_recipes().items()}, extended)
        bad = replace(recipe, stages=(recipe.stages[0],replace(stage, operation_id='threshold.dual_channels')))
        with self.assertRaises(runner.NativeRecipeError):runner._validate_recipe(bad)

    def test_shared_admission_and_exact_replay_resource_boundaries(self):
        controls = dict(channel='CMYK',lower=[0]*4,upper=[255]*4)
        plan = replay._plan((887,887),factory.OPERATION,controls,7*887*887)
        self.assertEqual(plan['cuda_bytes'],factory.retained_bound(887*887))
        self.assertEqual(plan['work_details']['primitive_output_bytes'],4*887*887)
        self.assertGreater(plan['work_details']['tile_scratch_bytes'],0)
        replay._plan((1,7_190_235),factory.OPERATION,controls,7*7_190_235)
        with self.assertRaises(replay.NativeStageReplayUnavailable):
            replay._plan((1,7_190_236),factory.OPERATION,controls,7*7_190_236)
        replay._plan((1,7_340_032),factory.OPERATION,controls,1)
        with self.assertRaises(replay.NativeStageReplayUnavailable):
            replay._plan((1,7_340_033),factory.OPERATION,controls,1)
        with self.assertRaises(primitive.RangeResourceError):primitive._range_admission(64_000_001)
        for changed in (dict(channel='other'),dict(lower=[0]),dict(upper=[256]*4),dict(lower=[False]*4)):
            with self.subTest(changed=changed),self.assertRaises(archives.NativeResultExportError):
                replay._plan((31,37),factory.OPERATION,{**controls,**changed},1)


@unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
@functional_units
class RangeExecutionTests(CatalogueTestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(dir='artifacts')
        cls.root = Path(cls.temporary.name)
        gray,rgb,_ = _fixture()
        # Endpoints include unequal R/B slots and the full neutral byte axis.
        flat = rgb.reshape(-1,3)
        flat[:256] = np.repeat(np.arange(256,dtype=np.uint8)[:,None],3,axis=1)
        flat[256:262] = [[250,30,10],[10,30,250],[0,0,0],[255,255,255],[0,255,0],[20,110,210]]
        cls.gray,cls.rgb = torch.tensor(gray,device='cuda'),torch.tensor(rgb,device='cuda')
        cls.recipe = factory.build_native_range_recipe()
        cls.cases = []
        channels = _reference_channel_bytes(np.ascontiguousarray(rgb[...,::-1]))
        for mode in ('RGB','HSV','LAB','CMYK','GRAY'):
            count = 4 if mode=='CMYK' else 1 if mode=='GRAY' else 3
            values = channels[mode.lower()][256].astype(int).tolist()
            cls.cases.extend((dict(name=mode+'-range',channel=mode,
                lower=[0,30,0,20] if mode=='CMYK' else [30]*count,
                upper=[180,220,180,200] if mode=='CMYK' else [180]*count),
                dict(name=mode+'-equal',channel=mode,lower=values,upper=values),
                dict(name=mode+'-reversed',channel=mode,lower=[180]*count,upper=[30]*count)))
        cls.reference = _reference_custom_range(np.ascontiguousarray(rgb[...,::-1]),cls.cases)

    @classmethod
    def tearDownClass(cls):cls.temporary.cleanup()

    def save(self, recipe=None, run=None, *, change=None, runtime=None):
        recipe = recipe or self.recipe
        run = run or execute(recipe,self.gray,self.rgb)
        if change:
            run = replace(run,nodes=tuple(replace(n,outputs={**n.outputs,**change})
                if n.stage_id==factory.STAGE_ID else n for n in run.nodes))
        if runtime is not None:run=replace(run,runtime=runtime)
        directory = self.root/str(len(tuple(self.root.iterdir())))
        archives.export_native_recipe_result(directory,recipe,run)
        return directory,run

    def check_test_fifteen_mode_endpoint_and_reversed_bound_cases_match_both_public_outputs(self):
        for case in self.cases:
            with self.subTest(case=case['name']):
                recipe = _with_parameters(self.recipe,**{key:value for key,value in case.items() if key!='name'})
                run = execute(recipe,self.gray,self.rgb)
                self.assertEqual(run.status,'executed',[(n.status,n.reason) for n in run.nodes])
                outputs = run.node(factory.STAGE_ID).outputs
                np.testing.assert_array_equal(outputs['mask'].cpu().numpy(),self.reference[case['name']+'_mask'])
                np.testing.assert_array_equal(outputs['masked_rgb'].cpu().numpy(),
                    self.reference[case['name']+'_masked'][...,::-1])
                self.assertTrue(all(t.is_cuda and t.is_contiguous() for t in outputs.values()))
                if case['name'].endswith('equal'):
                    self.assertEqual(outputs['mask'].view(-1)[256].item(),255)
                if case['name'].endswith('range'):
                    count=torch.count_nonzero(outputs['mask']).item()
                    self.assertGreater(count,0);self.assertLess(count,self.gray.numel())
                if case['name'].endswith('reversed'):
                    self.assertEqual(torch.count_nonzero(outputs['mask']).item(),0)

    def check_test_each_control_changes_calculation_and_only_threshold_is_recomputed(self):
        cache=PipelineAnalysisCache()
        recipe=_with_parameters(self.recipe,lower=[0,0,0],upper=[100,180,255])
        previous=execute(recipe,self.gray,self.rgb,cache)
        self.assertEqual(execute(recipe,self.gray,self.rgb,cache).status,'executed')
        self.assertEqual(cache.last_computed_nodes,())
        for update in (dict(lower=[70,0,0]),dict(upper=[255,180,255]),dict(channel='HSV')):
            changed=_with_parameters(recipe,**update)
            current=execute(changed,self.gray,self.rgb,cache)
            self.assertEqual(current.status,'executed')
            self.assertFalse(torch.equal(current.node(factory.STAGE_ID).outputs['mask'],previous.node(factory.STAGE_ID).outputs['mask']))
            self.assertEqual(cache.last_computed_nodes,(factory.STAGE_ID,))
            self.assertEqual(cache.last_reused_nodes,(recipe.stages[0].stage_id,))
            previous,recipe=current,changed

    def test_additional_numerical_regressions(self):
        self.scenario(self.check_test_fifteen_mode_endpoint_and_reversed_bound_cases_match_both_public_outputs)
        self.scenario(self.check_test_each_control_changes_calculation_and_only_threshold_is_recomputed)










class RangeQtTests(CatalogueTestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
        from PySide6.QtWidgets import QApplication
        cls.app=QApplication.instance() or QApplication([])



if __name__=='__main__':unittest.main()
