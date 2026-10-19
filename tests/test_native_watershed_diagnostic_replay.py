
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

import cv2
import torch

from seedvision.benchmarking import native_recipe as runner
from seedvision.benchmarking import native_stage_replay as replay
from seedvision.benchmarking import native_watershed_diagnostic_recipe as factory
from seedvision.benchmarking import native_watershed_diagnostic_replay as compact
from seedvision.export import native_results as archives
from tests import test_native_recipe as seeds
from tests import test_native_calibrated_seed_recipe as calibrated
from tests.test_native_watershed_diagnostic_recipe import controls, selected


def binding(value):
    if isinstance(value,dict):
        return {'kind':'mapping','items':[[key,binding(item)] for key,item in value.items()]}
    if isinstance(value,(tuple,list)):
        return {'kind':'tuple','items':[binding(item) for item in value]}
    return value


@functional_units
class DiagnosticReplayPlanTests(CatalogueTestCase):
    def check_test_new_plan_admits_full_fixture_and_preserves_old_contract_envelopes(self):
        operation = factory.OPERATION
        for pixels in (1,1_000_000,1_000_001,2080*3120):
            plan = replay._plan((1,pixels),operation,{},10*pixels)
            self.assertEqual(plan['cuda_bytes'],max(128*pixels+128*1024*1024,
                13*pixels+256*1024*1024+64*min(pixels,1_000_000)+3*1024*1024))
            self.assertIsNone(plan['work_limit'])
            self.assertEqual(plan['selected_limit_bytes'],128*1024*1024)
            self.assertEqual(plan['work_details']['max_metadata_bytes'],256*1024)
        maximum=(1024*1024*1024-128*1024*1024)//128
        self.assertEqual(replay._plan((1,maximum),operation,{},1)['cuda_bytes'],1024*1024*1024)
        for shape,amount in (((1,maximum+1),1),((96,128),128*1024*1024+1)):
            with self.assertRaises(runner.NativeRecipeError):
                replay._plan(shape,operation,{},amount)
        for operation in replay.WATERSHED_TYPED_OPERATIONS:
            p = {'distance':8} if operation.endswith('local_maxima') else {}
            self.assertEqual(replay._plan((96,128),operation,p,1)['cuda_bytes'],128*96*128+128*1024*1024)
        self.assertEqual(replay._plan((96,128),'threshold.binary',{},1)['selected_limit_bytes'],48*1024*1024)

    def check_test_decoder_counts_keys_depth_bytes_and_rejects_nonprimitive_tags(self):
        bad = [dict(kind='array',name='array_0000'),dict(kind='record',type='anything',fields={}),
               {'kind':'mapping','items':[['a',1],['a',2]]},1.0,{'kind':'tuple','items':[2**65]},
               'x'*100_000,'\u20ac'*342,'\ud800',{'kind':'x'*100_000,'items':[]}]
        deep = 0
        for _ in range(compact.MAX_METADATA_DEPTH+2):
            deep = {'kind':'tuple','items':[deep]}
        bad.append(deep)
        for value in bad:
            with self.assertRaises(ValueError):
                compact.decode(value,ValueError)
        with patch.object(compact,'MAX_METADATA_ITEMS',2):
            with self.assertRaises(ValueError):
                compact.decode(binding({'a':1}),ValueError)
        with patch.object(compact,'MAX_METADATA_BYTES',33):
            with self.assertRaises(ValueError):
                compact.decode(binding((255,255)),ValueError)

    def check_test_comparison_counts_each_extra_palette_channel_and_finishes_after_path_cap(self):
        actual={'palette':((1,2,3),(4,5,6)),'ids':(1,2)}
        expected={'palette':((1,2,3),),'ids':(1,)}
        result=compact.compare(actual,expected,lambda:False)
        self.assertEqual(result['compared_elements'],8)
        self.assertEqual(result['mismatched_elements'],4)
        result=compact.compare({'palette':tuple((1,2,3) for _ in range(100))},
            {'palette':()},lambda:False)
        self.assertEqual(result['compared_elements'],300)
        self.assertEqual(result['mismatched_elements'],300)
        self.assertEqual(len(result['mismatch_paths']),32)
        self.assertEqual(compact.compare((),None,lambda:False)['mismatched_elements'],1)

    def test_workflow_lifecycle(self):
        """2 related DiagnosticReplayPlanTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_new_plan_admits_full_fixture_and_preserves_old_contract_envelopes)
        self.scenario(self.check_test_comparison_counts_each_extra_palette_channel_and_finishes_after_path_cap)

    def test_validation_and_recovery(self):
        """1 related DiagnosticReplayPlanTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_decoder_counts_keys_depth_bytes_and_rejects_nonprimitive_tags)


@unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
class DiagnosticReplayTests(CatalogueTestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(dir='artifacts')
        cls.root = Path(cls.temporary.name)
        cls.runs = []
        for kind in ('binary','mean','otsu'):
            for local in (False,True):
                recipe = selected(kind,local=local)
                if local:
                    path=cls.root/'card.png'
                    assert cv2.imwrite(str(path),calibrated.fixture())
                    saved=runner.run_native_recipe_path(recipe,path)
                else:
                    gray,rgb=seeds.fixture()
                    saved=runner.run_native_recipe(recipe,torch.tensor(gray,device='cuda'),
                        torch.tensor(rgb,device='cuda'),source_sha256=seeds.SOURCE,
                        coordinate_frame=seeds.FRAME,preparation=seeds.PREPARATION)
                directory=cls.root/(kind+('-local' if local else '-source'))
                archives.export_native_recipe_result(directory,recipe,saved)
                cls.runs.append((recipe,saved,directory))

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def changed(self, index=0, stage=None, **outputs):
        recipe,saved,_=self.runs[index]
        stage=stage or factory.STAGE
        node=replace(saved.node(stage),outputs={**dict(saved.node(stage).outputs),**outputs})
        result=replace(saved,nodes=tuple(node if item.stage_id==stage else item for item in saved.nodes))
        directory=self.root/('changed-'+str(len(tuple(self.root.iterdir()))))
        archives.export_native_recipe_result(directory,recipe,result)
        return directory

    def attempt(self,directory,cancel=None):
        calls=[]
        def profile(frame,event,arg):
            if event=='call' and frame.f_code is runner._dispatch.__code__:
                calls.append((frame.f_locals['operation'],set(frame.f_locals['inputs'])))
        from tests.observation import observe_functions
        with observe_functions(profile, runner._dispatch):
            result=replay.replay_native_stage(directory,factory.STAGE,cancel=cancel)
        return result,calls





    def test_palette_maximum_and_nested_compact_immutability(self):
        facts=dict(self.runs[0][1].node(factory.STAGE).outputs['metadata'])
        palette,metadata=runner._watershed_diagnostic.watershed_palette(4097,random_seed=0xffffffff)
        facts.update(metadata);facts['label_ids']=tuple(range(1,4097))
        parameters={'random_seed':0xffffffff,'color_scale':'gist_rainbow','line_thickness':5}
        frozen=compact.validate(compact.decode(binding(facts),ValueError),parameters,96*128,ValueError)
        self.assertEqual(len(frozen['palette_source_triples']),4097)
        with self.assertRaises(TypeError):frozen['palette_count']=2
        with self.assertRaises(TypeError):frozen['palette_source_triples'][0][0]=1
