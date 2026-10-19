"""Typed observations, public colour parity, isolated replay and Qt charts."""

from tests.functional_units import functional_units
from dataclasses import fields,is_dataclass,replace
from hashlib import sha256
import inspect
import json
import math
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

from seedvision.benchmarking import native_recipe as runner,native_stage_replay as replay
from seedvision.benchmarking import native_color_replay as compact
from seedvision.benchmarking import color_histogram as chart
from seedvision.cuda import plantcv_color as color
from seedvision.export import native_results as archives
from tests import test_native_recipe as seeds
from tests import test_native_calibrated_range_recipe as calibrated
from tests.test_native_card_recipe import _parameter
from tests import test_plantcv_color as public_color

COLOR='pcv_seed_binary_analyze_color'
PARAMETERS={'n_labels':1,'colorspaces':'rgb','label':'graph'}


def tagged(value):
    if is_dataclass(value):
        return {'kind':'record','type':type(value).__module__+'.'+type(value).__qualname__,
            'fields':{field.name:tagged(getattr(value,field.name)) for field in fields(value)}}
    if type(value) is tuple:return {'kind':'tuple','items':[tagged(item) for item in value]}
    if type(value) is float and not math.isfinite(value):
        return {'kind':'nonfinite','value':'nan' if math.isnan(value) else 'positive_infinity' if value>0 else 'negative_infinity'}
    return value


def zero_result():
    histograms=tuple(color.ColorHistogram(variable,variable.replace('_',' '),color._METHOD,
        'frequency','percent of selected pixels',"<class 'list'>",tuple(range(256)),(0.0,)*256)
        for variable in ('blue_frequencies','green_frequencies','red_frequencies'))
    statistics=tuple(color._statistic(variable,math.nan,'degrees')
        for variable in ('hue_circular_mean','hue_circular_std','hue_median'))
    return color.ColorAnalysisResult('available','rgb',(color.ColorLabelResult(1,'graph_1',0,histograms,statistics),))


@functional_units
class ColourReplayContractTests(CatalogueTestCase):
    def check_test_known_records_nonfinite_tokens_and_immutable_expected_metadata(self):
        result=zero_result();decoded=compact.decode(tagged(result),ValueError)
        self.assertIs(compact.validate(decoded,'observations',PARAMETERS,1,ValueError),decoded)
        self.assertEqual(decoded,compact.canonical(result))
        self.assertEqual(compact.compare(compact.canonical(result),decoded,lambda:False)['mismatched_elements'],0)
        with self.assertRaises(TypeError):decoded['fields']['status']='changed'
        for bad in ({'kind':'array','name':'array_0000'},{'kind':'record','type':'unknown','fields':{}},
                {'kind':'record','type':'seedvision.cuda.plantcv_color.ColorAnalysisResult','fields':{}},
                {'kind':'nonfinite','value':'invalid'},float('nan'),2**65,'x'*2049,'\ud800'):
            with self.assertRaises(ValueError):compact.decode(bad,ValueError)
        deep=0
        for _ in range(compact.MAX_METADATA_DEPTH+2):deep={'kind':'tuple','items':[deep]}
        with self.assertRaises(ValueError):compact.decode(deep,ValueError)
        with patch.object(compact,'MAX_METADATA_ITEMS',1):
            with self.assertRaises(ValueError):compact.decode({'kind':'tuple','items':[]},ValueError)
        with patch.object(compact,'MAX_METADATA_BYTES',10):
            with self.assertRaises(ValueError):compact.decode(tagged(result),ValueError)
        boundary='é'*1024
        self.assertEqual(compact.decode(boundary,ValueError),boundary)
        with self.assertRaises(ValueError):compact.decode('é'*1025,ValueError)
        compact.controls({**PARAMETERS,'label':'é'*512},ValueError)
        with self.assertRaises(ValueError):compact.controls({**PARAMETERS,'label':'é'*513},ValueError)
        for text,is_label in (('x'*2049,False),('x'*1025,True)):
            encoded=[]
            def encoding_profile(frame,event,arg):
                if event=='c_call' and getattr(arg,'__name__',None)=='encode' and getattr(arg,'__self__',None) is text:
                    encoded.append(True)
            previous=sys.getprofile();sys.setprofile(encoding_profile)
            try:
                with self.assertRaises(ValueError):
                    if is_label:compact.controls({**PARAMETERS,'label':text},ValueError)
                    else:compact.decode(text,ValueError)
            finally:sys.setprofile(previous)
            self.assertFalse(encoded,'Over-limit characters must reject before UTF-8 allocation.')

    def check_test_nonfinite_wrong_field_bad_bin_shapes_and_inconsistent_availability_rejected(self):
        first=zero_result();label=first.labels[0];statistic=label.statistics[0]
        for changed in (replace(first,status=('nonfinite','nan')),
            replace(first,labels=(replace(label,histograms=(replace(label.histograms[0],values=(0.0,)),*label.histograms[1:])),)),
            replace(first,labels=(replace(label,statistics=(replace(statistic,available=True),*label.statistics[1:])),))):
            with self.assertRaises(ValueError):compact.validate(compact.canonical(changed),'observations',PARAMETERS,1,ValueError)

    def check_test_comparison_counts_all_values_nonfinite_kinds_and_capped_paths(self):
        actual=compact.canonical(zero_result());result=zero_result();label=result.labels[0]
        changed=replace(result,labels=(replace(label,histograms=(replace(label.histograms[0],values=(1.0,)*256),*label.histograms[1:])),))
        expected=compact.canonical(changed);comparison=compact.compare(actual,expected,lambda:False)
        self.assertEqual(comparison['mismatched_elements'],256);self.assertEqual(len(comparison['mismatch_paths']),32)
        self.assertGreater(comparison['compared_elements'],1500)
        self.assertEqual(compact.compare(('nonfinite','nan'),('nonfinite','positive_infinity'),lambda:False)['mismatched_elements'],1)

    def check_test_effective_controls_and_allocation_free_resource_boundaries(self):
        for values in ({**PARAMETERS,'n_labels':True},{**PARAMETERS,'n_labels':129},
                {**PARAMETERS,'colorspaces':'CMYK'},{**PARAMETERS,'label':['wrong','length']}):
            with self.assertRaises(archives.NativeResultExportError):replay._plan((3,4),'analysis.color',values,84)
        for pixels in (65535,65536,65537):
            plan=replay._plan((1,pixels),'analysis.color',{**PARAMETERS,'n_labels':128},7*pixels)
            self.assertEqual(plan['cuda_bytes'],128*pixels+128*1024*1024)
            self.assertEqual(plan['selected_limit_bytes'],48*1024*1024)
            self.assertEqual(plan['work_details']['n_labels'],128)
        maximum=(1024*1024*1024-128*1024*1024)//128
        replay._plan((1,maximum),'analysis.color',PARAMETERS,7*maximum if 7*maximum<=48*1024*1024 else 1)
        with self.assertRaises(runner.NativeRecipeError):replay._plan((1,maximum+1),'analysis.color',PARAMETERS,1)
        with self.assertRaises(runner.NativeRecipeError):replay._plan((3,4),'analysis.color',PARAMETERS,48*1024*1024+1)

    def test_workflow_lifecycle(self):
        """3 related ColourReplayContractTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_known_records_nonfinite_tokens_and_immutable_expected_metadata)
        self.scenario(self.check_test_comparison_counts_all_values_nonfinite_kinds_and_capped_paths)
        self.scenario(self.check_test_effective_controls_and_allocation_free_resource_boundaries)

    def test_validation_and_recovery(self):
        """1 related ColourReplayContractTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_nonfinite_wrong_field_bad_bin_shapes_and_inconsistent_availability_rejected)


@unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
class ColourReplayTests(CatalogueTestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary=tempfile.TemporaryDirectory(dir='artifacts');cls.root=Path(cls.temporary.name);cls.runs={}
        gray,rgb=seeds.fixture();cls.rgb=rgb
        for space in ('rgb','lab','hsv','all'):
            recipe=_parameter(seeds.selected_recipe(),COLOR,'colorspaces',space)
            saved=runner.run_native_recipe(recipe,torch.tensor(gray,device='cuda'),torch.tensor(rgb,device='cuda'),
                source_sha256=seeds.SOURCE,coordinate_frame=seeds.FRAME,preparation=seeds.PREPARATION)
            assert saved.status=='executed',[(n.stage_id,n.status,n.reason) for n in saved.nodes]
            directory=cls.root/space;archives.export_native_recipe_result(directory,recipe,saved)
            cls.runs[space]=(recipe,saved,directory)

    @classmethod
    def tearDownClass(cls):cls.temporary.cleanup()

    def changed(self,space='rgb',**outputs):
        recipe,saved,_=self.runs[space]
        node=replace(saved.node(COLOR),outputs={**dict(saved.node(COLOR).outputs),**outputs})
        result=replace(saved,nodes=tuple(node if n.stage_id==COLOR else n for n in saved.nodes))
        directory=self.root/('changed-'+str(len(tuple(self.root.iterdir()))));archives.export_native_recipe_result(directory,recipe,result)
        return directory

    def selected_input_archive(self,space,labels,outputs=None,count=1):
        recipe,saved,_=self.runs[space];recipe=_parameter(recipe,COLOR,'n_labels',count)
        parameters=recipe.parameters_for_runner(COLOR)
        if outputs is None:
            outputs=runner._dispatch('analysis.color',{'rgb':torch.tensor(self.rgb,device='cuda'),
                'labels':labels},parameters,None)
        nodes=tuple(replace(n,outputs={**dict(n.outputs),'labels':labels})
            if n.stage_id=='pcv_seed_binary_ws_masked_flood' else
            replace(n,outputs=outputs,parameters=parameters) if n.stage_id==COLOR else n for n in saved.nodes)
        altered=replace(saved,recipe_identity=recipe.identity,nodes=nodes)
        directory=self.root/('selected-'+str(len(tuple(self.root.iterdir()))))
        archives.export_native_recipe_result(directory,recipe,altered)
        return directory

    def attempt(self,directory,stage=COLOR,**kwargs):
        calls=[];selected=[]
        def profile(frame,event,arg):
            if event=='call' and frame.f_code is runner._dispatch.__code__:calls.append(frame.f_locals['operation'])
            if event=='call' and frame.f_code is archives.load_native_recipe_result.__code__:selected.append(frame.f_locals['array_names'])
        from tests.observation import observe_functions
        with observe_functions(profile, runner._dispatch, archives.load_native_recipe_result):
            result=replay.replay_native_stage(directory,stage,**kwargs)
        return result,calls,selected

    def test_four_spaces_compare_every_public_observation_and_replay_only_selected_arrays(self):
        oracle=public_color.PlantCVColorParityTests()
        for space,(recipe,saved,directory) in self.runs.items():
            outputs=saved.node(COLOR).outputs;labels=saved.node('pcv_seed_binary_ws_masked_flood').outputs['labels']
            parameters=dict(saved.node(COLOR).parameters)
            public=oracle.reference(np.ascontiguousarray(self.rgb[...,::-1]),labels.cpu().numpy(),parameters['n_labels'],space,parameters['label'])
            oracle.compare(outputs['observations'],public)
            before={p.name:sha256(p.read_bytes()).hexdigest() for p in directory.iterdir() if p.is_file()}
            result,calls,selected=self.attempt(directory)
            self.assertEqual(result.status,'matched',result.reason);self.assertEqual(calls,['analysis.color'])
            self.assertEqual(len(selected[0]),2);self.assertEqual(result.metadata['plan']['selected_bytes'],7*96*128)
            self.assertEqual(result.metadata['comparison']['mode'],'exact_typed_colour_metadata')
            self.assertGreater(result.metadata['comparison']['compared_elements'],3000)
            self.assertEqual(before,{p.name:sha256(p.read_bytes()).hexdigest() for p in directory.iterdir() if p.is_file()})










class ColourReplayQtTests(CatalogueTestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
        from PySide6.QtWidgets import QApplication
        cls.app=QApplication.instance() or QApplication([])



if __name__=='__main__':unittest.main()
