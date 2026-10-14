"""Typed observations, public colour parity, isolated replay and Qt charts."""
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


class ColourReplayContractTests(unittest.TestCase):
    def test_known_records_nonfinite_tokens_and_immutable_expected_metadata(self):
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

    def test_nonfinite_wrong_field_bad_bin_shapes_and_inconsistent_availability_rejected(self):
        first=zero_result();label=first.labels[0];statistic=label.statistics[0]
        for changed in (replace(first,status=('nonfinite','nan')),
            replace(first,labels=(replace(label,histograms=(replace(label.histograms[0],values=(0.0,)),*label.histograms[1:])),)),
            replace(first,labels=(replace(label,statistics=(replace(statistic,available=True),*label.statistics[1:])),))):
            with self.assertRaises(ValueError):compact.validate(compact.canonical(changed),'observations',PARAMETERS,1,ValueError)

    def test_comparison_counts_all_values_nonfinite_kinds_and_capped_paths(self):
        actual=compact.canonical(zero_result());result=zero_result();label=result.labels[0]
        changed=replace(result,labels=(replace(label,histograms=(replace(label.histograms[0],values=(1.0,)*256),*label.histograms[1:])),))
        expected=compact.canonical(changed);comparison=compact.compare(actual,expected,lambda:False)
        self.assertEqual(comparison['mismatched_elements'],256);self.assertEqual(len(comparison['mismatch_paths']),32)
        self.assertGreater(comparison['compared_elements'],1500)
        self.assertEqual(compact.compare(('nonfinite','nan'),('nonfinite','positive_infinity'),lambda:False)['mismatched_elements'],1)

    def test_effective_controls_and_allocation_free_resource_boundaries(self):
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


@unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
class ColourReplayTests(unittest.TestCase):
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
        previous=sys.getprofile();sys.setprofile(profile)
        try:result=replay.replay_native_stage(directory,stage,**kwargs)
        finally:sys.setprofile(previous)
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

    def test_local_calibrated_range_lineage_and_automatic_count(self):
        recipe=calibrated.selected_recipe();stage=calibrated.PREFIX+'analyze_color'
        path=self.root/'card.png';cv2.imwrite(str(path),calibrated.calibrated.fixture())
        saved=runner.run_native_recipe_path(recipe,path);directory=self.root/'local';archives.export_native_recipe_result(directory,recipe,saved)
        result,calls,_=self.attempt(directory,stage)
        self.assertEqual(result.status,'matched',result.reason);self.assertEqual(calls,['analysis.color'])
        self.assertEqual(result.metadata['frame_lineage']['local_to_parent_xy'],(600,260))
        self.assertEqual(result.metadata['plan']['pixels'],96*128)
        self.assertEqual(result.metadata['parameters']['n_labels'],saved.node(stage).parameters['n_labels'])

    def test_automatic_count_fault_rejected_before_dispatch(self):
        recipe,saved,_=self.runs['rgb'];node=saved.node(COLOR)
        outputs={'observations':replace(node.outputs['observations'],labels=node.outputs['observations'].labels[:1]),
            'histogram':node.outputs['histogram']}
        changed=replace(node,parameters={**dict(node.parameters),'n_labels':1},outputs=outputs)
        altered=replace(saved,nodes=tuple(changed if n.stage_id==COLOR else n for n in saved.nodes))
        directory=self.root/'bad-auto-count';archives.export_native_recipe_result(directory,recipe,altered)
        result,calls,selected=self.attempt(directory)
        self.assertEqual(result.status,'invalid_archive',result.reason);self.assertFalse(calls)
        self.assertEqual(len(selected[0]),2);self.assertFalse(result.outputs)

    def test_empty_modes_and_explicit_binary_remap_ignore_semantics(self):
        empty=torch.zeros((96,128),device='cuda',dtype=torch.int32)
        oracle=public_color.PlantCVColorParityTests()
        for space in ('rgb','lab'):
            directory=self.selected_input_archive(space,empty)
            result,calls,_=self.attempt(directory)
            self.assertEqual(result.status,'matched',result.reason);self.assertEqual(calls,['analysis.color'])
            observations=result.outputs['observations'];self.assertEqual(observations.labels[0].pixel_count,0)
            self.assertTrue(all(not stat.available and math.isnan(stat.value) for stat in observations.labels[0].statistics))
            public=oracle.reference(np.ascontiguousarray(self.rgb[...,::-1]),empty.cpu().numpy(),1,space,'graph')
            oracle.compare(observations,public)
        for space in ('hsv','all'):
            source=self.runs[space][1].node(COLOR).outputs
            observations=replace(source['observations'],labels=source['observations'].labels[:1])
            histogram=replace(source['histogram'],series=source['histogram'].series[:1])
            directory=self.selected_input_archive(space,empty,{'observations':observations,'histogram':histogram})
            result,calls,_=self.attempt(directory)
            self.assertEqual(result.status,'unavailable',result.reason);self.assertEqual(calls,['analysis.color'])
            self.assertIn('UnboundLocalError',result.reason);self.assertFalse(result.outputs)
        binary=empty.clone();binary[12:30,20:40]=255
        for labels in (binary,torch.full_like(empty,255),
                self.runs['rgb'][1].node('pcv_seed_binary_ws_masked_flood').outputs['labels']):
            directory=self.selected_input_archive('rgb',labels)
            result,calls,_=self.attempt(directory)
            self.assertEqual(result.status,'matched',result.reason);self.assertEqual(calls,['analysis.color'])
            public=oracle.reference(np.ascontiguousarray(self.rgb[...,::-1]),labels.cpu().numpy(),1,'rgb','graph')
            oracle.compare(result.outputs['observations'],public)

    def test_private_adapter_source_mutations_and_strict_callbacks_withhold_products(self):
        directory=self.runs['rgb'][2]
        for target in ('bgr_image','rgb','labels'):
            for mode in ('value','storage','raise'):
                changed=[]
                def mutate():
                    frame=inspect.currentframe().f_back
                    while frame is not None:
                        if frame.f_code is color.analyze_color_cuda.__code__ and not changed:
                            tensor=frame.f_locals.get('labeled_mask' if target=='labels' else target)
                            if target=='rgb':
                                outer=frame.f_back
                                while outer is not None and outer.f_code is not runner._dispatch.__code__:outer=outer.f_back
                                tensor=outer.f_locals['inputs']['rgb'] if outer is not None else None
                            if tensor is not None:
                                changed.append(True)
                                if mode=='storage':tensor.data=tensor.clone()
                                else:tensor.reshape(-1)[0]=0
                                if mode=='raise':raise RuntimeError('callback failure after mutation')
                                break
                        frame=frame.f_back
                    return False
                result,_,_=self.attempt(directory,cancel=mutate)
                self.assertTrue(changed,(target,mode));self.assertEqual(result.status,'failed',result.reason)
                self.assertIn('changed during execution',result.reason);self.assertFalse(result.outputs)
        for value in (1,None):
            result,_,_=self.attempt(directory,cancel=lambda:value)
            self.assertEqual(result.status,'failed',result.reason);self.assertFalse(result.outputs)

    def test_well_typed_wrong_histogram_statistic_bins_pixel_count_chart_and_reason_mismatch(self):
        original=self.runs['rgb'][1].node(COLOR).outputs['observations'];label=original.labels[0]
        first=label.histograms[0];values=list(first.values);values[0]=1.0 if values[0]!=1.0 else 2.0
        bins=list(first.bin_labels);bins[1]=0
        statistic=label.statistics[0]
        wrong_stat=replace(statistic,value=1.0,available=True,unavailable_reason=None)
        cases=[replace(label,histograms=(replace(first,values=tuple(values)),*label.histograms[1:])),
            replace(label,histograms=(replace(first,bin_labels=tuple(bins)),*label.histograms[1:])),
            replace(label,statistics=(wrong_stat,*label.statistics[1:])),replace(label,pixel_count=0),
            replace(label,statistics=(replace(statistic,value=math.nan,available=False,unavailable_reason='wrong unavailable reason'),*label.statistics[1:]))]
        for row in cases:
            result,calls,_=self.attempt(self.changed(observations=replace(original,labels=(row,*original.labels[1:]))))
            self.assertEqual(result.status,'mismatch',result.reason);self.assertEqual(calls,['analysis.color'])
            self.assertGreater(result.metadata['comparison']['mismatched_elements'],0)
        hist=self.runs['hsv'][1].node(COLOR).outputs['histogram'];first=hist.series[0];values=list(first.frequencies);values[0]=1.0
        changed=replace(hist,series=(replace(first,frequencies=tuple(values)),*hist.series[1:]))
        result,calls,_=self.attempt(self.changed('hsv',histogram=changed))
        self.assertEqual(result.status,'mismatch',result.reason);self.assertEqual(calls,['analysis.color'])

    def test_malformed_record_is_rejected_before_array_loading(self):
        original=self.runs['rgb'][1].node(COLOR).outputs['observations'];label=original.labels[0]
        changed=replace(original,labels=(replace(label,histograms=(replace(label.histograms[0],values=(1.0,)),*label.histograms[1:])),*original.labels[1:]))
        result,calls,selected=self.attempt(self.changed(observations=changed))
        self.assertEqual(result.status,'invalid_archive',result.reason);self.assertFalse(calls);self.assertFalse(selected)

    def test_128_label_all_bundle_fits_decoder_and_matches_public_and_replay(self):
        recipe,saved,_=self.runs['all'];recipe=_parameter(recipe,COLOR,'n_labels',128)
        labels=torch.arange(96*128,device='cuda',dtype=torch.int32).reshape(96,128)%128+1
        parameters=recipe.parameters_for_runner(COLOR)
        outputs=runner._dispatch('analysis.color',{'rgb':torch.tensor(self.rgb,device='cuda'),'labels':labels},parameters,None)
        oracle=public_color.PlantCVColorParityTests();public=oracle.reference(np.ascontiguousarray(self.rgb[...,::-1]),labels.cpu().numpy(),128,'all',parameters['label'])
        oracle.compare(outputs['observations'],public)
        nodes=tuple(replace(n,outputs={**dict(n.outputs),'labels':labels}) if n.stage_id=='pcv_seed_binary_ws_masked_flood'
            else replace(n,outputs=outputs,parameters=parameters) if n.stage_id==COLOR else n for n in saved.nodes)
        altered=replace(saved,recipe_identity=recipe.identity,nodes=nodes)
        directory=self.root/'all128';archives.export_native_recipe_result(directory,recipe,altered)
        result,calls,_=self.attempt(directory)
        self.assertEqual(result.status,'matched',result.reason);self.assertEqual(calls,['analysis.color'])
        self.assertGreater(result.metadata['comparison']['compared_elements'],620000)

    def test_cancellation_mutation_contract_and_frozen_output_changes_withhold_products(self):
        directory=self.runs['rgb'][2]
        result,_,_=self.attempt(directory,cancel=lambda:True);self.assertEqual(result.status,'cancelled');self.assertFalse(result.outputs)
        handoff=[]
        def cancel_handoff():
            frame=inspect.currentframe().f_back
            while frame is not None:
                if frame.f_code is runner._dispatch.__code__ and 'result' in frame.f_locals:
                    handoff.append(True);return True
                frame=frame.f_back
            return False
        result,_,_=self.attempt(directory,cancel=cancel_handoff)
        self.assertTrue(handoff);self.assertEqual(result.status,'cancelled',result.reason);self.assertFalse(result.outputs)
        for final_callback in (False,True):
            changed=[]
            def mutate():
                frame=inspect.currentframe().f_back
                while frame is not None:
                    if (frame.f_code.co_name=='replay_native_stage' and frame.f_locals.get('color_products') is not None
                            and (not final_callback or 'comparisons' in frame.f_locals)):
                        value=frame.f_locals['outputs']['observations'];object.__setattr__(value,'status','changed');changed.append(True);break
                    frame=frame.f_back
                return False
            result,_,_=self.attempt(directory,cancel=mutate)
            self.assertTrue(changed);self.assertEqual(result.status,'failed',result.reason);self.assertFalse(result.outputs)
        original=compact.MAX_METADATA_ITEMS
        def change_contract():compact.MAX_METADATA_ITEMS=original+1;return False
        try:result,_,_=self.attempt(directory,cancel=change_contract)
        finally:compact.MAX_METADATA_ITEMS=original
        self.assertEqual(result.status,'failed',result.reason);self.assertFalse(result.outputs)


class ColourReplayQtTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
        from PySide6.QtWidgets import QApplication
        cls.app=QApplication.instance() or QApplication([])

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_actual_async_colour_replay_shows_chart_and_preserves_active_run(self):
        from seedvision.ui.native_recipe_dialog import NativeRecipeDialog
        with tempfile.TemporaryDirectory(dir='artifacts') as temporary:
            root=Path(temporary);image=root/'seeds.png';_,rgb=seeds.fixture();cv2.imwrite(str(image),rgb[...,::-1])
            dialog=NativeRecipeDialog(seeds.selected_recipe());dialog.show()
            def wait():
                end=time.monotonic()+40
                while dialog._worker is not None and time.monotonic()<end:self.app.processEvents();time.sleep(.01)
                self.assertIsNone(dialog._worker)
            try:
                dialog.set_source_path(image);dialog.start_run();wait();current=dialog.result
                archives.export_native_recipe_result(root/'run',dialog.recipe,current)
                dialog._select_stage(COLOR);dialog.start_stage_replay(root/'run');wait()
                self.assertEqual(dialog.replay_result.status,'matched',dialog.status_label.text())
                self.assertIs(dialog.result,current);self.assertFalse(dialog.histogram.isHidden())
                self.assertIn('Colour observations: available',dialog.facts.toPlainText())
                self.assertNotIn('ColorHistogram(',dialog.facts.toPlainText())
            finally:dialog.reject();wait();dialog.deleteLater()


if __name__=='__main__':unittest.main()
