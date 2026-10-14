"""Public size products, bounded typed replay, physical units and Qt state."""
from dataclasses import replace
from hashlib import sha256
import inspect
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
from seedvision.benchmarking import native_size_replay as compact
from seedvision.benchmarking import native_calibrated_seed_recipe as units
from seedvision.export import native_results as archives
from seedvision.cuda import plantcv_size as size
from tests import test_native_recipe as seeds,test_native_calibrated_seed_recipe as calibrated
from tests.test_native_card_recipe import _parameter

SIZE='pcv_seed_binary_analyze_size'
LOCAL_SIZE=SIZE
PARAMETERS={'n_labels':1,'label':'graph'}


def tagged(value):
    if isinstance(value,dict) or hasattr(value,'items'):
        return {'kind':'mapping','items':[[key,tagged(item)] for key,item in value.items()]}
    if type(value) is tuple:return {'kind':'tuple','items':[tagged(item) for item in value]}
    return value


def zero_rows():
    return {'graph_1':{item.variable:{'trait':item.trait,'method':item.method,'scale':item.scale,
        'datatype':str(item.datatype),'value':item.value,'label':item.label}
        for item in size._plantcv_size_output_observations(size._zero_plantcv_size_bundle(True))}}


class SizeReplayContractTests(unittest.TestCase):
    def test_exact_nominal_datatypes_zero_bundle_immutable_decoder_and_caps(self):
        original=zero_rows();original['graph_1']['area']['value']=12.5
        decoded=compact.decode(tagged(original),ValueError)
        compact.observations(decoded,PARAMETERS,ValueError)
        self.assertEqual(decoded['graph_1']['area']['datatype'],"<class 'int'>")
        self.assertEqual(decoded['graph_1']['in_bounds']['trait'],'whether the plant goes out of bounds ')
        self.assertEqual(compact.compare(decoded,compact.canonical(original),lambda:False)['mismatched_elements'],0)
        with self.assertRaises(TypeError):decoded['graph_1']['area']['value']=0
        for bad in ({'kind':'array','name':'array_0000'},float('nan'),float('inf'),2**65,'x'*2049,'\ud800',
                {'kind':'mapping','items':[['x',0],['x',1]]}):
            with self.assertRaises(ValueError):compact.decode(bad,ValueError)
        for name,limit in (('MAX_METADATA_ITEMS',1),('MAX_METADATA_BYTES',10),('MAX_METADATA_DEPTH',1)):
            with patch.object(compact,name,limit):
                with self.assertRaises(ValueError):compact.decode(tagged(original),ValueError)
        for bad in ({**PARAMETERS,'n_labels':True},{**PARAMETERS,'n_labels':129},
                {**PARAMETERS,'line_thickness':32768},{**PARAMETERS,'label':'é'*513}):
            with self.assertRaises(ValueError):compact.controls(bad,ValueError)
        compact.controls({**PARAMETERS,'line_thickness':32767,'label':'é'*512},ValueError)

    def test_anisotropic_all_variables_pixel_units_underflow_and_resource_plans(self):
        original=zero_rows()
        for variable in units.AREA_VARIABLES+units.LINEAR_VARIABLES:original['graph_1'][variable]['value']=10.0
        converted=units._scaled_size({'observations':original,'scale':{'px_width':2.0,'px_height':3.0,'unit':'mm'}})
        compact.observations(converted['observations'],PARAMETERS,ValueError,'mm')
        compact.metadata(converted['metadata'],ValueError)
        for variable in units.AREA_VARIABLES:self.assertEqual(converted['observations']['graph_1'][variable]['value'],60.0)
        for variable in units.LINEAR_VARIABLES:self.assertEqual(converted['observations']['graph_1'][variable]['value'],20.0)
        for variable in units.UNSCALED_VARIABLES:self.assertEqual(converted['observations']['graph_1'][variable],original['graph_1'][variable])
        for variable in units.AREA_VARIABLES+units.LINEAR_VARIABLES:
            self.assertEqual(converted['observations']['graph_1'][variable]['scale'],'mm')
        tiny={'px_width':1e-300,'px_height':1e-300,'unit':'mm'}
        compact.scale(tiny,ValueError)
        self.assertEqual(units._scaled_size({'observations':original,'scale':tiny})['metadata']['area_factor'],0.0)
        pixels=units._scaled_size({'observations':original,'scale':{'px_width':None,'px_height':None,'unit':None}})
        self.assertEqual(pixels['observations'],original)
        self.assertEqual(pixels['metadata']['unit'],'pixels')
        maximum=(replay.MAX_REPLAY_CUDA_BYTES-128*1024*1024)//128
        plan=replay._plan((1,maximum),'analysis.size',{**PARAMETERS,'n_labels':128},10*maximum)
        self.assertEqual(plan['cuda_bytes'],replay.MAX_REPLAY_CUDA_BYTES)
        self.assertIsNone(plan['work']);self.assertIsNone(plan['work_limit'])
        with self.assertRaises(replay.NativeStageReplayUnavailable):replay._plan((1,maximum+1),'analysis.size',PARAMETERS,1)
        unit_plan=replay._plan((96,128),'calibrated_seed.size_units',{},0)
        self.assertEqual(unit_plan['cuda_bytes'],64*1024)
        self.assertEqual(unit_plan['work_details']['conversion_cuda_bytes'],0)
        self.assertIsNone(unit_plan['work'])
        with patch.object(replay,'MAX_REPLAY_CUDA_BYTES',64*1024-1):
            with self.assertRaises(replay.NativeStageReplayUnavailable):
                replay._plan((96,128),'calibrated_seed.size_units',{},0)
        with self.assertRaises(archives.NativeResultExportError):replay._plan((96,128),'calibrated_seed.size_units',{},1)

    def test_128_sample_bundle_counts_every_leaf_and_caps_mismatch_paths(self):
        row=zero_rows()['graph_1'];rows={f'graph_{i}':row for i in range(1,129)}
        parameters={**PARAMETERS,'n_labels':128}
        expected=compact.decode(tagged(rows),ValueError);compact.observations(expected,parameters,ValueError)
        changed={sample:{variable:dict(value) for variable,value in values.items()} for sample,values in rows.items()}
        for values in changed.values():values['area']['value']=1
        compared=compact.compare(compact.canonical(changed),expected,lambda:False)
        self.assertEqual(compared['mismatched_elements'],128);self.assertEqual(len(compared['mismatch_paths']),32)
        self.assertGreater(compared['compared_elements'],128*17*6)
        text='x'*2049;encodes=[]
        def profile(frame,event,arg):
            if event=='c_call' and getattr(arg,'__name__',None)=='encode' and getattr(arg,'__self__',None) is text:encodes.append(True)
        previous=sys.getprofile();sys.setprofile(profile)
        try:
            with self.assertRaises(ValueError):compact.decode(text,ValueError)
        finally:sys.setprofile(previous)
        self.assertFalse(encodes)


@unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
class SizeReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary=tempfile.TemporaryDirectory(dir='artifacts');cls.root=Path(cls.temporary.name)
        gray,rgb=seeds.fixture();cls.rgb=rgb;cls.recipe=seeds.selected_recipe()
        cls.saved=runner.run_native_recipe(cls.recipe,torch.tensor(gray,device='cuda'),torch.tensor(rgb,device='cuda'),
            source_sha256=seeds.SOURCE,coordinate_frame=seeds.FRAME,preparation=seeds.PREPARATION)
        assert cls.saved.status=='executed',[(n.stage_id,n.status,n.reason) for n in cls.saved.nodes]
        cls.directory=cls.root/'source';archives.export_native_recipe_result(cls.directory,cls.recipe,cls.saved)
        cls.local_recipe=calibrated.selected_recipe();cls.image=calibrated.fixture()
        path=cls.root/'card.png';cv2.imwrite(str(path),cls.image)
        cls.local_saved=runner.run_native_recipe_path(cls.local_recipe,path)
        assert cls.local_saved.status=='executed',[(n.stage_id,n.status,n.reason) for n in cls.local_saved.nodes]
        cls.local_directory=cls.root/'local';archives.export_native_recipe_result(cls.local_directory,cls.local_recipe,cls.local_saved)

    @classmethod
    def tearDownClass(cls):cls.temporary.cleanup()

    def changed(self,saved=None,recipe=None,**replacements):
        saved=saved or self.saved;recipe=recipe or self.recipe
        altered=replace(saved,nodes=tuple(replacements.get(n.stage_id,n) for n in saved.nodes))
        if any(node.status=='partial' for node in altered.nodes):altered=replace(altered,status='partial')
        directory=self.root/('changed-'+str(len(tuple(self.root.iterdir()))))
        archives.export_native_recipe_result(directory,recipe,altered);return directory

    def attempt(self,directory=None,stage=SIZE,**kwargs):
        calls=[];selected=[]
        def profile(frame,event,arg):
            if event=='call' and frame.f_code is runner._dispatch.__code__:calls.append(frame.f_locals['operation'])
            if event=='call' and frame.f_code is archives.load_native_recipe_result.__code__:selected.append(frame.f_locals['array_names'])
        previous=sys.getprofile();sys.setprofile(profile)
        try:result=replay.replay_native_stage(directory or self.directory,stage,**kwargs)
        finally:sys.setprofile(previous)
        return result,calls,selected

    def test_source_public_observations_diagnostic_and_selected_only_replay(self):
        reference=seeds.NativeRecipeTests();reference.rgb=self.saved.node('pcv_seed_binary_external_inputs').outputs['rgb_image']
        expected,observations=reference.reference()
        reference.assert_size_observations(self.saved.node(SIZE),observations['__size_observations__'])
        np.testing.assert_array_equal(self.saved.node(SIZE).outputs['diagnostic'].cpu().numpy(),expected['size_diagnostic_rgb'])
        before={str(p.relative_to(self.directory)):sha256(p.read_bytes()).hexdigest() for p in self.directory.rglob('*') if p.is_file()}
        result,calls,selected=self.attempt()
        self.assertEqual(result.status,'matched',result.reason);self.assertEqual(calls,['analysis.size'])
        self.assertEqual(len(selected[0]),3);self.assertEqual(result.metadata['plan']['selected_bytes'],10*96*128)
        self.assertGreater(result.metadata['comparison']['compared_elements'],3*96*128+200)
        self.assertEqual(before,{str(p.relative_to(self.directory)):sha256(p.read_bytes()).hexdigest() for p in self.directory.rglob('*') if p.is_file()})

    def test_local_size_and_metadata_only_units_match_public_physical_products(self):
        for stage in (LOCAL_SIZE,units.SCALED_SIZE):
            result,calls,selected=self.attempt(self.local_directory,stage)
            self.assertEqual(result.status,'matched',result.reason)
            self.assertEqual(calls,['analysis.size' if stage==LOCAL_SIZE else 'calibrated_seed.size_units'])
            self.assertEqual(result.metadata['frame_lineage']['local_to_parent_xy'],(600,260))
            if stage==units.SCALED_SIZE:
                self.assertEqual(selected,[()]);self.assertEqual(result.metadata['plan']['work_details']['conversion_cuda_bytes'],0)
                self.assertIn('not recomputed',result.metadata['size_units_lineage']['scope'])
        with patch.object(replay,'MAX_REPLAY_CUDA_BYTES',64*1024-1),patch.object(replay,'_runtime') as runtime:
            rejected,calls,selected=self.attempt(self.local_directory,units.SCALED_SIZE)
        self.assertEqual(rejected.status,'unavailable',rejected.reason)
        self.assertFalse(calls);self.assertFalse(selected);runtime.assert_not_called()
        oracle=calibrated.CalibratedSeedTests();oracle.root=self.root;oracle.image=self.image
        _,public=oracle.reference('binary')
        oracle.assert_observations(self.local_saved.node(units.SCALED_SIZE).outputs['observations'],public['__size_observations__'])
        node=self.local_saved.node(runner._card_recipe.SCALE)
        recipe=_parameter(self.local_recipe,node.stage_id,'color_chip_size',None)
        raw=self.local_saved.node(LOCAL_SIZE).outputs['observations']
        scale={'px_width':None,'px_height':None,'unit':None}
        scaled=units._scaled_size({'observations':raw,'scale':scale})
        saved=replace(self.local_saved,recipe_identity=recipe.identity)
        directory=self.changed(saved,recipe,**{node.stage_id:replace(node,parameters={'color_chip_size':None},outputs={'metadata':scale}),
            units.SCALED_SIZE:replace(saved.node(units.SCALED_SIZE),outputs=scaled)})
        result,calls,selected=self.attempt(directory,units.SCALED_SIZE)
        self.assertEqual(result.status,'matched',result.reason);self.assertEqual(selected,[()])
        self.assertEqual(result.outputs['observations'],raw)

    def test_well_typed_changed_metric_metadata_and_missing_diagnostic_are_mismatches(self):
        node=self.saved.node(SIZE);observations={sample:{variable:dict(row) for variable,row in values.items()} for sample,values in node.outputs['observations'].items()}
        observations[next(iter(observations))]['perimeter']['value']+=1.0
        result,_,_=self.attempt(self.changed(**{SIZE:replace(node,outputs={**dict(node.outputs),'observations':observations})}))
        self.assertEqual(result.status,'mismatch',result.reason);self.assertEqual(result.metadata['comparison']['mismatched_elements'],1)
        omitted=replace(node,status='partial',reason='Unavailable outputs: diagnostic',outputs={'observations':node.outputs['observations']})
        result,_,_=self.attempt(self.changed(**{SIZE:omitted}))
        self.assertEqual(result.status,'mismatch',result.reason)
        self.assertEqual(result.metadata['comparison']['by_port']['diagnostic']['kind'],'output_availability')
        node=self.local_saved.node(units.SCALED_SIZE)
        result,_,_=self.attempt(self.changed(self.local_saved,self.local_recipe,**{node.stage_id:replace(node,
            outputs={**dict(node.outputs),'metadata':{**dict(node.outputs['metadata']),'area_factor':1.5}})}),units.SCALED_SIZE)
        self.assertEqual(result.status,'mismatch',result.reason);self.assertEqual(result.metadata['comparison']['mismatched_elements'],1)

    def test_malformed_descriptor_and_scale_lineage_reject_before_loading(self):
        node=self.saved.node(SIZE);rows={sample:{variable:dict(row) for variable,row in values.items()} for sample,values in node.outputs['observations'].items()}
        rows[next(iter(rows))]['in_bounds']['trait']='trimmed trailing space'
        result,calls,selected=self.attempt(self.changed(**{SIZE:replace(node,outputs={**dict(node.outputs),'observations':rows})}))
        self.assertEqual(result.status,'invalid_archive',result.reason);self.assertFalse(calls);self.assertFalse(selected)
        node=self.local_saved.node(runner._card_recipe.SCALE)
        changed=replace(node,outputs={'metadata':{**dict(node.outputs['metadata']),'px_width':2.0}})
        result,calls,selected=self.attempt(self.changed(self.local_saved,self.local_recipe,**{node.stage_id:changed}),units.SCALED_SIZE)
        self.assertEqual(result.status,'invalid_archive',result.reason);self.assertFalse(calls);self.assertFalse(selected)

    def test_auto_count_fault_and_explicit_empty_label_zero_bundle(self):
        node=self.saved.node(SIZE);first=next(iter(node.outputs['observations']))
        altered=replace(node,parameters={**dict(node.parameters),'n_labels':1},outputs={**dict(node.outputs),'observations':{first:node.outputs['observations'][first]}})
        result,calls,_=self.attempt(self.changed(**{SIZE:altered}))
        self.assertEqual(result.status,'invalid_archive',result.reason);self.assertFalse(calls)
        recipe=_parameter(self.recipe,SIZE,'n_labels',1);parameters=recipe.parameters_for_runner(SIZE)
        flood=self.saved.node('pcv_seed_binary_ws_masked_flood');empty=torch.zeros_like(flood.outputs['labels'])
        outputs=runner._dispatch('analysis.size',{'image':torch.tensor(self.rgb,device='cuda'),'labels':empty},parameters,None)
        saved=replace(self.saved,recipe_identity=recipe.identity)
        directory=self.changed(saved,recipe,**{SIZE:replace(node,parameters=parameters,outputs=outputs),
            flood.stage_id:replace(flood,outputs={**dict(flood.outputs),'labels':empty})})
        result,calls,_=self.attempt(directory)
        self.assertEqual(result.status,'matched',result.reason);self.assertEqual(calls,['analysis.size'])
        self.assertEqual(result.outputs['observations']['graph_1']['area']['value'],0)

    def test_private_mask_canvas_and_original_inputs_mutations_withhold_outputs(self):
        for target in ('mask','image','labels','canvas'):
            for mode in ('value','storage','raise','dtype','dtype_raise'):
                changed=[]
                def mutate():
                    frame=inspect.currentframe().f_back
                    while frame is not None:
                        if not changed:
                            tensor=frame.f_locals.get(target)
                            if target in ('image','labels') and frame.f_code is runner._dispatch.__code__:
                                tensor=frame.f_locals['inputs'].get(target)
                            if torch.is_tensor(tensor):
                                changed.append(True)
                                if mode in ('dtype','dtype_raise'):tensor.data=tensor.view(torch.int8 if tensor.dtype==torch.uint8 else torch.float32)
                                elif mode=='storage':tensor.set_(tensor.clone())
                                else:tensor.view(-1)[0]^=1
                                if mode in ('raise','dtype_raise'):raise RuntimeError('callback changed '+target)
                                return False
                        frame=frame.f_back
                    return False
                result,_,_=self.attempt(cancel=mutate)
                self.assertTrue(changed,target);self.assertEqual(result.status,'failed',(target,mode,result.reason));self.assertFalse(result.outputs)
        result,_,_=self.attempt(cancel=lambda:1)
        self.assertEqual(result.status,'failed',result.reason);self.assertFalse(result.outputs)
        cancelled=[]
        def cancel_in_size():
            frame=inspect.currentframe().f_back
            while frame is not None:
                if frame.f_code is size.premeasure_plantcv_binary_mask.__code__:cancelled.append(True);return True
                frame=frame.f_back
            return False
        result,_,_=self.attempt(cancel=cancel_in_size)
        self.assertTrue(cancelled);self.assertEqual(result.status,'cancelled',result.reason);self.assertFalse(result.outputs)

    def test_controlled_diagnostic_omission_and_numerical_unavailability_classification(self):
        node=self.saved.node(SIZE)
        omitted=replace(node,status='partial',reason='Unavailable outputs: diagnostic',outputs={'observations':node.outputs['observations']})
        directory=self.changed(**{SIZE:omitted})
        runtime=replay._runtime(self.recipe,torch.device('cuda',torch.cuda.current_device()))
        with patch.object(runner,'_runtime_records',return_value=runtime),patch.object(size,'render_plantcv_size_diagnostic_rgb',side_effect=size.RasterSpanLimitError('controlled drawing limit')):
            result,_,_=self.attempt(directory)
        self.assertEqual(result.status,'matched',result.reason);self.assertEqual(set(result.outputs),{'observations'})
        self.assertEqual(result.outputs['observations'],node.outputs['observations'])
        with patch.object(runner,'_runtime_records',return_value=runtime),patch.object(size,'premeasure_plantcv_binary_mask',side_effect=NotImplementedError('controlled numerical limit')):
            result,_,_=self.attempt()
        self.assertEqual(result.status,'unavailable',result.reason);self.assertFalse(result.outputs)

    def test_final_contract_or_diagnostic_mutation_and_units_cancellation_withhold_outputs(self):
        original=compact.MAX_METADATA_ITEMS
        for target in ('contract','diagnostic'):
            changed=[]
            def mutate_final():
                frame=inspect.currentframe().f_back
                while frame is not None:
                    if not changed and target=='contract' and frame.f_code is compact.compare.__code__:
                        compact.MAX_METADATA_ITEMS=original+1;changed.append(True);return False
                    if not changed and target=='diagnostic' and frame.f_code is replay._compare_product.__code__:
                        tensor=frame.f_locals.get('actual')
                        if torch.is_tensor(tensor):tensor.view(-1)[0]^=1;changed.append(True);return False
                    frame=frame.f_back
                return False
            try:result,_,_=self.attempt(cancel=mutate_final)
            finally:compact.MAX_METADATA_ITEMS=original
            self.assertTrue(changed);self.assertEqual(result.status,'failed',result.reason);self.assertFalse(result.outputs)
        cancelled=[]
        def cancel_units():
            frame=inspect.currentframe().f_back
            while frame is not None:
                if frame.f_code is compact.compare.__code__:cancelled.append(True);return True
                frame=frame.f_back
            return False
        result,_,selected=self.attempt(self.local_directory,units.SCALED_SIZE,cancel=cancel_units)
        self.assertTrue(cancelled);self.assertEqual(result.status,'cancelled',result.reason)
        self.assertEqual(selected,[()]);self.assertFalse(result.outputs)


class SizeReplayQtTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
        from PySide6.QtWidgets import QApplication
        cls.app=QApplication.instance() or QApplication([])

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_actual_async_size_and_units_replay_preserves_active_run(self):
        from seedvision.ui.native_recipe_dialog import NativeRecipeDialog
        with tempfile.TemporaryDirectory(dir='artifacts') as temporary:
            root=Path(temporary);path=root/'card.png';cv2.imwrite(str(path),calibrated.fixture())
            dialog=NativeRecipeDialog(calibrated.selected_recipe());dialog.show()
            def wait():
                end=time.monotonic()+60
                while dialog._worker is not None and time.monotonic()<end:self.app.processEvents();time.sleep(.01)
                self.assertIsNone(dialog._worker)
            try:
                dialog.set_source_path(path);dialog.start_run();wait();current=dialog.result
                archives.export_native_recipe_result(root/'run',dialog.recipe,current)
                for stage in (LOCAL_SIZE,units.SCALED_SIZE):
                    dialog._select_stage(stage);dialog.start_stage_replay(root/'run');wait()
                    self.assertEqual(dialog.replay_result.status,'matched',dialog.status_label.text())
                    self.assertIs(dialog.result,current);self.assertIn('Size observations:',dialog.facts.toPlainText())
                    self.assertNotIn('Compact outputs:',dialog.facts.toPlainText())
            finally:dialog.reject();wait();dialog.deleteLater()


if __name__=='__main__':unittest.main()
