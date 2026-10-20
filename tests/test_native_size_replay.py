"""Public size products, bounded typed replay, physical units and Qt state."""

from tests.functional_units import functional_units
from dataclasses import replace
import inspect
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
from seedvision.benchmarking import native_size_replay as compact
from seedvision.benchmarking import native_calibrated_seed_recipe as units
from seedvision.export import native_results as archives
from seedvision.cuda import plantcv_size as size
from tests import test_native_recipe as seeds,test_native_calibrated_seed_recipe as calibrated
from tests.test_native_card_recipe import _parameter
from tests.replay_fixtures import archive_hashes, changed_archive, replay_attempt

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


@functional_units
class SizeReplayContractTests(CatalogueTestCase):
    def check_test_exact_nominal_datatypes_zero_bundle_immutable_decoder_and_caps(self):
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
        for bad in ({**PARAMETERS,'n_labels':True},{**PARAMETERS,'n_labels':1025},
                {**PARAMETERS,'line_thickness':32768},{**PARAMETERS,'label':'é'*513}):
            with self.assertRaises(ValueError):compact.controls(bad,ValueError)
        compact.controls({**PARAMETERS,'line_thickness':32767,'label':'é'*512},ValueError)
        self.assertEqual(len(compact.controls({**PARAMETERS,'n_labels':1024},ValueError)),runner.MAX_LABELS)

    def check_test_anisotropic_all_variables_pixel_units_underflow_and_resource_plans(self):
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

    def check_test_128_sample_bundle_counts_every_leaf_and_caps_mismatch_paths(self):
        row=zero_rows()['graph_1'];rows={f'graph_{i}':row for i in range(1,129)}
        parameters={**PARAMETERS,'n_labels':128}
        expected=compact.decode(tagged(rows),ValueError);compact.observations(expected,parameters,ValueError)
        changed={sample:{variable:dict(value) for variable,value in values.items()} for sample,values in rows.items()}
        for values in changed.values():values['area']['value']=1
        compared=compact.compare(compact.canonical(changed),expected,lambda:False)
        self.assertEqual(compared['mismatched_elements'],128);self.assertEqual(len(compared['mismatch_paths']),32)
        self.assertGreater(compared['compared_elements'],128*17*6)
        # Current-sized compact observations survive source-absent transport and units.
        rows={f'graph_{i}':row for i in range(1,273)}
        parameters={**PARAMETERS,'n_labels':272}
        expected=compact.decode(tagged(rows),ValueError)
        compact.observations(expected,parameters,ValueError)
        converted=units._scaled_size({'observations':expected,
            'scale':{'px_width':2.0,'px_height':3.0,'unit':'mm'}})
        compact.observations(converted['observations'],parameters,ValueError,'mm')
        self.assertEqual(len(converted['observations']),272)
        self.assertEqual(compact.compare(compact.canonical(rows),expected,lambda:False)['mismatched_elements'],0)
        # The count change does not relax metadata traversal admission.
        with patch.object(compact,'MAX_METADATA_ITEMS',100):
            with self.assertRaisesRegex(ValueError,'traversal bound'):compact.decode(tagged(rows),ValueError)
        with self.assertRaises(ValueError):
            units._scaled_size({'observations':{f'graph_{i}':row for i in range(1025)},
                'scale':{'px_width':None,'px_height':None,'unit':None}})
        text='x'*2049;encodes=[]
        def profile(frame,event,arg):
            if event=='c_call' and getattr(arg,'__name__',None)=='encode' and getattr(arg,'__self__',None) is text:encodes.append(True)
        previous=sys.getprofile();sys.setprofile(profile)
        try:
            with self.assertRaises(ValueError):compact.decode(text,ValueError)
        finally:sys.setprofile(previous)
        self.assertFalse(encodes)

    def test_workflow_lifecycle(self):
        """3 related SizeReplayContractTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_exact_nominal_datatypes_zero_bundle_immutable_decoder_and_caps)
        self.scenario(self.check_test_anisotropic_all_variables_pixel_units_underflow_and_resource_plans)
        self.scenario(self.check_test_128_sample_bundle_counts_every_leaf_and_caps_mismatch_paths)


@unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
class SizeReplayTests(CatalogueTestCase):
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
        return changed_archive(self.root,recipe,saved,replacements,partial=True)

    def attempt(self,directory=None,stage=SIZE,**kwargs):
        return replay_attempt(directory or self.directory,stage,**kwargs)







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
        self.check_above_128_source_absent_size_replay()

    def check_above_128_source_absent_size_replay(self):
        gray=np.zeros((96,128),dtype=np.uint8)
        for y in range(12):
            for x in range(12):gray[2+7*y:7+7*y,2+7*x:7+7*x]=210
        rgb=np.repeat(gray[...,None],3,axis=2)
        recipe=seeds.selected_recipe(**{'filter.median_blur':{'ksize':1},
            'morphology.fill':{'size':1},'morphology.erode':{'ksize':3},
            'morphology.dilate':{'ksize':3},'reference.internal.watershed.local_maxima':{'distance':1}})
        saved=runner.run_native_recipe(recipe,torch.tensor(gray,device='cuda'),torch.tensor(rgb,device='cuda'),
            source_sha256=seeds.SOURCE,coordinate_frame=seeds.FRAME,preparation=seeds.PREPARATION)
        self.assertEqual(saved.status,'executed',[(node.stage_id,node.reason) for node in saved.nodes])
        expected=saved.node(SIZE).outputs
        self.assertEqual(len(expected['observations']),144)
        directory=self.root/'above128-source-absent'
        archives.export_native_recipe_result(directory,recipe,saved)
        result,_,_=self.attempt(directory)
        self.assertEqual(result.status,'matched',result.reason)
        self.assertEqual(result.outputs['observations'],expected['observations'])
        self.assertTrue(torch.equal(result.outputs['diagnostic'],expected['diagnostic']))



class SizeReplayQtTests(CatalogueTestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
        from PySide6.QtWidgets import QApplication
        cls.app=QApplication.instance() or QApplication([])



if __name__=='__main__':unittest.main()
