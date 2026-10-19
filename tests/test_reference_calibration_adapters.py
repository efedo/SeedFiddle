"""Explicit public ROI and size-scale contexts for a composed reference workflow."""

from tests.catalogue_fixtures import CatalogueTestCase

from tests.functional_units import functional_units
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import cv2
import numpy as np

from seedvision.benchmarking.plantcv_worker import WorkerError, _run_operation
from seedvision.benchmarking.reference_runner import PlantCVReferenceRunner
from tests.test_reference_runner import HAS_REFERENCE_ENV, REFERENCE_PYTHON
from tests.test_plantcv_card_detection import fixture as card_fixture


ROOT = Path(__file__).resolve().parents[1]

DIRECT_SCRIPT = r'''
import hashlib, importlib, json, sys
from pathlib import Path
import numpy as np
from plantcv import plantcv as pcv
for name, digest in (
    ('plantcv.plantcv.transform.auto_correct_color', 'bed2eb9e50e42e39f7ad4bbef157150190585268178f06083092edb19ffaf57a'),
    ('plantcv.plantcv._helpers', 'c05b0c944210657332ce307f2066caab55f2ebc8d8fea11988db3e476e7cdfba'),
    ('plantcv.plantcv.analyze.size', 'db5942d8cda707952e5b43777f3641a195383f62f79ab33a81b4f053bc1eb482'),
):
    assert hashlib.sha256(Path(importlib.import_module(name).__file__).read_bytes()).hexdigest() == digest
def primitive(value):
    if isinstance(value, np.ndarray): return primitive(value.tolist())
    if isinstance(value, np.generic): return primitive(value.item())
    if isinstance(value, dict): return {str(k):primitive(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [primitive(v) for v in value]
    if isinstance(value, type): return value.__name__
    return value
def observations():
    return {name:primitive(getattr(pcv.outputs,name))
            for name in ('observations','measurements','metadata') if getattr(pcv.outputs,name,None)}
with np.load(sys.argv[1], allow_pickle=False) as loaded:
    rgb,labels = loaded['rgb'].copy(),loaded['labels'].copy()
params = json.loads(Path(sys.argv[2]).read_text(encoding='utf-8'))
pcv.params.debug = None
pcv.params.verbose = False
pcv.params.sample_label = 'SeedFiddle'
pcv.params.line_thickness = 1
pcv.params.px_width = pcv.params.px_height = 1.0
pcv.params.unit = 'pixels'
pcv.outputs.clear()
kwargs = dict(adaptive_method=0, color_chip_size=[7,11])
if params['bounds'] is not None:
    x0,y0,x1,y1 = params['bounds']
    kwargs['roi'] = pcv.Objects(
        contours=[[np.array([[[x0,y0]],[[x0,y1]],[[x1,y1]],[[x1,y0]]],dtype=np.int32)]],
        hierarchy=[np.array([[[-1,-1,-1,-1]]],dtype=np.int32)])
corrected = pcv.transform.auto_correct_color(np.ascontiguousarray(rgb[...,::-1]), **kwargs)
correction_observations = observations()
scale = dict(px_width=pcv.params.px_width,px_height=pcv.params.px_height,unit=pcv.params.unit)
pcv.outputs.clear()
size_image = pcv.analyze.size(img=corrected,labeled_mask=labels,n_labels=1,label='SeedFiddle')
np.savez(sys.argv[3],corrected_rgb=np.ascontiguousarray(corrected[...,::-1]),
         size_rgb=np.ascontiguousarray(size_image[...,::-1]))
Path(sys.argv[4]).write_text(json.dumps(primitive(dict(scale=scale,
    correction_observations=correction_observations,size_observations=observations())),
    allow_nan=False,sort_keys=True),encoding='utf-8')
'''


def run_adapter(operation, arrays, parameters, pcv):
    return _run_operation({'operation': operation, 'parameters': parameters}, arrays, Path('.'), pcv)


@functional_units
class ReferenceCalibrationAdapterTests(CatalogueTestCase):
    def setUp(self):
        self.rgb = np.broadcast_to(np.array([250,30,10],np.uint8),(8,12,3)).copy()
        self.labels = np.ones((8,12),np.int32)
        self.outputs = SimpleNamespace(observations={},measurements={},metadata={})
        self.params = SimpleNamespace(px_width=1.0,px_height=1.0,unit='pixels')

    def check_test_roi_bounds_reject_before_object_construction_or_public_call(self):
        pcv = SimpleNamespace(Objects=Mock(),transform=SimpleNamespace(auto_correct_color=Mock()))
        for bounds in ([],[0,0,12],[0,0,12,8,9],[0,0,0,8],[2,1,1,8],[-1,0,12,8],
                       [0,0,13,8],[0,0,12,9],[False,0,12,8],[0.0,0,12,8],'full'):
            with self.subTest(bounds=bounds), self.assertRaises(WorkerError):
                run_adapter('transform.auto_correct_color',{'rgb':self.rgb},{'roi_bounds_xyxy':bounds},pcv)
        pcv.Objects.assert_not_called()
        pcv.transform.auto_correct_color.assert_not_called()

    def check_test_roi_is_exact_half_open_public_object_and_full_frame_rgb_adapter(self):
        seen = {}
        def objects(**kwargs):
            seen.update(kwargs)
            return SimpleNamespace(**kwargs)
        def correct(*,rgb_img,roi,**kwargs):
            seen['roi'] = roi
            np.testing.assert_array_equal(rgb_img,self.rgb[...,::-1])
            return rgb_img.copy()
        pcv = SimpleNamespace(Objects=objects,transform=SimpleNamespace(auto_correct_color=correct),
                              outputs=self.outputs,params=self.params)
        result = run_adapter('transform.auto_correct_color',{'rgb':self.rgb},{'roi_bounds_xyxy':[2,1,12,8]},pcv)
        np.testing.assert_array_equal(seen['contours'][0][0],[[[2,1]],[[2,8]],[[12,8]],[[12,1]]])
        self.assertEqual(seen['contours'][0][0].dtype,np.int32)
        np.testing.assert_array_equal(seen['hierarchy'][0],[[[-1,-1,-1,-1]]])
        np.testing.assert_array_equal(result[0]['corrected_image'],self.rgb)
        self.assertEqual(result[2]['roi_adapter']['effective_bounds_xyxy'],[2,1,12,8])
        self.assertEqual(result[2]['roi_adapter']['corrected_output_frame'],'full input frame')

    def check_test_absent_roi_keeps_old_call_and_explicit_null_declares_full_frame(self):
        calls = []
        def correct(**kwargs):
            calls.append(kwargs)
            return kwargs['rgb_img'].copy()
        pcv = SimpleNamespace(transform=SimpleNamespace(auto_correct_color=correct),
                              outputs=self.outputs,params=self.params)
        old = run_adapter('transform.auto_correct_color',{'rgb':self.rgb},{},pcv)
        selected = run_adapter('transform.auto_correct_color',{'rgb':self.rgb},{'roi_bounds_xyxy':None},pcv)
        self.assertNotIn('roi',calls[0])
        self.assertIsNone(calls[1]['roi'])
        self.assertNotIn('roi_adapter',old[2])
        self.assertEqual(selected[2]['roi_adapter']['effective_bounds_xyxy'],[0,0,12,8])
        selected[2].pop('roi_adapter')
        self.assertEqual(old[1:],selected[1:])
        np.testing.assert_array_equal(old[0]['corrected_image'],selected[0]['corrected_image'])

    def check_test_calibration_changes_public_calculation_and_restores_before_default_call(self):
        seen = []
        def size(*,img,**kwargs):
            context = (self.params.px_width,self.params.px_height,self.params.unit)
            seen.append(context)
            self.outputs.observations = {'area':96*context[0]*context[1],'width':12*context[0],
                                         'center_of_mass':[6,4]}
            return img.copy()
        pcv = SimpleNamespace(analyze=SimpleNamespace(size=size),outputs=self.outputs,params=self.params)
        arrays = {'image':self.rgb,'labels':self.labels}
        selected = run_adapter('analysis.size',arrays,{'calibration':{'px_width':2,'px_height':3,'unit':'mm'}},pcv)
        self.assertEqual(selected[1]['observations'],{'area':576.0,'width':24.0,'center_of_mass':[6,4]})
        self.assertEqual((self.params.px_width,self.params.px_height,self.params.unit),(1.0,1.0,'pixels'))
        default = run_adapter('analysis.size',arrays,{},pcv)
        self.assertEqual(seen,[(2.0,3.0,'mm'),(1.0,1.0,'pixels')])
        self.assertNotIn('calibration',default[2])
        self.assertTrue(selected[2]['calibration_context_restored'])
        self.assertEqual(selected[2]['calibration_origin'],'explicit_pcv.params_adapter')

    def check_test_public_failure_restores_actual_previous_scale_fields(self):
        self.params.px_width,self.params.px_height,self.params.unit = 4.5,6.25,'previous'
        def size(**kwargs):
            self.assertEqual((self.params.px_width,self.params.px_height,self.params.unit),(2.0,3.0,'mm'))
            raise LookupError('actual source failure')
        pcv = SimpleNamespace(analyze=SimpleNamespace(size=size),outputs=self.outputs,params=self.params)
        with self.assertRaisesRegex(LookupError,'actual source failure'):
            run_adapter('analysis.size',{'image':self.rgb,'labels':self.labels},
                        {'calibration':{'px_width':2,'px_height':3,'unit':'mm'}},pcv)
        self.assertEqual((self.params.px_width,self.params.px_height,self.params.unit),(4.5,6.25,'previous'))

    def check_test_invalid_calibration_rejects_before_call_and_preserves_globals(self):
        size = Mock()
        pcv = SimpleNamespace(analyze=SimpleNamespace(size=size),outputs=self.outputs,params=self.params)
        valid = {'px_width':2,'px_height':3,'unit':'mm'}
        invalid = [None,{}, {'px_width':2,'px_height':3},dict(valid,extra=1),dict(valid,px_width=0),
                   dict(valid,px_height=-1),dict(valid,px_width=True),dict(valid,px_width='2'),
                   dict(valid,px_width=float('inf')),dict(valid,px_height=float('nan')),
                   dict(valid,unit=[]),dict(valid,unit='cm')]
        for calibration in invalid:
            with self.subTest(calibration=calibration), self.assertRaises(WorkerError):
                run_adapter('analysis.size',{'image':self.rgb,'labels':self.labels},{'calibration':calibration},pcv)
            self.assertEqual((self.params.px_width,self.params.px_height,self.params.unit),(1.0,1.0,'pixels'))
        size.assert_not_called()

    def test_validation_and_recovery(self):
        """3 related ReferenceCalibrationAdapterTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_roi_bounds_reject_before_object_construction_or_public_call)
        self.scenario(self.check_test_public_failure_restores_actual_previous_scale_fields)
        self.scenario(self.check_test_invalid_calibration_rejects_before_call_and_preserves_globals)

    def test_workflow_lifecycle(self):
        """3 related ReferenceCalibrationAdapterTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_roi_is_exact_half_open_public_object_and_full_frame_rgb_adapter)
        self.scenario(self.check_test_absent_roi_keeps_old_call_and_explicit_null_declares_full_frame)
        self.scenario(self.check_test_calibration_changes_public_calculation_and_restores_before_default_call)


@unittest.skipUnless(HAS_REFERENCE_ENV,'Pinned PlantCV reference environment is required')
class ActualPublicCalibrationAdapterTests(unittest.TestCase):
    def direct(self,rgb,labels,bounds):
        with tempfile.TemporaryDirectory(prefix='public-calibration-adapter-',dir=ROOT/'artifacts') as name:
            root = Path(name)
            np.savez(root/'inputs.npz',rgb=rgb,labels=labels)
            (root/'params.json').write_text(json.dumps({'bounds':bounds}),encoding='utf-8')
            result = subprocess.run([str(REFERENCE_PYTHON),'-c',DIRECT_SCRIPT,str(root/'inputs.npz'),
                str(root/'params.json'),str(root/'outputs.npz'),str(root/'observations.json')],
                cwd=ROOT,capture_output=True,text=True,timeout=120)
            self.assertEqual(result.returncode,0,result.stderr)
            with np.load(root/'outputs.npz',allow_pickle=False) as values:
                arrays = {key:values[key].copy() for key in values.files}
            return arrays,json.loads((root/'observations.json').read_text(encoding='utf-8'))

    def labels(self,shape):
        labels = np.zeros(shape,np.int32)
        cv2.ellipse(labels,(shape[1]//2,shape[0]//2),(24,11),25,0,360,1,-1)
        return labels

    def assert_public(self,rgb,bounds):
        labels = self.labels(rgb.shape[:2])
        expected,metadata = self.direct(rgb,labels,bounds)
        runner = PlantCVReferenceRunner(REFERENCE_PYTHON)
        params = {'adaptive_method':0,'color_chip_size':[7,11]}
        if bounds is not None:
            params['roi_bounds_xyxy'] = bounds
        corrected = runner.run('transform.auto_correct_color',{'rgb':rgb},params)
        np.testing.assert_array_equal(corrected.arrays['corrected_image'],expected['corrected_rgb'])
        self.assertEqual(corrected.observations,metadata['correction_observations'])
        effects = corrected.effective_parameters['calibration_effects']
        calibration = {key:effects[key] for key in ('px_width','px_height','unit')}
        self.assertEqual(calibration,metadata['scale'])
        size = runner.run('analysis.size',{'image':corrected.arrays['corrected_image'],'labels':labels},
                          {'n_labels':1,'calibration':calibration})
        self.assertEqual(size.observations,metadata['size_observations'])
        np.testing.assert_array_equal(size.arrays['analysis_image'],expected['size_rgb'])
        self.assertEqual(size.effective_parameters['calibration'],calibration)
        self.assertTrue(size.effective_parameters['calibration_context_restored'])
        self.assertEqual(size.identity['identity_digest'],corrected.identity['identity_digest'])
        return corrected





if __name__ == '__main__':
    unittest.main()
