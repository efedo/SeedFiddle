"""Hook semantics, strict transport and actual pinned public watershed products."""

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

from seedvision.benchmarking import reference_watershed_capture as capture
from seedvision.benchmarking import reference_runner as protocol
from seedvision.benchmarking import plantcv_worker as worker


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "artifacts/plantcv-reference-env/Scripts/python.exe"


def sample(mask, rgb_img=None, *, missing=False, error=None, invalid=None):
    """Synthetic Python frame for hook tests, never a public-reference oracle."""
    dist_transform = np.ones(mask.shape, np.float32)
    local_max_coordinates = np.array([[1, 2]], dtype=np.intp)
    local_max = np.zeros(mask.shape, bool)
    local_max[1, 2] = True
    markers = local_max.astype(np.int32)
    dist_transform1 = -dist_transform
    labels = (mask > 0).astype(np.int32)
    plt_img = np.full((*mask.shape, 3), (11, 57, 203), np.uint8)
    rand_color = [[3, 51, 201], [207, 91, 5]]
    estimated_object_count = 1
    if missing:
        del dist_transform
    if error is not None:
        raise error
    if invalid == 'outside': labels[0,0] = 1
    if invalid == 'negative': labels[1,2] = -1
    if invalid == 'dtype': labels = labels.astype(np.int64)
    if invalid == 'shape': labels = labels[:2]
    return labels


class CaptureHookTests(unittest.TestCase):
    def setUp(self):
        self.mask = np.zeros((4, 6), np.uint8)
        self.mask[1:3, 1:5] = 255

    def call(self, function=sample, **kwargs):
        return capture.capture_watershed(function, mask=self.mask,
            kwargs={"mask": self.mask, **kwargs}, max_bytes=10000, max_elements=10000)

    def test_one_actual_target_call_owned_products_prior_profile_and_trace_restore(self):
        old_profile, old_trace = sys.getprofile(), sys.gettrace()
        events, frames, traced = [], [], []
        def prior(frame, event, arg):
            if frame.f_code is sample.__code__:
                events.append(event)
                if event == "return": frames.append(dict(frame.f_locals))
        def trace(frame, event, arg):
            if frame.f_code is sample.__code__: traced.append(event)
            return trace
        try:
            sys.setprofile(prior);sys.settrace(trace)
            result = self.call()
            self.assertIs(sys.getprofile(), prior);self.assertIs(sys.gettrace(), trace)
        finally:
            sys.setprofile(old_profile);sys.settrace(old_trace)
        self.assertEqual(events.count("call"), 1)
        self.assertEqual(events.count("return"), 1)
        self.assertIn("line", traced)
        self.assertEqual(result.metadata["status"], "available")
        self.assertIs(result.labels, frames[0]["labels"])
        for name, array in result.arrays.items():
            original = frames[0][capture.SOURCE_LOCALS[name]]
            if isinstance(original, np.ndarray):
                self.assertFalse(np.shares_memory(array, original), name)
                original.fill(0)
        self.assertTrue(np.all(result.arrays["distance"] == 1))
        self.assertEqual(result.arrays["palette"].tolist(), [[3,51,201],[207,91,5]])

    def test_missing_local_retains_labels_but_no_partial_capture(self):
        observed = []
        old = sys.getprofile()
        def prior(frame,event,arg):
            if frame.f_code is sample.__code__ and event == 'return': observed.append(arg)
        try:
            sys.setprofile(prior);result = self.call(missing=True)
        finally:sys.setprofile(old)
        self.assertEqual(result.metadata["status"], "unavailable")
        self.assertIn("dist_transform", result.metadata["reason"])
        self.assertEqual(result.arrays, {})
        np.testing.assert_array_equal(result.labels, (self.mask > 0).astype(np.int32))
        self.assertFalse(np.shares_memory(result.labels,observed[0]))
        observed[0].fill(0)
        self.assertTrue(np.any(result.labels))

    def test_malformed_fallback_labels_reject_and_reentrant_capture_is_unavailable(self):
        for invalid in ('outside','negative','dtype','shape'):
            with self.subTest(invalid=invalid):
                with self.assertRaises(capture.WatershedCaptureError):self.call(missing=True,invalid=invalid)
        def reentrant(mask,level=0):
            if level==0:reentrant(mask,1)
            return (mask>0).astype(np.int32)
        result=self.call(reentrant)
        self.assertEqual(result.arrays,{})
        self.assertEqual(result.metadata['status'],'unavailable')
        self.assertIn('more than once',result.metadata['reason'])

    def test_public_and_prior_profiler_errors_propagate_and_restore_hooks(self):
        sentinel = RuntimeError("public error")
        old_profile, old_trace = sys.getprofile(), sys.gettrace()
        def prior(frame, event, arg): pass
        try:
            sys.setprofile(prior)
            with self.assertRaises(RuntimeError) as cm: self.call(error=sentinel)
            self.assertIs(cm.exception, sentinel)
            self.assertIs(sys.getprofile(), prior);self.assertIs(sys.gettrace(), old_trace)
            def broken(frame, event, arg):
                if frame.f_code is sample.__code__: raise sentinel
            sys.setprofile(broken)
            with self.assertRaises(RuntimeError) as cm: self.call()
            self.assertIs(cm.exception, sentinel);self.assertIs(sys.getprofile(), broken)
        finally:
            sys.setprofile(old_profile);sys.settrace(old_trace)

    def test_code_drift_aborts_instead_of_reporting_capture_unavailable(self):
        code = sample.__code__
        def different(**kwargs): return np.zeros((4, 6), np.int32)
        old = sys.getprofile()
        def prior(frame, event, arg):
            if frame.f_code is code and event == "return": sample.__code__ = different.__code__
        try:
            sys.setprofile(prior)
            with self.assertRaises(capture.WatershedCaptureIdentityError): self.call()
            self.assertIs(sys.getprofile(), prior)
        finally:
            sample.__code__ = code;sys.setprofile(old)

    def test_snapshot_admission_precedes_public_call_and_contiguous_copy(self):
        p = self.mask.size
        for max_bytes, max_elements in [(42*p+2,16*p+3),(42*p+3,16*p+2)]:
            with mock.patch.object(capture.np, "array", side_effect=AssertionError("allocation")):
                with self.assertRaisesRegex(capture.WatershedCaptureError, "envelope"):
                    capture.capture_watershed(sample, mask=self.mask, kwargs={"mask":self.mask}, max_bytes=max_bytes, max_elements=max_elements)
        capture.capture_watershed(sample, mask=self.mask, kwargs={"mask":self.mask}, max_bytes=42*p+3, max_elements=16*p+3)

    def test_malformed_count_coordinates_palette_and_sparse_ids_are_unavailable(self):
        observed = {}
        def prior(frame,event,arg):
            if frame.f_code is sample.__code__ and event == "return": observed.update(frame.f_locals)
        old=sys.getprofile()
        try:
            sys.setprofile(prior);self.call()
        finally:sys.setprofile(old)
        def check(change, message):
            local=copy.deepcopy(observed);change(local)
            with self.assertRaisesRegex(capture.WatershedCaptureError,message):
                capture._snapshot(local,local['labels'],self.mask,max_bytes=10000,max_elements=10000)
        check(lambda d:d.update(estimated_object_count=2),"count")
        check(lambda d:d.update(local_max_coordinates=np.array([[99,2]],np.intp)),"frame")
        check(lambda d:d.update(rand_color=[[0,0,0],[256,0,0]]),"rand_color")
        check(lambda d:d['dist_transform'].fill(np.nan),"nonfinite")
        def sparse(d):
            d['markers'][d['markers']>0]=7;d['labels'][d['labels']>0]=7
        check(sparse,"palette")
        # Charge actual descriptors too, even when a direct validator caller
        # does not perform the conservative public-call envelope check.
        with self.assertRaisesRegex(capture.WatershedCaptureError,"aggregate"):
            capture._snapshot(observed,observed['labels'],self.mask,max_bytes=1,max_elements=10000)


class CaptureProtocolTests(unittest.TestCase):
    def products(self, empty=False):
        mask=np.zeros((4,6),np.uint8);mask[1:3,1:5]=255
        result=capture.capture_watershed(sample,mask=mask,kwargs={'mask':mask},max_bytes=10000,max_elements=10000)
        arrays={k:v.copy() for k,v in result.arrays.items()};metadata=copy.deepcopy(result.metadata)
        if empty:
            mask.fill(0);arrays['labels'].fill(0);arrays['markers'].fill(0);arrays['peaks'].fill(False)
            arrays['coordinates']=np.empty((0,2),np.intp);arrays['palette']=arrays['palette'][:1].copy()
            metadata.update(estimated_object_count=0,positive_marker_ids=[],positive_label_ids=[],label_max=0)
        arrays['diagnostic']=arrays['diagnostic_bgr'][...,::-1].copy()
        metadata['call_id']='call1'
        metadata['artifacts']={k:{'source_local':capture.SOURCE_LOCALS.get(k,'plt_img'),'shape':list(v.shape),'dtype':v.dtype.str,'nbytes':v.nbytes,'sha256':hashlib.sha256(v.tobytes()).hexdigest()} for k,v in arrays.items()}
        metadata['artifacts']['diagnostic']['adapter']='BGR to RGB channel reversal'
        if empty:
            del arrays['coordinates'];metadata['empty_coordinates']=copy.deepcopy(capture.EMPTY_COORDINATES)
        return mask,arrays,{'capture':metadata}

    def test_empty_coordinate_record_reconstructs_exact_array_without_relaxing_npz(self):
        mask,arrays,effective=self.products(empty=True);decl=[]
        protocol._validate_watershed_capture(arrays,decl,effective,'call1',{'mask':mask})
        self.assertEqual(arrays['coordinates'].shape,(0,2));self.assertEqual(arrays['coordinates'].dtype,np.int64)
        self.assertEqual(decl,[{'name':'coordinates','shape':[0,2],'dtype':'<i8','nbytes':0}])
        with self.assertRaisesRegex(protocol.ReferenceProtocolError,"shape"):
            protocol.validate_array_mapping({'coordinates':arrays['coordinates']})

    def test_nonempty_artifacts_have_exact_digest_channels_frame_and_count(self):
        mask,arrays,effective=self.products();protocol._validate_watershed_capture(arrays,[],effective,'call1',{'mask':mask})
        arrays['diagnostic'][0,0,0]^=1
        with self.assertRaisesRegex(protocol.ReferenceProtocolError,"identity"):
            protocol._validate_watershed_capture(arrays,[],effective,'call1',{'mask':mask})

    def test_malformed_empty_sentinels_and_partial_capture_reject(self):
        changes={'schema':'wrong','name':'other','shape':[0,3],'dtype':'<i4','nbytes':True,'sha256':'0'*64}
        for key,value in changes.items():
            with self.subTest(key=key):
                mask,arrays,effective=self.products(empty=True);effective['capture']['empty_coordinates'][key]=value
                with self.assertRaisesRegex(protocol.ReferenceProtocolError,'empty-coordinate'):
                    protocol._validate_watershed_capture(arrays,[],effective,'call1',{'mask':mask})
        mask,arrays,effective=self.products();effective['capture']['status']='unavailable'
        with self.assertRaisesRegex(protocol.ReferenceProtocolError,'only public labels'):
            protocol._validate_watershed_capture(arrays,[],effective,'call1',{'mask':mask})
        with mock.patch.object(protocol,'MAX_ARRAYS',8):
            mask,arrays,effective=self.products(empty=True)
            with self.assertRaisesRegex(protocol.ReferenceProtocolError,'inventory'):
                protocol._validate_watershed_capture(arrays,[],effective,'call1',{'mask':mask})

    def test_capture_pins_are_additive_and_contradictory_source_records_reject(self):
        records=capture.capture_source_records();self.assertEqual(worker._capture_source_inventory(records),records)
        records[0]['sha256']='0'*64
        with self.assertRaisesRegex(worker.WorkerError,'contradicts'):worker._capture_source_inventory(records)


ORACLE = r'''
import hashlib, importlib, json, pathlib, sys
import numpy as np
from plantcv import plantcv as pcv
from seedvision.benchmarking.reference_watershed_capture import capture_watershed, capture_source_records, SOURCE_LOCALS
root=pathlib.Path(pcv.__file__).parent
for record in capture_source_records():
 raw=(root/pathlib.Path(record['path']).name).read_bytes()
 assert len(raw)==record['bytes'] and hashlib.sha256(raw).hexdigest()==record['sha256']
pcv.params.debug=None;pcv.params.line_thickness=1;pcv.params.sample_label='oracle'
mask=np.zeros((48,72),np.uint8);yy,xx=np.ogrid[:48,:72]
mask[((xx-25)**2+(yy-24)**2<12**2)|((xx-47)**2+(yy-24)**2<12**2)]=255
bgr=np.full((48,72,3),(13,83,209),np.uint8)
function=pcv.watershed.watershed_segmentation;locals_={};calls=[]
def prior(frame,event,arg):
 if frame.f_code is function.__code__:
  calls.append(event)
  if event=='return' and arg is not None:locals_.update(frame.f_locals)
np.random.seed(2117);sys.setprofile(prior)
original=function(bgr,mask,distance=5,label='oracle');sys.setprofile(None)
expected={name:np.array(locals_[local],copy=True) for name,local in SOURCE_LOCALS.items()}
state=np.random.get_state();np.random.seed(2117)
captured=capture_watershed(function,mask=mask,kwargs=dict(rgb_img=bgr,mask=mask,distance=5,label='oracle'),max_bytes=1000000,max_elements=1000000)
assert calls.count('call')==1 and calls.count('return')==1
assert captured.metadata['status']=='available'
for name,array in captured.arrays.items():assert np.array_equal(array,expected[name]),name
after=np.random.get_state();assert state[0]==after[0] and np.array_equal(state[1],after[1]) and state[2:]==after[2:]
assert np.array_equal(captured.labels,original)
np.savez(sys.argv[1],**captured.arrays)
print(json.dumps({'status':captured.metadata['status'],'count':captured.metadata['estimated_object_count'],'rng_advancement_preserved':True}))
'''


@unittest.skipUnless(REFERENCE.is_file(),"pinned public environment is not provisioned")
class ActualPublicCaptureTests(unittest.TestCase):
    def test_post_call_runtime_drift_withholds_all_outputs(self):
        original=protocol._run_child
        script="""
from seedvision.benchmarking import plantcv_worker as worker
original=worker._verify_identity
calls=0
def changed(expected):
 global calls
 calls+=1
 result=original(expected)
 if calls==2:result['version']='changed-after-call'
 return result
worker._verify_identity=changed
raise SystemExit(worker.main())
"""
        def child(command,**kwargs):
            return original([command[0],'-c',script,*command[3:]],**kwargs)
        with mock.patch.object(protocol,'_run_child',side_effect=child):
            with self.assertRaisesRegex(protocol.ReferenceExecutionError,'runtime identity changed'):
                protocol.PlantCVReferenceRunner(REFERENCE).run('segmentation.watershed.capture',
                    {'rgb':np.zeros((4,6,3),np.uint8),'mask':np.zeros((4,6),np.uint8)},{'distance':1})

    def test_actual_pinned_locals_diagnostics_rng_and_runner_match(self):
        mask=np.zeros((48,72),np.uint8);yy,xx=np.ogrid[:48,:72]
        mask[((xx-25)**2+(yy-24)**2<12**2)|((xx-47)**2+(yy-24)**2<12**2)]=255
        rgb=np.full((48,72,3),(209,83,13),np.uint8)
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'oracle.npz'
            process=subprocess.run([str(REFERENCE),'-c',ORACLE,str(path)],cwd=ROOT,capture_output=True,text=True,timeout=120)
            self.assertEqual(process.returncode,0,process.stderr)
            self.assertTrue(json.loads(process.stdout)['rng_advancement_preserved'])
            expected=dict(np.load(path,allow_pickle=False))
        runner=protocol.PlantCVReferenceRunner(REFERENCE)
        captured=runner.run('segmentation.watershed.capture',{'rgb':rgb,'mask':mask},{'distance':5})
        self.assertEqual(captured.effective_parameters['capture']['status'],'available')
        for name in ('distance','coordinates','peaks','markers','negative_distance','labels'):
            np.testing.assert_array_equal(captured.arrays[name],expected[name])
        np.testing.assert_array_equal(captured.arrays['diagnostic'],captured.arrays['diagnostic_bgr'][...,::-1])
        ordinary=runner.run('segmentation.watershed',{'rgb':rgb,'mask':mask},{'distance':5})
        self.assertEqual(set(ordinary.arrays),{'labels'});self.assertNotIn('capture',ordinary.effective_parameters)
        np.testing.assert_array_equal(ordinary.arrays['labels'],captured.arrays['labels'])

    def test_actual_empty_capture_and_downstream_use_returned_labels(self):
        runner=protocol.PlantCVReferenceRunner(REFERENCE)
        rgb=np.full((48,72,3),(83,139,197),np.uint8)
        empty=runner.run('segmentation.watershed.capture',{'rgb':rgb,'mask':np.zeros((48,72),np.uint8)},{'distance':5})
        self.assertEqual(empty.arrays['coordinates'].shape,(0,2))
        self.assertEqual(empty.effective_parameters['capture']['estimated_object_count'],0)
        self.assertFalse(np.any(empty.arrays['labels']))
        yy,xx=np.ogrid[:48,:72];mask=(((xx-36)**2+(yy-24)**2)<14**2).astype(np.uint8)*255
        result=runner.run('segmentation.watershed.capture',{'rgb':rgb,'mask':mask},{'distance':5})
        labels=result.arrays['labels'];n_labels=int(labels.max())
        size=runner.run('analysis.size',{'image':rgb,'labels':labels},{'n_labels':n_labels})
        color=runner.run('analysis.color',{'rgb':rgb,'labels':labels},{'n_labels':n_labels,'colorspaces':'all'})
        self.assertEqual(size.effective_parameters['n_labels'],n_labels);self.assertTrue(size.observations)
        self.assertEqual(color.effective_parameters['n_labels'],n_labels);self.assertTrue(color.observations)
        np.testing.assert_array_equal(labels,result.arrays['labels'])

    def test_cancel_after_child_return_withholds_capture_and_cleans_transport(self):
        class Cancel:
            value=False
            def is_set(self):return self.value
        cancel=Cancel();original=protocol._run_child
        def child(*args,**kwargs):
            result=original(*args,**kwargs);cancel.value=True;return result
        with tempfile.TemporaryDirectory() as temp:
            runner=protocol.PlantCVReferenceRunner(REFERENCE,temporary_root=temp)
            with mock.patch.object(protocol,'_run_child',side_effect=child):
                with self.assertRaises(protocol.ReferenceCancelledError):
                    runner.run('segmentation.watershed.capture',{'rgb':np.zeros((4,6,3),np.uint8),'mask':np.zeros((4,6),np.uint8)},{'distance':1},cancel_event=cancel)
            self.assertEqual(list(Path(temp).iterdir()),[])


if __name__ == '__main__':unittest.main()
