"""Pinned public reader semantics and shared native preparation boundaries."""
from dataclasses import replace
from hashlib import sha256
from importlib.metadata import version
import importlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image
import torch

from seedvision.benchmarking import native_image_inputs as inputs
from seedvision.benchmarking import native_recipe as runner
from seedvision.segmentation.baseline import PipelineAnalysisCache

ROOT=Path(__file__).resolve().parents[1]
REFERENCE=ROOT/'artifacts/plantcv-reference-env/Scripts/python.exe'
READER_SHA='09035b0caa1f91a503ecda82e39e082dfec2d84f78c8f8fdc8c3fe3854fbe770'
ARMS=({}, {'_automatic_card':True}, {'_calibrated':True}, {'_core_recovery':True},
      {'_watershed_debug':True}, {'_smpts_paper':True}, {'_calibrated':True,'_calibrated_range':True},
      {'_watershed_debug':True,'_calibrated':True},
      {'_watershed_debug':True,'_calibrated':True,'_calibrated_range':True})


def fixtures(directory):
    y,x=np.indices((17,23))
    rgb=np.stack(((x*13+y*7)%256,(x*3+y*19)%256,(x*29+y*11)%256),axis=-1).astype(np.uint8)
    image=Image.fromarray(rgb)
    paths=[]
    def save(name,value,**kw):
        path=directory/name;value.save(path,**kw);paths.append(path)
    for suffix in ('png','jpg','bmp','tif'):
        save('color.'+suffix,image)
    gray=Image.fromarray(((x*17+y*23)%256).astype(np.uint8))
    save('gray.png',gray);save('gray.tif',gray)
    palette=gray.convert('P')
    palette.putpalette([v for i in range(256) for v in (i,(i*3)%256,255-i)])
    save('palette.png',palette);save('palette.tif',palette)
    save('transparent-palette.png',palette,transparency=3)
    rgba=np.concatenate((rgb,((x+y)%2*255).astype(np.uint8)[...,None]),axis=-1)
    save('alpha.png',Image.fromarray(rgba))
    values=np.resize(np.array([0,1,255,256,257,32767,32768,65279,65280,65535],np.uint16),(17,23))
    save('depth16.png',Image.fromarray(values));save('depth16.tif',Image.fromarray(values))
    for orientation in range(2,9):
        exif=Image.Exif();exif[274]=orientation
        save(f'exif-{orientation}.jpg',image,exif=exif)
    save('cmyk.jpg',image.convert('CMYK'))
    save('pages.tif',image,save_all=True,append_images=[Image.fromarray(255-rgb)])
    for original,name in [('color.png','png-content.jpg'),('color.jpg','jpeg-content.png')]:
        path=directory/name;path.write_bytes((directory/original).read_bytes());paths.append(path)
    unicode_path=directory/'seed-色.png';unicode_path.write_bytes((directory/'color.png').read_bytes());paths.append(unicode_path)
    for name,raw in [('empty.png',b''),('corrupt.jpg',b'\xff\xd8broken'),
                     ('unsupported.png',b'ordinary text, not an image')]:
        path=directory/name;path.write_bytes(raw);paths.append(path)
    paths.append(directory/'missing.png')
    return paths


REFERENCE_SCRIPT=r'''
from hashlib import sha256
from importlib.metadata import version
import importlib,json,sys
from pathlib import Path
import cv2,numpy as np
from plantcv import plantcv as pcv
assert version('plantcv')=='4.11.3'
assert version('opencv-python')=='4.11.0.86'
assert version('numpy')=='2.2.6'
reader=importlib.import_module('plantcv.plantcv.readimage')
assert sha256(Path(reader.__file__).read_bytes()).hexdigest()==sys.argv[3]
pcv.params.debug=None;pcv.params.verbose=False
paths=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
arrays={};rows=[]
for i,name in enumerate(paths):
    path=Path(name)
    row={'name':path.name,'source_sha256':sha256(path.read_bytes()).hexdigest() if path.exists() else None}
    try:
        bgr,source_dir,source_name=pcv.readimage(filename=name,mode='rgb')
        assert source_name==path.name
        arrays[f'bgr_{i}']=bgr
        arrays[f'gray_{i}']=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
        row.update(success=True,shape=list(bgr.shape),dtype=str(bgr.dtype),
                   pixel_sha256=sha256(bgr.tobytes()).hexdigest())
    except Exception as error:
        row.update(success=False,error_type=type(error).__name__)
    rows.append(row)
np.savez(sys.argv[2]+'.npz',**arrays)
Path(sys.argv[2]+'.json').write_text(json.dumps(rows),encoding='utf-8')
'''


class ImageInputContractTests(unittest.TestCase):
    def test_source_pin_and_explicit_mode(self):
        pin=json.loads((ROOT/'config/peer_input_decode_contract.json').read_text(encoding='utf-8'))
        self.assertEqual(pin['reader']['source_sha256'],READER_SHA)
        self.assertEqual(inputs.PLANTCV_READER_SHA256,READER_SHA)
        self.assertEqual(pin['reader']['mode'],inputs.READER_MODE)
        self.assertEqual(pin['native']['decode_flags'],inputs.DECODE_FLAGS)
        self.assertEqual(pin['native']['max_encoded_bytes'],inputs.MAX_ENCODED_BYTES)
        self.assertEqual(pin['reader']['color_order'],'BGR')
        self.assertIn('not codec-direct',inputs.GRAY_CONTRACT)

    def test_snapshot_identity_size_bounds_and_changed_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'source.png';raw=b'12345678';path.write_bytes(raw)
            identity=inputs.source_identity(path)
            self.assertEqual((identity.size,identity.sha256),(len(raw),sha256(raw).hexdigest()))
            self.assertEqual(inputs.snapshot(path,identity),raw)
            path.write_bytes(b'87654321')
            with self.assertRaises(inputs.ImageInputError):inputs.snapshot(path,identity)
            with patch.object(inputs,'MAX_ENCODED_BYTES',7):
                with self.assertRaises(inputs.ImageInputError):inputs.source_identity(path)
                with self.assertRaises(inputs.ImageInputError):inputs.decode(raw)
            path.write_bytes(b'')
            with self.assertRaises(inputs.ImageInputError):inputs.source_identity(path)

    def test_pinned_public_format_depth_alpha_orientation_and_failure_matrix(self):
        self.assertTrue(REFERENCE.is_file(),'Pinned PlantCV interpreter is required.')
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary);paths=fixtures(directory)
            manifest=directory/'paths.json';manifest.write_text(json.dumps([str(p) for p in paths]),encoding='utf-8')
            output=directory/'reference'
            process=subprocess.run([str(REFERENCE),'-c',REFERENCE_SCRIPT,str(manifest),str(output),READER_SHA],
                                   capture_output=True,text=True,timeout=120)
            self.assertEqual(process.returncode,0,process.stderr)
            rows=json.loads(output.with_suffix('.json').read_text(encoding='utf-8'))
            self.assertEqual(len(rows),len(paths))
            with np.load(output.with_suffix('.npz'),allow_pickle=False) as expected:
                for i,(path,row) in enumerate(zip(paths,rows)):
                    with self.subTest(case=path.name):
                        if path.name=='seed-色.png':
                            # Pinned Windows imread cannot open this Unicode path.
                            # The native byte reader supports it as an explicit extension.
                            self.assertFalse(row['success'])
                            self.assertTrue(row['error_type'])
                            identity=inputs.source_identity(path)
                            np.testing.assert_array_equal(inputs.decode(inputs.snapshot(path,identity)),expected['bgr_0'])
                            if torch.cuda.is_available():
                                prepared=runner.prepare_native_recipe_inputs(path)
                                np.testing.assert_array_equal(prepared.rgb.cpu().numpy(),expected['bgr_0'][...,::-1])
                                np.testing.assert_array_equal(prepared.gray.cpu().numpy(),expected['gray_0'])
                        elif row['success']:
                            raw=path.read_bytes();identity=inputs.source_identity(path)
                            self.assertEqual(identity.sha256,row['source_sha256'])
                            bgr=inputs.decode(inputs.snapshot(path,identity))
                            np.testing.assert_array_equal(bgr,expected[f'bgr_{i}'])
                            self.assertEqual(sha256(bgr.tobytes()).hexdigest(),row['pixel_sha256'])
                            if torch.cuda.is_available():
                                prepared=runner.prepare_native_recipe_inputs(path)
                                np.testing.assert_array_equal(prepared.rgb.cpu().numpy(),expected[f'bgr_{i}'][...,::-1])
                                np.testing.assert_array_equal(prepared.gray.cpu().numpy(),expected[f'gray_{i}'])
                                self.assertEqual(prepared.source_sha256,row['source_sha256'])
                                self.assertEqual(prepared.coordinate_frame.shape,tuple(row['shape'][:2]))
                        else:
                            with self.assertRaises((inputs.ImageInputError,OSError)):
                                identity=inputs.source_identity(path)
                                inputs.decode(inputs.snapshot(path,identity))
            self.assertEqual(sum(row['success'] for row in rows),23)


@unittest.skipUnless(torch.cuda.is_available(),'CUDA preparation is required')
class NativePreparationBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory()
        self.path=Path(self.temporary.name)/'source.png'
        self.bgr=np.arange(13*17*3,dtype=np.uint8).reshape(13,17,3)
        self.assertTrue(cv2.imwrite(str(self.path),self.bgr))

    def tearDown(self):
        self.temporary.cleanup()

    def test_cache_reuses_pixels_repairs_detached_facts_and_equal_size_replacement(self):
        cache=PipelineAnalysisCache()
        first=runner.prepare_native_recipe_inputs(self.path,cache=cache)
        with patch.object(cv2,'imdecode',side_effect=AssertionError('Cache must not decode')):
            second=runner.prepare_native_recipe_inputs(self.path,cache=cache)
        self.assertIs(second.gray,first.gray);self.assertIs(second.rgb,first.rgb)
        object.__setattr__(first,'source_sha256','f'*64)
        repaired=runner.prepare_native_recipe_inputs(self.path,cache=cache)
        self.assertIsNot(repaired.gray,first.gray)
        same_path=self.path.with_suffix('.bmp')
        self.assertTrue(cv2.imwrite(str(same_path),self.bgr))
        same_size_baseline=runner.prepare_native_recipe_inputs(same_path,cache=cache)
        original=same_path.read_bytes()
        replacement=cv2.imencode('.bmp',255-self.bgr)[1].tobytes()
        self.assertEqual(len(original),len(replacement))
        # Byte identity, not file size, determines reuse.
        same_path.write_bytes(replacement)
        changed=runner.prepare_native_recipe_inputs(same_path,cache=cache)
        self.assertNotEqual(changed.source_sha256,same_size_baseline.source_sha256)
        np.testing.assert_array_equal(changed.rgb.cpu().numpy(),(255-self.bgr)[...,::-1])
        self.assertNotEqual(original,replacement)

    def test_strict_bool_cancellation_and_caller_exception_identity(self):
        for stopped in (None,0,1,np.bool_(False)):
            with self.subTest(value=repr(stopped)),self.assertRaises(TypeError):
                runner.prepare_native_recipe_inputs(self.path,cancel=lambda:stopped)
        cache=PipelineAnalysisCache()
        with self.assertRaises(runner.NativeRecipeCancelled):
            runner.prepare_native_recipe_inputs(self.path,cache=cache,cancel=lambda:True)
        self.assertNotIn('reference.prepared',cache.values)
        failure=ValueError('Original caller exception')
        def raises():raise failure
        with self.assertRaises(ValueError) as caught:
            runner.prepare_native_recipe_inputs(self.path,cancel=raises)
        self.assertIs(caught.exception,failure)

    def test_initial_source_and_cached_fields_mutation_priority_all_recipe_arms(self):
        for arm in ARMS:
            for raises in (False,True):
                cache=PipelineAnalysisCache()
                prepared=runner.prepare_native_recipe_inputs(self.path,cache=cache,**arm)
                def mutate():
                    object.__setattr__(prepared.coordinate_frame,'frame_id','changed')
                    if raises:raise ValueError('Caller after metadata change')
                    return True
                with self.subTest(arm=arm,raises=raises),self.assertRaises(runner.NativePreparationMutation):
                    runner.prepare_native_recipe_inputs(self.path,cache=cache,cancel=mutate,**arm)
                self.assertNotIn('reference.prepared',cache.values)
        raw=self.path.read_bytes()
        for raises in (False,True):
            def change_source():
                self.path.write_bytes(raw+b'changed')
                if raises:raise ValueError('Caller after source change')
                return True
            try:
                with self.assertRaises(runner.NativePreparationMutation):
                    runner.prepare_native_recipe_inputs(self.path,cancel=change_source)
            finally:self.path.write_bytes(raw)

    def test_private_cpu_pixels_headers_and_cuda_buffers_are_guarded_for_every_arm(self):
        original_decode=cv2.imdecode;original_tensor=torch.as_tensor;original_empty=torch.empty
        for arm in ARMS:
            for target in ('cpu_pixels','cpu_header','bgr','gray'):
                for raises in (False,True):
                    captured=[];fired=False;cache=PipelineAnalysisCache()
                    def decode(*args,**kwargs):
                        value=original_decode(*args,**kwargs)
                        if target.startswith('cpu'):captured.append(value)
                        return value
                    def as_tensor(*args,**kwargs):
                        value=original_tensor(*args,**kwargs)
                        if target=='bgr':captured.append(value)
                        return value
                    def empty(*args,**kwargs):
                        value=original_empty(*args,**kwargs)
                        if target=='gray':captured.append(value)
                        return value
                    def mutate():
                        nonlocal fired
                        if captured and not fired:
                            fired=True
                            if target=='cpu_header':captured[0].shape=(17,13,3)
                            elif target=='cpu_pixels':captured[0][0,0,0]^=1
                            else:captured[0].add_(1)
                            if raises:raise ValueError('Caller after private mutation')
                            return True
                        return False
                    with self.subTest(arm=arm,target=target,raises=raises), \
                         patch.object(cv2,'imdecode',decode),patch.object(torch,'as_tensor',as_tensor), \
                         patch.object(torch,'empty',empty):
                        with self.assertRaises(runner.NativePreparationMutation):
                            runner.prepare_native_recipe_inputs(self.path,cache=cache,cancel=mutate,**arm)
                    self.assertTrue(fired);self.assertNotIn('reference.prepared',cache.values)

    def test_admission_before_upload_and_final_source_audit(self):
        with patch.object(runner,'MAX_RASTER_PIXELS',self.bgr.shape[0]*self.bgr.shape[1]-1), \
             patch.object(torch,'as_tensor',side_effect=AssertionError('No CUDA upload')):
            with self.assertRaisesRegex(runner.NativeRecipeError,'raster limit'):
                runner.prepare_native_recipe_inputs(self.path)
        original=runner.NativeRecipeInputs;captured=[]
        def capture(*args,**kwargs):
            value=original(*args,**kwargs);captured.append(value);return value
        raw=self.path.read_bytes()
        def replace_source():
            if captured:self.path.write_bytes(raw+b'changed')
            return False
        try:
            with patch.object(runner,'NativeRecipeInputs',capture):
                with self.assertRaises(runner.NativePreparationMutation):
                    runner.prepare_native_recipe_inputs(self.path,cancel=replace_source)
        finally:self.path.write_bytes(raw)

    def test_initial_middle_final_cancellation_and_cached_metadata_tensor_binding_guards(self):
        calls=0
        def count():
            nonlocal calls
            calls+=1;return False
        runner.prepare_native_recipe_inputs(self.path,cancel=count)
        total=calls
        for boundary in (1,max(2,total//2),total):
            calls=0;cache=PipelineAnalysisCache()
            def cancel():
                nonlocal calls
                calls+=1;return calls==boundary
            with self.subTest(boundary=boundary),self.assertRaises(runner.NativeRecipeCancelled):
                runner.prepare_native_recipe_inputs(self.path,cache=cache,cancel=cancel)
            self.assertNotIn('reference.prepared',cache.values)
        for target in ('gray','rgb','source_sha256','source_path','preparation','coordinate_frame','binding'):
            for raises in (False,True):
                cache=PipelineAnalysisCache()
                prepared=runner.prepare_native_recipe_inputs(self.path,cache=cache)
                def mutate():
                    if target in ('gray','rgb'):getattr(prepared,target).add_(1)
                    elif target=='source_path':object.__setattr__(prepared,target,self.path.with_name('changed.png'))
                    elif target=='source_sha256':object.__setattr__(prepared,target,'f'*64)
                    elif target=='preparation':object.__setattr__(prepared,target,{'gray':'changed','rgb':'changed'})
                    elif target=='coordinate_frame':
                        object.__setattr__(prepared,target,replace(prepared.coordinate_frame,origin='changed'))
                    else:cache.values.pop('reference.prepared')
                    if raises:raise ValueError('Caller after cached mutation')
                    return False
                with self.subTest(target=target,raises=raises),self.assertRaises(runner.NativePreparationMutation):
                    runner.prepare_native_recipe_inputs(self.path,cache=cache,cancel=mutate)
                self.assertNotIn('reference.prepared',cache.values)

    def test_path_entry_uses_strict_shared_guards_before_recipe_model_selection(self):
        for stopped in (None,0,1,np.bool_(False)):
            with self.subTest(value=repr(stopped)),self.assertRaises(TypeError):
                runner.run_native_recipe_path(None,self.path,cancel=lambda:stopped)
        for target in ('gray','metadata','binding','source'):
            cache=PipelineAnalysisCache()
            prepared=runner.prepare_native_recipe_inputs(self.path,cache=cache)
            raw=self.path.read_bytes()
            def mutate():
                if target=='gray':prepared.gray.add_(1)
                elif target=='metadata':object.__setattr__(prepared,'source_sha256','f'*64)
                elif target=='binding':cache.values.pop('reference.prepared')
                else:self.path.write_bytes(raw+b'changed')
                raise ValueError('Caller after path-entry mutation')
            try:
                with self.subTest(target=target),self.assertRaises(runner.NativePreparationMutation):
                    runner.run_native_recipe_path(None,self.path,cache=cache,cancel=mutate)
                self.assertNotIn('reference.prepared',cache.values)
            finally:self.path.write_bytes(raw)
        failure=ValueError('Original path-entry caller exception')
        def raises():raise failure
        with self.assertRaises(ValueError) as caught:
            runner.run_native_recipe_path(None,self.path,cancel=raises)
        self.assertIs(caught.exception,failure)

    def test_loaded_helper_changes_withhold_and_retry_recovers(self):
        original=inputs.decode;cache=PipelineAnalysisCache();fired=False
        def changed(raw):return original(raw)
        def callback():
            nonlocal fired
            if not fired:
                fired=True;inputs.decode=changed
                raise ValueError('Caller after helper change')
            return False
        try:
            with self.assertRaises(runner.NativePreparationMutation):
                runner.prepare_native_recipe_inputs(self.path,cache=cache,cancel=callback)
        finally:inputs.decode=original
        self.assertNotIn('reference.prepared',cache.values)
        self.assertEqual(runner.prepare_native_recipe_inputs(self.path,cache=cache).gray.device.type,'cuda')
