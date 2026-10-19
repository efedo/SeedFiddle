"""Pinned public reader semantics and shared native preparation boundaries."""

from tests.catalogue_fixtures import CatalogueTestCase

from tests.functional_units import functional_units
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
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


class ImageInputContractTests(CatalogueTestCase):
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

@unittest.skipUnless(torch.cuda.is_available(),'CUDA preparation is required')
@functional_units
class NativePreparationBoundaryTests(CatalogueTestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory()
        self.path=Path(self.temporary.name)/'source.png'
        self.bgr=np.arange(13*17*3,dtype=np.uint8).reshape(13,17,3)
        self.assertTrue(cv2.imwrite(str(self.path),self.bgr))

    def tearDown(self):
        self.temporary.cleanup()

    def check_test_cache_reuses_pixels_repairs_detached_facts_and_equal_size_replacement(self):
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

    def check_test_loaded_helper_changes_withhold_and_retry_recovers(self):
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

    def test_additional_numerical_regressions(self):
        self.scenario(self.check_test_cache_reuses_pixels_repairs_detached_facts_and_equal_size_replacement)
        self.scenario(self.check_test_loaded_helper_changes_withhold_and_retry_recovers)
