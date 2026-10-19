
from tests.catalogue_fixtures import CatalogueTestCase
from dataclasses import replace
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from seedvision.benchmarking.checkpoints import CoordinateFrame
from seedvision.benchmarking.native_correction_recipe import (
    NativeCorrectionCancelled, NativeCorrectionError, run_native_correction_recipe,
)
from seedvision.benchmarking.recipe_catalogue import build_recipe_catalogue
from seedvision.segmentation.baseline import PipelineAnalysisCache


_ROOT = Path(__file__).resolve().parents[1]
_REFERENCE = Path(os.environ.get('SEEDVISION_PLANTCV_REFERENCE_PYTHON',
    str(_ROOT/'artifacts'/'plantcv-reference-env'/'Scripts'/'python.exe')))
_SCRIPT = r"""
import sys, tempfile, importlib, hashlib
from importlib.metadata import version
from pathlib import Path
import cv2
import numpy as np
from plantcv import plantcv as pcv
assert version('plantcv') == '4.11.3'
assert version('numpy') == '2.2.6'
assert version('opencv-python') == '4.11.0.86'
module = importlib.import_module('plantcv.plantcv.transform.color_correction')
assert hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() == 'c0c3cf636db7a881aef820743501614fc3dccd0c1b732ac7df5d54a7607cce68'
pcv.params.debug = None
values = np.load(sys.argv[1], allow_pickle=False)
target_bgr = cv2.cvtColor(values['target_rgb'], cv2.COLOR_RGB2BGR)
source_bgr = cv2.cvtColor(values['source_rgb'], cv2.COLOR_RGB2BGR)
with tempfile.TemporaryDirectory() as output:
    target, source, matrix, corrected = pcv.transform.correct_color(
        target_bgr, values['target_mask'], source_bgr, values['source_mask'], output)
np.savez(sys.argv[2], target_matrix=target, source_matrix=source,
    transformation_matrix=matrix, corrected_rgb=cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB))
"""
_TARGET_SHA, _SOURCE_SHA = 'a'*64, 'b'*64
_TARGET_FRAME = CoordinateFrame('target.image', (17, 25), 'top-left', 'x-right,y-down', 'pixel')
_SOURCE_FRAME = CoordinateFrame('source.image', (17, 25), 'top-left', 'x-right,y-down', 'pixel')
_RECIPE = build_recipe_catalogue().recipe('plantcv_color_card_correct_color')
_STAGE = 'pcv_color_correct_color'


def _fixture():
    rng = np.random.default_rng(309)
    source = rng.integers(20, 235, (17, 25, 3), dtype=np.uint8)
    target = np.clip(source.astype(np.int16)+[9, -7, 5], 0, 255).astype(np.uint8)
    mask = np.zeros((17, 25), dtype=np.uint8)
    for index in range(24):
        row, column = divmod(index, 6)
        ys, xs = slice(1+row*4, 4+row*4), slice(1+column*4, 4+column*4)
        source[ys, xs] = rng.integers(20, 235, 3, dtype=np.uint8)
        target[ys, xs] = np.clip(source[ys, xs].astype(np.int16)+[9, -7, 5], 0, 255)
        mask[ys, xs] = (index+1)*10
    return target, mask.copy(), source, mask.copy()


def _cuda_inputs():
    return tuple(torch.as_tensor(value.copy(), device='cuda') for value in _fixture())


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA is required for native correction recipes')
class NativeCorrectionRecipeTests(CatalogueTestCase):
    def reference(self, arrays):
        self.assertTrue(_REFERENCE.is_file(), 'Pinned PlantCV interpreter is required.')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            np.savez(root/'input.npz', **dict(zip(
                ('target_rgb', 'target_mask', 'source_rgb', 'source_mask'), arrays)))
            completed = subprocess.run([str(_REFERENCE), '-c', _SCRIPT,
                str(root/'input.npz'), str(root/'output.npz')],
                capture_output=True, text=True, timeout=90)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            with np.load(root/'output.npz', allow_pickle=False) as output:
                return {name: output[name].copy() for name in output.files}

    def run_recipe(self, inputs, *, recipe=_RECIPE, cache=None, cancel=None,
            target_source_sha256=_TARGET_SHA, source_sha256=_SOURCE_SHA,
            target_coordinate_frame=_TARGET_FRAME, coordinate_frame=_SOURCE_FRAME):
        return run_native_correction_recipe(recipe, *inputs,
            target_source_sha256=target_source_sha256, source_sha256=source_sha256,
            target_coordinate_frame=target_coordinate_frame, coordinate_frame=coordinate_frame,
            cache=cache, cancel=cancel)





    def test_identity_height_workspace_and_source_contract_fail_before_calculation(self):
        inputs = _cuda_inputs()
        with self.assertRaises(NativeCorrectionError):
            self.run_recipe(inputs, target_source_sha256=None)
        target = torch.cat((inputs[0], inputs[0][:1])).contiguous()
        mask = torch.cat((inputs[1], inputs[1][:1])).contiguous()
        with self.assertRaisesRegex(NativeCorrectionError, 'equal image heights'):
            self.run_recipe((target, mask, inputs[2], inputs[3]),
                target_coordinate_frame=replace(_TARGET_FRAME, shape=(18, 25)))
        with patch('seedvision.benchmarking.native_correction_recipe.MAX_RETAINED_CUDA_BYTES', 32*1024*1024):
            # Isolate workspace preflight: changing an imported runtime limit
            # intentionally fails full API provenance verification first.
            from seedvision.benchmarking.native_correction_recipe import _fit_cache_budget
            with self.assertRaisesRegex(NativeCorrectionError, 'workspace budget'):
                _fit_cache_budget(PipelineAnalysisCache(), 'reference.recipe:test:',
                    inputs[1].numel(), inputs[3].numel())
        ports = tuple(replace(port, data_type='WrongImage') if port.name == 'target_rgb' else port
            for port in _RECIPE.stages[0].ports)
        with self.assertRaises(NativeCorrectionError):
            self.run_recipe(inputs, recipe=replace(_RECIPE, stages=(replace(_RECIPE.stages[0], ports=ports),)))


if __name__ == '__main__':
    unittest.main()
