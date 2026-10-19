"""Public rectangle-helper equality and integrated native ROI morphology."""

from tests.functional_units import functional_units
from dataclasses import replace
import hashlib
import inspect
import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from tests.catalogue_fixtures import CatalogueTestCase
from unittest.mock import patch

import cv2
import numpy as np
import torch

from seedvision.benchmarking import native_recipe as runner
from seedvision.benchmarking import native_morphology_roi_recipe as factory
from seedvision.benchmarking import native_stage_replay as replay
from seedvision.benchmarking.recipes import RecipeSpec
from seedvision.cuda import plantcv_morphology as primitive
from seedvision.cuda.tensor_callback import TensorCallbackMutation
from seedvision.export import native_results as archives
from seedvision.segmentation.baseline import PipelineAnalysisCache
from tests.test_native_recipe import fixture, selected_recipe, FRAME, SOURCE, PREPARATION, REFERENCE

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')


def _change(recipe, operation, **values):
    return replace(recipe, stages=tuple(replace(s, parameters=tuple(
        replace(p, value=values.get(p.parameter_id, p.value)) for p in s.parameters))
        if s.operation_id == operation else s for s in recipe.stages))


def _composition_orders(base, controls, add):
    """Exercise every order, constructing each identical immutable prefix once."""
    recipes = {(): base}
    snapshots = {(): base.to_json()}
    for order in itertools.permutations(controls):
        for length in range(1, len(order) + 1):
            prefix = order[:length]
            if prefix not in recipes:
                recipes[prefix] = add(recipes[prefix[:-1]], prefix[-1])
                snapshots[prefix] = recipes[prefix].to_json()
        yield order, recipes[order]
    for prefix, recipe in recipes.items():
        if recipe.to_json() != snapshots[prefix]:
            raise AssertionError(f'Composition mutated shared input {prefix!r}')


def _recipe():
    return factory.build_native_morphology_roi_recipe(runner.build_native_seed_recipe(selected_recipe()))


REFERENCE_SCRIPT = r'''
import hashlib, importlib, json, sys
from importlib.metadata import version
from pathlib import Path
import numpy as np
from plantcv import plantcv as pcv
from plantcv.plantcv.classes import Objects
assert version('plantcv') == '4.11.3'
assert version('opencv-python') == '4.11.0.86'
helper=importlib.import_module('plantcv.plantcv._helpers')
assert hashlib.sha256(Path(helper.__file__).read_bytes()).hexdigest() == 'c05b0c944210657332ce307f2066caab55f2ebc8d8fea11988db3e476e7cdfba'
for name,digest in [('erode','6ae07bc734ae7226b9f39f5b970d549319e49b45ca9559b0ae3064e02a67487a'),('dilate','e0bf2fcf8ca96e8d7aa5cb05cf3b7e4dbdb163450466126bc8fb124549f2472f')]:
 module=importlib.import_module('plantcv.plantcv.'+name)
 assert hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()==digest
pcv.params.debug=None
source=np.load(sys.argv[1],allow_pickle=False)['gray']
cases=json.loads(Path(sys.argv[2]).read_text())
results={}
for index,case in enumerate(cases):
 bounds=case['roi'];roi=None
 if bounds is not None:
  x0,y0,x1,y1=bounds
  contour=[np.array([[[x0,y0]],[[x0,y1]],[[x1,y1]],[[x1,y0]]],np.int32)]
  hierarchy=np.array([[[-1,-1,-1,-1]]],np.int32)
  roi=Objects(contours=[contour],hierarchy=[hierarchy])
 output=getattr(pcv,case['operation'])(gray_img=source,ksize=case['ksize'],i=case['iterations'],roi=roi)
 results[str(index)]=output
 # Public rectangle uses x+w-1/y+h-1 as the helper's exclusive endpoints.
 if bounds is not None and x1<source.shape[1] and y1<source.shape[0]:
  rectangle=pcv.roi.rectangle(source,x=x0,y=y0,w=x1-x0+1,h=y1-y0+1)
  check=getattr(pcv,case['operation'])(gray_img=source,ksize=case['ksize'],i=case['iterations'],roi=rectangle)
  assert np.array_equal(output,check)
np.savez(sys.argv[3],**results)
'''


@functional_units
class MorphologyROIContractTests(CatalogueTestCase):
    def check_test_historical_mask_identity_tracks_shared_callback_implementation(self):
        from seedvision.benchmarking import native_mask_stage as bridge
        from seedvision.cuda import tensor_callback
        bridge._implementation_digest()
        with patch.object(tensor_callback, 'checked_cancel', lambda *args, **kwargs: False):
            with self.assertRaises(bridge.NativeMaskStageError):
                bridge._implementation_digest()

    def check_test_all_fourteen_variants_roundtrip_without_changing_base_definitions(self):
        for base_id in factory.BASE_IDS:
            with self.subTest(base=base_id):
                base = factory.canonical_base(base_id)
                before = base.to_dict()
                recipe = factory.build_native_morphology_roi_recipe(base)
                runner._validate_recipe(recipe)
                restored = RecipeSpec.from_json(recipe.to_json())
                self.assertEqual(restored.identity, recipe.identity)
                self.assertEqual(base.to_dict(), before)
                self.assertEqual(runner._diagnostic_recipe.base_recipe_id(recipe.recipe_id),
                                 runner._diagnostic_recipe.base_recipe_id(base_id))
                for stage, prior in zip(recipe.stages, base.stages, strict=True):
                    if stage.operation_id in factory.OPERATIONS:
                        self.assertEqual(stage.parameters[:-1], prior.parameters)
                        self.assertIsNone(stage.parameters[-1].value)
                    else:
                        self.assertEqual(stage, prior)

    def check_test_bounds_and_full_canvas_replay_admission_remain_strict(self):
        parameters = {'ksize': 4, 'iterations': 2, factory.PARAMETER: [2, 3, 12, 13]}
        plan = replay._plan((20, 30), 'morphology.erode', parameters, 1200)
        self.assertEqual(plan['work_details']['roi_pixels'], 100)
        self.assertEqual(plan['work'], 100*4*2*2+600)
        for value in ([0, 0, 31, 20], [0, 0, 0, 1], [True, 0, 1, 1], [0., 0, 1, 1], '0,0,1,1'):
            with self.subTest(value=value), self.assertRaises((TypeError, ValueError)):
                replay._plan((20, 30), 'morphology.erode', dict(parameters, roi_bounds_xyxy=value), 1200)
        with self.assertRaises(runner.NativeRecipeError):
            replay._plan((1, 7_340_033), 'morphology.erode',
                         dict(parameters, roi_bounds_xyxy=[0, 0, 1, 1]), 14_680_066)

    def test_workflow_lifecycle(self):
        """3 related MorphologyROIContractTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_historical_mask_identity_tracks_shared_callback_implementation)
        self.scenario(self.check_test_all_fourteen_variants_roundtrip_without_changing_base_definitions)
        self.scenario(self.check_test_bounds_and_full_canvas_replay_admission_remain_strict)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
@functional_units
class MorphologyROIIntegrationTests(CatalogueTestCase):

    def check_test_crop_boundary_is_independent_and_real_controls_change_pixels(self):
        source = torch.zeros((40, 40), dtype=torch.uint8, device='cuda')
        source[12:28, 12:28] = 255
        roi = (12, 12, 28, 28)
        crop_first = primitive.erode(source, 4, 1, roi=roi)
        full_first = primitive.erode(source, 4, 1)
        self.assertFalse(torch.equal(crop_first[12:28, 12:28], full_first[12:28, 12:28]))
        roi = (16, 16, 32, 32)
        outputs = (primitive.erode(source, 2, 1, roi=roi), primitive.erode(source, 4, 1, roi=roi),
                   primitive.erode(source, 4, 2, roi=roi), primitive.dilate(source, 4, 1, roi=roi))
        self.assertEqual(len({hashlib.sha256(t.cpu().numpy().tobytes()).hexdigest() for t in outputs}), 4)

    def check_test_invalid_bounds_and_cancellation_never_publish_or_change_input(self):
        source = torch.arange(24, device='cuda', dtype=torch.uint8).reshape(4, 6)
        before = source.clone()
        for roi in ((0, 0, 7, 4), (0, 0, 0, 1), (True, 0, 1, 1), [0, 0, 1, 1]):
            with self.subTest(roi=roi), patch.object(primitive, '_axis_max', side_effect=AssertionError('allocated work')):
                with self.assertRaises((TypeError, ValueError)):
                    primitive.erode(source, 3, roi=roi)
        polls = 0
        def cancel():
            nonlocal polls
            polls += 1
            return polls >= 3
        with self.assertRaises(primitive.MorphologyCancelledError):
            primitive.dilate(source, 2, 4, roi=(1, 1, 5, 4), cancel=cancel)
        with self.assertRaises(TypeError):
            primitive.dilate(source, 2, cancel=lambda: 1)
        torch.testing.assert_close(source, before)

    def check_test_callback_mutation_of_original_private_buffers_or_storage_withholds_output(self):
        for target, poll in (('gray', 1), ('values', 2), ('result', 3), ('output', 4)):
            for outcome in ('false', 'true', 'raise'):
                with self.subTest(target=target, outcome=outcome):
                    source = torch.ones((8, 10), dtype=torch.uint8, device='cuda')
                    polls = 0
                    def cancel():
                        nonlocal polls
                        polls += 1
                        if polls != poll:
                            return False
                        frame = inspect.currentframe().f_back
                        while frame.f_code is not primitive._morphology.__code__:
                            frame = frame.f_back
                        frame.f_locals[target].fill_(7)
                        if outcome == 'raise':
                            raise LookupError('caller callback failed')
                        return outcome == 'true'
                    with self.assertRaises(TensorCallbackMutation):
                        primitive.dilate(source, 2, roi=(1, 1, 9, 7), cancel=cancel)
        source = torch.ones((8, 10), dtype=torch.uint8, device='cuda')
        def resize_storage():
            storage = source.untyped_storage()
            storage.resize_(storage.nbytes()+1)
            raise LookupError('caller callback failed after resizing storage')
        with self.assertRaises(TensorCallbackMutation):
            primitive.erode(source, 2, cancel=resize_storage)
        def unchanged_error():
            raise LookupError('unchanged caller error')
        with self.assertRaisesRegex(LookupError, 'unchanged caller error'):
            primitive.erode(source, 2, cancel=unchanged_error)





    def test_validation_and_recovery(self):
        """3 related MorphologyROIIntegrationTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_crop_boundary_is_independent_and_real_controls_change_pixels)
        self.scenario(self.check_test_invalid_bounds_and_cancellation_never_publish_or_change_input)
        self.scenario(self.check_test_callback_mutation_of_original_private_buffers_or_storage_withholds_output)


if __name__ == '__main__':
    unittest.main()
