"""Automatic-card working graph, CUDA cache boundaries and saved outputs."""

from tests.catalogue_fixtures import CatalogueTestCase

from tests.functional_units import functional_units
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch

from seedvision.benchmarking import native_card_recipe as factory, native_recipe as runner
from seedvision.benchmarking.checkpoints import CoordinateFrame
from seedvision.benchmarking.recipes import LayoutPosition
from seedvision.segmentation.baseline import PipelineAnalysisCache
from tests.test_native_kmeans_recipe import _parameter
from tests import test_plantcv_card_detection as card_tests

fixture = card_tests.fixture


class CardContractTests(CatalogueTestCase):
    def test_additive_recipe_contract_and_retained_bound(self):
        from seedvision.benchmarking.recipe_catalogue import build_recipe_catalogue
        before = build_recipe_catalogue().to_dict()
        recipe = factory.build_native_card_recipe()
        runner._validate_recipe(recipe)
        self.assertEqual(len(recipe.stages), 9)
        self.assertEqual(recipe.parameters_for_runner(factory.STANDARD)['pos'], 3)
        self.assertEqual(build_recipe_catalogue().to_dict(), before)
        maximum = (1024*1024*1024-256*1024*1024)//32
        self.assertEqual(factory.retained_bound(maximum), 1024*1024*1024)
        with self.assertRaises(factory.card.AutomaticCardResourceError):
            factory.retained_bound(maximum+1)
        for module in factory.implementation_modules():
            runner._module_record(module)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
@functional_units
class NativeCardRecipeTests(CatalogueTestCase):
    oracle = card_tests.AutomaticCardTests.oracle

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.image = fixture(varied=True)
        self.path = self.root/'card.png'
        self.assertTrue(cv2.imwrite(str(self.path), self.image))
        self.recipe = factory.build_native_card_recipe()
        self.cache = PipelineAnalysisCache()

    def tearDown(self):
        self.temp.cleanup()

    def run_recipe(self, recipe=None, **kwargs):
        return runner.run_native_recipe_path(recipe or self.recipe, self.path, cache=self.cache, **kwargs)

    def check_test_actual_public_auto_and_roi_default_position(self):
        for bounds in (None, [40,40,420,580]):
            recipe = _parameter(self.recipe, factory.PREPROCESS, 'roi_bounds_xyxy', bounds)
            actual = self.run_recipe(recipe)
            self.assertEqual(actual.status, 'executed', [(n.stage_id,n.reason) for n in actual.nodes])
            expected, _ = self.oracle(self.image, operation='auto', roi_bounds_xyxy=bounds)
            mask = actual.node(factory.MASK).outputs['labeled_mask']
            np.testing.assert_array_equal(mask.cpu().numpy(), expected['mask'])
            np.testing.assert_array_equal(actual.node(factory.MASK).outputs['bounding_mask'].cpu(),expected['bbox'])
            np.testing.assert_allclose(actual.node(factory.MEANS).outputs['source_matrix'],expected['source'],atol=2e-15,rtol=0)
            np.testing.assert_array_equal(actual.node(factory.STANDARD).outputs['target_matrix'],expected['target'])
            corrected = actual.node(factory.AFFINE).outputs['corrected_rgb']
            difference = np.abs(corrected.cpu().numpy().astype(int)-expected['corrected'][...,::-1].astype(int))
            self.assertLessEqual(int(difference.max()),1)
            self.assertEqual(corrected.device.type,'cuda')
            self.assertEqual(mask.dtype,torch.float64)
            geometry = actual.node(factory.GEOMETRY).outputs['geometry']
            np.testing.assert_array_equal(geometry['centers_xy'],expected['centers'])
            np.testing.assert_array_equal(geometry['corners_xy'],expected['corners'])
            with self.assertRaises(TypeError): geometry['detected_chips']=0
            self.assertIsInstance(actual.node(factory.MEANS).outputs['source_matrix'],tuple)

    def check_test_controls_recompute_only_their_dependents(self):
        first = self.run_recipe()
        self.assertTrue(all(node.reused for node in self.run_recipe().nodes))
        cases = (
            (factory.MASK,'radius',22,{factory.MASK,factory.MEANS,factory.AFFINE}),
            (factory.SCALE,'color_chip_size',[7,11],{factory.SCALE}),
            (factory.STANDARD,'pos',2,{factory.STANDARD,factory.AFFINE}),
            (factory.GEOMETRY,'min_size',1200,{factory.GEOMETRY,factory.MASK,factory.SCALE,factory.MEANS,factory.AFFINE}),
            (factory.GEOMETRY,'aspect_ratio',1.4,{factory.GEOMETRY,factory.MASK,factory.SCALE,factory.MEANS,factory.AFFINE}),
            (factory.GEOMETRY,'solidity',.75,{factory.GEOMETRY,factory.MASK,factory.SCALE,factory.MEANS,factory.AFFINE}),
            (factory.PREPROCESS,'block_size',35,{factory.PREPROCESS,factory.GEOMETRY,factory.MASK,factory.SCALE,factory.MEANS,factory.AFFINE}),
            (factory.PREPROCESS,'adaptive_method',0,{factory.PREPROCESS,factory.GEOMETRY,factory.MASK,factory.SCALE,factory.MEANS,factory.AFFINE}),
            (factory.PREPROCESS,'roi_bounds_xyxy',[40,40,420,580],{factory.PREPROCESS,factory.GEOMETRY,factory.MASK,factory.SCALE,factory.MEANS,factory.AFFINE}),
        )
        selected = self.recipe
        for stage,name,value,computed in cases:
            selected = _parameter(selected,stage,name,value)
            result = self.run_recipe(selected)
            self.assertEqual(result.status,'executed', [(n.stage_id,n.reason) for n in result.nodes])
            self.assertEqual(set(self.cache.last_computed_nodes),computed)
            if stage == factory.MASK:
                self.assertFalse(torch.equal(first.node(factory.MASK).outputs['labeled_mask'],result.node(factory.MASK).outputs['labeled_mask']))
            if stage == factory.SCALE:
                self.assertEqual(result.node(factory.SCALE).outputs['metadata']['unit'],'mm')
        layout = replace(selected, title='Working card', layout=tuple(LayoutPosition(s.stage_id,100.+i*200,80.) for i,s in enumerate(selected.stages)))
        self.assertTrue(all(node.reused for node in self.run_recipe(layout).nodes))





    def check_test_loaded_geometry_and_fixed_metadata_identity_checked(self):
        with patch.object(factory.minrect,'_no_guard',lambda: False):
            with self.assertRaises(ValueError): self.run_recipe()
        changed = factory.card.GRID_BOX_411.copy(); changed.setflags(write=False)
        with patch.object(factory.card,'GRID_BOX_411',changed):
            # A byte-identical immutable replacement is permitted; changed bytes are not.
            factory.metadata_identity()
        changed = factory.card.GRID_BOX_411.copy();changed[0,0]=1;changed.setflags(write=False)
        with patch.object(factory.card,'GRID_BOX_411',changed):
            with self.assertRaisesRegex(ValueError,'Fixed grid'): self.run_recipe()

    def check_test_workspace_rejection_precedes_cuda_upload(self):
        with patch.object(factory,'retained_bound',side_effect=factory.card.AutomaticCardResourceError('fixture cap')):
            with patch.object(torch,'as_tensor',side_effect=AssertionError('must not allocate')):
                with self.assertRaisesRegex(factory.card.AutomaticCardResourceError,'fixture cap'):
                    self.run_recipe()



    def test_workflow_lifecycle(self):
        """3 related NativeCardRecipeTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_actual_public_auto_and_roi_default_position)
        self.scenario(self.check_test_controls_recompute_only_their_dependents)
        self.scenario(self.check_test_loaded_geometry_and_fixed_metadata_identity_checked)

    def test_validation_and_recovery(self):
        """1 related NativeCardRecipeTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_workspace_rejection_precedes_cuda_upload)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class NativeCardDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def wait(self, dialog):
        deadline = time.monotonic()+30
        while dialog._worker is not None and time.monotonic()<deadline:
            self.app.processEvents();time.sleep(.01)
        self.app.processEvents();self.assertIsNone(dialog._worker)
