
from tests.catalogue_fixtures import CatalogueTestCase
from dataclasses import replace
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch

from seedvision.benchmarking import native_recipe as runner
from seedvision.benchmarking import native_watershed_diagnostic_recipe as factory
from seedvision.benchmarking.recipes import RecipeSpec
from seedvision.benchmarking.native_stage_replay import replay_native_stage
from seedvision.export import native_results as archives
from seedvision.persistence.native_recipes import native_recipes_from_payload, native_recipes_to_payload
from seedvision.segmentation.baseline import PipelineAnalysisCache
from tests import test_native_recipe as seeds
from tests import test_native_calibrated_seed_recipe as calibrated
from tests.test_plantcv_watershed_diagnostic import oracle


def selected(kind='binary', local=False):
    base = calibrated.selected_recipe(kind) if local else runner.build_native_seed_recipe(seeds.selected_recipe(kind))
    return factory.build_native_watershed_diagnostic_recipe(base)


def controls(recipe, **values):
    return replace(recipe,stages=tuple(replace(stage,parameters=tuple(
        replace(p,value=values.get(p.parameter_id,p.value)) for p in stage.parameters))
        if stage.stage_id == factory.STAGE else stage for stage in recipe.stages))


class WatershedDiagnosticRecipeContractTests(CatalogueTestCase):

    def test_joint_admission_is_explicit_and_full_committed_frame_fits(self):
        for local in (False,True):
            bound = runner._diagnostic_retained_bound(2080*3120,calibrated=local)
            self.assertLessEqual(bound,runner.MAX_RETAINED_CUDA_BYTES)
            with self.assertRaises(runner.NativeRecipeError):
                runner._diagnostic_retained_bound(16_000_001,calibrated=local)


@unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
class WatershedDiagnosticRecipeCalculationTests(CatalogueTestCase):

    def run_fixture(self, recipe, cache=None):
        if factory.base_recipe_id(recipe.recipe_id).startswith('plantcv_calibrated_'):
            bgr = calibrated.fixture()
            rgb = bgr[...,::-1].copy()
            gray = cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
            frame = replace(seeds.FRAME,shape=gray.shape)
        else:
            gray,rgb = seeds.fixture()
            frame = seeds.FRAME
        return runner.run_native_recipe(recipe,torch.as_tensor(gray,device='cuda'),
            torch.as_tensor(rgb,device='cuda'),source_sha256=seeds.SOURCE,
            coordinate_frame=frame,preparation=seeds.PREPARATION,cache=cache)

    def test_all_six_actual_graphs_draw_exact_public_pixels_in_their_source_or_roi_frame(self):
        cases,actuals = [],[]
        for kind in ('binary','mean','otsu'):
            for local in (False,True):
                recipe = selected(kind,local)
                result = self.run_fixture(recipe)
                diagnostic = result.node(factory.STAGE)
                self.assertEqual(diagnostic.status,'executed',diagnostic.reason)
                image_edge = next(e for e in recipe.dependencies if e.target_stage==factory.STAGE and e.target_port=='image')
                image = result.node(image_edge.source_stage).outputs[image_edge.source_port]
                cases.append(dict(rgb=image.cpu().tolist(),labels=result.labels.cpu().tolist(),
                    color_scale='gist_rainbow',random_seed=0,line_thickness=5))
                actuals.append(diagnostic.outputs['diagnostic'].cpu().numpy())
                self.assertEqual(tuple(diagnostic.outputs['diagnostic'].shape[:2]),result.labels_coordinate_frame.shape)
                flood = next(n for n in result.nodes if n.operation_id=='reference.internal.watershed.masked_flood')
                self.assertEqual(set(flood.outputs),{'labels','flood_complete'})
        for actual,expected in zip(actuals,oracle('overlay',cases)):
            np.testing.assert_array_equal(actual,expected['image'])




    def run_fixture_with_cancel(self,recipe):
        gray,rgb = seeds.fixture()
        return runner.run_native_recipe(recipe,torch.tensor(gray,device='cuda'),torch.tensor(rgb,device='cuda'),
            source_sha256=seeds.SOURCE,coordinate_frame=seeds.FRAME,
            preparation=seeds.PREPARATION,cancel=lambda:True)


@unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
class WatershedDiagnosticDesktopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])
