
from tests.catalogue_fixtures import CatalogueTestCase

from tests.functional_units import functional_units
from dataclasses import replace
import unittest

import cv2
import numpy as np
import torch

from seedvision.benchmarking import native_recipe as native
from seedvision.benchmarking.recipe_catalogue import build_recipe_catalogue
from seedvision.benchmarking.recipes import RecipeSpec
from seedvision.segmentation.baseline import PipelineAnalysisCache
from tests.test_native_recipe import selected_recipe, fixture, SOURCE, FRAME, PREPARATION, PREFIX
from tests.test_plantcv_size import _pinned_public_size
from tests.test_plantcv_size_diagnostic import reference, _SIZE_IMAGE_SCRIPT


def with_thickness(recipe, thickness):
    return replace(recipe, stages=tuple(replace(stage, parameters=tuple(
        replace(parameter, value=thickness) if parameter.parameter_id == 'line_thickness'
        else parameter for parameter in stage.parameters))
        if stage.operation_id == 'analysis.size' else stage for stage in recipe.stages))


class NativeWorkingSeedContractTests(CatalogueTestCase):
    def test_working_copy_adds_only_effective_size_control_and_preserves_history(self):
        catalogue = build_recipe_catalogue()
        original = catalogue.to_json()
        for kind in ('binary','mean','otsu'):
            selected = selected_recipe(kind)
            recipe = native.build_native_seed_recipe(selected)
            self.assertEqual(recipe.recipe_id, selected.recipe_id+'_working')
            self.assertEqual(RecipeSpec.from_json(recipe.to_json()),recipe)
            self.assertEqual(recipe.dependencies,selected.dependencies)
            self.assertEqual(recipe.layout,selected.layout)
            for old,new in zip(selected.stages,recipe.stages):
                if old.operation_id != 'analysis.size':
                    self.assertEqual(old,new)
                else:
                    self.assertEqual(old.parameters,new.parameters[:-1])
                    parameter = new.parameters[-1]
                    self.assertEqual((parameter.parameter_id,parameter.value),('line_thickness',5))
                    self.assertTrue(parameter.editable and parameter.affects_calculation)
            native._validate_recipe(with_thickness(recipe,1))
        self.assertEqual(build_recipe_catalogue().to_json(),original)


@unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
@functional_units
class NativeWorkingSeedCalculationTests(CatalogueTestCase):


    pass

    def check_test_changed_thickness_recomputes_only_size_and_preserves_cuda_upstream_products(self):
        gray,rgb = fixture()
        gray,rgb = torch.as_tensor(gray,device='cuda'),torch.as_tensor(rgb,device='cuda')
        recipe = native.build_native_seed_recipe(selected_recipe())
        cache = PipelineAnalysisCache()
        arguments = dict(source_sha256=SOURCE,coordinate_frame=FRAME,preparation=PREPARATION,cache=cache)
        first = native.run_native_recipe(recipe,gray,rgb,**arguments)
        changed = native.run_native_recipe(with_thickness(recipe,1),gray,rgb,**arguments)
        stage_id = PREFIX+'analyze_size'
        self.assertEqual(first.status,'executed')
        self.assertEqual(changed.status,'executed')
        self.assertEqual(cache.last_computed_nodes,(stage_id,))
        self.assertEqual(len(cache.last_reused_nodes),len(recipe.stages)-1)
        self.assertEqual(changed.node(stage_id).parameters['line_thickness'],1)
        self.assertIs(first.labels,changed.labels)
        self.assertFalse(torch.equal(first.node(stage_id).outputs['diagnostic'],
            changed.node(stage_id).outputs['diagnostic']))

    def test_additional_numerical_regressions(self):
        self.scenario(self.check_test_changed_thickness_recomputes_only_size_and_preserves_cuda_upstream_products)
