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


class NativeWorkingSeedContractTests(unittest.TestCase):
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
class NativeWorkingSeedCalculationTests(unittest.TestCase):
    def test_actual_dispatch_both_thicknesses_match_public_measurements_and_pixels(self):
        labels = np.zeros((64,80),np.uint8)
        cv2.fillPoly(labels,[np.asarray([[9,45],[49,9],[57,18],[22,55]],np.int32)],1)
        bgr = np.full((*labels.shape,3),[19,73,121],np.uint8)
        image = torch.as_tensor(bgr[...,::-1].copy(),device='cuda')
        inputs = {'image':image,'labels':torch.as_tensor(labels.astype(np.int32),device='cuda')}
        results = {}
        for thickness in (1,5):
            outputs = native._dispatch('analysis.size',inputs,
                dict(n_labels=1,label='graph',line_thickness=thickness),None)
            expected = _pinned_public_size(labels,line_thickness=thickness)
            actual = outputs['observations']['graph_1']
            self.assertEqual(set(actual),set(expected))
            for variable in expected:
                np.testing.assert_allclose(actual[variable]['value'],expected[variable]['value'],
                    rtol=1e-6,atol=1e-6,err_msg=variable)
            expected_image = reference(_SIZE_IMAGE_SCRIPT,dict(mask=labels.tolist(),
                image=bgr.tolist(),n_labels=1,line_thickness=thickness))
            np.testing.assert_array_equal(outputs['diagnostic'].cpu().numpy(),expected_image[...,::-1])
            results[thickness] = outputs
        self.assertNotEqual(results[1]['observations']['graph_1']['longest_path']['value'],
            results[5]['observations']['graph_1']['longest_path']['value'])
        default = native._dispatch('analysis.size',inputs,dict(n_labels=1,label='graph'),None)
        torch.testing.assert_close(default['diagnostic'],results[5]['diagnostic'])

    def test_changed_thickness_recomputes_only_size_and_preserves_cuda_upstream_products(self):
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

    def test_real_isolated_worker_thickness_one_matches_native_public_contract(self):
        from seedvision.benchmarking.reference_runner import PlantCVReferenceRunner
        from tests.test_reference_runner import REFERENCE_PYTHON
        labels = np.zeros((64,80),np.int32)
        cv2.fillPoly(labels,[np.asarray([[9,45],[49,9],[57,18],[22,55]],np.int32)],1)
        rgb = np.full((*labels.shape,3),[121,73,19],np.uint8)
        expected = PlantCVReferenceRunner(REFERENCE_PYTHON).run('analysis.size',
            {'image':rgb,'labels':labels},{'n_labels':1,'label':'graph'})
        actual = native._dispatch('analysis.size',dict(image=torch.as_tensor(rgb,device='cuda'),
            labels=torch.as_tensor(labels,device='cuda')),dict(n_labels=1,label='graph',line_thickness=1),None)
        np.testing.assert_array_equal(actual['diagnostic'].cpu().numpy(),expected.arrays['analysis_image'])
        records = expected.observations['observations']['graph_1']
        self.assertEqual(len(records),17)
        for variable,record in records.items():
            np.testing.assert_allclose(actual['observations']['graph_1'][variable]['value'],
                record['value'],rtol=1e-6,atol=1e-6,err_msg=variable)
