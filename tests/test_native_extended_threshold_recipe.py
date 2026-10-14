import json
import subprocess
import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import torch

from seedvision.benchmarking.checkpoints import CoordinateFrame
from seedvision.benchmarking.native_extended_threshold_recipes import build_native_extended_threshold_recipes
from seedvision.benchmarking.native_recipe import NativeRecipeCancelled, NativeRecipeError, run_native_recipe
from seedvision.benchmarking.recipe_catalogue import build_recipe_catalogue
from seedvision.benchmarking.recipes import RecipeSpec, RecipeContractError
from seedvision.segmentation.baseline import PipelineAnalysisCache
from seedvision.cuda import plantcv_extended_thresholds as extended
from seedvision.benchmarking import native_extended_threshold_recipes as recipe_factory
from seedvision.benchmarking import native_recipe as native_recipe_module
from tests.test_plantcv_extended_thresholds import REFERENCE, SCRIPT

SOURCE = 'a' * 64
PREPARATION = {
    'gray': 'Explicit synthetic uint8 grayscale boundary for native extended threshold test.',
    'rgb': 'Explicit synthetic RGB boundary; dual-channel adapter converts RGB to BGR.',
}


def _fixture():
    height, width = 31, 37
    yy, xx = np.indices((height, width))
    gray = (35 + ((7*xx + 11*yy + (xx*yy) % 47) % 180)).astype(np.uint8)
    gray[6:25, 8:30] = np.where(
        ((xx[6:25, 8:30]-19)**2 + (yy[6:25, 8:30]-15)**2) < 90, 195, 45)
    rgb = np.stack((40+(5*xx+3*yy) % 200,
        25+(7*yy+xx) % 210, 15+(2*xx+11*yy) % 220), axis=-1).astype(np.uint8)
    triangle = np.full((height, width), 30, dtype=np.uint8)
    triangle[7:24, 9:29] = 180
    return gray, rgb, triangle


def _reference_mask(image, operation, parameters):
    completed = subprocess.run([str(REFERENCE), '-c', SCRIPT],
        input=json.dumps({'image': image.tolist(), 'operation': operation, 'cases': [parameters]}),
        capture_output=True, text=True, timeout=90)
    if completed.returncode:
        raise AssertionError(completed.stderr)
    return np.asarray(json.loads(completed.stdout)[0], dtype=np.uint8)


def _with_parameters(recipe, **updates):
    stage = recipe.stages[1]
    changed = replace(stage, parameters=tuple(
        replace(parameter, value=updates.get(parameter.parameter_id, parameter.value))
        for parameter in stage.parameters))
    return replace(recipe, stages=(recipe.stages[0], changed))


class NativeExtendedThresholdContractTests(unittest.TestCase):
    def test_four_distinct_typed_recipes_round_trip_and_preserve_historical_catalogue(self):
        original = build_recipe_catalogue().to_json()
        recipes = build_native_extended_threshold_recipes()
        self.assertEqual(set(recipes), {
            'plantcv_threshold_gaussian', 'plantcv_threshold_dual_channels',
            'plantcv_threshold_texture', 'plantcv_threshold_triangle'})
        self.assertEqual(len({item.identity.graph_sha256 for item in recipes.values()}), 4)
        for recipe in recipes.values():
            self.assertEqual(RecipeSpec.from_json(recipe.to_json()), recipe)
            operation = recipe.stages[1]
            self.assertTrue(all(parameter.affects_calculation and parameter.editable
                for parameter in operation.parameters))
            self.assertEqual(operation.ports[-1].data_type, 'BinaryMask')
            self.assertEqual(len(operation.source_mappings), 1)
            self.assertEqual(operation.source_mappings[0].resolution.value, 'pinned')
        triangle = recipes['plantcv_threshold_triangle'].stages[1]
        self.assertEqual([item.parameter_id for item in triangle.parameters], ['object_type'])
        self.assertEqual(build_recipe_catalogue().to_json(), original)

    def test_parameter_construction_is_verified_from_loaded_code(self):
        native_recipe_module._module_record(recipe_factory)
        with patch.object(recipe_factory, '_parameter_contracts', lambda: {}):
            with self.assertRaisesRegex(NativeRecipeError, 'Loaded calculation'):
                native_recipe_module._module_record(recipe_factory)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA is required for extended threshold recipes')
class NativeExtendedThresholdRecipeTests(unittest.TestCase):
    def setUp(self):
        self.gray_np, self.rgb_np, self.triangle_np = _fixture()
        self.gray = torch.as_tensor(self.gray_np.copy(), device='cuda')
        self.rgb = torch.as_tensor(self.rgb_np.copy(), device='cuda')
        self.frame = CoordinateFrame('test.native-extended-threshold', self.gray_np.shape,
            'top-left', 'x-right,y-down', 'pixel')
        self.cache = PipelineAnalysisCache()
        self.recipes = build_native_extended_threshold_recipes()

    def run_recipe(self, recipe, *, gray=None, source=SOURCE, cache=None, cancel=None):
        return run_native_recipe(recipe, self.gray if gray is None else gray, self.rgb,
            source_sha256=source, coordinate_frame=self.frame, preparation=PREPARATION,
            cache=self.cache if cache is None else cache, cancel=cancel)

    def test_each_recipe_matches_pinned_public_operation_and_returns_cuda_mask(self):
        cases = (
            ('plantcv_threshold_gaussian', self.gray_np, 'gaussian',
             {'ksize': 5, 'offset': 1.25, 'object_type': 'light'},
             {'ksize': 5, 'offset': 1.25, 'object_type': 'light'}),
            ('plantcv_threshold_triangle', self.triangle_np, 'triangle',
             {'object_type': 'dark'}, {'object_type': 'dark', 'xstep': 1}),
            ('plantcv_threshold_texture', self.gray_np, 'texture',
             {'ksize': 3, 'offset': 1, 'texture_method': 'dissimilarity',
              'borders': 'nearest', 'threshold': 30},
             {'ksize': 3, 'offset': 1, 'texture_method': 'dissimilarity',
              'borders': 'nearest', 'threshold': 30}),
            ('plantcv_threshold_dual_channels', self.gray_np, 'dual_channels',
             {'x_channel': 'R', 'y_channel': 'G', 'points': [[40, 40], [180, 150]], 'above': True},
             {'x_channel': 'R', 'y_channel': 'G', 'points': [[40, 40], [180, 150]], 'above': True}))
        for recipe_id, gray_np, operation, native_parameters, public_parameters in cases:
            with self.subTest(operation=operation):
                recipe = _with_parameters(self.recipes[recipe_id], **native_parameters)
                gray = torch.as_tensor(gray_np.copy(), device='cuda')
                image = np.ascontiguousarray(self.rgb_np[..., ::-1]) if operation == 'dual_channels' else gray_np
                expected = _reference_mask(image, operation, public_parameters)
                result = self.run_recipe(recipe, gray=gray)
                node = result.node(recipe.stages[1].stage_id)
                self.assertEqual(node.status, 'executed', node.reason)
                self.assertEqual(result.status, 'executed')
                self.assertEqual(result.recipe_identity, recipe.identity)
                self.assertEqual(node.outputs['mask'].device, self.gray.device)
                self.assertEqual(node.outputs['mask'].dtype, torch.uint8)
                np.testing.assert_array_equal(node.outputs['mask'].cpu().numpy(), expected)

    def test_cache_controls_source_scope_and_tensor_version_are_recipe_local(self):
        recipe = _with_parameters(self.recipes['plantcv_threshold_gaussian'], offset=0.0)
        input_id, operation_id = (stage.stage_id for stage in recipe.stages)
        first = self.run_recipe(recipe)
        second = self.run_recipe(recipe)
        self.assertIs(first.node(operation_id).outputs['mask'], second.node(operation_id).outputs['mask'])
        self.assertEqual(self.cache.last_computed_nodes, ())
        self.assertEqual(set(self.cache.last_reused_nodes), {input_id, operation_id})
        changed = _with_parameters(recipe, offset=1.0)
        self.run_recipe(changed)
        self.assertNotEqual(changed.identity.settings_sha256, recipe.identity.settings_sha256)
        self.assertEqual(self.cache.last_computed_nodes, (operation_id,))
        self.assertEqual(self.cache.last_reused_nodes, (input_id,))
        triangle = self.recipes['plantcv_threshold_triangle']
        self.run_recipe(triangle, gray=torch.as_tensor(self.triangle_np.copy(), device='cuda'))
        self.assertNotIn(operation_id, self.cache.last_reused_nodes)
        self.assertNotEqual(triangle.identity.graph_sha256, recipe.identity.graph_sha256)
        self.run_recipe(changed, source='b'*64)
        self.assertEqual(set(self.cache.last_computed_nodes), {input_id, operation_id})
        self.gray.add_(1)
        self.run_recipe(changed, source='b'*64)
        self.assertEqual(set(self.cache.last_computed_nodes), {input_id, operation_id})

    def test_larger_gaussian_control_matches_public_and_recomputes_only_threshold(self):
        image = np.full(self.gray_np.shape,37,dtype=np.uint8)
        image[:,19:] = 219
        gray = torch.as_tensor(image,device='cuda')
        recipe = self.recipes['plantcv_threshold_gaussian']
        input_id, operation_id = (stage.stage_id for stage in recipe.stages)
        self.assertEqual(next(p.maximum for p in recipe.stages[1].parameters
            if p.parameter_id == 'ksize'),15)
        self.run_recipe(_with_parameters(recipe,ksize=11,offset=12),gray=gray)
        changed = _with_parameters(recipe,ksize=15,offset=12)
        result = self.run_recipe(changed,gray=gray)
        node = result.node(operation_id)
        self.assertEqual(node.status,'executed',node.reason)
        self.assertEqual(self.cache.last_computed_nodes,(operation_id,))
        self.assertEqual(self.cache.last_reused_nodes,(input_id,))
        expected = _reference_mask(image,'gaussian',dict(ksize=15,offset=12,object_type='light'))
        np.testing.assert_array_equal(node.outputs['mask'].cpu().numpy(),expected)

    def test_uncertain_large_gaussian_clears_cached_mask_and_can_recover(self):
        image = np.random.default_rng(411015).integers(0,256,self.gray_np.shape,dtype=np.uint8)
        gray = torch.as_tensor(image,device='cuda')
        recipe = self.recipes['plantcv_threshold_gaussian']
        operation_id = recipe.stages[1].stage_id
        self.assertEqual(self.run_recipe(recipe,gray=gray).node(operation_id).status,'executed')
        uncertain = _with_parameters(recipe,ksize=15)
        failed = self.run_recipe(uncertain,gray=gray)
        self.assertEqual(failed.node(operation_id).status,'failed')
        self.assertFalse(failed.node(operation_id).outputs)
        recovered = self.run_recipe(recipe,gray=gray)
        self.assertEqual(recovered.node(operation_id).status,'executed')
        self.assertFalse(recovered.node(operation_id).reused)

    def test_loaded_implementation_and_factory_are_bound_to_runtime_identity(self):
        records, _ = native_recipe_module._runtime_records(self.gray.device)
        self.assertIn(extended.__name__, records)
        self.assertIn(recipe_factory.__name__, records)

    def test_invalid_control_recovery_and_cancellation_never_publish_stale_mask(self):
        recipe = self.recipes['plantcv_threshold_gaussian']
        operation_id = recipe.stages[1].stage_id
        first = self.run_recipe(recipe)
        self.assertIn('mask', first.node(operation_id).outputs)
        with self.assertRaises(RecipeContractError):
            _with_parameters(recipe, ksize=2)
        recovered = self.run_recipe(recipe)
        self.assertEqual(recovered.node(operation_id).status, 'executed')
        # An invalid typed edit never reaches execution or evicts valid results.
        self.assertTrue(recovered.node(operation_id).reused)
        self.assertIs(recovered.node(operation_id).outputs['mask'], first.node(operation_id).outputs['mask'])
        event = __import__('threading').Event()
        event.set()
        with self.assertRaises(NativeRecipeCancelled):
            self.run_recipe(recipe, cache=PipelineAnalysisCache(), cancel=event.is_set)
        calls = 0
        def cancel_inside_gaussian():
            nonlocal calls
            calls += 1
            return calls >= 5
        cache = PipelineAnalysisCache()
        with self.assertRaises(NativeRecipeCancelled):
            self.run_recipe(recipe, cache=cache, cancel=cancel_inside_gaussian)
        self.assertNotIn(f'reference.recipe:{recipe.recipe_id}:{operation_id}', cache.values)
