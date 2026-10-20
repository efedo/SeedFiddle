"""Actual backend choices, public CPU adapters and node-local cache contracts."""

from tests.functional_units import functional_units
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import unittest
from tests.catalogue_fixtures import CatalogueTestCase
from unittest.mock import patch

import numpy as np
import torch

from seedvision.benchmarking import reference_runner as reference
from seedvision.benchmarking.native_recipe import build_native_seed_recipe, run_native_recipe, NativeRecipeCancelled
from seedvision.pipeline.backend_execution import (BackendExecution, BackendUnavailable,
    WATERSHED_OPERATIONS, backend_capability, resolve_backend, validate_backend_selection,
    validate_backend_provenance, validate_public_otsu_calculation)
from seedvision.pipeline.workflows import PipelineWorkflow
from seedvision.segmentation.baseline import PipelineAnalysisCache
from tests.test_native_recipe import selected_recipe, fixture, FRAME, PREPARATION, SOURCE


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / 'artifacts' / 'plantcv-reference-env' / 'Scripts' / 'python.exe'


def workflow(kind='binary', **kwargs):
    recipe = build_native_seed_recipe(selected_recipe(kind,
        **({'threshold.mean': {'ksize': 255, 'offset': 0}} if kind == 'mean' else {})))
    return PipelineWorkflow(recipe, recipe, **kwargs)


def checked_identity(recipe):
    return {'reference': 'plantcv', 'version': reference.PINNED_PLANTCV_VERSION,
        'commit': reference.PINNED_PLANTCV_COMMIT, 'python': reference.PINNED_REFERENCE_PYTHON,
        'platform': reference.PINNED_REFERENCE_PLATFORM,
        'source_audit_sha256': reference.PINNED_SOURCE_AUDIT_SHA256,
        'requirements_lock_sha256': reference.PINNED_REQUIREMENTS_SHA256,
        'identity_digest': '1' * 64,
        'actual_distributions': {name.lower().replace('_', '-'): value
            for name, value in recipe.provenance.environment_versions.items()}}


class FakeRunner:
    backend_identity = {'fixture': 'checked-public-boundary-v1'}

    def __init__(self, recipe, response=None):
        self.recipe, self.response, self.calls = recipe, response, []

    def run(self, operation, arrays, parameters, *, cancel_event=None):
        self.calls.append((operation, arrays, parameters))
        if self.response is not None:
            return self.response(operation, arrays, parameters)
        return reference.ReferenceResult(operation, {'mask': (arrays['gray'] > parameters['threshold']).astype(np.uint8) * 255},
            {}, {'callable': 'plantcv.plantcv.threshold.binary', **parameters}, (), {}, {},
            checked_identity(self.recipe), 1, 2)


@functional_units
class PipelineBackendPolicyTests(CatalogueTestCase):
    def check_test_native_default_and_public_only_where_equivalent(self):
        selected = workflow()
        self.assertTrue(all(resolve_backend(selected, stage.stage_id) == 'native' for stage in selected.recipe.stages))
        public = replace(selected, backend_default='plantcv')
        image = next(stage for stage in public.recipe.stages if stage.operation_id == 'external.image_inputs')
        threshold = next(stage for stage in public.recipe.stages if stage.operation_id == 'threshold.binary')
        self.assertEqual(resolve_backend(public, image.stage_id), 'native')
        self.assertEqual(resolve_backend(public, threshold.stage_id), 'plantcv')
        self.assertIn('input boundary', backend_capability(public.recipe, image.stage_id).reason)
        from seedvision.benchmarking.recipe_catalogue import build_recipe_catalogue
        unselected = build_native_seed_recipe(build_recipe_catalogue().recipe('plantcv_seed_binary'))
        median = next(stage for stage in unselected.stages if stage.operation_id == 'filter.median_blur')
        self.assertIsNone(unselected.parameters_for_runner(median.stage_id)['ksize'])
        self.assertIn('plantcv', backend_capability(unselected, median.stage_id).backends)

    def check_test_public_watershed_requires_coherent_atomic_choices(self):
        selected = workflow()
        group = tuple(stage.stage_id for stage in selected.recipe.stages if stage.operation_id in WATERSHED_OPERATIONS)
        self.assertEqual(set(backend_capability(selected.recipe, group[0]).atomic_group), set(group))
        with self.assertRaisesRegex(BackendUnavailable, 'atomic'):
            validate_backend_selection(replace(selected, node_backends={group[0]: 'plantcv'}))
        validate_backend_selection(replace(selected, node_backends={identifier: 'plantcv' for identifier in group}))

    def check_test_native_only_explicit_override_is_rejected(self):
        selected = workflow()
        image = next(stage.stage_id for stage in selected.recipe.stages if stage.operation_id == 'external.image_inputs')
        with self.assertRaisesRegex(BackendUnavailable, 'input boundary'):
            validate_backend_selection(replace(selected, node_backends={image: 'plantcv'}))

    def check_test_edited_watershed_mask_route_cannot_claim_public_equivalence(self):
        selected = workflow()
        flood = next(stage.stage_id for stage in selected.recipe.stages if stage.operation_id.endswith('.masked_flood'))
        erode = next(stage.stage_id for stage in selected.recipe.stages if stage.operation_id == 'morphology.erode')
        altered = replace(selected.recipe, dependencies=tuple(replace(edge, source_stage=erode)
            if edge.target_stage == flood and edge.target_port == 'mask' else edge for edge in selected.recipe.dependencies))
        self.assertEqual(backend_capability(altered, flood).backends, ('native',))
        self.assertIn('connections', backend_capability(altered, flood).reason)

    def check_test_public_binary_adapter_calls_runner_and_returns_device_products(self):
        selected = workflow(backend_default='plantcv')
        stage = next(stage for stage in selected.recipe.stages if stage.operation_id == 'threshold.binary')
        runner = FakeRunner(selected.recipe)
        executor = BackendExecution(selected.recipe, workflow=selected, runner=runner)
        pixels = torch.tensor([[0, 100, 101, 255]], dtype=torch.uint8)
        outputs, provenance, calculation = executor.execute(stage, {'gray': pixels},
            selected.recipe.parameters_for_runner(stage.stage_id), {})
        self.assertEqual(len(runner.calls), 1)
        self.assertEqual(runner.calls[0][0], 'threshold.binary')
        self.assertTrue(torch.equal(outputs['mask'], torch.tensor([[0, 0, 255, 255]], dtype=torch.uint8)))
        self.assertEqual(outputs['mask'].device, pixels.device)
        self.assertEqual(provenance['backend'], 'plantcv')
        self.assertIsNone(calculation)
        validate_backend_provenance(provenance)

    def check_test_public_runtime_unavailable_never_calls_native_dispatch(self):
        selected = workflow(backend_default='plantcv')
        stage = next(stage for stage in selected.recipe.stages if stage.operation_id == 'threshold.binary')
        executor = BackendExecution(selected.recipe, workflow=selected,
            runner=reference.PlantCVReferenceRunner(ROOT / 'artifacts' / 'missing-reference-python.exe'))
        with patch.object(reference, 'plantcv_reference_status', return_value={'available': False, 'reason': 'missing pinned runtime'}):
            with self.assertRaisesRegex(BackendUnavailable, 'missing pinned runtime'):
                executor.execute(stage, {'gray': torch.zeros((2, 2), dtype=torch.uint8)}, {'threshold': 100, 'object_type': 'light'}, {})

    def check_test_cancelled_public_adapter_publishes_no_outputs(self):
        selected = workflow(backend_default='plantcv')
        stage = next(stage for stage in selected.recipe.stages if stage.operation_id == 'threshold.binary')
        runner = FakeRunner(selected.recipe)
        executor = BackendExecution(selected.recipe, workflow=selected, runner=runner, cancel=lambda: True)
        with self.assertRaises(NativeRecipeCancelled):
            executor.execute(stage, {'gray': torch.zeros((2, 2), dtype=torch.uint8)}, {'threshold': 100, 'object_type': 'light'}, {})
        self.assertFalse(runner.calls)

    def check_test_public_input_mutation_is_not_published(self):
        selected = workflow(backend_default='plantcv')
        stage = next(stage for stage in selected.recipe.stages if stage.operation_id == 'threshold.binary')
        pixels = torch.zeros((2, 2), dtype=torch.uint8)
        runner = FakeRunner(selected.recipe)
        original = runner.run
        def mutate(*args, **kwargs):
            result = original(*args, **kwargs)
            pixels[0, 0] = 255
            return result
        runner.run = mutate
        executor = BackendExecution(selected.recipe, workflow=selected, runner=runner)
        with self.assertRaisesRegex(ValueError, 'input changed'):
            executor.execute(stage, {'gray': pixels}, {'threshold': 100, 'object_type': 'light'}, {})

    def check_test_public_provenance_rejects_wrong_source_identity(self):
        selected = workflow(backend_default='plantcv')
        runner = FakeRunner(selected.recipe)
        stage = next(stage for stage in selected.recipe.stages if stage.operation_id == 'threshold.binary')
        _, provenance, _ = BackendExecution(selected.recipe, workflow=selected, runner=runner).execute(
            stage, {'gray': torch.zeros((2, 2), dtype=torch.uint8)}, {'threshold': 100, 'object_type': 'light'}, {})
        forged = dict(provenance) | {'identity': dict(provenance['identity']) | {'version': 'different'}}
        with self.assertRaisesRegex(ValueError, 'pinned'):
            validate_backend_provenance(forged)

    def test_workflow_lifecycle(self):
        """4 related PipelineBackendPolicyTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_native_default_and_public_only_where_equivalent)
        self.scenario(self.check_test_public_watershed_requires_coherent_atomic_choices)
        self.scenario(self.check_test_edited_watershed_mask_route_cannot_claim_public_equivalence)
        self.scenario(self.check_test_public_binary_adapter_calls_runner_and_returns_device_products)

    def test_validation_and_recovery(self):
        """5 related PipelineBackendPolicyTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_native_only_explicit_override_is_rejected)
        self.scenario(self.check_test_public_runtime_unavailable_never_calls_native_dispatch)
        self.scenario(self.check_test_cancelled_public_adapter_publishes_no_outputs)
        self.scenario(self.check_test_public_input_mutation_is_not_published)
        self.scenario(self.check_test_public_provenance_rejects_wrong_source_identity)


class RecordingRunner(reference.PlantCVReferenceRunner):
    def __init__(self):
        super().__init__(REFERENCE)
        self.calls = []

    def run(self, operation, *args, **kwargs):
        self.calls.append(operation)
        return super().run(operation, *args, **kwargs)


@unittest.skipUnless(torch.cuda.is_available() and REFERENCE.is_file(), 'CUDA and the pinned public environment are required')
@functional_units
class PipelineBackendPublicIntegrationTests(CatalogueTestCase):
    def setUp(self):
        gray, rgb = fixture()
        self.gray = torch.as_tensor(gray, device='cuda')
        self.rgb = torch.as_tensor(rgb, device='cuda')
        self.cache = PipelineAnalysisCache()
        self.runner = RecordingRunner()

    def execute(self, selected, target=None):
        return run_native_recipe(selected.recipe, self.gray, self.rgb, source_sha256=SOURCE,
            coordinate_frame=FRAME, preparation=PREPARATION, cache=self.cache,
            working_workflow=selected, target_node_id=target, reference_runner=self.runner)

    def check_test_all_three_seed_factories_execute_the_actual_public_complete_pipeline(self):
        for kind in ('binary', 'mean', 'otsu'):
            with self.subTest(kind=kind):
                self.cache = PipelineAnalysisCache()
                self.runner.calls.clear()
                selected = workflow(kind, backend_default='plantcv')
                result = self.execute(selected)
                self.assertEqual(result.status, 'executed', [(node.stage_id, node.status, node.reason) for node in result.nodes])
                self.assertEqual(self.runner.calls.count('segmentation.watershed.capture'), 1)
                for node in result.nodes:
                    if node.operation_id != 'external.image_inputs':
                        self.assertEqual(node.backend, 'plantcv')
                        validate_backend_provenance(node.backend_provenance)
                    for output in node.outputs.values():
                        if torch.is_tensor(output):
                            self.assertEqual(output.device.type, 'cuda')
                calls = tuple(self.runner.calls)
                warm = self.execute(selected)
                self.assertEqual(tuple(self.runner.calls), calls)
                self.assertTrue(all(node.reused for node in warm.nodes))
                if kind == 'otsu':
                    node = next(node for node in result.nodes if node.operation_id == 'threshold.otsu')
                    validate_public_otsu_calculation(node.calculation, 'light')

    def check_test_single_backend_change_reuses_native_ancestors(self):
        selected = workflow()
        before = self.execute(selected)
        threshold = next(stage for stage in selected.recipe.stages if stage.operation_id == 'threshold.binary')
        changed = replace(selected, node_backends={threshold.stage_id: 'plantcv'})
        after = self.execute(changed)
        self.assertEqual(after.status, 'executed')
        self.assertEqual(self.runner.calls, ['threshold.binary'])
        ancestors = [stage.stage_id for stage in selected.recipe.stages
            if stage.operation_id in ('external.image_inputs', 'filter.median_blur')]
        self.assertTrue(all(after.node(identifier).reused for identifier in ancestors))
        self.assertFalse(after.node(threshold.stage_id).reused)
        self.assertTrue(torch.equal(before.node(threshold.stage_id).outputs['mask'], after.node(threshold.stage_id).outputs['mask']))
        from seedvision.benchmarking.checkpoints import _freeze_json
        public_node = after.node(threshold.stage_id)
        forged = dict(public_node.backend_provenance)
        forged['effective_parameters'] = dict(forged['effective_parameters']) | {'threshold': 101}
        object.__setattr__(public_node, 'backend_provenance', _freeze_json(forged, name='forged cache fixture'))
        self.runner.calls.clear()
        repaired = self.execute(changed)
        self.assertEqual(self.runner.calls, ['threshold.binary'])
        self.assertFalse(repaired.node(threshold.stage_id).reused)
        self.assertEqual(repaired.node(threshold.stage_id).backend_provenance['effective_parameters']['threshold'], 100)
        self.assertTrue(all(repaired.node(identifier).reused for identifier in ancestors))
        object.__setattr__(repaired.node(threshold.stage_id), 'backend', 'native')
        self.runner.calls.clear()
        repaired_again = self.execute(changed)
        self.assertEqual(self.runner.calls, ['threshold.binary'])
        self.assertEqual(repaired_again.node(threshold.stage_id).backend, 'plantcv')
        from seedvision.pipeline.workflows import with_foreground_region
        region = with_foreground_region(changed, threshold.stage_id, (30,20,70,60))
        bounded = self.execute(region)
        expected = torch.zeros_like(before.node(threshold.stage_id).outputs['mask'])
        expected[20:60,30:70] = before.node(threshold.stage_id).outputs['mask'][20:60,30:70]
        self.assertTrue(torch.equal(bounded.node(threshold.stage_id).outputs['mask'], expected))
        self.assertEqual(bounded.node(threshold.stage_id).backend, 'plantcv')
        self.assertIn('foreground_region_adapter', bounded.node(threshold.stage_id).backend_provenance)
        self.assertTrue(all(bounded.node(identifier).reused for identifier in ancestors))



    def check_test_mutated_public_capture_invalidates_whole_group_only(self):
        selected = workflow(backend_default='plantcv')
        first = self.execute(selected)
        markers = next(node for node in first.nodes if node.operation_id.endswith('.connected_markers'))
        markers.outputs['markers'][0, 0] += 1
        self.runner.calls.clear()
        second = self.execute(selected)
        self.assertEqual(second.status, 'executed')
        self.assertEqual(self.runner.calls, ['segmentation.watershed.capture', 'analysis.size', 'analysis.color'])
        self.assertTrue(all(second.node(stage.stage_id).reused for stage in selected.recipe.stages
            if stage.operation_id in ('external.image_inputs', 'filter.median_blur', 'threshold.binary',
                'morphology.fill', 'morphology.erode', 'morphology.dilate')))

    def test_workflow_lifecycle(self):
        """2 related PipelineBackendPublicIntegrationTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_all_three_seed_factories_execute_the_actual_public_complete_pipeline)
        self.scenario(self.check_test_single_backend_change_reuses_native_ancestors)

    def test_validation_and_recovery(self):
        """1 related PipelineBackendPublicIntegrationTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_mutated_public_capture_invalidates_whole_group_only)


if __name__ == '__main__':
    unittest.main()
