"""Pinned public-call adapters beyond the seed template's elementary stages."""

from tests.functional_units import functional_units
from dataclasses import replace
import tempfile
import unittest
from tests.catalogue_fixtures import CatalogueTestCase

import numpy as np
import torch

from seedvision.pipeline.backend_execution import BackendExecution, backend_capability
from seedvision.pipeline.workflows import PipelineWorkflow
from seedvision.benchmarking.native_recipe import _dispatch
from tests.test_pipeline_backend_execution import REFERENCE, RecordingRunner


@unittest.skipUnless(torch.cuda.is_available() and REFERENCE.is_file(), 'CUDA and pinned public Python required')
@functional_units
class PipelinePublicAdapterTests(CatalogueTestCase):
    def setUp(self):
        self.runner = RecordingRunner()

    def check_test_extended_threshold_and_range_public_calls_preserve_masks_and_rgb(self):
        from seedvision.benchmarking.native_extended_threshold_recipes import build_native_extended_threshold_recipes
        from seedvision.benchmarking.native_range_recipe import build_native_range_recipe
        gray = torch.arange(20 * 24, device='cuda').remainder(256).to(torch.uint8).reshape(20, 24)
        rgb = torch.stack((gray, 255 - gray, gray // 2), -1).contiguous()
        recipes = [*build_native_extended_threshold_recipes().values(), build_native_range_recipe()]
        for recipe in recipes:
            with self.subTest(operation=recipe.stages[-1].operation_id):
                selected = PipelineWorkflow(recipe, recipe, backend_default='plantcv')
                stage = recipe.stages[-1]
                source = gray
                if stage.operation_id == 'threshold.triangle':
                    source = torch.full((12, 12), 30, dtype=torch.uint8, device='cuda')
                    source.reshape(-1)[::3] = 180
                elif stage.operation_id == 'threshold.texture':
                    source = torch.full((4, 5), 42, dtype=torch.uint8, device='cuda')
                inputs = {'rgb': rgb} if stage.operation_id in ('threshold.dual_channels', 'threshold.custom_range') else {'gray': source}
                parameters = recipe.parameters_for_runner(stage.stage_id)
                actual, provenance, _ = BackendExecution(recipe, workflow=selected, runner=self.runner).execute(stage, inputs, parameters, {})
                expected = _dispatch(stage.operation_id, inputs, parameters, None)
                for name in actual:
                    self.assertTrue(torch.equal(actual[name], expected[name]), stage.operation_id + '.' + name)
                    self.assertEqual(actual[name].device.type, 'cuda')
                self.assertEqual(provenance['backend'], 'plantcv')
        self.assertEqual(len(self.runner.calls), 5)

    def check_test_portable_bayes_tables_execute_original_public_classifier(self):
        from tests.test_native_naive_bayes_recipe import _recipe_with_model
        from tests.test_plantcv_naive_bayes import _pdf, _image
        from seedvision.benchmarking.native_recipe import run_native_recipe_path
        import cv2
        with tempfile.TemporaryDirectory() as temporary:
            from pathlib import Path
            root = Path(temporary)
            (root / 'model.tsv').write_bytes(_pdf())
            recipe, _ = _recipe_with_model(root / 'model.tsv')
            image = root / 'image.png'
            self.assertTrue(cv2.imwrite(str(image), _image()))
            native = run_native_recipe_path(recipe, image)
            selected = PipelineWorkflow(recipe, recipe, backend_default='plantcv')
            public = run_native_recipe_path(recipe, image, working_workflow=selected, reference_runner=self.runner)
            self.assertEqual(public.status, 'executed', [(n.stage_id, n.reason) for n in public.nodes])
            actual = public.node('analysis.naive_bayes_classifier')
            expected = native.node('analysis.naive_bayes_classifier')
            self.assertEqual(actual.backend, 'plantcv')
            self.assertEqual(actual.outputs['class_names'], expected.outputs['class_names'])
            for name in actual.outputs['class_names']:
                self.assertTrue(torch.equal(actual.outputs['class_masks'][name], expected.outputs['class_masks'][name]))
            self.assertEqual(self.runner.calls, ['analysis.naive_bayes_classifier'])
            self.assertIn('round-trip', actual.backend_provenance['effective_parameters']['pdf_transport'])

    def check_test_four_input_correction_public_backend_preserves_frames_and_matrices(self):
        from tests.test_native_correction_recipe import _cuda_inputs, _TARGET_FRAME, _SOURCE_FRAME
        from seedvision.benchmarking.native_correction_recipe import build_native_correction_recipe, run_native_correction_recipe, STAGE_ID
        from seedvision.segmentation.baseline import PipelineAnalysisCache
        recipe = build_native_correction_recipe()
        selected = PipelineWorkflow(recipe, recipe, backend_default='plantcv')
        inputs, cache = _cuda_inputs(), PipelineAnalysisCache()
        kwargs = dict(target_source_sha256='a' * 64, source_sha256='b' * 64,
            target_coordinate_frame=_TARGET_FRAME, coordinate_frame=_SOURCE_FRAME)
        native = run_native_correction_recipe(recipe, *inputs, **kwargs)
        public = run_native_correction_recipe(recipe, *inputs, **kwargs, cache=cache,
            working_workflow=selected, reference_runner=self.runner)
        self.assertEqual(public.node(STAGE_ID).backend, 'plantcv')
        self.assertEqual(public.coordinate_frame, native.coordinate_frame)
        self.assertEqual(public.target_coordinate_frame, native.target_coordinate_frame)
        for name in ('target_matrix', 'source_matrix'):
            np.testing.assert_allclose(public.node(STAGE_ID).outputs[name], native.node(STAGE_ID).outputs[name], rtol=0, atol=2e-15)
        np.testing.assert_allclose(public.node(STAGE_ID).outputs['transformation_matrix'],
            native.node(STAGE_ID).outputs['transformation_matrix'], rtol=1e-8, atol=1e-8)
        difference = (public.node(STAGE_ID).outputs['corrected_image'].to(torch.int16)
            - native.node(STAGE_ID).outputs['corrected_image'].to(torch.int16)).abs()
        self.assertLessEqual(int(difference.max().item()), 1)
        warm = run_native_correction_recipe(recipe, *inputs, **kwargs, cache=cache,
            working_workflow=selected, reference_runner=self.runner)
        self.assertTrue(warm.node(STAGE_ID).reused)
        self.assertEqual(self.runner.calls, ['transform.correct_color'])
        from seedvision.benchmarking.checkpoints import _freeze_json
        original = public.node(STAGE_ID)
        altered = dict(original.backend_provenance)
        altered['effective_parameters'] = dict(altered['effective_parameters']) | {'output_directory': 'forged cached call'}
        object.__setattr__(original, 'backend_provenance', _freeze_json(altered, name='forged correction cache fixture'))
        repaired = run_native_correction_recipe(recipe, *inputs, **kwargs, cache=cache,
            working_workflow=selected, reference_runner=self.runner)
        self.assertFalse(repaired.node(STAGE_ID).reused)
        self.assertEqual(self.runner.calls, ['transform.correct_color', 'transform.correct_color'])
        self.assertNotEqual(repaired.node(STAGE_ID).backend_provenance['effective_parameters']['output_directory'], 'forged cached call')

    def check_test_public_size_and_all_color_observations_match_native_measurements(self):
        from tests.test_pipeline_backend_execution import workflow
        from tests.test_native_recipe import fixture
        selected = workflow(backend_default='plantcv')
        _, image = fixture()
        rgb = torch.as_tensor(image, device='cuda')
        labels = torch.zeros(image.shape[:2], dtype=torch.int32, device='cuda')
        labels[20:70, 20:60] = 1
        labels[30:65, 60:100] = 2
        executor = BackendExecution(selected.recipe, workflow=selected, runner=self.runner)
        size = next(stage for stage in selected.recipe.stages if stage.operation_id == 'analysis.size')
        size_inputs = {'image': rgb, 'labels': labels}
        size_parameters = {'n_labels': 2, 'label': 'graph', 'line_thickness': 5}
        actual, _, _ = executor.execute(size, size_inputs, size_parameters, {})
        expected = _dispatch('analysis.size', size_inputs, size_parameters, None)
        for sample in actual['observations']:
            for variable in ('area', 'width', 'height', 'center_of_mass', 'perimeter'):
                np.testing.assert_allclose(actual['observations'][sample][variable]['value'],
                    expected['observations'][sample][variable]['value'], rtol=1e-6, atol=1e-6)
        color = next(stage for stage in selected.recipe.stages if stage.operation_id == 'analysis.color')
        color_inputs = {'rgb': rgb, 'labels': labels}
        color_parameters = {'n_labels': 2, 'label': 'graph', 'colorspaces': 'all'}
        actual, _, _ = executor.execute(color, color_inputs, color_parameters, {})
        expected = _dispatch('analysis.color', color_inputs, color_parameters, None)
        self.assertEqual(actual['histogram'].data_records(), expected['histogram'].data_records())
        for sample in expected['observations'].labels:
            for histogram in sample.histograms:
                record = actual['observations'][sample.sample][histogram.variable]
                self.assertEqual(tuple(record['label']), histogram.bin_labels)
                np.testing.assert_allclose(record['value'], histogram.values, rtol=0, atol=1e-12)
        self.assertEqual(self.runner.calls, ['analysis.size', 'analysis.color'])

    def check_test_public_card_matrix_and_affine_helpers_keep_native_detector_context(self):
        from seedvision.benchmarking import native_card_recipe as card
        from tests.test_native_correction_recipe import _cuda_inputs
        _, _, rgb, mask = _cuda_inputs()
        bgr = rgb.flip(-1).contiguous()
        recipe = card.build_native_card_recipe()
        selected = PipelineWorkflow(recipe, recipe, backend_default='plantcv')
        executor = BackendExecution(recipe, workflow=selected, runner=self.runner)
        means, _, _ = executor.execute(recipe.stage(card.MEANS), {'bgr': bgr, 'labeled_mask': mask.to(torch.float64)}, {}, {})
        expected = _dispatch('color_card.means', {'bgr': bgr, 'labeled_mask': mask.to(torch.float64)}, {}, None)
        np.testing.assert_allclose(means['source_matrix'], expected['source_matrix'], rtol=0, atol=2e-15)
        target, _, _ = executor.execute(recipe.stage(card.STANDARD), {'bgr': bgr}, {'pos': 3}, {})
        expected_target = _dispatch('color_card.standard', {'bgr': bgr}, {'pos': 3}, None)
        self.assertEqual(target, expected_target)
        inputs = {'bgr': bgr, **means, **target}
        actual, provenance, _ = executor.execute(recipe.stage(card.AFFINE), inputs, {}, {})
        expected_affine = _dispatch('color_card.affine', inputs, {}, None)
        np.testing.assert_allclose(actual['transformation_matrix'], expected_affine['transformation_matrix'], rtol=1e-12, atol=1e-12)
        difference = (actual['corrected_rgb'].to(torch.int16) - expected_affine['corrected_rgb'].to(torch.int16)).abs()
        self.assertLessEqual(int(difference.max().item()), 1)
        self.assertIn('actual ar/ag/ab', provenance['effective_parameters']['coefficient_origin'])
        self.assertEqual(backend_capability(recipe, card.GEOMETRY).backends, ('native',))

    def test_additional_numerical_regressions(self):
        self.scenario(self.check_test_extended_threshold_and_range_public_calls_preserve_masks_and_rgb)
        self.scenario(self.check_test_portable_bayes_tables_execute_original_public_classifier)
        self.scenario(self.check_test_four_input_correction_public_backend_preserves_frames_and_matrices)
        self.scenario(self.check_test_public_size_and_all_color_observations_match_native_measurements)
        self.scenario(self.check_test_public_card_matrix_and_affine_helpers_keep_native_detector_context)







if __name__ == '__main__':
    unittest.main()
