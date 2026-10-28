"""Native selection/cache/report witnesses for the frozen trait classifier."""

from tests.functional_units import functional_units
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import unittest

import numpy as np
import torch

from seedvision.export.results import export_report, result_report, select_result
from seedvision.learning.instance_selection import select_instances
from seedvision.learning.trait_pipeline import (
    TraitClassifierSettings, run_trait_classifier,
)
from seedvision.pipeline import build_default_pipeline
from seedvision.pipeline.settings import production_settings
import tests.test_trait_classifier as core_fixture


@unittest.skipUnless(torch.cuda.is_available(), 'Native classifier inference requires CUDA')
@functional_units
class TraitPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        core_fixture.TraitClassifierTests.setUpClass()
        cls.model = core_fixture.TraitClassifierTests.model

    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.path = self.root / 'traits.json'
        self.path.write_text(self.model.to_json(), encoding='utf-8')
        self.labels = np.zeros((12, 12), np.int32)
        self.labels[2:4, 2:4] = 1
        self.labels[7:9, 7:9] = 2
        self.corrected = np.zeros((24, 24, 3), np.uint8)
        self.corrected[4:8, 4:8, 2] = 30
        self.corrected[14:18, 14:18, 2] = 225
        self.instances = SimpleNamespace(labels=self.labels, source_shape=(24, 24),
                                         instance_confidences=np.array([.8, .9]))
        self.source = torch.tensor(self.corrected, device='cuda').float().permute(2, 0, 1)[None]
        self.valid = torch.ones((1, 1, 24, 24), device='cuda', dtype=torch.bool)
        self.cache = {}

    def run_classifier(self, *, threshold=.8, path=None, instances=None, source=None, valid=None):
        return run_trait_classifier('procedural_instances', instances or self.instances,
            corrected_bgr=self.corrected, crop_offset=(0, 0), affine=np.eye(3),
            gpu_source=self.source if source is None else source,
            gpu_valid=self.valid if valid is None else valid,
            settings=TraitClassifierSettings(str(path or self.path), threshold), root=self.root, cache=self.cache)

    def analysis(self, result):
        return SimpleNamespace(image_path=None, crop_offset=(0, 0), warnings=(),
            calibration=SimpleNamespace(corrected_bgr=self.corrected, affine_matrix=np.eye(3)),
            layers=SimpleNamespace(foreground_colour_profile=object()),
            procedural_instances=self.instances, trait_classifier=result)

    def check_test_selected_method_preference_is_shared_and_filters_disabled_branches(self):
        values = dict(unet=object(), stardist=object(), cellpose=object(), procedural=object())
        analysis = SimpleNamespace(unet_instances=values['unet'], stardist_instances=values['stardist'],
            cellpose_sam_instances=values['cellpose'], procedural_instances=values['procedural'],
            layers=SimpleNamespace(foreground_colour_profile=object()))
        for enabled in (None, {'procedural_instances'}, {'stardist_instances', 'procedural_instances'},
                        {'cellpose_sam_instances', 'procedural_instances'}):
            self.assertEqual(select_instances(**values, enabled=enabled), select_result(analysis, enabled))
        analysis.layers.foreground_colour_profile = None
        self.assertIsNone(select_result(analysis, {'procedural_instances'})[1])

    def check_test_actual_cuda_resize_colour_features_predict_without_raster_download(self):
        with patch.object(torch.Tensor, 'cpu', side_effect=AssertionError('raster downloaded')):
            result, computed = self.run_classifier()
        self.assertTrue(computed)
        rows = result.report['predictions']
        self.assertEqual([x['predicted_trait_id'] for x in rows], ['dark', 'light'])
        features = self.cache['trait_classifier.features']
        self.assertTrue(features.values.is_cuda)
        torch.testing.assert_close(features.values[:, 15], torch.full((2,), np.log(16), device='cuda', dtype=torch.float64))
        self.assertEqual(result.report['input_binding']['crop_offset'], [0, 0])
        self.assertFalse(result.report['scientifically_validated'])

    def check_test_threshold_only_reuses_features_and_scores_then_identical_run_reuses_result(self):
        first, _ = self.run_classifier(threshold=0)
        features = self.cache['trait_classifier.features']
        with patch('seedvision.learning.trait_pipeline.extract_trait_features', side_effect=AssertionError('features ran')), \
             patch('seedvision.learning.trait_pipeline.predict_traits', side_effect=AssertionError('scoring ran')):
            second, computed = self.run_classifier(threshold=1)
            same, reused = self.run_classifier(threshold=1)
        self.assertTrue(computed)
        self.assertFalse(reused)
        self.assertIs(second, same)
        self.assertIs(features, self.cache['trait_classifier.features'])
        self.assertNotEqual(first.report['abstention_threshold'], second.report['abstention_threshold'])

    def check_test_model_content_change_rescores_without_reextracting_features(self):
        first, _ = self.run_classifier()
        payload = self.model.payload
        payload['weights'][0][0] *= -1
        from seedvision.learning.trait_classifier import FrozenTraitModel
        changed = FrozenTraitModel(json.dumps(payload))
        self.path.write_text(changed.to_json(), encoding='utf-8')
        with patch('seedvision.learning.trait_pipeline.extract_trait_features', side_effect=AssertionError('features ran')):
            second, computed = self.run_classifier()
        self.assertTrue(computed)
        self.assertNotEqual(first.report['model_sha256'], second.report['model_sha256'])

    def check_test_label_corrected_pixels_and_support_changes_invalidate_features(self):
        self.run_classifier()
        old = self.cache['trait_classifier.features']
        self.labels[4, 4] = 1
        self.run_classifier()
        self.assertIsNot(old, self.cache['trait_classifier.features'])
        old = self.cache['trait_classifier.features']
        self.corrected[4, 4, 2] = 31
        self.source[0, 2, 4, 4] = 31
        self.run_classifier()
        self.assertIsNot(old, self.cache['trait_classifier.features'])
        self.valid[0, 0, 4, 4] = False
        result, _ = self.run_classifier()
        self.assertEqual(result.report['predictions'][0]['abstention_reason'], 'outside_valid_crop')

    def check_test_missing_invalid_models_and_frame_boundary_inputs_abstain(self):
        with patch('seedvision.learning.trait_pipeline.extract_trait_features', side_effect=AssertionError('features ran')):
            absent, _ = self.run_classifier(path='None')
            missing, _ = self.run_classifier(path=self.root / 'missing.json')
        self.assertEqual(absent.report['predictions'][0]['abstention_reason'], 'no_frozen_model')
        self.assertEqual(missing.report['predictions'][0]['abstention_reason'], 'model_file_missing')
        self.path.write_text('{}')
        invalid, _ = self.run_classifier()
        self.assertEqual(invalid.report['predictions'][0]['abstention_reason'], 'invalid_model')
        self.path.write_text(self.model.to_json())
        wrong = SimpleNamespace(labels=self.labels, source_shape=(25, 24))
        mismatched, _ = self.run_classifier(instances=wrong)
        self.assertEqual(mismatched.report['predictions'][0]['abstention_reason'], 'incompatible_frame')
        self.labels[0, 0] = 1
        boundary, _ = self.run_classifier()
        self.assertEqual(boundary.report['predictions'][0]['abstention_reason'], 'crop_boundary_instance')

    def check_test_report_persists_actual_traits_and_threshold_changes_stale_review(self):
        prediction, _ = self.run_classifier(threshold=0)
        analysis = self.analysis(prediction)
        with patch('seedvision.learning.trait_pipeline.run_trait_classifier', side_effect=AssertionError('export recomputed')):
            report = result_report(analysis)
            destination = export_report(self.root / 'results', analysis, report)
        self.assertEqual([x['trait_assignment'] for x in report['instances']], ['dark', 'light'])
        stored = json.loads((destination / 'results.json').read_text())
        self.assertEqual(stored['trait_classifier'], stored['identity']['trait_classifier'])
        review = dict(result_revision=report['result_revision'], decisions={'1': 'accepted'})
        self.assertEqual(result_report(analysis, review=review)['accepted_count'], 1)
        analysis.trait_classifier = self.run_classifier(threshold=.99)[0]
        self.assertEqual(result_report(analysis, review=review)['accepted_count'], 0)

    def check_test_changed_exported_labels_crop_pixels_or_model_reject_before_report(self):
        result, _ = self.run_classifier()
        analysis = self.analysis(result)
        self.labels[5, 5] = 1
        with self.assertRaisesRegex(ValueError, 'inputs changed'): result_report(analysis)
        self.labels[5, 5] = 0
        analysis.crop_offset = (1, 0)
        with self.assertRaisesRegex(ValueError, 'inputs changed'): result_report(analysis)
        analysis.crop_offset = (0, 0)
        self.corrected[0, 0, 0] = 1
        with self.assertRaisesRegex(ValueError, 'inputs changed'): result_report(analysis)
        self.corrected[0, 0, 0] = 0
        self.path.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'model changed'): result_report(analysis)

    def check_test_graph_settings_roundtrip_and_authored_dependencies(self):
        graph = build_default_pipeline()
        self.assertIn('trait_classifier', graph.nodes)
        node = graph.node('trait_classifier')
        node.set_parameter('trait_model_path', str(self.path))
        node.set_parameter('trait_abstention_threshold', .7)
        settings = production_settings(graph)['trait_settings']
        self.assertEqual(settings, TraitClassifierSettings(str(self.path), .7))
        from seedvision.persistence.analysis_settings import (
            analysis_settings_profile_from_graph, analysis_settings_profile_to_payload,
            analysis_settings_profile_from_payload,
        )
        profile = analysis_settings_profile_from_graph(graph)
        restored = analysis_settings_profile_from_payload(analysis_settings_profile_to_payload(profile))
        self.assertEqual(restored, profile)
        self.assertIn('trait_model_path', {x.key for x in node.parameter_specs})
        self.assertIn('trait_abstention_threshold', {x.key for x in node.parameter_specs})
        self.assertEqual(set(graph.upstream('trait_classifier')), {'deskew_colour', 'layout_detection',
            'procedural_instances', 'unet_instances', 'stardist_instances', 'cellpose_sam_instances'})

    def check_test_classifier_enables_with_alternative_branches_and_required_inputs_remain_honest(self):
        graph = build_default_pipeline()
        graph.set_enabled('trait_classifier', True)
        self.assertFalse(graph.node('unet_instances').enabled)
        self.assertFalse(graph.node('stardist_instances').enabled)
        graph.set_enabled('procedural_instances', False)
        self.assertTrue(graph.node('trait_classifier').enabled)
        graph.set_enabled('procedural_instances', True)
        edge = graph.connection_for_input('trait_classifier', 'unet')
        graph.disconnect(edge)
        self.assertFalse(graph.node('trait_classifier').enabled)
        with self.assertRaisesRegex(ValueError, 'Reconnect required'):
            graph.set_enabled('trait_classifier', True)
        graph.connect(edge.source, edge.source_port, edge.target, edge.target_port)
        self.assertTrue(graph.node('trait_classifier').enabled)
        graph.set_enabled('layout_detection', False)
        self.assertFalse(graph.node('trait_classifier').enabled)

    def check_test_native_worker_forwards_classifier_settings(self):
        import inspect
        from seedvision.ui.main_window import _AnalysisTask
        arguments = {name: None for name, parameter in inspect.signature(_AnalysisTask).parameters.items()
            if parameter.default is inspect.Parameter.empty}
        settings = TraitClassifierSettings(str(self.path), .63)
        arguments.update(path=self.root / 'capture.png', learning_root=self.root, pipeline_revision=1,
            species='unknown', seed_trait_species='', background_colour_enabled=True,
            background_reference_points=(), foreground_reference_points=(), seed_instance_traits=(),
            seed_trait_coat_patterns=(), seed_trait_conditions=(), enabled_nodes=frozenset({'trait_classifier'}),
            dirty_nodes=frozenset(), trait_settings=settings)
        task = _AnalysisTask(**arguments)
        completed, failures = [], []
        task.signals.completed.connect(lambda result, revision: completed.append((result, revision)))
        task.signals.failed.connect(lambda *args: failures.append(args))
        expected = SimpleNamespace()
        with patch('seedvision.segmentation.baseline.analyze_path', return_value=expected) as analyzer:
            task.run()
        self.assertEqual(failures, [])
        self.assertEqual(completed, [(expected, 1)])
        self.assertIs(analyzer.call_args.kwargs['trait_settings'], settings)

    def check_test_inference_mode_tensor_mutations_cannot_reuse_unversioned_features(self):
        with torch.inference_mode():
            source = self.source.clone()
        first, _ = self.run_classifier(source=source)
        features = self.cache['trait_classifier.features']
        with torch.inference_mode():
            source[0, 2, 4:8, 4:8] = 225
        second, computed = self.run_classifier(source=source)
        self.assertTrue(computed)
        self.assertIsNot(features, self.cache['trait_classifier.features'])
        self.assertNotEqual(first.report['predictions'][0]['probabilities'],
                            second.report['predictions'][0]['probabilities'])

    def check_test_gpu_storage_replacement_without_version_increment_invalidates_features(self):
        first, _ = self.run_classifier()
        old = self.cache['trait_classifier.features']
        version = self.source._version
        replacement = self.source.clone()
        replacement[0, 2, 4:8, 4:8] = 225
        self.source.data = replacement
        self.assertEqual(self.source._version, version)
        second, computed = self.run_classifier()
        self.assertTrue(computed)
        self.assertIsNot(old, self.cache['trait_classifier.features'])
        self.assertNotEqual(first.report['predictions'][0]['probabilities'],
                            second.report['predictions'][0]['probabilities'])
        old = self.cache['trait_classifier.features']
        version = self.valid._version
        replacement = self.valid.clone()
        replacement[0, 0, 4, 4] = False
        self.valid.data = replacement
        self.assertEqual(self.valid._version, version)
        rejected, computed = self.run_classifier()
        self.assertTrue(computed)
        self.assertIsNot(old, self.cache['trait_classifier.features'])
        self.assertEqual(rejected.report['predictions'][0]['abstention_reason'], 'outside_valid_crop')

    def check_test_model_growth_after_stat_is_bounded_and_stales_known_export(self):
        import io
        from seedvision.learning.trait_pipeline import _MAX_MODEL_BYTES
        prediction, _ = self.run_classifier()
        original_open = Path.open
        requested = []
        class GrowingFile(io.BytesIO):
            def read(self, size=-1):
                requested.append(size)
                return super().read(size)
        def opened(path, *args, **kwargs):
            if path == self.path:
                return GrowingFile(self.model.to_json().encode('utf-8') + b' ' * _MAX_MODEL_BYTES)
            return original_open(path, *args, **kwargs)
        with patch.object(Path, 'open', opened):
            rejected, _ = self.run_classifier()
            self.assertEqual(rejected.report['model_unavailability_reason'], 'invalid_model')
            self.assertIsNone(rejected.report['model_artifact_sha256'])
            with self.assertRaisesRegex(ValueError, 'model changed'):
                result_report(self.analysis(prediction))
        self.assertEqual(requested, [_MAX_MODEL_BYTES+1, _MAX_MODEL_BYTES+1])

    def check_test_version_twenty_one_profile_adopts_only_disabled_classifier_defaults(self):
        from seedvision.persistence.analysis_settings import (
            analysis_settings_profile_from_graph, analysis_settings_profile_to_payload,
            analysis_settings_profile_from_payload, apply_analysis_settings_profile,
            IncompatibleAnalysisSettingsProfile,
        )
        payload = analysis_settings_profile_to_payload(analysis_settings_profile_from_graph(build_default_pipeline()))
        payload['nodes'] = [node for node in payload['nodes'] if node['id'] != 'trait_classifier']
        payload['connections'] = [edge for edge in payload['connections'] if edge['target'] != 'trait_classifier']
        for version in (20, 21):
            legacy = json.loads(json.dumps(payload))
            legacy['version'] = version
            graph = build_default_pipeline()
            graph.node('trait_classifier').parameters['trait_model_path'] = 'unrelated-current-choice.json'
            apply_analysis_settings_profile(graph, analysis_settings_profile_from_payload(legacy))
            self.assertFalse(graph.node('trait_classifier').enabled)
            self.assertEqual(graph.node('trait_classifier').parameters['trait_model_path'], 'None')
            self.assertEqual(graph.missing_input_ports('trait_classifier'), ())
            legacy['nodes'] = [node for node in legacy['nodes'] if node['id'] != 'deskew_colour']
            with self.assertRaisesRegex(IncompatibleAnalysisSettingsProfile, 'missing nodes'):
                apply_analysis_settings_profile(build_default_pipeline(), analysis_settings_profile_from_payload(legacy))
        with self.assertRaisesRegex(IncompatibleAnalysisSettingsProfile, 'missing nodes'):
            apply_analysis_settings_profile(build_default_pipeline(), analysis_settings_profile_from_payload(payload))

    def check_test_missing_model_becoming_available_requires_reanalysis_before_export(self):
        path = self.root / 'later.json'
        prediction, _ = self.run_classifier(path=path)
        analysis = self.analysis(prediction)
        self.assertEqual(result_report(analysis)['instances'][0]['trait_assignment'], 'abstained: model_file_missing')
        path.write_text(self.model.to_json(), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'became available'):
            result_report(analysis)

    def check_test_saved_project_batch_forwards_model_and_threshold_and_persists_traits(self):
        import shutil
        import cv2
        from seedvision.persistence import ProjectAnalysisStore, ProjectImageSpec, analysis_settings_profile_from_graph
        from seedvision.export.project_runner import run_project
        (self.root / 'config').mkdir()
        shutil.copyfile(Path(__file__).parents[1] / 'config' / 'traits.json', self.root / 'config' / 'traits.json')
        source = self.root / 'capture.png'
        self.assertTrue(cv2.imwrite(str(source), self.corrected))
        graph = build_default_pipeline()
        graph.set_parameter('trait_classifier', 'trait_model_path', str(self.path))
        graph.set_parameter('trait_classifier', 'trait_abstention_threshold', .61)
        graph.set_enabled('trait_classifier', True)
        master = ProjectAnalysisStore(self.root).capture_and_save(
            analysis_settings=analysis_settings_profile_from_graph(graph),
            images=(ProjectImageSpec(source, self.corrected.shape[:2]),),
            destination=self.root / 'test.seedfiddle-project.json')
        prediction, _ = self.run_classifier(threshold=.61)
        analysis = self.analysis(prediction)
        analysis.image_path = source
        with patch('seedvision.export.project_runner.analyze_path', return_value=analysis) as analyzer:
            output = run_project(self.root, master, self.root / 'batch')[0]
        forwarded = analyzer.call_args.kwargs
        self.assertEqual(forwarded['trait_settings'], TraitClassifierSettings(str(self.path), .61))
        self.assertIn('trait_classifier', forwarded['enabled_nodes'])
        stored = json.loads((output / 'results.json').read_text())
        self.assertEqual(stored['trait_classifier']['abstention_threshold'], .61)
        self.assertEqual([row['trait_assignment'] for row in stored['instances']], ['dark', 'light'])

    def check_test_file_analysis_cache_and_crops_use_stored_classifier_revision(self):
        import cv2
        from seedvision.calibration import image as calibration
        from seedvision.calibration.geometry import DishDetectionSettings
        from seedvision.segmentation.baseline import analyze_path, BaselineSettings, PipelineAnalysisCache
        from seedvision.segmentation.procedural import ProceduralInstanceResult
        from seedvision.export.crops import extract_seed_crops
        from seedvision.export.crop_artifacts import export_seed_crop_artifacts, CropArtifactError
        from tests.test_neutral_calibration import fixture as image_fixture, ruler

        image, card = image_fixture()
        source = self.root / 'capture.png'
        self.assertTrue(cv2.imwrite(str(source), image))
        foreground = np.zeros(image.shape[:2], bool)
        foreground[28:36, 44:52] = True
        cache = PipelineAnalysisCache()
        settings = TraitClassifierSettings(str(self.path), 0)
        options = dict(settings=BaselineSettings(),
            calibration_settings=calibration.CalibrationSettings(apply_perspective_correction=False),
            dish_settings=DishDetectionSettings(expected_outer_diameter_mm=0),
            foreground_reference_mask=foreground, enabled_nodes={'procedural_instances', 'trait_classifier'},
            learning_root=self.root, node_cache=cache, trait_settings=settings)

        def decoder(valid, *args, **kwargs):
            height, width = valid.shape[-2:]
            labels = np.zeros((height, width), np.int32)
            labels[height//2-2:height//2+2, width//3-2:width//3+2] = 1
            labels[height//2-2:height//2+2, 2*width//3-2:2*width//3+2] = 2
            zeros = np.zeros(labels.shape, np.float32)
            return ProceduralInstanceResult(labels=labels, centres_xy=np.zeros((2, 2), np.float32),
                marker_scores=np.ones(2, np.float32), instance_confidences=np.ones(2, np.float32),
                occupancy_likelihood=zeros, occupancy_mask=zeros.astype(bool), boundary_cost=zeros,
                centre_likelihood=zeros, source_shape=(height, width), working_scale=1.)

        with (patch.object(calibration, 'detect_colour_card', return_value=card),
              patch.object(calibration, 'detect_ruler', side_effect=ruler),
              patch('seedvision.segmentation.baseline.procedural_seed_instances', side_effect=decoder) as decoded):
            analysis = analyze_path(source, **options)
            features = cache.values['trait_classifier.features']
            self.assertEqual(len(analysis.trait_classifier.report['predictions']), 2)
            self.assertTrue(all(row['probabilities'] is not None for row in analysis.trait_classifier.report['predictions']))
            self.assertTrue(features.values.is_cuda)
            cached = analyze_path(source, **options)
            self.assertIs(analysis.trait_classifier, cached.trait_classifier)
            self.assertIn('trait_classifier', cache.last_reused_nodes)
            options['trait_settings'] = TraitClassifierSettings(str(self.path), 1)
            with patch('seedvision.learning.trait_pipeline.extract_trait_features', side_effect=AssertionError('reextracted')):
                changed = analyze_path(source, dirty_nodes={'trait_classifier'}, **options)
            self.assertEqual(decoded.call_count, 1)
            self.assertIs(cache.values['trait_classifier.features'], features)
        report = result_report(analysis)
        changed_report = result_report(changed)
        self.assertNotEqual(report['result_revision'], changed_report['result_revision'])
        with patch.object(torch.Tensor, 'cpu', side_effect=AssertionError('raster downloaded')):
            crops = extract_seed_crops(analysis, method='procedural_instances', seed_ids=(1,),
                padding_px=0, result_revision=report['result_revision'])
        self.assertEqual(crops.records[0].status, 'available')
        with self.assertRaisesRegex(CropArtifactError, 'Result changed'):
            export_seed_crop_artifacts(self.root / 'stale-crops', changed, report, seed_ids=(1,), padding_px=0)
        self.assertFalse((self.root / 'stale-crops').exists())

    def test_workflow_lifecycle(self):
        """10 related TraitPipelineTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_selected_method_preference_is_shared_and_filters_disabled_branches)
        self.scenario(self.check_test_actual_cuda_resize_colour_features_predict_without_raster_download)
        self.scenario(self.check_test_threshold_only_reuses_features_and_scores_then_identical_run_reuses_result)
        self.scenario(self.check_test_model_content_change_rescores_without_reextracting_features)
        self.scenario(self.check_test_graph_settings_roundtrip_and_authored_dependencies)
        self.scenario(self.check_test_classifier_enables_with_alternative_branches_and_required_inputs_remain_honest)
        self.scenario(self.check_test_native_worker_forwards_classifier_settings)
        self.scenario(self.check_test_version_twenty_one_profile_adopts_only_disabled_classifier_defaults)
        self.scenario(self.check_test_saved_project_batch_forwards_model_and_threshold_and_persists_traits)
        self.scenario(self.check_test_file_analysis_cache_and_crops_use_stored_classifier_revision)

    def test_validation_and_recovery(self):
        """8 related TraitPipelineTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_label_corrected_pixels_and_support_changes_invalidate_features)
        self.scenario(self.check_test_missing_invalid_models_and_frame_boundary_inputs_abstain)
        self.scenario(self.check_test_report_persists_actual_traits_and_threshold_changes_stale_review)
        self.scenario(self.check_test_changed_exported_labels_crop_pixels_or_model_reject_before_report)
        self.scenario(self.check_test_inference_mode_tensor_mutations_cannot_reuse_unversioned_features)
        self.scenario(self.check_test_gpu_storage_replacement_without_version_increment_invalidates_features)
        self.scenario(self.check_test_model_growth_after_stat_is_bounded_and_stales_known_export)
        self.scenario(self.check_test_missing_model_becoming_available_requires_reanalysis_before_export)


if __name__ == '__main__': unittest.main()
