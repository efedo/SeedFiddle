"""Actual shape-update outcomes persist and bind the authoritative result revision."""

from tests.functional_units import functional_units
from dataclasses import asdict, replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import cv2

from seedvision.export.results import export_report, result_report
from seedvision.measurement import shape_model as model
from seedvision.measurement.lineage import facts_sha256
from tests.test_local_shape_units import measurements
from tests.test_reviewed_measurement_lineage import analysis_fixture
from tests.test_shape_update_availability import fixture, update


def analysis_with_outcome(outcome, *, summary=None):
    analysis = analysis_fixture()
    if summary is not None:
        analysis.seed_measurement_summary = summary
    family = model.SeedPoseShapeFamily('flat', outcome.component, 2, outcome.source,
        prior_probability=1., out_of_family_distance=5., update_state=outcome.state,
        update_unavailability_reason=outcome.unavailability_reason,
        pooling_method=outcome.pooling_method, conditioning_scales=outcome.conditioning_scales)
    analysis.seed_dimensions_shape_model = model.SeedDimensionsShapeModel((family,),
        analysis.seed_measurement_summary, 'species_library_prior_and_current', ('soybean',), 24., 24.,
        in_sample_local_update=outcome.state in {'applied', 'local_only'})
    return analysis


def conditioned_analysis():
    values = measurements(scale=4.)
    prior = model._local_component(values, 'flat')
    outcome = model._update_component_result(prior, values, 'flat',
        shrinkage_seed_count=3., maximum_contour_modes=0)
    assert outcome.pooling_method == 'coordinate_conditioned_precision'
    return analysis_with_outcome(outcome)


@functional_units
class ShapeModelExportTests(unittest.TestCase):
    def check_test_actual_file_analysis_and_cached_model_export_without_cuda_transfer(self):
        import torch
        from seedvision.calibration import image as calibration
        from seedvision.calibration.geometry import DishDetectionSettings
        from seedvision.segmentation.baseline import analyze_path, BaselineSettings, PipelineAnalysisCache
        from tests.test_neutral_calibration import fixture as image_fixture, ruler
        from tests.test_reviewed_measurement_lineage import annotation, labels_fixture
        image, card = image_fixture()
        with TemporaryDirectory() as temporary:
            source = Path(temporary) / 'capture.png'
            self.assertTrue(cv2.imwrite(str(source), image))
            cache = PipelineAnalysisCache()
            options = dict(settings=BaselineSettings(shape_minimum_component_seeds=1),
                calibration_settings=calibration.CalibrationSettings(apply_perspective_correction=False),
                dish_settings=DishDetectionSettings(expected_outer_diameter_mm=0), enabled_nodes=set(),
                seed_instance_annotations=labels_fixture(), seed_instance_traits=(annotation(),),
                reference_transform=np.eye(3), node_cache=cache)
            with (patch.object(calibration, 'detect_colour_card', return_value=card),
                patch.object(calibration, 'detect_ruler', side_effect=ruler)):
                analysis = analyze_path(source, **options)
                cached = analyze_path(source, **options)
            self.assertIs(analysis.seed_dimensions_shape_model, cached.seed_dimensions_shape_model)
            self.assertIn('seed_scale_estimation', cache.last_reused_nodes)
            with patch.object(torch.Tensor, 'cpu', side_effect=AssertionError('CUDA transfer')):
                report = result_report(cached)
            self.assertEqual(report['shape_model']['reviewed_summary_sha256'],
                facts_sha256(report['reviewed_measurements']))
            self.assertEqual(report['shape_model']['families'][0]['update_state'], 'not_attempted')
            self.assertEqual(analysis.calibration.gpu_corrected_bgr.device.type, 'cuda')

    def check_test_absent_model_remains_absent_without_invented_diagnostics(self):
        report = result_report(analysis_fixture())
        self.assertNotIn('shape_model', report)
        self.assertIsNone(report['identity']['shape_model'])

    def check_test_actual_local_assembly_serializes_without_refitting_or_writing_library(self):
        analysis = analysis_fixture()
        analysis.seed_dimensions_shape_model = model.assemble_seed_dimensions_shape_model(
            local_summary=analysis.seed_measurement_summary, legacy_processing_diameter_px=24.,
            source_mode='current_image_only', minimum_component_seeds=1)
        with (patch.object(model, '_local_component', side_effect=AssertionError('refitted')),
            patch.object(model, 'fit_species_dimensions_shape_bank', side_effect=AssertionError('refitted')),
            patch('seedvision.reference_library.persistence.SpeciesLibraryStore.publish', side_effect=AssertionError('published'))):
            report = result_report(analysis)
        snapshot = report['shape_model']
        self.assertEqual(snapshot['families'][0]['update_state'], 'not_attempted')
        self.assertEqual(snapshot['families'][0]['source'], 'local')
        self.assertEqual(snapshot['reviewed_summary_sha256'], facts_sha256(report['reviewed_measurements']))
        self.assertFalse(snapshot['scientifically_validated'])
        self.assertFalse(snapshot['physical_measurements_eligible'])

    def check_test_conditioned_actual_mask_outcome_roundtrips_statistics_scales_and_support(self):
        analysis = conditioned_analysis()
        family = analysis.seed_dimensions_shape_model.families[0]
        before = asdict(family.component)
        with TemporaryDirectory() as temporary:
            report = result_report(analysis)
            destination = export_report(Path(temporary), analysis, report)
            stored = json.loads((destination / 'results.json').read_text(encoding='utf-8'))
        self.assertEqual(stored['shape_model'], stored['identity']['shape_model'])
        saved = stored['shape_model']['families'][0]
        self.assertEqual(saved['pooling_method'], 'coordinate_conditioned_precision')
        self.assertEqual(saved['conditioning_scales'], list(family.conditioning_scales))
        self.assertEqual(saved['component'], json.loads(json.dumps(before)))
        self.assertEqual(asdict(family.component), before)

    def check_test_ordinary_mixed_local_only_and_unavailable_actual_outcomes_roundtrip(self):
        values, prior = fixture()
        _, local = fixture(covariance=2.)
        _, dimensionless = fixture(False)
        singular = replace(prior, covariance=tuple(tuple(row) for row in np.zeros((19, 19))))
        outcomes = [update(prior, local, values), update(dimensionless, local, values),
            update(None, local, values), update(singular, local, values)]
        self.assertEqual([x.state for x in outcomes], ['applied', 'applied', 'local_only', 'unavailable'])
        for outcome in outcomes:
            with self.subTest(state=outcome.state, method=outcome.pooling_method):
                saved = result_report(analysis_with_outcome(outcome))['shape_model']['families'][0]
                self.assertEqual(saved['update_state'], outcome.state)
                self.assertEqual(saved['pooling_method'], outcome.pooling_method)
                self.assertEqual(saved['update_unavailability_reason'], outcome.unavailability_reason)
                self.assertEqual(saved['source'], outcome.source)
                self.assertEqual(saved['component'], json.loads(json.dumps(asdict(outcome.component))))

    def check_test_legacy_unknown_update_and_missing_summary_binding_remain_unknown(self):
        analysis = conditioned_analysis()
        family = replace(analysis.seed_dimensions_shape_model.families[0], update_state=None,
            pooling_method=None, conditioning_scales=())
        analysis.seed_dimensions_shape_model = replace(analysis.seed_dimensions_shape_model,
            families=(family,), measurement_summary=None)
        saved = result_report(analysis)['shape_model']
        self.assertEqual(saved['reviewed_summary_binding_state'], 'legacy_not_recorded')
        self.assertIsNone(saved['reviewed_summary_sha256'])
        self.assertEqual(saved['families'][0]['update_diagnostic_state'], 'legacy_not_recorded')
        self.assertIsNone(saved['families'][0]['update_state'])
        self.assertIsNone(saved['families'][0]['pooling_method'])
        self.assertIsNone(saved['families'][0]['update_unavailability_reason'])

    def check_test_equal_detached_summary_passes_but_changed_measurement_or_eligibility_rejects(self):
        analysis = conditioned_analysis()
        summary = analysis.seed_measurement_summary
        analysis.seed_dimensions_shape_model = replace(analysis.seed_dimensions_shape_model,
            measurement_summary=replace(summary))
        self.assertEqual(result_report(analysis)['shape_model']['reviewed_summary_binding_state'], 'recorded')
        observation = summary.observations[0]
        variants = [replace(summary, eligible_count=summary.eligible_count+1),
            replace(summary, observations=(replace(observation, eligible=False),)),
            replace(summary, observations=(replace(observation, measurement=replace(
                observation.measurement, maximum_span=observation.measurement.maximum_span+1)),))]
        for other in variants:
            with self.subTest(summary=other.eligible_count), self.assertRaisesRegex(ValueError, 'measurements changed'):
                analysis.seed_dimensions_shape_model = replace(analysis.seed_dimensions_shape_model, measurement_summary=other)
                result_report(analysis)

    def check_test_changed_known_model_lineage_rejects_before_result_persistence(self):
        analysis = conditioned_analysis()
        summary = analysis.seed_measurement_summary
        stale = replace(summary, lineage=replace(summary.lineage, corrected_image_sha256='a'*64))
        analysis.seed_dimensions_shape_model = replace(analysis.seed_dimensions_shape_model, measurement_summary=stale)
        with self.assertRaisesRegex(ValueError, 'corrected_image_sha256 changed'):
            result_report(analysis)

    def check_test_only_model_facts_changing_invalidates_previously_saved_instance_review(self):
        analysis = conditioned_analysis()
        original = analysis.seed_dimensions_shape_model
        family = original.families[0]
        baseline = result_report(analysis)
        review = {'result_revision': baseline['result_revision'], 'decisions': {'1': 'accepted'}}
        self.assertEqual(result_report(analysis, review=review)['accepted_count'], 1)
        covariance = np.asarray(family.component.covariance)*1.01
        replacements = [replace(original, warnings=('numeric diagnostic',)),
            replace(original, in_sample_local_update=False),
            replace(original, selected_hierarchy_path=('soybean', 'lot:heldout')),
            replace(original, families=(replace(family, local_observation_count=3),)),
            replace(original, families=(replace(family, pooling_method='legacy_precision', conditioning_scales=()),)),
            replace(original, families=(replace(family, conditioning_scales=tuple(x*2 for x in family.conditioning_scales)),)),
            replace(original, families=(replace(family, component=replace(family.component,
                covariance=tuple(tuple(row) for row in covariance))),))]
        for other in replacements:
            analysis.seed_dimensions_shape_model = other
            changed = result_report(analysis, review=review)
            self.assertNotEqual(changed['result_revision'], baseline['result_revision'])
            self.assertEqual(changed['accepted_count'], 0)

    def check_test_update_reason_and_physical_fit_reason_are_independent_revision_facts(self):
        values, prior = fixture(False)
        prior = replace(prior, covariance=tuple(tuple(row) for row in np.zeros((19, 19))),
            physical_fit_unavailability_reason='numeric_range', calibrated_physical_seed_count=7.)
        outcome = update(prior, prior, values)
        analysis = analysis_with_outcome(outcome)
        saved = result_report(analysis)
        family = saved['shape_model']['families'][0]
        self.assertEqual(family['update_unavailability_reason'], 'singular_precision')
        self.assertEqual(family['component']['physical_fit_unavailability_reason'], 'numeric_range')
        self.assertEqual(family['component']['calibrated_physical_seed_count'], 7.)
        original = analysis.seed_dimensions_shape_model
        analysis.seed_dimensions_shape_model = replace(original, families=(replace(original.families[0],
            update_unavailability_reason='numeric_range'),))
        self.assertNotEqual(result_report(analysis)['result_revision'], saved['result_revision'])

    def check_test_conditioned_scale_domains_and_contradictory_diagnostics_reject(self):
        analysis = conditioned_analysis()
        original = analysis.seed_dimensions_shape_model
        family = original.families[0]
        bad = [replace(family, conditioning_scales=x) for x in (
            (), (1.,)*18, (0.,)*19, (-1.,)*19, (float('inf'),)*19, (float('nan'),)*19, (True,)*19)]
        bad += [replace(family, update_state=None), replace(family, update_state='unavailable'),
            replace(family, pooling_method='unsupported'), replace(family, source='library'),
            replace(family, update_unavailability_reason='numeric_range')]
        for other in bad:
            with self.subTest(state=other.update_state, method=other.pooling_method), self.assertRaises(ValueError):
                analysis.seed_dimensions_shape_model = replace(original, families=(other,))
                result_report(analysis)

    def check_test_mutable_nested_runtime_data_cannot_change_an_assembled_json_snapshot(self):
        analysis = conditioned_analysis()
        original = analysis.seed_dimensions_shape_model
        family = original.families[0]
        mean = list(family.component.mean)
        covariance = [list(row) for row in family.component.covariance]
        other = replace(family, component=replace(family.component, mean=mean, covariance=covariance))
        analysis.seed_dimensions_shape_model = replace(original, families=(other,))
        report = result_report(analysis)
        before = json.dumps(report, sort_keys=True)
        mean[0] += 1; covariance[0][0] += 1
        self.assertEqual(json.dumps(report, sort_keys=True), before)
        self.assertNotEqual(result_report(analysis)['result_revision'], report['result_revision'])

    def check_test_nonfinite_counts_diameters_and_mismatched_pose_are_not_persisted(self):
        analysis = conditioned_analysis()
        original = analysis.seed_dimensions_shape_model
        family = original.families[0]
        bad = [replace(original, compatibility_seed_diameter_px=float('inf')),
            replace(original, legacy_processing_diameter_px=0),
            replace(original, families=(replace(family, local_observation_count=True),)),
            replace(original, families=(replace(family, component=replace(family.component,
                effective_physical_seed_count=float('nan'))),)),
            replace(original, families=(replace(family, pose='side'),)),
            replace(original, families=(family, family))]
        for other in bad:
            with self.assertRaises(ValueError):
                analysis.seed_dimensions_shape_model = other
                result_report(analysis)

    def check_test_shape_diagnostic_only_change_rejects_stale_crop_export_before_output(self):
        from seedvision.export.crop_artifacts import export_seed_crop_artifacts
        from tests.test_seed_crop_artifacts import _analysis as crop_analysis
        analysis = crop_analysis()
        current = conditioned_analysis()
        analysis = replace(analysis, seed_dimensions_shape_model=replace(current.seed_dimensions_shape_model,
            measurement_summary=None))
        report = result_report(analysis)
        analysis = replace(analysis, seed_dimensions_shape_model=replace(analysis.seed_dimensions_shape_model,
            warnings=('changed local update history',)))
        with TemporaryDirectory() as temporary:
            output = Path(temporary) / 'output'
            with self.assertRaisesRegex(ValueError, 'Result changed'):
                export_seed_crop_artifacts(output, analysis, report, seed_ids=(1,), padding_px=0)
            self.assertFalse(output.exists())

    def test_workflow_lifecycle(self):
        """7 related ShapeModelExportTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_actual_file_analysis_and_cached_model_export_without_cuda_transfer)
        self.scenario(self.check_test_absent_model_remains_absent_without_invented_diagnostics)
        self.scenario(self.check_test_actual_local_assembly_serializes_without_refitting_or_writing_library)
        self.scenario(self.check_test_conditioned_actual_mask_outcome_roundtrips_statistics_scales_and_support)
        self.scenario(self.check_test_update_reason_and_physical_fit_reason_are_independent_revision_facts)
        self.scenario(self.check_test_mutable_nested_runtime_data_cannot_change_an_assembled_json_snapshot)
        self.scenario(self.check_test_nonfinite_counts_diameters_and_mismatched_pose_are_not_persisted)

    def test_validation_and_recovery(self):
        """7 related ShapeModelExportTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_ordinary_mixed_local_only_and_unavailable_actual_outcomes_roundtrip)
        self.scenario(self.check_test_legacy_unknown_update_and_missing_summary_binding_remain_unknown)
        self.scenario(self.check_test_equal_detached_summary_passes_but_changed_measurement_or_eligibility_rejects)
        self.scenario(self.check_test_changed_known_model_lineage_rejects_before_result_persistence)
        self.scenario(self.check_test_only_model_facts_changing_invalidates_previously_saved_instance_review)
        self.scenario(self.check_test_conditioned_scale_domains_and_contradictory_diagnostics_reject)
        self.scenario(self.check_test_shape_diagnostic_only_change_rejects_stale_crop_export_before_output)


if __name__ == '__main__':
    unittest.main()
