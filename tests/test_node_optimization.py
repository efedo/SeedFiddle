from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from seedvision.pipeline import build_default_pipeline
from seedvision.optimization.registry import OBJECTIVES, EXEMPT, capability, candidate_values, affected_nodes
from seedvision.optimization.search import optimize, apply_report
from seedvision.optimization.runtime import (Sample, ProductionEvaluator, eligible_instances,
    production_settings, raster_loss, load_targets, array_digest, write_record)
from seedvision.persistence.reference_regions import SeedInstanceAnnotation


class FakeEvaluator:
    mode = 'test adaptation'
    provenance = {'objective_version': 'test'}

    def __init__(self):
        self.seen = []
        self.checks = 0

    def check_snapshot(self):
        self.checks += 1

    def parameters(self, graph, identifier, names):
        return names, {}

    def score(self, graph, identifier):
        value = graph.node(identifier).parameters['wavelet_level_count']
        self.seen.append(value)
        return {'image-a': float((value-2)**2), 'image-b': float((value-4)**2)}

    def accept(self, graph, identifier):
        pass

    def final_scores(self, graph):
        return {'image-a': 1., 'image-b': 1.}


class OptimizationContractTests(unittest.TestCase):
    def test_every_card_and_control_has_a_disposition(self):
        graph = build_default_pipeline()
        nodes = {**graph.nodes, **graph.unused_nodes}
        self.assertEqual(set(nodes), set(OBJECTIVES) | set(EXEMPT))
        for node in nodes.values():
            contract = capability(node)
            self.assertEqual(set(node.parameters), set(contract.searchable) | {key for key, _ in contract.fixed})
            self.assertFalse(set(contract.searchable) & {key for key, _ in contract.fixed})
        proc = capability(graph.node('procedural_instances'))
        self.assertIn('candidate_hypotheses_per_marker', proc.searchable)
        self.assertIn('centre_validated_oval_weight', proc.searchable)
        self.assertNotIn('reference_error_overreach_weight', proc.searchable)
        self.assertNotIn('reference_texture_weight', proc.searchable)

    def test_integer_boolean_and_categorical_candidates_are_typed(self):
        graph = build_default_pipeline()
        for node in graph.nodes.values():
            for spec in node.parameter_specs:
                if spec.key not in capability(node).searchable:
                    continue
                for value in candidate_values(spec, node.parameters[spec.key]):
                    self.assertEqual(node.validated_parameter_value(spec.key, value), value)
                    if spec.kind == 'int':
                        self.assertIs(type(value), int)

    def test_edge_controls_invalidate_true_prototype_owner(self):
        graph = build_default_pipeline()
        affected = affected_nodes(graph, 'reference_edge_probability')
        self.assertIn('reference_texture_prototypes', affected)
        self.assertIn('procedural_instances', affected)
        self.assertNotIn('ruler_detection', affected)

    def test_shared_proposal_optimizes_both_images_not_last_image(self):
        graph = build_default_pipeline()
        graph.node('wavelet_decomposition').set_parameter('wavelet_level_count', 4)
        evaluator = FakeEvaluator()
        report = optimize(graph, evaluator, selected={'wavelet_decomposition'})
        self.assertEqual(graph.node('wavelet_decomposition').parameters['wavelet_level_count'], 4)
        self.assertEqual(report.changes['wavelet_decomposition']['wavelet_level_count'], 3)
        row = next(row for row in report.nodes if row.node_id == 'wavelet_decomposition')
        self.assertEqual(set(row.before), {'image-a', 'image-b'})
        self.assertEqual(row.after, {'image-a': 1., 'image-b': 1.})
        apply_report(graph, report)
        self.assertEqual(graph.node('wavelet_decomposition').parameters['wavelet_level_count'], 3)
        with self.assertRaisesRegex(ValueError, 'stale'):
            apply_report(graph, report)

    def test_cancellation_never_installs_partial_changes(self):
        graph = build_default_pipeline()
        evaluator = FakeEvaluator()
        report = optimize(graph, evaluator, selected={'wavelet_decomposition'}, cancelled=lambda: len(evaluator.seen) > 1)
        self.assertTrue(report.cancelled)
        self.assertFalse(report.changes)
        with self.assertRaises(ValueError):
            apply_report(graph, report)

    def test_candidate_cannot_win_by_dropping_an_image(self):
        graph = build_default_pipeline()
        evaluator = FakeEvaluator()
        original = evaluator.score
        def score(graph, key):
            scores = original(graph, key)
            return scores if len(evaluator.seen) == 1 else {'image-a': 0.}
        evaluator.score = score
        report = optimize(graph, evaluator, selected={'wavelet_decomposition'})
        self.assertFalse(report.changes)
        row = next(row for row in report.nodes if row.node_id == 'wavelet_decomposition')
        self.assertTrue(any('invalid' in trial for trial in row.trials))

    def test_complete_sweep_does_not_spend_budget_on_fixed_values(self):
        graph = build_default_pipeline()
        evaluator = FakeEvaluator()
        report = optimize(graph, evaluator, selected={'wavelet_decomposition'}, maximum_evaluations=1)
        row = next(row for row in report.nodes if row.node_id == 'wavelet_decomposition')
        self.assertIn('Budget exhausted', row.reason)
        self.assertEqual(len(row.trials), 1)

    def test_downstream_search_observes_accepted_upstream_values(self):
        graph = build_default_pipeline()
        graph.node('wavelet_decomposition').set_parameter('wavelet_level_count', 4)
        evaluator = FakeEvaluator()
        downstream_seen = []
        original = evaluator.score
        def score(current, identifier):
            if identifier == 'wavelet_decomposition':
                return original(current, identifier)
            upstream = current.node('wavelet_decomposition').parameters['wavelet_level_count']
            downstream_seen.append(upstream)
            value = current.node(identifier).parameters['edge_blur_sigma']
            return {'image-a': (value - upstream*.1)**2, 'image-b': (value - upstream*.1)**2}
        evaluator.score = score
        evaluator.parameters = lambda current, key, names: ((('edge_blur_sigma',) if key == 'edge_gradients' else names), {})
        optimize(graph, evaluator, selected={'edge_gradients', 'wavelet_decomposition'})
        self.assertTrue(downstream_seen)
        self.assertEqual(set(downstream_seen), {3})
        self.assertEqual(graph.node('wavelet_decomposition').parameters['wavelet_level_count'], 4)

    def test_settings_match_controller_parameter_owners(self):
        graph = build_default_pipeline()
        graph.node('procedural_instances').set_parameter('trace_minimum_length_fraction', .72)
        graph.node('edge_traces').set_parameter('trace_minimum_length_fraction', .4)
        settings = production_settings(graph)
        self.assertEqual(settings['procedural_settings'].trace_minimum_length_fraction, .72)
        self.assertEqual(settings['layer_settings'].trace_minimum_length_fraction, .4)


class OptimizationReferenceTests(unittest.TestCase):
    def test_only_complete_reviewed_connected_shapes_are_eligible(self):
        labels = np.zeros((20,20), np.uint16)
        labels[2:6,2:6] = 1
        labels[8:11,8:11] = 2
        labels[12:14,12:14] = 3
        labels[16:18,16:18] = 3
        annotations = (SeedInstanceAnnotation(1, shape_reviewed=True, outline_visibility='complete'),
                       SeedInstanceAnnotation(2, shape_reviewed=True, outline_visibility='partly_occluded'),
                       SeedInstanceAnnotation(3, shape_reviewed=True, outline_visibility='complete'))
        result = eligible_instances(labels, annotations)
        self.assertEqual(set(np.unique(result)), {0,1})
        self.assertFalse(result.flags.writeable)

    def test_raster_reduction_does_not_download_gpu_raster(self):
        import torch
        from seedvision.cuda.ops import GpuRaster
        raster = GpuRaster(torch.full((1,1,4,4), 204., dtype=torch.float32))
        target = np.full((8,8), np.nan, np.float32)
        target[2:6,2:6] = 1
        result = SimpleNamespace(crop_offset=(2,2), layers=SimpleNamespace(valid_mask=np.zeros((4,4))))
        value = raster_loss(raster, target, result)
        self.assertAlmostEqual(value, -np.log(.8), places=5)
        self.assertEqual(raster.download_count, 0)

    def test_target_file_is_source_and_transform_bound(self):
        with TemporaryDirectory() as directory:
            source = Path(directory)/'source'
            source.write_bytes(b'fixture')
            sample = Sample('a', source, {}, (4,4), np.eye(3))
            path = Path(directory)/'targets.npz'
            np.savez(path, source_sha256=sample.fingerprint,
                     transform_sha256=array_digest(sample.transform),
                     image_quality__sensor_noise=np.ones((4,4)))
            self.assertIn('image_quality__sensor_noise', load_targets(path, sample))
            sample.transform[0,2] = 1
            with self.assertRaisesRegex(ValueError, 'transform'):
                load_targets(path, sample)

    def test_atomic_journal_refuses_nonfinite_values_without_overwrite(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)/'record.json'
            write_record(path, {'value': 1.})
            before = path.read_bytes()
            with self.assertRaises(ValueError):
                write_record(path, {'value': float('nan')})
            self.assertEqual(path.read_bytes(), before)

    def test_failed_candidate_cache_is_discarded(self):
        graph = build_default_pipeline()
        with TemporaryDirectory() as directory:
            source = Path(directory)/'source'
            source.write_bytes(b'fixture')
            sample = Sample('a', source, {}, (4,4), np.eye(3))
            evaluator = ProductionEvaluator(graph, [sample])
            with patch('seedvision.segmentation.baseline.analyze_path', side_effect=ValueError('invalid candidate')):
                with self.assertRaises(ValueError):
                    evaluator.run(graph, sample)
            self.assertNotIn('a', evaluator.caches)
            self.assertNotIn('a', evaluator.snapshots)

    def test_proposal_matching_uses_one_to_one_bounded_matches(self):
        labels = np.zeros((40,40),np.uint16)
        labels[5:10,5:10] = 1
        labels[25:30,25:30] = 2
        sample = SimpleNamespace(labels=labels,seed_diameter=10,complete=True)
        result = SimpleNamespace(proposals=(SimpleNamespace(center_x=7,center_y=7,confidence=1.),
                                            SimpleNamespace(center_x=27,center_y=27,confidence=1.)))
        evaluator = object.__new__(ProductionEvaluator)
        self.assertEqual(evaluator.proposal_loss(sample,result),0.)
        self.assertEqual(evaluator.reference_proposal_map(sample,result),{1:1,2:2})
        result.proposals = result.proposals[:1]
        self.assertGreater(evaluator.proposal_loss(sample,result),0.)

    def test_gradient_ceiling_objective_scores_its_actual_output(self):
        evaluator = object.__new__(ProductionEvaluator)
        evaluator.targets = {('a','boundary'): np.ones((4,4),np.float32)}
        sample = SimpleNamespace(key='a',transform=np.eye(3))
        output = object()
        result = SimpleNamespace(layers=SimpleNamespace(weak_lightening_surface_gradient=output))
        with patch('seedvision.optimization.runtime.raster_loss',return_value=.25) as loss:
            self.assertEqual(evaluator.score_result('lightening_gradient_ceiling',sample,result),.25)
        self.assertIs(loss.call_args.args[0],output)

    def test_scalar_scores_are_json_serializable(self):
        import json
        graph = build_default_pipeline()
        with TemporaryDirectory() as directory:
            source = Path(directory)/'source'
            source.write_bytes(b'fixture')
            sample = Sample('a',source,{},(4,4),np.eye(3),custom={'image_quality__sensor_noise':np.ones((4,4))})
            evaluator = ProductionEvaluator(graph,[sample])
            with patch.object(evaluator,'run',return_value=None), patch.object(evaluator,'score_result',return_value=np.float32(.25)):
                self.assertEqual(json.loads(json.dumps(evaluator.score(graph,'image_quality'))),{'a':.25})

    def test_candidate_cache_repair_keeps_unrelated_nodes_clean(self):
        graph = build_default_pipeline()
        with TemporaryDirectory() as directory:
            source = Path(directory)/'source'
            source.write_bytes(b'fixture')
            sample = Sample('a', source, {}, (4,4), np.eye(3))
            evaluator = ProductionEvaluator(graph, [sample])
            result = SimpleNamespace(calibration=SimpleNamespace(affine_matrix=np.eye(3)))
            with patch('seedvision.segmentation.baseline.analyze_path', return_value=result) as analyze:
                evaluator.run(graph, sample)
                graph.node('procedural_instances').set_parameter('foreground_threshold_scale', .7)
                evaluator.run(graph, sample)
                dirty = analyze.call_args.kwargs['dirty_nodes']
                self.assertIn('procedural_instances', dirty)
                self.assertNotIn('reference_texture_prototypes', dirty)
                self.assertNotIn('ruler_detection', dirty)
                graph.node('procedural_instances').set_parameter('foreground_threshold_scale', .8)
                evaluator.run(graph, sample)
                self.assertIn('procedural_instances', analyze.call_args.kwargs['dirty_nodes'])

    def test_inactive_semantic_trials_are_explicitly_excluded(self):
        graph = build_default_pipeline()
        with TemporaryDirectory() as directory:
            source = Path(directory)/'source'
            source.write_bytes(b'fixture')
            labels = np.zeros((8,8),np.uint16)
            labels[2:6,2:6] = 1
            sample = Sample('a', source, {'seed_instance_traits': (SeedInstanceAnnotation(1, shape_reviewed=True, outline_visibility='complete'),)}, (8,8), np.eye(3), labels=labels)
            evaluator = ProductionEvaluator(graph, [sample])
            with patch.object(evaluator, 'run', return_value=SimpleNamespace(layers=SimpleNamespace())):
                included, excluded = evaluator.parameters(graph, 'procedural_instances', capability(graph.node('procedural_instances')).searchable)
            for name in ('boundary_semantic_floor','boundary_nonphysical_discount','boundary_physical_ridge_weight','boundary_trace_weight'):
                self.assertNotIn(name, included)
                self.assertIn('No nonzero', excluded[name])


if __name__ == '__main__':
    unittest.main()
