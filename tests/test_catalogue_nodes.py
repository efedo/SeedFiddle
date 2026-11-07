"""Independent catalogue instances, typed projections and portable project state."""


from tests.functional_units import functional_units
from dataclasses import replace
from pathlib import Path
from hashlib import sha256
import tempfile
import unittest
from unittest.mock import patch
import json

from seedvision.pipeline.catalogue_nodes import (catalogue_entries, create_instance, default_graph,
    instance_graph, sync_instance, NodeInstance, node_instances_to_payload,
    node_instances_from_payload, validate_node_instances)
from seedvision.pipeline.model import PipelineGraph, PipelineConnection, build_default_pipeline
from seedvision.persistence.analysis_settings import analysis_settings_profile_from_graph
from seedvision.persistence.project_analysis import ProjectAnalysisStore
from seedvision.persistence.snapshot import create_snapshot, restore_snapshot


@functional_units
class CatalogueNodeTests(unittest.TestCase):
    def check_test_catalogue_lists_each_default_node_once(self):
        entries = catalogue_entries()
        default = build_default_pipeline()
        self.assertEqual(len(entries), 52)
        self.assertEqual(len({entry.key for entry in entries}), len(entries))
        self.assertTrue(all(entry.key == 'default:' + entry.target_node for entry in entries))
        self.assertEqual({entry.target_node for entry in entries}, set(default.nodes) | set(default.unused_nodes))

    def check_test_duplicate_default_instances_keep_independent_controls_scope_and_round_trip(self):
        key = 'default:seed_scale_estimation'
        first, second = create_instance(key), create_instance(key)
        left, right = instance_graph(first), instance_graph(second)
        target = first.instance_id + '::' + first.target_node
        left.node(target).parameters['reference_max_components'] = 7
        left.node(target).x = 123.5
        edited = sync_instance(first, left)
        edited = replace(edited, selected_node=first.target_node, selected_output='seed_scale_estimation')
        self.assertEqual(default_graph(edited).node(first.target_node).parameters['reference_max_components'], 7)
        self.assertEqual(default_graph(second).node(second.target_node).parameters['reference_max_components'], 3)
        self.assertFalse(set(left.nodes) & set(right.nodes))
        self.assertTrue(all(identifier.startswith(first.instance_id + '::') for identifier in left.nodes))
        self.assertLess(len(left.nodes), len(default_graph(first).nodes))
        restored = node_instances_from_payload(node_instances_to_payload((edited, second)))
        self.assertEqual(restored, (edited, second))
        self.assertEqual((restored[0].selected_node, restored[0].selected_output),
                         (first.target_node, 'seed_scale_estimation'))
        self.assertEqual((NodeInstance.from_dict(second.to_dict()).selected_node,
                          NodeInstance.from_dict(second.to_dict()).selected_output), (None, ''))
        self.assertEqual(instance_graph(restored[0]).node(target).x, 123.5)
        self.assertEqual(sync_instance(restored[0], instance_graph(restored[0])), restored[0])

    def check_test_cross_instance_edges_and_duplicate_or_unregistered_descriptors_reject(self):
        first = create_instance('default:seed_scale_estimation')
        second = create_instance('default:seed_scale_estimation')
        left, right = instance_graph(first), instance_graph(second)
        merged = PipelineGraph((), ())
        merged.nodes = {**left.nodes, **right.nodes}
        merged.connections = (*left.connections, *right.connections,
            PipelineConnection(next(iter(left.nodes)), next(iter(right.nodes)), 'RawImage', 'image', 'image'))
        with self.assertRaisesRegex(ValueError, 'explicit adapter'):
            sync_instance(first, merged)
        with self.assertRaisesRegex(ValueError, 'unique'):
            validate_node_instances((first, first))
        with self.assertRaises(ValueError):
            NodeInstance.from_dict(first.to_dict() | {'catalogue_key': 'default:invented'})
        with self.assertRaises(ValueError):
            NodeInstance.from_dict(first.to_dict() | {'workflow': None})
        with self.assertRaises(ValueError):
            replace(first, selected_node='foreign_node')
        with self.assertRaises(ValueError):
            replace(first, selected_output='x' * 257)

    def check_test_duplicate_default_instances_survive_project_and_snapshot_reopen(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = ProjectAnalysisStore(root)
            instances = (create_instance('default:seed_scale_estimation'),
                         create_instance('default:seed_scale_estimation'))
            document = store.capture(analysis_settings=analysis_settings_profile_from_graph(build_default_pipeline()),
                                     images=(), node_instances=instances)
            master = store.save(document, root / 'private.seedfiddle-project.json')
            self.assertEqual(store.load(master).document.node_instances, instances)
            snapshot = create_snapshot(root, master, root / 'snapshots')
            restored = restore_snapshot(snapshot, root / 'restored')
            self.assertEqual(ProjectAnalysisStore(restored.parent).load(restored).document.node_instances, instances)
            # Batch orchestration must retain two duplicate rows and separate
            # caches without needing numerical analysis for this contract.
            from seedvision.pipeline import catalogue_nodes as backend
            from seedvision.export import results
            from seedvision.export.node_instances import export_project_node_instances
            source = root / 'batch-source.bin'
            source.write_bytes(b'private batch identity')
            calls = []
            reviewed_keys = ('background_reference_mask', 'foreground_reference_mask',
                'background_exclusion_mask', 'foreground_exclusion_mask', 'seed_instance_annotations',
                'seed_instance_traits', 'reference_transform', 'reference_archive_source_sha256',
                'reference_annotation_snapshot_source_sha256', 'manual_seed_centres', 'seed_trait_species',
                'seed_trait_coat_patterns', 'seed_trait_conditions', 'learning_root', 'species',
                'biological_context', 'species_library', 'library_shape_bank')
            reviewed = {key: object() for key in reviewed_keys}
            private = {key: object() for key in ('settings', 'layer_settings', 'enabled_nodes',
                'node_cache', 'dirty_nodes', 'cancellation_requested', 'background_colour_enabled')}
            context = backend.instance_default_context(reviewed | private)
            self.assertEqual(set(context), set(reviewed))
            def execute(instance, path, **options):
                calls.append((instance.instance_id, options['cache']))
                self.assertEqual(set(options['default_context']), set(reviewed))
                for key, value in reviewed.items():
                    self.assertIs(options['default_context'][key], value)
                return object()
            def publish(directory, result, report):
                directory.mkdir()
                (directory / 'results.json').write_text('{}')
                return directory
            with patch.object(backend, 'execute_instance', side_effect=execute), \
                    patch.object(results, 'result_report', return_value={'state': 'available_for_review'}), \
                    patch.object(results, 'export_report', side_effect=publish):
                ledger = export_project_node_instances(instances, source, root / 'batch',
                    expected_source_sha256=sha256(source.read_bytes()).hexdigest(), default_context=context)
            rows = json.loads(ledger.read_text())['instances']
            self.assertEqual([row['instance_id'] for row in rows], [instance.instance_id for instance in instances])
            self.assertTrue(all(row['report'] is not None for row in rows))
            self.assertIsNot(calls[0][1], calls[1][1])

    def test_workflow_lifecycle(self):
        """3 related CatalogueNodeTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_catalogue_lists_each_default_node_once)
        self.scenario(self.check_test_duplicate_default_instances_keep_independent_controls_scope_and_round_trip)
        self.scenario(self.check_test_duplicate_default_instances_survive_project_and_snapshot_reopen)

    def test_validation_and_recovery(self):
        """1 related CatalogueNodeTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_cross_instance_edges_and_duplicate_or_unregistered_descriptors_reject)
