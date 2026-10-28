"""Independent catalogue instances, typed projections and portable project state."""

from tests.catalogue_fixtures import CatalogueTestCase

from tests.functional_units import functional_units
from dataclasses import replace
from pathlib import Path
from hashlib import sha256
import tempfile
import unittest
from unittest.mock import patch
import json

from seedvision.pipeline.catalogue_nodes import (catalogue_entries, create_instance,
    instance_graph, sync_instance, NodeInstance, node_instances_to_payload,
    node_instances_from_payload, validate_node_instances)
from seedvision.pipeline.model import PipelineGraph, PipelineConnection, build_default_pipeline
from seedvision.persistence.analysis_settings import analysis_settings_profile_from_graph
from seedvision.persistence.project_analysis import ProjectAnalysisStore
from seedvision.persistence.snapshot import create_snapshot, restore_snapshot


@functional_units
class CatalogueNodeTests(CatalogueTestCase):
    def check_test_catalogue_includes_default_smpts_models_and_four_input_correction(self):
        entries = catalogue_entries()
        self.assertEqual(sum(entry.backend == 'default' for entry in entries), 51)
        self.assertEqual(len({entry.key for entry in entries}), len(entries))
        operations = {entry.recipe.stage(entry.target_node).operation_id
                      for entry in entries if entry.recipe is not None}
        self.assertTrue({'smpts_paper.mean', 'smpts_paper.crops',
            'external.plantcv.naive_bayes_pdf', 'external.plantcv.kmeans_centres'} <= operations)
        self.assertTrue(any(entry.recipe is not None and
            entry.recipe.recipe_id == 'plantcv_color_card_correct_color' for entry in entries))

    def check_test_duplicate_native_thresholds_keep_independent_controls_scope_and_round_trip(self):
        entry = next(entry for entry in catalogue_entries() if entry.recipe is not None
                     and entry.recipe.stage(entry.target_node).operation_id == 'threshold.binary')
        first, second = create_instance(entry.key), create_instance(entry.key)
        left, right = instance_graph(first), instance_graph(second)
        target = first.instance_id + '::' + first.target_node
        left.node(target).parameters['threshold'] = 91
        left.node(target).x = 123.5
        edited = sync_instance(first, left)
        edited = replace(edited, selected_node=first.target_node, selected_output='mask')
        self.assertEqual(edited.workflow.recipe.parameters_for_runner(first.target_node)['threshold'], 91)
        self.assertIsNone(second.workflow.recipe.parameters_for_runner(second.target_node)['threshold'])
        self.assertFalse(set(left.nodes) & set(right.nodes))
        self.assertTrue(all(identifier.startswith(first.instance_id + '::') for identifier in left.nodes))
        self.assertLess(len(left.nodes), len(first.workflow.recipe.stages))
        restored = node_instances_from_payload(node_instances_to_payload((edited, second)))
        self.assertEqual(restored, (edited, second))
        self.assertEqual((restored[0].selected_node, restored[0].selected_output), (first.target_node, 'mask'))
        self.assertEqual((NodeInstance.from_dict(second.to_dict()).selected_node,
                          NodeInstance.from_dict(second.to_dict()).selected_output), (None, ''))
        self.assertEqual(instance_graph(restored[0]).node(target).x, 123.5)
        # Bypassing the owned median by rewiring must keep that disconnected
        # card owned, editable and serializable on subsequent saves.
        projected = instance_graph(edited)
        old = next(edge for edge in projected.connections if edge.target == target and edge.target_port == 'gray')
        source = next(identifier for identifier in projected.nodes if identifier.endswith('external_inputs'))
        projected.connections = tuple(edge for edge in projected.connections if edge != old) + (
            replace(old, source=source, source_port='gray_image'),)
        rewired = sync_instance(edited, projected)
        reopened = NodeInstance.from_dict(rewired.to_dict())
        self.assertEqual(set(instance_graph(reopened).nodes), set(projected.nodes))
        self.assertEqual(sync_instance(reopened, instance_graph(reopened)).workflow, reopened.workflow)

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
            NodeInstance.from_dict(first.to_dict() | {'catalogue_key': 'native:invented'})
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
            from seedvision.export.project_native_graphs import export_project_node_instances
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

    def check_test_identical_correction_recipes_keep_owned_inputs_through_portable_reopen(self):
        import cv2
        import numpy as np
        from seedvision.benchmarking.checkpoints import CoordinateFrame
        from seedvision.benchmarking.native_correction_recipe import STAGE_ID
        from seedvision.persistence.project_analysis import ProjectImageSpec
        from seedvision.persistence.correction_inputs import (PORTS, CorrectionInputAsset,
            ProjectCorrectionBinding, validate_correction_bindings)
        entry = next(entry for entry in catalogue_entries() if entry.recipe is not None
                     and entry.recipe.recipe_id == 'plantcv_color_card_correct_color' and entry.target_node == STAGE_ID)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'source.png'
            self.assertTrue(cv2.imwrite(str(source), np.full((4, 6, 3), 80, np.uint8)))
            instances = []
            paths_by_owner = {}
            for index in range(2):
                paths = {'source_rgb': source}
                for role in ('target_rgb', 'target_mask', 'source_mask'):
                    path = root / f'{role}-{index}.png'
                    shape = (4, 6, 3) if role.endswith('rgb') else (4, 6)
                    self.assertTrue(cv2.imwrite(str(path), np.full(shape, 20 + index, np.uint8)))
                    paths[role] = path
                instance = create_instance(entry.key)
                instance = replace(instance, workflow=replace(instance.workflow,
                    input_paths={role: str(path.resolve()) for role, path in paths.items()}))
                instances.append(instance)
                paths_by_owner[instance.instance_id] = paths
            instances = tuple(instances)
            store = ProjectAnalysisStore(root)
            profile = analysis_settings_profile_from_graph(build_default_pipeline())
            initial = store.capture(analysis_settings=profile, images=(ProjectImageSpec(source, (4, 6)),),
                                    node_instances=instances)
            bindings = []
            for instance in instances:
                assets = []
                for role in PORTS:
                    path = paths_by_owner[instance.instance_id][role]
                    raw = path.read_bytes()
                    digest = sha256(raw).hexdigest()
                    frame = CoordinateFrame('target.decoded' if role.startswith('target_') else 'source.decoded',
                        (4, 6), 'top-left', 'x-right,y-down', 'pixel')
                    assets.append(CorrectionInputAsset(role, str(path.resolve()), digest, len(raw), frame,
                        'Declared prior preparation for ' + role, 'native-correction/' + digest + '.image'))
                recipe = instance.workflow.recipe
                bindings.append(ProjectCorrectionBinding(initial.images[0].identifier, recipe.recipe_id,
                    recipe.recipe_version, recipe.identity.recipe_sha256, tuple(assets), instance.instance_id))
            captured = store.capture(analysis_settings=profile, images=(ProjectImageSpec(source, (4, 6)),),
                node_instances=instances, correction_bindings=tuple(bindings), existing_correction_bindings=())
            master = store.save(captured, root / 'owned.seedfiddle-project.json')
            self.assertEqual(len({binding.key for binding in captured.correction_bindings}), 2)
            with self.assertRaises(ValueError):
                validate_correction_bindings(captured.correction_bindings, captured.images, node_instances=instances[:1])
            for paths in paths_by_owner.values():
                for role, path in paths.items():
                    if role != 'source_rgb':
                        path.unlink()
            snapshot = create_snapshot(root, master, root / 'snapshots')
            workspace = root / 'portable'
            restored = restore_snapshot(snapshot, workspace)
            loaded = ProjectAnalysisStore(workspace).load(restored)
            self.assertFalse(loaded.issues)
            self.assertEqual({binding.owner_id for binding in loaded.document.correction_bindings},
                             {instance.instance_id for instance in instances})
            self.assertEqual(len({binding.selection_sha256 for binding in loaded.document.correction_bindings}), 2)

    def test_workflow_lifecycle(self):
        """4 related CatalogueNodeTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_catalogue_includes_default_smpts_models_and_four_input_correction)
        self.scenario(self.check_test_duplicate_native_thresholds_keep_independent_controls_scope_and_round_trip)
        self.scenario(self.check_test_duplicate_default_instances_survive_project_and_snapshot_reopen)
        self.scenario(self.check_test_identical_correction_recipes_keep_owned_inputs_through_portable_reopen)

    def test_validation_and_recovery(self):
        """1 related CatalogueNodeTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_cross_instance_edges_and_duplicate_or_unregistered_descriptors_reject)
