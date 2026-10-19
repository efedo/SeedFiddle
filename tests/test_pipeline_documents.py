"""Standalone graph/layout transport, factory protection and frozen file loads."""

from tests.catalogue_fixtures import CatalogueTestCase

from tests.functional_units import functional_units
from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QApplication, QComboBox, QMainWindow, QMessageBox

from seedvision.persistence.analysis_settings import analysis_settings_profile_from_graph
from seedvision.persistence.pipeline_documents import (
    PIPELINE_EXTENSION, LAYOUT_EXTENSION, PipelineDocument, PipelineDocumentError,
    PipelineDocumentStore, PipelineLayoutDocument,
)
from seedvision.persistence.reference_models import ReferenceModelAsset
from seedvision.pipeline import build_default_pipeline
from seedvision.pipeline.catalogue_nodes import create_instance, display_id
from seedvision.pipeline.workflows import create_workflow, PipelineWorkflow, workflow_graph, workflow_from_graph
from seedvision.ui.pipeline_canvas import PipelineCanvas
from seedvision.ui.pipeline_documents import PipelineDocumentsController
from tests.test_native_recipe import selected_recipe


@functional_units
class PipelineDocumentTests(CatalogueTestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def check_test_full_roundtrip_preserves_both_primaries_instances_and_backend_choices(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            default = build_default_pipeline()
            default.set_parameter('ruler_detection', 'ruler_length_mm', 183.0)
            profile = analysis_settings_profile_from_graph(default)
            workflow = create_workflow(selected_recipe())
            self.assertNotIn('backend_default', workflow.to_dict())
            legacy = PipelineWorkflow.from_dict(workflow.to_dict())
            self.assertEqual(legacy.backend_default, 'native')
            self.assertEqual(dict(legacy.node_backends), {})
            median = next(stage.stage_id for stage in workflow.recipe.stages if stage.operation_id == 'filter.median_blur')
            workflow = replace(workflow, backend_default='plantcv', node_backends={median: 'native'},
                               display_state={'pipeline_name': 'Custom backend comparison'})
            graph = workflow_graph(workflow)
            graph.node(median).enabled = False
            graph.node(median).x = 321.5
            edited = workflow_from_graph(workflow, graph)
            self.assertEqual(edited.backend_for(median), 'native')
            self.assertEqual(edited.backend_for(next(stage.stage_id for stage in edited.recipe.stages if stage.stage_id != median)), 'plantcv')
            self.assertEqual(PipelineWorkflow.from_json(edited.to_json()), edited)
            for updates in ({'backend_default': 'unknown'}, {'node_backends': {'unknown': 'plantcv'}},
                            {'node_backends': {median: 'unknown'}}):
                with self.assertRaises(ValueError):
                    replace(edited, **updates)
            instances = (create_instance('default:seed_scale_estimation'),
                         create_instance('default:seed_scale_estimation'))
            selection = display_id(instances[1], instances[1].target_node)
            layout = PipelineLayoutDocument(((selection, 454.5, -31.0),), selected_node=selection,
                                            selected_output='diameter', bundle_cables=True)
            store = PipelineDocumentStore(root)
            for primary in (None, edited):
                with self.subTest(primary='default' if primary is None else 'native'):
                    document = PipelineDocument('Custom measured seeds', profile, primary, instances, layout,
                                                backend_default='plantcv', node_backends={selection: 'native'})
                    path = root / ('default' if primary is None else 'native') / ('measured' + PIPELINE_EXTENSION)
                    store.save_pipeline(document, path)
                    reopened = store.load_pipeline(path)
                    self.assertEqual(reopened.to_dict(), document.to_dict())
                    self.assertEqual(reopened.node_instances, instances)
                    self.assertEqual(reopened.pipeline_workflow, primary)
                    self.assertEqual(reopened.layout.selected_node, selection)

    def check_test_assets_remain_explicit_and_merge_does_not_erase_unresolved_project_refs(self):
        from seedvision.benchmarking.checkpoints import CoordinateFrame
        from seedvision.persistence.correction_inputs import CorrectionInputAsset, ProjectCorrectionBinding, PORTS, RECIPE_ID
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = ReferenceModelAsset('old-model', '1', 'classification', 'a' * 64, sha256(b'model').hexdigest(), 5,
                                        'models/selected.model')
            existing = replace(model, stage_id='unresolved-old-selection', path='models/unresolved.model')
            inputs = tuple(CorrectionInputAsset(
                role, str((root / (role + '.png')).resolve()), 'b' * 64, 7,
                CoordinateFrame('target.decoded' if role.startswith('target_') else 'source.decoded',
                                (17, 25), 'top-left', 'x-right,y-down', 'pixel'),
                'Declared prior decoded input', 'correction/' + role + '.png') for role in PORTS)
            binding = ProjectCorrectionBinding('c' * 64, RECIPE_ID, '1', 'd' * 64, inputs)
            document = PipelineDocument('Custom retained assets', analysis_settings_profile_from_graph(build_default_pipeline()),
                                        reference_model_assets=(model,), correction_bindings=(binding,), asset_root=str(root))
            store = PipelineDocumentStore(root)
            path = store.save_pipeline(document, root / ('retained' + PIPELINE_EXTENSION))
            loaded = store.load_pipeline(path)
            self.assertEqual(loaded.reference_model_assets, (model,))
            self.assertEqual(loaded.correction_bindings, (binding,))
            self.assertEqual({row['status'] for row in loaded.asset_statuses()}, {'missing'})
            (root / model.path).parent.mkdir()
            (root / model.path).write_bytes(b'model')
            self.assertEqual(loaded.asset_statuses(verify=True)[0]['status'], 'available')
            (root / model.path).write_bytes(b'other')
            self.assertEqual(loaded.asset_statuses(verify=True)[0]['status'], 'fingerprint_mismatch')
            models, bindings = loaded.merge_asset_records((existing,), ())
            self.assertEqual(models, (existing, model))
            self.assertEqual(bindings, (binding,))
            with self.assertRaises(PipelineDocumentError):
                loaded.merge_asset_records((replace(model, path='models/conflicting.model'),), ())
            with self.assertRaises(PipelineDocumentError):
                loaded.merge_asset_records((replace(model, stage_id='other-selection', source_sha256='f' * 64),), ())

    def check_test_invalid_files_atomic_failure_and_factory_names_cannot_replace_existing_bytes(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            document = PipelineDocument('Custom profile', analysis_settings_profile_from_graph(build_default_pipeline()))
            protected = root / ('bundled' + PIPELINE_EXTENSION)
            store = PipelineDocumentStore(root, factory_names=('Official seed pipeline',), factory_paths=(protected,))
            path = store.save_pipeline(document, root / ('custom' + PIPELINE_EXTENSION))
            original = path.read_bytes()
            for name in ('../escape', 'folder\\name', 'name:stream', 'line\nname', 'dot.'):
                with self.subTest(name=name), self.assertRaises(PipelineDocumentError):
                    replace(document, name=name)
            with patch('seedvision.persistence.pipeline_documents.os.replace', side_effect=OSError('simulated replacement failure')):
                with self.assertRaises(PipelineDocumentError):
                    store.save_pipeline(replace(document, name='Changed custom'), path)
            self.assertEqual(path.read_bytes(), original)
            self.assertFalse(tuple(root.glob('*.tmp')))
            for target, name in ((protected, 'Custom profile'),
                                 (root / ('official seed pipeline' + PIPELINE_EXTENSION), 'Custom profile'),
                                 (root / ('custom' + PIPELINE_EXTENSION), '  Official seed pipeline  '),
                                 (root / 'config' / ('custom' + PIPELINE_EXTENSION), 'Custom profile')):
                with self.subTest(target=target, name=name), self.assertRaises(ValueError):
                    store.save_pipeline(replace(document, name=name), target)
            self.assertEqual(path.read_bytes(), original)
            malformed = root / ('invalid' + PIPELINE_EXTENSION)
            malformed.write_text('{"format":"seedfiddle-pipeline","format":"duplicate"}', encoding='utf-8')
            with self.assertRaises(PipelineDocumentError):
                store.load_pipeline(malformed)
            with patch('seedvision.persistence.pipeline_documents.MAX_DOCUMENT_BYTES', 32):
                with self.assertRaises(PipelineDocumentError):
                    store.load_pipeline(path)
            with self.assertRaises(PipelineDocumentError):
                PipelineLayoutDocument((('project', float('nan'), 0),))

    def check_test_controller_loads_frozen_and_layout_does_not_invalidate_calculations(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            window = QMainWindow()
            window._root = root
            window._background_work_is_active = Mock(return_value=False)
            window._thread_pool = SimpleNamespace(activeThreadCount=lambda: 0)
            window._set_project_dirty = Mock()
            graph = build_default_pipeline()
            canvas = PipelineCanvas(graph, window)
            window.pipeline_canvas = canvas
            window.pipeline_workflow = SimpleNamespace(active=False)
            window.catalogue_nodes = None
            window.image_workspace_action = QAction(window)
            window.pipeline_workspace_action = QAction(window)
            for action in (window.image_workspace_action, window.pipeline_workspace_action):
                action.setCheckable(True)
                action.setChecked(True)
            window.overlay_combo = QComboBox(window)
            window.overlay_combo.addItem('Raw', 'raw_image')
            profile = analysis_settings_profile_from_graph(graph)
            document = PipelineDocument('Custom inspection', profile)
            store = PipelineDocumentStore(root, factory_names=('Official source',))
            path = store.save_pipeline(document, root / ('inspection' + PIPELINE_EXTENSION))
            install = Mock()
            controller = PipelineDocumentsController(window, capture=lambda name: replace(document, name=name),
                                                      install=install, store=store)
            locks = Mock()
            controller.frozen_changed.connect(locks)
            live_name = {'value': 'Original live name'}
            controller.pipeline_saved.connect(lambda document, path: live_name.update(value=document.name))
            self.assertTrue(controller.frozen)
            self.assertFalse(controller.can_edit())
            controller.set_frozen(False)
            self.assertTrue(controller.can_edit())
            controller.loaded('Official source')
            self.assertTrue(controller.frozen)
            self.assertTrue(locks.call_args.args[0])
            with patch.object(QMessageBox, 'warning'):
                self.assertFalse(controller.save_pipeline_as(root / ('copy' + PIPELINE_EXTENSION), 'Official source'))
                self.assertFalse(controller.load_pipeline(root / 'does-not-exist.json'))
                with patch.object(store, 'save_pipeline', side_effect=PipelineDocumentError('Simulated save failure')):
                    self.assertFalse(controller.save_pipeline_as(root / ('copy' + PIPELINE_EXTENSION), 'Custom failed name'))
            self.assertEqual(live_name['value'], 'Original live name')
            self.assertEqual(controller.current_name, 'Official source')
            install.assert_not_called()
            self.assertTrue(controller.save_pipeline_as(root / ('copy' + PIPELINE_EXTENSION), 'Custom saved name'))
            self.assertEqual(live_name['value'], 'Custom saved name')
            self.assertTrue(controller.load_pipeline(path))
            install.assert_called_once()
            self.assertTrue(controller.frozen)
            revision, edges = graph.revision, graph.connections
            cached = object()
            window._analysis_caches = {'test-image': cached}
            graph.node('project').calculation_seconds = 12.1
            layout = PipelineLayoutDocument((('project', 231.5, -98.0), ('absent-stable-id', 1, 2)),
                                            selected_node='project', selected_output='raw_image', bundle_cables=True)
            layout_path = store.save_layout(layout, root / ('arrangement' + LAYOUT_EXTENSION))
            with patch.object(graph, 'invalidate', wraps=graph.invalidate) as invalidate:
                self.assertTrue(controller.load_layout(layout_path))
            invalidate.assert_not_called()
            self.assertEqual((graph.node('project').x, graph.node('project').y), (231.5, -98.0))
            self.assertIs(window._analysis_caches['test-image'], cached)
            self.assertEqual(graph.node('project').calculation_seconds, 12.1)
            self.assertIs(graph.connections, edges)
            self.assertEqual(graph.revision, revision)
            self.assertEqual(analysis_settings_profile_from_graph(graph), profile)
            self.assertTrue(controller.frozen)
            self.assertTrue(controller.save_layout_as(root / ('resaved' + LAYOUT_EXTENSION)))
            window.close()
            self.application.processEvents()

    def test_workflow_lifecycle(self):
        """2 related PipelineDocumentTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_full_roundtrip_preserves_both_primaries_instances_and_backend_choices)
        self.scenario(self.check_test_assets_remain_explicit_and_merge_does_not_erase_unresolved_project_refs)

    def test_validation_and_recovery(self):
        """2 related PipelineDocumentTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_invalid_files_atomic_failure_and_factory_names_cannot_replace_existing_bytes)
        self.scenario(self.check_test_controller_loads_frozen_and_layout_does_not_invalidate_calculations)
