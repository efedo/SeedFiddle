"""Desktop freeze, factory copies and complete graph/layout file integration."""

from tests.catalogue_fixtures import CatalogueTestCase

from tests.functional_units import functional_units
from dataclasses import replace
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtWidgets import QApplication, QMessageBox

from seedvision.persistence.pipeline_documents import PipelineLayoutDocument
from seedvision.ui.pipeline_workspace import capture_pipeline
from tests.review_fixtures import isolated_window, dispose_window


@functional_units
class PipelineWorkspaceFileTests(CatalogueTestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.window = isolated_window(self.root, editable=False)
        self.presenter = self.window.pipeline_workflow
        self.files = self.window.pipeline_documents

    def tearDown(self):
        dispose_window(self.window)
        self.application.processEvents()
        self.temporary.cleanup()

    def binary(self):
        index = next(index for index in range(1, self.presenter.selector.count())
                     if self.presenter.selector.itemData(index).recipe_id == 'plantcv_seed_binary_working')
        self.presenter.selector.setCurrentIndex(index)
        return next(stage.stage_id for stage in self.presenter.workflow.recipe.stages
                    if stage.operation_id == 'threshold.binary')

    def check_test_freeze_guards_edits_but_allows_layout_run_and_backend_choice(self):
        self.assertTrue(self.files.frozen)
        self.assertTrue(self.files.freeze_button.isChecked())
        self.assertFalse(hasattr(self.window, 'reference_graphs_action'))
        menus = [action.text().replace('&', '') for action in self.window.menuBar().actions()]
        self.assertNotIn('View', menus)
        identifier = self.binary()
        graph = self.presenter.graph
        before = dict(graph.node(identifier).parameters)
        self.window.pipeline_canvas.select_node(identifier)
        self.assertTrue(self.window.pipeline_canvas.editing_locked)
        self.assertTrue(self.window.pipeline_inspector.editing_locked)
        self.assertFalse(self.window.pipeline_inspector.reset_parameters_button.isEnabled())
        self.presenter.parameter_changed(identifier, 'threshold', 119)
        self.window.pipeline_canvas._inline_parameter_changed(identifier, 'threshold', 121)
        self.assertEqual(graph.node(identifier).parameters, before)
        self.assertFalse(self.window.catalogue_nodes.add_entry('default:seed_scale_estimation'))
        edges = tuple(graph.connections)
        self.window.pipeline_canvas._disconnect_connection(edges[0])
        self.assertEqual(tuple(graph.connections), edges)
        with patch.object(self.presenter, 'run') as run:
            self.window.pipeline_canvas._request_run_to_node(identifier)
            run.assert_called_once_with(identifier)
        layout = PipelineLayoutDocument(((identifier, 120.0, 230.0),), selected_node=identifier)
        revision, cache = graph.revision, self.presenter.cache
        self.files.apply_layout(layout)
        self.assertEqual((graph.node(identifier).x, graph.node(identifier).y), (120.0, 230.0))
        self.assertEqual(graph.revision, revision)
        self.assertIs(self.presenter.cache, cache)
        with patch('sys.excepthook') as qt_error:
            self.window.pipeline_backends.global_selector.setCurrentIndex(1)
            qt_error.assert_not_called()
        self.assertEqual(self.presenter.workflow.backend_default, 'plantcv')
        self.assertTrue(self.files.frozen)
        with self.assertRaisesRegex(ValueError, 'editable'):
            self.window._activate_trait_model(self.root / 'not-read.json')
        self.files.set_frozen(False)
        self.presenter.parameter_changed(identifier, 'threshold', 119)
        self.assertEqual(self.presenter.workflow.recipe.parameters_for_runner(identifier)['threshold'], 119)

    def check_test_factory_save_as_and_fresh_template_never_modify_factory(self):
        identifier = self.binary()
        original = self.presenter.workflow.source_recipe.to_json()
        factory = self.presenter.workflow.source_recipe.title
        self.files.set_frozen(False)
        self.presenter.parameter_changed(identifier, 'threshold', 117)
        destination = self.root / 'custom.seedfiddle-pipeline.json'
        with patch.object(QMessageBox, 'warning') as warning:
            self.assertFalse(self.files.save_pipeline_as(destination, factory))
            warning.assert_called_once()
        self.assertFalse(destination.exists())
        self.assertTrue(self.files.save_pipeline_as(destination, 'My seed comparison'))
        self.assertEqual(self.presenter.workflow.display_state['pipeline_name'], 'My seed comparison')
        self.presenter.selector.setCurrentIndex(0)
        self.binary()
        self.assertTrue(self.files.frozen)
        self.assertEqual(self.presenter.workflow.source_recipe.to_json(), original)
        self.assertNotEqual(self.presenter.workflow.recipe.parameters_for_runner(identifier)['threshold'], 117)
        self.assertTrue(self.files.load_pipeline(destination))
        self.assertTrue(self.files.frozen)
        self.assertEqual(self.presenter.workflow.recipe.parameters_for_runner(identifier)['threshold'], 117)

    def check_test_pipeline_roundtrip_retains_independent_instances_and_layout_cache_is_inert(self):
        identifier = self.binary()
        self.files.set_frozen(False)
        first = self.window.catalogue_nodes.add_entry('default:seed_scale_estimation')
        second = self.window.catalogue_nodes.add_entry('default:seed_scale_estimation')
        self.assertNotEqual(first, second)
        self.window.catalogue_nodes.output_requested(second, '')
        graph = self.window.pipeline_canvas.graph
        graph.node(second).x, graph.node(second).y = 1450.0, -210.0
        self.window.pipeline_canvas.select_node(second)
        instances = self.window.catalogue_nodes.snapshot_instances()
        destination = self.root / 'two-seed-nodes.seedfiddle-pipeline.json'
        self.assertTrue(self.files.save_pipeline_as(destination, 'Two seed scale estimates'))
        self.presenter.install(None)
        self.assertTrue(self.files.load_pipeline(destination))
        self.assertEqual(self.window.catalogue_nodes.snapshot_instances(), instances)
        self.assertTrue(self.files.frozen)
        self.assertEqual(self.window.catalogue_nodes._selected, second)
        self.assertIn(identifier, self.window.pipeline_canvas.graph.nodes)
        component = self.window.catalogue_nodes._component(second)
        cached_result, cache = component.result, component.cache
        sentinel = object()
        cache.values['retained-fixture-product'] = sentinel
        revision = self.window.pipeline_canvas.graph.revision
        layout_path = self.root / 'alternate.seedfiddle-layout.json'
        layout = replace(self.files.capture_layout(), positions=((second, 1800.0, -420.0),))
        self.files.store.save_layout(layout, layout_path)
        self.assertTrue(self.files.load_layout(layout_path))
        self.assertIs(component.result, cached_result)
        self.assertIs(component.cache, cache)
        self.assertIs(cache.values['retained-fixture-product'], sentinel)
        self.assertEqual(self.window.pipeline_canvas.graph.revision, revision)
        self.assertEqual(self.window.pipeline_canvas.graph.node(second).x, 1800.0)

    def check_test_invalid_backend_load_is_atomic_and_unavailable_assets_survive_resave(self):
        from seedvision.persistence.reference_models import ReferenceModelAsset
        identifier = self.binary()
        original_graph = self.window.pipeline_canvas.graph
        original = capture_pipeline(self.window, 'Validated graph')
        payload = original.to_dict()
        external = next(stage.stage_id for stage in self.presenter.workflow.recipe.stages
                        if stage.operation_id == 'external.image_inputs')
        payload['pipeline_workflow']['node_backends'] = {external: 'plantcv'}
        bad = self.root / 'invalid.seedfiddle-pipeline.json'
        bad.write_text(json.dumps(payload), encoding='utf-8')
        with patch.object(QMessageBox, 'warning') as warning:
            self.assertFalse(self.files.load_pipeline(bad))
            warning.assert_called_once()
        self.assertIs(self.window.pipeline_canvas.graph, original_graph)
        self.assertEqual(self.presenter.workflow.recipe, original.pipeline_workflow.recipe)
        from seedvision.persistence.project_analysis import ProjectAnalysisStore
        project = self.root / 'invalid-choice.seedfiddle-project.json'
        ProjectAnalysisStore(self.root).capture_and_save(analysis_settings=original.analysis_settings,
            images=(), pipeline_workflow=original.pipeline_workflow, destination=project)
        project_payload = json.loads(project.read_text(encoding='utf-8'))
        project_payload['pipeline_workflow']['node_backends'] = {external: 'plantcv'}
        project.write_text(json.dumps(project_payload), encoding='utf-8')
        with patch.object(QMessageBox, 'critical') as critical:
            self.assertFalse(self.window._open_project(project))
            critical.assert_called_once()
        self.assertIs(self.window.pipeline_canvas.graph, original_graph)
        retained = ReferenceModelAsset('old-model', '1', 'classification', 'a' * 64, 'b' * 64, 5, 'models/missing.model')
        document = replace(original, reference_model_assets=(retained,))
        good = self.root / 'retained.seedfiddle-pipeline.json'
        self.files.store.save_pipeline(document, good)
        self.assertTrue(self.files.load_pipeline(good))
        resaved = capture_pipeline(self.window, 'Retained references')
        self.assertIn(retained, resaved.reference_model_assets)
        self.assertEqual(resaved.asset_statuses(self.root)[0]['status'], 'missing')
        self.assertIn(identifier, self.presenter.graph.nodes)
        from hashlib import sha256
        origin = self.root / 'other-workspace'
        (origin / 'models').mkdir(parents=True)
        (origin / 'models' / 'portable.model').write_bytes(b'model')
        transported = replace(retained, stage_id='portable-classification',
            source_sha256=sha256(b'model').hexdigest(), path='models/portable.model')
        document = replace(original, name='Portable asset reference', reference_model_assets=(transported,),
            asset_root=str(origin.resolve()))
        portable = self.root / 'portable.seedfiddle-pipeline.json'
        self.files.store.save_pipeline(document, portable)
        self.assertEqual(document.asset_statuses(self.root)[0]['status'], 'missing')
        installer = self.files._install
        def install_then_hide_origin(selected):
            result = installer(selected)
            (origin / 'models' / 'portable.model').rename(origin / 'models' / 'moved.model')
            return result
        with patch.object(self.files, '_install', side_effect=install_then_hide_origin):
            self.assertTrue(self.files.load_pipeline(portable))
        self.assertEqual((self.root / 'models' / 'portable.model').read_bytes(), b'model')
        self.assertNotIn('remain unavailable', self.window.statusBar().currentMessage())
        self.assertEqual(document.asset_statuses(self.root)[0]['status'], 'unchecked')
        self.assertEqual(document.asset_statuses()[0]['status'], 'missing')

    def check_test_node_backend_updates_only_affected_execution_and_atomic_watershed(self):
        identifier = self.binary()
        from seedvision.pipeline import NodeStatus
        from seedvision.pipeline.backend_execution import WATERSHED_OPERATIONS
        for node in self.presenter.graph.nodes.values():
            node.status = NodeStatus.COMPLETE
        self.window.pipeline_canvas.select_node(identifier)
        cache = self.presenter.cache
        control = self.window.pipeline_backends.node_selector
        control.setCurrentIndex(control.findData('plantcv'))
        self.assertEqual(self.presenter.workflow.node_backends[identifier], 'plantcv')
        self.assertIs(self.presenter.cache, cache)
        upstream = self.presenter.graph.upstream(identifier)
        self.assertTrue(all(self.presenter.graph.node(node).status == NodeStatus.COMPLETE for node in upstream))
        group = tuple(stage.stage_id for stage in self.presenter.workflow.recipe.stages
                      if stage.operation_id in WATERSHED_OPERATIONS)
        self.window.pipeline_canvas.select_node(group[0])
        control.setCurrentIndex(control.findData('plantcv'))
        self.assertEqual({self.presenter.workflow.node_backends[node] for node in group}, {'plantcv'})
        self.assertTrue(self.files.frozen)
        self.files.set_frozen(False)
        from seedvision.pipeline.catalogue_nodes import catalogue_entries
        entry = next(entry for entry in catalogue_entries() if entry.backend == 'native'
                     and entry.recipe.stage(entry.target_node).operation_id == 'threshold.binary')
        added = self.window.catalogue_nodes.add_entry(entry.key)
        self.window.pipeline_canvas.select_node(added)
        with patch('sys.excepthook') as qt_error:
            self.window.pipeline_backends.global_selector.setCurrentIndex(1)
            qt_error.assert_not_called()
        component = self.window.catalogue_nodes._component(added)
        self.assertEqual(component.instance.workflow.backend_default, 'plantcv')
        control.setCurrentIndex(control.findData('native'))
        document = capture_pipeline(self.window, 'Mixed execution choices')
        path = self.root / 'mixed.seedfiddle-pipeline.json'
        self.files.store.save_pipeline(document, path)
        with patch('sys.excepthook') as qt_error:
            self.window.pipeline_backends.global_selector.setCurrentIndex(0)
            qt_error.assert_not_called()
        self.assertTrue(self.files.load_pipeline(path))
        restored = self.window.catalogue_nodes._component(added).instance.workflow
        self.assertEqual(restored.backend_default, 'plantcv')
        self.assertEqual(restored.node_backends[entry.target_node], 'native')
        self.assertTrue(self.files.frozen)

    def test_workflow_lifecycle(self):
        """4 related PipelineWorkspaceFileTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_freeze_guards_edits_but_allows_layout_run_and_backend_choice)
        self.scenario(self.check_test_factory_save_as_and_fresh_template_never_modify_factory)
        self.scenario(self.check_test_pipeline_roundtrip_retains_independent_instances_and_layout_cache_is_inert)
        self.scenario(self.check_test_node_backend_updates_only_affected_execution_and_atomic_watershed)

    def test_validation_and_recovery(self):
        """1 related PipelineWorkspaceFileTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_invalid_backend_load_is_atomic_and_unavailable_assets_survive_resave)
