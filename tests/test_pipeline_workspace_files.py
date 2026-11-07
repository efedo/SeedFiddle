"""Desktop default-pipeline lock, named copies and complete graph/layout file integration."""


from tests.functional_units import functional_units
from dataclasses import replace
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtWidgets import QApplication, QMessageBox

from seedvision.persistence.pipeline_documents import PipelineLayoutDocument
from seedvision.ui.pipeline_workspace import capture_pipeline
from tests.review_fixtures import editable_copy, isolated_window, dispose_window

NODE, KEY = 'ruler_detection', 'ruler_length_mm'


@functional_units
class PipelineWorkspaceFileTests(unittest.TestCase):
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

    def value(self):
        return self.window.pipeline.node(NODE).parameters[KEY]

    def check_test_lock_guards_edits_but_allows_layout_and_run_to_node(self):
        self.assertTrue(self.files.locked)
        self.assertFalse(hasattr(self.window, 'reference_graphs_action'))
        self.assertFalse(hasattr(self.window, 'pipeline_backends'))
        menus = [action.text().replace('&', '') for action in self.window.menuBar().actions()]
        self.assertNotIn('View', menus)
        graph = self.window.pipeline
        before = dict(graph.node(NODE).parameters)
        canvas = self.window.pipeline_canvas
        canvas.select_node(NODE)
        self.assertTrue(canvas.editing_locked)
        self.assertTrue(self.window.pipeline_inspector.editing_locked)
        self.assertFalse(self.window.pipeline_inspector.reset_parameters_button.isEnabled())
        # There is no lock control, and the inspector says why controls are disabled.
        self.assertFalse(hasattr(self.files, 'freeze_button'))
        inspector = self.window.pipeline_inspector
        self.assertTrue(inspector.lock_notice.isVisibleTo(inspector))
        commands = [action.text() for action in self.presenter.more_button.menu().actions()]
        self.assertNotIn('Load layout…', commands)
        self.assertNotIn('Save layout as…', commands)
        with patch.object(QMessageBox, 'question', return_value=QMessageBox.StandardButton.No) as question:
            self.window._pipeline_parameter_changed(NODE, KEY, 119.0)
            self.assertFalse(self.window.catalogue_nodes.add_entry('default:seed_scale_estimation'))
        self.assertEqual(question.call_count, 2)
        canvas._inline_parameter_changed(NODE, KEY, 121.0)
        self.assertEqual(graph.node(NODE).parameters, before)
        edges = tuple(graph.connections)
        canvas._disconnect_connection(edges[0])
        self.assertEqual(tuple(graph.connections), edges)
        # Running needs no copy: it changes no setting.
        with patch.object(self.window, '_analyze_current_image') as analyze:
            canvas._request_run_to_node(NODE)
        self.assertEqual(analyze.call_args.kwargs['rerun_node'], NODE)
        self.assertIn(NODE, analyze.call_args.kwargs['enabled_node_scope'])
        layout = PipelineLayoutDocument(((NODE, 120.0, 230.0),), selected_node=NODE)
        revision, caches = graph.revision, dict(self.window._analysis_caches)
        self.files.apply_layout(layout)
        self.assertEqual((graph.node(NODE).x, graph.node(NODE).y), (120.0, 230.0))
        self.assertEqual(graph.revision, revision)
        self.assertEqual(self.window._analysis_caches, caches)
        self.assertTrue(self.files.locked)
        with patch.object(QMessageBox, 'question', return_value=QMessageBox.StandardButton.No):
            with self.assertRaisesRegex(ValueError, 'editable'):
                self.window._activate_trait_model(self.root / 'not-read.json')
        editable_copy(self.window)
        self.assertFalse(inspector.lock_notice.isVisibleTo(inspector))
        self.window._pipeline_parameter_changed(NODE, KEY, 119.0)
        self.assertEqual(self.value(), 119.0)

    def check_test_factory_save_as_and_fresh_template_never_modify_factory(self):
        factory_value = self.value()
        self.assertNotEqual(factory_value, 117.0)
        factory = self.presenter.graph_pipeline.title
        editable_copy(self.window)
        self.window._pipeline_parameter_changed(NODE, KEY, 117.0)
        destination = self.root / 'custom.seedfiddle-pipeline.json'
        with patch.object(QMessageBox, 'warning') as warning:
            self.assertFalse(self.files.save_pipeline_as(destination, factory))
            warning.assert_called_once()
        self.assertFalse(destination.exists())
        self.assertTrue(self.files.save_pipeline_as(destination, 'My seed comparison'))
        self.assertEqual(self.files.current_name, 'My seed comparison')
        self.presenter.select_template(None)
        self.assertTrue(self.files.locked)
        self.assertEqual(self.value(), factory_value)
        self.assertTrue(self.files.load_pipeline(destination))
        self.assertFalse(self.files.locked)
        self.assertEqual(self.value(), 117.0)

    def check_test_pipeline_roundtrip_retains_independent_instances_and_layout_cache_is_inert(self):
        editable_copy(self.window)
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
        self.presenter.select_template(None)
        self.assertEqual(self.window.catalogue_nodes.instances, ())
        self.assertTrue(self.files.load_pipeline(destination))
        self.assertEqual(self.window.catalogue_nodes.snapshot_instances(), instances)
        self.assertFalse(self.files.locked)
        self.assertEqual(self.window.catalogue_nodes._selected, second)
        self.assertIn(NODE, self.window.pipeline_canvas.graph.nodes)
        component = self.window.catalogue_nodes._component(second)
        cached_result, cache = component.result, component.cache
        sentinel = object()
        cache.values['retained-fixture-product'] = sentinel
        revision = self.window.pipeline_canvas.graph.revision
        layout = replace(self.files.capture_layout(), positions=((second, 1800.0, -420.0),))
        self.files.apply_layout(layout)
        self.assertIs(component.result, cached_result)
        self.assertIs(component.cache, cache)
        self.assertIs(cache.values['retained-fixture-product'], sentinel)
        self.assertEqual(self.window.pipeline_canvas.graph.revision, revision)
        self.assertEqual(self.window.pipeline_canvas.graph.node(second).x, 1800.0)

    def check_test_refused_pipeline_and_project_files_leave_the_working_graph(self):
        editable_copy(self.window)
        self.window._pipeline_parameter_changed(NODE, KEY, 133.0)
        graph, name = self.window.pipeline, self.files.current_name
        payload = capture_pipeline(self.window, 'Validated graph').to_dict()
        layout = payload['layout']
        # An earlier format, a removed recipe workflow and a layout naming no node are each refused.
        for label, changed in (('earlier version', payload | {'version': 1}),
                               ('removed workflow', payload | {'pipeline_workflow': None}),
                               ('unknown node', payload | {'layout': layout | {'selected_node': 'invented'}})):
            with self.subTest(label):
                bad = self.root / 'invalid.seedfiddle-pipeline.json'
                bad.write_text(json.dumps(changed), encoding='utf-8')
                with patch.object(QMessageBox, 'warning') as warning:
                    self.assertFalse(self.files.load_pipeline(bad))
                    warning.assert_called_once()
                self.assertIs(self.window.pipeline, graph)
                self.assertIs(self.window.pipeline_canvas.graph, graph)
                self.assertEqual(self.files.current_name, name)
                self.assertEqual(self.value(), 133.0)
        from seedvision.persistence.analysis_settings import analysis_settings_profile_from_graph
        from seedvision.persistence.project_analysis import ProjectAnalysisStore
        project = self.root / 'removed-field.seedfiddle-project.json'
        ProjectAnalysisStore(self.root).capture_and_save(
            analysis_settings=analysis_settings_profile_from_graph(graph), images=(), destination=project)
        project_payload = json.loads(project.read_text(encoding='utf-8'))
        project_payload['native_recipes'] = []
        project.write_text(json.dumps(project_payload), encoding='utf-8')
        with patch.object(QMessageBox, 'critical') as critical:
            self.assertFalse(self.window._open_project(project))
            critical.assert_called_once()
        self.assertIs(self.window.pipeline_canvas.graph, graph)
        self.assertEqual(self.value(), 133.0)

    def test_workflow_lifecycle(self):
        """3 related PipelineWorkspaceFileTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_lock_guards_edits_but_allows_layout_and_run_to_node)
        self.scenario(self.check_test_factory_save_as_and_fresh_template_never_modify_factory)
        self.scenario(self.check_test_pipeline_roundtrip_retains_independent_instances_and_layout_cache_is_inert)

    def test_validation_and_recovery(self):
        """1 related PipelineWorkspaceFileTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_refused_pipeline_and_project_files_leave_the_working_graph)


