"""CPU/Qt ownership, routing and stale-publication checks for added nodes."""


from tests.functional_units import functional_units
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PySide6.QtWidgets import QApplication, QWidget
from seedvision.pipeline import build_default_pipeline, NodeStatus
from seedvision.pipeline.catalogue_nodes import catalogue_entries
from seedvision.ui.catalogue_nodes import CatalogueNodesController


class _Pool:
    def __init__(self):
        self.active = 0
        self.jobs = []

    def activeThreadCount(self):
        return self.active

    def start(self, job):
        self.jobs.append(job)


class _Window(QWidget):
    def __init__(self):
        super().__init__()
        self._thread_pool = _Pool()
        self.image_view = SimpleNamespace(image_path=Path('fixture.png'))
        self.pipeline_canvas = Mock()
        self.pipeline_inspector = Mock()
        self.calibration_section = Mock()
        self.baseline_section = Mock()
        self.image_display_stack = Mock()
        self.pipeline_workflow = SimpleNamespace(preview=Mock(), facts=Mock(), render=Mock())
        self._root = Path(__file__).resolve().parents[1]
        self._source_fingerprints = {}
        self._set_project_dirty = Mock()
        self._check_source_revision = Mock(return_value=True)
        self.messages = Mock()

    def statusBar(self):
        return self.messages


@functional_units
class CatalogueNodesControllerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.window = _Window()
        self.controller = CatalogueNodesController(self.window)
        self.primary = build_default_pipeline()
        self.controller.install_primary(self.primary)

    def tearDown(self):
        self.controller.stop()
        self.window.deleteLater()
        self.app.processEvents()

    def check_test_duplicate_default_instances_own_controls_cache_and_prerequisite_presence(self):
        first = self.controller.add_entry('default:background_likelihood')
        second = self.controller.add_entry('default:background_likelihood')
        a, b = self.controller._component(first), self.controller._component(second)
        self.assertNotEqual(first, second)
        self.assertIsNot(a.cache, b.cache)
        self.assertIsNot(a.graph.node(first).parameters, b.graph.node(second).parameters)
        self.assertIs(self.controller.graph.node('background_likelihood'), self.primary.node('background_likelihood'))
        self.controller.parameter_changed(first, 'background_colour_enabled', False)
        self.assertFalse(a.graph.node(first).parameters['background_colour_enabled'])
        self.assertTrue(b.graph.node(second).parameters['background_colour_enabled'])
        self.assertTrue(self.primary.node('background_likelihood').parameters['background_colour_enabled'])
        menu = {item['key']: item for item in self.controller.menu_entries()}
        self.assertEqual(menu['default:background_likelihood']['count'], 3)
        self.assertEqual(menu['default:project']['count'], 3)
        saved = self.controller.snapshot_instances()
        self.controller.set_instances(saved)
        self.assertEqual(self.controller.snapshot_instances(), saved)

    def check_test_default_target_runs_only_its_scope_with_its_own_cache(self):
        identifier = self.controller.add_entry('default:seed_scale_estimation')
        component = self.controller._component(identifier)
        result = object()
        with patch.object(self.controller, '_default_context', return_value={}), \
             patch('seedvision.segmentation.baseline.analyze_path', return_value=result) as runner:
            function = self.controller._prepare_run(component, identifier, Path('fixture.png'))
            self.assertIs(function(lambda: False), result)
        options = runner.call_args.kwargs
        self.assertIs(options['node_cache'], component.cache)
        self.assertIn('seed_scale_estimation', options['enabled_nodes'])
        self.assertFalse(any('::' in node for node in options['enabled_nodes']))
        self.assertLess(len(options['enabled_nodes']), len(self.primary.nodes))
        self.assertIsNone(self.controller._component(identifier.split('::')[0] + '::unregistered'))

    def check_test_queue_waits_for_primary_and_stop_withholds_old_result(self):
        identifier = self.controller.add_entry('default:seed_scale_estimation')
        component = self.controller._component(identifier)
        component.result = object()
        self.window._thread_pool.active = 1
        with patch.object(self.controller, '_prepare_run', return_value=lambda cancel: object()),              patch.object(self.controller, '_context_fingerprint', return_value='context'):
            self.controller.run(identifier)
            self.controller._timer.stop()
            self.assertIsNone(component.result)
            self.controller._advance()
            self.controller._timer.stop()
            self.assertEqual(self.window._thread_pool.jobs, [])
            self.window._thread_pool.active = 0
            self.controller._advance()
        job = self.window._thread_pool.jobs[0]
        self.assertTrue(self.controller.running)
        self.controller.stop()
        self.controller._completed(job, identifier, object())
        self.assertIsNone(component.result)
        self.assertFalse(self.controller.busy)
        self.assertEqual(component.graph.node(identifier).status, NodeStatus.IDLE)

    def check_test_instance_limit_and_planned_node_reject_before_execution(self):
        from seedvision.pipeline.catalogue_nodes import MAX_NODE_INSTANCES
        with patch.object(self.controller, '_components', {str(index): None for index in range(MAX_NODE_INSTANCES)}), \
             patch('seedvision.pipeline.catalogue_nodes.create_instance') as factory:
            self.assertIsNone(self.controller.add_entry('default:project'))
            factory.assert_not_called()
        planned = next(entry for entry in catalogue_entries()
            if not ({**self.primary.nodes, **self.primary.unused_nodes}[entry.target_node].implemented))
        identifier = self.controller.add_entry(planned.key)
        self.controller.run(identifier)
        self.controller._timer.stop()
        self.controller._advance()
        self.assertFalse(self.controller.running)
        self.assertEqual(self.window._thread_pool.jobs, [])
        self.assertEqual(self.controller.graph.node(identifier).status, NodeStatus.BLOCKED)
        self.assertIn('planned', self.controller.graph.node(identifier).status_detail)

    def check_test_primary_wiring_revision_changes_only_when_its_edges_change(self):
        self.controller.add_entry('default:project')
        initial = self.primary.revision
        self.controller.sync_primary()
        self.assertEqual(self.primary.revision, initial)
        edge = self.primary.connections[0]
        self.controller.graph.connections = tuple(value for value in self.controller.graph.connections if value != edge)
        self.controller.graph.revision += 1
        self.controller.sync_primary()
        self.assertNotIn(edge, self.primary.connections)
        self.assertGreater(self.primary.revision, initial)
        revision = self.primary.revision
        self.controller.sync_primary()
        self.assertEqual(self.primary.revision, revision)

    def test_workflow_lifecycle(self):
        """3 related CatalogueNodesControllerTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_duplicate_default_instances_own_controls_cache_and_prerequisite_presence)
        self.scenario(self.check_test_default_target_runs_only_its_scope_with_its_own_cache)
        self.scenario(self.check_test_primary_wiring_revision_changes_only_when_its_edges_change)

    def test_validation_and_recovery(self):
        """2 related CatalogueNodesControllerTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_queue_waits_for_primary_and_stop_withholds_old_result)
        self.scenario(self.check_test_instance_limit_and_planned_node_reject_before_execution)


if __name__ == '__main__':
    unittest.main()
