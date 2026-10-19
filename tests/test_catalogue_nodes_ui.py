"""CPU/Qt ownership, routing and stale-publication checks for added nodes."""

from tests.catalogue_fixtures import CatalogueTestCase

from tests.functional_units import functional_units
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PySide6.QtWidgets import QApplication, QHBoxLayout, QWidget
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
        actions = QWidget(self)
        actions.setLayout(QHBoxLayout())
        self.pipeline_workflow = SimpleNamespace(workflow=None, node_actions=actions,
            preview=Mock(), facts=Mock(), histogram=Mock(), crop_selector=Mock(), render=Mock())
        self._root = Path(__file__).resolve().parents[1]
        self._project_reference_model_assets = ()
        self._project_correction_bindings = ()
        self._source_fingerprints = {}
        self._set_project_dirty = Mock()
        self._check_source_revision = Mock(return_value=True)
        self.messages = Mock()

    def statusBar(self):
        return self.messages


@functional_units
class CatalogueNodesControllerTests(CatalogueTestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls.native_threshold = next(entry for entry in catalogue_entries()
            if entry.backend == 'native' and entry.recipe.stage(entry.target_node).operation_id == 'threshold.binary')

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

    def check_test_native_target_uses_own_cache_and_existing_runner(self):
        identifier = self.controller.add_entry(self.native_threshold.key)
        component = self.controller._component(identifier)
        result = object()
        with patch('seedvision.persistence.reference_models.model_asset_paths', return_value={}), \
             patch('seedvision.pipeline.workflows.run_workflow_path', return_value=result) as runner:
            function = self.controller._prepare_run(component, identifier, Path('fixture.png'))
            self.assertIs(function(lambda: False), result)
        self.assertIs(runner.call_args.kwargs['cache'], component.cache)
        self.assertEqual(runner.call_args.kwargs['target_node_id'], component.instance.target_node)
        self.assertIs(runner.call_args.args[0], component.instance.workflow)
        self.assertIsNone(self.controller._component(identifier.split('::')[0] + '::unregistered'))

    def check_test_queue_waits_for_primary_and_stop_withholds_old_result(self):
        identifier = self.controller.add_entry(self.native_threshold.key)
        component = self.controller._component(identifier)
        component.result = object()
        self.window._thread_pool.active = 1
        with patch.object(self.controller, '_prepare_run', return_value=lambda cancel: object()):
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
        planned = next(entry for entry in catalogue_entries() if entry.backend == 'default'
            and not ({**self.primary.nodes, **self.primary.unused_nodes}[entry.target_node].implemented))
        identifier = self.controller.add_entry(planned.key)
        self.controller.run(identifier)
        self.controller._timer.stop()
        self.controller._advance()
        self.assertFalse(self.controller.running)
        self.assertEqual(self.window._thread_pool.jobs, [])
        self.assertEqual(self.controller.graph.node(identifier).status, NodeStatus.BLOCKED)
        self.assertIn('planned', self.controller.graph.node(identifier).status_detail)

    def check_test_correction_instances_never_borrow_primary_binding_or_sibling_paths(self):
        entry = next(entry for entry in catalogue_entries() if entry.backend == 'native'
            and entry.recipe.recipe_id == 'plantcv_color_card_correct_color')
        identifiers = [self.controller.add_entry(entry.key) for _ in range(2)]
        components = [self.controller._component(identifier) for identifier in identifiers]
        recipe = components[0].instance.workflow.recipe
        context = SimpleNamespace(root=self.window._root, image=SimpleNamespace(identifier='image'), source_path=Path('fixture.png'))
        self.window._capture_correction_project_context = Mock(return_value=context)
        self.window._project_correction_bindings = (SimpleNamespace(image_id='image', recipe_id=recipe.recipe_id,
            recipe_sha256=recipe.identity.recipe_sha256, owner_id=None),)
        with patch('seedvision.benchmarking.native_correction_inputs.run_native_correction_files', return_value=object()) as runner:
            for index, (identifier, component) in enumerate(zip(identifiers, components)):
                paths = dict(target_rgb=f'target-{index}.png', target_mask=f'target-mask-{index}.png', source_mask=f'source-mask-{index}.png')
                component.instance = replace(component.instance,
                    workflow=replace(component.instance.workflow, input_paths=paths))
                function = self.controller._prepare_run(component, identifier, Path('fixture.png'))
                function(lambda: False)
                self.assertEqual(runner.call_args.args[1], dict(paths, source_rgb='fixture.png'))
                self.assertIs(runner.call_args.kwargs['cache'], component.cache)

        primary = self.window._project_correction_bindings[0]
        left, right = (SimpleNamespace(owner_id=component.instance.instance_id) for component in components)
        self.window._project_correction_bindings = (primary, left, right)
        self.controller._selected = identifiers[0]
        self.assertTrue(self.controller.remove_selected())
        self.assertEqual(self.window._project_correction_bindings, (primary, right))

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
        """4 related CatalogueNodesControllerTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_duplicate_default_instances_own_controls_cache_and_prerequisite_presence)
        self.scenario(self.check_test_native_target_uses_own_cache_and_existing_runner)
        self.scenario(self.check_test_correction_instances_never_borrow_primary_binding_or_sibling_paths)
        self.scenario(self.check_test_primary_wiring_revision_changes_only_when_its_edges_change)

    def test_validation_and_recovery(self):
        """2 related CatalogueNodesControllerTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_queue_waits_for_primary_and_stop_withholds_old_result)
        self.scenario(self.check_test_instance_limit_and_planned_node_reject_before_execution)


if __name__ == '__main__':
    unittest.main()
