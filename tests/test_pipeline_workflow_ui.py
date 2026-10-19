"""Main-workspace routing, editing and queued-delivery ownership."""

from tests.functional_units import functional_units
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from tests.catalogue_fixtures import CatalogueTestCase
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtWidgets import QApplication, QComboBox, QSpinBox
from seedvision.ui.main_window import MainWindow
from seedvision.ui.pipeline_workflow import _Job
from tests.review_fixtures import dispose_window


@functional_units
class PipelineWorkflowUiTests(CatalogueTestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def setUp(self):
        self.directory = TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / 'artifacts')
        self.root = Path(self.directory.name)
        from tests.test_project_analysis_ui import ProjectAnalysisUiTests
        self.window = ProjectAnalysisUiTests._window(self.root)
        self.controller = self.window.pipeline_workflow

    def tearDown(self):
        dispose_window(self.window)
        self.application.processEvents()
        self.directory.cleanup()

    def select_binary(self):
        index = next(i for i in range(1, self.controller.selector.count())
            if self.controller.selector.itemData(i).recipe_id == 'plantcv_seed_binary_working')
        self.controller.selector.setCurrentIndex(index)
        self.assertTrue(self.controller.active)
        self.window.pipeline_documents.set_frozen(False)
        return self.controller.workflow

    def seed_old_display(self):
        from seedvision.benchmarking.color_histogram import HueHistogramChart
        self.controller.result = object()
        self.controller.facts.setPlainText('Old measured area: 42 pixels')
        self.controller.histogram.set_chart(HueHistogramChart('available', ()))
        self.controller.histogram.show()
        self.controller._preview_candidates = [('old output', object(), {})]

    def assert_display_cleared(self):
        self.assertIsNone(self.controller.result)
        self.assertEqual(self.controller._preview_candidates, [])
        self.assertNotIn('Old measured area', self.controller.facts.toPlainText())
        self.assertTrue(self.controller.facts.isHidden())
        self.assertIsNone(self.controller.histogram.chart)
        self.assertTrue(self.controller.histogram.isHidden())
        self.assertTrue(self.controller.crop_selector.isHidden())

    def load_image(self, name='current.png'):
        from PySide6.QtGui import QImage
        image = QImage(32, 32, QImage.Format.Format_RGB888)
        image.fill(0x408020)
        path = self.root/name
        self.assertTrue(image.save(str(path)))
        self.assertTrue(self.window.image_view.load_image(path)[0])
        return path

    def check_test_actual_selector_swaps_canvas_and_restores_default_graph(self):
        default = self.window.pipeline
        self.window.pipeline_canvas.select_node('reference_texture_prototypes')
        baseline_selection = self.window._selected_pipeline_node
        workflow = self.select_binary()
        self.assertIs(self.window.pipeline_canvas.graph, self.controller.graph)
        self.assertEqual(set(self.controller.graph.nodes), {s.stage_id for s in workflow.recipe.stages})
        self.assertTrue(self.controller.facts.isHidden())
        self.assertEqual(self.controller.facts.toPlainText(), '')
        self.seed_old_display()
        self.controller.selector.setCurrentIndex(0)
        self.assertIs(self.window.pipeline_canvas.graph, default)
        self.assertFalse(self.controller.active)
        self.assertTrue(self.window.pipeline_inspector._unified_optimization)
        self.assertEqual(self.window._selected_pipeline_node, baseline_selection)
        self.assert_display_cleared()
        self.controller.install(None)
        self.assertEqual(self.window._selected_pipeline_node, baseline_selection)

    def check_test_startup_pipeline_toolbar_and_compact_workflow_choices(self):
        self.window.resize(1440, 900)
        self.window.show()
        self.application.processEvents()
        self.assertTrue(self.window.pipeline_workspace_action.isChecked())
        self.assertTrue(self.window.pipeline_canvas.isVisible())
        selected_item = self.window.pipeline_canvas.node_items['project']
        visible = self.window.pipeline_canvas.mapToScene(self.window.pipeline_canvas.viewport().rect()).boundingRect()
        self.assertTrue(visible.intersects(selected_item.sceneBoundingRect()))
        fraction = self.window.image_list.height() / self.window.height()
        self.assertGreater(fraction, .27)
        self.assertLess(fraction, .39)
        self.assertIs(self.controller.strip.parent(), self.window.pipeline_canvas.control_bar)
        self.assertLessEqual(self.controller.selector.width(), 130)
        self.assertFalse(hasattr(self.controller, 'run_button'))
        self.assertFalse(hasattr(self.controller, 'stop_button'))
        actions = self.window.workflow_toolbar.actions()
        self.assertIs(actions[actions.index(self.window.analyze_action)+1], self.window.stop_action)
        identifiers = {self.controller.selector.itemData(index).recipe_id
            for index in range(1, self.controller.selector.count())}
        self.assertEqual(identifiers, {'plantcv_color_card_correct_color', 'plantcv_seed_binary_working',
            'plantcv_seed_mean_working', 'plantcv_seed_otsu_working', 'smpts_paper_fixed_mean'})
        self.window.resize(1050, 650)
        self.application.processEvents()
        self.assertGreaterEqual(self.controller.strip.width(), self.controller.strip.sizeHint().width())
        self.assertEqual(self.controller.export_menu_action.isVisible(), self.controller.export_button.isHidden())
        self.assertTrue(self.window.pipeline_canvas.add_node_button.isVisible())
        self.assertLessEqual(self.controller.strip.geometry().right(), self.window.pipeline_canvas.control_bar.width())
        canvas = self.window.pipeline_canvas
        canvas._populate_node_catalogue()
        self.assertEqual(canvas.node_catalogue_list.count(), 122)
        colours = {canvas.node_catalogue_list.item(index).foreground().color().name()
            for index in range(canvas.node_catalogue_list.count())}
        self.assertEqual(colours, {'#1768ad', '#218044'})

    def check_test_typed_controls_and_choice_edit_reach_engine_contract(self):
        self.select_binary()
        stage = next(s for s in self.controller.workflow.recipe.stages if s.operation_id == 'threshold.binary')
        self.window.pipeline_canvas.select_node(stage.stage_id)
        node = self.controller.graph.node(stage.stage_id)
        node.validate_parameter_sections()
        choice = next(spec for spec in node.parameter_specs if spec.kind == 'choice')
        widget = self.window.pipeline_inspector.findChild(QComboBox, 'parameter_'+choice.key)
        self.assertIsNotNone(widget)
        index = (widget.currentIndex()+1) % widget.count()
        selected_value = widget.itemText(index)
        self.seed_old_display()
        widget.setCurrentIndex(index)
        self.assert_display_cleared()
        self.assertEqual(self.controller.workflow.recipe.parameters_for_runner(stage.stage_id)[choice.key], selected_value)
        for spec in node.parameter_specs:
            if spec.kind == 'int':
                self.assertIsNotNone(self.window.pipeline_inspector.findChild(QSpinBox, 'parameter_'+spec.key))

    def check_test_actual_run_click_uses_current_image_and_full_scope(self):
        self.select_binary()
        self.load_image()
        self.controller.sync_buttons()
        self.seed_old_display()
        with patch.object(self.controller, 'start_job') as start, patch('seedvision.pipeline.workflows.run_workflow_path', return_value='computed') as run:
            self.window.analyze_action.trigger()
            self.assert_display_cleared()
            function = start.call_args.args[0]
            self.assertEqual(function(lambda: False), 'computed')
        self.assertEqual(run.call_args.args[1], self.window.image_view.image_path)
        self.assertIsNone(run.call_args.kwargs['target_node_id'])

    def check_test_canvas_run_to_node_signal_routes_selected_scope(self):
        self.select_binary()
        selected = self.controller._selected
        with patch.object(self.controller, 'run') as run:
            self.window.pipeline_canvas.run_to_node_requested.emit(selected)
        run.assert_called_once_with(selected)

    def check_test_queued_delivery_after_image_switch_is_discarded(self):
        self.select_binary()
        path = self.load_image('old.png')
        job = _Job(lambda cancel: 'obsolete', (self.controller._generation, path))
        self.controller._job = job
        from seedvision.pipeline import NodeStatus
        for node in self.controller.graph.nodes.values():
            node.status = NodeStatus.COMPLETE
        self.load_image('replacement.png')
        self.controller.image_changed()
        self.assertTrue(all(node.status == NodeStatus.IDLE for node in self.controller.graph.nodes.values()))
        completion = unittest.mock.Mock()
        self.controller.job_completed(job, 'obsolete', completion)
        completion.assert_not_called()
        self.assertIsNone(self.controller._job)

    def check_test_same_path_reopen_reuses_unchanged_source_and_clears_changed_source(self):
        from PySide6.QtGui import QImage
        from seedvision.pipeline import NodeStatus
        import torch
        self.select_binary()
        path = self.load_image()
        self.window._add_and_open_image(str(path))
        threshold = next(stage for stage in self.controller.workflow.recipe.stages
            if stage.operation_id == 'threshold.binary')
        self.window.pipeline_canvas.select_node(threshold.stage_id)
        old_result = SimpleNamespace(coordinate_frame=SimpleNamespace(frame_id='same-source'),
            node=lambda identifier: SimpleNamespace(status='executed', reason='Existing result',
                operation_id='threshold.binary', outputs={'mask': torch.zeros((32, 32), dtype=torch.uint8)}))
        self.controller.result = old_result
        old_cache = self.controller.cache
        old_cache.values['test-owned-product'] = object()
        prepared = object()
        self.controller._correction_prepared = prepared
        generation = self.controller._generation
        self.window._open_path(path)
        self.assertIs(self.controller.result, old_result)
        self.assertIs(self.controller.cache, old_cache)
        self.assertIs(self.controller._correction_prepared, prepared)
        self.assertEqual(self.controller._generation, generation)

        self.seed_old_display()
        self.controller.result = old_result
        bypass = next(node for node in self.controller.graph.nodes.values() if node.bypassable)
        for node in self.controller.graph.nodes.values():
            node.status = NodeStatus.COMPLETE
        bypass.enabled = False
        bypass.status = NodeStatus.BYPASSED
        job = _Job(lambda cancel: 'obsolete same-path result', (generation, path))
        self.controller._job = job
        replacement = QImage(48, 24, QImage.Format.Format_RGB888)
        replacement.fill(0x906020)
        self.assertTrue(replacement.save(str(path)))
        self.window._open_path(path)
        self.assertEqual(self.window.image_view.image_size, (48, 24))
        self.assert_display_cleared()
        self.assertIsNot(self.controller.cache, old_cache)
        self.assertIsNone(self.controller._correction_prepared)
        self.assertGreater(self.controller._generation, generation)
        self.assertFalse(self.controller.export_button.isEnabled())
        self.assertTrue(all(node.status == (NodeStatus.BYPASSED if node is bypass else NodeStatus.IDLE)
            for node in self.controller.graph.nodes.values()))
        completion = unittest.mock.Mock()
        self.controller.job_completed(job, 'obsolete same-path result', completion)
        completion.assert_not_called()

    def check_test_project_reset_restores_default_and_drops_copies(self):
        self.select_binary()
        self.controller._copies['old'] = self.controller.workflow
        self.seed_old_display()
        self.window._clear_project_session_state()
        self.assertIs(self.window.pipeline_canvas.graph, self.window.pipeline)
        self.assertFalse(self.controller.active)
        self.assertEqual(self.controller._copies, {})
        self.assertEqual(self.controller.selector.currentIndex(), 0)
        self.assert_display_cleared()

    def check_test_install_reopened_workflow_updates_selector(self):
        from seedvision.pipeline.workflows import create_workflow
        recipe = self.controller.selector.itemData(3)
        self.controller.install(create_workflow(recipe))
        self.assertEqual(self.controller.selector.currentIndex(), 3)
        self.assertIs(self.window.pipeline_canvas.graph, self.controller.graph)

    def check_test_correction_inputs_are_contextual_and_source_is_current_image(self):
        index = next(i for i in range(1, self.controller.selector.count())
            if self.controller.selector.itemData(i).recipe_id == 'plantcv_color_card_correct_color')
        self.controller.selector.setCurrentIndex(index)
        self.window.pipeline_documents.set_frozen(False)
        from PySide6.QtWidgets import QPushButton
        texts = {w.text() for w in self.controller.node_actions.findChildren(QPushButton)}
        self.assertIn('Target image…', texts)
        self.assertIn('Source chip mask…', texts)
        self.assertNotIn('Source image…', texts)
        self.load_image()
        from dataclasses import replace
        self.controller.workflow = replace(self.controller.workflow, input_paths={
            'target_rgb': str(self.root/'target.png'), 'target_mask': str(self.root/'target-mask.png'),
            'source_mask': str(self.root/'source-mask.png')})
        with patch.object(self.window, '_capture_correction_project_context', return_value=None), patch.object(self.controller, 'start_job') as start, patch('seedvision.benchmarking.native_correction_inputs.run_native_correction_files', return_value=('result', None)) as run:
            self.controller.run()
            self.assertEqual(start.call_args.args[0](lambda: False), ('result', None))
        self.assertEqual(run.call_args.args[1]['source_rgb'], str(self.window.image_view.image_path))

    def check_test_actual_save_and_open_restore_native_working_values(self):
        self.select_binary()
        path = self.load_image()
        self.window._add_and_open_image(str(path))
        stage = next(s for s in self.controller.workflow.recipe.stages if s.operation_id == 'threshold.binary')
        self.controller.parameter_changed(stage.stage_id, 'threshold', 117)
        self.window.pipeline_canvas.select_node(stage.stage_id)
        from seedvision.pipeline.catalogue_nodes import catalogue_entries
        entry = next(entry for entry in catalogue_entries() if entry.backend == 'native'
            and entry.recipe.stage(entry.target_node).operation_id == 'threshold.binary')
        added = self.window.catalogue_nodes.add_entry(entry.key)
        self.window.catalogue_nodes.parameter_changed(added, 'threshold', 143)
        self.window.catalogue_nodes.output_requested(added, 'mask')
        self.window.image_workspace_action.setChecked(False)
        destination = self.root/'temporary.seedfiddle-project.json'
        from PySide6.QtWidgets import QFileDialog, QMessageBox
        with patch.object(QFileDialog, 'getSaveFileName', return_value=(str(destination), '')), patch.object(QMessageBox, 'warning'), patch.object(QMessageBox, 'critical') as critical:
            self.window.save_project_action.trigger()
            critical.assert_not_called()
        self.assertTrue(destination.is_file())
        import json
        self.assertEqual(json.loads(destination.read_text())['node_instances'][0].get('selected_node'), entry.target_node)
        self.controller.selector.setCurrentIndex(0)
        with patch.object(self.window, '_confirm_project_replacement', return_value=True), patch.object(QMessageBox, 'warning'), patch.object(QMessageBox, 'critical') as critical:
            self.assertTrue(self.window._open_project(destination))
            critical.assert_not_called()
        self.assertTrue(self.controller.active)
        self.assertEqual(self.controller.workflow.recipe.parameters_for_runner(stage.stage_id)['threshold'], 117)
        self.assertEqual(self.controller.selector.currentData().recipe_id, 'plantcv_seed_binary_working')
        self.assertEqual(self.controller._selected, stage.stage_id)
        self.assertFalse(self.window.image_workspace_action.isChecked())
        self.assertTrue(self.window.pipeline_workspace_action.isChecked())
        self.assertEqual(self.window.catalogue_nodes._selected, added)
        self.assertEqual(self.window.catalogue_nodes._selected_output, 'mask')
        component = self.window.catalogue_nodes._component(added)
        self.assertEqual(component.instance.workflow.recipe.parameters_for_runner(entry.target_node)['threshold'], 143)

    def check_test_structured_points_preserve_fractional_values(self):
        from seedvision.ui.workflow_parameters import WorkflowParameterEditor
        editor = WorkflowParameterEditor('points', '[[1.25, 2.5], [10.75, 20.5]]')
        emitted = []
        editor.changed.connect(emitted.append)
        editor.fields[0].setValue(1.375)
        editor.emit_value()
        import json
        self.assertEqual(json.loads(emitted[-1]), [[1.375, 2.5], [10.75, 20.5]])
        editor.deleteLater()

    def check_test_range_colorspace_change_adapts_bounds_field_count(self):
        recipe = next(recipe for recipe in self.controller._all_templates if recipe.recipe_id == 'plantcv_threshold_custom_range')
        self.controller.load_recipe(recipe)
        self.window.pipeline_documents.set_frozen(False)
        stage = next(s for s in self.controller.workflow.recipe.stages if s.operation_id == 'threshold.custom_range')
        self.controller.parameter_changed(stage.stage_id, 'channel', 'CMYK')
        values = self.controller.workflow.recipe.parameters_for_runner(stage.stage_id)
        self.assertEqual(len(values['lower']), 4)
        self.assertEqual(len(values['upper']), 4)

    def check_test_actual_enabled_checkbox_bypasses_only_selected_node(self):
        self.select_binary()
        stage = next(s for s in self.controller.workflow.recipe.stages if s.operation_id == 'filter.median_blur')
        self.window.pipeline_canvas.select_node(stage.stage_id)
        self.assertTrue(self.window.pipeline_inspector.enabled_checkbox.isEnabled())
        self.window.pipeline_inspector.enabled_checkbox.click()
        self.assertEqual(self.controller.workflow.disabled_nodes, (stage.stage_id,))
        self.assertTrue(all(node.enabled for identifier, node in self.controller.graph.nodes.items() if identifier != stage.stage_id))

    def check_test_learning_menu_actions_preserve_native_training_routes(self):
        with patch.object(self.controller, 'train_model') as train:
            self.window.train_native_bayes_action.trigger()
            self.window.train_native_kmeans_action.trigger()
        self.assertEqual(train.call_args_list, [unittest.mock.call('bayes'), unittest.mock.call('kmeans')])

    def test_workflow_lifecycle(self):
        """15 related PipelineWorkflowUiTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_actual_selector_swaps_canvas_and_restores_default_graph)
        self.scenario(self.check_test_startup_pipeline_toolbar_and_compact_workflow_choices)
        self.scenario(self.check_test_typed_controls_and_choice_edit_reach_engine_contract)
        self.scenario(self.check_test_actual_run_click_uses_current_image_and_full_scope)
        self.scenario(self.check_test_canvas_run_to_node_signal_routes_selected_scope)
        self.scenario(self.check_test_queued_delivery_after_image_switch_is_discarded)
        self.scenario(self.check_test_same_path_reopen_reuses_unchanged_source_and_clears_changed_source)
        self.scenario(self.check_test_project_reset_restores_default_and_drops_copies)
        self.scenario(self.check_test_install_reopened_workflow_updates_selector)
        self.scenario(self.check_test_correction_inputs_are_contextual_and_source_is_current_image)
        self.scenario(self.check_test_actual_save_and_open_restore_native_working_values)
        self.scenario(self.check_test_structured_points_preserve_fractional_values)
        self.scenario(self.check_test_range_colorspace_change_adapts_bounds_field_count)
        self.scenario(self.check_test_actual_enabled_checkbox_bypasses_only_selected_node)
        self.scenario(self.check_test_learning_menu_actions_preserve_native_training_routes)


if __name__ == '__main__':
    unittest.main()
