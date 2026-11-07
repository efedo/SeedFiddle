"""Main-workspace routing between the default pipelines and the shared project nodes."""

from tests.functional_units import functional_units
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from seedvision.pipeline.default_pipelines import CELLPOSE_SEED_ANALYSIS, PROCEDURAL_SEED_ANALYSIS
from seedvision.ui.pipeline_workflow import DEFAULTS_HEADING, ENTRY_KIND, SELECTOR_WIDTH
from tests.review_fixtures import dispose_window, editable_copy
from seedvision.pipeline.project_context import CONTEXT_NODE_IDS

INSTALLED = 'seedvision.learning.cellpose_sam.cellpose_installed'


@functional_units
class PipelineWorkflowUiTests(unittest.TestCase):
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

    def choose(self, pipeline):
        with patch(INSTALLED, return_value=True):
            self.controller.select_template(pipeline.recipe_id)
        self.application.processEvents()

    def load_image(self, name='current.png'):
        from PySide6.QtGui import QImage
        image = QImage(32, 32, QImage.Format.Format_RGB888)
        image.fill(0x408020)
        path = self.root/name
        self.assertTrue(image.save(str(path)))
        self.assertTrue(self.window.image_view.load_image(path)[0])
        return path

    def check_test_shared_project_nodes_and_two_way_image_selection(self):
        first, second = self.load_image('first.png'), self.load_image('second.png')
        for path in (first, second):
            self.window._add_image(path, open_now=False)
        self.window._open_path(first)
        for pipeline in (PROCEDURAL_SEED_ANALYSIS, CELLPOSE_SEED_ANALYSIS):
            with self.subTest(pipeline=pipeline.title):
                self.choose(pipeline)
                self.assertIs(self.controller.graph_pipeline, pipeline)
                self.assertIs(self.window.pipeline_canvas.graph, self.window.pipeline)
                self.assertTrue(CONTEXT_NODE_IDS <= self.window.pipeline_canvas.graph.nodes.keys())
                self.window.pipeline_canvas.select_node('image_selector')
                combo = self.window.pipeline_inspector.image_selector_combo
                self.assertFalse(combo.isHidden())
                self.assertTrue(self.window.pipeline_documents.locked)
                self.assertTrue(combo.isEnabled())
                combo.setCurrentIndex(combo.findData(str(second)))
                self.assertEqual(self.window.image_view.image_path, second)
                self.assertEqual(self.window.image_list.currentItem().data(Qt.ItemDataRole.UserRole), str(second))
                self.window._open_list_item(self.window.image_list.item(0))
                self.assertEqual(self.window.image_view.image_path, first)
                self.assertEqual(combo.currentData(), str(first))
                self.window.pipeline_canvas.select_node('metadata')
                self.assertFalse(self.window.pipeline_inspector.context_info_label.isHidden())
                self.window.pipeline_canvas.select_node('species_reference_library')
                self.assertFalse(self.window.pipeline_inspector.species_library_button.isHidden())
                self.assertIn('No species library pinned', self.window.pipeline_inspector.context_info_label.text())
                self.window.pipeline_canvas.select_node('project')
                self.assertFalse(self.window.pipeline_inspector.project_info_container.isHidden())
                with patch.object(self.window, '_analyze_current_image') as analyze:
                    self.window._run_pipeline_to_node('image_selector')
                    analyze.assert_not_called()
        self.window._remove_image(second)
        self.assertEqual(self.window.pipeline_inspector.image_selector_combo.count(), 1)

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
        self.assertFalse(hasattr(self.controller, 'run_button'))
        self.assertFalse(hasattr(self.controller, 'stop_button'))
        actions = self.window.workflow_toolbar.actions()
        self.assertIs(actions[actions.index(self.window.analyze_action)+1], self.window.stop_action)
        # One grouped list holds the default pipelines, then the project's copies.
        selector = self.controller.selector
        templates = [index for index in range(selector.count()) if selector.itemData(index, ENTRY_KIND) == 'template']
        self.assertEqual([selector.itemData(index).recipe_id for index in templates],
                         [PROCEDURAL_SEED_ANALYSIS.recipe_id, CELLPOSE_SEED_ANALYSIS.recipe_id])
        headings = [selector.itemText(index) for index in range(selector.count())
                    if selector.itemData(index, ENTRY_KIND) == 'heading']
        self.assertEqual(headings, [DEFAULTS_HEADING.upper()])
        self.assertFalse(any(selector.model().item(index).isEnabled() for index in range(selector.count())
                             if selector.itemData(index, ENTRY_KIND) in ('heading', 'saved-heading')))
        self.assertFalse(any(action.menu() for action in self.controller.more_button.menu().actions()))
        self.assertEqual(selector.itemData(selector.currentIndex(), ENTRY_KIND), 'custom')
        self.assertEqual(selector.currentText(), self.window.pipeline_documents.current_name)
        self.window.resize(1050, 650)
        self.application.processEvents()
        self.assertGreaterEqual(self.controller.strip.width(), self.controller.strip.sizeHint().width())
        # The list only gives up width once every optional strip widget is hidden.
        if self.controller.selector.width() < SELECTOR_WIDTH:
            self.assertTrue(self.controller.export_button.isHidden())
        # The compact icons never hide: nothing else reaches their commands.
        self.assertFalse(self.controller.load_button.isHidden())
        self.assertFalse(self.controller.save_button.isHidden())
        self.window.resize(1440, 900)
        self.application.processEvents()
        self.assertEqual(self.controller.selector.width(), SELECTOR_WIDTH)
        self.window.resize(1050, 650)
        self.application.processEvents()
        self.assertEqual(self.controller.export_menu_action.isVisible(), self.controller.export_button.isHidden())
        self.assertTrue(self.window.pipeline_canvas.add_node_button.isVisible())
        self.assertLessEqual(self.controller.strip.geometry().right(), self.window.pipeline_canvas.control_bar.width())

    def check_test_project_reset_restores_default_and_drops_copies(self):
        self.choose(CELLPOSE_SEED_ANALYSIS)
        editable_copy(self.window, 'Cellpose copy')
        documents = self.window.pipeline_documents
        self.assertFalse(documents.locked)
        self.assertTrue(self.controller._entries('custom'))
        self.window._clear_project_session_state()
        self.assertIs(self.window.pipeline_canvas.graph, self.window.pipeline)
        self.assertTrue(documents.locked)
        self.assertEqual(documents.custom, {})
        self.assertEqual(self.controller._entries('custom', 'saved-heading'), [])
        self.assertEqual(self.controller.selector.currentIndex(), self.controller.index_of(None))
        self.assertEqual(self.controller.selector.currentText(), self.controller.graph_pipeline.title)

    def check_test_templates_are_arranged_when_chosen_and_reloads_keep_their_layout(self):
        canvas = self.window.pipeline_canvas
        self.controller.select_template(None)
        # The startup project's Procedural seed analysis is laid out in columns.
        self.assertEqual({round(node.x) % 270 for node in self.window.pipeline.nodes.values()}, {0})
        with patch.object(canvas, 'auto_arrange', wraps=canvas.auto_arrange) as arrange:
            self.choose(CELLPOSE_SEED_ANALYSIS)
            arrange.assert_called_once_with()
            self.assertEqual({round(node.x) % 270 for node in self.window.pipeline.nodes.values()}, {0})
            identifier = CELLPOSE_SEED_ANALYSIS.selected_node
            canvas.node_items[identifier].setPos(1234.0, -567.0)
            arrange.reset_mock()
            # Reinstalling the graph, as Open project and Load pipeline do, keeps its layout.
            self.controller.install()
            arrange.assert_not_called()
        node = self.window.pipeline.node(identifier)
        self.assertEqual((node.x, node.y), (1234.0, -567.0))
        self.assertTrue(canvas.auto_arrange_button.isVisibleTo(canvas.control_bar))

    def check_test_actual_save_and_open_restore_working_values_and_added_nodes(self):
        path = self.load_image()
        self.window._add_and_open_image(str(path))
        self.window.pipeline.set_parameter('ruler_detection', 'ruler_length_mm', 183.0)
        self.window.pipeline_canvas.select_node('ruler_detection')
        catalogue = self.window.catalogue_nodes
        added = catalogue.add_entry('default:seed_scale_estimation')
        self.assertTrue(catalogue.parameter_changed(added, 'reference_max_components', 7))
        port = catalogue._component(added).graph.node(added).output_ports[0][0]
        catalogue.output_requested(added, port)
        self.window.image_workspace_action.setChecked(False)
        destination = self.root/'temporary.seedfiddle-project.json'
        from PySide6.QtWidgets import QFileDialog, QMessageBox
        with patch.object(QFileDialog, 'getSaveFileName', return_value=(str(destination), '')), \
             patch.object(QMessageBox, 'warning'), patch.object(QMessageBox, 'critical') as critical:
            self.window.save_project_action.trigger()
            critical.assert_not_called()
        self.assertTrue(destination.is_file())
        saved = json.loads(destination.read_text(encoding='utf-8'))
        self.assertEqual(saved['node_instances'][0].get('selected_node'), 'seed_scale_estimation')
        self.controller.select_template(None)
        with patch.object(self.window, '_confirm_project_replacement', return_value=True), \
             patch.object(QMessageBox, 'warning'), patch.object(QMessageBox, 'critical') as critical:
            self.assertTrue(self.window._open_project(destination))
            critical.assert_not_called()
        self.assertEqual(self.window.pipeline.node('ruler_detection').parameters['ruler_length_mm'], 183.0)
        self.assertEqual(self.controller.selector.currentText(), self.window.pipeline_documents.current_name)
        self.assertFalse(self.window.image_workspace_action.isChecked())
        self.assertTrue(self.window.pipeline_workspace_action.isChecked())
        self.assertEqual(catalogue._selected, added)
        self.assertEqual(catalogue._selected_output, port)
        restored = catalogue._component(added)
        self.assertEqual(restored.graph.node(added).parameters['reference_max_components'], 7)

    def test_workflow_lifecycle(self):
        """5 related PipelineWorkflowUiTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_shared_project_nodes_and_two_way_image_selection)
        self.scenario(self.check_test_startup_pipeline_toolbar_and_compact_workflow_choices)
        self.scenario(self.check_test_project_reset_restores_default_and_drops_copies)
        self.scenario(self.check_test_templates_are_arranged_when_chosen_and_reloads_keep_their_layout)
        self.scenario(self.check_test_actual_save_and_open_restore_working_values_and_added_nodes)


if __name__ == '__main__':
    unittest.main()
