"""Procedural and Cellpose seed analysis are one catalogue with different active nodes."""


from tests.functional_units import functional_units
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtWidgets import QApplication, QFileDialog, QInputDialog, QMessageBox

from seedvision.persistence.analysis_settings import (
    analysis_settings_profile_from_graph, apply_analysis_settings_profile,
)
from seedvision.pipeline import build_default_pipeline
from seedvision.pipeline.default_pipelines import (
    CELLPOSE_NODE, CELLPOSE_SEED_ANALYSIS, PROCEDURAL_SEED_ANALYSIS,
    build_cellpose_pipeline, build_procedural_pipeline, graph_pipeline_of,
)
from tests.review_fixtures import dispose_window, isolated_window

YES = QMessageBox.StandardButton.Yes
CELLPOSE_NODES = {'project', 'image_selector', 'metadata', 'deskew_colour', 'species_reference_library',
                  'ruler_detection', 'layout_detection', 'seed_scale_estimation', CELLPOSE_NODE}
INSTALLED = 'seedvision.learning.cellpose_sam.cellpose_installed'


@functional_units
class GraphDefaultPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.window = isolated_window(self.root, editable=False)
        self.presenter = self.window.pipeline_workflow
        self.documents = self.window.pipeline_documents
        self.selector = self.presenter.selector

    def tearDown(self):
        dispose_window(self.window)
        self.application.processEvents()
        self.temporary.cleanup()

    def choose(self, pipeline):
        with patch(INSTALLED, return_value=True):
            self.selector.setCurrentIndex(self.presenter.index_of(pipeline.recipe_id))
        self.application.processEvents()

    def assert_on(self, pipeline, active):
        self.assertEqual(set(self.window.pipeline.nodes), active)
        self.assertEqual(self.selector.currentText(), pipeline.title)
        self.assertEqual(self.documents.current_name, pipeline.title)
        self.assertTrue(self.documents.locked)
        self.assertEqual(set(self.window.pipeline_canvas.node_items), active)

    def save_project(self, destination):
        with patch.object(QFileDialog, 'getSaveFileName', return_value=(str(destination), '')):
            self.assertTrue(self.window._save_project_as())

    def check_test_procedural_excludes_only_cellpose(self):
        whole, graph = build_default_pipeline(), build_procedural_pipeline()
        self.assertEqual(set(graph.nodes), set(whole.nodes) - {CELLPOSE_NODE})
        self.assertIn(CELLPOSE_NODE, graph.unused_nodes)
        self.assertFalse(graph.node(CELLPOSE_NODE).enabled)
        self.assertTrue(graph.node('procedural_instances').enabled)
        for identifier, node in graph.nodes.items():
            self.assertEqual(node.enabled, whole.node(identifier).enabled, identifier)
        analysis_settings_profile_from_graph(graph)

    def check_test_cellpose_keeps_only_what_cellpose_needs(self):
        graph = build_cellpose_pipeline()
        self.assertEqual(set(graph.nodes), CELLPOSE_NODES)
        self.assertTrue(all(node.enabled for node in graph.nodes.values()))
        self.assertFalse(any(node.enabled for node in graph.unused_nodes.values()))
        # Nothing it needs was left out, and nothing is left without a consumer.
        for identifier in graph.nodes:
            self.assertEqual(graph.missing_input_ports(identifier), ())
            if identifier != CELLPOSE_NODE:
                self.assertIn(CELLPOSE_NODE, graph.downstream(identifier, recursive=True))
        # The same catalogue, so either pipeline's settings load onto either graph.
        whole = build_default_pipeline()
        self.assertEqual({**graph.nodes, **graph.unused_nodes}.keys(), {**whole.nodes, **whole.unused_nodes}.keys())
        apply_analysis_settings_profile(whole, analysis_settings_profile_from_graph(graph))
        self.assertEqual(set(whole.nodes), CELLPOSE_NODES)

    def check_test_graph_says_which_pipeline_it_came_from(self):
        self.assertIs(graph_pipeline_of(build_procedural_pipeline().nodes), PROCEDURAL_SEED_ANALYSIS)
        self.assertIs(graph_pipeline_of(build_cellpose_pipeline().nodes), CELLPOSE_SEED_ANALYSIS)

    def check_test_list_changes_the_graph_and_back(self):
        procedural = set(build_procedural_pipeline().nodes)
        self.assert_on(PROCEDURAL_SEED_ANALYSIS, procedural)
        self.choose(CELLPOSE_SEED_ANALYSIS)
        self.assert_on(CELLPOSE_SEED_ANALYSIS, CELLPOSE_NODES)
        self.assertEqual(self.window._selected_pipeline_node, CELLPOSE_NODE)
        self.assertEqual(self.window.pipeline_inspector._node.identifier, CELLPOSE_NODE)
        self.assertTrue(self.window.pipeline.node(CELLPOSE_NODE).enabled)
        self.choose(PROCEDURAL_SEED_ANALYSIS)
        self.assert_on(PROCEDURAL_SEED_ANALYSIS, procedural)
        self.assertFalse(self.window.pipeline.node(CELLPOSE_NODE).enabled)
        self.assertEqual(self.window._selected_pipeline_node, 'seed_scale_estimation')

    def check_test_changing_pipeline_keeps_shared_nodes_and_clears_the_open_result(self):
        key = self.window._current_image_key() or 'image'
        self.window._analysis_caches[key] = object()
        self.window._analyses[key] = object()
        self.choose(CELLPOSE_SEED_ANALYSIS)
        self.assertNotIn(key, self.window._analyses)
        dirty = self.window._cache_dirty_nodes[key]
        self.assertIn(CELLPOSE_NODE, dirty)
        self.assertIn('procedural_instances', dirty)
        self.assertFalse(dirty & (CELLPOSE_NODES - {CELLPOSE_NODE}))
        del self.window._analysis_caches[key]

    def check_test_overlays_follow_the_chosen_pipeline(self):
        def modes():
            return {self.window.overlay_combo.itemData(index) for index in range(self.window.overlay_combo.count())}
        self.assertNotIn('cellpose_sam_instances', modes())
        self.assertIn('procedural_instances', modes())
        self.choose(CELLPOSE_SEED_ANALYSIS)
        self.assertIn('cellpose_sam_instances', modes())
        self.assertNotIn('procedural_instances', modes())

    def check_test_missing_package_keeps_the_current_pipeline(self):
        with (patch(INSTALLED, return_value=False), patch.object(QMessageBox, 'warning') as warning):
            self.selector.setCurrentIndex(self.presenter.index_of(CELLPOSE_SEED_ANALYSIS.recipe_id))
        self.assertEqual(warning.call_args.args[1], 'Cellpose-SAM is not installed')
        self.assert_on(PROCEDURAL_SEED_ANALYSIS, set(build_procedural_pipeline().nodes))

    def check_test_a_copy_keeps_its_graph_across_pipeline_changes(self):
        self.choose(CELLPOSE_SEED_ANALYSIS)
        with (patch.object(QMessageBox, 'question', return_value=YES),
              patch.object(QInputDialog, 'getText', return_value=('Mine', True)) as prompt):
            self.presenter.rename_button.click()
        self.assertEqual(prompt.call_args.kwargs['text'], 'Cellpose seed analysis copy')
        self.assertEqual(self.documents.factory_name, CELLPOSE_SEED_ANALYSIS.title)
        self.assertFalse(self.documents.locked)
        self.window.pipeline.restore_unused_node('hue_only')
        self.choose(PROCEDURAL_SEED_ANALYSIS)
        self.assert_on(PROCEDURAL_SEED_ANALYSIS, set(build_procedural_pipeline().nodes))
        self.selector.setCurrentIndex(self.selector.findText('Mine'))
        self.assertEqual(self.documents.current_name, 'Mine')
        self.assertEqual(self.documents.factory_name, CELLPOSE_SEED_ANALYSIS.title)
        self.assertEqual(set(self.window.pipeline.nodes), CELLPOSE_NODES | {'hue_only'})
        self.choose(CELLPOSE_SEED_ANALYSIS)
        self.assert_on(CELLPOSE_SEED_ANALYSIS, CELLPOSE_NODES)

    def check_test_titles_are_reserved_for_copies(self):
        for pipeline in (PROCEDURAL_SEED_ANALYSIS, CELLPOSE_SEED_ANALYSIS):
            self.assertEqual(self.documents.unique_name(pipeline.title), pipeline.title + ' 2')

    def check_test_project_reopens_on_the_pipeline_it_was_saved_on(self):
        self.choose(CELLPOSE_SEED_ANALYSIS)
        destination = self.root / 'cellpose.seedfiddle-project.json'
        self.save_project(destination)
        saved = json.loads(destination.read_text(encoding='utf-8'))
        self.assertNotIn('active_pipeline', saved)
        self.choose(PROCEDURAL_SEED_ANALYSIS)
        self.window._set_project_dirty(False)
        self.assertTrue(self.window._open_project(destination))
        self.assert_on(CELLPOSE_SEED_ANALYSIS, CELLPOSE_NODES)

    def check_test_new_project_starts_on_procedural(self):
        self.choose(CELLPOSE_SEED_ANALYSIS)
        self.window._set_project_dirty(False)
        self.window._install_new_project()
        self.assert_on(PROCEDURAL_SEED_ANALYSIS, set(build_procedural_pipeline().nodes))

    def check_test_training_asks_for_procedural(self):
        self.choose(CELLPOSE_SEED_ANALYSIS)
        with patch.object(QMessageBox, 'information') as information:
            self.window._start_learning_training()
        self.assertEqual(information.call_args.args[1], 'Choose Procedural seed analysis')

    def test_the_two_graphs(self):
        """3 related GraphDefaultPipelineTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_procedural_excludes_only_cellpose)
        self.scenario(self.check_test_cellpose_keeps_only_what_cellpose_needs)
        self.scenario(self.check_test_graph_says_which_pipeline_it_came_from)

    def test_choosing_between_them(self):
        """9 related GraphDefaultPipelineTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_list_changes_the_graph_and_back)
        self.scenario(self.check_test_changing_pipeline_keeps_shared_nodes_and_clears_the_open_result)
        self.scenario(self.check_test_overlays_follow_the_chosen_pipeline)
        self.scenario(self.check_test_missing_package_keeps_the_current_pipeline)
        self.scenario(self.check_test_a_copy_keeps_its_graph_across_pipeline_changes)
        self.scenario(self.check_test_titles_are_reserved_for_copies)
        self.scenario(self.check_test_project_reopens_on_the_pipeline_it_was_saved_on)
        self.scenario(self.check_test_new_project_starts_on_procedural)
        self.scenario(self.check_test_training_asks_for_procedural)


if __name__ == '__main__':
    unittest.main()
