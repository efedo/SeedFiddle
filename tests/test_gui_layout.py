"""Desktop layout contracts from the GUI review: menus, toolbar, panels, canvas."""
from __future__ import annotations

from tests.functional_units import functional_units

from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


@functional_units
class GuiLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from PySide6.QtWidgets import QApplication

        cls.application = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        from tests.review_fixtures import dispose_window, isolated_window

        self._directory = TemporaryDirectory(prefix="seedfiddle-layout-")
        self.addCleanup(self._directory.cleanup)
        root = Path(self._directory.name)
        (root / "images").mkdir()
        self.window = isolated_window(root)
        self.addCleanup(dispose_window, self.window)

    def check_test_every_shortcut_is_unique(self) -> None:
        from PySide6.QtGui import QAction

        shortcuts = Counter(
            action.shortcut().toString()
            for action in self.window.findChildren(QAction)
            if not action.shortcut().isEmpty()
        )
        self.assertEqual({key: count for key, count in shortcuts.items() if count > 1}, {})

    def check_test_toolbar_keeps_run_view_and_annotation_controls_only(self) -> None:
        actions = self.window.workflow_toolbar.actions()
        for moved in (
            self.window.results_action,
            self.window.search_node_action,
            self.window.redo_reference_action,
        ):
            self.assertNotIn(moved, actions)
        self.assertIs(actions[actions.index(self.window.analyze_action) + 1], self.window.stop_action)

    def check_test_menus_group_commands_and_end_file_with_exit(self) -> None:
        menus = {action.text().replace("&", ""): action.menu() for action in self.window.menuBar().actions()}
        self.assertEqual(list(menus), ["File", "Edit", "Analysis", "Learning", "Help"])
        file_actions = [action for action in menus["File"].actions() if not action.isSeparator()]
        self.assertIs(file_actions[-1], self.window.exit_action)
        edit = menus["Edit"].actions()
        self.assertIn(self.window.undo_reference_edit_action, edit)
        self.assertIn(self.window.redo_reference_action, edit)
        help_actions = menus["Help"].actions()
        for action in (self.window.operator_guide_action, self.window.shortcuts_action, self.window.about_action):
            self.assertIn(action, help_actions)

    def check_test_shortcut_reference_lists_window_commands(self) -> None:
        from seedvision.ui.help_menu import operator_guide_path, shortcut_rows, show_shortcuts

        rows = dict(shortcut_rows(self.window))
        self.assertEqual(rows["Run pipeline"], self.window.analyze_action.shortcut().toString())
        dialog = show_shortcuts(self.window)
        self.addCleanup(dialog.close)
        self.assertEqual(dialog.findChild(type(dialog.layout().itemAt(0).widget()), "shortcutTable").rowCount(), len(rows))
        self.assertTrue(operator_guide_path(Path(__file__).resolve().parents[1]).is_file())

    def check_test_overlay_filter_hides_non_matching_entries(self) -> None:
        menu = self.window.overlay_menu
        self.window._rebuild_overlay_menu("raw_image")
        self.window._overlay_filter_field.setText("wavelet")
        visible = [
            action for action in menu.actions()
            if action.isVisible() and not action.isSeparator()
            and action is not self.window._overlay_filter_action
        ]
        self.assertTrue(visible)
        self.assertTrue(all("wavelet" in action.text().casefold() for action in visible))
        self.window._overlay_filter_field.clear()
        self.assertGreater(
            sum(action.isVisible() for action in menu.actions()), len(visible)
        )

    def check_test_reference_summary_is_compact_and_details_are_on_demand(self) -> None:
        from seedvision.ui.main_window import _reference_summary_html

        html = _reference_summary_html("saved", seed_count=42, draft=True, bundled_available=True)
        self.assertIn("References saved", html)
        self.assertIn("42 seeds outlined", html)
        self.assertIn("Unapplied edits", html)
        self.assertNotIn("Bundled", html)
        self.assertTrue(self.window.reference_association_scroll.isHidden())
        self.window.reference_association_details_button.setChecked(True)
        self.assertFalse(self.window.reference_association_scroll.isHidden())

    def check_test_spin_decimals_never_round_a_value(self) -> None:
        from types import SimpleNamespace
        from seedvision.ui.spin_format import spin_decimals

        spec = SimpleNamespace(step=0.05, minimum=0.0, maximum=1.0)
        self.assertEqual(spin_decimals(spec, 0.5), 2)
        self.assertEqual(spin_decimals(spec, 0.125), 3)
        self.assertEqual(spin_decimals(SimpleNamespace(step=1, minimum=0, maximum=10), 3.0), 0)

    def check_test_node_cards_fold_viewer_outputs_behind_an_expander(self) -> None:
        item = self.window.pipeline_canvas.node_items["background_likelihood"]
        hidden = item.hidden_output_count()
        self.assertGreater(hidden, 0)
        collapsed = item.height
        item.set_outputs_expanded(True)
        self.assertEqual(item.hidden_output_count(), 0)
        self.assertGreater(item.height, collapsed)
        item.set_outputs_expanded(False)
        self.assertEqual(item.height, collapsed)
        for port_id, _ in item.visible_output_ports():
            self.assertFalse(
                item.node.output_port_types.get(port_id, "").startswith("ViewerOverlay:")
                and not item._output_connected(port_id)
            )

    def check_test_right_panel_pins_results_above_node_and_image_tabs(self) -> None:
        tabs = self.window.details_tabs
        self.assertEqual([tabs.tabText(index) for index in range(tabs.count())], ["Node", "Image"])
        self.assertTrue(self.window.right_panel.isAncestorOf(self.window.baseline_section))
        self.assertFalse(tabs.isAncestorOf(self.window.baseline_section))
        tabs.setCurrentIndex(1)
        self.window._pipeline_node_selected("background_likelihood")
        self.assertIs(tabs.currentWidget(), self.window.metadata_scroll)

    def test_workflow_lifecycle(self):
        """9 related GuiLayoutTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_every_shortcut_is_unique)
        self.scenario(self.check_test_toolbar_keeps_run_view_and_annotation_controls_only)
        self.scenario(self.check_test_menus_group_commands_and_end_file_with_exit)
        self.scenario(self.check_test_shortcut_reference_lists_window_commands)
        self.scenario(self.check_test_overlay_filter_hides_non_matching_entries)
        self.scenario(self.check_test_reference_summary_is_compact_and_details_are_on_demand)
        self.scenario(self.check_test_spin_decimals_never_round_a_value)
        self.scenario(self.check_test_node_cards_fold_viewer_outputs_behind_an_expander)
        self.scenario(self.check_test_right_panel_pins_results_above_node_and_image_tabs)


if __name__ == "__main__":
    unittest.main()
