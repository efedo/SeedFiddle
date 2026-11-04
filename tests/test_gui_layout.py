"""Desktop layout contracts from the GUI review: menus, toolbar, panels, canvas."""
from __future__ import annotations

from tests.functional_units import functional_units

from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


# Wide enough for every header control in full, even with offscreen Qt's wide box glyphs.
WIDE_WINDOW = 2400


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

    def check_test_overlay_sections_follow_workflow_categories(self) -> None:
        from seedvision.pipeline import NODE_CATEGORIES, build_default_pipeline

        graph = build_default_pipeline()
        unknown = {node_id: node.category for node_id, node in graph.nodes.items()
                   if node.category not in NODE_CATEGORIES}
        unused = {node_id: node.category for node_id, node in graph.unused_nodes.items()
                  if node.category not in NODE_CATEGORIES}
        self.assertEqual({**unknown, **unused}, {})
        self.window._rebuild_overlay_menu("raw_image")
        sections = [action.text() for action in self.window.overlay_menu.actions() if action.isSeparator()]
        self.assertEqual(sections, sorted(sections, key=NODE_CATEGORIES.index))
        self.assertEqual(sections[0], "Inputs")

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

    def check_test_image_list_marks_a_run_without_a_count(self) -> None:
        from types import SimpleNamespace
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QListWidgetItem
        from seedvision.ui.main_window import _path_identity

        source = Path(self._directory.name) / "images" / "plate.png"
        item = QListWidgetItem(source.name)
        item.setData(Qt.ItemDataRole.UserRole, str(source))
        self.window.image_list.addItem(item)
        key = _path_identity(source)
        analysis = SimpleNamespace(
            procedural_instances=object(), layers=SimpleNamespace(foreground_colour_profile=None)
        )
        self.window._analyses[key] = analysis
        self.addCleanup(self.window._analyses.pop, key, None)
        self.window._update_analysis_availability()
        self.assertTrue(item.data(Qt.ItemDataRole.AccessibleDescriptionRole).startswith(
            "Computed; no count: Foreground evidence is missing"))
        analysis.layers.foreground_colour_profile = object()
        self.window._update_analysis_availability()
        self.assertEqual(item.data(Qt.ItemDataRole.AccessibleDescriptionRole), "Computed; review available")

    def check_test_run_stop_and_inspector_share_one_work_state(self) -> None:
        window = self.window
        window._pipeline_node_selected("procedural_instances")
        status = window.pipeline_inspector.optimization_status
        self.assertFalse(window.stop_action.isEnabled())
        window._procedural_fit_task = object()
        try:
            window._update_analysis_availability()
            self.assertTrue(window.stop_action.isEnabled())
            self.assertFalse(window.analyze_action.isEnabled())
            window.pipeline_workflow.sync_buttons()
            self.assertFalse(window.analyze_action.isEnabled())
            self.assertEqual(status.text(), "Wait for the current calculation to finish.")
            window._optimization_task = object()
            window._update_analysis_availability()
            self.assertEqual(status.text(), "Optimisation is running.")
        finally:
            window._procedural_fit_task = window._optimization_task = None
        window._update_analysis_availability()
        self.assertFalse(window.stop_action.isEnabled())
        self.assertNotIn("running", status.text())
        # A stopped analysis says so, instead of reading as a superseded one.
        from pathlib import Path
        from unittest.mock import patch

        from seedvision.pipeline import NodeStatus

        path = Path("capture.jpg")
        window.pipeline.set_status("layout_detection", NodeStatus.RUNNING, "Running")
        window._stop_current_work()
        with patch.object(window.image_view, "_image_path", path):
            window._finish_analysis_cancellation(path, window.pipeline.revision)
        self.assertEqual(window.statusBar().currentMessage(), "Stopped analysis of capture.jpg.")
        node = window.pipeline.node("layout_detection")
        self.assertEqual((node.status, node.status_detail), (NodeStatus.IDLE, "Stopped"))

    def check_test_disabled_nodes_say_why_they_are_bypassed(self) -> None:
        from seedvision.pipeline import build_default_pipeline

        graph = build_default_pipeline()
        self.assertEqual(
            graph.disabled_status_detail("cellpose_sam_instances"), "Disabled; enable to run Cellpose-SAM"
        )
        source = graph.node("frequency_noise_masks")
        graph.set_enabled("frequency_noise_masks", False)
        self.assertEqual(source.status_detail, "Disabled")
        self.assertEqual(
            graph.node("reference_texture_prototypes").status_detail, f"Disabled; requires {source.title}"
        )

    def check_test_foreground_readiness_button_opens_foreground_painting(self) -> None:
        from unittest.mock import patch

        button = self.window.reference_readiness_button
        foreground = self.window.foreground_point_button
        self.assertEqual(button.text(), "Paint Foreground references…")
        self.assertFalse(foreground.isEnabled())
        button.click()
        self.assertEqual(self.window._selected_pipeline_node, "background_likelihood")
        self.assertFalse(foreground.isChecked())
        foreground.setEnabled(True)
        self.addCleanup(foreground.setEnabled, False)
        with patch.object(foreground, "setChecked") as check:
            button.click()
        check.assert_called_once_with(True)

    def _settled(self, width: int) -> None:
        self.window.resize(width, 700)
        for _ in range(6):
            self.application.processEvents()

    def _pane_edges(self, widget) -> tuple[int, int]:
        left = widget.mapTo(self.window.image_display_stack, widget.rect().topLeft()).x()
        return left, left + widget.width()

    def check_test_overlay_controls_sit_at_the_left_of_the_image_header(self) -> None:
        from PySide6.QtGui import QColor
        from PySide6.QtWidgets import QLabel
        from seedvision.ui.canvas_controls import CANVAS_CONTROL_BACKGROUND

        window, view = self.window, self.window.image_view
        window.show()
        self._settled(WIDE_WINDOW)
        controls = window.overlay_header_controls
        self.assertIs(controls.parentWidget(), view.zoom_controls)
        self.assertIs(view.header_layout.itemAt(0).widget(), controls)
        for widget in (window.overlay_button, window.overlay_opacity_slider, window.overlay_opacity_label):
            self.assertTrue(widget.isVisible())
            self.assertTrue(controls.isAncestorOf(widget))
            self.assertFalse(window.workflow_toolbar.isAncestorOf(widget))
        labels = {label.text() for label in view.zoom_controls.findChildren(QLabel) if label.isVisible()}
        self.assertNotIn("Image zoom", labels)
        self.assertLessEqual({"Overlay:", "Overlay opacity:", "68%"}, labels)
        # Controls, then file name, then zoom, all on the header's one row.
        order = (controls, view.current_file_label, view.zoom_out_button, view.actual_size_button)
        edges = [self._pane_edges(widget) for widget in order]
        self.assertEqual(edges, sorted(edges))
        self.assertLessEqual(view.zoom_controls.rect().height(), view.viewportMargins().top())
        self.assertTrue(view.zoom_controls.rect().contains(controls.geometry()))
        # Light text on the dark header, as a platform style would otherwise not give it.
        background = QColor(CANVAS_CONTROL_BACKGROUND).lightness()
        for label in (controls.selector_heading, controls.opacity_heading, window.overlay_opacity_label):
            self.assertGreater(label.palette().windowText().color().lightness(), background + 100)
        # Behaviour did not move with it: the slider blends and the button still opens the menu.
        self.assertIs(window.overlay_button.menu(), window.overlay_menu)
        window.overlay_opacity_slider.setValue(23)
        self.assertEqual(window.overlay_opacity_label.text(), "23%")
        self.assertAlmostEqual(view._overlay_opacity, 0.23)

    def check_test_toolbar_closes_up_after_the_overlay_controls_left(self) -> None:
        window = self.window
        window.show()
        self._settled(WIDE_WINDOW)

        def visible_separators_touch() -> bool:
            visible = [action for action in window.workflow_toolbar.actions() if action.isVisible()]
            return any(left.isSeparator() and right.isSeparator() for left, right in zip(visible, visible[1:]))

        actions = window.workflow_toolbar.actions()
        self.assertFalse(visible_separators_touch())
        self.assertFalse(window.hsv_separator_action.isVisible())
        self.assertLess(actions.index(window.image_workspace_action), actions.index(window.paint_background_action))
        window._sync_overlay_display_controls("background_colour_gamut")
        self.assertTrue(window.hsv_value_slider.isVisible())
        self.assertTrue(window.hsv_separator_action.isVisible())
        self.assertFalse(visible_separators_touch())
        # A full-pane overlay cannot be blended, so opacity leaves the header with it.
        self.assertTrue(window.overlay_button.isVisible())
        self.assertFalse(window.overlay_opacity_slider.isVisible())
        self.assertFalse(window.overlay_header_controls.opacity_heading.isVisible())
        window._sync_overlay_display_controls("raw_image")
        self.application.processEvents()
        self.assertFalse(window.hsv_value_slider.isVisible())
        self.assertFalse(window.hsv_separator_action.isVisible())
        self.assertFalse(visible_separators_touch())
        self.assertTrue(window.overlay_opacity_slider.isVisible())
        self.assertTrue(window.overlay_header_controls.opacity_heading.isVisible())
        # The annotation controls stayed and still work.
        self.assertTrue(window.workflow_toolbar.isAncestorOf(window.annotation_opacity_slider))
        window.annotation_opacity_slider.setValue(31)
        self.assertEqual(window.annotation_opacity_label.text(), "31%")
        self.assertAlmostEqual(window.image_view._annotation_opacity, 0.31)

    def check_test_image_header_never_overlaps_at_narrow_widths(self) -> None:
        from PySide6.QtCore import Qt

        window, view = self.window, self.window.image_view
        controls = window.overlay_header_controls
        file_name = "Scanner_export_batch_seven_soybean_plate_2026-11-03_IMG_0002c.JPG"
        title = "Cellpose-SAM seed instances"
        window.show()
        self._settled(WIDE_WINDOW)
        controls.set_title(title, "Choose an overlay.")
        self.assertEqual(window.overlay_button.accessibleName(), title)
        self.assertIn(title, window.overlay_button.toolTip())
        for name in ("IMG_0002c.JPG", file_name):
            view.current_file_label.setText(name)
            view._fit_header_controls()
            for width in (WIDE_WINDOW, 1540, 1280, 1100, 1000, window.minimumWidth()):
                self._settled(width)
                row = [widget for widget in (
                    controls.selector_heading, window.overlay_button, controls.opacity_heading,
                    controls.short_opacity_heading, window.overlay_opacity_slider, window.overlay_opacity_label,
                    view.current_file_label, view.zoom_out_button, view.zoom_label, view.zoom_in_button,
                    view.fit_button, view.actual_size_button,
                ) if widget.isVisible()]
                with self.subTest(file=name, width=width, shown=[w.objectName() or type(w).__name__ for w in row]):
                    # The selector, the file name and every zoom button always stay.
                    for widget in (window.overlay_button, view.current_file_label, view.zoom_out_button,
                                   view.zoom_label, view.zoom_in_button, view.fit_button, view.actual_size_button):
                        self.assertIn(widget, row)
                        self.assertGreater(widget.width(), 0)
                    edges = [self._pane_edges(widget) for widget in row]
                    for (_, right), (left, _) in zip(edges, edges[1:]):
                        self.assertLessEqual(right, left)
                    self.assertLessEqual(edges[-1][1], window.image_display_stack.width())
                    self.assertEqual(view.current_file_label.text(), name)
                    self.assertEqual(window.overlay_button.accessibleName(), title)
        # A wide pane shows a short overlay name and the file name in full.
        view.current_file_label.setText("IMG_0002c.JPG")
        view._fit_header_controls()
        controls.set_title("Raw image", "Choose an overlay.")
        self._settled(WIDE_WINDOW)
        self.assertEqual(window.overlay_button.text(), "Raw image")
        self.assertTrue(controls.selector_heading.isVisible())
        self.assertTrue(controls.opacity_heading.isVisible())
        self.assertGreaterEqual(view.current_file_label.width(), view.current_file_label.sizeHint().width())
        # A vertical scroll bar takes its width from the header, not from the zoom buttons.
        view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self._settled(window.minimumWidth())
        self.assertLessEqual(self._pane_edges(view.actual_size_button)[1], view.viewport().width())
        self.assertLessEqual(self._pane_edges(view.zoom_controls)[1], view.viewport().width())
        self.assertLessEqual(self._pane_edges(window.overlay_button)[1], self._pane_edges(view.current_file_label)[0])
        # The header's minimum fits the panes' minimums inside the smallest window.
        side_panels = sum(window.main_splitter.widget(index).minimumWidth() for index in (0, 2))
        self.assertLessEqual(side_panels + window.image_display_stack.minimumSizeHint().width(), window.minimumWidth())

    def check_test_overlay_controls_follow_the_showing_display(self) -> None:
        window, view = self.window, self.window.image_view
        preview = window.pipeline_workflow.preview
        controls = window.overlay_header_controls
        window.show()
        self._settled(WIDE_WINDOW)
        window.overlay_opacity_slider.setValue(41)
        window.overlay_opacity_slider.setEnabled(True)
        window.image_display_stack.setCurrentWidget(preview)
        self.application.processEvents()
        self.assertTrue(preview.isAncestorOf(controls))
        self.assertFalse(view.isAncestorOf(controls))
        self.assertIs(preview.header_layout.itemAt(0).widget(), controls)
        self.assertTrue(window.overlay_button.isVisible())
        self.assertTrue(window.overlay_opacity_slider.isVisible())
        self.assertEqual(window.overlay_opacity_slider.value(), 41)
        self.assertTrue(window.overlay_opacity_slider.isEnabled())
        self.assertLessEqual(self._pane_edges(controls)[0], 8)
        # A long message wraps below the header instead of narrowing these controls.
        preview.message.setText("Run this independent node on the selected project image to inspect outputs. " * 3)
        self._settled(1000)
        self.assertEqual(preview.header_layout.parentWidget().height(), 38)
        self.assertGreater(window.overlay_button.width(), 0)
        self._settled(WIDE_WINDOW)
        window.image_display_stack.setCurrentWidget(view)
        self.application.processEvents()
        self.assertIs(controls.parentWidget(), view.zoom_controls)
        self.assertIs(view.header_layout.itemAt(0).widget(), controls)
        self.assertIsNone(preview.header_layout.itemAt(0).widget())
        self.assertEqual(window.overlay_opacity_slider.value(), 41)
        self.assertTrue(window.overlay_button.isVisible())
        # They go with the image pane; the pipeline can be shown alone.
        window.image_workspace_action.setChecked(False)
        self.application.processEvents()
        self.assertFalse(window.overlay_button.isVisible())
        window.image_workspace_action.setChecked(True)
        self.application.processEvents()
        self.assertTrue(window.overlay_button.isVisible())

    def check_test_window_fixtures_share_templates_and_opting_out_builds_them(self) -> None:
        from unittest.mock import patch

        from seedvision.pipeline import workflows
        from tests.review_fixtures import (_workflow_template_key, _workflow_templates, dispose_window,
                                           isolated_window, shared_workflow_templates, workspace_window)

        root = Path(self._directory.name)
        shared, identities = _workflow_templates(_workflow_template_key(), workflows.workflow_catalogue)
        self.assertIs(self.window.pipeline_workflow._all_templates, shared)
        window = workspace_window(root)
        self.addCleanup(dispose_window, window)
        self.assertIs(window.pipeline_workflow._all_templates, shared)
        # Every other window fixture shares; this one keeps the real startup build covered.
        real = isolated_window(root, reuse_templates=False, images=False,
                               settings_path=root / "real-startup-settings.ini")
        self.addCleanup(dispose_window, real)
        self.assertIsNot(real.pipeline_workflow._all_templates, shared)
        self.assertEqual(tuple(recipe.to_json() for recipe in real.pipeline_workflow._all_templates), identities)
        with patch("tests.review_fixtures._workflow_templates", return_value=(shared, ("changed",))):
            with self.assertRaisesRegex(AssertionError, "changed before construction"):
                with shared_workflow_templates():
                    pass
        with patch.object(workflows, "workflow_catalogue") as mocked, shared_workflow_templates():
            self.assertIs(workflows.workflow_catalogue, mocked)

    def test_workflow_lifecycle(self):
        """19 related GuiLayoutTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_every_shortcut_is_unique)
        self.scenario(self.check_test_toolbar_keeps_run_view_and_annotation_controls_only)
        self.scenario(self.check_test_menus_group_commands_and_end_file_with_exit)
        self.scenario(self.check_test_shortcut_reference_lists_window_commands)
        self.scenario(self.check_test_overlay_filter_hides_non_matching_entries)
        self.scenario(self.check_test_overlay_sections_follow_workflow_categories)
        self.scenario(self.check_test_reference_summary_is_compact_and_details_are_on_demand)
        self.scenario(self.check_test_spin_decimals_never_round_a_value)
        self.scenario(self.check_test_node_cards_fold_viewer_outputs_behind_an_expander)
        self.scenario(self.check_test_right_panel_pins_results_above_node_and_image_tabs)
        self.scenario(self.check_test_image_list_marks_a_run_without_a_count)
        self.scenario(self.check_test_run_stop_and_inspector_share_one_work_state)
        self.scenario(self.check_test_disabled_nodes_say_why_they_are_bypassed)
        self.scenario(self.check_test_foreground_readiness_button_opens_foreground_painting)
        self.scenario(self.check_test_overlay_controls_sit_at_the_left_of_the_image_header)
        self.scenario(self.check_test_toolbar_closes_up_after_the_overlay_controls_left)
        self.scenario(self.check_test_image_header_never_overlaps_at_narrow_widths)
        self.scenario(self.check_test_overlay_controls_follow_the_showing_display)
        self.scenario(self.check_test_window_fixtures_share_templates_and_opting_out_builds_them)


if __name__ == "__main__":
    unittest.main()
