from __future__ import annotations

from copy import deepcopy
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np


class ProjectAnalysisUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        try:
            from PySide6.QtWidgets import QApplication
        except ImportError as error:
            raise unittest.SkipTest(f"PySide6 unavailable: {error}")
        cls.application = QApplication.instance() or QApplication([])

    @staticmethod
    def _write_image(
        path: Path,
        *,
        size: tuple[int, int] = (48, 36),
        colour: str = "#6f624a",
    ) -> Path:
        from PySide6.QtGui import QColor, QImage

        path.parent.mkdir(parents=True, exist_ok=True)
        image = QImage(size[0], size[1], QImage.Format.Format_RGB32)
        image.fill(QColor(colour))
        if not image.save(str(path)):
            raise RuntimeError(f"Could not write test image {path}.")
        return path.resolve()

    @staticmethod
    def _window(root: Path):
        """Construct a window whose recent-project settings are test-local."""

        from PySide6.QtCore import QSettings

        from seedvision.ui.main_window import MainWindow

        settings = QSettings(
            str(root / "test-application-settings.ini"),
            QSettings.Format.IniFormat,
        )
        settings.clear()
        with patch("seedvision.ui.main_window.QSettings", return_value=settings):
            window = MainWindow(root)
        window._test_application_settings = settings
        return window

    @classmethod
    def _dispose(cls, window) -> None:
        """Close without allowing a test's intentionally dirty state to prompt."""

        window._project_tracking_enabled = False
        window._project_dirty = False
        window._reference_masks_dirty.clear()
        window._instance_annotations_dirty.clear()
        window._active_tasks.clear()
        window._learning_training_task = None
        window._procedural_fit_task = None
        window.close()
        cls.application.processEvents()

    @staticmethod
    def _image_paths(window) -> tuple[Path, ...]:
        from PySide6.QtCore import Qt

        return tuple(
            Path(window.image_list.item(index).data(Qt.ItemDataRole.UserRole)).resolve()
            for index in range(window.image_list.count())
        )

    def test_actions_and_profile_load_replace_only_portable_analysis_state(self) -> None:
        from PySide6.QtGui import QKeySequence
        from PySide6.QtWidgets import QMessageBox

        from seedvision.persistence import save_analysis_settings_profile
        from seedvision.pipeline import build_default_pipeline

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = self._write_image(root / "images" / "capture.png")
            window = self._window(root)
            try:
                self.assertEqual(
                    window.open_action.shortcut().toString(),
                    QKeySequence(QKeySequence.StandardKey.Open).toString(),
                )
                self.assertEqual(
                    window.save_project_action.shortcut().toString(),
                    QKeySequence(QKeySequence.StandardKey.Save).toString(),
                )
                self.assertEqual(
                    window.save_project_as_action.shortcut().toString(),
                    QKeySequence(QKeySequence.StandardKey.SaveAs).toString(),
                )
                self.assertEqual(
                    window.open_project_action.shortcut().toString(), "Ctrl+Shift+O"
                )
                self.assertTrue(window.save_reference_regions_action.shortcut().isEmpty())

                source = build_default_pipeline()
                source.set_parameter(
                    "ruler_detection", "ruler_length_mm", 181.0
                )
                source.restore_unused_node("circle_candidates")
                profile_path = save_analysis_settings_profile(
                    root / "laboratory.seedfiddle-settings.json", source
                )

                key = window._current_image_key()
                self.assertIsNotNone(key)
                assert key is not None
                foreground = np.zeros((36, 48), dtype=bool)
                foreground[5:12, 7:16] = True
                foreground.flags.writeable = False
                window._applied_foreground_reference_masks[key] = foreground
                manual_sentinel = object()
                window._manual_seed_centre_states["unrelated"] = manual_sentinel
                window.pipeline.node("ruler_detection").x = 812.5
                window.pipeline.node("ruler_detection").y = -74.25
                window.pipeline_canvas.node_items["ruler_detection"].setPos(
                    812.5, -74.25
                )
                window.pipeline_canvas.set_cable_bundling_enabled(True)
                window.pipeline_canvas.set_obstacle_routing_enabled(True)
                before_images = self._image_paths(window)
                before_image = window.image_view.image_path
                before_species = window.species_combo.currentText()

                with (
                    patch.object(
                        QMessageBox,
                        "question",
                        return_value=QMessageBox.StandardButton.Yes,
                    ) as question,
                    patch.object(window, "_analyze_current_image") as analyze,
                ):
                    loaded = window._load_analysis_settings_profile(profile_path)

                self.assertTrue(loaded)
                question.assert_called_once()
                analyze.assert_not_called()
                self.assertEqual(
                    window.pipeline.node("ruler_detection").parameters[
                        "ruler_length_mm"
                    ],
                    181.0,
                )
                self.assertIn("circle_candidates", window.pipeline.nodes)
                self.assertEqual(window.pipeline.node("ruler_detection").x, 812.5)
                self.assertEqual(window.pipeline.node("ruler_detection").y, -74.25)
                self.assertTrue(window.pipeline_canvas.cable_bundling_enabled)
                self.assertTrue(window.pipeline_canvas.obstacle_routing_enabled)
                self.assertEqual(self._image_paths(window), before_images)
                self.assertEqual(window.image_view.image_path, before_image)
                self.assertEqual(window.image_view.image_path, image_path)
                self.assertIs(
                    window._applied_foreground_reference_masks[key], foreground
                )
                self.assertIs(
                    window._manual_seed_centre_states["unrelated"], manual_sentinel
                )
                self.assertEqual(window.species_combo.currentText(), before_species)
            finally:
                self._dispose(window)

    def test_incompatible_profile_is_rejected_before_confirmation_and_is_atomic(self) -> None:
        from PySide6.QtWidgets import QMessageBox

        from seedvision.persistence import (
            analysis_settings_profile_from_graph,
            analysis_settings_profile_to_payload,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_image(root / "images" / "capture.png")
            window = self._window(root)
            try:
                before = analysis_settings_profile_to_payload(
                    analysis_settings_profile_from_graph(window.pipeline)
                )
                payload = deepcopy(before)
                ruler = next(
                    node
                    for node in payload["nodes"]
                    if node["id"] == "ruler_detection"
                )
                ruler["parameters"]["ruler_length_mm"] = -1.0
                source = root / "invalid.seedfiddle-settings.json"
                source.write_text(json.dumps(payload), encoding="utf-8")
                image_before = window.image_view.image_path

                with (
                    patch.object(QMessageBox, "question") as question,
                    patch.object(QMessageBox, "critical") as critical,
                ):
                    loaded = window._load_analysis_settings_profile(source)

                self.assertFalse(loaded)
                question.assert_not_called()
                critical.assert_called_once()
                self.assertEqual(
                    analysis_settings_profile_to_payload(
                        analysis_settings_profile_from_graph(window.pipeline)
                    ),
                    before,
                )
                self.assertEqual(window.image_view.image_path, image_before)
            finally:
                self._dispose(window)

    def test_project_round_trip_restores_state_and_purges_every_old_image_value(self) -> None:
        from PySide6.QtCore import QSignalBlocker

        from seedvision.persistence import (
            ManualSeedCentreStore,
            ReferenceRegionBundle,
            ReferenceRegionStore,
        )
        from seedvision.segmentation import PipelineAnalysisCache

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = self._write_image(
                root / "images" / "a.png", colour="#6b5941"
            )
            second = self._write_image(
                root / "images" / "b.png", colour="#7c694e"
            )
            stale = self._write_image(
                root / "outside" / "stale.png", colour="#4a687c"
            )
            window = self._window(root)
            try:
                foreground = np.zeros((36, 48), dtype=bool)
                foreground[4:16, 6:20] = True
                instances = np.zeros((36, 48), dtype=np.uint16)
                instances[18:29, 23:37] = 7
                ReferenceRegionStore(root).save(
                    second,
                    ReferenceRegionBundle(
                        shape=(36, 48),
                        foreground=foreground,
                        annotated_seeds=instances,
                        annotation_origin="manual",
                    ),
                )
                ManualSeedCentreStore(root).save(
                    second,
                    np.asarray(((11.5, 13.25),), dtype=np.float64),
                    mode="replace_automatic",
                    source_shape=(36, 48),
                )
                window._open_path(second)
                window.pipeline.set_parameter(
                    "ruler_detection", "ruler_length_mm", 187.0
                )
                window.pipeline.node("ruler_detection").x = 733.0
                window.pipeline.node("ruler_detection").y = -93.0
                window.pipeline_canvas.node_items["ruler_detection"].setPos(
                    733.0, -93.0
                )
                window.pipeline_canvas.set_cable_bundling_enabled(True)
                window.pipeline_canvas.set_obstacle_routing_enabled(True)
                window.pipeline_canvas.select_node("reference_edge_probability")
                window.species_combo.addItem("Project test species")
                window.species_combo.setCurrentText("Project test species")
                overlay_index = window.overlay_combo.findData(
                    "physical_edge_probability"
                )
                self.assertGreaterEqual(overlay_index, 0)
                window.overlay_combo.setCurrentIndex(overlay_index)
                destination = root / "projects" / "roundtrip.seedfiddle-project.json"
                self.assertTrue(window._write_project(destination))

                stale_key = str(stale.resolve()).casefold()
                stale_mask = np.ones((3, 4), dtype=bool)
                for values in (
                    window._draft_background_reference_masks,
                    window._draft_foreground_reference_masks,
                    window._applied_background_reference_masks,
                    window._applied_foreground_reference_masks,
                    window._draft_background_exclusion_masks,
                    window._draft_foreground_exclusion_masks,
                    window._applied_background_exclusion_masks,
                    window._applied_foreground_exclusion_masks,
                    window._draft_instance_annotations,
                    window._applied_instance_annotations,
                ):
                    values[stale_key] = stale_mask
                window._draft_instance_annotation_origins[stale_key] = "stale"
                window._applied_instance_annotation_origins[stale_key] = "stale"
                window._reference_dirty_classes[stale_key] = {"foreground"}
                window._reference_undo_histories[stale_key] = object()
                window._instance_undo_histories[stale_key] = object()
                window._manual_seed_centre_states[stale_key] = object()
                window._manual_seed_centre_histories[stale_key] = [object()]
                window._instance_continuity_cache[stale_key] = object()
                window._reference_masks_dirty.add(stale_key)
                window._instance_annotations_dirty.add(stale_key)
                window._reference_region_autoload_attempted.add(stale_key)
                window._manual_seed_centre_autoload_attempted.add(stale_key)
                window._analyses[stale_key] = SimpleNamespace()
                window._analysis_caches[stale_key] = PipelineAnalysisCache()
                window._cache_dirty_nodes[stale_key] = {"edge_gradients"}
                window._add_image(stale, open_now=False)
                with QSignalBlocker(window.paint_background_action):
                    window.paint_background_action.setChecked(True)
                with QSignalBlocker(window.annotate_instances_action):
                    window.annotate_instances_action.setChecked(True)
                window.reference_panel.show()
                window.image_view._reference_point_mode = "foreground"
                window.image_view._manual_seed_centre_editing = True
                window.pipeline.set_parameter(
                    "ruler_detection", "ruler_length_mm", 199.0
                )
                window.pipeline.node("ruler_detection").x = -400.0
                window.pipeline.node("ruler_detection").y = 500.0
                window.pipeline_canvas.set_cable_bundling_enabled(False)
                window.pipeline_canvas.set_obstacle_routing_enabled(False)
                window.species_combo.setCurrentIndex(0)

                with patch.object(
                    window, "_confirm_project_replacement", return_value=True
                ):
                    opened = window._open_project(destination)

                self.assertTrue(opened)
                self.assertEqual(self._image_paths(window), (first, second))
                self.assertEqual(window.image_view.image_path, second)
                self.assertEqual(
                    window.pipeline.node("ruler_detection").parameters[
                        "ruler_length_mm"
                    ],
                    187.0,
                )
                self.assertEqual(window.pipeline.node("ruler_detection").x, 733.0)
                self.assertEqual(window.pipeline.node("ruler_detection").y, -93.0)
                self.assertTrue(window.pipeline_canvas.cable_bundling_enabled)
                self.assertTrue(window.pipeline_canvas.obstacle_routing_enabled)
                self.assertEqual(
                    window._selected_pipeline_node, "reference_texture_prototypes"
                )
                self.assertEqual(
                    window.overlay_combo.currentData(), "physical_edge_probability"
                )
                self.assertEqual(
                    window.species_combo.currentText(), "Project test species"
                )
                second_key = str(second.resolve()).casefold()
                pending = window._pending_unbound_reference_bundles[second_key]
                np.testing.assert_array_equal(pending.foreground, foreground)
                np.testing.assert_array_equal(pending.annotated_seeds, instances)
                np.testing.assert_allclose(
                    window._manual_seed_centre_states[
                        second_key
                    ].centres_source_xy,
                    ((11.5, 13.25),),
                )
                for values in (
                    window._draft_background_reference_masks,
                    window._draft_foreground_reference_masks,
                    window._applied_background_reference_masks,
                    window._applied_foreground_reference_masks,
                    window._draft_background_exclusion_masks,
                    window._draft_foreground_exclusion_masks,
                    window._applied_background_exclusion_masks,
                    window._applied_foreground_exclusion_masks,
                    window._draft_instance_annotations,
                    window._applied_instance_annotations,
                    window._draft_instance_annotation_origins,
                    window._applied_instance_annotation_origins,
                    window._reference_dirty_classes,
                    window._reference_undo_histories,
                    window._instance_undo_histories,
                    window._manual_seed_centre_states,
                    window._manual_seed_centre_histories,
                    window._instance_continuity_cache,
                    window._analyses,
                    window._analysis_caches,
                    window._cache_dirty_nodes,
                ):
                    self.assertNotIn(stale_key, values)
                self.assertNotIn(stale_key, window._reference_masks_dirty)
                self.assertNotIn(stale_key, window._instance_annotations_dirty)
                self.assertNotIn(
                    stale_key, window._reference_region_autoload_attempted
                )
                self.assertNotIn(
                    stale_key, window._manual_seed_centre_autoload_attempted
                )
                self.assertFalse(window.paint_background_action.isChecked())
                self.assertFalse(window.annotate_instances_action.isChecked())
                self.assertFalse(window.reference_panel.isVisible())
                self.assertIsNone(window.image_view._reference_point_mode)
                self.assertFalse(window.image_view.manual_seed_centre_editing)
                self.assertEqual(window._current_project_path, destination.resolve())
                self.assertFalse(window._project_dirty)
                self.assertEqual(window._recent_project_paths[0], destination.resolve())
            finally:
                self._dispose(window)

    def test_absent_and_partial_ui_state_reset_unspecified_presentation_to_defaults(self) -> None:
        from PySide6.QtWidgets import QMessageBox

        from seedvision.persistence import (
            ProjectAnalysisStore,
            ProjectUiState,
            analysis_settings_profile_from_graph,
        )
        from seedvision.pipeline import build_default_pipeline

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            window = self._window(root)
            try:
                store = ProjectAnalysisStore(root)
                default_graph = build_default_pipeline()
                profile = analysis_settings_profile_from_graph(default_graph)
                absent_path = store.save(
                    store.capture(
                        analysis_settings=profile,
                        images=(),
                        ui_state=None,
                    ),
                    root / "projects" / "absent.seedfiddle-project.json",
                )
                window.pipeline.node("ruler_detection").x = -900.0
                window.pipeline.node("ruler_detection").y = 910.0
                window.pipeline_canvas.set_cable_bundling_enabled(True)
                window.pipeline_canvas.set_obstacle_routing_enabled(True)
                with patch.object(
                    window, "_confirm_project_replacement", return_value=True
                ):
                    self.assertTrue(window._open_project(absent_path))
                self.assertEqual(
                    window.pipeline.node("ruler_detection").x,
                    default_graph.node("ruler_detection").x,
                )
                self.assertEqual(
                    window.pipeline.node("ruler_detection").y,
                    default_graph.node("ruler_detection").y,
                )
                self.assertFalse(window.pipeline_canvas.cable_bundling_enabled)
                self.assertFalse(window.pipeline_canvas.obstacle_routing_enabled)
                self.assertFalse(window._project_dirty)

                partial_path = store.save(
                    store.capture(
                        analysis_settings=profile,
                        images=(),
                        ui_state=ProjectUiState(
                            node_positions={"ruler_detection": (123.5, -44.0)},
                            bundle_cables=True,
                            route_around_nodes=False,
                            selected_node=None,
                            selected_overlay=None,
                        ),
                    ),
                    root / "projects" / "partial.seedfiddle-project.json",
                )
                window.pipeline.node("edge_gradients").x = -777.0
                window.pipeline.node("edge_gradients").y = 888.0
                with (
                    patch.object(
                        window, "_confirm_project_replacement", return_value=True
                    ),
                    patch.object(QMessageBox, "warning"),
                ):
                    self.assertTrue(window._open_project(partial_path))
                self.assertEqual(
                    (
                        window.pipeline.node("ruler_detection").x,
                        window.pipeline.node("ruler_detection").y,
                    ),
                    (123.5, -44.0),
                )
                self.assertEqual(
                    window.pipeline.node("edge_gradients").x,
                    default_graph.node("edge_gradients").x,
                )
                self.assertEqual(
                    window.pipeline.node("edge_gradients").y,
                    default_graph.node("edge_gradients").y,
                )
                self.assertTrue(window.pipeline_canvas.cable_bundling_enabled)
                self.assertFalse(window.pipeline_canvas.obstacle_routing_enabled)
                self.assertFalse(window._project_dirty)
            finally:
                self._dispose(window)

    def test_tracked_untitled_close_prompts_and_save_routes_through_save_as(self) -> None:
        from PySide6.QtWidgets import QMessageBox

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            window = self._window(root)
            try:
                with patch.object(
                    window, "_confirm_project_replacement", return_value=True
                ):
                    self.assertTrue(window._new_project())
                self.assertTrue(window._project_tracking_enabled)
                self.assertIsNone(window._current_project_path)

                window._set_project_dirty()
                cancel_event = Mock()
                with patch.object(
                    QMessageBox,
                    "warning",
                    return_value=QMessageBox.StandardButton.Cancel,
                ) as warning:
                    window.closeEvent(cancel_event)
                cancel_event.ignore.assert_called_once()
                warning.assert_called_once()

                window._reference_masks_dirty.add("unapplied")
                draft_event = Mock()
                with patch.object(
                    QMessageBox,
                    "warning",
                    return_value=QMessageBox.StandardButton.Cancel,
                ):
                    window.closeEvent(draft_event)
                draft_event.ignore.assert_called_once()
                self.assertIn("unapplied", window._reference_masks_dirty)

                window._reference_masks_dirty.clear()
                save_event = Mock()
                with (
                    patch.object(
                        QMessageBox,
                        "warning",
                        return_value=QMessageBox.StandardButton.Save,
                    ),
                    patch.object(
                        window, "_save_project_as", return_value=False
                    ) as save_as,
                ):
                    window.closeEvent(save_event)
                save_as.assert_called_once_with(drafts_resolved=True)
                save_event.ignore.assert_called_once()
            finally:
                self._dispose(window)

    def test_new_project_purges_session_and_restores_default_graph_and_ui(self) -> None:
        from PySide6.QtCore import QSignalBlocker

        from seedvision.pipeline import build_default_pipeline
        from seedvision.ui.main_window import _ManualSeedCentreState
        from seedvision.ui.reference_history import RasterUndoHistory

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_image(root / "images" / "capture.png")
            window = self._window(root)
            try:
                self.assertTrue(
                    window._write_project(
                        root / "projects" / "old.seedfiddle-project.json"
                    )
                )
                key = window._current_image_key()
                self.assertIsNotNone(key)
                assert key is not None
                window.pipeline.set_parameter(
                    "ruler_detection", "ruler_length_mm", 191.0
                )
                window.pipeline.node("ruler_detection").x = -600.0
                window.pipeline.node("ruler_detection").y = 700.0
                window.pipeline_canvas.set_cable_bundling_enabled(True)
                window.pipeline_canvas.set_obstacle_routing_enabled(True)
                window._applied_foreground_reference_masks[key] = np.ones(
                    (36, 48), dtype=bool
                )
                window._manual_seed_centre_states[key] = _ManualSeedCentreState(
                    np.asarray(((8.0, 9.0),), dtype=np.float64), "augment"
                )
                window._reference_undo_histories[key] = RasterUndoHistory()
                with QSignalBlocker(window.paint_background_action):
                    window.paint_background_action.setChecked(True)
                with QSignalBlocker(window.annotate_instances_action):
                    window.annotate_instances_action.setChecked(True)
                window.reference_panel.show()

                with patch.object(
                    window, "_confirm_project_replacement", return_value=True
                ):
                    self.assertTrue(window._new_project())

                defaults = build_default_pipeline()
                self.assertEqual(window.image_list.count(), 0)
                self.assertIsNone(window.image_view.image_path)
                self.assertEqual(window._image_paths, {})
                self.assertEqual(window._applied_foreground_reference_masks, {})
                self.assertEqual(window._manual_seed_centre_states, {})
                self.assertEqual(window._reference_undo_histories, {})
                self.assertEqual(
                    window.pipeline.node("ruler_detection").parameters[
                        "ruler_length_mm"
                    ],
                    defaults.node("ruler_detection").parameters[
                        "ruler_length_mm"
                    ],
                )
                self.assertEqual(
                    window.pipeline.node("ruler_detection").x,
                    defaults.node("ruler_detection").x,
                )
                self.assertEqual(
                    window.pipeline.node("ruler_detection").y,
                    defaults.node("ruler_detection").y,
                )
                self.assertFalse(window.pipeline_canvas.cable_bundling_enabled)
                self.assertFalse(window.pipeline_canvas.obstacle_routing_enabled)
                self.assertFalse(window.paint_background_action.isChecked())
                self.assertFalse(window.annotate_instances_action.isChecked())
                self.assertTrue(window.reference_panel.isHidden())
                self.assertTrue(window._project_tracking_enabled)
                self.assertIsNone(window._current_project_path)
                self.assertFalse(window._project_dirty)
            finally:
                self._dispose(window)

    def test_ordinary_analysis_allows_project_save_but_guards_context_replacement(self) -> None:
        from PySide6.QtWidgets import QMessageBox

        from seedvision.persistence import (
            ProjectAnalysisStore,
            analysis_settings_profile_from_graph,
            save_analysis_settings_profile,
        )
        from seedvision.pipeline import build_default_pipeline

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            window = self._window(root)
            try:
                graph = build_default_pipeline()
                profile_path = save_analysis_settings_profile(
                    root / "busy.seedfiddle-settings.json", graph
                )
                project_path = ProjectAnalysisStore(root).save(
                    ProjectAnalysisStore(root).capture(
                        analysis_settings=analysis_settings_profile_from_graph(graph),
                        images=(),
                    ),
                    root / "projects" / "busy.seedfiddle-project.json",
                )
                before_revision = window.pipeline.revision
                window._project_tracking_enabled = True
                window._set_project_path(project_path)
                window._active_tasks["busy"] = object()
                window._update_analysis_availability()
                self.assertFalse(window.new_project_action.isEnabled())
                self.assertFalse(window.open_project_action.isEnabled())
                self.assertTrue(window.save_project_action.isEnabled())
                self.assertTrue(window.save_project_as_action.isEnabled())
                self.assertFalse(window.load_analysis_settings_action.isEnabled())
                self.assertFalse(window.save_analysis_settings_action.isEnabled())

                with patch.object(QMessageBox, "information") as information:
                    self.assertFalse(
                        window._load_analysis_settings_profile(profile_path)
                    )
                    self.assertFalse(window._open_project(project_path))
                    self.assertTrue(window._save_project())
                    self.assertFalse(window._new_project())
                self.assertEqual(information.call_count, 3)
                self.assertEqual(window.pipeline.revision, before_revision)
                self.assertEqual(window._current_project_path, project_path)
                self.assertEqual(window._recent_project_paths, [project_path])
            finally:
                window._active_tasks.clear()
                self._dispose(window)

    def test_ordinary_recompute_keeps_reference_and_annotation_entry_points_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_image(root / "images" / "capture.png")
            window = self._window(root)
            try:
                window.image_view._analysis_result = object()
                window._active_tasks["ordinary-analysis"] = object()
                window._sync_background_controls()
                window._update_analysis_availability()

                self.assertTrue(window.paint_background_action.isEnabled())
                self.assertTrue(window.annotate_instances_action.isEnabled())
                self.assertTrue(window.background_point_button.isEnabled())
                self.assertTrue(window.instance_smart_fill_button.isEnabled())
                self.assertTrue(window.save_project_action.isEnabled())
                self.assertTrue(window.save_project_as_action.isEnabled())
            finally:
                window._active_tasks.clear()
                self._dispose(window)

    def test_changed_source_cancel_is_nonmutating_and_failed_paths_are_not_recent(self) -> None:
        from PySide6.QtWidgets import QMessageBox

        from seedvision.persistence import (
            ProjectAnalysisStore,
            ProjectImageSpec,
            analysis_settings_profile_from_graph,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = self._write_image(root / "images" / "capture.png")
            window = self._window(root)
            try:
                store = ProjectAnalysisStore(root)
                project_path = store.capture_and_save(
                    analysis_settings=analysis_settings_profile_from_graph(
                        window.pipeline
                    ),
                    images=(ProjectImageSpec(image, (36, 48)),),
                    destination=(
                        root / "projects" / "changed.seedfiddle-project.json"
                    ),
                )
                self._write_image(
                    image, colour="#265f78"
                )
                window.pipeline.set_parameter(
                    "ruler_detection", "ruler_length_mm", 193.0
                )
                before_images = self._image_paths(window)
                before_recent = list(window._recent_project_paths)
                with (
                    patch.object(
                        QMessageBox,
                        "warning",
                        return_value=QMessageBox.StandardButton.Cancel,
                    ),
                    patch.object(window, "_confirm_project_replacement") as confirm,
                ):
                    opened = window._open_project(project_path)
                self.assertFalse(opened)
                confirm.assert_not_called()
                self.assertEqual(self._image_paths(window), before_images)
                self.assertEqual(
                    window.pipeline.node("ruler_detection").parameters[
                        "ruler_length_mm"
                    ],
                    193.0,
                )
                self.assertEqual(window._recent_project_paths, before_recent)

                missing = root / "does-not-exist.seedfiddle-project.json"
                with patch.object(QMessageBox, "critical"):
                    self.assertFalse(window._open_project(missing))
                self.assertNotIn(missing.resolve(), window._recent_project_paths)

                for index in range(10):
                    window._remember_recent_project(
                        root / f"recent-{index}.seedfiddle-project.json"
                    )
                self.assertEqual(len(window._recent_project_paths), 8)
                repeated = root / "recent-5.seedfiddle-project.json"
                window._remember_recent_project(repeated)
                self.assertEqual(window._recent_project_paths[0], repeated.resolve())
                self.assertEqual(len(window._recent_project_paths), 8)
            finally:
                self._dispose(window)

    def test_live_source_corruption_aborts_master_save_without_replacing_it(self) -> None:
        from PySide6.QtWidgets import QMessageBox

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = self._write_image(root / "images" / "capture.png")
            window = self._window(root)
            try:
                destination = root / "projects" / "source.seedfiddle-project.json"
                self.assertTrue(window._write_project(destination))
                original_master = destination.read_bytes()
                window._set_project_dirty()
                image.write_bytes(b"not a decodable image")

                with patch.object(QMessageBox, "critical") as critical:
                    self.assertFalse(window._save_project())

                critical.assert_called_once()
                self.assertTrue(window._project_dirty)
                self.assertEqual(destination.read_bytes(), original_master)
            finally:
                self._dispose(window)

    def test_unavailable_selected_image_survives_fallback_until_user_selects(self) -> None:
        from PySide6.QtWidgets import QMessageBox

        from seedvision.persistence import (
            ProjectAnalysisStore,
            ProjectImageSpec,
            analysis_settings_profile_from_graph,
        )
        from seedvision.pipeline import build_default_pipeline

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            missing = self._write_image(root / "images" / "intended.png")
            fallback = self._write_image(root / "images" / "fallback.png")
            store = ProjectAnalysisStore(root)
            project_path = store.capture_and_save(
                analysis_settings=analysis_settings_profile_from_graph(
                    build_default_pipeline()
                ),
                images=(
                    ProjectImageSpec(missing, (36, 48)),
                    ProjectImageSpec(fallback, (36, 48)),
                ),
                selected_image=missing,
                destination=root / "projects" / "selection.seedfiddle-project.json",
            )
            intended_id = store.load(project_path).document.selected_image_id
            missing.unlink()

            window = self._window(root)
            try:
                with (
                    patch.object(
                        QMessageBox,
                        "warning",
                        return_value=QMessageBox.StandardButton.Yes,
                    ),
                    patch.object(
                        window, "_confirm_project_replacement", return_value=True
                    ),
                ):
                    self.assertTrue(window._open_project(project_path))
                self.assertEqual(window.image_view.image_path, fallback)
                self.assertEqual(
                    window._project_unresolved_selected_image_id, intended_id
                )

                self.assertTrue(window._save_project())
                self.assertEqual(
                    store.load(project_path, verify_files=False).document.selected_image_id,
                    intended_id,
                )
                self.assertFalse(window._project_dirty)

                window._open_path(fallback)
                self.assertIsNone(window._project_unresolved_selected_image_id)
                self.assertTrue(window._project_dirty)
                self.assertTrue(window._save_project())
                saved = store.load(project_path, verify_files=False)
                fallback_id = next(
                    image.record.identifier
                    for image in saved.images
                    if image.path == fallback
                )
                self.assertEqual(
                    saved.document.selected_image_id,
                    fallback_id,
                )
            finally:
                self._dispose(window)

    def test_failed_applied_and_manual_sidecar_autosaves_are_retried_before_master_save(self) -> None:
        from PySide6.QtWidgets import QMessageBox

        from seedvision.persistence import (
            MANUAL_SEED_CENTRES_SIDECAR,
            REFERENCE_REGIONS_SIDECAR,
            ManualSeedCentreStoreError,
            ProjectAnalysisStore,
            ProjectFileStatus,
            ReferenceRegionError,
        )
        from seedvision.ui.main_window import _ManualSeedCentreState

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = self._write_image(root / "images" / "capture.png")
            window = self._window(root)
            try:
                destination = root / "projects" / "retry.seedfiddle-project.json"
                self.assertTrue(window._write_project(destination))
                original_master = destination.read_bytes()
                key = window._current_image_key()
                self.assertIsNotNone(key)
                assert key is not None

                foreground = np.zeros((36, 48), dtype=bool)
                foreground[6:18, 8:21] = True
                window._draft_foreground_reference_masks[key] = foreground
                window._reference_masks_dirty.add(key)
                window._reference_dirty_classes[key] = {"foreground"}
                with (
                    patch.object(
                        window._reference_region_store,
                        "save",
                        side_effect=ReferenceRegionError("reference disk failure"),
                    ),
                    patch.object(QMessageBox, "critical"),
                    patch.object(window, "_analyze_current_image"),
                ):
                    window._apply_reference_masks()

                with (
                    patch.object(
                        window._manual_seed_centre_store,
                        "save",
                        side_effect=ManualSeedCentreStoreError(
                            "centre disk failure"
                        ),
                    ),
                    patch.object(QMessageBox, "critical"),
                    patch.object(window, "_analyze_current_image"),
                ):
                    window._apply_manual_seed_centre_state(
                        key,
                        _ManualSeedCentreState(
                            np.asarray(((14.0, 12.5),), dtype=np.float64),
                            "augment",
                        ),
                        label="added test centre",
                        record_undo=True,
                    )

                self.assertIn(key, window._unsaved_reference_sidecars)
                self.assertIn(key, window._unsaved_manual_centre_sidecars)
                self.assertTrue(window._project_dirty)
                self.assertFalse(window._reference_region_store.path_for(image).exists())
                self.assertFalse(window._manual_seed_centre_store.path_for(image).exists())

                with (
                    patch.object(
                        window._reference_region_store,
                        "save",
                        side_effect=ReferenceRegionError("still unavailable"),
                    ),
                    patch.object(QMessageBox, "critical"),
                ):
                    self.assertFalse(window._save_project())
                self.assertEqual(destination.read_bytes(), original_master)
                self.assertIn(key, window._unsaved_reference_sidecars)
                self.assertIn(key, window._unsaved_manual_centre_sidecars)
                self.assertTrue(window._project_dirty)

                self.assertTrue(window._save_project())
                self.assertNotIn(key, window._unsaved_reference_sidecars)
                self.assertNotIn(key, window._unsaved_manual_centre_sidecars)
                self.assertTrue(window._reference_region_store.path_for(image).is_file())
                self.assertTrue(window._manual_seed_centre_store.path_for(image).is_file())
                loaded = ProjectAnalysisStore(root).load(destination)
                self.assertEqual(len(loaded.images), 1)
                reference = loaded.images[0].sidecar(REFERENCE_REGIONS_SIDECAR)
                centres = loaded.images[0].sidecar(MANUAL_SEED_CENTRES_SIDECAR)
                self.assertIsNotNone(reference)
                self.assertIsNotNone(centres)
                assert reference is not None and centres is not None
                self.assertEqual(reference.status, ProjectFileStatus.AVAILABLE)
                self.assertEqual(centres.status, ProjectFileStatus.AVAILABLE)
                self.assertFalse(window._project_dirty)
            finally:
                self._dispose(window)

    def test_bulk_draft_clear_persists_an_empty_corrected_shape_archive(self) -> None:
        from seedvision.persistence import ReferenceRegionStore

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = self._write_image(root / "images" / "capture.png")
            window = self._window(root)
            try:
                key = window._current_image_key()
                self.assertIsNotNone(key)
                assert key is not None
                corrected_shape = (29, 41)
                prior_foreground = np.zeros(corrected_shape, dtype=bool)
                prior_foreground[3:8, 5:11] = True
                prior_instances = np.zeros(corrected_shape, dtype=np.uint16)
                prior_instances[12:18, 19:27] = 4
                window._applied_foreground_reference_masks[key] = prior_foreground
                window._applied_instance_annotations[key] = prior_instances
                window._applied_instance_annotation_origins[key] = "manual"
                window._reference_layer_shapes[key] = corrected_shape
                window._draft_foreground_reference_masks[key] = np.zeros(
                    corrected_shape, dtype=bool
                )
                window._draft_instance_annotations[key] = np.zeros(
                    corrected_shape, dtype=np.uint16
                )
                window._reference_masks_dirty.add(key)
                window._instance_annotations_dirty.add(key)

                self.assertTrue(
                    window._apply_and_save_all_project_drafts(
                        {key}, recompute=False
                    )
                )

                stored = ReferenceRegionStore(root).load_if_present(
                    image, corrected_shape
                )
                self.assertIsNotNone(stored)
                assert stored is not None
                self.assertEqual(stored.shape, corrected_shape)
                self.assertIsNone(stored.background)
                self.assertIsNone(stored.foreground)
                self.assertIsNone(stored.other)
                self.assertIsNone(stored.annotated_seeds)
                self.assertEqual(window._reference_layer_shapes[key], corrected_shape)
            finally:
                self._dispose(window)

    def test_user_image_node_and_overlay_selection_dirty_tracked_project_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = self._write_image(root / "images" / "a.png")
            second = self._write_image(
                root / "images" / "b.png", colour="#435f72"
            )
            window = self._window(root)
            try:
                destination = root / "projects" / "selections.seedfiddle-project.json"
                self.assertTrue(window._write_project(destination))
                self.assertFalse(window._project_dirty)

                window._open_path(second)
                self.assertTrue(window._project_dirty)
                self.assertTrue(window._write_project(destination))
                self.assertFalse(window._project_dirty)

                window.pipeline_canvas.select_node("reference_edge_probability")
                self.assertTrue(window._project_dirty)
                self.assertTrue(window._write_project(destination))
                self.assertFalse(window._project_dirty)

                raw_index = window.overlay_combo.findData("raw_image")
                self.assertGreaterEqual(raw_index, 0)
                window.overlay_combo.setCurrentIndex(raw_index)
                self.assertTrue(window._project_dirty)
                self.assertEqual(window.image_view.image_path, second)
                self.assertEqual(self._image_paths(window), (first, second))

                with patch.object(
                    window, "_confirm_project_replacement", return_value=True
                ):
                    self.assertTrue(window._open_project(destination))
                self.assertFalse(window._project_dirty)
            finally:
                self._dispose(window)

    def test_project_sidecar_binding_omits_unlisted_and_loads_only_listed_files(self) -> None:
        from seedvision.persistence import (
            ManualSeedCentreStore,
            ProjectAnalysisStore,
            ProjectImageSpec,
            ReferenceRegionBundle,
            ReferenceRegionStore,
            analysis_settings_profile_from_graph,
        )
        from seedvision.pipeline import build_default_pipeline

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = self._write_image(root / "images" / "capture.png")
            reference_store = ReferenceRegionStore(root)
            foreground = np.zeros((36, 48), dtype=bool)
            foreground[7:16, 10:23] = True
            reference_store.save(
                image,
                ReferenceRegionBundle(shape=(36, 48), foreground=foreground),
            )
            ManualSeedCentreStore(root).save(
                image,
                np.asarray(((9.5, 11.25),), dtype=np.float64),
                mode="augment",
                source_shape=(36, 48),
            )
            store = ProjectAnalysisStore(root)
            profile = analysis_settings_profile_from_graph(build_default_pipeline())
            omitted_path = store.capture_and_save(
                analysis_settings=profile,
                images=(
                    ProjectImageSpec(
                        image,
                        (36, 48),
                        include_reference_regions=False,
                        include_manual_seed_centres=False,
                    ),
                ),
                selected_image=image,
                destination=root / "projects" / "omitted.seedfiddle-project.json",
            )
            listed_path = store.capture_and_save(
                analysis_settings=profile,
                images=(ProjectImageSpec(image, (36, 48)),),
                selected_image=image,
                destination=root / "projects" / "listed.seedfiddle-project.json",
            )
            window = self._window(root)
            try:
                key = str(image.resolve()).casefold()
                with patch.object(
                    window, "_confirm_project_replacement", return_value=True
                ):
                    self.assertTrue(window._open_project(omitted_path))
                self.assertNotIn(key, window._applied_foreground_reference_masks)
                self.assertNotIn(key, window._manual_seed_centre_states)
                self.assertFalse(window._project_dirty)

                with patch.object(
                    window, "_confirm_project_replacement", return_value=True
                ):
                    self.assertTrue(window._open_project(listed_path))
                pending = window._pending_unbound_reference_bundles[key]
                np.testing.assert_array_equal(pending.foreground, foreground)
                np.testing.assert_allclose(
                    window._manual_seed_centre_states[key].centres_source_xy,
                    ((9.5, 11.25),),
                )
                self.assertFalse(window._project_dirty)
            finally:
                self._dispose(window)

    def test_changed_valid_sidecars_load_but_keep_master_dirty(self) -> None:
        from PySide6.QtWidgets import QMessageBox

        from seedvision.persistence import (
            ManualSeedCentreStore,
            ProjectAnalysisStore,
            ProjectImageSpec,
            ReferenceRegionBundle,
            ReferenceRegionStore,
            analysis_settings_profile_from_graph,
        )
        from seedvision.pipeline import build_default_pipeline

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = self._write_image(root / "images" / "capture.png")
            reference_store = ReferenceRegionStore(root)
            original = np.zeros((36, 48), dtype=bool)
            original[2:6, 3:8] = True
            reference_store.save(
                image, ReferenceRegionBundle(shape=(36, 48), foreground=original)
            )
            centre_store = ManualSeedCentreStore(root)
            centre_store.save(
                image,
                np.asarray(((5.0, 6.0),), dtype=np.float64),
                mode="augment",
                source_shape=(36, 48),
            )
            store = ProjectAnalysisStore(root)
            project_path = store.capture_and_save(
                analysis_settings=analysis_settings_profile_from_graph(
                    build_default_pipeline()
                ),
                images=(ProjectImageSpec(image, (36, 48)),),
                selected_image=image,
                destination=root / "projects" / "changed-sidecar.seedfiddle-project.json",
            )
            changed = np.zeros((36, 48), dtype=bool)
            changed[18:30, 20:36] = True
            reference_store.save(
                image, ReferenceRegionBundle(shape=(36, 48), foreground=changed)
            )
            centre_store.save(
                image,
                np.asarray(((24.0, 21.5),), dtype=np.float64),
                mode="replace_automatic",
                source_shape=(36, 48),
            )

            window = self._window(root)
            try:
                with (
                    patch.object(
                        window, "_confirm_project_replacement", return_value=True
                    ),
                    patch.object(
                        QMessageBox,
                        "warning",
                        return_value=QMessageBox.StandardButton.Ok,
                    ),
                ):
                    self.assertTrue(window._open_project(project_path))
                key = str(image.resolve()).casefold()
                pending = window._pending_unbound_reference_bundles[key]
                np.testing.assert_array_equal(pending.foreground, changed)
                np.testing.assert_allclose(
                    window._manual_seed_centre_states[key].centres_source_xy,
                    ((24.0, 21.5),),
                )
                self.assertEqual(
                    window._manual_seed_centre_states[key].mode,
                    "replace_automatic",
                )
                self.assertTrue(window._project_dirty)
                affected = window._resolve_pending_project_reference_bundle(
                    key,
                    SimpleNamespace(
                        image_path=image,
                        calibration=SimpleNamespace(
                            corrected_bgr=np.zeros((36, 48, 3), dtype=np.uint8)
                        ),
                    ),
                )
                self.assertIn("project", affected)
                np.testing.assert_array_equal(
                    window._applied_foreground_reference_masks[key], changed
                )
            finally:
                self._dispose(window)

    def test_invalid_listed_sidecar_is_withheld_preserved_then_explicitly_superseded(self) -> None:
        from hashlib import sha256

        from PySide6.QtWidgets import QMessageBox

        from seedvision.persistence import (
            REFERENCE_REGIONS_SIDECAR,
            ProjectAnalysisStore,
            ProjectImageSpec,
            ReferenceRegionBundle,
            ReferenceRegionStore,
            analysis_settings_profile_from_graph,
        )
        from seedvision.pipeline import build_default_pipeline

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = self._write_image(root / "images" / "capture.png")
            reference_store = ReferenceRegionStore(root)
            mask = np.zeros((36, 48), dtype=bool)
            mask[5:13, 9:18] = True
            sidecar_path = reference_store.save(
                image, ReferenceRegionBundle(shape=(36, 48), foreground=mask)
            )
            store = ProjectAnalysisStore(root)
            project_path = store.capture_and_save(
                analysis_settings=analysis_settings_profile_from_graph(
                    build_default_pipeline()
                ),
                images=(ProjectImageSpec(image, (36, 48)),),
                selected_image=image,
                destination=root / "projects" / "invalid-sidecar.seedfiddle-project.json",
            )
            corrupt_bytes = b"not a valid categorical NPZ"
            sidecar_path.write_bytes(corrupt_bytes)
            payload = json.loads(project_path.read_text(encoding="utf-8"))
            payload["images"][0]["sidecars"][REFERENCE_REGIONS_SIDECAR][
                "sha256"
            ] = sha256(corrupt_bytes).hexdigest()
            project_path.write_text(json.dumps(payload), encoding="utf-8")

            # The loose startup workspace is intentionally independent of a
            # project manifest. Suppress its deterministic autoload so this
            # test reaches the project's INVALID_SIDECAR policy without a
            # startup modal.
            with patch(
                "seedvision.ui.main_window.ReferenceRegionStore.load_if_present",
                return_value=None,
            ):
                window = self._window(root)
            try:
                with (
                    patch.object(
                        window, "_confirm_project_replacement", return_value=True
                    ),
                    patch.object(
                        QMessageBox,
                        "warning",
                        return_value=QMessageBox.StandardButton.Ok,
                    ),
                ):
                    self.assertTrue(window._open_project(project_path))
                key = str(image.resolve()).casefold()
                self.assertNotIn(key, window._applied_foreground_reference_masks)
                self.assertTrue(window._project_dirty)
                self.assertEqual(sidecar_path.read_bytes(), corrupt_bytes)

                self.assertTrue(window._save_project())
                preserved = store.load(project_path)
                preserved_sidecar = preserved.images[0].sidecar(
                    REFERENCE_REGIONS_SIDECAR
                )
                self.assertIsNotNone(preserved_sidecar)
                assert preserved_sidecar is not None
                self.assertEqual(preserved_sidecar.path, sidecar_path.resolve())
                self.assertEqual(
                    preserved_sidecar.reference.sha256,
                    sha256(corrupt_bytes).hexdigest(),
                )
                self.assertEqual(sidecar_path.read_bytes(), corrupt_bytes)

                replacement = np.zeros((36, 48), dtype=bool)
                replacement[20:31, 22:39] = True
                replacement.flags.writeable = False
                window._applied_foreground_reference_masks[key] = replacement
                window._reference_layer_shapes[key] = (36, 48)
                self.assertIsNotNone(
                    window._persist_applied_reference_regions(automatic=False)
                )
                window._set_project_dirty()
                self.assertTrue(window._save_project())
                restored = store.load(project_path)
                self.assertIsNotNone(
                    restored.images[0].sidecar(REFERENCE_REGIONS_SIDECAR)
                )
                self.assertFalse(window._project_dirty)
            finally:
                self._dispose(window)

    def test_degraded_save_preserves_unresolved_record_until_locator_is_readded(self) -> None:
        from PySide6.QtWidgets import QMessageBox

        from seedvision.persistence import (
            ProjectAnalysisStore,
            ProjectImageSpec,
            analysis_settings_profile_from_graph,
        )
        from seedvision.pipeline import build_default_pipeline

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = self._write_image(root / "images" / "a.png")
            changed = self._write_image(
                root / "images" / "b.png", colour="#73583e"
            )
            store = ProjectAnalysisStore(root)
            project_path = store.capture_and_save(
                analysis_settings=analysis_settings_profile_from_graph(
                    build_default_pipeline()
                ),
                images=(
                    ProjectImageSpec(first, (36, 48)),
                    ProjectImageSpec(changed, (36, 48)),
                ),
                selected_image=first,
                destination=root / "projects" / "degraded.seedfiddle-project.json",
            )
            original = store.load(project_path, verify_files=False).document
            self._write_image(changed, colour="#284f67")
            window = self._window(root)
            try:
                def warning_answer(_parent, title, *_args, **_kwargs):
                    if title == "Some project images are unavailable":
                        return QMessageBox.StandardButton.Yes
                    return QMessageBox.StandardButton.Ok

                with (
                    patch.object(
                        window, "_confirm_project_replacement", return_value=True
                    ),
                    patch.object(QMessageBox, "warning", side_effect=warning_answer),
                ):
                    self.assertTrue(window._open_project(project_path))
                self.assertEqual(self._image_paths(window), (first,))
                self.assertTrue(window._project_dirty)
                self.assertTrue(window._save_project())
                preserved = store.load(project_path, verify_files=False).document
                self.assertEqual(
                    tuple(record.identifier for record in preserved.images),
                    tuple(record.identifier for record in original.images),
                )
                self.assertEqual(preserved.images[1], original.images[1])

                window._add_image(changed, open_now=False)
                self.assertTrue(window._save_project())
                replaced = store.load(project_path).document
                self.assertEqual(len(replaced.images), 2)
                self.assertNotEqual(
                    replaced.images[1].source.sha256,
                    original.images[1].source.sha256,
                )
            finally:
                self._dispose(window)

    def test_restored_source_keeps_unavailable_record_explicit_sidecar_paths(self) -> None:
        from PySide6.QtWidgets import QMessageBox

        from seedvision.persistence import (
            MANUAL_SEED_CENTRES_SIDECAR,
            REFERENCE_REGIONS_SIDECAR,
            ManualSeedCentreStore,
            ProjectAnalysisStore,
            ProjectImageSpec,
            ReferenceRegionBundle,
            ReferenceRegionStore,
            analysis_settings_profile_from_graph,
        )
        from seedvision.pipeline import build_default_pipeline
        from seedvision.ui.main_window import _path_identity

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = self._write_image(root / "images" / "restored.png")
            image_bytes = image.read_bytes()
            reference_store = ReferenceRegionStore(root)
            foreground = np.zeros((36, 48), dtype=bool)
            foreground[4:13, 6:19] = True
            canonical_reference = reference_store.save(
                image,
                ReferenceRegionBundle(shape=(36, 48), foreground=foreground),
            )
            centre_store = ManualSeedCentreStore(root)
            canonical_centres = centre_store.save(
                image,
                np.asarray(((12.5, 9.25),), dtype=np.float64),
                mode="augment",
                source_shape=(36, 48),
            )
            explicit_reference = (
                root / "projects" / "imported" / "restored-reference.npz"
            )
            explicit_centres = (
                root / "projects" / "imported" / "restored-centres.npz"
            )
            explicit_reference.parent.mkdir(parents=True, exist_ok=True)
            canonical_reference.replace(explicit_reference)
            canonical_centres.replace(explicit_centres)

            store = ProjectAnalysisStore(root)
            project_path = store.capture_and_save(
                analysis_settings=analysis_settings_profile_from_graph(
                    build_default_pipeline()
                ),
                images=(
                    ProjectImageSpec(
                        image,
                        (36, 48),
                        reference_regions_path=explicit_reference,
                        manual_seed_centres_path=explicit_centres,
                    ),
                ),
                selected_image=image,
                destination=root / "projects" / "restored.seedfiddle-project.json",
            )
            image.unlink()
            window = self._window(root)
            try:
                with (
                    patch.object(
                        QMessageBox,
                        "warning",
                        return_value=QMessageBox.StandardButton.Yes,
                    ),
                    patch.object(
                        window, "_confirm_project_replacement", return_value=True
                    ),
                ):
                    self.assertTrue(window._open_project(project_path))
                key = _path_identity(image)
                self.assertEqual(
                    window._project_reference_sidecar_paths[key],
                    explicit_reference.resolve(),
                )
                self.assertEqual(
                    window._project_manual_centre_sidecar_paths[key],
                    explicit_centres.resolve(),
                )
                self.assertIn(key, window._withheld_reference_sidecars)
                self.assertIn(key, window._withheld_manual_centre_sidecars)

                image.write_bytes(image_bytes)
                window._add_image(image, open_now=False)
                self.assertTrue(window._save_project())
                restored = store.load(project_path)
                reference = restored.images[0].sidecar(REFERENCE_REGIONS_SIDECAR)
                centres = restored.images[0].sidecar(MANUAL_SEED_CENTRES_SIDECAR)
                self.assertIsNotNone(reference)
                self.assertIsNotNone(centres)
                assert reference is not None and centres is not None
                self.assertEqual(reference.path, explicit_reference.resolve())
                self.assertEqual(centres.path, explicit_centres.resolve())
                self.assertFalse(reference_store.path_for(image).exists())
                self.assertFalse(centre_store.path_for(image).exists())
            finally:
                self._dispose(window)

    def test_new_project_image_does_not_capture_unseen_loose_sidecars(self) -> None:
        from seedvision.persistence import (
            ManualSeedCentreStore,
            ProjectAnalysisStore,
            ProjectImageSpec,
            ReferenceRegionBundle,
            ReferenceRegionStore,
            analysis_settings_profile_from_graph,
        )
        from seedvision.pipeline import build_default_pipeline
        from seedvision.ui.main_window import _path_identity

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = self._write_image(root / "images" / "a.png")
            added = self._write_image(root / "images" / "b.png")
            reference_store = ReferenceRegionStore(root)
            mask = np.zeros((36, 48), dtype=bool)
            mask[7:15, 9:20] = True
            reference_path = reference_store.save(
                added, ReferenceRegionBundle(shape=(36, 48), foreground=mask)
            )
            centre_store = ManualSeedCentreStore(root)
            centre_path = centre_store.save(
                added,
                np.asarray(((15.0, 11.0),), dtype=np.float64),
                mode="augment",
                source_shape=(36, 48),
            )
            store = ProjectAnalysisStore(root)
            project_path = store.capture_and_save(
                analysis_settings=analysis_settings_profile_from_graph(
                    build_default_pipeline()
                ),
                images=(ProjectImageSpec(first, (36, 48)),),
                selected_image=first,
                destination=root / "projects" / "unseen.seedfiddle-project.json",
            )
            window = self._window(root)
            try:
                with patch.object(
                    window, "_confirm_project_replacement", return_value=True
                ):
                    self.assertTrue(window._open_project(project_path))
                key = _path_identity(added)
                window._add_image(added, open_now=False)
                self.assertIn(key, window._withheld_reference_sidecars)
                self.assertIn(key, window._withheld_manual_centre_sidecars)
                self.assertNotIn(key, window._project_reference_sidecar_paths)
                self.assertNotIn(key, window._project_manual_centre_sidecar_paths)

                self.assertTrue(window._save_project())
                saved = store.load(project_path)
                added_record = next(image for image in saved.images if image.path == added)
                self.assertEqual(added_record.sidecars, ())
                self.assertTrue(reference_path.is_file())
                self.assertTrue(centre_path.is_file())
            finally:
                self._dispose(window)

    def test_explicit_noncanonical_sidecar_loads_and_locator_is_preserved_until_resaved(self) -> None:
        from seedvision.persistence import (
            REFERENCE_REGIONS_SIDECAR,
            ProjectAnalysisStore,
            ProjectImageSpec,
            ReferenceRegionBundle,
            ReferenceRegionStore,
            analysis_settings_profile_from_graph,
        )
        from seedvision.pipeline import build_default_pipeline

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = self._write_image(root / "images" / "capture.png")
            reference_store = ReferenceRegionStore(root)
            mask = np.zeros((36, 48), dtype=bool)
            mask[8:19, 12:27] = True
            canonical = reference_store.save(
                image, ReferenceRegionBundle(shape=(36, 48), foreground=mask)
            )
            explicit = root / "projects" / "imported" / "legacy-reference.npz"
            explicit.parent.mkdir(parents=True, exist_ok=True)
            canonical.replace(explicit)
            store = ProjectAnalysisStore(root)
            project_path = store.capture_and_save(
                analysis_settings=analysis_settings_profile_from_graph(
                    build_default_pipeline()
                ),
                images=(
                    ProjectImageSpec(
                        image,
                        (36, 48),
                        include_reference_regions=True,
                        include_manual_seed_centres=False,
                        reference_regions_path=explicit,
                    ),
                ),
                selected_image=image,
                destination=root / "projects" / "explicit.seedfiddle-project.json",
            )
            window = self._window(root)
            try:
                with patch.object(
                    window, "_confirm_project_replacement", return_value=True
                ):
                    self.assertTrue(window._open_project(project_path))
                key = str(image.resolve()).casefold()
                self.assertEqual(
                    window._project_reference_sidecar_paths[key], explicit.resolve()
                )
                np.testing.assert_array_equal(
                    window._pending_unbound_reference_bundles[key].foreground,
                    mask,
                )
                self.assertTrue(window._save_project())
                preserved = store.load(project_path)
                preserved_sidecar = preserved.images[0].sidecar(
                    REFERENCE_REGIONS_SIDECAR
                )
                self.assertIsNotNone(preserved_sidecar)
                assert preserved_sidecar is not None
                self.assertEqual(preserved_sidecar.path, explicit.resolve())

                window._resolve_pending_project_reference_bundle(
                    key,
                    SimpleNamespace(
                        image_path=image,
                        calibration=SimpleNamespace(
                            corrected_bgr=np.zeros((36, 48, 3), dtype=np.uint8)
                        ),
                    ),
                )
                self.assertIsNotNone(
                    window._persist_applied_reference_regions(automatic=False)
                )
                window._set_project_dirty()
                self.assertTrue(window._save_project())
                rebound = store.load(project_path)
                rebound_sidecar = rebound.images[0].sidecar(
                    REFERENCE_REGIONS_SIDECAR
                )
                self.assertIsNotNone(rebound_sidecar)
                assert rebound_sidecar is not None
                self.assertEqual(
                    rebound_sidecar.path,
                    reference_store.path_for(image).resolve(),
                )
                self.assertTrue(explicit.is_file())
            finally:
                self._dispose(window)

    def test_corrected_shape_sidecar_waits_for_calibration_then_installs_or_withholds(self) -> None:
        from PySide6.QtWidgets import QMessageBox

        from seedvision.persistence import (
            ProjectAnalysisStore,
            ProjectImageSpec,
            ReferenceRegionBundle,
            ReferenceRegionStore,
            analysis_settings_profile_from_graph,
        )
        from seedvision.pipeline import build_default_pipeline

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = self._write_image(root / "images" / "capture.png")
            corrected_shape = (29, 41)
            mask = np.zeros(corrected_shape, dtype=bool)
            mask[5:16, 7:24] = True
            ReferenceRegionStore(root).save(
                image,
                ReferenceRegionBundle(
                    shape=corrected_shape,
                    foreground=mask,
                ),
            )
            store = ProjectAnalysisStore(root)
            project_path = store.capture_and_save(
                analysis_settings=analysis_settings_profile_from_graph(
                    build_default_pipeline()
                ),
                images=(ProjectImageSpec(image, (36, 48)),),
                selected_image=image,
                destination=root / "projects" / "corrected.seedfiddle-project.json",
            )
            # Loose-workspace autoload has no calibration result and therefore
            # cannot validate a corrected-coordinate shape. The project path
            # below is the contract under test.
            with patch(
                "seedvision.ui.main_window.ReferenceRegionStore.load_if_present",
                return_value=None,
            ):
                window = self._window(root)
            try:
                with patch.object(
                    window, "_confirm_project_replacement", return_value=True
                ):
                    self.assertTrue(window._open_project(project_path))
                key = str(image.resolve()).casefold()
                self.assertIn(key, window._pending_unbound_reference_bundles)
                self.assertNotIn(key, window._applied_foreground_reference_masks)
                self.assertFalse(window._project_dirty)

                matching_result = SimpleNamespace(
                    image_path=image,
                    calibration=SimpleNamespace(
                        corrected_bgr=np.zeros(
                            (*corrected_shape, 3), dtype=np.uint8
                        )
                    ),
                )
                window.image_view.show_analysis(matching_result, render=False)
                affected = window._resolve_pending_project_reference_bundle(
                    key, matching_result
                )
                self.assertIn("project", affected)
                np.testing.assert_array_equal(
                    window._applied_foreground_reference_masks[key], mask
                )

                with patch.object(
                    window, "_confirm_project_replacement", return_value=True
                ):
                    self.assertTrue(window._open_project(project_path))
                self.assertIn(key, window._pending_unbound_reference_bundles)
                mismatching_result = SimpleNamespace(
                    image_path=image,
                    calibration=SimpleNamespace(
                        corrected_bgr=np.zeros((31, 41, 3), dtype=np.uint8)
                    ),
                )
                window.image_view.show_analysis(mismatching_result, render=False)
                with patch.object(QMessageBox, "warning") as warning:
                    affected = window._resolve_pending_project_reference_bundle(
                        key, mismatching_result
                    )
                self.assertEqual(affected, set())
                warning.assert_called_once()
                self.assertNotIn(key, window._applied_foreground_reference_masks)
                self.assertIn(key, window._withheld_reference_sidecars)
                self.assertTrue(window._project_dirty)
            finally:
                self._dispose(window)


if __name__ == "__main__":
    unittest.main()
