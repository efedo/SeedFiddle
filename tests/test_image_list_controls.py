"""Native image-list actions, file preservation and edit ownership."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import os
import unittest
from unittest.mock import patch

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox

from seedvision.persistence import (
    BiologicalContext, ProjectAnalysisStore, ProjectImageSpec, ReferenceRegionBundle,
    ReferenceRegionStore, analysis_settings_profile_from_graph,
)
from seedvision.pipeline import build_default_pipeline
from seedvision.ui.main_window import MainWindow, _path_identity
from tests.review_fixtures import dispose_window, isolated_window
from tests import test_project_analysis_ui


class ImageListControlsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def test_header_open_and_row_removal_preserve_files_and_unresolved_project_records(self):
        with TemporaryDirectory() as temporary, patch.object(MainWindow, "_release_unused_cuda_blocks"):
            root = Path(temporary)
            first, removed, missing = (
                test_project_analysis_ui.ProjectAnalysisUiTests._write_image(root / "images" / name)
                for name in ("a.png", "b.png", "c.png")
            )
            foreground = np.zeros((36, 48), dtype=bool)
            foreground[3:9, 4:12] = True
            sidecar = ReferenceRegionStore(root).save(removed, ReferenceRegionBundle(
                shape=(36, 48), source_to_corrected=np.eye(3), foreground=foreground))
            preserved_bytes = {path: path.read_bytes() for path in (first, removed, sidecar)}
            store = ProjectAnalysisStore(root)
            project = store.capture_and_save(
                analysis_settings=analysis_settings_profile_from_graph(build_default_pipeline()),
                images=tuple(ProjectImageSpec(path, (36, 48), biological_context=BiologicalContext("unknown"))
                             for path in (first, removed, missing)),
                selected_image=missing, destination=root / "project.seedfiddle-project.json")
            original = store.load(project, verify_files=False).document
            removed_id, missing_id = original.images[1].identifier, original.images[2].identifier
            missing.unlink()
            window = isolated_window(root)
            try:
                with patch.object(window, "_confirm_project_replacement", return_value=True), \
                     patch.object(QMessageBox, "warning", return_value=QMessageBox.StandardButton.Yes):
                    self.assertTrue(window._open_project(project))
                window.resize(1440, 900)
                window.show()
                self.application.processEvents()
                self.assertNotIn(window.open_action, window.workflow_toolbar.actions())
                self.assertIs(window.open_images_button.defaultAction(), window.open_action)
                with patch("seedvision.ui.main_window.QFileDialog.getOpenFileNames", return_value=([], "")) as choose:
                    window.open_images_button.click()
                choose.assert_called_once()
                self.assertGreater(window.image_list.height() / window.height(), .27)
                self.assertLess(window.image_list.height() / window.height(), .39)
                key = _path_identity(removed)
                window._draft_foreground_reference_masks[key] = foreground
                window._optimization_targets[key] = {"area": 12}
                window._image_review_states[key] = "reviewed"
                window._source_fingerprints[key] = "removed fingerprint"
                window._project_correction_bindings = (SimpleNamespace(image_id=removed_id),)
                row = window.image_list.item(1)
                close_rect = window._image_list_delegate.remove_rect(window.image_list.visualItemRect(row))
                QTest.mouseClick(window.image_list.viewport(), Qt.MouseButton.LeftButton, pos=close_rect.center())
                self.assertEqual(window.image_view.image_path, first)
                self.assertEqual(window.image_list.count(), 1)
                self.assertNotIn(key, window._image_paths)
                self.assertNotIn(key, window._draft_foreground_reference_masks)
                self.assertNotIn(key, window._optimization_targets)
                self.assertNotIn(key, window._image_review_states)
                self.assertNotIn(key, window._source_fingerprints)
                self.assertEqual(window._project_correction_bindings, ())
                self.assertNotIn(removed_id, window._project_original_image_records)
                self.assertEqual(window._project_unresolved_image_records[missing_id], original.images[2])
                self.assertEqual(window._project_unresolved_selected_image_id, missing_id)
                self.assertTrue(window._project_dirty)
                self.assertTrue(window._save_project())
                saved = store.load(project, verify_files=False).document
                self.assertEqual(saved.images, (original.images[0], original.images[2]))
                self.assertEqual(saved.selected_image_id, missing_id)
                for path, contents in preserved_bytes.items():
                    self.assertEqual(path.read_bytes(), contents)
            finally:
                dispose_window(window)
                self.application.processEvents()

    def test_removal_protects_edits_and_busy_work_then_clears_selected_native_display(self):
        with TemporaryDirectory() as temporary, patch.object(MainWindow, "_release_unused_cuda_blocks"):
            root = Path(temporary)
            first, second = (
                test_project_analysis_ui.ProjectAnalysisUiTests._write_image(root / "images" / name)
                for name in ("a.png", "b.png")
            )
            window = isolated_window(root)
            try:
                key = _path_identity(first)
                second_key = _path_identity(second)
                from seedvision.persistence.project_analysis import PROJECT_RELATIVE, _image_identifier
                window._project_correction_bindings = (SimpleNamespace(
                    image_id=_image_identifier(PROJECT_RELATIVE, "images/a.png")),)
                foreground = np.zeros((36, 48), dtype=bool)
                foreground[2:8, 5:11] = True
                window._draft_foreground_reference_masks[key] = foreground
                window._reference_masks_dirty.add(key)
                with patch.object(window, "_background_work_is_active", return_value=True), \
                     patch.object(QMessageBox, "information") as busy:
                    self.assertFalse(window._remove_image(first))
                busy.assert_called_once()
                with patch.object(QMessageBox, "warning", return_value=QMessageBox.StandardButton.Cancel):
                    self.assertFalse(window._remove_image(first))
                self.assertIs(window._draft_foreground_reference_masks[key], foreground)
                with patch.object(QMessageBox, "warning", return_value=QMessageBox.StandardButton.Save), \
                     patch.object(window, "_apply_and_save_all_project_drafts", return_value=False):
                    self.assertFalse(window._remove_image(first))
                self.assertEqual(window.image_view.image_path, first)
                # An unrelated failed sidecar must not block saving this row's
                # edits, or be retried as a side effect of its removal.
                window._unsaved_reference_sidecars.add(second_key)
                with patch.object(QMessageBox, "warning", return_value=QMessageBox.StandardButton.Save):
                    self.assertTrue(window._remove_image(first))
                self.assertEqual(window.image_view.image_path, second)
                self.assertEqual(window._project_correction_bindings, ())
                self.assertIn(second_key, window._unsaved_reference_sidecars)
                sidecar = ReferenceRegionStore(root).path_for(first)
                self.assertTrue(sidecar.is_file())
                saved_sidecar = sidecar.read_bytes()
                window._unsaved_manual_centre_sidecars.add(second_key)
                with patch.object(QMessageBox, "warning", return_value=QMessageBox.StandardButton.Save), \
                     patch.object(window, "_retry_unsaved_project_sidecars", return_value=False):
                    self.assertFalse(window._remove_image(second))
                controller = window.pipeline_workflow
                controller.selector.setCurrentIndex(next(
                    index for index in range(1, controller.selector.count())
                    if controller.selector.itemData(index).recipe_id == "plantcv_seed_binary_working"))
                controller.result = SimpleNamespace(node=unittest.mock.Mock(side_effect=StopIteration))
                controller.facts.setPlainText("Old image result")
                with patch.object(QMessageBox, "warning", return_value=QMessageBox.StandardButton.Discard):
                    self.assertTrue(window._remove_image(second))
                self.assertEqual(window.image_list.count(), 0)
                self.assertIsNone(window.image_view.image_path)
                self.assertIsNone(controller.result)
                self.assertNotIn("Old image result", controller.facts.toPlainText())
                self.assertIsNone(window._current_image_key())
                self.assertFalse(window.analyze_action.isEnabled())
                self.assertFalse(window._unsaved_manual_centre_sidecars)
                self.assertTrue(first.is_file())
                self.assertTrue(second.is_file())
                self.assertEqual(sidecar.read_bytes(), saved_sidecar)
            finally:
                dispose_window(window)
                self.application.processEvents()
