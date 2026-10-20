from __future__ import annotations

from tests.functional_units import functional_units

from tests.review_fixtures import dispose_window

import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


@functional_units
class InstanceMaskImportUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        try:
            from PySide6.QtWidgets import QApplication
        except ImportError as error:
            raise unittest.SkipTest(f"PySide6 unavailable: {error}")
        cls.application = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        from PySide6.QtGui import QColor, QImage

        from seedvision.ui.main_window import MainWindow

        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.image_path = self.root / "images" / "capture.png"
        self.image_path.parent.mkdir(parents=True)
        image = QImage(80, 60, QImage.Format.Format_RGB888)
        image.fill(QColor("#8b927f"))
        self.assertTrue(image.save(str(self.image_path)))
        self.window = MainWindow(self.root)
        self.addCleanup(dispose_window, self.window)
        self.window.image_view.set_overlay_mode("none")
        self.result = SimpleNamespace(
            calibration=SimpleNamespace(
                corrected_bgr=np.full((60, 80, 3), 128, dtype=np.uint8),
                affine_matrix=np.eye(3, dtype=np.float64),
            ),
            layers=SimpleNamespace(
                edge_likelihood=np.zeros((60, 80), dtype=np.float32),
                undirected_edge_hue=np.zeros((60, 80), dtype=np.float32),
                directed_edge_hue=np.zeros((60, 80), dtype=np.float32),
                offset_x=0,
                offset_y=0,
            ),
            crop_offset=(0, 0),
            estimated_seed_diameter_px=20.0,
        )
        key = self.window._current_image_key()
        self.assertIsNotNone(key)
        assert key is not None
        self.key = key
        self.window._analyses[key] = self.result
        self.window.image_view._analysis_result = self.result
        self.window.image_view._render_analysis = lambda: None
        self.window._sync_background_controls()
        self.application.processEvents()

    @staticmethod
    def _imported(labels: np.ndarray, *, bundled: bool = True):
        from seedvision.persistence import ImportedInstanceMask

        return ImportedInstanceMask(
            labels=labels,
            source_path=Path("bundle.png" if bundled else "manual.npz"),
            coordinate_space=("source->corrected" if bundled else "corrected"),
            origin=(
                "bundled:seed-instance-references/masks/capture.png;reviewed=false"
                if bundled
                else "import:C:/manual.npz"
            ),
            reviewed=(False if bundled else None),
            external_reference_count=(1 if bundled else None),
        )

    def check_test_panel_import_is_draft_with_provenance_warning_and_undo(self) -> None:
        labels = np.zeros((60, 80), dtype=np.uint16)
        labels[4:9, 5:10] = 7
        labels[30:35, 45:50] = 7
        labels[12:22, 20:31] = 9

        self.window.instance_smart_fill_button.setChecked(True)
        self.window.show_instance_annotations_checkbox.setChecked(False)
        imported = self._imported(labels)
        with patch(
            "seedvision.ui.main_window.load_bundled_instance_mask",
            return_value=imported,
        ), patch(
            "seedvision.ui.main_window.QFileDialog.getOpenFileName"
        ) as chooser:
            self.window.load_instance_reference_button.click()

        chooser.assert_not_called()
        self.assertTrue(
            np.array_equal(self.window._draft_instance_annotations[self.key], labels)
        )
        self.assertEqual(
            self.window._draft_instance_annotation_origins[self.key], imported.origin
        )
        self.assertIn(self.key, self.window._instance_annotations_dirty)
        self.assertTrue(self.window.annotate_instances_action.isChecked())
        self.assertTrue(self.window.instance_smart_fill_button.isChecked())
        self.assertFalse(self.window.show_instance_annotations_checkbox.isChecked())
        self.assertEqual(
            self.window._instance_undo_histories[self.key].next_label,
            "imported seed labels",
        )
        self.assertIn(
            "1 seed ID has disconnected areas: 7",
            self.window.image_view.instance_continuity_warning_banner.text(),
        )
        self.assertIn("unreviewed", self.window.statusBar().currentMessage())
        self.assertIn("IDs were preserved exactly", self.window.statusBar().currentMessage())

        self.window._undo_instance_reference_edit()
        current = self.window._draft_instance_annotations.get(
            self.key, self.window._applied_instance_annotations.get(self.key)
        )
        self.assertTrue(current is None or not np.any(current))
        self.assertEqual(len(self.window._instance_undo_histories[self.key]),0)
        self.assertTrue(self.window._instance_undo_histories[self.key].can_redo)

    def check_test_bundle_import_is_explicit_and_preserves_applied_labels(self) -> None:
        from PySide6.QtWidgets import QMessageBox
        applied = np.zeros((60, 80), np.uint16)
        applied[5:15, 8:20] = 1
        bundled = np.zeros_like(applied)
        bundled[24:38, 31:47] = 9
        self.window._applied_instance_annotations[self.key] = applied
        self.window._set_instance_draft_state(self.key, applied, 'manual')
        with patch('seedvision.ui.main_window.load_bundled_instance_mask',
                   return_value=self._imported(bundled)) as loader, \
             patch.object(QMessageBox, 'question', return_value=QMessageBox.StandardButton.Yes):
            self.window.load_instance_labels_action.trigger()
        loader.assert_called_once()
        np.testing.assert_array_equal(self.window._applied_instance_annotations[self.key], applied)
        np.testing.assert_array_equal(self.window._draft_instance_annotations[self.key], bundled)
        self.assertIn(self.key, self.window._instance_annotations_dirty)
        self.window._undo_instance_reference_edit()
        np.testing.assert_array_equal(self.window._draft_instance_annotations.get(
            self.key, self.window._applied_instance_annotations[self.key]), applied)

    def check_test_no_bundle_explains_explicit_file_import(self) -> None:
        from PySide6.QtWidgets import QMessageBox
        with patch('seedvision.ui.main_window.load_bundled_instance_mask', return_value=None), \
             patch('seedvision.ui.main_window.QFileDialog.getOpenFileName') as chooser, \
             patch.object(QMessageBox, 'information') as information:
            self.window.load_instance_labels_action.trigger()
        chooser.assert_not_called()
        self.assertIn('Import seed labels from file', information.call_args.args[2])
        self.assertNotIn(self.key, self.window._instance_annotations_dirty)

    def check_test_explicit_file_chooser_bypasses_an_available_bundle(self) -> None:
        bundled_labels = np.zeros((60, 80), dtype=np.uint16)
        bundled_labels[2:10, 3:12] = 4
        manual_labels = np.zeros_like(bundled_labels)
        manual_labels[22:39, 31:52] = 27
        selected = self.root / "reviewed-correction.npz"
        np.savez_compressed(selected, labels=manual_labels)

        with patch(
            "seedvision.ui.main_window.load_bundled_instance_mask",
            return_value=self._imported(bundled_labels),
        ) as bundled_loader, patch(
            "seedvision.ui.main_window.QFileDialog.getOpenFileName",
            return_value=(str(selected), "Lossless categorical masks"),
        ):
            self.window.choose_instance_labels_action.trigger()

        bundled_loader.assert_not_called()
        self.assertTrue(
            np.array_equal(
                self.window._draft_instance_annotations[self.key], manual_labels
            )
        )
        self.assertEqual(
            self.window._draft_instance_annotation_origins[self.key],
            f"import:{selected.resolve()}",
        )
        self.assertIn(self.key, self.window._instance_annotations_dirty)
        self.assertEqual(
            self.window._instance_undo_histories[self.key].next_label,
            "imported seed labels",
        )

    def check_test_validation_failure_warns_without_mutating_draft_or_history(self) -> None:
        from seedvision.persistence import InstanceMaskImportError

        baseline = np.zeros((60, 80), dtype=np.uint16)
        baseline[10:20, 10:20] = 4
        self.window._set_instance_draft_state(self.key, baseline, "manual:existing")
        self.window.image_view.set_instance_annotations(
            baseline, copy=False, render=False
        )
        before = baseline.copy()
        with patch(
            "seedvision.ui.main_window.load_bundled_instance_mask",
            side_effect=InstanceMaskImportError("fingerprint or dimensions mismatch"),
        ), patch(
            "seedvision.ui.main_window.QMessageBox.critical"
        ) as warning, patch(
            "seedvision.ui.main_window.QFileDialog.getOpenFileName"
        ) as chooser:
            self.window._load_instance_reference_mask()

        warning.assert_called_once()
        chooser.assert_not_called()
        self.assertTrue(
            np.array_equal(self.window._draft_instance_annotations[self.key], before)
        )
        self.assertEqual(
            self.window._draft_instance_annotation_origins[self.key],
            "manual:existing",
        )
        self.assertNotIn(self.key, self.window._instance_undo_histories)

    def check_test_replacement_cancel_leaves_existing_draft_untouched(self) -> None:
        from PySide6.QtWidgets import QMessageBox

        existing = np.zeros((60, 80), dtype=np.uint16)
        existing[2:7, 2:7] = 1
        replacement = np.zeros_like(existing)
        replacement[30:40, 30:40] = 2
        self.window._set_instance_draft_state(self.key, existing, "manual:existing")
        with patch.object(
            QMessageBox,
            "question",
            return_value=QMessageBox.StandardButton.No,
        ):
            changed = self.window._install_imported_instance_mask(
                self._imported(replacement)
            )
        self.assertFalse(changed)
        self.assertTrue(
            np.array_equal(self.window._draft_instance_annotations[self.key], existing)
        )
        self.assertNotIn(self.key, self.window._instance_undo_histories)

    def check_test_training_export_readiness_and_revision_identity(self) -> None:
        from PySide6.QtWidgets import QDialog, QMessageBox
        from seedvision.persistence.reference_regions import SeedInstanceAnnotation
        from seedvision.learning.data import LearningManifest, LearningSample
        from seedvision.learning.contracts import FeatureStackSpec
        labels = np.zeros((60, 80), np.uint16)
        labels[10:20, 10:20] = 4
        labels[30:40, 30:40] = 8
        self.window._applied_instance_annotations[self.key] = labels
        self.window._applied_seed_annotations[self.key] = {
            4: SeedInstanceAnnotation(4, shape_reviewed=True, outline_visibility='complete')}
        with patch.object(QMessageBox, 'warning') as warning, \
             patch('seedvision.ui.main_window.LearningExportDialog') as dialog:
            self.window._export_learning_sample()
        dialog.assert_not_called()
        self.assertIn('1 excluded', warning.call_args.args[2])
        self.assertIn('8', warning.call_args.args[2])
        self.window._applied_seed_annotations[self.key][8] = SeedInstanceAnnotation(
            8, shape_reviewed=True, outline_visibility='complete')
        self.window._instance_annotations_dirty.add(self.key)
        with patch.object(QMessageBox, 'warning') as warning:
            self.window._export_learning_sample()
        self.assertIn('unapplied draft', warning.call_args.args[2])
        self.window._instance_annotations_dirty.clear()
        destination = self.root / 'learning-data'
        manifest = LearningManifest('test', FeatureStackSpec(), (
            LearningSample('capture', '', '', '', 'g', 'train', False),))
        manifest.save(destination / 'manifest.json')
        (destination / '.learning-revisions' / 'capture_r2').mkdir(parents=True)
        from seedvision.ui.learning_workflow import LearningExportDialog
        with patch('seedvision.ui.main_window.LearningExportDialog', wraps=LearningExportDialog) as constructor, \
             patch.object(QDialog, 'exec', return_value=QDialog.DialogCode.Rejected):
            self.window._export_learning_sample()
        self.assertEqual(constructor.call_args.kwargs['default_identifier'], 'capture_r3')
        self.assertEqual(constructor.call_args.kwargs['default_revision'], '3')
        def reviewed_dialog(**kwargs):
            dialog = LearningExportDialog(**kwargs)
            dialog.group_edit.setText('capture-group')
            dialog.author_edit.setText('Reviewer')
            dialog.reviewed_checkbox.setChecked(True)
            dialog.species_reviewed_checkbox.setChecked(True)
            return dialog
        with patch('seedvision.ui.main_window.LearningExportDialog', side_effect=reviewed_dialog), \
             patch.object(QDialog, 'exec', return_value=QDialog.DialogCode.Accepted), \
             patch('seedvision.learning.export.export_analysis_sample', return_value=SimpleNamespace(
                 identifier='capture_r3', reviewed=True)) as exporter, \
             patch('seedvision.ui.main_window.run_persistence', side_effect=lambda parent, call, text: call()), \
             patch.object(QMessageBox, 'information') as information:
            self.window._export_learning_sample()
        exporter.assert_called_once()
        self.assertIs(exporter.call_args.args[0], self.result)
        np.testing.assert_array_equal(exporter.call_args.args[1], labels)
        self.assertTrue(exporter.call_args.kwargs['reviewed'])
        self.assertEqual(exporter.call_args.kwargs['annotation_author'], 'Reviewer')
        self.assertIn('Train or refine', information.call_args.args[2])

    def check_test_seed_label_export_uses_applied_16_bit_labels(self) -> None:
        import cv2
        from PySide6.QtWidgets import QFileDialog
        applied = np.zeros((60, 80), np.uint16)
        applied[10:20, 10:20] = 31777
        draft = applied.copy()
        draft[30:40, 30:40] = 42
        self.window._applied_instance_annotations[self.key] = applied
        self.window._draft_instance_annotations[self.key] = draft
        invalid = self.root / 'lossy.jpg'
        from PySide6.QtWidgets import QMessageBox
        with patch.object(QFileDialog, 'getSaveFileName', return_value=(str(invalid), '')), \
             patch.object(QMessageBox, 'critical') as warning:
            self.window._save_instance_labels()
        self.assertFalse(invalid.exists())
        self.assertIn('lossless 16-bit', warning.call_args.args[2])
        destination = self.root / 'export.png'
        with patch.object(QFileDialog, 'getSaveFileName', return_value=(str(destination), '')):
            self.window._save_instance_labels()
        loaded = cv2.imread(str(destination), cv2.IMREAD_UNCHANGED)
        self.assertEqual(loaded.dtype, np.uint16)
        np.testing.assert_array_equal(loaded, applied)
        self.window._unsaved_reference_sidecars.add(self.key)
        self.window._sync_background_controls()
        self.assertIn('disk save failed', self.window.instance_annotation_confirmation_label.text())

    def test_workflow_lifecycle(self):
        """6 related InstanceMaskImportUiTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_panel_import_is_draft_with_provenance_warning_and_undo)
        self.scenario(self.check_test_bundle_import_is_explicit_and_preserves_applied_labels)
        self.scenario(self.check_test_no_bundle_explains_explicit_file_import)
        self.scenario(self.check_test_explicit_file_chooser_bypasses_an_available_bundle)
        self.scenario(self.check_test_training_export_readiness_and_revision_identity)
        self.scenario(self.check_test_seed_label_export_uses_applied_16_bit_labels)

    def test_validation_and_recovery(self):
        """2 related InstanceMaskImportUiTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_validation_failure_warns_without_mutating_draft_or_history)
        self.scenario(self.check_test_replacement_cancel_leaves_existing_draft_untouched)


if __name__ == "__main__":
    unittest.main()
