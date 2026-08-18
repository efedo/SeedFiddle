from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


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
        self.addCleanup(self.window.close)
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

    def test_panel_import_is_draft_with_provenance_warning_and_undo(self) -> None:
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
            "imported seed-instance reference mask",
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
        self.assertNotIn(self.key, self.window._instance_undo_histories)

    def test_no_bundle_falls_back_to_npz_file_chooser(self) -> None:
        labels = np.zeros((60, 80), dtype=np.uint16)
        labels[5:20, 8:24] = 31_777
        selected = self.root / "manual.npz"
        np.savez_compressed(selected, annotated_seeds=labels)

        with patch(
            "seedvision.ui.main_window.load_bundled_instance_mask",
            return_value=None,
        ), patch(
            "seedvision.ui.main_window.QFileDialog.getOpenFileName",
            return_value=(str(selected), "Lossless categorical masks"),
        ):
            self.window.load_instance_labels_action.trigger()

        self.assertTrue(
            np.array_equal(self.window._draft_instance_annotations[self.key], labels)
        )
        self.assertEqual(
            self.window._draft_instance_annotation_origins[self.key],
            f"import:{selected.resolve()}",
        )

    def test_explicit_file_chooser_bypasses_an_available_bundle(self) -> None:
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
            "imported seed-instance reference mask",
        )

    def test_validation_failure_warns_without_mutating_draft_or_history(self) -> None:
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

    def test_replacement_cancel_leaves_existing_draft_untouched(self) -> None:
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


if __name__ == "__main__":
    unittest.main()
