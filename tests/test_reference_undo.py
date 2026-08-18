from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from seedvision.persistence import ReferenceRegionBundle


class ReferenceUndoUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        try:
            from PySide6.QtWidgets import QApplication
        except ImportError as error:
            raise unittest.SkipTest(f"PySide6 unavailable: {error}")
        cls.application = QApplication.instance() or QApplication([])

    @staticmethod
    def _write_image(path: Path, colour: str = "#8b927f") -> None:
        from PySide6.QtGui import QColor, QImage

        path.parent.mkdir(parents=True, exist_ok=True)
        image = QImage(96, 72, QImage.Format.Format_RGB888)
        image.fill(QColor(colour))
        if not image.save(str(path)):
            raise RuntimeError(f"Could not write test image {path}.")

    def setUp(self) -> None:
        from seedvision.ui.main_window import MainWindow

        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.first_path = self.root / "images" / "a.png"
        self.second_path = self.root / "images" / "b.png"
        self._write_image(self.first_path)
        self._write_image(self.second_path, "#7f8692")
        self.window = MainWindow(self.root)
        self.addCleanup(self.window.close)
        self.window.resize(960, 700)
        self.window.show()
        self._make_current_image_editable()

    def _make_current_image_editable(self) -> None:
        height, width = self._shape()
        # Select the no-overlay path while no synthetic result is installed;
        # the normal renderer legitimately expects the complete production
        # result contract. Undo tests only need an analysis-presence sentinel.
        self.window.image_view.set_overlay_mode("none")
        result = SimpleNamespace(
            calibration=SimpleNamespace(
                corrected_bgr=np.full((height, width, 3), 128, dtype=np.uint8)
            ),
            layers=SimpleNamespace(
                edge_likelihood=np.zeros((height, width), dtype=np.float32),
                undirected_edge_hue=np.zeros((height, width), dtype=np.float32),
                directed_edge_hue=np.zeros((height, width), dtype=np.float32),
                offset_x=0,
                offset_y=0,
            ),
            crop_offset=(0, 0),
            estimated_seed_diameter_px=24.0,
        )
        self.window.image_view._analysis_result = result
        self.window.image_view._render_analysis = lambda: None
        self.window._sync_background_controls()
        self.application.processEvents()
        self.window.image_view.fit_image()

    def _shape(self) -> tuple[int, int]:
        width, height = self.window.image_view.image_size or (0, 0)
        return height, width

    def _key(self) -> str:
        key = self.window._current_image_key()
        self.assertIsNotNone(key)
        assert key is not None
        return key

    def _reference_mask(self, class_name: str) -> np.ndarray:
        values = self.window.image_view.reference_mask(class_name)
        return (
            np.zeros(self._shape(), dtype=bool)
            if values is None
            else np.asarray(values, dtype=bool)
        )

    def _activate_reference_mode(self) -> None:
        self.window.paint_background_action.setChecked(True)
        self.window._sync_background_controls()
        self.application.processEvents()

    def _activate_instance_mode(self) -> None:
        self.window.annotate_instances_action.setChecked(True)
        self.window._sync_background_controls()
        self.application.processEvents()

    def _commit_reference(
        self,
        class_name: str,
        region: np.ndarray,
        *,
        erase: bool = False,
    ) -> None:
        context = (
            "material"
            if class_name in {"background", "foreground", "other"}
            else "boundary"
        )
        before = self.window._reference_group_state(self._key(), context)
        after = [np.asarray(values).copy() for values in before]
        names = (
            ("background", "foreground", "other")
            if context == "material"
            else ("physical_edge", "non_edge")
        )
        selected = names.index(class_name)
        if erase:
            after[selected][region] = False
        else:
            after[selected][region] = True
            for index, values in enumerate(after):
                if index != selected:
                    values[region] = False
        if context == "material":
            self.window.image_view.set_reference_masks(
                after[0],
                after[1],
                after[2],
                after[2],
                copy=True,
                render=False,
                normalize_material=False,
            )
        else:
            material = self.window._reference_group_state(self._key(), "material")
            self.window.image_view.set_reference_masks(
                material[0],
                material[1],
                material[2],
                material[2],
                physical_edge_mask=after[0],
                non_edge_mask=after[1],
                copy=True,
                render=False,
                normalize_material=False,
            )
        self.window._reference_mask_edited(class_name, after[selected])

    def _commit_instances(self, labels: np.ndarray) -> None:
        self.window.image_view.set_instance_annotations(
            labels, copy=True, render=False
        )
        self.window._instance_annotations_edited(
            self.window.image_view._instance_annotations
        )

    def test_material_exclusivity_multi_undo_returns_to_clean_baseline(self) -> None:
        from PySide6.QtCore import Qt
        from PySide6.QtTest import QTest

        self._activate_reference_mode()
        key = self._key()
        height, width = self._shape()
        first = np.zeros((height, width), dtype=bool)
        first[18:25, 22:31] = True
        second = np.zeros_like(first)
        second[44:50, 62:69] = True

        self.assertFalse(self.window.reference_undo_button.isEnabled())
        self._commit_reference("foreground", first)
        self._commit_reference("background", first)
        self._commit_reference("other", second)

        history = self.window._reference_undo_histories[key]
        self.assertEqual(len(history), 3)
        self.assertEqual(self.window.reference_undo_button.text(), "Undo (3)")
        self.assertTrue(self.window.undo_reference_edit_action.isEnabled())
        self.window.reference_undo_button.click()
        self.assertFalse(
            np.any(self._reference_mask("other")[second])
        )
        self.assertTrue(
            np.all(self._reference_mask("background")[first])
        )

        self.window.activateWindow()
        self.window.image_view.viewport().setFocus()
        QTest.keyClick(
            self.window.image_view.viewport(),
            Qt.Key.Key_Z,
            Qt.KeyboardModifier.ControlModifier,
        )
        self.application.processEvents()
        self.assertTrue(
            np.all(self._reference_mask("foreground")[first])
        )
        self.assertFalse(
            np.any(self._reference_mask("background")[first])
        )

        self.window.undo_reference_edit_action.trigger()
        self.assertNotIn(key, self.window._reference_undo_histories)
        self.assertNotIn(key, self.window._reference_masks_dirty)
        self.assertNotIn(key, self.window._reference_dirty_classes)
        self.assertFalse(self.window.reference_undo_button.isEnabled())
        self.assertFalse(self.window.apply_reference_masks_button.isEnabled())
        self.assertFalse(self.window.revert_reference_masks_button.isEnabled())

    def test_retired_boundary_reference_edits_are_ignored(self) -> None:
        height, width = self._shape()
        region = np.zeros((height, width), dtype=bool)
        region[31:35, 35:57] = True
        key = self._key()

        self.window._reference_mask_edited("physical_edge", region)
        self.window._reference_mask_edited("non_edge", region)

        self.assertNotIn(key, self.window._reference_masks_dirty)
        self.assertNotIn(key, self.window._reference_undo_histories)
        self.assertNotIn(key, self.window._draft_physical_edge_reference_masks)
        self.assertNotIn(key, self.window._draft_non_edge_reference_masks)
        self.assertFalse(np.any(self._reference_mask("physical_edge")))
        self.window._undo_active_reference_edit()
        self.assertNotIn(key, self.window._reference_masks_dirty)

    def test_instance_history_restores_labels_origin_and_clear_commands(self) -> None:
        key = self._key()
        height, width = self._shape()
        applied = np.zeros((height, width), dtype=np.uint16)
        applied[10:17, 12:20] = 1
        applied.flags.writeable = False
        self.window._applied_instance_annotations[key] = applied
        self.window._applied_instance_annotation_origins[key] = (
            "pipeline:procedural_instances"
        )
        self.window.image_view.set_instance_annotations(
            applied, copy=False, render=False
        )
        self._activate_instance_mode()

        added = applied.copy()
        added[38:46, 58:67] = 2
        self._commit_instances(added)
        self.window.instance_id_spin.setValue(2)
        self.window._clear_current_instance_annotation()
        self.assertFalse(np.any(self.window.image_view._instance_annotations == 2))
        self.window._undo_active_reference_edit()
        self.assertTrue(np.any(self.window.image_view._instance_annotations == 2))
        self.assertEqual(
            self.window._draft_instance_annotation_origins[key],
            "pipeline:procedural_instances",
        )

        self.window._clear_all_instance_annotations()
        self.assertFalse(np.any(self.window.image_view._instance_annotations))
        self.assertEqual(self.window._draft_instance_annotation_origins[key], "manual")
        self.window._undo_active_reference_edit()
        self.assertTrue(np.array_equal(self.window.image_view._instance_annotations, added))
        self.assertEqual(
            self.window._draft_instance_annotation_origins[key],
            "pipeline:procedural_instances",
        )

        self.window._undo_active_reference_edit()
        self.assertIs(
            self.window.image_view._instance_annotations,
            self.window._applied_instance_annotations[key],
        )
        self.assertNotIn(key, self.window._draft_instance_annotations)
        self.assertNotIn(key, self.window._instance_annotations_dirty)
        self.assertNotIn(key, self.window._instance_undo_histories)

    def test_apply_and_revert_are_history_barriers(self) -> None:
        self._activate_reference_mode()
        key = self._key()
        height, width = self._shape()
        foreground = np.zeros((height, width), dtype=bool)
        foreground[12:20, 18:28] = True
        self._commit_reference("foreground", foreground)
        analyses: list[bool] = []
        self.window._analyze_current_image = lambda **_kwargs: analyses.append(True)

        self.window._apply_reference_masks()
        self.assertEqual(analyses, [True])
        self.assertNotIn(key, self.window._reference_undo_histories)
        self.assertFalse(self.window._applied_foreground_reference_masks[key].flags.writeable)

        background = np.zeros_like(foreground)
        background[42:49, 55:66] = True
        self._commit_reference("background", background)
        self.assertIn(key, self.window._reference_undo_histories)
        self.window._revert_reference_masks()
        self.assertNotIn(key, self.window._reference_undo_histories)
        self.assertNotIn(key, self.window._reference_masks_dirty)
        self.assertTrue(
            np.array_equal(
                self.window.image_view.reference_mask("foreground"), foreground
            )
        )
        self.assertFalse(
            np.any(self.window.image_view.reference_mask("background"))
        )

    def test_histories_are_isolated_per_image_and_autoload_is_a_barrier(self) -> None:
        self._activate_reference_mode()
        first_key = self._key()
        first_region = np.zeros(self._shape(), dtype=bool)
        first_region[8:14, 10:18] = True
        self._commit_reference("foreground", first_region)

        self.window._open_path(self.second_path)
        self._make_current_image_editable()
        self._activate_reference_mode()
        second_key = self._key()
        second_region = np.zeros(self._shape(), dtype=bool)
        second_region[44:51, 70:78] = True
        self._commit_reference("background", second_region)
        self.assertEqual(len(self.window._reference_undo_histories[first_key]), 1)
        self.assertEqual(len(self.window._reference_undo_histories[second_key]), 1)

        self.window._undo_active_reference_edit()
        self.assertNotIn(second_key, self.window._reference_undo_histories)
        self.assertEqual(len(self.window._reference_undo_histories[first_key]), 1)

        self.window._open_path(self.first_path)
        self._make_current_image_editable()
        self._activate_reference_mode()
        self.window._undo_active_reference_edit()
        self.assertNotIn(first_key, self.window._reference_undo_histories)

        self._commit_reference("foreground", first_region)
        loaded = np.zeros(self._shape(), dtype=bool)
        loaded[30:36, 40:47] = True
        self.window._install_reference_region_bundle(
            first_key,
            ReferenceRegionBundle(shape=self._shape(), background=loaded),
        )
        self.window._sync_background_controls()
        self.assertNotIn(first_key, self.window._reference_undo_histories)
        self.assertNotIn(first_key, self.window._instance_undo_histories)
        self.assertFalse(self.window.reference_undo_button.isEnabled())
        self.assertTrue(
            np.array_equal(
                self.window.image_view.reference_mask("background"), loaded
            )
        )

    def test_ui_queue_is_bounded_and_no_op_does_not_consume_it(self) -> None:
        self._activate_reference_mode()
        key = self._key()
        height, width = self._shape()
        empty = np.zeros((height, width), dtype=bool)
        self._commit_reference("background", empty)
        self.assertNotIn(key, self.window._reference_undo_histories)

        for index in range(self.window.REFERENCE_UNDO_LIMIT + 3):
            point = np.zeros_like(empty)
            point[2 + index // 12, 2 + index % 12] = True
            self._commit_reference("background", point)
        history = self.window._reference_undo_histories[key]
        self.assertEqual(len(history), self.window.REFERENCE_UNDO_LIMIT)
        self.assertEqual(
            self.window.reference_undo_button.text(),
            f"Undo ({self.window.REFERENCE_UNDO_LIMIT})",
        )

        for _ in range(self.window.REFERENCE_UNDO_LIMIT):
            self.window._undo_active_reference_edit()
        remaining = self.window.image_view.reference_mask("background")
        self.assertTrue(remaining[2, 2])
        self.assertTrue(remaining[2, 3])
        self.assertTrue(remaining[2, 4])
        self.assertEqual(int(np.count_nonzero(remaining)), 3)
        self.assertNotIn(key, self.window._reference_undo_histories)
        self.assertIn(key, self.window._reference_masks_dirty)
        self.assertFalse(self.window.reference_undo_button.isEnabled())

    def test_right_drag_and_explicit_eraser_are_single_undoable_strokes(self) -> None:
        from PySide6.QtCore import QPointF, Qt
        from PySide6.QtTest import QTest

        self._activate_reference_mode()
        key = self._key()
        view = self.window.image_view
        view.set_reference_brush_radius(4.0)
        centre = view.mapFromScene(QPointF(48.0, 36.0))

        QTest.mouseClick(view.viewport(), Qt.MouseButton.LeftButton, pos=centre)
        self.assertTrue(self._reference_mask("background")[36, 48])
        QTest.mouseClick(view.viewport(), Qt.MouseButton.RightButton, pos=centre)
        self.assertFalse(self._reference_mask("background")[36, 48])
        self.assertEqual(len(self.window._reference_undo_histories[key]), 2)
        self.window._undo_active_reference_edit()
        self.assertTrue(self._reference_mask("background")[36, 48])

        self.window.reference_eraser_button.setChecked(True)
        QTest.mouseClick(view.viewport(), Qt.MouseButton.LeftButton, pos=centre)
        self.assertFalse(self._reference_mask("background")[36, 48])
        self.window._undo_active_reference_edit()
        self.assertTrue(self._reference_mask("background")[36, 48])

    def test_assisted_instance_commits_are_atomic_and_anchor_is_a_no_op(self) -> None:
        from PySide6.QtCore import QPointF

        from seedvision.annotation import SmartFillRegion

        self._activate_instance_mode()
        key = self._key()
        view = self.window.image_view
        view.set_instance_annotation_tool("edge_trace")
        first = QPointF(14.0, 22.0)
        view._instance_preview_point = QPointF(first)
        view._instance_preview_endpoint = QPointF(first)
        self.assertFalse(view._commit_instance_assisted_preview(first))
        self.assertNotIn(key, self.window._instance_undo_histories)

        second = QPointF(66.0, 22.0)
        view._instance_preview_point = QPointF(second)
        view._instance_preview_endpoint = QPointF(second)
        view._instance_preview_geometry = np.asarray(
            ((14, 22), (30, 22), (48, 22), (66, 22)), dtype=np.float32
        )
        self.assertTrue(view._commit_instance_assisted_preview(second))
        self.assertEqual(len(self.window._instance_undo_histories[key]), 1)
        self.window._undo_active_reference_edit()
        self.assertFalse(np.any(view._instance_annotations))

        view.set_instance_annotation_tool("smart_fill")
        fill = np.ones((8, 9), dtype=bool)
        view._instance_preview_point = QPointF(30.0, 54.0)
        view._instance_preview_region = SmartFillRegion(
            x=26, y=50, mask=fill, added_count=int(fill.sum())
        )
        self.assertTrue(
            view._commit_instance_assisted_preview(QPointF(30.0, 54.0))
        )
        self.assertEqual(len(self.window._instance_undo_histories[key]), 1)
        self.window._undo_active_reference_edit()
        self.assertFalse(np.any(view._instance_annotations))


if __name__ == "__main__":
    unittest.main()
