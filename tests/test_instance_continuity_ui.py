from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


class InstanceContinuityUiTests(unittest.TestCase):
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

        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        image_path = self.root / "images" / "seeds.png"
        image_path.parent.mkdir(parents=True)
        image = QImage(80, 60, QImage.Format.Format_RGB32)
        image.fill(QColor("white"))
        self.assertTrue(image.save(str(image_path)))
        self.window = MainWindow(self.root)
        self.addCleanup(self.window.close)
        self.window.image_view._analysis_result = SimpleNamespace()

    def _install_draft(self, labels: np.ndarray) -> str:
        key = self.window._current_image_key()
        self.assertIsNotNone(key)
        assert key is not None
        self.window._set_instance_draft_state(key, labels, "manual")
        self.window.image_view.set_instance_annotations(
            self.window._draft_instance_annotations.get(key),
            copy=False,
            render=False,
        )
        self.window._sync_background_controls()
        self.application.processEvents()
        return key

    def test_selected_seed_warning_is_exact_and_does_not_block_apply(self) -> None:
        labels = np.zeros((60, 80), dtype=np.uint16)
        labels[4:9, 5:10] = 7
        labels[30:35, 45:50] = 7
        self.window.instance_id_spin.setValue(7)

        key = self._install_draft(labels)

        warning = self.window.instance_continuity_warning_label
        banner = self.window.image_view.instance_continuity_warning_banner
        self.assertFalse(warning.isHidden())
        self.assertEqual(warning.text(), "⚠ Seed 7 has 2 disconnected areas.")
        self.assertIn("8-neighbour", warning.toolTip())
        self.assertFalse(banner.isHidden())
        self.assertEqual(banner.text(), warning.text())
        self.assertEqual(banner.toolTip(), warning.toolTip())
        self.assertIn("background-color: rgba(176, 24, 24", banner.styleSheet())
        from PySide6.QtCore import Qt

        self.assertTrue(
            banner.testAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        )
        self.window.image_view.resize(420, 260)
        self.application.processEvents()
        self.assertGreaterEqual(
            banner.geometry().top(), self.window.image_view.zoom_controls.height()
        )
        self.assertLessEqual(
            banner.geometry().right(), self.window.image_view.width()
        )
        self.assertTrue(self.window.apply_instance_annotations_button.isEnabled())
        self.assertIn(key, self.window._instance_annotations_dirty)

        connected = labels.copy()
        for row, column in zip(range(8, 32), range(9, 47), strict=False):
            connected[row, column] = 7
        # A one-pixel interpolation above can skip columns; use a continuous
        # OpenCV line to establish an unambiguous 8-connected bridge.
        import cv2

        cv2.line(connected, (8, 7), (47, 32), 7, thickness=1, lineType=cv2.LINE_8)
        self._install_draft(connected)
        self.assertTrue(warning.isHidden())
        self.assertTrue(banner.isHidden())
        self.assertEqual(banner.text(), "")

    def test_global_warning_and_identity_cache_update_for_a_new_draft(self) -> None:
        labels = np.zeros((60, 80), dtype=np.uint16)
        labels[3:6, 3:6] = 2
        labels[20:23, 20:23] = 2
        labels[40:44, 50:54] = 5
        key = self._install_draft(labels)
        self.window.instance_id_spin.setValue(5)
        self.window._sync_background_controls()

        warning = self.window.instance_continuity_warning_label
        banner = self.window.image_view.instance_continuity_warning_banner
        self.assertIn("1 seed ID has disconnected areas: 2", warning.text())
        self.assertEqual(banner.text(), warning.text())
        self.window.instance_id_spin.setValue(2)
        self.assertEqual(banner.text(), "⚠ Seed 2 has 2 disconnected areas.")
        self.window.instance_id_spin.setValue(5)
        self.assertEqual(banner.text(), warning.text())
        first = self.window._instance_continuity_summary(key, labels)
        second = self.window._instance_continuity_summary(key, labels)
        self.assertIs(first, second)

        replacement = labels.copy()
        replacement[3:21, 4] = 2
        replacement[20, 4:21] = 2
        self.window._set_instance_draft_state(key, replacement, "manual")
        third = self.window._instance_continuity_summary(key, replacement)
        self.assertIsNot(third, first)
        self.assertEqual(third.component_count(2), 1)


if __name__ == "__main__":
    unittest.main()
