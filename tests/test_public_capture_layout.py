"""Preview geometry and clearing contracts, independent of source calculations."""

from tests.catalogue_fixtures import CatalogueTestCase

from tests.functional_units import functional_units
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import unittest
from dataclasses import replace
import numpy as np
from PySide6.QtWidgets import QApplication

from seedvision.benchmarking.recipe_catalogue import build_recipe_catalogue
from seedvision.ui.public_reference_capture_dialog import (
    CaptureReply, DisplayArtifact, PublicReferenceCaptureDialog,
)


@functional_units
class PublicCaptureLayoutTests(CatalogueTestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.dialog = PublicReferenceCaptureDialog(build_recipe_catalogue().recipe("plantcv_seed_binary"))
        self.dialog.show()
        image = np.zeros((96, 128, 3), dtype=np.uint8)
        self.dialog._reply = CaptureReply(None, (
            DisplayArtifact("display_fixture", "0" * 64, "display fixture", image, "RGB test image"),
            DisplayArtifact("compact_fixture", "1" * 64, "compact fixture", None, "No raster"),
        ))
        self.dialog.artifact_selector.addItems(["display_fixture", "compact_fixture"])
        self.app.processEvents()

    def tearDown(self):
        self.dialog.close()
        self.app.processEvents()

    def check_test_preview_fits_both_window_sizes_and_keeps_aspect_ratio(self):
        for width, height in ((1440, 900), (1050, 650), (1440, 900)):
            with self.subTest(size=(width, height)):
                self.dialog.resize(width, height)
                self.app.processEvents()
                self.assertEqual((self.dialog.width(), self.dialog.height()), (width, height))
                pixmap = self.dialog.preview.pixmap()
                rectangle = self.dialog.preview.contentsRect()
                self.assertFalse(pixmap.isNull())
                self.assertLessEqual(pixmap.width(), rectangle.width())
                self.assertLessEqual(pixmap.height(), rectangle.height())
                self.assertLessEqual(abs(pixmap.width() - pixmap.height() * 128 / 96), 1.5)

    def check_test_compact_selection_and_invalidation_cannot_restore_stale_image_on_resize(self):
        self.dialog.artifact_selector.setCurrentIndex(1)
        self.dialog.resize(1440, 900)
        self.app.processEvents()
        self.assertTrue(self.dialog.preview.pixmap().isNull())
        self.dialog.artifact_selector.setCurrentIndex(0)
        self.assertFalse(self.dialog.preview.pixmap().isNull())
        self.dialog._invalidate()
        self.dialog.resize(1050, 650)
        self.app.processEvents()
        self.assertTrue(self.dialog.preview.pixmap().isNull())
        self.assertEqual(self.dialog.preview.text(), "No current verified capture")

    def check_test_same_viewport_replacement_and_clear_reload_display_new_pixels(self):
        original = self.dialog.preview.pixmap()
        viewport = self.dialog.preview.contentsRect().size()
        red = np.zeros((96, 128, 3), dtype=np.uint8)
        red[..., 0] = 255
        artifacts = self.dialog._reply.artifacts
        self.dialog._reply = CaptureReply(None, (replace(artifacts[0], image=red), artifacts[1]))
        self.dialog._show_artifact(0)
        self.app.processEvents()
        self.assertEqual(self.dialog.preview.contentsRect().size(), viewport)
        replacement = self.dialog.preview.pixmap()
        self.assertEqual(replacement.toImage().pixelColor(0, 0).getRgb()[:3], (255, 0, 0))
        self.dialog.preview.setPixmap(original)
        self.assertEqual(self.dialog.preview.pixmap().toImage().pixelColor(0, 0).getRgb()[:3], (0, 0, 0))
        self.dialog.preview.clear()
        self.assertTrue(self.dialog.preview.pixmap().isNull())
        self.dialog.preview.setPixmap(replacement)
        self.assertEqual(self.dialog.preview.pixmap().toImage().pixelColor(0, 0).getRgb()[:3], (255, 0, 0))

    def test_workflow_lifecycle(self):
        """2 related PublicCaptureLayoutTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_preview_fits_both_window_sizes_and_keeps_aspect_ratio)
        self.scenario(self.check_test_same_viewport_replacement_and_clear_reload_display_new_pixels)

    def test_validation_and_recovery(self):
        """1 related PublicCaptureLayoutTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_compact_selection_and_invalidation_cannot_restore_stale_image_on_resize)


if __name__ == "__main__":
    unittest.main()
