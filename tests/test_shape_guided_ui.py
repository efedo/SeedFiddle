from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np


class ShapeGuidedImageViewTests(unittest.TestCase):
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

        from seedvision.annotation import ShapeGuidedFillOptions
        from seedvision.ui.image_view import ImageView

        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "rotated-seed.png"
        qimage = QImage(190, 170, QImage.Format.Format_RGB32)
        qimage.fill(QColor("white"))
        self.assertTrue(qimage.save(str(self.path)))
        image = np.full((170, 190, 3), (220, 224, 228), np.uint8)
        seed = np.zeros((170, 190), np.uint8)
        cv2.ellipse(seed, (98, 82), (39, 21), 37, 0, 360, 1, -1)
        image[seed > 0] = (66, 101, 146)
        edge = np.zeros((170, 190), np.uint8)
        cv2.ellipse(edge, (98, 82), (39, 21), 37, 0, 360, 255, 2)
        layers = SimpleNamespace(
            edge_ridges=edge,
            offset_x=0,
            offset_y=0,
        )
        result = SimpleNamespace(
            layers=layers,
            crop_offset=(0, 0),
            estimated_seed_diameter_px=78.0,
            calibration=SimpleNamespace(corrected_bgr=image),
        )
        self.view = ImageView()
        self.addCleanup(self.view.close)
        succeeded, error = self.view.load_image(self.path)
        self.assertTrue(succeeded, error)
        self.view.show_analysis(result, render=False)
        labels = np.zeros((170, 190), np.uint16)
        labels[77:81, 114:118] = 9
        self.view.set_instance_annotations(labels, render=False)
        self.view.set_active_instance_id(3)
        self.view.set_instance_annotation_editing(True)
        self.view.set_instance_annotation_tool("shape_guided_fill")
        self.view.set_shape_guided_fill_options(
            ShapeGuidedFillOptions(),
            edge_source="ridges",
        )

    def test_preview_shows_prior_refined_boundary_and_commits_cached_mask(self) -> None:
        from PySide6.QtCore import QPointF

        edits: list[np.ndarray] = []
        self.view.instance_annotations_edited.connect(edits.append)
        point = QPointF(95.0, 84.0)

        self.view._update_instance_assisted_preview(point)

        region = self.view._instance_shape_guided_region
        self.assertIsNotNone(region)
        assert region is not None
        self.assertTrue(region.accepted, region.reason)
        self.assertIsNotNone(region.prior_polygon)
        self.assertIsNotNone(region.boundary_polygon)
        self.assertGreaterEqual(len(self.view._instance_preview_items), 3)
        cached_region = region

        self.assertTrue(self.view._commit_instance_assisted_preview(point))
        self.assertIs(cached_region, region)
        self.assertTrue(edits)
        self.assertEqual(int(edits[-1][82, 98]), 3)
        self.assertTrue(np.all(edits[-1][77:81, 114:118] == 9))

    def test_empty_evidence_refuses_without_changing_labels(self) -> None:
        from PySide6.QtCore import QPointF

        result = self.view._analysis_result
        result.layers.edge_ridges = np.zeros((170, 190), np.uint8)
        self.view.show_analysis(SimpleNamespace(**vars(result)), render=False)
        before = self.view.instance_annotations()
        point = QPointF(95.0, 84.0)

        self.view._update_instance_assisted_preview(point)

        region = self.view._instance_shape_guided_region
        self.assertIsNotNone(region)
        assert region is not None
        self.assertFalse(region.accepted)
        self.assertIn("No edge evidence", region.reason)
        self.assertFalse(self.view._commit_instance_assisted_preview(point))
        self.assertTrue(np.array_equal(self.view.instance_annotations(), before))

    def test_mouse_wheel_changes_visible_shape_preference_without_zooming(self) -> None:
        from PySide6.QtCore import QPoint, QPointF, Qt
        from PySide6.QtGui import QWheelEvent

        changed: list[float] = []
        self.view.shape_fill_size_preference_changed.connect(changed.append)
        before_zoom = self.view.transform().m11()
        event = QWheelEvent(
            QPointF(95.0, 84.0),
            QPointF(95.0, 84.0),
            QPoint(),
            QPoint(0, 120),
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
            Qt.ScrollPhase.NoScrollPhase,
            False,
        )

        self.view.wheelEvent(event)

        self.assertTrue(event.isAccepted())
        self.assertEqual(changed, [1.05])
        self.assertAlmostEqual(
            self.view._shape_guided_fill_options.preferred_scale, 1.05
        )
        self.assertAlmostEqual(self.view.transform().m11(), before_zoom)

    def test_retired_plain_shape_snap_is_not_an_image_view_tool(self) -> None:
        self.assertFalse(hasattr(self.view, "_shape_snap_options"))
        with self.assertRaisesRegex(ValueError, "Unknown instance annotation tool"):
            self.view.set_instance_annotation_tool("shape_snap")


if __name__ == "__main__":
    unittest.main()
