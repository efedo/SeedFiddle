from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


class ReferenceTextureCollageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        try:
            from PySide6.QtWidgets import QApplication
        except ImportError as error:
            raise unittest.SkipTest(f"PySide6 unavailable: {error}")
        cls.application = QApplication.instance() or QApplication([])

    def test_edge_strip_markers_use_source_pixel_offsets_and_lengths(self) -> None:
        from PySide6.QtCore import QRectF

        from seedvision.ui.image_view import _edge_strip_annotation_lines

        centre, sides = _edge_strip_annotation_lines(
            QRectF(5.0, 3.0, 68.0, 68.0),
            (20, 20),
            normal_offset_px=4.0,
            tangent_half_length_px=6.0,
        )

        self.assertAlmostEqual(centre.x1(), 18.6)
        self.assertAlmostEqual(centre.x2(), 59.4)
        self.assertAlmostEqual(centre.y1(), 37.0)
        self.assertEqual(len(sides), 2)
        self.assertAlmostEqual(sides[0].y1(), 23.4)
        self.assertAlmostEqual(sides[1].y1(), 50.6)
        self.assertTrue(all(line.x1() == centre.x1() for line in sides))
        self.assertTrue(all(line.x2() == centre.x2() for line in sides))

        _centre, clipped_sides = _edge_strip_annotation_lines(
            QRectF(0.0, 0.0, 40.0, 40.0),
            (20, 20),
            normal_offset_px=12.0,
            tangent_half_length_px=30.0,
        )
        self.assertEqual(clipped_sides, ())
        self.assertEqual((_centre.x1(), _centre.x2()), (0.0, 40.0))

    def test_edge_collage_marks_descriptor_lines_and_explains_context(self) -> None:
        from PySide6.QtGui import QColor, QImage

        from seedvision.ui.image_view import ImageView
        from seedvision.visualization import (
            ReferenceTextureProfile,
            ReferenceTexturePrototype,
        )

        patch = np.full((20, 20, 3), 48, np.uint8)
        patch.flags.writeable = False
        profile = ReferenceTextureProfile(
            prototypes=(
                ReferenceTexturePrototype(
                    class_name="physical_edge",
                    patch_bgr=patch,
                    weight=1.0,
                    sample_count=32,
                    centre_xy=(50.0, 50.0),
                    tangent_degrees=25.0,
                ),
            ),
            class_sample_counts=(("physical_edge", 32),),
            edge_strip_normal_offset_px=4.0,
            edge_strip_tangent_half_length_px=6.0,
            patch_size_px=20,
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "source.png"
            source = QImage(80, 60, QImage.Format.Format_RGB32)
            source.fill(QColor("white"))
            self.assertTrue(source.save(str(path)))
            result = SimpleNamespace(
                calibration=SimpleNamespace(
                    corrected_bgr=np.full((60, 80, 3), 210, np.uint8)
                ),
                layers=SimpleNamespace(reference_texture_profile=profile),
            )
            view = ImageView()
            succeeded, error = view.load_image(path)
            self.assertTrue(succeeded, error)
            view.set_overlay_mode("reference_texture_prototypes")
            view.show_analysis(result)

            tooltip = view._image_item.toolTip()
            self.assertIn("not matched as image patches", tooltip)
            self.assertIn("both possible side assignments", tooltip)
            image = view._image_item.pixmap().toImage().convertToFormat(
                QImage.Format.Format_RGBA8888
            )
            pixels = np.frombuffer(
                image.constBits(), dtype=np.uint8, count=image.sizeInBytes()
            ).reshape(image.height(), image.bytesPerLine())
            rgb = pixels[:, : image.width() * 4].reshape(
                image.height(), image.width(), 4
            )[:, :, :3]
            centre_colour = np.array((255, 240, 106), np.uint8)
            side_colour = np.array((56, 221, 255), np.uint8)
            signed_rgb = rgb.astype(np.int16)
            self.assertLess(
                int(
                    np.min(
                        np.max(
                            np.abs(signed_rgb - centre_colour.astype(np.int16)),
                            axis=2,
                        )
                    )
                ),
                40,
            )
            self.assertLess(
                int(
                    np.min(
                        np.max(
                            np.abs(signed_rgb - side_colour.astype(np.int16)),
                            axis=2,
                        )
                    )
                ),
                40,
            )

            view.set_overlay_mode("raw_image")
            self.application.processEvents()
            self.assertEqual(view._image_item.toolTip(), "")
            view.close()


if __name__ == "__main__":
    unittest.main()
