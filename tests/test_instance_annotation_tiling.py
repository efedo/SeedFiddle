from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np


class InstanceAnnotationTilingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        try:
            from PySide6.QtWidgets import QApplication
        except ImportError as error:
            raise unittest.SkipTest(f"PySide6 unavailable: {error}")
        cls.application = QApplication.instance() or QApplication([])

    @staticmethod
    def _make_item(labels: np.ndarray, *, selected_identifier: int | None = None):
        from seedvision.ui.image_view import ImageView, _InstanceAnnotationTileItem

        return _InstanceAnnotationTileItem(
            labels,
            colour_for_identifier=ImageView.instance_colour,
            selected_identifier=selected_identifier,
        )

    @staticmethod
    def _assert_pixel_matches_identifier(image, x: int, y: int, identifier: int) -> None:
        from seedvision.ui.image_view import ImageView

        actual = image.pixelColor(int(x), int(y))
        expected = ImageView.instance_colour(identifier)
        if actual.alpha() != 156:
            raise AssertionError(
                f"Expected full-resolution annotation alpha 156 at {(x, y)}, "
                f"got {actual.alpha()}."
            )
        if actual.getRgb()[:3] != expected.getRgb()[:3]:
            raise AssertionError(
                f"Expected ID {identifier} colour {expected.getRgb()[:3]} at "
                f"{(x, y)}, got {actual.getRgb()[:3]}."
            )

    def test_6240_by_4160_tiles_preserve_exact_one_pixel_marks_and_ridges(self) -> None:
        labels = np.zeros((4160, 6240), dtype=np.uint16)
        isolated = (
            (0, 0, 1),
            (511, 400, 2),
            (512, 400, 3),
            (6239, 4159, 4),
        )
        for x, y, identifier in isolated:
            labels[y, x] = identifier
        # One-pixel ridges immediately beside horizontal and vertical tile
        # boundaries catch both resampling and off-by-one tile coverage.
        labels[1000:1010, 3071] = 5
        labels[1024, 3000:3010] = 6
        # Keep the neighbouring tile materialized while leaving the pixel next
        # to the vertical ridge transparent.
        labels[900, 3100] = 7

        item = self._make_item(labels)

        bounds = item.boundingRect()
        self.assertEqual((bounds.x(), bounds.y()), (0.0, 0.0))
        self.assertEqual((bounds.width(), bounds.height()), (6240.0, 4160.0))
        self.assertTrue(item.transform().isIdentity())
        for x, y, identifier in isolated:
            tile_x = x // item.tile_size
            tile_y = y // item.tile_size
            tile = item.render_tile(tile_x, tile_y)
            self._assert_pixel_matches_identifier(
                tile,
                x - tile_x * item.tile_size,
                y - tile_y * item.tile_size,
                identifier,
            )

        vertical = item.render_tile(5, 1)
        for y in range(1000, 1010):
            local_y = y - item.tile_size
            self.assertEqual(vertical.pixelColor(510, local_y).alpha(), 0)
            self._assert_pixel_matches_identifier(vertical, 511, local_y, 5)
        right_neighbour = item.render_tile(6, 1)
        for y in range(1000, 1010):
            self.assertEqual(
                right_neighbour.pixelColor(0, y - item.tile_size).alpha(), 0
            )

        horizontal = item.render_tile(5, 2)
        for x in range(3000, 3010):
            local_x = x - 5 * item.tile_size
            self._assert_pixel_matches_identifier(horizontal, local_x, 0, 6)
            self.assertEqual(horizontal.pixelColor(local_x, 1).alpha(), 0)

        edge_tile = item.render_tile(12, 8)
        self.assertEqual((edge_tile.width(), edge_tile.height()), (96, 64))

    def test_cache_is_lru_bounded_independently_of_image_extent(self) -> None:
        from PySide6.QtGui import QImage

        height, width = 4160, 6240
        labels = np.zeros((height, width), dtype=np.uint16)
        tiles_x = (width + 511) // 512
        tiles_y = (height + 511) // 512
        for tile_y in range(tiles_y):
            for tile_x in range(tiles_x):
                labels[tile_y * 512, tile_x * 512] = 1
        item = self._make_item(labels)

        for tile_y in range(tiles_y):
            for tile_x in range(tiles_x):
                image = item.render_tile(tile_x, tile_y)
                self.assertFalse(image.isNull())

        self.assertEqual(item.cache_size, tiles_x * tiles_y)
        self.assertLessEqual(item.cache_size, item.cache_capacity)
        self.assertEqual(
            set(item.cache_keys),
            {
                (tile_x, tile_y)
                for tile_y in range(tiles_y)
                for tile_x in range(tiles_x)
            },
        )
        self.assertLessEqual(item.cache_bytes, item.cache_byte_capacity)
        self.assertEqual(item.cache_bytes, labels.size)
        self.assertEqual(
            item.cache_bytes,
            sum(
                int(image.sizeInBytes())
                for image in item._tile_cache.values()
                if not image.isNull()
            ),
        )
        self.assertTrue(
            all(
                image.format() == QImage.Format.Format_Indexed8
                for image in item._tile_cache.values()
            )
        )

    def test_item_paint_keeps_one_source_pixel_as_a_hard_four_by_four_zoom_block(self) -> None:
        from PySide6.QtCore import QRectF
        from PySide6.QtGui import QImage, QPainter
        from PySide6.QtWidgets import QStyleOptionGraphicsItem

        labels = np.zeros((4, 4), dtype=np.uint16)
        labels[1, 1] = 9
        item = self._make_item(labels)
        canvas = QImage(16, 16, QImage.Format.Format_RGBA8888)
        canvas.fill(0)
        painter = QPainter(canvas)
        painter.scale(4.0, 4.0)
        option = QStyleOptionGraphicsItem()
        option.exposedRect = QRectF(0.0, 0.0, 4.0, 4.0)
        item.paint(painter, option)
        painter.end()

        for y in range(16):
            for x in range(16):
                expected_opaque = 4 <= x < 8 and 4 <= y < 8
                alpha = canvas.pixelColor(x, y).alpha()
                self.assertEqual(alpha, 156 if expected_opaque else 0)

    def test_selected_filter_and_in_place_edit_invalidate_cached_pixels(self) -> None:
        from seedvision.ui.image_view import ImageView, _InstanceAnnotationTileItem

        labels = np.zeros((96, 128), dtype=np.uint16)
        labels[20, 30] = 1
        labels[20, 31] = 2
        labels[21, 31] = 2
        view = ImageView()
        self.addCleanup(view.close)
        view.set_instance_annotations(labels, render=False)
        view.set_active_instance_id(1)
        view.set_show_selected_instance_only(True)

        item = view._instance_annotation_overlay_item
        self.assertIsInstance(item, _InstanceAnnotationTileItem)
        assert item is not None
        first = item.render_tile(0, 0)
        self._assert_pixel_matches_identifier(first, 30, 20, 1)
        self.assertEqual(first.pixelColor(31, 20).alpha(), 0)
        generation = item.generation

        view.set_active_instance_id(2)
        self.assertIs(view._instance_annotation_overlay_item, item)
        self.assertGreater(item.generation, generation)
        self.assertEqual(item.cache_size, 0)
        second = item.render_tile(0, 0)
        self.assertEqual(second.pixelColor(30, 20).alpha(), 0)
        self._assert_pixel_matches_identifier(second, 31, 20, 2)

        # Normal brush commits mutate the active copy-on-write label buffer in
        # place before the overlay refresh. The previous cached tile must not
        # survive that refresh.
        assert view._instance_annotations is not None
        view._instance_annotations[20, 31] = 0
        previous_generation = item.generation
        view._refresh_instance_annotation_overlay()
        self.assertGreater(item.generation, previous_generation)
        self.assertEqual(item.cache_size, 0)
        erased = item.render_tile(0, 0)
        self.assertEqual(erased.pixelColor(31, 20).alpha(), 0)

    def test_repeated_refresh_visibility_and_clear_leave_no_stale_scene_items(self) -> None:
        from seedvision.ui.image_view import ImageView, _InstanceAnnotationTileItem

        view = ImageView()
        self.addCleanup(view.close)
        labels = np.zeros((80, 100), dtype=np.uint16)
        labels[10:20, 15:25] = 1
        view.set_instance_annotations(labels, render=False)
        view._refresh_instance_annotation_overlay()
        item = view._instance_annotation_overlay_item
        self.assertIsInstance(item, _InstanceAnnotationTileItem)
        assert item is not None
        item.render_tile(0, 0)

        for _ in range(5):
            view._refresh_instance_annotation_overlay()
            tiled_items = [
                candidate
                for candidate in view.scene().items()
                if isinstance(candidate, _InstanceAnnotationTileItem)
            ]
            self.assertEqual(tiled_items, [item])
            self.assertEqual(view._overlay_items.count(item), 1)

        view.set_instance_annotations_visible(False)
        self.assertIsNone(view._instance_annotation_overlay_item)
        self.assertIsNone(item.scene())
        self.assertFalse(
            any(
                isinstance(candidate, _InstanceAnnotationTileItem)
                for candidate in view.scene().items()
            )
        )

        view.set_instance_annotations_visible(True)
        replacement = view._instance_annotation_overlay_item
        self.assertIsInstance(replacement, _InstanceAnnotationTileItem)
        self.assertIsNot(replacement, item)
        view._clear_overlay_items()
        self.assertIsNone(view._instance_annotation_overlay_item)
        assert replacement is not None
        self.assertIsNone(replacement.scene())

    def test_new_tile_item_honours_current_overlay_opacity(self) -> None:
        from seedvision.ui.image_view import ImageView

        view = ImageView()
        self.addCleanup(view.close)
        view.set_overlay_opacity(0.37)
        labels = np.zeros((32, 32), dtype=np.uint16)
        labels[4, 5] = 1
        view.set_instance_annotations(labels, render=False)
        view._refresh_instance_annotation_overlay()

        item = view._instance_annotation_overlay_item
        self.assertIsNotNone(item)
        assert item is not None
        self.assertAlmostEqual(item.opacity(), 0.37)

    def test_actual_size_pan_and_high_zoom_materialize_exposed_tiles(self) -> None:
        from PySide6.QtGui import QColor, QImage

        from seedvision.ui.image_view import ImageView

        with tempfile.TemporaryDirectory() as temporary_directory:
            image_path = Path(temporary_directory) / "viewport.png"
            source = QImage(2048, 1536, QImage.Format.Format_RGB32)
            source.fill(QColor("white"))
            self.assertTrue(source.save(str(image_path)))

            view = ImageView()
            self.addCleanup(view.close)
            view.resize(420, 340)
            succeeded, error = view.load_image(image_path)
            self.assertTrue(succeeded, error)
            labels = np.zeros((1536, 2048), dtype=np.uint16)
            for tile_y in range(3):
                for tile_x in range(4):
                    labels[tile_y * 512 + 256, tile_x * 512 + 256] = 1
            view.set_instance_annotations(labels, render=False)
            view._refresh_instance_annotation_overlay()
            item = view._instance_annotation_overlay_item
            self.assertIsNotNone(item)
            assert item is not None
            view.show()
            self.application.processEvents()

            view.actual_size()
            view.centerOn(256.0, 256.0)
            item.invalidate_tiles()
            view.viewport().repaint()
            self.application.processEvents()
            left_coordinates = {key[-2:] for key in item.cache_keys}
            self.assertIn((0, 0), left_coordinates)

            view.centerOn(1792.0, 1280.0)
            view.viewport().repaint()
            self.application.processEvents()
            right_coordinates = {key[-2:] for key in item.cache_keys}
            self.assertIn((3, 2), right_coordinates)
            self.assertGreater(len(right_coordinates - left_coordinates), 0)

            view.resetTransform()
            view.scale(4.0, 4.0)
            view.centerOn(1280.0, 768.0)
            item.invalidate_tiles()
            view.viewport().repaint()
            self.application.processEvents()
            self.assertIn((2, 1), {key[-2:] for key in item.cache_keys})
            self.assertLessEqual(item.cache_size, item.cache_capacity)


if __name__ == "__main__":
    unittest.main()
