"""Native image-list actions, thumbnails, file preservation and edit ownership."""
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Timer, get_ident
from types import SimpleNamespace
import os
import shutil
import time
import unittest
from unittest.mock import patch

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtCore import QEvent, QPoint, QRect, Qt
from PySide6.QtGui import (
    QColor, QFontMetrics, QHelpEvent, QImage, QImageIOHandler, QImageWriter, QPalette,
)
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox

from seedvision.persistence import (
    BiologicalContext, ProjectAnalysisStore, ProjectImageSpec, ReferenceRegionBundle,
    ReferenceRegionStore, analysis_settings_profile_from_graph,
)
from seedvision.export.project_runner import run_project
from seedvision.pipeline import build_default_pipeline
from seedvision.segmentation.baseline import PipelineAnalysisCache
from seedvision.ui import image_list
from seedvision.ui.image_list import (
    IMAGE_OPEN_ROLE, IMAGE_SPECIES_ROLE, THUMBNAIL_SIZE, ImageThumbnailLoader, name_lines,
    read_thumbnail, row_text, species_text,
)
from seedvision.ui.canvas_controls import COMPACT_ICON_SIZE
from seedvision.ui.main_window import MainWindow, UNKNOWN_SPECIES, _path_identity
from tests.functional_units import functional_units
from tests.review_fixtures import dispose_window, isolated_window
from tests import test_project_analysis_ui


class _StopBatch(Exception):
    """Ends a batch run once its analysis request is built."""


def _write_image(path, **options):
    return test_project_analysis_ui.ProjectAnalysisUiTests._write_image(path, **options)


def _wait_until(condition, timeout=10.0):
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("Timed out waiting for background thumbnails")
        QTest.qWait(5)


@contextmanager
def _held_decodes():
    """Hold every background decode until the gate opens, recording its thread."""

    gate, threads = Event(), []

    def held_read(path, box):
        threads.append(get_ident())
        gate.wait(10)
        return read_thumbnail(path, box)

    with patch.object(image_list, "read_thumbnail", held_read):
        try:
            yield gate, threads
        finally:
            gate.set()


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
                # A named, icon-only plus in the pipeline editor's flat colour, whose
                # middle stays filled where its two arms cross.
                button = window.open_images_button
                self.assertIs(button.defaultAction(), window.open_action)
                self.assertEqual(button.toolButtonStyle(), Qt.ToolButtonStyle.ToolButtonIconOnly)
                self.assertEqual(button.iconSize(), COMPACT_ICON_SIZE)
                self.assertLessEqual(button.width(), 30)
                self.assertEqual(button.accessibleName(), "Add images")
                self.assertTrue(button.toolTip().startswith("Add one or more images"))
                plus = button.icon().pixmap(COMPACT_ICON_SIZE).toImage()
                middle = plus.pixelColor(plus.width() // 2, plus.height() // 2)
                self.assertEqual((middle.name(), middle.alpha()), ("#f2f6fa", 255))
                self.assertEqual(plus.pixelColor(1, 1).alpha(), 0)
                self.assertFalse(window.open_action.isIconVisibleInMenu())
                with patch("seedvision.ui.main_window.QFileDialog.getOpenFileNames", return_value=([], "")) as choose:
                    button.click()
                choose.assert_called_once()
                self.assertGreater(window.image_list.height() / window.height(), .27)
                self.assertLess(window.image_list.height() / window.height(), .39)
                key = _path_identity(removed)
                window._draft_foreground_reference_masks[key] = foreground
                window._optimization_targets[key] = {"area": 12}
                window._image_review_states[key] = "reviewed"
                window._source_fingerprints[key] = "removed fingerprint"
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

    def test_removal_protects_edits_and_busy_work_then_clears_the_selection(self):
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
                foreground = np.zeros((36, 48), dtype=bool)
                foreground[2:8, 5:11] = True
                window._draft_foreground_reference_masks[key] = foreground
                window._reference_masks_dirty.add(key)
                window._reference_transforms[key] = np.eye(3)
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
                self.assertIn(second_key, window._unsaved_reference_sidecars)
                sidecar = ReferenceRegionStore(root).path_for(first)
                self.assertTrue(sidecar.is_file())
                saved_sidecar = sidecar.read_bytes()
                window._unsaved_manual_centre_sidecars.add(second_key)
                with patch.object(QMessageBox, "warning", return_value=QMessageBox.StandardButton.Save), \
                     patch.object(window, "_retry_unsaved_project_sidecars", return_value=False):
                    self.assertFalse(window._remove_image(second))
                with patch.object(QMessageBox, "warning", return_value=QMessageBox.StandardButton.Discard):
                    self.assertTrue(window._remove_image(second))
                self.assertEqual(window.image_list.count(), 0)
                self.assertIsNone(window.image_view.image_path)
                self.assertIsNone(window._current_image_key())
                self.assertFalse(window.analyze_action.isEnabled())
                self.assertFalse(window._unsaved_manual_centre_sidecars)
                self.assertTrue(first.is_file())
                self.assertTrue(second.is_file())
                self.assertEqual(sidecar.read_bytes(), saved_sidecar)
            finally:
                dispose_window(window)
                self.application.processEvents()


@functional_units
class ImageThumbnailLoaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.loader = ImageThumbnailLoader()
        self.addCleanup(self.loader.stop)

    def _delivered(self, path):
        _wait_until(lambda: str(path) not in self.loader._wanted)
        return self.loader._entries.get(str(path))

    def check_reader_fits_the_box_and_reports_what_it_cannot_read(self):
        box = THUMBNAIL_SIZE
        landscape, error = read_thumbnail(
            str(_write_image(self.root / "\u00e9 seeds.png", size=(650, 420))), box)
        self.assertEqual(error, "")
        self.assertEqual((landscape.width(), landscape.height()), (48, 31))
        small, _ = read_thumbnail(str(_write_image(self.root / "small.png", size=(10, 8))), box)
        self.assertEqual((small.width(), small.height()), (10, 8))

        # A deep source must not cost more than an ordinary thumbnail to hold.
        deep = QImage(96, 64, QImage.Format.Format_RGBX64)
        deep.fill(QColor("#336699"))
        self.assertTrue(deep.save(str(self.root / "deep.png")))
        thumbnail, _ = read_thumbnail(str(self.root / "deep.png"), box)
        self.assertEqual(thumbnail.depth(), 32)

        rotated = self.root / "rotated.jpg"
        writer = QImageWriter(str(rotated))
        writer.setTransformation(QImageIOHandler.Transformation.TransformationRotate90)
        source = QImage(240, 120, QImage.Format.Format_RGB32)
        source.fill(QColor("#336699"))
        self.assertTrue(writer.write(source))
        upright, _ = read_thumbnail(str(rotated), box)
        self.assertEqual((upright.width(), upright.height()), (16, 32))

        corrupt = self.root / "corrupt.jpg"
        corrupt.write_bytes(b"not an image")
        for unreadable in (corrupt, self.root / "missing.jpg"):
            image, error = read_thumbnail(str(unreadable), box)
            self.assertTrue(image.isNull())
            self.assertTrue(error)

    def check_decoding_stays_off_the_gui_thread_and_queues_each_image_once(self):
        path = _write_image(self.root / "a.png")
        changed = []
        self.loader.changed.connect(lambda: changed.append(True))
        with _held_decodes() as (gate, threads):
            self.assertIsNone(self.loader.get(str(path)))
            self.assertIsNone(self.loader.get(str(path)))
            _wait_until(lambda: threads)
            self.assertNotEqual(threads[0], get_ident())
            self.assertIsNone(self.loader.get(str(path)))
            gate.set()
            self.assertFalse(self._delivered(path).image.isNull())
        self.assertEqual(len(threads), 1)
        self.assertEqual(changed, [True])

        broken = self.root / "broken.jpg"
        broken.write_bytes(b"not an image")
        self.loader.get(str(broken))
        self.assertTrue(self._delivered(broken).image.isNull())
        self.assertTrue(self.loader.error(str(broken)))
        self.assertEqual(self.loader.error(str(path)), "")

    def check_abandoned_work_is_dropped_and_the_cache_stays_bounded(self):
        first, second, third = (
            _write_image(self.root / name) for name in ("a.png", "b.png", "c.png"))
        with _held_decodes() as (gate, threads):
            self.loader.get(str(first))
            _wait_until(lambda: threads)
            self.loader.forget(str(first))
            self.loader.get(str(second))
            self.loader.clear()
            gate.set()
            _wait_until(lambda: self.loader._pool.activeThreadCount() == 0)
            QTest.qWait(20)
        self.assertEqual(len(threads), 1)
        self.assertFalse(self.loader._entries)
        self.assertFalse(self.loader._wanted)

        with patch.object(ImageThumbnailLoader, "CACHE_LIMIT", 2):
            for path in (first, second):
                self.loader.get(str(path))
                self._delivered(path)
            self.assertIsNotNone(self.loader.get(str(first)))
            self.loader.get(str(third))
            self._delivered(third)
        self.assertEqual(tuple(self.loader._entries), (str(first), str(third)))

    def check_a_changed_source_file_is_decoded_again(self):
        path = _write_image(self.root / "a.png", size=(48, 32), colour="#d02020")
        self.loader.get(str(path))
        before = self._delivered(path)
        self.loader.refresh(str(path))
        self.assertIs(self.loader._entries[str(path)], before)
        _write_image(path, size=(96, 64), colour="#2020d0")
        changed = []
        self.loader.changed.connect(lambda: changed.append(True))
        self.loader.refresh(str(path))
        self.assertEqual(changed, [True])
        self.assertIsNone(self.loader.get(str(path)))
        after = self._delivered(path)
        self.assertEqual(after.image.pixelColor(5, 5), QColor("#2020d0"))

    def check_stop_waits_for_the_decode_in_flight_without_blocking_it(self):
        path = _write_image(self.root / "a.png")
        with _held_decodes() as (gate, threads):
            self.loader.get(str(path))
            _wait_until(lambda: threads)
            release = Timer(.05, gate.set)
            release.start()
            started = time.monotonic()
            self.loader.stop()
            release.join()
            self.assertLess(time.monotonic() - started, ImageThumbnailLoader.STOP_WAIT_MS / 1000)
            self.assertEqual(self.loader._pool.activeThreadCount(), 0)
        QTest.qWait(20)
        self.assertFalse(self.loader._entries)

    def check_names_fit_two_lines_and_keep_their_distinguishing_end(self):
        metrics = QFontMetrics(QApplication.font())
        self.assertEqual(name_lines("IMG_9403.JPG", metrics, 500), ("IMG_9403.JPG",))
        width = metrics.horizontalAdvance("IMG_9403.J")
        self.assertEqual(name_lines("IMG_9403.JPG", metrics, width), ("IMG_9403.", "JPG"))
        long_name = "2026-10-12_soybean_tray_A_replicate_3_IMG_9403_corrected.jpg"
        width = 2 * metrics.horizontalAdvance("corrected.jpg")
        top, bottom = name_lines(long_name, metrics, width)
        self.assertLessEqual(metrics.horizontalAdvance(top), width)
        self.assertLessEqual(metrics.horizontalAdvance(bottom), width)
        self.assertTrue(long_name.startswith(top))
        self.assertTrue(bottom.endswith(".jpg"))
        self.assertEqual(len(name_lines("abc", metrics, 1)), 2)

    def check_species_text_shortens_the_genus_before_the_epithet(self):
        metrics = QFontMetrics(QApplication.font())
        self.assertEqual(species_text("Lupinus mexicanus", metrics, 500), "Lupinus mexicanus")
        width = metrics.horizontalAdvance("L. polyphyllus") + 1
        self.assertLess(width, metrics.horizontalAdvance("Lupinus mexicanus"))
        self.assertEqual(species_text("Lupinus mexicanus", metrics, width), "L. mexicanus")
        self.assertEqual(species_text("Lupinus polyphyllus", metrics, width), "L. polyphyllus")
        for name in ("Lupinus polyphyllus", "Soybean", UNKNOWN_SPECIES):
            tight = species_text(name, metrics, 30)
            self.assertLessEqual(metrics.horizontalAdvance(tight), 30)
            self.assertNotEqual(tight, name)

    def check_species_is_a_column_when_wide_and_a_second_line_when_narrow(self):
        metrics = QFontMetrics(QApplication.font())
        species = "Lupinus polyphyllus"
        area = QRect(60, 100, metrics.horizontalAdvance(f"{species}  IMG_0000c.JPG") + 40, 38)

        wide = [row_text(area, name, species, metrics) for name in ("a.png", "IMG_9403.JPG")]
        self.assertEqual(wide[0].species_rect, wide[1].species_rect)
        self.assertEqual(wide[0].species_rect.right(), area.right())
        self.assertEqual(wide[0].species, species)
        for layout in wide:
            self.assertFalse(layout.name_rect.intersects(layout.species_rect))
            self.assertTrue(area.contains(layout.species_rect))
        self.assertEqual(wide[1].name, ("IMG_9403.JPG",))

        narrow_area = QRect(60, 100, metrics.horizontalAdvance("IMG_9403.JPG") + 2, 38)
        narrow = row_text(narrow_area, "IMG_9403.JPG", species, metrics)
        self.assertEqual(narrow.name, ("IMG_9403.JPG",))
        self.assertEqual(narrow.species_rect.top(), narrow.name_rect.bottom() + 1)
        self.assertEqual(narrow.species_rect.left(), narrow.name_rect.left())
        self.assertTrue(narrow_area.contains(narrow.name_rect))
        self.assertTrue(narrow_area.contains(narrow.species_rect))
        self.assertLessEqual(metrics.horizontalAdvance(narrow.species), narrow_area.width())

    def test_background_decoding_and_cache(self):
        self.scenario(self.check_reader_fits_the_box_and_reports_what_it_cannot_read)
        self.scenario(self.check_decoding_stays_off_the_gui_thread_and_queues_each_image_once)
        self.scenario(self.check_abandoned_work_is_dropped_and_the_cache_stays_bounded)
        self.scenario(self.check_a_changed_source_file_is_decoded_again)
        self.scenario(self.check_stop_waits_for_the_decode_in_flight_without_blocking_it)
        self.scenario(self.check_names_fit_two_lines_and_keep_their_distinguishing_end)
        self.scenario(self.check_species_text_shortens_the_genus_before_the_epithet)
        self.scenario(self.check_species_is_a_column_when_wide_and_a_second_line_when_narrow)


@functional_units
class ImageListRowTests(unittest.TestCase):
    COLOURS = ("#d02020", "#20a020", "#2020d0", "#d0a020")

    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.paths = [
            _write_image(self.root / "images" / f"{name}.png", size=(480, 320), colour=colour)
            for name, colour in zip("abcd", self.COLOURS)
        ]
        release = patch.object(MainWindow, "_release_unused_cuda_blocks")
        release.start()
        self.addCleanup(release.stop)
        self.window = isolated_window(self.root)
        self.addCleanup(dispose_window, self.window)
        self.window.resize(1440, 900)
        self.window.show()
        self.list = self.window.image_list
        self.loader = self.window._image_thumbnails
        self.delegate = self.window._image_list_delegate

    def _settle(self):
        self.application.processEvents()
        _wait_until(lambda: not self.loader._wanted)
        self.application.processEvents()

    def _row(self, row):
        return self.list.visualItemRect(self.list.item(row))

    def _picture(self):
        self.application.processEvents()
        return self.list.viewport().grab().toImage()

    def _tile_colour(self, row):
        return self._picture().pixelColor(self.delegate.tile_rect(self._row(row)).center())

    def _margin_colour(self, row):
        rect = self._row(row)
        return self._picture().pixelColor(rect.left() + 2, rect.center().y())

    def _accent(self):
        return self.list.palette().color(QPalette.ColorGroup.Active, QPalette.ColorRole.Highlight)

    def _open_rows(self):
        return [row for row in range(self.list.count())
                if self.list.item(row).data(IMAGE_OPEN_ROLE)]

    def _species(self, window=None):
        rows = (window or self.window).image_list
        return [rows.item(row).data(IMAGE_SPECIES_ROLE) for row in range(rows.count())]

    def _changed_area(self, before, after):
        """The smallest rectangle holding every pixel that differs between two grabs."""

        area = QRect()
        for y in range(before.height()):
            for x in range(before.width()):
                if before.pixel(x, y) != after.pixel(x, y):
                    area = area.united(QRect(x, y, 1, 1))
        return area

    def _species_change_area(self, row, rect):
        """Where a row's picture changes when it names another species."""

        item, before = self.list.item(row), self._picture()
        species = item.data(IMAGE_SPECIES_ROLE)
        # Short enough never to be shortened, because the offscreen platform may draw
        # every letter alike and only the length of the text then shows.
        item.setData(IMAGE_SPECIES_ROLE, "Pea")
        changed = self._changed_area(before, self._picture()).intersected(rect)
        item.setData(IMAGE_SPECIES_ROLE, species)
        self.assertEqual(self._picture(), before)
        return changed

    def _tooltip(self, row):
        viewport = self.list.viewport()
        position = self._row(row).center()
        with patch.object(image_list.QToolTip, "showText") as tooltip:
            QApplication.sendEvent(viewport, QHelpEvent(
                QEvent.Type.ToolTip, position, viewport.mapToGlobal(position)))
        return tooltip.call_args.args[1]

    def check_every_row_shows_its_own_image_replacing_the_blank_document(self):
        self._settle()
        self.assertEqual(self.list.count(), 4)
        for row, colour in enumerate(self.COLOURS):
            self.assertEqual(self._tile_colour(row), QColor(colour))
        idle = self.list.item(1)
        self.assertTrue(idle.icon().isNull())
        self.assertEqual(idle.data(Qt.ItemDataRole.AccessibleDescriptionRole), "Analysis required")

    def check_analysis_status_stays_visible_on_the_thumbnail(self):
        self._settle()
        tile = self.delegate.tile_rect(self._row(1))
        spot = tile.bottomRight() - QPoint(6, 6)
        self.assertEqual(self._picture().pixelColor(spot), QColor(self.COLOURS[1]))
        key = _path_identity(self.paths[1])
        self.window._active_tasks[key] = SimpleNamespace()
        self.addCleanup(self.window._active_tasks.clear)
        self.window._update_analysis_availability()
        item = self.list.item(1)
        self.assertFalse(item.icon().isNull())
        self.assertEqual(item.data(Qt.ItemDataRole.AccessibleDescriptionRole), "Computing")
        self.assertIn("Computing", item.toolTip())
        self.assertNotEqual(self._picture().pixelColor(spot), QColor(self.COLOURS[1]))
        self.assertEqual(self._tile_colour(1), QColor(self.COLOURS[1]))
        self.window._active_tasks.clear()
        self.window._unsaved_manual_centre_sidecars.add(key)
        self.addCleanup(self.window._unsaved_manual_centre_sidecars.clear)
        self.window._update_analysis_availability()
        self.assertIn("Save required", item.toolTip())
        self.assertFalse(item.icon().isNull())

    def check_placeholder_and_unreadable_files_do_not_hold_up_or_break_the_list(self):
        self._settle()
        placeholder = self.list.palette().color(QPalette.ColorRole.Midlight)
        self.loader.clear()
        with _held_decodes() as (gate, threads):
            self.list.viewport().update()
            _wait_until(lambda: threads)
            self.assertEqual(self._tile_colour(1).rgb(), placeholder.rgb())
            self.assertNotIn(get_ident(), threads)
            # The analysis pool stays free, so nothing reports the window as busy.
            self.assertEqual(self.window._thread_pool.activeThreadCount(), 0)
            self.window._open_list_item(self.list.item(2))
            self.assertEqual(self.window.image_view.image_path, self.paths[2])
            with patch.object(QMessageBox, "information") as busy:
                self.assertTrue(self.window._remove_image(self.paths[3]))
            busy.assert_not_called()
            self.assertEqual(self.list.count(), 3)
            gate.set()
            self._settle()
        self.assertEqual(self._tile_colour(1), QColor(self.COLOURS[1]))

        broken = self.root / "images" / "broken.jpg"
        broken.write_bytes(b"not an image")
        self.window._add_image(broken, open_now=False, mark_project_dirty=False)
        self.list.viewport().update()
        self._settle()
        self.assertEqual(self.list.count(), 4)
        self.assertTrue(self.loader.error(str(broken.resolve())))
        tile = self.delegate.tile_rect(self._row(3))
        picture = self._picture()
        mid = self.list.palette().color(QPalette.ColorRole.Mid)
        crossed = sum(
            picture.pixelColor(x, y).rgb() == mid.rgb()
            for x in range(tile.left() + 2, tile.right() - 1)
            for y in range(tile.top() + 2, tile.bottom() - 1)
        )
        self.assertGreater(crossed, 20)
        self.assertEqual(self._tile_colour(1), QColor(self.COLOURS[1]))
        viewport = self.list.viewport()
        position = self._row(3).center()
        with patch.object(image_list.QToolTip, "showText") as tooltip:
            QApplication.sendEvent(viewport, QHelpEvent(
                QEvent.Type.ToolTip, position, viewport.mapToGlobal(position)))
        self.assertIn("No thumbnail:", tooltip.call_args.args[1])
        self.assertIn("broken.jpg", tooltip.call_args.args[1])

    def check_only_painted_rows_are_decoded(self):
        for index in range(40):
            self.window._add_image(
                _write_image(self.root / "more" / f"{index:02}.png"),
                open_now=False, mark_project_dirty=False,
            )
        self.list.viewport().update()
        self._settle()
        painted = len(self.loader._entries)
        self.assertGreater(painted, 3)
        self.assertLess(painted, 15)
        self.list.scrollToBottom()
        self._settle()
        self.assertGreater(len(self.loader._entries), painted)
        self.assertIn(str((self.root / "more" / "39.png").resolve()), self.loader._entries)

    def check_highlight_follows_the_open_image_however_it_changes(self):
        accent = self._accent()
        self.assertEqual(self._open_rows(), [0])
        self.assertEqual(self._margin_colour(0), accent)
        self.assertNotEqual(self._margin_colour(1), accent)
        # The alternating base colours remain, and do not affect the highlight.
        self.assertNotEqual(self._margin_colour(1), self._margin_colour(2))

        self.window._open_list_item(self.list.item(1))
        self.assertEqual(self._open_rows(), [1])
        self.assertEqual(self._margin_colour(1), accent)
        self.assertNotEqual(self._margin_colour(0), accent)
        self.assertIn("open in the viewer",
                      self.list.item(1).data(Qt.ItemDataRole.AccessibleDescriptionRole))

        # It does not depend on the list having keyboard focus.
        self.window.species_combo.setFocus()
        self.assertEqual(self._margin_colour(1), accent)

        # A click or arrow key moves only the list cursor: the viewer has not changed.
        viewport = self.list.viewport()
        QTest.mouseClick(viewport, Qt.MouseButton.LeftButton, pos=self._row(2).center())
        self.assertIs(self.list.currentItem(), self.list.item(2))
        self.assertEqual(self.window.image_view.image_path, self.paths[1])
        self.assertEqual(self._open_rows(), [1])
        cursor = self._margin_colour(2)
        self.assertNotEqual(cursor, accent)
        self.assertNotEqual(cursor, self._margin_colour(0))
        self.assertEqual(self._picture().pixelColor(self._row(2).topLeft()), accent)
        self.list.setFocus()
        QTest.keyClick(self.list, Qt.Key.Key_Down)
        self.assertIs(self.list.currentItem(), self.list.item(3))
        self.assertEqual(self._open_rows(), [1])

        # Enter and a real double click both open it.
        QTest.keyClick(self.list, Qt.Key.Key_Return)
        self.assertEqual(self.window.image_view.image_path, self.paths[3])
        self.assertEqual(self._open_rows(), [3])
        self.assertEqual(self._margin_colour(3), accent)
        position = self._row(0).center()
        QTest.mouseClick(viewport, Qt.MouseButton.LeftButton, pos=position)
        QTest.mouseDClick(viewport, Qt.MouseButton.LeftButton, pos=position)
        QTest.mouseRelease(viewport, Qt.MouseButton.LeftButton, pos=position)
        self.assertEqual(self._open_rows(), [0])

        # So do the Image selector node, adding an image and removing the open one.
        self.window._select_pipeline_image(2)
        self.assertEqual(self._open_rows(), [2])
        self.assertEqual(self._margin_colour(2), accent)
        added = _write_image(self.root / "images" / "e.png")
        self.window._add_image(added, open_now=True)
        self.assertEqual(self._open_rows(), [4])
        self.assertTrue(self.window._remove_image(added))
        self.assertEqual(self.list.count(), 4)
        self.assertEqual(self._open_rows(), [3])
        self.assertEqual(self._margin_colour(3), accent)
        for path in self.paths:
            self.assertTrue(self.window._remove_image(path))
        self.assertEqual(self.list.count(), 0)
        self.assertIsNone(self.window.image_view.image_path)

    def check_rows_fit_the_narrow_panel(self):
        long_name = "2026-10-12_soybean_tray_A_replicate_3_IMG_9403_corrected_copy.png"
        self.window._add_image(
            _write_image(self.root / "images" / long_name), open_now=False,
            mark_project_dirty=False,
        )
        self.list.viewport().update()
        self._settle()
        last = self.list.count() - 1
        rect = self._row(last)
        self.assertEqual(rect.width(), self.list.viewport().width())
        self.assertFalse(self.list.horizontalScrollBar().isVisible())
        self.assertTrue(rect.contains(self.delegate.remove_rect(rect)))
        self.assertTrue(rect.contains(self.delegate.tile_rect(rect)))
        self.assertFalse(
            self.delegate.tile_rect(rect).intersects(self.delegate.remove_rect(rect)))
        self.assertGreaterEqual(rect.height(), THUMBNAIL_SIZE.height() + 2)
        # The Images list keeps its share of the window, so this is what it shows.
        self.assertGreaterEqual(self.list.viewport().height() / rect.height(), 6.5)

    def check_every_row_always_names_a_species_starting_at_unknown(self):
        self.assertEqual(UNKNOWN_SPECIES, "Unknown")
        self.assertEqual(self.window.species_combo.currentText(), UNKNOWN_SPECIES)
        self.assertEqual(self._species(), [UNKNOWN_SPECIES] * 4)
        self._settle()
        self.assertIn("Species: Unknown", self._tooltip(1))
        self.assertFalse(self._species_change_area(1, self._row(1)).isNull())

        # A row that was given no species still reads as unknown, not blank.
        before = self._picture()
        self.list.item(1).setData(IMAGE_SPECIES_ROLE, None)
        self.assertEqual(self._picture(), before)
        self.assertIn("Species: Unknown", self._tooltip(1))
        self.assertFalse(self.window._project_dirty)

    def _install_traits(self):
        config = self.root / "config"
        config.mkdir()
        shutil.copy(Path(__file__).resolve().parents[1] / "config" / "traits.json", config)

    def _open_mixed_species_project(self):
        self._install_traits()
        contexts = (BiologicalContext("soybean"), BiologicalContext("lupinus_mexicanus"),
                    BiologicalContext("unknown"), None)
        project = ProjectAnalysisStore(self.root).capture_and_save(
            analysis_settings=analysis_settings_profile_from_graph(build_default_pipeline()),
            images=tuple(ProjectImageSpec(path, (320, 480), biological_context=context)
                         for path, context in zip(self.paths, contexts)),
            species="Lupinus mutabilis", selected_image=self.paths[0],
            destination=self.root / "project.seedfiddle-project.json")
        return self._reopen(project), project

    def _reopen(self, project):
        window = isolated_window(self.root)
        self.addCleanup(dispose_window, window)
        with patch.object(window, "_confirm_project_replacement", return_value=True):
            self.assertTrue(window._open_project(project))
        return window

    def check_each_row_names_the_species_recorded_for_its_own_image(self):
        window, _ = self._open_mixed_species_project()
        self.assertEqual(self._species(window), [
            "Soybean", "Lupinus mexicanus", UNKNOWN_SPECIES, "Lupinus mutabilis"])

        # A species missing from the catalogue is still named rather than hidden.
        window._project_image_biological_contexts[_path_identity(self.paths[3])] = (
            BiologicalContext("vicia_faba"))
        window._update_analysis_availability()
        self.assertEqual(self._species(window)[3], "vicia_faba")

    def check_the_selector_shows_and_sets_the_open_images_species(self):
        window, project = self._open_mixed_species_project()
        combo = window.species_combo

        # It follows the open image, and looking at an image changes nothing.
        for path, species in zip(self.paths, (
                "Soybean", "Lupinus mexicanus", UNKNOWN_SPECIES, "Lupinus mutabilis")):
            window._open_path(path, mark_project_dirty=False)
            self.assertEqual(combo.currentText(), species)
        self.assertEqual(self._species(window), [
            "Soybean", "Lupinus mexicanus", UNKNOWN_SPECIES, "Lupinus mutabilis"])
        self.assertEqual(window.pipeline.node("metadata").status_detail, "Lupinus mutabilis")

        # Changing it changes the open image and no other.
        window._open_path(self.paths[1], mark_project_dirty=False)
        combo.setCurrentText("Lupinus polyphyllus")
        self.assertEqual(self._species(window), [
            "Soybean", "Lupinus polyphyllus", UNKNOWN_SPECIES, "Lupinus mutabilis"])
        self.assertEqual(window._effective_biological_context().species_id, "lupinus_polyphyllus")

        # It does not carry over to the next image, and analysis uses that image's own.
        window._open_path(self.paths[0], mark_project_dirty=False)
        self.assertEqual(combo.currentText(), "Soybean")
        self.assertEqual(window._current_seed_trait_species_id(), "soybean")
        window._open_path(self.paths[1], mark_project_dirty=False)
        self.assertEqual(combo.currentText(), "Lupinus polyphyllus")

        # New images follow the project, and giving one image a species leaves them there.
        added = _write_image(self.root / "images" / "e.png")
        window._add_image(added, open_now=False, mark_project_dirty=False)
        self.assertEqual(self._species(window)[-1], "Lupinus mutabilis")
        window._open_path(self.paths[3], mark_project_dirty=False)
        combo.setCurrentText("Soybean")
        self.assertEqual(self._species(window)[3:], ["Soybean", "Lupinus mutabilis"])

        # One image can be set back to unknown on its own.
        window._open_path(self.paths[0], mark_project_dirty=False)
        combo.setCurrentIndex(0)
        expected = [UNKNOWN_SPECIES, "Lupinus polyphyllus", UNKNOWN_SPECIES, "Soybean", "Lupinus mutabilis"]
        self.assertEqual(self._species(window), expected)

        # Each image's own survives a save; the one without any keeps following the project.
        self.assertTrue(window._save_project())
        loaded = ProjectAnalysisStore(self.root).load(project)
        self.assertEqual(loaded.document.species, "Lupinus mutabilis")
        self.assertEqual(loaded.document.biological_context.species_id, "lupinus_mutabilis")
        self.assertEqual(
            [None if image.record.biological_context is None
             else image.record.biological_context.species_id for image in loaded.images],
            ["unknown", "lupinus_polyphyllus", "unknown", "soybean", None])
        reopened = self._reopen(project)
        self.assertEqual(self._species(reopened), expected)
        self.assertEqual(reopened.species_combo.currentText(), UNKNOWN_SPECIES)

        # Removing the open image moves to the next one's species, then the project's.
        reopened._open_path(self.paths[1], mark_project_dirty=False)
        reopened._remove_image(self.paths[1])
        self.assertEqual(reopened.species_combo.currentText(), UNKNOWN_SPECIES)
        for path in (self.paths[0], self.paths[2], self.paths[3], added):
            reopened._remove_image(path)
        self.assertIsNone(reopened._current_image_key())
        self.assertEqual(reopened.species_combo.currentText(), "Lupinus mutabilis")

        # With no image open it sets the project's, which the next new image follows.
        reopened.species_combo.setCurrentText("Lupinus polyphyllus")
        reopened._add_image(added, open_now=False, mark_project_dirty=False)
        self.assertEqual(self._species(reopened), ["Lupinus polyphyllus"])

    def check_the_first_species_chosen_also_covers_images_without_one(self):
        self._install_traits()
        window = isolated_window(self.root)
        self.addCleanup(dispose_window, window)
        for path in self.paths:
            window._add_image(path, open_now=False, mark_project_dirty=False)
        window._open_path(self.paths[0], mark_project_dirty=False)
        self.assertEqual(self._species(window), [UNKNOWN_SPECIES] * 4)
        follower = _path_identity(self.paths[2])
        window._analysis_caches[follower] = PipelineAnalysisCache()

        window.species_combo.setCurrentText("Soybean")
        self.assertEqual(self._species(window), ["Soybean"] * 4)
        self.assertEqual(window._project_species, "Soybean")
        self.assertTrue(window._cache_dirty_nodes[follower])

        # From then on a choice stays with its image.
        window._cache_dirty_nodes.pop(follower)
        window._open_path(self.paths[1], mark_project_dirty=False)
        self.assertEqual(window.species_combo.currentText(), "Soybean")
        window.species_combo.setCurrentText("Lupinus mexicanus")
        self.assertEqual(self._species(window),
                         ["Soybean", "Lupinus mexicanus", "Soybean", "Soybean"])
        self.assertEqual(window._project_species, "Soybean")
        self.assertNotIn(follower, window._cache_dirty_nodes)

    def check_a_library_build_records_its_species_on_included_images_without_one(self):
        self._install_traits()
        window = isolated_window(self.root)
        self.addCleanup(dispose_window, window)
        for path in self.paths:
            window._add_image(path, open_now=False, mark_project_dirty=False)
        window._open_path(self.paths[0], mark_project_dirty=False)
        window._open_species_library_manager()
        dialog = window._species_library_dialog
        self.addCleanup(dialog.close)

        # A new project has no species yet, so its images are listed under any.
        dialog.species_combo.setCurrentIndex(dialog.species_combo.findData("soybean"))
        self.assertEqual(dialog.source_list.count(), 4)
        self.assertTrue(dialog.source_list.item(0).text().endswith(
            "species not set, no saved annotations"))
        self.assertIn("records the checked ones as Soybean", dialog.unset_species_label.text())
        self.assertFalse(dialog.other_species_label.isVisibleTo(dialog))

        # Building from two of them records Soybean on those two only.
        with patch("seedvision.ui.main_window._SpeciesLibraryBuildTask") as task_type, \
                patch.object(window._thread_pool, "start") as start:
            window._start_species_library_build(
                self.paths[:2], "1", None, (), BiologicalContext("soybean"), None)
        self.addCleanup(window._species_library_build_progress.close)
        start.assert_called_once()
        self.assertEqual(self._species(window),
                         ["Soybean", "Soybean", UNKNOWN_SPECIES, UNKNOWN_SPECIES])
        self.assertEqual(window._project_species, UNKNOWN_SPECIES)
        self.assertEqual(window.species_combo.currentText(), "Soybean")
        self.assertEqual(window.pipeline.node("metadata").status_detail, "Soybean")
        self.assertTrue(window._project_dirty)
        contexts = task_type.call_args.kwargs["source_contexts"]
        self.assertEqual(
            [contexts[_path_identity(path)][0].species_id for path in self.paths],
            ["soybean", "soybean", "unknown", "unknown"])
        # The open image is now Soybean, so the published library can be used here.
        self.assertEqual(dialog.project_species_id, "soybean")
        window._species_library_build_task = None

    def check_batch_analyses_each_image_as_its_own_species(self):
        self._install_traits()
        # Of these only Lupinus mutabilis has coat patterns, and the project is not it.
        for index, (context, expected, species_id) in enumerate((
                (BiologicalContext("lupinus_mutabilis"), "Lupinus mutabilis", "lupinus_mutabilis"),
                (None, "Lupinus mexicanus", "lupinus_mexicanus"),
                (BiologicalContext("unknown"), "unknown", "unknown"))):
            project = ProjectAnalysisStore(self.root).capture_and_save(
                analysis_settings=analysis_settings_profile_from_graph(build_default_pipeline()),
                images=(ProjectImageSpec(self.paths[index], (320, 480), biological_context=context),),
                species="Lupinus mexicanus", biological_context=BiologicalContext("lupinus_mexicanus"),
                selected_image=self.paths[index],
                destination=self.root / f"batch-{index}.seedfiddle-project.json")
            with patch("seedvision.export.project_runner.run_native_analysis",
                       side_effect=_StopBatch) as runner, self.assertRaises(_StopBatch):
                run_project(self.root, project, self.root / f"batch-{index}")
            request = runner.call_args.kwargs
            self.assertEqual(request["species"], expected)
            self.assertEqual(request["biological_context"].species_id, species_id)
            self.assertEqual(bool(request["seed_trait_coat_patterns"]), index == 0)

    def check_species_fits_the_narrow_panel_and_is_a_column_in_a_wide_one(self):
        window, panel = self.window, self.window.images_panel_splitter.parentWidget()
        self.window.main_splitter.setSizes((panel.minimumWidth(), 1000, 250))
        self._settle()
        self.assertEqual(panel.width(), panel.minimumWidth())
        self.assertFalse(self.list.horizontalScrollBar().isVisible())
        heading = window.open_images_button.parentWidget()
        self.assertTrue(heading.rect().contains(window.open_images_button.geometry()))

        metrics = self.list.fontMetrics()
        rect = self._row(1)
        area = self.delegate.text_rect(rect)
        narrow = row_text(area, "b.png", "Lupinus polyphyllus", metrics)
        self.assertEqual(narrow.name, ("b.png",))
        self.assertGreaterEqual(narrow.species_rect.top(), narrow.name_rect.bottom())
        self.assertTrue(rect.contains(narrow.species_rect))
        for target in (self.delegate.tile_rect(rect), self.delegate.remove_rect(rect)):
            self.assertFalse(area.intersects(target))
        changed = self._species_change_area(1, rect)
        self.assertFalse(changed.isNull())
        self.assertFalse(changed.intersects(self.delegate.tile_rect(rect)))
        self.assertFalse(changed.intersects(self.delegate.remove_rect(rect)))

        # The test platform's font can be wider than the panel's limit allows for.
        panel.setMaximumWidth(1000)
        wide_width = metrics.horizontalAdvance("Lupinus polyphyllus  IMG_0000c.JPG") + 160
        self.window.main_splitter.setSizes((wide_width, 1440 - wide_width - 250, 250))
        self._settle()
        rect = self._row(1)
        wide = row_text(self.delegate.text_rect(rect), "b.png", UNKNOWN_SPECIES, metrics)
        self.assertGreater(wide.species_rect.left(), wide.name_rect.right())
        changed = self._species_change_area(1, rect)
        # A short name stays put, so only the species column changes.
        self.assertTrue(wide.species_rect.contains(changed))
        self.assertFalse(changed.isNull())

    def check_thumbnails_reach_neither_the_project_nor_the_disk(self):
        before = {path for path in self.root.rglob("*") if path.is_file()}
        self.assertFalse(self.window._project_dirty)
        self._settle()
        self.assertTrue(self.loader._entries)
        self.assertFalse(self.window._project_dirty)
        self.assertEqual({path for path in self.root.rglob("*") if path.is_file()}, before)
        project = self.root / "project.seedfiddle-project.json"
        self.assertTrue(self.window._write_project(project))
        self.assertNotIn("thumbnail", project.read_text(encoding="utf-8").lower())

    def check_closing_finishes_the_work_only_when_the_window_really_closes(self):
        self._settle()
        self.window._set_project_dirty()
        self.loader.clear()
        self.list.viewport().update()
        self.application.processEvents()
        with patch.object(QMessageBox, "warning", return_value=QMessageBox.StandardButton.Cancel):
            self.window.close()
        self.assertTrue(self.window.isVisible())
        self._settle()
        self.assertTrue(self.loader._entries)

        self.window._set_project_dirty(False)
        self.loader.clear()
        with _held_decodes() as (gate, threads):
            self.list.viewport().update()
            _wait_until(lambda: threads)
            release = Timer(.05, gate.set)
            release.start()
            dispose_window(self.window)
            release.join()
            self.assertFalse(self.window.isVisible())
            self.assertEqual(self.loader._pool.activeThreadCount(), 0)
            self.assertFalse(self.loader._wanted)
        self.application.processEvents()
        self.assertFalse(self.loader._entries)

    def test_rows_show_thumbnails_status_and_the_open_image(self):
        self.scenario(self.check_every_row_shows_its_own_image_replacing_the_blank_document)
        self.scenario(self.check_analysis_status_stays_visible_on_the_thumbnail)
        self.scenario(self.check_placeholder_and_unreadable_files_do_not_hold_up_or_break_the_list)
        self.scenario(self.check_only_painted_rows_are_decoded)
        self.scenario(self.check_highlight_follows_the_open_image_however_it_changes)
        self.scenario(self.check_rows_fit_the_narrow_panel)
        self.scenario(self.check_every_row_always_names_a_species_starting_at_unknown)
        self.scenario(self.check_each_row_names_the_species_recorded_for_its_own_image)
        self.scenario(self.check_the_selector_shows_and_sets_the_open_images_species)
        self.scenario(self.check_the_first_species_chosen_also_covers_images_without_one)
        self.scenario(
            self.check_a_library_build_records_its_species_on_included_images_without_one)
        self.scenario(self.check_batch_analyses_each_image_as_its_own_species)
        self.scenario(self.check_species_fits_the_narrow_panel_and_is_a_column_in_a_wide_one)
        self.scenario(self.check_thumbnails_reach_neither_the_project_nor_the_disk)
        self.scenario(self.check_closing_finishes_the_work_only_when_the_window_really_closes)
