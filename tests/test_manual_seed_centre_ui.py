from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


class ManualSeedCentreUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        try:
            from PySide6.QtWidgets import QApplication
        except ImportError as error:
            raise unittest.SkipTest(f"PySide6 unavailable: {error}")
        cls.application = QApplication.instance() or QApplication([])

    @staticmethod
    def _write_image(path: Path, *, size: tuple[int, int] = (100, 100)) -> None:
        from PySide6.QtGui import QColor, QImage

        path.parent.mkdir(parents=True, exist_ok=True)
        image = QImage(size[0], size[1], QImage.Format.Format_RGB32)
        image.fill(QColor("#61563d"))
        if not image.save(str(path)):
            raise RuntimeError(f"Could not write test image {path}.")

    @staticmethod
    def _procedural_result(
        centres_xy: tuple[tuple[float, float], ...],
        sources: tuple[int, ...],
        *,
        affine_matrix: np.ndarray | None = None,
    ) -> SimpleNamespace:
        procedural = SimpleNamespace(
            marker_centres_xy=np.asarray(centres_xy, dtype=np.float32).reshape(-1, 2),
            marker_sources=np.asarray(sources, dtype=np.uint8),
            # Deliberately different watershed-region centroids: hit testing
            # and editing must use marker metadata, never these display centres.
            centres_xy=(
                np.asarray(centres_xy, dtype=np.float32).reshape(-1, 2) + 19.0
            ),
            rejected_manual_centres_xy=np.empty((0, 2), dtype=np.float32),
            rejected_manual_centre_reasons=(),
            working_scale=1.0,
        )
        return SimpleNamespace(
            procedural_instances=procedural,
            crop_offset=(0, 0),
            calibration=SimpleNamespace(
                affine_matrix=(
                    np.eye(3, dtype=np.float64)
                    if affine_matrix is None
                    else np.asarray(affine_matrix, dtype=np.float64)
                )
            ),
        )

    def test_qt_add_manual_drag_auto_drag_and_right_delete_use_marker_semantics(self) -> None:
        from PySide6.QtCore import QPointF, Qt
        from PySide6.QtTest import QTest

        from seedvision.ui.image_view import ImageView

        with tempfile.TemporaryDirectory() as temporary:
            image_path = Path(temporary) / "image.png"
            self._write_image(image_path)
            view = ImageView()
            view.resize(420, 360)
            succeeded, error = view.load_image(image_path)
            self.assertTrue(succeeded, error)
            view.show()
            self.application.processEvents()
            view.fit_image()
            edits: list[tuple[np.ndarray, str, str]] = []
            view.manual_seed_centres_edited.connect(
                lambda values, mode, label: edits.append(
                    (np.asarray(values).copy(), mode, label)
                )
            )

            with patch.object(view, "_render_analysis"):
                # Empty-space click adds a manual override without changing
                # Augment mode.
                view._analysis_result = self._procedural_result(((25.0, 25.0),), (0,))
                view.set_manual_seed_centres(None, mode="augment", render=False)
                view.set_manual_seed_centre_editing(True)
                add = view.mapFromScene(QPointF(70.0, 60.0))
                QTest.mouseClick(
                    view.viewport(), Qt.MouseButton.LeftButton, pos=add
                )
                np.testing.assert_allclose(edits[-1][0], ((70.0, 60.0),), atol=0.6)
                self.assertEqual(edits[-1][1], "augment")

                # Dragging a manual marker preserves its current mode.
                view._analysis_result = self._procedural_result(
                    ((25.0, 25.0), (70.0, 60.0)), (0, 1)
                )
                view.set_manual_seed_centres(
                    np.asarray(((70.0, 60.0),)), mode="augment", render=False
                )
                start = view.mapFromScene(QPointF(70.0, 60.0))
                end = view.mapFromScene(QPointF(80.0, 72.0))
                QTest.mousePress(
                    view.viewport(), Qt.MouseButton.LeftButton, pos=start
                )
                QTest.mouseMove(view.viewport(), end, delay=5)
                QTest.mouseRelease(
                    view.viewport(), Qt.MouseButton.LeftButton, pos=end
                )
                np.testing.assert_allclose(edits[-1][0], ((80.0, 72.0),), atol=0.8)
                self.assertEqual(edits[-1][1], "augment")

                # Dragging an automatic marker seeds Replace with every
                # surviving non-annotation marker, then moves only the hit one.
                view._analysis_result = self._procedural_result(
                    ((25.0, 25.0), (80.0, 72.0)), (0, 1)
                )
                view.set_manual_seed_centres(
                    np.asarray(((80.0, 72.0),)), mode="augment", render=False
                )
                start = view.mapFromScene(QPointF(25.0, 25.0))
                end = view.mapFromScene(QPointF(38.0, 31.0))
                QTest.mousePress(
                    view.viewport(), Qt.MouseButton.LeftButton, pos=start
                )
                QTest.mouseMove(view.viewport(), end, delay=5)
                QTest.mouseRelease(
                    view.viewport(), Qt.MouseButton.LeftButton, pos=end
                )
                self.assertEqual(edits[-1][1], "replace_automatic")
                np.testing.assert_allclose(
                    edits[-1][0], ((38.0, 31.0), (80.0, 72.0)), atol=0.8
                )

                # Right-deleting an automatic marker has the same safe mode
                # conversion and retains the other real marker.
                view._analysis_result = self._procedural_result(
                    ((25.0, 25.0), (80.0, 72.0)), (0, 1)
                )
                view.set_manual_seed_centres(
                    np.asarray(((80.0, 72.0),)), mode="augment", render=False
                )
                automatic = view.mapFromScene(QPointF(25.0, 25.0))
                QTest.mouseClick(
                    view.viewport(), Qt.MouseButton.RightButton, pos=automatic
                )
                self.assertEqual(edits[-1][1], "replace_automatic")
                np.testing.assert_allclose(edits[-1][0], ((80.0, 72.0),), atol=0.6)

            view.close()

    def test_completed_edit_inverse_projects_autosaves_and_invalidates_only_dependents(self) -> None:
        from seedvision.persistence import ManualSeedCentreStore
        from seedvision.ui.main_window import MainWindow

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "images" / "capture.png"
            self._write_image(image_path, size=(80, 60))
            window = MainWindow(root)
            key = window._current_image_key()
            self.assertIsNotNone(key)
            assert key is not None
            transform = np.asarray(
                ((2.0, 0.0, 5.0), (0.0, 1.5, 7.0), (0.0, 0.0, 1.0)),
                dtype=np.float64,
            )
            result = self._procedural_result(
                ((20.0, 20.0),), (0,), affine_matrix=transform
            )
            window._analyses[key] = result
            window.image_view._analysis_result = result
            expected_affected = {
                "procedural_instances",
                *window.pipeline.downstream("procedural_instances", recursive=True),
            }
            revision = window.pipeline.revision

            with (
                patch.object(window.image_view, "_render_analysis"),
                patch.object(window.pipeline, "invalidate", wraps=window.pipeline.invalidate) as invalidate,
                patch.object(window, "_analyze_current_image") as analyze,
            ):
                window._manual_seed_centres_edited(
                    np.asarray(((25.0, 22.0),), dtype=np.float64),
                    "augment",
                    "added manual seed centre",
                )

            state = window._manual_seed_centre_states[key]
            np.testing.assert_allclose(state.centres_source_xy, ((10.0, 10.0),))
            np.testing.assert_allclose(
                window._manual_seed_centres_in_corrected_coordinates(key, result),
                ((25.0, 22.0),),
            )
            self.assertEqual(window.pipeline.revision, revision + 1)
            invalidate.assert_called_once_with(expected_affected)
            analyze.assert_called_once_with(dirty_nodes=expected_affected)
            self.assertNotIn("background_likelihood", expected_affected)
            self.assertNotIn("edge_gradients", expected_affected)
            self.assertIn("procedural_instances", expected_affected)
            self.assertNotIn(key, window._analyses)
            self.assertEqual(
                window._cache_dirty_nodes[key],
                expected_affected,
            )

            stored = ManualSeedCentreStore(root).load_if_present(image_path, (60, 80))
            self.assertIsNotNone(stored)
            assert stored is not None
            np.testing.assert_allclose(stored.centres_xy, ((10.0, 10.0),))
            self.assertEqual(stored.mode, "augment")
            window.close()

    def test_qt_mode_switch_undo_and_reset_are_autosaved_per_image(self) -> None:
        from PySide6.QtCore import Qt
        from PySide6.QtTest import QTest

        from seedvision.persistence import ManualSeedCentreStore
        from seedvision.ui.main_window import MainWindow

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "images" / "capture.png"
            self._write_image(image_path)
            window = MainWindow(root)
            key = window._current_image_key()
            self.assertIsNotNone(key)
            assert key is not None
            result = self._procedural_result(
                ((20.0, 30.0), (70.0, 65.0)), (0, 0)
            )
            window._analyses[key] = result
            window.image_view._analysis_result = result
            inspector = window.pipeline_inspector
            replace_index = inspector.procedural_centres_mode_combo.findData(
                "replace_automatic"
            )
            self.assertGreaterEqual(replace_index, 0)

            with (
                patch.object(window.image_view, "_render_analysis"),
                patch.object(window, "_analyze_current_image") as analyze,
            ):
                window._pipeline_node_selected("procedural_instances")
                window._sync_procedural_centres_controls()
                # The combo is a real Qt action: Augment -> Replace copies all
                # currently editable markers before disabling discovery.
                inspector.procedural_centres_mode_combo.setCurrentIndex(replace_index)
                self.application.processEvents()
                replacement = window._manual_seed_centre_states[key]
                self.assertEqual(replacement.mode, "replace_automatic")
                np.testing.assert_allclose(
                    replacement.centres_source_xy,
                    ((20.0, 30.0), (70.0, 65.0)),
                )

                QTest.mouseClick(
                    inspector.procedural_centres_undo_button,
                    Qt.MouseButton.LeftButton,
                )
                undone = window._manual_seed_centre_states[key]
                self.assertEqual(undone.mode, "augment")
                self.assertEqual(undone.centres_source_xy.shape, (0, 2))

                # Reinstall the result removed by each invalidation, create an
                # edit, then drive Reset through its visible Qt control.
                window._analyses[key] = result
                window.image_view._analysis_result = result
                window._manual_seed_centres_edited(
                    np.asarray(((44.0, 42.0),)),
                    "augment",
                    "added manual seed centre",
                )
                QTest.mouseClick(
                    inspector.procedural_centres_reset_button,
                    Qt.MouseButton.LeftButton,
                )
                reset = window._manual_seed_centre_states[key]
                self.assertEqual(reset.mode, "augment")
                self.assertEqual(reset.centres_source_xy.shape, (0, 2))

            self.assertEqual(analyze.call_count, 4)
            loaded = ManualSeedCentreStore(root).load_if_present(image_path, (100, 100))
            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertEqual(loaded.mode, "augment")
            self.assertEqual(loaded.centres_xy.shape, (0, 2))
            window.close()

    def test_autosave_failure_keeps_the_edit_and_still_recomputes(self) -> None:
        from seedvision.ui.main_window import MainWindow, QMessageBox

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "images" / "capture.png"
            self._write_image(image_path)
            window = MainWindow(root)
            key = window._current_image_key()
            self.assertIsNotNone(key)
            assert key is not None
            result = self._procedural_result(((30.0, 30.0),), (0,))
            window._analyses[key] = result
            window.image_view._analysis_result = result

            with (
                patch.object(window.image_view, "_render_analysis"),
                patch.object(
                    window._manual_seed_centre_store,
                    "save",
                    side_effect=OSError("disk full"),
                ),
                patch.object(QMessageBox, "critical") as critical,
                patch.object(window, "_analyze_current_image") as analyze,
            ):
                window._manual_seed_centres_edited(
                    np.asarray(((44.0, 41.0),)),
                    "augment",
                    "added manual seed centre",
                )

            critical.assert_called_once()
            self.assertIn("remains active", critical.call_args.args[2])
            np.testing.assert_allclose(
                window._manual_seed_centre_states[key].centres_source_xy,
                ((44.0, 41.0),),
            )
            analyze.assert_called_once()
            self.assertIn("save failed", window.statusBar().currentMessage())
            window.close()

    def test_running_node_switch_and_image_switch_stop_qt_edit_mode(self) -> None:
        from seedvision.ui.main_window import MainWindow

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "images" / "a.png"
            second = root / "images" / "b.png"
            self._write_image(first)
            self._write_image(second)
            window = MainWindow(root)
            key = window._current_image_key()
            self.assertIsNotNone(key)
            assert key is not None
            result = self._procedural_result(((30.0, 30.0),), (0,))
            window._analyses[key] = result
            window.image_view._analysis_result = result

            with patch.object(window.image_view, "_render_analysis"):
                window._pipeline_node_selected("procedural_instances")
                window._sync_procedural_centres_controls()
                window.pipeline_inspector.procedural_centres_edit_button.click()
                self.assertTrue(window.image_view.manual_seed_centre_editing)

                window._active_tasks[first] = object()
                window._sync_procedural_centres_controls()
                self.assertFalse(window.image_view.manual_seed_centre_editing)
                self.assertFalse(
                    window.pipeline_inspector.procedural_centres_edit_button.isEnabled()
                )
                window._active_tasks.clear()
                window._sync_procedural_centres_controls()
                window.pipeline_inspector.procedural_centres_edit_button.click()
                self.assertTrue(window.image_view.manual_seed_centre_editing)

                window._pipeline_node_selected("background_likelihood")
                self.assertFalse(window.image_view.manual_seed_centre_editing)
                window._pipeline_node_selected("procedural_instances")
                window._sync_procedural_centres_controls()
                window.pipeline_inspector.procedural_centres_edit_button.click()
                self.assertTrue(window.image_view.manual_seed_centre_editing)

                window._open_path(second)
                self.assertFalse(window.image_view.manual_seed_centre_editing)

            window.close()


if __name__ == "__main__":
    unittest.main()
