from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np


class EdgeTraceCommitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        try:
            from PySide6.QtWidgets import QApplication
        except ImportError as error:
            raise unittest.SkipTest(f"PySide6 unavailable: {error}")
        cls.application = QApplication.instance() or QApplication([])

    def _image_view(self, root: Path, name: str = "trace.png"):
        from PySide6.QtGui import QColor, QImage

        from seedvision.ui.image_view import ImageView

        path = root / name
        image = QImage(72, 64, QImage.Format.Format_RGB32)
        image.fill(QColor("white"))
        self.assertTrue(image.save(str(path)))
        view = ImageView()
        self.addCleanup(view.close)
        succeeded, error = view.load_image(path)
        self.assertTrue(succeeded, error)
        view.set_instance_annotation_editing(True)
        view.set_instance_annotation_tool("edge_trace")
        return view

    @staticmethod
    def _set_preview(view, end_xy: tuple[int, int], geometry) -> None:
        from PySide6.QtCore import QPointF

        endpoint = QPointF(float(end_xy[0]), float(end_xy[1]))
        view._instance_preview_point = QPointF(endpoint)
        view._instance_preview_endpoint = QPointF(endpoint)
        view._instance_preview_geometry = np.asarray(geometry, dtype=np.int32)

    def test_open_trace_is_exactly_one_pixel_independent_of_brush_radius(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            view = self._image_view(Path(temporary_directory))
            view.set_active_instance_id(4)
            view.set_reference_brush_radius(40.0)

            changed = view._paint_instance_geometry(
                np.asarray(((8, 30), (56, 30)), dtype=np.int32),
                filled=False,
            )

            labels = view.instance_annotations()
            self.assertEqual(changed, 49)
            self.assertEqual(int(np.count_nonzero(labels == 4)), 49)
            self.assertTrue(np.all(labels[30, 8:57] == 4))
            self.assertFalse(np.any(labels[:30] == 4))
            self.assertFalse(np.any(labels[31:] == 4))

    def test_closing_trace_fills_only_inward_and_preserves_other_ids(self) -> None:
        from PySide6.QtCore import QPointF

        from seedvision.annotation import EdgeTraceOptions

        with tempfile.TemporaryDirectory() as temporary_directory:
            view = self._image_view(Path(temporary_directory))
            labels = np.zeros((64, 72), dtype=np.uint16)
            labels[22:25, 22:25] = 9
            view.set_instance_annotations(labels)
            view.set_active_instance_id(3)
            view.set_edge_trace_options(
                EdgeTraceOptions(tangent_mode="off", fill_closed_loops=True)
            )

            first = QPointF(10.0, 10.0)
            self._set_preview(view, (10, 10), ((10, 10),))
            self.assertFalse(view._commit_instance_assisted_preview(first))

            for endpoint, segment in (
                ((40, 10), ((10, 10), (40, 10))),
                ((40, 40), ((40, 10), (40, 40))),
                ((10, 40), ((40, 40), (10, 40))),
            ):
                self._set_preview(view, endpoint, segment)
                self.assertTrue(
                    view._commit_instance_assisted_preview(
                        QPointF(float(endpoint[0]), float(endpoint[1]))
                    )
                )

            open_labels = view.instance_annotations()
            self.assertEqual(int(open_labels[20, 20]), 0)

            self._set_preview(view, (10, 10), ((10, 40), (10, 10)))
            self.assertTrue(view._commit_instance_assisted_preview(first))

            filled = view.instance_annotations()
            self.assertEqual(int(filled[20, 20]), 3)
            self.assertTrue(np.all(filled[22:25, 22:25] == 9))
            self.assertEqual(int(filled[9, 20]), 0)
            self.assertEqual(int(filled[41, 20]), 0)
            self.assertEqual(int(filled[20, 9]), 0)
            self.assertEqual(int(filled[20, 41]), 0)
            rows, columns = np.nonzero(filled == 3)
            self.assertEqual((int(columns.min()), int(columns.max())), (10, 40))
            self.assertEqual((int(rows.min()), int(rows.max())), (10, 40))
            self.assertIsNone(view._instance_trace_anchor)
            self.assertIsNone(view._instance_trace_geometry)

    def test_closed_fill_can_be_disabled_and_trace_session_resets(self) -> None:
        from PySide6.QtCore import QPointF

        from seedvision.annotation import EdgeTraceOptions

        with tempfile.TemporaryDirectory() as temporary_directory:
            view = self._image_view(Path(temporary_directory))
            view.set_edge_trace_options(
                EdgeTraceOptions(tangent_mode="off", fill_closed_loops=False)
            )

            first = QPointF(12.0, 12.0)
            self._set_preview(view, (12, 12), ((12, 12),))
            self.assertFalse(view._commit_instance_assisted_preview(first))
            for endpoint, segment in (
                ((36, 12), ((12, 12), (36, 12))),
                ((36, 36), ((36, 12), (36, 36))),
                ((12, 36), ((36, 36), (12, 36))),
                ((12, 12), ((12, 36), (12, 12))),
            ):
                self._set_preview(view, endpoint, segment)
                self.assertTrue(
                    view._commit_instance_assisted_preview(
                        QPointF(float(endpoint[0]), float(endpoint[1]))
                    )
                )

            labels = view.instance_annotations()
            self.assertEqual(int(labels[24, 24]), 0)
            self.assertEqual(int(labels[12, 24]), 1)
            self.assertIsNone(view._instance_trace_anchor)
            self.assertIsNone(view._instance_trace_geometry)

            self._set_preview(view, (14, 14), ((14, 14),))
            self.assertFalse(
                view._commit_instance_assisted_preview(QPointF(14.0, 14.0))
            )
            self.assertIsNotNone(view._instance_trace_geometry)
            view.set_active_instance_id(2)
            self.assertIsNone(view._instance_trace_anchor)
            self.assertIsNone(view._instance_trace_geometry)

    def test_fill_closed_loops_checkbox_defaults_on_and_updates_options(self) -> None:
        from seedvision.annotation import EdgeTraceOptions
        from seedvision.ui.main_window import MainWindow

        with tempfile.TemporaryDirectory() as temporary_directory:
            window = MainWindow(Path(temporary_directory))
            self.addCleanup(window.close)

            self.assertTrue(EdgeTraceOptions().fill_closed_loops)
            self.assertTrue(window.edge_trace_fill_closed_checkbox.isChecked())
            self.assertTrue(window.image_view._edge_trace_options.fill_closed_loops)

            window.edge_trace_fill_closed_checkbox.setChecked(False)
            self.assertFalse(window.image_view._edge_trace_options.fill_closed_loops)
            window.edge_trace_fill_closed_checkbox.setChecked(True)
            self.assertTrue(window.image_view._edge_trace_options.fill_closed_loops)


if __name__ == "__main__":
    unittest.main()
