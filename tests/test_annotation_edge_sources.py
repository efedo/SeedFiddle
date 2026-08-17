from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


class AnnotationEdgeSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        try:
            from PySide6.QtWidgets import QApplication
        except ImportError as error:
            raise unittest.SkipTest(f"PySide6 unavailable: {error}")
        cls.application = QApplication.instance() or QApplication([])

    def test_trace_and_fill_selectors_default_to_ridges_and_update_options(self) -> None:
        from PySide6.QtGui import QColor, QImage

        from seedvision.ui.main_window import MainWindow

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image_path = root / "images" / "seed.png"
            image_path.parent.mkdir(parents=True)
            image = QImage(48, 36, QImage.Format.Format_RGB32)
            image.fill(QColor("white"))
            self.assertTrue(image.save(str(image_path)))

            window = MainWindow(root)
            self.addCleanup(window.close)

            self.assertEqual(window.edge_trace_evidence_combo.currentData(), "ridges")
            self.assertEqual(window.smart_fill_evidence_combo.currentData(), "ridges")
            self.assertEqual(window.image_view._edge_trace_options.edge_source, "ridges")
            self.assertEqual(window.image_view._smart_fill_options.edge_source, "ridges")

            reference_index = window.edge_trace_evidence_combo.findData(
                "reference_ridges"
            )
            self.assertGreaterEqual(reference_index, 0)
            self.assertEqual(
                window.edge_trace_evidence_combo.itemText(reference_index),
                "Thinned reference edge ridge",
            )
            window.edge_trace_evidence_combo.setCurrentIndex(reference_index)
            self.application.processEvents()
            self.assertEqual(
                window.image_view._edge_trace_options.edge_source,
                "reference_ridges",
            )

            reference_fill_index = window.smart_fill_evidence_combo.findData(
                "reference_ridges"
            )
            self.assertGreaterEqual(reference_fill_index, 0)
            window.smart_fill_evidence_combo.setCurrentIndex(reference_fill_index)
            self.application.processEvents()
            self.assertEqual(
                window.image_view._smart_fill_options.edge_source,
                "reference_ridges",
            )

            trace_index = window.edge_trace_evidence_combo.findData("traces")
            self.assertGreaterEqual(trace_index, 0)
            window.edge_trace_evidence_combo.setCurrentIndex(trace_index)
            self.application.processEvents()
            self.assertEqual(window.image_view._edge_trace_options.edge_source, "traces")

            fill_index = window.smart_fill_evidence_combo.findData("adaptive")
            self.assertGreaterEqual(fill_index, 0)
            window.smart_fill_evidence_combo.setCurrentIndex(fill_index)
            self.application.processEvents()
            self.assertEqual(window.image_view._smart_fill_options.edge_source, "adaptive")

    def test_image_view_adapts_each_edge_source_and_invalidates_cached_values(self) -> None:
        from PySide6.QtGui import QColor, QImage

        from seedvision.ui.image_view import ImageView

        with tempfile.TemporaryDirectory() as temporary_directory:
            image_path = Path(temporary_directory) / "edge-sources.png"
            image = QImage(8, 6, QImage.Format.Format_RGB32)
            image.fill(QColor("white"))
            self.assertTrue(image.save(str(image_path)))

            ridges = np.zeros((6, 8), dtype=np.uint8)
            ridges[1, 1] = 180
            traces = np.zeros((6, 8), dtype=np.uint16)
            traces[2, 2] = 1
            traces[3, 3] = 300
            magnitude = np.zeros((6, 8), dtype=np.uint8)
            magnitude[4, 4] = 64
            physical = np.zeros((6, 8), dtype=np.float32)
            physical[5, 5] = 1.0
            reference_ridges = np.zeros((6, 8), dtype=np.uint8)
            reference_ridges[0, 0] = 203

            result = self._analysis_result(
                ridges=ridges,
                reference_ridges=reference_ridges,
                traces=traces,
                magnitude=magnitude,
                physical=physical,
            )
            view = ImageView()
            self.addCleanup(view.close)
            succeeded, error = view.load_image(image_path)
            self.assertTrue(succeeded, error)
            view.show_analysis(result, render=False)

            selected = {
                source: view._annotation_edge_evidence(source)
                for source in (
                    "ridges", "reference_ridges", "traces", "magnitude", "physical"
                )
            }
            for first_index, first in enumerate(selected):
                for second in tuple(selected)[first_index + 1 :]:
                    self.assertFalse(
                        np.array_equal(selected[first], selected[second]),
                        f"{first} and {second} unexpectedly selected the same raster",
                    )

            trace_evidence = selected["traces"]
            self.assertEqual(trace_evidence.dtype, np.uint8)
            self.assertEqual(int(trace_evidence[2, 2]), 255)
            self.assertEqual(int(trace_evidence[3, 3]), 255)
            self.assertEqual(np.unique(trace_evidence).tolist(), [0, 255])

            adaptive = view._annotation_edge_evidence("adaptive")
            expected_adaptive = np.maximum.reduce(
                (
                    ridges,
                    trace_evidence,
                    np.rint(selected["physical"] * 0.45).astype(np.uint8),
                    np.rint(magnitude * 0.45).astype(np.uint8),
                )
            )
            self.assertTrue(np.array_equal(adaptive, expected_adaptive))
            self.assertEqual(int(adaptive[2, 2]), 255)
            self.assertEqual(int(adaptive[3, 3]), 255)
            self.assertEqual(int(adaptive[5, 5]), 115)

            old_cached_ridges = selected["ridges"]
            replacement_ridges = np.zeros_like(ridges)
            replacement_ridges[0, 7] = 231
            replacement = self._analysis_result(
                ridges=replacement_ridges,
                reference_ridges=reference_ridges,
                traces=traces,
                magnitude=magnitude,
                physical=physical,
            )
            view.show_analysis(replacement, render=False)
            self.assertEqual(view._annotation_evidence_cache, {})
            refreshed_ridges = view._annotation_edge_evidence("ridges")
            self.assertIsNot(refreshed_ridges, old_cached_ridges)
            self.assertEqual(int(refreshed_ridges[0, 7]), 231)
            self.assertEqual(int(refreshed_ridges[1, 1]), 0)

    @staticmethod
    def _analysis_result(
        *,
        ridges: np.ndarray,
        reference_ridges: np.ndarray,
        traces: np.ndarray,
        magnitude: np.ndarray,
        physical: np.ndarray,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            layers=SimpleNamespace(
                edge_ridges=ridges,
                reference_edge_ridges=reference_ridges,
                edge_trace_labels=traces,
                edge_likelihood=magnitude,
                physical_edge_probability=physical,
                offset_x=0,
                offset_y=0,
            ),
            crop_offset=(0, 0),
        )


if __name__ == "__main__":
    unittest.main()
