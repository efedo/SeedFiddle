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

    def test_trace_and_fill_selectors_expose_only_applicable_edge_sources(self) -> None:
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
            self.assertEqual(window.shape_fill_evidence_combo.currentData(), "adaptive")
            self.assertEqual(window.image_view._edge_trace_options.edge_source, "ridges")
            self.assertEqual(window.image_view._smart_fill_options.edge_source, "ridges")
            self.assertEqual(window.image_view._smart_fill_edge_source, "ridges")
            self.assertEqual(window.image_view._shape_guided_edge_source, "adaptive")
            self.assertFalse(hasattr(window, "shape_snap_evidence_combo"))
            self.assertFalse(hasattr(window.image_view, "_shape_snap_edge_source"))

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
            self.assertEqual(
                window.smart_fill_evidence_combo.itemText(fill_index),
                "Combined (ridge priority)",
            )
            self.assertIn("45% strength", window.smart_fill_evidence_combo.toolTip())
            window.smart_fill_evidence_combo.setCurrentIndex(fill_index)
            self.application.processEvents()
            self.assertEqual(window.image_view._smart_fill_options.edge_source, "adaptive")

            shape_trace_index = window.shape_fill_evidence_combo.findData("traces")
            self.assertGreaterEqual(shape_trace_index, 0)
            window.shape_fill_evidence_combo.setCurrentIndex(shape_trace_index)
            self.application.processEvents()
            self.assertEqual(window.image_view._shape_guided_edge_source, "traces")

            smart_net_index = window.smart_fill_evidence_combo.findData(
                "net_physical"
            )
            shape_net_index = window.shape_fill_evidence_combo.findData(
                "net_physical"
            )
            self.assertGreaterEqual(smart_net_index, 0)
            self.assertGreaterEqual(shape_net_index, 0)
            self.assertEqual(
                window.smart_fill_evidence_combo.itemText(smart_net_index),
                "Net physical-edge probability",
            )
            window.smart_fill_evidence_combo.setCurrentIndex(smart_net_index)
            window.shape_fill_evidence_combo.setCurrentIndex(shape_net_index)
            self.application.processEvents()
            self.assertEqual(
                window.image_view._smart_fill_edge_source, "net_physical"
            )
            # SmartFillOptions retains its core-library-compatible physical
            # source while ImageView remembers the exact net raster selected.
            self.assertEqual(
                window.image_view._smart_fill_options.edge_source, "physical"
            )
            self.assertEqual(
                window.image_view._shape_guided_edge_source, "net_physical"
            )
            self.assertEqual(
                window.edge_trace_evidence_combo.findData("net_physical"), -1
            )

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
            non_physical = np.zeros((6, 8), dtype=np.float32)
            non_physical[5, 5] = 0.25
            physical[0, 6] = 0.50
            non_physical[0, 6] = 0.75
            reference_ridges = np.zeros((6, 8), dtype=np.uint8)
            reference_ridges[0, 0] = 203

            result = self._analysis_result(
                ridges=ridges,
                reference_ridges=reference_ridges,
                traces=traces,
                magnitude=magnitude,
                physical=physical,
                non_physical=non_physical,
                net_scale=0.5,
            )
            view = ImageView()
            self.addCleanup(view.close)
            succeeded, error = view.load_image(image_path)
            self.assertTrue(succeeded, error)
            view.show_analysis(result, render=False)

            selected = {
                source: view._annotation_edge_evidence(source)
                for source in (
                    "ridges",
                    "reference_ridges",
                    "traces",
                    "magnitude",
                    "physical",
                    "net_physical",
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

            expected_net = np.rint(
                np.clip(
                    selected["physical"].astype(np.float32)
                    - 0.5
                    * (non_physical * 255.0).astype(np.uint8).astype(np.float32),
                    0.0,
                    255.0,
                )
            ).astype(np.uint8)
            self.assertTrue(
                np.array_equal(selected["net_physical"], expected_net)
            )
            self.assertEqual(int(selected["net_physical"][5, 5]), 224)
            self.assertEqual(int(selected["net_physical"][0, 6]), 32)
            self.assertIs(
                view._annotation_edge_evidence("net_physical"),
                selected["net_physical"],
            )

            # The node's display-only coefficient is live evidence, not a
            # baked analysis product. A changed scale must not return the
            # raster cached for the previous scale.
            result.layers.net_physical_edge_internal_scale = 1.0
            updated_net = view._annotation_edge_evidence("net_physical")
            self.assertIsNot(updated_net, selected["net_physical"])
            self.assertEqual(int(updated_net[5, 5]), 192)
            self.assertEqual(int(updated_net[0, 6]), 0)

            old_cached_ridges = selected["ridges"]
            replacement_ridges = np.zeros_like(ridges)
            replacement_ridges[0, 7] = 231
            replacement = self._analysis_result(
                ridges=replacement_ridges,
                reference_ridges=reference_ridges,
                traces=traces,
                magnitude=magnitude,
                physical=physical,
                non_physical=non_physical,
                net_scale=0.25,
            )
            view.show_analysis(replacement, render=False)
            self.assertEqual(view._annotation_evidence_cache, {})
            refreshed_ridges = view._annotation_edge_evidence("ridges")
            self.assertIsNot(refreshed_ridges, old_cached_ridges)
            self.assertEqual(int(refreshed_ridges[0, 7]), 231)
            self.assertEqual(int(refreshed_ridges[1, 1]), 0)
            refreshed_net = view._annotation_edge_evidence("net_physical")
            self.assertEqual(int(refreshed_net[5, 5]), 239)

    @staticmethod
    def _analysis_result(
        *,
        ridges: np.ndarray,
        reference_ridges: np.ndarray,
        traces: np.ndarray,
        magnitude: np.ndarray,
        physical: np.ndarray,
        non_physical: np.ndarray,
        net_scale: float,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            layers=SimpleNamespace(
                edge_ridges=ridges,
                reference_edge_ridges=reference_ridges,
                edge_trace_labels=traces,
                edge_likelihood=magnitude,
                physical_edge_probability=physical,
                non_edge_probability=non_physical,
                net_physical_edge_internal_scale=net_scale,
                offset_x=0,
                offset_y=0,
            ),
            crop_offset=(0, 0),
        )


if __name__ == "__main__":
    unittest.main()
