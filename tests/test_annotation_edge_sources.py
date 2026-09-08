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

        from seedvision.annotation import (
            ANNOTATION_EDGE_SOURCES,
            EDGE_TRACE_EDGE_SOURCES,
        )
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
                "Reference-edge probability",
            )
            window.smart_fill_evidence_combo.setCurrentIndex(smart_net_index)
            window.shape_fill_evidence_combo.setCurrentIndex(shape_net_index)
            self.application.processEvents()
            self.assertEqual(
                window.image_view._smart_fill_edge_source, "net_physical"
            )
            self.assertEqual(
                window.image_view._smart_fill_options.edge_source,
                "net_physical",
            )
            self.assertEqual(
                window.image_view._shape_guided_edge_source, "net_physical"
            )
            self.assertEqual(
                window.edge_trace_evidence_combo.findData("net_physical"), -1
            )
            for combo in (window.smart_fill_evidence_combo, window.shape_fill_evidence_combo):
                conservative_index = combo.findData("conservative_net_physical")
                self.assertGreaterEqual(conservative_index, 0)
                self.assertEqual(
                    combo.itemText(conservative_index),
                    "Conservative net physical-edge evidence",
                )
            self.assertEqual(
                window.smart_fill_evidence_combo.findData("physical"), -1
            )
            self.assertEqual(
                window.shape_fill_evidence_combo.findData("physical"), -1
            )
            for combo in (
                window.edge_trace_evidence_combo,
                window.smart_fill_evidence_combo,
                window.shape_fill_evidence_combo,
            ):
                self.assertGreaterEqual(
                    combo.findData("normalized_net_physical"), 0
                )
                self.assertGreaterEqual(
                    combo.findData("normalized_reference_ridges"), 0
                )

            def sources(combo) -> set[str]:
                return {
                    str(combo.itemData(index)) for index in range(combo.count())
                }

            self.assertSetEqual(
                sources(window.edge_trace_evidence_combo),
                set(EDGE_TRACE_EDGE_SOURCES),
            )
            self.assertSetEqual(
                sources(window.smart_fill_evidence_combo),
                set(ANNOTATION_EDGE_SOURCES),
            )
            self.assertSetEqual(
                sources(window.shape_fill_evidence_combo),
                set(ANNOTATION_EDGE_SOURCES),
            )

            # Every value offered by a selector must survive the exact settings
            # synchronization that runs while a completed analysis is installed.
            for index in range(window.edge_trace_evidence_combo.count()):
                window.edge_trace_evidence_combo.setCurrentIndex(index)
                self.application.processEvents()
                self.assertEqual(
                    window.image_view._edge_trace_options.edge_source,
                    window.edge_trace_evidence_combo.itemData(index),
                )
            for index in range(window.smart_fill_evidence_combo.count()):
                window.smart_fill_evidence_combo.setCurrentIndex(index)
                self.application.processEvents()
                selected = str(window.smart_fill_evidence_combo.itemData(index))
                self.assertEqual(window.image_view._smart_fill_edge_source, selected)
                self.assertEqual(
                    window.image_view._smart_fill_options.edge_source,
                    selected,
                )
            for index in range(window.shape_fill_evidence_combo.count()):
                window.shape_fill_evidence_combo.setCurrentIndex(index)
                self.application.processEvents()
                self.assertEqual(
                    window.image_view._shape_guided_edge_source,
                    window.shape_fill_evidence_combo.itemData(index),
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
            ridges[5, 5] = 128
            ridges[0, 6] = 192
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
            normalized = np.zeros((6, 8), dtype=np.uint8)
            normalized[5, 5] = 140
            normalized[0, 6] = 96
            normalized_ridges = np.zeros((6, 8), dtype=np.uint8)
            normalized_ridges[0, 1] = 211

            result = self._analysis_result(
                ridges=ridges,
                reference_ridges=reference_ridges,
                traces=traces,
                magnitude=magnitude,
                physical=physical,
                non_physical=non_physical,
                normalized=normalized,
                normalized_ridges=normalized_ridges,
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
                    "conservative_reference_ridges",
                    "normalized_reference_ridges",
                    "traces",
                    "magnitude",
                    "net_physical",
                    "conservative_net_physical",
                    "normalized_net_physical",
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
                    normalized_ridges,
                    trace_evidence,
                    np.rint(normalized * 0.45).astype(np.uint8),
                    np.rint(magnitude * 0.45).astype(np.uint8),
                )
            )
            self.assertTrue(np.array_equal(adaptive, expected_adaptive))
            self.assertEqual(int(adaptive[2, 2]), 255)
            self.assertEqual(int(adaptive[3, 3]), 255)
            self.assertEqual(int(adaptive[5, 5]), 128)

            expected_net = np.asarray(
                result.layers.reference_edge_probability, dtype=np.uint8
            )
            self.assertTrue(
                np.array_equal(selected["net_physical"], expected_net)
            )
            self.assertEqual(int(selected["net_physical"][5, 5]), 128)
            self.assertEqual(int(selected["net_physical"][0, 6]), 96)
            self.assertEqual(int(selected["conservative_net_physical"][5, 5]), 112)
            self.assertEqual(int(selected["conservative_net_physical"][0, 6]), 24)
            np.testing.assert_array_equal(
                selected["conservative_reference_ridges"], result.layers.net_reference_edge_ridges
            )
            self.assertIs(
                view._annotation_edge_evidence("net_physical"),
                selected["net_physical"],
            )

            # Only conservative evidence includes the subtraction coefficient.
            # Both sources use their cached outputs. Merely changing adjacent
            # metadata cannot make fill disagree with the displayed field.
            result.layers.net_physical_edge_internal_scale = 1.0
            updated_net = view._annotation_edge_evidence("net_physical")
            self.assertIs(updated_net, selected["net_physical"])
            self.assertEqual(int(updated_net[5, 5]), 128)
            self.assertEqual(int(updated_net[0, 6]), 96)
            self.assertIs(
                view._annotation_edge_evidence("conservative_net_physical"),
                selected["conservative_net_physical"],
            )

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
                normalized=normalized,
                normalized_ridges=normalized_ridges,
                net_scale=0.25,
            )
            view.show_analysis(replacement, render=False)
            self.assertEqual(view._annotation_evidence_cache, {})
            refreshed_ridges = view._annotation_edge_evidence("ridges")
            self.assertIsNot(refreshed_ridges, old_cached_ridges)
            self.assertEqual(int(refreshed_ridges[0, 7]), 231)
            self.assertEqual(int(refreshed_ridges[1, 1]), 0)
            refreshed_net = view._annotation_edge_evidence("net_physical")
            self.assertEqual(int(refreshed_net[5, 5]), 0)
            self.assertEqual(int(refreshed_net[0, 7]), 0)

    def test_compact_fallback_separates_probability_from_conservative_evidence(self) -> None:
        from unittest.mock import patch

        from seedvision.ui.image_view import ImageView

        view = ImageView()
        self.addCleanup(view.close)
        layers = SimpleNamespace(net_physical_edge_internal_scale=0.5)
        view._analysis_result = SimpleNamespace(layers=layers)
        rasters = {
            "physical_edge_probability": np.asarray(((200, 200, 40),), np.uint8),
            "non_edge_probability": np.asarray(((160, 160, 200),), np.uint8),
            "edge_ridges": np.asarray(((128, 0, 255),), np.uint8),
        }
        with patch.object(view, "_full_annotation_evidence", side_effect=rasters.__getitem__):
            probability = view._annotation_edge_evidence("net_physical")
            conservative = view._annotation_edge_evidence("conservative_net_physical")
            np.testing.assert_array_equal(probability, ((100, 0, 40),))
            np.testing.assert_array_equal(conservative, ((60, 0, 0),))
            layers.net_physical_edge_internal_scale = 2.0
            self.assertIs(view._annotation_edge_evidence("net_physical"), probability)
            np.testing.assert_array_equal(
                view._annotation_edge_evidence("conservative_net_physical"), ((0, 0, 0),)
            )

    @staticmethod
    def _analysis_result(
        *,
        ridges: np.ndarray,
        reference_ridges: np.ndarray,
        traces: np.ndarray,
        magnitude: np.ndarray,
        physical: np.ndarray,
        non_physical: np.ndarray,
        normalized: np.ndarray,
        normalized_ridges: np.ndarray,
        net_scale: float,
    ) -> SimpleNamespace:
        physical_u8 = np.rint(
            np.asarray(physical, dtype=np.float32)
            * (255.0 if np.asarray(physical).max(initial=0.0) <= 1.0 else 1.0)
        ).clip(0.0, 255.0).astype(np.uint8)
        non_physical_u8 = np.rint(
            np.asarray(non_physical, dtype=np.float32)
            * (
                255.0
                if np.asarray(non_physical).max(initial=0.0) <= 1.0
                else 1.0
            )
        ).clip(0.0, 255.0).astype(np.uint8)
        net_physical = np.rint(
            np.clip(
                physical_u8.astype(np.float32)
                - float(net_scale) * non_physical_u8.astype(np.float32),
                0.0,
                255.0,
            )
        ).astype(np.uint8)
        reference_edge_probability = np.rint(
            physical_u8.astype(np.float32)
            * np.asarray(ridges, dtype=np.float32)
            / 255.0
        ).astype(np.uint8)
        conservative_evidence = np.rint(
            net_physical.astype(np.float32)
            * np.asarray(ridges, dtype=np.float32)
            / 255.0
        ).astype(np.uint8)
        return SimpleNamespace(
            layers=SimpleNamespace(
                edge_ridges=ridges,
                reference_edge_ridges=reference_ridges,
                net_reference_edge_ridges=np.rint(reference_ridges * 0.5).astype(np.uint8),
                edge_trace_labels=traces,
                edge_likelihood=magnitude,
                physical_edge_probability=physical,
                non_edge_probability=non_physical,
                net_physical_edge_probability=net_physical,
                reference_edge_probability=reference_edge_probability,
                conservative_net_physical_edge_evidence=conservative_evidence,
                locally_normalized_net_physical_edge=normalized,
                normalized_net_reference_edge_ridges=normalized_ridges,
                net_physical_edge_internal_scale=net_scale,
                offset_x=0,
                offset_y=0,
            ),
            crop_offset=(0, 0),
        )


if __name__ == "__main__":
    unittest.main()
