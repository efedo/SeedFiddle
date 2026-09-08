"""Regression contracts for visibility-aware measurements and annotation tools."""
from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import cv2
import numpy as np

from seedvision.measurement import measure_shape_mask, measure_reviewed_seed_instances
from seedvision.persistence.reference_regions import (
    SeedInstanceAnnotation, ReferenceRegionBundle, ReferenceRegionStore,
    _seed_annotations_from_payload,
)


def reviewed(i=1, **kw):
    return SeedInstanceAnnotation(i, shape_reviewed=True, outline_visibility="complete",
                                  pose="flat", **kw)


class ShapeRevisionTests(unittest.TestCase):
    def test_old_shape_libraries_cannot_be_relabelled_after_source_exclusion(self):
        from tests.test_reference_library import _manifest, _contribution
        from seedvision.reference_library.aggregation import aggregate_contributions, contributions_from_artifact
        from seedvision.reference_library.contracts import SpeciesLibraryArtifact, SpeciesLibraryPin, BiologicalContext
        from seedvision.reference_library.service import SpeciesLibraryService
        aggregated = aggregate_contributions((_contribution(0), _contribution(1)))
        artifact = SpeciesLibraryArtifact(_manifest(aggregated),
            foreground_colour=aggregated["foreground_colour"],
            shape_summary=replace(aggregated["shape_summary"], schema_id="species-shape-summary-v2"),
            dimensions_shape=replace(aggregated["dimensions_shape"], schema_id="species-dimensions-shape-bank-v2"))
        service = SpeciesLibraryService(SimpleNamespace(load=lambda _pin: artifact))
        resolved = service.resolve(SpeciesLibraryPin("library-1", "1", "lupinus_mutabilis", "d"*64),
            context=BiologicalContext("lupinus_mutabilis"),
            current_source_sha256=artifact.manifest.sources[0].source_sha256,
            trait_vocabulary_sha256="a"*64)
        self.assertIsNone(resolved.artifact.dimensions_shape)
        self.assertIsNone(resolved.artifact.shape_summary)
        self.assertIsNotNone(resolved.artifact.foreground_colour)
        self.assertTrue(any("rebuild" in warning for warning in resolved.provenance.warnings))
        self.assertTrue(all(not item.shape_observations for item in contributions_from_artifact(artifact)))

    def test_curvature_prior_changes_only_trace_and_downstream_cache(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        from seedvision.visualization import build_analysis_layers
        image = np.full((96, 96, 3), 225, np.uint8)
        cv2.ellipse(image, (48, 48), (28, 18), 0, 0, 360, (30, 75, 170), -1)
        valid = np.full((96, 96), 255, np.uint8)
        cache = {}
        args = (image, valid, np.asarray(((48, 48),), np.float32), np.asarray((23,), np.float32), 46.)
        build_analysis_layers(*args, offset_x=0, offset_y=0, cache_values=cache)
        gradients = cache["layer.edge_gradients"]
        first = cache["layer.seed_edge_curves"]
        # The consumer needs only compact statistical values, no seed IDs or coordinates.
        summary = SimpleNamespace(boundary_curvature_times_diameter=(.1, .2, .3, .4))
        build_analysis_layers(*args, offset_x=0, offset_y=0, cache_values=cache,
                             dirty_nodes={"edge_traces", "seed_edge_curves"},
                             seed_measurement_summary=summary)
        changed = cache["layer.seed_edge_curves"]
        self.assertIs(cache["layer.edge_gradients"], gradients)
        self.assertIs(changed.ridge_state, first.ridge_state)
        self.assertNotEqual(changed.trace_state.signature, first.trace_state.signature)
        self.assertFalse(torch.equal(changed.trace_state.trace_confidence,
                                     first.trace_state.trace_confidence))

    def test_ellipse_major_axis_orientation_and_residual_match_outline(self):
        for angle in (0, 27, 75, 135):
            mask = np.zeros((180, 200), np.uint8)
            cv2.ellipse(mask, (100, 90), (50, 25), angle, 0, 360, 1, -1)
            m = measure_shape_mask(mask)
            difference = (m.ellipse.orientation_degrees-angle+90) % 180-90
            self.assertLess(abs(difference), 2.)
            self.assertLess(m.non_ellipticity, .03)
            self.assertGreater(m.measurement_uncertainty[0], .5)

    def test_partial_full_length_contributes_only_size_and_not_shape(self):
        labels = np.zeros((140, 230), np.uint16)
        cv2.ellipse(labels, (55, 70), (36, 24), 0, 0, 360, 1, -1)
        cv2.ellipse(labels, (160, 70), (30, 22), 0, 0, 360, 2, -1)
        labels[76:, 120:] = 0
        partial = replace(reviewed(2), outline_visibility="partly_occluded", full_length_visible=True)
        summary = measure_reviewed_seed_instances(labels, (reviewed(), partial))
        self.assertEqual(summary.eligible_count, 1)
        self.assertEqual(len(summary.size_observations), 2)
        self.assertEqual(summary.vectors.shape[0], 1)
        self.assertEqual(len(summary.boundary_turn_degrees), 128)
        self.assertIsNotNone(summary.mean_internal_concavity)
        without = measure_reviewed_seed_instances(labels, (reviewed(), replace(partial, full_length_visible=False)))
        self.assertEqual(len(without.size_observations), 1)
        mean, sd, error = summary.size_statistics
        self.assertAlmostEqual(mean, np.mean([i.measurement.maximum_span for i in summary.size_observations]))
        self.assertGreater(sd, 0)
        self.assertGreater(error, 0)

    def test_visibility_must_be_explicit_even_for_large_uncropped_mask(self):
        labels = np.zeros((80, 100), np.uint16)
        cv2.ellipse(labels, (50, 40), (32, 22), 0, 0, 360, 1, -1)
        summary = measure_reviewed_seed_instances(labels, (SeedInstanceAnnotation(1),))
        self.assertIsNone(summary.size_statistics)
        from seedvision.segmentation.baseline import _annotated_seed_diameter_measurements
        self.assertFalse(_annotated_seed_diameter_measurements(labels)[0].selected)

    def test_boundary_distribution_is_seed_balanced_and_rotation_invariant(self):
        summaries = []
        for angle in (0, 47):
            labels = np.zeros((220, 220), np.uint16)
            cv2.ellipse(labels, (110, 110), (64, 40), angle, 0, 360, 1, -1)
            summaries.append(measure_reviewed_seed_instances(labels, (reviewed(),)))
        self.assertAlmostEqual(sum(summaries[0].boundary_turn_degrees), 360., delta=1.)
        a, b = [np.quantile(s.boundary_curvature_times_diameter, [.1, .5, .9]) for s in summaries]
        np.testing.assert_allclose(a, b, atol=.5)

    def test_archive_round_trip_and_version_four_default(self):
        from PySide6.QtGui import QImage
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "image.png"
            image = QImage(30, 20, QImage.Format.Format_RGB888)
            image.fill(0)
            image.save(str(path))
            labels = np.zeros((20, 30), np.uint16)
            labels[5:15, 8:22] = 1
            item = replace(reviewed(), outline_visibility="partly_occluded", full_length_visible=True)
            store = ReferenceRegionStore(root)
            store.save(path, ReferenceRegionBundle(labels.shape, annotated_seeds=labels, seed_annotations=(item,)))
            self.assertTrue(store.load_if_present(path, expected_shape=labels.shape).seed_annotations[0].full_length_visible)
            from dataclasses import asdict
            old = asdict(item)
            old.pop("full_length_visible")
            old["conditions"] = []
            loaded = _seed_annotations_from_payload([old], frozenset({1}), version=4)
            self.assertFalse(loaded[0].full_length_visible)

    def test_old_top_fraction_profile_migrates_without_retaining_control(self):
        from seedvision.pipeline import build_default_pipeline
        from seedvision.persistence import (analysis_settings_profile_from_graph,
            analysis_settings_profile_to_payload, analysis_settings_profile_from_payload,
            apply_analysis_settings_profile)
        graph = build_default_pipeline()
        payload = analysis_settings_profile_to_payload(analysis_settings_profile_from_graph(graph))
        payload["version"] = 17
        node = next(n for n in payload["nodes"] if n["id"] == "seed_scale_estimation")
        node["parameters"]["annotated_seed_top_fraction"] = .25
        payload["connections"] = [c for c in payload["connections"] if c["target_port"] != "reference_curvature"]
        apply_analysis_settings_profile(graph, analysis_settings_profile_from_payload(payload))
        self.assertNotIn("annotated_seed_top_fraction", graph.node("seed_scale_estimation").parameters)
        for target in ("edge_traces", "seed_edge_curves"):
            self.assertTrue(any(c.target == target and c.target_port == "reference_curvature" for c in graph.connections))


class AnnotationWidgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def test_distribution_has_axes_units_pose_key_and_error_bars(self):
        from PySide6.QtGui import QImage
        from PySide6.QtWidgets import QGraphicsTextItem, QGraphicsLineItem
        from seedvision.ui.image_view import ImageView
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.png"
            img = QImage(1000, 800, QImage.Format.Format_RGB888)
            img.fill(0)
            img.save(str(path))
            view = ImageView()
            view.load_image(path)
            labels = np.zeros((800, 1000), np.uint16)
            cv2.ellipse(labels, (300, 450), (65, 40), 35, 0, 360, 1, -1)
            summary = measure_reviewed_seed_instances(labels, (reviewed(),), pixels_per_mm=10)
            view._render_size_ovality_distribution(SimpleNamespace(), summary, None)
            text = "\n".join(i.toPlainText() for i in view._overlay_items if isinstance(i, QGraphicsTextItem))
            self.assertIn("Maximum span (mm)", text)
            self.assertIn("Ovality = body length / width", text)
            self.assertIn("±1σ", text)
            self.assertIn("Green: flat", text)
            self.assertGreaterEqual(sum(isinstance(i, QGraphicsLineItem) for i in view._overlay_items), 12)
            view.close()

    def test_imperial_measurement_uses_its_own_dividers_and_transform(self):
        from seedvision.calibration.image import ImageCalibration, RulerDetection
        ruler = RulerDetection((0, 5), (150, 5), 75, 15, 150, 25, 0, 1.,
            imperial_tick_segments=tuple((((x, 35), (x, 20))) for x in (10, 30, 50, 70)),
            imperial_unit_divider_indices=(1, 3))
        calibration = ImageCalibration(np.zeros((80, 180, 3), np.uint8), None, ruler,
            np.asarray(((1., 0., 4.), (0., 1., -2.), (0., 0., 1.))), 0., False, 0., (1., 1., 1.),
            10., 1., 10., 10., (), ruler_in_corrected_coordinates=False)
        np.testing.assert_allclose(calibration.imperial_measurement_endpoints_corrected(),
                                   ((34, 33), (74, 33)))

    def test_no_defects_empty_selector_and_shape_fields(self):
        from PySide6.QtGui import QImage
        from seedvision.ui.main_window import MainWindow
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "images").mkdir()
            image = QImage(100, 80, QImage.Format.Format_RGB888)
            image.fill(0)
            image.save(str(root / "images" / "example.png"))
            window = MainWindow(root)
            try:
                key = window._current_image_key()
                self.assertIsNotNone(key)
                labels = np.zeros((80, 100), np.uint16)
                labels[20:60, 25:75] = 1
                window._instance_annotations_edited(labels)
                window.instance_id_spin.setValue(1)
                window._sync_seed_trait_controls()
                self.assertTrue(window.instance_empty_label.isHidden())
                self.assertEqual(window.existing_instance_combo.count(), 2)
                window.seed_conditions_reviewed_checkbox.setChecked(True)
                self.assertTrue(window._draft_seed_annotations[key][1].conditions_reviewed)
                damage = next(iter(window.seed_condition_checkboxes))
                window.seed_condition_checkboxes[damage].setChecked(True)
                self.assertFalse(window.seed_conditions_reviewed_checkbox.isChecked())
                self.assertIn(damage, window._draft_seed_annotations[key][1].conditions)
                window.seed_conditions_reviewed_checkbox.setChecked(True)
                self.assertEqual(window._draft_seed_annotations[key][1].conditions, ())
                self.assertFalse(hasattr(window, "seed_physical_id_edit"))
                window.seed_outline_visibility_combo.setCurrentIndex(window.seed_outline_visibility_combo.findData("complete"))
                self.assertTrue(window.seed_full_length_checkbox.isHidden())
                window.seed_outline_visibility_combo.setCurrentIndex(window.seed_outline_visibility_combo.findData("partly_occluded"))
                self.assertFalse(window.seed_full_length_checkbox.isHidden())
                window.seed_full_length_checkbox.setChecked(True)
                self.assertTrue(window._draft_seed_annotations[key][1].full_length_visible)
                window.instance_id_spin.setValue(2)
                self.assertFalse(window.instance_empty_label.isHidden())
                window._select_existing_instance(1)
                self.assertEqual(window.instance_id_spin.value(), 1)
            finally:
                window.close()

    def test_hilum_drag_is_not_a_paint_stroke_and_survives_direction_edit(self):
        from PySide6.QtCore import Qt, QPoint
        from PySide6.QtGui import QImage
        from PySide6.QtTest import QTest
        from seedvision.ui.image_view import ImageView
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.png"
            img = QImage(200, 160, QImage.Format.Format_RGB888)
            img.fill(0)
            img.save(str(path))
            view = ImageView()
            view.resize(500, 440)
            view.show()
            view.load_image(path)
            self.app.processEvents()
            view.set_hilum_editing(True)
            edits, strokes = [], []
            view.hilum_landmark_edited.connect(lambda p, d: edits.append((p, d)))
            view.instance_annotations_edited.connect(strokes.append)
            start = view.mapFromScene(80., 90.)
            end = view.mapFromScene(120., 90.)
            QTest.mousePress(view.viewport(), Qt.MouseButton.LeftButton, pos=start)
            QTest.mouseMove(view.viewport(), end)
            QTest.mouseRelease(view.viewport(), Qt.MouseButton.LeftButton, pos=end)
            self.assertEqual(len(edits), 1)
            np.testing.assert_allclose(edits[0][1], (1., 0.), atol=.05)
            self.assertEqual(strokes, [])
            self.assertEqual(len(view._hilum_items), 2)
            QTest.keyClick(view, Qt.Key.Key_Escape)
            self.assertFalse(view._hilum_editing)
            cancellations = []
            view.hilum_editing_cancelled.connect(lambda: cancellations.append(True))
            view.set_hilum_editing(True)
            view.load_image(path)
            self.assertFalse(view._hilum_editing)
            self.assertEqual(cancellations, [True])
            view.close()

    def test_brush_stack_uses_only_current_page_and_resize_is_remembered(self):
        from PySide6.QtCore import Qt, QPoint
        from PySide6.QtTest import QTest
        from seedvision.ui.main_window import MainWindow
        window = MainWindow(Path(__file__).resolve().parents[1])
        try:
            window.show()
            window.annotate_instances_action.setEnabled(True)
            window.annotate_instances_action.setChecked(True)
            self.app.processEvents()
            stack = window.instance_tool_options_stack
            self.assertLess(stack.height(), 120)
            grip = window.reference_panel.resize_grip
            start = grip.rect().center()
            QTest.mousePress(grip, Qt.MouseButton.LeftButton, pos=start)
            QTest.mouseMove(grip, start+QPoint(65, -40))
            QTest.mouseRelease(grip, Qt.MouseButton.LeftButton, pos=start+QPoint(65, -40))
            self.assertIsNotNone(window.reference_panel.user_size)
            size = window.reference_panel.size()
            window.image_view._layout_context_panel()
            self.assertEqual(window.reference_panel.size(), size)
        finally:
            window.close()
