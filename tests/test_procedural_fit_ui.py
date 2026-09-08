from __future__ import annotations

import os
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


class ProceduralFitMainWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        try:
            from PySide6.QtWidgets import QApplication
        except ImportError as error:
            raise unittest.SkipTest(f"PySide6 unavailable: {error}")
        cls.application = QApplication.instance() or QApplication([])

    def _window_and_key(self):
        from seedvision.ui.main_window import MainWindow

        # Never let a developer's ignored per-image archive open a modal or
        # influence these controller-only tests.
        with patch(
            "seedvision.ui.main_window.ReferenceRegionStore.load_if_present",
            return_value=None,
        ):
            window = MainWindow(ROOT)
        def dispose() -> None:
            window._project_tracking_enabled = False
            window._project_dirty = False
            window._procedural_fit_task = None
            window.close()

        self.addCleanup(dispose)
        key = window._current_image_key()
        if key is None:
            self.skipTest("No workspace image is present")
        # Local ignored archives must not make these controller tests depend on
        # a developer's current annotation work.
        window._applied_instance_annotations.pop(key, None)
        window._draft_instance_annotations.pop(key, None)
        window._instance_annotations_dirty.discard(key)
        window._instance_continuity_cache.pop(key, None)
        window._project_tracking_enabled = True
        window._update_project_chrome()
        return window, key

    @staticmethod
    def _fit_report(initial_settings, *, improved: bool = True):
        from seedvision.segmentation.procedural_fit import (
            ProceduralFitResult,
            ProceduralFitScore,
        )

        initial_score = ProceduralFitScore(
            loss=0.42,
            true_positive_pixels=800,
            false_positive_pixels=120,
            distance_weighted_false_positive_pixels=156.0,
            false_negative_pixels=80,
            overreach_distance_scale_px=24.0,
            matched_instances=2,
            annotated_instances=2,
            evaluated_predictions=2,
            false_positive_instances=0,
            false_negative_instances=0,
            pixel_precision=0.87,
            pixel_recall=0.91,
        )
        proposed_score = replace(
            initial_score,
            loss=0.19 if improved else 0.42,
            false_positive_pixels=35 if improved else 120,
            distance_weighted_false_positive_pixels=(
                41.5 if improved else 156.0
            ),
            false_negative_pixels=42 if improved else 80,
            pixel_precision=0.96 if improved else 0.87,
            pixel_recall=0.95 if improved else 0.91,
        )
        proposed_settings = (
            replace(
                initial_settings,
                foreground_threshold_scale=(
                    initial_settings.foreground_threshold_scale + 0.08
                ),
            )
            if improved
            else initial_settings
        )
        return ProceduralFitResult(
            initial_settings=initial_settings,
            proposed_settings=proposed_settings,
            initial_score=initial_score,
            proposed_score=proposed_score,
            trials=(),
        )

    def _install_completed_task(
        self,
        window,
        key,
        annotations,
        *,
        annotations_are_complete: bool = False,
    ) -> None:
        window._applied_instance_annotations[key] = annotations
        window._selected_pipeline_node = "procedural_instances"
        window.pipeline_inspector.set_node(
            window.pipeline.node("procedural_instances")
        )
        window._procedural_fit_task = SimpleNamespace(
            image_key=key,
            pipeline_revision=window.pipeline.revision,
            annotation_source=annotations,
            annotations_are_complete=annotations_are_complete,
        )
        window._procedural_fit_progress = None

    def test_fit_action_forwards_annotation_coverage_mode(self) -> None:
        from seedvision.ui.pipeline_inspector import PipelineInspector

        window, _key = self._window_and_key()
        requests: list[tuple[float, float, bool]] = []
        window._start_procedural_fit = (
            lambda penalty, distance_scale, complete: requests.append(
                (penalty, distance_scale, complete)
            )
        )

        window._pipeline_node_action_requested(
            "procedural_instances",
            PipelineInspector.PROCEDURAL_FIT_ACTION,
            {
                "false_positive_weight": 3.25,
                "overreach_distance_scale_fraction": 0.30,
                "annotations_are_complete": True,
            },
        )

        self.assertEqual(requests, [(3.25, 0.30, True)])

    def test_fit_task_passes_complete_coverage_to_core_options(self) -> None:
        from seedvision.segmentation.procedural import ProceduralInstanceSettings
        from seedvision.ui.main_window import _ProceduralFitTask

        annotations = np.zeros((8, 8), np.uint16)
        annotations[2:6, 2:6] = 1
        background = np.full((8, 8), 80, np.uint8)
        refined_background = np.full((8, 8), 96, np.uint8)
        layers = SimpleNamespace(
            valid_mask=np.ones((8, 8), np.uint8) * 255,
            foreground_noise_likelihood=None,
            background_likelihood=background,
            refined_background_likelihood=refined_background,
            edge_likelihood=np.zeros((8, 8), np.uint8),
            edge_ridges=np.zeros((8, 8), np.uint8),
            physical_edge_probability=None,
            non_edge_probability=None,
            reference_edge_ridges=object(),
            net_reference_edge_ridges=object(),
            locally_normalized_net_physical_edge=object(),
            normalized_net_reference_edge_ridges=object(),
            edge_trace_labels=object(),
            edge_trace_continuity=object(),
            reference_seed_surface_probability=None,
            darkening_surface_gradient=object(),
        )
        analysis = SimpleNamespace(
            layers=layers,
            estimated_seed_diameter_px=24.0,
            foreground_probability=None,
            advanced=SimpleNamespace(
                rasters={
                    "sensor_noise": None,
                    "shadow_likelihood": None,
                    "flattened_grayscale": object(),
                }
            ),
        )
        task = _ProceduralFitTask(
            analysis,
            annotations,
            ProceduralInstanceSettings(
                reference_error_minimum_match_iou=0.35,
                reference_error_missed_seed_weight=0.25,
                reference_error_concavity_weight=3.5,
            ),
            false_positive_weight=2.5,
            overreach_distance_scale_fraction=0.25,
            annotations_are_complete=True,
            image_key="fixture",
            pipeline_revision=7,
            annotation_source=annotations,
        )
        report = SimpleNamespace(cancelled=False)
        prepared = SimpleNamespace(
            annotation_targets=lambda labels, mask_to_valid: labels,
            seed_diameter_px=19.5,
        )
        with patch(
            "seedvision.ui.main_window.prepare_procedural_instance_inputs",
            return_value=prepared,
        ) as prepare, patch(
            "seedvision.ui.main_window.fit_procedural_settings",
            return_value=report,
        ) as fit:
            task.run()

        options = fit.call_args.kwargs["options"]
        self.assertTrue(options.annotations_are_complete)
        self.assertEqual(options.false_positive_weight, 2.5)
        self.assertEqual(options.overreach_distance_scale_fraction, 0.25)
        self.assertEqual(options.minimum_match_iou, 0.35)
        self.assertEqual(options.missed_seed_weight, 0.25)
        self.assertEqual(options.incorrect_concavity_weight, 3.5)
        self.assertEqual(fit.call_args.kwargs["seed_diameter_px"], 19.5)
        prepare_kwargs = prepare.call_args.kwargs
        self.assertIsNone(prepare_kwargs["material_probability"])
        self.assertIsNone(prepare_kwargs["physical_edge_probability"])
        self.assertIsNone(prepare_kwargs["non_edge_probability"])
        self.assertIsNone(
            prepare_kwargs["normalized_net_physical_edge_probability"]
        )
        self.assertIsNone(prepare_kwargs["thinned_reference_edge_ridges"])
        self.assertIsNone(prepare_kwargs["oriented_edge_trace_labels"])
        self.assertIsNone(prepare_kwargs["oriented_edge_trace_continuity"])
        self.assertIsNone(prepare_kwargs["reference_surface_probability"])
        np.testing.assert_array_equal(
            prepare_kwargs["foreground_probability"],
            np.full((8, 8), 175, np.uint8),
        )
        np.testing.assert_array_equal(
            prepare_kwargs["foreground_noise_probability"],
            np.zeros((8, 8), np.uint8),
        )
        self.assertNotIn("sensor_noise", prepare_kwargs)
        self.assertNotIn("shadow_likelihood", prepare_kwargs)

    def test_instance_boundary_supervision_is_automatic_without_checkbox(self) -> None:
        window, _key = self._window_and_key()
        self.assertFalse(
            hasattr(window, "use_annotated_instance_boundaries_checkbox")
        )
        self.assertIn(
            "automatically supply physical contours",
            window.instance_boundary_supervision_label.text(),
        )

    def test_fit_eligibility_requires_applied_connected_annotations(self) -> None:
        window, key = self._window_and_key()
        window.pipeline_inspector.set_node(
            window.pipeline.node("procedural_instances")
        )
        window._analyses[key] = object()
        annotations = np.zeros((12, 12), np.uint16)
        annotations[2:6, 3:7] = 1
        window._applied_instance_annotations[key] = annotations

        window._sync_procedural_fit_controls()
        self.assertTrue(window.pipeline_inspector.procedural_fit_button.isEnabled())

        window._instance_annotations_dirty.add(key)
        window._sync_procedural_fit_controls()
        self.assertFalse(window.pipeline_inspector.procedural_fit_button.isEnabled())
        self.assertIn(
            "Apply + save", window.pipeline_inspector.procedural_fit_status_label.text()
        )

        window._instance_annotations_dirty.discard(key)
        disconnected = np.zeros((12, 12), np.uint16)
        disconnected[1:3, 1:3] = 1
        disconnected[8:10, 8:10] = 1
        window._applied_instance_annotations[key] = disconnected
        window._instance_continuity_cache.pop(key, None)
        window._sync_procedural_fit_controls()
        self.assertFalse(window.pipeline_inspector.procedural_fit_button.isEnabled())
        self.assertIn(
            "disconnected",
            window.pipeline_inspector.procedural_fit_status_label.text().lower(),
        )

    def test_rejecting_completed_fit_keeps_parameters_and_revision(self) -> None:
        from PySide6.QtWidgets import QMessageBox

        window, key = self._window_and_key()
        annotations = np.zeros((10, 10), np.uint16)
        annotations[2:8, 2:8] = 1
        self._install_completed_task(window, key, annotations)
        window._analyses[key] = object()
        initial_settings = window._procedural_settings()
        report = self._fit_report(initial_settings)
        initial_parameters = dict(
            window.pipeline.node("procedural_instances").parameters
        )
        initial_revision = window.pipeline.revision
        pending: list[bool] = []
        window._start_pending_analysis = lambda: pending.append(True)

        with patch(
            "seedvision.ui.main_window.QMessageBox.question",
            return_value=QMessageBox.StandardButton.No,
        ):
            window._procedural_fit_completed(report)

        self.assertEqual(
            window.pipeline.node("procedural_instances").parameters,
            initial_parameters,
        )
        self.assertEqual(window.pipeline.revision, initial_revision)
        self.assertEqual(pending, [True])
        self.assertIn(
            "Kept current settings",
            window.pipeline_inspector.procedural_fit_status_label.text(),
        )
        self.assertIn(
            "disjoint predictions unscored",
            window.pipeline_inspector.procedural_fit_status_label.text(),
        )

    def test_completed_whole_dish_fit_labels_reported_metrics(self) -> None:
        window, key = self._window_and_key()
        annotations = np.zeros((10, 10), np.uint16)
        annotations[2:8, 2:8] = 1
        self._install_completed_task(
            window,
            key,
            annotations,
            annotations_are_complete=True,
        )
        window._analyses[key] = object()
        report = self._fit_report(window._procedural_settings(), improved=False)
        window._start_pending_analysis = lambda: None

        with patch("seedvision.ui.main_window.QMessageBox.information"):
            window._procedural_fit_completed(report)

        self.assertIn(
            "Whole-dish metrics",
            window.pipeline_inspector.procedural_fit_status_label.text(),
        )

    def test_accepting_completed_fit_batches_one_update_and_recompute(self) -> None:
        from PySide6.QtWidgets import QMessageBox

        from seedvision.segmentation.baseline import PipelineAnalysisCache

        window, key = self._window_and_key()
        annotations = np.zeros((10, 10), np.uint16)
        annotations[2:8, 2:8] = 1
        self._install_completed_task(window, key, annotations)
        window._analyses[key] = object()
        window._analysis_caches[key] = PipelineAnalysisCache()
        initial_settings = window._procedural_settings()
        report = self._fit_report(initial_settings)
        initial_revision = window.pipeline.revision
        analyses: list[set[str]] = []
        window._analyze_current_image = lambda **kwargs: analyses.append(
            set(kwargs.get("dirty_nodes", ()))
        )
        original_set_parameters = window.pipeline.set_parameters
        calls: list[tuple[str, dict[str, object]]] = []

        def set_parameters(node_id, values):
            calls.append((node_id, dict(values)))
            return original_set_parameters(node_id, values)

        window.pipeline.set_parameters = set_parameters
        with patch(
            "seedvision.ui.main_window.QMessageBox.question",
            return_value=QMessageBox.StandardButton.Yes,
        ):
            window._procedural_fit_completed(report)

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], "procedural_instances")
        self.assertEqual(window.pipeline.revision, initial_revision + 1)
        self.assertAlmostEqual(
            window.pipeline.node("procedural_instances").parameters[
                "foreground_threshold_scale"
            ],
            report.proposed_settings.foreground_threshold_scale,
        )
        self.assertEqual(len(analyses), 1)
        self.assertIn("procedural_instances", analyses[0])
        self.assertIn("procedural_instances", window._cache_dirty_nodes[key])
        self.assertIn(
            "Applied fitted settings",
            window.pipeline_inspector.procedural_fit_status_label.text(),
        )

    def test_stale_completed_fit_is_discarded_without_prompt_or_mutation(self) -> None:
        window, key = self._window_and_key()
        annotations = np.zeros((10, 10), np.uint16)
        annotations[2:8, 2:8] = 1
        self._install_completed_task(window, key, annotations)
        window._procedural_fit_task.pipeline_revision -= 1
        window._analyses[key] = object()
        initial_settings = window._procedural_settings()
        report = self._fit_report(initial_settings)
        initial_parameters = dict(
            window.pipeline.node("procedural_instances").parameters
        )
        initial_revision = window.pipeline.revision
        pending: list[bool] = []
        window._start_pending_analysis = lambda: pending.append(True)

        with patch("seedvision.ui.main_window.QMessageBox.question") as question:
            window._procedural_fit_completed(report)

        question.assert_not_called()
        self.assertEqual(
            window.pipeline.node("procedural_instances").parameters,
            initial_parameters,
        )
        self.assertEqual(window.pipeline.revision, initial_revision)
        self.assertEqual(pending, [True])
        self.assertIn(
            "Discarded a fit",
            window.pipeline_inspector.procedural_fit_status_label.text(),
        )


if __name__ == "__main__":
    unittest.main()
