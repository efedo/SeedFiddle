"""Production integration on two generated images; no accuracy claims."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import os
from unittest.mock import patch

import cv2
import numpy as np
import torch

from seedvision.pipeline import build_default_pipeline
from seedvision.optimization.runtime import Sample, ProductionEvaluator, production_settings
from seedvision.optimization.search import optimize, apply_report
from seedvision.persistence.reference_regions import SeedInstanceAnnotation
from seedvision.segmentation.baseline import analyze_path


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA production pipeline required')
class ProductionOptimizationTests(unittest.TestCase):
    def test_ui_samples_fit_local_references_without_a_library_pin(self):
        """One bounded real UI -> frozen samples -> production fit witness."""
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
        from PySide6.QtWidgets import QApplication
        from seedvision.ui.main_window import MainWindow, _path_identity
        from seedvision.ui.optimization import FitTask
        from tests.review_fixtures import isolated_window, dispose_window
        app = QApplication.instance() or QApplication([])
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(MainWindow, '_load_workspace_images'):
                window = isolated_window(Path(__file__).resolve().parents[1], settings_path=root / 'ui.ini')
            try:
                window._project_tracking_enabled = False
                window._project_species_library_pin = None
                keep = {'project', 'background_likelihood', 'layout_detection', 'seed_scale_estimation'}
                for node in window.pipeline.nodes.values():
                    node.enabled = node.identifier in keep
                window.pipeline.node('background_likelihood').set_parameter('background_colour_components', 2)
                window.pipeline.node('background_likelihood').set_parameter('foreground_reference_components', 2)
                window.pipeline.node('layout_detection').set_parameter('expected_outer_diameter_mm', 0)
                window._image_paths.clear()
                for index in range(2):
                    image = np.full((96, 128, 3), 180 + index * 5, np.uint8)
                    labels = np.zeros((96, 128), np.uint16)
                    cv2.ellipse(labels, (60 + index * 3, 45), (17, 11), 0, 0, 360, 1, -1)
                    image[labels > 0] = (45 + index * 8, 80, 120)
                    path = root / f'local_{index}.png'
                    self.assertTrue(cv2.imwrite(str(path), image))
                    key = _path_identity(path)
                    window._image_paths[key] = path
                    window._applied_instance_annotations[key] = labels
                    window._applied_seed_annotations[key] = {1: SeedInstanceAnnotation(
                        1, shape_reviewed=True, outline_visibility='complete')}
                    background = np.zeros_like(labels, bool)
                    background[8:20, 8:30] = True
                    window._applied_background_reference_masks[key] = background
                    window._reference_transforms[key] = np.eye(3)
                samples = window._optimization_samples()
                self.assertEqual(len(samples), 2)
                self.assertTrue(all(sample.kwargs['species_library'] is None for sample in samples))
                task = FitTask(window.pipeline, samples, {'background_likelihood'}, passes=1, budget=2)
                reports, failures = [], []
                task.signals.completed.connect(reports.append)
                task.signals.failed.connect(failures.append)
                task.run()
                app.processEvents()
                self.assertFalse(failures, failures)
                self.assertEqual(len(reports), 1)
                report = reports[0]
                self.assertFalse(report.cancelled)
                row = next(row for row in report.nodes if row.node_id == 'background_likelihood')
                self.assertIn(row.status, {'improved', 'unchanged'})
                self.assertEqual(set(row.before), set(window._image_paths))
                self.assertEqual(set(row.after), set(row.before))
                self.assertTrue(all(np.isfinite(value) for value in row.before.values()))
                self.assertGreaterEqual(len(row.trials), 2)
                self.assertIn('scores', row.trials[1], row.trials[1])
                self.assertEqual(set(row.trials[1]['scores']), set(row.before))
                self.assertTrue(all(np.isfinite(value) for value in row.trials[1]['scores'].values()))
                self.assertEqual(len(report.provenance['sources']), 2)
                self.assertEqual(report.provenance['libraries'], ['none', 'none'])
                self.assertFalse(report.provenance.get('excluded_images', {}).get('background_likelihood'))
            finally:
                window._instance_annotations_dirty.clear()
                window._reference_masks_dirty.clear()
                window._unsaved_reference_sidecars.clear()
                dispose_window(window)


if __name__ == '__main__':
    unittest.main()
