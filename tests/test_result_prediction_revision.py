"""Exact prediction facts must travel with review and exported revisions."""

import csv
import json
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np


def _analysis(source=None):
    labels = np.zeros((16, 16), np.int32)
    labels[4:8, 4:8] = 1
    return SimpleNamespace(image_path=source, crop_offset=(0, 0), warnings=(),
        calibration=SimpleNamespace(corrected_bgr=np.zeros((16, 16, 3), np.uint8),
                                    affine_matrix=np.eye(3)),
        layers=SimpleNamespace(foreground_colour_profile=object()),
        procedural_instances=SimpleNamespace(labels=labels, source_shape=(16, 16),
            instance_confidences=np.array([.8]), candidate_scores=np.array([.7, .6]),
            candidate_selected=np.array([True, False]),
            rejected_manual_centres_xy=np.array([[2., 3.]]),
            rejected_manual_centre_reasons=('outside_foreground',)))


class ResultPredictionRevisionTests(unittest.TestCase):
    def test_every_reported_prediction_fact_stales_old_review_with_unchanged_labels(self):
        from seedvision.export.results import result_report, RESULT_REVISION_CONTRACT
        mutations = {
            'confidence': lambda i: i.instance_confidences.__setitem__(0, .3),
            'candidate_score': lambda i: i.candidate_scores.__setitem__(1, .2),
            'candidate_selection': lambda i: i.candidate_selected.__setitem__(1, True),
            'rejected_xy': lambda i: i.rejected_manual_centres_xy.__setitem__((0, 0), 9.),
            'rejected_reason': lambda i: setattr(i, 'rejected_manual_centre_reasons', ('duplicate',)),
        }
        for name, mutate in mutations.items():
            with self.subTest(fact=name):
                analysis = _analysis()
                before = result_report(analysis)
                review = {'result_revision': before['result_revision'], 'decisions': {'1': 'accepted'}}
                self.assertEqual(result_report(analysis, review=review)['accepted_count'], 1)
                labels = analysis.procedural_instances.labels.copy()
                mutate(analysis.procedural_instances)
                after = result_report(analysis, review=review)
                np.testing.assert_array_equal(analysis.procedural_instances.labels, labels)
                self.assertNotEqual(after['result_revision'], before['result_revision'])
                self.assertEqual(after['accepted_count'], 0)
                self.assertEqual(after['instances'][0]['review'], 'unreviewed')
                self.assertEqual(after['schema_version'], 3)
                self.assertEqual(after['identity']['revision_contract'], RESULT_REVISION_CONTRACT)

    def test_exact_dtype_and_values_bind_revision_but_array_layout_does_not(self):
        from seedvision.export.results import result_report
        analysis = _analysis()
        analysis.procedural_instances.instance_confidences = np.array([.5], np.float64)
        first = result_report(analysis)
        analysis.procedural_instances.instance_confidences = np.array([.5, 99.])[::2]
        self.assertEqual(result_report(analysis)['result_revision'], first['result_revision'])
        analysis.procedural_instances.instance_confidences = np.array([.5], np.float32)
        changed = result_report(analysis)
        self.assertEqual(changed['instances'][0]['uncalibrated_instance_score'], .5)
        self.assertNotEqual(changed['result_revision'], first['result_revision'])

    def test_report_uses_detached_compact_prediction_values(self):
        from seedvision.export.results import result_report
        analysis = _analysis()
        report = result_report(analysis)
        serialized = json.dumps(report, sort_keys=True)
        instances = analysis.procedural_instances
        instances.instance_confidences[0] = .1
        instances.candidate_scores[:] = .1
        instances.candidate_selected[:] = False
        instances.rejected_manual_centres_xy[:] = 42
        self.assertEqual(json.dumps(report, sort_keys=True), serialized)
        self.assertEqual(report['instances'][0]['uncalibrated_instance_score'], .8)
        self.assertEqual(report['candidate_diagnostics']['candidate_scores'], [.7, .6])
        self.assertEqual(report['rejected_manual_centres'][0]['xy_px'], [2., 3.])

    def test_malformed_prediction_facts_are_unreviewable(self):
        from seedvision.export.results import result_report
        mutations = (
            ('instance_confidences', np.array([np.nan])),
            ('instance_confidences', np.empty(0)),
            ('candidate_scores', np.array(['bad'], object)),
            ('candidate_scores', np.array([.1])),
            ('candidate_selected', np.array([[True, False]])),
            ('candidate_selected', np.array([np.nan, 0.])),
            ('rejected_manual_centres_xy', np.array([[1., 2., 3.]])),
            ('rejected_manual_centre_reasons', ()),
            ('rejected_manual_centre_reasons', (123,)),
        )
        for field, value in mutations:
            with self.subTest(field=field, value=str(value)):
                analysis = _analysis()
                old = result_report(analysis)
                setattr(analysis.procedural_instances, field, value)
                with self.assertRaises(ValueError):
                    result_report(analysis, review={'result_revision': old['result_revision'],
                                                  'decisions': {'1': 'accepted'}})

    def test_empty_and_unavailable_remain_distinct_stable_states(self):
        from seedvision.export.results import result_report
        analysis = _analysis()
        analysis.procedural_instances.labels[:] = 0
        analysis.procedural_instances.instance_confidences = np.empty(0)
        empty = result_report(analysis)
        self.assertEqual(empty['state'], 'available_for_review')
        self.assertEqual(empty['proposed_count'], 0)
        self.assertEqual(result_report(analysis)['result_revision'], empty['result_revision'])
        analysis.layers.foreground_colour_profile = None
        unavailable = result_report(analysis, review={'result_revision': empty['result_revision'],
                                                    'decisions': {'1': 'accepted'}})
        self.assertEqual(unavailable['state'], 'unavailable')
        self.assertIsNone(unavailable['proposed_count'])
        self.assertNotEqual(unavailable['result_revision'], empty['result_revision'])
        self.assertEqual(unavailable['accepted_count'], 0)

    def test_json_csv_exports_preserve_the_same_prediction_snapshot(self):
        from seedvision.export.results import result_report, export_report
        analysis = _analysis()
        report = result_report(analysis)
        with TemporaryDirectory() as directory:
            output = export_report(directory, analysis, report)
            saved = json.loads((output / 'results.json').read_text())
            with (output / 'instances.csv').open(encoding='utf-8-sig', newline='') as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(saved['result_revision'], report['result_revision'])
            self.assertEqual(saved['identity']['prediction_facts'], report['identity']['prediction_facts'])
            self.assertEqual(float(row['uncalibrated_instance_score']), .8)
            self.assertEqual(saved['candidate_diagnostics'], report['candidate_diagnostics'])
            self.assertFalse(saved['scientifically_validated'])

    def test_desktop_score_change_uses_new_review_file_and_preserves_old_decisions(self):
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
        from PySide6.QtWidgets import QApplication
        from seedvision.ui.main_window import MainWindow
        app = QApplication.instance() or QApplication([])
        with TemporaryDirectory() as directory:
            window = MainWindow(Path(directory))
            try:
                analysis = _analysis()
                window._analyses[None] = analysis
                old = window._current_report(analysis)
                old_path = window._review_path(old)
                old_path.parent.mkdir(parents=True, exist_ok=True)
                review = {'result_revision': old['result_revision'], 'decisions': {'1': 'accepted'}}
                old_path.write_text(json.dumps(review), encoding='utf-8')
                self.assertEqual(window._current_report(analysis)['accepted_count'], 1)
                analysis.procedural_instances.instance_confidences[0] = .2
                fresh = window._current_report(analysis)
                self.assertEqual(fresh['accepted_count'], 0)
                self.assertNotEqual(window._review_path(fresh), old_path)
                self.assertEqual(json.loads(old_path.read_text()), review)
            finally:
                window.close()
                app.processEvents()

    def test_saved_project_batch_reports_score_and_diagnostic_changes_with_new_revision(self):
        from seedvision.persistence import ProjectAnalysisStore, ProjectImageSpec, analysis_settings_profile_from_graph
        from seedvision.pipeline import build_default_pipeline
        from seedvision.export.project_runner import run_project
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'config').mkdir()
            shutil.copyfile(Path(__file__).parents[1] / 'config' / 'traits.json', root / 'config' / 'traits.json')
            source = root / 'source.png'
            cv2.imwrite(str(source), np.zeros((16, 16, 3), np.uint8))
            master = ProjectAnalysisStore(root).capture_and_save(
                analysis_settings=analysis_settings_profile_from_graph(build_default_pipeline()),
                images=(ProjectImageSpec(source, (16, 16)),), destination=root / 'project.seedfiddle-project.json')
            analysis = _analysis(source)
            with patch('seedvision.export.project_runner.analyze_path', return_value=analysis):
                first = run_project(root, master, root / 'output')[0]
                analysis.procedural_instances.instance_confidences[0] = .2
                analysis.procedural_instances.candidate_selected[1] = True
                second = run_project(root, master, root / 'output')[0]
            before = json.loads((first / 'results.json').read_text())
            after = json.loads((second / 'results.json').read_text())
            self.assertNotEqual(before['result_revision'], after['result_revision'])
            self.assertEqual(before['instances'][0]['uncalibrated_instance_score'], .8)
            self.assertEqual(after['instances'][0]['uncalibrated_instance_score'], .2)
            self.assertEqual(after['candidate_diagnostics']['selected'], [True, True])


class PredictionRevisionCropTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        if not torch.cuda.is_available():
            raise unittest.SkipTest('CUDA is required for seed crop export integration.')

    def test_confidence_only_stale_crop_export_rejects_before_output(self):
        from tests.test_seed_crop_artifacts import _analysis as crop_analysis
        from seedvision.export.results import result_report
        from seedvision.export.crop_artifacts import export_seed_crop_artifacts
        analysis = crop_analysis()
        report = result_report(analysis)
        analysis.procedural_instances.instance_confidences[0] = .2
        with TemporaryDirectory() as directory:
            output = Path(directory) / 'output'
            with self.assertRaisesRegex(ValueError, 'Result changed'):
                export_seed_crop_artifacts(output, analysis, report, seed_ids=(1,), padding_px=0)
            self.assertFalse(output.exists())

    def test_confidence_change_during_encoding_prevents_crop_publication(self):
        from tests.test_seed_crop_artifacts import _analysis as crop_analysis
        from seedvision.export.results import result_report
        from seedvision.export.crop_artifacts import export_seed_crop_artifacts
        analysis = crop_analysis()
        report = result_report(analysis)
        encode = cv2.imwrite
        def changed(path, raster):
            analysis.procedural_instances.instance_confidences[0] = .2
            return encode(path, raster)
        with TemporaryDirectory() as directory:
            with patch('seedvision.export.crop_artifacts.cv2.imwrite', side_effect=changed):
                with self.assertRaisesRegex(ValueError, 'Result changed during'):
                    export_seed_crop_artifacts(directory, analysis, report, seed_ids=(1,), padding_px=0)
            self.assertFalse(list(Path(directory).glob('seed-crops-*')))


if __name__ == '__main__':
    unittest.main()
