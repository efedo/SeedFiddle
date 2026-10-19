"""Actual frozen CUDA trait predictions must reach the native review widgets."""

from tests.functional_units import functional_units
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

import tests.test_trait_classifier as core_fixture
from seedvision.export.results import export_report
from seedvision.learning.trait_pipeline import TraitClassifierSettings, run_trait_classifier


@unittest.skipUnless(torch.cuda.is_available(), 'Trait review integration requires CUDA')
@functional_units
class TraitReviewUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])
        core_fixture.TraitClassifierTests.setUpClass()
        cls.model = core_fixture.TraitClassifierTests.model

    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root / 'traits.json'
        self.path.write_text(self.model.to_json(), encoding='utf-8')
        self.labels = np.zeros((12, 12), np.int32)
        self.labels[2:4, 2:4] = 1
        self.labels[7:9, 7:9] = 2
        self.corrected = np.zeros((24, 24, 3), np.uint8)
        self.corrected[4:8, 4:8, 2] = 30
        self.corrected[14:18, 14:18, 2] = 170
        self.instances = SimpleNamespace(labels=self.labels, source_shape=(24, 24),
                                         instance_confidences=np.array([.8, .9]))
        self.source = torch.tensor(self.corrected, device='cuda').float().permute(2, 0, 1)[None]
        self.valid = torch.ones((1, 1, 24, 24), device='cuda', dtype=torch.bool)
        self.cache = {}
        from seedvision.ui.main_window import MainWindow
        self.window = MainWindow(self.root)
        self.addCleanup(self.window.close)
        self.window.pipeline.node('trait_classifier').enabled = True

    def classify(self, threshold=0, path=None):
        selected = str(self.path if path is None else path)
        self.window.pipeline.set_parameters('trait_classifier', {
            'trait_model_path': selected, 'trait_abstention_threshold': threshold})
        prediction, _ = run_trait_classifier('procedural_instances', self.instances,
            corrected_bgr=self.corrected, crop_offset=(0, 0), affine=np.eye(3),
            gpu_source=self.source, gpu_valid=self.valid,
            settings=TraitClassifierSettings(selected, threshold), root=self.root, cache=self.cache)
        self.analysis = SimpleNamespace(image_path=None, crop_offset=(0, 0), warnings=(),
            calibration=SimpleNamespace(corrected_bgr=self.corrected, affine_matrix=np.eye(3)),
            layers=SimpleNamespace(foreground_colour_profile=object()),
            procedural_instances=self.instances, trait_classifier=prediction)
        self.window._analyses[None] = self.analysis
        return self.window._current_report(self.analysis)

    def observe_dialog(self, exercise=None):
        from PySide6.QtWidgets import QDialog, QLabel, QTableWidget
        observed = {}

        def inspect(dialog):
            table = dialog.findChild(QTableWidget)
            column = next(i for i in range(table.columnCount())
                          if table.horizontalHeaderItem(i).text() == 'Predicted trait')
            observed['rows'] = {int(table.item(i, 0).text()): {
                'text': table.item(i, column).text(), 'tooltip': table.item(i, column).toolTip(),
                'review': table.cellWidget(i, 1).currentText()}
                for i in range(table.rowCount())}
            observed['summary'] = next(item.text() for item in dialog.findChildren(QLabel)
                                       if 'Trait predictions:' in item.text())
            if exercise:
                exercise(dialog, table)
            return QDialog.DialogCode.Rejected

        with patch.object(QDialog, 'exec', new=inspect), \
             patch('seedvision.learning.trait_pipeline.run_trait_classifier',
                   side_effect=AssertionError('Review recomputed classifier')):
            self.window._review_results()
        return observed

    def check_test_actual_mixed_predictions_definitions_and_scores_reach_review_without_refitting(self):
        baseline = self.classify()
        confidences = [row['trait_prediction']['confidence'] for row in baseline['instances']]
        self.assertNotEqual(*confidences)
        report = self.classify(sum(confidences) / 2)
        self.assertEqual(sum(row['trait_prediction']['abstained'] for row in report['instances']), 1)
        before = json.dumps(report, sort_keys=True)
        observed = self.observe_dialog()
        vocabulary = self.model.payload['vocabulary']['traits']
        for row in report['instances']:
            cell = observed['rows'][row['seed_id']]
            prediction = row['trait_prediction']
            if prediction['abstained']:
                self.assertIn('abstained', cell['text'].lower())
                self.assertIn(prediction['abstention_reason'].replace('_', ' '), cell['text'])
                self.assertIn(prediction['abstention_reason'], cell['tooltip'])
            else:
                self.assertEqual(cell['text'], prediction['predicted_trait_id'])
                definition = next(item for item in vocabulary
                                  if item['identifier'] == prediction['predicted_trait_id'])
                self.assertIn(definition['definition'], cell['tooltip'])
                self.assertIn(definition['evidence_reference'], cell['tooltip'])
            self.assertIn(f"{prediction['confidence']:.3f}", cell['tooltip'])
            self.assertIn('not a probability of correctness', cell['tooltip'])
            self.assertIn(self.model.model_sha256, cell['tooltip'])
        self.assertIn('1 assigned, 1 abstained', observed['summary'])
        self.assertNotIn('trait assignments withheld', observed['summary'])
        self.assertEqual(json.dumps(self.window._current_report(self.analysis), sort_keys=True), before)
        destination = export_report(self.root / 'exports', self.analysis, report)
        stored = json.loads((destination / 'results.json').read_text(encoding='utf-8'))
        self.assertEqual(stored['trait_classifier'], report['trait_classifier'])
        self.assertEqual(stored['instances'], report['instances'])
        self.assertFalse(stored['scientifically_validated'])

    def check_test_actual_absent_invalid_and_outside_support_abstentions_stay_explanatory(self):
        invalid = self.root / 'invalid.json'
        invalid.write_text('{}', encoding='utf-8')
        cases = [('None', 'no_frozen_model'),
                 (self.root / 'missing.json', 'model_file_missing'),
                 (invalid, 'invalid_model'), (self.path, 'outside_valid_crop')]
        for path, expected in cases:
            with self.subTest(reason=expected):
                self.valid.fill_(True)
                if expected == 'outside_valid_crop':
                    self.valid[0, 0, 4, 4] = False
                report = self.classify(path=path)
                observed = self.observe_dialog()
                self.assertEqual(set(observed['rows']), {1, 2})
                row = report['instances'][0]
                self.assertEqual(row['trait_prediction']['abstention_reason'], expected)
                self.assertIn(expected.replace('_', ' '), observed['rows'][1]['text'])
                self.assertIn(expected, observed['rows'][1]['tooltip'])
                if expected != 'outside_valid_crop':
                    self.assertIn('0 assigned, 2 abstained', observed['summary'])
                self.assertFalse(report['scientifically_validated'])

    def check_test_threshold_change_stales_review_preserves_probabilities_and_historical_bytes(self):
        first = self.classify()
        self.assertTrue(all(not row['trait_prediction']['abstained'] for row in first['instances']))
        self.observe_dialog(lambda dialog, table: table.cellWidget(0, 1).setCurrentText('accepted'))
        reviewed = self.window._current_report(self.analysis)
        self.assertEqual(reviewed['accepted_count'], 1)
        review_path = self.window._review_path(reviewed)
        old_bytes = review_path.read_bytes()
        second = self.classify(threshold=1)
        self.assertNotEqual(second['result_revision'], first['result_revision'])
        self.assertEqual(second['accepted_count'], 0)
        self.assertEqual([row['trait_prediction']['probabilities'] for row in first['instances']],
                         [row['trait_prediction']['probabilities'] for row in second['instances']])
        expected = [row['trait_prediction']['confidence'] < 1 for row in first['instances']]
        self.assertTrue(any(expected))
        self.assertEqual([row['trait_prediction']['abstained'] for row in second['instances']], expected)
        observed = self.observe_dialog()
        self.assertTrue(all(row['review'] == 'unreviewed' for row in observed['rows'].values()))
        self.assertIn(f'{2-sum(expected)} assigned, {sum(expected)} abstained', observed['summary'])
        self.assertEqual(review_path.read_bytes(), old_bytes)
        self.assertFalse(self.window._review_path(second).exists())

    def test_workflow_lifecycle(self):
        """1 related TraitReviewUiTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_actual_mixed_predictions_definitions_and_scores_reach_review_without_refitting)

    def test_validation_and_recovery(self):
        """2 related TraitReviewUiTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_actual_absent_invalid_and_outside_support_abstentions_stay_explanatory)
        self.scenario(self.check_test_threshold_change_stales_review_preserves_probabilities_and_historical_bytes)


if __name__ == '__main__':
    unittest.main()
