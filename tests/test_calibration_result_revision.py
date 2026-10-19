"""Actual calibration state must travel with authoritative review revisions."""

from tests.functional_units import functional_units
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from seedvision.export.results import result_report, export_report
from tests.test_result_prediction_revision import _analysis
from tests import test_neutral_calibration as neutral_fixture


def analysis():
    fixture = neutral_fixture.NeutralCalibrationTests()
    fixture.setUp()
    result = _analysis()
    result.calibration = fixture.run_calibration()
    return result


@functional_units
class CalibrationResultRevisionTests(unittest.TestCase):

    def check_test_neutral_request_and_witness_changes_bind_revision(self):
        current = analysis(); old = result_report(current)
        calibration = current.calibration
        settings = replace(calibration.calibration_settings, neutral_sample_frame='corrected')
        neutral = replace(calibration.neutral_balance, sample_frame='corrected')
        current.calibration = replace(calibration, calibration_settings=settings, neutral_balance=neutral)
        frame_revision = result_report(current)['result_revision']
        self.assertNotEqual(old['result_revision'], frame_revision)
        witness = current.calibration.neutral_balance.patches[0]
        changed = replace(witness, sampled_mean_bgr=tuple(v+.1 for v in witness.sampled_mean_bgr))
        current.calibration = replace(current.calibration, neutral_balance=replace(neutral, patches=(changed, *neutral.patches[1:])))
        self.assertNotEqual(frame_revision, result_report(current)['result_revision'])

    def check_test_report_calibration_is_detached_and_physical_claims_remain_withheld(self):
        current = analysis(); report = result_report(current)
        snapshot = json.dumps(report, sort_keys=True)
        current.calibration.affine_matrix[0, 2] = 3
        self.assertEqual(json.dumps(report, sort_keys=True), snapshot)
        facts = report['identity']['calibration_facts']
        self.assertEqual(facts['state'], 'recorded')
        self.assertEqual(facts['settings']['neutral_patch_selection'], 'automatic')
        self.assertEqual(facts['seed_valid_region']['state'], 'unknown')
        self.assertFalse(facts['physical_measurements_eligible'])
        self.assertFalse(report['scientifically_validated'])
        self.assertIsNone(report['instances'][0]['physical_length_mm'])
        self.assertIsNone(report['instances'][0]['physical_area_mm2'])

    def check_test_invalid_calibration_numbers_or_inconsistent_neutral_facts_are_unreviewable(self):
        mutations = (lambda c: replace(c, pixels_per_mm=np.nan), lambda c: replace(c, pixels_per_mm=np.inf),
                     lambda c: replace(c, pixels_per_mm=True), lambda c: replace(c, scale_confidence=2.),
                     lambda c: replace(c, channel_gains_bgr=(1., 1., 1.)),
                     lambda c: replace(c, neutral_balance=replace(c.neutral_balance, state='unavailable')),
                     lambda c: replace(c, neutral_balance=replace(c.neutral_balance, patches=())),
                     lambda c: replace(c, affine_matrix=np.full((3, 3), np.nan)))
        for mutate in mutations:
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                current = analysis(); current.calibration = mutate(current.calibration); result_report(current)

    def check_test_legacy_calibration_is_explicit_stable_and_binds_any_recorded_scale(self):
        current = _analysis(); report = result_report(current)
        self.assertEqual(report['identity']['calibration_facts']['state'], 'legacy_unavailable')
        self.assertNotIn('neutral_balance', report['identity']['calibration_facts'])
        self.assertNotIn('settings', report['identity']['calibration_facts'])
        self.assertEqual(result_report(current)['result_revision'], report['result_revision'])
        current.calibration.pixels_per_mm = 2.
        self.assertNotEqual(result_report(current)['result_revision'], report['result_revision'])

    def check_test_export_round_trip_preserves_the_same_calibration_snapshot(self):
        current = analysis(); report = result_report(current)
        with TemporaryDirectory() as directory:
            output = export_report(directory, current, report)
            saved = json.loads((Path(output)/'results.json').read_text(encoding='utf-8'))
        self.assertEqual(saved['identity']['calibration_facts'], report['identity']['calibration_facts'])
        self.assertEqual(saved['result_revision'], report['result_revision'])
        self.assertEqual(saved['schema_version'], 3)


    def test_workflow_lifecycle(self):
        """4 related CalibrationResultRevisionTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_neutral_request_and_witness_changes_bind_revision)
        self.scenario(self.check_test_report_calibration_is_detached_and_physical_claims_remain_withheld)
        self.scenario(self.check_test_legacy_calibration_is_explicit_stable_and_binds_any_recorded_scale)
        self.scenario(self.check_test_export_round_trip_preserves_the_same_calibration_snapshot)

    def test_validation_and_recovery(self):
        """1 related CalibrationResultRevisionTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_invalid_calibration_numbers_or_inconsistent_neutral_facts_are_unreviewable)
