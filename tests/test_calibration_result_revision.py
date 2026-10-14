"""Actual calibration state must travel with authoritative review revisions."""
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


class CalibrationResultRevisionTests(unittest.TestCase):
    def test_scale_confidence_ruler_and_actual_settings_stale_review_without_pixel_changes(self):
        mutations = (lambda c: replace(c, pixels_per_mm=3.),
                     lambda c: replace(c, scale_confidence=.4),
                     lambda c: replace(c, ruler=replace(c.ruler, matched_tick_count=15)),
                     lambda c: replace(c, calibration_settings=replace(c.calibration_settings, ruler_length_mm=160.)),
                     lambda c: replace(c, ruler_in_corrected_coordinates=False),
                     lambda c: replace(c, warnings=('Scale assumption changed.',)))
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                current = analysis(); old = result_report(current)
                review = {'result_revision': old['result_revision'], 'decisions': {'1': 'accepted'}}
                self.assertEqual(result_report(current, review=review)['accepted_count'], 1)
                pixels = current.calibration.corrected_bgr.copy()
                current.calibration = mutate(current.calibration)
                new = result_report(current, review=review)
                np.testing.assert_array_equal(pixels, current.calibration.corrected_bgr)
                self.assertNotEqual(old['result_revision'], new['result_revision'])
                self.assertEqual(new['accepted_count'], 0)

    def test_neutral_request_and_witness_changes_bind_revision(self):
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

    def test_report_calibration_is_detached_and_physical_claims_remain_withheld(self):
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

    def test_invalid_calibration_numbers_or_inconsistent_neutral_facts_are_unreviewable(self):
        mutations = (lambda c: replace(c, pixels_per_mm=np.nan), lambda c: replace(c, pixels_per_mm=np.inf),
                     lambda c: replace(c, pixels_per_mm=True), lambda c: replace(c, scale_confidence=2.),
                     lambda c: replace(c, channel_gains_bgr=(1., 1., 1.)),
                     lambda c: replace(c, neutral_balance=replace(c.neutral_balance, state='unavailable')),
                     lambda c: replace(c, neutral_balance=replace(c.neutral_balance, patches=())),
                     lambda c: replace(c, affine_matrix=np.full((3, 3), np.nan)))
        for mutate in mutations:
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                current = analysis(); current.calibration = mutate(current.calibration); result_report(current)

    def test_legacy_calibration_is_explicit_stable_and_binds_any_recorded_scale(self):
        current = _analysis(); report = result_report(current)
        self.assertEqual(report['identity']['calibration_facts']['state'], 'legacy_unavailable')
        self.assertNotIn('neutral_balance', report['identity']['calibration_facts'])
        self.assertNotIn('settings', report['identity']['calibration_facts'])
        self.assertEqual(result_report(current)['result_revision'], report['result_revision'])
        current.calibration.pixels_per_mm = 2.
        self.assertNotEqual(result_report(current)['result_revision'], report['result_revision'])

    def test_export_round_trip_preserves_the_same_calibration_snapshot(self):
        current = analysis(); report = result_report(current)
        with TemporaryDirectory() as directory:
            output = export_report(directory, current, report)
            saved = json.loads((Path(output)/'results.json').read_text(encoding='utf-8'))
        self.assertEqual(saved['identity']['calibration_facts'], report['identity']['calibration_facts'])
        self.assertEqual(saved['result_revision'], report['result_revision'])
        self.assertEqual(saved['schema_version'], 3)

    def test_actual_detected_card_and_ruler_facts_are_report_serializable(self):
        from seedvision.calibration.image import calibrate_image, CalibrationSettings
        from tests import test_image_calibration as image_fixture

        image = image_fixture.ImageCalibrationTests._synthetic_calibration_image(rotation_degrees=2.)
        current = _analysis()
        current.calibration = calibrate_image(image, CalibrationSettings(
            neutral_patch_selection='1,5,9', neutral_sample_frame='corrected'))
        self.assertIsNotNone(current.calibration.colour_card)
        self.assertIsNotNone(current.calibration.ruler)
        report = result_report(current)
        facts = report['identity']['calibration_facts']
        self.assertEqual(facts['state'], 'recorded')
        self.assertEqual(facts['neutral_balance']['state'], 'applied')
        self.assertEqual(facts['neutral_balance']['used_patch_ids'], [1, 5, 9])
        self.assertGreater(facts['pixels_per_mm'], 0)
        self.assertEqual(json.loads(json.dumps(facts, allow_nan=False)), facts)
