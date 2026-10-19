"""Actual backend outcomes, cache recovery, Qt/batch parity and immutable review binding."""

from tests.functional_units import functional_units
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from seedvision.export import backend_outcomes as outcomes
from seedvision.export.results import result_report
import tests.test_authoritative_parity as parity_fixture
import tests.test_result_prediction_revision as report_fixture


@functional_units
class BackendOutcomeTests(unittest.TestCase):
    def analysis(self):
        value = report_fixture._analysis()
        value.procedural_instances.centres_xy = np.array([[6., 6.]])
        return value

    def capture(self, analysis, failures=None, enabled=None):
        analysis.backend_outcomes_json = outcomes.capture_native_outcomes(
            enabled={'procedural_instances', 'unet_instances'} if enabled is None else enabled,
            results={node: getattr(analysis, node, None) for node in outcomes.BACKENDS},
            failures=failures or {'unet_instances': ('unavailable', 'checkpoint_missing', None, None)})
        return analysis.backend_outcomes_json

    def check_test_available_empty_missing_and_disabled_are_distinct_and_round_trip(self):
        analysis = self.analysis()
        analysis.procedural_instances.labels.fill(0)
        analysis.procedural_instances.centres_xy = np.zeros((0, 2))
        analysis.procedural_instances.instance_confidences = np.zeros(0)
        raw = self.capture(analysis)
        ledger = outcomes.verified_native_outcomes(raw, analysis)
        records = ledger['outcomes']
        self.assertEqual(records['procedural_instances']['state'], 'available')
        self.assertEqual(records['procedural_instances']['prediction_facts']['output_state'], 'empty')
        self.assertEqual(records['procedural_instances']['prediction_facts']['instance_count'], 0)
        self.assertEqual(records['unet_instances']['state'], 'unavailable')
        self.assertEqual(records['stardist_instances']['state'], 'not_run')
        self.assertEqual(outcomes.read_native_outcomes(json.dumps(json.loads(raw))), ledger)
        self.assertEqual(result_report(analysis)['proposed_count'], 0)

    def check_test_legacy_is_unknown_diagnostics_do_not_hash_and_outcome_changes_stale_review(self):
        analysis = self.analysis()
        legacy = result_report(analysis)
        self.assertEqual(legacy['backend_outcomes']['evidence_state'], 'not_recorded')
        self.assertIsNone(legacy['backend_outcomes']['outcomes'])
        self.capture(analysis, {'unet_instances': ('failed', 'inference_failed', 'ValueError', 'A:/one.pt')})
        first = result_report(analysis)
        review = {'result_revision': first['result_revision'], 'decisions': {'1': 'accepted'}}
        self.assertEqual(result_report(analysis, review=review)['accepted_count'], 1)
        self.capture(analysis, {'unet_instances': ('failed', 'inference_failed', 'ValueError', 'B:/two.pt')})
        second = result_report(analysis, review=review)
        self.assertEqual(first['identity'], second['identity'])
        self.assertNotEqual(first['backend_outcomes'], second['backend_outcomes'])
        self.assertEqual(second['accepted_count'], 1)
        self.capture(analysis)
        self.assertNotEqual(result_report(analysis)['result_revision'], first['result_revision'])
        self.assertEqual(result_report(analysis, review=review)['accepted_count'], 0)

    def check_test_nonselected_prediction_mutation_and_forged_producer_states_reject(self):
        analysis = self.analysis()
        analysis.unet_instances = SimpleNamespace(labels=analysis.procedural_instances.labels.copy(),
            centres_xy=np.array([[6., 6.]]), instance_confidences=np.array([.9]))
        # Explicit available producers, independent of selected method.
        raw = outcomes.capture_native_outcomes(enabled={'procedural_instances', 'unet_instances'},
            results={node: getattr(analysis, node, None) for node in outcomes.BACKENDS})
        analysis.backend_outcomes_json = raw
        result_report(analysis, enabled={'procedural_instances'})
        analysis.unet_instances.instance_confidences[0] = .4
        with self.assertRaisesRegex(ValueError, 'changed after analysis'):
            result_report(analysis, enabled={'procedural_instances'})
        forged = json.loads(raw)
        forged['outcomes']['unet_instances'].update(state='failed', reason_code='decoder_failed',
            prediction_facts=None, error_type='ValueError', diagnostic='attempted failure')
        with self.assertRaisesRegex(ValueError, 'disagrees'):
            outcomes.verified_native_outcomes(json.dumps(forged), analysis)

    def check_test_malformed_inventory_states_counts_frames_and_duplicate_keys_reject(self):
        raw = self.capture(self.analysis())
        mutations = [
            lambda p: p['outcomes'].pop('unet_instances'),
            lambda p: p['outcomes']['unet_instances'].update(reason_code=[]),
            lambda p: p['outcomes']['unet_instances'].update(backend_id='original_SMPTS'),
            lambda p: p['outcomes']['unet_instances'].update(enabled=False),
            lambda p: p['outcomes']['procedural_instances']['prediction_facts'].update(instance_count=True),
            lambda p: p['outcomes']['procedural_instances']['prediction_facts'].update(instance_count=-1),
            lambda p: p['outcomes']['procedural_instances']['prediction_facts'].update(output_state='empty'),
            lambda p: p['outcomes']['procedural_instances']['prediction_facts'].update(source_shape=[0, 2]),
            lambda p: p['outcomes']['procedural_instances']['prediction_facts']['labels'].update(dtype='object'),
            lambda p: p['outcomes']['procedural_instances']['prediction_facts']['centres_xy'].update(shape=[1, 3]),
            lambda p: p['outcomes']['procedural_instances']['prediction_facts']['labels'].update(sha256='x' * 64),
        ]
        for mutate in mutations:
            payload = json.loads(raw)
            mutate(payload)
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                outcomes.read_native_outcomes(json.dumps(payload))
        for malformed in ('{"contract":1,"contract":2}', 'NaN', 'x' * (1024 * 1024 + 1)):
            with self.assertRaises(ValueError):
                outcomes.read_native_outcomes(malformed)
        with self.assertRaisesRegex(ValueError, 'explicit producer evidence'):
            outcomes.capture_native_outcomes(enabled={'unet_instances'}, results={})

    def check_test_missing_foreground_withholds_computed_procedural_prediction_and_cannot_be_refreshed(self):
        analysis = self.analysis()
        analysis.layers.foreground_colour_profile = None
        analysis.backend_outcomes_json = outcomes.capture_native_outcomes(
            enabled={'procedural_instances'}, results={'procedural_instances': analysis.procedural_instances},
            foreground_available=False)
        report = result_report(analysis)
        record = report['backend_outcomes']['outcomes']['procedural_instances']
        self.assertEqual(record['state'], 'unavailable')
        self.assertEqual(record['reason_code'], 'foreground_missing')
        self.assertIsNone(record['prediction_facts'])
        self.assertIsNone(report['proposed_count'])
        analysis.layers.foreground_colour_profile = object()
        with self.assertRaisesRegex(ValueError, 'prerequisite changed'):
            result_report(analysis)

    def test_validation_and_recovery(self):
        """5 related BackendOutcomeTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_available_empty_missing_and_disabled_are_distinct_and_round_trip)
        self.scenario(self.check_test_legacy_is_unknown_diagnostics_do_not_hash_and_outcome_changes_stale_review)
        self.scenario(self.check_test_nonselected_prediction_mutation_and_forged_producer_states_reject)
        self.scenario(self.check_test_malformed_inventory_states_counts_frames_and_duplicate_keys_reject)
        self.scenario(self.check_test_missing_foreground_withholds_computed_procedural_prediction_and_cannot_be_refreshed)


@unittest.skipUnless(torch.cuda.is_available(), 'Native outcome parity requires CUDA')
class NativeBackendOutcomeTests(unittest.TestCase):
    setUpClass = classmethod(parity_fixture.AuthoritativeParityTests.setUpClass.__func__)
    setUp = parity_fixture.AuthoritativeParityTests.setUp
    save_project = parity_fixture.AuthoritativeParityTests.save_project
    install_profile = parity_fixture.AuthoritativeParityTests.install_profile
    decode = staticmethod(parity_fixture.AuthoritativeParityTests.decode)
    paired = parity_fixture.AuthoritativeParityTests.paired
    save_desktop_review = parity_fixture.AuthoritativeParityTests.save_desktop_review

    def enable_unet(self, *, checkpoint=True):
        path = self.root / 'selected.pt'
        if checkpoint:
            path.write_bytes(b'explicit non-executable checkpoint fixture')
        self.graph.set_parameter('unet_instances', 'checkpoint_path', str(path))
        self.graph.set_enabled('unet_instances', True)
        self.save_project()
        self.install_profile()
        return path
