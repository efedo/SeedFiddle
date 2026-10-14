"""Actual backend outcomes, cache recovery, Qt/batch parity and immutable review binding."""
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

    def test_available_empty_missing_and_disabled_are_distinct_and_round_trip(self):
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

    def test_legacy_is_unknown_diagnostics_do_not_hash_and_outcome_changes_stale_review(self):
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

    def test_nonselected_prediction_mutation_and_forged_producer_states_reject(self):
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

    def test_malformed_inventory_states_counts_frames_and_duplicate_keys_reject(self):
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

    def test_missing_foreground_withholds_computed_procedural_prediction_and_cannot_be_refreshed(self):
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

    def test_real_routes_preserve_missing_checkpoint_and_successful_fallback_then_reject_stale_availability(self):
        path = self.enable_unet(checkpoint=False)
        with patch('seedvision.segmentation.baseline.predict_pipeline_model', side_effect=AssertionError('missing checkpoint ran')):
            desktop, batch = self.paired()
        ledger = desktop['backend_outcomes']
        self.assertEqual(ledger, batch['backend_outcomes'])
        self.assertEqual(ledger['selected_backend'], 'procedural_instances')
        self.assertEqual(ledger['outcomes']['unet_instances']['state'], 'unavailable')
        self.assertEqual(ledger['outcomes']['procedural_instances']['prediction_facts']['instance_count'], 2)
        self.assertEqual(ledger['outcomes']['stardist_instances']['state'], 'not_run')
        path.write_bytes(b'newly available selection')
        with self.assertRaisesRegex(ValueError, 'changed after analysis'):
            self.window._current_report(self.result)

    def test_actual_failed_inference_is_visible_in_qt_comparison_with_path_diagnostics_outside_identity(self):
        self.enable_unet()
        with patch('seedvision.segmentation.baseline.predict_pipeline_model',
                   side_effect=[ValueError('batch/model.pt failed'), ValueError('desktop/model.pt failed')]):
            desktop, batch = self.paired()
        self.assertEqual(desktop['method'], 'procedural_instances')
        for report in (desktop, batch):
            record = report['backend_outcomes']['outcomes']['unet_instances']
            self.assertEqual(record['state'], 'failed')
            self.assertEqual(record['reason_code'], 'inference_failed')
            self.assertIsNone(record['prediction_facts'])
        self.assertNotEqual(desktop['backend_outcomes'], batch['backend_outcomes'])
        self.assertEqual(desktop['identity'], batch['identity'])
        from PySide6.QtWidgets import QDialog, QLabel, QPushButton, QMessageBox
        observed = []
        def inspect(dialog):
            observed.extend(label.text() for label in dialog.findChildren(QLabel))
            next(button for button in dialog.findChildren(QPushButton)
                 if button.text().startswith('Compare completed')).click()
            return QDialog.DialogCode.Rejected
        with patch.object(QDialog, 'exec', new=inspect), patch.object(QMessageBox, 'information') as information:
            self.window._review_results()
        self.assertTrue(any('U-Net: failed' in text and 'desktop/model.pt failed' in text for text in observed))
        self.assertIn('U-Net: failed', information.call_args.args[2])
        self.assertIn('StarDist: not run', information.call_args.args[2])
        self.window._mark_analysis_complete(self.result)
        from seedvision.pipeline import NodeStatus
        self.assertEqual(self.window.pipeline.node('unet_instances').status, NodeStatus.FAILED)

    def test_actual_decoder_failure_remains_distinct_and_empty_success_is_available(self):
        self.enable_unet()
        with (patch('seedvision.segmentation.baseline.predict_pipeline_model', return_value=(object(), 'opaque_fixture')),
              patch('seedvision.segmentation.baseline.decode_pipeline_model', side_effect=ValueError('decode failed'))):
            desktop, _ = self.paired()
        self.assertEqual(desktop['backend_outcomes']['outcomes']['unet_instances']['reason_code'], 'decoder_failed')
        from seedvision.learning.contracts import LearnedInstanceResult, ModelFamily
        empty = LearnedInstanceResult(ModelFamily.UNET_WATERSHED, np.zeros((52, 52), np.int32),
            np.zeros((0, 2), np.float32), np.zeros(0, np.float32), checkpoint_id='opaque_fixture')
        with (patch('seedvision.segmentation.baseline.predict_pipeline_model', return_value=(object(), 'opaque_fixture')),
              patch('seedvision.segmentation.baseline.decode_pipeline_model', return_value=empty)):
            desktop, _ = self.paired()
        self.assertEqual(desktop['method'], 'unet_instances')
        self.assertEqual(desktop['proposed_count'], 0)
        self.assertEqual(desktop['backend_outcomes']['outcomes']['unet_instances']['state'], 'available')
        self.assertEqual(desktop['backend_outcomes']['outcomes']['unet_instances']['prediction_facts']['output_state'], 'empty')

    def test_actual_invalid_checkpoint_load_preserves_failure_and_format_unproven_bytes(self):
        self.enable_unet()
        desktop, batch = self.paired()
        self.assertEqual(desktop['backend_outcomes'], batch['backend_outcomes'])
        record = desktop['backend_outcomes']['outcomes']['unet_instances']
        self.assertEqual(record['state'], 'failed')
        self.assertEqual(record['reason_code'], 'inference_failed')
        self.assertIsNotNone(record['error_type'])
        self.assertIsNone(record['prediction_facts'])
        asset = desktop['identity']['native_artifact_selections']['unet_instances']
        self.assertEqual(asset['state'], 'bytes_recorded_format_unproven')
        self.assertFalse(asset['used'])

    def test_success_failure_recovery_and_disable_clear_only_own_cache_and_stale_reviews(self):
        checkpoint_path = self.enable_unet()
        from seedvision.learning.contracts import LearnedInstanceResult, ModelFamily
        from seedvision.segmentation.baseline import analyze_path, PipelineAnalysisCache
        from seedvision.pipeline.settings import production_settings
        settings = production_settings(self.graph)
        enabled = frozenset(node for node, value in self.graph.nodes.items() if value.enabled)
        settings.update(enabled_nodes=enabled, learning_root=self.root, foreground_reference_mask=self.labels > 0,
                        reference_transform=self.reference_transform)
        cache = PipelineAnalysisCache()
        learned = LearnedInstanceResult(ModelFamily.UNET_WATERSHED, np.ones((52, 52), np.int32),
            np.array([[26., 26.]], np.float32), np.array([.8], np.float32), checkpoint_id='opaque_fixture')
        with (patch('seedvision.segmentation.baseline.predict_pipeline_model', return_value=(object(), 'opaque_fixture')) as infer,
              patch('seedvision.segmentation.baseline.decode_pipeline_model', return_value=learned)):
            first = analyze_path(self.source, **settings, node_cache=cache)
            calibration = cache.values['calibration']
            first_capture = first.backend_outcomes_json
            second = analyze_path(self.source, **settings, node_cache=cache)
            self.assertEqual(infer.call_count, 1)
            self.assertIn('unet_instances', cache.last_reused_nodes)
            self.assertEqual(first_capture, second.backend_outcomes_json)
        before = result_report(second)
        review = {'result_revision': before['result_revision'], 'decisions': {'1': 'accepted'}}
        changed = dict(settings, unet_settings=replace(settings['unet_settings'], tile_size=256))
        with patch('seedvision.segmentation.baseline.predict_pipeline_model', side_effect=ValueError('new inference failed')):
            failed = analyze_path(self.source, **changed, node_cache=cache)
        self.assertIs(cache.values['calibration'], calibration)
        self.assertFalse(any(key.startswith('learning.unet_instances.') for key in cache.values))
        self.assertIsNone(failed.unet_instances)
        self.assertEqual(first.backend_outcomes_json, first_capture)
        self.assertEqual(result_report(failed, review=review)['accepted_count'], 0)
        with (patch('seedvision.segmentation.baseline.predict_pipeline_model', return_value=(object(), 'opaque_fixture')),
              patch('seedvision.segmentation.baseline.decode_pipeline_model', return_value=learned)):
            recovered = analyze_path(self.source, **changed, node_cache=cache)
        self.assertEqual(outcomes.read_native_outcomes(recovered.backend_outcomes_json)['outcomes']['unet_instances']['state'], 'available')
        checkpoint_path.unlink()
        missing = analyze_path(self.source, **changed, node_cache=cache)
        self.assertFalse(any(key.startswith('learning.unet_instances.') for key in cache.values))
        self.assertEqual(outcomes.read_native_outcomes(missing.backend_outcomes_json)['outcomes']['unet_instances']['state'], 'unavailable')
        self.assertIs(cache.values['calibration'], calibration)
        disabled = analyze_path(self.source, **dict(changed, enabled_nodes=enabled - {'unet_instances'}), node_cache=cache)
        self.assertFalse(any(key.startswith('learning.unet_instances.') for key in cache.values))
        self.assertEqual(outcomes.read_native_outcomes(disabled.backend_outcomes_json)['outcomes']['unet_instances']['state'], 'not_run')

    def test_cancellation_and_device_fatal_errors_propagate_without_successful_fallback(self):
        self.enable_unet()
        from seedvision.segmentation.baseline import analyze_path, PipelineAnalysisCache
        from seedvision.pipeline.settings import production_settings
        from seedvision.timing import AnalysisCancelled
        settings = production_settings(self.graph)
        settings.update(enabled_nodes=frozenset(node for node, value in self.graph.nodes.items() if value.enabled),
                        learning_root=self.root, foreground_reference_mask=self.labels > 0,
                        reference_transform=self.reference_transform)
        for error in (AnalysisCancelled('cancelled'), torch.cuda.OutOfMemoryError('out of memory'),
                      RuntimeError('CUDA device-side assert')):
            cache = PipelineAnalysisCache()
            with self.subTest(error=error), patch('seedvision.segmentation.baseline.predict_pipeline_model', side_effect=error):
                with self.assertRaises(type(error)):
                    analyze_path(self.source, **settings, node_cache=cache)
            self.assertFalse(any(key.startswith('learning.unet_instances.') for key in cache.values))
            self.assertNotIn('result', cache.values)
