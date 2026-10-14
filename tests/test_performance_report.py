"""Actual synchronized CUDA/resource/cache facts and explicit correction timing."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch

from seedvision.benchmarking import performance as p
from seedvision.benchmarking.native_recipe import build_native_seed_recipe
from tests.test_native_recipe import selected_recipe, fixture


class PerformanceContractTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='performance-report-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / 'source.png'
        gray, rgb = fixture()
        rgb = np.stack((gray, np.minimum(gray, 173), np.maximum(gray, 61)), axis=-1)
        self.assertTrue(cv2.imwrite(str(self.source), rgb))
        self.recipe = build_native_seed_recipe(selected_recipe())

    def test_resource_controls_reject_invalid_and_unbounded_requests_before_cuda(self):
        with patch.object(p.torch.cuda, 'synchronize', side_effect=AssertionError('No CUDA execution')):
            for kwargs in ({'repeats': True}, {'repeats': 65}, {'warmups': -1},
                           {'modes': ('warm',), 'warmups': 0}, {'modes': ('cold', 'cold')},
                           {'maximum_elapsed_seconds': float('inf')}, {'maximum_peak_bytes': True},
                           {'device': 'cpu'}):
                with self.subTest(kwargs=kwargs), self.assertRaises((ValueError, TypeError)):
                    p.benchmark_native_recipe(self.recipe, self.source, **kwargs)

    def test_unavailable_cuda_has_explicit_report_and_no_cpu_substitution(self):
        with patch.object(p.torch.cuda, 'is_available', return_value=False), patch.object(
                p, 'run_native_recipe_path', side_effect=AssertionError('No substitute')):
            report = p.benchmark_native_recipe(self.recipe, self.source)
        self.assertEqual(report['status'], 'unavailable')
        self.assertFalse(report['runs'])
        self.assertIsNone(report['operator_correction']['records'] or None)
        self.assertFalse(report['scientifically_validated'])

    def test_explicit_correction_timer_preserves_completed_and_interrupted_work(self):
        binding = dict(operator='reviewer', acceptance_standard='visible-outline-review-v1',
            source_sha256=sha256(self.source.read_bytes()).hexdigest(), recipe_sha256=self.recipe.identity.recipe_sha256)
        with p.measure_operator_correction(**binding) as complete:
            sum(range(1000))
        self.assertEqual(complete['status'], 'completed')
        self.assertGreater(complete['seconds'], 0)
        error = RuntimeError('Interrupted correction')
        with self.assertRaises(RuntimeError) as failed:
            with p.measure_operator_correction(**binding) as interrupted:
                raise error
        self.assertIs(failed.exception, error)
        self.assertEqual(interrupted['status'], 'interrupted')
        with patch.object(p.torch.cuda, 'is_available', return_value=False):
            report = p.benchmark_native_recipe(self.recipe, self.source, correction_records=(complete, interrupted))
        self.assertEqual(report['operator_correction']['records'], [complete, interrupted])
        stale = deepcopy(complete)
        stale['source_sha256'] = 'b'*64
        with self.assertRaises(ValueError):
            p.benchmark_native_recipe(self.recipe, self.source, correction_records=(stale,))

    def test_report_export_is_bounded_finite_and_preserves_prior_destination(self):
        path = self.root / 'report.json'
        p.export_performance_report({'status': 'failed', 'scientifically_validated': False}, path)
        original = path.read_bytes()
        with self.assertRaises(FileExistsError):
            p.export_performance_report({'status': 'executed'}, path)
        self.assertEqual(path.read_bytes(), original)
        with self.assertRaises(ValueError):
            p.export_performance_report({'seconds': float('nan')}, self.root / 'invalid.json')
        self.assertFalse((self.root / 'invalid.json').exists())


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class ActualPerformanceTests(unittest.TestCase):
    setUp = PerformanceContractTests.setUp
    def test_actual_cold_warm_timings_peaks_and_transfer_trace_preserve_native_products(self):
        observed = []
        actual = p.run_native_recipe_path
        def record(*args, **kwargs):
            result = actual(*args, **kwargs)
            observed.append((result.labels.data_ptr(), tuple(node.reused for node in result.nodes),
                             result.labels.device.type, result.labels.sum().item()))
            return result
        with patch.object(p, 'run_native_recipe_path', record):
            report = p.benchmark_native_recipe(self.recipe, self.source, repeats=1, warmups=1)
        self.assertEqual(report['status'], 'executed', report['runs'])
        cold, warm = report['runs']
        self.assertEqual(observed[0][0], observed[2][0])
        self.assertTrue(all(observed[2][1]))
        self.assertEqual({row[2] for row in observed}, {'cuda'})
        self.assertEqual(len({row[3] for row in observed}), 1)
        self.assertGreater(cold['transfers']['cpu_to_cuda_tensor_bytes'], 0)
        self.assertEqual(warm['transfers']['cpu_to_cuda_tensor_bytes'], 0)
        for row in (cold, warm):
            self.assertGreaterEqual(row['wall_seconds'], 0)
            self.assertGreaterEqual(row['cuda_event_seconds'], 0)
            self.assertGreaterEqual(row['peak_allocated_bytes'], row['baseline_allocated_bytes'])
            self.assertIn('input_preparation', row['node_timings'])
            self.assertTrue(all(value['cuda_event_seconds'] is not None for value in row['node_timings'].values()))
        self.assertFalse(report['scientifically_validated'])
        p.export_performance_report(report, self.root / 'actual.json')
        self.assertEqual(json.loads((self.root / 'actual.json').read_bytes()), report)

    def test_measured_threshold_stops_following_runs_and_cancel_is_distinct(self):
        report = p.benchmark_native_recipe(self.recipe, self.source, repeats=3, warmups=0,
            modes=('cold',), maximum_elapsed_seconds=1e-12)
        self.assertEqual(report['status'], 'threshold_exceeded')
        self.assertEqual(len(report['runs']), 1)
        self.assertEqual(report['requested_measured_runs'], 3)
        cancelled = p.benchmark_native_recipe(self.recipe, self.source, cancel=lambda: True)
        self.assertEqual(cancelled['status'], 'cancelled')
        self.assertFalse(cancelled['runs'])

    def test_actual_producer_failure_retains_synchronized_measurements_without_products(self):
        error = RuntimeError('Controlled path producer failure')
        with patch.object(p, 'run_native_recipe_path', side_effect=error):
            report = p.benchmark_native_recipe(self.recipe, self.source, repeats=2, warmups=0, modes=('cold',))
        self.assertEqual(report['status'], 'failed')
        self.assertEqual(len(report['runs']), 1)
        self.assertIn('Controlled path producer failure', report['runs'][0]['reason'])
        self.assertIsNone(report['runs'][0]['nodes'])
        self.assertGreaterEqual(report['runs'][0]['peak_allocated_bytes'], 0)

    def test_failed_measurement_keeps_original_failure_and_unknown_gpu_facts(self):
        with patch.object(p, 'run_native_recipe_path', side_effect=RuntimeError('Original producer error')), \
                patch.object(p.torch.cuda, 'synchronize', side_effect=(None, RuntimeError('Sync error'))):
            report = p.benchmark_native_recipe(self.recipe, self.source, repeats=1, warmups=0, modes=('cold',))
        row = report['runs'][0]
        self.assertEqual(report['status'], 'failed')
        self.assertIn('Original producer error', row['reason'])
        self.assertIn('Sync error', row['measurement_error'])
        self.assertIsNone(row['cuda_event_seconds'])
        self.assertIsNone(row['peak_allocated_bytes'])
        self.assertFalse(row['synchronized'])
