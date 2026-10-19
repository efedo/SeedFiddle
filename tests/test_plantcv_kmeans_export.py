"""Compact fitted-centre export, identity and transfer/cancellation boundaries."""

from tests.functional_units import functional_units
from dataclasses import replace
from hashlib import sha256
import tracemalloc
import unittest
from unittest.mock import patch

import torch

from seedvision.cuda import plantcv_kmeans_export as native
from seedvision.cuda import plantcv_kmeans_fit as fit
from seedvision.cuda import plantcv_kmeans_model as models


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
@functional_units
class KMeansFitExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        features = torch.zeros((31, 4), dtype=torch.uint16, device='cuda')
        cls.result = fit.fit_kmeans_training_rows(features, 2, source_n_threads=1,
            batch_size=7, n_init=1)

    def check_test_early_stopped_fit_and_ordered_fractional_signed_zero_centres_roundtrip(self):
        self.assertLess(self.result.n_steps, self.result.plan.n_steps)
        centres = torch.tensor([[-0., 2.125, 255., 17.], [9., 1.5, 100.25, 99.875]], dtype=torch.float64, device='cuda')
        result = replace(self.result, centres=centres)
        exported = native.export_kmeans_fit(result, 2, 1)
        self.assertEqual(exported.model, models.parse_kmeans_model(exported.payload))
        self.assertEqual(exported.payload, models.encode_kmeans_model(exported.model))
        self.assertEqual(exported.model.source_sha256, sha256(exported.payload).hexdigest())
        self.assertEqual(exported.model.source_bytes, len(exported.payload))
        self.assertEqual(exported.model.centres, tuple(tuple(row) for row in centres.cpu().tolist()))
        self.assertIn(b'-0.0', exported.payload)
        native.export_kmeans_fit(self.result, 2, 1)

    def check_test_only_bounded_centres_transfer_and_source_tensors_remain_unchanged(self):
        original = native._copy_centres_to_cpu
        values = (self.result.centres, self.result.counts, self.result.labels, self.result.row_distances)
        versions = tuple(value._version for value in values)
        with patch.object(native, '_copy_centres_to_cpu', wraps=original) as transfer:
            native.export_kmeans_fit(self.result, 2, 1)
        self.assertEqual(transfer.call_count, 1)
        self.assertIs(transfer.call_args.args[0], self.result.centres)
        self.assertTrue(all(value.device.type == 'cuda' for value in values))
        self.assertEqual(versions, tuple(value._version for value in values))

    def check_test_invalid_result_geometry_and_cpu_budget_reject_before_transfer(self):
        invalid = [None, replace(self.result, plan=None), replace(self.result, centres=self.result.centres.cpu()),
            replace(self.result, centres=self.result.centres.float()),
            replace(self.result, centres=self.result.centres.t()),
            replace(self.result, plan=replace(self.result.plan, clusters=3)),
            replace(self.result, plan=replace(self.result.plan, features=True))]
        with patch.object(native, '_copy_centres_to_cpu', side_effect=AssertionError('transfer before validation')):
            for result in invalid:
                with self.subTest(value=type(result).__name__), self.assertRaises(native.KMeansExportError):
                    native.export_kmeans_fit(result, 2, 1)
            for size, channels in ((True, 1), (0, 1), (32, 1), (2, True), (2, 2), (1, 1)):
                with self.assertRaises(native.KMeansExportError):
                    native.export_kmeans_fit(self.result, size, channels)
            with patch.object(native, 'MAX_CPU_METADATA_BYTES', 1), self.assertRaises(native.KMeansExportError):
                native.export_kmeans_fit(self.result, 2, 1)

    def check_test_actual_largest_model_geometry_cpu_peak_and_incremental_payload_cap(self):
        centres = torch.full((256, 2883), .125, dtype=torch.float64, device='cuda')
        result = replace(self.result, centres=centres, plan=replace(self.result.plan, rows=256, clusters=256, features=2883))
        tracemalloc.start()
        try:
            exported = native.export_kmeans_fit(result, 31, 3)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        bound = native.CPU_BYTES_PER_CENTRE*centres.numel()+4*models.MAX_MODEL_BYTES+native.CPU_HEADROOM_BYTES
        self.assertLessEqual(peak+8*centres.numel(), bound)
        self.assertLessEqual(bound, native.MAX_CPU_METADATA_BYTES)
        self.assertEqual(len(exported.model.centres), 256)
        self.assertLess(len(exported.payload), models.MAX_MODEL_BYTES)
        centres.fill_(.12345678901234568)
        with self.assertRaisesRegex(native.KMeansExportError, 'byte limit'):
            native.export_kmeans_fit(result, 31, 3)

    def check_test_nonfinite_values_bad_transfer_and_encoder_failure_publish_no_export(self):
        for value in (float('nan'), float('inf'), -1., 256.):
            result = replace(self.result, centres=torch.full_like(self.result.centres, value))
            with self.assertRaises(native.KMeansExportError):
                native.export_kmeans_fit(result, 2, 1)
        with patch.object(native, '_copy_centres_to_cpu', return_value=torch.zeros((1, 1))), self.assertRaises(native.KMeansExportError):
            native.export_kmeans_fit(self.result, 2, 1)
        with patch.object(models, 'encode_kmeans_model', side_effect=models.KMeansModelError('bad encoder')), self.assertRaisesRegex(native.KMeansExportError, 'bad encoder'):
            native.export_kmeans_fit(self.result, 2, 1)

    def check_test_every_callback_mutation_and_cancellation_detected_before_next_stage(self):
        total = 0
        def count():
            nonlocal total
            total += 1
            return False
        native.export_kmeans_fit(self.result, 2, 1, cancel=count)
        for target in range(1, total+1):
            result = replace(self.result, centres=self.result.centres.clone())
            calls = 0
            def mutate():
                nonlocal calls
                calls += 1
                if calls == target:
                    result.centres.resize_(0)
                return False
            with self.assertRaisesRegex(native.KMeansExportError, 'changed'):
                native.export_kmeans_fit(result, 2, 1, cancel=mutate)
            self.assertEqual(calls, target)
        for target in (1, total//2, total):
            calls = 0
            def cancel():
                nonlocal calls
                calls += 1
                return calls == target
            with self.assertRaises(native.KMeansExportCancelled):
                native.export_kmeans_fit(self.result, 2, 1, cancel=cancel)
            self.assertEqual(calls, target)
        with patch.object(native, '_copy_centres_to_cpu', side_effect=AssertionError('transfer after cancel')):
            with self.assertRaises(native.KMeansExportCancelled):
                native.export_kmeans_fit(self.result, 2, 1, cancel=lambda: True)
        with self.assertRaisesRegex(native.KMeansExportError, 'bool'):
            native.export_kmeans_fit(self.result, 2, 1, cancel=lambda: 1)
        result = replace(self.result, centres=self.result.centres.clone())
        original = native._copy_centres_to_cpu
        def mutated_copy(centres):
            cpu = original(centres)
            centres.resize_(0)
            return cpu
        with patch.object(native, '_copy_centres_to_cpu', side_effect=mutated_copy):
            with self.assertRaisesRegex(native.KMeansExportError, 'changed'):
                native.export_kmeans_fit(result, 2, 1)

    def check_test_loaded_source_binds_export_budget_and_transfer_implementation(self):
        from seedvision.benchmarking.native_recipe import _module_record
        self.assertTrue(_module_record(native)['loaded_code'])
        with patch.object(native, 'CPU_BYTES_PER_CENTRE', 1), self.assertRaises(ValueError):
            _module_record(native)

    def test_validation_and_recovery(self):
        """5 related KMeansFitExportTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_early_stopped_fit_and_ordered_fractional_signed_zero_centres_roundtrip)
        self.scenario(self.check_test_invalid_result_geometry_and_cpu_budget_reject_before_transfer)
        self.scenario(self.check_test_nonfinite_values_bad_transfer_and_encoder_failure_publish_no_export)
        self.scenario(self.check_test_every_callback_mutation_and_cancellation_detected_before_next_stage)
        self.scenario(self.check_test_loaded_source_binds_export_budget_and_transfer_implementation)

    def test_workflow_lifecycle(self):
        """2 related KMeansFitExportTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_only_bounded_centres_transfer_and_source_tensors_remain_unchanged)
        self.scenario(self.check_test_actual_largest_model_geometry_cpu_peak_and_incremental_payload_cap)
