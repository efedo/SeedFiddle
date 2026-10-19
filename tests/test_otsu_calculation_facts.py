"""Exact binary domains and single-pass Otsu cutoff facts through native execution."""

from tests.functional_units import functional_units
from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch

from seedvision.benchmarking import native_recipe as runner
from seedvision.benchmarking import native_stage_replay as replay
from seedvision.benchmarking import native_threshold_stage as bridge
from seedvision.benchmarking.checkpoints import load_checkpoint, publish_checkpoint
from seedvision.cuda import plantcv_thresholds as thresholds
from seedvision.export import native_results as archives
from seedvision.segmentation.baseline import PipelineAnalysisCache
from tests.test_native_recipe import fixture, selected_recipe, FRAME, SOURCE, PREPARATION
from tests.test_native_threshold_stage import _fixture as checkpoint_fixture
from tests.test_adaptive_kernel_normalization import public_masks
from tests.test_plantcv_extended_thresholds import SCRIPT, REFERENCE

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

PUBLIC_OTSU_SCRIPT = SCRIPT[:SCRIPT.index('payload =')] + r'''
import cv2
payload = json.load(sys.stdin)
outcomes = []
for case in payload:
    image = np.asarray(case['image'], dtype=np.uint8)
    public = pcv.threshold.otsu(image, object_type=case['object_type'])
    flags = cv2.THRESH_OTSU | (cv2.THRESH_BINARY if case['object_type'] == 'light' else cv2.THRESH_BINARY_INV)
    cutoff, adapter = cv2.threshold(image, 0, 255, flags)
    assert np.array_equal(adapter, public), 'The scalar adapter must preserve the public mask'
    outcomes.append(dict(mask=public.tolist(), reference_cutoff=cutoff))
print(json.dumps(outcomes))
'''


def legacy_archive(current, destination):
    """Republish a genuine v4 report and bound checkpoint groups, without facts."""
    report, _ = archives.load_native_recipe_result(current)
    payload = json.loads(json.dumps(report))
    payload.pop('result_revision')
    groups = payload.pop('groups')
    payload['version'] = 4
    for node in payload['nodes']:
        node.pop('calculation')
    binding = sha256(archives._json_bytes(payload)).hexdigest()
    destination.mkdir()
    saved = []
    for group in groups:
        checkpoint = load_checkpoint(current/'checkpoints', group['checkpoint_sha256'])
        metadata = dict(checkpoint.metadata)
        metadata.update(version=4, run_binding_sha256=binding)
        reference = publish_checkpoint(destination/'checkpoints', checkpoint.identity, checkpoint.arrays,
            execution=checkpoint.execution, measurement=checkpoint.measurement,
            raster_roles=checkpoint.raster_roles, metadata=metadata, allow_cpu_transfer=True)
        saved.append(dict(checkpoint_sha256=reference.checkpoint_sha256, arrays=group['arrays']))
    body = payload | {'groups': saved}
    body['result_revision'] = sha256(archives._json_bytes(body)).hexdigest()
    (destination/'native-results.json').write_bytes(archives._json_bytes(body))


@functional_units
class OtsuCalculationContractTests(unittest.TestCase):
    def check_test_facts_reject_bool_out_of_range_profile_polarity_and_false_source_claim(self):
        expected = thresholds.otsu_calculation_facts(35, 'LIGHT')
        self.assertEqual(expected['object_type'], 'light')
        self.assertEqual(expected['comparison_rule'], 'gray > cutoff')
        self.assertEqual(thresholds.validate_otsu_calculation_facts(expected, 'light'), expected)
        cases = [None, dict(expected, native_selected_cutoff=True),
            dict(expected, native_selected_cutoff=-1), dict(expected, native_selected_cutoff=256),
            dict(expected, native_selected_cutoff=35.0),
            dict(expected, cutoff_selection_profile='other'), dict(expected, object_type='LIGHT'),
            dict(expected, comparison_rule='gray >= cutoff'),
            dict(expected, reference_scalar_conformance='exact'), dict(expected, unexpected=1)]
        for value in cases:
            with self.subTest(value=value), self.assertRaises((ValueError, TypeError)):
                thresholds.validate_otsu_calculation_facts(value, 'light')

    def check_test_source_scalar_adapter_rejects_wrong_origin_method_and_unverified_mask(self):
        from seedvision.benchmarking.reference_stage import validate_otsu_cutoff_adapter, ReferenceStageError
        record=dict(reference_selected_cutoff=10,origin='pinned_cv2.threshold_adapter',
            public_return='mask_only',object_type='light',method='THRESH_OTSU|THRESH_BINARY',
            public_mask_match=True)
        self.assertEqual(validate_otsu_cutoff_adapter(record,'light'),record)
        for bad in (dict(record,reference_selected_cutoff=True),dict(record,public_mask_match=False),
                dict(record,origin='PlantCV_public_return'),dict(record,method='THRESH_BINARY'),
                dict(record,public_return='mask_and_scalar')):
            with self.subTest(bad=bad),self.assertRaises(ReferenceStageError):
                validate_otsu_cutoff_adapter(bad,'light')

    def check_test_near_tie_and_rare_class_fallback_facts_keep_existing_selection_guards(self):
        with self.assertRaises(thresholds.OtsuAmbiguityError):
            thresholds._otsu_cutoff_from_histogram([10_000_000,10_000_000,10_000_001]+[0]*253)
        counts = [0]*256
        counts[127], counts[128] = 8_388_608, 1
        cutoff = thresholds._otsu_cutoff_from_histogram(counts)
        self.assertEqual(cutoff, 0)
        self.assertEqual(thresholds.otsu_calculation_facts(cutoff, 'dark')['comparison_rule'], 'gray <= cutoff')

    def test_validation_and_recovery(self):
        """2 related OtsuCalculationContractTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_facts_reject_bool_out_of_range_profile_polarity_and_false_source_claim)
        self.scenario(self.check_test_source_scalar_adapter_rejects_wrong_origin_method_and_unverified_mask)

    def test_workflow_lifecycle(self):
        """1 related OtsuCalculationContractTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_near_tie_and_rare_class_fallback_facts_keep_existing_selection_guards)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
@functional_units
class OtsuCalculationCudaTests(unittest.TestCase):
    def run_recipe(self, *, recipe=None, cache=None, cancel=None, gray=None, rgb=None):
        values, colors = fixture()
        return runner.run_native_recipe(recipe or runner.build_native_seed_recipe(selected_recipe('otsu')),
            torch.as_tensor(values, device='cuda') if gray is None else gray,
            torch.as_tensor(colors, device='cuda') if rgb is None else rgb,
            source_sha256=SOURCE, coordinate_frame=FRAME, preparation=PREPARATION, cache=cache, cancel=cancel)

    @staticmethod
    def stage(recipe):
        return next(stage.stage_id for stage in recipe.stages if stage.operation_id == 'threshold.otsu')



    def check_test_result_and_compatibility_wrapper_each_use_one_histogram(self):
        gray = torch.tensor([[10,10,10,200,200,200]],dtype=torch.uint8,device='cuda')
        original = torch.bincount
        with patch.object(torch,'bincount',wraps=original) as histogram:
            result = thresholds.threshold_otsu_result(gray,'LIGHT')
            self.assertEqual(histogram.call_count,1)
        self.assertEqual(result.native_selected_cutoff,10)
        with patch.object(torch,'bincount',wraps=original) as histogram:
            mask = thresholds.threshold_otsu(gray,'light')
            self.assertEqual(histogram.call_count,1)
        self.assertTrue(torch.equal(mask,result.mask))
        with self.assertRaises((AttributeError,TypeError)):
            result.native_selected_cutoff = 200

    def check_test_tied_partition_returns_no_result_or_calculation(self):
        gray = torch.tensor([[0,0,1,2,2]],dtype=torch.uint8,device='cuda')
        for operation in (thresholds.threshold_otsu_result,thresholds.threshold_otsu):
            with self.assertRaises(thresholds.OtsuAmbiguityError):
                operation(gray,'light')
        with self.assertRaises(thresholds.OtsuAmbiguityError):
            runner._dispatch('threshold.otsu',{'gray':gray},{'object_type':'light'},None)




    def check_test_new_archive_roundtrip_and_legacy_replay_preserve_mask_only_ports(self):
        recipe = runner.build_native_seed_recipe(selected_recipe('otsu'))
        run = self.run_recipe(recipe=recipe)
        stage = self.stage(recipe)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary);current=root/'current';legacy=root/'legacy'
            archives.export_native_recipe_result(current,recipe,run)
            report,_ = archives.load_native_recipe_result(current)
            self.assertEqual(report['version'],5)
            saved = next(node for node in report['nodes'] if node['stage_id']==stage)
            self.assertEqual(replay._compact_frame_metadata(saved['calculation']),dict(run.node(stage).calculation))
            before = {str(path.relative_to(current)):sha256(path.read_bytes()).hexdigest()
                for path in current.rglob('*') if path.is_file()}
            result = replay.replay_native_stage(current,stage)
            self.assertEqual(result.status,'matched',result.reason)
            self.assertEqual(set(result.outputs),{'mask'})
            self.assertEqual(result.metadata['calculation']['comparison_status'],'matched')
            self.assertEqual(result.metadata['comparison']['calculation']['compared_elements'],1)
            legacy_archive(current,legacy)
            legacy_report,_ = archives.load_native_recipe_result(legacy)
            self.assertEqual(legacy_report['version'],4)
            result = replay.replay_native_stage(legacy,stage)
            self.assertEqual(result.status,'matched',result.reason)
            self.assertIsNone(result.metadata['calculation']['archived'])
            self.assertEqual(result.metadata['calculation']['comparison_status'],'unrecorded_legacy_archive')
            self.assertEqual(result.metadata['comparison']['calculation']['compared_elements'],0)
            payload = json.loads(json.dumps(legacy_report));payload.pop('result_revision');payload.pop('groups')
            next(node for node in payload['nodes'] if node['stage_id']==stage)['calculation'] = saved['calculation']
            with self.assertRaises(archives.NativeResultExportError): archives._validate_payload(payload)
            after = {str(path.relative_to(current)):sha256(path.read_bytes()).hexdigest()
                for path in current.rglob('*') if path.is_file()}
            self.assertEqual(before,after)





    def test_workflow_lifecycle(self):
        """3 related OtsuCalculationCudaTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_result_and_compatibility_wrapper_each_use_one_histogram)
        self.scenario(self.check_test_tied_partition_returns_no_result_or_calculation)
        self.scenario(self.check_test_new_archive_roundtrip_and_legacy_replay_preserve_mask_only_ports)


if __name__ == '__main__':
    unittest.main()
