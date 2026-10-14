"""Exact binary domains and single-pass Otsu cutoff facts through native execution."""
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


class OtsuCalculationContractTests(unittest.TestCase):
    def test_facts_reject_bool_out_of_range_profile_polarity_and_false_source_claim(self):
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

    def test_source_scalar_adapter_rejects_wrong_origin_method_and_unverified_mask(self):
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

    def test_near_tie_and_rare_class_fallback_facts_keep_existing_selection_guards(self):
        with self.assertRaises(thresholds.OtsuAmbiguityError):
            thresholds._otsu_cutoff_from_histogram([10_000_000,10_000_000,10_000_001]+[0]*253)
        counts = [0]*256
        counts[127], counts[128] = 8_388_608, 1
        cutoff = thresholds._otsu_cutoff_from_histogram(counts)
        self.assertEqual(cutoff, 0)
        self.assertEqual(thresholds.otsu_calculation_facts(cutoff, 'dark')['comparison_rule'], 'gray <= cutoff')


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
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

    def test_binary_all_256_values_at_all_256_cutoffs_both_polarities_match_public(self):
        image = np.arange(256, dtype=np.uint8).reshape(16,16)
        cases = [dict(threshold=value, object_type=polarity)
            for value in range(256) for polarity in ('light','dark')]
        expected = public_masks(self, image, 'binary', cases)
        gray = torch.from_numpy(image).cuda().t()
        self.assertFalse(gray.is_contiguous())
        for params, original in zip(cases, expected):
            actual = thresholds.threshold_binary(gray, **params)
            self.assertEqual(actual.device, gray.device)
            self.assertEqual(actual.dtype, torch.uint8)
            np.testing.assert_array_equal(actual.t().cpu().numpy(), original)

    def test_public_masks_constants_singletons_thin_imbalance_and_empty_bin_plateaus(self):
        images = [np.zeros((1,1), np.uint8), np.full((1,1),255,np.uint8),
            np.full((7,9),117,np.uint8),
            np.array([[10,10,10,200,200,200]],np.uint8),
            np.array([[10],[10],[200],[200]],np.uint8),
            np.asarray([0]*999+[255],np.uint8).reshape(10,100),
            np.asarray([0]*4+[30]*6+[180]*9+[255]*10,np.uint8).reshape(1,29),
            np.random.default_rng(847).integers(0,256,(17,23),dtype=np.uint8)]
        cases = [dict(image=image.tolist(),object_type=polarity)
            for image in images for polarity in ('light','dark')]
        self.assertTrue(REFERENCE.is_file())
        process = subprocess.run([str(REFERENCE),'-c',PUBLIC_OTSU_SCRIPT],
            input=json.dumps(cases), capture_output=True, text=True, timeout=90)
        self.assertEqual(process.returncode,0,process.stderr)
        originals = json.loads(process.stdout)
        for case, original in zip(cases, originals):
            image = torch.tensor(case['image'],dtype=torch.uint8,device='cuda')
            result = thresholds.threshold_otsu_result(image,case['object_type'])
            np.testing.assert_array_equal(result.mask.cpu().numpy(),np.asarray(original['mask'],np.uint8))
            expected = (image > result.native_selected_cutoff if case['object_type']=='light'
                else image <= result.native_selected_cutoff).to(torch.uint8)*255
            self.assertTrue(torch.equal(result.mask,expected))
            self.assertEqual(result.mask.device,image.device)
            self.assertEqual(type(result.native_selected_cutoff),int)
            self.assertEqual(thresholds.otsu_calculation_facts(result.native_selected_cutoff,
                result.object_type)['reference_scalar_conformance'],'not_established')

    def test_result_and_compatibility_wrapper_each_use_one_histogram(self):
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

    def test_tied_partition_returns_no_result_or_calculation(self):
        gray = torch.tensor([[0,0,1,2,2]],dtype=torch.uint8,device='cuda')
        for operation in (thresholds.threshold_otsu_result,thresholds.threshold_otsu):
            with self.assertRaises(thresholds.OtsuAmbiguityError):
                operation(gray,'light')
        with self.assertRaises(thresholds.OtsuAmbiguityError):
            runner._dispatch('threshold.otsu',{'gray':gray},{'object_type':'light'},None)

    def test_node_cache_reuses_facts_and_cutoff_mutation_recomputes_only_dependents(self):
        recipe = runner.build_native_seed_recipe(selected_recipe('otsu'))
        stage = self.stage(recipe)
        values, colors = fixture()
        gray, rgb = torch.tensor(values,device='cuda'),torch.tensor(colors,device='cuda')
        cache = PipelineAnalysisCache()
        def run():
            calls = []
            def profile(frame,event,arg):
                if event == 'call' and frame.f_code is thresholds.threshold_otsu_result.__code__:
                    calls.append('otsu')
            sys.setprofile(profile)
            try:
                result = self.run_recipe(recipe=recipe,cache=cache,gray=gray,rgb=rgb)
            finally: sys.setprofile(None)
            return result,calls
        first,calls = run()
        self.assertEqual(calls,['otsu'])
        node = first.node(stage)
        self.assertEqual(set(node.outputs),{'mask'})
        self.assertEqual(node.calculation['native_selected_cutoff'],35)
        with self.assertRaises(TypeError): node.calculation['native_selected_cutoff'] = 36
        second,calls = run()
        self.assertEqual(calls,[])
        self.assertTrue(second.node(stage).reused)
        self.assertIs(second.node(stage).calculation,node.calculation)
        self.assertIs(second.node(stage).outputs['mask'],node.outputs['mask'])
        forged = thresholds.otsu_calculation_facts(36,'light')
        object.__setattr__(cache.values[f'reference.recipe:{recipe.recipe_id}:{stage}'].node,'calculation',forged)
        third,calls = run()
        self.assertEqual(calls,['otsu'])
        self.assertEqual(third.node(stage).calculation['native_selected_cutoff'],35)
        for previous in first.nodes:
            if previous.stage_id == stage: break
            self.assertTrue(third.node(previous.stage_id).reused)
            for port,value in previous.outputs.items():
                self.assertIs(third.node(previous.stage_id).outputs[port],value)
        self.assertNotEqual(third.node(stage).product_key,node.product_key)

    def test_pinned_scalar_adapter_and_exact_mask_plus_cutoff_conformance(self):
        from seedvision.benchmarking import threshold_conformance as pairs
        from seedvision.benchmarking.conformance import compare_masks, ConformanceError
        from tests.test_threshold_conformance import _fixture as pair_fixture
        for polarity in ('light','dark'):
            with self.subTest(polarity=polarity), tempfile.TemporaryDirectory() as temporary:
                image=np.array([[10,10,10,200,200,200]],np.uint8)
                args=pair_fixture(temporary,'threshold.otsu',
                    selected={'object_type':polarity},values=image)
                captured={}
                def capture(root,pair,**kwargs):
                    captured.update(root=root,pair=pair,kwargs=kwargs)
                    return compare_masks(root,pair,**kwargs)
                with patch.object(pairs,'compare_masks',side_effect=capture):
                    outcome=pairs.run_threshold_conformance_pair(**args)
                self.assertEqual(outcome.report['pair_status'],'evaluated_pass',outcome.report)
                result=outcome.report['conformance']['result']['conformance']
                self.assertTrue(result['selected_cutoff']['exactly_equal'])
                self.assertTrue(result['selected_cutoff']['mask_and_cutoff_equal'])
                source=load_checkpoint(args['output_root'],
                    outcome.reference_outcome.checkpoint_reference.checkpoint_sha256)
                adapter=source.metadata['observations']['otsu_cutoff_adapter']
                self.assertEqual(adapter['reference_selected_cutoff'],10)
                self.assertEqual(adapter['origin'],'pinned_cv2.threshold_adapter')
                self.assertEqual(adapter['public_return'],'mask_only')
                self.assertIs(adapter['public_mask_match'],True)
                native=load_checkpoint(args['output_root'],
                    outcome.native_outcome.checkpoint_reference.checkpoint_sha256)
                metadata=dict(native.metadata)
                metadata['calculation']=thresholds.otsu_calculation_facts(11,polarity)
                different=publish_checkpoint(args['output_root'],native.identity,native.arrays,
                    execution=native.execution,measurement=native.measurement,
                    raster_roles=native.raster_roles,metadata=metadata,allow_cpu_transfer=True)
                comparison=compare_masks(captured['root'],
                    replace(captured['pair'],candidate_checkpoint_sha256=different.checkpoint_sha256),
                    **captured['kwargs'])
                self.assertTrue(comparison['conformance']['exactly_equal'])
                self.assertEqual(comparison['conformance']['mismatch_pixel_count'],0)
                self.assertFalse(comparison['conformance']['selected_cutoff']['exactly_equal'])
                self.assertFalse(comparison['conformance']['passed'])
                metadata['calculation']=dict(native.metadata['calculation'],native_selected_cutoff=True)
                malformed=publish_checkpoint(args['output_root'],native.identity,native.arrays,
                    execution=native.execution,measurement=native.measurement,
                    raster_roles=native.raster_roles,metadata=metadata,allow_cpu_transfer=True)
                with self.assertRaisesRegex(ConformanceError,'cutoff metadata'):
                    compare_masks(captured['root'],
                        replace(captured['pair'],candidate_checkpoint_sha256=malformed.checkpoint_sha256),
                        **captured['kwargs'])

    def test_bridge_checkpoint_records_cutoff_from_same_calculation(self):
        with tempfile.TemporaryDirectory() as temporary:
            args,_,parameters = checkpoint_fixture(Path(temporary),'threshold.otsu',
                values=np.array([[10,10,10,200,200,200]],np.uint8))
            outcome = bridge.run_native_threshold_stage(**args)
            self.assertIsNotNone(outcome.checkpoint_reference,outcome.run_record.reason)
            checkpoint = load_checkpoint(args['output_root'],outcome.checkpoint_reference.checkpoint_sha256)
            self.assertEqual(checkpoint.metadata['effective_parameters'],parameters)
            facts = checkpoint.metadata['calculation']
            self.assertEqual(facts['native_selected_cutoff'],10)
            self.assertEqual(thresholds.validate_otsu_calculation_facts(facts,parameters['object_type']),dict(facts))

    def test_new_archive_roundtrip_and_legacy_replay_preserve_mask_only_ports(self):
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

    def test_facts_only_mismatch_is_reported_even_with_identical_plateau_mask(self):
        recipe = runner.build_native_seed_recipe(selected_recipe('otsu'))
        run = self.run_recipe(recipe=recipe);stage=self.stage(recipe);node=run.node(stage)
        changed = replace(node,calculation=thresholds.otsu_calculation_facts(36,'light'))
        altered = replace(run,nodes=tuple(changed if item.stage_id==stage else item for item in run.nodes))
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary)/'altered'
            archives.export_native_recipe_result(directory,recipe,altered)
            result = replay.replay_native_stage(directory,stage)
            self.assertEqual(result.status,'mismatch',result.reason)
            self.assertTrue(torch.equal(result.outputs['mask'],node.outputs['mask']))
            self.assertEqual(result.metadata['comparison']['by_port']['mask']['mismatched_elements'],0)
            self.assertEqual(result.metadata['comparison']['calculation']['mismatched_elements'],1)

    def test_malformed_or_missing_new_facts_reject_before_publication(self):
        recipe=runner.build_native_seed_recipe(selected_recipe('otsu'))
        run=self.run_recipe(recipe=recipe);stage=self.stage(recipe);node=run.node(stage)
        values=[None,dict(node.calculation,native_selected_cutoff=True),
            dict(node.calculation,cutoff_selection_profile='other'),
            dict(node.calculation,reference_scalar_conformance='exact')]
        with tempfile.TemporaryDirectory() as temporary:
            for index,value in enumerate(values):
                changed=replace(node,calculation=value)
                altered=replace(run,nodes=tuple(changed if item.stage_id==stage else item for item in run.nodes))
                directory=Path(temporary)/str(index)
                with patch.object(archives,'publish_checkpoint') as publish:
                    with self.assertRaises(archives.NativeResultExportError):
                        archives.export_native_recipe_result(directory,recipe,altered)
                    publish.assert_not_called()
                self.assertFalse(directory.exists())

    def test_callback_fact_mutation_takes_priority_and_evicts_descendants(self):
        recipe=runner.build_native_seed_recipe(selected_recipe('otsu'))
        stage=self.stage(recipe);cache=PipelineAnalysisCache();changed=[]
        def callback():
            entry=cache.values.get(f'reference.recipe:{recipe.recipe_id}:{stage}')
            if entry is not None and not changed:
                changed.append(True)
                object.__setattr__(entry.node,'calculation',thresholds.otsu_calculation_facts(36,'light'))
                raise RuntimeError('caller exception')
            return False
        with self.assertRaisesRegex(runner.NativeRecipeError,'cutoff facts changed'):
            self.run_recipe(recipe=recipe,cache=cache,cancel=callback)
        self.assertTrue(changed)
        self.assertNotIn(f'reference.recipe:{recipe.recipe_id}:{stage}',cache.values)
        for previous in recipe.stages:
            if previous.stage_id==stage:break
            self.assertIn(f'reference.recipe:{recipe.recipe_id}:{previous.stage_id}',cache.values)

    def test_async_qt_cutoff_facts_and_replay_after_origin_deleted(self):
        from PySide6.QtWidgets import QApplication
        from seedvision.ui.native_recipe_dialog import NativeRecipeDialog
        application=QApplication.instance() or QApplication([])
        recipe=runner.build_native_seed_recipe(selected_recipe('otsu'));stage=self.stage(recipe)
        image,_=fixture()
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);source=root/'seed.png';saved=root/'saved'
            self.assertTrue(cv2.imwrite(str(source),np.repeat(image[...,None],3,axis=2)))
            dialog=NativeRecipeDialog(recipe);dialog.show()
            try:
                dialog.set_source_path(source);dialog.start_run()
                deadline=time.monotonic()+45
                while dialog._worker is not None and time.monotonic()<deadline:
                    application.processEvents();time.sleep(.01)
                application.processEvents()
                self.assertIsNone(dialog._worker)
                self.assertIsNotNone(dialog.result,dialog.status_label.text())
                node=dialog.result.node(stage)
                self.assertEqual(node.status,'executed',node.reason)
                dialog._select_stage(stage)
                self.assertIn(f"Native selected cutoff: {node.calculation['native_selected_cutoff']}",dialog.facts.toPlainText())
                self.assertIn('scalar parity is not established',dialog.facts.toPlainText())
                self.assertFalse(dialog.preview.pixmap().isNull())
                archives.export_native_recipe_result(saved,recipe,dialog.result)
                source.unlink()
                result=replay.replay_native_stage(saved,stage)
                self.assertEqual(result.status,'matched',result.reason)
                self.assertTrue(torch.equal(result.outputs['mask'],node.outputs['mask']))
                self.assertIs(dialog.result.node(stage),node)
            finally:
                dialog.close();application.processEvents()


if __name__ == '__main__':
    unittest.main()
