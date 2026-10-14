"""Actual public composition checks, followed by explicitly injected faults.

The direct oracle calls pinned PlantCV, never the adapters or coordinator.
Replay tests below check orchestration only; they are not numerical oracles.
"""
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import tempfile
import threading
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from seedvision.benchmarking import reference_calibrated_seed as implementation
from seedvision.benchmarking import reference_calibrated_seed_reader as reader
from seedvision.benchmarking.checkpoints import (
    CheckpointCancelled, CheckpointLimits, ExecutionRecord, ExecutionStatus,
    RasterRole, _canonical_json, _thaw_json, load_checkpoint, publish_checkpoint,
)
from seedvision.benchmarking.recipe_catalogue import build_recipe_catalogue
from seedvision.benchmarking.reference_runner import (
    PlantCVReferenceRunner, ReferenceCancelledError, ReferenceExecutionError,
)
from seedvision.benchmarking.reference_seed_chain import ReferenceRasterInput
from tests.test_native_calibrated_seed_recipe import fixture, reference_script
from tests.test_native_recipe import selected_recipe
from tests.test_reference_preprocessing import (
    HAS_REFERENCE_ENV, REFERENCE_PYTHON, _ReferencePreprocessingFixtures, _measurement,
)


def spec(kind='binary', **changes):
    values = {'seed_recipe':selected_recipe(kind, **{'threshold.binary':{'threshold':124}}),
        'card_parameters':{'adaptive_method':0,'roi_bounds_xyxy':[0,0,460,620],
                           'color_chip_size':[7,11]},
        'seed_roi_xyxy':(600,260,728,356)}
    return implementation.ReferenceCalibratedSeedSpec(**(values|changes))


def arguments(fixtures, selected=None, *, bgr=None):
    selected = spec() if selected is None else selected
    bgr = fixture() if bgr is None else bgr
    ok, encoded = cv2.imencode('.png',bgr)
    if not ok:
        raise AssertionError('PNG fixture encoding failed.')
    rgb = cv2.imdecode(encoded,cv2.IMREAD_COLOR)[...,::-1].copy()
    old = fixtures._arguments(selected.seed_recipe,shape=rgb.shape[:2])
    identity = replace(old['expected_input_identity'],stage_id='prepared_rgb_input',
        source_sha256=sha256(encoded.tobytes()).hexdigest(),
        image_semantics={**dict(old['expected_input_identity'].image_semantics),
                        'color_order':'RGB','channel_conversion':'local OpenCV BGR decode reversed to RGB'})
    role = RasterRole('image','|u1','HWC',(0,255))
    measurement = _measurement('prepared_full_rgb')
    published = publish_checkpoint(old['input_root'],identity,{'rgb':rgb},
        execution=ExecutionRecord(ExecutionStatus.AVAILABLE),measurement=measurement,
        raster_roles={'rgb':role},metadata={'fixture_only':True,'actual_public_reader':False},
        allow_cpu_transfer=True)
    return {'parent':ReferenceRasterInput(old['input_root'],published.checkpoint_sha256,
                identity,measurement,'rgb',role),'spec':selected,'runner':fixtures.runner,
            'output_root':old['output_root'],'expected_spec_sha256':selected.sha256,
            'expected_source_sha256':identity.source_sha256,
            'expected_parent_frame':identity.coordinate_frame,
            'output_measurement':_measurement('actual_public_seed_products'),'allow_cpu_transfer':True}


def read_arguments(args):
    return {k:args[k] for k in ('parent','spec','expected_spec_sha256','expected_source_sha256',
                              'expected_parent_frame','output_measurement')}


class CalibratedSpecTests(unittest.TestCase):
    def setUp(self):
        self.fixtures = _ReferencePreprocessingFixtures()
        self.fixtures.setUp()
        self.addCleanup(self.fixtures.tearDown)

    def test_composition_roundtrip_preserves_original_catalogue_and_selected_recipes(self):
        before = build_recipe_catalogue().to_dict()
        for kind in ('binary','mean','otsu'):
            selected = spec(kind)
            restored = implementation.ReferenceCalibratedSeedSpec.from_dict(selected.to_dict())
            self.assertEqual(restored.sha256,selected.sha256)
            self.assertEqual(restored.seed_recipe.identity,selected.seed_recipe.identity)
            with self.assertRaises(TypeError):
                restored.card_parameters['radius'] = 7
        self.assertEqual(build_recipe_catalogue().to_dict(),before)

    def test_selected_calculations_and_crop_change_composition_identity(self):
        original = spec()
        alternatives = [spec('mean'),spec('otsu'),spec(seed_roi_xyxy=(601,260,729,356)),
            spec(card_parameters={'adaptive_method':0,'color_chip_size':[7,13]}),
            spec(seed_recipe=selected_recipe('binary', **{'threshold.binary':{'threshold':125}}))]
        self.assertTrue(all(value.sha256!=original.sha256 for value in alternatives))
        value = original.to_dict()
        value['adapters']['public_roi_helper_sha256'] = 'f'*64
        with self.assertRaises(implementation.ReferenceCalibratedSeedError):
            implementation.ReferenceCalibratedSeedSpec.from_dict(value)

    def test_malformed_selected_controls_are_rejected(self):
        for bounds in (None,[0,0,0,1],[-1,0,2,3],[0,0,True,3],[0,0,2.0,3]):
            with self.subTest(bounds=bounds),self.assertRaises(ValueError):
                spec(seed_roi_xyxy=bounds)
        for parameters in ({'block_size':4},{'radius':0},{'min_size':True},
                {'color_chip_size':[0,11]},{'roi_bounds_xyxy':[0,0,0,1]},
                {'aspect_ratio':float('nan')},{'solidity':2},{'deprecated_orientation':3}):
            with self.subTest(parameters=parameters),self.assertRaises(ValueError):
                spec(card_parameters=parameters)

    def test_preflight_rejects_wrong_revision_frame_and_out_of_frame_rois(self):
        args = arguments(self.fixtures)
        changes = [dict(expected_spec_sha256='0'*64),dict(expected_source_sha256='0'*64),
            dict(expected_parent_frame=replace(args['expected_parent_frame'],origin='elsewhere')),
            dict(allow_cpu_transfer=False),dict(output_measurement=None)]
        oversized = spec(seed_roi_xyxy=(600,260,901,356))
        changes.append(dict(spec=oversized,expected_spec_sha256=oversized.sha256))
        with patch.object(PlantCVReferenceRunner,'run') as run:
            for change in changes:
                with self.subTest(change=change),self.assertRaises((ValueError,TypeError)):
                    implementation.run_reference_calibrated_seed(**(args|change))
            run.assert_not_called()

    def test_cancelled_preflight_and_reader_invalid_context_do_not_open_files(self):
        args = arguments(self.fixtures)
        event = threading.Event()
        event.set()
        with patch.object(implementation,'_input',side_effect=AssertionError('input read')):
            with self.assertRaises(ReferenceCancelledError):
                implementation.run_reference_calibrated_seed(**args,cancel=event)
        read_args = read_arguments(args)
        with patch.object(Path,'open',side_effect=AssertionError('manifest read')):
            with self.assertRaises(CheckpointCancelled):
                reader.read_reference_calibrated_seed(args['output_root'],'a'*64,**read_args,cancel=event)
            for change in (dict(output_measurement=None),dict(limits=None),
                    dict(expected_parent_frame=replace(args['expected_parent_frame'],units='mm'))):
                with self.subTest(change=change),self.assertRaises((ValueError,TypeError)):
                    reader.read_reference_calibrated_seed(args['output_root'],'a'*64,**(read_args|change))

    def _mutation_cancellation(self,error_type,signal_event):
        args = arguments(self.fixtures)
        event = threading.Event()
        def run(runner,operation,arrays,parameters=None,**kwargs):
            arrays['rgb'] = arrays['rgb'].copy()
            arrays['rgb'][0,0,0] ^= 1
            if signal_event:
                event.set()
            raise error_type('Declared source failure after bound input replacement.')
        with patch.object(PlantCVReferenceRunner,'run',run):
            outcome = implementation.run_reference_calibrated_seed(**args,cancel=event)
        self.assertEqual(outcome.correction.status,'failed')
        self.assertTrue(outcome.correction.details['input_contract_violated'])
        self.assertIn('Bound input changed',outcome.correction.reason)
        self.assertIsNone(outcome.correction.checkpoint_reference)
        self.assertIsNone(outcome.child)
        self.assertIsNone(outcome.manifest_sha256)
        self.assertEqual(outcome.status,'cancelled' if signal_event else 'partial')

    def test_bound_input_mutation_remains_failed_when_cancellation_is_true(self):
        self._mutation_cancellation(ReferenceExecutionError,True)

    def test_bound_input_mutation_remains_failed_on_cancellation_exception(self):
        self._mutation_cancellation(ReferenceCancelledError,False)


@unittest.skipUnless(HAS_REFERENCE_ENV,'Pinned public PlantCV environment is required.')
class ActualCalibratedCompositionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixtures = _ReferencePreprocessingFixtures()
        cls.fixtures.setUp()
        cls.addClassCleanup(cls.fixtures.tearDown)
        cls.cases = {}
        cls.source_results = []
        cls.actual_requests = []
        original = PlantCVReferenceRunner.run
        for kind in ('binary','mean','otsu'):
            args = arguments(cls.fixtures,spec(kind))
            results,requests = [],[]
            def observe(runner,operation,arrays,parameters=None,**kwargs):
                requests.append((operation,deepcopy(parameters)))
                result = original(runner,operation,arrays,parameters,**kwargs)
                results.append(result)
                return result
            with patch.object(PlantCVReferenceRunner,'run',observe):
                outcome = implementation.run_reference_calibrated_seed(**args)
            cls.cases[kind] = (args,outcome,results,requests)
            if kind=='binary':
                cls.source_results = results
                cls.actual_requests = requests

    def direct(self,kind):
        with tempfile.TemporaryDirectory(prefix='direct-calibrated-public-') as name:
            root = Path(name)
            np.savez(root/'input.npz',bgr=fixture())
            script = reference_script().replace('pcv.params.line_thickness = 5','pcv.params.line_thickness = 1')
            script = script.replace("observations['__size_observations__'] = dict(pcv.outputs.observations)",
                "observations['__size_observations__'] = dict(pcv.outputs.observations)\n"
                "pcv.outputs.clear()\npcv.params.px_width = pcv.params.px_height = 1.0\n"
                "pcv.params.unit = 'pixels'\n"
                "pixel_image = pcv.analyze.size(img=bgr,labeled_mask=outputs['labels'],"
                "n_labels=int(outputs['labels'].max()),label='graph')\n"
                "outputs['pixel_size_diagnostic_rgb'] = pixel_image[...,::-1].copy()\n"
                "observations['__pixel_size_observations__'] = dict(pcv.outputs.observations)")
            completed = subprocess.run([str(REFERENCE_PYTHON),'-c',script,str(root/'input.npz'),
                str(root/'output.npz'),str(root/'observations.json'),kind,'rgb'],
                capture_output=True,text=True,encoding='utf-8',timeout=180)
            self.assertEqual(completed.returncode,0,completed.stderr[-4000:])
            with np.load(root/'output.npz',allow_pickle=False) as archive:
                arrays = {k:archive[k].copy() for k in archive.files}
            return arrays,json.loads((root/'observations.json').read_text(encoding='utf-8'))

    def verified(self,kind='binary',**changes):
        args,outcome,_,_ = self.cases[kind]
        self.assertEqual(outcome.status,'executed',_thaw_json(outcome.record))
        return reader.read_reference_calibrated_seed(args['output_root'],outcome.manifest_sha256,
                                                    **(read_arguments(args)|changes))

    def test_actual_binary_mean_otsu_match_direct_corrected_pixels_labels_and_observations(self):
        for kind,(args,outcome,results,requests) in self.cases.items():
            with self.subTest(kind=kind):
                self.assertEqual(outcome.status,'executed',_thaw_json(outcome.record))
                verified = self.verified(kind)
                expected,observations = self.direct(kind)
                checkpoints = verified.checkpoints
                np.testing.assert_array_equal(checkpoints['correction'].arrays['corrected_image'],expected['corrected'][...,::-1])
                np.testing.assert_array_equal(checkpoints['crop.rgb'].arrays['rgb'],expected['seed_rgb'])
                np.testing.assert_array_equal(checkpoints['crop.gray'].arrays['gray'],expected['seed_gray'])
                stages = (('median_blur','image'),('binary_threshold','mask'),('small_object_filter','mask'),
                          ('erode','mask'),('dilate','mask'))
                for i,(name,port) in enumerate(stages):
                    np.testing.assert_array_equal(checkpoints[f'seed.preprocessing.{i}'].arrays[port],expected[name])
                for name in ('distance','coordinates','peaks','markers','negative_distance','labels'):
                    np.testing.assert_array_equal(checkpoints['seed.watershed'].arrays[name],expected[name])
                np.testing.assert_array_equal(checkpoints['scaled_size'].arrays['analysis_image'],expected['size_diagnostic_rgb'])
                np.testing.assert_array_equal(checkpoints['seed.size'].arrays['analysis_image'],expected['pixel_size_diagnostic_rgb'])
                self.assertEqual(_thaw_json(checkpoints['scaled_size'].metadata['observations']['observations']),observations.pop('__size_observations__'))
                self.assertEqual(_thaw_json(checkpoints['seed.size'].metadata['observations']['observations']),observations.pop('__pixel_size_observations__'))
                scale = observations.pop('__scale__')
                observations.pop('__returned_hue_chart__')
                self.assertEqual(_thaw_json(checkpoints['seed.color'].metadata['observations']['observations']),observations)
                calibration = _thaw_json(checkpoints['scaled_size'].metadata['effective_parameters']['calibration'])
                self.assertEqual([calibration['px_width'],calibration['px_height'],calibration['unit']],scale)
                self.assertEqual(len(requests),10)
                self.assertEqual(requests[-1][0],'analysis.size')
                self.assertNotIn('calibration',requests[-3][1])
                self.assertEqual(outcome.record['complete_required_verified_public_calls'],10)
                self.assertEqual(len(checkpoints),15)
                self.assertFalse(outcome.record['scientifically_validated'])

    def test_typed_reader_enforces_actual_remaining_aggregate_budget(self):
        verified = self.verified()
        count = sum(a.nbytes for checkpoint in verified.checkpoints.values() for a in checkpoint.arrays.values())
        self.assertEqual(verified.retained_array_bytes,count)
        limits = CheckpointLimits(max_total_array_bytes=count,max_array_bytes=min(count,CheckpointLimits().max_array_bytes))
        self.assertEqual(self.verified(limits=limits).retained_array_bytes,count)
        with self.assertRaises(ValueError):
            self.verified(limits=replace(limits,max_total_array_bytes=count-1,
                                        max_array_bytes=min(limits.max_array_bytes,count-1)))

    def test_actual_empty_default_skips_analyses_and_explicit_seven_is_passed_unchanged(self):
        original = PlantCVReferenceRunner.run
        for explicit in (False,True):
            changes = {'threshold.binary':{'threshold':255}}
            if explicit:
                changes.update({'analysis.size':{'n_labels':7},'analysis.color':{'n_labels':7}})
            selected = spec(seed_recipe=selected_recipe('binary',**changes))
            args = arguments(self.fixtures,selected)
            calls = []
            def observe(runner,operation,arrays,parameters=None,**kwargs):
                calls.append((operation,deepcopy(parameters)))
                return original(runner,operation,arrays,parameters,**kwargs)
            with patch.object(PlantCVReferenceRunner,'run',observe):
                outcome = implementation.run_reference_calibrated_seed(**args)
            with self.subTest(explicit=explicit):
                self.assertEqual(outcome.child.record['label_facts']['maximum_label_id'],0)
                self.assertEqual(len(calls),10 if explicit else 7)
                self.assertEqual(outcome.status,'executed' if explicit else 'empty',outcome.record['phase_failure'])
                self.assertEqual(outcome.scaled_size.status,'executed' if explicit else 'unavailable',outcome.scaled_size.reason)
                if explicit:
                    self.assertEqual([p['n_labels'] for op,p in calls if op in ('analysis.size','analysis.color')],[7,7,7])
                    self.assertEqual(outcome.child.calls[1].status,'executed')
                    self.assertEqual(outcome.child.calls[2].status,'executed')
                    verified = reader.read_reference_calibrated_seed(args['output_root'],outcome.manifest_sha256,
                                                                   **read_arguments(args))
                    self.assertFalse(np.any(verified.checkpoints['seed.watershed'].arrays['labels']))
                else:
                    self.assertIsNone(outcome.manifest_sha256)
                    self.assertTrue(all(op not in ('analysis.size','analysis.color') for op,_ in calls))

    def _manifest_copy(self,record):
        args = self.cases['binary'][0]
        raw = _canonical_json(record,name='declared modified manifest fixture')
        digest = sha256(raw).hexdigest()
        path = args['output_root']/'calibrated'/'chains'/(digest+'.json')
        path.write_bytes(raw)
        self.addCleanup(path.unlink)
        return digest

    def test_changed_spec_source_crop_scale_and_duplicate_calls_are_rejected(self):
        args,outcome,_,_ = self.cases['binary']
        for path,value in ((('composition_sha256',),'0'*64),(('source_sha256',),'0'*64),
                (('crop','local_to_parent_xy'),[601,260]),
                (('scaled_size','details','calibration','px_height'),3.0),
                (('scaled_size','call_id'),outcome.correction.call_id),
                (('correction','checkpoint_sha256'),'0'*64)):
            record = _thaw_json(outcome.record)
            target = record
            for part in path[:-1]:
                target = target[part]
            target[path[-1]] = value
            digest = self._manifest_copy(record)
            with self.subTest(path=path),self.assertRaises((ValueError,FileNotFoundError)):
                reader.read_reference_calibrated_seed(args['output_root'],digest,**read_arguments(args))

    def replay(self, *, fault=None,event=None,cancel_after=None,invalid_scale=False,mutate=False,mutate_and_raise=False):
        """Declared replay of real results for orchestration, not public execution."""
        selected = spec()
        args = arguments(self.fixtures,selected)
        calls = []
        source_by_operation = {r.operation:r for r in self.source_results}
        size_results = [r for r in self.source_results if r.operation=='analysis.size']
        def run(runner,operation,arrays,parameters=None,**kwargs):
            calls.append((operation,deepcopy(parameters)))
            if operation==fault and (operation!='analysis.size' or 'calibration' not in parameters):
                raise ReferenceExecutionError('Declared injected source failure.')
            original = source_by_operation[operation]
            if operation=='analysis.size':
                original = size_results[1 if 'calibration' in parameters else 0]
            result = replace(original,arrays={k:v.copy() for k,v in original.arrays.items()},
                effective_parameters=deepcopy(dict(original.effective_parameters)),
                observations=deepcopy(dict(original.observations)),identity=deepcopy(dict(original.identity)))
            if invalid_scale and operation=='transform.auto_correct_color':
                result.effective_parameters['calibration_effects']['px_width'] = 0
            if (mutate or mutate_and_raise) and operation=='transform.auto_correct_color':
                arrays['rgb'] = arrays['rgb'].copy()
                arrays['rgb'][0,0,0] ^= 1
                if mutate_and_raise:
                    raise ReferenceExecutionError('Declared failure after input mutation.')
            if cancel_after==len(calls):
                event.set()
            return result
        with patch.object(PlantCVReferenceRunner,'run',run):
            outcome = implementation.run_reference_calibrated_seed(**args,cancel=event)
        return args,outcome,calls

    def test_declared_correction_failure_and_invalid_scale_withhold_parent(self):
        for changes in (dict(fault='transform.auto_correct_color'),dict(invalid_scale=True),dict(mutate=True)):
            args,outcome,calls = self.replay(**changes)
            with self.subTest(changes=changes):
                self.assertEqual(outcome.status,'partial')
                self.assertIsNone(outcome.manifest_sha256)
                self.assertEqual(len(calls),1)
                self.assertIsNone(outcome.child)
                self.assertEqual(outcome.record['extra_runner_request_count'],1)

    def test_declared_child_failures_still_attempt_independent_scaled_size(self):
        for fault in ('analysis.size','analysis.color'):
            args,outcome,calls = self.replay(fault=fault)
            with self.subTest(fault=fault):
                self.assertEqual(outcome.status,'partial',outcome.record['phase_failure'])
                self.assertIsNone(outcome.manifest_sha256)
                self.assertIsNotNone(outcome.correction.checkpoint_reference)
                self.assertIsNotNone(outcome.child.calls[0].checkpoint_reference)
                self.assertEqual(outcome.scaled_size.status,'executed',outcome.scaled_size.reason)
                self.assertEqual(calls[-1][0],'analysis.size')
                self.assertIn('calibration',calls[-1][1])

    def test_declared_mutation_on_raised_public_call_is_explicit_and_withholds_parent(self):
        args,outcome,calls = self.replay(mutate_and_raise=True)
        self.assertEqual(outcome.status,'partial')
        self.assertIsNone(outcome.manifest_sha256)
        self.assertEqual(len(calls),1)
        self.assertTrue(outcome.correction.details['input_contract_violated'])
        self.assertIn('Bound input changed',outcome.correction.reason)

    def test_declared_cancellation_retains_correction_without_parent(self):
        event = threading.Event()
        args,outcome,calls = self.replay(event=event,cancel_after=2)
        self.assertEqual(outcome.status,'cancelled')
        self.assertEqual(len(calls),2)
        self.assertIsNotNone(outcome.correction.checkpoint_reference)
        self.assertIsNone(outcome.manifest_sha256)

    def test_precommit_verification_failure_preserves_individual_checkpoints(self):
        with patch.object(reader,'_verify_complete_record',side_effect=ValueError('Declared precommit fault.')):
            args,outcome,calls = self.replay()
        self.assertEqual(outcome.status,'partial',outcome.record['phase_failure'])
        self.assertIsNone(outcome.manifest_sha256)
        self.assertIsNotNone(outcome.child.manifest_sha256)
        self.assertIsNotNone(outcome.scaled_size.checkpoint_reference)
        self.assertFalse(list((args['output_root']/'calibrated'/'chains').glob('*.json')))
        self.assertFalse(list((args['output_root']/'calibrated'/'chains').glob('*.tmp')))

    def test_late_mutating_callback_cannot_publish_a_broken_verified_revision(self):
        audited = {'path':None}
        original = reader._verify_complete_record
        def audit(record,store,**kwargs):
            verified = original(record,store,**kwargs)
            audited['path'] = Path(store)/(record['scaled_size']['checkpoint_sha256']+'.checkpoint')/'manifest.json'
            return verified
        def callback():
            if audited['path'] is not None:
                audited['path'].write_bytes(b'declared late callback mutation')
            return False
        with patch.object(reader,'_verify_complete_record',audit):
            args,outcome,calls = self.replay(event=callback)
        self.assertEqual(outcome.status,'executed',outcome.record['phase_failure'])
        verified = reader.read_reference_calibrated_seed(args['output_root'],outcome.manifest_sha256,
                                                       **read_arguments(args))
        self.assertEqual(verified.record['composition_sha256'],args['spec'].sha256)


if __name__=='__main__':
    unittest.main()
