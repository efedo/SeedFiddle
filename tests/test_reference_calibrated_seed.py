"""Actual public composition checks, followed by explicitly injected faults.

The direct oracle calls pinned PlantCV, never the adapters or coordinator.
Replay tests below check orchestration only; they are not numerical oracles.
"""

from tests.catalogue_fixtures import CatalogueTestCase
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


class CalibratedSpecTests(CatalogueTestCase):
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




    def _manifest_copy(self,record):
        args = self.cases['binary'][0]
        raw = _canonical_json(record,name='declared modified manifest fixture')
        digest = sha256(raw).hexdigest()
        path = args['output_root']/'calibrated'/'chains'/(digest+'.json')
        path.write_bytes(raw)
        self.addCleanup(path.unlink)
        return digest


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








if __name__=='__main__':
    unittest.main()
