"""One actual capture; replayed results test session orchestration only."""

from tests.functional_units import functional_units
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from shutil import copytree
import os
import tempfile
import threading
import unittest
from tests.catalogue_fixtures import CatalogueTestCase
from unittest.mock import patch

import cv2
import numpy as np

from seedvision.benchmarking import reference_calibrated_capture_session as session
from seedvision.benchmarking.checkpoints import CheckpointCancelled, CheckpointLimits, _canonical_json, _thaw_json
from seedvision.benchmarking.reference_runner import PlantCVReferenceRunner
from tests.test_native_calibrated_seed_recipe import fixture
from tests.test_reference_calibrated_seed import spec
from tests.test_reference_preprocessing import HAS_REFERENCE_ENV, REFERENCE_PYTHON


@functional_units
class CalibratedSessionAdmissionTests(CatalogueTestCase):
    def check_test_crop_watershed_bound_does_not_reject_large_parent(self):
        selected = spec(seed_roi_xyxy=(600,260,728,356))
        plan = session._resource_plan((1000,4800), selected, CheckpointLimits())
        self.assertEqual(plan['watershed_shape'],[96,128])
        self.assertEqual(plan['prepared_numeric_envelope_bytes'],6*4_800_000)
        # A full parent capture would exceed the public watershed limit.
        self.assertGreater(42*4_800_000+3,128*1024*1024)

    def check_test_parent_transmission_roi_and_retained_limits_are_separate(self):
        selected = spec()
        with self.assertRaisesRegex(ValueError,'transmission'):
            session._resource_plan((4000,6000), selected, CheckpointLimits())
        with self.assertRaisesRegex(ValueError,'inside'):
            session._resource_plan((300,700), selected, CheckpointLimits())
        total = session._resource_plan((640,900),selected,CheckpointLimits())['complete_retained_numeric_envelope_bytes']
        exact = replace(CheckpointLimits(), max_total_array_bytes=total,max_array_bytes=total)
        self.assertEqual(session._resource_plan((640,900),selected,exact)['complete_retained_numeric_envelope_bytes'],total)
        with self.assertRaisesRegex(ValueError,'Complete retained'):
            session._resource_plan((640,900),selected,replace(exact,max_total_array_bytes=total-1,max_array_bytes=total-1))

    def check_test_cancelled_or_invalid_preflight_does_not_read_source(self):
        event = threading.Event()
        event.set()
        runner = PlantCVReferenceRunner(REFERENCE_PYTHON)
        with patch.object(session.inputs,'source_identity',side_effect=AssertionError('source read')):
            with self.assertRaises(CheckpointCancelled):
                session.capture_calibrated_reference_path(spec(),'missing.png','missing',runner,cancel=event)
            for change in ({'spec':None},{'runner':None},{'limits':None}):
                values = {'spec':spec(),'source_path':'missing.png','store':'missing','runner':runner}|change
                with self.subTest(change=change),self.assertRaises(TypeError):
                    session.capture_calibrated_reference_path(**values)
        with patch.object(Path,'open',side_effect=AssertionError('saved read')):
            with self.assertRaises(CheckpointCancelled):
                session.open_calibrated_reference_capture('missing.json',cancel=event)

    def test_validation_and_recovery(self):
        """3 related CalibratedSessionAdmissionTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_crop_watershed_bound_does_not_reject_large_parent)
        self.scenario(self.check_test_parent_transmission_roi_and_retained_limits_are_separate)
        self.scenario(self.check_test_cancelled_or_invalid_preflight_does_not_read_source)


@unittest.skipUnless(HAS_REFERENCE_ENV,'Pinned public PlantCV environment is required.')
class CalibratedCaptureSessionTests(CatalogueTestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix='calibrated-capture-session-')
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name)
        cls.source = cls.root/'source.png'
        cls.bgr = fixture()
        ok,encoded = cv2.imencode('.png',cls.bgr)
        if not ok:
            raise AssertionError('Fixture encoding failed.')
        cls.source_bytes = encoded.tobytes()
        cls.source.write_bytes(cls.source_bytes)
        cls.selected = spec()
        cls.store = cls.root/'actual'
        cls.results,cls.requests = [],[]
        original = PlantCVReferenceRunner.run
        def observe(runner,operation,arrays,parameters=None,**kwargs):
            result = original(runner,operation,arrays,parameters,**kwargs)
            cls.results.append(result)
            cls.requests.append((operation,deepcopy(parameters)))
            return result
        with patch.object(PlantCVReferenceRunner,'run',observe),patch.object(session.inputs,'decode',wraps=session.inputs.decode) as decode:
            cls.captured = session.capture_calibrated_reference_path(cls.selected,cls.source,cls.store,
                PlantCVReferenceRunner(REFERENCE_PYTHON),expected_stamp=session.inputs.stamp(cls.source))
            cls.decode_calls = decode.call_count
        if cls.captured.verified is None:
            raise AssertionError(_thaw_json(cls.captured.outcome.record))

    def _replay(self, mutate=None):
        index = 0
        def run(runner,operation,arrays,parameters=None,**kwargs):
            nonlocal index
            expected = self.requests[index]
            self.assertEqual((operation,parameters),expected)
            result = self.results[index]
            index += 1
            if mutate:
                mutate(index,operation,arrays)
            return replace(result,arrays={name:value.copy() for name,value in result.arrays.items()},
                effective_parameters=deepcopy(dict(result.effective_parameters)),identity=deepcopy(dict(result.identity)))
        return run

    def _source(self,name):
        source = self.root/(name+'.png')
        source.write_bytes(self.source_bytes)
        return source

    def _capture(self,name,**kwargs):
        return session.capture_calibrated_reference_path(self.selected,self._source(name),self.root/name,
            PlantCVReferenceRunner(REFERENCE_PYTHON),**kwargs)

    def _copy(self,name):
        store = self.root/name
        copytree(self.store,store)
        return store,store/'calibrated'/'chains'/(self.captured.manifest_sha256+'.json')

    def _assert_uncommitted(self,store):
        chains = store/'calibrated'/'chains'
        self.assertEqual(list(chains.glob('*.json')),[])
        self.assertEqual(list(chains.glob('*.tmp')),[])
        self.assertTrue(any(store.glob('*.checkpoint')))
        self.assertTrue(any((store/'seed'/'chains').glob('*.json')))





    def test_partial_manifest_and_bad_preparation_metadata_reject(self):
        store,path = self._copy('partial')
        record = session._strict_json_loads(path.read_bytes())
        record['status'] = 'partial'
        raw = _canonical_json(record,name='declared partial')
        digest = sha256(raw).hexdigest()
        changed_path = path.with_name(digest+'.json')
        changed_path.write_bytes(raw)
        selected = {'schema':session.SIDECAR_SCHEMA,'manifest_sha256':digest,
            'composition_sha256':self.selected.sha256,'composition':self.selected.to_dict()}
        changed_path.with_name(digest+'.composition.json').write_bytes(_canonical_json(selected,name='partial sidecar'))
        with self.assertRaises(ValueError):
            session.open_calibrated_reference_capture(changed_path)
        verified = self.captured.verified
        parent = verified.checkpoints['parent']
        for mutation in (lambda m:m.update(actual_public_reader=True),
                         lambda m:m['preparation']['decoder'].update(opencv_build_sha256='f'*64),
                         lambda m:m['preparation']['code'].update(session_sha256='f'*64)):
            metadata = _thaw_json(parent.metadata)
            mutation(metadata)
            altered = replace(parent,metadata=metadata)
            changed = replace(verified,checkpoints={**dict(verified.checkpoints),'parent':altered})
            with self.subTest(mutation=mutation),self.assertRaisesRegex(ValueError,'preparation'):
                session._verify_preparation(changed,self.selected,CheckpointLimits())




    def test_final_source_rehash_detects_bytes_even_with_declared_constant_stamp(self):
        source = self._source('changed-byte-hash')
        store = self.root/'changed-byte-hash'
        stamp = session.inputs.stamp(source)
        def mutate(index,operation,arrays):
            if index == 10:
                source.write_bytes(bytes([self.source_bytes[0]^1])+self.source_bytes[1:])
        with (patch.object(PlantCVReferenceRunner,'run',self._replay(mutate)),
             patch.object(session.inputs,'stamp',return_value=stamp),self.assertRaisesRegex(ValueError,'original bytes')):
            session.capture_calibrated_reference_path(self.selected,source,store,PlantCVReferenceRunner(REFERENCE_PYTHON))
        self._assert_uncommitted(store)








if __name__ == '__main__':
    unittest.main()
