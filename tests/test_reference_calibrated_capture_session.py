"""One actual capture; replayed results test session orchestration only."""
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from shutil import copytree
import os
import tempfile
import threading
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from seedvision.benchmarking import reference_calibrated_capture_session as session
from seedvision.benchmarking.checkpoints import CheckpointCancelled, CheckpointLimits, _canonical_json, _thaw_json
from seedvision.benchmarking.reference_runner import PlantCVReferenceRunner
from tests.test_native_calibrated_seed_recipe import fixture
from tests.test_reference_calibrated_seed import spec
from tests.test_reference_preprocessing import HAS_REFERENCE_ENV, REFERENCE_PYTHON


class CalibratedSessionAdmissionTests(unittest.TestCase):
    def test_crop_watershed_bound_does_not_reject_large_parent(self):
        selected = spec(seed_roi_xyxy=(600,260,728,356))
        plan = session._resource_plan((1000,4800), selected, CheckpointLimits())
        self.assertEqual(plan['watershed_shape'],[96,128])
        self.assertEqual(plan['prepared_numeric_envelope_bytes'],6*4_800_000)
        # A full parent capture would exceed the public watershed limit.
        self.assertGreater(42*4_800_000+3,128*1024*1024)

    def test_parent_transmission_roi_and_retained_limits_are_separate(self):
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

    def test_cancelled_or_invalid_preflight_does_not_read_source(self):
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


@unittest.skipUnless(HAS_REFERENCE_ENV,'Pinned public PlantCV environment is required.')
class CalibratedCaptureSessionTests(unittest.TestCase):
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

    def test_actual_ten_calls_decode_once_and_rgb_only_parent(self):
        self.assertEqual(len(self.requests),10)
        self.assertEqual(self.requests[0][0],'transform.auto_correct_color')
        self.assertEqual(self.requests[-1][0],'analysis.size')
        self.assertEqual(self.decode_calls,1)
        verified = self.captured.verified
        parent = verified.checkpoints['parent']
        self.assertEqual(set(parent.arrays),{'rgb'})
        np.testing.assert_array_equal(parent.arrays['rgb'],self.bgr[...,::-1])
        self.assertEqual(parent.identity.source_sha256,sha256(self.source_bytes).hexdigest())
        self.assertIs(parent.metadata['actual_public_reader'],False)
        self.assertEqual(verified.record['crop']['local_to_parent_xy'],(600,260))
        self.assertEqual(verified.record['scaled_measurement']['units'],'mm')
        self.assertFalse(verified.record['scientifically_validated'])
        self.assertEqual(len(verified.checkpoints),15)
        plan = parent.metadata['resource_plan']
        self.assertLessEqual(verified.retained_array_bytes,plan['complete_retained_numeric_envelope_bytes'])
        for checkpoint in verified.checkpoints.values():
            for array in checkpoint.arrays.values():
                self.assertFalse(array.flags.writeable)

    def test_saved_revision_opens_after_store_move_and_without_original_image(self):
        store,path = self._copy('moved')
        with (patch.object(session.inputs,'snapshot',side_effect=AssertionError('original source snapshot')),
             patch.object(session.inputs,'decode',side_effect=AssertionError('original source decode')),
             patch.object(session.inputs,'source_identity',side_effect=AssertionError('original source hash'))):
            opened = session.open_calibrated_reference_capture(path)
        self.assertIsNone(opened.outcome)
        self.assertEqual(opened.spec,self.selected)
        self.assertEqual(opened.manifest_sha256,self.captured.manifest_sha256)
        self.assertEqual(opened.verified.record,self.captured.verified.record)
        self.assertEqual(opened.verified.retained_array_bytes,
            sum(a.nbytes for checkpoint in opened.verified.checkpoints.values() for a in checkpoint.arrays.values()))

    def test_open_actual_retained_budget_and_one_byte_below(self):
        total = self.captured.verified.retained_array_bytes
        limits = replace(CheckpointLimits(),max_total_array_bytes=total,max_array_bytes=total)
        opened = session.open_calibrated_reference_capture(self.captured.manifest_path,limits=limits)
        self.assertEqual(opened.verified.retained_array_bytes,total)
        with self.assertRaises(ValueError):
            session.open_calibrated_reference_capture(self.captured.manifest_path,
                limits=replace(limits,max_total_array_bytes=total-1,max_array_bytes=total-1))

    def test_sidecar_rejects_swapped_spec_digest_revision_and_noncanonical_bytes(self):
        store,path = self._copy('sidecar-faults')
        sidecar = path.with_name(path.stem+'.composition.json')
        raw = sidecar.read_bytes()
        value = session._strict_json_loads(raw)
        changes = [lambda s:s.update(composition_sha256='f'*64),
                   lambda s:s.update(manifest_sha256='f'*64),
                   lambda s:s.update(composition=spec('mean').to_dict()),
                   lambda s:s['composition'].update(seed_roi_xyxy=[601,260,729,356])]
        for mutate in changes:
            changed = deepcopy(value)
            mutate(changed)
            sidecar.write_bytes(_canonical_json(changed,name='declared sidecar fault'))
            with self.subTest(change=mutate),self.assertRaises(ValueError):
                session.open_calibrated_reference_capture(path)
        for invalid in (raw+b'\n',b'{"schema":1,"schema":2}',b'{"schema":NaN}'):
            sidecar.write_bytes(invalid)
            with self.subTest(raw=invalid[:30]),self.assertRaises(ValueError):
                session.open_calibrated_reference_capture(path)
        sidecar.write_bytes(raw)
        self.assertEqual(session.open_calibrated_reference_capture(path).spec,self.selected)

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

    def test_manifest_checkpoint_caps_and_path_reject(self):
        store,path = self._copy('corruption')
        with self.assertRaises(ValueError):
            session.open_calibrated_reference_capture(path,limits=replace(CheckpointLimits(),max_metadata_bytes=100))
        with patch.object(Path,'open',side_effect=AssertionError('unexpected manifest I/O')):
            for bad in (self.root/'a.json',path.with_name(path.stem+'.composition.json'),self.root/(path.stem+'.json')):
                with self.subTest(path=bad),self.assertRaises(ValueError):
                    session.open_calibrated_reference_capture(bad)
        parent = self.captured.verified.checkpoints['parent']
        (store/'inputs'/(parent.checkpoint_sha256+'.checkpoint')/'manifest.json').write_bytes(b'corrupt')
        with self.assertRaises(ValueError):
            session.open_calibrated_reference_capture(path)

    def test_source_changed_before_preparation_makes_no_public_calls(self):
        with patch.object(PlantCVReferenceRunner,'run') as public,self.assertRaises(ValueError):
            self._capture('bad-stamp',expected_stamp=(0,0,0,0,0))
        public.assert_not_called()

    def test_source_changed_after_first_call_withholds_saved_revision(self):
        source = self._source('changed-between-calls')
        store = self.root/'changed-between-calls'
        def mutate(index,operation,arrays):
            if index == 1:
                source.write_bytes(self.source_bytes+b'changed')
        with patch.object(PlantCVReferenceRunner,'run',self._replay(mutate)),self.assertRaises(ValueError):
            session.capture_calibrated_reference_path(self.selected,source,store,PlantCVReferenceRunner(REFERENCE_PYTHON))
        self.assertEqual(list(store.glob('calibrated/chains/*.json')),[])

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

    def test_write_fsync_and_replace_failures_retract_only_owned_parent(self):
        original_replace = os.replace
        def replace_failure(source,destination):
            if str(destination).endswith('.composition.json'):
                raise OSError('Declared sidecar replacement failure.')
            return original_replace(source,destination)
        failures = [patch.object(session.tempfile,'NamedTemporaryFile',side_effect=OSError('Declared sidecar write failure.')),
                    patch.object(session.os,'fsync',side_effect=OSError('Declared fsync failure.')),
                    patch.object(session.os,'replace',replace_failure)]
        for index,failure in enumerate(failures):
            name = 'publish-fault-'+str(index)
            # Apply file faults only after the coordinator completes, so its
            # independently committed checkpoints/child remain valid evidence.
            actual = session.run_reference_calibrated_seed
            active = []
            def complete(*args,**kwargs):
                outcome = actual(*args,**kwargs)
                failure.start()
                active.append(failure)
                return outcome
            try:
                with (self.subTest(index=index),patch.object(PlantCVReferenceRunner,'run',self._replay()),
                     patch.object(session,'run_reference_calibrated_seed',complete),self.assertRaises(OSError)):
                    self._capture(name)
            finally:
                for current in active:
                    current.stop()
            self._assert_uncommitted(self.root/name)

    def test_verification_cancel_or_failure_preserves_checkpoints(self):
        for reason in (CheckpointCancelled('Declared saved audit cancellation.'),ValueError('Declared saved audit failure.')):
            name = 'audit-'+type(reason).__name__
            with (patch.object(PlantCVReferenceRunner,'run',self._replay()),
                 patch.object(session,'read_reference_calibrated_seed',side_effect=reason),self.assertRaises(type(reason))):
                self._capture(name)
            self._assert_uncommitted(self.root/name)

    def test_changed_parent_replacement_survives_failure_cleanup(self):
        name = 'replacement-preserved'
        replacement = b'declared changed replacement'
        changed = []
        def fail(store,digest,**kwargs):
            path = Path(store)/'calibrated'/'chains'/(digest+'.json')
            path.write_bytes(replacement)
            changed.append(path)
            raise ValueError('Declared audit failure after replacement.')
        with (patch.object(PlantCVReferenceRunner,'run',self._replay()),
             patch.object(session,'read_reference_calibrated_seed',fail),self.assertRaises(ValueError)):
            self._capture(name)
        self.assertEqual(changed[0].read_bytes(),replacement)
        self.assertEqual(list(changed[0].parent.glob('*.tmp')),[])
        self.assertEqual(list(changed[0].parent.glob('*.composition.json')),[])

    def test_last_caller_callback_precedes_saved_graph_audit(self):
        name = 'callback-boundary'
        audited = False
        late_calls = []
        actual = session.read_reference_calibrated_seed
        def audit(*args,**kwargs):
            nonlocal audited
            self.assertIsNone(kwargs['cancel'])
            verified = actual(*args,**kwargs)
            audited = True
            return verified
        def cancel():
            if audited:
                late_calls.append(True)
                return True
            return False
        with patch.object(PlantCVReferenceRunner,'run',self._replay()),patch.object(session,'read_reference_calibrated_seed',audit):
            captured = self._capture(name,cancel=cancel)
        self.assertEqual(late_calls,[])
        self.assertEqual(session.open_calibrated_reference_capture(captured.manifest_path).manifest_sha256,captured.manifest_sha256)

    def test_cancellation_after_commit_keeps_saved_revision(self):
        event = threading.Event()
        actual = os.replace
        def after_commit(source,destination):
            actual(source,destination)
            if str(destination).endswith('.composition.json'):
                event.set()
        with patch.object(PlantCVReferenceRunner,'run',self._replay()),patch.object(session.os,'replace',after_commit):
            captured = self._capture('post-commit-cancel',cancel=event)
        self.assertTrue(event.is_set())
        self.assertTrue(captured.manifest_path.exists())
        self.assertEqual(session.open_calibrated_reference_capture(captured.manifest_path).spec,self.selected)

    def test_exception_after_sidecar_replace_preserves_committed_pair(self):
        actual = os.replace
        saved = []
        def after_commit(source,destination):
            actual(source,destination)
            if str(destination).endswith('.composition.json'):
                saved.append(Path(destination))
                raise OSError('Declared unknown outcome after completed replacement.')
        with (patch.object(PlantCVReferenceRunner,'run',self._replay()),
              patch.object(session.os,'replace',after_commit),self.assertRaises(OSError)):
            self._capture('post-commit-exception')
        path = saved[0].with_name(saved[0].name.removesuffix('.composition.json')+'.json')
        self.assertTrue(path.exists())
        self.assertEqual(session.open_calibrated_reference_capture(path).spec,self.selected)


if __name__ == '__main__':
    unittest.main()
