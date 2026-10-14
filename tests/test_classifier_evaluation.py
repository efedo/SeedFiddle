"""CPU artifact/evidence binding; synthetic facts do not certify study conduct."""
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from hashlib import sha256
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from seedvision.benchmarking import classifier_evaluation as adapter
from seedvision.benchmarking.manifest import AccessPurpose, AccessRecord, SplitRole
from seedvision.cuda import plantcv_naive_bayes as bayes
from seedvision.model_io import kmeans_model_artifact as codec
from seedvision.model_io import naive_bayes_provenance as provenance
from seedvision.model_io import plantcv_kmeans_model as models
from tests.test_benchmark_manifest import _ready_manifest, _sample
from tests.test_kmeans_model_artifact import _conversion, _native


def _manifest(source='2'*64, *, purpose=AccessPurpose.TRAINING):
    samples = (
        _sample('development', source_sha256=source, biological_lot_id='lot_dev',
                specimen_group_id='specimen_dev', capture_session_id='capture_dev'),
        _sample('locked', SplitRole.LOCKED_TEST, biological_lot_id='lot_test',
                specimen_group_id='specimen_test', capture_session_id='capture_test'),
    )
    return _ready_manifest(samples=samples, access_records=(
        AccessRecord('train', purpose, ('development',), 'declared_operator'),))


def _bayes_sample(raw):
    """Typed synthetic sample-table provenance, deliberately no image membership."""
    model = bayes.parse_plantcv_pdf(raw)
    count, total = len(model.class_names), 2*len(model.class_names)
    scratch = 3*256*256*8 + count*3*256*8 + 512 + 2*1024*1024
    calculation = {}
    for module in provenance.CALCULATION_MODULES:
        calculation[module] = dict(name=module.rsplit('.', 1)[-1]+'.py',
            source_sha256='4'*64, dataclasses=[], imported_aliases=[],
            loaded_code=[dict(name='recorded_function', code_sha256='5'*64,
                             defaults_sha256='6'*64)], runtime_constants={})
        if module == 'seedvision.cuda.plantcv_range':
            calculation[module]['computed_constants'] = '7'*64
    channel = dict(samples=2, weight=.5, mean=10., covariance=1., effective_samples=2.,
                   scott_factor=1., bandwidth=1., covariance_absolute_budget=0.,
                   pdf_absolute_budget=0., numerical_profile=provenance.NUMERICAL_PROFILE)
    return dict(kind='native_bayes_sample_training', parser_id=provenance.PARSER_ID,
        class_names=list(model.class_names), sample_counts=[2]*count,
        model_sha256=model.source_sha256, tables_sha256=model.tables_sha256,
        sample_origin_path=None, sample_sha256='1'*64, sample_bytes=100,
        parsed_samples_sha256='2'*64, headers=list(model.class_names),
        fit_plan=dict(classes=count, sample_counts=[2]*count, samples=total,
            work=total*780+count*(3*256*256+3*256), scratch_bytes=scratch,
            live_bytes=scratch+total*3),
        channel_fits=[[dict(channel) for _ in range(3)] for _ in range(count)],
        calculation=calculation, scientific_validation=False,
        runtime={name:'recorded' for name in ('torch', 'numpy', 'torch_cuda', 'device',
                                            'device_name', 'nvrtc', 'cuda_driver_api')},
        source_pins=dict(provenance.SOURCE_PINS))


class ClassifierEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.model = models.model_from_centres(((0.,),), 1, 1)
        self.metadata = _native(self.model)
        self.path = self.root/'frozen.json'
        self.raw = codec.encode_kmeans_artifact(self.model, self.metadata)
        self.path.write_bytes(self.raw)

    def bind(self, manifest=None, **overrides):
        arguments = dict(family='kmeans', model_path=self.path, method_id='candidate',
            backend_id='native_cuda', recipe_sha256='a'*64, training_access_ids=('train',))
        arguments.update(overrides)
        binding, receipt = adapter.bind_classifier_evaluation(manifest or _manifest(), **arguments)
        return binding, json.loads(receipt), receipt

    def test_native_membership_binds_actual_full_artifact_and_training_bytes(self):
        binding, receipt, text = self.bind()
        self.assertEqual(binding.model_sha256, sha256(self.raw).hexdigest())
        self.assertEqual(binding.training_evidence_sha256,
                         sha256(codec._canonical_metadata(self.metadata)).hexdigest())
        self.assertEqual(receipt['source_membership'],
                         [{'source_sha256':'2'*64, 'sample_ids':['development']}])
        self.assertTrue(receipt['evaluation_eligible'])
        self.assertFalse(receipt['scientifically_validated'])
        self.assertEqual(receipt['manifest_sha256'], _manifest().manifest_sha256)
        self.assertEqual(text, json.dumps(receipt, sort_keys=True, ensure_ascii=False,
                                         separators=(',', ':'), allow_nan=False))
        with self.assertRaises(FrozenInstanceError):
            binding.model_sha256 = '0'*64

    def test_relocated_same_bytes_bind_identically_without_opening_training_images(self):
        copied = self.root/'copied.json'
        copied.write_bytes(self.raw)
        with patch.object(adapter, '_read', wraps=adapter._read) as reader:
            first = self.bind()
            second = self.bind(model_path=copied)
        self.assertEqual([call.args[0] for call in reader.call_args_list], [self.path, copied])
        self.assertEqual(first, second)

    def test_source_membership_must_belong_to_selected_access_not_any_manifest_sample(self):
        manifest = _manifest(source='3'*64)
        extra = _sample('other', source_sha256='2'*64, biological_lot_id='lot_other',
                        specimen_group_id='specimen_other', capture_session_id='capture_other')
        manifest = replace(manifest, samples=manifest.samples+(extra,))
        with self.assertRaisesRegex(ValueError, 'outside selected training'):
            self.bind(manifest)

    def test_training_access_absent_wrong_purpose_and_empty_reject_before_artifact_read(self):
        for manifest, identifiers in ((_manifest(), ('absent',)),
            (_manifest(purpose=AccessPurpose.PARAMETER_DEVELOPMENT), ('train',)),
            (_manifest(), ())):
            with self.subTest(identifiers=identifiers), patch.object(adapter, '_read') as reader:
                with self.assertRaises(ValueError):
                    self.bind(manifest, training_access_ids=identifiers)
                reader.assert_not_called()

    def test_training_validation_samples_reject_even_when_global_audit_permits_access(self):
        manifest = _manifest()
        development = replace(manifest.samples[0], split_role=SplitRole.VALIDATION)
        manifest = replace(manifest, samples=(development, manifest.samples[1]))
        with self.assertRaisesRegex(ValueError, 'development samples only'):
            self.bind(manifest)

    def test_locked_training_and_connected_group_leakage_block_before_read(self):
        manifest = _manifest()
        locked_access = replace(manifest.access_records[0], sample_ids=('locked',))
        group_alias = replace(manifest.samples[1], biological_lot_id='lot_dev')
        for bad in (replace(manifest, access_records=(locked_access,)),
                    replace(manifest, samples=(manifest.samples[0], group_alias)),
                    replace(manifest, access_log_complete=False,
                            access_log_unavailable_reason='not supplied')):
            with self.subTest(bad=bad), patch.object(adapter, '_read') as reader:
                with self.assertRaisesRegex(ValueError, 'not evaluation eligible'):
                    self.bind(bad)
                reader.assert_not_called()

    def test_conversion_binds_actual_typed_bytes_but_membership_remains_unavailable(self):
        self.path.write_bytes(codec.encode_kmeans_artifact(self.model, _conversion(self.model)))
        _, receipt, _ = self.bind()
        self.assertFalse(receipt['evaluation_eligible'])
        self.assertIsNone(receipt['source_membership'])
        self.assertEqual(receipt['source_membership_state'], 'unavailable')

    def test_legacy_numeric_model_requires_typed_evidence_and_can_use_matching_external_record(self):
        self.path.write_bytes(models.encode_kmeans_model(self.model))
        with self.assertRaisesRegex(ValueError, 'requires typed training evidence'):
            self.bind()
        training = self.root/'training.json'
        raw = codec._canonical_metadata(self.metadata)
        training.write_bytes(raw)
        binding, receipt, _ = self.bind(training_path=training)
        self.assertTrue(receipt['evaluation_eligible'])
        self.assertEqual(binding.training_evidence_sha256, sha256(raw).hexdigest())

    def test_external_evidence_cannot_replace_embedded_evidence(self):
        training = self.root/'training.json'
        training.write_bytes(codec._canonical_metadata(_conversion(self.model)))
        with self.assertRaisesRegex(ValueError, 'differs from embedded'):
            self.bind(training_path=training)

    def test_wrong_centres_identity_and_corrupt_numeric_artifacts_reject(self):
        training = self.root/'training.json'
        other = models.model_from_centres(((1.,),), 1, 1)
        training.write_bytes(codec._canonical_metadata(_native(other)))
        with self.assertRaises(ValueError):
            self.bind(training_path=training)
        self.path.write_bytes(self.raw.replace(b'0.0', b'NaN', 1))
        with self.assertRaises(ValueError):
            self.bind()

    def test_duplicate_noncanonical_and_nonfinite_external_json_reject(self):
        training = self.root/'training.json'
        for raw in (b'{"kind":"x","kind":"y"}', b'{"kind":NaN}',
                    json.dumps(self.metadata, indent=2).encode()):
            training.write_bytes(raw)
            with self.subTest(raw=raw[:40]), self.assertRaises(ValueError):
                self.bind(training_path=training)

    def test_bounded_reader_observes_growth_without_trusting_stat(self):
        with patch.object(Path, 'open', return_value=io.BytesIO(b'x'*17)):
            with self.assertRaisesRegex(ValueError, 'byte limit'):
                adapter._read(self.path, 16)
        with patch.object(Path, 'open', side_effect=OSError('unreadable')):
            with self.assertRaises(ValueError):
                adapter._read(self.path, 16)

    def test_bayes_sample_record_is_typed_model_bound_and_honestly_unverified(self):
        rows = ['class\tchannel\t'+'\t'.join(map(str, range(256)))]
        rows.extend('\t'.join(('leaf', channel, *(['0.5']*256))) for channel in bayes.CHANNELS)
        raw = ('\n'.join(rows)+'\n').encode()
        self.path.write_bytes(raw)
        training = self.root/'bayes-training.json'
        metadata = _bayes_sample(raw)
        evidence = json.dumps(metadata, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
        training.write_bytes(evidence)
        binding, receipt, _ = self.bind(family='naive_bayes', training_path=training)
        self.assertEqual(binding.model_sha256, sha256(raw).hexdigest())
        self.assertEqual(binding.training_evidence_sha256, sha256(evidence).hexdigest())
        self.assertEqual(receipt['numeric_identity']['class_names'], ['leaf'])
        self.assertFalse(receipt['evaluation_eligible'])
        self.assertIsNone(receipt['source_membership'])
        metadata['tables_sha256'] = '0'*64
        training.write_text(json.dumps(metadata, sort_keys=True, separators=(',', ':')), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'bytes/tables'):
            self.bind(family='naive_bayes', training_path=training)

    def test_bayes_directory_membership_adapter_preserves_repeated_consumed_occurrences(self):
        # This isolates membership plumbing; typed parser acceptance has its own
        # production parser checks and the unmocked sample-record test above.
        with patch.object(adapter, '_artifacts', return_value=(
                '4'*64, '5'*64, 'native_bayes_directory_training',
                {'class_names':['plant', 'background'], 'tables_sha256':'6'*64},
                ['2'*64, '2'*64])):
            _, receipt, _ = self.bind(family='naive_bayes')
        self.assertTrue(receipt['evaluation_eligible'])
        self.assertEqual(len(receipt['source_membership']), 2)

    def test_bad_family_and_mutable_or_duplicate_access_ids_reject(self):
        for changes in ({'family':'pickle'}, {'training_access_ids':['train']},
                        {'training_access_ids':('train','train')}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.bind(**changes)

    def test_actual_native_cuda_training_artifact_binds_consumed_source(self):
        # Actual trainer witness, separate from synthetic codec fixtures above.
        # The supplied study identities are engineering fixtures, not lab truth.
        import cv2
        import numpy as np
        import torch
        from seedvision.benchmarking.native_kmeans_training import (
            KMeansTrainingRequest, train_kmeans_directory,
        )
        if not torch.cuda.is_available():
            self.skipTest('Actual native directory training requires CUDA.')
        training_dir = self.root/'actual-training'
        training_dir.mkdir()
        image = training_dir/'source.png'
        self.assertTrue(cv2.imwrite(str(image), np.arange(64, dtype=np.uint8).reshape(8, 8)))
        source_sha = sha256(image.read_bytes()).hexdigest()
        completion = train_kmeans_directory(KMeansTrainingRequest(
            str(training_dir), patch_size=1, clusters=1, seed=1, n_init=1))
        self.assertTrue(completion.cache.rows.features.is_cuda)
        self.path.write_bytes(completion.artifact_payload)
        binding, receipt, _ = self.bind(_manifest(source_sha))
        self.assertTrue(receipt['evaluation_eligible'])
        self.assertEqual(binding.model_sha256, sha256(completion.artifact_payload).hexdigest())
        self.assertEqual(receipt['source_membership'],
                         [{'source_sha256':source_sha, 'sample_ids':['development']}])
        self.assertFalse(receipt['scientifically_validated'])
