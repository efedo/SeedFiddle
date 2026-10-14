"""Persistent model metadata, legacy compatibility and full-file identity."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from seedvision.benchmarking import kmeans_model_artifact as native
from seedvision.benchmarking import native_kmeans_recipe as recipes
from seedvision.cuda import plantcv_kmeans_model as models


def _conversion(model):
    return dict(kind='plantcv_conversion', numeric_identity=native.numeric_model_identity(model),
        input=dict(source_sha256='1'*64, source_bytes=123,
            estimator_class='sklearn.cluster._kmeans.MiniBatchKMeans'),
        environment=dict(plantcv='4.11.3', numpy='2.2.6', sklearn='1.9.1', joblib='1.6.0'),
        geometry=dict(patch_size=model.patch_size, channels=model.channels,
            method='unique-supported-feature-width'))


def _native(model, sampling=None):
    request = dict(directory='D:\\recorded-training', prefix='', num_imgs=0,
        patch_size=1, sampling=sampling,
        sampling_kind='all' if sampling is None else 'count' if type(sampling) is int else 'fraction',
        clusters=1, seed=1, n_init=1)
    manifest = dict(directory=request['directory'], listing=['a.png'], selected_names=['a.png'],
        prefix='', num_imgs=0, occurrences=[dict(index=0, name='a.png', status='accepted',
            source_sha256='2'*64, source_bytes=123, shape=[2, 2])], source_bytes=123, decoded_bytes=16)
    manifest['manifest_sha256'] = sha256(native._canonical_metadata(manifest)).hexdigest()
    rows = 4 if sampling is None else min(sampling, 4) if type(sampling) is int else int(4*sampling)
    return dict(kind='native_training', numeric_identity=native.numeric_model_identity(model),
        request=request, manifest=manifest, patch_seed=1,
        fit_parameters=dict(init='k-means++', source_n_threads=None, seed=1, max_iter=100,
            batch_size=1024, n_init=1, init_size=None, tol=0.0, max_no_improvement=10,
            reassignment_ratio=.01, compute_labels=True),
        fit_plan=dict(rows=rows, features=1, clusters=1, batch_size=rows, init_size=rows,
            n_init=1, n_steps=100, adjusted_init_size=False, work=1, peak_live_bytes=1, cpu_metadata_bytes=1),
        fit_result=dict(n_steps=12, n_iter=12, inertia=dict(lo=0.0, hi=0.0),
            scaled_tolerance=dict(lo=0.0, hi=0.0), source_n_threads=None,
            source_reduction_profile='all-valid-openmp-reductions-v1',
            random_identity=dict(schema='seedfiddle.kmeans-randomstate.v1', numpy='2.5.1', seed=1,
                algorithm='MT19937', calls=1, state_sha256='3'*64, draw_sha256='4'*64)),
        calculation=dict(row_sha256='5'*64, fit_modules={name: '6'*64 for name in native.FIT_MODULES},
            runtime=dict(torch='2.13.0', torch_cuda='13.0', numpy='2.5.1', opencv='5.0.0',
                qt='6.11.1', device='cuda:0', device_name='recorded device')),
        source_pins=dict(plantcv_version='4.11.3',
            train_sha256='dbfb8a528e92984a20e73c7f265a41efa9fa088b42ed4f65addbe6ab40ee8b4d',
            predict_sha256='09019da00e0713c51cff31ffd240f7bf8aa0b4e228f6d88f2bed9c1d40430392',
            sklearn_kmeans_sha256='0cf335b8b9b4abe67259ccdcbaf07f02bd7feee31e63a06a1cce32a87fcd375d'))


class KMeansModelArtifactTests(unittest.TestCase):
    def setUp(self):
        self.model = models.model_from_centres(((-0.0,), (2.5,), (-0.0,)), 1, 1)

    def test_legacy_codec_bytes_and_identity_remain_unchanged(self):
        raw = models.encode_kmeans_model(self.model)
        artifact = native.parse_kmeans_artifact(raw)
        self.assertEqual(artifact.model, models.parse_kmeans_model(raw))
        self.assertIsNone(artifact.provenance)
        self.assertIsNone(artifact.provenance_sha256)
        self.assertEqual(models.encode_kmeans_model(artifact.model), raw)

    def test_roundtrip_keeps_centres_signed_zero_and_outer_file_identity(self):
        provenance = _conversion(self.model)
        raw = native.encode_kmeans_artifact(self.model, provenance)
        artifact = native.parse_kmeans_artifact(raw)
        self.assertEqual(artifact.model.centres, self.model.centres)
        self.assertIn(b'-0.0', raw)
        self.assertEqual(artifact.model.centres_sha256, self.model.centres_sha256)
        self.assertEqual(artifact.model.source_sha256, sha256(raw).hexdigest())
        self.assertEqual(artifact.model.source_bytes, len(raw))
        self.assertEqual(artifact.provenance, provenance)
        artifact.provenance['input']['source_bytes'] = 999
        self.assertEqual(artifact.provenance, provenance)
        self.assertEqual(native.encode_kmeans_artifact(artifact.model, artifact.provenance), raw)
        with self.assertRaises(models.KMeansModelError):
            models.parse_kmeans_model(raw)

    def test_native_manifest_sampling_defaults_rng_and_none_profile_roundtrip(self):
        model = models.model_from_centres(((0.0,),), 1, 1)
        for sample in (None, 1, .5):
            with self.subTest(sample=sample):
                provenance = _native(model, sample)
                loaded = native.parse_kmeans_artifact(native.encode_kmeans_artifact(model, provenance))
                self.assertEqual(loaded.provenance, provenance)
                self.assertIs(type(loaded.provenance['request']['sampling']), type(sample))
                self.assertIsNone(loaded.provenance['fit_result']['source_n_threads'])

    def test_checksum_and_numeric_tampering_rejects(self):
        value = json.loads(native.encode_kmeans_artifact(self.model, _conversion(self.model)))
        for change in ('checksum', 'centres', 'identity', 'geometry'):
            bad = deepcopy(value)
            if change == 'checksum':
                bad['provenance_sha256'] = '0'*64
            elif change == 'centres':
                bad['numeric_model']['centres'][0][0] = 1.
            elif change == 'identity':
                bad['provenance']['numeric_identity']['source_bytes'] += 1
            else:
                bad['provenance']['geometry']['channels'] = True
            if change in ('identity', 'geometry'):
                bad['provenance_sha256'] = sha256(native._canonical_metadata(bad['provenance'])).hexdigest()
            with self.subTest(change=change), self.assertRaises(models.KMeansModelError):
                native.parse_kmeans_artifact(json.dumps(bad))

    def test_inconsistent_native_facts_reject_even_with_recomputed_checksum(self):
        model = models.model_from_centres(((0.0,),), 1, 1)
        mutations = (
            lambda p: p['request'].update(sampling=1., sampling_kind='fraction'),
            lambda p: p['request'].update(seed=True),
            lambda p: p['fit_parameters'].update(batch_size=2048),
            lambda p: p['fit_plan'].update(rows=5),
            lambda p: p['fit_result'].update(n_steps=101),
            lambda p: p['fit_result'].update(n_iter=11),
            lambda p: p['fit_result'].update(source_n_threads=1),
            lambda p: p['fit_result'].update(source_reduction_profile='recorded-openmp-threads-v1'),
            lambda p: p['fit_result']['inertia'].update(lo=1.),
            lambda p: p['fit_result']['random_identity'].update(seed=2),
            lambda p: p['fit_result']['random_identity'].update(state_sha256='bad'),
            lambda p: p['calculation']['fit_modules'].pop(native.FIT_MODULES[0]),
            lambda p: p['source_pins'].update(train_sha256='0'*64),
            lambda p: p['manifest']['occurrences'][0].update(source_bytes=124),
            lambda p: p['manifest']['selected_names'].append('a.png'),
            lambda p: p['manifest']['occurrences'][0].update(name={}),
        )
        for mutation in mutations:
            provenance = _native(model)
            mutation(provenance)
            with self.subTest(provenance=provenance), self.assertRaises(models.KMeansModelError):
                native.encode_kmeans_artifact(model, provenance)

    def test_duplicate_nonfinite_unknown_deep_and_oversized_metadata_rejects(self):
        raw = native.encode_kmeans_artifact(self.model, _conversion(self.model))
        for bad in (raw.replace(b'{"schema":', b'{"schema":"duplicate","schema":', 1),
                raw.replace(b'2.5', b'NaN'), b'['*1100+b'0'+b']'*1100,
                b'x'*(models.MAX_MODEL_BYTES+1), None):
            with self.subTest(kind=type(bad)), self.assertRaises(models.KMeansModelError):
                native.parse_kmeans_artifact(bad)
        provenance = _conversion(self.model)
        provenance['unknown'] = 1
        with self.assertRaises(models.KMeansModelError):
            native.encode_kmeans_artifact(self.model, provenance)
        value = 'leaf'
        for _ in range(native.MAX_METADATA_DEPTH+1):
            value = [value]
        for bad in (value, ['x']*(native.MAX_METADATA_ITEMS+1), 'x'*(native.MAX_METADATA_STRING+1), float('inf'), object()):
            with self.subTest(kind=type(bad)), self.assertRaises(models.KMeansModelError):
                native._canonical_metadata(bad)
        with patch.object(native, 'MAX_PROVENANCE_BYTES', 20), self.assertRaises(models.KMeansModelError):
            native.encode_kmeans_artifact(self.model, _conversion(self.model))
        with patch.object(models, 'MAX_MODEL_BYTES', len(raw)-1), self.assertRaises(models.KMeansModelError):
            native.encode_kmeans_artifact(self.model, _conversion(self.model))

    def test_recipe_selection_binds_entire_artifact_and_detects_provenance_edits(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'model.json'
            provenance = _conversion(self.model)
            path.write_bytes(native.encode_kmeans_artifact(self.model, provenance))
            selected = recipes.model_source_for_path(path)
            model = recipes.load_selected_model(selected)
            self.assertEqual(model.centres, self.model.centres)
            self.assertEqual(selected['source_sha256'], sha256(path.read_bytes()).hexdigest())
            provenance['environment']['joblib'] = 'changed'
            path.write_bytes(native.encode_kmeans_artifact(self.model, provenance))
            with self.assertRaisesRegex(ValueError, 'changed'):
                recipes.load_selected_model(selected)
            self.assertEqual(recipes.model_source_for_path(path)['centres_sha256'], selected['centres_sha256'])

    def test_load_bounds_actual_read(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'model.json'
            path.write_bytes(b'x'*100)
            with patch.object(models, 'MAX_MODEL_BYTES', 20), self.assertRaises(models.KMeansModelError):
                native.load_kmeans_artifact(path)
        with self.assertRaises(models.KMeansModelError):
            native.load_kmeans_artifact(path)
