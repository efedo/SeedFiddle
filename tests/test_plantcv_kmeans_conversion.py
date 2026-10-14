"""Offline source-model conversion and shared-codec import compatibility."""
from hashlib import sha256
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from seedvision.model_io import convert_plantcv_kmeans as conversion
from seedvision.model_io import kmeans_model_artifact as artifacts
from seedvision.model_io import plantcv_kmeans_model as models
from seedvision.cuda import plantcv_kmeans_predict as prediction
from seedvision.cuda import plantcv_kmeans as masks
from seedvision.cuda import plantcv_kmeans_classifier as scoring

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT/'artifacts/plantcv-reference-env/Scripts/python.exe'

PUBLIC_SCRIPT = r'''
import hashlib,importlib,importlib.metadata,json,sys
from pathlib import Path
import cv2,joblib,numpy as np,sklearn
train=importlib.import_module('plantcv.learn.train_kmeans')
classifier=importlib.import_module('plantcv.plantcv.kmeans_classifier')
from seedvision.model_io.convert_plantcv_kmeans import convert_plantcv_kmeans
assert importlib.metadata.version('plantcv')=='4.11.3'
assert np.__version__=='2.2.6'
assert sklearn.__version__=='1.9.1'
assert cv2.__version__=='4.11.0'
assert hashlib.sha256(Path(train.__file__).read_bytes()).hexdigest()=='dbfb8a528e92984a20e73c7f265a41efa9fa088b42ed4f65addbe6ab40ee8b4d'
assert hashlib.sha256(Path(classifier.__file__).read_bytes()).hexdigest()=='09019da00e0713c51cff31ffd240f7bf8aa0b4e228f6d88f2bed9c1d40430392'
sk=importlib.import_module('sklearn.cluster._kmeans')
assert hashlib.sha256(Path(sk.__file__).read_bytes()).hexdigest()=='0cf335b8b9b4abe67259ccdcbaf07f02bd7feee31e63a06a1cce32a87fcd375d'
root=Path(sys.argv[1]); results=[]
def digest(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
for index,(p,c,k) in enumerate(((1,1,2),(2,3,2),(3,1,1),(10,3,2),(31,3,1))):
    folder=root/f'images-{index}'; folder.mkdir()
    shape=(max(p+3,17),max(p+4,18))+(() if c==1 else (3,))
    for number,value in enumerate((0,180)):
        image=np.full(shape,value,np.uint8) if c==1 else np.broadcast_to(
            np.array((0,40,80) if number==0 else (140,180,220),np.uint8),shape).copy()
        assert cv2.imwrite(str(folder/f'{number:03}.png'),image)
    source=root/f'public-{index}.fit'; target=root/f'converted-{index}.json'
    fitted=train.train_kmeans(str(folder),k,out_path=str(source),patch_size=p,sampling=7,seed=31,n_init=2)
    loaded=joblib.load(source)
    assert type(loaded) is type(fitted)
    assert np.array_equal(loaded.cluster_centers_,fitted.cluster_centers_)
    assert not hasattr(loaded,'patch_size') and not hasattr(loaded,'channels')
    assert loaded.cluster_centers_.dtype==np.float64
    convert_plantcv_kmeans(source,target)
    heldout=np.empty((9,11)+(() if c==1 else (3,)),np.uint8)
    heldout[:,:5]=10 if c==1 else (10,50,90)
    heldout[:,5:]=170 if c==1 else (130,170,210)
    heldout+=np.random.default_rng(217+index).integers(0,7,heldout.shape,dtype=np.uint8)
    labels=classifier.predict_kmeans(heldout,str(source),patch_size=p)
    products=classifier.mask_kmeans(labels,k)
    results.append(dict(path=str(target),p=p,c=c,k=k,image=heldout.tolist(),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),source_bytes=source.stat().st_size,
        centres=digest(loaded.cluster_centers_),labels=digest(labels),classes_present=int(np.unique(labels).size),
        masks=[[key,digest(value)] for key,value in products.items()]))
# Exact binary64/signed-zero/order preservation is also checked independently of
# the public trainer's nonnegative image-derived centres, including a strided table.
from sklearn.cluster import MiniBatchKMeans
estimator=MiniBatchKMeans(n_clusters=3)
table=np.array([[-0.,99.],[2.125,98.],[-0.,97.]],dtype=np.float64)
estimator.cluster_centers_=table[:,::2]
estimator.n_features_in_=1
source=root/'signed.fit'; target=root/'signed.json'; joblib.dump(estimator,source)
convert_plantcv_kmeans(source,target)
signed_sha=digest(estimator.cluster_centers_)
estimator=MiniBatchKMeans(n_clusters=2,random_state=1,n_init=1).fit(np.array([[0.],[2.],[0.],[2.]]))
estimator.cluster_centers_=np.array([[.5],[1.5]],np.float64)
source=root/'ambiguous.fit'; ambiguous=root/'ambiguous.json'; joblib.dump(estimator,source)
convert_plantcv_kmeans(source,ambiguous)
source_labels=classifier.predict_kmeans(np.ones((6,7),np.uint8),str(source),patch_size=1)
assert np.all(source_labels==0)
print(json.dumps(dict(cases=results,signed_path=str(target),signed_sha=signed_sha,ambiguous_path=str(ambiguous))))
'''

INVALID_SCRIPT = r'''
import json,sys
from pathlib import Path
import joblib,numpy as np
from sklearn.cluster import MiniBatchKMeans,KMeans
from seedvision.model_io.convert_plantcv_kmeans import convert_plantcv_kmeans,KMeansConversionError
from seedvision.model_io.plantcv_kmeans_model import KMeansModelError
root=Path(sys.argv[1]); source=root/'invalid.fit'; target=root/'existing.json'
def model():
    m=MiniBatchKMeans(n_clusters=2); m.n_features_in_=1
    m.cluster_centers_=np.zeros((2,1),dtype=np.float64)
    return m
cases=[KMeans(n_clusters=2),MiniBatchKMeans(n_clusters=2)]
for attribute,value in (('n_clusters',True),('n_clusters',2.),('n_features_in_',1.),
        ('n_features_in_',True),('cluster_centers_',np.zeros((2,1),np.float32)),
        ('cluster_centers_',np.zeros(2,np.float64)),('cluster_centers_',np.zeros((1,1))),
        ('cluster_centers_',np.array([[float('nan')],[0.]])),
        ('cluster_centers_',np.array([[float('inf')],[0.]])),
        ('cluster_centers_',np.array([[-1.],[0.]])),('cluster_centers_',np.array([[256.],[0.]]))):
    m=model(); setattr(m,attribute,value); cases.append(m)
for k,width,value in ((257,1,0.),(2,4097,0.),(2,2,0.),(2,1024,0.),(256,2883,.12345678901234568)):
    m=model(); m.n_clusters=k; m.n_features_in_=width
    m.cluster_centers_=np.full((k,width),value,np.float64); cases.append(m)
for m in cases:
    joblib.dump(m,source); target.write_bytes(b'existing')
    try: convert_plantcv_kmeans(source,target)
    except (KMeansConversionError,KMeansModelError): pass
    else: raise AssertionError('unsupported model converted')
    assert target.read_bytes()==b'existing'
    assert not list(root.glob('.existing.json.*.tmp'))
source.write_bytes(b'not a fitted source model')
try: convert_plantcv_kmeans(source,target)
except KMeansConversionError: pass
else: raise AssertionError('invalid serialization converted')
assert target.read_bytes()==b'existing'
print(json.dumps(dict(rejected=len(cases)+1)))
'''


class KMeansConversionStructureTests(unittest.TestCase):
    def test_shared_codec_and_help_imports_never_load_desktop_or_ml_dependencies(self):
        script = r'''
import importlib.abc,runpy,sys
blocked=('torch','cv2','numpy','PySide6','joblib','sklearn','seedvision.cuda','seedvision.benchmarking')
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if any(fullname==name or fullname.startswith(name+'.') for name in blocked):
            raise AssertionError('unexpected dependency import: '+fullname)
sys.meta_path.insert(0,Guard())
from seedvision.model_io import plantcv_kmeans_model,kmeans_model_artifact
sys.argv=['converter','--help']
try: runpy.run_module('seedvision.model_io.convert_plantcv_kmeans',run_name='__main__')
except SystemExit as error: assert error.code==0
assert not any(any(key==name or key.startswith(name+'.') for name in blocked) for key in sys.modules)
'''
        result = subprocess.run([sys.executable, '-c', script], cwd=ROOT,
            capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('destination JSON model artifact', result.stdout)

    def test_old_import_paths_are_identical_modules_with_shared_patchable_globals(self):
        old_model = importlib.import_module('seedvision.cuda.plantcv_kmeans_model')
        old_artifact = importlib.import_module('seedvision.benchmarking.kmeans_model_artifact')
        self.assertIs(old_model, models)
        self.assertIs(old_artifact, artifacts)
        self.assertIs(importlib.import_module('seedvision.cuda').plantcv_kmeans_model, models)
        self.assertIs(importlib.import_module('seedvision.benchmarking').kmeans_model_artifact, artifacts)
        self.assertIs(old_model.KMeansModel, models.KMeansModel)
        with patch.object(old_model, 'MAX_MODEL_BYTES', 1), self.assertRaises(models.KMeansModelError):
            models.parse_kmeans_model(b'{}')
        self.assertEqual(models.MAX_MODEL_BYTES, 8*1024*1024)

    def test_historical_producer_labels_remain_stable_and_real_codec_source_is_verified(self):
        from seedvision.benchmarking.native_recipe import _module_record
        from seedvision.benchmarking.native_threshold_stage import NativeThresholdStageError
        self.assertIn('seedvision.benchmarking.kmeans_model_artifact', artifacts.FIT_MODULES)
        self.assertNotIn(artifacts.__name__, artifacts.FIT_MODULES)
        self.assertEqual(Path(artifacts.__file__).parent, ROOT/'seedvision/model_io')
        _module_record(artifacts)
        _module_record(models)
        with patch.object(models, 'MAX_FEATURES', 1), self.assertRaises(NativeThresholdStageError):
            _module_record(models)
        with patch.object(artifacts, 'MAX_METADATA_DEPTH', 1), self.assertRaises(NativeThresholdStageError):
            _module_record(artifacts)

    def test_geometry_bounds_exact_integer_shapes_and_unsupported_widths(self):
        for p in (1, 2, 10, 31):
            for c in (1, 3):
                self.assertEqual(conversion._geometry(p*p*c), (p, c))
        for width in (True, 1., 0, -1, 2, 1024, 4097):
            with self.subTest(width=width), self.assertRaises(conversion.KMeansConversionError):
                conversion._geometry(width)

    def test_atomic_replace_failure_and_output_cap_preserve_existing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'model.json'
            path.write_bytes(b'existing')
            with patch.object(conversion.os, 'replace', side_effect=OSError('replace failed')):
                with self.assertRaises(conversion.KMeansConversionError):
                    conversion._atomic_write(path, b'new')
            self.assertEqual(path.read_bytes(), b'existing')
            self.assertEqual(list(Path(directory).glob('.*.tmp')), [])
            with patch.object(models, 'MAX_MODEL_BYTES', 1), self.assertRaises(conversion.KMeansConversionError):
                conversion._atomic_write(path, b'new')
            self.assertEqual(path.read_bytes(), b'existing')
            conversion._atomic_write(path, b'new')
            self.assertEqual(path.read_bytes(), b'new')

    def test_input_cap_and_same_file_aliases_reject_before_loading_model(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)/'model.fit'
            source.write_bytes(b'not a source estimator')
            with patch.object(conversion, 'MAX_INPUT_BYTES', 1), self.assertRaises(conversion.KMeansConversionError):
                conversion._read_bounded(source)
            with self.assertRaisesRegex(conversion.KMeansConversionError, 'must differ'):
                conversion.convert_plantcv_kmeans(source, source)
            alias = Path(directory)/'same.fit'
            alias.hardlink_to(source)
            with self.assertRaisesRegex(conversion.KMeansConversionError, 'must differ'):
                conversion.convert_plantcv_kmeans(source, alias)


@unittest.skipUnless(REFERENCE.exists(), 'pinned source training environment required')
class KMeansPublicConversionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(dir=ROOT/'artifacts')
        cls.addClassCleanup(cls.temp.cleanup)
        result = subprocess.run([str(REFERENCE), '-c', PUBLIC_SCRIPT, cls.temp.name],
            cwd=ROOT, capture_output=True, text=True, timeout=120)
        if result.returncode:
            raise AssertionError(result.stderr)
        cls.oracle = json.loads(result.stdout)

    def test_actual_public_training_models_preserve_exact_centres_and_conversion_provenance(self):
        for case in self.oracle['cases']:
            with self.subTest(p=case['p'], c=case['c']):
                artifact = artifacts.load_kmeans_artifact(case['path'])
                model = artifact.model
                self.assertEqual((model.patch_size, model.channels, len(model.centres)),
                    (case['p'], case['c'], case['k']))
                self.assertEqual(sha256(np.asarray(model.centres, dtype='<f8').tobytes()).hexdigest(), case['centres'])
                provenance = artifact.provenance
                self.assertEqual(provenance['input']['source_sha256'], case['source_sha256'])
                self.assertEqual(provenance['input']['source_bytes'], case['source_bytes'])
                self.assertEqual(provenance['environment'],
                    dict(plantcv='4.11.3', numpy='2.2.6', sklearn='1.9.1', joblib='1.6.0'))
                self.assertEqual(provenance['numeric_identity'], artifacts.numeric_model_identity(model))
                self.assertEqual(model.source_sha256, sha256(Path(case['path']).read_bytes()).hexdigest())
                if case['k'] == 2:
                    self.assertEqual(case['classes_present'], 2)
        signed = artifacts.load_kmeans_artifact(self.oracle['signed_path']).model
        self.assertEqual(sha256(np.asarray(signed.centres, dtype='<f8').tobytes()).hexdigest(), self.oracle['signed_sha'])

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_converted_models_match_public_heldout_labels_and_ordered_masks(self):
        for case in self.oracle['cases']:
            with self.subTest(p=case['p'], c=case['c']):
                model = artifacts.load_kmeans_artifact(case['path']).model
                image = torch.tensor(case['image'], dtype=torch.uint8, device='cuda')
                labels = prediction.predict_kmeans(image, model)
                self.assertEqual(sha256(labels.cpu().numpy().tobytes()).hexdigest(), case['labels'])
                products = masks.mask_kmeans(labels, case['k'])
                self.assertEqual([[key, sha256(value.cpu().numpy().tobytes()).hexdigest()]
                    for key, value in products.items()], case['masks'])

    def test_unfitted_wrong_type_malformed_and_oversized_models_leave_output_untouched(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as directory:
            result = subprocess.run([str(REFERENCE), '-c', INVALID_SCRIPT, directory], cwd=ROOT,
                capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertGreaterEqual(json.loads(result.stdout)['rejected'], 19)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_source_fractional_tie_remains_withheld_after_conversion(self):
        model = artifacts.load_kmeans_artifact(self.oracle['ambiguous_path']).model
        image = torch.ones((6, 7), dtype=torch.uint8, device='cuda')
        with self.assertRaises(scoring.KMeansDecisionAmbiguous):
            prediction.predict_kmeans(image, model)

    def test_actual_source_environment_cli_writes_a_loadable_artifact(self):
        source = Path(self.oracle['signed_path']).with_suffix('.fit')
        target = source.with_name('cli.json')
        result = subprocess.run([str(REFERENCE), '-m', 'seedvision.model_io.convert_plantcv_kmeans',
            str(source), str(target)], cwd=ROOT, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        model = artifacts.load_kmeans_artifact(target).model
        self.assertEqual(sha256(np.asarray(model.centres, dtype='<f8').tobytes()).hexdigest(), self.oracle['signed_sha'])
