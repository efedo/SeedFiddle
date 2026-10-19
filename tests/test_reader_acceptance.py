"""Frozen public/native reader pixels, outcomes and preparation-cache evidence.

This engineering corpus is separate from recipe-output and scientific gates.
The expected record is refreshed explicitly; tests never rewrite its baseline.
"""

from tests.catalogue_fixtures import CatalogueTestCase
from hashlib import sha256
from importlib.metadata import version
import inspect
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import cv2
import numpy as np
import torch

from seedvision.benchmarking import native_image_inputs as inputs
from seedvision.benchmarking import native_recipe as runner
from seedvision.segmentation.baseline import PipelineAnalysisCache
from tests.test_native_image_inputs import fixtures, REFERENCE, READER_SHA
from tests.acceptance_diff import assert_json_equal


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = ROOT / 'artifacts/reader-acceptance-baseline-v1.json'
EXPECTED_SHA256 = '3f06d1f4a9ed2ba4568ad3c14797bbb72c81ca9a422be294df98ba0503c4cd68'
# The unchanged reader matrix predates these loaded-code identities. A complete
# current reader capture found only these two provenance differences; root
# independently recomputed both against the V3 frozen source closure. The
# immutable transition receipt is complete-software-reader-provenance-transition-v1.
# No pixels, outcomes, cache facts, environment pins or other fields migrate.
REVIEWED_PREPARATION_IDENTITIES = (
    (1, '56855e000ef3c85e47158e3bb523927df47b198d91ee02d975dc85107c3bc7f7',
        'eff2c85ff0275b162287753b9045eb595928518a9f055a4564e58e00dd3c3851'),
    (4, '629f1b436abf9a9a3a981c53b0512c627357c78301a0ebbdf5a3702859b14d55',
        '7d61d4aa056930967984a785e36323d9b4a048f2299930057f507f27ec557d58'),
)
WORKSPACE_TRANSITION = ROOT / 'artifacts/pipeline-workspace-reader-provenance-v1.json'
WORKSPACE_TRANSITION_SHA256 = '60354886d03d57b15a6c22540e1dee43c8edb3492e57daca9e6b9e276bb43d37'
PREVIOUS_TRANSITION = ROOT / 'artifacts/complete-software-reader-provenance-transition-v1.json'
PREVIOUS_TRANSITION_SHA256 = '8ab40a88449508459fe168c503ab6e60c27690c2c19d96f23a8235ad7b4b50c6'
CASE_NAMES = (
    'color.png', 'color.jpg', 'color.bmp', 'color.tif', 'gray.png', 'gray.tif',
    'palette.png', 'palette.tif', 'transparent-palette.png', 'alpha.png',
    'depth16.png', 'depth16.tif', 'exif-2.jpg', 'exif-3.jpg', 'exif-4.jpg',
    'exif-5.jpg', 'exif-6.jpg', 'exif-7.jpg', 'exif-8.jpg', 'cmyk.jpg',
    'pages.tif', 'png-content.jpg', 'jpeg-content.png', 'seed-色.png',
    'empty.png', 'corrupt.jpg', 'unsupported.png', 'missing.png',
)
MODES = ('color-derived', 'codec-gray')


def _digest_json(value):
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                             separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def _array(value):
    return {'shape': list(value.shape), 'dtype': str(value.dtype),
            'pixels_sha256': sha256(np.ascontiguousarray(value).tobytes()).hexdigest()}


def _frozen_record():
    if not EXPECTED.is_file():
        raise AssertionError('Frozen reader evidence is required at artifacts/reader-acceptance-baseline-v1.json; restore the reviewed baseline before this acceptance gate.')
    raw = EXPECTED.read_bytes()
    if sha256(raw).hexdigest() != EXPECTED_SHA256:
        raise AssertionError('The frozen reader acceptance baseline bytes changed.')
    return json.loads(raw)


def _reviewed_expected():
    """Apply reviewed code identities while preserving the complete reader matrix."""
    expected = _frozen_record()
    identity = expected['pins']['native_preparation_identity']
    for index, previous, current in REVIEWED_PREPARATION_IDENTITIES:
        if identity[index] != previous:
            raise AssertionError('The previous reader code transition no longer applies.')
        identity[index] = current
    raw = WORKSPACE_TRANSITION.read_bytes()
    if sha256(raw).hexdigest() != WORKSPACE_TRANSITION_SHA256:
        raise AssertionError('The reviewed workspace reader transition bytes changed.')
    record = json.loads(raw)
    if (record['schema'] != 'seedfiddle.pipeline-workspace-reader-code-provenance-v1'
            or record['baseline'] != EXPECTED.relative_to(ROOT).as_posix()
            or record['baseline_sha256'] != EXPECTED_SHA256
            or record['previous_transition'] != PREVIOUS_TRANSITION.relative_to(ROOT).as_posix()
            or record['previous_transition_sha256'] != PREVIOUS_TRANSITION_SHA256
            or sha256(PREVIOUS_TRANSITION.read_bytes()).hexdigest() != PREVIOUS_TRANSITION_SHA256):
        raise AssertionError('The workspace reader transition history changed.')
    source = 'seedvision/benchmarking/native_recipe.py'
    from tests.acceptance_provenance import verify_previous_source, apply
    if set(record['source_files']) != {source}:
        raise AssertionError('The reviewed reader implementation inventory changed.')
    verify_previous_source(source, record['source_files'][source])
    transition = {
        'path': '/pins/native_preparation_identity/1',
        'old': 'eff2c85ff0275b162287753b9045eb595928518a9f055a4564e58e00dd3c3851',
        'new': '4d1cfc6307f49ad55f7b4bdd9d63154ed49de52ddb55081a1ae413b115803624',
    }
    if (record['transitions'] != [transition]
            or identity != record['identity_before'] or identity[1] != transition['old']):
        raise AssertionError('Only the reviewed preparation-code identity may migrate.')
    identity[1] = transition['new']
    if identity != record['identity_after']:
        raise AssertionError('The reviewed reader identity closure changed.')
    # The consolidated matrix also checks the public reader's returned name.
    # Only this assertion-only oracle edit migrates; all recorded facts stay fixed.
    old_oracle = '3747e5c2f1ec5b256ab747e11a6e3beb637fdf3876d4ef9263fe2bef23fdd395'
    new_oracle = 'd4406e598a6494243776a3ba294c02f9c62cb07dd0e43035049863f9d849778b'
    if (expected['pins']['public_oracle_sha256'] != old_oracle
            or sha256(ORACLE.encode('utf-8')).hexdigest() != new_oracle):
        raise AssertionError('The reviewed filename-assertion oracle transition changed.')
    expected['pins']['public_oracle_sha256'] = new_oracle
    return apply(expected, 'reader')


def _inventory(paths):
    if tuple(path.name for path in paths) != CASE_NAMES:
        raise AssertionError('The reader corpus inventory changed.')
    return [{'case_id': path.name, 'suffix': path.suffix,
             'path_class': 'unicode' if path.name == 'seed-色.png' else 'ascii',
             'exists': path.is_file(),
             'source_bytes': path.stat().st_size if path.is_file() else None,
             'source_sha256': sha256(path.read_bytes()).hexdigest() if path.is_file() else None}
            for path in paths]


ORACLE = r'''
from hashlib import sha256
from importlib.metadata import version
import importlib,json,sys
from pathlib import Path
import cv2,numpy as np
from plantcv import plantcv as pcv
assert version('plantcv')=='4.11.3'
assert version('opencv-python')=='4.11.0.86'
assert version('numpy')=='2.2.6'
reader=importlib.import_module('plantcv.plantcv.readimage')
reader_sha=sha256(Path(reader.__file__).read_bytes()).hexdigest()
assert reader_sha==sys.argv[3]
pcv.params.debug=None;pcv.params.verbose=False
arrays={};rows=[]
for i,name in enumerate(json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))):
    row={}
    for mode in ('rgb','gray'):
        try:
            raster,_,source_name=pcv.readimage(filename=name,mode=mode)
            assert source_name==Path(name).name
            assert raster.dtype==np.uint8
            arrays[f'{mode}_{i}']=raster
            row[mode]={'status':'success','shape':list(raster.shape),'dtype':str(raster.dtype),
                       'pixels_sha256':sha256(np.ascontiguousarray(raster).tobytes()).hexdigest()}
        except Exception as error:
            row[mode]={'status':'failure','phase':'public.readimage','error_type':type(error).__name__}
    rows.append(row)
np.savez(sys.argv[2]+'.npz',**arrays)
pins={'plantcv':version('plantcv'),'opencv-python':version('opencv-python'),
      'opencv':cv2.__version__,'numpy':np.__version__,'flags':{'rgb':1,'gray':0},
      'reader_sha256':reader_sha,
      'opencv_build_sha256':sha256(cv2.getBuildInformation().encode('utf-8')).hexdigest()}
Path(sys.argv[2]+'.json').write_text(json.dumps({'pins':pins,'cases':rows}),encoding='utf-8')
'''


def capture_reader_acceptance():
    """Execute the finite corpus and return compact, path-independent evidence."""
    if not REFERENCE.is_file():
        raise AssertionError('The pinned public reader interpreter is required.')
    if not torch.cuda.is_available():
        raise AssertionError('CUDA is required for native reader acceptance.')
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        paths = fixtures(directory)
        inventory = _inventory(paths)
        request = directory / 'paths.json'
        request.write_text(json.dumps([str(path) for path in paths], ensure_ascii=False), encoding='utf-8')
        output = directory / 'public'
        process = subprocess.run([str(REFERENCE), '-c', ORACLE, str(request), str(output), READER_SHA],
                                 capture_output=True, text=True, timeout=120)
        if process.returncode:
            raise AssertionError(process.stderr)
        public = json.loads(output.with_suffix('.json').read_text(encoding='utf-8'))
        if len(public['cases']) != len(paths):
            raise AssertionError('The public reader returned an incomplete corpus.')
        cases = []
        with np.load(output.with_suffix('.npz'), allow_pickle=False) as rasters:
            for index, (path, source, reference) in enumerate(zip(paths, inventory, public['cases'], strict=True)):
                row = {'source': source, 'public': reference, 'native': {}}
                equivalent = 0 if path.name == 'seed-色.png' else index
                cache = PipelineAnalysisCache()
                prior_rgb = None
                prior_rgb_pixels = None
                selected_grays = {}
                for mode in MODES:
                    try:
                        prepared = runner.prepare_native_recipe_inputs(path, cache=cache, gray_source_mode=mode)
                    except (runner.NativeRecipeError, OSError) as error:
                        if any(value['status'] == 'success' for value in reference.values()):
                            raise AssertionError(f'Native preparation rejected public success {path.name}') from error
                        row['native'][mode] = {'status': 'rejected', 'phase': 'native.prepare',
                                               'error_type': type(error).__name__}
                        continue
                    if path.name != 'seed-色.png' and any(value['status'] != 'success' for value in reference.values()):
                        raise AssertionError(f'Unexpected native-only reader success {path.name}')
                    if prepared.source_sha256 != source['source_sha256']:
                        raise AssertionError('Native preparation used different source bytes.')
                    bgr = rasters[f'rgb_{equivalent}']
                    reference_gray = (rasters[f'gray_{equivalent}'] if mode == 'codec-gray'
                                      else cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
                    gray = prepared.gray.cpu().numpy()
                    rgb = prepared.rgb.cpu().numpy()
                    np.testing.assert_array_equal(gray, reference_gray)
                    np.testing.assert_array_equal(rgb, bgr[..., ::-1])
                    if prepared.gray.dtype != torch.uint8 or prepared.rgb.dtype != torch.uint8:
                        raise AssertionError('Native reader output dtype changed.')
                    warm = runner.prepare_native_recipe_inputs(path, cache=cache, gray_source_mode=mode)
                    same_mode_reused = warm is prepared
                    mode_rgb_reused = None if prior_rgb is None else prepared.rgb is prior_rgb
                    mode_rgb_preserved = (None if prior_rgb_pixels is None else
                                          bool(np.array_equal(rgb, prior_rgb_pixels)))
                    if not same_mode_reused or mode_rgb_reused is False:
                        raise AssertionError('Native preparation failed its same-source cache contract.')
                    np.testing.assert_array_equal(warm.gray.cpu().numpy(), gray)
                    np.testing.assert_array_equal(warm.rgb.cpu().numpy(), rgb)
                    if mode_rgb_preserved is False:
                        raise AssertionError('Switching gray modes changed cached RGB pixels.')
                    row['native'][mode] = {
                        'status': 'success', 'source_sha256': prepared.source_sha256,
                        'frame': prepared.coordinate_frame.to_dict(),
                        'gray': _array(gray), 'rgb': _array(rgb),
                        'reference_gray': _array(reference_gray),
                        'reference_rgb': _array(bgr[..., ::-1]),
                        'gray_reference_method': 'public.readimage(gray)' if mode == 'codec-gray'
                                                 else 'OpenCV BGR2GRAY of public.readimage(rgb)',
                        'comparison': 'native_path_extension' if equivalent != index else 'exact_pixels',
                        'equivalent_ascii_case': paths[equivalent].name if equivalent != index else None,
                        'gray_mismatches': 0, 'rgb_mismatches': 0,
                        'same_mode_preparation_reused': same_mode_reused,
                        'same_mode_output_pixels_preserved': True,
                        'mode_switch_rgb_reused': mode_rgb_reused,
                        'mode_switch_rgb_pixels_preserved': mode_rgb_preserved,
                    }
                    selected_grays[mode] = np.array(gray, copy=True)
                    prior_rgb = prepared.rgb
                    prior_rgb_pixels = np.array(rgb, copy=True)
                row['gray_modes_different_pixels'] = (int(np.count_nonzero(
                    selected_grays[MODES[0]] != selected_grays[MODES[1]]))
                    if len(selected_grays) == 2 else None)
                cache.values.clear()
                cases.append(row)
        return {
            'schema': 'seedfiddle.reader-acceptance.v1',
            'scope': {'fixtures': len(paths), 'native_modes': list(MODES), 'public_modes': ['rgb', 'gray'],
                      'formats': ['png', 'jpeg', 'bmp', 'tiff'],
                      'claim': 'Exact recorded pixels and outcomes for these fixtures and pinned readers; Unicode loading is a native extension.',
                      'native_environment': 'Acceptance is limited to the measured Torch/CUDA/device environment in this record.',
                      'recipe_output_freeze': False, 'scientific_validation': False,
                      'other_reader_modes': 'separate open contracts'},
            'pins': {'public': public['pins'], 'native': inputs.versions(), 'pillow': version('pillow'),
                     'fixture_builder_sha256': sha256(inspect.getsource(fixtures).encode('utf-8')).hexdigest(),
                     'public_oracle_sha256': sha256(ORACLE.encode('utf-8')).hexdigest(),
                     'native_preparation_identity': list(runner._preparation_identity()),
                     'native_input_helper_sha256': sha256(Path(inputs.__file__).read_bytes()).hexdigest()},
            'measured_native_environment': {'torch': str(torch.__version__), 'torch_cuda': torch.version.cuda,
                                            'device_name': torch.cuda.get_device_name()},
            'inventory_sha256': _digest_json(inventory), 'cases': cases,
        }


class ReaderAcceptanceTests(CatalogueTestCase):
    def test_frozen_inventory_and_exact_encoded_fixture_bytes(self):
        expected = _frozen_record()
        _reviewed_expected()
        self.assertEqual(expected['schema'], 'seedfiddle.reader-acceptance.v1')
        with tempfile.TemporaryDirectory() as temporary:
            paths = fixtures(Path(temporary))
            inventory = _inventory(paths)
            cases = {case['source']['case_id']: case for case in expected['cases']}
            self.assertEqual(sum(case['public']['rgb']['status'] == 'success'
                                 for case in cases.values()), 23)
            for path in paths:
                with self.subTest(cpu_decode=path.name):
                    case = cases[path.name]
                    public = case['public']['rgb']
                    if path.name == 'seed-色.png':
                        self.assertEqual(public['status'], 'failure')
                        self.assertTrue(public['error_type'])
                        public = cases['color.png']['public']['rgb']
                    elif public['status'] != 'success':
                        with self.assertRaises((inputs.ImageInputError, OSError)):
                            identity = inputs.source_identity(path)
                            inputs.decode(inputs.snapshot(path, identity))
                        continue
                    identity = inputs.source_identity(path)
                    self.assertEqual(identity.sha256, case['source']['source_sha256'])
                    bgr = inputs.decode(inputs.snapshot(path, identity))
                    self.assertEqual(_array(bgr),
                                     {key: public[key] for key in ('shape', 'dtype', 'pixels_sha256')})
        self.assertEqual([row['source'] for row in expected['cases']], inventory)
        self.assertEqual(expected['inventory_sha256'], _digest_json(inventory))
        self.assertEqual(expected['pins']['pillow'], version('pillow'))
        self.assertEqual(expected['pins']['fixture_builder_sha256'],
                         sha256(inspect.getsource(fixtures).encode('utf-8')).hexdigest())
        self.assertEqual(expected['scope']['native_modes'], list(MODES))
        self.assertEqual(len(expected['cases']), 28)

    def test_actual_public_native_pixels_outcomes_and_cache_match_frozen_matrix(self):
        expected = _reviewed_expected()
        actual = capture_reader_acceptance()
        assert_json_equal(self, actual, expected)
        public_successes = [case for case in actual['cases']
                            if case['public']['rgb']['status'] == 'success']
        self.assertEqual(len(public_successes), 23)
        successful = [case for case in actual['cases'] if case['gray_modes_different_pixels'] is not None]
        self.assertTrue(any(case['gray_modes_different_pixels'] > 0 for case in successful),
                        'A codec fixture must prove the grayscale control affects pixels.')
        unicode = next(case for case in actual['cases'] if case['source']['path_class'] == 'unicode')
        self.assertEqual({value['status'] for value in unicode['public'].values()}, {'failure'})
        self.assertTrue(all(value['error_type'] for value in unicode['public'].values()))
        self.assertEqual({value['comparison'] for value in unicode['native'].values()}, {'native_path_extension'})
        for name in ('empty.png', 'corrupt.jpg', 'unsupported.png', 'missing.png'):
            case = next(case for case in actual['cases'] if case['source']['case_id'] == name)
            self.assertEqual({value['status'] for value in case['public'].values()}, {'failure'})
            self.assertEqual({value['status'] for value in case['native'].values()}, {'rejected'})


if __name__ == '__main__':
    unittest.main()
