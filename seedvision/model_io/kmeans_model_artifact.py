"""Bounded single-file K-means models with compact, persistent provenance.

The centres-only codec remains unchanged. A model returned here binds its source
identity to the entire selected file; its numeric digest still covers only the
ordered centres and prediction contract. This package uses only the standard
library and can also run in the source model's separate training environment.
Provenance describes a recorded calculation, not biological model validation.
"""
from dataclasses import dataclass, replace
from hashlib import sha256
import json
import math
from pathlib import Path
import re

from seedvision.model_io import plantcv_kmeans_model as _model

SCHEMA = 'seedfiddle.plantcv-kmeans-artifact.v1'
MAX_PROVENANCE_BYTES = 512 * 1024
MAX_METADATA_ITEMS = 50_000
MAX_METADATA_DEPTH = 16
MAX_METADATA_STRING = 32_768
FIT_MODULES = (
    'seedvision.cuda.plantcv_kmeans_fit', 'seedvision.cuda.plantcv_kmeans_control',
    'seedvision.cuda.plantcv_kmeans_intervals', 'seedvision.cuda.plantcv_kmeans_minibatch',
    'seedvision.cuda.plantcv_kmeans_plusplus', 'seedvision.cuda.plantcv_kmeans_random',
    'seedvision.cuda.plantcv_kmeans_tolerance', 'seedvision.cuda.plantcv_kmeans_training_score',
    'seedvision.cuda.plantcv_kmeans_export', 'seedvision.benchmarking.kmeans_model_artifact',
    'seedvision.benchmarking.native_kmeans_training',
)


@dataclass(frozen=True, slots=True)
class KMeansArtifact:
    model: _model.KMeansModel
    provenance_json: bytes | None
    provenance_sha256: str | None

    @property
    def provenance(self):
        """Return an independent metadata copy, never mutable shared state."""
        return None if self.provenance_json is None else json.loads(self.provenance_json)


def _error(message):
    raise _model.KMeansModelError('invalid K-means provenance: '+message)


def _keys(value, fields):
    if type(value) is not dict or set(value) != set(fields):
        _error('unexpected or missing fields')


def _int(value, low, high):
    if type(value) is not int or not low <= value <= high:
        _error('integer outside its supported bound')


def _sha(value):
    if type(value) is not str or re.fullmatch('[0-9a-f]{64}', value) is None:
        _error('expected lowercase SHA-256')


def _text(value):
    if type(value) is not str or not value:
        _error('expected nonempty text')


def _canonical_metadata(value):
    pending, items = [(value, 0)], 0
    while pending:
        current, depth = pending.pop()
        items += 1
        if items > MAX_METADATA_ITEMS or depth > MAX_METADATA_DEPTH:
            _error('metadata item/depth bound exceeded')
        if type(current) is dict:
            if items+len(pending)+len(current) > MAX_METADATA_ITEMS:
                _error('metadata item bound exceeded')
            if any(type(key) is not str or len(key) > MAX_METADATA_STRING for key in current):
                _error('invalid metadata key')
            pending.extend((child, depth+1) for child in current.values())
        elif type(current) in (list, tuple):
            if items+len(pending)+len(current) > MAX_METADATA_ITEMS:
                _error('metadata item bound exceeded')
            pending.extend((child, depth+1) for child in current)
        elif type(current) is str:
            if len(current) > MAX_METADATA_STRING:
                _error('metadata text bound exceeded')
        elif type(current) is float:
            if not math.isfinite(current):
                _error('nonfinite metadata')
        elif current is not None and type(current) not in (int, bool):
            _error('metadata must contain only JSON primitives')
    encoded = bytearray()
    try:
        for part in json.JSONEncoder(ensure_ascii=False, sort_keys=True,
                separators=(',', ':'), allow_nan=False).iterencode(value):
            chunk = part.encode('utf-8')
            if len(chunk) > MAX_PROVENANCE_BYTES-len(encoded):
                _error('metadata byte bound exceeded')
            encoded.extend(chunk)
    except (ValueError, UnicodeError, OverflowError) as error:
        raise _model.KMeansModelError('invalid K-means metadata encoding') from error
    return bytes(encoded)


def numeric_model_identity(model):
    raw = _model.encode_kmeans_model(model)
    return dict(source_sha256=sha256(raw).hexdigest(), source_bytes=len(raw),
        centres_sha256=model.centres_sha256)


def _manifest(value, request, model):
    _keys(value, ('directory', 'listing', 'selected_names', 'prefix', 'num_imgs',
        'occurrences', 'source_bytes', 'decoded_bytes', 'manifest_sha256'))
    if any(value[key] != request[key] for key in ('directory', 'prefix', 'num_imgs')):
        _error('request differs from the selected directory')
    names, selected, occurrences = value['listing'], value['selected_names'], value['occurrences']
    if (type(names) is not list or not 1 <= len(names) <= 4096
            or any(type(name) is not str or not name or '/' in name or '\\' in name
                or name in ('.', '..') for name in names)
            or names != sorted(set(names)) or type(selected) is not list
            or not 1 <= len(selected) <= 256
            or any(type(name) is not str or name not in names for name in selected)
            or type(occurrences) is not list
            or len(occurrences) != len(selected)
            or (request['num_imgs'] == 0 and selected != names)
            or (request['num_imgs'] != 0 and len(selected) != request['num_imgs'])):
        _error('invalid directory listing/selection')
    unique, rows = {}, 0
    for index, item in enumerate(occurrences):
        _keys(item, ('index', 'name', 'status', 'source_sha256', 'source_bytes', 'shape'))
        _int(item['index'], index, index)
        if type(item['name']) is not str or item['name'] != selected[index] or item['name'] not in names:
            _error('occurrence order differs from selection')
        status = item['status']
        if type(status) is not str or status not in ('prefix_skipped', 'non_file', 'decode_skipped', 'accepted'):
            _error('invalid occurrence status')
        if (request['prefix'] not in item['name']) != (status == 'prefix_skipped'):
            _error('occurrence prefix differs from request')
        _int(item['source_bytes'], 0, 64*1024*1024)
        if status in ('prefix_skipped', 'non_file'):
            if item['source_sha256'] is not None or item['source_bytes'] != 0:
                _error('skipped occurrence has file identity')
        else:
            _sha(item['source_sha256'])
        shape = item['shape']
        if status == 'accepted':
            _int(item['source_bytes'], 1, 64*1024*1024)
            if type(shape) is not list or len(shape) not in (2, 3):
                _error('invalid accepted image shape')
            for size in shape:
                _int(size, 1, 16_000_000)
            channels = 1 if len(shape) == 2 else shape[2]
            if channels != model.channels or shape[0]*shape[1] > 16_000_000:
                _error('accepted shape differs from model domain')
            p = model.patch_size
            if min(shape[:2]) < p:
                _error('patch does not fit the training image')
            total = (shape[0]-p+1)*(shape[1]-p+1)
            sample = request['sampling']
            count = total if sample is None else (min(sample, total)
                if type(sample) is int else int(sample*total))
            _int(count, 1, 1_000_000)
            rows += count
        elif shape is not None:
            _error('skipped occurrence has decoded geometry')
        if item['name'] in unique and item != dict(unique[item['name']], index=index):
            _error('duplicate occurrences have inconsistent file facts')
        unique[item['name']] = item
    if occurrences[0]['status'] != 'accepted':
        _error('source accumulator requires an accepted first occurrence')
    _int(value['source_bytes'], 0, 128*1024*1024)
    _int(value['decoded_bytes'], 1, 128*1024*1024)
    if (value['source_bytes'] != sum(item['source_bytes'] for item in unique.values())
            or value['decoded_bytes'] != sum(4*item['shape'][0]*item['shape'][1]
                for item in unique.values() if item['status'] == 'accepted')):
        _error('invalid aggregate directory bytes')
    fields = {key: child for key, child in value.items() if key != 'manifest_sha256'}
    if value['manifest_sha256'] != sha256(_canonical_metadata(fields)).hexdigest():
        _error('directory manifest digest differs')
    return rows


def _native_provenance(value, model):
    _keys(value, ('kind', 'numeric_identity', 'request', 'manifest', 'patch_seed',
        'fit_parameters', 'fit_plan', 'fit_result', 'calculation', 'source_pins'))
    request = value['request']
    _keys(request, ('directory', 'prefix', 'num_imgs', 'patch_size', 'sampling',
        'sampling_kind', 'clusters', 'seed', 'n_init'))
    _text(request['directory'])
    if type(request['prefix']) is not str:
        _error('prefix must be text')
    _int(request['num_imgs'], 0, 256)
    _int(request['seed'], 0, 0xFFFFFFFF)
    _int(request['n_init'], 1, 32)
    _int(request['patch_size'], model.patch_size, model.patch_size)
    _int(request['clusters'], len(model.centres), len(model.centres))
    sample = request['sampling']
    kind = 'all' if sample is None else ('count' if type(sample) is int else 'fraction')
    if request['sampling_kind'] != kind:
        _error('sampling type identity differs')
    if kind == 'count':
        _int(sample, 1, 1_000_000)
    elif kind == 'fraction' and (type(sample) is not float or not 0 < sample < 1):
        _error('fraction must be strictly between zero and one')
    _int(value['patch_seed'], 1, 1)
    rows = _manifest(value['manifest'], request, model)
    parameters = value['fit_parameters']
    expected = dict(init='k-means++', source_n_threads=None, seed=request['seed'],
        max_iter=100, batch_size=1024, n_init=request['n_init'], init_size=None,
        tol=0.0, max_no_improvement=10, reassignment_ratio=.01, compute_labels=True)
    _keys(parameters, expected)
    if any(type(parameters[key]) is not type(child) or parameters[key] != child
            for key, child in expected.items()):
        _error('unsupported effective native fit parameters')
    plan = value['fit_plan']
    _keys(plan, ('rows', 'features', 'clusters', 'batch_size', 'init_size', 'n_init',
        'n_steps', 'adjusted_init_size', 'work', 'peak_live_bytes', 'cpu_metadata_bytes'))
    expected_plan = dict(rows=rows, features=model.patch_size**2*model.channels,
        clusters=len(model.centres), batch_size=min(1024, rows),
        init_size=min(max(3*min(1024, rows), len(model.centres)), rows),
        n_init=request['n_init'], n_steps=100*rows//min(1024, rows))
    for key, child in expected_plan.items():
        _int(plan[key], child, child)
    _int(rows, len(model.centres), 1_000_000)
    if 2*rows*plan['features'] > 128*1024*1024 or plan['adjusted_init_size'] is not False:
        _error('invalid fit dataset/initialization plan')
    _int(plan['n_steps'], 1, 1_000_000)
    _int(plan['work'], 1, 20_000_000_000)
    _int(plan['peak_live_bytes'], 1, 512*1024*1024)
    _int(plan['cpu_metadata_bytes'], 1, 128*1024*1024)
    fit = value['fit_result']
    _keys(fit, ('n_steps', 'n_iter', 'inertia', 'scaled_tolerance',
        'source_n_threads', 'source_reduction_profile', 'random_identity'))
    _int(fit['n_steps'], 1, plan['n_steps'])
    iterations = (fit['n_steps']*plan['batch_size']+rows-1)//rows
    _int(fit['n_iter'], iterations, iterations)
    if (fit['source_n_threads'] is not None
            or fit['source_reduction_profile'] != 'all-valid-openmp-reductions-v1'):
        _error('native fit requires the explicit all-reduction profile')
    for key in ('inertia', 'scaled_tolerance'):
        interval = fit[key]
        _keys(interval, ('lo', 'hi'))
        if any(type(child) is not float or not math.isfinite(child) or child < 0
                for child in interval.values()) or interval['lo'] > interval['hi']:
            _error('invalid scalar interval')
    if fit['scaled_tolerance'] != dict(lo=0.0, hi=0.0):
        _error('native zero tolerance differs from its interval')
    rng = fit['random_identity']
    _keys(rng, ('schema', 'numpy', 'seed', 'algorithm', 'calls', 'state_sha256', 'draw_sha256'))
    if rng['schema'] != 'seedfiddle.kmeans-randomstate.v1' or rng['algorithm'] != 'MT19937':
        _error('unsupported estimator RNG')
    _int(rng['seed'], request['seed'], request['seed'])
    _int(rng['calls'], 1, 100_000_000)
    _sha(rng['state_sha256'])
    _sha(rng['draw_sha256'])
    calculation = value['calculation']
    _keys(calculation, ('row_sha256', 'fit_modules', 'runtime'))
    _sha(calculation['row_sha256'])
    modules = calculation['fit_modules']
    if type(modules) is not dict or set(modules) != set(FIT_MODULES):
        _error('invalid loaded producer identities')
    for name, digest in modules.items():
        if not name.startswith('seedvision.'):
            _error('invalid producer module name')
        _sha(digest)
    runtime = calculation['runtime']
    _keys(runtime, ('torch', 'torch_cuda', 'numpy', 'opencv', 'qt', 'device', 'device_name'))
    for child in runtime.values():
        _text(child)
    if rng['numpy'] != runtime['numpy']:
        _error('RNG and calculation NumPy identities differ')
    pins = value['source_pins']
    _keys(pins, ('plantcv_version', 'train_sha256', 'predict_sha256', 'sklearn_kmeans_sha256'))
    if pins['plantcv_version'] != '4.11.3':
        _error('unsupported PlantCV source version')
    expected_pins = dict(train_sha256='dbfb8a528e92984a20e73c7f265a41efa9fa088b42ed4f65addbe6ab40ee8b4d',
        predict_sha256='09019da00e0713c51cff31ffd240f7bf8aa0b4e228f6d88f2bed9c1d40430392',
        sklearn_kmeans_sha256='0cf335b8b9b4abe67259ccdcbaf07f02bd7feee31e63a06a1cce32a87fcd375d')
    if any(pins[key] != digest for key, digest in expected_pins.items()):
        _error('source pin differs from the supported calculation')


def _validate_provenance(value, model):
    if type(value) is not dict or value.get('kind') not in ('native_training', 'plantcv_conversion'):
        _error('unsupported provenance kind')
    identity = value.get('numeric_identity')
    _keys(identity, ('source_sha256', 'source_bytes', 'centres_sha256'))
    _int(identity['source_bytes'], 1, _model.MAX_MODEL_BYTES)
    _sha(identity['source_sha256'])
    _sha(identity['centres_sha256'])
    if identity != numeric_model_identity(model):
        _error('embedded numeric model identity differs')
    if value['kind'] == 'native_training':
        _native_provenance(value, model)
    else:
        _keys(value, ('kind', 'numeric_identity', 'input', 'environment', 'geometry'))
        source = value['input']
        _keys(source, ('source_sha256', 'source_bytes', 'estimator_class'))
        _sha(source['source_sha256'])
        _int(source['source_bytes'], 1, 64*1024*1024)
        if source['estimator_class'] != 'sklearn.cluster._kmeans.MiniBatchKMeans':
            _error('unsupported source estimator class')
        environment = value['environment']
        _keys(environment, ('plantcv', 'numpy', 'sklearn', 'joblib'))
        for child in environment.values():
            _text(child)
        geometry = value['geometry']
        _keys(geometry, ('patch_size', 'channels', 'method'))
        _int(geometry['patch_size'], model.patch_size, model.patch_size)
        _int(geometry['channels'], model.channels, model.channels)
        if geometry['method'] != 'unique-supported-feature-width':
            _error('unsupported source geometry inference')


def encode_kmeans_artifact(model, provenance):
    """Finish/validate one bounded artifact before its atomic save begins."""
    metadata = _canonical_metadata(provenance)
    normalized = json.loads(metadata)
    _validate_provenance(normalized, model)
    numeric = _model.encode_kmeans_model(model)
    raw = (b'{"schema":'+json.dumps(SCHEMA).encode('ascii')+b',"numeric_model":'+numeric
        +b',"provenance":'+metadata+b',"provenance_sha256":"'
        +sha256(metadata).hexdigest().encode('ascii')+b'"}')
    if len(raw) > _model.MAX_MODEL_BYTES:
        _error('complete artifact exceeds the eight-MiB bound')
    return raw


def parse_kmeans_artifact(source):
    """Accept legacy numeric JSON or a provenance envelope without file I/O."""
    try:
        if type(source) is str and len(source) > _model.MAX_MODEL_BYTES:
            _error('oversized model artifact')
        raw = source.encode('utf-8') if type(source) is str else source
        if type(raw) is not bytes or not 0 < len(raw) <= _model.MAX_MODEL_BYTES:
            _error('empty or oversized model artifact')
        value = json.loads(raw, object_pairs_hook=_model._object_pairs, parse_constant=_model._nonfinite)
    except (ValueError, UnicodeError, RecursionError) as error:
        raise _model.KMeansModelError(f'invalid K-means artifact: {error}') from error
    if type(value) is dict and value.get('schema') == _model.SCHEMA:
        return KMeansArtifact(_model.parse_kmeans_model(raw), None, None)
    _keys(value, ('schema', 'numeric_model', 'provenance', 'provenance_sha256'))
    if value['schema'] != SCHEMA:
        _error('unsupported model artifact schema')
    # The existing codec alone owns numeric shape/value validation. Re-encode
    # through its canonical encoder before checking the embedded numeric hash.
    try:
        inner = json.dumps(value['numeric_model'], separators=(',', ':'), allow_nan=False).encode('utf-8')
        model = _model.parse_kmeans_model(inner)
    except (ValueError, OverflowError, RecursionError) as error:
        raise _model.KMeansModelError('invalid embedded numeric model') from error
    metadata = _canonical_metadata(value['provenance'])
    digest = sha256(metadata).hexdigest()
    if value['provenance_sha256'] != digest:
        _error('provenance digest differs')
    _validate_provenance(value['provenance'], model)
    return KMeansArtifact(replace(model, source_sha256=sha256(raw).hexdigest(), source_bytes=len(raw)), metadata, digest)


def load_kmeans_artifact(path):
    try:
        with Path(path).open('rb') as stream:
            raw = stream.read(_model.MAX_MODEL_BYTES+1)
    except OSError as error:
        raise _model.KMeansModelError(f'unable to read K-means model artifact: {error}') from error
    return parse_kmeans_artifact(raw)
