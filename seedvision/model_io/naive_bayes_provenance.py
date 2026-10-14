"""Compact project-held training records for interoperable PlantCV TSV models.

The record is descriptive provenance, not permission to access its original
samples and not a scientific or numerical certificate. No training files are
opened here. Prediction uses the separately verified TSV asset.
"""
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import PurePosixPath
import re
import struct

MAX_PROVENANCE_BYTES = 512 * 1024
MAX_METADATA_ITEMS = 50_000
MAX_METADATA_DEPTH = 20
MAX_METADATA_STRING = 32_768
NUMERICAL_PROFILE = 'ordered-f64; reduction-derived plus empirical transcendental allowance'
BUFFERED_NUMERICAL_PROFILE = ('ordered-f64; numpy-2.2.6-buffer8192-v1; '
    'reduction-derived plus empirical transcendental allowance')
CONSTANT_NUMERICAL_PROFILE = ('ordered-f64; numpy-2.2.6-buffer8192-constant-v1; '
    'positive-sum covariance plus empirical normalization allowance')
PARSER_ID = 'plantcv-4.11.3-ordered-sampled-rgb-v1'
DIRECTORY_PARSER_ID = 'plantcv-4.11.3-ordered-binary-directories-v1'
COORDINATE_DIRECTORY_PARSER_ID = 'plantcv-4.11.3-ordered-binary-directories-v2'
SOURCE_PINS = {
    'plantcv': '4.11.3', 'scipy': '1.15.3', 'numpy': '2.2.6',
    'plantcv_train_sha256': 'f59c95c493d189b6110d47a60a97c86e306ab87583a2f128eff8c389aa6b56eb',
    'scipy_kde_sha256': 'e08717ede16febe4d374f21348036fae6a4739f0de88fa32e9c0e8a097ad6839',
    'scipy_stats_sha256': 'de1368dab08c654cce5a321fb47a590b9dc415336a61defd97e4c34be484f7dc',
    'numpy_pairwise_sha256': 'ed85d5b99f0b457cb7a316bf8c4acb884034bbbb81a89c5cedb62be80bf7c4f8',
}
CALCULATION_MODULES = (
    'seedvision.cuda.plantcv_naive_bayes_training', 'seedvision.cuda._nvrtc',
    'seedvision.cuda.plantcv_range', 'seedvision.cuda.plantcv_naive_bayes',
    'seedvision.benchmarking.native_naive_bayes_training',
)
DIRECTORY_CALCULATION_MODULES = (
    'seedvision.cuda.plantcv_naive_bayes_sampling',
    'seedvision.cuda.plantcv_naive_bayes_training', 'seedvision.cuda._nvrtc',
    'seedvision.cuda.plantcv_range', 'seedvision.cuda.plantcv_naive_bayes',
    'seedvision.benchmarking.naive_bayes_training_directory',
)


@dataclass(frozen=True, slots=True)
class BayesTrainingRecord:
    json_text: str
    sha256: str

    @property
    def metadata(self):
        return json.loads(self.json_text)

    def to_binding(self):
        return {'json': self.json_text, 'sha256': self.sha256}


def _keys(value, names):
    if type(value) is not dict or set(value) != set(names):
        raise ValueError('Bayes training provenance has missing or unexpected fields')


def _integer(value, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError('Bayes training provenance integer is outside its bound')


def _digest(value):
    if type(value) is not str or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise ValueError('Bayes training provenance requires a lowercase SHA-256')


def _number(value, *, positive=False):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError('Bayes training provenance requires bounded finite numeric facts')


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Bayes training provenance has duplicate JSON keys')
        result[key] = value
    return result


def _bounded_metadata(value):
    pending, count = [(value, 0)], 0
    while pending:
        current, depth = pending.pop()
        count += 1
        if count > MAX_METADATA_ITEMS or depth > MAX_METADATA_DEPTH:
            raise ValueError('Bayes training provenance exceeds its item/depth bound')
        if type(current) is dict:
            if any(type(k) is not str or len(k) > MAX_METADATA_STRING for k in current):
                raise ValueError('Bayes training provenance has invalid metadata keys')
            pending.extend((v, depth+1) for v in current.values())
        elif type(current) is list:
            pending.extend((v, depth+1) for v in current)
        elif type(current) is str:
            if len(current) > MAX_METADATA_STRING:
                raise ValueError('Bayes training provenance exceeds its text bound')
        elif type(current) is float:
            if not math.isfinite(current):
                raise ValueError('Bayes training provenance contains nonfinite metadata')
        elif type(current) is int:
            if not -(2**63) <= current < 2**63:
                raise ValueError('Bayes training provenance integer exceeds its metadata bound')
        elif current is not None and type(current) not in (int, bool):
            raise ValueError('Bayes training provenance must contain JSON metadata only')
        if count+len(pending) > MAX_METADATA_ITEMS:
            raise ValueError('Bayes training provenance exceeds its item bound')


def _constant_moments(n, value):
    """Compact repeat-term audit of the fixed source's weighted scalar facts.

    Uniform leaves have eight equal accumulators. Memoize each distinct subtree
    length, retaining the actual additions and outer buffer order. No n-element
    input, reduction tensor, or original source file is materialized or read.
    """
    def repeated_sum(term):
        memo = {}
        def pair(rows):
            if rows in memo:
                return memo[rows]
            if rows > 128:
                half = rows//2
                half -= half%8
                result = pair(half)+pair(rows-half)
            elif rows < 8:
                result = -0.0
                for _ in range(rows):
                    result += term
            else:
                accumulator = term
                for _ in range(rows//8-1):
                    accumulator += term
                r = accumulator
                result = ((r+r)+(r+r))+((r+r)+(r+r))
                for _ in range(rows%8):
                    result += term
            memo[rows] = result
            return result
        result = 0.0
        for start in range(0, n, 8192):
            result += pair(min(8192, n-start))
        return result
    weight = 1.0/n
    sumw = repeated_sum(weight)
    sumw2 = repeated_sum(weight*weight)
    mean = repeated_sum(value*weight)/sumw
    return weight, sumw, sumw2, mean, sumw-sumw2/sumw


def _constant_channel(channel, n):
    value = channel['constant_value']
    _integer(value, 0, 255)
    weight, sumw, sumw2, mean, fact = _constant_moments(n, value)
    covariance = channel['covariance']
    bandwidth, factor = channel['bandwidth'], channel['scott_factor']
    u = 2.0**-53
    gamma = (n+32)*u/(1.0-(n+32)*u)
    residual = value-mean
    term = residual*(residual*weight)
    # This compact product is only the centre of a relative interval, never a
    # substitute for the actual ordered CUDA covariance calculation.
    centre = (n*term)*(1.0/fact)
    covariance_budget = 4*gamma*covariance
    if (channel['mean'] != mean or mean == value
            or channel['effective_samples'] != 1.0/sumw2
            or abs(factor-math.pow(1.0/sumw2,-1.0/5.0)) > 64*u*factor
            or abs(covariance-centre) > 2*gamma*centre
            or channel['covariance_absolute_budget'] != covariance_budget
            or covariance*(1.0+4*gamma) >= 2.0**-12 or factor > 1.0
            or bandwidth != math.sqrt(covariance)*factor):
        raise ValueError('Bayes training provenance has inconsistent constant-channel facts')
    relative_bandwidth = -0.5*math.log1p(-covariance_budget/covariance)+64*u
    if (1.0-1024*u)/(bandwidth*math.exp(relative_bandwidth)) <= 64.0:
        raise ValueError('Bayes training provenance exceeds the constant whitening interval')
    norm = math.pow(2.0*math.pi, -0.5)/bandwidth
    expected_pdf_budget = (norm*math.exp(relative_bandwidth)*sumw*(1.0+4*gamma)
        *math.expm1(relative_bandwidth+4*gamma+64*u)+256*math.ulp(0.0))
    if channel['pdf_absolute_budget'] != expected_pdf_budget:
        raise ValueError('Bayes training provenance has an inconsistent constant PDF budget')


def _relative(value, *, directory=False):
    if (type(value) is not str or not value or '\0' in value
            or PurePosixPath(value).is_absolute() or '..' in PurePosixPath(value).parts
            or PurePosixPath(value).as_posix() != value or (value == '.' and not directory)):
        raise ValueError('Bayes directory provenance requires canonical relative paths')
    return value


def _entry_name(value):
    _relative(value)
    if PurePosixPath(value).name != value:
        raise ValueError('Bayes directory provenance requires flat entry names')


def _rng_state(value):
    _keys(value, ('algorithm', 'keys', 'position', 'has_gauss', 'cached_gaussian', 'sha256'))
    if (value['algorithm'] != 'MT19937' or type(value['keys']) is not str
            or re.fullmatch('[0-9a-f]{4992}', value['keys']) is None
            or type(value['cached_gaussian']) is not str
            or re.fullmatch('[0-9a-f]{16}', value['cached_gaussian']) is None):
        raise ValueError('Bayes directory provenance has an invalid legacy RNG state')
    _integer(value['position'], 0, 624)
    _integer(value['has_gauss'], 0, 1)
    _digest(value['sha256'])
    raw = (b'MT19937\0'+bytes.fromhex(value['keys'])+
        struct.pack('<II', value['position'], value['has_gauss'])+
        bytes.fromhex(value['cached_gaussian']))
    if sha256(raw).hexdigest() != value['sha256']:
        raise ValueError('Bayes directory provenance RNG digest differs from its state')
    return value['sha256']


def _directory_pair(value, index):
    _keys(value, ('index', 'image_relative_path', 'mask_relative_path'))
    _integer(value['index'], index, index)
    _relative(value['image_relative_path'])
    _entry_name(value['mask_relative_path'])
    if PurePosixPath(value['image_relative_path']).name != value['mask_relative_path']:
        raise ValueError('Bayes directory provenance requires a flat matching mask basename')


def _directory_record(value, total, fit_work, fit_scratch):
    selection = value['selection']
    _keys(selection, ('image_root_origin', 'mask_root_origin', 'walk', 'pairs',
        'skipped_missing_masks', 'selection_sha256'))
    for name in ('image_root_origin', 'mask_root_origin'):
        if type(selection[name]) is not str or not selection[name] or '\0' in selection[name]:
            raise ValueError('Bayes directory provenance has an invalid descriptive origin')
    walk, pairs, skipped = selection['walk'], selection['pairs'], selection['skipped_missing_masks']
    if (type(walk) is not list or not 1 <= len(walk) <= 4096
            or type(pairs) is not list or not 1 <= len(pairs) <= 256 or type(skipped) is not list):
        raise ValueError('Bayes directory provenance exceeds its traversal/pair bounds')
    _digest(selection['selection_sha256'])
    selected_fields = {key: item for key, item in selection.items() if key != 'selection_sha256'}
    canonical = json.dumps(selected_fields, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode('utf-8')
    if sha256(canonical).hexdigest() != selection['selection_sha256']:
        raise ValueError('Bayes directory provenance selection digest differs from its manifest')
    for index, pair in enumerate(pairs):
        _directory_pair(pair, index)
    for relative in skipped:
        _relative(relative)
    candidates, entries, seen = [], 0, set()
    for item in walk:
        if type(item) is not list or len(item) != 3:
            raise ValueError('Bayes directory provenance has invalid walk records')
        directory, directories, filenames = item
        _relative(directory, directory=True)
        if directory in seen or (not seen and directory != '.'):
            raise ValueError('Bayes directory provenance has invalid walk-directory order')
        seen.add(directory)
        if type(directories) is not list or type(filenames) is not list:
            raise ValueError('Bayes directory provenance needs ordered entry lists')
        for name in directories+filenames:
            _entry_name(name)
        if len(set(directories+filenames)) != len(directories)+len(filenames):
            raise ValueError('Bayes directory provenance repeats an entry within a directory')
        entries += len(directories)+len(filenames)
        if entries > 4096:
            raise ValueError('Bayes directory provenance exceeds its entry bound')
        for filename in filenames:
            if filename[-3:] in ('png', 'jpg', 'jpeg'):
                candidates.append((PurePosixPath(directory)/filename).as_posix())
    pair_index = skip_index = 0
    for relative in candidates:
        if pair_index < len(pairs) and relative == pairs[pair_index]['image_relative_path']:
            pair_index += 1
        elif skip_index < len(skipped) and relative == skipped[skip_index]:
            skip_index += 1
        else:
            raise ValueError('Bayes directory provenance pairing differs from source candidate order')
    if pair_index != len(pairs) or skip_index != len(skipped):
        raise ValueError('Bayes directory provenance has unmatched selected/skipped candidates')
    occurrences = value['occurrences']
    if type(occurrences) is not list or len(occurrences) != len(pairs):
        raise ValueError('Bayes directory provenance occurrence count differs from its selection')
    prior_rng, final_rng = _rng_state(value['rng_before']), _rng_state(value['rng_after'])
    source_identities, rows, work, index_dtype = {}, 0, 0, None
    coordinates = value['parser_id'] == COORDINATE_DIRECTORY_PARSER_ID
    for index, occurrence in enumerate(occurrences):
        _keys(occurrence, ('pair', 'image', 'mask', 'image_shape', 'mask_shape', 'sampling_plan',
            'draws_sha256', 'random_index_dtype', 'random_calls', 'rng_before_sha256', 'rng_after_sha256'))
        _directory_pair(occurrence['pair'], index)
        if occurrence['pair'] != pairs[index]:
            raise ValueError('Bayes directory provenance occurrence order differs from its selection')
        for kind in ('image', 'mask'):
            identity = occurrence[kind]
            _keys(identity, ('sha256', 'bytes'))
            _digest(identity['sha256'])
            _integer(identity['bytes'], 1, 64*1024*1024)
            key = kind, pairs[index][kind+'_relative_path']
            if key in source_identities and source_identities[key] != identity:
                raise ValueError('Bayes directory provenance repeated source identity differs')
            source_identities[key] = identity
        image_shape, mask_shape = occurrence['image_shape'], occurrence['mask_shape']
        if (type(image_shape) is not list or len(image_shape) != 3
                or type(mask_shape) is not list or len(mask_shape) != 2):
            raise ValueError('Bayes directory provenance has invalid decoded dimensions')
        for dimension in image_shape+mask_shape:
            _integer(dimension, 1, 16_000_000)
        height, width, channels = image_shape
        mask_height, mask_width = mask_shape
        pixels, mask_pixels = height*width, mask_height*mask_width
        if (channels != 3 or pixels > 16_000_000 or mask_pixels > 16_000_000
                or (not coordinates and image_shape != [mask_height, mask_width, 3])):
            raise ValueError('Bayes directory provenance dimensions differ from its bounded parser contract')
        plan = occurrence['sampling_plan']
        plan_fields = ('height', 'width', 'foreground_pixels', 'background_pixels', 'samples_per_class',
            'mask_tiles', 'conversion_tile', 'gather_tile', 'output_bytes', 'cpu_metadata_bytes',
            'work', 'peak_live_bytes')
        _keys(plan, plan_fields + (('mask_height', 'mask_width') if coordinates else ()))
        foreground, background = plan['foreground_pixels'], plan['background_pixels']
        # Completed categorical indexing selects unique coordinates only in
        # the image/mask intersection; ignored values outside it are allowed.
        intersection = min(height, mask_height)*min(width, mask_width)
        _integer(foreground, 0, intersection)
        _integer(background, 0, intersection-foreground)
        n, tiles = foreground//10, (mask_pixels+4095)//4096
        if n and background < 2:
            raise ValueError('Bayes directory provenance reports an impossible completed source draw')
        conversion, gather = plan['conversion_tile'], plan['gather_tile']
        _integer(conversion, 1, min(65_536, pixels))
        _integer(gather, 1 if n else 0, min(65_536, n))
        gathers = 0 if not n else 6*((n+gather-1)//gather)
        current_work = (96*pixels+(16*mask_pixels+16*tiles if coordinates else 8*tiles)
            +2048*((pixels+conversion-1)//conversion)+12*n+256*gathers)
        peak = (6*pixels+mask_pixels+4*(foreground+background)
            +(32 if coordinates else 16)*tiles+6*n+128*conversion+16*gather+4*1024*1024)
        expected = dict(height=height, width=width, samples_per_class=n, mask_tiles=tiles,
            output_bytes=6*n, cpu_metadata_bytes=24*n+(80 if coordinates else 64)*tiles+128*1024,
            work=current_work, peak_live_bytes=peak)
        if coordinates:
            expected.update(mask_height=mask_height, mask_width=mask_width)
        for key, exact in expected.items():
            _integer(plan[key], exact, exact)
        if (plan['cpu_metadata_bytes'] > 16*1024*1024 or 3*rows+peak > 256*1024*1024
                or rows+2*n > 1_000_000 or work+current_work > 2_000_000_000):
            raise ValueError('Bayes directory provenance exceeds its cumulative sampling bound')
        _digest(occurrence['draws_sha256'])
        _digest(occurrence['rng_before_sha256'])
        _digest(occurrence['rng_after_sha256'])
        _integer(occurrence['random_calls'], 6, 6)
        if occurrence['random_index_dtype'] not in ('<i4', '<i8'):
            raise ValueError('Bayes directory provenance has unsupported C-long random indices')
        if index_dtype is not None and occurrence['random_index_dtype'] != index_dtype:
            raise ValueError('Bayes directory provenance changes C-long dtype within one job')
        index_dtype = occurrence['random_index_dtype']
        if (occurrence['rng_before_sha256'] != prior_rng
                or (not n and occurrence['rng_after_sha256'] != prior_rng)):
            raise ValueError('Bayes directory provenance has an inconsistent ordered RNG chain')
        prior_rng = occurrence['rng_after_sha256']
        rows += 2*n
        work += current_work
    if prior_rng != final_rng or value['sample_counts'] != [rows//2, rows//2] or rows != total:
        raise ValueError('Bayes directory provenance sample/RNG totals are inconsistent')
    source_bytes = sum(identity['bytes'] for identity in source_identities.values())
    _integer(value['source_bytes'], source_bytes, source_bytes)
    _integer(source_bytes, 1, 128*1024*1024)
    _integer(value['sampling_work'], work, work)
    if work+fit_work > 2_000_000_000:
        raise ValueError('Bayes directory provenance exceeds the combined sampling/fit work bound')
    preflight = value['fit_preflight']
    _keys(preflight, ('work', 'peak_live_bytes', 'scratch_bytes'))
    for key, exact in (('work', fit_work), ('scratch_bytes', fit_scratch),
            ('peak_live_bytes', 6*total+fit_scratch)):
        _integer(preflight[key], exact, exact)
    _integer(preflight['peak_live_bytes'], 1, 128*1024*1024)
    _integer(value['fit_plan']['live_bytes'], 3*total+fit_scratch, 3*total+fit_scratch)


def parse_training_record(source, model):
    """Validate canonical JSON, its fit facts and its exact TSV/table/class binding."""
    if type(source) is bytes:
        if not 0 < len(source) <= MAX_PROVENANCE_BYTES:
            raise ValueError('Bayes training provenance exceeds its byte bound')
        try:
            text = source.decode('utf-8')
        except UnicodeError as error:
            raise ValueError('Bayes training provenance must be UTF-8') from error
    elif type(source) is str:
        if not 0 < len(source) <= MAX_PROVENANCE_BYTES:
            raise ValueError('Bayes training provenance exceeds its byte bound')
        text = source
    else:
        raise ValueError('Bayes training provenance must be immutable JSON text or bytes')
    try:
        raw = text.encode('utf-8')
        if len(raw) > MAX_PROVENANCE_BYTES:
            raise ValueError('Bayes training provenance exceeds its byte bound')
        value = json.loads(text, object_pairs_hook=_pairs)
        _bounded_metadata(value)
        directory = type(value) is dict and value.get('kind') == 'native_bayes_directory_training'
        canonical = json.dumps(value, ensure_ascii=not directory, sort_keys=True,
            separators=(',', ':'), allow_nan=False)
        if canonical != text:
            raise ValueError('Bayes training provenance must use canonical JSON')
    except (UnicodeError, RecursionError, OverflowError) as error:
        raise ValueError('Bayes training provenance has unsupported encoding/depth/numbers') from error
    shared = ('kind', 'parser_id', 'class_names', 'sample_counts', 'model_sha256',
        'tables_sha256', 'fit_plan', 'channel_fits', 'calculation', 'scientific_validation',
        'runtime', 'source_pins')
    additional = (('selection', 'occurrences', 'source_bytes', 'rng_before', 'rng_after',
        'sampling_work', 'fit_preflight') if directory else
        ('sample_origin_path', 'sample_sha256', 'sample_bytes', 'parsed_samples_sha256', 'headers'))
    _keys(value, shared+additional)
    if (value['kind'] != ('native_bayes_directory_training' if directory else 'native_bayes_sample_training')
            or value['parser_id'] not in ((DIRECTORY_PARSER_ID, COORDINATE_DIRECTORY_PARSER_ID)
                if directory else (PARSER_ID,))
            or value['scientific_validation'] is not False or value['source_pins'] != SOURCE_PINS):
        raise ValueError('Bayes training provenance has an unsupported training contract')
    if not directory:
        origin = value['sample_origin_path']
        if origin is not None and (type(origin) is not str or not origin):
            raise ValueError('Bayes training provenance has an invalid descriptive sample origin')
        _integer(value['sample_bytes'], 1, 16*1024*1024)
        for name in ('sample_sha256', 'parsed_samples_sha256'):
            _digest(value[name])
    for name in ('model_sha256', 'tables_sha256'):
        _digest(value[name])
    classes, counts = value['class_names'], value['sample_counts']
    if (type(classes) is not list or tuple(classes) != model.class_names
            or not 1 <= len(classes) <= 32 or type(counts) is not list or len(counts) != len(classes)):
        raise ValueError('Bayes training provenance class/header/count order differs from the model')
    if directory:
        if classes != ['plant', 'background']:
            raise ValueError('Bayes directory provenance needs source binary class order')
    else:
        headers = value['headers']
        if (type(headers) is not list or not 1 <= len(headers) <= 32
                or any(type(h) is not str or not h or len(h.encode('utf-8')) > 255 for h in headers)
                or list(dict.fromkeys(headers)) != classes):
            raise ValueError('Bayes training provenance class/header/count order differs from the model')
    if value['model_sha256'] != model.source_sha256 or value['tables_sha256'] != model.tables_sha256:
        raise ValueError('Bayes training provenance differs from the selected model bytes/tables')
    for n in counts:
        _integer(n, 2, 1_000_000)
    total, count = sum(counts), len(classes)
    _integer(total, 2, 1_000_000)
    plan = value['fit_plan']
    _keys(plan, ('classes', 'sample_counts', 'samples', 'work', 'scratch_bytes', 'live_bytes'))
    expected_work = total*780 + count*(3*256*256+3*256)
    expected_scratch = 3*256*256*8 + count*3*256*8 + 512 + 2*1024*1024
    if (plan['classes'] != count or plan['sample_counts'] != counts or plan['samples'] != total
            or plan['work'] != expected_work or plan['scratch_bytes'] != expected_scratch):
        raise ValueError('Bayes training provenance has inconsistent fit-plan facts')
    for name in ('classes', 'samples', 'work', 'scratch_bytes', 'live_bytes'):
        _integer(plan[name], 1, 128*1024*1024 if name.endswith('bytes') else 1_000_000_000)
    _integer(plan['scratch_bytes'], 1, 16*1024*1024)
    _integer(plan['live_bytes'], expected_scratch+total*3, 128*1024*1024)
    channel_fits = value['channel_fits']
    if type(channel_fits) is not list or len(channel_fits) != count:
        raise ValueError('Bayes training provenance has invalid channel-fit count')
    profile_families = set()
    for n, channels in zip(counts, channel_fits):
        if type(channels) is not list or len(channels) != 3:
            raise ValueError('Bayes training provenance needs three ordered HSV fit records')
        for channel in channels:
            if type(channel) is not dict:
                raise ValueError('Bayes training provenance needs compact channel records')
            profile = channel.get('numerical_profile')
            fields = ('samples', 'weight', 'mean', 'covariance', 'effective_samples',
                'scott_factor', 'bandwidth', 'covariance_absolute_budget', 'pdf_absolute_budget',
                'numerical_profile')
            constant = profile == CONSTANT_NUMERICAL_PROFILE
            _keys(channel, fields+('constant_value',) if constant else fields)
            _integer(channel['samples'], n, n)
            profile = channel['numerical_profile']
            if (channel['weight'] != 1.0/n or type(profile) is not str
                    or profile not in (NUMERICAL_PROFILE, BUFFERED_NUMERICAL_PROFILE, CONSTANT_NUMERICAL_PROFILE)):
                raise ValueError('Bayes training provenance has unsupported weights/numerical profile')
            profile_families.add('legacy' if profile == NUMERICAL_PROFILE else 'buffer8192')
            for key in ('weight', 'covariance', 'effective_samples', 'scott_factor', 'bandwidth'):
                _number(channel[key], positive=True)
            for key in ('mean', 'covariance_absolute_budget', 'pdf_absolute_budget'):
                _number(channel[key])
            if ((not constant and channel['mean'] > 255)
                    or channel['covariance_absolute_budget'] >= channel['covariance']):
                raise ValueError('Bayes training provenance reports an unsupported covariance domain')
            if constant:
                _constant_channel(channel, n)
    if len(profile_families) != 1:
        raise ValueError('Bayes training provenance mixes numerical profiles within one fit')
    if directory:
        _directory_record(value, total, expected_work, expected_scratch)
    calculation = value['calculation']
    _keys(calculation, DIRECTORY_CALCULATION_MODULES if directory else CALCULATION_MODULES)
    for module, record in calculation.items():
        fields = ('name', 'source_sha256', 'dataclasses', 'imported_aliases',
            'loaded_code', 'runtime_constants')
        if module == 'seedvision.cuda.plantcv_range':
            fields += ('computed_constants',)
        _keys(record, fields)
        if 'computed_constants' in record:
            _digest(record['computed_constants'])
        if (record['name'] != module.rsplit('.', 1)[-1]+'.py'
                or type(record['dataclasses']) is not list
                or type(record['imported_aliases']) is not list
                or type(record['runtime_constants']) is not dict):
            raise ValueError('Bayes training provenance has invalid calculation-module facts')
        _digest(record.get('source_sha256'))
        if type(record.get('loaded_code')) is not list or not record['loaded_code']:
            raise ValueError('Bayes training provenance lacks loaded calculation identities')
        for code in record['loaded_code']:
            _keys(code, ('name', 'code_sha256', 'defaults_sha256'))
            if type(code['name']) is not str or not code['name']:
                raise ValueError('Bayes training provenance has an invalid loaded callable')
            _digest(code['code_sha256'])
            _digest(code['defaults_sha256'])
    runtime = value['runtime']
    runtime_fields = ('torch', 'numpy', 'torch_cuda', 'device', 'device_name', 'nvrtc', 'cuda_driver_api')
    if directory:
        runtime_fields += ('opencv', 'opencv_build_sha256', 'pyside', 'qt')
    _keys(runtime, runtime_fields)
    if any(type(item) is not str or not item for item in runtime.values()):
        raise ValueError('Bayes training provenance has invalid descriptive runtime facts')
    if directory:
        _digest(runtime['opencv_build_sha256'])
    return BayesTrainingRecord(text, sha256(raw).hexdigest())


def validate_training_binding(binding, model):
    _keys(binding, ('json', 'sha256'))
    _digest(binding['sha256'])
    record = parse_training_record(binding['json'], model)
    if record.sha256 != binding['sha256']:
        raise ValueError('Bayes training provenance digest differs from its exact JSON bytes')
    return record
