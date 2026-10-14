"""Primitive archived-model contracts and allocation-free replay admission.

Model origins are descriptive identity. Only verified archived bytes are parsed;
this module never resolves or opens an original model path. Stage dispatch and
CUDA product comparison belong to the guarded replay runner.
"""
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import re
from types import MappingProxyType

import numpy as np

from seedvision.cuda import plantcv_naive_bayes as bayes
from seedvision.cuda import plantcv_kmeans as masks
from seedvision.cuda import plantcv_kmeans_predict as predict
from seedvision.model_io import plantcv_kmeans_model as centres
from seedvision.model_io import kmeans_model_artifact as artifact
from seedvision.model_io.asset_paths import validate_model_origin
from . import native_naive_bayes_recipe as bayes_recipe
from . import native_kmeans_recipe as kmeans_recipe

MAX_METADATA_BYTES = 8 * 1024 * 1024
MAX_METADATA_ITEMS = 4096
MAX_METADATA_DEPTH = 8
MAX_SELECTED_BYTES = 128 * 1024 * 1024
MAX_CUDA_BYTES = 1024 * 1024 * 1024
COMPARE_ELEMENTS = 262_144
AUDIT_CUDA_BYTES = 64 * 1024
CUDA_ALLOCATION_BYTES = 512
REFERENCE_VERSION = '4.11.3'
SOURCE_HASHES = MappingProxyType({
    'bayes': 'd3f40429ead7aa5deb298fa78d484d0f6d89c9cf05e706e07fdf7b97fedc3049',
    'kmeans': '09019da00e0713c51cff31ffd240f7bf8aa0b4e228f6d88f2bed9c1d40430392',
})
MODEL_FIELDS = MappingProxyType({
    'bayes': bayes_recipe.MODEL_FIELDS,
    'kmeans': kmeans_recipe.MODEL_FIELDS,
})


class NativeModelReplayError(ValueError):
    """An archived model or product binding violates its typed contract."""


class NativeModelReplayUnavailable(NativeModelReplayError):
    """A valid requested calculation exceeds its admitted resource domain."""


@dataclass(frozen=True, slots=True)
class ModelAssetBinding:
    kind: str
    identity: Mapping[str, object]
    array_name: str


@dataclass(frozen=True, slots=True)
class ModelReplayPlan:
    """Replay charges plus the source primitive's own admitted live buffers.

    ``native_live_bytes`` follows each primitive's existing cap: Bayes/masks
    exclude their caller-owned input, while prediction includes its input.
    ``cuda_live_bytes`` includes all retained replay inputs and comparisons.
    Raw model and Python metadata bounds do not guarantee process RSS.
    """
    selected_bytes: int
    cuda_live_bytes: int
    native_live_bytes: int
    scratch_bytes: int
    work: int | None
    work_details: tuple[tuple[str, int], ...]


def state():
    return (MAX_METADATA_BYTES, MAX_METADATA_ITEMS, MAX_METADATA_DEPTH,
            MAX_SELECTED_BYTES, MAX_CUDA_BYTES, COMPARE_ELEMENTS, AUDIT_CUDA_BYTES, CUDA_ALLOCATION_BYTES,
            REFERENCE_VERSION, tuple(SOURCE_HASHES.items()), tuple(MODEL_FIELDS.items()),
            bayes.MAX_MODEL_BYTES, bayes.MAX_CLASSES, bayes.MAX_PIXELS,
            bayes.MAX_OUTPUT_BYTES, bayes.MAX_SCRATCH_BYTES, bayes.MAX_CUDA_LIVE_BYTES,
            bayes.TILE_PIXELS, bayes.SCRATCH_ENVELOPE_BYTES_PER_PIXEL,
            masks.MAX_PIXELS, masks.MAX_OUTPUT_BYTES, masks.MAX_TILE_PIXELS, masks.MAX_CAT_LIST,
            centres.MAX_MODEL_BYTES, centres.MAX_CLUSTERS, centres.MAX_FEATURES,
            centres.MAX_CENTRE_VALUES, centres.PREPROCESSING,
            predict.MAX_WORK, predict.MAX_SCRATCH_BYTES, predict.MAX_CUDA_LIVE_BYTES,
            predict.DEFAULT_FEATURE_TILE_PIXELS, predict.DEFAULT_GAUSSIAN_TILE_SIDE,
            predict._preprocess.MAX_IMAGE_PIXELS, predict._preprocess.MAX_PADDED_PIXELS,
            predict._preprocess.MAX_OUTPUT_BYTES, predict._preprocess.MAX_TILE_SIDE,
            predict._preprocess.MAX_SCRATCH_BYTES, predict._preprocess.MAX_FILTER_WORK,
            predict._classifier.MAX_WORK, predict._classifier.MAX_SCRATCH_BYTES,
            predict._classifier.SCRATCH_HEADROOM_BYTES,
            predict._features.MAX_FEATURE_TILE_PIXELS, predict._features.MAX_FEATURE_TILE_BYTES,
            predict._features.FEATURE_TILE_ROW_OVERHEAD_BYTES,
            artifact.SCHEMA, artifact.MAX_PROVENANCE_BYTES, artifact.MAX_METADATA_ITEMS,
            artifact.MAX_METADATA_DEPTH, artifact.MAX_METADATA_STRING)


PINNED_STATE = state()


def _walk(root, tagged, poll=None):
    visits = encoded = 0
    byte_cap, item_cap, depth_cap = MAX_METADATA_BYTES, MAX_METADATA_ITEMS, MAX_METADATA_DEPTH

    def charge(amount):
        nonlocal encoded
        encoded += amount
        if encoded > byte_cap:
            raise NativeModelReplayError('Model metadata exceeds its encoded-byte bound.')

    def walk(value, depth=0):
        nonlocal visits
        visits += 1
        if visits > item_cap or depth > depth_cap:
            raise NativeModelReplayError('Model metadata exceeds its traversal bound.')
        if poll is not None and visits % 256 == 0:
            poll()
        if type(value) is str:
            # Source class names have no smaller per-name cap than model bytes.
            if len(value) > byte_cap:
                raise NativeModelReplayError('Model metadata text exceeds its byte bound.')
            try:
                raw = value.encode('utf-8')
            except UnicodeEncodeError as error:
                raise NativeModelReplayError('Model metadata must be UTF-8.') from error
            if len(raw) > byte_cap:
                raise NativeModelReplayError('Model metadata text exceeds its byte bound.')
            charge(len(json.dumps(value, ensure_ascii=False).encode('utf-8')))
            return value
        if value is None or type(value) in (bool, int, float):
            if type(value) is int and value.bit_length() > 64:
                raise NativeModelReplayError('Model identity integer exceeds its bound.')
            if type(value) is float and not math.isfinite(value):
                raise NativeModelReplayError('Model metadata must contain finite numbers.')
            charge(len(json.dumps(value)))
            return value
        if tagged:
            if (type(value) is not dict or set(value) != {'kind', 'items'}
                    or type(value['items']) is not list or type(value['kind']) is not str):
                raise NativeModelReplayError('Model metadata requires tagged primitive containers.')
            items, kind = value['items'], value['kind']
            if len(items) > item_cap - visits:
                raise NativeModelReplayError('Model metadata exceeds its traversal bound.')
            charge(32 + len(items))
            if kind in ('tuple', 'list'):
                return tuple(walk(item, depth + 1) for item in items)
            if kind != 'mapping':
                raise NativeModelReplayError('Model metadata contains an unsupported tag.')
            result = {}
            for item in items:
                if (type(item) is not list or len(item) != 2
                        or type(item[0]) is not str or item[0] in result):
                    raise NativeModelReplayError('Model metadata has an invalid mapping entry.')
                key = walk(item[0], depth + 1)
                result[key] = walk(item[1], depth + 1)
                charge(4)
            return MappingProxyType(result)
        if type(value) in (tuple, list):
            if len(value) > item_cap - visits:
                raise NativeModelReplayError('Model metadata exceeds its traversal bound.')
            charge(32 + len(value))
            return tuple(walk(item, depth + 1) for item in value)
        if isinstance(value, Mapping):
            if len(value) > item_cap - visits:
                raise NativeModelReplayError('Model metadata exceeds its traversal bound.')
            charge(32 + len(value))
            result = {}
            for key, item in value.items():
                if type(key) is not str:
                    raise NativeModelReplayError('Model metadata keys must be strings.')
                result[walk(key, depth + 1)] = walk(item, depth + 1)
                charge(4)
            return MappingProxyType(result)
        raise NativeModelReplayError('Model metadata contains an unknown type.')

    return walk(root)


def decode(binding, poll=None):
    return _walk(binding, True, poll)


def canonical(value, poll=None):
    return _walk(value, False, poll)


def class_names(value, maximum):
    if type(maximum) is not int or not 1 <= maximum <= centres.MAX_CLUSTERS:
        raise NativeModelReplayError('Class-name count bound is invalid.')
    value = canonical(value)
    if (type(value) is not tuple or not 1 <= len(value) <= maximum
            or any(type(name) is not str or not name for name in value)
            or len(set(value)) != len(value)):
        raise NativeModelReplayError('Class names must be ordered, unique, nonempty text.')
    return value


def identity(value, kind):
    if kind not in MODEL_FIELDS:
        raise NativeModelReplayError('Unknown archived model kind.')
    value = canonical(value)
    fields = set(MODEL_FIELDS[kind]) | {'reference_version', 'reference_source_sha256'}
    if not isinstance(value, Mapping) or set(value) != fields:
        raise NativeModelReplayError('Model identity has missing or unknown fields.')
    try:
        validate_model_origin(value['path'])
    except (TypeError, ValueError) as error:
        raise NativeModelReplayError('Model origin must be canonical absolute path text.') from error
    for name in ('source_sha256', 'tables_sha256' if kind == 'bayes' else 'centres_sha256'):
        if type(value[name]) is not str or re.fullmatch('[0-9a-f]{64}', value[name]) is None:
            raise NativeModelReplayError('Model identity requires lowercase SHA-256 values.')
    source_cap = bayes.MAX_MODEL_BYTES if kind == 'bayes' else centres.MAX_MODEL_BYTES
    if (type(value['source_bytes']) is not int or not 1 <= value['source_bytes'] <= source_cap
            or value['reference_version'] != REFERENCE_VERSION
            or value['reference_source_sha256'] != SOURCE_HASHES[kind]):
        raise NativeModelReplayError('Model source size or reference identity is unsupported.')
    if kind == 'bayes':
        class_names(value['class_names'], bayes.MAX_CLASSES)
    else:
        patch, channels, count = value['patch_size'], value['channels'], value['cluster_count']
        if (type(patch) is not int or not 1 <= patch <= 31
                or type(channels) is not int or channels not in (1, 3)
                or type(count) is not int or not 1 <= count <= centres.MAX_CLUSTERS
                or patch * patch * channels > centres.MAX_FEATURES
                or count * patch * patch * channels > centres.MAX_CENTRE_VALUES
                or value['preprocessing'] != centres.PREPROCESSING):
            raise NativeModelReplayError('Model geometry or preprocessing identity is unsupported.')
    return value


def _selection_matches(value, selected, kind):
    if kind == 'bayes':
        try:
            bayes_recipe.validate_model_source_metadata(selected)
        except (TypeError, ValueError) as error:
            raise NativeModelReplayError('Saved Bayes selection is invalid.') from error
        selected = bayes_recipe.calculation_model_source(selected)
    if not isinstance(selected, Mapping) or set(selected) != set(MODEL_FIELDS[kind]):
        raise NativeModelReplayError('Saved model selection has missing or unknown fields.')
    selected = canonical(selected)
    if any(type(selected[name]) is not type(value[name]) or selected[name] != value[name]
           for name in MODEL_FIELDS[kind]):
        raise NativeModelReplayError('Archived model identity differs from the saved selection.')


def _array_reference(binding, specs, shape, *, data_hash=None):
    if (type(binding) is not dict or set(binding) != {'kind', 'name'}
            or type(binding['kind']) is not str or binding['kind'] != 'array'
            or type(binding['name']) is not str):
        raise NativeModelReplayError('Model products require direct archived array references.')
    spec = specs.get(binding['name'])
    if (not isinstance(spec, Mapping) or type(spec.get('shape')) not in (tuple, list)
            or any(type(item) is not int or item < 1 for item in spec['shape'])
            or tuple(spec['shape']) != tuple(shape)
            or spec.get('dtype') != '|u1' or spec.get('layout') != 'C-contiguous'
            or type(spec.get('nbytes')) is not int or spec['nbytes'] != math.prod(shape)
            or data_hash is not None and spec.get('data_sha256') != data_hash):
        raise NativeModelReplayError('Archived model/product header differs from its byte contract.')
    return binding['name']


def asset_binding(binding, kind, specs, selected, poll=None):
    if (type(binding) is not dict or set(binding) != {'kind', 'identity', 'bytes'}
            or type(binding['kind']) is not str or binding['kind'] != 'model_asset'):
        raise NativeModelReplayError('Model input requires its archived byte-bound asset.')
    value = identity(decode(binding['identity'], poll), kind)
    _selection_matches(value, selected, kind)
    name = _array_reference(binding['bytes'], specs, (value['source_bytes'],),
                            data_hash=value['source_sha256'])
    return ModelAssetBinding(kind, value, name)


def _model_matches(value, model, kind):
    facts = {'source_bytes': model.source_bytes, 'source_sha256': model.source_sha256}
    if kind == 'bayes':
        facts.update(tables_sha256=model.tables_sha256, class_names=model.class_names)
    else:
        facts.update(centres_sha256=model.centres_sha256, patch_size=model.patch_size,
                     channels=model.channels, cluster_count=len(model.centres))
    if any(type(item) is not type(value[name]) or item != value[name] for name, item in facts.items()):
        raise NativeModelReplayError('Parsed numerical model differs from its archived identity.')


def parse_asset(binding, raw_array):
    value = identity(binding.identity, binding.kind)
    if (type(raw_array) is not np.ndarray or raw_array.dtype != np.uint8
            or raw_array.shape != (value['source_bytes'],) or not raw_array.flags.c_contiguous):
        raise NativeModelReplayError('Archived model bytes must be a bounded contiguous CPU byte vector.')
    raw = raw_array.tobytes()
    if sha256(raw).hexdigest() != value['source_sha256']:
        raise NativeModelReplayError('Archived model bytes differ from their SHA-256 identity.')
    try:
        model = bayes.parse_plantcv_pdf(raw) if binding.kind == 'bayes' else artifact.parse_kmeans_artifact(raw).model
    except ValueError as error:
        raise NativeModelReplayError('Archived model bytes violate their numerical format.') from error
    _model_matches(value, model, binding.kind)
    return model


def class_mask_bindings(names_binding, masks_binding, specs, shape, *, count, maximum, poll=None):
    if type(count) is not int or not 1 <= count <= maximum:
        raise NativeModelReplayError('Class-mask count must match a nonempty bounded model output.')
    if (type(names_binding) is not dict or type(names_binding.get('kind')) is not str
            or names_binding.get('kind') != 'tuple'
            or set(names_binding) != {'kind', 'items'}):
        raise NativeModelReplayError('Ordered class names require their tuple binding.')
    names = class_names(decode(names_binding, poll), maximum)
    if len(names) != count:
        raise NativeModelReplayError('Archived class-mask count differs from its operation domain.')
    if (type(masks_binding) is not dict or set(masks_binding) != {'kind', 'items'}
            or type(masks_binding['kind']) is not str or masks_binding['kind'] != 'mapping'
            or type(masks_binding['items']) is not list
            or len(masks_binding['items']) != count):
        raise NativeModelReplayError('Class masks require one ordered array mapping per class.')
    pairs, seen = [], set()
    for expected_name, item in zip(names, masks_binding['items']):
        if type(item) is not list or len(item) != 2 or type(item[0]) is not str or item[0] != expected_name:
            raise NativeModelReplayError('Class-mask keys and ordered class names disagree.')
        name = _array_reference(item[1], specs, shape)
        if name in seen:
            raise NativeModelReplayError('Class masks must bind distinct archived arrays.')
        seen.add(name)
        pairs.append((expected_name, name))
        if poll is not None:
            poll()
    return names, tuple(pairs)


def mask_names(parameters, cluster_count):
    if type(cluster_count) is not int or not 1 <= cluster_count <= centres.MAX_CLUSTERS:
        raise NativeModelReplayError('Mask replay requires a nonempty bound cluster count.')
    if not isinstance(parameters, Mapping) or set(parameters) != {'cat_list'}:
        raise NativeModelReplayError('K-means mask replay requires its exact category control.')
    cats = parameters['cat_list']
    if cats is None:
        return tuple(str(index) for index in range(cluster_count)), cluster_count
    if (type(cats) not in (list, tuple) or not 1 <= len(cats) <= masks.MAX_CAT_LIST
            or any(type(item) is not int for item in cats)):
        raise NativeModelReplayError('Category selection requires a nonempty exact-integer sequence.')
    return ('selected',), len(cats)


def _pixels(shape, maximum):
    if (type(shape) not in (tuple, list) or len(shape) != 2
            or any(type(item) is not int or item < 1 for item in shape)):
        raise NativeModelReplayError('Replay frame requires positive integer height and width.')
    pixels = math.prod(shape)
    if pixels > maximum:
        raise NativeModelReplayUnavailable('Replay raster exceeds the native pixel limit.')
    return pixels


def _admit(plan, selected_limit, cuda_limit):
    if (type(selected_limit) is not int or not 1 <= selected_limit <= MAX_SELECTED_BYTES
            or type(cuda_limit) is not int or not 1 <= cuda_limit <= MAX_CUDA_BYTES):
        raise NativeModelReplayError('Replay resource caps may only lower the declared limits.')
    if plan.selected_bytes > selected_limit or plan.cuda_live_bytes > cuda_limit:
        raise NativeModelReplayUnavailable('Model replay exceeds selected-array or CUDA live-storage admission.')
    return plan


def _allocation_bytes(size):
    """Admit the default native CUDA allocator's minimum allocation rounding."""
    return ((size + CUDA_ALLOCATION_BYTES - 1) // CUDA_ALLOCATION_BYTES) * CUDA_ALLOCATION_BYTES


def _comparison_bytes(pixels):
    # count_nonzero includes a byte inequality tile and its int64 reduction
    # input; charge its scalar too. Binary validation needs three boolean tiles.
    tile = min(pixels, COMPARE_ELEMENTS)
    return max(3 * _allocation_bytes(tile),
               _allocation_bytes(tile) + _allocation_bytes(8 * tile) + CUDA_ALLOCATION_BYTES)


def bayes_plan(shape, value, *, selected_limit=MAX_SELECTED_BYTES, cuda_limit=MAX_CUDA_BYTES):
    value = identity(value, 'bayes')
    pixels = _pixels(shape, bayes.MAX_PIXELS)
    count = len(value['class_names'])
    output, table = count * pixels, count * 3 * 256 * 8
    scratch = min(pixels, bayes.TILE_PIXELS) * bayes.SCRATCH_ENVELOPE_BYTES_PER_PIXEL
    native = output + table + scratch
    if (output > bayes.MAX_OUTPUT_BYTES or scratch > bayes.MAX_SCRATCH_BYTES
            or native > bayes.MAX_CUDA_LIVE_BYTES):
        raise NativeModelReplayUnavailable('Bayes replay exceeds the native output/scratch/live domain.')
    live = (2 * _allocation_bytes(3 * pixels) + count * _allocation_bytes(pixels)
            + _allocation_bytes(output) + _allocation_bytes(table) + scratch
            + _comparison_bytes(pixels) + AUDIT_CUDA_BYTES)
    plan = ModelReplayPlan((3 + count) * pixels + value['source_bytes'], live, native, scratch,
                           None, (('class_pixel_visits', pixels * count),))
    return _admit(plan, selected_limit, cuda_limit)


def masks_plan(shape, value, parameters, *, selected_limit=MAX_SELECTED_BYTES, cuda_limit=MAX_CUDA_BYTES):
    value = identity(value, 'kmeans')
    pixels = _pixels(shape, masks.MAX_PIXELS)
    names, scans = mask_names(parameters, value['cluster_count'])
    output = pixels * len(names)
    if output > masks.MAX_OUTPUT_BYTES:
        raise NativeModelReplayUnavailable('K-means masks exceed the native aggregate output limit.')
    scratch = 2 * min(pixels, masks.MAX_TILE_PIXELS)
    live = (_allocation_bytes(pixels) + 2 * len(names) * _allocation_bytes(pixels)
            + 2 * _allocation_bytes(min(pixels, masks.MAX_TILE_PIXELS))
            + _comparison_bytes(pixels) + AUDIT_CUDA_BYTES)
    plan = ModelReplayPlan(pixels + output, live, output + scratch, scratch,
                           None, (('category_pixel_visits', pixels * scans),))
    return _admit(plan, selected_limit, cuda_limit)


def prediction_plan(shape, value, model, *, selected_limit=MAX_SELECTED_BYTES, cuda_limit=MAX_CUDA_BYTES):
    value = identity(value, 'kmeans')
    pixels = _pixels(shape, predict._preprocess.MAX_IMAGE_PIXELS)
    try:
        centres.validate_kmeans_model(model)
    except ValueError as error:
        raise NativeModelReplayError('Prediction requires a validated immutable numerical model.') from error
    _model_matches(value, model, 'kmeans')
    try:
        native = predict.preflight_kmeans_prediction_shape(*shape, model)
    except ValueError as error:
        raise NativeModelReplayUnavailable('K-means prediction exceeds its native shape/work/storage domain.') from error
    channels = value['channels']
    live = (_allocation_bytes(channels * pixels) + (_allocation_bytes(3 * pixels) if channels == 3 else 0)
            + _allocation_bytes(native.gaussian_output_bytes) + 2 * _allocation_bytes(pixels) + native.scratch_bytes
            + _comparison_bytes(pixels) + AUDIT_CUDA_BYTES)
    plan = ModelReplayPlan((channels + 1) * pixels + value['source_bytes'], live,
                           native.live_bytes, native.scratch_bytes, native.work,
                           (('feature_count', native.feature_count), ('feature_tile_pixels', native.tile_pixels)))
    return _admit(plan, selected_limit, cuda_limit)


def compare(actual, expected, poll):
    """Visit every validated primitive leaf, including every ordered name."""
    actual, expected = canonical(actual), canonical(expected)
    missing = object()
    def visit(left, right):
        poll()
        if isinstance(left, Mapping) and isinstance(right, Mapping):
            rows = [visit(left.get(key,missing),right.get(key,missing)) for key in sorted(set(left)|set(right))]
        elif type(left) is tuple and type(right) is tuple:
            rows = [visit(left[index] if index<len(left) else missing,
                          right[index] if index<len(right) else missing) for index in range(max(len(left),len(right)))]
        else:
            return int(type(left) is not type(right) or left!=right),1
        return sum(row[0] for row in rows),sum(row[1] for row in rows)
    mismatch,count=visit(actual,expected)
    return {'mismatched_elements':mismatch,'compared_elements':count,'shape_equal':True,'kind':'compact_model_metadata'}
