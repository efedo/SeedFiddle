"""Durable native graph products, separate from reviewed instance reports.

CUDA products transfer only at this explicit persistence boundary. A run may
span several existing bounded checkpoints; those are output archives, not
per-stage replay certificates or historical StageRunRecord replacements.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile

import numpy as np
import torch

from seedvision.benchmarking.checkpoints import (
    CheckpointCancelled, CheckpointIdentity, CheckpointLimits, ExecutionRecord, ExecutionStatus,
    MeasurementContract, RasterRole, load_checkpoint, publish_checkpoint,
    _array_descriptor, _estimate_npz_size, _read_manifest, _validate_array_specs,
)
from seedvision.benchmarking.native_recipe import NativeRecipeCancelled, NativeRecipeResult
from seedvision.benchmarking.native_correction_recipe import NativeCorrectionResult, RECIPE_ID as CORRECTION_RECIPE_ID
from seedvision.benchmarking.native_correction_inputs import (
    validate_correction_input_sources, correction_file_implementation_sha256,
)
from seedvision.benchmarking.recipes import RecipeIdentity, RecipeSpec
from seedvision.persistence.correction_inputs import (
    CorrectionRunOrigins, validate_correction_run_origins, PROJECT_RUNTIME_KEYS,
)


FORMAT = 'seedfiddle.native-analysis-result'
VERSION = 5
MAX_REPORT_BYTES = 8 * 1024 * 1024
MAX_METADATA_ITEMS = 1_000_000
MAX_METADATA_DEPTH = 64
MAX_RESULT_ARRAYS = 1024
_SUCCESS = frozenset(('provided', 'executed', 'empty', 'partial', 'bypassed'))
_STATES = _SUCCESS | frozenset(('failed', 'not_run', 'unavailable'))


def _public_provenance(stage, value):
    from seedvision.pipeline.backend_execution import (
        validate_backend_provenance, WATERSHED_OPERATIONS,
    )
    checked = validate_backend_provenance(value)
    operation = {
        'color_card.means': 'transform.get_color_matrix',
        'color_card.standard': 'transform.std_color_matrix',
        'color_card.affine': 'transform.affine_color_correction',
    }.get(stage.operation_id, stage.operation_id)
    if stage.operation_id in WATERSHED_OPERATIONS:
        operation = 'segmentation.watershed.capture'
    if checked['operation'] != operation:
        raise NativeResultExportError('Public provenance belongs to another operation.')
    return checked


class NativeResultExportError(ValueError):
    """A result cannot be archived with its exact declared identity."""


def _poll(cancel):
    if cancel is not None and cancel():
        raise NativeRecipeCancelled('Native result export cancelled before publication.')


def _json_bytes(value):
    try:
        raw = json.dumps(value, ensure_ascii=False, sort_keys=True,
            separators=(',', ':'), allow_nan=False).encode('utf-8')
    except (ValueError, TypeError, UnicodeError, RecursionError) as error:
        raise NativeResultExportError('Native result metadata is not strict JSON.') from error
    if len(raw) > MAX_REPORT_BYTES:
        raise NativeResultExportError('Native result report exceeds its 8 MiB bound.')
    return raw


class _Collector:
    def __init__(self, frame, limits, cancel):
        self.frame, self.limits, self.cancel = frame, limits, cancel
        self.arrays, self.roles, self.seen = {}, {}, {}
        self.signatures = []
        self.active = set()
        self.encoded_bytes = 0
        self.items = self.text_bytes = self.total_bytes = 0

    def array(self, value, *, cpu_asset=False, cpu_matrix=False):
        if torch.is_tensor(value):
            if value.device.type != 'cuda' or value.layout != torch.strided or value.requires_grad:
                raise NativeResultExportError('Analysis products require strided CUDA tensors without gradients.')
            self.signatures.append((value, self.signature(value)))
        elif cpu_matrix:
            _immutable_correction_matrix(value)
            self.signatures.append((value, self.signature(value)))
        elif (not cpu_asset or not isinstance(value, np.ndarray) or value.ndim != 1
              or value.dtype != np.uint8 or value.flags.writeable or not isinstance(value.base, bytes)):
            raise NativeResultExportError('Only verified immutable model bytes may be CPU arrays.')
        previous = self.seen.get(id(value))
        if previous is not None:
            return {'kind': 'array', 'name': previous}
        if len(self.arrays) >= MAX_RESULT_ARRAYS:
            raise NativeResultExportError('Native result exceeds its array-count bound.')
        name = f'array_{len(self.arrays):04d}'
        shape, dtype, size = _array_descriptor(value, limits=self.limits, name=name)
        self.total_bytes += size
        if self.total_bytes > self.limits.max_total_array_bytes:
            raise NativeResultExportError('Native result exceeds its aggregate array-byte bound.')
        self.arrays[name] = value
        self.seen[id(value)] = name
        # Mixed graph products have different mask/image/label semantics. Bind
        # their frame here, retaining exact port semantics in the saved recipe.
        hw = self.frame.shape
        is_hwc = len(shape) == 3 and shape[:2] == hw
        is_chw = len(shape) == 3 and shape[1:] == hw
        layout = ('HW' if shape == hw else 'HWC' if is_hwc and not is_chw
                  else 'CHW' if is_chw and not is_hwc else None)
        if layout is not None and not cpu_matrix:
            self.roles[name] = RasterRole('raster', dtype.str, layout)
        return {'kind': 'array', 'name': name}

    @staticmethod
    def signature(value):
        if isinstance(value, np.ndarray):
            return (tuple(value.shape), value.dtype.str, value.strides, value.flags.writeable,
                    value.__array_interface__['data'][0], sha256(value).hexdigest())
        try:
            return (value._version, tuple(value.shape), value.data_ptr(), value.dtype,
                    tuple(value.stride()), value.storage_offset())
        except RuntimeError as error:
            raise NativeResultExportError('Native output tensor lacks a verifiable mutation version.') from error

    def charge(self, amount):
        self.encoded_bytes += amount
        if self.encoded_bytes > MAX_REPORT_BYTES:
            raise NativeResultExportError('Native metadata exceeds its encoded-byte collection bound.')

    def encode(self, value, depth=0):
        _poll(self.cancel)
        self.items += 1
        if self.items > MAX_METADATA_ITEMS or depth > MAX_METADATA_DEPTH:
            raise NativeResultExportError('Native result exceeds its metadata traversal bound.')
        container = isinstance(value, (Mapping, tuple, list)) or (is_dataclass(value) and not isinstance(value, type))
        if container:
            if id(value) in self.active:
                raise NativeResultExportError('Native result metadata contains a cycle.')
            self.active.add(id(value))
        try:
            # Charge structural/tag overhead before allocating the encoded tree;
            # primitive sizes also account for JSON escaping below. The final
            # complete report is checked again before any CUDA transfer.
            self.charge(80 if container else 4)
            return self._encode_value(value, depth)
        finally:
            if container:
                self.active.remove(id(value))

    def _encode_value(self, value, depth):
        if type(value) is int and value.bit_length() > 256:
            raise NativeResultExportError('Native metadata integer exceeds its compact bound.')
        if value is None or type(value) in (bool, int):
            self.charge(len(str(value)))
            return value
        if type(value) is str:
            self.text_bytes += len(value.encode('utf-8'))
            if self.text_bytes > MAX_REPORT_BYTES:
                raise NativeResultExportError('Native result text exceeds its metadata bound.')
            self.charge(len(json.dumps(value, ensure_ascii=False).encode('utf-8')))
            return value
        if type(value) is float:
            self.charge(60 if not math.isfinite(value) else len(repr(value)))
            if math.isfinite(value):
                return value
            return {'kind': 'nonfinite', 'value': 'nan' if math.isnan(value)
                    else 'positive_infinity' if value > 0 else 'negative_infinity'}
        if torch.is_tensor(value) or isinstance(value, np.ndarray):
            return self.array(value)
        if isinstance(value, Mapping):
            if any(type(key) is not str for key in value):
                raise NativeResultExportError('Native output mappings require text keys.')
            return {'kind': 'mapping', 'items': [[self.encode(key, depth+1),
                self.encode(item, depth+1)] for key, item in value.items()]}
        if isinstance(value, (tuple, list)):
            return {'kind': 'tuple' if isinstance(value, tuple) else 'list',
                    'items': [self.encode(item, depth+1) for item in value]}
        if is_dataclass(value) and not isinstance(value, type):
            return {'kind': 'record', 'type': type(value).__module__+'.'+type(value).__qualname__,
                    'fields': {field.name: self.encode(getattr(value, field.name), depth+1)
                               for field in fields(value)}}
        raise NativeResultExportError(f'Unsupported native output metadata type: {type(value).__name__}.')


def _model_asset(recipe, node, collector, model_asset_paths):
    """Bind actual selected raw bytes without duplicating large numeric tables."""
    from seedvision.benchmarking import native_kmeans_recipe as kmeans
    from seedvision.benchmarking import native_naive_bayes_recipe as bayes
    factory = {'external.plantcv.kmeans_centres': kmeans,
               'external.plantcv.naive_bayes_pdf': bayes}.get(node.operation_id)
    if factory is None:
        raise NativeResultExportError('No raw model binding is defined for this output.')
    selection = recipe.parameters_for_runner(node.stage_id)['model_source']
    options = {} if node.stage_id not in model_asset_paths else {
        'resolved_path': model_asset_paths[node.stage_id]}
    model = factory.load_selected_model(selection, **options)
    identity = factory.selected_model_identity(selection, model, **options)
    if model != node.outputs['model'] or dict(identity) != dict(node.outputs['model_identity']):
        raise NativeResultExportError('Selected model differs from the model used by this run.')
    with Path(model_asset_paths.get(node.stage_id, identity['path'])).open('rb') as stream:
        raw = stream.read(identity['source_bytes']+1)
    if len(raw) != identity['source_bytes'] or sha256(raw).hexdigest() != identity['source_sha256']:
        raise NativeResultExportError('Selected model bytes changed before result export.')
    return {'kind': 'model_asset', 'identity': collector.encode(identity),
            'bytes': collector.array(np.frombuffer(raw, dtype=np.uint8), cpu_asset=True)}


def _correction_spec(port, shape, dtype, frame):
    shape, dtype = tuple(shape), np.dtype(dtype)
    valid = (shape == (*frame.shape, 3) and dtype == np.uint8) if port == 'corrected_image' else (
        dtype == np.float64 and (shape == (9, 9) if port == 'transformation_matrix' else
            len(shape) == 2 and shape[1] == 4 and 1 <= shape[0] <= 255))
    if not valid:
        raise NativeResultExportError('Correction output shape or dtype differs from its declared port.')


def _immutable_correction_matrix(value):
    if (not isinstance(value, np.ndarray) or value.ndim != 2 or value.dtype != np.float64
            or value.flags.writeable or not value.flags.c_contiguous
            or not (value.shape == (9, 9) or value.shape[1] == 4 and 1 <= value.shape[0] <= 255)):
        raise NativeResultExportError('Correction fit matrices require bounded immutable float64 tables.')
    base = value
    for _ in range(4):
        if not isinstance(base, np.ndarray):
            break
        if base.flags.writeable:
            raise NativeResultExportError('Correction fit matrix backing must be immutable.')
        base = base.base
    if not isinstance(base, bytes) or len(base) != value.nbytes or not np.isfinite(value).all():
        raise NativeResultExportError('Correction fit matrices require finite values with immutable bytes backing.')


def _correction_sources(recipe, result):
    from seedvision.benchmarking.native_correction_recipe import _validate_recipe
    _validate_recipe(recipe)
    sources = validate_correction_input_sources(result.input_sources)
    project_keys = {name for name in PROJECT_RUNTIME_KEYS if name in result.runtime}
    if (result.input_origins is None and project_keys
            or result.input_origins is not None and project_keys != set(PROJECT_RUNTIME_KEYS)
            or any(type(result.runtime[name]) is not str or re.fullmatch('[0-9a-f]{64}', result.runtime[name]) is None
                   for name in project_keys)):
        raise NativeResultExportError('Project correction producer identity requires its complete origin binding.')
    if (sources['source_rgb']['sha256'] != result.source_sha256
            or sources['target_rgb']['sha256'] != result.target_source_sha256
            or sources['source_rgb']['frame'] != result.coordinate_frame.to_dict()
            or sources['target_rgb']['frame'] != result.target_coordinate_frame.to_dict()
            or dict(result.preparation) != {name: row['decode'] for name, row in sources.items()}
            or type(result.implementation_sha256) is not str
            or re.fullmatch('[0-9a-f]{64}', result.implementation_sha256) is None
            or result.implementation_sha256 != correction_file_implementation_sha256(result.runtime)):
        raise NativeResultExportError('Correction result differs from its four input-source identities.')
    return sources


def _identity(recipe, result):
    selected = recipe.identity
    semantics = {'color_order': 'RGB', 'dtype': 'uint8', 'value_range': [0, 255],
        'channel_conversion': dict(result.preparation), 'geometric_transform': 'none',
        'interpolation': 'none'}
    if isinstance(result, NativeCorrectionResult):
        semantics['correction_inputs'] = _correction_sources(recipe, result)
        semantics['correction_implementation_sha256'] = result.implementation_sha256
        if result.input_origins is not None:
            origins = validate_correction_run_origins(recipe, result.input_origins, result.input_sources)
            semantics['correction_input_origins'] = origins.to_dict()
    return CheckpointIdentity(source_sha256=result.source_sha256,
        coordinate_frame=result.coordinate_frame, stage_id='native.analysis.result',
        recipe_id=selected.recipe_id, recipe_version=selected.recipe_version,
        recipe_sha256=selected.recipe_sha256, graph_sha256=selected.graph_sha256,
        settings_sha256=selected.settings_sha256, backend_id=selected.backend_id,
        backend_version=selected.backend_version, environment_versions=dict(result.runtime),
        image_semantics=semantics,
        mask_semantics={'foreground_polarity': 'declared_by_output_port',
            'representation': 'declared_by_output_port', 'connectivity': 'declared_by_output_port',
            'validity_region': 'coordinate_frame', 'border_policy': 'declared_by_output_port'},
        instance_semantics={'label_dtype': 'declared_by_output_port', 'background_label': 0,
            'label_id_mapping': 'declared_by_output_port',
            'split_boundary_ownership': 'declared_by_output_port',
            'visibility': 'visible_output_pixels', 'eligibility': 'not_scientifically_validated'})


def _validate_payload(payload):
    """Check semantic claims as well as bytes, before restoring any arrays."""
    expected = {'format', 'version', 'recipe', 'recipe_identity', 'source_sha256',
        'coordinate_frame', 'preparation', 'runtime', 'status', 'nodes',
        'checkpoint_identity', 'scientifically_validated', 'reference_conformance', 'checkpoint_scope'}
    if type(payload) is dict and payload.get('version') in (2, 3, 4, VERSION):
        expected.add('model_asset_paths')
    if type(payload) is dict and payload.get('version') in (3, 4, VERSION):
        expected.update(('correction_inputs', 'correction_implementation_sha256'))
    if type(payload) is dict and payload.get('version') in (4, VERSION):
        expected.add('correction_input_origins')
    if type(payload) is not dict or set(payload) != expected:
        raise NativeResultExportError('Native result payload has missing or unknown fields.')
    if (payload['format'] != FORMAT or type(payload['version']) is not int or payload['version'] not in (1, 2, 3, 4, VERSION)
            or payload['scientifically_validated'] is not False
            or payload['reference_conformance'] != 'not_evaluated_by_export'
            or payload['checkpoint_scope'] != 'whole_run_outputs; not per-stage replay certificates'):
        raise NativeResultExportError('Native archive scope or validation claims are invalid.')
    recipe = RecipeSpec.from_json(_json_bytes(payload['recipe']))
    from seedvision.benchmarking.native_recipe import _checked_model_asset_paths
    paths = payload.get('model_asset_paths', {})
    if type(paths) is not dict:
        raise NativeResultExportError('Native model transport paths must be a mapping.')
    _checked_model_asset_paths(recipe, paths)
    recorded = RecipeIdentity.from_dict(payload['recipe_identity'])
    identity = CheckpointIdentity.from_dict(payload['checkpoint_identity'])
    preparation = payload['preparation']
    runtime = payload['runtime']
    correction = recipe.recipe_id == CORRECTION_RECIPE_ID
    if (type(runtime) is not dict or not runtime or any(type(key) is not str or not key.strip()
            or type(value) is not str or not value.strip() for key, value in runtime.items())):
        raise NativeResultExportError('Native runtime identity must be a nonempty text mapping.')
    if (type(preparation) is not dict or set(preparation) != (
            {'target_rgb', 'target_mask', 'source_rgb', 'source_mask'} if correction else {'gray', 'rgb'})
            or any(type(value) is not str or not value.strip() for value in preparation.values())
            or recorded.analytical_dict() != recipe.identity.analytical_dict()):
        raise NativeResultExportError('Native preparation or selected recipe identity is inconsistent.')
    if correction:
        sources = validate_correction_input_sources(payload.get('correction_inputs'))
        from seedvision.benchmarking.checkpoints import CoordinateFrame
        origins = payload.get('correction_input_origins')
        if origins is not None:
            origins = CorrectionRunOrigins.from_dict(origins)
        result = NativeCorrectionResult(recorded, sources['target_rgb']['sha256'], payload['source_sha256'],
            CoordinateFrame.from_dict(sources['target_rgb']['frame']), identity.coordinate_frame,
            preparation, runtime, payload.get('correction_implementation_sha256'), (), payload['status'], sources, origins)
    else:
        if (payload.get('correction_inputs') is not None
                or payload.get('correction_implementation_sha256') is not None
                or payload.get('correction_input_origins') is not None):
            raise NativeResultExportError('Only correction results may contain four-file correction identities.')
        result = NativeRecipeResult(recorded, payload['source_sha256'], identity.coordinate_frame,
            preparation, runtime, (), payload['status'])
    if identity != _identity(recipe, result) or identity.coordinate_frame.to_dict() != payload['coordinate_frame']:
        raise NativeResultExportError('Native checkpoint source/frame/backend/runtime identity is inconsistent.')
    nodes, references, model_assets, model_identities = payload['nodes'], set(), [], []
    correction_arrays = {}
    if type(nodes) is not list or len(nodes) != len(recipe.stages):
        raise NativeResultExportError('Native report must contain every ordered recipe stage.')
    visits = 0

    def decode(value, depth=0, *, permit_arrays=True, permit_model=False):
        nonlocal visits
        visits += 1
        if visits > MAX_METADATA_ITEMS or depth > MAX_METADATA_DEPTH:
            raise NativeResultExportError('Native encoded output traversal exceeds its bound.')
        if value is None or type(value) in (str, bool, int, float):
            if type(value) is int and value.bit_length() > 256:
                raise NativeResultExportError('Native encoded integer exceeds its bound.')
            return value
        if type(value) is not dict:
            raise NativeResultExportError('Native output metadata must use its tagged encoding.')
        kind = value.get('kind')
        if kind == 'array' and set(value) == {'kind', 'name'} and permit_arrays:
            name = value['name']
            if type(name) is not str or re.fullmatch(r'array_[0-9]{4}', name) is None:
                raise NativeResultExportError('Invalid native output array binding.')
            references.add(name)
            return value
        if kind == 'nonfinite' and set(value) == {'kind', 'value'}:
            if value['value'] not in ('nan', 'positive_infinity', 'negative_infinity'):
                raise NativeResultExportError('Invalid nonfinite native observation.')
            return value
        if kind in ('mapping', 'tuple', 'list') and set(value) == {'kind', 'items'}:
            if type(value['items']) is not list:
                raise NativeResultExportError('Native encoded items must be an ordered list.')
            if kind == 'mapping':
                decoded = {}
                for item in value['items']:
                    if (type(item) is not list or len(item) != 2 or type(item[0]) is not str
                            or item[0] in decoded):
                        raise NativeResultExportError('Invalid native mapping key or duplicate item.')
                    decoded[item[0]] = decode(item[1], depth+1, permit_arrays=permit_arrays)
                return decoded
            items = [decode(item, depth+1, permit_arrays=permit_arrays) for item in value['items']]
            return tuple(items) if kind == 'tuple' else items
        if kind == 'record' and set(value) == {'kind', 'type', 'fields'}:
            if type(value['type']) is not str or type(value['fields']) is not dict:
                raise NativeResultExportError('Invalid native compact record.')
            return {key: decode(item, depth+1, permit_arrays=permit_arrays)
                    for key, item in value['fields'].items()}
        if kind == 'model_asset' and set(value) == {'kind', 'identity', 'bytes'} and permit_arrays and permit_model:
            selected = decode(value['identity'], depth+1, permit_arrays=False)
            array = decode(value['bytes'], depth+1)
            if type(selected) is not dict or type(array) is not dict or array.get('kind') != 'array':
                raise NativeResultExportError('Invalid native model asset binding.')
            model_assets.append((selected, array['name']))
            return value
        raise NativeResultExportError('Unknown or malformed native output metadata tag.')

    complete = True
    for stage, node in zip(recipe.stages, nodes):
        node_fields = {'stage_id', 'operation_id', 'status', 'reason',
            'reused', 'product_key', 'input_keys', 'parameters', 'outputs'}
        if payload['version'] == VERSION:
            node_fields.add('calculation')
            if isinstance(node, dict) and ('backend' in node or 'backend_provenance' in node):
                node_fields.update(('backend', 'backend_provenance'))
        if (type(node) is not dict or set(node) != node_fields
                or node['stage_id'] != stage.stage_id or node['operation_id'] != stage.operation_id
                or node['status'] not in _STATES or type(node['reused']) is not bool
                or type(node['reason']) is not str or node['reason'] != node['reason'].strip()):
            raise NativeResultExportError('Native report node differs from its exact stage contract.')
        public = node.get('backend') == 'plantcv'
        if 'backend' in node:
            if not public:
                raise NativeResultExportError('Explicit archived backend must identify a public call.')
            try:
                provenance = decode(node['backend_provenance'], permit_arrays=False)
                if provenance is not None or node['status'] in _SUCCESS:
                    _public_provenance(stage, provenance)
            except (ValueError, TypeError) as error:
                raise NativeResultExportError(str(error)) from error
        if node['status'] == 'bypassed':
            _validate_bypass_stage(stage)
        available = node['status'] in _SUCCESS
        complete &= node['status'] in ('provided', 'executed', 'empty', 'bypassed')
        if (available and (type(node['product_key']) is not str
                or re.fullmatch(r'[0-9a-f]{64}', node['product_key']) is None)
                or not available and node['product_key'] is not None
                or type(node['input_keys']) is not dict
                or any(type(key) is not str or type(value) is not str
                    or re.fullmatch(r'[0-9a-f]{64}', value) is None for key, value in node['input_keys'].items())):
            raise NativeResultExportError('Native node cache lineage is invalid.')
        if ((node['status'] in ('provided', 'executed', 'empty', 'bypassed') and node['reason'])
                or (node['status'] not in ('provided', 'executed', 'empty', 'bypassed') and not node['reason'])):
            raise NativeResultExportError('Native node availability and reason disagree.')
        selected = recipe.parameters_for_runner(stage.stage_id)
        resolved = decode(node['parameters'], permit_arrays=False)
        if type(resolved) is not dict or set(resolved) != set(selected):
            raise NativeResultExportError('Native effective parameter inventory differs from its recipe.')
        for name, value in selected.items():
            if name == 'n_labels' and value is None and stage.operation_id in ('analysis.color', 'analysis.size'):
                # Blocked nodes can retain unresolved None; executed nodes must
                # carry the positive object count calculated by the executor.
                actual = resolved[name]
                if actual is None and not available:
                    continue
                if type(actual) is not int or not (1 <= actual <= 128 if available
                        else 0 <= actual <= math.prod(identity.coordinate_frame.shape)):
                    raise NativeResultExportError('Invalid resolved native object count.')
            elif _json_bytes(resolved[name]) != _json_bytes(value):
                raise NativeResultExportError('Native effective parameters differ from the saved selection.')
        calculation = node.get('calculation')
        if stage.operation_id == 'threshold.otsu' and node['status'] in ('executed', 'empty'):
            if payload['version'] == VERSION:
                from seedvision.cuda.plantcv_thresholds import validate_otsu_calculation_facts
                from seedvision.pipeline.backend_execution import validate_public_otsu_calculation
                try:
                    (validate_public_otsu_calculation if public else validate_otsu_calculation_facts)(
                        decode(calculation, permit_arrays=False), resolved['object_type'])
                except (ValueError, TypeError) as error:
                    raise NativeResultExportError(str(error)) from error
        elif calculation is not None:
            raise NativeResultExportError('Only a completed Otsu node can record cutoff calculation facts.')
        outputs = node['outputs']
        if type(outputs) is not list or any(type(item) is not list or len(item) != 2
                or type(item[0]) is not str for item in outputs):
            raise NativeResultExportError('Native output ports require ordered named bindings.')
        ports = {item[0] for item in outputs}
        declared = {port.name for port in stage.ports if port.direction == 'output'}
        if (len(ports) != len(outputs) or ports-declared or (not available and ports)
                or (node['status'] in ('provided', 'executed', 'empty', 'bypassed') and ports != declared)):
            raise NativeResultExportError('Native archived outputs contradict their port availability.')
        for name, value in outputs:
            model_port = name == 'model' and stage.operation_id in (
                'external.plantcv.kmeans_centres', 'external.plantcv.naive_bayes_pdf')
            decoded = decode(value, permit_model=model_port)
            if correction:
                if type(decoded) is not dict or decoded.get('kind') != 'array' or decoded['name'] in correction_arrays:
                    raise NativeResultExportError('Correction outputs require distinct array bindings for every port.')
                correction_arrays[decoded['name']] = name
            if name == 'model_identity':
                model_identities.append(decoded)
            if name == 'model' and available:
                selection = selected.get('model_source')
                if stage.operation_id == 'external.plantcv.naive_bayes_pdf' and isinstance(selection, Mapping):
                    from seedvision.benchmarking import native_naive_bayes_recipe as bayes
                    bayes.validate_model_source_metadata(selection)
                    # Project-held training history stays in recipe/parameters.
                    # The raw TSV's numerical identity contains the original
                    # five source fields and remains interoperable with PlantCV.
                    selection = bayes.calculation_model_source(selection)
                if (not model_port or type(value) is not dict or value.get('kind') != 'model_asset' or not isinstance(selection, Mapping)
                        or any(_json_bytes(dict(model_assets[-1][0]).get(key)) != _json_bytes(item)
                               for key, item in selection.items())):
                    raise NativeResultExportError('Native raw model asset differs from its selected identity.')
    if payload['status'] != ('executed' if complete else 'partial'):
        raise NativeResultExportError('Native report run status contradicts its node availability.')
    if model_identities and (len(model_assets) != 1 or any(
            _json_bytes(value) != _json_bytes(model_assets[0][0]) for value in model_identities)):
        raise NativeResultExportError('Native model identity outputs differ from the archived selected model.')
    return identity, references, model_assets, correction_arrays


def export_native_recipe_result(directory, recipe, result, *,
                                limits=CheckpointLimits(), cancel=None):
    """Atomically publish ordered products and compact observations from one run.

    No project metadata or StageRunRecord is mutated. Partial runs retain only
    products actually present; missing/failed outputs stay explicitly missing.
    All groups and metadata are preflighted before the first CUDA-to-CPU copy.
    """
    _poll(cancel)
    if not isinstance(recipe, RecipeSpec) or not isinstance(result, (NativeRecipeResult, NativeCorrectionResult)):
        raise TypeError('Export requires the selected RecipeSpec and native recipe/correction result.')
    correction = isinstance(result, NativeCorrectionResult)
    if correction != (recipe.recipe_id == CORRECTION_RECIPE_ID):
        raise NativeResultExportError('Correction recipes require a four-input correction result.')
    sources = _correction_sources(recipe, result) if correction else None
    if recipe.identity.analytical_dict() != result.recipe_identity.analytical_dict():
        raise NativeResultExportError('Native result belongs to different recipe settings or topology.')
    if tuple(node.stage_id for node in result.nodes) != tuple(stage.stage_id for stage in recipe.stages):
        raise NativeResultExportError('Native result does not contain the exact ordered recipe stages.')
    collector = _Collector(result.coordinate_frame, limits, cancel)
    nodes = []
    for stage, node in zip(recipe.stages, result.nodes):
        if node.status == 'bypassed':
            _validate_bypass_stage(stage)
        if node.operation_id != stage.operation_id or node.status not in _STATES:
            raise NativeResultExportError('Native node operation or state differs from its recipe.')
        if node.status not in _SUCCESS and node.outputs:
            raise NativeResultExportError('A failed/unavailable node cannot archive output products.')
        declared = {port.name for port in stage.ports if port.direction == 'output'}
        if (set(node.outputs) - declared or
                (node.status in ('provided', 'executed', 'empty', 'bypassed') and set(node.outputs) != declared)):
            raise NativeResultExportError('Native node products differ from its declared output ports.')
        outputs = []
        for name, value in node.outputs.items():
            if correction:
                _correction_spec(name, value.shape, value.dtype if isinstance(value, np.ndarray) else
                    str(value.dtype).removeprefix('torch.'), result.coordinate_frame)
                if name == 'corrected_image':
                    if not torch.is_tensor(value):
                        raise NativeResultExportError('Corrected image must remain on CUDA until export.')
                    encoded = collector.encode(value)
                    collector.roles[encoded['name']] = RasterRole('image', 'uint8', 'HWC', (0, 255))
                else:
                    encoded = collector.array(value, cpu_matrix=True)
            else:
                encoded = _model_asset(recipe, node, collector, result.model_asset_paths or {}) if name == 'model' else collector.encode(value)
            outputs.append([name, encoded])
        nodes.append({'stage_id': node.stage_id, 'operation_id': node.operation_id,
            'status': node.status, 'reason': node.reason, 'reused': node.reused,
            'product_key': node.product_key, 'input_keys': dict(node.input_keys),
            'parameters': collector.encode(node.parameters), 'outputs': outputs,
            'calculation': collector.encode(getattr(node, 'calculation', None))})
        backend = getattr(node, 'backend', 'native')
        provenance = getattr(node, 'backend_provenance', None)
        if backend == 'plantcv':
            nodes[-1].update(backend=backend,
                backend_provenance=collector.encode(_public_provenance(stage, provenance)
                    if provenance is not None or node.status in _SUCCESS else None))
        elif backend != 'native' or provenance is not None:
            raise NativeResultExportError('Native results cannot assert public execution provenance.')
    complete = all(node.status in ('provided', 'executed', 'empty', 'bypassed') for node in result.nodes)
    if result.status != ('executed' if complete else 'partial'):
        raise NativeResultExportError('Native run status contradicts its node availability.')
    identity = _identity(recipe, result)
    payload = {'format': FORMAT, 'version': VERSION, 'recipe': recipe.to_dict(),
        'recipe_identity': result.recipe_identity.to_dict(), 'source_sha256': result.source_sha256,
        'coordinate_frame': result.coordinate_frame.to_dict(), 'preparation': dict(result.preparation),
        'runtime': dict(result.runtime), 'status': result.status, 'nodes': nodes,
        'model_asset_paths': dict(result.model_asset_paths or {}),
        'correction_inputs': sources,
        'correction_implementation_sha256': result.implementation_sha256 if correction else None,
        'correction_input_origins': result.input_origins.to_dict()
            if correction and result.input_origins is not None else None,
        'checkpoint_identity': identity.to_dict(), 'scientifically_validated': False,
        'reference_conformance': 'not_evaluated_by_export',
        'checkpoint_scope': 'whole_run_outputs; not per-stage replay certificates'}
    _validate_payload(payload)
    binding = sha256(_json_bytes(payload)).hexdigest()
    names = tuple(collector.arrays)
    groups = [names[start:start+limits.max_arrays] for start in range(0, len(names), limits.max_arrays)]
    for group in groups:
        inventory = {name: _array_descriptor(collector.arrays[name], limits=limits, name=name)
                     for name in group}
        if _estimate_npz_size(inventory) > limits.max_npz_bytes:
            raise NativeResultExportError('Native result archive group exceeds its byte bound.')
    # Reserve every eventual group address before transferring/publishing.
    _json_bytes(payload | {'groups': [{'checkpoint_sha256': '0'*64, 'arrays': list(group)}
                                    for group in groups], 'result_revision': '0'*64})
    destination = Path(directory).expanduser().absolute()
    if destination.exists():
        raise FileExistsError(f'Native result destination already exists: {destination}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.native-result-', suffix='.incomplete', dir=destination.parent))
    try:
        saved = []
        for index, group in enumerate(groups):
            _poll(cancel)
            reference = publish_checkpoint(staging/'checkpoints', identity,
                {name: collector.arrays[name] for name in group},
                execution=ExecutionRecord(ExecutionStatus.AVAILABLE),
                measurement=MeasurementContract('seedfiddle.native_output_archive', '1',
                    'mixed_declared_outputs', 'declared_by_observations', 'uncalibrated',
                    'not_estimated', False, 'Output persistence does not certify scientific eligibility.'),
                raster_roles={name: collector.roles[name] for name in group if name in collector.roles},
                metadata={'format': FORMAT, 'version': VERSION, 'run_binding_sha256': binding,
                          'group_index': index, 'arrays': list(group)},
                allow_cpu_transfer=True, limits=limits, cancel=cancel)
            saved.append({'checkpoint_sha256': reference.checkpoint_sha256, 'arrays': list(group)})
        report = payload | {'groups': saved}
        report['result_revision'] = sha256(_json_bytes(report)).hexdigest()
        if any(collector.signature(value) != signature for value, signature in collector.signatures):
            raise NativeResultExportError('Native output tensor changed during result export.')
        with (staging/'native-results.json').open('wb') as stream:
            stream.write(_json_bytes(report))
            stream.flush()
            os.fsync(stream.fileno())
        # QC describes the actual stored products, not just in-memory outputs.
        # Stream-check every array without retaining rasters; publish both files
        # in the same staging rename so a failed QC write cannot complete a run.
        from seedvision.export.method_qc import qc_from_native_archive, export_method_qc, FILENAME
        export_method_qc(qc_from_native_archive(staging, limits=limits, cancel=cancel), staging/FILENAME)
        _poll(cancel)
        if any(collector.signature(value) != signature for value, signature in collector.signatures):
            raise NativeResultExportError('Native output tensor changed before result publication.')
        staging.rename(destination)
        return destination/'native-results.json'
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _native_result_inventory(directory, limits, cancel, expected_revision):
    """Preflight the complete report and manifests; payload bytes are not read."""
    _poll(cancel)
    root = Path(directory)
    path = root/'native-results.json'
    with path.open('rb') as stream:
        raw = stream.read(MAX_REPORT_BYTES+1)
    if len(raw) > MAX_REPORT_BYTES:
        raise NativeResultExportError('Native result report exceeds its byte bound.')
    def pairs(rows):
        result = {}
        for key, value in rows:
            if key in result:
                raise NativeResultExportError('Duplicate native report field.')
            result[key] = value
        return result
    try:
        report = json.loads(raw, object_pairs_hook=pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
        revision = report.pop('result_revision')
        if expected_revision is not None and revision != expected_revision:
            raise ValueError('Native result changed since its inventory was selected.')
        if report['format'] != FORMAT or type(report['version']) is not int or report['version'] not in (1, 2, 3, 4, VERSION):
            raise ValueError('Unsupported native result format.')
        if sha256(_json_bytes(report)).hexdigest() != revision:
            raise ValueError('Native result revision differs from its report bytes.')
        groups = report.pop('groups')
        identity, references, model_assets, correction_arrays = _validate_payload(report)
        binding = sha256(_json_bytes(report)).hexdigest()
        if type(groups) is not list or len(groups) > MAX_RESULT_ARRAYS:
            raise ValueError('Native result group inventory exceeds its bound.')
        specs_by_name, total, seen = {}, 0, set()
        # Inventory ALL archives before loading any product arrays. A later
        # group cannot cause an aggregate-cap failure only after allocation.
        for group in groups:
            _poll(cancel)
            if type(group) is not dict or set(group) != {'checkpoint_sha256', 'arrays'}:
                raise ValueError('Invalid native result group schema.')
            names = group['arrays']
            if (type(names) is not list or not names or len(names) > limits.max_arrays
                    or any(type(name) is not str or name in seen for name in names)
                    or len(set(names)) != len(names)):
                raise ValueError('Invalid or duplicate native result array inventory.')
            digest = group['checkpoint_sha256']
            if (type(digest) is not str or len(digest) != 64
                    or any(char not in '0123456789abcdef' for char in digest)):
                raise ValueError('Invalid native checkpoint address.')
            checkpoint_path = root/'checkpoints'/f'{digest}.checkpoint'
            if checkpoint_path.is_symlink() or not checkpoint_path.is_dir():
                raise ValueError('Native checkpoint must be a regular directory.')
            manifest = _read_manifest(checkpoint_path/'manifest.json', limits)
            if manifest['checkpoint_sha256'] != digest:
                raise ValueError('Native checkpoint address differs from its manifest.')
            if CheckpointIdentity.from_dict(manifest['identity']) != identity:
                raise ValueError('Native checkpoint identity differs from the report.')
            specs = _validate_array_specs(manifest['arrays'], limits, identity)
            for item in specs:
                if item['name'] in correction_arrays:
                    port = correction_arrays[item['name']]
                    _correction_spec(port, item['shape'], item['dtype'], identity.coordinate_frame)
                    if port != 'corrected_image' and item['role'] != {'kind': 'compact'}:
                        raise ValueError('Compact correction matrices cannot claim raster coordinate semantics.')
                    if port == 'corrected_image' and item['role'] != RasterRole('image', 'uint8', 'HWC', (0, 255)).to_dict():
                        raise ValueError('Corrected image requires its explicit source-frame RGB raster role.')
            if {item['name'] for item in specs} != set(names):
                raise ValueError('Native checkpoint inventory differs from the report.')
            total += sum(item['nbytes'] for item in specs)
            specs_by_name.update({item['name']:item for item in specs})
            seen.update(names)
            if total > limits.max_total_array_bytes or len(seen) > MAX_RESULT_ARRAYS:
                raise ValueError('Native result exceeds its aggregate array bound.')
        if seen != references:
            raise ValueError('Native array inventory differs from the actual output references.')
        for selected, name in model_assets:
            spec = specs_by_name[name]
            if (spec['shape'] != [selected['source_bytes']] or spec['dtype'] != np.dtype(np.uint8).str
                    or spec['nbytes'] != selected['source_bytes']
                    or spec['data_sha256'] != selected['source_sha256']):
                raise ValueError('Archived native model descriptor differs from its selected source identity.')
        report = report | {'groups':groups, 'result_revision':revision}
        return report, specs_by_name, identity, binding, model_assets, correction_arrays
    except (ValueError, TypeError, KeyError, RecursionError) as error:
        raise NativeResultExportError(f'Native result verification failed: {error}') from error


def read_native_recipe_inventory(directory, *, limits=CheckpointLimits(), cancel=None):
    """Read admitted report/array descriptors without verifying payload bytes.

    Use the returned revision as expected_revision on the full loader. Inventory
    alone does not establish that any stored pixels can be loaded or replayed.
    """
    report, specs, *_ = _native_result_inventory(directory, limits, cancel, None)
    return report, specs


def load_native_recipe_result(directory, *, limits=CheckpointLimits(),
                              array_names=None, expected_revision=None, cancel=None):
    """Verify every stored product, optionally retaining only named CPU arrays.

    None preserves the all-array behavior. A tuple (including ()) selects the
    returned array map; the report and aggregate admission still cover the whole
    run. Every unselected payload is streamed and verified too. Host memory also
    includes one bounded compressed group snapshot and a selected-payload copy.
    This archive boundary does not execute recorded types/models or a stage.
    """
    report, specs, identity, binding, model_assets, correction_arrays = _native_result_inventory(
        directory, limits, cancel, expected_revision)
    root = Path(directory)
    groups = report['groups']
    if array_names is not None and (type(array_names) is not tuple
            or any(type(name) is not str for name in array_names)
            or len(set(array_names)) != len(array_names) or set(array_names)-set(specs)):
        raise NativeResultExportError('Selected arrays must be distinct names in the full native inventory.')
    requested = None if array_names is None else frozenset(array_names)
    # Compact correction fits have an additional finite-value contract; load
    # these bounded auxiliary matrices even if they are not requested, then
    # discard them after verification. Models need no auxiliary full allocation:
    # their source digest was bound to the manifest and every payload is hashed.
    auxiliary = {name for name,port in correction_arrays.items() if port != 'corrected_image'}
    arrays = {}
    try:
        for index, group in enumerate(groups):
            _poll(cancel)
            options = {}
            if requested is not None:
                options['array_names'] = tuple(name for name in group['arrays']
                                              if name in requested or name in auxiliary)
            if cancel is not None:
                options['cancel'] = cancel
            checkpoint = load_checkpoint(root/'checkpoints', group['checkpoint_sha256'],
                expected_identity=identity, limits=limits, **options)
            if (dict(checkpoint.metadata) != {'format': FORMAT, 'version': report['version'],
                    'run_binding_sha256': binding, 'group_index': index,
                    'arrays': tuple(group['arrays'])}
                    or set(checkpoint.arrays) != (set(group['arrays']) if requested is None else
                        (requested|auxiliary)&set(group['arrays']))):
                raise ValueError('Native checkpoint differs from its report binding.')
            for name, value in checkpoint.arrays.items():
                if name in arrays or len(arrays) >= MAX_RESULT_ARRAYS:
                    raise ValueError('Duplicate or excessive native output array.')
                arrays[name] = value
        for selected, name in model_assets:
            if name not in arrays:
                continue  # Full streamed digest and source binding were checked.
            array = arrays[name]
            if (array.ndim != 1 or array.dtype != np.uint8 or array.nbytes != selected['source_bytes']
                    or sha256(array).hexdigest() != selected['source_sha256']):
                raise ValueError('Archived native model bytes differ from their selected source identity.')
        for name, port in correction_arrays.items():
            if port != 'corrected_image' and not np.isfinite(arrays[name]).all():
                raise ValueError('Archived correction fit matrices must be finite.')
        from seedvision.export.method_qc import load_method_qc, _qc_from_verified_native_report, FILENAME
        quality_path = root/FILENAME
        if quality_path.exists():
            quality = load_method_qc(quality_path)
            if quality.payload['identity'] != _qc_from_verified_native_report(report).payload['identity']:
                raise ValueError('Recipe QC differs from the fully verified native archive.')
        _poll(cancel)
        return report, (arrays if requested is None else
                        {name:value for name,value in arrays.items() if name in requested})
    except CheckpointCancelled as error:
        raise NativeRecipeCancelled('Native result loading cancelled before completion.') from error
    except (ValueError, TypeError, KeyError, RecursionError) as error:
        raise NativeResultExportError(f'Native result verification failed: {error}') from error


def _validate_bypass_stage(stage):
    from seedvision.pipeline.workflows import bypass_ports
    if (stage.condition is None or stage.condition.expression != 'workspace.disabled'
            or stage.condition.outcome_when_true != 'bypass' or not bypass_ports(stage)):
        raise NativeResultExportError('Native bypass requires its registered typed working condition.')
