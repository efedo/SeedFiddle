"""Execute canonical PlantCV seed graphs with CUDA products and node-local caches.

Prepared grayscale/RGB inputs are explicit caller boundaries. This evaluator uses
the existing recipe topology and PipelineAnalysisCache; it does not manufacture
original-reference checkpoints or promote measurements to scientific validation.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, is_dataclass, replace
from hashlib import sha256
from importlib.metadata import version
from importlib import import_module
from importlib.util import resolve_name
import json
from pathlib import Path
import re
import struct
import time
from types import MappingProxyType
from types import CodeType
from typing import Mapping

import torch
import cv2

from seedvision.benchmarking.checkpoints import CoordinateFrame, _freeze_json
from seedvision.benchmarking.recipe_catalogue import build_recipe_catalogue
from seedvision.benchmarking.recipes import RecipeSpec, RecipeIdentity, ParameterContract, RecipeProvenanceMode
from seedvision.benchmarking import native_threshold_stage as _identity
from seedvision.benchmarking import color_histogram as _chart
from seedvision.benchmarking import native_extended_threshold_recipes as _extended_recipe_specs
from seedvision.benchmarking import native_naive_bayes_recipe as _naive_bayes_recipe
from seedvision.benchmarking import native_kmeans_recipe as _kmeans_recipe
from seedvision.benchmarking import native_card_recipe as _card_recipe
from seedvision.benchmarking import native_calibrated_seed_recipe as _calibrated_seed
from seedvision.benchmarking import native_core_recovery_recipe as _recovery_recipe
from seedvision.benchmarking import native_smpts_recipe as _paper_recipe
from seedvision.benchmarking import native_range_recipe as _range_recipe
from seedvision.benchmarking import native_watershed_diagnostic_recipe as _diagnostic_recipe
from seedvision.benchmarking import native_morphology_roi_recipe as _morphology_roi_recipe
from seedvision.benchmarking import native_fill_roi_recipe as _fill_roi_recipe
from seedvision.benchmarking import native_median_roi_recipe as _median_roi_recipe
from seedvision.benchmarking import native_reader_recipe as _reader_recipe
from seedvision.cuda import plantcv_watershed_diagnostic as _watershed_diagnostic
from seedvision.benchmarking import kmeans_model_artifact as _kmeans_artifact
from seedvision.cuda import plantcv_extended_thresholds as _extended_thresholds
from seedvision.cuda import (plantcv_thresholds as _threshold, plantcv_median as _median,
    plantcv_morphology as _morphology, plantcv_fill as _fill,
    plantcv_distance as _distance, plantcv_peaks as _peaks,
    plantcv_watershed as _watershed, plantcv_color as _color,
    plantcv_range as _range, plantcv_size as _size,
    plantcv_raster_spans as _raster, _nvrtc, ops as _ops)
from seedvision.cuda import plantcv_naive_bayes as _naive_bayes
from seedvision.cuda import tensor_callback as _tensor_callback
from seedvision.cuda import (plantcv_kmeans as _kmeans_masks,
    plantcv_kmeans_preprocess as _kmeans_preprocess, plantcv_kmeans_features as _kmeans_features,
    plantcv_kmeans_classifier as _kmeans_classifier, plantcv_kmeans_model as _kmeans_model,
    plantcv_kmeans_predict as _kmeans_predict)
from seedvision.segmentation.baseline import PipelineAnalysisCache
from seedvision.benchmarking import native_image_inputs as _image_inputs
from seedvision.model_io import asset_paths as _asset_paths
from seedvision.model_io import naive_bayes_provenance as _bayes_provenance


MAX_RETAINED_CUDA_BYTES = 1024*1024*1024
MAX_RASTER_PIXELS = 64_000_000
MAX_MARKER_PIXELS = 65_536
MAX_LABELS = 128
_SEED_RECIPES = ('plantcv_seed_binary', 'plantcv_seed_mean', 'plantcv_seed_otsu')
_WORKING_SEED_RECIPES = {
    'plantcv_seed_binary_working': 'plantcv_seed_binary',
    'plantcv_seed_mean_working': 'plantcv_seed_mean',
    'plantcv_seed_otsu_working': 'plantcv_seed_otsu',
}
_NAIVE_BAYES_RECIPES = ('plantcv_naive_bayes_classifier',)
_KMEANS_RECIPES = ('plantcv_kmeans_predict_masks',)


class NativeRecipeError(ValueError):
    pass


class NativePreparationMutation(NativeRecipeError,_tensor_callback.TensorCallbackMutation):
    """A preparation callback changed the selected source, buffers or identity."""


class NativeRecipeCancelled(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class NativeRecipeInputs:
    source_path: Path
    source_sha256: str
    coordinate_frame: CoordinateFrame
    gray: torch.Tensor
    rgb: torch.Tensor
    preparation: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class NativeRecipeNode:
    stage_id: str
    operation_id: str
    status: str

    outputs: Mapping[str, object]
    parameters: Mapping[str, object]
    input_keys: Mapping[str, str]
    product_key: str | None
    reused: bool = False
    reason: str = ''
    calculation: Mapping[str, object] | None = None


@dataclass(frozen=True, slots=True)
class NativeRecipeResult:
    recipe_identity: RecipeIdentity
    source_sha256: str
    coordinate_frame: CoordinateFrame
    preparation: Mapping[str, str]
    runtime: Mapping[str, str]
    nodes: tuple[NativeRecipeNode, ...]
    status: str

    # Transport metadata never changes the selected recipe or product cache keys.
    model_asset_paths: Mapping[str, str] | None = None

    def node(self, stage_id):
        return next(node for node in self.nodes if node.stage_id == stage_id)

    @property
    def labels(self):
        for node in self.nodes:
            if (node.operation_id in ('reference.internal.watershed.masked_flood', 'recovery.partition')
                    or node.stage_id == _paper_recipe.RELOCALIZE and node.operation_id == 'smpts_paper.components'):
                return node.outputs.get('labels')
        return None

    @property
    def labels_coordinate_frame(self):
        """Actual label frame; corrected seed crops have their own pixel origin."""
        if self.labels is None:
            return None
        crop = next((node for node in self.nodes if node.operation_id == 'calibrated_seed.crop'),None)
        if crop is not None and 'metadata' in crop.outputs:
            return CoordinateFrame.from_dict(dict(crop.outputs['metadata']['coordinate_frame']))
        return self.coordinate_frame


@dataclass(slots=True)
class _CacheEntry:
    key: str
    node: NativeRecipeNode
    signatures: tuple
    calculation_signature: str | None = None


def _digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        allow_nan=False).encode('utf-8')).hexdigest()


def _cancel(cancel):
    if cancel is not None and cancel():
        raise NativeRecipeCancelled('Native recipe cancelled before a complete run.')


def _tensor_signature(tensor):
    storage = tensor.untyped_storage()
    return (str(tensor.device), str(tensor.dtype), tuple(tensor.shape),
        tuple(tensor.stride()), tensor.storage_offset(), tensor.data_ptr(),
        storage.data_ptr(), storage.nbytes(), tensor._version)


def _nested_signature(path, value):
    if torch.is_tensor(value):
        return ((path, 'tensor', _tensor_signature(value)),)
    if isinstance(value, Mapping):
        keys = tuple(sorted(value, key=lambda item: (type(item).__name__, repr(item))))
        rows = [(path, 'mapping', keys)]
        for key in keys:
            rows.extend(_nested_signature(path+(('key', type(key).__name__, repr(key)),), value[key]))
        return tuple(rows)
    if isinstance(value, (tuple, list)):
        rows = [(path, type(value).__name__, len(value))]
        for index, child in enumerate(value):
            rows.extend(_nested_signature(path+(('index', index),), child))
        return tuple(rows)
    if isinstance(value, _naive_bayes.NaiveBayesModel):
        # Include the actual deeply immutable tables, so replacing fields on a
        # forged cached dataclass cannot hide behind an unchanged digest string.
        return ((path, 'model', value.source_sha256, value.tables_sha256,
            value.class_names, value.probabilities, value.source_bytes),)
    if isinstance(value, _kmeans_model.KMeansModel):
        return ((path, 'kmeans_model', value.source_sha256, value.centres_sha256,
            value.patch_size, value.channels, value.centres, value.source_bytes),)
    if value is None or type(value) in (str, bool, int, float):
        return ((path, type(value).__name__, value),)
    return ((path, type(value).__module__, type(value).__qualname__),)


def _compact_recovery_signature(outputs):
    """Stream all recovery fields without retaining per-leaf key paths."""
    digest=sha256()
    visits=0
    active=set()
    def token(kind,payload=b''):
        digest.update(kind)
        digest.update(len(payload).to_bytes(8,'big'))
        digest.update(payload)
    def visit(value,depth=0):
        nonlocal visits
        visits+=1
        if depth>32 or visits>4_194_304:
            raise NativeRecipeError('Recovery cache metadata exceeds its traversal bound.')
        if torch.is_tensor(value):
            token(b'X')
            visit((id(value),*_tensor_signature(value)),depth+1)
        elif isinstance(value,Mapping) or type(value) in (tuple,list):
            identity=id(value)
            if identity in active:
                raise NativeRecipeError('Recovery cache metadata contains a cycle.')
            active.add(identity)
            try:
                if isinstance(value,Mapping):
                    if any(type(key) is not str for key in value):
                        raise NativeRecipeError('Recovery cache metadata requires string keys.')
                    token(b'M',str(len(value)).encode('ascii'))
                    for key in sorted(value):
                        visit(key,depth+1);visit(value[key],depth+1)
                else:
                    token(b'T' if type(value) is tuple else b'L',str(len(value)).encode('ascii'))
                    for child in value:visit(child,depth+1)
            finally:
                active.remove(identity)
        elif value is None:token(b'N')
        elif type(value) is bool:token(b'B',b'1' if value else b'0')
        elif type(value) is int:
            if value.bit_length()>256:
                raise NativeRecipeError('Recovery cache integer exceeds its bound.')
            token(b'I',str(value).encode('ascii'))
        elif type(value) is float:token(b'F',value.hex().encode('ascii'))
        elif type(value) is str:
            if len(value)>65_536:
                raise NativeRecipeError('Recovery cache text exceeds its bound.')
            token(b'S',value.encode('utf-8','surrogatepass'))
        else:
            raise NativeRecipeError('Unsupported recovery cache metadata value.')
    token(b'V',b'seedfiddle.recovery-cache.v1')
    visit(outputs)
    return (('recovery-v1',digest.hexdigest()),)


def _output_signatures(outputs, *, compact_recovery=False):
    return _compact_recovery_signature(outputs) if compact_recovery else _nested_signature((), outputs)


class _CardTensorGuard:
    """Reject reentrant mutation before publishing any automatic-card result."""
    def __init__(self, gray, rgb, callback, cache, recipe_id, label='Automatic card',
                 clear_on_callback_error=False):
        self.callback = callback
        self.cache = cache
        self.prefix = 'reference.recipe:' + recipe_id + ':'
        self.label = label
        self.clear_on_callback_error = clear_on_callback_error
        self.watched = {}
        self.watch({'gray': gray, 'rgb': rgb})

    def watch(self, value):
        for tensor in _iter_nested_tensors(value):
            previous = self.watched.get(id(tensor))
            signature = _tensor_signature(tensor)
            if previous is not None and previous[1] != signature:
                self._changed()
            self.watched[id(tensor)] = (tensor, signature)

    def _clear(self):
        if isinstance(self.cache, PipelineAnalysisCache):
            for name in tuple(self.cache.values):
                if name.startswith(self.prefix):
                    del self.cache.values[name]

    def _changed(self):
        self._clear()
        raise NativeRecipeError(self.label+' source or reusable product changed during execution.')

    def __call__(self):
        try:
            return self.callback() if self.callback is not None else False
        except BaseException:
            if self.clear_on_callback_error:
                self._clear()
            raise
        finally:
            for tensor, expected in self.watched.values():
                if _tensor_signature(tensor) != expected:
                    self._changed()


def _evict_recipe_descendants(recipe, stage_id, cache, prefix):
    dirty = {stage_id}
    while True:
        expanded = dirty | {edge.target_stage for edge in recipe.dependencies if edge.source_stage in dirty}
        if expanded == dirty:
            break
        dirty = expanded
    for dependent in dirty:
        cache.values.pop(prefix + dependent, None)


def _iter_nested_tensors(value):
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, Mapping):
        for child in value.values():
            yield from _iter_nested_tensors(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            yield from _iter_nested_tensors(child)


def _nested_cuda_storage(outputs):
    storage = {}
    for tensor in _iter_nested_tensors(outputs):
        if tensor.device.type == 'cuda':
            allocation = tensor.untyped_storage()
            storage[(str(tensor.device), allocation.data_ptr())] = allocation.nbytes()
    return storage


def _clear_naive_bayes_cache(cache):
    prefix = f'reference.recipe:{_naive_bayes_recipe.NAIVE_BAYES_RECIPE_ID}:'
    for stage in (_naive_bayes_recipe.MODEL_STAGE_ID, _naive_bayes_recipe.CLASSIFIER_STAGE_ID):
        cache.values.pop(prefix+stage, None)


def _resolve_naive_bayes_binding(recipe, cache, model_asset_paths=None):
    try:
        source = recipe.parameters_for_runner(_naive_bayes_recipe.MODEL_STAGE_ID)[
            _naive_bayes_recipe.MODEL_PARAMETER_ID]
        paths = {} if model_asset_paths is None else model_asset_paths
        options = {} if _naive_bayes_recipe.MODEL_STAGE_ID not in paths else {
            'resolved_path': paths[_naive_bayes_recipe.MODEL_STAGE_ID]}
        model = _naive_bayes_recipe.load_selected_model(source, **options)
        identity = _naive_bayes_recipe.selected_model_identity(source, model, **options)
        return model, identity
    except Exception as error:
        _clear_naive_bayes_cache(cache)
        raise NativeRecipeError(
            f'Naive Bayes model selection is unavailable: {type(error).__name__}: {error}') from error


def _clear_kmeans_cache(cache):
    prefix = f'reference.recipe:{_kmeans_recipe.KMEANS_RECIPE_ID}:'
    for stage in (_kmeans_recipe.MODEL_STAGE_ID, _kmeans_recipe.PREDICT_STAGE_ID,
                  _kmeans_recipe.MASK_STAGE_ID):
        cache.values.pop(prefix+stage, None)


def _resolve_kmeans_binding(recipe, cache, model_asset_paths=None):
    try:
        source = recipe.parameters_for_runner(_kmeans_recipe.MODEL_STAGE_ID)[
            _kmeans_recipe.MODEL_PARAMETER_ID]
        paths = {} if model_asset_paths is None else model_asset_paths
        options = {} if _kmeans_recipe.MODEL_STAGE_ID not in paths else {
            'resolved_path': paths[_kmeans_recipe.MODEL_STAGE_ID]}
        model = _kmeans_recipe.load_selected_model(source, **options)
        identity = _kmeans_recipe.selected_model_identity(source, model, **options)
        return model, identity
    except Exception as error:
        _clear_kmeans_cache(cache)
        raise NativeRecipeError(
            f'K-means model selection is unavailable: {type(error).__name__}: {error}') from error


def _retained_recipe_bound(pixels, model=None, categories=None):
    bound = 30*pixels + 16*min(pixels, _peaks.MAX_CANDIDATES)
    if pixels < 1 or pixels > MAX_RASTER_PIXELS:
        raise NativeRecipeError('Native recipe exceeds its raster limit.')
    if model is None:
        # Nonclassifier working recipes share this conservative preparation
        # envelope, including the new retained size RGB copy and paint scratch.
        bound += 3*pixels + _size.MAX_SIZE_DIAGNOSTIC_SCRATCH_BYTES
    if isinstance(model, _naive_bayes.NaiveBayesModel):
        output_bytes = len(model.class_names)*pixels
        table_bytes = len(model.class_names)*3*256*8
        scratch_bytes = _naive_bayes.SCRATCH_ENVELOPE_BYTES_PER_PIXEL*min(pixels, _naive_bayes.TILE_PIXELS)
        if (pixels > _naive_bayes.MAX_PIXELS or output_bytes > _naive_bayes.MAX_OUTPUT_BYTES
                or scratch_bytes > _naive_bayes.MAX_SCRATCH_BYTES
                or output_bytes+table_bytes+scratch_bytes > _naive_bayes.MAX_CUDA_LIVE_BYTES):
            raise NativeRecipeError('Selected classifier exceeds its raster, masks or scratch budget.')
        bound += output_bytes+table_bytes+scratch_bytes
    elif isinstance(model, _kmeans_model.KMeansModel):
        if categories is not None and (not isinstance(categories, (tuple, list))
                or not 1 <= len(categories) <= _kmeans_masks.MAX_CAT_LIST
                or any(type(value) is not int for value in categories)):
            raise NativeRecipeError('K-means categories must be null or a nonempty bounded integer list.')
        output_bytes = pixels * (len(model.centres) if categories is None else 1)
        if pixels > _kmeans_masks.MAX_PIXELS or output_bytes > _kmeans_masks.MAX_OUTPUT_BYTES:
            raise NativeRecipeError('Selected K-means masks exceed their raster/output budget.')
        # Conservative session envelope also covers retained old masks while a
        # changed predictor runs. The predictor preflights geometry/work before
        # Gaussian allocation; no full padded image is made by this executor.
        bound += (output_bytes + pixels + _kmeans_preprocess.MAX_OUTPUT_BYTES
                  + _kmeans_predict.MAX_SCRATCH_BYTES + _kmeans_masks.MAX_SCRATCH_BYTES)
    if bound > MAX_RETAINED_CUDA_BYTES:
        raise NativeRecipeError('Native recipe exceeds its retained-product byte budget.')
    return bound


def _code_constant(value):
    if isinstance(value, CodeType):
        return _code_record(value)
    if isinstance(value, bytes):
        return {'bytes': value.hex()}
    if isinstance(value, tuple):
        return {'tuple': [_code_constant(item) for item in value]}
    if isinstance(value, frozenset):
        return {'frozenset': sorted((_code_constant(item) for item in value), key=repr)}
    if isinstance(value, slice):
        return {'slice': [_code_constant(item) for item in (value.start, value.stop, value.step)]}
    if value is Ellipsis:
        return {'ellipsis': True}
    if isinstance(value, complex):
        return {'complex': [value.real.hex(), value.imag.hex()]}
    if isinstance(value, float):
        return {'float': value.hex()}
    if value is None or type(value) in (bool, int, str):
        return value
    raise NativeRecipeError(f'Unsupported loaded-code constant type: {type(value).__name__}.')


def _code_record(code):
    # Explicit fields avoid marshal's interning/reference flags changing after
    # a function has run. CUDA slice/Ellipsis constants remain fully represented.
    fields = ('co_argcount', 'co_posonlyargcount', 'co_kwonlyargcount', 'co_nlocals',
        'co_stacksize', 'co_flags', 'co_names', 'co_varnames', 'co_freevars',
        'co_cellvars', 'co_filename', 'co_name', 'co_qualname', 'co_firstlineno')
    return {**{name: getattr(code, name) for name in fields},
        'co_code': code.co_code.hex(), 'co_linetable': code.co_linetable.hex(),
        'co_exceptiontable': code.co_exceptiontable.hex(),
        'co_consts': [_code_constant(value) for value in code.co_consts]}


def _module_record(module, *, only=None):
    """Reuse loaded-code/source verification, including all defined helpers."""
    path = Path(module.__file__).resolve()
    source = _identity._read_source_bounded(path)
    tree = ast.parse(source, filename=str(path))
    namespace = dict(vars(module))
    names, constants, dataclasses = [], set(), []
    for statement in tree.body:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if only is None or statement.name in only:
                names.append(statement.name)
        elif isinstance(statement, ast.ClassDef) and only is None:
            cls = namespace[statement.name]
            if is_dataclass(cls):
                dataclasses.append(statement.name)
            for member in statement.body:
                if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    name = f'{statement.name}.{member.name}'
                    namespace[name] = getattr(cls, member.name)
                    names.append(name)
        elif isinstance(statement, (ast.Assign, ast.AnnAssign)) and only is None:
            targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
            for target in targets:
                if isinstance(target, ast.Name) and target.id.upper() == target.id:
                    try:
                        _identity._normalize_identity_value(_identity._literal_node(statement.value, namespace))
                    except (ValueError, TypeError):
                        continue  # The range module's computed tables are checked below.
                    constants.add(target.id)
    # Code objects also contain Ellipsis and slice constants in CUDA indexing.
    # Python compares their full executable structure without restricting those
    # constants to the older threshold recorder's scalar-only inventory.
    fresh_codes = {}
    def visit(code):
        fresh_codes[code.co_qualname] = code
        for child in code.co_consts:
            if isinstance(child, CodeType):
                visit(child)
    visit(compile(source, str(path), 'exec', dont_inherit=True))
    definitions = _identity._source_callable_nodes(tree)
    loaded_records = []
    for name in names:
        function = namespace[name]
        function = getattr(function, '__wrapped__', function)
        if isinstance(function, property):
            function = function.fget
        code = getattr(function, '__code__', None)
        fresh = fresh_codes.get(name)
        if (code is None or fresh is None or code != fresh
                or code.co_filename != fresh.co_filename or code.co_firstlineno != fresh.co_firstlineno
                or code.co_linetable != fresh.co_linetable or code.co_exceptiontable != fresh.co_exceptiontable):
            raise NativeRecipeError(f'Loaded calculation {path.name}:{name} differs from current source.')
        defaults = _identity._verify_function_defaults(name, function, definitions[name], namespace)
        loaded_records.append({'name': name, 'code_sha256': _digest(_code_record(code)),
            'defaults_sha256': _digest(defaults)})
    record = _identity._module_source_record(path.name, path, namespace, (),
        frozenset(constants), dataclass_names=tuple(dataclasses))
    record['loaded_code'] = loaded_records
    aliases = []
    for statement in tree.body:
        if not isinstance(statement, ast.ImportFrom):
            continue
        imported_name = ('.'*statement.level)+(statement.module or '')
        qualified = resolve_name(imported_name, module.__package__) if statement.level else imported_name
        if not qualified.startswith('seedvision.'):
            continue
        imported_module = import_module(qualified)
        for alias in statement.names:
            if alias.name == '*':
                raise NativeRecipeError('Wildcard calculation imports cannot be verified.')
            local = alias.asname or alias.name
            if namespace[local] is not getattr(imported_module, alias.name):
                raise NativeRecipeError(f'Loaded calculation alias {path.name}:{local} differs from its module.')
            aliases.append({'name': local, 'module': qualified, 'attribute': alias.name})
    record['imported_aliases'] = aliases
    if module is _range:
        # Only compact source-derived constants are regenerated on CPU. The
        # source-defined generators above have already passed loaded-code checks.
        assignment = next(statement for statement in tree.body if isinstance(statement, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == '_OPENCV_CBRT_COEFFICIENTS'
                for target in statement.targets))
        raw_bits = _identity._literal_node(assignment.value.args[0].generators[0].iter, namespace)
        coefficients = tuple(struct.unpack('=d', struct.pack('=Q', int(raw, 16)))[0] for raw in raw_bits)
        if namespace['_OPENCV_CBRT_COEFFICIENTS'] != coefficients:
            raise NativeRecipeError('Loaded OpenCV cube-root constants differ from current source.')
        expected = {'_LAB_GAMMA_VALUES': module._lab_gamma_table(),
            '_LAB_CBRT_VALUES': module._lab_cbrt_table()}
        for name, values in expected.items():
            if namespace[name] != values:
                raise NativeRecipeError(f'Loaded calculation table {name} differs from current source.')
        record['computed_constants'] = _digest({'coefficients': coefficients, **expected})
    return record


def _verify_cached_lab_tables(device):
    actual = _range._lab_tables(device)
    expected = (_range._LAB_GAMMA_VALUES, _range._LAB_CBRT_VALUES)
    if not isinstance(actual, tuple) or len(actual) != 2:
        raise NativeRecipeError('Cached LAB conversion tables have an invalid structure.')
    for tensor, values in zip(actual, expected):
        if (not torch.is_tensor(tensor) or tensor.device != device or tensor.dtype != torch.int32
                or tuple(tensor.shape) != (len(values),) or not tensor.is_contiguous()
                or not torch.equal(tensor, torch.tensor(values, dtype=torch.int32, device=device))):
            raise NativeRecipeError('Cached LAB conversion table differs from its verified source values.')


def _runtime_records(device, *, card=False, calibrated=False, recovery=False, diagnostic=False, paper=False):
    modules = (_threshold, _median, _morphology, _fill, _distance, _peaks,
        _watershed, _color, _range, _size, _raster, _nvrtc, _chart,
        _extended_thresholds, _extended_recipe_specs, _range_recipe, _naive_bayes, _naive_bayes_recipe,
        _kmeans_recipe, _kmeans_masks, _kmeans_preprocess, _kmeans_features,
        _kmeans_classifier, _kmeans_model, _kmeans_predict, _kmeans_artifact, _asset_paths,
        _bayes_provenance, _tensor_callback, _image_inputs, _morphology_roi_recipe, _fill_roi_recipe,
        _median_roi_recipe, _reader_recipe)
    records = {module.__name__: _digest(_module_record(module)) for module in modules}
    if diagnostic:
        records.update({module.__name__: _digest(_module_record(module)) for module in
            (_diagnostic_recipe, _watershed_diagnostic, _watershed_diagnostic.contours)})
        records['watershed_palette_runtime'] = _digest(_watershed_diagnostic.palette_runtime_identity())
    if card:
        records.update({module.__name__: _digest(_module_record(module))
            for module in _card_recipe.implementation_modules()})
        records['automatic_card_metadata'] = _digest(_card_recipe.metadata_identity())
    if calibrated:
        records.update({module.__name__: _digest(_module_record(module))
            for module in _calibrated_seed.implementation_modules()})
    if recovery:
        records.update({module.__name__: _digest(_module_record(module))
            for module in (_recovery_recipe, _recovery_recipe.core)})
        for module, path, name in (
            (_recovery_recipe, Path(_recovery_recipe.__file__).parents[1]/'cuda/_core_recovery_graph.cu', 'core_graph_kernel'),
            (_recovery_recipe.core, Path(_recovery_recipe.core.__file__).with_name('_core_recovery.cu'), 'core_recovery_kernel')):
            kernel_source=path.read_text(encoding='utf-8')
            if kernel_source != module._CUDA_SOURCE:
                raise NativeRecipeError('Loaded recovery kernel differs from its current source.')
            records[name]=sha256(kernel_source.encode('utf-8')).hexdigest()
    if paper:
        records.update({module.__name__: _digest(_module_record(module))
            for module in (_paper_recipe, _paper_recipe.paper)})
        kernel_source = Path(_paper_recipe.paper.__file__).with_name('_smpts_paper.cu').read_text(encoding='utf-8')
        if kernel_source != _paper_recipe.paper._CUDA_SOURCE:
            raise NativeRecipeError('Loaded paper kernel differs from its current source.')
        records['smpts_paper_kernel'] = sha256(kernel_source.encode('utf-8')).hexdigest()
    _verify_cached_lab_tables(device)
    records[_ops.__name__] = _digest(_module_record(_ops, only=('connected_components',)))
    kernel = Path(_watershed.__file__).with_name('_plantcv_watershed.cu').read_text(encoding='utf-8')
    if kernel != _watershed._CUDA_SOURCE:
        raise NativeRecipeError('Loaded watershed kernel differs from its current source.')
    records['watershed_kernel'] = sha256(kernel.encode('utf-8')).hexdigest()
    runtime = {'torch': str(torch.__version__), 'torch_cuda': str(torch.version.cuda),
        'device': str(device), 'device_name': torch.cuda.get_device_name(device),
        'numpy': version('numpy'), 'opencv-python': version('opencv-python'),
        'executor': _digest(_module_record(__import__(__name__, fromlist=['_cancel'])))}
    runtime.update(_nvrtc.runtime_versions())
    # Persist the complete calculation closure, not just the executor module.
    # Old archives remain readable but cannot assert this exact replay binding.
    runtime['calculation_modules_sha256'] = _digest(records)
    return records, runtime


def build_native_seed_recipe(recipe):
    """Copy selected historical seed settings into an additive native working graph."""
    if not isinstance(recipe, RecipeSpec) or recipe.recipe_id not in _SEED_RECIPES:
        raise NativeRecipeError('Select a historical PlantCV seed recipe to make a working copy.')
    _validate_recipe(recipe)
    stages = tuple(replace(stage, parameters=stage.parameters + (
        ParameterContract('line_thickness', 'integer', 5, True, editable=True,
            minimum=1, maximum=32767, units='pixels',
            description='Integer stroke width; also affects the source longest-path measurement.'),
    )) if stage.operation_id == 'analysis.size' else stage for stage in recipe.stages)
    provenance = replace(recipe.provenance, mode=RecipeProvenanceMode.NATIVE_COMPATIBLE,
        backend_id='seedvision.benchmarking.native_recipe', backend_version='1.1.0',
        evidence_note='Native working seed graph with explicit size line thickness. '
            'Supported domains are declared separately; no biological validation is implied.')
    return replace(recipe, recipe_id=recipe.recipe_id+'_working', recipe_version='1.1.0',
        provenance=provenance, stages=stages)


def _validate_recipe(recipe):
    if not isinstance(recipe, RecipeSpec):
        raise NativeRecipeError('Select a supported PlantCV RecipeSpec.')
    if recipe.recipe_id in _SEED_RECIPES:
        canonical = build_recipe_catalogue().recipe(recipe.recipe_id)
    elif recipe.recipe_id in _WORKING_SEED_RECIPES:
        canonical = build_native_seed_recipe(build_recipe_catalogue().recipe(
            _WORKING_SEED_RECIPES[recipe.recipe_id]))
    elif recipe.recipe_id in _diagnostic_recipe.RECIPE_IDS:
        canonical = _diagnostic_recipe.build_native_watershed_diagnostic_recipe(
            _diagnostic_recipe.canonical_base(recipe.recipe_id))
    elif recipe.recipe_id in _morphology_roi_recipe.RECIPE_IDS:
        canonical = _morphology_roi_recipe.build_native_morphology_roi_recipe(
            _morphology_roi_recipe.canonical_base(recipe.recipe_id))
    elif recipe.recipe_id in _fill_roi_recipe.RECIPE_IDS:
        canonical = _fill_roi_recipe.build_native_fill_roi_recipe(
            _fill_roi_recipe.canonical_base(recipe.recipe_id))
    elif recipe.recipe_id in _NAIVE_BAYES_RECIPES:
        canonical = _naive_bayes_recipe.build_native_naive_bayes_recipe()
    elif recipe.recipe_id in _KMEANS_RECIPES:
        canonical = _kmeans_recipe.build_native_kmeans_recipe()
    elif recipe.recipe_id == _card_recipe.RECIPE_ID:
        canonical = _card_recipe.build_native_card_recipe()
    elif recipe.recipe_id in _calibrated_seed.RECIPE_IDS:
        canonical = _calibrated_seed.build_native_calibrated_seed_recipe(recipe.recipe_id.rsplit('_',1)[1])
    elif recipe.recipe_id == _recovery_recipe.RECIPE_ID:
        canonical = _recovery_recipe.build_native_core_recovery_recipe()
    elif recipe.recipe_id == _paper_recipe.RECIPE_ID:
        canonical = _paper_recipe.build_native_smpts_recipe()
    elif recipe.recipe_id == _range_recipe.RECIPE_ID:
        canonical = _range_recipe.build_native_range_recipe()
    else:
        canonical = _extended_recipe_specs.build_native_extended_threshold_recipes().get(recipe.recipe_id)
        if canonical is None:
            raise NativeRecipeError('Select a canonical seed, extended threshold or frozen-model classifier recipe.')
    if _reader_recipe.is_reader_recipe(recipe):
        if recipe.recipe_version == _reader_recipe.MEDIAN_RECIPE_VERSION:
            canonical = _median_roi_recipe._upgrade(canonical)
        canonical = _reader_recipe._upgrade(canonical)
    elif _median_roi_recipe.is_median_roi_recipe(recipe):
        canonical = _median_roi_recipe._upgrade(canonical)
    actual = recipe.to_dict()
    expected = canonical.to_dict()
    # Layout/presentation and selected parameter values do not redefine a graph.
    for document in (actual, expected):
        document.pop('layout')
        document.pop('title')
        document.pop('description')
        for stage in document['stages']:
            stage.pop('title')
            stage.pop('description')
            for port in stage['ports']:
                port.pop('label')
            for parameter in stage['parameters']:
                parameter.pop('value')
    if actual != expected:
        raise NativeRecipeError('Selected recipe differs from canonical source, topology or parameter contracts.')


def _topological_stages(recipe):
    remaining = {stage.stage_id: stage for stage in recipe.stages}
    completed = set()
    while remaining:
        ready = [stage for stage in remaining.values() if all(
            edge.source_stage in completed for edge in recipe.dependencies if edge.target_stage == stage.stage_id)]
        if not ready:
            raise NativeRecipeError('Recipe contains an unresolved dependency cycle.')
        for stage in ready:
            yield stage
            completed.add(stage.stage_id)
            del remaining[stage.stage_id]


def _required_parameters(parameters, *, dynamic_count=False, optional=()):
    missing = [name for name, value in parameters.items()
        if value is None and name not in optional and not (dynamic_count and name == 'n_labels')]
    if missing:
        raise NativeRecipeError('Required parameter values are unselected: '+', '.join(missing))


def _model_dispatch(operation, inputs, parameters, cancel):
    """Guard model contents, source pixels, adapters and nested returned masks."""
    original = _nested_signature((), inputs)
    model = inputs.get('model')
    model_fields = (('probabilities', 'class_names') if isinstance(model, _naive_bayes.NaiveBayesModel)
                    else ('centres',) if isinstance(model, _kmeans_model.KMeansModel) else ())
    field_ids = tuple(id(getattr(model, name)) for name in model_fields)
    watched, returned = [], []

    def intact():
        if (_nested_signature((), inputs) != original
                or tuple(id(getattr(model, name)) for name in model_fields) != field_ids
                or any(_tensor_signature(value) != signature for value, signature in watched)
                or any((_nested_signature((), value), tuple(value.get('class_masks', ()))) != signature
                       for value, signature in returned)):
            raise NativeRecipeError('Model calculation input, adapter or output changed during execution.')

    def poll():
        intact()
        try:
            stopped = cancel() if cancel is not None else False
            if type(stopped) is not bool:
                raise NativeRecipeError('Model cancellation callback must return a bool.')
            return stopped
        finally:
            intact()

    if poll():
        raise NativeRecipeCancelled('Model calculation cancelled before completion.')
    if operation == 'analysis.naive_bayes_classifier':
        bgr = inputs['rgb_image'].flip(-1).contiguous()
        watched.append((bgr, _tensor_signature(bgr)))
        result = _naive_bayes.naive_bayes_masks(bgr, model, cancel=poll)
        outputs = {'class_masks': MappingProxyType(dict(result.masks)),
                   'class_names': tuple(result.class_names), 'model_identity': inputs['model_identity']}
    elif operation == 'analysis.kmeans_predict':
        image = inputs['gray_image'] if model.channels == 1 else inputs['rgb_image'].flip(-1).contiguous()
        watched.append((image, _tensor_signature(image)))
        outputs = {'labels': _kmeans_predict.predict_kmeans(image, model, cancel=poll),
                   'model_identity': inputs['model_identity']}
    else:
        masks = _kmeans_masks.mask_kmeans(inputs['labels'], inputs['model_identity']['cluster_count'],
                                         parameters['cat_list'], cancel=poll)
        if torch.is_tensor(masks):
            masks = {'selected': masks}
        outputs = {'class_masks': MappingProxyType(dict(masks)), 'class_names': tuple(masks),
                   'model_identity': inputs['model_identity']}
    returned.append((outputs, (_nested_signature((), outputs), tuple(outputs.get('class_masks', ())))))
    if poll():
        raise NativeRecipeCancelled('Model calculation cancelled before completion.')
    intact()
    return outputs


class _CalculatedOutputs(dict):
    """Keep compact calculation facts separate from declared recipe ports."""
    def __init__(self, values, calculation):
        super().__init__(values)
        self.calculation = calculation


def _dispatch_calculation(operation, outputs, parameters):
    if operation != 'threshold.otsu':
        return None
    try:
        facts = _threshold.validate_otsu_calculation_facts(
            getattr(outputs, 'calculation', None), parameters['object_type'])
    except (ValueError, TypeError) as error:
        raise NativeRecipeError(str(error)) from error
    return _freeze_json(facts, name='native Otsu calculation facts')


def _calculation_signature(node):
    if node.calculation is None:
        return 'unrecorded' if node.operation_id == 'threshold.otsu' and node.status in ('executed', 'empty') else None
    if node.operation_id != 'threshold.otsu' or node.status not in ('executed', 'empty'):
        return 'invalid'
    try:
        return _digest(_threshold.validate_otsu_calculation_facts(
            node.calculation, node.parameters['object_type']))
    except (ValueError, TypeError, KeyError):
        return 'invalid'


def _dispatch(operation, inputs, parameters, cancel):
    if operation == _diagnostic_recipe.OPERATION:
        try:
            result = _watershed_diagnostic.render_watershed_diagnostic_rgb(
                inputs['image'], inputs['labels'], **parameters, cancel=cancel)
        except (_watershed_diagnostic.WatershedDiagnosticResourceError,
                _watershed_diagnostic.WatershedDiagnosticUnsupported) as error:
            return {'metadata': _freeze_json({'status': 'unavailable', 'reason': str(error),
                'kind': 'native_watershed_debug_diagnostic', 'public_return': 'labels only'},
                name='watershed diagnostic unavailability')}
        return {'diagnostic': result.image, 'metadata': _freeze_json(dict(result.metadata),
            name='watershed diagnostic provenance')}
    if operation.startswith('recovery.'):
        return _recovery_recipe.execute_stage(operation, inputs, parameters, cancel)
    if operation.startswith('smpts_paper.'):
        return _paper_recipe.execute_stage(operation, inputs, parameters, cancel)
    if operation.startswith('calibrated_seed.'):
        return _calibrated_seed.execute_stage(operation, inputs, parameters, cancel)
    if operation.startswith('color_card.'):
        return _card_recipe.execute_card_stage(operation, inputs, parameters, cancel)
    if operation == 'filter.median_blur':
        kernel = parameters['ksize']
        if isinstance(kernel, list):
            kernel = tuple(kernel)  # JSON representation of the public tuple kernel.
        roi = _median_roi_recipe.selected_bounds(parameters.get('roi_bounds_xyxy'), inputs['gray'].shape)
        return {'image': _median.median_blur(inputs['gray'], kernel, roi=roi, cancel=cancel)}
    if operation == 'threshold.otsu':
        result = _threshold.threshold_otsu_result(inputs['gray'], **parameters)
        return _CalculatedOutputs({'mask': result.mask}, _threshold.otsu_calculation_facts(
            result.native_selected_cutoff, result.object_type))
    if operation in ('threshold.binary', 'threshold.mean'):
        function = {'threshold.binary': _threshold.threshold_binary,
            'threshold.mean': _threshold.threshold_mean}[operation]
        return {'mask': function(inputs['gray'], **parameters)}
    if operation == 'threshold.gaussian':
        return {'mask': _extended_thresholds.threshold_gaussian(inputs['gray'], **parameters, cancel=cancel)}
    if operation == 'threshold.triangle':
        return {'mask': _extended_thresholds.threshold_triangle(inputs['gray'], **parameters, cancel=cancel)}
    if operation == 'threshold.texture':
        return {'mask': _extended_thresholds.threshold_texture(inputs['gray'], **parameters, cancel=cancel)}
    if operation == 'threshold.dual_channels':
        bgr = inputs['rgb'].flip(-1).contiguous()
        return {'mask': _extended_thresholds.threshold_dual_channels(bgr, **parameters, cancel=cancel)}
    if operation == _range_recipe.OPERATION:
        try:
            return _range_recipe.execute_stage(inputs, parameters, cancel)
        except _range.RangeResourceError as error:
            raise NotImplementedError(str(error)) from error
    if operation in ('analysis.naive_bayes_classifier', 'analysis.kmeans_predict', 'analysis.kmeans_masks'):
        return _model_dispatch(operation, inputs, parameters, cancel)
    if operation == 'morphology.fill':
        roi = _fill_roi_recipe.selected_bounds(parameters.get('roi_bounds_xyxy'), inputs['mask'].shape)
        return {'mask': _fill.fill(inputs['mask'], parameters['size'], roi=roi, cancel=cancel)}
    if operation in ('morphology.erode', 'morphology.dilate'):
        function = _morphology.erode if operation.endswith('erode') else _morphology.dilate
        selected = dict(parameters)
        roi = _morphology_roi_recipe.selected_bounds(selected.pop('roi_bounds_xyxy', None), inputs['gray'].shape)
        return {'mask': function(inputs['gray'], **selected, roi=roi, cancel=cancel)}
    if operation == 'reference.internal.watershed.distance_map':
        return {'distance': _distance.distance_map(inputs['mask'], cancel=cancel)}
    if operation == 'reference.internal.watershed.local_maxima':
        coordinates, peaks = _peaks.local_maxima(inputs['distance'], inputs['mask'],
            parameters['distance'], cancel=cancel)
        return {'coordinates': coordinates, 'peaks': peaks}
    if operation == 'reference.internal.watershed.connected_markers':
        if int(torch.count_nonzero(inputs['peaks']).item()) > MAX_MARKER_PIXELS:
            raise NativeRecipeError('Connected-marker compact statistics exceed their point limit.')
        markers, _ = _ops.connected_components(inputs['peaks'])
        return {'markers': markers.reshape(inputs['peaks'].shape).contiguous()}
    if operation == 'reference.internal.watershed.negate_distance':
        return {'negative_distance': -inputs['distance']}
    if operation == 'reference.internal.watershed.masked_flood':
        return {'labels': _watershed.masked_watershed(inputs['negative_distance'],
            inputs['markers'], inputs['mask'], cancel=cancel), 'flood_complete': True}
    if operation == 'reference.internal.watershed.marker_count_observation':
        count = int(torch.unique(inputs['markers']).numel())-1
        return {'marker_count': _freeze_json({'sample': parameters['label'], 'variable': 'estimated_object_count',
            'trait': 'estimated object count', 'method': 'plantcv.plantcv.watershed',
            'scale': 'none', 'datatype': "<class 'int'>", 'value': count}, name='marker count observation')}
    if operation == 'analysis.color':
        bgr = inputs['rgb'].flip(-1).contiguous()
        guard = _CardTensorGuard(inputs['labels'], bgr, None, None, 'analysis.color',
            label='Colour analysis')
        guard.watch(inputs['rgb'])
        def color_poll():
            guard()
            try:
                cancelled = cancel() if cancel is not None else False
                if type(cancelled) is not bool:
                    raise NativeRecipeError('Colour cancellation callback must return a bool.')
                return cancelled
            finally:
                guard()
        result = _color.analyze_color_cuda(bgr, inputs['labels'],
            n_labels=parameters['n_labels'], colorspaces=parameters['colorspaces'],
            label=parameters['label'], cancel=color_poll)
        _color._cancel(color_poll)
        if result.status != 'available':
            raise NativeRecipeError(f'Original colour failure: {result.error_type}: {result.error}')
        return {'observations': result, 'histogram': _chart.hue_histogram_chart(result)}
    if operation == 'analysis.size':
        guard = _CardTensorGuard(inputs['labels'], inputs['image'], None, None, 'analysis.size',
            label='Size analysis')
        mask_guard = None
        def size_poll():
            guard()
            if mask_guard is not None:mask_guard()
            try:
                cancelled=cancel() if cancel is not None else False
                if type(cancelled) is not bool:
                    raise NativeRecipeError('Size cancellation callback must return a bool.')
                return cancelled
            finally:
                guard()
                if mask_guard is not None:mask_guard()
        line_thickness = _size._validate_line_thickness(parameters.get('line_thickness', 5))
        labels = parameters['label']
        if isinstance(labels, str):
            labels = (labels,)*parameters['n_labels']
        if (not isinstance(labels, (tuple, list)) or len(labels) != parameters['n_labels']
                or any(not isinstance(value, str) for value in labels)):
            raise NativeRecipeError('Size sample labels must match the selected object count.')
        observations = {}
        label_results = []
        diagnostic_available = True
        diagnostic_points = diagnostic_boundary_points = diagnostic_contours = 0
        for label_id in range(1, parameters['n_labels']+1):
            _cancel(size_poll)
            mask=(inputs['labels'] == label_id).to(torch.uint8)*255
            mask_guard=_CardTensorGuard(mask, None, None, None, 'analysis.size',label='Size private mask')
            try:
                result = _size.premeasure_plantcv_binary_mask(mask, cancel=size_poll,
                    line_thickness=line_thickness)
                _cancel(size_poll)
            finally:
                mask_guard=None
                del mask
            if not result.plantcv_bundle_available:
                raise NotImplementedError(result.plantcv_bundle_unavailable_reason)
            if diagnostic_available:
                try:
                    points = _size._preflight_size_diagnostic_points((result,), cancel=size_poll)
                    topology = result.boundary_topology
                    records = topology.xy_neighbor_bits if topology is not None else None
                    boundary_points = int(records.shape[0]) if records is not None else 0
                    contours = len(topology.contours_xy or ()) if topology is not None else 0
                    if (diagnostic_points + points > _size.MAX_SIZE_DIAGNOSTIC_POINTS
                            or diagnostic_boundary_points + boundary_points > _size.MAX_SIZE_DIAGNOSTIC_POINTS
                            or diagnostic_contours + contours > _size.MAX_CONTOUR_COMPONENTS):
                        raise _size.RasterSpanLimitError('Retained diagnostic geometry exceeds its shared bound.')
                    diagnostic_points += points
                    diagnostic_boundary_points += boundary_points
                    diagnostic_contours += contours
                    label_results.append(result)
                except (NotImplementedError, _size.RasterSpanLimitError):
                    label_results.clear()
                    diagnostic_available = False
            observations[f'{labels[label_id-1]}_{label_id}'] = {
                item.variable: {'trait': item.trait, 'method': item.method,
                    'scale': item.scale, 'datatype': str(item.datatype),
                    'value': item.value, 'label': item.label}
                for item in result.output_observations}
        outputs = {'observations': _freeze_json(observations, name='size observations')}
        if not diagnostic_available:
            _cancel(size_poll)
            return outputs
        try:
            outputs['diagnostic'] = _size.render_plantcv_size_diagnostic_rgb(
                inputs['image'], tuple(label_results), cancel=size_poll, line_thickness=line_thickness)
        except (NotImplementedError, _size.RasterSpanLimitError):
            # Accurate numerical observations survive a diagnostic-only limit.
            # The declared missing output makes this a partial node.
            pass
        _cancel(size_poll)
        return outputs
    raise NotImplementedError(f'Native calculation is unavailable for {operation}.')


def _diagnostic_retained_bound(pixels, *, calibrated=False, range_arm=False):
    if not 1 <= pixels <= _watershed_diagnostic.MAX_PIXELS:
        raise NativeRecipeError('Watershed diagnostic graph exceeds its raster bound.')
    base = (_calibrated_seed.retained_bound(pixels,range_arm=True) if range_arm else
            _calibrated_seed.retained_bound(pixels) if calibrated else _retained_recipe_bound(pixels))
    # The new RGB copy is retained. Contour extraction and paint scratch do not
    # overlap; calibrated admission already includes 256MiB scratch.
    # Use the composed seed graph's larger dense envelope for source variants
    # too: greedy NMS scratch may dominate before this branch starts drawing.
    composed = 64*pixels + 16*min(pixels, _peaks.MAX_CANDIDATES) + 256*1024*1024
    bound = max(base, composed) + 3*pixels
    if pixels > _watershed_diagnostic.MAX_PIXELS or bound > MAX_RETAINED_CUDA_BYTES:
        raise NativeRecipeError('Watershed diagnostic graph exceeds its joint CUDA byte bound.')
    return bound


def _preparation_identity():
    executor=__import__(__name__,fromlist=['_preparation_identity'])
    helper=_digest(_module_record(_image_inputs))
    preparation=_digest(_module_record(executor,only=(
        '_preparation_identity','_prepared_facts','_audit_preparation','_preparation_callback','_path_preparation_poll',
        '_preparation_bound','_gray_preparation_prefix','prepare_native_recipe_inputs')))
    converter=_digest(_module_record(_range))
    return (helper,preparation,converter,_digest(_image_inputs.versions()),
        _digest(_module_record(_card_recipe.correction)),_digest(_module_record(_tensor_callback)),
        _digest(_module_record(_reader_recipe)))


def _prepared_facts(prepared):
    return (id(prepared),type(prepared).__qualname__,id(prepared.source_path),str(prepared.source_path),
        prepared.source_sha256,id(prepared.coordinate_frame),
        _card_recipe.correction._metadata_signature(prepared.coordinate_frame),
        _card_recipe.correction._metadata_signature(prepared.preparation),
        id(prepared.gray),_tensor_signature(prepared.gray),
        id(prepared.rgb),_tensor_signature(prepared.rgb))


def _audit_preparation(identity, decoder):
    try:
        changed=cv2.imdecode is not decoder or _preparation_identity()!=identity
    except NativeRecipeError as error:
        raise NativePreparationMutation('Loaded image preparation changed during calculation.') from error
    if changed:
        raise NativePreparationMutation('Loaded image preparation or decoder changed during calculation.')


def _preparation_callback(cancel, *, path=None, tensors=(), metadata=()):
    before=None if path is None else _image_inputs.stamp(path)
    owned=tuple((value,_prepared_facts(value)) for value in metadata
        if type(value).__name__=='NativeRecipeInputs')
    guard=_card_recipe.correction._guarded(cancel)
    try:
        try:
            guard(*tensors,metadata=metadata)
        except _card_recipe.correction.CorrectionMutation as error:
            raise NativePreparationMutation(str(error)) from error
        except _card_recipe.correction.CorrectionCancelled as error:
            raise NativeRecipeCancelled('Native image preparation was cancelled.') from error
    finally:
        if any(_prepared_facts(value)!=facts for value,facts in owned):
            raise NativePreparationMutation('Prepared source/frame facts changed during its callback.')
        if path is not None and _image_inputs.stamp(path)!=before:
            raise NativePreparationMutation('Source image changed during its preparation callback.')
    return False


def _path_preparation_poll(path, cache, cancel):
    entry=cache.values.get('reference.prepared')
    prepared=(entry[1] if isinstance(entry,tuple) and len(entry)>1
        and isinstance(entry[1],NativeRecipeInputs) else None)
    try:
        try:
            _preparation_callback(cancel,path=Path(path).resolve(strict=False),
                tensors=() if prepared is None else (prepared.gray,prepared.rgb),
                metadata=() if prepared is None else (prepared,))
        except NativePreparationMutation:
            cache.values.pop('reference.prepared',None)
            raise
    finally:
        if cache.values.get('reference.prepared') is not entry:
            cache.values.pop('reference.prepared',None)
            raise NativePreparationMutation('Prepared image cache binding changed at the path entry.')


def _preparation_bound(pixels, model, categories, automatic_card, calibrated, core_recovery,
                       watershed_debug, smpts_paper, calibrated_range):
    if watershed_debug:
        return _diagnostic_retained_bound(pixels,calibrated=calibrated,range_arm=calibrated_range)
    if smpts_paper:return _paper_recipe.retained_bound(pixels)
    if core_recovery:return _recovery_recipe.retained_bound(pixels)
    if calibrated:return _calibrated_seed.retained_bound(pixels,range_arm=calibrated_range)
    if automatic_card:return _card_recipe.retained_bound(pixels)
    return _retained_recipe_bound(pixels,model,categories)


def _gray_preparation_prefix(mode):
    if type(mode) is not str or mode not in _reader_recipe.MODES:
        raise NativeRecipeError('Select color-derived or codec-gray grayscale preparation.')
    flags = (_image_inputs.GRAY_DECODE_FLAGS if mode == 'codec-gray' else _image_inputs.DECODE_FLAGS)
    contract = (_image_inputs.CODEC_GRAY_CONTRACT if mode == 'codec-gray' else _image_inputs.GRAY_CONTRACT)
    return f'gray_source_mode={mode}; decode_flags={flags}; '+contract


def _validate_preparation_mode(recipe, preparation):
    if _reader_recipe.is_reader_recipe(recipe):
        mode = _reader_recipe.selected_mode(recipe)
        prefix = _gray_preparation_prefix(mode)
        suffix = (r' decoder build/flags identity [0-9a-f]{64}; immutable encoded snapshot '
            r'[1-9][0-9]* bytes; input helper [0-9a-f]{64}; preparation [0-9a-f]{64}; full decoded frame\.'
            if mode == 'codec-gray' else r' CUDA converter [0-9a-f]{64}; full decoded frame\.')
        if (not isinstance(preparation, Mapping) or type(preparation.get('gray')) is not str
                or re.fullmatch(re.escape(prefix)+suffix, preparation['gray']) is None):
            raise NativeRecipeError('Prepared grayscale facts do not match the selected reader mode and flags.')


def prepare_native_recipe_inputs(path, *, device='cuda', cache=None, cancel=None, _model=None,
                                 _categories=None, _automatic_card=False, _calibrated=False, _core_recovery=False,
                                 _watershed_debug=False, _smpts_paper=False, _calibrated_range=False,
                                 gray_source_mode='color-derived'):
    """Decode identified bytes; select CUDA-derived or codec-direct grayscale.

    All recipes share strict callback/source/code guards. Cached pixels retain
    detached frame/source/preparation facts as well as tensor signatures.
    Encoded file limits do not certify decoder-internal or process RSS memory.
    """
    if cache is not None and not isinstance(cache,PipelineAnalysisCache):
        raise TypeError('cache must be a PipelineAnalysisCache or None.')
    if cancel is not None and not callable(cancel):raise TypeError('cancel must be callable or None.')
    cache_name='reference.prepared'
    identity=_preparation_identity();decoder=cv2.imdecode
    candidate=None;entry=None
    if cache is not None:
        entry=cache.values.get(cache_name)
        if (isinstance(entry,tuple) and len(entry)==4 and isinstance(entry[1],NativeRecipeInputs)
                and torch.is_tensor(entry[1].gray) and torch.is_tensor(entry[1].rgb)):
            candidate=entry[1]
            if (entry[2]!=(_tensor_signature(candidate.gray),_tensor_signature(candidate.rgb))
                    or entry[3]!=_prepared_facts(candidate)):
                cache.values.pop(cache_name,None);candidate=entry=None
        elif entry is not None:
            cache.values.pop(cache_name,None);entry=None
    resolved=Path(path).resolve(strict=False)
    def poll(*tensors,metadata=()):
        watched=() if candidate is None else (candidate.gray,candidate.rgb)
        compact=metadata if candidate is None else (*metadata,candidate)
        facts=None if candidate is None else _prepared_facts(candidate)
        try:
            return _preparation_callback(cancel,path=resolved,tensors=(*watched,*tensors),metadata=compact)
        finally:
            if candidate is not None and (cache.values.get(cache_name) is not entry
                    or _prepared_facts(candidate)!=facts):
                raise NativePreparationMutation('Prepared image cache binding changed during a callback.')
    try:
        poll()
        gray_prefix = _gray_preparation_prefix(gray_source_mode)
        if gray_source_mode != 'color-derived' and any((_automatic_card, _calibrated, _core_recovery,
                                                       _smpts_paper, _calibrated_range, _model is not None)):
            raise NativeRecipeError('Codec grayscale is supported only by uncalibrated seed workflows.')
        if isinstance(_model,_kmeans_model.KMeansModel):_kmeans_model.validate_kmeans_model(_model)
        elif _model is not None:_naive_bayes._validate_model(_model)
        resolved=Path(path).resolve(strict=True)
        source=_image_inputs.source_identity(resolved,poll)
        device=torch.device(device)
        if device.type!='cuda':raise NativeRecipeError('Native recipe preparation requires CUDA.')
        if device.index is None:device=torch.device('cuda',torch.cuda.current_device())
        key=(str(resolved),source.size,source.sha256,str(device),identity,gray_source_mode)
        bounds=(_model,_categories,_automatic_card,_calibrated,_core_recovery,
            _watershed_debug,_smpts_paper,_calibrated_range)
        if candidate is not None and entry[0]==key:
            _preparation_bound(candidate.gray.numel(),*bounds)
            poll(metadata=(source,))
            if _image_inputs.source_identity(resolved,poll)!=source:
                raise NativePreparationMutation('Source image changed during cached preparation.')
            if (entry[2]!=(_tensor_signature(candidate.gray),_tensor_signature(candidate.rgb))
                    or entry[3]!=_prepared_facts(candidate)):
                raise NativePreparationMutation('Prepared image facts changed during reuse.')
            return candidate
        reuse_color = (candidate is not None and type(entry[0]) is tuple and len(entry[0]) == 6
            and entry[0][:5] == key[:5] and candidate.rgb.dtype == torch.uint8
            and candidate.rgb.device == device and candidate.rgb.is_contiguous()
            and candidate.rgb.ndim == 3 and candidate.rgb.shape[-1] == 3
            and tuple(candidate.rgb.shape[:2]) == candidate.coordinate_frame.shape
            and candidate.gray.dtype == torch.uint8 and candidate.gray.device == device
            and candidate.gray.is_contiguous() and tuple(candidate.gray.shape) == candidate.coordinate_frame.shape)
        if not reuse_color:
            if cache is not None:cache.values.pop(cache_name,None)
            candidate=entry=None
        raw=_image_inputs.snapshot(resolved,source,poll)
        poll(metadata=(source,))
        decoded = None if reuse_color else _image_inputs.decode(raw)
        shape = tuple(candidate.rgb.shape[:2]) if reuse_color else tuple(decoded.shape[:2])
        poll(metadata=(source,decoded))
        pixels=int(shape[0])*int(shape[1])
        retained = _preparation_bound(pixels,*bounds)
        if reuse_color:
            storage = _nested_cuda_storage((candidate.gray,candidate.rgb))
            candidate_bytes = sum(storage.values())
            for cached in cache.values.values():
                if isinstance(cached, _CacheEntry):
                    storage.update(_nested_cuda_storage(cached.node.outputs))
            # Retain the old cache until publication. Charge its unique backing
            # buffers plus the new BGR/gray and conservative conversion scratch.
            replacement = sum(storage.values())+4*pixels+_size.MAX_SIZE_DIAGNOSTIC_SCRATCH_BYTES
            if max(retained+candidate_bytes,replacement) > MAX_RETAINED_CUDA_BYTES:
                raise NativeRecipeError('Reader replacement exceeds the retained-product byte budget.')
        _image_inputs.decoded_bound(pixels, color=not reuse_color, gray=gray_source_mode == 'codec-gray')
        decoded_gray = (_image_inputs.decode_gray(raw,shape) if gray_source_mode == 'codec-gray' else None)
        del raw
        if ((0 if decoded is None else decoded.nbytes)+(0 if decoded_gray is None else decoded_gray.nbytes)
                > _image_inputs.MAX_DECODED_BYTES):
            raise NativeRecipeError('Owned decoded image arrays exceed the 256MiB limit.')
        poll(metadata=(source,decoded,decoded_gray))
        if reuse_color:
            rgb=candidate.rgb
            bgr=rgb.flip(-1).contiguous()
        else:
            bgr=torch.as_tensor(decoded,device=device)
            rgb=bgr.flip(-1).contiguous()
        del decoded
        if decoded_gray is not None:
            poll(bgr,rgb,metadata=(source,decoded_gray))
            gray=torch.as_tensor(decoded_gray,device=device)
            poll(bgr,rgb,gray,metadata=(source,decoded_gray))
            del decoded_gray
        else:
            gray=torch.empty(tuple(bgr.shape[:2]),dtype=torch.uint8,device=device)
            flat_bgr,flat_gray=bgr.reshape(-1,3),gray.reshape(-1)
            for start in range(0,pixels,_color.TILE_PIXELS):
                poll(bgr,rgb,gray,metadata=(source,))
                flat_gray[start:start+_color.TILE_PIXELS]=_range._channels(
                    flat_bgr[start:start+_color.TILE_PIXELS],'GRAY').reshape(-1)
                poll(bgr,rgb,gray,metadata=(source,))
        helper,preparer,converter,decoder_identity,*guard_identity=identity
        preparation=MappingProxyType({
            'gray':gray_prefix+(f' decoder build/flags identity {decoder_identity}; immutable encoded '
                f'snapshot {source.size} bytes; input helper {helper}; preparation {preparer}; full decoded frame.'
                if gray_source_mode == 'codec-gray' else f' CUDA converter {converter}; full decoded frame.'),
            'rgb':_image_inputs.COLOR_CONTRACT+f' OpenCV {cv2.__version__}; immutable encoded snapshot '
                f'{source.size} bytes; decoder build/flags identity {decoder_identity}; '
                f'input helper {helper}; preparation {preparer}; CUDA BGR-to-RGB channel reversal; no rectification.'})
        prepared=NativeRecipeInputs(resolved,source.sha256,
            CoordinateFrame('source.decoded',tuple(gray.shape),'top-left','x-right,y-down','pixel'),
            gray,rgb,preparation)
        poll(bgr,rgb,gray,metadata=(source,prepared))
        if _image_inputs.source_identity(resolved,lambda **kw:poll(bgr,rgb,gray,
                metadata=(source,prepared,*kw.get('metadata',()))))!=source:
            raise NativePreparationMutation('Source image changed during preparation.')
        publication=(key,prepared,(_tensor_signature(gray),_tensor_signature(rgb)),_prepared_facts(prepared))
    except NativePreparationMutation:
        if cache is not None:cache.values.pop(cache_name,None)
        raise
    except _image_inputs.ImageInputError as error:
        if cache is not None:cache.values.pop(cache_name,None)
        raise NativeRecipeError(str(error)) from error
    finally:
        try:
            _audit_preparation(identity,decoder)
        except NativePreparationMutation:
            if cache is not None:cache.values.pop(cache_name,None)
            raise
    if cache is not None:cache.values[cache_name]=publication
    return prepared


def _checked_model_asset_paths(recipe, value):
    if value is None:
        return MappingProxyType({})
    declared = {stage.stage_id for stage in recipe.stages if stage.operation_id in (
        'external.plantcv.kmeans_centres', 'external.plantcv.naive_bayes_pdf')}
    if not isinstance(value, Mapping) or set(value)-declared:
        raise NativeRecipeError('Model asset paths must bind only declared model stages.')
    return MappingProxyType({stage: _asset_paths.validate_model_origin(path) for stage, path in value.items()})


def run_native_recipe_path(recipe, path, *, device='cuda', cache=None, cancel=None,
                           model_asset_paths=None, timing_recorder=None):
    """Shared desktop/batch path entry point; the selected RecipeSpec is unchanged."""
    if cache is None:
        cache = PipelineAnalysisCache()
    if not isinstance(cache, PipelineAnalysisCache):
        raise TypeError('cache must be a PipelineAnalysisCache or None.')
    _path_preparation_poll(path,cache,cancel)
    _validate_recipe(recipe)
    model_asset_paths = _checked_model_asset_paths(recipe, model_asset_paths)
    model = None
    categories = None
    external_inputs = None
    if recipe.recipe_id in _NAIVE_BAYES_RECIPES:
        model, identity = _resolve_naive_bayes_binding(recipe, cache, model_asset_paths)
        external_inputs = {_naive_bayes_recipe.MODEL_STAGE_ID: (model, identity)}
    elif recipe.recipe_id in _KMEANS_RECIPES:
        model, identity = _resolve_kmeans_binding(recipe, cache, model_asset_paths)
        categories = recipe.parameters_for_runner(_kmeans_recipe.MASK_STAGE_ID)['cat_list']
        external_inputs = {_kmeans_recipe.MODEL_STAGE_ID: (model, identity)}
    if timing_recorder is not None:
        from seedvision.timing import NodeTimingRecorder
        if not isinstance(timing_recorder, NodeTimingRecorder):
            raise TypeError('Benchmark timing requires a NodeTimingRecorder.')
    span = timing_recorder.start('input_preparation', report_progress=False) if timing_recorder else None
    try:
        prepared = prepare_native_recipe_inputs(path, device=device, cache=cache, cancel=cancel,
            _model=model, _categories=categories, _automatic_card=recipe.recipe_id == _card_recipe.RECIPE_ID,
            _calibrated=_diagnostic_recipe.base_recipe_id(recipe.recipe_id) in _calibrated_seed.RECIPE_IDS,
            _calibrated_range=_diagnostic_recipe.base_recipe_id(recipe.recipe_id) == _calibrated_seed.RANGE_RECIPE_ID,
            _watershed_debug=_morphology_roi_recipe.base_recipe_id(recipe.recipe_id) in _diagnostic_recipe.RECIPE_IDS,
            _smpts_paper=recipe.recipe_id == _paper_recipe.RECIPE_ID,
            _core_recovery=recipe.recipe_id == _recovery_recipe.RECIPE_ID,
            gray_source_mode=_reader_recipe.selected_mode(recipe))
    finally:
        if span is not None:
            timing_recorder.stop(span, completed=False)
    return run_native_recipe(recipe, prepared.gray, prepared.rgb, source_sha256=prepared.source_sha256,
        coordinate_frame=prepared.coordinate_frame, preparation=prepared.preparation,
        cache=cache, cancel=cancel, external_inputs=external_inputs, model_asset_paths=model_asset_paths,
        timing_recorder=timing_recorder)


def run_native_recipe(recipe, gray, rgb, *, source_sha256, coordinate_frame,
                      preparation, cache=None, cancel=None, external_inputs=None, model_asset_paths=None,
                      timing_recorder=None):
    """Execute the exact selected graph; retain valid per-node CUDA products.

    Preparation records must explain the caller's grayscale/RGB boundaries. A
    null analysis n_labels is resolved from the actual final label raster and
    recorded in the node result. The returned hue chart retains exact compact
    observation rows with native rendering; size overlays remain unavailable.
    Failures are explicit per node; siblings can complete, descendants cannot
    reuse an older output. Cancellation raises and never returns a complete run.
    """
    if cancel is not None and not callable(cancel):
        raise TypeError('cancel must be callable or None.')
    if timing_recorder is not None:
        from seedvision.timing import NodeTimingRecorder
        if not isinstance(timing_recorder, NodeTimingRecorder):
            raise TypeError('Benchmark timing requires a NodeTimingRecorder.')
    diagnostic_run = isinstance(recipe, RecipeSpec) and _morphology_roi_recipe.base_recipe_id(recipe.recipe_id) in _diagnostic_recipe.RECIPE_IDS
    calibrated = isinstance(recipe, RecipeSpec) and _diagnostic_recipe.base_recipe_id(recipe.recipe_id) in _calibrated_seed.RECIPE_IDS
    calibrated_range = calibrated and _diagnostic_recipe.base_recipe_id(recipe.recipe_id)==_calibrated_seed.RANGE_RECIPE_ID
    card_run = calibrated or isinstance(recipe, RecipeSpec) and recipe.recipe_id == _card_recipe.RECIPE_ID
    recovery_run = isinstance(recipe, RecipeSpec) and recipe.recipe_id == _recovery_recipe.RECIPE_ID
    paper_run = isinstance(recipe, RecipeSpec) and recipe.recipe_id == _paper_recipe.RECIPE_ID
    range_run = isinstance(recipe, RecipeSpec) and recipe.recipe_id == _range_recipe.RECIPE_ID
    guarded_run = card_run or recovery_run or diagnostic_run or paper_run or range_run
    card_guard = (_CardTensorGuard(gray, rgb, cancel, cache, recipe.recipe_id,
                    label='Custom range' if range_run else 'SMPTS paper reconstruction' if paper_run else 'Core recovery' if recovery_run else 'Watershed diagnostic' if diagnostic_run else 'Automatic card',
                    clear_on_callback_error=recovery_run or diagnostic_run or paper_run or range_run or calibrated_range) if guarded_run else None)
    if guarded_run:
        cancel = card_guard
    _cancel(cancel)
    _validate_recipe(recipe)
    model_asset_paths = _checked_model_asset_paths(recipe, model_asset_paths)
    if cache is None:
        cache = PipelineAnalysisCache()
    if guarded_run:
        card_guard.cache = cache
    if not isinstance(cache, PipelineAnalysisCache):
        raise TypeError('cache must be an existing PipelineAnalysisCache.')
    if external_inputs is not None and not isinstance(external_inputs, Mapping):
        raise NativeRecipeError('External recipe inputs must be a mapping.')
    selected_model = selected_identity = None
    if recipe.recipe_id in _NAIVE_BAYES_RECIPES:
        selected_model, selected_identity = _resolve_naive_bayes_binding(recipe, cache, model_asset_paths)
        if external_inputs is not None:
            try:
                if set(external_inputs) != {_naive_bayes_recipe.MODEL_STAGE_ID}:
                    raise ValueError('Unknown or missing external model input stage.')
                provided = external_inputs[_naive_bayes_recipe.MODEL_STAGE_ID]
                if not isinstance(provided, (tuple, list)) or len(provided) != 2:
                    raise ValueError('External model input must carry the model and selected identity.')
                candidate, identity = provided
                _naive_bayes._validate_model(candidate)
                if (candidate != selected_model or not isinstance(identity, Mapping)
                        or dict(identity) != dict(selected_identity)):
                    raise ValueError('External model differs from the currently selected file bytes and identity.')
            except Exception as error:
                _clear_naive_bayes_cache(cache)
                raise NativeRecipeError(f'External model input is invalid: {error}') from error
    elif recipe.recipe_id in _KMEANS_RECIPES:
        selected_model, selected_identity = _resolve_kmeans_binding(recipe, cache, model_asset_paths)
        if external_inputs is not None:
            try:
                if set(external_inputs) != {_kmeans_recipe.MODEL_STAGE_ID}:
                    raise ValueError('Unknown or missing external model input stage.')
                provided = external_inputs[_kmeans_recipe.MODEL_STAGE_ID]
                if not isinstance(provided, (tuple, list)) or len(provided) != 2:
                    raise ValueError('External model input must carry the model and selected identity.')
                candidate, identity = provided
                _kmeans_model.validate_kmeans_model(candidate)
                if (candidate != selected_model or not isinstance(identity, Mapping)
                        or dict(identity) != dict(selected_identity)):
                    raise ValueError('External model differs from the currently selected bytes and identity.')
            except Exception as error:
                _clear_kmeans_cache(cache)
                raise NativeRecipeError(f'External model input is invalid: {error}') from error
    elif external_inputs:
        raise NativeRecipeError('This recipe does not accept external model inputs.')
    if not isinstance(source_sha256, str) or not re.fullmatch('[0-9a-f]{64}', source_sha256):
        raise NativeRecipeError('Prepared inputs require a lowercase source SHA-256.')
    if not isinstance(coordinate_frame, CoordinateFrame):
        raise NativeRecipeError('Prepared inputs require their exact coordinate frame.')
    if (not isinstance(preparation, Mapping) or set(preparation) != {'gray', 'rgb'}
            or any(not isinstance(value, str) or not value.strip() for value in preparation.values())):
        raise NativeRecipeError('Declare the gray and RGB preparation independently.')
    _validate_preparation_mode(recipe, preparation)
    if (not all(torch.is_tensor(value) and value.layout == torch.strided
            and value.dtype == torch.uint8 and value.device.type == 'cuda' and value.is_contiguous()
            for value in (gray, rgb)) or gray.ndim != 2 or rgb.ndim != 3 or rgb.shape[-1] != 3
            or tuple(gray.shape) != coordinate_frame.shape or tuple(rgb.shape[:2]) != coordinate_frame.shape
            or gray.device != rgb.device):
        raise NativeRecipeError('Require same-frame contiguous CUDA uint8 HW gray and HWC RGB inputs.')
    pixels = gray.numel()
    # Bound retained dense products and the maximum compact peak-coordinate tensor
    # before any stage allocations; individual primitives also bound scratch.
    categories = (recipe.parameters_for_runner(_kmeans_recipe.MASK_STAGE_ID)['cat_list']
                  if recipe.recipe_id in _KMEANS_RECIPES else None)
    retained_bound = (_diagnostic_retained_bound(pixels, calibrated=calibrated,range_arm=calibrated_range) if diagnostic_run else
                      _range_recipe.retained_bound(pixels) if range_run else
                      _paper_recipe.retained_bound(pixels) if paper_run else
                      _recovery_recipe.retained_bound(pixels) if recovery_run else
                      _calibrated_seed.retained_bound(pixels,range_arm=True) if calibrated_range else
                      _calibrated_seed.retained_bound(pixels) if calibrated else
                      _card_recipe.retained_bound(pixels) if card_run
                      else _retained_recipe_bound(pixels, selected_model, categories))
    if recovery_run or diagnostic_run or paper_run or range_run or calibrated_range:
        actual_inputs=_nested_cuda_storage({'gray':gray,'rgb':rgb})
        retained_bound += max(0,sum(actual_inputs.values())-4*pixels)
        if retained_bound > MAX_RETAINED_CUDA_BYTES:
            label = 'Custom range' if range_run else 'SMPTS paper reconstruction' if paper_run else 'Watershed diagnostic' if diagnostic_run else 'Calibrated range' if calibrated_range else 'Recovery'
            raise NativeRecipeError(label+' source backing exceeds the joint graph byte bound.')
    scope = _digest({'source': source_sha256, 'frame': coordinate_frame.to_dict(),
        'preparation': dict(preparation), 'device': str(gray.device)})
    if cache.values.get('reference.source.scope') != scope:
        for name in tuple(cache.values):
            if name.startswith('reference.recipe:'):
                del cache.values[name]
        cache.values['reference.source.scope'] = scope
    prefix = f'reference.recipe:{recipe.recipe_id}:'
    calculation_watched = []
    if any(stage.operation_id == 'threshold.otsu' for stage in recipe.stages):
        original_cancel = cancel
        def calculation_poll():
            try:
                return original_cancel() if original_cancel is not None else False
            finally:
                for watched_node, signature in calculation_watched:
                    if _calculation_signature(watched_node) != signature:
                        _evict_recipe_descendants(recipe, watched_node.stage_id, cache, prefix)
                        raise NativeRecipeError('Native Otsu cutoff facts changed during execution.')
        cancel = calculation_poll
    # Retain alternate recipe caches when they fit, evict them before new
    # allocations if their actual CUDA storage plus this run's bound exceeds
    # the session budget. Unrelated production-cache keys are never changed.
    storage = {}
    other_names = []
    for name, entry in cache.values.items():
        if name.startswith('reference.recipe:') and not name.startswith(prefix):
            other_names.append(name)
            if isinstance(entry, _CacheEntry):
                storage.update(_nested_cuda_storage(entry.node.outputs))
    if sum(storage.values())+retained_bound > MAX_RETAINED_CUDA_BYTES:
        for name in other_names:
            del cache.values[name]
    runtime_audit = _runtime_records
    module_records, runtime = (runtime_audit(gray.device, card=card_run, calibrated=calibrated,
                                           diagnostic=True) if diagnostic_run else
                               runtime_audit(gray.device, paper=True) if paper_run else
                               runtime_audit(gray.device, recovery=True) if recovery_run else
                               _runtime_records(gray.device, card=True, calibrated=True) if calibrated else
                               _runtime_records(gray.device, card=True) if card_run
                               else _runtime_records(gray.device))
    execution = _digest({'runtime': runtime, 'modules': module_records})
    base = {'source': source_sha256, 'frame': coordinate_frame.to_dict(),
        'preparation': dict(preparation), 'graph': recipe.identity.graph_sha256, 'execution': execution}
    nodes, computed, reused, timings = {}, [], [], {}
    for stage in _topological_stages(recipe):
        _cancel(cancel)
        parameters = recipe.parameters_for_runner(stage.stage_id)
        incoming = [edge for edge in recipe.dependencies if edge.target_stage == stage.stage_id]
        inputs, input_keys = {}, {}
        unavailable = []
        for edge in incoming:
            upstream = nodes[edge.source_stage]
            if edge.source_port not in upstream.outputs or upstream.product_key is None:
                unavailable.append(f'{edge.source_stage}.{edge.source_port}')
            else:
                inputs[edge.target_port] = upstream.outputs[edge.source_port]
                input_keys[edge.target_port] = _digest([upstream.product_key, edge.source_port])
        cache_name = prefix+stage.stage_id
        if unavailable:
            cache.values.pop(cache_name, None)
            nodes[stage.stage_id] = NativeRecipeNode(stage.stage_id, stage.operation_id, 'not_run',
                MappingProxyType({}), MappingProxyType(parameters), MappingProxyType(input_keys), None,
                reason='Unavailable inputs: '+', '.join(unavailable))
            continue
        if stage.operation_id in ('analysis.color', 'analysis.size') and parameters.get('n_labels') is None:
            parameters['n_labels'] = int(inputs['labels'].max().item())
        calculation_parameters = parameters
        if stage.operation_id == 'external.plantcv.naive_bayes_pdf':
            calculation_parameters = {**parameters, 'model_source':
                _naive_bayes_recipe.calculation_model_source(parameters['model_source'])}
        key_payload = {**base, 'stage': stage.stage_id, 'parameters': calculation_parameters, 'inputs': input_keys}
        if stage.operation_id == 'external.image_inputs':
            key_payload['tensors'] = [_tensor_signature(gray), _tensor_signature(rgb)]
        key = _digest(key_payload)
        entry = cache.values.get(cache_name)
        if isinstance(entry, _CacheEntry) and entry.key == key and entry.signatures == _output_signatures(
                entry.node.outputs,compact_recovery=recovery_run or paper_run) and (
                entry.calculation_signature == _calculation_signature(entry.node)):
            previous = entry.node
            node = NativeRecipeNode(previous.stage_id, previous.operation_id, previous.status,
                previous.outputs, _freeze_json(parameters, name='resolved parameters'),
                previous.input_keys, previous.product_key,
                reused=True, reason=previous.reason, calculation=previous.calculation)
            nodes[stage.stage_id] = node
            if node.calculation is not None:
                calculation_watched.append((node, _calculation_signature(node)))
            if guarded_run:
                card_guard.watch(node.outputs)
            reused.append(stage.stage_id)
            timings[stage.stage_id] = 0.0
            if timing_recorder:
                timing_recorder.add_seconds(stage.stage_id, 0.0)
            continue
        cache.values.pop(cache_name, None)
        if guarded_run:
            _evict_recipe_descendants(recipe, stage.stage_id, cache, prefix)
        if recipe.recipe_id in _KMEANS_RECIPES:
            # Drop invalid downstream products before allocating a replacement
            # predictor, preserving the independent image/preparation branch.
            downstream = {
                _kmeans_recipe.MODEL_STAGE_ID: (_kmeans_recipe.PREDICT_STAGE_ID, _kmeans_recipe.MASK_STAGE_ID),
                _kmeans_recipe.PREDICT_STAGE_ID: (_kmeans_recipe.MASK_STAGE_ID,),
            }.get(stage.stage_id, ())
            for dependent in downstream:
                cache.values.pop(prefix+dependent, None)
        entry = None  # Release stale cached masks before allocating their replacement.
        started = time.perf_counter()
        span = timing_recorder.start(stage.stage_id, report_progress=False) if timing_recorder else None
        try:
            optional = {'analysis.kmeans_masks': ('cat_list',),
                'recovery.validity': ('validity_roi',),
                'color_card.preprocess': ('roi_bounds_xyxy',),
                'color_card.scale': ('color_chip_size',)}.get(stage.operation_id, ())
            if stage.operation_id in _morphology_roi_recipe.OPERATIONS:
                optional = (_morphology_roi_recipe.PARAMETER,)
            if stage.operation_id == _fill_roi_recipe.OPERATION:
                optional = (_fill_roi_recipe.PARAMETER,)
            if stage.operation_id == _median_roi_recipe.OPERATION:
                optional = (_median_roi_recipe.PARAMETER,)
            _required_parameters(parameters, optional=optional)
            if stage.operation_id == 'external.image_inputs':
                outputs, status = {'gray_image': gray, 'rgb_image': rgb}, 'provided'
                if any(port.name == 'source_frame' and port.direction == 'output' for port in stage.ports):
                    outputs['source_frame'] = _freeze_json({'coordinate_frame':coordinate_frame.to_dict(),
                        'source_sha256':source_sha256},name='prepared source frame')
            elif stage.operation_id == 'external.plantcv.naive_bayes_pdf':
                outputs, status = {'model': selected_model, 'model_identity': selected_identity}, 'provided'
            elif stage.operation_id == 'external.plantcv.kmeans_centres':
                outputs, status = {'model': selected_model, 'model_identity': selected_identity}, 'provided'
            else:
                if stage.operation_id in ('analysis.color', 'analysis.size') and not 1 <= parameters['n_labels'] <= MAX_LABELS:
                    raise NativeRecipeError('Analysis requires an actual object count in [1,128].')
                outputs, status = _dispatch(stage.operation_id, inputs, parameters, cancel), 'executed'
            calculation = _dispatch_calculation(stage.operation_id, outputs, parameters)
            if guarded_run:
                card_guard.watch(outputs)
            _cancel(cancel)
            # A fresh generation prevents descendants reusing products computed
            # from a mutated cache tensor, even if the lineage parameters match.
            generation = int(cache.values.get(prefix+'generation', 0))+1
            cache.values[prefix+'generation'] = generation
            product_key = _digest([key, generation] if calculation is None else [key, generation, dict(calculation)])
            missing_outputs = sorted(port.name for port in stage.ports
                if port.direction == 'output' and port.name not in outputs)
            reason = ''
            if missing_outputs:
                status = 'partial'
                reason = 'Unavailable outputs: '+', '.join(missing_outputs)
            node = NativeRecipeNode(stage.stage_id, stage.operation_id, status, MappingProxyType(outputs),
                _freeze_json(parameters, name='resolved parameters'), MappingProxyType(input_keys), product_key,
                reason=reason, calculation=calculation)
            cache.values[cache_name] = _CacheEntry(key, node,
                _output_signatures(outputs,compact_recovery=recovery_run or paper_run), _calculation_signature(node))
            if calculation is not None:
                calculation_watched.append((node, _calculation_signature(node)))
            computed.append(stage.stage_id)
        except (NativeRecipeCancelled, _median.MedianCancelledError, _fill.FillCancelledError,
                _distance.DistanceCancelledError, _peaks.PeakCancelledError,
                _watershed.WatershedCancelled, _watershed_diagnostic.WatershedDiagnosticCancelled,
                _color.ColorAnalysisCancelled,
                _recovery_recipe.core.CoreRecoveryCancelled,
                _paper_recipe.paper.PaperCancelled,
                _range.RangeCancelledError, _size.PlantCVSizeCancelledError,
                _extended_thresholds.ExtendedThresholdCancelled, _naive_bayes.NaiveBayesCancelled,
                _kmeans_masks.NativeKMeansCancelled, _kmeans_preprocess.KMeansPreprocessCancelled,
                _kmeans_features.KMeansFeatureCancelled, _kmeans_classifier.KMeansClassificationCancelled,
                _kmeans_predict.KMeansPredictionCancelled,
                _card_recipe.card.AutomaticCardCancelled, _card_recipe.preprocess.ColorCardPreprocessCancelled,
                _card_recipe.contours.BinaryContourCancelled, _card_recipe.correction.CorrectionCancelled) as error:
            raise NativeRecipeCancelled(str(error)) from error
        except NotImplementedError as error:
            node = NativeRecipeNode(stage.stage_id, stage.operation_id, 'unavailable', MappingProxyType({}),
                MappingProxyType(parameters), MappingProxyType(input_keys), None, reason=str(error))
            cache.values[cache_name] = _CacheEntry(key, node, ())
        except Exception as error:
            _cancel(cancel)
            node = NativeRecipeNode(stage.stage_id, stage.operation_id, 'failed', MappingProxyType({}),
                MappingProxyType(parameters), MappingProxyType(input_keys), None,
                reason=f'{type(error).__name__}: {error}')
        finally:
            if span is not None:
                timing_recorder.stop(span, completed=False)
        nodes[stage.stage_id] = node
        timings[stage.stage_id] = time.perf_counter()-started
    _cancel(cancel)
    if recovery_run or diagnostic_run or paper_run or range_run or calibrated_range:
        # Caller callbacks have finished. Recheck the actual loaded calculation
        # closure before publishing products under its initial runtime digest.
        try:
            final_records,final_runtime=(runtime_audit(gray.device, card=card_run, calibrated=calibrated,
                diagnostic=True) if diagnostic_run else runtime_audit(gray.device,paper=True) if paper_run
                else runtime_audit(gray.device,card=True,calibrated=True) if calibrated_range
                else runtime_audit(gray.device) if range_run else runtime_audit(gray.device,recovery=True))
            if final_records != module_records or final_runtime != runtime:
                raise NativeRecipeError('Guarded graph implementation changed during execution.')
        except BaseException:
            card_guard._clear()
            raise
    cache.last_computed_nodes, cache.last_reused_nodes = tuple(computed), tuple(reused)
    cache.node_timings_seconds = timings
    ordered = tuple(nodes[stage.stage_id] for stage in recipe.stages)
    complete = all(node.status in ('executed', 'provided', 'empty') for node in ordered)
    return NativeRecipeResult(recipe.identity, source_sha256, coordinate_frame,
        MappingProxyType(dict(preparation)), MappingProxyType(runtime), ordered,
        'executed' if complete else 'partial', model_asset_paths)
