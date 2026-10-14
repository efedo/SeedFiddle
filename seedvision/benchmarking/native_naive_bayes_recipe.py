"""Working graph and byte-bound asset selection for frozen PlantCV Bayes PDFs."""
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
import re
from types import MappingProxyType, SimpleNamespace

from seedvision.benchmarking.recipe_catalogue import (
    _image_semantics, _load_source_config, _plantcv_input_stage,
    _plantcv_mapping, _plantcv_provenance,
)
from seedvision.benchmarking.recipes import (
    DependencyContract, LayoutPosition, ParameterContract, PortContract,
    RecipeProvenanceMode, RecipeSpec, RecipeStage,
)
from seedvision.cuda.plantcv_naive_bayes import (
    MAX_CLASSES, MAX_MODEL_BYTES, _validate_model, load_plantcv_pdf, parse_plantcv_pdf,
)
from seedvision.model_io.asset_paths import resolve_model_path
from seedvision.model_io import naive_bayes_provenance as _training_record

NATIVE_NAIVE_BAYES_VERSION = '1.0.0'
NAIVE_BAYES_RECIPE_ID = 'plantcv_naive_bayes_classifier'
IMAGE_STAGE_ID = 'external.image_inputs'
MODEL_STAGE_ID = 'external.plantcv_naive_bayes_pdf'
CLASSIFIER_STAGE_ID = 'analysis.naive_bayes_classifier'
MODEL_PARAMETER_ID = 'model_source'
SOURCE_PATH = 'plantcv/plantcv/naive_bayes_classifier.py'
SOURCE_SYMBOL = 'plantcv.plantcv.naive_bayes_classifier.naive_bayes_classifier'
MODEL_FIELDS = ('path', 'source_sha256', 'tables_sha256', 'class_names', 'source_bytes')
TRAINING_FIELD = 'training_provenance'


def model_source_for_path(path):
    """Build the single saved selection value from a bounded, parsed model."""
    resolved = Path(path).expanduser().resolve(strict=True)
    model = load_plantcv_pdf(resolved, max_model_bytes=MAX_MODEL_BYTES, max_classes=MAX_CLASSES)
    _validate_model(model)
    return {
        'path': str(resolved),
        'source_sha256': model.source_sha256,
        'tables_sha256': model.tables_sha256,
        'class_names': list(model.class_names),
        'source_bytes': model.source_bytes,
    }


def model_source_for_trained_model(path, tsv_bytes, provenance_json):
    """Attach a typed project-held record only to the exact saved TSV asset."""
    expected = parse_plantcv_pdf(tsv_bytes)
    record = _training_record.parse_training_record(provenance_json, expected)
    selected = model_source_for_path(path)
    if (selected['source_sha256'] != expected.source_sha256
            or selected['source_bytes'] != expected.source_bytes
            or selected['tables_sha256'] != expected.tables_sha256
            or tuple(selected['class_names']) != expected.class_names):
        raise ValueError('Saved trained model differs from the completed TSV bytes.')
    return {**selected, TRAINING_FIELD: record.to_binding()}


def calculation_model_source(value):
    """Training history is recorded metadata; it does not change inference."""
    return {name: value[name] for name in MODEL_FIELDS}


def validate_model_source_metadata(value):
    """Validate the saved selection/record without opening either source file."""
    if not isinstance(value, Mapping):
        raise ValueError('Select a PlantCV PDF model file before running this recipe.')
    if set(value) not in (set(MODEL_FIELDS), set(MODEL_FIELDS) | {TRAINING_FIELD}):
        raise ValueError('The selected model binding has missing or unknown fields.')
    path = value['path']
    if not isinstance(path, str) or not path or path != path.strip():
        raise ValueError('The selected model path must be trimmed text.')
    for name in ('source_sha256', 'tables_sha256'):
        if not isinstance(value[name], str) or re.fullmatch(r'[0-9a-f]{64}', value[name]) is None:
            raise ValueError(f'The selected model {name} must be a lowercase SHA-256.')
    names = value['class_names']
    if not isinstance(names, (list, tuple)) or not 1 <= len(names) <= MAX_CLASSES:
        raise ValueError(f'The selected model must bind 1..{MAX_CLASSES} ordered class names.')
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError('Selected model class names must be non-empty text.')
    if len(set(names)) != len(names):
        raise ValueError('Selected model class names must be unique and ordered.')
    size = value['source_bytes']
    if type(size) is not int or not 1 <= size <= MAX_MODEL_BYTES:
        raise ValueError('Selected model byte length is outside the bounded PDF domain.')
    if TRAINING_FIELD in value:
        identity = SimpleNamespace(source_sha256=value['source_sha256'],
            tables_sha256=value['tables_sha256'], class_names=tuple(names))
        _training_record.validate_training_binding(value[TRAINING_FIELD], identity)
    return path, tuple(names)


def _validated_selection(value, *, resolved_path=None):
    path, names = validate_model_source_metadata(value)
    return resolve_model_path(path, resolved_path), names


def _check_selected_model(value, model, *, resolved_path=None):
    path, names = _validated_selection(value, resolved_path=resolved_path)
    _validate_model(model)
    if (model.source_sha256 != value['source_sha256']
            or model.tables_sha256 != value['tables_sha256']
            or model.class_names != names or model.source_bytes != value['source_bytes']):
        raise ValueError('Selected model file changed; choose it again to bind the new bytes.')
    if TRAINING_FIELD in value:
        _training_record.validate_training_binding(value[TRAINING_FIELD], model)
    return path


def load_selected_model(value, *, resolved_path=None):
    """Reread, reparse and compare all asset identity before any cache reuse."""
    canonical_path, _ = _validated_selection(value, resolved_path=resolved_path)
    model = load_plantcv_pdf(canonical_path, max_model_bytes=MAX_MODEL_BYTES, max_classes=MAX_CLASSES)
    _check_selected_model(value, model, resolved_path=resolved_path)
    return model


def selected_model_identity(value, model, *, resolved_path=None):
    """Freeze selection metadata only after checking it against parsed tables."""
    _check_selected_model(value, model, resolved_path=resolved_path)
    return MappingProxyType({
        'path': value['path'], 'source_sha256': model.source_sha256,
        'tables_sha256': model.tables_sha256, 'class_names': tuple(model.class_names),
        'source_bytes': model.source_bytes,
        'reference_version': '4.11.3',
        'reference_source_sha256': 'd3f40429ead7aa5deb298fa78d484d0f6d89c9cf05e706e07fdf7b97fedc3049',
    })


def _parameter_contracts():
    return (ParameterContract(
        parameter_id=MODEL_PARAMETER_ID, value_type='json', value=None,
        affects_calculation=True, editable=True,
        description='Selected local PlantCV PDF model, bound to exact file/table hashes and ordered classes. '
                    'The saved recipe references the file; it does not embed the model.',
    ),)


def build_native_naive_bayes_recipe():
    config = _load_source_config()
    original = _plantcv_provenance(config)
    provenance = replace(original, mode=RecipeProvenanceMode.NATIVE_COMPATIBLE,
        backend_id='seedvision.cuda.plantcv_naive_bayes',
        backend_version=NATIVE_NAIVE_BAYES_VERSION,
        evidence_note=('Frozen-model CUDA adapter for the pinned PlantCV 4.11.3 classifier. '
            'The model is user-selected and byte-bound; training and biological validity are not asserted.'))
    mapping = (_plantcv_mapping(config, SOURCE_PATH, SOURCE_SYMBOL),)
    model_semantics = {'representation': 'validated immutable PlantCV PDF probability tables',
        'source': 'single byte-bound local model asset selection'}
    identity_semantics = {'source': 'reread PDF file and recomputed tables'}
    masks_semantics = {'representation': 'ordered mapping from class name to CUDA uint8 HW 0/255 mask',
        'class_order': 'exact model insertion order'}
    image_stage = replace(_plantcv_input_stage('pcv_naive_bayes'), stage_id=IMAGE_STAGE_ID)
    model_stage = RecipeStage(
        stage_id=MODEL_STAGE_ID, title='Selected PlantCV PDF model',
        operation_id='external.plantcv.naive_bayes_pdf',
        description='Read and validate the selected bounded model before image preparation or cache reuse.',
        ports=(
            PortContract(name='model', direction='output', data_type='NaiveBayesModel',
                label='Validated probability tables', semantics=model_semantics),
            PortContract(name='model_identity', direction='output', data_type='ModelIdentity',
                label='Bound model identity', semantics=identity_semantics),
        ), parameters=_parameter_contracts(), source_mappings=mapping)
    classifier_stage = RecipeStage(
        stage_id=CLASSIFIER_STAGE_ID, title='Naive Bayes class masks',
        operation_id='analysis.naive_bayes_classifier',
        description='Apply exact PlantCV BGR-to-HSV likelihood products and first-class argmax ties.',
        ports=(
            PortContract(name='rgb_image', direction='input', data_type='RGBImage', label='RGB image',
                semantics={**_image_semantics('RGB'), 'channel_boundary': 'RGB uint8 HWC'}),
            PortContract(name='model', direction='input', data_type='NaiveBayesModel',
                label='Frozen PDF model', semantics=model_semantics),
            PortContract(name='model_identity', direction='input', data_type='ModelIdentity',
                label='Bound model identity', semantics=identity_semantics),
            PortContract(name='class_masks', direction='output', data_type='ClassMaskMap',
                label='Per-class masks', semantics=masks_semantics),
            PortContract(name='class_names', direction='output', data_type='OrderedClassNames',
                label='Model class order', semantics={'order': 'PDF insertion order'}),
            PortContract(name='model_identity', direction='output', data_type='ModelIdentity',
                label='Bound model identity', semantics=identity_semantics),
        ), source_mappings=mapping)
    return RecipeSpec(
        recipe_id=NAIVE_BAYES_RECIPE_ID, recipe_version=NATIVE_NAIVE_BAYES_VERSION,
        title='PlantCV frozen Naive Bayes classifier',
        description='Apply a selected, byte-bound PlantCV probability PDF to a decoded seed image.',
        provenance=provenance, stages=(image_stage, model_stage, classifier_stage),
        dependencies=(
            DependencyContract(source_stage=IMAGE_STAGE_ID, source_port='rgb_image',
                target_stage=CLASSIFIER_STAGE_ID, target_port='rgb_image', data_type='RGBImage'),
            DependencyContract(source_stage=MODEL_STAGE_ID, source_port='model',
                target_stage=CLASSIFIER_STAGE_ID, target_port='model', data_type='NaiveBayesModel'),
            DependencyContract(source_stage=MODEL_STAGE_ID, source_port='model_identity',
                target_stage=CLASSIFIER_STAGE_ID, target_port='model_identity', data_type='ModelIdentity'),
        ), layout=(LayoutPosition(stage_id=IMAGE_STAGE_ID, x=0, y=0),
            LayoutPosition(stage_id=MODEL_STAGE_ID, x=0, y=180),
            LayoutPosition(stage_id=CLASSIFIER_STAGE_ID, x=350, y=90)))
