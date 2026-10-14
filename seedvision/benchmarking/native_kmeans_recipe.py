"""Working graph and byte-bound asset selection for frozen K-means centres."""
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
import re
from types import MappingProxyType

from seedvision.benchmarking.recipe_catalogue import (
    _image_semantics, _load_source_config, _plantcv_input_stage,
    _plantcv_mapping, _plantcv_provenance,
)
from seedvision.benchmarking.recipes import (
    DependencyContract, LayoutPosition, ParameterContract, PortContract,
    RecipeProvenanceMode, RecipeSpec, RecipeStage,
)
from seedvision.cuda import plantcv_kmeans_model as _model
from seedvision.benchmarking import kmeans_model_artifact as _artifact
from seedvision.model_io.asset_paths import resolve_model_path

NATIVE_KMEANS_VERSION = '1.0.0'
KMEANS_RECIPE_ID = 'plantcv_kmeans_predict_masks'
IMAGE_STAGE_ID = 'external.image_inputs'
MODEL_STAGE_ID = 'external.plantcv_kmeans_centres'
PREDICT_STAGE_ID = 'analysis.kmeans_predict'
MASK_STAGE_ID = 'analysis.kmeans_masks'
CLASSIFIER_STAGE_ID = MASK_STAGE_ID
MODEL_PARAMETER_ID = 'model_source'
CAT_LIST_PARAMETER_ID = 'cat_list'
SOURCE_PATH = 'plantcv/plantcv/kmeans_classifier.py'
PREDICT_SYMBOL = 'plantcv.plantcv.kmeans_classifier.predict_kmeans'
MASK_SYMBOL = 'plantcv.plantcv.kmeans_classifier.mask_kmeans'
MODEL_FIELDS = (
    'path', 'source_sha256', 'source_bytes', 'centres_sha256',
    'patch_size', 'channels', 'cluster_count', 'preprocessing',
)


def model_source_for_path(path):
    """Build one saved selection from a bounded, parsed numeric-centres file."""
    resolved = Path(path).expanduser().resolve(strict=True)
    model = _artifact.load_kmeans_artifact(resolved).model
    _model.validate_kmeans_model(model)
    return {
        'path': str(resolved), 'source_sha256': model.source_sha256,
        'source_bytes': model.source_bytes, 'centres_sha256': model.centres_sha256,
        'patch_size': model.patch_size, 'channels': model.channels,
        'cluster_count': len(model.centres), 'preprocessing': _model.PREPROCESSING,
    }


def _validated_selection(value, *, resolved_path=None):
    if not isinstance(value, Mapping) or set(value) != set(MODEL_FIELDS):
        raise ValueError('Select one complete frozen K-means model asset before running this recipe.')
    path = value['path']
    if type(path) is not str or not path or path != path.strip():
        raise ValueError('The selected model path must be trimmed text.')
    for field in ('source_sha256', 'centres_sha256'):
        if type(value[field]) is not str or re.fullmatch(r'[0-9a-f]{64}', value[field]) is None:
            raise ValueError(f'The selected model {field} must be a lowercase SHA-256.')
    if (type(value['source_bytes']) is not int
            or not 1 <= value['source_bytes'] <= _model.MAX_MODEL_BYTES
            or type(value['patch_size']) is not int or not 1 <= value['patch_size'] <= 31
            or type(value['channels']) is not int or value['channels'] not in (1, 3)
            or type(value['cluster_count']) is not int
            or not 1 <= value['cluster_count'] <= _model.MAX_CLUSTERS
            or value['preprocessing'] != _model.PREPROCESSING):
        raise ValueError('Selected model geometry, size or preprocessing identity is unsupported.')
    feature_count = value['patch_size'] * value['patch_size'] * value['channels']
    if (feature_count > _model.MAX_FEATURES
            or feature_count * value['cluster_count'] > _model.MAX_CENTRE_VALUES):
        raise ValueError('Selected model exceeds the bounded feature or centre-table domain.')
    try:
        canonical = resolve_model_path(path, resolved_path)
    except (OSError, RuntimeError, ValueError) as error:
        raise ValueError(f'The selected K-means model path is unavailable: {error}') from error
    return canonical


def _check_selected_model(value, model, *, resolved_path=None):
    path = _validated_selection(value, resolved_path=resolved_path)
    _model.validate_kmeans_model(model)
    if (model.source_sha256 != value['source_sha256']
            or model.source_bytes != value['source_bytes']
            or model.centres_sha256 != value['centres_sha256']
            or model.patch_size != value['patch_size'] or model.channels != value['channels']
            or len(model.centres) != value['cluster_count']):
        raise ValueError('Selected model file changed; choose it again to bind the new bytes.')
    return path


def load_selected_model(value, *, resolved_path=None):
    """Reread and validate selected file contents before any cache lookup/reuse."""
    model = _artifact.load_kmeans_artifact(_validated_selection(value, resolved_path=resolved_path)).model
    _check_selected_model(value, model, resolved_path=resolved_path)
    return model


def selected_model_identity(value, model, *, resolved_path=None):
    """Freeze facts checked against selected bytes and the ordered centre table."""
    _check_selected_model(value, model, resolved_path=resolved_path)
    return MappingProxyType({
        'path': value['path'], 'source_sha256': model.source_sha256,
        'source_bytes': model.source_bytes, 'centres_sha256': model.centres_sha256,
        'patch_size': model.patch_size, 'channels': model.channels,
        'cluster_count': len(model.centres), 'preprocessing': _model.PREPROCESSING,
        'reference_version': '4.11.3',
        'reference_source_sha256': '09019da00e0713c51cff31ffd240f7bf8aa0b4e228f6d88f2bed9c1d40430392',
    })


def build_native_kmeans_recipe():
    config = _load_source_config()
    provenance = replace(_plantcv_provenance(config),
        mode=RecipeProvenanceMode.NATIVE_COMPATIBLE,
        backend_id='seedvision.cuda.plantcv_kmeans_predict', backend_version=NATIVE_KMEANS_VERSION,
        evidence_note='Frozen-centre CUDA prediction and masks for pinned PlantCV 4.11.3 sigma5. '
            'Selected centres are numeric metadata; native training and biological validity are separate.')
    predict_mapping = (_plantcv_mapping(config, SOURCE_PATH, PREDICT_SYMBOL),)
    mask_mapping = (_plantcv_mapping(config, SOURCE_PATH, MASK_SYMBOL),)
    model_semantics = {'representation': 'validated immutable ordered float64 centre rows',
        'identity': 'selected source bytes, numeric digest, patch geometry and preprocessing'}
    identity_semantics = {'representation': 'immutable reread file/table identity'}
    label_semantics = {'representation': 'CUDA uint8 HW cluster-ID raster',
        'range': '0..cluster_count-1', 'order': 'estimator centre order'}
    masks_semantics = {'representation': 'ordered mapping to CUDA uint8 HW 0/255 masks',
        'selection': 'null returns decimal IDs0..k-1; explicit nonempty categories return the public '
            'raw union wrapped as {"selected": tensor}. Duplicate/absent IDs retain source semantics.'}
    image_stage = replace(_plantcv_input_stage('pcv_kmeans'), stage_id=IMAGE_STAGE_ID)
    model_stage = RecipeStage(
        stage_id=MODEL_STAGE_ID, title='Selected frozen K-means centres',
        operation_id='external.plantcv.kmeans_centres',
        description='Reread and validate the selected bounded model before image preparation/cache reuse.',
        ports=(PortContract(name='model', direction='output', data_type='KMeansModel',
                   label='Validated numeric centres', semantics=model_semantics),
               PortContract(name='model_identity', direction='output', data_type='ModelIdentity',
                   label='Bound model identity', semantics=identity_semantics)),
        parameters=(ParameterContract(parameter_id=MODEL_PARAMETER_ID, value_type='json',
            value=None, affects_calculation=True, editable=True,
            description='One bounded numeric-centres JSON asset bound to local path, file/table hashes, '
                'patch geometry and sigma5 preprocessing. Saved recipes reference this file.'),),
        source_mappings=predict_mapping)
    predict_stage = RecipeStage(
        stage_id=PREDICT_STAGE_ID, title='PlantCV K-means prediction',
        operation_id='analysis.kmeans_predict',
        description='Use prepared gray for a one-channel model or CUDA RGB-to-BGR for a three-channel '
            'model, then apply sigma5 uint16 patch features and the frozen centres.',
        ports=(PortContract(name='rgb_image', direction='input', data_type='RGBImage',
                   label='RGB image', semantics={**_image_semantics('RGB'),
                       'channel_boundary': 'RGB uint8 HWC'}),
               PortContract(name='gray_image', direction='input', data_type='GrayImage',
                   label='Grayscale image', semantics=_image_semantics('GRAY')),
               PortContract(name='model', direction='input', data_type='KMeansModel',
                   label='Frozen centre table', semantics=model_semantics),
               PortContract(name='model_identity', direction='input', data_type='ModelIdentity',
                   label='Bound model identity', semantics=identity_semantics),
               PortContract(name='labels', direction='output', data_type='LabelRaster',
                   label='Predicted cluster IDs', semantics=label_semantics),
               PortContract(name='model_identity', direction='output', data_type='ModelIdentity',
                   label='Bound model identity', semantics=identity_semantics)),
        source_mappings=predict_mapping)
    mask_stage = RecipeStage(
        stage_id=MASK_STAGE_ID, title='PlantCV K-means category masks',
        operation_id='analysis.kmeans_masks',
        description='Create ordered per-ID masks or the selected category union from predicted IDs.',
        ports=(PortContract(name='labels', direction='input', data_type='LabelRaster',
                   label='Predicted cluster IDs', semantics=label_semantics),
               PortContract(name='model_identity', direction='input', data_type='ModelIdentity',
                   label='Bound model identity', semantics=identity_semantics),
               PortContract(name='class_masks', direction='output', data_type='ClassMaskMap',
                   label='Category masks', semantics=masks_semantics),
               PortContract(name='class_names', direction='output', data_type='OrderedClassNames',
                   label='Category display names', semantics={'order': 'native adapter result order'}),
               PortContract(name='model_identity', direction='output', data_type='ModelIdentity',
                   label='Bound model identity', semantics=identity_semantics)),
        parameters=(ParameterContract(parameter_id=CAT_LIST_PARAMETER_ID, value_type='json',
            value=None, affects_calculation=True, editable=True,
            description='null returns every cluster mask; a nonempty integer list returns one union '
                'named selected. Duplicates/absent IDs follow PlantCV; empty lists are rejected.'),),
        source_mappings=mask_mapping)
    dependencies = tuple(DependencyContract(source_stage=IMAGE_STAGE_ID, source_port=name,
        target_stage=PREDICT_STAGE_ID, target_port=name, data_type=kind)
        for name, kind in (('rgb_image', 'RGBImage'), ('gray_image', 'GrayImage')))
    dependencies += tuple(DependencyContract(source_stage=MODEL_STAGE_ID, source_port=name,
        target_stage=PREDICT_STAGE_ID, target_port=name, data_type=kind)
        for name, kind in (('model', 'KMeansModel'), ('model_identity', 'ModelIdentity')))
    dependencies += tuple(DependencyContract(source_stage=PREDICT_STAGE_ID, source_port=name,
        target_stage=MASK_STAGE_ID, target_port=name, data_type=kind)
        for name, kind in (('labels', 'LabelRaster'), ('model_identity', 'ModelIdentity')))
    return RecipeSpec(recipe_id=KMEANS_RECIPE_ID, recipe_version=NATIVE_KMEANS_VERSION,
        title='PlantCV frozen K-means prediction and masks',
        description='Apply a selected numeric centre model through pinned sigma5 prediction and category masks.',
        provenance=provenance, stages=(image_stage, model_stage, predict_stage, mask_stage),
        dependencies=dependencies,
        layout=(LayoutPosition(stage_id=IMAGE_STAGE_ID, x=0, y=0),
                LayoutPosition(stage_id=MODEL_STAGE_ID, x=0, y=180),
                LayoutPosition(stage_id=PREDICT_STAGE_ID, x=360, y=60),
                LayoutPosition(stage_id=MASK_STAGE_ID, x=700, y=60)))
