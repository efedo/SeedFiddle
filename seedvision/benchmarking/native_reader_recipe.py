"""Versioned grayscale source selection for uncalibrated seed workflows."""
from dataclasses import replace

from .recipes import ParameterContract, RecipeSpec
from . import native_median_roi_recipe as median
from . import native_fill_roi_recipe as fill
from . import native_morphology_roi_recipe as morphology

VERSION = '1.0.0'
RECIPE_VERSION = '3.0.0'
MEDIAN_RECIPE_VERSION = '3.1.0'
BASE_IDS = tuple(name for name in median.BASE_IDS if name.startswith('plantcv_seed_'))
OPERATION = 'external.image_inputs'
PARAMETER = 'gray_source_mode'
MODES = ('color-derived', 'codec-gray')


def is_reader_recipe(recipe):
    return (isinstance(recipe, RecipeSpec) and recipe.recipe_id in BASE_IDS
            and recipe.recipe_version in (RECIPE_VERSION, MEDIAN_RECIPE_VERSION))


def selected_mode(recipe):
    if not is_reader_recipe(recipe):
        return MODES[0]
    source = next(s for s in recipe.stages if s.operation_id == OPERATION)
    value = recipe.parameters_for_runner(source.stage_id)[PARAMETER]
    if type(value) is not str or value not in MODES:
        raise ValueError('Select color-derived or codec-gray grayscale preparation.')
    return value


def _upgrade(recipe):
    """Pure canonical construction; the public builder performs admission."""
    stages = tuple(replace(stage, parameters=stage.parameters + (
        ParameterContract(PARAMETER, 'string', MODES[0], True, editable=True, choices=MODES,
            description='color-derived converts decoded color on CUDA; codec-gray '
                'decodes the same image bytes with OpenCV IMREAD_GRAYSCALE. Both retain the full decoded frame.'),
    )) if stage.operation_id == OPERATION else stage for stage in recipe.stages)
    provenance = replace(recipe.provenance,
        backend_id='seedvision.benchmarking.native_reader_recipe', backend_version=VERSION,
        evidence_note=recipe.provenance.evidence_note + ' Explicit grayscale source selection '
            'uses the pinned color or grayscale decoder on one immutable byte snapshot; '
            'calibrated recipes continue deriving grayscale from corrected RGB.')
    return replace(recipe, recipe_version=(MEDIAN_RECIPE_VERSION
        if recipe.recipe_version == median.RECIPE_VERSION else RECIPE_VERSION),
        title=recipe.title+' + grayscale reader', provenance=provenance, stages=stages)


def build_native_reader_recipe(recipe):
    from .native_recipe import _validate_recipe
    if not isinstance(recipe, RecipeSpec) or recipe.recipe_id not in BASE_IDS:
        raise ValueError('Select an uncalibrated native working seed recipe.')
    _validate_recipe(recipe)
    if is_reader_recipe(recipe):
        raise ValueError('Grayscale reader controls are already present.')
    return _upgrade(recipe)


def compose(recipe, *, add_median=False, add_fill=False, add_morphology=False):
    """Rebuild one canonical profile and retain every existing selected value."""
    from .native_recipe import _validate_recipe
    if not is_reader_recipe(recipe):
        raise ValueError('Select a supported grayscale reader recipe.')
    _validate_recipe(recipe)
    has_fill = recipe.recipe_id in fill.RECIPE_IDS
    without_fill = fill.base_recipe_id(recipe.recipe_id)
    has_morphology = without_fill in morphology.RECIPE_IDS
    has_median = recipe.recipe_version == MEDIAN_RECIPE_VERSION
    if (add_fill and has_fill or add_morphology and has_morphology or add_median and has_median):
        raise ValueError('Requested ROI controls are already present.')
    base = morphology.canonical_base(morphology.base_recipe_id(recipe.recipe_id))
    if has_morphology or add_morphology:
        base = morphology.build_native_morphology_roi_recipe(base)
    if has_fill or add_fill:
        base = fill.build_native_fill_roi_recipe(base)
    if has_median or add_median:
        base = median._upgrade(base)
    composed = _upgrade(base)
    values = {(s.stage_id, p.parameter_id): p.value for s in recipe.stages for p in s.parameters}
    destinations = {(s.stage_id, p.parameter_id) for s in composed.stages for p in s.parameters}
    if values.keys()-destinations:
        raise ValueError('Composition could not preserve every selected calculation parameter.')
    stages = tuple(replace(s, parameters=tuple(replace(p, value=values.get(
        (s.stage_id, p.parameter_id), p.value)) for p in s.parameters)) for s in composed.stages)
    composed = replace(composed, stages=stages, layout=recipe.layout, description=recipe.description)
    _validate_recipe(composed)
    return composed
