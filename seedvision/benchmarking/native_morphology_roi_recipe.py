"""Add independent rectangular morphology controls to native working graphs."""

from dataclasses import replace

from .recipes import ParameterContract, RecipeSpec
from . import native_watershed_diagnostic_recipe as diagnostic

VERSION = '1.0.0'
SUFFIX = '_with_morphology_roi'
BASE_IDS = diagnostic.BASE_IDS + diagnostic.RECIPE_IDS
RECIPE_IDS = tuple(name + SUFFIX for name in BASE_IDS)
OPERATIONS = ('morphology.erode', 'morphology.dilate')
PARAMETER = 'roi_bounds_xyxy'


def base_recipe_id(recipe_id):
    from .native_fill_roi_recipe import base_recipe_id as fill_base
    recipe_id = fill_base(recipe_id)
    return recipe_id[:-len(SUFFIX)] if recipe_id in RECIPE_IDS else recipe_id


def canonical_base(recipe_id):
    from .recipe_catalogue import build_recipe_catalogue
    from .native_recipe import build_native_seed_recipe
    from .native_calibrated_seed_recipe import build_native_calibrated_seed_recipe
    base = base_recipe_id(recipe_id)
    if base not in BASE_IDS:
        raise ValueError('Select a supported native working seed recipe.')
    if base in diagnostic.RECIPE_IDS:
        return diagnostic.build_native_watershed_diagnostic_recipe(diagnostic.canonical_base(base))
    if base.startswith('plantcv_calibrated_seed_'):
        return build_native_calibrated_seed_recipe(base.rsplit('_', 1)[1])
    return build_native_seed_recipe(build_recipe_catalogue().recipe(base[:-len('_working')]))


def selected_bounds(value, shape):
    """Normalize JSON bounds without coercing numeric coordinates."""
    from seedvision.cuda.plantcv_morphology import _roi_bounds
    if value is None:
        return None
    if type(value) not in (list, tuple):
        raise TypeError('ROI bounds must be null or [x0,y0,x1,y1].')
    return _roi_bounds(tuple(value), *map(int, shape))


def build_native_morphology_roi_recipe(recipe):
    """Keep historical definitions intact and add source crop/filter/replace controls."""
    from .native_recipe import _validate_recipe
    from . import native_reader_recipe as reader
    if reader.is_reader_recipe(recipe):
        return reader.compose(recipe, add_morphology=True)
    if not isinstance(recipe, RecipeSpec) or recipe.recipe_id not in BASE_IDS:
        raise ValueError('Select a supported native working seed recipe.')
    _validate_recipe(recipe)
    stages = tuple(replace(stage, parameters=stage.parameters + (
        ParameterContract(PARAMETER, 'json', None, True, editable=True,
            description='null for whole image, or nonempty half-open [x0,y0,x1,y1] '
                'inside this node input frame. Filter the crop independently; preserve exterior pixels.'),
    )) if stage.operation_id in OPERATIONS else stage for stage in recipe.stages)
    provenance = replace(recipe.provenance,
        backend_id='seedvision.benchmarking.native_morphology_roi_recipe', backend_version=VERSION,
        evidence_note=recipe.provenance.evidence_note + ' Rectangular morphology uses the pinned '
            'helper crop/filter/replace rule in the node input frame; arbitrary contours are unsupported.')
    return replace(recipe, recipe_id=recipe.recipe_id+SUFFIX, recipe_version=VERSION,
        title=recipe.title+' + morphology ROI', provenance=provenance, stages=stages)
