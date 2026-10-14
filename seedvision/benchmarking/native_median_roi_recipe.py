"""Versioned median ROI controls on existing native working graph identities."""

from dataclasses import replace

from .recipes import ParameterContract, RecipeSpec
from . import native_fill_roi_recipe as fill
from . import native_morphology_roi_recipe as morphology

VERSION = '1.0.0'
RECIPE_VERSION = '2.0.0'
BASE_IDS = fill.BASE_IDS + fill.RECIPE_IDS
OPERATION = 'filter.median_blur'
PARAMETER = 'roi_bounds_xyxy'


def is_median_roi_recipe(recipe):
    return (isinstance(recipe, RecipeSpec) and recipe.recipe_id in BASE_IDS
            and (recipe.recipe_version == RECIPE_VERSION
                 or recipe.recipe_id.startswith('plantcv_seed_') and recipe.recipe_version == '3.1.0'))


def canonical_base(recipe_id):
    if recipe_id in fill.RECIPE_IDS:
        return fill.build_native_fill_roi_recipe(fill.canonical_base(recipe_id))
    if recipe_id in morphology.RECIPE_IDS:
        return morphology.build_native_morphology_roi_recipe(morphology.canonical_base(recipe_id))
    if recipe_id in morphology.BASE_IDS:
        return morphology.canonical_base(recipe_id)
    raise ValueError('Select a supported native working seed recipe.')


def selected_bounds(value, shape):
    """Normalize JSON rectangles without coercing coordinates or geometry."""
    from seedvision.cuda.plantcv_median import _roi_bounds
    if value is None:
        return None
    if type(value) not in (list, tuple):
        raise TypeError('ROI bounds must be null or [x0,y0,x1,y1].')
    return _roi_bounds(tuple(value), *map(int, shape))


def _upgrade(recipe):
    """Pure canonical construction; admission belongs to the public builder."""
    stages = tuple(replace(stage, parameters=stage.parameters + (
        ParameterContract(PARAMETER, 'json', None, True, editable=True,
            description='null for whole image, or nonempty half-open [x0,y0,x1,y1] '
                'inside this node input frame. Median-filter the crop independently; preserve exterior pixels.'),
    )) if stage.operation_id == OPERATION else stage for stage in recipe.stages)
    provenance = replace(recipe.provenance,
        backend_id='seedvision.benchmarking.native_median_roi_recipe', backend_version=VERSION,
        evidence_note=recipe.provenance.evidence_note + ' Median ROI uses the pinned helper '
            'crop/filter/replace rule; existing supported kernel and crop geometry limits remain in force.')
    return replace(recipe, recipe_version=RECIPE_VERSION,
        title=recipe.title+' + median ROI', provenance=provenance, stages=stages)


def build_native_median_roi_recipe(recipe):
    """Add one versioned stage contract without introducing new recipe IDs."""
    from .native_recipe import _validate_recipe
    from . import native_reader_recipe as reader
    if reader.is_reader_recipe(recipe):
        return reader.compose(recipe, add_median=True)
    if not isinstance(recipe, RecipeSpec) or recipe.recipe_id not in BASE_IDS:
        raise ValueError('Select a supported native working seed recipe.')
    _validate_recipe(recipe)
    if is_median_roi_recipe(recipe):
        raise ValueError('Median ROI controls are already present.')
    return _upgrade(recipe)


def _without_median(recipe):
    from .native_recipe import _validate_recipe
    if not is_median_roi_recipe(recipe):
        raise ValueError('Select a supported median ROI recipe.')
    _validate_recipe(recipe)
    base = canonical_base(recipe.recipe_id)
    stages = tuple(replace(stage, parameters=tuple(
        p for p in stage.parameters if p.parameter_id != PARAMETER))
        if stage.operation_id == OPERATION else stage for stage in recipe.stages)
    return replace(base, stages=stages, layout=recipe.layout, description=recipe.description)


def _restore_median(recipe, selected):
    bounds = next(p for stage in selected.stages if stage.operation_id == OPERATION
                  for p in stage.parameters if p.parameter_id == PARAMETER)
    upgraded = build_native_median_roi_recipe(recipe)
    return replace(upgraded, stages=tuple(replace(stage, parameters=tuple(
        bounds if p.parameter_id == PARAMETER else p for p in stage.parameters))
        if stage.operation_id == OPERATION else stage for stage in upgraded.stages))


def add_fill_roi(recipe):
    """Preserve the median selection when fill controls are added afterward."""
    from . import native_reader_recipe as reader
    if reader.is_reader_recipe(recipe):
        return reader.compose(recipe, add_fill=True)
    base = _without_median(recipe)
    return _restore_median(fill.build_native_fill_roi_recipe(base), recipe)


def add_morphology_roi(recipe):
    """Preserve all selections in either existing fill/morphology action order."""
    from . import native_reader_recipe as reader
    if reader.is_reader_recipe(recipe):
        return reader.compose(recipe, add_morphology=True)
    base = _without_median(recipe)
    composed = (fill.add_morphology_roi(base) if base.recipe_id in fill.RECIPE_IDS
                else morphology.build_native_morphology_roi_recipe(base))
    return _restore_median(composed, recipe)
