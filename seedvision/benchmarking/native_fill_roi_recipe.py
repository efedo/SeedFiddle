"""Add independent rectangular small-object controls to native working graphs."""

from dataclasses import replace

from .recipes import ParameterContract, RecipeSpec
from . import native_morphology_roi_recipe as morphology

VERSION = '1.0.0'
SUFFIX = '_with_fill_roi'
BASE_IDS = morphology.BASE_IDS + morphology.RECIPE_IDS
RECIPE_IDS = tuple(name + SUFFIX for name in BASE_IDS)
OPERATION = 'morphology.fill'
PARAMETER = 'roi_bounds_xyxy'


def base_recipe_id(recipe_id):
    return recipe_id[:-len(SUFFIX)] if recipe_id in RECIPE_IDS else recipe_id


def canonical_base(recipe_id):
    base = base_recipe_id(recipe_id)
    if base not in BASE_IDS:
        raise ValueError('Select a supported native working seed recipe.')
    if base in morphology.RECIPE_IDS:
        return morphology.build_native_morphology_roi_recipe(morphology.canonical_base(base))
    return morphology.canonical_base(base)


def selected_bounds(value, shape):
    """Normalize JSON bounds without coercing numeric coordinates."""
    from seedvision.cuda.plantcv_fill import _roi_bounds
    if value is None:
        return None
    if type(value) not in (list, tuple):
        raise TypeError('ROI bounds must be null or [x0,y0,x1,y1].')
    return _roi_bounds(tuple(value), *map(int, shape))


def build_native_fill_roi_recipe(recipe):
    """Preserve historical graphs and add crop-local component area filtering."""
    from .native_recipe import _validate_recipe
    from . import native_reader_recipe as reader
    if reader.is_reader_recipe(recipe):
        return reader.compose(recipe, add_fill=True)
    if not isinstance(recipe, RecipeSpec) or recipe.recipe_id not in BASE_IDS:
        raise ValueError('Select a supported native working seed recipe.')
    _validate_recipe(recipe)
    stages = tuple(replace(stage, parameters=stage.parameters + (
        ParameterContract(PARAMETER, 'json', None, True, editable=True,
            description='null for whole image, or nonempty half-open [x0,y0,x1,y1] '
                'inside this node input frame. Count 4-connected objects in the crop; preserve exterior pixels.'),
    )) if stage.operation_id == OPERATION else stage for stage in recipe.stages)
    provenance = replace(recipe.provenance,
        backend_id='seedvision.benchmarking.native_fill_roi_recipe', backend_version=VERSION,
        evidence_note=recipe.provenance.evidence_note + ' Rectangular fill uses the pinned '
            'helper crop/filter/replace rule with default 4-connectivity; arbitrary contours are unsupported. '
            'Native BinaryMask output stays uint8; the public explicit-ROI platform integer result '
            'is normalized without changing its 0/255 values.')
    return replace(recipe, recipe_id=recipe.recipe_id+SUFFIX, recipe_version=VERSION,
        title=recipe.title+' + small-object ROI', provenance=provenance, stages=stages)


def add_morphology_roi(recipe):
    """Canonicalize the reverse desktop action order while retaining selections."""
    from .native_recipe import _validate_recipe
    from . import native_reader_recipe as reader
    if reader.is_reader_recipe(recipe):
        return reader.compose(recipe, add_morphology=True)
    if not isinstance(recipe, RecipeSpec) or recipe.recipe_id not in RECIPE_IDS:
        raise ValueError('Select a supported small-object ROI recipe.')
    _validate_recipe(recipe)
    base_id = base_recipe_id(recipe.recipe_id)
    if base_id not in morphology.BASE_IDS:
        raise ValueError('Morphology ROI controls are already present.')
    selected = next(s for s in recipe.stages if s.operation_id == OPERATION)
    bounds = next(p for p in selected.parameters if p.parameter_id == PARAMETER)
    base = canonical_base(recipe.recipe_id)
    stages = tuple(replace(s, parameters=tuple(p for p in s.parameters if p.parameter_id != PARAMETER))
                   if s.operation_id == OPERATION else s for s in recipe.stages)
    base = replace(base, stages=stages, layout=recipe.layout, description=recipe.description)
    composed = build_native_fill_roi_recipe(morphology.build_native_morphology_roi_recipe(base))
    return replace(composed, stages=tuple(replace(s, parameters=tuple(
        bounds if p.parameter_id == PARAMETER else p for p in s.parameters))
        if s.operation_id == OPERATION else s for s in composed.stages))
