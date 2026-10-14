"""Add the public watershed debug image as an independent native graph branch."""
from dataclasses import replace

from .recipes import (DependencyContract, LayoutPosition, ParameterContract,
                      PortContract, RecipeSpec, RecipeStage, SourceMapping, SourceResolution)
from seedvision.cuda import plantcv_watershed_diagnostic as drawing

VERSION = '1.0.0'
SUFFIX = '_with_watershed_diagnostic'
BASE_IDS = tuple('plantcv_seed_' + kind + '_working' for kind in ('binary', 'mean', 'otsu')) + tuple(
    'plantcv_calibrated_seed_' + kind for kind in ('binary', 'mean', 'otsu', 'range'))
RECIPE_IDS = tuple(name + SUFFIX for name in BASE_IDS)
STAGE = 'pcv_watershed_debug_diagnostic'
OPERATION = 'native.watershed_debug_diagnostic'


def base_recipe_id(recipe_id):
    from .native_morphology_roi_recipe import base_recipe_id as morphology_base
    recipe_id = morphology_base(recipe_id)
    return recipe_id[:-len(SUFFIX)] if recipe_id in RECIPE_IDS else recipe_id


def canonical_base(recipe_id):
    from .native_recipe import build_native_seed_recipe
    from .recipe_catalogue import build_recipe_catalogue
    from .native_calibrated_seed_recipe import build_native_calibrated_seed_recipe
    base = base_recipe_id(recipe_id)
    if base not in BASE_IDS:
        raise ValueError('Select a supported working or calibrated seed recipe.')
    if base.startswith('plantcv_calibrated_seed_'):
        return build_native_calibrated_seed_recipe(base.rsplit('_', 1)[1])
    return build_native_seed_recipe(build_recipe_catalogue().recipe(base[:-len('_working')]))


def build_native_watershed_diagnostic_recipe(recipe):
    """Preserve selected base settings, old factories and historical graph identities."""
    from .native_recipe import _validate_recipe
    if not isinstance(recipe, RecipeSpec) or recipe.recipe_id not in BASE_IDS:
        raise ValueError('Select a supported working or calibrated seed recipe.')
    _validate_recipe(recipe)
    flood = next(stage for stage in recipe.stages
                 if stage.operation_id == 'reference.internal.watershed.masked_flood')
    rgb_edge = next(edge for edge in recipe.dependencies
                    if edge.target_stage == flood.stage_id and edge.target_port == 'rgb')
    rgb_port = next(p for p in flood.ports if p.name == 'rgb' and p.direction == 'input')
    label_port = next(p for p in flood.ports if p.name == 'labels' and p.direction == 'output')
    mapping = SourceMapping(reference_id='plantcv', reference_version='4.11.3',
        resolution=SourceResolution.PINNED, source_path='plantcv/plantcv/watershed.py',
        source_symbol='plantcv.plantcv.watershed.watershed_segmentation: debug contour drawing',
        source_url='https://github.com/danforthcenter/plantcv/blob/v4.11.3/plantcv/plantcv/watershed.py',
        source_sha256=drawing.SOURCE_SHA256, license_id='MPL-2.0',
        note='Debug side effect, not a public returned raster. A local explicit seed reproduces '
             'the source shuffle with the same ambient legacy RNG seed.')
    stage = RecipeStage(STAGE, 'Watershed debug image', OPERATION,
        description='Draw every source contour with the shuffled palette on a private RGB copy. '
            'Labels and object count remain analytical outputs. Source palette[label_id] indexing '
            'is preserved; unsupported/resource-limited drawing is unavailable.',
        ports=(replace(rgb_port, name='image'), replace(label_port, direction='input'),
            replace(rgb_port, name='diagnostic', direction='output', label='Native watershed debug RGB',
                    semantics={**dict(rgb_port.semantics), 'role': 'debug-only native diagnostic'}),
            PortContract('metadata', 'output', 'Metadata', 'Palette and drawing provenance')),
        parameters=(ParameterContract('color_scale', 'string', 'gist_rainbow', True,
                editable=True, description='Registered Matplotlib colormap used for source byte sampling.'),
            ParameterContract('random_seed', 'integer', 0, True, editable=True, minimum=0,
                maximum=0xffffffff, description='Local legacy NumPy seed for reproducible source palette shuffle.'),
            ParameterContract('line_thickness', 'integer', 5, True, editable=True, minimum=1,
                maximum=32767, units='pixels', description='Source drawContours closed LINE_8 stroke width.')),
        source_mappings=(mapping,))
    edges = recipe.dependencies + (
        DependencyContract(rgb_edge.source_stage, rgb_edge.source_port, STAGE, 'image', rgb_port.data_type),
        DependencyContract(flood.stage_id, 'labels', STAGE, 'labels', label_port.data_type))
    position = next((p for p in recipe.layout if p.stage_id == flood.stage_id), None)
    layout = recipe.layout + (LayoutPosition(STAGE,
        max((p.x for p in recipe.layout), default=0) + 280, 0 if position is None else position.y),)
    provenance = replace(recipe.provenance,
        backend_id='seedvision.benchmarking.native_watershed_diagnostic_recipe', backend_version=VERSION,
        evidence_note='Native seed graph with a separate reproducible watershed debug image. '
            'The original public return remains labels only; no scientific validation is implied.')
    return replace(recipe, recipe_id=recipe.recipe_id+SUFFIX, recipe_version=VERSION,
        title=recipe.title+' + watershed debug image', provenance=provenance,
        stages=recipe.stages+(stage,), dependencies=edges, layout=layout)
