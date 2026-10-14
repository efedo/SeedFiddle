"""Additive executable SMPTS paper port with explicit, provisional choices."""
from dataclasses import replace

from .checkpoints import _freeze_json
from .recipe_catalogue import build_recipe_catalogue
from .recipes import (DependencyContract, LayoutPosition, ParameterContract,
    PortContract, RecipeSpec, RecipeStage, SourceMapping, SourceResolution)
from seedvision.cuda import smpts_paper as paper

VERSION = '1.0.0-provisional'
RECIPE_ID = 'smpts_paper_fixed_mean'
SOURCE = 'paper_source'
MEAN = 'paper_adaptive_mean'
MEDIAN = 'paper_binary_median'
GLOBAL = 'paper_fixed_global'
INTERSECTION = 'paper_intersection'
COMPONENTS = 'paper_source_components'
CONTACTS = 'paper_contact_selection'
EROSION = 'paper_targeted_erosion'
RELOCALIZE = 'paper_relocalized_components'
CROPS = 'paper_original_rgb_crops'


def retained_bound(pixels):
    return paper.retained_bound(pixels)


def is_executable_paper_recipe(recipe):
    return isinstance(recipe, RecipeSpec) and recipe.recipe_id == RECIPE_ID


def _port(name, direction, kind):
    semantics = {}
    if kind in ('BinaryMask', 'InstanceLabels'):
        semantics = {'frame': 'source.decoded', 'background': 0,
            'dtype': 'uint8' if kind == 'BinaryMask' else 'int32'}
        semantics.update({'foreground': 255} if kind == 'BinaryMask' else
            {'connectivity': 8, 'id_order': 'first foreground pixel, row-major'})
    elif kind in ('PackedRGB', 'PackedMask'):
        semantics = {'frame': None, 'dtype': 'uint8', 'layout': 'flat contiguous packed bytes',
            'indexing': 'crop table offset_pixels and crop_shape; RGB interleaved, three bytes per pixel'}
    elif kind in ('GrayImage', 'RGBImage'):
        semantics = {'color_order': 'GRAY' if kind == 'GrayImage' else 'RGB', 'dtype': 'uint8',
            'shape': 'HxW' if kind == 'GrayImage' else 'HxWx3', 'range': (0, 255)}
        if kind == 'RGBImage':
            semantics['channel_boundary'] = 'RGB uint8 HWC'
    return PortContract(name, direction, kind, name.replace('_', ' '), semantics=semantics)


def _stage(identifier, title, operation, inputs, outputs, description, parameters=()):
    mapping = SourceMapping('seedfiddle_smpts_paper_port', VERSION, SourceResolution.PINNED,
        source_path='seedvision/cuda/_smpts_paper.cu', source_symbol=operation,
        source_sha256=paper.CUDA_SOURCE_SHA256,
        note='Executable declared reconstruction; native implementation hash, not original capsule code. '
             f'Manuscript SHA256 {paper.PAPER_SHA256}. Unspecified choices are assumptions.')
    return RecipeStage(identifier, title, 'smpts_paper.'+operation, description=description,
        ports=tuple(_port(n, 'input', k) for n, k in inputs)+tuple(_port(n, 'output', k) for n, k in outputs),
        parameters=parameters, source_mappings=(mapping,))


def build_native_smpts_recipe():
    catalogue = build_recipe_catalogue()
    historical = catalogue.recipe('smpts_paper_reconstruction')
    source = replace(catalogue.recipe('plantcv_seed_binary').stages[0], stage_id=SOURCE,
        title='Prepared source RGB and gray')
    mean = _stage(MEAN, 'Adaptive mean branch', 'mean', (('gray', 'GrayImage'),),
        (('mask', 'BinaryMask'),), 'Mean branch precedes binary median. Offset and polarity are assumptions.', (
        ParameterContract('ksize', 'integer', 25, True, editable=True, minimum=3, maximum=255,
            units='pixels', description='Odd local mean window; the paper selected25.'),
        ParameterContract('offset', 'number', 2.0, True, editable=True, minimum=-255, maximum=255,
            description='Assumed adaptive offset; affects each binary decision.'),
        ParameterContract('object_type', 'string', 'light', True, editable=True, choices=('light', 'dark'))))
    median = _stage(MEDIAN, 'Median on adaptive mask', 'median', (('mask', 'BinaryMask'),),
        (('mask', 'BinaryMask'),), 'Exact binary median with assumed replicated border.', (
        ParameterContract('ksize', 'integer', 5, True, editable=True, minimum=1, maximum=31,
            units='pixels', description='Odd binary median window; the paper selected5.'),))
    fixed = _stage(GLOBAL, 'Fixed global branch', 'fixed', (('gray', 'GrayImage'),),
        (('mask', 'BinaryMask'),), 'This variant explicitly chooses fixed binary. The paper also mentions Otsu ambiguously.', (
        ParameterContract('threshold', 'integer', 100, True, editable=True, minimum=0, maximum=255,
            description='Assumed fixed cutoff; not a recovered original setting.'),
        ParameterContract('object_type', 'string', 'light', True, editable=True, choices=('light', 'dark'))))
    combine = _stage(INTERSECTION, 'Intersect binary branches', 'intersection',
        (('adaptive', 'BinaryMask'), ('global', 'BinaryMask')), (('mask', 'BinaryMask'),),
        'Logical binary intersection preserves0/255; multiplication never overflows uint8.')
    def component_stage(identifier, title):
        return _stage(identifier, title, 'components', (('mask', 'BinaryMask'),),
            (('labels', 'InstanceLabels'), ('components', 'Metadata')),
            'Assumed eight-connectivity and axis-aligned half-open boxes; IDs follow the first source pixel. At most4096 components.')
    source_components = component_stage(COMPONENTS, 'Locate source components')
    contacts = _stage(CONTACTS, 'Screen suspected contacts', 'contacts',
        (('components', 'Metadata'), ('labels', 'InstanceLabels')), (('selected_parent_ids', 'Metadata'),),
        'Select a suspected contact only when axis-aligned box width*height strictly exceeds the assumed cutoff.', (
        ParameterContract('contact_box_area', 'integer', 10_000, True, editable=True, minimum=0,
            maximum=paper.MAX_PIXELS, units='pixels squared',
            description='Assumed strict box-area cutoff, not an original or biologically validated rule.'),))
    erosion = _stage(EROSION, 'Erode only selected parents', 'erosion',
        (('mask', 'BinaryMask'), ('labels', 'InstanceLabels'), ('components', 'Metadata'),
         ('selected_parent_ids', 'Metadata')),
        (('mask', 'BinaryMask'), ('selected_mask', 'BinaryMask'), ('eroded_selected_mask', 'BinaryMask')),
        'One square erosion with assumed replicated border; all unselected source components stay unchanged.', (
        ParameterContract('ksize', 'integer', 13, True, editable=True, minimum=1, maximum=31,
            units='pixels', description='Odd square erosion window; the paper selected13. Shrunk regions are not recovered outlines.'),))
    relocalize = component_stage(RELOCALIZE, 'Relocalize eroded components')
    crops = _stage(CROPS, 'Extract original masked RGB', 'crops',
        (('rgb', 'RGBImage'), ('source_labels', 'InstanceLabels'), ('source_components', 'Metadata'),
         ('final_labels', 'InstanceLabels'), ('final_components', 'Metadata')),
        (('labels', 'InstanceLabels'), ('packed_rgb', 'PackedRGB'), ('packed_mask', 'PackedMask'),
         ('crops', 'Metadata'), ('disappeared_parent_ids', 'Metadata'), ('metadata', 'Metadata')),
        'Packed CUDA crops use original RGB, zeroed outside each final component. Half-open source bounds, offsets, parent lineage and disappeared IDs stay explicit.')
    stages = (source, mean, median, fixed, combine, source_components, contacts, erosion, relocalize, crops)
    edges = (
        (SOURCE, 'gray_image', MEAN, 'gray', 'GrayImage'),
        (MEAN, 'mask', MEDIAN, 'mask', 'BinaryMask'),
        (SOURCE, 'gray_image', GLOBAL, 'gray', 'GrayImage'),
        (MEDIAN, 'mask', INTERSECTION, 'adaptive', 'BinaryMask'),
        (GLOBAL, 'mask', INTERSECTION, 'global', 'BinaryMask'),
        (INTERSECTION, 'mask', COMPONENTS, 'mask', 'BinaryMask'),
        (COMPONENTS, 'components', CONTACTS, 'components', 'Metadata'),
        (COMPONENTS, 'labels', CONTACTS, 'labels', 'InstanceLabels'),
        (INTERSECTION, 'mask', EROSION, 'mask', 'BinaryMask'),
        (COMPONENTS, 'labels', EROSION, 'labels', 'InstanceLabels'),
        (COMPONENTS, 'components', EROSION, 'components', 'Metadata'),
        (CONTACTS, 'selected_parent_ids', EROSION, 'selected_parent_ids', 'Metadata'),
        (EROSION, 'mask', RELOCALIZE, 'mask', 'BinaryMask'),
        (SOURCE, 'rgb_image', CROPS, 'rgb', 'RGBImage'),
        (COMPONENTS, 'labels', CROPS, 'source_labels', 'InstanceLabels'),
        (COMPONENTS, 'components', CROPS, 'source_components', 'Metadata'),
        (RELOCALIZE, 'labels', CROPS, 'final_labels', 'InstanceLabels'),
        (RELOCALIZE, 'components', CROPS, 'final_components', 'Metadata'),
    )
    provenance = replace(historical.provenance, backend_id='seedvision.benchmarking.native_smpts_recipe',
        backend_version=VERSION, evidence_note='Executable provisional fixed-global/mean paper reconstruction. '
        'Paper-supported topology and default windows25/5/13. Offset, polarity, fixed cutoff, '
        'eight-connectivity, axis-aligned boxes, strict box-area cutoff, replicated borders, '
        'square single erosion and masked half-open crops are declared assumptions. '
        'Original capsule unavailable; no original-code conformance or scientific validation.')
    return RecipeSpec(RECIPE_ID, VERSION, 'SMPTS paper reconstruction — fixed/mean',
        'Provisional paper port with explicit assumptions, targeted erosion and original RGB crops.',
        provenance, stages, tuple(DependencyContract(*e) for e in edges),
        tuple(LayoutPosition(s.stage_id, (i%4)*290, (i//4)*270) for i, s in enumerate(stages)))


def execute_stage(operation, inputs, parameters, cancel):
    try:
        if operation == 'smpts_paper.mean':
            return {'mask': paper.thresholds.threshold_mean(inputs['gray'], **parameters)}
        if operation == 'smpts_paper.fixed':
            return {'mask': paper.thresholds.threshold_binary(inputs['gray'], **parameters)}
        if operation == 'smpts_paper.median':
            return {'mask': paper.binary_median(inputs['mask'], cancel=cancel, **parameters)}
        if operation == 'smpts_paper.intersection':
            return {'mask': paper.intersection(inputs['adaptive'], inputs['global'], cancel=cancel)}
        if operation == 'smpts_paper.components':
            return paper.components(inputs['mask'], cancel=cancel)
        if operation == 'smpts_paper.contacts':
            return {'selected_parent_ids': paper.select_contacts(inputs['components'], inputs['labels'].shape,
                parameters['contact_box_area'])}
        if operation == 'smpts_paper.erosion':
            return paper.targeted_erosion(inputs['mask'], inputs['labels'], inputs['components'],
                inputs['selected_parent_ids'], cancel=cancel, **parameters)
        if operation == 'smpts_paper.crops':
            output = paper.pack_crops(inputs['rgb'], inputs['source_labels'], inputs['source_components'],
                inputs['final_labels'], inputs['final_components'], cancel=cancel)
            return {'labels': inputs['final_labels'], **output, 'metadata': _freeze_json({
                'status': 'provisional_paper_reconstruction', 'profile': paper.PROFILE,
                'paper_sha256': paper.PAPER_SHA256, 'original_execution': False, 'scientific_validation': False,
                'component_connectivity': 8, 'contact_metric': 'axis-aligned bbox width*height',
                'contact_comparison': 'strict > selected cutoff', 'median_border': 'replicate',
                'erosion': 'square, one iteration, replicate', 'bounds': 'source-frame half-open xyxy',
                'crop_pixels': 'original RGB masked by final component; packed row-major',
                'crop_count': len(output['crops']), 'not_recovered_measurement_outlines': True},
                name='paper reconstruction assumptions')}
    except (paper.PaperResourceError, paper.thresholds.AdaptiveMeanPrecisionError) as error:
        raise NotImplementedError(str(error)) from error
    raise NotImplementedError('Unknown declared paper-port operation.')
