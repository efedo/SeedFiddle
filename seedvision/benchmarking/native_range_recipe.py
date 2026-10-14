"""Separate working graph for the pinned PlantCV custom-range operation."""
from dataclasses import replace

import torch

from seedvision.benchmarking.recipe_catalogue import (
    _image_semantics, _load_source_config, _mask_semantics,
    _plantcv_input_stage, _plantcv_mapping, _plantcv_provenance,
)
from seedvision.benchmarking.recipes import (
    DependencyContract, LayoutPosition, ParameterContract, PortContract,
    RecipeProvenanceMode, RecipeSpec, RecipeStage,
)
from seedvision.cuda import plantcv_range as native


RECIPE_ID = 'plantcv_threshold_custom_range'
VERSION = '1.0.0'
OPERATION = 'threshold.custom_range'
STAGE_ID = RECIPE_ID + '_threshold'


def build_native_range_recipe():
    config = _load_source_config()
    provenance = replace(_plantcv_provenance(config),
        mode=RecipeProvenanceMode.NATIVE_COMPATIBLE,
        backend_id='seedvision.cuda.plantcv_range', backend_version=VERSION,
        evidence_note='Pinned single-operation byte-range adapter; acquisition settings and scientific validity are not established.')
    source = _plantcv_input_stage(RECIPE_ID)
    rgb = {**_image_semantics('RGB'), 'channel_boundary': 'RGB uint8 HWC'}
    stage = RecipeStage(STAGE_ID, 'PlantCV custom-range threshold', OPERATION,
        description='Inclusive byte bounds in the selected channel space. The RGB selection follows the pinned B,G,R storage-slot order; CMYK preserves its historical C/Y swap. The masked output retains original RGB pixels, not converted channel values. Reversed bounds produce an empty mask.',
        ports=(PortContract('rgb', 'input', 'RGBImage', 'Original RGB image', semantics=rgb),
            PortContract('mask', 'output', 'BinaryMask', 'Binary foreground mask', semantics=_mask_semantics()),
            PortContract('masked_rgb', 'output', 'RGBImage', 'Masked original RGB image', semantics=rgb)),
        parameters=(
            ParameterContract('channel', 'string', 'RGB', True, editable=True,
                choices=('RGB', 'HSV', 'LAB', 'CMYK', 'GRAY'),
                description='Channel space. RGB bounds use B,G,R slot order; CMYK uses the pinned C/Y swap.'),
            ParameterContract('lower', 'json', [0, 0, 0], True, editable=True,
                description='Inclusive integer byte lower bounds: 3 values for RGB/HSV/LAB, 4 for CMYK, 1 for GRAY.'),
            ParameterContract('upper', 'json', [255, 255, 255], True, editable=True,
                description='Inclusive integer byte upper bounds, with the same channel count as lower.')),
        source_mappings=(_plantcv_mapping(config,
            'plantcv/plantcv/threshold/threshold_methods.py',
            'plantcv.plantcv.threshold.threshold_methods.custom_range',
            note='Pinned public function; original RGB masking and inclusive inRange bounds.'),))
    return RecipeSpec(RECIPE_ID, VERSION, 'PlantCV custom-range threshold',
        'Working native range threshold with mask and masked RGB outputs; separate from the historical reference catalogue.',
        provenance, (source, stage),
        (DependencyContract(source.stage_id, 'rgb_image', STAGE_ID, 'rgb', 'RGBImage'),),
        layout=(LayoutPosition(source.stage_id, 0., 0.), LayoutPosition(STAGE_ID, 260., 0.)))


def retained_bound(pixels):
    admission = native._range_admission(pixels)
    # Include source boundaries, all output/conversion buffers, tile overlap,
    # compact lookup tables and comparison scratch. Same envelope as replay.
    return max(128*pixels+128*1024*1024,
        17*pixels+admission['tile_scratch_bytes']+admission['fixed_scratch_bytes'])


def execute_stage(inputs, parameters, cancel):
    rgb = inputs['rgb']
    native._range_controls(**parameters)
    native._range_admission(rgb.shape[0]*rgb.shape[1])
    rgb.record_stream(torch.cuda.current_stream(rgb.device))
    bgr = torch.empty((1, 3, *rgb.shape[:2]), dtype=torch.uint8, device=rgb.device)
    for channel in range(3):
        bgr[0, channel].copy_(rgb[..., 2-channel])
    mask, masked = native.threshold_custom_range(bgr, **parameters, cancel=cancel)
    del bgr
    masked_rgb = torch.empty_like(rgb)
    for channel in range(3):
        masked_rgb[..., channel].copy_(masked[0, 2-channel])
    return {'mask': mask, 'masked_rgb': masked_rgb}
