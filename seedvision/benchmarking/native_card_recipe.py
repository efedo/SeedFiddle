"""Editable automatic-card graph; bounded primitives and exact cache boundaries."""
from dataclasses import replace
import hashlib
import sys

import numpy as np
import torch

from .checkpoints import _freeze_json
from .recipe_catalogue import _load_source_config, _plantcv_input_stage, _plantcv_provenance, _plantcv_mapping
from .recipes import (DependencyContract, LayoutPosition, ParameterContract, PortContract,
                      RecipeSpec, RecipeStage, RecipeProvenanceMode, SourceMapping, SourceResolution)
from seedvision.cuda import (plantcv_card_detection as card, plantcv_card_preprocess as preprocess,
    plantcv_contours as contours, plantcv_minrect as minrect, plantcv_card_kernel_pins as kernel_pins,
    plantcv_color_card as color_card, plantcv_correction as correction)

RECIPE_ID = 'plantcv_automatic_color_card'
VERSION = '1.0.0'
PREFIX = 'pcv_auto_card_'
PREPROCESS = PREFIX + 'preprocess'
GEOMETRY = PREFIX + 'geometry'
MASK = PREFIX + 'mask'
SCALE = PREFIX + 'scale'
MEANS = PREFIX + 'means'
STANDARD = PREFIX + 'standard'
AFFINE = PREFIX + 'affine'


def _source(symbol, note=''):
    return SourceMapping(reference_id='plantcv', reference_version='4.11.3',
        resolution=SourceResolution.PINNED,
        source_path='plantcv/plantcv/transform/detect_color_card.py',
        source_symbol='plantcv.plantcv.transform.detect_color_card.' + symbol,
        source_url='https://github.com/danforthcenter/plantcv/blob/v4.11.3/plantcv/plantcv/transform/detect_color_card.py',
        source_sha256=card.DETECT_SOURCE_SHA256, license_id='MIT', note=note)


def _port(name, direction, data_type, semantics=None):
    return PortContract(name, direction, data_type, name.replace('_', ' '), semantics=semantics or {})


def build_native_card_recipe():
    """Add a working graph without changing historical catalogue recipes."""
    config = _load_source_config()
    provenance = replace(_plantcv_provenance(config), mode=RecipeProvenanceMode.NATIVE_COMPATIBLE,
        backend_id='seedvision.benchmarking.native_card_recipe', backend_version=VERSION,
        evidence_note='Editable decomposition of pinned automatic card detection and affine correction. '
            'Defaults use standard orientation3 as in auto_correct_color; selecting another orientation '
            'explicitly composes std_color_matrix with affine_color_correction. No scientific validation is implied.')
    source = _plantcv_input_stage(PREFIX.rstrip('_'))
    rgb = {'color_order':'RGB','dtype':'uint8','shape':'HxWx3','range':[0,255],
           'channel_boundary':'RGB uint8 HWC'}
    bgr = {'color_order':'BGR','dtype':'uint8','shape':'HxWx3','range':[0,255]}
    local = {**bgr,'frame':'roi_bounds_xyxy'}
    local_gray = {'dtype':'uint8','shape':'HxW','frame':'roi_bounds_xyxy'}
    labels = {'dtype':'float64','shape':'HxW','background':0,'positive_values':'chip IDs10..240'}
    stages = [source, RecipeStage(PREFIX+'bgr', 'Card image', 'color_card.bgr',
        ports=(_port('rgb','input','RGBImage',rgb),_port('bgr','output','BGRImage',bgr)),
        description='Convert the explicit RGB boundary to the BGR bytes consumed by the pinned detector.',
        source_mappings=(_source('_color_card_detection','Explicit native RGB-to-BGR boundary.'),))]
    stages.append(RecipeStage(PREPROCESS, 'Find chip edges', 'color_card.preprocess',
        ports=(_port('bgr','input','BGRImage',bgr),_port('roi_bgr','output','BGRImage',local),
               *(_port(name,'output','GrayImage',local_gray) for name in ('gray','blurred','threshold')),
               _port('bounds','output','RegionBounds')),
        parameters=(ParameterContract('roi_bounds_xyxy','json',None,True,editable=True,
            description='null for full frame, or half-open [x0,y0,x1,y1]. Crop before filtering.'),
            ParameterContract('adaptive_method','integer',1,True,editable=True,minimum=0,maximum=1,
                description='0: adaptive mean; 1: adaptive Gaussian. Offset2 is fixed by the source.'),
            ParameterContract('block_size','integer',51,True,editable=True,minimum=3,maximum=255,
                description='Odd adaptive kernel3..255; uncertain decisions are explicitly unavailable.')),
        description='Crop the selected ROI, then CUDA grayscale, byte Gaussian11 and inverse adaptive filtering.',
        source_mappings=(_source('_color_card_detection'),)))
    stages.append(RecipeStage(GEOMETRY, 'Detect card geometry', 'color_card.geometry',
        ports=(_port('roi_bgr','input','BGRImage',local),_port('threshold','input','GrayImage',local_gray),
               _port('geometry','output','CardGeometry'),_port('metadata','output','Metadata')),
        parameters=(ParameterContract('min_size','number',1000,True,editable=True,minimum=0,
            description='Strict minimum contour area in pixels squared.'),
            ParameterContract('aspect_ratio','number',1.27,True,editable=True,minimum=1e-12,
                description='Strict rectangle aspect limit; reciprocal values below1 are supported.'),
            ParameterContract('solidity','number',.8,True,editable=True,minimum=0,maximum=1,
                description='Strict contour area divided by rotated-rectangle area.')),
        description='Trace source-ordered contours, filter chips and derive the ordered24-center grid.',
        source_mappings=(_source('_color_card_detection'),)))
    stages.append(RecipeStage(MASK,'Sample circles','color_card.mask',
        ports=(_port('bgr','input','BGRImage',bgr),_port('bounds','input','RegionBounds'),
               _port('geometry','input','CardGeometry'),_port('labeled_mask','output','ChipMask',labels),
               _port('bounding_mask','output','BinaryMask',{'dtype':'uint8','shape':'HxW','background':0,'foreground':255})),
        parameters=(ParameterContract('radius','integer',20,True,editable=True,minimum=0,maximum=4096,
            description='Filled chip-circle radius in pixels; later chip labels overwrite overlaps.'),),
        description='Draw full-frame CUDA chip labels and the inclusive clipped bounding mask.',
        source_mappings=(_source('_draw_color_chips'),)))
    stages.append(RecipeStage(SCALE,'Chip size calibration','color_card.scale',
        ports=(_port('geometry','input','CardGeometry'),_port('metadata','output','Metadata')),
        parameters=(ParameterContract('color_chip_size','json',None,True,editable=True,
            description='null, named classic/passport/cameratrax/nano/mini, or [width,height] in mm.'),),
        description='Calculate mm per pixel from measured chip dimensions. This does not change image pixels or application units.',
        source_mappings=(_source('_set_size_scale_from_chip'),)))
    correction_path='plantcv/plantcv/transform/color_correction.py'
    def mapping(symbol):
        return (_plantcv_mapping(config,correction_path,'plantcv.plantcv.transform.color_correction.'+symbol),)
    stages.append(RecipeStage(MEANS,'Measure chip colours','color_card.means',
        ports=(_port('bgr','input','BGRImage',bgr),_port('labeled_mask','input','ChipMask',labels),
               _port('source_matrix','output','ColorMatrix')),
        description='CUDA integer chip reductions; compact rows retain chip ID and normalized RGB means.',
        source_mappings=mapping('get_color_matrix')))
    stages.append(RecipeStage(STANDARD,'Standard chip colours','color_card.standard',
        ports=(_port('bgr','input','BGRImage',bgr),_port('target_matrix','output','ColorMatrix')),
        parameters=(ParameterContract('pos','integer',3,True,editable=True,minimum=0,maximum=3,
            description='Standard-card orientation0..3; automatic correction uses3 by default.'),),
        description='Standard normalized RGB targets in the selected public matrix orientation.',
        source_mappings=mapping('std_color_matrix')))
    stages.append(RecipeStage(AFFINE,'Correct image colours','color_card.affine',
        ports=(_port('bgr','input','BGRImage',bgr),_port('source_matrix','input','ColorMatrix'),
               _port('target_matrix','input','ColorMatrix'),
               _port('corrected_rgb','output','RGBImage',rgb),_port('transformation_matrix','output','ColorMatrix'),
               _port('metadata','output','Metadata')),
        description='Fit source-to-target affine RGB values and correct the whole image on CUDA.',
        source_mappings=mapping('affine_color_correction')))
    edges=[]
    def link(start,port,end,target=None):
        original=next(p for s in stages if s.stage_id==start for p in s.ports if p.name==port and p.direction=='output')
        edges.append(DependencyContract(start,port,end,target or port,original.data_type))
    link(source.stage_id,'rgb_image',PREFIX+'bgr','rgb')
    for stage in (PREPROCESS,MASK,MEANS,STANDARD,AFFINE):link(PREFIX+'bgr','bgr',stage)
    for port in ('roi_bgr','threshold'):link(PREPROCESS,port,GEOMETRY)
    link(PREPROCESS,'bounds',MASK)
    for stage in (MASK,SCALE):link(GEOMETRY,'geometry',stage)
    link(MASK,'labeled_mask',MEANS)
    link(MEANS,'source_matrix',AFFINE);link(STANDARD,'target_matrix',AFFINE)
    positions=((0,0),(260,0),(520,0),(780,0),(1040,0),(1040,270),(1300,0),(780,-270),(1560,0))
    return RecipeSpec(RECIPE_ID,VERSION,'Automatic colour-card correction',
        'Editable automatic card detection and affine correction. Reference agreement and scientific colour accuracy remain separate.',
        provenance,tuple(stages),tuple(edges),tuple(LayoutPosition(s.stage_id,*p) for s,p in zip(stages,positions)))


def retained_bound(pixels):
    """Bound graph-owned CUDA storage, including replacement-stage scratch.

    At most25N bytes are retained across accepted nodes (gray/RGB4N, BGR3N,
    ROI copy/filters6N, labels/bbox9N, corrected RGB3N). Affine application
    temporarily adds another3N BGR bytes; chip conversion adds1N in its earlier
    stage.32N leaves dense slack. The additional256MiB covers contour extraction
    scratch (already including its foreground), or the nonconcurrent64MiB filter
    scratch and bounded affine/reduction tiles. Dirty descendants are released
    before replacement allocation. Caller-retained old results are not cache-owned.
    """
    bound=32*pixels+256*1024*1024
    if not 1<=pixels<=64_000_000 or bound>1024*1024*1024:
        raise card.AutomaticCardResourceError('Automatic card graph exceeds its retained CUDA workspace bound')
    return bound


def implementation_modules():
    return (sys.modules[__name__],card,preprocess,contours,minrect,kernel_pins,color_card,correction)


def metadata_identity():
    if (card.GRID_BOX_411.flags.writeable or card.GRID_BOX_411.shape!=(4,2)
            or card.GRID_BOX_411.dtype!=np.dtype('<f4')
            or hashlib.sha256(card.GRID_BOX_411.tobytes()).hexdigest()!=card.GRID_BOX_SHA256):
        raise card.AutomaticCardError('Fixed grid metadata differs from the pinned source')
    return {'grid_box_sha256':card.GRID_BOX_SHA256,'reference_sources':card.REFERENCE_SOURCES}


def _matrix(value):
    return tuple(tuple(float(x) for x in row) for row in value)


def execute_card_stage(operation,inputs,parameters,cancel):
    """Return CUDA rasters or deeply immutable scalar metadata, never NumPy views."""
    image=inputs.get('bgr',inputs.get('roi_bgr'))
    guard=correction._guarded(cancel,tuple(value for value in inputs.values() if torch.is_tensor(value)),metadata=inputs)
    def finish(outputs,*private,metadata=()):
        guard(*private,*(value for value in outputs.values() if torch.is_tensor(value)),metadata=(outputs,metadata))
        return outputs
    guard()
    if operation=='color_card.bgr':
        return finish({'bgr':inputs['rgb'].flip(-1).contiguous()})
    if operation=='color_card.preprocess':
        height,width=map(int,image.shape[:2])
        bounds=card._bounds(parameters['roi_bounds_xyxy'],height,width)
        card._preflight(height,width,bounds,parameters['adaptive_method'],parameters['block_size'],0,correction=True)
        x0,y0,x1,y1=bounds
        local=image[y0:y1,x0:x1].contiguous()
        result=preprocess.preprocess_color_card_cuda(local,parameters['adaptive_method'],parameters['block_size'],cancel=lambda:guard(local))
        return finish({'roi_bgr':local,'gray':result.gray,'blurred':result.blurred,'threshold':result.threshold,'bounds':bounds})
    if operation=='color_card.geometry':
        card._parameters(0,parameters['min_size'],parameters['aspect_ratio'],parameters['solidity'],None)
        forest=contours.trace_binary_contours_cuda(inputs['threshold'],cancel=guard)
        selected,bbox,corners,centers,transform=card._geometry(image,forest,parameters['min_size'],
            parameters['aspect_ratio'],parameters['solidity'],lambda:guard(metadata=(forest,)))
        areas=tuple(float(c[2]) for c in selected)
        widths=tuple(float(c[3][1][0]) for c in selected);heights=tuple(float(c[3][1][1]) for c in selected)
        metadata={'detected_chips':len(selected),'median_color_chip_size':float(np.median(areas)),
            'median_color_chip_width':float(np.median(widths)),'median_color_chip_height':float(np.median(heights))}
        geometry={**metadata,'selected_contour_indices':tuple(c[0] for c in selected),'bounding_rect':bbox,
            'centers_xy':tuple(tuple(map(int,row)) for row in centers),'corners_xy':tuple(tuple(map(int,row)) for row in corners),
            'transformation':_matrix(transform),'areas':areas,'widths':widths,'heights':heights,
            'opencv_version':forest.opencv_version,'numpy_version':forest.numpy_version,**metadata_identity()}
        return finish({'geometry':_freeze_json(geometry,name='card geometry'),
                'metadata':_freeze_json(metadata,name='card measurements')},metadata=(corners,centers,transform))
    if operation=='color_card.mask':
        radius=parameters['radius'];card._parameters(radius,0,1.27,.8,None)
        height,width=map(int,image.shape[:2]);bounds=card._bounds(inputs['bounds'],height,width)
        card._circle_budget(bounds[3]-bounds[1],bounds[2]-bounds[0],radius)
        geometry=inputs['geometry']
        labels,bounding=card._rasterize_geometry(height,width,bounds,geometry['bounding_rect'],
            geometry['centers_xy'],radius,image.device,guard)
        return finish({'labeled_mask':labels,'bounding_mask':bounding})
    if operation=='color_card.scale':
        geometry=inputs['geometry'];physical=card._parameters(0,0,1.27,.8,parameters['color_chip_size'])
        px_width,px_height,unit=card._scale_metadata(geometry['median_color_chip_width'],
            geometry['median_color_chip_height'],physical)
        return finish({'metadata':_freeze_json({'px_width':px_width,'px_height':px_height,'unit':unit},name='chip scale')})
    if operation=='color_card.means':
        mask=inputs['labeled_mask'].to(torch.uint8)
        _,matrix=correction.get_color_matrix_cuda(image,mask,cancel=lambda:guard(mask))
        return finish({'source_matrix':_matrix(matrix)},mask,metadata=(matrix,))
    if operation=='color_card.standard':
        cuda_matrix=color_card.std_color_matrix(parameters['pos'],device=image.device)
        matrix=cuda_matrix.cpu().numpy()
        return finish({'target_matrix':_matrix(matrix)},cuda_matrix,metadata=(matrix,))
    if operation=='color_card.affine':
        result=correction.affine_color_correction_cuda(image,inputs['source_matrix'],inputs['target_matrix'],cancel=guard)
        return finish({'corrected_rgb':result.corrected_bgr.flip(-1).contiguous(),
            'transformation_matrix':_matrix(result.transformation_matrix),
            'metadata':_freeze_json({'fit':result.fit,'numpy_version':result.numpy_version},name='affine fit')},
            result.corrected_bgr,metadata=(result.source_matrix,result.target_matrix,result.transformation_matrix))
    raise NotImplementedError('Unknown card graph operation: '+operation)
