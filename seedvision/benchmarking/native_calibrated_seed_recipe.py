"""Compose card correction, an explicit seed crop and the working seed graphs."""
from dataclasses import replace
import hashlib
import json
import math
import sys
from collections.abc import Mapping

import torch

from . import native_card_recipe as card_recipe
from .checkpoints import CoordinateFrame, _freeze_json
from .recipe_catalogue import build_recipe_catalogue
from .recipes import (DependencyContract, LayoutPosition, ParameterContract, PortContract,
                      RecipeSpec, SourceMapping, SourceResolution)
from seedvision.cuda import plantcv_range as color_range
from seedvision.cuda import plantcv_peaks as peaks
from seedvision.cuda import plantcv_size as size

VERSION = '1.0.0'
RANGE_RECIPE_ID = 'plantcv_calibrated_seed_range'
RECIPE_IDS = tuple('plantcv_calibrated_seed_' + kind for kind in ('binary', 'mean', 'otsu', 'range'))
CROP = 'pcv_calibrated_seed_crop'
SCALED_SIZE = 'pcv_calibrated_seed_size_units'
LOCAL_FRAME = 'seed_roi_xyxy'
GRAY_CONVERSION = 'Pinned OpenCV BGR2GRAY byte coefficients applied to corrected RGB after cropping.'
LOCAL_COORDINATES = 'ROI-local pixels; parent xy=local xy+local_to_parent_xy'
AREA_VARIABLES = ('area', 'convex_hull_area')
LINEAR_VARIABLES = ('perimeter', 'total_edge_length', 'width', 'height', 'longest_path',
                    'ellipse_major_axis', 'ellipse_minor_axis')
UNSCALED_VARIABLES = ('in_bounds', 'solidity', 'center_of_mass', 'convex_hull_vertices',
                      'object_in_frame', 'ellipse_center', 'ellipse_angle', 'ellipse_eccentricity')


def _helper_mapping(symbol):
    return SourceMapping(reference_id='plantcv', reference_version='4.11.3',
        resolution=SourceResolution.PINNED, source_path='plantcv/plantcv/_helpers.py',
        source_symbol='plantcv.plantcv._helpers.' + symbol,
        source_url='https://github.com/danforthcenter/plantcv/blob/v4.11.3/plantcv/plantcv/_helpers.py',
        source_sha256=color_range.PLANTCV_HELPER_REFERENCE_ENV_SHA256, license_id='MPL-2.0',
        note='Explicit composition; frame metadata and execution orchestration are native.')


def build_native_calibrated_seed_recipe(kind='binary'):
    """Add a graph without redefining either the card or historical seed recipe."""
    from .native_recipe import build_native_seed_recipe
    if kind == 'range':
        return _build_range_recipe()
    if kind not in ('binary', 'mean', 'otsu'):
        raise ValueError('Select binary, mean, otsu or range for calibrated seed analysis.')
    card = card_recipe.build_native_card_recipe()
    seed = build_native_seed_recipe(build_recipe_catalogue().recipe('plantcv_seed_' + kind))
    source = card.stages[0]
    source = replace(source, ports=source.ports + (PortContract('source_frame', 'output',
        'Metadata', 'Decoded source frame'),))
    seed_source = seed.stages[0]
    source_ports = {port.name: port for port in seed_source.ports}
    local_rgb = {**dict(source_ports['rgb_image'].semantics), 'frame': LOCAL_FRAME}
    local_gray = {**dict(source_ports['gray_image'].semantics), 'frame': LOCAL_FRAME}
    def port(name, direction, data_type, semantics=None):
        return PortContract(name, direction, data_type, name.replace('_', ' '), semantics=semantics or {})
    from .recipes import RecipeStage
    crop = RecipeStage(CROP, 'Select seed region', 'calibrated_seed.crop',
        description='Required half-open [x0,y0,x1,y1] in the corrected source frame. '
            'Crop before grayscale and seed preprocessing. Labels and coordinates stay local to this crop; '
            'parent coordinates are local+(x0,y0). Card overlap is reported, never silently removed.',
        ports=(port('corrected_rgb','input','RGBImage',dict(source_ports['rgb_image'].semantics)),
            port('source_frame','input','Metadata'), port('geometry','input','CardGeometry'),
            port('bounds','input','RegionBounds'), port('rgb_image','output','RGBImage',local_rgb),
            port('gray_image','output','GrayImage',local_gray), port('metadata','output','Metadata')),
        parameters=(ParameterContract('seed_roi_xyxy','json',None,True,editable=True,
            description='Required integer half-open bounds in corrected full-frame pixels; select full bounds explicitly for the whole image.'),),
        source_mappings=(_helper_mapping('_rect_filter'),
            card_recipe._source('_color_card_detection', 'Pinned BGR-to-gray byte conversion after the explicit crop.')))
    # The default seed controls remain explicitly unselected where the source
    # catalogue requires an image-specific choice. All old nodes keep their IDs.
    local_stages = tuple(replace(stage, ports=tuple(replace(p,
        semantics={**dict(p.semantics), 'frame': LOCAL_FRAME})
        for p in stage.ports)) for stage in seed.stages[1:])
    size = next(stage for stage in local_stages if stage.operation_id == 'analysis.size')
    observations = next(p for p in size.ports if p.name == 'observations' and p.direction == 'output')
    scaled = RecipeStage(SCALED_SIZE, 'Size in calibrated units', 'calibrated_seed.size_units',
        description='Apply the pinned size conversion to compact pixel observations. '
            'Linear traits use mm/pixel width; areas use width*height. Source scale/label fields '
            'use mm for both, as PlantCV does. Coordinates remain local pixels. Without a chip size, retain pixel units.',
        ports=(replace(observations, direction='input'), port('scale','input','Metadata'),
            observations, port('metadata','output','Metadata')),
        source_mappings=(_helper_mapping('_scale_size'),) + size.source_mappings)
    stages = (source,) + card.stages[1:] + (crop,) + local_stages + (scaled,)
    edges = list(card.dependencies)
    edges.extend((DependencyContract(card_recipe.AFFINE,'corrected_rgb',CROP,'corrected_rgb','RGBImage'),
        DependencyContract(source.stage_id,'source_frame',CROP,'source_frame','Metadata'),
        DependencyContract(card_recipe.GEOMETRY,'geometry',CROP,'geometry','CardGeometry'),
        DependencyContract(card_recipe.PREPROCESS,'bounds',CROP,'bounds','RegionBounds')))
    edges.extend(replace(edge, source_stage=CROP) if edge.source_stage == seed_source.stage_id else edge
                 for edge in seed.dependencies)
    edges.extend((DependencyContract(size.stage_id,'observations',SCALED_SIZE,'observations',observations.data_type),
        DependencyContract(card_recipe.SCALE,'metadata',SCALED_SIZE,'scale','Metadata')))
    provenance = replace(card.provenance,
        backend_id='seedvision.benchmarking.native_calibrated_seed_recipe', backend_version=VERSION,
        evidence_note='Native composition of pinned automatic affine colour correction, explicit ROI crop, '
            'seed preprocessing/watershed and size/colour observations. Physical conversion is an independent '
            'compact branch. Image-specific settings must be selected; no scientific validation is implied.')
    layout = tuple(card.layout) + (LayoutPosition(CROP,1820,0),) + tuple(
        LayoutPosition(s.stage_id,2080+(i%5)*260,(i//5)*270) for i,s in enumerate(local_stages)) + (
        LayoutPosition(SCALED_SIZE,3120,810),)
    return RecipeSpec('plantcv_calibrated_seed_'+kind, VERSION,
        'Calibrated seed analysis ('+kind+')',
        'Detect and correct the colour card, crop a selected seed region, segment seeds and measure local traits.',
        provenance, stages, tuple(edges), layout)


def _build_range_recipe():
    """Add explicit RGB range -> binary median; leave old arms byte-identical."""
    from .native_range_recipe import build_native_range_recipe
    base = build_native_calibrated_seed_recipe('binary')
    median = next(s for s in base.stages if s.operation_id == 'filter.median_blur')
    threshold = next(s for s in base.stages if s.operation_id == 'threshold.binary')
    rename = {s.stage_id:s.stage_id.replace('pcv_seed_binary_', 'pcv_seed_range_') for s in base.stages}
    rename[threshold.stage_id] = 'pcv_seed_range_custom_range'
    range_stage=build_native_range_recipe().stages[1]
    ranged = replace(range_stage,stage_id=rename[threshold.stage_id],
        ports=tuple(replace(p,semantics={**dict(p.semantics),'frame':LOCAL_FRAME})
            for p in range_stage.ports))
    mask_semantics = dict(next(p for p in ranged.ports if p.name=='mask').semantics)
    binary_median = replace(median,stage_id=rename[median.stage_id],title='Smooth range mask',
        description='Apply the pinned median operation to the binary range mask before fill and morphology. The kernel affects this mask and its segmentation descendants.',
        ports=tuple(replace(p,data_type='BinaryMask',semantics=mask_semantics) for p in median.ports))
    stages=[]
    for stage in base.stages:
        if stage.stage_id==median.stage_id:stages.extend((ranged,binary_median))
        elif stage.stage_id!=threshold.stage_id:stages.append(replace(stage,stage_id=rename[stage.stage_id]))
    edges=[]
    for edge in base.dependencies:
        if edge.target_stage in (median.stage_id,threshold.stage_id):continue
        if edge.source_stage==threshold.stage_id:
            edges.append(replace(edge,source_stage=rename[median.stage_id],source_port='image',target_stage=rename[edge.target_stage]))
        else:
            edges.append(replace(edge,source_stage=rename[edge.source_stage],target_stage=rename[edge.target_stage]))
    edges.extend((DependencyContract(CROP,'rgb_image',ranged.stage_id,'rgb','RGBImage'),
        DependencyContract(ranged.stage_id,'mask',binary_median.stage_id,'gray','BinaryMask')))
    layout=tuple(replace(p,stage_id=(ranged.stage_id if p.stage_id==median.stage_id
        else binary_median.stage_id if p.stage_id==threshold.stage_id else rename[p.stage_id])) for p in base.layout)
    provenance=replace(base.provenance,evidence_note='Native working composition: automatic card correction, explicit corrected RGB seed ROI, pinned custom-range mask and original masked RGB, binary-mask median, fill/morphology/watershed and size/colour observations. Trait input remains unmasked corrected RGB. No historical fixed range recipe or scientific validation is claimed.')
    return replace(base,recipe_id=RANGE_RECIPE_ID,title='Calibrated seed analysis (range)',
        description='Correct card colours, select an RGB seed region and channel bounds, smooth the range mask, segment and measure local traits.',
        provenance=provenance,stages=tuple(stages),dependencies=tuple(edges),layout=layout)


def retained_bound(pixels, *, range_arm=False):
    """Joint graph-owned bound before upload or replacement allocations.

    Card products retain25N bytes; the contiguous crop/gray adds4N. Seed gray,
    five byte masks, distance/markers/negative/labels and diagnostics retain at
    most25N more.64N leaves dense/temporary slack, plus16 bytes per possible
    peak.256MiB covers the largest nonconcurrent contour/filter/size scratch.
    Views count by backing storage in the shared executor. Old caller-held runs
    remain outside cache ownership and must be released by the caller.
    """
    bound=64*pixels+16*min(pixels,peaks.MAX_CANDIDATES)+256*1024*1024
    # Range retains an additional masked RGB and transient BGR/primitive output
    # products. Charge a full-source16N margin even for a much smaller seed ROI;
    # its <=132MiB tiled scratch fits the existing nonconcurrent256MiB reserve.
    if range_arm:bound+=16*pixels
    if not 1<=pixels<=64_000_000 or bound>1024*1024*1024:
        raise card_recipe.card.AutomaticCardResourceError('Calibrated seed graph exceeds its joint CUDA workspace bound')
    return bound


def implementation_modules():
    return (sys.modules[__name__],)


def _crop_frame(parent, selected):
    """Derive the declared ROI frame without allocating or cropping pixels."""
    if not isinstance(parent,CoordinateFrame):
        raise TypeError('Seed crop requires an explicit parent CoordinateFrame.')
    height,width=parent.shape
    if (not isinstance(selected,(tuple,list)) or len(selected)!=4
            or any(type(v) is not int for v in selected)):
        raise ValueError('Seed ROI requires four integer half-open bounds.')
    x0,y0,x1,y1=selected
    if not (0<=x0<x1<=width and 0<=y0<y1<=height):
        raise ValueError('Seed ROI must be nonempty and inside the corrected image.')
    if (parent.origin,parent.axes,parent.units)!=('top-left','x-right,y-down','pixel'):
        raise ValueError('Seed crop requires top-left x-right,y-down pixel coordinates.')
    translation={'parent_frame':parent.to_dict(),'bounds_xyxy':list(selected),'local_to_parent_xy':[x0,y0]}
    transform=hashlib.sha256(json.dumps(translation,sort_keys=True,separators=(',',':')).encode('utf-8')).hexdigest()
    local=CoordinateFrame(parent.frame_id+'.seed_roi',(y1-y0,x1-x0),parent.origin,parent.axes,parent.units,transform)
    return local,translation


def _crop_overlap(selected, bounds, bounding_rect):
    """Intersect the seed ROI with the detector's clipped inclusive mask."""
    x0,y0,x1,y1=selected
    bx,by,bw,bh=bounding_rect
    rx,ry,rx1,ry1=bounds
    left,top=max(rx,bx+rx),max(ry,by+ry)
    right,bottom=min(rx1,bx+rx+bw+1),min(ry1,by+ry+bh+1)
    return max(0,min(x1,right)-max(x0,left))*max(0,min(y1,bottom)-max(y0,top))


def _crop(inputs, parameters, cancel):
    image=inputs['corrected_rgb']
    height,width=map(int,image.shape[:2])
    parent=CoordinateFrame.from_dict(dict(inputs['source_frame']['coordinate_frame']))
    if parent.shape!=(height,width):
        raise ValueError('Corrected image differs from the declared parent frame.')
    local,translation=_crop_frame(parent,parameters['seed_roi_xyxy'])
    x0,y0,x1,y1=translation['bounds_xyxy']
    guard=card_recipe.correction._guarded(cancel,(image,),metadata=(inputs,parameters))
    guard()
    rgb=image[y0:y1,x0:x1].contiguous()
    gray=torch.empty((y1-y0,x1-x0),device=image.device,dtype=torch.uint8)
    # Conversion uses the same pinned integer coefficients as detector/path input
    # preparation; only bounded pixel tiles receive a temporary BGR copy.
    flat_rgb,flat_gray=rgb.reshape(-1,3),gray.reshape(-1)
    for start in range(0,gray.numel(),262144):
        guard(rgb,gray)
        end=min(gray.numel(),start+262144)
        flat_gray[start:end]=color_range._channels(flat_rgb[start:end].flip(-1),'GRAY')[:,0].to(torch.uint8)
        guard(rgb,gray)
    # The detector's bounding mask includes x+w and y+h, clipped within its ROI.
    overlap=_crop_overlap(translation['bounds_xyxy'],inputs['bounds'],inputs['geometry']['bounding_rect'])
    metadata={**translation,'coordinate_frame':local.to_dict(),
        'source_sha256':inputs['source_frame']['source_sha256'],
        'gray_conversion':GRAY_CONVERSION,
        'card_bounding_overlap_pixels':overlap,'labels_and_coordinates':LOCAL_COORDINATES}
    outputs={'rgb_image':rgb,'gray_image':gray,'metadata':_freeze_json(metadata,name='seed crop frame')}
    guard(rgb,gray,metadata=(translation,outputs))
    return outputs


def _scaled_size(inputs):
    scale=inputs['scale']
    if not isinstance(scale,Mapping) or set(scale)!= {'px_width','px_height','unit'}:
        raise ValueError('Size conversion requires the explicit card scale record.')
    if all(scale[name] is None for name in ('px_width','px_height','unit')):
        width,height,unit=1.0,1.0,'pixels'
    else:
        width,height,unit=scale['px_width'],scale['px_height'],scale['unit']
        if (type(width) not in (int,float) or type(height) not in (int,float)
                or not math.isfinite(width) or not math.isfinite(height) or width<=0 or height<=0
                or unit!='mm'):
            raise ValueError('Size conversion requires finite positive mm per pixel factors.')
    area=width*height
    if not math.isfinite(area):
        raise ValueError('Size conversion area factor overflowed.')
    observations=inputs['observations']
    if not isinstance(observations,Mapping) or not 1<=len(observations)<=size.MAX_SIZE_DIAGNOSTIC_LABELS:
        raise ValueError('Size observations exceed their compact sample bound.')
    converted={}
    for sample,values in observations.items():
        if not isinstance(values,Mapping) or set(values)!=set(AREA_VARIABLES+LINEAR_VARIABLES+UNSCALED_VARIABLES):
            raise ValueError('Size conversion requires the pinned seventeen observation variables.')
        converted[sample]={}
        for variable,item in values.items():
            row=dict(item)
            if variable in AREA_VARIABLES+LINEAR_VARIABLES:
                factor=area if variable in AREA_VARIABLES else width
                value=row['value']
                if type(value) not in (int,float) or not math.isfinite(value):
                    raise ValueError('Size conversion requires finite scalar source measurements.')
                value=value*factor
                if not math.isfinite(value):raise ValueError('Size measurement overflowed during conversion.')
                row.update(value=value,scale=unit,label=unit)
            converted[sample][variable]=row
    return {'observations':_freeze_json(converted,name='scaled size observations'),
        'metadata':_freeze_json({'px_width':width,'px_height':height,'unit':unit,'area_factor':area,
            'area_variables':AREA_VARIABLES,'linear_variables':LINEAR_VARIABLES,
            'coordinate_units':'ROI-local pixels','source_area_scale_field':unit},name='size conversion')}


def execute_stage(operation,inputs,parameters,cancel):
    if operation=='calibrated_seed.crop':
        try:
            return _crop(inputs,parameters,cancel)
        except card_recipe.correction.CorrectionCancelled as error:
            raise card_recipe.card.AutomaticCardCancelled(
                'Seed crop was cancelled before a complete result.') from error
    if operation=='calibrated_seed.size_units':return _scaled_size(inputs)
    raise NotImplementedError('Unknown calibrated seed operation: '+operation)
