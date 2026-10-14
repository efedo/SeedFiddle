"""Bounded compact colour-card contracts and allocation-free stage admission.

The replay runner owns ancestry, CUDA uploads, execution and raster comparison.
This module keeps source and ROI frames distinct and never constructs archived
Python types. Declared caps cover encoded data and modeled CUDA storage, not RSS.
"""
from collections.abc import Mapping
from dataclasses import dataclass
import math
from types import MappingProxyType

from . import native_card_recipe as native

MAX_METADATA_BYTES = 8 * 1024 * 1024
MAX_METADATA_ITEMS = 1_000_000
MAX_METADATA_DEPTH = 16
MAX_SELECTED_BYTES = 128 * 1024 * 1024
MAX_ARRAY_BYTES = 128 * 1024 * 1024
MAX_TOTAL_ARRAY_BYTES = 256 * 1024 * 1024
MAX_CUDA_BYTES = 1024 * 1024 * 1024
COMPARE_ELEMENTS = 262_144
AUDIT_CUDA_BYTES = 64 * 1024
ALLOCATION_BYTES = 512
MAX_PIXELS = 64_000_000

# The port's role declares both its compact schema and its coordinate frame.
PORTS = MappingProxyType({
    'color_card.bgr': ({'rgb':'source_rgb'}, {'bgr':'source_bgr'}, 'bgr'),
    'color_card.preprocess': ({'bgr':'source_bgr'},
        {'roi_bgr':'roi_bgr','gray':'roi_gray','blurred':'roi_gray','threshold':'roi_binary','bounds':'bounds'}, 'threshold'),
    'color_card.geometry': ({'roi_bgr':'roi_bgr','threshold':'roi_binary'},
        {'geometry':'geometry','metadata':'measurements'}, None),
    'color_card.mask': ({'bgr':'source_bgr','bounds':'bounds','geometry':'geometry'},
        {'labeled_mask':'chip_labels','bounding_mask':'source_binary'}, 'labeled_mask'),
    'color_card.scale': ({'geometry':'geometry'}, {'metadata':'scale'}, None),
    'color_card.means': ({'bgr':'source_bgr','labeled_mask':'chip_labels'}, {'source_matrix':'chip_matrix'}, None),
    'color_card.standard': ({'bgr':'source_bgr'}, {'target_matrix':'standard_matrix'}, None),
    'color_card.affine': ({'bgr':'source_bgr','source_matrix':'fit_matrix','target_matrix':'fit_matrix'},
        {'corrected_rgb':'source_rgb','transformation_matrix':'affine_matrix','metadata':'affine_metadata'}, 'corrected_rgb'),
})
PORTS = MappingProxyType({key:(MappingProxyType(inputs),MappingProxyType(outputs),preview)
                         for key,(inputs,outputs,preview) in PORTS.items()})
RASTER_ROLES = frozenset(('source_rgb','source_bgr','source_binary','roi_bgr','roi_gray','roi_binary','chip_labels'))
GEOMETRY_FIELDS = frozenset(('detected_chips','median_color_chip_size','median_color_chip_width',
    'median_color_chip_height','selected_contour_indices','bounding_rect','centers_xy','corners_xy',
    'transformation','areas','widths','heights','opencv_version','numpy_version','grid_box_sha256','reference_sources'))
MEASUREMENT_FIELDS = frozenset(('detected_chips','median_color_chip_size','median_color_chip_width','median_color_chip_height'))


class NativeCardReplayError(ValueError):
    """An archived product or control violates its declared card contract."""


class NativeCardReplayUnavailable(NativeCardReplayError):
    """A valid operation exceeds its source or replay resource domain."""


def state():
    return (MAX_METADATA_BYTES,MAX_METADATA_ITEMS,MAX_METADATA_DEPTH,MAX_SELECTED_BYTES,
        MAX_ARRAY_BYTES,MAX_TOTAL_ARRAY_BYTES,MAX_CUDA_BYTES,
        COMPARE_ELEMENTS,AUDIT_CUDA_BYTES,ALLOCATION_BYTES,MAX_PIXELS,
        tuple((key,tuple(inputs.items()),tuple(outputs.items()),preview) for key,(inputs,outputs,preview) in PORTS.items()),
        tuple(sorted(GEOMETRY_FIELDS)),tuple(sorted(MEASUREMENT_FIELDS)),
        native.card.MAX_RETAINED_BYTES,native.card.MAX_GEOMETRY_WORK,native.card.MAX_CIRCLE_WORK,
        native.card.MAX_RADIUS,native.card.MAX_ROW_WRITES,native.card.GRID_BOX_SHA256,native.card.REFERENCE_SOURCES,
        native.preprocess.MAX_TILE_SIDE,native.preprocess.MAX_FILTER_WORK,native.preprocess.MAX_SCRATCH_BYTES,
        native.preprocess.MAX_RETAINED_BYTES,native.contours.MAX_SCRATCH_BYTES,
        native.contours.MAX_BOUNDARY_POINTS,native.contours.MAX_TILE_PIXELS,
        native.contours.size.MAX_CONTOUR_COMPONENTS,native.contours.size.MAX_CONTOUR_SCAN_ROWS,
        native.correction.TILE_PIXELS,native.correction.MAX_CHIPS,native.correction.MAX_PIXELS)


def _compact(value, *, encoded, poll=None):
    items=bytes_used=0
    active=set()
    def charge(amount,visits=0):
        nonlocal items,bytes_used
        items+=visits;bytes_used+=amount
        if items>MAX_METADATA_ITEMS or bytes_used>MAX_METADATA_BYTES:
            raise NativeCardReplayError('Card metadata exceeds its traversal/encoded-byte bound.')
        if visits and poll is not None and items%1024==0:poll()
    def text(value):
        if len(value)>MAX_METADATA_BYTES:raise NativeCardReplayError('Card metadata text exceeds its bound.')
        # Exact ensure_ascii=False JSON size, without allocating an escaped copy.
        charge(2)
        for index,char in enumerate(value):
            code=ord(char)
            if 0xd800<=code<=0xdfff:raise NativeCardReplayError('Invalid card metadata text.')
            size=(2 if char in '\\"\b\f\n\r\t' else 6 if code<32 else
                  1 if code<128 else 2 if code<2048 else 3 if code<65536 else 4)
            charge(size)
            if poll is not None and index and index%1024==0:poll()
        return value
    def visit(value,depth):
        if depth>MAX_METADATA_DEPTH:raise NativeCardReplayError('Card metadata exceeds its depth bound.')
        charge(0,1)
        if value is None or type(value) is bool:
            charge(4 if value is None or value is True else 5);return value
        if type(value) is int:
            if value.bit_length()>256:raise NativeCardReplayError('Card metadata integer exceeds its bound.')
            charge(len(str(value)));return value
        if type(value) is float:
            if not math.isfinite(value):raise NativeCardReplayError('Card metadata requires finite numbers.')
            charge(len(repr(value)));return value
        if type(value) is str:return text(value)
        if encoded:
            if (type(value) is not dict or set(value)!={'kind','items'}
                    or type(value['kind']) is not str or value['kind'] not in ('mapping','tuple','list')
                    or type(value['items']) is not list):
                raise NativeCardReplayError('Card metadata requires primitive tagged mappings/sequences.')
            kind,entries=value['kind'],value['items']
        elif isinstance(value,Mapping):
            if len(value)>MAX_METADATA_ITEMS:raise NativeCardReplayError('Card metadata container exceeds its item bound.')
            kind,entries='mapping',value.items()
        elif type(value) in (tuple,list):kind,entries=('tuple' if type(value) is tuple else 'list'),value
        else:raise NativeCardReplayError('Card metadata contains an unsupported Python value.')
        if len(entries)>MAX_METADATA_ITEMS:raise NativeCardReplayError('Card metadata container exceeds its item bound.')
        if id(value) in active:raise NativeCardReplayError('Card metadata contains a cycle.')
        active.add(id(value))
        try:
            # Complete tagged-container envelope and entry punctuation.
            charge(len('{"kind":"'+kind+'","items":[]}')+max(0,len(entries)-1))
            if kind=='mapping':
                result={}
                for entry in entries:
                    if (type(entry) not in (tuple,list) or len(entry)!=2 or type(entry[0]) is not str or entry[0] in result):
                        raise NativeCardReplayError('Card metadata mapping keys must be unique strings.')
                    if depth+1>MAX_METADATA_DEPTH:raise NativeCardReplayError('Card metadata exceeds its depth bound.')
                    charge(3,1)
                    result[text(entry[0])]=visit(entry[1],depth+1)
                return MappingProxyType(result)
            result=[visit(entry,depth+1) for entry in entries]
            return tuple(result) if kind=='tuple' else result
        finally:active.remove(id(value))
    return visit(value,0)


def decode(binding,poll=None):
    return _compact(binding,encoded=True,poll=poll)


def canonical(value,poll=None):
    return _compact(value,encoded=False,poll=poll)


def _number(value, *, positive=False):
    return type(value) in (int,float) and math.isfinite(value) and (not positive or value>0)


def _sequence(value,length=None):
    return type(value) in (tuple,list) and (length is None or len(value)==length)


def _matrix(value,rows,columns, *, normalized=False):
    return (_sequence(value) and rows[0]<=len(value)<=rows[1]
        and all(_sequence(row,columns) and all(_number(v) for v in row)
                and (not normalized or all(0<=v<=1 for v in row[1:])) for row in value))


def frames(source_shape,bounds=None):
    if (type(source_shape) not in (tuple,list) or len(source_shape)!=2
            or any(type(v) is not int or v<=0 for v in source_shape)):
        raise NativeCardReplayError('Card source shape must contain positive integer height/width.')
    height,width=source_shape
    try:bounds=native.card._bounds(bounds,height,width)
    except ValueError as error:raise NativeCardReplayError(str(error)) from error
    return tuple(source_shape),(bounds[3]-bounds[1],bounds[2]-bounds[0]),bounds


def raster_spec(role,source_shape,bounds=None):
    source,local,_=frames(source_shape,bounds)
    if role not in RASTER_ROLES:raise NativeCardReplayError('Unknown card raster role.')
    shape=local if role.startswith('roi_') else source
    if role.endswith(('rgb','bgr')):shape=(*shape,3)
    return shape,('<f8' if role=='chip_labels' else '|u1')


def controls(operation,parameters):
    keys={'color_card.bgr':(), 'color_card.preprocess':('roi_bounds_xyxy','adaptive_method','block_size'),
          'color_card.geometry':('min_size','aspect_ratio','solidity'),'color_card.mask':('radius',),
          'color_card.scale':('color_chip_size',),'color_card.means':(), 'color_card.standard':('pos',),
          'color_card.affine':()}
    if operation not in keys or not isinstance(parameters,Mapping) or set(parameters)!=set(keys[operation]):
        raise NativeCardReplayError('Card controls differ from the operation contract.')
    try:
        if operation=='color_card.preprocess':
            native.preprocess._parameters(parameters['adaptive_method'],parameters['block_size'],native.preprocess.MAX_TILE_SIDE,None)
        elif operation=='color_card.geometry':native.card._parameters(0,parameters['min_size'],parameters['aspect_ratio'],parameters['solidity'],None)
        elif operation=='color_card.mask':native.card._parameters(parameters['radius'],0,1.27,.8,None)
        elif operation=='color_card.scale':native.card._parameters(0,0,1.27,.8,parameters['color_chip_size'])
        elif operation=='color_card.standard':
            if type(parameters['pos']) is not int or not 0<=parameters['pos']<=3:
                raise NativeCardReplayError('The card graph requires integer pos0..3.')
    except (ValueError,TypeError,OverflowError) as error:raise NativeCardReplayError(str(error)) from error
    return parameters


def validate(value,role,source_shape,bounds=None,*,poll=None,check_versions=True):
    """Validate structural roles; consistent changed numerics remain comparable."""
    value=canonical(value,poll)
    source,local,resolved=frames(source_shape,bounds)
    if role=='bounds':
        if not _sequence(value,4) or any(type(v) is not int for v in value):raise NativeCardReplayError('Invalid card bounds.')
        frames(source,value)
    elif role in ('geometry','measurements'):
        expected=GEOMETRY_FIELDS if role=='geometry' else MEASUREMENT_FIELDS
        if not isinstance(value,Mapping) or set(value)!=expected:raise NativeCardReplayError('Invalid card geometry/measurement fields.')
        count=value['detected_chips'];maximum=native.contours.size.MAX_CONTOUR_COMPONENTS
        if type(count) is not int or not 1<=count<=maximum:raise NativeCardReplayError('Invalid detected contour count.')
        if not all(_number(value[key],positive=True) for key in MEASUREMENT_FIELDS-{'detected_chips'}):
            raise NativeCardReplayError('Card medians must be positive finite numbers.')
        if role=='geometry':
            indices=value['selected_contour_indices']
            if (not _sequence(indices,count) or any(type(v) is not int or not 0<=v<maximum for v in indices)
                    or any(a>=b for a,b in zip(indices,indices[1:]))):
                raise NativeCardReplayError('Invalid source-ordered selected contour IDs.')
            for key in ('areas','widths','heights'):
                if not _sequence(value[key],count) or not all(_number(v,positive=True) for v in value[key]):
                    raise NativeCardReplayError('Selected contour dimensions differ from their count.')
            rect=value['bounding_rect']
            if (not _sequence(rect,4) or any(type(v) is not int for v in rect)
                    or min(rect[:2])<0 or min(rect[2:])<=0 or rect[0]+rect[2]>local[1] or rect[1]+rect[3]>local[0]):
                raise NativeCardReplayError('Card bounding rectangle lies outside the ROI frame.')
            for key,count_points in (('centers_xy',24),('corners_xy',4)):
                if (not _sequence(value[key],count_points) or any(not _sequence(row,2)
                        or any(type(v) is not int or not -(2**31)<=v<2**31 for v in row) for row in value[key])):
                    raise NativeCardReplayError('Invalid ordered card points.')
            if any(not 0<=x<local[1] or not 0<=y<local[0] for x,y in value['corners_xy']):
                raise NativeCardReplayError('Card corners lie outside the exclusive ROI frame.')
            if not _matrix(value['transformation'],(3,3),3):raise NativeCardReplayError('Invalid card3x3 transformation.')
            if (value['grid_box_sha256']!=native.card.GRID_BOX_SHA256
                    or value['reference_sources']!=canonical(native.card.REFERENCE_SOURCES,poll)
                    or type(value['opencv_version']) is not str or type(value['numpy_version']) is not str
                    or check_versions and (value['opencv_version']!=native.card.cv2.__version__
                        or value['numpy_version']!=native.np.__version__)):
                raise NativeCardReplayError('Card geometry provenance differs from the loaded primitive.')
    elif role=='scale':
        if not isinstance(value,Mapping) or set(value)!={'px_width','px_height','unit'}:
            raise NativeCardReplayError('Invalid card scale fields.')
        if not (all(value[key] is None for key in value)
                or value['unit']=='mm' and all(_number(value[key],positive=True) for key in ('px_width','px_height'))):
            raise NativeCardReplayError('Card scale must be absent or positive finite millimetres per pixel.')
    elif role in ('chip_matrix','standard_matrix','fit_matrix'):
        rows=(24,24) if role=='standard_matrix' else (0 if role=='chip_matrix' else 1,native.correction.MAX_CHIPS)
        if not _matrix(value,rows,4,normalized=True):raise NativeCardReplayError('Invalid finite normalized card colour matrix.')
    elif role=='affine_matrix':
        if not _matrix(value,(4,4),3):raise NativeCardReplayError('Invalid card4x3 affine matrix.')
    elif role=='affine_metadata':
        if (not isinstance(value,Mapping) or set(value)!={'fit','numpy_version'}
                or value['fit']!='affine' or type(value['numpy_version']) is not str
                or check_versions and value['numpy_version']!=native.np.__version__):
            raise NativeCardReplayError('Invalid card affine-fit provenance.')
    else:raise NativeCardReplayError('Unknown compact card product role.')
    return value


def compare(actual,expected,poll=None):
    """Compare every finite compact leaf and ordered sequence element."""
    actual,expected=canonical(actual,poll),canonical(expected,poll)
    missing=object()
    shape_equal=True
    def visit(left,right):
        nonlocal shape_equal
        if isinstance(left,Mapping) or isinstance(right,Mapping):
            if not isinstance(left,Mapping) or not isinstance(right,Mapping):
                shape_equal=False;return 1,1
            if set(left)!=set(right):shape_equal=False
            rows=[visit(left.get(key,missing),right.get(key,missing)) for key in sorted(set(left)|set(right))]
        elif type(left) in (tuple,list) or type(right) in (tuple,list):
            if type(left) is not type(right):
                shape_equal=False;return 1,1
            if len(left)!=len(right):shape_equal=False
            rows=[visit(left[i] if i<len(left) else missing,right[i] if i<len(right) else missing)
                  for i in range(max(len(left),len(right)))]
        else:
            if poll is not None:poll()
            return int(type(left) is not type(right) or left!=right),1
        return sum(v[0] for v in rows),sum(v[1] for v in rows)
    mismatch,count=visit(actual,expected)
    return {'kind':'card_compact','mismatched_elements':mismatch,'compared_elements':count,'shape_equal':shape_equal}


def _rounded(size):return ((size+ALLOCATION_BYTES-1)//ALLOCATION_BYTES)*ALLOCATION_BYTES


@dataclass(frozen=True,slots=True)
class CardReplayPlan:
    selected_bytes:int
    cuda_bytes:int
    primitive_scratch_bytes:int
    work:int|None
    work_details:tuple[tuple[str,int],...]


def plan(operation,source_shape,parameters,*,bounds=None,selected_limit=MAX_SELECTED_BYTES,cuda_limit=MAX_CUDA_BYTES):
    controls(operation,parameters)
    if operation=='color_card.preprocess':bounds=parameters['roi_bounds_xyxy']
    source,local,bounds=frames(source_shape,bounds)
    p,r=math.prod(source),math.prod(local)
    if p>MAX_PIXELS:raise NativeCardReplayUnavailable('Card source frame exceeds its pixel domain.')
    if any(type(cap) is not int or cap<0 for cap in (selected_limit,cuda_limit)):
        raise NativeCardReplayError('Card replay limits must be nonnegative integers.')
    inputs,outputs,_=PORTS[operation]
    def nbytes(role):
        shape,dtype=raster_spec(role,source,bounds)
        return math.prod(shape)*(8 if dtype=='<f8' else 1)
    retained=[nbytes(role) for role in (*inputs.values(),*outputs.values()) if role in RASTER_ROLES]
    expected=[nbytes(role) for role in outputs.values() if role in RASTER_ROLES]
    selected=sum(retained);live=sum(_rounded(size) for size in retained)+sum(_rounded(size) for size in expected)
    if any(size>MAX_ARRAY_BYTES for size in retained) or selected>MAX_TOTAL_ARRAY_BYTES:
        raise NativeCardReplayUnavailable('Card replay exceeds checkpoint array admission.')
    largest=max(expected,default=0)
    # A float64 raster still needs one comparison element per pixel.
    if operation=='color_card.mask':largest=p
    tile=min(largest,COMPARE_ELEMENTS)
    compare_bytes=(_rounded(tile)+_rounded(8*tile)+ALLOCATION_BYTES) if tile else 0
    scratch=work=None;details=()
    try:
        if operation=='color_card.preprocess':
            native.card._preflight(*source,bounds,parameters['adaptive_method'],parameters['block_size'],0,correction=True)
            work,scratch=native.preprocess._preflight(*local,parameters['block_size'],native.preprocess.MAX_TILE_SIDE,
                                                   method=parameters['adaptive_method'],complete=True)
            details=(('roi_pixels',r),)
        elif operation=='color_card.geometry':
            extraction=max(3*r,r+64*native.contours.MAX_BOUNDARY_POINTS+64*native.contours.MAX_TILE_PIXELS)
            if local[0]>native.contours.size.MAX_CONTOUR_SCAN_ROWS or extraction>native.contours.MAX_SCRATCH_BYTES:
                raise NativeCardReplayUnavailable('Card contour extraction exceeds its row/scratch domain.')
            scratch=native.contours.MAX_SCRATCH_BYTES+4096
            details=(('max_geometry_work',native.card.MAX_GEOMETRY_WORK),('max_boundary_points',native.contours.MAX_BOUNDARY_POINTS))
        elif operation=='color_card.mask':
            radius=parameters['radius'];native.card._circle_budget(*local,radius)
            work=24*min(local[0],2*radius+1)*min(local[1],2*radius+1)
        elif operation=='color_card.means':
            live+=_rounded(p)
            scratch=40*min(p,native.correction.TILE_PIXELS)+256*64+native.correction.MAX_CHIPS*128+2*1024*1024
            # Float64 label validation precedes the narrowing cast.
            compare_bytes=max(compare_bytes,16*min(p,COMPARE_ELEMENTS)+4096)
            details=(('chip_pixel_scans',p),)
        elif operation=='color_card.standard':scratch=8192
        elif operation=='color_card.affine':
            live+=_rounded(3*p)
            scratch=192*min(p,native.correction.TILE_PIXELS)+2*1024*1024
            details=(('affine_pixel_applications',p),)
    except ValueError as error:raise NativeCardReplayUnavailable(str(error)) from error
    scratch=scratch or 0
    # Mask labels are checked for finite integral chip IDs before calculation.
    if operation=='color_card.mask':compare_bytes=max(compare_bytes,16*min(p,COMPARE_ELEMENTS)+4096)
    live+=scratch+compare_bytes+AUDIT_CUDA_BYTES
    if selected>selected_limit or live>cuda_limit:
        raise NativeCardReplayUnavailable('Card replay exceeds selected-array or CUDA admission.')
    return CardReplayPlan(selected,live,scratch,work,details)


PINNED_STATE=state()
