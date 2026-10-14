"""Typed saved-input contracts for the provisional SMPTS paper reconstruction.

Agreement establishes deterministic engineering replay, not original capsule
execution, recovered seed outlines or scientific validation.
"""
from collections.abc import Mapping
import math
from types import MappingProxyType
from . import native_smpts_recipe as native

MAX_METADATA_BYTES = 32*1024*1024
MAX_METADATA_ITEMS = 1_000_000
MAX_METADATA_DEPTH = 16
MAX_SELECTED_BYTES = 128*1024*1024
MAX_CUDA_BYTES = 1024*1024*1024
COMPARE_ELEMENTS = 262_144
AUDIT_CUDA_BYTES = 64*1024
ALLOCATION_BYTES = 512
PORTS = MappingProxyType({
    'smpts_paper.mean': ({'gray':'gray'}, {'mask':'binary'}, 'mask'),
    'smpts_paper.fixed': ({'gray':'gray'}, {'mask':'binary'}, 'mask'),
    'smpts_paper.median': ({'mask':'binary'}, {'mask':'binary'}, 'mask'),
    'smpts_paper.intersection': ({'adaptive':'binary','global':'binary'}, {'mask':'binary'}, 'mask'),
    'smpts_paper.components': ({'mask':'binary'}, {'labels':'labels','components':'inventory'}, 'labels'),
    'smpts_paper.contacts': ({'labels':'labels','components':'inventory'}, {'selected_parent_ids':'selected'}, None),
    'smpts_paper.erosion': ({'mask':'binary','labels':'labels','components':'inventory','selected_parent_ids':'selected'},
        {'mask':'binary','selected_mask':'binary','eroded_selected_mask':'binary'}, 'mask'),
    'smpts_paper.crops': ({'rgb':'rgb','source_labels':'labels','source_components':'inventory',
        'final_labels':'labels','final_components':'inventory'}, {'labels':'labels','packed_rgb':'packed_rgb',
        'packed_mask':'packed_mask','crops':'crops','disappeared_parent_ids':'disappeared','metadata':'metadata'}, 'labels'),
})
PORTS = MappingProxyType({key:(MappingProxyType(inputs),MappingProxyType(outputs),preview)
    for key,(inputs,outputs,preview) in PORTS.items()})
RASTER_ROLES = frozenset(('gray','binary','labels','rgb','packed_rgb','packed_mask'))
COMPONENT_FIELDS = ('label_id','area_pixels','bounds_xyxy')
CROP_FIELDS = (*COMPONENT_FIELDS,'parent_id','offset_pixels','crop_shape')


class NativePaperReplayError(ValueError):
    """Malformed archived paper contract."""


class NativePaperReplayUnavailable(NativePaperReplayError):
    """Source or selected replay resource domain exceeded."""


def state():
    paper = native.paper
    return (MAX_METADATA_BYTES,MAX_METADATA_ITEMS,MAX_METADATA_DEPTH,MAX_SELECTED_BYTES,
        MAX_CUDA_BYTES,COMPARE_ELEMENTS,AUDIT_CUDA_BYTES,ALLOCATION_BYTES,
        tuple((key,tuple(inputs.items()),tuple(outputs.items()),preview)
            for key,(inputs,outputs,preview) in PORTS.items()),tuple(sorted(RASTER_ROLES)),
        COMPONENT_FIELDS,CROP_FIELDS,native.RECIPE_ID,native.VERSION,
        paper.MAX_PIXELS,paper.MAX_COMPONENTS,paper.MAX_CUDA_BYTES,paper.MAX_CROP_BYTES,
        paper.MAX_CPU_METADATA_BYTES,paper.BACKEND_BYTES,paper.MAX_COMPONENT_WORK,
        paper.MAX_WINDOW_WORK,paper.TILE,paper.UNION_ROUNDS,paper.JUMPS_PER_ROUND,
        paper.FINAL_JUMPS,paper.MAX_JUMP_PASSES,paper.PROFILE,paper.PAPER_SHA256,
        paper.CUDA_SOURCE_SHA256)


def _compact(root,encoded,poll):
    visits=bytes_used=0
    active=set()
    max_bytes,max_items,max_depth=MAX_METADATA_BYTES,MAX_METADATA_ITEMS,MAX_METADATA_DEPTH
    def charge(size,items=0):
        nonlocal visits,bytes_used
        visits+=items;bytes_used+=size
        if visits>max_items or bytes_used>max_bytes:
            raise NativePaperReplayError('Paper metadata exceeds its encoded-byte/item bound.')
        if items and poll is not None and visits%1024==0:poll()
    def text(value):
        if len(value)>max_bytes:raise NativePaperReplayError('Paper text exceeds its bound.')
        charge(2)
        for index,char in enumerate(value):
            code=ord(char)
            if 0xd800<=code<=0xdfff:raise NativePaperReplayError('Paper text must be valid UTF-8.')
            charge(2 if char in '\\"\b\f\n\r\t' else 6 if code<32 else
                1 if code<128 else 2 if code<2048 else 3 if code<65536 else 4)
            if poll is not None and index and index%1024==0:poll()
        return value
    def visit(value,depth):
        if depth>max_depth:raise NativePaperReplayError('Paper metadata exceeds its depth bound.')
        charge(0,1)
        if value is None or type(value) is bool:
            charge(4 if value is None or value is True else 5);return value
        if type(value) is int:
            if value.bit_length()>256:raise NativePaperReplayError('Paper integer exceeds its bound.')
            charge(len(str(value)));return value
        if type(value) is float:
            if not math.isfinite(value):raise NativePaperReplayError('Paper numbers must be finite.')
            charge(len(repr(value)));return value
        if type(value) is str:return text(value)
        if encoded:
            if (type(value) is not dict or set(value)!={'kind','items'}
                    or type(value['kind']) is not str or value['kind'] not in ('mapping','tuple','list')
                    or type(value['items']) is not list):
                raise NativePaperReplayError('Paper metadata requires primitive tagged mappings/sequences.')
            kind,entries=value['kind'],value['items']
        elif isinstance(value,Mapping):kind,entries='mapping',value.items()
        elif type(value) in (tuple,list):kind,entries=('tuple' if type(value) is tuple else 'list'),value
        else:raise NativePaperReplayError('Paper metadata contains an unsupported value.')
        if len(entries)>max_items:raise NativePaperReplayError('Paper container exceeds its item bound.')
        if id(value) in active:raise NativePaperReplayError('Paper metadata contains a cycle.')
        active.add(id(value))
        try:
            charge(len('{"kind":"'+kind+'","items":[]}')+max(0,len(entries)-1))
            if kind=='mapping':
                result={}
                for entry in entries:
                    if (type(entry) not in (tuple,list) or len(entry)!=2
                            or type(entry[0]) is not str or entry[0] in result):
                        raise NativePaperReplayError('Paper mapping keys must be unique strings.')
                    if depth+1>max_depth:raise NativePaperReplayError('Paper metadata exceeds its depth bound.')
                    charge(3,1);result[text(entry[0])]=visit(entry[1],depth+1)
                return MappingProxyType(result)
            values=[visit(entry,depth+1) for entry in entries]
            return tuple(values) if kind=='tuple' else values
        finally:active.remove(id(value))
    return visit(root,0)


def decode(binding,poll=None):return _compact(binding,True,poll)


def canonical(value,poll=None):return _compact(value,False,poll)



def _keys(value,keys,name):
    if not isinstance(value,Mapping) or set(value)!=set(keys):
        raise NativePaperReplayError(name+' has missing or unknown fields.')


def _integer(value,low,high,name):
    if type(value) is not int or not low<=value<=high:
        raise NativePaperReplayError(name+' requires a bounded integer.')
    return value


def controls(operation,parameters):
    wanted = {
        'smpts_paper.mean':('ksize','offset','object_type'),
        'smpts_paper.fixed':('threshold','object_type'),
        'smpts_paper.median':('ksize',), 'smpts_paper.erosion':('ksize',),
        'smpts_paper.contacts':('contact_box_area',),
    }.get(operation,())
    if operation not in PORTS:raise NativePaperReplayError('Unknown paper operation.')
    _keys(parameters,wanted,'Paper controls')
    if 'ksize' in parameters:
        k=_integer(parameters['ksize'],3 if operation=='smpts_paper.mean' else 1,
            255 if operation=='smpts_paper.mean' else 31,'Window')
        if not k%2:raise NativePaperReplayError('Paper window must be odd.')
    if 'offset' in parameters:
        value=parameters['offset']
        if type(value) not in (int,float) or not math.isfinite(value) or not -255<=value<=255:
            raise NativePaperReplayError('Mean offset must be finite and in [-255,255].')
    if 'object_type' in parameters and (type(parameters['object_type']) is not str
            or parameters['object_type'] not in ('light','dark')):
        raise NativePaperReplayError('Paper polarity requires light or dark.')
    if 'threshold' in parameters:_integer(parameters['threshold'],0,255,'Fixed cutoff')
    if 'contact_box_area' in parameters:_integer(parameters['contact_box_area'],0,native.paper.MAX_PIXELS,'Contact cutoff')


def inventory(value,shape):
    if type(value) is not tuple or len(value)>native.paper.MAX_COMPONENTS:
        raise NativePaperReplayError('Component inventory requires a bounded ordered tuple.')
    area=0
    for index,row in enumerate(value):
        _keys(row,COMPONENT_FIELDS,'Component row')
        _integer(row['label_id'],index+1,index+1,'Consecutive component ID')
        box=row['bounds_xyxy']
        if type(box) is not tuple or len(box)!=4 or any(type(x) is not int for x in box):
            raise NativePaperReplayError('Component bounds require four integer coordinates.')
        x0,y0,x1,y1=box
        if not 0<=x0<x1<=shape[1] or not 0<=y0<y1<=shape[0]:
            raise NativePaperReplayError('Component box must be nonempty and inside the source frame.')
        area+=_integer(row['area_pixels'],1,(x1-x0)*(y1-y0),'Component area')
    if area>math.prod(shape):raise NativePaperReplayError('Component areas exceed the source frame.')
    return value


def ids(value,count,name):
    if (type(value) is not tuple or len(value)>count
            or any(type(i) is not int or not 1<=i<=count for i in value)
            or tuple(sorted(set(value)))!=value):
        raise NativePaperReplayError(name+' requires sorted unique inventory IDs.')
    return value


def crop_total(rows):
    return sum((row['bounds_xyxy'][2]-row['bounds_xyxy'][0])
        *(row['bounds_xyxy'][3]-row['bounds_xyxy'][1]) for row in rows)


def assumptions(count):
    return {'status':'provisional_paper_reconstruction','profile':native.paper.PROFILE,
        'paper_sha256':native.paper.PAPER_SHA256,'original_execution':False,'scientific_validation':False,
        'component_connectivity':8,'contact_metric':'axis-aligned bbox width*height',
        'contact_comparison':'strict > selected cutoff','median_border':'replicate',
        'erosion':'square, one iteration, replicate','bounds':'source-frame half-open xyxy',
        'crop_pixels':'original RGB masked by final component; packed row-major',
        'crop_count':count,'not_recovered_measurement_outlines':True}


def validate(value,role,shape,context=None):
    context=context or {}
    if role=='inventory':return inventory(value,shape)
    if role in ('selected','disappeared'):
        rows=context.get('source_components' if role=='disappeared' else 'components',())
        return ids(value,len(rows),role)
    if role=='crops':
        final=context['final_components'];source=context['source_components']
        if type(value) is not tuple or len(value)!=len(final):
            raise NativePaperReplayError('Crop rows must match the complete final inventory.')
        offset=0
        for row,component in zip(value,final):
            _keys(row,CROP_FIELDS,'Crop row')
            # These are transport identities, not remeasured ancestor statistics.
            if any(type(row[k]) is not type(component[k]) or row[k]!=component[k]
                    for k in COMPONENT_FIELDS if k!='bounds_xyxy') or row['bounds_xyxy']!=component['bounds_xyxy']:
                raise NativePaperReplayError('Crop row differs from its final component producer.')
            if (type(row['bounds_xyxy']) is not tuple or len(row['bounds_xyxy'])!=4
                    or any(type(x) is not int for x in row['bounds_xyxy'])):
                raise NativePaperReplayError('Crop bounds require four exact integer coordinates.')
            _integer(row['parent_id'],1,len(source),'Crop source parent')
            _integer(row['offset_pixels'],offset,offset,'Consecutive packed offset')
            x0,y0,x1,y1=row['bounds_xyxy']
            if (type(row['crop_shape']) is not tuple or len(row['crop_shape'])!=2
                    or any(type(x) is not int for x in row['crop_shape'])
                    or row['crop_shape']!=(y1-y0,x1-x0)):
                raise NativePaperReplayError('Crop shape differs from its half-open bounds.')
            offset+=(x1-x0)*(y1-y0)
        if 4*offset>native.paper.MAX_CROP_BYTES:
            raise NativePaperReplayUnavailable('Packed crops exceed their aggregate source byte limit.')
        return value
    if role=='metadata':
        expected=assumptions(len(context['final_components']))
        _keys(value,expected,'Paper assumptions')
        if any(type(value[k]) is not type(v) or value[k]!=v for k,v in expected.items()):
            raise NativePaperReplayError('Paper assumption metadata differs from its provisional contract.')
        return value
    raise NativePaperReplayError('Unknown paper compact role.')


def raster_spec(role,shape,context=None):
    context=context or {}
    if role in ('packed_rgb','packed_mask'):
        total=crop_total(context['final_components'])
        if 4*total>native.paper.MAX_CROP_BYTES:
            raise NativePaperReplayUnavailable('Packed crop headers exceed the source byte limit.')
        return ((3*total if role=='packed_rgb' else total,), '|u1')
    return ((*shape,3) if role=='rgb' else tuple(shape),'<i4' if role=='labels' else '|u1')


def compare(actual,expected,poll=None):
    # Compare every primitive leaf and preserve sequence types and ordering.
    missing=object();shape_equal=True;visits=0
    def visit(left,right):
        nonlocal shape_equal,visits
        visits+=1
        if poll is not None and visits%1024==0:poll()
        if isinstance(left,Mapping) or isinstance(right,Mapping):
            if not isinstance(left,Mapping) or not isinstance(right,Mapping):shape_equal=False;return 1,1
            if set(left)!=set(right):shape_equal=False
            children=((left.get(k,missing),right.get(k,missing)) for k in sorted(set(left)|set(right)))
        elif type(left) in (tuple,list) or type(right) in (tuple,list):
            if type(left) is not type(right):shape_equal=False;return 1,1
            if len(left)!=len(right):shape_equal=False
            children=((left[i] if i<len(left) else missing,right[i] if i<len(right) else missing)
                for i in range(max(len(left),len(right))))
        else:return int(type(left) is not type(right) or left!=right),1
        mismatch=count=0
        for a,b in children:
            m,n=visit(a,b);mismatch+=m;count+=n
        return mismatch,count
    mismatch,count=visit(actual,expected)
    if poll is not None:poll()
    return {'kind':'paper_compact','mismatched_elements':mismatch,'compared_elements':count,'shape_equal':shape_equal}


def rounded(size):return ((size+ALLOCATION_BYTES-1)//ALLOCATION_BYTES)*ALLOCATION_BYTES


def plan(operation,shape,parameters,*,selected_allocations,input_allocations,crop_pixels=0,
         selected_limit=MAX_SELECTED_BYTES,cuda_limit=MAX_CUDA_BYTES):
    controls(operation,parameters)
    if (type(shape) not in (tuple,list) or len(shape)!=2
            or any(type(v) is not int or v<=0 for v in shape)):
        raise NativePaperReplayError('Paper frame requires positive integer height and width.')
    for sizes in (selected_allocations,input_allocations):
        if (type(sizes) not in (tuple,list) or len(sizes)>16
                or any(type(v) is not int or v<0 for v in sizes)):
            raise NativePaperReplayError('Paper allocation facts require bounded nonnegative byte counts.')
    for cap in (selected_limit,cuda_limit):_integer(cap,0,2**63-1,'Replay cap')
    _integer(crop_pixels,0,native.paper.MAX_CROP_BYTES//4,'Packed pixels')
    selected_bytes=sum(selected_allocations);retained=sum(rounded(v) for v in input_allocations)
    selected=sum(rounded(v) for v in selected_allocations)
    if retained>selected:raise NativePaperReplayError('Input allocation facts exceed selected allocations.')
    pixels=math.prod(shape);paper=native.paper
    try:work=paper.component_work(pixels)
    except paper.PaperResourceError as error:raise NativePaperReplayUnavailable(str(error)) from error
    window=0;extra=0;source_extra=0
    if operation=='smpts_paper.mean':
        # Prefix/index-select int64 planes, byte predicates and axis vectors.
        # At most 64P bytes of certification planes and 32P bytes of
        # prefix planes plus <=65*max(H,W) axis-vector bytes are live.
        # The 128P envelope also covers allocation rounding and temporaries.
        # This conservative envelope includes the certification branch. The
        # source uses O(P) prefix sums, not a P*k*k sample-domain restriction.
        extra=source_extra=128*pixels+paper.BACKEND_BYTES
    elif operation=='smpts_paper.fixed':extra=source_extra=3*rounded(pixels)
    elif operation in ('smpts_paper.median','smpts_paper.intersection'):
        source_extra=pixels+paper.BACKEND_BYTES
        extra=rounded(pixels)+paper.BACKEND_BYTES+2*ALLOCATION_BYTES
        if operation.endswith('median'):window=pixels*parameters['ksize']**2
    elif operation=='smpts_paper.components':
        source_extra=24*pixels+paper.BACKEND_BYTES+paper.MAX_COMPONENTS*32
        extra=rounded(24*pixels)+paper.BACKEND_BYTES+paper.MAX_COMPONENTS*32+16*ALLOCATION_BYTES
    elif operation=='smpts_paper.erosion':
        window=2*parameters['ksize']*pixels
        source_extra=4*pixels+paper.BACKEND_BYTES+paper.MAX_COMPONENTS*32
        extra=4*rounded(pixels)+paper.BACKEND_BYTES+paper.MAX_COMPONENTS*32+16*ALLOCATION_BYTES
    elif operation=='smpts_paper.crops':
        source_extra=4*crop_pixels+paper.BACKEND_BYTES+paper.MAX_COMPONENTS*64
        extra=rounded(3*crop_pixels)+rounded(crop_pixels)+paper.BACKEND_BYTES+paper.MAX_COMPONENTS*64+16*ALLOCATION_BYTES
    if window>paper.MAX_WINDOW_WORK:
        raise NativePaperReplayUnavailable('Paper operation exceeds its selected window-sample limit.')
    if retained+source_extra>paper.MAX_CUDA_BYTES:
        raise NativePaperReplayUnavailable('Paper operation exceeds its source CUDA admission.')
    tile=min(max(pixels,3*crop_pixels),COMPARE_ELEMENTS)
    comparison=rounded(tile)+rounded(8*tile)+ALLOCATION_BYTES
    live=selected+extra+comparison+AUDIT_CUDA_BYTES
    if selected_bytes>selected_limit or live>cuda_limit:
        raise NativePaperReplayUnavailable('Paper operation exceeds selected-array or CUDA replay admission.')
    return {'selected_bytes':selected_bytes,'selected_limit_bytes':selected_limit,'cuda_bytes':live,
        'cuda_limit_bytes':cuda_limit,'primitive_scratch_bytes':extra,'comparison_scratch_bytes':comparison,
        'runtime_audit_cuda_reserve_bytes':AUDIT_CUDA_BYTES,'component_work':work,
        'window_sample_work':window,'crop_pixels':crop_pixels,'compact_metadata_limit_bytes':MAX_METADATA_BYTES,
        'bounds':'encoded metadata and modeled tensor storage; not process RSS or elapsed time'}


PINNED_STATE=state()
