"""Primitive contracts and allocation-free admission for recovery stage replay.

These are experimental native operations. Saved-input agreement establishes
computational reproducibility, not recovery accuracy or SMPTS conformance.
"""
from collections.abc import Mapping
import math
from types import MappingProxyType

from . import native_core_recovery_recipe as native
from .checkpoints import CoordinateFrame

MAX_METADATA_BYTES = 64*1024*1024
MAX_METADATA_ITEMS = 1_000_000
MAX_METADATA_DEPTH = 16
MAX_SELECTED_BYTES = 128*1024*1024
MAX_CUDA_BYTES = 1024*1024*1024
COMPARE_ELEMENTS = 262_144
AUDIT_CUDA_BYTES = 64*1024
ALLOCATION_BYTES = 512
PORTS = MappingProxyType({
    'recovery.core_labels': (MappingProxyType({'mask':'binary'}),
        MappingProxyType({'markers':'markers','inventory':'inventory'}),'markers'),
    'recovery.validity': (MappingProxyType({'mask':'binary','source_frame':'source_binding'}),
        MappingProxyType({'validity':'binary','metadata':'validity_metadata'}),'validity'),
    'recovery.clip_cores': (MappingProxyType({'markers':'markers','validity':'binary'}),
        MappingProxyType({'markers':'markers'}),'markers'),
    'recovery.partition': (MappingProxyType({'foreground':'binary','markers':'markers',
        'validity':'binary','elevation':'elevation','inventory':'inventory','source_frame':'source_binding'}),
        MappingProxyType({'partition':'partition','labels':'markers','metadata':'partition_metadata'}),'labels'),
    'recovery.annotation': (MappingProxyType({'partition':'partition'}),
        MappingProxyType({'annotation':'annotation','uncertainty_band':'unit_binary','metadata':'annotation_metadata'}),'uncertainty_band'),
    'recovery.measurements': (MappingProxyType({'annotation':'annotation_metadata_only'}),
        MappingProxyType({'observations':'observations','metadata':'annotation_metadata'}),None),
})
RASTER_ROLES = frozenset(('binary','markers','unit_binary','elevation','ids'))
PARTITION_TENSORS = MappingProxyType({'original_foreground':'binary','markers':'markers',
    'validity':'binary','elevation':'elevation','labels':'markers','domain':'unit_binary',
    'unresolved':'unit_binary','interface':'unit_binary','label_ids':'ids'})
STATUSES = ('empty_foreground','all_foreground_excluded_by_validity','unresolved_foreground','partitioned')
REASONS = ('inferred_shared_boundary','canvas_boundary','invalid_support','unresolved_neighbor')
OBJECT_FIELDS = ('label_id','seed_pixels','assigned_pixels','interface_pixels','bbox',
    'touches_canvas','touches_invalid_support','touches_unresolved')
PLAN_FIELDS = ('height','width','original_foreground_pixels','domain_pixels',
    'validity_clipped_foreground_pixels','seed_pixels','labels','input_backing_bytes',
    'watershed_workspace_bytes','peak_live_bytes','cpu_metadata_bytes','work')
TRAITS = ('assigned_area','bounding_width','bounding_height')


class NativeRecoveryReplayError(ValueError):
    """An archived recovery contract is malformed."""


class NativeRecoveryReplayUnavailable(NativeRecoveryReplayError):
    """The operation exceeds its source or replay resource domain."""


def state():
    return (MAX_METADATA_BYTES,MAX_METADATA_ITEMS,MAX_METADATA_DEPTH,MAX_SELECTED_BYTES,
        MAX_CUDA_BYTES,COMPARE_ELEMENTS,AUDIT_CUDA_BYTES,ALLOCATION_BYTES,
        tuple((key,tuple(inputs.items()),tuple(outputs.items()),preview)
            for key,(inputs,outputs,preview) in PORTS.items()),tuple(sorted(RASTER_ROLES)),
        native.MAX_RASTER_PIXELS,native.MAX_CORE_LABELS,native.CORE_BATCH,native.VALIDATION_TILE,
        native.core.MAX_LIVE_BYTES,native.core.MAX_CPU_METADATA_BYTES,native.core.MAX_WORK,
        native.core.BACKEND_HEADROOM_BYTES,native.core.MAX_LABELS,native.core.MAX_ID_PROBES,
        native.core.ID_TABLE_SLOTS,native.core.VALIDATION_TILE_PIXELS,native.core.PROFILE,
        native.core._watershed.MAX_WORKSPACE_BYTES,native.GRAPH_METADATA_BYTES_PER_LABEL,
        native.GRAPH_METADATA_FIXED_BYTES,tuple(PARTITION_TENSORS.items()),STATUSES,REASONS,
        OBJECT_FIELDS,PLAN_FIELDS,TRAITS)


def _compact(root,encoded,poll):
    visits=bytes_used=0
    active=set()
    max_bytes,max_items,max_depth=MAX_METADATA_BYTES,MAX_METADATA_ITEMS,MAX_METADATA_DEPTH
    def charge(size,items=0):
        nonlocal visits,bytes_used
        visits+=items;bytes_used+=size
        if visits>max_items or bytes_used>max_bytes:
            raise NativeRecoveryReplayError('Recovery metadata exceeds its encoded-byte/item bound.')
        if items and poll is not None and visits%1024==0:poll()
    def text(value):
        if len(value)>max_bytes:raise NativeRecoveryReplayError('Recovery text exceeds its bound.')
        charge(2)
        for index,char in enumerate(value):
            code=ord(char)
            if 0xd800<=code<=0xdfff:raise NativeRecoveryReplayError('Recovery text must be valid UTF-8.')
            charge(2 if char in '\\"\b\f\n\r\t' else 6 if code<32 else
                1 if code<128 else 2 if code<2048 else 3 if code<65536 else 4)
            if poll is not None and index and index%1024==0:poll()
        return value
    def visit(value,depth):
        if depth>max_depth:raise NativeRecoveryReplayError('Recovery metadata exceeds its depth bound.')
        charge(0,1)
        if value is None or type(value) is bool:
            charge(4 if value is None or value is True else 5);return value
        if type(value) is int:
            if value.bit_length()>256:raise NativeRecoveryReplayError('Recovery integer exceeds its bound.')
            charge(len(str(value)));return value
        if type(value) is float:
            if not math.isfinite(value):raise NativeRecoveryReplayError('Recovery numbers must be finite.')
            charge(len(repr(value)));return value
        if type(value) is str:return text(value)
        if encoded:
            if (type(value) is not dict or set(value)!={'kind','items'}
                    or type(value['kind']) is not str or value['kind'] not in ('mapping','tuple','list')
                    or type(value['items']) is not list):
                raise NativeRecoveryReplayError('Recovery metadata requires primitive tagged mappings/sequences.')
            kind,entries=value['kind'],value['items']
        elif isinstance(value,Mapping):kind,entries='mapping',value.items()
        elif type(value) in (tuple,list):kind,entries=('tuple' if type(value) is tuple else 'list'),value
        else:raise NativeRecoveryReplayError('Recovery metadata contains an unsupported value.')
        if len(entries)>max_items:raise NativeRecoveryReplayError('Recovery container exceeds its item bound.')
        if id(value) in active:raise NativeRecoveryReplayError('Recovery metadata contains a cycle.')
        active.add(id(value))
        try:
            charge(len('{"kind":"'+kind+'","items":[]}')+max(0,len(entries)-1))
            if kind=='mapping':
                result={}
                for entry in entries:
                    if (type(entry) not in (tuple,list) or len(entry)!=2
                            or type(entry[0]) is not str or entry[0] in result):
                        raise NativeRecoveryReplayError('Recovery mapping keys must be unique strings.')
                    if depth+1>max_depth:raise NativeRecoveryReplayError('Recovery metadata exceeds its depth bound.')
                    charge(3,1);result[text(entry[0])]=visit(entry[1],depth+1)
                return MappingProxyType(result)
            values=[visit(entry,depth+1) for entry in entries]
            return tuple(values) if kind=='tuple' else values
        finally:active.remove(id(value))
    return visit(root,0)


def decode(binding,poll=None):return _compact(binding,True,poll)


def canonical(value,poll=None):return _compact(value,False,poll)


def bounds(shape,parameters):
    if (type(shape) not in (tuple,list) or len(shape)!=2
            or any(type(v) is not int or v<=0 for v in shape)):
        raise NativeRecoveryReplayError('Recovery shape requires positive integer height/width.')
    if not isinstance(parameters,Mapping) or set(parameters)!={'validity_roi'}:
        raise NativeRecoveryReplayError('Validity requires its exact ROI control.')
    height,width=shape;roi=parameters['validity_roi']
    if roi is None:return (0,0,width,height)
    if (not isinstance(roi,Mapping) or set(roi)!={'shape','bounds'}
            or type(roi['shape']) not in (tuple,list) or tuple(roi['shape'])!=tuple(shape)
            or any(type(v) is not int for v in roi['shape'])
            or type(roi['bounds']) not in (tuple,list) or len(roi['bounds'])!=4
            or any(type(v) is not int for v in roi['bounds'])):
        raise NativeRecoveryReplayError('Validity ROI requires exact shape and four integer bounds.')
    x0,y0,x1,y1=roi['bounds']
    if not 0<=x0<x1<=width or not 0<=y0<y1<=height:
        raise NativeRecoveryReplayError('Validity ROI must be nonempty and inside the full source frame.')
    return (x0,y0,x1,y1)


def _keys(value,keys,name):
    if not isinstance(value,Mapping) or set(value)!=set(keys):
        raise NativeRecoveryReplayError(name+' has missing or unknown fields.')


def _integer(value,low,high,name):
    if type(value) is not int or not low<=value<=high:
        raise NativeRecoveryReplayError(name+' requires a bounded integer.')
    return value


def downstream_count(count):
    try:return native.graph_metadata_bound(count)
    except native.core.CoreRecoveryResourceError as error:
        raise NativeRecoveryReplayUnavailable(str(error)) from error


def annotation_controls(parameters):
    _keys(parameters,('uncertainty_radius','measurement_policy'),'Annotation controls')
    _integer(parameters['uncertainty_radius'],0,8,'Uncertainty radius')
    if type(parameters['measurement_policy']) is not str or parameters['measurement_policy'] not in ('withhold_uncertain','flag_uncertain'):
        raise NativeRecoveryReplayError('Unknown annotation measurement policy.')


def _box(value,frame):
    if type(value) is not tuple or len(value)!=4 or any(type(v) is not int for v in value):
        raise NativeRecoveryReplayError('Object bbox requires four integer half-open coordinates.')
    x0,y0,x1,y1=value
    if not 0<=x0<x1<=frame.shape[1] or not 0<=y0<y1<=frame.shape[0]:
        raise NativeRecoveryReplayError('Object bbox lies outside the full source frame.')
    return (x1-x0)*(y1-y0)


def _object(value,frame):
    pixels=math.prod(frame.shape)
    _integer(value['label_id'],1,pixels,'Object ID')
    _integer(value['seed_pixels'],1,pixels,'Seed pixels')
    _integer(value['assigned_pixels'],value['seed_pixels'],pixels,'Assigned pixels')
    _integer(value['interface_pixels'],0,value['assigned_pixels'],'Interface pixels')
    if value['assigned_pixels']>_box(value['bbox'],frame):
        raise NativeRecoveryReplayError('Assigned pixels exceed the bounding rectangle.')
    if any(type(value[name]) is not bool for name in OBJECT_FIELDS[-3:]):
        raise NativeRecoveryReplayError('Object boundary facts require booleans.')


def _partition_metadata(value,frame,binding,poll):
    _keys(value,('schema','objects','plan','status','profile','unresolved_pixels',
        'interface_pixels','inventory','binding'),'Partition metadata')
    if value['schema']!='seedfiddle.core-recovery-partition.v1' or value['profile']!=native.core.PROFILE:
        raise NativeRecoveryReplayError('Partition schema/profile differs from the native source.')
    validate(value['binding'],'source_binding',frame,binding,{},poll)
    validate(value['inventory'],'inventory',frame,binding,{},poll)
    downstream_count(len(value['inventory']['objects']))
    expected={row['label_id']:row['original_core_pixels'] for row in value['inventory']['objects']}
    objects=value['objects'];plan=value['plan'];pixels=math.prod(frame.shape)
    if type(objects) is not tuple or len(objects)>len(expected):
        raise NativeRecoveryReplayError('Partition objects require a bounded tuple.')
    _keys(plan,PLAN_FIELDS,'Partition plan')
    for name in PLAN_FIELDS:_integer(plan[name],0,2**63-1,'Plan '+name)
    if (plan['height'],plan['width'])!=frame.shape or plan['labels']!=len(objects):
        raise NativeRecoveryReplayError('Partition plan shape/ID count differs from its records.')
    for name in ('original_foreground_pixels','domain_pixels','validity_clipped_foreground_pixels','seed_pixels'):
        _integer(plan[name],0,pixels,'Plan '+name)
    _integer(value['unresolved_pixels'],0,pixels,'Unresolved pixels')
    _integer(value['interface_pixels'],0,pixels,'Interface pixels')
    previous=seed=assigned=contacts=0
    for index,row in enumerate(objects):
        if poll is not None and index%1024==0:poll()
        _keys(row,OBJECT_FIELDS,'Partition object');_object(row,frame)
        if row['label_id']<=previous or row['label_id'] not in expected or row['seed_pixels']>expected[row['label_id']]:
            raise NativeRecoveryReplayError('Partition IDs/cores differ from their original inventory.')
        previous=row['label_id'];seed+=row['seed_pixels'];assigned+=row['assigned_pixels'];contacts+=row['interface_pixels']
    if (seed!=plan['seed_pixels'] or assigned+value['unresolved_pixels']!=plan['domain_pixels']
            or contacts!=value['interface_pixels']
            or plan['domain_pixels']+plan['validity_clipped_foreground_pixels']!=plan['original_foreground_pixels']):
        raise NativeRecoveryReplayError('Partition counts do not describe one valid foreground partition.')
    status=('empty_foreground' if not plan['original_foreground_pixels'] else
        'all_foreground_excluded_by_validity' if not plan['domain_pixels'] else
        'unresolved_foreground' if value['unresolved_pixels'] else 'partitioned')
    if value['status']!=status:raise NativeRecoveryReplayError('Partition status differs from its counts.')


def _annotation_metadata(value,frame,binding,parameters,poll):
    _keys(value,('schema','binding','partition_status','original_foreground_pixels',
        'validity_clipped_foreground_pixels','unresolved_pixels','uncertainty_radius',
        'measurement_policy','objects'),'Annotation metadata')
    if value['schema']!='seedfiddle.core-recovery-annotation.v1' or value['partition_status'] not in STATUSES:
        raise NativeRecoveryReplayError('Unknown annotation schema/partition status.')
    validate(value['binding'],'source_binding',frame,binding,{},poll)
    controls={name:value[name] for name in ('uncertainty_radius','measurement_policy')};annotation_controls(controls)
    if parameters and controls!=parameters:
        raise NativeRecoveryReplayError('Annotation metadata differs from its declared controls.')
    pixels=math.prod(frame.shape)
    for name in ('original_foreground_pixels','validity_clipped_foreground_pixels','unresolved_pixels'):
        _integer(value[name],0,pixels,name)
    rows=value['objects']
    if type(rows) is not tuple:raise NativeRecoveryReplayError('Annotation objects require a tuple.')
    downstream_count(len(rows));previous=assigned=0
    common=(*OBJECT_FIELDS[:5],'original_core_pixels','uncertainty_pixels','reasons',
        'measurement_permitted','uncontested_visible_partition','scientific_validation','clipped_core_pixels')
    for index,row in enumerate(rows):
        if poll is not None and index%1024==0:poll()
        if not isinstance(row,Mapping) or 'bbox' not in row:raise NativeRecoveryReplayError('Malformed annotation object.')
        disappeared=row['bbox'] is None
        _keys(row,common if disappeared else (*common,*OBJECT_FIELDS[-3:]),'Annotation object')
        _integer(row['label_id'],previous+1,pixels,'Annotation ID');previous=row['label_id']
        _integer(row['original_core_pixels'],1,pixels,'Original core pixels')
        for name in ('measurement_permitted','uncontested_visible_partition','scientific_validation'):
            if type(row[name]) is not bool:raise NativeRecoveryReplayError('Annotation eligibility requires booleans.')
        if row['scientific_validation'] is not False:raise NativeRecoveryReplayError('Recovery cannot claim scientific validation.')
        if disappeared:
            if any(type(row[name]) is not int or row[name]!=0 for name in ('seed_pixels','assigned_pixels','interface_pixels','uncertainty_pixels')):
                raise NativeRecoveryReplayError('Disappeared objects must have zero active counts.')
            reasons=('disappeared_by_validity',);allowed=uncontested=False
        else:
            _object(row,frame)
            _integer(row['uncertainty_pixels'],0,row['assigned_pixels'],'Uncertainty pixels')
            reasons=tuple(name for name,yes in zip(REASONS,(row['uncertainty_pixels']>0,
                row['touches_canvas'],row['touches_invalid_support'],row['touches_unresolved'])) if yes)
            allowed=not reasons or value['measurement_policy']=='flag_uncertain';uncontested=not reasons
        if (type(row['reasons']) is not tuple or row['reasons']!=reasons
                or row['measurement_permitted']!=allowed or row['uncontested_visible_partition']!=uncontested
                or type(row['clipped_core_pixels']) is not int
                or row['clipped_core_pixels']!=row['original_core_pixels']-row['seed_pixels']
                or row['clipped_core_pixels']<0):
            raise NativeRecoveryReplayError('Annotation reasons, eligibility or clipped core counts disagree.')
        assigned+=row['assigned_pixels']
    if assigned+value['unresolved_pixels']+value['validity_clipped_foreground_pixels']!=value['original_foreground_pixels']:
        raise NativeRecoveryReplayError('Annotation coverage counts differ from its original foreground.')


def _observations(value,frame,poll):
    if not isinstance(value,Mapping):raise NativeRecoveryReplayError('Measurements require a sample mapping.')
    downstream_count(len(value));pixels=math.prod(frame.shape);previous=0
    for index,(sample,rows) in enumerate(value.items()):
        if poll is not None and index%1024==0:poll()
        if (type(sample) is not str or len(sample)>5+len(str(pixels)) or not sample.startswith('core_')
                or not sample[5:].isascii() or not sample[5:].isdigit()):
            raise NativeRecoveryReplayError('Measurement sample must name one core ID.')
        label_id=int(sample[5:]);_integer(label_id,previous+1,pixels,'Measurement ID');previous=label_id
        if sample!='core_'+str(label_id):raise NativeRecoveryReplayError('Measurement IDs require canonical decimal names.')
        _keys(rows,TRAITS,'Measurement traits')
        for name,row in rows.items():
            _keys(row,('value','available','trait','method','scale','datatype','reasons','scientific_validation'),'Measurement record')
            if (type(row['available']) is not bool or row['scientific_validation'] is not False
                    or row['trait']!=name.replace('_',' ') or row['method']!='seedfiddle.experimental.core_recovery'
                    or row['scale']!=('pixel_squared' if name=='assigned_area' else 'pixel')
                    or row['datatype']!="<class 'int'>" or type(row['reasons']) is not tuple
                    or row['reasons'] not in (('disappeared_by_validity',),tuple(r for r in REASONS if r in row['reasons']))):
                raise NativeRecoveryReplayError('Measurement descriptors/reasons differ from their declared contract.')
            if row['available']:
                _integer(row['value'],1,pixels if name=='assigned_area' else frame.shape[1 if name=='bounding_width' else 0],'Measurement value')
            elif row['value'] is not None:raise NativeRecoveryReplayError('Withheld measurements require explicit unavailable values.')


def validate(value,role,frame,source_binding,parameters,poll=None):
    pixels=math.prod(frame.shape)
    if role=='source_binding':
        if (not isinstance(value,Mapping) or set(value)!={'coordinate_frame','source_sha256'}
                or value!=source_binding):
            raise NativeRecoveryReplayError('Recovery binding differs from the archived source/frame.')
        try:bound=CoordinateFrame.from_dict(dict(value['coordinate_frame']))
        except (ValueError,TypeError,KeyError) as error:raise NativeRecoveryReplayError(str(error)) from error
        if bound!=frame or (frame.origin,frame.axes,frame.units)!=('top-left','x-right,y-down','pixel'):
            raise NativeRecoveryReplayError('Recovery requires the full source pixel frame.')
    elif role=='inventory':
        if (not isinstance(value,Mapping) or set(value)!={'connectivity','core_pixels','objects'}
                or type(value['connectivity']) is not int or value['connectivity']!=4
                or type(value['core_pixels']) is not int or not 0<=value['core_pixels']<=pixels
                or type(value['objects']) is not tuple or len(value['objects'])>native.MAX_CORE_LABELS):
            raise NativeRecoveryReplayError('Core inventory violates its four-connected primitive schema.')
        previous=total=0
        for index,item in enumerate(value['objects']):
            if poll is not None and index%1024==0:poll()
            if (not isinstance(item,Mapping) or set(item)!={'label_id','original_core_pixels'}
                    or type(item['label_id']) is not int or not previous<item['label_id']<=pixels
                    or type(item['original_core_pixels']) is not int or not 1<=item['original_core_pixels']<=pixels):
                raise NativeRecoveryReplayError('Core records require ascending sparse IDs and positive integer areas.')
            previous=item['label_id'];total+=item['original_core_pixels']
        if total!=value['core_pixels']:
            raise NativeRecoveryReplayError('Core inventory areas differ from its pixel total.')
    elif role=='validity_metadata':
        if (not isinstance(value,Mapping) or set(value)!={'binding','bounds_xyxy'}
                or type(value['bounds_xyxy']) is not tuple or value['bounds_xyxy']!=bounds(frame.shape,parameters)
                or any(type(v) is not int for v in value['bounds_xyxy'])):
            raise NativeRecoveryReplayError('Validity metadata differs from its declared half-open ROI.')
        validate(value['binding'],'source_binding',frame,source_binding,parameters,poll)
    elif role=='partition_metadata':_partition_metadata(value,frame,source_binding,poll)
    elif role=='annotation_metadata':_annotation_metadata(value,frame,source_binding,parameters,poll)
    elif role=='observations':_observations(value,frame,poll)
    else:raise NativeRecoveryReplayError('Unknown compact recovery role.')
    return value


def compare(actual,expected,poll=None):
    actual,expected=canonical(actual,poll),canonical(expected,poll)
    missing=object();shape_equal=True;visits=0
    def visit(left,right):
        nonlocal shape_equal,visits
        visits+=1
        if poll is not None and visits%1024==0:poll()
        if isinstance(left,Mapping) or isinstance(right,Mapping):
            if not isinstance(left,Mapping) or not isinstance(right,Mapping):shape_equal=False;return 1,1
            if set(left)!=set(right):shape_equal=False
            children=((left.get(key,missing),right.get(key,missing)) for key in sorted(set(left)|set(right)))
        elif type(left) in (tuple,list) or type(right) in (tuple,list):
            if type(left) is not type(right):shape_equal=False;return 1,1
            if len(left)!=len(right):shape_equal=False
            children=((left[i] if i<len(left) else missing,right[i] if i<len(right) else missing)
                for i in range(max(len(left),len(right))))
        else:
            return int(type(left) is not type(right) or left!=right),1
        mismatch=count=0
        for child_left,child_right in children:
            m,n=visit(child_left,child_right);mismatch+=m;count+=n
        return mismatch,count
    mismatch,count=visit(actual,expected)
    if poll is not None:poll()
    return {'kind':'recovery_compact','mismatched_elements':mismatch,'compared_elements':count,'shape_equal':shape_equal}


def _rounded(size):return ((size+ALLOCATION_BYTES-1)//ALLOCATION_BYTES)*ALLOCATION_BYTES


def plan(operation,shape,parameters,*,selected_bytes,selected_limit=MAX_SELECTED_BYTES,cuda_limit=MAX_CUDA_BYTES,
         selected_allocations=None,input_allocations=None,label_count=0,inventory_count=0):
    if operation not in PORTS:raise NativeRecoveryReplayError('Unknown recovery replay operation.')
    bounds(shape,parameters if operation=='recovery.validity' else {'validity_roi':None})
    if operation=='recovery.annotation':annotation_controls(parameters)
    elif operation!='recovery.validity' and (not isinstance(parameters,Mapping) or parameters):
        raise NativeRecoveryReplayError('Core labeling and clipping have no editable controls.')
    if any(type(cap) is not int or cap<0 for cap in (selected_bytes,selected_limit,cuda_limit)):
        raise NativeRecoveryReplayError('Replay byte counts and limits require nonnegative integers.')
    pixels=math.prod(shape)
    if pixels>native.MAX_RASTER_PIXELS:raise NativeRecoveryReplayUnavailable('Recovery exceeds its raster domain.')
    tile=min(pixels,COMPARE_ELEMENTS)
    compare_bytes=_rounded(tile)+_rounded(8*tile)+ALLOCATION_BYTES
    if selected_allocations is not None:
        if (type(selected_allocations) not in (tuple,list) or len(selected_allocations)>32
                or any(type(size) is not int or size<0 for size in selected_allocations)
                or sum(selected_allocations)!=selected_bytes):
            raise NativeRecoveryReplayError('Selected allocation facts differ from their byte total.')
    selected=sum(_rounded(size) for size in selected_allocations or ())
    if operation in ('recovery.partition','recovery.annotation','recovery.measurements'):
        if selected_allocations is None:
            raise NativeRecoveryReplayError('Recovery requires unique selected allocation facts.')
        downstream_count(inventory_count)
        _integer(label_count,0,native.core.MAX_LABELS,'Selected ID-vector length')
        if (type(input_allocations) not in (tuple,list) or len(input_allocations)>32
                or any(type(size) is not int or size<0 for size in input_allocations)
                or sum(input_allocations)>selected_bytes):
            raise NativeRecoveryReplayError('Recovery requires admitted unique input allocation facts.')
        retained=sum(_rounded(size) for size in input_allocations)
        work=native.core._work(pixels) if operation=='recovery.partition' else (
            pixels*(4*parameters['uncertainty_radius']+3*max(1,label_count.bit_length())+32)+64*label_count
            if operation=='recovery.annotation' else 3*inventory_count)
        if operation=='recovery.partition':
            metadata=4*native.core.ID_TABLE_SLOTS+native.core.MAX_LABELS*80+native.core.BACKEND_HEADROOM_BYTES
            workspace=16*pixels+16
            try:native.core._admit(retained+23*pixels+metadata,native.core.MAX_LABELS*1536+1024*1024,work)
            except native.core.CoreRecoveryResourceError as error:raise NativeRecoveryReplayUnavailable(str(error)) from error
            if workspace>native.core._watershed.MAX_WORKSPACE_BYTES:
                raise NativeRecoveryReplayUnavailable('Conservative partition queue exceeds the source workspace limit.')
            # selected already retains the input arrays and every expected output.
            scratch=_rounded(23*pixels)+_rounded(metadata)+32*ALLOCATION_BYTES;fresh=0
            cpu=native.core.MAX_LABELS*1536+1024*1024
        elif operation=='recovery.annotation':
            source_peak=retained+2*pixels+44*label_count+36+native.core.BACKEND_HEADROOM_BYTES
            cpu=label_count*1536+1024*1024
            try:native.core._admit(source_peak,cpu,work)
            except native.core.CoreRecoveryResourceError as error:raise NativeRecoveryReplayUnavailable(str(error)) from error
            scratch=_rounded(44*label_count)+_rounded(36)+native.core.BACKEND_HEADROOM_BYTES
            fresh=2*_rounded(pixels)
        else:
            if selected_bytes or selected or retained:raise NativeRecoveryReplayError('Measurement replay must select no raster arrays.')
            scratch=fresh=compare_bytes=0;cpu=downstream_count(inventory_count)
        live=selected+fresh+scratch+compare_bytes+AUDIT_CUDA_BYTES
        if selected_bytes>selected_limit or live>cuda_limit:
            raise NativeRecoveryReplayUnavailable('Recovery exceeds selected-array or CUDA replay admission.')
        return {'selected_bytes':selected_bytes,'selected_limit_bytes':selected_limit,'cuda_bytes':live,
            'cuda_limit_bytes':cuda_limit,'primitive_scratch_bytes':scratch,'comparison_scratch_bytes':compare_bytes,
            'runtime_audit_cuda_reserve_bytes':AUDIT_CUDA_BYTES,'work':work,'primitive_cpu_metadata_bytes':cpu,
            'compact_metadata_limit_bytes':MAX_METADATA_BYTES,'queue_admission':'worst-case full domain before upload'
                if operation=='recovery.partition' else 'no partition queue allocated',
            'bounds':'encoded metadata and modeled tensor storage; not process RSS or elapsed time'}
    if operation=='recovery.core_labels':
        try:native.retained_bound(pixels)
        except native.core.CoreRecoveryResourceError as error:raise NativeRecoveryReplayUnavailable(str(error)) from error
        # Worst-case foreground queue is safe before any selected raster upload.
        # The producer retains its actual-foreground admission at dispatch.
        scratch=_rounded(4*pixels)+2*_rounded(4*native.MAX_CORE_LABELS)+_rounded(32)+native.core.BACKEND_HEADROOM_BYTES
        fresh=_rounded(4*pixels);selected=_rounded(pixels)+_rounded(4*pixels)
        work=16*pixels
        cpu=native.MAX_CORE_LABELS*256+1024*1024
    elif operation=='recovery.clip_cores':
        scratch=0;fresh=_rounded(4*pixels)
        selected=2*_rounded(4*pixels)+_rounded(pixels);work=pixels;cpu=0
    else:
        scratch=0;fresh=_rounded(pixels);selected=2*_rounded(pixels);work=pixels;cpu=4096
    live=selected+fresh+scratch+compare_bytes+AUDIT_CUDA_BYTES
    if selected_bytes>selected_limit or live>cuda_limit:
        raise NativeRecoveryReplayUnavailable('Recovery exceeds selected-array or CUDA replay admission.')
    return {'selected_bytes':selected_bytes,'selected_limit_bytes':selected_limit,'cuda_bytes':live,
        'cuda_limit_bytes':cuda_limit,'primitive_scratch_bytes':scratch,'comparison_scratch_bytes':compare_bytes,
        'runtime_audit_cuda_reserve_bytes':AUDIT_CUDA_BYTES,'work':work,'primitive_cpu_metadata_bytes':cpu,
        'compact_metadata_limit_bytes':MAX_METADATA_BYTES,
        'queue_admission':('worst-case foreground before upload; actual count checked by producer'
            if operation=='recovery.core_labels' else 'no queue required'),
        'bounds':'encoded metadata and modeled tensor storage; not process RSS or elapsed time'}


PINNED_STATE=state()
