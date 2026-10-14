"""Distinct experimental native graph for recovery from explicit eroded cores.

Public node products are mappings of CUDA tensors and compact JSON facts. Session
snapshot records remain transient, so ordinary native cache/export traversal can
see every raster without persisting pointers or mutation counters.
"""
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from collections.abc import Mapping
from types import MappingProxyType

import torch

from .checkpoints import CoordinateFrame, _freeze_json
from .recipe_catalogue import build_recipe_catalogue
from .recipes import (DependencyContract, LayoutPosition, ParameterContract, PortContract,
                      RecipeProvenance, RecipeProvenanceMode, RecipeSpec, RecipeStage,
                      SourceMapping, SourceResolution)
from seedvision.cuda import core_recovery as core
from seedvision.cuda._nvrtc import fixed_kernel_program

VERSION = '1.0.0'
RECIPE_ID = 'seedfiddle_experimental_core_recovery'
SOURCE = 'recovery_source'
FOREGROUND = 'recovery_foreground'
ERODE = 'recovery_erode'
CORES = 'recovery_core_labels'
DISTANCE = 'recovery_distance'
ELEVATION = 'recovery_elevation'
VALIDITY = 'recovery_validity'
CLIP = 'recovery_clip_cores'
PARTITION = 'recovery_partition'
ANNOTATION = 'recovery_annotation'
MEASUREMENTS = 'recovery_measurements'
MAX_RASTER_PIXELS = 64_000_000
MAX_CORE_LABELS = 32_768
GRAPH_METADATA_BYTES_PER_LABEL = 8192
GRAPH_METADATA_FIXED_BYTES = 1024*1024
CORE_BATCH = 4096
VALIDATION_TILE = 65_536
_CUDA_SOURCE = Path(__file__).parents[1].joinpath('cuda/_core_recovery_graph.cu').read_text(encoding='utf-8')
CUDA_SOURCE_SHA256 = sha256(_CUDA_SOURCE.encode('utf-8')).hexdigest()
_TENSOR_FIELDS = ('original_foreground', 'markers', 'validity', 'elevation', 'labels',
                  'domain', 'unresolved', 'interface', 'label_ids')


def retained_bound(pixels):
    # All graph-owned retained products, replacement scratch, core BFS and flood
    # queue fit concurrently. Caller-held old results are outside cache ownership.
    bound = 64*pixels+32*1024*1024+MAX_CORE_LABELS*256
    if not 1 <= pixels <= MAX_RASTER_PIXELS:
        raise core.CoreRecoveryResourceError('Recovery graph exceeds its raster limit.')
    core._admit(bound, core.MAX_CPU_METADATA_BYTES,
                core._work(pixels)+64*pixels+MAX_CORE_LABELS*64)
    return bound


def graph_metadata_bound(count):
    """Bound all graph-owned compact records/cache, excluding explicit export."""
    if type(count) is not int or not 0<=count<=MAX_CORE_LABELS:
        raise core.CoreRecoveryResourceError('Recovery graph core inventory exceeds its label limit.')
    bound=GRAPH_METADATA_FIXED_BYTES+GRAPH_METADATA_BYTES_PER_LABEL*count
    if bound>core.MAX_CPU_METADATA_BYTES:
        raise core.CoreRecoveryResourceError('Recovery graph exceeds its compact CPU metadata label limit.')
    return bound


def _native_mapping(symbol):
    return SourceMapping('seedfiddle_enhancement', VERSION, SourceResolution.PINNED,
        source_path='seedvision/cuda/_core_recovery_graph.cu', source_symbol=symbol,
        source_sha256=CUDA_SOURCE_SHA256,
        note='Native experimental graph producer; not an original SMPTS operation.')


def _port(name, direction, kind, semantics=None):
    return PortContract(name, direction, kind, name.replace('_', ' '),
                        semantics=_semantics(kind) if semantics is None else semantics)


def _semantics(kind):
    if kind == 'BinaryMask':
        return {'dtype':'uint8','foreground':255,'background':0,'frame':'source.decoded'}
    if kind == 'MarkerLabels':
        return {'dtype':'int32','background':0,'positive_values':'first core pixel index plus one','frame':'source.decoded'}
    if kind in ('DistanceMap','NegativeDistanceMap'):
        return {'dtype':'float32','frame':'source.decoded','method':'pinned OpenCV five-by-five chamfer',
                'sign':'negative' if kind == 'NegativeDistanceMap' else 'positive'}
    return {}


def build_native_core_recovery_recipe():
    """Add an explicit core/elevation/validity graph without altering history."""
    from dataclasses import replace
    old = build_recipe_catalogue().recipe('plantcv_seed_binary')
    source = replace(old.stages[0], stage_id=SOURCE, ports=old.stages[0].ports+(
        _port('source_frame', 'output', 'Metadata'),))
    threshold = replace(next(s for s in old.stages if s.operation_id == 'threshold.binary'),
                        stage_id=FOREGROUND, title='Original foreground')
    threshold = replace(threshold,ports=tuple(replace(p,label='Original grayscale')
        if p.direction == 'input' and p.name == 'gray' else p for p in threshold.ports))
    erosion = replace(next(s for s in old.stages if s.operation_id == 'morphology.erode'),
                      stage_id=ERODE, title='Explicit eroded cores')
    distance = replace(next(s for s in old.stages if s.operation_id.endswith('.distance_map')),
                       stage_id=DISTANCE)
    elevation = replace(next(s for s in old.stages if s.operation_id.endswith('.negate_distance')),
                        stage_id=ELEVATION, title='Negative 5x5 distance elevation')
    threshold,erosion,distance,elevation=tuple(replace(s,ports=tuple(
        replace(p,semantics=_semantics(p.data_type)) if p.data_type in
        ('BinaryMask','DistanceMap','NegativeDistanceMap') else p for p in s.ports))
        for s in (threshold,erosion,distance,elevation))
    def stage(stage_id, title, operation, ports, description, parameters=()):
        return RecipeStage(stage_id, title, operation, description=description,
            ports=ports, parameters=parameters, source_mappings=(_native_mapping(operation),))
    cores = stage(CORES, 'Four-connected core IDs', 'recovery.core_labels', (
        _port('mask','input','BinaryMask'), _port('markers','output','MarkerLabels'),
        _port('inventory','output','Metadata')),
        'Label eroded components with stable first-pixel IDs and retain their pre-validity inventory. Fixed four-connectivity.')
    validity = stage(VALIDITY, 'Valid source support', 'recovery.validity', (
        _port('mask','input','BinaryMask'), _port('source_frame','input','Metadata'),
        _port('validity','output','BinaryMask'), _port('metadata','output','Metadata')),
        'Keep the full source frame. An optional dimension-bound pixel rectangle marks valid support; it never rescales a batch image.', (
        ParameterContract('validity_roi','json',None,True,editable=True,
            description='Null uses the full frame; otherwise {"shape":[height,width],"bounds":[x0,y0,x1,y1]} with integer half-open pixels.'),))
    clip = stage(CLIP, 'Clip cores to valid support', 'recovery.clip_cores', (
        _port('markers','input','MarkerLabels'), _port('validity','input','BinaryMask'),
        _port('markers','output','MarkerLabels')),
        'Clip core pixels after labeling, preserving original IDs and their separate expected inventory.')
    partition = stage(PARTITION, 'Recover original foreground', 'recovery.partition', (
        _port('foreground','input','BinaryMask'), _port('markers','input','MarkerLabels'),
        _port('validity','input','BinaryMask'), _port('elevation','input','NegativeDistanceMap'),
        _port('inventory','input','Metadata'), _port('source_frame','input','Metadata'),
        _port('partition','output','RecoveryPartition'), _port('labels','output','InstanceLabels'),
        _port('metadata','output','Metadata')),
        'Partition valid original foreground from supplied core IDs and explicit elevation. Preserve the original union, unresolved coverage and inferred boundary flags.')
    annotation = stage(ANNOTATION, 'Boundary uncertainty', 'recovery.annotation', (
        _port('partition','input','RecoveryPartition'), _port('annotation','output','RecoveryAnnotation'),
        _port('uncertainty_band','output','BinaryMask',{'positive_value':1}),
        _port('metadata','output','Metadata')),
        'Reuse labels. Radius0 includes the symmetric interface; further four-neighbor steps stay inside valid foreground. Policy controls computational measurements.', (
        ParameterContract('uncertainty_radius','integer',0,True,editable=True,minimum=0,maximum=8,units='pixels'),
        ParameterContract('measurement_policy','string','withhold_uncertain',True,editable=True,
            choices=('withhold_uncertain','flag_uncertain'))))
    measurements = stage(MEASUREMENTS, 'Experimental pixel measurements', 'recovery.measurements', (
        _port('annotation','input','RecoveryAnnotation'), _port('observations','output','Observations'),
        _port('metadata','output','Metadata')),
        'Compute permitted assigned area and bounding width/height in pixels. Withheld or disappeared objects have explicit unavailable values; these are not PlantCV size traits or biological claims.')
    stages = (source,threshold,erosion,cores,distance,elevation,validity,clip,partition,annotation,measurements)
    edges = (
        (SOURCE,'gray_image',FOREGROUND,'gray','GrayImage'),
        (FOREGROUND,'mask',ERODE,'gray','BinaryMask'),
        (ERODE,'mask',CORES,'mask','BinaryMask'),
        (FOREGROUND,'mask',DISTANCE,'mask','BinaryMask'),
        (DISTANCE,'distance',ELEVATION,'distance','DistanceMap'),
        (FOREGROUND,'mask',VALIDITY,'mask','BinaryMask'),
        (SOURCE,'source_frame',VALIDITY,'source_frame','Metadata'),
        (CORES,'markers',CLIP,'markers','MarkerLabels'),
        (VALIDITY,'validity',CLIP,'validity','BinaryMask'),
        (FOREGROUND,'mask',PARTITION,'foreground','BinaryMask'),
        (CLIP,'markers',PARTITION,'markers','MarkerLabels'),
        (VALIDITY,'validity',PARTITION,'validity','BinaryMask'),
        (ELEVATION,'negative_distance',PARTITION,'elevation','NegativeDistanceMap'),
        (CORES,'inventory',PARTITION,'inventory','Metadata'),
        (SOURCE,'source_frame',PARTITION,'source_frame','Metadata'),
        (PARTITION,'partition',ANNOTATION,'partition','RecoveryPartition'),
        (ANNOTATION,'annotation',MEASUREMENTS,'annotation','RecoveryAnnotation'),
    )
    provenance = RecipeProvenance(RecipeProvenanceMode.NATIVE_COMPATIBLE,
        'seedvision.benchmarking.native_core_recovery_recipe',VERSION,
        reference_id='seedfiddle_enhancement',reference_version=VERSION,
        evidence_status=SourceResolution.PINNED,
        evidence_note='Distinct native experimental core recovery with explicit erosion, fixed four-connectivity and named negative distance elevation. No SMPTS reconstruction, measured gain or scientific validation.')
    return RecipeSpec(RECIPE_ID,VERSION,'Experimental core recovery',
        'Recover original foreground from explicit eroded cores; retain unresolved objects and boundary uncertainty.',
        provenance,stages,tuple(DependencyContract(*e) for e in edges),
        tuple(LayoutPosition(s.stage_id,(i%4)*290,(i//4)*270) for i,s in enumerate(stages)))


def _core_labels(mask, cancel):
    if (not torch.is_tensor(mask) or mask.device.type != 'cuda' or mask.dtype != torch.uint8
            or mask.ndim != 2 or not mask.is_contiguous()):
        raise ValueError('Core labeling requires a contiguous CUDA uint8 HW mask.')
    retained_bound(mask.numel())
    height,width = mask.shape
    with torch.cuda.device(mask.device):
        guard = core._Guard(cancel,(mask,))
        guard.checkpoint()
        state = torch.zeros(8,dtype=torch.int32,device=mask.device)
        guard.watch(state)
        program = fixed_kernel_program(_CUDA_SOURCE,mask.device)
        for start in range(0,mask.numel(),VALIDATION_TILE):
            guard.checkpoint()
            count=min(VALIDATION_TILE,mask.numel()-start)
            program.launch_1d('recovery_graph_validate',(mask,state),(start,count),count)
            if int(state[5].item()):
                raise ValueError('Core mask must contain only0/255.')
        foreground=int(state[7].item())
        capacity=max(1,foreground)
        # Include the actual input backing, output, queue, ID/area tables, state
        # and a bounded backend allowance before the full-sized allocations.
        core._admit(core._backing_bytes((mask,))+4*mask.numel()+4*capacity+
                    8*MAX_CORE_LABELS+32+core.BACKEND_HEADROOM_BYTES,
                    MAX_CORE_LABELS*256+1024*1024,8*(mask.numel()+foreground))
        labels=torch.zeros(mask.shape,dtype=torch.int32,device=mask.device)
        queue=torch.empty(capacity,dtype=torch.int32,device=mask.device)
        ids=torch.empty(MAX_CORE_LABELS,dtype=torch.int32,device=mask.device)
        areas=torch.empty_like(ids)
        guard.watch(labels,queue,ids,areas)
        for _ in range(0,mask.numel()+foreground,CORE_BATCH):
            guard.checkpoint()
            program.launch('recovery_graph_cores',(mask,labels,queue,ids,areas,state),
                           (height,width,capacity,MAX_CORE_LABELS,CORE_BATCH))
            progress=state.cpu().tolist()
            if progress[5]:
                raise core.CoreRecoveryResourceError('Core component count or queue exceeded its bound.')
            if progress[0] == mask.numel() and progress[1] == progress[2]:
                break
        else:
            progress=state.cpu().tolist()
        if (progress[0] != mask.numel() or progress[1] != progress[2]
                or progress[6] != foreground):
            raise core.CoreRecoveryResourceError('Core labeling exceeded its checked linear work bound.')
        guard.checkpoint()
        count=progress[4]
        inventory=_freeze_json({'connectivity':4,'core_pixels':foreground,'objects':tuple(
            {'label_id':i,'original_core_pixels':a}
            for i,a in zip(ids[:count].cpu().tolist(),areas[:count].cpu().tolist()))},name='core inventory')
        guard.checkpoint()
        return {'markers':labels,'inventory':inventory}


def _source_frame(value, shape):
    if not isinstance(value,Mapping) or set(value) != {'coordinate_frame','source_sha256'}:
        raise ValueError('Recovery requires its exact prepared source/frame binding.')
    frame=CoordinateFrame.from_dict(dict(value['coordinate_frame']))
    if frame.shape != tuple(shape) or (frame.origin,frame.axes,frame.units) != ('top-left','x-right,y-down','pixel'):
        raise ValueError('Recovery products require the declared full pixel frame.')
    return _freeze_json(dict(value),name='recovery source binding')


def _validity(inputs, parameters, cancel):
    mask=inputs['mask']
    binding=_source_frame(inputs['source_frame'],mask.shape)
    height,width=mask.shape
    selected=parameters['validity_roi']
    bounds=(0,0,width,height)
    if selected is not None:
        if (not isinstance(selected,Mapping) or set(selected) != {'shape','bounds'}
                or not isinstance(selected['shape'],(tuple,list)) or tuple(selected['shape']) != (height,width)
                or any(type(v) is not int for v in selected['shape'])
                or not isinstance(selected['bounds'],(tuple,list)) or len(selected['bounds']) != 4
                or any(type(v) is not int for v in selected['bounds'])):
            raise ValueError('Validity ROI requires exact image shape and four integer pixel bounds.')
        bounds=tuple(selected['bounds'])
        x0,y0,x1,y1=bounds
        if not 0<=x0<x1<=width or not 0<=y0<y1<=height:
            raise ValueError('Validity ROI must be nonempty and inside its declared frame.')
    guard=core._Guard(cancel,(mask,))
    guard.checkpoint()
    validity=torch.empty_like(mask)
    guard.watch(validity)
    with torch.cuda.device(mask.device):
        fixed_kernel_program(_CUDA_SOURCE,mask.device).launch_1d('recovery_graph_validity',
            (validity,),(height,width,*bounds),mask.numel())
    guard.checkpoint()
    return {'validity':validity,'metadata':_freeze_json({'binding':binding,'bounds_xyxy':bounds},name='valid support')}


def _pack(partition, inventory, binding):
    return MappingProxyType({'tensors':MappingProxyType({name:getattr(partition,name) for name in _TENSOR_FIELDS}),
        'metadata':_freeze_json({'schema':'seedfiddle.core-recovery-partition.v1',
            'objects':tuple(asdict(obj) for obj in partition.objects),'plan':asdict(partition.plan),
            'status':partition.status,'profile':partition.profile,
            'unresolved_pixels':partition.unresolved_pixels,'interface_pixels':partition.interface_pixels,
            'inventory':inventory,'binding':binding},name='recovery partition')})


def _unpack(value):
    if not isinstance(value,Mapping) or set(value) != {'tensors','metadata'}:
        raise ValueError('Recovery partition transport has missing or unknown fields.')
    tensors,metadata=value['tensors'],value['metadata']
    if (not isinstance(tensors,Mapping) or set(tensors) != set(_TENSOR_FIELDS)
            or not isinstance(metadata,Mapping) or set(metadata) != {'schema','objects','plan','status',
                'profile','unresolved_pixels','interface_pixels','inventory','binding'}
            or metadata['schema'] != 'seedfiddle.core-recovery-partition.v1' or metadata['profile'] != core.PROFILE):
        raise ValueError('Recovery partition transport differs from its declared schema.')
    core._inputs(*(tensors[name] for name in _TENSOR_FIELDS[:4]))
    shape,device=tensors['labels'].shape,tensors['original_foreground'].device
    for name in ('labels','domain','unresolved','interface'):
        tensor=tensors[name]
        if (not torch.is_tensor(tensor) or tensor.device != device or tuple(tensor.shape) != tuple(shape)
                or not tensor.is_contiguous() or tensor.dtype != (torch.int32 if name == 'labels' else torch.uint8)):
            raise ValueError('Recovery partition output tensor contract differs.')
    if tuple(shape) != tuple(tensors['original_foreground'].shape):
        raise ValueError('Recovery partition frame differs from its source.')
    ids=tensors['label_ids']
    if ids.device != device or ids.dtype != torch.int32 or ids.ndim != 1 or not ids.is_contiguous() or ids.numel()>core.MAX_LABELS:
        raise ValueError('Recovery partition ID table differs.')
    _source_frame(metadata['binding'],shape)
    objects=tuple(core.CoreRecoveredObject(**dict(row)) for row in metadata['objects'])
    partition=core.CoreRecoveryPartition(*(tensors[name] for name in _TENSOR_FIELDS),objects,
        core.CoreRecoveryPlan(**dict(metadata['plan'])),metadata['unresolved_pixels'],metadata['interface_pixels'],
        metadata['status'],metadata['profile'],())
    object.__setattr__(partition,'_snapshot',core._partition_snapshot(partition))
    return partition


def _annotate(value, parameters, cancel):
    graph_metadata_bound(len(value['metadata']['inventory']['objects']))
    partition=_unpack(value)
    result=core.annotate_core_recovery(partition,**parameters,cancel=cancel)
    facts={obj.label_id:obj for obj in partition.objects}
    statuses={obj.label_id:obj for obj in result.objects}
    records=[]
    for expected in value['metadata']['inventory']['objects']:
        label_id=expected['label_id']
        obj,status=facts.get(label_id),statuses.get(label_id)
        record=dict(expected)
        if obj is None:
            record.update(seed_pixels=0,assigned_pixels=0,interface_pixels=0,bbox=None,
                uncertainty_pixels=0,reasons=('disappeared_by_validity',),measurement_permitted=False,
                uncontested_visible_partition=False,scientific_validation=False)
        else:
            record.update(asdict(obj))
            record.update(uncertainty_pixels=status.uncertainty_pixels,reasons=status.reasons,
                measurement_permitted=status.segmentation_measurement_allowed,
                uncontested_visible_partition=status.uncontested_visible_partition,scientific_validation=False)
        record['clipped_core_pixels']=record['original_core_pixels']-record['seed_pixels']
        records.append(record)
    metadata=_freeze_json({'schema':'seedfiddle.core-recovery-annotation.v1',
        'binding':value['metadata']['binding'],'partition_status':partition.status,
        'original_foreground_pixels':partition.plan.original_foreground_pixels,
        'validity_clipped_foreground_pixels':partition.plan.validity_clipped_foreground_pixels,
        'unresolved_pixels':partition.unresolved_pixels,'uncertainty_radius':result.uncertainty_radius,
        'measurement_policy':result.measurement_policy,'objects':records},name='recovery annotation')
    annotation=MappingProxyType({'uncertainty_band':result.uncertainty_band,'metadata':metadata})
    return {'annotation':annotation,'uncertainty_band':result.uncertainty_band,'metadata':metadata}


def _measure(annotation):
    graph_metadata_bound(len(annotation['metadata']['objects']))
    observations={}
    for obj in annotation['metadata']['objects']:
        allowed=obj['measurement_permitted']
        # Ineligible records do not execute the size calculation. Diagnostic
        # counts remain in metadata, separately from these permitted traits.
        values={'assigned_area':None,'bounding_width':None,'bounding_height':None}
        if allowed:
            x0,y0,x1,y1=obj['bbox']
            values={'assigned_area':obj['assigned_pixels'],'bounding_width':x1-x0,'bounding_height':y1-y0}
        observations['core_'+str(obj['label_id'])]={name:{'value':value,'available':allowed,
            'trait':name.replace('_',' '),'method':'seedfiddle.experimental.core_recovery',
            'scale':'pixel_squared' if name=='assigned_area' else 'pixel','datatype':"<class 'int'>",
            'reasons':obj['reasons'],'scientific_validation':False} for name,value in values.items()}
    return {'observations':_freeze_json(observations,name='experimental pixel measurements'),
            'metadata':annotation['metadata']}


def execute_stage(operation, inputs, parameters, cancel):
    if operation == 'recovery.core_labels':
        return _core_labels(inputs['mask'],cancel)
    if operation == 'recovery.validity':
        return _validity(inputs,parameters,cancel)
    if operation == 'recovery.clip_cores':
        markers,validity=inputs['markers'],inputs['validity']
        guard=core._Guard(cancel,(markers,validity))
        guard.checkpoint()
        output=torch.empty_like(markers)
        guard.watch(output)
        with torch.cuda.device(markers.device):
            fixed_kernel_program(_CUDA_SOURCE,markers.device).launch_1d('recovery_graph_clip',
                (markers,validity,output),(markers.numel(),),markers.numel())
        guard.checkpoint()
        return {'markers':output}
    if operation == 'recovery.partition':
        graph_metadata_bound(len(inputs['inventory']['objects']))
        binding=_source_frame(inputs['source_frame'],inputs['foreground'].shape)
        result=core.recover_core_labels(inputs['foreground'],inputs['markers'],inputs['validity'],
                                       inputs['elevation'],cancel=cancel)
        partition=_pack(result,inputs['inventory'],binding)
        return {'partition':partition,'labels':result.labels,'metadata':partition['metadata']}
    if operation == 'recovery.annotation':
        return _annotate(inputs['partition'],parameters,cancel)
    if operation == 'recovery.measurements':
        if cancel is not None and cancel():
            raise core.CoreRecoveryCancelled('Recovery measurements cancelled.')
        return _measure(inputs['annotation'])
    raise NotImplementedError('Unknown experimental recovery operation.')
