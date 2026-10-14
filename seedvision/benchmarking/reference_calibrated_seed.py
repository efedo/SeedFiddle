"""Compose actual color correction, a declared crop, and public seed analysis.

This is a SeedFiddle composition of pinned public calls, not a newly discovered
PlantCV recipe. The original authored child and its pixel measurements remain
unchanged. Native CUDA execution and node caches do not use this CPU path.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
import math
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

import cv2
import numpy as np

from .checkpoint_inspection import _verify_runner_identity
from .checkpoints import (
    CheckpointCancelled, CheckpointIdentity, CheckpointLimits, CoordinateFrame,
    ExecutionRecord, ExecutionStatus, MeasurementContract, RasterRole,
    _canonical_json, _cancelled, _freeze_json, _thaw_json, load_stage_input, publish_checkpoint,
)
from .recipes import RecipeSpec
from .reference_mask_stage import _selected_parameters
from .reference_preprocessing import _chain_stages, _validate_pinned_recipe
from .reference_runner import PlantCVReferenceRunner, ReferenceCancelledError, ReferenceResult, _validate_watershed_capture
from .reference_seed_chain import (
    ReferenceRasterInput, ReferenceSeedChainOutcome, ReferenceSourceCall,
    _checked_result, _digest, _identity as _seed_identity, _input, _manifest, _parameters,
    run_reference_seed_chain,
)
from .reference_seed_chain_reader import _result, _same, read_reference_seed_chain
from .reference_stage import _calculation_parameters
from .reference_watershed_capture import capture_source_records


SPEC_SCHEMA = 'seedfiddle.public-calibrated-seed-spec.v1'
CHAIN_SCHEMA = 'seedfiddle.public-calibrated-seed-chain.v1'
HELPER_SHA256 = 'c05b0c944210657332ce307f2066caab55f2ebc8d8fea11988db3e476e7cdfba'
ADAPTER_CONTRACT = {
    'origin':'seedfiddle_composition_of_pinned_public_calls',
    'auto_correct_color_sha256':'bed2eb9e50e42e39f7ad4bbef157150190585268178f06083092edb19ffaf57a',
    'public_roi_helper_sha256':HELPER_SHA256,
    'crop':'half-open source-frame slice after full-image correction; no interpolation',
    'gray':'local cv2.COLOR_BGR2GRAY after RGB to BGR reversal',
    'size':'original pixel child plus separate actual public size under correction-returned params',
    'version':'1',
}
_CARD_DEFAULTS = {'color_chip_size':None,'adaptive_method':1,'block_size':51,'radius':20,
                  'min_size':1000,'aspect_ratio':1.27,'solidity':0.8,'roi_bounds_xyxy':None}


class ReferenceCalibratedSeedError(ValueError):
    """The explicit composition or source lineage is inconsistent."""


def _number(value, name):
    if isinstance(value,bool) or not isinstance(value,(int,float)):
        raise ReferenceCalibratedSeedError(f'{name} must be a finite number.')
    try:
        number = float(value)
    except (ValueError,OverflowError):
        raise ReferenceCalibratedSeedError(f'{name} must be a finite number.') from None
    if not math.isfinite(number):
        raise ReferenceCalibratedSeedError(f'{name} must be a finite number.')
    return number


def _bounds(value, shape=None):
    if not isinstance(value,(list,tuple)) or len(value)!=4 or any(type(v) is not int for v in value):
        raise ReferenceCalibratedSeedError('ROI requires four integer half-open bounds.')
    x0,y0,x1,y1 = value
    if not (0<=x0<x1 and 0<=y0<y1) or (shape is not None and (x1>shape[1] or y1>shape[0])):
        raise ReferenceCalibratedSeedError('ROI must be nonempty and inside its parent frame.')
    return tuple(value)


def _card_parameters(value):
    if not isinstance(value,Mapping) or set(value)-set(_CARD_DEFAULTS):
        raise ReferenceCalibratedSeedError('Unsupported public color-card parameters.')
    p = dict(_CARD_DEFAULTS)|dict(value)
    for name,low,high in (('adaptive_method',0,1),('block_size',3,255),('radius',1,512),('min_size',1,None)):
        v = p[name]
        if type(v) is not int or v<low or (high is not None and v>high):
            raise ReferenceCalibratedSeedError(f'{name} is outside the public adapter domain.')
    if p['block_size']%2!=1:
        raise ReferenceCalibratedSeedError('block_size must be odd.')
    for name in ('aspect_ratio','solidity'):
        p[name] = _number(p[name],name)
    if p['aspect_ratio']<=0 or not 0<=p['solidity']<=1:
        raise ReferenceCalibratedSeedError('Invalid card geometry parameters.')
    size = p['color_chip_size']
    if isinstance(size,str):
        if size not in {'classic','passport','cameratrax'}:
            raise ReferenceCalibratedSeedError('Unsupported public card name.')
    elif size is not None:
        if not isinstance(size,(list,tuple)) or len(size)!=2:
            raise ReferenceCalibratedSeedError('Chip dimensions require width and height.')
        size = [_number(v,'color_chip_size') for v in size]
        if min(size)<=0:
            raise ReferenceCalibratedSeedError('Chip dimensions must be positive.')
        p['color_chip_size'] = size
    if p['roi_bounds_xyxy'] is not None:
        p['roi_bounds_xyxy'] = list(_bounds(p['roi_bounds_xyxy']))
    return p


@dataclass(frozen=True,slots=True)
class ReferenceCalibratedSeedSpec:
    seed_recipe: RecipeSpec
    card_parameters: Mapping[str,Any]
    seed_roi_xyxy: tuple[int,int,int,int]

    def __post_init__(self):
        if type(self.seed_recipe) is not RecipeSpec:
            raise TypeError('An exact original authored seed recipe is required.')
        _validate_pinned_recipe(self.seed_recipe)
        _parameters(self.seed_recipe)
        for stage in _chain_stages(self.seed_recipe):
            if stage.operation_id.startswith('threshold.'):
                _calculation_parameters(self.seed_recipe,stage.stage_id,stage.operation_id)
            else:
                _selected_parameters(self.seed_recipe,stage.stage_id,stage.operation_id)
        object.__setattr__(self,'card_parameters',_freeze_json(_card_parameters(self.card_parameters),name='card parameters'))
        object.__setattr__(self,'seed_roi_xyxy',_bounds(self.seed_roi_xyxy))

    def to_dict(self):
        return {'schema':SPEC_SCHEMA,'adapters':dict(ADAPTER_CONTRACT),'seed_recipe':self.seed_recipe.to_dict(),
                'card_parameters':_thaw_json(self.card_parameters),'seed_roi_xyxy':list(self.seed_roi_xyxy)}

    @property
    def sha256(self):
        return sha256(_canonical_json(self.to_dict(),name='calibrated public composition')).hexdigest()

    @classmethod
    def from_dict(cls,value):
        if not isinstance(value,dict) or set(value)!={'schema','adapters','seed_recipe','card_parameters','seed_roi_xyxy'}:
            raise ReferenceCalibratedSeedError('Composition spec schema differs.')
        if value['schema']!=SPEC_SCHEMA or value['adapters']!=ADAPTER_CONTRACT:
            raise ReferenceCalibratedSeedError('Composition adapter contract differs.')
        recipe = RecipeSpec.from_json(_canonical_json(value['seed_recipe'],name='selected child recipe').decode('utf-8'))
        return cls(recipe,value['card_parameters'],value['seed_roi_xyxy'])


@dataclass(frozen=True,slots=True)
class ReferenceCalibratedSeedOutcome:
    status: str
    correction: ReferenceSourceCall
    child: ReferenceSeedChainOutcome|None
    scaled_size: ReferenceSourceCall
    record: Mapping[str,Any]
    manifest_path: Path|None = None
    manifest_sha256: str|None = None

    def __post_init__(self):
        if self.status not in {'executed','empty','partial','cancelled'}:
            raise ValueError('Invalid calibrated composition outcome.')
        if (self.manifest_path is None)!=(self.manifest_sha256 is None) or (self.manifest_path is not None and self.status!='executed'):
            raise ValueError('Only complete compositions publish a parent manifest.')
        object.__setattr__(self,'record',_freeze_json(self.record,name='calibrated source composition'))


def _identity(parent,spec,stage,upstream,frame,code_sha):
    return replace(parent.identity,coordinate_frame=frame,stage_id=stage,
        recipe_id='seedfiddle_public_calibrated_seed',recipe_version='1',
        recipe_sha256=spec.sha256,graph_sha256=spec.sha256,settings_sha256=spec.sha256,
        backend_id='seedfiddle.public_reference_composition',backend_version='1',
        environment_versions={**dict(spec.seed_recipe.provenance.environment_versions),
            'local-opencv':str(cv2.__version__),'local-numpy':str(np.__version__),
            'composition-code-sha256':code_sha},upstream_checkpoint_sha256=upstream)


def _scale(checkpoint,spec):
    metadata = checkpoint.metadata
    _verify_runner_identity(metadata,spec.seed_recipe)
    effective = metadata['effective_parameters']
    effects = effective.get('calibration_effects')
    if not isinstance(effects,Mapping):
        raise ReferenceCalibratedSeedError('Correction omitted its actual returned scale.')
    scale = {name:_number(effects.get(name),name) for name in ('px_width','px_height')}
    scale['unit'] = effects.get('unit')
    expected_unit = 'pixels' if spec.card_parameters['color_chip_size'] is None else 'mm'
    if scale['unit']!=expected_unit or min(scale['px_width'],scale['px_height'])<=0:
        raise ReferenceCalibratedSeedError('Correction returned unusable scale parameters.')
    if expected_unit=='pixels' and (scale['px_width']!=1.0 or scale['px_height']!=1.0):
        raise ReferenceCalibratedSeedError('Unselected chip dimensions changed pixel scale.')
    _same(effects.get('chip_measurements'),metadata['observations'].get('metadata',{}),'actual chip observations')
    return scale


def _crop(parent,spec,reference,store,measurement,limits,cancel,code_sha):
    checkpoint = load_stage_input(store,reference.checkpoint_sha256,expected_identity=reference.identity,
                                 expected_measurement=measurement,limits=limits)
    scale = _scale(checkpoint,spec)
    corrected = checkpoint.arrays['corrected_image']
    x0,y0,x1,y1 = spec.seed_roi_xyxy
    crop_rgb = np.ascontiguousarray(corrected[y0:y1,x0:x1])
    gray = cv2.cvtColor(np.ascontiguousarray(crop_rgb[...,::-1]),cv2.COLOR_BGR2GRAY)
    transform = {'parent_frame':parent.identity.coordinate_frame.to_dict(),'bounds_xyxy':list(spec.seed_roi_xyxy),
        'corrected_rgb_sha256':_digest(corrected),'composition_sha256':spec.sha256,
        'gray_conversion':ADAPTER_CONTRACT['gray'],'opencv':str(cv2.__version__),'adapter_version':'1'}
    frame = CoordinateFrame('corrected_public_seed_roi',(y1-y0,x1-x0),'top_left_pixel_centres',
                            'x_right_y_down','pixel',sha256(_canonical_json(transform,name='corrected crop frame')).hexdigest())
    record = {'status':'prepared','bounds_xyxy':list(spec.seed_roi_xyxy),'local_to_parent_xy':[x0,y0],
        'coordinate_frame':frame.to_dict(),'parent_frame':parent.identity.coordinate_frame.to_dict(),
        'correction_checkpoint_sha256':reference.checkpoint_sha256,'transform':transform,'inputs':{}}
    bindings = {}
    for name,array in (('rgb',crop_rgb),('gray',gray)):
        if _cancelled(cancel):
            raise CheckpointCancelled('Cancelled before corrected crop publication.')
        identity = _identity(parent,spec,'corrected_crop_'+name,reference.checkpoint_sha256,frame,code_sha)
        identity = replace(identity,image_semantics={**dict(identity.image_semantics),
            'color_order':'RGB' if name=='rgb' else 'GRAY',
            'channel_conversion':'RGB slice of actual public corrected image' if name=='rgb' else ADAPTER_CONTRACT['gray'],
            'geometric_transform':'half-open seed ROI after color correction','interpolation':'none'})
        role = RasterRole('image','|u1','HWC' if name=='rgb' else 'HW',(0,255))
        input_measurement = replace(measurement,definition_id='corrected_public_seed_'+name)
        published = publish_checkpoint(store/'inputs',identity,{name:array},
            execution=ExecutionRecord(ExecutionStatus.AVAILABLE),measurement=input_measurement,
            raster_roles={name:role},metadata={'schema':'seedfiddle.corrected-public-crop.v1',
                'composition_sha256':spec.sha256,'lineage':{k:v for k,v in record.items() if k!='inputs'},
                'actual_public_reader':False,'actual_public_crop':False,'adapter_code_sha256':code_sha},
            allow_cpu_transfer=True,limits=limits,cancel=cancel)
        bindings[name] = ReferenceRasterInput(store/'inputs',published.checkpoint_sha256,identity,input_measurement,name,role)
        record['inputs'][name] = bindings[name].to_dict()
    return bindings,frame,record,scale


def run_reference_calibrated_seed(parent:ReferenceRasterInput,spec:ReferenceCalibratedSeedSpec,
    runner:PlantCVReferenceRunner,output_root:Path|str,*,expected_spec_sha256:str,
    expected_source_sha256:str,expected_parent_frame:CoordinateFrame,output_measurement:MeasurementContract,
    allow_cpu_transfer:bool,limits:CheckpointLimits=CheckpointLimits(),timeout_seconds=None,cancel=None):
    """Execute correction, crop, the unchanged seed child, and source-scaled size.

Only complete verified children and successful extra calls publish a parent
manifest. Partial children may still produce an independent scaled size call.
Per-checkpoint numeric admission does not bound public/process peak memory.
"""
    if type(spec) is not ReferenceCalibratedSeedSpec or spec.sha256!=expected_spec_sha256:
        raise ReferenceCalibratedSeedError('The exact selected composition identity is required.')
    if type(runner) is not PlantCVReferenceRunner or not isinstance(limits,CheckpointLimits):
        raise TypeError('An explicit public runner and checkpoint limits are required.')
    if not isinstance(output_measurement,MeasurementContract) or output_measurement.eligible or allow_cpu_transfer is not True:
        raise ReferenceCalibratedSeedError('Public CPU artifacts require an explicitly ineligible measurement contract.')
    if not isinstance(expected_parent_frame,CoordinateFrame) or (
        expected_parent_frame.origin,expected_parent_frame.axes,expected_parent_frame.units)!=(
            'top_left_pixel_centres','x_right_y_down','pixel'):
        raise ReferenceCalibratedSeedError('The parent requires the prepared public pixel-frame convention.')
    _bounds(spec.seed_roi_xyxy,expected_parent_frame.shape)
    card_parameters = _thaw_json(spec.card_parameters)
    if card_parameters['roi_bounds_xyxy'] is not None:
        _bounds(card_parameters['roi_bounds_xyxy'],expected_parent_frame.shape)
    if _cancelled(cancel):
        raise ReferenceCancelledError('Calibrated public composition cancelled before input loading.')
    parent_array = _input(parent,source=expected_source_sha256,frame=expected_parent_frame,channels='RGB',limits=limits)
    code_sha = sha256(Path(__file__).read_bytes()).hexdigest()
    store = Path(output_root)
    child_store = store/'seed'
    parent_call_id = uuid4().hex
    child = None
    crop_record = None
    crop_failure = ''
    scale = None
    scaled_measurement = None

    def poll():
        if _cancelled(cancel):
            raise ReferenceCancelledError('Calibrated public composition cancelled.')
        if sha256(Path(__file__).read_bytes()).hexdigest()!=code_sha:
            raise ReferenceCalibratedSeedError('Composition adapter source changed during execution.')

    def stopped(operation,status,reason,details=None):
        return ReferenceSourceCall(operation,uuid4().hex,status,False,reason,None,details or {})

    def source_call(operation,arrays,parameters,identity,measurement,extra):
        call_id = uuid4().hex
        details = {'requested_parameters':parameters,'parameters_passed':parameters,**extra}
        invoked = False
        digests = {name:_digest(array) for name,array in arrays.items()}
        try:
            poll()
            invoked = True
            result = runner.run(operation,arrays,parameters,timeout_seconds=timeout_seconds,cancel_event=cancel)
            if type(result) is not ReferenceResult or result.operation!=operation:
                raise ReferenceCalibratedSeedError('Source returned a different result type or operation.')
            _verify_runner_identity({'runner_identity':dict(result.identity)},spec.seed_recipe)
            if operation=='analysis.size':
                _checked_result(result,operation,parameters,spec.seed_recipe)
                if result.effective_parameters.get('calibration_origin')!='explicit_pcv.params_adapter' or result.effective_parameters.get('calibration_context_restored') is not True:
                    raise ReferenceCalibratedSeedError('Public size did not verify its temporary scale context.')
            else:
                effective = result.effective_parameters
                if effective.get('callable')!='plantcv.plantcv.transform.auto_correct_color':
                    raise ReferenceCalibratedSeedError('Correction public callable differs.')
                for key,value in parameters.items():
                    if key!='roi_bounds_xyxy':
                        _same(effective.get(key),value,'correction parameter '+key)
                roi = effective.get('roi_adapter',{})
                expected_bounds = parameters['roi_bounds_xyxy'] or [0,0,expected_parent_frame.shape[1],expected_parent_frame.shape[0]]
                expected_roi = {'requested_bounds_xyxy':parameters['roi_bounds_xyxy'],
                    'effective_bounds_xyxy':expected_bounds,'bounds_semantics':'half-open source-frame x0,y0,x1,y1',
                    'origin':'explicit_pcv.Objects_adapter','public_helper':'plantcv.plantcv._helpers._rect_filter',
                    'public_helper_sha256':HELPER_SHA256,'corrected_output_frame':'full input frame'}
                _same(roi,expected_roi,'actual correction ROI adapter')
            for value in (result.computation_runtime_ns,result.wall_runtime_ns):
                if type(value) is not int or value<0:
                    raise ReferenceCalibratedSeedError('Source omitted its actual runtime.')
            details['verified_public_result_received'] = True
            poll()
            if any(_digest(arrays[name])!=digest for name,digest in digests.items()):
                raise ReferenceCalibratedSeedError('Source call mutated a bound input.')
            products = dict(result.arrays)
            name = 'corrected_image' if operation=='transform.auto_correct_color' else 'analysis_image'
            shape = arrays['rgb'].shape if operation=='transform.auto_correct_color' else arrays['image'].shape
            image = products.get(name)
            if set(products)!={name} or not isinstance(image,np.ndarray) or image.dtype!=np.uint8 or image.shape!=shape:
                raise ReferenceCalibratedSeedError('Source output raster inventory, dtype or frame differs.')
            roles = {name:RasterRole('image','|u1','HWC',(0,255))}
            origins = {name:'RGB_adapter_of_actual_public_return'}
            if operation=='analysis.size':
                products['labels_context'] = arrays['labels']
                roles['labels_context'] = RasterRole('instances','<i4','HW')
                origins['labels_context'] = 'input_context_not_computed_output'
            metadata = {'schema':'seedfiddle.public-calibrated-source-call.v1','operation':operation,
                'call_id':call_id,'call_id_origin':'seedfiddle_coordinator_attempt',
                'composition_attempt_id':parent_call_id,'composition_sha256':spec.sha256,
                'composition':spec.to_dict(),'parent_input':parent.to_dict(),
                'source_sha256':expected_source_sha256,'coordinate_frame':identity.coordinate_frame.to_dict(),
                'upstream_checkpoint_sha256':identity.upstream_checkpoint_sha256,'input_array_sha256':digests,
                'parameters_passed':parameters,'effective_parameters':dict(result.effective_parameters),
                'runner_identity':dict(result.identity),'observations':dict(result.observations),
                'visualizations':dict(result.visualizations),'channel_semantics':dict(result.channel_semantics),
                'array_origins':origins,'lineage':extra,'adapter_code_sha256':code_sha,
                'computation_runtime_ns':result.computation_runtime_ns,'wall_runtime_ns':result.wall_runtime_ns,
                'scientifically_validated':False}
            reference = publish_checkpoint(store,identity,products,execution=ExecutionRecord(
                ExecutionStatus.AVAILABLE,elapsed_ms=result.wall_runtime_ns/1e6),measurement=measurement,
                raster_roles=roles,metadata=metadata,allow_cpu_transfer=True,limits=limits,cancel=cancel)
            details['reference_identity_digest'] = result.identity['identity_digest']
            return ReferenceSourceCall(operation,call_id,'executed',True,'',reference,details)
        except Exception as error:
            cancelled = isinstance(error,(ReferenceCancelledError,CheckpointCancelled)) or _cancelled(cancel)
            violated = any(_digest(arrays[name])!=digest for name,digest in digests.items())
            if violated:
                details['input_contract_violated'] = True
            reason = ' '.join((
                f'ReferenceCalibratedSeedError: Bound input changed during source execution; {type(error).__name__}: {error}'
                if violated else f'{type(error).__name__}: {error}').split())[:512]
            return ReferenceSourceCall(operation,call_id,'cancelled' if cancelled and not violated else 'failed',
                                       invoked,reason,None,details)

    correction = source_call('transform.auto_correct_color',{'rgb':parent_array},card_parameters,
        _identity(parent,spec,'public_auto_correction',parent.checkpoint_sha256,expected_parent_frame,code_sha),
        output_measurement,{'parent_rgb':parent.to_dict()})
    del parent_array
    scaled = stopped('analysis.size','not_run','Correction did not provide a usable checkpoint.')
    if correction.checkpoint_reference is not None:
        try:
            poll()
            bindings,local_frame,crop_record,scale = _crop(parent,spec,correction.checkpoint_reference,store,
                output_measurement,limits,cancel,code_sha)
            poll()
            child = run_reference_seed_chain(bindings['gray'],bindings['rgb'],spec.seed_recipe,runner,child_store,
                expected_source_sha256=expected_source_sha256,expected_coordinate_frame=local_frame,
                expected_recipe_identity=spec.seed_recipe.identity,output_measurement=output_measurement,
                allow_cpu_transfer=True,limits=limits,timeout_seconds=timeout_seconds,cancel=cancel)
            if type(child) is not ReferenceSeedChainOutcome or child.preprocessing.recipe_identity!=spec.seed_recipe.identity:
                raise ReferenceCalibratedSeedError('Child returned a different recipe or outcome contract.')
            if child.manifest_sha256 is not None:
                verified = read_reference_seed_chain(child_store,child.manifest_sha256,gray=bindings['gray'],rgb=bindings['rgb'],
                    recipe=spec.seed_recipe,expected_source_sha256=expected_source_sha256,
                    expected_coordinate_frame=local_frame,output_measurement=output_measurement,limits=limits,cancel=cancel)
                del verified
            poll()
            watershed = child.calls[0]
            reference = watershed.checkpoint_reference
            final = child.preprocessing.final_checkpoint
            if reference is None or final is None:
                scaled = stopped('analysis.size','not_run','Child did not publish a usable watershed labels checkpoint.')
            else:
                rgb_array = _input(bindings['rgb'],source=expected_source_sha256,frame=local_frame,channels='RGB',limits=limits)
                ws_identity = _seed_identity(bindings['rgb'],spec.seed_recipe,'segmentation.watershed.capture',final.checkpoint_sha256)
                if reference.identity!=ws_identity:
                    raise ReferenceCalibratedSeedError('Child watershed identity differs from its corrected crop.')
                retained = rgb_array.nbytes
                def load_remaining(root,ref):
                    nonlocal retained
                    remaining = limits.max_total_array_bytes-retained
                    if remaining<1:
                        raise ReferenceCalibratedSeedError('Retained scaled-call context exceeds its numeric budget.')
                    reduced = replace(limits,max_total_array_bytes=remaining,max_array_bytes=min(limits.max_array_bytes,remaining))
                    value = load_stage_input(root,ref.checkpoint_sha256,expected_identity=ref.identity,
                                             expected_measurement=output_measurement,limits=reduced)
                    retained += sum(a.nbytes for a in value.arrays.values())
                    return value
                checkpoint = load_remaining(child_store,reference)
                mask_checkpoint = load_remaining(child_store,final)
                _verify_runner_identity(checkpoint.metadata,spec.seed_recipe)
                _same(checkpoint.metadata.get('call_id'),watershed.call_id,'child watershed call identity')
                _same(checkpoint.metadata.get('rgb_input'),bindings['rgb'].to_dict(),'corrected child RGB')
                ws_parameters,size_parameters,_ = _parameters(spec.seed_recipe)
                result = _checked_result(_result(checkpoint,'segmentation.watershed.capture'),
                                         'segmentation.watershed.capture',ws_parameters,spec.seed_recipe)
                capture = result.effective_parameters['capture']
                arrays = dict(result.arrays)
                if 'empty_coordinates' in capture:
                    coords = arrays.pop('coordinates',None)
                    if coords is None or coords.shape!=(0,2) or coords.dtype!=np.dtype('<i8'):
                        raise ReferenceCalibratedSeedError('Child empty-coordinate transport differs.')
                _validate_watershed_capture(arrays,[],result.effective_parameters,capture['call_id'],{'mask':mask_checkpoint.arrays['mask']})
                _same(result.identity.get('capture_source_files'),capture_source_records(),'child capture source pins')
                _same(checkpoint.metadata.get('input_array_sha256'),
                      {'rgb':_digest(rgb_array),'mask':_digest(mask_checkpoint.arrays['mask'])},'child corrected inputs')
                labels = checkpoint.arrays['labels'].copy()
                labels_sha = _digest(labels)
                label_ids = np.unique(labels)
                positive_ids = label_ids[label_ids>0].tolist()
                marker_ids = capture.get('positive_marker_ids')
                label_facts = {'positive_label_ids':positive_ids,'distinct_positive_label_count':len(positive_ids),
                    'maximum_label_id':int(labels.max()),'labels_sha256':labels_sha,
                    'positive_marker_ids':marker_ids,'estimated_object_count':capture.get('estimated_object_count'),
                    'positive_marker_count':None if marker_ids is None else len(marker_ids)}
                _same(label_facts,child.record['label_facts'],'child labels lineage')
                _same(watershed.details.get('label_facts'),label_facts,'child call label facts')
                _same(watershed.details.get('worker_call_id'),capture['call_id'],'child worker call')
                _same(watershed.details.get('capture_status'),capture['status'],'child capture status')
                expected_roles = {key:RasterRole('instances' if key in {'markers','labels'} else
                    'image' if key=='diagnostic' else 'raster',value.dtype.str,
                    'HWC' if value.ndim==3 else 'HW',(0,255) if key=='diagnostic' else None)
                    for key,value in checkpoint.arrays.items() if key not in {'coordinates','palette'}}
                if dict(checkpoint.raster_roles)!=expected_roles:
                    raise ReferenceCalibratedSeedError('Child watershed raster roles differ.')
                _same(checkpoint.metadata.get('array_origins'),{key:'actual_public_return' if key=='labels' else
                    'BGR_to_RGB_adapter_of_actual_plt_img' if key=='diagnostic' else 'actual_watershed_local'
                    for key in checkpoint.arrays},'child watershed array origins')
                _same(checkpoint.execution.status.value,'empty' if not np.any(labels) else 'available',
                      'child watershed execution status')
                del arrays,result,checkpoint,mask_checkpoint
                parameters = dict(size_parameters)
                if parameters['n_labels'] is None:
                    parameters['n_labels'] = int(labels.max())
                if parameters['n_labels']==0:
                    scaled = stopped('analysis.size','unavailable','Default empty labels require no public size call.',
                        {'effective_n_labels':0,'n_labels_rule':'int(public_labels.max())'})
                else:
                    parameters['calibration'] = scale
                    scaled_measurement = replace(output_measurement,definition_id='public_color_card_scaled_size',
                        units=scale['unit'],calibration_status='public_source_scale_not_scientifically_validated')
                    extra = {'correction_checkpoint_sha256':correction.checkpoint_reference.checkpoint_sha256,
                        'corrected_crop_rgb':bindings['rgb'].to_dict(),'watershed_checkpoint_sha256':reference.checkpoint_sha256,
                        'labels_sha256':labels_sha,'scale_origin':'actual_auto_correct_color_checkpoint',
                        'calibration':scale,'n_labels_rule':'int(public_labels.max())' if size_parameters['n_labels'] is None else 'explicit_positive_selection',
                        'coordinate_units':'ROI-local pixels','linear_scale_rule':'px_width',
                        'area_scale_rule':'px_width * px_height','measurement':scaled_measurement.to_dict()}
                    scaled = source_call('analysis.size',{'image':rgb_array,'labels':labels},parameters,
                        _identity(parent,spec,'public_corrected_scaled_size',reference.checkpoint_sha256,local_frame,code_sha),
                        scaled_measurement,extra)
                del rgb_array,labels
        except Exception as error:
            cancelled = isinstance(error,(ReferenceCancelledError,CheckpointCancelled)) or _cancelled(cancel)
            crop_failure = ' '.join(f'{type(error).__name__}: {error}'.split())[:512]
            scaled = stopped('analysis.size','cancelled' if cancelled else 'not_run',crop_failure)
    cancelled = (_cancelled(cancel) or correction.status=='cancelled' or scaled.status=='cancelled'
                 or (child is not None and child.status=='cancelled'))
    complete = (not cancelled and not crop_failure and correction.status=='executed' and child is not None
                and child.status=='executed' and child.manifest_sha256 is not None and scaled.status=='executed')
    empty = not cancelled and not crop_failure and child is not None and child.status=='empty' and scaled.status=='unavailable'
    status = 'cancelled' if cancelled else 'executed' if complete else 'empty' if empty else 'partial'
    record = {'schema':CHAIN_SCHEMA,'status':status,'complete':complete,'composition':spec.to_dict(),
        'composition_sha256':spec.sha256,'composition_attempt_id':parent_call_id,
        'source_sha256':expected_source_sha256,'parent_frame':expected_parent_frame.to_dict(),
        'parent_input':parent.to_dict(),'correction':correction.to_dict(),'crop':crop_record,'phase_failure':crop_failure,
        'child_record':None if child is None else _thaw_json(child.record),
        'child_manifest':None if child is None or child.manifest_sha256 is None else {'store':'seed','sha256':child.manifest_sha256},
        'scaled_size':scaled.to_dict(),'output_measurement':output_measurement.to_dict(),
        'scaled_measurement':None if scaled_measurement is None else scaled_measurement.to_dict(),
        'extra_runner_request_count':int(correction.invoked)+int(scaled.invoked),
        'extra_verified_public_result_count':sum(bool(c.details.get('verified_public_result_received')) for c in (correction,scaled)),
        'counter_scope':'extra correction/scaled-size requests only; original child retains its own scoped counters',
        'complete_required_verified_public_calls':10 if complete else None,
        'adapter_code_sha256':code_sha,'scientifically_validated':False}
    path = digest = None
    if complete:
        try:
            def before_commit():
                # A caller callback can change stored inputs. Run it before
                # the final audit, which deliberately invokes no caller code.
                poll()
                from .reference_calibrated_seed_reader import _verify_complete_record
                verified = _verify_complete_record(record,store,parent=parent,spec=spec,
                    expected_source_sha256=expected_source_sha256,expected_parent_frame=expected_parent_frame,
                    output_measurement=output_measurement,limits=limits,cancel=None)
                del verified
                if sha256(Path(__file__).read_bytes()).hexdigest()!=code_sha:
                    raise ReferenceCalibratedSeedError('Composition adapter source changed before commit.')
                return False
            path,digest = _manifest(store/'calibrated',record,limits,before_commit)
        except Exception as error:
            status = 'cancelled' if isinstance(error,(ReferenceCancelledError,CheckpointCancelled)) or _cancelled(cancel) else 'partial'
            record.update(status=status,complete=False,complete_required_verified_public_calls=None,
                          phase_failure=' '.join(f'{type(error).__name__}: {error}'.split())[:512])
    return ReferenceCalibratedSeedOutcome(status,correction,child,scaled,record,path,digest)


__all__ = ['ReferenceCalibratedSeedSpec','ReferenceCalibratedSeedOutcome',
           'ReferenceCalibratedSeedError','run_reference_calibrated_seed']
