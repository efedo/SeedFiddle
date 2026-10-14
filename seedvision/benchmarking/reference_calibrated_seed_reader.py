"""Verify a caller-addressed complete calibrated public composition."""
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Mapping

import cv2
import numpy as np

from .checkpoint_inspection import _verify_runner_identity
from .checkpoints import (BenchmarkCheckpoint, CheckpointCancelled, CheckpointLimits, CoordinateFrame,
    ExecutionStatus, MeasurementContract, RasterRole, _canonical_json, _cancelled, _freeze_json, load_checkpoint)
from .reference_calibrated_seed import (ADAPTER_CONTRACT,CHAIN_SCHEMA,HELPER_SHA256,
    ReferenceCalibratedSeedError,ReferenceCalibratedSeedSpec,_bounds,_identity,_scale)
from .reference_runner import _strict_json_loads
from .reference_seed_chain import ReferenceRasterInput,_checked_result,_digest,_parameters
from .reference_seed_chain_reader import _result,_same,read_reference_seed_chain


@dataclass(frozen=True,slots=True)
class VerifiedReferenceCalibratedSeed:
    record: Mapping[str, Any]
    checkpoints: Mapping[str, BenchmarkCheckpoint]
    retained_array_bytes: int


def _validate_caller(parent,spec,source,frame,measurement,limits,cancel):
    if _cancelled(cancel):
        raise CheckpointCancelled('Calibrated composition verification cancelled.')
    if type(spec) is not ReferenceCalibratedSeedSpec or not isinstance(limits,CheckpointLimits):
        raise TypeError('An exact composition spec and checkpoint limits are required.')
    if not isinstance(measurement,MeasurementContract) or measurement.eligible:
        raise ReferenceCalibratedSeedError('Reference measurements remain scientifically ineligible.')
    if not isinstance(frame,CoordinateFrame) or (frame.origin,frame.axes,frame.units)!=(
        'top_left_pixel_centres','x_right_y_down','pixel'):
        raise ReferenceCalibratedSeedError('The parent requires the prepared public pixel-frame convention.')
    if not isinstance(parent,ReferenceRasterInput) or parent.identity.source_sha256!=source or parent.identity.coordinate_frame!=frame:
        raise ReferenceCalibratedSeedError('Trusted parent input source/frame differs.')
    _bounds(spec.seed_roi_xyxy,frame.shape)
    if spec.card_parameters['roi_bounds_xyxy'] is not None:
        _bounds(spec.card_parameters['roi_bounds_xyxy'],frame.shape)


def _verify_complete_record(record,store,*,parent,spec,expected_source_sha256,expected_parent_frame,
                            output_measurement,limits,cancel=None):
    """Shared precommit/read verification; descriptors use a remaining byte budget."""
    def poll():
        if _cancelled(cancel):
            raise CheckpointCancelled('Calibrated composition verification cancelled.')
    poll()
    _validate_caller(parent,spec,expected_source_sha256,expected_parent_frame,output_measurement,limits,cancel)
    required = {'schema','status','complete','composition','composition_sha256','composition_attempt_id',
        'source_sha256','parent_frame','parent_input','correction','crop','phase_failure','child_record',
        'child_manifest','scaled_size','output_measurement','scaled_measurement','extra_runner_request_count',
        'extra_verified_public_result_count','counter_scope','complete_required_verified_public_calls',
        'adapter_code_sha256','scientifically_validated'}
    if not isinstance(record,dict) or set(record)!=required:
        raise ReferenceCalibratedSeedError('Calibrated parent manifest schema differs.')
    expected = {'schema':CHAIN_SCHEMA,'status':'executed','complete':True,'composition':spec.to_dict(),
        'composition_sha256':spec.sha256,'source_sha256':expected_source_sha256,
        'parent_frame':expected_parent_frame.to_dict(),'parent_input':parent.to_dict(),'phase_failure':'',
        'output_measurement':output_measurement.to_dict(),'extra_runner_request_count':2,
        'extra_verified_public_result_count':2,'complete_required_verified_public_calls':10,
        'counter_scope':'extra correction/scaled-size requests only; original child retains its own scoped counters',
        'scientifically_validated':False}
    for key,value in expected.items():
        _same(record[key],value,key)
    attempt = record['composition_attempt_id']
    if not isinstance(attempt,str) or not re.fullmatch(r'[0-9a-f]{32}',attempt):
        raise ReferenceCalibratedSeedError('Invalid composition attempt identity.')
    code_path = Path(__file__).with_name('reference_calibrated_seed.py')
    code_sha = sha256(code_path.read_bytes()).hexdigest()
    _same(record['adapter_code_sha256'],code_sha,'composition adapter code')
    retained = 0
    checkpoints = {}
    def reduced_limits():
        remaining = limits.max_total_array_bytes-retained
        if remaining<1:
            raise ReferenceCalibratedSeedError('Retained parent arrays exceed the remaining numeric budget.')
        return replace(limits,max_total_array_bytes=remaining,max_array_bytes=min(limits.max_array_bytes,remaining))
    def load(name,root,digest,identity,measurement):
        nonlocal retained
        poll()
        checkpoint = load_checkpoint(root,digest,expected_identity=identity,
                                     expected_measurement=measurement,limits=reduced_limits())
        retained += sum(a.nbytes for a in checkpoint.arrays.values())
        checkpoints[name] = checkpoint
        poll()
        return checkpoint
    parent_checkpoint = load('parent',parent.root,parent.checkpoint_sha256,parent.identity,parent.measurement)
    rgb_role = RasterRole('image','|u1','HWC',(0,255))
    parent_rgb = parent_checkpoint.arrays.get(parent.array_name)
    if parent.role!=rgb_role or parent_checkpoint.raster_roles.get(parent.array_name)!=rgb_role or parent.identity.image_semantics['color_order']!='RGB' or parent_rgb is None or parent_rgb.dtype!=np.uint8 or parent_rgb.shape!=(*expected_parent_frame.shape,3):
        raise ReferenceCalibratedSeedError('Parent RGB raster binding differs.')

    def public_call(name,call,identity,measurement,parameters,lineage,input_digests,output_name,shape):
        call_keys = {'operation','call_id','call_id_origin','status','invoked','reason','invoked_meaning',
                     'checkpoint_sha256','checkpoint_identity','details'}
        if set(call)!=call_keys:
            raise ReferenceCalibratedSeedError('Calibrated source-call record schema differs.')
        call_id = call['call_id']
        if not isinstance(call_id,str) or not re.fullmatch(r'[0-9a-f]{32}',call_id):
            raise ReferenceCalibratedSeedError('Invalid source call identity.')
        operation = 'transform.auto_correct_color' if name=='correction' else 'analysis.size'
        for key,value in {'operation':operation,'status':'executed','invoked':True,'reason':'',
            'call_id_origin':'seedfiddle_coordinator_attempt',
            'invoked_meaning':'reference_runner_requested_not_proof_of_public_completion',
            'checkpoint_identity':identity.to_dict()}.items():
            _same(call[key],value,'source '+key)
        checkpoint = load(name,store,call['checkpoint_sha256'],identity,measurement)
        metadata = checkpoint.metadata
        metadata_keys = {'schema','operation','call_id','call_id_origin','composition_attempt_id',
            'composition_sha256','composition','parent_input','source_sha256','coordinate_frame',
            'upstream_checkpoint_sha256','input_array_sha256','parameters_passed','effective_parameters',
            'runner_identity','observations','visualizations','channel_semantics','array_origins','lineage',
            'adapter_code_sha256','computation_runtime_ns','wall_runtime_ns','scientifically_validated'}
        if set(metadata)!=metadata_keys:
            raise ReferenceCalibratedSeedError('Source checkpoint metadata schema differs.')
        fixed = {'schema':'seedfiddle.public-calibrated-source-call.v1','operation':operation,'call_id':call_id,
            'call_id_origin':'seedfiddle_coordinator_attempt','composition_attempt_id':attempt,
            'composition_sha256':spec.sha256,'composition':spec.to_dict(),'parent_input':parent.to_dict(),
            'source_sha256':expected_source_sha256,'coordinate_frame':identity.coordinate_frame.to_dict(),
            'upstream_checkpoint_sha256':identity.upstream_checkpoint_sha256,'input_array_sha256':input_digests,
            'parameters_passed':parameters,'lineage':lineage,'adapter_code_sha256':code_sha,'scientifically_validated':False}
        for key,value in fixed.items():
            _same(metadata.get(key),value,'source metadata '+key)
        _verify_runner_identity(metadata,spec.seed_recipe)
        expected_details = {'requested_parameters':parameters,'parameters_passed':parameters,**lineage,
            'verified_public_result_received':True,'reference_identity_digest':metadata['runner_identity']['identity_digest']}
        _same(call['details'],expected_details,'source call details')
        image = checkpoint.arrays.get(output_name)
        if image is None or image.dtype!=np.uint8 or image.shape!=shape:
            raise ReferenceCalibratedSeedError('Source output frame/dtype differs.')
        if checkpoint.execution.status!=ExecutionStatus.AVAILABLE:
            raise ReferenceCalibratedSeedError('Complete source output is unavailable.')
        for key in ('computation_runtime_ns','wall_runtime_ns'):
            if type(metadata[key]) is not int or metadata[key]<0:
                raise ReferenceCalibratedSeedError('Source runtime is invalid.')
        _same(checkpoint.execution.elapsed_ms,metadata['wall_runtime_ns']/1e6,'source runtime')
        expected_roles = {output_name:rgb_role}
        origins = {output_name:'RGB_adapter_of_actual_public_return'}
        if name=='scaled_size':
            expected_roles['labels_context'] = RasterRole('instances','<i4','HW')
            origins['labels_context'] = 'input_context_not_computed_output'
        if set(checkpoint.arrays)!=set(expected_roles) or dict(checkpoint.raster_roles)!=expected_roles:
            raise ReferenceCalibratedSeedError('Source output inventory or roles differ.')
        _same(metadata['array_origins'],origins,'source output origins')
        return checkpoint

    card_parameters = {k:v for k,v in spec.to_dict()['card_parameters'].items()}
    correction_identity = _identity(parent,spec,'public_auto_correction',parent.checkpoint_sha256,expected_parent_frame,code_sha)
    correction = public_call('correction',record['correction'],correction_identity,output_measurement,
        card_parameters,{'parent_rgb':parent.to_dict()},{'rgb':_digest(parent_rgb)},
        'corrected_image',parent_rgb.shape)
    effective = correction.metadata['effective_parameters']
    _same(effective.get('callable'),'plantcv.plantcv.transform.auto_correct_color','correction callable')
    for key,value in card_parameters.items():
        if key!='roi_bounds_xyxy':
            _same(effective.get(key),value,'correction '+key)
    bounds = card_parameters['roi_bounds_xyxy']
    expected_roi = {'requested_bounds_xyxy':bounds,
        'effective_bounds_xyxy':bounds or [0,0,expected_parent_frame.shape[1],expected_parent_frame.shape[0]],
        'bounds_semantics':'half-open source-frame x0,y0,x1,y1','origin':'explicit_pcv.Objects_adapter',
        'public_helper':'plantcv.plantcv._helpers._rect_filter','public_helper_sha256':HELPER_SHA256,
        'corrected_output_frame':'full input frame'}
    _same(effective.get('roi_adapter'),expected_roi,'correction ROI adapter')
    scale = _scale(correction,spec)
    corrected = correction.arrays['corrected_image']
    x0,y0,x1,y1 = spec.seed_roi_xyxy
    transform = {'parent_frame':expected_parent_frame.to_dict(),'bounds_xyxy':list(spec.seed_roi_xyxy),
        'corrected_rgb_sha256':_digest(corrected),'composition_sha256':spec.sha256,
        'gray_conversion':ADAPTER_CONTRACT['gray'],'opencv':str(cv2.__version__),'adapter_version':'1'}
    frame = CoordinateFrame('corrected_public_seed_roi',(y1-y0,x1-x0),'top_left_pixel_centres',
        'x_right_y_down','pixel',sha256(_canonical_json(transform,name='corrected crop frame')).hexdigest())
    crop = record['crop']
    crop_lineage = {'status':'prepared','bounds_xyxy':list(spec.seed_roi_xyxy),'local_to_parent_xy':[x0,y0],
        'coordinate_frame':frame.to_dict(),'parent_frame':expected_parent_frame.to_dict(),
        'correction_checkpoint_sha256':correction.checkpoint_sha256,'transform':transform}
    if not isinstance(crop,dict) or set(crop)!=set(crop_lineage)|{'inputs'} or set(crop['inputs'])!={'rgb','gray'}:
        raise ReferenceCalibratedSeedError('Crop lineage schema differs.')
    for key,value in crop_lineage.items():
        _same(crop[key],value,'crop '+key)
    bindings = {}
    for name in ('rgb','gray'):
        value = crop['inputs'][name]
        identity = _identity(parent,spec,'corrected_crop_'+name,correction.checkpoint_sha256,frame,code_sha)
        identity = replace(identity,image_semantics={**dict(identity.image_semantics),
            'color_order':'RGB' if name=='rgb' else 'GRAY',
            'channel_conversion':'RGB slice of actual public corrected image' if name=='rgb' else ADAPTER_CONTRACT['gray'],
            'geometric_transform':'half-open seed ROI after color correction','interpolation':'none'})
        measurement = replace(output_measurement,definition_id='corrected_public_seed_'+name)
        role = rgb_role if name=='rgb' else RasterRole('image','|u1','HW',(0,255))
        binding = ReferenceRasterInput(store/'inputs',value['checkpoint_sha256'],identity,measurement,name,role)
        _same(value,binding.to_dict(),'crop binding '+name)
        checkpoint = load('crop.'+name,binding.root,binding.checkpoint_sha256,identity,measurement)
        _same(checkpoint.metadata,{'schema':'seedfiddle.corrected-public-crop.v1','composition_sha256':spec.sha256,
            'lineage':crop_lineage,'actual_public_reader':False,'actual_public_crop':False,
            'adapter_code_sha256':code_sha},'crop metadata '+name)
        if set(checkpoint.arrays)!={name} or dict(checkpoint.raster_roles)!={name:role} or checkpoint.execution.status!=ExecutionStatus.AVAILABLE:
            raise ReferenceCalibratedSeedError('Crop raster contract differs.')
        expected_pixels = corrected[y0:y1,x0:x1] if name=='rgb' else cv2.cvtColor(
            np.ascontiguousarray(corrected[y0:y1,x0:x1,::-1]),cv2.COLOR_BGR2GRAY)
        if not np.array_equal(checkpoint.arrays[name],expected_pixels):
            raise ReferenceCalibratedSeedError('Crop pixels differ from their actual corrected parent.')
        bindings[name] = binding
    child_manifest = record['child_manifest']
    if not isinstance(child_manifest,dict) or set(child_manifest)!={'store','sha256'} or child_manifest['store']!='seed':
        raise ReferenceCalibratedSeedError('Child manifest address differs.')
    child = read_reference_seed_chain(store/'seed',child_manifest['sha256'],gray=bindings['gray'],rgb=bindings['rgb'],
        recipe=spec.seed_recipe,expected_source_sha256=expected_source_sha256,expected_coordinate_frame=frame,
        output_measurement=output_measurement,limits=reduced_limits(),cancel=cancel)
    retained += child.retained_array_bytes
    checkpoints.update({'seed.'+name:value for name,value in child.checkpoints.items()})
    _same(record['child_record'],child.record,'verified original child record')
    labels = child.checkpoints['watershed'].arrays['labels']
    _,parameters,_ = _parameters(spec.seed_recipe)
    rule = 'int(public_labels.max())' if parameters['n_labels'] is None else 'explicit_positive_selection'
    if parameters['n_labels'] is None:
        parameters['n_labels'] = int(labels.max())
    parameters['calibration'] = scale
    scaled_measurement = replace(output_measurement,definition_id='public_color_card_scaled_size',units=scale['unit'],
                                calibration_status='public_source_scale_not_scientifically_validated')
    _same(record['scaled_measurement'],scaled_measurement.to_dict(),'scaled measurement contract')
    extra = {'correction_checkpoint_sha256':correction.checkpoint_sha256,'corrected_crop_rgb':bindings['rgb'].to_dict(),
        'watershed_checkpoint_sha256':child.checkpoints['watershed'].checkpoint_sha256,'labels_sha256':_digest(labels),
        'scale_origin':'actual_auto_correct_color_checkpoint','calibration':scale,'n_labels_rule':rule,
        'coordinate_units':'ROI-local pixels','linear_scale_rule':'px_width','area_scale_rule':'px_width * px_height',
        'measurement':scaled_measurement.to_dict()}
    identity = _identity(parent,spec,'public_corrected_scaled_size',child.checkpoints['watershed'].checkpoint_sha256,frame,code_sha)
    size = public_call('scaled_size',record['scaled_size'],identity,scaled_measurement,parameters,extra,
        {'image':_digest(checkpoints['crop.rgb'].arrays['rgb']),'labels':_digest(labels)},'analysis_image',(*frame.shape,3))
    if not np.array_equal(size.arrays['labels_context'],labels):
        raise ReferenceCalibratedSeedError('Scaled-size labels differ from the actual child watershed.')
    result = _checked_result(_result(size,'analysis.size',arrays={'analysis_image':size.arrays['analysis_image']}),
                             'analysis.size',parameters,spec.seed_recipe)
    _same(result.effective_parameters.get('calibration_origin'),'explicit_pcv.params_adapter','scale context origin')
    _same(result.effective_parameters.get('calibration_context_restored'),True,'scale context restoration')
    call_ids = [record['correction']['call_id'],record['scaled_size']['call_id'],
                *[value['call_id'] for value in record['child_record']['source_calls']]]
    if len(set(call_ids+[attempt]))!=6:
        raise ReferenceCalibratedSeedError('Composition and source-call identities must remain distinct.')
    poll()
    return VerifiedReferenceCalibratedSeed(_freeze_json(record,name='verified calibrated composition'),
                                          MappingProxyType(checkpoints),retained)


def read_reference_calibrated_seed(output_root,manifest_sha256,*,parent,spec,expected_spec_sha256,
    expected_source_sha256,expected_parent_frame,output_measurement,limits=CheckpointLimits(),cancel=None):
    """Read only a trusted digest in the supplied store, never infer a current image."""
    if type(spec) is not ReferenceCalibratedSeedSpec or spec.sha256!=expected_spec_sha256:
        raise ReferenceCalibratedSeedError('The exact selected composition identity is required.')
    _validate_caller(parent,spec,expected_source_sha256,expected_parent_frame,output_measurement,limits,cancel)
    if not isinstance(manifest_sha256,str) or not re.fullmatch(r'[0-9a-f]{64}',manifest_sha256):
        raise ReferenceCalibratedSeedError('A trusted lowercase parent manifest SHA-256 is required.')
    root = Path(output_root).resolve()
    with (root/'calibrated'/'chains'/(manifest_sha256+'.json')).open('rb') as stream:
        raw = stream.read(limits.max_metadata_bytes+1)
    if len(raw)>limits.max_metadata_bytes or sha256(raw).hexdigest()!=manifest_sha256:
        raise ReferenceCalibratedSeedError('Parent manifest differs from its trusted revision or metadata cap.')
    record = _strict_json_loads(raw)
    if not isinstance(record,dict):
        raise ReferenceCalibratedSeedError('Parent manifest is not an object.')
    return _verify_complete_record(record,root,parent=parent,spec=spec,expected_source_sha256=expected_source_sha256,
        expected_parent_frame=expected_parent_frame,output_measurement=output_measurement,limits=limits,cancel=cancel)


__all__ = ['VerifiedReferenceCalibratedSeed','read_reference_calibrated_seed']
