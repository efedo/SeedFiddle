"""Replay a selected native raster stage from verified saved input products.

This is separate from historical StageRunRecords and never changes an archive.
Dense execution/comparison stays CUDA; archived bytes enter through the explicit
bounded checkpoint reader. Agreement is computational, not scientific validation.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
import sys
from types import MappingProxyType
from typing import Mapping

import torch

from seedvision.benchmarking import native_recipe as runner
from seedvision.benchmarking import checkpoints as checkpoint_library
from seedvision.benchmarking.checkpoints import CheckpointLimits, CoordinateFrame, _freeze_json
from seedvision.benchmarking.recipes import RecipeSpec
from seedvision.export import native_results as archives
from seedvision.benchmarking import native_watershed_diagnostic_replay as diagnostic_contract
from seedvision.benchmarking import native_color_replay as color_contract
from seedvision.benchmarking import native_size_replay as size_contract
from seedvision.benchmarking import native_model_replay as model_contract
from seedvision.benchmarking import native_card_replay as card_contract
from seedvision.benchmarking import native_recovery_replay as recovery_contract
from seedvision.benchmarking import native_paper_replay as paper_contract
from seedvision.benchmarking import native_crop_replay as crop_contract


SUPPORTED_REPLAY_OPERATIONS = MappingProxyType({
    'filter.median_blur':(('gray',),('image',),'image'),
    'threshold.binary':(('gray',),('mask',),'mask'),
    'threshold.mean':(('gray',),('mask',),'mask'),
    'threshold.otsu':(('gray',),('mask',),'mask'),
    'threshold.gaussian':(('gray',),('mask',),'mask'),
    'threshold.triangle':(('gray',),('mask',),'mask'),
    'threshold.texture':(('gray',),('mask',),'mask'),
    'threshold.dual_channels':(('rgb',),('mask',),'mask'),
    'threshold.custom_range':(('rgb',),('mask','masked_rgb'),'masked_rgb'),
    'morphology.fill':(('mask',),('mask',),'mask'),
    'morphology.erode':(('gray',),('mask',),'mask'),
    'morphology.dilate':(('gray',),('mask',),'mask'),
    'reference.internal.watershed.distance_map':(('mask',),('distance',),'distance'),
    'reference.internal.watershed.local_maxima':(('distance','mask'),('coordinates','peaks'),'peaks'),
    'reference.internal.watershed.connected_markers':(('peaks',),('markers',),'markers'),
    'reference.internal.watershed.negate_distance':(('distance',),('negative_distance',),'negative_distance'),
    'reference.internal.watershed.masked_flood':(('negative_distance','markers','mask','rgb'),('labels','flood_complete'),'labels'),
    'reference.internal.watershed.marker_count_observation':(('markers','flood_complete'),('marker_count',),None),
    'native.watershed_debug_diagnostic':(('image','labels'),('diagnostic','metadata'),'diagnostic'),
    'analysis.color':(('rgb','labels'),('observations','histogram'),None),
    'analysis.size':(('image','labels'),('observations','diagnostic'),'diagnostic'),
    'calibrated_seed.size_units':(('observations','scale'),('observations','metadata'),None),
    'calibrated_seed.crop':(('corrected_rgb','source_frame','geometry','bounds'),('rgb_image','gray_image','metadata'),'rgb_image'),
    'analysis.naive_bayes_classifier':(('rgb_image','model','model_identity'),('class_masks','class_names','model_identity'),'class_masks'),
    'analysis.kmeans_predict':(('rgb_image','gray_image','model','model_identity'),('labels','model_identity'),'labels'),
    'analysis.kmeans_masks':(('labels','model_identity'),('class_masks','class_names','model_identity'),'class_masks'),
    **{operation:(tuple(inputs),tuple(outputs),preview)
       for operation,(inputs,outputs,preview) in card_contract.PORTS.items()},
    **{operation:(tuple(inputs),tuple(outputs),preview)
       for operation,(inputs,outputs,preview) in recovery_contract.PORTS.items()},
    **{operation:(tuple(inputs),tuple(outputs),preview)
       for operation,(inputs,outputs,preview) in paper_contract.PORTS.items()},
})
PRODUCT_CONTRACTS = MappingProxyType({
    'GrayImage':('raster','|u1'), 'BinaryMask':('raster','|u1'),
    'RGBImage':('rgb','|u1'), 'DistanceMap':('raster','<f4'),
    'NegativeDistanceMap':('raster','<f4'), 'PeakMask':('raster','|b1'),
    'MarkerLabels':('raster','<i4'), 'InstanceLabels':('raster','<i4'),
    'PeakCoordinates':('coordinates','<i8'), 'OperationCompletion':('token',None),
    'Observation':('observation',None),
    'Metadata':('diagnostic_metadata',None),
    'ColorObservations':('color_observations',None),
    'Visualization':('color_chart',None),
    'SizeObservations':('size_observations',None),
    'NaiveBayesModel':('model',None), 'KMeansModel':('model',None),
    'ModelIdentity':('model_identity',None), 'ClassMaskMap':('class_masks',None),
    'OrderedClassNames':('class_names',None), 'LabelRaster':('raster','|u1'),
})
MODEL_OPERATIONS = ('analysis.naive_bayes_classifier','analysis.kmeans_predict','analysis.kmeans_masks')
WATERSHED_TYPED_OPERATIONS = (
    'reference.internal.watershed.local_maxima',
    'reference.internal.watershed.connected_markers',
    'reference.internal.watershed.negate_distance',
    'reference.internal.watershed.masked_flood',
    'reference.internal.watershed.marker_count_observation',
)
EXTENDED_THRESHOLD_OPERATIONS = (
    'threshold.gaussian', 'threshold.triangle',
    'threshold.texture', 'threshold.dual_channels',
)
MAX_SELECTED_BYTES = 48*1024*1024
MAX_TYPED_SELECTED_BYTES = 128*1024*1024
MAX_REPLAY_CUDA_BYTES = 1024*1024*1024
MAX_COMPACT_RUNTIME_AUDIT_CUDA_BYTES = 64*1024
MAX_REPLAY_WORK = 8_000_000_000
MAX_DISTANCE_SCAN_CHUNKS = 65_536
MAX_FRAME_METADATA_ITEMS = 256
MAX_FRAME_METADATA_DEPTH = 8
COMPARE_ELEMENTS = 262_144
ARCHIVE_LIMITS = CheckpointLimits(max_total_array_bytes=256*1024*1024,
    max_array_bytes=128*1024*1024, max_npz_bytes=256*1024*1024)


@dataclass(frozen=True, slots=True)
class NativeStageReplayResult:
    status: str
    stage_id: str
    reason: str
    outputs: Mapping[str, object]
    metadata: Mapping[str, object]
    scientifically_validated: bool = False


class _ReplayCallbackError(RuntimeError):
    """A caller error is not a source-derived unavailable calculation."""


def _result(status, stage_id, reason='', outputs=None, metadata=None):
    return NativeStageReplayResult(status, stage_id, reason,
        MappingProxyType(dict(outputs or {})), _freeze_json(metadata or {}, name='stage replay facts'))



def _canonical_replay_source(recipe):
    """Recover the registered source provenance before validating a working archive."""
    name = recipe.recipe_id
    if name in runner._SEED_RECIPES:
        canonical = runner.build_recipe_catalogue().recipe(name)
    elif name in runner._WORKING_SEED_RECIPES:
        canonical = runner.build_native_seed_recipe(runner.build_recipe_catalogue().recipe(runner._WORKING_SEED_RECIPES[name]))
    elif name in runner._diagnostic_recipe.RECIPE_IDS:
        canonical = runner._diagnostic_recipe.build_native_watershed_diagnostic_recipe(runner._diagnostic_recipe.canonical_base(name))
    elif name in runner._morphology_roi_recipe.RECIPE_IDS:
        canonical = runner._morphology_roi_recipe.build_native_morphology_roi_recipe(runner._morphology_roi_recipe.canonical_base(name))
    elif name in runner._fill_roi_recipe.RECIPE_IDS:
        canonical = runner._fill_roi_recipe.build_native_fill_roi_recipe(runner._fill_roi_recipe.canonical_base(name))
    elif name in runner._NAIVE_BAYES_RECIPES:
        canonical = runner._naive_bayes_recipe.build_native_naive_bayes_recipe()
    elif name in runner._KMEANS_RECIPES:
        canonical = runner._kmeans_recipe.build_native_kmeans_recipe()
    elif name == runner._card_recipe.RECIPE_ID:
        canonical = runner._card_recipe.build_native_card_recipe()
    elif name in runner._calibrated_seed.RECIPE_IDS:
        canonical = runner._calibrated_seed.build_native_calibrated_seed_recipe(name.rsplit('_', 1)[1])
    elif name == runner._recovery_recipe.RECIPE_ID:
        canonical = runner._recovery_recipe.build_native_core_recovery_recipe()
    elif name == runner._paper_recipe.RECIPE_ID:
        canonical = runner._paper_recipe.build_native_smpts_recipe()
    elif name == runner._range_recipe.RECIPE_ID:
        canonical = runner._range_recipe.build_native_range_recipe()
    else:
        canonical = runner._extended_recipe_specs.build_native_extended_threshold_recipes().get(name)
        if canonical is None:
            raise runner.NativeRecipeError('Working replay source is not registered.')
    if runner._reader_recipe.is_reader_recipe(recipe):
        if recipe.recipe_version == runner._reader_recipe.MEDIAN_RECIPE_VERSION:
            canonical = runner._median_roi_recipe._upgrade(canonical)
        canonical = runner._reader_recipe._upgrade(canonical)
    elif runner._median_roi_recipe.is_median_roi_recipe(recipe):
        canonical = runner._median_roi_recipe._upgrade(canonical)
    return canonical


def _runtime(recipe, device):
    calibrated = runner._diagnostic_recipe.base_recipe_id(recipe.recipe_id) in runner._calibrated_seed.RECIPE_IDS
    card = calibrated or recipe.recipe_id == runner._card_recipe.RECIPE_ID
    recovery = recipe.recipe_id == runner._recovery_recipe.RECIPE_ID
    options = {'card': card, 'calibrated': calibrated, 'recovery': recovery,
        'paper': recipe.recipe_id == runner._paper_recipe.RECIPE_ID}
    if runner._morphology_roi_recipe.base_recipe_id(recipe.recipe_id) in runner._diagnostic_recipe.RECIPE_IDS:
        options['diagnostic'] = True
    records, runtime = runner._runtime_records(device, **options)
    return records, runtime


def _compact_frame_metadata(binding):
    """Decode only bounded primitive frame metadata, never arrays or types."""
    visits = 0
    def decode(value, depth=0):
        nonlocal visits
        visits += 1
        if visits > MAX_FRAME_METADATA_ITEMS or depth > MAX_FRAME_METADATA_DEPTH:
            raise archives.NativeResultExportError('Crop frame metadata exceeds its compact traversal bound.')
        if value is None or type(value) in (str,bool,int):
            return value
        if type(value) is not dict or set(value)!={'kind','items'} or type(value['items']) is not list:
            raise archives.NativeResultExportError('Crop frame metadata must contain only tagged primitive mappings/sequences.')
        if value['kind'] in ('list','tuple'):
            return [decode(item,depth+1) for item in value['items']]
        if value['kind']=='mapping':
            result = {}
            for item in value['items']:
                if (type(item) is not list or len(item)!=2 or type(item[0]) is not str or item[0] in result):
                    raise archives.NativeResultExportError('Crop frame metadata has an invalid mapping entry.')
                result[item[0]] = decode(item[1],depth+1)
            return result
        raise archives.NativeResultExportError('Crop frame metadata contains an unsupported tag.')
    return decode(binding)


def _bound_frame_inputs(recipe, nodes, stage):
    incoming = [edge for edge in recipe.dependencies if edge.target_stage==stage.stage_id]
    node = nodes[stage.stage_id]
    if (node['status'] not in ('provided','executed','empty','partial') or node['product_key'] is None
            or set(node['input_keys'])!={edge.target_port for edge in incoming}):
        raise archives.NativeResultExportError('Crop frame ancestry has unavailable or inconsistent inputs.')
    for edge in incoming:
        source = nodes[edge.source_stage]
        if (source['status'] not in ('provided','executed','empty','partial')
                or source['product_key'] is None or edge.source_port not in dict(source['outputs'])
                or node['input_keys'][edge.target_port] != runner._digest([source['product_key'],edge.source_port])):
            raise archives.NativeResultExportError('Crop frame ancestry differs from its declared producer-port keys.')
    return incoming


def _frame_raster(binding, specs, shape, dtype='|u1'):
    if (type(binding) is not dict or set(binding)!={'kind','name'} or binding['kind']!='array'):
        raise archives.NativeResultExportError('Frame ancestry requires a direct raster array.')
    spec = specs[binding['name']]
    if (tuple(spec['shape'])!=shape or spec['dtype']!=dtype or spec['layout']!='C-contiguous'):
        raise archives.NativeResultExportError('Frame ancestry raster violates its shape/dtype/layout contract.')


def _compact_frame_field(binding, name, decoder=None):
    """Select a bounded primitive field without decoding unrelated geometry."""
    if (type(binding) is not dict or set(binding)!={'kind','items'} or binding['kind']!='mapping'
            or type(binding['items']) is not list or len(binding['items'])>MAX_FRAME_METADATA_ITEMS):
        raise archives.NativeResultExportError('Crop geometry requires a compact tagged mapping.')
    seen, selected = set(), None
    for item in binding['items']:
        if type(item) is not list or len(item)!=2 or type(item[0]) is not str or item[0] in seen:
            raise archives.NativeResultExportError('Crop geometry has an invalid mapping entry.')
        seen.add(item[0])
        if item[0]==name:
            selected = item[1]
    if name not in seen:
        raise archives.NativeResultExportError('Crop geometry omits its required frame field.')
    return _compact_frame_metadata(selected) if decoder is None else decoder(selected,archives.NativeResultExportError)


def _port(stage, name, direction):
    return next(item for item in stage.ports if item.name==name and item.direction==direction)




def _foreground_region_context(recipe, stage, node, frame):
    """Read only the archive-validated working mask adapter, in this input frame."""
    key = 'foreground_region_bounds_xyxy'
    binding = node['parameters']
    if not any(item[0] == key for item in binding.get('items', ())):
        return None
    from seedvision.pipeline.workflows import FOREGROUND_OPERATIONS
    bounds = _compact_frame_field(binding, key)
    if (recipe.provenance.mode.value != 'enhanced' or stage.operation_id not in FOREGROUND_OPERATIONS
            or type(bounds) is not list or len(bounds) != 4
            or any(type(value) is not int or value < 0 for value in bounds)
            or bounds[0] >= bounds[2] or bounds[1] >= bounds[3]):
        raise archives.NativeResultExportError('Replay foreground region context is invalid.')
    provenance = _compact_frame_metadata(node['backend_provenance'])
    archives._foreground_provenance(stage, provenance, bounds, frame.shape)
    return tuple(bounds)


def _foreground_region_plan(plan, frame):
    """Charge the full-frame byte clone retained alongside primitive output."""
    pixels = math.prod(frame.shape)
    extra = ((pixels + 511) // 512) * 512
    result = dict(plan)
    result['cuda_bytes'] += extra
    if result.get('work') is not None:
        result['work'] += 2 * pixels
    if (result['cuda_bytes'] > MAX_REPLAY_CUDA_BYTES
            or result.get('work') is not None and result['work'] > MAX_REPLAY_WORK):
        raise NativeStageReplayUnavailable('Foreground adapter exceeds replay CUDA/work admission.')
    result['foreground_region_adapter_cuda_bytes'] = extra
    result['foreground_region_adapter_work'] = 2 * pixels
    return result


def _paper_context(recipe,nodes,stage,frame,poll):
    context={}
    for edge in recipe.dependencies:
        if edge.target_stage==stage.stage_id and edge.target_port in ('components','source_components','final_components'):
            value=paper_contract.decode(dict(nodes[edge.source_stage]['outputs'])[edge.source_port],poll)
            context[edge.target_port]=paper_contract.inventory(value,frame.shape)
    if stage.operation_id=='smpts_paper.components':
        context['components']=paper_contract.inventory(paper_contract.decode(
            dict(nodes[stage.stage_id]['outputs'])['components'],poll),frame.shape)
    context['label_counts']={'labels':len(context.get('final_components',context.get('components',()))),
        'source_labels':len(context.get('source_components',())),
        'final_labels':len(context.get('final_components',()))}
    return context


def _paper_binding(binding,role,specs,frame,context,poll):
    if role in paper_contract.RASTER_ROLES:
        shape,dtype=paper_contract.raster_spec(role,frame.shape,context)
        _frame_raster(binding,specs,shape,dtype)
        return 'array',binding['name']
    return 'compact',paper_contract.validate(paper_contract.decode(binding,poll),role,frame.shape,context)


def _paper_frame(recipe,report,stage,specs,poll):
    factory=runner._paper_recipe
    if recipe.recipe_id!=factory.RECIPE_ID:
        raise archives.NativeResultExportError('Paper replay requires its canonical provisional graph.')
    canonical=factory.build_native_smpts_recipe()
    expected={item.stage_id:item for item in canonical.stages}
    nodes={node['stage_id']:node for node in report['nodes']}
    frame=CoordinateFrame.from_dict(report['coordinate_frame'])
    # Direct native callers may supply an explicit frame identity/transform.
    # Bind its exact archived identity; no new frame-name restriction is imposed.
    seen=set();path=[]
    def visit(current):
        if current.stage_id in seen:return
        if current.stage_id not in expected or current.operation_id!=expected[current.stage_id].operation_id:
            raise archives.NativeResultExportError('Paper ancestry crosses a noncanonical stage.')
        seen.add(current.stage_id)
        incoming=_bound_frame_inputs(recipe,nodes,current)
        wanted={(e.target_port,e.source_stage,e.source_port,e.data_type)
            for e in canonical.dependencies if e.target_stage==current.stage_id}
        if {(e.target_port,e.source_stage,e.source_port,e.data_type) for e in incoming}!=wanted:
            raise archives.NativeResultExportError('Paper ancestry differs from canonical producer ports.')
        parameters=paper_contract.canonical(recipe.parameters_for_runner(current.stage_id),poll)
        saved_parameters = dict(paper_contract.decode(nodes[current.stage_id]['parameters'], poll))
        region = _foreground_region_context(recipe, current, nodes[current.stage_id], frame)
        if region is not None:
            saved_parameters.pop('foreground_region_bounds_xyxy')
        if saved_parameters != parameters:
            raise archives.NativeResultExportError('Saved paper ancestry controls differ from declared controls.')
        outputs=dict(nodes[current.stage_id]['outputs'])
        if current.stage_id==factory.SOURCE:
            if current.operation_id!='external.image_inputs' or incoming:
                raise archives.NativeResultExportError('Paper ancestry omits its provided image source.')
            _frame_raster(outputs.get('gray_image'),specs,frame.shape)
            _frame_raster(outputs.get('rgb_image'),specs,(*frame.shape,3))
        else:
            paper_contract.controls(current.operation_id,parameters)
            context=_paper_context(recipe,nodes,current,frame,poll)
            roles=paper_contract.PORTS[current.operation_id][1]
            if set(outputs)!=set(roles):
                raise archives.NativeResultExportError('Paper ancestor output ports differ from their exact contract.')
            for name,role in roles.items():_paper_binding(outputs[name],role,specs,frame,context,poll)
            if current.operation_id=='smpts_paper.crops':
                parent=next(e for e in incoming if e.target_port=='final_labels')
                if outputs['labels']!=dict(nodes[parent.source_stage]['outputs'])[parent.source_port]:
                    raise archives.NativeResultExportError('Packed extraction must retain its final-label input alias.')
        for edge in incoming:
            path.append({'stage_id':current.stage_id,'input_port':edge.target_port,
                'producer_stage':edge.source_stage,'producer_port':edge.source_port,
                'input_key':nodes[current.stage_id]['input_keys'][edge.target_port]})
            visit(recipe.stage(edge.source_stage))
    visit(stage)
    return frame,{'frame_scope':'source','frame_lineage':{'paper_ancestry':path},
        'paper_reconstruction':'provisional declared assumptions; original capsule execution not established'},\
        _paper_context(recipe,nodes,stage,frame,poll)


def _validate_paper_product(value,role,frame,context,name,device,poll,error_type,owned_storage):
    if role not in paper_contract.RASTER_ROLES:
        try:paper_contract.validate(paper_contract.canonical(value,poll),role,frame.shape,context)
        except paper_contract.NativePaperReplayError as error:raise error_type(str(error)) from error
        return
    shape,dtype=paper_contract.raster_spec(role,frame.shape,context)
    if (not torch.is_tensor(value) or value.layout!=torch.strided or value.requires_grad
            or value.device!=device or value.dtype!=_torch_dtype(dtype)
            or tuple(value.shape)!=shape or not value.is_contiguous()):
        raise error_type('Paper tensor violates its frame/packed-shape/device/dtype/layout contract.')
    storage=value.untyped_storage()
    if storage.data_ptr() not in (owned_storage or ()) and storage.nbytes()>paper_contract.rounded(value.numel()*value.element_size()):
        raise error_type('Paper tensor retains unadmitted backing storage.')
    flat=value.view(-1)
    # Fresh output label IDs are validated independently of possibly different
    # archived inventory cardinality, then all compact leaves are compared.
    limit=(runner._paper_recipe.paper.MAX_COMPONENTS if error_type is runner.NativeRecipeError
        else context['label_counts'].get(name,runner._paper_recipe.paper.MAX_COMPONENTS))
    for start in range(0,value.numel(),COMPARE_ELEMENTS):
        poll();tile=flat[start:start+COMPARE_ELEMENTS]
        if role in ('binary','packed_mask'):valid=(tile==0)|(tile==255)
        elif role=='labels':valid=(tile>=0)&(tile<=limit)
        else:continue
        if not bool(valid.all().item()):
            raise error_type('Paper tensor violates its binary mask or consecutive-ID domain.')
        del tile,valid


def _paper_relationships(operation,products,inputs,parameters,error_type):
    if operation=='smpts_paper.contacts' and error_type is runner.NativeRecipeError:
        wanted=runner._paper_recipe.paper.select_contacts(inputs['components'],inputs['labels'].shape,
            parameters['contact_box_area'])
        if products['selected_parent_ids']!=wanted:
            raise error_type('Fresh contact IDs differ from the strict bounding-box cutoff.')
    if operation=='smpts_paper.crops':
        if products['labels'] is not inputs['final_labels']:
            raise error_type('Crop labels must alias the final-label input.')
        present={row['parent_id'] for row in products['crops']}
        missing=tuple(row['label_id'] for row in inputs['source_components'] if row['label_id'] not in present)
        if products['disappeared_parent_ids']!=missing:
            raise error_type('Disappeared parent IDs differ from the packed crop lineage.')


def _recovery_frame(recipe,report,stage,specs,poll):
    """Bind full-frame canonical ancestry without executing any ancestor."""
    factory=runner._recovery_recipe
    if recipe.recipe_id!=factory.RECIPE_ID:
        raise archives.NativeResultExportError('Recovery replay requires its canonical experimental graph.')
    canonical=factory.build_native_core_recovery_recipe()
    expected={item.stage_id:item for item in canonical.stages}
    frame=CoordinateFrame.from_dict(report['coordinate_frame'])
    binding=recovery_contract.canonical(runner._freeze_json({'coordinate_frame':frame.to_dict(),
        'source_sha256':report['source_sha256']},name='recovery replay source binding'),poll)
    recovery_contract.validate(binding,'source_binding',frame,binding,{},poll)
    nodes={node['stage_id']:node for node in report['nodes']};seen=set();path=[]
    def visit(current):
        if current.stage_id in seen:return
        if current.stage_id not in expected or current.operation_id!=expected[current.stage_id].operation_id:
            raise archives.NativeResultExportError('Recovery ancestry crosses a noncanonical producer.')
        seen.add(current.stage_id)
        incoming=_bound_frame_inputs(recipe,nodes,current)
        wanted={(edge.target_port,edge.source_stage,edge.source_port,edge.data_type)
            for edge in canonical.dependencies if edge.target_stage==current.stage_id}
        if {(edge.target_port,edge.source_stage,edge.source_port,edge.data_type) for edge in incoming}!=wanted:
            raise archives.NativeResultExportError('Recovery ancestry differs from canonical producer ports.')
        saved=recovery_contract.decode(nodes[current.stage_id]['parameters'],poll)
        parameters=recovery_contract.canonical(runner._freeze_json(recipe.parameters_for_runner(current.stage_id),
            name='recovery replay parameters'),poll)
        if saved!=parameters:
            raise archives.NativeResultExportError('Saved recovery ancestry controls differ from their declared values.')
        outputs=dict(nodes[current.stage_id]['outputs'])
        if current.stage_id==factory.CORES:
            recovery_contract.validate(recovery_contract.decode(outputs.get('inventory'),poll),
                'inventory',frame,binding,parameters,poll)
        elif current.stage_id==factory.VALIDITY:
            recovery_contract.validate(recovery_contract.decode(outputs.get('metadata'),poll),
                'validity_metadata',frame,binding,parameters,poll)
        elif current.stage_id in (factory.PARTITION,factory.ANNOTATION):
            partition_stage=current.stage_id==factory.PARTITION
            port,role=('partition','partition') if partition_stage else ('annotation','annotation')
            _,transport=_recovery_binding(outputs.get(port),role,specs,frame,binding,parameters,poll)
            metadata_role='partition_metadata' if partition_stage else 'annotation_metadata'
            metadata=recovery_contract.validate(recovery_contract.decode(outputs.get('metadata'),poll),
                metadata_role,frame,binding,parameters,poll)
            raster='labels' if partition_stage else 'uncertainty_band'
            _,array=_recovery_binding(outputs.get(raster),'markers' if partition_stage else 'unit_binary',
                specs,frame,binding,{},poll)
            if transport['arrays'][raster]!=array or transport['metadata']!=metadata:
                raise archives.NativeResultExportError('Recovery inner/separate output aliases disagree.')
            if partition_stage:
                by_port={edge.target_port:dict(nodes[edge.source_stage]['outputs'])[edge.source_port] for edge in incoming}
                for target,name in (('original_foreground','foreground'),('markers','markers'),('validity','validity'),('elevation','elevation')):
                    if transport['arrays'][target]!=by_port[name].get('name'):
                        raise archives.NativeResultExportError('Partition retained inputs differ from their canonical producers.')
                if metadata['inventory']!=recovery_contract.decode(by_port['inventory'],poll):
                    raise archives.NativeResultExportError('Partition inventory differs from its core producer.')
            else:
                parent_edge=next(edge for edge in incoming if edge.target_port=='partition')
                parent=_recovery_binding(dict(nodes[parent_edge.source_stage]['outputs'])[parent_edge.source_port],
                    'partition',specs,frame,binding,{},poll)[1]['metadata']
                if (tuple((row['label_id'],row['original_core_pixels']) for row in metadata['objects'])
                        !=tuple((row['label_id'],row['original_core_pixels']) for row in parent['inventory']['objects'])
                        or metadata['partition_status']!=parent['status']
                        or any(metadata[name]!=parent['plan'][name] for name in
                            ('original_foreground_pixels','validity_clipped_foreground_pixels'))
                        or metadata['unresolved_pixels']!=parent['unresolved_pixels']):
                    raise archives.NativeResultExportError('Annotation identity/counts differ from its retained partition.')
        if current.stage_id==factory.SOURCE:
            _frame_raster(outputs.get('gray_image'),specs,frame.shape)
            _frame_raster(outputs.get('rgb_image'),specs,(*frame.shape,3))
            recovery_contract.validate(recovery_contract.decode(outputs.get('source_frame'),poll),
                'source_binding',frame,binding,{},poll)
        for edge in incoming:
            source=recipe.stage(edge.source_stage);value=dict(nodes[source.stage_id]['outputs']).get(edge.source_port)
            if edge.source_port=='source_frame':
                recovery_contract.validate(recovery_contract.decode(value,poll),'source_binding',frame,binding,{},poll)
            elif edge.data_type in ('RecoveryPartition','RecoveryAnnotation'):
                _recovery_binding(value,'partition' if edge.data_type=='RecoveryPartition' else 'annotation',
                    specs,frame,binding,{} if edge.data_type=='RecoveryPartition' else recipe.parameters_for_runner(source.stage_id),poll)
            elif edge.source_port=='inventory':
                recovery_contract.validate(recovery_contract.decode(value,poll),'inventory',frame,binding,{},poll)
            else:_frame_raster(value,specs,frame.shape,'<i4' if edge.data_type in ('MarkerLabels','InstanceLabels')
                else '<f4' if edge.data_type in ('DistanceMap','NegativeDistanceMap') else '|u1')
            path.append({'stage_id':current.stage_id,'input_port':edge.target_port,
                'producer_stage':source.stage_id,'producer_port':edge.source_port,
                'input_key':nodes[current.stage_id]['input_keys'][edge.target_port]})
            visit(source)
    visit(stage)
    return frame,{'frame_scope':'source','frame_lineage':{'recovery_ancestry':path}},binding


def _recovery_binding(binding,role,specs,frame,source_binding,parameters,poll):
    if role in recovery_contract.RASTER_ROLES:
        dtype='<i4' if role in ('markers','ids') else '<f4' if role=='elevation' else '|u1'
        shape=frame.shape
        if role=='ids':
            if type(binding) is not dict or set(binding)!={'kind','name'} or binding['kind']!='array':
                raise archives.NativeResultExportError('Recovery IDs require a direct array.')
            shape=tuple(specs[binding['name']]['shape'])
            if len(shape)!=1 or not 0<=shape[0]<=runner._recovery_recipe.core.MAX_LABELS:
                raise archives.NativeResultExportError('Recovery ID vector exceeds its bounded shape.')
        _frame_raster(binding,specs,shape,dtype)
        return 'array',binding['name']
    if role in ('partition','annotation','annotation_metadata_only'):
        fields=_recovery_mapping(binding,('tensors','metadata') if role=='partition' else ('uncertainty_band','metadata'))
        metadata_role='partition_metadata' if role=='partition' else 'annotation_metadata'
        metadata=recovery_contract.validate(recovery_contract.decode(fields['metadata'],poll),metadata_role,
            frame,source_binding,parameters if role!='partition' else {},poll)
        if role=='partition':
            tensors=_recovery_mapping(fields['tensors'],recovery_contract.PARTITION_TENSORS)
            arrays={name:_recovery_binding(value,recovery_contract.PARTITION_TENSORS[name],specs,
                frame,source_binding,{},poll)[1] for name,value in tensors.items()}
            if specs[arrays['label_ids']]['shape'][0]!=len(metadata['objects']):
                raise archives.NativeResultExportError('Partition ID header differs from its object records.')
        else:
            band=_recovery_binding(fields['uncertainty_band'],'unit_binary',specs,frame,source_binding,{},poll)[1]
            arrays={} if role=='annotation_metadata_only' else {'uncertainty_band':band}
        descriptor=MappingProxyType({'arrays':MappingProxyType(arrays),'metadata':metadata})
        return role,descriptor
    return 'compact',recovery_contract.validate(recovery_contract.decode(binding,poll),role,
        frame,source_binding,parameters,poll)


def _recovery_mapping(binding,keys):
    if (type(binding) is not dict or set(binding)!={'kind','items'} or binding['kind']!='mapping'
            or type(binding['items']) is not list or len(binding['items'])!=len(keys)):
        raise archives.NativeResultExportError('Recovery transport requires its exact tagged mapping.')
    result={}
    for entry in binding['items']:
        if type(entry) is not list or len(entry)!=2 or type(entry[0]) is not str or entry[0] in result:
            raise archives.NativeResultExportError('Recovery transport has invalid/duplicate fields.')
        result[entry[0]]=entry[1]
    if set(result)!=set(keys):raise archives.NativeResultExportError('Recovery transport has missing or unknown fields.')
    return result


def _recovery_array_names(products):
    for kind,value in products.values():
        if kind=='array':yield value
        elif kind in ('partition','annotation','annotation_metadata_only'):
            yield from value['arrays'].values()


def _recovery_materialize(kind,value,uploaded):
    if kind=='array':return uploaded[value]
    if kind=='partition':
        return MappingProxyType({'tensors':MappingProxyType({name:uploaded[array] for name,array in value['arrays'].items()}),
            'metadata':value['metadata']})
    if kind=='annotation':return MappingProxyType({'uncertainty_band':uploaded[value['arrays']['uncertainty_band']],
        'metadata':value['metadata']})
    if kind=='annotation_metadata_only':return MappingProxyType({'metadata':value['metadata']})
    return value


def _compare_recovery_product(actual,expected,poll):
    if torch.is_tensor(actual):return _compare_product(actual,expected,poll)
    if isinstance(actual,Mapping) and ('tensors' in actual or 'uncertainty_band' in actual):
        rows={}
        if 'tensors' in actual:
            rows.update({name:_compare_product(value,expected['tensors'][name],poll) for name,value in actual['tensors'].items()})
        else:rows['uncertainty_band']=_compare_product(actual['uncertainty_band'],expected['uncertainty_band'],poll)
        rows['metadata']=recovery_contract.compare(actual['metadata'],expected['metadata'],poll)
        return {'kind':'recovery_transport','by_field':rows,
            'mismatched_elements':sum(row['mismatched_elements'] for row in rows.values()),
            'compared_elements':sum(row['compared_elements'] for row in rows.values()),
            'shape_equal':all(row['shape_equal'] for row in rows.values())}
    return recovery_contract.compare(actual,expected,poll)


@dataclass(frozen=True,slots=True)
class _CardFrames:
    source: CoordinateFrame
    local: CoordinateFrame
    bounds: tuple[int,int,int,int]


def _card_binding(binding,role,specs,frames,poll=None,*,check_versions=True):
    if role in card_contract.RASTER_ROLES:
        shape,dtype=card_contract.raster_spec(role,frames.source.shape,frames.bounds)
        _frame_raster(binding,specs,shape,dtype)
        return 'array',binding['name']
    value=card_contract.validate(card_contract.decode(binding,poll),role,
        frames.source.shape,frames.bounds,poll=poll,check_versions=check_versions)
    if role=='bounds' and tuple(value)!=frames.bounds:
        raise archives.NativeResultExportError('Card bounds differ from the declared ROI control.')
    return 'compact',value


def _card_frame(recipe,report,stage,specs,poll,*,check_versions=False):
    """Bind canonical card ancestry without loading or executing its ancestors."""
    factory=runner._card_recipe
    calibrated=runner._diagnostic_recipe.base_recipe_id(recipe.recipe_id) in runner._calibrated_seed.RECIPE_IDS
    if not calibrated and recipe.recipe_id!=factory.RECIPE_ID:
        raise archives.NativeResultExportError('Card replay requires a canonical shared card graph.')
    canonical=factory.build_native_card_recipe()
    anchor=canonical.stages[0]
    expected={current.stage_id:current for current in canonical.stages}
    if stage.stage_id not in expected or expected[stage.stage_id].operation_id!=stage.operation_id:
        raise archives.NativeResultExportError('Card operation differs from its canonical stage ID.')
    parent=CoordinateFrame.from_dict(report['coordinate_frame'])
    _,shape,bounds=card_contract.frames(parent.shape,
        recipe.parameters_for_runner(factory.PREPROCESS)['roi_bounds_xyxy'])
    translation={'parent_frame':parent.to_dict(),'bounds_xyxy':bounds,'local_to_parent_xy':bounds[:2]}
    local=CoordinateFrame(parent.frame_id+'.color_card_roi',shape,'ROI top-left',parent.axes,parent.units,
        runner._digest(translation))
    frames=_CardFrames(parent,local,bounds)
    nodes={node['stage_id']:node for node in report['nodes']}
    seen=set();path=[]
    def visit(current):
        if current.stage_id in seen:return
        seen.add(current.stage_id)
        if current.stage_id not in expected or current.operation_id!=expected[current.stage_id].operation_id:
            raise archives.NativeResultExportError('Card ancestry crosses a noncanonical producer.')
        incoming=_bound_frame_inputs(recipe,nodes,current)
        wanted={(edge.target_port,edge.source_stage,edge.source_port,edge.data_type)
            for edge in canonical.dependencies if edge.target_stage==current.stage_id}
        if {(edge.target_port,edge.source_stage,edge.source_port,edge.data_type) for edge in incoming}!=wanted:
            raise archives.NativeResultExportError('Card ancestry differs from canonical producer ports.')
        outputs=dict(nodes[current.stage_id]['outputs'])
        if current.stage_id==anchor.stage_id:
            for name,header in (('rgb_image',(*parent.shape,3)),('gray_image',parent.shape)):
                _frame_raster(outputs.get(name),specs,header)
            if calibrated:
                source=_compact_frame_metadata(outputs.get('source_frame'))
                if source!={'coordinate_frame':parent.to_dict(),'source_sha256':report['source_sha256']}:
                    raise archives.NativeResultExportError('Card image anchor differs from its saved source/frame.')
        else:
            parameters=recipe.parameters_for_runner(current.stage_id)
            card_contract.controls(current.operation_id,parameters)
            saved=card_contract.decode(nodes[current.stage_id]['parameters'],poll)
            if saved!=card_contract.canonical(_freeze_json(parameters,name='card parameters'),poll):
                raise archives.NativeResultExportError('Card effective controls differ from the selected recipe.')
            roles=card_contract.PORTS[current.operation_id][1]
            if set(outputs)!=set(roles):
                raise archives.NativeResultExportError('Card ancestor omits or adds declared output ports.')
            for name,role in roles.items():
                _card_binding(outputs[name],role,specs,frames,poll,check_versions=check_versions)
        for edge in incoming:
            path.append({'stage_id':current.stage_id,'input_port':edge.target_port,
                'producer_stage':edge.source_stage,'producer_port':edge.source_port,
                'input_key':nodes[current.stage_id]['input_keys'][edge.target_port]})
            visit(recipe.stage(edge.source_stage))
    visit(stage)
    inputs,outputs,_=card_contract.PORTS[stage.operation_id]
    port_frames={direction:{name:(local if role.startswith('roi_') or role=='geometry' else parent).to_dict()
        for name,role in roles.items() if role in card_contract.RASTER_ROLES or role=='geometry'}
        for direction,roles in (('input',inputs),('output',outputs))}
    primary=local if stage.operation_id in ('color_card.preprocess','color_card.geometry') else parent
    facts={'frame_scope':'card_roi' if primary is local else 'source',
        'source_coordinate_frame':parent.to_dict(),'card_roi_coordinate_frame':local.to_dict(),
        'card_roi_bounds_xyxy':bounds,'card_local_to_parent_xy':bounds[:2],
        'card_port_frames':port_frames,'frame_lineage':{'card_ancestry':path},
        'upstream_calculations':'verified recorded ancestry only; ancestors are not recomputed'}
    return primary,facts,frames


def _crop_frame(recipe,report,stage,specs,poll,*,check_versions=False):
    factory=runner._calibrated_seed
    base=runner._diagnostic_recipe.base_recipe_id(recipe.recipe_id)
    if base not in factory.RECIPE_IDS or stage.stage_id!=factory.CROP or stage.operation_id!=crop_contract.OPERATION:
        raise archives.NativeResultExportError('Crop replay requires its exact canonical calibrated stage.')
    canonical=factory.build_native_calibrated_seed_recipe(base.removeprefix('plantcv_calibrated_seed_'))
    nodes={row['stage_id']:row for row in report['nodes']}
    incoming=_bound_frame_inputs(recipe,nodes,stage)
    wanted={(edge.target_port,edge.source_stage,edge.source_port,edge.data_type)
        for edge in canonical.dependencies if edge.target_stage==stage.stage_id}
    if {(edge.target_port,edge.source_stage,edge.source_port,edge.data_type) for edge in incoming}!=wanted:
        raise archives.NativeResultExportError('Crop ancestry differs from canonical producer ports.')
    _,card_facts,card_frames=_card_frame(recipe,report,recipe.stage(runner._card_recipe.AFFINE),
        specs,poll,check_versions=check_versions)
    parameters=recipe.parameters_for_runner(stage.stage_id)
    if crop_contract.compare(crop_contract.decode(nodes[stage.stage_id]['parameters'],poll),
            crop_contract.canonical(_freeze_json(parameters,name='crop parameters'),poll),poll)['mismatched_elements']:
        raise archives.NativeResultExportError('Crop effective controls differ from the selected recipe.')
    context=crop_contract.frames(card_frames.source,parameters,report['source_sha256'],card_frames.bounds)
    poll=runner._card_recipe.correction._guarded(poll,metadata=(context,))
    bindings={edge.target_port:dict(nodes[edge.source_stage]['outputs'])[edge.source_port] for edge in incoming}
    for name,role in crop_contract.INPUT_ROLES.items():
        _crop_binding(bindings[name],role,specs,context,poll,check_versions=check_versions)
    outputs=dict(nodes[stage.stage_id]['outputs'])
    if set(outputs)!=set(crop_contract.OUTPUT_ROLES):
        raise archives.NativeResultExportError('Crop outputs differ from their exact contract.')
    for name,role in crop_contract.OUTPUT_ROLES.items():
        _crop_binding(outputs[name],role,specs,context,poll,check_versions=check_versions)
    path=[{'stage_id':stage.stage_id,'input_port':edge.target_port,
        'producer_stage':edge.source_stage,'producer_port':edge.source_port,
        'input_key':nodes[stage.stage_id]['input_keys'][edge.target_port]} for edge in incoming]
    facts={'frame_scope':'calibrated_seed_roi','parent_coordinate_frame':context.parent.to_dict(),
        'card_roi_coordinate_frame':card_frames.local.to_dict(),'card_roi_bounds_xyxy':context.card_bounds,
        'crop_port_frames':{'corrected_rgb':context.parent.to_dict(),'geometry':card_frames.local.to_dict(),
            'rgb_image':context.local.to_dict(),'gray_image':context.local.to_dict()},
        'frame_lineage':{'bounds_xyxy':context.translation['bounds_xyxy'],
            'local_to_parent_xy':context.translation['local_to_parent_xy'],
            'crop_ancestry':path,'card_ancestry':card_facts['frame_lineage']['card_ancestry']},
        'upstream_calculations':'verified recorded ancestry only; card detection and correction are not recomputed'}
    return context.local,facts,context


def _crop_binding(binding,role,specs,context,poll=None,*,check_versions=True):
    if role in ('parent_rgb','local_rgb','local_gray'):
        shape,dtype=crop_contract.raster_spec(role,context)
        _frame_raster(binding,specs,shape,dtype)
        return 'array',binding['name']
    return 'compact',crop_contract.validate(crop_contract.decode(binding,poll),role,
        context,poll,check_versions=check_versions)


def _validate_crop_product(value,role,context,device,poll,error_type,owned_storage):
    if role not in ('parent_rgb','local_rgb','local_gray'):
        try:crop_contract.validate(value,role,context,poll)
        except crop_contract.NativeCropReplayError as error:raise error_type(str(error)) from error
        return
    shape,dtype=crop_contract.raster_spec(role,context)
    if (not torch.is_tensor(value) or value.layout!=torch.strided or value.requires_grad
            or value.device!=device or value.dtype!=_torch_dtype(dtype)
            or tuple(value.shape)!=shape or not value.is_contiguous()):
        raise error_type('Crop raster violates its exact parent/local frame/device/dtype/layout.')
    storage=value.untyped_storage()
    if storage.data_ptr() not in owned_storage and storage.nbytes()>crop_contract.rounded(value.numel()*value.element_size()):
        raise error_type('Crop raster retains unadmitted backing storage.')


def _card_plan(frames,operation,parameters,selected_bytes):
    plan=card_contract.plan(operation,frames.source.shape,parameters,bounds=frames.bounds,
        selected_limit=MAX_TYPED_SELECTED_BYTES,cuda_limit=MAX_REPLAY_CUDA_BYTES)
    if selected_bytes>MAX_TYPED_SELECTED_BYTES:
        raise card_contract.NativeCardReplayUnavailable('Card selected arrays exceed replay admission.')
    return {'pixels':frames.source.shape[0]*frames.source.shape[1],
        'roi_pixels':frames.local.shape[0]*frames.local.shape[1],'selected_bytes':selected_bytes,
        'selected_admission_bytes':plan.selected_bytes,'cuda_bytes':plan.cuda_bytes,
        'primitive_scratch_bytes':plan.primitive_scratch_bytes,'work':plan.work,'work_details':dict(plan.work_details),
        'selected_limit_bytes':MAX_TYPED_SELECTED_BYTES,'cuda_limit_bytes':MAX_REPLAY_CUDA_BYTES,
        'runtime_audit_cuda_reserve_bytes':card_contract.AUDIT_CUDA_BYTES,
        'compact_metadata_limit_bytes':card_contract.MAX_METADATA_BYTES,
        'bounds':'modeled admission; not host RSS or elapsed-time guarantees'}


def _observation(value, label, pixels, error_type):
    expected = {'sample':label,'variable':'estimated_object_count','trait':'estimated object count',
        'method':'plantcv.plantcv.watershed','scale':'none','datatype':"<class 'int'>"}
    if (not isinstance(value,Mapping) or set(value)!=set(expected)|{'value'}
            or any(type(value[key]) is not str or value[key]!=item for key,item in expected.items())
            or type(value['value']) is not int or not 0<=value['value']<pixels):
        raise error_type('Marker-count record violates its exact observation contract.')
    return _freeze_json(value,name='replayed marker count')


def _binding_product(binding, data_type, specs, frame, *, label=None, parameters=None, poll=None,
                     metadata_role=None, size_parameters=None, size_unit='pixels',
                     card_role=None,card_frames=None,check_versions=True,
                     recovery_role=None,recovery_binding=None,paper_role=None,paper_context=None,paper_name=None,crop_role=None,crop_context=None):
    if crop_role is not None:
        return _crop_binding(binding,crop_role,specs,crop_context,poll,check_versions=check_versions)
    if paper_role is not None:
        return _paper_binding(binding,paper_role,specs,frame,paper_context,poll)
    if recovery_role is not None:
        return _recovery_binding(binding,recovery_role,specs,frame,recovery_binding,parameters,poll)
    if card_role is not None:
        return _card_binding(binding,card_role,specs,card_frames,poll,check_versions=check_versions)
    kind,dtype = PRODUCT_CONTRACTS[data_type]
    if metadata_role is not None or kind=='size_observations':
        value=size_contract.decode(binding,archives.NativeResultExportError,poll)
        if kind=='size_observations':
            size_contract.observations(value,size_parameters or parameters,archives.NativeResultExportError,size_unit)
        elif metadata_role=='size_scale':size_contract.scale(value,archives.NativeResultExportError)
        elif metadata_role=='size_units':size_contract.metadata(value,archives.NativeResultExportError)
        else:raise archives.NativeResultExportError('Unknown size metadata role.')
        return 'compact',value
    if kind in ('color_observations','color_chart'):
        port='observations' if kind=='color_observations' else 'histogram'
        value=color_contract.decode(binding,archives.NativeResultExportError,poll)
        return 'compact',color_contract.validate(value,port,parameters,frame.shape[0]*frame.shape[1],archives.NativeResultExportError)
    if kind=='diagnostic_metadata':
        value = diagnostic_contract.validate(diagnostic_contract.decode(binding,archives.NativeResultExportError),
            parameters,frame.shape[0]*frame.shape[1],archives.NativeResultExportError)
        return 'compact',value
    if kind=='token':
        if binding is not True:
            raise archives.NativeResultExportError('Flood completion must be the exact completed token.')
        return 'compact',True
    if kind=='observation':
        value = _observation(_compact_frame_metadata(binding),label,frame.shape[0]*frame.shape[1],
            archives.NativeResultExportError)
        return 'compact',value
    if kind=='coordinates':
        if (type(binding) is not dict or set(binding)!={'kind','name'} or binding['kind']!='array'):
            raise archives.NativeResultExportError('Peak coordinates require a direct array.')
        spec = specs[binding['name']]
        shape = tuple(spec['shape'])
        if (len(shape)!=2 or shape[1]!=2 or not 0<=shape[0]<=runner._peaks.MAX_CANDIDATES
                or spec['dtype']!=dtype or spec['layout']!='C-contiguous'):
            raise archives.NativeResultExportError('Peak coordinate shape/dtype/layout exceeds its contract.')
    else:
        _frame_raster(binding,specs,(*frame.shape,3) if kind=='rgb' else frame.shape,dtype)
    return 'array',binding['name']


def _replay_frame(recipe, report, stage, specs):
    """Resolve the selected port's source or explicitly declared seed ROI frame."""
    parent = CoordinateFrame.from_dict(report['coordinate_frame'])
    factory = runner._calibrated_seed
    if stage.operation_id=='calibrated_seed.size_units':
        return _size_units_frame(recipe,report,stage,specs)
    if stage.operation_id in MODEL_OPERATIONS:
        return _model_frame(recipe,report,stage,specs)[:2]
    frames = [item.semantics.get('frame') for item in stage.ports
              if not (stage.operation_id==runner._diagnostic_recipe.OPERATION and item.data_type=='Metadata')]
    local = factory.LOCAL_FRAME in frames
    if local and (runner._diagnostic_recipe.base_recipe_id(recipe.recipe_id) not in factory.RECIPE_IDS
                  or any(value!=factory.LOCAL_FRAME for value in frames)):
        raise archives.NativeResultExportError('Selected raster ports disagree on their declared ROI frame.')
    nodes = {node['stage_id']:node for node in report['nodes']}
    frame, translation, crop = parent,None,None
    if local:
        crop = recipe.stage(factory.CROP)
        if crop.operation_id!='calibrated_seed.crop':
            raise archives.NativeResultExportError('ROI frame requires the canonical crop producer.')
        frame,translation = factory._crop_frame(parent,recipe.parameters_for_runner(crop.stage_id)['seed_roi_xyxy'])
    path,seen,visiting = [],set(),set()
    def visit(current):
        if current.stage_id in seen or crop is not None and current.stage_id==crop.stage_id:
            return
        if current.stage_id in visiting:
            raise archives.NativeResultExportError('Replay frame ancestry contains a cycle.')
        visiting.add(current.stage_id)
        incoming = _bound_frame_inputs(recipe,nodes,current)
        for ancestor in incoming:
            source = recipe.stage(ancestor.source_stage)
            target_port = _port(current,ancestor.target_port,'input')
            source_port = _port(source,ancestor.source_port,'output')
            if (target_port.data_type!=source_port.data_type
                    or any((item.semantics.get('frame')==factory.LOCAL_FRAME)!=local
                           for item in (target_port,source_port))):
                raise archives.NativeResultExportError('Replay ancestry crosses an undeclared coordinate frame/type.')
            _binding_product(dict(nodes[source.stage_id]['outputs'])[ancestor.source_port],source_port.data_type,specs,frame)
            path.append({'stage_id':current.stage_id,'input_port':ancestor.target_port,
                'producer_stage':source.stage_id,'producer_port':ancestor.source_port,
                'input_key':nodes[current.stage_id]['input_keys'][ancestor.target_port]})
            visit(source)
        if not incoming and current.operation_id!='external.image_inputs':
            raise archives.NativeResultExportError('Replay ancestry has no canonical image input/crop anchor.')
        visiting.remove(current.stage_id);seen.add(current.stage_id)
    visit(stage)
    if not local:
        return frame,{'frame_scope':'source','frame_lineage':{'raster_ancestry':path}}
    crop_inputs = _bound_frame_inputs(recipe,nodes,crop)
    crop_bindings = {item.target_port:dict(nodes[item.source_stage]['outputs'])[item.source_port]
        for item in crop_inputs}
    _frame_raster(crop_bindings['corrected_rgb'],specs,(*parent.shape,3))
    _frame_raster(dict(nodes[crop.stage_id]['outputs']).get('rgb_image'),specs,(*frame.shape,3))
    source_record = _compact_frame_metadata(crop_bindings['source_frame'])
    if archives._json_bytes(source_record)!=archives._json_bytes({
            'coordinate_frame':parent.to_dict(),'source_sha256':report['source_sha256']}):
        raise archives.NativeResultExportError('Crop source record differs from the archived source/frame.')
    metadata = _compact_frame_metadata(dict(nodes[crop.stage_id]['outputs']).get('metadata'))
    bounds = _compact_frame_metadata(crop_bindings['bounds'])
    card_parameters = recipe.parameters_for_runner(runner._card_recipe.PREPROCESS)
    expected_bounds = runner._card_recipe.card._bounds(card_parameters['roi_bounds_xyxy'],*parent.shape)
    if (type(bounds) is not list or len(bounds)!=4 or any(type(value) is not int for value in bounds)
            or tuple(bounds)!=tuple(expected_bounds)):
        raise archives.NativeResultExportError('Saved card bounds differ from the declared parent ROI.')
    rect = _compact_frame_field(crop_bindings['geometry'],'bounding_rect')
    if (type(rect) is not list or len(rect)!=4 or any(type(value) is not int for value in rect)
            or min(rect[:2])<0 or min(rect[2:])<=0
            or rect[0]+rect[2]>bounds[2]-bounds[0] or rect[1]+rect[3]>bounds[3]-bounds[1]):
        raise archives.NativeResultExportError('Saved card bounding rectangle differs from its ROI frame.')
    overlap = factory._crop_overlap(translation['bounds_xyxy'],bounds,rect)
    expected = {**translation,'coordinate_frame':frame.to_dict(),
        'source_sha256':report['source_sha256'],'gray_conversion':factory.GRAY_CONVERSION,
        'labels_and_coordinates':factory.LOCAL_COORDINATES}
    if (type(metadata) is not dict or set(metadata)!=set(expected)|{'card_bounding_overlap_pixels'}
            or archives._json_bytes({key:metadata[key] for key in expected})!=archives._json_bytes(expected)
            or type(metadata['card_bounding_overlap_pixels']) is not int
            or metadata['card_bounding_overlap_pixels']!=overlap):
        raise archives.NativeResultExportError('Crop metadata differs from its exact ROI/source/frame/translation contract.')
    return frame, {'frame_scope':'calibrated_seed_roi','parent_coordinate_frame':parent.to_dict(),
        'frame_lineage':{'crop_stage':crop.stage_id,'crop_product_key':nodes[crop.stage_id]['product_key'],
            'crop_input_keys':nodes[crop.stage_id]['input_keys'],'raster_ancestry':path,
            'bounds_xyxy':translation['bounds_xyxy'],'local_to_parent_xy':translation['local_to_parent_xy'],
            'card_bounding_overlap_pixels':overlap,
            'crop_metadata_sha256':runner._digest(metadata)}}


def _model_frame(recipe, report, stage, specs):
    """Bind canonical image ancestry and the separate frame-less model root."""
    kind = 'bayes' if stage.operation_id=='analysis.naive_bayes_classifier' else 'kmeans'
    factory = runner._naive_bayes_recipe if kind=='bayes' else runner._kmeans_recipe
    if (recipe.recipe_id != (factory.NAIVE_BAYES_RECIPE_ID if kind=='bayes' else factory.KMEANS_RECIPE_ID)
            or stage.stage_id!=stage.operation_id):
        raise archives.NativeResultExportError('Model replay requires its canonical working recipe and stage.')
    frame = CoordinateFrame.from_dict(report['coordinate_frame'])
    nodes = {row['stage_id']:row for row in report['nodes']}
    image = recipe.stage(factory.IMAGE_STAGE_ID)
    model = recipe.stage(factory.MODEL_STAGE_ID)
    operation = 'external.plantcv.naive_bayes_pdf' if kind=='bayes' else 'external.plantcv.kmeans_centres'
    if (image.operation_id!='external.image_inputs' or model.operation_id!=operation
            or _bound_frame_inputs(recipe,nodes,image) or _bound_frame_inputs(recipe,nodes,model)):
        raise archives.NativeResultExportError('Model replay roots differ from canonical image/model producers.')
    image_outputs = dict(nodes[image.stage_id]['outputs'])
    for name,shape in (('rgb_image',(*frame.shape,3)),('gray_image',frame.shape)):
        _frame_raster(image_outputs.get(name),specs,shape)
    model_outputs = dict(nodes[model.stage_id]['outputs'])
    asset = model_contract.asset_binding(model_outputs.get('model'),kind,specs,
        recipe.parameters_for_runner(model.stage_id)['model_source'])
    root_identity = model_contract.identity(model_contract.decode(model_outputs.get('model_identity')),kind)
    if root_identity!=asset.identity:
        raise archives.NativeResultExportError('Model root identity differs from its byte-bound asset.')
    predictor = recipe.stage(factory.CLASSIFIER_STAGE_ID if kind=='bayes' else factory.PREDICT_STAGE_ID)
    expected_edges = {('rgb_image',image.stage_id,'rgb_image'),('model',model.stage_id,'model'),
                      ('model_identity',model.stage_id,'model_identity')}
    if kind=='kmeans':expected_edges.add(('gray_image',image.stage_id,'gray_image'))
    incoming = _bound_frame_inputs(recipe,nodes,predictor)
    if {(edge.target_port,edge.source_stage,edge.source_port) for edge in incoming}!=expected_edges:
        raise archives.NativeResultExportError('Model prediction ancestry differs from its canonical producer ports.')
    identity_out = model_contract.identity(model_contract.decode(
        dict(nodes[predictor.stage_id]['outputs']).get('model_identity')),kind)
    if identity_out!=asset.identity:
        raise archives.NativeResultExportError('Prediction identity differs from the archived model root.')
    if stage.operation_id=='analysis.kmeans_masks':
        incoming_masks = _bound_frame_inputs(recipe,nodes,stage)
        if {(edge.target_port,edge.source_stage,edge.source_port) for edge in incoming_masks}!={
                ('labels',predictor.stage_id,'labels'),('model_identity',predictor.stage_id,'model_identity')}:
            raise archives.NativeResultExportError('Mask replay differs from its canonical prediction ancestry.')
        _frame_raster(dict(nodes[predictor.stage_id]['outputs']).get('labels'),specs,frame.shape)
        incoming += incoming_masks
    lineage = [{'stage_id':edge.target_stage,'input_port':edge.target_port,
        'producer_stage':edge.source_stage,'producer_port':edge.source_port,
        'input_key':nodes[edge.target_stage]['input_keys'][edge.target_port]} for edge in incoming]
    return frame, {'frame_scope':'source','frame_lineage':{'raster_and_model_ancestry':lineage},
        'model_lineage':{'kind':kind,'producer_stage':model.stage_id,
            'producer_product_key':nodes[model.stage_id]['product_key'],'source_sha256':asset.identity['source_sha256'],
            'numerical_sha256':asset.identity['tables_sha256' if kind=='bayes' else 'centres_sha256'],
            'origin_access':'canonical text only; original path is not opened'}}, asset


def _select_model_products(stage, input_bindings, output_bindings, specs, frame, asset, parameters, poll):
    value,kind = asset.identity,asset.kind
    identity = model_contract.identity(model_contract.decode(input_bindings['model_identity'],poll),kind)
    output_identity = model_contract.identity(model_contract.decode(output_bindings['model_identity'],poll),kind)
    if identity!=value or output_identity!=value:
        raise archives.NativeResultExportError('Selected model identities differ from their canonical asset.')
    inputs={'model_identity':('compact',identity)}
    if stage.operation_id=='analysis.kmeans_masks':
        _frame_raster(input_bindings['labels'],specs,frame.shape)
        inputs['labels']=('array',input_bindings['labels']['name'])
    else:
        if parameters:
            raise archives.NativeResultExportError('Model inference has no independent numerical controls.')
        port='rgb_image' if kind=='bayes' or value['channels']==3 else 'gray_image'
        for name in ('rgb_image',) if kind=='bayes' else ('rgb_image','gray_image'):
            _frame_raster(input_bindings[name],specs,(*frame.shape,3) if name=='rgb_image' else frame.shape)
        inputs[port]=('array',input_bindings[port]['name'])
        inputs['model']=('model',asset)
    outputs={'model_identity':('compact',output_identity)}
    if stage.operation_id=='analysis.kmeans_predict':
        _frame_raster(output_bindings['labels'],specs,frame.shape)
        outputs['labels']=('array',output_bindings['labels']['name'])
    else:
        count=len(value['class_names']) if kind=='bayes' else len(model_contract.mask_names(parameters,value['cluster_count'])[0])
        names,pairs=model_contract.class_mask_bindings(output_bindings['class_names'],output_bindings['class_masks'],
            specs,frame.shape,count=count,maximum=runner._naive_bayes.MAX_CLASSES if kind=='bayes' else runner._kmeans_model.MAX_CLUSTERS,poll=poll)
        outputs.update(class_names=('compact',names),class_masks=('mask_map',pairs))
    return inputs,outputs


def _model_array_names(products):
    for kind,value in products.values():
        if kind=='array':yield value
        elif kind=='model':yield value.array_name
        elif kind=='mask_map':yield from (name for _,name in value)


def _model_plan(frame, operation, parameters, asset, model=None):
    options={'selected_limit':MAX_TYPED_SELECTED_BYTES,'cuda_limit':MAX_REPLAY_CUDA_BYTES}
    if operation=='analysis.naive_bayes_classifier':plan=model_contract.bayes_plan(frame.shape,asset.identity,**options)
    elif operation=='analysis.kmeans_masks':plan=model_contract.masks_plan(frame.shape,asset.identity,parameters,**options)
    else:plan=model_contract.prediction_plan(frame.shape,asset.identity,model,**options)
    return {'pixels':frame.shape[0]*frame.shape[1],'selected_bytes':plan.selected_bytes,'cuda_bytes':plan.cuda_live_bytes,
        'selected_admission_bytes':plan.selected_bytes,'selected_limit_bytes':MAX_TYPED_SELECTED_BYTES,
        'cuda_limit_bytes':MAX_REPLAY_CUDA_BYTES,'work_limit':runner._kmeans_predict.MAX_WORK if plan.work is not None else None,
        'native_admitted_live_bytes':plan.native_live_bytes,'primitive_scratch_bytes':plan.scratch_bytes,
        'work':plan.work,'work_details':dict(plan.work_details),
        'runtime_audit_cuda_reserve_bytes':model_contract.AUDIT_CUDA_BYTES,
        'raw_model_cuda_bytes':0,'model_origin_io':False,'bounds':'admission, not RSS/runtime guarantees'}


def _model_signature(value):
    model=value.get('model') if isinstance(value,Mapping) else None
    ids = ((id(model.probabilities),id(model.class_names)) if isinstance(model,runner._naive_bayes.NaiveBayesModel)
           else (id(model.centres),) if isinstance(model,runner._kmeans_model.KMeansModel) else ())
    order=tuple(value.get('class_masks',())) if isinstance(value,Mapping) else ()
    return runner._nested_signature((),value),ids,order


def _effective_traits(recipe, node, contract):
    parameters=recipe.parameters_for_runner(node['stage_id'])
    resolved=size_contract.decode(node['parameters'],archives.NativeResultExportError) if contract is size_contract else _compact_frame_metadata(node['parameters'])
    if (not isinstance(resolved,Mapping) or set(resolved)!=set(parameters)
            or any(resolved[key]!=value and not (isinstance(resolved[key],tuple) and isinstance(value,list)
                    and resolved[key]==tuple(value)) for key,value in parameters.items()
                if key!='n_labels' or value is not None)):
        raise archives.NativeResultExportError('Trait effective parameters differ from the selected recipe.')
    contract.controls(resolved,archives.NativeResultExportError)
    return dict(resolved)


def _size_units_frame(recipe, report, stage, specs):
    """Units retain size's ROI frame and independently bind the card-scale branch."""
    nodes={node['stage_id']:node for node in report['nodes']}
    incoming=_bound_frame_inputs(recipe,nodes,stage)
    size_edge=next(edge for edge in incoming if edge.target_port=='observations')
    source=recipe.stage(size_edge.source_stage)
    scale_edge=next(edge for edge in incoming if edge.target_port=='scale')
    card=runner._card_recipe
    if (source.operation_id!='analysis.size' or size_edge.source_port!='observations'
            or scale_edge.source_stage!=card.SCALE or scale_edge.source_port!='metadata'):
        raise archives.NativeResultExportError('Size units require their canonical size and card-scale producers.')
    frame,facts=_replay_frame(recipe,report,source,specs)
    _effective_traits(recipe,nodes[source.stage_id],size_contract)
    parent=CoordinateFrame.from_dict(report['coordinate_frame'])
    path=[]
    for stage_id in (card.SCALE,card.GEOMETRY,card.PREPROCESS,card.PREFIX+'bgr'):
        current=recipe.stage(stage_id)
        for edge in _bound_frame_inputs(recipe,nodes,current):
            path.append({'stage_id':stage_id,'input_port':edge.target_port,'producer_stage':edge.source_stage,
                'producer_port':edge.source_port,'input_key':nodes[stage_id]['input_keys'][edge.target_port]})
    geometry=dict(nodes[card.GEOMETRY]['outputs'])['geometry']
    dimensions=tuple(_compact_frame_field(geometry,key,size_contract.decode) for key in (
        'median_color_chip_width','median_color_chip_height'))
    for value in dimensions:size_contract._number(value,archives.NativeResultExportError,True)
    parameters=recipe.parameters_for_runner(card.SCALE)
    resolved=size_contract.decode(nodes[card.SCALE]['parameters'],archives.NativeResultExportError)
    if not isinstance(resolved,Mapping) or set(resolved)!={'color_chip_size'}:
        raise archives.NativeResultExportError('Card scale requires its exact effective settings.')
    if size_contract.canonical(parameters)!=resolved:
        # Recipe JSON lists have the same semantics as decoded tagged tuples.
        expected=parameters['color_chip_size'];actual=resolved.get('color_chip_size')
        if not (set(resolved)=={'color_chip_size'} and isinstance(expected,list) and actual==tuple(expected)):
            raise archives.NativeResultExportError('Card scale effective settings differ from the selected recipe.')
    try:
        physical=card.card._parameters(0,0,1.27,.8,parameters['color_chip_size'])
        width,height,unit=card.card._scale_metadata(*dimensions,physical)
    except (ValueError,TypeError,ZeroDivisionError) as error:
        raise archives.NativeResultExportError('Invalid archived card scale settings or dimensions.') from error
    saved=size_contract.decode(dict(nodes[card.SCALE]['outputs'])['metadata'],archives.NativeResultExportError)
    size_contract.scale(saved,archives.NativeResultExportError)
    if saved!={'px_width':width,'px_height':height,'unit':unit}:
        raise archives.NativeResultExportError('Card scale differs from its archived geometry and physical settings.')
    bounds=card.card._bounds(recipe.parameters_for_runner(card.PREPROCESS)['roi_bounds_xyxy'],*parent.shape)
    roi_shape=(bounds[3]-bounds[1],bounds[2]-bounds[0])
    for port in ('roi_bgr','threshold'):
        _frame_raster(dict(nodes[card.PREPROCESS]['outputs'])[port],specs,
            (*roi_shape,3) if port=='roi_bgr' else roi_shape)
    _frame_raster(dict(nodes[card.PREFIX+'bgr']['outputs'])['bgr'],specs,(*parent.shape,3))
    anchor=next(edge for edge in recipe.dependencies if edge.target_stage==card.PREFIX+'bgr')
    if recipe.stage(anchor.source_stage).operation_id!='external.image_inputs':
        raise archives.NativeResultExportError('Card scale branch requires a canonical image anchor.')
    _frame_raster(dict(nodes[anchor.source_stage]['outputs'])[anchor.source_port],specs,(*parent.shape,3))
    facts['size_units_lineage']={'size_producer':source.stage_id,'size_product_key':nodes[source.stage_id]['product_key'],
        'scale_product_key':nodes[card.SCALE]['product_key'],'scale_ancestry':path,
        'chip_dimensions_pixels':dimensions,'physical_settings':parameters,
        'scope':'selected units calculation; upstream label maximum and card measurements are not recomputed'}
    return frame,facts


class NativeStageReplayUnavailable(runner.NativeRecipeError):
    """A valid selected operation exceeds its declared replay resource domain."""


def _extended_threshold_admission(pixels, operation, parameters):
    """Separate malformed controls from resource and source-precision limits."""
    native = runner._extended_thresholds
    try:
        if operation=='threshold.gaussian':
            ksize,offset = native._validate_mean_parameters(parameters['ksize'],parameters['offset'])
            native._object_type(parameters['object_type'])
            if ksize not in (*native.GAUSSIAN_WEIGHTS,*native.GAUSSIAN_FLOAT32_KERNEL_BITS):
                raise ValueError('Effective Gaussian kernels must be odd sizes 3..15 (even requests round up).')
        elif operation=='threshold.triangle':
            native._object_type(parameters['object_type'])
        elif operation=='threshold.texture':
            ksize,offset = parameters['ksize'],parameters['offset']
            if (type(ksize) is not int or ksize not in (3,5,7,9)
                    or type(offset) is not int or not 1<=offset<ksize
                    or type(parameters['threshold']) is not int or not 0<=parameters['threshold']<=255
                    or parameters['texture_method'] not in native.TEXTURE_PROPERTIES
                    or parameters['borders'] not in native.TEXTURE_BORDERS):
                raise ValueError('Texture controls exceed the supported kernel/offset/property/border domain.')
        else:
            if (parameters['x_channel'] not in native.CHANNELS or parameters['y_channel'] not in native.CHANNELS
                    or type(parameters['above']) is not bool):
                raise ValueError('Dual-channel names or above-line control are invalid.')
            native._line(parameters['points'])
    except (ValueError,TypeError,KeyError) as error:
        raise archives.NativeResultExportError(f'Invalid extended threshold controls: {error}') from error
    try:
        if operation=='threshold.gaussian':
            admission = native._gaussian_admission(pixels,ksize)
            admission.update(requested_kernel_size=parameters['ksize'],effective_kernel_size=ksize)
        elif operation=='threshold.texture':
            admission = native._texture_admission(pixels,ksize,offset)
        elif operation=='threshold.dual_channels':
            admission = native._dual_channel_admission(pixels,parameters['x_channel'],parameters['y_channel'])
        else:
            if pixels>native.MAX_TRIANGLE_PIXELS:
                raise native.ExtendedThresholdError('Triangle histogram exceeds its exact float32-count pixel domain.')
            admission = {'pixel_limit':native.MAX_TRIANGLE_PIXELS,
                'tile_scratch_bytes':0, 'fixed_scratch_bytes':65_536,
                'compact_cpu_histogram_bins':256,
                'source_preamble':'exclude byte255; require a rising-edge histogram peak; then guarded Otsu'}
    except native.ExtendedThresholdError as error:
        raise NativeStageReplayUnavailable(str(error)) from error
    return admission


def _plan(shape, operation, parameters, selected_bytes):
    pixels = shape[0]*shape[1]
    # Includes uploaded inputs/reference, result, primitive full-frame buffers,
    # tiled scratch and comparison. Primitive algorithms keep their own stricter
    # parameter/convergence checks. The cap includes owned replay allocations,
    # not unrelated CUDA caches or allocations in another task.
    cuda_bytes = 128*pixels+128*1024*1024
    work = 256*pixels
    work_details = {'model':'bounded tensor-element visits; not device instructions or elapsed time'}
    diagnostic = operation == runner._diagnostic_recipe.OPERATION
    selected_limit = MAX_TYPED_SELECTED_BYTES if diagnostic or operation=='analysis.size' or operation in WATERSHED_TYPED_OPERATIONS else MAX_SELECTED_BYTES
    if not 1<=pixels<=runner.MAX_RASTER_PIXELS:
        if operation in (*EXTENDED_THRESHOLD_OPERATIONS, 'threshold.custom_range'):
            raise NativeStageReplayUnavailable('Selected threshold exceeds replay raster admission.')
        raise runner.NativeRecipeError('Selected stage exceeds replay raster admission.')
    if operation=='calibrated_seed.size_units':
        if parameters or selected_bytes:
            raise archives.NativeResultExportError('Size units replay requires no controls or selected arrays.')
        cuda_bytes=MAX_COMPACT_RUNTIME_AUDIT_CUDA_BYTES
        work=None
        work_details={'model':'bounded compact conversion; fixed shared CUDA table audit is budgeted separately',
            'conversion_cuda_bytes':0,'runtime_audit_cuda_bytes':MAX_COMPACT_RUNTIME_AUDIT_CUDA_BYTES,
            'runtime_audit_table_values':sum(len(values) for values in (
                runner._range._LAB_GAMMA_VALUES,runner._range._LAB_CBRT_VALUES)),
            'max_metadata_bytes_per_product':size_contract.MAX_METADATA_BYTES,
            'max_metadata_items_per_product':size_contract.MAX_METADATA_ITEMS}
    elif operation=='analysis.size':
        size_contract.controls(parameters,archives.NativeResultExportError)
        count=parameters['n_labels'];native=runner._size
        # Sequential masks: do not multiply full-frame CUDA storage by count.
        # Sparse CPU geometry is bounded per measurement; optional retained
        # drawing geometry has its own aggregate cap and cannot suppress traits.
        work=None
        cuda_bytes=max(cuda_bytes,14*pixels+64*min(pixels,native.MAX_TILE_PIXELS)
            +64*native.MAX_TOPOLOGY_BOUNDARY_POINTS+native.MAX_SIZE_DIAGNOSTIC_SCRATCH_BYTES)
        work_details={'model':'operation-specific numerical and drawing hard caps; no composite instruction or elapsed-time estimate',
            'n_labels':count,'label_scan_pixels':count*pixels,
            'max_topology_points_per_label':native.MAX_TOPOLOGY_BOUNDARY_POINTS,
            'max_contour_work_per_label':native.MAX_CONTOUR_WORK_UNITS,
            'max_trace_steps_per_label':native.MAX_CONTOUR_TRACE_STEPS,
            'max_scan_rows_per_label':native.MAX_CONTOUR_SCAN_ROWS,
            'ellipse_hull_point_limit':native.MAX_CONTOUR_AGGREGATE_POINTS,
            'max_numerical_span_work_per_helper':native.MAX_RASTER_SPAN_WORK,
            'max_numerical_span_segments_per_helper':native.MAX_RASTER_SPAN_SEGMENTS,
            'max_diagnostic_geometry_work_aggregate':native.MAX_RASTER_SPAN_WORK,
            'max_diagnostic_points_aggregate':native.MAX_SIZE_DIAGNOSTIC_POINTS,
            'max_diagnostic_contours_aggregate':native.MAX_CONTOUR_COMPONENTS,
            'max_diagnostic_pixel_writes':native.MAX_SIZE_DIAGNOSTIC_PIXELS,
            'max_diagnostic_span_items':native.MAX_SIZE_DIAGNOSTIC_SPAN_ITEMS,
            'max_diagnostic_scratch_bytes':native.MAX_SIZE_DIAGNOSTIC_SCRATCH_BYTES,
            'compact_host_scope':'one bounded per-label topology plus bounded retained drawing geometry and observation metadata',
            'max_metadata_bytes_per_product':size_contract.MAX_METADATA_BYTES,
            'max_metadata_items_per_product':size_contract.MAX_METADATA_ITEMS}
    elif operation == 'analysis.color':
        color_contract.controls(parameters,archives.NativeResultExportError)
        count=parameters['n_labels'];tile=min(pixels,runner._color.TILE_PIXELS)
        tiles=(pixels+runner._color.TILE_PIXELS-1)//runner._color.TILE_PIXELS
        #7P uploaded RGB+labels,3P BGR adapter; all three channel spaces are
        # calculated by the pinned primitive even when one space is selected.
        scratch=512*tile+32768*(count+1)+4*1024*1024
        cuda_bytes=max(cuda_bytes,10*pixels+scratch)
        work=128*pixels+32*tiles*9*(count+1)*256
        work_details={'model':'declared scan/channel/histogram visits; not instructions or elapsed time',
            'tile_pixels':tile,'live_raster_bytes':10*pixels,'tile_and_table_scratch_bytes':scratch,
            'n_labels':count,'histogram_spaces_computed':3,
            'max_metadata_bytes_per_product':color_contract.MAX_METADATA_BYTES,
            'max_metadata_items_per_product':color_contract.MAX_METADATA_ITEMS}
    elif operation == 'threshold.custom_range':
        try:
            runner._range._range_controls(**parameters)
        except (ValueError,TypeError,KeyError) as error:
            raise archives.NativeResultExportError(f'Invalid custom-range controls: {error}') from error
        try:
            admission = runner._range._range_admission(pixels)
        except runner._range.RangeResourceError as error:
            raise NativeStageReplayUnavailable(str(error)) from error
        cuda_bytes = runner._range_recipe.retained_bound(pixels)
        work_details = {'model':'baseline replay admission units; not instructions or elapsed time',
            'baseline_units_per_pixel':256, 'live_raster_bytes':17*pixels, **admission}
    elif operation in EXTENDED_THRESHOLD_OPERATIONS:
        admission = _extended_threshold_admission(pixels,operation,parameters)
        # Inputs + expected mask + fresh mask; dual also owns a BGR copy.
        raster_bytes = (8 if operation=='threshold.dual_channels' else 3)*pixels
        cuda_bytes = max(cuda_bytes,raster_bytes+admission['tile_scratch_bytes']
            +admission['fixed_scratch_bytes']+4*1024*1024)
        work_details = {'model':'baseline replay admission units; primitive visits/limits recorded separately; no instruction or elapsed-time estimate',
            'baseline_units_per_pixel':256, 'live_raster_bytes':raster_bytes, **admission}
    elif diagnostic:
        #13P covers uploaded RGB/labels/expected RGB and the fresh RGB canvas.
        # Reserve contour CUDA scratch and tiled bbox scratch together, even
        # though they are separate phases. Keep the old envelope as a floor.
        cuda_bytes = max(cuda_bytes,13*pixels+256*1024*1024
            +64*min(pixels,runner._watershed_diagnostic.MAX_TILE_PIXELS)+3*1024*1024)
        work = None
        drawing = runner._watershed_diagnostic
        work_details = {'model':'native diagnostic operation-specific hard limits; no elapsed-time guarantee',
            'max_scan_work':drawing.MAX_SCAN_WORK,'max_topology_work':drawing.size.MAX_CONTOUR_WORK_UNITS,
            'max_geometry_work':drawing.MAX_GEOMETRY_WORK,'max_host_geometry_bytes':drawing.MAX_CPU_GEOMETRY_BYTES,
            'max_paint_scratch_bytes':96*1024*1024,'max_metadata_bytes':diagnostic_contract.MAX_METADATA_BYTES,
            'max_metadata_items':diagnostic_contract.MAX_METADATA_ITEMS,'live_raster_bytes':13*pixels}
    elif operation in WATERSHED_TYPED_OPERATIONS:
        # These primitives have their own bounded candidates/rounds/queue or
        # pointer-jump admission. Do not pretend the legacy tensor-visit model
        # estimates data-dependent pooling, sorting or CUDA heap instructions.
        work = None
        work_details = {'model':'native operation-specific hard limits; no elapsed-time guarantee'}
        if operation.endswith('local_maxima'):
            distance = parameters['distance']
            if type(distance) is not int or not 1<=distance<=runner._peaks.MAX_DISTANCE:
                raise runner.NativeRecipeError('Selected peak distance exceeds native admission.')
            work_details.update(max_candidates=runner._peaks.MAX_CANDIDATES,
                max_nms_rounds=runner._peaks.MAX_NMS_ROUNDS,max_distance=runner._peaks.MAX_DISTANCE,
                max_nms_sample_visits=runner._peaks.MAX_NMS_SAMPLE_VISITS,
                nms_admission='actual ROI/candidates determine affordable rounds before NMS allocation',
                primitive_tensor_workspace_bytes=runner._peaks.MAX_ESTIMATED_TENSOR_WORKSPACE_BYTES)
        elif operation.endswith('connected_markers'):
            work_details.update(max_marker_pixels=runner.MAX_MARKER_PIXELS,
                max_union_passes=40,pointer_jumps_per_pass=4,final_pointer_jumps=8,
                compact_statistics='at most one row per bounded marker component')
        elif operation.endswith('masked_flood'):
            work_details.update(max_foreground_queue=pixels,queue_entry_bytes=12,
                native_output_and_queue_bytes=16*pixels+16,
                native_workspace_limit_bytes=runner._watershed.MAX_WORKSPACE_BYTES,
                kernel_batch_pixels=runner._watershed.KERNEL_BATCH_PIXELS,
                queue_rule='each foreground pixel is inserted at most once')
            if 16*pixels+16>runner._watershed.MAX_WORKSPACE_BYTES:
                raise runner.NativeRecipeError('Selected flood exceeds its native queue workspace.')
        elif operation.endswith('marker_count_observation'):
            work_details['calculation'] = 'CUDA unique on one int32 marker raster; compact scalar observation'
        else:
            work_details['calculation'] = 'one float32 negation pass'
    elif operation == 'threshold.mean':
        ksize, _ = runner._threshold._validate_mean_parameters(parameters['ksize'],parameters['offset'])
        runner._threshold._object_type(parameters['object_type'])
        work_details.update(requested_kernel_size=parameters['ksize'],effective_kernel_size=ksize,
            border='replicate', calculation='exact integer box sums with scalar/SIMD decision certification')
    elif operation == 'filter.median_blur':
        kernel = parameters['ksize']
        kernel = tuple(kernel) if isinstance(kernel,list) else kernel
        kh, kw = runner._median._kernel_shape(kernel)
        roi = runner._median_roi_recipe.selected_bounds(parameters.get('roi_bounds_xyxy'), shape)
        crop_shape = shape if roi is None else (roi[3]-roi[1], roi[2]-roi[0])
        runner._median._validate_scipy_reflect_geometry(kh, kw, *crop_shape)
        crop_pixels = crop_shape[0]*crop_shape[1]
        window_values = crop_pixels*kh*kw
        work = window_values + (16*pixels if roi is not None else 0)
        work_details.update(kernel_shape=[kh,kw], window_values=window_values)
        if roi is not None:
            work_details.update(roi_bounds_xyxy=roi, roi_pixels=crop_pixels,
                output_frame='input full canvas', roi_rule='crop/filter/replace; exterior preserved',
                full_canvas_memory_floor=True)
    elif operation in ('morphology.erode','morphology.dilate'):
        roi = runner._morphology_roi_recipe.selected_bounds(parameters.get('roi_bounds_xyxy'), shape)
        roi_pixels = pixels if roi is None else (roi[2]-roi[0])*(roi[3]-roi[1])
        work = roi_pixels*parameters['ksize']*parameters['iterations']*2
        if roi is not None:
            work += pixels
            work_details.update(roi_bounds_xyxy=roi, roi_pixels=roi_pixels,
                output_frame='input full canvas', roi_rule='crop/filter/replace; exterior preserved')
        work_details.update(axis_passes=2*parameters['iterations'], kernel_size=parameters['ksize'])
    elif operation == 'morphology.fill':
        roi = runner._fill_roi_recipe.selected_bounds(parameters.get('roi_bounds_xyxy'), shape)
        crop_shape = shape if roi is None else (roi[3]-roi[1], roi[2]-roi[0])
        crop_pixels = crop_shape[0]*crop_shape[1]
        if type(parameters['size']) is not int or parameters['size'] < 1:
            raise runner.NativeRecipeError('Fill size must be a positive integer.')
        if parameters['size'] == 1 or parameters['size'] > crop_pixels:
            work = 16*crop_pixels
            work_details['area_shortcut'] = True
        else:
            # Every possible edge batch can invoke a complete parent-raster
            # compression on each union pass, including both directions.
            # Four visits cover the pointer gather and equality reduction;
            # 32 cover each edge's gathers, comparisons, hooks and indexing.
            horizontal = crop_shape[0]*max(0,crop_shape[1]-1)
            vertical = max(0,crop_shape[0]-1)*crop_shape[1]
            chunk = runner._fill.MAX_EDGE_CHUNK
            batches = sum((edges+chunk-1)//chunk for edges in (horizontal,vertical))
            compress = 4*crop_pixels*runner._fill.MAX_ROOT_JUMPS
            work = (128*crop_pixels+compress + runner._fill.MAX_UNION_PASSES*
                    (32*(horizontal+vertical)+batches*compress))
            work_details.update(edge_batches=batches, possible_edges=horizontal+vertical,
                union_passes_per_batch=runner._fill.MAX_UNION_PASSES,
                root_jumps_per_compression=runner._fill.MAX_ROOT_JUMPS,
                full_parent_compressions=batches*runner._fill.MAX_UNION_PASSES+1)
        if roi is not None:
            # Full-frame binary validation, clone/replace and comparison remain
            # admitted even when the union/find working crop is tiny.
            work += 16*pixels
            work_details.update(roi_bounds_xyxy=roi, roi_pixels=crop_pixels,
                output_frame='input full canvas', roi_rule='crop/filter/replace; exterior preserved',
                connectivity=4, area_rule='remove strictly smaller than size')
    elif operation == 'reference.internal.watershed.distance_map':
        # Exactly two row sweeps: eight-neighbor construction/reduction,
        # bounded prefix scan, row carry and flips. Charge scan depth and
        # per-chunk setup separately, including very tall/narrow rasters.
        chunk = runner._distance.SCAN_CHUNK_PIXELS
        depth = (min(shape[1],chunk)-1).bit_length()
        scans = 2*shape[0]*((shape[1]+chunk-1)//chunk)
        work = pixels*(128+8*depth)+512*scans+4*chunk
        work_details.update(sweeps=2, neighbors_per_sweep=8, scan_depth=depth,
            scan_chunks=scans, scan_chunk_limit=MAX_DISTANCE_SCAN_CHUNKS)
        if scans > MAX_DISTANCE_SCAN_CHUNKS:
            raise runner.NativeRecipeError('Selected distance stage exceeds replay row-scan admission.')
    if (selected_bytes > selected_limit or cuda_bytes > MAX_REPLAY_CUDA_BYTES
            or work is not None and work > MAX_REPLAY_WORK):
        if operation in (*EXTENDED_THRESHOLD_OPERATIONS, 'threshold.custom_range','analysis.size','calibrated_seed.size_units'):
            raise NativeStageReplayUnavailable('Selected operation exceeds replay input/CUDA/work admission.')
        raise runner.NativeRecipeError('Selected stage exceeds replay input/CUDA/work admission.')
    # Host execution retains <=48/128MiB selected bytes, at most one selected-copy,
    # a <=256MiB compressed snapshot, plus streaming and admitted JSON metadata.
    return {'pixels':pixels, 'selected_bytes':selected_bytes,
        'cuda_bytes':cuda_bytes, 'work':work, 'work_details':work_details,
        'selected_limit_bytes':selected_limit,
        'cuda_limit_bytes':MAX_REPLAY_CUDA_BYTES, 'work_limit':None if work is None else MAX_REPLAY_WORK,
        'compressed_snapshot_limit_bytes':ARCHIVE_LIMITS.max_npz_bytes,
        'host_scope':'snapshot + selected payloads + one selected copy + bounded metadata/chunks'}


def _torch_dtype(dtype):
    return {'|u1':torch.uint8,'|b1':torch.bool,'<f4':torch.float32,'<f8':torch.float64,
            '<i4':torch.int32,'<i8':torch.int64}[dtype]


def _validate_card_product(value,role,frames,device,poll,error_type,owned_storage):
    if role not in card_contract.RASTER_ROLES:
        try:card_contract.validate(value,role,frames.source.shape,frames.bounds,poll=poll)
        except card_contract.NativeCardReplayError as error:raise error_type(str(error)) from error
        if role=='bounds' and tuple(value)!=frames.bounds:
            raise error_type('Card bounds differ from the declared ROI control.')
        return
    shape,dtype=card_contract.raster_spec(role,frames.source.shape,frames.bounds)
    if (not torch.is_tensor(value) or value.layout!=torch.strided or value.requires_grad
            or value.device!=device or value.dtype!=_torch_dtype(dtype)
            or tuple(value.shape)!=shape or not value.is_contiguous()):
        raise error_type('Card raster violates its declared frame/device/dtype/layout.')
    storage=value.untyped_storage()
    if storage.data_ptr() not in owned_storage and storage.nbytes()>card_contract._rounded(value.numel()*value.element_size()):
        raise error_type('Card raster retains unadmitted backing storage.')
    flat=value.view(-1)
    if role in ('chip_labels','source_binary','roi_binary'):
        for start in range(0,value.numel(),COMPARE_ELEMENTS):
            poll();tile=flat[start:start+COMPARE_ELEMENTS]
            if role=='chip_labels':
                valid=torch.isfinite(tile)&((tile==0)|((tile>=10)&(tile<=240)&(torch.remainder(tile,10)==0)))
            else:valid=(tile==0)|(tile==255)
            if not bool(valid.all().item()):
                raise error_type('Chip labels require finite IDs0/10..240; binary masks require0/255.')
            del tile,valid


def _validate_product(value, data_type, frame, device, poll, error_type, *, label=None, parameters=None,
                      metadata_role=None, size_parameters=None, size_unit='pixels',
                      model_kind=None, model_identity=None, model_count=None,
                      card_role=None,card_frames=None,owned_storage=None,
                      recovery_role=None,recovery_binding=None,paper_role=None,paper_context=None,paper_name=None,crop_role=None,crop_context=None):
    if crop_role is not None:
        return _validate_crop_product(value,crop_role,crop_context,device,poll,error_type,owned_storage)
    if paper_role is not None:
        return _validate_paper_product(value,paper_role,frame,paper_context,paper_name,device,poll,error_type,owned_storage)
    if recovery_role is not None:
        if recovery_role in ('partition','annotation','annotation_metadata_only'):
            try:
                keys=('tensors','metadata') if recovery_role=='partition' else (
                    ('metadata',) if recovery_role=='annotation_metadata_only' else ('uncertainty_band','metadata'))
                recovery_contract._keys(value,keys,'Recovery transport')
                if recovery_role=='partition':
                    recovery_contract._keys(value['tensors'],recovery_contract.PARTITION_TENSORS,'Partition tensors')
                    rasters=value['tensors'];roles=recovery_contract.PARTITION_TENSORS
                else:
                    rasters={} if recovery_role=='annotation_metadata_only' else {'uncertainty_band':value['uncertainty_band']}
                    roles={'uncertainty_band':'unit_binary'}
                for name,raster in rasters.items():
                    _validate_product(raster,data_type,frame,device,poll,error_type,
                        recovery_role=roles[name],recovery_binding=recovery_binding,owned_storage=owned_storage)
                recovery_contract.validate(recovery_contract.canonical(value['metadata'],poll),
                    'partition_metadata' if recovery_role=='partition' else 'annotation_metadata',
                    frame,recovery_binding,{} if recovery_role=='partition' else parameters,poll)
                if recovery_role=='partition':
                    # This is bounded compact ID metadata, not a dense raster transfer.
                    ids=tuple(rasters['label_ids'].cpu().tolist())
                    if ids!=tuple(row['label_id'] for row in value['metadata']['objects']):
                        raise recovery_contract.NativeRecoveryReplayError('Partition ID vector differs from its ordered records.')
            except recovery_contract.NativeRecoveryReplayError as error:raise error_type(str(error)) from error
            return
        if recovery_role not in recovery_contract.RASTER_ROLES:
            try:recovery_contract.validate(recovery_contract.canonical(value,poll),recovery_role,
                frame,recovery_binding,parameters,poll)
            except recovery_contract.NativeRecoveryReplayError as error:raise error_type(str(error)) from error
            return
        dtype=torch.int32 if recovery_role in ('markers','ids') else torch.float32 if recovery_role=='elevation' else torch.uint8
        shape_ok=(torch.is_tensor(value) and value.ndim==1 and value.numel()<=runner._recovery_recipe.core.MAX_LABELS
            if recovery_role=='ids' else torch.is_tensor(value) and tuple(value.shape)==frame.shape)
        if (not torch.is_tensor(value) or value.layout!=torch.strided or value.requires_grad
                or value.device!=device or value.dtype!=dtype or not shape_ok
                or not value.is_contiguous()):
            raise error_type('Recovery raster violates its full-frame/device/dtype/layout contract.')
        storage=value.untyped_storage()
        if storage.data_ptr() not in (owned_storage or ()) and storage.nbytes()>recovery_contract._rounded(value.numel()*value.element_size()):
            raise error_type('Recovery raster retains unadmitted backing storage.')
        flat=value.view(-1)
        for start in range(0,value.numel(),COMPARE_ELEMENTS):
            poll();tile=flat[start:start+COMPARE_ELEMENTS]
            if recovery_role=='elevation':valid=torch.isfinite(tile)
            elif recovery_role in ('markers','ids'):
                valid=(tile>0 if recovery_role=='ids' else tile>=0)&(tile<=math.prod(frame.shape))
                if recovery_role=='ids' and tile.numel()>1 and not bool((tile[1:]>tile[:-1]).all().item()):
                    raise error_type('Recovery IDs require strictly ascending positive source-pixel IDs.')
            else:valid=(tile==0)|(tile==(1 if recovery_role=='unit_binary' else 255))
            if not bool(valid.all().item()):
                raise error_type('Recovery values violate the declared ID, finite elevation or binary mask encoding.')
            del tile,valid
        return
    if card_role is not None:
        return _validate_card_product(value,card_role,card_frames,device,poll,error_type,owned_storage or ())
    kind,dtype = PRODUCT_CONTRACTS[data_type]
    if kind in ('model','model_identity','class_names','class_masks'):
        try:
            if kind=='model':
                (runner._naive_bayes._validate_model if model_kind=='bayes' else runner._kmeans_model.validate_kmeans_model)(value)
                model_contract._model_matches(model_identity,value,model_kind)
            elif kind=='model_identity':
                if model_contract.identity(value,model_kind)!=model_identity:
                    raise model_contract.NativeModelReplayError('Model identity differs from its archived asset.')
            elif kind=='class_names':
                names=model_contract.class_names(value,runner._naive_bayes.MAX_CLASSES if model_kind=='bayes' else runner._kmeans_model.MAX_CLUSTERS)
                if len(names)!=model_count:raise model_contract.NativeModelReplayError('Class-name count differs from its model operation.')
                if error_type is runner.NativeRecipeError:
                    source_names=(model_identity['class_names'] if model_kind=='bayes'
                                  else model_contract.mask_names(parameters,model_identity['cluster_count'])[0])
                    if names!=source_names:raise model_contract.NativeModelReplayError('Fresh class names differ from the model/selection order.')
            else:
                if not isinstance(value,Mapping) or len(value)!=model_count:
                    raise model_contract.NativeModelReplayError('Class masks require the complete ordered mapping.')
                model_contract.class_names(tuple(value),runner._naive_bayes.MAX_CLASSES if model_kind=='bayes' else runner._kmeans_model.MAX_CLUSTERS)
                for raster in value.values():
                    _validate_product(raster,'BinaryMask' if error_type is runner.NativeRecipeError else 'GrayImage',
                        frame,device,poll,error_type)
        except ValueError as error:
            raise error_type(str(error)) from error
        return
    if metadata_role is not None or kind=='size_observations':
        value=size_contract.canonical(value,error_type,poll)
        if kind=='size_observations':size_contract.observations(value,size_parameters or parameters,error_type,size_unit)
        elif metadata_role=='size_scale':size_contract.scale(value,error_type)
        elif metadata_role=='size_units':size_contract.metadata(value,error_type)
        else:raise error_type('Unknown size metadata role.')
        return
    if kind in ('color_observations','color_chart'):
        color_contract.validate(color_contract.canonical(value),
            'observations' if kind=='color_observations' else 'histogram',parameters,frame.shape[0]*frame.shape[1],error_type)
        return
    if kind=='diagnostic_metadata':
        diagnostic_contract.validate(value,parameters,frame.shape[0]*frame.shape[1],error_type)
        return
    if kind=='token':
        if value is not True:
            raise error_type('Flood completion must be the exact completed token.')
        return
    if kind=='observation':
        _observation(value,label,frame.shape[0]*frame.shape[1],error_type)
        return
    if (not torch.is_tensor(value) or value.device!=device or value.dtype!=_torch_dtype(dtype)
            or not value.is_contiguous()):
        raise error_type('Replayed product violates its device/dtype/layout contract.')
    shape = tuple(value.shape)
    if kind=='coordinates':
        if len(shape)!=2 or shape[1]!=2 or not 0<=shape[0]<=runner._peaks.MAX_CANDIDATES:
            raise error_type('Replayed coordinates violate their Kx2 candidate contract.')
        for start in range(0,shape[0],COMPARE_ELEMENTS):
            poll();tile=value[start:start+COMPARE_ELEMENTS]
            if not bool(((tile[:,0]>=0)&(tile[:,0]<frame.shape[0])
                    &(tile[:,1]>=0)&(tile[:,1]<frame.shape[1])).all().item()):
                raise error_type('Peak coordinates lie outside their resolved row/column frame.')
        return
    if shape!=((*frame.shape,3) if kind=='rgb' else frame.shape):
        raise error_type('Replayed raster violates its resolved coordinate frame.')
    flat = value.view(-1)
    for start in range(0,value.numel(),COMPARE_ELEMENTS):
        poll();tile=flat[start:start+COMPARE_ELEMENTS]
        if dtype=='<f4' and not bool(torch.isfinite(tile).all().item()):
            raise error_type('Replayed float raster must be finite.')
        if data_type=='BinaryMask' and not bool(((tile==0)|(tile==255)).all().item()):
            raise error_type('Prepared foreground must contain only zero and255.')
        if data_type in ('MarkerLabels','InstanceLabels') and not bool(((tile>=0)&(tile<=runner.MAX_MARKER_PIXELS)).all().item()):
            raise error_type('Marker or instance IDs exceed their bounded nonnegative domain.')
        if data_type=='LabelRaster' and not bool((tile<model_identity['cluster_count']).all().item()):
            raise error_type('Predicted cluster IDs exceed the bound model domain.')


def _validate_relationships(operation, products, inputs, frame, poll, error_type):
    if operation=='recovery.partition':
        transport=products['partition'];metadata=transport['metadata']
        if (transport['tensors']['labels'] is not products['labels'] or metadata!=products['metadata']
                or metadata['inventory']!=inputs['inventory'] or metadata['binding']!=inputs['source_frame']
                or any(transport['tensors'][field] is not inputs[port] for field,port in
                    (('original_foreground','foreground'),('markers','markers'),('validity','validity'),('elevation','elevation')))):
            raise error_type('Partition retained input/output aliases or metadata bindings disagree.')
    elif operation=='recovery.annotation':
        transport=products['annotation']
        if transport['uncertainty_band'] is not products['uncertainty_band'] or transport['metadata']!=products['metadata']:
            raise error_type('Annotation inner/separate band or metadata aliases disagree.')
    elif operation=='recovery.measurements':
        metadata=inputs['annotation']['metadata']
        if products['metadata']!=metadata:
            raise error_type('Measurements must retain their selected annotation metadata.')
        if error_type is runner.NativeRecipeError:
            rows=metadata['objects'];observations=products['observations']
            if tuple(observations)!=tuple('core_'+str(row['label_id']) for row in rows):
                raise error_type('Fresh measurements differ from the annotation inventory order.')
            for row in rows:
                poll();allowed=row['measurement_permitted']
                values=(row['assigned_pixels'],row['bbox'][2]-row['bbox'][0],row['bbox'][3]-row['bbox'][1]) if allowed else (None,None,None)
                for name,value in zip(recovery_contract.TRAITS,values):
                    record=observations['core_'+str(row['label_id'])][name]
                    if record['value']!=value or record['available']!=allowed or record['reasons']!=row['reasons']:
                        raise error_type('Fresh measurement values or eligibility differ from annotation facts.')
    elif operation in MODEL_OPERATIONS:
        if products['model_identity']!=inputs['model_identity']:
            raise error_type('Returned model identity differs from the selected input.')
        if 'class_masks' in products:
            if tuple(products['class_masks'])!=tuple(products['class_names']):
                raise error_type('Class-mask key order differs from the returned names.')
    elif operation == 'threshold.custom_range' and error_type is runner.NativeRecipeError:
        # Well-typed historical mistakes are compared as mismatches. Only a
        # fresh calculation must satisfy the operation's masking relationship.
        mask, image, source = products['mask'].view(-1), products['masked_rgb'].view(-1,3), inputs['rgb'].view(-1,3)
        for start in range(0,mask.numel(),COMPARE_ELEMENTS):
            poll();end=start+COMPARE_ELEMENTS
            expected = torch.where(mask[start:end,None]!=0,source[start:end],0)
            if not torch.equal(image[start:end],expected):
                raise error_type('Fresh range masked RGB differs from original pixels and the foreground mask.')
    elif operation.endswith('local_maxima'):
        coordinates,peaks = products['coordinates'],products['peaks']
        count = int(coordinates.shape[0])
        poll()
        if int(torch.count_nonzero(peaks).item())!=count:
            raise error_type('Peak coordinate count differs from its painted boolean mask.')
        if count:
            ids = coordinates[:,0]*frame.shape[1]+coordinates[:,1]
            if int(torch.unique(ids).numel())!=count or not bool(peaks[coordinates[:,0],coordinates[:,1]].all().item()):
                raise error_type('Peak coordinates must be unique and paint their exact mask.')
            intensities = inputs['distance'][coordinates[:,0],coordinates[:,1]]
            if count>1 and not bool(((intensities[:-1]>intensities[1:])
                    |((intensities[:-1]==intensities[1:])&(ids[:-1]<ids[1:]))).all().item()):
                raise error_type('Peak coordinates violate descending distance/stable row-major order.')
    elif operation.endswith('masked_flood'):
        labels,markers,mask = (products['labels'].view(-1),inputs['markers'].view(-1),inputs['mask'].view(-1))
        present = torch.zeros(runner.MAX_MARKER_PIXELS+1,dtype=torch.bool,device=markers.device)
        present[0] = True
        for start in range(0,labels.numel(),COMPARE_ELEMENTS):
            poll();end=start+COMPARE_ELEMENTS
            values,seeds,foreground = labels[start:end],markers[start:end],mask[start:end]!=0
            if bool((values[~foreground]!=0).any().item()):
                raise error_type('Watershed output labels extend outside the archived foreground.')
            selected = foreground&(seeds!=0)
            if not bool((values[selected]==seeds[selected]).all().item()):
                raise error_type('Watershed output does not preserve its archived in-mask seeds.')
            present[seeds[selected].to(torch.int64)] = True
        for start in range(0,labels.numel(),COMPARE_ELEMENTS):
            poll()
            if not bool(present[labels[start:start+COMPARE_ELEMENTS].to(torch.int64)].all().item()):
                raise error_type('Watershed output contains an ID absent from its archived in-mask seeds.')


def _compare_product(actual, expected, poll):
    if type(actual) in tuple(color_contract.RECORDS.values()):
        return color_contract.compare(color_contract.canonical(actual),expected,poll)
    if isinstance(actual,Mapping) and actual.get('kind')=='native_watershed_debug_diagnostic':
        return diagnostic_contract.compare(actual,expected,poll)
    if not torch.is_tensor(actual):
        poll()
        mismatch = int(actual!=expected)
        return {'mismatched_elements':mismatch,'compared_elements':1,'shape_equal':True,'kind':'compact'}
    common = min(actual.numel(),expected.numel())
    mismatch = abs(actual.numel()-expected.numel())
    for start in range(0,common,COMPARE_ELEMENTS):
        poll();end=min(common,start+COMPARE_ELEMENTS)
        mismatch += int(torch.count_nonzero(actual.view(-1)[start:end]!=expected.view(-1)[start:end]).item())
    return {'mismatched_elements':mismatch,'compared_elements':max(actual.numel(),expected.numel()),
        'shape_equal':tuple(actual.shape)==tuple(expected.shape),'kind':'tensor'}


def _compare_model_product(actual, expected, poll):
    if isinstance(actual,Mapping) and actual and all(torch.is_tensor(value) for value in actual.values()):
        rows=[]
        for (name,raster),(expected_name,expected_raster) in zip(actual.items(),expected.items()):
            comparison=_compare_product(raster,expected_raster,poll)
            comparison.update(actual_class=name,expected_class=expected_name)
            comparison['mismatched_elements']+=int(name!=expected_name)
            comparison['compared_elements']+=1
            rows.append(comparison)
        return {'kind':'ordered_cuda_class_masks','by_class':rows,'shape_equal':True,
            'mismatched_elements':sum(row['mismatched_elements'] for row in rows),
            'compared_elements':sum(row['compared_elements'] for row in rows)}
    if torch.is_tensor(actual):return _compare_product(actual,expected,poll)
    return model_contract.compare(actual,expected,poll)


def replay_native_stage(directory, stage_id, *, device='cuda', cancel=None,
                        expected_revision=None, expected_recipe_identity=None):
    """Verify an archive and replay every output of one exact selected stage.

    Dense inputs, expected products, calculation and comparisons stay CUDA.
    Compact tokens/observations remain immutable metadata. History is untouched.
    """
    if type(stage_id) is not str or not stage_id:
        raise TypeError('Select an explicit stage ID.')
    if cancel is not None and not callable(cancel):
        raise TypeError('cancel must be callable or None.')
    watched, nested_watched, recovery_watched, facts = [],[],[],{}
    calculation_watched = []
    crop_context=crop_context_baseline=None

    def intact(*,compact=True):
        if any(runner._compact_recovery_signature(value) != signature for value,signature in calculation_watched):
            raise runner.NativeRecipeError("Replay Otsu cutoff facts changed during execution.")
        if crop_context is not None and runner._card_recipe.correction._metadata_signature(crop_context)!=crop_context_baseline:
            raise runner.NativeRecipeError('Replay crop frame context changed during execution.')
        if any(runner._tensor_signature(value)!=signature for value,signature in watched):
            raise runner.NativeRecipeError('Replay input or output changed during execution.')
        if any(_model_signature(value)!=signature for value,signature in nested_watched):
            raise runner.NativeRecipeError('Replay model contents or nested outputs changed during execution.')
        if compact and any(runner._compact_recovery_signature(value)!=signature for value,signature in recovery_watched):
            raise runner.NativeRecipeError('Replay recovery contents or nested outputs changed during execution.')

    def poll():
        # Large immutable inventories need full checks around caller callbacks
        # and at publication, not around each callback-free kernel checkpoint.
        intact(compact=cancel is not None)
        try:
            if cancel is not None:
                try:stopped = cancel()
                except runner.NativeRecipeCancelled:raise
                except Exception as error:raise _ReplayCallbackError(f'Callback failed: {type(error).__name__}: {error}') from error
                if type(stopped) is not bool:
                    raise TypeError('Replay cancellation must return bool.')
                if stopped:
                    raise runner.NativeRecipeCancelled('Stage replay cancelled before completion.')
        finally:
            intact(compact=cancel is not None)
        return False

    try:
        poll()
        report,specs = archives.read_native_recipe_inventory(directory,limits=ARCHIVE_LIMITS,cancel=poll)
        recipe = RecipeSpec.from_json(archives._json_bytes(report['recipe']))
        if (expected_revision is not None and report['result_revision']!=expected_revision
                or expected_recipe_identity is not None and recipe.identity.analytical_dict()!=
                    expected_recipe_identity.analytical_dict()):
            raise archives.NativeResultExportError('Saved run changed from the exact selected recipe/archive.')
        try:
            validation_recipe = recipe
            if recipe.provenance.mode.value == 'enhanced' and 'workspace_workflow_adapter' in report['runtime']:
                from seedvision.pipeline.workflows import _working_provenance
                canonical = _canonical_replay_source(recipe)
                if recipe.provenance != _working_provenance(canonical):
                    raise runner.NativeRecipeError('Working replay provenance differs from its registered source.')
                # ROI is recorded separately; this still rejects altered graph/port/parameter contracts.
                validation_recipe = replace(recipe, provenance=canonical.provenance)
            runner._validate_recipe(validation_recipe)
            runner._validate_preparation_mode(recipe, report['preparation'])
        except runner.NativeRecipeError as error:
            if (recipe.recipe_id in runner._NAIVE_BAYES_RECIPES + runner._KMEANS_RECIPES
                    or recipe.recipe_id==runner._card_recipe.RECIPE_ID
                    or recipe.recipe_id==runner._recovery_recipe.RECIPE_ID
                    or recipe.recipe_id==runner._paper_recipe.RECIPE_ID
                    or runner._diagnostic_recipe.base_recipe_id(recipe.recipe_id) in runner._calibrated_seed.RECIPE_IDS):
                raise archives.NativeResultExportError(f'Replay recipe verification failed: {error}') from error
            raise
        stage = recipe.stage(stage_id)
        if stage.operation_id not in SUPPORTED_REPLAY_OPERATIONS:
            return _result('unsupported',stage_id,'No saved-input replay contract for this operation.')
        node = next(row for row in report['nodes'] if row['stage_id']==stage_id)
        facts = {'stage_id':stage_id,'archive_revision':report['result_revision'],
            'recipe_identity':report['recipe_identity'],'source_sha256':report['source_sha256'],
            'coordinate_frame':report['coordinate_frame'],'operation_id':stage.operation_id,
            'recorded_runtime':report['runtime'],'scientifically_validated':False,
            'reference_conformance':'not_evaluated_by_replay'}
        if node.get('backend') == 'plantcv':
            return _result('unavailable',stage_id,
                'This stage was executed by PlantCV. Native replay cannot verify the public call.',
                metadata=facts | {'recorded_backend': 'plantcv', 'backend_provenance': node['backend_provenance']})
        if node['status'] not in ('executed','empty','partial'):
            return _result('unavailable',stage_id,'Selected historical products are unavailable.',metadata=facts)
        input_ports,output_ports,preview = SUPPORTED_REPLAY_OPERATIONS[stage.operation_id]
        if ({port.name for port in stage.ports if port.direction=='input'}!=set(input_ports)
                or {port.name for port in stage.ports if port.direction=='output'}!=set(output_ports)):
            raise archives.NativeResultExportError('Selected ports differ from the typed replay contract.')
        incoming = [edge for edge in recipe.dependencies if edge.target_stage==stage_id]
        if {edge.target_port for edge in incoming}!=set(input_ports) or set(node['input_keys'])!=set(input_ports):
            raise archives.NativeResultExportError('Selected input lineage differs from the exact recipe edges.')
        nodes = {row['stage_id']:row for row in report['nodes']}
        input_bindings,lineage = {},{}
        for edge in incoming:
            producer = nodes[edge.source_stage]
            if (producer['product_key'] is None or node['input_keys'][edge.target_port]!=
                    runner._digest([producer['product_key'],edge.source_port])):
                raise archives.NativeResultExportError('Selected input key does not bind its declared producer port.')
            binding = dict(producer['outputs']).get(edge.source_port)
            if binding is None:
                return _result('unavailable',stage_id,'An exact archived input is absent.',metadata=facts)
            input_bindings[edge.target_port] = binding
            lineage[edge.target_port] = {'producer_stage':edge.source_stage,'producer_port':edge.source_port,
                'producer_product_key':producer['product_key'],'input_key':node['input_keys'][edge.target_port]}
        output_bindings = dict(node['outputs'])
        size_operation=stage.operation_id in ('analysis.size','calibrated_seed.size_units')
        units_operation=stage.operation_id=='calibrated_seed.size_units'
        model_operation=stage.operation_id in MODEL_OPERATIONS
        card_operation=stage.operation_id in card_contract.PORTS
        recovery_operation=stage.operation_id in recovery_contract.PORTS
        paper_operation=stage.operation_id in paper_contract.PORTS
        crop_operation=stage.operation_id==crop_contract.OPERATION
        crop_context=crop_identity=crop_state=None
        if crop_operation:
            crop_identity=runner._digest(runner._module_record(crop_contract))
            crop_state=crop_contract.state()
            if crop_state!=crop_contract.PINNED_STATE:
                raise runner.NativeRecipeError('Loaded crop replay contract differs from its pinned state.')
        paper_context=paper_identity=paper_state=None
        if paper_operation:
            paper_identity=runner._digest(runner._module_record(paper_contract))
            paper_state=paper_contract.state()
            if paper_state!=paper_contract.PINNED_STATE:
                raise runner.NativeRecipeError("Loaded paper replay contract differs from its pinned state.")
        recovery_binding=recovery_identity=recovery_state=None
        if recovery_operation:
            recovery_identity=runner._digest(runner._module_record(recovery_contract))
            recovery_state=recovery_contract.state()
            if recovery_state!=recovery_contract.PINNED_STATE:
                raise runner.NativeRecipeError('Loaded recovery replay contract differs from its pinned state.')
        card_frames=card_identity=card_state=None
        if card_operation or crop_operation:
            card_identity=runner._digest(runner._module_record(card_contract))
            card_state=card_contract.state()
            if card_state!=card_contract.PINNED_STATE:
                raise runner.NativeRecipeError('Loaded card replay contract differs from its pinned state.')
        model_asset=parsed_model=model_identity=model_state=None
        if model_operation:
            model_identity=runner._digest(runner._module_record(model_contract))
            model_state=model_contract.state()
            if model_state!=model_contract.PINNED_STATE:
                raise runner.NativeRecipeError('Loaded model replay contract differs from its pinned state.')
        required_outputs=('observations',) if stage.operation_id=='analysis.size' else output_ports
        if any(port not in output_bindings for port in required_outputs):
            return _result('unavailable',stage_id,'An exact archived output is absent.',metadata=facts)
        selected_output_ports=tuple(name for name in output_ports if name in output_bindings)
        try:
            if crop_operation:frame,frame_facts,crop_context=_crop_frame(recipe,report,stage,specs,poll)
            elif card_operation:frame,frame_facts,card_frames=_card_frame(recipe,report,stage,specs,poll)
            elif recovery_operation:frame,frame_facts,recovery_binding=_recovery_frame(recipe,report,stage,specs,poll)
            elif paper_operation:frame,frame_facts,paper_context=_paper_frame(recipe,report,stage,specs,poll)
            elif model_operation:frame,frame_facts,model_asset=_model_frame(recipe,report,stage,specs)
            else:frame,frame_facts = _replay_frame(recipe,report,stage,specs)
        except (recovery_contract.NativeRecoveryReplayUnavailable,paper_contract.NativePaperReplayUnavailable,crop_contract.NativeCropReplayUnavailable):
            raise
        except runner._tensor_callback.TensorCallbackMutation:
            raise
        except (ValueError,TypeError,StopIteration) as error:
            raise archives.NativeResultExportError(f'Replay frame verification failed: {error}') from error
        if crop_operation:crop_context_baseline=runner._card_recipe.correction._metadata_signature(crop_context)
        facts.update(frame_facts);facts['coordinate_frame']=frame.to_dict()
        parameters = recipe.parameters_for_runner(stage_id)
        foreground_region = _foreground_region_context(recipe, stage, node, frame)
        workspace_runtime = 'workspace_workflow_adapter' in report['runtime']
        if foreground_region is not None:
            if not workspace_runtime:
                raise archives.NativeResultExportError('Replay foreground adapter omits its working-runtime binding.')
            facts['foreground_region_bounds_xyxy'] = foreground_region
            facts['foreground_region_adapter'] = 'CUDA full-frame binary mask exclusion v1'
        expected_calculation = None
        if stage.operation_id == 'threshold.otsu' and node.get('calculation') is not None:
            expected_calculation = runner._threshold.validate_otsu_calculation_facts(
                _compact_frame_metadata(node['calculation']), parameters['object_type'])
            calculation_watched.append((expected_calculation, runner._compact_recovery_signature(expected_calculation)))
        calculation_comparison = None
        if recovery_operation or paper_operation or crop_operation:
            recovery_watched.append((parameters,runner._compact_recovery_signature(parameters)))
        automatic_trait_count=stage.operation_id in ('analysis.color','analysis.size') and parameters['n_labels'] is None
        if stage.operation_id in ('analysis.color','analysis.size'):
            parameters=_effective_traits(recipe,node,color_contract if stage.operation_id=='analysis.color' else size_contract)
        size_parameters=parameters
        size_unit='pixels'
        if units_operation:
            edge=next(edge for edge in incoming if edge.target_port=='observations')
            size_parameters=_effective_traits(recipe,nodes[edge.source_stage],size_contract)
            scale=size_contract.decode(input_bindings['scale'],archives.NativeResultExportError,poll)
            _,_,size_unit=size_contract.scale(scale,archives.NativeResultExportError)
        def options(name,direction):
            values={'label':parameters.get('label'),'parameters':parameters,
                'size_parameters':size_parameters,'size_unit':size_unit if units_operation and direction=='output' else 'pixels',
                'metadata_role':('size_scale' if direction=='input' else 'size_units')
                    if units_operation and name in ('scale','metadata') else None}
            if model_operation:
                count=(len(model_asset.identity['class_names']) if model_asset.kind=='bayes'
                       else len(model_contract.mask_names(parameters,model_asset.identity['cluster_count'])[0])
                       if stage.operation_id=='analysis.kmeans_masks' else None)
                values.update(model_kind=model_asset.kind,model_identity=model_asset.identity,model_count=count)
            if card_operation:
                roles=card_contract.PORTS[stage.operation_id][0 if direction=='input' else 1]
                values.update(card_role=roles[name],card_frames=card_frames)
            if recovery_operation:
                roles=recovery_contract.PORTS[stage.operation_id][0 if direction=='input' else 1]
                values.update(recovery_role=roles[name],recovery_binding=recovery_binding)
            if crop_operation:
                roles=crop_contract.INPUT_ROLES if direction=='input' else crop_contract.OUTPUT_ROLES
                values.update(crop_role=roles[name],crop_context=crop_context)
            if paper_operation:
                roles=paper_contract.PORTS[stage.operation_id][0 if direction=="input" else 1]
                values.update(paper_role=roles[name],paper_context=paper_context,paper_name=name)
            return values
        optional=(('cat_list',) if stage.operation_id=='analysis.kmeans_masks' else
                  ('roi_bounds_xyxy',) if stage.operation_id=='color_card.preprocess' else
                  ('validity_roi',) if stage.operation_id=='recovery.validity' else
                  ('color_chip_size',) if stage.operation_id=='color_card.scale' else ())
        if stage.operation_id in runner._morphology_roi_recipe.OPERATIONS:
            optional=(runner._morphology_roi_recipe.PARAMETER,)
        if stage.operation_id == runner._fill_roi_recipe.OPERATION:
            optional=(runner._fill_roi_recipe.PARAMETER,)
        if stage.operation_id == runner._median_roi_recipe.OPERATION:
            optional=(runner._median_roi_recipe.PARAMETER,)
        runner._required_parameters(parameters,optional=optional)
        facts.update(parameters=parameters,input_lineage=lineage,preview_port=preview)
        if units_operation:
            # Admit the fixed audit reserve before a cold runtime check creates
            # cached CUDA tables. Conversion itself has no selected arrays.
            facts['plan']=_plan(frame.shape,stage.operation_id,parameters,0)
        if recovery_operation:
            selected_inputs={name:_binding_product(input_bindings[name],_port(stage,name,'input').data_type,
                specs,frame,poll=poll,**options(name,'input')) for name in input_ports}
            selected_outputs={name:_binding_product(output_bindings[name],_port(stage,name,'output').data_type,
                specs,frame,poll=poll,**options(name,'output')) for name in output_ports}
            names=tuple(dict.fromkeys((*_recovery_array_names(selected_inputs),*_recovery_array_names(selected_outputs))))
            recovery_plan_options={}
            if stage.operation_id in ('recovery.partition','recovery.annotation','recovery.measurements'):
                descriptor=(selected_outputs['partition'][1] if stage.operation_id=='recovery.partition' else
                    selected_inputs['partition'][1] if stage.operation_id=='recovery.annotation' else selected_inputs['annotation'][1])
                metadata=descriptor['metadata'];input_names=tuple(dict.fromkeys(_recovery_array_names(selected_inputs)))
                inventory_count=len(metadata['inventory']['objects']) if 'inventory' in metadata else len(metadata['objects'])
                recovery_plan_options={'selected_allocations':tuple(specs[name]['nbytes'] for name in names),
                    'input_allocations':tuple(specs[name]['nbytes'] for name in input_names),
                    'label_count':specs[descriptor['arrays']['label_ids']]['shape'][0] if 'label_ids' in descriptor['arrays'] else 0,
                    'inventory_count':inventory_count}
                if stage.operation_id=='recovery.measurements':
                    facts['selected_dependency']='annotation metadata only; band headers and aliases bound without selecting band pixels'
                    facts['upstream_statistics']='archived partition/annotation statistics; upstream pixels and counts are not recomputed'
            facts['plan']=recovery_contract.plan(stage.operation_id,frame.shape,parameters,
                selected_bytes=sum(specs[name]['nbytes'] for name in names),
                selected_limit=MAX_TYPED_SELECTED_BYTES,cuda_limit=MAX_REPLAY_CUDA_BYTES,**recovery_plan_options)
            facts.update(selected_arrays=names,recovery_contract_sha256=recovery_identity)
            if recovery_contract.state()!=recovery_state or runner._digest(runner._module_record(recovery_contract))!=recovery_identity:
                raise runner.NativeRecipeError('Recovery replay contract changed before runtime/upload.')

        if paper_operation:
            selected_inputs={name:_binding_product(input_bindings[name],_port(stage,name,'input').data_type,
                specs,frame,poll=poll,**options(name,'input')) for name in input_ports}
            selected_outputs={name:_binding_product(output_bindings[name],_port(stage,name,'output').data_type,
                specs,frame,poll=poll,**options(name,'output')) for name in output_ports}
            names=tuple(dict.fromkeys(value for kind,value in (*selected_inputs.values(),*selected_outputs.values()) if kind=='array'))
            input_names=tuple(dict.fromkeys(value for kind,value in selected_inputs.values() if kind=='array'))
            facts['plan']=paper_contract.plan(stage.operation_id,frame.shape,parameters,
                selected_allocations=tuple(specs[name]['nbytes'] for name in names),
                input_allocations=tuple(specs[name]['nbytes'] for name in input_names),
                crop_pixels=paper_contract.crop_total(paper_context.get('final_components',())),
                selected_limit=MAX_TYPED_SELECTED_BYTES,cuda_limit=MAX_REPLAY_CUDA_BYTES)
            facts.update(selected_arrays=names,paper_contract_sha256=paper_identity)
            if stage.operation_id=='smpts_paper.contacts':
                facts['selected_dependency']='archived label input uploaded; calculation uses only its frame shape and component inventory'
            if paper_contract.state()!=paper_state or runner._digest(runner._module_record(paper_contract))!=paper_identity:
                raise runner.NativeRecipeError('Paper replay contract changed before runtime/upload.')
        if crop_operation:
            selected_inputs={name:_binding_product(input_bindings[name],_port(stage,name,'input').data_type,
                specs,frame,poll=poll,check_versions=False,**options(name,'input')) for name in input_ports}
            selected_outputs={name:_binding_product(output_bindings[name],_port(stage,name,'output').data_type,
                specs,frame,poll=poll,check_versions=False,**options(name,'output')) for name in output_ports}
            names=tuple(dict.fromkeys(value for kind,value in (*selected_inputs.values(),*selected_outputs.values()) if kind=='array'))
            facts['plan']=crop_contract.plan(crop_context,selected_allocations=tuple(specs[name]['nbytes'] for name in names),
                selected_limit=MAX_TYPED_SELECTED_BYTES,cuda_limit=MAX_REPLAY_CUDA_BYTES)
            facts.update(selected_arrays=names,crop_contract_sha256=crop_identity)
            if crop_contract.state()!=crop_state or runner._digest(runner._module_record(crop_contract))!=crop_identity:
                raise runner.NativeRecipeError('Crop replay contract changed before runtime/upload.')
        if card_operation:
            selected_inputs={name:_binding_product(input_bindings[name],_port(stage,name,'input').data_type,
                specs,frame,poll=poll,check_versions=False,**options(name,'input')) for name in input_ports}
            selected_outputs={name:_binding_product(output_bindings[name],_port(stage,name,'output').data_type,
                specs,frame,poll=poll,check_versions=False,**options(name,'output')) for name in output_ports}
            names=tuple(dict.fromkeys(value for kind,value in (*selected_inputs.values(),*selected_outputs.values()) if kind=='array'))
            facts['plan']=_card_plan(card_frames,stage.operation_id,parameters,sum(specs[name]['nbytes'] for name in names))
            facts.update(selected_arrays=names,card_contract_sha256=card_identity)
            if (card_contract.state()!=card_state or runner._digest(runner._module_record(card_contract))!=card_identity):
                raise runner.NativeRecipeError('Card replay contract changed before runtime/upload.')
        if model_operation:
            selected_inputs,selected_outputs=_select_model_products(stage,input_bindings,output_bindings,
                specs,frame,model_asset,parameters,poll)
            names=tuple(dict.fromkeys((*_model_array_names(selected_inputs),*_model_array_names(selected_outputs))))
            selected_bytes=sum(specs[name]['nbytes'] for name in names)
            if selected_bytes>MAX_TYPED_SELECTED_BYTES:
                raise model_contract.NativeModelReplayUnavailable('Model replay exceeds selected-array admission before loading.')
            if stage.operation_id!='analysis.kmeans_predict':
                facts['plan']=_model_plan(frame,stage.operation_id,parameters,model_asset)
            loaded_report,arrays=archives.load_native_recipe_result(directory,limits=ARCHIVE_LIMITS,
                array_names=names,expected_revision=report['result_revision'],cancel=poll)
            if loaded_report!=report:raise archives.NativeResultExportError('Archive report changed during model replay selection.')
            if stage.operation_id!='analysis.kmeans_masks':
                parsed_model=model_contract.parse_asset(model_asset,arrays[model_asset.array_name])
            facts['plan']=_model_plan(frame,stage.operation_id,parameters,model_asset,parsed_model)
            facts['plan']['selected_bytes']=selected_bytes
            facts.update(selected_arrays=names,model_contract_sha256=model_identity,
                active_image_port=next((name for name in selected_inputs if name in ('rgb_image','gray_image')),None),
                model_bytes_selected=stage.operation_id!='analysis.kmeans_masks')
            if (model_contract.state()!=model_state
                    or runner._digest(runner._module_record(model_contract))!=model_identity):
                raise runner.NativeRecipeError('Model replay contract changed before runtime/upload.')
        if not torch.cuda.is_available():
            return _result('unavailable',stage_id,'CUDA is unavailable.',metadata=facts)
        selected_device = torch.device(device)
        if selected_device.type!='cuda':
            return _result('unsupported',stage_id,'Stage replay requires CUDA.',metadata=facts)
        if selected_device.index is None:
            selected_device = torch.device('cuda',torch.cuda.current_device())
        runtime_factory,audit = _runtime,runner._module_record
        initial_records,initial_runtime = runtime_factory(recipe,selected_device)
        if workspace_runtime:
            runner._bind_working_runtime(initial_records, initial_runtime)
        # Expected compact provenance uses the recorded calculation runtime.
        # Reject incompatible runtimes before checking it against current code.
        if (report['runtime'].get('calculation_modules_sha256') is None or report['runtime']!=initial_runtime):
            return _result('unavailable',stage_id,'Recorded calculation code/CUDA runtime differs or omits replay binding.',metadata=facts)
        if crop_operation:_crop_frame(recipe,report,stage,specs,poll,check_versions=True)
        if card_operation:
            _card_frame(recipe,report,stage,specs,poll,check_versions=True)
            for direction,bindings in (('input',input_bindings),('output',output_bindings)):
                for name in (input_ports if direction=='input' else output_ports):
                    _binding_product(bindings[name],_port(stage,name,direction).data_type,specs,frame,
                        poll=poll,**options(name,direction))
        color_identity=runner._digest(audit(color_contract)) if stage.operation_id=='analysis.color' else None
        color_state=color_contract.state() if color_identity is not None else None
        if color_state is not None and color_state!=color_contract.PINNED_STATE:
            raise runner.NativeRecipeError('Loaded colour compact contract differs from its pinned state.')
        size_identity=runner._digest(audit(size_contract)) if size_operation else None
        size_state=size_contract.state() if size_identity is not None else None
        if size_state is not None and size_state!=size_contract.PINNED_STATE:
            raise runner.NativeRecipeError('Loaded size compact contract differs from its pinned state.')
        if not model_operation and not card_operation and not recovery_operation and not paper_operation and not crop_operation:
            selected_inputs = {name:_binding_product(input_bindings[name],_port(stage,name,'input').data_type,
                specs,frame,poll=poll,**options(name,'input')) for name in input_ports}
            selected_outputs = {name:_binding_product(output_bindings[name],_port(stage,name,'output').data_type,
                specs,frame,poll=poll,**options(name,'output')) for name in selected_output_ports}
            names = tuple(dict.fromkeys(value for kind,value in (*selected_inputs.values(),*selected_outputs.values())
                if kind=='array'))
            facts['plan'] = _plan(frame.shape,stage.operation_id,parameters,sum(specs[name]['nbytes'] for name in names))
        if foreground_region is not None:
            facts['plan'] = _foreground_region_plan(facts['plan'], frame)
        replay_module = sys.modules[__name__]
        replay_identity = runner._digest(audit(replay_module))
        archive_identity = runner._digest(audit(archives))
        checkpoint_identity = runner._digest(audit(checkpoint_library))
        diagnostic_identity = (runner._digest(audit(diagnostic_contract))
            if stage.operation_id==runner._diagnostic_recipe.OPERATION else None)
        facts.update(replay_implementation_sha256=replay_identity,current_runtime=initial_runtime)
        if diagnostic_identity is not None:
            facts['diagnostic_contract_sha256'] = diagnostic_identity
        if color_identity is not None:
            facts['color_contract_sha256']=color_identity
        if size_identity is not None:
            facts['size_contract_sha256']=size_identity
            facts['size_effective_parameters']=size_parameters
            facts['archived_output_ports']=selected_output_ports
        if not model_operation:
            loaded_report,arrays = archives.load_native_recipe_result(directory,limits=ARCHIVE_LIMITS,
                array_names=names,expected_revision=report['result_revision'],cancel=poll)
            if loaded_report!=report:
                raise archives.NativeResultExportError('Archive report changed during replay input selection.')
        poll()
        with torch.cuda.device(selected_device):
            uploaded = {name:torch.tensor(arrays[name],device=selected_device) for name in names
                if not model_operation or name!=model_asset.array_name}
            watched.extend((value,runner._tensor_signature(value)) for value in uploaded.values())
            def materialize(kind,value):
                if recovery_operation:return _recovery_materialize(kind,value,uploaded)
                if kind=='array':return uploaded[value]
                if kind=='model':return parsed_model
                if kind=='mask_map':return MappingProxyType({key:uploaded[name] for key,name in value})
                return value
            inputs = MappingProxyType({name:materialize(kind,value) for name,(kind,value) in selected_inputs.items()})
            expected_products = MappingProxyType({name:materialize(kind,value) for name,(kind,value) in selected_outputs.items()})
            if model_operation or card_operation:
                nested_watched.extend((value,_model_signature(value)) for value in (inputs,expected_products))
            if recovery_operation or paper_operation or crop_operation:
                recovery_watched.extend((value,runner._compact_recovery_signature(value)) for value in (inputs,expected_products))
            owned_storage={value.untyped_storage().data_ptr() for value in uploaded.values()}
            def validation_options(name,direction):
                values=options(name,direction)
                if card_operation or recovery_operation or paper_operation or crop_operation:values['owned_storage']=owned_storage
                return values
            # Preserve the first-input guard alias used by existing callers/tests.
            source = next((value for value in inputs.values() if torch.is_tensor(value)),None)
            poll()
            for name,value in inputs.items():
                _validate_product(value,_port(stage,name,'input').data_type,frame,selected_device,poll,
                    archives.NativeResultExportError,**validation_options(name,'input'))
            for name,value in expected_products.items():
                if color_identity is None:
                    _validate_product(value,_port(stage,name,'output').data_type,frame,selected_device,poll,
                        archives.NativeResultExportError,**validation_options(name,'output'))
            if automatic_trait_count and int(inputs['labels'].max().item())!=parameters['n_labels']:
                raise archives.NativeResultExportError('Trait resolved object count differs from its archived label maximum.')
            if diagnostic_identity is not None:
                poll()
                ids = torch.unique(inputs['labels'])
                if (ids.numel()>runner._watershed_diagnostic.MAX_LABELS+1
                        or int(ids[0].item())!=0 or int(ids[-1].item())>=ids.numel()):
                    raise archives.NativeResultExportError('Diagnostic input IDs exceed the source palette-index domain.')
                facts['diagnostic_input_label_ids'] = tuple(ids[1:].cpu().tolist())
                facts['diagnostic_input_palette_count'] = int(ids.numel())
                del ids
            _validate_relationships(stage.operation_id,expected_products,inputs,frame,poll,archives.NativeResultExportError)
            if paper_operation:_paper_relationships(stage.operation_id,expected_products,inputs,parameters,archives.NativeResultExportError)
            try:
                dispatched = runner._dispatch(stage.operation_id,inputs,parameters,poll)
                calculation = runner._dispatch_calculation(stage.operation_id,dispatched,parameters)
                if foreground_region is not None:
                    from seedvision.pipeline.workflows import restrict_foreground
                    poll()
                    dispatched = dict(dispatched)
                    dispatched['mask'] = restrict_foreground(dispatched['mask'], foreground_region)
                    poll()
                outputs = MappingProxyType(dict(dispatched))
                if stage.operation_id == 'threshold.otsu':
                    actual_calculation = dict(calculation)
                    calculation_comparison = {
                        'mismatched_elements': int(expected_calculation is not None
                            and actual_calculation != expected_calculation),
                        'compared_elements': int(expected_calculation is not None), 'kind': 'compact'}
                    facts['calculation'] = {
                        'native': actual_calculation, 'archived': expected_calculation,
                        'comparison_status': ('unrecorded_legacy_archive' if expected_calculation is None
                            else 'mismatch' if calculation_comparison['mismatched_elements'] else 'matched')}
                    calculation_watched.append((facts['calculation'], runner._compact_recovery_signature(facts['calculation'])))

            except NotImplementedError as error:
                if not paper_operation and stage.operation_id not in ('threshold.custom_range','analysis.size'):
                    raise
                poll()
                raise NativeStageReplayUnavailable(str(error)) from error
            except runner.NativeRecipeError as error:
                if color_identity is None or not str(error).startswith('Original colour failure:'):
                    raise
                poll()
                raise NativeStageReplayUnavailable(str(error)) from error
            except (runner._extended_thresholds.ExtendedThresholdError,runner._threshold.OtsuAmbiguityError) as error:
                if stage.operation_id not in EXTENDED_THRESHOLD_OPERATIONS:
                    raise
                poll()
                raise NativeStageReplayUnavailable(str(error)) from error
            except runner._tensor_callback.TensorCallbackMutation:
                raise
            except runner._paper_recipe.paper.PaperMutationError:
                raise
            except runner._recovery_recipe.core.CoreRecoveryMutationError:
                raise
            except runner._recovery_recipe.core.CoreRecoveryResourceError as error:
                if not recovery_operation:raise
                poll()
                raise NativeStageReplayUnavailable(str(error)) from error
            except ValueError as error:
                if model_operation or card_operation:
                    poll()
                    raise NativeStageReplayUnavailable(str(error)) from error
                if not units_operation or str(error)!='Size measurement overflowed during conversion.':
                    raise
                poll()
                raise NativeStageReplayUnavailable(str(error)) from error
            intact()
            if (diagnostic_identity is not None and set(outputs)=={'metadata'}
                    and outputs['metadata'].get('status')=='unavailable'):
                return _result('unavailable',stage_id,outputs['metadata']['reason'],metadata=facts)
            if (not set(required_outputs)<=set(outputs) or not set(outputs)<=set(output_ports)
                    or stage.operation_id!='analysis.size' and set(outputs)!=set(output_ports)):
                raise runner.NativeRecipeError('Replayed output ports differ from the exact selected stage.')
            watched.extend((value,runner._tensor_signature(value)) for value in outputs.values() if torch.is_tensor(value))
            if model_operation or card_operation:nested_watched.append((outputs,_model_signature(outputs)))
            if recovery_operation or paper_operation or crop_operation:recovery_watched.append((outputs,runner._compact_recovery_signature(outputs)))
            for name,value in outputs.items():
                _validate_product(value,_port(stage,name,'output').data_type,frame,selected_device,poll,
                    runner.NativeRecipeError,**validation_options(name,'output'))
            _validate_relationships(stage.operation_id,outputs,inputs,frame,poll,runner.NativeRecipeError)
            if paper_operation:_paper_relationships(stage.operation_id,outputs,inputs,parameters,runner.NativeRecipeError)
            color_products=MappingProxyType({name:color_contract.canonical(value) for name,value in outputs.items()}) if color_identity is not None else None
            size_products=MappingProxyType({name:size_contract.canonical(value,runner.NativeRecipeError,poll)
                for name,value in outputs.items() if not torch.is_tensor(value)}) if size_identity is not None else None
            comparisons={}
            for name in output_ports:
                if name not in outputs or name not in expected_products:
                    comparisons[name]={'kind':'output_availability','mismatched_elements':int((name in outputs)!=(name in expected_products)),
                        'compared_elements':1,'shape_equal':True,'actual_available':name in outputs,'expected_available':name in expected_products}
                elif model_operation:
                    comparisons[name]=_compare_model_product(outputs[name],expected_products[name],poll)
                elif crop_operation and not torch.is_tensor(outputs[name]):
                    comparisons[name]=crop_contract.compare(outputs[name],expected_products[name],poll)
                elif card_operation and not torch.is_tensor(outputs[name]):
                    comparisons[name]=card_contract.compare(outputs[name],expected_products[name],poll)
                elif paper_operation and not torch.is_tensor(outputs[name]):
                    comparisons[name]=paper_contract.compare(paper_contract.canonical(outputs[name],poll),expected_products[name],poll)
                elif recovery_operation:
                    comparisons[name]=_compare_recovery_product(outputs[name],expected_products[name],poll)
                elif color_identity is not None:
                    comparisons[name]=color_contract.compare(color_products[name],expected_products[name],poll)
                elif size_products is not None and name in size_products:
                    comparisons[name]=size_contract.compare(size_products[name],expected_products[name],poll)
                else:comparisons[name]=_compare_product(outputs[name],expected_products[name],poll)
            poll()
        final_records,final_runtime = runtime_factory(recipe,selected_device)
        if workspace_runtime:
            runner._bind_working_runtime(final_records, final_runtime)
        if (final_records!=initial_records or final_runtime!=initial_runtime
                or runner._digest(audit(replay_module))!=replay_identity
                or runner._digest(audit(archives))!=archive_identity
                or runner._digest(audit(checkpoint_library))!=checkpoint_identity
                or color_identity is not None and (runner._digest(audit(color_contract))!=color_identity
                    or color_contract.state()!=color_state)
                or size_identity is not None and (runner._digest(audit(size_contract))!=size_identity
                    or size_contract.state()!=size_state)
                or model_identity is not None and (runner._digest(audit(model_contract))!=model_identity
                    or model_contract.state()!=model_state)
                or card_identity is not None and (runner._digest(audit(card_contract))!=card_identity
                    or card_contract.state()!=card_state)
                or recovery_identity is not None and (runner._digest(audit(recovery_contract))!=recovery_identity
                    or recovery_contract.state()!=recovery_state)
                or crop_identity is not None and (runner._digest(audit(crop_contract))!=crop_identity
                    or crop_contract.state()!=crop_state)
                or paper_identity is not None and (runner._digest(audit(paper_contract))!=paper_identity
                    or paper_contract.state()!=paper_state)
                or diagnostic_identity is not None and runner._digest(audit(diagnostic_contract))!=diagnostic_identity):
            raise runner.NativeRecipeError('Loaded replay calculation changed during execution.')
        intact()
        if color_products is not None and any(color_contract.canonical(outputs[name])!=value for name,value in color_products.items()):
            raise runner.NativeRecipeError('Replay colour output changed during execution.')
        if size_products is not None and any(size_contract.canonical(outputs[name],runner.NativeRecipeError)!=value for name,value in size_products.items()):
            raise runner.NativeRecipeError('Replay size output changed during execution.')
        mismatch = sum(item['mismatched_elements'] for item in comparisons.values())
        if calculation_comparison is not None:
            mismatch += calculation_comparison['mismatched_elements']
        facts['comparison'] = {'mode':('exact_calibrated_crop_parent_local_cuda_and_metadata' if crop_operation else 'exact_paper_cuda_rasters_packed_crops_and_compact_metadata' if paper_operation else 'exact_recovery_cuda_rasters_and_compact_metadata' if recovery_operation else 'exact_card_cuda_rasters_and_compact_metadata' if card_operation else 'exact_ordered_cuda_class_products_and_model_metadata' if model_operation else 'exact_typed_size_units_metadata' if units_operation else 'exact_typed_size_metadata_and_optional_cuda_diagnostic' if size_identity is not None else 'exact_typed_colour_metadata' if color_identity is not None else 'exact_cuda_rasters_and_compact_metadata' if diagnostic_identity is not None
                                      else 'exact_on_cuda'),'mismatched_elements':mismatch,
            'compared_elements':sum(item['compared_elements'] for item in comparisons.values()),
            'archive_payloads_verified':True,'by_port':comparisons,
            'floating_zero':'numeric equality; signed zeros compare equal'}
        if calculation_comparison is not None:
            facts['comparison']['calculation'] = calculation_comparison
            facts['comparison']['compared_elements'] += calculation_comparison['compared_elements']
        return _result('matched' if mismatch==0 else 'mismatch',stage_id,outputs=outputs,metadata=facts)
    except runner.NativeRecipeCancelled as error:
        return _result('cancelled',stage_id,str(error),metadata=facts)
    except (runner._median.MedianCancelledError,runner._fill.FillCancelledError,
            runner._distance.DistanceCancelledError,runner._peaks.PeakCancelledError,
             runner._watershed.WatershedCancelled,runner._watershed_diagnostic.WatershedDiagnosticCancelled,
             runner._range.RangeCancelledError,
             runner._color.ColorAnalysisCancelled,
             runner._size.PlantCVSizeCancelledError,
             runner._naive_bayes.NaiveBayesCancelled,runner._kmeans_masks.NativeKMeansCancelled,
             runner._kmeans_predict.KMeansPredictionCancelled,runner._kmeans_preprocess.KMeansPreprocessCancelled,
             runner._kmeans_features.KMeansFeatureCancelled,runner._kmeans_classifier.KMeansClassificationCancelled,
             runner._card_recipe.card.AutomaticCardCancelled,runner._card_recipe.preprocess.ColorCardPreprocessCancelled,
             runner._card_recipe.contours.BinaryContourCancelled,runner._card_recipe.correction.CorrectionCancelled,
             runner._recovery_recipe.core.CoreRecoveryCancelled,runner._paper_recipe.paper.PaperCancelled,
            runner._extended_thresholds.ExtendedThresholdCancelled) as error:
        return _result('cancelled',stage_id,str(error),metadata=facts)
    except (NativeStageReplayUnavailable,model_contract.NativeModelReplayUnavailable,card_contract.NativeCardReplayUnavailable,
            recovery_contract.NativeRecoveryReplayUnavailable,paper_contract.NativePaperReplayUnavailable,crop_contract.NativeCropReplayUnavailable) as error:
        return _result('unavailable',stage_id,str(error),metadata=facts)
    except (archives.NativeResultExportError,model_contract.NativeModelReplayError,card_contract.NativeCardReplayError,
            recovery_contract.NativeRecoveryReplayError,paper_contract.NativePaperReplayError,crop_contract.NativeCropReplayError,KeyError,OSError) as error:
        return _result('invalid_archive',stage_id,str(error),metadata=facts)
    except Exception as error:
        return _result('failed',stage_id,f'{type(error).__name__}: {error}',metadata=facts)
