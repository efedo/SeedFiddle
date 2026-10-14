"""Primitive dual-frame contracts for saved calibrated RGB crop replay.

Only verified direct inputs are executed. Full-parent RGB uploads and fresh local
RGB/gray buffers are admitted separately; metadata limits are not RSS promises.
"""
from collections.abc import Mapping
from dataclasses import dataclass

from . import native_card_replay as card
from . import native_calibrated_seed_recipe as native
from .checkpoints import CoordinateFrame, _freeze_json

OPERATION = 'calibrated_seed.crop'
INPUT_ROLES = {'corrected_rgb':'parent_rgb','source_frame':'source_binding',
    'geometry':'geometry','bounds':'bounds'}
OUTPUT_ROLES = {'rgb_image':'local_rgb','gray_image':'local_gray','metadata':'crop_metadata'}
MAX_PIXELS = 64_000_000
MAX_SELECTED_BYTES = 128*1024*1024
MAX_CUDA_BYTES = 1024*1024*1024
TILE_PIXELS = 262_144
COMPARE_ELEMENTS = 262_144
AUDIT_CUDA_BYTES = 64*1024
ALLOCATION_BYTES = 512


class NativeCropReplayError(ValueError):
    """An archived crop violates its frame, primitive or producer contract."""


class NativeCropReplayUnavailable(NativeCropReplayError):
    """A structurally valid crop exceeds its declared replay resources."""


@dataclass(frozen=True,slots=True)
class CropFrames:
    parent: CoordinateFrame
    local: CoordinateFrame
    translation: Mapping
    source_sha256: str
    card_bounds: tuple[int,int,int,int]


def state():
    return (OPERATION,tuple(INPUT_ROLES.items()),tuple(OUTPUT_ROLES.items()),MAX_PIXELS,
        MAX_SELECTED_BYTES,MAX_CUDA_BYTES,TILE_PIXELS,COMPARE_ELEMENTS,AUDIT_CUDA_BYTES,
        ALLOCATION_BYTES,native.GRAY_CONVERSION,native.LOCAL_COORDINATES,card.state())


def decode(value,poll=None):
    return card.decode(value,poll)


def canonical(value,poll=None):
    return card.canonical(value,poll)


def compare(actual,expected,poll=None):
    result=card.compare(actual,expected,poll)
    return {**result,'kind':'calibrated_crop_compact'}


def _same(actual,expected,poll=None):
    return compare(actual,expected,poll)['mismatched_elements']==0


def frames(parent,parameters,source_sha256,card_bounds):
    if not isinstance(parameters,Mapping) or set(parameters)!={'seed_roi_xyxy'}:
        raise NativeCropReplayError('Crop controls differ from their exact contract.')
    try:
        local,translation=native._crop_frame(parent,parameters['seed_roi_xyxy'])
        _,_,bounds=card.frames(parent.shape,card_bounds)
    except (ValueError,TypeError) as error:
        raise NativeCropReplayError(str(error)) from error
    if type(source_sha256) is not str or len(source_sha256)!=64 or any(c not in '0123456789abcdef' for c in source_sha256):
        raise NativeCropReplayError('Crop source identity requires a SHA-256.')
    return CropFrames(parent,local,_freeze_json(translation,name='crop translation'),source_sha256,bounds)


def raster_spec(role,context):
    if role=='parent_rgb':return (*context.parent.shape,3),'|u1'
    if role=='local_rgb':return (*context.local.shape,3),'|u1'
    if role=='local_gray':return context.local.shape,'|u1'
    raise NativeCropReplayError('Unknown calibrated crop raster role.')


def fixed_metadata(context):
    return _freeze_json({**context.translation,'coordinate_frame':context.local.to_dict(),
        'source_sha256':context.source_sha256,'gray_conversion':native.GRAY_CONVERSION,
        'labels_and_coordinates':native.LOCAL_COORDINATES},name='crop fixed metadata')


def validate(value,role,context,poll=None,*,check_versions=True):
    value=canonical(value,poll)
    if role=='source_binding':
        expected=_freeze_json({'coordinate_frame':context.parent.to_dict(),
            'source_sha256':context.source_sha256},name='crop source binding')
        if not _same(value,expected,poll):
            raise NativeCropReplayError('Crop source binding differs from its parent frame/source.')
    elif role in ('geometry','bounds'):
        try:
            value=card.validate(value,role,context.parent.shape,context.card_bounds,
                poll=poll,check_versions=check_versions)
        except card.NativeCardReplayError as error:
            raise NativeCropReplayError(str(error)) from error
        if role=='bounds' and tuple(value)!=context.card_bounds:
            raise NativeCropReplayError('Crop card bounds differ from their producer control.')
    elif role=='crop_metadata':
        expected=fixed_metadata(context)
        if not isinstance(value,Mapping) or set(value)!=set(expected)|{'card_bounding_overlap_pixels'}:
            raise NativeCropReplayError('Crop metadata has missing or additional fields.')
        if not _same({key:value[key] for key in expected},expected,poll):
            raise NativeCropReplayError('Crop metadata differs from its exact source/frame/translation contract.')
        count=value['card_bounding_overlap_pixels']
        pixels=context.local.shape[0]*context.local.shape[1]
        if type(count) is not int or not 0<=count<=pixels:
            raise NativeCropReplayError('Crop overlap must be an exact integer inside the local pixel domain.')
        # The saved count is an output to compare, never an admission claim.
    else:
        raise NativeCropReplayError('Unknown calibrated crop compact role.')
    return value


def rounded(size):
    return ((size+ALLOCATION_BYTES-1)//ALLOCATION_BYTES)*ALLOCATION_BYTES


def plan(context,*,selected_allocations,selected_limit=MAX_SELECTED_BYTES,cuda_limit=MAX_CUDA_BYTES):
    parent=context.parent.shape[0]*context.parent.shape[1]
    local=context.local.shape[0]*context.local.shape[1]
    if not 1<=local<=parent<=MAX_PIXELS:
        raise NativeCropReplayUnavailable('Crop parent/local pixels exceed replay admission.')
    if (type(selected_allocations) not in (tuple,list) or any(type(size) is not int or size<=0
            for size in selected_allocations)):
        raise NativeCropReplayError('Crop selected allocations require positive exact integers.')
    selected=sum(selected_allocations)
    uploads=sum(rounded(size) for size in selected_allocations)
    tile=min(local,TILE_PIXELS)
    # BGR flip3T, three int32 channels12T, arithmetic temporaries and uint8
    # result fit64T.32 allocator slots cover each phase's small allocations.
    conversion=64*tile+32*ALLOCATION_BYTES
    comparison=9*min(3*local,COMPARE_ELEMENTS)+32*ALLOCATION_BYTES
    fresh=rounded(3*local)+rounded(local)  # Conservative even for parent aliases.
    cuda=uploads+fresh+max(conversion,comparison)+AUDIT_CUDA_BYTES
    if selected>selected_limit or cuda>cuda_limit:
        raise NativeCropReplayUnavailable('Crop selected arrays or modeled CUDA peak exceed replay admission.')
    return {'pixels':parent,'roi_pixels':local,'selected_bytes':selected,
        'selected_upload_bytes':uploads,'fresh_output_bytes':fresh,'cuda_bytes':cuda,
        'conversion_tile_pixels':tile,'primitive_scratch_bytes':conversion,
        'comparison_scratch_bytes':comparison,'runtime_audit_cuda_reserve_bytes':AUDIT_CUDA_BYTES,
        'selected_limit_bytes':selected_limit,'cuda_limit_bytes':cuda_limit,
        'compact_metadata_limit_bytes':card.MAX_METADATA_BYTES,
        'bounds':'modeled CUDA allocation admission; not RSS or elapsed-time guarantees',
        'alias_accounting':'unique selected array names; fresh RGB copy charged even when the calculation returns a parent view'}


PINNED_STATE=state()
