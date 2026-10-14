"""Experimental recovery from supplied eroded cores and explicit elevation.

This is a distinct engineering feature, not the unavailable SMPTS erosion or
contact rule. The known foreground union is retained; individual boundaries
between cores are inferred. Four-connected partitioning preserves the existing
pinned scikit-image heap behavior (see SCIKIT_IMAGE_WATERSHED_NOTICES.md).
Only bounded scalar and per-ID facts cross to CPU. No per-object dense stack is
created. Cancellation guards catch ordinary PyTorch tensor/storage mutation;
direct writes through external raw pointers are outside that guard contract.
"""
from dataclasses import dataclass, fields
from hashlib import sha256
from pathlib import Path

import torch

from seedvision.cuda._nvrtc import fixed_kernel_program
from seedvision.cuda import plantcv_watershed as _watershed


MAX_RASTER_PIXELS = 64_000_000
MAX_LABELS = 32_768
ID_TABLE_SLOTS = 65_536
MAX_ID_PROBES = 128
VALIDATION_TILE_PIXELS = 65_536
KERNEL_BATCH_PIXELS = 4096
MAX_LIVE_BYTES = 1024*1024*1024
MAX_CPU_METADATA_BYTES = 64*1024*1024
MAX_WORK = 8_000_000_000
BACKEND_HEADROOM_BYTES = 8*1024*1024
PROFILE = 'experimental-core-recovery; explicit-f32-elevation; pinned-skimage-0.25.2; connectivity-4-v1'
_CUDA_SOURCE = Path(__file__).with_name('_core_recovery.cu').read_text(encoding='utf-8')
CUDA_SOURCE_SHA256 = sha256(_CUDA_SOURCE.encode('utf-8')).hexdigest()


class CoreRecoveryCancelled(RuntimeError):
    pass


class CoreRecoveryResourceError(ValueError):
    pass


class CoreRecoveryMutationError(ValueError):
    pass


@dataclass(frozen=True)
class CoreRecoveredObject:
    label_id: int
    seed_pixels: int
    assigned_pixels: int
    interface_pixels: int
    bbox: tuple
    touches_canvas: bool
    touches_invalid_support: bool
    touches_unresolved: bool


@dataclass(frozen=True)
class CoreRecoveryPlan:
    height: int
    width: int
    original_foreground_pixels: int
    domain_pixels: int
    validity_clipped_foreground_pixels: int
    seed_pixels: int
    labels: int
    input_backing_bytes: int
    watershed_workspace_bytes: int
    peak_live_bytes: int
    cpu_metadata_bytes: int
    work: int


@dataclass(frozen=True)
class CoreRecoveryPartition:
    original_foreground: torch.Tensor
    markers: torch.Tensor
    validity: torch.Tensor
    elevation: torch.Tensor
    labels: torch.Tensor
    domain: torch.Tensor
    unresolved: torch.Tensor
    interface: torch.Tensor
    label_ids: torch.Tensor
    objects: tuple
    plan: CoreRecoveryPlan
    unresolved_pixels: int
    interface_pixels: int
    status: str
    profile: str
    _snapshot: tuple


@dataclass(frozen=True)
class CoreRecoveryMeasurementStatus:
    label_id: int
    uncertainty_pixels: int
    reasons: tuple
    segmentation_measurement_allowed: bool
    uncontested_visible_partition: bool
    scientific_validation: bool


@dataclass(frozen=True)
class CoreRecoveryAnnotationPlan:
    retained_partition_bytes: int
    peak_live_bytes: int
    cpu_metadata_bytes: int
    work: int


@dataclass(frozen=True)
class CoreRecoveryAnnotation:
    partition: CoreRecoveryPartition
    uncertainty_band: torch.Tensor
    uncertainty_radius: int
    measurement_policy: str
    objects: tuple
    plan: CoreRecoveryAnnotationPlan
    _snapshot: tuple


def _signature(tensor):
    if not torch.is_tensor(tensor):
        raise CoreRecoveryMutationError('A retained recovery tensor was replaced.')
    storage = tensor.untyped_storage()
    return (id(tensor), tensor._version, tensor.data_ptr(), storage.data_ptr(),
            storage.nbytes(), tuple(tensor.shape), tuple(tensor.stride()),
            tensor.storage_offset(), tensor.dtype, tensor.device, tensor.layout)


def _backing_bytes(tensors):
    backings = {}
    for tensor in tensors:
        storage = tensor.untyped_storage()
        backings[(tensor.device, storage.data_ptr())] = storage.nbytes()
    return sum(backings.values())


class _Guard:
    def __init__(self, cancel, tensors, extra=None):
        if cancel is not None and not callable(cancel):
            raise TypeError('cancel must be a zero-argument callable or None.')
        self.cancel, self.extra, self.watched = cancel, extra, []
        self.watch(*tensors)

    def watch(self, *tensors):
        for tensor in tensors:
            tensor.record_stream(torch.cuda.current_stream(tensor.device))
            self.watched.append((tensor, _signature(tensor)))

    def check(self):
        if self.extra is not None:
            self.extra()
        for tensor, signature in self.watched:
            if _signature(tensor) != signature:
                raise CoreRecoveryMutationError('Recovery input or workspace changed during execution.')

    def checkpoint(self):
        self.check()
        if self.cancel is not None:
            try:
                decision = self.cancel()
            finally:
                self.check()
            if type(decision) is not bool:
                raise TypeError('cancel must return a bool.')
            if decision:
                raise CoreRecoveryCancelled('Core recovery cancelled before publication.')


def _inputs(foreground, markers, validity, elevation):
    tensors = (foreground, markers, validity, elevation)
    if not all(torch.is_tensor(t) for t in tensors):
        raise TypeError('Core recovery inputs must be PyTorch tensors.')
    if (foreground.layout != torch.strided or foreground.device.type != 'cuda'
            or foreground.ndim != 2 or foreground.dtype != torch.uint8):
        raise ValueError('Foreground must be a CUDA HxW uint8 binary raster.')
    if (markers.dtype != torch.int32 or validity.dtype != torch.uint8
            or elevation.dtype != torch.float32
            or any(t.layout != torch.strided or t.device != foreground.device
                   or tuple(t.shape) != tuple(foreground.shape) or not t.is_contiguous()
                   for t in tensors)):
        raise ValueError('Require contiguous same-frame CUDA uint8 masks, int32 cores and float32 elevation.')
    pixels = foreground.numel()
    if not 1 <= pixels <= MAX_RASTER_PIXELS:
        raise CoreRecoveryResourceError('Core recovery exceeds its nonempty raster limit.')
    return tensors


def _admit(peak, cpu, work):
    if peak > MAX_LIVE_BYTES:
        raise CoreRecoveryResourceError('Core recovery exceeds its combined live CUDA byte limit.')
    if cpu > MAX_CPU_METADATA_BYTES:
        raise CoreRecoveryResourceError('Core recovery exceeds its compact CPU metadata limit.')
    if work > MAX_WORK:
        raise CoreRecoveryResourceError('Core recovery exceeds its bounded work limit.')


def _work(pixels):
    # Conservative visits/comparisons, including bounded probing and heap steps.
    return (pixels*(MAX_ID_PROBES+3*MAX_LABELS.bit_length()+8*pixels.bit_length()+32)
            + ID_TABLE_SLOTS*16+MAX_LABELS*64
            + ((pixels+VALIDATION_TILE_PIXELS-1)//VALIDATION_TILE_PIXELS)*2048)


def _flags(state):
    value = int(state[0].item())
    if value & 31:
        raise ValueError('Require binary 0/255 masks, nonnegative cores inside valid foreground and finite elevation.')
    if value & 96:
        raise CoreRecoveryResourceError('Core IDs exceed the bounded count or lookup-probe domain.')
    if value & ~127:
        raise CoreRecoveryMutationError('Recovered labels violate marker, ID or foreground invariants.')


def _partition_tensors(partition):
    return (partition.original_foreground, partition.markers, partition.validity,
            partition.elevation, partition.labels, partition.domain,
            partition.unresolved, partition.interface, partition.label_ids)


def _partition_snapshot(partition):
    return (tuple(_signature(t) for t in _partition_tensors(partition)),
            tuple(_record_snapshot(obj,CoreRecoveredObject) for obj in partition.objects),
            _record_snapshot(partition.plan,CoreRecoveryPlan),partition.unresolved_pixels,
            partition.interface_pixels, partition.status, partition.profile)


def _record_snapshot(value,record_type):
    if type(value) is not record_type:
        raise CoreRecoveryMutationError('A retained recovery record was replaced.')
    return tuple(getattr(value,field.name) for field in fields(record_type))


def _check_partition(partition):
    if (type(partition) is not CoreRecoveryPartition
            or _partition_snapshot(partition) != partition._snapshot):
        raise CoreRecoveryMutationError('Retained recovery partition or facts changed after publication.')


def _flood(elevation, markers, domain, capacity, guard):
    height, width = domain.shape
    output = torch.empty_like(markers)
    values = torch.empty(capacity, dtype=torch.float32, device=domain.device)
    ages = torch.empty(capacity, dtype=torch.int32, device=domain.device)
    indices = torch.empty_like(ages)
    state = torch.tensor([0, 1, 0, 0], dtype=torch.int32, device=domain.device)
    guard.watch(output, values, ages, indices, state)
    guard.checkpoint()
    program = fixed_kernel_program(_watershed._CUDA_SOURCE, domain.device)
    arguments = (elevation, markers, domain, output, values, ages, indices, state)
    integers = (height, width, capacity, KERNEL_BATCH_PIXELS)
    for _ in range(0, domain.numel(), KERNEL_BATCH_PIXELS):
        guard.checkpoint()
        program.launch('seedvision_ws_init', arguments, integers)
        if int(state[3].item()):
            raise CoreRecoveryResourceError('Core watershed exceeded its checked queue capacity.')
    # Each domain pixel is queued once at most; never use an unbounded host loop.
    for _ in range(0, capacity, KERNEL_BATCH_PIXELS):
        guard.checkpoint()
        if not int(state[0].item()):
            break
        program.launch('seedvision_ws_step', arguments, integers)
        if int(state[3].item()):
            raise CoreRecoveryResourceError('Core watershed exceeded its checked queue capacity.')
    if int(state[0].item()):
        raise CoreRecoveryResourceError('Core watershed exceeded its checked flood-work bound.')
    guard.checkpoint()
    return output


def _summarize(foreground, markers, validity, labels, domain, unresolved,
               interface, band, ids, guard, program, *, publish):
    height, width = labels.shape
    size = ids.numel()
    stats = torch.empty((size, 11), dtype=torch.int32, device=labels.device)
    state = torch.zeros(9, dtype=torch.int32, device=labels.device)
    guard.watch(stats, state)
    guard.checkpoint()
    if size:
        program.launch_1d('recovery_stats_init', (stats,), (size, height, width), size)
    tensors = (foreground, markers, validity, labels, domain, unresolved,
               interface, band, ids, stats, state)
    for start in range(0, labels.numel(), VALIDATION_TILE_PIXELS):
        guard.checkpoint()
        count = min(VALIDATION_TILE_PIXELS, labels.numel()-start)
        program.launch_1d('recovery_summarize', tensors,
                         (height, width, size, start, count, int(publish)), count)
        _flags(state)
    guard.checkpoint()
    host_ids, host_stats, counts = ids.cpu().tolist(), stats.cpu().tolist(), state.cpu().tolist()
    objects = tuple(CoreRecoveredObject(label_id, row[0], row[1], row[2],
        (row[3], row[4], row[5]+1, row[6]+1), bool(row[7]), bool(row[8]), bool(row[9]))
        for label_id, row in zip(host_ids, host_stats))
    if any(o.seed_pixels < 1 or o.assigned_pixels < o.seed_pixels for o in objects):
        raise CoreRecoveryMutationError('Recovered object lost its supplied core.')
    return objects, tuple(row[10] for row in host_stats), counts[6], counts[8]


def recover_core_labels(foreground, markers, validity, elevation, *, cancel=None):
    """Partition valid original foreground from supplied positive core IDs.

    All four inputs are same-frame contiguous CUDA HxW tensors. Masks contain
    only 0/255. IDs may be sparse positive int32 values, with a bounded distinct
    count and lookup domain. A core outside valid foreground is an error.
    Unseeded four-connected components remain unresolved zero. The retained
    original mask is the known union, not evidence of hidden individual outlines.
    """
    tensors = _inputs(foreground, markers, validity, elevation)
    height, width = foreground.shape
    pixels, backing = foreground.numel(), _backing_bytes(tensors)
    work = _work(pixels)
    metadata = 4*ID_TABLE_SLOTS+MAX_LABELS*80+BACKEND_HEADROOM_BYTES
    _admit(backing+metadata, MAX_LABELS*1536+1024*1024, work)
    with torch.cuda.device(foreground.device):
        guard = _Guard(cancel, tensors)
        guard.checkpoint()
        table = torch.zeros(ID_TABLE_SLOTS, dtype=torch.int32, device=foreground.device)
        state = torch.zeros(9, dtype=torch.int32, device=foreground.device)
        guard.watch(table, state)
        program = fixed_kernel_program(_CUDA_SOURCE, foreground.device)
        for start in range(0, pixels, VALIDATION_TILE_PIXELS):
            guard.checkpoint()
            count = min(VALIDATION_TILE_PIXELS, pixels-start)
            program.launch_1d('recovery_validate', tensors+(table, state),
                (start, count, ID_TABLE_SLOTS, MAX_ID_PROBES, MAX_LABELS), count)
            _flags(state)
        counts = state.cpu().tolist()
        domain_pixels, size = counts[1], counts[2]
        capacity = max(1, domain_pixels)
        workspace = 4*pixels+12*capacity+16
        if workspace > _watershed.MAX_WORKSPACE_BYTES:
            raise CoreRecoveryResourceError('Core watershed exceeds its fixed queue workspace limit.')
        peak, cpu = backing+11*pixels+12*capacity+metadata, size*1536+1024*1024
        _admit(peak, cpu, work)
        plan = CoreRecoveryPlan(height, width, counts[4], domain_pixels, counts[4]-domain_pixels, counts[5],
                                size, backing, workspace, peak, cpu, work)
        plan_snapshot=_record_snapshot(plan,CoreRecoveryPlan)
        def check_plan():
            if _record_snapshot(plan,CoreRecoveryPlan)!=plan_snapshot:
                raise CoreRecoveryMutationError('Recovery resource facts changed during execution.')
        guard.extra=check_plan
        unsorted_ids = torch.empty(size, dtype=torch.int32, device=foreground.device)
        guard.watch(unsorted_ids)
        if size:
            guard.checkpoint()
            program.launch_1d('recovery_compact', (table, unsorted_ids, state),
                             (ID_TABLE_SLOTS, size), ID_TABLE_SLOTS)
            _flags(state)
            if int(state[3].item()) != size:
                raise CoreRecoveryMutationError('Core ID inventory changed during compaction.')
            ids = torch.sort(unsorted_ids).values
        else:
            ids = unsorted_ids
        domain = torch.empty_like(foreground)
        unresolved, interface = torch.empty_like(foreground), torch.empty_like(foreground)
        guard.watch(ids, domain, unresolved, interface)
        guard.checkpoint()
        program.launch_1d('recovery_domain', (foreground, validity, domain), (pixels,), pixels)
        labels = _flood(elevation, markers, domain, capacity, guard)
        objects, _, missing, contacts = _summarize(foreground, markers, validity,
            labels, domain, unresolved, interface, interface, ids, guard, program, publish=True)
        if sum(o.assigned_pixels for o in objects)+missing != domain_pixels:
            raise CoreRecoveryMutationError('Recovered coverage does not partition valid foreground.')
        status = ('empty_foreground' if not counts[4] else 'all_foreground_excluded_by_validity' if not domain_pixels
                  else 'unresolved_foreground' if missing else 'partitioned')
        partition = CoreRecoveryPartition(foreground, markers, validity, elevation,
            labels, domain, unresolved, interface, ids, objects, plan,
            missing, contacts, status, PROFILE, ())
        partition_snapshot = _partition_snapshot(partition)
        object.__setattr__(partition, '_snapshot', partition_snapshot)
        def check_final_partition():
            check_plan()
            if partition._snapshot != partition_snapshot:
                raise CoreRecoveryMutationError('Recovery partition snapshot changed during execution.')
            _check_partition(partition)
        guard.extra = check_final_partition
        guard.checkpoint()
        _check_partition(partition)
        return partition


def annotate_core_recovery(partition, *, uncertainty_radius=0,
                           measurement_policy='withhold_uncertain', cancel=None):
    """Annotate cached labels without repartitioning or modifying their tensors.

    Radius zero includes the symmetric shared interface; each additional step
    expands it by one four-neighbor pixel inside the valid foreground domain.
    Policies control permission for computational segmentation measurements.
    Neither policy establishes biological identity, hidden individual outlines
    or scientific validation. An uncontested visible partition means only that
    these input-domain boundary checks found no uncertainty for its supplied ID.
    """
    _check_partition(partition)
    partition_snapshot = partition._snapshot
    if type(uncertainty_radius) is not int or not 0 <= uncertainty_radius <= 8:
        raise ValueError('uncertainty_radius must be an integer from 0 through 8.')
    if (type(measurement_policy) is not str
            or measurement_policy not in ('withhold_uncertain', 'flag_uncertain')):
        raise ValueError('Unknown recovery measurement policy.')
    tensors = _partition_tensors(partition)
    height, width = partition.labels.shape
    pixels, size = partition.labels.numel(), partition.label_ids.numel()
    retained = _backing_bytes(tensors)
    peak = retained+2*pixels+size*44+36+BACKEND_HEADROOM_BYTES
    cpu = size*1536+1024*1024
    work = pixels*(4*uncertainty_radius+3*max(1, size.bit_length())+32)+size*64
    _admit(peak, cpu, work)
    plan = CoreRecoveryAnnotationPlan(retained, peak, cpu, work)
    plan_snapshot=_record_snapshot(plan,CoreRecoveryAnnotationPlan)
    def check_facts():
        if partition._snapshot != partition_snapshot:
            raise CoreRecoveryMutationError('Recovery partition snapshot changed during annotation.')
        _check_partition(partition)
        if _record_snapshot(plan,CoreRecoveryAnnotationPlan)!=plan_snapshot:
            raise CoreRecoveryMutationError('Recovery annotation resource facts changed during execution.')
    with torch.cuda.device(partition.labels.device):
        guard = _Guard(cancel, tensors, check_facts)
        guard.checkpoint()
        band = partition.interface.clone()
        spare = torch.empty_like(band)
        guard.watch(band, spare)
        program = fixed_kernel_program(_CUDA_SOURCE, band.device)
        for _ in range(uncertainty_radius):
            guard.checkpoint()
            program.launch_1d('recovery_dilate', (band, partition.domain, spare),
                             (height, width), pixels)
            band, spare = spare, band
        base, band_counts, missing, contacts = _summarize(partition.original_foreground,
            partition.markers, partition.validity, partition.labels, partition.domain,
            partition.unresolved, partition.interface, band, partition.label_ids,
            guard, program, publish=False)
        if (base != partition.objects or missing != partition.unresolved_pixels
                or contacts != partition.interface_pixels):
            raise CoreRecoveryMutationError('Retained partition facts do not reproduce.')
        objects = []
        for facts, uncertain in zip(base, band_counts):
            reasons = tuple(name for name, yes in (
                ('inferred_shared_boundary', uncertain > 0),
                ('canvas_boundary', facts.touches_canvas),
                ('invalid_support', facts.touches_invalid_support),
                ('unresolved_neighbor', facts.touches_unresolved)) if yes)
            objects.append(CoreRecoveryMeasurementStatus(facts.label_id, uncertain,
                reasons, not reasons or measurement_policy == 'flag_uncertain',
                not reasons, False))
        objects = tuple(objects)
        object_snapshot=tuple(_record_snapshot(obj,CoreRecoveryMeasurementStatus) for obj in objects)
        def check_final_facts():
            check_facts()
            if tuple(_record_snapshot(obj,CoreRecoveryMeasurementStatus) for obj in objects)!=object_snapshot:
                raise CoreRecoveryMutationError('Recovery annotation facts changed during execution.')
        guard.extra=check_final_facts
        guard.checkpoint()
        snapshot = (partition._snapshot, _signature(band), uncertainty_radius,
                    measurement_policy, object_snapshot,plan_snapshot)
        return CoreRecoveryAnnotation(partition, band, uncertainty_radius,
                                      measurement_policy, objects, plan, snapshot)
