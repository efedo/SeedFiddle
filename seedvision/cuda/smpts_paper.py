"""Executable, explicitly assumed SMPTS paper reconstruction on CUDA.

This does not reproduce the unavailable original Code Ocean capsule. Eight-
connectivity, axis-aligned boxes, strict box-area screening, replicated borders,
square single-pass erosion and masked crops are declared reconstruction choices.
The paper supports the branch topology and default window sizes25/5/13.
Only bounded per-component records cross to CPU; crop pixels remain packed on
CUDA. No recovered full outline or validated biological measurement is implied.
Guards detect ordinary PyTorch tensor/storage mutation; external raw-pointer
writes are outside this contract. Work admission is not an elapsed-time promise.
"""
from collections.abc import Mapping
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType

import torch

from ._nvrtc import fixed_kernel_program
from . import plantcv_thresholds as thresholds

MAX_PIXELS = 64_000_000
MAX_COMPONENTS = 4096
MAX_CUDA_BYTES = 1024*1024*1024
MAX_CROP_BYTES = 128*1024*1024
MAX_CPU_METADATA_BYTES = 32*1024*1024
BACKEND_BYTES = 64*1024*1024
MAX_COMPONENT_WORK = 32_000_000_000
MAX_WINDOW_WORK = 1_000_000_000
TILE = 65_536
UNION_ROUNDS = 64
JUMPS_PER_ROUND = 4
FINAL_JUMPS = 26
MAX_JUMP_PASSES = 512
PROFILE = 'smpts-paper-fixed-mean; assumed-ccl8-box-area-replicate-packed-v1'
PAPER_SHA256 = '9b7773b86257b7e2bac8860066033d95380abe736623e5608fc4c056b967dd19'
_CUDA_SOURCE = Path(__file__).with_name('_smpts_paper.cu').read_text(encoding='utf-8')
CUDA_SOURCE_SHA256 = sha256(_CUDA_SOURCE.encode('utf-8')).hexdigest()


class PaperResourceError(ValueError):
    pass


class PaperCancelled(RuntimeError):
    pass


class PaperMutationError(ValueError):
    pass


def _signature(tensor):
    storage = tensor.untyped_storage()
    return (id(tensor), tensor._version, tensor.data_ptr(), storage.data_ptr(),
            storage.nbytes(), tuple(tensor.shape), tuple(tensor.stride()),
            tensor.storage_offset(), tensor.dtype, tensor.device, tensor.layout)


def _backing_bytes(tensors):
    stores = {}
    for tensor in tensors:
        storage = tensor.untyped_storage()
        stores[(tensor.device, storage.data_ptr())] = storage.nbytes()
    return sum(stores.values())


class _Guard:
    def __init__(self, cancel, *tensors):
        if cancel is not None and not callable(cancel):
            raise TypeError('cancel must be callable or None.')
        self.cancel, self.watched = cancel, []
        self.watch(*tensors)

    def watch(self, *tensors):
        for tensor in tensors:
            tensor.record_stream(torch.cuda.current_stream(tensor.device))
            self.watched.append((tensor, _signature(tensor)))

    def check(self):
        for tensor, signature in self.watched:
            if _signature(tensor) != signature:
                raise PaperMutationError('Paper reconstruction input or retained storage changed.')

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
                raise PaperCancelled('Paper reconstruction cancelled before publication.')

    def nested_cancel(self):
        self.checkpoint()
        return False


def _integer(value, name, minimum, maximum):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f'{name} must be an integer in [{minimum}, {maximum}].')
    return value


def _mask(mask):
    if (not torch.is_tensor(mask) or mask.layout != torch.strided
            or mask.device.type != 'cuda' or mask.dtype != torch.uint8
            or mask.ndim != 2 or not mask.is_contiguous()):
        raise ValueError('Require a contiguous CUDA uint8 HW raster.')
    component_work(mask.numel())
    return mask


def _same_frame(a, b, dtype):
    if (not torch.is_tensor(b) or b.layout != torch.strided or b.device != a.device
            or b.dtype != dtype or b.shape != a.shape or not b.is_contiguous()):
        raise ValueError('Require contiguous same-device, same-frame rasters.')


def component_work(pixels):
    _integer(pixels, 'pixels', 1, MAX_PIXELS)
    # Four pointer reads/writes per jump; four edges with bounded atomic accesses.
    # Admission units count conservative scalar array/atomic visits, not seconds
    # or CUDA instructions. Includes initialization, verification, scan and stats.
    work = pixels*(4*MAX_JUMP_PASSES+UNION_ROUNDS*(24+12)+48)
    if work > MAX_COMPONENT_WORK:
        raise PaperResourceError('Paper component labeling exceeds its bounded work limit.')
    return work


def retained_bound(pixels):
    component_work(pixels)
    # Graph-retained rasters + one CCL/scan phase + replacement scratch, all
    # packed crops and a fixed backend allowance. Caller backing excess is added
    # by the graph runner. Caller-held previous results are outside cache ownership.
    bound = 64*pixels+MAX_CROP_BYTES+BACKEND_BYTES
    if bound > MAX_CUDA_BYTES:
        raise PaperResourceError('Paper graph exceeds its composed CUDA byte limit.')
    return bound


def metadata_bound(count):
    _integer(count, 'component count', 0, MAX_COMPONENTS)
    # Covers simultaneous source/final/crop records and compact stats copies.
    # CUDA-backed tensor wrappers/cache records share this modeled host allowance.
    bound = 1024*1024+4096*count
    if bound > MAX_CPU_METADATA_BYTES:
        raise PaperResourceError('Paper compact metadata exceeds its host byte limit.')
    return bound


def _admit(tensors, extra):
    if _backing_bytes(tensors)+extra > MAX_CUDA_BYTES:
        raise PaperResourceError('Paper stage exceeds its live CUDA byte limit.')


def _program(mask):
    if Path(__file__).with_name('_smpts_paper.cu').read_text(encoding='utf-8') != _CUDA_SOURCE:
        raise PaperMutationError('Loaded paper CUDA source no longer matches its file.')
    return fixed_kernel_program(_CUDA_SOURCE, mask.device)


def _tiles(program, name, tensors, integers, pixels, guard):
    for start in range(0, pixels, TILE):
        guard.checkpoint()
        count = min(TILE, pixels-start)
        program.launch_1d(name, tensors, (*integers, start, count), count)


def _state(program, device):
    state = torch.empty(1, dtype=torch.int32, device=device)
    program.launch('paper_reset', (state,), ())
    return state


def _valid_state(state, guard, message):
    value = int(state.item())
    guard.checkpoint()
    if value:
        raise ValueError(message)


def binary_median(mask, ksize=5, *, cancel=None):
    """Exact binary majority with a replicated border, including size1."""
    mask = _mask(mask)
    _integer(ksize, 'ksize', 1, 31)
    if not ksize % 2:
        raise ValueError('Median ksize must be odd.')
    if mask.numel()*ksize*ksize > MAX_WINDOW_WORK:
        raise PaperResourceError('Paper median exceeds its window-sample limit.')
    _admit((mask,), mask.numel()+BACKEND_BYTES)
    with torch.cuda.device(mask.device), torch.no_grad():
        guard = _Guard(cancel, mask)
        guard.checkpoint()
        program = _program(mask)
        result = torch.empty_like(mask)
        state = _state(program, mask.device)
        guard.watch(result, state)
        _tiles(program, 'paper_median', (mask, result, state), (*mask.shape, ksize), mask.numel(), guard)
        _valid_state(state, guard, 'Median input must contain only0/255.')
        return result


def intersection(a, b, *, cancel=None):
    a = _mask(a)
    _same_frame(a, b, torch.uint8)
    _admit((a, b), a.numel()+BACKEND_BYTES)
    with torch.cuda.device(a.device), torch.no_grad():
        guard = _Guard(cancel, a, b)
        guard.checkpoint()
        program = _program(a)
        result, state = torch.empty_like(a), _state(program, a.device)
        guard.watch(result, state)
        _tiles(program, 'paper_intersection', (a, b, result, state), (), a.numel(), guard)
        _valid_state(state, guard, 'Intersection inputs must contain only0/255.')
        return result


def _statistics(labels, count, program, guard):
    height, width = labels.shape
    stats = torch.empty((count+1, 5), dtype=torch.int32, device=labels.device)
    state = _state(program, labels.device)
    guard.watch(stats, state)
    program.launch_1d('paper_stats_initialize', (stats,), (count+1, height, width), count+1)
    _tiles(program, 'paper_stats', (labels, stats, state), (count+1, width), labels.numel(), guard)
    _valid_state(state, guard, 'Component IDs exceed their declared inventory.')
    rows = stats[1:].cpu().tolist()  # At most4096 rows, five bounded int32 facts.
    guard.checkpoint()
    if any(row[0] <= 0 for row in rows):
        raise ValueError('Component inventory contains an absent ID.')
    return tuple(MappingProxyType({'label_id': i+1, 'area_pixels': row[0],
        'bounds_xyxy': tuple(row[1:])}) for i, row in enumerate(rows))


def components(mask, *, cancel=None):
    """Bounded CCL8 with canonical IDs in first-foreground-pixel order."""
    mask = _mask(mask)
    pixels = mask.numel()
    _admit((mask,), 24*pixels+BACKEND_BYTES+MAX_COMPONENTS*32)
    with torch.cuda.device(mask.device), torch.no_grad():
        guard = _Guard(cancel, mask)
        guard.checkpoint()
        program = _program(mask)
        parent = torch.empty(mask.shape, dtype=torch.int32, device=mask.device)
        other = torch.empty_like(parent)
        state = _state(program, mask.device)
        flags = torch.empty(mask.shape, dtype=torch.uint8, device=mask.device)
        guard.watch(parent, other, state, flags)
        _tiles(program, 'paper_initialize', (mask, parent, state), (), pixels, guard)
        _valid_state(state, guard, 'Component input must contain only0/255.')
        jumps = 0
        def compress(passes):
            nonlocal parent, other, jumps
            if jumps+passes > MAX_JUMP_PASSES:
                raise PaperResourceError('Paper components exceeded512 pointer-jump passes.')
            for _ in range(passes):
                _tiles(program, 'paper_jump', (parent, other), (), pixels, guard)
                parent, other = other, parent
            jumps += passes
        for _ in range(UNION_ROUNDS):
            compress(JUMPS_PER_ROUND)
            program.launch('paper_reset', (state,), ())
            _tiles(program, 'paper_hook', (parent, state), tuple(mask.shape), pixels, guard)
            changed = int(state.item())
            guard.checkpoint()
            if not changed:
                # Approximate-root hooks can have a false fixed point. Fully
                # compress, verify every edge, then resume bounded union rounds
                # if necessary instead of rejecting a resolvable component.
                compress(FINAL_JUMPS)
                program.launch('paper_reset', (state,), ())
                _tiles(program, 'paper_verify_roots', (parent, flags, state), tuple(mask.shape), pixels, guard)
                failed = int(state.item())
                guard.checkpoint()
                if not failed:
                    break
        else:
            raise PaperResourceError('Paper components did not converge within64 union rounds.')
        prefix = torch.cumsum(flags.reshape(-1), dim=0, dtype=torch.int32)
        guard.watch(prefix)
        count = int(prefix[-1].item())
        guard.checkpoint()
        if count > MAX_COMPONENTS:
            raise PaperResourceError('Paper component inventory exceeds4096 objects.')
        metadata_bound(count)
        labels = torch.empty_like(parent)
        guard.watch(labels)
        _tiles(program, 'paper_compact', (parent, prefix, labels), (), pixels, guard)
        table = _statistics(labels, count, program, guard)
        guard.checkpoint()
        return {'labels': labels, 'components': table}


def _inventory(value, shape):
    if type(value) is not tuple or len(value) > MAX_COMPONENTS:
        raise ValueError('Require a bounded immutable component inventory.')
    height, width = shape
    for i, row in enumerate(value):
        if not isinstance(row, Mapping) or set(row) != {'label_id', 'area_pixels', 'bounds_xyxy'}:
            raise ValueError('Component record has an invalid schema.')
        bounds = row['bounds_xyxy']
        if (type(row['label_id']) is not int or row['label_id'] != i+1
                or type(bounds) is not tuple or len(bounds) != 4
                or any(type(v) is not int for v in bounds)):
            raise ValueError('Component IDs/bounds must be canonical integer records.')
        x0, y0, x1, y1 = bounds
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError('Component bounds exceed the source frame.')
        _integer(row['area_pixels'], 'area_pixels', 1, (x1-x0)*(y1-y0))
    return value


def select_contacts(inventory, shape, contact_box_area):
    """Declare suspected contacts by strict axis-aligned box area comparison."""
    inventory = _inventory(inventory, shape)
    _integer(contact_box_area, 'contact_box_area', 0, MAX_PIXELS)
    return tuple(row['label_id'] for row in inventory
        if (row['bounds_xyxy'][2]-row['bounds_xyxy'][0])
        *(row['bounds_xyxy'][3]-row['bounds_xyxy'][1]) > contact_box_area)


def targeted_erosion(mask, labels, inventory, selected_ids, ksize=13, *, cancel=None):
    mask = _mask(mask)
    _same_frame(mask, labels, torch.int32)
    inventory = _inventory(inventory, mask.shape)
    if (type(selected_ids) is not tuple or any(type(i) is not int or not 1 <= i <= len(inventory)
            for i in selected_ids) or tuple(sorted(set(selected_ids))) != selected_ids):
        raise ValueError('Selected contact IDs must be unique, sorted inventory IDs.')
    _integer(ksize, 'ksize', 1, 31)
    if not ksize % 2:
        raise ValueError('Erosion ksize must be odd.')
    if 2*ksize*mask.numel() > MAX_WINDOW_WORK:
        raise PaperResourceError('Paper erosion exceeds its axis-sample limit.')
    _admit((mask, labels), 4*mask.numel()+BACKEND_BYTES+MAX_COMPONENTS*32)
    with torch.cuda.device(mask.device), torch.no_grad():
        guard = _Guard(cancel, mask, labels)
        guard.checkpoint()
        program = _program(mask)
        actual = _statistics(labels, len(inventory), program, guard)
        if actual != inventory:
            raise ValueError('Component records do not describe the supplied labels.')
        lookup = torch.tensor([False]+[i in selected_ids for i in range(1, len(inventory)+1)],
            dtype=torch.uint8, device=mask.device)
        target, horizontal, eroded, result = (torch.empty_like(mask) for _ in range(4))
        state = _state(program, mask.device)
        guard.watch(lookup, target, horizontal, eroded, result, state)
        _tiles(program, 'paper_select', (mask, labels, lookup, target, state),
            (len(inventory)+1,), mask.numel(), guard)
        _valid_state(state, guard, 'Source labels/mask disagree or exceed their inventory.')
        _tiles(program, 'paper_erode_axis', (target, horizontal), (*mask.shape, ksize, 1), mask.numel(), guard)
        _tiles(program, 'paper_erode_axis', (horizontal, eroded), (*mask.shape, ksize, 0), mask.numel(), guard)
        _tiles(program, 'paper_merge', (mask, labels, lookup, eroded, result), (), mask.numel(), guard)
        guard.checkpoint()
        return {'mask': result, 'selected_mask': target, 'eroded_selected_mask': eroded}


def pack_crops(rgb, source_labels, source_inventory, final_labels, final_inventory, *, cancel=None):
    if (not torch.is_tensor(rgb) or rgb.layout != torch.strided or rgb.device.type != 'cuda'
            or rgb.dtype != torch.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3
            or not rgb.is_contiguous()):
        raise ValueError('Crops require a contiguous CUDA RGB uint8 image.')
    shape = tuple(rgb.shape[:2])
    component_work(shape[0]*shape[1])
    # A HW view allows exact dtype/frame checks without allocating any raster.
    _same_frame(rgb[:, :, 0], source_labels, torch.int32)
    _same_frame(source_labels, final_labels, torch.int32)
    source_inventory = _inventory(source_inventory, shape)
    final_inventory = _inventory(final_inventory, shape)
    total = sum((r['bounds_xyxy'][2]-r['bounds_xyxy'][0])
        *(r['bounds_xyxy'][3]-r['bounds_xyxy'][1]) for r in final_inventory)
    if 4*total > MAX_CROP_BYTES:
        raise PaperResourceError('Packed crops exceed the aggregate128MiB byte limit.')
    _admit((rgb, source_labels, final_labels), 4*total+BACKEND_BYTES+MAX_COMPONENTS*64)
    with torch.cuda.device(rgb.device), torch.no_grad():
        guard = _Guard(cancel, rgb, source_labels, final_labels)
        guard.checkpoint()
        program = _program(source_labels)
        for labels, inventory in ((source_labels, source_inventory), (final_labels, final_inventory)):
            if _statistics(labels, len(inventory), program, guard) != inventory:
                raise ValueError('Crop component records do not describe their labels.')
        slots, source_slots = len(final_inventory)+1, len(source_inventory)+1
        minimum = torch.empty(slots, dtype=torch.int32, device=rgb.device)
        maximum = torch.empty_like(minimum)
        state = _state(program, rgb.device)
        guard.watch(minimum, maximum, state)
        program.launch_1d('paper_parent_initialize', (minimum, maximum), (slots, source_slots), slots)
        _tiles(program, 'paper_parents', (source_labels, final_labels, minimum, maximum, state),
            (slots, source_slots), source_labels.numel(), guard)
        _valid_state(state, guard, 'Final labels contain unsupported source pixels or invalid IDs.')
        low, high = minimum[1:].cpu().tolist(), maximum[1:].cpu().tolist()
        guard.checkpoint()
        if any(a != b or not 1 <= a < source_slots for a, b in zip(low, high)):
            raise ValueError('A final component crosses source-parent boundaries.')
        packed_rgb = torch.empty(3*total, dtype=torch.uint8, device=rgb.device)
        packed_mask = torch.empty(total, dtype=torch.uint8, device=rgb.device)
        guard.watch(packed_rgb, packed_mask)
        rows, offset = [], 0
        for row, parent in zip(final_inventory, low):
            x0, y0, x1, y1 = row['bounds_xyxy']
            size = (x1-x0)*(y1-y0)
            _tiles(program, 'paper_crop', (rgb, final_labels, packed_rgb, packed_mask),
                (shape[1], row['label_id'], x0, y0, x1-x0, offset), size, guard)
            rows.append(MappingProxyType(dict(row, parent_id=parent, offset_pixels=offset,
                crop_shape=(y1-y0, x1-x0))))
            offset += size
        present = set(low)
        guard.checkpoint()
        return {'packed_rgb': packed_rgb, 'packed_mask': packed_mask, 'crops': tuple(rows),
            'disappeared_parent_ids': tuple(i for i in range(1, source_slots) if i not in present)}


def extract_from_intersection(mask, rgb, contact_box_area=10_000, erosion_ksize=13, *, cancel=None):
    """Execute screening, targeted erosion, relocalization and packed extraction."""
    mask = _mask(mask)
    retained_bound(mask.numel())
    _integer(contact_box_area, 'contact_box_area', 0, MAX_PIXELS)
    _integer(erosion_ksize, 'erosion_ksize', 1, 31)
    if not erosion_ksize % 2:
        raise ValueError('Erosion ksize must be odd.')
    if 2*erosion_ksize*mask.numel() > MAX_WINDOW_WORK:
        raise PaperResourceError('Paper erosion exceeds its axis-sample limit.')
    if (not torch.is_tensor(rgb) or rgb.device != mask.device or rgb.dtype != torch.uint8
            or tuple(rgb.shape) != (*mask.shape, 3) or not rgb.is_contiguous()):
        raise ValueError('Original RGB must be contiguous and bound to the mask frame/device.')
    extra_backing = max(0, _backing_bytes((mask, rgb))-4*mask.numel())
    if retained_bound(mask.numel())+extra_backing > MAX_CUDA_BYTES:
        raise PaperResourceError('Paper graph input backing exceeds its composed CUDA limit.')
    with torch.cuda.device(mask.device), torch.no_grad():
        guard = _Guard(cancel, mask, rgb)
        guard.checkpoint()
        source = components(mask, cancel=guard.nested_cancel)
        guard.watch(source['labels'])
        selected = select_contacts(source['components'], mask.shape, contact_box_area)
        erosion = targeted_erosion(mask, source['labels'], source['components'], selected,
            erosion_ksize, cancel=guard.nested_cancel)
        guard.watch(*erosion.values())
        final = components(erosion['mask'], cancel=guard.nested_cancel)
        guard.watch(final['labels'])
        crops = pack_crops(rgb, source['labels'], source['components'], final['labels'],
            final['components'], cancel=guard.nested_cancel)
        guard.watch(crops['packed_rgb'], crops['packed_mask'])
        guard.checkpoint()
        return dict(source_labels=source['labels'], source_components=source['components'],
            selected_parent_ids=selected, final_labels=final['labels'], final_components=final['components'],
            **erosion, **crops)


def reconstruct(gray, rgb, *, adaptive_ksize=25, adaptive_offset=2.0,
        object_type='light', global_threshold=100, median_ksize=5,
        contact_box_area=10_000, erosion_ksize=13, cancel=None):
    """Run the fixed-global/mean variant, with every selected control active."""
    gray = _mask(gray)  # Gray values are unrestricted; binary validation follows thresholding.
    if (not torch.is_tensor(rgb) or rgb.layout != torch.strided or rgb.device != gray.device
            or rgb.dtype != torch.uint8 or tuple(rgb.shape) != (*gray.shape, 3)
            or not rgb.is_contiguous()):
        raise ValueError('Require contiguous same-frame CUDA RGB uint8 input.')
    extra_backing = max(0, _backing_bytes((gray, rgb))-4*gray.numel())
    if retained_bound(gray.numel())+extra_backing > MAX_CUDA_BYTES:
        raise PaperResourceError('Paper graph input backing exceeds its composed CUDA limit.')
    # Reject invalid controls before any full-frame branch allocation.
    thresholds._validate_mean_parameters(adaptive_ksize, adaptive_offset)
    thresholds._object_type(object_type)
    _integer(global_threshold, 'global_threshold', 0, 255)
    _integer(median_ksize, 'median_ksize', 1, 31)
    _integer(erosion_ksize, 'erosion_ksize', 1, 31)
    _integer(contact_box_area, 'contact_box_area', 0, MAX_PIXELS)
    if not median_ksize % 2 or not erosion_ksize % 2:
        raise ValueError('Median and erosion windows must be odd.')
    if max(gray.numel()*median_ksize**2, 2*erosion_ksize*gray.numel()) > MAX_WINDOW_WORK:
        raise PaperResourceError('Paper graph exceeds its selected window-sample limit.')
    with torch.cuda.device(gray.device), torch.no_grad():
        guard = _Guard(cancel, gray, rgb)
        guard.checkpoint()
        adaptive = thresholds.threshold_mean(gray, adaptive_ksize, adaptive_offset, object_type)
        guard.watch(adaptive)
        guard.checkpoint()
        median = binary_median(adaptive, median_ksize, cancel=guard.nested_cancel)
        guard.watch(median)
        fixed = thresholds.threshold_binary(gray, global_threshold, object_type)
        guard.watch(fixed)
        combined = intersection(median, fixed, cancel=guard.nested_cancel)
        guard.watch(combined)
        extracted = extract_from_intersection(combined, rgb, contact_box_area, erosion_ksize,
            cancel=guard.nested_cancel)
        guard.watch(*(v for v in extracted.values() if torch.is_tensor(v)))
        guard.checkpoint()
        return dict(adaptive_mask=adaptive, median_mask=median, global_mask=fixed,
            intersection_mask=combined, **extracted)
