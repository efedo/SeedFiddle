"""Source-ordered CUDA image/mask samples for PlantCV 4.11.3 Bayes training.

Full images, HSV, mask populations and selected values remain on CUDA. Only
bounded per-tile counts and the source's RandomState indices/state use CPU.
The three output columns are sampled independently, as in the public trainer.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import struct

import numpy as np
import torch

from seedvision.cuda import _nvrtc
from seedvision.cuda import plantcv_range as _range
from seedvision.cuda.plantcv_naive_bayes_training import (
    MAX_SAMPLES, PLANTCV_TRAIN_SHA256, _tensor_signature)

MAX_IMAGE_PIXELS = 16_000_000
MAX_CUDA_LIVE_BYTES = 256 * 1024 * 1024
MAX_CPU_METADATA_BYTES = 16 * 1024 * 1024
MAX_WORK = 2_000_000_000
MAX_TILE_PIXELS = 65_536
MASK_TILE_PIXELS = 4096
CUDA_HEADROOM_BYTES = 4 * 1024 * 1024
CPU_HEADROOM_BYTES = 128 * 1024
RANDOM_INDEX_DTYPE = 'l'

_CUDA_SOURCE = r'''
extern "C" __global__ void count_tiles(const unsigned char* mask, int* counts,
    int pixels, int mask_width, int image_height, int image_width,
    int tile_size, int tiles) {
    int t=blockIdx.x*blockDim.x+threadIdx.x;
    if(t>=tiles) return;
    int fg=0,bg=0,bad_fg=0,bad_bg=0,stop=min(pixels,(t+1)*tile_size);
    for(int i=t*tile_size;i<stop;++i) {
        bool outside=i/mask_width>=image_height || i%mask_width>=image_width;
        fg+=mask[i]==255; bg+=mask[i]==0;
        bad_fg+=(mask[i]==255 && outside); bad_bg+=(mask[i]==0 && outside);
    }
    counts[4*t]=fg; counts[4*t+1]=bg;
    counts[4*t+2]=bad_fg; counts[4*t+3]=bad_bg;
}
extern "C" __global__ void positions(const unsigned char* mask, const int* offsets,
    int* foreground, int* background, int pixels, int mask_width, int image_width,
    int tile_size, int tiles) {
    int t=blockIdx.x*blockDim.x+threadIdx.x;
    if(t>=tiles) return;
    int fg=offsets[4*t],bg=offsets[4*t+1],stop=min(pixels,(t+1)*tile_size);
    for(int i=t*tile_size;i<stop;++i) {
        // Categorical bounds were audited before launching this kernel.
        if(mask[i]==255) foreground[fg++]=(i/mask_width)*image_width+i%mask_width;
        else if(mask[i]==0) background[bg++]=(i/mask_width)*image_width+i%mask_width;
    }
}
extern "C" __global__ void gather_channel(const unsigned char* hsv,
    const int* population, const long long* ordinals, unsigned char* output,
    int start, int rows, int channel) {
    int r=blockIdx.x*blockDim.x+threadIdx.x;
    if(r>=rows) return;
    int pixel=population[ordinals[r]];
    output[(long long)(start+r)*3+channel]=hsv[(long long)pixel*3+channel];
}
'''


class BayesSamplingError(ValueError):
    """Unsupported image/mask, mutation or bounded-resource domain."""


class BayesSamplingSourceError(BayesSamplingError):
    """The public sequence reaches invalid mask coordinates or randint bounds."""


class BayesSamplingCancelled(RuntimeError):
    """Cancelled without publishing samples; consumed RNG draws stay consumed."""


@dataclass(frozen=True, slots=True)
class BayesRandomState:
    algorithm: str
    keys: bytes
    position: int
    has_gauss: int
    cached_gaussian: bytes

    @property
    def sha256(self):
        return sha256(self.algorithm.encode('ascii')+b'\0'+self.keys+
            struct.pack('<II', self.position, self.has_gauss)+self.cached_gaussian).hexdigest()


@dataclass(frozen=True, slots=True)
class BayesImageSamplingPlan:
    height: int
    width: int
    mask_height: int
    mask_width: int
    foreground_pixels: int
    background_pixels: int
    samples_per_class: int
    mask_tiles: int
    conversion_tile: int
    gather_tile: int
    output_bytes: int
    cpu_metadata_bytes: int
    work: int
    peak_live_bytes: int


@dataclass(frozen=True, slots=True)
class BayesImageSamples:
    samples: tuple[torch.Tensor, torch.Tensor]
    plan: BayesImageSamplingPlan
    rng_before: BayesRandomState
    rng_after: BayesRandomState
    draws_sha256: str
    random_index_dtype: str
    random_calls: int = 6
    class_names: tuple[str, str] = ('plant', 'background')


@dataclass(frozen=True, slots=True)
class _JobBudget:
    """Internal directory-job reservations; these are not recipe controls."""
    retained_backing_bytes: int = 0
    retained_rows: int = 0
    consumed_work: int = 0

    def validate(self):
        for value, maximum in ((self.retained_backing_bytes, MAX_CUDA_LIVE_BYTES),
                (self.retained_rows, MAX_SAMPLES), (self.consumed_work, MAX_WORK)):
            if type(value) is not int or not 0 <= value <= maximum:
                raise BayesSamplingError('invalid cumulative Bayes sampling budget')


def _state(rng):
    algorithm, keys, position, has_gauss, cached = rng.get_state()
    if (algorithm != 'MT19937' or keys.dtype != np.uint32 or keys.shape != (624,)
            or not 0 <= position <= 624 or has_gauss not in (0, 1)):
        raise BayesSamplingError('require the source legacy MT19937 RandomState layout')
    return BayesRandomState(algorithm, keys.astype('<u4', copy=False).tobytes(),
        int(position), int(has_gauss), struct.pack('<d', cached))


class _Guard:
    def __init__(self, rng, cancel, tensors):
        self.rng, self.cancel = rng, cancel
        self.tensors = {}
        self.watch(*tensors)
        self.expected_rng = _state(rng)

    def watch(self, *tensors):
        for tensor in tensors:
            self.tensors[id(tensor)] = tensor, _tensor_signature(tensor)

    def check(self):
        if any(_tensor_signature(x) != signature for x, signature in self.tensors.values()):
            raise BayesSamplingError('Bayes sampling tensor changed during calculation or callback')
        if _state(self.rng) != self.expected_rng:
            raise BayesSamplingError('Bayes sampling RandomState changed outside its ordered draws')

    def checkpoint(self):
        self.check()
        try:
            answer = self.cancel() if self.cancel is not None else False
        finally:
            self.check()
        if type(answer) is not bool:
            raise BayesSamplingError('cancel callback must return bool')
        if answer:
            raise BayesSamplingCancelled('Bayes image sampling cancelled; consumed draws are not rolled back')


def _raster_work(pixels, mask_pixels, tiles, conversion_tile):
    conversions = (pixels+conversion_tile-1)//conversion_tile
    # Charge repeated reciprocal-table creation/upload and conversion setup,
    # so tiny tiles cannot bypass the whole-job work cap.
    # Image conversion and both categorical mask passes have separate sizes.
    # This is a conservative visit/setup metric, not an instruction count.
    return pixels*96+mask_pixels*16+tiles*16+conversions*2048


def _input_domain(bgr, mask, conversion_tile, gather_tile, budget=None):
    budget = _JobBudget() if budget is None else budget
    if type(budget) is not _JobBudget:
        raise BayesSamplingError('invalid cumulative Bayes sampling budget')
    budget.validate()
    if (not torch.is_tensor(bgr) or not torch.is_tensor(mask)
            or bgr.device.type != 'cuda' or mask.device != bgr.device
            or bgr.layout != torch.strided or mask.layout != torch.strided
            or bgr.dtype != torch.uint8 or mask.dtype != torch.uint8
            or bgr.ndim != 3 or bgr.shape[-1] != 3 or mask.ndim != 2
            or not bgr.is_contiguous() or not mask.is_contiguous()):
        raise BayesSamplingError('require contiguous CUDA uint8 BGR HWC and mask HW')
    height, width = map(int, bgr.shape[:2])
    mask_height, mask_width = map(int, mask.shape)
    pixels, mask_pixels = height*width, mask_height*mask_width
    if (not 1 <= pixels <= MAX_IMAGE_PIXELS
            or not 1 <= mask_pixels <= MAX_IMAGE_PIXELS):
        raise BayesSamplingError('Bayes sampling image/mask is empty or exceeds the pixel cap')
    for value in (conversion_tile, gather_tile):
        if type(value) is not int or not 1 <= value <= MAX_TILE_PIXELS:
            raise BayesSamplingError('conversion/gather tiles must be bounded positive integers')
    storages = {(x.device, x.untyped_storage().data_ptr()): x.untyped_storage().nbytes()
        for x in (bgr, mask)}
    backing = sum(storages.values())
    tiles = (mask_pixels+MASK_TILE_PIXELS-1)//MASK_TILE_PIXELS
    # First phase allocates only explicit compact counts. No PyTorch nonzero/
    # prefix-scan workspace or full CPU mask is needed.
    if budget.retained_backing_bytes+backing+tiles*16+CUDA_HEADROOM_BYTES > MAX_CUDA_LIVE_BYTES:
        raise BayesSamplingError('Bayes sampling input/count storage exceeds the CUDA bound')
    if (tiles*80+CPU_HEADROOM_BYTES > MAX_CPU_METADATA_BYTES
            or budget.consumed_work+_raster_work(pixels, mask_pixels, tiles,
                min(conversion_tile, pixels)) > MAX_WORK):
        raise BayesSamplingError('Bayes sampling count metadata/work exceeds its bound')
    return height, width, mask_height, mask_width, tiles, backing


def _plan(domain, foreground, background, conversion_tile, gather_tile, budget=None):
    budget = _JobBudget() if budget is None else budget
    if type(budget) is not _JobBudget:
        raise BayesSamplingError('invalid cumulative Bayes sampling budget')
    budget.validate()
    height, width, mask_height, mask_width, tiles, backing = domain
    pixels, mask_pixels = height*width, mask_height*mask_width
    count = foreground//10
    if budget.retained_rows+2*count > MAX_SAMPLES:
        raise BayesSamplingError('Bayes sampling exceeds the combined KDE sample cap')
    conversion = min(conversion_tile, pixels)
    gather = min(gather_tile, count)
    output = 6*count
    cpu = count*24+tiles*80+CPU_HEADROOM_BYTES
    gathers = 0 if not count else 6*((count+gather-1)//gather)
    work = _raster_work(pixels, mask_pixels, tiles, conversion)+count*12+gathers*256
    # Explicit int32 mapped population positions total at most4M, HSV3P,
    # two four-column count/offset tables32T. Conversion uses bounded scratch;
    # only a single ordinal-index tile is uploaded for each gather.
    live = (backing+pixels*3+4*(foreground+background)+tiles*32+output+
        conversion*128+gather*16+CUDA_HEADROOM_BYTES)
    if (cpu > MAX_CPU_METADATA_BYTES or budget.consumed_work+work > MAX_WORK
            or budget.retained_backing_bytes+live > MAX_CUDA_LIVE_BYTES):
        raise BayesSamplingError('Bayes sampling exceeds the CPU metadata/work/live CUDA bound')
    return BayesImageSamplingPlan(height, width, mask_height, mask_width,
        foreground, background, count,
        tiles, conversion, gather, output, cpu, work, live)


def sample_naive_bayes_image(bgr, mask, *, rng=None, cancel=None,
                            conversion_tile=16_384, gather_tile=16_384, _budget=None):
    """Sample plant/background exactly in Hfg,Hbg,Sfg,Sbg,Vfg,Vbg order.

    None uses the public global NumPy RandomState; an explicit legacy
    RandomState can replay a captured source state. There is no invented seed.
    The public upper bound is population-1 (exclusive), with replacement.
    Zero-row samples are valid here; the KDE fitter separately requires >=2.
    Mask rows/columns address the image; ignored mask values need not be in
    bounds. Categorical out-of-bounds coordinates fail before this pair's RNG.
    Cancellation/source failures do not roll back already consumed RNG draws.
    """
    if cancel is not None and not callable(cancel):
        raise BayesSamplingError('cancel must be callable or None')
    if rng is not None and type(rng) is not np.random.RandomState:
        raise BayesSamplingError('rng must be a legacy RandomState or None for the public global state')
    # Copy validated immutable counters before any caller callback. The plan
    # still describes this image; the directory owner records cumulative work.
    budget = _JobBudget() if _budget is None else _budget
    if type(budget) is not _JobBudget:
        raise BayesSamplingError('invalid cumulative Bayes sampling budget')
    budget.validate()
    budget = _JobBudget(budget.retained_backing_bytes, budget.retained_rows, budget.consumed_work)
    domain = _input_domain(bgr, mask, conversion_tile, gather_tile, budget)
    rng = np.random if rng is None else rng
    guard = _Guard(rng, cancel, (bgr, mask))
    before = guard.expected_rng
    # A shadow predicts the exact six calls, detecting concurrent RNG writes
    # even between a callback check and a draw. Its seed is overwritten before
    # use and does not set/reseed the caller's source stream.
    shadow = np.random.RandomState(0)
    shadow.set_state(rng.get_state())
    guard.checkpoint()
    height, width, mask_height, mask_width, tiles, _ = domain
    pixels, mask_pixels = height*width, mask_height*mask_width
    with torch.cuda.device(bgr.device):
        stream = torch.cuda.current_stream(bgr.device)
        bgr.record_stream(stream)
        mask.record_stream(stream)
        program = _nvrtc.fixed_kernel_program(_CUDA_SOURCE, bgr.device)
        guard.checkpoint()
        counts = torch.empty((tiles, 4), dtype=torch.int32, device=bgr.device)
        guard.watch(counts)
        program.launch_1d('count_tiles', (mask, counts),
            (mask_pixels, mask_width, height, width, MASK_TILE_PIXELS, tiles), tiles)
        guard.checkpoint()
        compact = counts.cpu().numpy()
        foreground, background, bad_foreground, bad_background = map(int,
            compact.sum(axis=0, dtype=np.int64))
        plan = _plan(domain, foreground, background, conversion_tile, gather_tile, budget)
        hsv = torch.empty((pixels, 3), dtype=torch.uint8, device=bgr.device)
        guard.watch(hsv)
        for start in range(0, pixels, plan.conversion_tile):
            guard.checkpoint()
            end = min(pixels, start+plan.conversion_tile)
            hsv[start:end].copy_(_range._channels(bgr.reshape(-1, 3)[start:end], 'HSV'))
            guard.watch(hsv)  # trusted PyTorch copy advances its version
        guard.checkpoint()
        # The source converts first, then extracts hue foreground/background
        # before either randint, even for zero sampled rows. Never gather an
        # unaudited image coordinate. Foreground has source failure priority.
        if bad_foreground:
            raise BayesSamplingSourceError('public hue foreground mask coordinate is out of bounds')
        if bad_background:
            raise BayesSamplingSourceError('public hue background mask coordinate is out of bounds')
        offsets_cpu = np.zeros_like(compact, dtype=np.int32)
        np.cumsum(compact[:-1], axis=0, dtype=np.int32, out=offsets_cpu[1:])
        offsets = torch.tensor(offsets_cpu, dtype=torch.int32, device=bgr.device)
        del compact, offsets_cpu
        positions = tuple(torch.empty(n, dtype=torch.int32, device=bgr.device)
            for n in (foreground, background))
        outputs = tuple(torch.empty((plan.samples_per_class, 3), dtype=torch.uint8,
            device=bgr.device) for _ in range(2))
        guard.watch(offsets, *positions, hsv, *outputs)
        guard.checkpoint()
        program.launch_1d('positions', (mask, offsets, *positions),
            (mask_pixels, mask_width, width, MASK_TILE_PIXELS, tiles), tiles)
        guard.checkpoint()
        draws = sha256()
        for channel in range(3):
            for category, population in enumerate((foreground, background)):
                guard.checkpoint()
                try:
                    indices = rng.randint(0, population-1, plan.samples_per_class)
                except ValueError as error:
                    guard.check()
                    raise BayesSamplingSourceError(
                        f'public randint failed for channel{channel}/class{category}: {error}') from error
                # Source draw happened; preserve it on every subsequent failure.
                guard.expected_rng = _state(rng)
                expected = shadow.randint(0, population-1, plan.samples_per_class)
                if (_state(shadow) != guard.expected_rng or not np.array_equal(indices, expected)):
                    raise BayesSamplingError('Bayes sampling RandomState changed during an ordered draw')
                del expected
                if (indices.dtype != np.dtype(RANDOM_INDEX_DTYPE) or not indices.flags.c_contiguous
                        or indices.shape != (plan.samples_per_class,)):
                    raise BayesSamplingError('source randint index layout differs from the C-long contract')
                draws.update(struct.pack('<iiii', channel, category, population-1, plan.samples_per_class))
                draws.update(indices.dtype.str.encode('ascii')+b'\0')
                draws.update(memoryview(indices).cast('B'))
                guard.checkpoint()
                for start in range(0, plan.samples_per_class, max(1, plan.gather_tile)):
                    guard.checkpoint()
                    stop = min(plan.samples_per_class, start+plan.gather_tile)
                    ordinals = torch.as_tensor(indices[start:stop], dtype=torch.int64, device=bgr.device)
                    guard.watch(ordinals)
                    program.launch_1d('gather_channel', (hsv, positions[category], ordinals, outputs[category]),
                        (start, stop-start, channel), stop-start)
                    guard.checkpoint()
                    del guard.tensors[id(ordinals)]
                    del ordinals
                del indices
        guard.checkpoint()
        return BayesImageSamples(outputs, plan, before, guard.expected_rng,
            draws.hexdigest(), np.dtype(RANDOM_INDEX_DTYPE).str)
