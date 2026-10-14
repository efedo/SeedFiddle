"""CUDA sampled sigma5 training patches with pinned source RandomState draws."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch

from seedvision.cuda import plantcv_kmeans_features as _features
from seedvision.cuda import plantcv_kmeans_preprocess as _preprocess
from seedvision.cuda.tensor_callback import TensorCallbackMutation

MAX_SAMPLE_INDEX_BYTES = 16 * 1024 * 1024
MAX_DATASET_BYTES = 128 * 1024 * 1024
MAX_WORK = 2_000_000_000
MAX_CUDA_LIVE_BYTES = 512 * 1024 * 1024
INDEX_DEVICE_BYTES_PER_ROW = 16
TILE_ROW_OVERHEAD_BYTES = 128
ALLOCATION_HEADROOM_BYTES = 2 * 1024 * 1024
CPU_INDEX_HEADROOM_BYTES = 128 * 1024
RANDOM_INDEX_DTYPE = 'l'


class KMeansSampledDataError(ValueError):
    """Unsupported sample request or whole-sample resource limit."""


class KMeansSampledDataCancelled(RuntimeError):
    """Sampled extraction cancelled without a complete matrix."""


@dataclass(frozen=True, slots=True)
class KMeansSampledDataPlan:
    height: int
    width: int
    channels: int
    patch_size: int
    feature_count: int
    patch_count: int
    tile_pixels: int
    gaussian_output_bytes: int
    dataset_bytes: int
    cpu_index_bytes: int
    work: int
    peak_live_bytes: int


def _poll(cancel):
    if cancel is not None and cancel():
        raise KMeansSampledDataCancelled('sampled K-means patch extraction was cancelled')


def _sample_count(total, sampling):
    if type(sampling) is int:
        if sampling <= 0:
            raise KMeansSampledDataError('sample count must be a positive integer')
        return min(sampling, total)
    if type(sampling) is float:
        if not math.isfinite(sampling) or not 0.0 < sampling < 1.0:
            raise KMeansSampledDataError('fractional sampling must be finite and strictly between zero and one')
        count = int(sampling * total)
        if count == 0:
            raise KMeansSampledDataError('zero sampled rows: pinned PlantCV fails at reshape(0, -1)')
        return count
    raise KMeansSampledDataError('sampling must be a positive integer or a float in (0,1)')


def preflight_sampled_kmeans_training_patches(image, patch_size, sampling, *, seed=1,
                                              cancel=None, tile_pixels=4096,
                                              gaussian_tile_side=256):
    """Check sampled output, index, Gaussian, work and live bounds before allocation."""
    if cancel is not None and not callable(cancel):
        raise TypeError('cancel must be a zero-argument callable or None')
    _poll(cancel)
    if type(seed) is not int or not 0 <= seed <= 0xFFFFFFFF:
        raise KMeansSampledDataError('RandomState seed must be an integer in [0,2**32-1]')
    if type(patch_size) is not int or not 1 <= patch_size <= 31:
        raise KMeansSampledDataError('patch_size must be an integer in [1,31]')
    channels, height, width, _, _ = _preprocess._preflight(image, 1, gaussian_tile_side)
    if height < patch_size or width < patch_size:
        raise KMeansSampledDataError('patch_size must fit inside the unpadded training image')
    feature_count = patch_size * patch_size * channels
    if feature_count > _features.MAX_PATCH_FEATURES:
        raise KMeansSampledDataError('patch feature width exceeds the supported domain')
    total = (height-patch_size+1) * (width-patch_size+1)
    count = _sample_count(total, sampling)
    if not 1 <= count <= _features.MAX_PATCH_PIXELS:
        raise KMeansSampledDataError('sample row count exceeds the supported domain')
    # Legacy RandomState defaults to C long, even when NumPy's default int is
    # int64. Preserve those exact draws; widen only bounded device-index tiles.
    cpu_index_bytes = count * 2 * np.dtype(RANDOM_INDEX_DTYPE).itemsize
    if cpu_index_bytes + CPU_INDEX_HEADROOM_BYTES > MAX_SAMPLE_INDEX_BYTES:
        raise KMeansSampledDataError('compact row/column indices exceed the CPU metadata cap')
    dataset_bytes = count * feature_count * 2
    if dataset_bytes > MAX_DATASET_BYTES:
        raise KMeansSampledDataError('sampled patch rows exceed the output byte cap')
    requested = _features._feature_tile_pixels(feature_count, tile_pixels)
    tile_row_bytes = feature_count*2 + INDEX_DEVICE_BYTES_PER_ROW + TILE_ROW_OVERHEAD_BYTES
    selected = min(requested, _features.MAX_FEATURE_TILE_BYTES // tile_row_bytes, count)
    if selected < 1:
        raise KMeansSampledDataError('one sampled patch exceeds the gather tile limit')
    tile_bytes = selected * tile_row_bytes
    gaussian_bytes = height * width * channels * 2
    image_bytes = image.numel() * image.element_size()
    gaussian_scratch = _preprocess._preflight_scratch_bytes(channels, gaussian_tile_side)
    columns = (width+gaussian_tile_side-1) // gaussian_tile_side
    gaussian_work = (height*(width+40*columns) + height*width) * channels * 41
    work = gaussian_work + count * (2*feature_count+2)
    peak_live = max(image_bytes+gaussian_bytes+gaussian_scratch,
        image_bytes+gaussian_bytes+dataset_bytes+tile_bytes+ALLOCATION_HEADROOM_BYTES)
    if work > MAX_WORK or peak_live > MAX_CUDA_LIVE_BYTES:
        raise KMeansSampledDataError('sampled extraction exceeds total work/live-storage limits')
    _poll(cancel)
    return KMeansSampledDataPlan(height, width, channels, patch_size, feature_count,
        count, selected, gaussian_bytes, dataset_bytes, cpu_index_bytes, work, peak_live)


def extract_sampled_kmeans_training_patches(image, patch_size, sampling, *, seed=1,
                                            cancel=None, tile_pixels=4096,
                                            gaussian_tile_side=256):
    """Return direct patch_extract sampled rows as a complete CUDA uint16 matrix.

    Draw all row indices, then all columns, independently with replacement and
    RandomState's original default C-long dtype. Only bounded index slices widen
    to CUDA int64. Same-storage int16 views support byte-preserving CUDA gathers
    without a full raster cast. The train_kmeans wrapper does not forward its fit
    seed to patch_extract, whose seed therefore remains one in that path.
    None/falsy sampling belongs to the separate full-patch producer.
    """
    if not torch.is_tensor(image):
        raise KMeansSampledDataError('image must be a CUDA uint8 tensor')
    if cancel is not None and not callable(cancel):
        raise TypeError('cancel must be a zero-argument callable or None')
    version = image._version

    def checked_cancel():
        cancelled = cancel() if cancel is not None else False
        if image._version != version:
            raise KMeansSampledDataError('training image changed during sampled extraction')
        return cancelled

    plan = preflight_sampled_kmeans_training_patches(image, patch_size, sampling,
        seed=seed, cancel=checked_cancel, tile_pixels=tile_pixels,
        gaussian_tile_side=gaussian_tile_side)
    rng = np.random.RandomState(seed)
    rows_cpu = rng.randint(plan.height-patch_size+1, size=plan.patch_count)
    columns_cpu = rng.randint(plan.width-patch_size+1, size=plan.patch_count)
    expected_dtype = np.dtype(RANDOM_INDEX_DTYPE)
    if (rows_cpu.dtype != expected_dtype or columns_cpu.dtype != expected_dtype
            or rows_cpu.nbytes+columns_cpu.nbytes != plan.cpu_index_bytes):
        raise KMeansSampledDataError('RandomState index layout differs from the preflight contract')
    _poll(checked_cancel)
    try:
        try:
            blurred = _preprocess.gaussian_patch_raster(image, 1, cancel=checked_cancel,
                tile_side=gaussian_tile_side)
        except TensorCallbackMutation as error:
            raise KMeansSampledDataError('live calculation tensor changed during sampled extraction') from error
        _poll(checked_cancel)
        output = torch.empty((plan.patch_count, plan.feature_count), dtype=torch.uint16,
            device=image.device)
        source = blurred.view(torch.int16)
        if source.ndim == 2:
            source = source.unsqueeze(-1)
        windows = source.unfold(0, patch_size, 1).unfold(1, patch_size, 1).permute(0, 1, 3, 4, 2)
        signed_output = output.view(torch.int16)
        for start in range(0, plan.patch_count, plan.tile_pixels):
            _poll(checked_cancel)
            stop = min(plan.patch_count, start+plan.tile_pixels)
            rows = torch.as_tensor(rows_cpu[start:stop], dtype=torch.int64, device=image.device)
            columns = torch.as_tensor(columns_cpu[start:stop], dtype=torch.int64, device=image.device)
            selected = windows[rows, columns]
            signed_output[start:stop].view(stop-start, patch_size, patch_size,
                plan.channels).copy_(selected)
            del selected, rows, columns
        _poll(checked_cancel)
        return output
    except _preprocess.KMeansPreprocessCancelled as error:
        raise KMeansSampledDataCancelled(str(error)) from error
