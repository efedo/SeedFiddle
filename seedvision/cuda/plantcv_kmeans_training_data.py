"""Bounded CUDA full-patch data for the pinned PlantCV K-means training path."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from seedvision.cuda import plantcv_kmeans_features as _features
from seedvision.cuda import plantcv_kmeans_preprocess as _preprocess

MAX_PATCHES = 16_000_000
MAX_DATASET_BYTES = 128 * 1024 * 1024
MAX_WORK = 2_000_000_000
MAX_CUDA_LIVE_BYTES = 512 * 1024 * 1024
ALLOCATION_HEADROOM_BYTES = 2 * 1024 * 1024


class KMeansTrainingDataError(ValueError):
    """Unsupported input or whole-dataset resource limit."""


class KMeansTrainingDataCancelled(RuntimeError):
    """Training-patch extraction was cancelled without returning a matrix."""


@dataclass(frozen=True, slots=True)
class KMeansTrainingDataPlan:
    height: int
    width: int
    channels: int
    patch_size: int
    feature_count: int
    patch_count: int
    tile_pixels: int
    gaussian_output_bytes: int
    dataset_bytes: int
    work: int
    peak_live_bytes: int


def _poll(cancel):
    if cancel is not None and cancel():
        raise KMeansTrainingDataCancelled('K-means training-patch extraction was cancelled')


def preflight_kmeans_training_data(image, patch_size, *, cancel=None,
                                   tile_pixels=4096, gaussian_tile_side=256):
    """Check Gaussian, patch, output, work and live-storage limits before allocation."""
    if cancel is not None and not callable(cancel):
        raise TypeError('cancel must be a zero-argument callable or None')
    _poll(cancel)
    if type(patch_size) is not int or not 1 <= patch_size <= 31:
        raise KMeansTrainingDataError('patch_size must be an integer in [1,31]')
    channels, height, width, _, _ = _preprocess._preflight(image, 1, gaussian_tile_side)
    if height < patch_size or width < patch_size:
        raise KMeansTrainingDataError('patch_size must fit inside the unpadded training image')
    feature_count = patch_size * patch_size * channels
    if feature_count > _features.MAX_PATCH_FEATURES:
        raise KMeansTrainingDataError('patch feature width exceeds the supported domain')
    patch_count = (height-patch_size+1) * (width-patch_size+1)
    if not 1 <= patch_count <= MAX_PATCHES:
        raise KMeansTrainingDataError('full-patch dataset exceeds the supported row count')
    selected = _features._feature_tile_pixels(feature_count, tile_pixels)
    if selected < 1:
        raise KMeansTrainingDataError('one patch feature row exceeds the tile scratch limit')
    selected = min(selected, patch_count)
    tile_bytes = selected * (feature_count*2 + _features.FEATURE_TILE_ROW_OVERHEAD_BYTES)
    if tile_bytes > _features.MAX_FEATURE_TILE_BYTES:
        raise KMeansTrainingDataError('patch feature tile exceeds its byte cap')
    dataset_bytes = patch_count * feature_count * 2
    if dataset_bytes > MAX_DATASET_BYTES:
        raise KMeansTrainingDataError('complete full-patch matrix exceeds the output byte cap')
    # Public training filters the unpadded image, then extracts valid windows.
    tile_columns = (width+gaussian_tile_side-1) // gaussian_tile_side
    gaussian_work = (height*(width+40*tile_columns) + height*width) * channels * 41
    work = gaussian_work + patch_count*feature_count
    gaussian_bytes = height*width*channels*2
    image_bytes = image.numel()*image.element_size()
    gaussian_scratch = _preprocess._preflight_scratch_bytes(channels, gaussian_tile_side)
    # Gaussian scratch and feature tiles do not overlap. Matrix allocation
    # follows the Gaussian; only one feature tile coexists with the output.
    peak_live = max(image_bytes+gaussian_bytes+gaussian_scratch,
        image_bytes+gaussian_bytes+dataset_bytes+tile_bytes+ALLOCATION_HEADROOM_BYTES)
    if work > MAX_WORK or peak_live > MAX_CUDA_LIVE_BYTES:
        raise KMeansTrainingDataError('full-patch extraction exceeds its total work/live-storage cap')
    _poll(cancel)
    return KMeansTrainingDataPlan(height, width, channels, patch_size, feature_count,
        patch_count, selected, gaussian_bytes, dataset_bytes, work, peak_live)


def extract_full_kmeans_training_patches(image, patch_size, *, cancel=None,
                                         tile_pixels=4096, gaussian_tile_side=256):
    """Return unpadded sigma5/sampling=None patches as a CUDA uint16 matrix.

    Every valid window is emitted row-major; rows flatten patch-y, patch-x,
    channel. Input channel order is preserved; PlantCV training-file inputs use
    OpenCV BGR order. Directory selection, other sigma
    values, sampled-patch RNG and estimator fitting are separate operations.
    """
    version = image._version if torch.is_tensor(image) else None
    plan = preflight_kmeans_training_data(image, patch_size, cancel=cancel,
        tile_pixels=tile_pixels, gaussian_tile_side=gaussian_tile_side)
    if image._version != version:
        raise KMeansTrainingDataError('training image changed during preflight')
    try:
        # patch_size1 removes prediction padding without changing sigma5.
        blurred = _preprocess.gaussian_patch_raster(image, 1, cancel=cancel,
            tile_side=gaussian_tile_side)
        _poll(cancel)
        patches = torch.empty((plan.patch_count, plan.feature_count),
            dtype=torch.uint16, device=image.device)
        expected_start = 0
        for start, tile in _features.iter_kmeans_patch_tiles(blurred, patch_size,
                cancel=cancel, tile_pixels=plan.tile_pixels):
            _poll(cancel)
            stop = start + int(tile.shape[0])
            if start != expected_start or stop > plan.patch_count:
                raise KMeansTrainingDataError('patch tiles do not cover the training matrix in order')
            patches[start:stop].copy_(tile)
            expected_start = stop
            del tile
        _poll(cancel)
        if expected_start != plan.patch_count or image._version != version:
            raise KMeansTrainingDataError('training image changed or patch coverage is incomplete')
        return patches
    except (_preprocess.KMeansPreprocessCancelled, _features.KMeansFeatureCancelled) as error:
        raise KMeansTrainingDataCancelled(str(error)) from error
