"""Source-ordered CUDA image patches and complete MiniBatch training composition.

Input channel order is preserved: PlantCV's file path supplies OpenCV BGR.
Directory selection and native desktop/project bindings are separate layers.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from seedvision.cuda import plantcv_kmeans_fit as _fit
from seedvision.cuda import plantcv_kmeans_training_data as _full
from seedvision.cuda import plantcv_kmeans_sampled_data as _sampled
from seedvision.cuda.tensor_callback import TensorCallbackMutation

MAX_IMAGES = 256
MAX_DATASET_BYTES = 128 * 1024 * 1024
MAX_CUDA_LIVE_BYTES = 512 * 1024 * 1024
MAX_COMBINED_WORK = 24_000_000_000
PATCH_SEED = 1
PLANTCV_TRAIN_SHA256 = 'dbfb8a528e92984a20e73c7f265a41efa9fa088b42ed4f65addbe6ab40ee8b4d'


class KMeansImageTrainingError(ValueError):
    """Invalid ordered images or a combined training resource bound."""


class KMeansImageTrainingCancelled(RuntimeError):
    """Image training cancelled without publishing a partial result."""


@dataclass(frozen=True, slots=True)
class KMeansImageDataPlan:
    rows: int
    features: int
    channels: int
    input_bytes: int
    dataset_bytes: int
    work: int
    peak_live_bytes: int
    image_plans: tuple


@dataclass(frozen=True, slots=True)
class KMeansImageFitPlan:
    data: KMeansImageDataPlan
    fit: _fit.KMeansFitPlan
    work: int
    peak_live_bytes: int


@dataclass(frozen=True, slots=True)
class KMeansTrainingRows:
    features: torch.Tensor
    plan: KMeansImageDataPlan


@dataclass(frozen=True, slots=True)
class KMeansImageFitResult:
    training_rows: KMeansTrainingRows
    fit: _fit.KMeansFitResult
    plan: KMeansImageFitPlan


def _images_and_guard(images, cancel):
    if type(images) not in (tuple, list) or not 1 <= len(images) <= MAX_IMAGES:
        raise KMeansImageTrainingError('provide an ordered image list within the image-count bound')
    images = tuple(images)
    if any(not torch.is_tensor(image) for image in images):
        raise KMeansImageTrainingError('every image must be a CUDA tensor')
    if cancel is not None and not callable(cancel):
        raise KMeansImageTrainingError('cancel must be callable or None')
    versions = tuple(image._version for image in images)

    def guard():
        if tuple(image._version for image in images) != versions:
            raise KMeansImageTrainingError('training image changed during calculation')
        try:
            cancelled = cancel() if cancel is not None else False
        finally:
            if tuple(image._version for image in images) != versions:
                raise KMeansImageTrainingError('training image changed during callback')
        if type(cancelled) is not bool:
            raise KMeansImageTrainingError('cancel callback must return bool')
        if cancelled:
            raise KMeansImageTrainingCancelled('K-means image training was cancelled')
        return False
    return images, guard


def preflight_kmeans_training_images(images, patch_size=10, *, sampling=None,
        cancel=None, tile_pixels=4096, gaussian_tile_side=256):
    """Bound all ordered image extraction and copies before Gaussian/RNG work."""
    images, guard = _images_and_guard(images, cancel)
    guard()
    plans = []
    for image in images:
        if sampling is None:
            plan = _full.preflight_kmeans_training_data(image, patch_size, cancel=guard,
                tile_pixels=tile_pixels, gaussian_tile_side=gaussian_tile_side)
        else:
            plan = _sampled.preflight_sampled_kmeans_training_patches(image, patch_size,
                sampling, seed=PATCH_SEED, cancel=guard, tile_pixels=tile_pixels,
                gaussian_tile_side=gaussian_tile_side)
        if plans and (plan.channels != plans[0].channels or image.device != images[0].device):
            raise KMeansImageTrainingError('training images must share channel count and CUDA device')
        plans.append(plan)
        guard()
    rows = sum(plan.patch_count for plan in plans)
    width = plans[0].feature_count
    dataset_bytes = 2*rows*width
    if rows > _fit.MAX_ROWS or dataset_bytes > MAX_DATASET_BYTES:
        raise KMeansImageTrainingError('combined training dataset exceeds its row/byte bound')
    storages = {(str(image.device), image.untyped_storage().data_ptr()): image.untyped_storage().nbytes()
        for image in images}
    input_bytes = sum(storages.values())
    work = sum(plan.work for plan in plans)+rows*width
    extra = max(plan.peak_live_bytes-image.numel()*image.element_size()
        for image, plan in zip(images, plans))
    peak = input_bytes+dataset_bytes+extra
    if work > MAX_COMBINED_WORK or peak > MAX_CUDA_LIVE_BYTES:
        raise KMeansImageTrainingError('image assembly exceeds its combined work/live bound')
    guard()
    return KMeansImageDataPlan(rows, width, plans[0].channels, input_bytes,
        dataset_bytes, work, peak, tuple(plans))


def _assemble(images, plan, patch_size, sampling, guard, tile_pixels, gaussian_tile_side):
    guard()
    with torch.cuda.device(images[0].device):
        stream = torch.cuda.current_stream(images[0].device)
        for image in images:
            image.record_stream(stream)
        guard()
        output = torch.empty((plan.rows, plan.features), dtype=torch.uint16, device=images[0].device)
        start = 0
        for image, expected in zip(images, plan.image_plans):
            guard()
            try:
                if sampling is None:
                    patches = _full.extract_full_kmeans_training_patches(image, patch_size,
                        cancel=guard, tile_pixels=tile_pixels, gaussian_tile_side=gaussian_tile_side)
                else:
                    patches = _sampled.extract_sampled_kmeans_training_patches(image, patch_size,
                        sampling, seed=PATCH_SEED, cancel=guard, tile_pixels=tile_pixels,
                        gaussian_tile_side=gaussian_tile_side)
            except TensorCallbackMutation as error:
                raise KMeansImageTrainingError('training image changed during patch extraction') from error
            except _sampled.KMeansSampledDataError as error:
                if isinstance(error.__cause__, TensorCallbackMutation):
                    raise KMeansImageTrainingError('training image changed during patch extraction') from error.__cause__
                raise
            guard()
            if (not torch.is_tensor(patches) or patches.device != output.device
                    or patches.dtype != torch.uint16 or patches.layout != torch.strided
                    or tuple(patches.shape) != (expected.patch_count, plan.features)
                    or not patches.is_contiguous()):
                raise KMeansImageTrainingError('patch producer returned an invalid complete matrix')
            stop = start+expected.patch_count
            output[start:stop].copy_(patches)
            del patches
            start = stop
            guard()
        if start != plan.rows:
            raise KMeansImageTrainingError('ordered patches do not cover the combined dataset')
        guard()
        return KMeansTrainingRows(output, plan)


def assemble_kmeans_training_images(images, patch_size=10, *, sampling=None,
        cancel=None, tile_pixels=4096, gaussian_tile_side=256):
    """Return reusable CUDA rows; each sampled image starts RandomState(1)."""
    images, guard = _images_and_guard(images, cancel)
    plan = preflight_kmeans_training_images(images, patch_size, sampling=sampling,
        cancel=guard, tile_pixels=tile_pixels, gaussian_tile_side=gaussian_tile_side)
    return _assemble(images, plan, patch_size, sampling, guard, tile_pixels, gaussian_tile_side)


def preflight_kmeans_image_fit(images, n_clusters, *, source_n_threads,
        patch_size=10, sampling=None, seed=1, max_iter=100, batch_size=1024,
        n_init=10, init_size=None, tol=0.0, max_no_improvement=10,
        reassignment_ratio=.01, compute_labels=True, cancel=None,
        tile_pixels=4096, gaussian_tile_side=256):
    """Bound extraction and every fit phase before allocating combined rows."""
    images, guard = _images_and_guard(images, cancel)
    data = preflight_kmeans_training_images(images, patch_size, sampling=sampling,
        cancel=guard, tile_pixels=tile_pixels, gaussian_tile_side=gaussian_tile_side)
    fit = _fit.preflight_kmeans_fit_shape(data.rows, data.features, n_clusters,
        source_n_threads=source_n_threads, seed=seed, max_iter=max_iter,
        batch_size=batch_size, n_init=n_init, init_size=init_size, tol=tol,
        max_no_improvement=max_no_improvement, reassignment_ratio=reassignment_ratio,
        compute_labels=compute_labels)
    work = data.work+fit.work
    peak = max(data.peak_live_bytes, data.input_bytes+fit.peak_live_bytes)
    if work > MAX_COMBINED_WORK or peak > MAX_CUDA_LIVE_BYTES:
        raise KMeansImageTrainingError('image fitting exceeds its combined work/live bound')
    guard()
    return KMeansImageFitPlan(data, fit, work, peak)


def fit_kmeans_training_images(images, n_clusters, *, source_n_threads,
        patch_size=10, sampling=None, seed=1, max_iter=100, batch_size=1024,
        n_init=10, init_size=None, tol=0.0, max_no_improvement=10,
        reassignment_ratio=.01, compute_labels=True, cancel=None,
        tile_pixels=4096, gaussian_tile_side=256):
    """Compose ordered image patches and fitting, retaining reusable CUDA rows."""
    images, guard = _images_and_guard(images, cancel)
    parameters = dict(source_n_threads=source_n_threads, seed=seed, max_iter=max_iter,
        batch_size=batch_size, n_init=n_init, init_size=init_size, tol=tol,
        max_no_improvement=max_no_improvement, reassignment_ratio=reassignment_ratio,
        compute_labels=compute_labels)
    plan = preflight_kmeans_image_fit(images, n_clusters, patch_size=patch_size,
        sampling=sampling, cancel=guard, tile_pixels=tile_pixels,
        gaussian_tile_side=gaussian_tile_side, **parameters)
    rows = _assemble(images, plan.data, patch_size, sampling, guard, tile_pixels, gaussian_tile_side)
    guard()
    fit = _fit.fit_kmeans_training_rows(rows.features, n_clusters, cancel=guard, **parameters)
    guard()
    return KMeansImageFitResult(rows, fit, plan)
