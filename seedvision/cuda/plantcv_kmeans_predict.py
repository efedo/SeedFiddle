"""Native frozen-model PlantCV K-means image prediction, with bounded CUDA tiles."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from seedvision.cuda import plantcv_kmeans_classifier as _classifier
from seedvision.cuda import plantcv_kmeans_features as _features
from seedvision.cuda import plantcv_kmeans_model as _model
from seedvision.cuda import plantcv_kmeans_preprocess as _preprocess
from seedvision.cuda.tensor_callback import checked_cancel, TensorCallbackMutation

MAX_WORK = 2_000_000_000
MAX_SCRATCH_BYTES = 128 * 1024 * 1024
MAX_CUDA_LIVE_BYTES = 512 * 1024 * 1024
DEFAULT_FEATURE_TILE_PIXELS = 4096
DEFAULT_GAUSSIAN_TILE_SIDE = 256


class KMeansPredictionError(ValueError):
    """Invalid image/model binding or a whole-prediction resource limit."""


class KMeansPredictionCancelled(RuntimeError):
    """Prediction was cancelled without returning a partial label raster."""


class KMeansPredictionMutation(KMeansPredictionError, TensorCallbackMutation):
    """A caller changed prediction buffers or immutable model metadata."""


@dataclass(frozen=True, slots=True)
class KMeansPredictionPlan:
    height: int
    width: int
    feature_count: int
    tile_pixels: int
    gaussian_output_bytes: int
    label_output_bytes: int
    scratch_bytes: int
    live_bytes: int
    work: int


def _poll(cancel, *tensors):
    try:
        stopped = checked_cancel(cancel, tensors)
    except TensorCallbackMutation as error:
        raise KMeansPredictionMutation(str(error)) from error
    if stopped:
        raise KMeansPredictionCancelled('K-means prediction was cancelled')


def preflight_kmeans_prediction(image, model, *, cancel=None,
                                 tile_pixels=DEFAULT_FEATURE_TILE_PIXELS,
                                 gaussian_tile_side=DEFAULT_GAUSSIAN_TILE_SIDE):
    """Validate the entire model/image/work/live-storage plan before allocation."""
    if cancel is not None and not callable(cancel):
        raise TypeError('cancel must be callable or None')
    _poll(cancel, image)
    _model.validate_kmeans_model(model)
    channels, height, width, padded_height, padded_width = _preprocess._preflight(
        image, model.patch_size, gaussian_tile_side)
    if channels != model.channels:
        raise KMeansPredictionError('image channels differ from the frozen model')
    plan = _plan_prediction_dimensions(channels, height, width, padded_height, padded_width,
                                      model, tile_pixels, gaussian_tile_side)
    _poll(cancel, image)
    return plan


def preflight_kmeans_prediction_shape(height, width, model, *,
                                      tile_pixels=DEFAULT_FEATURE_TILE_PIXELS,
                                      gaussian_tile_side=DEFAULT_GAUSSIAN_TILE_SIDE):
    """Plan the same native calculation from validated metadata, without CUDA."""
    _model.validate_kmeans_model(model)
    dimensions = _preprocess._preflight_shape(model.channels, height, width,
                                             model.patch_size, gaussian_tile_side)
    return _plan_prediction_dimensions(*dimensions, model, tile_pixels, gaussian_tile_side)


def _plan_prediction_dimensions(channels, height, width, padded_height, padded_width,
                                model, tile_pixels, gaussian_tile_side):
    feature_count = model.patch_size * model.patch_size * channels
    unique, _, _, passes = _classifier._prepare_centres(model.centres, feature_count, height * width)
    _classifier._kmeans_tile_rows(feature_count, len(unique) * feature_count)
    selected = _features._feature_tile_pixels(feature_count, tile_pixels)
    if selected < 1:
        raise KMeansPredictionError('one feature tile exceeds the scratch bound')
    selected = min(height * width, selected)
    gaussian_bytes, label_bytes = padded_height * padded_width * channels * 2, height * width
    # Features coexist as uint16 and byte-equivalent uint8 while scoring. The
    # classifier's budget already includes its centre table and row temporaries.
    feature_bytes = selected * (3 * feature_count + _features.FEATURE_TILE_ROW_OVERHEAD_BYTES)
    scratch = max(_preprocess._preflight_scratch_bytes(channels, gaussian_tile_side),
                  feature_bytes + _classifier.MAX_SCRATCH_BYTES)
    live = height * width * channels + gaussian_bytes + label_bytes + scratch
    tile_columns = (padded_width + gaussian_tile_side - 1) // gaussian_tile_side
    gaussian_work = (padded_height * (padded_width + 40 * tile_columns)
                     + padded_height * padded_width) * channels * 41
    work = (gaussian_work + len(unique) * feature_count
            + height * width * feature_count * (1 + len(unique) * passes))
    if work > MAX_WORK or scratch > MAX_SCRATCH_BYTES or live > MAX_CUDA_LIVE_BYTES:
        raise KMeansPredictionError('complete prediction exceeds its work/scratch/live-storage bound')
    return KMeansPredictionPlan(height, width, feature_count, selected, gaussian_bytes,
                                 label_bytes, scratch, live, work)


def predict_kmeans(image, model, *, cancel=None, tile_pixels=DEFAULT_FEATURE_TILE_PIXELS,
                   gaussian_tile_side=DEFAULT_GAUSSIAN_TILE_SIDE):
    """Return a complete CUDA uint8 HW label raster for a frozen numeric model.

    Channel order is preserved exactly as supplied, matching the public image
    input. The model fixes patch size; preprocessing is always the public sigma5
    path. Work and live storage are checked globally before Gaussian allocation.
    """
    if cancel is not None and not callable(cancel):
        raise TypeError('cancel must be callable or None')
    caller_cancel = cancel
    active = {'image': image}
    prepared = None
    def identity():
        return (model.patch_size, model.channels, id(model.centres), model.source_sha256,
                model.source_bytes, model.centres_sha256,
                _classifier._prepared_identity(prepared) if prepared is not None else None)
    def guarded_cancel():
        try:
            return checked_cancel(caller_cancel, tuple(active.values()), identity=identity)
        except TensorCallbackMutation as error:
            raise KMeansPredictionMutation(str(error)) from error
    cancel = guarded_cancel if caller_cancel is not None else None
    plan = preflight_kmeans_prediction(image, model, cancel=cancel, tile_pixels=tile_pixels,
                                       gaussian_tile_side=gaussian_tile_side)
    input_version = image._version
    blurred = _preprocess.gaussian_patch_raster(image, model.patch_size, cancel=cancel,
                                               tile_side=gaussian_tile_side)
    active['blurred'] = blurred
    _poll(cancel)
    unique, indices, _, passes = _classifier._prepare_centres(model.centres, plan.feature_count,
                                                             plan.height * plan.width)
    prepared = _classifier._prepare_cuda_centres(unique, indices, passes, image.device)
    active.update(centres=prepared.centres_gpu, norms=prepared.norms_gpu, original=prepared.original_gpu)
    _poll(cancel)
    labels = torch.empty((plan.height * plan.width,), dtype=torch.uint8, device=image.device)
    active['labels'] = labels
    expected_start = 0
    for start, features in _features.iter_kmeans_patch_tiles(blurred, model.patch_size,
                                                            cancel=cancel, tile_pixels=plan.tile_pixels):
        active['features'] = features
        _poll(cancel)
        if start != expected_start:
            raise KMeansPredictionError('patch tiles do not cover the raster in order')
        # The fixed nonnegative Gaussian and guarded byte rounding guarantee
        # values0..255; this narrowing preserves every public uint16 feature value.
        byte_features = features.to(torch.uint8)
        active['byte_features'] = byte_features
        tile_labels = _classifier._classify_prepared_features(byte_features, prepared, cancel=cancel)
        stop = start + int(tile_labels.numel())
        labels[start:stop].copy_(tile_labels)
        expected_start = stop
        active.pop('features')
        active.pop('byte_features')
        del features, byte_features, tile_labels
    _poll(cancel)
    if expected_start != labels.numel() or image._version != input_version:
        raise KMeansPredictionError('prediction coverage or input identity changed during calculation')
    return labels.reshape(plan.height, plan.width)
