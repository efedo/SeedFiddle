"""Bounded CUDA nearest-centre scoring for pinned sklearn K-means models.

The source compares squared-centre-norm minus twice the feature dot product,
omitting the feature norm common to every centre. No float32/TF32 matrix
multiplication is used. Only compact numeric centre metadata enters from CPU.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from seedvision.cuda.tensor_callback import checked_cancel, TensorCallbackReturnError

MAX_CLUSTERS = 256
MAX_FEATURES = 4096
MAX_CENTRE_VALUES = 1_048_576
MAX_ROWS = 16_000_000
MAX_SCRATCH_BYTES = 64 * 1024 * 1024
MAX_WORK = 1_000_000_000
SCRATCH_HEADROOM_BYTES = 2 * 1024 * 1024
SKLEARN_VERSION = '1.9.1'
SKLEARN_KMEANS_SHA256 = '0cf335b8b9b4abe67259ccdcbaf07f02bd7feee31e63a06a1cce32a87fcd375d'


class KMeansClassificationError(ValueError):
    """Invalid features or centre metadata."""


class KMeansClassificationCancelled(RuntimeError):
    """Classification was cancelled before returning a complete vector."""


class KMeansDecisionAmbiguous(KMeansClassificationError):
    """The source winner is not established within the arithmetic bound."""


class KMeansResourceLimit(KMeansClassificationError):
    """Classification exceeds a declared work or storage limit."""


@dataclass(frozen=True, slots=True)
class _PreparedCentres:
    unique: tuple[tuple[float, ...], ...]
    centres_gpu: torch.Tensor
    norms_gpu: torch.Tensor
    original_gpu: torch.Tensor
    passes: int


def _poll(cancel,*tensors):
    if cancel is not None:
        try:
            result = checked_cancel(cancel,tensors)
        except TensorCallbackReturnError as error:
            raise KMeansClassificationError('cancel callback must return bool') from error
        if result:
            raise KMeansClassificationCancelled('K-means classification was cancelled')


def _prepare_centres(centres, feature_count, row_count):
    if type(centres) is not tuple or not 1 <= len(centres) <= MAX_CLUSTERS:
        raise KMeansClassificationError('centres must contain 1..256 immutable rows')
    if len(centres) * feature_count > MAX_CENTRE_VALUES:
        raise KMeansResourceLimit('centre matrix exceeds the compact metadata limit')
    unique, indices, seen = [], [], set()
    integral = True
    for index, row in enumerate(centres):
        if type(row) is not tuple or len(row) != feature_count:
            raise KMeansClassificationError('centre rows must be immutable and match the feature width')
        normalized = []
        for value in row:
            if type(value) not in (int, float):
                raise KMeansClassificationError('centre values must be numeric scalars')
            try:
                value = float(value)
            except (ValueError, OverflowError) as error:
                raise KMeansClassificationError('unsupported centre value') from error
            if not math.isfinite(value) or not 0.0 <= value <= 255.0:
                raise KMeansClassificationError('centre values must be finite and in [0,255]')
            integral = integral and value.is_integer()
            normalized.append(value)
        key = tuple(normalized)
        if key not in seen:
            seen.add(key)
            unique.append(key)
            indices.append(index)
    passes = 1 if integral or len(unique) == 1 else 2
    if row_count * len(unique) * feature_count * passes > MAX_WORK:
        raise KMeansResourceLimit('prediction work exceeds the supported bound')
    return tuple(unique), tuple(indices), integral, passes


def _kmeans_tile_rows(feature_count, centre_values=0):
    # Include two centre tables during norm reduction, GPU row metadata, one
    # float64 feature tile/product tile, and all row-wise temporaries. The norm
    # product is gone before feature scoring; counting both tables throughout
    # is deliberately conservative.
    available = MAX_SCRATCH_BYTES - SCRATCH_HEADROOM_BYTES - 16 * centre_values
    rows = available // (16 * feature_count + 512)
    if rows < 1:
        raise KMeansResourceLimit('one feature tile exceeds the scratch limit')
    return rows


def _decision_margin_bound(feature_count):
    """Conservative source/GPU binary64 score-comparison error in byte space.

    For F nonnegative byte features/centres, sum(c*c)+2*sum(x*c) <=195075*F.
    Each backend's norm/dot/score error is bounded by gamma_(2F+8) times
    that magnitude. Four score errors cover two backends/two competitors;
    the GPU margin subtraction adds at most 2*u*(1+gamma_(2F+8)) times
    the magnitude. Using gamma_(2F+32) reserves more than that extra error,
    including underflow-sized absolute terms. Integral products/sums stay below2**53
    throughout this domain, so they need no floating winner guard.
    """
    unit = 2.0 ** -53
    product = (2 * feature_count + 32) * unit
    gamma = product / (1.0 - product)
    return math.nextafter(4.0 * gamma * 195075.0 * feature_count, math.inf)


def _prepare_cuda_centres(unique, indices, passes, device):
    """Create reusable CUDA model intermediates after complete work preflight."""
    centres_gpu = torch.tensor(unique, dtype=torch.float64, device=device)
    norms_gpu = centres_gpu.square().sum(dim=1, dtype=torch.float64)
    original_gpu = torch.tensor(indices, dtype=torch.uint8, device=device)
    return _PreparedCentres(unique, centres_gpu, norms_gpu, original_gpu, passes)


def _prepared_identity(prepared):
    return (id(prepared.unique), prepared.passes, id(prepared.centres_gpu),
            id(prepared.norms_gpu), id(prepared.original_gpu))


def classify_kmeans_features(features, centres, *, cancel=None, tile_rows=None):
    """Return complete CUDA uint8 labels for byte-equivalent feature rows.

    Integer-centre ties and identical rows retain the first original cluster.
    For fractional centres every distinct competitor must have a certified gap;
    one uncertain row withholds the entire vector, including earlier tiles.
    """
    if cancel is not None and not callable(cancel):
        raise KMeansClassificationError('cancel must be callable or None')
    _poll(cancel,features)
    if (not torch.is_tensor(features) or features.device.type != 'cuda'
            or features.layout != torch.strided or features.dtype != torch.uint8
            or features.ndim != 2 or not features.is_contiguous()):
        raise KMeansClassificationError('features must be contiguous strided CUDA uint8 rows')
    row_count, feature_count = map(int, features.shape)
    if not 0 <= row_count <= MAX_ROWS or not 1 <= feature_count <= MAX_FEATURES:
        raise KMeansResourceLimit('feature row count or width exceeds the supported bound')
    unique, indices, integral, passes = _prepare_centres(centres, feature_count, row_count)
    selected_tile = _kmeans_tile_rows(feature_count, len(unique) * feature_count)
    if tile_rows is not None:
        if type(tile_rows) is not int or not 1 <= tile_rows <= selected_tile:
            raise KMeansResourceLimit('tile_rows must be positive and fit the scratch bound')
        selected_tile = tile_rows
    _poll(cancel,features)
    if row_count == 0:
        return torch.empty((0,), dtype=torch.uint8, device=features.device)
    prepared = _prepare_cuda_centres(unique, indices, passes, features.device)
    return _classify_prepared_features(features, prepared, cancel=cancel, tile_rows=selected_tile)


def _classify_prepared_features(features, prepared, *, cancel=None, tile_rows=None):
    """Internal scoring for validated byte rows and a privately prepared model."""
    caller_cancel = cancel
    if caller_cancel is not None:
        def prepared_cancel():
            return checked_cancel(caller_cancel, (prepared.centres_gpu, prepared.norms_gpu,
                                  prepared.original_gpu), identity=lambda: _prepared_identity(prepared))
        cancel = prepared_cancel
    row_count, feature_count = map(int, features.shape)
    unique = prepared.unique
    centres_gpu, norms_gpu, original_gpu = prepared.centres_gpu, prepared.norms_gpu, prepared.original_gpu
    selected_tile = _kmeans_tile_rows(feature_count, len(unique) * feature_count)
    if tile_rows is not None:
        selected_tile = min(selected_tile, tile_rows)
    labels = torch.empty((row_count,), dtype=torch.uint8, device=features.device)
    margin_bound = _decision_margin_bound(feature_count)
    _poll(cancel,features,centres_gpu,norms_gpu,original_gpu,labels)
    for begin in range(0, row_count, selected_tile):
        _poll(cancel,features,centres_gpu,norms_gpu,original_gpu,labels)
        end = min(row_count, begin + selected_tile)
        x = features[begin:end].to(torch.float64)
        best = torch.full((end - begin,), math.inf, dtype=torch.float64, device=features.device)
        winner = torch.zeros((end - begin,), dtype=torch.int64, device=features.device)
        for index in range(len(unique)):
            _poll(cancel,features,centres_gpu,norms_gpu,original_gpu,labels,x,best,winner)
            dot = (x * centres_gpu[index]).sum(dim=1, dtype=torch.float64)
            distance = norms_gpu[index] - 2.0 * dot
            better = distance < best
            best = torch.where(better, distance, best)
            winner = torch.where(better, index, winner)
        if prepared.passes == 2:
            uncertain = torch.zeros((), dtype=torch.bool, device=features.device)
            for index in range(len(unique)):
                _poll(cancel,features,centres_gpu,norms_gpu,original_gpu,labels,x,best,winner,uncertain)
                dot = (x * centres_gpu[index]).sum(dim=1, dtype=torch.float64)
                distance = norms_gpu[index] - 2.0 * dot
                uncertain.logical_or_(((winner != index) &
                                      (distance - best <= margin_bound)).any())
            _poll(cancel,features,centres_gpu,norms_gpu,original_gpu,labels,x,best,winner,uncertain)
            if bool(uncertain.item()):
                raise KMeansDecisionAmbiguous('K-means winner is within the source/GPU binary64 error interval')
            del uncertain
        labels[begin:end].copy_(original_gpu[winner])
        del x, best, winner, dot, distance, better
    _poll(cancel,features,centres_gpu,norms_gpu,original_gpu,labels)
    return labels
