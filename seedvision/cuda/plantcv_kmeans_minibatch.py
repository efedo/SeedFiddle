"""Source-ordered CUDA unit-weight MiniBatchKMeans centre/count updates.

This is the dense update component for supplied labels. Initial centre choice,
label assignment, sampled batch indices, reassignment and convergence belong to
the complete training loop and are not performed by this function.
The dense-update port retains the BSD notice in docs/SKLEARN_KMEANS_NOTICES.md.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from seedvision.cuda import _nvrtc

MAX_ROWS = 4096
MAX_FEATURES = 4096
MAX_CLUSTERS = 256
MAX_CENTRE_VALUES = 1_048_576
MAX_WORK = 1_000_000_000
MAX_SCRATCH_BYTES = 32 * 1024 * 1024
MAX_CUDA_LIVE_BYTES = 256 * 1024 * 1024
VALUE_TILE = 65_536
UPDATE_TILE_ROWS = 1024
EXACT_COUNT_LIMIT = 9_007_199_254_740_992
SKLEARN_VERSION = '1.9.1'
SKLEARN_MINIBATCH_SHA256 = 'b6ffd63a5a9c6f9162d77e1568911fd3540c4ceaaa8d5243cb341470e758fff5'
_CUDA_SOURCE = r'''
extern "C" __global__ void init_accum(
    const double* centres, const double* counts, double* accum, int k, int f) {
    int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q < k*f) accum[q] = centres[q] * counts[q/f];
}
extern "C" __global__ void count_rows(
    const unsigned char* labels, double* counts, int start, int rows, int k) {
    int cluster = blockIdx.x * blockDim.x + threadIdx.x;
    if (cluster >= k) return;
    double count = counts[cluster];
    for (int r=0; r<rows; ++r)
        if ((int)labels[start+r] == cluster) count = count + 1.0;
    counts[cluster] = count;
}
extern "C" __global__ void add_rows(
    const double* x, const unsigned char* labels, double* accum,
    int start, int rows, int k, int f) {
    int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= k*f) return;
    int cluster = q/f, feature = q%f;
    double value = accum[q];
    for (int r=0; r<rows; ++r) {
        int sample = start+r;
        if ((int)labels[sample] == cluster)
            value = value + x[(long long)sample*f+feature];
    }
    accum[q] = value;
}
extern "C" __global__ void finish_update(
    const double* old_centres, const double* old_counts, const double* batch_counts,
    const double* accum, double* centres, double* counts, int k, int f) {
    int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= k*f) return;
    int cluster = q/f, feature = q%f;
    double total = batch_counts[cluster] == 0.0 ? old_counts[cluster]
        : old_counts[cluster] + batch_counts[cluster];
    if (feature == 0) counts[cluster] = total;
    if (batch_counts[cluster] == 0.0) centres[q] = old_centres[q];
    else {
        double alpha = 1.0 / total;
        centres[q] = accum[q] * alpha;
    }
}
'''


class KMeansMiniBatchError(ValueError):
    """Invalid unit-weight update or an update resource limit."""


class KMeansMiniBatchCancelled(RuntimeError):
    """Update cancelled without returning partial centres or counts."""


@dataclass(frozen=True, slots=True)
class KMeansMiniBatchUpdate:
    centres: torch.Tensor
    counts: torch.Tensor


def _poll(cancel):
    if cancel is not None and cancel():
        raise KMeansMiniBatchCancelled('K-means minibatch centre update was cancelled')


def _checked_poll(cancel, values, versions):
    _poll(cancel)
    if tuple(value._version for value in values) != versions:
        raise KMeansMiniBatchError('minibatch inputs changed during the update')


def _bounded_values(value, minimum, maximum, cancel):
    flat = value.reshape(-1)
    for start in range(0, flat.numel(), VALUE_TILE):
        _poll(cancel)
        tile = flat[start:start+VALUE_TILE]
        if (not bool(torch.isfinite(tile).all().item())
                or bool(((tile < minimum) | (tile > maximum)).any().item())):
            raise KMeansMiniBatchError('numeric values are outside the bounded update domain')


def _preflight(features, labels, centres, counts, cancel):
    if cancel is not None and not callable(cancel):
        raise TypeError('cancel must be a zero-argument callable or None')
    _poll(cancel)
    values = (features, labels, centres, counts)
    if not all(torch.is_tensor(value) and value.device.type == 'cuda'
            and value.layout == torch.strided and value.is_contiguous() for value in values):
        raise KMeansMiniBatchError('all update inputs must be contiguous strided CUDA tensors')
    if (features.dtype != torch.float64 or centres.dtype != torch.float64
            or counts.dtype != torch.float64 or labels.dtype != torch.uint8
            or features.ndim != 2 or centres.ndim != 2 or counts.ndim != 1 or labels.ndim != 1
            or any(value.device != features.device for value in values)):
        raise KMeansMiniBatchError('require same-device float64 rows/centres/counts and byte labels')
    rows, width = map(int, features.shape)
    clusters = int(centres.shape[0])
    if (not 1 <= rows <= MAX_ROWS or not 1 <= width <= MAX_FEATURES
            or not 1 <= clusters <= MAX_CLUSTERS or clusters*width > MAX_CENTRE_VALUES
            or tuple(centres.shape) != (clusters, width)
            or labels.numel() != rows or counts.numel() != clusters):
        raise KMeansMiniBatchError('update shape/count limits are exceeded')
    # Charge row visits in both kernels, plus init/final centre-element visits.
    work = clusters * (rows*(width+1) + 2*width)
    if work > MAX_WORK:
        raise KMeansMiniBatchError('minibatch centre-update work exceeds its bound')
    # Output/accumulator pair plus count vectors and bounded value-check scratch.
    scratch = 16*clusters*width + 16*clusters + 3*VALUE_TILE + 2*1024*1024
    live = sum(value.numel()*value.element_size() for value in values) + scratch
    if scratch > MAX_SCRATCH_BYTES or live > MAX_CUDA_LIVE_BYTES:
        raise KMeansMiniBatchError('minibatch update exceeds its scratch/live-storage bound')
    _bounded_values(features, 0.0, 255.0, cancel)
    _bounded_values(centres, 0.0, 255.0, cancel)
    _bounded_values(counts, 0.0, float(EXACT_COUNT_LIMIT-rows), cancel)
    if bool((counts != counts.round()).any().item()):
        raise KMeansMiniBatchError('unit-weight accumulated counts must be exact nonnegative integers')
    if int(labels.max().item()) >= clusters:
        raise KMeansMiniBatchError('supplied labels must name a selected cluster')
    _poll(cancel)
    return rows, width, clusters


def update_unit_weight_centres(features, labels, centres, counts, *, cancel=None):
    """Return new CUDA float64 centres/counts using source row-ordered additions.

    All weights are one, matching MiniBatchKMeans.fit's sampled batches. Supplied
    labels are not recomputed here. Empty clusters preserve old centres/counts.
    No inputs are mutated; all outputs remain private until the final checks.
    """
    values = (features, labels, centres, counts)
    versions = tuple(value._version if torch.is_tensor(value) else None for value in values)
    rows, width, clusters = _preflight(features, labels, centres, counts, cancel)
    if tuple(value._version for value in values) != versions:
        raise KMeansMiniBatchError('minibatch inputs changed during preflight')
    with torch.cuda.device(features.device):
        for value in values:
            value.record_stream(torch.cuda.current_stream(features.device))
        program = _nvrtc.fixed_kernel_program(_CUDA_SOURCE, features.device)
        _checked_poll(cancel, values, versions)
        accum = torch.empty_like(centres)
        batch_counts = torch.zeros_like(counts)
        program.launch_1d('init_accum', (centres, counts, accum), (clusters, width), clusters*width)
        for start in range(0, rows, UPDATE_TILE_ROWS):
            _checked_poll(cancel, values, versions)
            selected = min(UPDATE_TILE_ROWS, rows-start)
            program.launch_1d('count_rows', (labels, batch_counts), (start, selected, clusters), clusters)
            program.launch_1d('add_rows', (features, labels, accum),
                (start, selected, clusters, width), clusters*width)
        _checked_poll(cancel, values, versions)
        result = KMeansMiniBatchUpdate(torch.empty_like(centres), torch.empty_like(counts))
        program.launch_1d('finish_update', (centres, counts, batch_counts, accum,
            result.centres, result.counts), (clusters, width), clusters*width)
        _checked_poll(cancel, values, versions)
        return result
