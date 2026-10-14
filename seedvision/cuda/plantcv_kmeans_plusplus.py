"""Exact CUDA k-means++ initialization for rounded byte-valued unit-weight rows.

The pinned sklearn algorithm and BSD notice are recorded in
docs/SKLEARN_KMEANS_NOTICES.md. This component selects initial centres; it does
not perform MiniBatch fitting, initialization-subset selection or validation.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch

from seedvision.cuda import _nvrtc
from seedvision.cuda import plantcv_kmeans_random as _random

MAX_SAMPLES = 1_000_000
MAX_FEATURES = 4096
MAX_CLUSTERS = 256
MAX_CENTRE_VALUES = 1_048_576
MAX_DATASET_BYTES = 128 * 1024 * 1024
MAX_WORK = 1_000_000_000
MAX_CUDA_LIVE_BYTES = 512 * 1024 * 1024
MAX_TILE_ROWS = 4096
VALUE_TILE = 65_536
ALLOCATION_HEADROOM_BYTES = 2 * 1024 * 1024
EXACT_INTEGER_LIMIT = 9_007_199_254_740_992
SKLEARN_VERSION = '1.9.1'
SKLEARN_KMEANS_SHA256 = '0cf335b8b9b4abe67259ccdcbaf07f02bd7feee31e63a06a1cce32a87fcd375d'
_CUDA_SOURCE = r'''
extern "C" __global__ void squared_rows(
    const unsigned short* x, const double* closest, double* output,
    int f, int centre, int start, int rows, int use_minimum) {
    int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= rows) return;
    long long distance = 0;
    for (int feature=0; feature<f; ++feature) {
        int delta = (int)x[(long long)(start+q)*f+feature]
            - (int)x[(long long)centre*f+feature];
        distance += (long long)delta * delta;
    }
    double value = (double)distance;
    if (use_minimum && closest[start+q] < value) value = closest[start+q];
    output[q] = value;
}
'''


class KMeansPlusPlusError(ValueError):
    """Invalid byte-valued initialization or whole-operation limit."""


class KMeansPlusPlusCancelled(RuntimeError):
    """Initialization cancelled without returning centres."""


@dataclass(frozen=True, slots=True)
class KMeansPlusPlusPlan:
    rows: int
    features: int
    clusters: int
    local_trials: int
    tile_rows: int
    work: int
    workspace_bytes: int
    peak_live_bytes: int


@dataclass(frozen=True, slots=True)
class KMeansPlusPlusResult:
    centre_indices: tuple[int, ...]
    centres: torch.Tensor
    potential: float


def _checkpoint(features, version, cancel):
    if features._version != version:
        raise KMeansPlusPlusError('feature matrix changed during initialization')
    cancelled = cancel() if cancel is not None else False
    if features._version != version:
        raise KMeansPlusPlusError('feature matrix changed during initialization callback')
    if cancelled:
        raise KMeansPlusPlusCancelled('K-means++ initialization was cancelled')


def preflight_kmeans_plusplus(features, n_clusters, *, tile_rows=4096):
    """Bound geometry, integer exactness, work and simultaneous CUDA allocations."""
    if (not torch.is_tensor(features) or features.device.type != 'cuda'
            or features.layout != torch.strided or features.dtype != torch.uint16
            or not features.is_contiguous() or features.ndim != 2):
        raise KMeansPlusPlusError('features must be contiguous strided CUDA uint16 rows')
    rows, width = map(int, features.shape)
    if (not 1 <= rows <= MAX_SAMPLES or not 1 <= width <= MAX_FEATURES
            or type(n_clusters) is not int or not 1 <= n_clusters <= min(MAX_CLUSTERS, rows)
            or n_clusters*width > MAX_CENTRE_VALUES):
        raise KMeansPlusPlusError('initialization shape/cluster limits are exceeded')
    if type(tile_rows) is not int or not 1 <= tile_rows <= MAX_TILE_ROWS:
        raise KMeansPlusPlusError('tile_rows must be an integer in [1,4096]')
    data_bytes = rows*width*2
    if data_bytes > MAX_DATASET_BYTES:
        raise KMeansPlusPlusError('byte-valued feature dataset exceeds its byte cap')
    if rows*width*65025 >= EXACT_INTEGER_LIMIT:
        raise KMeansPlusPlusError('integer distances/potentials exceed the exact float64 domain')
    trials = 2 + int(math.log(n_clusters))
    passes = 1 + (n_clusters-1)*(trials+1)
    # Value validation, all distance/reduction/minimum passes and prefix visits.
    work = rows*width + rows*(width+2)*passes + rows*(n_clusters-1)
    if work > MAX_WORK:
        raise KMeansPlusPlusError('initialization work exceeds its global bound')
    selected = min(rows, tile_rows)
    # Closest/prefix coexist; distance tiles use one binary64 output. A fixed
    # margin covers validation predicates, tiny candidates and scan/reduce scratch.
    workspace = 16*rows + 8*n_clusters*width + 8*selected + 3*VALUE_TILE + ALLOCATION_HEADROOM_BYTES
    peak_live = data_bytes + workspace
    if peak_live > MAX_CUDA_LIVE_BYTES:
        raise KMeansPlusPlusError('initialization exceeds its global live-storage bound')
    return KMeansPlusPlusPlan(rows, width, n_clusters, trials, selected, work, workspace, peak_live)


def _distance_tile(program, features, closest, centre, start, rows, use_minimum):
    output = torch.empty((rows,), dtype=torch.float64, device=features.device)
    program.launch_1d('squared_rows', (features, closest, output),
        (int(features.shape[1]), centre, start, rows, int(use_minimum)), rows)
    return output


def initialize_kmeans_plusplus(features, n_clusters, random_state, *, cancel=None, tile_rows=4096):
    """Return source-ordered float64 CUDA centres and compact selected row IDs.

    Every squared byte difference, positive prefix and potential is an integer
    below 2**53 within preflight bounds. Source binary64 norms/dots and the CUDA
    integer distance kernel therefore agree exactly. Preserve normalized unit-
    weight first choice, binary64 uniform*potential, left search and first ties.
    Cancellation may consume RNG state but never publishes incomplete centres.
    """
    if cancel is not None and not callable(cancel):
        raise TypeError('cancel must be a zero-argument callable or None')
    if not isinstance(random_state, _random.MiniBatchRandomSchedule):
        raise KMeansPlusPlusError('random_state must be a MiniBatchRandomSchedule')
    if not torch.is_tensor(features):
        raise KMeansPlusPlusError('features must be CUDA uint16 rows')
    version = features._version
    _checkpoint(features, version, cancel)
    plan = preflight_kmeans_plusplus(features, n_clusters, tile_rows=tile_rows)
    signed = features.view(torch.int16).reshape(-1)
    for start in range(0, signed.numel(), VALUE_TILE):
        _checkpoint(features, version, cancel)
        tile = signed[start:start+VALUE_TILE]
        if bool(((tile < 0) | (tile > 255)).any().item()):
            raise KMeansPlusPlusError('features must contain rounded byte values in [0,255]')
        del tile
    _checkpoint(features, version, cancel)
    with torch.cuda.device(features.device):
        features.record_stream(torch.cuda.current_stream(features.device))
        program = _nvrtc.fixed_kernel_program(_CUDA_SOURCE, features.device)
        _checkpoint(features, version, cancel)
        first = random_state.first_center(plan.rows)
        _checkpoint(features, version, cancel)
        if type(first) is not int or not 0 <= first < plan.rows:
            raise KMeansPlusPlusError('RNG returned an invalid first centre index')
        centres = torch.empty((plan.clusters, plan.features), dtype=torch.float64, device=features.device)
        centres[0].copy_(features[first])
        closest = torch.empty((plan.rows,), dtype=torch.float64, device=features.device)
        for start in range(0, plan.rows, plan.tile_rows):
            _checkpoint(features, version, cancel)
            count = min(plan.tile_rows, plan.rows-start)
            distances = _distance_tile(program, features, closest, first, start, count, False)
            closest[start:start+count].copy_(distances)
            del distances
        potential = float(closest.sum().item())
        ids = [first]
        for centre_number in range(1, plan.clusters):
            _checkpoint(features, version, cancel)
            uniforms = random_state.local_trials(plan.local_trials)
            _checkpoint(features, version, cancel)
            if (not isinstance(uniforms, np.ndarray) or uniforms.dtype != np.float64
                    or uniforms.shape != (plan.local_trials,)
                    or not bool(np.isfinite(uniforms).all())
                    or bool(((uniforms < 0) | (uniforms >= 1)).any())):
                raise KMeansPlusPlusError('RNG returned invalid local-trial uniforms')
            targets = [float(value)*potential for value in uniforms]
            cumulative = torch.cumsum(closest, dim=0)
            target_tensor = torch.tensor(targets, dtype=torch.float64, device=features.device)
            candidates_gpu = torch.searchsorted(cumulative, target_tensor, right=False)
            candidates_gpu.clamp_(max=plan.rows-1)
            candidates = candidates_gpu.cpu().tolist()
            del candidates_gpu, target_tensor, cumulative
            potentials = []
            for candidate in candidates:
                _checkpoint(features, version, cancel)
                candidate_potential = torch.zeros((), dtype=torch.float64, device=features.device)
                for start in range(0, plan.rows, plan.tile_rows):
                    _checkpoint(features, version, cancel)
                    count = min(plan.tile_rows, plan.rows-start)
                    distances = _distance_tile(program, features, closest, candidate, start, count, True)
                    candidate_potential.add_(distances.sum())
                    del distances
                potentials.append(float(candidate_potential.item()))
                del candidate_potential
            best = min(range(plan.local_trials), key=potentials.__getitem__)
            selected = candidates[best]
            potential = potentials[best]
            _checkpoint(features, version, cancel)
            centres[centre_number].copy_(features[selected])
            ids.append(selected)
            for start in range(0, plan.rows, plan.tile_rows):
                _checkpoint(features, version, cancel)
                count = min(plan.tile_rows, plan.rows-start)
                distances = _distance_tile(program, features, closest, selected, start, count, True)
                closest[start:start+count].copy_(distances)
                del distances
        _checkpoint(features, version, cancel)
        return KMeansPlusPlusResult(tuple(ids), centres, potential)
