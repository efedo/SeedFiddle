"""Source-ordered bounded CUDA unit-weight MiniBatchKMeans fitting.

This implements the byte-feature k-means++ estimator path used by PlantCV.
Image/directory training ingestion and desktop/model export are separate layers.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import warnings

import numpy as np
import torch

from seedvision.cuda import plantcv_kmeans_control as _control
from seedvision.cuda import plantcv_kmeans_intervals as _intervals
from seedvision.cuda import plantcv_kmeans_minibatch as _update
from seedvision.cuda import plantcv_kmeans_plusplus as _plusplus
from seedvision.cuda import plantcv_kmeans_random as _random
from seedvision.cuda import plantcv_kmeans_tolerance as _tolerance
from seedvision.cuda import plantcv_kmeans_training_score as _score

MAX_ROWS = 1_000_000
MAX_FEATURES = 4096
MAX_CLUSTERS = 256
MAX_BATCH_ROWS = 4096
MAX_INITIALIZATIONS = 32
MAX_STEPS = 1_000_000
MAX_DATASET_BYTES = 128 * 1024 * 1024
MAX_TOTAL_WORK = 20_000_000_000
MAX_CUDA_LIVE_BYTES = 512 * 1024 * 1024
MAX_CPU_METADATA_BYTES = 128 * 1024 * 1024
VALUE_TILE = 65_536
HEADROOM_BYTES = 2 * 1024 * 1024
SKLEARN_KMEANS_SHA256 = '0cf335b8b9b4abe67259ccdcbaf07f02bd7feee31e63a06a1cce32a87fcd375d'


class KMeansFitError(ValueError):
    """Invalid fit parameters or whole-job resource bounds."""


class KMeansFitCancelled(RuntimeError):
    """Fitting cancelled without publishing a partial model."""


@dataclass(frozen=True, slots=True)
class KMeansFitPlan:
    rows: int
    features: int
    clusters: int
    batch_size: int
    init_size: int
    n_init: int
    n_steps: int
    adjusted_init_size: bool
    work: int
    peak_live_bytes: int
    cpu_metadata_bytes: int


@dataclass(frozen=True, slots=True)
class KMeansFitResult:
    centres: torch.Tensor
    counts: torch.Tensor
    labels: torch.Tensor | None
    row_distances: torch.Tensor | None
    inertia: _intervals.F64Interval
    scaled_tolerance: _intervals.F64Interval
    n_steps: int
    n_iter: int
    source_n_threads: int | None
    random_identity: dict
    plan: KMeansFitPlan

    @property
    def source_reduction_profile(self):
        return _intervals.source_reduction_profile(self.source_n_threads)


def _integer(value, name, maximum=None):
    if type(value) is not int or value < 1 or (maximum is not None and value > maximum):
        raise KMeansFitError(f'{name} must be a positive integer within the supported bound')
    return value


def _number(value, name):
    try:
        return _intervals._number(value)
    except ValueError as error:
        raise KMeansFitError(f'{name} must be finite and nonnegative') from error


def _score_bounds(rows, width, clusters):
    work = 2*rows*width+clusters*width+clusters*(clusters-1)//2*width+rows*clusters*(width+8)+rows
    workspace = _score.ALLOCATION_HEADROOM_BYTES+min(rows, _score.MAX_TILE_ROWS)*(8*clusters+128)+11*clusters+9*rows+8
    return work, workspace


def preflight_kmeans_fit(features, n_clusters, *, source_n_threads, seed=1, max_iter=100,
        batch_size=1024, n_init='auto', init_size=None, tol=0.0,
        max_no_improvement=10, reassignment_ratio=.01, compute_labels=True):
    if (not torch.is_tensor(features) or features.device.type != 'cuda'
            or features.layout != torch.strided or features.dtype != torch.uint16
            or features.ndim != 2 or not features.is_contiguous()):
        raise KMeansFitError('features must be contiguous strided CUDA uint16 byte rows')
    return preflight_kmeans_fit_shape(*map(int, features.shape), n_clusters,
        source_n_threads=source_n_threads, seed=seed, max_iter=max_iter,
        batch_size=batch_size, n_init=n_init, init_size=init_size, tol=tol,
        max_no_improvement=max_no_improvement, reassignment_ratio=reassignment_ratio,
        compute_labels=compute_labels)


def preflight_kmeans_fit_shape(rows, width, n_clusters, *, source_n_threads, seed=1,
        max_iter=100, batch_size=1024, n_init='auto', init_size=None, tol=0.0,
        max_no_improvement=10, reassignment_ratio=.01, compute_labels=True):
    """Resolve the identical fit envelope before a training dataset is allocated."""
    if (type(rows) is not int or type(width) is not int
            or not 1 <= rows <= MAX_ROWS or not 1 <= width <= MAX_FEATURES
            or 2*rows*width > MAX_DATASET_BYTES):
        raise KMeansFitError('feature dataset exceeds its geometry/byte bound')
    clusters = _integer(n_clusters, 'n_clusters', min(rows, MAX_CLUSTERS))
    if source_n_threads is not None:
        _integer(source_n_threads, 'source_n_threads')
    if type(seed) is not int or not 0 <= seed <= 0xFFFFFFFF:
        raise KMeansFitError('seed must be an integer in [0,2**32-1]')
    _integer(max_iter, 'max_iter')
    batch = min(_integer(batch_size, 'batch_size'), rows)
    if batch > MAX_BATCH_ROWS:
        raise KMeansFitError('resolved batch size exceeds its supported bound')
    initializations = 1 if type(n_init) is str and n_init == 'auto' else _integer(n_init, 'n_init', MAX_INITIALIZATIONS)
    adjusted = init_size is not None and _integer(init_size, 'init_size') < clusters
    initial = 3*batch if init_size is None else init_size
    if initial < clusters:
        initial = 3*clusters
    initial = min(initial, rows)
    tolerance = _number(tol, 'tol')
    _number(reassignment_ratio, 'reassignment_ratio')
    if max_no_improvement is not None and (type(max_no_improvement) is not int or max_no_improvement < 0):
        raise KMeansFitError('max_no_improvement must be nonnegative or None')
    if type(compute_labels) is not bool:
        raise KMeansFitError('compute_labels must be bool')
    steps = max_iter*rows//batch
    if steps > MAX_STEPS:
        raise KMeansFitError('fitting exceeds its total step bound')
    trials = 2+int(math.log(clusters))
    plus_work = initial*width + initial*(width+2)*(1+(clusters-1)*(trials+1)) + initial*(clusters-1)
    validation_work, validation_workspace = _score_bounds(initial, width, clusters)
    batch_work, batch_workspace = _score_bounds(batch, width, clusters)
    final_work, final_workspace = _score_bounds(rows, width, clusters)
    update_work = clusters*(batch*(width+1)+2*width)
    update_scratch = 16*clusters*width+16*clusters+3*_update.VALUE_TILE+2*1024*1024
    update_live = 8*batch*width+batch+8*clusters*width+8*clusters+update_scratch
    if (max(plus_work, validation_work, batch_work, final_work if compute_labels else 0) > _score.MAX_WORK
            or plus_work > _plusplus.MAX_WORK or update_work > _update.MAX_WORK
            or update_scratch > _update.MAX_SCRATCH_BYTES or update_live > _update.MAX_CUDA_LIVE_BYTES):
        raise KMeansFitError('a complete fitting stage exceeds its work/scratch/live bound')
    # Include all gathers, metadata/RNG probability/CDF visits, each initializer,
    # every scoring/update step and the optional final full-data pass. Nonzero
    # tolerance adds its complete moment and centre-shift work before any draw.
    work = rows*width + initializations*(plus_work+validation_work+initial*width+4*initial)
    work += initial*width + steps*(batch_work+update_work+2*batch*width+4*rows+8*batch+clusters)
    if compute_labels:
        work += final_work
    if tolerance > 0:
        work += 4*rows*width+steps*8*clusters*width
    if work > MAX_TOTAL_WORK:
        raise KMeansFitError('fitting exceeds its whole-job work bound')
    data, centre_bytes = 2*rows*width, 8*clusters*width
    plus_workspace = 16*initial+centre_bytes+8*min(initial, _plusplus.MAX_TILE_ROWS)+3*_plusplus.VALUE_TILE+_plusplus.ALLOCATION_HEADROOM_BYTES
    init_peak = data+4*initial*width+2*centre_bytes+8*initial+max(plus_workspace, validation_workspace)
    train_peak = data+10*batch*width+2*centre_bytes+64*clusters+max(update_scratch, batch_workspace, HEADROOM_BYTES+8*clusters*width)
    final_peak = data+centre_bytes+8*clusters+(final_workspace if compute_labels else 0)
    moment_blocks = (rows+_tolerance.MOMENT_TILE_ROWS-1)//_tolerance.MOMENT_TILE_ROWS
    tolerance_peak = data+17*moment_blocks*width+17*width+HEADROOM_BYTES if tolerance > 0 else data
    peak = max(init_peak, train_peak, final_peak, tolerance_peak)+HEADROOM_BYTES
    cpu_bytes = _random.CPU_HEADROOM_BYTES+48*rows+24*initial+16*batch+32*clusters+16*width
    if peak > MAX_CUDA_LIVE_BYTES or cpu_bytes > MAX_CPU_METADATA_BYTES:
        raise KMeansFitError('fitting exceeds its whole-job CUDA/CPU storage bound')
    return KMeansFitPlan(rows, width, clusters, batch, initial, initializations, steps, adjusted, work, peak, cpu_bytes)


def _gather(features, indices, count, guard):
    guard()
    if (type(indices) is not np.ndarray or indices.ndim != 1 or indices.size != count
            or not np.issubdtype(indices.dtype, np.integer)
            or bool(((indices < 0) | (indices >= features.shape[0])).any())):
        raise KMeansFitError('random schedule returned invalid training row indices')
    gpu_indices = torch.tensor(indices, dtype=torch.int64, device=features.device)
    guard()
    result = features.view(torch.int16).index_select(0, gpu_indices).view(torch.uint16)
    guard()
    return result


def fit_kmeans_training_rows(features, n_clusters, *, source_n_threads, seed=1, max_iter=100,
        batch_size=1024, n_init='auto', init_size=None, tol=0.0,
        max_no_improvement=10, reassignment_ratio=.01, compute_labels=True, cancel=None):
    """Fit the complete bounded byte-feature k-means++ MiniBatch estimator path.

    The source thread count is explicit provenance. Certified intervals propagate
    through final inertia and convergence; ambiguous decisions publish no model.
    This operation mutates no caller tensor and keeps every dense matrix CUDA.
    """
    if not torch.is_tensor(features):
        raise KMeansFitError('features must be a CUDA tensor')
    if cancel is not None and not callable(cancel):
        raise KMeansFitError('cancel must be callable or None')
    version = features._version
    def guard():
        if features._version != version:
            raise KMeansFitError('training features changed during fitting')
        cancelled = cancel() if cancel is not None else False
        if features._version != version:
            raise KMeansFitError('training features changed during callback')
        if type(cancelled) is not bool:
            raise KMeansFitError('cancel callback must return bool')
        if cancelled:
            raise KMeansFitCancelled('K-means fitting was cancelled')
        return False
    guard()
    plan = preflight_kmeans_fit(features, n_clusters, source_n_threads=source_n_threads, seed=seed,
        max_iter=max_iter, batch_size=batch_size, n_init=n_init, init_size=init_size, tol=tol,
        max_no_improvement=max_no_improvement, reassignment_ratio=reassignment_ratio, compute_labels=compute_labels)
    flat = features.view(torch.int16).reshape(-1)
    for start in range(0, flat.numel(), VALUE_TILE):
        guard()
        tile = flat[start:start+VALUE_TILE]
        if bool(((tile < 0) | (tile > 255)).any().item()):
            raise KMeansFitError('training features must contain rounded bytes in [0,255]')
    guard()
    scaled_tol = _tolerance.scaled_kmeans_tolerance(features, tol, cancel=guard)
    if plan.adjusted_init_size:
        warnings.warn(f'init_size={init_size} should be larger than n_clusters={n_clusters}. '
            'Setting it to min(3*n_clusters, n_samples)', RuntimeWarning, stacklevel=2)
    schedule = _random.MiniBatchRandomSchedule(seed)
    with torch.cuda.device(features.device):
        features.record_stream(torch.cuda.current_stream(features.device))
        indices = schedule.validation_indices(plan.rows, plan.init_size)
        guard()
        validation = _gather(features, indices, plan.init_size, guard)
        del indices
        best_centres, best_inertia = None, None
        for _ in range(plan.n_init):
            guard()
            indices = schedule.init_subset(plan.rows, plan.init_size)
            guard()
            subset = features if indices is None else _gather(features, indices, plan.init_size, guard)
            candidate = _plusplus.initialize_kmeans_plusplus(subset, plan.clusters, schedule, cancel=guard)
            del subset, indices
            guard()
            score = _score.score_kmeans_training_rows(validation, candidate.centres,
                source_n_threads=source_n_threads, cancel=guard)
            if best_inertia is None or _intervals.strict_interval_less(score.inertia, best_inertia):
                best_centres, best_inertia = candidate.centres, score.inertia
            del candidate, score
        del validation
        centres = best_centres
        del best_centres
        counts = torch.zeros((plan.clusters,), dtype=torch.float64, device=features.device)
        timer = _control.MiniBatchControl(plan.batch_size, plan.clusters)
        convergence = _intervals.IntervalMiniBatchControl(plan.batch_size, tol=scaled_tol,
            max_no_improvement=max_no_improvement)
        for step in range(plan.n_steps):
            guard()
            indices = schedule.minibatch_indices(plan.rows, plan.batch_size)
            guard()
            batch = _gather(features, indices, plan.batch_size, guard)
            del indices
            due = timer.random_reassign_due(counts.cpu().numpy())
            guard()
            score = _score.score_kmeans_training_rows(batch, centres, source_n_threads=source_n_threads, cancel=guard)
            inertia, labels = score.inertia, score.labels
            del score
            batch_float = batch.to(torch.float64)
            guard()
            updated = _update.update_unit_weight_centres(batch_float, labels, centres, counts, cancel=guard)
            del labels
            guard()
            if due and reassignment_ratio > 0:
                snapshot = updated.counts.cpu().numpy()
                reassignment = _control.select_low_count_reassignments(snapshot, batch_rows=plan.batch_size,
                    reassignment_ratio=reassignment_ratio, schedule=schedule)
                guard()
                mask = torch.tensor(reassignment.to_reassign, dtype=torch.bool, device=features.device)
                if reassignment.n_reassigns:
                    row_ids = torch.tensor(reassignment.new_rows, dtype=torch.int64, device=features.device)
                    updated.centres[mask] = batch_float[row_ids]
                    del row_ids
                # Preserve source writes/RNG before the all-selected min(empty)
                # failure. Even an empty selection evaluates this RHS minimum.
                updated.counts[mask] = _control.low_count_reset_value(snapshot, reassignment)
                del mask, snapshot, reassignment
            guard()
            difference = _tolerance.centre_shift_interval(centres, updated.centres, cancel=guard) if scaled_tol.hi > 0 else None
            centres, counts = updated.centres, updated.counts
            del updated, batch, batch_float
            stop = convergence.convergence(step=step, n_steps=plan.n_steps, n_samples=plan.rows,
                batch_inertia=inertia, centres_squared_diff=difference)
            guard()
            if stop:
                break
        completed = step+1
        if compute_labels:
            score = _score.score_kmeans_training_rows(features, centres, source_n_threads=source_n_threads, cancel=guard)
            labels, row_distances, inertia = score.labels, score.row_distances, score.inertia
        else:
            labels, row_distances = None, None
            if convergence.ewa_inertia is None:
                # The pinned estimator also raises after a one-step fit when
                # compute_labels=False, rather than inventing an EWA value.
                raise TypeError("unsupported operand type(s) for *: 'NoneType' and 'float'")
            inertia = _intervals.interval_scale(convergence.ewa_inertia, float(plan.rows))
        guard()
        return KMeansFitResult(centres, counts, labels, row_distances, inertia, scaled_tol, completed,
            (completed*plan.batch_size+plan.rows-1)//plan.rows, source_n_threads, schedule.identity(), plan)
