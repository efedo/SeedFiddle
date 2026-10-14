"""Pinned MiniBatchKMeans control arithmetic and compact reassignment planning.

Only scalar state and at most256 count values cross this boundary. Feature rows,
centres and counts used by the fitting loop remain on CUDA. The sklearn source
algorithm and BSD notice are recorded in docs/SKLEARN_KMEANS_NOTICES.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import math

import numpy as np

from seedvision.cuda import plantcv_kmeans_random as _random

MAX_BATCH_ROWS = 4096
MAX_CLUSTERS = 256
MAX_SAMPLES = 1_000_000
EXACT_COUNT_LIMIT = 9_007_199_254_740_992
SKLEARN_VERSION = '1.9.1'
SKLEARN_KMEANS_SHA256 = '0cf335b8b9b4abe67259ccdcbaf07f02bd7feee31e63a06a1cce32a87fcd375d'


class KMeansControlError(ValueError):
    """Invalid compact control state or reassignment snapshot."""


def _nonnegative_number(value, name):
    if type(value) not in (int, float):
        raise KMeansControlError(f'{name} must be a finite nonnegative number')
    try:
        result = float(value)
    except (OverflowError, ValueError) as error:
        raise KMeansControlError(f'{name} is outside its numeric domain') from error
    if not math.isfinite(result) or result < 0:
        raise KMeansControlError(f'{name} must be finite and nonnegative')
    return result


def _count_snapshot(counts):
    if (type(counts) is not np.ndarray or counts.dtype != np.float64
            or counts.ndim != 1 or not counts.flags.c_contiguous
            or not 1 <= counts.size <= MAX_CLUSTERS):
        raise KMeansControlError('counts must be a contiguous compact float64 vector')
    if (not bool(np.isfinite(counts).all()) or bool(((counts < 0) | (counts > EXACT_COUNT_LIMIT)).any())
            or bool((counts != np.floor(counts)).any())):
        raise KMeansControlError('unit-weight counts must be exact finite nonnegative integers')
    return sha256(memoryview(counts).cast('B')).hexdigest()


@dataclass
class MiniBatchControl:
    batch_size: int
    n_clusters: int
    # This is the source's internal already-scaled _tol, not its public tol.
    tol: float = 0.0
    max_no_improvement: int | None = 10
    n_since_last_reassign: int = 0
    ewa_inertia: float | None = None
    ewa_inertia_min: float | None = None
    no_improvement: int = 0

    def __post_init__(self):
        self._validate_state()

    def _validate_state(self):
        if type(self.batch_size) is not int or not 1 <= self.batch_size <= MAX_BATCH_ROWS:
            raise KMeansControlError('batch_size is outside the supported bound')
        if type(self.n_clusters) is not int or not 1 <= self.n_clusters <= MAX_CLUSTERS:
            raise KMeansControlError('n_clusters is outside the supported bound')
        _nonnegative_number(self.tol, 'scaled tolerance')
        if self.max_no_improvement is not None and (
                type(self.max_no_improvement) is not int or self.max_no_improvement < 0):
            raise KMeansControlError('max_no_improvement must be a nonnegative integer or None')
        for value in (self.n_since_last_reassign, self.no_improvement):
            if type(value) is not int or value < 0:
                raise KMeansControlError('control counters must be nonnegative integers')
        for value in (self.ewa_inertia, self.ewa_inertia_min):
            if value is not None:
                _nonnegative_number(value, 'smoothed inertia state')

    def random_reassign_due(self, counts):
        """Use pre-update counts after the minibatch draw, before the update."""
        self._validate_state()
        _count_snapshot(counts)
        if counts.size != self.n_clusters:
            raise KMeansControlError('count snapshot does not match n_clusters')
        next_count = self.n_since_last_reassign + self.batch_size
        due = bool((counts == 0).any()) or next_count >= 10*self.n_clusters
        self.n_since_last_reassign = 0 if due else next_count
        return due

    def convergence(self, *, step, n_steps, n_samples, centres_squared_diff, batch_inertia):
        """Apply the source scalar branch/order for a known batch-inertia value.

        The full fitter must certify its inertia/reduction contract before using
        this scalar method; a backend-dependent aggregate is not silently exact.
        """
        self._validate_state()
        if (type(step) is not int or type(n_steps) is not int or not 0 <= step < n_steps
                or type(n_samples) is not int or not 1 <= n_samples <= MAX_SAMPLES
                or self.batch_size > n_samples):
            raise KMeansControlError('step/sample geometry is outside the fitting loop')
        diff = _nonnegative_number(centres_squared_diff, 'centre difference')
        inertia = _nonnegative_number(batch_inertia, 'batch inertia')
        inertia /= self.batch_size
        if step == 0:
            return False
        if self.ewa_inertia is None:
            next_ewa = inertia
        else:
            alpha = self.batch_size*2.0/(n_samples+1)
            alpha = min(alpha, 1)
            next_ewa = self.ewa_inertia*(1-alpha) + inertia*alpha
        self.ewa_inertia = next_ewa
        if self.tol > 0.0 and diff <= self.tol:
            return True
        if self.ewa_inertia_min is None or self.ewa_inertia < self.ewa_inertia_min:
            self.no_improvement = 0
            self.ewa_inertia_min = self.ewa_inertia
        else:
            self.no_improvement += 1
        return self.max_no_improvement is not None and self.no_improvement >= self.max_no_improvement


@dataclass(frozen=True, slots=True)
class LowCountReassignment:
    to_reassign: tuple[bool, ...]
    new_rows: tuple[int, ...] | None
    counts_sha256: str

    @property
    def n_reassigns(self):
        return sum(self.to_reassign)


def select_low_count_reassignments(counts, *, batch_rows, reassignment_ratio, schedule):
    """Plan source-ordered centre writes from a post-update compact snapshot.

    Call only when random_reassign is due and ratio>0. The caller writes selected
    minibatch rows to CUDA centres before evaluating low_count_reset_value, which
    preserves the source's unconditional minimum expression and failure ordering.
    """
    digest = _count_snapshot(counts)
    if type(batch_rows) is not int or not 1 <= batch_rows <= MAX_BATCH_ROWS:
        raise KMeansControlError('batch_rows is outside the supported bound')
    ratio = _nonnegative_number(reassignment_ratio, 'reassignment ratio')
    if ratio == 0:
        raise KMeansControlError('call the reassignment branch only for a positive ratio')
    if not isinstance(schedule, _random.MiniBatchRandomSchedule):
        raise KMeansControlError('schedule must be a MiniBatchRandomSchedule')
    selected = counts < ratio*counts.max()
    if selected.sum() > 0.5*batch_rows:
        keep = np.argsort(counts)[int(0.5*batch_rows):]
        selected[keep] = False
    count = int(selected.sum())
    new_rows = None
    if count:
        rows = schedule.reassignment_indices(batch_rows, count)
        if (not isinstance(rows, np.ndarray) or rows.ndim != 1 or rows.size != count
                or not np.issubdtype(rows.dtype, np.integer)
                or bool(((rows < 0) | (rows >= batch_rows)).any())
                or np.unique(rows).size != count):
            raise KMeansControlError('schedule returned invalid without-replacement row IDs')
        new_rows = tuple(map(int, rows))
    if _count_snapshot(counts) != digest:
        raise KMeansControlError('count snapshot changed while selecting reassignments')
    return LowCountReassignment(tuple(map(bool, selected)), new_rows, digest)


def low_count_reset_value(counts, plan):
    """Evaluate after centre writes, even when the selected mask is empty."""
    digest = _count_snapshot(counts)
    if (type(plan) is not LowCountReassignment or type(plan.to_reassign) is not tuple
            or len(plan.to_reassign) != counts.size
            or any(type(value) is not bool for value in plan.to_reassign)
            or plan.counts_sha256 != digest):
        raise KMeansControlError('reassignment plan does not bind this count snapshot')
    return float(np.min(counts[~np.asarray(plan.to_reassign, dtype=bool)]))
