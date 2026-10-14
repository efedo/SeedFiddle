"""Bounded compact CPU RandomState calls for pinned unit-weight CUDA fitting.

Dense features, centres, norms and distances belong on CUDA. Only the source's
random index vectors and small RNG/provenance state live here. Calls are explicit
so the fitting loop can preserve source branch order and RNG consumption.
"""
from __future__ import annotations

from hashlib import sha256
import struct

import numpy as np

MAX_SAMPLES = 1_000_000
MAX_BATCH_SIZE = 4096
MAX_LOCAL_TRIALS = 64
MAX_CPU_BYTES = 128 * 1024 * 1024
CPU_HEADROOM_BYTES = 64 * 1024
SKLEARN_VERSION = '1.9.1'
SKLEARN_KMEANS_SHA256 = '0cf335b8b9b4abe67259ccdcbaf07f02bd7feee31e63a06a1cce32a87fcd375d'


class KMeansRandomError(ValueError):
    """Unsupported seed, draw geometry or compact metadata budget."""


class MiniBatchRandomSchedule:
    """One seeded legacy RandomState stream; this helper does not fit a model."""

    def __init__(self, seed):
        if type(seed) is not int or not 0 <= seed <= 0xFFFFFFFF:
            raise KMeansRandomError('native RandomState requires an integer seed in [0,2**32-1]')
        self._seed = seed
        self._rng = np.random.RandomState(seed)
        self._draw_digest = sha256()
        self._calls = 0
        self._peak_cpu_bytes = 0

    @property
    def peak_cpu_bytes(self):
        """Conservative per-call allocation envelope, excluding caller-held outputs."""
        return self._peak_cpu_bytes

    @staticmethod
    def _sample_domain(n_samples):
        if type(n_samples) is not int or not 1 <= n_samples <= MAX_SAMPLES:
            raise KMeansRandomError('unsupported training sample count')

    def _sizes(self, n_samples, size, *, maximum=None, allow_zero=False):
        self._sample_domain(n_samples)
        upper = n_samples if maximum is None else maximum
        if type(size) is not int or not (0 if allow_zero else 1) <= size <= upper:
            raise KMeansRandomError('unsupported random draw size')

    def _budget(self, n_samples, size, *, weighted=False, without_replacement=False):
        # Eight bytes covers either C-long or pointer-width returned indices.
        # Weighted choice also holds a probability vector, CDF and uniform/
        # search-index temporaries. Without replacement may permute all N rows.
        peak = CPU_HEADROOM_BYTES + size*24
        if weighted:
            peak += n_samples*16
        if without_replacement:
            peak += n_samples*8
        if peak > MAX_CPU_BYTES:
            raise KMeansRandomError('random metadata allocation exceeds the CPU byte cap')
        self._peak_cpu_bytes = max(self._peak_cpu_bytes, peak)

    def _record(self, name, value):
        array = np.asarray(value)
        if not array.flags.c_contiguous:
            raise KMeansRandomError('random draw must be contiguous for bounded recording')
        self._draw_digest.update(name.encode('ascii')+b'\0')
        self._draw_digest.update(array.dtype.str.encode('ascii')+b'\0')
        self._draw_digest.update(struct.pack('<I', array.ndim))
        for dimension in array.shape:
            self._draw_digest.update(struct.pack('<Q', dimension))
        self._draw_digest.update(memoryview(array).cast('B'))
        self._calls += 1
        return value

    def validation_indices(self, n_samples, init_size):
        self._sizes(n_samples, init_size)
        self._budget(n_samples, init_size)
        # fit always draws this vector, even when init_size equals n_samples.
        return self._record('validation.randint', self._rng.randint(0, n_samples, init_size))

    def init_subset(self, n_samples, init_size):
        self._sizes(n_samples, init_size)
        if init_size == n_samples:
            # _init_centroids does not sample or consume state in this branch.
            return None
        self._budget(n_samples, init_size)
        return self._record('init.randint', self._rng.randint(0, n_samples, init_size))

    def first_center(self, n_samples):
        self._sample_domain(n_samples)
        self._budget(n_samples, 1, weighted=True)
        probabilities = np.ones(n_samples, dtype=np.float64)
        probabilities /= probabilities.sum()
        index = self._rng.choice(n_samples, p=probabilities)
        self._record('kmeans_plus_plus.first.choice', index)
        return int(index)

    def local_trials(self, count):
        if type(count) is not int or not 1 <= count <= MAX_LOCAL_TRIALS:
            raise KMeansRandomError('unsupported k-means++ local-trial count')
        self._budget(0, count)
        return self._record('kmeans_plus_plus.local.uniform', self._rng.uniform(size=count))

    def minibatch_indices(self, n_samples, batch_size):
        self._sizes(n_samples, batch_size, maximum=min(n_samples, MAX_BATCH_SIZE))
        self._budget(n_samples, batch_size, weighted=True)
        probabilities = np.ones(n_samples, dtype=np.float64)
        probabilities /= probabilities.sum()
        # Explicit p is essential: unit-weight choice consumes a different
        # stream from randint or choice(p=None).
        return self._record('minibatch.choice.weighted', self._rng.choice(
            n_samples, batch_size, p=probabilities, replace=True))

    def reassignment_indices(self, batch_size, count):
        self._sizes(batch_size, count, allow_zero=True)
        if batch_size > MAX_BATCH_SIZE:
            raise KMeansRandomError('unsupported reassignment batch size')
        self._budget(batch_size, count, without_replacement=True)
        return self._record('reassignment.choice.without_replacement', self._rng.choice(
            batch_size, replace=False, size=count))

    def identity(self):
        """Return bounded provenance and an exact post-call MT19937 state digest."""
        algorithm, keys, position, has_gauss, cached = self._rng.get_state()
        state = sha256(algorithm.encode('ascii')+b'\0')
        state.update(memoryview(keys.astype('<u4', copy=False)).cast('B'))
        state.update(struct.pack('<IId', int(position), int(has_gauss), float(cached)))
        return dict(schema='seedfiddle.kmeans-randomstate.v1', numpy=np.__version__,
            seed=self._seed, algorithm=algorithm, calls=self._calls,
            state_sha256=state.hexdigest(), draw_sha256=self._draw_digest.hexdigest())
