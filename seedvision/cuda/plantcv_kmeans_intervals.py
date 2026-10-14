"""Compact certified inertia/control arithmetic for pinned CUDA MiniBatch fitting.

An interval encloses the source's binary64 value; it is not a replacement value.
Known point values execute the original scalar operations without widening.
Dense row losses and their serial fold are produced on CUDA by the scorer.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math

from seedvision.cuda import plantcv_kmeans_control as _control
from seedvision.cuda.plantcv_kmeans_classifier import _decision_margin_bound

MAX_ROWS = 1_000_000
MAX_FEATURES = 4096
EXACT_INTEGER_LIMIT = 9_007_199_254_740_992
ALL_REDUCTIONS_PROFILE = 'all-valid-openmp-reductions-v1'
RECORDED_THREADS_PROFILE = 'recorded-openmp-threads-v1'


class KMeansIntervalError(ValueError):
    """Invalid compact numeric interval or control state."""


class KMeansNumericalAmbiguity(KMeansIntervalError):
    """A source comparison is not determined by its certified intervals."""


def source_reduction_profile(source_n_threads):
    """Name explicit all-order certification or observed source provenance.

    A recorded count does not narrow the proof: both profiles use the same
    bound over all legal reduction orders. None never infers a host setting.
    """
    if source_n_threads is None:
        return ALL_REDUCTIONS_PROFILE
    if type(source_n_threads) is int and source_n_threads > 0:
        return RECORDED_THREADS_PROFILE
    raise KMeansIntervalError('source thread provenance must be None or an observed positive integer')


def _number(value):
    if type(value) not in (int, float):
        raise KMeansIntervalError('interval values must be finite nonnegative numbers')
    try:
        value = float(value)
    except (OverflowError, ValueError) as error:
        raise KMeansIntervalError('interval value exceeds binary64') from error
    if not math.isfinite(value) or value < 0.0:
        raise KMeansIntervalError('interval values must be finite and nonnegative')
    return value


@dataclass(frozen=True, slots=True)
class F64Interval:
    lo: float
    hi: float

    def __post_init__(self):
        lo, hi = _number(self.lo), _number(self.hi)
        if hi < lo:
            raise KMeansIntervalError('interval endpoints must be ordered')
        object.__setattr__(self, 'lo', lo)
        object.__setattr__(self, 'hi', hi)

    @classmethod
    def point(cls, value):
        value = _number(value)
        return cls(value, value)

    @property
    def is_point(self):
        return self.lo == self.hi


def _interval(value):
    if type(value) is not F64Interval:
        raise KMeansIntervalError('expected an immutable F64Interval')
    return value


def _down(value):
    return max(0.0, math.nextafter(value, -math.inf))


def _up(value):
    return math.nextafter(value, math.inf)


def interval_add(left, right):
    left, right = _interval(left), _interval(right)
    if left.is_point and right.is_point:
        return F64Interval.point(left.lo + right.lo)
    return F64Interval(_down(left.lo + right.lo), _up(left.hi + right.hi))


def interval_scale(value, factor):
    value, factor = _interval(value), _number(factor)
    if value.is_point:
        return F64Interval.point(value.lo * factor)
    if factor == 0.0:
        return F64Interval.point(0.0)
    return F64Interval(_down(value.lo * factor), _up(value.hi * factor))


def interval_divide(value, divisor):
    value = _interval(value)
    if type(divisor) is not int or divisor < 1:
        raise KMeansIntervalError('interval divisor must be a positive integer')
    if value.is_point:
        return F64Interval.point(value.lo / divisor)
    return F64Interval(_down(value.lo / divisor), _up(value.hi / divisor))


def strict_interval_less(candidate, incumbent):
    """Certify source '<', including exact equality as no improvement."""
    candidate, incumbent = _interval(candidate), _interval(incumbent)
    if candidate.hi < incumbent.lo:
        return True
    if candidate.lo >= incumbent.hi:
        return False
    raise KMeansNumericalAmbiguity('source strict inertia improvement is within the certified interval')


def _directed_fraction(value, *, upper):
    rounded = float(value)
    if not math.isfinite(rounded):
        raise KMeansIntervalError('certified bound exceeds finite binary64')
    actual = Fraction.from_float(rounded)
    if upper and actual < value:
        rounded = math.nextafter(rounded, math.inf)
    elif not upper and actual > value:
        rounded = math.nextafter(rounded, -math.inf)
    return max(0.0, rounded)


def source_inertia_interval(serial_sum, *, row_count, feature_count, source_n_threads, integral):
    """Bound the pinned source aggregate from a CUDA serial row-loss fold.

    Integral centres are the caller's proven property, not inferred from the
    scalar sum. Byte features/centres give exact integer row losses and every
    possible partial total is below2**53, regardless of OpenMP reduction order.

    For fractional centres, each backend's subtraction, square and grouped sum
    has at most(2F+32) rounded operations along a term's path, with byte-space
    absolute magnitude bounded by195075F. Four gamma-weighted magnitudes in the
    shared winner bound exceed both row-error bounds, including contraction and
    the bounded subtraction/product underflow terms (far below that reserve).
    Thus Erow also bounds the difference of the direct squared-distance rows.

    The serial CUDA sum is rounded, not the real sum of its binary64 rows. Undo
    that fold's gamma bound, add N*Erow for source row arithmetic, then expand for
    any valid source reduction. Nonnegative binary64 additions that remain
    subnormal are exact multiples of the minimum subnormal; no extra underflow
    term is needed for these additions. Overflow is excluded by the byte domain.
    Exact Fraction intermediates and directed final conversion prevent an
    inward-rounded bound. One-thread fractional cases are still intervals;
    fixture agreement alone does not prove compiler-wide row equality.
    """
    serial_sum = _number(serial_sum)
    source_reduction_profile(source_n_threads)
    if (type(row_count) is not int or not 1 <= row_count <= MAX_ROWS
            or type(feature_count) is not int or not 1 <= feature_count <= MAX_FEATURES
            or type(integral) is not bool):
        raise KMeansIntervalError('inertia geometry/thread provenance is outside its domain')
    largest = row_count * feature_count * 65025
    if largest >= EXACT_INTEGER_LIMIT:
        raise KMeansIntervalError('inertia domain exceeds exact integer totals')
    if integral:
        if not serial_sum.is_integer() or serial_sum > largest:
            raise KMeansIntervalError('integral inertia does not satisfy the proved byte bound')
        return F64Interval.point(serial_sum)
    error = row_count * Fraction.from_float(_decision_margin_bound(feature_count))
    observed = Fraction.from_float(serial_sum)
    if observed > largest + error:
        raise KMeansIntervalError('serial inertia exceeds its byte-domain bound')
    additions = row_count - 1
    gamma = Fraction(additions, EXACT_INTEGER_LIMIT - additions)
    lower_gpu = observed / (1 + gamma)
    upper_gpu = observed / (1 - gamma)
    lower_source = max(Fraction(0), lower_gpu - error) * (1 - gamma)
    upper_source = (upper_gpu + error) * (1 + gamma)
    return F64Interval(_directed_fraction(lower_source, upper=False),
        _directed_fraction(upper_source, upper=True))


@dataclass
class IntervalMiniBatchControl:
    """Source-ordered EWA decisions, committed only after certification."""
    batch_size: int
    # Already-scaled internal _tol; an exposed public tol needs its own producer.
    tol: F64Interval | float = 0.0
    max_no_improvement: int | None = 10
    ewa_inertia: F64Interval | None = None
    ewa_inertia_min: F64Interval | None = None
    no_improvement: int = 0

    def __post_init__(self):
        if type(self.tol) in (int, float):
            self.tol = F64Interval.point(self.tol)
        self._validate_state()

    def _validate_state(self):
        if type(self.batch_size) is not int or not 1 <= self.batch_size <= _control.MAX_BATCH_ROWS:
            raise KMeansIntervalError('batch size exceeds the compact control domain')
        _interval(self.tol)
        if self.max_no_improvement is not None and (
                type(self.max_no_improvement) is not int or self.max_no_improvement < 0):
            raise KMeansIntervalError('max_no_improvement must be nonnegative or None')
        if type(self.no_improvement) is not int or self.no_improvement < 0:
            raise KMeansIntervalError('no_improvement must be a nonnegative integer')
        for value in (self.ewa_inertia, self.ewa_inertia_min):
            if value is not None:
                _interval(value)

    def convergence(self, *, step, n_steps, n_samples, batch_inertia, centres_squared_diff=None):
        self._validate_state()
        if (type(step) is not int or type(n_steps) is not int or not 0 <= step < n_steps
                or type(n_samples) is not int or not self.batch_size <= n_samples <= MAX_ROWS):
            raise KMeansIntervalError('step/sample geometry exceeds the fitting domain')
        inertia = interval_divide(batch_inertia, self.batch_size)
        if self.tol.hi > 0.0:
            _interval(centres_squared_diff)
        if step == 0:
            return False
        if self.ewa_inertia is None:
            next_ewa = inertia
        else:
            alpha = self.batch_size * 2.0 / (n_samples + 1)
            alpha = min(alpha, 1)
            next_ewa = interval_add(interval_scale(self.ewa_inertia, 1-alpha),
                interval_scale(inertia, alpha))
        if self.tol.hi > 0.0:
            if centres_squared_diff.lo > self.tol.hi:
                pass
            elif self.tol.lo > 0.0 and centres_squared_diff.hi <= self.tol.lo:
                self.ewa_inertia = next_ewa
                return True
            else:
                raise KMeansNumericalAmbiguity('source centre-tolerance decision is within the certified interval')
        if self.ewa_inertia_min is None or strict_interval_less(next_ewa, self.ewa_inertia_min):
            next_min, next_count = next_ewa, 0
        else:
            next_min, next_count = self.ewa_inertia_min, self.no_improvement+1
        self.ewa_inertia = next_ewa
        self.ewa_inertia_min = next_min
        self.no_improvement = next_count
        return self.max_no_improvement is not None and next_count >= self.max_no_improvement
