"""CUDA byte moments and certified source tolerance/centre-shift producers."""
from __future__ import annotations

from fractions import Fraction

import torch

from seedvision.cuda import _nvrtc
from seedvision.cuda import plantcv_kmeans_intervals as _intervals

MAX_ROWS = 1_000_000
MAX_FEATURES = 4096
MAX_CENTRE_VALUES = 1_048_576
MAX_DATASET_BYTES = 128 * 1024 * 1024
MAX_WORK = 2_000_000_000
MAX_SCRATCH_BYTES = 32 * 1024 * 1024
MAX_CUDA_LIVE_BYTES = 512 * 1024 * 1024
MOMENT_TILE_ROWS = 4096
SHIFT_TILE_VALUES = 256
VALUE_TILE = 65_536
HEADROOM_BYTES = 2 * 1024 * 1024
EXACT_INTEGER_LIMIT = 9_007_199_254_740_992
NUMPY_VERSION = '2.2.6'
NUMPY_METHODS_SHA256 = '41068be3404b05b582867995683f73619b41a9fc72b9f0cc7983f55f53131146'
SKLEARN_KMEANS_SHA256 = '0cf335b8b9b4abe67259ccdcbaf07f02bd7feee31e63a06a1cce32a87fcd375d'
_CUDA_SOURCE = r'''
extern "C" __global__ void moment_partials(const unsigned short* x,
    long long* sums, long long* squares, unsigned char* valid,
    int rows, int f, int tile, int blocks) {
    int q=blockIdx.x*blockDim.x+threadIdx.x;
    if (q>=blocks*f) return;
    int j=q%f, block=q/f;
    int start=block*tile, end=start+tile;
    if (end>rows) end=rows;
    long long s=0, sq=0;
    unsigned char ok=1;
    for (int r=start; r<end; ++r) {
        long long v=(long long)x[(long long)r*f+j];
        if (v>255) ok=0;
        s+=v; sq+=v*v;
    }
    sums[q]=s; squares[q]=sq; valid[q]=ok;
}
extern "C" __global__ void finish_moments(const long long* sums,
    const long long* squares, const unsigned char* valid,
    long long* totals, unsigned char* facts, int f, int blocks) {
    int j=blockIdx.x*blockDim.x+threadIdx.x;
    if (j>=f) return;
    long long s=0, sq=0; unsigned char ok=1;
    for (int b=0; b<blocks; ++b) {
        s+=sums[(long long)b*f+j]; sq+=squares[(long long)b*f+j];
        if (!valid[(long long)b*f+j]) ok=0;
    }
    totals[j]=s; totals[f+j]=sq; facts[j]=ok;
}
extern "C" __global__ void shift_partials(const double* old_centres,
    const double* new_centres, double* partials, int values, int tile, int blocks) {
    int b=blockIdx.x*blockDim.x+threadIdx.x;
    if (b>=blocks) return;
    int start=b*tile, end=start+tile;
    if (end>values) end=values;
    double total=0.0;
    for (int j=start; j<end; ++j) {
        double d=__dsub_rn(new_centres[j],old_centres[j]);
        total=__dadd_rn(total,__dmul_rn(d,d));
    }
    partials[b]=total;
}
extern "C" __global__ void finish_shift(const double* partials, double* output, int blocks) {
    if (blockIdx.x!=0 || threadIdx.x!=0) return;
    double total=0.0;
    for (int b=0; b<blocks; ++b) total=__dadd_rn(total,partials[b]);
    output[0]=total;
}
'''


class KMeansToleranceError(ValueError):
    """Invalid tolerance tensors or producer resource/numeric bounds."""


class KMeansToleranceCancelled(RuntimeError):
    """Tolerance/shift cancelled before publishing a certified interval."""


def _checkpoint(values, versions, cancel):
    if tuple(value._version for value in values) != versions:
        raise KMeansToleranceError('tolerance inputs changed during calculation')
    result = cancel() if cancel is not None else False
    if tuple(value._version for value in values) != versions:
        raise KMeansToleranceError('tolerance inputs changed during callback')
    if type(result) is not bool:
        raise KMeansToleranceError('cancel callback must return bool')
    if result:
        raise KMeansToleranceCancelled('K-means tolerance calculation was cancelled')


def _gamma(additions):
    return Fraction(additions, EXACT_INTEGER_LIMIT-additions)


def _bounds(lower, upper):
    try:
        return _intervals.F64Interval(_intervals._directed_fraction(max(Fraction(0), lower), upper=False),
            _intervals._directed_fraction(upper, upper=True))
    except (OverflowError, ValueError) as error:
        raise KMeansToleranceError('tolerance interval exceeds finite binary64') from error


def _shape(value, dtype):
    if (not torch.is_tensor(value) or value.device.type != 'cuda'
            or value.layout != torch.strided or value.dtype != dtype
            or value.ndim != 2 or not value.is_contiguous()):
        raise KMeansToleranceError('expected contiguous strided CUDA matrix of the declared dtype')


def _budget(work, scratch, input_bytes):
    if work > MAX_WORK or scratch > MAX_SCRATCH_BYTES or input_bytes+scratch > MAX_CUDA_LIVE_BYTES:
        raise KMeansToleranceError('tolerance producer exceeds its global work/scratch/live bound')


def scaled_kmeans_tolerance(features, public_tol, *, cancel=None):
    """Bound pinned mean(var(X,axis=0))*tol from exact CUDA integer moments.

    At most4096 sums and4096 square sums cross to CPU for exact rational bounds.
    NumPy's two-pass floating variance is enclosed, not replaced by the algebraic
    variance. Constant byte columns prove an exactly-zero source variance.
    Zero tolerance bypasses content scans and all CUDA producer allocations.
    """
    if cancel is not None and not callable(cancel):
        raise KMeansToleranceError('cancel must be callable or None')
    _shape(features, torch.uint16)
    versions = (features._version,)
    _checkpoint((features,), versions, cancel)
    rows, width = map(int, features.shape)
    if not 1 <= rows <= MAX_ROWS or not 1 <= width <= MAX_FEATURES or rows*width*2 > MAX_DATASET_BYTES:
        raise KMeansToleranceError('tolerance dataset exceeds its geometry/byte bound')
    try:
        tolerance = _intervals._number(public_tol)
    except ValueError as error:
        raise KMeansToleranceError('public tolerance must be finite and nonnegative') from error
    if tolerance == 0.0:
        return _intervals.F64Interval.point(0.0)
    blocks = (rows+MOMENT_TILE_ROWS-1)//MOMENT_TILE_ROWS
    partial_count = blocks*width
    scratch = 17*partial_count + 17*width + HEADROOM_BYTES
    _budget(3*rows*width+3*partial_count+width, scratch, rows*width*2)
    _checkpoint((features,), versions, cancel)
    with torch.cuda.device(features.device):
        features.record_stream(torch.cuda.current_stream(features.device))
        program = _nvrtc.fixed_kernel_program(_CUDA_SOURCE, features.device)
        _checkpoint((features,), versions, cancel)
        sums = torch.empty((partial_count,), dtype=torch.int64, device=features.device)
        squares = torch.empty_like(sums)
        valid = torch.empty((partial_count,), dtype=torch.uint8, device=features.device)
        totals = torch.empty((2*width,), dtype=torch.int64, device=features.device)
        facts = torch.empty((width,), dtype=torch.uint8, device=features.device)
        program.launch_1d('moment_partials', (features, sums, squares, valid),
            (rows, width, MOMENT_TILE_ROWS, blocks), partial_count)
        _checkpoint((features,), versions, cancel)
        program.launch_1d('finish_moments', (sums, squares, valid, totals, facts), (width, blocks), width)
        _checkpoint((features,), versions, cancel)
        if not all(facts.cpu().tolist()):
            raise KMeansToleranceError('tolerance features must be rounded bytes in [0,255]')
        compact = totals.cpu().tolist()
        numerator = sum(rows*compact[width+j]-compact[j]*compact[j] for j in range(width))
        _checkpoint((features,), versions, cancel)
        if numerator == 0:
            return _intervals.F64Interval.point(0.0)
        exact_variance = Fraction(numerator, rows*rows*width)
        tol = Fraction.from_float(tolerance)
        u, eta = Fraction(1, EXACT_INTEGER_LIMIT), Fraction(1, 2**1074)
        # Mean rounding, subtraction/square error, arbitrary positive variance
        # reductions, final feature mean/division and tolerance multiplication.
        mean_error = 255*u+eta
        deviation_error = mean_error+256*u+eta
        variance_error = deviation_error*511 + _gamma(rows-1)*65536 + 2*u*65536 + 3*eta
        mean_error = variance_error + _gamma(width-1)*(65536+variance_error) + u*(65536+variance_error)+eta
        error = tol*mean_error + u*tol*(65536+mean_error)+eta
        target = exact_variance*tol
        result = _bounds(target-error, target+error)
        _checkpoint((features,), versions, cancel)
        return result


def centre_shift_interval(old_centres, new_centres, *, cancel=None):
    """Bound the source's np.sum((new-old)**2); caller bypasses this at tol0."""
    if cancel is not None and not callable(cancel):
        raise KMeansToleranceError('cancel must be callable or None')
    for value in (old_centres, new_centres):
        _shape(value, torch.float64)
    if old_centres.device != new_centres.device or old_centres.shape != new_centres.shape:
        raise KMeansToleranceError('centre-shift matrices must share shape and CUDA device')
    clusters, width = map(int, old_centres.shape)
    values = old_centres.numel()
    if not 1 <= clusters <= 256 or not 1 <= width <= MAX_FEATURES or values > MAX_CENTRE_VALUES:
        raise KMeansToleranceError('centre-shift geometry exceeds its bound')
    inputs, versions = (old_centres, new_centres), (old_centres._version, new_centres._version)
    _checkpoint(inputs, versions, cancel)
    blocks = (values+SHIFT_TILE_VALUES-1)//SHIFT_TILE_VALUES
    _budget(7*values+blocks, 8*blocks+8+HEADROOM_BYTES, 16*values)
    equal = True
    for start in range(0, values, VALUE_TILE):
        _checkpoint(inputs, versions, cancel)
        a, b = old_centres.reshape(-1)[start:start+VALUE_TILE], new_centres.reshape(-1)[start:start+VALUE_TILE]
        if (not bool(torch.isfinite(a).all().item()) or not bool(torch.isfinite(b).all().item())
                or bool(((a < 0) | (a > 255) | (b < 0) | (b > 255)).any().item())):
            raise KMeansToleranceError('centre values must be finite and in [0,255]')
        equal = bool((a == b).all().item()) and equal
    _checkpoint(inputs, versions, cancel)
    if equal:
        return _intervals.F64Interval.point(0.0)
    with torch.cuda.device(old_centres.device):
        for value in inputs:
            value.record_stream(torch.cuda.current_stream(old_centres.device))
        program = _nvrtc.fixed_kernel_program(_CUDA_SOURCE, old_centres.device)
        _checkpoint(inputs, versions, cancel)
        partials = torch.empty((blocks,), dtype=torch.float64, device=old_centres.device)
        scalar = torch.empty((1,), dtype=torch.float64, device=old_centres.device)
        program.launch_1d('shift_partials', (old_centres, new_centres, partials), (values, SHIFT_TILE_VALUES, blocks), blocks)
        _checkpoint(inputs, versions, cancel)
        program.launch_1d('finish_shift', (partials, scalar), (blocks,), 1)
        _checkpoint(inputs, versions, cancel)
        observed = Fraction.from_float(float(scalar.item()))
        u, eta = Fraction(1, EXACT_INTEGER_LIMIT), Fraction(1, 2**1074)
        error = values*((255*u+eta)*511+65536*u+2*eta)
        gamma = _gamma(values-1)
        lower = max(Fraction(0), (observed-error)/(1+gamma))
        upper = (observed+error)/(1-gamma)
        result = _bounds(lower*(1-gamma)-error, upper*(1+gamma)+error)
        _checkpoint(inputs, versions, cancel)
        return result
