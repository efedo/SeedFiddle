"""Bounded CUDA Gaussian PDFs for PlantCV's sampled-pixel Bayes trainer.

Byte samples, covariance reductions, lookups and ordered density sums stay on
CUDA. Only compact fit statistics cross to CPU. NumPy's pairwise sum structure
and SciPy's Gaussian operation order are attributed in docs/BAYES_KDE_NOTICES.md.
The linked BLAS reduction and host/device transcendental functions are not
claimed byte-exact; the result carries an explicit numerical conformance budget.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch

from seedvision.cuda import _nvrtc
from seedvision.cuda.plantcv_naive_bayes import MAX_CLASSES

PLANTCV_TRAIN_SHA256 = 'f59c95c493d189b6110d47a60a97c86e306ab87583a2f128eff8c389aa6b56eb'
SCIPY_KDE_SHA256 = 'e08717ede16febe4d374f21348036fae6a4739f0de88fa32e9c0e8a097ad6839'
SCIPY_STATS_SHA256 = 'de1368dab08c654cce5a321fb47a590b9dc415336a61defd97e4c34be484f7dc'
NUMPY_PAIRWISE_SHA256 = 'ed85d5b99f0b457cb7a316bf8c4acb884034bbbb81a89c5cedb62be80bf7c4f8'
NUMPY_REDUCTION_SHA256 = '21dedc0e398bbc87ae422ad3465f7d619919d8bdb387b91e70a88a5cac1bf84a'
NUMPY_REDUCTION_LOOP_SHA256 = 'bf5476975362c8bd86a62a8ad4dd3c646d3fbb069a2bae7ded61b295fcfe1a96'
NUMPY_ARITHMETIC_SHA256 = '3703b91ea1add9427e303f96c9fa9aa91842bb867a45bc5a3e868f3667145b11'
NUMPY_ITERATOR_SHA256 = '79425fe80f245fa2106da34458655c94befd7af707d3945df1c452fe98125c9e'
NUMPY_TYPES_SHA256 = 'aa79c2eb417ea1e1b390ceb9bd5f1570cb1384b60a7035a484b40139f2778d99'
# This is the pinned reference's default reduction buffer, independent of the
# installed NumPy version or its process/thread-local setbufsize setting.
NUMPY_REDUCTION_BUFFER = 8192
NUMERICAL_PROFILE = ('ordered-f64; numpy-2.2.6-buffer8192-v1; '
    'reduction-derived plus empirical transcendental allowance')
CONSTANT_NUMERICAL_PROFILE = ('ordered-f64; numpy-2.2.6-buffer8192-constant-v1; '
    'positive-sum covariance plus empirical normalization allowance')
MAX_SAMPLES = 1_000_000
MAX_WORK = 1_000_000_000
MAX_SCRATCH_BYTES = 16 * 1024 * 1024
MAX_CUDA_LIVE_BYTES = 128 * 1024 * 1024
SAMPLE_TILE = 4096
HEADROOM_BYTES = 2 * 1024 * 1024

_CUDA_SOURCE = '#define NUMPY_REDUCTION_BUFFER '+str(NUMPY_REDUCTION_BUFFER)+'\n'+r'''
// NumPy 2.2.6 DOUBLE_pairwise_sum, with virtual weighted byte/constant inputs.
__device__ double term(const unsigned char* x, int row, int channel,
                       double w, int kind) {
    if (kind == 0) return w;
    if (kind == 1) return w*w;
    return (double)x[(long long)row*3+channel]*w;
}
__device__ double leaf(const unsigned char* x, int start, int n,
                       int channel, double w, int kind) {
    if (n < 8) {
        double r=-0.0;
        for (int i=0; i<n; ++i) r=r+term(x,start+i,channel,w,kind);
        return r;
    }
    double r[8];
    for (int j=0; j<8; ++j) r[j]=term(x,start+j,channel,w,kind);
    int i=8;
    for (; i<n-n%8; i+=8)
        for (int j=0; j<8; ++j) r[j]=r[j]+term(x,start+i+j,channel,w,kind);
    double out=((r[0]+r[1])+(r[2]+r[3]))+((r[4]+r[5])+(r[6]+r[7]));
    for (; i<n; ++i) out=out+term(x,start+i,channel,w,kind);
    return out;
}
__device__ double pair_sum(const unsigned char* x, int start, int n, int channel,
                           double w, int kind) {
    // Explicit bounded DFS stack retains the recursive left+right tree.
    int starts[32], lengths[32], states[32]; double left[32];
    int d=0; starts[0]=start; lengths[0]=n; states[0]=0;
    while (true) {
        if (lengths[d]>128 && states[d]==0) {
            int half=lengths[d]/2; half-=half%8;
            states[d]=1;
            starts[d+1]=starts[d]; lengths[d+1]=half; states[d+1]=0; ++d;
            continue;
        }
        double value=leaf(x,starts[d],lengths[d],channel,w,kind);
        while (true) {
            if (d==0) return value;
            --d;
            if (states[d]==1) {
                left[d]=value; states[d]=2;
                int half=lengths[d]/2; half-=half%8;
                starts[d+1]=starts[d]+half;
                lengths[d+1]=lengths[d]-half; states[d+1]=0; ++d;
                break;
            }
            value=left[d]+value;
        }
    }
}
__device__ double buffered_sum(const unsigned char* x, int n, int channel,
                               double w, int kind) {
    // NumPy's reduction iterator calls the inner pairwise loop per buffer;
    // loops_arithm_fp then adds each result to the existing scalar output.
    double total=0.0;
    for (int start=0; start<n; start+=NUMPY_REDUCTION_BUFFER) {
        int rows=n-start;
        if (rows>NUMPY_REDUCTION_BUFFER) rows=NUMPY_REDUCTION_BUFFER;
        total=total+pair_sum(x,start,rows,channel,w,kind);
    }
    return total;
}
extern "C" __global__ void initial_stats(const unsigned char* x, double* stats, int n) {
    int c=blockIdx.x*blockDim.x+threadIdx.x;
    if (c>=3) return;
    double w=1.0/(double)n;
    double sumw=buffered_sum(x,n,c,w,0), sumw2=buffered_sum(x,n,c,w,1);
    double avg=buffered_sum(x,n,c,w,2)/sumw;
    double* s=stats+c*8;
    s[0]=w; s[1]=sumw; s[2]=sumw2; s[3]=avg;
    s[4]=sumw-sumw2/sumw; s[5]=0.0; s[6]=255.0; s[7]=0.0;
}
extern "C" __global__ void covariance_rows(const unsigned char* x, double* stats,
                                            int start, int rows) {
    int c=blockIdx.x*blockDim.x+threadIdx.x;
    if (c>=3) return;
    double* s=stats+c*8;
    double total=s[5], lo=s[6], hi=s[7];
    for (int i=start; i<start+rows; ++i) {
        double value=(double)x[(long long)i*3+c];
        double residual=value-s[3];
        // np.dot(centered, (centered*aweights).T), without reassociation/FMA.
        total=total+residual*(residual*s[0]);
        if (value<lo) lo=value;
        if (value>hi) hi=value;
    }
    s[5]=total; s[6]=lo; s[7]=hi;
}
extern "C" __global__ void gaussian_lookup(const double* params, double* lookup) {
    int q=blockIdx.x*blockDim.x+threadIdx.x;
    if (q>=3*256*256) return;
    int c=q/(256*256), value=(q/256)%256, query=q%256;
    // The pinned 1x1 BLAS triangular solve multiplies by the reciprocal.
    double a=(double)value*params[c*3], b=(double)query*params[c*3];
    double residual=a-b;
    double arg=0.0; arg=arg+residual*residual;
    lookup[q]=exp(-arg/2.0)*params[c*3+1];
}
extern "C" __global__ void density_rows(const unsigned char* x, const double* lookup,
    const double* params, double* pdf, int start, int rows) {
    int q=blockIdx.x*blockDim.x+threadIdx.x;
    if (q>=3*256) return;
    int c=q/256, query=q%256;
    double total=pdf[q], weight=params[c*3+2];
    for (int i=start; i<start+rows; ++i) {
        int value=(int)x[(long long)i*3+c];
        double kernel=lookup[(c*256+value)*256+query];
        total=total+weight*kernel;
    }
    pdf[q]=total;
}
'''


class BayesTrainingError(ValueError):
    """Unsupported samples, numeric domain or resource budget."""


class BayesTrainingCancelled(RuntimeError):
    """Training cancelled before a complete model was published."""


@dataclass(frozen=True, slots=True)
class BayesTrainingPlan:
    classes: int
    sample_counts: tuple[int, ...]
    samples: int
    work: int
    scratch_bytes: int
    live_bytes: int


@dataclass(frozen=True, slots=True)
class BayesChannelFit:
    samples: int
    weight: float
    mean: float
    covariance: float
    effective_samples: float
    scott_factor: float
    bandwidth: float
    covariance_absolute_budget: float
    pdf_absolute_budget: float
    numerical_profile: str = NUMERICAL_PROFILE


@dataclass(frozen=True, slots=True)
class BayesConstantChannelFit:
    """Positive roundoff covariance from a source-constant byte column."""
    samples: int
    weight: float
    mean: float
    covariance: float
    effective_samples: float
    scott_factor: float
    bandwidth: float
    covariance_absolute_budget: float
    pdf_absolute_budget: float
    constant_value: int
    numerical_profile: str = CONSTANT_NUMERICAL_PROFILE

    def __post_init__(self):
        if (self.numerical_profile != CONSTANT_NUMERICAL_PROFILE
                or type(self.constant_value) is not int or not 0 <= self.constant_value <= 255):
            raise BayesTrainingError('constant-channel fit requires its fixed profile and byte value')


@dataclass(frozen=True, slots=True)
class BayesTrainingFit:
    class_names: tuple[str, ...]
    probabilities: torch.Tensor
    channels: tuple[tuple[BayesChannelFit | BayesConstantChannelFit, ...], ...]
    plan: BayesTrainingPlan


def _tensor_signature(x):
    storage = x.untyped_storage()
    return (x._version, x.data_ptr(), storage.data_ptr(), storage.nbytes(),
        tuple(x.shape), tuple(x.stride()), x.storage_offset(), x.dtype, x.device, x.layout)


def _checkpoint(values, versions, cancel):
    def unchanged():
        if tuple(_tensor_signature(x) for x in values) != versions:
            raise BayesTrainingError('Bayes training samples changed during calculation or callback')
    unchanged()
    try:
        answer = cancel() if cancel is not None else False
    finally:
        unchanged()
    if type(answer) is not bool:
        raise BayesTrainingError('cancel callback must return bool')
    if answer:
        raise BayesTrainingCancelled('Bayes training was cancelled before a complete result')


def _preflight(samples, class_names):
    if (type(samples) is not tuple or type(class_names) is not tuple
            or not 1 <= len(samples) <= MAX_CLASSES or len(class_names) != len(samples)
            or any(type(name) is not str or not name or '\t' in name or '\n' in name
                   or '\r' in name for name in class_names)
            or len(set(class_names)) != len(class_names)):
        raise BayesTrainingError('require ordered unique nonempty model class names and sample tuples')
    if not all(torch.is_tensor(x) and x.device.type == 'cuda' and x.layout == torch.strided
               and x.dtype == torch.uint8 and x.ndim == 2 and x.shape[1] == 3
               and x.is_contiguous() and x.device == samples[0].device for x in samples):
        raise BayesTrainingError('require same-device contiguous CUDA uint8 Nx3 HSV samples')
    counts = tuple(int(x.shape[0]) for x in samples)
    total = sum(counts)
    if any(n < 2 for n in counts) or total > MAX_SAMPLES:
        raise BayesTrainingError('each class needs at least two samples within the global sample cap')
    # Three ordered covariance scans, three pairwise reductions per channel,
    # three 256-query scans, and the lookup/finite scans. No Nx256 allocation.
    work = total*(3*3+3+3*256) + len(samples)*(3*256*256+3*256)
    scratch = 3*256*256*8 + len(samples)*3*256*8 + 512 + HEADROOM_BYTES
    # Charge complete backing storage, including tensors that alias larger inputs.
    storages = {(x.device, x.untyped_storage().data_ptr()): x.untyped_storage().nbytes()
                for x in samples}
    live = sum(storages.values()) + scratch
    if (work > MAX_WORK or scratch > MAX_SCRATCH_BYTES or live > MAX_CUDA_LIVE_BYTES):
        raise BayesTrainingError('Bayes training exceeds the work/scratch/live CUDA budget')
    return BayesTrainingPlan(len(samples), counts, total, work, scratch, live)


def _fit_parameters(statistics, n, class_name):
    channels, params = [], []
    u = 2.0**-53
    gamma = (n+32)*u/(1.0-(n+32)*u)
    # Each weighted sum uses a bounded pairwise chunk followed by sequential
    # outer additions. gamma_(d+m-1) conservatively covers composition of the
    # chunk's gamma_d and the outer gamma_(m-1), without subtractive cancellation.
    # The factor four below also covers the rounded product and mean division.
    chunks = (n+NUMPY_REDUCTION_BUFFER-1)//NUMPY_REDUCTION_BUFFER
    depth = (32 + max(0, math.ceil(math.log2(max(1, min(n, NUMPY_REDUCTION_BUFFER)/128))))
        + chunks-1)
    mean_gamma = depth*u/(1.0-depth*u)
    for channel, (weight, sumw, sumw2, mean, fact, total, lo, hi) in zip(
            ('hue', 'saturation', 'value'), statistics):
        context = f'class {class_name!r}, channel {channel}: '
        constant = hi == lo
        if constant and mean == lo:
            raise BayesTrainingError(context+'constant HSV channel has singular source covariance')
        covariance = total*(1.0/fact)
        neff = 1.0/sumw2
        factor = float(np.power(neff, -1.0/5.0))
        bandwidth = math.sqrt(covariance)*factor
        if not all(math.isfinite(x) and x > 0 for x in (fact, covariance, neff, factor, bandwidth)):
            raise BayesTrainingError(context+'Gaussian KDE requires finite positive covariance and bandwidth')
        # Conservative positive-sum/dot error plus possible weighted-mean error.
        # This component bounds reduction differences; pow/exp/libm compatibility
        # remains empirical and is identified separately in the numerical profile.
        mean_error = 4*255*mean_gamma
        covariance_budget = 4*gamma*covariance
        if not constant:
            covariance_budget += sumw*(510*mean_error+mean_error*mean_error)/fact
        if covariance_budget >= covariance:
            raise BayesTrainingError(context+'source covariance uncertainty reaches zero; KDE is unsupported')
        relative_bandwidth = -0.5*math.log1p(-covariance_budget/covariance) + 64*u
        norm = math.pow(2.0*math.pi, -0.5)/bandwidth
        max_norm = norm*math.exp(relative_bandwidth)
        # |d (exp(-t²/2)/h) / d log(h)| <= 1/h. Roundoff in separate
        # whitening is included. The final 64u allowance for host/device exp,
        # pow and normalization is empirical, not a vendor-independent theorem.
        if constant:
            # The fixed source/native mean is identical. Nonzero residual terms
            # residual*(residual*w) are normal and positive for byte inputs and
            # n<=1M. Separate source/native positive sums and covariance scaling
            # differ by less than 4*gamma; no generic mean uncertainty is needed.
            # At query==byte, both separate whitenings are identical. At every
            # other byte, this guarded upper covariance gives an exponent below
            # -2048 even after whitening error, far past binary64 underflow.
            if covariance*(1.0+4*gamma) >= 2.0**-12 or factor > 1.0:
                raise BayesTrainingError(context+'constant-channel off-peak underflow domain is unsupported')
            max_bandwidth = bandwidth*math.exp(relative_bandwidth)
            if (1.0-1024*u)/max_bandwidth <= 64.0:
                raise BayesTrainingError(context+'constant-channel whitening interval is unsupported')
            # A multiplicative peak interval covers covariance normalization and
            # both ordered positive density sums. The two 64u contributions are
            # explicitly empirical allowances for the pinned normalization path.
            log_ratio = relative_bandwidth + 4*gamma + 64*u
            pdf_budget = (max_norm*sumw*(1.0+4*gamma)*math.expm1(log_ratio)
                + 256*math.ulp(0.0))
            channels.append(BayesConstantChannelFit(n, weight, mean, covariance,
                neff, factor, bandwidth, covariance_budget, pdf_budget, constant_value=int(lo)))
        else:
            pdf_budget = max_norm*(relative_bandwidth + 4*gamma
                + 16*u*255/bandwidth + 64*u)*sumw + 256*math.ulp(0.0)
            channels.append(BayesChannelFit(n, weight, mean, covariance, neff, factor,
                bandwidth, covariance_budget, pdf_budget))
        params.append((1.0/bandwidth, norm, weight))
    return tuple(channels), params


def fit_naive_bayes_hsv(samples: tuple[torch.Tensor, ...], class_names: tuple[str, ...],
                        *, cancel=None) -> BayesTrainingFit:
    """Fit each class/channel and evaluate its Gaussian PDF at bytes 0..255.

    Class and original sample order are retained. One class is supported, as in
    the public implementation. No regularization or histogram substitution is
    applied. Constant columns preserve source singular failures; positive source
    roundoff covariance uses its own guarded numerical profile.
    """
    if cancel is not None and not callable(cancel):
        raise BayesTrainingError('cancel must be callable or None')
    plan = _preflight(samples, class_names)
    versions = tuple(_tensor_signature(x) for x in samples)
    _checkpoint(samples, versions, cancel)
    device = samples[0].device
    with torch.cuda.device(device):
        for x in samples:
            x.record_stream(torch.cuda.current_stream(device))
        program = _nvrtc.fixed_kernel_program(_CUDA_SOURCE, device)
        _checkpoint(samples, versions, cancel)
        output = torch.zeros((len(samples), 3, 256), dtype=torch.float64, device=device)
        # Raw CUDA writes do not advance PyTorch's version counter. Keep the
        # output's signature fixed across those trusted writes and guard it at
        # every callback, so a changed partial PDF cannot become the final model.
        watched = (*samples, output)
        watched_versions = (*versions, _tensor_signature(output))
        stats = torch.empty((3, 8), dtype=torch.float64, device=device)
        lookup = torch.empty((3, 256, 256), dtype=torch.float64, device=device)
        fits = []
        for index, (x, n) in enumerate(zip(samples, plan.sample_counts)):
            _checkpoint(watched, watched_versions, cancel)
            program.launch_1d('initial_stats', (x, stats), (n,), 3)
            for start in range(0, n, SAMPLE_TILE):
                _checkpoint(watched, watched_versions, cancel)
                program.launch_1d('covariance_rows', (x, stats), (start, min(SAMPLE_TILE, n-start)), 3)
            # Exactly 24 compact scalars, never a dense sample transfer.
            channels, params = _fit_parameters(stats.cpu().tolist(), n, class_names[index])
            fits.append(channels)
            parameters = torch.tensor(params, dtype=torch.float64, device=device)
            program.launch_1d('gaussian_lookup', (parameters, lookup), (), 3*256*256)
            for start in range(0, n, SAMPLE_TILE):
                _checkpoint(watched, watched_versions, cancel)
                program.launch_1d('density_rows', (x, lookup, parameters, output[index]),
                    (start, min(SAMPLE_TILE, n-start)), 3*256)
            if not bool(torch.isfinite(output[index]).all().item()):
                raise BayesTrainingError('Gaussian KDE produced a nonfinite PDF')
        _checkpoint(watched, watched_versions, cancel)
        return BayesTrainingFit(class_names, output, tuple(fits), plan)
