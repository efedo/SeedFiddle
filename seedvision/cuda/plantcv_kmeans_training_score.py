"""Bounded mutable-centre CUDA labels and certified pinned-source inertia.

The source Lloyd score is centre_norm - 2*dot, with no feature norm or clamp.
Only compact centre validity flags and the final scalar fold cross to CPU.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from seedvision.cuda import _nvrtc
from seedvision.cuda.plantcv_kmeans_classifier import _decision_margin_bound
from seedvision.cuda.plantcv_kmeans_intervals import F64Interval, source_inertia_interval, source_reduction_profile

MAX_ROWS = 1_000_000
MAX_FEATURES = 4096
MAX_CLUSTERS = 256
MAX_DATASET_BYTES = 128 * 1024 * 1024
MAX_WORK = 1_000_000_000
MAX_SCRATCH_BYTES = 64 * 1024 * 1024
MAX_CUDA_LIVE_BYTES = 512 * 1024 * 1024
MAX_TILE_ROWS = 4096
VALUE_TILE = 65_536
ALLOCATION_HEADROOM_BYTES = 2 * 1024 * 1024
SKLEARN_VERSION = '1.9.1'
SKLEARN_KMEANS_SHA256 = '0cf335b8b9b4abe67259ccdcbaf07f02bd7feee31e63a06a1cce32a87fcd375d'
SKLEARN_COMMON_SHA256 = '9a05c0184b411fd1d68c9033e265d4b41824f01f1a66d6c494cfb11f1780b489'
SKLEARN_LLOYD_SHA256 = '66ba7cad80c8000e350c0dd4014e790b424e7c0157b479c98fc7610071aa2a92'
_CUDA_SOURCE = r'''
extern "C" __global__ void centre_facts(
    const double* centres, double* norms, unsigned char* facts, int f, int k) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= k) return;
    double norm = 0.0;
    int valid = 1, integral = 1;
    for (int j=0; j<f; ++j) {
        double value = centres[(long long)c*f+j];
        if (!isfinite(value) || value<0.0 || value>255.0) valid=0;
        if (value != floor(value)) integral=0;
        norm=__dadd_rn(norm,__dmul_rn(value,value));
    }
    norms[c]=norm;
    facts[c]=(unsigned char)((valid ? 1 : 0) | (integral ? 2 : 0));
}
extern "C" __global__ void duplicate_flags(
    const double* centres, unsigned char* duplicates, int f, int k) {
    int later = blockIdx.x * blockDim.x + threadIdx.x;
    if (later >= k) return;
    int duplicate=0;
    for (int earlier=0; earlier<later; ++earlier) {
        int equal=1;
        for (int j=0; j<f; ++j) {
            if (centres[(long long)earlier*f+j] != centres[(long long)later*f+j]) {
                equal=0; break;
            }
        }
        if (equal) { duplicate=1; break; }
    }
    duplicates[later]=(unsigned char)duplicate;
}
extern "C" __global__ void score_rows(
    const unsigned short* x, const double* centres, const double* norms,
    double* scores, int start, int rows, int f, int k) {
    int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= rows*k) return;
    int r=q/k, c=q%k;
    double dot=0.0;
    for (int j=0; j<f; ++j) {
        dot=__dadd_rn(dot,__dmul_rn((double)x[(long long)(start+r)*f+j],centres[(long long)c*f+j]));
    }
    scores[q]=__dsub_rn(norms[c],__dmul_rn(2.0,dot));
}
extern "C" __global__ void assigned_distances(
    const unsigned short* x, const double* centres, const unsigned char* labels,
    double* distances, int start, int rows, int f) {
    int q=blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= rows) return;
    int row=start+q, c=(int)labels[row], j=0;
    double total=0.0;
    for (; j+3<f; j+=4) {
        double d0=__dsub_rn((double)x[(long long)row*f+j],centres[(long long)c*f+j]);
        double d1=__dsub_rn((double)x[(long long)row*f+j+1],centres[(long long)c*f+j+1]);
        double d2=__dsub_rn((double)x[(long long)row*f+j+2],centres[(long long)c*f+j+2]);
        double d3=__dsub_rn((double)x[(long long)row*f+j+3],centres[(long long)c*f+j+3]);
        double group=__dadd_rn(__dadd_rn(__dadd_rn(__dmul_rn(d0,d0),__dmul_rn(d1,d1)),__dmul_rn(d2,d2)),__dmul_rn(d3,d3));
        total=__dadd_rn(total,group);
    }
    for (; j<f; ++j) {
        double d=__dsub_rn((double)x[(long long)row*f+j],centres[(long long)c*f+j]);
        total=__dadd_rn(total,__dmul_rn(d,d));
    }
    distances[row]=total;
}
extern "C" __global__ void serial_inertia(
    const double* distances, double* output, int rows) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    double total=0.0;
    for (int row=0; row<rows; ++row) total=__dadd_rn(total,distances[row]);
    output[0]=total;
}
'''


class KMeansTrainingScoreError(ValueError):
    """Invalid byte-domain tensors or whole-operation bounds."""


class KMeansTrainingScoreCancelled(RuntimeError):
    """No complete result was published after cancellation."""


class KMeansTrainingScorePrecisionError(KMeansTrainingScoreError):
    """A fractional nearest-centre decision cannot be certified."""


@dataclass(frozen=True, slots=True)
class KMeansTrainingScorePlan:
    rows: int
    features: int
    clusters: int
    tile_rows: int
    work: int
    workspace_bytes: int
    peak_live_bytes: int


@dataclass(frozen=True, slots=True)
class KMeansTrainingScore:
    labels: torch.Tensor
    row_distances: torch.Tensor
    inertia: F64Interval
    source_n_threads: int | None

    @property
    def source_reduction_profile(self):
        return source_reduction_profile(self.source_n_threads)


def _checkpoint(features, centres, versions, cancel):
    if (features._version, centres._version) != versions:
        raise KMeansTrainingScoreError('scoring inputs changed during execution')
    cancelled = cancel() if cancel is not None else False
    if (features._version, centres._version) != versions:
        raise KMeansTrainingScoreError('scoring inputs changed during callback')
    if type(cancelled) is not bool:
        raise KMeansTrainingScoreError('cancel callback must return bool')
    if cancelled:
        raise KMeansTrainingScoreCancelled('K-means training scoring was cancelled')


def preflight_kmeans_training_score(features, centres, *, source_n_threads, tile_rows=4096):
    if (not torch.is_tensor(features) or not torch.is_tensor(centres)
            or features.device.type != 'cuda' or centres.device != features.device
            or features.layout != torch.strided or centres.layout != torch.strided
            or features.dtype != torch.uint16 or centres.dtype != torch.float64
            or features.ndim != 2 or centres.ndim != 2
            or not features.is_contiguous() or not centres.is_contiguous()):
        raise KMeansTrainingScoreError('expected contiguous CUDA uint16 rows and float64 centres on one device')
    rows, width = map(int, features.shape)
    clusters, centre_width = map(int, centres.shape)
    if (not 1 <= rows <= MAX_ROWS or not 1 <= width <= MAX_FEATURES
            or not 1 <= clusters <= MAX_CLUSTERS or centre_width != width
            or (source_n_threads is not None and (type(source_n_threads) is not int or source_n_threads < 1))):
        raise KMeansTrainingScoreError('scoring geometry/thread provenance exceeds its domain')
    if type(tile_rows) is not int or not 1 <= tile_rows <= MAX_TILE_ROWS:
        raise KMeansTrainingScoreError('tile_rows must be an integer in [1,4096]')
    input_bytes = rows*width*2 + clusters*width*8
    if rows*width*2 > MAX_DATASET_BYTES:
        raise KMeansTrainingScoreError('feature dataset exceeds its byte cap')
    # Include validation, norm/duplicate checks, score and assigned-distance
    # visits, serial fold, and all label/margin reduction passes together.
    work = 2*rows*width + clusters*width + clusters*(clusters-1)//2*width + rows*clusters*(width+8) + rows
    if work > MAX_WORK:
        raise KMeansTrainingScoreError('scoring work exceeds its global bound')
    selected = min(rows, tile_rows)
    scratch = ALLOCATION_HEADROOM_BYTES + selected*(8*clusters+128) + 11*clusters
    if scratch > MAX_SCRATCH_BYTES:
        raise KMeansTrainingScoreError('scoring scratch exceeds its byte cap')
    workspace = scratch + 9*rows + 8
    peak = input_bytes + workspace
    if peak > MAX_CUDA_LIVE_BYTES:
        raise KMeansTrainingScoreError('scoring exceeds its global live-storage bound')
    return KMeansTrainingScorePlan(rows, width, clusters, selected, work, workspace, peak)


def score_kmeans_training_rows(features, centres, *, source_n_threads, cancel=None, tile_rows=4096):
    """Return complete CUDA labels/losses and the certified source inertia.

    Identical centres preserve their earliest source index. Fractional winners
    need a gap against every distinct competitor; integral ties remain exact.
    Input versions are captured before callbacks and checked around every raw
    launch. Norms and duplicate flags are reused across all row tiles in this
    call, then recomputed for the next mutable centre version.
    """
    if cancel is not None and not callable(cancel):
        raise KMeansTrainingScoreError('cancel must be callable or None')
    if not torch.is_tensor(features) or not torch.is_tensor(centres):
        raise KMeansTrainingScoreError('scoring inputs must be tensors')
    versions = (features._version, centres._version)
    _checkpoint(features, centres, versions, cancel)
    plan = preflight_kmeans_training_score(features, centres, source_n_threads=source_n_threads, tile_rows=tile_rows)
    signed = features.view(torch.int16).reshape(-1)
    for start in range(0, signed.numel(), VALUE_TILE):
        _checkpoint(features, centres, versions, cancel)
        tile = signed[start:start+VALUE_TILE]
        if bool(((tile < 0) | (tile > 255)).any().item()):
            raise KMeansTrainingScoreError('feature values must be rounded bytes in [0,255]')
    _checkpoint(features, centres, versions, cancel)
    with torch.cuda.device(features.device):
        stream = torch.cuda.current_stream(features.device)
        features.record_stream(stream)
        centres.record_stream(stream)
        program = _nvrtc.fixed_kernel_program(_CUDA_SOURCE, features.device)
        _checkpoint(features, centres, versions, cancel)
        norms = torch.empty((plan.clusters,), dtype=torch.float64, device=features.device)
        facts = torch.empty((plan.clusters,), dtype=torch.uint8, device=features.device)
        program.launch_1d('centre_facts', (centres, norms, facts), (plan.features, plan.clusters), plan.clusters)
        _checkpoint(features, centres, versions, cancel)
        compact_facts = facts.cpu().tolist()
        if any((value & 1) == 0 for value in compact_facts):
            raise KMeansTrainingScoreError('centres must be finite and in [0,255]')
        integral = all((value & 2) != 0 for value in compact_facts)
        duplicates = torch.empty_like(facts)
        program.launch_1d('duplicate_flags', (centres, duplicates), (plan.features, plan.clusters), plan.clusters)
        _checkpoint(features, centres, versions, cancel)
        duplicate_mask = duplicates.to(torch.bool)
        labels = torch.empty((plan.rows,), dtype=torch.uint8, device=features.device)
        row_distances = torch.empty((plan.rows,), dtype=torch.float64, device=features.device)
        scores = torch.empty((plan.tile_rows, plan.clusters), dtype=torch.float64, device=features.device)
        margin_bound = _decision_margin_bound(plan.features)
        for start in range(0, plan.rows, plan.tile_rows):
            _checkpoint(features, centres, versions, cancel)
            rows = min(plan.tile_rows, plan.rows-start)
            score_tile = scores[:rows]
            program.launch_1d('score_rows', (features, centres, norms, score_tile),
                (start, rows, plan.features, plan.clusters), rows*plan.clusters)
            _checkpoint(features, centres, versions, cancel)
            score_tile.masked_fill_(duplicate_mask, float('inf'))
            best_values, best_labels = score_tile.min(dim=1)
            labels[start:start+rows].copy_(best_labels)
            if not integral and plan.clusters > 1:
                score_tile.scatter_(1, best_labels[:, None], float('inf'))
                second_values = score_tile.min(dim=1).values
                if bool(((second_values-best_values) <= margin_bound).any().item()):
                    raise KMeansTrainingScorePrecisionError('fractional winner is within the source/GPU arithmetic bound')
                del second_values
            del best_values, best_labels
            _checkpoint(features, centres, versions, cancel)
            program.launch_1d('assigned_distances', (features, centres, labels, row_distances),
                (start, rows, plan.features), rows)
            _checkpoint(features, centres, versions, cancel)
        scalar = torch.empty((1,), dtype=torch.float64, device=features.device)
        program.launch_1d('serial_inertia', (row_distances, scalar), (plan.rows,), 1)
        _checkpoint(features, centres, versions, cancel)
        inertia = source_inertia_interval(float(scalar.item()), row_count=plan.rows,
            feature_count=plan.features, source_n_threads=source_n_threads, integral=integral)
        _checkpoint(features, centres, versions, cancel)
        return KMeansTrainingScore(labels, row_distances, inertia, source_n_threads)
