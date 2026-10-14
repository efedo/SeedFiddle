"""Export compact CUDA fit centres through the existing numeric model format."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from seedvision.cuda import plantcv_kmeans_fit as _fit
from seedvision.cuda import plantcv_kmeans_model as _model

MAX_CPU_METADATA_BYTES = 128 * 1024 * 1024
CPU_BYTES_PER_CENTRE = 64
CPU_HEADROOM_BYTES = 2 * 1024 * 1024


class KMeansExportError(ValueError):
    """Invalid fitted-centre interpretation or export metadata bound."""


class KMeansExportCancelled(RuntimeError):
    """Export cancelled without returning model metadata or bytes."""


@dataclass(frozen=True, slots=True)
class KMeansFitExport:
    model: _model.KMeansModel
    payload: bytes


def _copy_centres_to_cpu(centres):
    """The sole, essential compact device transfer in this module."""
    return centres.detach().to(device='cpu', dtype=torch.float64, copy=True)


def export_kmeans_fit(result, patch_size, channels, *, cancel=None):
    """Encode ordered centres; leave features/counts/labels/distances on CUDA.

    Only centre interpretation is validated here. This does not certify scientific
    model quality or reconstruct unrelated training-result fields.
    """
    if type(result) is not _fit.KMeansFitResult or type(result.plan) is not _fit.KMeansFitPlan:
        raise KMeansExportError('expected a complete fit result and its plan')
    if cancel is not None and not callable(cancel):
        raise KMeansExportError('cancel must be callable or None')
    centres = result.centres
    if (not torch.is_tensor(centres) or centres.device.type != 'cuda'
            or centres.layout != torch.strided or centres.dtype != torch.float64
            or centres.ndim != 2 or not centres.is_contiguous()):
        raise KMeansExportError('fit centres must be contiguous strided CUDA float64 rows')
    version = centres._version

    def guard():
        if centres._version != version:
            raise KMeansExportError('fit centres changed during export')
        cancelled = cancel() if cancel is not None else False
        if centres._version != version:
            raise KMeansExportError('fit centres changed during callback')
        if type(cancelled) is not bool:
            raise KMeansExportError('cancel callback must return bool')
        if cancelled:
            raise KMeansExportCancelled('K-means export was cancelled')

    guard()
    if type(patch_size) is not int or not 1 <= patch_size <= 31 or type(channels) is not int or channels not in (1, 3):
        raise KMeansExportError('model geometry requires patch_size1..31 and channels1or3')
    clusters, width = map(int, centres.shape)
    if (not 1 <= clusters <= _model.MAX_CLUSTERS or width != patch_size*patch_size*channels
            or width > _model.MAX_FEATURES or clusters*width > _model.MAX_CENTRE_VALUES
            or type(result.plan.clusters) is not int or result.plan.clusters != clusters
            or type(result.plan.features) is not int or result.plan.features != width):
        raise KMeansExportError('fit centre shape and plan do not match model geometry')
    # At most one row list coexists with the tuple table. Encoding uses the
    # existing bounded incremental encoder; release the first table before parse.
    # 64/value covers CPU f64, float/tuple storage and bounded row temporaries;
    # four model-byte caps cover encoder/byte/string temporaries.
    cpu_bound = CPU_BYTES_PER_CENTRE*clusters*width+4*_model.MAX_MODEL_BYTES+CPU_HEADROOM_BYTES
    if cpu_bound > MAX_CPU_METADATA_BYTES:
        raise KMeansExportError('centre export exceeds its CPU metadata bound')
    guard()
    cpu = _copy_centres_to_cpu(centres)
    guard()
    if (not torch.is_tensor(cpu) or cpu.device.type != 'cpu' or cpu.dtype != torch.float64
            or cpu.layout != torch.strided or tuple(cpu.shape) != (clusters, width)
            or not cpu.is_contiguous()):
        raise KMeansExportError('compact centre transfer returned invalid data')
    rows = []
    for row in cpu:
        guard()
        rows.append(tuple(row.tolist()))
    table = tuple(rows)
    del rows, row, cpu
    guard()
    try:
        digest = _model._centres_digest(patch_size, channels, table)
        # This temporary value is never published. Canonical parsing below
        # establishes the actual raw-byte identity of the exported artifact.
        model = _model.KMeansModel(patch_size, channels, table, '0'*64, 1, digest)
        _model.validate_kmeans_model(model)
        guard()
        payload = _model.encode_kmeans_model(model)
        del model, table
        guard()
        parsed = _model.parse_kmeans_model(payload)
        if parsed.centres_sha256 != digest:
            raise KMeansExportError('canonical export changed the numeric table identity')
    except _model.KMeansModelError as error:
        raise KMeansExportError(str(error)) from error
    guard()
    return KMeansFitExport(parsed, payload)
