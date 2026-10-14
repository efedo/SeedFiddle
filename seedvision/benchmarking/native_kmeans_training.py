"""Native directory training, reusable CUDA rows and compact model export."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json

from PySide6.QtCore import qVersion

import torch

from seedvision.benchmarking import kmeans_training_directory as _directory
from seedvision.benchmarking import kmeans_model_artifact as _artifact
from seedvision.cuda import plantcv_kmeans_export as _export
from seedvision.cuda import plantcv_kmeans_fit as _fit
from seedvision.cuda import plantcv_kmeans_training as _training


@dataclass(frozen=True, slots=True)
class KMeansTrainingRequest:
    directory: str
    prefix: str = ''
    num_imgs: int = 0
    patch_size: int = 10
    sampling: int | float | None = None
    clusters: int = 2
    seed: int = 1
    n_init: int = 10


@dataclass(frozen=True, slots=True)
class KMeansTrainingCache:
    selection: tuple
    manifest: _directory.KMeansTrainingDirectory
    key: tuple | None
    rows: _training.KMeansTrainingRows | None
    version: int | None

    def without_rows(self):
        """Invalidate extraction while retaining the upstream file selection."""
        return KMeansTrainingCache(self.selection, self.manifest, None, None, None)


@dataclass(frozen=True, slots=True)
class KMeansTrainingSummary:
    rows: int
    steps: int
    iterations: int
    source_reduction_profile: str
    source_n_threads: int | None
    directory_sha256: str


@dataclass(frozen=True, slots=True)
class KMeansTrainingCompletion:
    cache: KMeansTrainingCache
    export: _export.KMeansFitExport
    summary: KMeansTrainingSummary
    artifact_payload: bytes


FIT_DEFAULTS = dict(max_iter=100, batch_size=1024, init_size=None, tol=0.0,
    max_no_improvement=10, reassignment_ratio=.01, compute_labels=True)


def _training_provenance(request, manifest, row_identity, fitted, model, device):
    """Project compact facts explicitly; never traverse a CUDA result/cache."""
    from seedvision.benchmarking.native_recipe import _module_record
    modules = (_fit, _fit._control, _fit._intervals, _fit._update, _fit._plusplus,
        _fit._random, _fit._tolerance, _fit._score, _export, _artifact,
        __import__(__name__, fromlist=['train_kmeans_directory']))
    # Stable persisted labels survive moving the shared codec into model_io.
    # The digest still verifies the actual loaded implementation and source.
    records = {('seedvision.benchmarking.kmeans_model_artifact' if module is _artifact
        else module.__name__): sha256(json.dumps(_module_record(module),
        sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()
        for module in modules}
    sampling_kind = ('all' if request.sampling is None else
        'count' if type(request.sampling) is int else 'fraction')
    effective_request = asdict(request)
    effective_request.update(directory=manifest.directory, sampling_kind=sampling_kind)
    return dict(kind='native_training', numeric_identity=_artifact.numeric_model_identity(model),
        request=effective_request, manifest=asdict(manifest), patch_seed=_training.PATCH_SEED,
        fit_parameters=dict(FIT_DEFAULTS, init='k-means++', source_n_threads=fitted.source_n_threads,
            seed=request.seed, n_init=request.n_init), fit_plan=asdict(fitted.plan),
        fit_result=dict(n_steps=fitted.n_steps, n_iter=fitted.n_iter,
            inertia=dict(lo=fitted.inertia.lo, hi=fitted.inertia.hi),
            scaled_tolerance=dict(lo=fitted.scaled_tolerance.lo, hi=fitted.scaled_tolerance.hi),
            source_n_threads=fitted.source_n_threads,
            source_reduction_profile=fitted.source_reduction_profile,
            random_identity=dict(fitted.random_identity)),
        calculation=dict(row_sha256=row_identity, fit_modules=records,
            runtime=dict(torch=str(torch.__version__), torch_cuda=str(torch.version.cuda),
                numpy=_directory.np.__version__, opencv=_directory.cv2.__version__, qt=qVersion(),
                device=str(device), device_name=torch.cuda.get_device_name(device))),
        source_pins=dict(plantcv_version='4.11.3', train_sha256=_directory.PLANTCV_TRAIN_SHA256,
            predict_sha256='09019da00e0713c51cff31ffd240f7bf8aa0b4e228f6d88f2bed9c1d40430392',
            sklearn_kmeans_sha256=_fit.SKLEARN_KMEANS_SHA256))


def _row_calculation_identity():
    from seedvision.benchmarking.native_recipe import _module_record
    modules = (_directory, _training, _training._full, _training._sampled,
        _training._full._features, _training._full._preprocess)
    records = [_module_record(module) for module in modules]
    runtime = (str(torch.__version__), _directory.np.__version__, _directory.cv2.__version__, qVersion())
    return sha256(json.dumps((records, runtime), sort_keys=True, separators=(',', ':'),
        allow_nan=False).encode('utf-8')).hexdigest()


def train_kmeans_directory(request, *, cached_rows=None, cancel=None):
    """Train effective parameters; fit edits can reuse verified CUDA patch rows.

    Fit/extraction edits retain the upstream selection and revalidate its bytes.
    Selection edits resolve files again; repeated choices remain repeated rows.
    No source thread count is inferred: numerical
    decisions must hold for the explicit all-reduction compatibility profile.
    """
    if type(request) is not KMeansTrainingRequest:
        raise ValueError('expected a native K-means training request')
    if cancel is not None and not callable(cancel):
        raise ValueError('cancel must be callable or None')
    if cached_rows is not None and type(cached_rows) is not KMeansTrainingCache:
        raise ValueError('expected a native K-means row cache or None')
    if type(request.patch_size) is not int or not 1 <= request.patch_size <= 31:
        raise ValueError('patch size must be an integer in [1,31]')
    if request.sampling is not None and (type(request.sampling) not in (int, float)
            or (type(request.sampling) is int and not 1 <= request.sampling <= _fit.MAX_ROWS)
            or (type(request.sampling) is float and not 0 < request.sampling < 1)):
        raise ValueError('sampling must be None, a positive patch count, or a fraction in (0,1)')
    watched = None
    version = None

    def guard():
        if watched is not None and watched._version != version:
            raise ValueError('cached training rows changed during calculation')
        answer = cancel() if cancel is not None else False
        if watched is not None and watched._version != version:
            raise ValueError('cached training rows changed during callback')
        if type(answer) is not bool:
            raise ValueError('cancel callback must return bool')
        if answer:
            raise _directory.KMeansDirectoryCancelled('K-means training was cancelled')
        return False

    guard()
    selection = (request.directory, request.prefix, request.num_imgs)
    if cached_rows is not None and cached_rows.selection == selection:
        manifest = _directory.validate_kmeans_training_directory(cached_rows.manifest, cancel=guard)
    else:
        manifest = _directory.resolve_kmeans_training_directory(request.directory,
            prefix=request.prefix, num_imgs=request.num_imgs, cancel=guard)
    key = (manifest.manifest_sha256, request.patch_size, type(request.sampling).__name__,
        request.sampling, _training.PATCH_SEED, torch.cuda.current_device(), _row_calculation_identity())
    fit_parameters = dict(FIT_DEFAULTS, source_n_threads=None, seed=request.seed, n_init=request.n_init)
    if cached_rows is not None and cached_rows.key == key and cached_rows.rows is not None:
        rows = cached_rows.rows
        if (type(rows) is not _training.KMeansTrainingRows or not torch.is_tensor(rows.features)
                or type(rows.plan) is not _training.KMeansImageDataPlan
                or tuple(rows.features.shape) != (rows.plan.rows, rows.plan.features)
                or rows.plan.features != request.patch_size**2*rows.plan.channels
                or type(cached_rows.version) is not int):
            raise ValueError('cached training rows have inconsistent metadata')
        watched, version = rows.features, cached_rows.version
        guard()
        fitted = _fit.fit_kmeans_training_rows(rows.features, request.clusters,
            cancel=guard, **fit_parameters)
    else:
        materialized = _directory.materialize_kmeans_training_directory(manifest, cancel=guard)
        plan = _training.preflight_kmeans_image_fit(materialized.images, request.clusters,
            patch_size=request.patch_size, sampling=request.sampling, cancel=guard, **fit_parameters)
        # A caller may still own old rows during a changed extraction. Include
        # that storage in this operation's peak rather than assuming it vanished.
        retained = (cached_rows.rows.features.untyped_storage().nbytes()
            if cached_rows is not None and cached_rows.rows is not None else 0)
        if plan.peak_live_bytes+retained > _training.MAX_CUDA_LIVE_BYTES:
            raise ValueError('retained row cache exceeds the combined CUDA live bound')
        trained = _training.fit_kmeans_training_images(materialized.images, request.clusters,
            patch_size=request.patch_size, sampling=request.sampling, cancel=guard, **fit_parameters)
        rows, fitted = trained.training_rows, trained.fit
        watched, version = rows.features, rows.features._version
        del trained, materialized
    guard()
    exported = _export.export_kmeans_fit(fitted, request.patch_size, rows.plan.channels, cancel=guard)
    summary = KMeansTrainingSummary(rows.plan.rows, fitted.n_steps, fitted.n_iter,
        fitted.source_reduction_profile, fitted.source_n_threads, manifest.manifest_sha256)
    provenance = _training_provenance(request, manifest, key[-1], fitted, exported.model, rows.features.device)
    artifact_payload = _artifact.encode_kmeans_artifact(exported.model, provenance)
    guard()
    _directory.validate_kmeans_training_directory(manifest, cancel=guard)
    return KMeansTrainingCompletion(KMeansTrainingCache(selection, manifest, key, rows, version),
        exported, summary, artifact_payload)
