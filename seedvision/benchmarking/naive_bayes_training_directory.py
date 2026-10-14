"""Source-ordered binary Bayes directory training with CUDA sampling/KDE.

OpenCV decodes one bounded image/mask pair using the public colour/gray flags.
Images, masks, HSV, populations and retained samples then stay on CUDA. The
ordered traversal and source bytes are audited after the last caller callback.
This API's distinct provenance is descriptive, not a scientific certificate.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import stat

import cv2
import numpy as np
import PySide6
from PySide6.QtCore import QByteArray, QBuffer, QIODevice, qVersion
from PySide6.QtGui import QImageReader
import torch

from seedvision.cuda import _nvrtc
from seedvision.cuda import plantcv_naive_bayes as _model
from seedvision.cuda import plantcv_naive_bayes_sampling as _sampling
from seedvision.cuda import plantcv_naive_bayes_training as _fit
from seedvision.cuda import plantcv_range as _range

MAX_DIRECTORY_ENTRIES = 4096
MAX_OCCURRENCES = 256
MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_SOURCE_BYTES = 128 * 1024 * 1024
MAX_DECODED_BYTES = 128 * 1024 * 1024
MAX_MANIFEST_BYTES = 512 * 1024
MAX_JOB_WORK = 2_000_000_000
PARSER_ID = 'plantcv-4.11.3-ordered-binary-directories-v2'
CLASS_NAMES = ('plant', 'background')
# Other source-decodable formats remain outside this native header contract.
RASTER_FORMATS = ('png', 'jpeg', 'jpg')


class BayesDirectoryError(ValueError):
    """Changed assets or an unsupported native directory/resource domain."""


class BayesDirectorySourceError(BayesDirectoryError):
    """Ordered processing reaches an undecodable source image or mask."""


@dataclass(frozen=True, slots=True)
class BayesImageMaskPair:
    index: int
    image_relative_path: str
    mask_relative_path: str


@dataclass(frozen=True, slots=True)
class BayesDirectorySelection:
    image_root_origin: str
    mask_root_origin: str
    walk: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...]
    pairs: tuple[BayesImageMaskPair, ...]
    skipped_missing_masks: tuple[str, ...]
    selection_sha256: str


@dataclass(frozen=True, slots=True)
class BayesDirectoryTrainingCompletion:
    selection: BayesDirectorySelection
    samples: tuple[torch.Tensor, torch.Tensor]
    fit: _fit.BayesTrainingFit
    model: _model.NaiveBayesModel
    tsv_bytes: bytes
    provenance_json: bytes


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode('utf-8')


def _digest(value):
    return sha256(_json(value)).hexdigest()


def _root(directory):
    try:
        root = Path(directory).expanduser().resolve(strict=True)
        if not root.is_dir():
            raise BayesDirectoryError('training roots must be directories')
        return root
    except (OSError, RuntimeError, ValueError, TypeError) as error:
        raise BayesDirectoryError(f'training root is unavailable: {error}') from error


def _path(root, relative):
    try:
        path = (root/relative).resolve(strict=True)
        if not path.is_relative_to(root):
            raise BayesDirectoryError('training assets must resolve inside their selected root')
        return path
    except (OSError, RuntimeError, ValueError) as error:
        raise BayesDirectoryError(f'training asset is unavailable: {error}') from error


def _selection(image_root, mask_root):
    walk, pairs, skipped = [], [], []
    entries = 0
    def failed(error):
        raise BayesDirectoryError(f'unable to traverse training directory: {error}') from error
    # Do not sort: the public implementation consumes the actual os.walk order.
    for directory, directories, filenames in os.walk(image_root, onerror=failed):
        entries += len(directories)+len(filenames)
        if entries > MAX_DIRECTORY_ENTRIES or len(walk) >= MAX_DIRECTORY_ENTRIES:
            raise BayesDirectoryError('training traversal exceeds the entry-count bound')
        relative_directory = Path(directory).relative_to(image_root).as_posix()
        walk.append((relative_directory, tuple(directories), tuple(filenames)))
        for filename in filenames:
            if filename[-3:] not in ('png', 'jpg', 'jpeg'):
                continue
            relative = (Path(relative_directory)/filename).as_posix()
            if not os.path.exists(mask_root/filename):
                skipped.append(relative)
                continue
            if len(pairs) >= MAX_OCCURRENCES:
                raise BayesDirectoryError('training selection exceeds the pair-occurrence bound')
            pairs.append(BayesImageMaskPair(len(pairs), relative, filename))
    fields = dict(image_root_origin=str(image_root), mask_root_origin=str(mask_root),
        walk=walk, pairs=[asdict(pair) for pair in pairs], skipped_missing_masks=skipped)
    encoded = _json(fields)
    if len(encoded) > MAX_MANIFEST_BYTES:
        raise BayesDirectoryError('training traversal exceeds the manifest byte bound')
    return BayesDirectorySelection(str(image_root), str(mask_root), tuple(walk),
        tuple(pairs), tuple(skipped), sha256(encoded).hexdigest())


def resolve_bayes_training_directories(image_directory, mask_directory):
    """Freeze traversal/pairing without prematurely decoding later occurrences.

    A missing flat mask skips its candidate. Duplicate basenames in different
    image subdirectories remain separate occurrences, using the same flat mask.
    Source suffix matching is case-sensitive last-three-character matching.
    """
    return _selection(_root(image_directory), _root(mask_directory))


def _read(root, relative):
    path = _path(root, relative)
    try:
        before = path.stat()
        if not stat.S_ISREG(before.st_mode):
            raise BayesDirectorySourceError('source image/mask is not a decodable regular file')
        if before.st_size > MAX_FILE_BYTES:
            raise BayesDirectoryError('training source exceeds the per-file byte bound')
        with path.open('rb') as stream:
            raw = stream.read(MAX_FILE_BYTES+1)
        after = path.stat()
    except OSError as error:
        raise BayesDirectoryError(f'unable to read training source: {error}') from error
    if len(raw) > MAX_FILE_BYTES:
        raise BayesDirectoryError('training source exceeds the per-file byte bound')
    if (before.st_size != len(raw) or after.st_size != len(raw)
            or before.st_mtime_ns != after.st_mtime_ns or _path(root, relative) != path):
        raise BayesDirectoryError('training source changed while reading')
    return raw, dict(sha256=sha256(raw).hexdigest(), bytes=len(raw))


def _header(raw):
    if not raw:
        raise BayesDirectorySourceError('source image/mask is undecodable')
    buffer = QBuffer()
    buffer.setData(QByteArray(raw))
    buffer.open(QIODevice.OpenModeFlag.ReadOnly)
    reader = QImageReader(buffer)
    encoded_format = bytes(reader.format()).decode('ascii', errors='replace').lower()
    if not encoded_format:
        raise BayesDirectorySourceError('source image/mask is undecodable')
    if encoded_format not in RASTER_FORMATS:
        raise BayesDirectoryError(f'unsupported native training raster header: {encoded_format!r}')
    size = reader.size()
    height, width = size.height(), size.width()
    if height <= 0 or width <= 0:
        raise BayesDirectorySourceError('source image/mask header is undecodable')
    if height*width > _sampling.MAX_IMAGE_PIXELS:
        raise BayesDirectoryError('training source exceeds the header pixel bound')
    return height, width


def _decode(raw, header, flags):
    try:
        image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), flags)
    except cv2.error as error:
        raise BayesDirectorySourceError('source image/mask is undecodable') from error
    if image is None:
        raise BayesDirectorySourceError('source image/mask is undecodable')
    expected_dimensions = 3 if flags == cv2.IMREAD_COLOR else 2
    # Source default flags may apply EXIF rotation; pixel count stays bounded.
    if (image.dtype != np.uint8 or image.ndim != expected_dimensions
            or (expected_dimensions == 3 and image.shape[-1] != 3)
            or tuple(image.shape[:2]) not in (header, header[::-1])
            or not image.flags.c_contiguous):
        raise BayesDirectoryError('decoded source differs from the bounded BGR/gray header contract')
    return image


def _backing(tensors):
    return sum({(x.device, x.untyped_storage().data_ptr()): x.untyped_storage().nbytes()
        for x in tensors}.values())


def _calculation_records():
    from seedvision.benchmarking.native_recipe import _module_record
    modules = (_sampling, _fit, _nvrtc, _range, _model,
        __import__(__name__, fromlist=['train_bayes_directories']))
    return {module.__name__: _module_record(module) for module in modules}


def _rng_record(state):
    return dict(algorithm=state.algorithm, keys=state.keys.hex(), position=state.position,
        has_gauss=state.has_gauss, cached_gaussian=state.cached_gaussian.hex(), sha256=state.sha256)


def _fit_preflight(total_rows, retained_bytes, consumed_work):
    if total_rows < 4 or total_rows > _fit.MAX_SAMPLES:
        raise _fit.BayesTrainingError('each binary class needs at least two samples within the global cap')
    scratch = 3*256*256*8 + 2*3*256*8 + 512 + _fit.HEADROOM_BYTES
    work = total_rows*(3*3+3+3*256) + 2*(3*256*256+3*256)
    # The concatenated samples coexist with retained chunks during torch.cat.
    peak = retained_bytes+3*total_rows+scratch
    if (scratch > _fit.MAX_SCRATCH_BYTES or peak > _fit.MAX_CUDA_LIVE_BYTES
            or work > _fit.MAX_WORK or consumed_work+work > MAX_JOB_WORK):
        raise _fit.BayesTrainingError('directory concatenation and fit exceed the combined resource bound')
    return dict(work=work, peak_live_bytes=peak, scratch_bytes=scratch)


def train_bayes_directories(image_directory, mask_directory, *, device='cuda', rng=None,
                            cancel=None, conversion_tile=16_384, gather_tile=16_384):
    """Fit binary PDFs from source-ordered pairs, returning interoperable TSV.

    Uses the existing global legacy NumPy stream unless a captured RandomState
    is supplied. No seed is invented. Native caps, cancellation and late source
    failures preserve draws already consumed and publish no completion. PNG and
    JPEG contents use source row/column mask indexing, including different
    dimensions when every categorical coordinate is in bounds. Decoder output
    storage is bounded; this is not a
    guarantee about codec-internal or whole-process CPU working memory.
    """
    if cancel is not None and not callable(cancel):
        raise BayesDirectoryError('cancel must be callable or None')
    if rng is not None and type(rng) is not np.random.RandomState:
        raise BayesDirectoryError('rng must be a legacy RandomState or None')
    device = torch.device(device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise BayesDirectoryError('Bayes directory training requires an available CUDA device')
    rng = np.random if rng is None else rng
    guard = _sampling._Guard(rng, cancel, ())
    before = guard.expected_rng
    def checkpoint():
        guard.checkpoint()
        return False
    checkpoint()
    selection = resolve_bayes_training_directories(image_directory, mask_directory)
    image_root, mask_root = Path(selection.image_root_origin), Path(selection.mask_root_origin)
    records = _calculation_records()
    sources, occurrences, parts = {}, [], [[], []]
    budget = _sampling._JobBudget()
    def retained_unchanged():
        if any(_fit._tensor_signature(x) != signature for x, signature in guard.tensors.values()):
            raise BayesDirectoryError('retained Bayes directory samples changed during calculation or callback')
    def sampling_callback():
        # The inner sampler guards the changing RNG on every actual draw. The
        # outer guard checks only the earlier pairs' retained sample tensors.
        retained_unchanged()
        try:
            return cancel() if cancel is not None else False
        finally:
            retained_unchanged()
    with torch.cuda.device(device):
        for pair in selection.pairs:
            checkpoint()
            image_raw, image_identity = _read(image_root, pair.image_relative_path)
            image_header = _header(image_raw)
            mask_raw, mask_identity = _read(mask_root, pair.mask_relative_path)
            mask_header = _header(mask_raw)
            for kind, relative, identity in (('image', pair.image_relative_path, image_identity),
                    ('mask', pair.mask_relative_path, mask_identity)):
                key = kind, relative
                if key in sources and sources[key] != identity:
                    raise BayesDirectoryError('a repeated training source changed between occurrences')
                sources[key] = identity
            if sum(identity['bytes'] for identity in sources.values()) > MAX_SOURCE_BYTES:
                raise BayesDirectoryError('training selection exceeds the aggregate source-byte bound')
            if 3*image_header[0]*image_header[1]+mask_header[0]*mask_header[1] > MAX_DECODED_BYTES:
                raise BayesDirectoryError('training pair exceeds the decoded-output byte bound')
            image = _decode(image_raw, image_header, cv2.IMREAD_COLOR)
            mask = _decode(mask_raw, mask_header, cv2.IMREAD_GRAYSCALE)
            del image_raw, mask_raw
            pixels = int(mask.size)
            tiles = (pixels+_sampling.MASK_TILE_PIXELS-1)//_sampling.MASK_TILE_PIXELS
            if (budget.retained_backing_bytes+image.nbytes+mask.nbytes+tiles*16+
                    _sampling.CUDA_HEADROOM_BYTES > _sampling.MAX_CUDA_LIVE_BYTES):
                raise BayesDirectoryError('training pair upload/count storage exceeds the cumulative CUDA bound')
            checkpoint()
            source = torch.tensor(image, dtype=torch.uint8, device=device)
            binary = torch.tensor(mask, dtype=torch.uint8, device=device)
            del image, mask
            # The inner API applies the retained backing/work/rows reservation
            # before current scratch allocation and before its first RNG draw.
            sampled = _sampling.sample_naive_bayes_image(source, binary,
                rng=None if rng is np.random else rng, cancel=sampling_callback,
                conversion_tile=conversion_tile, gather_tile=gather_tile, _budget=budget)
            retained_unchanged()
            guard.expected_rng = sampled.rng_after
            guard.watch(*sampled.samples)
            for category in range(2):
                parts[category].append(sampled.samples[category])
            occurrences.append(dict(pair=asdict(pair), image=image_identity, mask=mask_identity,
                image_shape=tuple(source.shape), mask_shape=tuple(binary.shape),
                sampling_plan=asdict(sampled.plan), draws_sha256=sampled.draws_sha256,
                random_index_dtype=sampled.random_index_dtype, random_calls=sampled.random_calls,
                rng_before_sha256=sampled.rng_before.sha256, rng_after_sha256=sampled.rng_after.sha256))
            retained = tuple(x for category in parts for x in category)
            budget = _sampling._JobBudget(_backing(retained),
                sum(int(x.shape[0]) for x in retained), budget.consumed_work+sampled.plan.work)
            del retained, source, binary, sampled
            checkpoint()
        fit_preflight = _fit_preflight(budget.retained_rows, budget.retained_backing_bytes, budget.consumed_work)
        checkpoint()
        samples = tuple(torch.cat(category, dim=0) for category in parts)
        guard.watch(*samples)
        for category in parts:
            for part in category:
                del guard.tensors[id(part)]
        del part, category, parts
        fitted = _fit.fit_naive_bayes_hsv(samples, CLASS_NAMES, cancel=checkpoint)
        guard.watch(fitted.probabilities)
        checkpoint()
        compact = fitted.probabilities.cpu().numpy()
        rows = ['class\tchannel\t'+'\t'.join(map(str, range(256)))]
        for channel_index, channel in enumerate(_model.CHANNELS):
            for category_index, name in enumerate(CLASS_NAMES):
                rows.append('\t'.join((name, channel, *map(str, compact[category_index, channel_index]))))
        payload = ('\n'.join(rows)+'\n').encode('utf-8')
        model = _model.parse_plantcv_pdf(payload)
        provenance = dict(kind='native_bayes_directory_training', parser_id=PARSER_ID,
            selection=asdict(selection), occurrences=occurrences,
            source_bytes=sum(identity['bytes'] for identity in sources.values()),
            class_names=CLASS_NAMES, sample_counts=tuple(int(x.shape[0]) for x in samples),
            rng_before=_rng_record(before), rng_after=_rng_record(guard.expected_rng),
            sampling_work=budget.consumed_work, fit_preflight=fit_preflight,
            model_sha256=model.source_sha256, tables_sha256=model.tables_sha256,
            fit_plan=asdict(fitted.plan), channel_fits=[[asdict(f) for f in cs] for cs in fitted.channels],
            calculation=records, scientific_validation=False,
            runtime=dict(torch=str(torch.__version__), numpy=np.__version__, torch_cuda=str(torch.version.cuda),
                opencv=cv2.__version__, opencv_build_sha256=sha256(cv2.getBuildInformation().encode()).hexdigest(),
                pyside=PySide6.__version__, qt=qVersion(), device=str(samples[0].device),
                device_name=torch.cuda.get_device_name(samples[0].device), **_nvrtc.runtime_versions()),
            source_pins=dict(plantcv='4.11.3', scipy='1.15.3', numpy='2.2.6',
                plantcv_train_sha256=_fit.PLANTCV_TRAIN_SHA256, scipy_kde_sha256=_fit.SCIPY_KDE_SHA256,
                scipy_stats_sha256=_fit.SCIPY_STATS_SHA256, numpy_pairwise_sha256=_fit.NUMPY_PAIRWISE_SHA256))
        encoded = _json(provenance)
        if len(encoded) > MAX_MANIFEST_BYTES:
            raise BayesDirectoryError('directory provenance exceeds the byte bound')
        checkpoint()
        # No caller callbacks follow this audit. New/missing/reordered pairs,
        # added masks and edits during fitting or the last callback are caught.
        if _selection(image_root, mask_root) != selection:
            raise BayesDirectoryError('training directory traversal or pairing changed during training')
        for (kind, relative), identity in sources.items():
            _, checked = _read(image_root if kind == 'image' else mask_root, relative)
            if checked != identity:
                raise BayesDirectoryError('selected training source changed during training')
        guard.check()
        if _calculation_records() != records:
            raise BayesDirectoryError('directory training implementation changed before publication')
        return BayesDirectoryTrainingCompletion(selection, samples, fitted, model, payload, encoded)
