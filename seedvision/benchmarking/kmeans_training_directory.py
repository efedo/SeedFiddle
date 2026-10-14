"""Bounded, byte-identified directory selection for native K-means training.

Qt reads raster headers only. OpenCV supplies unchanged gray/BGR pixels, which
are uploaded once per selected unique file; all analysis continues on CUDA.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import random
import stat

import cv2
import numpy as np
from PySide6.QtCore import QByteArray, QBuffer, QIODevice
from PySide6.QtGui import QImageReader
import torch

from seedvision.cuda import plantcv_kmeans_training as _training

MAX_DIRECTORY_ENTRIES = 4096
MAX_OCCURRENCES = 256
MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_SOURCE_BYTES = 128 * 1024 * 1024
MAX_DECODED_BYTES = 128 * 1024 * 1024
MAX_IMAGE_PIXELS = 16_000_000
DECODED_BYTES_PER_PIXEL = 4
PLANTCV_TRAIN_SHA256 = 'dbfb8a528e92984a20e73c7f265a41efa9fa088b42ed4f65addbe6ab40ee8b4d'
RASTER_FORMATS = ('bmp', 'cur', 'gif', 'icns', 'ico', 'jpeg', 'jpg', 'jfif',
    'pbm', 'pgm', 'png', 'ppm', 'tga', 'tif', 'tiff', 'wbmp', 'webp', 'xbm', 'xpm')
BYTE_FORMATS = ('Format_Mono', 'Format_MonoLSB', 'Format_Indexed8', 'Format_RGB32',
    'Format_ARGB32', 'Format_ARGB32_Premultiplied', 'Format_RGB16', 'Format_RGB666',
    'Format_ARGB6666_Premultiplied', 'Format_RGB555', 'Format_ARGB8555_Premultiplied',
    'Format_RGB888', 'Format_RGB444', 'Format_ARGB4444_Premultiplied',
    'Format_RGBX8888', 'Format_RGBA8888', 'Format_RGBA8888_Premultiplied',
    'Format_Alpha8', 'Format_Grayscale8', 'Format_BGR888', 'Format_CMYK8888')
# Qt's WebP plugin reports dimensions but no QImage format before decoding.
# VP8/VP8L WebP stores byte components, including optional alpha; four bytes
# per canvas pixel still bounds OpenCV's unchanged decoded output.
BYTE_ONLY_FORMATS = ('webp',)


class KMeansDirectoryError(ValueError):
    """Invalid/changed directory assets or unsupported resource/image domains."""


class KMeansDirectorySourceError(KMeansDirectoryError):
    """The pinned source cannot initialize its training-patch accumulator."""


class KMeansDirectoryCancelled(RuntimeError):
    """Directory work cancelled without publishing images or a fitted model."""


@dataclass(frozen=True, slots=True)
class TrainingImageOccurrence:
    index: int
    name: str
    status: str
    source_sha256: str | None
    source_bytes: int
    shape: tuple[int, ...] | None


@dataclass(frozen=True, slots=True)
class KMeansTrainingDirectory:
    directory: str
    listing: tuple[str, ...]
    selected_names: tuple[str, ...]
    prefix: str
    num_imgs: int
    occurrences: tuple[TrainingImageOccurrence, ...]
    source_bytes: int
    decoded_bytes: int
    manifest_sha256: str


@dataclass(frozen=True, slots=True)
class KMeansDirectoryImages:
    manifest: KMeansTrainingDirectory
    images: tuple[torch.Tensor, ...]


@dataclass(frozen=True, slots=True)
class KMeansDirectoryFitResult:
    manifest: KMeansTrainingDirectory
    training: _training.KMeansImageFitResult


def _poll(cancel):
    answer = cancel() if cancel is not None else False
    if type(answer) is not bool:
        raise KMeansDirectoryError('cancel callback must return bool')
    if answer:
        raise KMeansDirectoryCancelled('K-means directory training was cancelled')


def _listing(root):
    try:
        names = []
        with os.scandir(root) as entries:
            for entry in entries:
                if len(names) >= MAX_DIRECTORY_ENTRIES:
                    raise KMeansDirectoryError('training directory exceeds the entry-count bound')
                names.append(entry.name)
        return tuple(sorted(names))
    except OSError as error:
        raise KMeansDirectoryError(f'unable to list training directory: {error}') from error


def _read_source(path):
    try:
        before = path.stat()
        if not stat.S_ISREG(before.st_mode):
            return None
        if before.st_size > MAX_FILE_BYTES:
            raise KMeansDirectoryError('training source exceeds the per-file byte bound')
        with path.open('rb') as stream:
            raw = stream.read(MAX_FILE_BYTES+1)
        after = path.stat()
    except OSError as error:
        raise KMeansDirectoryError(f'unable to read training source: {error}') from error
    if len(raw) > MAX_FILE_BYTES:
        raise KMeansDirectoryError('training source exceeds the per-file byte bound')
    if (before.st_size != len(raw) or after.st_size != len(raw)
            or (before.st_mtime_ns, before.st_size) != (after.st_mtime_ns, after.st_size)):
        raise KMeansDirectoryError('training source changed while reading')
    return raw


def _header(raw):
    if not raw:
        return None
    buffer = QBuffer()
    buffer.setData(QByteArray(raw))
    buffer.open(QIODevice.OpenModeFlag.ReadOnly)
    reader = QImageReader(buffer)
    encoded_format = bytes(reader.format()).decode('ascii', errors='replace').lower()
    if not encoded_format:
        return None
    if encoded_format not in RASTER_FORMATS:
        raise KMeansDirectoryError(f'no bounded raster-header contract for {encoded_format!r}')
    size = reader.size()
    width, height = size.width(), size.height()
    if width <= 0 or height <= 0:
        return None
    if width*height > MAX_IMAGE_PIXELS:
        raise KMeansDirectoryError('training image exceeds the header pixel bound')
    if encoded_format not in BYTE_ONLY_FORMATS and reader.imageFormat().name not in BYTE_FORMATS:
        raise KMeansDirectoryError('training requires an eight-bit gray/BGR image header')
    # Palette/CMYK and packed byte formats may expand on decode. No pixel data
    # is read by Qt; all allowed byte outputs fit four bytes per header pixel.
    return height, width, DECODED_BYTES_PER_PIXEL*height*width


def _decode(raw, header):
    if header is None:
        return None
    try:
        image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    except cv2.error:
        return None
    if image is None:
        return None
    if (image.dtype != np.uint8 or image.ndim not in (2, 3)
            or (image.ndim == 3 and image.shape[2] != 3)):
        raise KMeansDirectoryError('training images must decode to uint8 gray or BGR')
    if tuple(image.shape[:2]) != header[:2] or image.nbytes > header[2]:
        raise KMeansDirectoryError('decoded training image differs from its bounded header')
    return image


def _manifest_digest(fields):
    return sha256(json.dumps(fields, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def resolve_kmeans_training_directory(directory, *, prefix='', num_imgs=0, cancel=None):
    """Freeze sorted selection; Python file choices stay independent of fit seed."""
    if type(prefix) is not str or type(num_imgs) is not int or not 0 <= num_imgs <= MAX_OCCURRENCES:
        raise KMeansDirectoryError('prefix must be text and num_imgs an integer in the selection bound')
    if cancel is not None and not callable(cancel):
        raise KMeansDirectoryError('cancel must be callable or None')
    _poll(cancel)
    try:
        root = Path(directory).expanduser().resolve(strict=True)
    except (OSError, RuntimeError, ValueError, TypeError) as error:
        raise KMeansDirectoryError(f'training directory is unavailable: {error}') from error
    listing = _listing(root)
    if not listing:
        raise KMeansDirectorySourceError('source training has no patch matrix: empty directory')
    if num_imgs == 0 and len(listing) > MAX_OCCURRENCES:
        raise KMeansDirectoryError('selected image occurrences exceed their count bound')
    _poll(cancel)
    selected = listing if num_imgs == 0 else tuple(random.choices(list(listing), k=num_imgs))
    _poll(cancel)
    facts, occurrences = {}, []
    source_bytes = decoded_bytes = 0
    for index, name in enumerate(selected):
        _poll(cancel)
        if prefix not in name:
            occurrences.append(TrainingImageOccurrence(index, name, 'prefix_skipped', None, 0, None))
            continue
        if name not in facts:
            raw = _read_source(root/name)
            _poll(cancel)
            if raw is None:
                facts[name] = ('non_file', None, 0, None)
            else:
                source_bytes += len(raw)
                if source_bytes > MAX_SOURCE_BYTES:
                    raise KMeansDirectoryError('selected source bytes exceed the aggregate bound')
                digest = sha256(raw).hexdigest()
                header = _header(raw)
                if header is None and raw and cv2.haveImageReader(str(root/name)):
                    raise KMeansDirectoryError('source image format lacks a bounded Qt raster header')
                _poll(cancel)
                if header is not None and decoded_bytes+header[2] > MAX_DECODED_BYTES:
                    raise KMeansDirectoryError('decoded images exceed the aggregate header bound')
                image = _decode(raw, header)
                _poll(cancel)
                if image is None:
                    facts[name] = ('decode_skipped', digest, len(raw), None)
                else:
                    shape = tuple(map(int, image.shape))
                    decoded_bytes += header[2]
                    facts[name] = ('accepted', digest, len(raw), shape)
                    del image
                del raw
        occurrences.append(TrainingImageOccurrence(index, name, *facts[name]))
        if occurrences[-1].status == 'accepted' and occurrences[0].status != 'accepted':
            raise KMeansDirectorySourceError('source training patch accumulator was never initialized at selected index zero')
    if not any(item.status == 'accepted' for item in occurrences):
        raise KMeansDirectorySourceError('source training has no patch matrix')
    fields = dict(directory=str(root), listing=listing, selected_names=selected, prefix=prefix,
        num_imgs=num_imgs, occurrences=tuple(occurrences), source_bytes=source_bytes,
        decoded_bytes=decoded_bytes)
    encoded = dict(fields, occurrences=[asdict(item) for item in occurrences])
    _poll(cancel)
    return KMeansTrainingDirectory(**fields, manifest_sha256=_manifest_digest(encoded))


def validate_kmeans_training_directory(manifest, *, cancel=None):
    """Reread every matched asset, including decoder skips, before GPU reuse."""
    if type(manifest) is not KMeansTrainingDirectory or type(manifest.occurrences) is not tuple:
        raise KMeansDirectoryError('expected an immutable training-directory manifest')
    if cancel is not None and not callable(cancel):
        raise KMeansDirectoryError('cancel must be callable or None')
    if (type(manifest.directory) is not str or type(manifest.prefix) is not str
            or type(manifest.listing) is not tuple or type(manifest.selected_names) is not tuple
            or not 1 <= len(manifest.occurrences) <= MAX_OCCURRENCES
            or not 1 <= len(manifest.listing) <= MAX_DIRECTORY_ENTRIES
            or len(manifest.selected_names) != len(manifest.occurrences)
            or type(manifest.num_imgs) is not int or not 0 <= manifest.num_imgs <= MAX_OCCURRENCES
            or (manifest.num_imgs and len(manifest.selected_names) != manifest.num_imgs)
            or (not manifest.num_imgs and manifest.selected_names != manifest.listing)
            or type(manifest.source_bytes) is not int or not 0 <= manifest.source_bytes <= MAX_SOURCE_BYTES
            or type(manifest.decoded_bytes) is not int or not 0 < manifest.decoded_bytes <= MAX_DECODED_BYTES
            or type(manifest.manifest_sha256) is not str
            or any(type(name) is not str or Path(name).name != name or name in ('.', '..')
                for name in (*manifest.listing, *manifest.selected_names))
            or manifest.listing != tuple(sorted(set(manifest.listing)))):
        raise KMeansDirectoryError('invalid training-directory manifest structure')
    unique = {}
    for index, item in enumerate(manifest.occurrences):
        if (type(item) is not TrainingImageOccurrence or type(item.index) is not int or item.index != index
                or type(item.name) is not str or item.name != manifest.selected_names[index]
                or item.name not in manifest.listing
                or type(item.status) is not str
                or item.status not in ('prefix_skipped', 'non_file', 'decode_skipped', 'accepted')
                or type(item.source_bytes) is not int or not 0 <= item.source_bytes <= MAX_FILE_BYTES
                or (manifest.prefix not in item.name) != (item.status == 'prefix_skipped')):
            raise KMeansDirectoryError('invalid training occurrence metadata')
        if item.status in ('prefix_skipped', 'non_file'):
            if item.source_sha256 is not None or item.source_bytes != 0 or item.shape is not None:
                raise KMeansDirectoryError('skipped training occurrence has invalid file facts')
        elif (type(item.source_sha256) is not str or len(item.source_sha256) != 64
                or any(char not in '0123456789abcdef' for char in item.source_sha256)):
            raise KMeansDirectoryError('invalid training source hash')
        if item.status == 'accepted':
            if (type(item.shape) is not tuple or len(item.shape) not in (2, 3)
                    or any(type(size) is not int or size <= 0 for size in item.shape)
                    or (len(item.shape) == 3 and item.shape[2] != 3)
                    or item.shape[0]*item.shape[1] > MAX_IMAGE_PIXELS):
                raise KMeansDirectoryError('invalid accepted training image geometry')
        elif item.shape is not None:
            raise KMeansDirectoryError('skipped training occurrence has decoded shape facts')
        facts = (item.status, item.source_sha256, item.source_bytes, item.shape)
        if item.name in unique and unique[item.name] != facts:
            raise KMeansDirectoryError('duplicate training occurrences have inconsistent facts')
        unique[item.name] = facts
    if not manifest.occurrences or manifest.occurrences[0].status != 'accepted':
        raise KMeansDirectoryError('manifest cannot initialize the source training accumulator')
    if (manifest.source_bytes != sum(facts[2] for facts in unique.values())
            or manifest.decoded_bytes != sum(DECODED_BYTES_PER_PIXEL*facts[3][0]*facts[3][1]
                for facts in unique.values() if facts[0] == 'accepted')):
        raise KMeansDirectoryError('training-directory aggregate byte facts are invalid')
    fields = asdict(manifest)
    digest = fields.pop('manifest_sha256')
    if digest != _manifest_digest(fields):
        raise KMeansDirectoryError('training-directory manifest identity is invalid')
    if (not 1 <= len(manifest.occurrences) <= MAX_OCCURRENCES
            or len(manifest.listing) > MAX_DIRECTORY_ENTRIES
            or manifest.source_bytes > MAX_SOURCE_BYTES or manifest.decoded_bytes > MAX_DECODED_BYTES):
        raise KMeansDirectoryError('training-directory manifest exceeds its metadata bounds')
    root = Path(manifest.directory)
    try:
        if str(root.resolve(strict=True)) != manifest.directory or _listing(root) != manifest.listing:
            raise KMeansDirectoryError('training directory listing changed; select it again')
    except (OSError, RuntimeError) as error:
        raise KMeansDirectoryError('training directory is unavailable') from error
    _verify_sources(manifest, root, cancel=cancel)
    _poll(cancel)
    # No callback may mutate an already checked file after the final snapshot
    # check. A second byte/listing pass follows all user callbacks, including
    # callbacks that changed an earlier file while a later file was checked.
    _verify_sources(manifest, root)
    return manifest


def _verify_sources(manifest, root, *, cancel=None):
    if _listing(root) != manifest.listing:
        raise KMeansDirectoryError('training directory listing changed; select it again')
    seen = set()
    for index, item in enumerate(manifest.occurrences):
        _poll(cancel)
        if (type(item) is not TrainingImageOccurrence or item.index != index
                or item.name != manifest.selected_names[index] or item.name not in manifest.listing
                or Path(item.name).name != item.name or item.name in ('.', '..')
                or item.status not in ('prefix_skipped', 'non_file', 'decode_skipped', 'accepted')):
            raise KMeansDirectoryError('invalid training occurrence metadata')
        if (manifest.prefix not in item.name) != (item.status == 'prefix_skipped'):
            raise KMeansDirectoryError('training occurrence prefix state is invalid')
        if item.status == 'prefix_skipped' or item.name in seen:
            continue
        seen.add(item.name)
        raw = _read_source(root/item.name)
        if item.status == 'non_file':
            if raw is not None:
                raise KMeansDirectoryError('training entry changed from a non-file')
        elif raw is None or len(raw) != item.source_bytes or sha256(raw).hexdigest() != item.source_sha256:
            raise KMeansDirectoryError('training source bytes changed; select the directory again')
        del raw
        _poll(cancel)
    if _listing(root) != manifest.listing:
        raise KMeansDirectoryError('training directory listing changed; select it again')


def materialize_kmeans_training_directory(manifest, *, device='cuda', cancel=None):
    """Verify the complete byte snapshot first, then upload unique decoded inputs."""
    validate_kmeans_training_directory(manifest, cancel=cancel)
    target = torch.device(device)
    if target.type != 'cuda':
        raise KMeansDirectoryError('training images must be materialized on CUDA')
    uploaded = {}
    for item in manifest.occurrences:
        if item.status != 'accepted' or item.name in uploaded:
            continue
        _poll(cancel)
        raw = _read_source(Path(manifest.directory)/item.name)
        if raw is None or len(raw) != item.source_bytes or sha256(raw).hexdigest() != item.source_sha256:
            raise KMeansDirectoryError('training source changed before image upload')
        image = _decode(raw, _header(raw))
        if image is None or tuple(image.shape) != item.shape:
            raise KMeansDirectoryError('training decode changed before image upload')
        _poll(cancel)
        uploaded[item.name] = torch.from_numpy(np.ascontiguousarray(image)).to(target)
        del image, raw
        _poll(cancel)
    validate_kmeans_training_directory(manifest, cancel=cancel)
    return KMeansDirectoryImages(manifest, tuple(uploaded[item.name]
        for item in manifest.occurrences if item.status == 'accepted'))


def fit_kmeans_training_directory(manifest, n_clusters, *, source_n_threads,
        device='cuda', cancel=None, **parameters):
    """Fit the byte-verified selected occurrences without implicit model writes."""
    materialized = materialize_kmeans_training_directory(manifest, device=device, cancel=cancel)
    def guard():
        _poll(cancel)
        return False
    result = _training.fit_kmeans_training_images(materialized.images, n_clusters,
        source_n_threads=source_n_threads, cancel=guard, **parameters)
    validate_kmeans_training_directory(manifest, cancel=cancel)
    return KMeansDirectoryFitResult(manifest, result)
