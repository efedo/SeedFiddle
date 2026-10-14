"""Bounded CUDA inference for frozen PlantCV 4.11.3 Naive Bayes PDFs.

The public rgb_img argument is converted with COLOR_BGR2HSV. This API therefore
requires contiguous CUDA uint8 BGR HWC pixels. Only bounded compact TSV model
data is parsed on CPU; full images, HSV tiles, likelihoods and masks stay CUDA.
Engineering conformance does not establish scientific model validation.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import io
import math
from pathlib import Path
import re
import struct
from types import MappingProxyType
from typing import Mapping

import torch
from seedvision.cuda.tensor_callback import checked_cancel

from seedvision.cuda import plantcv_range as _range

PLANTCV_VERSION = '4.11.3'
PLANTCV_SOURCE_SHA256 = 'd3f40429ead7aa5deb298fa78d484d0f6d89c9cf05e706e07fdf7b97fedc3049'
CHANNELS = ('hue', 'saturation', 'value')
MAX_MODEL_BYTES = 8 * 1024 * 1024
MAX_CLASSES = 32
MAX_PIXELS = 16_000_000
MAX_OUTPUT_BYTES = 256 * 1024 * 1024
MAX_SCRATCH_BYTES = 64 * 1024 * 1024
MAX_CUDA_LIVE_BYTES = 512 * 1024 * 1024
SCRATCH_ENVELOPE_BYTES_PER_PIXEL = 1024
TILE_PIXELS = 16_384


class NaiveBayesError(ValueError):
    """Input, model, device, or resource failure."""


class NaiveBayesCancelled(RuntimeError):
    """Inference was cancelled before a complete class-mask bundle existed."""


@dataclass(frozen=True, slots=True)
class NaiveBayesModel:
    class_names: tuple[str, ...]
    probabilities: tuple[tuple[tuple[float, ...], ...], ...]
    source_sha256: str
    source_bytes: int
    tables_sha256: str


@dataclass(frozen=True, slots=True)
class NaiveBayesResult:
    masks: Mapping[str, torch.Tensor]
    class_names: tuple[str, ...]
    model_sha256: str
    reference_version: str = PLANTCV_VERSION
    reference_source_sha256: str = PLANTCV_SOURCE_SHA256


def _poll(cancel, *tensors):
    if checked_cancel(cancel,tensors):
        raise NaiveBayesCancelled('Naive Bayes inference was cancelled before a complete result.')


def _validate_limits(max_model_bytes, max_classes):
    if (type(max_model_bytes) is not int or not 0 < max_model_bytes <= MAX_MODEL_BYTES
            or type(max_classes) is not int or not 0 < max_classes <= MAX_CLASSES):
        raise NaiveBayesError('Model limits must be positive and cannot exceed implementation caps.')


def _table_digest(class_names, probabilities):
    digest = sha256()
    for class_name, channels in zip(class_names, probabilities):
        encoded = class_name.encode('utf-8')
        digest.update(struct.pack('>I', len(encoded)))
        digest.update(encoded)
        for channel in channels:
            for value in channel:
                digest.update(struct.pack('>d', value))
    return digest.hexdigest()


def parse_plantcv_pdf(source: bytes | str, *, max_model_bytes: int = MAX_MODEL_BYTES,
                      max_classes: int = MAX_CLASSES) -> NaiveBayesModel:
    """Parse the source's header-plus-258-column TSV with bounded model data.

    Class insertion order and last-row replacement match PlantCV. Unknown
    channels are parsed and validated but ignored by inference, as in source.
    Supported likelihoods are finite nonnegative binary64 values whose largest
    left-associated product is finite; NaNs and overflow are withheld.
    """
    _validate_limits(max_model_bytes, max_classes)
    if isinstance(source, bytes):
        raw = source
        if len(raw) > max_model_bytes:
            raise NaiveBayesError('PlantCV PDF exceeds the compact model byte limit.')
        try:
            text = raw.decode('utf-8')
        except UnicodeDecodeError as error:
            raise NaiveBayesError('PlantCV PDF must be UTF-8 text.') from error
    elif isinstance(source, str):
        if len(source) > max_model_bytes:
            raise NaiveBayesError('PlantCV PDF exceeds the compact model byte limit.')
        try:
            raw = source.encode('utf-8')
        except UnicodeEncodeError as error:
            raise NaiveBayesError('PlantCV PDF must be UTF-8 text.') from error
        if len(raw) > max_model_bytes:
            raise NaiveBayesError('PlantCV PDF exceeds the compact model byte limit.')
        text = source
    else:
        raise NaiveBayesError('Provide the frozen PDF as UTF-8 text or bytes.')
    if not raw:
        raise NaiveBayesError('PlantCV PDF is empty.')

    classes: dict[str, dict[str, tuple[float, ...]]] = {}
    stream = io.StringIO(text, newline=None)
    stream.readline()
    for line_number, row in enumerate(stream, start=2):
        columns = row.rstrip('\n').split('\t')
        if len(columns) != 258:
            raise NaiveBayesError(f'PlantCV PDF row {line_number} has {len(columns)} fields; expected 258.')
        class_name, channel = columns[:2]
        if not class_name:
            raise NaiveBayesError(f'PlantCV PDF row {line_number} has an empty class name.')
        if class_name not in classes:
            if len(classes) >= max_classes:
                raise NaiveBayesError('PlantCV PDF exceeds the class-count limit.')
            classes[class_name] = {}
        try:
            values = tuple(float(value) for value in columns[2:])
        except ValueError as error:
            raise NaiveBayesError(f'PlantCV PDF row {line_number} contains a nonnumeric likelihood.') from error
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise NaiveBayesError(f'PlantCV PDF row {line_number} requires 256 finite nonnegative likelihoods.')
        classes[class_name][channel] = values
    if not classes:
        raise NaiveBayesError('PlantCV PDF contains no class rows.')
    frozen = []
    for class_name, channels in classes.items():
        missing = [name for name in CHANNELS if name not in channels]
        if missing:
            raise NaiveBayesError(f'Class {class_name!r} lacks required HSV channels: {", ".join(missing)}.')
        ordered = tuple(channels[name] for name in CHANNELS)
        maxima = tuple(max(values) for values in ordered)
        if not math.isfinite(maxima[0]*maxima[1]*maxima[2]):
            raise NaiveBayesError(f'Class {class_name!r} can overflow a source likelihood product.')
        frozen.append(ordered)
    class_names, tables = tuple(classes), tuple(frozen)
    return NaiveBayesModel(class_names, tables, sha256(raw).hexdigest(), len(raw),
        _table_digest(class_names, tables))


def load_plantcv_pdf(path: str | Path, *, max_model_bytes: int = MAX_MODEL_BYTES,
                     max_classes: int = MAX_CLASSES) -> NaiveBayesModel:
    """Read at most the configured model byte cap plus one sentinel byte."""
    _validate_limits(max_model_bytes, max_classes)
    try:
        with Path(path).open('rb') as stream:
            raw = stream.read(max_model_bytes+1)
    except OSError as error:
        raise NaiveBayesError(f'Unable to read the frozen PlantCV PDF: {error}') from error
    if len(raw) > max_model_bytes:
        raise NaiveBayesError('PlantCV PDF exceeds the compact model byte limit.')
    return parse_plantcv_pdf(raw, max_model_bytes=max_model_bytes, max_classes=max_classes)


def _validate_model(model: NaiveBayesModel) -> None:
    if not isinstance(model, NaiveBayesModel):
        raise NaiveBayesError('Expected a parsed immutable PlantCV PDF model.')
    if type(model.class_names) is not tuple or type(model.probabilities) is not tuple:
        raise NaiveBayesError('Parsed model names and tables must be immutable tuples.')
    count = len(model.class_names)
    if (count == 0 or count > MAX_CLASSES
            or any(type(name) is not str or not name for name in model.class_names)
            or len(set(model.class_names)) != count
            or type(model.source_bytes) is not int or not 0 < model.source_bytes <= MAX_MODEL_BYTES
            or len(model.probabilities) != count
            or not isinstance(model.source_sha256, str)
            or not re.fullmatch(r'[0-9a-f]{64}', model.source_sha256)
            or not isinstance(model.tables_sha256, str)
            or not re.fullmatch(r'[0-9a-f]{64}', model.tables_sha256)):
        raise NaiveBayesError('Parsed PlantCV PDF model metadata is invalid.')
    for channels in model.probabilities:
        if (type(channels) is not tuple or len(channels) != 3
                or any(type(values) is not tuple or len(values) != 256 for values in channels)):
            raise NaiveBayesError('Parsed PlantCV PDF table shape is invalid.')
        if any(type(value) is not float or not math.isfinite(value) or value < 0.0
                for values in channels for value in values):
            raise NaiveBayesError('Parsed PlantCV PDF tables contain unsupported likelihood values.')
        maxima = tuple(max(values) for values in channels)
        if not math.isfinite(maxima[0]*maxima[1]*maxima[2]):
            raise NaiveBayesError('Parsed PlantCV PDF can overflow a three-channel product.')
    try:
        digest = _table_digest(model.class_names, model.probabilities)
    except (UnicodeError, struct.error) as error:
        raise NaiveBayesError('Parsed PlantCV PDF class names or table encoding is invalid.') from error
    if digest != model.tables_sha256:
        raise NaiveBayesError('Parsed PlantCV PDF tables differ from their frozen table digest.')


def naive_bayes_masks(bgr_image: torch.Tensor, model: NaiveBayesModel, *,
                      cancel=None, tile_pixels: int = TILE_PIXELS) -> NaiveBayesResult:
    """Return complete per-class CUDA uint8 0/255 masks in source class order."""
    if cancel is not None and not callable(cancel):
        raise NaiveBayesError('cancel must be callable or None.')
    _poll(cancel,bgr_image)
    _validate_model(model)
    if (not torch.is_tensor(bgr_image) or bgr_image.layout != torch.strided
            or bgr_image.device.type != 'cuda' or bgr_image.dtype != torch.uint8
            or bgr_image.ndim != 3 or bgr_image.shape[-1] != 3 or not bgr_image.is_contiguous()):
        raise NaiveBayesError('Require contiguous CUDA uint8 BGR HWC input.')
    if type(tile_pixels) is not int or not 0 < tile_pixels <= TILE_PIXELS:
        raise NaiveBayesError(f'tile_pixels must be in 1..{TILE_PIXELS}.')
    height, width, _ = bgr_image.shape
    pixels = height*width
    class_count = len(model.class_names)
    output_bytes = pixels*class_count
    table_bytes = class_count*3*256*8
    tile = min(tile_pixels, pixels)
    # 1 KiB per active pixel conservatively covers the bounded converter's int32
    # intermediates/gathers, int64 HSV indexes, float64 lookups/products/best,
    # winner/comparison vectors and elementwise temporaries, including tile overlap.
    tile_scratch_bytes = tile*SCRATCH_ENVELOPE_BYTES_PER_PIXEL
    if (pixels == 0 or pixels > MAX_PIXELS or output_bytes > MAX_OUTPUT_BYTES
            or tile_scratch_bytes > MAX_SCRATCH_BYTES
            or output_bytes+table_bytes+tile_scratch_bytes > MAX_CUDA_LIVE_BYTES):
        raise NaiveBayesError('Image, model table, masks, or tiled scratch exceeds the CUDA resource budget.')
    _poll(cancel,bgr_image)
    table = torch.tensor(model.probabilities, dtype=torch.float64, device=bgr_image.device)
    masks = torch.zeros((class_count, pixels), dtype=torch.uint8, device=bgr_image.device)
    flat = bgr_image.view(-1, 3)
    for start in range(0, pixels, tile):
        _poll(cancel,bgr_image,table,masks)
        end = min(start+tile, pixels)
        hsv = _range._channels(flat[start:end], 'HSV')
        if (not torch.is_tensor(hsv) or hsv.device != bgr_image.device
                or hsv.dtype != torch.uint8 or tuple(hsv.shape) != (end-start, 3)):
            raise NaiveBayesError('Verified HSV conversion returned an invalid CUDA byte tile.')
        hue, saturation, value = (channel.to(torch.int64) for channel in hsv.unbind(dim=1))
        best = None
        winner = torch.zeros((end-start,), dtype=torch.int64, device=bgr_image.device)
        for class_index in range(class_count):
            _poll(cancel,bgr_image,table,masks,hsv,hue,saturation,value,best,winner)
            channels = table[class_index]
            probability = channels[0].index_select(0, hue)
            selected = channels[1].index_select(0, saturation)
            probability.mul_(selected)
            del selected
            selected = channels[2].index_select(0, value)
            probability.mul_(selected)
            del selected
            if best is None:
                best = probability
            else:
                better = probability > best
                winner.masked_fill_(better, class_index)
                # Separate kernels retain left-associated binary64 multiplication
                # and strict first-index selection when likelihoods are equal.
                torch.maximum(best, probability, out=best)
        for class_index in range(class_count):
            masks[class_index, start:end].copy_((winner == class_index).to(torch.uint8).mul_(255))
    _poll(cancel,bgr_image,table,masks)
    class_masks = {name: masks[index].view(height, width)
        for index, name in enumerate(model.class_names)}
    return NaiveBayesResult(MappingProxyType(class_masks), model.class_names, model.source_sha256)
