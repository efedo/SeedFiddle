"""Bounded CUDA measurements reproducing pinned PlantCV 4.11.3 analyze.color.

Dense HWC uint8 BGR images and uint8/int32 label masks stay on CUDA. Only compact
per-label integer histograms cross to CPU for observation/statistic metadata.
The public empty-HSV/ALL failure is explicit and never yields a complete bundle.
"""
from dataclasses import dataclass
import math
from typing import Callable, Sequence

import torch

from .plantcv_range import _channels


MAX_IMAGE_PIXELS = 64_000_000
PLANTCV_VERSION = '4.11.3'
PLANTCV_COLOR_SOURCE_SHA256 = '1a28a86a82af612f65ffd55a142df8d23673cd5ba80ec9a2fbfda10606f7b6b8'
MAX_LABELS = 1024
MAX_OUTPUT_BYTES = 128 * 1024 * 1024
TILE_PIXELS = 65_536
_METHOD = 'plantcv.plantcv.analyze.color'
_MODES = ('RGB', 'LAB', 'HSV')
_VARIABLES = {
    'RGB': ('blue_frequencies', 'green_frequencies', 'red_frequencies'),
    'LAB': ('lightness_frequencies', 'green-magenta_frequencies', 'blue-yellow_frequencies'),
    'HSV': ('hue_frequencies', 'saturation_frequencies', 'value_frequencies'),
}


class ColorAnalysisError(ValueError):
    pass


class ColorAnalysisCancelled(ColorAnalysisError):
    pass


class ColorAnalysisResourceError(ColorAnalysisError):
    pass


@dataclass(frozen=True, slots=True)
class ColorHistogram:
    variable: str
    trait: str
    method: str
    scale: str
    units: str
    datatype: str
    bin_labels: tuple[float, ...]
    values: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class ColorStatistic:
    variable: str
    trait: str
    method: str
    scale: str
    units: str
    datatype: str
    value: float
    available: bool
    unavailable_reason: str | None = None


@dataclass(frozen=True, slots=True)
class ColorLabelResult:
    label_id: int
    sample: str
    pixel_count: int
    histograms: tuple[ColorHistogram, ...]
    statistics: tuple[ColorStatistic, ...]


@dataclass(frozen=True, slots=True)
class ColorAnalysisResult:
    status: str
    colorspace: str
    labels: tuple[ColorLabelResult, ...]
    error_type: str | None = None
    error: str | None = None
    failed_sample: str | None = None


def _cancel(cancel):
    if cancel is None:
        return
    checker = cancel if callable(cancel) else getattr(cancel, 'is_set', None)
    if not callable(checker):
        raise ColorAnalysisError('cancel must be callable or provide is_set()')
    if checker():
        raise ColorAnalysisCancelled('Colour analysis cancelled before a complete result.')


def _weighted_median(counts: Sequence[int]) -> float:
    total = sum(counts)
    if not total:
        return math.nan
    lower_target, upper_target = (total-1)//2, total//2
    cumulative, lower = 0, None
    for value, count in enumerate(counts):
        cumulative += int(count)
        if lower is None and cumulative > lower_target:
            lower = value
        if cumulative > upper_target:
            return (lower+value)/2.0
    raise AssertionError('Histogram count did not reach its median.')


def _hue_statistics(hue_counts):
    counts = [0, *hue_counts[1:180]]
    total = sum(counts)
    if not total:
        return math.nan, math.nan, math.nan
    factor = 2.0 * math.pi / 179.0
    sine = math.fsum(count * math.sin(value*factor) for value, count in enumerate(counts) if count)
    cosine = math.fsum(count * math.cos(value*factor) for value, count in enumerate(counts) if count)
    angle = math.atan2(sine, cosine) % (2.0*math.pi)
    mean = angle * 179.0 / (2.0*math.pi) * 2.0
    resultant = min(1.0, math.hypot(sine/total, cosine/total))
    deviation = (math.inf if resultant == 0 else
        math.sqrt(-2.0*math.log(resultant)) * 179.0 / (2.0*math.pi) * 2.0)
    return mean, deviation, _weighted_median(counts)*2.0


def _positive_byte_statistics(counts):
    counts = [0, *counts[1:]]
    total = sum(counts)
    if not total:
        return math.nan, math.nan
    # Mean of positive byte/255 values, matching the declared source scale.
    mean = math.fsum(value/255.0 * count for value, count in enumerate(counts) if count) / total * 100.0
    median_byte = _weighted_median(counts)
    # For even populations NumPy averages already-normalized middle samples.
    lower, upper = math.floor(median_byte), math.ceil(median_byte)
    median = (lower/255.0 + upper/255.0)/2.0 * 100.0
    return mean, median


def _statistic(variable, value, scale):
    trait = ('hue circular standard deviation' if variable == 'hue_circular_std'
             else variable.replace('_', ' '))
    reason = None if math.isfinite(value) else ('source statistic is NaN' if math.isnan(value)
                                               else 'source statistic is infinite')
    return ColorStatistic(variable, trait, _METHOD, scale, scale, "<class 'float'>",
        value, math.isfinite(value), reason)


def analyze_color_cuda(bgr_image, labeled_mask, n_labels=1, colorspaces='hsv', label=None,
                       *, sample_label='default', cancel=None):
    """Compute exact byte histograms and numerically compared compact statistics.

    Labels 1..n_labels follow the original helper, including its two-unique-value
    mask remap: only 255 becomes 1 and every other value becomes zero. All-255
    singleton masks are not remapped. None uses explicitly supplied sample_label;
    callers can pass PlantCV's current global sample label. Empty RGB/LAB is a
    valid zero histogram plus NaN hue statistics; empty HSV/ALL returns the public
    UnboundLocalError state without presenting partial observations as complete.
    """
    _cancel(cancel)
    if (not torch.is_tensor(bgr_image) or not torch.is_tensor(labeled_mask)
            or bgr_image.layout != torch.strided or labeled_mask.layout != torch.strided
            or bgr_image.device.type != 'cuda' or labeled_mask.device != bgr_image.device
            or bgr_image.dtype != torch.uint8 or labeled_mask.dtype not in (torch.uint8, torch.int32)
            or bgr_image.ndim != 3 or bgr_image.shape[-1] != 3
            or tuple(labeled_mask.shape) != tuple(bgr_image.shape[:2])
            or not bgr_image.is_contiguous() or not labeled_mask.is_contiguous()):
        raise ColorAnalysisError('Require contiguous CUDA HxWx3 uint8 BGR and same-device HxW uint8/int32 labels.')
    if type(n_labels) is not int or not 1 <= n_labels <= MAX_LABELS:
        raise ColorAnalysisResourceError(f'n_labels must be 1..{MAX_LABELS}')
    if not isinstance(colorspaces, str) or colorspaces.lower() not in ('rgb', 'lab', 'hsv', 'all'):
        raise ColorAnalysisError('colorspaces must be rgb, lab, hsv or all')
    colorspaces = colorspaces.lower()
    pixel_total = labeled_mask.numel()
    if not 1 <= pixel_total <= MAX_IMAGE_PIXELS:
        raise ColorAnalysisResourceError('Image exceeds its nonempty raster limit.')
    if n_labels*9*256*32 > MAX_OUTPUT_BYTES:
        raise ColorAnalysisResourceError('Histogram output exceeds its byte limit.')
    if not isinstance(sample_label, str):
        raise ColorAnalysisError('sample_label must be a string')
    names = sample_label if label is None else label
    if isinstance(names, str):
        names = (names,) * n_labels
    elif not isinstance(names, Sequence) or len(names) != n_labels or any(not isinstance(v, str) for v in names):
        raise ColorAnalysisError('label must be a string or one string per label ID')
    names = tuple(f'{value}_{i}' for i, value in enumerate(names, 1))
    pixels, masks = bgr_image.reshape(-1, 3), labeled_mask.reshape(-1)
    minimum = torch.full((), torch.iinfo(labeled_mask.dtype).max, dtype=labeled_mask.dtype, device=bgr_image.device)
    maximum = torch.full((), torch.iinfo(labeled_mask.dtype).min, dtype=labeled_mask.dtype, device=bgr_image.device)
    for start in range(0, pixel_total, TILE_PIXELS):
        _cancel(cancel)
        tile = masks[start:start+TILE_PIXELS]
        minimum = torch.minimum(minimum, tile.min())
        maximum = torch.maximum(maximum, tile.max())
    low, high = int(minimum.item()), int(maximum.item())
    remap = low != high and high == 255
    if remap:
        for start in range(0, pixel_total, TILE_PIXELS):
            _cancel(cancel)
            tile = masks[start:start+TILE_PIXELS]
            if not bool(((tile == low) | (tile == high)).all().item()):
                remap = False
                break
    histograms = {mode: torch.zeros((n_labels+1, 3, 256), dtype=torch.int64, device=bgr_image.device)
                  for mode in _MODES}
    counts = torch.zeros(n_labels+1, dtype=torch.int64, device=bgr_image.device)
    channels = torch.arange(3, dtype=torch.int64, device=bgr_image.device)[:, None]
    for start in range(0, pixel_total, TILE_PIXELS):
        _cancel(cancel)
        ids = masks[start:start+TILE_PIXELS].to(torch.int64)
        if remap:
            ids = torch.where(ids == 255, 1, 0)
        selected = (ids > 0) & (ids <= n_labels)
        ids = ids[selected]
        counts += torch.bincount(ids, minlength=n_labels+1)
        source = pixels[start:start+TILE_PIXELS]
        for mode in _MODES:
            values = _channels(source, mode)[selected].transpose(0, 1).to(torch.int64)
            keys = ((channels*(n_labels+1)+ids[None, :])*256+values).reshape(-1)
            histogram = torch.bincount(keys, minlength=3*(n_labels+1)*256)
            histograms[mode] += histogram.reshape(3, n_labels+1, 256).transpose(0, 1)
    _cancel(cancel)
    compact = {mode: value[1:].cpu().tolist() for mode, value in histograms.items()}
    counts = counts[1:].cpu().tolist()
    _cancel(cancel)
    if colorspaces in ('hsv', 'all'):
        for index, count in enumerate(counts):
            if count == 0:
                return ColorAnalysisResult('reference_error', colorspaces, (), 'UnboundLocalError',
                    'Pinned PlantCV reads unbound s/v for an empty selected HSV/ALL mask.', names[index])
    byte_bins = tuple(range(256))
    percent_bins = tuple(round(i/255.0*100.0, 2) for i in range(256))
    opponent_bins = tuple(range(-128, 128))
    hue_bins = tuple(range(1, 360, 2))
    output = []
    for index, (sample, count) in enumerate(zip(names, counts)):
        _cancel(cancel)
        items, statistics = [], []
        for mode in _MODES:
            if colorspaces != 'all' and mode.lower() != colorspaces:
                continue
            for channel, variable in enumerate(_VARIABLES[mode]):
                values = compact[mode][index][channel]
                bins = byte_bins
                if mode == 'HSV' and channel == 0:
                    values, bins = values[:180], hue_bins
                elif mode == 'LAB' and channel > 0:
                    bins = opponent_bins
                elif mode in ('HSV', 'LAB'):
                    bins = percent_bins
                percentages = tuple(value/count*100.0 for value in values) if count else (0.0,)*len(values)
                items.append(ColorHistogram(variable, variable.replace('_', ' '), _METHOD, 'frequency',
                    'percent of selected pixels', "<class 'list'>", tuple(bins), percentages))
        hsv = compact['HSV'][index]
        if colorspaces in ('hsv', 'all'):
            for name, values in (('saturation', hsv[1]), ('value', hsv[2])):
                mean, median = _positive_byte_statistics(values)
                statistics.extend((_statistic(name+'_mean', mean, 'percent'),
                                   _statistic(name+'_median', median, 'percent')))
        mean, deviation, median = _hue_statistics(hsv[0])
        statistics.extend((_statistic('hue_circular_mean', mean, 'degrees'),
                           _statistic('hue_circular_std', deviation, 'degrees'),
                           _statistic('hue_median', median, 'degrees')))
        output.append(ColorLabelResult(index+1, sample, count, tuple(items), tuple(statistics)))
    _cancel(cancel)
    return ColorAnalysisResult('available', colorspaces, tuple(output))
