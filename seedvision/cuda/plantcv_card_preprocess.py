"""Bounded CUDA preprocessing for PlantCV 4.11.3 automatic colour cards.

The detector uses BGR2GRAY, an 11-tap byte Gaussian with REFLECT_101 borders,
then inverse adaptive mean/Gaussian thresholding with replicated borders and
offset two. Only compact, SHA-verified OpenCV kernel metadata is computed on
CPU. Full rasters and filter intermediates remain on the input CUDA device.

The byte filter follows OpenCV 4.11.0 smooth.dispatch.cpp/smooth.simd.hpp:
two nonnegative fixed-point axes sum to 256, followed by half-up shift 16.
Adaptive Gaussian uses CV_32F taps/intermediates. Larger kernels return only
when an outward interval certifies the final binary predicate, including
nearest-even byte conversion. See docs/OPENCV_MATH_NOTICES.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import math
import struct

import cv2
import torch

from .plantcv_card_kernel_pins import GAUSSIAN_KERNEL_SHA256
from . import tensor_callback
from .plantcv_range import _gray_from_bgr
from .plantcv_thresholds import (
    AdaptiveMeanPrecisionError, _adaptive_mean_condition, _replicated_box_sum,
)

MAX_IMAGE_PIXELS = 64 * 1024 * 1024
MAX_RETAINED_BYTES = 512 * 1024 * 1024
MAX_SCRATCH_BYTES = 64 * 1024 * 1024
MAX_FILTER_WORK = 1_000_000_000
MAX_TILE_SIDE = 256
BYTE11_WEIGHTS = (2, 7, 17, 31, 45, 52, 45, 31, 17, 7, 2)


class ColorCardPreprocessError(ValueError):
    """Invalid input, unsupported reference metadata or a resource limit."""


class ColorCardPreprocessCancelled(RuntimeError):
    """Cancellation with no published partial result."""


class ColorCardPreprocessPrecisionError(ColorCardPreprocessError):
    """The pinned float32 binary decision cannot be certified."""


class ColorCardPreprocessMutation(ColorCardPreprocessError,tensor_callback.TensorCallbackMutation):
    """A callback changed an input or live private filter buffer."""


@dataclass(frozen=True)
class ColorCardPreprocessing:
    gray: torch.Tensor
    blurred: torch.Tensor
    threshold: torch.Tensor
    adaptive_method: int
    block_size: int


def _validate_source(image, *, bgr=False):
    if (not torch.is_tensor(image) or image.layout != torch.strided
            or image.device.type != 'cuda' or image.dtype != torch.uint8
            or not image.is_contiguous()):
        raise ColorCardPreprocessError('Require contiguous strided CUDA uint8 input')
    if (bgr and (image.ndim != 3 or image.shape[-1] != 3)) or (not bgr and image.ndim != 2):
        raise ColorCardPreprocessError('Require HxWx3 BGR' if bgr else 'Require HxW gray')
    height, width = map(int, image.shape[:2])
    if min(height, width) < 1 or height * width > MAX_IMAGE_PIXELS:
        raise ColorCardPreprocessError('Image pixel count exceeds the supported bound')
    return height, width


def _signature(image):
    try:
        return tensor_callback._signature(image)
    except RuntimeError as error:
        raise ColorCardPreprocessError('Input must retain a mutation version counter') from error


def _guard(image, signature, cancel,private=()):
    def intact():
        if _signature(image)!=signature:
            raise ColorCardPreprocessMutation('Input changed during colour-card preprocessing')
    intact()
    try:
        try:cancelled=tensor_callback.checked_cancel(cancel,(image,*private))
        except tensor_callback.TensorCallbackMutation as error:
            raise ColorCardPreprocessMutation(str(error)) from error
    finally:intact()
    if cancelled:
        raise ColorCardPreprocessCancelled('Colour-card preprocessing cancelled')
    return False


def _parameters(method, size, tile_side, cancel):
    if cancel is not None and not callable(cancel):
        raise TypeError('cancel must be a zero-argument callable or None')
    if type(method) is not int or method not in (0, 1):
        raise ColorCardPreprocessError('adaptive_method must be 0 (mean) or 1 (Gaussian)')
    if type(size) is not int or size < 3 or size > 255 or size % 2 != 1:
        raise ColorCardPreprocessError('block_size must be odd in [3,255]')
    if type(tile_side) is not int or not 1 <= tile_side <= MAX_TILE_SIDE:
        raise ColorCardPreprocessError(f'tile_side must be in [1,{MAX_TILE_SIDE}]')


def _filter_work(height, width, size, tile_side, *, mean=False):
    radius = size // 2
    tile_rows = (height + tile_side - 1) // tile_side
    tile_columns = (width + tile_side - 1) // tile_side
    gathered = (height + 2 * radius * tile_rows) * (width + 2 * radius * tile_columns)
    if mean:
        # Prefix scans, gathers and border corrections in both box passes.
        return 32 * gathered
    return gathered + size * ((height + 2 * radius * tile_rows) * width + height * width)


def _scratch_bytes(height, width, size, tile_side):
    h, w = min(height, tile_side), min(width, tile_side)
    halo = (h + size - 1) * (w + size - 1)
    # Includes gather/casts, two prefix/filter passes, independent products,
    # endpoint/rounding/boolean temporaries, index vectors and allocator slack.
    return 80 * halo + 128 * h * w + 2 * 1024 * 1024


def _preflight(height, width, size, tile_side, *, method=1, complete=False):
    pixels = height * width
    retained = pixels * (6 if complete else 2)
    work = _filter_work(height, width, size, tile_side, mean=method == 0)
    scratch = _scratch_bytes(height, width, size, tile_side)
    if complete:
        work += _filter_work(height, width, 11, tile_side) + pixels
        scratch = max(scratch, _scratch_bytes(height, width, 11, tile_side))
    if retained > MAX_RETAINED_BYTES or work > MAX_FILTER_WORK:
        raise ColorCardPreprocessError('Colour-card retained storage or filter work exceeds its bound')
    if scratch > MAX_SCRATCH_BYTES:
        raise ColorCardPreprocessError('Colour-card tile exceeds its scratch bound')
    return work, scratch


def _weights(size):
    values = cv2.getGaussianKernel(size, 0, cv2.CV_32F).reshape(-1)
    raw = values.astype('<f4').tobytes()
    if hashlib.sha256(raw).hexdigest() != GAUSSIAN_KERNEL_SHA256[size]:
        raise ColorCardPreprocessError('Loaded OpenCV Gaussian taps differ from pinned 4.11.0')
    return struct.unpack('<' + 'f' * size, raw)


def _round_up(value):
    result = float(value)
    if Fraction.from_float(result) < value:
        result = math.nextafter(result, math.inf)
    return result


def _gaussian_error(weights):
    # Exact rational bound construction avoids a rounded-down bound. The source
    # symmetric scalar/SIMD reductions use fewer than 4*k rounded operations per
    # axis. Native separate multiply/add uses fewer than 2*k binary64 operations.
    # All taps and pixels are nonnegative, normal finite values; float32 taps are
    # exactly representable as binary64. No intermediate byte conversion occurs.
    size = len(weights)
    total = sum((Fraction.from_float(w) for w in weights), Fraction())
    def gamma(operations, unit):
        return operations * unit / (1 - operations * unit)
    g32 = gamma(4 * size, Fraction(1, 1 << 24))
    g64 = gamma(2 * size, Fraction(1, 1 << 53))
    return _round_up(255 * total * total * (2*g32 + g32*g32 + 2*g64 + g64*g64))


def _indices(start, end, radius, length, device, *, reflect=False):
    index = torch.arange(start-radius, end+radius, dtype=torch.int64, device=device)
    if not reflect:
        return index.clamp_(0, length-1)
    if length == 1:
        return index.zero_()
    period = 2 * (length-1)
    folded = index.remainder_(period)
    return torch.where(folded < length, folded, period-folded)


def _separable(tile, weights, height, width, *, integer=False, guard):
    radius = len(weights) // 2
    dtype = torch.int32 if integer else torch.float64
    horizontal = torch.zeros((height+2*radius, width), dtype=dtype, device=tile.device)
    for offset, weight in enumerate(weights):
        guard(tile,horizontal)
        # Keep multiplication and addition separate; release each product.
        product = tile[:, offset:offset+width].to(dtype) * weight
        horizontal.add_(product)
        del product
    del tile
    vertical = torch.zeros((height, width), dtype=dtype, device=horizontal.device)
    for offset, weight in enumerate(weights):
        guard(horizontal,vertical)
        product = horizontal[offset:offset+height] * weight
        vertical.add_(product)
        del product
    return vertical


def gaussian_blur11_cuda(gray, *, cancel=None, tile_side=MAX_TILE_SIDE):
    """Exact pinned uint8 GaussianBlur(11,11), sigma0, BORDER_REFLECT_101."""
    height, width = _validate_source(gray)
    _parameters(1, 11, tile_side, cancel)
    signature = _signature(gray)
    guard = lambda *private: _guard(gray, signature, cancel,private)
    guard()
    _preflight(height, width, 11, tile_side)
    guard()
    output = torch.empty_like(gray)
    for top in range(0, height, tile_side):
        for left in range(0, width, tile_side):
            guard(output)
            bottom, right = min(height, top+tile_side), min(width, left+tile_side)
            h, w = bottom-top, right-left
            rows = _indices(top, bottom, 5, height, gray.device, reflect=True)
            columns = _indices(left, right, 5, width, gray.device, reflect=True)
            tile = gray[rows[:, None], columns[None, :]]
            numerator = _separable(tile, BYTE11_WEIGHTS, h, w, integer=True,
                guard=lambda *private:guard(output,rows,columns,tile,*private))
            # 255*256**2 + 32768 < 2**24 < int32 maximum.
            output[top:bottom, left:right] = ((numerator+32768) >> 16).to(torch.uint8)
            guard(output,tile,numerator,rows,columns)
            del tile, numerator, rows, columns
    guard(output)
    return output


def _certified_decision(estimate, source, error):
    if error == 0:
        return source <= estimate.round().clamp_(0, 255).sub_(2)
    lower = estimate - error
    upper = estimate + error
    negative = torch.tensor(-math.inf, dtype=torch.float64, device=estimate.device)
    positive = torch.tensor(math.inf, dtype=torch.float64, device=estimate.device)
    lower.nextafter_(negative).round_().clamp_(0, 255).sub_(2)
    upper.nextafter_(positive).round_().clamp_(0, 255).sub_(2)
    low_decision, high_decision = source <= lower, source <= upper
    if bool((low_decision != high_decision).any().item()):
        raise ColorCardPreprocessPrecisionError('Adaptive Gaussian binary decision is uncertain')
    return low_decision


def adaptive_card_threshold_cuda(gray, adaptive_method=1, block_size=51, *,
                                 cancel=None, tile_side=MAX_TILE_SIDE):
    """Inverse replicated adaptive threshold, fixed offset2, odd kernels3..255.

    For non-dyadic Gaussian kernels an interval must certify every binary
    decision. Uncertain decisions withhold the whole mask; they are not rounded
    heuristically. tile_side is an internal execution bound, not a node control.
    """
    height, width = _validate_source(gray)
    _parameters(adaptive_method, block_size, tile_side, cancel)
    signature = _signature(gray)
    guard = lambda *private: _guard(gray, signature, cancel,private)
    guard()
    _preflight(height, width, block_size, tile_side, method=adaptive_method)
    weights = _weights(block_size) if adaptive_method == 1 else None
    # For k<=9 the dyadic source numerator is below2**24, so both passes
    # and native binary64 arithmetic are exact, including half-byte ties.
    error = _gaussian_error(weights) if adaptive_method == 1 and block_size > 9 else 0
    guard()
    output = torch.empty_like(gray)
    radius = block_size // 2
    for top in range(0, height, tile_side):
        for left in range(0, width, tile_side):
            guard(output)
            bottom, right = min(height, top+tile_side), min(width, left+tile_side)
            h, w = bottom-top, right-left
            rows = _indices(top, bottom, radius, height, gray.device)
            columns = _indices(left, right, radius, width, gray.device)
            tile = gray[rows[:, None], columns[None, :]]
            if adaptive_method == 0:
                horizontal = _replicated_box_sum(tile, 1, radius)[:, radius:radius+w].contiguous()
                guard(output,rows,columns,tile,horizontal)
                sums = _replicated_box_sum(horizontal, 0, radius)[radius:radius+h]
                try:
                    selected = _adaptive_mean_condition(sums, gray[top:bottom, left:right],
                        block_size, 2, 'dark', frame_shape=(height,width))
                except AdaptiveMeanPrecisionError as error:
                    raise ColorCardPreprocessPrecisionError(str(error)) from error
                estimate = None
                del horizontal, sums
            else:
                estimate = _separable(tile, weights, h, w,
                    guard=lambda *private:guard(output,rows,columns,tile,*private))
                selected = _certified_decision(estimate, gray[top:bottom, left:right], error)
            output[top:bottom, left:right] = selected.to(torch.uint8)*255
            guard(output,tile,estimate,selected,rows,columns)
            del tile, estimate, selected, rows, columns
    guard(output)
    return output


def preprocess_color_card_cuda(bgr, adaptive_method=1, block_size=51, *,
                               cancel=None, tile_side=MAX_TILE_SIDE):
    """Return reusable gray, byte-blurred and threshold rasters on CUDA."""
    height, width = _validate_source(bgr, bgr=True)
    _parameters(adaptive_method, block_size, tile_side, cancel)
    signature = _signature(bgr)
    guard = lambda *private: _guard(bgr, signature, cancel,private)
    guard()
    _preflight(height, width, block_size, tile_side, method=adaptive_method, complete=True)
    guard()
    gray = torch.empty((height, width), dtype=torch.uint8, device=bgr.device)
    flat, target = bgr.reshape(-1, 3), gray.reshape(-1)
    for start in range(0, height*width, tile_side*tile_side):
        guard(gray)
        end = min(height*width, start+tile_side*tile_side)
        target[start:end] = _gray_from_bgr(flat[start:end])
    blurred = gaussian_blur11_cuda(gray, cancel=lambda:guard(gray), tile_side=tile_side)
    threshold = adaptive_card_threshold_cuda(blurred, adaptive_method, block_size,
        cancel=lambda:guard(gray,blurred), tile_side=tile_side)
    guard(gray,blurred,threshold)
    return ColorCardPreprocessing(gray, blurred, threshold, adaptive_method, block_size)
