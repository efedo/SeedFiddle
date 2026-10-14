"""Bounded CUDA ports of selected PlantCV 4.11.3 extended thresholds.

Gaussian uses OpenCV's exact dyadic 3/5/7/9-tap default kernels and replicated
borders. Kernels 11/13/15 use pinned float32 coefficients with a conservative
rounding guard; uncertain masks are withheld. Dual-channel thresholding retains
the pinned BGR storage slots, CMYK C/Y naming and strict line comparisons.
Full rasters and tiled scratch remain on CUDA; no image transfer is required.
Sources and retained OpenCV notices: docs/OPENCV_MATH_NOTICES.md.
"""
from __future__ import annotations

import math
import struct
import warnings

import torch

from .plantcv_range import _channels
from .plantcv_thresholds import (
    PLANTCV_VERSION, PLANTCV_COMMIT, PLANTCV_THRESHOLD_SOURCE_SHA256,
    _object_type, _validate_gray, _validate_mean_parameters, threshold_otsu,
)


MAX_RASTER_PIXELS = 64_000_000
TILE_PIXELS = 65_536
MAX_POINT_MAGNITUDE = 1_000_000.0
MAX_LINE_COEFFICIENT = 1e16
MAX_TRIANGLE_PIXELS = 1 << 24
MAX_TEXTURE_PIXELS = 4_000_000
MAX_TEXTURE_SCRATCH_BYTES = 128 * 1024 * 1024
TEXTURE_PROPERTIES = ('contrast', 'dissimilarity', 'homogeneity', 'ASM', 'energy', 'correlation')
TEXTURE_BORDERS = ('constant', 'nearest', 'reflect', 'mirror', 'wrap')
TEXTURE_ERROR_BOUNDS = {
    'contrast': 1e-5, 'dissimilarity': 1e-7, 'homogeneity': 1e-8,
    'ASM': 1e-8, 'energy': 1e-7, 'correlation': 1e-2,
}
GAUSSIAN_WEIGHTS = {
    3: (1, 2, 1),
    5: (1, 4, 6, 4, 1),
    7: (2, 7, 14, 18, 14, 7, 2),
    9: (4, 13, 30, 51, 60, 51, 30, 13, 4),
}
GAUSSIAN_FLOAT32_KERNEL_BITS = {
    11: (0x3c10612b, 0x3cde5c35, 0x3d855a85, 0x3df92326, 0x3e353f0f, 0x3e4d6105,
         0x3e353f0f, 0x3df92326, 0x3d855a85, 0x3cde5c35, 0x3c10612b),
    13: (0x3bbe0483, 0x3c865c11, 0x3d1d47f4, 0x3d986692, 0x3df478f5, 0x3e224f6b,
         0x3e32665d, 0x3e224f6b, 0x3df478f5, 0x3d986692, 0x3d1d47f4, 0x3c865c11, 0x3bbe0483),
    15: (0x3b869623, 0x3c300526, 0x3cc68dcb, 0x3d412caa, 0x3da218f7, 0x3deaa1ed,
         0x3e12764e, 0x3e1db4b1, 0x3e12764e, 0x3deaa1ed, 0x3da218f7, 0x3d412caa,
         0x3cc68dcb, 0x3c300526, 0x3b869623),
}
MAX_GAUSSIAN_FLOAT_SCRATCH_BYTES = 64 * 1024 * 1024
MAX_GAUSSIAN_FLOAT_WORK = 1_000_000_000
CHANNELS = {
    'R': ('RGB', 2), 'G': ('RGB', 1), 'B': ('RGB', 0),
    'l': ('LAB', 0), 'a': ('LAB', 1), 'b': ('LAB', 2),
    'h': ('HSV', 0), 's': ('HSV', 1), 'v': ('HSV', 2),
    'c': ('CMYK', 0), 'm': ('CMYK', 1), 'y': ('CMYK', 2),
    'k': ('CMYK', 3), 'gray': ('GRAY', 0), 'index': ('INDEX', 0),
}


class ExtendedThresholdError(ValueError):
    pass


class ExtendedThresholdCancelled(RuntimeError):
    pass


class ExtendedThresholdPrecisionError(ExtendedThresholdError):
    """The bounded CUDA calculation cannot prove the public uint8 mask decision."""


class TrianglePreambleError(ExtendedThresholdError):
    """The pinned triangle implementation fails before its final Otsu call."""


def _cancel(cancel):
    if cancel is not None:
        if not callable(cancel):
            raise TypeError('cancel must be a zero-argument callable or None')
        if cancel():
            raise ExtendedThresholdCancelled('Extended threshold cancelled before a complete output.')


def _pixel_limit(count):
    if not 1 <= count <= MAX_RASTER_PIXELS:
        raise ExtendedThresholdError(f'Raster must contain 1..{MAX_RASTER_PIXELS} pixels.')


def _gaussian_admission(pixels, ksize):
    """Allocation-free resource facts, shared by execution and saved replay."""
    _pixel_limit(pixels)
    samples = pixels*ksize*ksize
    floating = ksize in GAUSSIAN_FLOAT32_KERNEL_BITS
    if floating:
        if samples > MAX_GAUSSIAN_FLOAT_WORK:
            raise ExtendedThresholdError('Large Gaussian threshold exceeds its bounded work limit.')
        scratch_per_pixel = 25*ksize*ksize + 64*ksize + 128
        tile = min(TILE_PIXELS, MAX_GAUSSIAN_FLOAT_SCRATCH_BYTES//scratch_per_pixel)
        if tile < 1:
            raise ExtendedThresholdError('Large Gaussian threshold exceeds its scratch-memory limit.')
    else:
        # Integer path keeps scalar tile vectors, never a window-value plane.
        scratch_per_pixel, tile = 128, TILE_PIXELS
    return {'mode':'guarded_float32' if floating else 'exact_dyadic',
        'window_sample_visits':samples, 'tile_pixels':tile,
        'scratch_per_pixel':scratch_per_pixel,
        'tile_scratch_bytes':min(pixels,tile)*scratch_per_pixel,
        'fixed_scratch_bytes':65_536,
        'window_sample_limit':MAX_GAUSSIAN_FLOAT_WORK if floating else None,
        'pixel_limit':MAX_RASTER_PIXELS}


def _texture_admission(pixels, ksize, offset):
    """Share the exact pair/scratch/tile domain without allocating a raster."""
    if not 1 <= pixels <= MAX_TEXTURE_PIXELS:
        raise ExtendedThresholdError(f'Texture raster exceeds {MAX_TEXTURE_PIXELS} pixels.')
    pairs = ksize*(ksize-offset)
    oriented = 2*pairs
    scratch_per_pixel = 96*oriented + oriented*oriented + 1024
    tile = min(TILE_PIXELS, MAX_TEXTURE_SCRATCH_BYTES//scratch_per_pixel)
    if tile < 1:
        raise ExtendedThresholdError('Texture kernel exceeds the scratch-memory budget.')
    return {'pair_count':pairs, 'oriented_pairs':oriented,
        'pixel_pair_visits':pixels*pairs, 'tile_pixels':tile,
        'scratch_per_pixel':scratch_per_pixel,
        'tile_scratch_bytes':min(pixels,tile)*scratch_per_pixel,
        'fixed_scratch_bytes':65_536, 'pixel_limit':MAX_TEXTURE_PIXELS,
        'tile_scratch_limit_bytes':MAX_TEXTURE_SCRATCH_BYTES,
        'sorting_scope':'per pixel, at most oriented_pairs keys and two searches per key'}


def _dual_channel_admission(pixels, x_channel, y_channel):
    """Conservative tiled conversion allowance, including both selected modes."""
    _pixel_limit(pixels)
    modes = tuple(dict.fromkeys(CHANNELS[name][0] for name in (x_channel,y_channel)))
    # Reserves include conversion inputs, intermediates and retained outputs.
    # Charge both modes in full although only one conversion executes at a time.
    conversion = {'RGB':0, 'GRAY':64, 'LAB':192, 'HSV':160, 'CMYK':256, 'INDEX':8}
    scratch_per_pixel = 64 + sum(conversion[mode] for mode in modes)
    return {'conversion_modes':modes, 'tile_pixels':TILE_PIXELS,
        'scratch_per_pixel':scratch_per_pixel,
        'tile_scratch_bytes':min(pixels,TILE_PIXELS)*scratch_per_pixel,
        'fixed_scratch_bytes':65_536, 'pixel_limit':MAX_RASTER_PIXELS,
        'channel_conversion_rule':'one RGB-to-BGR dispatch boundary, then at most two distinct tiled modes'}


def _texture_coordinate(index, length, mode):
    """Map SciPy ndimage border modes without transferring raster coordinates."""
    if mode == 'constant':
        return index.clamp(0, length-1), (index >= 0) & (index < length)
    if mode == 'nearest':
        return index.clamp(0, length-1), None
    if mode == 'wrap':
        return torch.remainder(index, length), None
    if mode == 'reflect':
        folded = torch.remainder(index, 2*length)
        return torch.where(folded < length, folded, 2*length-1-folded), None
    if mode == 'mirror':
        if length == 1:
            return torch.zeros_like(index), None
        period = 2*(length-1)
        folded = torch.remainder(index, period)
        return torch.where(folded < length, folded, period-folded), None
    raise ExtendedThresholdError('Unsupported texture border mode')


def _gaussian_float32_weights(bits):
    return tuple(struct.unpack('<f', struct.pack('<I', value))[0] for value in bits)


def _gaussian_float32_error_bound(weights, ksize):
    # Two positive separable CV_32F sums. 4*k rounded operations per pass
    # bounds the source's symmetric scalar/SIMD pair, multiply and add ordering.
    # gamma_n=n*u/(1-n*u) propagates each pass's absolute rounding error.
    weight_sum = math.fsum(weights)
    operations = 4 * ksize
    gamma = (operations * 2.0**-24) / (1.0 - operations * 2.0**-24)
    # Binary64 coefficient products and reduction contribute <1.3e-11 for k<=15.
    return 255.0 * weight_sum**2 * (2.0*gamma + gamma**2) + 1e-9


def _threshold_gaussian_float32(gray, ksize, delta, polarity, cancel):
    pixels = gray.numel()
    tile_pixels = _gaussian_admission(pixels,ksize)['tile_pixels']
    host_weights = _gaussian_float32_weights(GAUSSIAN_FLOAT32_KERNEL_BITS[ksize])
    error = _gaussian_float32_error_bound(host_weights, ksize)
    bits = torch.tensor(GAUSSIAN_FLOAT32_KERNEL_BITS[ksize], dtype=torch.int32, device=gray.device)
    weights = bits.view(torch.float32).to(torch.float64)
    outer = weights[:, None] * weights[None, :]
    offsets = torch.arange(-(ksize//2), ksize//2+1, dtype=torch.int64, device=gray.device)
    height, width = gray.shape
    flat = gray.reshape(-1)
    output = torch.empty_like(gray)
    target = output.reshape(-1)
    for start in range(0, pixels, tile_pixels):
        _cancel(cancel)
        end = min(start+tile_pixels, pixels)
        positions = torch.arange(start, end, dtype=torch.int64, device=gray.device)
        rows = torch.div(positions, width, rounding_mode='floor')
        columns = positions.remainder(width)
        y_index = (rows[:, None] + offsets).clamp(0, height-1)
        x_index = (columns[:, None] + offsets).clamp(0, width-1)
        indices = y_index[:, :, None] * width + x_index[:, None, :]
        estimate = (flat[indices].to(torch.float64) * outer).sum(dim=(1, 2), dtype=torch.float64)
        del indices, y_index, x_index
        fraction = estimate - torch.floor(estimate)
        if bool((torch.abs(fraction-0.5) <= error).any().item()):
            raise ExtendedThresholdPrecisionError(
                'OpenCV float32 Gaussian mean may cross a uint8 rounding boundary.')
        local_threshold = torch.round(estimate).clamp_(0, 255).to(torch.float32).sub_(delta)
        source = flat[start:end]
        selected = source > local_threshold if polarity == 'light' else source <= local_threshold
        target[start:end] = selected.to(torch.uint8)*255
    _cancel(cancel)
    return output


def threshold_gaussian(gray, ksize, offset, object_type='light', *, cancel=None):
    """Pinned adaptive Gaussian threshold, CUDA uint8 HW, guarded kernels 3..15.

    Even integer requests round up to the next odd kernel, as in PlantCV.
    For effective kernels 3/5/7/9, the two-dimensional weighted numerator
    is an exact int32 sum below 2**24.
    Its dyadic float32 mean rounds to nearest even, matching OpenCV's CV_32F
    GaussianBlur -> uint8 conversion. Light uses ceil(offset) and strict greater;
    dark uses floor(offset) and less-or-equal. Kernels 11/13/15 return only if a
    float32 rounding-error interval proves the byte mean. Offsets are in [-255,255].
    """
    _cancel(cancel)
    gray = _validate_gray(gray)
    if not gray.is_contiguous():
        raise ExtendedThresholdError('gray must be contiguous')
    _pixel_limit(gray.numel())
    ksize, offset = _validate_mean_parameters(ksize, offset)
    polarity = _object_type(object_type)
    delta = math.ceil(offset) if polarity == 'light' else math.floor(offset)
    if ksize not in GAUSSIAN_WEIGHTS:
        if ksize not in GAUSSIAN_FLOAT32_KERNEL_BITS:
            raise ExtendedThresholdError('Effective Gaussian kernels are limited to odd sizes 3..15 (even requests round up).')
        return _threshold_gaussian_float32(gray, ksize, delta, polarity, cancel)
    weights = GAUSSIAN_WEIGHTS[ksize]
    tile_pixels = _gaussian_admission(gray.numel(),ksize)['tile_pixels']
    denominator = sum(weights)**2
    radius = ksize//2
    height, width = gray.shape
    flat = gray.reshape(-1)
    output = torch.empty_like(gray)
    target = output.reshape(-1)
    for start in range(0, gray.numel(), tile_pixels):
        _cancel(cancel)
        end = min(start+tile_pixels, gray.numel())
        positions = torch.arange(start, end, dtype=torch.int64, device=gray.device)
        rows = torch.div(positions, width, rounding_mode='floor')
        columns = positions.remainder(width)
        numerator = torch.zeros(end-start, dtype=torch.int32, device=gray.device)
        for y, wy in enumerate(weights):
            _cancel(cancel)
            row_base = (rows+y-radius).clamp(0, height-1)*width
            horizontal = torch.zeros_like(numerator)
            for x, wx in enumerate(weights):
                indices = row_base+(columns+x-radius).clamp(0, width-1)
                horizontal.add_(flat[indices].to(torch.int32), alpha=wx)
            numerator.add_(horizontal, alpha=wy)
        local_threshold = numerator.to(torch.float32).mul_(1.0/denominator).round_().sub_(delta)
        selected = flat[start:end] > local_threshold if polarity == 'light' else flat[start:end] <= local_threshold
        target[start:end] = selected.to(torch.uint8)*255
    _cancel(cancel)
    return output


def _line(points):
    if not isinstance(points, (list, tuple)) or len(points) < 2:
        raise ExtendedThresholdError('points must contain at least two coordinate pairs')
    selected = []
    for point in points[:2]:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ExtendedThresholdError('Each selected point must contain x and y')
        if any(type(value) not in (int, float) or not math.isfinite(value)
                or abs(value) > MAX_POINT_MAGNITUDE for value in point):
            raise ExtendedThresholdError('Point coordinates must be finite numbers within +/-1,000,000')
        selected.append(tuple(float(value) for value in point))
    (x0, y0), (x1, y1) = selected
    denominator = x1-x0+1e-10
    if denominator == 0:
        raise ExtendedThresholdError('Line denominator is zero after the reference epsilon')
    slope = (y1-y0)/denominator
    intercept = y0-slope*x0
    if not all(math.isfinite(value) and abs(value) <= MAX_LINE_COEFFICIENT for value in (slope, intercept)):
        raise ExtendedThresholdError('Line coefficients exceed the supported finite domain')
    if len(points) > 2:
        warnings.warn('Only the first two points are used, as in pinned PlantCV.', UserWarning, stacklevel=3)
    return slope, intercept


def threshold_dual_channels(bgr_image, x_channel, y_channel, points, above=True, *, cancel=None):
    """Pinned two-channel straight-line threshold for contiguous CUDA uint8 HWC.

    Names are case-sensitive: R/G/B, l/a/b, h/s/v, c/m/y/k, gray and index.
    `index` is the global row-major pixel index. The reference adds 1e-10 to the
    x difference even for vertical lines. Above/below are strict >/<; points exactly
    on the line are zero for both. Extra points are ignored with a warning.
    """
    _cancel(cancel)
    if (not torch.is_tensor(bgr_image) or bgr_image.layout != torch.strided
            or bgr_image.device.type != 'cuda' or bgr_image.dtype != torch.uint8
            or bgr_image.ndim != 3 or bgr_image.shape[-1] != 3 or not bgr_image.is_contiguous()):
        raise ExtendedThresholdError('Require a contiguous CUDA HxWx3 uint8 BGR image')
    _pixel_limit(bgr_image.numel()//3)
    if x_channel not in CHANNELS or y_channel not in CHANNELS:
        raise ExtendedThresholdError('Unsupported channel name or case')
    if type(above) is not bool:
        raise ExtendedThresholdError('above must be a bool')
    slope, intercept = _line(points)
    tile_pixels = _dual_channel_admission(bgr_image.numel()//3,x_channel,y_channel)['tile_pixels']
    pixels = bgr_image.reshape(-1, 3)
    output = torch.empty(bgr_image.shape[:2], dtype=torch.uint8, device=bgr_image.device)
    target = output.reshape(-1)
    for start in range(0, pixels.shape[0], tile_pixels):
        _cancel(cancel)
        end = min(start+tile_pixels, pixels.shape[0])
        converted = {}
        selected = []
        for name in (x_channel, y_channel):
            mode, slot = CHANNELS[name]
            if mode == 'INDEX':
                value = torch.arange(start, end, dtype=torch.float64, device=pixels.device)
            else:
                if mode not in converted:
                    converted[mode] = _channels(pixels[start:end], mode)
                value = converted[mode][:, slot].to(torch.float64)
            selected.append(value)
        # Separate operations retain NumPy's multiply then add, without FMA.
        y_line = selected[0]*slope
        y_line.add_(intercept)
        chosen = selected[1] > y_line if above else selected[1] < y_line
        target[start:end] = chosen.to(torch.uint8)*255
    _cancel(cancel)
    return output


def threshold_triangle(gray, object_type='light', *, cancel=None):
    """Pinned triangle masks at fixed xstep=1, within the Otsu supported domain.

    PlantCV 4.11.3 computes a triangle seed then passes THRESH_OTSU, which ignores
    that seed. No xstep mask control is exposed. Preserve its earlier failure:
    calcHist uses 256 bins over [0,255], excluding byte255; the active rising-edge
    peak detector needs at least one peak. A peak proves the first global maximum
    is below bin255, so the default-step distance loop is nonempty and finite.
    The existing Otsu primitive withholds ambiguous partitions. Full pixels stay
    on CUDA and only compact 256-bin counts transfer. The 2**24 pixel cap keeps
    the reference float32 histogram counts exact. Cancellation returns no mask.

    Source: threshold_methods.py, PLANTCV_THRESHOLD_SOURCE_SHA256, PlantCV4.11.3.
    """
    _cancel(cancel)
    gray = _validate_gray(gray)
    if not gray.is_contiguous():
        raise ExtendedThresholdError('gray must be contiguous')
    if gray.numel() > MAX_TRIANGLE_PIXELS:
        raise ExtendedThresholdError(f'Triangle histogram exceeds {MAX_TRIANGLE_PIXELS} pixels.')
    histogram = torch.bincount(gray.reshape(-1), minlength=256)
    _cancel(cancel)
    counts = [int(value) for value in histogram.detach().cpu().tolist()]
    counts[255] = 0
    peaks = [index for index in range(1, 256)
        if counts[index]-counts[index-1] > 0
        and (counts[index+1]-counts[index] if index < 255 else 0) <= 0]
    if not peaks:
        raise TrianglePreambleError('Pinned triangle preamble found no rising-edge histogram peak.')
    _cancel(cancel)
    # Source polarity validation occurs after its preamble; preserve that order.
    polarity = _object_type(object_type)
    mask = threshold_otsu(gray, polarity)
    _cancel(cancel)
    return mask


def threshold_texture(gray, ksize, threshold, offset=3,
                      texture_method='dissimilarity', borders='nearest', *, cancel=None):
    """Bounded binary-mask decisions for PlantCV 4.11.3 texture thresholding.

    Supports CUDA uint8 HW, kernels 3/5/7/9, integer offset in [1,ksize), byte
    thresholds, six documented properties and five SciPy border modes. The
    internal texture map is not claimed byte-equivalent. Indeterminate decisions
    raise ExtendedThresholdPrecisionError; no partial mask is returned.

    The pinned reference normalizes a symmetric GLCM twice in float64, computes
    graycoprops, casts to uint8, then applies strict THRESH_BINARY. There are
    P=ksize*(ksize-offset)<=72 pairs and M=2P<=144 symmetric observations. Bound
    dense reductions by gamma=n*u/(1-n*u), n=65535, u=2**-52: gamma<1.46e-11.
    An 8*gamma L1 allowance for normalization/reductions bounds contrast error
    by 65025*8gamma<1e-5, dissimilarity by 255*8gamma<1e-7, and homogeneity/ASM
    by 1e-8. ASM>=1/144 makes the energy sqrt derivative <=6 (bound 1e-7).
    Nonconstant integer marginals have variance >=143/144**2>0.00689; the
    corresponding moment/ratio allowance is below 0.01 for correlation.
    Decisions near threshold+1 or correlation endpoints +/-1 are withheld.
    Only a single diagonal GLCM cell proves an exact value of one through both
    reference normalizations; multiple diagonal cells do not prove that cast.
    Contrast >=256-1e-5 is outside the supported uint8 conversion domain.

    Oracle pins: PlantCV4.11.3 threshold_methods.py SHA256
    04a671632e7e4decf5a662779cd88473b5aff750554afeee98eabe02eaa49907;
    SciPy1.15.3 ndimage/_filters.py
    919dcbfd3cccf9c2b61210214d27c828a49529db75a6432f2d9c99442946ec65;
    skimage0.25.2 feature/texture.py
    0bc7c6ea025fa7d69cd23c5c66e34252eb0eda8ec503117406d542a4c93e50d7.
    """
    _cancel(cancel)
    gray = _validate_gray(gray)
    if not gray.is_contiguous():
        raise ExtendedThresholdError('gray must be contiguous')
    if gray.numel() > MAX_TEXTURE_PIXELS:
        raise ExtendedThresholdError(f'Texture raster exceeds {MAX_TEXTURE_PIXELS} pixels.')
    if type(ksize) is not int or ksize not in (3, 5, 7, 9):
        raise ExtendedThresholdError('Texture ksize must be one of 3, 5, 7 or 9.')
    if type(offset) is not int or not 1 <= offset < ksize:
        raise ExtendedThresholdError('Texture offset must be an integer in [1, ksize).')
    if type(threshold) is not int or not 0 <= threshold <= 255:
        raise ExtendedThresholdError('Texture threshold must be an integer byte.')
    if texture_method not in TEXTURE_PROPERTIES:
        raise ExtendedThresholdError('Unsupported documented texture property.')
    if borders not in TEXTURE_BORDERS:
        raise ExtendedThresholdError('Unsupported SciPy texture border mode.')

    height, width = gray.shape
    admission = _texture_admission(gray.numel(),ksize,offset)
    pair_count, oriented_pairs = admission['pair_count'], admission['oriented_pairs']
    # Include simultaneously live pair arrays, sorting/search buffers and
    # reductions. Retain an M*M allowance as conservative allocator headroom;
    # no quadratic equality plane or dtype-conversion plane is constructed.
    tile_pixels = admission['tile_pixels']
    _cancel(cancel)
    if ((texture_method in ('homogeneity', 'ASM', 'energy') and threshold >= 1)
            or (texture_method == 'correlation' and threshold == 255)):
        result = torch.zeros_like(gray)
        _cancel(cancel)
        return result

    flat = gray.reshape(-1)
    output = torch.empty_like(gray)
    target = output.reshape(-1)
    radius = ksize//2
    cutoff = float(threshold+1)
    error_bound = TEXTURE_ERROR_BOUNDS[texture_method]
    for start in range(0, gray.numel(), tile_pixels):
        _cancel(cancel)
        end = min(start+tile_pixels, gray.numel())
        positions = torch.arange(start, end, dtype=torch.int64, device=gray.device)
        rows = torch.div(positions, width, rounding_mode='floor')
        columns = positions.remainder(width)
        left_columns, right_columns = [], []
        for dy in range(ksize):
            _cancel(cancel)
            row_index, row_valid = _texture_coordinate(rows+dy-radius, height, borders)
            for dx in range(ksize-offset):
                col_left, valid_left = _texture_coordinate(columns+dx-radius, width, borders)
                col_right, valid_right = _texture_coordinate(columns+dx+offset-radius, width, borders)
                left = flat[row_index*width+col_left]
                right = flat[row_index*width+col_right]
                if borders == 'constant':
                    left = torch.where(row_valid & valid_left, left, 0)
                    right = torch.where(row_valid & valid_right, right, 0)
                left_columns.append(left)
                right_columns.append(right)
        left = torch.stack(left_columns, dim=1)
        right = torch.stack(right_columns, dim=1)
        single_cell = (left == right).all(dim=1) & (left == left[:, :1]).all(dim=1)
        left_i, right_i = left.to(torch.int32), right.to(torch.int32)
        delta = left_i-right_i
        squared = delta.to(torch.float64).square()
        if texture_method == 'contrast':
            values = squared.mean(dim=1)
            if bool((values >= 256.0-error_bound).any().item()):
                raise ExtendedThresholdPrecisionError('Contrast reaches the unsupported uint8 conversion range.')
        elif texture_method == 'dissimilarity':
            values = delta.abs().to(torch.float64).mean(dim=1)
        elif texture_method == 'homogeneity':
            values = torch.reciprocal(1.0+squared).mean(dim=1)
        elif texture_method in ('ASM', 'energy'):
            keys = torch.cat((left_i*256+right_i, right_i*256+left_i), dim=1)
            ordered = keys.sort(dim=1).values
            lower = torch.searchsorted(ordered, keys, right=False)
            upper = torch.searchsorted(ordered, keys, right=True)
            # Each observation contributes its cell multiplicity. Summing
            # these multiplicities equals sum(count**2), without an M*M plane
            # or torch.sum's otherwise much larger int64 input conversion.
            asm = (upper-lower).sum(dim=1, dtype=torch.int64).to(torch.float64)
            del ordered, lower, upper, keys
            asm = asm/(oriented_pairs*oriented_pairs)
            values = asm if texture_method == 'ASM' else torch.sqrt(asm)
        else:
            x = torch.cat((left, right), dim=1).to(torch.float64)
            y = torch.cat((right, left), dim=1).to(torch.float64)
            centered_x = x-x.mean(dim=1, keepdim=True)
            centered_y = y-y.mean(dim=1, keepdim=True)
            variance_x = centered_x.square().mean(dim=1)
            variance_y = centered_y.square().mean(dim=1)
            covariance = (centered_x*centered_y).mean(dim=1)
            denominator = torch.sqrt(variance_x*variance_y)
            corr = torch.where(denominator < 1e-15, 1.0, covariance/denominator)
            near_endpoint = (((corr-1.0).abs() <= error_bound)
                             | ((corr+1.0).abs() <= error_bound))
            if bool((near_endpoint & ~single_cell).any().item()):
                raise ExtendedThresholdPrecisionError('Correlation is too close to the uint8 endpoints +/-1.')
            selected = single_cell if threshold == 0 else torch.zeros_like(single_cell)
            target[start:end] = selected.to(torch.uint8)*255
            continue
        exact_one = single_cell if texture_method in ('homogeneity', 'ASM', 'energy') else torch.zeros_like(single_cell)
        near_cutoff = (values-cutoff).abs() <= error_bound
        if bool((near_cutoff & ~exact_one).any().item()):
            raise ExtendedThresholdPrecisionError(f'{texture_method} is too close to the uint8 cutoff {cutoff:g}.')
        selected = values >= cutoff
        if threshold == 0:
            selected = selected | exact_one
        target[start:end] = selected.to(torch.uint8)*255
    _cancel(cancel)
    return output
