"""Bounded CUDA preprocessing for the pinned PlantCV K-means predictor.

The source pads with edge values before sigma-five Gaussian filtering, rounds
byte-scaled values to even, and stores uint16 before extracting image patches.
Only tile halos are gathered here; the padded source is virtual. See
docs/SCIPY_KMEANS_NOTICES.md for the symmetric correlation attribution.
"""
from __future__ import annotations

import math
import struct

import torch
from seedvision.cuda.tensor_callback import checked_cancel

MAX_IMAGE_PIXELS = 16_000_000
MAX_PADDED_PIXELS = 16_000_000
MAX_OUTPUT_BYTES = 128 * 1024 * 1024
MAX_TILE_SIDE = 256
MAX_SCRATCH_BYTES = 64 * 1024 * 1024
MAX_FILTER_WORK = 1_000_000_000
SIGMA5_WEIGHT_BITS = (
    '3efc113e67a34f9a', '3f0e9d347af7ba2c', '3f200a91aed84201',
    '3f3026ceaaef5d9c', '3f3f3ffe5366298d', '3f4d0bb4c23b8d53',
    '3f59f03a798bae24', '3f664156b94ff939', '3f7258a96c00a508',
    '3f7d0fdc1a91a71b', '3f861d971cc0d07e', '3f902b6d98acd25a',
    '3f96b7adf708e81b', '3f9eaa58a4ba7225', '3fa3e2acd55166da',
    '3fa8c75f2fc165cb', '3fadaa6517492b60', '3fb10fd11517a7f6',
    '3fb2db2f1c27e704', '3fb405ae2f8a8257', '3fb46d39dcd3d08c',
    '3fb405ae2f8a8257', '3fb2db2f1c27e704', '3fb10fd11517a7f6',
    '3fadaa6517492b60', '3fa8c75f2fc165cb', '3fa3e2acd55166da',
    '3f9eaa58a4ba7225', '3f96b7adf708e81b', '3f902b6d98acd25a',
    '3f861d971cc0d07e', '3f7d0fdc1a91a71b', '3f7258a96c00a508',
    '3f664156b94ff939', '3f59f03a798bae24', '3f4d0bb4c23b8d53',
    '3f3f3ffe5366298d', '3f3026ceaaef5d9c', '3f200a91aed84201',
    '3f0e9d347af7ba2c', '3efc113e67a34f9a',
)


class KMeansPreprocessError(ValueError):
    """Invalid input or a preprocessing resource limit."""


class KMeansPreprocessCancelled(RuntimeError):
    """Cancelled without publishing a partial raster."""


class KMeansPrecisionError(KMeansPreprocessError):
    """The reference byte-rounding decision is not established."""


def _check_cancel(cancel,*tensors):
    if checked_cancel(cancel,tensors):
        raise KMeansPreprocessCancelled('K-means preprocessing was cancelled')


def _weights():
    # Decode inside verified code rather than caching computed module globals.
    return tuple(struct.unpack('>d', bytes.fromhex(bits))[0]
                 for bits in SIGMA5_WEIGHT_BITS)


def _byte_error_bound():
    """Bound both binary64 two-axis reductions through byte rescaling.

    Each backend's 41-term pass needs at most 81 rounded operations; SciPy's
    symmetric-pair path needs fewer. With u=2**-53, gamma_81 bounds each pass.
    Nonnegative coefficients and normalized inputs bound both axes by S**2.
    Eight gamma terms conservatively cover the two backends/two axes, including
    the second pass propagating the first pass's error. Four u terms cover input
    normalization and final scaling. The final bound is rounded upward.
    """
    unit = 2.0 ** -53
    total = math.fsum(abs(value) for value in _weights()) + 2.0 * unit
    gamma = 81.0 * unit / (1.0 - 81.0 * unit)
    return math.nextafter(255.0 * total * total * (8.0 * gamma + 4.0 * unit),
                          math.inf)


def _preflight_scratch_bytes(channels, tile_side):
    # Gather/cast halo, one pair at a time, two pass outputs and rounding
    # temporaries; extra headroom includes indices and allocator granularity.
    halo = (tile_side + 40) ** 2 * channels
    pixels = tile_side ** 2 * channels
    return halo * 32 + pixels * 64 + 2 * 1024 * 1024


def _preflight(image, patch_size, tile_side):
    if not torch.is_tensor(image) or image.device.type != 'cuda':
        raise KMeansPreprocessError('image must already be a CUDA tensor')
    if (image.layout != torch.strided or image.dtype != torch.uint8
            or image.ndim not in (2, 3) or not image.is_contiguous()):
        raise KMeansPreprocessError('image must be contiguous strided uint8 HW or HWC')
    channels = 1 if image.ndim == 2 else int(image.shape[2])
    if channels not in (1, 3) or (image.ndim == 3 and channels != 3):
        raise KMeansPreprocessError('HWC images must have three channels')
    height, width = map(int, image.shape[:2])
    return _preflight_shape(channels, height, width, patch_size, tile_side)


def _preflight_shape(channels, height, width, patch_size, tile_side):
    """Apply the raster resource contract without allocating or inspecting CUDA."""
    if type(channels) is not int or channels not in (1, 3):
        raise KMeansPreprocessError('image channels must be one or three')
    if (type(height) is not int or type(width) is not int
            or height < 1 or width < 1 or height * width > MAX_IMAGE_PIXELS):
        raise KMeansPreprocessError('image pixel count exceeds the supported bounds')
    if type(patch_size) is not int or not 1 <= patch_size <= 31:
        raise KMeansPreprocessError('patch_size must be an integer in [1,31]')
    if type(tile_side) is not int or not 1 <= tile_side <= MAX_TILE_SIDE:
        raise KMeansPreprocessError(f'tile_side must be an integer in [1,{MAX_TILE_SIDE}]')
    padded_height, padded_width = height + patch_size - 1, width + patch_size - 1
    padded_pixels = padded_height * padded_width
    if padded_pixels > MAX_PADDED_PIXELS or padded_pixels * channels * 2 > MAX_OUTPUT_BYTES:
        raise KMeansPreprocessError('padded raster exceeds the output limit')
    # Vertical filtering repeats the horizontal halo for each tile column.
    # Charge 41 coefficient terms per pass output, including those halos.
    tile_columns = (padded_width + tile_side - 1) // tile_side
    vertical_pixels = padded_height * (padded_width + 40 * tile_columns)
    if (vertical_pixels + padded_pixels) * channels * 41 > MAX_FILTER_WORK:
        raise KMeansPreprocessError('Gaussian filter work exceeds the supported bound')
    if _preflight_scratch_bytes(channels, tile_side) > MAX_SCRATCH_BYTES:
        raise KMeansPreprocessError('Gaussian tile exceeds the scratch limit')
    return channels, height, width, padded_height, padded_width


def gaussian_patch_raster(image, patch_size=10, *, cancel=None, tile_side=MAX_TILE_SIDE):
    """Return the padded, blurred CUDA uint16 raster for patch extraction.

    A byte decision whose binary64 error interval reaches a half-integer
    withholds the complete raster. Input tensors are never mutated.
    """
    if cancel is not None and not callable(cancel):
        raise TypeError('cancel must be a zero-argument callable or None')
    _check_cancel(cancel,image)
    channels, height, width, out_height, out_width = _preflight(image, patch_size, tile_side)
    weights, error_bound = _weights(), _byte_error_bound()
    _check_cancel(cancel,image)
    shape = (out_height, out_width) if image.ndim == 2 else (out_height, out_width, channels)
    output = torch.empty(shape, dtype=torch.uint16, device=image.device)
    destination = output.unsqueeze(2) if image.ndim == 2 else output
    source = image.unsqueeze(2) if image.ndim == 2 else image
    before = (patch_size - 1) // 2
    uncertain = torch.zeros((), dtype=torch.bool, device=image.device)
    for top in range(0, out_height, tile_side):
        for left in range(0, out_width, tile_side):
            _check_cancel(cancel,image,output,uncertain)
            bottom, right = min(out_height, top + tile_side), min(out_width, left + tile_side)
            tile_height, tile_width = bottom - top, right - left
            rows = (torch.arange(top - 20, bottom + 20, device=image.device)
                    - before).clamp_(0, height - 1)
            columns = (torch.arange(left - 20, right + 20, device=image.device)
                       - before).clamp_(0, width - 1)
            tile = source[rows[:, None], columns[None, :], :].to(torch.float64)
            tile.mul_(1.0 / 255.0)
            vertical = tile[20:20 + tile_height] * weights[20]
            # SciPy NI_Correlate1D's symmetric-pair order: -20 through -1.
            for offset in range(-20, 0):
                _check_cancel(cancel,image,output,uncertain,rows,columns,tile,vertical)
                pair = (tile[20 + offset:20 + offset + tile_height]
                        + tile[20 - offset:20 - offset + tile_height])
                vertical.add_(pair.mul_(weights[20 + offset]))
                del pair
            del tile
            horizontal = vertical[:, 20:20 + tile_width] * weights[20]
            for offset in range(-20, 0):
                _check_cancel(cancel,image,output,uncertain,rows,columns,vertical,horizontal)
                pair = (vertical[:, 20 + offset:20 + offset + tile_width]
                        + vertical[:, 20 - offset:20 - offset + tile_width])
                horizontal.add_(pair.mul_(weights[20 + offset]))
                del pair
            del vertical
            horizontal.mul_(255.0)
            near_half = (horizontal - (torch.floor(horizontal) + 0.5)).abs_() <= error_bound
            uncertain.logical_or_(near_half.any())
            destination[top:bottom, left:right].copy_(torch.round(horizontal).to(torch.uint16))
            del horizontal, near_half, rows, columns
    _check_cancel(cancel,image,output,uncertain)
    if bool(uncertain.item()):
        raise KMeansPrecisionError('Gaussian byte rounding is within the binary64 error interval')
    _check_cancel(cancel,image,output,uncertain)
    return output
