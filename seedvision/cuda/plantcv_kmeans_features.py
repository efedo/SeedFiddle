"""CUDA row-major patch tiles for PlantCV's K-means image predictor."""
from __future__ import annotations

import torch
from seedvision.cuda.tensor_callback import checked_cancel

MAX_PATCH_PIXELS = 16_000_000
MAX_PADDED_PIXELS = 16_000_000
MAX_PATCH_FEATURES = 4096
MAX_PATCH_SIZE = 31
MAX_FEATURE_TILE_PIXELS = 4096
MAX_FEATURE_TILE_BYTES = 32 * 1024 * 1024
FEATURE_TILE_ROW_OVERHEAD_BYTES = 128


class KMeansFeatureError(ValueError):
    """Invalid patch raster or a patch resource limit."""


class KMeansFeatureCancelled(RuntimeError):
    """Patch extraction was cancelled."""


def _poll(cancel,*tensors):
    if checked_cancel(cancel,tensors):
        raise KMeansFeatureCancelled('K-means patch extraction was cancelled')


def _feature_tile_pixels(feature_count, requested):
    if type(requested) is not int or not 1 <= requested <= MAX_FEATURE_TILE_PIXELS:
        raise KMeansFeatureError(f'tile_pixels must be an integer in [1,{MAX_FEATURE_TILE_PIXELS}]')
    return min(requested, MAX_FEATURE_TILE_BYTES // (feature_count * 2 + FEATURE_TILE_ROW_OVERHEAD_BYTES))


def iter_kmeans_patch_tiles(blurred_padded, patch_size, *, cancel=None,
                            tile_pixels=MAX_FEATURE_TILE_PIXELS):
    """Yield independent uint16 feature tiles in (patch_y,patch_x,channel) order.

    Each start offset describes one contiguous range in the flattened output
    raster. Full windows are views; only a bounded tile is copied. uint16 values
    above the byte range remain intact. Consumers should release each tile
    before requesting the next one to retain the declared live storage bound.
    """
    if cancel is not None and not callable(cancel):
        raise TypeError('cancel must be a zero-argument callable or None')
    _poll(cancel,blurred_padded)
    if not torch.is_tensor(blurred_padded) or blurred_padded.device.type != 'cuda':
        raise KMeansFeatureError('blurred_padded must already be a CUDA tensor')
    if (blurred_padded.layout != torch.strided or blurred_padded.dtype != torch.uint16
            or not blurred_padded.is_contiguous()):
        raise KMeansFeatureError('blurred_padded must be contiguous strided CUDA uint16')
    if type(patch_size) is not int or not 1 <= patch_size <= MAX_PATCH_SIZE:
        raise KMeansFeatureError(f'patch_size must be an integer in [1,{MAX_PATCH_SIZE}]')
    if blurred_padded.ndim == 2:
        padded_height, padded_width = map(int, blurred_padded.shape)
        channels, source = 1, blurred_padded.unsqueeze(-1)
    elif blurred_padded.ndim == 3 and blurred_padded.shape[2] == 3:
        padded_height, padded_width, channels = map(int, blurred_padded.shape)
        source = blurred_padded
    else:
        raise KMeansFeatureError('blurred_padded must be HW or HWC3')
    height, width = padded_height - patch_size + 1, padded_width - patch_size + 1
    if height < 1 or width < 1:
        raise KMeansFeatureError('blurred_padded is smaller than patch_size')
    if height * width > MAX_PATCH_PIXELS or padded_height * padded_width > MAX_PADDED_PIXELS:
        raise KMeansFeatureError('patch or padded pixel count exceeds the supported bounds')
    feature_count = patch_size * patch_size * channels
    if feature_count > MAX_PATCH_FEATURES:
        raise KMeansFeatureError('patch feature width exceeds the supported bound')
    selected = _feature_tile_pixels(feature_count, tile_pixels)
    if selected < 1:
        raise KMeansFeatureError('one patch exceeds the feature tile limit')
    windows = source.unfold(0, patch_size, 1).unfold(1, patch_size, 1).permute(0, 1, 3, 4, 2)
    columns_per_tile = min(width, selected)
    rows_per_tile = 1 if width > selected else max(1, selected // width)
    for row in range(0, height, rows_per_tile):
        row_stop = min(height, row + rows_per_tile)
        for column in range(0, width, columns_per_tile):
            _poll(cancel,blurred_padded)
            column_stop = min(width, column + columns_per_tile)
            count = (row_stop - row) * (column_stop - column)
            features = torch.empty((count, feature_count), dtype=torch.uint16, device=source.device)
            # A shaped destination copies the strided view without uint16
            # advanced indexing or an intermediate reshape allocation. It also
            # guarantees p=1 tiles do not alias the caller's source tensor.
            features.view(row_stop - row, column_stop - column, patch_size,
                          patch_size, channels).copy_(windows[row:row_stop, column:column_stop])
            _poll(cancel,blurred_padded,features)
            yield row * width + column, features
            del features
    _poll(cancel,blurred_padded)
