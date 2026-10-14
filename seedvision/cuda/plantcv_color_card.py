"""Pinned PlantCV color-card constants and bounded CUDA mask helpers."""
from __future__ import annotations

import math
import operator
from typing import Callable, Sequence

import torch

MAX_IMAGE_PIXELS = 64 * 1024 * 1024
MAX_CHIPS = 25
MAX_RADIUS = 4096
MAX_ROW_WRITES = 250_000

_STANDARD_RGB = (
    (115, 82, 68), (194, 150, 130), (98, 122, 157), (87, 108, 67),
    (133, 128, 177), (103, 189, 170), (214, 126, 44), (80, 91, 166),
    (193, 90, 99), (94, 60, 108), (157, 188, 64), (224, 163, 46),
    (56, 61, 150), (70, 148, 73), (175, 54, 60), (231, 199, 31),
    (187, 86, 149), (8, 133, 161), (243, 243, 242), (200, 200, 200),
    (160, 160, 160), (122, 122, 121), (85, 85, 85), (52, 52, 52),
)


class ColorCardError(ValueError):
    """Invalid or unsupported color-card input."""


class ColorCardCancelled(ColorCardError):
    """Raised when cancellation is observed before returning a mask."""


class ColorCardResourceError(ColorCardError):
    """Raised when declared color-card resource bounds are exceeded."""


def _cancelled(cancel: Callable[[], bool] | object | None) -> bool:
    if cancel is None:
        return False
    if callable(cancel):
        return bool(cancel())
    is_set = getattr(cancel, 'is_set', None)
    if callable(is_set):
        return bool(is_set())
    raise ColorCardError('cancel must be a callable or event exposing is_set()')


def _index(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise ColorCardError(f'{name} must be an integer')
    try:
        return operator.index(value)
    except TypeError as error:
        raise ColorCardError(f'{name} must be an integer') from error


def std_color_matrix(pos: int | float = 0, *, device: torch.device | str = 'cuda') -> torch.Tensor:
    """Return PlantCV's 24x4 standard RGB matrix as float64 on device.

    The position is floored before validation. The 4x6 chip-index matrix is
    rotated counterclockwise, while returned chip IDs remain 10 through 240.
    """
    try:
        orientation = math.floor(pos)
    except (TypeError, ValueError, OverflowError) as error:
        raise ColorCardError('pos must be a finite number with floor in 0..3') from error
    if orientation < 0 or orientation > 3:
        raise ColorCardError('pos must have a floor of 0, 1, 2, or 3')
    target = torch.device(device)
    # These are compact reference constants. Normalize with correctly rounded
    # scalar division before upload; CUDA scalar division may fold to a rounded
    # reciprocal multiplication and differ from NumPy by one float64 ULP.
    rgb = torch.tensor(tuple(tuple(value/255.0 for value in row)
        for row in _STANDARD_RGB), dtype=torch.float64, device=target)
    indices = torch.arange(1, 25, dtype=torch.int64, device=target).reshape(4, 6)
    rotated = torch.rot90(indices, k=orientation, dims=(0, 1)).reshape(-1)
    ordered_rgb = rgb.index_select(0, rotated-1)
    chip_ids = torch.arange(10, 241, 10, dtype=torch.float64, device=target).reshape(24, 1)
    return torch.cat((chip_ids, ordered_rgb), dim=1)


def _circle_row_spans(radius: int) -> tuple[tuple[int, int, int], ...]:
    """Return (y offset, x minimum, x maximum) spans for a filled cv2 circle."""
    spans: dict[int, tuple[int, int]] = {}

    def add(y_offset: int, x_min: int, x_max: int) -> None:
        old = spans.get(y_offset)
        if old is None:
            spans[y_offset] = (x_min, x_max)
        else:
            spans[y_offset] = (min(old[0], x_min), max(old[1], x_max))

    # OpenCV's integer circle scan emits four filled horizontal spans.
    err, dx, dy, plus, minus = 0, radius, 0, 1, (radius << 1)-1
    while dx >= dy:
        add(dy, -dx, dx)
        add(-dy, -dx, dx)
        add(dx, -dy, dy)
        add(-dx, -dy, dy)
        dy += 1
        err += plus
        plus += 2
        step = 0 if err <= 0 else -1
        err -= minus & step
        dx += step
        minus -= step & 2
    return tuple((y_offset, span[0], span[1]) for y_offset, span in sorted(spans.items()))


def _pair(value: Sequence[object], name: str) -> tuple[int, int]:
    try:
        if len(value) != 2:
            raise ColorCardError(f'{name} must contain exactly two integers')
        return _index(value[0], f'{name}[0]'), _index(value[1], f'{name}[1]')
    except TypeError as error:
        raise ColorCardError(f'{name} must contain exactly two integers') from error


def create_color_card_mask(rgb_img: torch.Tensor, radius: int, start_coord: Sequence[int],
        spacing: Sequence[int], nrows: int, ncols: int, exclude: Sequence[int] | None = None,
        *, cancel: Callable[[], bool] | object | None = None) -> torch.Tensor:
    """Create a PlantCV-style uint8 labeled circle mask on CUDA.

    Centers are row-major from (x,y) and (horizontal,vertical) spacing.
    Descending exclusions retain Python negative-index/duplicate deletion.
    Survivors are relabeled 10,20,... and later circles overwrite earlier ones.
    Exclusions are copied rather than sorted in the caller's list. Empty clipped
    ROIs are rejected, preserving the public call's failure instead of returning
    a completed mask.
    """
    if not isinstance(rgb_img, torch.Tensor) or rgb_img.device.type != 'cuda':
        raise ColorCardError('rgb_img must be a CUDA tensor')
    if rgb_img.ndim < 2:
        raise ColorCardError('rgb_img must have height and width dimensions')
    height, width = int(rgb_img.shape[0]), int(rgb_img.shape[1])
    pixel_count = height*width
    if height <= 0 or width <= 0 or pixel_count > MAX_IMAGE_PIXELS:
        raise ColorCardResourceError(f'image must contain 1..{MAX_IMAGE_PIXELS} pixels')
    radius, nrows, ncols = _index(radius, 'radius'), _index(nrows, 'nrows'), _index(ncols, 'ncols')
    if radius < 0:
        raise ColorCardError('radius must be non-negative')
    if radius > MAX_RADIUS:
        raise ColorCardResourceError(f'radius exceeds the {MAX_RADIUS}-pixel limit')
    if nrows < 1 or ncols < 1:
        raise ColorCardError('nrows and ncols must be positive')
    if nrows*ncols > MAX_CHIPS:
        raise ColorCardResourceError(f'at most {MAX_CHIPS} chip circles are supported')
    start_x, start_y = _pair(start_coord, 'start_coord')
    space_x, space_y = _pair(spacing, 'spacing')
    if _cancelled(cancel):
        raise ColorCardCancelled('color-card mask cancelled before rasterization')
    centers = []
    # Validate all ROIs before excluding chips, preserving the source > bound.
    for row in range(nrows):
        y = start_y+row*space_y
        for col in range(ncols):
            x = start_x+col*space_x
            if x-radius < 0 or x+radius > width or y-radius < 0 or y+radius > height:
                raise ColorCardError('a circle ROI extends outside the image')
            if radius == 0 and (x >= width or y >= height):
                raise ColorCardError('a clipped circle ROI contains no pixels')
            centers.append((x, y))
    if len(centers)*(2*radius+1) > MAX_ROW_WRITES:
        raise ColorCardResourceError('circle raster work exceeds the bounded row-write limit')
    if exclude is None:
        excluded = []
    else:
        try:
            excluded = [_index(item, 'exclude index') for item in exclude]
        except TypeError as error:
            raise ColorCardError('exclude must be a sequence of integer chip indexes') from error
    for chip_index in sorted(excluded, reverse=True):
        try:
            del centers[chip_index]
        except IndexError as error:
            raise ColorCardError(f'exclude index {chip_index} is outside the current chip list') from error
    result = torch.zeros((height, width), dtype=torch.uint8, device=rgb_img.device)
    spans = _circle_row_spans(radius)
    for chip_number, (center_x, center_y) in enumerate(centers, start=1):
        if _cancelled(cancel):
            raise ColorCardCancelled('color-card mask cancelled before result commit')
        for y_offset, x_min, x_max in spans:
            if _cancelled(cancel):
                raise ColorCardCancelled('color-card mask cancelled before result commit')
            image_y = center_y+y_offset
            if image_y < 0 or image_y >= height:
                continue
            left, right = max(0, center_x+x_min), min(width-1, center_x+x_max)
            if left <= right:
                result[image_y, left:right+1] = chip_number*10
    if _cancelled(cancel):
        raise ColorCardCancelled('color-card mask cancelled before result commit')
    return result
