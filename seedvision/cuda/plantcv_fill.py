"""CUDA-resident small-object removal for the pinned PlantCV ``fill`` call.

PlantCV 4.11.3's public ``fill.py`` calls scikit-image
``remove_small_objects`` with its default 2-D connectivity of one. That means
4-connected foreground components with area strictly smaller than ``size`` are
removed. Despite the public name, this operation does not fill enclosed holes.
With no ROI, the complete input canvas participates, including components that
touch any image border.

Nonempty half-open ROI bounds filter a crop independently and replace it in a
fresh full-frame uint8 mask. The public wrapper's explicit-ROI parent expression
uses NumPy's platform integer dtype; this native BinaryMask API normalizes that
result to uint8 while preserving every 0/255 value. Typed-output parity for that
public ROI quirk is not claimed.

This implementation supports the explicit uint8 0/255 mask domain. It uses
CUDA union/find and device-side component-area accumulation; no pixel raster,
label map, or area table is copied to the CPU. Chunked edge processing bounds
temporary edge storage independently of foreground density. The accepted raster
limit matches the isolated PlantCV runner's 64,000,000-pixel protocol cap.

The wrapper source is pinned to PlantCV 4.11.3, commit
``77e3322fd0cd5a9bef18bb99402f20c8f2db87a9``. The source wrapper does not pin
its transitive scikit-image helper source; the exact dependency archive is part
of the committed reference environment lock.
"""

from __future__ import annotations

from math import gcd
from typing import Callable

import torch

from seedvision.cuda import tensor_callback as _tensor_callback


PLANTCV_VERSION = "4.11.3"
PLANTCV_COMMIT = "77e3322fd0cd5a9bef18bb99402f20c8f2db87a9"
PLANTCV_FILL_SOURCE_SHA256 = "73c003887b771504d9e89a8b63bab6211f0f2f22474d1ad9d429aea167d05803"
MIN_OBJECT_AREA = 1
MAX_RASTER_PIXELS = 64_000_000
MAX_EDGE_CHUNK = 1_000_000
MAX_UNION_PASSES = 64
MAX_ROOT_JUMPS = 26
PIXEL_CHUNK = 1_000_000


class FillResourceError(ValueError):
    """Raised when a raster exceeds the isolated runner's pixel cap."""


class FillCancelledError(RuntimeError):
    """Filtering was cancelled without returning a partial output mask."""


class FillConvergenceError(RuntimeError):
    """CUDA connected-component union/find did not converge within its bound."""


def _validate_mask(mask: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(mask):
        raise TypeError("mask must be a PyTorch tensor")
    if mask.layout != torch.strided:
        raise TypeError("mask must be a dense strided tensor")
    if mask.device.type != "cuda":
        raise ValueError("mask must already reside on a CUDA device")
    if mask.dtype != torch.uint8:
        raise TypeError("mask must have dtype torch.uint8")
    if mask.ndim != 2:
        raise ValueError("mask must be a two-dimensional HxW raster")
    height, width = (int(value) for value in mask.shape)
    if height < 1 or width < 1:
        raise ValueError("mask must have non-empty height and width")
    if height * width > MAX_RASTER_PIXELS:
        raise FillResourceError(
            f"mask exceeds the reference runner limit of {MAX_RASTER_PIXELS:,} pixels"
        )
    stride_y, stride_x = (int(value) for value in mask.stride())
    if (height > 1 and stride_y <= 0) or (width > 1 and stride_x <= 0):
        raise ValueError("mask must not use overlapping or reversed raster strides")
    if height > 1 and width > 1:
        common = gcd(stride_y, stride_x)
        if stride_x // common < height and stride_y // common < width:
            raise ValueError("mask must not use overlapping raster strides")
    return mask


def _positive_area(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("size must be an integer")
    if value < MIN_OBJECT_AREA:
        raise ValueError("size must be a positive integer")
    return value


def _roi_bounds(roi, height: int, width: int):
    if roi is None:
        return (0, 0, width, height)
    if type(roi) is not tuple or len(roi) != 4 or any(type(v) is not int for v in roi):
        raise TypeError('roi must be None or a four-integer (x0, y0, x1, y1) tuple')
    x0, y0, x1, y1 = roi
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError('roi must be a non-empty half-open rectangle inside the image')
    return roi


def _check_cancel(cancel: Callable[[], bool] | None, tensors=()) -> None:
    if _tensor_callback.checked_cancel(cancel, tensors):
        raise FillCancelledError("Small-object filtering was cancelled.")


def _union_edge_batch(
    parent: torch.Tensor,
    edge_a: torch.Tensor,
    edge_b: torch.Tensor,
    cancel: Callable[[], bool] | None,
    watched=(),
) -> torch.Tensor:
    """Merge one bounded edge batch until all its endpoint pairs agree."""
    for _ in range(MAX_UNION_PASSES):
        _check_cancel(cancel, (*watched, parent, edge_a, edge_b))
        # The parent raster is fully flattened before this batch and after
        # every concurrent hook, so these gather results are exact roots.
        root_a = parent[edge_a]
        root_b = parent[edge_b]
        different = root_a != root_b
        if not bool(different.any().item()):
            return parent
        higher = torch.maximum(root_a[different], root_b[different])
        lower = torch.minimum(root_a[different], root_b[different])
        high_indices = higher.to(torch.int64)
        parent.scatter_reduce_(
            0,
            high_indices,
            lower,
            reduce="amin",
            include_self=True,
        )
        parent = _compress_roots(parent, cancel,
            (*watched, edge_a, edge_b, root_a, root_b, different, higher, lower, high_indices))
        del root_a, root_b, different, higher, lower, high_indices
    raise FillConvergenceError(
        f"Connected-component edge batch exceeded {MAX_UNION_PASSES} union passes."
    )


def _union_direction(
    foreground: torch.Tensor,
    parent: torch.Tensor,
    *,
    horizontal: bool,
    cancel: Callable[[], bool] | None,
    watched=(),
) -> torch.Tensor:
    height, width = (int(value) for value in foreground.shape)
    if horizontal:
        if width < 2:
            return parent
        adjacent = foreground[:, :-1] & foreground[:, 1:]
        row_width = width - 1
        vertical_step = 0
    else:
        if height < 2:
            return parent
        adjacent = foreground[:-1, :] & foreground[1:, :]
        row_width = width
        vertical_step = width
    adjacent_flat = adjacent.contiguous().view(-1)
    edge_count = int(adjacent_flat.numel())
    for start in range(0, edge_count, MAX_EDGE_CHUNK):
        _check_cancel(cancel, (*watched, foreground, parent, adjacent, adjacent_flat))
        stop = min(edge_count, start + MAX_EDGE_CHUNK)
        local_edges = torch.nonzero(adjacent_flat[start:stop], as_tuple=False).flatten()
        if local_edges.numel() == 0:
            del local_edges
            continue
        linear = local_edges + start
        y = torch.div(linear, row_width, rounding_mode="floor")
        x = linear - y * row_width
        first = (y * width + x).to(torch.int32)
        second = first + (1 if horizontal else vertical_step)
        parent = _union_edge_batch(parent, first, second, cancel,
            (*watched, foreground, adjacent, adjacent_flat, local_edges, linear, y, x))
        del local_edges, linear, y, x, first, second
    return parent


def _compress_roots(
    parent: torch.Tensor,
    cancel: Callable[[], bool] | None,
    watched=(),
) -> torch.Tensor:
    """Fully flatten the parent forest with bounded device pointer-doubling."""
    for _ in range(MAX_ROOT_JUMPS):
        _check_cancel(cancel, (*watched, parent))
        compressed = parent[parent]
        if torch.equal(compressed, parent):
            return compressed
        parent = compressed
    raise FillConvergenceError("Connected-component flattening exceeded its bound.")


def _component_areas(
    foreground_flat: torch.Tensor,
    parent: torch.Tensor,
    cancel: Callable[[], bool] | None,
    watched=(),
) -> torch.Tensor:
    """Count foreground members at each root, keeping the table on CUDA."""
    areas = torch.zeros_like(parent)
    pixel_count = int(foreground_flat.numel())
    for start in range(0, pixel_count, PIXEL_CHUNK):
        _check_cancel(cancel, (*watched, foreground_flat, parent, areas))
        stop = min(pixel_count, start + PIXEL_CHUNK)
        active = foreground_flat[start:stop]
        roots = parent[start:stop][active]
        areas.scatter_add_(
            0,
            roots.to(torch.int64),
            torch.ones_like(roots),
        )
        del active, roots
    return areas


def _filter_crop(mask, size, cancel, watched):
    """Filter one independent crop with callback-scoped live-buffer guards."""
    pixel_count = int(mask.numel())
    _check_cancel(cancel, (*watched, mask))
    if size == 1:
        output = mask.clone(memory_format=torch.contiguous_format)
        _check_cancel(cancel, (*watched, mask, output))
        return output
    if size > pixel_count:
        output = torch.zeros_like(mask, memory_format=torch.contiguous_format)
        _check_cancel(cancel, (*watched, mask, output))
        return output

    foreground = (mask == 255).contiguous()
    _check_cancel(cancel, (*watched, mask, foreground))
    if not bool(foreground.any().item()):
        output = mask.clone(memory_format=torch.contiguous_format)
        _check_cancel(cancel, (*watched, mask, foreground, output))
        return output

    parent = torch.arange(pixel_count, device=mask.device, dtype=torch.int32)
    live = (*watched, mask, foreground)
    parent = _union_direction(foreground, parent, horizontal=True, cancel=cancel, watched=live)
    parent = _union_direction(foreground, parent, horizontal=False, cancel=cancel, watched=live)
    parent = _compress_roots(parent, cancel, live)
    foreground_flat = foreground.view(-1)
    areas = _component_areas(foreground_flat, parent, cancel, live)
    retained_roots = areas >= size
    output = torch.empty_like(mask, memory_format=torch.contiguous_format)
    output_flat = output.view(-1)
    for start in range(0, pixel_count, PIXEL_CHUNK):
        _check_cancel(cancel, (*live, parent, foreground_flat, areas, retained_roots, output, output_flat))
        stop = min(pixel_count, start + PIXEL_CHUNK)
        roots = parent[start:stop]
        keep = foreground_flat[start:stop] & retained_roots[roots.to(torch.int64)]
        output_flat[start:stop] = keep.to(torch.uint8) * 255
        del roots, keep
    _check_cancel(cancel, (*live, parent, foreground_flat, areas, retained_roots, output, output_flat))
    return output


def fill(
    mask: torch.Tensor,
    size: int,
    *,
    roi=None,
    cancel: Callable[[], bool] | None = None,
) -> torch.Tensor:
    """Remove 4-connected foreground objects whose area is less than ``size``.

    ``mask`` is a dense CUDA ``uint8`` HxW tensor with values exactly 0 or 255.
    ``size`` is any positive Python integer, matching PlantCV's unbounded
    positive area-threshold contract. Pixels on the canvas border participate
    normally. Optional nonempty half-open ROI bounds count and filter components
    inside the crop independently, then replace it in a fresh full-frame copy.
    Exterior pixels remain unchanged. The input
    is never modified, and the returned mask remains uint8 0/255 on the same
    CUDA device. ``cancel`` may be a callback such as ``Event.is_set``; it is
    checked before allocation, during edge/area/output passes, and before
    return, with strict bool results and mutation checks even on exceptions.
    Arbitrary contours and empty/out-of-frame rectangles are unsupported.
    """
    mask = _validate_mask(mask)
    size = _positive_area(size)
    x0, y0, x1, y1 = _roi_bounds(roi, *map(int, mask.shape))
    if cancel is not None and not callable(cancel):
        raise TypeError("cancel must be a callable or None")
    _check_cancel(cancel, (mask,))

    binary_values = (mask == 0) | (mask == 255)
    if not bool(binary_values.all().item()):
        raise ValueError("mask must contain only the binary uint8 values 0 and 255")
    del binary_values
    crop = mask[y0:y1, x0:x1]
    filtered = _filter_crop(crop, size, cancel, (mask,))
    _check_cancel(cancel, (mask, crop, filtered))
    if (x0, y0, x1, y1) == (0, 0, int(mask.shape[1]), int(mask.shape[0])):
        return filtered
    output = mask.clone(memory_format=torch.contiguous_format)
    output[y0:y1, x0:x1] = filtered
    _check_cancel(cancel, (mask, crop, filtered, output))
    return output


__all__ = [
    "MAX_EDGE_CHUNK",
    "MAX_RASTER_PIXELS",
    "MIN_OBJECT_AREA",
    "FillCancelledError",
    "FillConvergenceError",
    "FillResourceError",
    "PLANTCV_COMMIT",
    "PLANTCV_FILL_SOURCE_SHA256",
    "PLANTCV_VERSION",
    "fill",
]
