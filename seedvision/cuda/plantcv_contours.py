"""Bounded RETR_TREE contours from CUDA masks without a dense CPU raster.

Only row-major boundary coordinates and neighbour occupancy leave CUDA. The
existing sparse tracer supplies source-ordered contours and hierarchy. Simple
approximation removes cyclic runs of equal directions, retaining repeated visits
at direction changes. CPU topology state has the tracer's explicit point, row,
component and work limits; it is not limited to the packed boundary byte count.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch

from . import plantcv_size as size
from . import tensor_callback
from .plantcv_correction import _metadata_signature

MAX_BOUNDARY_POINTS = size.MAX_TOPOLOGY_BOUNDARY_POINTS
MAX_TILE_PIXELS = size.MAX_TILE_PIXELS
MAX_SCRATCH_BYTES = 256 * 1024 * 1024


class BinaryContourError(ValueError):
    """Invalid raster or a changed input."""


class BinaryContourResourceError(BinaryContourError):
    """A bounded extraction or sparse tracing limit was exceeded."""


class BinaryContourUnsupportedError(BinaryContourError):
    """Sparse topology did not yield a complete contour forest."""


class BinaryContourCancelled(RuntimeError):
    """Cancellation discards the entire forest."""


class BinaryContourMutation(BinaryContourError,tensor_callback.TensorCallbackMutation):
    """A callback changed a contour input or scoped extraction buffer."""


@dataclass(frozen=True)
class BinaryContourForest:
    contours: tuple[np.ndarray, ...]
    hierarchy: np.ndarray
    frame_shape: tuple[int, int]
    boundary_point_count: int
    approximation: str
    topology_status: str
    opencv_version: str
    numpy_version: str


def _signature(mask):
    try:
        return tensor_callback._signature(mask)
    except (AttributeError, RuntimeError) as error:
        raise BinaryContourError('Input must be a tensor with a mutation version counter') from error


def _immutable(array, *, shape=None):
    array = np.asarray(array, dtype=np.int32)
    # Bytes backing prevents setflags(write=True), unlike an owning array with
    # only its WRITEABLE flag cleared.
    return np.frombuffer(array.tobytes(), dtype=np.int32).reshape(
        array.shape if shape is None else shape)


def _simple(points, guard):
    count = len(points)
    if count <= 1:
        return _immutable(points, shape=(count, 1, 2))
    output = np.empty_like(points)
    parent_guard=guard
    def guard():
        try:return tensor_callback.checked_cancel(parent_guard,identity=lambda:_metadata_signature((points,output)))
        except tensor_callback.TensorCallbackMutation as error:raise BinaryContourMutation(str(error)) from error
    kept = 0
    for i in range(count):
        if i % 4096 == 0:
            guard()
        previous = points[i] - points[(i - 1) % count]
        following = points[(i + 1) % count] - points[i]
        if (int(np.sign(previous[0])), int(np.sign(previous[1]))) != (
                int(np.sign(following[0])), int(np.sign(following[1]))):
            output[kept] = points[i]
            kept += 1
    if not kept:
        output[0] = points[0]
        kept = 1
    guard()
    return _immutable(output[:kept], shape=(kept, 1, 2))


def trace_binary_contours_cuda(mask, *, approximation='simple', cancel=None,
        tile_pixels=MAX_TILE_PIXELS, work_budget=None):
    """Return immutable compact contours for a binary CUDA uint8 HxW mask.

    ``simple`` matches CHAIN_APPROX_SIMPLE; ``none`` retains every border step.
    Both use RETR_TREE. Empty masks return an empty forest. Resource limits and
    unsupported topology raise distinct errors, never a fake empty result.
    """
    if cancel is not None and not callable(cancel):
        raise TypeError('cancel must be callable or None')
    if approximation not in ('simple', 'none'):
        raise BinaryContourError('approximation must be simple or none')
    if type(tile_pixels) is not int or not 1 <= tile_pixels <= MAX_TILE_PIXELS:
        raise BinaryContourResourceError('tile_pixels exceeds the extraction tile bound')
    if work_budget is not None and (type(work_budget) is not list or len(work_budget) != 1
            or type(work_budget[0]) is not int or not 0 <= work_budget[0] <= size.MAX_CONTOUR_WORK_UNITS):
        raise BinaryContourResourceError('Shared contour work budget is invalid.')
    signature = _signature(mask)
    live={}

    def guard():
        def intact():
            if _signature(mask)!=signature:raise BinaryContourMutation('Input changed during contour extraction')
        intact()
        try:
            try:cancelled=tensor_callback.checked_cancel(cancel,(mask,),identity=lambda:_metadata_signature(live))
            except tensor_callback.TensorCallbackMutation as error:raise BinaryContourMutation(str(error)) from error
        finally:intact()
        if cancelled:
            raise BinaryContourCancelled('Contour extraction cancelled')
        return False

    guard()
    # Shape-only admission precedes binary validation and every dense allocation.
    if mask.ndim != 2:
        raise BinaryContourError('Require an HxW binary CUDA raster')
    height, width = map(int, mask.shape)
    if height > size.MAX_CONTOUR_SCAN_ROWS:
        raise BinaryContourResourceError('Contour scan-row bound exceeded')
    scratch = max(3 * height * width,
                  height * width + 64 * MAX_BOUNDARY_POINTS + 64 * tile_pixels)
    if scratch > MAX_SCRATCH_BYTES:
        raise BinaryContourResourceError('CUDA extraction scratch bound exceeded')
    try:
        size._validate_mask(mask)
    except size.PlantCVSizeResourceError as error:
        raise BinaryContourResourceError(str(error)) from error
    guard()
    tile_width = min(width, tile_pixels)
    tile_height = max(1, tile_pixels // tile_width)
    chunks, total = [], 0
    with torch.no_grad():
        foreground_full = mask != 0
        live.update(foreground_full=foreground_full,chunks=chunks)
        for y0 in range(0, height, tile_height):
            for x0 in range(0, width, tile_width):
                guard()
                foreground, bits = size._tile_neighbor_bits(foreground_full,
                    y0=y0, y1=min(height, y0 + tile_height),
                    x0=x0, x1=min(width, x0 + tile_width))
                boundary = foreground & (bits != 255)
                local = torch.nonzero(boundary, as_tuple=False)
                live.update(foreground=foreground,bits=bits,boundary=boundary,local=local)
                count = int(local.shape[0])
                if total + count > MAX_BOUNDARY_POINTS:
                    raise BinaryContourResourceError('Boundary point bound exceeded')
                total += count
                if count:
                    xs, ys = local[:, 1] + x0, local[:, 0] + y0
                    occupancy = bits[local[:, 0], local[:, 1]].to(torch.int32)
                    chunks.append(torch.stack((xs, ys, occupancy), dim=1).to(torch.int32))
                    live.update(xs=xs,ys=ys,occupancy=occupancy)
                guard()
                for key in ('foreground','bits','boundary','local','xs','ys','occupancy'):live.pop(key,None)
                del foreground, bits, boundary, local
                if count:del xs,ys,occupancy
        guard()
        if chunks:
            records = torch.cat(chunks, dim=0)
            chunks.clear()
            keys = records[:, 1].to(torch.int64) * width + records[:, 0]
            order = torch.argsort(keys)
            live.update(records=records,keys=keys,order=order)
            guard()
            records = _immutable(records[order].cpu().numpy())
            del keys,order
        else:
            records = _immutable(np.empty((0, 3), dtype=np.int32))
        live.clear();live['records']=records
        del foreground_full
    guard()
    try:
        options = {} if work_budget is None else {'work_budget': work_budget}
        contours, hierarchy, status = size._trace_sparse_contours(
            records, width=width, height=height, cancel=guard, **options)
    except size.PlantCVSizeCancelledError as error:
        raise BinaryContourCancelled(str(error)) from error
    if contours is None or hierarchy is None:
        error_type = (BinaryContourResourceError if status.startswith('unsupported_resource_bound')
                      else BinaryContourUnsupportedError)
        raise error_type(status)
    accepted = {'empty_mask', 'nested_contours', 'single_external_contour',
                'multiple_external_contours'}
    if status not in accepted:
        raise BinaryContourUnsupportedError(status)
    if (hierarchy.dtype != np.int32 or hierarchy.shape != (1, len(contours), 4)
            or not size._valid_retr_hierarchy(hierarchy, cancel=guard)
            or sum(len(c) for c in contours) > size.MAX_CONTOUR_AGGREGATE_POINTS):
        raise BinaryContourUnsupportedError('Invalid sparse contour forest')
    result = []
    live.update(contours=contours,hierarchy=hierarchy,result=result)
    for points in contours:
        guard()
        if (points.dtype != np.int32 or points.ndim != 2 or points.shape[1] != 2
                or not len(points) or np.any(points < 0)
                or np.any(points[:, 0] >= width) or np.any(points[:, 1] >= height)
                or np.any(np.abs(points - np.roll(points, 1, axis=0)) > 1)):
            raise BinaryContourUnsupportedError('Invalid sparse contour coordinates')
        result.append(_simple(points, guard) if approximation == 'simple'
                      else _immutable(points, shape=(len(points), 1, 2)))
    forest = BinaryContourForest(tuple(result), _immutable(hierarchy),
        (height, width), total, approximation, status, cv2.__version__, np.__version__)
    live['forest']=forest
    guard()
    return forest
