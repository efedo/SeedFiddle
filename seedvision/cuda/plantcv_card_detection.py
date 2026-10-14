"""Automatic PlantCV colour-card geometry over CUDA-resident image rasters.

The pinned 4.11.3 detector filters RETR_TREE/SIMPLE contours, orders rectangle
corners by the white chip and uses int32 cv2.transform WITHOUT a perspective
divide. Sparse contours, four sampled pixels and compact matrices use CPU
geometry; full image/filter/mask/correction arrays stay on CUDA. Actual OpenCV
and NumPy versions are retained; conformance is limited to tested domains.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

import cv2
import numpy as np
import torch

from . import plantcv_card_preprocess as preprocessing
from . import tensor_callback
from .plantcv_contours import BinaryContourForest, trace_binary_contours_cuda
from .plantcv_minrect import (
    ROTCALIPERS_SOURCE_SHA256, TYPES_SOURCE_SHA256, box_points_411, min_area_rect_411,
)
from .plantcv_color_card import MAX_RADIUS, MAX_ROW_WRITES, _circle_row_spans, std_color_matrix
from .plantcv_correction import (
    ColorCorrectionResult, affine_color_correction_cuda, get_color_matrix_cuda, _metadata_signature,
)

DETECT_SOURCE_SHA256 = 'e3ad4028e8a831e523815e33ee057f08ba73f5a3f60b3933f11504dd71b7bb49'
AUTO_CORRECT_SOURCE_SHA256 = 'bed2eb9e50e42e39f7ad4bbef157150190585268178f06083092edb19ffaf57a'
MAX_RETAINED_BYTES = 512 * 1024 * 1024
MAX_GEOMETRY_WORK = 128_000_000
MAX_CIRCLE_WORK = 128_000_000
# Actual cv2.boxPoints(cv2.minAreaRect(fixed int32 4x6 grid)) in the pinned
# OpenCV 4.11.0 wheel. Order and tiny negative first x are significant.
GRID_BOX_411 = np.frombuffer(bytes.fromhex(
    '0be289a8000000000000964300000000000096430000fa43000000000000fa43'),
    dtype='<f4').reshape(4, 2)
GRID_BOX_SHA256 = '39c9018bf2b16570e4c2871891f031b76f5cdfb52131b1673b26e0672cec0ac6'
REFERENCE_SOURCES = (
    ('plantcv/transform/detect_color_card.py', DETECT_SOURCE_SHA256),
    ('plantcv/transform/auto_correct_color.py', AUTO_CORRECT_SOURCE_SHA256),
    ('opencv/imgproc/rotcalipers.cpp', ROTCALIPERS_SOURCE_SHA256),
    ('opencv/core/types.cpp', TYPES_SOURCE_SHA256),
)


class AutomaticCardError(ValueError):
    """Invalid input, changed data or unavailable detector geometry."""


class AutomaticCardNotFound(AutomaticCardError):
    """The source detector has no admissible card or one-to-one chip mapping."""


class AutomaticCardResourceError(AutomaticCardError):
    """Full detection/correction storage or work exceeds its declared limit."""


class AutomaticCardCancelled(RuntimeError):
    """No partial detection or corrected image is published."""


class AutomaticCardMutation(AutomaticCardError,tensor_callback.TensorCallbackMutation):
    """A callback changed a live card input, tensor or compact geometry."""


def _checked(guard,tensors=(),metadata=()):
    try:return tensor_callback.checked_cancel(guard,tensors,identity=lambda:_metadata_signature(metadata))
    except tensor_callback.TensorCallbackMutation as error:raise AutomaticCardMutation(str(error)) from error


@dataclass(frozen=True)
class AutomaticCardGeometry:
    selected_contour_indices: tuple[int, ...]
    areas: tuple[float, ...]
    widths: tuple[float, ...]
    heights: tuple[float, ...]
    bounding_rect: tuple[int, int, int, int]
    corners_xy: np.ndarray
    centers_xy: np.ndarray
    transformation: np.ndarray
    median_color_chip_size: float
    median_color_chip_width: float
    median_color_chip_height: float
    px_width: float | None
    px_height: float | None
    unit: str | None


@dataclass(frozen=True)
class AutomaticCardDetection:
    labeled_mask: torch.Tensor
    bounding_mask: torch.Tensor
    preprocessing: preprocessing.ColorCardPreprocessing
    contours: BinaryContourForest
    geometry: AutomaticCardGeometry
    roi_bounds_xyxy: tuple[int, int, int, int]
    opencv_version: str
    numpy_version: str
    reference_sources: tuple[tuple[str, str], ...]
    grid_box_sha256: str


@dataclass(frozen=True)
class AutomaticColorCorrection:
    detection: AutomaticCardDetection
    correction: ColorCorrectionResult


def _immutable(array):
    array = np.asarray(array)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _guarded(image, cancel):
    if cancel is not None and not callable(cancel):
        raise TypeError('cancel must be callable or None')
    if not torch.is_tensor(image):
        raise AutomaticCardError('Input must be a CUDA tensor')
    watched = [(image, preprocessing._signature(image))]
    def watch(tensor):
        watched.append((tensor, preprocessing._signature(tensor)))
    def guard():
        def intact():
            if any(preprocessing._signature(tensor)!=signature for tensor,signature in watched):
                raise AutomaticCardMutation('Input or reusable output changed during automatic card analysis')
        intact()
        try:cancelled=_checked(cancel,tuple(tensor for tensor,_ in watched))
        finally:intact()
        if cancelled:
            raise AutomaticCardCancelled('Automatic card analysis cancelled')
        return False
    return guard, watch


def _bounds(roi, height, width):
    if roi is None:
        return (0, 0, width, height)
    if (not isinstance(roi, (tuple, list)) or len(roi) != 4
            or any(type(v) is not int for v in roi)):
        raise AutomaticCardError('roi_bounds_xyxy must contain four integers')
    x0, y0, x1, y1 = roi
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise AutomaticCardError('ROI must be nonempty half-open bounds inside the image')
    return tuple(roi)


def _parameters(radius, min_size, aspect_ratio, solidity, color_chip_size):
    if type(radius) is not int or not 0 <= radius <= MAX_RADIUS:
        raise AutomaticCardResourceError(f'radius must be an integer in [0,{MAX_RADIUS}]')
    for value, name in ((min_size, 'min_size'), (aspect_ratio, 'aspect_ratio'), (solidity, 'solidity')):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise AutomaticCardError(f'{name} must be finite numeric data')
    if min_size < 0 or aspect_ratio <= 0 or not 0 <= solidity <= 1:
        raise AutomaticCardError('Require min_size>=0, aspect_ratio>0 and solidity in [0,1]')
    known = {'CLASSIC': (40, 40), 'PASSPORT': (12, 12), 'CAMERATRAX': (11, 11),
             'NANO': (4, 3), 'MINI': (12, 12)}
    if color_chip_size is None:
        return None
    if isinstance(color_chip_size, str):
        if color_chip_size.upper() not in known:
            raise AutomaticCardError('Unknown color_chip_size')
        return (*known[color_chip_size.upper()], True)
    if not isinstance(color_chip_size, (tuple, list)) or len(color_chip_size) != 2:
        raise AutomaticCardError('color_chip_size must be a known card or (width,height) in mm')
    try:
        dimensions = tuple(float(v) for v in color_chip_size)
    except (ValueError, TypeError, OverflowError) as error:
        raise AutomaticCardError('Invalid color chip dimensions') from error
    if any(not math.isfinite(v) or v <= 0 for v in dimensions):
        raise AutomaticCardError('Color chip dimensions must be positive finite numbers')
    return (*dimensions, False)


def _circle_budget(height, width, radius):
    if (24 * (2 * radius + 1) > MAX_ROW_WRITES
            or 24 * min(height, 2 * radius + 1) * min(width, 2 * radius + 1) > MAX_CIRCLE_WORK):
        raise AutomaticCardResourceError('Automatic card circle raster work bound exceeded')


def _preflight(height, width, bounds, method, block_size, radius, *, correction=False):
    x0, y0, x1, y1 = bounds
    local_h, local_w = y1 - y0, x1 - x0
    pixels, roi_pixels = height * width, local_h * local_w
    # Input3N, result mask8N+bboxN, reusable filters3R; ROI copy3R
    # may coexist. Auto correction additionally retains castmaskN+output3N.
    retained = (16 if correction else 12) * pixels + 6 * roi_pixels
    if retained > MAX_RETAINED_BYTES:
        raise AutomaticCardResourceError('Automatic card retained storage bound exceeded')
    preprocessing._parameters(method, block_size, preprocessing.MAX_TILE_SIDE, None)
    preprocessing._preflight(local_h, local_w, block_size, preprocessing.MAX_TILE_SIDE,
                             method=method, complete=True)
    _circle_budget(local_h, local_w, radius)


def _geometry(image, forest, min_size, aspect_ratio, solidity, guard):
    parent_guard=guard
    live={'forest':forest}
    def guard():
        return _checked(parent_guard,tuple(value for value in live.values() if torch.is_tensor(value)),
            tuple(value for value in live.values() if not torch.is_tensor(value)))
    if (GRID_BOX_411.dtype != np.dtype('<f4') or GRID_BOX_411.shape != (4, 2)
            or GRID_BOX_411.flags.writeable
            or hashlib.sha256(GRID_BOX_411.tobytes()).hexdigest() != GRID_BOX_SHA256):
        raise AutomaticCardError('Fixed grid metadata differs from pinned OpenCV 4.11.0')
    points = sum(len(c) for c in forest.contours)
    # Covers hull/rectangle/area passes and 24 point-polygon traversals per
    # selected contour, plus sorting compact candidate records.
    if 96 * points + 256 * len(forest.contours) > MAX_GEOMETRY_WORK:
        raise AutomaticCardResourceError('Automatic card geometry work bound exceeded')
    ratio_limit = max(aspect_ratio, 1) / min(aspect_ratio, 1)
    candidates = []
    for i, contour in enumerate(forest.contours):
        guard()
        area = cv2.contourArea(contour)
        if area <= min_size:
            continue
        rect = min_area_rect_411(contour, guard=guard)
        width, height = rect[1]
        if min(width, height) <= 0:
            raise AutomaticCardError('Degenerate candidate rectangle')
        if max(width, height) / min(width, height) < ratio_limit and area / np.prod(rect[1]) > solidity:
            candidates.append((i, contour, area, rect))
    if not candidates:
        raise AutomaticCardNotFound('No color card found')
    median = np.median([c[2] for c in candidates])
    selected = [c for c in candidates if 0.8 < c[2] / median < 1.2]
    if not selected:
        raise AutomaticCardNotFound('No color card found')
    live['selected']=selected
    stacked = np.vstack([c[1] for c in selected])
    bbox = tuple(cv2.boundingRect(stacked))
    del stacked
    centers = _immutable(np.concatenate([[np.array(c[3][0]).astype(int)] for c in selected]))
    live['detected_centers']=centers
    corners = np.intp(box_points_411(min_area_rect_411(centers, guard=guard)))
    live['corners']=corners
    height, width = map(int, image.shape[:2])
    if np.any(corners < 0) or np.any(corners[:, 0] > width) or np.any(corners[:, 1] > height):
        raise AutomaticCardNotFound('Color card corners could not be detected accurately')
    # The source's inclusive admission still fails when indexing exactly W/H.
    # Preserve that failure rather than clamp it to a different sampled pixel.
    if np.any(corners[:, 0] == width) or np.any(corners[:, 1] == height):
        raise AutomaticCardNotFound('Color card corner lies on the exclusive image edge')
    guard()
    indices = torch.as_tensor(corners.copy(), dtype=torch.int64, device=image.device)
    samples = image[indices[:, 1], indices[:, 0], :].cpu().numpy()
    live.update(indices=indices,samples=samples)
    guard()
    white_index = np.argmin([np.mean(math.dist(pixel, (255, 255, 255))) for pixel in samples])
    corners = corners[np.argsort([math.dist(corner, corners[white_index]) for corner in corners])[[0, 1, 3, 2]]]
    grid = np.array([(i * 100, j * 100) for j in range(6) for i in range(4)], dtype=np.int32)
    transformation = cv2.getPerspectiveTransform(GRID_BOX_411, corners.astype('float32'))
    if not np.isfinite(transformation).all():
        raise AutomaticCardError('Nonfinite color card transformation')
    new_centers = cv2.transform(np.array([grid]), transformation)[0][:, :2]
    live.update(corners=corners,grid=grid,transformation=transformation,new_centers=new_centers)
    # Source checks transformed (not detected) centers strictly inside each
    # retained contour. It does not require exactly 24 retained contours.
    for _, contour, _, _ in selected:
        guard()
        if sum(cv2.pointPolygonTest(contour, (int(pt[0]), int(pt[1])), False) == 1
               for pt in new_centers) != 1:
            raise AutomaticCardNotFound('Centers do not map 1 to 1 with detected color chips')
    guard()
    return selected, bbox, corners, new_centers, transformation


def _rasterize_geometry(height, width, bounds, bbox, centers, radius, device, guard):
    x0, y0, x1, y1 = bounds
    guard()
    labels = torch.zeros((height, width), dtype=torch.float64, device=device)
    bounding_mask = torch.zeros((height, width), dtype=torch.uint8, device=device)
    x, y, w, h = bbox
    # Source rectangle endpoint is (x+w,y+h), inclusive and clipped locally.
    bounding_mask[y0+y:y0+min(y+h+1,y1-y0), x0+x:x0+min(x+w+1,x1-x0)] = 255
    spans = _circle_row_spans(radius)
    for i, (cx, cy) in enumerate(centers):
        _checked(guard,(labels,bounding_mask),(bounds,bbox,centers))
        for row, (dy, left, right) in enumerate(spans):
            if row % 256 == 0:
                _checked(guard,(labels,bounding_mask),(bounds,bbox,centers))
            local_y = int(cy) + dy
            begin, end = max(0, int(cx)+left), min(x1-x0, int(cx)+right+1)
            if 0 <= local_y < y1-y0 and begin < end:
                labels[y0+local_y, x0+begin:x0+end] = (i + 1) * 10
    _checked(guard,(labels,bounding_mask),(bounds,bbox,centers))
    return labels, bounding_mask


def _scale_metadata(width, height, physical):
    if physical is None:
        return None, None, None
    known_width, known_height, named = physical
    px_width = max(known_width, known_height) / max(width, height) if named else known_width / width
    px_height = min(known_width, known_height) / min(width, height) if named else known_height / height
    return px_width, px_height, 'mm'


def detect_color_card_cuda(bgr, *, adaptive_method=1, block_size=51, radius=20,
        min_size=1000, aspect_ratio=1.27, solidity=.8, roi_bounds_xyxy=None,
        color_chip_size=None, cancel=None):
    """Detect a 4x6 card and return float64 CUDA labels and compact metadata.

    ROI bounds are explicitly half-open `(x0,y0,x1,y1)`. PlantCV's rectangle
    object stores x+w-1/y+h-1 but slices those coordinates exclusively; use those
    effective bounds when translating an existing rectangle. Full-frame edge
    bounds are allowed and correspond to source rectangular Objects directly.
    Scaling is returned as metadata, without changing global application units.
    """
    try:
        guard, watch = _guarded(bgr, cancel)
        guard()
        height, width = preprocessing._validate_source(bgr, bgr=True)
        bounds = _bounds(roi_bounds_xyxy, height, width)
        physical = _parameters(radius, min_size, aspect_ratio, solidity, color_chip_size)
        _preflight(height, width, bounds, adaptive_method, block_size, radius)
        x0, y0, x1, y1 = bounds
        local = bgr[y0:y1, x0:x1].contiguous()
        prep = preprocessing.preprocess_color_card_cuda(local, adaptive_method, block_size, cancel=guard)
        for tensor in (prep.gray, prep.blurred, prep.threshold):
            watch(tensor)
        forest = trace_binary_contours_cuda(prep.threshold, cancel=guard)
        selected, bbox, corners, centers, transformation = _geometry(
            local, forest, min_size, aspect_ratio, solidity, guard)
        labels, bounding_mask = _rasterize_geometry(height, width, bounds, bbox,
            centers, radius, bgr.device, guard)
        watch(labels)
        watch(bounding_mask)
        areas = tuple(float(c[2]) for c in selected)
        widths = tuple(float(c[3][1][0]) for c in selected)
        heights = tuple(float(c[3][1][1]) for c in selected)
        median_area, median_width, median_height = map(float, (np.median(areas), np.median(widths), np.median(heights)))
        px_width, px_height, unit = _scale_metadata(median_width, median_height, physical)
        geometry = AutomaticCardGeometry(tuple(c[0] for c in selected), areas, widths, heights,
            bbox, _immutable(corners), _immutable(centers), _immutable(transformation),
            median_area, median_width, median_height, px_width, px_height, unit)
        result = AutomaticCardDetection(labels, bounding_mask, prep, forest, geometry,
            bounds, cv2.__version__, np.__version__, REFERENCE_SOURCES, GRID_BOX_SHA256)
        _checked(guard,metadata=(result,corners,centers,transformation))
        return result
    except tensor_callback.TensorCallbackMutation as error:
        if isinstance(error,AutomaticCardMutation):raise
        raise AutomaticCardMutation(str(error)) from error


def mask_color_card_cuda(bgr, **kwargs):
    """Return the public full-frame automatic bounding-box mask on CUDA."""
    if 'roi_bounds_xyxy' in kwargs or 'color_chip_size' in kwargs:
        raise AutomaticCardError('mask_color_card accepts full-frame detector parameters')
    return detect_color_card_cuda(bgr, **kwargs).bounding_mask


def auto_correct_color_cuda(bgr, *, cancel=None, **kwargs):
    """Detect chips and apply the source affine fit to standard matrix pos=3."""
    try:
        guard, watch = _guarded(bgr, cancel)
        guard()
        height, width = preprocessing._validate_source(bgr, bgr=True)
        bounds = _bounds(kwargs.get('roi_bounds_xyxy'), height, width)
        _parameters(kwargs.get('radius', 20), kwargs.get('min_size', 1000),
                    kwargs.get('aspect_ratio', 1.27), kwargs.get('solidity', .8), kwargs.get('color_chip_size'))
        _preflight(height, width, bounds, kwargs.get('adaptive_method', 1),
                   kwargs.get('block_size', 51), kwargs.get('radius', 20), correction=True)
        detection = detect_color_card_cuda(bgr, cancel=guard, **kwargs)
        for tensor in (detection.labeled_mask, detection.bounding_mask, detection.preprocessing.gray,
                       detection.preprocessing.blurred, detection.preprocessing.threshold):
            watch(tensor)
        _, source = get_color_matrix_cuda(bgr, detection.labeled_mask.to(torch.uint8), cancel=guard)
        target = std_color_matrix(pos=3, device=bgr.device).cpu().numpy()
        _checked(guard,metadata=(detection,source,target))
        raw = affine_color_correction_cuda(bgr, source, target, cancel=guard)
        watch(raw.corrected_bgr)
        correction = ColorCorrectionResult(raw.corrected_bgr, _immutable(raw.source_matrix),
            _immutable(raw.target_matrix), _immutable(raw.transformation_matrix), raw.fit, raw.numpy_version)
        result = AutomaticColorCorrection(detection, correction)
        _checked(guard,metadata=(result,raw,source,target))
        return result
    except tensor_callback.TensorCallbackMutation as error:
        if isinstance(error,AutomaticCardMutation):raise
        raise AutomaticCardMutation(str(error)) from error
