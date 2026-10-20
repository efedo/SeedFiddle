"""Bounded compact scanline ports for OpenCV 4.11 drawing.cpp rasterization.

Only integer points and inclusive row intervals are retained. No frame-sized or
object-bounds CPU raster is created. See docs/OPENCV_MATH_NOTICES.md.
https://raw.githubusercontent.com/opencv/opencv/4.11.0/modules/imgproc/src/drawing.cpp
"""
from __future__ import annotations

import math
import sys
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from itertools import chain

MAX_RASTER_PIXELS = 64_000_000
MAX_RASTER_SPAN_ROWS = 1_000_000
MAX_RASTER_SPAN_SEGMENTS = 1_000_000
MAX_RASTER_SPAN_POINTS = 1_000_000
MAX_RASTER_SPAN_WORK = 16_000_000
XY_SHIFT = 16
XY_ONE = 1 << XY_SHIFT
_INT32_MIN = -(1 << 31)
_INT32_MAX = (1 << 31) - 1
_INT64_MIN = -(1 << 63)
_INT64_MAX = (1 << 63) - 1


class RasterSpanLimitError(ValueError):
    """Compact raster work exceeded an explicit per-call bound."""


class RasterSpanCancelled(RuntimeError):
    """Compact raster work was cancelled by the caller."""


def _poll_cancel(cancel: Callable[[], bool] | None, work: int) -> None:
    if cancel is not None and work % 4096 == 0 and cancel():
        raise RasterSpanCancelled("OpenCV drawing-span work was cancelled")


def _frame(width: int, height: int) -> tuple[int, int]:
    width, height = int(width), int(height)
    if width < 1 or height < 1 or width > _INT32_MAX or height > _INT32_MAX:
        raise ValueError("raster dimensions must be positive OpenCV int values")
    if width * height > MAX_RASTER_PIXELS:
        raise RasterSpanLimitError("raster exceeds the 64-million-pixel bound")
    if height > MAX_RASTER_SPAN_ROWS:
        raise RasterSpanLimitError("raster height exceeds the scan-row bound")
    return width, height


def _trunc_div(numerator: int, denominator: int) -> int:
    """C++ signed integer division, truncating toward zero."""
    if denominator == 0:
        raise ZeroDivisionError("integer division by zero")
    q = abs(numerator) // abs(denominator)
    return -q if (numerator < 0) != (denominator < 0) else q


def _clip_delta(a: int, b: int, denominator: int) -> int:
    """Match drawing.cpp's double expression followed by int64 truncation."""
    if denominator == 0:
        raise ZeroDivisionError("clipLine received a parallel clipping edge")
    return int(float(a) * float(b) / float(denominator))


def _clip_line_2l(width: int, height: int, first: tuple[int, int], last: tuple[int, int]):
    """OpenCV drawing.cpp clipLine(Size2l) for integer coordinates."""
    if width <= 0 or height <= 0:
        return False, first, last
    x1, y1 = map(int, first)
    x2, y2 = map(int, last)
    if any(v < _INT64_MIN or v > _INT64_MAX for v in (x1, y1, x2, y2)):
        raise RasterSpanLimitError("clipLine coordinates exceed OpenCV int64 range")
    right, bottom = width - 1, height - 1

    def code(x: int, y: int) -> int:
        return int(x < 0) | (int(x > right) << 1) | (int(y < 0) << 2) | (int(y > bottom) << 3)

    c1, c2 = code(x1, y1), code(x2, y2)
    if (c1 & c2) == 0 and (c1 | c2) != 0:
        if c1 & 12:
            edge_y = 0 if c1 < 8 else bottom
            x1 += _clip_delta(edge_y - y1, x2 - x1, y2 - y1)
            y1 = edge_y
            c1 = int(x1 < 0) | (int(x1 > right) << 1)
        if c2 & 12:
            edge_y = 0 if c2 < 8 else bottom
            x2 += _clip_delta(edge_y - y2, x2 - x1, y2 - y1)
            y2 = edge_y
            c2 = int(x2 < 0) | (int(x2 > right) << 1)
        if (c1 & c2) == 0 and (c1 | c2) != 0:
            if c1:
                edge_x = 0 if c1 == 1 else right
                y1 += _clip_delta(edge_x - x1, y2 - y1, x2 - x1)
                x1 = edge_x
                c1 = 0
            if c2:
                edge_x = 0 if c2 == 1 else right
                y2 += _clip_delta(edge_x - x2, y2 - y1, x2 - x1)
                x2 = edge_x
                c2 = 0
    return (c1 | c2) == 0, (x1, y1), (x2, y2)


def line8_points(width: int, height: int, start: tuple[int, int], end: tuple[int, int], *,
                 cancel: Callable[[], bool] | None = None) -> tuple[tuple[int, int], ...]:
    """OpenCV LineIterator 8-connectivity pixels after full-frame clipLine."""
    width, height = _frame(width, height)
    x0, y0 = map(int, start)
    x1, y1 = map(int, end)
    if any(v < _INT32_MIN or v > _INT32_MAX for v in (x0, y0, x1, y1)):
        raise RasterSpanLimitError("Line8 endpoint exceeds OpenCV Point int range")
    visible, first, last = _clip_line_2l(width, height, (x0, y0), (x1, y1))
    if not visible:
        return ()
    x0, y0 = first
    x1, y1 = last
    if x1 < x0:
        x0, y0, x1, y1 = x1, y1, x0, y0
    dx, signed_dy = x1 - x0, y1 - y0
    sy, dy = (1 if signed_dy >= 0 else -1), abs(signed_dy)
    major = max(dx, dy)
    if major + 1 > MAX_RASTER_SPAN_POINTS:
        raise RasterSpanLimitError("Line8 exceeds the point bound")
    steep = dy > dx
    major, minor = (dy, dx) if steep else (dx, dy)
    err = major - 2 * minor
    x, y = x0, y0
    output: list[tuple[int, int]] = []
    for index in range(major + 1):
        _poll_cancel(cancel, index)
        output.append((x, y))
        if index == major:
            break
        diagonal = err < 0
        err += -2 * minor + (2 * major if diagonal else 0)
        if steep:
            y += sy
            if diagonal:
                x += 1
        else:
            x += 1
            if diagonal:
                y += sy
    return tuple(output)


def line2_points(width: int, height: int, start_fixed: tuple[int, int], end_fixed: tuple[int, int], *,
                 cancel: Callable[[], bool] | None = None) -> tuple[tuple[int, int], ...]:
    """OpenCV drawing.cpp Line2 for 16-bit fixed-point endpoints."""
    width, height = _frame(width, height)
    p0, p1 = tuple(map(int, start_fixed)), tuple(map(int, end_fixed))
    if any(v < _INT64_MIN or v > _INT64_MAX for v in (*p0, *p1)):
        raise RasterSpanLimitError("Line2 endpoint exceeds OpenCV int64 range")
    visible, p0, p1 = _clip_line_2l(width << XY_SHIFT, height << XY_SHIFT, p0, p1)
    if not visible:
        return ()
    x0, y0 = p0
    x1, y1 = p1
    dx, dy = x1 - x0, y1 - y0
    ax, ay = abs(dx), abs(dy)
    if ax > ay:
        if dx < 0:
            dy = -dy
            x0, y0, x1, y1 = x1, y1, x0, y0
        x_step = XY_ONE
        y_step = _trunc_div(dy * XY_ONE, ax | 1)
        count = (x1 - x0) >> XY_SHIFT
        x_major = True
    else:
        if dy < 0:
            dx = -dx
            x0, y0, x1, y1 = x1, y1, x0, y0
        x_step = _trunc_div(dx * XY_ONE, ay | 1)
        y_step = XY_ONE
        count = (y1 - y0) >> XY_SHIFT
        x_major = False
    if count < 0 or count + 2 > MAX_RASTER_SPAN_POINTS:
        raise RasterSpanLimitError("Line2 exceeds the point bound")
    x, y = x0 + (XY_ONE >> 1), y0 + (XY_ONE >> 1)
    output: list[tuple[int, int]] = []

    def emit(px: int, py: int) -> None:
        px, py = int(px), int(py)
        if 0 <= px < width and 0 <= py < height:
            output.append((px, py))

    emit((x1 + (XY_ONE >> 1)) >> XY_SHIFT, (y1 + (XY_ONE >> 1)) >> XY_SHIFT)
    for index in range(count + 1):
        _poll_cancel(cancel, index)
        if x_major:
            emit(x >> XY_SHIFT, y >> XY_SHIFT)
            x = ((x >> XY_SHIFT) + 1) << XY_SHIFT
            y += y_step
        else:
            emit(x >> XY_SHIFT, y >> XY_SHIFT)
            y = ((y >> XY_SHIFT) + 1) << XY_SHIFT
            x += x_step
    return tuple(output)


def normalize_spans(spans: Iterable[tuple[int, int, int]], *, width: int, height: int,
                    cancel: Callable[[], bool] | None = None,
                    max_work: int = MAX_RASTER_SPAN_SEGMENTS) -> tuple[tuple[int, int, int], ...]:
    """Clip and merge inclusive (y,x0,x1) intervals; every input item is charged."""
    width, height = _frame(width, height)
    rows: dict[int, list[tuple[int, int]]] = defaultdict(list)
    work = 0
    for y, x0, x1 in spans:
        _poll_cancel(cancel, work)
        work += 1
        if work > max_work:
            raise RasterSpanLimitError("scanline span input work exceeds the bound")
        y, x0, x1 = int(y), int(x0), int(x1)
        if 0 <= y < height and x1 >= 0 and x0 < width:
            x0, x1 = max(0, x0), min(width - 1, x1)
            if x0 <= x1:
                rows[y].append((x0, x1))
    result: list[tuple[int, int, int]] = []
    for row_index, y in enumerate(sorted(rows)):
        _poll_cancel(cancel, row_index)
        intervals = sorted(rows[y])
        left, right = intervals[0]
        for next_left, next_right in intervals[1:]:
            if next_left <= right + 1:
                right = max(right, next_right)
            else:
                result.append((y, left, right))
                left, right = next_left, next_right
        result.append((y, left, right))
        if len(result) > MAX_RASTER_SPAN_SEGMENTS:
            raise RasterSpanLimitError("normalized span count exceeds the bound")
    return tuple(result)


def _point_spans(points: Iterable[tuple[int, int]]) -> list[tuple[int, int, int]]:
    return [(int(y), int(x), int(x)) for x, y in points]


def fill_convex_poly_spans(vertices: Sequence[tuple[int, int]], width: int, height: int, *,
                           shift: int = XY_SHIFT,
                           cancel: Callable[[], bool] | None = None) -> tuple[tuple[int, int, int], ...]:
    """OpenCV 4.11 FillConvexPoly spans for LINE_8 and 0..16 fixed-point shift."""
    width, height = _frame(width, height)
    if not 0 <= shift <= XY_SHIFT:
        raise ValueError("polygon shift must be between zero and 16")
    vertices = tuple((int(x), int(y)) for x, y in vertices)
    n = len(vertices)
    if n > MAX_RASTER_SPAN_SEGMENTS:
        raise RasterSpanLimitError("convex polygon vertex count exceeds the bound")
    scale = 1 << (XY_SHIFT - shift)
    fixed = tuple((x * scale, y * scale) for x, y in vertices)
    outline: list[tuple[int, int, int]] = []
    line_work = 0
    for i in range(n):
        _poll_cancel(cancel, i)
        p0, p1 = fixed[i - 1], fixed[i]
        if shift == 0:
            points = line8_points(width, height, (p0[0] >> XY_SHIFT, p0[1] >> XY_SHIFT),
                                  (p1[0] >> XY_SHIFT, p1[1] >> XY_SHIFT), cancel=cancel)
        else:
            points = line2_points(width, height, p0, p1, cancel=cancel)
        line_work += len(points)
        if line_work > MAX_RASTER_SPAN_POINTS:
            raise RasterSpanLimitError("convex polygon outline exceeds the point bound")
        outline.extend(_point_spans(points))
        if len(outline) > MAX_RASTER_SPAN_SEGMENTS:
            raise RasterSpanLimitError("convex polygon outline span count exceeds the bound")
    if n < 3:
        return normalize_spans(outline, width=width, height=height, cancel=cancel)
    delta = (1 << shift) >> 1
    rounded = tuple(((x + delta) >> shift, (y + delta) >> shift) for x, y in vertices)
    xmin, xmax = min(p[0] for p in rounded), max(p[0] for p in rounded)
    ymin, ymax = min(p[1] for p in rounded), max(p[1] for p in rounded)
    if xmax < 0 or ymax < 0 or xmin >= width or ymin >= height:
        return normalize_spans(outline, width=width, height=height, cancel=cancel)
    ymax = min(ymax, height - 1)
    if ymax - ymin + 1 > MAX_RASTER_SPAN_ROWS:
        raise RasterSpanLimitError("convex polygon scan rows exceed the bound")
    imin = min(range(n), key=lambda i: vertices[i][1])
    edge_idx, edge_dir = [imin, imin], [1, n - 1]
    edge_y_end, edge_x, edge_dx = [ymin, ymin], [-XY_ONE, -XY_ONE], [0, 0]
    fill: list[tuple[int, int, int]] = []
    for row_index, y in enumerate(range(ymin, ymax + 1)):
        _poll_cancel(cancel, row_index)
        found = [False, False]
        for side in range(2):
            if y < edge_y_end[side]:
                found[side] = True
                continue
            idx0 = edge_idx[side]
            idx = (idx0 + edge_dir[side]) % n
            for _ in range(n):
                ty = (vertices[idx][1] + delta) >> shift
                if ty > y:
                    xs, xe = fixed[idx0][0], fixed[idx][0]
                    dy = ty - y
                    edge_dx[side] = _trunc_div((xe - xs) * 2 + dy, 2 * dy)
                    edge_x[side] = xs
                    edge_y_end[side] = ty
                    edge_idx[side] = idx
                    found[side] = True
                    break
                idx0 = idx
                idx = (idx + edge_dir[side]) % n
        if found[0] and found[1] and 0 <= y < height:
            left, right = (0, 1) if edge_x[0] <= edge_x[1] else (1, 0)
            x0 = (edge_x[left] + (XY_ONE >> 1)) >> XY_SHIFT
            x1 = (edge_x[right] + (XY_ONE >> 1)) >> XY_SHIFT
            if x1 >= 0 and x0 < width:
                x0, x1 = max(0, x0), min(width - 1, x1)
                if x0 <= x1:
                    fill.append((y, x0, x1))
                    if len(outline) + len(fill) > MAX_RASTER_SPAN_SEGMENTS:
                        raise RasterSpanLimitError("convex polygon span count exceeds the bound")
        edge_x[0] += edge_dx[0]
        edge_x[1] += edge_dx[1]
    return normalize_spans(chain(outline, fill), width=width, height=height, cancel=cancel)


def filled_circle_spans(width: int, height: int, center: tuple[int, int], radius: int, *,
                        cancel: Callable[[], bool] | None = None) -> tuple[tuple[int, int, int], ...]:
    """OpenCV drawing.cpp Circle(fill=1) intervals, clipped at actual frame edges."""
    width, height = _frame(width, height)
    cx, cy = map(int, center)
    radius = int(radius)
    if radius < 0:
        raise ValueError("circle radius must be nonnegative")
    if radius + 1 > MAX_RASTER_SPAN_ROWS or 4 * (radius + 1) > MAX_RASTER_SPAN_SEGMENTS:
        raise RasterSpanLimitError("circle row work exceeds the bound")
    err, dx, dy, plus, minus = 0, radius, 0, 1, (radius << 1) - 1
    raw: list[tuple[int, int, int]] = []
    iteration = 0
    while dx >= dy:
        _poll_cancel(cancel, iteration)
        raw.extend(((cy-dy, cx-dx, cx+dx), (cy+dy, cx-dx, cx+dx),
                    (cy-dx, cx-dy, cx+dy), (cy+dx, cx-dy, cx+dy)))
        dy += 1
        err += plus
        plus += 2
        mask = 0 if err <= 0 else -1
        err -= minus & mask
        dx += mask
        minus -= mask & 2
        iteration += 1
    return normalize_spans(raw, width=width, height=height, cancel=cancel)


def circle_outline_spans(width, height, center, radius, *, cancel=None, work_budget=None):
    """OpenCV 4.11 Circle(fill=0), LINE_8 and integer coordinates."""
    width, height = _frame(width, height)
    cx, cy = map(int, center)
    if isinstance(radius, bool) or not isinstance(radius, int) or radius < 0:
        raise ValueError('circle radius must be a nonnegative integer')
    if radius + 1 > MAX_RASTER_SPAN_ROWS or 8 * (radius + 1) > MAX_RASTER_SPAN_SEGMENTS:
        raise RasterSpanLimitError('circle outline work exceeds the bound')
    if work_budget is None:
        work_budget = [0]
    _charge_geometry_work(work_budget, 8 * (radius + 1), cancel)
    err, dx, dy, plus, minus = 0, radius, 0, 1, (radius << 1) - 1
    raw = []
    iteration = 0
    while dx >= dy:
        _poll_cancel(cancel, iteration)
        for x, y in ((cx-dx, cy-dy), (cx+dx, cy-dy),
                     (cx-dx, cy+dy), (cx+dx, cy+dy),
                     (cx-dy, cy-dx), (cx+dy, cy-dx),
                     (cx-dy, cy+dx), (cx+dy, cy+dx)):
            if 0 <= x < width and 0 <= y < height:
                raw.append((y, x, x))
        dy += 1
        err += plus
        plus += 2
        mask = 0 if err <= 0 else -1
        err -= minus & mask
        dx += mask
        minus -= mask & 2
        iteration += 1
    _charge_geometry_work(work_budget, max(1, len(raw)) * max(1, len(raw).bit_length()), cancel)
    return normalize_spans(raw, width=width, height=height, cancel=cancel,
                           max_work=MAX_RASTER_SPAN_WORK)


def thick_line_spans(width: int, height: int, start: tuple[int, int], end: tuple[int, int], *,
                     thickness: int = 5,
                     cancel: Callable[[], bool] | None = None) -> tuple[tuple[int, int, int], ...]:
    """OpenCV 4.11 ThickLine for integer points, LINE_8, flags=3, shift=0."""
    width, height = _frame(width, height)
    if thickness <= 0 or thickness > 32767:
        raise ValueError("line thickness must be in [1, 32767]")
    x0, y0 = map(int, start)
    x1, y1 = map(int, end)
    if any(v < _INT32_MIN or v > _INT32_MAX for v in (x0, y0, x1, y1)):
        raise RasterSpanLimitError("line endpoint exceeds OpenCV Point int range")
    if thickness == 1:
        return normalize_spans(_point_spans(line8_points(width, height, (x0,y0), (x1,y1), cancel=cancel)),
                               width=width, height=height, cancel=cancel)
    p0, p1 = (x0 << XY_SHIFT, y0 << XY_SHIFT), (x1 << XY_SHIFT, y1 << XY_SHIFT)
    dx, dy = float(x0 - x1), float(y1 - y0)
    length2 = dx * dx + dy * dy
    thickness_fixed = thickness << (XY_SHIFT - 1)
    odd = thickness & 1
    raw: list[tuple[int, int, int]] = []
    if abs(length2) > sys.float_info.epsilon:
        scale = (thickness_fixed + odd * XY_ONE * 0.5) / math.sqrt(length2)
        dpx, dpy = int(round(dy * scale)), int(round(dx * scale))
        quad = ((p0[0]+dpx,p0[1]+dpy),(p0[0]-dpx,p0[1]-dpy),
                (p1[0]-dpx,p1[1]-dpy),(p1[0]+dpx,p1[1]+dpy))
        raw.extend(fill_convex_poly_spans(quad, width, height, shift=XY_SHIFT, cancel=cancel))
    radius = (thickness_fixed + (XY_ONE >> 1)) >> XY_SHIFT
    for index, point in enumerate((p0, p1)):
        _poll_cancel(cancel, index)
        center = ((point[0] + (XY_ONE >> 1)) >> XY_SHIFT,
                  (point[1] + (XY_ONE >> 1)) >> XY_SHIFT)
        raw.extend(filled_circle_spans(width, height, center, radius, cancel=cancel))
        if len(raw) > MAX_RASTER_SPAN_SEGMENTS:
            raise RasterSpanLimitError("thick-line span count exceeds the bound")
    return normalize_spans(raw, width=width, height=height, cancel=cancel)


def _charge_geometry_work(work_budget, amount, cancel):
    if cancel is not None and cancel():
        raise RasterSpanCancelled('OpenCV drawing-span work was cancelled')
    amount = max(1, int(amount))
    if work_budget[0] + amount > MAX_RASTER_SPAN_WORK:
        raise RasterSpanLimitError('aggregate drawing geometry work exceeds the bound')
    work_budget[0] += amount


def thick_line_fixed_spans(width, height, start_fixed, end_fixed, *, thickness=5,
                           flags=3, cancel=None, work_budget=None):
    """OpenCV 4.11 ThickLine for endpoints in 16.16 fixed-point units."""
    width, height = _frame(width, height)
    if (isinstance(thickness, bool) or not isinstance(thickness, int)
            or thickness < 1 or thickness > 32767 or flags not in (0, 1, 2, 3)):
        raise ValueError('fixed thick-line arguments are outside the supported source domain')
    if len(start_fixed) != 2 or len(end_fixed) != 2:
        raise ValueError('fixed line endpoints require exactly two coordinates')
    if work_budget is None:
        work_budget = [0]
    p0, p1 = tuple(map(int, start_fixed)), tuple(map(int, end_fixed))
    if any(v < _INT64_MIN or v > _INT64_MAX for v in (*p0, *p1)):
        raise RasterSpanLimitError('fixed line endpoint exceeds OpenCV int64 range')
    dx_px = (abs(p0[0] - p1[0]) + XY_ONE - 1) // XY_ONE
    dy_px = (abs(p0[1] - p1[1]) + XY_ONE - 1) // XY_ONE
    # Charge unclipped line/row work before allocating any geometry.
    _charge_geometry_work(work_budget, 4 * (dx_px + dy_px + 2 * thickness + 4), cancel)
    if thickness == 1:
        raw = _point_spans(line2_points(width, height, p0, p1, cancel=cancel))
        _charge_geometry_work(work_budget,
            max(1, len(raw)) * max(1, len(raw).bit_length()), cancel)
        return normalize_spans(raw, width=width, height=height, cancel=cancel,
                               max_work=MAX_RASTER_SPAN_WORK)
    dx = float(p0[0] - p1[0]) * (1.0 / XY_ONE)
    dy = float(p1[1] - p0[1]) * (1.0 / XY_ONE)
    length2 = dx * dx + dy * dy
    thickness_fixed = int(thickness) << (XY_SHIFT - 1)
    odd = int(thickness) & 1
    raw = []
    if abs(length2) > sys.float_info.epsilon:
        scale = (thickness_fixed + odd * XY_ONE * 0.5) / math.sqrt(length2)
        dpx, dpy = int(round(dy * scale)), int(round(dx * scale))
        quad = ((p0[0] + dpx, p0[1] + dpy), (p0[0] - dpx, p0[1] - dpy),
                (p1[0] - dpx, p1[1] - dpy), (p1[0] + dpx, p1[1] + dpy))
        if any(v < _INT64_MIN or v > _INT64_MAX for point in quad for v in point):
            raise RasterSpanLimitError('fixed thick-line quad exceeds OpenCV int64 range')
        raw.extend(fill_convex_poly_spans(quad, width, height, shift=XY_SHIFT, cancel=cancel))
    if cancel is not None and cancel():
        raise RasterSpanCancelled('OpenCV drawing-span work was cancelled')
    radius = (thickness_fixed + (XY_ONE >> 1)) >> XY_SHIFT
    for bit, point in ((1, p0), (2, p1)):
        if flags & bit:
            center = ((point[0] + (XY_ONE >> 1)) >> XY_SHIFT,
                      (point[1] + (XY_ONE >> 1)) >> XY_SHIFT)
            raw.extend(filled_circle_spans(width, height, center, radius, cancel=cancel))
            if len(raw) > MAX_RASTER_SPAN_SEGMENTS:
                raise RasterSpanLimitError('fixed thick-line span count exceeds the bound')
    _charge_geometry_work(work_budget, max(1, len(raw)) * max(1, len(raw).bit_length()), cancel)
    return normalize_spans(raw, width=width, height=height, cancel=cancel,
                           max_work=MAX_RASTER_SPAN_WORK)


def closed_polyline_spans(vertices, width, height, *, thickness=5, cancel=None, work_budget=None):
    """OpenCV 4.11 drawContours closed LINE_8 stroke (each edge flags=2)."""
    width, height = _frame(width, height)
    if (isinstance(thickness, bool) or not isinstance(thickness, int)
            or not 1 <= thickness <= 32767):
        raise ValueError('line thickness must be an integer in [1, 32767]')
    if not hasattr(vertices, '__len__') or len(vertices) > MAX_RASTER_SPAN_POINTS:
        raise RasterSpanLimitError('polyline point count exceeds the bound')
    points = tuple((int(x), int(y)) for x, y in vertices)
    if not points:
        return ()
    if work_budget is None:
        work_budget = [0]
    raw = []
    for index, first in enumerate(points):
        if index % 256 == 0 and cancel is not None and cancel():
            raise RasterSpanCancelled('OpenCV drawing-span work was cancelled')
        last = points[(index + 1) % len(points)]
        if any(v < _INT32_MIN or v > _INT32_MAX for v in (*first, *last)):
            raise RasterSpanLimitError('contour point exceeds OpenCV Point int range')
        if thickness == 1:
            _charge_geometry_work(work_budget,
                abs(first[0]-last[0]) + abs(first[1]-last[1]) + 1, cancel)
            edge = _point_spans(line8_points(width, height, first, last, cancel=cancel))
        else:
            edge = thick_line_fixed_spans(width, height,
                (first[0] << XY_SHIFT, first[1] << XY_SHIFT),
                (last[0] << XY_SHIFT, last[1] << XY_SHIFT),
                thickness=thickness, flags=2, cancel=cancel, work_budget=work_budget)
        if len(raw) + len(edge) > MAX_RASTER_SPAN_SEGMENTS:
            raise RasterSpanLimitError('aggregate contour stroke spans exceed the bound')
        raw.extend(edge)
    _charge_geometry_work(work_budget, max(1, len(raw)) * max(1, len(raw).bit_length()), cancel)
    return normalize_spans(raw, width=width, height=height, cancel=cancel,
                           max_work=MAX_RASTER_SPAN_WORK)


def iter_adjacent_closed_polyline_spans(vertices, width, height, *, thickness=5,
                                       cancel=None, work_budget=None, batch_edges=256,
                                       cache_directions=False):
    """Bounded batches of the same closed stroke for CHAIN_APPROX_NONE points.

    Every edge, including closure, must be 8-neighbor adjacent. Painting all
    batches with one color equals painting their union. At most8192 estimated
    edge rows are retained per batch, plus one primitive's temporary geometry.
    A single edge may exceed that target; its rows remain bounded by the frame.
    Optional direction reuse retains at most nine interior edge templates;
    their rows are bounded by the fixed thickness and the source frame.
    Raw span and work limits cover the whole contour, not each yielded batch.
    """
    width, height = _frame(width, height)
    if type(thickness) is not int or not 1 <= thickness <= 32767:
        raise ValueError('line thickness must be an integer in [1, 32767]')
    if type(batch_edges) is not int or not 1 <= batch_edges <= 4096:
        raise ValueError('batch_edges must be an integer in [1, 4096]')
    if type(cache_directions) is not bool:
        raise ValueError('cache_directions must be a bool')
    if not hasattr(vertices, '__len__') or len(vertices) > MAX_RASTER_SPAN_POINTS:
        raise RasterSpanLimitError('polyline point count exceeds the bound')
    points = tuple((int(x), int(y)) for x, y in vertices)
    if not points:
        return
    if work_budget is None:
        work_budget = [0]
    rows = min(height, thickness + 4)
    batch_edges = min(batch_edges, max(1, 8192 // rows))
    raw, raw_count, templates = [], 0, {}
    for index, first in enumerate(points):
        last = points[(index + 1) % len(points)]
        if (any(v < _INT32_MIN or v > _INT32_MAX for v in (*first, *last))
                or max(abs(first[0]-last[0]), abs(first[1]-last[1])) > 1):
            raise ValueError('batched contour requires adjacent OpenCV int points')
        if thickness == 1:
            _charge_geometry_work(work_budget,
                abs(first[0]-last[0]) + abs(first[1]-last[1]) + 1, cancel)
            edge = _point_spans(line8_points(width, height, first, last, cancel=cancel))
        else:
            direction = (last[0]-first[0], last[1]-first[1])
            # Interior adjacent edges differ only by integer translation.
            # Keep at most nine compact templates; boundary edges use the
            # original primitive because its clipping can affect rounding.
            margin = thickness + 2
            interior = cache_directions and all(margin <= x < width-margin and margin <= y < height-margin
                for x, y in (first, last))
            template = templates.get(direction) if interior else None
            if template is None:
                edge = thick_line_fixed_spans(width, height,
                    (first[0] << XY_SHIFT, first[1] << XY_SHIFT),
                    (last[0] << XY_SHIFT, last[1] << XY_SHIFT),
                    thickness=thickness, flags=2, cancel=cancel, work_budget=work_budget)
                if interior:
                    templates[direction] = tuple((y-first[1], x0-first[0], x1-first[0])
                        for y, x0, x1 in edge)
            else:
                _charge_geometry_work(work_budget, len(template), cancel)
                edge = tuple((y+first[1], x0+first[0], x1+first[0])
                    for y, x0, x1 in template)
        raw_count += len(edge)
        if raw_count > MAX_RASTER_SPAN_SEGMENTS:
            raise RasterSpanLimitError('aggregate contour stroke spans exceed the bound')
        raw.extend(edge)
        del edge
        if (index + 1) % batch_edges == 0 or index + 1 == len(points):
            _charge_geometry_work(work_budget,
                max(1, len(raw)) * max(1, len(raw).bit_length()), cancel)
            normalized = normalize_spans(raw, width=width, height=height, cancel=cancel,
                                         max_work=MAX_RASTER_SPAN_WORK)
            raw = []
            yield normalized
            del normalized


def _insert_active_edge_bounded(active, entry, *, work: int,
                                cancel: Callable[[], bool] | None) -> int:
    """Preserve insertion-before-equals while charging search and list movement."""
    insert_at = 0
    while insert_at < len(active):
        if work >= MAX_RASTER_SPAN_WORK:
            raise RasterSpanLimitError("polygon active-edge work exceeds the bound")
        work += 1
        _poll_cancel(cancel, work)
        if not active[insert_at][1] < entry[1]:
            break
        insert_at += 1
    # Charge suffix movement before Python's list insertion mutates the row.
    shift_cost = len(active) - insert_at
    if work + shift_cost > MAX_RASTER_SPAN_WORK:
        raise RasterSpanLimitError("polygon active-edge insertion work exceeds the bound")
    previous_work = work
    work += shift_cost
    if cancel is not None and work // 4096 > previous_work // 4096:
        _poll_cancel(cancel, (previous_work // 4096 + 1) * 4096)
    active.insert(insert_at, entry)
    return work


def fill_polygon_spans(vertices: Sequence[tuple[int, int]], width: int, height: int, *,
                       cancel: Callable[[], bool] | None = None) -> tuple[tuple[int, int, int], ...]:
    """OpenCV 4.11 CollectPolyEdges/FillEdgeCollection spans, LINE_8, shift=0."""
    width, height = _frame(width, height)
    vertices = tuple((int(x), int(y)) for x, y in vertices)
    n = len(vertices)
    if n > MAX_RASTER_SPAN_SEGMENTS:
        raise RasterSpanLimitError("polygon vertex count exceeds the bound")
    boundary: list[tuple[int, int, int]] = []
    edges: list[list[int]] = []
    points_seen = 0
    for index in range(n):
        _poll_cancel(cancel, index)
        x0,y0 = vertices[index - 1]
        x1,y1 = vertices[index]
        if any(v < _INT32_MIN or v > _INT32_MAX for v in (x0,y0,x1,y1)):
            raise RasterSpanLimitError("polygon vertex exceeds OpenCV Point int range")
        points = line8_points(width, height, (x0,y0), (x1,y1), cancel=cancel)
        points_seen += len(points)
        if points_seen > MAX_RASTER_SPAN_POINTS:
            raise RasterSpanLimitError("polygon outline exceeds the point bound")
        boundary.extend(_point_spans(points))
        if len(boundary) > MAX_RASTER_SPAN_SEGMENTS:
            raise RasterSpanLimitError("polygon outline span count exceeds the bound")
        if y0 == y1:
            continue
        pt0c_x, pt0c_y = x0 << XY_SHIFT, y0
        pt1c_x, pt1c_y = x1 << XY_SHIFT, y1
        if not (0 <= x0 < width and 0 <= x1 < width and 0 <= y0 < height and 0 <= y1 < height):
            _, clipped0, clipped1 = _clip_line_2l(width, height, (x0,y0), (x1,y1))
            if clipped0[1] != clipped1[1]:
                pt0c_y, pt1c_y = clipped0[1], clipped1[1]
            pt0c_x, pt1c_x = clipped0[0] << XY_SHIFT, clipped1[0] << XY_SHIFT
        edge_dx = _trunc_div(pt1c_x - pt0c_x, pt1c_y - pt0c_y)
        if y0 < y1:
            ey0, ey1 = y0, y1
            ex = pt0c_x + (y0 - pt0c_y) * edge_dx
        else:
            ey0, ey1 = y1, y0
            ex = pt1c_x + (y1 - pt1c_y) * edge_dx
        edges.append([ey0, ey1, ex, edge_dx])
    if len(edges) < 2:
        return normalize_spans(boundary, width=width, height=height, cancel=cancel)
    edges.sort(key=lambda edge: (edge[0], edge[2], edge[3]))
    ymin = min(edge[0] for edge in edges)
    ymax = min(max(edge[1] for edge in edges), height)
    if ymax <= ymin:
        return normalize_spans(boundary, width=width, height=height, cancel=cancel)
    if ymax - ymin > MAX_RASTER_SPAN_ROWS:
        raise RasterSpanLimitError("polygon scan rows exceed the bound")
    x_min = min(min(e[2], e[2] + (e[1]-e[0])*e[3]) for e in edges)
    x_max = max(max(e[2], e[2] + (e[1]-e[0])*e[3]) for e in edges)
    if ymax < 0 or ymin >= height or x_max < 0 or x_min >= (width << XY_SHIFT):
        return normalize_spans(boundary, width=width, height=height, cancel=cancel)
    fill: list[tuple[int, int, int]] = []
    active: list[list[int]] = []
    next_edge = 0
    work = 0
    delta = XY_ONE - 1
    for row_index, y in enumerate(range(ymin, ymax)):
        _poll_cancel(cancel, row_index)
        active = [entry for entry in active if edges[entry[0]][1] != y]
        while next_edge < len(edges) and edges[next_edge][0] == y:
            edge = edges[next_edge]
            entry = [next_edge, edge[2]]
            work = _insert_active_edge_bounded(active, entry, work=work, cancel=cancel)
            next_edge += 1
        work += len(active)
        if work > MAX_RASTER_SPAN_WORK:
            raise RasterSpanLimitError("polygon active-edge work exceeds the bound")
        for pair in range(0, len(active) - 1, 2):
            left, right = active[pair], active[pair + 1]
            if y >= 0:
                x0 = (left[1] + delta) >> XY_SHIFT
                x1 = right[1] >> XY_SHIFT
                if x0 < width and x1 >= 0:
                    x0, x1 = max(0, x0), min(width - 1, x1)
                    if x0 <= x1:
                        fill.append((y, x0, x1))
                        if len(boundary) + len(fill) > MAX_RASTER_SPAN_SEGMENTS:
                            raise RasterSpanLimitError("polygon span count exceeds the bound")
            left[1] += edges[left[0]][3]
            right[1] += edges[right[0]][3]
        active.sort(key=lambda entry: entry[1])
    return normalize_spans(chain(boundary, fill), width=width, height=height, cancel=cancel)


def intersect_scanline_spans(first: Sequence[tuple[int,int,int]],
                             second: Sequence[tuple[int,int,int]], *,
                             cancel: Callable[[], bool] | None = None):
    """Return lexicographic (x,y) endpoint pixels of the span intersection."""
    if len(first) + len(second) > MAX_RASTER_SPAN_SEGMENTS:
        raise RasterSpanLimitError("span intersection input exceeds the bound")
    by_y_a: dict[int, list[tuple[int,int]]] = defaultdict(list)
    by_y_b: dict[int, list[tuple[int,int]]] = defaultdict(list)
    for index, (y,x0,x1) in enumerate(first):
        _poll_cancel(cancel,index)
        by_y_a[int(y)].append((int(x0),int(x1)))
    for index, (y,x0,x1) in enumerate(second):
        _poll_cancel(cancel,index)
        by_y_b[int(y)].append((int(x0),int(x1)))
    minimum = maximum = None
    work = 0
    for row_index, y in enumerate(sorted(by_y_a.keys() & by_y_b.keys())):
        _poll_cancel(cancel,row_index)
        a, b = by_y_a[y], by_y_b[y]
        i = j = 0
        while i < len(a) and j < len(b):
            work += 1
            if work > MAX_RASTER_SPAN_WORK:
                raise RasterSpanLimitError("span intersection work exceeds the bound")
            left, right = max(a[i][0],b[j][0]), min(a[i][1],b[j][1])
            if left <= right:
                for point in ((left,y),(right,y)):
                    if minimum is None or point < minimum:
                        minimum = point
                    if maximum is None or point > maximum:
                        maximum = point
            if a[i][1] <= b[j][1]:
                i += 1
            else:
                j += 1
    return None if minimum is None else (minimum, maximum)


__all__ = [
    "MAX_RASTER_PIXELS", "MAX_RASTER_SPAN_POINTS", "MAX_RASTER_SPAN_ROWS",
    "MAX_RASTER_SPAN_SEGMENTS", "MAX_RASTER_SPAN_WORK", "RasterSpanCancelled",
    "RasterSpanLimitError", "fill_convex_poly_spans", "fill_polygon_spans",
    "filled_circle_spans", "intersect_scanline_spans", "line2_points",
    "line8_points", "normalize_spans", "thick_line_spans",
]
