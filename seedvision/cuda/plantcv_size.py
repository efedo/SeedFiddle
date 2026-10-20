"""CUDA premeasurements and sparse boundary topology for PlantCV size parity.

Dense per-object masks and raw reductions stay on CUDA. Only bounded boundary
coordinates/neighbor bits transfer for sparse contour tracing and compact OpenCV
geometry. Supported external, hole and nested contours return pixel-unit
observations; resource-limited topology remains unavailable.
Bounded nonverbose diagnostic drawing keeps full-resolution RGB pixels on CUDA.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Callable
from dataclasses import dataclass
from math import atan2, gcd, isclose, isfinite, pi, sin, sqrt

import cv2
import numpy as np
import torch
from .plantcv_raster_spans import (
    RasterSpanCancelled, RasterSpanLimitError,
    fill_polygon_spans, thick_line_spans, intersect_scanline_spans,
    closed_polyline_spans, iter_adjacent_closed_polyline_spans,
    thick_line_fixed_spans, normalize_spans, circle_outline_spans,
    MAX_RASTER_SPAN_WORK, MAX_RASTER_SPAN_SEGMENTS, XY_SHIFT, XY_ONE,
)


PLANTCV_VERSION = "4.11.3"
PLANTCV_OPENCV_VERSION = "4.11.0"
PLANTCV_OPENCV_PYTHON_VERSION = "4.11.0.86"
PLANTCV_SIZE_SOURCE_REFERENCE_ENV_SHA256 = (
    "db5942d8cda707952e5b43777f3641a195383f62f79ab33a81b4f053bc1eb482"
)
MAX_RASTER_PIXELS = 64_000_000
MAX_TOPOLOGY_BOUNDARY_POINTS = 1_000_000
MAX_TILE_PIXELS = 1_000_000
MAX_CONTOUR_TRACE_STEPS = MAX_TOPOLOGY_BOUNDARY_POINTS * 16
MAX_CONTOUR_SCAN_ROWS = 1_000_000
MAX_CONTOUR_COMPONENTS = 65_536
MAX_CONTOUR_AGGREGATE_POINTS = MAX_TOPOLOGY_BOUNDARY_POINTS
MAX_CONTOUR_WORK_UNITS = MAX_TOPOLOGY_BOUNDARY_POINTS * 32

# Bit order is clockwise: N, NE, E, SE, S, SW, W, NW.
_NEIGHBOR_OFFSETS = (
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
)


class PlantCVSizeResourceError(ValueError):
    """The input raster or requested sparse topology exceeds its fixed bound."""


class PlantCVSizeCancelledError(RuntimeError):
    """A bounded mask/topology pass was cancelled before a complete result."""


def _check_cancel(cancel: Callable[[], bool] | None) -> bool:
    return bool(cancel is not None and cancel())


class _DisjointSet:
    def __init__(self, count: int) -> None:
        self.parent = list(range(count))
        self.rank = [0] * count

    def find(self, item: int) -> int:
        parent = self.parent[item]
        while parent != self.parent[parent]:
            self.parent[parent] = self.parent[self.parent[parent]]
            parent = self.parent[parent]
        self.parent[item] = parent
        return parent

    def union(self, left: int, right: int) -> None:
        left, right = self.find(left), self.find(right)
        if left == right:
            return
        if self.rank[left] < self.rank[right]:
            left, right = right, left
        self.parent[right] = left
        if self.rank[left] == self.rank[right]:
            self.rank[left] += 1


def _foreground_spans_from_boundary(records: np.ndarray, *, width: int,
        height: int, cancel: Callable[[], bool] | None = None):
    """Recover horizontal runs from compact W/E boundary endpoints."""
    values = np.asarray(records)
    if values.ndim != 2 or values.shape[1] != 3 or values.dtype.kind not in "iu":
        raise ValueError("boundary records must be an integer Nx3 array")
    if len(values) > MAX_TOPOLOGY_BOUNDARY_POINTS or height > MAX_CONTOUR_SCAN_ROWS:
        raise PlantCVSizeResourceError("boundary or contour scan-row limit exceeded")
    if not len(values):
        return ()
    keys = values[:, 1].astype(np.int64)*width+values[:, 0]
    if np.any(keys[1:] <= keys[:-1]) or np.any(values[:, 2] > 255):
        raise ValueError("boundary records must be unique row-major byte occupancy rows")
    breaks = np.flatnonzero(values[1:, 1] != values[:-1, 1])+1
    spans = []
    for group_index, (begin, end) in enumerate(zip(np.r_[0, breaks], np.r_[breaks, len(values)])):
        if group_index % 4096 == 0 and _check_cancel(cancel):
            raise PlantCVSizeCancelledError("foreground span reconstruction cancelled")
        row = values[begin:end]
        y = int(row[0, 1])
        if y < 0 or y >= height:
            raise ValueError("boundary row lies outside the image")
        xs, bits = row[:, 0], row[:, 2]
        starts, ends = xs[(bits & 64) == 0], xs[(bits & 4) == 0]
        if len(starts) != len(ends):
            raise PlantCVSizeResourceError("boundary table does not define closed row spans")
        for x0, x1 in zip(starts.tolist(), ends.tolist()):
            if x0 < 0 or x1 < x0 or x1 >= width:
                raise ValueError("boundary endpoints lie outside the ordered image row")
            spans.append((y, int(x0), int(x1)))
    return tuple(spans)


def _span_components(spans, *, width: int, height: int,
        diagonal_connectivity: bool, background: bool,
        cancel: Callable[[], bool] | None = None,
        charge_work: Callable[[int], None] | None = None):
    """Union compact horizontal runs; background uses four-connectivity."""
    if height > MAX_CONTOUR_SCAN_ROWS or len(spans) > MAX_TOPOLOGY_BOUNDARY_POINTS:
        raise PlantCVSizeResourceError("scanline topology resource bound exceeded")
    if not spans:
        return (), frozenset()
    by_row = {}
    for index, (y, x0, x1) in enumerate(spans):
        if index % 4096 == 0 and _check_cancel(cancel):
            raise PlantCVSizeCancelledError("scanline topology cancelled")
        if charge_work is not None:
            charge_work(1)
        by_row.setdefault(y, []).append((x0, x1, index))
    dsu, rows, work = _DisjointSet(len(spans)), sorted(by_row), 0
    tolerance = int(diagonal_connectivity and not background)
    if charge_work is not None:
        charge_work(len(rows))
    for previous_y, current_y in zip(rows, rows[1:]):
        if current_y != previous_y+1:
            continue
        previous, current = by_row[previous_y], by_row[current_y]
        left = right = 0
        while left < len(previous) and right < len(current):
            if work % 4096 == 0 and _check_cancel(cancel):
                raise PlantCVSizeCancelledError("scanline union cancelled")
            work += 1
            if charge_work is not None:
                charge_work(1)
            ax0, ax1, ai = previous[left]
            bx0, bx1, bi = current[right]
            if ax1+tolerance >= bx0 and bx1+tolerance >= ax0:
                dsu.union(ai, bi)
            if ax1 < bx1:
                left += 1
            else:
                right += 1
    component_ids, exterior = [], set()
    for i, (y, x0, x1) in enumerate(spans):
        if i % 4096 == 0 and _check_cancel(cancel):
            raise PlantCVSizeCancelledError("scanline component labeling cancelled")
        if charge_work is not None:
            charge_work(1)
        component = dsu.find(i)
        component_ids.append(component)
        if y == 0 or y == height-1 or x0 == 0 or x1 == width-1:
            exterior.add(component)
    # The complement builder includes every background run in every image
    # row. An absent row is entirely foreground and cannot connect a hole to
    # the frame. Only an actual frame-touching run proves exterior membership.
    return tuple(component_ids), frozenset(exterior)


def _background_spans_from_foreground(foreground_spans, *, width: int,
        height: int, cancel: Callable[[], bool] | None = None,
        charge_work: Callable[[int], None] | None = None):
    if height > MAX_CONTOUR_SCAN_ROWS:
        raise PlantCVSizeResourceError("contour scan-row limit exceeded")
    rows = {}
    for index, (y, x0, x1) in enumerate(foreground_spans):
        if index % 4096 == 0 and _check_cancel(cancel):
            raise PlantCVSizeCancelledError("background span reconstruction cancelled")
        if charge_work is not None:
            charge_work(1)
        rows.setdefault(y, []).append((x0, x1))
    result = []
    for y in range(height):
        if y % 4096 == 0 and _check_cancel(cancel):
            raise PlantCVSizeCancelledError("background span reconstruction cancelled")
        if charge_work is not None:
            charge_work(1)
        cursor = 0
        for x0, x1 in rows.get(y, ()):
            if cursor < x0:
                if charge_work is not None:
                    charge_work(1)
                result.append((y, cursor, x0-1))
                if len(result) > MAX_TOPOLOGY_BOUNDARY_POINTS:
                    raise PlantCVSizeResourceError("background-span limit exceeded")
            cursor = x1+1
        if cursor < width:
            if charge_work is not None:
                charge_work(1)
            result.append((y, cursor, width-1))
        if len(result) > MAX_TOPOLOGY_BOUNDARY_POINTS:
            raise PlantCVSizeResourceError("background-span limit exceeded")
    return tuple(result)


def _moore_trace(start_xy, backtrack_xy, *, width: int, height: int,
        foreground_contains: Callable[[int, int], bool],
        cancel: Callable[[], bool] | None = None,
        max_steps: int = MAX_CONTOUR_TRACE_STEPS,
        max_points: int = MAX_CONTOUR_AGGREGATE_POINTS,
        step_count: list[int] | None = None,
        charge_work: Callable[[int], None] | None = None) -> np.ndarray:
    """Preserve each border transition, including self-touch revisits."""
    start = tuple(map(int, start_xy))
    if max_points < 1:
        raise PlantCVSizeResourceError("aggregate contour-point limit exceeded")
    if not foreground_contains(*start):
        raise ValueError("contour start is not foreground")
    contour, first_next, current = [start], None, start
    backtrack = backtrack_xy
    for step in range(max_steps):
        if step_count is not None:
            step_count[0] += 1
        if charge_work is not None:
            charge_work(1)
        if step % 1024 == 0 and _check_cancel(cancel):
            raise PlantCVSizeCancelledError("Moore contour tracing cancelled")
        cx, cy = current
        delta = (backtrack[1]-cy, backtrack[0]-cx)
        try:
            back_index = _NEIGHBOR_OFFSETS.index(delta)
        except ValueError as error:
            raise PlantCVSizeResourceError("Moore trace lost adjacent backtrack state") from error
        next_xy = None
        for turn in range(1, 9):
            if charge_work is not None:
                charge_work(1)
            direction = (back_index-turn) % 8
            dy, dx = _NEIGHBOR_OFFSETS[direction]
            nx, ny = cx+dx, cy+dy
            if 0 <= nx < width and 0 <= ny < height and foreground_contains(nx, ny):
                py, px = _NEIGHBOR_OFFSETS[(direction+1) % 8]
                next_xy, next_backtrack = (nx, ny), (cx+px, cy+py)
                break
        if next_xy is None:
            return np.asarray(contour, dtype=np.int32)
        if first_next is None:
            first_next = next_xy
        elif current == start and next_xy == first_next:
            if contour[-1] == start:
                contour.pop()
            return np.asarray(contour, dtype=np.int32)
        # The final repeated start is removed by the closure check on the next
        # transition. Permit that one temporary point at the exact output cap;
        # a non-closing revisit still fails before another point is appended.
        if len(contour) >= max_points and next_xy != start:
            raise PlantCVSizeResourceError("aggregate contour-point limit exceeded")
        current, backtrack = next_xy, next_backtrack
        contour.append(current)
    raise PlantCVSizeResourceError("aggregate contour trace-step budget exceeded")


def _valid_retr_hierarchy(hierarchy, *, cancel=None, charge_work=None):
    """Validate reciprocal sibling links and a complete rooted contour forest."""
    if _check_cancel(cancel):
        raise PlantCVSizeCancelledError("contour hierarchy validation cancelled")
    if not isinstance(hierarchy, np.ndarray) or hierarchy.dtype.kind not in "iu":
        return False
    if hierarchy.ndim == 3 and hierarchy.shape[0] == 1 and hierarchy.shape[2] == 4:
        links = hierarchy[0]
    elif hierarchy.ndim == 2 and hierarchy.shape[1] == 4:
        links = hierarchy
    else:
        return False
    count = len(links)
    if count > MAX_CONTOUR_COMPONENTS:
        return False
    work = 0
    def charge():
        nonlocal work
        if charge_work is not None:
            charge_work()
        else:
            work += 1
            if work % 4096 == 1 and _check_cancel(cancel):
                raise PlantCVSizeCancelledError("contour hierarchy validation cancelled")
    if not count:
        return True
    for index, (next_i, prev_i, child_i, parent_i) in enumerate(links):
        charge()
        if any(link < -1 or link >= count or link == index
                for link in (next_i, prev_i, child_i, parent_i)):
            return False
        if next_i >= 0 and (links[next_i, 1] != index or links[next_i, 3] != parent_i):
            return False
        if prev_i >= 0 and (links[prev_i, 0] != index or links[prev_i, 3] != parent_i):
            return False
        if child_i >= 0 and (links[child_i, 3] != index or links[child_i, 1] != -1):
            return False
    child_seen = set()
    for parent_i, row in enumerate(links):
        child_i, previous = int(row[2]), -1
        while child_i >= 0:
            charge()
            if (child_i in child_seen or int(links[child_i, 3]) != parent_i
                    or int(links[child_i, 1]) != previous):
                return False
            child_seen.add(child_i)
            previous, child_i = child_i, int(links[child_i, 0])
    if child_seen != set(np.flatnonzero(links[:, 3] >= 0).tolist()):
        return False
    roots = set(np.flatnonzero(links[:, 3] == -1).tolist())
    heads = [index for index in roots if links[index, 1] == -1]
    if len(heads) != 1:
        return False
    root_seen, current, previous = set(), heads[0], -1
    while current >= 0:
        charge()
        if (current in root_seen or current not in roots
                or int(links[current, 1]) != previous):
            return False
        root_seen.add(current)
        previous, current = current, int(links[current, 0])
    if root_seen != roots:
        return False
    # Child chains alone also admit a detached parent/child cycle beside a
    # valid root. Require every node to be reached once from the root forest.
    reachable, stack = set(), list(roots)
    while stack:
        charge()
        node = stack.pop()
        if node in reachable:
            return False
        reachable.add(node)
        child = int(links[node, 2])
        while child >= 0:
            charge()
            stack.append(child)
            child = int(links[child, 0])
    if _check_cancel(cancel):
        raise PlantCVSizeCancelledError("contour hierarchy validation cancelled")
    return len(reachable) == count


def _trace_sparse_contours(records: np.ndarray, *, width: int, height: int,
        cancel: Callable[[], bool] | None = None, work_budget=None):
    """Trace a bounded RETR_TREE forest from compact runs, without a CPU mask."""
    if work_budget is not None and (type(work_budget) is not list or len(work_budget) != 1
            or type(work_budget[0]) is not int or not 0 <= work_budget[0] <= MAX_CONTOUR_WORK_UNITS):
        raise ValueError('contour work_budget must be a bounded one-element integer list')
    if width < 1 or height < 1 or height > MAX_CONTOUR_SCAN_ROWS:
        return None, None, "unsupported_scan_extent"
    values = np.asarray(records)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("compact boundary records must be Nx3")
    initial_work = 0 if work_budget is None else work_budget[0]
    if _check_cancel(cancel):
        raise PlantCVSizeCancelledError("contour tracing cancelled")
    if not len(values):
        return (), np.empty((1, 0, 4), dtype=np.int32), "empty_mask"
    try:
        work = initial_work
        next_poll = (work // 4096 + 1) * 4096
        def charge(amount=1):
            nonlocal work, next_poll
            if amount < 0 or work + amount > MAX_CONTOUR_WORK_UNITS:
                raise PlantCVSizeResourceError("contour work-unit limit exceeded")
            work += amount
            if work_budget is not None:
                work_budget[0] = work
            if work >= next_poll or amount >= 4096:
                if _check_cancel(cancel):
                    raise PlantCVSizeCancelledError("contour topology work cancelled")
                next_poll = (work // 4096 + 1) * 4096

        charge(len(values))
        fg_spans = _foreground_spans_from_boundary(
            values, width=width, height=height, cancel=cancel)
        fg_ids, _ = _span_components(fg_spans, width=width, height=height,
            diagonal_connectivity=True, background=False, cancel=cancel, charge_work=charge)
        bg_spans = _background_spans_from_foreground(fg_spans, width=width,
            height=height, cancel=cancel, charge_work=charge)
        bg_ids, exterior = _span_components(bg_spans, width=width, height=height,
            diagonal_connectivity=False, background=True, cancel=cancel, charge_work=charge)

        fg_rows, bg_rows, fg_first, hole_first = {}, {}, {}, {}
        for (y, x0, x1), component in zip(fg_spans, fg_ids):
            charge()
            fg_rows.setdefault(y, []).append((x0, x1, component))
            if component not in fg_first:
                fg_first[component] = (x0, y)
        for (y, x0, x1), component in zip(bg_spans, bg_ids):
            charge()
            bg_rows.setdefault(y, []).append((x0, x1, component))
            if component not in exterior and component not in hole_first:
                hole_first[component] = (x0, y)
        node_count = len(fg_first)+len(hole_first)
        if node_count > MAX_CONTOUR_COMPONENTS:
            raise PlantCVSizeResourceError("contour component limit exceeded")
        charge(len(fg_spans)+len(bg_spans))
        fg_starts = {y: tuple(x0 for x0, _, _ in row) for y, row in fg_rows.items()}
        bg_starts = {y: tuple(x0 for x0, _, _ in row) for y, row in bg_rows.items()}

        def component_at(rows, starts, x, y):
            row = rows.get(y, ())
            charge(1+len(row).bit_length())
            index = bisect_right(starts.get(y, ()), x)-1
            if index < 0:
                return None
            x0, x1, component = row[index]
            return component if x0 <= x <= x1 else None
        def fg_at(x, y):
            return component_at(fg_rows, fg_starts, x, y)
        def bg_at(x, y):
            return component_at(bg_rows, bg_starts, x, y)
        def fg_contains(x, y):
            return fg_at(x, y) is not None

        # Pinned OpenCV discovers external borders at their first foreground
        # pixel and holes at their first background pixel (start one pixel left).
        # Tree::addChild prepends each sibling, and its iterator uses DFS order.
        charge(32*node_count)
        fg_order = sorted(fg_first, key=lambda c: (fg_first[c][1], fg_first[c][0]))
        hole_order = sorted(hole_first, key=lambda c: (hole_first[c][1], hole_first[c][0]))
        node_parent, node_event = {}, {}
        for component in fg_order:
            charge()
            x0, y = fg_first[component]
            left_bg = bg_at(x0-1, y) if x0 > 0 else None
            if x0 > 0 and left_bg is None:
                return None, None, "unsupported_missing_background_parent"
            node = ("f", component)
            node_parent[node] = ("h", left_bg) if left_bg in hole_first else None
            node_event[node] = (y, x0)
        for component in hole_order:
            charge()
            x0, y = hole_first[component]
            parent_fg = fg_at(x0-1, y) if x0 > 0 else None
            if parent_fg is None:
                return None, None, "unsupported_missing_hole_parent"
            node = ("h", component)
            node_parent[node], node_event[node] = ("f", parent_fg), (y, x0)

        charge(node_count)
        trace_nodes = [("f", c) for c in fg_order]+[("h", c) for c in hole_order]
        contours_by_node, traced_keys = {}, set()
        total_points = total_steps = 0
        for node_index, node in enumerate(trace_nodes):
            if node_index % 1024 == 0 and _check_cancel(cancel):
                raise PlantCVSizeCancelledError("component contour tracing cancelled")
            kind, component = node
            if kind == "f":
                start = fg_first[component]
                backtrack, expected_component = (start[0]-1, start[1]), component
            else:
                x0, y = hole_first[component]
                start, backtrack = (x0-1, y), (x0, y)
                expected_component = node_parent[node][1]
            if fg_at(*start) != expected_component:
                return None, None, "unsupported_contour_parent_transition"
            steps = [0]
            contour = _moore_trace(start, backtrack, width=width, height=height,
                foreground_contains=fg_contains, cancel=cancel,
                max_steps=MAX_CONTOUR_TRACE_STEPS-total_steps,
                max_points=MAX_CONTOUR_AGGREGATE_POINTS-total_points,
                step_count=steps, charge_work=charge)
            total_steps += steps[0]
            total_points += len(contour)
            if total_points > MAX_CONTOUR_AGGREGATE_POINTS:
                raise PlantCVSizeResourceError("aggregate contour-point limit exceeded")
            for x, y in contour:
                if fg_at(int(x), int(y)) != expected_component:
                    return None, None, "unsupported_non_simple_contour_parent"
                charge()
                traced_keys.add(int(y)*width+int(x))
            contours_by_node[node] = contour

        boundary_keys = set()
        for x, y, bits in values:
            charge()
            key = int(y)*width+int(x)
            boundary_keys.add(key)
            if int(bits) & 85 != 85 and key not in traced_keys:
                return None, None, "unsupported_non_simple_boundary"
        for key in traced_keys:
            charge()
            if key not in boundary_keys:
                return None, None, "unsupported_non_simple_boundary"

        children = {}
        for node, parent in node_parent.items():
            charge()
            if parent is not None and parent not in node_parent:
                return None, None, "unsupported_missing_contour_parent"
            children.setdefault(parent, []).append(node)
        for siblings in children.values():
            charge(32*len(siblings))
            siblings.sort(key=lambda node: node_event[node], reverse=True)
        roots = children.get(None, ())
        if not roots:
            return None, None, "unsupported_contour_tree_without_root"
        ordered, seen, stack = [], set(), list(reversed(roots))
        while stack:
            charge()
            node = stack.pop()
            if node in seen:
                return None, None, "unsupported_cyclic_contour_hierarchy"
            seen.add(node)
            ordered.append(node)
            node_children = children.get(node, ())
            charge(len(node_children))
            stack.extend(reversed(node_children))
        if len(seen) != len(node_parent):
            return None, None, "unsupported_disconnected_contour_hierarchy"

        index_of = {}
        for index, node in enumerate(ordered):
            charge()
            index_of[node] = index
        hierarchy = np.full((1, len(ordered), 4), -1, dtype=np.int32)
        for siblings in children.values():
            for position, node in enumerate(siblings):
                charge()
                current = index_of[node]
                if position+1 < len(siblings):
                    hierarchy[0, current, 0] = index_of[siblings[position+1]]
                if position:
                    hierarchy[0, current, 1] = index_of[siblings[position-1]]
        for index, node in enumerate(ordered):
            charge()
            parent = node_parent[node]
            if parent is not None:
                hierarchy[0, index, 3] = index_of[parent]
            node_children = children.get(node, ())
            if node_children:
                hierarchy[0, index, 2] = index_of[node_children[0]]
        if not _valid_retr_hierarchy(hierarchy, cancel=cancel, charge_work=charge):
            return None, None, "unsupported_invalid_contour_forest"
        charge(len(ordered))
        contours = tuple(contours_by_node[node] for node in ordered)
        status = ("nested_contours" if hole_order else
            "single_external_contour" if len(roots) == 1 else "multiple_external_contours")
        if _check_cancel(cancel):
            raise PlantCVSizeCancelledError("contour tracing cancelled")
        return contours, hierarchy, status
    except PlantCVSizeResourceError as error:
        return None, None, f"unsupported_resource_bound:{error}"

@dataclass(frozen=True)
class CompactBoundaryTopology:
    """CPU metadata rows ``(x, y, eight-neighbor occupancy bits)``.

    Rows are sorted in image row-major order. They are not OpenCV contour
    points, and cannot be used as a contour without tracing and hierarchy
    conformance work.
    """

    xy_neighbor_bits: torch.Tensor | None
    boundary_point_count: int | None
    point_limit: int
    status: str
    contours_xy: tuple[torch.Tensor, ...] | None = None
    hierarchy: torch.Tensor | None = None
    contour_status: str = "not_traced"


@dataclass(frozen=True)
class PlantCVSizeObservationBundle:
    """PlantCV 4.11.3 ``analyze.size`` values in pixel units."""

    area: float
    convex_hull_area: float
    solidity: float
    perimeter: float
    total_edge_length: float
    width: int
    height: int
    longest_path: float
    center_of_mass: tuple[float, float]
    convex_hull_vertices: int
    object_in_frame: bool
    ellipse_center: tuple[float, float]
    ellipse_major_axis: float
    ellipse_minor_axis: float
    ellipse_angle: float
    ellipse_eccentricity: float

    @property
    def in_bounds(self):
        return self.object_in_frame



PLANTCV_SIZE_METHOD = "plantcv.plantcv.analyze.size"
PLANTCV_SIZE_DEFAULT_LINE_THICKNESS = 5
OPENCV_RUNTIME_VERSION = str(cv2.__version__)


@dataclass(frozen=True)
class PlantCVSizeOutputObservation:
    variable: str
    value: float | int | bool | tuple[float, float]
    trait: str
    method: str
    scale: str
    datatype: type
    label: str | tuple[str, str]


def _plantcv_size_output_observations(bundle):
    # Pixel-unit metadata and ordering from the pinned public _analyze_size.
    schema = (
        ("area", "area", "pixels", int, "pixels"),
        ("convex_hull_area", "convex hull area", "pixels", int, "pixels"),
        ("solidity", "solidity", "none", float, "none"),
        ("perimeter", "perimeter", "pixels", int, "pixels"),
        ("total_edge_length", "total length of object edges", "pixels", int, "pixels"),
        ("width", "width", "pixels", int, "pixels"),
        ("height", "height", "pixels", int, "pixels"),
        ("longest_path", "longest path", "pixels", int, "pixels"),
        ("center_of_mass", "center of mass", "none", tuple, ("x", "y")),
        ("convex_hull_vertices", "convex hull vertices", "none", int, "none"),
        ("object_in_frame", "object in frame", "none", bool, "none"),
        ("ellipse_center", "ellipse center", "none", tuple, ("x", "y")),
        ("ellipse_major_axis", "ellipse major axis length", "pixels", int, "pixels"),
        ("ellipse_minor_axis", "ellipse minor axis length", "pixels", int, "pixels"),
        ("ellipse_angle", "ellipse major axis angle", "degrees", float, "degrees"),
        ("ellipse_eccentricity", "ellipse eccentricity", "none", float, "none"),
    )
    frame_observation = PlantCVSizeOutputObservation('in_bounds', bundle.in_bounds,
        'whether the plant goes out of bounds ', 'plantcv.plantcv.within_frame',
        'none', bool, 'none')
    return (frame_observation,)+tuple(PlantCVSizeOutputObservation(variable, getattr(bundle, variable),
        trait, PLANTCV_SIZE_METHOD, scale, datatype, label)
        for variable, trait, scale, datatype, label in schema)


def _zero_plantcv_size_bundle(object_in_frame):
    return PlantCVSizeObservationBundle(
        area=0, convex_hull_area=0, solidity=0, perimeter=0,
        total_edge_length=0, width=0, height=0, longest_path=0.0,
        center_of_mass=(0, 0), convex_hull_vertices=0,
        object_in_frame=object_in_frame, ellipse_center=(0, 0),
        ellipse_major_axis=0, ellipse_minor_axis=0, ellipse_angle=0.0,
        ellipse_eccentricity=0.0)


class _UnsupportedSizeObservation(Exception):
    pass


def _charge_size_geometry_work(work_budget, amount, cancel=None):
    if _check_cancel(cancel):
        raise PlantCVSizeCancelledError('PlantCV size diagnostic geometry cancelled')
    if work_budget is not None:
        amount = max(1, int(amount))
        if work_budget[0] + amount > MAX_RASTER_SPAN_WORK:
            raise RasterSpanLimitError('aggregate size diagnostic geometry exceeds the bound')
        work_budget[0] += amount


def _validate_line_thickness(value):
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 32767:
        raise ValueError('line_thickness must be an integer in [1, 32767]')
    return value


def _plantcv_longest_path_details(*, hull, width, height, cmx, cmy, cancel=None,
                                work_budget=None, line_thickness=5):
    if _check_cancel(cancel):
        raise PlantCVSizeCancelledError("PlantCV size longest-path cancelled")
    center_x, center_y = int(cmx), int(cmy)
    # At most 11x11 CPU pixels for the source's frame-clipped radius-four disk.
    x0, x1 = max(0, center_x-5), min(width, center_x+6)
    y0, y1 = max(0, center_y-5), min(height, center_y+6)
    disk = np.zeros((y1-y0, x1-x0), dtype=np.uint8)
    cv2.circle(disk, (center_x-x0, center_y-y0), 4, 255, -1)
    circle_contours, _ = cv2.findContours(disk, cv2.RETR_TREE,
        cv2.CHAIN_APPROX_NONE, offset=(x0, y0))
    if not circle_contours:
        raise _UnsupportedSizeObservation("clipped radius-four circle has no contour")
    vertices = np.vstack(hull)
    _charge_size_geometry_work(work_budget, 32 * len(vertices), cancel)
    distances = []
    for index, point in enumerate(vertices):
        if index % 1024 == 0 and _check_cancel(cancel):
            raise PlantCVSizeCancelledError("PlantCV size hull search cancelled")
        distances.append(cv2.pointPolygonTest(circle_contours[0],
            tuple(int(value) for value in point), measureDist=True))
    max_x, max_y = (int(value) for value in vertices[int(np.argmax(np.absolute(distances)))])
    xdiff, ydiff = float(max_x-center_x), float(max_y-center_y)
    slope = float(ydiff/xdiff) if xdiff != 0 else 1
    intercept = center_y-slope*center_x
    caliper_thickness = line_thickness
    if slope != 0:
        left, right = int(-intercept/slope), int((height-intercept)/slope)
        if 0 <= left <= width and 0 <= right <= width:
            start, end = (right, height), (left, 0)
        else:
            start, end = (0, int(intercept)), (width, int(slope*width+intercept))
            # Pinned _longest_axis hard-codes five in this measurement branch.
            # Its later diagnostic overlay still uses the requested thickness.
            caliper_thickness = 5
    else:
        start, end = (width, center_y), (0, center_y)
    try:
        min_y = max(0, min(int(point[1]) for point in vertices))
        max_y = min(height-1, max(int(point[1]) for point in vertices))
        _charge_size_geometry_work(work_budget,
            4 * len(vertices) + 8 * max(0, max_y-min_y+1), cancel)
        hull_spans = fill_polygon_spans(vertices, width, height, cancel=cancel)
        _charge_size_geometry_work(work_budget,
            4 * (abs(end[0]-start[0]) + abs(end[1]-start[1])
                 + 2 * caliper_thickness + 4), cancel)
        line_spans = thick_line_spans(width, height, start, end,
            thickness=caliper_thickness, cancel=cancel)
        _charge_size_geometry_work(work_budget, len(hull_spans)+len(line_spans), cancel)
        endpoints = intersect_scanline_spans(hull_spans, line_spans, cancel=cancel)
    except RasterSpanCancelled as error:
        raise PlantCVSizeCancelledError(str(error)) from error
    if endpoints is None:
        raise _UnsupportedSizeObservation("longest-path line does not intersect the hull")
    # Existing NumPy computes the same compact two-coordinate Euclidean metric;
    # SciPy is confined to the isolated reference interpreter.
    length = float(np.linalg.norm(np.asarray(endpoints[1], dtype=np.float64)
        - np.asarray(endpoints[0], dtype=np.float64)))
    return length, tuple(endpoints[1]), tuple(endpoints[0])


def _plantcv_longest_path(*, hull, width, height, cmx, cmy, cancel=None, line_thickness=5):
    return _plantcv_longest_path_details(hull=hull, width=width, height=height,
        cmx=cmx, cmy=cmy, cancel=cancel, line_thickness=line_thickness)[0]


def _fitellipse_uses_411_perturbation(points, *, cancel=None):
    """Detect the pinned fitEllipseNoDirect singular-value perturbation branch.

    OpenCV 4.11 shapedescr.cpp uses deterministic getOfs(i,eps), while 5.0 uses
    RNG offsets. This detects when the compact deterministic compatibility
    path is required. Other changed runtimes withhold that observation bundle.
    Source: https://github.com/opencv/opencv/blob/4.11.0/modules/imgproc/src/shapedescr.cpp
    Retained Intel license: docs/OPENCV_MATH_NOTICES.md.
    """
    if _check_cancel(cancel):
        raise PlantCVSizeCancelledError('ellipse conditioning check cancelled')
    xy = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(xy) <= 5:
        return False
    cx = cy = np.float32(0.0)
    for index, (x, y) in enumerate(xy):
        if index % 4096 == 0 and _check_cancel(cancel):
            raise PlantCVSizeCancelledError('ellipse centroid check cancelled')
        cx = np.float32(cx+x)
        cy = np.float32(cy+y)
    cx, cy = np.float32(cx/np.float32(len(xy))), np.float32(cy/np.float32(len(xy)))
    centered = np.empty_like(xy)
    absolute_sum = 0.0
    for index, (x, y) in enumerate(xy):
        if index % 4096 == 0 and _check_cancel(cancel):
            raise PlantCVSizeCancelledError('ellipse scale check cancelled')
        dx, dy = np.float32(x-cx), np.float32(y-cy)
        centered[index] = (dx, dy)
        absolute_sum += abs(float(dx))+abs(float(dy))
    epsilon = float(np.finfo(np.float32).eps)
    scale = 100.0/max(absolute_sum, epsilon)
    design = np.empty((len(xy), 5), dtype=np.float64)
    for index, (dx, dy) in enumerate(centered):
        if index % 4096 == 0 and _check_cancel(cancel):
            raise PlantCVSizeCancelledError('ellipse design check cancelled')
        px, py = float(dx)*scale, float(dy)*scale
        design[index] = (-px*px, -py*py, -px*py, px, py)
    singular = np.asarray(cv2.SVDecomp(design, flags=cv2.SVD_NO_UV)[0]).reshape(-1)
    if _check_cancel(cancel):
        raise PlantCVSizeCancelledError('ellipse conditioning check cancelled')
    return float(singular[0])*epsilon > float(singular[4])


def _fitellipse_no_direct_411_details(points, *, cancel=None):
    """Port the deterministic 4.11 ill-conditioned no-direct compact-point path.

    The caller applies the original >5-point gate and SVD predicate. Retained
    OpenCV source notices are in docs/OPENCV_MATH_NOTICES.md.
    """
    raw = np.asarray(points)
    if raw.dtype.kind not in "iu" or raw.size % 2:
        raise ValueError("fitEllipse points must be an integer Nx2 contour")
    n = int(raw.size // 2)
    if n < 6:
        raise ValueError("the pinned no-direct ellipse path requires more than five points")
    if n > MAX_CONTOUR_AGGREGATE_POINTS:
        raise PlantCVSizeResourceError("fitEllipse compact point limit exceeded")
    if _check_cancel(cancel):
        raise PlantCVSizeCancelledError("fitEllipse 4.11 port cancelled")
    xy = raw.reshape(n, 2).astype(np.float32, copy=True)
    cx = cy = np.float32(0.0)
    for index, (x, y) in enumerate(xy):
        if index % 4096 == 0 and _check_cancel(cancel):
            raise PlantCVSizeCancelledError("fitEllipse centroid pass cancelled")
        cx = np.float32(cx+x)
        cy = np.float32(cy+y)
    cx = np.float32(cx/np.float32(n))
    cy = np.float32(cy/np.float32(n))
    absolute_sum = 0.0
    for index, (x, y) in enumerate(xy):
        if index % 4096 == 0 and _check_cancel(cancel):
            raise PlantCVSizeCancelledError("fitEllipse scale pass cancelled")
        dx, dy = np.float32(x-cx), np.float32(y-cy)
        absolute_sum += abs(float(dx))+abs(float(dy))
    flt_epsilon = float(np.finfo(np.float32).eps)
    scale = 100.0/(absolute_sum if absolute_sum > flt_epsilon else flt_epsilon)
    eps = np.float32(absolute_sum/(n*2)*1e-3)
    for index in range(n):
        if index % 4096 == 0 and _check_cancel(cancel):
            raise PlantCVSizeCancelledError("fitEllipse deterministic perturbation cancelled")
        dx = np.float32(-eps if (index & 1) == 0 else eps)
        dy = np.float32(-eps if (index & 2) == 0 else eps)
        xy[index, 0] = np.float32(xy[index, 0]+dx)
        xy[index, 1] = np.float32(xy[index, 1]+dy)

    design = np.empty((n, 5), dtype=np.float64)
    rhs = np.full((n, 1), 10000.0, dtype=np.float64)
    for index, (x, y) in enumerate(xy):
        if index % 4096 == 0 and _check_cancel(cancel):
            raise PlantCVSizeCancelledError("fitEllipse coefficient pass cancelled")
        dx, dy = np.float32(x-cx), np.float32(y-cy)
        px, py = float(dx)*scale, float(dy)*scale
        design[index] = (-px*px, -py*py, -px*py, px, py)
    if _check_cancel(cancel):
        raise PlantCVSizeCancelledError("fitEllipse SVD cancelled")
    singular, u, vt = cv2.SVDecomp(design)
    if _check_cancel(cancel):
        raise PlantCVSizeCancelledError("fitEllipse SVD cancelled")
    gfp = np.asarray(cv2.SVBackSubst(singular, u, vt, rhs), dtype=np.float64).reshape(5)
    center_matrix = np.asarray(((2.0*gfp[0], gfp[2]), (gfp[2], 2.0*gfp[1])), dtype=np.float64)
    center_rhs = np.asarray(((gfp[3],), (gfp[4],)), dtype=np.float64)
    center_solution = np.zeros((2, 1), dtype=np.float64)
    if _check_cancel(cancel):
        raise PlantCVSizeCancelledError("fitEllipse center solve cancelled")
    _, center_solution = cv2.solve(center_matrix, center_rhs, center_solution, cv2.DECOMP_SVD)
    if _check_cancel(cancel):
        raise PlantCVSizeCancelledError("fitEllipse center solve cancelled")
    rp = np.zeros(5, dtype=np.float64)
    rp[:2] = np.asarray(center_solution, dtype=np.float64).reshape(2)

    design = np.empty((n, 3), dtype=np.float64)
    rhs = np.ones((n, 1), dtype=np.float64)
    for index, (x, y) in enumerate(xy):
        if index % 4096 == 0 and _check_cancel(cancel):
            raise PlantCVSizeCancelledError("fitEllipse final coefficient pass cancelled")
        dx, dy = np.float32(x-cx), np.float32(y-cy)
        px, py = float(dx)*scale-rp[0], float(dy)*scale-rp[1]
        design[index] = (px*px, py*py, px*py)
    if _check_cancel(cancel):
        raise PlantCVSizeCancelledError("fitEllipse final solve cancelled")
    prior_gfp = np.asarray(gfp[:3], dtype=np.float64).reshape(3, 1).copy()
    _, coefficients = cv2.solve(design, rhs, prior_gfp, cv2.DECOMP_SVD)
    if _check_cancel(cancel):
        raise PlantCVSizeCancelledError("fitEllipse final solve cancelled")
    gfp = np.asarray(coefficients, dtype=np.float64).reshape(3)
    min_eps = 1e-8
    rp[4] = -0.5*atan2(float(gfp[2]), float(gfp[1]-gfp[0]))
    t = (float(gfp[2])/sin(-2.0*float(rp[4]))
        if abs(float(gfp[2])) > min_eps else float(gfp[1]-gfp[0]))
    rp[2] = abs(float(gfp[0]+gfp[1]-t))
    if rp[2] > min_eps:
        rp[2] = sqrt(2.0/float(rp[2]))
    rp[3] = abs(float(gfp[0]+gfp[1]+t))
    if rp[3] > min_eps:
        rp[3] = sqrt(2.0/float(rp[3]))
    center_x = np.float32(np.float32(rp[0]/scale)+cx)
    center_y = np.float32(np.float32(rp[1]/scale)+cy)
    width, height = np.float32(rp[2]*2.0/scale), np.float32(rp[3]*2.0/scale)
    angle = np.float32(0.0)
    if width > height:
        width, height = height, width
        angle = np.float32(90.0+rp[4]*180.0/pi)
    if angle < -180.0:
        angle = np.float32(angle+360.0)
    if angle > 360.0:
        angle = np.float32(angle-360.0)
    ellipse = ((float(center_x), float(center_y)), (float(width), float(height)), float(angle))
    return ellipse, float(gfp[2]), float(gfp[1]-gfp[0])


def _fitellipse_no_direct_411(points, *, cancel=None):
    """Return the deterministic compact-point fit, without claiming solver parity."""
    return _fitellipse_no_direct_411_details(points, cancel=cancel)[0]


def _fitellipse_orientation_ambiguous(cross, diagonal_difference):
    """Conservatively reject the source's negative-x atan2 discontinuity.

    OpenCV core SVD implementations can change the sign of a near-zero cross
    coefficient. The pinned 1e-8 axis-aligned threshold does not normalize that
    sign: atan2 can return opposite sides of pi and change the published angle.
    Near-isotropic coefficients also leave orientation numerically undefined.
    """
    return (not isfinite(cross) or not isfinite(diagonal_difference)
        or abs(cross) <= 1e-8 and diagonal_difference <= 1e-8)


def _plantcv_size_observations(*, contours, hierarchy, contour_status, width,
        height, pixel_count, moment_m10, moment_m01, object_in_frame, cancel=None,
        line_thickness=5):
    if _check_cancel(cancel):
        raise PlantCVSizeCancelledError("PlantCV size observations cancelled")
    if contour_status == "empty_mask":
        composed, original_contours = np.empty((0, 1, 2), dtype=np.int32), ()
    elif contour_status in {"single_external_contour", "multiple_external_contours", "nested_contours"}:
        if contours is None or hierarchy is None or not contours:
            return None, "unsupported inconsistent contour-tree topology"
        links = np.asarray(hierarchy, dtype=np.int32).reshape(-1, 4)
        count = len(contours)
        if len(links) != count or not _valid_retr_hierarchy(links, cancel=cancel):
            return None, "unsupported contour-tree hierarchy"
        roots = int(np.count_nonzero(links[:, 3] == -1))
        has_children = bool(np.any(links[:, 2] != -1))
        if ((contour_status == "single_external_contour" and (count != 1 or roots != 1 or has_children))
                or (contour_status == "multiple_external_contours" and (roots < 2 or has_children))
                or (contour_status == "nested_contours" and (not has_children or roots < 1))):
            return None, "unsupported inconsistent contour-tree status"
        original_contours = tuple(np.asarray(c, dtype=np.int32) for c in contours)
        if any(c.ndim != 2 or c.shape[1] != 2 or len(c) == 0 for c in original_contours):
            return None, "unsupported malformed contour"
        retained = [i for i, row in enumerate(links) if not (row[2] == -1 and row[3] > -1)]
        if not retained:
            return None, "unsupported empty PlantCV object composition"
        # Pinned _object_composition drops non-root leaf contours; preserve
        # every other contour in the exact cv2 return order.
        composed = np.vstack([original_contours[i] for i in retained])
    else:
        return None, f"contour topology unavailable: {contour_status}"
    if len(composed) <= 5:
        return _zero_plantcv_size_bundle(object_in_frame), ""
    if pixel_count <= 0:
        return None, "nonempty composed contour has no foreground pixels"
    if _check_cancel(cancel):
        raise PlantCVSizeCancelledError("PlantCV size geometry cancelled")
    hull = cv2.convexHull(composed)
    hull_area, area = float(cv2.contourArea(hull)), float(pixel_count)
    solidity = area/hull_area if hull_area != 0 else 1
    perimeter, total_edge_length = float(cv2.arcLength(composed, True)), 0.0
    for index, contour in enumerate(original_contours):
        if index % 1024 == 0 and _check_cancel(cancel):
            raise PlantCVSizeCancelledError("PlantCV total-edge calculation cancelled")
        total_edge_length += cv2.arcLength(contour, True)
    _, _, width_px, height_px = cv2.boundingRect(composed)
    cmx, cmy = moment_m10/pixel_count, moment_m01/pixel_count
    ellipse = None
    if (OPENCV_RUNTIME_VERSION != PLANTCV_OPENCV_VERSION
            and _fitellipse_uses_411_perturbation(composed, cancel=cancel)):
        if OPENCV_RUNTIME_VERSION != "5.0.0":
            return None, 'unsupported fitEllipse runtime outside the reviewed 5.0.0 port'
        try:
            ellipse, cross, diagonal_difference = _fitellipse_no_direct_411_details(
                composed, cancel=cancel)
            if _fitellipse_orientation_ambiguous(cross, diagonal_difference):
                return None, ('unsupported fitEllipse orientation numerically unstable '
                    'at pinned atan2 branch')
        except PlantCVSizeResourceError as error:
            return None, f'unsupported_fitEllipse_resource_bound:{error}'
    if ellipse is None:
        ellipse = cv2.fitEllipse(composed)
    ellipse_center, axes, ellipse_angle = ellipse
    major_index = int(np.argmax(axes))
    minor_index = 1-major_index
    # Preserve source Python-float division, including its zero-major failure.
    eccentricity = float(np.sqrt(1-(axes[minor_index]/axes[major_index])**2))
    try:
        longest_path = _plantcv_longest_path(hull=hull, width=width, height=height,
            cmx=cmx, cmy=cmy, cancel=cancel, line_thickness=line_thickness)
    except (RasterSpanLimitError, _UnsupportedSizeObservation) as error:
        return None, f"unsupported_longest_path_resource_or_geometry:{error}"
    return PlantCVSizeObservationBundle(area, hull_area, float(solidity),
        perimeter, float(total_edge_length), int(width_px), int(height_px),
        longest_path, (float(cmx), float(cmy)), len(hull), object_in_frame,
        (float(ellipse_center[0]), float(ellipse_center[1])),
        float(axes[major_index]), float(axes[minor_index]), float(ellipse_angle),
        eccentricity), ""

@dataclass(frozen=True)
class PlantCVSizePremeasurements:
    """Raw mask-domain values, intentionally separate from PlantCV outputs."""

    pixel_count: int
    moment_m10_px: int
    moment_m01_px: int
    centroid_xy_px: tuple[float, float] | None
    mask_bbox_xywh_px: tuple[int, int, int, int] | None
    touches_frame_candidate: bool
    in_bounds_candidate: bool
    boundary_topology: CompactBoundaryTopology | None
    plantcv_observations: PlantCVSizeObservationBundle | None
    plantcv_bundle_available: bool
    plantcv_bundle_unavailable_reason: str
    output_observations: tuple[PlantCVSizeOutputObservation, ...] | None = None
    line_thickness: int = PLANTCV_SIZE_DEFAULT_LINE_THICKNESS


MAX_SIZE_DIAGNOSTIC_POINTS = MAX_CONTOUR_AGGREGATE_POINTS
MAX_SIZE_DIAGNOSTIC_PIXELS = 16_000_000
MAX_SIZE_DIAGNOSTIC_BATCH_PIXELS = 1_000_000
MAX_SIZE_DIAGNOSTIC_SPAN_ITEMS = MAX_RASTER_SPAN_SEGMENTS
MAX_SIZE_DIAGNOSTIC_SCRATCH_BYTES = 96 * 1024 * 1024
MAX_SIZE_DIAGNOSTIC_LABELS = 1024


@dataclass(frozen=True, slots=True)
class _PlantCVSizeDrawGeometry:
    contours_xy: tuple[tuple[tuple[int, int], ...], ...]
    hull_xy: tuple[tuple[int, int], ...]
    bbox_xywh: tuple[int, int, int, int]
    centroid_xy: tuple[int, int]
    longest_path_start_xy: tuple[int, int]
    longest_path_end_xy: tuple[int, int]


def _preflight_size_diagnostic_points(label_results, *, cancel=None):
    if not isinstance(label_results, (tuple, list)):
        raise TypeError('size diagnostics require a bounded list or tuple of measurements')
    if not 1 <= len(label_results) <= MAX_SIZE_DIAGNOSTIC_LABELS:
        raise RasterSpanLimitError('size diagnostic label count exceeds the bound')
    total = 0
    for result in label_results:
        if _check_cancel(cancel):
            raise PlantCVSizeCancelledError('PlantCV size diagnostic preflight cancelled')
        if not isinstance(result, PlantCVSizePremeasurements):
            raise TypeError('size diagnostics require PlantCV size measurements')
        topology = result.boundary_topology
        if topology is None or topology.contours_xy is None:
            continue
        contours = topology.contours_xy
        if not isinstance(contours, (tuple, list)) or len(contours) > MAX_CONTOUR_COMPONENTS:
            raise RasterSpanLimitError('diagnostic contour count exceeds the bound')
        for contour in contours:
            shape = getattr(contour, 'shape', ())
            if len(shape) != 2 or shape[1] != 2:
                raise NotImplementedError('compact size contour shape is unavailable')
            if contour.dtype not in (torch.int32, np.dtype('int32')):
                raise NotImplementedError('compact size contours require int32 coordinates')
            count = int(shape[0])
            if count > MAX_SIZE_DIAGNOSTIC_POINTS - total:
                raise RasterSpanLimitError('aggregate diagnostic contour points exceed the bound')
            total += count
        hierarchy = topology.hierarchy
        if hierarchy is not None:
            shape = tuple(getattr(hierarchy, 'shape', ()))
            if shape not in ((len(contours), 4), (1, len(contours), 4)):
                raise NotImplementedError('compact size hierarchy shape is unavailable')
            if hierarchy.dtype not in (torch.int32, np.dtype('int32')):
                raise NotImplementedError('compact size hierarchy requires int32 links')
    return total


def _opencv_411_circle10_sine_samples():
    # drawing.cpp SinTable indices 0,18,...,450: explicit C++ binary32 literals.
    values = (0.0000000, 0.3090170, 0.5877853, 0.8090170, 0.9510565, 1.0000000,
        0.9510565, 0.8090170, 0.5877853, 0.3090170, 0.0000000, -0.3090170,
        -0.5877853, -0.8090170, -0.9510565, -1.0000000, -0.9510565, -0.8090170,
        -0.5877853, -0.3090170, -0.0000000, 0.3090170, 0.5877853, 0.8090170,
        0.9510565, 1.0000000)
    return tuple(float(np.float32(value)) for value in values)


def _ellipse_ex_circle10_fixed_points(center, *, cancel=None):
    cx, cy = map(int, center)
    center_x, center_y = cx << XY_SHIFT, cy << XY_SHIFT
    axes = 10 << XY_SHIFT
    sine = _opencv_411_circle10_sine_samples()
    points = []
    for index in range(21):
        if index % 8 == 0 and _check_cancel(cancel):
            raise PlantCVSizeCancelledError('PlantCV size diagnostic circle cancelled')
        vx = float(center_x) + axes * sine[25-index]
        vy = float(center_y) + axes * sine[index]
        base_x = int(round(vx / XY_ONE)) << XY_SHIFT
        base_y = int(round(vy / XY_ONE)) << XY_SHIFT
        point = (base_x + int(round(vx-base_x)), base_y + int(round(vy-base_y)))
        if not points or point != points[-1]:
            points.append(point)
    return tuple(points) if len(points) > 1 else ((center_x, center_y), (center_x, center_y))


def _plantcv_circle_outline_spans(width, height, center, *, thickness=5, cancel=None,
                                 work_budget=None):
    if work_budget is None:
        work_budget = [0]
    if thickness == 1:
        return circle_outline_spans(width, height, center, 10,
            cancel=cancel, work_budget=work_budget)
    points = _ellipse_ex_circle10_fixed_points(center, cancel=cancel)
    raw = []
    for index, (first, last) in enumerate(zip(points, points[1:])):
        edge = thick_line_fixed_spans(width, height, first, last, thickness=thickness,
            flags=3 if index == 0 else 2, cancel=cancel, work_budget=work_budget)
        if len(raw) + len(edge) > MAX_RASTER_SPAN_SEGMENTS:
            raise RasterSpanLimitError('centroid circle stroke spans exceed the bound')
        raw.extend(edge)
    _charge_size_geometry_work(work_budget,
        max(1, len(raw)) * max(1, len(raw).bit_length()), cancel)
    return normalize_spans(raw, width=width, height=height, cancel=cancel,
        max_work=MAX_RASTER_SPAN_WORK)


def _size_draw_geometry(result, width, height, *, cancel=None, work_budget=None,
                        line_thickness=5):
    bundle, topology = result.plantcv_observations, result.boundary_topology
    if not result.plantcv_bundle_available or bundle is None:
        raise NotImplementedError('complete size diagnostic observations are unavailable')
    if bundle.area == 0:
        return None  # Public source draws nothing for <=5 composed points.
    if topology is None or topology.contours_xy is None or topology.hierarchy is None:
        raise NotImplementedError('complete size diagnostic contours are unavailable')
    def compact_numpy(value):
        return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
    hierarchy = compact_numpy(topology.hierarchy).reshape(-1, 4)
    if not _valid_retr_hierarchy(hierarchy, cancel=cancel):
        raise NotImplementedError('size diagnostic contour hierarchy is inconsistent')
    contours = []
    for contour in topology.contours_xy:
        _charge_size_geometry_work(work_budget, len(contour), cancel)
        points = compact_numpy(contour)
        if (len(points) == 0 or np.any(points < 0)
                or np.any(points[:, 0] >= width) or np.any(points[:, 1] >= height)):
            raise NotImplementedError('size diagnostic contour coordinates are outside the image')
        contours.append(points.reshape(-1, 1, 2))
    retained = [points for index, points in enumerate(contours)
        if not (hierarchy[index, 3] >= 0 and hierarchy[index, 2] < 0)]
    if not retained:
        raise NotImplementedError('size diagnostic composition is empty')
    composed = np.vstack(retained)
    if len(composed) <= 5:
        return None
    _charge_size_geometry_work(work_budget,
        len(composed) * max(1, len(composed).bit_length()), cancel)
    hull = cv2.convexHull(composed).reshape(-1, 2)
    if _check_cancel(cancel):
        raise PlantCVSizeCancelledError('PlantCV size diagnostic hull cancelled')
    bbox = tuple(int(value) for value in cv2.boundingRect(composed))
    if bbox[2:] != (bundle.width, bundle.height) or bbox != result.mask_bbox_xywh_px:
        raise NotImplementedError('size diagnostic contour and mask bounds differ')
    cmx, cmy = bundle.center_of_mass
    length, start, end = _plantcv_longest_path_details(hull=hull, width=width,
        height=height, cmx=cmx, cmy=cmy, cancel=cancel, work_budget=work_budget,
        line_thickness=line_thickness)
    if not isclose(length, bundle.longest_path, rel_tol=0.0, abs_tol=1e-10):
        raise NotImplementedError('size diagnostic longest-path geometry changed')
    return _PlantCVSizeDrawGeometry(
        tuple(tuple(tuple(map(int, point)) for point in contour[:, 0, :]) for contour in contours),
        tuple(tuple(map(int, point)) for point in hull), bbox, (int(cmx), int(cmy)),
        tuple(map(int, start)), tuple(map(int, end)))


def _size_thick_line_spans(width, height, start, end, *, work_budget, cancel=None,
                           thickness=5):
    _charge_size_geometry_work(work_budget,
        4 * (abs(end[0]-start[0]) + abs(end[1]-start[1]) + 2*thickness + 4), cancel)
    return thick_line_spans(width, height, start, end, thickness=thickness, cancel=cancel)


def _paint_rgb_spans_(canvas, spans, bgr, budget, *, cancel=None, after_write=None):
    """Paint bounded CUDA batches; color arguments are literal OpenCV BGR."""
    height, width = map(int, canvas.shape[:2])
    rgb = tuple(reversed(bgr))
    budget[1] += len(spans)
    if budget[1] > MAX_SIZE_DIAGNOSTIC_SPAN_ITEMS:
        raise RasterSpanLimitError('aggregate diagnostic span count exceeds the bound')
    batch, batch_pixels = [], 0
    flat = canvas.view(-1, 3)
    def flush():
        nonlocal batch, batch_pixels
        if not batch:
            return
        if _check_cancel(cancel):
            raise PlantCVSizeCancelledError('PlantCV size diagnostic painting cancelled')
        # <=1M pixels/batch, <=1M span items. Conservative scratch envelope96MiB
        # includes int64 starts/lengths/offsets and expanded indexing temporaries.
        lengths = torch.as_tensor([right-left+1 for _, left, right in batch],
            dtype=torch.int64, device=canvas.device)
        starts = torch.as_tensor([row*width+left for row, left, _ in batch],
            dtype=torch.int64, device=canvas.device)
        offsets = torch.cumsum(lengths, 0)-lengths
        positions = torch.arange(batch_pixels, dtype=torch.int64, device=canvas.device)
        indices = (torch.repeat_interleave(starts, lengths, output_size=batch_pixels)
            + positions - torch.repeat_interleave(offsets, lengths, output_size=batch_pixels))
        flat[indices] = torch.as_tensor(rgb, dtype=torch.uint8, device=canvas.device)
        if after_write is not None:
            after_write()
        batch, batch_pixels = [], 0
    for row, left, right in spans:
        if _check_cancel(cancel):
            raise PlantCVSizeCancelledError('PlantCV size diagnostic painting cancelled')
        if not 0 <= row < height or not 0 <= left <= right < width:
            raise ValueError('diagnostic paint spans must be clipped to the image')
        while left <= right:
            take = min(right-left+1, MAX_SIZE_DIAGNOSTIC_BATCH_PIXELS-batch_pixels)
            budget[0] += take
            if budget[0] > MAX_SIZE_DIAGNOSTIC_PIXELS:
                raise RasterSpanLimitError('aggregate diagnostic stroke pixels exceed the bound')
            batch.append((row, left, left+take-1))
            batch_pixels += take
            left += take
            if batch_pixels == MAX_SIZE_DIAGNOSTIC_BATCH_PIXELS:
                flush()
    flush()


def render_plantcv_size_diagnostic_rgb(image_rgb, label_results, *, cancel=None,
                                      line_thickness=5):
    """Pinned nonverbose size drawing on a private CUDA RGB copy."""
    line_thickness = _validate_line_thickness(line_thickness)
    if (not torch.is_tensor(image_rgb) or image_rgb.device.type != 'cuda'
            or image_rgb.dtype != torch.uint8 or image_rgb.ndim != 3
            or image_rgb.shape[-1] != 3 or not image_rgb.is_contiguous()):
        raise ValueError('size diagnostics require contiguous CUDA RGB uint8 HxWx3')
    height, width = map(int, image_rgb.shape[:2])
    if min(width, height) < 1 or width*height > MAX_RASTER_PIXELS:
        raise RasterSpanLimitError('diagnostic image exceeds the raster bound')
    _preflight_size_diagnostic_points(label_results, cancel=cancel)
    if any(result.line_thickness != line_thickness for result in label_results):
        raise ValueError('diagnostic thickness must match its size measurements')
    work, geometries = [0], []
    for result in label_results:
        geometries.append(_size_draw_geometry(result, width, height,
            cancel=cancel, work_budget=work, line_thickness=line_thickness))
    if _check_cancel(cancel):
        raise PlantCVSizeCancelledError('PlantCV size diagnostic cancelled before output allocation')
    canvas = image_rgb.clone()
    original_cancel=cancel
    def signature():
        storage=canvas.untyped_storage()
        return (id(canvas),str(canvas.device),str(canvas.dtype),str(canvas.layout),
            tuple(canvas.shape),tuple(canvas.stride()),canvas.storage_offset(),canvas.data_ptr(),
            storage.data_ptr(),storage.nbytes(),canvas._version)
    expected_signature=signature()
    def check_canvas():
        if signature()!=expected_signature:
            raise ValueError('Size diagnostic canvas changed during calculation.')
    def after_write():
        nonlocal expected_signature
        current=signature()
        if current[:-1]!=expected_signature[:-1]:
            raise ValueError('Size diagnostic canvas metadata changed during painting.')
        expected_signature=current
    def paint_cancel():
        check_canvas()
        try:
            stopped=original_cancel() if original_cancel is not None else False
            if type(stopped) is not bool:raise ValueError('Size diagnostic cancellation must return bool.')
            return stopped
        finally:
            check_canvas()
    # Keep the preallocation geometry phase unchanged. Subsequent callbacks
    # guard the owned CUDA canvas as well as the runner's original inputs.
    cancel=paint_cancel
    paint_budget = [0, 0]
    # Geometry preparation and raster painting are separate bounded phases.
    # Each admits at most16M geometry visits; the former retains only bounded
    # compact contours and the latter uses the existing96MiB paint batches.
    # Charging preparation again to painting made hundreds of valid objects
    # exhaust the shared counter before any stroke bound was reached.
    work = [0]
    magenta, blue = (255, 0, 255), (255, 0, 0)
    try:
        for geometry in geometries:
            if geometry is None:
                continue
            spans = closed_polyline_spans(geometry.hull_xy, width, height, thickness=line_thickness,
                cancel=cancel, work_budget=work)
            _paint_rgb_spans_(canvas, spans, magenta, paint_budget, cancel=cancel, after_write=after_write)
            for contour in geometry.contours_xy:
                # CHAIN_APPROX_NONE contours have adjacent points. Normalize
                # their strokes in bounded batches; a shared budget still
                # accounts for every batch and any repainted overlap.
                for spans in iter_adjacent_closed_polyline_spans(contour, width, height,
                        thickness=line_thickness, cancel=cancel, work_budget=work,
                        cache_directions=True):
                    _paint_rgb_spans_(canvas, spans, blue, paint_budget,
                        cancel=cancel, after_write=after_write)
            x, y, bw, bh = geometry.bbox_xywh
            cx, cy = geometry.centroid_xy
            for start, end in (((x, y), (x+bw, y)), ((cx, y), (cx, y+bh))):
                spans = _size_thick_line_spans(width, height, start, end,
                    work_budget=work, cancel=cancel, thickness=line_thickness)
                _paint_rgb_spans_(canvas, spans, magenta, paint_budget, cancel=cancel, after_write=after_write)
            spans = _plantcv_circle_outline_spans(width, height, (cx, cy),
                cancel=cancel, work_budget=work, thickness=line_thickness)
            _paint_rgb_spans_(canvas, spans, magenta, paint_budget, cancel=cancel, after_write=after_write)
            spans = _size_thick_line_spans(width, height, geometry.longest_path_start_xy,
                geometry.longest_path_end_xy, work_budget=work, cancel=cancel,
                thickness=line_thickness)
            _paint_rgb_spans_(canvas, spans, magenta, paint_budget, cancel=cancel, after_write=after_write)
        if _check_cancel(cancel):
            raise PlantCVSizeCancelledError('PlantCV size diagnostic cancelled')
        return canvas
    except RasterSpanCancelled as error:
        raise PlantCVSizeCancelledError(str(error)) from error


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
        raise PlantCVSizeResourceError(
            f"mask exceeds the {MAX_RASTER_PIXELS:,}-pixel reference bound"
        )
    stride_y, stride_x = (int(value) for value in mask.stride())
    if (height > 1 and stride_y <= 0) or (width > 1 and stride_x <= 0):
        raise ValueError("mask must not use overlapping or reversed raster strides")
    if height > 1 and width > 1:
        common = gcd(stride_y, stride_x)
        if stride_x // common < height and stride_y // common < width:
            raise ValueError("mask must not use overlapping raster strides")
    if not bool(((mask == 0) | (mask == 255)).all().item()):
        raise ValueError("mask must contain only binary values 0 and 255")
    return mask


def _tile_neighbor_bits(
    foreground_full: torch.Tensor,
    *,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a tile's foreground raster and occupancy bits including seams."""

    image_height, image_width = (int(value) for value in foreground_full.shape)
    foreground = foreground_full[y0:y1, x0:x1]
    bits = torch.zeros(foreground.shape, dtype=torch.uint8, device=foreground.device)
    for bit_index, (dy, dx) in enumerate(_NEIGHBOR_OFFSETS):
        dst_y0 = max(y0, -dy)
        dst_y1 = min(y1, image_height - dy)
        dst_x0 = max(x0, -dx)
        dst_x1 = min(x1, image_width - dx)
        if dst_y0 >= dst_y1 or dst_x0 >= dst_x1:
            continue
        destination = bits[
            dst_y0 - y0 : dst_y1 - y0,
            dst_x0 - x0 : dst_x1 - x0,
        ]
        source = foreground_full[
            dst_y0 + dy : dst_y1 + dy,
            dst_x0 + dx : dst_x1 + dx,
        ]
        destination.bitwise_or_(source.to(torch.uint8) * (1 << bit_index))
    return foreground, bits


def premeasure_plantcv_binary_mask(mask: torch.Tensor, *,
        cancel: Callable[[], bool] | None = None,
        line_thickness: int = PLANTCV_SIZE_DEFAULT_LINE_THICKNESS) -> PlantCVSizePremeasurements:
    """Compute raw binary-mask statistics on CUDA and export sparse boundary data.

    Empty, external, hole and nested topology can return pixel-unit observations,
    preserving the composed-point >5 gate and source longest-path rule.
    Unsupported topology has explicit unavailable outputs. Compact OpenCV
    geometry records its actual runtime; broad conformance requires same-mask
    comparisons against the isolated pinned reference.
    """

    line_thickness = _validate_line_thickness(line_thickness)
    if cancel is not None and not callable(cancel):
        raise TypeError("cancel must be callable or None")
    if _check_cancel(cancel):
        raise PlantCVSizeCancelledError("size premeasurements cancelled")
    mask = _validate_mask(mask)
    height, width = (int(value) for value in mask.shape)
    foreground_full = mask != 0
    pixel_count = 0
    moment_m10 = 0
    moment_m01 = 0
    min_x, min_y = width, height
    max_x, max_y = -1, -1
    touches_frame = False
    topology_count = 0
    topology_chunks: list[torch.Tensor] = []
    topology_over_limit = False
    # Scan only the occupied bounding rectangle. Neighborhood reads still use
    # the complete CUDA raster and all coordinates remain in the original
    # frame, preserving border tests, moments, contours and ellipse arithmetic.
    occupied_rows = torch.nonzero(foreground_full.any(dim=1), as_tuple=False).flatten()
    occupied_columns = torch.nonzero(foreground_full.any(dim=0), as_tuple=False).flatten()
    scan_y0 = int(occupied_rows[0].item()) if occupied_rows.numel() else 0
    scan_y1 = int(occupied_rows[-1].item())+1 if occupied_rows.numel() else 0
    scan_x0 = int(occupied_columns[0].item()) if occupied_columns.numel() else 0
    scan_x1 = int(occupied_columns[-1].item())+1 if occupied_columns.numel() else 0
    tile_width = max(1, min(scan_x1-scan_x0, MAX_TILE_PIXELS))
    tile_height = max(1, MAX_TILE_PIXELS // tile_width)

    with torch.no_grad():
        for y0 in range(scan_y0, scan_y1, tile_height):
            y1 = min(scan_y1, y0 + tile_height)
            rows = torch.arange(y0, y1, device=mask.device, dtype=torch.int64)
            for x0 in range(scan_x0, scan_x1, tile_width):
                if _check_cancel(cancel):
                    raise PlantCVSizeCancelledError("size premeasurements cancelled")
                x1 = min(scan_x1, x0 + tile_width)
                columns = torch.arange(x0, x1, device=mask.device, dtype=torch.int64)
                foreground, neighbor_bits = _tile_neighbor_bits(
                    foreground_full,
                    y0=y0,
                    y1=y1,
                    x0=x0,
                    x1=x1,
                )
                row_counts = foreground.sum(dim=1, dtype=torch.int64)
                column_counts = foreground.sum(dim=0, dtype=torch.int64)
                pixel_count += int(row_counts.sum().item())
                moment_m10 += int((column_counts * columns).sum().item())
                moment_m01 += int((row_counts * rows).sum().item())

                active_rows = torch.nonzero(row_counts, as_tuple=False).flatten()
                if active_rows.numel():
                    min_y = min(min_y, y0 + int(active_rows[0].item()))
                    max_y = max(max_y, y0 + int(active_rows[-1].item()))
                active_columns = torch.nonzero(column_counts, as_tuple=False).flatten()
                if active_columns.numel():
                    min_x = min(min_x, x0 + int(active_columns[0].item()))
                    max_x = max(max_x, x0 + int(active_columns[-1].item()))

                if y0 == 0 and bool(foreground[0].any().item()):
                    touches_frame = True
                if y1 == height and bool(foreground[-1].any().item()):
                    touches_frame = True
                if x0 == 0 and bool(foreground[:, 0].any().item()):
                    touches_frame = True
                if x1 == width and bool(foreground[:, -1].any().item()):
                    touches_frame = True

                if not topology_over_limit:
                    boundary = foreground & (neighbor_bits != 255)
                    local_yx = torch.nonzero(boundary, as_tuple=False)
                    count = int(local_yx.shape[0])
                    if topology_count + count > MAX_TOPOLOGY_BOUNDARY_POINTS:
                        topology_over_limit = True
                        topology_chunks.clear()
                    else:
                        topology_count += count
                        if count:
                            ys = local_yx[:, 0] + y0
                            xs = local_yx[:, 1] + x0
                            bits = neighbor_bits[local_yx[:, 0], local_yx[:, 1]].to(torch.int32)
                            topology_chunks.append(torch.stack((xs, ys, bits), dim=1).to(torch.int32))

    if pixel_count:
        bbox = (min_x, min_y, max_x - min_x + 1, max_y - min_y + 1)
        centroid = (moment_m10 / pixel_count, moment_m01 / pixel_count)
    else:
        bbox = None
        centroid = None

    if topology_over_limit:
        contours, hierarchy, contour_status = None, None, 'boundary_point_limit_exceeded'
        topology = CompactBoundaryTopology(
            xy_neighbor_bits=None,
            boundary_point_count=None,
            point_limit=MAX_TOPOLOGY_BOUNDARY_POINTS,
            status="boundary_point_limit_exceeded",
        )
    else:
        if topology_chunks:
            records = torch.cat(topology_chunks, dim=0)
            order = torch.argsort(records[:, 1].to(torch.int64) * width + records[:, 0])
            records = records[order].cpu()
        else:
            records = torch.empty((0, 3), dtype=torch.int32, device="cpu")
        if _check_cancel(cancel):
            raise PlantCVSizeCancelledError("size topology export cancelled")
        contours, hierarchy, contour_status = _trace_sparse_contours(
            records.numpy(), width=width, height=height, cancel=cancel)
        topology = CompactBoundaryTopology(
            xy_neighbor_bits=records,
            boundary_point_count=topology_count,
            point_limit=MAX_TOPOLOGY_BOUNDARY_POINTS,
            status="sparse_boundary_row_major_not_contour_order",
            contours_xy=None if contours is None else tuple(torch.from_numpy(c.copy()) for c in contours),
            hierarchy=None if hierarchy is None else torch.from_numpy(hierarchy.copy()),
            contour_status=contour_status,
        )

    if _check_cancel(cancel):
        raise PlantCVSizeCancelledError("size premeasurements cancelled")
    bundle, reason = _plantcv_size_observations(
        contours=contours, hierarchy=hierarchy, contour_status=contour_status,
        width=width, height=height, pixel_count=pixel_count,
        moment_m10=moment_m10, moment_m01=moment_m01,
        object_in_frame=not touches_frame, cancel=cancel, line_thickness=line_thickness)
    if _check_cancel(cancel):
        raise PlantCVSizeCancelledError('PlantCV size observations cancelled')
    return PlantCVSizePremeasurements(
        pixel_count=pixel_count,
        moment_m10_px=moment_m10,
        moment_m01_px=moment_m01,
        centroid_xy_px=centroid,
        mask_bbox_xywh_px=bbox,
        touches_frame_candidate=touches_frame,
        in_bounds_candidate=not touches_frame,
        boundary_topology=topology,
        plantcv_observations=bundle,
        plantcv_bundle_available=bundle is not None,
        plantcv_bundle_unavailable_reason=reason,
        output_observations=None if bundle is None else _plantcv_size_output_observations(bundle),
        line_thickness=line_thickness,
    )


__all__ = [
    "MAX_RASTER_PIXELS",
    "MAX_TILE_PIXELS",
    "MAX_TOPOLOGY_BOUNDARY_POINTS",
    "PLANTCV_OPENCV_PYTHON_VERSION",
    "PLANTCV_SIZE_SOURCE_REFERENCE_ENV_SHA256",
    "PLANTCV_VERSION",
    "CompactBoundaryTopology",
    "PlantCVSizePremeasurements",
    "PlantCVSizeObservationBundle",
    "PlantCVSizeOutputObservation",
    "OPENCV_RUNTIME_VERSION",
    "PlantCVSizeResourceError",
    "PlantCVSizeCancelledError",
    "premeasure_plantcv_binary_mask",
]
