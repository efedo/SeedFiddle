"""Find a grid colour chart anywhere in an image, at any orientation.

The chart is found from its swatches rather than from its card: uniform,
square-ish regions of similar size whose centres form a regular grid. A
homography from grid coordinates to the image absorbs perspective. Each row's
and column's shared printed edges are then refined on the full-resolution
tensor, and the swatch colours identify a known chart and which way up it
lies. Pixel work stays on the tensor device; only per-component statistics,
edge profiles and swatch means are downloaded.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from seedvision.calibration.geometry import CalibrationDetectionError
from seedvision.cuda import (
    binary_erode,
    connected_components,
    gaussian_blur,
    gradient_magnitude,
    resize,
)


@dataclass(frozen=True, slots=True)
class ChartReference:
    """A known chart in its upright orientation, swatches in reading order."""

    name: str
    rows: int
    columns: int
    rgb: tuple[tuple[int, int, int], ...]


# Median source colours of the card in the image fixtures (its upright
# orientation: white top left, greys down the left column). Identification
# compares the order of swatches in each channel, so camera exposure, white
# balance and tone curve do not need to match.
SPYDERCHECKR_24 = ChartReference(
    "Datacolor SpyderCHECKR 24",
    6,
    4,
    (
        (218, 221, 227), (41, 136, 190), (211, 133, 29), (116, 198, 192),
        (191, 193, 200), (182, 76, 156), (74, 81, 188), (138, 133, 200),
        (158, 158, 166), (217, 208, 50), (194, 80, 93), (74, 111, 63),
        (112, 111, 118), (167, 38, 39), (74, 44, 110), (98, 124, 177),
        (64, 63, 68), (63, 154, 77), (150, 197, 71), (184, 149, 128),
        (30, 30, 35), (41, 34, 153), (207, 162, 36), (91, 60, 45),
    ),
)

# X-Rite (now Calibrite) published sRGB values; the Passport and Mini share
# this layout.
COLORCHECKER_CLASSIC_24 = ChartReference(
    "X-Rite ColorChecker Classic 24",
    4,
    6,
    (
        (115, 82, 68), (194, 150, 130), (98, 122, 157), (87, 108, 67),
        (133, 128, 177), (103, 189, 170),
        (214, 126, 44), (80, 91, 166), (193, 90, 99), (94, 60, 108),
        (157, 188, 64), (224, 163, 46),
        (56, 61, 150), (70, 148, 73), (175, 54, 60), (231, 199, 31),
        (187, 86, 149), (8, 133, 161),
        (243, 243, 242), (200, 200, 200), (160, 160, 160), (122, 122, 121),
        (85, 85, 85), (52, 52, 52),
    ),
)

KNOWN_CHARTS = (SPYDERCHECKR_24, COLORCHECKER_CLASSIC_24)


@dataclass(frozen=True, slots=True)
class ChartGrid:
    """A located swatch grid in image reading order (row 0 at the top)."""

    rows: int
    columns: int
    corners: np.ndarray
    centres: np.ndarray
    mean_bgr: np.ndarray
    outline: np.ndarray
    detected: np.ndarray
    residual: float


@dataclass(frozen=True, slots=True)
class ChartIdentity:
    """Which known chart a grid is, and how its upright layout maps onto it."""

    chart: ChartReference | None
    rotation_degrees: int
    canonical_index: np.ndarray
    residual: float


_WORKING_SIDE = 1200.0
_MINIMUM_SWATCHES = 12
_MINIMUM_PER_AXIS = 3
_CANONICAL_PITCH = 100


def locate_chart_grid(source) -> ChartGrid:
    """Find the most complete colourful swatch grid in a BGR NCHW tensor."""

    import torch

    height, width = source.shape[-2:]
    scale = min(1.0, _WORKING_SIDE / max(height, width))
    small = (
        resize(source, (max(32, round(height * scale)), max(32, round(width * scale))), mode="area")
        if scale < 1.0
        else source
    )
    smooth = gaussian_blur(small, 1.0)
    channels = smooth[0][:, None]
    gradient, _, _ = gradient_magnitude(channels, scharr=True)
    edge = gradient.amax(dim=0, keepdim=True)
    threshold = torch.clamp(torch.median(edge) * 3.0, 3.0, 10.0)
    uniform = binary_erode(edge < threshold, 3)
    labels, statistics = connected_components(uniform)
    candidates = _swatch_candidates(labels, statistics, smooth)
    excluded = np.zeros(len(candidates["centre"]), bool)
    for _ in range(3):
        lattice = _fit_lattice(candidates, excluded)
        if lattice is None:
            break
        grid = _refine_grid(source, lattice, candidates, scale)
        if grid is not None and _is_colour_chart(grid.mean_bgr):
            return grid
        excluded |= lattice["members"]
    raise CalibrationDetectionError("Colour calibration chart was not detected.")


def identify_chart(grid: ChartGrid) -> ChartIdentity:
    """Match the grid's colours to a known chart, its rotation and position.

    Each channel's swatches are compared by rank, which any monotonic camera
    response and white balance preserve. A grid smaller than the chart, such
    as one cut by the image edge, is matched against every window of the
    rotated chart. The best match must be clearly better than every other
    placement; otherwise the chart is reported as unknown.
    """

    measured = grid.mean_bgr[..., ::-1].reshape(-1, 3).astype(np.float64)
    measured_ranks = np.column_stack([_ranks(measured[:, channel]) for channel in range(3)])
    reading_order = np.arange(grid.rows * grid.columns).reshape(grid.rows, grid.columns)
    scored = []
    for chart in KNOWN_CHARTS:
        reference = np.asarray(chart.rgb, np.float64).reshape(chart.rows, chart.columns, 3)
        index = np.arange(chart.rows * chart.columns).reshape(chart.rows, chart.columns)
        for quarter in range(4):
            rotated = np.rot90(reference, k=-quarter)
            rotated_index = np.rot90(index, k=-quarter)
            for top in range(rotated.shape[0] - grid.rows + 1):
                for left in range(rotated.shape[1] - grid.columns + 1):
                    window = rotated[top:top + grid.rows, left:left + grid.columns].reshape(-1, 3)
                    disagreement = _rank_disagreement(window, measured_ranks)
                    canonical = rotated_index[top:top + grid.rows, left:left + grid.columns]
                    scored.append((disagreement, chart, quarter * 90, canonical))
    scored.sort(key=lambda item: item[0])
    if scored:
        disagreement, chart, rotation, canonical = scored[0]
        runner_up = scored[1][0] if len(scored) > 1 else np.inf
        if disagreement <= 0.15 and runner_up >= 3.0 * disagreement:
            return ChartIdentity(chart, rotation, np.ascontiguousarray(canonical), float(disagreement))
    return ChartIdentity(None, 0, reading_order, float(scored[0][0]) if scored else float("nan"))


def _ranks(values: np.ndarray) -> np.ndarray:
    """Ranks with ties sharing their mean rank."""

    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    _, inverse = np.unique(values, return_inverse=True)
    totals = np.bincount(inverse, weights=ranks)
    counts = np.bincount(inverse)
    return totals[inverse] / counts[inverse]


def _rank_disagreement(reference: np.ndarray, measured_ranks: np.ndarray) -> float:
    """One minus the mean per-channel Spearman correlation."""

    correlations = []
    for channel in range(3):
        first = _ranks(reference[:, channel]) - (len(reference) - 1) / 2.0
        second = measured_ranks[:, channel] - (len(reference) - 1) / 2.0
        scale = float(np.sqrt(np.sum(first**2) * np.sum(second**2)))
        correlations.append(float(np.sum(first * second)) / scale if scale > 0.0 else 0.0)
    return float(1.0 - np.mean(correlations))


def _is_colour_chart(mean_bgr: np.ndarray) -> bool:
    """Reject regular grids of near-identical cells, such as graph paper."""

    colours = mean_bgr.reshape(-1, 3)
    chroma = colours.max(axis=1) - colours.min(axis=1)
    lightness = colours.mean(axis=1)
    return bool(np.count_nonzero(chroma > 30.0) >= 3 and np.ptp(lightness) > 80.0)


def _swatch_candidates(labels, statistics, image) -> dict[str, np.ndarray]:
    """Moment statistics of compact uniform components, filtered on device."""

    import torch

    count = int(len(statistics["area"]))
    height, width = labels.shape[-2:]
    flat = labels.reshape(-1).long()
    area = torch.bincount(flat, minlength=count).double()
    minimum_area = 60.0
    maximum_area = 0.06 * height * width
    keep = (area >= minimum_area) & (area <= maximum_area)
    keep[0] = False
    if not bool(keep.any()):
        return {"centre": np.zeros((0, 2)), "size": np.zeros(0), "axes": np.zeros((0, 2, 2)), "colour": np.zeros((0, 3))}
    yy, xx = torch.meshgrid(
        torch.arange(height, device=labels.device, dtype=torch.float64),
        torch.arange(width, device=labels.device, dtype=torch.float64),
        indexing="ij",
    )
    x = xx.reshape(-1)
    y = yy.reshape(-1)

    def summed(values):
        return torch.bincount(flat, weights=values, minlength=count)

    safe = area.clamp_min(1.0)
    mean_x = summed(x) / safe
    mean_y = summed(y) / safe
    var_x = summed(x * x) / safe - mean_x**2
    var_y = summed(y * y) / safe - mean_y**2
    cov_xy = summed(x * y) / safe - mean_x * mean_y
    colour = torch.stack(
        [summed(image[0, channel].reshape(-1).double()) / safe for channel in range(3)],
        dim=1,
    )
    selected = torch.nonzero(keep).flatten()
    table = torch.stack(
        (area, mean_x, mean_y, var_x, var_y, cov_xy, colour[:, 0], colour[:, 1], colour[:, 2]),
        dim=1,
    )[selected].cpu().numpy()
    area_np, cx, cy, vx, vy, cxy = (table[:, index] for index in range(6))
    trace = vx + vy
    determinant = np.clip(vx * vy - cxy**2, 1e-9, None)
    root = np.sqrt(np.clip((trace / 2.0) ** 2 - determinant, 0.0, None))
    major, minor = trace / 2.0 + root, np.clip(trace / 2.0 - root, 1e-9, None)
    elongation = np.sqrt(major / minor)
    # A filled rectangle has area == 12 sqrt(det(covariance)); ellipses are
    # close to this as well, but triangles, rings and ragged blobs are not.
    rectangularity = area_np / (12.0 * np.sqrt(determinant))
    usable = (elongation <= 2.0) & (rectangularity >= 0.82) & (rectangularity <= 1.15)
    covariance = np.stack((np.stack((vx, cxy), -1), np.stack((cxy, vy), -1)), -2)
    return {
        "centre": np.column_stack((cx, cy))[usable],
        "size": np.sqrt(area_np[usable]),
        "axes": covariance[usable],
        "colour": table[usable, 6:9],
    }


def _local_lattice_hypotheses(offset, distance, neighbour) -> list[tuple[int, np.ndarray]]:
    """One grid basis per candidate, from its nearest roughly square corner.

    Each candidate proposes the displacement to its nearest neighbour and the
    nearest neighbour roughly perpendicular to it. Bases are local, so seeds
    of swatch size scattered beside the chart cannot outvote its axes.
    """

    hypotheses = []
    for origin in range(len(offset)):
        near = np.flatnonzero(neighbour[origin])
        if near.size < 2:
            continue
        near = near[np.argsort(distance[origin, near])][:8]
        first = offset[origin, near[0]]
        first_length = float(np.linalg.norm(first))
        for other in near[1:]:
            second = offset[origin, other]
            second_length = float(np.linalg.norm(second))
            cosine = abs(float(first @ second)) / max(first_length * second_length, 1e-9)
            if cosine <= 0.35 and 0.6 <= second_length / max(first_length, 1e-9) <= 1.67:
                hypotheses.append((origin, np.column_stack((first, second))))
                break
    return hypotheses


def _lattice_members(offset_from_origin, affine, compatible, tolerance_fraction) -> dict[int, tuple[int, int]]:
    """Candidates on the lattice through the origin, connected to it."""

    pitch = float(np.linalg.norm(affine, axis=0).min())
    grid = offset_from_origin @ np.linalg.inv(affine).T
    rounded = np.round(grid)
    residual = np.linalg.norm((grid - rounded) @ affine.T, axis=1)
    inlier = np.flatnonzero((residual <= tolerance_fraction * pitch) & compatible)
    nodes = rounded[inlier].astype(int)
    kept = _connected_nodes(nodes, inlier, (0, 0))
    return {index: tuple(rounded[index].astype(int)) for index in kept}


def _reading_order_transform(affine: np.ndarray) -> np.ndarray:
    """Signed axis permutation taking lattice steps to image reading order.

    Columns advance along the axis nearer the image horizontal (rightwards),
    rows along the other (downwards), so grid coordinates read like the image.
    """

    first, second = affine[:, 0], affine[:, 1]
    first_horizontal = abs(first[0]) / max(np.linalg.norm(first), 1e-9)
    second_horizontal = abs(second[0]) / max(np.linalg.norm(second), 1e-9)
    if first_horizontal >= second_horizontal:
        transform = np.eye(2, dtype=int)
        column_step, row_step = first, second
    else:
        transform = np.asarray(((0, 1), (1, 0)), int)
        column_step, row_step = second, first
    signs = np.asarray((1 if column_step[0] >= 0 else -1, 1 if row_step[1] >= 0 else -1))
    return signs[:, None] * transform


def _fit_lattice(candidates, excluded) -> dict | None:
    """Fit a regular grid to candidate centres; return its homography and nodes."""

    centres = candidates["centre"]
    sizes = candidates["size"]
    available = np.flatnonzero(~excluded)
    if available.size < _MINIMUM_SWATCHES:
        return None
    centre = centres[available]
    size = sizes[available]
    offset = centre[None, :, :] - centre[:, None, :]
    distance = np.linalg.norm(offset, axis=-1)
    pair_size = (size[:, None] + size[None, :]) * 0.5
    ratio = size[None, :] / size[:, None]
    compatible = (ratio >= 0.7) & (ratio <= 1.43)
    neighbour = compatible & (distance >= 1.02 * pair_size) & (distance <= 2.0 * pair_size)
    hypotheses = _local_lattice_hypotheses(offset, distance, neighbour)
    if not hypotheses:
        return None
    # A coarse count ranks the hypotheses; the best few are refitted on their
    # connected members and recounted at a tighter tolerance, so a chance
    # alignment of seeds cannot outscore a printed grid.
    coarse = []
    for origin, affine in hypotheses:
        if abs(np.linalg.det(affine)) < 1e-6:
            continue
        members = _lattice_members(offset[origin], affine, compatible[origin], 0.25)
        coarse.append((len(members), origin, affine, members))
    coarse.sort(key=lambda item: -item[0])
    best = None
    for count, origin, affine, members in coarse[:40]:
        if count < _MINIMUM_SWATCHES or (best is not None and count <= len(best[2])):
            continue
        nodes = np.asarray(list(members.values()), np.float64)
        design = np.column_stack((nodes, np.ones(len(nodes))))
        solution, *_ = np.linalg.lstsq(design, offset[origin][list(members)], rcond=None)
        refined = solution[:2].T
        if abs(np.linalg.det(refined)) < 1e-6:
            continue
        shifted = offset[origin] - solution[2]
        tight = _lattice_members(shifted, refined, compatible[origin], 0.15)
        if best is None or len(tight) > len(best[2]):
            best = (origin, refined, tight)
    if best is None or len(best[2]) < _MINIMUM_SWATCHES:
        return None
    origin, affine, members = best
    transform = _reading_order_transform(affine)
    members = {index: tuple(int(value) for value in transform @ node) for index, node in members.items()}
    affine = affine @ transform.T
    pitches = [float(value) for value in np.linalg.norm(affine, axis=0)]
    homography = None
    for _ in range(3):
        node_array = np.asarray(list(members.values()), np.float64)
        point_array = centre[list(members)]
        homography = _fit_homography(node_array, point_array)
        if homography is None:
            return None
        local_inverse = np.linalg.inv(homography)
        projected = _apply_homography(local_inverse, centre)
        nearest_node = np.round(projected)
        error = np.linalg.norm(_apply_homography(homography, nearest_node) - centre, axis=1)
        member_size = float(np.median(size[list(members)]))
        size_ok = (size >= 0.7 * member_size) & (size <= 1.43 * member_size)
        accepted = np.flatnonzero((error <= 0.2 * min(pitches)) & size_ok)
        kept = _connected_nodes(nearest_node[accepted].astype(int), accepted, members.get(origin))
        updated = {index: tuple(nearest_node[index].astype(int)) for index in kept}
        if len(updated) < _MINIMUM_SWATCHES:
            return None
        if updated == members:
            break
        members = updated
    members = _trim_sparse_border(members)
    if len(members) < _MINIMUM_SWATCHES:
        return None
    homography = _fit_homography(
        np.asarray(list(members.values()), np.float64), centre[list(members)]
    )
    if homography is None:
        return None
    node_array = np.asarray(list(members.values()), int)
    low = node_array.min(axis=0)
    high = node_array.max(axis=0)
    columns, rows = (high - low + 1).tolist()
    occupied = {tuple(node) for node in node_array}
    if (
        rows < _MINIMUM_PER_AXIS
        or columns < _MINIMUM_PER_AXIS
        or len(occupied) < max(_MINIMUM_SWATCHES, 0.6 * rows * columns)
    ):
        return None
    shift = np.asarray(((1.0, 0.0, low[0]), (0.0, 1.0, low[1]), (0.0, 0.0, 1.0)))
    homography = homography @ shift
    detected = np.zeros((rows, columns), bool)
    for column, row in node_array - low:
        detected[row, column] = True
    members_mask = np.zeros(len(centres), bool)
    members_mask[available[list(members)]] = True
    member_indices = available[list(members)]
    return {
        "homography": homography,
        "rows": rows,
        "columns": columns,
        "detected": detected,
        "members": members_mask,
        "member_indices": member_indices,
        "member_nodes": node_array - low,
        "pitches": pitches,
    }


def _trim_sparse_border(members: dict[int, tuple[int, int]]) -> dict[int, tuple[int, int]]:
    """Drop outer rows and columns that are mostly empty.

    Seeds or printed marks that happen to sit on lattice points beside the
    chart would otherwise widen the grid; a chart's own outer rows and columns
    are full apart from at most an occasional missed swatch.
    """

    members = dict(members)
    while members:
        nodes = np.asarray(list(members.values()), int)
        low, high = nodes.min(axis=0), nodes.max(axis=0)
        span = high - low + 1
        worst = None
        for axis in (0, 1):
            for value in (low[axis], high[axis]):
                occupied = len({tuple(node) for node in nodes if node[axis] == value})
                fraction = occupied / span[1 - axis]
                if fraction < 0.5 and (worst is None or fraction < worst[0]):
                    worst = (fraction, axis, value)
        if worst is None:
            break
        _, axis, value = worst
        members = {index: node for index, node in members.items() if node[axis] != value}
    return members


def _connected_nodes(
    nodes: np.ndarray, indices: np.ndarray, start: tuple[int, int] | None
) -> list[int]:
    """Candidates on the grid component reachable from the start node.

    Nodes may be one missing swatch apart, so a chart with an undetected
    swatch stays one component, but seeds far from the chart do not join it.
    """

    if len(nodes) == 0:
        return []
    by_node: dict[tuple[int, int], int] = {}
    for node, index in zip(map(tuple, nodes), indices):
        by_node.setdefault((int(node[0]), int(node[1])), int(index))
    origin_node = (
        (int(start[0]), int(start[1]))
        if start is not None and (int(start[0]), int(start[1])) in by_node
        else next(iter(by_node))
    )
    seen = {origin_node}
    queue = deque([origin_node])
    while queue:
        column, row = queue.popleft()
        for step_column in range(-2, 3):
            for step_row in range(-2, 3):
                neighbour = (column + step_column, row + step_row)
                if neighbour in by_node and neighbour not in seen:
                    seen.add(neighbour)
                    queue.append(neighbour)
    return [by_node[node] for node in seen]


def _fit_homography(source: np.ndarray, target: np.ndarray) -> np.ndarray | None:
    """Normalised least-squares homography from grid nodes to image points."""

    if len(source) < 4:
        return None

    def normaliser(points):
        mean = points.mean(axis=0)
        spread = np.sqrt(np.mean(np.sum((points - mean) ** 2, axis=1)))
        factor = np.sqrt(2.0) / max(spread, 1e-9)
        return np.asarray(((factor, 0.0, -factor * mean[0]), (0.0, factor, -factor * mean[1]), (0.0, 0.0, 1.0)))

    source_norm, target_norm = normaliser(source), normaliser(target)
    a = _apply_homography(source_norm, source)
    b = _apply_homography(target_norm, target)
    rows = []
    for (x, y), (u, v) in zip(a, b):
        rows.append((-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u))
        rows.append((0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v))
    _, singular, vt = np.linalg.svd(np.asarray(rows))
    if singular[-2] < 1e-9:
        return None
    matrix = vt[-1].reshape(3, 3)
    matrix = np.linalg.inv(target_norm) @ matrix @ source_norm
    if abs(matrix[2, 2]) < 1e-12:
        return None
    return matrix / matrix[2, 2]


def _apply_homography(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, np.float64).reshape(-1, 2)
    homogeneous = np.column_stack((points, np.ones(len(points)))) @ matrix.T
    return homogeneous[:, :2] / homogeneous[:, 2:3]


def _refine_grid(source, lattice, candidates, scale) -> ChartGrid | None:
    """Snap rows and columns to shared printed edges on the full image."""

    import torch
    import torch.nn.functional as functional

    rows, columns = lattice["rows"], lattice["columns"]
    to_full = np.diag((1.0 / scale, 1.0 / scale, 1.0))
    grid_to_image = to_full @ lattice["homography"]
    # Patch half-sizes in grid units from the members' second moments along
    # the local grid axes (the erosion removed about one pixel per side).
    half_sizes = []
    member_indices = lattice["member_indices"]
    member_nodes = lattice["member_nodes"]
    small_homography = lattice["homography"]
    for axis in (0, 1):
        values = []
        for index, node in zip(member_indices, member_nodes):
            step = np.zeros(2)
            step[axis] = 1.0
            ends = _apply_homography(small_homography, np.asarray((node - step * 0.5, node + step * 0.5), np.float64))
            direction = ends[1] - ends[0]
            pitch = float(np.linalg.norm(direction))
            unit = direction / max(pitch, 1e-9)
            variance = float(unit @ candidates["axes"][index] @ unit)
            side = np.sqrt(12.0 * max(variance, 0.0)) + 2.0
            values.append(side / max(pitch, 1e-9) / 2.0)
        half_sizes.append(float(np.clip(np.median(values), 0.2, 0.48)))
    pitch_px = _CANONICAL_PITCH
    canvas_width, canvas_height = columns * pitch_px, rows * pitch_px
    device, dtype = source.device, source.dtype
    u = torch.arange(canvas_width, device=device, dtype=torch.float64) + 0.5
    v = torch.arange(canvas_height, device=device, dtype=torch.float64) + 0.5
    grid_v, grid_u = torch.meshgrid(v, u, indexing="ij")
    node_u = grid_u / pitch_px - 0.5
    node_v = grid_v / pitch_px - 0.5
    matrix = torch.as_tensor(grid_to_image, device=device, dtype=torch.float64)
    mapped = torch.stack((node_u, node_v, torch.ones_like(node_u)), dim=-1) @ matrix.T
    denominator = mapped[..., 2].clamp_min(1e-9)
    image_x = mapped[..., 0] / denominator
    image_y = mapped[..., 1] / denominator
    height, width = source.shape[-2:]
    sample_grid = torch.stack(
        (image_x / max(width - 1, 1) * 2.0 - 1.0, image_y / max(height - 1, 1) * 2.0 - 1.0),
        dim=-1,
    )[None].to(dtype)
    rectified = functional.grid_sample(
        source, sample_grid, mode="bilinear", padding_mode="border", align_corners=True
    )[0]
    step = 2
    vertical = functional.pad(
        torch.sqrt((rectified[:, :, 2 * step:] - rectified[:, :, :-2 * step]).square().mean(dim=0) + 1e-6),
        (step, step, 0, 0),
    )
    horizontal = functional.pad(
        torch.sqrt((rectified[:, 2 * step:, :] - rectified[:, :-2 * step, :]).square().mean(dim=0) + 1e-6),
        (0, 0, step, step),
    )
    edges = []
    for axis, count, other_count, response in (
        (0, columns, rows, vertical),
        (1, rows, columns, horizontal),
    ):
        half = half_sizes[axis] * pitch_px
        other_half = half_sizes[1 - axis] * pitch_px
        gap = pitch_px - 2.0 * half
        into_gap = max(2, int(gap // 2))
        into_swatch = max(into_gap, int(round(0.4 * half)))
        length = into_gap + into_swatch + 1
        expected, starts = [], []
        for index in range(count):
            centre = (index + 0.5) * pitch_px
            opening, closing = round(centre - half), round(centre + half)
            expected.extend((opening, closing))
            starts.extend((opening - into_gap, closing - into_swatch))
        band = max(2, int(0.7 * other_half))
        profiles = []
        for start in starts:
            positions = torch.arange(start, start + length, device=device).clamp(0, (canvas_width if axis == 0 else canvas_height) - 1)
            support = []
            for other in range(other_count):
                middle = int((other + 0.5) * pitch_px)
                if axis == 0:
                    support.append(response[middle - band:middle + band + 1][:, positions].mean(dim=0))
                else:
                    support.append(response[positions][:, middle - band:middle + band + 1].mean(dim=1))
            profiles.append(torch.stack(support).mean(dim=0))
        table = torch.stack(profiles).cpu().numpy()
        refined = [_profile_peak(profile, start, float(guess)) for profile, start, guess in zip(table, starts, expected)]
        pairs = []
        for index in range(count):
            opening, closing = refined[2 * index:2 * index + 2]
            if not 0.7 * 2.0 * half <= closing - opening <= 1.3 * 2.0 * half:
                opening, closing = float(expected[2 * index]), float(expected[2 * index + 1])
            pairs.append((opening, closing))
        edges.append(pairs)
    column_edges, row_edges = edges
    corners = np.zeros((rows, columns, 4, 2))
    centres = np.zeros((rows, columns, 2))
    for row, (top, bottom) in enumerate(row_edges):
        for column, (left, right) in enumerate(column_edges):
            canonical = np.asarray(((left, top), (right, top), (right, bottom), (left, bottom)), np.float64)
            corners[row, column] = _apply_homography(grid_to_image, canonical / pitch_px - 0.5)
            middle = np.asarray((((left + right) / 2.0, (top + bottom) / 2.0),), np.float64)
            centres[row, column] = _apply_homography(grid_to_image, middle / pitch_px - 0.5)[0]
    # A chart cut by the image edge keeps only its whole rows and columns.
    inside = (
        (corners[..., 0] >= -0.5) & (corners[..., 0] <= width - 0.5)
        & (corners[..., 1] >= -0.5) & (corners[..., 1] <= height - 0.5)
    ).all(axis=-1)
    window = _inside_window(inside)
    if window is None:
        return None
    top, bottom, left, right = window
    rows, columns = bottom - top, right - left
    corners = corners[top:bottom, left:right]
    centres = centres[top:bottom, left:right]
    detected = lattice["detected"][top:bottom, left:right]
    outline_canonical = np.asarray(
        (
            (column_edges[left][0], row_edges[top][0]),
            (column_edges[right - 1][1], row_edges[top][0]),
            (column_edges[right - 1][1], row_edges[bottom - 1][1]),
            (column_edges[left][0], row_edges[bottom - 1][1]),
        ),
        np.float64,
    )
    outline = _apply_homography(grid_to_image, outline_canonical / pitch_px - 0.5)
    means = _quadrilateral_means(source, corners.reshape(-1, 4, 2)).reshape(rows, columns, 3)
    kept = (
        (member_nodes[:, 0] >= left) & (member_nodes[:, 0] < right)
        & (member_nodes[:, 1] >= top) & (member_nodes[:, 1] < bottom)
    )
    if not bool(kept.any()):
        return None
    member_centres = candidates["centre"][member_indices[kept]] / scale
    predicted = centres[member_nodes[kept, 1] - top, member_nodes[kept, 0] - left]
    pitch_full = min(lattice["pitches"]) / scale
    residual = float(np.mean(np.linalg.norm(predicted - member_centres, axis=1)) / max(pitch_full, 1e-9))
    return ChartGrid(rows, columns, corners, centres, means, outline, detected, residual)


def _inside_window(inside: np.ndarray) -> tuple[int, int, int, int] | None:
    """Largest-first trim of border rows and columns with cells outside the image."""

    top, bottom, left, right = 0, inside.shape[0], 0, inside.shape[1]
    while not bool(inside[top:bottom, left:right].all()):
        outside = ~inside[top:bottom, left:right]
        _, side = max(
            (int(outside[0].sum()), "top"),
            (int(outside[-1].sum()), "bottom"),
            (int(outside[:, 0].sum()), "left"),
            (int(outside[:, -1].sum()), "right"),
        )
        if side == "top":
            top += 1
        elif side == "bottom":
            bottom -= 1
        elif side == "left":
            left += 1
        else:
            right -= 1
        rows, columns = bottom - top, right - left
        if min(rows, columns) < _MINIMUM_PER_AXIS or rows * columns < _MINIMUM_SWATCHES:
            return None
    return top, bottom, left, right


def _profile_peak(profile: np.ndarray, start: int, expected: float) -> float:
    peak = int(np.argmax(profile))
    baseline = float(np.quantile(profile, 0.30))
    if float(profile[peak]) < baseline + 1.0:
        return expected
    low, high = max(0, peak - 2), min(len(profile), peak + 3)
    weights = np.clip(profile[low:high] - baseline, 0.0, None)
    if float(weights.sum()) <= 1e-6:
        return float(start + peak)
    return float(np.average(np.arange(start + low, start + high, dtype=np.float64), weights=weights))


def _quadrilateral_means(source, corners: np.ndarray) -> np.ndarray:
    """Mean BGR over the central 76% of each quadrilateral, on device."""

    import torch
    import torch.nn.functional as functional

    device = source.device
    quads = torch.as_tensor(corners, device=device, dtype=torch.float32)
    axis = torch.linspace(0.12, 0.88, 11, device=device)
    vv, uu = torch.meshgrid(axis, axis, indexing="ij")
    weights = torch.stack(
        ((1 - uu) * (1 - vv), uu * (1 - vv), uu * vv, (1 - uu) * vv), dim=-1
    )
    points = torch.einsum("hwk,pkc->phwc", weights, quads)
    height, width = source.shape[-2:]
    grid = torch.stack(
        (points[..., 0] / max(width - 1, 1) * 2.0 - 1.0, points[..., 1] / max(height - 1, 1) * 2.0 - 1.0),
        dim=-1,
    ).reshape(1, -1, 1, 2).to(source.dtype)
    samples = functional.grid_sample(source, grid, mode="bilinear", padding_mode="border", align_corners=True)
    count = corners.shape[0]
    return samples[0, :, :, 0].reshape(3, count, -1).mean(dim=-1).T.cpu().numpy().astype(np.float64)
