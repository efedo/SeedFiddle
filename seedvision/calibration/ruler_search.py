"""Find straight ruler bodies anywhere in an image, at any angle.

A ruler's two long sides are the strongest pair of parallel straight edges a
ruler-width apart. Edge points vote for lines at every angle in one GPU
accumulator; each angle's histogram is paired with itself across the
plausible ruler widths. Regions supplied by the caller, such as the colour
chart, do not vote. Only the few best line pairs and their extents along the
ruler are downloaded.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from seedvision.cuda import gradient_magnitude

_ANGLE_STEPS = 720
_SUPPRESSED_DEGREES = 3.0


@dataclass(frozen=True, slots=True)
class RulerCandidate:
    """A ruler-width band in the working image, as two parallel lines.

    ``u`` runs along the band at ``angle_degrees`` from the image x axis and
    ``v`` across it, so a point is ``x = u cos a - v sin a``,
    ``y = u sin a + v cos a``.
    """

    angle_degrees: float
    top_v: float
    bottom_v: float
    u_min: float
    u_max: float
    pair_score: float


def polygon_mask(shape: tuple[int, int], polygon: np.ndarray, device):
    """Boolean mask of a convex polygon's interior on the tensor device."""

    import torch

    height, width = shape
    points = np.asarray(polygon, np.float64).reshape(-1, 2)
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    signed_area = float(np.sum(points[:, 0] * np.roll(points[:, 1], -1) - np.roll(points[:, 0], -1) * points[:, 1]))
    orientation = 1.0 if signed_area >= 0.0 else -1.0
    inside = torch.ones((height, width), dtype=torch.bool, device=device)
    for (x0, y0), (x1, y1) in zip(points, np.roll(points, -1, axis=0)):
        cross = (x1 - x0) * (yy - y0) - (y1 - y0) * (xx - x0)
        inside &= cross * orientation >= 0.0
    return inside


def ruler_candidates(gray, excluded=(), *, count: int = 4) -> list[RulerCandidate]:
    """The strongest ruler-width line pairs in a 1x1xHxW grey tensor.

    Widths are taken from the short image side, so a ruler along either side
    of a portrait or landscape photograph is treated alike. Candidates are
    returned strongest first; neighbours within a few degrees and one ruler
    width of a stronger pair are suppressed.
    """

    import torch
    import torch.nn.functional as functional

    height, width = gray.shape[-2:]
    gradient, _, _ = gradient_magnitude(gray, scharr=True)
    magnitude = gradient[0, 0]
    allowed = torch.ones_like(magnitude, dtype=torch.bool)
    for polygon in excluded:
        allowed &= ~polygon_mask((height, width), polygon, magnitude.device)
    if int(allowed.sum()) < 64:
        return []
    if float(magnitude[allowed].max()) < 4.0:
        return []
    # Most of a plain background has no gradient at all, so the quantile
    # alone can fall below any real edge; it is floored.
    threshold = torch.clamp(torch.quantile(magnitude[allowed], 0.92), min=2.0)
    edge = allowed & (magnitude > threshold)
    points = torch.nonzero(edge, as_tuple=False).to(torch.float32)
    if points.shape[0] < 80:
        return []
    y, x = points[:, 0], points[:, 1]

    # Angles cover every line orientation once: a horizontal ruler reads left
    # to right and a vertical one top to bottom.
    angles = torch.linspace(-90.0 + 180.0 / _ANGLE_STEPS, 90.0, _ANGLE_STEPS, device=magnitude.device)
    bins = height + 2 * width + 1
    histogram = torch.zeros((_ANGLE_STEPS, bins), device=magnitude.device)
    chunk = 48
    for start in range(0, _ANGLE_STEPS, chunk):
        radians = torch.deg2rad(angles[start:start + chunk])[:, None]
        v = -torch.sin(radians) * x[None] + torch.cos(radians) * y[None]
        index = (torch.round(v).long() + width).clamp(0, bins - 1)
        rows = radians.shape[0]
        flat = (torch.arange(rows, device=magnitude.device)[:, None] * bins + index).reshape(-1)
        histogram[start:start + rows] = torch.bincount(flat, minlength=rows * bins).reshape(rows, bins).float()

    short_side = min(height, width)
    minimum_width = max(12, round(short_side * 0.025))
    maximum_width = max(minimum_width + 2, round(short_side * 0.23))
    span = maximum_width - minimum_width + 1
    # min(H[v], H[v + s]) maximised over s equals min(H[v], max_s H[v + s]).
    shifted = functional.pad(histogram, (0, maximum_width))[:, minimum_width:minimum_width + bins + span - 1]
    partner, partner_index = functional.max_pool1d(shifted[:, None], span, stride=1, return_indices=True)
    partner = partner[:, 0, :bins]
    partner_index = partner_index[:, 0, :bins] + minimum_width
    pair = torch.minimum(histogram, partner)

    angle_values = angles.detach().cpu().numpy()
    suppressed = int(round(_SUPPRESSED_DEGREES * _ANGLE_STEPS / 180.0))
    found: list[tuple[float, float, float, float]] = []
    for _ in range(count):
        flat_index = int(torch.argmax(pair))
        angle_index, low_bin = divmod(flat_index, bins)
        score = float(pair[angle_index, low_bin])
        if score <= 0.0:
            break
        high_bin = int(partner_index[angle_index, low_bin])
        found.append((score, float(angle_values[angle_index]), float(low_bin - width), float(high_bin - width)))
        for step in range(-suppressed, suppressed + 1):
            row = angle_index + step
            low, high = low_bin - width - maximum_width, high_bin - width + maximum_width
            if not 0 <= row < _ANGLE_STEPS:
                # Orientations wrap at 180 degrees, where v changes sign.
                row %= _ANGLE_STEPS
                low, high = -high, -low
            pair[row, max(0, low + width):max(0, min(bins, high + width + 1))] = 0.0

    candidates = []
    for score, angle, top_v, bottom_v in found:
        extent = _band_extent(x, y, angle, top_v, bottom_v)
        if extent is None:
            continue
        angle, top_v, bottom_v = _refine_band(x, y, angle, top_v, bottom_v, *extent)
        extent = _band_extent(x, y, angle, top_v, bottom_v)
        if extent is not None:
            candidates.append(RulerCandidate(angle, top_v, bottom_v, extent[0], extent[1], score))
    return candidates


def _refine_band(
    x, y, angle: float, top_v: float, bottom_v: float, u_min: float, u_max: float
) -> tuple[float, float, float]:
    """Least-squares direction and offsets of a band's two long edges.

    Accumulator angles are a quarter degree apart, and where several score
    alike the first is taken, so a tilted ruler can come out a degree off. A
    line through each edge's points within the band's extent recovers the
    ruler's own direction; the two slopes are averaged, weighted by points.
    """

    for tolerance in (4.0, 3.0, 2.0):
        radians = np.deg2rad(angle)
        cosine, sine = float(np.cos(radians)), float(np.sin(radians))
        v = -sine * x + cosine * y
        u = cosine * x + sine * y
        within = (u >= u_min) & (u <= u_max)
        slopes, weights, centroids = [], [], []
        for line_v in (top_v, bottom_v):
            near = within & ((v - line_v).abs() <= tolerance)
            points = int(near.sum())
            if points < 20:
                return angle, top_v, bottom_v
            along = u[near].double()
            across = v[near].double()
            centred = along - along.mean()
            slopes.append(float((centred * (across - across.mean())).sum() / centred.square().sum().clamp_min(1e-9)))
            weights.append(points)
            centroids.append((float(x[near].double().mean()), float(y[near].double().mean())))
        slope = float(np.average(slopes, weights=weights))
        refined = angle + float(np.degrees(np.arctan(slope)))
        flipped = refined > 90.0 or refined <= -90.0
        if flipped:
            # Orientations wrap at 180 degrees, where u and v change sign.
            refined -= 180.0 if refined > 90.0 else -180.0
            u_min, u_max = -u_max, -u_min
        radians = np.deg2rad(refined)
        cosine, sine = float(np.cos(radians)), float(np.sin(radians))
        offsets = sorted(-sine * cx + cosine * cy for cx, cy in centroids)
        angle, top_v, bottom_v = refined, offsets[0], offsets[1]
    return angle, top_v, bottom_v


def _band_extent(x, y, angle: float, top_v: float, bottom_v: float) -> tuple[float, float] | None:
    """The longest stretch along a band where both of its edges are present.

    A ruler's two long sides run its whole length. Other structures crossing
    the band, such as a colour chart whose edge happens to continue one side,
    or seeds beyond the ruler's end, lack one side or both and are excluded;
    gaps of up to two ruler widths, such as glare, are bridged.
    """

    radians = np.deg2rad(angle)
    cosine, sine = float(np.cos(radians)), float(np.sin(radians))
    v = -sine * x + cosine * y
    along = cosine * x + sine * y
    tolerance = 2.5
    on_top = (v - top_v).abs() <= tolerance
    on_bottom = (v - bottom_v).abs() <= tolerance
    top_u = along[on_top].detach().cpu().numpy()
    bottom_u = along[on_bottom].detach().cpu().numpy()
    if top_u.size < 20 or bottom_u.size < 20:
        return None
    band_width = max(bottom_v - top_v, 1.0)
    bin_width = max(4.0, band_width * 0.5)
    origin = float(min(top_u.min(), bottom_u.min()))
    size = int((max(top_u.max(), bottom_u.max()) - origin) / bin_width) + 1
    top_counts = np.bincount(((top_u - origin) / bin_width).astype(int), minlength=size)
    bottom_counts = np.bincount(((bottom_u - origin) / bin_width).astype(int), minlength=size)
    counts = top_counts + bottom_counts
    occupied = np.flatnonzero((top_counts > 0) & (bottom_counts > 0))
    if occupied.size == 0:
        return None
    gap = max(2, int(round(2.0 * band_width / bin_width)))
    breaks = np.flatnonzero(np.diff(occupied) > gap)
    starts = np.concatenate(((0,), breaks + 1))
    stops = np.concatenate((breaks, (occupied.size - 1,)))
    totals = [counts[occupied[first]:occupied[last] + 1].sum() for first, last in zip(starts, stops)]
    best = int(np.argmax(totals))
    first_bin, last_bin = occupied[starts[best]], occupied[stops[best]]
    return origin + first_bin * bin_width, origin + (last_bin + 1) * bin_width
