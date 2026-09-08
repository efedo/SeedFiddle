"""Compact geometry derived from editable CPU annotation labels, not predictions."""

from __future__ import annotations

import math
import numpy as np


def annotation_centres(labels) -> dict[int, tuple[float, float]]:
    """Area centroids in image coordinates, with bounded row-chunk temporaries.

    Include all painted pixels of each ID, including disconnected fragments.
    Partial annotations describe the visible painted area, not an inferred body.
    These coordinates are display/metadata products, never detection markers.
    """
    if labels is None:
        return {}
    values = np.asarray(labels)
    if values.ndim != 2 or values.dtype.kind not in "ui":
        raise ValueError("Seed annotation labels must be a 2-D integer raster.")
    if not values.size:
        return {}
    maximum = int(values.max())
    if maximum > 65535 or int(values.min()) < 0:
        raise ValueError("Seed annotation IDs must fit uint16.")
    if maximum == 0:
        return {}
    count = np.zeros(maximum + 1, np.float64)
    sum_x = np.zeros_like(count)
    sum_y = np.zeros_like(count)
    for start in range(0, values.shape[0], 128):
        block = values[start:start + 128]
        y, x = np.nonzero(block)
        ids = block[y, x].astype(np.intp, copy=False)
        count += np.bincount(ids, minlength=maximum + 1)
        sum_x += np.bincount(ids, weights=x, minlength=maximum + 1)
        sum_y += np.bincount(ids, weights=y + start, minlength=maximum + 1)
    return {int(i): (float(sum_x[i] / count[i]), float(sum_y[i] / count[i]))
            for i in np.flatnonzero(count) if i > 0}


def outward_hilum_direction(centre, point):
    """Unit vector from the painted-area centroid to the hilum, or unknown."""
    if centre is None or point is None:
        return None
    dx, dy = float(point[0] - centre[0]), float(point[1] - centre[1])
    length = math.hypot(dx, dy)
    if not math.isfinite(length) or length <= 1e-9:
        return None
    return dx / length, dy / length
