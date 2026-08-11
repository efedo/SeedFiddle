"""Ground-truth target construction for both learned instance models."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class DenseSeedTargets:
    interior: np.ndarray
    physical_boundary: np.ndarray
    pattern_boundary: np.ndarray
    pattern_valid: np.ndarray
    centre: np.ndarray
    distance: np.ndarray
    valid: np.ndarray

    def as_channels(self) -> np.ndarray:
        return np.stack(
            (
                self.interior,
                self.physical_boundary,
                self.pattern_boundary,
                self.pattern_valid,
                self.centre,
                self.distance,
                self.valid,
            ),
            axis=0,
        ).astype(np.float32, copy=False)


@dataclass(frozen=True, slots=True)
class StarDistTargets:
    object_probability: np.ndarray
    radial_distances: np.ndarray
    radial_valid: np.ndarray
    valid: np.ndarray


def relabel_consecutive(labels: np.ndarray) -> np.ndarray:
    values = np.asarray(labels)
    if values.ndim != 2:
        raise ValueError("Instance labels must be a two-dimensional array.")
    if np.any(values < 0):
        raise ValueError("Instance labels cannot be negative.")
    identifiers = np.unique(values)
    identifiers = identifiers[identifiers > 0]
    lookup = np.zeros(int(identifiers[-1]) + 1 if len(identifiers) else 1, np.int32)
    lookup[identifiers] = np.arange(1, len(identifiers) + 1, dtype=np.int32)
    return lookup[values.astype(np.int64, copy=False)]


def physical_boundary_mask(labels: np.ndarray, *, width: int = 2) -> np.ndarray:
    """Return foreground-side boundaries, including contacts between IDs."""

    labels = relabel_consecutive(labels)
    foreground = labels > 0
    boundary = np.zeros(labels.shape, dtype=bool)
    boundary[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    boundary[:, :-1] |= labels[:, 1:] != labels[:, :-1]
    boundary[1:, :] |= labels[1:, :] != labels[:-1, :]
    boundary[:-1, :] |= labels[1:, :] != labels[:-1, :]
    boundary &= foreground
    width = max(1, int(width))
    if width > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * width - 1,) * 2)
        boundary = cv2.dilate(np.uint8(boundary), kernel) > 0
        boundary &= foreground
    return boundary.astype(np.float32)


def build_dense_targets(
    labels: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
    pattern_boundary: np.ndarray | None = None,
    pattern_valid: np.ndarray | None = None,
    boundary_width: int = 2,
) -> DenseSeedTargets:
    """Derive dense U-Net targets without inventing missing pattern labels."""

    labels = relabel_consecutive(labels)
    shape = labels.shape
    valid = (
        np.ones(shape, dtype=bool)
        if valid_mask is None
        else np.asarray(valid_mask, dtype=bool)
    )
    if valid.shape != shape:
        raise ValueError("The valid mask must have the same shape as the labels.")
    if pattern_boundary is None:
        pattern = np.zeros(shape, dtype=np.float32)
        pattern_known = np.zeros(shape, dtype=np.float32)
    else:
        pattern = np.asarray(pattern_boundary, dtype=bool)
        if pattern.shape != shape:
            raise ValueError("The pattern-boundary mask must match the labels.")
        pattern_known = (
            (labels > 0)
            if pattern_valid is None
            else np.asarray(pattern_valid, dtype=bool)
        )
        if pattern_known.shape != shape:
            raise ValueError("The pattern-valid mask must match the labels.")
        pattern &= labels > 0
        pattern_known &= labels > 0
        pattern = pattern.astype(np.float32)
        pattern_known = pattern_known.astype(np.float32)

    interior = (labels > 0).astype(np.float32)
    boundary = physical_boundary_mask(labels, width=boundary_width)
    centre = np.zeros(shape, dtype=np.float32)
    distance = np.zeros(shape, dtype=np.float32)
    for identifier in range(1, int(labels.max(initial=0)) + 1):
        region = np.uint8(labels == identifier)
        if not np.any(region):
            continue
        transform = cv2.distanceTransform(region, cv2.DIST_L2, 5)
        maximum = float(transform.max())
        if maximum <= 0:
            continue
        normalized = transform / maximum
        distance[region > 0] = normalized[region > 0]
        peak_y, peak_x = np.unravel_index(int(np.argmax(transform)), shape)
        area = int(np.count_nonzero(region))
        equivalent_diameter = 2.0 * np.sqrt(area / np.pi)
        sigma = max(1.0, equivalent_diameter * 0.075)
        radius = max(2, int(np.ceil(3.0 * sigma)))
        y0, y1 = max(0, peak_y - radius), min(shape[0], peak_y + radius + 1)
        x0, x1 = max(0, peak_x - radius), min(shape[1], peak_x + radius + 1)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        gaussian = np.exp(
            -((xx - peak_x) ** 2 + (yy - peak_y) ** 2) / (2.0 * sigma * sigma)
        ).astype(np.float32)
        local_region = region[y0:y1, x0:x1] > 0
        centre[y0:y1, x0:x1] = np.maximum(
            centre[y0:y1, x0:x1], gaussian * local_region
        )

    valid_float = valid.astype(np.float32)
    return DenseSeedTargets(
        interior=interior * valid_float,
        physical_boundary=boundary * valid_float,
        pattern_boundary=pattern * valid_float,
        pattern_valid=pattern_known * valid_float,
        centre=centre * valid_float,
        distance=distance * valid_float,
        valid=valid_float,
    )

def build_stardist_targets(
    labels: np.ndarray,
    *,
    ray_count: int = 32,
    valid_mask: np.ndarray | None = None,
    maximum_distance: int | None = None,
) -> StarDistTargets:
    """Calculate first-exit radial distances from every labelled pixel.

    This dependency-free reference implementation is intentionally exact at
    pixel resolution and is suitable for cached target generation. Training
    code should cache its output rather than recomputing it every epoch.
    """

    if not 8 <= int(ray_count) <= 256:
        raise ValueError("ray_count must be between 8 and 256.")
    labels = relabel_consecutive(labels)
    height, width = labels.shape
    valid = (
        np.ones(labels.shape, dtype=bool)
        if valid_mask is None
        else np.asarray(valid_mask, dtype=bool)
    )
    if valid.shape != labels.shape:
        raise ValueError("The valid mask must have the same shape as the labels.")
    dense = build_dense_targets(labels, valid_mask=valid, boundary_width=1)
    object_probability = dense.distance.copy()
    radial = np.zeros((int(ray_count), height, width), dtype=np.float32)
    radial_valid = np.zeros_like(radial)
    rows, columns = np.nonzero((labels > 0) & valid)
    if not len(rows):
        return StarDistTargets(object_probability, radial, radial_valid, valid.astype(np.float32))
    source_ids = labels[rows, columns]
    limit = int(maximum_distance or np.ceil(np.hypot(height, width)))
    angles = np.arange(int(ray_count), dtype=np.float32) * (
        2.0 * np.pi / int(ray_count)
    )
    for ray_index, angle in enumerate(angles):
        active = np.ones(len(rows), dtype=bool)
        found = np.zeros(len(rows), dtype=bool)
        values = np.zeros(len(rows), dtype=np.float32)
        cosine, sine = float(np.cos(angle)), float(np.sin(angle))
        for step in range(1, limit + 1):
            if not np.any(active):
                break
            active_indices = np.flatnonzero(active)
            sample_x = np.rint(columns[active_indices] + cosine * step).astype(np.int32)
            sample_y = np.rint(rows[active_indices] + sine * step).astype(np.int32)
            inside = (
                (sample_x >= 0)
                & (sample_x < width)
                & (sample_y >= 0)
                & (sample_y < height)
            )
            same = np.zeros(len(active_indices), dtype=bool)
            valid_indices = active_indices[inside]
            if len(valid_indices):
                same[inside] = (
                    labels[sample_y[inside], sample_x[inside]]
                    == source_ids[valid_indices]
                )
            exited = ~same
            exited_indices = active_indices[exited]
            if len(exited_indices):
                values[exited_indices] = max(0.5, step - 0.5)
                found[exited_indices] = True
                active[exited_indices] = False
        radial[ray_index, rows, columns] = values
        radial_valid[ray_index, rows, columns] = found.astype(np.float32)
    return StarDistTargets(
        object_probability=object_probability,
        radial_distances=radial,
        radial_valid=radial_valid,
        valid=valid.astype(np.float32),
    )
