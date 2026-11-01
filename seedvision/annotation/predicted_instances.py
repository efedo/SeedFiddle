"""Seed-outline drafts built from a model's predicted instances.

Predictions live in the analysis dish crop; seed outlines are edited in the
whole corrected image. These CPU helpers place one in the other and add
predictions alongside outlines the operator has already drawn.
"""

from __future__ import annotations

import numpy as np

_MAXIMUM_LABEL = int(np.iinfo(np.uint16).max)


def outlines_from_crop_instances(crop_labels, crop_offset, corrected_shape) -> np.ndarray:
    """Place dish-crop instance labels into an empty corrected-image label map."""

    crop = np.asarray(crop_labels)
    if crop.ndim != 2:
        raise ValueError("Predicted instances must be a 2-D label map.")
    if crop.size and (int(crop.min()) < 0 or int(crop.max()) > _MAXIMUM_LABEL):
        raise ValueError("Predicted instance IDs must fit 16-bit seed labels.")
    offset_x, offset_y = (int(value) for value in crop_offset)
    height, width = (int(value) for value in corrected_shape)
    if (offset_x < 0 or offset_y < 0 or offset_y + crop.shape[0] > height
            or offset_x + crop.shape[1] > width):
        raise ValueError("The predicted instance crop lies outside the corrected image.")
    labels = np.zeros((height, width), dtype=np.uint16)
    labels[offset_y:offset_y + crop.shape[0], offset_x:offset_x + crop.shape[1]] = crop
    return labels


def add_unoutlined_instances(existing, predicted) -> tuple[np.ndarray, int]:
    """Add each predicted seed that is mostly not outlined yet.

    Existing outlines and their IDs are never changed: an added seed takes only
    unlabelled pixels and the next free ID. A prediction more than half covered
    by existing outlines is treated as already outlined.
    """

    existing = np.asarray(existing, dtype=np.uint16)
    predicted = np.asarray(predicted)
    if existing.shape != predicted.shape:
        raise ValueError("Existing and predicted seed labels must share one frame.")
    count = int(predicted.max(initial=0))
    free = existing == 0
    identifiers = predicted.ravel()
    area = np.bincount(identifiers, minlength=count + 1)
    free_area = np.bincount(identifiers, weights=free.ravel(), minlength=count + 1)
    added = np.flatnonzero(free_area * 2 > area)
    added = added[added > 0]
    first = int(existing.max(initial=0)) + 1
    if first + len(added) - 1 > _MAXIMUM_LABEL:
        raise ValueError("Adding these seeds would exceed 16-bit seed label IDs.")
    mapping = np.zeros(count + 1, dtype=np.uint16)
    mapping[added] = np.arange(first, first + len(added), dtype=np.uint16)
    relabelled = mapping[predicted]
    merged = existing.copy()
    target = free & (relabelled > 0)
    merged[target] = relabelled[target]
    return merged, int(len(added))
