"""Derive conservative boundary-reference evidence from reviewed instances."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class InstanceBoundaryReferences:
    """Boundary supervision derived only from complete annotated instances.

    ``physical_edge`` is the one-pixel inner contour of every instance.
    ``safe_interior`` is the region far enough inside those contours that an
    annotation/rasterisation error cannot turn a real boundary into a negative
    example.  ``non_edge`` is the subset of that safe interior selected by an
    externally supplied candidate-edge mask; flat interior pixels are never
    negative edge examples merely because they lie inside an instance.
    """

    physical_edge: np.ndarray
    non_edge: np.ndarray
    safe_interior: np.ndarray
    interior_buffer_px: float


def instance_boundary_references(
    instance_labels: np.ndarray,
    seed_diameter: float,
    *,
    interior_buffer_fraction: float = 0.08,
    internal_edge_candidates: np.ndarray | None = None,
) -> InstanceBoundaryReferences:
    """Convert complete labelled seeds into conservative boundary references.

    The one-pixel *inner* contour of every positive instance ID is physical-edge
    evidence, including both sides of a contact between two differently labelled
    seeds. Pixels farther than ``interior_buffer_fraction`` of a seed diameter
    from that contour are eligible for non-physical-edge supervision, but only
    where ``internal_edge_candidates`` identifies an actual edge candidate.
    This prevents the much larger population of flat seed-interior pixels from
    overwhelming the classifier. The pixels between the contour and safe
    interior are intentionally left unlabelled so rasterization error or an
    imprecise annotation boundary cannot become false negative evidence.

    Applied instance annotations are treated as reviewed, complete regions. A
    partial painted patch has a real raster contour but that contour does not
    necessarily coincide with a physical seed boundary, so it must remain a
    draft rather than being applied as a complete instance.
    """

    labels = np.asarray(instance_labels)
    if labels.ndim != 2:
        raise ValueError("Seed instance annotations must be a two-dimensional raster.")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("Seed instance annotations must contain integer IDs.")
    if np.any(labels < 0):
        raise ValueError("Seed instance annotation IDs cannot be negative.")
    if not np.isfinite(seed_diameter) or float(seed_diameter) <= 0.0:
        raise ValueError("Seed diameter must be positive and finite.")
    if not 0.0 <= float(interior_buffer_fraction) <= 0.50:
        raise ValueError("The instance-interior buffer must be between 0 and 0.5 diameter.")

    labelled = labels > 0
    physical = np.zeros(labels.shape, dtype=bool)
    if bool(labelled.any()):
        # Padding with ID zero means an instance touching the image boundary still
        # receives an inner contour.  Comparing IDs rather than a binary union is
        # essential: it preserves the physical contact between adjacent instances.
        padded = np.pad(labels, 1, mode="constant", constant_values=0)
        centre = padded[1:-1, 1:-1]
        for y_offset in (-1, 0, 1):
            for x_offset in (-1, 0, 1):
                if x_offset == 0 and y_offset == 0:
                    continue
                neighbour = padded[
                    1 + y_offset : 1 + y_offset + labels.shape[0],
                    1 + x_offset : 1 + x_offset + labels.shape[1],
                ]
                physical |= labelled & (neighbour != centre)

    interior_buffer_px = max(
        1.0,
        float(seed_diameter) * float(interior_buffer_fraction),
    )
    if bool(labelled.any()):
        # Setting the exact contour to zero lets one distance transform measure
        # distance from outer borders *and* contacts between distinct IDs.
        interior_candidates = np.uint8(labelled & ~physical)
        distance = cv2.distanceTransform(
            interior_candidates,
            cv2.DIST_L2,
            cv2.DIST_MASK_PRECISE,
        )
        safe_interior = labelled & (distance >= interior_buffer_px)
    else:
        safe_interior = np.zeros(labels.shape, dtype=bool)

    if internal_edge_candidates is None:
        non_edge = np.zeros(labels.shape, dtype=bool)
    else:
        candidates = np.asarray(internal_edge_candidates, dtype=bool)
        if candidates.shape != labels.shape:
            raise ValueError(
                "Internal-edge candidates must match the instance raster."
            )
        non_edge = safe_interior & candidates

    physical.flags.writeable = False
    non_edge.flags.writeable = False
    safe_interior.flags.writeable = False
    return InstanceBoundaryReferences(
        physical_edge=physical,
        non_edge=non_edge,
        safe_interior=safe_interior,
        interior_buffer_px=interior_buffer_px,
    )


def internal_edge_candidate_mask(
    edge_strength: np.ndarray,
    ridge_strength: np.ndarray,
    safe_interior: np.ndarray,
) -> np.ndarray:
    """Select sparse, locally maximal edge evidence inside reviewed seeds.

    This CPU form mirrors the bounded CUDA prototype-selection rule and is
    used only while exporting explicit training targets.  Flat seed interior
    is retained as reviewed negative context through ``safe_interior`` but is
    never mislabeled as a non-physical edge positive.
    """

    safe = np.asarray(safe_interior, dtype=bool)
    edge = np.asarray(edge_strength, dtype=np.float32)
    ridge = np.asarray(ridge_strength, dtype=np.float32)
    if edge.shape != safe.shape or ridge.shape != safe.shape:
        raise ValueError("Edge, ridge, and safe-interior rasters must match.")
    if not np.isfinite(edge).all() or not np.isfinite(ridge).all():
        raise ValueError("Edge and ridge evidence must be finite.")
    if float(edge.max(initial=0.0)) > 1.0:
        edge = edge / 255.0
    if float(ridge.max(initial=0.0)) > 1.0:
        ridge = ridge / 255.0
    edge = np.clip(edge, 0.0, 1.0)
    ridge = np.clip(ridge, 0.0, 1.0)
    if not np.any(safe):
        result = np.zeros(safe.shape, dtype=bool)
        result.flags.writeable = False
        return result

    threshold = max(0.08, float(np.quantile(edge[safe], 0.65)))
    kernel = np.ones((3, 3), np.uint8)
    local_high = cv2.dilate(edge, kernel)
    local_low = cv2.erode(edge, kernel)
    local_maximum = (edge >= local_high - 1e-6) & (
        local_high - local_low >= 0.02
    )
    result = safe & ((ridge >= 0.05) | (local_maximum & (edge >= threshold)))
    result.flags.writeable = False
    return result


def instance_boundary_references_from_edge_evidence(
    instance_labels: np.ndarray,
    seed_diameter: float,
    edge_strength: np.ndarray,
    ridge_strength: np.ndarray,
    *,
    interior_buffer_fraction: float = 0.08,
) -> InstanceBoundaryReferences:
    """Derive physical contours and internal non-physical edge candidates."""

    base = instance_boundary_references(
        instance_labels,
        seed_diameter,
        interior_buffer_fraction=interior_buffer_fraction,
    )
    non_edge = internal_edge_candidate_mask(
        edge_strength,
        ridge_strength,
        base.safe_interior,
    )
    return InstanceBoundaryReferences(
        physical_edge=base.physical_edge,
        non_edge=non_edge,
        safe_interior=base.safe_interior,
        interior_buffer_px=base.interior_buffer_px,
    )
