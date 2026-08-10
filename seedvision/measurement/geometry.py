"""Geometric measurements calculated from reviewed seed masks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from seedvision.cuda import (
    CudaContext,
    binary_erode,
    connected_components,
    image_to_tensor,
)


@dataclass(frozen=True, slots=True)
class ShapeMeasurements:
    """Visible-face shape measurements in pixels or calibrated millimetres."""

    maximum_feret: float
    minimum_feret: float
    area: float
    perimeter: float
    equivalent_diameter: float
    aspect_ratio: float
    circularity: float
    roundness: float
    solidity: float
    convexity: float
    length_unit: str
    area_unit: str


def measure_mask(
    mask: np.ndarray,
    *,
    pixels_per_mm: float | None = None,
    cuda_context: CudaContext | None = None,
) -> ShapeMeasurements:
    """Measure the largest external object in a binary mask.

    Physical dimensions are returned only when ``pixels_per_mm`` is supplied.
    The caller remains responsible for withholding measurements from occluded
    or otherwise invalid masks.
    """

    if mask is None or mask.ndim != 2:
        raise ValueError("A two-dimensional binary mask is required.")
    if pixels_per_mm is not None and pixels_per_mm <= 0:
        raise ValueError("pixels_per_mm must be positive.")

    import torch

    context = cuda_context or CudaContext.resolve()
    binary = image_to_tensor(np.uint8(mask > 0), context) > 0
    labels, statistics = connected_components(binary)
    if int(labels.max().item()) == 0:
        raise ValueError("The mask contains no foreground object.")
    selected = int(np.argmax(statistics["area"][1:]) + 1)
    component = labels == selected
    area_px = float(component.sum().item())
    values = component[0, 0]
    horizontal = torch.count_nonzero(values[:, 1:] != values[:, :-1])
    vertical = torch.count_nonzero(values[1:, :] != values[:-1, :])
    diagonal_a = torch.count_nonzero(values[1:, 1:] != values[:-1, :-1])
    diagonal_b = torch.count_nonzero(values[1:, :-1] != values[:-1, 1:])
    # Crofton's four-direction estimate is substantially less grid-biased than
    # summing horizontal and vertical pixel-face lengths.
    perimeter_px = float(
        (
            np.pi
            / 8.0
            * (
                horizontal.float()
                + vertical.float()
                + (diagonal_a.float() + diagonal_b.float()) / np.sqrt(2.0)
            )
        ).item()
    )
    if area_px <= 0 or perimeter_px <= 0:
        raise ValueError("The foreground object has no measurable area.")

    boundary = component & ~binary_erode(component, 3)
    boundary_yx = torch.nonzero(boundary[0, 0], as_tuple=False)
    points = boundary_yx[:, (1, 0)].detach().cpu().numpy().astype(np.float64)
    hull = _convex_hull(points)
    hull_area = max(area_px, _polygon_area(hull))
    hull_perimeter = _polygon_perimeter(hull)
    maximum_feret_px = _maximum_pairwise_distance(hull)
    minimum_feret_px = _minimum_caliper_width(hull)
    equivalent_diameter_px = float(np.sqrt(4.0 * area_px / np.pi))

    scale = pixels_per_mm or 1.0
    length_unit = "mm" if pixels_per_mm is not None else "px"
    area_unit = "mm²" if pixels_per_mm is not None else "px²"
    return ShapeMeasurements(
        maximum_feret=maximum_feret_px / scale,
        minimum_feret=minimum_feret_px / scale,
        area=area_px / (scale * scale),
        perimeter=perimeter_px / scale,
        equivalent_diameter=equivalent_diameter_px / scale,
        aspect_ratio=maximum_feret_px / max(minimum_feret_px, 1e-9),
        circularity=4.0 * np.pi * area_px / (perimeter_px * perimeter_px),
        roundness=4.0 * area_px / (np.pi * maximum_feret_px * maximum_feret_px),
        solidity=area_px / max(hull_area, 1e-9),
        convexity=hull_perimeter / perimeter_px,
        length_unit=length_unit,
        area_unit=area_unit,
    )


def _convex_hull(points: np.ndarray) -> np.ndarray:
    """Andrew monotone-chain hull for the compact downloaded boundary."""

    unique = sorted({(float(x), float(y)) for x, y in points})
    if len(unique) <= 1:
        return np.asarray(unique, np.float64).reshape(-1, 2)

    def cross(origin, first, second):
        return (
            (first[0] - origin[0]) * (second[1] - origin[1])
            - (first[1] - origin[1]) * (second[0] - origin[0])
        )

    lower: list[tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1], np.float64)


def _polygon_area(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0
    x, y = points[:, 0], points[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) * 0.5)


def _polygon_perimeter(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1).sum())


def _maximum_pairwise_distance(points: np.ndarray) -> float:
    if len(points) == 1:
        return 0.0
    maximum_squared = 0.0
    for start in range(0, len(points), 256):
        chunk = points[start : start + 256]
        differences = chunk[:, None, :] - points[None, :, :]
        maximum_squared = max(
            maximum_squared,
            float(np.max(np.einsum("ijk,ijk->ij", differences, differences))),
        )
    return float(np.sqrt(maximum_squared))


def _minimum_caliper_width(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    edges = np.roll(points, -1, axis=0) - points
    lengths = np.linalg.norm(edges, axis=1)
    valid_edges = edges[lengths > 1e-9]
    valid_lengths = lengths[lengths > 1e-9]
    if not len(valid_edges):
        return 0.0
    normals = np.column_stack((-valid_edges[:, 1], valid_edges[:, 0]))
    normals /= valid_lengths[:, None]
    minimum = float("inf")
    for normal in normals:
        projections = points @ normal
        minimum = min(minimum, float(np.max(projections) - np.min(projections)))
    return minimum
