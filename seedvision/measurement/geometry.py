"""Geometric measurements calculated from reviewed seed masks."""

from __future__ import annotations

from dataclasses import dataclass
import math

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


@dataclass(frozen=True, slots=True)
class RobustBodyEllipse:
    center_xy: tuple[float, float]
    body_length: float
    body_width: float
    orientation_degrees: float
    orientation_uncertainty_degrees: float
    support_fraction: float


@dataclass(frozen=True, slots=True)
class LocalContourFeature:
    angle_degrees: float
    arc_fraction: float
    signed_height_fraction: float
    support_confidence: float
    hilum_associated: bool


@dataclass(frozen=True, slots=True)
class SeedShapeMeasurement:
    """Auditable observed 2-D silhouette geometry for one reviewed mask."""

    maximum_span: float
    maximum_span_endpoints: tuple[tuple[float, float], tuple[float, float]]
    ellipse: RobustBodyEllipse
    projected_area: float
    ellipse_relative_area: float
    non_ellipticity: float
    asymmetry: float
    solidity: float
    concavity_fraction: float
    protrusion_fraction: float
    neck_width_fraction: float
    thin_extension_fraction: float
    local_feature: LocalContourFeature | None
    contour_signature: tuple[float, ...]
    measurement_uncertainty: tuple[float, ...]
    length_unit: str
    area_unit: str

    @property
    def ovality(self) -> float:
        return self.ellipse.body_length / max(self.ellipse.body_width, 1e-9)

    def vector(self) -> tuple[float, ...]:
        feature = self.local_feature
        feature_angle = (
            0.0 if feature is None else np.deg2rad(feature.angle_degrees * 2.0)
        )
        return (
            self.maximum_span,
            self.ellipse.body_length,
            self.ellipse.body_width,
            self.ovality,
            self.projected_area,
            self.ellipse_relative_area,
            self.non_ellipticity,
            self.asymmetry,
            self.solidity,
            self.concavity_fraction,
            self.protrusion_fraction,
            self.neck_width_fraction,
            self.thin_extension_fraction,
            0.0 if feature is None else feature.signed_height_fraction,
            0.0 if feature is None else feature.arc_fraction,
            0.0 if feature is None else float(np.cos(feature_angle)),
            0.0 if feature is None else float(np.sin(feature_angle)),
            0.0 if feature is None else feature.support_confidence,
            0.0 if feature is None else float(feature.hilum_associated),
        )


def measure_shape_mask(
    mask: np.ndarray,
    *,
    pixels_per_mm: float | None = None,
    calibration_relative_uncertainty: float = 0.0,
    hilum_point: tuple[float, float] | None = None,
    contour_samples: int = 128,
    perturbation_radii: tuple[int, ...] = (1, 2),
) -> SeedShapeMeasurement:
    """Measure a reviewed silhouette with robust ellipse and boundary sensitivity.

    Only compact contour points leave the source raster. Morphological opening
    and closing supply spatially coherent boundary alternatives; independent
    per-pixel jitter is deliberately not used.
    """

    import cv2

    values = np.asarray(mask, dtype=np.uint8)
    if values.ndim != 2 or not np.any(values):
        raise ValueError("A non-empty two-dimensional seed mask is required.")
    if pixels_per_mm is not None and pixels_per_mm <= 0:
        raise ValueError("pixels_per_mm must be positive.")
    if not 0.0 <= calibration_relative_uncertainty <= 0.5:
        raise ValueError("Calibration relative uncertainty must be between 0 and 0.5.")
    if contour_samples < 32 or contour_samples > 1024:
        raise ValueError("contour_samples must be between 32 and 1024.")
    # Qt annotations are CPU rasters, but contour fitting/sensitivity needs only
    # this seed's compact bounding crop, not repeated full-image morphology.
    rows = np.flatnonzero(values.any(axis=1))
    columns = np.flatnonzero(values.any(axis=0))
    padding = max((int(r) for r in perturbation_radii), default=0) + 2
    offset_x, offset_y = int(columns[0])-padding, int(rows[0])-padding
    values = np.pad(values[rows[0]:rows[-1]+1, columns[0]:columns[-1]+1], padding)
    if hilum_point is not None:
        hilum_point = (hilum_point[0]-offset_x, hilum_point[1]-offset_y)
    contour = _largest_contour(values)
    measurement_px = _shape_from_contour(
        values, contour, hilum_point=hilum_point, contour_samples=contour_samples
    )
    alternatives = []
    for radius in perturbation_radii:
        radius = int(radius)
        if radius <= 0:
            continue
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1)
        )
        for operation in (cv2.MORPH_OPEN, cv2.MORPH_CLOSE, cv2.MORPH_ERODE, cv2.MORPH_DILATE):
            changed = cv2.morphologyEx(values, operation, kernel)
            if not np.any(changed):
                continue
            try:
                alternatives.append(
                    _shape_from_contour(
                        changed,
                        _largest_contour(changed),
                        hilum_point=hilum_point,
                        contour_samples=contour_samples,
                    ).vector()
                )
            except ValueError:
                continue
    baseline = np.asarray(measurement_px.vector(), np.float64)
    boundary_uncertainty = (
        np.zeros_like(baseline)
        if not alternatives
        else np.sqrt(
            np.mean((np.asarray(alternatives, np.float64) - baseline) ** 2, axis=0)
        )
    )
    calibration_uncertainty = np.zeros_like(baseline)
    # Lengths and areas share image-level ruler uncertainty. Dimensionless
    # shape quantities do not acquire fabricated scale error.
    calibration_uncertainty[[0, 1, 2]] = (
        baseline[[0, 1, 2]] * calibration_relative_uncertainty
    )
    calibration_uncertainty[4] = (
        baseline[4] * 2.0 * calibration_relative_uncertainty
    )
    uncertainty = np.sqrt(
        np.square(boundary_uncertainty) + np.square(calibration_uncertainty)
    )
    scale = 1.0 if pixels_per_mm is None else float(pixels_per_mm)
    area_scale = scale * scale
    ellipse = measurement_px.ellipse
    return SeedShapeMeasurement(
        maximum_span=measurement_px.maximum_span / scale,
        maximum_span_endpoints=tuple((x+offset_x, y+offset_y)
                                     for x, y in measurement_px.maximum_span_endpoints),
        ellipse=RobustBodyEllipse(
            (ellipse.center_xy[0]+offset_x, ellipse.center_xy[1]+offset_y),
            ellipse.body_length / scale,
            ellipse.body_width / scale,
            ellipse.orientation_degrees,
            ellipse.orientation_uncertainty_degrees,
            ellipse.support_fraction,
        ),
        projected_area=measurement_px.projected_area / area_scale,
        ellipse_relative_area=measurement_px.ellipse_relative_area,
        non_ellipticity=measurement_px.non_ellipticity,
        asymmetry=measurement_px.asymmetry,
        solidity=measurement_px.solidity,
        concavity_fraction=measurement_px.concavity_fraction,
        protrusion_fraction=measurement_px.protrusion_fraction,
        neck_width_fraction=measurement_px.neck_width_fraction,
        thin_extension_fraction=measurement_px.thin_extension_fraction,
        local_feature=measurement_px.local_feature,
        contour_signature=measurement_px.contour_signature,
        measurement_uncertainty=tuple(
            float(value / scale if index in {0, 1, 2} else value / area_scale if index == 4 else value)
            for index, value in enumerate(uncertainty)
        ),
        length_unit="mm" if pixels_per_mm is not None else "px",
        area_unit="mm²" if pixels_per_mm is not None else "px²",
    )


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


def _largest_contour(mask: np.ndarray) -> np.ndarray:
    import cv2

    contours, _hierarchy = cv2.findContours(
        np.asarray(mask, np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        raise ValueError("The seed mask contains no external contour.")
    contour = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
    if len(contour) < 5:
        raise ValueError("At least five contour points are required for shape fitting.")
    return contour


def _robust_ellipse(points: np.ndarray):
    import cv2

    current = points.astype(np.float32).reshape(-1, 1, 2)
    support = np.ones(len(points), dtype=bool)
    ellipse = cv2.fitEllipse(current)
    for _ in range(3):
        center, axes, angle = ellipse
        major = max(float(axes[0]), float(axes[1]))
        minor = min(float(axes[0]), float(axes[1]))
        orientation = float(angle + (90.0 if axes[1] > axes[0] else 0.0)) % 180.0
        radians = np.deg2rad(orientation)
        cosine, sine = np.cos(radians), np.sin(radians)
        delta = points - np.asarray(center, np.float64)
        local_x = delta[:, 0] * cosine + delta[:, 1] * sine
        local_y = -delta[:, 0] * sine + delta[:, 1] * cosine
        radius = np.sqrt(
            (local_x / max(major * 0.5, 1e-6)) ** 2
            + (local_y / max(minor * 0.5, 1e-6)) ** 2
        )
        residual = np.abs(radius - 1.0)
        cutoff = max(0.025, float(np.quantile(residual, 0.85)))
        support = residual <= cutoff
        if int(support.sum()) < 5:
            support[:] = True
            break
        ellipse = cv2.fitEllipse(
            points[support].astype(np.float32).reshape(-1, 1, 2)
        )
    center, axes, angle = ellipse
    major = max(float(axes[0]), float(axes[1]))
    minor = min(float(axes[0]), float(axes[1]))
    orientation = float(angle + (90.0 if axes[1] > axes[0] else 0.0)) % 180.0
    return (
        (float(center[0]), float(center[1])),
        major,
        minor,
        orientation,
        float(np.mean(support)),
    )


def _shape_from_contour(
    mask: np.ndarray,
    contour: np.ndarray,
    *,
    hilum_point: tuple[float, float] | None,
    contour_samples: int,
) -> SeedShapeMeasurement:
    import cv2

    center, major, minor, orientation, support = _robust_ellipse(contour)
    if major <= 0 or minor <= 0:
        raise ValueError("The robust body ellipse is degenerate.")
    hull = cv2.convexHull(contour.astype(np.float32)).reshape(-1, 2).astype(np.float64)
    maximum_span = _maximum_pairwise_distance(hull)
    first, second = _maximum_pairwise_endpoints(hull)
    area = float(np.count_nonzero(mask))
    hull_mask = np.zeros_like(mask, dtype=np.uint8)
    cv2.fillConvexPoly(hull_mask, np.rint(hull).astype(np.int32), 1)
    hull_area = max(area, float(np.count_nonzero(hull_mask)))
    ellipse_area = np.pi * major * minor * 0.25
    solidity = area / max(hull_area, 1e-9)
    concavity = max(0.0, hull_area - area) / max(area, 1e-9)

    radians = np.deg2rad(orientation)
    cosine, sine = np.cos(radians), np.sin(radians)
    delta = contour - np.asarray(center, np.float64)
    local_x = delta[:, 0] * cosine + delta[:, 1] * sine
    local_y = -delta[:, 0] * sine + delta[:, 1] * cosine
    angles = np.mod(np.arctan2(local_y, local_x), 2.0 * np.pi)
    observed_radius = np.hypot(local_x, local_y)
    expected_radius = 1.0 / np.sqrt(
        (np.cos(angles) / (major * 0.5)) ** 2
        + (np.sin(angles) / (minor * 0.5)) ** 2
    )
    residual = observed_radius / np.maximum(expected_radius, 1e-6) - 1.0
    non_ellipticity = float(np.sqrt(np.mean(np.square(residual))))
    signature = _angular_signature(angles, residual, contour_samples)
    half = contour_samples // 2
    asymmetry = float(
        np.mean(np.abs(signature[:half] - signature[half : half * 2]))
    )
    positive = np.maximum(signature, 0.0)
    protrusion_fraction = float(np.mean(positive))
    feature = _local_feature(
        signature, center, orientation, hilum_point=hilum_point
    )

    ellipse_mask = np.zeros_like(mask, dtype=np.uint8)
    cv2.ellipse(
        ellipse_mask,
        (round(center[0]), round(center[1])),
        (max(1, round(major * 0.5)), max(1, round(minor * 0.5))),
        orientation,
        0,
        360,
        1,
        thickness=-1,
    )
    extension = (mask > 0) & (ellipse_mask == 0)
    if np.any(extension):
        distance = cv2.distanceTransform(np.asarray(mask > 0, np.uint8), cv2.DIST_L2, 5)
        extension_widths = 2.0 * distance[extension]
        neck_width_fraction = float(
            np.quantile(extension_widths, 0.10) / max(minor, 1e-6)
        )
        thin_extension_fraction = float(
            np.mean(extension_widths < 0.18 * minor)
            * np.count_nonzero(extension)
            / max(area, 1.0)
        )
    else:
        neck_width_fraction = 1.0
        thin_extension_fraction = 0.0
    orientation_uncertainty = float(
        min(90.0, 4.0 + 70.0 * np.exp(-8.0 * (major / minor - 1.0)))
    )
    return SeedShapeMeasurement(
        maximum_span=float(maximum_span),
        maximum_span_endpoints=(first, second),
        ellipse=RobustBodyEllipse(
            center,
            float(major),
            float(minor),
            float(orientation),
            orientation_uncertainty,
            support,
        ),
        projected_area=area,
        ellipse_relative_area=area / max(ellipse_area, 1e-9),
        non_ellipticity=non_ellipticity,
        asymmetry=asymmetry,
        solidity=solidity,
        concavity_fraction=concavity,
        protrusion_fraction=protrusion_fraction,
        neck_width_fraction=neck_width_fraction,
        thin_extension_fraction=thin_extension_fraction,
        local_feature=feature,
        contour_signature=tuple(float(value) for value in signature),
        measurement_uncertainty=(0.0,) * 19,
        length_unit="px",
        area_unit="px²",
    )


def _maximum_pairwise_endpoints(
    points: np.ndarray,
) -> tuple[tuple[float, float], tuple[float, float]]:
    if len(points) == 1:
        point = tuple(float(value) for value in points[0])
        return point, point
    best = (0, 0)
    maximum = -1.0
    for start in range(0, len(points), 256):
        differences = points[start : start + 256, None, :] - points[None, :, :]
        squared = np.einsum("ijk,ijk->ij", differences, differences)
        local = int(np.argmax(squared))
        first, second = np.unravel_index(local, squared.shape)
        if float(squared[first, second]) > maximum:
            maximum = float(squared[first, second])
            best = (start + int(first), int(second))
    return (
        tuple(float(value) for value in points[best[0]]),
        tuple(float(value) for value in points[best[1]]),
    )


def _angular_signature(
    angles: np.ndarray, residual: np.ndarray, sample_count: int
) -> np.ndarray:
    positions = np.floor(angles / (2.0 * np.pi) * sample_count).astype(int)
    positions = np.clip(positions, 0, sample_count - 1)
    signature = np.full(sample_count, np.nan, np.float64)
    for index in range(sample_count):
        values = residual[positions == index]
        if len(values):
            signature[index] = float(np.median(values))
    available = np.flatnonzero(np.isfinite(signature))
    if not len(available):
        return np.zeros(sample_count, np.float64)
    extended_x = np.concatenate((available - sample_count, available, available + sample_count))
    extended_y = np.tile(signature[available], 3)
    return np.interp(np.arange(sample_count), extended_x, extended_y)


def _local_feature(
    signature: np.ndarray,
    center: tuple[float, float],
    orientation: float,
    *,
    hilum_point: tuple[float, float] | None,
) -> LocalContourFeature | None:
    absolute = np.abs(signature)
    peak = int(np.argmax(absolute))
    height = float(signature[peak])
    robust = float(np.median(absolute) + 1.4826 * np.median(np.abs(absolute - np.median(absolute))))
    if abs(height) < max(0.025, robust * 1.5):
        return None
    threshold = max(0.015, abs(height) * 0.45)
    support = 1
    for direction in (-1, 1):
        for step in range(1, len(signature) // 4):
            if abs(signature[(peak + direction * step) % len(signature)]) < threshold:
                break
            support += 1
    angle = (peak + 0.5) / len(signature) * 360.0
    associated = False
    if hilum_point is not None:
        hilum_angle = (
            math.degrees(
                math.atan2(hilum_point[1] - center[1], hilum_point[0] - center[0])
            )
            - orientation
        ) % 360.0
        separation = abs((angle - hilum_angle + 180.0) % 360.0 - 180.0)
        associated = separation <= max(15.0, support / len(signature) * 360.0)
    return LocalContourFeature(
        angle_degrees=float(angle),
        arc_fraction=float(min(1.0, support / len(signature))),
        signed_height_fraction=height,
        support_confidence=float(min(1.0, abs(height) / max(robust * 3.0, 1e-6))),
        hilum_associated=associated,
    )
