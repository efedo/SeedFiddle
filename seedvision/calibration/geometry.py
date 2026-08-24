"""Geometric reference detection for the controlled laboratory layout."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from seedvision.cuda import (
    CudaContext,
    bgr_to_gray,
    bilinear_sample,
    gaussian_blur,
    gradient_magnitude,
    image_to_tensor,
    resize,
)


class CalibrationDetectionError(RuntimeError):
    """Raised when a required geometric reference cannot be located."""


@dataclass(frozen=True, slots=True)
class DishDetectionSettings:
    """User-adjustable CUDA rim search and weak layout-prior settings."""

    downsample_max_dimension: int = 1600
    hough_accumulator_threshold: int = 32
    min_radius_fraction: float = 0.16
    max_radius_fraction: float = 0.29
    expected_center_x_fraction: float = 0.58
    expected_center_y_fraction: float = 0.40
    expected_radius_fraction: float = 0.225
    expected_outer_diameter_mm: float = 96.0
    calibrated_outer_radius_tolerance_fraction: float = 0.025
    rim_pair_search_fraction: float = 0.14
    rim_pair_min_separation_fraction: float = 0.025
    rim_pair_max_separation_fraction: float = 0.12
    rim_pair_expected_separation_fraction: float = 0.045
    rim_pair_secondary_support_fraction: float = 0.30

    def __post_init__(self) -> None:
        if not 512 <= self.downsample_max_dimension <= 8000:
            raise ValueError("downsample_max_dimension must be between 512 and 8000.")
        if not 5 <= self.hough_accumulator_threshold <= 200:
            raise ValueError(
                "hough_accumulator_threshold must be between 5 and 200."
            )
        if not 0.05 <= self.min_radius_fraction < self.max_radius_fraction <= 0.48:
            raise ValueError("Dish radius fractions must satisfy 0.05 ≤ min < max ≤ 0.48.")
        for name, value in (
            ("expected_center_x_fraction", self.expected_center_x_fraction),
            ("expected_center_y_fraction", self.expected_center_y_fraction),
            ("expected_radius_fraction", self.expected_radius_fraction),
        ):
            if not 0.05 <= value <= 0.95:
                raise ValueError(f"{name} must be between 0.05 and 0.95.")
        if not 0.0 <= self.expected_outer_diameter_mm <= 1000.0:
            raise ValueError(
                "expected_outer_diameter_mm must be between 0 and 1000."
            )
        if not 0.0 <= self.calibrated_outer_radius_tolerance_fraction <= 0.25:
            raise ValueError(
                "calibrated_outer_radius_tolerance_fraction must be between 0 and 0.25."
            )
        if not 0.03 <= self.rim_pair_search_fraction <= 0.30:
            raise ValueError("rim_pair_search_fraction must be between 0.03 and 0.30.")
        if not (
            0.005
            <= self.rim_pair_min_separation_fraction
            < self.rim_pair_max_separation_fraction
            <= 0.30
        ):
            raise ValueError(
                "Petri-rim separation fractions must satisfy 0.005 <= min < max <= 0.30."
            )
        if not (
            self.rim_pair_min_separation_fraction
            <= self.rim_pair_expected_separation_fraction
            <= self.rim_pair_max_separation_fraction
        ):
            raise ValueError(
                "Expected Petri-rim separation must lie between its minimum and maximum."
            )
        if not 0.05 <= self.rim_pair_secondary_support_fraction <= 1.0:
            raise ValueError(
                "rim_pair_secondary_support_fraction must be between 0.05 and 1.0."
            )


@dataclass(frozen=True, slots=True)
class DishCircle:
    """Detected dual-edge Petri dish in original image coordinates.

    ``radius`` is the legacy alias for the upper/outer analysis boundary so
    older downstream consumers do not clip content at the lower glass edge.
    The two named radii preserve the physical glass-edge model needed by the
    layout overlay and future vessel-specific dispatch.
    """

    center_x: int
    center_y: int
    radius: int
    confidence: float
    lower_edge_radius: int | None = None
    upper_edge_radius: int | None = None
    lower_edge_confidence: float = 0.0
    upper_edge_confidence: float = 0.0
    vessel_type: str = "petri_dish"
    rim_pair_detected: bool = False
    outer_radius_x: float | None = None
    outer_radius_y: float | None = None
    inner_radius_x: float | None = None
    inner_radius_y: float | None = None
    ellipse_angle_degrees: float = 0.0
    ellipse_confidence: float = 0.0

    @property
    def inner_radius(self) -> int:
        return int(self.lower_edge_radius or self.radius)

    @property
    def outer_radius(self) -> int:
        return int(self.upper_edge_radius or self.radius)

    @property
    def outer_axes(self) -> tuple[float, float]:
        return (
            float(self.outer_radius_x or self.outer_radius),
            float(self.outer_radius_y or self.outer_radius),
        )

    @property
    def inner_axes(self) -> tuple[float, float]:
        return (
            float(self.inner_radius_x or self.inner_radius),
            float(self.inner_radius_y or self.inner_radius),
        )

    @property
    def bounds(self) -> tuple[int, int, int, int]:
        axis_x, axis_y = self.outer_axes
        angle = np.deg2rad(float(self.ellipse_angle_degrees))
        cosine, sine = float(np.cos(angle)), float(np.sin(angle))
        extent_x = np.sqrt((axis_x * cosine) ** 2 + (axis_y * sine) ** 2)
        extent_y = np.sqrt((axis_x * sine) ** 2 + (axis_y * cosine) ** 2)
        return (
            int(np.floor(self.center_x - extent_x)),
            int(np.floor(self.center_y - extent_y)),
            int(np.ceil(self.center_x + extent_x)),
            int(np.ceil(self.center_y + extent_y)),
        )


def detect_dish(
    image: np.ndarray,
    settings: DishDetectionSettings | None = None,
    *,
    cuda_context: CudaContext | None = None,
    image_tensor=None,
    pixels_per_mm: float | None = None,
) -> DishCircle:
    """Locate the Petri dish using its circular rim and the pilot layout prior.

    The layout prior is intentionally weak: it is used to select among CUDA
    candidates, not to invent a result when no circular rim is present.
    """

    if image is None or image.ndim != 3 or image.shape[2] < 3:
        raise ValueError("A BGR colour image is required.")

    settings = settings or DishDetectionSettings()
    height, width = image.shape[:2]
    scale = min(1.0, settings.downsample_max_dimension / max(height, width))
    import torch

    context = cuda_context or CudaContext.resolve()
    tensor = image_to_tensor(image, context) if image_tensor is None else image_tensor
    small_height = max(16, round(height * scale))
    small_width = max(16, round(width * scale))
    small = resize(tensor, (small_height, small_width), mode="area")
    gray = gaussian_blur(bgr_to_gray(small), 2.0)
    gradient, gradient_x, gradient_y = gradient_magnitude(gray)
    high = torch.quantile(gradient.reshape(-1), 0.995).clamp_min(1.0)
    edge = (gradient / high).clamp(0.0, 1.0)

    expected_x = small_width * settings.expected_center_x_fraction
    expected_y = small_height * settings.expected_center_y_fraction
    expected_radius = small_height * settings.expected_radius_fraction
    minimum_radius = max(4.0, small_height * settings.min_radius_fraction)
    maximum_radius = max(minimum_radius + 2.0, small_height * settings.max_radius_fraction)
    calibrated_primary_radius_range: tuple[float, float] | None = None
    calibrated_outer_radius_range: tuple[float, float] | None = None
    if (
        pixels_per_mm is not None
        and float(pixels_per_mm) > 0.0
        and settings.expected_outer_diameter_mm > 0.0
    ):
        # The photographed layout includes a ruler and a consistent exterior
        # dish diameter. Use that calibrated physical size to prevent the
        # generic circle bank from treating a crowded seed-mass boundary as
        # glass, then require the outer member of the resolved pair to remain
        # within the editable physical tolerance.
        calibrated_expected_outer_radius = (
            float(pixels_per_mm)
            * settings.expected_outer_diameter_mm
            * 0.5
            * scale
        )
        calibrated_outer_minimum = (
            calibrated_expected_outer_radius
            * (1.0 - settings.calibrated_outer_radius_tolerance_fraction)
        )
        calibrated_outer_maximum = (
            calibrated_expected_outer_radius
            * (1.0 + settings.calibrated_outer_radius_tolerance_fraction)
        )
        calibrated_primary_minimum = max(
            minimum_radius,
            calibrated_expected_outer_radius
            * (1.0 - settings.rim_pair_max_separation_fraction),
        )
        calibrated_primary_maximum = min(
            maximum_radius,
            calibrated_outer_maximum,
        )
        if calibrated_primary_maximum <= calibrated_primary_minimum + 2.0:
            raise CalibrationDetectionError(
                "The calibrated dish diameter conflicts with the configured "
                "image-relative radius range."
            )
        calibrated_primary_radius_range = (
            calibrated_primary_minimum,
            calibrated_primary_maximum,
        )
        calibrated_outer_radius_range = (
            calibrated_outer_minimum,
            calibrated_outer_maximum,
        )
    center_step = max(3.0, small_height / 115.0)
    radius_step = max(2.0, (maximum_radius - minimum_radius) / 36.0)
    x_values = torch.arange(
        max(maximum_radius, expected_x - small_width * 0.16),
        min(small_width - maximum_radius, expected_x + small_width * 0.16) + 0.1,
        center_step,
        device=context.device,
    )
    y_values = torch.arange(
        max(maximum_radius, expected_y - small_height * 0.16),
        min(small_height - maximum_radius, expected_y + small_height * 0.16) + 0.1,
        center_step,
        device=context.device,
    )
    radius_values = torch.arange(
        minimum_radius,
        maximum_radius + 0.1,
        radius_step,
        device=context.device,
    )
    if not x_values.numel() or not y_values.numel() or not radius_values.numel():
        raise CalibrationDetectionError("Petri-dish search geometry is empty.")
    grid_x, grid_y, grid_radius = torch.meshgrid(
        x_values, y_values, radius_values, indexing="ij"
    )
    candidates = torch.stack(
        (grid_x.reshape(-1), grid_y.reshape(-1), grid_radius.reshape(-1)),
        dim=1,
    )
    angles = torch.arange(0, 192, device=context.device, dtype=torch.float32)
    angles *= 2.0 * torch.pi / 192.0
    cosine, sine = torch.cos(angles), torch.sin(angles)
    best_score = torch.tensor(-1.0, device=context.device)
    best_candidate = None
    best_support = torch.tensor(0.0, device=context.device)
    for start in range(0, len(candidates), 2048):
        candidate = candidates[start : start + 2048]
        radius = candidate[:, 2:3]
        sample_x = candidate[:, 0:1] + radius * cosine[None]
        sample_y = candidate[:, 1:2] + radius * sine[None]
        sampled_edge = bilinear_sample(edge, sample_x, sample_y)
        sampled_x = bilinear_sample(gradient_x, sample_x, sample_y)
        sampled_y = bilinear_sample(gradient_y, sample_x, sample_y)
        alignment = torch.abs(
            sampled_x * cosine[None] + sampled_y * sine[None]
        ) / torch.sqrt(sampled_x.square() + sampled_y.square() + 1e-6)
        angular_support = sampled_edge * alignment.square()
        sector_support = angular_support.reshape(-1, 12, 16).mean(dim=2)
        support = (
            angular_support.mean(dim=1) * 0.65
            + torch.quantile(sector_support, 0.25, dim=1) * 0.35
        )
        error = (
            torch.abs(candidate[:, 0] - expected_x) / small_width
            + torch.abs(candidate[:, 1] - expected_y) / small_height
            + 1.5 * torch.abs(candidate[:, 2] - expected_radius) / small_height
        )
        score = support - error * 0.22
        if calibrated_primary_radius_range is not None:
            # Ruler scale is valuable when the image evidence is ambiguous, but
            # it is not infallible: a partly occluded ruler or a perspective
            # correction can bias px/mm while the circular glass rim remains
            # exceptionally clear.  Keep the physical diameter as a soft
            # tie-breaker instead of excluding every circle outside a narrow
            # calibrated interval.  The full-circle sector statistic above is
            # what prevents a crowded seed-mass edge from winning.
            calibrated_minimum_radius, calibrated_maximum_radius = (
                calibrated_primary_radius_range
            )
            below = torch.clamp(calibrated_minimum_radius - candidate[:, 2], min=0.0)
            above = torch.clamp(candidate[:, 2] - calibrated_maximum_radius, min=0.0)
            calibrated_deviation = (below + above) / max(small_height, 1)
            score = score - calibrated_deviation * 0.04
        index = torch.argmax(score)
        if score[index] > best_score:
            best_score = score[index]
            best_support = support[index]
            best_candidate = candidate[index]
    minimum_support = 0.035 + settings.hough_accumulator_threshold / 200.0 * 0.16
    if best_candidate is None or float(best_support.item()) < minimum_support:
        raise CalibrationDetectionError("Petri-dish rim not found.")
    selected = best_candidate.detach().cpu().numpy()
    error = float(
        abs(selected[0] - expected_x) / small_width
        + abs(selected[1] - expected_y) / small_height
        + 1.5 * abs(selected[2] - expected_radius) / small_height
    )
    inverse_scale = 1.0 / scale
    (
        lower_radius_small,
        upper_radius_small,
        lower_support,
        upper_support,
        rim_pair_detected,
    ) = (
        _detect_petri_rim_pair(
            edge,
            gradient_x,
            gradient_y,
            float(selected[0]),
            float(selected[1]),
            float(selected[2]),
            minimum_radius,
            maximum_radius,
            settings,
            context,
            outer_radius_range=None,
        )
    )
    # If the visual profile cannot resolve two coherent rims, retry with the
    # calibrated outer interval as a fallback. A successfully detected visual
    # pair is never moved merely to agree with a potentially biased ruler.
    if not rim_pair_detected and calibrated_outer_radius_range is not None:
        calibrated_pair = _detect_petri_rim_pair(
            edge,
            gradient_x,
            gradient_y,
            float(selected[0]),
            float(selected[1]),
            float(selected[2]),
            minimum_radius,
            maximum_radius,
            settings,
            context,
            outer_radius_range=calibrated_outer_radius_range,
        )
        if calibrated_pair[4]:
            (
                lower_radius_small,
                upper_radius_small,
                lower_support,
                upper_support,
                rim_pair_detected,
            ) = calibrated_pair
    ellipse_fit = _fit_mild_dish_ellipse(
        edge,
        float(selected[0]),
        float(selected[1]),
        float(upper_radius_small),
    )
    if ellipse_fit is None:
        fitted_center_x_small = float(selected[0])
        fitted_center_y_small = float(selected[1])
        outer_axis_x_small = float(upper_radius_small)
        outer_axis_y_small = float(upper_radius_small)
        ellipse_angle_degrees = 0.0
        ellipse_confidence = 0.0
    else:
        (
            fitted_center_x_small,
            fitted_center_y_small,
            outer_axis_x_small,
            outer_axis_y_small,
            ellipse_angle_degrees,
            ellipse_confidence,
        ) = ellipse_fit
    lower_ratio = float(lower_radius_small) / max(float(upper_radius_small), 1e-6)
    outer_axis_x = outer_axis_x_small * inverse_scale
    outer_axis_y = outer_axis_y_small * inverse_scale
    inner_axis_x = outer_axis_x * lower_ratio
    inner_axis_y = outer_axis_y * lower_ratio
    lower_radius = round(max(inner_axis_x, inner_axis_y))
    upper_radius = int(np.ceil(max(outer_axis_x, outer_axis_y)))
    minimum_pair_support = minimum_support * 0.70

    def edge_confidence(support: float) -> float:
        relative = support / max(float(best_support.item()), 1e-6)
        absolute = support / max(minimum_pair_support, 1e-6)
        return float(np.clip(relative * 0.55 + (absolute - 0.5) * 0.45, 0.0, 1.0))

    return DishCircle(
        center_x=round(fitted_center_x_small * inverse_scale),
        center_y=round(fitted_center_y_small * inverse_scale),
        radius=upper_radius,
        confidence=max(
            0.0,
            min(
                1.0,
                (float(best_support.item()) / minimum_support - 1.0) * 0.70
                + max(0.0, 1.0 - error / 0.30) * 0.55,
            ),
        ),
        lower_edge_radius=lower_radius,
        upper_edge_radius=upper_radius,
        lower_edge_confidence=edge_confidence(lower_support),
        upper_edge_confidence=edge_confidence(upper_support),
        rim_pair_detected=rim_pair_detected,
        outer_radius_x=float(outer_axis_x),
        outer_radius_y=float(outer_axis_y),
        inner_radius_x=float(inner_axis_x),
        inner_radius_y=float(inner_axis_y),
        ellipse_angle_degrees=float(ellipse_angle_degrees),
        ellipse_confidence=float(ellipse_confidence),
    )


def _fit_mild_dish_ellipse(
    edge,
    center_x: float,
    center_y: float,
    radius: float,
) -> tuple[float, float, float, float, float, float] | None:
    """Fit a conservative, mildly elliptical outer rim to bounded edge pixels."""

    import cv2
    import torch

    height, width = edge.shape[-2:]
    yy, xx = torch.meshgrid(
        torch.arange(height, device=edge.device, dtype=torch.float32),
        torch.arange(width, device=edge.device, dtype=torch.float32),
        indexing="ij",
    )
    radial = torch.sqrt((xx - center_x).square() + (yy - center_y).square())
    half_band = max(3.0, radius * 0.045)
    annulus = (radial >= radius - half_band) & (radial <= radius + half_band)
    values = edge[0, 0][annulus]
    if values.numel() < 80:
        return None
    threshold = torch.quantile(values, 0.62).clamp_min(0.06)
    selected = annulus & (edge[0, 0] >= threshold)
    points_yx = torch.nonzero(selected, as_tuple=False)
    if points_yx.shape[0] < 80:
        return None
    if points_yx.shape[0] > 24000:
        step = max(1, int(points_yx.shape[0]) // 24000)
        points_yx = points_yx[::step]
    points_xy = points_yx[:, (1, 0)].detach().cpu().numpy().astype(np.float32)
    (_fit_center, _fit_size, _fit_angle) = cv2.fitEllipse(points_xy[:, None, :])
    fit_center_x, fit_center_y = (float(value) for value in _fit_center)
    diameter_x, diameter_y = (float(value) for value in _fit_size)
    axis_x, axis_y = diameter_x * 0.5, diameter_y * 0.5
    mean_radius = (axis_x + axis_y) * 0.5
    axis_ratio = max(axis_x, axis_y) / max(1e-6, min(axis_x, axis_y))
    center_error = float(np.hypot(fit_center_x - center_x, fit_center_y - center_y))
    if not 0.90 <= mean_radius / max(radius, 1e-6) <= 1.10:
        return None
    if axis_ratio > 1.15 or center_error > radius * 0.045:
        return None
    # Circular fits do not need an arbitrary unstable orientation.
    angle = float(_fit_angle) if axis_ratio >= 1.006 else 0.0
    confidence = float(
        np.clip(
            1.0
            - center_error / max(radius * 0.045, 1e-6) * 0.35
            - abs(mean_radius - radius) / max(radius * 0.10, 1e-6) * 0.35,
            0.0,
            1.0,
        )
    )
    return (
        fit_center_x,
        fit_center_y,
        axis_x,
        axis_y,
        angle,
        confidence,
    )


def _detect_petri_rim_pair(
    edge,
    gradient_x,
    gradient_y,
    center_x: float,
    center_y: float,
    primary_radius: float,
    minimum_radius: float,
    maximum_radius: float,
    settings: DishDetectionSettings,
    context: CudaContext,
    *,
    outer_radius_range: tuple[float, float] | None = None,
) -> tuple[float, float, float, float, bool]:
    """Select lower/inner and upper/outer glass edges from one radial profile."""

    import torch

    search = settings.rim_pair_search_fraction
    radius_start = max(minimum_radius, primary_radius * (1.0 - search))
    radius_end = min(maximum_radius, primary_radius * (1.0 + search))
    radius_step = max(0.35, primary_radius / 700.0)
    radii = torch.arange(
        radius_start,
        radius_end + radius_step * 0.5,
        radius_step,
        device=context.device,
    )
    angles = torch.arange(0, 256, device=context.device, dtype=torch.float32)
    angles *= 2.0 * torch.pi / 256.0
    cosine, sine = torch.cos(angles), torch.sin(angles)
    sample_x = center_x + radii[:, None] * cosine[None]
    sample_y = center_y + radii[:, None] * sine[None]
    sampled_edge = bilinear_sample(edge, sample_x, sample_y)
    sampled_x = bilinear_sample(gradient_x, sample_x, sample_y)
    sampled_y = bilinear_sample(gradient_y, sample_x, sample_y)
    alignment = torch.abs(
        sampled_x * cosine[None] + sampled_y * sine[None]
    ) / torch.sqrt(sampled_x.square() + sampled_y.square() + 1e-6)
    angular_support = sampled_edge * alignment.square()
    sector_support = angular_support.reshape(-1, 16, 16).mean(dim=2)
    support = (
        angular_support.mean(dim=1) * 0.65
        + torch.quantile(sector_support, 0.25, dim=1) * 0.35
    )
    compact = torch.stack((radii, support), dim=1).detach().cpu().numpy()
    radius_values = compact[:, 0]
    raw_support_values = compact[:, 1]
    # A glass rim creates several narrow reflections. Suppress single-radius
    # noise before selecting the physical inner and exterior boundaries.
    smoothing_kernel = np.asarray((1.0, 2.0, 3.0, 2.0, 1.0), np.float32)
    smoothing_kernel /= smoothing_kernel.sum()
    support_values = np.convolve(
        np.pad(raw_support_values, (2, 2), mode="edge"),
        smoothing_kernel,
        mode="valid",
    )
    selection_support = support_values * 0.65 + raw_support_values * 0.35
    if len(radius_values) < 5:
        primary_support = float(support_values.max())
        return (
            primary_radius,
            primary_radius,
            primary_support,
            primary_support,
            False,
        )

    peak_indices = {
        index
        for index in range(2, len(support_values) - 2)
        if support_values[index]
        >= float(np.max(support_values[index - 2 : index + 3]))
    }
    # Preserve narrow but coherent exterior reflections that the smoothing
    # kernel can merge into a neighbouring glass highlight.
    peak_indices.update(
        index
        for index in range(2, len(raw_support_values) - 2)
        if raw_support_values[index]
        >= float(np.max(raw_support_values[index - 2 : index + 3]))
    )
    peak_indices = sorted(peak_indices)
    if not peak_indices:
        peak_indices = [int(np.argmax(selection_support))]
    maximum_support = max(float(np.max(selection_support)), 1e-6)
    peak_indices = [
        index
        for index in peak_indices
        if float(selection_support[index])
        >= maximum_support * settings.rim_pair_secondary_support_fraction
    ]

    def valid_outer(index: int) -> bool:
        if outer_radius_range is None:
            return True
        return (
            outer_radius_range[0]
            <= float(radius_values[index])
            <= outer_radius_range[1]
        )

    anchor_tolerance = primary_radius * 0.04
    anchor_candidates = [
        index
        for index in peak_indices
        if abs(float(radius_values[index]) - primary_radius) <= anchor_tolerance
    ]
    if anchor_candidates:
        anchor = max(
            anchor_candidates,
            key=lambda index: (
                float(selection_support[index]) / maximum_support
                - abs(float(radius_values[index]) - primary_radius)
                / max(anchor_tolerance, 1e-6)
                * 0.12
            ),
        )
        outer_candidates = []
        for index in peak_indices:
            if not valid_outer(index):
                continue
            separation_fraction = float(
                radius_values[index] - radius_values[anchor]
            ) / max(primary_radius, 1e-6)
            if (
                settings.rim_pair_min_separation_fraction
                <= separation_fraction
                <= settings.rim_pair_max_separation_fraction
            ):
                separation_error = abs(
                    separation_fraction
                    - settings.rim_pair_expected_separation_fraction
                ) / max(
                    settings.rim_pair_max_separation_fraction
                    - settings.rim_pair_min_separation_fraction,
                    1e-6,
                )
                score = (
                    float(selection_support[index]) / maximum_support
                    - separation_error * 0.18
                    - max(
                        0.0,
                        settings.rim_pair_expected_separation_fraction
                        - separation_fraction,
                    )
                    / max(
                        settings.rim_pair_expected_separation_fraction
                        - settings.rim_pair_min_separation_fraction,
                        1e-6,
                    )
                    * 0.28
                )
                outer_candidates.append((score, index))
        if outer_candidates:
            _, outer = max(outer_candidates)
            return (
                float(radius_values[anchor]),
                float(radius_values[outer]),
                float(selection_support[anchor]),
                float(selection_support[outer]),
                True,
            )
    best_pair: tuple[float, int, int] | None = None
    for first_position, first in enumerate(peak_indices):
        for second in peak_indices[first_position + 1 :]:
            if not valid_outer(second):
                continue
            separation = float(radius_values[second] - radius_values[first])
            separation_fraction = separation / max(primary_radius, 1e-6)
            if not (
                settings.rim_pair_min_separation_fraction
                <= separation_fraction
                <= settings.rim_pair_max_separation_fraction
            ):
                continue
            strength = (
                float(selection_support[first]) + float(selection_support[second])
            ) / (2.0 * maximum_support)
            separation_error = abs(
                separation_fraction - settings.rim_pair_expected_separation_fraction
            ) / max(
                settings.rim_pair_max_separation_fraction
                - settings.rim_pair_min_separation_fraction,
                1e-6,
            )
            score = (
                strength
                - separation_error * 0.18
                - max(
                    0.0,
                    settings.rim_pair_expected_separation_fraction
                    - separation_fraction,
                )
                / max(
                    settings.rim_pair_expected_separation_fraction
                    - settings.rim_pair_min_separation_fraction,
                    1e-6,
                )
                * 0.28
            )
            if best_pair is None or score > best_pair[0]:
                best_pair = (score, first, second)
    if best_pair is None:
        primary_index = int(np.argmax(selection_support))
        separated = [
            index
            for index in peak_indices
            if index > primary_index
            and valid_outer(index)
            and settings.rim_pair_min_separation_fraction
            <= (float(radius_values[index] - radius_values[primary_index]))
            / max(primary_radius, 1e-6)
            <= settings.rim_pair_max_separation_fraction
        ]
        if separated:
            secondary = max(separated, key=lambda index: selection_support[index])
            first, second = sorted((primary_index, secondary))
        else:
            inferred_gap = (
                primary_radius * settings.rim_pair_expected_separation_fraction
            )
            second_radius = min(radius_end, primary_radius + inferred_gap * 0.5)
            if outer_radius_range is not None:
                second_radius = float(
                    np.clip(
                        second_radius,
                        outer_radius_range[0],
                        outer_radius_range[1],
                    )
                )
            first_radius = max(radius_start, second_radius - inferred_gap)
            primary_support = float(selection_support[primary_index])
            return (
                first_radius,
                second_radius,
                primary_support * 0.5,
                primary_support * 0.5,
                False,
            )
    else:
        _, first, second = best_pair
    return (
        float(radius_values[first]),
        float(radius_values[second]),
        float(selection_support[first]),
        float(selection_support[second]),
        True,
    )
