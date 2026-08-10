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
    rim_pair_search_fraction: float = 0.14
    rim_pair_min_separation_fraction: float = 0.015
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

    @property
    def inner_radius(self) -> int:
        return int(self.lower_edge_radius or self.radius)

    @property
    def outer_radius(self) -> int:
        return int(self.upper_edge_radius or self.radius)

    @property
    def bounds(self) -> tuple[int, int, int, int]:
        radius = self.outer_radius
        return (
            self.center_x - radius,
            self.center_y - radius,
            self.center_x + radius,
            self.center_y + radius,
        )


def detect_dish(
    image: np.ndarray,
    settings: DishDetectionSettings | None = None,
    *,
    cuda_context: CudaContext | None = None,
    image_tensor=None,
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
        )
    )
    lower_radius = round(lower_radius_small * inverse_scale)
    upper_radius = round(upper_radius_small * inverse_scale)
    minimum_pair_support = minimum_support * 0.70

    def edge_confidence(support: float) -> float:
        relative = support / max(float(best_support.item()), 1e-6)
        absolute = support / max(minimum_pair_support, 1e-6)
        return float(np.clip(relative * 0.55 + (absolute - 0.5) * 0.45, 0.0, 1.0))

    return DishCircle(
        center_x=round(float(selected[0]) * inverse_scale),
        center_y=round(float(selected[1]) * inverse_scale),
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
    support_values = compact[:, 1]
    if len(radius_values) < 5:
        primary_support = float(support_values.max())
        return (
            primary_radius,
            primary_radius,
            primary_support,
            primary_support,
            False,
        )

    peak_indices = [
        index
        for index in range(2, len(support_values) - 2)
        if support_values[index]
        >= float(np.max(support_values[index - 2 : index + 3]))
    ]
    if not peak_indices:
        peak_indices = [int(np.argmax(support_values))]
    maximum_support = max(float(np.max(support_values)), 1e-6)
    peak_indices = [
        index
        for index in peak_indices
        if float(support_values[index])
        >= maximum_support * settings.rim_pair_secondary_support_fraction
    ]
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
                float(support_values[index]) / maximum_support
                - abs(float(radius_values[index]) - primary_radius)
                / max(anchor_tolerance, 1e-6)
                * 0.12
            ),
        )
        outer_candidates = []
        for index in peak_indices:
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
                    float(support_values[index]) / maximum_support
                    - separation_error * 0.18
                )
                outer_candidates.append((score, index))
        if outer_candidates:
            _, outer = max(outer_candidates)
            return (
                float(radius_values[anchor]),
                float(radius_values[outer]),
                float(support_values[anchor]),
                float(support_values[outer]),
                True,
            )
    best_pair: tuple[float, int, int] | None = None
    for first_position, first in enumerate(peak_indices):
        for second in peak_indices[first_position + 1 :]:
            separation = float(radius_values[second] - radius_values[first])
            separation_fraction = separation / max(primary_radius, 1e-6)
            if not (
                settings.rim_pair_min_separation_fraction
                <= separation_fraction
                <= settings.rim_pair_max_separation_fraction
            ):
                continue
            strength = (
                float(support_values[first]) + float(support_values[second])
            ) / (2.0 * maximum_support)
            separation_error = abs(
                separation_fraction - settings.rim_pair_expected_separation_fraction
            ) / max(
                settings.rim_pair_max_separation_fraction
                - settings.rim_pair_min_separation_fraction,
                1e-6,
            )
            score = strength - separation_error * 0.18
            if best_pair is None or score > best_pair[0]:
                best_pair = (score, first, second)
    if best_pair is None:
        primary_index = int(np.argmax(support_values))
        separated = [
            index
            for index in peak_indices
            if settings.rim_pair_min_separation_fraction
            <= abs(float(radius_values[index] - radius_values[primary_index]))
            / max(primary_radius, 1e-6)
            <= settings.rim_pair_max_separation_fraction
        ]
        if separated:
            secondary = max(separated, key=lambda index: support_values[index])
            first, second = sorted((primary_index, secondary))
        else:
            inferred_gap = (
                primary_radius * settings.rim_pair_expected_separation_fraction
            )
            first_radius = max(radius_start, primary_radius - inferred_gap * 0.5)
            second_radius = min(radius_end, primary_radius + inferred_gap * 0.5)
            primary_support = float(support_values[primary_index])
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
        float(support_values[first]),
        float(support_values[second]),
        True,
    )
