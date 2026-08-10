"""Public diagnostic-layer types backed by PyTorch CUDA kernels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class NoiseFrequencyProfile:
    """Per-image robust texture statistics learned from colour pseudo-labels."""

    band_scales_px: tuple[float, float, float]
    background_log_rms: tuple[float, float, float]
    nonbackground_log_rms: tuple[float, float, float]
    background_sample_count: int
    nonbackground_sample_count: int
    separation: float
    background_log_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    nonbackground_log_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)


@dataclass(frozen=True, slots=True)
class BackgroundColourProfile:
    """Robust colour centre and observed likely-background range."""

    centre_lab: tuple[float, float, float]
    scale_lab: tuple[float, float, float]
    bgr_low: tuple[int, int, int]
    bgr_high: tuple[int, int, int]
    sample_count: int
    sample_fraction: float
    component_centres_lab: tuple[tuple[float, float, float], ...] = ()
    component_scales_lab: tuple[tuple[float, float, float], ...] = ()
    component_weights: tuple[float, ...] = ()
    refinement_iterations: int = 0


@dataclass(frozen=True, slots=True)
class ForegroundColourProfile:
    """Robust colour modes represented by the current foreground model."""

    centre_lab: tuple[float, float, float]
    scale_lab: tuple[float, float, float]
    bgr_low: tuple[int, int, int]
    bgr_high: tuple[int, int, int]
    sample_count: int
    sample_fraction: float
    component_centres_lab: tuple[tuple[float, float, float], ...] = ()
    component_scales_lab: tuple[tuple[float, float, float], ...] = ()
    component_weights: tuple[float, ...] = ()
    refinement_iterations: int = 0


@dataclass(frozen=True, slots=True)
class AnalysisLayerSettings:
    """User-adjustable settings for diagnostic and provisional mask layers."""

    background_sample_radius_fraction: float = 0.10
    background_chroma_percentile: float = 50.0
    background_lightness_percentile: float = 55.0
    background_minimum_sample_fraction: float = 0.002
    background_prior_tolerance: float = 24.0
    background_lightness_scale_floor: float = 8.0
    background_chroma_scale_floor: float = 3.0
    background_colour_components: int = 4
    background_distribution_fit_iterations: int = 6
    background_refinement_iterations: int = 2
    background_refinement_min_probability: float = 0.82
    background_frequency_weight_power: float = 0.35
    background_distribution_scale_multiplier: float = 1.25
    noise_medium_scale_fraction: float = 0.03
    noise_coarse_scale_fraction: float = 0.08
    noise_direction_step_degrees: int = 15
    noise_vector_length_fraction: float = 0.55
    noise_vector_sample_count: int = 9
    noise_vector_decay: float = 0.86
    noise_direction_integration: str = "maximum"
    noise_background_min_likelihood: int = 190
    noise_nonbackground_max_likelihood: int = 65
    noise_working_maximum_dimension: int = 1280
    edge_blur_sigma: float = 1.2
    edge_chroma_weight: float = 1.5
    edge_normalization_percentile: float = 99.0
    edge_strength_gamma: float = 0.65
    instance_min_extent_fraction: float = 0.72
    instance_max_extent_fraction: float = 0.95
    instance_radius_extent_multiplier: float = 1.55
    curve_radius_low_fraction: float = 0.38
    curve_radius_nominal_fraction: float = 0.50
    curve_radius_high_fraction: float = 0.64
    curve_arc_angle_degrees: float = 28.0
    curve_orientation_tolerance_degrees: float = 11.0
    curve_near_distance_fraction: float = 0.035
    curve_middle_distance_fraction: float = 0.12
    curve_deep_distance_fraction: float = 0.24
    curve_min_darkening: float = 1.0
    curve_full_darkening: float = 20.0
    curve_diameter_multiplier: float = 1.0
    curve_lightness_boost: float = 0.60
    ridge_nms_step_px: float = 1.0
    ridge_low_threshold: float = 0.10
    ridge_high_threshold: float = 0.24
    ridge_hysteresis_iterations: int = 8
    trace_tangent_tolerance_degrees: float = 24.0
    trace_maximum_gap_px: int = 2
    trace_window_fraction: float = 0.20
    trace_sample_count: int = 7
    trace_minimum_length_fraction: float = 0.18
    trace_junction_max_neighbors: int = 4
    boundary_radius_min_fraction: float = 0.25
    boundary_radius_max_fraction: float = 0.80
    boundary_radius_sample_count: int = 9
    boundary_arc_span_degrees: float = 70.0
    boundary_arc_sample_count: int = 15
    boundary_orientation_tolerance_degrees: float = 20.0
    boundary_missing_support_floor: float = 0.15
    boundary_circle_residual_tolerance: float = 0.16
    boundary_ellipse_residual_tolerance: float = 0.18
    boundary_max_axis_ratio: float = 2.20
    boundary_radius_log_tolerance: float = 0.45
    boundary_center_vote_weight: float = 0.35
    boundary_center_vote_blur_fraction: float = 0.08
    boundary_semantic_weight: float = 0.25
    boundary_polarity_boost: float = 0.20
    boundary_minimum_confidence: float = 0.12
    boundary_geometry_max_candidates: int = 256
    boundary_working_maximum_dimension: int = 1280

    def __post_init__(self) -> None:
        percentages = (
            self.background_chroma_percentile,
            self.background_lightness_percentile,
            self.edge_normalization_percentile,
        )
        if any(not 1.0 <= value <= 99.9 for value in percentages):
            raise ValueError("Layer percentiles must be between 1 and 99.9.")
        if not (
            0.05
            <= self.curve_radius_low_fraction
            < self.curve_radius_nominal_fraction
            < self.curve_radius_high_fraction
            <= 1.20
        ):
            raise ValueError("Curve radius fractions must be strictly increasing.")
        if not (
            0.0
            <= self.curve_near_distance_fraction
            < self.curve_middle_distance_fraction
            < self.curve_deep_distance_fraction
            <= 0.80
        ):
            raise ValueError("Curve profile distances must be strictly increasing.")
        if not 5.0 <= self.curve_arc_angle_degrees <= 80.0:
            raise ValueError("curve_arc_angle_degrees must be between 5 and 80.")
        if not 1.0 <= self.curve_orientation_tolerance_degrees <= 45.0:
            raise ValueError("curve_orientation_tolerance_degrees must be between 1 and 45.")
        if self.curve_full_darkening <= 0.0:
            raise ValueError("curve_full_darkening must be positive.")
        if not 0.25 <= self.curve_diameter_multiplier <= 3.0:
            raise ValueError("curve_diameter_multiplier must be between 0.25 and 3.")
        if not 0.0 <= self.curve_lightness_boost <= 3.0:
            raise ValueError("curve_lightness_boost must be between 0 and 3.")
        if self.noise_medium_scale_fraction >= self.noise_coarse_scale_fraction:
            raise ValueError("Medium noise scale must be below coarse noise scale.")
        if self.noise_nonbackground_max_likelihood >= self.noise_background_min_likelihood:
            raise ValueError("Non-background likelihood maximum must be below background minimum.")
        if self.instance_min_extent_fraction >= self.instance_max_extent_fraction:
            raise ValueError("Minimum instance extent must be below maximum extent.")
        if not 0.0001 <= self.background_minimum_sample_fraction <= 0.25:
            raise ValueError("Background minimum sample fraction is out of range.")
        if not 2.0 <= self.background_prior_tolerance <= 100.0:
            raise ValueError("Background prior tolerance is out of range.")
        if not 5 <= self.noise_direction_step_degrees <= 90:
            raise ValueError("Noise direction step must be between 5 and 90 degrees.")
        if not 0.05 <= self.noise_vector_length_fraction <= 2.0:
            raise ValueError("Noise vector length fraction is out of range.")
        if not 2 <= self.noise_vector_sample_count <= 32:
            raise ValueError("Noise vector sample count must be between 2 and 32.")
        if not 512 <= self.noise_working_maximum_dimension <= 4096:
            raise ValueError("Noise working dimension must be between 512 and 4096.")
        if not 0.10 <= self.noise_vector_decay <= 1.0:
            raise ValueError("Noise vector decay must be between 0.10 and 1.")
        if self.background_lightness_scale_floor <= 0.0:
            raise ValueError("Background lightness scale floor must be positive.")
        if self.background_chroma_scale_floor <= 0.0:
            raise ValueError("Background chroma scale floor must be positive.")
        if not 1 <= self.background_colour_components <= 8:
            raise ValueError("Background colour components must be between 1 and 8.")
        if not 1 <= self.background_distribution_fit_iterations <= 20:
            raise ValueError("Background distribution fit iterations must be between 1 and 20.")
        if not 0 <= self.background_refinement_iterations <= 8:
            raise ValueError("Background refinement iterations must be between 0 and 8.")
        if not 0.50 <= self.background_refinement_min_probability <= 0.99:
            raise ValueError("Background refinement probability must be between 0.50 and 0.99.")
        if not 0.0 <= self.background_frequency_weight_power <= 1.0:
            raise ValueError("Background frequency influence must be between 0 and 1.")
        if not 0.50 <= self.background_distribution_scale_multiplier <= 3.0:
            raise ValueError("Background colour tolerance multiplier must be between 0.5 and 3.")
        if self.noise_direction_integration not in {"mean", "maximum", "minimum", "median"}:
            raise ValueError("Unknown directional background integration method.")
        if not 0.25 <= self.ridge_nms_step_px <= 3.0:
            raise ValueError("Ridge NMS step must be between 0.25 and 3 pixels.")
        if not 0.0 <= self.ridge_low_threshold < self.ridge_high_threshold <= 1.0:
            raise ValueError("Ridge thresholds must satisfy 0 <= low < high <= 1.")
        if not 1 <= self.ridge_hysteresis_iterations <= 32:
            raise ValueError("Ridge hysteresis iterations must be between 1 and 32.")
        if not 2 <= self.trace_sample_count <= 32:
            raise ValueError("Trace sample count must be between 2 and 32.")
        if not 1 <= self.trace_maximum_gap_px <= 5:
            raise ValueError("Trace maximum gap must be between 1 and 5 pixels.")
        if not 1 <= self.trace_junction_max_neighbors <= 8:
            raise ValueError("Trace junction neighbour limit must be between 1 and 8.")
        if not 0.05 <= self.boundary_radius_min_fraction < self.boundary_radius_max_fraction <= 1.5:
            raise ValueError("Boundary radius fractions are invalid.")
        if not 3 <= self.boundary_radius_sample_count <= 25:
            raise ValueError("Boundary radius sample count must be between 3 and 25.")
        if not 512 <= self.boundary_working_maximum_dimension <= 4096:
            raise ValueError("Boundary working dimension must be between 512 and 4096.")
        if not 5 <= self.boundary_arc_sample_count <= 41:
            raise ValueError("Boundary arc sample count must be between 5 and 41.")
        if not 10.0 <= self.boundary_arc_span_degrees <= 160.0:
            raise ValueError("Boundary arc span must be between 10 and 160 degrees.")
        if not 1.0 <= self.boundary_max_axis_ratio <= 4.0:
            raise ValueError("Boundary maximum axis ratio must be between 1 and 4.")
        for name, value in (
            ("boundary_center_vote_weight", self.boundary_center_vote_weight),
            ("boundary_semantic_weight", self.boundary_semantic_weight),
            ("boundary_polarity_boost", self.boundary_polarity_boost),
            ("boundary_minimum_confidence", self.boundary_minimum_confidence),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1.")


@dataclass(frozen=True, slots=True)
class AnalysisLayers:
    """GPU-resident raster layers aligned to the vessel-centred crop.

    Selecting an overlay materializes only the raster(s) required by that Qt
    view. The authoritative tensors remain resident on their original device.
    """

    offset_x: int
    offset_y: int
    instance_labels: object
    instance_colours: np.ndarray
    background_likelihood: object
    refined_background_likelihood: object
    noise_frequency_profile: NoiseFrequencyProfile
    edge_likelihood: object
    directed_edge_hue: object
    undirected_edge_hue: object
    seed_edge_curve_likelihood: object
    seed_edge_curve_radius_px: object
    valid_mask: object
    undirected_edge_likelihood: object | None = None
    background_mode: str = "automatic"
    background_reference_count: int = 0
    background_colour_profile: BackgroundColourProfile | None = None
    foreground_colour_profile: ForegroundColourProfile | None = None
    directional_background_likelihoods: tuple[np.ndarray, ...] = ()
    directional_background_angles_degrees: tuple[float, ...] = ()
    edge_ridges: object | None = None
    edge_trace_labels: object | None = None
    edge_trace_continuity: object | None = None
    edge_trace_gap_confidence: object | None = None
    edge_radius_ratio_hue: object | None = None
    edge_radius_confidence: object | None = None
    edge_circle_confidence: object | None = None
    edge_ellipse_confidence: object | None = None
    edge_fit_residual: object | None = None
    edge_centre_votes: object | None = None
    edge_semantic_sides: object | None = None
    edge_rejection_hue: object | None = None
    edge_rejection_strength: object | None = None
    edge_fit_geometry: object | None = None
    surrounding_noise_likelihood: object | None = None
    surrounding_noise_valid_mask: object | None = None
    surrounding_noise_offset_x: int = 0
    surrounding_noise_offset_y: int = 0
    gpu_source: object | None = None
    gpu_valid: object | None = None

    def instance_rgba(self) -> np.ndarray:
        labels = np.asarray(self.instance_labels)
        rgb = self.instance_colours[labels]
        alpha = np.uint8(labels > 0) * 255
        return np.dstack((rgb, alpha))

    def background_rgba(self) -> np.ndarray:
        gray = 255 - np.asarray(self.background_likelihood)
        alpha = np.uint8(np.asarray(self.valid_mask) > 0) * 255
        return np.dstack((gray, gray, gray, alpha))

    def refined_background_rgba(self) -> np.ndarray:
        gray = 255 - np.asarray(self.refined_background_likelihood)
        alpha = np.uint8(np.asarray(self.valid_mask) > 0) * 255
        return np.dstack((gray, gray, gray, alpha))

    def directional_background_rgba(self, index: int) -> np.ndarray:
        gray = 255 - np.asarray(self.directional_background_likelihoods[index])
        alpha = np.uint8(np.asarray(self.valid_mask) > 0) * 255
        return np.dstack((gray, gray, gray, alpha))

    def surrounding_noise_rgba(self) -> np.ndarray | None:
        if (
            self.surrounding_noise_likelihood is None
            or self.surrounding_noise_valid_mask is None
        ):
            return None
        gray = 255 - np.asarray(self.surrounding_noise_likelihood)
        alpha = np.uint8(np.asarray(self.surrounding_noise_valid_mask) > 0) * 255
        return np.dstack((gray, gray, gray, alpha))

    def directed_edge_rgba(self) -> np.ndarray:
        return self._edge_rgba(self.directed_edge_hue)

    def edge_magnitude_rgba(self) -> np.ndarray:
        likelihood = np.asarray(self.edge_likelihood)
        alpha = np.uint8(np.asarray(self.valid_mask) > 0) * 255
        return np.dstack(
            (likelihood, likelihood, likelihood, alpha)
        )

    def undirected_edge_rgba(self) -> np.ndarray:
        return self._edge_rgba(self.undirected_edge_hue, self.undirected_edge_likelihood)

    def seed_edge_curve_rgba(self) -> np.ndarray:
        return self._edge_rgba(self.directed_edge_hue, self.seed_edge_curve_likelihood)

    def edge_ridges_rgba(self) -> np.ndarray:
        return self._heat_rgba(self.edge_ridges)

    def edge_traces_rgba(self) -> np.ndarray:
        labels = np.asarray(self.edge_trace_labels, dtype=np.int64)
        hue = np.uint8(np.mod(labels * 67, 179))
        strength = np.uint8(labels > 0) * 255
        return self._edge_rgba(hue, strength)

    def edge_radius_confirmation_rgba(self) -> np.ndarray:
        return self._edge_rgba(
            self.edge_radius_ratio_hue, self.edge_radius_confidence
        )

    def edge_rejection_rgba(self) -> np.ndarray:
        return self._edge_rgba(
            self.edge_rejection_hue, self.edge_rejection_strength
        )

    def edge_rgba(self) -> np.ndarray:
        return self.directed_edge_rgba()

    def _edge_rgba(self, hue: np.ndarray, likelihood: np.ndarray | None = None) -> np.ndarray:
        likelihood = self.edge_likelihood if likelihood is None else likelihood
        rgb = _hsv_full_saturation_to_rgb(
            np.asarray(hue), np.asarray(likelihood)
        )
        alpha = np.uint8(np.asarray(self.valid_mask) > 0) * 255
        return np.dstack((rgb, alpha))

    def _heat_rgba(self, raster) -> np.ndarray:
        values = np.asarray(raster, dtype=np.uint8)
        normalized = values.astype(np.float32) / 255.0
        red = np.uint8(np.clip(normalized * 2.0, 0.0, 1.0) * 255)
        green = np.uint8(
            np.clip(1.0 - np.abs(normalized * 2.0 - 1.0), 0.0, 1.0)
            * 255
        )
        blue = np.uint8(np.clip((1.0 - normalized) * 1.7, 0.0, 1.0) * 255)
        alpha = np.uint8(np.asarray(self.valid_mask) > 0) * 255
        return np.dstack((red, green, blue, alpha))


def _hsv_full_saturation_to_rgb(hue: np.ndarray, value: np.ndarray) -> np.ndarray:
    """Render OpenCV-scale HSV results for Qt without an analysis dependency."""

    hue_sector = hue.astype(np.float32) / 30.0
    sector = np.floor(hue_sector).astype(np.int16) % 6
    fraction = hue_sector - np.floor(hue_sector)
    maximum = value.astype(np.float32)
    rising = maximum * fraction
    falling = maximum * (1.0 - fraction)
    zero = np.zeros_like(maximum)
    red = np.choose(sector, (maximum, falling, zero, zero, rising, maximum))
    green = np.choose(sector, (rising, maximum, maximum, falling, zero, zero))
    blue = np.choose(sector, (zero, zero, rising, maximum, maximum, falling))
    return np.uint8(np.clip(np.stack((red, green, blue), axis=-1), 0, 255))


def build_analysis_layers(
    crop: np.ndarray,
    valid_mask: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray,
    seed_diameter: float,
    *,
    offset_x: int,
    offset_y: int,
    background_reference_points: tuple[tuple[float, float], ...] = (),
    foreground_reference_points: tuple[tuple[float, float], ...] = (),
    background_reference_mask: np.ndarray | None = None,
    foreground_reference_mask: np.ndarray | None = None,
    background_reference_samples: np.ndarray | None = None,
    background_reference_sample_count: int = 0,
    background_prior_lab: tuple[float, float, float] | None = None,
    background_colour_enabled: bool = True,
    foreground_probability=None,
    foreground_colour_profile: ForegroundColourProfile | None = None,
    surrounding_noise_source_tensor=None,
    surrounding_noise_valid_tensor=None,
    surrounding_noise_offset_x: int = 0,
    surrounding_noise_offset_y: int = 0,
    settings: AnalysisLayerSettings | None = None,
    cache_values: dict[str, object] | None = None,
    dirty_nodes: set[str] | frozenset[str] = frozenset(),
    cuda_context=None,
    timing_recorder=None,
) -> AnalysisLayers:
    from seedvision.cuda.layers import build_cuda_analysis_layers

    return build_cuda_analysis_layers(
        crop,
        valid_mask,
        centers,
        radii,
        seed_diameter,
        offset_x=offset_x,
        offset_y=offset_y,
        background_reference_points=background_reference_points,
        foreground_reference_points=foreground_reference_points,
        background_reference_mask=background_reference_mask,
        foreground_reference_mask=foreground_reference_mask,
        background_reference_samples=background_reference_samples,
        background_reference_sample_count=background_reference_sample_count,
        background_prior_lab=background_prior_lab,
        background_colour_enabled=background_colour_enabled,
        foreground_probability=foreground_probability,
        foreground_colour_profile=foreground_colour_profile,
        surrounding_noise_source_tensor=surrounding_noise_source_tensor,
        surrounding_noise_valid_tensor=surrounding_noise_valid_tensor,
        surrounding_noise_offset_x=surrounding_noise_offset_x,
        surrounding_noise_offset_y=surrounding_noise_offset_y,
        settings=settings,
        cache_values=cache_values,
        dirty_nodes=dirty_nodes,
        cuda_context=cuda_context,
        timing_recorder=timing_recorder,
    )


# Kept as importable helpers for focused diagnostics and existing callers.
from seedvision.cuda.layers import (  # noqa: E402
    directional_edges as _directional_edges,
    seed_edge_curve_likelihood as _seed_edge_curve_likelihood,
)
