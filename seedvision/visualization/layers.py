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
    excluded_component_centres_lab: tuple[tuple[float, float, float], ...] = ()
    excluded_component_scales_lab: tuple[tuple[float, float, float], ...] = ()
    excluded_component_weights: tuple[float, ...] = ()
    exclusion_strength: float = 0.95


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
    excluded_component_centres_lab: tuple[tuple[float, float, float], ...] = ()
    excluded_component_scales_lab: tuple[tuple[float, float, float], ...] = ()
    excluded_component_weights: tuple[float, ...] = ()
    exclusion_strength: float = 0.95
    source: str = "unknown"
    source_sample_count: int = 0


@dataclass(frozen=True, slots=True)
class ReferenceTexturePrototype:
    """One feature medoid plus a display-only source-context thumbnail."""

    class_name: str
    patch_bgr: np.ndarray
    weight: float
    sample_count: int
    centre_xy: tuple[float, float]
    tangent_degrees: float | None = None


@dataclass(frozen=True, slots=True)
class ReferenceTextureProfile:
    """Compact display metadata for the GPU-resident prototype banks."""

    prototypes: tuple[ReferenceTexturePrototype, ...] = ()
    class_sample_counts: tuple[tuple[str, int], ...] = ()
    class_sample_count_units: tuple[tuple[str, str], ...] = ()
    material_feature_names: tuple[str, ...] = ()
    edge_feature_names: tuple[str, ...] = ()
    working_scale: float = 1.0
    edge_working_scale: float = 1.0
    edge_working_seed_diameter_px: float = 0.0
    edge_strip_normal_offset_px: float = 0.0
    edge_strip_tangent_half_length_px: float = 0.0
    patch_size_px: int = 0

    def count_for(self, class_name: str) -> int:
        return sum(
            prototype.class_name == class_name
            for prototype in self.prototypes
        )

    def sample_count_unit_for(self, class_name: str) -> str:
        return dict(self.class_sample_count_units).get(
            class_name, "reference pixels"
        )


@dataclass(frozen=True, slots=True)
class AnalysisLayerSettings:
    """User-adjustable settings for diagnostic and provisional mask layers."""

    perimeter_background_buffer_cm: float = 0.35
    perimeter_background_band_thickness_cm: float = 0.50
    background_sample_radius_fraction: float = 0.10
    background_chroma_percentile: float = 50.0
    background_lightness_percentile: float = 55.0
    background_minimum_sample_fraction: float = 0.002
    background_prior_tolerance: float = 24.0
    background_lightness_scale_floor: float = 8.0
    background_chroma_scale_floor: float = 3.0
    background_colour_components: int = 32
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
    foreground_noise_medium_scale_fraction: float = 0.03
    foreground_noise_coarse_scale_fraction: float = 0.08
    foreground_noise_direction_step_degrees: int = 15
    foreground_noise_vector_length_fraction: float = 0.55
    foreground_noise_vector_sample_count: int = 9
    foreground_noise_vector_decay: float = 0.86
    foreground_noise_direction_integration: str = "1st tertile"
    foreground_noise_foreground_min_likelihood: int = 190
    foreground_noise_nonforeground_max_likelihood: int = 65
    foreground_noise_working_maximum_dimension: int = 1280
    edge_blur_sigma: float = 1.2
    edge_chroma_weight: float = 1.5
    edge_normalization_percentile: float = 99.0
    edge_strength_gamma: float = 0.65
    surface_gradient_blur_sigma: float = 1.6
    surface_gradient_radius_fraction: float = 0.45
    surface_gradient_direction_step_degrees: int = 15
    surface_gradient_sample_count: int = 8
    surface_gradient_normalization_percentile: float = 99.0
    surface_gradient_strength_gamma: float = 0.60
    surface_gradient_working_maximum_dimension: int = 1280
    lightening_gradient_maximum_slope: float = 1.0
    darkening_gradient_maximum_slope: float = 1.0
    frequency_noise_fine_scale_fraction: float = 0.008
    frequency_noise_medium_scale_fraction: float = 0.030
    frequency_noise_coarse_scale_fraction: float = 0.100
    frequency_noise_context_fraction: float = 0.025
    frequency_noise_normalization_percentile: float = 99.0
    frequency_noise_strength_gamma: float = 0.65
    frequency_noise_working_maximum_dimension: int = 1280
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
    reference_edge_ridge_weight: float = 0.35
    reference_ridge_nms_step_px: float = 1.0
    reference_ridge_low_threshold: float = 0.10
    reference_ridge_high_threshold: float = 0.24
    reference_ridge_hysteresis_iterations: int = 8
    reference_ridge_working_maximum_dimension: int = 1280
    reference_texture_material_prototypes_per_class: int = 64
    reference_texture_edge_prototypes_per_class: int = 256
    reference_texture_minimum_samples_per_prototype: int = 16
    reference_texture_fit_iterations: int = 4
    reference_texture_similarity_scale: float = 1.0
    reference_texture_context_fraction: float = 0.04
    reference_texture_patch_fraction: float = 0.28
    reference_texture_working_maximum_dimension: int = 960
    reference_texture_edge_working_maximum_dimension: int = 2048
    reference_edge_minimum_working_seed_diameter_px: float = 28.0
    reference_edge_strip_normal_offset_fraction: float = 0.05
    reference_edge_strip_tangent_half_length_fraction: float = 0.08
    # Annotated instances always provide boundary supervision when present;
    # this controls how far inside each contour internal-edge examples begin.
    reference_texture_instance_interior_buffer_fraction: float = 0.08
    # Display-only evidence margin: the internal/non-physical classifier is
    # intentionally adjustable because true physical contours can receive
    # support from both image-local prototype banks.
    net_physical_edge_internal_scale: float = 0.50
    trace_tangent_tolerance_degrees: float = 24.0
    trace_maximum_gap_px: int = 2
    trace_curvature_policy: str = "prefer"
    trace_curvature_tolerance_degrees: float = 6.0
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
    boundary_instance_edge_influence: float = 0.35
    boundary_nonphysical_edge_discount: float = 0.80
    boundary_polarity_boost: float = 0.20
    boundary_minimum_confidence: float = 0.12
    boundary_geometry_max_candidates: int = 256
    boundary_working_maximum_dimension: int = 1280

    def __post_init__(self) -> None:
        if not 0.0 <= self.perimeter_background_buffer_cm <= 2.0:
            raise ValueError("Perimeter background buffer must be between 0 and 2 cm.")
        if not 0.05 <= self.perimeter_background_band_thickness_cm <= 2.0:
            raise ValueError(
                "Perimeter background band thickness must be between 0.05 and 2 cm."
            )
        percentages = (
            self.background_chroma_percentile,
            self.background_lightness_percentile,
            self.edge_normalization_percentile,
            self.surface_gradient_normalization_percentile,
            self.frequency_noise_normalization_percentile,
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
        if self.foreground_noise_medium_scale_fraction >= self.foreground_noise_coarse_scale_fraction:
            raise ValueError("Medium foreground-noise scale must be below its coarse scale.")
        if self.foreground_noise_nonforeground_max_likelihood >= self.foreground_noise_foreground_min_likelihood:
            raise ValueError("Non-foreground likelihood maximum must be below foreground minimum.")
        if self.instance_min_extent_fraction >= self.instance_max_extent_fraction:
            raise ValueError("Minimum instance extent must be below maximum extent.")
        if not 0.0001 <= self.background_minimum_sample_fraction <= 0.25:
            raise ValueError("Background minimum sample fraction is out of range.")
        if not 2.0 <= self.background_prior_tolerance <= 100.0:
            raise ValueError("Background prior tolerance is out of range.")
        if not 5 <= self.noise_direction_step_degrees <= 90:
            raise ValueError("Noise direction step must be between 5 and 90 degrees.")
        if not 5 <= self.foreground_noise_direction_step_degrees <= 90:
            raise ValueError("Foreground-noise direction step must be between 5 and 90 degrees.")
        if not 0.05 <= self.noise_vector_length_fraction <= 2.0:
            raise ValueError("Noise vector length fraction is out of range.")
        if not 0.05 <= self.foreground_noise_vector_length_fraction <= 2.0:
            raise ValueError("Foreground-noise vector length fraction is out of range.")
        if not 2 <= self.noise_vector_sample_count <= 32:
            raise ValueError("Noise vector sample count must be between 2 and 32.")
        if not 2 <= self.foreground_noise_vector_sample_count <= 32:
            raise ValueError("Foreground-noise vector sample count must be between 2 and 32.")
        if not 512 <= self.noise_working_maximum_dimension <= 4096:
            raise ValueError("Noise working dimension must be between 512 and 4096.")
        if not 512 <= self.foreground_noise_working_maximum_dimension <= 4096:
            raise ValueError("Foreground-noise working dimension must be between 512 and 4096.")
        if not 8 <= self.reference_texture_material_prototypes_per_class <= 256:
            raise ValueError(
                "Material reference-texture prototypes per class must be between 8 and 256."
            )
        if not 8 <= self.reference_texture_edge_prototypes_per_class <= 1024:
            raise ValueError(
                "Edge reference-texture prototypes per class must be between 8 and 1024."
            )
        if not 4 <= self.reference_texture_minimum_samples_per_prototype <= 512:
            raise ValueError("Reference texture prototype support must be between 4 and 512 pixels.")
        if not 1 <= self.reference_texture_fit_iterations <= 12:
            raise ValueError("Reference texture fit iterations must be between 1 and 12.")
        if not 0.25 <= self.reference_texture_similarity_scale <= 4.0:
            raise ValueError("Reference texture tolerance must be between 0.25 and 4.")
        if not 0.005 <= self.reference_texture_context_fraction <= 0.30:
            raise ValueError("Reference texture context must be between 0.005 and 0.30 seed diameters.")
        if not 0.10 <= self.reference_texture_patch_fraction <= 0.80:
            raise ValueError("Reference texture patch size must be between 0.10 and 0.80 seed diameters.")
        if not 256 <= self.reference_texture_working_maximum_dimension <= 2048:
            raise ValueError("Reference texture working dimension must be between 256 and 2048.")
        if not 512 <= self.reference_texture_edge_working_maximum_dimension <= 4096:
            raise ValueError(
                "Reference edge-strip working dimension must be between 512 and 4096."
            )
        if not 8.0 <= self.reference_edge_minimum_working_seed_diameter_px <= 96.0:
            raise ValueError(
                "Minimum edge-strip seed diameter must be between 8 and 96 pixels."
            )
        if not 0.01 <= self.reference_edge_strip_normal_offset_fraction <= 0.20:
            raise ValueError(
                "Edge-strip normal offset must be between 0.01 and 0.2 diameter."
            )
        if not 0.01 <= self.reference_edge_strip_tangent_half_length_fraction <= 0.25:
            raise ValueError(
                "Edge-strip tangent half-length must be between 0.01 and 0.25 diameter."
            )
        if not (
            0.0
            <= self.reference_texture_instance_interior_buffer_fraction
            <= 0.50
        ):
            raise ValueError(
                "Annotated-instance interior buffer must be between 0 and 0.5 diameter."
            )
        if not 0.0 <= self.net_physical_edge_internal_scale <= 2.0:
            raise ValueError(
                "Net physical-edge internal subtraction must be between 0 and 2."
            )
        if not 0.10 <= self.noise_vector_decay <= 1.0:
            raise ValueError("Noise vector decay must be between 0.10 and 1.")
        if not 0.10 <= self.foreground_noise_vector_decay <= 1.0:
            raise ValueError("Foreground-noise vector decay must be between 0.10 and 1.")
        if self.background_lightness_scale_floor <= 0.0:
            raise ValueError("Background lightness scale floor must be positive.")
        if self.background_chroma_scale_floor <= 0.0:
            raise ValueError("Background chroma scale floor must be positive.")
        if not 1 <= self.background_colour_components <= 256:
            raise ValueError(
                "Background colour modes must be between 1 and 256."
            )
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
        direction_integrations = {"mean", "maximum", "minimum", "median"}
        if self.noise_direction_integration not in direction_integrations:
            raise ValueError("Unknown directional background integration method.")
        if self.foreground_noise_direction_integration not in {
            *direction_integrations, "1st tertile"
        }:
            raise ValueError("Unknown directional foreground integration method.")
        if not 0.05 <= self.surface_gradient_radius_fraction <= 2.0:
            raise ValueError("Surface-gradient ray length fraction is out of range.")
        if not 5 <= self.surface_gradient_direction_step_degrees <= 90:
            raise ValueError("Surface-gradient direction step must be between 5 and 90 degrees.")
        if not 2 <= self.surface_gradient_sample_count <= 32:
            raise ValueError("Surface-gradient sample count must be between 2 and 32.")
        if not 512 <= self.surface_gradient_working_maximum_dimension <= 4096:
            raise ValueError("Surface-gradient working dimension must be between 512 and 4096.")
        if self.surface_gradient_blur_sigma <= 0.0:
            raise ValueError("Surface-gradient blur sigma must be positive.")
        if self.surface_gradient_strength_gamma <= 0.0:
            raise ValueError("Surface-gradient display gamma must be positive.")
        if self.lightening_gradient_maximum_slope <= 0.0:
            raise ValueError("Lightening-gradient slope ceiling must be positive.")
        if self.darkening_gradient_maximum_slope <= 0.0:
            raise ValueError("Darkening-gradient slope ceiling must be positive.")
        if not (
            0.0
            < self.frequency_noise_fine_scale_fraction
            < self.frequency_noise_medium_scale_fraction
            < self.frequency_noise_coarse_scale_fraction
        ):
            raise ValueError("Frequency-noise scale fractions must be strictly increasing.")
        if self.frequency_noise_context_fraction <= 0.0:
            raise ValueError("Frequency-noise RMS context must be positive.")
        if self.frequency_noise_strength_gamma <= 0.0:
            raise ValueError("Frequency-noise display gamma must be positive.")
        if not 512 <= self.frequency_noise_working_maximum_dimension <= 4096:
            raise ValueError("Frequency-noise working dimension must be between 512 and 4096.")
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
        if self.trace_curvature_policy not in {"off", "prefer", "require"}:
            raise ValueError(
                "Trace curvature policy must be 'off', 'prefer', or 'require'."
            )
        if not 0.0 <= self.trace_curvature_tolerance_degrees <= 45.0:
            raise ValueError(
                "Trace curvature tolerance must be between 0 and 45 degrees."
            )
        if not 1 <= self.trace_junction_max_neighbors <= 8:
            raise ValueError("Trace junction neighbour limit must be between 1 and 8.")
        if not 0.0 <= self.reference_edge_ridge_weight <= 1.0:
            raise ValueError("Reference-edge ridge support must be between 0 and 1.")
        if not 0.25 <= self.reference_ridge_nms_step_px <= 3.0:
            raise ValueError(
                "Reference-ridge NMS step must be between 0.25 and 3 pixels."
            )
        if not (
            0.0
            <= self.reference_ridge_low_threshold
            < self.reference_ridge_high_threshold
            <= 1.0
        ):
            raise ValueError(
                "Reference-ridge thresholds must satisfy 0 <= low < high <= 1."
            )
        if not 1 <= self.reference_ridge_hysteresis_iterations <= 32:
            raise ValueError(
                "Reference-ridge hysteresis iterations must be between 1 and 32."
            )
        if not 256 <= self.reference_ridge_working_maximum_dimension <= 4096:
            raise ValueError(
                "Reference-ridge working dimension must be between 256 and 4096."
            )
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
            ("boundary_instance_edge_influence", self.boundary_instance_edge_influence),
            ("boundary_nonphysical_edge_discount", self.boundary_nonphysical_edge_discount),
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
    lightening_surface_gradient: object | None = None
    lightening_surface_direction: object | None = None
    darkening_surface_gradient: object | None = None
    darkening_surface_direction: object | None = None
    weak_lightening_surface_gradient: object | None = None
    weak_lightening_surface_direction: object | None = None
    weak_darkening_surface_gradient: object | None = None
    weak_darkening_surface_direction: object | None = None
    frequency_noise_band_scales_px: tuple[float, float, float] = (0.0, 0.0, 0.0)
    darkness_frequency_noise_masks: tuple[object, ...] = ()
    colour_frequency_noise_masks: tuple[object, ...] = ()
    foreground_noise_likelihood: object | None = None
    foreground_noise_frequency_profile: NoiseFrequencyProfile | None = None
    other_colour_probability: object | None = None
    other_noise_probability: object | None = None
    other_noise_frequency_profile: NoiseFrequencyProfile | None = None
    reference_seed_surface_probability: object | None = None
    reference_background_texture_probability: object | None = None
    reference_other_texture_probability: object | None = None
    reference_texture_profile: ReferenceTextureProfile | None = None
    physical_edge_probability: object | None = None
    non_edge_probability: object | None = None
    net_physical_edge_internal_scale: float = 0.50
    reference_edge_ridges: object | None = None
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
    surrounding_background_likelihood: object | None = None
    surrounding_background_valid_mask: object | None = None
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

    def foreground_noise_rgba(self) -> np.ndarray:
        gray = (
            np.zeros_like(np.asarray(self.valid_mask), dtype=np.uint8)
            if self.foreground_noise_likelihood is None
            else np.asarray(self.foreground_noise_likelihood)
        )
        alpha = np.uint8(np.asarray(self.valid_mask) > 0) * 255
        return np.dstack((gray, gray, gray, alpha))

    def other_colour_rgba(self) -> np.ndarray:
        """Show raw colour membership in the explicitly painted Other model."""

        gray = (
            np.zeros_like(np.asarray(self.valid_mask), dtype=np.uint8)
            if self.other_colour_probability is None
            else np.asarray(self.other_colour_probability, dtype=np.uint8)
        )
        alpha = np.uint8(np.asarray(self.valid_mask) > 0) * 255
        return np.dstack((gray, gray, gray, alpha))

    def other_noise_rgba(self) -> np.ndarray:
        """Show the directional Other-vs-non-Other noise/colour score."""

        gray = (
            np.zeros_like(np.asarray(self.valid_mask), dtype=np.uint8)
            if self.other_noise_probability is None
            else np.asarray(self.other_noise_probability, dtype=np.uint8)
        )
        alpha = np.uint8(np.asarray(self.valid_mask) > 0) * 255
        return np.dstack((gray, gray, gray, alpha))

    def reference_seed_surface_rgba(self) -> np.ndarray:
        return self.reference_material_probability_rgba("foreground")

    def reference_material_probability_rgba(
        self, class_name: str
    ) -> np.ndarray:
        rasters = {
            "foreground": self.reference_seed_surface_probability,
            "background": self.reference_background_texture_probability,
            "other": self.reference_other_texture_probability,
        }
        if class_name not in rasters:
            raise ValueError(f"Unknown reference material class {class_name!r}.")
        raster = rasters[class_name]
        values = (
            np.zeros_like(np.asarray(self.valid_mask), dtype=np.uint8)
            if raster is None
            else np.asarray(raster, dtype=np.uint8)
        )
        alpha = np.uint8(np.asarray(self.valid_mask) > 0) * 255
        return np.dstack((values, values, values, alpha))

    def reference_edge_probability_rgba(self, physical: bool = True) -> np.ndarray:
        raster = (
            self.physical_edge_probability if physical else self.non_edge_probability
        )
        values = (
            np.zeros_like(np.asarray(self.valid_mask), dtype=np.uint8)
            if raster is None
            else np.asarray(raster, dtype=np.uint8)
        )
        normalized = values.astype(np.float32) / 255.0
        if physical:
            red = np.uint8(normalized * 255.0)
            green = np.uint8(normalized * 210.0)
            blue = np.uint8(normalized * 45.0)
        else:
            red = np.uint8(normalized * 75.0)
            green = np.uint8(normalized * 150.0)
            blue = np.uint8(normalized * 255.0)
        alpha = np.uint8(np.asarray(self.valid_mask) > 0) * 255
        return np.dstack((red, green, blue, alpha))

    def reference_edge_comparison_rgba(self) -> np.ndarray:
        """Compare both learned edge classes without materializing a new raster.

        Physical-edge probability is blue and non-physical-edge probability is red, so
        pixels supported by both classes appear magenta.  The two lazy source
        rasters are downloaded only when this diagnostic is selected.
        """

        valid = np.asarray(self.valid_mask)
        shape = valid.shape
        physical = (
            np.zeros(shape, dtype=np.uint8)
            if self.physical_edge_probability is None
            else np.asarray(self.physical_edge_probability, dtype=np.uint8)
        )
        non_edge = (
            np.zeros(shape, dtype=np.uint8)
            if self.non_edge_probability is None
            else np.asarray(self.non_edge_probability, dtype=np.uint8)
        )
        green = np.zeros(shape, dtype=np.uint8)
        alpha = np.uint8(valid > 0) * 255
        return np.dstack((non_edge, green, physical, alpha))

    def net_physical_edge_probability_rgba(self) -> np.ndarray:
        """Render the scaled positive physical-edge evidence margin in blue."""

        valid = np.asarray(self.valid_mask)
        shape = valid.shape
        physical = (
            np.zeros(shape, dtype=np.uint8)
            if self.physical_edge_probability is None
            else np.asarray(self.physical_edge_probability, dtype=np.uint8)
        )
        non_edge = (
            np.zeros(shape, dtype=np.uint8)
            if self.non_edge_probability is None
            else np.asarray(self.non_edge_probability, dtype=np.uint8)
        )
        net_physical = np.rint(
            np.maximum(
                physical.astype(np.float32)
                - float(self.net_physical_edge_internal_scale)
                * non_edge.astype(np.float32),
                0.0,
            )
        ).astype(np.uint8)
        black = np.zeros(shape, dtype=np.uint8)
        alpha = np.uint8(valid > 0) * 255
        return np.dstack((black, black, net_physical, alpha))

    def reference_edge_ridges_rgba(self) -> np.ndarray:
        values = (
            np.zeros_like(np.asarray(self.valid_mask), dtype=np.uint8)
            if self.reference_edge_ridges is None
            else np.asarray(self.reference_edge_ridges, dtype=np.uint8)
        )
        normalized = values.astype(np.float32) / 255.0
        red = np.uint8(normalized * 255.0)
        green = np.uint8(normalized * 210.0)
        blue = np.uint8(normalized * 45.0)
        alpha = np.uint8(np.asarray(self.valid_mask) > 0) * 255
        return np.dstack((red, green, blue, alpha))

    def surrounding_noise_rgba(self) -> np.ndarray | None:
        if (
            self.surrounding_noise_likelihood is None
            or self.surrounding_noise_valid_mask is None
        ):
            return None
        gray = 255 - np.asarray(self.surrounding_noise_likelihood)
        alpha = np.uint8(np.asarray(self.surrounding_noise_valid_mask) > 0) * 255
        return np.dstack((gray, gray, gray, alpha))

    def surrounding_background_rgba(self) -> np.ndarray | None:
        if (
            self.surrounding_background_likelihood is None
            or self.surrounding_background_valid_mask is None
        ):
            return None
        gray = 255 - np.asarray(self.surrounding_background_likelihood)
        alpha = (
            np.uint8(np.asarray(self.surrounding_background_valid_mask) > 0)
            * 255
        )
        return np.dstack((gray, gray, gray, alpha))

    def directed_edge_rgba(self) -> np.ndarray:
        return self._edge_rgba(self.directed_edge_hue)

    def edge_magnitude_rgba(self) -> np.ndarray:
        likelihood = np.asarray(self.edge_likelihood)
        alpha = np.uint8(np.asarray(self.valid_mask) > 0) * 255
        return np.dstack(
            (likelihood, likelihood, likelihood, alpha)
        )

    def surface_gradient_rgba(
        self, polarity: str, *, filtered: bool = False
    ) -> np.ndarray:
        prefix = "weak_" if filtered else ""
        if polarity not in {"lightening", "darkening"}:
            raise ValueError("Surface-gradient polarity must be lightening or darkening.")
        strength = getattr(self, f"{prefix}{polarity}_surface_gradient")
        hue = getattr(self, f"{prefix}{polarity}_surface_direction")
        return self._edge_rgba(hue, strength)

    def surface_gradient_magnitude_rgba(
        self, polarity: str, *, filtered: bool = False
    ) -> np.ndarray:
        prefix = "weak_" if filtered else ""
        if polarity not in {"lightening", "darkening"}:
            raise ValueError("Surface-gradient polarity must be lightening or darkening.")
        strength = np.asarray(
            getattr(self, f"{prefix}{polarity}_surface_gradient")
        )
        alpha = np.uint8(np.asarray(self.valid_mask) > 0) * 255
        return np.dstack((strength, strength, strength, alpha))

    def frequency_noise_rgba(self, channel: str, band_index: int) -> np.ndarray:
        if channel == "darkness":
            raster = self.darkness_frequency_noise_masks[band_index]
        elif channel == "colour":
            raster = self.colour_frequency_noise_masks[band_index]
        else:
            raise ValueError("Frequency-noise channel must be darkness or colour.")
        return self._heat_rgba(raster)

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
    background_exclusion_mask: np.ndarray | None = None,
    foreground_exclusion_mask: np.ndarray | None = None,
    seed_instance_annotations: np.ndarray | None = None,
    background_reference_samples: np.ndarray | None = None,
    background_reference_sample_count: int = 0,
    background_prior_lab: tuple[float, float, float] | None = None,
    background_prior_samples_lab=None,
    background_colour_enabled: bool = True,
    foreground_noise_enabled: bool = True,
    reference_edge_probability_enabled: bool = True,
    reference_edge_ridges_enabled: bool = True,
    reference_texture_prototypes_enabled: bool = True,
    surface_darkness_gradients_enabled: bool = True,
    lightening_gradient_ceiling_enabled: bool = True,
    darkening_gradient_ceiling_enabled: bool = True,
    frequency_noise_masks_enabled: bool = True,
    instance_masks_enabled: bool = True,
    seed_edge_curves_enabled: bool = True,
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
        background_exclusion_mask=background_exclusion_mask,
        foreground_exclusion_mask=foreground_exclusion_mask,
        seed_instance_annotations=seed_instance_annotations,
        background_reference_samples=background_reference_samples,
        background_reference_sample_count=background_reference_sample_count,
        background_prior_lab=background_prior_lab,
        background_prior_samples_lab=background_prior_samples_lab,
        background_colour_enabled=background_colour_enabled,
        foreground_noise_enabled=foreground_noise_enabled,
        reference_edge_probability_enabled=reference_edge_probability_enabled,
        reference_edge_ridges_enabled=reference_edge_ridges_enabled,
        reference_texture_prototypes_enabled=reference_texture_prototypes_enabled,
        surface_darkness_gradients_enabled=surface_darkness_gradients_enabled,
        lightening_gradient_ceiling_enabled=lightening_gradient_ceiling_enabled,
        darkening_gradient_ceiling_enabled=darkening_gradient_ceiling_enabled,
        frequency_noise_masks_enabled=frequency_noise_masks_enabled,
        instance_masks_enabled=instance_masks_enabled,
        seed_edge_curves_enabled=seed_edge_curves_enabled,
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
