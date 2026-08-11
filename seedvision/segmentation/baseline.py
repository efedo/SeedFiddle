"""Transparent PyTorch seed proposals for bootstrapping human review.

This is deliberately a proposal generator rather than the final instance model.
It gives the annotation UI useful starting points before trained weights exist,
and it labels every result as approximate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from seedvision.cuda import (
    CudaContext,
    GpuRaster,
    bgr_to_gray,
    bgr_to_lab,
    binary_close,
    binary_open,
    connected_components,
    distance_transform as cuda_distance_transform,
    gaussian_blur,
    gradient_magnitude,
    image_to_tensor,
    lab_colour_distribution,
    lab_colour_frequency_distribution,
    otsu_threshold,
)
from seedvision.cuda.layers import directional_edges
from seedvision.segmentation.procedural import (
    ProceduralInstanceResult,
    ProceduralInstanceSettings,
    procedural_seed_instances,
)
from seedvision.learning.contracts import LearnedInstanceResult, ModelFamily
from seedvision.learning.pipeline import (
    StarDistPipelineSettings,
    UNetPipelineSettings,
    decode_pipeline_model,
    predict_pipeline_model,
)

from seedvision.calibration.geometry import (
    DishCircle,
    DishDetectionSettings,
    detect_dish,
)
from seedvision.calibration.image import (
    CalibrationSettings,
    ImageCalibration,
    calibrate_image,
)
from seedvision.visualization import (
    ADVANCED_NODE_MODES,
    AdvancedAnalysisLayers,
    AdvancedAnalysisSettings,
    AnalysisLayerSettings,
    AnalysisLayers,
    ForegroundColourProfile,
    build_advanced_analysis_layers,
    build_analysis_layers,
)
from seedvision.timing import NodeTimingRecorder
from seedvision.visualization.advanced import (
    local_lighting_evidence_tensors,
    sensor_noise_likelihood_tensor,
)


@dataclass(frozen=True, slots=True)
class BaselineSettings:
    """User-adjustable parameters exposed by the identification pipeline node."""

    inner_radius_fraction: float = 0.90
    reference_scale_factor: float = 0.72
    foreground_otsu_fraction: float = 0.90
    circle_accumulator_threshold: int = 22
    dense_circle_relaxation: int = 4
    dense_distance_candidate_threshold: int = 30
    merge_distance_fraction: float = 0.48
    reference_roi_x_min: float = 0.43
    reference_roi_x_max: float = 0.60
    reference_roi_y_min: float = 0.66
    reference_roi_y_max: float = 0.84
    reference_colour_distance_threshold: float = 10.0
    reference_max_aspect_ratio: float = 3.0
    reference_max_components: int = 3
    fallback_diameter_fraction: float = 0.16
    foreground_chroma_weight: float = 1.8
    foreground_morphology_fraction: float = 0.06
    foreground_background_prior_tolerance: float = 24.0
    foreground_probability_softness_fraction: float = 0.18
    foreground_reference_weight: float = 0.75
    foreground_local_contrast_scale_fraction: float = 0.18
    foreground_shadow_rejection_strength: float = 1.0
    foreground_reference_components: int = 64
    foreground_distribution_fit_iterations: int = 6
    foreground_refinement_iterations: int = 2
    foreground_refinement_min_probability: float = 0.82
    foreground_frequency_weight_power: float = 0.0
    foreground_distribution_scale_multiplier: float = 1.50
    distance_blur_fraction: float = 0.025
    distance_neighborhood_fraction: float = 0.58
    distance_min_depth_fraction: float = 0.14
    distance_proposal_radius_fraction: float = 0.43
    circle_min_distance_fraction: float = 0.58
    circle_min_radius_fraction: float = 0.22
    circle_max_radius_fraction: float = 0.62
    circle_edge_threshold: int = 80
    circle_working_maximum_dimension: int = 1280
    circle_edge_magnitude_weight: float = 0.50
    circle_sensor_noise_weight: float = 0.20
    circle_flattened_grayscale_weight: float = 0.10
    circle_shadow_weight: float = 0.10
    circle_highlight_weight: float = 0.10
    circle_confidence: float = 0.62
    distance_confidence: float = 0.48

    def __post_init__(self) -> None:
        ranges = {
            "inner_radius_fraction": (self.inner_radius_fraction, 0.75, 0.96),
            "reference_scale_factor": (self.reference_scale_factor, 0.50, 1.00),
            "foreground_otsu_fraction": (self.foreground_otsu_fraction, 0.40, 1.40),
            "circle_accumulator_threshold": (
                self.circle_accumulator_threshold,
                10,
                40,
            ),
            "merge_distance_fraction": (self.merge_distance_fraction, 0.25, 0.75),
            "reference_colour_distance_threshold": (
                self.reference_colour_distance_threshold,
                2.0,
                50.0,
            ),
            "reference_max_aspect_ratio": (
                self.reference_max_aspect_ratio,
                1.0,
                8.0,
            ),
            "fallback_diameter_fraction": (
                self.fallback_diameter_fraction,
                0.05,
                0.40,
            ),
            "foreground_chroma_weight": (self.foreground_chroma_weight, 0.5, 5.0),
            "foreground_morphology_fraction": (
                self.foreground_morphology_fraction,
                0.01,
                0.20,
            ),
            "foreground_background_prior_tolerance": (
                self.foreground_background_prior_tolerance,
                2.0,
                100.0,
            ),
            "foreground_probability_softness_fraction": (
                self.foreground_probability_softness_fraction,
                0.02,
                1.0,
            ),
            "foreground_reference_weight": (
                self.foreground_reference_weight,
                0.0,
                1.0,
            ),
            "foreground_local_contrast_scale_fraction": (
                self.foreground_local_contrast_scale_fraction,
                0.03,
                0.60,
            ),
            "foreground_shadow_rejection_strength": (
                self.foreground_shadow_rejection_strength,
                0.0,
                3.0,
            ),
            "foreground_reference_components": (
                self.foreground_reference_components,
                1,
                256,
            ),
            "foreground_distribution_fit_iterations": (
                self.foreground_distribution_fit_iterations,
                1,
                20,
            ),
            "foreground_refinement_iterations": (
                self.foreground_refinement_iterations,
                0,
                8,
            ),
            "foreground_refinement_min_probability": (
                self.foreground_refinement_min_probability,
                0.50,
                0.99,
            ),
            "foreground_frequency_weight_power": (
                self.foreground_frequency_weight_power,
                0.0,
                1.0,
            ),
            "foreground_distribution_scale_multiplier": (
                self.foreground_distribution_scale_multiplier,
                0.50,
                3.0,
            ),
            "distance_blur_fraction": (self.distance_blur_fraction, 0.0, 0.12),
            "distance_neighborhood_fraction": (
                self.distance_neighborhood_fraction,
                0.20,
                1.20,
            ),
            "distance_min_depth_fraction": (
                self.distance_min_depth_fraction,
                0.03,
                0.40,
            ),
            "distance_proposal_radius_fraction": (
                self.distance_proposal_radius_fraction,
                0.10,
                0.80,
            ),
            "circle_min_distance_fraction": (
                self.circle_min_distance_fraction,
                0.20,
                1.20,
            ),
            "circle_min_radius_fraction": (
                self.circle_min_radius_fraction,
                0.05,
                0.80,
            ),
            "circle_max_radius_fraction": (
                self.circle_max_radius_fraction,
                0.10,
                1.20,
            ),
            "circle_edge_threshold": (self.circle_edge_threshold, 10, 200),
            "circle_working_maximum_dimension": (
                self.circle_working_maximum_dimension,
                512,
                4096,
            ),
            "circle_edge_magnitude_weight": (
                self.circle_edge_magnitude_weight,
                0.0,
                1.0,
            ),
            "circle_sensor_noise_weight": (
                self.circle_sensor_noise_weight,
                0.0,
                1.0,
            ),
            "circle_flattened_grayscale_weight": (
                self.circle_flattened_grayscale_weight,
                0.0,
                1.0,
            ),
            "circle_shadow_weight": (
                self.circle_shadow_weight,
                0.0,
                1.0,
            ),
            "circle_highlight_weight": (
                self.circle_highlight_weight,
                0.0,
                1.0,
            ),
            "circle_confidence": (self.circle_confidence, 0.10, 1.00),
            "distance_confidence": (self.distance_confidence, 0.10, 1.00),
        }
        for name, (value, minimum, maximum) in ranges.items():
            if not minimum <= value <= maximum:
                raise ValueError(f"{name} must be between {minimum} and {maximum}.")
        if (
            self.circle_edge_magnitude_weight
            + self.circle_sensor_noise_weight
            + self.circle_flattened_grayscale_weight
            + self.circle_shadow_weight
            + self.circle_highlight_weight
            <= 0
        ):
            raise ValueError(
                "At least one circle evidence weight must be greater than zero."
            )
        if not 0.0 <= self.reference_roi_x_min < self.reference_roi_x_max <= 1.0:
            raise ValueError("Reference ROI X fractions must satisfy 0 ≤ min < max ≤ 1.")
        if not 0.0 <= self.reference_roi_y_min < self.reference_roi_y_max <= 1.0:
            raise ValueError("Reference ROI Y fractions must satisfy 0 ≤ min < max ≤ 1.")
        if not 1 <= self.reference_max_components <= 10:
            raise ValueError("reference_max_components must be between 1 and 10.")
        if not 0 <= self.dense_circle_relaxation <= 10:
            raise ValueError("dense_circle_relaxation must be between 0 and 10.")
        if not 10 <= self.dense_distance_candidate_threshold <= 300:
            raise ValueError(
                "dense_distance_candidate_threshold must be between 10 and 300."
            )
        if self.circle_min_radius_fraction >= self.circle_max_radius_fraction:
            raise ValueError("Circle minimum radius must be below maximum radius.")


@dataclass(frozen=True, slots=True)
class SeedProposal:
    """Approximate seed center and extent in original image coordinates."""

    identifier: int
    center_x: float
    center_y: float
    radius: float
    confidence: float
    source: str


@dataclass(frozen=True, slots=True)
class BackgroundSamplingBand:
    """Exact Petri-dish annulus used for the initial background prior."""

    inner_radius_px: float
    outer_radius_px: float
    outside_vessel: bool
    sample_count: int
    buffer_cm: float = 0.0
    thickness_cm: float = 0.0
    buffer_px: float = 0.0
    thickness_px: float = 0.0


@dataclass(frozen=True, slots=True)
class BaselineAnalysis:
    """Result of the untrained classical baseline."""

    image_path: Path | None
    dish: DishCircle
    proposals: tuple[SeedProposal, ...]
    estimated_seed_diameter_px: float
    reference_seed_count: int
    foreground_threshold: float
    foreground_pixel_count: int
    analysis_region_pixel_count: int
    distance_candidate_count: int
    circle_candidate_count: int
    crowding: str
    warnings: tuple[str, ...]
    calibration: ImageCalibration
    layers: AnalysisLayers
    advanced: AdvancedAnalysisLayers
    crop_offset: tuple[int, int]
    foreground_feature: GpuRaster
    foreground_probability: GpuRaster
    foreground_mask: GpuRaster
    distance_transform: GpuRaster
    distance_candidate_geometry: np.ndarray
    circle_candidate_geometry: np.ndarray
    reference_roi: tuple[int, int, int, int]
    perimeter_background_lab: tuple[float, float, float] | None
    perimeter_background_band: BackgroundSamplingBand | None
    background_prior_deviation: float
    foreground_reference_count: int
    node_timings_seconds: dict[str, float] = field(default_factory=dict)
    procedural_instances: ProceduralInstanceResult | None = None
    unet_instances: LearnedInstanceResult | None = None
    stardist_instances: LearnedInstanceResult | None = None
    method: str = "classical fused review proposals"
    approximate: bool = True

    @property
    def count(self) -> int:
        return len(self.proposals)

    @property
    def pixels_per_mm(self) -> float | None:
        return self.calibration.pixels_per_mm


@dataclass(frozen=True, slots=True)
class _Candidate:
    x: float
    y: float
    radius: float
    confidence: float
    source: str


@dataclass(slots=True)
class PipelineAnalysisCache:
    """Per-image stage products reused across graph-scoped recomputations."""

    values: dict[str, object] = field(default_factory=dict)
    last_computed_nodes: tuple[str, ...] = ()
    last_reused_nodes: tuple[str, ...] = ()
    node_timings_seconds: dict[str, float] = field(default_factory=dict)


def analyze_path(
    path: Path,
    settings: BaselineSettings | None = None,
    calibration_settings: CalibrationSettings | None = None,
    dish_settings: DishDetectionSettings | None = None,
    layer_settings: AnalysisLayerSettings | None = None,
    advanced_settings: AdvancedAnalysisSettings | None = None,
    procedural_settings: ProceduralInstanceSettings | None = None,
    unet_settings: UNetPipelineSettings | None = None,
    stardist_settings: StarDistPipelineSettings | None = None,
    *,
    background_reference_points: tuple[tuple[float, float], ...] = (),
    foreground_reference_points: tuple[tuple[float, float], ...] = (),
    background_reference_mask: np.ndarray | None = None,
    foreground_reference_mask: np.ndarray | None = None,
    background_exclusion_mask: np.ndarray | None = None,
    foreground_exclusion_mask: np.ndarray | None = None,
    seed_instance_annotations: np.ndarray | None = None,
    background_colour_enabled: bool = True,
    enabled_nodes: set[str] | frozenset[str] | None = None,
    node_cache: PipelineAnalysisCache | None = None,
    dirty_nodes: set[str] | frozenset[str] = frozenset(),
    progress_callback=None,
    learning_root: Path | None = None,
    species: str = "unknown",
) -> BaselineAnalysis:
    """Read and analyze an image from disk."""

    cache_values = None if node_cache is None else node_cache.values
    resolved_path = str(path.resolve()).casefold()
    raw_is_dirty = "raw_images" in dirty_nodes
    raw_elapsed = None
    if (
        cache_values is not None
        and not raw_is_dirty
        and cache_values.get("raw.path") == resolved_path
        and "raw.image" in cache_values
    ):
        image = cache_values["raw.image"]
    else:
        if progress_callback is not None:
            try:
                progress_callback("raw_images", "started")
            except Exception:
                pass
        raw_started = perf_counter()
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        raw_elapsed = perf_counter() - raw_started
        if image is not None and progress_callback is not None:
            try:
                progress_callback("raw_images", "completed")
            except Exception:
                pass
        if cache_values is not None and image is not None:
            cache_values["raw.path"] = resolved_path
            cache_values["raw.image"] = image
    if image is None:
        raise RuntimeError(f"The image decoder could not read {path}")
    return analyze_image(
        image,
        image_path=path,
        settings=settings,
        calibration_settings=calibration_settings,
        dish_settings=dish_settings,
        layer_settings=layer_settings,
        advanced_settings=advanced_settings,
        procedural_settings=procedural_settings,
        unet_settings=unet_settings,
        stardist_settings=stardist_settings,
        background_reference_points=background_reference_points,
        foreground_reference_points=foreground_reference_points,
        background_reference_mask=background_reference_mask,
        foreground_reference_mask=foreground_reference_mask,
        background_exclusion_mask=background_exclusion_mask,
        foreground_exclusion_mask=foreground_exclusion_mask,
        seed_instance_annotations=seed_instance_annotations,
        background_colour_enabled=background_colour_enabled,
        enabled_nodes=enabled_nodes,
        node_cache=node_cache,
        dirty_nodes=dirty_nodes,
        progress_callback=progress_callback,
        learning_root=learning_root,
        species=species,
        _initial_timings=(
            {"raw_images": raw_elapsed}
            if raw_elapsed is not None
            else None
        ),
    )


def analyze_image(
    image: np.ndarray,
    *,
    image_path: Path | None = None,
    settings: BaselineSettings | None = None,
    calibration_settings: CalibrationSettings | None = None,
    dish_settings: DishDetectionSettings | None = None,
    layer_settings: AnalysisLayerSettings | None = None,
    advanced_settings: AdvancedAnalysisSettings | None = None,
    procedural_settings: ProceduralInstanceSettings | None = None,
    unet_settings: UNetPipelineSettings | None = None,
    stardist_settings: StarDistPipelineSettings | None = None,
    background_reference_points: tuple[tuple[float, float], ...] = (),
    foreground_reference_points: tuple[tuple[float, float], ...] = (),
    background_reference_mask: np.ndarray | None = None,
    foreground_reference_mask: np.ndarray | None = None,
    background_exclusion_mask: np.ndarray | None = None,
    foreground_exclusion_mask: np.ndarray | None = None,
    seed_instance_annotations: np.ndarray | None = None,
    background_colour_enabled: bool = True,
    enabled_nodes: set[str] | frozenset[str] | None = None,
    node_cache: PipelineAnalysisCache | None = None,
    dirty_nodes: set[str] | frozenset[str] = frozenset(),
    progress_callback=None,
    learning_root: Path | None = None,
    species: str = "unknown",
    _initial_timings: dict[str, float] | None = None,
) -> BaselineAnalysis:
    """Generate approximate seed proposals for a controlled-layout image."""

    settings = settings or BaselineSettings()
    layer_settings = layer_settings or AnalysisLayerSettings()
    advanced_settings = advanced_settings or AdvancedAnalysisSettings()
    procedural_settings = procedural_settings or ProceduralInstanceSettings()
    unet_settings = unet_settings or UNetPipelineSettings()
    stardist_settings = stardist_settings or StarDistPipelineSettings()
    learning_root = Path.cwd() if learning_root is None else Path(learning_root)
    cuda_context = CudaContext.resolve(
        requested=advanced_settings.compute_device,
        allow_cpu_fallback=advanced_settings.allow_cpu_fallback
    )
    timings = NodeTimingRecorder(
        cuda_context.device, progress_callback=progress_callback
    )
    for node_id, elapsed in (_initial_timings or {}).items():
        timings.add_seconds(node_id, elapsed)
    values = {} if node_cache is None else node_cache.values
    dirty = set(dirty_nodes)
    computed: list[str] = []
    reused: list[str] = []

    def node_enabled(node_id: str) -> bool:
        return enabled_nodes is None or node_id in enabled_nodes

    calibration_nodes = {
        "colour_reference", "ruler_detection", "deskew_colour", "scale_calibration"
    }
    calibration_dirty = bool(dirty & calibration_nodes) or "calibration" not in values
    if calibration_dirty:
        calibration = calibrate_image(
            image,
            calibration_settings,
            cuda_context=cuda_context,
            timing_recorder=timings,
        )
        values["calibration"] = calibration
        computed.extend(sorted(dirty & calibration_nodes or {"deskew_colour"}))
    else:
        calibration = values["calibration"]
        reused.append("deskew_colour")
    analysis_image = calibration.corrected_bgr
    background_reference_mask = _aligned_reference_mask(
        background_reference_mask, analysis_image.shape[:2]
    )
    foreground_reference_mask = _aligned_reference_mask(
        foreground_reference_mask, analysis_image.shape[:2]
    )
    background_exclusion_mask = _aligned_reference_mask(
        background_exclusion_mask, analysis_image.shape[:2]
    )
    foreground_exclusion_mask = _aligned_reference_mask(
        foreground_exclusion_mask, analysis_image.shape[:2]
    )
    seed_instance_annotations = _aligned_instance_annotations(
        seed_instance_annotations, analysis_image.shape[:2]
    )

    layout_dirty = calibration_dirty or "layout_detection" in dirty or "dish" not in values
    if layout_dirty:
        with timings.measure("layout_detection"):
            dish = detect_dish(
                analysis_image,
                dish_settings,
                cuda_context=cuda_context,
                image_tensor=calibration.gpu_corrected_bgr,
                pixels_per_mm=calibration.pixels_per_mm,
            )
        values["dish"] = dish
        computed.append("layout_detection")
    else:
        dish = values["dish"]
        reused.append("layout_detection")

    seed_scale_dirty = (
        calibration_dirty
        or layout_dirty
        or "seed_scale_estimation" in dirty
        or "seed_scale" not in values
    )
    if seed_scale_dirty:
        with timings.measure("seed_scale_estimation"):
            reference_diameter, reference_count = _reference_seed_diameter(
                analysis_image, dish, settings, cuda_context
            )
        values["seed_scale"] = (reference_diameter, reference_count)
        computed.append("seed_scale_estimation")
    else:
        reference_diameter, reference_count = values["seed_scale"]
        reused.append("seed_scale_estimation")
    seed_diameter = (
        reference_diameter
        or dish.outer_radius * settings.fallback_diameter_fraction
    )

    crop_dirty = calibration_dirty or layout_dirty or "crop" not in values
    if crop_dirty:
        crop, offset_x, offset_y = _dish_crop(analysis_image, dish)
        values["crop"] = (crop, offset_x, offset_y)
    else:
        crop, offset_x, offset_y = values["crop"]

    reference_radius = max(
        2,
        round(seed_diameter * layer_settings.background_sample_radius_fraction),
    )
    local_background_points = tuple(
        (float(x) - offset_x, float(y) - offset_y)
        for x, y in background_reference_points
    )
    local_foreground_points = tuple(
        (float(x) - offset_x, float(y) - offset_y)
        for x, y in foreground_reference_points
    )
    crop_height, crop_width = crop.shape[:2]
    gpu_crop_inputs = values.get("segmentation.gpu_inputs")
    if crop_dirty or gpu_crop_inputs is None:
        gpu_input_timing = timings.start(
            "foreground_segmentation", report_progress=False
        )
        if calibration.gpu_corrected_bgr is None:
            gpu_crop_source = image_to_tensor(crop, cuda_context)
        else:
            gpu_crop_source = calibration.gpu_corrected_bgr[
                :,
                :,
                offset_y : offset_y + crop_height,
                offset_x : offset_x + crop_width,
            ]
        gpu_crop_lab = bgr_to_lab(gpu_crop_source)
        gpu_crop_inputs = gpu_crop_source, gpu_crop_lab
        values["segmentation.gpu_inputs"] = gpu_crop_inputs
        timings.stop(gpu_input_timing)
    else:
        gpu_crop_source, gpu_crop_lab = gpu_crop_inputs
    local_background_reference_mask = (
        None
        if background_reference_mask is None
        else background_reference_mask[
            offset_y : offset_y + crop_height,
            offset_x : offset_x + crop_width,
        ]
    )
    local_foreground_reference_mask = (
        None
        if foreground_reference_mask is None
        else foreground_reference_mask[
            offset_y : offset_y + crop_height,
            offset_x : offset_x + crop_width,
        ]
    )
    local_background_exclusion_mask = (
        None
        if background_exclusion_mask is None
        else background_exclusion_mask[
            offset_y : offset_y + crop_height,
            offset_x : offset_x + crop_width,
        ]
    )
    local_foreground_exclusion_mask = (
        None
        if foreground_exclusion_mask is None
        else foreground_exclusion_mask[
            offset_y : offset_y + crop_height,
            offset_x : offset_x + crop_width,
        ]
    )
    local_seed_instance_annotations = (
        None
        if seed_instance_annotations is None
        else seed_instance_annotations[
            offset_y : offset_y + crop_height,
            offset_x : offset_x + crop_width,
        ]
    )
    reference_samples_dirty = (
        crop_dirty
        or seed_scale_dirty
        or "foreground_segmentation" in dirty
        or "background_likelihood" in dirty
        or "reference.samples" not in values
    )
    if reference_samples_dirty:
        reference_owner = (
            "background_likelihood"
            if "background_likelihood" in dirty or "reference.samples" not in values
            else "foreground_segmentation"
        )
        reference_timing = timings.start(reference_owner, report_progress=False)
        background_samples, accepted_background_points = _reference_colour_samples(
            analysis_image,
            background_reference_points,
            reference_radius,
            cuda_context,
            reference_mask=(
                background_reference_mask
                if background_exclusion_mask is None
                else (
                    np.asarray(background_reference_mask, dtype=bool)
                    if background_reference_mask is not None
                    else np.zeros(analysis_image.shape[:2], dtype=bool)
                )
                & ~np.asarray(background_exclusion_mask, dtype=bool)
            ),
            image_tensor=calibration.gpu_corrected_bgr,
        )
        values["reference.samples"] = (
            background_samples,
            accepted_background_points,
        )
        timings.stop(reference_timing)
    else:
        background_samples, accepted_background_points = values["reference.samples"]
    accepted_foreground_points = sum(
        1
        for x, y in local_foreground_points
        if 0 <= x < crop.shape[1] and 0 <= y < crop.shape[0]
    )

    perimeter_background_dirty = (
        calibration_dirty
        or layout_dirty
        or "perimeter_background_reference" in dirty
        or "perimeter_background" not in values
        or "perimeter.full_lab" not in values
    )
    if perimeter_background_dirty:
        with timings.measure("perimeter_background_reference"):
            full_perimeter_lab = (
                bgr_to_lab(
                    calibration.gpu_corrected_bgr
                    if calibration.gpu_corrected_bgr is not None
                    else image_to_tensor(analysis_image, cuda_context)
                )
                if calibration_dirty or "perimeter.full_lab" not in values
                else values["perimeter.full_lab"]
            )
            values["perimeter.full_lab"] = full_perimeter_lab
            perimeter_background = _dish_perimeter_background_lab(
                analysis_image,
                dish,
                calibration.pixels_per_mm,
                cuda_context,
                buffer_cm=layer_settings.perimeter_background_buffer_cm,
                thickness_cm=(
                    layer_settings.perimeter_background_band_thickness_cm
                ),
                image_tensor=calibration.gpu_corrected_bgr,
                lab_tensor=full_perimeter_lab,
            )
        values["perimeter_background"] = perimeter_background
        computed.append("perimeter_background_reference")
    else:
        perimeter_background = values["perimeter_background"]
        full_perimeter_lab = values["perimeter.full_lab"]
        reused.append("perimeter_background_reference")
    (
        background_perimeter_prior_lab,
        background_perimeter_band,
        background_perimeter_samples_lab,
    ) = perimeter_background

    foreground_dirty = (
        crop_dirty
        or seed_scale_dirty
        or "foreground_segmentation" in dirty
        or "foreground" not in values
    )
    if foreground_dirty:
        foreground_timing = timings.start("foreground_segmentation")
        (
            foreground_perimeter_prior_lab,
            foreground_perimeter_band,
            foreground_perimeter_samples_lab,
        ) = _dish_perimeter_background_lab(
            analysis_image,
            dish,
            calibration.pixels_per_mm,
            cuda_context,
            band_radii_px=_legacy_background_band_radii(
                dish, calibration.pixels_per_mm
            ),
            image_tensor=calibration.gpu_corrected_bgr,
            lab_tensor=full_perimeter_lab,
        )
        (
            feature,
            foreground_probability,
            valid,
            analysis_valid,
            foreground_threshold,
            foreground_perimeter_background_lab,
            segmentation_background_lab,
            foreground_colour_profile,
        ) = _foreground_feature(
            crop,
            settings,
            cuda_context,
            pixels_per_mm=calibration.pixels_per_mm,
            background_reference_points=local_background_points,
            background_reference_samples=background_samples,
            background_reference_mask=local_background_reference_mask,
            foreground_reference_points=local_foreground_points,
            foreground_reference_mask=local_foreground_reference_mask,
            foreground_exclusion_mask=local_foreground_exclusion_mask,
            reference_radius=reference_radius,
            seed_diameter=seed_diameter,
            analysis_radius=float(dish.outer_radius),
            perimeter_background_lab=foreground_perimeter_prior_lab,
            perimeter_background_samples_lab=foreground_perimeter_samples_lab,
            source_tensor=gpu_crop_source,
            lab_tensor=gpu_crop_lab,
        )
        mask = _foreground_mask(
            foreground_probability,
            analysis_valid,
            seed_diameter,
            settings,
            cuda_context,
        )
        values["foreground"] = (
            feature,
            foreground_probability,
            valid,
            analysis_valid,
            mask,
            foreground_threshold,
            foreground_perimeter_background_lab,
            foreground_perimeter_band,
            foreground_perimeter_samples_lab,
            segmentation_background_lab,
            foreground_colour_profile,
        )
        timings.stop(foreground_timing)
        computed.append("foreground_segmentation")
    else:
        (
            feature,
            foreground_probability,
            valid,
            analysis_valid,
            mask,
            foreground_threshold,
            foreground_perimeter_background_lab,
            foreground_perimeter_band,
            foreground_perimeter_samples_lab,
            segmentation_background_lab,
            foreground_colour_profile,
        ) = values["foreground"]
        reused.append("foreground_segmentation")
    perimeter_background_lab = background_perimeter_prior_lab
    perimeter_background_band = background_perimeter_band
    perimeter_background_samples_lab = background_perimeter_samples_lab
    if local_foreground_reference_mask is not None:
        reference_tensor = image_to_tensor(
            np.asarray(local_foreground_reference_mask, np.uint8), cuda_context
        ) > 0
        accepted_foreground_points = int(
            (
                reference_tensor
                & (image_to_tensor(valid, cuda_context) > 0)
            ).sum().item()
        )

    # The dormant circle toolbox node consumes the same cached evidence shown
    # by Edge gradients and Image-quality diagnostics. Compute those products
    # early only when the circle node is explicitly restored and enabled; the
    # normal default DAG retains its existing calculation order and cost.
    circle_edge_products = None
    circle_sensor_noise = None
    circle_local_lighting = None
    precomputed_edge_gradients = False
    circle_evidence_dirty = False
    if node_enabled("circle_candidates"):
        valid_tensor = image_to_tensor(valid, cuda_context) > 0
        edge_evidence_dirty = (
            node_enabled("edge_gradients")
            and (
                crop_dirty
                or "edge_gradients" in dirty
                or "layer.edge_gradients" not in values
            )
        )
        if node_enabled("edge_gradients"):
            if edge_evidence_dirty:
                edge_timing = timings.start("edge_gradients")
                circle_edge_products = directional_edges(
                    crop,
                    valid,
                    layer_settings,
                    cuda_context=cuda_context,
                    source_tensor=gpu_crop_source,
                    lab_tensor=gpu_crop_lab,
                    valid_tensor=valid_tensor,
                )
                values["layer.edge_gradients"] = circle_edge_products
                timings.stop(edge_timing)
                precomputed_edge_gradients = True
            else:
                circle_edge_products = values["layer.edge_gradients"]

        sensor_evidence_dirty = (
            node_enabled("image_quality")
            and (
                crop_dirty
                or seed_scale_dirty
                or "image_quality" in dirty
                or "circle.sensor_noise_evidence" not in values
            )
        )
        if node_enabled("image_quality"):
            if sensor_evidence_dirty:
                previous_advanced = values.get("advanced.layers")
                previous_sensor_noise = (
                    None
                    if previous_advanced is None
                    else previous_advanced.rasters.get("sensor_noise")
                )
                if (
                    previous_sensor_noise is not None
                    and not crop_dirty
                    and not seed_scale_dirty
                    and "image_quality" not in dirty
                ):
                    circle_sensor_noise = (
                        image_to_tensor(previous_sensor_noise, cuda_context) / 255.0
                    )
                else:
                    image_quality_timing = timings.start(
                        "image_quality", report_progress=False
                    )
                    circle_sensor_noise = sensor_noise_likelihood_tensor(
                        gpu_crop_source,
                        valid_tensor,
                        seed_diameter,
                        advanced_settings,
                    )
                    timings.stop(image_quality_timing)
                values["circle.sensor_noise_evidence"] = circle_sensor_noise
            else:
                circle_sensor_noise = values["circle.sensor_noise_evidence"]

        lighting_evidence_dirty = (
            node_enabled("illumination_decomposition")
            and (
                crop_dirty
                or seed_scale_dirty
                or "illumination_decomposition" in dirty
                or "circle.local_lighting_evidence" not in values
            )
        )
        if node_enabled("illumination_decomposition"):
            if lighting_evidence_dirty:
                previous_advanced = values.get("advanced.layers")
                lighting_names = (
                    "illumination_field",
                    "flattened_grayscale",
                    "shadow_likelihood",
                    "highlight_likelihood",
                    "reflectance_image",
                )
                previous_lighting = (
                    None
                    if previous_advanced is None
                    else tuple(
                        previous_advanced.rasters.get(name)
                        for name in lighting_names
                    )
                )
                if (
                    previous_lighting is not None
                    and all(item is not None for item in previous_lighting)
                    and not crop_dirty
                    and not seed_scale_dirty
                    and "illumination_decomposition" not in dirty
                ):
                    circle_local_lighting = tuple(
                        image_to_tensor(item, cuda_context) / 255.0
                        for item in previous_lighting
                    )
                else:
                    lighting_timing = timings.start(
                        "illumination_decomposition", report_progress=False
                    )
                    circle_local_lighting = local_lighting_evidence_tensors(
                        gpu_crop_source,
                        valid_tensor,
                        seed_diameter,
                        advanced_settings,
                    )
                    timings.stop(lighting_timing)
                values["circle.local_lighting_evidence"] = (
                    circle_local_lighting
                )
            else:
                circle_local_lighting = values[
                    "circle.local_lighting_evidence"
                ]
        circle_evidence_dirty = (
            edge_evidence_dirty
            or sensor_evidence_dirty
            or lighting_evidence_dirty
        )

    distance_dirty = (
        foreground_dirty
        or "distance_candidates" in dirty
        or "distance" not in values
    )
    if distance_dirty:
        if node_enabled("distance_candidates"):
            with timings.measure("distance_candidates"):
                marker_candidates, distance_transform = _distance_candidates(
                    mask, seed_diameter, settings, cuda_context
                )
            computed.append("distance_candidates")
        else:
            import torch

            marker_candidates = []
            distance_transform = GpuRaster(
                torch.zeros(
                    (1, 1, crop_height, crop_width),
                    device=cuda_context.device,
                    dtype=torch.float32,
                ),
                numpy_dtype=np.float32,
                name="disabled distance transform",
            )
        values["distance"] = (marker_candidates, distance_transform)
    else:
        marker_candidates, distance_transform = values["distance"]
        reused.append("distance_candidates")
    circle_threshold = settings.circle_accumulator_threshold
    if len(marker_candidates) >= settings.dense_distance_candidate_threshold:
        circle_threshold = max(
            10, circle_threshold - settings.dense_circle_relaxation
        )
    circle_dirty = (
        crop_dirty
        or seed_scale_dirty
        or circle_evidence_dirty
        or "circle_candidates" in dirty
        or "circles" not in values
    )
    if circle_dirty:
        if node_enabled("circle_candidates"):
            with timings.measure("circle_candidates"):
                circle_candidates = _circle_candidates(
                    crop,
                    seed_diameter,
                    settings,
                    cuda_context,
                    accumulator_threshold=circle_threshold,
                    source_tensor=gpu_crop_source,
                    edge_magnitude_tensor=(
                        None
                        if circle_edge_products is None
                        else circle_edge_products.strength
                    ),
                    sensor_noise_tensor=circle_sensor_noise,
                    flattened_grayscale_tensor=(
                        None
                        if circle_local_lighting is None
                        else circle_local_lighting[1]
                    ),
                    shadow_likelihood_tensor=(
                        None
                        if circle_local_lighting is None
                        else circle_local_lighting[2]
                    ),
                    highlight_likelihood_tensor=(
                        None
                        if circle_local_lighting is None
                        else circle_local_lighting[3]
                    ),
                    analysis_radius=float(dish.outer_radius),
                )
            computed.append("circle_candidates")
        else:
            circle_candidates = []
        values["circles"] = circle_candidates
    else:
        circle_candidates = values["circles"]
        reused.append("circle_candidates")

    identification_dirty = (
        distance_dirty
        or circle_dirty
        or "identification" in dirty
        or "fused" not in values
    )
    if identification_dirty:
        if node_enabled("identification"):
            identification_timing = timings.start("identification")
            circle_candidates = _filter_circle_candidates(
                circle_candidates,
                marker_candidates,
                distance_transform,
                seed_diameter,
                settings,
            )
            values["circles"] = circle_candidates
            fused = _fuse_candidates(
                marker_candidates,
                circle_candidates,
                seed_diameter,
                settings,
            )
            timings.stop(identification_timing)
            computed.append("identification")
        else:
            fused = []
        values["fused"] = fused
    else:
        fused = values["fused"]
        reused.append("identification")

    ordered_candidates = tuple(sorted(fused, key=lambda item: (item.y, item.x)))
    proposals = tuple(
        SeedProposal(
            identifier=index,
            center_x=candidate.x + offset_x,
            center_y=candidate.y + offset_y,
            radius=candidate.radius,
            confidence=candidate.confidence,
            source=candidate.source,
        )
        for index, candidate in enumerate(ordered_candidates, start=1)
    )
    layer_dirty = set(dirty)
    if crop_dirty:
        values.pop("layer.gpu_inputs", None)
        layer_dirty.update(
            {
                "background_likelihood",
                "refined_background_likelihood",
                "foreground_noise_likelihood",
                "instance_masks",
                "edge_gradients",
                "surface_darkness_gradients",
                "lightening_gradient_ceiling",
                "darkening_gradient_ceiling",
                "frequency_noise_masks",
                "directed_edges",
                "undirected_edges",
                "edge_ridges",
                "edge_traces",
                "seed_edge_curves",
            }
        )
    if seed_scale_dirty:
        layer_dirty.update(
            {
                "background_likelihood",
                "refined_background_likelihood",
                "foreground_noise_likelihood",
                "surface_darkness_gradients",
                "lightening_gradient_ceiling",
                "darkening_gradient_ceiling",
                "frequency_noise_masks",
                "instance_masks",
                "seed_edge_curves",
            }
        )
    if identification_dirty:
        layer_dirty.update({"instance_masks", "seed_edge_curves"})
    if foreground_dirty:
        layer_dirty.update({"foreground_noise_likelihood", "seed_edge_curves"})
    if perimeter_background_dirty:
        layer_dirty.update(
            {
                "background_likelihood",
                "refined_background_likelihood",
                "instance_masks",
                "seed_edge_curves",
            }
        )
    if "edge_gradients" in layer_dirty:
        layer_dirty.update(
            {
                "directed_edges",
                "undirected_edges",
                "edge_ridges",
                "edge_traces",
                "seed_edge_curves",
            }
        )
    if "surface_darkness_gradients" in layer_dirty:
        layer_dirty.update(
            {"lightening_gradient_ceiling", "darkening_gradient_ceiling"}
        )
    if "edge_ridges" in layer_dirty:
        layer_dirty.update({"edge_traces", "seed_edge_curves"})
    if "edge_traces" in layer_dirty:
        layer_dirty.add("seed_edge_curves")
    if "background_likelihood" in layer_dirty:
        layer_dirty.update(
            {
                "refined_background_likelihood",
                "instance_masks",
                "seed_edge_curves",
            }
        )
    if "refined_background_likelihood" in layer_dirty:
        layer_dirty.update({"instance_masks", "seed_edge_curves"})
    if "instance_masks" in layer_dirty:
        layer_dirty.add("seed_edge_curves")
    if precomputed_edge_gradients:
        # Downstream invalidation above still applies, but the shared gradient
        # field itself has already been refreshed for the circle consumer.
        layer_dirty.discard("edge_gradients")

    layer_cache_keys = {
        "background_likelihood": "layer.background",
        "refined_background_likelihood": "layer.refined_background",
        "foreground_noise_likelihood": "layer.foreground_noise",
        "instance_masks": "layer.instances",
        "edge_gradients": "layer.edge_gradients",
        "surface_darkness_gradients": "layer.surface_darkness_gradients",
        "lightening_gradient_ceiling": "layer.lightening_gradient_ceiling",
        "darkening_gradient_ceiling": "layer.darkening_gradient_ceiling",
        "frequency_noise_masks": "layer.frequency_noise_masks",
        "directed_edges": "layer.directed_edges",
        "undirected_edges": "layer.undirected_edges",
        "edge_ridges": "layer.edge_ridges",
        "edge_traces": "layer.edge_traces",
        "seed_edge_curves": "layer.seed_edge_curves",
    }
    for node_id, cache_key in layer_cache_keys.items():
        if node_id == "edge_gradients" and precomputed_edge_gradients:
            computed.append(node_id)
        elif node_id in layer_dirty or cache_key not in values:
            computed.append(node_id)
        else:
            reused.append(node_id)

    if crop_dirty or "layer.gpu_inputs" not in values:
        values["layer.gpu_inputs"] = (
            gpu_crop_source,
            gpu_crop_lab,
            image_to_tensor(valid, cuda_context) > 0,
        )
    if (
        crop_dirty
        or perimeter_background_dirty
        or "layer.surrounding_noise_inputs" not in values
    ):
        values["layer.surrounding_noise_inputs"] = _surrounding_noise_inputs(
            analysis_image.shape[:2],
            calibration.gpu_corrected_bgr,
            analysis_image,
            dish,
            perimeter_background_band,
            cuda_context,
        )
    (
        surrounding_noise_source,
        surrounding_noise_valid,
        surrounding_noise_offset_x,
        surrounding_noise_offset_y,
    ) = values["layer.surrounding_noise_inputs"]
    layers = build_analysis_layers(
        crop,
        valid,
        np.asarray(
            [(candidate.x, candidate.y) for candidate in ordered_candidates],
            dtype=np.float32,
        ),
        np.asarray(
            [candidate.radius for candidate in ordered_candidates], dtype=np.float32
        ),
        seed_diameter,
        offset_x=offset_x,
        offset_y=offset_y,
        background_reference_points=local_background_points,
        foreground_reference_points=local_foreground_points,
        background_reference_mask=local_background_reference_mask,
        foreground_reference_mask=local_foreground_reference_mask,
        background_exclusion_mask=local_background_exclusion_mask,
        foreground_exclusion_mask=local_foreground_exclusion_mask,
        seed_instance_annotations=local_seed_instance_annotations,
        background_reference_samples=background_samples,
        background_reference_sample_count=accepted_background_points,
        background_prior_lab=perimeter_background_lab,
        background_prior_samples_lab=perimeter_background_samples_lab,
        background_colour_enabled=background_colour_enabled,
        foreground_noise_enabled=node_enabled("foreground_noise_likelihood"),
        surface_darkness_gradients_enabled=node_enabled(
            "surface_darkness_gradients"
        ),
        lightening_gradient_ceiling_enabled=node_enabled(
            "lightening_gradient_ceiling"
        ),
        darkening_gradient_ceiling_enabled=node_enabled(
            "darkening_gradient_ceiling"
        ),
        frequency_noise_masks_enabled=node_enabled("frequency_noise_masks"),
        instance_masks_enabled=node_enabled("instance_masks"),
        seed_edge_curves_enabled=node_enabled("seed_edge_curves"),
        foreground_probability=foreground_probability,
        foreground_colour_profile=foreground_colour_profile,
        surrounding_noise_source_tensor=surrounding_noise_source,
        surrounding_noise_valid_tensor=surrounding_noise_valid,
        surrounding_noise_offset_x=surrounding_noise_offset_x,
        surrounding_noise_offset_y=surrounding_noise_offset_y,
        settings=layer_settings,
        cache_values=values,
        dirty_nodes=layer_dirty,
        cuda_context=cuda_context,
        timing_recorder=timings,
    )
    background_prior_deviation = 0.0
    if (
        perimeter_background_lab is not None
        and layers.background_colour_profile is not None
    ):
        detected = np.asarray(
            layers.background_colour_profile.centre_lab, dtype=np.float32
        )
        prior = np.asarray(perimeter_background_lab, dtype=np.float32)
        difference = detected - prior
        background_prior_deviation = float(
            np.sqrt(
                difference[0] ** 2
                + 1.5 * difference[1] ** 2
                + 1.5 * difference[2] ** 2
            )
        )

    advanced_node_ids = set(ADVANCED_NODE_MODES)
    enabled_advanced_nodes = (
        advanced_node_ids
        if enabled_nodes is None
        else advanced_node_ids & set(enabled_nodes)
    )
    requested_advanced_nodes = dirty & enabled_advanced_nodes
    previous_advanced = values.get("advanced.layers")
    if previous_advanced is not None and any(
        node_id in dirty and node_id not in enabled_advanced_nodes
        for node_id in advanced_node_ids
    ):
        # A newly disabled branch must not retain stale visible products. Rebuild
        # compact zero placeholders and the independent enabled diagnostics.
        previous_advanced = None
    upstream_advanced_dirty = (
        calibration_dirty
        or foreground_dirty
        or distance_dirty
        or identification_dirty
        or bool(
            layer_dirty
            & {
                "foreground_noise_likelihood",
                "instance_masks",
                "directed_edges",
            }
        )
    )
    if previous_advanced is None:
        requested_advanced_nodes = set(enabled_advanced_nodes)
    elif upstream_advanced_dirty and not requested_advanced_nodes:
        # Direct API callers may name only a legacy source node, whereas the
        # Qt graph supplies the full recursive dirty set.
        requested_advanced_nodes = set(enabled_advanced_nodes)
    advanced_dirty = (
        bool(requested_advanced_nodes)
        or upstream_advanced_dirty
        or previous_advanced is None
    )
    if advanced_dirty:
        calibration_anchors: list[tuple[float, float, float]] = []
        if calibration.colour_card is not None:
            corrected_bounds = calibration.transform_points(
                calibration.colour_card.bounds
            )
            card_center = np.mean(corrected_bounds, axis=0)
            calibration_anchors.append(
                (
                    float(card_center[0]),
                    float(card_center[1]),
                    1.0 - float(calibration.colour_card.confidence),
                )
            )
        if calibration.ruler is not None:
            corrected_endpoints = calibration.ruler_endpoints_corrected()
            ruler_risk = 1.0 - float(calibration.ruler.confidence)
            calibration_anchors.extend(
                (float(point[0]), float(point[1]), ruler_risk)
                for point in corrected_endpoints
            )
        advanced = build_advanced_analysis_layers(
            crop,
            valid,
            feature,
            foreground_threshold,
            mask,
            distance_transform,
            layers.instance_labels,
            np.asarray(
                [(candidate.x, candidate.y) for candidate in ordered_candidates],
                dtype=np.float32,
            ),
            np.asarray(
                [candidate.radius for candidate in ordered_candidates],
                dtype=np.float32,
            ),
            np.asarray(
                [(item.x, item.y, item.radius) for item in circle_candidates],
                dtype=np.float32,
            ).reshape(-1, 3),
            np.asarray(
                [(item.x, item.y, item.radius) for item in marker_candidates],
                dtype=np.float32,
            ).reshape(-1, 3),
            seed_diameter,
            offset_x=offset_x,
            offset_y=offset_y,
            foreground_probability=foreground_probability,
            foreground_noise_probability=layers.foreground_noise_likelihood,
            background_colour_probability=layers.background_likelihood,
            background_noise_probability=layers.refined_background_likelihood,
            full_image_shape=analysis_image.shape[:2],
            calibration_anchors=tuple(calibration_anchors),
            settings=advanced_settings,
            previous=previous_advanced,
            dirty_nodes=requested_advanced_nodes,
            enabled_nodes=enabled_advanced_nodes,
            source_tensor=layers.gpu_source,
            valid_tensor=layers.gpu_valid,
            sensor_noise_tensor=circle_sensor_noise,
            local_lighting_tensors=circle_local_lighting,
            timing_recorder=timings,
        )
        values["advanced.layers"] = advanced
        computed.extend(
            node_id for node_id in ADVANCED_NODE_MODES
            if node_id in requested_advanced_nodes
        )
        reused.extend(
            node_id for node_id in ADVANCED_NODE_MODES
            if node_id in enabled_advanced_nodes
            and node_id not in requested_advanced_nodes
        )
    else:
        advanced = values["advanced.layers"]
        reused.extend(
            node_id for node_id in ADVANCED_NODE_MODES
            if node_id in enabled_advanced_nodes
        )

    # This new review branch is opt-in for API callers that omit graph state;
    # the desktop always passes its explicit active-node set.
    procedural_enabled = (
        enabled_nodes is not None and "procedural_instances" in enabled_nodes
    )
    procedural_dirty = (
        calibration_dirty
        or foreground_dirty
        or advanced_dirty
        or bool(
            layer_dirty
            & {
                "background_likelihood",
                "refined_background_likelihood",
                "foreground_noise_likelihood",
                "edge_gradients",
                "edge_ridges",
            }
        )
        or "procedural_instances" in dirty
        or "segmentation.procedural_instances" not in values
    )
    if procedural_enabled and procedural_dirty:
        with timings.measure("procedural_instances"):
            procedural_result = procedural_seed_instances(
                layers.valid_mask,
                seed_diameter,
                foreground_probability=foreground_probability,
                foreground_noise_probability=layers.foreground_noise_likelihood,
                background_probability=layers.background_likelihood,
                refined_background_probability=layers.refined_background_likelihood,
                edge_magnitude=layers.edge_likelihood,
                edge_ridges=layers.edge_ridges,
                sensor_noise=advanced.rasters["sensor_noise"],
                shadow_likelihood=advanced.rasters["shadow_likelihood"],
                seed_instance_annotations=local_seed_instance_annotations,
                settings=procedural_settings,
            )
        values["segmentation.procedural_instances"] = procedural_result
        computed.append("procedural_instances")
    elif procedural_enabled:
        procedural_result = values["segmentation.procedural_instances"]
        reused.append("procedural_instances")
    else:
        procedural_result = None
        values.pop("segmentation.procedural_instances", None)

    learned_evidence = {
        "foreground_colour": foreground_probability,
        "foreground_noise": layers.foreground_noise_likelihood,
        "background_colour": layers.background_likelihood,
        "background_noise": layers.refined_background_likelihood,
        "edge_magnitude": layers.edge_likelihood,
        "sensor_noise": advanced.rasters["sensor_noise"],
        "flattened_grayscale": advanced.rasters["flattened_grayscale"],
        "shadow": advanced.rasters["shadow_likelihood"],
        "highlight": advanced.rasters["highlight_likelihood"],
    }
    learned_upstream_dirty = (
        calibration_dirty
        or layout_dirty
        or seed_scale_dirty
        or foreground_dirty
        or advanced_dirty
        or bool(
            layer_dirty
            & {
                "background_likelihood",
                "refined_background_likelihood",
                "foreground_noise_likelihood",
                "edge_gradients",
            }
        )
    )

    def learned_branch(node_id, family, branch_settings):
        enabled = enabled_nodes is not None and node_id in enabled_nodes
        prefix = f"learning.{node_id}"
        if not enabled:
            for suffix in ("outputs", "checkpoint", "inference_signature", "decoder_signature", "result"):
                values.pop(f"{prefix}.{suffix}", None)
            return None
        configured_checkpoint = Path(branch_settings.checkpoint_path).expanduser()
        resolved_checkpoint = (
            configured_checkpoint.resolve()
            if configured_checkpoint.is_absolute()
            else (learning_root / configured_checkpoint).resolve()
        )
        checkpoint_fingerprint = (
            resolved_checkpoint.stat().st_mtime_ns
            if resolved_checkpoint.is_file()
            else None
        )
        inference_signature = (
            *branch_settings.inference_signature,
            str(resolved_checkpoint).casefold(),
            checkpoint_fingerprint,
            str(species),
            round(float(seed_diameter), 5),
        )
        inference_dirty = (
            learned_upstream_dirty
            or values.get(f"{prefix}.inference_signature") != inference_signature
            or f"{prefix}.outputs" not in values
        )
        decoder_signature = (
            *branch_settings.decoder_signature,
            round(float(seed_diameter), 5),
            None
            if local_seed_instance_annotations is None
            else hash(local_seed_instance_annotations.tobytes()),
        )
        decoder_dirty = (
            inference_dirty
            or node_id in dirty
            or values.get(f"{prefix}.decoder_signature") != decoder_signature
            or f"{prefix}.result" not in values
        )
        if decoder_dirty:
            with timings.measure(node_id):
                if inference_dirty:
                    outputs, checkpoint_id = predict_pipeline_model(
                        family,
                        root=learning_root,
                        settings=branch_settings,
                        source_bgr=layers.gpu_source,
                        valid_mask=layers.gpu_valid,
                        evidence=learned_evidence,
                        species=species,
                        seed_diameter_px=seed_diameter,
                    )
                    values[f"{prefix}.outputs"] = outputs
                    values[f"{prefix}.checkpoint"] = checkpoint_id
                    values[f"{prefix}.inference_signature"] = inference_signature
                else:
                    outputs = values[f"{prefix}.outputs"]
                    checkpoint_id = values[f"{prefix}.checkpoint"]
                decoded = decode_pipeline_model(
                    family,
                    outputs,
                    settings=branch_settings,
                    seed_diameter_px=seed_diameter,
                    painted_instances=(
                        local_seed_instance_annotations
                        if family is ModelFamily.UNET_WATERSHED
                        else None
                    ),
                    valid_mask=layers.valid_mask,
                    checkpoint_id=checkpoint_id,
                )
            values[f"{prefix}.result"] = decoded
            values[f"{prefix}.decoder_signature"] = decoder_signature
            computed.append(node_id)
            return decoded
        reused.append(node_id)
        return values[f"{prefix}.result"]

    unet_result = learned_branch(
        "unet_instances", ModelFamily.UNET_WATERSHED, unet_settings
    )
    stardist_result = learned_branch(
        "stardist_instances", ModelFamily.STARDIST, stardist_settings
    )

    reported_instance_count = (
        unet_result.count
        if unet_result is not None
        else stardist_result.count
        if stardist_result is not None
        else procedural_result.count
        if procedural_result is not None
        else len(proposals)
    )
    nominal_seed_area = np.pi * (seed_diameter * 0.5) ** 2
    usable_dish_area = np.pi * (
        dish.outer_radius * settings.inner_radius_fraction
    ) ** 2
    packing_estimate = reported_instance_count * nominal_seed_area / max(1.0, usable_dish_area)
    if packing_estimate >= 0.62 or reported_instance_count >= 250:
        crowding = "high"
    elif packing_estimate >= 0.28 or reported_instance_count >= 70:
        crowding = "moderate"
    else:
        crowding = "low"

    warnings = [
        "Learned instances are not publication-validated; review and correction are required."
        if unet_result is not None or stardist_result is not None
        else "Untrained procedural instances; review and correction are required."
        if procedural_result is not None
        else "Untrained classical proposals; review and correction are required."
    ]
    warnings.extend(calibration.warnings)
    if layers.background_mode == "disabled":
        warnings.append(
            "Background colour and noise-frequency analysis were disabled for this run."
        )
    elif (
        background_reference_points
        or (
            background_reference_mask is not None
            and np.any(background_reference_mask)
        )
    ) and layers.background_mode != "manual":
        warnings.append(
            "No painted background area fell inside the corrected image; automatic "
            "background selection was used."
        )
    if (
        foreground_reference_points
        or (
            foreground_reference_mask is not None
            and np.any(foreground_reference_mask)
        )
    ) and not accepted_foreground_points:
        warnings.append(
            "No painted foreground area fell inside the detected dish crop."
        )
    if background_prior_deviation >= 18.0:
        warnings.append(
            "Detected background colour deviates substantially from the median "
            f"colour in the {perimeter_background_band.thickness_cm:.2f} cm band "
            f"after the {perimeter_background_band.buffer_cm:.2f} cm dish buffer "
            f"(weighted Lab distance {background_prior_deviation:.1f})."
        )
    if calibration.pixels_per_mm is None:
        warnings.append("Absolute ruler scale was not assigned.")
    if reference_diameter is None:
        warnings.append(
            "Reference seeds were not isolated reliably; seed size was estimated from the dish."
        )
    if crowding != "low":
        warnings.append(
            "Touching/overlapping seeds make the proposal count especially uncertain."
        )
    if procedural_result is not None:
        low_confidence_fraction = float(
            np.mean(procedural_result.instance_confidences < 0.55)
        ) if procedural_result.count else 1.0
        if low_confidence_fraction >= 0.20:
            warnings.append(
                "Procedural instance separation contains substantial low-confidence "
                "structure; review the confidence overlay before using its count."
            )

    measured_timings = timings.finalize()
    node_timings_seconds = (
        dict(node_cache.node_timings_seconds)
        if node_cache is not None
        else {}
    )
    node_timings_seconds.update(measured_timings)

    result = BaselineAnalysis(
        image_path=image_path,
        dish=dish,
        proposals=proposals,
        estimated_seed_diameter_px=float(seed_diameter),
        reference_seed_count=reference_count,
        foreground_threshold=foreground_threshold,
        foreground_pixel_count=(
            mask.count_above(1)
            if hasattr(mask, "count_above")
            else int(np.count_nonzero(mask))
        ),
        analysis_region_pixel_count=(
            analysis_valid.count_above(1)
            if hasattr(analysis_valid, "count_above")
            else int(np.count_nonzero(analysis_valid))
        ),
        distance_candidate_count=len(marker_candidates),
        circle_candidate_count=len(circle_candidates),
        crowding=crowding,
        warnings=tuple(warnings),
        calibration=calibration,
        layers=layers,
        advanced=advanced,
        crop_offset=(offset_x, offset_y),
        foreground_feature=feature,
        foreground_probability=foreground_probability,
        foreground_mask=mask,
        distance_transform=distance_transform,
        distance_candidate_geometry=np.asarray(
            [(item.x, item.y, item.radius) for item in marker_candidates],
            dtype=np.float32,
        ).reshape(-1, 3),
        circle_candidate_geometry=np.asarray(
            [(item.x, item.y, item.radius) for item in circle_candidates],
            dtype=np.float32,
        ).reshape(-1, 3),
        reference_roi=(
            round(analysis_image.shape[1] * settings.reference_roi_x_min),
            round(analysis_image.shape[0] * settings.reference_roi_y_min),
            round(analysis_image.shape[1] * settings.reference_roi_x_max),
            round(analysis_image.shape[0] * settings.reference_roi_y_max),
        ),
        perimeter_background_lab=perimeter_background_lab,
        perimeter_background_band=perimeter_background_band,
        background_prior_deviation=background_prior_deviation,
        foreground_reference_count=accepted_foreground_points,
        node_timings_seconds=node_timings_seconds,
        procedural_instances=procedural_result,
        unet_instances=unet_result,
        stardist_instances=stardist_result,
        method=(
            "multi-head U-Net plus pattern-aware watershed (review required)"
            if unet_result is not None
            else "StarDist star-convex polygons (review required)"
            if stardist_result is not None
            else "procedural marker-controlled watershed (review required)"
            if procedural_result is not None
            else "classical fused review proposals"
        ),
    )
    if node_cache is not None:
        values["result"] = result
        node_cache.last_computed_nodes = tuple(dict.fromkeys(computed))
        node_cache.last_reused_nodes = tuple(dict.fromkeys(reused))
        node_cache.node_timings_seconds = node_timings_seconds
    return result


def _dish_crop(
    image: np.ndarray, dish: DishCircle
) -> tuple[np.ndarray, int, int]:
    radius = dish.outer_radius
    x0 = max(0, dish.center_x - radius)
    y0 = max(0, dish.center_y - radius)
    x1 = min(image.shape[1], dish.center_x + radius)
    y1 = min(image.shape[0], dish.center_y + radius)
    return image[y0:y1, x0:x1], x0, y0


def _background_band_radii(
    dish: DishCircle,
    pixels_per_mm: float | None,
    *,
    buffer_cm: float = 0.35,
    thickness_cm: float = 0.50,
) -> tuple[float, float]:
    """Return the exact buffered outside-rim annulus in corrected-image pixels."""

    vessel_radius = float(dish.outer_radius)
    pixels_per_cm = (
        10.0 * float(pixels_per_mm)
        if pixels_per_mm is not None and pixels_per_mm > 0
        else vessel_radius / 4.8
    )
    buffer_px = float(buffer_cm) * pixels_per_cm
    thickness_px = float(thickness_cm) * pixels_per_cm
    inner_radius = vessel_radius + buffer_px
    return inner_radius, inner_radius + thickness_px


def _legacy_background_band_radii(
    dish: DishCircle,
    pixels_per_mm: float | None,
) -> tuple[float, float]:
    """Retain the established compact prior used only by foreground segmentation."""

    vessel_radius = float(dish.outer_radius)
    band_width = (
        5.0 * float(pixels_per_mm)
        if pixels_per_mm is not None and pixels_per_mm > 0
        else vessel_radius * 0.08
    )
    band_width = max(3.0, min(band_width, vessel_radius * 0.25))
    return vessel_radius * 1.005, vessel_radius + band_width


def _surrounding_noise_inputs(
    image_shape: tuple[int, int],
    full_image_tensor,
    image: np.ndarray,
    dish: DishCircle,
    band: BackgroundSamplingBand,
    cuda_context: CudaContext,
) -> tuple[object, object, int, int]:
    """Crop and mask only the annulus needed by the noise-profile node."""

    import torch

    margin = int(np.ceil(band.outer_radius_px))
    x0 = max(0, dish.center_x - margin)
    y0 = max(0, dish.center_y - margin)
    x1 = min(image_shape[1], dish.center_x + margin)
    y1 = min(image_shape[0], dish.center_y + margin)
    if full_image_tensor is None:
        source = image_to_tensor(image[y0:y1, x0:x1], cuda_context)
    else:
        source = full_image_tensor[:, :, y0:y1, x0:x1]
    height, width = source.shape[-2:]
    yy, xx = torch.meshgrid(
        torch.arange(height, device=cuda_context.device, dtype=torch.float32),
        torch.arange(width, device=cuda_context.device, dtype=torch.float32),
        indexing="ij",
    )
    center_x = float(dish.center_x - x0)
    center_y = float(dish.center_y - y0)
    distance = torch.sqrt(
        (xx - center_x).square() + (yy - center_y).square()
    )
    annulus = (
        (distance >= float(band.inner_radius_px))
        & (distance <= float(band.outer_radius_px))
    )[None, None]
    return source, annulus, x0, y0


def _reference_colour_samples(
    image: np.ndarray,
    points: tuple[tuple[float, float], ...],
    radius: int,
    cuda_context: CudaContext,
    *,
    reference_mask: np.ndarray | None = None,
    image_tensor=None,
) -> tuple[object, int]:
    """Collect compact colour patches anywhere in the corrected image."""

    import torch

    if not points and (reference_mask is None or not np.any(reference_mask)):
        return torch.empty(
            (0, 3), device=cuda_context.device, dtype=torch.float32
        ), 0

    height, width = image.shape[:2]
    yy, xx = torch.meshgrid(
        torch.arange(height, device=cuda_context.device, dtype=torch.float32),
        torch.arange(width, device=cuda_context.device, dtype=torch.float32),
        indexing="ij",
    )
    sample_mask = torch.zeros((height, width), device=cuda_context.device, dtype=torch.bool)
    accepted = 0
    if reference_mask is not None:
        mask_values = np.asarray(reference_mask, dtype=np.uint8)
        if mask_values.shape != (height, width):
            raise ValueError("Reference mask must match the corrected image dimensions.")
        sample_mask |= image_to_tensor(mask_values, cuda_context)[0, 0] > 0
        accepted = int(np.count_nonzero(mask_values))
    for point_x, point_y in points:
        x = round(float(point_x))
        y = round(float(point_y))
        if not (0 <= x < image.shape[1] and 0 <= y < image.shape[0]):
            continue
        sample_mask |= (xx - x).square() + (yy - y).square() <= max(1, int(radius)) ** 2
        if reference_mask is None:
            accepted += 1
    if not accepted:
        return torch.empty(
            (0, 3), device=cuda_context.device, dtype=torch.float32
        ), 0
    tensor = (
        image_to_tensor(image, cuda_context)
        if image_tensor is None
        else image_tensor
    )[0].permute(1, 2, 0)
    samples = tensor[sample_mask]
    return samples, accepted


def _aligned_reference_mask(
    mask: np.ndarray | None,
    shape: tuple[int, int],
) -> np.ndarray | None:
    """Return a Boolean reference mask aligned to the corrected image."""

    if mask is None:
        return None
    values = np.asarray(mask, dtype=np.uint8)
    if values.ndim != 2:
        raise ValueError("Reference masks must be two-dimensional binary arrays.")
    if values.shape != shape:
        values = cv2.resize(
            values,
            (shape[1], shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    return values > 0


def _aligned_instance_annotations(
    annotations: np.ndarray | None,
    shape: tuple[int, int],
) -> np.ndarray | None:
    """Return an unsigned seed-identity label map aligned to the corrected image."""

    if annotations is None:
        return None
    values = np.asarray(annotations)
    if values.ndim != 2:
        raise ValueError("Seed instance annotations must be a two-dimensional label map.")
    if np.any(values < 0) or np.any(values > np.iinfo(np.uint16).max):
        raise ValueError("Seed instance annotation IDs must be between 0 and 65,535.")
    values = values.astype(np.uint16, copy=False)
    if values.shape != shape:
        values = cv2.resize(
            values,
            (shape[1], shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    return values


def _sample_count(samples) -> int:
    """Return a sample tensor's known size without materializing it on CPU."""

    if samples is None:
        return 0
    try:
        import torch

        if torch.is_tensor(samples):
            return int(samples.numel() // 3)
    except ImportError:
        pass
    return int(np.asarray(samples).size // 3)


def _bgr_samples_to_lab(samples, cuda_context: CudaContext):
    """Convert an N x 3 BGR collection to Lab without a host round-trip."""

    import torch

    if torch.is_tensor(samples):
        values = samples.to(
            device=cuda_context.device, dtype=torch.float32
        ).reshape(-1, 3)
        nchw = values.T[None, :, :, None]
    else:
        nchw = image_to_tensor(
            np.asarray(samples, np.uint8).reshape(-1, 1, 3), cuda_context
        )
    return bgr_to_lab(nchw)[0, :, :, 0].T


def _reference_seed_diameter(
    image: np.ndarray,
    dish: DishCircle,
    settings: BaselineSettings,
    cuda_context: CudaContext,
) -> tuple[float | None, int]:
    """Estimate sample seed size from the isolated seeds above the ruler."""

    height, width = image.shape[:2]
    x0, x1 = (
        round(width * settings.reference_roi_x_min),
        round(width * settings.reference_roi_x_max),
    )
    y0, y1 = (
        round(height * settings.reference_roi_y_min),
        round(height * settings.reference_roi_y_max),
    )
    crop = image[y0:y1, x0:x1]
    if crop.size == 0:
        return None, 0

    import torch

    tensor = image_to_tensor(crop, cuda_context)
    lab = bgr_to_lab(tensor)
    background = torch.median(lab.reshape(3, -1), dim=1).values
    delta = torch.sqrt(
        (lab[:, 0:1] - background[0]).square()
        + 1.8 * (lab[:, 1:2] - background[1]).square()
        + 1.8 * (lab[:, 2:3] - background[2]).square()
    )
    mask = delta >= settings.reference_colour_distance_threshold
    kernel_scale = max(3, round(max(image.shape[:2]) / 900.0))
    if kernel_scale % 2 == 0:
        kernel_scale += 1
    mask = binary_close(binary_open(mask, kernel_scale), kernel_scale)
    labels, stats = connected_components(mask)
    component_count = int(labels.max().item()) + 1

    diameters: list[float] = []
    minimum_area = max(100.0, image.shape[0] * image.shape[1] * 0.00002)
    maximum_area = crop.shape[0] * crop.shape[1] * 0.10
    for index in range(1, component_count):
        x = int(stats["left"][index])
        y = int(stats["top"][index])
        component_width = int(stats["width"][index])
        component_height = int(stats["height"][index])
        area = int(stats["area"][index])
        touches_edge = (
            x <= 1
            or y <= 1
            or x + component_width >= crop.shape[1] - 1
            or y + component_height >= crop.shape[0] - 1
        )
        aspect = max(component_width, component_height) / max(
            1, min(component_width, component_height)
        )
        equivalent_diameter = float(np.sqrt(4.0 * area / np.pi))
        plausible_for_dish = (
            dish.outer_radius * 0.07
            <= equivalent_diameter
            <= dish.outer_radius * 0.35
        )
        if (
            minimum_area <= area <= maximum_area
            and not touches_edge
            and aspect <= settings.reference_max_aspect_ratio
            and plausible_for_dish
        ):
            diameters.append(equivalent_diameter)

    if not diameters:
        return None, 0
    # The layout normally contains two seeds. Limit the influence of incidental
    # small components without hard-coding exactly two.
    diameters = sorted(diameters, reverse=True)[: settings.reference_max_components]
    # The low-contrast shadow belongs to the thresholded component and inflates
    # its equivalent diameter.  The sparse hand-counted pilot calibrates this
    # foreground-component diameter to the effective center-spacing diameter.
    return float(np.median(diameters) * settings.reference_scale_factor), len(diameters)


def _foreground_feature(
    crop: np.ndarray,
    settings: BaselineSettings,
    cuda_context: CudaContext,
    *,
    pixels_per_mm: float | None = None,
    background_reference_points: tuple[tuple[float, float], ...] = (),
    background_reference_samples=None,
    background_reference_mask: np.ndarray | None = None,
    foreground_reference_points: tuple[tuple[float, float], ...] = (),
    foreground_reference_mask: np.ndarray | None = None,
    foreground_exclusion_mask: np.ndarray | None = None,
    reference_radius: int = 3,
    seed_diameter: float = 24.0,
    valid_radius: float | None = None,
    analysis_radius: float | None = None,
    center_x: float | None = None,
    center_y: float | None = None,
    perimeter_background_lab: tuple[float, float, float] | None = None,
    perimeter_background_samples_lab=None,
    source_tensor=None,
    lab_tensor=None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    tuple[float, float, float],
    tuple[float, float, float],
    ForegroundColourProfile | None,
]:
    height, width = crop.shape[:2]
    import torch

    yy, xx = torch.meshgrid(
        torch.arange(height, device=cuda_context.device, dtype=torch.float32),
        torch.arange(width, device=cuda_context.device, dtype=torch.float32),
        indexing="ij",
    )
    dish_radius = (
        min(height, width) * 0.5
        if valid_radius is None
        else float(valid_radius)
    )
    seed_area_radius = (
        dish_radius if analysis_radius is None else float(analysis_radius)
    )
    local_center_x = width * 0.5 if center_x is None else float(center_x)
    local_center_y = height * 0.5 if center_y is None else float(center_y)
    radial_distance = torch.sqrt(
        (xx - local_center_x).square() + (yy - local_center_y).square()
    )
    # Diagnostic layers cover the complete visible dish. Proposal generation
    # retains the configured inner restriction so the glass rim cannot become
    # a chain of false seeds.
    valid_pixels = radial_distance <= dish_radius * 0.995
    analysis_pixels = (
        radial_distance <= seed_area_radius * settings.inner_radius_fraction
    )
    source = image_to_tensor(crop, cuda_context) if source_tensor is None else source_tensor
    lab_nchw = bgr_to_lab(source) if lab_tensor is None else lab_tensor
    lab = lab_nchw[0].permute(1, 2, 0)

    perimeter_width = (
        5.0 * pixels_per_mm
        if pixels_per_mm is not None and pixels_per_mm > 0
        else dish_radius * 0.08
    )
    perimeter_width = max(3.0, min(float(perimeter_width), dish_radius * 0.25))
    perimeter_ring = valid_pixels & (
        radial_distance >= dish_radius * 0.995 - perimeter_width
    )
    if int(perimeter_ring.sum().item()) < 16:
        perimeter_ring = valid_pixels
    if perimeter_background_lab is None:
        perimeter_background = torch.median(lab[perimeter_ring], dim=0).values
    else:
        perimeter_background = torch.as_tensor(
            perimeter_background_lab,
            device=cuda_context.device,
            dtype=lab.dtype,
        )

    foreground_reference_tensor = _reference_point_mask(
        height,
        width,
        foreground_reference_points,
        reference_radius,
        cuda_context,
    ) & valid_pixels
    if foreground_reference_mask is not None:
        foreground_reference_tensor |= (
            image_to_tensor(
                np.asarray(foreground_reference_mask, np.uint8), cuda_context
            )[0, 0]
            > 0
        ) & valid_pixels
    foreground_exclusion_tensor = torch.zeros_like(valid_pixels)
    if foreground_exclusion_mask is not None:
        foreground_exclusion_tensor = image_to_tensor(
            np.asarray(foreground_exclusion_mask, np.uint8), cuda_context
        )[0, 0] > 0
        foreground_exclusion_tensor &= valid_pixels
        foreground_reference_tensor &= ~foreground_exclusion_tensor
    manual_mask = _reference_point_mask(
        height,
        width,
        background_reference_points,
        reference_radius,
        cuda_context,
    )
    if background_reference_mask is not None:
        manual_mask |= image_to_tensor(
            np.asarray(background_reference_mask, np.uint8), cuda_context
        )[0, 0] > 0
    manual_mask &= valid_pixels
    supplied_background = None
    if _sample_count(background_reference_samples):
        supplied_background = _bgr_samples_to_lab(
            background_reference_samples, cuda_context
        )
    elif int(manual_mask.sum().item()) >= 1:
        supplied_background = lab[manual_mask]
    if supplied_background is None and _sample_count(perimeter_background_samples_lab):
        supplied_background = perimeter_background_samples_lab
        background_refinement_iterations = 0
    else:
        background_refinement_iterations = settings.foreground_refinement_iterations
    if supplied_background is not None:
        (
            _,
            background_centres,
            _,
            background_weights,
            _,
            _,
        ) = lab_colour_distribution(
            lab,
            supplied_background,
            valid_pixels & ~foreground_reference_tensor,
            maximum_components=min(8, settings.foreground_reference_components),
            fit_iterations=settings.foreground_distribution_fit_iterations,
            refinement_iterations=background_refinement_iterations,
            refinement_min_probability=settings.foreground_refinement_min_probability,
            frequency_weight_power=settings.foreground_frequency_weight_power,
            scale_multiplier=settings.foreground_distribution_scale_multiplier,
            scale_floors=(8.0, 3.0, 3.0),
            distance_weights=(
                1.0,
                settings.foreground_chroma_weight,
                settings.foreground_chroma_weight,
            ),
        )
        background = background_centres[int(torch.argmax(background_weights).item())]
        background_delta = lab[:, :, None, :] - background_centres[
            None, None, :, :
        ]
        background_distances = (
            background_delta[:, :, :, 0].square()
            + settings.foreground_chroma_weight
            * background_delta[:, :, :, 1].square()
            + settings.foreground_chroma_weight
            * background_delta[:, :, :, 2].square()
        )
        feature = torch.sqrt(torch.min(background_distances, dim=2).values)
    else:
        prior_distance = torch.sqrt(
            (lab[:, :, 0] - perimeter_background[0]).square()
            + settings.foreground_chroma_weight
            * (lab[:, :, 1] - perimeter_background[1]).square()
            + settings.foreground_chroma_weight
            * (lab[:, :, 2] - perimeter_background[2]).square()
        )
        candidate_background = valid_pixels & (
            prior_distance <= settings.foreground_background_prior_tolerance
        )
        if int(candidate_background.sum().item()) < 16:
            # A seed-filled dish may contain no visible patch of the surrounding
            # tray colour. In that case the measured outside-dish prior is the
            # evidence; selecting an arbitrary fraction of the dish would turn
            # the dominant seed colour into "background".
            background = perimeter_background
        else:
            background = torch.median(lab[candidate_background], dim=0).values
        feature = torch.sqrt(
            (lab[:, :, 0] - background[0]).square()
            + settings.foreground_chroma_weight
            * (lab[:, :, 1] - background[1]).square()
            + settings.foreground_chroma_weight
            * (lab[:, :, 2] - background[2]).square()
        )
    feature = torch.where(valid_pixels, feature, torch.zeros_like(feature))

    otsu = otsu_threshold(feature[analysis_pixels])
    threshold = max(8.0, float(otsu) * settings.foreground_otsu_fraction)
    softness = max(
        2.0,
        threshold * settings.foreground_probability_softness_fraction,
    )
    # Distance from a pale tray alone makes the darkest inter-seed shadows look
    # more foreground-like than the seed coats. Seed surfaces are normally local
    # lightness maxima at the seed scale, whereas gaps are local minima. Add that
    # evidence in log-odds space so shadows are rejected without requiring every
    # legitimate seed colour to be close to one global prototype.
    lightness = lab[:, :, 0]
    contrast_sigma = max(
        1.0,
        seed_diameter * settings.foreground_local_contrast_scale_fraction,
    )
    local_lightness = gaussian_blur(
        lightness[None, None], contrast_sigma
    )[0, 0]
    contrast_softness = max(2.0, softness * 0.35)
    visible_background_fraction = torch.mean(
        (feature[valid_pixels] <= settings.foreground_background_prior_tolerance).float()
    )
    # Sparse dishes already supply abundant true background and do not need the
    # crowded-dish shadow heuristic. Ramp it in only as visible tray disappears.
    crowded_dish_gate = torch.clamp(
        (0.30 - visible_background_fraction) / 0.20,
        0.0,
        1.0,
    )
    probability_logit = (feature - threshold) / softness
    probability_logit = probability_logit + (
        settings.foreground_shadow_rejection_strength
        * crowded_dish_gate
        * (lightness - local_lightness)
        / contrast_softness
    )
    probability = torch.sigmoid(probability_logit)

    foreground_centres = None
    foreground_scales = None
    foreground_weights = None
    foreground_fitted_sample_count = 0
    foreground_refinement_rounds = 0
    foreground_profile_sample_mask = None
    excluded_centres = None
    excluded_scales = None
    excluded_weights = None
    exclusion_strength = 0.95

    # Painted foreground pixels form an empirical colour-frequency table.
    # Individual light/dark pattern colours remain separate anchors; cautious
    # refinement can widen their tolerances but cannot move or invent modes.
    if foreground_reference_tensor.any():
        (
            prototype_probability,
            foreground_centres,
            foreground_scales,
            foreground_weights,
            foreground_fitted_sample_count,
            foreground_refinement_rounds,
        ) = lab_colour_frequency_distribution(
            lab,
            lab[foreground_reference_tensor],
            valid_pixels & ~manual_mask & ~foreground_exclusion_tensor,
            maximum_bins=settings.foreground_reference_components,
            refinement_iterations=settings.foreground_refinement_iterations,
            refinement_min_probability=settings.foreground_refinement_min_probability,
            frequency_weight_power=settings.foreground_frequency_weight_power,
            scale_multiplier=settings.foreground_distribution_scale_multiplier,
            scale_floors=(8.0, 4.0, 4.0),
            distance_weights=(
                1.0,
                settings.foreground_chroma_weight,
                settings.foreground_chroma_weight,
            ),
        )
        # Reference evidence participates in the same calculation everywhere.
        # Nothing underneath the brush is forced to one; a matching unpainted
        # pixel receives exactly the same colour-derived log-odds contribution.
        probability = torch.sigmoid(
            torch.logit(probability.clamp(1e-5, 1.0 - 1e-5))
            + 6.0
            * settings.foreground_reference_weight
            * prototype_probability
        )
        foreground_profile_sample_mask = foreground_reference_tensor

    if foreground_exclusion_tensor.any():
        (
            excluded_membership,
            excluded_centres,
            excluded_scales,
            excluded_weights,
            _,
            _,
        ) = lab_colour_frequency_distribution(
            lab,
            lab[foreground_exclusion_tensor],
            valid_pixels,
            maximum_bins=max(16, settings.foreground_reference_components),
            refinement_iterations=0,
            frequency_weight_power=0.0,
            scale_multiplier=settings.foreground_distribution_scale_multiplier,
            scale_floors=(8.0, 4.0, 4.0),
            distance_weights=(
                1.0,
                settings.foreground_chroma_weight,
                settings.foreground_chroma_weight,
            ),
        )
        # Painted exclusions define a negative colour-frequency model. Apply
        # it identically at every matching colour rather than zeroing pixels
        # merely because the brush passed over their coordinates.
        probability = probability * (
            1.0 - exclusion_strength * excluded_membership
        )

    # Without a painted anchor, summarize the colours that the current
    # foreground calculation itself considers confidently seed-like. This is a
    # diagnostic fit only: it does not feed back into foreground probability.
    if foreground_centres is None:
        automatic_foreground = (
            valid_pixels
            & ~manual_mask
            & ~foreground_exclusion_tensor
            & (probability >= max(0.70, settings.foreground_refinement_min_probability))
        )
        if int(automatic_foreground.sum().item()) < 16:
            eligible_probability = torch.where(
                valid_pixels & ~manual_mask,
                probability,
                torch.zeros_like(probability),
            )
            if int((valid_pixels & ~manual_mask & ~foreground_exclusion_tensor).sum().item()):
                cutoff = torch.quantile(
                    eligible_probability[
                        valid_pixels & ~manual_mask & ~foreground_exclusion_tensor
                    ],
                    0.90,
                )
                automatic_foreground = (
                    valid_pixels
                    & ~manual_mask
                    & ~foreground_exclusion_tensor
                    & (probability >= cutoff)
                )
        if int(automatic_foreground.sum().item()):
            (
                _,
                foreground_centres,
                foreground_scales,
                foreground_weights,
                foreground_fitted_sample_count,
                foreground_refinement_rounds,
            ) = lab_colour_distribution(
                lab,
                lab[automatic_foreground],
                valid_pixels & ~manual_mask & ~foreground_exclusion_tensor,
                maximum_components=min(8, settings.foreground_reference_components),
                fit_iterations=settings.foreground_distribution_fit_iterations,
                refinement_iterations=0,
                refinement_min_probability=settings.foreground_refinement_min_probability,
                frequency_weight_power=settings.foreground_frequency_weight_power,
                scale_multiplier=settings.foreground_distribution_scale_multiplier,
                scale_floors=(8.0, 4.0, 4.0),
                distance_weights=(
                    1.0,
                    settings.foreground_chroma_weight,
                    settings.foreground_chroma_weight,
                ),
            )
            foreground_profile_sample_mask = automatic_foreground

    foreground_colour_profile = None
    if foreground_centres is not None and foreground_profile_sample_mask is not None:
        sample_bgr = source[0].permute(1, 2, 0)[foreground_profile_sample_mask]
        if int(sample_bgr.shape[0]) > 200000:
            sample_bgr = sample_bgr[:: max(1, int(sample_bgr.shape[0]) // 200000)]
        low = torch.quantile(sample_bgr, 0.05, dim=0).round().clamp(0, 255)
        high = torch.quantile(sample_bgr, 0.95, dim=0).round().clamp(0, 255)
        dominant = int(torch.argmax(foreground_weights).item())
        foreground_colour_profile = ForegroundColourProfile(
            centre_lab=tuple(
                float(value) for value in foreground_centres[dominant].cpu().tolist()
            ),
            scale_lab=tuple(
                float(value) for value in foreground_scales[dominant].cpu().tolist()
            ),
            bgr_low=tuple(int(value) for value in low.cpu().tolist()),
            bgr_high=tuple(int(value) for value in high.cpu().tolist()),
            sample_count=int(foreground_fitted_sample_count),
            sample_fraction=int(foreground_fitted_sample_count)
            / max(int(valid_pixels.sum().item()), 1),
            component_centres_lab=tuple(
                tuple(float(value) for value in row)
                for row in foreground_centres.cpu().tolist()
            ),
            component_scales_lab=tuple(
                tuple(float(value) for value in row)
                for row in foreground_scales.cpu().tolist()
            ),
            component_weights=tuple(
                float(value) for value in foreground_weights.cpu().tolist()
            ),
            refinement_iterations=int(foreground_refinement_rounds),
            excluded_component_centres_lab=(
                ()
                if excluded_centres is None
                else tuple(
                    tuple(float(value) for value in row)
                    for row in excluded_centres.cpu().tolist()
                )
            ),
            excluded_component_scales_lab=(
                ()
                if excluded_scales is None
                else tuple(
                    tuple(float(value) for value in row)
                    for row in excluded_scales.cpu().tolist()
                )
            ),
            excluded_component_weights=(
                ()
                if excluded_weights is None
                else tuple(
                    float(value) for value in excluded_weights.cpu().tolist()
                )
            ),
            exclusion_strength=exclusion_strength,
        )
    probability = torch.where(
        valid_pixels,
        probability,
        torch.zeros_like(probability),
    )

    return (
        GpuRaster(
            feature[None, None], numpy_dtype=np.float32,
            name="foreground feature",
        ),
        GpuRaster(
            torch.round(probability[None, None] * 255.0).to(torch.uint8),
            numpy_dtype=np.uint8,
            name="foreground probability",
        ),
        GpuRaster(
            valid_pixels[None, None].to(torch.uint8) * 255,
            numpy_dtype=np.uint8,
            name="foreground valid region",
        ),
        GpuRaster(
            analysis_pixels[None, None].to(torch.uint8) * 255,
            numpy_dtype=np.uint8,
            name="foreground proposal region",
        ),
        threshold,
        tuple(float(value) for value in perimeter_background.cpu().tolist()),
        tuple(float(value) for value in background.cpu().tolist()),
        foreground_colour_profile,
    )


def _dish_perimeter_background_lab(
    image: np.ndarray,
    dish: DishCircle,
    pixels_per_mm: float | None,
    cuda_context: CudaContext,
    *,
    buffer_cm: float = 0.35,
    thickness_cm: float = 0.50,
    band_radii_px: tuple[float, float] | None = None,
    image_tensor=None,
    lab_tensor=None,
) -> tuple[tuple[float, float, float], BackgroundSamplingBand, object]:
    """Return the colour summary and sampled Lab pixels outside a dish.

    The bounded sample remains on the analysis device so downstream colour
    models retain the annulus's multimodal distribution instead of reducing it
    to one regional average.
    """

    import torch

    height, width = image.shape[:2]
    yy, xx = torch.meshgrid(
        torch.arange(height, device=cuda_context.device, dtype=torch.float32),
        torch.arange(width, device=cuda_context.device, dtype=torch.float32),
        indexing="ij",
    )
    distance = torch.sqrt(
        (xx - float(dish.center_x)).square()
        + (yy - float(dish.center_y)).square()
    )
    vessel_radius = float(dish.outer_radius)
    inner_radius, outer_radius = (
        _background_band_radii(
            dish,
            pixels_per_mm,
            buffer_cm=buffer_cm,
            thickness_cm=thickness_cm,
        )
        if band_radii_px is None
        else (float(band_radii_px[0]), float(band_radii_px[1]))
    )
    buffer_width = inner_radius - vessel_radius
    band_width = outer_radius - inner_radius
    pixels_per_cm = (
        10.0 * float(pixels_per_mm)
        if pixels_per_mm is not None and pixels_per_mm > 0
        else vessel_radius / 4.8
    )
    reported_buffer_cm = buffer_width / max(pixels_per_cm, 1e-6)
    reported_thickness_cm = band_width / max(pixels_per_cm, 1e-6)
    ring = (distance >= inner_radius) & (distance <= outer_radius)
    sample_count = int(ring.sum().item())
    outside_vessel = True
    if sample_count < 16:
        outer_radius = max(0.0, vessel_radius - buffer_width)
        inner_radius = max(0.0, outer_radius - band_width)
        ring = (distance <= outer_radius) & (distance >= inner_radius)
        sample_count = int(ring.sum().item())
        outside_vessel = False
    if lab_tensor is None:
        source = (
            image_to_tensor(image, cuda_context)
            if image_tensor is None
            else image_tensor
        )
        lab_tensor = bgr_to_lab(source)
    lab = lab_tensor[0].permute(1, 2, 0)
    samples = lab[ring]
    if int(samples.shape[0]) > 32768:
        indices = torch.linspace(
            0,
            int(samples.shape[0]) - 1,
            32768,
            device=cuda_context.device,
        ).round().long()
        samples = samples[indices]
    median = torch.median(samples, dim=0).values
    return (
        tuple(float(value) for value in median.cpu().tolist()),
        BackgroundSamplingBand(
            inner_radius_px=float(inner_radius),
            outer_radius_px=float(outer_radius),
            outside_vessel=outside_vessel,
            sample_count=sample_count,
            buffer_cm=float(reported_buffer_cm),
            thickness_cm=float(reported_thickness_cm),
            buffer_px=float(buffer_width),
            thickness_px=float(band_width),
        ),
        samples,
    )


def _foreground_mask(
    probability: np.ndarray,
    valid: np.ndarray,
    seed_diameter: float,
    settings: BaselineSettings,
    cuda_context: CudaContext,
) -> object:
    import torch

    tensor = image_to_tensor(probability, cuda_context) / 255.0
    valid_tensor = image_to_tensor(valid, cuda_context) > 0
    mask = (tensor >= 0.5) & valid_tensor
    kernel_size = max(3, round(seed_diameter * settings.foreground_morphology_fraction))
    if kernel_size % 2 == 0:
        kernel_size += 1
    mask = binary_close(binary_open(mask, kernel_size), kernel_size)
    return GpuRaster(
        mask.to(torch.uint8) * 255,
        numpy_dtype=np.uint8,
        name="foreground binary mask",
    )


def _reference_point_mask(
    height: int,
    width: int,
    points: tuple[tuple[float, float], ...],
    radius: int,
    cuda_context: CudaContext,
):
    """Create a hard circular reference mask on the selected tensor device."""

    import torch

    mask = torch.zeros(
        (height, width), device=cuda_context.device, dtype=torch.bool
    )
    if not points:
        return mask
    yy, xx = torch.meshgrid(
        torch.arange(height, device=cuda_context.device, dtype=torch.float32),
        torch.arange(width, device=cuda_context.device, dtype=torch.float32),
        indexing="ij",
    )
    radius_squared = float(max(1, radius) ** 2)
    for point_x, point_y in points:
        if 0 <= point_x < width and 0 <= point_y < height:
            mask |= (
                (xx - float(point_x)).square()
                + (yy - float(point_y)).square()
                <= radius_squared
            )
    return mask


def _distance_candidates(
    mask: np.ndarray,
    seed_diameter: float,
    settings: BaselineSettings,
    cuda_context: CudaContext,
) -> tuple[list[_Candidate], object]:
    import torch.nn.functional as functional

    mask_tensor = image_to_tensor(mask, cuda_context) > 0
    distance = cuda_distance_transform(mask_tensor)
    distance = gaussian_blur(
        distance, max(0.1, seed_diameter * settings.distance_blur_fraction)
    )
    neighborhood = max(5, round(seed_diameter * settings.distance_neighborhood_fraction))
    if neighborhood % 2 == 0:
        neighborhood += 1
    dilated = functional.max_pool2d(
        distance, neighborhood, stride=1, padding=neighborhood // 2
    )
    peaks = (
        (distance >= dilated - 1e-4)
        & (distance >= max(2.0, seed_diameter * settings.distance_min_depth_fraction))
    )
    labels, stats = connected_components(peaks)
    component_count = int(labels.max().item()) + 1
    candidates: list[_Candidate] = []
    for index in range(1, component_count):
        if stats["area"][index] > neighborhood * neighborhood:
            continue
        candidates.append(
            _Candidate(
                x=float(stats["centroid_x"][index]),
                y=float(stats["centroid_y"][index]),
                radius=seed_diameter * settings.distance_proposal_radius_fraction,
                confidence=settings.distance_confidence,
                source="distance",
            )
        )
    return candidates, GpuRaster(
        distance,
        numpy_dtype=np.float32,
        name="distance transform",
    )


def _circle_candidates(
    crop: np.ndarray,
    seed_diameter: float,
    settings: BaselineSettings,
    cuda_context: CudaContext,
    *,
    accumulator_threshold: int | None = None,
    source_tensor=None,
    edge_magnitude_tensor=None,
    sensor_noise_tensor=None,
    flattened_grayscale_tensor=None,
    shadow_likelihood_tensor=None,
    highlight_likelihood_tensor=None,
    analysis_radius: float | None = None,
    analysis_center: tuple[float, float] | None = None,
) -> list[_Candidate]:
    import torch
    import torch.nn.functional as functional

    source = (
        image_to_tensor(crop, cuda_context)
        if source_tensor is None
        else source_tensor
    )
    source_height, source_width = source.shape[-2:]
    working_scale = min(
        1.0,
        float(settings.circle_working_maximum_dimension)
        / max(source_height, source_width),
    )
    working_height = max(16, round(source_height * working_scale))
    working_width = max(16, round(source_width * working_scale))
    working_source = (
        source
        if (working_height, working_width) == (source_height, source_width)
        else functional.interpolate(
            source, (working_height, working_width), mode="area"
        )
    )
    working_diameter = seed_diameter * working_scale

    def working_evidence(values):
        if values is None:
            return None
        if isinstance(values, GpuRaster):
            tensor = values.gpu_tensor(
                device=cuda_context.device, dtype=torch.float32
            )
        elif torch.is_tensor(values):
            tensor = values.to(
                device=cuda_context.device, dtype=torch.float32
            )
        else:
            tensor = image_to_tensor(np.asarray(values), cuda_context)
        if tensor.ndim == 2:
            tensor = tensor[None, None]
        elif tensor.ndim == 3:
            tensor = tensor[None]
        if tensor.shape[1] > 1:
            tensor = torch.mean(tensor, dim=1, keepdim=True)
        if float(tensor.max().item()) > 1.5:
            tensor = tensor / 255.0
        if tensor.shape[-2:] != (working_height, working_width):
            tensor = functional.interpolate(
                tensor,
                (working_height, working_width),
                mode="bilinear",
                align_corners=False,
            )
        return tensor.clamp(0.0, 1.0)

    edge = working_evidence(edge_magnitude_tensor)
    if edge is None:
        # Preserve a direct-call fallback, while the authored pipeline wiring
        # supplies the shared multichannel edge magnitude in normal use.
        gray = gaussian_blur(bgr_to_gray(working_source), 1.2)
        gradient, _, _ = gradient_magnitude(gray)
        high = torch.quantile(gradient.reshape(-1), 0.995).clamp_min(1.0)
        edge = (gradient / high).clamp(0.0, 1.0)
    edge_floor = settings.circle_edge_threshold / 255.0 * 0.45
    edge = torch.where(edge >= edge_floor, edge, torch.zeros_like(edge))

    evidence_weight = settings.circle_edge_magnitude_weight
    evidence = edge * evidence_weight

    def add_boundary_evidence(values, weight: float) -> None:
        nonlocal evidence, evidence_weight
        raster = working_evidence(values)
        if raster is None or weight <= 0:
            return
        boundary, _, _ = gradient_magnitude(
            gaussian_blur(raster, 0.8), scharr=True
        )
        high = torch.quantile(boundary.reshape(-1), 0.995).clamp_min(1e-6)
        boundary = (boundary / high).clamp(0.0, 1.0)
        boundary = torch.where(
            boundary >= edge_floor * 0.65,
            boundary,
            torch.zeros_like(boundary),
        )
        evidence += boundary * weight
        evidence_weight += weight

    # These inputs are region/intensity rasters, so their spatial transitions
    # supply ring support rather than their filled interiors.
    add_boundary_evidence(
        sensor_noise_tensor, settings.circle_sensor_noise_weight
    )
    add_boundary_evidence(
        flattened_grayscale_tensor,
        settings.circle_flattened_grayscale_weight,
    )
    add_boundary_evidence(
        shadow_likelihood_tensor, settings.circle_shadow_weight
    )
    add_boundary_evidence(
        highlight_likelihood_tensor, settings.circle_highlight_weight
    )
    edge = evidence / max(evidence_weight, 1e-6)
    minimum_radius = max(
        4, round(working_diameter * settings.circle_min_radius_fraction)
    )
    maximum_radius = max(
        minimum_radius + 2,
        round(working_diameter * settings.circle_max_radius_fraction),
    )
    radius_count = min(10, max(3, maximum_radius - minimum_radius + 1))
    radii = torch.linspace(
        minimum_radius,
        maximum_radius,
        radius_count,
        device=cuda_context.device,
    )
    half_size = maximum_radius + 3
    yy, xx = torch.meshgrid(
        torch.arange(-half_size, half_size + 1, device=cuda_context.device),
        torch.arange(-half_size, half_size + 1, device=cuda_context.device),
        indexing="ij",
    )
    radial_distance = torch.sqrt(xx.float().square() + yy.float().square())
    kernels = []
    for radius in radii:
        ring = torch.exp(-0.5 * ((radial_distance - radius) / 1.25).square())
        ring /= ring.sum().clamp_min(1.0)
        kernels.append(ring)
    kernel_bank = torch.stack(kernels)[:, None]
    response_bank = functional.conv2d(edge, kernel_bank, padding=half_size)
    response, radius_index = torch.max(response_bank, dim=1, keepdim=True)
    minimum_distance = max(
        8, round(working_diameter * settings.circle_min_distance_fraction)
    )
    if minimum_distance % 2 == 0:
        minimum_distance += 1
    local_maximum = functional.max_pool2d(
        response,
        minimum_distance,
        stride=1,
        padding=minimum_distance // 2,
    )
    applied_threshold = (
        settings.circle_accumulator_threshold
        if accumulator_threshold is None
        else accumulator_threshold
    )
    # Map the familiar 0..200 accumulator control onto the normalized CUDA
    # ring-support response. This wider slope preserves useful sensitivity at
    # the default while making strict settings materially reject weak circles.
    response_threshold = 0.127 + applied_threshold * 0.0090
    if analysis_center is None:
        center_x, center_y = working_width * 0.5, working_height * 0.5
    else:
        center_x = float(analysis_center[0]) * working_scale
        center_y = float(analysis_center[1]) * working_scale
    maximum_distance = (
        (
            min(working_height, working_width) * 0.5
            if analysis_radius is None
            else float(analysis_radius) * working_scale
        )
        * settings.inner_radius_fraction
    )
    coordinate_y, coordinate_x = torch.meshgrid(
        torch.arange(working_height, device=cuda_context.device),
        torch.arange(working_width, device=cuda_context.device),
        indexing="ij",
    )
    inside = (
        (coordinate_x - center_x).float().square()
        + (coordinate_y - center_y).float().square()
        <= maximum_distance * maximum_distance
    )
    peaks = (
        (response[0, 0] >= local_maximum[0, 0] - 1e-6)
        & (response[0, 0] >= response_threshold)
        & inside
    )
    positions = torch.nonzero(peaks, as_tuple=False)
    if positions.numel() == 0:
        return []
    scores = response[0, 0, positions[:, 0], positions[:, 1]]
    order = torch.argsort(scores, descending=True)[:600]
    positions = positions[order]
    selected_radii = radii[
        radius_index[0, 0, positions[:, 0], positions[:, 1]]
    ]
    position_values = positions.detach().cpu().numpy()
    radius_values = selected_radii.detach().cpu().numpy()
    return [
        _Candidate(
            x=float(position[1]) / working_scale,
            y=float(position[0]) / working_scale,
            radius=float(radius) / working_scale,
            confidence=settings.circle_confidence,
            source="circle",
        )
        for position, radius in zip(position_values, radius_values, strict=True)
    ]


def _fuse_candidates(
    primary: list[_Candidate],
    secondary: list[_Candidate],
    seed_diameter: float,
    settings: BaselineSettings,
) -> list[_Candidate]:
    """Fuse complementary proposals with deterministic distance suppression."""

    accepted = list(primary)
    minimum_distance = seed_diameter * settings.merge_distance_fraction
    for candidate in sorted(secondary, key=lambda item: item.confidence, reverse=True):
        nearest_index: int | None = None
        nearest_distance = float("inf")
        for index, existing in enumerate(accepted):
            distance = float(np.hypot(candidate.x - existing.x, candidate.y - existing.y))
            if distance < nearest_distance:
                nearest_index = index
                nearest_distance = distance
        if nearest_index is None or nearest_distance >= minimum_distance:
            accepted.append(candidate)
            continue

        existing = accepted[nearest_index]
        total = existing.confidence + candidate.confidence
        accepted[nearest_index] = _Candidate(
            x=(existing.x * existing.confidence + candidate.x * candidate.confidence) / total,
            y=(existing.y * existing.confidence + candidate.y * candidate.confidence) / total,
            radius=max(existing.radius, candidate.radius),
            confidence=max(existing.confidence, candidate.confidence),
            source="fused",
        )
    return accepted


def _filter_circle_candidates(
    circles: list[_Candidate],
    distance_candidates: list[_Candidate],
    distance_transform: object,
    seed_diameter: float,
    settings: BaselineSettings,
) -> list[_Candidate]:
    """Reject ring responses unsupported by seed-interior geometry.

    A circle near a distance peak is retained for centre/radius fusion. A new
    circle-only seed must have appreciable foreground depth and be far enough
    from every existing distance peak to represent a distinct seed rather than
    an internal lupin coat boundary.
    """

    if not distance_candidates:
        return circles
    merge_distance = seed_diameter * settings.merge_distance_fraction
    distinct_distance = seed_diameter * 0.75
    minimum_depth = seed_diameter * 0.19
    if not circles:
        return []
    if isinstance(distance_transform, GpuRaster):
        import torch

        distance = distance_transform.gpu_tensor(dtype=torch.float32)
        if distance.ndim == 4:
            distance = distance[0, 0]
        height, width = distance.shape
        circle_xy = torch.as_tensor(
            [(item.x, item.y) for item in circles],
            device=distance.device,
            dtype=torch.float32,
        )
        distance_xy = torch.as_tensor(
            [(item.x, item.y) for item in distance_candidates],
            device=distance.device,
            dtype=torch.float32,
        )
        nearest = torch.cdist(circle_xy, distance_xy).amin(dim=1)
        x = circle_xy[:, 0].round().long().clamp(0, width - 1)
        y = circle_xy[:, 1].round().long().clamp(0, height - 1)
        depth = distance[y, x]
        keep = (nearest <= merge_distance) | (
            (nearest >= distinct_distance) & (depth >= minimum_depth)
        )
        keep_values = keep.detach().cpu().tolist()
        return [
            circle for circle, retained in zip(circles, keep_values, strict=True)
            if retained
        ]

    height, width = distance_transform.shape
    accepted: list[_Candidate] = []
    for circle in circles:
        nearest = min(
            np.hypot(circle.x - candidate.x, circle.y - candidate.y)
            for candidate in distance_candidates
        )
        x = min(width - 1, max(0, round(circle.x)))
        y = min(height - 1, max(0, round(circle.y)))
        depth = float(distance_transform[y, x])
        if nearest <= merge_distance or (
            nearest >= distinct_distance and depth >= minimum_depth
        ):
            accepted.append(circle)
    return accepted
