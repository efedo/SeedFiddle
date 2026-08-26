"""CUDA-first diagnostic analyses for seed segmentation and coat assessment.

The algorithms in this module are transparent classical tensor operations. They
do not pretend to be trained probabilities; every output is a normalized
image-derived confidence map intended for troubleshooting and annotation.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np

from seedvision.cuda.ops import GpuRaster

if TYPE_CHECKING:
    import torch


ADVANCED_NODE_MODES = {
    "boundary_normals": "boundary_confidence",
    "touching_split": "touching_split_likelihood",
    "ellipse_likelihood": "ellipse_likelihood",
    "proposal_disagreement": "proposal_disagreement",
    "assignment_confidence": "instance_assignment_confidence",
    "contact_graph": "contact_graph",
    "illumination_decomposition": "flattened_grayscale",
    "image_quality": "image_quality_risk",
    "radial_profile": "radial_profile_residual",
    "wrinkling": "wrinkling_likelihood",
    "coat_damage": "coat_damage_likelihood",
    "pattern_decomposition": "pattern_classes",
    "colour_probabilities": "colour_classes",
    "calibration_residuals": "calibration_residual_risk",
}

ADVANCED_OVERLAY_LABELS = (
    ("Boundary confidence + normals", "boundary_confidence"),
    ("Boundary magnitude", "boundary_magnitude"),
    ("Touching-seed split likelihood", "touching_split_likelihood"),
    ("Multiscale ellipse likelihood", "ellipse_likelihood"),
    ("Proposal-source disagreement", "proposal_disagreement"),
    ("Instance-assignment confidence", "instance_assignment_confidence"),
    ("Contested instance pixels", "contested_pixels"),
    ("Occlusion/contact graph", "contact_graph"),
    ("Estimated illumination field", "illumination_field"),
    ("Flattened grayscale", "flattened_grayscale"),
    ("Local shadow likelihood", "shadow_likelihood"),
    ("Local highlight likelihood", "highlight_likelihood"),
    ("Reflectance image", "reflectance_image"),
    ("Specular/glare likelihood", "glare_likelihood"),
    ("Image-quality risk", "image_quality_risk"),
    ("Local focus", "focus_quality"),
    ("Clipped highlights", "clipped_highlights"),
    ("Underexposure", "underexposure"),
    ("Sensor/noise likelihood", "sensor_noise"),
    ("Radial profile residual", "radial_profile_residual"),
    ("Normalized radial coordinate", "radial_coordinate"),
    ("Wrinkling likelihood", "wrinkling_likelihood"),
    ("Seed-coat damage likelihood", "coat_damage_likelihood"),
    ("Pattern decomposition", "pattern_classes"),
    ("Pattern confidence", "pattern_confidence"),
    ("Colour-class probabilities", "colour_classes"),
    ("Colour uncertainty", "colour_uncertainty"),
    ("Calibration residual risk", "calibration_residual_risk"),
)

COLOUR_CLASS_NAMES = ("white", "yellow", "green", "red", "brown", "black")
PATTERN_CLASS_NAMES = ("plain", "spots", "mottled", "patches", "striped", "bicolour")


@dataclass(frozen=True, slots=True)
class AdvancedAnalysisSettings:
    """User controls shared by the CUDA diagnostic branch."""

    compute_device: str = "cuda"
    allow_cpu_fallback: bool = False
    maximum_dimension: int = 1024
    interior_smoothing_fraction: float = 0.055
    interior_background_weight: float = 0.55
    interior_foreground_noise_weight: float = 0.35
    interior_reference_texture_weight: float = 0.35
    boundary_width_fraction: float = 0.025
    split_neck_fraction: float = 0.36
    ellipse_radial_tolerance: float = 0.22
    disagreement_scale_fraction: float = 0.16
    assignment_boundary_penalty: float = 0.70
    contact_distance_multiplier: float = 1.28
    illumination_scale_fraction: float = 0.55
    flattening_contrast_gain: float = 1.80
    lighting_deviation_scale_fraction: float = 0.10
    shadow_z_threshold: float = 0.75
    highlight_z_threshold: float = 0.75
    lighting_extreme_softness: float = 0.30
    quality_noise_scale_fraction: float = 0.025
    radial_bin_count: int = 24
    wrinkle_scale_fraction: float = 0.035
    damage_anomaly_scale_fraction: float = 0.12
    pattern_scale_fraction: float = 0.18
    colour_temperature: float = 18.0
    calibration_residual_gain: float = 1.0

    def __post_init__(self) -> None:
        if self.compute_device not in {"cuda", "auto", "cpu"}:
            raise ValueError("compute_device must be cuda, auto, or cpu.")
        if not 256 <= self.maximum_dimension <= 4096:
            raise ValueError("maximum_dimension must be between 256 and 4096.")
        if not 4 <= self.radial_bin_count <= 128:
            raise ValueError("radial_bin_count must be between 4 and 128.")
        if not 0.0 <= self.interior_background_weight <= 1.0:
            raise ValueError("interior_background_weight must be between 0 and 1.")
        if not 0.0 <= self.interior_foreground_noise_weight <= 1.0:
            raise ValueError(
                "interior_foreground_noise_weight must be between 0 and 1."
            )
        if not 0.0 <= self.interior_reference_texture_weight <= 1.0:
            raise ValueError(
                "interior_reference_texture_weight must be between 0 and 1."
            )
        positive = (
            self.interior_smoothing_fraction,
            self.boundary_width_fraction,
            self.split_neck_fraction,
            self.ellipse_radial_tolerance,
            self.disagreement_scale_fraction,
            self.contact_distance_multiplier,
            self.illumination_scale_fraction,
            self.flattening_contrast_gain,
            self.lighting_deviation_scale_fraction,
            self.shadow_z_threshold,
            self.highlight_z_threshold,
            self.lighting_extreme_softness,
            self.quality_noise_scale_fraction,
            self.wrinkle_scale_fraction,
            self.damage_anomaly_scale_fraction,
            self.pattern_scale_fraction,
            self.colour_temperature,
            self.calibration_residual_gain,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("Advanced analysis scale and gain settings must be positive.")


@dataclass(frozen=True, slots=True)
class ComputeBackendInfo:
    requested: str
    used: str
    device_name: str
    cuda_available: bool
    fallback_reason: str = ""
    elapsed_seconds: float = 0.0

    @property
    def summary(self) -> str:
        base = f"{self.used.upper()}: {self.device_name}"
        if self.fallback_reason:
            return f"{base} (fallback: {self.fallback_reason})"
        return base


@dataclass(frozen=True, slots=True)
class SeedDiagnosticSummary:
    """GPU-aggregated diagnostic and broad trait values for one proposal."""

    identifier: int
    pixel_count: int
    assignment_confidence: float
    image_quality_risk: float
    wrinkling_likelihood: float
    coat_damage_likelihood: float
    dominant_colour: str
    colour_proportions: tuple[float, ...]
    dominant_pattern: str
    pattern_proportions: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class AdvancedAnalysisLayers:
    """Advanced rasters aligned to the dish crop."""

    offset_x: int
    offset_y: int
    valid_mask: np.ndarray | GpuRaster
    rasters: dict[str, np.ndarray | GpuRaster]
    hue_rasters: dict[str, tuple[np.ndarray | GpuRaster, np.ndarray | GpuRaster]]
    backend: ComputeBackendInfo
    contact_pairs: tuple[tuple[int, int, float], ...]
    colour_class_names: tuple[str, ...] = COLOUR_CLASS_NAMES
    colour_probabilities: tuple[np.ndarray, ...] = ()
    pattern_class_names: tuple[str, ...] = PATTERN_CLASS_NAMES
    pattern_probabilities: tuple[np.ndarray, ...] = ()
    instance_summaries: tuple[SeedDiagnosticSummary, ...] = ()

    def rgba(self, mode: str) -> np.ndarray:
        alpha = np.uint8(np.asarray(self.valid_mask) > 0) * 255
        if mode.startswith("colour_probability:"):
            index = int(mode.partition(":")[2])
            return _heat_rgba(np.asarray(self.colour_probabilities[index]), alpha)
        if mode.startswith("pattern_probability:"):
            index = int(mode.partition(":")[2])
            return _heat_rgba(np.asarray(self.pattern_probabilities[index]), alpha)
        if mode in self.hue_rasters:
            hue, strength = (
                np.asarray(value) for value in self.hue_rasters[mode]
            )
            rgb = _hsv_full_saturation_to_rgb(hue, strength)
            return np.dstack((rgb, alpha))
        raster = np.asarray(self.rasters[mode])
        if mode in {
            "illumination_field",
            "flattened_grayscale",
            "reflectance_image",
            "radial_coordinate",
        }:
            return np.dstack((raster, raster, raster, alpha))
        return _heat_rgba(raster, alpha)

    @property
    def seed_colour_proportions(self) -> tuple[float, ...]:
        return _dominant_class_proportions(
            (summary.dominant_colour for summary in self.instance_summaries),
            self.colour_class_names,
        )

    @property
    def seed_pattern_proportions(self) -> tuple[float, ...]:
        return _dominant_class_proportions(
            (summary.dominant_pattern for summary in self.instance_summaries),
            self.pattern_class_names,
        )

    @property
    def mean_wrinkling_likelihood(self) -> float:
        return _summary_mean(
            summary.wrinkling_likelihood for summary in self.instance_summaries
        )

    @property
    def mean_coat_damage_likelihood(self) -> float:
        return _summary_mean(
            summary.coat_damage_likelihood for summary in self.instance_summaries
        )


def _local_lighting_evidence(
    torch,
    functional,
    luminance,
    valid,
    seed_diameter: float,
    settings: AdvancedAnalysisSettings,
):
    """Flatten luminance and classify locally unusual dark/bright areas."""

    illumination_sigma = max(
        3.0, seed_diameter * settings.illumination_scale_fraction
    )
    illumination_support = _gaussian(
        torch, functional, valid, illumination_sigma
    ).clamp_min(1e-4)
    illumination = _gaussian(
        torch, functional, luminance * valid, illumination_sigma
    ) / illumination_support
    illumination = illumination.clamp(0.0, 1.0) * valid

    # A log ratio removes multiplicative lighting while retaining a stable
    # mid-gray reference. This is intentionally a simple, interpretable
    # flattening rather than histogram equalization.
    log_ratio = torch.log(
        (luminance + 0.02) / (illumination + 0.02)
    ) * valid
    flattened = (
        0.5 + log_ratio * settings.flattening_contrast_gain
    ).clamp(0.0, 1.0) * valid

    deviation_sigma = max(
        0.8, seed_diameter * settings.lighting_deviation_scale_fraction
    )
    deviation_support = _gaussian(
        torch, functional, valid, deviation_sigma
    ).clamp_min(1e-4)
    local_deviation = _gaussian(
        torch, functional, torch.abs(log_ratio) * valid, deviation_sigma
    ) / deviation_support
    standardized = log_ratio / local_deviation.clamp_min(0.025)
    softness = max(0.05, settings.lighting_extreme_softness)
    shadow = torch.sigmoid(
        (-standardized - settings.shadow_z_threshold) / softness
    ) * valid
    highlight = torch.sigmoid(
        (standardized - settings.highlight_z_threshold) / softness
    ) * valid
    reflectance = _normalize(
        torch, luminance / (illumination + 0.08), valid
    ) * valid
    return illumination, flattened, shadow, highlight, reflectance


def local_lighting_evidence_tensors(
    source_tensor,
    valid_tensor,
    seed_diameter: float,
    settings: AdvancedAnalysisSettings,
):
    """Return full-resolution GPU tensors owned by the illumination node."""

    import torch
    import torch.nn.functional as functional

    source = source_tensor.to(dtype=torch.float32)
    valid_full = valid_tensor.to(device=source.device).bool()
    source_height, source_width = source.shape[-2:]
    scale = min(
        1.0,
        float(settings.maximum_dimension) / max(source_height, source_width),
    )
    height = max(8, round(source_height * scale))
    width = max(8, round(source_width * scale))
    rgb = source[:, (2, 1, 0)] / 255.0
    if (height, width) != (source_height, source_width):
        rgb = functional.interpolate(
            rgb, (height, width), mode="bilinear", align_corners=False
        )
        valid = functional.interpolate(
            valid_full.float(), (height, width), mode="nearest"
        )
    else:
        valid = valid_full.float()
    luminance = (
        0.2126 * rgb[:, 0:1]
        + 0.7152 * rgb[:, 1:2]
        + 0.0722 * rgb[:, 2:3]
    )
    products = _local_lighting_evidence(
        torch,
        functional,
        luminance,
        valid,
        seed_diameter * scale,
        settings,
    )
    if (height, width) == (source_height, source_width):
        return tuple(product * valid_full.float() for product in products)
    return tuple(
        functional.interpolate(
            product,
            (source_height, source_width),
            mode="bilinear",
            align_corners=False,
        )
        * valid_full.float()
        for product in products
    )


def sensor_noise_likelihood_tensor(
    source_tensor,
    valid_tensor,
    seed_diameter: float,
    settings: AdvancedAnalysisSettings,
):
    """Return the image-quality node's full-resolution sensor/noise tensor.

    This extraction lets another node consume the exact image-quality evidence
    without forcing it to recompute a private approximation from the source
    image. Work is performed at the advanced-analysis resolution and restored
    to the source raster on the same device.
    """

    import torch
    import torch.nn.functional as functional

    source = source_tensor.to(dtype=torch.float32)
    valid_full = valid_tensor.to(device=source.device).bool()
    source_height, source_width = source.shape[-2:]
    scale = min(
        1.0,
        float(settings.maximum_dimension) / max(source_height, source_width),
    )
    height = max(8, round(source_height * scale))
    width = max(8, round(source_width * scale))
    rgb = source[:, (2, 1, 0)] / 255.0
    if (height, width) != (source_height, source_width):
        rgb = functional.interpolate(
            rgb, (height, width), mode="bilinear", align_corners=False
        )
        valid = functional.interpolate(
            valid_full.float(), (height, width), mode="nearest"
        )
    else:
        valid = valid_full.float()
    luminance = (
        0.2126 * rgb[:, 0:1]
        + 0.7152 * rgb[:, 1:2]
        + 0.0722 * rgb[:, 2:3]
    )
    noise_sigma = max(
        0.8,
        seed_diameter * scale * settings.quality_noise_scale_fraction,
    )
    sensor_noise = _normalize(
        torch,
        torch.abs(
            luminance - _gaussian(torch, functional, luminance, noise_sigma)
        ),
        valid,
    ) * valid
    if (height, width) != (source_height, source_width):
        sensor_noise = functional.interpolate(
            sensor_noise,
            (source_height, source_width),
            mode="bilinear",
            align_corners=False,
        )
    return sensor_noise * valid_full.float()


def build_advanced_analysis_layers(
    crop: np.ndarray,
    valid_mask: np.ndarray,
    foreground_feature: object,
    foreground_threshold: float,
    foreground_mask: object,
    distance_transform: object,
    instance_labels: object,
    centers: np.ndarray,
    radii: np.ndarray,
    circle_candidates: np.ndarray,
    distance_candidates: np.ndarray,
    seed_diameter: float,
    *,
    offset_x: int,
    offset_y: int,
    full_image_shape: tuple[int, int],
    foreground_probability: object | None = None,
    foreground_noise_probability: object | None = None,
    reference_surface_probability: object | None = None,
    background_colour_probability: object | None = None,
    background_noise_probability: object | None = None,
    calibration_anchors: tuple[tuple[float, float, float], ...] = (),
    settings: AdvancedAnalysisSettings | None = None,
    previous: AdvancedAnalysisLayers | None = None,
    dirty_nodes: set[str] | frozenset[str] = frozenset(),
    enabled_nodes: set[str] | frozenset[str] | None = None,
    source_tensor=None,
    valid_tensor=None,
    sensor_noise_tensor=None,
    local_lighting_tensors=None,
    timing_recorder=None,
) -> AdvancedAnalysisLayers:
    """Build all fifteen diagnostic products on one PyTorch device."""

    import torch
    import torch.nn.functional as functional

    settings = settings or AdvancedAnalysisSettings()
    dirty = set(dirty_nodes)

    def enabled(node_id: str) -> bool:
        return enabled_nodes is None or node_id in enabled_nodes

    def active(node_id: str) -> bool:
        return enabled(node_id) and (previous is None or node_id in dirty)

    active_order = tuple(
        node_id for node_id in ADVANCED_NODE_MODES if active(node_id)
    )
    setup_owner = active_order[0] if active_order else None

    def start_timing(node_id: str, *, report_progress: bool = True):
        return (
            None
            if timing_recorder is None
            else timing_recorder.start(
                node_id, report_progress=report_progress
            )
        )

    def stop_timing(span) -> None:
        if span is not None:
            timing_recorder.stop(span)

    device, backend = _resolve_device(torch, settings)
    started = perf_counter()
    source_height, source_width = crop.shape[:2]
    scale = min(1.0, settings.maximum_dimension / max(source_height, source_width))
    height = max(8, round(source_height * scale))
    width = max(8, round(source_width * scale))
    diameter = max(4.0, seed_diameter * scale)
    setup_timing = (
        start_timing(setup_owner, report_progress=False)
        if setup_owner is not None
        else None
    )

    if source_tensor is None:
        rgb = torch.from_numpy(crop[:, :, ::-1].copy()).to(
            device=device, dtype=torch.float32
        )
        rgb = rgb.permute(2, 0, 1).unsqueeze(0) / 255.0
    else:
        rgb = source_tensor.to(device=device, dtype=torch.float32)[:, (2, 1, 0)] / 255.0
    rgb = functional.interpolate(rgb, (height, width), mode="bilinear", align_corners=False)
    valid = _resize_numpy(
        torch,
        functional,
        valid_mask if valid_tensor is None else GpuRaster(
            valid_tensor.to(torch.uint8) * 255,
            numpy_dtype=np.uint8,
            name="shared valid mask",
        ),
        device,
        height,
        width,
        "nearest",
    )
    valid = valid.clamp(0.0, 1.0)
    fg_mask = _resize_numpy(torch, functional, foreground_mask, device, height, width, "nearest")
    fg_feature = _resize_numpy(
        torch, functional, foreground_feature, device, height, width, "bilinear",
        normalize=False,
    )
    fg_probability = (
        None
        if foreground_probability is None
        else _resize_numpy(
            torch,
            functional,
            foreground_probability,
            device,
            height,
            width,
            "bilinear",
        ).clamp(0.0, 1.0)
    )
    distance = _resize_numpy(
        torch, functional, distance_transform, device, height, width, "bilinear",
        normalize=False,
    ) * scale
    labels = _resize_numpy(torch, functional, instance_labels, device, height, width, "nearest", normalize=False).long()

    def cached_scalar(name: str):
        if previous is None:
            return torch.zeros(
                (1, 1, height, width), device=device, dtype=torch.float32
            )
        return _resize_numpy(
            torch, functional, previous.rasters[name], device, height, width,
            "bilinear",
        )

    def cached_hue(name: str):
        if previous is None:
            return torch.zeros(
                (1, 1, height, width), device=device, dtype=torch.uint8
            )
        return _resize_numpy(
            torch, functional, previous.hue_rasters[name][0], device, height,
            width, "bilinear", normalize=False,
        ).clamp(0.0, 179.0).to(torch.uint8)

    def cached_probabilities(values):
        channels = [
            _resize_numpy(
                torch, functional, value, device, height, width, "bilinear"
            )
            for value in values
        ]
        return torch.cat(channels, dim=1)

    luminance = 0.2126 * rgb[:, 0:1] + 0.7152 * rgb[:, 1:2] + 0.0722 * rgb[:, 2:3]
    grad_x, grad_y = _sobel(torch, functional, luminance)
    gradient = torch.sqrt(grad_x.square() + grad_y.square() + 1e-8)
    gradient_n = _normalize(torch, gradient, valid)
    stop_timing(setup_timing)
    interior_consumers = {
        "boundary_normals",
        "proposal_disagreement",
        "assignment_confidence",
        "wrinkling",
        "coat_damage",
        "pattern_decomposition",
        "colour_probabilities",
    }
    if any(active(node_id) for node_id in interior_consumers):
        node_timing = None
        smoothing = max(0.8, diameter * settings.interior_smoothing_fraction)
        if fg_probability is None:
            smooth_feature = _gaussian(torch, functional, fg_feature, smoothing)
            feature_probability = torch.sigmoid(
                (smooth_feature - foreground_threshold)
                / max(2.0, foreground_threshold * 0.22)
            )
        else:
            # Preserve painted foreground-distribution evidence and the same soft evidence
            # shown in the foreground-probability viewer.
            feature_probability = _gaussian(
                torch, functional, fg_probability, smoothing
            )
        # Foreground colour, directional texture, material prototypes,
        # Background, and Other have already been reconciled by the
        # hierarchical material decision. Recombining the raw channels here
        # double-counted several sources and could undo explicit ambiguity.
        # This diagnostic now has one job: produce a seed-scale-smoothed view
        # of the authoritative resolved Seed mass.
        interior = feature_probability.clamp(0.0, 1.0) * valid
        stop_timing(node_timing)
    else:
        interior = cached_scalar("seed_interior_probability")

    if active("boundary_normals"):
        node_timing = start_timing("boundary_normals")
        boundary_radius = max(
            1, round(diameter * settings.boundary_width_fraction)
        )
        kernel = boundary_radius * 2 + 1
        outer = functional.max_pool2d(
            interior, kernel, stride=1, padding=boundary_radius
        )
        inner = -functional.max_pool2d(
            -interior, kernel, stride=1, padding=boundary_radius
        )
        probability_boundary = (outer - inner).clamp(0.0, 1.0)
        boundary = torch.sqrt(
            (probability_boundary * 0.70).square()
            + (gradient_n * 0.30).square()
        ) * valid
        interior_gx, interior_gy = _sobel(torch, functional, interior)
        normal_hue = _angle_hue(torch, interior_gx, interior_gy)
        stop_timing(node_timing)
    else:
        boundary = cached_scalar("boundary_magnitude")
        normal_hue = cached_hue("boundary_confidence")

    distance_n = (distance / max(diameter * 0.5, 1.0)).clamp(0.0, 1.5)
    if active("touching_split"):
        node_timing = start_timing("touching_split")
        dist_laplacian = _laplacian(torch, functional, distance_n)
        neck = torch.exp(-distance_n / settings.split_neck_fraction) * fg_mask
        saddle = _normalize(torch, torch.relu(-dist_laplacian), valid)
        split = (boundary * 0.45 + neck * saddle * 0.55) * fg_mask * valid
        stop_timing(node_timing)
    else:
        split = cached_scalar("touching_split_likelihood")

    anisotropy = None
    ellipse_hue = None
    if active("ellipse_likelihood") or active("pattern_decomposition"):
        structure_owner = (
            "ellipse_likelihood"
            if active("ellipse_likelihood")
            else "pattern_decomposition"
        )
        structure_timing = start_timing(
            structure_owner, report_progress=False
        )
        tensor_sigma = max(1.0, diameter * 0.08)
        jxx = _gaussian(torch, functional, grad_x.square(), tensor_sigma)
        jyy = _gaussian(torch, functional, grad_y.square(), tensor_sigma)
        jxy = _gaussian(torch, functional, grad_x * grad_y, tensor_sigma)
        anisotropy = torch.sqrt(
            (jxx - jyy).square() + 4.0 * jxy.square()
        ) / (jxx + jyy + 1e-6)
        ellipse_hue = _angle_hue(torch, jxx - jyy, 2.0 * jxy, axial=True)
        stop_timing(structure_timing)

    centers_array = np.asarray(centers, np.float32).reshape(-1, 2)
    radii_array = np.asarray(radii, np.float32).reshape(-1)
    radial, assigned_valid = _assigned_radial(
        torch, labels, centers_array, radii_array, scale, height, width, device
    )
    if active("ellipse_likelihood"):
        node_timing = start_timing("ellipse_likelihood")
        radial_support = torch.exp(
            -((radial - 1.0) / settings.ellipse_radial_tolerance).square()
        )
        ellipse = (
            boundary * (0.35 + 0.65 * anisotropy)
            * radial_support * assigned_valid * valid
        )
        stop_timing(node_timing)
    else:
        ellipse = (
            cached_scalar("ellipse_likelihood")
            if previous is None
            else _resize_numpy(
                torch,
                functional,
                previous.hue_rasters["ellipse_likelihood"][1],
                device,
                height,
                width,
                "bilinear",
            )
        )
        ellipse_hue = cached_hue("ellipse_likelihood")

    if active("proposal_disagreement"):
        node_timing = start_timing("proposal_disagreement")
        proposal_sigma = max(
            1.0, diameter * settings.disagreement_scale_fraction
        )
        circle_map = _candidate_map(
            torch, functional, circle_candidates, scale, height, width, device,
            proposal_sigma,
        )
        distance_map = _candidate_map(
            torch, functional, distance_candidates, scale, height, width,
            device, proposal_sigma,
        )
        evidence = torch.cat((circle_map, distance_map, interior, ellipse), dim=1)
        disagreement = torch.std(evidence, dim=1, keepdim=True) * 2.4 * valid
        stop_timing(node_timing)
    else:
        disagreement = cached_scalar("proposal_disagreement")

    if active("assignment_confidence"):
        node_timing = start_timing("assignment_confidence")
        contested = _contested_pixels(torch, labels) * fg_mask * valid
        assignment = interior * (
            1.0 - settings.assignment_boundary_penalty * boundary
        ) * (1.0 - contested * 0.75) * assigned_valid
        stop_timing(node_timing)
    else:
        contested = cached_scalar("contested_pixels")
        assignment = cached_scalar("instance_assignment_confidence")

    if active("contact_graph"):
        node_timing = start_timing("contact_graph")
        contact_pairs = _contact_pairs(
            torch, centers_array, radii_array,
            settings.contact_distance_multiplier, device,
        )
        contact_raster = torch.maximum(contested, boundary * contested)
        stop_timing(node_timing)
    else:
        contact_pairs = () if previous is None else previous.contact_pairs
        contact_raster = cached_scalar("contact_graph")

    maximum_rgb = torch.max(rgb, dim=1, keepdim=True).values
    minimum_rgb = torch.min(rgb, dim=1, keepdim=True).values
    saturation = (maximum_rgb - minimum_rgb) / (maximum_rgb + 1e-4)
    if active("illumination_decomposition"):
        node_timing = start_timing("illumination_decomposition")
        if local_lighting_tensors is None:
            (
                illumination,
                flattened,
                shadow,
                highlight,
                reflectance,
            ) = _local_lighting_evidence(
                torch,
                functional,
                luminance,
                valid,
                diameter,
                settings,
            )
        else:
            resized_lighting = []
            for tensor in local_lighting_tensors:
                values = tensor.to(device=device, dtype=torch.float32)
                if values.ndim == 2:
                    values = values[None, None]
                elif values.ndim == 3:
                    values = values[None]
                resized_lighting.append(
                    functional.interpolate(
                        values,
                        (height, width),
                        mode="bilinear",
                        align_corners=False,
                    ).clamp(0.0, 1.0)
                    * valid
                )
            (
                illumination,
                flattened,
                shadow,
                highlight,
                reflectance,
            ) = resized_lighting
        glare = (
            torch.sigmoid((maximum_rgb - 0.90) * 35.0)
            * torch.sigmoid((0.20 - saturation) * 18.0) * valid
        )
        stop_timing(node_timing)
    else:
        illumination = cached_scalar("illumination_field")
        flattened = cached_scalar("flattened_grayscale")
        shadow = cached_scalar("shadow_likelihood")
        highlight = cached_scalar("highlight_likelihood")
        reflectance = cached_scalar("reflectance_image")
        glare = cached_scalar("glare_likelihood")

    if active("image_quality"):
        node_timing = start_timing("image_quality")
        focus = _normalize(torch, gradient, valid)
        clipped = torch.sigmoid((maximum_rgb - 0.965) * 80.0) * valid
        underexposure = torch.sigmoid((0.10 - luminance) * 45.0) * valid
        if sensor_noise_tensor is None:
            noise_sigma = max(
                0.8, diameter * settings.quality_noise_scale_fraction
            )
            sensor_noise = _normalize(
                torch,
                torch.abs(
                    luminance
                    - _gaussian(torch, functional, luminance, noise_sigma)
                ),
                valid,
            )
        else:
            sensor_noise = sensor_noise_tensor.to(
                device=device, dtype=torch.float32
            )
            if sensor_noise.ndim == 2:
                sensor_noise = sensor_noise[None, None]
            elif sensor_noise.ndim == 3:
                sensor_noise = sensor_noise[None]
            sensor_noise = functional.interpolate(
                sensor_noise,
                (height, width),
                mode="bilinear",
                align_corners=False,
            ).clamp(0.0, 1.0) * valid
        quality_risk = torch.maximum(
            torch.maximum(1.0 - focus, clipped),
            torch.maximum(underexposure, sensor_noise * 0.7),
        ) * valid
        stop_timing(node_timing)
    else:
        focus = cached_scalar("focus_quality")
        clipped = cached_scalar("clipped_highlights")
        underexposure = cached_scalar("underexposure")
        sensor_noise = cached_scalar("sensor_noise")
        quality_risk = cached_scalar("image_quality_risk")

    radial_clamped = radial.clamp(0.0, 1.499)
    if active("radial_profile"):
        node_timing = start_timing("radial_profile")
        bin_count = settings.radial_bin_count
        bin_index = torch.clamp(
            (radial_clamped * (bin_count / 1.5)).long(), 0, bin_count - 1
        )
        weights = (assigned_valid * valid).reshape(-1)
        seed_count = max(1, len(centers_array))
        profile_index = labels.clamp(0, seed_count) * bin_count + bin_index
        profile_size = (seed_count + 1) * bin_count
        sums = torch.zeros(profile_size, device=device).scatter_add_(
            0, profile_index.reshape(-1), luminance.reshape(-1) * weights
        )
        counts = torch.zeros(profile_size, device=device).scatter_add_(
            0, profile_index.reshape(-1), weights
        )
        expected_profile = sums / (counts + 1e-6)
        radial_residual = _normalize(
            torch,
            torch.abs(luminance - expected_profile[profile_index]),
            valid,
        ) * assigned_valid
        stop_timing(node_timing)
    else:
        radial_residual = cached_scalar("radial_profile_residual")

    if active("wrinkling"):
        node_timing = start_timing("wrinkling")
        wrinkle_sigma = max(
            0.7, diameter * settings.wrinkle_scale_fraction
        )
        wrinkle_base = _gaussian(
            torch, functional, luminance, wrinkle_sigma
        )
        wrinkle = _normalize(
            torch,
            torch.abs(_laplacian(torch, functional, wrinkle_base)),
            valid,
        )
        wrinkle *= interior * (1.0 - boundary)
        stop_timing(node_timing)
    else:
        wrinkle = cached_scalar("wrinkling_likelihood")

    if active("coat_damage"):
        node_timing = start_timing("coat_damage")
        anomaly_sigma = max(
            1.2, diameter * settings.damage_anomaly_scale_fraction
        )
        local_colour = _gaussian(torch, functional, rgb, anomaly_sigma)
        colour_anomaly = torch.sqrt(
            torch.mean((rgb - local_colour).square(), dim=1, keepdim=True)
            + 1e-8
        )
        colour_anomaly = _normalize(torch, colour_anomaly, valid)
        damage = (
            torch.sqrt(
                colour_anomaly * torch.maximum(gradient_n, radial_residual)
            ) * interior * (1.0 - 0.65 * boundary)
        )
        stop_timing(node_timing)
    else:
        damage = cached_scalar("coat_damage_likelihood")

    if active("pattern_decomposition"):
        node_timing = start_timing("pattern_decomposition")
        pattern_sigma = max(
            1.5, diameter * settings.pattern_scale_fraction
        )
        fine_difference = torch.abs(
            luminance
            - _gaussian(
                torch, functional, luminance,
                max(0.8, pattern_sigma * 0.25),
            )
        )
        medium_difference = torch.abs(
            luminance
            - _gaussian(torch, functional, luminance, pattern_sigma)
        )
        coarse_difference = torch.abs(
            luminance
            - _gaussian(torch, functional, luminance, pattern_sigma * 2.2)
        )
        stripe = anisotropy * _normalize(torch, medium_difference, valid)
        pattern_scores = torch.cat(
            (
                (
                    1.0
                    - _normalize(
                        torch, fine_difference + medium_difference, valid
                    )
                ).clamp(0.0, 1.0),
                _normalize(torch, fine_difference, valid),
                _normalize(torch, medium_difference, valid),
                _normalize(torch, coarse_difference, valid),
                stripe,
                torch.abs(
                    _gaussian(
                        torch, functional, rgb[:, 0:1] - rgb[:, 2:3],
                        pattern_sigma * 1.5,
                    )
                ),
            ),
            dim=1,
        ) * interior
        pattern_probability = (
            torch.softmax(pattern_scores * 4.0, dim=1) * interior
        )
        pattern_confidence, pattern_class = torch.max(
            pattern_probability, dim=1, keepdim=True
        )
        pattern_hue = (
            pattern_class.float() * (179.0 / len(PATTERN_CLASS_NAMES))
        ).to(torch.uint8)
        stop_timing(node_timing)
    else:
        pattern_probability = (
            torch.zeros(
                (1, len(PATTERN_CLASS_NAMES), height, width),
                device=device,
                dtype=torch.float32,
            )
            if previous is None
            else cached_probabilities(previous.pattern_probabilities)
        )
        pattern_confidence = cached_scalar("pattern_confidence")
        pattern_hue = cached_hue("pattern_classes")

    if active("colour_probabilities"):
        node_timing = start_timing("colour_probabilities")
        prototypes = torch.tensor(
            (
                (0.92, 0.92, 0.88),
                (0.88, 0.76, 0.18),
                (0.24, 0.52, 0.22),
                (0.67, 0.18, 0.12),
                (0.40, 0.24, 0.12),
                (0.08, 0.08, 0.07),
            ),
            device=device,
            dtype=rgb.dtype,
        )
        colour_distance = torch.sum(
            (rgb[:, None] - prototypes[None, :, :, None, None]) ** 2,
            dim=2,
        )
        colour_probability = (
            torch.softmax(
                -colour_distance * settings.colour_temperature, dim=1
            ) * interior
        )
        colour_confidence, colour_class = torch.max(
            colour_probability, dim=1, keepdim=True
        )
        colour_hue = (
            colour_class.float() * (179.0 / len(COLOUR_CLASS_NAMES))
        ).to(torch.uint8)
        entropy = -torch.sum(
            colour_probability * torch.log(colour_probability + 1e-7),
            dim=1,
            keepdim=True,
        )
        entropy /= np.log(len(COLOUR_CLASS_NAMES))
        stop_timing(node_timing)
    else:
        colour_probability = (
            torch.zeros(
                (1, len(COLOUR_CLASS_NAMES), height, width),
                device=device,
                dtype=torch.float32,
            )
            if previous is None
            else cached_probabilities(previous.colour_probabilities)
        )
        colour_confidence = (
            cached_scalar("colour_confidence")
            if previous is None
            else _resize_numpy(
                torch,
                functional,
                previous.hue_rasters["colour_classes"][1],
                device,
                height,
                width,
                "bilinear",
            )
        )
        colour_hue = cached_hue("colour_classes")
        entropy = cached_scalar("colour_uncertainty")

    if active("calibration_residuals"):
        node_timing = start_timing("calibration_residuals")
        full_height, full_width = full_image_shape
        yy, xx = torch.meshgrid(
            torch.arange(height, device=device, dtype=torch.float32),
            torch.arange(width, device=device, dtype=torch.float32),
            indexing="ij",
        )
        global_x = xx / scale + offset_x
        global_y = yy / scale + offset_y
        if calibration_anchors:
            anchor_distances = []
            anchor_risks = []
            for anchor_x, anchor_y, anchor_risk in calibration_anchors:
                distance_anchor = torch.sqrt(
                    (global_x - anchor_x) ** 2
                    + (global_y - anchor_y) ** 2
                )
                anchor_distances.append(distance_anchor)
                anchor_risks.append(float(anchor_risk))
            stacked_distances = torch.stack(anchor_distances)
            nearest_distance, nearest_index = torch.min(
                stacked_distances, dim=0
            )
            risks = torch.tensor(anchor_risks, device=device)[nearest_index]
            calibration_risk = (
                risks
                + nearest_distance
                / max(np.hypot(full_width, full_height), 1.0) * 0.35
            )
        else:
            calibration_risk = (
                torch.ones((height, width), device=device) * 0.75
            )
        calibration_risk = (
            calibration_risk * settings.calibration_residual_gain
        ).clamp(0.0, 1.0)[None, None] * valid
        stop_timing(node_timing)
    else:
        calibration_risk = cached_scalar("calibration_residual_risk")

    rasters: dict[str, np.ndarray | GpuRaster] = {}
    hue_rasters: dict[
        str, tuple[np.ndarray | GpuRaster, np.ndarray | GpuRaster]
    ] = {}
    scalar_values = {
        "seed_interior_probability": interior,
        "boundary_magnitude": boundary,
        "touching_split_likelihood": split,
        "proposal_disagreement": disagreement,
        "instance_assignment_confidence": assignment,
        "contested_pixels": contested,
        "contact_graph": contact_raster,
        "illumination_field": illumination,
        "flattened_grayscale": flattened,
        "shadow_likelihood": shadow,
        "highlight_likelihood": highlight,
        "reflectance_image": reflectance,
        "glare_likelihood": glare,
        "image_quality_risk": quality_risk,
        "focus_quality": focus,
        "clipped_highlights": clipped,
        "underexposure": underexposure,
        "sensor_noise": sensor_noise,
        "radial_profile_residual": radial_residual,
        "radial_coordinate": radial_clamped / 1.5,
        "wrinkling_likelihood": wrinkle,
        "coat_damage_likelihood": damage,
        "pattern_confidence": pattern_confidence,
        "colour_uncertainty": entropy,
        "calibration_residual_risk": calibration_risk,
    }
    scalar_owners = {
        "seed_interior_probability": "boundary_normals",
        "boundary_magnitude": "boundary_normals",
        "touching_split_likelihood": "touching_split",
        "proposal_disagreement": "proposal_disagreement",
        "instance_assignment_confidence": "assignment_confidence",
        "contested_pixels": "assignment_confidence",
        "contact_graph": "contact_graph",
        "illumination_field": "illumination_decomposition",
        "flattened_grayscale": "illumination_decomposition",
        "shadow_likelihood": "illumination_decomposition",
        "highlight_likelihood": "illumination_decomposition",
        "reflectance_image": "illumination_decomposition",
        "glare_likelihood": "illumination_decomposition",
        "image_quality_risk": "image_quality",
        "focus_quality": "image_quality",
        "clipped_highlights": "image_quality",
        "underexposure": "image_quality",
        "sensor_noise": "image_quality",
        "radial_profile_residual": "radial_profile",
        "radial_coordinate": "radial_profile",
        "wrinkling_likelihood": "wrinkling",
        "coat_damage_likelihood": "coat_damage",
        "pattern_confidence": "pattern_decomposition",
        "colour_uncertainty": "colour_probabilities",
        "calibration_residual_risk": "calibration_residuals",
    }
    for name, tensor in scalar_values.items():
        owner = scalar_owners[name]
        if previous is not None and not active(owner):
            rasters[name] = previous.rasters[name]
        else:
            output_timing = (
                start_timing(owner, report_progress=False)
                if active(owner)
                else None
            )
            rasters[name] = _to_u8(
                torch, functional, tensor, source_height, source_width
            )
            stop_timing(output_timing)

    def hue_output(name, owner, hue, strength):
        if previous is not None and not active(owner):
            return previous.hue_rasters[name]
        output_timing = (
            start_timing(owner, report_progress=False)
            if active(owner)
            else None
        )
        result = (
            _to_u8(
                torch, functional, hue.float() / 179.0,
                source_height, source_width, scale_255=False, hue=True,
            ),
            strength,
        )
        stop_timing(output_timing)
        return result

    hue_rasters["boundary_confidence"] = hue_output(
        "boundary_confidence", "boundary_normals", normal_hue,
        rasters["boundary_magnitude"],
    )
    if previous is not None and not active("ellipse_likelihood"):
        ellipse_strength = previous.hue_rasters["ellipse_likelihood"][1]
    else:
        output_timing = (
            start_timing("ellipse_likelihood", report_progress=False)
            if active("ellipse_likelihood")
            else None
        )
        ellipse_strength = _to_u8(
            torch, functional, ellipse, source_height, source_width
        )
        stop_timing(output_timing)
    hue_rasters["ellipse_likelihood"] = hue_output(
        "ellipse_likelihood", "ellipse_likelihood", ellipse_hue,
        ellipse_strength,
    )
    hue_rasters["pattern_classes"] = hue_output(
        "pattern_classes", "pattern_decomposition", pattern_hue,
        rasters["pattern_confidence"],
    )
    if previous is not None and not active("colour_probabilities"):
        hue_rasters["colour_classes"] = previous.hue_rasters["colour_classes"]
        colour_outputs = previous.colour_probabilities
    else:
        output_timing = (
            start_timing("colour_probabilities", report_progress=False)
            if active("colour_probabilities")
            else None
        )
        hue_rasters["colour_classes"] = (
            _to_u8(
                torch, functional, colour_hue.float() / 179.0,
                source_height, source_width, scale_255=False, hue=True,
            ),
            _to_u8(
                torch, functional, colour_confidence,
                source_height, source_width,
            ),
        )
        colour_outputs = tuple(
            _to_u8(
                torch, functional, colour_probability[:, index:index + 1],
                source_height, source_width,
            )
            for index in range(len(COLOUR_CLASS_NAMES))
        )
        stop_timing(output_timing)
    if previous is not None and not active("pattern_decomposition"):
        pattern_outputs = previous.pattern_probabilities
    else:
        output_timing = (
            start_timing("pattern_decomposition", report_progress=False)
            if active("pattern_decomposition")
            else None
        )
        pattern_outputs = tuple(
            _to_u8(
                torch, functional, pattern_probability[:, index:index + 1],
                source_height, source_width,
            )
            for index in range(len(PATTERN_CLASS_NAMES))
        )
        stop_timing(output_timing)
    summary_owner = setup_owner or "boundary_normals"
    summary_timing = start_timing(summary_owner, report_progress=False)
    instance_summaries = _summarize_instances(
        torch,
        labels,
        assignment,
        quality_risk,
        wrinkle,
        damage,
        colour_probability,
        pattern_probability,
        len(centers_array),
    )
    valid_output = _to_u8(
        torch, functional, valid, source_height, source_width
    )
    stop_timing(summary_timing)
    elapsed = perf_counter() - started
    backend = ComputeBackendInfo(
        requested=backend.requested,
        used=backend.used,
        device_name=backend.device_name,
        cuda_available=backend.cuda_available,
        fallback_reason=backend.fallback_reason,
        elapsed_seconds=elapsed,
    )
    return AdvancedAnalysisLayers(
        offset_x=offset_x,
        offset_y=offset_y,
        valid_mask=valid_output,
        rasters=rasters,
        hue_rasters=hue_rasters,
        backend=backend,
        contact_pairs=contact_pairs,
        colour_probabilities=colour_outputs,
        pattern_probabilities=pattern_outputs,
        instance_summaries=instance_summaries,
    )


def _resolve_device(torch, settings: AdvancedAnalysisSettings):
    cuda_available = bool(torch.cuda.is_available())
    requested = settings.compute_device
    if requested == "cpu":
        device = torch.device("cpu")
        return device, ComputeBackendInfo(requested, "cpu", "CPU", cuda_available)
    if cuda_available:
        device = torch.device("cuda")
        return device, ComputeBackendInfo(
            requested, "cuda", torch.cuda.get_device_name(device), True
        )
    if not settings.allow_cpu_fallback:
        raise RuntimeError(
            "CUDA analysis was requested but PyTorch cannot access a CUDA device."
        )
    return torch.device("cpu"), ComputeBackendInfo(
        requested, "cpu", "CPU", False, "CUDA unavailable"
    )


def _resize_numpy(torch, functional, array, device, height, width, mode, *, normalize=True):
    if isinstance(array, GpuRaster):
        tensor = array.gpu_tensor(device=device, dtype=torch.float32)
        if tensor.ndim == 2:
            tensor = tensor[None, None]
        elif tensor.ndim == 3:
            tensor = tensor[None]
        if normalize and np.issubdtype(array.dtype, np.integer):
            tensor = tensor / 255.0
    else:
        tensor = torch.from_numpy(np.asarray(array).copy()).to(
            device=device, dtype=torch.float32
        )
        tensor = tensor[None, None]
        if normalize and float(tensor.max()) > 1.5:
            tensor = tensor / 255.0
    kwargs = {} if mode == "nearest" else {"align_corners": False}
    return functional.interpolate(tensor, (height, width), mode=mode, **kwargs)


def _gaussian(torch, functional, tensor, sigma):
    sigma = float(max(0.35, sigma))
    radius = min(96, max(1, round(sigma * 3.0)))
    positions = torch.arange(-radius, radius + 1, device=tensor.device, dtype=tensor.dtype)
    kernel = torch.exp(-0.5 * (positions / sigma) ** 2)
    kernel /= torch.sum(kernel)
    channels = tensor.shape[1]
    horizontal = kernel.reshape(1, 1, 1, -1).repeat(channels, 1, 1, 1)
    vertical = kernel.reshape(1, 1, -1, 1).repeat(channels, 1, 1, 1)
    result = functional.conv2d(tensor, horizontal, padding=(0, radius), groups=channels)
    return functional.conv2d(result, vertical, padding=(radius, 0), groups=channels)


def _sobel(torch, functional, tensor):
    kernel_x = torch.tensor(
        ((-1.0, 0.0, 1.0), (-2.0, 0.0, 2.0), (-1.0, 0.0, 1.0)),
        device=tensor.device,
        dtype=tensor.dtype,
    )[None, None] / 8.0
    kernel_y = kernel_x.transpose(-1, -2)
    return (
        functional.conv2d(tensor, kernel_x, padding=1),
        functional.conv2d(tensor, kernel_y, padding=1),
    )


def _laplacian(torch, functional, tensor):
    kernel = torch.tensor(
        ((0.0, 1.0, 0.0), (1.0, -4.0, 1.0), (0.0, 1.0, 0.0)),
        device=tensor.device,
        dtype=tensor.dtype,
    )[None, None]
    return functional.conv2d(tensor, kernel, padding=1)


def _normalize(torch, tensor, valid):
    selected = tensor[valid.expand_as(tensor) > 0.5]
    if selected.numel() < 16:
        return torch.zeros_like(tensor)
    high = torch.quantile(selected, 0.99).clamp_min(1e-6)
    return (tensor / high).clamp(0.0, 1.0)


def _angle_hue(torch, x_component, y_component, *, axial=False):
    angle = torch.atan2(y_component, x_component)
    if axial:
        angle = torch.remainder(angle, torch.pi)
        return torch.round(angle / torch.pi * 179.0).to(torch.uint8)
    angle = torch.remainder(angle, torch.pi * 2.0)
    return torch.round(angle / (torch.pi * 2.0) * 179.0).to(torch.uint8)


def _assigned_radial(torch, labels, centers, radii, scale, height, width, device):
    if len(centers) == 0:
        zeros = torch.zeros((1, 1, height, width), device=device)
        return zeros, zeros
    scaled_centers = torch.from_numpy(centers * scale).to(device=device)
    scaled_radii = torch.from_numpy(np.maximum(radii * scale, 1.0)).to(device=device)
    scaled_centers = torch.cat((torch.zeros((1, 2), device=device), scaled_centers), dim=0)
    scaled_radii = torch.cat((torch.ones(1, device=device), scaled_radii), dim=0)
    safe_labels = labels.clamp(0, len(centers))
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    assigned_centers = scaled_centers[safe_labels[0, 0]]
    assigned_radii = scaled_radii[safe_labels[0, 0]]
    radial = torch.sqrt(
        (xx - assigned_centers[:, :, 0]).square()
        + (yy - assigned_centers[:, :, 1]).square()
    ) / assigned_radii
    return radial[None, None], (safe_labels > 0).float()


def _candidate_map(torch, functional, geometry, scale, height, width, device, sigma):
    impulse = torch.zeros((1, 1, height, width), device=device)
    values = np.asarray(geometry, np.float32).reshape(-1, 3)
    if len(values):
        x = torch.from_numpy(np.clip(np.rint(values[:, 0] * scale), 0, width - 1).astype(np.int64)).to(device)
        y = torch.from_numpy(np.clip(np.rint(values[:, 1] * scale), 0, height - 1).astype(np.int64)).to(device)
        impulse[0, 0].index_put_((y, x), torch.ones_like(x, dtype=torch.float32), accumulate=True)
    return _normalize(torch, _gaussian(torch, functional, impulse, sigma), torch.ones_like(impulse))


def _contested_pixels(torch, labels):
    contested = torch.zeros_like(labels, dtype=torch.bool)
    for shift_y, shift_x in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        shifted = torch.roll(labels, (shift_y, shift_x), dims=(-2, -1))
        contested |= (labels > 0) & (shifted > 0) & (labels != shifted)
    return contested.float()


def _contact_pairs(torch, centers, radii, multiplier, device):
    if len(centers) < 2:
        return ()
    center_tensor = torch.from_numpy(centers).to(device=device)
    radius_tensor = torch.from_numpy(radii).to(device=device)
    distances = torch.cdist(center_tensor, center_tensor)
    limits = (radius_tensor[:, None] + radius_tensor[None, :]) * multiplier
    candidate = torch.triu((distances > 0) & (distances <= limits), diagonal=1)
    indices = torch.nonzero(candidate, as_tuple=False)
    if indices.numel() == 0:
        return ()
    confidence = (1.0 - distances[indices[:, 0], indices[:, 1]] / limits[indices[:, 0], indices[:, 1]]).clamp(0.0, 1.0)
    indices_cpu = indices[:4000].detach().cpu().numpy()
    confidence_cpu = confidence[:4000].detach().cpu().numpy()
    return tuple(
        (int(pair[0]) + 1, int(pair[1]) + 1, float(score))
        for pair, score in zip(indices_cpu, confidence_cpu, strict=True)
    )


def _summarize_instances(
    torch,
    labels,
    assignment,
    quality,
    wrinkle,
    damage,
    colour_probability,
    pattern_probability,
    seed_count,
):
    if seed_count <= 0:
        return ()
    label_values = labels.reshape(-1).clamp(0, seed_count)
    counts = torch.bincount(label_values, minlength=seed_count + 1).float()

    def means(values):
        sums = torch.zeros(seed_count + 1, device=labels.device).scatter_add_(
            0, label_values, values.reshape(-1)
        )
        return sums / counts.clamp_min(1.0)

    assignment_mean = means(assignment)
    quality_mean = means(quality)
    wrinkle_mean = means(wrinkle)
    damage_mean = means(damage)

    def class_proportions(probability):
        class_count = probability.shape[1]
        rows = []
        for class_index in range(class_count):
            rows.append(means(probability[:, class_index:class_index + 1]))
        values = torch.stack(rows, dim=1)
        return values / values.sum(dim=1, keepdim=True).clamp_min(1e-6)

    colour_values = class_proportions(colour_probability)
    pattern_values = class_proportions(pattern_probability)
    packed = torch.cat(
        (
            counts[:, None],
            assignment_mean[:, None],
            quality_mean[:, None],
            wrinkle_mean[:, None],
            damage_mean[:, None],
            colour_values,
            pattern_values,
        ),
        dim=1,
    )[1:].detach().cpu().numpy()
    summaries = []
    colour_start = 5
    pattern_start = colour_start + len(COLOUR_CLASS_NAMES)
    for index, row in enumerate(packed, start=1):
        colour = tuple(float(value) for value in row[colour_start:pattern_start])
        pattern = tuple(float(value) for value in row[pattern_start:])
        summaries.append(
            SeedDiagnosticSummary(
                identifier=index,
                pixel_count=int(round(float(row[0]))),
                assignment_confidence=float(row[1]),
                image_quality_risk=float(row[2]),
                wrinkling_likelihood=float(row[3]),
                coat_damage_likelihood=float(row[4]),
                dominant_colour=COLOUR_CLASS_NAMES[int(np.argmax(colour))],
                colour_proportions=colour,
                dominant_pattern=PATTERN_CLASS_NAMES[int(np.argmax(pattern))],
                pattern_proportions=pattern,
            )
        )
    return tuple(summaries)


def _to_u8(torch, functional, tensor, height, width, *, scale_255=True, hue=False):
    resized = functional.interpolate(tensor.float(), (height, width), mode="bilinear", align_corners=False)
    values = resized[0, 0].clamp(0.0, 1.0)
    multiplier = 179.0 if hue else (255.0 if scale_255 else 179.0)
    return GpuRaster(
        torch.round(values * multiplier).to(torch.uint8)[None, None],
        numpy_dtype=np.uint8,
        name="advanced overlay",
    )


def _heat_rgba(value: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    normalized = value.astype(np.float32) / 255.0
    red = np.uint8(np.clip(normalized * 2.0, 0.0, 1.0) * 255.0)
    green = np.uint8(np.clip(1.0 - np.abs(normalized * 2.0 - 1.0), 0.0, 1.0) * 255.0)
    blue = np.uint8(np.clip((1.0 - normalized) * 1.7, 0.0, 1.0) * 255.0)
    return np.dstack((red, green, blue, alpha))


def _hsv_full_saturation_to_rgb(hue: np.ndarray, value: np.ndarray) -> np.ndarray:
    """Convert downloaded overlay hue/value arrays for Qt rendering."""

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


def _dominant_class_proportions(values, names):
    counts = {name: 0 for name in names}
    total = 0
    for value in values:
        counts[value] += 1
        total += 1
    if total == 0:
        return tuple(0.0 for _ in names)
    return tuple(counts[name] / total for name in names)


def _summary_mean(values) -> float:
    sequence = tuple(values)
    return float(sum(sequence) / len(sequence)) if sequence else 0.0
