"""PyTorch implementations of Seed Fiddle's diagnostic layers."""

from __future__ import annotations

import colorsys
from dataclasses import dataclass, field

import numpy as np

from seedvision.cuda.ops import (
    CudaContext,
    GpuRaster,
    bgr_to_lab,
    bilinear_sample,
    gaussian_blur,
    gradient_magnitude,
    image_to_tensor,
    lab_colour_distribution,
    lab_colour_frequency_distribution,
    oriented_connected_components,
)
from seedvision.cuda.material import hierarchical_material_evidence


@dataclass(slots=True)
class EdgeGradientProducts:
    """Continuous shared gradient field plus lazy display rasters."""

    source: object
    lab: object
    valid: object
    strength: object
    normal_x: object
    normal_y: object
    tangent_x: object
    tangent_y: object
    directed_hue_float: object
    undirected_hue_float: object
    strength_raster: GpuRaster
    directed_hue_raster: GpuRaster
    undirected_hue_raster: GpuRaster

    def __iter__(self):
        yield self.strength_raster
        yield self.directed_hue_raster
        yield self.undirected_hue_raster


@dataclass(slots=True)
class ReferenceEdgeProducts:
    """Prototype diagnostics and true-edge-supported boundary evidence."""

    physical_prototype_compatibility: GpuRaster
    nonphysical_prototype_compatibility: GpuRaster
    net_prototype_compatibility: GpuRaster
    edge_supported_physical_compatibility: GpuRaster
    edge_supported_nonphysical_compatibility: GpuRaster
    reference_edge_probability: GpuRaster
    conservative_net_physical_edge_evidence: GpuRaster
    net_prototype_field: GpuRaster
    reference_edge_field: GpuRaster
    conservative_net_edge_field: GpuRaster
    interior_direction_x: GpuRaster
    interior_direction_y: GpuRaster
    interior_direction_confidence: GpuRaster
    subtraction_weight: float
    physical_sample_count: int
    non_edge_sample_count: int

    # Source-compatible aliases for compact callers. These remain raw
    # prototype diagnostics; production consumers use the explicitly named
    # edge-supported fields above.
    @property
    def physical_probability(self) -> GpuRaster:
        return self.physical_prototype_compatibility

    @property
    def non_edge_probability(self) -> GpuRaster:
        return self.nonphysical_prototype_compatibility

    @property
    def net_probability(self) -> GpuRaster:
        return self.net_prototype_compatibility


@dataclass(slots=True)
class ReferenceTextureProducts:
    """Multi-prototype material and boundary evidence learned per image."""

    seed_surface_probability: GpuRaster | None
    background_probability: GpuRaster | None
    other_probability: GpuRaster | None
    physical_edge_probability: GpuRaster
    non_edge_probability: GpuRaster
    physical_edge_field: GpuRaster
    non_edge_field: GpuRaster
    physical_edge_interior_direction_x: GpuRaster
    physical_edge_interior_direction_y: GpuRaster
    physical_edge_interior_direction_confidence: GpuRaster
    profile: object
    foreground_sample_count: int
    background_sample_count: int
    other_sample_count: int
    physical_sample_count: int
    non_edge_sample_count: int


@dataclass(slots=True)
class ReferenceSeedTraitProducts:
    """Independent coat-pattern and condition probabilities for seed material."""

    coat_probabilities: tuple[tuple[str, GpuRaster], ...]
    condition_probabilities: tuple[tuple[str, GpuRaster], ...]
    profile: object


@dataclass(slots=True)
class _FeaturePrototypeBank:
    """GPU-resident diagonal feature prototypes for one painted class."""

    class_name: str
    centres: object
    scales: object
    weights: object
    sample_counts: object
    positions_yx: object
    sample_count: int
    # Number of independently balanced image sources represented by this
    # compact bank.  A current-image bank is one source regardless of how many
    # painted pixels or medoids it contains.
    source_count: int = 1


@dataclass(slots=True)
class SurfaceGradientProducts:
    """Maximum one-sided L* slopes retained at a bounded GPU working size."""

    source_height: int
    source_width: int
    work_scale: float
    valid: object
    lightening_magnitude: object
    darkening_magnitude: object
    lightening_strength: object
    darkening_strength: object
    lightening_hue_float: object
    darkening_hue_float: object
    lightening_strength_raster: GpuRaster
    darkening_strength_raster: GpuRaster
    lightening_hue_raster: GpuRaster
    darkening_hue_raster: GpuRaster


@dataclass(slots=True)
class FrequencyNoiseMaskProducts:
    """Fine/medium/coarse local RMS masks for darkness and Lab chroma."""

    band_scales_px: tuple[float, float, float]
    darkness_masks: tuple[GpuRaster, GpuRaster, GpuRaster]
    colour_masks: tuple[GpuRaster, GpuRaster, GpuRaster]


@dataclass(slots=True)
class BoundaryTraceProducts:
    final_likelihood: GpuRaster
    selected_radius: GpuRaster
    ridges: GpuRaster
    trace_labels: GpuRaster
    trace_continuity: GpuRaster
    gap_confidence: GpuRaster
    radius_ratio_hue: GpuRaster
    radius_confidence: GpuRaster
    circle_confidence: GpuRaster
    ellipse_confidence: GpuRaster
    fit_residual: GpuRaster
    centre_votes: GpuRaster
    oval_centre_probability: GpuRaster
    semantic_sides: GpuRaster
    rejection_hue: GpuRaster
    rejection_strength: GpuRaster
    geometry: object | None = None
    ridge_state: object | None = None
    trace_state: object | None = None


@dataclass(slots=True)
class GpuRidgeState:
    """Continuous ridge tensors retained for downstream GPU stages."""

    signature: tuple[object, ...]
    scale: float
    height: int
    width: int
    edge: object
    valid: object
    normal_x: object
    normal_y: object
    tangent_x: object
    tangent_y: object
    lightness: object
    yy: object
    xx: object
    nms: object
    weak: object
    accepted: object


@dataclass(slots=True)
class GpuTraceState:
    """Oriented trace tensors retained for boundary confirmation."""

    signature: tuple[object, ...]
    trace_seed: object
    trace_labels: object
    continuity: object
    gap_confidence: object
    trace_confidence: object


@dataclass(slots=True)
class GpuBoundaryGeometry:
    """Compact fit geometry retained on-device until its overlay is selected."""

    vote_centres_xy: object
    vote_confidence: object
    instance_centres_xy: object
    ellipse_axes_xy: object
    ellipse_angle_radians: object
    ellipse_confidence: object
    oval_centres_xy: object
    oval_axes_xy: object
    oval_angle_radians: object
    oval_confidence: object
    _host_cache: dict[str, np.ndarray] | None = field(
        default=None, init=False, repr=False
    )
    download_count: int = field(default=0, init=False)

    def materialize(self) -> dict[str, np.ndarray]:
        if self._host_cache is None:
            self._host_cache = {
                "vote_centres_xy": self.vote_centres_xy.detach().cpu().numpy(),
                "vote_confidence": self.vote_confidence.detach().cpu().numpy(),
                "instance_centres_xy": self.instance_centres_xy.detach().cpu().numpy(),
                "ellipse_axes_xy": self.ellipse_axes_xy.detach().cpu().numpy(),
                "ellipse_angle_radians": self.ellipse_angle_radians.detach().cpu().numpy(),
                "ellipse_confidence": self.ellipse_confidence.detach().cpu().numpy(),
                "oval_centres_xy": self.oval_centres_xy.detach().cpu().numpy(),
                "oval_axes_xy": self.oval_axes_xy.detach().cpu().numpy(),
                "oval_angle_radians": self.oval_angle_radians.detach().cpu().numpy(),
                "oval_confidence": self.oval_confidence.detach().cpu().numpy(),
            }
            self.download_count += 1
        return self._host_cache

    def release_host_cache(self) -> None:
        """Drop compact CPU geometry mirrors when their owning cache is evicted."""

        self._host_cache = None


def _lazy_u8(tensor, name: str) -> GpuRaster:
    import torch

    values = tensor
    if values.dtype != torch.uint8:
        values = torch.round(values).clamp(0, 255).to(torch.uint8)
    if values.ndim == 2:
        values = values[None, None]
    elif values.ndim == 3:
        values = values[None]
    return GpuRaster(values, numpy_dtype=np.uint8, name=name)


def _lazy_float(tensor, name: str) -> GpuRaster:
    values = tensor
    if values.ndim == 2:
        values = values[None, None]
    elif values.ndim == 3:
        values = values[None]
    return GpuRaster(values, numpy_dtype=np.float32, name=name)


def hue_only_rgb(source_tensor) -> GpuRaster:
    """Encode only source hue at fixed 62% value; achromatic pixels stay gray."""

    import torch

    bgr = source_tensor.float().clamp(0.0, 255.0) / 255.0
    blue, green, red = bgr[:, 0:1], bgr[:, 1:2], bgr[:, 2:3]
    maximum = torch.maximum(torch.maximum(red, green), blue)
    minimum = torch.minimum(torch.minimum(red, green), blue)
    delta = maximum - minimum
    safe_delta = delta.clamp_min(1e-6)
    hue6 = torch.where(
        maximum == red,
        torch.remainder((green - blue) / safe_delta, 6.0),
        torch.where(
            maximum == green,
            (blue - red) / safe_delta + 2.0,
            (red - green) / safe_delta + 4.0,
        ),
    )
    hue6 = torch.remainder(hue6, 6.0)
    chroma = torch.full_like(hue6, 0.62)
    x = chroma * (1.0 - torch.abs(torch.remainder(hue6, 2.0) - 1.0))
    zero = torch.zeros_like(chroma)
    sector = torch.floor(hue6).long().clamp(0, 5)
    red_hue = torch.where(
        sector == 0, chroma,
        torch.where(sector == 1, x, torch.where(sector == 5, chroma, zero)),
    )
    green_hue = torch.where(
        sector == 0, x,
        torch.where(
            sector == 1, chroma,
            torch.where(sector == 2, chroma, torch.where(sector == 3, x, zero)),
        ),
    )
    blue_hue = torch.where(
        sector == 2, x,
        torch.where(
            sector == 3, chroma,
            torch.where(sector == 4, chroma, torch.where(sector == 5, x, zero)),
        ),
    )
    achromatic = delta <= (1.0 / 255.0)
    neutral = torch.full_like(chroma, 0.62)
    rgb = torch.cat(
        (
            torch.where(achromatic, neutral, red_hue),
            torch.where(achromatic, neutral, green_hue),
            torch.where(achromatic, neutral, blue_hue),
        ),
        dim=1,
    )
    return _lazy_u8(rgb * 255.0, "hue-only fixed-darkness RGB")


def stationary_wavelet_decomposition(
    source_tensor,
    level_count: int = 4,
) -> tuple[tuple[GpuRaster, ...], GpuRaster]:
    """Return an exact-reconstruction undecimated B3-spline à trous pyramid."""

    import torch
    import torch.nn.functional as functional

    levels = max(1, min(4, int(level_count)))
    current = source_tensor.float()
    base = torch.as_tensor(
        (1.0, 4.0, 6.0, 4.0, 1.0),
        device=current.device,
        dtype=current.dtype,
    ) / 16.0
    details: list[GpuRaster] = []
    for level in range(4):
        if level < levels:
            dilation = 2**level
            size = 1 + 4 * dilation
            kernel_1d = torch.zeros(size, device=current.device, dtype=current.dtype)
            kernel_1d[::dilation] = base
            kernel_2d = torch.outer(kernel_1d, kernel_1d)
            kernel = kernel_2d[None, None].repeat(current.shape[1], 1, 1, 1)
            padding = size // 2
            padded = functional.pad(
                current,
                (padding, padding, padding, padding),
                mode="reflect",
            )
            smooth = functional.conv2d(
                padded, kernel, groups=current.shape[1]
            )
            detail = current - smooth
            current = smooth
        else:
            detail = torch.zeros_like(current)
        details.append(
            _lazy_float(detail, f"stationary wavelet detail level {level + 1}")
        )
    return tuple(details), _lazy_float(current, "stationary wavelet residual")


def _lazy_int(tensor, name: str) -> GpuRaster:
    import torch

    values = tensor.to(torch.int32)
    if values.ndim == 2:
        values = values[None, None]
    elif values.ndim == 3:
        values = values[None]
    return GpuRaster(values, numpy_dtype=np.int32, name=name)


def _raster_tensor(value, context, *, normalized: bool = False):
    import torch

    tensor = image_to_tensor(value, context)
    if normalized:
        tensor = tensor / 255.0
    return tensor.to(torch.float32)


def _sample_count(samples) -> int:
    if samples is None:
        return 0
    try:
        import torch

        if torch.is_tensor(samples):
            return int(samples.numel() // 3)
    except ImportError:
        pass
    return int(np.asarray(samples).size // 3)


def _bgr_samples_to_lab(samples, context):
    import torch

    if torch.is_tensor(samples):
        values = samples.to(
            device=context.device, dtype=torch.float32
        ).reshape(-1, 3)
        nchw = values.T[None, :, :, None]
    else:
        nchw = image_to_tensor(
            np.asarray(samples, np.uint8).reshape(-1, 1, 3), context
        )
    return bgr_to_lab(nchw)[0, :, :, 0].T


def build_cuda_analysis_layers(
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
    foreground_reference_source_mask=None,
    background_exclusion_mask: np.ndarray | None = None,
    foreground_exclusion_mask: np.ndarray | None = None,
    seed_instance_annotations: np.ndarray | None = None,
    seed_instance_traits: tuple[object, ...] = (),
    seed_trait_species: str = "",
    seed_trait_coat_patterns: tuple[str, ...] = (),
    seed_trait_conditions: tuple[str, ...] = (),
    background_reference_samples: np.ndarray | None = None,
    background_reference_sample_count: int = 0,
    background_prior_lab: tuple[float, float, float] | None = None,
    background_prior_samples_lab=None,
    background_prior_source_mask=None,
    background_colour_enabled: bool = True,
    background_noise_enabled: bool = True,
    foreground_noise_enabled: bool = True,
    reference_edge_probability_enabled: bool = True,
    reference_edge_ridges_enabled: bool = True,
    reference_texture_prototypes_enabled: bool = True,
    reference_seed_traits_enabled: bool = True,
    hue_only_enabled: bool = True,
    wavelet_decomposition_enabled: bool = True,
    surface_darkness_gradients_enabled: bool = True,
    lightening_gradient_ceiling_enabled: bool = True,
    darkening_gradient_ceiling_enabled: bool = True,
    frequency_noise_masks_enabled: bool = True,
    instance_masks_enabled: bool = True,
    seed_edge_curves_enabled: bool = True,
    foreground_probability=None,
    automatic_foreground_probability=None,
    reviewed_foreground_probability=None,
    foreground_automatic_evidence_authority: float = 1.0,
    material_valid_mask=None,
    material_proposal_valid_mask=None,
    instance_nonseed_probability=None,
    foreground_colour_profile=None,
    species_library=None,
    seed_shape_model=None,
    seed_measurement_summary=None,
    material_evidence_enabled: bool = True,
    surrounding_noise_source_tensor=None,
    surrounding_noise_valid_tensor=None,
    surrounding_noise_offset_x: int = 0,
    surrounding_noise_offset_y: int = 0,
    despeckled_flattened_tensor=None,
    settings=None,
    cache_values: dict[str, object] | None = None,
    dirty_nodes: set[str] | frozenset[str] = frozenset(),
    cuda_context: CudaContext | None = None,
    timing_recorder=None,
):
    """Build cached overlay products while all raster math stays on the GPU."""

    from seedvision.visualization.layers import AnalysisLayers, AnalysisLayerSettings

    settings = settings or AnalysisLayerSettings()
    context = cuda_context or CudaContext.resolve()
    values = {} if cache_values is None else cache_values
    dirty = set(dirty_nodes)
    if crop.ndim != 3 or crop.shape[2] < 3:
        raise ValueError("A BGR colour crop is required.")
    if valid_mask.shape != crop.shape[:2]:
        raise ValueError("valid_mask must match the crop dimensions.")
    centers = np.asarray(centers, np.float32).reshape(-1, 2)
    radii = np.asarray(radii, np.float32).reshape(-1)
    if len(centers) != len(radii):
        raise ValueError("Each seed center requires one radius.")
    if seed_instance_annotations is not None:
        seed_instance_annotations = np.asarray(seed_instance_annotations)
        if seed_instance_annotations.shape != crop.shape[:2]:
            raise ValueError("Seed instance annotations must match the crop dimensions.")

    gpu_inputs = values.get("layer.gpu_inputs")
    if gpu_inputs is None:
        source_tensor = image_to_tensor(crop, context)
        lab_tensor = bgr_to_lab(source_tensor)
        valid_tensor = image_to_tensor(valid_mask, context) > 0
        gpu_inputs = source_tensor, lab_tensor, valid_tensor
        values["layer.gpu_inputs"] = gpu_inputs
    else:
        source_tensor, lab_tensor, valid_tensor = gpu_inputs
    hue_dirty = "hue_only" in dirty or "layer.hue_only" not in values
    if hue_dirty:
        hue_timing = (
            None
            if timing_recorder is None
            else timing_recorder.start("hue_only")
        )
        values["layer.hue_only"] = (
            hue_only_rgb(source_tensor)
            if hue_only_enabled
            else _lazy_u8(
                source_tensor[:, :3] * 0.0, "disabled hue-only visualization"
            )
        )
        if hue_timing is not None:
            timing_recorder.stop(hue_timing)
    hue_only = values["layer.hue_only"]

    wavelet_signature = int(settings.wavelet_level_count)
    wavelet_dirty = (
        "wavelet_decomposition" in dirty
        or "layer.wavelet_decomposition" not in values
        or values.get("layer.wavelet_signature") != wavelet_signature
    )
    if wavelet_dirty:
        wavelet_timing = (
            None
            if timing_recorder is None
            else timing_recorder.start("wavelet_decomposition")
        )
        if wavelet_decomposition_enabled:
            wavelet_details, wavelet_residual = stationary_wavelet_decomposition(
                source_tensor, wavelet_signature
            )
        else:
            zero = _lazy_float(
                source_tensor[:, :3] * 0.0, "disabled wavelet detail"
            )
            wavelet_details = (zero, zero, zero, zero)
            wavelet_residual = _lazy_float(
                source_tensor[:, :3] * 0.0, "disabled wavelet residual"
            )
        values["layer.wavelet_decomposition"] = (
            wavelet_details,
            wavelet_residual,
        )
        values["layer.wavelet_signature"] = wavelet_signature
        if wavelet_timing is not None:
            timing_recorder.stop(wavelet_timing)
    else:
        wavelet_details, wavelet_residual = values["layer.wavelet_decomposition"]
    painted_reference_dirty = "project" in dirty or "reference_layers" in dirty
    background_dirty = (
        painted_reference_dirty
        or "background_likelihood" in dirty
        or "layer.background" not in values
    )
    if background_dirty and background_colour_enabled:
        background_timing = (
            None
            if timing_recorder is None
            else timing_recorder.start("background_likelihood")
        )
        background_result = background_colour_likelihood(
            crop,
            valid_mask,
            background_reference_points=background_reference_points,
            foreground_reference_points=foreground_reference_points,
            background_reference_mask=background_reference_mask,
            foreground_reference_mask=foreground_reference_mask,
            background_exclusion_mask=background_exclusion_mask,
            seed_instance_annotations=seed_instance_annotations,
            background_reference_samples=background_reference_samples,
            background_reference_sample_count=background_reference_sample_count,
            background_prior_lab=background_prior_lab,
            background_prior_samples_lab=background_prior_samples_lab,
            background_prior_source_mask=background_prior_source_mask,
            seed_diameter=seed_diameter,
            sample_radius=max(2, round(seed_diameter * settings.background_sample_radius_fraction)),
            settings=settings,
            cuda_context=context,
            source_tensor=source_tensor,
            lab_tensor=lab_tensor,
            valid_tensor=valid_tensor,
            include_other_probability=True,
            include_reference_source_mask=True,
        )
        if background_timing is not None:
            timing_recorder.stop(background_timing)
        values["layer.background"] = background_result
    elif background_dirty:
        background_result = (
            _lazy_u8(
                source_tensor[:, :1] * 0.0, "disabled background likelihood"
            ),
            "disabled",
            0,
            None,
            None,
            None,
        )
        values["layer.background"] = background_result
    else:
        background_result = values["layer.background"]
    (
        background,
        background_mode,
        reference_count,
        colour_profile,
        other_colour_probability,
        background_reference_source_mask,
    ) = background_result

    surrounding_background_dirty = (
        background_dirty
        or "layer.surrounding_background" not in values
    )
    if (
        surrounding_background_dirty
        and background_colour_enabled
        and colour_profile is not None
        and surrounding_noise_source_tensor is not None
        and surrounding_noise_valid_tensor is not None
    ):
        surrounding_background_result = background_colour_profile_likelihood(
            surrounding_noise_source_tensor,
            surrounding_noise_valid_tensor,
            colour_profile,
            settings,
            cuda_context=context,
        )
        values["layer.surrounding_background"] = surrounding_background_result
    elif surrounding_background_dirty:
        surrounding_background_result = (None, None)
        values["layer.surrounding_background"] = surrounding_background_result
    else:
        surrounding_background_result = values["layer.surrounding_background"]
    (
        surrounding_background,
        surrounding_background_valid,
    ) = surrounding_background_result

    refined_dirty = (
        background_dirty
        or "refined_background_likelihood" in dirty
        or "layer.refined_background" not in values
    )
    refined_timing = None
    if refined_dirty and background_noise_enabled:
        refined_timing = (
            None
            if timing_recorder is None
            else timing_recorder.start("refined_background_likelihood")
        )
        refined_result = noise_frequency_background_likelihood(
            crop,
            valid_mask,
            None,
            seed_diameter,
            settings,
            background_reference_points=background_reference_points,
            foreground_reference_points=foreground_reference_points,
            background_reference_mask=background_reference_mask,
            automatic_target_reference_mask=(
                background_reference_source_mask
            ),
            automatic_target_authority=(
                1.0
                if colour_profile is None
                else colour_profile.automatic_evidence_authority
            ),
            external_target_source_tensor=surrounding_noise_source_tensor,
            external_target_valid_tensor=surrounding_noise_valid_tensor,
            automatic_nontarget_reference_mask=(
                foreground_reference_source_mask
            ),
            foreground_reference_mask=foreground_reference_mask,
            target_exclusion_mask=None,
            other_reference_mask=background_exclusion_mask,
            other_colour_likelihood=None,
            include_other_probability=True,
            reference_radius=max(
                2,
                round(seed_diameter * settings.background_sample_radius_fraction),
            ),
            cuda_context=context,
            source_tensor=source_tensor,
            lab_tensor=lab_tensor,
            valid_tensor=valid_tensor,
        )
        values["layer.refined_background"] = refined_result
    elif refined_dirty:
        refined_result = (
            _lazy_u8(
                source_tensor[:, :1] * 0.0,
                "disabled refined background likelihood",
            ),
            empty_noise_frequency_profile(seed_diameter, settings),
            (),
            (),
            None,
            None,
        )
        values["layer.refined_background"] = refined_result
    else:
        refined_result = values["layer.refined_background"]
    (
        refined_background,
        noise_profile,
        directional_background,
        directional_angles,
        other_noise_probability,
        other_noise_profile,
    ) = refined_result
    foreground_noise_signature = (
        None if species_library is None else species_library.manifest.content_sha256,
        str(settings.foreground_noise_reference_source),
        float(settings.foreground_noise_current_reference_weight),
    )
    foreground_noise_dirty = (
        painted_reference_dirty
        or "foreground_segmentation" in dirty
        or "refined_background_likelihood" in dirty
        or "foreground_noise_likelihood" in dirty
        or "layer.foreground_noise" not in values
        or values.get("layer.foreground_noise_signature")
        != foreground_noise_signature
    )
    if (
        foreground_noise_dirty
        and foreground_noise_enabled
        and foreground_probability is not None
    ):
        foreground_noise_timing = (
            None
            if timing_recorder is None
            else timing_recorder.start("foreground_noise_likelihood")
        )
        foreground_noise_result = noise_frequency_foreground_likelihood(
            crop,
            valid_mask,
            None,
            seed_diameter,
            settings,
            background_reference_points=background_reference_points,
            foreground_reference_points=foreground_reference_points,
            background_reference_mask=background_reference_mask,
            foreground_reference_mask=foreground_reference_mask,
            automatic_foreground_reference_mask=(
                foreground_reference_source_mask
            ),
            foreground_exclusion_mask=foreground_exclusion_mask,
            reference_radius=max(
                2,
                round(seed_diameter * settings.background_sample_radius_fraction),
            ),
            cuda_context=context,
            source_tensor=source_tensor,
            lab_tensor=lab_tensor,
            valid_tensor=valid_tensor,
            library_target_bank=(
                None if species_library is None else species_library.foreground_noise
            ),
        )
        values["layer.foreground_noise"] = foreground_noise_result
        values["layer.foreground_noise_signature"] = foreground_noise_signature
        if foreground_noise_timing is not None:
            timing_recorder.stop(foreground_noise_timing)
    elif foreground_noise_dirty:
        foreground_noise_result = (
            _lazy_u8(
                source_tensor[:, :1] * 0.0,
                "disabled foreground noise likelihood",
            ),
            empty_noise_frequency_profile(
                seed_diameter, _foreground_noise_settings(settings)
            ),
            (),
            (),
        )
        values["layer.foreground_noise"] = foreground_noise_result
        values["layer.foreground_noise_signature"] = foreground_noise_signature
    else:
        foreground_noise_result = values["layer.foreground_noise"]
    (
        foreground_noise,
        foreground_noise_profile,
        _directional_foreground,
        _directional_foreground_angles,
    ) = foreground_noise_result
    surrounding_dirty = (
        refined_dirty
        or "layer.surrounding_noise" not in values
    )
    if (
        surrounding_dirty
        and background_noise_enabled
        and surrounding_noise_source_tensor is not None
        and surrounding_noise_valid_tensor is not None
    ):
        surrounding_result = surrounding_band_noise_likelihood(
            surrounding_noise_source_tensor,
            surrounding_noise_valid_tensor,
            seed_diameter,
            settings,
            noise_profile,
            colour_likelihood=None,
            cuda_context=context,
        )
        values["layer.surrounding_noise"] = surrounding_result
    elif surrounding_dirty:
        surrounding_result = (None, None)
        values["layer.surrounding_noise"] = surrounding_result
    else:
        surrounding_result = values["layer.surrounding_noise"]
    surrounding_noise, surrounding_noise_valid = surrounding_result
    if refined_timing is not None:
        timing_recorder.stop(refined_timing)

    instance_dirty = (
        refined_dirty or "instance_masks" in dirty or "layer.instances" not in values
    )
    if instance_dirty and instance_masks_enabled:
        import torch

        instance_timing = (
            None
            if timing_recorder is None
            else timing_recorder.start("instance_masks")
        )

        agreed_background = (
            instance_nonseed_probability
            if instance_nonseed_probability is not None
            else GpuRaster(
                torch.minimum(
                    _raster_tensor(background, context),
                    _raster_tensor(refined_background, context),
                ).to(torch.uint8),
                numpy_dtype=np.uint8,
                name="legacy agreed background likelihood",
            )
        )
        (
            instance_centers,
            instance_radii,
            annotation_identifiers,
        ) = _merge_annotated_instance_seeds(
            centers,
            radii,
            seed_instance_annotations,
            seed_diameter,
        )
        labels = instance_voronoi(
            valid_mask,
            agreed_background,
            instance_centers,
            instance_radii,
            seed_diameter,
            settings,
            cuda_context=context,
            valid_tensor=valid_tensor,
            seed_instance_annotations=seed_instance_annotations,
            annotation_identifiers=annotation_identifiers,
        )
        colours = spatially_contrasting_colours(instance_centers)
        values["layer.instances"] = labels, colours
        if instance_timing is not None:
            timing_recorder.stop(instance_timing)
    elif instance_dirty:
        import torch

        labels = GpuRaster(
            torch.zeros(
                valid_tensor.shape,
                device=context.device,
                dtype=torch.int32,
            ),
            numpy_dtype=np.int32,
            name="disabled instance masks",
        )
        colours = np.zeros((1, 3), np.uint8)
        values["layer.instances"] = labels, colours
    else:
        labels, colours = values["layer.instances"]

    gradients_dirty = (
        wavelet_dirty
        or "edge_gradients" in dirty
        or "layer.edge_gradients" not in values
    )
    if gradients_dirty:
        gradient_timing = (
            None
            if timing_recorder is None
            else timing_recorder.start("edge_gradients")
        )
        gradient_result = directional_edges(
            crop,
            valid_mask,
            settings,
            cuda_context=context,
            source_tensor=source_tensor,
            lab_tensor=lab_tensor,
            valid_tensor=valid_tensor,
            despeckled_flattened_tensor=despeckled_flattened_tensor,
            wavelet_details=wavelet_details,
            wavelet_residual=wavelet_residual,
        )
        if gradient_timing is not None:
            timing_recorder.stop(gradient_timing)
        values["layer.edge_gradients"] = gradient_result
    else:
        gradient_result = values["layer.edge_gradients"]
    shared_edge_likelihood = gradient_result.strength_raster
    shared_directed_hue = gradient_result.directed_hue_raster
    shared_undirected_hue = gradient_result.undirected_hue_raster

    surface_gradients_dirty = (
        "surface_darkness_gradients" in dirty
        or "layer.surface_darkness_gradients" not in values
    )
    if surface_gradients_dirty and surface_darkness_gradients_enabled:
        surface_timing = (
            None
            if timing_recorder is None
            else timing_recorder.start("surface_darkness_gradients")
        )
        surface_gradients = surface_directional_darkness_gradients(
            crop,
            valid_mask,
            seed_diameter,
            settings,
            cuda_context=context,
            lab_tensor=lab_tensor,
            valid_tensor=valid_tensor,
        )
        if surface_timing is not None:
            timing_recorder.stop(surface_timing)
        values["layer.surface_darkness_gradients"] = surface_gradients
    elif surface_gradients_dirty:
        zero = valid_tensor.float() * 0.0
        zero_raster = _lazy_u8(zero, "disabled surface darkness gradients")
        surface_gradients = SurfaceGradientProducts(
            source_height=int(zero.shape[-2]),
            source_width=int(zero.shape[-1]),
            work_scale=1.0,
            valid=valid_tensor[0, 0],
            lightening_magnitude=zero[0, 0],
            darkening_magnitude=zero[0, 0],
            lightening_strength=zero[0, 0],
            darkening_strength=zero[0, 0],
            lightening_hue_float=zero[0, 0],
            darkening_hue_float=zero[0, 0],
            lightening_strength_raster=zero_raster,
            darkening_strength_raster=zero_raster,
            lightening_hue_raster=zero_raster,
            darkening_hue_raster=zero_raster,
        )
        values["layer.surface_darkness_gradients"] = surface_gradients
    else:
        surface_gradients = values["layer.surface_darkness_gradients"]

    lightening_ceiling_dirty = (
        surface_gradients_dirty
        or "lightening_gradient_ceiling" in dirty
        or "layer.lightening_gradient_ceiling" not in values
    )
    if (
        lightening_ceiling_dirty
        and surface_darkness_gradients_enabled
        and lightening_gradient_ceiling_enabled
    ):
        lightening_timing = (
            None
            if timing_recorder is None
            else timing_recorder.start("lightening_gradient_ceiling")
        )
        lightening_ceiling = upper_magnitude_surface_gradient(
            surface_gradients,
            settings.lightening_gradient_maximum_slope,
            "lightening",
        )
        if lightening_timing is not None:
            timing_recorder.stop(lightening_timing)
        values["layer.lightening_gradient_ceiling"] = lightening_ceiling
    elif lightening_ceiling_dirty:
        zero = _lazy_u8(valid_tensor.float() * 0.0, "disabled weak lightening gradient")
        lightening_ceiling = (zero, zero)
        values["layer.lightening_gradient_ceiling"] = lightening_ceiling
    else:
        lightening_ceiling = values["layer.lightening_gradient_ceiling"]

    darkening_ceiling_dirty = (
        surface_gradients_dirty
        or "darkening_gradient_ceiling" in dirty
        or "layer.darkening_gradient_ceiling" not in values
    )
    if (
        darkening_ceiling_dirty
        and surface_darkness_gradients_enabled
        and darkening_gradient_ceiling_enabled
    ):
        darkening_timing = (
            None
            if timing_recorder is None
            else timing_recorder.start("darkening_gradient_ceiling")
        )
        darkening_ceiling = upper_magnitude_surface_gradient(
            surface_gradients,
            settings.darkening_gradient_maximum_slope,
            "darkening",
        )
        if darkening_timing is not None:
            timing_recorder.stop(darkening_timing)
        values["layer.darkening_gradient_ceiling"] = darkening_ceiling
    elif darkening_ceiling_dirty:
        zero = _lazy_u8(valid_tensor.float() * 0.0, "disabled weak darkening gradient")
        darkening_ceiling = (zero, zero)
        values["layer.darkening_gradient_ceiling"] = darkening_ceiling
    else:
        darkening_ceiling = values["layer.darkening_gradient_ceiling"]

    frequency_noise_dirty = (
        "frequency_noise_masks" in dirty
        or "layer.frequency_noise_masks" not in values
    )
    if frequency_noise_dirty and frequency_noise_masks_enabled:
        frequency_timing = (
            None
            if timing_recorder is None
            else timing_recorder.start("frequency_noise_masks")
        )
        frequency_noise = multiscale_frequency_noise_masks(
            crop,
            valid_mask,
            seed_diameter,
            settings,
            cuda_context=context,
            lab_tensor=lab_tensor,
            valid_tensor=valid_tensor,
        )
        if frequency_timing is not None:
            timing_recorder.stop(frequency_timing)
        values["layer.frequency_noise_masks"] = frequency_noise
    elif frequency_noise_dirty:
        zero = _lazy_u8(valid_tensor.float() * 0.0, "disabled frequency noise mask")
        frequency_noise = FrequencyNoiseMaskProducts(
            band_scales_px=(0.0, 0.0, 0.0),
            darkness_masks=(zero, zero, zero),
            colour_masks=(zero, zero, zero),
        )
        values["layer.frequency_noise_masks"] = frequency_noise
    else:
        frequency_noise = values["layer.frequency_noise_masks"]

    # Directed and undirected tangent visualizations are products of the shared
    # Edge gradients calculation itself. Their former pass-through cache nodes
    # performed no computation and have been removed from the graph.
    edge_likelihood = shared_edge_likelihood
    directed_edge_hue = shared_directed_hue
    undirected_edge_likelihood = shared_edge_likelihood
    undirected_edge_hue = shared_undirected_hue

    previous_curve_result = values.get("layer.seed_edge_curves")
    ridges_dirty = (
        gradients_dirty
        or "edge_ridges" in dirty
        or previous_curve_result is None
    )
    traces_dirty = (
        ridges_dirty
        or "edge_traces" in dirty
        or previous_curve_result is None
    )
    if ridges_dirty or traces_dirty:
        trace_products = seed_boundary_tracing(
            gradient_result,
            seed_diameter,
            settings,
            cuda_context=context,
            previous=previous_curve_result,
            recompute_ridges=ridges_dirty,
            recompute_traces=traces_dirty,
            compute_final=False,
            timing_recorder=timing_recorder,
            seed_shape_model=seed_shape_model,
            seed_measurement_summary=seed_measurement_summary,
        )
        values["layer.edge_ridges"] = trace_products.ridges
        values["layer.edge_traces"] = (
            trace_products.trace_labels,
            trace_products.trace_continuity,
            trace_products.gap_confidence,
        )
    else:
        trace_products = previous_curve_result

    reference_texture_signature = (
        None if species_library is None else species_library.manifest.content_sha256,
        str(settings.material_prototype_reference_source),
        float(settings.material_prototype_current_reference_weight),
        str(settings.edge_prototype_reference_source),
        float(settings.edge_prototype_current_reference_weight),
        int(settings.reference_texture_material_prototypes_per_class),
        int(settings.reference_texture_edge_prototypes_per_class),
        int(settings.reference_texture_minimum_samples_per_prototype),
        int(settings.reference_texture_fit_iterations),
        float(settings.reference_texture_similarity_scale),
        float(settings.reference_texture_class_contrast),
        int(settings.reference_edge_minimum_samples_per_prototype),
        int(settings.reference_edge_fit_iterations),
        float(settings.reference_edge_similarity_scale),
        float(settings.reference_edge_class_contrast),
        float(settings.reference_texture_context_fraction),
        float(settings.reference_texture_patch_fraction),
        int(settings.reference_texture_working_maximum_dimension),
        int(settings.reference_texture_edge_working_maximum_dimension),
        float(settings.reference_edge_minimum_working_seed_diameter_px),
        float(settings.reference_edge_strip_normal_offset_fraction),
        float(settings.reference_edge_strip_tangent_half_length_fraction),
        float(settings.reference_edge_ridge_weight),
        float(settings.reference_texture_instance_interior_buffer_fraction),
    )
    reference_texture_dirty = (
        background_dirty
        or painted_reference_dirty
        or "foreground_segmentation" in dirty
        or gradients_dirty
        or frequency_noise_dirty
        or ridges_dirty
        or "reference_texture_prototypes" in dirty
        or "layer.reference_texture_prototypes" not in values
        or values.get("layer.reference_texture_signature")
        != reference_texture_signature
    )
    if reference_texture_dirty and reference_texture_prototypes_enabled:
        texture_timing = (
            None
            if timing_recorder is None
            else timing_recorder.start("reference_texture_prototypes")
        )
        if background_exclusion_mask is None:
            other_reference_mask = foreground_exclusion_mask
        elif foreground_exclusion_mask is None:
            other_reference_mask = background_exclusion_mask
        else:
            other_reference_mask = np.logical_or(
                background_exclusion_mask, foreground_exclusion_mask
            )
        reference_textures = reference_texture_probabilities(
            crop,
            gradient_result,
            trace_products.ridges,
            frequency_noise,
            seed_diameter,
            settings,
            background_reference_mask=background_reference_mask,
            background_reference_source_mask=(
                background_reference_source_mask
            ),
            background_automatic_authority=(
                1.0
                if colour_profile is None
                else colour_profile.automatic_evidence_authority
            ),
            foreground_reference_mask=foreground_reference_mask,
            foreground_reference_source_mask=(
                foreground_reference_source_mask
            ),
            other_reference_mask=other_reference_mask,
            seed_instance_annotations=seed_instance_annotations,
            species_library=species_library,
            cuda_context=context,
        )
        if texture_timing is not None:
            timing_recorder.stop(texture_timing)
        values["layer.reference_texture_prototypes"] = reference_textures
        values["layer.reference_texture_signature"] = reference_texture_signature
    elif reference_texture_dirty:
        from seedvision.visualization.layers import ReferenceTextureProfile

        zero = _lazy_u8(
            valid_tensor.float() * 0.0,
            "disabled reference texture prototypes",
        )
        reference_textures = ReferenceTextureProducts(
            seed_surface_probability=None,
            background_probability=None,
            other_probability=None,
            physical_edge_probability=zero,
            non_edge_probability=zero,
            physical_edge_field=_lazy_float(
                valid_tensor.float() * 0.0,
                "disabled continuous physical prototype compatibility",
            ),
            non_edge_field=_lazy_float(
                valid_tensor.float() * 0.0,
                "disabled continuous non-physical prototype compatibility",
            ),
            physical_edge_interior_direction_x=_lazy_float(
                valid_tensor.float() * 0.0,
                "disabled physical-edge interior direction x",
            ),
            physical_edge_interior_direction_y=_lazy_float(
                valid_tensor.float() * 0.0,
                "disabled physical-edge interior direction y",
            ),
            physical_edge_interior_direction_confidence=_lazy_float(
                valid_tensor.float() * 0.0,
                "disabled physical-edge interior direction confidence",
            ),
            profile=ReferenceTextureProfile(),
            foreground_sample_count=0,
            background_sample_count=0,
            other_sample_count=0,
            physical_sample_count=0,
            non_edge_sample_count=0,
        )
        values["layer.reference_texture_prototypes"] = reference_textures
        values["layer.reference_texture_signature"] = reference_texture_signature
    else:
        reference_textures = values["layer.reference_texture_prototypes"]

    material_evidence_dirty = (
        background_dirty
        or refined_dirty
        or foreground_noise_dirty
        or reference_texture_dirty
        or "material_evidence_decision" in dirty
        or "layer.material_evidence" not in values
    )
    if material_evidence_dirty:
        material_timing = (
            None
            if timing_recorder is None
            else timing_recorder.start("material_evidence_decision")
        )
        decision_valid_source = (
            valid_tensor.float() * 255.0
            if material_valid_mask is None
            else material_valid_mask
        )
        decision_valid = (
            decision_valid_source
            if material_evidence_enabled
            else valid_tensor.float() * 0.0
        )
        material_evidence = hierarchical_material_evidence(
            decision_valid,
            seed_diameter,
            foreground_colour=(
                foreground_probability
                if foreground_probability is not None
                else valid_tensor.float() * 0.0
            ),
            foreground_noise=foreground_noise,
            background_colour=background,
            background_noise=refined_background,
            other_colour=other_colour_probability,
            other_noise=other_noise_probability,
            reference_foreground=(
                reference_textures.seed_surface_probability
            ),
            reference_background=reference_textures.background_probability,
            reference_other=reference_textures.other_probability,
            seed_reference_mask=foreground_reference_mask,
            additional_seed_reference_mask=(
                foreground_reference_source_mask
            ),
            background_reference_mask=background_reference_mask,
            other_reference_mask=background_exclusion_mask,
            calibration_valid_mask=valid_tensor.float() * 255.0,
            proposal_valid_mask=material_proposal_valid_mask,
            colour_weight=settings.material_colour_weight,
            noise_weight=settings.material_noise_weight,
            prototype_weight=settings.material_prototype_weight,
            unknown_weight=settings.material_unknown_weight,
            temperature=settings.material_decision_temperature,
            seed_threshold=settings.material_seed_threshold,
            morphology_fraction=settings.material_morphology_fraction,
            cuda_context=context,
        )
        values["layer.material_evidence"] = material_evidence
        if material_timing is not None:
            timing_recorder.stop(material_timing)
    else:
        material_evidence = values["layer.material_evidence"]

    seed_trait_signature = (
        None if species_library is None else species_library.manifest.content_sha256,
        str(settings.seed_trait_reference_source),
        float(settings.seed_trait_current_reference_weight),
        id(seed_instance_annotations),
        str(seed_trait_species),
        tuple(str(value) for value in seed_trait_coat_patterns),
        tuple(str(value) for value in seed_trait_conditions),
        tuple(
            (
                int(value.seed_id),
                getattr(value, "coat_pattern", None),
                tuple(getattr(value, "conditions", ())),
                bool(getattr(value, "conditions_reviewed", False)),
            )
            for value in seed_instance_traits
        ),
        int(settings.reference_seed_trait_prototypes_per_class),
        int(settings.reference_seed_trait_minimum_samples_per_prototype),
        int(settings.reference_seed_trait_fit_iterations),
        float(settings.reference_seed_trait_similarity_scale),
        float(settings.reference_seed_trait_class_contrast),
        float(settings.reference_seed_trait_context_fraction),
        float(settings.reference_seed_trait_interior_buffer_fraction),
        int(settings.reference_seed_trait_working_maximum_dimension),
    )
    seed_traits_dirty = (
        gradients_dirty
        or ridges_dirty
        or frequency_noise_dirty
        or material_evidence_dirty
        or "reference_seed_traits" in dirty
        or "layer.reference_seed_traits" not in values
        or values.get("layer.reference_seed_traits_signature")
        != seed_trait_signature
    )
    if seed_traits_dirty and reference_seed_traits_enabled:
        seed_trait_timing = (
            None
            if timing_recorder is None
            else timing_recorder.start("reference_seed_traits")
        )
        seed_traits = reference_seed_trait_probabilities(
            gradient_result,
            trace_products.ridges,
            frequency_noise,
            material_evidence.seed_mask,
            seed_diameter,
            settings,
            seed_instance_annotations=seed_instance_annotations,
            seed_instance_traits=tuple(seed_instance_traits),
            annotation_species=seed_trait_species,
            coat_patterns=tuple(seed_trait_coat_patterns),
            conditions=tuple(seed_trait_conditions),
            species_library=species_library,
            cuda_context=context,
        )
        if seed_trait_timing is not None:
            timing_recorder.stop(seed_trait_timing)
        values["layer.reference_seed_traits"] = seed_traits
        values["layer.reference_seed_traits_signature"] = seed_trait_signature
    elif seed_traits_dirty:
        from seedvision.visualization.layers import ReferenceSeedTraitProfile

        zero = _lazy_u8(
            valid_tensor.float() * 0.0,
            "disabled reference seed trait probability",
        )
        seed_traits = ReferenceSeedTraitProducts(
            coat_probabilities=tuple(
                (str(name), zero) for name in seed_trait_coat_patterns
            ),
            condition_probabilities=tuple(
                (str(name), zero) for name in seed_trait_conditions
            ),
            profile=ReferenceSeedTraitProfile(
                species_id=str(seed_trait_species),
                coat_patterns=tuple(seed_trait_coat_patterns),
                conditions=tuple(seed_trait_conditions),
            ),
        )
        values["layer.reference_seed_traits"] = seed_traits
        values["layer.reference_seed_traits_signature"] = seed_trait_signature
    else:
        seed_traits = values["layer.reference_seed_traits"]

    reference_edge_dirty = (
        reference_texture_dirty
        or "reference_edge_probability" in dirty
        or "layer.reference_edge_probability" not in values
    )
    if reference_edge_dirty and reference_edge_probability_enabled:
        reference_timing = (
            None
            if timing_recorder is None
            else timing_recorder.start("reference_edge_probability")
        )
        import torch

        subtraction_weight = float(settings.net_physical_edge_internal_scale)
        physical_compatibility = (
            reference_textures.physical_edge_probability.gpu_tensor(
                dtype=torch.float32
            )
            / 255.0
        )
        nonphysical_compatibility = (
            reference_textures.non_edge_probability.gpu_tensor(
                dtype=torch.float32
            )
            / 255.0
        )
        net_prototype_compatibility = torch.clamp(
            physical_compatibility
            - subtraction_weight * nonphysical_compatibility,
            min=0.0,
            max=1.0,
        )
        # The prototype descriptor is intentionally wider than a true image
        # edge. It classifies local edge context but cannot itself establish a
        # barrier. The authoritative reference-edge probability is therefore
        # supported only on the already thinned, image-derived edge ridge:
        #
        #   R * Pphysical
        #
        # Pphysical already includes class competition and unknown confidence.
        # A second subtraction is a conservative evidence policy, not another
        # probability calibration. Keep it separate and explicitly opt-in.
        # Both products make broad descriptor/restoration halos harmless.
        true_edge_support = (
            trace_products.ridges.gpu_tensor(dtype=torch.float32) / 255.0
        ).clamp(0.0, 1.0)
        edge_supported_physical = (
            true_edge_support * physical_compatibility
        ).clamp(0.0, 1.0)
        edge_supported_nonphysical = (
            true_edge_support * nonphysical_compatibility
        ).clamp(0.0, 1.0)
        conservative_net_edge_evidence = (
            true_edge_support * net_prototype_compatibility
        ).clamp(0.0, 1.0)
        reference_edge_probability = _lazy_u8(
            edge_supported_physical * 255.0,
            "reference-edge probability",
        )
        reference_edges = ReferenceEdgeProducts(
            physical_prototype_compatibility=(
                reference_textures.physical_edge_probability
            ),
            nonphysical_prototype_compatibility=(
                reference_textures.non_edge_probability
            ),
            net_prototype_compatibility=_lazy_u8(
                net_prototype_compatibility * 255.0,
                "net physical-edge prototype compatibility",
            ),
            edge_supported_physical_compatibility=reference_edge_probability,
            edge_supported_nonphysical_compatibility=_lazy_u8(
                edge_supported_nonphysical * 255.0,
                "edge-supported non-physical prototype compatibility",
            ),
            reference_edge_probability=reference_edge_probability,
            conservative_net_physical_edge_evidence=_lazy_u8(
                conservative_net_edge_evidence * 255.0,
                "conservative net physical-edge evidence",
            ),
            net_prototype_field=_lazy_float(
                net_prototype_compatibility,
                "continuous net physical-edge prototype compatibility",
            ),
            reference_edge_field=_lazy_float(
                edge_supported_physical,
                "continuous reference-edge probability",
            ),
            conservative_net_edge_field=_lazy_float(
                conservative_net_edge_evidence,
                "continuous conservative net physical-edge evidence",
            ),
            interior_direction_x=(
                reference_textures.physical_edge_interior_direction_x
            ),
            interior_direction_y=(
                reference_textures.physical_edge_interior_direction_y
            ),
            interior_direction_confidence=(
                reference_textures.physical_edge_interior_direction_confidence
            ),
            subtraction_weight=subtraction_weight,
            physical_sample_count=reference_textures.physical_sample_count,
            non_edge_sample_count=reference_textures.non_edge_sample_count,
        )
        if reference_timing is not None:
            timing_recorder.stop(reference_timing)
        values["layer.reference_edge_probability"] = reference_edges
    elif reference_edge_dirty:
        zero = _lazy_u8(
            valid_tensor.float() * 0.0, "disabled reference edge probability"
        )
        zero_float = _lazy_float(
            valid_tensor.float() * 0.0,
            "disabled physical-edge direction",
        )
        reference_edges = ReferenceEdgeProducts(
            physical_prototype_compatibility=zero,
            nonphysical_prototype_compatibility=zero,
            net_prototype_compatibility=zero,
            edge_supported_physical_compatibility=zero,
            edge_supported_nonphysical_compatibility=zero,
            reference_edge_probability=zero,
            conservative_net_physical_edge_evidence=zero,
            net_prototype_field=_lazy_float(
                valid_tensor.float() * 0.0,
                "disabled continuous net prototype compatibility",
            ),
            reference_edge_field=_lazy_float(
                valid_tensor.float() * 0.0,
                "disabled continuous reference-edge probability",
            ),
            conservative_net_edge_field=zero_float,
            interior_direction_x=zero_float,
            interior_direction_y=zero_float,
            interior_direction_confidence=zero_float,
            subtraction_weight=float(settings.net_physical_edge_internal_scale),
            physical_sample_count=0,
            non_edge_sample_count=0,
        )
        values["layer.reference_edge_probability"] = reference_edges
    else:
        reference_edges = values["layer.reference_edge_probability"]

    reference_ridges_dirty = (
        reference_edge_dirty
        or gradients_dirty
        or "reference_edge_ridges" in dirty
        or "layer.reference_edge_ridges" not in values
        or "layer.net_reference_edge_ridges" not in values
        or "layer.locally_normalized_net_physical_edge" not in values
        or "layer.normalized_net_reference_edge_ridges" not in values
    )
    if reference_ridges_dirty and reference_edge_ridges_enabled:
        import torch

        reference_ridge_timing = (
            None
            if timing_recorder is None
            else timing_recorder.start("reference_edge_ridges")
        )
        reference_edge_ridge = reference_probability_ridges(
            reference_edges.reference_edge_field,
            gradient_result,
            settings,
            include_gradient_strength=False,
            restrict_to_source_support=True,
            cuda_context=context,
        )
        net_reference_edge_ridge = reference_probability_ridges(
            reference_edges.conservative_net_edge_field,
            gradient_result,
            settings,
            include_gradient_strength=False,
            restrict_to_source_support=True,
            cuda_context=context,
        )
        # Retain the normalized_net_* storage/port IDs for saved graphs, but
        # normalize the authoritative Physical probability, never the lambda
        # margin. Lambda must not affect this default downstream branch.
        normalized_net_reference_field = (
            locally_normalized_reference_edge_probability(
                reference_edges.physical_prototype_compatibility.gpu_tensor(
                    dtype=torch.float32
                ) / 255.0,
                trace_products.ridges,
                gradient_result,
                seed_diameter,
                settings,
                cuda_context=context,
            )
        )
        locally_normalized_net_physical = _lazy_u8(
            normalized_net_reference_field.gpu_tensor(dtype=torch.float32) * 255.0,
            "normalized reference-edge probability",
        )
        normalized_net_reference_edge_ridge = reference_probability_ridges(
            normalized_net_reference_field,
            gradient_result,
            settings,
            include_gradient_strength=False,
            restrict_to_source_support=True,
            cuda_context=context,
        )
        if reference_ridge_timing is not None:
            timing_recorder.stop(reference_ridge_timing)
        values["layer.reference_edge_ridges"] = reference_edge_ridge
        values["layer.net_reference_edge_ridges"] = net_reference_edge_ridge
        values["layer.locally_normalized_net_physical_edge"] = (
            locally_normalized_net_physical
        )
        values["layer.normalized_net_reference_edge_ridges"] = (
            normalized_net_reference_edge_ridge
        )
    elif reference_ridges_dirty:
        reference_edge_ridge = _lazy_u8(
            valid_tensor.float() * 0.0,
            "disabled thinned reference-edge ridge",
        )
        values["layer.reference_edge_ridges"] = reference_edge_ridge
        net_reference_edge_ridge = _lazy_u8(
            valid_tensor.float() * 0.0,
            "disabled thinned conservative net physical-edge ridge",
        )
        values["layer.net_reference_edge_ridges"] = net_reference_edge_ridge
        locally_normalized_net_physical = _lazy_u8(
            valid_tensor.float() * 0.0,
            "disabled normalized reference-edge probability",
        )
        values["layer.locally_normalized_net_physical_edge"] = (
            locally_normalized_net_physical
        )
        normalized_net_reference_edge_ridge = _lazy_u8(
            valid_tensor.float() * 0.0,
            "disabled thinned normalized reference-edge ridge",
        )
        values["layer.normalized_net_reference_edge_ridges"] = (
            normalized_net_reference_edge_ridge
        )
    else:
        reference_edge_ridge = values["layer.reference_edge_ridges"]
        net_reference_edge_ridge = values["layer.net_reference_edge_ridges"]
        locally_normalized_net_physical = values[
            "layer.locally_normalized_net_physical_edge"
        ]
        normalized_net_reference_edge_ridge = values[
            "layer.normalized_net_reference_edge_ridges"
        ]

    trace_source_name = str(settings.trace_edge_source)
    trace_source_raster = None
    selected_trace_dirty = False
    if trace_source_name != "generic_ridges":
        trace_source_raster = {
            "reference_ridges": reference_edge_ridge,
            "net_reference_ridges": net_reference_edge_ridge,
            "normalized_net_reference_ridges": normalized_net_reference_edge_ridge,
        }[trace_source_name]
        selected_trace_dirty = (
            traces_dirty
            or reference_ridges_dirty
            or "edge_traces" in dirty
            or previous_curve_result is None
        )
        trace_products = seed_boundary_tracing(
            gradient_result,
            seed_diameter,
            settings,
            cuda_context=context,
            previous=trace_products,
            recompute_ridges=False,
            recompute_traces=selected_trace_dirty,
            compute_final=False,
            trace_ridge_override=trace_source_raster,
            trace_source_name=trace_source_name,
            timing_recorder=timing_recorder,
            seed_shape_model=seed_shape_model,
            seed_measurement_summary=seed_measurement_summary,
        )
        values["layer.edge_traces"] = (
            trace_products.trace_labels,
            trace_products.trace_continuity,
            trace_products.gap_confidence,
        )

    curve_dirty = (
        traces_dirty
        or reference_edge_dirty
        or background_dirty
        or "seed_edge_curves" in dirty
        or previous_curve_result is None
    )
    if curve_dirty:
        curve_result = seed_boundary_tracing(
            gradient_result,
            seed_diameter,
            settings,
            background_likelihood=(
                material_evidence.nonseed_probability
                if material_evidence_enabled
                else background
            ),
            foreground_probability=(
                material_evidence.seed_probability
                if material_evidence_enabled
                else foreground_probability
            ),
            physical_edge_probability=(
                reference_edges.edge_supported_physical_compatibility
            ),
            non_edge_probability=(
                reference_edges.edge_supported_nonphysical_compatibility
            ),
            seed_shape_model=seed_shape_model,
            seed_measurement_summary=seed_measurement_summary,
            cuda_context=context,
            previous=trace_products,
            recompute_ridges=False,
            recompute_traces=False,
            compute_final=seed_edge_curves_enabled,
            trace_ridge_override=trace_source_raster,
            trace_source_name=trace_source_name,
            timing_recorder=timing_recorder,
        )
        values["layer.seed_edge_curves"] = curve_result
    else:
        curve_result = values["layer.seed_edge_curves"]
    curve_likelihood = curve_result.final_likelihood
    curve_radius = curve_result.selected_radius

    return AnalysisLayers(
        offset_x=offset_x,
        offset_y=offset_y,
        instance_labels=labels,
        instance_colours=colours,
        background_likelihood=background,
        refined_background_likelihood=refined_background,
        noise_frequency_profile=noise_profile,
        foreground_noise_likelihood=foreground_noise,
        foreground_noise_frequency_profile=foreground_noise_profile,
        automatic_foreground_colour_probability=(
            automatic_foreground_probability
        ),
        reviewed_foreground_colour_probability=(
            reviewed_foreground_probability
        ),
        foreground_automatic_evidence_authority=float(
            foreground_automatic_evidence_authority
        ),
        background_automatic_evidence_authority=(
            1.0
            if colour_profile is None
            else float(colour_profile.automatic_evidence_authority)
        ),
        other_colour_probability=other_colour_probability,
        other_noise_probability=other_noise_probability,
        other_noise_frequency_profile=other_noise_profile,
        seed_evidence_support=material_evidence.seed_support,
        background_evidence_support=material_evidence.background_support,
        other_evidence_support=material_evidence.other_support,
        nonseed_evidence_support=material_evidence.nonseed_support,
        seed_material_probability=material_evidence.seed_probability,
        nonseed_material_probability=material_evidence.nonseed_probability,
        material_ambiguity_probability=(
            material_evidence.ambiguity_probability
        ),
        material_unknown_probability=material_evidence.unknown_probability,
        conditional_background_probability=(
            material_evidence.conditional_background_probability
        ),
        conditional_other_probability=(
            material_evidence.conditional_other_probability
        ),
        conditional_material_subtype_ambiguity=(
            material_evidence.conditional_ambiguity_probability
        ),
        conditional_nonseed_unknown_probability=(
            material_evidence.conditional_unknown_probability
        ),
        material_source_reliabilities=(
            material_evidence.source_reliabilities
        ),
        seed_material_mask=material_evidence.seed_mask,
        background_reference_source_mask=(
            background_reference_source_mask
        ),
        foreground_reference_source_mask=(
            foreground_reference_source_mask
        ),
        reference_seed_surface_probability=(
            reference_textures.seed_surface_probability
        ),
        reference_background_texture_probability=(
            reference_textures.background_probability
        ),
        reference_other_texture_probability=reference_textures.other_probability,
        reference_texture_profile=reference_textures.profile,
        reference_seed_coat_probabilities=(
            seed_traits.coat_probabilities
        ),
        reference_seed_condition_probabilities=(
            seed_traits.condition_probabilities
        ),
        reference_seed_trait_profile=seed_traits.profile,
        physical_edge_probability=(
            reference_edges.physical_prototype_compatibility
        ),
        non_edge_probability=(
            reference_edges.nonphysical_prototype_compatibility
        ),
        net_physical_edge_probability=(
            reference_edges.net_prototype_compatibility
        ),
        reference_edge_probability=(
            reference_edges.reference_edge_probability
        ),
        conservative_net_physical_edge_evidence=(
            reference_edges.conservative_net_physical_edge_evidence
        ),
        edge_supported_physical_compatibility=(
            reference_edges.edge_supported_physical_compatibility
        ),
        edge_supported_nonphysical_compatibility=(
            reference_edges.edge_supported_nonphysical_compatibility
        ),
        physical_edge_interior_direction_x=(
            reference_edges.interior_direction_x
        ),
        physical_edge_interior_direction_y=(
            reference_edges.interior_direction_y
        ),
        physical_edge_interior_direction_confidence=(
            reference_edges.interior_direction_confidence
        ),
        net_physical_edge_internal_scale=(
            reference_edges.subtraction_weight
        ),
        reference_edge_ridges=reference_edge_ridge,
        net_reference_edge_ridges=net_reference_edge_ridge,
        locally_normalized_net_physical_edge=(
            locally_normalized_net_physical
        ),
        normalized_net_reference_edge_ridges=(
            normalized_net_reference_edge_ridge
        ),
        hue_only_rgb=hue_only,
        wavelet_details=wavelet_details,
        wavelet_residual=wavelet_residual,
        edge_likelihood=edge_likelihood,
        directed_edge_hue=directed_edge_hue,
        undirected_edge_hue=undirected_edge_hue,
        seed_edge_curve_likelihood=curve_likelihood,
        seed_edge_curve_radius_px=curve_radius,
        lightening_surface_gradient=surface_gradients.lightening_strength_raster,
        lightening_surface_direction=surface_gradients.lightening_hue_raster,
        darkening_surface_gradient=surface_gradients.darkening_strength_raster,
        darkening_surface_direction=surface_gradients.darkening_hue_raster,
        weak_lightening_surface_gradient=lightening_ceiling[0],
        weak_lightening_surface_direction=lightening_ceiling[1],
        weak_darkening_surface_gradient=darkening_ceiling[0],
        weak_darkening_surface_direction=darkening_ceiling[1],
        frequency_noise_band_scales_px=frequency_noise.band_scales_px,
        darkness_frequency_noise_masks=frequency_noise.darkness_masks,
        colour_frequency_noise_masks=frequency_noise.colour_masks,
        valid_mask=_lazy_u8(valid_tensor.float() * 255.0, "valid dish mask"),
        undirected_edge_likelihood=undirected_edge_likelihood,
        background_mode=background_mode,
        background_reference_count=reference_count,
        background_colour_profile=colour_profile,
        foreground_colour_profile=foreground_colour_profile,
        directional_background_likelihoods=directional_background,
        directional_background_angles_degrees=directional_angles,
        edge_ridges=curve_result.ridges,
        edge_trace_labels=curve_result.trace_labels,
        edge_trace_continuity=curve_result.trace_continuity,
        edge_trace_gap_confidence=curve_result.gap_confidence,
        edge_radius_ratio_hue=curve_result.radius_ratio_hue,
        edge_radius_confidence=curve_result.radius_confidence,
        edge_circle_confidence=curve_result.circle_confidence,
        edge_ellipse_confidence=curve_result.ellipse_confidence,
        edge_fit_residual=curve_result.fit_residual,
        edge_centre_votes=curve_result.centre_votes,
        oval_centre_probability=curve_result.oval_centre_probability,
        edge_semantic_sides=curve_result.semantic_sides,
        edge_rejection_hue=curve_result.rejection_hue,
        edge_rejection_strength=curve_result.rejection_strength,
        edge_fit_geometry=curve_result.geometry,
        surrounding_background_likelihood=surrounding_background,
        surrounding_background_valid_mask=surrounding_background_valid,
        surrounding_noise_likelihood=surrounding_noise,
        surrounding_noise_valid_mask=surrounding_noise_valid,
        surrounding_noise_offset_x=int(surrounding_noise_offset_x),
        surrounding_noise_offset_y=int(surrounding_noise_offset_y),
        gpu_source=source_tensor,
        gpu_valid=valid_tensor,
    )


def _point_mask(height, width, points, radius, context):
    import torch

    mask = torch.zeros(
        (height, width), device=context.device, dtype=torch.bool
    )
    if not points:
        return mask
    yy, xx = torch.meshgrid(
        torch.arange(height, device=context.device, dtype=torch.float32),
        torch.arange(width, device=context.device, dtype=torch.float32),
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


def background_colour_likelihood(
    crop: np.ndarray,
    valid_mask: np.ndarray,
    *,
    background_reference_points=(),
    foreground_reference_points=(),
    background_reference_mask=None,
    foreground_reference_mask=None,
    background_exclusion_mask=None,
    seed_instance_annotations=None,
    background_reference_samples=None,
    background_reference_sample_count=0,
    background_prior_lab=None,
    background_prior_samples_lab=None,
    background_prior_source_mask=None,
    seed_diameter=None,
    sample_radius=3,
    settings=None,
    cuda_context=None,
    source_tensor=None,
    lab_tensor=None,
    valid_tensor=None,
    include_other_probability=False,
    include_reference_source_mask=False,
):
    from seedvision.visualization.layers import BackgroundColourProfile, AnalysisLayerSettings
    import torch

    settings = settings or AnalysisLayerSettings()
    context = cuda_context or CudaContext.resolve()
    source = image_to_tensor(crop, context) if source_tensor is None else source_tensor
    base_lab = bgr_to_lab(source) if lab_tensor is None else lab_tensor
    lab = base_lab[0].permute(1, 2, 0)
    valid = (
        image_to_tensor(valid_mask, context)[0, 0] > 0
        if valid_tensor is None
        else valid_tensor[0, 0].bool()
    )
    height, width = valid.shape
    background_point_mask = _point_mask(
        height,
        width,
        background_reference_points,
        sample_radius,
        context,
    ) & valid
    foreground_point_mask = _point_mask(
        height,
        width,
        foreground_reference_points,
        sample_radius,
        context,
    ) & valid
    if background_reference_mask is not None:
        background_point_mask |= image_to_tensor(
            np.asarray(background_reference_mask, np.uint8), context
        )[0, 0] > 0
        background_point_mask &= valid
    if foreground_reference_mask is not None:
        foreground_point_mask |= image_to_tensor(
            np.asarray(foreground_reference_mask, np.uint8), context
        )[0, 0] > 0
        foreground_point_mask &= valid
    background_exclusion = torch.zeros_like(valid)
    if background_exclusion_mask is not None:
        background_exclusion = image_to_tensor(
            np.asarray(background_exclusion_mask, np.uint8), context
        )[0, 0] > 0
        background_exclusion &= valid
        background_point_mask &= ~background_exclusion
    annotated_seed_mask = torch.zeros_like(valid)
    if seed_instance_annotations is not None:
        annotation_values = np.asarray(seed_instance_annotations)
        if annotation_values.shape != (height, width):
            raise ValueError(
                "Seed instance annotations must match the crop dimensions."
            )
        annotated_seed_mask = image_to_tensor(
            np.asarray(annotation_values > 0, np.uint8), context
        )[0, 0] > 0
        annotated_seed_mask &= valid
    mode = "automatic"
    reference_count = 0
    manual_samples = None
    source_values = None
    eligible = valid & ~foreground_point_mask & ~background_exclusion
    eligible &= ~annotated_seed_mask

    def limited_rows(values, maximum=32768):
        """Bound one already-materialized sample group deterministically."""

        count = int(values.shape[0])
        if count <= maximum:
            return values
        indices = torch.linspace(
            0, count - 1, maximum, device=values.device
        ).round().long()
        return values[indices]

    def sampled_mask_indices(mask, maximum=32768):
        """Select spatially distributed true pixels without a huge nonzero list."""

        flat = mask.reshape(-1)
        count = int(flat.sum().item())
        if not count:
            return torch.empty(
                (0,), device=mask.device, dtype=torch.long
            )
        wanted_count = min(count, int(maximum))
        wanted_ranks = np.rint(
            np.linspace(0, count - 1, wanted_count)
        ).astype(np.int64)
        selected = []
        cumulative = 0
        chunk_size = 1_048_576
        for start in range(0, int(flat.numel()), chunk_size):
            stop = min(start + chunk_size, int(flat.numel()))
            chunk = flat[start:stop]
            chunk_count = int(chunk.sum().item())
            if not chunk_count:
                continue
            left = int(np.searchsorted(wanted_ranks, cumulative, side="left"))
            right = int(
                np.searchsorted(
                    wanted_ranks, cumulative + chunk_count, side="left"
                )
            )
            if right > left:
                local_true = torch.nonzero(chunk, as_tuple=False).flatten()
                local_ranks = torch.as_tensor(
                    wanted_ranks[left:right] - cumulative,
                    device=mask.device,
                    dtype=torch.long,
                )
                selected.append(local_true[local_ranks] + start)
            cumulative += chunk_count
        return torch.cat(selected) if selected else torch.empty(
            (0,), device=mask.device, dtype=torch.long
        )

    if _sample_count(background_reference_samples):
        if torch.is_tensor(background_reference_samples):
            source_values = background_reference_samples.to(
                device=context.device, dtype=torch.float32
            ).reshape(-1, 3)
        else:
            source_values = torch.as_tensor(
                np.asarray(background_reference_samples).reshape(-1, 3),
                device=context.device,
                dtype=torch.float32,
            )
        source_values = limited_rows(source_values)
        manual_samples = _bgr_samples_to_lab(source_values, context)
        mode = "manual"
        reference_count = int(background_reference_sample_count)
    elif background_reference_points or background_point_mask.any():
        if background_point_mask.any():
            manual_indices = sampled_mask_indices(background_point_mask)
            manual_samples = lab.reshape(-1, 3)[manual_indices]
            source_values = source[0].permute(1, 2, 0).reshape(-1, 3)[
                manual_indices
            ]
            mode = "manual"
            reference_count = (
                int(background_point_mask.sum().item())
                if background_reference_mask is not None
                else len(background_reference_points)
            )

    def lab_sample_tensor(values):
        if not _sample_count(values):
            return None
        if torch.is_tensor(values):
            return limited_rows(values.to(
                device=context.device, dtype=lab.dtype
            ).reshape(-1, 3))
        return limited_rows(torch.as_tensor(
            np.asarray(values).reshape(-1, 3),
            device=context.device,
            dtype=lab.dtype,
        ))

    def balanced_samples(groups, maximum=32768, relative_weights=None):
        """Build a bounded pool with explicit source-authority weights."""

        available = [
            values.reshape(-1, 3)
            for values in groups
            if values is not None and int(values.shape[0])
        ]
        if not available:
            return None
        if len(available) == 1:
            return limited_rows(available[0], maximum)
        target = min(
            int(maximum), sum(int(values.shape[0]) for values in available)
        )
        weights = (
            [1.0] * len(available)
            if relative_weights is None
            else [max(0.0, float(value)) for value in relative_weights]
        )
        if len(weights) != len(available):
            raise ValueError("Sample groups and source weights must align.")
        weight_total = sum(weights)
        if weight_total <= 0.0:
            weights = [1.0] * len(available)
            weight_total = float(len(available))
        allocations = [
            max(1, round(target * weight / weight_total))
            for weight in weights
        ]
        while sum(allocations) > target:
            index = max(range(len(allocations)), key=allocations.__getitem__)
            if allocations[index] <= 1:
                break
            allocations[index] -= 1
        while sum(allocations) < target:
            index = max(
                range(len(allocations)),
                key=lambda item: weights[item] / max(allocations[item], 1),
            )
            allocations[index] += 1
        retained = []
        for values, count in zip(available, allocations, strict=True):
            if count == int(values.shape[0]):
                retained.append(values)
            else:
                indices = torch.linspace(
                    0,
                    int(values.shape[0]) - 1,
                    count,
                    device=values.device,
                ).round().long()
                retained.append(values[indices])
        return torch.cat(retained, dim=0)

    perimeter_samples = lab_sample_tensor(background_prior_samples_lab)
    prior = (
        torch.as_tensor(
            background_prior_lab,
            device=context.device,
            dtype=lab.dtype,
        )
        if background_prior_lab is not None
        else (
            None
            if perimeter_samples is None
            else torch.median(perimeter_samples, dim=0).values
        )
    )
    keep_perimeter = bool(
        settings.background_keep_perimeter_reference and prior is not None
    )
    manual_reviewed_count = (
        0 if manual_samples is None else int(manual_samples.shape[0])
    )
    automatic_evidence_authority = 1.0
    if manual_reviewed_count:
        effective_seed_diameter = (
            max(1.0, float(seed_diameter))
            if seed_diameter is not None
            else max(10.0, float(sample_radius) * 10.0)
        )
        nominal_seed_area = max(
            1.0, np.pi * (effective_seed_diameter * 0.5) ** 2
        )
        reviewed_seed_areas = manual_reviewed_count / nominal_seed_area
        floor = float(settings.background_automatic_evidence_floor)
        half_life = max(
            0.05,
            float(
                settings.background_reviewed_authority_half_life_seed_areas
            ),
        )
        automatic_evidence_authority = floor + (1.0 - floor) * 2.0 ** (
            -reviewed_seed_areas / half_life
        )
    automatic_reference_mask = None
    no_eligible_background_source = False
    if keep_perimeter:
        if background_prior_source_mask is not None:
            if hasattr(background_prior_source_mask, "gpu_tensor"):
                prior_source_values = background_prior_source_mask.gpu_tensor(
                    device=context.device
                )
                if prior_source_values.ndim == 2:
                    prior_source_values = prior_source_values[None, None]
                elif prior_source_values.ndim == 3:
                    prior_source_values = prior_source_values[None]
            elif torch.is_tensor(background_prior_source_mask):
                prior_source_values = background_prior_source_mask.to(
                    device=context.device
                )
                if prior_source_values.ndim == 2:
                    prior_source_values = prior_source_values[None, None]
                elif prior_source_values.ndim == 3:
                    prior_source_values = prior_source_values[None]
            else:
                prior_source_values = image_to_tensor(
                    np.asarray(background_prior_source_mask, np.uint8), context
                )
            if prior_source_values.shape[-2:] != (height, width):
                raise ValueError(
                    "The perimeter background source mask must match the crop dimensions."
                )
            # Retain only the exact colour-filtered sampling ring. Do not use
            # its median colour to recruit similar pixels from inside the dish.
            # Authored Foreground/Other/instance regions retain precedence if
            # the emergency inside-rim sampling fallback overlaps them.
            automatic_reference_mask = (
                prior_source_values[0, 0] > 0
            ) & eligible & ~background_point_mask & ~annotated_seed_mask

        # Preserve the accepted ring's observed multimodality without allowing
        # its raw area to drown painted Background anchors.
        ring_anchor = (
            perimeter_samples
            if perimeter_samples is not None
            else prior[None, :]
        )
        supplied = balanced_samples(
            (manual_samples, ring_anchor),
            relative_weights=(1.0, automatic_evidence_authority),
        )
        automatic_prior_samples = manual_samples is None
    elif manual_samples is not None:
        supplied = balanced_samples((manual_samples,))
        automatic_prior_samples = False
    elif not bool(eligible.any().item()):
        # Every valid coordinate is explicitly Foreground or Other. Retain a
        # single technical fit sample so the compact profile stays well formed,
        # but mask the model to zero below rather than stealing a negative
        # reference pixel or silently re-enabling the perimeter prior.
        first_valid = torch.nonzero(
            valid.reshape(-1), as_tuple=False
        ).flatten()[:1]
        supplied = lab.reshape(-1, 3)[first_valid]
        source_values = source[0].permute(1, 2, 0).reshape(-1, 3)[first_valid]
        automatic_prior_samples = False
        no_eligible_background_source = True
    else:
        # An explicit perimeter opt-out is honest even without paint: neither
        # the median nor the annulus samples influence this independent
        # low-chroma/light-background fallback.
        valid_lab = lab[eligible]
        chroma = torch.sqrt(
            (valid_lab[:, 1] - 128.0).square()
            + (valid_lab[:, 2] - 128.0).square()
        )
        chroma_limit = torch.quantile(
            chroma, settings.background_chroma_percentile / 100.0
        )
        lightness_limit = torch.quantile(
            valid_lab[:, 0], settings.background_lightness_percentile / 100.0
        )
        candidates = eligible & (
            torch.sqrt(
                (lab[:, :, 1] - 128.0).square()
                + (lab[:, :, 2] - 128.0).square()
            )
            <= chroma_limit
        ) & (lab[:, :, 0] >= lightness_limit)
        minimum = max(
            32,
            round(
                valid.sum().item()
                * settings.background_minimum_sample_fraction
            ),
        )
        if int(candidates.sum().item()) < minimum:
            score = lab[:, :, 0] - torch.sqrt(
                (lab[:, :, 1] - 128.0).square()
                + (lab[:, :, 2] - 128.0).square()
            )
            threshold = torch.quantile(
                score[eligible],
                max(
                    0.0,
                    1.0 - minimum / max(int(eligible.sum().item()), 1),
                ),
            )
            candidates = eligible & (score >= threshold)
        candidate_indices = sampled_mask_indices(candidates)
        supplied = lab.reshape(-1, 3)[candidate_indices]
        source_values = source[0].permute(1, 2, 0).reshape(-1, 3)[
            candidate_indices
        ]
        automatic_prior_samples = False
    del annotated_seed_mask

    (
        likelihood,
        component_centres,
        component_scales,
        component_weights,
        fitted_sample_count,
        refinement_rounds,
    ) = lab_colour_distribution(
        lab,
        supplied,
        eligible,
        maximum_components=settings.background_colour_components,
        fit_iterations=settings.background_distribution_fit_iterations,
        refinement_iterations=0,
        frequency_weight_power=settings.background_frequency_weight_power,
        scale_multiplier=settings.background_distribution_scale_multiplier,
        combine_modes="maximum",
        scale_floors=(
            settings.background_lightness_scale_floor,
            settings.background_chroma_scale_floor,
            settings.background_chroma_scale_floor,
        ),
        # Reference classes restrict fitting, but every valid coordinate is
        # evaluated by the resulting colour model. This prevents annotation
        # coordinates from becoming hard-coded output values.
        output_mask=(
            eligible if no_eligible_background_source else valid
        ),
    )
    dominant_component = int(torch.argmax(component_weights).item())
    centre = component_centres[dominant_component]
    scale = component_scales[dominant_component]
    excluded_membership = None
    excluded_centres = None
    excluded_scales = None
    excluded_weights = None
    exclusion_strength = 0.95
    if bool(background_exclusion.any().item()):
        (
            excluded_membership,
            excluded_centres,
            excluded_scales,
            excluded_weights,
            _,
            _,
        ) = lab_colour_frequency_distribution(
            lab,
            lab[background_exclusion],
            valid,
            # Keep the competing Other model expressive without multiplying a
            # now high-capacity positive-model control into hundreds of costly
            # full-image frequency passes. Sixty-four preserves the previous
            # default Other capacity when the positive default was four.
            maximum_bins=max(
                16, min(64, settings.background_colour_components * 16)
            ),
            refinement_iterations=0,
            frequency_weight_power=0.0,
            scale_multiplier=settings.background_distribution_scale_multiplier,
            scale_floors=(
                settings.background_lightness_scale_floor,
                settings.background_chroma_scale_floor,
                settings.background_chroma_scale_floor,
            ),
        )
        # Preserve the raw Background fit.  Other is positive Non-seed evidence
        # in the hierarchical decision and a conditional subtype here; it must
        # not erase Background support on glass that legitimately fits both.

    if keep_perimeter:
        # The bounded, source-balanced anchor pool is the exact colour fit
        # input, including retained ring Lab samples. Convert that compact
        # metadata pool so the reported BGR range cannot omit the ring merely
        # because painted BGR values were also available.
        import cv2

        fitted_source_lab = (
            supplied.detach().to(device="cpu", dtype=torch.float32).numpy()
        )
        fitted_source_lab_u8 = np.clip(
            np.rint(fitted_source_lab.reshape(-1, 1, 3)), 0, 255
        ).astype(np.uint8)
        fitted_source_bgr = cv2.cvtColor(
            fitted_source_lab_u8, cv2.COLOR_LAB2BGR
        ).reshape(-1, 3)
        range_values = torch.as_tensor(
            fitted_source_bgr,
            device=context.device,
            dtype=torch.float32,
        )
    elif source_values is None:
        # Convert the selected Lab samples' corresponding input colours only for
        # the compact UI range. Manual sample BGR values are already supplied.
        if _sample_count(background_reference_samples):
            if torch.is_tensor(background_reference_samples):
                range_values = background_reference_samples.to(
                    device=context.device, dtype=torch.float32
                ).reshape(-1, 3)
            else:
                range_values = torch.as_tensor(
                    np.asarray(background_reference_samples).reshape(-1, 3),
                    device=context.device,
                    dtype=torch.float32,
                )
        elif automatic_prior_samples and _sample_count(background_prior_samples_lab):
            # The fitted automatic samples come from the exterior annulus in
            # Lab space. Convert that compact sample itself for the inspector's
            # observed BGR range; using arbitrary in-dish valid pixels here made
            # the caption look as though the annulus seed had been discarded.
            import cv2

            prior_lab = (
                background_prior_samples_lab.detach()
                .to(device="cpu", dtype=torch.float32)
                .numpy()
                if torch.is_tensor(background_prior_samples_lab)
                else np.asarray(background_prior_samples_lab, dtype=np.float32)
            )
            prior_lab_u8 = np.clip(
                np.rint(prior_lab.reshape(-1, 1, 3)), 0, 255
            ).astype(np.uint8)
            prior_bgr = cv2.cvtColor(
                prior_lab_u8, cv2.COLOR_LAB2BGR
            ).reshape(-1, 3)
            range_values = torch.as_tensor(
                prior_bgr, device=context.device, dtype=torch.float32
            )
        else:
            range_values = source[0].permute(1, 2, 0)[valid][: min(4096, int(valid.sum().item()))]
    else:
        range_values = source_values
    if int(range_values.shape[0]) > 32768:
        range_indices = torch.linspace(
            0,
            int(range_values.shape[0]) - 1,
            32768,
            device=range_values.device,
        ).round().long()
        range_values = range_values[range_indices]
    low = torch.quantile(range_values, 0.05, dim=0).round().clamp(0, 255).cpu().numpy()
    high = torch.quantile(range_values, 0.95, dim=0).round().clamp(0, 255).cpu().numpy()
    sample_count = int(fitted_sample_count)
    profile = BackgroundColourProfile(
        centre_lab=tuple(float(value) for value in centre.cpu().tolist()),
        scale_lab=tuple(float(value) for value in scale.cpu().tolist()),
        bgr_low=tuple(int(value) for value in low),
        bgr_high=tuple(int(value) for value in high),
        sample_count=sample_count,
        sample_fraction=sample_count / max(int(valid.sum().item()), 1),
        component_centres_lab=tuple(
            tuple(float(value) for value in row)
            for row in component_centres.cpu().tolist()
        ),
        component_scales_lab=tuple(
            tuple(float(value) for value in row)
            for row in component_scales.cpu().tolist()
        ),
        component_weights=tuple(
            float(value) for value in component_weights.cpu().tolist()
        ),
        refinement_iterations=refinement_rounds,
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
            else tuple(float(value) for value in excluded_weights.cpu().tolist())
        ),
        exclusion_strength=exclusion_strength,
        automatic_evidence_authority=float(automatic_evidence_authority),
        reviewed_sample_count=int(manual_reviewed_count),
    )
    other_probability = (
        None
        if excluded_membership is None
        else _lazy_u8(
            excluded_membership[None, None] * 255.0,
            "other colour probability",
        )
    )
    result = (
        _lazy_u8(
            likelihood[None, None] * 255.0, "background colour likelihood"
        ),
        mode,
        reference_count,
        profile,
    )
    if include_other_probability:
        result += (other_probability,)
    if include_reference_source_mask:
        result += (
            None
            if automatic_reference_mask is None
            else _lazy_u8(
                automatic_reference_mask[None, None].to(torch.uint8) * 255,
                "automatic background reference source",
            ),
        )
    return result


def empty_noise_frequency_profile(seed_diameter, settings):
    from seedvision.visualization.layers import NoiseFrequencyProfile

    scales = _noise_scales(seed_diameter, settings)
    return NoiseFrequencyProfile(
        band_scales_px=scales,
        descriptor_centre=(0.0,) * len(_NOISE_TEXTURE_DESCRIPTOR_NAMES),
        target_sample_count=0,
        descriptor_scale=(1.0,) * len(_NOISE_TEXTURE_DESCRIPTOR_NAMES),
        descriptor_names=_NOISE_TEXTURE_DESCRIPTOR_NAMES,
    )


def background_colour_profile_likelihood(
    source_tensor,
    valid_tensor,
    profile,
    settings,
    *,
    cuda_context=None,
):
    """Evaluate an existing background colour model over a bounded GPU crop.

    This intentionally reuses the exact fitted positive components from
    :func:`background_colour_likelihood`. It does
    not refit the model from the surrounding band or expand the canonical dish
    crop used by downstream calculations.
    """

    import torch

    context = cuda_context or CudaContext.resolve()
    source = source_tensor.to(device=context.device, dtype=torch.float32)
    valid = valid_tensor.to(device=context.device).bool()
    if source.ndim != 4 or source.shape[0] != 1 or source.shape[1] < 3:
        raise ValueError("source_tensor must be one NCHW BGR image.")
    if valid.shape != (1, 1, source.shape[-2], source.shape[-1]):
        raise ValueError("valid_tensor must be an NCHW mask matching the source.")
    lab = bgr_to_lab(source)[0].permute(1, 2, 0)
    metric_weights = torch.as_tensor(
        (1.0, 1.25, 1.25), device=context.device, dtype=lab.dtype
    )

    def evaluate(
        centres_values,
        scales_values,
        weights_values,
        frequency_power,
        *,
        combine_modes: str,
    ):
        centres = torch.as_tensor(
            centres_values, device=context.device, dtype=lab.dtype
        ).reshape(-1, 3)
        scales = torch.as_tensor(
            scales_values, device=context.device, dtype=lab.dtype
        ).reshape(-1, 3)
        weights = torch.as_tensor(
            weights_values, device=context.device, dtype=lab.dtype
        ).reshape(-1)
        if not int(centres.shape[0]):
            return torch.zeros(lab.shape[:2], device=context.device, dtype=lab.dtype)
        if scales.shape != centres.shape or weights.shape[0] != centres.shape[0]:
            raise ValueError("Background colour profile components are inconsistent.")
        adjusted = weights.clamp_min(1e-6).pow(max(0.0, float(frequency_power)))
        adjusted /= adjusted.max().clamp_min(1e-6)
        probability = torch.zeros(
            lab.shape[:2], device=context.device, dtype=lab.dtype
        )
        for start in range(0, int(centres.shape[0]), 4):
            stop = min(start + 4, int(centres.shape[0]))
            effective_scales = scales[start:stop] * max(
                0.05, float(settings.background_distribution_scale_multiplier)
            )
            delta = (
                lab[:, :, None, :] - centres[None, None, start:stop, :]
            ) / effective_scales[None, None, :, :]
            distance = torch.sum(
                delta.square() * metric_weights[None, None, None, :], dim=3
            )
            membership = torch.exp(-0.5 * distance)
            weighted = membership * adjusted[None, None, start:stop]
            if combine_modes == "maximum":
                probability = torch.maximum(
                    probability, torch.max(weighted, dim=2).values
                )
            else:
                probability += torch.sum(weighted, dim=2)
        return probability.clamp(0.0, 1.0)

    probability = evaluate(
        profile.component_centres_lab,
        profile.component_scales_lab,
        profile.component_weights,
        settings.background_frequency_weight_power,
        combine_modes="maximum",
    )
    probability *= valid[0, 0].float()
    return (
        _lazy_u8(
            probability[None, None] * 255.0,
            "surrounding-band background colour likelihood",
        ),
        _lazy_u8(
            valid.float() * 255.0,
            "surrounding-band background colour valid mask",
        ),
    )


def _integrate_directional_noise(stack, method: str):
    """Merge directional continuation probabilities on their GPU device."""

    import torch

    if method == "mean":
        return stack.mean(dim=0)
    if method == "minimum":
        return stack.min(dim=0).values
    if method == "median":
        return stack.median(dim=0).values
    if method == "1st tertile":
        # The first tertile is the exact one-third quantile: one third of the
        # directional continuations lie below it and two thirds lie above it.
        # Interpolate the two adjacent order statistics explicitly. This is
        # numerically equivalent to torch.quantile's linear rule but avoids a
        # full sorted copy of the 24-direction full-raster stack on CUDA.
        position = (int(stack.shape[0]) - 1) / 3.0
        lower_index = int(np.floor(position))
        upper_index = int(np.ceil(position))
        lower = torch.kthvalue(stack, lower_index + 1, dim=0).values
        if lower_index == upper_index:
            return lower
        upper = torch.kthvalue(stack, upper_index + 1, dim=0).values
        return torch.lerp(lower, upper, position - lower_index)
    if method == "maximum":
        return stack.max(dim=0).values
    raise ValueError(f"Unknown directional noise integration method: {method!r}.")


_NOISE_TEXTURE_DESCRIPTOR_NAMES = tuple(
    f"{band} {measurement}"
    for band in ("fine", "medium", "coarse")
    for measurement in (
        "residual RMS",
        "principal-axis variation",
        "cross-axis variation",
    )
)


def _noise_texture_features(lab, scales):
    """Return colour-independent multiscale, orientation-aware texture features.

    Each frequency residual contributes its existing isotropic RMS plus the two
    eigenvalues of a local spatial structure tensor.  The latter describe
    variation along and across the locally dominant texture orientation without
    making the classifier sensitive to the absolute rotation of a seed.
    """

    import torch
    import torch.nn.functional as functional

    fine = gaussian_blur(lab, max(0.55, scales[0]))
    medium = gaussian_blur(lab, scales[1])
    coarse = gaussian_blur(lab, scales[2])
    residuals = (lab - fine, fine - medium, medium - coarse)
    features = []
    for residual, sigma in zip(residuals, scales, strict=True):
        local_sigma = max(0.65, min(2.5, sigma * 0.35))
        residual_energy = torch.mean(residual.square(), dim=1, keepdim=True)
        residual_energy = gaussian_blur(residual_energy, local_sigma)

        padded_x = functional.pad(residual, (1, 1, 0, 0), mode="replicate")
        padded_y = functional.pad(residual, (0, 0, 1, 1), mode="replicate")
        gradient_x = 0.5 * (padded_x[..., 2:] - padded_x[..., :-2])
        gradient_y = 0.5 * (padded_y[..., 2:, :] - padded_y[..., :-2, :])
        tensor_xx = gaussian_blur(
            torch.mean(gradient_x.square(), dim=1, keepdim=True),
            local_sigma,
        )
        tensor_yy = gaussian_blur(
            torch.mean(gradient_y.square(), dim=1, keepdim=True),
            local_sigma,
        )
        tensor_xy = gaussian_blur(
            torch.mean(gradient_x * gradient_y, dim=1, keepdim=True),
            local_sigma,
        )
        trace = tensor_xx + tensor_yy
        discriminant = torch.sqrt(
            ((tensor_xx - tensor_yy).square() + 4.0 * tensor_xy.square())
            .clamp_min(0.0)
        )
        principal = 0.5 * (trace + discriminant)
        cross_axis = 0.5 * (trace - discriminant).clamp_min(0.0)
        features.extend(
            (
                torch.log1p(torch.sqrt(residual_energy.clamp_min(1e-10)) * 255.0),
                torch.log1p(torch.sqrt(principal.clamp_min(1e-10)) * 255.0),
                torch.log1p(torch.sqrt(cross_axis.clamp_min(1e-10)) * 255.0),
            )
        )
    return torch.cat(features, dim=1)


def _reference_material_feature_tensor(
    lab,
    edge,
    ridge,
    noise_channels,
    *,
    context_sigma: float,
):
    """Build the production material/trait prototype descriptor tensor.

    Library extraction calls this same seam.  Keeping local residual units,
    channel order, and normalization in one function prevents a persisted bank
    from being trained in a subtly different feature space than runtime maps.
    """

    import torch

    if len(noise_channels) != 6:
        raise ValueError("Material descriptors require three darkness and three colour bands.")
    local_lab = gaussian_blur(lab, max(0.7, float(context_sigma)))
    residual = lab - local_lab
    colour_residual = torch.sqrt(
        residual[:, 1:2].square() + residual[:, 2:3].square()
    )
    local_edge = gaussian_blur(edge, max(0.7, float(context_sigma))).clamp(0.0, 1.0)
    return torch.cat(
        (
            lab[:, 0:1] / 255.0,
            (lab[:, 1:2] - 128.0) / 128.0,
            (lab[:, 2:3] - 128.0) / 128.0,
            *noise_channels,
            edge,
            ridge,
            residual[:, 0:1].abs() / 40.0,
            colour_residual / 55.0,
            local_edge,
        ),
        dim=1,
    )


def _noise_target_distance(values, centre, scale):
    """Return mean robust standardized distance from one target class."""

    return ((values - centre) / scale).square().mean(
        dim=-1, keepdim=True
    )


def _noise_target_calibration(values, centre, scale):
    """Fit a target-only compatibility scale from positive examples.

    The 95th-percentile positive distance is the half-support distance. A
    minimum of one robust squared scale keeps a nearly uniform reference from
    creating a numerically brittle zero-width acceptance function.
    """

    import torch

    distances = _noise_target_distance(values, centre, scale).reshape(-1)
    if not int(distances.numel()):
        return 0.0, 1.0
    median = float(torch.quantile(distances, 0.50).item())
    half_distance = max(1.0, float(torch.quantile(distances, 0.95).item()))
    return median, half_distance


def _noise_target_compatibility(values, centre, scale, half_distance):
    """Map target distance to independent class compatibility in ``[0, 1]``."""

    import torch

    distance = _noise_target_distance(values, centre, scale)
    return torch.exp(
        -float(np.log(2.0)) * distance / max(float(half_distance), 1e-6)
    )


def _library_noise_compatibility(values, bank, context):
    """Evaluate source-balanced immutable target profiles in production units."""

    import torch

    if bank is None or not len(bank.centres):
        return torch.zeros_like(values[..., :1])
    if tuple(bank.feature_names) != _NOISE_TEXTURE_DESCRIPTOR_NAMES:
        raise ValueError("Species foreground-noise bank has an incompatible descriptor schema.")
    centres = torch.as_tensor(
        np.asarray(bank.centres), device=context.device, dtype=values.dtype
    )
    scales = torch.as_tensor(
        np.asarray(bank.scales), device=context.device, dtype=values.dtype
    ).clamp_min(1e-4)
    halves = torch.as_tensor(
        np.asarray(bank.half_distances), device=context.device, dtype=values.dtype
    ).clamp_min(1e-6)
    weights = torch.as_tensor(
        np.asarray(bank.weights), device=context.device, dtype=values.dtype
    )
    best = torch.zeros_like(values[..., :1])
    for start in range(0, int(centres.shape[0]), 8):
        distance = torch.mean(
            ((values[..., None, :] - centres[start : start + 8])
             / scales[start : start + 8]).square(),
            dim=-1,
        )
        compatibility = torch.exp(
            -float(np.log(2.0))
            * distance
            / halves[start : start + 8]
        )
        support = weights[start : start + 8]
        support = 0.78 + 0.22 * torch.sqrt(
            support / support.max().clamp_min(1e-6)
        )
        best = torch.maximum(
            best, (compatibility * support).amax(dim=-1, keepdim=True)
        )
    return best


def noise_frequency_background_likelihood(
    crop,
    valid_mask,
    colour_likelihood,
    seed_diameter,
    settings,
    *,
    background_reference_points=(),
    foreground_reference_points=(),
    background_reference_mask=None,
    automatic_target_reference_mask=None,
    automatic_target_authority=1.0,
    external_target_source_tensor=None,
    external_target_valid_tensor=None,
    automatic_nontarget_reference_mask=None,
    foreground_reference_mask=None,
    target_exclusion_mask=None,
    other_reference_mask=None,
    other_colour_likelihood=None,
    target_name="background",
    reference_radius=3,
    cuda_context=None,
    source_tensor=None,
    lab_tensor=None,
    valid_tensor=None,
    target_reference_precedence=False,
    require_target_reference=False,
    include_other_probability=False,
    library_target_bank=None,
    reference_source_mode="Current image only",
    current_reference_weight=1.0,
):
    from seedvision.visualization.layers import NoiseFrequencyProfile
    import torch
    import torch.nn.functional as functional

    context = cuda_context or CudaContext.resolve()
    full_source = image_to_tensor(crop, context) if source_tensor is None else source_tensor
    full_lab = (bgr_to_lab(full_source) if lab_tensor is None else lab_tensor) / 255.0
    full_valid = (
        image_to_tensor(valid_mask, context) > 0
        if valid_tensor is None
        else valid_tensor.bool()
    )
    full_colour = (
        full_valid.float() * 0.5
        if colour_likelihood is None
        else _raster_tensor(colour_likelihood, context, normalized=True)
    )
    # Retain the legacy argument for direct-call compatibility. Texture
    # evidence is deliberately independent of every colour probability.
    del other_colour_likelihood
    source_height, source_width = full_valid.shape[-2:]
    full_external_source = (
        None
        if external_target_source_tensor is None
        else external_target_source_tensor.to(
            device=context.device, dtype=full_source.dtype
        )
    )
    full_external_valid = (
        None
        if external_target_valid_tensor is None
        else external_target_valid_tensor.to(device=context.device).bool()
    )
    if (full_external_source is None) != (full_external_valid is None):
        raise ValueError(
            "External background texture source and valid mask must be supplied together."
        )
    if full_external_source is not None and (
        full_external_source.ndim != 4
        or full_external_source.shape[0] != 1
        or full_external_source.shape[1] < 3
        or full_external_valid.shape
        != (
            1,
            1,
            full_external_source.shape[-2],
            full_external_source.shape[-1],
        )
    ):
        raise ValueError(
            "External background texture source must be NCHW with a matching mask."
        )
    # The background-named inputs represent the target class. The foreground
    # wrapper swaps its painted masks into those positions so both classifiers
    # share the same direct-supervision implementation.
    full_target_reference = _point_mask(
        source_height,
        source_width,
        background_reference_points,
        reference_radius,
        context,
    )
    full_nontarget_reference = _point_mask(
        source_height,
        source_width,
        foreground_reference_points,
        reference_radius,
        context,
    )
    if background_reference_mask is not None:
        full_target_reference |= image_to_tensor(
            np.asarray(background_reference_mask, np.uint8), context
        )[0, 0] > 0
    if foreground_reference_mask is not None:
        full_nontarget_reference |= image_to_tensor(
            np.asarray(foreground_reference_mask, np.uint8), context
        )[0, 0] > 0
    if automatic_nontarget_reference_mask is not None:
        if hasattr(automatic_nontarget_reference_mask, "gpu_tensor"):
            automatic_nontarget_values = (
                automatic_nontarget_reference_mask.gpu_tensor(
                    device=context.device
                )
            )
            if automatic_nontarget_values.ndim == 2:
                automatic_nontarget_values = automatic_nontarget_values[None, None]
            elif automatic_nontarget_values.ndim == 3:
                automatic_nontarget_values = automatic_nontarget_values[None]
        else:
            automatic_nontarget_values = image_to_tensor(
                np.asarray(automatic_nontarget_reference_mask, np.uint8), context
            )
        full_nontarget_reference |= automatic_nontarget_values[0, 0] > 0
    full_target_reference &= full_valid[0, 0]
    full_nontarget_reference &= full_valid[0, 0]
    if target_reference_precedence:
        full_nontarget_reference &= ~full_target_reference
    else:
        # Foreground wins accidental positive-mask overlap, matching the colour
        # models' foreground-precedence convention.
        full_target_reference &= ~full_nontarget_reference
    full_target_exclusion = torch.zeros_like(full_target_reference)
    if target_exclusion_mask is not None:
        full_target_exclusion = image_to_tensor(
            np.asarray(target_exclusion_mask, np.uint8), context
        )[0, 0] > 0
        full_target_exclusion &= full_valid[0, 0]
        full_target_reference &= ~full_target_exclusion
    full_manual_target_reference = full_target_reference.clone()
    full_automatic_target_reference = torch.zeros_like(
        full_target_reference
    )
    if automatic_target_reference_mask is not None:
        if hasattr(automatic_target_reference_mask, "gpu_tensor"):
            automatic_values = automatic_target_reference_mask.gpu_tensor(
                device=context.device
            )
            if automatic_values.ndim == 2:
                automatic_values = automatic_values[None, None]
            elif automatic_values.ndim == 3:
                automatic_values = automatic_values[None]
            automatic_target = automatic_values[0, 0] > 0
        else:
            automatic_target = image_to_tensor(
                automatic_target_reference_mask, context
            )[0, 0] > 0
        full_automatic_target_reference = automatic_target & (
            full_valid[0, 0]
            & ~full_nontarget_reference
            & ~full_target_exclusion
            & ~full_manual_target_reference
        )
        full_target_reference |= full_automatic_target_reference
    if (
        require_target_reference
        and not bool(full_target_reference.any().item())
        and library_target_bank is None
    ):
        result = (
            _lazy_u8(
                full_colour * 0.0,
                f"no user-authored {target_name} texture reference",
            ),
            empty_noise_frequency_profile(seed_diameter, settings),
            (),
            (),
        )
        return result + (None, None) if include_other_probability else result
    full_other_reference = torch.zeros_like(full_target_reference)
    if other_reference_mask is not None:
        full_other_reference = image_to_tensor(
            np.asarray(other_reference_mask, np.uint8), context
        )[0, 0] > 0
        full_other_reference &= full_valid[0, 0]
    elif bool(full_target_exclusion.any().item()):
        # Backward-compatible direct-call convention: historically the target
        # exclusion mask also supplied the separately modelled Other class.
        full_other_reference = full_target_exclusion.clone()

    largest_work_dimension = max(source_height, source_width)
    if full_external_source is not None:
        largest_work_dimension = max(
            largest_work_dimension,
            int(full_external_source.shape[-2]),
            int(full_external_source.shape[-1]),
        )
    work_scale = min(
        1.0,
        float(settings.noise_working_maximum_dimension)
        / largest_work_dimension,
    )
    height = max(8, round(source_height * work_scale))
    width = max(8, round(source_width * work_scale))
    if (height, width) == (source_height, source_width):
        source, lab, valid, colour = (
            full_source,
            full_lab,
            full_valid,
            full_colour,
        )
        manual_target_reference = full_manual_target_reference
        automatic_target_reference = full_automatic_target_reference
        target_reference = (
            manual_target_reference | automatic_target_reference
        )
        nontarget_reference = full_nontarget_reference
        target_exclusion = full_target_exclusion
        other_reference = full_other_reference
    else:
        source = functional.interpolate(
            full_source, (height, width), mode="area"
        )
        lab = functional.interpolate(
            full_lab, (height, width), mode="bilinear", align_corners=False
        )
        valid = functional.interpolate(
            full_valid.float(), (height, width), mode="nearest"
        ) > 0.5
        colour = functional.interpolate(
            full_colour, (height, width), mode="bilinear", align_corners=False
        )
        # Preserve even narrow painted regions when the noise classifier works
        # at a bounded resolution; nearest-neighbour downsampling could drop a
        # small ground-truth stroke entirely.
        manual_target_reference = functional.adaptive_max_pool2d(
            full_manual_target_reference[None, None].float(), (height, width)
        )[0, 0] > 0.0
        automatic_target_reference = functional.adaptive_max_pool2d(
            full_automatic_target_reference[None, None].float(),
            (height, width),
        )[0, 0] > 0.0
        target_reference = (
            manual_target_reference | automatic_target_reference
        )
        nontarget_reference = functional.adaptive_max_pool2d(
            full_nontarget_reference[None, None].float(), (height, width)
        )[0, 0] > 0.0
        target_exclusion = functional.adaptive_max_pool2d(
            full_target_exclusion[None, None].float(), (height, width)
        )[0, 0] > 0.0
        other_reference = functional.adaptive_max_pool2d(
            full_other_reference[None, None].float(), (height, width)
        )[0, 0] > 0.0
    external_source = None
    external_valid = None
    if full_external_source is not None:
        external_height = max(
            8, round(int(full_external_source.shape[-2]) * work_scale)
        )
        external_width = max(
            8, round(int(full_external_source.shape[-1]) * work_scale)
        )
        if (external_height, external_width) == tuple(
            int(value) for value in full_external_source.shape[-2:]
        ):
            external_source = full_external_source
            external_valid = full_external_valid
        else:
            external_source = functional.interpolate(
                full_external_source,
                (external_height, external_width),
                mode="area",
            )
            # The retained perimeter ring may be only a few working pixels
            # thick. Preserve its support while area-resampling the source.
            external_valid = functional.adaptive_max_pool2d(
                full_external_valid.float(),
                (external_height, external_width),
            ) > 0.0
    reported_scales = _noise_scales(seed_diameter, settings)
    scales = _noise_scales(seed_diameter * work_scale, settings)
    feature = _noise_texture_features(lab, scales)
    external_feature_values = None
    if (
        external_source is not None
        and external_valid is not None
        and bool(external_valid.any().item())
    ):
        external_lab = bgr_to_lab(external_source) / 255.0
        external_feature = _noise_texture_features(external_lab, scales)
        external_feature_values = external_feature.permute(0, 2, 3, 1)[
            external_valid.permute(0, 2, 3, 1).expand(
                -1, -1, -1, int(external_feature.shape[1])
            )
        ].reshape(-1, int(external_feature.shape[1]))
    eligible = valid
    if not bool(eligible.any().item()):
        result = (
            _lazy_u8(full_colour * 0.0, f"excluded {target_name} noise likelihood"),
            empty_noise_frequency_profile(seed_diameter, settings),
            (),
            (),
        )
        return result + (None, None) if include_other_probability else result
    if target_reference_precedence:
        nontarget_reference &= ~target_reference
    else:
        manual_target_reference &= ~nontarget_reference
        automatic_target_reference &= ~nontarget_reference
        target_reference = (
            manual_target_reference | automatic_target_reference
        )
    has_external_target = bool(
        external_feature_values is not None
        and int(external_feature_values.shape[0]) > 0
    )
    texture_energy = feature.mean(dim=1, keepdim=True)
    automatic_target = torch.zeros_like(eligible)
    if (
        not require_target_reference
        and reference_source_mode != "Species library only"
        and not bool(target_reference.any().item())
        and not has_external_target
    ):
        target_candidates = (
            eligible
            & ~target_exclusion[None, None]
            & ~nontarget_reference[None, None]
        )
        if bool(target_candidates.any().item()):
            threshold = torch.quantile(texture_energy[target_candidates], 0.25)
            automatic_target = target_candidates & (texture_energy <= threshold)
    if bool(target_reference.any().item()):
        # Painted target regions and any exact retained perimeter-ring pixels
        # inside this raster supply actual texture observations.
        confident_background = target_reference[None, None]
    elif has_external_target:
        # The exact outside-dish annulus has different coordinates. Its compact
        # feature rows are consumed below instead of inventing an aligned mask.
        confident_background = torch.zeros_like(automatic_target)
    else:
        confident_background = automatic_target
    feature_values = feature.permute(0, 2, 3, 1)

    def mask_feature_values(mask):
        feature_count = int(feature_values.shape[-1])
        return feature_values[
            mask.permute(0, 2, 3, 1).expand(-1, -1, -1, feature_count)
        ].reshape(-1, feature_count)

    def resampled_rows(rows, count):
        if int(rows.shape[0]) == count:
            return rows
        indices = torch.linspace(
            0,
            int(rows.shape[0]) - 1,
            count,
            device=rows.device,
        ).round().long()
        return rows[indices]

    manual_fit_mask = manual_target_reference[None, None]
    automatic_fit_mask = automatic_target_reference[None, None]
    external_fit_values = external_feature_values
    if (
        external_fit_values is not None
        and int(external_fit_values.shape[0]) > 32768
    ):
        external_fit_values = resampled_rows(external_fit_values, 32768)
    automatic_values = (
        external_fit_values
        if external_fit_values is not None and int(external_fit_values.shape[0])
        else mask_feature_values(automatic_fit_mask)
    )
    if bool(manual_fit_mask.any().item()) and int(automatic_values.shape[0]):
        manual_values = mask_feature_values(manual_fit_mask)
        fit_count = min(
            32768,
            int(manual_values.shape[0]) + int(automatic_values.shape[0]),
        )
        authority = max(0.0, float(automatic_target_authority))
        manual_count = max(
            1, round(fit_count / max(1.0 + authority, 1e-6))
        )
        manual_count = min(manual_count, fit_count - 1)
        automatic_count = fit_count - manual_count
        target_values = torch.cat(
            (
                resampled_rows(manual_values, manual_count),
                resampled_rows(automatic_values, automatic_count),
            ),
            dim=0,
        )
    elif external_fit_values is not None and int(external_fit_values.shape[0]):
        target_values = external_fit_values
    else:
        target_values = mask_feature_values(confident_background)
        if int(target_values.shape[0]) > 32768:
            target_values = resampled_rows(target_values, 32768)
    has_current_target = int(target_values.shape[0]) > 0
    if has_current_target:
        target_center, target_scale = _robust_tensor_distribution(target_values)
    else:
        target_center = torch.zeros(
            int(feature.shape[1]), device=context.device, dtype=feature.dtype
        )
        target_scale = torch.ones_like(target_center)
    positive_class_name = (
        "Foreground" if target_name == "foreground" else "Background"
    )
    target_distance_median, target_half_distance = (
        _noise_target_calibration(target_values, target_center, target_scale)
        if has_current_target
        else (0.0, 1.0)
    )
    values = feature.permute(0, 2, 3, 1)
    texture_probability = (
        _noise_target_compatibility(
            values,
            target_center,
            target_scale,
            target_half_distance,
        ).permute(0, 3, 1, 2)
        if has_current_target
        else torch.zeros_like(valid.float())
    )
    library_texture_probability = _library_noise_compatibility(
        values, library_target_bank, context
    ).permute(0, 3, 1, 2)
    if reference_source_mode == "Current image only":
        pass
    elif reference_source_mode == "Species library only":
        texture_probability = library_texture_probability
    elif reference_source_mode == "Species library + current image":
        texture_probability = 1.0 - (
            (1.0 - library_texture_probability.clamp(0.0, 1.0))
            * (1.0 - texture_probability.clamp(0.0, 1.0)).pow(
                float(current_reference_weight)
            )
        )
    else:
        raise ValueError(f"Unknown foreground-noise reference source: {reference_source_mode}")
    # Reference pixels fit a global compatibility function; their coordinates
    # are never overwritten. The output depends only on this target class.
    base = texture_probability.clamp(1e-4, 1.0 - (1.0 / 255.0)) * valid

    other_base = None
    other_profile = None
    if bool(other_reference.any().item()):
        # Other is a target-only positive texture class. Foreground and
        # Background references do not participate in either its fit or score.
        other_reference &= valid[0, 0]
        confident_other = other_reference[None, None]
        other_values = mask_feature_values(confident_other)
        if int(other_values.shape[0]) > 32768:
            other_values = resampled_rows(other_values, 32768)
        other_center, other_scale = _robust_tensor_distribution(other_values)
        other_distance_median, other_half_distance = (
            _noise_target_calibration(
                other_values, other_center, other_scale
            )
        )
        other_texture_probability = _noise_target_compatibility(
            values,
            other_center,
            other_scale,
            other_half_distance,
        ).permute(0, 3, 1, 2)
        other_base = other_texture_probability.clamp(
            1e-4, 1.0 - (1.0 / 255.0)
        ) * valid
        other_profile = NoiseFrequencyProfile(
            band_scales_px=reported_scales,
            descriptor_centre=tuple(
                float(value) for value in other_center.cpu().tolist()
            ),
            target_sample_count=int(confident_other.sum().item()),
            descriptor_scale=tuple(
                float(value) for value in other_scale.cpu().tolist()
            ),
            positive_class_name="Other",
            target_distance_median=other_distance_median,
            compatibility_half_distance=other_half_distance,
            descriptor_names=_NOISE_TEXTURE_DESCRIPTOR_NAMES,
        )
    yy, xx = torch.meshgrid(
        torch.arange(height, device=context.device, dtype=torch.float32),
        torch.arange(width, device=context.device, dtype=torch.float32),
        indexing="ij",
    )
    angles = tuple(float(angle) for angle in range(0, 360, settings.noise_direction_step_degrees))
    length = max(
        2.0,
        seed_diameter * work_scale * settings.noise_vector_length_fraction,
    )
    distances = torch.linspace(0.0, length, settings.noise_vector_sample_count, device=context.device)
    weights = torch.pow(
        torch.as_tensor(settings.noise_vector_decay, device=context.device),
        torch.arange(settings.noise_vector_sample_count, device=context.device),
    )
    weights /= weights.sum()
    def directional_refinement(current_base):
        # Preallocate one direction bank and finish it before evaluating another
        # class. Retaining two Python lists plus both stacked banks at 1280 px
        # would add hundreds of MiB to an otherwise bounded CUDA pass.
        stack = torch.empty(
            (len(angles), height, width),
            device=context.device,
            dtype=current_base.dtype,
        )
        for index, degrees in enumerate(angles):
            radians = np.deg2rad(degrees)
            distance_view = distances[:, None, None]
            sample_x = xx[None] + float(np.cos(radians)) * distance_view
            sample_y = yy[None] + float(np.sin(radians)) * distance_view
            sample_valid = bilinear_sample(valid.float(), sample_x, sample_y)
            weight_view = weights[:, None, None]
            support = (weight_view * sample_valid).sum(dim=0)
            sample = bilinear_sample(
                current_base, sample_x, sample_y
            ).clamp_min(1e-4)
            accumulated = (
                torch.log(sample) * weight_view * sample_valid
            ).sum(dim=0)
            stack[index] = (
                torch.exp(accumulated / support.clamp_min(0.15))
                * valid[0, 0]
            )
        return _integrate_directional_noise(
            stack, settings.noise_direction_integration
        )

    refined = directional_refinement(base)
    other_refined = (
        None
        if other_base is None
        else directional_refinement(other_base)
    )

    library_source_count = (
        0
        if library_target_bank is None
        else len(np.unique(np.asarray(library_target_bank.source_indices)))
    )
    library_profile_count = (
        0 if library_target_bank is None else len(library_target_bank.centres)
    )
    profile = NoiseFrequencyProfile(
        band_scales_px=reported_scales,
        descriptor_centre=tuple(
            float(value) for value in target_center.cpu().tolist()
        ),
        target_sample_count=int(target_values.shape[0]),
        descriptor_scale=tuple(
            float(value) for value in target_scale.cpu().tolist()
        ),
        positive_class_name=positive_class_name,
        target_distance_median=target_distance_median,
        compatibility_half_distance=target_half_distance,
        descriptor_names=_NOISE_TEXTURE_DESCRIPTOR_NAMES,
        reference_source_mode=reference_source_mode,
        current_target_sample_count=int(target_values.shape[0]),
        library_source_count=library_source_count,
        library_profile_count=library_profile_count,
    )
    def restore(item):
        values = item[None, None]
        if values.shape[-2:] != (source_height, source_width):
            values = functional.interpolate(
                values,
                (source_height, source_width),
                mode="bilinear",
                align_corners=False,
            )
        return values * full_valid.float()

    result = (
        _lazy_u8(restore(refined) * 255.0, f"{target_name} noise likelihood"),
        profile,
        (),
        angles,
    )
    other_result = (
        (
            None
            if other_refined is None
            else _lazy_u8(
                restore(other_refined) * 255.0,
                "other noise probability",
            )
        ),
        other_profile,
    )
    return result + other_result if include_other_probability else result


def surrounding_band_noise_likelihood(
    source_tensor,
    valid_tensor,
    seed_diameter,
    settings,
    profile,
    *,
    colour_likelihood=None,
    cuda_context=None,
):
    """Apply target-only texture compatibility to the sampled rim annulus."""

    import torch
    import torch.nn.functional as functional

    context = cuda_context or CudaContext.resolve()
    full_source = source_tensor.to(device=context.device, dtype=torch.float32)
    full_valid = valid_tensor.to(device=context.device).bool()
    source_height, source_width = full_source.shape[-2:]
    work_scale = min(
        1.0,
        float(settings.noise_working_maximum_dimension)
        / max(source_height, source_width),
    )
    height = max(8, round(source_height * work_scale))
    width = max(8, round(source_width * work_scale))
    if (height, width) == (source_height, source_width):
        source = full_source
        valid = full_valid
    else:
        source = functional.interpolate(full_source, (height, width), mode="area")
        valid = functional.interpolate(
            full_valid.float(), (height, width), mode="nearest"
        ) > 0.5
    lab = bgr_to_lab(source) / 255.0
    scales = _noise_scales(seed_diameter * work_scale, settings)
    features = _noise_texture_features(lab, scales)
    if len(profile.descriptor_centre) == 3:
        # Compatibility with synthetic/legacy in-memory profiles created before
        # orientation-aware descriptors were added.
        features = features[:, (0, 3, 6)]
    values = features.permute(0, 2, 3, 1)
    bg_center = torch.as_tensor(
        profile.descriptor_centre,
        device=context.device,
        dtype=values.dtype,
    )
    bg_scale = torch.as_tensor(
        profile.descriptor_scale,
        device=context.device,
        dtype=values.dtype,
    ).clamp_min(1e-4)
    texture_probability = _noise_target_compatibility(
        values,
        bg_center,
        bg_scale,
        getattr(profile, "compatibility_half_distance", 1.0),
    )[0, :, :, 0]
    probability = texture_probability
    probability = probability * valid[0, 0]
    raster = probability[None, None]
    if raster.shape[-2:] != (source_height, source_width):
        raster = functional.interpolate(
            raster,
            (source_height, source_width),
            mode="bilinear",
            align_corners=False,
        )
    raster *= full_valid.float()
    return (
        _lazy_u8(raster * 255.0, "surrounding-band noise likelihood"),
        _lazy_u8(
            full_valid.float() * 255.0,
            "surrounding-band noise valid mask",
        ),
    )


def _noise_scales(seed_diameter, settings):
    return (
        0.65,
        max(1.0, float(seed_diameter) * settings.noise_medium_scale_fraction),
        max(2.0, float(seed_diameter) * settings.noise_coarse_scale_fraction),
    )


def _robust_tensor_distribution(values):
    import torch

    if values.shape[0] > 200000:
        step = max(1, values.shape[0] // 200000)
        values = values[::step]
    centre = torch.median(values, dim=0).values
    scale = torch.median(torch.abs(values - centre), dim=0).values * 1.4826
    return centre, scale.clamp_min(0.045)


def _positive_percentile(torch, values, valid, percentile: float):
    """Return a stable positive response percentile without downloading values."""

    selected = values[valid.expand_as(values)]
    selected = selected[selected > 1e-8]
    if selected.numel() == 0:
        return torch.as_tensor(1.0, device=values.device, dtype=values.dtype)
    return torch.quantile(selected, float(percentile) / 100.0).clamp_min(1e-6)


def _restored_u8(
    functional,
    tensor,
    height: int,
    width: int,
    name: str,
    *,
    mode: str = "bilinear",
    multiplier: float = 255.0,
):
    """Restore a working tensor to the full crop while retaining it on-device."""

    values = tensor
    if values.ndim == 2:
        values = values[None, None]
    elif values.ndim == 3:
        values = values[None]
    if values.shape[-2:] != (height, width):
        kwargs = {} if mode == "nearest" else {"align_corners": False}
        values = functional.interpolate(values, (height, width), mode=mode, **kwargs)
    return _lazy_u8(values * multiplier, name)


def surface_directional_darkness_gradients(
    crop,
    valid_mask,
    seed_diameter,
    settings=None,
    *,
    cuda_context=None,
    lab_tensor=None,
    valid_tensor=None,
):
    """Find maximum query-to-target lightening and darkening slopes on rays."""

    from seedvision.visualization.layers import AnalysisLayerSettings
    import torch
    import torch.nn.functional as functional

    settings = settings or AnalysisLayerSettings()
    context = cuda_context or CudaContext.resolve()
    full_lab = (
        bgr_to_lab(image_to_tensor(crop, context))
        if lab_tensor is None
        else lab_tensor
    )
    full_valid = (
        image_to_tensor(valid_mask, context) > 0
        if valid_tensor is None
        else valid_tensor.bool()
    )
    source_height, source_width = full_valid.shape[-2:]
    work_scale = min(
        1.0,
        float(settings.surface_gradient_working_maximum_dimension)
        / max(source_height, source_width),
    )
    height = max(8, round(source_height * work_scale))
    width = max(8, round(source_width * work_scale))
    if (height, width) == (source_height, source_width):
        lab = full_lab
        valid = full_valid
    else:
        lab = functional.interpolate(
            full_lab, (height, width), mode="bilinear", align_corners=False
        )
        valid = functional.interpolate(
            full_valid.float(), (height, width), mode="nearest"
        ) > 0.5

    # OpenCV-compatible Lab tensors encode L*=0..100 as 0..255. Convert back so
    # the reported slope and downstream ceilings have stable physical units.
    lightness = lab[:, 0:1] * (100.0 / 255.0)
    lightness = gaussian_blur(
        lightness,
        max(0.10, float(settings.surface_gradient_blur_sigma) * work_scale),
    )
    yy, xx = torch.meshgrid(
        torch.arange(height, device=context.device, dtype=torch.float32),
        torch.arange(width, device=context.device, dtype=torch.float32),
        indexing="ij",
    )
    ray_length = max(
        1.0,
        float(seed_diameter)
        * work_scale
        * float(settings.surface_gradient_radius_fraction),
    )
    first_distance = min(1.0, ray_length)
    distances = torch.linspace(
        first_distance,
        ray_length,
        int(settings.surface_gradient_sample_count),
        device=context.device,
    )
    # Convert each working-grid displacement back to original-image pixels.
    physical_distances = distances / max(work_scale, 1e-6)
    query = lightness[0, 0]
    query_valid = valid[0, 0]
    lightening = torch.zeros_like(query)
    darkening = torch.zeros_like(query)
    lightening_angle = torch.zeros_like(query)
    darkening_angle = torch.zeros_like(query)
    distance_view = distances[:, None, None]
    denominator = physical_distances[:, None, None].clamp_min(1e-6)
    for degrees in range(
        0, 360, int(settings.surface_gradient_direction_step_degrees)
    ):
        radians = np.deg2rad(float(degrees))
        sample_x = xx[None] + float(np.cos(radians)) * distance_view
        sample_y = yy[None] + float(np.sin(radians)) * distance_view
        target = bilinear_sample(lightness, sample_x, sample_y)
        target_valid = bilinear_sample(valid.float(), sample_x, sample_y) > 0.999
        signed_slope = (target - query[None]) / denominator
        unavailable = torch.full_like(target, -float("inf"))
        local_lightening = torch.relu(
            torch.where(target_valid, signed_slope, unavailable).max(dim=0).values
        )
        local_darkening = torch.relu(
            torch.where(target_valid, -signed_slope, unavailable).max(dim=0).values
        )
        lightening_update = local_lightening > lightening
        darkening_update = local_darkening > darkening
        lightening = torch.where(lightening_update, local_lightening, lightening)
        darkening = torch.where(darkening_update, local_darkening, darkening)
        lightening_angle = torch.where(
            lightening_update,
            torch.full_like(lightening_angle, float(degrees)),
            lightening_angle,
        )
        darkening_angle = torch.where(
            darkening_update,
            torch.full_like(darkening_angle, float(degrees)),
            darkening_angle,
        )

    lightening *= query_valid
    darkening *= query_valid
    combined = torch.cat((lightening[None, None], darkening[None, None]), dim=0)
    combined_valid = query_valid[None, None].expand_as(combined)
    normalization = _positive_percentile(
        torch,
        combined,
        combined_valid,
        settings.surface_gradient_normalization_percentile,
    )
    gamma = float(settings.surface_gradient_strength_gamma)
    lightening_strength = (
        (lightening / normalization).clamp(0.0, 1.0).pow(gamma) * query_valid
    )
    darkening_strength = (
        (darkening / normalization).clamp(0.0, 1.0).pow(gamma) * query_valid
    )
    lightening_hue = torch.remainder(lightening_angle * 0.5, 180.0) * query_valid
    darkening_hue = torch.remainder(darkening_angle * 0.5, 180.0) * query_valid
    return SurfaceGradientProducts(
        source_height=source_height,
        source_width=source_width,
        work_scale=work_scale,
        valid=query_valid,
        lightening_magnitude=lightening,
        darkening_magnitude=darkening,
        lightening_strength=lightening_strength,
        darkening_strength=darkening_strength,
        lightening_hue_float=lightening_hue,
        darkening_hue_float=darkening_hue,
        lightening_strength_raster=_restored_u8(
            functional,
            lightening_strength,
            source_height,
            source_width,
            "maximum lightening surface slope",
        ),
        darkening_strength_raster=_restored_u8(
            functional,
            darkening_strength,
            source_height,
            source_width,
            "maximum darkening surface slope",
        ),
        lightening_hue_raster=_restored_u8(
            functional,
            lightening_hue,
            source_height,
            source_width,
            "lightening target direction",
            mode="nearest",
            multiplier=1.0,
        ),
        darkening_hue_raster=_restored_u8(
            functional,
            darkening_hue,
            source_height,
            source_width,
            "darkening target direction",
            mode="nearest",
            multiplier=1.0,
        ),
    )


def upper_magnitude_surface_gradient(
    products: SurfaceGradientProducts,
    maximum_slope: float,
    polarity: str,
):
    """Zero surface responses above a raw L*/original-pixel ceiling."""

    import torch.nn.functional as functional

    if polarity == "lightening":
        magnitude = products.lightening_magnitude
        strength = products.lightening_strength
        hue = products.lightening_hue_float
    elif polarity == "darkening":
        magnitude = products.darkening_magnitude
        strength = products.darkening_strength
        hue = products.darkening_hue_float
    else:
        raise ValueError("Surface-gradient polarity must be lightening or darkening.")
    retained = (magnitude > 0.0) & (magnitude <= float(maximum_slope)) & products.valid
    filtered_strength = strength * retained
    filtered_hue = hue * retained
    return (
        _restored_u8(
            functional,
            filtered_strength,
            products.source_height,
            products.source_width,
            f"weak {polarity} surface slope",
        ),
        _restored_u8(
            functional,
            filtered_hue,
            products.source_height,
            products.source_width,
            f"weak {polarity} target direction",
            mode="nearest",
            multiplier=1.0,
        ),
    )


def multiscale_frequency_noise_masks(
    crop,
    valid_mask,
    seed_diameter,
    settings=None,
    *,
    cuda_context=None,
    lab_tensor=None,
    valid_tensor=None,
):
    """Calculate local darkness and chroma RMS energy in three frequency bands."""

    from seedvision.visualization.layers import AnalysisLayerSettings
    import torch
    import torch.nn.functional as functional

    settings = settings or AnalysisLayerSettings()
    context = cuda_context or CudaContext.resolve()
    full_lab = (
        bgr_to_lab(image_to_tensor(crop, context))
        if lab_tensor is None
        else lab_tensor
    )
    full_valid = (
        image_to_tensor(valid_mask, context) > 0
        if valid_tensor is None
        else valid_tensor.bool()
    )
    source_height, source_width = full_valid.shape[-2:]
    work_scale = min(
        1.0,
        float(settings.frequency_noise_working_maximum_dimension)
        / max(source_height, source_width),
    )
    height = max(8, round(source_height * work_scale))
    width = max(8, round(source_width * work_scale))
    if (height, width) == (source_height, source_width):
        lab = full_lab
        valid = full_valid
    else:
        lab = functional.interpolate(
            full_lab, (height, width), mode="bilinear", align_corners=False
        )
        valid = functional.interpolate(
            full_valid.float(), (height, width), mode="nearest"
        ) > 0.5
    valid_float = valid.float()
    # Work in physical Lab units so darkness and chroma energies remain stable
    # across images even though each display mask is normalized independently.
    lab_physical = torch.cat(
        (
            lab[:, 0:1] * (100.0 / 255.0),
            lab[:, 1:3] - 128.0,
        ),
        dim=1,
    )
    reported_scales = (
        float(seed_diameter) * settings.frequency_noise_fine_scale_fraction,
        float(seed_diameter) * settings.frequency_noise_medium_scale_fraction,
        float(seed_diameter) * settings.frequency_noise_coarse_scale_fraction,
    )
    scales = tuple(max(0.35, value * work_scale) for value in reported_scales)
    context_sigma = max(
        0.35,
        float(seed_diameter)
        * work_scale
        * settings.frequency_noise_context_fraction,
    )

    def masked_blur(values, sigma):
        numerator = gaussian_blur(values * valid_float, sigma)
        denominator = gaussian_blur(valid_float, sigma).clamp_min(1e-4)
        return numerator / denominator

    fine = masked_blur(lab_physical, scales[0])
    medium = masked_blur(lab_physical, scales[1])
    coarse = masked_blur(lab_physical, scales[2])
    residuals = (lab_physical - fine, fine - medium, medium - coarse)
    darkness_masks = []
    colour_masks = []
    band_names = ("fine", "medium", "coarse")
    gamma = float(settings.frequency_noise_strength_gamma)
    for residual, band_name in zip(residuals, band_names, strict=True):
        darkness_energy = torch.sqrt(
            masked_blur(residual[:, 0:1].square(), context_sigma).clamp_min(0.0)
        ) * valid_float
        colour_energy = torch.sqrt(
            masked_blur(
                residual[:, 1:3].square().mean(dim=1, keepdim=True),
                context_sigma,
            ).clamp_min(0.0)
        ) * valid_float
        darkness_norm = _positive_percentile(
            torch,
            darkness_energy,
            valid,
            settings.frequency_noise_normalization_percentile,
        ).clamp_min(0.05)
        colour_norm = _positive_percentile(
            torch,
            colour_energy,
            valid,
            settings.frequency_noise_normalization_percentile,
        ).clamp_min(0.05)
        darkness_strength = (
            darkness_energy / darkness_norm
        ).clamp(0.0, 1.0).pow(gamma) * valid_float
        colour_strength = (
            colour_energy / colour_norm
        ).clamp(0.0, 1.0).pow(gamma) * valid_float
        darkness_masks.append(
            _restored_u8(
                functional,
                darkness_strength,
                source_height,
                source_width,
                f"{band_name} darkness noise energy",
            )
        )
        colour_masks.append(
            _restored_u8(
                functional,
                colour_strength,
                source_height,
                source_width,
                f"{band_name} colour noise energy",
            )
        )
    return FrequencyNoiseMaskProducts(
        band_scales_px=reported_scales,
        darkness_masks=tuple(darkness_masks),
        colour_masks=tuple(colour_masks),
    )


def directional_edges(
    crop,
    valid_mask,
    settings=None,
    *,
    cuda_context=None,
    source_tensor=None,
    lab_tensor=None,
    valid_tensor=None,
    despeckled_flattened_tensor=None,
    wavelet_details=(),
    wavelet_residual=None,
):
    from seedvision.visualization.layers import AnalysisLayerSettings
    import torch
    import torch.nn.functional as functional

    settings = settings or AnalysisLayerSettings()
    context = cuda_context or CudaContext.resolve()
    source = (
        image_to_tensor(crop, context)
        if source_tensor is None
        else source_tensor
    )
    base_lab = bgr_to_lab(source) if lab_tensor is None else lab_tensor
    valid = (
        image_to_tensor(valid_mask, context) > 0
        if valid_tensor is None
        else valid_tensor.bool()
    )

    def derivative(channel):
        method = settings.edge_gradient_method
        if method == "scharr":
            return gradient_magnitude(channel, scharr=True)
        if method == "sobel":
            return gradient_magnitude(channel, scharr=False)
        if method == "prewitt":
            kernel_values = ((-1.0, 0.0, 1.0),) * 3
            divisor = 6.0
        else:
            kernel_values = (
                (0.0, 0.0, 0.0),
                (-1.0, 0.0, 1.0),
                (0.0, 0.0, 0.0),
            )
            divisor = 2.0
        kernel_x = torch.tensor(
            kernel_values, device=channel.device, dtype=channel.dtype
        )[None, None] / divisor
        kernel_y = kernel_x.transpose(-1, -2)
        padded = functional.pad(channel, (1, 1, 1, 1), mode="replicate")
        gx_value = functional.conv2d(padded, kernel_x)
        gy_value = functional.conv2d(padded, kernel_y)
        return (
            torch.sqrt(gx_value.square() + gy_value.square() + 1e-8),
            gx_value,
            gy_value,
        )

    weights = (1.0, settings.edge_chroma_weight, settings.edge_chroma_weight)

    def source_vector(channels):
        if settings.edge_blur_sigma > 0.0:
            channels = gaussian_blur(channels, settings.edge_blur_sigma)
        components = [
            derivative(channels[:, index : index + 1]) for index in range(3)
        ]
        gx_value = sum(
            item[1] * weight
            for item, weight in zip(components, weights, strict=True)
        )
        gy_value = sum(
            item[2] * weight
            for item, weight in zip(components, weights, strict=True)
        )
        return gx_value, gy_value

    vectors = []
    if settings.edge_gradient_include_original:
        if settings.edge_gradient_use_despeckled_flattened:
            if despeckled_flattened_tensor is None:
                raise ValueError(
                    "Despeckled edge input was selected but the illumination node "
                    "did not supply despeckled flattened grayscale."
                )
            flattened = (
                despeckled_flattened_tensor.to(
                    device=source.device, dtype=torch.float32
                )
                if torch.is_tensor(despeckled_flattened_tensor)
                else _raster_tensor(
                    despeckled_flattened_tensor, context
                ).to(device=source.device, dtype=torch.float32)
            )
            if flattened.ndim == 2:
                flattened = flattened[None, None]
            elif flattened.ndim == 3:
                flattened = flattened[None]
            if flattened.shape[1] != 1:
                flattened = flattened[:, :1]
            # Local-lighting tensors use [0, 1].  L* uses [0, 100], while
            # the neutral a*/b* channels deliberately contribute no colour
            # edge when this grayscale replacement is selected.
            neutral_lab = torch.cat(
                (
                    flattened.clamp(0.0, 1.0) * 100.0,
                    torch.zeros_like(flattened),
                    torch.zeros_like(flattened),
                ),
                dim=1,
            )
            vectors.append(source_vector(neutral_lab))
        else:
            vectors.append(source_vector(base_lab))
    enabled_details = (
        settings.edge_gradient_include_wavelet_detail_1,
        settings.edge_gradient_include_wavelet_detail_2,
        settings.edge_gradient_include_wavelet_detail_3,
        settings.edge_gradient_include_wavelet_detail_4,
    )
    for enabled, detail in zip(enabled_details, wavelet_details, strict=False):
        if not enabled:
            continue
        values = _raster_tensor(detail, context).float()
        blue, green, red = values[:, 0:1], values[:, 1:2], values[:, 2:3]
        opponent = torch.cat(
            (
                0.114 * blue + 0.587 * green + 0.299 * red,
                0.5 * (red - green),
                0.25 * (red + green - 2.0 * blue),
            ),
            dim=1,
        ) * float(settings.edge_wavelet_detail_gain)
        vectors.append(source_vector(opponent))
    if settings.edge_gradient_include_wavelet_residual and wavelet_residual is not None:
        residual = _raster_tensor(wavelet_residual, context).float().clamp(0.0, 255.0)
        vectors.append(source_vector(bgr_to_lab(residual)))
    if not vectors:
        raise ValueError("Edge gradients require at least one enabled image source.")

    gx_stack = torch.stack([item[0] for item in vectors], dim=0)
    gy_stack = torch.stack([item[1] for item in vectors], dim=0)
    magnitude_stack = torch.sqrt(
        gx_stack.square() + gy_stack.square()
    ).clamp_min(1e-8)
    if settings.edge_gradient_source_fusion == "maximum":
        selected = torch.argmax(magnitude_stack, dim=0, keepdim=True)
        gx = torch.gather(gx_stack, 0, selected)[0]
        gy = torch.gather(gy_stack, 0, selected)[0]
    else:
        mean_x = torch.mean(gx_stack, dim=0)
        mean_y = torch.mean(gy_stack, dim=0)
        if settings.edge_gradient_source_fusion == "vector_sum":
            gx, gy = mean_x, mean_y
        else:
            target = torch.sqrt(torch.mean(magnitude_stack.square(), dim=0))
            direction_norm = torch.sqrt(
                mean_x.square() + mean_y.square()
            ).clamp_min(1e-8)
            gx = mean_x / direction_norm * target
            gy = mean_y / direction_norm * target
    magnitude = torch.sqrt(gx.square() + gy.square()).clamp_min(0.0)
    normalization = torch.quantile(
        magnitude[valid], settings.edge_normalization_percentile / 100.0
    ).clamp_min(1e-5)
    strength = (magnitude / normalization).clamp(0.0, 1.0).pow(settings.edge_strength_gamma) * valid
    normal_degrees = torch.rad2deg(torch.atan2(gy, gx))
    tangent_degrees = torch.remainder(normal_degrees - 90.0, 360.0)
    directed_hue = torch.remainder(tangent_degrees * 0.5, 180.0) * valid
    undirected_hue = torch.remainder(tangent_degrees, 180.0) * valid
    inverse_magnitude = magnitude.clamp_min(1e-6).reciprocal()
    normal_x = gx * inverse_magnitude
    normal_y = gy * inverse_magnitude
    tangent_x = -normal_y
    tangent_y = normal_x
    return EdgeGradientProducts(
        source=source,
        lab=base_lab,
        valid=valid,
        strength=strength,
        normal_x=normal_x,
        normal_y=normal_y,
        tangent_x=tangent_x,
        tangent_y=tangent_y,
        directed_hue_float=directed_hue,
        undirected_hue_float=undirected_hue,
        strength_raster=_lazy_u8(strength * 255.0, "edge magnitude"),
        directed_hue_raster=_lazy_u8(directed_hue, "directed edge tangent"),
        undirected_hue_raster=_lazy_u8(undirected_hue, "undirected edge tangent"),
    )


def _fit_feature_prototype_bank(
    features,
    mask,
    *,
    class_name: str,
    priority_mask=None,
    maximum_prototypes: int,
    minimum_support: int,
    iterations: int,
    scale_floor: float,
) -> _FeaturePrototypeBank | None:
    """Fit deterministic coverage-preserving robust prototypes on the GPU."""

    import torch

    locations = torch.nonzero(mask[0, 0], as_tuple=False)
    sample_count = int(locations.shape[0])
    if sample_count == 0:
        return None

    def resampled_locations(values, count):
        if int(values.shape[0]) == count:
            return values
        indices = torch.linspace(
            0,
            int(values.shape[0]) - 1,
            count,
            device=values.device,
        ).round().long()
        return values[indices]

    if priority_mask is not None:
        priority = mask & priority_mask.bool()
        secondary = mask & ~priority
        priority_locations = torch.nonzero(
            priority[0, 0], as_tuple=False
        )
        secondary_locations = torch.nonzero(
            secondary[0, 0], as_tuple=False
        )
        if int(priority_locations.shape[0]) and int(
            secondary_locations.shape[0]
        ):
            fit_count = min(32768, sample_count)
            priority_count = (fit_count + 1) // 2
            secondary_count = fit_count - priority_count
            locations = torch.cat(
                (
                    resampled_locations(
                        priority_locations, priority_count
                    ),
                    resampled_locations(
                        secondary_locations, secondary_count
                    ),
                ),
                dim=0,
            )
        elif sample_count > 32768:
            locations = resampled_locations(locations, 32768)
    elif sample_count > 32768:
        locations = resampled_locations(locations, 32768)
    samples = features[0, :, locations[:, 0], locations[:, 1]].T
    centre = torch.median(samples, dim=0).values
    spread = (
        1.4826 * torch.median(torch.abs(samples - centre), dim=0).values
    ).clamp_min(float(scale_floor))
    standardized = (samples - centre) / spread
    prototype_count = min(
        int(maximum_prototypes),
        max(1, int(samples.shape[0]) // max(1, int(minimum_support))),
    )

    first = int(torch.argmax(standardized.square().mean(dim=1)).item())
    selected = [first]
    nearest = torch.sum(
        (standardized - standardized[first]).square(), dim=1
    )
    for _ in range(1, prototype_count):
        if float(nearest.max().item()) <= 1e-8:
            break
        candidate = int(torch.argmax(nearest).item())
        selected.append(candidate)
        distance = torch.sum(
            (standardized - standardized[candidate]).square(), dim=1
        )
        nearest = torch.minimum(nearest, distance)
    normalized_centres = standardized[
        torch.as_tensor(selected, device=samples.device)
    ].clone()
    assignments = torch.zeros(
        samples.shape[0], device=samples.device, dtype=torch.long
    )
    def nearest_assignments():
        """Assign samples without allocating the complete samples x K matrix."""

        # 2,097,152 float distances are 8 MiB.  Keeping this bound independent
        # of the user-selected edge-bank capacity makes 1,024 prototypes an
        # honest option on the supported 8 GiB GPUs instead of a hidden
        # full-raster memory multiplier.
        maximum_distance_values = 2_097_152
        centre_count = max(1, int(normalized_centres.shape[0]))
        chunk_size = max(
            1,
            min(
                int(standardized.shape[0]),
                maximum_distance_values // centre_count,
            ),
        )
        chunks = []
        for start in range(0, int(standardized.shape[0]), chunk_size):
            distances = torch.cdist(
                standardized[start : start + chunk_size],
                normalized_centres,
            ).square()
            chunks.append(torch.argmin(distances, dim=1))
        return torch.cat(chunks, dim=0)

    for _ in range(max(1, int(iterations))):
        assignments = nearest_assignments()
        updated = []
        for index in range(normalized_centres.shape[0]):
            members = standardized[assignments == index]
            updated.append(
                normalized_centres[index]
                if not len(members)
                else torch.median(members, dim=0).values
            )
        normalized_centres = torch.stack(updated)
    # Associate samples with the final robust centres, rather than the centres
    # from the start of the last refinement pass.
    assignments = nearest_assignments()

    final_centres = []
    final_scales = []
    final_weights = []
    final_counts = []
    final_positions = []
    for index in range(normalized_centres.shape[0]):
        member_indices = torch.nonzero(
            assignments == index, as_tuple=False
        ).flatten()
        if not len(member_indices):
            continue
        members = samples[member_indices]
        local_centre = torch.median(members, dim=0).values
        local_scale = (
            1.4826
            * torch.median(torch.abs(members - local_centre), dim=0).values
        ).clamp_min(float(scale_floor))
        normalized_distance = torch.mean(
            ((members - local_centre) / local_scale).square(), dim=1
        )
        medoid = member_indices[torch.argmin(normalized_distance)]
        count = int(member_indices.shape[0])
        final_centres.append(local_centre)
        final_scales.append(local_scale)
        final_weights.append(count / max(1, int(samples.shape[0])))
        final_counts.append(count)
        final_positions.append(locations[medoid])
    if not final_centres:
        return None
    return _FeaturePrototypeBank(
        class_name=class_name,
        centres=torch.stack(final_centres),
        scales=torch.stack(final_scales),
        weights=torch.as_tensor(
            final_weights, device=samples.device, dtype=torch.float32
        ),
        sample_counts=torch.as_tensor(
            final_counts, device=samples.device, dtype=torch.int32
        ),
        positions_yx=torch.stack(final_positions),
        sample_count=sample_count,
        source_count=1,
    )


def _prototype_bank_similarity(features, bank, tolerance: float):
    """Evaluate nearest diagonal prototype distance without a KxFxHxW tensor."""

    import torch
    import torch.nn.functional as functional

    if bank is None:
        return torch.zeros_like(features[:, :1])
    feature_count = max(1, int(features.shape[1]))
    best = torch.zeros_like(features[:, :1])
    squared = features.square()
    for start in range(0, int(bank.centres.shape[0]), 12):
        centres = bank.centres[start : start + 12]
        scales = (
            bank.scales[start : start + 12] * float(tolerance)
        ).clamp_min(1e-4)
        inverse_variance = scales.reciprocal().square()
        quadratic = functional.conv2d(
            squared, inverse_variance[:, :, None, None]
        )
        linear = functional.conv2d(
            features,
            (-2.0 * centres * inverse_variance)[:, :, None, None],
        )
        constant = torch.sum(
            centres.square() * inverse_variance, dim=1
        )[None, :, None, None]
        distance = ((quadratic + linear + constant) / feature_count).clamp(
            0.0, 30.0
        )
        similarity = torch.exp(-0.5 * distance)
        # Preserve rare, genuinely different prototypes while giving strongly
        # supported modes a modest preference.
        support = bank.weights[start : start + 12]
        support = 0.78 + 0.22 * torch.sqrt(
            support / bank.weights.max().clamp_min(1e-6)
        )
        similarity *= support[None, :, None, None]
        best = torch.maximum(best, similarity.max(dim=1, keepdim=True).values)
    return best


def _library_class_prototype_bank(
    source_bank,
    class_name: str,
    feature_names: tuple[str, ...],
    context,
):
    """Upload one immutable library class as a compact GPU prototype bank."""

    import torch

    if source_bank is None:
        return None
    if tuple(source_bank.feature_names) != tuple(feature_names):
        raise ValueError(
            f"Species-library {class_name} prototype schema does not match the live descriptor."
        )
    class_table = tuple(source_bank.class_names)
    aliases = {
        "non_edge": ("non_edge", "non_physical_edge"),
        "physical_edge": ("physical_edge",),
    }.get(class_name, (class_name,))
    class_id = next(
        (class_table.index(value) for value in aliases if value in class_table),
        None,
    )
    if class_id is None:
        return None
    selected = np.flatnonzero(np.asarray(source_bank.class_ids) == class_id)
    if not len(selected):
        return None
    device = context.device
    return _FeaturePrototypeBank(
        class_name=class_name,
        centres=torch.as_tensor(
            np.asarray(source_bank.centres)[selected], device=device, dtype=torch.float32
        ),
        scales=torch.as_tensor(
            np.asarray(source_bank.scales)[selected], device=device, dtype=torch.float32
        ),
        weights=torch.as_tensor(
            np.asarray(source_bank.weights)[selected], device=device, dtype=torch.float32
        ),
        sample_counts=torch.as_tensor(
            np.asarray(source_bank.sample_counts)[selected], device=device, dtype=torch.int32
        ),
        # Negative locations explicitly mark descriptor-only prototypes. They
        # must never be rendered as if they came from this image.
        positions_yx=torch.full(
            (len(selected), 2), -1, device=device, dtype=torch.int64
        ),
        sample_count=int(np.asarray(source_bank.sample_counts)[selected].sum()),
        source_count=max(
            1,
            len(np.unique(np.asarray(source_bank.source_indices)[selected])),
        ),
    )


def _select_or_merge_prototype_banks(
    local_bank,
    library_bank,
    *,
    source_mode: str,
    current_weight: float,
    class_name: str,
):
    """Apply an explicit source mode and source-level current-image weight."""

    import torch

    if source_mode == "Current image only":
        return local_bank
    if source_mode == "Species library only":
        return library_bank
    if source_mode != "Species library + current image":
        raise ValueError(f"Unknown prototype source mode: {source_mode}")
    if local_bank is None or float(current_weight) <= 0.0:
        return library_bank
    if library_bank is None:
        return local_bank
    local_weights = local_bank.weights / local_bank.weights.sum().clamp_min(1e-8)
    library_weights = library_bank.weights / library_bank.weights.sum().clamp_min(1e-8)
    # Current-image weight is source-level authority.  At weight 1.0 the
    # current image receives the same total authority as each independently
    # balanced library image, rather than the same authority as the entire
    # library or as every individual prototype.
    library_source_count = max(1, int(library_bank.source_count))
    denominator = library_source_count + float(current_weight)
    library_weights = library_weights * library_source_count / denominator
    local_weights = local_weights * float(current_weight) / denominator
    combined_weights = torch.cat((library_weights, local_weights))
    return _FeaturePrototypeBank(
        class_name=class_name,
        centres=torch.cat((library_bank.centres, local_bank.centres), dim=0),
        scales=torch.cat((library_bank.scales, local_bank.scales), dim=0),
        weights=combined_weights,
        sample_counts=torch.cat(
            (library_bank.sample_counts, local_bank.sample_counts), dim=0
        ),
        positions_yx=torch.cat(
            (library_bank.positions_yx, local_bank.positions_yx), dim=0
        ),
        sample_count=library_bank.sample_count + local_bank.sample_count,
        source_count=library_source_count + 1,
    )


def reference_seed_trait_probabilities(
    gradients: EdgeGradientProducts,
    ridges,
    frequency_noise: FrequencyNoiseMaskProducts,
    seed_material_mask,
    seed_diameter: float,
    settings,
    *,
    seed_instance_annotations=None,
    seed_instance_traits: tuple[object, ...] = (),
    annotation_species: str = "",
    coat_patterns: tuple[str, ...] = (),
    conditions: tuple[str, ...] = (),
    species_library=None,
    cuda_context=None,
) -> ReferenceSeedTraitProducts:
    """Learn isolated, image-local material classifiers from labelled seeds.

    Coat classes compete only with one another and are normalized to sum to
    one on the resolved seed-material mask whenever at least two classes have
    training support. Each condition is a separate reviewed-positive versus
    reviewed-negative model, so condition outputs remain non-exclusive. No
    authored class mask is ever copied into or used to overwrite an output.
    """

    import cv2
    import torch
    import torch.nn.functional as functional

    from seedvision.visualization.layers import ReferenceSeedTraitProfile

    context = cuda_context or CudaContext.resolve()
    source_height, source_width = gradients.strength.shape[-2:]
    categories = tuple(dict.fromkeys(str(value) for value in coat_patterns))
    condition_names = tuple(dict.fromkeys(str(value) for value in conditions))
    zero_full = gradients.valid.float() * 0.0
    library_trait_bank = (
        None if species_library is None else species_library.seed_traits
    )

    def empty_products(profile=None):
        zero = _lazy_u8(zero_full, "unavailable reference seed trait probability")
        return ReferenceSeedTraitProducts(
            coat_probabilities=tuple((name, zero) for name in categories),
            condition_probabilities=tuple(
                (name, zero) for name in condition_names
            ),
            profile=profile
            or ReferenceSeedTraitProfile(
                species_id=str(annotation_species),
                coat_patterns=categories,
                conditions=condition_names,
            ),
        )

    has_local_traits = bool(
        seed_instance_annotations is not None
        and np.any(seed_instance_annotations)
        and seed_instance_traits
    )
    if (not has_local_traits and library_trait_bank is None) or (
        not categories and not condition_names
    ):
        return empty_products()
    labels = (
        np.zeros((source_height, source_width), np.uint16)
        if seed_instance_annotations is None
        else np.asarray(seed_instance_annotations)
    )
    if labels.shape != (source_height, source_width):
        raise ValueError("Seed trait annotations must match the corrected crop.")

    maximum = int(settings.reference_seed_trait_working_maximum_dimension)
    work_scale = min(1.0, maximum / max(source_height, source_width))
    height = max(16, round(source_height * work_scale))
    width = max(16, round(source_width * work_scale))
    if labels.shape != (height, width):
        work_labels = cv2.resize(
            labels.astype(np.float32, copy=False),
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.uint16, copy=False)
    else:
        work_labels = labels.astype(np.uint16, copy=False)

    diameter = max(6.0, float(seed_diameter) * work_scale)
    interior_distance = max(
        1.0,
        diameter * float(settings.reference_seed_trait_interior_buffer_fraction),
    )
    safe_interior = np.zeros((height, width), dtype=bool)
    extant_ids = set(int(value) for value in np.unique(work_labels) if value)
    annotations = {
        int(value.seed_id): value
        for value in seed_instance_traits
        if int(value.seed_id) in extant_ids
    }
    for seed_id in annotations:
        instance = np.uint8(work_labels == seed_id)
        distance = cv2.distanceTransform(instance, cv2.DIST_L2, 5)
        interior = distance >= interior_distance
        if not np.any(interior):
            # Tiny reviewed seeds still contribute only if they have a genuine
            # one-pixel inset; contour pixels are never treated as coat material.
            interior = cv2.erode(instance, np.ones((3, 3), np.uint8)) > 0
        safe_interior |= interior
    if not np.any(safe_interior) and library_trait_bank is None:
        return empty_products()

    def resized(values, *, mode="bilinear"):
        tensor = values.float()
        if tensor.ndim == 2:
            tensor = tensor[None, None]
        elif tensor.ndim == 3:
            tensor = tensor[None]
        if tensor.shape[-2:] == (height, width):
            return tensor
        arguments = {} if mode in {"nearest", "area"} else {"align_corners": False}
        return functional.interpolate(tensor, (height, width), mode=mode, **arguments)

    valid = resized(gradients.valid.float(), mode="nearest") > 0.5
    lab = resized(gradients.lab)
    edge = resized(gradients.strength).clamp(0.0, 1.0)
    ridge = resized(_raster_tensor(ridges, context, normalized=True)).clamp(0.0, 1.0)
    noise_channels = tuple(
        resized(_raster_tensor(raster, context, normalized=True))
        for raster in (
            *frequency_noise.darkness_masks,
            *frequency_noise.colour_masks,
        )
    )
    sigma = max(
        0.7,
        diameter * float(settings.reference_seed_trait_context_fraction),
    )
    features = _reference_material_feature_tensor(
        lab,
        edge,
        ridge,
        noise_channels,
        context_sigma=sigma,
    )
    class_seed_counts: list[tuple[str, int]] = []
    class_sample_counts: list[tuple[str, int]] = []
    prototype_counts: list[tuple[str, int]] = []

    def bank_for(seed_ids, class_name):
        ids = tuple(sorted(set(int(value) for value in seed_ids)))
        class_seed_counts.append((class_name, len(ids)))
        bank = None
        if ids:
            class_mask_numpy = np.isin(work_labels, ids) & safe_interior
            mask = (image_to_tensor(class_mask_numpy.astype(np.uint8), context) > 0) & valid
            bank = _fit_feature_prototype_bank(
                features,
                mask,
                class_name=class_name,
                maximum_prototypes=int(
                    settings.reference_seed_trait_prototypes_per_class
                ),
                minimum_support=int(
                    settings.reference_seed_trait_minimum_samples_per_prototype
                ),
                iterations=int(settings.reference_seed_trait_fit_iterations),
                scale_floor=0.045,
            )
        library_bank = _library_class_prototype_bank(
            library_trait_bank,
            class_name,
            tuple(library_trait_bank.feature_names) if library_trait_bank is not None else (),
            context,
        ) if library_trait_bank is not None else None
        if library_trait_bank is not None and tuple(library_trait_bank.feature_names) != (
            "Lab lightness", "Lab a*", "Lab b*",
            "fine darkness noise", "medium darkness noise", "coarse darkness noise",
            "fine colour noise", "medium colour noise", "coarse colour noise",
            "edge magnitude", "ridge support", "local lightness residual",
            "local colour residual", "local edge density",
        ):
            raise ValueError("Species-library trait descriptor does not match the live descriptor.")
        bank = _select_or_merge_prototype_banks(
            bank,
            library_bank,
            source_mode=str(settings.seed_trait_reference_source),
            current_weight=float(settings.seed_trait_current_reference_weight),
            class_name=class_name,
        )
        class_sample_counts.append(
            (class_name, 0 if bank is None else int(bank.sample_count))
        )
        prototype_counts.append(
            (
                class_name,
                0 if bank is None else int(bank.centres.shape[0]),
            )
        )
        return bank

    coat_banks = {
        category: bank_for(
            (
                seed_id
                for seed_id, annotation in annotations.items()
                if getattr(annotation, "coat_pattern", None) == category
            ),
            f"coat:{category}",
        )
        for category in categories
    }
    tolerance = float(settings.reference_seed_trait_similarity_scale)
    contrast = float(settings.reference_seed_trait_class_contrast)
    coat_scores = {
        name: (
            None
            if bank is None
            else _prototype_bank_similarity(features, bank, tolerance)
        )
        for name, bank in coat_banks.items()
    }
    available_coats = tuple(
        name for name in categories if coat_scores[name] is not None
    )
    coat_work = {name: torch.zeros_like(features[:, :1]) for name in categories}
    coat_model_available = len(available_coats) >= 2
    if coat_model_available:
        logits = torch.cat(
            [
                torch.log(coat_scores[name].clamp_min(1e-6)) * contrast
                for name in available_coats
            ],
            dim=1,
        )
        normalized = torch.softmax(logits, dim=1) * valid.float()
        for index, name in enumerate(available_coats):
            coat_work[name] = normalized[:, index : index + 1]

    condition_work: dict[str, object] = {}
    condition_availability: list[tuple[str, bool]] = []
    for condition in condition_names:
        reviewed = {
            seed_id: annotation
            for seed_id, annotation in annotations.items()
            if bool(getattr(annotation, "conditions_reviewed", False))
        }
        positive = bank_for(
            (
                seed_id
                for seed_id, annotation in reviewed.items()
                if condition in tuple(getattr(annotation, "conditions", ()))
            ),
            f"condition:{condition}:present",
        )
        negative = bank_for(
            (
                seed_id
                for seed_id, annotation in reviewed.items()
                if condition not in tuple(getattr(annotation, "conditions", ()))
            ),
            f"condition:{condition}:absent",
        )
        available = positive is not None and negative is not None
        condition_availability.append((condition, available))
        if not available:
            condition_work[condition] = torch.zeros_like(features[:, :1])
            continue
        positive_score = _prototype_bank_similarity(features, positive, tolerance)
        negative_score = _prototype_bank_similarity(features, negative, tolerance)
        positive_mass = positive_score.clamp_min(1e-6).pow(contrast)
        negative_mass = negative_score.clamp_min(1e-6).pow(contrast)
        condition_work[condition] = (
            positive_mass / (positive_mass + negative_mass).clamp_min(1e-6)
        ) * valid.float()

    full_support = (
        _raster_tensor(seed_material_mask, context, normalized=True) > 0.5
    ) & gradients.valid.bool()

    def restored(values, name):
        if values.shape[-2:] != (source_height, source_width):
            values = functional.interpolate(
                values,
                (source_height, source_width),
                mode="bilinear",
                align_corners=False,
            )
        return values.clamp(0.0, 1.0) * full_support.float()

    coat_full = {
        name: restored(values, f"seed coat {name} probability")
        for name, values in coat_work.items()
    }
    if coat_model_available:
        denominator = sum(coat_full.values()).clamp_min(1e-8)
        for name in available_coats:
            coat_full[name] = (
                coat_full[name] / denominator
            ) * full_support.float()
    condition_full = {
        name: restored(values, f"seed condition {name} probability")
        for name, values in condition_work.items()
    }
    profile = ReferenceSeedTraitProfile(
        species_id=str(annotation_species),
        coat_patterns=categories,
        conditions=condition_names,
        class_seed_counts=tuple(class_seed_counts),
        class_sample_counts=tuple(class_sample_counts),
        prototype_counts=tuple(prototype_counts),
        coat_model_available=coat_model_available,
        condition_models_available=tuple(condition_availability),
        reference_source_mode=str(settings.seed_trait_reference_source),
        current_seed_count=len(annotations),
        library_source_count=(
            0
            if library_trait_bank is None
            or str(settings.seed_trait_reference_source) == "Current image only"
            else len(np.unique(np.asarray(library_trait_bank.source_indices)))
        ),
        library_prototype_count=(
            0
            if library_trait_bank is None
            or str(settings.seed_trait_reference_source) == "Current image only"
            else len(library_trait_bank.centres)
        ),
    )
    return ReferenceSeedTraitProducts(
        coat_probabilities=tuple(
            (
                name,
                _lazy_u8(
                    coat_full[name] * 255.0,
                    f"seed coat {name} probability",
                ),
            )
            for name in categories
        ),
        condition_probabilities=tuple(
            (
                name,
                _lazy_u8(
                    condition_full[name] * 255.0,
                    f"seed condition {name} probability",
                ),
            )
            for name in condition_names
        ),
        profile=profile,
    )


def _prototype_class_probabilities(
    class_scores,
    valid,
    *,
    class_contrast: float,
    unknown_mass: float = 0.08,
):
    """Convert competing prototype-kernel similarities into class masses.

    A Gaussian-kernel similarity is expected to be only about ``exp(-0.5)``
    for an ordinary in-distribution sample one robust scale from its nearest
    medoid. Treating that value directly as a posterior made even reviewed
    class appearances look artificially middling. Keep two distinct questions
    separate instead:

    * the unmodified strongest similarity controls known-versus-unknown mass;
    * a bounded contrast exponent calibrates relative competition among the
      available semantic classes.

    This calculation receives no reference masks or coordinates. The same
    equation is therefore applied to painted and unpainted pixels, and it
    cannot hard-write authored labels into the result.
    """

    import torch

    available = tuple(
        (name, score)
        for name, score in class_scores.items()
        if score is not None
    )
    if len(available) < 2:
        return {}
    names = tuple(name for name, _score in available)
    scores = torch.stack(
        tuple(score.clamp(0.0, 1.0) for _name, score in available),
        dim=0,
    )
    epsilon = torch.finfo(scores.dtype).eps
    relative = scores.clamp_min(epsilon).pow(float(class_contrast))
    conditional = relative / relative.sum(dim=0).clamp_min(epsilon)
    strongest = scores.max(dim=0).values
    known_confidence = strongest / (
        strongest + float(unknown_mass)
    ).clamp_min(epsilon)
    masses = conditional * known_confidence * valid
    return {
        name: masses[index]
        for index, name in enumerate(names)
    }


def _candidate_internal_edge_mask(
    edge,
    ridge,
    tangent_x,
    tangent_y,
    safe_interior,
    *,
    ridge_weight: float = 0.35,
):
    """Select sparse edge-like negative examples inside reviewed instances.

    The safe interior is deliberately *not* itself a non-physical-edge class:
    flat pixels would outnumber the coat-pattern ridges that the classifier must
    learn to reject. A previously thinned ridge may have passed through one or
    more bilinear resizes before reaching this independently sized descriptor
    grid. Treating every positive halo pixel as a candidate moves prototype
    centres away from the transition they are meant to classify. Re-thin the
    configured edge/ridge blend along its live normal here, on the exact fitting
    grid, and apply the prominence threshold to *all* candidates. The adaptive
    threshold is evaluated only inside annotated safe interiors and retains an
    absolute floor so sensor variation in an otherwise flat seed does not
    become supervision.
    """

    import torch
    safe = safe_interior.bool()
    if not bool(safe.any().item()):
        return torch.zeros_like(safe)
    expected_shape = edge.shape
    if any(
        value.shape != expected_shape
        for value in (ridge, tangent_x, tangent_y, safe_interior)
    ):
        raise ValueError(
            "Internal-edge evidence, tangents, and safe interior must match."
        )
    ridge_weight = min(1.0, max(0.0, float(ridge_weight)))
    candidate_strength = (
        (1.0 - ridge_weight) * edge + ridge_weight * ridge
    ).clamp(0.0, 1.0)
    interior_strength = candidate_strength[safe]
    quantile_position = 0.65 * (int(interior_strength.numel()) - 1)
    lower_index = int(np.floor(quantile_position))
    upper_index = int(np.ceil(quantile_position))
    lower = torch.kthvalue(interior_strength, lower_index + 1).values
    if lower_index == upper_index:
        adaptive = lower
    else:
        upper = torch.kthvalue(interior_strength, upper_index + 1).values
        adaptive = torch.lerp(
            lower,
            upper,
            quantile_position - lower_index,
        )
    adaptive = adaptive.clamp_min(0.08)
    tangent_length = torch.sqrt(
        tangent_x.square() + tangent_y.square()
    ).clamp_min(1e-6)
    tx, ty = tangent_x / tangent_length, tangent_y / tangent_length
    normal_x, normal_y = -ty, tx
    height, width = candidate_strength.shape[-2:]
    yy, xx = torch.meshgrid(
        torch.arange(
            height, device=candidate_strength.device, dtype=torch.float32
        ),
        torch.arange(
            width, device=candidate_strength.device, dtype=torch.float32
        ),
        indexing="ij",
    )
    forward = bilinear_sample(
        candidate_strength,
        xx + normal_x[0, 0],
        yy + normal_y[0, 0],
    )
    backward = bilinear_sample(
        candidate_strength,
        xx - normal_x[0, 0],
        yy - normal_y[0, 0],
    )
    centre = candidate_strength[0, 0]
    # The asymmetric equality resolves an exactly flat two-pixel peak to one
    # deterministic side instead of retaining a doubled source footprint.
    normal_maximum = (centre >= forward) & (centre > backward)
    tangent_defined = (
        tangent_x.square() + tangent_y.square()
    )[0, 0] >= 1e-6
    prominent = centre >= adaptive
    return (
        safe
        & normal_maximum[None, None]
        & prominent
        & tangent_defined[None, None]
    )


def thin_probability_ridges(
    probability,
    normal_x,
    normal_y,
    valid,
    *,
    normal_sampling_step_px: float,
    low_threshold: float,
    high_threshold: float,
    hysteresis_iterations: int,
):
    """Thin a continuous probability field along its local edge normal on GPU.

    This is the probability-map analogue of the shared edge-ridge operation:
    subpixel non-maximum suppression preserves the original peak confidence,
    then high-confidence maxima seed a bounded reconstruction through connected
    weak maxima.  An asymmetric equality test gives plateaus one deterministic
    side instead of leaving a two-pixel-wide ridge.
    """

    import torch
    import torch.nn.functional as functional

    values = probability.float().clamp(0.0, 1.0)
    if values.ndim == 2:
        values = values[None, None]
    elif values.ndim == 3:
        values = values[None]
    nx = normal_x.float()
    ny = normal_y.float()
    if nx.ndim == 2:
        nx = nx[None, None]
    elif nx.ndim == 3:
        nx = nx[None]
    if ny.ndim == 2:
        ny = ny[None, None]
    elif ny.ndim == 3:
        ny = ny[None]
    support = valid.bool()
    if support.ndim == 2:
        support = support[None, None]
    elif support.ndim == 3:
        support = support[None]
    if not (
        values.shape == nx.shape == ny.shape == support.shape
        and values.shape[0] == values.shape[1] == 1
    ):
        raise ValueError(
            "Probability, normals, and valid mask must be matching 1-channel rasters."
        )
    if not 0.0 <= float(low_threshold) < float(high_threshold) <= 1.0:
        raise ValueError("Ridge thresholds must satisfy 0 <= low < high <= 1.")
    if int(hysteresis_iterations) < 1:
        raise ValueError("Ridge hysteresis must run for at least one iteration.")

    length = torch.sqrt(nx.square() + ny.square()).clamp_min(1e-6)
    nx, ny = nx / length, ny / length
    height, width = values.shape[-2:]
    yy, xx = torch.meshgrid(
        torch.arange(height, device=values.device, dtype=torch.float32),
        torch.arange(width, device=values.device, dtype=torch.float32),
        indexing="ij",
    )
    step = max(0.05, float(normal_sampling_step_px))
    forward = bilinear_sample(
        values,
        xx + nx[0, 0] * step,
        yy + ny[0, 0] * step,
    )
    backward = bilinear_sample(
        values,
        xx - nx[0, 0] * step,
        yy - ny[0, 0] * step,
    )
    centre = values[0, 0]
    nms = (
        centre
        * (centre >= forward)
        * (centre > backward)
        * support[0, 0]
    )
    weak = nms >= float(low_threshold)
    accepted = nms >= float(high_threshold)
    for _ in range(int(hysteresis_iterations)):
        reached = functional.max_pool2d(
            accepted.float()[None, None], 3, stride=1, padding=1
        )[0, 0] > 0
        accepted = weak & reached
    return (nms * accepted)[None, None]


def _reference_patch(
    crop: np.ndarray,
    centre_xy: tuple[float, float],
    size: int,
    tangent_degrees: float | None,
) -> np.ndarray:
    """Extract an axis- or tangent-aligned compact BGR prototype thumbnail."""

    import cv2

    size = max(12, int(size))
    if tangent_degrees is None:
        return np.ascontiguousarray(
            cv2.getRectSubPix(crop, (size, size), centre_xy)
        )
    outer = max(size + 4, int(np.ceil(size * 1.45)))
    local = cv2.getRectSubPix(crop, (outer, outer), centre_xy)
    centre = ((outer - 1) * 0.5, (outer - 1) * 0.5)
    # OpenCV image coordinates have +Y downward.  The tangent returned by the
    # edge descriptor uses that same convention, so rotating by the positive
    # image-space angle aligns the edge horizontally in the thumbnail.
    matrix = cv2.getRotationMatrix2D(centre, float(tangent_degrees), 1.0)
    aligned = cv2.warpAffine(
        local,
        matrix,
        (outer, outer),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return np.ascontiguousarray(
        cv2.getRectSubPix(aligned, (size, size), centre)
    )


def _adaptive_edge_working_shape(
    source_height: int,
    source_width: int,
    seed_diameter: float,
    *,
    material_maximum_dimension: int,
    edge_maximum_dimension: int,
    minimum_working_seed_diameter_px: float,
) -> tuple[float, int, int]:
    """Choose an honest, bounded edge-classifier working resolution.

    Material prototypes retain their independently configured working limit.
    Edge strips use enough pixels to represent a small seed whenever the
    explicit edge limit permits it, but never exceed either that hard limit or
    source resolution.
    """

    longest = max(1, int(source_height), int(source_width))
    diameter = max(1e-6, float(seed_diameter))
    target_longest = int(
        np.ceil(
            longest
            * max(1.0, float(minimum_working_seed_diameter_px))
            / diameter
        )
    )
    requested_longest = max(
        int(material_maximum_dimension),
        target_longest,
    )
    working_longest = min(
        longest,
        max(16, int(edge_maximum_dimension)),
        requested_longest,
    )
    scale = min(1.0, working_longest / longest)
    return (
        float(scale),
        max(16, round(int(source_height) * scale)),
        max(16, round(int(source_width) * scale)),
    )


def _instance_outward_normals(
    instance_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ID-aware inner contours and their local outward unit normals.

    A neighbouring different positive ID is exterior to the current instance,
    just like zero-valued background.  This preserves opposing normals on the
    two one-pixel contours at a seed contact instead of collapsing the labelled
    instances into one binary union.
    """

    labels = np.asarray(instance_labels)
    if labels.ndim != 2 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("Instance labels must be a two-dimensional integer raster.")
    if np.any(labels < 0):
        raise ValueError("Instance labels cannot contain negative IDs.")
    labelled = labels > 0
    padded = np.pad(labels, 1, mode="constant", constant_values=0)
    centre = padded[1:-1, 1:-1]
    normal_x = np.zeros(labels.shape, np.float32)
    normal_y = np.zeros(labels.shape, np.float32)
    contour = np.zeros(labels.shape, bool)
    for y_offset in (-1, 0, 1):
        for x_offset in (-1, 0, 1):
            if x_offset == 0 and y_offset == 0:
                continue
            neighbour = padded[
                1 + y_offset : 1 + y_offset + labels.shape[0],
                1 + x_offset : 1 + x_offset + labels.shape[1],
            ]
            different = labelled & (neighbour != centre)
            contour |= different
            inverse_length = 1.0 / float(np.hypot(x_offset, y_offset))
            normal_x[different] += float(x_offset) * inverse_length
            normal_y[different] += float(y_offset) * inverse_length
    length = np.hypot(normal_x, normal_y)
    usable = contour & (length > 1e-6)
    normal_x[usable] /= length[usable]
    normal_y[usable] /= length[usable]
    contour &= usable
    normal_x[~contour] = 0.0
    normal_y[~contour] = 0.0
    return contour, normal_x, normal_y


def _edge_strip_feature_maps(
    lab,
    edge,
    ridge,
    tangent_x,
    tangent_y,
    valid,
    seed_diameter: float,
    *,
    normal_offset_fraction: float,
    tangent_half_length_fraction: float,
    include_reverse: bool = True,
    include_normals: bool = True,
):
    """Build normal-oriented interior/edge/exterior strip descriptors.

    The first returned tensor treats ``+normal`` as exterior; the optional
    second tensor is the exact polarity reversal. Each zone is a short
    tangent-pooled line sample, not a square image patch. Validity participates
    in every weighted average, so padding or the dish boundary cannot
    manufacture a dark matching strip. Callers fitting an already oriented
    training bank can omit the reverse tensor and normal maps to lower peak
    memory.
    """

    import torch

    if lab.ndim != 4 or lab.shape[0] != 1 or lab.shape[1] != 3:
        raise ValueError("Strip Lab input must have shape 1 x 3 x H x W.")
    expected = lab.shape[-2:]
    scalar_inputs = (edge, ridge, tangent_x, tangent_y, valid)
    if any(value.shape[-2:] != expected for value in scalar_inputs):
        raise ValueError("All edge-strip inputs must have matching dimensions.")
    height, width = expected
    yy, xx = torch.meshgrid(
        torch.arange(height, device=lab.device, dtype=torch.float32),
        torch.arange(width, device=lab.device, dtype=torch.float32),
        indexing="ij",
    )
    tangent_length = torch.sqrt(
        tangent_x.square() + tangent_y.square()
    ).clamp_min(1e-6)
    tx = tangent_x / tangent_length
    ty = tangent_y / tangent_length
    nx = -ty
    ny = tx
    half_length = max(
        1.0,
        float(seed_diameter) * float(tangent_half_length_fraction),
    )
    normal_offset = max(
        1.0,
        float(seed_diameter) * float(normal_offset_fraction),
    )
    tangent_offsets = (-half_length, -0.5 * half_length, 0.0,
                       0.5 * half_length, half_length)
    tangent_weights = (1.0, 2.0, 3.0, 2.0, 1.0)
    valid_float = valid.float()

    def pooled_at(values, normal_distance: float):
        channels = []
        support = torch.zeros_like(xx)
        accumulated = [torch.zeros_like(xx) for _ in range(values.shape[1])]
        base_x = xx + nx[0, 0] * float(normal_distance)
        base_y = yy + ny[0, 0] * float(normal_distance)
        for tangent_distance, weight in zip(
            tangent_offsets, tangent_weights, strict=True
        ):
            sample_x = base_x + tx[0, 0] * float(tangent_distance)
            sample_y = base_y + ty[0, 0] * float(tangent_distance)
            sample_valid = bilinear_sample(
                valid_float, sample_x, sample_y
            ).clamp(0.0, 1.0)
            weighted_valid = sample_valid * float(weight)
            support += weighted_valid
            for channel in range(values.shape[1]):
                accumulated[channel] += (
                    bilinear_sample(
                        values[:, channel : channel + 1],
                        sample_x,
                        sample_y,
                    )
                    * weighted_valid
                )
        denominator = support.clamp_min(1e-5)
        for value in accumulated:
            channels.append((value / denominator)[None, None])
        maximum_support = float(sum(tangent_weights))
        return torch.cat(channels, dim=1), (
            support / maximum_support
        )[None, None]

    # Semantic classes describe which kind of transition is present. Absolute
    # edge/ridge amplitude is combined later by Reference edges when a precise
    # ridge is requested; embedding it here merely disguises edge strength as
    # a class probability.
    base_features = lab
    interior, interior_support = pooled_at(base_features, -normal_offset)
    centre, centre_support = pooled_at(base_features, 0.0)
    exterior, exterior_support = pooled_at(base_features, normal_offset)

    local_sigma = max(0.7, float(seed_diameter) * 0.025)
    local_lab = gaussian_blur(lab, local_sigma)
    residual = lab - local_lab
    residual_features = torch.cat(
        (
            residual[:, 0:1].abs() / 40.0,
            torch.sqrt(
                residual[:, 1:2].square() + residual[:, 2:3].square()
            ) / 55.0,
        ),
        dim=1,
    )
    interior_residual, _ = pooled_at(residual_features, -normal_offset)
    centre_residual, _ = pooled_at(residual_features, 0.0)
    exterior_residual, _ = pooled_at(residual_features, normal_offset)

    def normalized_zone(zone, zone_residual):
        return torch.cat(
            (
                zone[:, 0:1] / 255.0,
                (zone[:, 1:2] - 128.0) / 128.0,
                (zone[:, 2:3] - 128.0) / 128.0,
                zone_residual,
            ),
            dim=1,
        )

    interior_zone = normalized_zone(interior, interior_residual)
    centre_zone = normalized_zone(centre, centre_residual)
    exterior_zone = normalized_zone(exterior, exterior_residual)
    signed_contrast = torch.cat(
        (
            (interior[:, 0:1] - exterior[:, 0:1]) / 55.0,
            (interior[:, 1:2] - exterior[:, 1:2]) / 70.0,
            (interior[:, 2:3] - exterior[:, 2:3]) / 70.0,
        ),
        dim=1,
    )
    axial_x = tx.square() - ty.square()
    axial_y = 2.0 * tx * ty
    coherence = torch.sqrt(
        gaussian_blur(axial_x, max(0.7, half_length * 0.5)).square()
        + gaussian_blur(axial_y, max(0.7, half_length * 0.5)).square()
    ).clamp(0.0, 1.0)
    support = torch.minimum(
        interior_support,
        torch.minimum(centre_support, exterior_support),
    )
    forward = torch.cat(
        (
            interior_zone,
            centre_zone,
            exterior_zone,
            signed_contrast,
            coherence,
            support,
        ),
        dim=1,
    )
    # Zone width is five (Lab plus L/chroma residual).
    zone_width = 5
    reverse = (
        torch.cat(
            (
                forward[:, 2 * zone_width : 3 * zone_width],
                forward[:, zone_width : 2 * zone_width],
                forward[:, :zone_width],
                -forward[:, 3 * zone_width : 3 * zone_width + 3],
                forward[:, 3 * zone_width + 3 :],
            ),
            dim=1,
        )
        if include_reverse
        else None
    )
    strip_valid = support >= 0.60
    return (
        forward,
        reverse,
        strip_valid,
        nx if include_normals else None,
        ny if include_normals else None,
    )


def _physical_edge_interior_direction(
    forward_similarity,
    reverse_similarity,
    physical_probability,
    normal_x,
    normal_y,
    valid,
):
    """Recover a suggestive inward normal without changing edge probability.

    Physical prototypes are fitted with ``+normal`` pointing out of the
    annotated seed. At query time both polarities are scored and the existing
    classifier deliberately uses their maximum. The winning polarity therefore
    tells us which side resembles the learned interior, while the normalized
    score difference records how trustworthy that orientation is. This helper
    is intentionally downstream of ``physical_probability`` and never mutates
    or recalibrates it.
    """

    import torch

    shapes = {
        tuple(value.shape)
        for value in (
            forward_similarity,
            reverse_similarity,
            physical_probability,
            normal_x,
            normal_y,
            valid,
        )
    }
    if len(shapes) != 1:
        raise ValueError("Physical-edge direction inputs must have matching shapes.")
    polarity_separation = (
        (forward_similarity - reverse_similarity).abs()
        / (forward_similarity + reverse_similarity).clamp_min(1e-6)
    ).clamp(0.0, 1.0)
    confidence = (
        physical_probability.clamp(0.0, 1.0)
        * polarity_separation
        * valid.float()
    )
    # Forward descriptors treat +normal as exterior, hence -normal is inward.
    # A reverse win swaps those sides. Ties deliberately receive zero
    # confidence above, so their arbitrary sign is never presented as evidence.
    inward_sign = torch.where(
        reverse_similarity > forward_similarity,
        torch.ones_like(forward_similarity),
        -torch.ones_like(forward_similarity),
    )
    return normal_x * inward_sign, normal_y * inward_sign, confidence


def reference_texture_probabilities(
    crop,
    gradients: EdgeGradientProducts,
    ridges,
    frequency_noise: FrequencyNoiseMaskProducts,
    seed_diameter,
    settings,
    *,
    background_reference_mask=None,
    background_reference_source_mask=None,
    background_automatic_authority: float = 1.0,
    foreground_reference_mask=None,
    foreground_reference_source_mask=None,
    other_reference_mask=None,
    seed_instance_annotations=None,
    species_library=None,
    cuda_context=None,
) -> ReferenceTextureProducts:
    """Learn material prototypes and annotation-derived edge prototypes.

    Every prototype represents a robust feature medoid; its retained image
    patch is provenance/context for the collage, not the matching template.
    The banks are evaluated globally and reference coordinates are never
    overwritten. Physical/non-physical boundary supervision comes exclusively
    and automatically from complete instance annotations.
    """

    import torch
    import torch.nn.functional as functional

    from seedvision.visualization.layers import (
        ReferenceTextureProfile,
        ReferenceTexturePrototype,
    )

    context = cuda_context or CudaContext.resolve()
    has_instance_annotations = bool(
        seed_instance_annotations is not None
        and np.any(seed_instance_annotations)
    )
    source_height, source_width = gradients.strength.shape[-2:]
    material_feature_names = (
        "Lab lightness",
        "Lab a*",
        "Lab b*",
        "fine darkness noise",
        "medium darkness noise",
        "coarse darkness noise",
        "fine colour noise",
        "medium colour noise",
        "coarse colour noise",
        "edge magnitude",
        "ridge support",
        "local lightness residual",
        "local colour residual",
        "local edge density",
    )
    edge_zone_names = (
        "Lab lightness",
        "Lab a*",
        "Lab b*",
        "local lightness residual",
        "local colour residual",
    )
    edge_feature_names = tuple(
        f"{zone} strip {feature}"
        for zone in ("interior", "centre/edge", "exterior")
        for feature in edge_zone_names
    ) + (
        "signed cross-edge lightness (interior - exterior)",
        "signed cross-edge a* (interior - exterior)",
        "signed cross-edge b* (interior - exterior)",
        "strip axial tangent coherence",
        "strip valid support",
    )
    library_material_bank = (
        None if species_library is None else species_library.material_prototypes
    )
    library_edge_bank = (
        None if species_library is None else species_library.edge_prototypes
    )
    material_mode = str(settings.material_prototype_reference_source)
    edge_mode = str(settings.edge_prototype_reference_source)

    def selected_library_counts(bank, aliases, mode):
        if bank is None or mode == "Current image only":
            return 0, 0
        class_ids = [
            bank.class_names.index(name)
            for name in aliases
            if name in bank.class_names
        ]
        selected = np.isin(np.asarray(bank.class_ids), class_ids)
        if not np.any(selected):
            return 0, 0
        return (
            len(np.unique(np.asarray(bank.source_indices)[selected])),
            int(np.count_nonzero(selected)),
        )

    library_material_source_count, library_material_prototype_count = (
        selected_library_counts(
            library_material_bank, ("foreground",), material_mode
        )
    )
    library_edge_source_count, library_edge_prototype_count = (
        selected_library_counts(
            library_edge_bank,
            ("physical_edge", "non_edge", "non_physical_edge"),
            edge_mode,
        )
    )

    def full_reference_mask(values):
        if values is None:
            return torch.zeros(
                (1, 1, source_height, source_width),
                device=context.device,
                dtype=torch.bool,
            )
        if hasattr(values, "gpu_tensor"):
            source = values.gpu_tensor(device=context.device)
            if source.ndim == 2:
                source = source[None, None]
            elif source.ndim == 3:
                source = source[None]
        else:
            array = np.ascontiguousarray(
                np.asarray(values, dtype=np.uint8)
            )
            if array.ndim != 2:
                raise ValueError("Reference masks must be two-dimensional.")
            source = torch.from_numpy(array).to(
                device=context.device
            )[None, None]
        if source.shape[-2:] != (source_height, source_width):
            raise ValueError(
                "Reference masks must match the prototype source dimensions."
            )
        return source > 0

    full_painted_background = full_reference_mask(background_reference_mask)
    full_painted_foreground = full_reference_mask(foreground_reference_mask)
    full_automatic_foreground = full_reference_mask(
        foreground_reference_source_mask
    )
    full_other = full_reference_mask(other_reference_mask)
    full_annotations = full_reference_mask(
        None
        if seed_instance_annotations is None
        else np.asarray(seed_instance_annotations) > 0
    )
    full_source_valid = gradients.valid.bool()
    full_painted_background &= full_source_valid
    full_painted_foreground &= full_source_valid
    full_automatic_foreground &= full_source_valid
    full_other &= full_source_valid
    full_annotations &= full_source_valid
    full_manual_overlap = (
        full_painted_background.to(torch.uint8)
        + full_painted_foreground.to(torch.uint8)
        + full_other.to(torch.uint8)
    ) > 1
    full_painted_background &= ~full_manual_overlap
    full_painted_foreground &= ~full_manual_overlap
    full_other &= ~full_manual_overlap
    full_automatic_foreground &= (
        ~full_painted_background
        & ~full_painted_foreground
        & ~full_other
    )
    full_automatic_background = (
        full_reference_mask(background_reference_source_mask)
        & full_source_valid
        & ~full_painted_background
        & ~full_painted_foreground
        & ~full_automatic_foreground
        & ~full_other
        & ~full_annotations
    )
    full_background = full_painted_background | full_automatic_background
    full_foreground_source = (
        full_painted_foreground | full_automatic_foreground
    )

    source_counts = (
        ("background", int(full_background.sum().item())),
        ("foreground", int(full_foreground_source.sum().item())),
        ("other", int(full_other.sum().item())),
        ("physical_edge", 0),
        ("non_edge", 0),
    )
    del (
        full_background,
        full_foreground_source,
        full_manual_overlap,
        full_source_valid,
    )
    sample_count_units = (
        ("background", "source reference pixels"),
        ("foreground", "source reference pixels"),
        ("other", "source reference pixels"),
        ("physical_edge", "edge-working-resolution strip samples"),
        ("non_edge", "edge-working-resolution strip samples"),
    )
    patch_size = max(
        12,
        min(
            128,
            round(
                float(seed_diameter)
                * float(settings.reference_texture_patch_fraction)
            ),
        ),
    )
    maximum = int(settings.reference_texture_working_maximum_dimension)
    work_scale = min(1.0, maximum / max(source_height, source_width))
    height = max(16, round(source_height * work_scale))
    width = max(16, round(source_width * work_scale))
    edge_work_scale, edge_height, edge_width = _adaptive_edge_working_shape(
        source_height,
        source_width,
        float(seed_diameter),
        material_maximum_dimension=maximum,
        edge_maximum_dimension=int(
            settings.reference_texture_edge_working_maximum_dimension
        ),
        minimum_working_seed_diameter_px=float(
            settings.reference_edge_minimum_working_seed_diameter_px
        ),
    )
    edge_diameter = max(6.0, float(seed_diameter) * edge_work_scale)
    edge_strip_normal_offset_px = max(
        1.0,
        edge_diameter
        * float(settings.reference_edge_strip_normal_offset_fraction),
    ) / max(edge_work_scale, 1e-8)
    edge_strip_tangent_half_length_px = max(
        1.0,
        edge_diameter
        * float(settings.reference_edge_strip_tangent_half_length_fraction),
    ) / max(edge_work_scale, 1e-8)
    material_context_radius_px = max(
        0.7,
        max(6.0, float(seed_diameter) * work_scale)
        * float(settings.reference_texture_context_fraction),
    ) / max(work_scale, 1e-8)

    def resized(values, *, mode="bilinear"):
        tensor = values.float()
        if tensor.ndim == 2:
            tensor = tensor[None, None]
        elif tensor.ndim == 3:
            tensor = tensor[None]
        if tensor.shape[-2:] == (height, width):
            return tensor
        arguments = {} if mode in {"nearest", "area"} else {"align_corners": False}
        return functional.interpolate(
            tensor, (height, width), mode=mode, **arguments
        )

    valid = resized(gradients.valid.float(), mode="nearest") > 0.5
    diameter = max(6.0, float(seed_diameter) * work_scale)

    def working_reference_mask(values):
        return resized(values.float(), mode="area") > 0.001

    painted_background_mask = working_reference_mask(
        full_painted_background
    ) & valid
    painted_foreground_mask = working_reference_mask(
        full_painted_foreground
    ) & valid
    automatic_foreground_mask = working_reference_mask(
        full_automatic_foreground
    ) & valid
    other_mask = working_reference_mask(full_other) & valid
    automatic_background_mask = working_reference_mask(
        full_automatic_background
    ) & valid
    if bool(painted_background_mask.any().item()) and float(
        background_automatic_authority
    ) < 0.999:
        # Deterministic spatial thinning preserves the ring's colour coverage
        # while making its sample authority subordinate to reviewed paint.
        yy_authority, xx_authority = torch.meshgrid(
            torch.arange(height, device=context.device),
            torch.arange(width, device=context.device),
            indexing="ij",
        )
        stride = max(
            1, round(1.0 / max(float(background_automatic_authority), 1e-3))
        )
        automatic_background_mask &= (
            (xx_authority + yy_authority * width) % stride == 0
        )[None, None]
    del (
        full_painted_background,
        full_painted_foreground,
        full_automatic_foreground,
        full_other,
        full_annotations,
        full_automatic_background,
    )
    edge = resized(gradients.strength).clamp(0.0, 1.0)
    ridge = resized(
        _raster_tensor(ridges, context, normalized=True)
    ).clamp(0.0, 1.0)
    if (
        not any(count for _class_name, count in source_counts)
        and not has_instance_annotations
        and library_material_bank is None
        and library_edge_bank is None
    ):
        # Generic gradient/ridge support already has its own upstream products.
        # Publishing it again as a learned semantic probability caused the
        # procedural boundary cost to count the same evidence twice.
        zero_field = edge * 0.0
        physical_field = _lazy_float(
            zero_field,
            "unavailable annotation-derived physical prototype compatibility",
        )
        if zero_field.shape[-2:] != (source_height, source_width):
            zero_field = functional.interpolate(
                zero_field,
                (source_height, source_width),
                mode="bilinear",
                align_corners=False,
            )
        zero = _lazy_u8(
            zero_field,
            "unavailable annotation-derived edge prototype compatibility",
        )
        return ReferenceTextureProducts(
            seed_surface_probability=None,
            background_probability=None,
            other_probability=None,
            physical_edge_probability=zero,
            non_edge_probability=zero,
            physical_edge_field=physical_field,
            non_edge_field=physical_field,
            physical_edge_interior_direction_x=_lazy_float(
                gradients.valid.float() * 0.0,
                "unavailable physical-edge interior direction x",
            ),
            physical_edge_interior_direction_y=_lazy_float(
                gradients.valid.float() * 0.0,
                "unavailable physical-edge interior direction y",
            ),
            physical_edge_interior_direction_confidence=_lazy_float(
                gradients.valid.float() * 0.0,
                "unavailable physical-edge interior direction confidence",
            ),
            profile=ReferenceTextureProfile(
                class_sample_counts=source_counts,
                class_sample_count_units=sample_count_units,
                material_feature_names=material_feature_names,
                edge_feature_names=edge_feature_names,
                working_scale=work_scale,
                edge_working_scale=edge_work_scale,
                edge_working_seed_diameter_px=edge_diameter,
                material_context_radius_px=material_context_radius_px,
                edge_strip_normal_offset_px=edge_strip_normal_offset_px,
                edge_strip_tangent_half_length_px=(
                    edge_strip_tangent_half_length_px
                ),
                patch_size_px=patch_size,
            ),
            foreground_sample_count=0,
            background_sample_count=0,
            other_sample_count=0,
            physical_sample_count=0,
            non_edge_sample_count=0,
        )

    lab = resized(gradients.lab)
    sigma = max(
        0.7,
        diameter * float(settings.reference_texture_context_fraction),
    )
    noise_channels = tuple(
        resized(_raster_tensor(raster, context, normalized=True))
        for raster in (
            *frequency_noise.darkness_masks,
            *frequency_noise.colour_masks,
        )
    )
    material_features = _reference_material_feature_tensor(
        lab,
        edge,
        ridge,
        noise_channels,
        context_sigma=sigma,
    )

    # Automatic sources are subordinate to every explicitly painted semantic
    # class.  The annotated-instance source was already inset and cleaned by
    # Foreground segmentation, but the precedence is enforced again here so a
    # direct caller cannot introduce ambiguous material supervision.
    automatic_foreground_mask = (
        automatic_foreground_mask
        & ~painted_background_mask
        & ~painted_foreground_mask
        & ~other_mask
    )
    foreground_mask = painted_foreground_mask | automatic_foreground_mask
    automatic_background_mask = (
        automatic_background_mask
        & ~foreground_mask
        & ~other_mask
        & ~painted_background_mask
    )
    background_mask = painted_background_mask | automatic_background_mask
    material_overlap = (
        background_mask.to(torch.uint8)
        + foreground_mask.to(torch.uint8)
        + other_mask.to(torch.uint8)
    ) > 1
    background_mask &= ~material_overlap
    foreground_mask &= ~material_overlap
    other_mask &= ~material_overlap

    shared_fit_arguments = dict(
        minimum_support=int(
            settings.reference_texture_minimum_samples_per_prototype
        ),
        iterations=int(settings.reference_texture_fit_iterations),
        scale_floor=0.045,
    )
    material_fit_arguments = dict(
        shared_fit_arguments,
        maximum_prototypes=int(
            settings.reference_texture_material_prototypes_per_class
        ),
    )
    local_banks = {
        "background": _fit_feature_prototype_bank(
            material_features,
            background_mask,
            class_name="background",
            priority_mask=painted_background_mask,
            **material_fit_arguments,
        ),
        "foreground": _fit_feature_prototype_bank(
            material_features,
            foreground_mask,
            class_name="foreground",
            priority_mask=painted_foreground_mask,
            **material_fit_arguments,
        ),
        "other": _fit_feature_prototype_bank(
            material_features,
            other_mask,
            class_name="other",
            **material_fit_arguments,
        ),
    }
    banks = dict(local_banks)
    library_foreground_bank = _library_class_prototype_bank(
        library_material_bank,
        "foreground",
        material_feature_names,
        context,
    )
    banks["foreground"] = _select_or_merge_prototype_banks(
        local_banks["foreground"],
        library_foreground_bank,
        source_mode=material_mode,
        current_weight=float(settings.material_prototype_current_reference_weight),
        class_name="foreground",
    )
    tolerance = float(settings.reference_texture_similarity_scale)
    material_scores = {
        name: (
            None
            if bank is None
            else _prototype_bank_similarity(material_features, bank, tolerance)
        )
        for name, bank in banks.items()
    }

    # These outputs share one descriptor, metric, fit procedure, and calibrated
    # competition, unlike the independent foreground/background colour nodes.
    # Kernel similarity is not itself a posterior: ordinary positive examples
    # commonly sit near exp(-0.5). Separate relative class contrast from the
    # unsharpened best-match confidence that reserves unknown mass. A one-bank
    # similarity remains uncalibrated and therefore publishes no probability.
    material_probabilities = _prototype_class_probabilities(
        material_scores,
        valid.float(),
        class_contrast=float(settings.reference_texture_class_contrast),
    )

    foreground_probability = material_probabilities.get("foreground")
    background_probability = material_probabilities.get("background")
    other_probability = material_probabilities.get("other")

    def restored(values, name):
        if values is None:
            return None
        if values.shape[-2:] != (source_height, source_width):
            values = functional.interpolate(
                values,
                (source_height, source_width),
                mode="bilinear",
                align_corners=False,
            )
        return _lazy_u8(values * 255.0, name)

    if not has_instance_annotations and library_edge_bank is None:
        # Material references cannot train either semantic edge class. Avoid
        # both adaptive high-resolution strip passes completely; generic edge
        # and ridge evidence remains available from its owning upstream nodes.
        material_prototypes = []
        for class_name in ("background", "foreground", "other"):
            bank = banks[class_name]
            if bank is None:
                continue
            positions = bank.positions_yx.detach().cpu().numpy()
            weights = bank.weights.detach().cpu().numpy()
            counts = bank.sample_counts.detach().cpu().numpy()
            for position, weight, count in zip(
                positions, weights, counts, strict=True
            ):
                if int(position[0]) < 0 or int(position[1]) < 0:
                    continue
                y_work, x_work = int(position[0]), int(position[1])
                centre_xy = (
                    float(x_work) / max(work_scale, 1e-8),
                    float(y_work) / max(work_scale, 1e-8),
                )
                patch = _reference_patch(
                    crop, centre_xy, patch_size, None
                )
                patch.flags.writeable = False
                material_prototypes.append(
                    ReferenceTexturePrototype(
                        class_name=class_name,
                        patch_bgr=patch,
                        weight=float(weight),
                        sample_count=int(count),
                        centre_xy=centre_xy,
                        tangent_degrees=None,
                    )
                )
        zero_field = edge * 0.0
        zero = restored(
            zero_field,
            "unavailable annotation-derived edge prototype compatibility",
        )
        return ReferenceTextureProducts(
            seed_surface_probability=restored(
                foreground_probability,
                "reference prototype seed-surface probability",
            ),
            background_probability=restored(
                background_probability,
                "reference prototype background probability",
            ),
            other_probability=restored(
                other_probability,
                "reference prototype other probability",
            ),
            physical_edge_probability=zero,
            non_edge_probability=zero,
            physical_edge_field=_lazy_float(
                zero_field,
                "unavailable annotation-derived physical prototype compatibility",
            ),
            non_edge_field=_lazy_float(
                zero_field,
                "unavailable annotation-derived non-physical prototype compatibility",
            ),
            physical_edge_interior_direction_x=_lazy_float(
                gradients.valid.float() * 0.0,
                "unavailable physical-edge interior direction x",
            ),
            physical_edge_interior_direction_y=_lazy_float(
                gradients.valid.float() * 0.0,
                "unavailable physical-edge interior direction y",
            ),
            physical_edge_interior_direction_confidence=_lazy_float(
                gradients.valid.float() * 0.0,
                "unavailable physical-edge interior direction confidence",
            ),
            profile=ReferenceTextureProfile(
                prototypes=tuple(material_prototypes),
                class_sample_counts=source_counts,
                class_sample_count_units=sample_count_units,
                material_feature_names=material_feature_names,
                edge_feature_names=edge_feature_names,
                working_scale=work_scale,
                edge_working_scale=edge_work_scale,
                edge_working_seed_diameter_px=edge_diameter,
                material_context_radius_px=material_context_radius_px,
                edge_strip_normal_offset_px=edge_strip_normal_offset_px,
                edge_strip_tangent_half_length_px=(
                    edge_strip_tangent_half_length_px
                ),
                patch_size_px=patch_size,
                material_reference_source_mode=material_mode,
                edge_reference_source_mode=edge_mode,
                current_material_sample_count=sum(
                    count for name, count in source_counts
                    if name in {"background", "foreground", "other"}
                ),
                library_material_source_count=library_material_source_count,
                library_material_prototype_count=library_material_prototype_count,
                current_edge_sample_count=0,
                library_edge_source_count=library_edge_source_count,
                library_edge_prototype_count=library_edge_prototype_count,
            ),
            foreground_sample_count=dict(source_counts)["foreground"],
            background_sample_count=dict(source_counts)["background"],
            other_sample_count=dict(source_counts)["other"],
            physical_sample_count=0,
            non_edge_sample_count=0,
        )

    def edge_resized(values, *, mode="bilinear"):
        tensor = values.float()
        if tensor.ndim == 2:
            tensor = tensor[None, None]
        elif tensor.ndim == 3:
            tensor = tensor[None]
        if tensor.shape[-2:] == (edge_height, edge_width):
            return tensor
        arguments = {} if mode in {"nearest", "area"} else {
            "align_corners": False
        }
        return functional.interpolate(
            tensor,
            (edge_height, edge_width),
            mode=mode,
            **arguments,
        )

    edge_valid = edge_resized(
        gradients.valid.float(), mode="nearest"
    ) > 0.5
    edge_strength = edge_resized(gradients.strength).clamp(0.0, 1.0)
    edge_ridge = edge_resized(
        _raster_tensor(ridges, context, normalized=True)
    ).clamp(0.0, 1.0)
    edge_lab = edge_resized(gradients.lab)
    edge_tangent_x = edge_resized(gradients.tangent_x)
    edge_tangent_y = edge_resized(gradients.tangent_y)
    physical_mask = torch.zeros_like(edge_valid)
    non_edge_mask = torch.zeros_like(edge_valid)
    physical_tangent_degrees = None
    safe_interior = None
    strip_tangent_x = edge_tangent_x
    strip_tangent_y = edge_tangent_y
    if has_instance_annotations:
        import cv2
        from seedvision.annotation.instance_references import (
            instance_boundary_references,
        )

        labels = np.asarray(seed_instance_annotations)
        if labels.shape != (source_height, source_width):
            raise ValueError(
                "Seed instance annotations must match the corrected image."
            )
        if labels.shape != (edge_height, edge_width):
            edge_labels = cv2.resize(
                labels.astype(np.float32, copy=False),
                (edge_width, edge_height),
                interpolation=cv2.INTER_NEAREST,
            ).astype(labels.dtype, copy=False)
        else:
            edge_labels = labels
        edge_boundary_references = instance_boundary_references(
            edge_labels,
            edge_diameter,
            interior_buffer_fraction=float(
                settings.reference_texture_instance_interior_buffer_fraction
            ),
        )
        contour, outward_x_numpy, outward_y_numpy = _instance_outward_normals(
            edge_labels
        )
        # The helper's contour omits only degenerate pixels whose local
        # different-ID directions cancel exactly. Those pixels have no reliable
        # inside/outside orientation and must not become signed supervision.
        physical_mask = (
            image_to_tensor(contour.astype(np.uint8), context) > 0.5
        ) & edge_valid
        safe_interior = (
            image_to_tensor(
                edge_boundary_references.safe_interior.astype(np.uint8),
                context,
            ) > 0.5
        ) & edge_valid & ~physical_mask
        outward_x = image_to_tensor(outward_x_numpy, context)
        outward_y = image_to_tensor(outward_y_numpy, context)
        # Annotation geometry, not image-gradient polarity, defines the strip
        # frame at physical training contours. This remains correct at a weak
        # rim, a two-seed contact, or a coat pattern whose stronger gradient is
        # oblique to the reviewed boundary. The chosen tangent makes the
        # helper's +normal point outward exactly.
        strip_tangent_x = torch.where(
            physical_mask, outward_y, edge_tangent_x
        )
        strip_tangent_y = torch.where(
            physical_mask, -outward_x, edge_tangent_y
        )
        physical_tangent_degrees = torch.rad2deg(
            torch.atan2(-outward_x[0, 0], outward_y[0, 0])
        )

    (
        training_edge_features,
        _unused_training_reverse,
        training_strip_valid,
        _training_strip_normal_x,
        _training_strip_normal_y,
    ) = _edge_strip_feature_maps(
        edge_lab,
        edge_strength,
        edge_ridge,
        strip_tangent_x,
        strip_tangent_y,
        edge_valid,
        edge_diameter,
        normal_offset_fraction=float(
            settings.reference_edge_strip_normal_offset_fraction
        ),
        tangent_half_length_fraction=float(
            settings.reference_edge_strip_tangent_half_length_fraction
        ),
        include_reverse=False,
        include_normals=False,
    )
    physical_mask &= training_strip_valid
    if safe_interior is not None:
        non_edge_mask = _candidate_internal_edge_mask(
            edge_strength,
            edge_ridge,
            edge_tangent_x,
            edge_tangent_y,
            safe_interior,
            ridge_weight=float(settings.reference_edge_ridge_weight),
        ) & training_strip_valid

    edge_overlap = physical_mask & non_edge_mask
    physical_mask &= ~edge_overlap
    non_edge_mask &= ~edge_overlap
    source_counts = tuple(
        (
            class_name,
            (
                int(physical_mask.sum().item())
                if class_name == "physical_edge"
                else int(non_edge_mask.sum().item())
                if class_name == "non_edge"
                else count
            ),
        )
        for class_name, count in source_counts
    )
    edge_fit_arguments = dict(
        minimum_support=int(
            settings.reference_edge_minimum_samples_per_prototype
        ),
        iterations=int(settings.reference_edge_fit_iterations),
        maximum_prototypes=int(
            settings.reference_texture_edge_prototypes_per_class
        ),
        scale_floor=0.055,
    )
    local_edge_banks = {
        "physical_edge": _fit_feature_prototype_bank(
            training_edge_features,
            physical_mask,
            class_name="physical_edge",
            **edge_fit_arguments,
        ),
        "non_edge": _fit_feature_prototype_bank(
            training_edge_features,
            non_edge_mask,
            class_name="non_edge",
            **edge_fit_arguments,
        ),
    }
    edge_banks = {}
    edge_current_weight = float(settings.edge_prototype_current_reference_weight)
    for class_name in ("physical_edge", "non_edge"):
        library_class_bank = _library_class_prototype_bank(
            library_edge_bank,
            class_name,
            edge_feature_names,
            context,
        )
        edge_banks[class_name] = _select_or_merge_prototype_banks(
            local_edge_banks[class_name],
            library_class_bank,
            source_mode=edge_mode,
            current_weight=edge_current_weight,
            class_name=class_name,
        )
    del (
        training_edge_features,
        training_strip_valid,
        strip_tangent_x,
        strip_tangent_y,
    )

    # Reference geometry chooses the training frame only. Global evaluation,
    # including the reviewed coordinates themselves, must use image evidence
    # available at every unlabelled query pixel; otherwise the annotations
    # would leak into their own displayed probabilities. Both image-normal
    # polarities remain explicitly evaluated below.
    (
        edge_features,
        reversed_edge_features,
        strip_valid,
        query_strip_normal_x,
        query_strip_normal_y,
    ) = _edge_strip_feature_maps(
        edge_lab,
        edge_strength,
        edge_ridge,
        edge_tangent_x,
        edge_tangent_y,
        edge_valid,
        edge_diameter,
        normal_offset_fraction=float(
            settings.reference_edge_strip_normal_offset_fraction
        ),
        tangent_half_length_fraction=float(
            settings.reference_edge_strip_tangent_half_length_fraction
        ),
        include_normals=True,
    )

    edge_tolerance = float(settings.reference_edge_similarity_scale)

    def polarity_ambiguous_similarity(bank):
        if bank is None:
            return None, None, None
        forward_similarity = _prototype_bank_similarity(
            edge_features, bank, edge_tolerance
        )
        reverse_similarity = _prototype_bank_similarity(
            reversed_edge_features, bank, edge_tolerance
        )
        # Keep this exact maximum as the sole classifier input. The two
        # operands are returned only so a separate direction product can use
        # information that the probability path intentionally discards.
        return (
            torch.maximum(forward_similarity, reverse_similarity),
            forward_similarity,
            reverse_similarity,
        )

    (
        physical_similarity,
        physical_forward_similarity,
        physical_reverse_similarity,
    ) = polarity_ambiguous_similarity(edge_banks["physical_edge"])
    (
        non_edge_similarity,
        _non_edge_forward_similarity,
        _non_edge_reverse_similarity,
    ) = polarity_ambiguous_similarity(edge_banks["non_edge"])
    del _non_edge_forward_similarity, _non_edge_reverse_similarity
    semantic_valid = edge_valid & strip_valid
    # As for material prototypes, Gaussian-kernel similarity is not a
    # posterior. Keep absolute known-edge confidence tied to the unsharpened
    # strongest match, and apply edge-class contrast only to relative Physical
    # versus Non-physical competition. This function receives no annotation
    # targets or coordinates, so training pixels remain ordinary predictions.
    edge_probabilities = _prototype_class_probabilities(
        {
            "physical_edge": physical_similarity,
            "non_edge": non_edge_similarity,
        },
        semantic_valid.float(),
        class_contrast=float(settings.reference_edge_class_contrast),
        unknown_mass=0.10,
    )
    physical_probability = edge_probabilities.get(
        "physical_edge", torch.zeros_like(edge_strength)
    )
    non_edge_probability = edge_probabilities.get(
        "non_edge", torch.zeros_like(edge_strength)
    )
    physical_edge_field = _lazy_float(
        physical_probability,
        "continuous multi-prototype physical edge compatibility",
    )
    non_edge_field = _lazy_float(
        non_edge_probability,
        "continuous multi-prototype non-physical edge compatibility",
    )
    if physical_forward_similarity is None:
        inward_x = torch.zeros_like(edge_strength)
        inward_y = torch.zeros_like(edge_strength)
        inward_confidence = torch.zeros_like(edge_strength)
    else:
        inward_x, inward_y, inward_confidence = (
            _physical_edge_interior_direction(
                physical_forward_similarity,
                physical_reverse_similarity,
                physical_probability,
                query_strip_normal_x,
                query_strip_normal_y,
                semantic_valid,
            )
        )

    # Interpolate confidence-weighted vectors, not circular hue values. Nearby
    # conflicting orientations cancel naturally and therefore lose confidence
    # instead of producing a spurious colour halfway around the hue wheel.
    weighted_inward_x = inward_x * inward_confidence
    weighted_inward_y = inward_y * inward_confidence
    if weighted_inward_x.shape[-2:] != (source_height, source_width):
        weighted_inward_x = functional.interpolate(
            weighted_inward_x,
            (source_height, source_width),
            mode="bilinear",
            align_corners=False,
        )
        weighted_inward_y = functional.interpolate(
            weighted_inward_y,
            (source_height, source_width),
            mode="bilinear",
            align_corners=False,
        )
    restored_inward_confidence = torch.sqrt(
        weighted_inward_x.square() + weighted_inward_y.square()
    ).clamp(0.0, 1.0)
    restored_inward_confidence *= gradients.valid.float()
    safe_inward_confidence = restored_inward_confidence.clamp_min(1e-6)
    restored_inward_x = torch.where(
        restored_inward_confidence > 1e-6,
        weighted_inward_x / safe_inward_confidence,
        torch.zeros_like(weighted_inward_x),
    )
    restored_inward_y = torch.where(
        restored_inward_confidence > 1e-6,
        weighted_inward_y / safe_inward_confidence,
        torch.zeros_like(weighted_inward_y),
    )
    physical_edge_interior_direction_x = _lazy_float(
        restored_inward_x,
        "physical-edge predicted interior direction x",
    )
    physical_edge_interior_direction_y = _lazy_float(
        restored_inward_y,
        "physical-edge predicted interior direction y",
    )
    physical_edge_interior_direction_confidence = _lazy_float(
        restored_inward_confidence,
        "physical-edge interior direction confidence",
    )

    prototypes = []
    all_banks = {**banks, **edge_banks}
    edge_tangent_degrees = torch.rad2deg(
        torch.atan2(edge_tangent_y[0, 0], edge_tangent_x[0, 0])
    )
    for class_name in (
        "background",
        "foreground",
        "other",
        "physical_edge",
        "non_edge",
    ):
        bank = all_banks[class_name]
        if bank is None:
            continue
        bank_positions = bank.positions_yx
        local_rows = (bank_positions[:, 0] >= 0) & (bank_positions[:, 1] >= 0)
        if not bool(local_rows.any().item()):
            continue
        bank_positions = bank_positions[local_rows]
        positions = bank_positions.detach().cpu().numpy()
        weights = bank.weights[local_rows].detach().cpu().numpy()
        counts = bank.sample_counts[local_rows].detach().cpu().numpy()
        if class_name == "physical_edge":
            angle_field = physical_tangent_degrees
        elif class_name == "non_edge":
            angle_field = edge_tangent_degrees
        else:
            angle_field = None
        angles = (
            angle_field[
                bank_positions[:, 0], bank_positions[:, 1]
            ].detach().cpu().numpy()
            if angle_field is not None
            else np.full(len(positions), np.nan, np.float32)
        )
        for position, weight, count, raw_angle in zip(
            positions, weights, counts, angles, strict=True
        ):
            y_work, x_work = int(position[0]), int(position[1])
            prototype_scale = (
                edge_work_scale
                if class_name in {"physical_edge", "non_edge"}
                else work_scale
            )
            # ``interpolate(..., align_corners=False)`` maps pixel centres, not
            # integer corners. Preserve that same geometry when returning a
            # medoid to the full-resolution provenance overlay.
            centre_xy = (
                (float(x_work) + 0.5) / max(prototype_scale, 1e-8) - 0.5,
                (float(y_work) + 0.5) / max(prototype_scale, 1e-8) - 0.5,
            )
            angle = (
                float(raw_angle)
                if class_name in {"physical_edge", "non_edge"}
                else None
            )
            patch = _reference_patch(crop, centre_xy, patch_size, angle)
            patch.flags.writeable = False
            prototypes.append(
                ReferenceTexturePrototype(
                    class_name=class_name,
                    patch_bgr=patch,
                    weight=float(weight),
                    sample_count=int(count),
                    centre_xy=centre_xy,
                    tangent_degrees=angle,
                )
            )
    profile = ReferenceTextureProfile(
        prototypes=tuple(prototypes),
        class_sample_counts=source_counts,
        class_sample_count_units=sample_count_units,
        material_feature_names=material_feature_names,
        edge_feature_names=edge_feature_names,
        working_scale=work_scale,
        edge_working_scale=edge_work_scale,
        edge_working_seed_diameter_px=edge_diameter,
        material_context_radius_px=material_context_radius_px,
        edge_strip_normal_offset_px=edge_strip_normal_offset_px,
        edge_strip_tangent_half_length_px=(
            edge_strip_tangent_half_length_px
        ),
        patch_size_px=patch_size,
        material_reference_source_mode=material_mode,
        edge_reference_source_mode=edge_mode,
        current_material_sample_count=sum(
            count for name, count in source_counts
            if name in {"background", "foreground", "other"}
        ),
        library_material_source_count=library_material_source_count,
        library_material_prototype_count=library_material_prototype_count,
        current_edge_sample_count=sum(
            count for name, count in source_counts
            if name in {"physical_edge", "non_edge"}
        ),
        library_edge_source_count=library_edge_source_count,
        library_edge_prototype_count=library_edge_prototype_count,
    )
    return ReferenceTextureProducts(
        seed_surface_probability=restored(
            foreground_probability,
            "reference prototype seed-surface probability",
        ),
        background_probability=restored(
            background_probability,
            "reference prototype background probability",
        ),
        other_probability=restored(
            other_probability,
            "reference prototype other probability",
        ),
        physical_edge_probability=restored(
            physical_probability,
            "multi-prototype physical edge compatibility",
        ),
        non_edge_probability=restored(
            non_edge_probability,
            "multi-prototype non-physical edge compatibility",
        ),
        physical_edge_field=physical_edge_field,
        non_edge_field=non_edge_field,
        physical_edge_interior_direction_x=(
            physical_edge_interior_direction_x
        ),
        physical_edge_interior_direction_y=(
            physical_edge_interior_direction_y
        ),
        physical_edge_interior_direction_confidence=(
            physical_edge_interior_direction_confidence
        ),
        profile=profile,
        foreground_sample_count=dict(source_counts)["foreground"],
        background_sample_count=dict(source_counts)["background"],
        other_sample_count=dict(source_counts)["other"],
        physical_sample_count=dict(source_counts)["physical_edge"],
        non_edge_sample_count=dict(source_counts)["non_edge"],
    )


def locally_normalized_reference_edge_probability(
    physical_prototype_compatibility_field,
    thinned_true_edge_support,
    gradients: EdgeGradientProducts,
    seed_diameter: float,
    settings,
    *,
    cuda_context=None,
) -> GpuRaster:
    """Normalize true-edge support before applying Physical compatibility.

    The semantic input is Pphysical, including its known-versus-unknown
    confidence, not a ratio or conservative subtraction. It cannot create
    spatial support. A bounded, winsorized local RMS envelope raises weak *thinned true
    image-edge support* toward a local target without attenuating strong
    support. The final full-resolution mask is restricted to the original
    thinned support, so resize interpolation cannot recreate descriptor halos.
    """

    import torch
    import torch.nn.functional as functional

    context = cuda_context or CudaContext.resolve()
    source_height, source_width = gradients.strength.shape[-2:]
    maximum = int(settings.reference_ridge_working_maximum_dimension)
    scale = min(1.0, maximum / max(source_height, source_width))
    height = max(8, round(source_height * scale))
    width = max(8, round(source_width * scale))

    def resized(values, *, mode="bilinear"):
        tensor = (
            values.to(device=context.device, dtype=torch.float32)
            if torch.is_tensor(values)
            else _raster_tensor(values, context).float()
        )
        if tensor.ndim == 2:
            tensor = tensor[None, None]
        elif tensor.ndim == 3:
            tensor = tensor[None]
        if tensor.shape[-2:] == (height, width):
            return tensor
        arguments = {} if mode in {"nearest", "area"} else {
            "align_corners": False
        }
        return functional.interpolate(
            tensor,
            (height, width),
            mode=mode,
            **arguments,
        )

    def valid_box_mean(values, valid_values, radius):
        """Return a valid-weighted box mean using O(image) integral tensors."""

        radius = max(1, int(radius))
        kernel = radius * 2 + 1
        weighted = values * valid_values

        def window_sum(source):
            padded = functional.pad(
                source,
                (radius, radius, radius, radius),
                mode="constant",
                value=0.0,
            )
            integral = functional.pad(
                padded.cumsum(dim=-2).cumsum(dim=-1),
                (1, 0, 1, 0),
                mode="constant",
                value=0.0,
            )
            return (
                integral[..., kernel:, kernel:]
                - integral[..., :-kernel, kernel:]
                - integral[..., kernel:, :-kernel]
                + integral[..., :-kernel, :-kernel]
            )

        numerator = window_sum(weighted)
        denominator = window_sum(valid_values).clamp_min(1.0)
        return numerator / denominator

    physical_compatibility = resized(physical_prototype_compatibility_field).clamp(
        0.0, 1.0
    )
    valid = resized(gradients.valid.float(), mode="nearest") > 0.5
    valid_float = valid.float()
    epsilon = torch.finfo(physical_compatibility.dtype).eps
    physical_compatibility *= valid_float

    radius = max(
        1,
        round(
            float(seed_diameter)
            * scale
            * float(settings.reference_edge_normalization_radius_fraction)
        ),
    )
    edge_support = resized(
        _raster_tensor(
            thinned_true_edge_support,
            context,
            normalized=True,
        )
    ).clamp(0.0, 1.0) * valid_float
    preliminary_rms = torch.sqrt(
        valid_box_mean(
            edge_support.square(), valid_float, radius
        ).clamp_min(0.0)
    )
    absolute_floor = float(
        settings.reference_edge_normalization_absolute_floor
    )
    winsor_cap = torch.maximum(
        preliminary_rms * 2.5,
        torch.full_like(preliminary_rms, absolute_floor),
    )
    clipped_total = torch.minimum(edge_support, winsor_cap)
    local_envelope = torch.sqrt(
        valid_box_mean(
            clipped_total.square(), valid_float, radius
        ).clamp_min(0.0)
    )
    # A strong section inside the normalization window must not raise the
    # denominator enough to suppress its weak continuation.  Cap the local
    # envelope at the current supported response (with the absolute floor as
    # a stable lower bound); hysteresis, rather than a neighbouring strong
    # response leaking into this statistic, decides whether that weak point
    # belongs to a coherent ridge.
    gain_envelope = torch.minimum(
        local_envelope,
        edge_support.clamp_min(absolute_floor),
    )
    maximum_gain = float(settings.reference_edge_normalization_maximum_gain)
    gain = torch.clamp(
        float(settings.reference_edge_normalization_target_support)
        / (gain_envelope + epsilon),
        min=1.0,
        max=maximum_gain,
    )
    normalized_support = torch.clamp(edge_support * gain, 0.0, 1.0)
    if absolute_floor > 0.0:
        gate_coordinate = torch.clamp(
            edge_support / absolute_floor,
            min=0.0,
            max=1.0,
        )
        absolute_gate = gate_coordinate.square() * (
            3.0 - 2.0 * gate_coordinate
        )
    else:
        absolute_gate = (edge_support > 0.0).to(edge_support.dtype)
    normalized = (
        physical_compatibility
        * normalized_support
        * absolute_gate
        * valid_float
    ).clamp(0.0, 1.0)
    if normalized.shape[-2:] != (source_height, source_width):
        normalized = functional.interpolate(
            normalized,
            (source_height, source_width),
            mode="bilinear",
            align_corners=False,
        )
    # Bilinear restoration must not broaden the one-pixel true-edge support.
    # Reapply the exact full-resolution support footprint after every resize.
    full_support = _raster_tensor(
        thinned_true_edge_support,
        context,
        normalized=True,
    ).clamp(0.0, 1.0)
    normalized *= (full_support > 0.0).to(normalized.dtype)
    normalized *= gradients.valid.float()
    return GpuRaster(
        normalized,
        numpy_dtype=np.float32,
        name="continuous normalized reference-edge probability",
    )


def reference_probability_ridges(
    physical_probability_field,
    gradients: EdgeGradientProducts,
    settings,
    *,
    include_gradient_strength: bool = True,
    restrict_to_source_support: bool = False,
    cuda_context=None,
) -> GpuRaster:
    """Thin an edge-supported reference field without a CPU round trip."""

    import torch.nn.functional as functional

    context = cuda_context or CudaContext.resolve()
    source_height, source_width = gradients.strength.shape[-2:]
    maximum = int(settings.reference_ridge_working_maximum_dimension)
    scale = min(1.0, maximum / max(source_height, source_width))
    height = max(8, round(source_height * scale))
    width = max(8, round(source_width * scale))

    def resized(values, *, mode="bilinear"):
        tensor = values.float()
        if tensor.ndim == 2:
            tensor = tensor[None, None]
        elif tensor.ndim == 3:
            tensor = tensor[None]
        if tensor.shape[-2:] == (height, width):
            return tensor
        arguments = {} if mode in {"nearest", "area"} else {"align_corners": False}
        return functional.interpolate(
            tensor,
            (height, width),
            mode=mode,
            **arguments,
        )

    source_probability = _raster_tensor(
        physical_probability_field, context
    ).clamp(0.0, 1.0)
    probability = resized(source_probability).clamp(0.0, 1.0)
    if include_gradient_strength:
        probability *= resized(gradients.strength).clamp(0.0, 1.0)
    normal_x = resized(gradients.normal_x)
    normal_y = resized(gradients.normal_y)
    valid = resized(gradients.valid.float(), mode="nearest") > 0.5
    ridge = thin_probability_ridges(
        probability,
        normal_x,
        normal_y,
        valid,
        normal_sampling_step_px=(
            float(settings.reference_ridge_nms_step_px) * scale
        ),
        low_threshold=float(settings.reference_ridge_low_threshold),
        high_threshold=float(settings.reference_ridge_high_threshold),
        hysteresis_iterations=int(
            settings.reference_ridge_hysteresis_iterations
        ),
    )
    if ridge.shape[-2:] != (source_height, source_width):
        ridge = functional.interpolate(
            ridge,
            (source_height, source_width),
            mode="bilinear",
            align_corners=False,
        )
    if restrict_to_source_support:
        ridge *= (source_probability > 0.0).to(ridge.dtype)
    ridge *= gradients.valid.float()
    return _lazy_u8(
        ridge * 255.0,
        "thinned reference edge ridge",
    )


def seed_boundary_tracing(
    gradients: EdgeGradientProducts,
    seed_diameter,
    settings,
    *,
    background_likelihood=None,
    foreground_probability=None,
    physical_edge_probability=None,
    non_edge_probability=None,
    instance_labels=None,
    centers=(),
    radii=(),
    cuda_context=None,
    previous: BoundaryTraceProducts | None = None,
    recompute_ridges: bool = True,
    recompute_traces: bool = True,
    compute_final: bool = True,
    trace_ridge_override=None,
    seed_shape_model=None,
    seed_measurement_summary=None,
    trace_source_name: str = "generic_ridges",
    timing_recorder=None,
) -> BoundaryTraceProducts:
    """Trace thinned ridges and confirm circle/ellipse seed boundaries on GPU."""

    import torch
    import torch.nn.functional as functional

    context = cuda_context or CudaContext.resolve()
    source_height, source_width = gradients.strength.shape[-2:]
    maximum = int(settings.boundary_working_maximum_dimension)
    scale = min(1.0, maximum / max(source_height, source_width))
    height = max(8, round(source_height * scale))
    width = max(8, round(source_width * scale))

    def resized(tensor, mode="bilinear"):
        if tensor.shape[-2:] == (height, width):
            return tensor.float()
        kwargs = {} if mode == "nearest" else {"align_corners": False}
        return functional.interpolate(
            tensor.float(), (height, width), mode=mode, **kwargs
        )

    def full_u8(tensor, name, *, mode="bilinear", multiplier=255.0):
        values = tensor
        if values.shape[-2:] != (source_height, source_width):
            kwargs = {} if mode == "nearest" else {"align_corners": False}
            values = functional.interpolate(
                values.float(),
                (source_height, source_width),
                mode=mode,
                **kwargs,
            )
        return _lazy_u8(values * multiplier, name)

    def full_float(tensor, name, *, mode="bilinear"):
        values = tensor
        if values.shape[-2:] != (source_height, source_width):
            kwargs = {} if mode == "nearest" else {"align_corners": False}
            values = functional.interpolate(
                values.float(),
                (source_height, source_width),
                mode=mode,
                **kwargs,
            )
        return _lazy_float(values, name)

    ridge_signature = (
        source_height,
        source_width,
        maximum,
        float(settings.ridge_nms_step_px),
        float(settings.ridge_low_threshold),
        float(settings.ridge_high_threshold),
        int(settings.ridge_hysteresis_iterations),
    )
    cached_ridge = None if previous is None else previous.ridge_state
    if (
        not recompute_ridges
        and isinstance(cached_ridge, GpuRidgeState)
        and cached_ridge.signature == ridge_signature
    ):
        ridge_state = cached_ridge
        edge = ridge_state.edge
        valid = ridge_state.valid
        normal_x = ridge_state.normal_x
        normal_y = ridge_state.normal_y
        tangent_x = ridge_state.tangent_x
        tangent_y = ridge_state.tangent_y
        lightness = ridge_state.lightness
        yy, xx = ridge_state.yy, ridge_state.xx
        nms, weak, accepted = (
            ridge_state.nms,
            ridge_state.weak,
            ridge_state.accepted,
        )
    else:
        ridge_timing = (
            None
            if timing_recorder is None
            else timing_recorder.start("edge_ridges")
        )
        edge = resized(gradients.strength)
        valid = resized(gradients.valid.float(), "nearest") > 0.5
        normal_x = resized(gradients.normal_x)
        normal_y = resized(gradients.normal_y)
        tangent_x = resized(gradients.tangent_x)
        tangent_y = resized(gradients.tangent_y)
        vector_norm = torch.sqrt(
            tangent_x.square() + tangent_y.square()
        ).clamp_min(1e-6)
        tangent_x, tangent_y = tangent_x / vector_norm, tangent_y / vector_norm
        normal_x, normal_y = tangent_y, -tangent_x
        lightness = resized(gradients.lab[:, 0:1])
        yy, xx = torch.meshgrid(
            torch.arange(height, device=context.device, dtype=torch.float32),
            torch.arange(width, device=context.device, dtype=torch.float32),
            indexing="ij",
        )

        # One-pixel ridges from continuous float normals, followed by CUDA
        # hysteresis reconstruction. No display quantization participates.
        nms_step = settings.ridge_nms_step_px * scale
        forward = bilinear_sample(
            edge,
            xx + normal_x[0, 0] * nms_step,
            yy + normal_y[0, 0] * nms_step,
        )
        backward = bilinear_sample(
            edge,
            xx - normal_x[0, 0] * nms_step,
            yy - normal_y[0, 0] * nms_step,
        )
        nms = (
            edge[0, 0]
            * (edge[0, 0] >= forward)
            * (edge[0, 0] >= backward)
            * valid[0, 0]
        )
        weak = nms >= settings.ridge_low_threshold
        accepted = nms >= settings.ridge_high_threshold
        for _ in range(settings.ridge_hysteresis_iterations):
            reached = functional.max_pool2d(
                accepted.float()[None, None], 3, stride=1, padding=1
            )[0, 0] > 0
            accepted = weak & reached
        ridge_state = GpuRidgeState(
            signature=ridge_signature,
            scale=scale,
            height=height,
            width=width,
            edge=edge,
            valid=valid,
            normal_x=normal_x,
            normal_y=normal_y,
            tangent_x=tangent_x,
            tangent_y=tangent_y,
            lightness=lightness,
            yy=yy,
            xx=xx,
            nms=nms,
            weak=weak,
            accepted=accepted,
        )
        if ridge_timing is not None:
            timing_recorder.stop(ridge_timing)

    trace_diameter = max(
        4.0,
        float(seed_diameter) * float(scale) * settings.trace_diameter_multiplier,
    )
    # Seed-balanced, scale/rotation-invariant summary, never annotation locations.
    reference_curvatures = np.asarray(getattr(
        seed_measurement_summary, "boundary_curvature_times_diameter", ()), float)
    curvature_envelope = (tuple(float(v) for v in np.quantile(
        np.abs(reference_curvatures), [.10, .90])) if reference_curvatures.size else None)
    learned_turn = (0. if curvature_envelope is None else float(np.degrees(
        curvature_envelope[1] * max(1, settings.trace_maximum_gap_px + 1) / trace_diameter)))
    trace_tangent_tolerance = min(85., max(settings.trace_tangent_tolerance_degrees, learned_turn))
    trace_accepted = accepted
    if trace_ridge_override is not None:
        trace_ridge = _raster_tensor(
            trace_ridge_override, context, normalized=True
        )
        trace_ridge = resized(trace_ridge)
        # Bilinear restoration/resizing creates low-valued halos beside a
        # one-pixel semantic ridge. The former >0 rule promoted those halos to
        # parallel boundaries. Re-thin against the live continuous normal and
        # break flat ties deterministically.
        trace_values = trace_ridge[0, 0]
        trace_step = max(0.25, float(settings.ridge_nms_step_px) * scale)
        trace_forward = bilinear_sample(
            trace_ridge,
            xx + normal_x[0, 0] * trace_step,
            yy + normal_y[0, 0] * trace_step,
        )
        trace_backward = bilinear_sample(
            trace_ridge,
            xx - normal_x[0, 0] * trace_step,
            yy - normal_y[0, 0] * trace_step,
        )
        trace_accepted = (
            (trace_values > 0.0)
            & (trace_values >= trace_forward)
            & (trace_values > trace_backward)
            & valid[0, 0]
        )
    trace_signature = (
        ridge_signature,
        str(trace_source_name),
        float(trace_diameter),
        float(settings.trace_tangent_tolerance_degrees),
        int(settings.trace_maximum_gap_px),
        str(settings.trace_curvature_policy),
        float(settings.trace_curvature_tolerance_degrees),
        float(settings.trace_window_fraction),
        int(settings.trace_sample_count),
        float(settings.trace_minimum_length_fraction),
        int(settings.trace_junction_max_neighbors),
        curvature_envelope,
    )
    cached_trace = None if previous is None else previous.trace_state
    if (
        not recompute_traces
        and ridge_state is cached_ridge
        and isinstance(cached_trace, GpuTraceState)
        and cached_trace.signature == trace_signature
    ):
        trace_state = cached_trace
        trace_seed = trace_state.trace_seed
        trace_labels = trace_state.trace_labels
        continuity = trace_state.continuity
        gap_confidence = trace_state.gap_confidence
        trace_confidence = trace_state.trace_confidence
    else:
        trace_timing = (
            None
            if timing_recorder is None
            else timing_recorder.start("edge_traces")
        )
        neighbour_count = functional.conv2d(
            trace_accepted.float()[None, None],
            torch.ones((1, 1, 3, 3), device=context.device),
            padding=1,
        )[0, 0] - trace_accepted.float()
        junction_limit = (
            max(int(settings.trace_junction_max_neighbors), 7)
            if trace_ridge_override is not None
            else int(settings.trace_junction_max_neighbors)
        )
        trace_seed = trace_accepted & (neighbour_count <= junction_limit)
        trace_labels = oriented_connected_components(
            trace_seed[None, None],
            tangent_x,
            tangent_y,
            maximum_gap=settings.trace_maximum_gap_px,
            tangent_tolerance_degrees=trace_tangent_tolerance,
            curvature_policy=settings.trace_curvature_policy,
            curvature_tolerance_degrees=(
                settings.trace_curvature_tolerance_degrees
            ),
        ).long()
        component_area = torch.bincount(trace_labels.reshape(-1))
        minimum_trace_length = max(
            2, round(trace_diameter * settings.trace_minimum_length_fraction)
        )
        if trace_ridge_override is not None:
            # Semantic probability ridges are often interrupted by local
            # classifier uncertainty. Keep short oriented fragments visible
            # instead of applying the generic-ridge minimum so strictly that
            # every non-generic source renders black.
            minimum_trace_length = min(
                minimum_trace_length,
                max(2, round(trace_diameter * 0.06)),
            )
        retained_component = component_area >= minimum_trace_length
        retained_component[0] = False
        trace_seed &= retained_component[trace_labels[0, 0]]
        trace_labels = torch.where(
            trace_seed[None, None], trace_labels, torch.zeros_like(trace_labels)
        )

        # Tangent-following path integration measures continuity and allows gaps.
        window = max(2.0, trace_diameter * settings.trace_window_fraction)
        distances = torch.linspace(
            1.0,
            window,
            settings.trace_sample_count,
            device=context.device,
        )
        ridge_field = functional.max_pool2d(
            trace_seed.float()[None, None], 3, stride=1, padding=1
        )
        side_means = []
        side_gap_scores = []
        tolerance_cos = float(
            np.cos(np.deg2rad(trace_tangent_tolerance))
        )
        for sign in (-1.0, 1.0):
            support_samples = []
            for distance in distances:
                sx = xx + tangent_x[0, 0] * distance * sign
                sy = yy + tangent_y[0, 0] * distance * sign
                ridge_sample = bilinear_sample(ridge_field, sx, sy)
                sampled_tx = bilinear_sample(tangent_x, sx, sy)
                sampled_ty = bilinear_sample(tangent_y, sx, sy)
                alignment = torch.abs(
                    sampled_tx * tangent_x[0, 0]
                    + sampled_ty * tangent_y[0, 0]
                )
                if curvature_envelope is not None:
                    observed_k = torch.acos(alignment.clamp(0., 1.)) * trace_diameter / distance
                    lo, hi = curvature_envelope
                    outside = (lo - observed_k).clamp_min(0) + (observed_k - hi).clamp_min(0)
                    shape_support = .5 + .5 * torch.exp(-.5*(outside/max(hi-lo, .5))**2)
                else:
                    shape_support = 1.
                alignment = (
                    (alignment - tolerance_cos)
                    / max(1e-5, 1.0 - tolerance_cos)
                ).clamp(0.0, 1.0)
                support_samples.append(ridge_sample * alignment * shape_support)
            support_stack = torch.stack(support_samples)
            side_means.append(support_stack.mean(dim=0))
            side_gap_scores.append(
                torch.exp(
                    torch.mean(
                        torch.log(0.15 + 0.85 * support_stack), dim=0
                    )
                )
            )
        continuity = torch.sqrt(
            (side_means[0] * side_means[1]).clamp_min(0.0)
        )
        gap_confidence = torch.sqrt(
            (side_gap_scores[0] * side_gap_scores[1]).clamp_min(0.0)
        )
        continuity = continuity * trace_seed
        gap_confidence = gap_confidence * trace_seed
        trace_confidence = torch.sqrt(
            (continuity * gap_confidence).clamp_min(0.0)
        )
        trace_state = GpuTraceState(
            signature=trace_signature,
            trace_seed=trace_seed,
            trace_labels=trace_labels,
            continuity=continuity,
            gap_confidence=gap_confidence,
            trace_confidence=trace_confidence,
        )
        if trace_timing is not None:
            timing_recorder.stop(trace_timing)

    if not compute_final:
        zero = torch.zeros(
            (1, 1, source_height, source_width),
            device=context.device,
            dtype=torch.float32,
        )
        zero_u8 = _lazy_u8(zero, "disabled seed boundary confidence")
        return BoundaryTraceProducts(
            final_likelihood=zero_u8,
            selected_radius=_lazy_float(
                zero, "disabled selected boundary radius"
            ),
            ridges=full_u8(
                (nms * accepted)[None, None], "thinned edge ridges"
            ),
            trace_labels=_lazy_int(
                functional.interpolate(
                    trace_labels.float(),
                    (source_height, source_width),
                    mode="nearest",
                ).round(),
                "oriented edge trace labels",
            ),
            trace_continuity=full_u8(
                continuity[None, None], "trace continuity"
            ),
            gap_confidence=full_u8(
                gap_confidence[None, None], "trace gap confidence"
            ),
            radius_ratio_hue=zero_u8,
            radius_confidence=zero_u8,
            circle_confidence=zero_u8,
            ellipse_confidence=zero_u8,
            fit_residual=zero_u8,
            centre_votes=zero_u8,
            oval_centre_probability=zero_u8,
            semantic_sides=zero_u8,
            rejection_hue=zero_u8,
            rejection_strength=zero_u8,
            geometry=None,
            ridge_state=ridge_state,
            trace_state=trace_state,
        )

    final_timing = (
        None
        if timing_recorder is None
        else timing_recorder.start("seed_edge_curves")
    )

    background = (
        torch.zeros_like(edge)
        if background_likelihood is None
        else resized(_raster_tensor(background_likelihood, context, normalized=True))
    )
    foreground = (
        (1.0 - background).clamp(0.0, 1.0)
        if foreground_probability is None
        else resized(_raster_tensor(foreground_probability, context, normalized=True))
    )
    instance_physical_probability = (
        torch.zeros_like(edge)
        if physical_edge_probability is None
        else resized(
            _raster_tensor(physical_edge_probability, context, normalized=True)
        )[0, 0]
    )
    instance_nonphysical_probability = (
        torch.zeros_like(edge)
        if non_edge_probability is None
        else resized(
            _raster_tensor(non_edge_probability, context, normalized=True)
        )[0, 0]
    )
    ridge_support = functional.max_pool2d(
        (nms * trace_seed)[None, None], 3, stride=1, padding=1
    )
    diameter = max(
        4.0,
        float(seed_diameter) * float(scale) * settings.curve_diameter_multiplier,
    )
    radii_tested = torch.linspace(
        diameter * settings.boundary_radius_min_fraction,
        diameter * settings.boundary_radius_max_fraction,
        settings.boundary_radius_sample_count,
        device=context.device,
    )
    half_span = settings.boundary_arc_span_degrees * 0.5
    angles = torch.linspace(
        -half_span,
        half_span,
        settings.boundary_arc_sample_count,
        device=context.device,
    )
    angles = torch.deg2rad(angles[torch.abs(angles) > 0.5])
    orientation_sigma = np.deg2rad(
        settings.boundary_orientation_tolerance_degrees
    )
    best_circle = torch.zeros_like(nms)
    best_radius = torch.zeros_like(nms)
    best_interior_sign = torch.ones_like(nms)
    best_semantic = torch.zeros_like(nms)
    best_polarity = torch.zeros_like(nms)
    best_centre_x = torch.zeros_like(nms)
    best_centre_y = torch.zeros_like(nms)
    vote_accumulator = torch.zeros_like(nms)
    vote_y, vote_x = torch.nonzero(trace_seed, as_tuple=True)
    flat_shape_family = (
        None if seed_shape_model is None else seed_shape_model.family("flat")
    )
    shape_component = (
        None if flat_shape_family is None else flat_shape_family.component
    )
    prior_ovality = (
        1.0
        if shape_component is None
        else float(
            np.clip(
                shape_component.mean[3],
                1.0,
                settings.boundary_max_axis_ratio,
            )
        )
    )
    prior_ovality_sigma = (
        0.35
        if shape_component is None
        else max(
            0.08,
            float(np.sqrt(max(shape_component.covariance[3][3], 0.0))),
        )
    )
    # Ellipse centres do not generally lie exactly on a boundary point's
    # normal. Tangential offsets broaden the reverse transform just enough to
    # include plausible oval centres while retaining a centred soft prior.
    tangent_reach = min(0.95, 0.60 * np.sqrt(prior_ovality))
    oval_tangent_offsets = tuple(
        float(value)
        for value in np.linspace(-tangent_reach, tangent_reach, 5)
    )

    for radius in radii_tested:
        radius_prior = torch.exp(
            -0.5
            * (
                torch.log(radius / max(diameter * 0.5, 1e-4))
                / settings.boundary_radius_log_tolerance
            ).square()
        )
        if curvature_envelope is not None:
            lo, hi = curvature_envelope
            k = diameter / radius
            outside = (lo - k).clamp_min(0) + (k - hi).clamp_min(0)
            radius_prior *= .5 + .5 * torch.exp(-.5*(outside/max(hi-lo, .5))**2)
        for interior_sign in (-1.0, 1.0):
            inward_x = normal_x[0, 0] * interior_sign
            inward_y = normal_y[0, 0] * interior_sign
            angle_view = angles[:, None, None]
            sine, cosine = torch.sin(angle_view), torch.cos(angle_view)
            sx = (
                xx[None]
                + tangent_x[0, 0][None] * radius * sine
                + inward_x[None] * radius * (1.0 - cosine)
            )
            sy = (
                yy[None]
                + tangent_y[0, 0][None] * radius * sine
                + inward_y[None] * radius * (1.0 - cosine)
            )
            sample_edge = bilinear_sample(ridge_support, sx, sy)
            sample_tx = bilinear_sample(tangent_x, sx, sy)
            sample_ty = bilinear_sample(tangent_y, sx, sy)
            expected_tx = tangent_x[0, 0][None] * cosine + inward_x[None] * sine
            expected_ty = tangent_y[0, 0][None] * cosine + inward_y[None] * sine
            angular_agreement = torch.abs(
                sample_tx * expected_tx + sample_ty * expected_ty
            ).clamp(0.0, 1.0)
            angular_error = torch.acos(angular_agreement)
            orientation_support = torch.exp(
                -0.5 * (angular_error / max(orientation_sigma, 1e-4)).square()
            )
            sample_stack = sample_edge * orientation_support
            support = (
                settings.boundary_missing_support_floor
                + (1.0 - settings.boundary_missing_support_floor)
                * sample_stack.mean(dim=0)
            )
            coverage = (
                sample_edge >= settings.ridge_low_threshold
            ).float().mean(dim=0)
            circle = (
                nms
                * trace_confidence
                * support
                * (0.35 + 0.65 * coverage)
                * (0.55 + 0.45 * radius_prior)
                * trace_seed
            )
            inside_distance = max(1.0, diameter * 0.18)
            outside_distance = max(1.0, diameter * 0.10)
            inside_x, inside_y = xx + inward_x * inside_distance, yy + inward_y * inside_distance
            outside_x, outside_y = xx - inward_x * outside_distance, yy - inward_y * outside_distance
            fg_inside = bilinear_sample(foreground, inside_x, inside_y)
            bg_inside = bilinear_sample(background, inside_x, inside_y)
            bg_outside = bilinear_sample(background, outside_x, outside_y)
            semantic = (
                0.55 * fg_inside
                + 0.25 * (1.0 - bg_inside)
                + 0.20 * torch.relu(bg_outside - bg_inside)
            ).clamp(0.0, 1.0)
            boundary_lightness = lightness[0, 0]
            inside_lightness = bilinear_sample(lightness, inside_x, inside_y)
            polarity = torch.sigmoid((inside_lightness - boundary_lightness - 1.0) / 6.0)
            score = circle * (
                1.0 - settings.boundary_semantic_weight
                + settings.boundary_semantic_weight * semantic
            )
            score *= 1.0 + settings.boundary_polarity_boost * polarity
            update = score > best_circle
            best_circle = torch.where(update, score, best_circle)
            best_radius = torch.where(update, radius, best_radius)
            best_interior_sign = torch.where(
                update, torch.full_like(best_interior_sign, interior_sign), best_interior_sign
            )
            best_semantic = torch.where(update, semantic, best_semantic)
            best_polarity = torch.where(update, polarity, best_polarity)
            centre_x = xx + inward_x * radius
            centre_y = yy + inward_y * radius
            best_centre_x = torch.where(update, centre_x, best_centre_x)
            best_centre_y = torch.where(update, centre_y, best_centre_y)
            for tangent_offset in oval_tangent_offsets:
                offset_x = (
                    centre_x
                    + tangent_x[0, 0] * radius * tangent_offset
                )
                offset_y = (
                    centre_y
                    + tangent_y[0, 0] * radius * tangent_offset
                )
                rounded_x = (
                    torch.round(offset_x[vote_y, vote_x])
                    .long()
                    .clamp(0, width - 1)
                )
                rounded_y = (
                    torch.round(offset_y[vote_y, vote_x])
                    .long()
                    .clamp(0, height - 1)
                )
                flat_index = rounded_y * width + rounded_x
                offset_prior = float(
                    np.exp(-0.5 * (tangent_offset / 0.50) ** 2)
                )
                vote_accumulator.reshape(-1).scatter_add_(
                    0,
                    flat_index,
                    circle[vote_y, vote_x] * offset_prior,
                )

    vote_blur = max(0.7, diameter * settings.boundary_center_vote_blur_fraction)
    vote_accumulator = gaussian_blur(
        vote_accumulator[None, None], vote_blur
    )[0, 0]
    vote_scale = torch.quantile(
        vote_accumulator.reshape(-1), 0.995
    ).clamp_min(1e-6)
    vote_ranking = vote_accumulator / vote_scale
    vote_probability = vote_ranking.clamp(0.0, 1.0)
    centre_support = bilinear_sample(
        vote_probability, best_centre_x, best_centre_y
    ) * trace_seed

    # Batched proposal-conditioned circle and ellipse fits. All scatter
    # reductions and conic residuals remain on the tensor device.
    centers_array = np.asarray(centers, np.float32).reshape(-1, 2)
    proposal_radii = np.asarray(radii, np.float32).reshape(-1)
    ellipse_confidence = torch.zeros_like(nms)
    proposal_circle_confidence = torch.zeros_like(nms)
    ellipse_axes = torch.empty((0, 2), device=context.device)
    ellipse_angles = torch.empty((0,), device=context.device)
    ellipse_fit_values = torch.empty((0,), device=context.device)
    center_tensor = torch.empty((0, 2), device=context.device)
    if instance_labels is not None and len(centers_array):
        labels = resized(
            _raster_tensor(instance_labels, context), "nearest"
        )[0, 0].long().clamp(0, len(centers_array))
        center_tensor = torch.as_tensor(
            centers_array * scale, device=context.device, dtype=torch.float32
        )
        radius_tensor = torch.as_tensor(
            np.maximum(proposal_radii * scale, 1.0),
            device=context.device,
            dtype=torch.float32,
        )
        padded_centres = torch.cat(
            (torch.zeros((1, 2), device=context.device), center_tensor), dim=0
        )
        assigned_centres = padded_centres[labels]
        ux = xx - assigned_centres[:, :, 0]
        uy = yy - assigned_centres[:, :, 1]
        weights = (nms * trace_confidence * (labels > 0)).reshape(-1)
        flat_labels = labels.reshape(-1)
        count = len(centers_array) + 1

        def scatter(values):
            result = torch.zeros(count, device=context.device)
            return result.scatter_add_(0, flat_labels, values.reshape(-1) * weights)

        totals = scatter(torch.ones_like(nms)).clamp_min(1e-5)
        cxx = scatter(ux.square()) / totals
        cyy = scatter(uy.square()) / totals
        cxy = scatter(ux * uy) / totals
        trace_cov = cxx + cyy
        discriminant = torch.sqrt((cxx - cyy).square() + 4.0 * cxy.square())
        eigen_major = ((trace_cov + discriminant) * 0.5).clamp_min(1.0)
        eigen_minor = ((trace_cov - discriminant) * 0.5).clamp_min(1.0)
        major = torch.sqrt(2.0 * eigen_major)
        minor = torch.sqrt(2.0 * eigen_minor)
        fitted_angle = 0.5 * torch.atan2(2.0 * cxy, cxx - cyy)
        proposal_radius_padded = torch.cat(
            (torch.ones(1, device=context.device), radius_tensor), dim=0
        )
        major = 0.72 * major + 0.28 * proposal_radius_padded
        minor = 0.72 * minor + 0.28 * proposal_radius_padded
        axis_ratio = (major / minor.clamp_min(1.0)).clamp(
            1.0, settings.boundary_max_axis_ratio
        )
        minor = major / axis_ratio
        assigned_angle = fitted_angle[labels]
        cosine, sine = torch.cos(assigned_angle), torch.sin(assigned_angle)
        local_x = ux * cosine + uy * sine
        local_y = -ux * sine + uy * cosine
        assigned_major = major[labels].clamp_min(1.0)
        assigned_minor = minor[labels].clamp_min(1.0)
        ellipse_radius = torch.sqrt(
            (local_x / assigned_major).square()
            + (local_y / assigned_minor).square()
        )
        ellipse_residual = torch.abs(ellipse_radius - 1.0)
        ellipse_confidence = torch.exp(
            -0.5
            * (
                ellipse_residual
                / settings.boundary_ellipse_residual_tolerance
            ).square()
        ) * trace_confidence * trace_seed * (labels > 0)
        if shape_component is not None:
            assigned_axis_ratio = axis_ratio[labels]
            ovality_prior = torch.exp(
                -0.5
                * (
                    (assigned_axis_ratio - prior_ovality)
                    / prior_ovality_sigma
                ).square()
            )
            # This is deliberately suggestive: an atypical accession or pose
            # retains half of its image-derived ellipse confidence.
            ellipse_confidence *= 0.50 + 0.50 * ovality_prior
        radial_distance = torch.sqrt(ux.square() + uy.square())
        fitted_circle_radius = scatter(radial_distance) / totals
        fitted_circle_radius = (
            0.72 * fitted_circle_radius + 0.28 * proposal_radius_padded
        )
        circle_residual = torch.abs(
            radial_distance - fitted_circle_radius[labels]
        ) / fitted_circle_radius[labels].clamp_min(1.0)
        proposal_circle_confidence = torch.exp(
            -0.5
            * (
                circle_residual
                / settings.boundary_circle_residual_tolerance
            ).square()
        ) * trace_confidence * trace_seed * (labels > 0)
        ellipse_axes = torch.stack((major[1:], minor[1:]), dim=1)
        ellipse_angles = fitted_angle[1:]
        ellipse_fit_values = (
            totals[1:] / max(diameter * np.pi, 1.0)
        ).clamp(0.0, 1.0)

    circle_confidence = torch.maximum(best_circle, proposal_circle_confidence)
    shape_confidence = torch.maximum(circle_confidence, ellipse_confidence)
    combined_shape = torch.maximum(
        shape_confidence,
        centre_support * settings.boundary_center_vote_weight,
    )
    final = (
        nms
        * (0.20 + 0.80 * trace_confidence)
        * (0.25 + 0.75 * combined_shape)
        * (
            1.0
            - settings.boundary_semantic_weight
            + settings.boundary_semantic_weight * best_semantic
        )
        * (1.0 + settings.boundary_polarity_boost * best_polarity)
        * trace_seed
    ).clamp(0.0, 1.0)
    reference_factor = (
        1.0
        - settings.boundary_instance_edge_influence
        + settings.boundary_instance_edge_influence
        * instance_physical_probability
    ) * (
        1.0
        - settings.boundary_nonphysical_edge_discount
        * instance_nonphysical_probability
    )
    final = (final * reference_factor).clamp(0.0, 1.0)
    fit_residual = (1.0 - shape_confidence).clamp(0.0, 1.0) * trace_seed
    radius_ratio = best_radius / max(diameter * 0.5, 1e-4)
    radius_hue = (
        (torch.log2(radius_ratio.clamp(0.5, 2.0)) + 1.0) * 89.5
    ).clamp(0.0, 179.0) / 179.0

    rejection_reason = torch.zeros_like(nms)
    candidate = nms >= settings.ridge_low_threshold
    rejection_reason = torch.where(candidate & ~accepted, 1.0, rejection_reason)
    rejection_reason = torch.where(accepted & ~trace_seed, 2.0, rejection_reason)
    rejection_reason = torch.where(
        trace_seed & (trace_confidence < 0.20), 3.0, rejection_reason
    )
    rejection_reason = torch.where(
        trace_seed & (trace_confidence >= 0.20) & (shape_confidence < 0.16),
        4.0,
        rejection_reason,
    )
    rejection_reason = torch.where(
        trace_seed
        & (shape_confidence >= 0.16)
        & (final < settings.boundary_minimum_confidence),
        5.0,
        rejection_reason,
    )
    rejection_hue = (rejection_reason / 6.0).clamp(0.0, 1.0)
    rejection_strength = (rejection_reason > 0).float() * nms

    local_maxima = vote_ranking == functional.max_pool2d(
        vote_ranking[None, None],
        max(3, round(diameter * 0.25)) | 1,
        stride=1,
        padding=(max(3, round(diameter * 0.25)) | 1) // 2,
    )[0, 0]
    candidate_votes = torch.where(
        local_maxima & (vote_ranking >= 0.20),
        vote_ranking,
        torch.zeros_like(vote_ranking),
    )
    top_count = min(
        int(settings.boundary_geometry_max_candidates),
        int(candidate_votes.numel()),
    )
    vote_rank_values, vote_indices = torch.topk(
        candidate_votes.reshape(-1), top_count
    )
    vote_values = vote_rank_values.clamp(0.0, 1.0)
    vote_centres = torch.stack(
        ((vote_indices % width).float() / scale, (vote_indices // width).float() / scale),
        dim=1,
    )

    # Fit proposal-independent ovals by working backward from curved ridge
    # fragments. Each accepted ridge pixel already has a locally supported
    # normal/radius projection (best_centre_x/y). Associate those projections
    # with nearby centre-vote maxima, accumulate edge-coordinate moments, then
    # validate the resulting ellipse against its complete sampled perimeter.
    # Procedural proposals and annotation-derived instance labels are never
    # inputs to this fit.
    candidate_centres_working = torch.stack(
        ((vote_indices % width).float(), (vote_indices // width).float()), dim=1
    )
    candidate_enabled = vote_values >= max(
        0.10, float(settings.boundary_minimum_confidence) * 0.75
    )
    edge_mask = trace_seed & (nms >= float(settings.ridge_low_threshold))
    edge_yx = torch.nonzero(edge_mask, as_tuple=False)
    totals = torch.zeros(top_count, device=context.device)
    moment_xx = torch.zeros_like(totals)
    moment_xy = torch.zeros_like(totals)
    moment_yy = torch.zeros_like(totals)
    radius_totals = torch.zeros_like(totals)
    minimum_axis = diameter * float(settings.boundary_radius_min_fraction)
    maximum_axis = diameter * float(settings.boundary_radius_max_fraction)
    association_chunk = 32768
    for start in range(0, int(edge_yx.shape[0]), association_chunk):
        points_yx = edge_yx[start : start + association_chunk]
        point_y = points_yx[:, 0]
        point_x = points_yx[:, 1]
        edge_points = torch.stack((point_x.float(), point_y.float()), dim=1)
        centre_distance_sq = (
            edge_points[:, None, :]
            - candidate_centres_working[None, :, :]
        ).square().sum(dim=2)
        plausible_radius = (
            (centre_distance_sq >= minimum_axis**2)
            & (centre_distance_sq <= maximum_axis**2)
        )
        centre_distance_sq = torch.where(
            candidate_enabled[None, :] & plausible_radius,
            centre_distance_sq,
            torch.full_like(centre_distance_sq, float("inf")),
        )
        nearest_distance_sq, assigned = torch.min(centre_distance_sq, dim=1)
        weight = (
            nms[point_y, point_x]
            * (0.25 + 0.75 * trace_confidence[point_y, point_x])
            * torch.isfinite(nearest_distance_sq)
        )
        assigned_centres = candidate_centres_working[assigned]
        relative_x = point_x.float() - assigned_centres[:, 0]
        relative_y = point_y.float() - assigned_centres[:, 1]
        totals.scatter_add_(0, assigned, weight)
        moment_xx.scatter_add_(0, assigned, weight * relative_x.square())
        moment_xy.scatter_add_(0, assigned, weight * relative_x * relative_y)
        moment_yy.scatter_add_(0, assigned, weight * relative_y.square())
        radius_totals.scatter_add_(
            0, assigned, weight * best_radius[point_y, point_x]
        )

    safe_totals = totals.clamp_min(1e-5)
    covariance_xx = moment_xx / safe_totals
    covariance_xy = moment_xy / safe_totals
    covariance_yy = moment_yy / safe_totals
    covariance_trace = covariance_xx + covariance_yy
    covariance_discriminant = torch.sqrt(
        (covariance_xx - covariance_yy).square()
        + 4.0 * covariance_xy.square()
    )
    eigen_major = (
        (covariance_trace + covariance_discriminant) * 0.5
    ).clamp_min(1.0)
    eigen_minor = (
        (covariance_trace - covariance_discriminant) * 0.5
    ).clamp_min(1.0)
    moment_major = torch.sqrt(2.0 * eigen_major)
    moment_minor = torch.sqrt(2.0 * eigen_minor)
    supported_radius = radius_totals / safe_totals
    supported_radius = torch.where(
        totals > 0.0,
        supported_radius,
        torch.full_like(supported_radius, diameter * 0.5),
    )
    oval_major = 0.78 * moment_major + 0.22 * supported_radius
    oval_minor = 0.78 * moment_minor + 0.22 * supported_radius
    oval_major = oval_major.clamp(min=minimum_axis, max=maximum_axis)
    oval_minor = oval_minor.clamp(min=minimum_axis, max=maximum_axis)
    oval_major, oval_minor = torch.maximum(
        oval_major, oval_minor
    ), torch.minimum(oval_major, oval_minor)
    oval_minor = torch.maximum(
        oval_minor,
        oval_major / float(settings.boundary_max_axis_ratio),
    )
    oval_angles = 0.5 * torch.atan2(
        2.0 * covariance_xy, covariance_xx - covariance_yy
    )

    # Uniform angular sectors make a short, excellent arc insufficient by
    # itself: confidence requires distributed support around the proposed oval.
    perimeter_sector_count = 16
    perimeter_samples_per_sector = max(
        4, int(np.ceil(settings.boundary_arc_sample_count / 2.0))
    )
    perimeter_sample_count = (
        perimeter_sector_count * perimeter_samples_per_sector
    )
    perimeter_angles = torch.arange(
        perimeter_sample_count,
        device=context.device,
        dtype=torch.float32,
    ) * (2.0 * np.pi / perimeter_sample_count)
    perimeter_cos = torch.cos(perimeter_angles)[None, None, :]
    perimeter_sin = torch.sin(perimeter_angles)[None, None, :]

    # Moment fits provide orientation and one useful axis hypothesis, but
    # fragmented traces can bias their eccentricity. Search a compact bank of
    # seed-scaled semi-axis pairs at that orientation and retain the one with
    # the best complete-perimeter support.
    axis_ratio_count = max(
        3,
        int(np.ceil((settings.boundary_max_axis_ratio - 1.0) / 0.20)) + 1,
    )
    axis_ratios = torch.linspace(
        1.0,
        float(settings.boundary_max_axis_ratio),
        axis_ratio_count,
        device=context.device,
    )
    radius_bank, ratio_bank = torch.meshgrid(
        radii_tested,
        axis_ratios,
        indexing="ij",
    )
    ratio_root = torch.sqrt(ratio_bank)
    grid_major = (radius_bank * ratio_root).reshape(-1)
    grid_minor = (radius_bank / ratio_root).reshape(-1)
    grid_valid = (
        (grid_major <= maximum_axis)
        & (grid_minor >= minimum_axis)
    )
    hypothesis_major = torch.cat(
        (
            oval_major[:, None],
            grid_major[None, :].expand(top_count, -1),
        ),
        dim=1,
    )
    hypothesis_minor = torch.cat(
        (
            oval_minor[:, None],
            grid_minor[None, :].expand(top_count, -1),
        ),
        dim=1,
    )
    hypothesis_valid = torch.cat(
        (
            torch.ones((top_count, 1), device=context.device, dtype=torch.bool),
            grid_valid[None, :].expand(top_count, -1),
        ),
        dim=1,
    )
    angle_cos = torch.cos(oval_angles)[:, None, None]
    angle_sin = torch.sin(oval_angles)[:, None, None]
    local_x = hypothesis_major[:, :, None] * perimeter_cos
    local_y = hypothesis_minor[:, :, None] * perimeter_sin
    sample_x = (
        candidate_centres_working[:, 0:1, None]
        + local_x * angle_cos
        - local_y * angle_sin
    )
    sample_y = (
        candidate_centres_working[:, 1:2, None]
        + local_x * angle_sin
        + local_y * angle_cos
    )
    expected_tangent_x = (
        -hypothesis_major[:, :, None] * perimeter_sin * angle_cos
        - hypothesis_minor[:, :, None] * perimeter_cos * angle_sin
    )
    expected_tangent_y = (
        -hypothesis_major[:, :, None] * perimeter_sin * angle_sin
        + hypothesis_minor[:, :, None] * perimeter_cos * angle_cos
    )
    expected_tangent_norm = torch.sqrt(
        expected_tangent_x.square() + expected_tangent_y.square()
    ).clamp_min(1e-5)
    expected_tangent_x /= expected_tangent_norm
    expected_tangent_y /= expected_tangent_norm
    sampled_edge = bilinear_sample(ridge_support, sample_x, sample_y)
    sampled_tangent_x = bilinear_sample(tangent_x, sample_x, sample_y)
    sampled_tangent_y = bilinear_sample(tangent_y, sample_x, sample_y)
    tangent_agreement = torch.abs(
        sampled_tangent_x * expected_tangent_x
        + sampled_tangent_y * expected_tangent_y
    ).clamp(0.0, 1.0)
    tangent_floor = float(
        np.cos(np.deg2rad(settings.boundary_orientation_tolerance_degrees))
    )
    oriented_support = sampled_edge * (
        (tangent_agreement - tangent_floor)
        / max(1e-5, 1.0 - tangent_floor)
    ).clamp(0.0, 1.0)
    sector_support = oriented_support.reshape(
        top_count,
        hypothesis_major.shape[1],
        perimeter_sector_count,
        perimeter_samples_per_sector,
    ).amax(dim=3)
    hypothesis_coverage = (
        sector_support >= float(settings.ridge_low_threshold) * 0.70
    ).float().mean(dim=2)
    hypothesis_mean_support = sector_support.mean(dim=2)
    hypothesis_fit = (
        0.72 * hypothesis_coverage + 0.28 * hypothesis_mean_support
    ) * hypothesis_valid
    best_hypothesis = torch.argmax(hypothesis_fit, dim=1)
    oval_major = hypothesis_major.gather(
        1, best_hypothesis[:, None]
    )[:, 0]
    oval_minor = hypothesis_minor.gather(
        1, best_hypothesis[:, None]
    )[:, 0]
    distributed_coverage = hypothesis_coverage.gather(
        1, best_hypothesis[:, None]
    )[:, 0]
    mean_perimeter_support = hypothesis_mean_support.gather(
        1, best_hypothesis[:, None]
    )[:, 0]
    approximate_circumference = 2.0 * np.pi * torch.sqrt(
        (oval_major.square() + oval_minor.square()) * 0.5
    ).clamp_min(1.0)
    accumulated_support = (
        totals / approximate_circumference
    ).clamp(0.0, 1.0)
    perimeter_fit = (
        0.60 * distributed_coverage
        + 0.25 * mean_perimeter_support
        + 0.15 * accumulated_support
    ).clamp(0.0, 1.0)
    oval_scores = torch.sqrt(
        vote_values.clamp(0.0, 1.0) * perimeter_fit
    )
    oval_scores *= candidate_enabled

    # Suppress duplicate centres on the tensor device. A candidate survives
    # only if no stronger (or earlier equal-strength) hypothesis lies within a
    # seed-scaled neighbourhood.
    pairwise_distance_sq = (
        candidate_centres_working[:, None, :]
        - candidate_centres_working[None, :, :]
    ).square().sum(dim=2)
    candidate_indices = torch.arange(top_count, device=context.device)
    stronger = oval_scores[None, :] > oval_scores[:, None]
    equal_earlier = (
        (oval_scores[None, :] == oval_scores[:, None])
        & (candidate_indices[None, :] < candidate_indices[:, None])
    )
    duplicate_radius_sq = max(2.0, diameter * 0.30) ** 2
    suppressed = (
        (pairwise_distance_sq < duplicate_radius_sq)
        & (stronger | equal_earlier)
    ).any(dim=1)
    accepted_ovals = (
        ~suppressed
        & candidate_enabled
        & (oval_scores >= float(settings.boundary_minimum_confidence))
        & (distributed_coverage >= 0.20)
    )
    oval_scores = oval_scores * accepted_ovals

    # Preserve a raster diagnostic for the accepted ellipse fits. It is a
    # sampled fitted perimeter, not a copy of the source ridge strength.
    final_angle_cos = torch.cos(oval_angles)[:, None]
    final_angle_sin = torch.sin(oval_angles)[:, None]
    final_local_x = oval_major[:, None] * perimeter_cos[:, 0]
    final_local_y = oval_minor[:, None] * perimeter_sin[:, 0]
    final_sample_x = (
        candidate_centres_working[:, 0:1]
        + final_local_x * final_angle_cos
        - final_local_y * final_angle_sin
    )
    final_sample_y = (
        candidate_centres_working[:, 1:2]
        + final_local_x * final_angle_sin
        + final_local_y * final_angle_cos
    )
    oval_edge_indices = (
        torch.round(final_sample_y).long().clamp(0, height - 1) * width
        + torch.round(final_sample_x).long().clamp(0, width - 1)
    )
    oval_edge_values = oval_scores[:, None].expand_as(final_sample_x)
    oval_edge_confidence = torch.zeros(height * width, device=context.device)
    oval_edge_confidence.scatter_reduce_(
        0,
        oval_edge_indices.reshape(-1),
        oval_edge_values.reshape(-1),
        reduce="amax",
        include_self=True,
    )
    oval_edge_confidence = functional.max_pool2d(
        oval_edge_confidence.reshape(1, 1, height, width),
        3,
        stride=1,
        padding=1,
    )[0, 0]
    ellipse_confidence = torch.maximum(
        ellipse_confidence, oval_edge_confidence
    )

    oval_impulses = torch.zeros(height * width, device=context.device)
    oval_indices = (
        torch.round(candidate_centres_working[:, 1])
        .long()
        .clamp(0, height - 1)
        * width
        + torch.round(candidate_centres_working[:, 0])
        .long()
        .clamp(0, width - 1)
    )
    oval_impulses.scatter_reduce_(
        0,
        oval_indices,
        oval_scores,
        reduce="amax",
        include_self=True,
    )
    oval_impulses = oval_impulses.reshape(1, 1, height, width)
    oval_blur_sigma = max(
        0.7,
        diameter * float(settings.boundary_center_vote_blur_fraction),
    )
    oval_centre_probability = gaussian_blur(
        oval_impulses, oval_blur_sigma
    )[0, 0]
    blur_radius = min(96, max(1, round(oval_blur_sigma * 3.0)))
    blur_positions = torch.arange(
        -blur_radius,
        blur_radius + 1,
        device=context.device,
        dtype=torch.float32,
    )
    blur_kernel = torch.exp(
        -0.5 * (blur_positions / oval_blur_sigma).square()
    )
    blur_peak = (1.0 / blur_kernel.sum()).square()
    oval_centre_probability = (
        oval_centre_probability / blur_peak.clamp_min(1e-6)
    ).clamp(0.0, 1.0) * valid[0, 0]

    geometry = GpuBoundaryGeometry(
        vote_centres_xy=vote_centres,
        vote_confidence=vote_values,
        instance_centres_xy=center_tensor / max(scale, 1e-6),
        ellipse_axes_xy=ellipse_axes / max(scale, 1e-6),
        ellipse_angle_radians=ellipse_angles,
        ellipse_confidence=ellipse_fit_values,
        oval_centres_xy=(
            candidate_centres_working / max(scale, 1e-6)
        ),
        oval_axes_xy=(
            torch.stack((oval_major, oval_minor), dim=1)
            / max(scale, 1e-6)
        ),
        oval_angle_radians=oval_angles,
        oval_confidence=oval_scores,
    )

    products = BoundaryTraceProducts(
        final_likelihood=full_u8(final[None, None], "seed boundary confidence"),
        selected_radius=full_float(
            (best_radius / max(scale, 1e-6))[None, None],
            "selected boundary radius",
        ),
        ridges=full_u8((nms * accepted)[None, None], "thinned edge ridges"),
        trace_labels=_lazy_int(
            functional.interpolate(
                trace_labels.float(),
                (source_height, source_width),
                mode="nearest",
            ).round(),
            "oriented edge trace labels",
        ),
        trace_continuity=full_u8(continuity[None, None], "trace continuity"),
        gap_confidence=full_u8(gap_confidence[None, None], "trace gap confidence"),
        radius_ratio_hue=full_u8(radius_hue[None, None], "fitted radius ratio hue", multiplier=179.0),
        radius_confidence=full_u8(combined_shape[None, None], "radius confirmation"),
        circle_confidence=full_u8(circle_confidence[None, None], "circle fit confidence"),
        ellipse_confidence=full_u8(ellipse_confidence[None, None], "ellipse fit confidence"),
        fit_residual=full_u8(fit_residual[None, None], "shape fit residual"),
        centre_votes=full_u8(vote_probability[None, None], "seed centre votes"),
        oval_centre_probability=full_u8(
            oval_centre_probability[None, None],
            "oval-derived seed centre probability",
        ),
        semantic_sides=full_u8(best_semantic[None, None], "semantic side confidence"),
        rejection_hue=full_u8(rejection_hue[None, None], "boundary rejection reason hue", multiplier=179.0),
        rejection_strength=full_u8(rejection_strength[None, None], "boundary rejection strength"),
        geometry=geometry,
        ridge_state=ridge_state,
        trace_state=trace_state,
    )
    if final_timing is not None:
        timing_recorder.stop(final_timing)
    return products


def seed_edge_curve_likelihood(
    crop,
    valid_mask,
    edge_likelihood,
    directed_hue,
    seed_diameter,
    settings=None,
    *,
    cuda_context=None,
):
    from seedvision.visualization.layers import AnalysisLayerSettings
    import torch

    settings = settings or AnalysisLayerSettings()
    context = cuda_context or CudaContext.resolve()
    source = image_to_tensor(crop, context)
    lightness = bgr_to_lab(source)[:, 0:1]
    edge = image_to_tensor(edge_likelihood, context) / 255.0
    hue = image_to_tensor(directed_hue, context)
    valid = image_to_tensor(valid_mask, context) > 0
    height, width = valid.shape[-2:]
    yy, xx = torch.meshgrid(
        torch.arange(height, device=context.device, dtype=torch.float32),
        torch.arange(width, device=context.device, dtype=torch.float32),
        indexing="ij",
    )
    tangent = torch.deg2rad(hue[0, 0] * 2.0)
    tangent_x, tangent_y = torch.cos(tangent), torch.sin(tangent)
    inward_x, inward_y = tangent_y, -tangent_x
    arc = np.deg2rad(settings.curve_arc_angle_degrees)
    tolerance = settings.curve_orientation_tolerance_degrees
    diameter = seed_diameter * settings.curve_diameter_multiplier
    radii = (
        diameter * settings.curve_radius_low_fraction,
        diameter * settings.curve_radius_nominal_fraction,
        diameter * settings.curve_radius_high_fraction,
    )
    scores = []
    for radius in radii:
        tangent_offset = radius * np.sin(arc)
        inward_offset = radius * (1.0 - np.cos(arc))
        side_edges = []
        side_orientations = []
        for sign in (-1.0, 1.0):
            sample_x = xx + tangent_x * (sign * tangent_offset) + inward_x * inward_offset
            sample_y = yy + tangent_y * (sign * tangent_offset) + inward_y * inward_offset
            neighbor_edge = bilinear_sample(edge, sample_x, sample_y)
            neighbor_hue = bilinear_sample(hue, sample_x, sample_y) * 2.0
            # Image y increases downwards; displacement along the encoded
            # tangent therefore rotates the neighbouring tangent by -sign.
            expected = torch.remainder(
                hue[0, 0] * 2.0 - sign * settings.curve_arc_angle_degrees,
                360.0,
            )
            difference = torch.abs(torch.remainder(neighbor_hue - expected + 180.0, 360.0) - 180.0)
            orientation = torch.exp(-torch.square(difference / tolerance))
            neighbor_valid = bilinear_sample(valid.float(), sample_x, sample_y)
            side_edges.append(neighbor_edge * neighbor_valid)
            side_orientations.append(orientation * neighbor_valid)
        edge_support = torch.sqrt((side_edges[0] * side_edges[1]).clamp_min(0.0))
        orientation_support = torch.sqrt(
            (side_orientations[0] * side_orientations[1]).clamp_min(0.0)
        )
        score = torch.sqrt((edge[0, 0] * edge_support).clamp_min(0.0))
        score *= orientation_support

        outward_x, outward_y = -inward_x, -inward_y
        near_distance = max(1.0, diameter * settings.curve_near_distance_fraction)
        deep_distance = max(near_distance + 1.0, diameter * settings.curve_deep_distance_fraction)
        near = bilinear_sample(lightness, xx + inward_x * near_distance, yy + inward_y * near_distance)
        deep = bilinear_sample(lightness, xx + inward_x * deep_distance, yy + inward_y * deep_distance)
        outside = bilinear_sample(lightness, xx + outward_x * near_distance, yy + outward_y * near_distance)
        darkening = torch.maximum(deep - near, outside - near)
        boost = torch.clamp(
            (darkening - settings.curve_min_darkening)
            / max(settings.curve_full_darkening - settings.curve_min_darkening, 1e-5),
            0.0,
            1.0,
        )
        score *= 1.0 + settings.curve_lightness_boost * boost
        scores.append(score.clamp(0.0, 1.0) * valid[0, 0])
    stack = torch.stack(scores)
    best_score, best_index = torch.max(stack, dim=0)
    radius_values = torch.as_tensor(radii, device=context.device, dtype=torch.float32)
    selected_radius = radius_values[best_index] * (best_score > 0.01)
    return (
        _lazy_u8(
            best_score[None, None] * 255.0,
            "legacy seed-edge curve likelihood",
        ),
        _lazy_float(
            selected_radius[None, None],
            "legacy seed-edge selected radius",
        ),
    )


def _foreground_noise_settings(settings):
    """Present foreground-prefixed controls to the shared noise classifier."""

    from types import SimpleNamespace

    return SimpleNamespace(
        noise_medium_scale_fraction=settings.foreground_noise_medium_scale_fraction,
        noise_coarse_scale_fraction=settings.foreground_noise_coarse_scale_fraction,
        noise_direction_step_degrees=settings.foreground_noise_direction_step_degrees,
        noise_vector_length_fraction=settings.foreground_noise_vector_length_fraction,
        noise_vector_sample_count=settings.foreground_noise_vector_sample_count,
        noise_vector_decay=settings.foreground_noise_vector_decay,
        noise_direction_integration=settings.foreground_noise_direction_integration,
        noise_working_maximum_dimension=settings.foreground_noise_working_maximum_dimension,
    )


def noise_frequency_foreground_likelihood(
    crop,
    valid_mask,
    colour_likelihood,
    seed_diameter,
    settings,
    *,
    background_reference_points=(),
    foreground_reference_points=(),
    background_reference_mask=None,
    foreground_reference_mask=None,
    automatic_foreground_reference_mask=None,
    foreground_exclusion_mask=None,
    reference_radius=3,
    cuda_context=None,
    source_tensor=None,
    lab_tensor=None,
    valid_tensor=None,
    library_target_bank=None,
):
    """Classify foreground texture from direct painted or fallback evidence."""

    return noise_frequency_background_likelihood(
        crop,
        valid_mask,
        colour_likelihood,
        seed_diameter,
        _foreground_noise_settings(settings),
        background_reference_points=foreground_reference_points,
        foreground_reference_points=background_reference_points,
        background_reference_mask=foreground_reference_mask,
        automatic_target_reference_mask=automatic_foreground_reference_mask,
        foreground_reference_mask=background_reference_mask,
        target_exclusion_mask=foreground_exclusion_mask,
        target_name="foreground",
        reference_radius=reference_radius,
        cuda_context=cuda_context,
        source_tensor=source_tensor,
        lab_tensor=lab_tensor,
        valid_tensor=valid_tensor,
        target_reference_precedence=True,
        require_target_reference=True,
        library_target_bank=library_target_bank,
        reference_source_mode=str(settings.foreground_noise_reference_source),
        current_reference_weight=float(
            settings.foreground_noise_current_reference_weight
        ),
    )


def _merge_annotated_instance_seeds(
    centers: np.ndarray,
    radii: np.ndarray,
    annotations: np.ndarray | None,
    seed_diameter: float,
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    """Prepend annotated interiors as seeds and suppress duplicate proposals."""

    automatic_centers = np.asarray(centers, np.float32).reshape(-1, 2)
    automatic_radii = np.asarray(radii, np.float32).reshape(-1)
    if annotations is None or not np.any(annotations):
        return automatic_centers, automatic_radii, ()
    values = np.asarray(annotations)
    rows, columns = np.nonzero(values)
    painted_ids = values[rows, columns]
    identifiers, inverse = np.unique(painted_ids, return_inverse=True)
    counts = np.bincount(inverse).astype(np.float64)
    manual_centers = np.column_stack(
        (
            np.bincount(inverse, weights=columns) / counts,
            np.bincount(inverse, weights=rows) / counts,
        )
    ).astype(np.float32)
    manual_radii = np.full(
        len(manual_centers), max(1.0, float(seed_diameter) * 0.43), np.float32
    )
    if len(automatic_centers):
        displacement = automatic_centers[:, None, :] - manual_centers[None, :, :]
        nearest = np.sqrt(np.sum(displacement * displacement, axis=2)).min(axis=1)
        retain = nearest > max(2.0, float(seed_diameter) * 0.48)
        automatic_centers = automatic_centers[retain]
        automatic_radii = automatic_radii[retain]
    return (
        np.concatenate((manual_centers, automatic_centers), axis=0),
        np.concatenate((manual_radii, automatic_radii), axis=0),
        tuple(int(value) for value in identifiers),
    )


def instance_voronoi(
    valid_mask,
    background_likelihood,
    centers,
    radii,
    seed_diameter,
    settings,
    *,
    cuda_context=None,
    valid_tensor=None,
    seed_instance_annotations: np.ndarray | None = None,
    annotation_identifiers: tuple[int, ...] = (),
):
    import torch

    context = cuda_context or CudaContext.resolve()
    valid = (
        image_to_tensor(valid_mask, context)[0, 0] > 0
        if valid_tensor is None
        else valid_tensor[0, 0].bool()
    )
    background = _raster_tensor(
        background_likelihood, context, normalized=True
    )[0, 0]
    height, width = valid.shape
    if len(centers) == 0:
        return _lazy_int(
            torch.zeros((1, 1, height, width), device=context.device),
            "instance labels",
        )
    yy, xx = torch.meshgrid(
        torch.arange(height, device=context.device, dtype=torch.float32),
        torch.arange(width, device=context.device, dtype=torch.float32),
        indexing="ij",
    )
    best_distance = torch.full((height, width), float("inf"), device=context.device)
    best_label = torch.zeros((height, width), dtype=torch.int32, device=context.device)
    center_tensor = torch.as_tensor(centers, device=context.device, dtype=torch.float32)
    radius_tensor = torch.as_tensor(radii, device=context.device, dtype=torch.float32)
    base_extent = seed_diameter * settings.instance_min_extent_fraction
    maximum_extent = seed_diameter * settings.instance_max_extent_fraction
    extents = torch.clamp(
        radius_tensor * settings.instance_radius_extent_multiplier,
        min=base_extent,
        max=maximum_extent,
    )
    for index in range(0, len(centers), 64):
        local_centers = center_tensor[index : index + 64]
        local_extents = extents[index : index + 64]
        distance = (
            (xx[None] - local_centers[:, 0, None, None]).square()
            + (yy[None] - local_centers[:, 1, None, None]).square()
        ) / local_extents[:, None, None].square().clamp_min(1.0)
        chunk_distance, chunk_index = torch.min(distance, dim=0)
        update = chunk_distance < best_distance
        best_distance = torch.where(update, chunk_distance, best_distance)
        best_label = torch.where(update, (chunk_index + index + 1).to(torch.int32), best_label)
    foreground_gate = background < 0.82
    accepted = valid & foreground_gate & (best_distance <= 1.0)
    labels = torch.where(accepted, best_label, torch.zeros_like(best_label))
    if seed_instance_annotations is not None and annotation_identifiers:
        annotation_values = np.asarray(seed_instance_annotations)
        rows, columns = np.nonzero(annotation_values)
        identifiers = np.asarray(annotation_identifiers)
        mapped = np.zeros(annotation_values.shape, dtype=np.int32)
        painted_ids = annotation_values[rows, columns]
        mapped[rows, columns] = np.searchsorted(identifiers, painted_ids) + 1
        constraints = torch.as_tensor(mapped, device=context.device)
        labels = torch.where(
            valid & (constraints > 0), constraints.to(torch.int32), labels
        )
    return _lazy_int(labels[None, None], "instance labels")


def spatially_contrasting_colours(centers: np.ndarray) -> np.ndarray:
    count = len(centers)
    colours = np.zeros((count + 1, 3), np.uint8)
    if count == 0:
        return colours
    candidates = np.asarray(
        [
            tuple(round(channel * 255.0) for channel in colorsys.hsv_to_rgb((index * 0.61803398875) % 1.0, 0.78, 1.0))
            for index in range(max(64, count * 3))
        ],
        np.float32,
    )
    order = np.argsort(np.asarray(centers)[:, 0] + np.asarray(centers)[:, 1] * 0.37)
    assigned: list[int] = []
    for rank, seed_index in enumerate(order):
        if not assigned:
            choice = 0
        else:
            used = candidates[np.asarray(assigned)]
            colour_distance = np.sqrt(((candidates[:, None] - used[None]) ** 2).sum(axis=2)).min(axis=1)
            colour_distance[np.asarray(assigned)] = -1.0
            choice = int(np.argmax(colour_distance))
        assigned.append(choice)
        colours[int(seed_index) + 1] = candidates[choice].astype(np.uint8)
    return colours
