"""PyTorch implementations of Seed Fiddle's diagnostic layers."""

from __future__ import annotations

import colorsys
from dataclasses import dataclass, field

import numpy as np

from seedvision.cuda.ops import (
    apply_contrastive_negative_evidence,
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
    """Reference-trained physical and apparent-boundary probabilities."""

    physical_probability: GpuRaster
    non_edge_probability: GpuRaster
    physical_field: GpuRaster
    physical_sample_count: int
    non_edge_sample_count: int


@dataclass(slots=True)
class ReferenceTextureProducts:
    """Multi-prototype material and boundary evidence learned per image."""

    seed_surface_probability: GpuRaster | None
    background_probability: GpuRaster | None
    other_probability: GpuRaster | None
    physical_edge_probability: GpuRaster
    non_edge_probability: GpuRaster
    physical_edge_field: GpuRaster
    profile: object
    foreground_sample_count: int
    background_sample_count: int
    other_sample_count: int
    physical_sample_count: int
    non_edge_sample_count: int


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
    background_exclusion_mask: np.ndarray | None = None,
    foreground_exclusion_mask: np.ndarray | None = None,
    physical_edge_reference_mask: np.ndarray | None = None,
    non_edge_reference_mask: np.ndarray | None = None,
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
    foreground_colour_profile=None,
    surrounding_noise_source_tensor=None,
    surrounding_noise_valid_tensor=None,
    surrounding_noise_offset_x: int = 0,
    surrounding_noise_offset_y: int = 0,
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
    painted_reference_dirty = "reference_layers" in dirty
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
            background_reference_samples=background_reference_samples,
            background_reference_sample_count=background_reference_sample_count,
            background_prior_lab=background_prior_lab,
            background_prior_samples_lab=background_prior_samples_lab,
            sample_radius=max(2, round(seed_diameter * settings.background_sample_radius_fraction)),
            settings=settings,
            cuda_context=context,
            source_tensor=source_tensor,
            lab_tensor=lab_tensor,
            valid_tensor=valid_tensor,
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
        )
        values["layer.background"] = background_result
    else:
        background_result = values["layer.background"]
    background, background_mode, reference_count, colour_profile = background_result

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
    if refined_dirty and background_colour_enabled:
        refined_timing = (
            None
            if timing_recorder is None
            else timing_recorder.start("refined_background_likelihood")
        )
        refined_result = noise_frequency_background_likelihood(
            crop,
            valid_mask,
            background,
            seed_diameter,
            settings,
            background_reference_points=background_reference_points,
            foreground_reference_points=foreground_reference_points,
            background_reference_mask=background_reference_mask,
            foreground_reference_mask=foreground_reference_mask,
            target_exclusion_mask=background_exclusion_mask,
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
        )
        values["layer.refined_background"] = refined_result
    else:
        refined_result = values["layer.refined_background"]
    refined_background, noise_profile, directional_background, directional_angles = refined_result
    foreground_noise_dirty = (
        painted_reference_dirty
        or "foreground_segmentation" in dirty
        or "foreground_noise_likelihood" in dirty
        or "layer.foreground_noise" not in values
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
            foreground_probability,
            seed_diameter,
            settings,
            background_reference_points=background_reference_points,
            foreground_reference_points=foreground_reference_points,
            background_reference_mask=background_reference_mask,
            foreground_reference_mask=foreground_reference_mask,
            foreground_exclusion_mask=foreground_exclusion_mask,
            reference_radius=max(
                2,
                round(seed_diameter * settings.background_sample_radius_fraction),
            ),
            cuda_context=context,
            source_tensor=source_tensor,
            lab_tensor=lab_tensor,
            valid_tensor=valid_tensor,
        )
        values["layer.foreground_noise"] = foreground_noise_result
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
        and background_colour_enabled
        and surrounding_noise_source_tensor is not None
        and surrounding_noise_valid_tensor is not None
    ):
        surrounding_result = surrounding_band_noise_likelihood(
            surrounding_noise_source_tensor,
            surrounding_noise_valid_tensor,
            seed_diameter,
            settings,
            noise_profile,
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

        agreed_background = GpuRaster(
            torch.minimum(
                _raster_tensor(background, context),
                _raster_tensor(refined_background, context),
            ).to(torch.uint8),
            numpy_dtype=np.uint8,
            name="agreed background likelihood",
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
        "edge_gradients" in dirty or "layer.edge_gradients" not in values
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

    directed_dirty = (
        gradients_dirty
        or "directed_edges" in dirty
        or "layer.directed_edges" not in values
    )
    if directed_dirty:
        directed_timing = (
            None
            if timing_recorder is None
            else timing_recorder.start("directed_edges")
        )
        edge_likelihood = shared_edge_likelihood
        directed_edge_hue = shared_directed_hue
        values["layer.directed_edges"] = edge_likelihood, directed_edge_hue
        if directed_timing is not None:
            timing_recorder.stop(directed_timing)
    else:
        edge_likelihood, directed_edge_hue = values["layer.directed_edges"]

    undirected_dirty = (
        gradients_dirty
        or "undirected_edges" in dirty
        or "layer.undirected_edges" not in values
    )
    if undirected_dirty:
        undirected_timing = (
            None
            if timing_recorder is None
            else timing_recorder.start("undirected_edges")
        )
        undirected_edge_likelihood = shared_edge_likelihood
        undirected_edge_hue = shared_undirected_hue
        values["layer.undirected_edges"] = undirected_edge_likelihood, undirected_edge_hue
        if undirected_timing is not None:
            timing_recorder.stop(undirected_timing)
    else:
        undirected_edge_likelihood, undirected_edge_hue = values["layer.undirected_edges"]

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
        )
        values["layer.edge_ridges"] = trace_products.ridges
        values["layer.edge_traces"] = (
            trace_products.trace_labels,
            trace_products.trace_continuity,
            trace_products.gap_confidence,
        )
    else:
        trace_products = previous_curve_result

    reference_texture_dirty = (
        painted_reference_dirty
        or gradients_dirty
        or frequency_noise_dirty
        or ridges_dirty
        or "reference_texture_prototypes" in dirty
        or "layer.reference_texture_prototypes" not in values
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
            foreground_reference_mask=foreground_reference_mask,
            other_reference_mask=other_reference_mask,
            physical_reference_mask=physical_edge_reference_mask,
            non_edge_reference_mask=non_edge_reference_mask,
            seed_instance_annotations=seed_instance_annotations,
            cuda_context=context,
        )
        if texture_timing is not None:
            timing_recorder.stop(texture_timing)
        values["layer.reference_texture_prototypes"] = reference_textures
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
                "disabled continuous physical edge probability",
            ),
            profile=ReferenceTextureProfile(),
            foreground_sample_count=0,
            background_sample_count=0,
            other_sample_count=0,
            physical_sample_count=0,
            non_edge_sample_count=0,
        )
        values["layer.reference_texture_prototypes"] = reference_textures
    else:
        reference_textures = values["layer.reference_texture_prototypes"]

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
        reference_edges = ReferenceEdgeProducts(
            reference_textures.physical_edge_probability,
            reference_textures.non_edge_probability,
            reference_textures.physical_edge_field,
            reference_textures.physical_sample_count,
            reference_textures.non_edge_sample_count,
        )
        if reference_timing is not None:
            timing_recorder.stop(reference_timing)
        values["layer.reference_edge_probability"] = reference_edges
    elif reference_edge_dirty:
        zero = _lazy_u8(
            valid_tensor.float() * 0.0, "disabled reference edge probability"
        )
        reference_edges = ReferenceEdgeProducts(
            zero,
            zero,
            _lazy_float(
                valid_tensor.float() * 0.0,
                "disabled continuous reference edge probability",
            ),
            0,
            0,
        )
        values["layer.reference_edge_probability"] = reference_edges
    else:
        reference_edges = values["layer.reference_edge_probability"]

    reference_ridges_dirty = (
        reference_edge_dirty
        or gradients_dirty
        or "reference_edge_ridges" in dirty
        or "layer.reference_edge_ridges" not in values
    )
    if reference_ridges_dirty and reference_edge_ridges_enabled:
        reference_ridge_timing = (
            None
            if timing_recorder is None
            else timing_recorder.start("reference_edge_ridges")
        )
        reference_edge_ridge = reference_probability_ridges(
            reference_edges.physical_field,
            gradient_result,
            settings,
            cuda_context=context,
        )
        if reference_ridge_timing is not None:
            timing_recorder.stop(reference_ridge_timing)
        values["layer.reference_edge_ridges"] = reference_edge_ridge
    elif reference_ridges_dirty:
        reference_edge_ridge = _lazy_u8(
            valid_tensor.float() * 0.0,
            "disabled thinned reference edge ridge",
        )
        values["layer.reference_edge_ridges"] = reference_edge_ridge
    else:
        reference_edge_ridge = values["layer.reference_edge_ridges"]

    curve_dirty = (
        traces_dirty
        or reference_edge_dirty
        or background_dirty
        or instance_dirty
        or "seed_edge_curves" in dirty
        or previous_curve_result is None
    )
    if curve_dirty:
        curve_result = seed_boundary_tracing(
            gradient_result,
            seed_diameter,
            settings,
            background_likelihood=background,
            foreground_probability=foreground_probability,
            physical_edge_probability=reference_edges.physical_probability,
            non_edge_probability=reference_edges.non_edge_probability,
            instance_labels=labels,
            centers=centers,
            radii=radii,
            cuda_context=context,
            previous=trace_products,
            recompute_ridges=False,
            recompute_traces=False,
            compute_final=seed_edge_curves_enabled,
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
        reference_seed_surface_probability=(
            reference_textures.seed_surface_probability
        ),
        reference_background_texture_probability=(
            reference_textures.background_probability
        ),
        reference_other_texture_probability=reference_textures.other_probability,
        reference_texture_profile=reference_textures.profile,
        physical_edge_probability=reference_edges.physical_probability,
        non_edge_probability=reference_edges.non_edge_probability,
        reference_edge_ridges=reference_edge_ridge,
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
    background_reference_samples=None,
    background_reference_sample_count=0,
    background_prior_lab=None,
    background_prior_samples_lab=None,
    sample_radius=3,
    settings=None,
    cuda_context=None,
    source_tensor=None,
    lab_tensor=None,
    valid_tensor=None,
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
    mode = "automatic"
    reference_count = 0
    supplied = None
    source_values = None
    eligible = valid & ~foreground_point_mask & ~background_exclusion
    if _sample_count(background_reference_samples):
        supplied = _bgr_samples_to_lab(background_reference_samples, context)
        mode = "manual"
        reference_count = int(background_reference_sample_count)
    elif background_reference_points or background_point_mask.any():
        if background_point_mask.any():
            supplied = lab[background_point_mask]
            source_values = source[0].permute(1, 2, 0)[background_point_mask]
            mode = "manual"
            reference_count = (
                int(background_point_mask.sum().item())
                if background_reference_mask is not None
                else len(background_reference_points)
            )

    automatic_prior_samples = False
    if supplied is None and _sample_count(background_prior_samples_lab):
        supplied = background_prior_samples_lab
        automatic_prior_samples = True

    if supplied is None:
        if background_prior_lab is not None:
            prior = torch.as_tensor(
                background_prior_lab,
                device=context.device,
                dtype=lab.dtype,
            )
            prior_distance = torch.sqrt(
                (lab[:, :, 0] - prior[0]).square()
                + 1.5 * (lab[:, :, 1] - prior[1]).square()
                + 1.5 * (lab[:, :, 2] - prior[2]).square()
            )
            candidates = eligible & (
                prior_distance <= settings.background_prior_tolerance
            )
        else:
            valid_lab = lab[eligible]
            chroma = torch.sqrt((valid_lab[:, 1] - 128.0).square() + (valid_lab[:, 2] - 128.0).square())
            chroma_limit = torch.quantile(chroma, settings.background_chroma_percentile / 100.0)
            lightness_limit = torch.quantile(valid_lab[:, 0], settings.background_lightness_percentile / 100.0)
            candidates = eligible & (
                torch.sqrt((lab[:, :, 1] - 128.0).square() + (lab[:, :, 2] - 128.0).square()) <= chroma_limit
            ) & (lab[:, :, 0] >= lightness_limit)
        minimum = max(32, round(valid.sum().item() * settings.background_minimum_sample_fraction))
        if int(candidates.sum().item()) < minimum:
            if background_prior_lab is not None:
                # In a densely filled dish there may be no visible interior
                # tray. Do not satisfy the minimum by relabelling the closest
                # seed colours as background: retain the independently measured
                # outside-dish prior as the model centre.
                source_values = source[0].permute(1, 2, 0)[candidates]
                if not int(candidates.sum().item()):
                    nearest = torch.argmin(
                        torch.where(
                            eligible,
                            prior_distance,
                            torch.full_like(prior_distance, float("inf")),
                        )
                    )
                    source_values = source[0].permute(1, 2, 0).reshape(-1, 3)[
                        nearest : nearest + 1
                    ]
                supplied = prior[None, :]
            else:
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
        if supplied is None:
            supplied = lab[candidates]
            source_values = source[0].permute(1, 2, 0)[candidates]
    else:
        candidates = None

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
        refinement_iterations=(
            0 if automatic_prior_samples else settings.background_refinement_iterations
        ),
        refinement_min_probability=settings.background_refinement_min_probability,
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
        output_mask=valid,
    )
    dominant_component = int(torch.argmax(component_weights).item())
    centre = component_centres[dominant_component]
    scale = component_scales[dominant_component]
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
        # Other is a competing learned colour model, not a global veto. Glass
        # and seed/background colours can overlap, so suppress background only
        # where Other fits better than the positive background distribution.
        likelihood = apply_contrastive_negative_evidence(
            likelihood,
            excluded_membership,
            strength=exclusion_strength,
        )

    if source_values is None:
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
    )
    return _lazy_u8(
        likelihood[None, None] * 255.0, "background colour likelihood"
    ), mode, reference_count, profile


def empty_noise_frequency_profile(seed_diameter, settings):
    from seedvision.visualization.layers import NoiseFrequencyProfile

    scales = _noise_scales(seed_diameter, settings)
    return NoiseFrequencyProfile(scales, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0, 0, 0.0)


def background_colour_profile_likelihood(
    source_tensor,
    valid_tensor,
    profile,
    settings,
    *,
    cuda_context=None,
):
    """Evaluate an existing background colour model over a bounded GPU crop.

    This intentionally reuses the exact fitted positive and contrastive
    exclusion components from :func:`background_colour_likelihood`. It does
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
    if profile.excluded_component_centres_lab:
        excluded = evaluate(
            profile.excluded_component_centres_lab,
            profile.excluded_component_scales_lab,
            profile.excluded_component_weights,
            0.0,
            combine_modes="sum",
        )
        probability = apply_contrastive_negative_evidence(
            probability,
            excluded,
            strength=profile.exclusion_strength,
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
    foreground_reference_mask=None,
    target_exclusion_mask=None,
    target_name="background",
    reference_radius=3,
    cuda_context=None,
    source_tensor=None,
    lab_tensor=None,
    valid_tensor=None,
    target_reference_precedence=False,
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
    full_colour = _raster_tensor(colour_likelihood, context, normalized=True)
    source_height, source_width = full_valid.shape[-2:]
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

    work_scale = min(
        1.0,
        float(settings.noise_working_maximum_dimension)
        / max(source_height, source_width),
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
        target_reference = full_target_reference
        nontarget_reference = full_nontarget_reference
        target_exclusion = full_target_exclusion
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
        target_reference = functional.adaptive_max_pool2d(
            full_target_reference[None, None].float(), (height, width)
        )[0, 0] > 0.0
        nontarget_reference = functional.adaptive_max_pool2d(
            full_nontarget_reference[None, None].float(), (height, width)
        )[0, 0] > 0.0
        target_exclusion = functional.adaptive_max_pool2d(
            full_target_exclusion[None, None].float(), (height, width)
        )[0, 0] > 0.0
    reported_scales = _noise_scales(seed_diameter, settings)
    scales = _noise_scales(seed_diameter * work_scale, settings)
    fine = gaussian_blur(lab, max(0.55, scales[0]))
    medium = gaussian_blur(lab, scales[1])
    coarse = gaussian_blur(lab, scales[2])
    residuals = (lab - fine, fine - medium, medium - coarse)
    features = []
    for residual, sigma in zip(residuals, scales, strict=True):
        energy = torch.mean(residual.square(), dim=1, keepdim=True)
        local = gaussian_blur(energy, max(0.65, min(2.5, sigma * 0.35)))
        features.append(torch.log1p(torch.sqrt(local.clamp_min(1e-10)) * 255.0))
    feature = torch.cat(features, dim=1)
    eligible = valid
    if not bool(eligible.any().item()):
        return (
            _lazy_u8(full_colour * 0.0, f"excluded {target_name} noise likelihood"),
            empty_noise_frequency_profile(seed_diameter, settings),
            (),
            (),
        )
    if target_reference_precedence:
        nontarget_reference &= ~target_reference
    else:
        target_reference &= ~nontarget_reference
    automatic_target = (
        eligible
        & ~target_exclusion[None, None]
        & ~nontarget_reference[None, None]
        & (colour >= settings.noise_background_min_likelihood / 255.0)
    )
    automatic_nontarget = (
        eligible
        & ~target_reference[None, None]
        & (colour <= settings.noise_nonbackground_max_likelihood / 255.0)
    )
    minimum_samples = max(32, round(int(valid.sum().item()) * 0.002))
    if bool(target_reference.any().item()):
        # Painted target regions supply the actual texture observations. Colour
        # pseudo-labels are used only when the user has not painted this class.
        confident_background = target_reference[None, None]
    else:
        confident_background = automatic_target
    if (
        not bool(target_reference.any().item())
        and int(confident_background.sum().item()) < minimum_samples
    ):
        threshold = torch.quantile(colour[eligible], 0.75)
        confident_background = (
            eligible
            & ~target_exclusion[None, None]
            & ~nontarget_reference[None, None]
            & (colour >= threshold)
        )
    manual_nontarget = nontarget_reference | target_exclusion
    if bool(manual_nontarget.any().item()):
        confident_nonbackground = manual_nontarget[None, None]
    else:
        confident_nonbackground = automatic_nontarget
    if (
        not bool(manual_nontarget.any().item())
        and int(confident_nonbackground.sum().item()) < minimum_samples
    ):
        threshold = torch.quantile(colour[eligible], 0.25)
        confident_nonbackground = (
            eligible
            & ~target_reference[None, None]
            & (colour <= threshold)
        )

    bg_values = feature.permute(0, 2, 3, 1)[confident_background.permute(0, 2, 3, 1).expand(-1, -1, -1, 3)].reshape(-1, 3)
    non_values = feature.permute(0, 2, 3, 1)[confident_nonbackground.permute(0, 2, 3, 1).expand(-1, -1, -1, 3)].reshape(-1, 3)
    bg_center, bg_scale = _robust_tensor_distribution(bg_values)
    non_center, non_scale = _robust_tensor_distribution(non_values)
    values = feature.permute(0, 2, 3, 1)
    bg_log = -0.5 * (((values - bg_center) / bg_scale).square() + 2.0 * torch.log(bg_scale)).sum(dim=-1, keepdim=True)
    non_log = -0.5 * (((values - non_center) / non_scale).square() + 2.0 * torch.log(non_scale)).sum(dim=-1, keepdim=True)
    texture_probability = torch.sigmoid((bg_log - non_log) * 0.72).permute(0, 3, 1, 2)
    base = (0.72 * texture_probability + 0.28 * colour).clamp(1e-4, 1.0) * valid
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
    directional_tensors = []
    for degrees in angles:
        radians = np.deg2rad(degrees)
        distance_view = distances[:, None, None]
        sample_x = xx[None] + float(np.cos(radians)) * distance_view
        sample_y = yy[None] + float(np.sin(radians)) * distance_view
        sample = bilinear_sample(base, sample_x, sample_y).clamp_min(1e-4)
        sample_valid = bilinear_sample(valid.float(), sample_x, sample_y)
        weight_view = weights[:, None, None]
        accumulated = (
            torch.log(sample) * weight_view * sample_valid
        ).sum(dim=0)
        support = (weight_view * sample_valid).sum(dim=0)
        directional = torch.exp(accumulated / support.clamp_min(0.15)) * valid[0, 0]
        directional_tensors.append(directional)
    stack = torch.stack(directional_tensors)
    if settings.noise_direction_integration == "mean":
        refined = stack.mean(dim=0)
    elif settings.noise_direction_integration == "minimum":
        refined = stack.min(dim=0).values
    elif settings.noise_direction_integration == "median":
        refined = stack.median(dim=0).values
    else:
        refined = stack.max(dim=0).values

    pooled_scale = torch.sqrt(bg_scale.square() + non_scale.square()).clamp_min(1e-4)
    # Report the symmetric two-class separation (distance from each centre to
    # their midpoint), matching the scale used by the original UI diagnostic.
    separation = float(
        2.0 * torch.linalg.vector_norm((bg_center - non_center) / pooled_scale).item()
    )
    profile = NoiseFrequencyProfile(
        band_scales_px=reported_scales,
        background_log_rms=tuple(float(value) for value in bg_center.cpu().tolist()),
        nonbackground_log_rms=tuple(float(value) for value in non_center.cpu().tolist()),
        background_sample_count=int(confident_background.sum().item()),
        nonbackground_sample_count=int(confident_nonbackground.sum().item()),
        separation=separation,
        background_log_scale=tuple(
            float(value) for value in bg_scale.cpu().tolist()
        ),
        nonbackground_log_scale=tuple(
            float(value) for value in non_scale.cpu().tolist()
        ),
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

    return (
        _lazy_u8(restore(refined) * 255.0, f"{target_name} noise likelihood"),
        profile,
        (),
        angles,
    )


def surrounding_band_noise_likelihood(
    source_tensor,
    valid_tensor,
    seed_diameter,
    settings,
    profile,
    *,
    cuda_context=None,
):
    """Apply the learned noise classifier only to the sampled rim annulus."""

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
    fine = gaussian_blur(lab, max(0.55, scales[0]))
    medium = gaussian_blur(lab, scales[1])
    coarse = gaussian_blur(lab, scales[2])
    residuals = (lab - fine, fine - medium, medium - coarse)
    features = []
    for residual, sigma in zip(residuals, scales, strict=True):
        energy = torch.mean(residual.square(), dim=1, keepdim=True)
        local = gaussian_blur(energy, max(0.65, min(2.5, sigma * 0.35)))
        features.append(torch.log1p(torch.sqrt(local.clamp_min(1e-10)) * 255.0))
    values = torch.cat(features, dim=1).permute(0, 2, 3, 1)
    bg_center = torch.as_tensor(
        profile.background_log_rms,
        device=context.device,
        dtype=values.dtype,
    )
    non_center = torch.as_tensor(
        profile.nonbackground_log_rms,
        device=context.device,
        dtype=values.dtype,
    )
    bg_scale = torch.as_tensor(
        profile.background_log_scale,
        device=context.device,
        dtype=values.dtype,
    ).clamp_min(1e-4)
    non_scale = torch.as_tensor(
        profile.nonbackground_log_scale,
        device=context.device,
        dtype=values.dtype,
    ).clamp_min(1e-4)
    bg_log = -0.5 * (
        ((values - bg_center) / bg_scale).square()
        + 2.0 * torch.log(bg_scale)
    ).sum(dim=-1)
    non_log = -0.5 * (
        ((values - non_center) / non_scale).square()
        + 2.0 * torch.log(non_scale)
    ).sum(dim=-1)
    probability = torch.sigmoid((bg_log - non_log) * 0.72)[0] * valid[0, 0]
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
):
    from seedvision.visualization.layers import AnalysisLayerSettings
    import torch

    settings = settings or AnalysisLayerSettings()
    context = cuda_context or CudaContext.resolve()
    source = (
        image_to_tensor(crop, context)
        if source_tensor is None
        else source_tensor
    )
    base_lab = bgr_to_lab(source) if lab_tensor is None else lab_tensor
    lab = gaussian_blur(base_lab, settings.edge_blur_sigma)
    valid = (
        image_to_tensor(valid_mask, context) > 0
        if valid_tensor is None
        else valid_tensor.bool()
    )
    gradients = [gradient_magnitude(lab[:, channel : channel + 1], scharr=True) for channel in range(3)]
    weights = (1.0, settings.edge_chroma_weight, settings.edge_chroma_weight)
    gx = sum(item[1] * weight for item, weight in zip(gradients, weights, strict=True))
    gy = sum(item[2] * weight for item, weight in zip(gradients, weights, strict=True))
    magnitude = torch.sqrt(gx.square() + gy.square()).clamp_min(0.0)
    normalization = torch.quantile(magnitude[valid], settings.edge_normalization_percentile / 100.0).clamp_min(1e-5)
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
    if sample_count > 32768:
        step = max(1, sample_count // 32768)
        locations = locations[::step][:32768]
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
    for _ in range(max(1, int(iterations))):
        distances = torch.cdist(standardized, normalized_centres).square()
        assignments = torch.argmin(distances, dim=1)
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
    distances = torch.cdist(standardized, normalized_centres).square()
    assignments = torch.argmin(distances, dim=1)

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
            support / support.max().clamp_min(1e-6)
        )
        similarity *= support[None, :, None, None]
        best = torch.maximum(best, similarity.max(dim=1, keepdim=True).values)
    return best


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
    matrix = cv2.getRotationMatrix2D(centre, -float(tangent_degrees), 1.0)
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


def reference_texture_probabilities(
    crop,
    gradients: EdgeGradientProducts,
    ridges,
    frequency_noise: FrequencyNoiseMaskProducts,
    seed_diameter,
    settings,
    *,
    background_reference_mask=None,
    foreground_reference_mask=None,
    other_reference_mask=None,
    physical_reference_mask=None,
    non_edge_reference_mask=None,
    seed_instance_annotations=None,
    cuda_context=None,
) -> ReferenceTextureProducts:
    """Learn many material and edge prototypes from painted references.

    Every prototype represents a medoid patch in a robust feature cluster. The
    banks are evaluated globally; painted coordinates are never overwritten.
    """

    import torch
    import torch.nn.functional as functional

    from seedvision.visualization.layers import (
        ReferenceTextureProfile,
        ReferenceTexturePrototype,
    )

    context = cuda_context or CudaContext.resolve()
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
    edge_feature_names = material_feature_names + (
        "axial tangent coherence",
        "directed tangent coherence",
        "cross-normal lightness contrast",
        "cross-normal colour contrast",
    )

    def source_count(mask) -> int:
        return 0 if mask is None else int(np.count_nonzero(mask))

    if foreground_reference_mask is None:
        foreground_source_count = source_count(seed_instance_annotations)
    elif seed_instance_annotations is None:
        foreground_source_count = source_count(foreground_reference_mask)
    else:
        foreground_source_count = int(
            np.count_nonzero(
                np.logical_or(
                    np.asarray(foreground_reference_mask, dtype=bool),
                    np.asarray(seed_instance_annotations) > 0,
                )
            )
        )

    source_counts = (
        ("background", source_count(background_reference_mask)),
        ("foreground", foreground_source_count),
        ("other", source_count(other_reference_mask)),
        ("physical_edge", source_count(physical_reference_mask)),
        ("non_edge", source_count(non_edge_reference_mask)),
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
    edge = resized(gradients.strength).clamp(0.0, 1.0)
    ridge = resized(
        _raster_tensor(ridges, context, normalized=True)
    ).clamp(0.0, 1.0)
    if not any(count for _class_name, count in source_counts):
        ridge_weight = float(settings.reference_edge_ridge_weight)
        edge_support = (
            (1.0 - ridge_weight) * edge
            + ridge_weight * ridge
        ).clamp(0.0, 1.0) * valid
        physical_field = _lazy_float(
            edge_support,
            "continuous generic physical edge probability",
        )
        if edge_support.shape[-2:] != (source_height, source_width):
            edge_support = functional.interpolate(
                edge_support,
                (source_height, source_width),
                mode="bilinear",
                align_corners=False,
            )
        physical = _lazy_u8(
            edge_support * 255.0,
            "generic physical edge probability",
        )
        zero = _lazy_u8(
            edge_support * 0.0,
            "empty reference non-edge probability",
        )
        return ReferenceTextureProducts(
            seed_surface_probability=None,
            background_probability=None,
            other_probability=None,
            physical_edge_probability=physical,
            non_edge_probability=zero,
            physical_edge_field=physical_field,
            profile=ReferenceTextureProfile(
                class_sample_counts=source_counts,
                material_feature_names=material_feature_names,
                edge_feature_names=edge_feature_names,
                working_scale=work_scale,
                patch_size_px=patch_size,
            ),
            foreground_sample_count=0,
            background_sample_count=0,
            other_sample_count=0,
            physical_sample_count=0,
            non_edge_sample_count=0,
        )

    lab = resized(gradients.lab)
    tangent_x = resized(gradients.tangent_x)
    tangent_y = resized(gradients.tangent_y)
    tangent_length = torch.sqrt(
        tangent_x.square() + tangent_y.square()
    ).clamp_min(1e-6)
    tangent_x /= tangent_length
    tangent_y /= tangent_length
    normal_x = -tangent_y
    normal_y = tangent_x
    diameter = max(6.0, float(seed_diameter) * work_scale)
    sigma = max(
        0.7,
        diameter * float(settings.reference_texture_context_fraction),
    )
    local_lab = gaussian_blur(lab, sigma)
    lab_residual = lab - local_lab
    colour_residual = torch.sqrt(
        lab_residual[:, 1:2].square() + lab_residual[:, 2:3].square()
    )
    local_edge = gaussian_blur(edge, sigma).clamp(0.0, 1.0)
    noise_channels = tuple(
        resized(_raster_tensor(raster, context, normalized=True))
        for raster in (
            *frequency_noise.darkness_masks,
            *frequency_noise.colour_masks,
        )
    )
    material_features = torch.cat(
        (
            lab[:, 0:1] / 255.0,
            (lab[:, 1:2] - 128.0) / 128.0,
            (lab[:, 2:3] - 128.0) / 128.0,
            *noise_channels,
            edge,
            ridge,
            lab_residual[:, 0:1].abs() / 40.0,
            colour_residual / 55.0,
            local_edge,
        ),
        dim=1,
    )

    axial_x = tangent_x.square() - tangent_y.square()
    axial_y = 2.0 * tangent_x * tangent_y
    axial_coherence = torch.sqrt(
        gaussian_blur(axial_x, sigma).square()
        + gaussian_blur(axial_y, sigma).square()
    ).clamp(0.0, 1.0)
    directed_coherence = torch.sqrt(
        gaussian_blur(tangent_x, sigma).square()
        + gaussian_blur(tangent_y, sigma).square()
    ).clamp(0.0, 1.0)
    yy, xx = torch.meshgrid(
        torch.arange(height, device=context.device, dtype=torch.float32),
        torch.arange(width, device=context.device, dtype=torch.float32),
        indexing="ij",
    )
    cross_distance = max(1.0, diameter * 0.025)
    plus_x = xx + normal_x[0, 0] * cross_distance
    plus_y = yy + normal_y[0, 0] * cross_distance
    minus_x = xx - normal_x[0, 0] * cross_distance
    minus_y = yy - normal_y[0, 0] * cross_distance
    cross_lab = []
    for channel in range(3):
        positive = bilinear_sample(lab[:, channel : channel + 1], plus_x, plus_y)
        negative = bilinear_sample(lab[:, channel : channel + 1], minus_x, minus_y)
        cross_lab.append((positive - negative).abs()[None, None])
    cross_chroma = torch.sqrt(
        cross_lab[1].square() + cross_lab[2].square()
    )
    edge_features = torch.cat(
        (
            material_features,
            axial_coherence,
            directed_coherence,
            cross_lab[0] / 55.0,
            cross_chroma / 70.0,
        ),
        dim=1,
    )

    def reference_mask(values):
        if values is None:
            return torch.zeros_like(valid)
        source = image_to_tensor(np.asarray(values, dtype=np.uint8), context)
        return resized(source, mode="area") > 0.001

    background_mask = reference_mask(background_reference_mask) & valid
    foreground_mask = reference_mask(foreground_reference_mask) & valid
    other_mask = reference_mask(other_reference_mask) & valid
    physical_mask = reference_mask(physical_reference_mask) & valid
    non_edge_mask = reference_mask(non_edge_reference_mask) & valid
    if seed_instance_annotations is not None and np.any(seed_instance_annotations):
        annotation_mask = reference_mask(
            np.asarray(seed_instance_annotations) > 0
        ) & valid
        erosion = max(1, round(diameter * 0.06))
        kernel = erosion * 2 + 1
        eroded = 1.0 - functional.max_pool2d(
            1.0 - annotation_mask.float(),
            kernel,
            stride=1,
            padding=erosion,
        )
        if bool((eroded > 0.999).any().item()):
            foreground_mask |= eroded > 0.999
        else:
            foreground_mask |= annotation_mask
    material_overlap = (
        background_mask.to(torch.uint8)
        + foreground_mask.to(torch.uint8)
        + other_mask.to(torch.uint8)
    ) > 1
    background_mask &= ~material_overlap
    foreground_mask &= ~material_overlap
    other_mask &= ~material_overlap
    edge_overlap = physical_mask & non_edge_mask
    physical_mask &= ~edge_overlap
    non_edge_mask &= ~edge_overlap

    fit_arguments = dict(
        maximum_prototypes=int(settings.reference_texture_prototypes_per_class),
        minimum_support=int(
            settings.reference_texture_minimum_samples_per_prototype
        ),
        iterations=int(settings.reference_texture_fit_iterations),
        scale_floor=0.045,
    )
    banks = {
        "background": _fit_feature_prototype_bank(
            material_features,
            background_mask,
            class_name="background",
            **fit_arguments,
        ),
        "foreground": _fit_feature_prototype_bank(
            material_features,
            foreground_mask,
            class_name="foreground",
            **fit_arguments,
        ),
        "other": _fit_feature_prototype_bank(
            material_features,
            other_mask,
            class_name="other",
            **fit_arguments,
        ),
    }
    edge_fit_arguments = dict(fit_arguments)
    edge_fit_arguments["scale_floor"] = 0.055
    edge_banks = {
        "physical_edge": _fit_feature_prototype_bank(
            edge_features,
            physical_mask,
            class_name="physical_edge",
            **edge_fit_arguments,
        ),
        "non_edge": _fit_feature_prototype_bank(
            edge_features,
            non_edge_mask,
            class_name="non_edge",
            **edge_fit_arguments,
        ),
    }
    tolerance = float(settings.reference_texture_similarity_scale)
    material_scores = {
        name: (
            None
            if bank is None
            else _prototype_bank_similarity(material_features, bank, tolerance)
        )
        for name, bank in banks.items()
    }

    def class_probability(name, competitors):
        score = material_scores[name]
        if score is None:
            return None
        available = [
            material_scores[competitor]
            for competitor in competitors
            if material_scores[competitor] is not None
        ]
        if not available:
            return score * valid
        competing = torch.stack(available, dim=0).max(dim=0).values
        return apply_contrastive_negative_evidence(
            score, competing, strength=0.95
        ) * valid

    foreground_probability = class_probability(
        "foreground", ("background", "other")
    )
    background_probability = class_probability(
        "background", ("foreground", "other")
    )
    other_probability = class_probability(
        "other", ("foreground", "background")
    )
    physical_similarity = (
        None
        if edge_banks["physical_edge"] is None
        else _prototype_bank_similarity(
            edge_features, edge_banks["physical_edge"], tolerance
        )
    )
    non_edge_similarity = (
        None
        if edge_banks["non_edge"] is None
        else _prototype_bank_similarity(
            edge_features, edge_banks["non_edge"], tolerance
        )
    )
    ridge_weight = float(settings.reference_edge_ridge_weight)
    edge_support = (
        (1.0 - ridge_weight) * edge
        + ridge_weight * ridge
    ).clamp(0.0, 1.0)
    if physical_similarity is not None and non_edge_similarity is not None:
        normalizer = physical_similarity + non_edge_similarity + 0.10
        physical_probability = edge_support * physical_similarity / normalizer
        non_edge_probability = edge_support * non_edge_similarity / normalizer
    elif physical_similarity is not None:
        physical_probability = edge_support * (
            0.25 + 0.75 * physical_similarity
        )
        non_edge_probability = edge_support * (1.0 - physical_similarity) * 0.35
    elif non_edge_similarity is not None:
        non_edge_probability = edge_support * non_edge_similarity
        physical_probability = edge_support * (
            1.0 - 0.85 * non_edge_similarity
        )
    else:
        physical_probability = edge_support
        non_edge_probability = torch.zeros_like(edge_support)
    physical_probability *= valid
    non_edge_probability *= valid
    physical_edge_field = _lazy_float(
        physical_probability,
        "continuous multi-prototype physical edge probability",
    )

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

    prototypes = []
    all_banks = {**banks, **edge_banks}
    tangent_degrees = torch.rad2deg(
        torch.atan2(tangent_y[0, 0], tangent_x[0, 0])
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
        positions = bank_positions.detach().cpu().numpy()
        weights = bank.weights.detach().cpu().numpy()
        counts = bank.sample_counts.detach().cpu().numpy()
        angles = (
            tangent_degrees[
                bank_positions[:, 0], bank_positions[:, 1]
            ].detach().cpu().numpy()
            if class_name in {"physical_edge", "non_edge"}
            else np.full(len(positions), np.nan, np.float32)
        )
        for position, weight, count, raw_angle in zip(
            positions, weights, counts, angles, strict=True
        ):
            y_work, x_work = int(position[0]), int(position[1])
            centre_xy = (
                float(x_work) / max(work_scale, 1e-8),
                float(y_work) / max(work_scale, 1e-8),
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
        material_feature_names=material_feature_names,
        edge_feature_names=edge_feature_names,
        working_scale=work_scale,
        patch_size_px=patch_size,
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
            "multi-prototype physical edge probability",
        ),
        non_edge_probability=restored(
            non_edge_probability,
            "multi-prototype non-edge probability",
        ),
        physical_edge_field=physical_edge_field,
        profile=profile,
        foreground_sample_count=dict(source_counts)["foreground"],
        background_sample_count=dict(source_counts)["background"],
        other_sample_count=dict(source_counts)["other"],
        physical_sample_count=dict(source_counts)["physical_edge"],
        non_edge_sample_count=dict(source_counts)["non_edge"],
    )


def reference_edge_probabilities(
    gradients: EdgeGradientProducts,
    ridges,
    seed_diameter,
    settings,
    *,
    source_tensor=None,
    physical_reference_mask=None,
    non_edge_reference_mask=None,
    cuda_context=None,
) -> ReferenceEdgeProducts:
    """Fit sparse reviewed edge classes and evaluate them across the dish.

    The classifier is intentionally compact and image-local. It learns robust
    diagonal feature profiles from reviewed pixels, but never replaces output
    values at the painted coordinates. Rotation-invariant directed and axial
    tangent coherence make the reference useful around an entire curved seed.
    """

    import torch
    import torch.nn.functional as functional

    context = cuda_context or CudaContext.resolve()
    source_height, source_width = gradients.strength.shape[-2:]
    maximum = int(settings.reference_edge_working_maximum_dimension)
    scale = min(1.0, maximum / max(source_height, source_width))
    height = max(16, round(source_height * scale))
    width = max(16, round(source_width * scale))

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
    edge = resized(gradients.strength).clamp(0.0, 1.0)
    ridge = resized(
        _raster_tensor(ridges, context, normalized=True)
    ).clamp(0.0, 1.0)
    corrected_lab = (
        gradients.lab if source_tensor is None else bgr_to_lab(source_tensor)
    )
    lab = resized(corrected_lab)
    tangent_x = resized(gradients.tangent_x)
    tangent_y = resized(gradients.tangent_y)
    tangent_length = torch.sqrt(tangent_x.square() + tangent_y.square()).clamp_min(1e-6)
    tangent_x = tangent_x / tangent_length
    tangent_y = tangent_y / tangent_length
    sigma = max(
        0.7,
        float(seed_diameter)
        * scale
        * float(settings.reference_edge_context_fraction),
    )
    axial_x = tangent_x.square() - tangent_y.square()
    axial_y = 2.0 * tangent_x * tangent_y
    axial_coherence = torch.sqrt(
        gaussian_blur(axial_x, sigma).square()
        + gaussian_blur(axial_y, sigma).square()
    ).clamp(0.0, 1.0)
    directed_coherence = torch.sqrt(
        gaussian_blur(tangent_x, sigma).square()
        + gaussian_blur(tangent_y, sigma).square()
    ).clamp(0.0, 1.0)
    local_lab = gaussian_blur(lab, sigma)
    lab_residual = lab - local_lab
    colour_residual = torch.sqrt(
        lab_residual[:, 1:2].square() + lab_residual[:, 2:3].square()
    )
    features = torch.cat(
        (
            edge,
            ridge,
            axial_coherence,
            directed_coherence,
            lab[:, 0:1] / 255.0,
            (lab[:, 1:2] - 128.0) / 128.0,
            (lab[:, 2:3] - 128.0) / 128.0,
            lab_residual[:, 0:1].abs() / 32.0,
            colour_residual / 45.0,
        ),
        dim=1,
    )

    def reference_mask(values):
        if values is None:
            return torch.zeros_like(valid)
        source = image_to_tensor(np.asarray(values, dtype=np.uint8), context)
        # Area interpolation preserves thin painted strokes when the bounded
        # classifier works below full resolution.
        return resized(source, mode="area") > 0.001

    physical_mask = reference_mask(physical_reference_mask) & valid
    non_edge_mask = reference_mask(non_edge_reference_mask) & valid & ~physical_mask
    physical_count = int(physical_mask.sum().item())
    non_edge_count = int(non_edge_mask.sum().item())
    scale_floor = torch.tensor(
        (0.08, 0.08, 0.08, 0.08, 0.10, 0.10, 0.10, 0.10, 0.10),
        device=context.device,
        dtype=torch.float32,
    ) * float(settings.reference_edge_similarity_scale)

    def similarity(mask):
        samples = features[0, :, mask[0, 0]].T
        if int(samples.shape[0]) > 65536:
            samples = samples[:: max(1, int(samples.shape[0]) // 65536)][:65536]
        centre = torch.median(samples, dim=0).values
        spread = 1.4826 * torch.median(torch.abs(samples - centre), dim=0).values
        spread = torch.maximum(spread, scale_floor)
        standardized = (features - centre[None, :, None, None]) / spread[
            None, :, None, None
        ]
        distance = standardized.square().clamp_max(9.0).mean(dim=1, keepdim=True)
        return torch.exp(-0.5 * distance)

    ridge_weight = float(settings.reference_edge_ridge_weight)
    edge_support = (
        (1.0 - ridge_weight) * edge + ridge_weight * ridge
    ).clamp(0.0, 1.0)
    if physical_count and non_edge_count:
        physical_similarity = similarity(physical_mask)
        non_edge_similarity = similarity(non_edge_mask)
        normalizer = physical_similarity + non_edge_similarity + 0.10
        physical = edge_support * physical_similarity / normalizer
        non_edge = edge_support * non_edge_similarity / normalizer
    elif physical_count:
        physical_similarity = similarity(physical_mask)
        physical = edge_support * (0.25 + 0.75 * physical_similarity)
        non_edge = edge_support * (1.0 - physical_similarity) * 0.35
    elif non_edge_count:
        non_edge_similarity = similarity(non_edge_mask)
        non_edge = edge_support * non_edge_similarity
        physical = edge_support * (1.0 - 0.85 * non_edge_similarity)
    else:
        physical = edge_support
        non_edge = torch.zeros_like(edge_support)
    physical = physical * valid
    non_edge = non_edge * valid

    def restored(values, name):
        if values.shape[-2:] != (source_height, source_width):
            values = functional.interpolate(
                values,
                (source_height, source_width),
                mode="bilinear",
                align_corners=False,
            )
        return _lazy_u8(values * 255.0, name)

    return ReferenceEdgeProducts(
        physical_probability=restored(
            physical, "reference-trained physical edge probability"
        ),
        non_edge_probability=restored(
            non_edge, "reference-trained non-edge probability"
        ),
        physical_field=_lazy_float(
            physical,
            "continuous reference-trained physical edge probability",
        ),
        physical_sample_count=physical_count,
        non_edge_sample_count=non_edge_count,
    )


def reference_probability_ridges(
    physical_probability_field,
    gradients: EdgeGradientProducts,
    settings,
    *,
    cuda_context=None,
) -> GpuRaster:
    """Thin reference-trained physical-edge probability without a CPU round trip."""

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

    probability = resized(
        _raster_tensor(physical_probability_field, context)
    ).clamp(0.0, 1.0)
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

    diameter = max(
        4.0,
        float(seed_diameter) * float(scale) * settings.curve_diameter_multiplier,
    )
    trace_signature = (
        ridge_signature,
        float(diameter),
        float(settings.trace_tangent_tolerance_degrees),
        int(settings.trace_maximum_gap_px),
        str(settings.trace_curvature_policy),
        float(settings.trace_curvature_tolerance_degrees),
        float(settings.trace_window_fraction),
        int(settings.trace_sample_count),
        float(settings.trace_minimum_length_fraction),
        int(settings.trace_junction_max_neighbors),
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
            accepted.float()[None, None],
            torch.ones((1, 1, 3, 3), device=context.device),
            padding=1,
        )[0, 0] - accepted.float()
        trace_seed = accepted & (
            neighbour_count <= settings.trace_junction_max_neighbors
        )
        trace_labels = oriented_connected_components(
            trace_seed[None, None],
            tangent_x,
            tangent_y,
            maximum_gap=settings.trace_maximum_gap_px,
            tangent_tolerance_degrees=settings.trace_tangent_tolerance_degrees,
            curvature_policy=settings.trace_curvature_policy,
            curvature_tolerance_degrees=(
                settings.trace_curvature_tolerance_degrees
            ),
        ).long()
        component_area = torch.bincount(trace_labels.reshape(-1))
        minimum_trace_length = max(
            2, round(diameter * settings.trace_minimum_length_fraction)
        )
        retained_component = component_area >= minimum_trace_length
        retained_component[0] = False
        trace_seed &= retained_component[trace_labels[0, 0]]
        trace_labels = torch.where(
            trace_seed[None, None], trace_labels, torch.zeros_like(trace_labels)
        )

        # Tangent-following path integration measures continuity and allows gaps.
        window = max(2.0, diameter * settings.trace_window_fraction)
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
            np.cos(np.deg2rad(settings.trace_tangent_tolerance_degrees))
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
                alignment = (
                    (alignment - tolerance_cos)
                    / max(1e-5, 1.0 - tolerance_cos)
                ).clamp(0.0, 1.0)
                support_samples.append(ridge_sample * alignment)
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
    physical_reference = (
        edge
        if physical_edge_probability is None
        else resized(
            _raster_tensor(physical_edge_probability, context, normalized=True)
        )[0, 0]
    )
    non_edge_reference = (
        torch.zeros_like(edge)
        if non_edge_probability is None
        else resized(
            _raster_tensor(non_edge_probability, context, normalized=True)
        )[0, 0]
    )
    ridge_support = functional.max_pool2d(
        (nms * trace_seed)[None, None], 3, stride=1, padding=1
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

    for radius in radii_tested:
        radius_prior = torch.exp(
            -0.5
            * (
                torch.log(radius / max(diameter * 0.5, 1e-4))
                / settings.boundary_radius_log_tolerance
            ).square()
        )
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
            rounded_x = torch.round(centre_x).long().clamp(0, width - 1)
            rounded_y = torch.round(centre_y).long().clamp(0, height - 1)
            flat_index = (rounded_y * width + rounded_x).reshape(-1)
            vote_accumulator.reshape(-1).scatter_add_(
                0, flat_index, circle.reshape(-1)
            )

    vote_blur = max(0.7, diameter * settings.boundary_center_vote_blur_fraction)
    vote_accumulator = gaussian_blur(
        vote_accumulator[None, None], vote_blur
    )[0, 0]
    vote_scale = torch.quantile(
        vote_accumulator.reshape(-1), 0.995
    ).clamp_min(1e-6)
    vote_probability = (vote_accumulator / vote_scale).clamp(0.0, 1.0)
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
        - settings.boundary_reference_influence
        + settings.boundary_reference_influence * physical_reference
    ) * (
        1.0
        - settings.boundary_reference_nonedge_discount * non_edge_reference
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

    local_maxima = vote_probability == functional.max_pool2d(
        vote_probability[None, None],
        max(3, round(diameter * 0.25)) | 1,
        stride=1,
        padding=(max(3, round(diameter * 0.25)) | 1) // 2,
    )[0, 0]
    candidate_votes = torch.where(
        local_maxima & (vote_probability >= 0.20),
        vote_probability,
        torch.zeros_like(vote_probability),
    )
    top_count = min(
        int(settings.boundary_geometry_max_candidates),
        int(candidate_votes.numel()),
    )
    vote_values, vote_indices = torch.topk(candidate_votes.reshape(-1), top_count)
    vote_centres = torch.stack(
        ((vote_indices % width).float() / scale, (vote_indices // width).float() / scale),
        dim=1,
    )
    geometry = GpuBoundaryGeometry(
        vote_centres_xy=vote_centres,
        vote_confidence=vote_values,
        instance_centres_xy=center_tensor / max(scale, 1e-6),
        ellipse_axes_xy=ellipse_axes / max(scale, 1e-6),
        ellipse_angle_radians=ellipse_angles,
        ellipse_confidence=ellipse_fit_values,
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
        noise_background_min_likelihood=settings.foreground_noise_foreground_min_likelihood,
        noise_nonbackground_max_likelihood=settings.foreground_noise_nonforeground_max_likelihood,
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
    foreground_exclusion_mask=None,
    reference_radius=3,
    cuda_context=None,
    source_tensor=None,
    lab_tensor=None,
    valid_tensor=None,
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
        foreground_reference_mask=background_reference_mask,
        target_exclusion_mask=foreground_exclusion_mask,
        target_name="foreground",
        reference_radius=reference_radius,
        cuda_context=cuda_context,
        source_tensor=source_tensor,
        lab_tensor=lab_tensor,
        valid_tensor=valid_tensor,
        target_reference_precedence=True,
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
