"""PyTorch implementations of Seed Vision's legacy diagnostic layers."""

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
    background_reference_samples: np.ndarray | None = None,
    background_reference_sample_count: int = 0,
    background_prior_lab: tuple[float, float, float] | None = None,
    background_colour_enabled: bool = True,
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

    gpu_inputs = values.get("layer.gpu_inputs")
    if gpu_inputs is None:
        source_tensor = image_to_tensor(crop, context)
        lab_tensor = bgr_to_lab(source_tensor)
        valid_tensor = image_to_tensor(valid_mask, context) > 0
        gpu_inputs = source_tensor, lab_tensor, valid_tensor
        values["layer.gpu_inputs"] = gpu_inputs
    else:
        source_tensor, lab_tensor, valid_tensor = gpu_inputs
    background_dirty = "background_likelihood" in dirty or "layer.background" not in values
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
            background_reference_samples=background_reference_samples,
            background_reference_sample_count=background_reference_sample_count,
            background_prior_lab=background_prior_lab,
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
    if instance_dirty:
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
        labels = instance_voronoi(
            valid_mask,
            agreed_background,
            centers,
            radii,
            seed_diameter,
            settings,
            cuda_context=context,
            valid_tensor=valid_tensor,
        )
        colours = spatially_contrasting_colours(centers)
        values["layer.instances"] = labels, colours
        if instance_timing is not None:
            timing_recorder.stop(instance_timing)
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
    curve_dirty = (
        traces_dirty
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
            instance_labels=labels,
            centers=centers,
            radii=radii,
            cuda_context=context,
            previous=previous_curve_result,
            recompute_ridges=ridges_dirty,
            recompute_traces=traces_dirty,
            timing_recorder=timing_recorder,
        )
        values["layer.seed_edge_curves"] = curve_result
        values["layer.edge_ridges"] = curve_result.ridges
        values["layer.edge_traces"] = (
            curve_result.trace_labels,
            curve_result.trace_continuity,
            curve_result.gap_confidence,
        )
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
        edge_likelihood=edge_likelihood,
        directed_edge_hue=directed_edge_hue,
        undirected_edge_hue=undirected_edge_hue,
        seed_edge_curve_likelihood=curve_likelihood,
        seed_edge_curve_radius_px=curve_radius,
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
    background_reference_samples=None,
    background_reference_sample_count=0,
    background_prior_lab=None,
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
    mode = "automatic"
    reference_count = 0
    supplied = None
    source_values = None
    eligible = valid & ~foreground_point_mask
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
                int(np.count_nonzero(background_reference_mask))
                if background_reference_mask is not None
                else len(background_reference_points)
            )

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
        refinement_iterations=settings.background_refinement_iterations,
        refinement_min_probability=settings.background_refinement_min_probability,
        frequency_weight_power=settings.background_frequency_weight_power,
        scale_multiplier=settings.background_distribution_scale_multiplier,
        scale_floors=(
            settings.background_lightness_scale_floor,
            settings.background_chroma_scale_floor,
            settings.background_chroma_scale_floor,
        ),
    )
    dominant_component = int(torch.argmax(component_weights).item())
    centre = component_centres[dominant_component]
    scale = component_scales[dominant_component]
    # Reference regions are hard semantic constraints, not merely training
    # samples. Foreground wins if the user accidentally overlaps both classes.
    likelihood = torch.where(background_point_mask, torch.ones_like(likelihood), likelihood)
    likelihood = torch.where(foreground_point_mask, torch.zeros_like(likelihood), likelihood)

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
    )
    return _lazy_u8(
        likelihood[None, None] * 255.0, "background colour likelihood"
    ), mode, reference_count, profile


def empty_noise_frequency_profile(seed_diameter, settings):
    from seedvision.visualization.layers import NoiseFrequencyProfile

    scales = _noise_scales(seed_diameter, settings)
    return NoiseFrequencyProfile(scales, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0, 0, 0.0)


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
    reference_radius=3,
    cuda_context=None,
    source_tensor=None,
    lab_tensor=None,
    valid_tensor=None,
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
    full_hard_background = _point_mask(
        source_height,
        source_width,
        background_reference_points,
        reference_radius,
        context,
    )
    full_hard_foreground = _point_mask(
        source_height,
        source_width,
        foreground_reference_points,
        reference_radius,
        context,
    )
    if background_reference_mask is not None:
        full_hard_background |= image_to_tensor(
            np.asarray(background_reference_mask, np.uint8), context
        )[0, 0] > 0
    if foreground_reference_mask is not None:
        full_hard_foreground |= image_to_tensor(
            np.asarray(foreground_reference_mask, np.uint8), context
        )[0, 0] > 0
    full_hard_background &= full_valid[0, 0]
    full_hard_foreground &= full_valid[0, 0]

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
        hard_background = full_hard_background
        hard_foreground = full_hard_foreground
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
        hard_background = functional.interpolate(
            full_hard_background[None, None].float(),
            (height, width),
            mode="nearest",
        )[0, 0] > 0.5
        hard_foreground = functional.interpolate(
            full_hard_foreground[None, None].float(),
            (height, width),
            mode="nearest",
        )[0, 0] > 0.5
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
    confident_background = valid & (colour >= settings.noise_background_min_likelihood / 255.0)
    confident_nonbackground = valid & (colour <= settings.noise_nonbackground_max_likelihood / 255.0)
    minimum_samples = max(32, round(int(valid.sum().item()) * 0.002))
    if int(confident_background.sum().item()) < minimum_samples:
        threshold = torch.quantile(colour[valid], 0.75)
        confident_background = valid & (colour >= threshold)
    if int(confident_nonbackground.sum().item()) < minimum_samples:
        threshold = torch.quantile(colour[valid], 0.25)
        confident_nonbackground = valid & (colour <= threshold)

    bg_values = feature.permute(0, 2, 3, 1)[confident_background.permute(0, 2, 3, 1).expand(-1, -1, -1, 3)].reshape(-1, 3)
    non_values = feature.permute(0, 2, 3, 1)[confident_nonbackground.permute(0, 2, 3, 1).expand(-1, -1, -1, 3)].reshape(-1, 3)
    bg_center, bg_scale = _robust_tensor_distribution(bg_values)
    non_center, non_scale = _robust_tensor_distribution(non_values)
    values = feature.permute(0, 2, 3, 1)
    bg_log = -0.5 * (((values - bg_center) / bg_scale).square() + 2.0 * torch.log(bg_scale)).sum(dim=-1, keepdim=True)
    non_log = -0.5 * (((values - non_center) / non_scale).square() + 2.0 * torch.log(non_scale)).sum(dim=-1, keepdim=True)
    texture_probability = torch.sigmoid((bg_log - non_log) * 0.72).permute(0, 3, 1, 2)
    base = (0.72 * texture_probability + 0.28 * colour).clamp(1e-4, 1.0) * valid

    base = torch.where(hard_background[None, None], torch.ones_like(base), base)
    base = torch.where(hard_foreground[None, None], torch.zeros_like(base), base)
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
        directional = torch.where(hard_background, torch.ones_like(directional), directional)
        directional = torch.where(hard_foreground, torch.zeros_like(directional), directional)
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
    refined = torch.where(hard_background, torch.ones_like(refined), refined)
    refined = torch.where(hard_foreground, torch.zeros_like(refined), refined)

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
        values = values * full_valid.float()
        values = torch.where(
            full_hard_background[None, None], torch.ones_like(values), values
        )
        return torch.where(
            full_hard_foreground[None, None], torch.zeros_like(values), values
        )

    directional = tuple(
        _lazy_u8(
            restore(item) * 255.0,
            f"background ray {angle:g} degrees",
        )
        for item, angle in zip(directional_tensors, angles, strict=True)
    )
    return (
        _lazy_u8(restore(refined) * 255.0, "refined background likelihood"),
        profile,
        directional,
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


def seed_boundary_tracing(
    gradients: EdgeGradientProducts,
    seed_diameter,
    settings,
    *,
    background_likelihood=None,
    foreground_probability=None,
    instance_labels=None,
    centers=(),
    radii=(),
    cuda_context=None,
    previous: BoundaryTraceProducts | None = None,
    recompute_ridges: bool = True,
    recompute_traces: bool = True,
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
