"""Hierarchical Seed-versus-Non-seed material-evidence fusion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from seedvision.cuda.ops import (
    CudaContext,
    GpuRaster,
    binary_close,
    binary_open,
    image_to_tensor,
)


@dataclass(slots=True)
class MaterialEvidenceProducts:
    """GPU-resident supports and normalized decision masses.

    Raw evidence producers remain independent.  The four decision rasters are
    mutually exclusive and sum to one over valid pixels.
    """

    seed_support: GpuRaster
    background_support: GpuRaster
    other_support: GpuRaster
    nonseed_support: GpuRaster
    seed_probability: GpuRaster
    nonseed_probability: GpuRaster
    ambiguity_probability: GpuRaster
    unknown_probability: GpuRaster
    conditional_background_probability: GpuRaster
    conditional_other_probability: GpuRaster
    conditional_ambiguity_probability: GpuRaster
    conditional_unknown_probability: GpuRaster
    seed_mask: GpuRaster
    source_reliabilities: tuple[tuple[str, float], ...] = ()


def _probability_tensor(values, context: CudaContext):
    """Return one NCHW float probability tensor, or ``None``."""

    if values is None:
        return None
    import torch

    if hasattr(values, "gpu_tensor"):
        tensor = values.gpu_tensor(device=context.device)
    elif torch.is_tensor(values):
        tensor = values.to(device=context.device)
    else:
        tensor = image_to_tensor(np.asarray(values), context)
    if tensor.ndim == 2:
        tensor = tensor[None, None]
    elif tensor.ndim == 3:
        tensor = tensor[None]
    if tensor.ndim != 4 or tensor.shape[0] != 1 or tensor.shape[1] != 1:
        raise ValueError("Material evidence must be a single-channel raster.")
    if tensor.dtype == torch.bool:
        return tensor.float()
    tensor = tensor.float()
    if not tensor.is_floating_point() or float(tensor.detach().max().item()) > 1.0:
        tensor = tensor / 255.0
    return tensor.clamp(0.0, 1.0)


def _weighted_support(sources, weights, authorities=None):
    """Fuse available support while unreliable sources leave unknown mass."""

    if authorities is None:
        authorities = (1.0,) * len(sources)
    available = [
        (source, float(weight), min(1.0, max(0.0, float(authority))))
        for source, weight, authority in zip(
            sources, weights, authorities, strict=True
        )
        if source is not None and float(weight) > 0.0
    ]
    if not available:
        return None
    denominator = sum(weight for _source, weight, _authority in available)
    return sum(
        source * weight * authority
        for source, weight, authority in available
    ) / denominator


def _reference_mask(values, context: CudaContext, shape):
    tensor = _probability_tensor(values, context)
    if tensor is None:
        return None
    if tensor.shape != shape:
        raise ValueError("Material calibration masks must share evidence dimensions.")
    return tensor > 0.0


def _source_reliability(source, target, nontarget, valid) -> float:
    """Estimate held-in discrimination without changing the raw evidence map."""

    import torch

    if source is None or target is None or nontarget is None:
        return 1.0
    target_pixels = target & valid
    nontarget_pixels = nontarget & valid
    if int(target_pixels.sum().item()) < 16 or int(nontarget_pixels.sum().item()) < 16:
        return 1.0
    target_median = source[target_pixels].median()
    nontarget_high = torch.quantile(source[nontarget_pixels], 0.90)
    # A source must beat the high tail of its reviewed non-target response,
    # not merely the median. This specifically prevents a broadly fitted
    # Background/Other model from earning suppressive authority while it still
    # cross-matches a substantial minority of annotated seed interiors. A 0.20
    # margin receives full authority; equal or reversed values receive none.
    # Calibration changes influence only; the raw map remains visible.
    return float(
        ((target_median - nontarget_high) / 0.20)
        .clamp(0.0, 1.0)
        .item()
    )


def hierarchical_material_evidence(
    valid_mask,
    seed_diameter: float,
    *,
    foreground_colour,
    foreground_noise=None,
    background_colour=None,
    background_noise=None,
    other_colour=None,
    other_noise=None,
    reference_foreground=None,
    reference_background=None,
    reference_other=None,
    seed_reference_mask=None,
    additional_seed_reference_mask=None,
    background_reference_mask=None,
    other_reference_mask=None,
    calibration_valid_mask=None,
    proposal_valid_mask=None,
    colour_weight: float = 1.0,
    noise_weight: float = 0.55,
    prototype_weight: float = 0.70,
    unknown_weight: float = 0.35,
    temperature: float = 1.0,
    seed_threshold: float = 0.50,
    morphology_fraction: float = 0.06,
    cuda_context: CudaContext | None = None,
) -> MaterialEvidenceProducts:
    """Build the authoritative hierarchical material decision.

    Background and Other are positive Non-seed evidence.  Other never subtracts
    from Background here; their conditional subtype probabilities may overlap
    or remain unknown. Raw evidence maps are never rewritten from the class
    annotations used to calibrate their reliability.
    """

    import torch

    context = cuda_context or CudaContext.resolve()
    valid = _probability_tensor(valid_mask, context)
    if valid is None:
        raise ValueError("A valid-region mask is required.")
    valid = valid > 0.0
    calibration_valid = (
        valid
        if calibration_valid_mask is None
        else _probability_tensor(calibration_valid_mask, context) > 0.0
    )
    if calibration_valid.shape != valid.shape:
        raise ValueError(
            "Material calibration and decision masks must share dimensions."
        )
    proposal_valid = (
        valid
        if proposal_valid_mask is None
        else _probability_tensor(proposal_valid_mask, context) > 0.0
    )
    if proposal_valid.shape != valid.shape:
        raise ValueError("Material proposal and decision masks must share dimensions.")

    foreground_colour_tensor = _probability_tensor(foreground_colour, context)
    if foreground_colour_tensor is None:
        raise ValueError("Foreground colour evidence is required.")
    evidence_shape = foreground_colour_tensor.shape
    if valid.shape != evidence_shape:
        raise ValueError("Material evidence and valid mask dimensions must match.")

    def converted(values):
        tensor = _probability_tensor(values, context)
        if tensor is not None and tensor.shape != evidence_shape:
            raise ValueError("All material evidence rasters must share dimensions.")
        return tensor

    foreground_noise_tensor = converted(foreground_noise)
    background_colour_tensor = converted(background_colour)
    background_noise_tensor = converted(background_noise)
    other_colour_tensor = converted(other_colour)
    other_noise_tensor = converted(other_noise)
    reference_foreground_tensor = converted(reference_foreground)
    reference_background_tensor = converted(reference_background)
    reference_other_tensor = converted(reference_other)

    seed_reference = _reference_mask(seed_reference_mask, context, evidence_shape)
    additional_seed_reference = _reference_mask(
        additional_seed_reference_mask, context, evidence_shape
    )
    if seed_reference is None:
        seed_reference = additional_seed_reference
    elif additional_seed_reference is not None:
        seed_reference |= additional_seed_reference
    background_reference = _reference_mask(
        background_reference_mask, context, evidence_shape
    )
    other_reference = _reference_mask(
        other_reference_mask, context, evidence_shape
    )

    def union(first, second):
        if first is None:
            return second
        if second is None:
            return first
        return first | second

    seed_nontarget = union(background_reference, other_reference)
    # Background and Other are parallel Non-seed subtypes. They calibrate
    # against reviewed Seed, never against one another, so legitimate glass
    # overlap cannot reduce either source's top-level Non-seed authority.
    background_nontarget = seed_reference
    other_nontarget = seed_reference
    reliability_items = (
        (
            "foreground_colour",
            _source_reliability(
                foreground_colour_tensor,
                seed_reference,
                seed_nontarget,
                calibration_valid,
            ),
        ),
        (
            "foreground_noise",
            _source_reliability(
                foreground_noise_tensor,
                seed_reference,
                seed_nontarget,
                calibration_valid,
            ),
        ),
        (
            "reference_foreground",
            _source_reliability(
                reference_foreground_tensor,
                seed_reference,
                seed_nontarget,
                calibration_valid,
            ),
        ),
        (
            "background_colour",
            _source_reliability(
                background_colour_tensor,
                background_reference,
                background_nontarget,
                calibration_valid,
            ),
        ),
        (
            "background_noise",
            _source_reliability(
                background_noise_tensor,
                background_reference,
                background_nontarget,
                calibration_valid,
            ),
        ),
        (
            "reference_background",
            _source_reliability(
                reference_background_tensor,
                background_reference,
                background_nontarget,
                calibration_valid,
            ),
        ),
        (
            "other_colour",
            _source_reliability(
                other_colour_tensor,
                other_reference,
                other_nontarget,
                calibration_valid,
            ),
        ),
        (
            "other_noise",
            _source_reliability(
                other_noise_tensor,
                other_reference,
                other_nontarget,
                calibration_valid,
            ),
        ),
        (
            "reference_other",
            _source_reliability(
                reference_other_tensor,
                other_reference,
                other_nontarget,
                calibration_valid,
            ),
        ),
    )
    reliabilities = dict(reliability_items)

    seed_support = _weighted_support(
        (
            foreground_colour_tensor,
            foreground_noise_tensor,
            reference_foreground_tensor,
        ),
        (colour_weight, noise_weight, prototype_weight),
        (
            reliabilities["foreground_colour"],
            reliabilities["foreground_noise"],
            reliabilities["reference_foreground"],
        ),
    )
    background_support = _weighted_support(
        (
            background_colour_tensor,
            background_noise_tensor,
            reference_background_tensor,
        ),
        (colour_weight, noise_weight, prototype_weight),
        (
            reliabilities["background_colour"],
            reliabilities["background_noise"],
            reliabilities["reference_background"],
        ),
    )
    other_support = _weighted_support(
        (other_colour_tensor, other_noise_tensor, reference_other_tensor),
        (colour_weight, noise_weight, prototype_weight),
        (
            reliabilities["other_colour"],
            reliabilities["other_noise"],
            reliabilities["reference_other"],
        ),
    )
    zero = torch.zeros_like(foreground_colour_tensor)
    seed_support = zero if seed_support is None else seed_support
    background_support = zero if background_support is None else background_support
    other_support = zero if other_support is None else other_support

    seed_support = seed_support.clamp(0.0, 1.0) * valid
    background_support = background_support.clamp(0.0, 1.0) * valid
    other_support = other_support.clamp(0.0, 1.0) * valid
    nonseed_support = (
        1.0 - (1.0 - background_support) * (1.0 - other_support)
    ) * valid

    seed_mass = seed_support * (1.0 - nonseed_support)
    nonseed_mass = nonseed_support * (1.0 - seed_support)
    ambiguity_mass = 2.0 * seed_support * nonseed_support
    unknown_mass = (
        float(unknown_weight)
        * (1.0 - seed_support)
        * (1.0 - nonseed_support)
    )
    inverse_temperature = 1.0 / max(float(temperature), 1e-6)
    masses = torch.cat(
        (seed_mass, nonseed_mass, ambiguity_mass, unknown_mass), dim=1
    ).clamp_min(1e-8)
    masses = masses.pow(inverse_temperature)
    masses = masses / masses.sum(dim=1, keepdim=True).clamp_min(1e-8)
    masses *= valid
    seed_probability = masses[:, 0:1]
    nonseed_probability = masses[:, 1:2]
    ambiguity_probability = masses[:, 2:3]
    unknown_probability = masses[:, 3:4]

    background_mass = background_support * (1.0 - other_support)
    other_mass = other_support * (1.0 - background_support)
    subtype_ambiguity = 2.0 * background_support * other_support
    subtype_unknown = (
        float(unknown_weight)
        * (1.0 - background_support)
        * (1.0 - other_support)
    )
    subtype_denominator = (
        background_mass
        + other_mass
        + subtype_ambiguity
        + subtype_unknown
    ).clamp_min(1e-8)
    conditional_background = background_mass / subtype_denominator * valid
    conditional_other = other_mass / subtype_denominator * valid
    conditional_ambiguity = subtype_ambiguity / subtype_denominator * valid
    conditional_unknown = subtype_unknown / subtype_denominator * valid

    seed_mask = (seed_probability >= float(seed_threshold)) & proposal_valid
    kernel_size = max(1, round(float(seed_diameter) * float(morphology_fraction)))
    if kernel_size > 1:
        kernel_size = max(3, kernel_size)
        if kernel_size % 2 == 0:
            kernel_size += 1
        seed_mask = binary_close(
            binary_open(seed_mask, kernel_size), kernel_size
        )
        seed_mask &= proposal_valid

    def float_raster(values, name):
        return GpuRaster(
            values.float(), numpy_dtype=np.float32, name=name
        )

    def u8_raster(values, name):
        return GpuRaster(
            torch.round(values.clamp(0.0, 1.0) * 255.0).to(torch.uint8),
            numpy_dtype=np.uint8,
            name=name,
        )

    return MaterialEvidenceProducts(
        seed_support=u8_raster(seed_support, "aggregated seed evidence"),
        background_support=u8_raster(
            background_support, "aggregated background evidence"
        ),
        other_support=u8_raster(other_support, "aggregated other evidence"),
        nonseed_support=u8_raster(nonseed_support, "aggregated non-seed evidence"),
        seed_probability=u8_raster(
            seed_probability, "resolved seed probability"
        ),
        nonseed_probability=u8_raster(
            nonseed_probability, "resolved non-seed probability"
        ),
        ambiguity_probability=u8_raster(
            ambiguity_probability, "material evidence ambiguity"
        ),
        unknown_probability=u8_raster(
            unknown_probability, "unknown material evidence"
        ),
        conditional_background_probability=u8_raster(
            conditional_background, "conditional background probability"
        ),
        conditional_other_probability=u8_raster(
            conditional_other, "conditional other probability"
        ),
        conditional_ambiguity_probability=u8_raster(
            conditional_ambiguity, "conditional material-subtype ambiguity"
        ),
        conditional_unknown_probability=u8_raster(
            conditional_unknown, "conditional unknown non-seed subtype"
        ),
        seed_mask=GpuRaster(
            seed_mask.to(torch.uint8) * 255,
            numpy_dtype=np.uint8,
            name="seed material binary proposal mask",
        ),
        source_reliabilities=tuple(reliability_items),
    )


__all__ = ["MaterialEvidenceProducts", "hierarchical_material_evidence"]
