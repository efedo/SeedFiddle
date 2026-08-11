"""Checkpointed model feature-stack assembly on the tensor device."""

from __future__ import annotations

from dataclasses import replace
from typing import Mapping

from seedvision.cuda import GpuRaster, bgr_to_lab, image_to_tensor
from seedvision.learning.contracts import FeatureStackSpec


def colour_only_feature_spec(
    *,
    include_species_planes: bool = True,
    nominal_seed_diameter_px: float = 48.0,
) -> FeatureStackSpec:
    return replace(
        FeatureStackSpec(),
        channels=("lab_l", "lab_a", "lab_b", "valid_dish"),
        include_species_planes=include_species_planes,
        nominal_seed_diameter_px=nominal_seed_diameter_px,
    )


def _nchw(values, *, device):
    import torch

    if isinstance(values, GpuRaster):
        tensor = values.gpu_tensor(device=device, dtype=torch.float32)
    elif torch.is_tensor(values):
        tensor = values.to(device=device, dtype=torch.float32)
    else:
        tensor = image_to_tensor(values).to(device=device, dtype=torch.float32)
    if tensor.ndim == 2:
        tensor = tensor[None, None]
    elif tensor.ndim == 3:
        tensor = tensor[None]
    if tensor.ndim != 4:
        raise ValueError("Model feature rasters must be two-dimensional or NCHW tensors.")
    return tensor


def assemble_feature_stack(
    source_bgr,
    valid_mask,
    evidence: Mapping[str, object],
    *,
    spec: FeatureStackSpec = FeatureStackSpec(),
    species: str | None = None,
):
    """Assemble normalized NCHW features without downloading GPU rasters."""

    import torch

    source = _nchw(source_bgr, device=(
        source_bgr.device
        if isinstance(source_bgr, GpuRaster)
        else source_bgr.device
        if torch.is_tensor(source_bgr)
        else _nchw(valid_mask, device=None).device
    ))
    device = source.device
    valid = _nchw(valid_mask, device=device)[:, :1]
    if float(valid.detach().max().item()) > 1.5:
        valid = valid / 255.0
    valid = valid.clamp(0.0, 1.0)
    lab = bgr_to_lab(source)
    available = {
        "lab_l": lab[:, 0:1] / 127.5 - 1.0,
        "lab_a": lab[:, 1:2] / 127.5 - 1.0,
        "lab_b": lab[:, 2:3] / 127.5 - 1.0,
        "valid_dish": valid,
    }
    aliases = {
        "foreground_colour": "foreground_colour",
        "foreground_noise": "foreground_noise",
        "edge_magnitude": "edge_magnitude",
        "sensor_noise": "sensor_noise",
        "flattened_grayscale": "flattened_grayscale",
        "shadow": "shadow",
        "highlight": "highlight",
    }
    for output_name, evidence_name in aliases.items():
        if evidence_name in evidence and evidence[evidence_name] is not None:
            tensor = _nchw(evidence[evidence_name], device=device)[:, :1]
            if float(tensor.detach().max().item()) > 1.5:
                tensor = tensor / 255.0
            available[output_name] = tensor.clamp(0.0, 1.0)
    for output_name, evidence_name in (
        ("inverse_background_colour", "background_colour"),
        ("inverse_background_noise", "background_noise"),
    ):
        if evidence_name in evidence and evidence[evidence_name] is not None:
            tensor = _nchw(evidence[evidence_name], device=device)[:, :1]
            if float(tensor.detach().max().item()) > 1.5:
                tensor = tensor / 255.0
            available[output_name] = 1.0 - tensor.clamp(0.0, 1.0)

    missing = [name for name in spec.channels if name not in available]
    if missing:
        raise ValueError(
            "The checkpoint requires unavailable feature channels: "
            + ", ".join(missing)
        )
    channels = [available[name] for name in spec.channels]
    if spec.include_species_planes:
        species_index = spec.species_index(species)
        for index in range(len(spec.species)):
            channels.append(
                torch.full_like(valid, 1.0 if index == species_index else 0.0)
            )
    stack = torch.cat(channels, dim=1)
    if stack.shape[1] != spec.input_channels:
        raise RuntimeError("The assembled feature count does not match its specification.")
    return stack
