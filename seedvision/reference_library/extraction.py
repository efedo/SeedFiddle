"""Deterministic compact contribution extraction from reviewed image sidecars."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import cv2
import numpy as np

from seedvision.measurement import (
    measure_reviewed_seed_instances,
    shape_observations_from_summary,
)
from seedvision.persistence.reference_regions import (
    ReferenceRegionBundle,
    file_sha256,
)
from seedvision.reference_library.contracts import (
    BiologicalContext,
    EDGE_PROTOTYPE_SCHEMA,
    FOREGROUND_COLOUR_SCHEMA,
    FOREGROUND_NOISE_SCHEMA,
    MATERIAL_PROTOTYPE_SCHEMA,
    SEED_TRAIT_SCHEMA,
    LibrarySourceRecord,
    ShapeObservation,
    SpeciesEdgePrototypeBank,
    SpeciesForegroundColourBank,
    SpeciesForegroundNoiseBank,
    SpeciesMaterialPrototypeBank,
    SpeciesSeedTraitBank,
)


MATERIAL_FEATURE_NAMES = (
    "Lab lightness", "Lab a*", "Lab b*",
    "fine darkness noise", "medium darkness noise", "coarse darkness noise",
    "fine colour noise", "medium colour noise", "coarse colour noise",
    "edge magnitude", "ridge support", "local lightness residual",
    "local colour residual", "local edge density",
)
NOISE_FEATURE_NAMES = tuple(
    f"{band} {name}"
    for band in ("fine", "medium", "coarse")
    for name in ("residual RMS", "principal-axis variation", "cross-axis variation")
)
EDGE_FEATURE_NAMES = (
    *(f"{zone} strip {feature}" for zone in ("interior", "centre/edge", "exterior")
      for feature in (
          "Lab lightness", "Lab a*", "Lab b*",
          "local lightness residual", "local colour residual",
      )),
    "signed cross-edge lightness (interior - exterior)",
    "signed cross-edge a* (interior - exterior)",
    "signed cross-edge b* (interior - exterior)",
    "strip axial tangent coherence", "strip valid support",
)


@dataclass(frozen=True, slots=True)
class LibraryExtractionSettings:
    maximum_colour_modes_per_source: int = 64
    foreground_colour_frequency_weight_power: float = 0.0
    foreground_colour_scale_multiplier: float = 1.50
    foreground_colour_chroma_weight: float = 1.80
    maximum_material_prototypes_per_source: int = 64
    maximum_edge_prototypes_per_class_per_source: int = 256
    maximum_trait_prototypes_per_class_per_source: int = 64
    minimum_samples_per_prototype: int = 16
    seed_interior_fraction: float = 0.08
    noise_working_maximum_dimension: int = 1280
    noise_medium_scale_fraction: float = 0.03
    noise_coarse_scale_fraction: float = 0.08
    frequency_noise_working_maximum_dimension: int = 1280
    material_working_maximum_dimension: int = 960
    material_context_fraction: float = 0.04
    trait_working_maximum_dimension: int = 1280
    trait_context_fraction: float = 0.04
    edge_working_maximum_dimension: int = 2048
    edge_minimum_working_seed_diameter_px: float = 28.0
    edge_normal_offset_fraction: float = 0.05
    edge_tangent_half_length_fraction: float = 0.08
    edge_ridge_weight: float = 0.35
    edge_interior_buffer_fraction: float = 0.08
    contour_samples: int = 128

    def __post_init__(self) -> None:
        if not 1 <= self.maximum_colour_modes_per_source <= 256:
            raise ValueError("Colour modes per source must be 1--256.")
        if not 0.0 <= self.foreground_colour_frequency_weight_power <= 1.0:
            raise ValueError("Foreground colour frequency weight must be 0--1.")
        if not 0.5 <= self.foreground_colour_scale_multiplier <= 3.0:
            raise ValueError("Foreground colour scale multiplier must be 0.5--3.")
        if not 0.5 <= self.foreground_colour_chroma_weight <= 5.0:
            raise ValueError("Foreground colour chroma weight must be 0.5--5.")
        if not 1 <= self.maximum_material_prototypes_per_source <= 256:
            raise ValueError("Material prototypes per source must be 1--256.")
        if not 1 <= self.maximum_edge_prototypes_per_class_per_source <= 1024:
            raise ValueError("Edge prototypes per class/source must be 1--1024.")
        if not 1 <= self.maximum_trait_prototypes_per_class_per_source <= 256:
            raise ValueError("Trait prototypes per class/source must be 1--256.")
        if not 2 <= self.minimum_samples_per_prototype <= 2048:
            raise ValueError("Prototype support must be 2--2048 samples.")
        if not 0.0 <= self.seed_interior_fraction <= 0.4:
            raise ValueError("Seed-interior fraction must be 0--0.4.")
        dimension_ranges = {
            "noise_working_maximum_dimension": (512, 4096),
            "frequency_noise_working_maximum_dimension": (512, 4096),
            "material_working_maximum_dimension": (256, 2048),
            "trait_working_maximum_dimension": (256, 2048),
            "edge_working_maximum_dimension": (512, 4096),
        }
        for name, (minimum, maximum) in dimension_ranges.items():
            if not minimum <= int(getattr(self, name)) <= maximum:
                raise ValueError(f"{name} must be {minimum}--{maximum} pixels.")
        for name in (
            "noise_medium_scale_fraction", "noise_coarse_scale_fraction",
            "material_context_fraction", "trait_context_fraction",
            "edge_normal_offset_fraction", "edge_tangent_half_length_fraction",
            "edge_interior_buffer_fraction",
        ):
            if not 0.0 < float(getattr(self, name)) <= 0.5:
                raise ValueError(f"{name} must be greater than zero and at most 0.5.")
        if not 8.0 <= float(self.edge_minimum_working_seed_diameter_px) <= 96.0:
            raise ValueError("Minimum working seed diameter must be 8--96 pixels.")
        if not 0.0 <= float(self.edge_ridge_weight) <= 1.0:
            raise ValueError("Edge ridge weight must be 0--1.")


@dataclass(frozen=True, slots=True)
class LibrarySourceInput:
    source_path: Path
    corrected_bgr: np.ndarray
    references: ReferenceRegionBundle
    annotation_sha256: str
    species_display_name: str
    biological_context: BiologicalContext
    capture_group_id: str | None = None
    pixels_per_mm: float | None = None
    calibration_relative_uncertainty: float = 0.0
    coat_pattern_vocabulary: tuple[str, ...] = ()
    condition_vocabulary: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ExtractedSourceContribution:
    source_index: int
    source: LibrarySourceRecord
    foreground_colour: SpeciesForegroundColourBank | None
    foreground_noise: SpeciesForegroundNoiseBank | None
    material_prototypes: SpeciesMaterialPrototypeBank | None
    edge_prototypes: SpeciesEdgePrototypeBank | None
    seed_traits: SpeciesSeedTraitBank | None
    shape_observations: tuple[ShapeObservation, ...]


def extract_source_contribution(
    source: LibrarySourceInput,
    *,
    source_index: int,
    settings: LibraryExtractionSettings | None = None,
    cancellation_requested=None,
) -> ExtractedSourceContribution:
    """Extract source-tagged compact banks using only reviewed sample selectors."""

    settings = settings or LibraryExtractionSettings()
    image = np.asarray(source.corrected_bgr)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("Library extraction requires an 8-bit corrected BGR image.")
    if image.shape[:2] != source.references.shape:
        raise ValueError("Corrected image and reference sidecar dimensions differ.")
    if source.references.annotation_species not in {
        "", source.biological_context.species_id
    }:
        raise ValueError("Reference sidecar species does not match the library species.")
    _cancel(cancellation_requested)
    labels = np.zeros(image.shape[:2], np.uint16)
    if source.references.annotated_seeds is not None:
        labels = np.asarray(source.references.annotated_seeds, np.uint16)
    foreground = _foreground_source_mask(source.references, labels, settings)
    _cancel(cancellation_requested)
    colour, noise, material, edge_bank, trait_bank = _extract_production_banks(
        image,
        foreground,
        labels,
        source.references.seed_annotations,
        source_index,
        settings,
        coat_pattern_vocabulary=source.coat_pattern_vocabulary,
        condition_vocabulary=source.condition_vocabulary,
        cancellation_requested=cancellation_requested,
    )
    summary = measure_reviewed_seed_instances(
        labels,
        source.references.seed_annotations,
        pixels_per_mm=source.pixels_per_mm,
        calibration_relative_uncertainty=source.calibration_relative_uncertainty,
        contour_samples=settings.contour_samples,
    )
    shape_observations = shape_observations_from_summary(
        summary,
        source_index=source_index,
        biological_context=source.biological_context,
    )
    record = LibrarySourceRecord(
        source_sha256=file_sha256(source.source_path),
        annotation_sha256=source.annotation_sha256,
        display_label=source.source_path.name,
        source_shape=image.shape[:2],
        capture_group_id=source.capture_group_id,
        biological_context=source.biological_context,
        review_status="reviewed",
    )
    return ExtractedSourceContribution(
        source_index, record, colour, noise, material, edge_bank, trait_bank,
        shape_observations,
    )


def extraction_settings_sha256(settings: LibraryExtractionSettings) -> str:
    payload = "\n".join(
        f"{name}={getattr(settings, name)!r}"
        for name in settings.__dataclass_fields__
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def _foreground_source_mask(references, labels, settings):
    mask = np.zeros(labels.shape, bool)
    if references.foreground is not None:
        mask |= np.asarray(references.foreground, bool)
    if np.any(labels):
        radius = max(1, round(_reference_diameter(labels) * settings.seed_interior_fraction))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1,) * 2)
        for seed_id in np.unique(labels):
            if seed_id:
                mask |= cv2.erode(np.asarray(labels == seed_id, np.uint8), kernel) > 0
    for excluded in (references.background, references.other):
        if excluded is not None:
            mask &= ~np.asarray(excluded, bool)
    return mask


def _extract_colour_bank(lab, mask, source_index, maximum_modes):
    samples = lab[mask]
    if not len(samples):
        return None
    quantized = np.round(samples / np.asarray((4.0, 3.0, 3.0))).astype(np.int16)
    cells, counts = np.unique(quantized, axis=0, return_counts=True)
    centres = cells.astype(np.float32) * np.asarray((4.0, 3.0, 3.0), np.float32)
    selected = _farthest_first(centres / np.asarray((8.0, 4.0, 4.0)), counts, maximum_modes)
    centres = centres[selected]
    assigned = np.argmin(_squared_distances(samples, centres, (8.0, 4.0, 4.0)), axis=1)
    used_centres, scales, weights, sample_counts = [], [], [], []
    for index in range(len(centres)):
        values = samples[assigned == index]
        if not len(values):
            continue
        centre = centres[index]
        used_centres.append(centre)
        scales.append(np.maximum((8.0, 4.0, 4.0), 1.4826 * np.median(np.abs(values - centre), axis=0)))
        weights.append(len(values))
        sample_counts.append(len(values))
    weights = np.asarray(weights, np.float32)
    weights /= max(float(weights.sum()), 1.0)
    return SpeciesForegroundColourBank(
        np.asarray(used_centres, np.float32), np.asarray(scales, np.float32), weights,
        np.full(len(scales), source_index, np.int32), np.asarray(sample_counts, np.int64),
        FOREGROUND_COLOUR_SCHEMA, ("L*", "a*", "b*"),
    )


def _reference_diameter(labels):
    widths = []
    for seed_id in np.unique(labels):
        if seed_id:
            rows, columns = np.nonzero(labels == seed_id)
            if len(rows):
                widths.append(max(columns.max() - columns.min() + 1, rows.max() - rows.min() + 1))
    return float(np.median(widths)) if widths else max(8.0, min(labels.shape) * 0.05)


def _extract_production_banks(
    image,
    foreground,
    labels,
    annotations,
    source_index,
    settings,
    *,
    coat_pattern_vocabulary=(),
    condition_vocabulary=(),
    cancellation_requested=None,
):
    """Extract compact examples with the exact live descriptor equations.

    Full feature rasters stay on the selected PyTorch device.  Only bounded
    rows selected by reviewed masks, fitted prototypes, and per-seed compact
    descriptors cross to CPU for immutable persistence.
    """

    import torch
    import torch.nn.functional as functional

    from seedvision.annotation.instance_references import (
        instance_boundary_references,
    )
    from seedvision.cuda.layers import (
        _adaptive_edge_working_shape,
        _candidate_internal_edge_mask,
        _edge_strip_feature_maps,
        _instance_outward_normals,
        _noise_scales,
        _noise_texture_features,
        _raster_tensor,
        _reference_material_feature_tensor,
        directional_edges,
        multiscale_frequency_noise_masks,
        thin_probability_ridges,
    )
    from seedvision.cuda.ops import (
        CudaContext,
        bgr_to_lab,
        image_to_tensor,
        lab_colour_frequency_distribution,
    )
    from seedvision.visualization.layers import AnalysisLayerSettings

    analysis_settings = AnalysisLayerSettings(
        noise_medium_scale_fraction=float(settings.noise_medium_scale_fraction),
        noise_coarse_scale_fraction=float(settings.noise_coarse_scale_fraction),
        foreground_noise_medium_scale_fraction=float(
            settings.noise_medium_scale_fraction
        ),
        foreground_noise_coarse_scale_fraction=float(
            settings.noise_coarse_scale_fraction
        ),
        noise_working_maximum_dimension=int(settings.noise_working_maximum_dimension),
        foreground_noise_working_maximum_dimension=int(
            settings.noise_working_maximum_dimension
        ),
        frequency_noise_working_maximum_dimension=int(
            settings.frequency_noise_working_maximum_dimension
        ),
        reference_texture_working_maximum_dimension=int(
            settings.material_working_maximum_dimension
        ),
        reference_texture_context_fraction=float(
            settings.material_context_fraction
        ),
        reference_seed_trait_working_maximum_dimension=int(
            settings.trait_working_maximum_dimension
        ),
        reference_seed_trait_context_fraction=float(settings.trait_context_fraction),
        reference_texture_edge_working_maximum_dimension=int(
            settings.edge_working_maximum_dimension
        ),
        reference_edge_minimum_working_seed_diameter_px=float(
            settings.edge_minimum_working_seed_diameter_px
        ),
        reference_edge_strip_normal_offset_fraction=float(
            settings.edge_normal_offset_fraction
        ),
        reference_edge_strip_tangent_half_length_fraction=float(
            settings.edge_tangent_half_length_fraction
        ),
        reference_edge_ridge_weight=float(settings.edge_ridge_weight),
        reference_texture_instance_interior_buffer_fraction=float(
            settings.edge_interior_buffer_fraction
        ),
    )
    context = CudaContext.resolve()
    source_tensor = image_to_tensor(image, context)
    valid = torch.ones(
        (1, 1, image.shape[0], image.shape[1]),
        device=context.device,
        dtype=torch.bool,
    )
    lab = bgr_to_lab(source_tensor)
    foreground_tensor = image_to_tensor(
        np.asarray(foreground, np.uint8), context
    ) > 0
    colour = None
    if bool(foreground_tensor.any().item()):
        (
            _unused_probability,
            colour_centres,
            colour_scales,
            colour_weights,
            colour_sample_count,
            _unused_assignments,
        ) = lab_colour_frequency_distribution(
            lab[0].permute(1, 2, 0),
            lab[0].permute(1, 2, 0)[foreground_tensor[0, 0]],
            valid[0, 0],
            maximum_bins=int(settings.maximum_colour_modes_per_source),
            refinement_iterations=0,
            frequency_weight_power=float(
                settings.foreground_colour_frequency_weight_power
            ),
            scale_multiplier=float(settings.foreground_colour_scale_multiplier),
            scale_floors=(8.0, 4.0, 4.0),
            distance_weights=(
                1.0,
                float(settings.foreground_colour_chroma_weight),
                float(settings.foreground_colour_chroma_weight),
            ),
        )
        centre_values = colour_centres.detach().cpu().numpy().astype(np.float32)
        scale_values = colour_scales.detach().cpu().numpy().astype(np.float32)
        weight_values = colour_weights.detach().cpu().numpy().astype(np.float32)
        colour = SpeciesForegroundColourBank(
            centre_values,
            scale_values,
            weight_values,
            np.full(len(centre_values), source_index, np.int32),
            np.full(
                len(centre_values),
                max(1, int(colour_sample_count) // max(1, len(centre_values))),
                np.int64,
            ),
            FOREGROUND_COLOUR_SCHEMA,
            ("L*", "a*", "b*"),
        )
    gradients = directional_edges(
        image,
        np.ones(image.shape[:2], np.uint8),
        analysis_settings,
        cuda_context=context,
        source_tensor=source_tensor,
        lab_tensor=lab,
        valid_tensor=valid,
    )
    diameter = _reference_diameter(labels)
    frequency_noise = multiscale_frequency_noise_masks(
        image,
        np.ones(image.shape[:2], np.uint8),
        diameter,
        analysis_settings,
        cuda_context=context,
        lab_tensor=lab,
        valid_tensor=valid,
    )
    ridge = thin_probability_ridges(
        gradients.strength,
        gradients.normal_x,
        gradients.normal_y,
        gradients.valid,
        normal_sampling_step_px=float(analysis_settings.ridge_nms_step_px),
        low_threshold=float(analysis_settings.ridge_low_threshold),
        high_threshold=float(analysis_settings.ridge_high_threshold),
        hysteresis_iterations=int(analysis_settings.ridge_hysteresis_iterations),
    )
    _cancel(cancellation_requested)

    def resized(values, height, width, *, mode="bilinear"):
        tensor = values.float()
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
            tensor, (height, width), mode=mode, **arguments
        )

    def work_shape(maximum_dimension):
        scale = min(1.0, float(maximum_dimension) / max(image.shape[:2]))
        return (
            scale,
            max(8, round(image.shape[0] * scale)),
            max(8, round(image.shape[1] * scale)),
        )

    # Target-only noise distribution.  This is the same nine-channel
    # orientation-invariant feature used by the live material-noise node.
    noise_scale, noise_height, noise_width = work_shape(
        settings.noise_working_maximum_dimension
    )
    noise_lab = resized(lab, noise_height, noise_width) / 255.0
    noise_mask = resized(
        foreground_tensor.float(), noise_height, noise_width, mode="area"
    ) > 0.001
    noise_features = _noise_texture_features(
        noise_lab,
        _noise_scales(diameter * noise_scale, analysis_settings),
    )
    noise_samples = _compact_tensor_samples(noise_features, noise_mask, 200000)
    noise = _extract_profile_samples(
        noise_samples,
        source_index,
        FOREGROUND_NOISE_SCHEMA,
        NOISE_FEATURE_NAMES,
    )
    _cancel(cancellation_requested)

    def material_features_at(maximum_dimension, context_fraction):
        work_scale, height, width = work_shape(maximum_dimension)
        work_lab = resized(lab, height, width)
        work_edge = resized(gradients.strength, height, width).clamp(0.0, 1.0)
        work_ridge = resized(ridge, height, width).clamp(0.0, 1.0)
        work_noise = tuple(
            resized(
                _raster_tensor(value, context, normalized=True), height, width
            ).clamp(0.0, 1.0)
            for value in (
                *frequency_noise.darkness_masks,
                *frequency_noise.colour_masks,
            )
        )
        return (
            work_scale,
            _reference_material_feature_tensor(
                work_lab,
                work_edge,
                work_ridge,
                work_noise,
                context_sigma=max(0.7, diameter * work_scale * context_fraction),
            ),
        )

    material_scale, material_features = material_features_at(
        settings.material_working_maximum_dimension,
        settings.material_context_fraction,
    )
    material_height, material_width = material_features.shape[-2:]
    material_mask = resized(
        foreground_tensor.float(), material_height, material_width, mode="area"
    ) > 0.001
    material_samples = _compact_tensor_samples(
        material_features, material_mask, 32768
    )
    material = _prototype_bank_from_samples(
        {"foreground": material_samples},
        source_index,
        MATERIAL_PROTOTYPE_SCHEMA,
        MATERIAL_FEATURE_NAMES,
        settings.maximum_material_prototypes_per_source,
        settings.minimum_samples_per_prototype,
    )
    _cancel(cancellation_requested)

    trait_scale, trait_features = material_features_at(
        settings.trait_working_maximum_dimension,
        settings.trait_context_fraction,
    )
    trait_labels = _resize_labels(
        labels, trait_features.shape[-2], trait_features.shape[-1]
    )
    trait_bank = _extract_trait_bank_from_tensor(
        trait_features,
        trait_labels,
        annotations,
        source_index,
        settings,
        diameter * trait_scale,
        coat_pattern_vocabulary=coat_pattern_vocabulary,
        condition_vocabulary=condition_vocabulary,
    )
    _cancel(cancellation_requested)

    edge_scale, edge_height, edge_width = _adaptive_edge_working_shape(
        image.shape[0],
        image.shape[1],
        diameter,
        material_maximum_dimension=settings.material_working_maximum_dimension,
        edge_maximum_dimension=settings.edge_working_maximum_dimension,
        minimum_working_seed_diameter_px=(
            settings.edge_minimum_working_seed_diameter_px
        ),
    )
    edge_diameter = max(6.0, diameter * edge_scale)
    edge_labels = _resize_labels(labels, edge_height, edge_width)
    edge_valid = resized(valid.float(), edge_height, edge_width, mode="nearest") > 0.5
    edge_strength = resized(
        gradients.strength, edge_height, edge_width
    ).clamp(0.0, 1.0)
    edge_ridge = resized(ridge, edge_height, edge_width).clamp(0.0, 1.0)
    edge_lab = resized(lab, edge_height, edge_width)
    edge_tangent_x = resized(gradients.tangent_x, edge_height, edge_width)
    edge_tangent_y = resized(gradients.tangent_y, edge_height, edge_width)
    edge_bank = None
    if np.any(edge_labels):
        boundary = instance_boundary_references(
            edge_labels,
            edge_diameter,
            interior_buffer_fraction=settings.edge_interior_buffer_fraction,
        )
        contour, outward_x_numpy, outward_y_numpy = _instance_outward_normals(
            edge_labels
        )
        physical_mask = (
            image_to_tensor(contour.astype(np.uint8), context) > 0
        ) & edge_valid
        safe_interior = (
            image_to_tensor(boundary.safe_interior.astype(np.uint8), context) > 0
        ) & edge_valid & ~physical_mask
        outward_x = image_to_tensor(outward_x_numpy, context)
        outward_y = image_to_tensor(outward_y_numpy, context)
        strip_tangent_x = torch.where(
            physical_mask, outward_y, edge_tangent_x
        )
        strip_tangent_y = torch.where(
            physical_mask, -outward_x, edge_tangent_y
        )
        edge_features, _reverse, strip_valid, _nx, _ny = _edge_strip_feature_maps(
            edge_lab,
            edge_strength,
            edge_ridge,
            strip_tangent_x,
            strip_tangent_y,
            edge_valid,
            edge_diameter,
            normal_offset_fraction=settings.edge_normal_offset_fraction,
            tangent_half_length_fraction=settings.edge_tangent_half_length_fraction,
            include_reverse=False,
            include_normals=False,
        )
        physical_mask &= strip_valid
        non_edge_mask = _candidate_internal_edge_mask(
            edge_strength,
            edge_ridge,
            edge_tangent_x,
            edge_tangent_y,
            safe_interior,
            ridge_weight=settings.edge_ridge_weight,
        ) & strip_valid
        overlap = physical_mask & non_edge_mask
        physical_mask &= ~overlap
        non_edge_mask &= ~overlap
        edge_bank = _prototype_bank_from_samples(
            {
                "physical_edge": _compact_balanced_instance_samples(
                    edge_features, physical_mask, edge_labels, 32768
                ),
                "non_edge": _compact_balanced_instance_samples(
                    edge_features, non_edge_mask, edge_labels, 32768
                ),
            },
            source_index,
            EDGE_PROTOTYPE_SCHEMA,
            EDGE_FEATURE_NAMES,
            settings.maximum_edge_prototypes_per_class_per_source,
            settings.minimum_samples_per_prototype,
        )
    _cancel(cancellation_requested)
    return colour, noise, material, edge_bank, trait_bank


def _compact_tensor_samples(features, mask, maximum):
    """Download at most ``maximum`` deterministic reviewed descriptor rows."""

    import torch

    flat_mask = mask.bool().reshape(-1)
    indices = torch.nonzero(flat_mask, as_tuple=False).reshape(-1)
    if int(indices.numel()) > int(maximum):
        positions = torch.linspace(
            0,
            int(indices.numel()) - 1,
            int(maximum),
            device=indices.device,
        ).round().long()
        indices = indices.index_select(0, positions)
    if not int(indices.numel()):
        return np.empty((0, int(features.shape[1])), np.float32)
    rows = features.permute(0, 2, 3, 1).reshape(-1, int(features.shape[1]))
    return np.ascontiguousarray(
        rows.index_select(0, indices).detach().cpu().numpy(),
        dtype=np.float32,
    )


def _compact_balanced_instance_samples(features, mask, labels, maximum):
    """Select equal bounded edge authority from every reviewed physical seed."""

    import torch

    identifiers = tuple(int(value) for value in np.unique(labels) if value > 0)
    if not identifiers:
        return _compact_tensor_samples(features, mask, maximum)
    per_seed = max(1, int(np.ceil(int(maximum) / len(identifiers))))
    label_tensor = torch.as_tensor(
        labels, device=features.device, dtype=torch.int64
    )[None, None]
    parts = [
        _compact_tensor_samples(
            features, mask & (label_tensor == seed_id), per_seed
        )
        for seed_id in identifiers
    ]
    parts = [value for value in parts if len(value)]
    if not parts:
        return np.empty((0, int(features.shape[1])), np.float32)
    values = np.concatenate(parts, axis=0)
    if len(values) > int(maximum):
        values = values[
            np.linspace(0, len(values) - 1, int(maximum)).round().astype(int)
        ]
    return np.ascontiguousarray(values, dtype=np.float32)


def _resize_labels(labels, height, width):
    if labels.shape == (height, width):
        return np.asarray(labels, np.uint16)
    return cv2.resize(
        np.asarray(labels, np.float32),
        (int(width), int(height)),
        interpolation=cv2.INTER_NEAREST,
    ).astype(np.uint16, copy=False)


def _extract_profile_samples(samples, source_index, schema, names):
    values = np.asarray(samples, np.float32)
    if values.ndim != 2 or not len(values):
        return None
    centre = np.median(values, axis=0)
    scale = np.maximum(
        0.05, 1.4826 * np.median(np.abs(values - centre), axis=0)
    )
    distances = np.mean(((values - centre) / scale) ** 2, axis=1)
    half = max(1.0, float(np.quantile(distances, 0.95)))
    return SpeciesForegroundNoiseBank(
        centre[None], scale[None], np.ones(1, np.float32),
        np.asarray((source_index,), np.int32),
        np.asarray((len(values),), np.int64), schema, tuple(names),
        np.asarray((half,), np.float32),
    )


def _prototype_bank_from_samples(
    class_samples,
    source_index,
    schema_id,
    feature_names,
    maximum_per_class,
    minimum_support,
):
    """Fit source-tagged prototypes from already selected compact rows."""

    all_centres, all_scales, all_weights = [], [], []
    all_sources, all_seeds, all_classes, all_counts = [], [], [], []
    class_names = tuple(class_samples)
    for class_id, class_name in enumerate(class_names):
        samples = np.asarray(class_samples[class_name], np.float32)
        if samples.ndim != 2 or not len(samples):
            continue
        centres, scales, weights, counts = _fit_prototypes(
            samples, maximum_per_class, minimum_support
        )
        all_centres.extend(centres)
        all_scales.extend(scales)
        all_weights.extend(weights)
        all_counts.extend(counts)
        all_sources.extend((source_index,) * len(centres))
        all_seeds.extend((0,) * len(centres))
        all_classes.extend((class_id,) * len(centres))
    if not all_centres:
        return None
    cls = (
        SpeciesMaterialPrototypeBank
        if schema_id == MATERIAL_PROTOTYPE_SCHEMA
        else SpeciesEdgePrototypeBank
    )
    return cls(
        np.asarray(all_centres, np.float32),
        np.asarray(all_scales, np.float32),
        np.asarray(all_weights, np.float32),
        np.asarray(all_sources, np.int32),
        np.asarray(all_seeds, np.int32),
        np.asarray(all_classes, np.int16),
        np.asarray(all_counts, np.int64),
        schema_id,
        tuple(feature_names),
        class_names,
    )


def _extract_trait_bank_from_tensor(
    features,
    labels,
    annotations,
    source_index,
    settings,
    working_diameter,
    *,
    coat_pattern_vocabulary=(),
    condition_vocabulary=(),
):
    """Reduce exact production material descriptors to one vector per seed."""

    import torch

    by_id = {int(item.seed_id): item for item in annotations}
    known_conditions = tuple(condition_vocabulary) or tuple(sorted(
        {condition for item in annotations for condition in item.conditions}
    ))
    class_names = tuple(
        [f"coat:{value}" for value in coat_pattern_vocabulary]
        + [
            f"condition:{condition}:{state}"
            for condition in known_conditions
            for state in ("absent", "present")
        ]
    )
    radius = max(1, round(working_diameter * settings.seed_interior_fraction))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)
    rows_by_class: dict[str, list[tuple[int, np.ndarray]]] = {
        name: [] for name in class_names
    }
    device = features.device
    for seed_id, annotation in by_id.items():
        interior = cv2.erode(
            np.asarray(labels == seed_id, np.uint8), kernel
        ) > 0
        if not np.any(interior):
            continue
        mask = torch.as_tensor(interior, device=device, dtype=torch.bool)
        values = features[0, :, mask]
        if not int(values.shape[1]):
            continue
        descriptor = torch.median(values, dim=1).values.detach().cpu().numpy()
        descriptor = np.asarray(descriptor, np.float32)
        if annotation.coat_pattern:
            rows_by_class.setdefault(
                f"coat:{annotation.coat_pattern}", []
            ).append((seed_id, descriptor))
        if annotation.conditions_reviewed:
            for condition in known_conditions:
                state = "present" if condition in annotation.conditions else "absent"
                rows_by_class.setdefault(
                    f"condition:{condition}:{state}", []
                ).append((seed_id, descriptor))
    if not class_names:
        class_names = tuple(sorted(rows_by_class))
    centres, scales, weights = [], [], []
    sources, seeds, classes, counts = [], [], [], []
    for class_id, name in enumerate(class_names):
        descriptors = rows_by_class.get(name, ())
        if not descriptors:
            continue
        values = np.asarray([value for _seed, value in descriptors], np.float32)
        selected = _farthest_first(
            values,
            np.ones(len(values), np.float32),
            min(settings.maximum_trait_prototypes_per_class_per_source, len(values)),
        )
        class_scale = np.maximum(
            0.045,
            np.std(values, axis=0)
            if len(values) > 1
            else np.ones(values.shape[1], np.float32) * 0.1,
        )
        for index in selected:
            centres.append(values[index])
            scales.append(class_scale)
            weights.append(1.0 / len(selected))
            sources.append(source_index)
            seeds.append(descriptors[int(index)][0])
            classes.append(class_id)
            counts.append(1)
    if not centres:
        return None
    return SpeciesSeedTraitBank(
        np.asarray(centres, np.float32), np.asarray(scales, np.float32),
        np.asarray(weights, np.float32), np.asarray(sources, np.int32),
        np.asarray(seeds, np.int32), np.asarray(classes, np.int16),
        np.asarray(counts, np.int64), SEED_TRAIT_SCHEMA,
        MATERIAL_FEATURE_NAMES, class_names,
    )


def _cancel(callback):
    if callback is not None and callback():
        from seedvision.timing import AnalysisCancelled
        raise AnalysisCancelled("Species-library extraction was cancelled.")


def _squared_distances(samples, centres, scales):
    return np.sum(((samples[:, None] - centres[None]) / np.asarray(scales)) ** 2, axis=2)


def _farthest_first(values, counts, maximum):
    if len(values) <= maximum:
        return np.arange(len(values))
    selected = [int(np.argmax(counts))]
    nearest = np.sum((values - values[selected[0]]) ** 2, axis=1)
    frequency = 0.25 + 0.75 * np.sqrt(counts / max(float(np.max(counts)), 1.0))
    while len(selected) < maximum:
        score = nearest * frequency
        score[selected] = -1
        index = int(np.argmax(score))
        selected.append(index)
        nearest = np.minimum(nearest, np.sum((values - values[index]) ** 2, axis=1))
    return np.asarray(selected, np.int64)


def _fit_prototypes(samples, maximum, minimum_support):
    centre = np.median(samples, axis=0)
    scale = np.maximum(0.045, 1.4826 * np.median(np.abs(samples - centre), axis=0))
    standardized = (samples - centre) / scale
    count = min(maximum, max(1, len(samples) // minimum_support))
    selected = _farthest_first(
        standardized,
        np.ones(len(standardized), np.float32),
        count,
    )
    prototype_centres = samples[selected].copy()
    for _ in range(4):
        assignments = np.argmin(
            _squared_distances(samples, prototype_centres, scale), axis=1
        )
        for index in range(len(prototype_centres)):
            members = samples[assignments == index]
            if len(members):
                prototype_centres[index] = np.median(members, axis=0)
    assignments = np.argmin(
        _squared_distances(samples, prototype_centres, scale), axis=1
    )
    centres, scales, weights, counts = [], [], [], []
    for index in range(len(prototype_centres)):
        members = samples[assignments == index]
        if not len(members):
            continue
        local = np.median(members, axis=0)
        centres.append(local)
        scales.append(
            np.maximum(0.045, 1.4826 * np.median(np.abs(members - local), axis=0))
        )
        counts.append(len(members))
        weights.append(len(members) / len(samples))
    return centres, scales, weights, counts


def _extract_edge_bank(lab, edge, ridge, labels, source_index, settings):
    if not np.any(labels):
        return None
    diameter = _reference_diameter(labels)
    offset = max(1.0, diameter * settings.edge_normal_offset_fraction)
    physical_features = []
    nonphysical_features = []
    local_lab = cv2.GaussianBlur(
        lab, (0, 0), max(0.7, diameter * 0.04)
    )
    gx = cv2.Sobel(lab[..., 0] / 255.0, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(lab[..., 0] / 255.0, cv2.CV_32F, 0, 1, ksize=3)
    for seed_id in np.unique(labels):
        if not seed_id:
            continue
        mask = np.asarray(labels == seed_id, np.uint8)
        contours, _hierarchy = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea).reshape(-1, 2)
        step = max(1, len(contour) // 2048)
        for index in range(0, len(contour), step):
            point = contour[index]
            previous = contour[(index - step) % len(contour)]
            following = contour[(index + step) % len(contour)]
            tangent = following.astype(float) - previous.astype(float)
            feature = _edge_strip_feature(
                lab, local_lab, point, tangent, offset, mask
            )
            if feature is not None:
                physical_features.append(feature)
        erode_radius = max(2, round(diameter * settings.seed_interior_fraction))
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (erode_radius * 2 + 1,) * 2
        )
        interior = cv2.erode(mask, kernel) > 0
        threshold_low = max(1, int(np.quantile(edge[interior], 0.70) * 255)) if np.any(interior) else 255
        internal = (cv2.Canny(np.uint8(lab[..., 0]), threshold_low, min(255, threshold_low * 2)) > 0) & interior
        locations = np.column_stack(np.nonzero(internal))
        if len(locations) > 4096:
            locations = locations[
                np.linspace(0, len(locations) - 1, 4096).round().astype(int)
            ]
        for y, x in locations:
            normal = np.asarray((gx[y, x], gy[y, x]), float)
            norm = np.linalg.norm(normal)
            if norm <= 1e-8:
                continue
            tangent = np.asarray((-normal[1], normal[0])) / norm
            feature = _edge_strip_feature(
                lab, local_lab, np.asarray((x, y)), tangent, offset, mask
            )
            if feature is not None:
                nonphysical_features.append(feature)
    class_samples = {
        "physical_edge": np.asarray(physical_features, np.float32),
        "non_physical_edge": np.asarray(nonphysical_features, np.float32),
    }
    centres = []
    scales = []
    weights = []
    sources = []
    seeds = []
    classes = []
    counts = []
    for class_id, name in enumerate(class_samples):
        samples = class_samples[name]
        if samples.ndim != 2 or not len(samples):
            continue
        fitted = _fit_prototypes(
            samples,
            settings.maximum_edge_prototypes_per_class_per_source,
            settings.minimum_samples_per_prototype,
        )
        class_centres, class_scales, class_weights, class_counts = fitted
        centres.extend(class_centres)
        scales.extend(class_scales)
        weights.extend(class_weights)
        counts.extend(class_counts)
        sources.extend((source_index,) * len(class_centres))
        seeds.extend((0,) * len(class_centres))
        classes.extend((class_id,) * len(class_centres))
    if not centres:
        return None
    return SpeciesEdgePrototypeBank(
        np.asarray(centres, np.float32), np.asarray(scales, np.float32),
        np.asarray(weights, np.float32), np.asarray(sources, np.int32),
        np.asarray(seeds, np.int32), np.asarray(classes, np.int16),
        np.asarray(counts, np.int64), EDGE_PROTOTYPE_SCHEMA,
        EDGE_FEATURE_NAMES, tuple(class_samples),
    )


def _edge_strip_feature(lab, local_lab, point, tangent, offset, seed_mask):
    tangent = np.asarray(tangent, float)
    norm = np.linalg.norm(tangent)
    if norm <= 1e-8:
        return None
    tangent /= norm
    normal = np.asarray((-tangent[1], tangent[0]))
    center = np.asarray(point, float)
    first = center + normal * offset
    second = center - normal * offset

    def inside(position):
        x, y = np.round(position).astype(int)
        return 0 <= y < seed_mask.shape[0] and 0 <= x < seed_mask.shape[1] and bool(seed_mask[y, x])

    # The inside/outside assignment is suggestive metadata embedded in signed
    # strip features; it never changes the physical/non-physical class label.
    if inside(first) and not inside(second):
        interior, exterior = first, second
    elif inside(second) and not inside(first):
        interior, exterior = second, first
    else:
        interior, exterior = first, second
    samples = [_bilinear(lab, position) for position in (interior, center, exterior)]
    local_samples = [
        _bilinear(local_lab, position)
        for position in (interior, center, exterior)
    ]
    if any(value is None for value in (*samples, *local_samples)):
        return None
    inside_lab, edge_lab, outside_lab = samples
    zone_features = []
    for value, local in zip(samples, local_samples, strict=True):
        residual = value - local
        zone_features.extend(
            (
                value[0] / 255.0,
                (value[1] - 128.0) / 128.0,
                (value[2] - 128.0) / 128.0,
                abs(residual[0]) / 40.0,
                np.hypot(residual[1], residual[2]) / 55.0,
            )
        )
    signed = inside_lab - outside_lab
    return np.asarray(
        (*zone_features,
            signed[0] / 40.0, signed[1] / 30.0, signed[2] / 30.0,
            1.0, 1.0,
        ),
        np.float32,
    )


def _bilinear(values, point):
    x, y = float(point[0]), float(point[1])
    if x < 0 or y < 0 or x > values.shape[1] - 1 or y > values.shape[0] - 1:
        return None
    x0, y0 = int(np.floor(x)), int(np.floor(y))
    x1, y1 = min(x0 + 1, values.shape[1] - 1), min(y0 + 1, values.shape[0] - 1)
    wx, wy = x - x0, y - y0
    return (
        values[y0, x0] * (1 - wx) * (1 - wy)
        + values[y0, x1] * wx * (1 - wy)
        + values[y1, x0] * (1 - wx) * wy
        + values[y1, x1] * wx * wy
    )


def _extract_trait_bank(
    features,
    labels,
    annotations,
    source_index,
    settings,
    *,
    coat_pattern_vocabulary=(),
    condition_vocabulary=(),
):
    by_id = {item.seed_id: item for item in annotations}
    known_conditions = tuple(condition_vocabulary) or tuple(sorted(
        {condition for item in annotations for condition in item.conditions}
    ))
    class_names = tuple(
        [f"coat:{value}" for value in coat_pattern_vocabulary]
        + [
            f"condition:{condition}:{state}"
            for condition in known_conditions
            for state in ("absent", "present")
        ]
    )
    samples_by_class: dict[str, list[tuple[int, np.ndarray]]] = {
        name: [] for name in class_names
    }
    planes = np.moveaxis(features, 0, -1)
    diameter = _reference_diameter(labels)
    radius = max(1, round(diameter * settings.seed_interior_fraction))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1,) * 2)
    for seed_id, annotation in by_id.items():
        interior = cv2.erode(np.asarray(labels == seed_id, np.uint8), kernel) > 0
        values = planes[interior]
        if not len(values):
            continue
        descriptor = np.median(values, axis=0)
        if annotation.coat_pattern:
            samples_by_class.setdefault(f"coat:{annotation.coat_pattern}", []).append((seed_id, descriptor))
        if annotation.conditions_reviewed:
            for condition in known_conditions:
                state = "present" if condition in annotation.conditions else "absent"
                samples_by_class.setdefault(f"condition:{condition}:{state}", []).append((seed_id, descriptor))
    if not samples_by_class:
        return None
    if not class_names:
        class_names = tuple(sorted(samples_by_class))
    centres = []
    scales = []
    weights = []
    sources = []
    seeds = []
    classes = []
    counts = []
    for class_id, name in enumerate(class_names):
        seed_descriptors = samples_by_class[name]
        if not seed_descriptors:
            continue
        sample_values = np.asarray([value for _seed, value in seed_descriptors], np.float32)
        selected = _farthest_first(
            sample_values,
            np.ones(len(sample_values), np.float32),
            min(settings.maximum_trait_prototypes_per_class_per_source, len(sample_values)),
        )
        for index in selected:
            centres.append(sample_values[index])
            scales.append(np.maximum(0.045, np.std(sample_values, axis=0) if len(sample_values) > 1 else np.ones(sample_values.shape[1]) * 0.1))
            weights.append(1.0 / len(selected))
            sources.append(source_index)
            seeds.append(seed_descriptors[int(index)][0])
            classes.append(class_id)
            counts.append(1)
    return SpeciesSeedTraitBank(
        np.asarray(centres, np.float32), np.asarray(scales, np.float32),
        np.asarray(weights, np.float32), np.asarray(sources, np.int32),
        np.asarray(seeds, np.int32), np.asarray(classes, np.int16),
        np.asarray(counts, np.int64), SEED_TRAIT_SCHEMA,
        MATERIAL_FEATURE_NAMES, class_names,
    )
