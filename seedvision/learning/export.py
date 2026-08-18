"""Export corrected Seed Fiddle analyses as reproducible learning samples."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from seedvision.learning.contracts import FeatureStackSpec
from seedvision.learning.data import export_learning_sample
from seedvision.learning.features import assemble_feature_stack
from seedvision.learning.targets import relabel_consecutive


def analysis_evidence(result) -> dict[str, object]:
    """Return the same evidence mapping used by live learned DAG nodes."""

    return {
        "foreground_colour": result.foreground_probability,
        "foreground_noise": result.layers.foreground_noise_likelihood,
        "background_colour": result.layers.background_likelihood,
        "background_noise": result.layers.refined_background_likelihood,
        "edge_magnitude": result.layers.edge_likelihood,
        "sensor_noise": result.advanced.rasters["sensor_noise"],
        "flattened_grayscale": result.advanced.rasters["flattened_grayscale"],
        "shadow": result.advanced.rasters["shadow_likelihood"],
        "highlight": result.advanced.rasters["highlight_likelihood"],
    }


def _canonical_size(shape: tuple[int, int], scale: float) -> tuple[int, int]:
    height, width = shape
    return max(16, round(height * scale)), max(16, round(width * scale))


def annotation_proposal_to_corrected(result, proposal) -> np.ndarray:
    """Expand one pipeline instance result into editable corrected coordinates.

    Procedural topology deliberately remains at a bounded working resolution,
    while learned decoders normally return the analysis-crop resolution.  The
    annotation editor uses the complete corrected photograph.  This function
    is the single, explicit conversion between those coordinate systems and
    never presents the prediction as reviewed ground truth.
    """

    if proposal is None or not hasattr(proposal, "labels"):
        raise ValueError("An instance proposal with a label raster is required.")
    labels = np.asarray(proposal.labels)
    if labels.ndim != 2 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("Instance proposal labels must be a two-dimensional integer raster.")
    if np.any(labels < 0):
        raise ValueError("Instance proposal labels cannot be negative.")
    maximum = int(labels.max(initial=0))
    if maximum > np.iinfo(np.uint16).max:
        raise ValueError("Instance proposal contains more than 65,535 identifiers.")

    corrected_shape = tuple(int(value) for value in result.calibration.corrected_bgr.shape[:2])
    crop_shape = tuple(int(value) for value in result.layers.valid_mask.shape)
    if len(crop_shape) != 2:
        raise ValueError("Analysis valid mask must be two-dimensional.")
    if labels.shape != crop_shape:
        labels = cv2.resize(
            labels.astype(np.uint16, copy=False),
            (crop_shape[1], crop_shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    labels = relabel_consecutive(labels).astype(np.uint16, copy=False)

    offset_x, offset_y = (int(value) for value in result.crop_offset)
    crop_height, crop_width = crop_shape
    if (
        offset_x < 0
        or offset_y < 0
        or offset_x + crop_width > corrected_shape[1]
        or offset_y + crop_height > corrected_shape[0]
    ):
        raise ValueError("Analysis crop falls outside the corrected image.")
    corrected = np.zeros(corrected_shape, dtype=np.uint16)
    corrected[
        offset_y : offset_y + crop_height,
        offset_x : offset_x + crop_width,
    ] = labels
    return corrected


def export_analysis_sample(
    result,
    corrected_instance_labels: np.ndarray,
    manifest_path: Path | str,
    *,
    dataset_id: str,
    identifier: str,
    species: str,
    group: str,
    split: str = "train",
    reviewed: bool = False,
    annotation_author: str | None = None,
    annotation_revision: str | None = None,
    notes: str | None = None,
    feature_spec: FeatureStackSpec = FeatureStackSpec(),
    replace_existing: bool = False,
):
    """Export one dish crop normalized to the checkpoint's nominal seed scale.

    Applied instance masks provide the physical contours and constrain sparse
    internal generic edge/ridge candidates to non-physical pattern targets.
    No independently painted boundary raster can override the authoritative
    seed identities.
    """

    import torch
    import torch.nn.functional as functional

    corrected_labels = np.asarray(corrected_instance_labels)
    corrected_shape = result.calibration.corrected_bgr.shape[:2]
    if corrected_labels.shape != corrected_shape:
        raise ValueError("Instance labels must use corrected-image coordinates.")
    offset_x, offset_y = result.crop_offset
    crop_height, crop_width = result.layers.valid_mask.shape
    crop_slice = (
        slice(offset_y, offset_y + crop_height),
        slice(offset_x, offset_x + crop_width),
    )
    labels = relabel_consecutive(corrected_labels[crop_slice])
    if not labels.max(initial=0):
        raise ValueError("At least one annotated seed instance is required for export.")
    features = assemble_feature_stack(
        result.layers.gpu_source,
        result.layers.gpu_valid,
        analysis_evidence(result),
        spec=feature_spec,
        species=species,
    )
    actual_diameter = max(1.0, float(result.estimated_seed_diameter_px))
    scale = min(4.0, max(0.25, feature_spec.nominal_seed_diameter_px / actual_diameter))
    target_height, target_width = _canonical_size(labels.shape, scale)
    if (target_height, target_width) != labels.shape:
        features = functional.interpolate(
            features,
            size=(target_height, target_width),
            mode="bilinear",
            align_corners=False,
        )
        labels = cv2.resize(
            labels,
            (target_width, target_height),
            interpolation=cv2.INTER_NEAREST,
        )
    image = np.asarray(result.calibration.corrected_bgr[crop_slice], dtype=np.uint8)
    if image.shape[:2] != (target_height, target_width):
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
        image = cv2.resize(image, (target_width, target_height), interpolation=interpolation)

    from seedvision.annotation import (
        instance_boundary_references_from_edge_evidence,
    )

    edge = np.asarray(result.layers.edge_likelihood, dtype=np.uint8)
    ridges_source = result.layers.edge_ridges
    ridges = (
        np.zeros(edge.shape, np.uint8)
        if ridges_source is None
        else np.asarray(ridges_source, dtype=np.uint8)
    )
    if edge.shape != (crop_height, crop_width) or ridges.shape != edge.shape:
        raise ValueError(
            "Analysis edge evidence must share the exported dish-crop dimensions."
        )
    if edge.shape != (target_height, target_width):
        edge = cv2.resize(
            edge,
            (target_width, target_height),
            interpolation=cv2.INTER_AREA,
        )
        ridges = cv2.resize(
            ridges,
            (target_width, target_height),
            interpolation=cv2.INTER_AREA,
        )
    derived = instance_boundary_references_from_edge_evidence(
        labels,
        float(feature_spec.nominal_seed_diameter_px),
        edge,
        ridges,
    )
    pattern = derived.non_edge
    pattern_valid = derived.safe_interior

    feature_values = features[0].detach().to(device="cpu", dtype=torch.float32).numpy()
    return export_learning_sample(
        manifest_path,
        dataset_id=dataset_id,
        feature_spec=feature_spec,
        identifier=identifier,
        features=feature_values,
        labels=labels,
        image_bgr=image,
        species=species,
        group=group,
        split=split,
        reviewed=reviewed,
        pattern_boundary=pattern,
        pattern_valid=pattern_valid,
        annotation_author=annotation_author,
        annotation_revision=annotation_revision,
        notes=notes,
        replace_existing=replace_existing,
    )
