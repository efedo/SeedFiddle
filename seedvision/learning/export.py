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
    corrected_pattern_boundary: np.ndarray | None = None,
    corrected_pattern_valid: np.ndarray | None = None,
    replace_existing: bool = False,
):
    """Export one dish crop normalized to the checkpoint's nominal seed scale."""

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

    pattern = None
    pattern_valid = None
    if corrected_pattern_boundary is not None or corrected_pattern_valid is not None:
        if corrected_pattern_boundary is None or corrected_pattern_valid is None:
            raise ValueError("Pattern boundary and validity masks are required together.")
        full_pattern = np.asarray(corrected_pattern_boundary)
        full_valid = np.asarray(corrected_pattern_valid)
        if full_pattern.shape != corrected_shape or full_valid.shape != corrected_shape:
            raise ValueError("Pattern annotations must use corrected-image coordinates.")
        pattern = full_pattern[crop_slice]
        pattern_valid = full_valid[crop_slice]
        if pattern.shape != (target_height, target_width):
            pattern = cv2.resize(
                np.uint8(pattern > 0),
                (target_width, target_height),
                interpolation=cv2.INTER_NEAREST,
            )
            pattern_valid = cv2.resize(
                np.uint8(pattern_valid > 0),
                (target_width, target_height),
                interpolation=cv2.INTER_NEAREST,
            )

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
