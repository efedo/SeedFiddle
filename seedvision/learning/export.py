"""Export corrected Seed Fiddle analyses as reproducible learning samples."""

from __future__ import annotations

from pathlib import Path
from dataclasses import asdict

import cv2
import numpy as np

from seedvision.learning.contracts import FeatureStackSpec
from seedvision.learning.data import export_learning_sample, file_sha256
from seedvision.learning.features import assemble_feature_stack, pipeline_evidence
from seedvision.learning.targets import relabel_consecutive


def analysis_evidence(result) -> dict[str, object]:
    """Return the same evidence mapping used by live learned DAG nodes."""

    return pipeline_evidence(result.foreground_colour_probability, result.layers, result.advanced)


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
    replace_existing: bool = False,
    seed_annotations: tuple = (),
    species_reviewed: bool = False,
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
    from seedvision.annotation.eligibility import eligible_instances
    eligible = eligible_instances(corrected_labels, seed_annotations)
    if not np.array_equal(eligible, corrected_labels):
        raise ValueError('Learning export requires complete, connected, shape-reviewed contours for every supplied seed. Partial/unreviewed shapes cannot define physical-boundary truth.')
    if feature_spec.include_species_planes and (not species_reviewed or feature_spec.species_index(species) == 0):
        raise ValueError('Species-conditioned features require reviewed, assigned species metadata. Choose a verified species or export an explicitly unconditioned feature specification.')
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
        provenance={
            'feature_recipe': 'raw-foreground-colour-v2',
            'analysis_mode': 'image_local_adaptation',
            'species_reviewed': bool(species_reviewed),
            'source_sha256': None if getattr(result, 'image_path', None) is None else file_sha256(result.image_path),
            'source_to_corrected': getattr(result.calibration, 'affine_matrix', np.eye(3)).tolist(),
            'canonical_scale': scale,
            'annotation_metadata': [asdict(item) for item in seed_annotations],
        },
    )
