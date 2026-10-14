"""Snapshots of the existing CPU review inputs actually used for geometry."""
from dataclasses import asdict, dataclass, fields, replace
from hashlib import sha256
import json

import numpy as np

from seedvision.annotation.coordinates import validated_transform
from seedvision.persistence.reference_regions import SeedInstanceAnnotation, validated_source_sha256


REVIEWED_INPUT_DEFINITION = 'seedfiddle.reviewed-measurement-inputs-v1'


def facts_sha256(value):
    return sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                             separators=(',', ':')).encode('utf-8')).hexdigest()


def _matrix(value):
    if value is None:
        return None
    return tuple(tuple(float(x) for x in row) for row in validated_transform(value))


@dataclass(frozen=True, slots=True)
class ReviewedMeasurementContext:
    source_sha256: str | None
    corrected_image_sha256: str
    corrected_image_shape: tuple[int, ...]
    corrected_image_dtype: str
    current_source_to_corrected: tuple[tuple[float, ...], ...]
    reference_source_to_corrected: tuple[tuple[float, ...], ...] | None
    applied_label_mapping: tuple[tuple[float, ...], ...] | None
    calibration_facts_sha256: str | None = None
    reference_archive_source_sha256: str | None = None
    reference_annotation_snapshot_source_sha256: str | None = None
    reference_archive_source_identity_state: str = 'not_recorded'
    reference_annotation_snapshot_source_identity_state: str = 'not_recorded'


def reference_source_identity_state(source_sha256, known_sha256, name):
    """Verify only against the actual source identity supplied by analysis."""
    known = validated_source_sha256(known_sha256, name)
    if known is None:
        return 'not_recorded'
    source = validated_source_sha256(source_sha256, 'current source SHA-256')
    if source is None:
        return 'recorded_unverified'
    if known != source:
        raise ValueError(f'{name} differs from the current source; reload and revalidate references.')
    return 'verified'


def reviewed_measurement_context(source_sha256, corrected_bgr, current_transform,
                                 reference_transform=None, *, calibration_facts=None,
                                 transport_applied=False, reference_archive_source_sha256=None,
                                 reference_annotation_snapshot_source_sha256=None):
    """Record declared transport; never infer its existence from matching shapes."""
    image = np.ascontiguousarray(corrected_bgr)
    current, reference = _matrix(current_transform), _matrix(reference_transform)
    if transport_applied and reference is None:
        raise ValueError('Applied reference transport requires its original transform.')
    mapping = None if reference is None or not transport_applied else _matrix(
        np.asarray(current) @ np.linalg.inv(np.asarray(reference)))
    archive_state = reference_source_identity_state(source_sha256,
        reference_archive_source_sha256, 'reference archive source SHA-256')
    snapshot_state = reference_source_identity_state(source_sha256,
        reference_annotation_snapshot_source_sha256, 'reference annotation snapshot source SHA-256')
    return ReviewedMeasurementContext(source_sha256, sha256(image.tobytes()).hexdigest(),
        tuple(image.shape), image.dtype.str, current, reference, mapping,
        facts_sha256(calibration_facts) if calibration_facts is not None else None,
        reference_archive_source_sha256, reference_annotation_snapshot_source_sha256,
        archive_state, snapshot_state)


@dataclass(frozen=True, slots=True)
class ReviewedMeasurementLineage:
    input_definition_revision: str
    mask_sha256: str
    mask_shape: tuple[int, int]
    mask_dtype: str
    effective_annotations_sha256: str
    source_image_frame: str = 'unknown'
    transform_status: str = 'not_recorded'
    alignment: str = 'exact_shape'
    source_sha256: str | None = None
    corrected_image_sha256: str | None = None
    corrected_image_shape: tuple[int, ...] | None = None
    corrected_image_dtype: str | None = None
    current_source_to_corrected: tuple[tuple[float, ...], ...] | None = None
    reference_source_to_corrected: tuple[tuple[float, ...], ...] | None = None
    applied_label_mapping: tuple[tuple[float, ...], ...] | None = None
    calibration_facts_sha256: str | None = None
    reference_source_identity_state: str = 'not_recorded'
    reference_archive_source_sha256: str | None = None
    reference_annotation_snapshot_source_sha256: str | None = None
    reference_archive_source_identity_state: str = 'not_recorded'
    reference_annotation_snapshot_source_identity_state: str = 'not_recorded'


@dataclass(frozen=True, slots=True)
class PreparedReviewedInputs:
    labels: np.ndarray | None
    annotations: tuple[SeedInstanceAnnotation, ...]
    lineage: ReviewedMeasurementLineage | None
    cache_key: str


def effective_annotations(labels, annotations):
    """Detach last-wins annotations/defaults for sorted extant positive IDs."""
    by_id = {}
    for item in annotations:
        if (not isinstance(item, SeedInstanceAnnotation)
                or isinstance(item.seed_id, (bool, np.bool_))
                or not isinstance(item.seed_id, (int, np.integer))
                or not 1 <= item.seed_id <= 65535):
            raise ValueError('Reviewed annotation IDs must be integers in 1..65535.')
        by_id[int(item.seed_id)] = item
    result = []
    for seed_id in (int(x) for x in np.unique(labels) if x > 0):
        item = by_id.get(seed_id, SeedInstanceAnnotation(seed_id))
        if not isinstance(item, SeedInstanceAnnotation):
            raise ValueError('Reviewed measurement annotations must be typed records.')
        # Frozen dataclasses can still receive mutable sequences from callers.
        values = {field.name: getattr(item, field.name) for field in fields(item)}
        values['seed_id'] = seed_id
        values['conditions'] = tuple(item.conditions)
        for name in ('hilum_point', 'hilum_direction'):
            values[name] = None if values[name] is None else tuple(values[name])
        result.append(SeedInstanceAnnotation(**values))
    return tuple(result)


def prepare_reviewed_inputs(labels, annotations, *, context=None):
    """Hash and measure the same detached inputs; no CUDA raster is transferred."""
    if labels is None:
        key = facts_sha256({'input_definition_revision': REVIEWED_INPUT_DEFINITION,
                            'labels': None, 'context': asdict(context) if context else None})
        return PreparedReviewedInputs(None, (), None, key)
    values = np.asarray(labels)
    if (values.ndim != 2 or not np.issubdtype(values.dtype, np.integer)
            or np.any(values < 0) or np.any(values > 65535)):
        raise ValueError('Seed labels must be a two-dimensional integer raster with IDs 0..65535.')
    values = np.array(values, dtype=np.uint16, order='C', copy=True)
    values.flags.writeable = False
    annotations = effective_annotations(values, annotations)
    lineage = ReviewedMeasurementLineage(REVIEWED_INPUT_DEFINITION,
        sha256(values.tobytes()).hexdigest(), tuple(values.shape), values.dtype.str,
        facts_sha256([asdict(item) for item in annotations]))
    if context is not None:
        for name in ('reference_archive_source_sha256', 'reference_annotation_snapshot_source_sha256'):
            state = reference_source_identity_state(context.source_sha256, getattr(context, name), name)
            if state != getattr(context, name.removesuffix('_sha256') + '_identity_state'):
                raise ValueError(f'Reviewed measurement context has an inconsistent {name} state.')
        if tuple(values.shape) != context.corrected_image_shape[:2]:
            raise ValueError('Reviewed labels require the exact corrected-image canvas.')
        proven = context.applied_label_mapping is not None
        states = (context.reference_archive_source_identity_state,
                  context.reference_annotation_snapshot_source_identity_state)
        identity_state = ('recorded_unverified' if 'recorded_unverified' in states else
                          'verified' if 'verified' in states else 'not_recorded')
        lineage = replace(lineage, **asdict(context),
            source_image_frame='full_corrected' if proven else 'unknown',
            transform_status='proven_reprojection' if proven else 'legacy_unknown',
            reference_source_identity_state=identity_state)
    return PreparedReviewedInputs(values, annotations, lineage, facts_sha256(asdict(lineage)))
