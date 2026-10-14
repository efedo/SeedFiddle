"""Export stored reviewed geometry separately from proposed instance measurements."""
from dataclasses import asdict
import json

import numpy as np

from seedvision.measurement.lineage import facts_sha256, reference_source_identity_state
from seedvision.reference_library.contracts import SHAPE_MEASUREMENT_NAMES


def _bundle(measurement):
    if measurement is None:
        return None
    return dict(asdict(measurement), vector=list(measurement.vector()),
        feature_order=list(SHAPE_MEASUREMENT_NAMES),
        described_values=[asdict(value) for value in measurement.described_values])


def _validate_bindings(summary, identity):
    lineage = summary.lineage
    if lineage is None:
        return
    for name in ('reference_archive_source_sha256', 'reference_annotation_snapshot_source_sha256'):
        known = getattr(lineage, name)
        recorded_state = getattr(lineage, name.removesuffix('_sha256') + '_identity_state')
        stored_state = reference_source_identity_state(lineage.source_sha256, known, name)
        if recorded_state != stored_state:
            raise ValueError(f'Invalid stored reviewed measurement {name} state.')
        # A newly available export source does not rewrite a stored unknown state.
        actual_state = reference_source_identity_state(identity['source_sha256'], known, name)
        if recorded_state == 'verified' and actual_state != 'verified':
            raise ValueError(f'Stored reviewed measurement {name} cannot be verified; reanalyze before export.')
        if recorded_state not in {'not_recorded', 'recorded_unverified', 'verified'} or (
            (known is None) != (recorded_state == 'not_recorded')
        ):
            raise ValueError(f'Invalid stored reviewed measurement {name} state.')
    states = (lineage.reference_archive_source_identity_state,
              lineage.reference_annotation_snapshot_source_identity_state)
    aggregate = ('recorded_unverified' if 'recorded_unverified' in states else
                 'verified' if 'verified' in states else 'not_recorded')
    if lineage.reference_source_identity_state != aggregate:
        raise ValueError('Invalid stored reviewed measurement reference source identity state.')
    annotations = [asdict(item.annotation) for item in summary.observations]
    if facts_sha256(annotations) != lineage.effective_annotations_sha256:
        raise ValueError('Stored reviewed annotations disagree with their measurement lineage.')
    for name in ('source_sha256', 'corrected_image_sha256', 'corrected_image_shape',
                 'corrected_image_dtype'):
        known = getattr(lineage, name)
        actual = identity[name]
        if known is not None and (list(known) if name.endswith('_shape') else known) != actual:
            raise ValueError(f'Stored reviewed measurement {name} changed; reanalyze before export.')
    known = lineage.current_source_to_corrected
    if known is not None and not np.array_equal(known, identity['source_to_corrected']):
        raise ValueError('Stored reviewed measurement transform changed; reanalyze before export.')
    known = lineage.calibration_facts_sha256
    if known is not None and known != facts_sha256(identity['calibration_facts']):
        raise ValueError('Stored reviewed measurement calibration changed; reanalyze before export.')


def reviewed_measurements_report(summary, identity):
    """Only the analysis snapshot supplies rows; current editor state is not read."""
    if summary is None:
        return None
    _validate_bindings(summary, identity)
    rows = []
    for item in summary.observations:
        pixel = item.pixel_measurement
        scaled = item.calibrated_measurement
        # Legacy summaries may record only their primary bundle. Its units are
        # retained, while absent provenance stays explicitly unknown.
        if pixel is None and item.measurement is not None and item.measurement.length_unit == 'px':
            pixel = item.measurement
        if scaled is None and item.measurement is not None and item.measurement.length_unit == 'mm':
            scaled = item.measurement
        rows.append({'seed_id': item.seed_id, 'annotation': asdict(item.annotation),
            'shape_eligible': item.eligible, 'shape_exclusion_reason': item.exclusion_reason,
            'length_eligible': item.length_eligible,
            'calibration_state': item.calibration_state,
            'calibration_reason': item.calibration_reason,
            'pixel_bundle': _bundle(pixel), 'scaled_bundle': _bundle(scaled),
            'primary_bundle': _bundle(item.measurement)})
    report = {'schema_version': 1,
        'measurement_basis': 'stored reviewed observed-mask geometry',
        'scaled_value_semantics': 'calibration-derived 2-D estimates; seed-plane applicability unvalidated',
        'physical_measurements_eligible': False, 'scientifically_validated': False,
        'lineage': asdict(summary.lineage) if summary.lineage is not None else None,
        'lineage_state': 'recorded' if summary.lineage is not None else 'legacy_not_recorded',
        'measurement_names': list(summary.measurement_names),
        'eligible_count': summary.eligible_count, 'calibrated_count': summary.calibrated_count,
        'primary_length_unit': summary.length_unit,
        'calibration_state': summary.calibration_state, 'calibration_reason': summary.calibration_reason,
        'size_statistics': summary.size_statistics,
        'size_statistics_unavailable_reason': summary.size_statistics_unavailable_reason,
        'boundary_turn_degrees': summary.boundary_turn_degrees,
        'boundary_curvature_times_diameter': summary.boundary_curvature_times_diameter,
        'observations': rows}
    # A detached, finite JSON snapshot is shared by revision identity and export.
    return json.loads(json.dumps(report, allow_nan=False))
