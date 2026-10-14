"""Persist the actual compact runtime model without refitting its source evidence."""
from dataclasses import asdict
import json

import numpy as np

from seedvision.export.reviewed_measurements import reviewed_measurements_report
from seedvision.measurement.lineage import facts_sha256
from seedvision.measurement.shape_model import ShapeComponentUpdate
from seedvision.reference_library.contracts import SHAPE_MEASUREMENT_NAMES, ShapePopulationComponent


def _count(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError(f'{name} must be a non-negative integer.')
    return int(value)


def _number(value, name, *, positive=False, maximum=None):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f'{name} must be numeric.')
    number = float(value)
    if not np.isfinite(number) or number < 0 or (positive and number == 0) or (
        maximum is not None and number > maximum
    ):
        raise ValueError(f'{name} is outside its finite range.')
    return number


def shape_model_report(model, identity, reviewed_summary):
    """Bind a detached model snapshot to its own recorded measurement summary."""
    if model is None:
        return None
    stored_summary = getattr(model, 'measurement_summary', None)
    summary_sha256 = None
    if stored_summary is not None:
        stored_report = reviewed_measurements_report(stored_summary, identity)
        if stored_report != reviewed_summary:
            raise ValueError('Stored shape model measurements changed; reanalyze before export.')
        summary_sha256 = facts_sha256(stored_report)
    if model.source_mode not in {'current_image_only', 'species_library_only',
        'species_library_prior_and_current'}:
        raise ValueError('Stored shape model source mode is unsupported.')
    if not isinstance(model.in_sample_local_update, bool):
        raise ValueError('Stored shape model in-sample state must be Boolean.')
    families = []
    poses = set()
    for family in model.families:
        if family.pose not in {'flat', 'oblique', 'side', 'uncertain'} or family.pose in poses:
            raise ValueError('Stored shape model needs unique supported pose families.')
        poses.add(family.pose)
        component = family.component
        if not isinstance(component, ShapePopulationComponent) or component.pose != family.pose:
            raise ValueError('Stored shape family and component pose disagree.')
        if len(component.mean) != len(SHAPE_MEASUREMENT_NAMES):
            raise ValueError('Stored shape component does not match the measurement schema.')
        # Revalidate potentially mutable nested inputs without changing their
        # covariance: an unavailable update can legitimately retain bad evidence.
        ShapePopulationComponent(**asdict(component))
        _number(component.effective_physical_seed_count, 'Effective component support')
        _number(component.calibrated_physical_seed_count, 'Calibrated component support')
        _number(component.dimensionless_physical_seed_count, 'Dimensionless component support')
        _count(component.source_count, 'Component source count')
        if family.source not in {'local', 'library', 'library+local'}:
            raise ValueError('Stored shape family source is unsupported.')
        if family.update_state is None:
            if family.update_unavailability_reason is not None or family.pooling_method is not None or family.conditioning_scales:
                raise ValueError('Unrecorded shape updates cannot carry known update diagnostics.')
        else:
            ShapeComponentUpdate(component, family.update_state, family.source,
                family.update_unavailability_reason, family.pooling_method, family.conditioning_scales)
        families.append({
            'pose': family.pose, 'component': asdict(component), 'source': family.source,
            'local_observation_count': _count(family.local_observation_count, 'Local observation count'),
            'prior_probability': _number(family.prior_probability, 'Pose prior probability', maximum=1),
            'out_of_family_distance': _number(family.out_of_family_distance, 'Family distance threshold'),
            'update_diagnostic_state': 'legacy_not_recorded' if family.update_state is None else 'recorded',
            'update_state': family.update_state,
            'update_unavailability_reason': family.update_unavailability_reason,
            'pooling_method': family.pooling_method,
            'conditioning_scales': list(family.conditioning_scales),
        })
    report = {
        'schema_version': 1, 'model_basis': 'stored runtime seed dimensions and shape model',
        'scientifically_validated': False, 'physical_measurements_eligible': False,
        'measurement_names': list(SHAPE_MEASUREMENT_NAMES),
        'reviewed_summary_binding_state': 'recorded' if summary_sha256 is not None else 'legacy_not_recorded',
        'reviewed_summary_sha256': summary_sha256,
        'source_mode': model.source_mode,
        'selected_hierarchy_path': list(model.selected_hierarchy_path),
        'legacy_processing_diameter_px': _number(model.legacy_processing_diameter_px, 'Processing diameter', positive=True),
        'compatibility_seed_diameter_px': _number(model.compatibility_seed_diameter_px, 'Compatibility diameter', positive=True),
        'in_sample_local_update': model.in_sample_local_update,
        'warnings': list(model.warnings), 'families': families,
    }
    return json.loads(json.dumps(report, allow_nan=False))
