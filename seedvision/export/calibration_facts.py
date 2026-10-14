"""Detached calibration facts that bind review revisions without physical claims."""
from dataclasses import asdict, is_dataclass
import math
from numbers import Real

import numpy as np

from seedvision.calibration.image import CalibrationSettings, ColourCardDetection, RulerDetection
from seedvision.calibration.neutral import NeutralBalanceFacts, NeutralPatchWitness, normalize_neutral_patch_selection


def _number(value, name, *, minimum=None, maximum=None, positive=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f'{name} must be a finite number.')
    value = float(value)
    if (not math.isfinite(value) or (positive and value <= 0)
            or (minimum is not None and value < minimum)
            or (maximum is not None and value > maximum)):
        raise ValueError(f'{name} is outside its finite calibration domain.')
    return value


def _detached(value):
    if is_dataclass(value) and not isinstance(value, type):
        return _detached(asdict(value))
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Real):
        numeric = _number(value, 'Calibration fact')
        return int(value) if isinstance(value, int) else numeric
    if isinstance(value, (tuple, list)):
        return [_detached(item) for item in value]
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {key: _detached(item) for key, item in value.items()}
    raise ValueError('Calibration facts require typed finite scalars and compact sequences.')


def calibration_facts(calibration):
    settings = getattr(calibration, 'calibration_settings', None)
    neutral = getattr(calibration, 'neutral_balance', None)
    shape = getattr(calibration, 'source_shape_hw', None)
    complete = settings is not None and neutral is not None and shape is not None
    result = {'state': 'recorded' if complete else 'legacy_unavailable',
        'reason': '' if complete else 'Legacy calibration lacks recorded settings, source shape or neutral witnesses.',
        'length_unit': 'mm', 'scale_unit': 'pixels/mm', 'source_axis_order': 'height,width',
        'seed_valid_region': {'state': 'unknown', 'reason': 'Reference detection does not establish seed-plane applicability.'},
        'physical_measurements_eligible': False, 'scientifically_validated': False}
    matrix = np.asarray(calibration.affine_matrix)
    if matrix.shape != (3, 3) or matrix.dtype.kind not in 'iuf' or not np.isfinite(matrix).all():
        raise ValueError('Calibration transform must be a finite 3x3 numeric matrix.')
    result['source_to_corrected'] = matrix.tolist()
    for name in ('deskew_degrees', 'perspective_strength', 'scale_confidence', 'scale_bar_mm'):
        if hasattr(calibration, name):
            result[name] = _number(getattr(calibration, name), name,
                minimum=0 if name != 'deskew_degrees' else None,
                maximum=1 if name == 'scale_confidence' else None)
    for name in ('pixels_per_mm', 'ruler_tick_spacing_px'):
        if hasattr(calibration, name):
            value = getattr(calibration, name)
            result[name] = None if value is None else _number(value, name, positive=True)
    for name in ('perspective_corrected', 'ruler_in_corrected_coordinates'):
        if hasattr(calibration, name):
            value = getattr(calibration, name)
            if not isinstance(value, bool):
                raise ValueError(f'{name} must be a boolean calibration fact.')
            result[name] = value
    gains = getattr(calibration, 'channel_gains_bgr', None)
    if gains is not None:
        if not isinstance(gains, tuple) or len(gains) != 3:
            raise ValueError('Calibration BGR gains require three recorded values.')
        result['channel_gains_bgr'] = [_number(value, 'Channel gain', positive=True) for value in gains]
    if shape is not None:
        if (not isinstance(shape, tuple) or len(shape) != 2
                or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in shape)):
            raise ValueError('Recorded source shape requires two positive integers.')
        result['source_shape_hw'] = list(shape)
    if settings is not None:
        if not isinstance(settings, CalibrationSettings):
            raise ValueError('Recorded calibration settings must be typed.')
        result['settings'] = _detached(settings)
    if neutral is not None:
        if not isinstance(neutral, NeutralBalanceFacts) or not isinstance(neutral.enabled, bool):
            raise ValueError('Neutral balance facts must be typed.')
        if (neutral.state not in ('applied', 'disabled', 'unavailable')
                or neutral.sample_frame not in ('source', 'corrected')
                or normalize_neutral_patch_selection(neutral.requested_selection) != neutral.requested_selection):
            raise ValueError('Neutral balance state or sampling request is invalid.')
        available, used = neutral.available_patch_ids, neutral.used_patch_ids
        if (not isinstance(available, tuple) or not isinstance(used, tuple)
                or not isinstance(neutral.patches, tuple)
                or any(not isinstance(w, NeutralPatchWitness) for w in neutral.patches)):
            raise ValueError('Neutral patch witnesses require immutable typed sequences.')
        if neutral.selection_mode not in ('none', 'manual', 'automatic_threshold', 'automatic_fallback'):
            raise ValueError('Unknown neutral patch selection mode.')
        if (neutral.enabled == (neutral.state == 'disabled')
                or (neutral.state == 'applied' and neutral.requested_selection != 'automatic'
                    and (used != tuple(map(int, neutral.requested_selection.split(',')))
                         or neutral.selection_mode != 'manual'))):
            raise ValueError('Neutral availability disagrees with enablement or manual selection.')
        for witness in neutral.patches:
            for corners in (witness.source_corners, witness.sample_corners):
                if len(corners) != 4 or any(len(point) != 2 for point in corners):
                    raise ValueError('Neutral patches require four recorded XY corners.')
            for mean in (witness.source_mean_bgr, witness.sampled_mean_bgr):
                if len(mean) != 3:
                    raise ValueError('Neutral patches require three recorded BGR means.')
                for v in mean: _number(v, 'Neutral sample mean', minimum=0, maximum=255)
        if (any(isinstance(v, bool) or not isinstance(v, int) or not 1 <= v <= 24 for v in available+used)
                or len(set(available)) != len(available) or len(set(used)) != len(used)
                or not set(used) <= set(available)
                or tuple(w.patch_id for w in neutral.patches) != used
                or (neutral.state == 'applied') != bool(used)):
            raise ValueError('Neutral patch witnesses do not match the recorded selection.')
        if gains is None or neutral.channel_gains_bgr != gains:
            raise ValueError('Recorded neutral gains disagree with applied calibration gains.')
        if neutral.state != 'applied' and gains != (1., 1., 1.):
            raise ValueError('Disabled or unavailable neutral balance must use identity gains.')
        if settings is not None and (neutral.requested_selection != settings.neutral_patch_selection
                or neutral.sample_frame != settings.neutral_sample_frame
                or neutral.enabled != settings.apply_colour_balance):
            raise ValueError('Neutral facts disagree with recorded calibration settings.')
        result['neutral_balance'] = _detached(neutral)
    for name, expected_type in (('colour_card', ColourCardDetection), ('ruler', RulerDetection)):
        if hasattr(calibration, name):
            value = getattr(calibration, name)
            if value is not None and not isinstance(value, expected_type):
                raise ValueError(f'{name} must be a typed reference detection.')
            result[name] = _detached(value)
    if hasattr(calibration, 'warnings'):
        warnings = calibration.warnings
        if not isinstance(warnings, tuple) or any(not isinstance(v, str) for v in warnings):
            raise ValueError('Calibration warnings must be recorded strings.')
        result['warnings'] = list(warnings)
    return result
