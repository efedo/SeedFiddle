"""Producer evidence for native instance branches, separate from result selection."""
from hashlib import sha256
import json

import numpy as np


CONTRACT = 'seedfiddle.native-backend-outcomes-v1'
BACKENDS = {
    'procedural_instances': 'seedfiddle.native.procedural',
    'unet_instances': 'seedfiddle.native.unet_watershed',
    'stardist_instances': 'seedfiddle.native.stardist',
}
_FIELDS = {'backend_id', 'backend_version', 'mode', 'enabled', 'state',
           'reason_code', 'prediction_facts', 'error_type', 'diagnostic'}
_REASONS = {'available': {None}, 'not_run': {'disabled'},
            'unavailable': {'checkpoint_missing', 'foreground_missing'},
            'failed': {'inference_failed', 'decoder_failed'}}


def _json(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(',', ':'))


def _array_facts(value, name, ndim):
    # Decoders already provide these CPU arrays for display/compact metadata.
    # Do not transfer a GPU raster to manufacture outcome evidence.
    array = np.asarray(value)
    if array.ndim != ndim or array.dtype.kind not in 'biuf' or not np.isfinite(array).all():
        raise ValueError(f'Invalid backend prediction {name}.')
    return {'shape': list(array.shape), 'dtype': array.dtype.str,
            'sha256': sha256(np.ascontiguousarray(array).tobytes()).hexdigest()}


def prediction_facts(result):
    labels = np.asarray(result.labels)
    if labels.ndim != 2 or labels.dtype.kind not in 'iu' or np.any(labels < 0):
        raise ValueError('Backend labels require a nonnegative integer raster.')
    identifiers = np.unique(labels[labels > 0])
    return {'instance_count': int(len(identifiers)),
            'output_state': 'nonempty' if len(identifiers) else 'empty',
            'source_shape': list(getattr(result, 'source_shape', labels.shape)),
            'labels': _array_facts(labels, 'labels', 2),
            'centres_xy': _array_facts(result.centres_xy, 'centres_xy', 2),
            'instance_confidences': _array_facts(result.instance_confidences, 'instance_confidences', 1)}


def capture_native_outcomes(*, enabled, results, failures=None, foreground_available=True):
    """Freeze states observed at the producers; missing objects never imply failure."""
    from seedvision import __version__
    failures = failures or {}
    records = {}
    for node, backend in BACKENDS.items():
        active = node in enabled
        result = results.get(node)
        state, reason, error_type, diagnostic = 'not_run', 'disabled', None, None
        facts = None
        if active:
            if node in failures:
                state, reason, error_type, diagnostic = failures[node]
            elif node == 'procedural_instances' and not foreground_available:
                state, reason = 'unavailable', 'foreground_missing'
            elif result is not None:
                state, reason, facts = 'available', None, prediction_facts(result)
            else:
                raise ValueError('An enabled backend requires explicit producer evidence.')
        records[node] = {'backend_id': backend, 'backend_version': __version__,
            'mode': 'native_compatible', 'enabled': active, 'state': state,
            'reason_code': reason, 'prediction_facts': facts,
            'error_type': error_type, 'diagnostic': diagnostic}
    captured = _json({'contract': CONTRACT, 'outcomes': records})
    read_native_outcomes(captured)
    return captured


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate backend outcome key.')
        result[key] = value
    return result


def read_native_outcomes(captured):
    """Read a bounded strict ledger; legacy absence stays explicitly unknown."""
    if captured is None:
        return {'contract': CONTRACT, 'evidence_state': 'not_recorded', 'outcomes': None}
    if not isinstance(captured, str) or len(captured.encode('utf-8')) > 1024 * 1024:
        raise ValueError('Invalid backend outcome capture.')
    try:
        payload = json.loads(captured, object_pairs_hook=_unique,
                             parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError('Invalid backend outcome capture.') from error
    if not isinstance(payload, dict) or set(payload) != {'contract', 'outcomes'} or payload['contract'] != CONTRACT:
        raise ValueError('Unsupported backend outcome contract.')
    records = payload['outcomes']
    if not isinstance(records, dict) or set(records) != set(BACKENDS):
        raise ValueError('Native backend inventory is incomplete.')
    for node, record in records.items():
        if not isinstance(record, dict) or set(record) != _FIELDS:
            raise ValueError('Invalid backend outcome fields.')
        if (record['backend_id'] != BACKENDS[node] or record['mode'] != 'native_compatible'
                or not isinstance(record['backend_version'], str) or not record['backend_version']
                or type(record['enabled']) is not bool):
            raise ValueError('Invalid native backend identity.')
        state, reason = record['state'], record['reason_code']
        if (not isinstance(state, str) or state not in _REASONS
                or (reason is not None and not isinstance(reason, str)) or reason not in _REASONS[state]):
            raise ValueError('Invalid backend outcome state/reason.')
        if record['enabled'] != (state != 'not_run'):
            raise ValueError('Backend enablement disagrees with producer evidence.')
        if state == 'unavailable' and ((node == 'procedural_instances') != (reason == 'foreground_missing')):
            raise ValueError('Backend prerequisite does not match its producer.')
        if state == 'failed' and node == 'procedural_instances':
            raise ValueError('Procedural failures are not represented by a completed analysis.')
        for field in ('error_type', 'diagnostic'):
            if record[field] is not None and (not isinstance(record[field], str) or not record[field]):
                raise ValueError('Invalid backend diagnostic.')
        if (state == 'failed') != (record['error_type'] is not None):
            raise ValueError('Failed execution requires an explicit error type.')
        if state != 'failed' and record['diagnostic'] is not None:
            raise ValueError('Only attempted failures carry exception diagnostics.')
        facts = record['prediction_facts']
        if state != 'available':
            if facts is not None:
                raise ValueError('Unavailable backends cannot carry a prediction.')
            continue
        if not isinstance(facts, dict) or set(facts) != {
                'instance_count', 'output_state', 'source_shape', 'labels', 'centres_xy', 'instance_confidences'}:
            raise ValueError('Available backend prediction facts are incomplete.')
        count = facts['instance_count']
        if type(count) is not int or count < 0 or facts['output_state'] != ('nonempty' if count else 'empty'):
            raise ValueError('Invalid backend instance count/state.')
        shape = facts['source_shape']
        if not isinstance(shape, list) or len(shape) != 2 or any(type(x) is not int or x <= 0 for x in shape):
            raise ValueError('Invalid backend source frame.')
        for field, ndim in (('labels', 2), ('centres_xy', 2), ('instance_confidences', 1)):
            value = facts[field]
            if (not isinstance(value, dict) or set(value) != {'shape', 'dtype', 'sha256'}
                    or not isinstance(value['shape'], list) or len(value['shape']) != ndim
                    or any(type(x) is not int or x < 0 for x in value['shape'])
                    or not isinstance(value['dtype'], str)
                    or not isinstance(value['sha256'], str) or len(value['sha256']) != 64
                    or any(c not in '0123456789abcdef' for c in value['sha256'])):
                raise ValueError('Invalid backend array facts.')
            try:
                dtype = np.dtype(value['dtype'])
            except (TypeError, ValueError) as error:
                raise ValueError('Invalid backend array dtype.') from error
            if dtype.kind not in ('iu' if field == 'labels' else 'biuf'):
                raise ValueError('Invalid backend array dtype.')
        if facts['centres_xy']['shape'][1] != 2 or any(x <= 0 for x in facts['labels']['shape']):
            raise ValueError('Invalid backend coordinate frame.')
    return dict(payload, evidence_state='recorded')


def verified_native_outcomes(captured, analysis):
    payload = read_native_outcomes(captured)
    if payload['outcomes'] is None:
        return payload
    foreground = getattr(getattr(analysis, 'layers', None), 'foreground_colour_profile', None) is not None
    for node, record in payload['outcomes'].items():
        result = getattr(analysis, node, None)
        if record['state'] == 'available':
            if result is None or prediction_facts(result) != record['prediction_facts']:
                raise ValueError('Backend prediction changed after analysis. Recompute before export.')
            if node == 'procedural_instances' and not foreground:
                raise ValueError('Backend foreground prerequisite changed after analysis.')
        elif node == 'procedural_instances' and record['state'] == 'unavailable':
            if foreground:
                raise ValueError('Backend foreground prerequisite changed after analysis.')
        elif result is not None:
            raise ValueError('Backend output disagrees with captured execution state.')
    return payload


def semantic_native_outcomes(payload):
    if payload['outcomes'] is None:
        return payload
    return dict(payload, outcomes={node: {key: value for key, value in record.items() if key != 'diagnostic'}
                                   for node, record in payload['outcomes'].items()})


def outcome_descriptions(payload):
    """Plain-language summaries shared by Qt review and comparison."""
    if payload['outcomes'] is None:
        return ['Backend execution evidence was not recorded.']
    names = {'procedural_instances': 'Procedural', 'unet_instances': 'U-Net', 'stardist_instances': 'StarDist'}
    lines = []
    for node in BACKENDS:
        record = payload['outcomes'][node]
        state = record['state']
        detail = (f"{record['prediction_facts']['instance_count']} proposed instances"
                  if state == 'available' else str(record['reason_code']).replace('_', ' '))
        if record['diagnostic']:
            detail += ': ' + record['diagnostic']
        lines.append(f'{names[node]}: {state.replace("_", " ")} — {detail}')
    return lines
