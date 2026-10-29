"""Captured native artifact selections and content-based result identity.

Local paths remain execution evidence. Only a designated selection whose bytes
and producer correspondence are verified may be projected to a semantic token.
"""
import json
from pathlib import Path
import re

from seedvision.learning.data import file_sha256


CONTRACT = 'seedfiddle.native-artifact-selections-v1'
SLOTS = {'unet_instances': 'checkpoint_path', 'stardist_instances': 'checkpoint_path',
         'cellpose_sam_instances': 'pretrained_model', 'trait_classifier': 'trait_model_path'}


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _selection(root, node, selected, enabled=True):
    if not isinstance(selected, str) or not selected.strip():
        raise ValueError('Native artifact selection must be a nonempty string.')
    if node == 'trait_classifier':
        from seedvision.learning.trait_pipeline import TraitClassifierSettings, _model
        model, path, digest, reason = _model(TraitClassifierSettings(selected), root)
        return dict(resolved_path=path, source_sha256=digest,
                    model_sha256=None if model is None else model.model_sha256,
                    state=reason or 'valid_frozen_model')
    if node == 'cellpose_sam_instances':
        from seedvision.learning.cellpose_sam import resolve_weights, weights_sha256
        # A model name resolves into Cellpose's weights cache. The weights are
        # about 1.2 GB, so a disabled node records the location unhashed.
        path = resolve_weights(selected, root)
        if not enabled:
            return dict(resolved_path=str(path), source_sha256=None, model_sha256=None,
                        state='disabled_unhashed')
        return dict(resolved_path=str(path), source_sha256=weights_sha256(path) if path.is_file() else None,
                    model_sha256=None, state='bytes_recorded_format_unproven' if path.is_file() else 'file_missing')
    path = Path(selected).expanduser()
    path = path.resolve() if path.is_absolute() else (Path(root) / path).resolve()
    return dict(resolved_path=str(path), source_sha256=file_sha256(path) if path.is_file() else None,
                model_sha256=None, state='bytes_recorded_format_unproven' if path.is_file() else 'file_missing')


def capture_artifact_selections(root, *, selections, enabled, trait_result=None, checkpoints=None):
    """Capture actual selections after inference, without moving CUDA tensors."""
    records = {}
    checkpoints = checkpoints or {}
    for node, selected in selections.items():
        if node not in SLOTS:
            raise ValueError('Unsupported native artifact slot.')
        record = dict(selection=selected, enabled=node in enabled, used=False, producer_id=None,
                      **_selection(root, node, selected, node in enabled))
        if node == 'trait_classifier' and trait_result is not None:
            report = trait_result.report
            if (record['resolved_path'] != report['model_path']
                    or record['source_sha256'] != report['model_artifact_sha256']
                    or record['model_sha256'] != report['model_sha256']
                    or (None if record['state'] == 'valid_frozen_model' else record['state'])
                       != report['model_unavailability_reason']):
                raise ValueError('Trait artifact changed during inference.')
            record['used'] = True
        elif node in checkpoints:
            identifier = checkpoints[node]
            if '#sha256=' not in identifier:
                record['producer_id'] = identifier
                record['used'] = True
            elif (record['source_sha256'] is None or identifier !=
                    record['resolved_path'] + '#sha256=' + record['source_sha256']):
                raise ValueError('Checkpoint changed during inference.')
            else:
                record['used'] = True
                record['state'] = 'used_checkpoint_bytes_verified'
        if record['used'] and not record['enabled']:
            raise ValueError('Disabled native artifact cannot have a producer result.')
        records[node] = record
    return _canonical({'contract': CONTRACT, 'records': records})


def verified_artifact_facts(captured):
    """Recheck recorded locations and states; never silently refresh stale facts."""
    try:
        payload = json.loads(captured) if isinstance(captured, str) else None
        if (not isinstance(payload, dict) or set(payload) != {'contract', 'records'}
                or payload['contract'] != CONTRACT or not isinstance(payload['records'], dict)):
            raise ValueError('Malformed captured native artifact selections.')
        for node, record in payload['records'].items():
            if (node not in SLOTS or not isinstance(record, dict)
                    or set(record) != {'selection', 'enabled', 'used', 'resolved_path', 'producer_id',
                                      'source_sha256', 'model_sha256', 'state'}
                    or type(record['enabled']) is not bool or type(record['used']) is not bool
                    or record['used'] and not record['enabled']
                    or not isinstance(record['selection'], str) or not record['selection'].strip()
                    or record['resolved_path'] is not None and not isinstance(record['resolved_path'], str)
                    or record['producer_id'] is not None and (node == 'trait_classifier'
                       or not record['used'] or not isinstance(record['producer_id'], str)
                       or '#sha256=' in record['producer_id'])
                    or any(value is not None and (not isinstance(value, str)
                           or re.fullmatch('[0-9a-f]{64}', value) is None)
                           for value in (record['source_sha256'], record['model_sha256']))):
                raise ValueError('Malformed captured native artifact selections.')
            # Absolute recorded locations make verification independent of cwd.
            selected = record['resolved_path'] or record['selection']
            current = _selection(Path.cwd(), node, selected, record['enabled'])
            expected_state = ('bytes_recorded_format_unproven'
                              if record['state'] == 'used_checkpoint_bytes_verified' else record['state'])
            if (current['resolved_path'] != record['resolved_path']
                    or current['source_sha256'] != record['source_sha256']
                    or current['model_sha256'] != record['model_sha256']
                    or current['state'] != expected_state
                    or node != 'trait_classifier' and record['producer_id'] is None and record['used'] !=
                       (record['state'] == 'used_checkpoint_bytes_verified')):
                raise ValueError('Native artifact selection changed after analysis. Recompute before export.')
        return payload['records']
    except (TypeError, KeyError, json.JSONDecodeError) as error:
        raise ValueError('Malformed captured native artifact selections.') from error


def semantic_artifact(record, node):
    result = {key: record[key] for key in ('enabled', 'used', 'state', 'source_sha256', 'model_sha256')}
    result['node'] = node
    result['contract'] = CONTRACT
    if record['producer_id'] is not None:
        result['opaque_producer_id'] = record['producer_id']
    if record['source_sha256'] is None and record['state'] != 'no_frozen_model':
        # Missing/unreadable/oversized selections have no proven content identity.
        result['opaque_selection'] = record['selection']
    return result


def semantic_settings(settings, records):
    """Project exact recognized selections; arbitrary caller metadata stays opaque."""
    detached = json.loads(_canonical(settings))
    if not isinstance(detached, dict) or not isinstance(detached.get('nodes'), list):
        return detached
    for row in detached['nodes']:
        if not isinstance(row, dict) or row.get('id') not in records:
            continue
        node = row['id']
        parameters = row.get('parameters')
        if not isinstance(parameters, dict) or SLOTS[node] not in parameters:
            continue
        actual = parameters[SLOTS[node]]
        token = semantic_artifact(records[node], node)
        if actual != records[node]['selection'] and actual != token:
            raise ValueError('Native artifact settings differ from the captured selection. Recompute before export.')
        parameters[SLOTS[node]] = token
    return detached


def semantic_trait_report(report, records):
    if report is None:
        if records.get('trait_classifier', {}).get('used'):
            raise ValueError('Captured trait producer is missing from the result.')
        return None
    detached = json.loads(_canonical(report))
    record = records.get('trait_classifier')
    if record and record['used'] and record['producer_id'] is None:
        if (report['model_path'] != record['resolved_path']
                or report['model_artifact_sha256'] != record['source_sha256']):
            raise ValueError('Trait producer differs from the captured artifact.')
        if report['model_sha256'] != record['model_sha256']:
            raise ValueError('Trait producer differs from the captured artifact.')
        if record['source_sha256'] is not None or record['state'] == 'no_frozen_model':
            detached.pop('model_path')
    return detached


def semantic_checkpoint(identifier, node, records):
    record = records.get(node)
    if record and record['used'] and record['producer_id'] is None:
        if identifier != record['resolved_path'] + '#sha256=' + record['source_sha256']:
            raise ValueError('Checkpoint producer differs from the captured artifact.')
        return semantic_artifact(record, node)
    if record and record['producer_id'] is not None and identifier != record['producer_id']:
        raise ValueError('Opaque checkpoint producer differs from the captured result.')
    return identifier


def execution_settings(settings, records):
    """Retain executable path strings even when rebuilding a semantic report."""
    detached = json.loads(_canonical(settings))
    if isinstance(detached, dict) and isinstance(detached.get('nodes'), list):
        for row in detached['nodes']:
            if not isinstance(row, dict) or row.get('id') not in records:
                continue
            node = row['id']
            parameters = row.get('parameters', {})
            if isinstance(parameters, dict) and parameters.get(SLOTS[node]) == semantic_artifact(records[node], node):
                parameters[SLOTS[node]] = records[node]['selection']
    return detached
