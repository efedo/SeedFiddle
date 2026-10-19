"""Portable recorded recipe QC, distinct from seed or scientific approval.

Native records derive from fully verified output archives. Selected-stage
records capture an inspection already performed by the desktop; loading those
records checks their consistency, without claiming fresh artifact verification.
No raster is copied or retained by this module.
"""
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import re
from uuid import uuid4

import numpy as np

from seedvision.benchmarking.checkpoint_inspection import CheckpointInspection, InspectionStatus
from seedvision.benchmarking.checkpoints import CheckpointLimits, CoordinateFrame, RasterRole
from seedvision.benchmarking.recipes import RecipeSpec, StageRunRecord, StageRunState, validate_stage_run


FORMAT = 'seedfiddle.recipe-method-qc'
VERSION = 1
FILENAME = 'method-qc.json'
LIMIT = 4 * 1024 * 1024
ACK_LIMIT = 64 * 1024
ACK_CONTRACT = 'seedfiddle.recipe-qc-acknowledgment-v1'
_SHA = re.compile(r'[0-9a-f]{64}\Z')
_NATIVE_STATES = {'provided', 'executed', 'empty', 'partial', 'failed', 'not_run', 'unavailable', 'bypassed'}


def _plain(value, depth=0):
    if depth > 64:
        raise ValueError('Recipe QC metadata exceeds its depth bound.')
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise ValueError('Recipe QC fields must be text.')
        return {key: _plain(item, depth+1) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item, depth+1) for item in value]
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise ValueError('Recipe QC accepts compact JSON metadata only.')


def _bytes(value, limit=LIMIT):
    try:
        raw = json.dumps(_plain(value), ensure_ascii=False, sort_keys=True,
                         separators=(',', ':'), allow_nan=False).encode('utf-8')
    except (TypeError, UnicodeError, RecursionError) as error:
        raise ValueError('Recipe QC is not strict JSON.') from error
    if len(raw) > limit:
        raise ValueError('Recipe QC exceeds its byte bound.')
    return raw


def _exact(value, fields, name):
    if type(value) is not dict or set(value) != set(fields.split()):
        raise ValueError(f'Invalid {name} fields.')


def _digest(value):
    if type(value) is not str or not _SHA.fullmatch(value):
        raise ValueError('Invalid recipe QC SHA-256.')


def _read(path, limit):
    with Path(path).open('rb') as stream:
        raw = stream.read(limit+1)
    if len(raw) > limit:
        raise ValueError('Recipe QC record exceeds its byte bound.')
    def pairs(rows):
        result = {}
        for key, value in rows:
            if key in result:
                raise ValueError('Duplicate recipe QC field.')
            result[key] = value
        return result
    try:
        return json.loads(raw, object_pairs_hook=pairs,
            parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)))
    except (UnicodeError, RecursionError) as error:
        raise ValueError('Recipe QC record is not bounded UTF-8 JSON.') from error


@dataclass(frozen=True)
class MethodQC:
    """Immutable captured QC. Payload access returns a detached validated copy."""
    payload_json: str

    def __post_init__(self):
        if type(self.payload_json) is not str or len(self.payload_json.encode('utf-8')) > LIMIT:
            raise ValueError('Recipe QC exceeds its byte bound.')
        # Use the same duplicate-key parser for in-memory and persisted records.
        def pairs(rows):
            result = {}
            for key, value in rows:
                if key in result:
                    raise ValueError('Duplicate recipe QC field.')
                result[key] = value
            return result
        try:
            payload = json.loads(self.payload_json, object_pairs_hook=pairs,
                parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)))
        except (UnicodeError, RecursionError) as error:
            raise ValueError('Recipe QC record is not bounded UTF-8 JSON.') from error
        _validate(payload)
        object.__setattr__(self, 'payload_json', _bytes(payload).decode('utf-8'))

    @property
    def payload(self):
        return json.loads(self.payload_json)

    @property
    def revision(self):
        return self.payload['qc_revision']


def _finish(recipe, identity):
    payload = {'format': FORMAT, 'version': VERSION, 'recipe': recipe.to_dict(),
               'identity': identity, 'qc_revision': sha256(_bytes(identity)).hexdigest()}
    return MethodQC(_bytes(payload).decode('utf-8'))


def _base(recipe, scope):
    return {'scope': scope, 'recipe_mode': recipe.provenance.mode.value,
        'recipe_identity': recipe.identity.analytical_dict(),
        'native_artifact': None, 'stage_record': None, 'recorded_inspection': None,
        'node_outcomes': [], 'measurement_eligibility': None, 'scientific_validity': None}


def _native_executor(recipe):
    # The common archive writer accepts both executor result types. Its own
    # module name is not the calculation identity, even for historical recipes
    # whose declared backend predates their additive native working copies.
    from seedvision.benchmarking.native_correction_recipe import RECIPE_ID
    return ('seedvision.benchmarking.native_correction_recipe' if recipe.recipe_id == RECIPE_ID
            else 'seedvision.benchmarking.native_recipe')


def _qc_from_verified_native_report(report):
    """Internal: caller has already verified all report and stored product bytes."""
    recipe = RecipeSpec.from_json(_bytes(report['recipe']))
    identity = _base(recipe, 'whole_native_archive')
    identity['native_artifact'] = {'format': report['format'], 'version': report['version'],
        'result_revision': report['result_revision'], 'source_sha256': report['source_sha256'],
        'coordinate_frame': report['coordinate_frame'], 'runtime': report['runtime'],
        'preparation': report['preparation'], 'status': report['status'],
        'execution_backend': _native_executor(recipe),
        'execution_mode': 'paper_reconstruction' if recipe.provenance.mode.value == 'paper_reconstruction'
                          else 'native_compatible'}
    identity['node_outcomes'] = [{'stage_id': node['stage_id'], 'operation_id': node['operation_id'],
        'state': node['status'], 'reason': node['reason'],
        'output_ports': [item[0] for item in node['outputs']]} for node in report['nodes']]
    public = [node['stage_id'] for node in report['nodes'] if node.get('backend') == 'plantcv']
    if public:
        identity['native_artifact'].update(execution_backend='seedvision.pipeline.backend_execution',
            execution_mode='plantcv_native', public_stages=public)
    return _finish(recipe, identity)


def qc_from_native_archive(directory, *, limits=CheckpointLimits(), cancel=None):
    """Fully verify every stored array, retaining none, then derive recorded QC."""
    from seedvision.export.native_results import load_native_recipe_result
    report, _ = load_native_recipe_result(directory, array_names=(), limits=limits, cancel=cancel)
    return _qc_from_verified_native_report(report)


def qc_from_stage_inspection(recipe, record, inspection):
    """Capture the selected record and observed inspection, without running analysis."""
    if not isinstance(recipe, RecipeSpec) or not isinstance(record, StageRunRecord):
        raise TypeError('Recipe QC requires its exact recipe and selected stage record.')
    if not isinstance(inspection, CheckpointInspection):
        raise TypeError('Recipe QC requires the captured checkpoint inspection.')
    validate_stage_run(recipe, record)
    if inspection.status is InspectionStatus.LOADING:
        raise ValueError('An inspection still loading has no terminal QC record.')
    identity = _base(recipe, 'selected_stage_record')
    stored = record.to_dict()
    stored['recipe_identity'] = recipe.identity.analytical_dict()
    identity['stage_record'] = stored
    verified = inspection.status is InspectionStatus.VERIFIED
    identity['recorded_inspection'] = {'status': inspection.status.value, 'detail': inspection.detail,
        'checkpoint_sha256': inspection.checkpoint_sha256,
        'source_sha256': inspection.source_sha256,
        'coordinate_frame': None if inspection.coordinate_frame is None else inspection.coordinate_frame.to_dict(),
        'execution_status': None if inspection.execution_status is None else inspection.execution_status.value,
        'array_inventory': _plain(inspection.array_inventory) if verified else None,
        'compact_metadata': _plain(inspection.compact_metadata) if verified else None}
    identity['node_outcomes'] = [{'stage_id': record.stage_id,
        'operation_id': recipe.stage(record.stage_id).operation_id,
        'state': record.state.value, 'reason': record.reason, 'output_ports': None}]
    return _finish(recipe, identity)


def _validate(payload):
    _exact(payload, 'format version recipe identity qc_revision', 'recipe QC')
    if payload['format'] != FORMAT or type(payload['version']) is not int or payload['version'] not in (VERSION, 2, 3, 4, 5):
        raise ValueError('Unsupported recipe QC format.')
    if payload['version'] == 2:
        from seedvision.export.recipe_attempt_qc import validate_recipe_attempt
        validate_recipe_attempt(payload)
        return
    if payload['version'] == 3:
        from seedvision.export.recipe_batch_qc import validate_recipe_batch
        validate_recipe_batch(payload)
        return
    if payload['version'] in (4, 5):
        from seedvision.export.reference_capture_qc import validate_reference_capture_qc
        validate_reference_capture_qc(payload)
        return
    recipe = RecipeSpec.from_json(_bytes(payload['recipe']))
    value = payload['identity']
    _exact(value, 'scope recipe_mode recipe_identity native_artifact stage_record recorded_inspection '
                  'node_outcomes measurement_eligibility scientific_validity', 'recipe QC identity')
    if (value['recipe_identity'] != recipe.identity.analytical_dict()
            or value['recipe_mode'] != recipe.provenance.mode.value
            or value['measurement_eligibility'] is not None or value['scientific_validity'] is not None):
        raise ValueError('Recipe QC identity or eligibility claims contradict its contract.')
    _digest(payload['qc_revision'])
    if payload['qc_revision'] != sha256(_bytes(value)).hexdigest():
        raise ValueError('Recipe QC revision differs from its recorded facts.')
    rows = value['node_outcomes']
    if type(rows) is not list:
        raise ValueError('Recipe QC outcomes must be ordered records.')
    if value['scope'] == 'whole_native_archive':
        if value['stage_record'] is not None or value['recorded_inspection'] is not None:
            raise ValueError('Native QC cannot claim a selected-stage inspection.')
        artifact = value['native_artifact']
        public = artifact.get('public_stages')
        _exact(artifact, 'format version result_revision source_sha256 coordinate_frame runtime preparation '
                         'status execution_backend execution_mode' + (' public_stages' if public is not None else ''), 'native QC artifact')
        if public is not None:
            from seedvision.pipeline.backend_execution import backend_capability
            if (type(public) is not list or not public or len(public) != len(set(public))
                    or any(type(identifier) is not str or identifier not in {stage.stage_id for stage in recipe.stages}
                           or 'plantcv' not in backend_capability(recipe, identifier).backends for identifier in public)):
                raise ValueError('Public QC stages differ from the admitted recipe operations.')
        expected_executor = 'seedvision.pipeline.backend_execution' if public else _native_executor(recipe)
        expected_mode = 'plantcv_native' if public else ('paper_reconstruction' if value['recipe_mode'] == 'paper_reconstruction' else 'native_compatible')
        if (artifact['format'] != 'seedfiddle.native-analysis-result'
                or type(artifact['version']) is not int or artifact['version'] not in range(1, 6)
                or artifact['execution_backend'] != expected_executor
                or artifact['execution_mode'] != expected_mode):
            raise ValueError('Native QC execution route is inconsistent.')
        _digest(artifact['result_revision']); _digest(artifact['source_sha256'])
        CoordinateFrame.from_dict(artifact['coordinate_frame'])
        for name in ('runtime', 'preparation'):
            if (type(artifact[name]) is not dict or not artifact[name]
                    or any(type(k) is not str or not k.strip() or type(v) is not str or not v.strip()
                           for k, v in artifact[name].items())):
                raise ValueError('Native QC preparation/runtime facts must be text mappings.')
        stages = recipe.stages
        if len(rows) != len(stages):
            raise ValueError('Native QC does not contain the exact ordered stages.')
    elif value['scope'] == 'selected_stage_record':
        if value['native_artifact'] is not None:
            raise ValueError('Selected-stage QC cannot claim a native whole-run archive.')
        stored = value['stage_record']
        _exact(stored, 'stage_id recipe_identity state checkpoint_sha256 upstream_checkpoint_sha256 reason', 'QC stage record')
        if stored['recipe_identity'] != recipe.identity.analytical_dict():
            raise ValueError('Selected-stage QC recipe identity is stale.')
        record = StageRunRecord.from_dict(stored | {'recipe_identity': recipe.identity.to_dict()})
        validate_stage_run(recipe, record)
        inspection = value['recorded_inspection']
        _exact(inspection, 'status detail checkpoint_sha256 source_sha256 coordinate_frame execution_status '
                           'array_inventory compact_metadata', 'recorded QC inspection')
        status = InspectionStatus(inspection['status'])
        if status is InspectionStatus.LOADING or type(inspection['detail']) is not str or not inspection['detail'].strip():
            raise ValueError('Recipe QC requires a terminal inspection with a detail.')
        if status is InspectionStatus.VERIFIED:
            if (record.state not in (StageRunState.EXECUTED, StageRunState.EMPTY)
                    or inspection['checkpoint_sha256'] != record.checkpoint_sha256
                    or inspection['execution_status'] != ('empty' if record.state is StageRunState.EMPTY else 'available')
                    or type(inspection['array_inventory']) is not list or not inspection['array_inventory']
                    or type(inspection['compact_metadata']) is not dict):
                raise ValueError('Recorded QC verification contradicts its selected stage.')
            _digest(inspection['source_sha256'])
            CoordinateFrame.from_dict(inspection['coordinate_frame'])
            inventory = inspection['array_inventory']
            if len(inventory) > 16:
                raise ValueError('Recorded QC inspection inventory exceeds its bound.')
            seen = set()
            for item in inventory:
                _exact(item, 'artifact name dtype shape nbytes role', 'QC inspected array')
                if (item['artifact'] not in ('upstream', 'output') or type(item['name']) is not str
                        or not item['name'] or (item['artifact'], item['name']) in seen
                        or type(item['dtype']) is not str or type(item['shape']) is not list
                        or not 1 <= len(item['shape']) <= 8
                        or any(type(n) is not int or not 0 <= n <= 2**31 for n in item['shape'])
                        or type(item['nbytes']) is not int or item['nbytes'] < 0):
                    raise ValueError('Invalid recorded QC array descriptor.')
                try:
                    dtype = np.dtype(item['dtype'])
                except (TypeError, ValueError) as error:
                    raise ValueError('Invalid recorded QC array dtype.') from error
                count = 1
                for n in item['shape']:
                    count *= n
                if dtype.kind not in 'biuf' or count*dtype.itemsize != item['nbytes']:
                    raise ValueError('Recorded QC array bytes contradict its descriptor.')
                if item['role'] is not None:
                    RasterRole.from_dict(item['role'])
                seen.add((item['artifact'], item['name']))
        elif any(inspection[key] is not None for key in ('checkpoint_sha256', 'source_sha256',
                'coordinate_frame', 'execution_status', 'array_inventory', 'compact_metadata')):
            raise ValueError('Unverified QC cannot assert observed artifact facts.')
        stages = (recipe.stage(record.stage_id),)
        if (len(rows) != 1 or type(rows[0]) is not dict
                or rows[0].get('state') != record.state.value or rows[0].get('reason') != record.reason):
            raise ValueError('Selected-stage QC outcomes contradict the recorded execution.')
    else:
        raise ValueError('Unknown recipe QC scope.')
    for stage, row in zip(stages, rows):
        _exact(row, 'stage_id operation_id state reason output_ports', 'QC node outcome')
        if row['stage_id'] != stage.stage_id or row['operation_id'] != stage.operation_id or type(row['reason']) is not str:
            raise ValueError('Recipe QC stage order or operation differs from its recipe.')
        ports = row['output_ports']
        if value['scope'] == 'selected_stage_record':
            if ports is not None:
                raise ValueError('A stage record cannot infer its produced output ports.')
        else:
            declared = {port.name for port in stage.ports if port.direction == 'output'}
            if row['state'] == 'bypassed':
                from seedvision.pipeline.workflows import bypass_ports
                if (stage.condition is None or stage.condition.expression != 'workspace.disabled'
                        or stage.condition.outcome_when_true != 'bypass' or not bypass_ports(stage)):
                    raise ValueError('Native bypass requires its registered typed working condition.')
            if (row['state'] not in _NATIVE_STATES or type(ports) is not list
                    or any(type(port) is not str for port in ports) or len(set(ports)) != len(ports)
                    or set(ports)-declared
                    or row['state'] in {'failed', 'not_run', 'unavailable'} and ports
                    or row['state'] in {'provided', 'executed', 'empty', 'bypassed'} and set(ports) != declared):
                raise ValueError('Native QC states or output ports contradict the recipe.')
    if value['scope'] == 'whole_native_archive' and value['native_artifact']['status'] != (
            'executed' if all(row['state'] in {'provided', 'executed', 'empty', 'bypassed'} for row in rows) else 'partial'):
        raise ValueError('Native QC run state contradicts its node outcomes.')


def load_method_qc(path, *, native_archive=None, limits=CheckpointLimits(), cancel=None):
    """Load captured facts; optional native archive binding performs full fresh verification."""
    quality = MethodQC(_bytes(_read(path, LIMIT)).decode('utf-8'))
    if native_archive is not None:
        expected = qc_from_native_archive(native_archive, limits=limits, cancel=cancel)
        if quality.payload['identity'] != expected.payload['identity']:
            raise ValueError('Recipe QC differs from the fully verified native archive.')
    return quality


def _publish(path, payload, limit):
    raw = _bytes(payload, limit)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if _bytes(_read(destination, limit), limit) == raw:
            return destination
        raise FileExistsError('Conflicting recipe QC evidence already exists at this destination.')
    temporary = destination.with_name('.'+destination.name+'-'+uuid4().hex+'.incomplete')
    try:
        with temporary.open('xb') as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        temporary.rename(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def export_method_qc(quality, path):
    if not isinstance(quality, MethodQC):
        raise TypeError('Export requires validated recipe QC.')
    return _publish(path, quality.payload, LIMIT)


def _validate_ack(payload, quality):
    _exact(payload, 'contract qc_revision decision operator_account reviewed_at_utc', 'recipe QC acknowledgment')
    if (payload['contract'] != ACK_CONTRACT or payload['qc_revision'] != quality.revision
            or payload['decision'] != 'acknowledged' or type(payload['operator_account']) is not str
            or not payload['operator_account'].strip() or type(payload['reviewed_at_utc']) is not str):
        raise ValueError('Recipe QC acknowledgment is stale or contains an approval claim.')
    instant = datetime.fromisoformat(payload['reviewed_at_utc'])
    if instant.tzinfo is None or instant.utcoffset() != timezone.utc.utcoffset(instant):
        raise ValueError('Recipe QC acknowledgment requires an explicit UTC timestamp.')
    return payload


def acknowledge_method_qc(quality, path, operator, *, reviewed_at_utc=None):
    if not isinstance(quality, MethodQC):
        raise TypeError('Acknowledgment requires validated recipe QC.')
    payload = {'contract': ACK_CONTRACT, 'qc_revision': quality.revision, 'decision': 'acknowledged',
        'operator_account': operator, 'reviewed_at_utc': datetime.now(timezone.utc).isoformat()
            if reviewed_at_utc is None else reviewed_at_utc}
    _validate_ack(payload, quality)
    return _publish(path, payload, ACK_LIMIT)


def load_method_qc_acknowledgment(path, quality):
    if not isinstance(quality, MethodQC):
        raise TypeError('Acknowledgment loading requires its exact validated recipe QC.')
    return _validate_ack(_read(path, ACK_LIMIT), quality)
