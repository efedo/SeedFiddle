"""Recorded public capture facts, separate from native output archives.

The existing public-chain reader verifies complete captures. Historical QC
loading checks recorded metadata without executing calls or retaining arrays.
"""
from hashlib import sha256
from uuid import uuid4

from seedvision.benchmarking.checkpoints import CoordinateFrame, _canonical_json, _thaw_json
from seedvision.benchmarking.recipes import RecipeSpec, StageRunRecord, validate_stage_run
from seedvision.benchmarking.reference_capture_session import ReferenceCaptureSession
from seedvision.benchmarking.reference_preprocessing import _chain_stages, _validate_pinned_recipe
from seedvision.export import method_qc as qc, recipe_attempt_qc as attempts

VERSION = 4
EXECUTOR = 'seedvision.benchmarking.reference_capture_session.capture_reference_path'
PUBLIC_BACKEND = 'seedvision.benchmarking.reference_runner.PlantCVReferenceRunner'
_STATES = {'failed', 'rejected', 'unavailable', 'cancelled', 'partial', 'empty'}
_PHASES = {'admission', 'selection', 'capture', 'verification', 'publication', 'between_rows'}


def _same(actual, expected, name):
    if qc._bytes(actual) != qc._bytes(expected):
        raise ValueError(f'Recorded reference {name} differs from its binding.')


def _chain(recipe, record, *, complete):
    qc._exact(record, 'schema status complete recipe_identity source_sha256 coordinate_frame inputs '
        'preprocessing source_calls label_facts runner_request_count_after_preprocessing '
        'verified_public_result_count_after_preprocessing scientifically_validated', 'recorded public chain')
    _validate_pinned_recipe(recipe)
    _same(record['recipe_identity'], recipe.identity.to_dict(), 'recipe')
    qc._digest(record['source_sha256'])
    CoordinateFrame.from_dict(record['coordinate_frame'])
    if (type(record['complete']) is not bool or record['scientifically_validated'] is not False
            or record['schema'] != 'seedfiddle.public-seed-chain.v1'
            or record['status'] not in {'executed', 'empty', 'partial', 'cancelled'}
            or record['complete'] is not complete
            or (record['status'] == 'executed') is not complete):
        raise ValueError('Recorded reference completion/science claims contradict the capture.')
    qc._exact(record['inputs'], 'gray rgb', 'recorded public inputs')
    for name, binding in record['inputs'].items():
        qc._exact(binding, 'checkpoint_sha256 array_name identity measurement role', 'recorded public input')
        qc._digest(binding['checkpoint_sha256'])
        _same(binding['identity']['source_sha256'], record['source_sha256'], 'input source')
        _same(binding['identity']['coordinate_frame'], record['coordinate_frame'], 'input frame')
        if binding['array_name'] != name:
            raise ValueError('Recorded reference input names differ.')
    rows = record['preprocessing']
    if type(rows) is not list or len(rows) != 5:
        raise ValueError('Recorded reference preprocessing inventory differs.')
    for stage, row in zip(_chain_stages(recipe), rows, strict=True):
        qc._exact(row, 'operation status run_record', 'recorded public preprocessing')
        run = StageRunRecord.from_dict(row['run_record'])
        validate_stage_run(recipe, run)
        if row['operation'] != stage.operation_id or run.stage_id != stage.stage_id:
            raise ValueError('Recorded reference preprocessing order differs.')
        if complete and (row['status'] not in {'executed', 'empty'} or run.checkpoint_sha256 is None):
            raise ValueError('Complete reference QC has an uncompleted preprocessing call.')
    calls = record['source_calls']
    operations = ('segmentation.watershed.capture', 'analysis.size', 'analysis.color')
    if type(calls) is not list or len(calls) != 3:
        raise ValueError('Recorded reference public-call inventory differs.')
    seen = set()
    for operation, call in zip(operations, calls, strict=True):
        qc._exact(call, 'operation call_id call_id_origin status invoked reason invoked_meaning '
            'checkpoint_sha256 checkpoint_identity details', 'recorded public call')
        if (call['operation'] != operation or type(call['call_id']) is not str or not call['call_id']
                or call['call_id'] in seen or type(call['invoked']) is not bool
                or type(call['reason']) is not str or type(call['details']) is not dict
                or call['status'] not in {'executed', 'empty', 'failed', 'unavailable', 'cancelled', 'not_run'}):
            raise ValueError('Recorded public-call facts differ.')
        seen.add(call['call_id'])
        if call['status'] in {'executed', 'empty'}:
            if call['invoked'] is not True or call['checkpoint_identity'] is None:
                raise ValueError('Successful public call lacks its captured identity.')
            qc._digest(call['checkpoint_sha256'])
            _same(call['checkpoint_identity']['source_sha256'], record['source_sha256'], 'call source')
            _same(call['checkpoint_identity']['coordinate_frame'], record['coordinate_frame'], 'call frame')
        elif call['checkpoint_sha256'] is not None or call['checkpoint_identity'] is not None:
            raise ValueError('Uncompleted public call cannot claim a checkpoint.')
        if complete and (call['status'] not in {'executed', 'empty'}
                         or call['details'].get('verified_public_result_received') is not True):
            raise ValueError('Complete reference QC has an unverified public call.')
    for name, expected in (
        ('runner_request_count_after_preprocessing', sum(call['invoked'] for call in calls)),
        ('verified_public_result_count_after_preprocessing', sum(
            bool(call['details'].get('verified_public_result_received')) for call in calls)),
    ):
        _same(record[name], expected, 'call count')


def _finish(recipe, identity):
    version = (5 if identity['scope'] == 'reference_capture_attempt'
               and identity['reference_capture']['expected_source_sha256'] is None else VERSION)
    return qc.MethodQC(qc._bytes({'format': qc.FORMAT, 'version': version, 'recipe': recipe.to_dict(),
        'identity': identity, 'qc_revision': sha256(qc._bytes(identity)).hexdigest()}).decode('utf-8'))


def qc_from_reference_session(session):
    """Capture already verified complete facts or an actual returned partial run."""
    if not isinstance(session, ReferenceCaptureSession):
        raise TypeError('Reference QC requires the actual immutable capture session.')
    if session.verified is None:
        if session.outcome is None:
            raise ValueError('An uncaptured session has no returned execution facts.')
        return reference_capture_attempt_qc(session.recipe, state=session.outcome.status,
            phase='capture', detail='Returned public chain has no complete verified manifest.',
            expected_source_sha256=session.outcome.record['source_sha256'], invoked=True, session=session)
    record = _thaw_json(session.verified.record)
    digest = session.verified.manifest_sha256
    if sha256(_canonical_json(record, name='recorded public manifest')).hexdigest() != digest:
        raise ValueError('Verified reference record differs from its manifest revision.')
    identity = qc._base(session.recipe, 'whole_reference_capture')
    identity['node_outcomes'] = None
    identity['reference_capture'] = {'manifest_sha256': digest, 'record': record,
        'execution_backend': PUBLIC_BACKEND, 'verification': 'verified_complete_manifest'}
    return _finish(session.recipe, identity)


def reference_capture_attempt_qc(recipe, *, state, phase, detail, expected_source_sha256,
                                 inputs_before=(), invoked=False, session=None, exception=None):
    """Record the actual reference boundary, with uncaptured facts left null."""
    identity = qc._base(recipe, 'reference_capture_attempt')
    identity['node_outcomes'] = None
    identity['reference_capture'] = {
        'attempt_id': uuid4().hex, 'state': state, 'phase': phase,
        'detail': str(detail)[:4096], 'error_type': None if exception is None else type(exception).__name__,
        'expected_source_sha256': expected_source_sha256, 'inputs_before': list(inputs_before),
        'invoked_executor': EXECUTOR if invoked else None,
        'record': None if session is None else _thaw_json(session.outcome.record),
        'manifest_sha256': None, 'verification': 'not_verified_complete_manifest',
    }
    return _finish(recipe, identity)


def validate_reference_capture_qc(payload):
    try:
        _validate_reference_capture_qc(payload)
    except (KeyError, TypeError, IndexError, AttributeError) as error:
        raise ValueError('Malformed recorded reference capture QC.') from error


def _validate_reference_capture_qc(payload):
    recipe = RecipeSpec.from_json(qc._bytes(payload['recipe']))
    value = payload['identity']
    qc._exact(value, 'scope recipe_mode recipe_identity native_artifact stage_record recorded_inspection '
        'node_outcomes measurement_eligibility scientific_validity reference_capture', 'reference QC identity')
    _same(value['recipe_identity'], recipe.identity.analytical_dict(), 'analytical identity')
    if (value['recipe_mode'] != recipe.provenance.mode.value
            or any(value[key] is not None for key in ('native_artifact', 'stage_record',
                'recorded_inspection', 'node_outcomes', 'measurement_eligibility', 'scientific_validity'))):
        raise ValueError('Reference QC cannot assert native artifacts or biological eligibility.')
    row = value['reference_capture']
    if value['scope'] == 'whole_reference_capture':
        qc._exact(row, 'manifest_sha256 record execution_backend verification', 'complete reference QC')
        qc._digest(row['manifest_sha256'])
        if row['execution_backend'] != PUBLIC_BACKEND or row['verification'] != 'verified_complete_manifest':
            raise ValueError('Complete reference QC execution route differs.')
        _chain(recipe, row['record'], complete=True)
        if sha256(_canonical_json(row['record'], name='recorded public manifest')).hexdigest() != row['manifest_sha256']:
            raise ValueError('Reference QC manifest revision differs from its exact recorded bytes.')
    elif value['scope'] == 'reference_capture_attempt':
        qc._exact(row, 'attempt_id state phase detail error_type expected_source_sha256 inputs_before '
            'invoked_executor record manifest_sha256 verification', 'reference attempt QC')
        if type(row['attempt_id']) is not str or not attempts._TOKEN.fullmatch(row['attempt_id']):
            raise ValueError('Invalid reference attempt token.')
        if row['state'] not in _STATES or row['phase'] not in _PHASES:
            raise ValueError('Unknown reference attempt outcome or phase.')
        attempts._text(row['detail'], 4096, 'reference detail')
        if row['error_type'] is not None:
            attempts._text(row['error_type'], 256, 'reference exception')
        if row['expected_source_sha256'] is None:
            if payload['version'] != 5 or row['record'] is not None:
                raise ValueError('Unknown source requires an uncaptured reference attempt.')
        else:
            qc._digest(row['expected_source_sha256'])
        attempts._validate_inputs(row['inputs_before'])
        if (row['manifest_sha256'] is not None or row['verification'] != 'not_verified_complete_manifest'
                or row['invoked_executor'] not in (None, EXECUTOR)
                or row['phase'] in {'admission', 'selection', 'between_rows'} and row['invoked_executor'] is not None):
            raise ValueError('Reference attempt cannot assert a completed capture or unobserved invocation.')
        if row['record'] is not None:
            if row['invoked_executor'] != EXECUTOR:
                raise ValueError('Returned reference facts require actual wrapper invocation.')
            _chain(recipe, row['record'], complete=False)
            _same(row['record']['source_sha256'], row['expected_source_sha256'], 'returned source')
            _same(row['record']['status'], row['state'], 'returned state')
    else:
        raise ValueError('Unknown reference QC scope.')
    if payload['version'] == 5 and (value['scope'] != 'reference_capture_attempt'
                                  or row['expected_source_sha256'] is not None):
        raise ValueError('Reference QC-v5 is reserved for uncaptured source identity.')
    qc._digest(payload['qc_revision'])
    if payload['qc_revision'] != sha256(qc._bytes(value)).hexdigest():
        raise ValueError('Reference QC revision differs from its recorded facts.')
