"""Bounded captured recipe attempts; these records verify no output archive.

Byte observations describe selected files at an actual attempt boundary. They
do not establish image/model format validity, checkpoint execution or scientific
eligibility. Loading this evidence never opens the selected files.
"""
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import stat
from uuid import uuid4

from seedvision.benchmarking.checkpoints import CoordinateFrame
from seedvision.benchmarking.recipes import RecipeSpec
from seedvision.export import method_qc as qc


VERSION = 2
MAX_INPUTS = 128
MAX_FILE_BYTES = 256 * 1024 * 1024
MAX_TOTAL_BYTES = 512 * 1024 * 1024
NATIVE_PATH_EXECUTOR = 'seedvision.benchmarking.native_recipe.run_native_recipe_path'
CORRECTION_FILE_EXECUTOR = 'seedvision.benchmarking.native_correction_inputs.run_native_correction_files'
PROJECT_CORRECTION_EXECUTOR = 'seedvision.benchmarking.project_correction.run_project_correction'
_CORRECTION_EXECUTORS = {CORRECTION_FILE_EXECUTOR, PROJECT_CORRECTION_EXECUTOR}
_EXECUTORS = {NATIVE_PATH_EXECUTOR, *_CORRECTION_EXECUTORS}
_TOKEN = re.compile(r'[0-9a-f]{32}\Z')
_STATES = {'failed', 'rejected', 'cancelled', 'unavailable'}
_PHASES = {'request_capture', 'native_path', 'correction_path', 'delivery',
           'result_validation', 'batch_selection', 'batch_publication', 'archive_export'}
_OBSERVATIONS = {'hashed', 'missing', 'unreadable', 'not_file', 'over_limit',
                 'budget_exhausted', 'changed'}


def _text(value, maximum, name, *, empty=False):
    if (type(value) is not str or len(value) > maximum
            or not empty and not value.strip()):
        raise ValueError(f'Invalid recipe attempt {name}.')


def _validate_observation(value):
    qc._exact(value, 'state sha256 size_bytes error_type', 'recipe input observation')
    state = value['state']
    if type(state) is not str or state not in _OBSERVATIONS:
        raise ValueError('Unknown recipe input observation.')
    size = value['size_bytes']
    if size is not None and (type(size) is not int or not 0 <= size <= 2**63-1):
        raise ValueError('Invalid recipe input byte count.')
    if state == 'hashed':
        qc._digest(value['sha256'])
        if size is None or size > MAX_FILE_BYTES or value['error_type'] is not None:
            raise ValueError('Hashed recipe input contradicts its byte bounds.')
    elif value['sha256'] is not None:
        raise ValueError('Unhashed recipe input cannot assert a byte digest.')
    if state in {'missing', 'unreadable'}:
        _text(value['error_type'], 256, 'input exception')
        if state == 'missing' and size is not None:
            raise ValueError('Missing recipe input has no observed byte count.')
    elif value['error_type'] is not None:
        raise ValueError('Recipe input observation cannot invent an exception.')
    if state in {'over_limit', 'budget_exhausted', 'changed'} and size is None:
        raise ValueError('Bounded recipe observation requires its observed size.')
    if state == 'over_limit' and size <= MAX_FILE_BYTES:
        raise ValueError('Oversized recipe observation contradicts its limit.')


def _validate_inputs(rows):
    if type(rows) is not list or len(rows) > MAX_INPUTS:
        raise ValueError('Recipe attempt input inventory exceeds its bound.')
    names = set()
    total = 0
    for row in rows:
        qc._exact(row, 'name locator observation', 'selected recipe input')
        _text(row['name'], 256, 'input name')
        _text(row['locator'], 32768, 'input locator')
        if row['name'] in names:
            raise ValueError('Duplicate recipe attempt input.')
        names.add(row['name'])
        _validate_observation(row['observation'])
        if row['observation']['state'] == 'hashed':
            total += row['observation']['size_bytes']
    if total > MAX_TOTAL_BYTES:
        raise ValueError('Recipe attempt byte observations exceed their aggregate bound.')


def _selected_recipe(value):
    return None if value is None else RecipeSpec.from_json(qc._bytes(value))


def _validate_request(value):
    qc._exact(value, 'attempt_id recipe inputs', 'recipe attempt request')
    if type(value['attempt_id']) is not str or not _TOKEN.fullmatch(value['attempt_id']):
        raise ValueError('Invalid recipe attempt token.')
    _selected_recipe(value['recipe'])
    _validate_inputs(value['inputs'])


@dataclass(frozen=True)
class RecipeAttemptRequest:
    """Detached selected request and observations captured before invocation."""
    payload_json: str

    def __post_init__(self):
        # MethodQC owns the strict persisted parser. Requests are produced here,
        # never imported from a file, but still reject noncompact/deep metadata.
        if type(self.payload_json) is not str or len(self.payload_json.encode('utf-8')) > qc.LIMIT:
            raise ValueError('Recipe attempt request exceeds its byte bound.')
        def pairs(rows):
            value = {}
            for key, item in rows:
                if key in value:
                    raise ValueError('Duplicate recipe attempt request field.')
                value[key] = item
            return value
        value = json.loads(self.payload_json, object_pairs_hook=pairs,
            parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)))
        _validate_request(value)
        object.__setattr__(self, 'payload_json', qc._bytes(value).decode('utf-8'))

    @property
    def payload(self):
        return json.loads(self.payload_json)


def _signature(value):
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns


def _observe(path, remaining):
    size = None
    count = 0
    try:
        selected = Path(path)
        before = selected.stat()
        size = before.st_size
        if not stat.S_ISREG(before.st_mode):
            return {'state': 'not_file', 'sha256': None, 'size_bytes': size, 'error_type': None}, 0
        if size > MAX_FILE_BYTES:
            return {'state': 'over_limit', 'sha256': None, 'size_bytes': size, 'error_type': None}, 0
        if size > remaining:
            return {'state': 'budget_exhausted', 'sha256': None, 'size_bytes': size, 'error_type': None}, 0
        digest = sha256()
        with selected.open('rb') as stream:
            opened = os.fstat(stream.fileno())
            if _signature(opened) != _signature(before):
                return {'state': 'changed', 'sha256': None, 'size_bytes': size, 'error_type': None}, 0
            while count < size:
                chunk = stream.read(min(1024 * 1024, size-count))
                if not chunk:
                    break
                count += len(chunk)
                digest.update(chunk)
            after = os.fstat(stream.fileno())
        if (count != size or _signature(after) != _signature(before)
                or _signature(selected.stat()) != _signature(before)):
            return {'state': 'changed', 'sha256': None, 'size_bytes': size, 'error_type': None}, count
        return {'state': 'hashed', 'sha256': digest.hexdigest(), 'size_bytes': size, 'error_type': None}, count
    except OSError as error:
        missing = isinstance(error, FileNotFoundError)
        return {'state': 'missing' if missing else 'unreadable', 'sha256': None,
                'size_bytes': None if missing else size, 'error_type': type(error).__name__}, count


def _observe_inputs(selected):
    rows, remaining = [], MAX_TOTAL_BYTES
    for name, locator in selected:
        _text(name, 256, 'input name')
        _text(locator, 32768, 'input locator')
        observation, count = _observe(locator, remaining)
        remaining = max(0, remaining-count)
        rows.append({'name': name, 'locator': locator, 'observation': observation})
    return rows


def capture_recipe_attempt_request(recipe, inputs):
    """Capture bounded file-byte facts; failed reads remain explicit unknowns."""
    if recipe is not None and not isinstance(recipe, RecipeSpec):
        raise TypeError('Recipe attempt capture requires a parsed recipe or unknown recipe.')
    if not isinstance(inputs, Mapping) or len(inputs) > MAX_INPUTS:
        raise ValueError('Recipe attempt input inventory exceeds its bound.')
    selected = [(name, os.fspath(path)) for name, path in inputs.items()]
    value = {'attempt_id': uuid4().hex, 'recipe': None if recipe is None else recipe.to_dict(),
             'inputs': _observe_inputs(selected)}
    return RecipeAttemptRequest(qc._bytes(value).decode('utf-8'))


def _returned_inputs(request, result, executor):
    from seedvision.benchmarking.native_recipe import NativeRecipeResult
    from seedvision.benchmarking.native_correction_recipe import NativeCorrectionResult
    recipe = _selected_recipe(request['recipe'])
    expected = NativeCorrectionResult if executor in _CORRECTION_EXECUTORS else NativeRecipeResult
    if (recipe is None or not isinstance(result, expected)
            or result.recipe_identity.analytical_dict() != recipe.identity.analytical_dict()):
        return None
    facts = {'source_sha256': result.source_sha256, 'coordinate_frame': result.coordinate_frame.to_dict()}
    if executor in _CORRECTION_EXECUTORS:
        facts.update(target_source_sha256=result.target_source_sha256,
                     target_coordinate_frame=result.target_coordinate_frame.to_dict())
    return facts


def recipe_attempt_qc(request, *, state, phase, detail, exception=None,
                      invoked=False, result_returned=False, returned_result=None,
                      capture_completion=True, invoked_executor=None):
    """Finish an actual boundary record without retaining or verifying products.

    A late Qt delivery can set capture_completion=False: after-file observations
    then remain unknown, and no selected-file I/O occurs on the UI thread.
    """
    if not isinstance(request, RecipeAttemptRequest):
        raise TypeError('Recipe attempt QC requires its captured request.')
    if any(type(item) is not bool for item in (invoked, result_returned, capture_completion)):
        raise TypeError('Recipe attempt invocation, return and capture choices must be boolean.')
    if not invoked and invoked_executor is not None:
        raise ValueError('An uninvoked recipe cannot claim an executor.')
    executor = (NATIVE_PATH_EXECUTOR if invoked_executor is None else invoked_executor) if invoked else None
    value = request.payload
    recipe = _selected_recipe(value['recipe'])
    after = (_observe_inputs([(row['name'], row['locator']) for row in value['inputs']])
             if capture_completion else None)
    reason = {'failed': 'exception', 'rejected': 'request_rejected',
              'cancelled': 'cancellation', 'unavailable': 'executor_unavailable'}.get(state)
    identity = {'scope': 'recipe_attempt',
        'recipe_mode': None if recipe is None else recipe.provenance.mode.value,
        'recipe_identity': None if recipe is None else recipe.identity.analytical_dict(),
        'native_artifact': None, 'stage_record': None, 'recorded_inspection': None,
        'node_outcomes': None, 'measurement_eligibility': None, 'scientific_validity': None,
        'attempt_record': {'attempt_id': value['attempt_id'], 'state': state, 'phase': phase,
            'reason_code': reason, 'detail': str(detail)[:4096],
            'exception_type': None if exception is None else type(exception).__name__,
            'invoked_executor': executor,
            'result_returned': result_returned, 'inputs_before': value['inputs'], 'inputs_after': after,
            'returned_inputs': _returned_inputs(value, returned_result, executor) if result_returned else None}}
    payload = {'format': qc.FORMAT, 'version': VERSION, 'recipe': value['recipe'],
               'identity': identity, 'qc_revision': sha256(qc._bytes(identity)).hexdigest()}
    return qc.MethodQC(qc._bytes(payload).decode('utf-8'))


def validate_recipe_attempt(payload):
    """Strict v2 dispatch; historical consistency only, never fresh file checks."""
    recipe = _selected_recipe(payload['recipe'])
    value = payload['identity']
    qc._exact(value, 'scope recipe_mode recipe_identity native_artifact stage_record recorded_inspection '
                    'node_outcomes measurement_eligibility scientific_validity attempt_record', 'recipe attempt identity')
    if (value['scope'] != 'recipe_attempt'
            or value['recipe_mode'] != (None if recipe is None else recipe.provenance.mode.value)
            or value['recipe_identity'] != (None if recipe is None else recipe.identity.analytical_dict())
            or any(value[key] is not None for key in ('native_artifact', 'stage_record', 'recorded_inspection',
                   'node_outcomes', 'measurement_eligibility', 'scientific_validity'))):
        raise ValueError('Recipe attempt cannot claim archive, stage, output or eligibility facts.')
    row = value['attempt_record']
    qc._exact(row, 'attempt_id state phase reason_code detail exception_type invoked_executor result_returned '
                  'inputs_before inputs_after returned_inputs', 'recipe attempt record')
    _validate_request({'attempt_id': row['attempt_id'], 'recipe': payload['recipe'], 'inputs': row['inputs_before']})
    if (type(row['state']) is not str or row['state'] not in _STATES
            or type(row['phase']) is not str or row['phase'] not in _PHASES
            or row['reason_code'] != {'failed': 'exception', 'rejected': 'request_rejected',
                'cancelled': 'cancellation', 'unavailable': 'executor_unavailable'}[row['state']]
            or type(row['result_returned']) is not bool):
        raise ValueError('Invalid terminal recipe attempt state or phase.')
    _text(row['detail'], 4096, 'terminal detail')
    if row['exception_type'] is not None:
        _text(row['exception_type'], 256, 'terminal exception')
    if row['state'] == 'failed' and row['exception_type'] is None:
        raise ValueError('Failed recipe attempt requires its observed exception.')
    executor = row['invoked_executor']
    if executor is not None:
        from seedvision.benchmarking.native_correction_recipe import RECIPE_ID
        if (recipe is None or type(executor) is not str or executor not in _EXECUTORS
                or executor == NATIVE_PATH_EXECUTOR and row['phase'] == 'correction_path'
                or executor in _CORRECTION_EXECUTORS and
                (recipe.recipe_id != RECIPE_ID or row['phase'] == 'native_path')):
            raise ValueError('Recipe attempt executor contradicts its captured invocation boundary.')
    if (recipe is None and (row['state'] != 'rejected' or row['phase'] != 'request_capture')
            or row['phase'] in {'request_capture', 'batch_selection'} and executor is not None
            or row['state'] in {'rejected', 'unavailable'} and (executor is not None or row['result_returned'])
            or row['result_returned'] and executor is None
            or row['phase'] in {'delivery', 'result_validation'} and not row['result_returned']):
        raise ValueError('Recipe attempt invocation or return claims contradict its boundary.')
    if row['inputs_after'] is not None:
        _validate_inputs(row['inputs_after'])
        if [(r['name'], r['locator']) for r in row['inputs_before']] != [(r['name'], r['locator']) for r in row['inputs_after']]:
            raise ValueError('Recipe attempt changed its selected input inventory.')
    returned = row['returned_inputs']
    if returned is not None:
        if not row['result_returned']:
            raise ValueError('Unreturned recipe attempt cannot assert captured result facts.')
        keys = 'source_sha256 coordinate_frame'
        if executor in _CORRECTION_EXECUTORS:
            keys += ' target_source_sha256 target_coordinate_frame'
        qc._exact(returned, keys, 'returned recipe input facts')
        qc._digest(returned['source_sha256'])
        CoordinateFrame.from_dict(returned['coordinate_frame'])
        if executor in _CORRECTION_EXECUTORS:
            qc._digest(returned['target_source_sha256'])
            CoordinateFrame.from_dict(returned['target_coordinate_frame'])
    qc._digest(payload['qc_revision'])
    if payload['qc_revision'] != sha256(qc._bytes(value)).hexdigest():
        raise ValueError('Recipe QC revision differs from its recorded facts.')
