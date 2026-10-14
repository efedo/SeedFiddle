"""Terminal evidence for attempts whose analysis request was never captured."""
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
import logging
from pathlib import Path
from uuid import uuid4

from seedvision.export.run_outcomes import _json, _sha, _source_digest, load_run_outcome
from seedvision.learning.data import file_sha256
from seedvision.timing import AnalysisCancelled


FORMAT = 'seedfiddle-native-preflight-outcome'
CONTRACT = 'seedfiddle.native-preflight-outcome-v1'
PHASES = {
    'project': frozenset({'project_arguments', 'project_load', 'project_sources',
                          'project_settings', 'project_traits', 'desktop_project_load',
                          'desktop_project_settings', 'desktop_project_install'}),
    'image': frozenset({'request_capture', 'desktop_prepare', 'image_settings',
                        'image_references', 'image_manual_centres', 'image_library', 'image_request',
                        'desktop_source', 'desktop_bundled_reference_import'}),
    'settings_profile': frozenset({'desktop_profile_load', 'desktop_profile_settings', 'desktop_profile_install'}),
    'annotation': frozenset({'desktop_mask_import'}),
}
FILES = ('export/preflight_outcomes.py', 'export/run_outcomes.py',
         'export/project_runner.py', 'ui/main_window.py')
UNKNOWN = ('request_facts', 'image_facts', 'calibration_facts',
           'prediction_revision', 'backend_outcomes')
REASONS = {'rejected': 'preflight_invalid_input', 'failed': 'preflight_exception',
           'unavailable': 'preflight_io_unavailable', 'cancelled': 'preflight_cancelled'}


def _fact(digest):
    return {'sha256': digest, 'identity_state': 'observed' if digest else 'unavailable'}


def _comparison(before, after):
    if before is None or after is None:
        return 'unavailable'
    return 'unchanged' if before == after else 'changed'


def _state(error):
    if isinstance(error, AnalysisCancelled):
        return 'cancelled'
    cause = error
    seen = set()
    while cause is not None and id(cause) not in seen:
        seen.add(id(cause))
        if isinstance(cause, OSError):
            return 'unavailable'
        cause = cause.__cause__
    return 'rejected' if isinstance(error, (ValueError, TypeError)) else 'failed'


@dataclass(frozen=True, slots=True)
class PreflightRunOutcome:
    subject_path: Path
    payload_json: str

    def __post_init__(self):
        payload = self.payload
        if (payload['format'] != FORMAT or not isinstance(self.subject_path, Path)
                or str(self.subject_path) != payload['execution']['subject_path']):
            raise ValueError('Preflight outcome locator does not match its captured subject.')

    @property
    def payload(self):
        return load_run_outcome(self.payload_json)

    @property
    def image_path(self):
        return self.subject_path if self.payload['identity']['subject_kind'] == 'image' else None


@dataclass(frozen=True, slots=True)
class PreflightAttempt:
    subject_path: Path
    subject_kind: str
    phase: str
    attempt_id: str
    subject_sha256: str | None

    def outcome(self, error):
        if not isinstance(error, Exception):
            raise TypeError('Process-control exceptions are not preflight outcomes.')
        from seedvision import __version__
        package = Path(__file__).resolve().parents[1]
        current = _source_digest(self.subject_path)
        state = _state(error)
        identity = {'contract': CONTRACT, 'attempt_id': self.attempt_id,
            'subject_kind': self.subject_kind, 'phase': self.phase,
            'subject_at_request': _fact(self.subject_sha256), 'subject_at_completion': _fact(current),
            'subject_verification_at_completion': _comparison(self.subject_sha256, current),
            'state': state, 'reason_code': REASONS[state], 'error_type': type(error).__name__,
            **dict.fromkeys(UNKNOWN),
            'implementation': {'version': __version__,
                'files': {name: file_sha256(package / name) for name in FILES}}}
        # An incomplete request has no transferable prediction identity. Each
        # attempt owns its acknowledgment even when all known facts match.
        payload = {'format': FORMAT, 'version': 1, 'identity': identity,
            'outcome_revision': sha256(_json(identity).encode()).hexdigest(),
            'diagnostic': (str(error) or type(error).__name__)[:16384],
            'execution': {'subject_path': str(self.subject_path)}}
        return PreflightRunOutcome(self.subject_path, _json(payload))


def begin_preflight(path, subject_kind, phase):
    if (not isinstance(subject_kind, str) or subject_kind not in PHASES
            or not isinstance(phase, str) or phase not in PHASES[subject_kind]):
        raise ValueError('Unknown preflight subject or phase.')
    path = Path(path)
    return PreflightAttempt(path, subject_kind, phase, uuid4().hex, _source_digest(path))


def validate_preflight_payload(payload):
    """Validate a detached historical record without probing its selected files."""
    if (set(payload) != {'format', 'version', 'identity', 'outcome_revision', 'diagnostic', 'execution'}
            or payload['format'] != FORMAT or type(payload['version']) is not int or payload['version'] != 1):
        raise ValueError('Unsupported preflight outcome format.')
    identity = payload['identity']
    fields = {'contract', 'attempt_id', 'subject_kind', 'phase', 'subject_at_request',
              'subject_at_completion', 'subject_verification_at_completion',
              'state', 'reason_code', 'error_type', 'implementation', *UNKNOWN}
    if not isinstance(identity, dict) or set(identity) != fields or identity['contract'] != CONTRACT:
        raise ValueError('Invalid preflight outcome identity.')
    kind, phase, attempt, state = (identity[name] for name in ('subject_kind', 'phase', 'attempt_id', 'state'))
    if (not isinstance(kind, str) or kind not in PHASES or not isinstance(phase, str) or phase not in PHASES[kind]
            or not isinstance(attempt, str) or len(attempt) != 32 or any(c not in '0123456789abcdef' for c in attempt)
            or not isinstance(state, str) or state not in REASONS or identity['reason_code'] != REASONS[state]
            or not isinstance(identity['error_type'], str) or not identity['error_type']
            or any(identity[name] is not None for name in UNKNOWN)):
        raise ValueError('Preflight evidence cannot claim captured requests or completed analysis.')
    for name in ('subject_at_request', 'subject_at_completion'):
        fact = identity[name]
        if (not isinstance(fact, dict) or set(fact) != {'sha256', 'identity_state'}
                or (fact['sha256'] is None and fact['identity_state'] != 'unavailable')
                or (fact['sha256'] is not None and (not _sha(fact['sha256']) or fact['identity_state'] != 'observed'))):
            raise ValueError('Invalid preflight subject availability.')
    if identity['subject_verification_at_completion'] != _comparison(
            identity['subject_at_request']['sha256'], identity['subject_at_completion']['sha256']):
        raise ValueError('Invalid preflight subject verification.')
    implementation = identity['implementation']
    if (not isinstance(implementation, dict) or set(implementation) != {'version', 'files'}
            or not isinstance(implementation['version'], str) or not implementation['version']
            or not isinstance(implementation['files'], dict) or set(implementation['files']) != set(FILES)
            or any(not _sha(value) for value in implementation['files'].values())):
        raise ValueError('Invalid preflight implementation identity.')
    execution = payload['execution']
    if (not isinstance(execution, dict) or set(execution) != {'subject_path'}
            or not isinstance(execution['subject_path'], str) or not execution['subject_path']
            or not isinstance(payload['diagnostic'], str) or not payload['diagnostic']
            or len(payload['diagnostic']) > 16384 or not _sha(payload['outcome_revision'])
            or payload['outcome_revision'] != sha256(_json(identity).encode()).hexdigest()):
        raise ValueError('Invalid preflight execution evidence or revision.')
    return payload


@contextmanager
def record_preflight(path, subject_kind, phase, directory):
    """Publish rejection evidence and retain the original public exception."""
    attempt = begin_preflight(path, subject_kind, phase)
    try:
        yield
    except Exception as error:
        try:
            from seedvision.export.run_outcomes import export_run_outcome
            destination = export_run_outcome(directory, attempt.outcome(error))
        except Exception as recording_error:
            logging.getLogger(__name__).exception('Could not persist preflight terminal evidence; phase=%s', phase)
            error.add_note(f'Terminal recording also failed: {type(recording_error).__name__}: {recording_error}')
        else:
            error.add_note(f'Terminal record: {destination}')
        raise
