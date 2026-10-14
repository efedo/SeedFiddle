"""Detached desktop public-capture requests and historical terminal QC.

File observations occur in the worker. Delivery helpers consume those captured
facts only, and never reinterpret a completed capture as failed execution.
"""
from dataclasses import dataclass
import json

from seedvision.benchmarking.recipes import RecipeSpec
from seedvision.export.recipe_attempt_qc import capture_recipe_attempt_request, _observe_inputs
from seedvision.export.reference_capture_qc import reference_capture_attempt_qc, qc_from_reference_session


@dataclass(frozen=True, slots=True)
class PublicReferenceAttempt:
    request_json: str
    expected_source_sha256: str | None

    @property
    def recipe(self):
        return RecipeSpec.from_json(json.dumps(json.loads(self.request_json)['recipe']))

    @property
    def inputs_before(self):
        return json.loads(self.request_json)['inputs']


def capture_public_reference_attempt(recipe, source_path, *, expected_source_sha256=None):
    request = capture_recipe_attempt_request(recipe, {'source_image': source_path})
    observation = request.payload['inputs'][0]['observation']
    digest = expected_source_sha256
    if digest is None and observation['state'] == 'hashed':
        digest = observation['sha256']
    return PublicReferenceAttempt(request.payload_json, digest)


def source_matches_attempt(attempt):
    """Worker-only bounded re-observation; unknown is never verified equality."""
    rows = _observe_inputs([(row['name'], row['locator']) for row in attempt.inputs_before])
    return (attempt.expected_source_sha256 is not None
            and rows[0]['observation']['state'] == 'hashed'
            and rows[0]['observation']['sha256'] == attempt.expected_source_sha256)


def execution_quality(attempt, session):
    if session.verified is not None:
        return qc_from_reference_session(session)
    return reference_capture_attempt_qc(session.recipe, state=session.outcome.status,
        phase='capture', detail='Returned public chain has no complete verified manifest.',
        expected_source_sha256=session.outcome.record['source_sha256'],
        inputs_before=() if attempt is None else attempt.inputs_before, invoked=True, session=session)


def terminal_quality(attempt, *, state, phase, detail, invoked, exception=None):
    return reference_capture_attempt_qc(attempt.recipe, state=state, phase=phase,
        detail=detail, expected_source_sha256=attempt.expected_source_sha256,
        inputs_before=attempt.inputs_before, invoked=invoked, exception=exception)


def delivery_quality(recipe, execution, attempt, *, state, detail, invoked):
    """A separate delivery record; completed/partial execution QC is untouched."""
    capture = execution.payload['identity']['reference_capture']
    digest = (capture['record']['source_sha256'] if capture.get('record') is not None
              else capture.get('expected_source_sha256'))
    return reference_capture_attempt_qc(recipe, state=state, phase='verification',
        detail=f'Delivery withheld: {detail}. Execution QC {execution.revision} remains unchanged.',
        expected_source_sha256=digest, inputs_before=() if attempt is None else attempt.inputs_before,
        invoked=invoked)
