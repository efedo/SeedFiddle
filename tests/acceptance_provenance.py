"""Exact reviewed source-only migrations; immutable numerical baselines stay fixed."""
from hashlib import sha256
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / 'artifacts/pipeline-backend-code-provenance-v1.json'
RECORD_SHA256 = 'b5c30ee44f637e623ae7f4fec139b25020f629c342fc844c8e827e75a32e6d12'


def record():
    raw = RECORD.read_bytes()
    if sha256(raw).hexdigest() != RECORD_SHA256:
        raise AssertionError('Reviewed backend source-identity record changed.')
    value = json.loads(raw)
    if value['schema'] != 'seedfiddle.pipeline-backend-source-provenance.v1':
        raise AssertionError('Unexpected backend source-identity schema.')
    for name, digest in value['source_files'].items():
        if sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            if name.startswith('tests/'):
                # Test organization is not a calculation identity. Keep the
                # actual numerical fixture definitions exact while allowing
                # independent test methods to become functional scenarios.
                from tests.fixture_definitions import verify_fixture_definitions
                verify_fixture_definitions(name, digest)
            else:
                raise AssertionError('Reviewed backend source changed: ' + name)
    for name, digest in value['baselines'].items():
        if sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise AssertionError('Frozen numerical baseline changed: ' + name)
    return value


def verify_previous_source(name, digest):
    current = sha256((ROOT / name).read_bytes()).hexdigest()
    if current == digest:
        return
    if name.startswith('tests/'):
        from tests.fixture_definitions import verify_fixture_definitions
        verify_fixture_definitions(name, digest)
        return
    value = record()
    if value['legacy_source_files'].get(name) != digest or value['source_files'].get(name) != current:
        raise AssertionError('Unreviewed prior source transition: ' + name)


def apply(expected, selection):
    value = record()
    changes = value[selection + '_transitions']
    allowed = (re.compile(r'^/cases/(?:0|[1-9][0-9]*)/(?:runtime_calculation_pins/(?:executor|correction_tensor_executor|correction_input_preparation)|preparation/(?:rgb|gray))$')
               if selection == 'recipe' else re.compile(r'^/pins/native_preparation_identity/1$'))
    seen = set()
    for change in changes:
        path = change['path']
        capture = path.startswith('/capture_pins/source_files/') and selection == 'recipe'
        if path in seen or not (allowed.fullmatch(path) or capture):
            raise AssertionError('Backend transition changes a non-code field: ' + path)
        seen.add(path)
        fields = [part.replace('~1', '/').replace('~0', '~') for part in path.split('/')[1:]]
        target = expected
        for field in fields[:-1]:
            target = target[int(field)] if isinstance(target, list) else target[field]
        key = int(fields[-1]) if isinstance(target, list) else fields[-1]
        if target[key] != change['old']:
            raise AssertionError('Backend transition prior value differs: ' + path)
        if '/preparation/' in path:
            tokens = re.compile(r'((?:input helper|preparation|CUDA converter) )[0-9a-f]{64}')
            if tokens.sub(r'\1<CODE>', change['old']) != tokens.sub(r'\1<CODE>', change['new']):
                raise AssertionError('Backend transition changed preparation beyond source identities.')
        elif not all(re.fullmatch('[0-9a-f]{64}', change[name]) for name in ('old', 'new')):
            raise AssertionError('Backend source transition must contain exact SHA-256 values.')
        target[key] = change['new']
    return expected


def capture_pins(pins):
    # Reuse the same exact transition list without applying any runtime changes.
    value = {'capture_pins': pins}
    for change in record()['recipe_transitions']:
        if change['path'].startswith('/capture_pins/source_files/'):
            name = change['path'].removeprefix('/capture_pins/source_files/').replace('~1', '/').replace('~0', '~')
            current = value['capture_pins']['source_files'][name]
            if current != change['old']:
                raise AssertionError('Backend capture source differs: ' + name)
            value['capture_pins']['source_files'][name] = change['new']
    return value['capture_pins']
