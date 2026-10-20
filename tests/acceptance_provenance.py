"""Exact reviewed source-only migrations; immutable numerical baselines stay fixed."""
from hashlib import sha256
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / 'artifacts/pipeline-backend-code-provenance-v1.json'
RECORD_SHA256 = 'b5c30ee44f637e623ae7f4fec139b25020f629c342fc844c8e827e75a32e6d12'
# Filled only after source freeze and independent review. Tests never generate it.
SUCCESSOR_RECORD = ROOT / 'artifacts/full-resolution-code-provenance-v3.json'
SUCCESSOR_SHA256 = '2a93625ffb97309685b776cc1ad23bc6b3e4438ce2bf038469f829c3e865b924'
SUCCESSOR_SCHEMA = 'seedfiddle.full-resolution-source-provenance.v3'
ARCHIVED_SUCCESSOR = ROOT / 'artifacts/full-resolution-code-provenance-v1.json'
ARCHIVED_SUCCESSOR_SHA256 = '652865981554125d63ac7272784c01b157cc5b8728e86066b6edbe99dcb3daac'
ARCHIVED_V2 = ROOT / 'artifacts/full-resolution-code-provenance-v2.json'
ARCHIVED_V2_SHA256 = '3aa4298e979d60aa3ae3385017c3ffb4951aa94c7293d57fa6c57762bc734696'
WORKSPACE_RECORD = ROOT / 'artifacts/project-context-code-provenance-v1.json'
WORKSPACE_SHA256 = 'a46da97385b7f573509ecd72f1ab4b98253bbb1062946177385d1ee568b8bed9'


def _load(path, expected_sha256, schema):
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != expected_sha256:
        raise AssertionError('Reviewed source-identity record changed: ' + path.name)
    value = json.loads(raw)
    if value['schema'] != schema or value.get('numerical_baselines_modified') is not False:
        raise AssertionError('Unexpected source-only identity schema: ' + path.name)
    return value


def _records():
    previous = _load(RECORD, RECORD_SHA256, 'seedfiddle.pipeline-backend-source-provenance.v1')
    records = [previous]
    if SUCCESSOR_SHA256 is not None:
        current = _load(SUCCESSOR_RECORD, SUCCESSOR_SHA256, SUCCESSOR_SCHEMA)
        if SUCCESSOR_SCHEMA in ('seedfiddle.full-resolution-source-provenance.v2',
                               'seedfiddle.full-resolution-source-provenance.v3'):
            archived = _load(ARCHIVED_SUCCESSOR, ARCHIVED_SUCCESSOR_SHA256,
                             'seedfiddle.full-resolution-source-provenance.v1')
            if SUCCESSOR_SCHEMA.endswith('.v2'):
                _verify_supersession(current, archived)
            else:
                archived_v2 = _load(ARCHIVED_V2, ARCHIVED_V2_SHA256,
                                    'seedfiddle.full-resolution-source-provenance.v2')
                _verify_supersession(archived_v2, archived)
                _verify_supersession(current, archived_v2,
                    archive_path=ARCHIVED_V2, archive_sha256=ARCHIVED_V2_SHA256)
                if current.get('supersession_lineage') != [
                        dict(path=ARCHIVED_SUCCESSOR.relative_to(ROOT).as_posix(), sha256=ARCHIVED_SUCCESSOR_SHA256),
                        dict(path=ARCHIVED_V2.relative_to(ROOT).as_posix(), sha256=ARCHIVED_V2_SHA256)]:
                    raise AssertionError('Full-resolution V3 does not retain its exact V1/V2 lineage.')
        if (current['predecessor'] != RECORD.relative_to(ROOT).as_posix()
                or current['predecessor_sha256'] != RECORD_SHA256
                or current['baselines'] != previous['baselines']
                or current['previous_source_files'] != previous['source_files']
                or not set(previous['source_files']).issubset(current['source_files'])):
            raise AssertionError('Full-resolution record does not bind the exact predecessor and frozen baselines.')
        palette_audit = current['non_source_runtime_audit']
        from seedvision.benchmarking.native_recipe import _digest
        if (palette_audit['watershed_palette_sha256'] != palette_audit['previous_watershed_palette_sha256']
                or _digest(palette_audit['identity']) != palette_audit['watershed_palette_sha256']
                or _digest(palette_audit['reconstructed_previous_diagnostic_modules']) != palette_audit['frozen_diagnostic_modules_sha256']):
            raise AssertionError('Full-resolution source migration changes the palette runtime.')
        source_paths = [change['path'] for change in current['source_changes']]
        required = {name for name, digest in previous['source_files'].items()
                    if current['source_files'][name] != digest}
        if len(source_paths) != len(set(source_paths)) or not required.issubset(source_paths):
            raise AssertionError('Full-resolution source change inventory is incomplete or duplicated.')
        for change in current['source_changes']:
            name = change['path']
            if (change['old'] == change['new'] or current['source_files'].get(name) != change['new']
                    or not all(re.fullmatch('[0-9a-f]{64}', change[key]) for key in ('old', 'new'))
                    or (name in previous['source_files'] and previous['source_files'][name] != change['old'])):
                raise AssertionError('Full-resolution source transition differs from its exact inventory: ' + name)
        records.append(current)
        workspace = _load(WORKSPACE_RECORD, WORKSPACE_SHA256,
                          'seedfiddle.project-context-source-provenance.v1')
        _verify_supersession(workspace, current, archive_path=SUCCESSOR_RECORD,
                             archive_sha256=SUCCESSOR_SHA256)
        if ([row['path'] for row in workspace['superseded_source_changes']]
                != ['seedvision/pipeline/workflows.py']
                or any(workspace[key] != current[key] for key in (
                    'recipe_transitions', 'reader_transitions', 'reconstruction',
                    'non_source_runtime_audit', 'prior_transitions'))):
            raise AssertionError('Project context must not change numerical or runtime expectations.')
        # This is another replacement of the same base migration, not a
        # second application of its numerical source-field substitutions.
        records[-1] = workspace
    latest = records[-1]
    for name, digest in latest['source_files'].items():
        if sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            if name.startswith('tests/'):
                # Test organization is not a calculation identity. Keep the
                # actual numerical fixture definitions exact while allowing
                # independent test methods to become functional scenarios.
                from tests.fixture_definitions import verify_fixture_definitions
                verify_fixture_definitions(name, digest)
            else:
                raise AssertionError('Reviewed calculation source changed: ' + name)
    for name, digest in latest['baselines'].items():
        if sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise AssertionError('Frozen numerical baseline changed: ' + name)
    return records


def _verify_supersession(current, archived, *, archive_path=ARCHIVED_SUCCESSOR,
                         archive_sha256=ARCHIVED_SUCCESSOR_SHA256):
    """Versions replace the same base migration; never apply them in series."""
    if (current.get('supersedes') != archive_path.relative_to(ROOT).as_posix()
            or current.get('supersedes_sha256') != archive_sha256
            or current.get('superseded_source_files') != archived['source_files']
            or any(current[key] != archived[key] for key in
                   ('predecessor', 'predecessor_sha256', 'baselines', 'previous_source_files'))):
        raise AssertionError('Full-resolution replacement does not bind its immutable lineage.')
    names = sorted(set(archived['source_files']) | set(current['source_files']))
    exact = [dict(path=name, old=archived['source_files'].get(name),
                  new=current['source_files'].get(name)) for name in names
             if archived['source_files'].get(name) != current['source_files'].get(name)]
    if current.get('superseded_source_changes') != exact:
        raise AssertionError('Full-resolution replacement source delta differs from its prior version.')


def record():
    return _records()[-1]


def verify_previous_source(name, digest):
    current = sha256((ROOT / name).read_bytes()).hexdigest()
    if current == digest:
        return
    if name.startswith('tests/'):
        from tests.fixture_definitions import verify_fixture_definitions
        verify_fixture_definitions(name, digest)
        return
    records = _records()
    historical = {row['source_files'].get(name) for row in records}
    historical.update(row.get('superseded_source_files', {}).get(name) for row in records)
    if records[-1]['schema'] in ('seedfiddle.full-resolution-source-provenance.v3',
                                 'seedfiddle.project-context-source-provenance.v1'):
        archived_v1 = _load(ARCHIVED_SUCCESSOR, ARCHIVED_SUCCESSOR_SHA256,
                            'seedfiddle.full-resolution-source-provenance.v1')
        historical.add(archived_v1['source_files'].get(name))
        archived_v2 = _load(ARCHIVED_V2, ARCHIVED_V2_SHA256,
                            'seedfiddle.full-resolution-source-provenance.v2')
        historical.add(archived_v2['source_files'].get(name))
    historical.add(records[0]['legacy_source_files'].get(name))
    if digest not in historical or records[-1]['source_files'].get(name) != current:
        raise AssertionError('Unreviewed prior source transition: ' + name)


def _apply_changes(expected, changes, selection, *, successor=False):
    if selection not in ('recipe', 'reader'):
        raise AssertionError('Unknown acceptance transition selection.')
    fields = 'executor|correction_tensor_executor|correction_input_preparation'
    if successor:
        fields += '|calculation_modules_sha256'
    allowed = (re.compile(r'^/cases/(?:0|[1-9][0-9]*)/(?:runtime_calculation_pins/(?:' + fields + r')|preparation/(?:rgb|gray))$')
               if selection == 'recipe' else re.compile(r'^/pins/native_preparation_identity/1$'))
    seen = set()
    for change in changes:
        path = change['path']
        capture = path.startswith('/capture_pins/source_files/') and selection == 'recipe'
        if path in seen or not (allowed.fullmatch(path) or capture):
            raise AssertionError('Source transition changes a non-code field: ' + path)
        seen.add(path)
        names = [part.replace('~1', '/').replace('~0', '~') for part in path.split('/')[1:]]
        target = expected
        for name in names[:-1]:
            target = target[int(name)] if isinstance(target, list) else target[name]
        key = int(names[-1]) if isinstance(target, list) else names[-1]
        if target[key] != change['old']:
            raise AssertionError('Source transition prior value differs: ' + path)
        if '/preparation/' in path:
            tokens = re.compile(r'((?:input helper|preparation|CUDA converter) )[0-9a-f]{64}')
            if tokens.sub(r'\1<CODE>', change['old']) != tokens.sub(r'\1<CODE>', change['new']):
                raise AssertionError('Source transition changed preparation beyond source identities.')
        elif not all(re.fullmatch('[0-9a-f]{64}', change[name]) for name in ('old', 'new')):
            raise AssertionError('Source transition must contain exact SHA-256 values.')
        target[key] = change['new']
    return expected


def apply(expected, selection):
    for index, value in enumerate(_records()):
        _apply_changes(expected, value[selection + '_transitions'], selection, successor=index > 0)
    return expected


def capture_pins(pins):
    # Reuse the exact transition chains without applying any runtime changes.
    value = {'capture_pins': pins}
    for index, row in enumerate(_records()):
        changes = [change for change in row['recipe_transitions']
                   if change['path'].startswith('/capture_pins/source_files/')]
        _apply_changes(value, changes, 'recipe', successor=index > 0)
    return value['capture_pins']
