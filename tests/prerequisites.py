"""Read-only test prerequisite checks and optional local fixture provisioning.

Expected bytes are declared in config/test_fixtures.json. Tests never generate
or download baselines. An external fixture directory uses repository-relative
paths, including artifacts/; only missing hash-verified files are copied.
"""
from hashlib import sha256
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'config/test_fixtures.json'


def file_sha256(path):
    value = sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def check_prerequisites(mode, *, fixture_source=None, root=ROOT):
    required_groups = {'reference', 'acceptance'} if mode == 'full' else (
        {'reference'} if mode == 'cuda' else {'reference', 'acceptance'} if mode == 'acceptance'
        else {'acceptance'} if mode == 'history' else set())
    errors, copied, checked = [], [], []
    if not required_groups:
        return {'passed': True, 'errors': errors, 'checked_files': checked, 'copied_files': copied}
    record = json.loads((root / 'config/test_fixtures.json').read_text(encoding='utf-8'))
    for item in record['files']:
        if item['group'] not in required_groups:
            continue
        relative = Path(item['path'])
        destination = (root / relative).resolve()
        if relative.is_absolute() or not destination.is_relative_to((root / 'artifacts').resolve()):
            raise ValueError('Fixture destination must stay under artifacts/: ' + str(relative))
        if not destination.exists() and fixture_source is not None:
            source = (Path(fixture_source) / relative).resolve()
            if source.is_file() and file_sha256(source) == item['sha256']:
                destination.parent.mkdir(parents=True, exist_ok=True)
                # Exclusive creation preserves an existing reviewed file.
                with source.open('rb') as incoming, destination.open('xb') as output:
                    shutil.copyfileobj(incoming, output)
                copied.append(item['path'])
        if not destination.is_file():
            errors.append('Missing fixture: ' + item['path'])
        elif file_sha256(destination) != item['sha256']:
            errors.append('Fixture SHA256 mismatch: ' + item['path'])
        else:
            checked.append(item['path'])
    if 'reference' in required_groups:
        interpreter = root / 'artifacts/plantcv-reference-env/Scripts/python.exe'
        if not interpreter.is_file():
            errors.append('Missing installed pinned PlantCV interpreter: ' + str(interpreter))
    if mode in {'full', 'cuda', 'acceptance'}:
        import torch
        if not torch.cuda.is_available():
            errors.append('CUDA is required for ' + mode + '; use --tier cpu-qt for the explicit CPU subset.')
    return {'passed': not errors, 'errors': errors, 'checked_files': checked, 'copied_files': copied}
