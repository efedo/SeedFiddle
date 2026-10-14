"""Bounded, explicitly retained recipe QC with project-specific ownership.

These records are historical metadata. Loading or transporting them never
probes the original recipe inputs, output archives or checkpoint stores.
"""
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import re

from seedvision.export import method_qc as qc
from seedvision.export.run_outcome_history import _read


ENTRIES = 2048
RECORDS = 256
TOTAL_BYTES = 64 * 1024 * 1024
ACK_RECORDS = 1024
ACK_BYTES = 16 * 1024 * 1024
_SHA = re.compile(r'[0-9a-f]{64}\Z')
_NAMESPACES = ('method-qc', 'method-qc-acknowledgments')
_SUFFIXES = ('.method-qc.json', '.method-qc-ack.json')


def history_root(root, project_path=None):
    if project_path is None:
        return Path(root) / 'artifacts'
    project = Path(project_path)
    if not project.is_absolute():
        project = Path(root) / project
    return project.with_name(project.name + '.qc-history')


def _directory(path):
    info = Path(path).lstat()
    if (Path(path).is_symlink() or getattr(info, 'st_file_attributes', 0) & 0x400
            or not Path(path).is_dir()):
        raise ValueError('Linked or non-directory recipe QC inventory is withheld.')


def _quality(raw):
    if not isinstance(raw, bytes) or len(raw) > qc.LIMIT:
        raise ValueError('Recipe QC record exceeds its byte bound.')
    try:
        return qc.MethodQC(raw.decode('utf-8'))
    except UnicodeError as error:
        raise ValueError('Recipe QC record is not UTF-8 JSON.') from error


def _ack(raw, qualities):
    if not isinstance(raw, bytes) or len(raw) > qc.ACK_LIMIT:
        raise ValueError('Recipe QC acknowledgment exceeds its byte bound.')
    def pairs(rows):
        result = {}
        for key, value in rows:
            if key in result:
                raise ValueError('Duplicate recipe QC acknowledgment field.')
            result[key] = value
        return result
    try:
        value = json.loads(raw.decode('utf-8'), object_pairs_hook=pairs,
            parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)))
        qc._bytes(value, qc.ACK_LIMIT)
    except (UnicodeError, RecursionError) as error:
        raise ValueError('Recipe QC acknowledgment is not bounded UTF-8 JSON.') from error
    revision = value.get('qc_revision') if type(value) is dict else None
    if type(revision) is not str or revision not in qualities:
        raise ValueError('Recipe QC acknowledgment has no retained exact QC revision.')
    return qc._validate_ack(value, qualities[revision])


def _identities(records):
    records = tuple(records)
    qualities = {}
    for raw in records:
        quality = _quality(raw)
        previous = qualities.get(quality.revision)
        if previous is not None and previous.payload['identity'] != quality.payload['identity']:
            raise ValueError('Conflicting analytical recipe QC identities.')
        qualities[quality.revision] = quality
    from seedvision.export.recipe_batch_qc import validate_batch_qc_links
    exact = {sha256(raw).hexdigest():raw for raw in records}
    for quality in qualities.values():
        validate_batch_qc_links(quality, exact)
    return qualities


@dataclass(frozen=True, slots=True)
class MethodQCHistory:
    records: tuple[bytes, ...]
    acknowledgments: tuple[bytes, ...]
    diagnostics: tuple[str, ...]
    incomplete: bool
    bytes_read: int
    entries_examined: int


def load_project_method_qc(root, project_path=None):
    """Only this selected master's retained history, or unassigned workspace history."""
    base = history_root(root, project_path)
    records, acknowledgments, diagnostics = {}, {}, []
    incomplete = False
    total_read = examined = 0
    try:
        _directory(base)
    except FileNotFoundError:
        return MethodQCHistory((), (), (), False, 0, 0)
    except (ValueError, OSError) as error:
        return MethodQCHistory((), (), (str(error),), True, 0, 0)
    for namespace, record_limit, byte_limit, file_limit, collected in (
        (_NAMESPACES[0], RECORDS, TOTAL_BYTES, qc.LIMIT, records),
        (_NAMESPACES[1], ACK_RECORDS, ACK_BYTES, qc.ACK_LIMIT, acknowledgments),
    ):
        count = total = entries_count = 0
        directory = base / namespace
        try:
            _directory(directory)
            with os.scandir(directory) as entries:
                for entry in entries:
                    entries_count += 1
                    examined += 1
                    if entries_count > ENTRIES:
                        incomplete = True
                        break
                    if not entry.name.endswith('.json'):
                        continue
                    count += 1
                    if count > record_limit:
                        incomplete = True
                        break
                    try:
                        raw = _read(entry.path, min(file_limit, max(0, byte_limit-total)))
                        total += len(raw)
                        total_read += len(raw)
                        if total > byte_limit:
                            incomplete = True
                            break
                        if len(raw) > file_limit:
                            raise ValueError('Recipe QC history record exceeds its byte bound.')
                        if namespace == _NAMESPACES[0]:
                            _quality(raw)
                        collected.setdefault(sha256(raw).hexdigest(), raw)
                    except (ValueError, OSError) as error:
                        diagnostics.append(f'{namespace}/{entry.name}: {type(error).__name__}: {error}')
        except FileNotFoundError:
            continue
        except (ValueError, OSError) as error:
            incomplete = True
            diagnostics.append(f'Cannot inventory {namespace}: {error}')
        if incomplete:
            break
    if incomplete:
        diagnostics.append('Recipe QC inventory is incomplete; records and acknowledgments withheld.')
        return MethodQCHistory((), (), tuple(diagnostics), True, total_read, examined)
    try:
        qualities = _identities(records.values())
    except ValueError as error:
        return MethodQCHistory((), (), (*diagnostics, str(error)), True, total_read, examined)
    valid_acks = {}
    for digest, raw in acknowledgments.items():
        try:
            _ack(raw, qualities)
            valid_acks[digest] = raw
        except ValueError as error:
            diagnostics.append(f'QC acknowledgment {digest}: {error}')
    return MethodQCHistory(tuple(records[key] for key in sorted(records)),
        tuple(valid_acks[key] for key in sorted(valid_acks)), tuple(diagnostics), False, total_read, examined)


def retain_method_qc(root, quality, *, project_path=None, acknowledgment=None):
    """Explicit retention; QC and acknowledgment are separate publications."""
    if not isinstance(quality, qc.MethodQC):
        raise TypeError('Retention requires validated recipe QC.')
    raw = quality.payload_json.encode('utf-8')
    ack_raw = None if acknowledgment is None else qc._bytes(acknowledgment, qc.ACK_LIMIT)
    if ack_raw is not None:
        _ack(ack_raw, {quality.revision: quality})
    base = history_root(root, project_path)
    if quality.payload['identity']['scope'] == 'recipe_batch_attempt':
        from seedvision.export.recipe_batch_qc import validate_batch_qc_links
        existing = load_project_method_qc(root, project_path)
        if existing.incomplete or existing.diagnostics:
            raise ValueError('Batch QC cannot retain into incomplete historical evidence.')
        validate_batch_qc_links(quality, {sha256(raw).hexdigest():raw for raw in existing.records})
    base.mkdir(parents=True, exist_ok=True)
    _directory(base)
    paths = []
    for namespace, value, limit in ((_NAMESPACES[0], raw, qc.LIMIT),
                                  (_NAMESPACES[1], ack_raw, qc.ACK_LIMIT)):
        if value is None:
            continue
        directory = base / namespace
        directory.mkdir(exist_ok=True)
        _directory(directory)
        path = directory / (sha256(value).hexdigest()+'.json')
        if path.exists():
            if _read(path, limit) != value:
                raise FileExistsError('Conflicting retained recipe QC bytes.')
        else:
            qc._publish(path, json.loads(value), limit)
        paths.append(path)
    return tuple(paths)


def capture_method_qc_evidence(root, project_path):
    history = load_project_method_qc(root, project_path)
    if history.incomplete or history.diagnostics:
        raise ValueError('Recipe QC snapshot evidence is incomplete: '+ '; '.join(history.diagnostics[:8]))
    return tuple((namespace+'/'+sha256(raw).hexdigest()+'.json', raw)
        for namespace, records in zip(_NAMESPACES, (history.records, history.acknowledgments), strict=True)
        for raw in records)


def verify_method_qc_evidence(source, index):
    """Validate exact objects, namespaces, identity linkage and bounded inventory."""
    supplementary, files = index.get('supplementary_evidence', {}), index.get('files')
    if not isinstance(supplementary, dict) or not isinstance(files, dict):
        raise ValueError('Invalid recipe QC supplementary inventory.')
    rows, records, acknowledgments, seen = [], [], [], set()
    counts, totals = [0, 0], [0, 0]
    for name, object_path in supplementary.items():
        if not isinstance(name, str):
            raise ValueError('Invalid recipe QC evidence name.')
        if not name.casefold().startswith('method-qc'):
            continue
        relative = PurePosixPath(name)
        if (len(relative.parts) != 2 or relative.parts[0] not in _NAMESPACES
                or relative.as_posix() != name or '\\' in name or ':' in name
                or name.casefold() in seen or not relative.parts[1].endswith('.json')
                or not _SHA.fullmatch(relative.parts[1][:-5])):
            raise ValueError('Unsafe recipe QC evidence mapping.')
        seen.add(name.casefold())
        kind = _NAMESPACES.index(relative.parts[0])
        count_limit, byte_limit, file_limit = ((RECORDS, TOTAL_BYTES, qc.LIMIT),
                                            (ACK_RECORDS, ACK_BYTES, qc.ACK_LIMIT))[kind]
        counts[kind] += 1
        if counts[kind] > count_limit:
            raise ValueError('Portable recipe QC exceeds its record budget.')
        digest = files.get(object_path) if isinstance(object_path, str) else None
        if (name in files or not isinstance(digest, str) or not _SHA.fullmatch(digest)
                or digest != relative.parts[1][:-5] or object_path != 'objects/'+digest+_SUFFIXES[kind]):
            raise ValueError('Unsafe or unverified recipe QC evidence object.')
        path = Path(source) / object_path
        if not path.resolve().is_relative_to(Path(source).resolve()):
            raise ValueError('Recipe QC object escaped its snapshot.')
        limit = min(file_limit, max(0, byte_limit-totals[kind]))
        raw = _read(path, limit)
        if len(raw) > limit or sha256(raw).hexdigest() != digest:
            raise ValueError('Portable recipe QC exceeds its limit or changed bytes.')
        totals[kind] += len(raw)
        (records if kind == 0 else acknowledgments).append(raw)
        rows.append((name, object_path, digest, limit, raw))
    qualities = _identities(records)
    for raw in acknowledgments:
        _ack(raw, qualities)
    reserved = {name for name in files if isinstance(name, str) and name.endswith(_SUFFIXES)}
    if reserved != {row[1] for row in rows}:
        raise ValueError('Portable recipe QC inventory has omitted or extra evidence mappings.')
    return tuple(rows)
