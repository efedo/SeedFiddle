"""Portable historical terminal evidence, separate from predictions and approvals."""
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import re

from seedvision.export.run_outcome_history import (
    HISTORY_BYTES, HISTORY_ENTRIES, HISTORY_RECORDS, RunOutcomeHistory,
    _outcome, _read, coalesce_run_outcomes, load_run_outcome_history,
)
from seedvision.export.run_outcomes import LIMIT, _json


ACK_LIMIT = 64 * 1024
ACK_BYTES = 16 * 1024 * 1024
ACK_RECORDS = 1024
ACK_CONTRACT = 'seedfiddle.run-outcome-acknowledgment-v1'
_SHA = re.compile(r'[0-9a-f]{64}\Z')


def _roots(root, project_path, workspace_name, private_name):
    paths = [Path(root) / 'artifacts' / workspace_name]
    if project_path is not None:
        project = Path(project_path)
        if not project.is_absolute():
            project = Path(root) / project
        private = project.parent / private_name
        if private != paths[0]:
            paths.append(private)
    return paths


def load_project_terminal_history(root, project_path=None, *, live=()):
    """Only workspace and the selected project's designated private namespace."""
    outcomes, diagnostics = [], []
    total = examined = 0
    incomplete = False
    for directory in _roots(root, project_path, 'run-outcomes', 'terminal-history'):
        try:
            info = directory.lstat()
            if directory.is_symlink() or getattr(info, 'st_file_attributes', 0) & 0x400:
                raise ValueError('Linked terminal history inventory is withheld.')
        except FileNotFoundError:
            continue
        except (ValueError, OSError) as error:
            incomplete = True
            diagnostics.append(f'Cannot inventory project terminal history: {error}')
            continue
        history = load_run_outcome_history(directory,
            entry_limit=max(1, HISTORY_ENTRIES - examined),
            byte_limit=max(1, HISTORY_BYTES - total))
        total += history.bytes_read
        examined += history.entries_examined
        outcomes.extend(history.outcomes)
        diagnostics.extend(history.diagnostics)
        incomplete |= history.incomplete
    if total > HISTORY_BYTES or examined > HISTORY_ENTRIES:
        incomplete = True
        diagnostics.append('Combined project terminal history exceeds its inventory/byte budget.')
    coherent, conflicts = coalesce_run_outcomes((*outcomes, *live))
    diagnostics.extend(conflicts)
    if incomplete:
        coherent = ()
    return RunOutcomeHistory(coherent, tuple(diagnostics), incomplete, total, examined)


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate acknowledgment JSON field.')
        result[key] = value
    return result


def load_acknowledgment(raw):
    """Multiple actor/time events are retained; no authenticated authorship claim."""
    if not isinstance(raw, bytes) or len(raw) > ACK_LIMIT:
        raise ValueError('Acknowledgment exceeds its bounded JSON contract.')
    try:
        record = json.loads(raw.decode('utf-8'), object_pairs_hook=_unique,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Invalid JSON constant.')))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError('Acknowledgment is not UTF-8 JSON.') from error
    if (not isinstance(record, dict) or set(record) != {
            'contract', 'outcome_revision', 'decision', 'operator_account', 'reviewed_at_utc'}
            or record['contract'] != ACK_CONTRACT or record['decision'] != 'acknowledged'
            or not isinstance(record['outcome_revision'], str) or not _SHA.fullmatch(record['outcome_revision'])
            or not isinstance(record['operator_account'], str) or not record['operator_account'].strip()
            or not isinstance(record['reviewed_at_utc'], str)):
        raise ValueError('Invalid terminal acknowledgment fields.')
    try:
        timestamp = datetime.fromisoformat(record['reviewed_at_utc'])
        if timestamp.utcoffset() is None or timestamp.utcoffset().total_seconds() != 0:
            raise ValueError('Timestamp must identify UTC.')
    except ValueError as error:
        raise ValueError('Invalid terminal acknowledgment UTC timestamp.') from error
    return record


@dataclass(frozen=True, slots=True)
class AcknowledgmentHistory:
    records: tuple[bytes, ...]
    diagnostics: tuple[str, ...]
    incomplete: bool


def load_project_acknowledgments(root, project_path=None):
    records, diagnostics = {}, []
    examined = total = count = 0
    incomplete = False
    for directory in _roots(root, project_path, 'run-outcome-acknowledgments', 'terminal-acknowledgments'):
        try:
            info = directory.lstat()
            if directory.is_symlink() or getattr(info, 'st_file_attributes', 0) & 0x400:
                raise ValueError('Linked acknowledgment inventory is withheld.')
            with os.scandir(directory) as entries:
                for entry in entries:
                    examined += 1
                    if examined > HISTORY_ENTRIES:
                        incomplete = True
                        break
                    if not entry.name.endswith('.json'):
                        continue
                    count += 1
                    if count > ACK_RECORDS:
                        incomplete = True
                        break
                    try:
                        raw = _read(entry.path, min(ACK_LIMIT, max(0, ACK_BYTES - total)))
                        total += len(raw)
                        if total > ACK_BYTES:
                            incomplete = True
                            break
                        record = load_acknowledgment(raw)
                        records.setdefault(_json(record), raw)
                    except (ValueError, OSError) as error:
                        diagnostics.append(f'{entry.name}: {type(error).__name__}: {error}')
        except FileNotFoundError:
            continue
        except (ValueError, OSError) as error:
            incomplete = True
            diagnostics.append(f'Cannot inventory terminal acknowledgments: {error}')
        if incomplete:
            break
    if incomplete:
        diagnostics.append('Terminal acknowledgment inventory is incomplete; records withheld.')
        records.clear()
    return AcknowledgmentHistory(tuple(records[key] for key in sorted(records)), tuple(diagnostics), incomplete)


def capture_terminal_evidence(root, project_path):
    """Fail incomplete/conflicting capture; each acknowledgment needs its outcome."""
    history = load_project_terminal_history(root, project_path)
    acknowledgments = load_project_acknowledgments(root, project_path)
    diagnostics = (*history.diagnostics, *acknowledgments.diagnostics)
    if history.incomplete or acknowledgments.incomplete or diagnostics:
        raise ValueError('Terminal snapshot evidence is incomplete: ' + '; '.join(diagnostics[:8]))
    rows, revisions = [], set()
    for outcome in history.outcomes:
        revision = outcome.payload['outcome_revision']
        revisions.add(revision)
        rows.append((f'terminal-history/{revision}/run-outcome.json', outcome.payload_json.encode('utf-8')))
    for raw in acknowledgments.records:
        if load_acknowledgment(raw)['outcome_revision'] not in revisions:
            raise ValueError('Terminal acknowledgment has no included outcome; export its terminal record into history before snapshotting.')
        rows.append((f'terminal-acknowledgments/{sha256(raw).hexdigest()}.json', raw))
    return tuple(rows)


def evidence_bytes(path, digest, limit):
    raw = _read(path, limit)
    if len(raw) > limit or sha256(raw).hexdigest() != digest:
        raise ValueError('Portable terminal evidence exceeds its limit or changed bytes.')
    return raw


def verify_terminal_evidence(source, index):
    """Verify generated names, exact objects, schemas, pairing and aggregate caps."""
    supplementary = index.get('supplementary_evidence', {})
    if not isinstance(supplementary, dict) or not isinstance(index.get('files'), dict):
        raise ValueError('Invalid supplementary evidence inventory.')
    rows, outcomes, acknowledgments, seen = [], [], [], set()
    terminal_bytes = ack_bytes = 0
    for name, object_path in supplementary.items():
        if not isinstance(name, str):
            raise ValueError('Invalid supplementary evidence name.')
        if not name.casefold().startswith(('terminal-history', 'terminal-acknowledgments')):
            continue
        relative = PurePosixPath(name)
        is_terminal = relative.parts[0] == 'terminal-history'
        if (name.casefold() in seen or '\\' in name or ':' in name
                or relative.as_posix() != name or not isinstance(object_path, str)):
            raise ValueError('Unsafe portable terminal evidence mapping.')
        seen.add(name.casefold())
        if is_terminal:
            valid = len(relative.parts) == 3 and _SHA.fullmatch(relative.parts[1]) and relative.parts[2] == 'run-outcome.json'
            limit, suffix = LIMIT, '.terminal.json'
            if len(outcomes) >= HISTORY_RECORDS:
                raise ValueError('Portable terminal evidence exceeds its record budget.')
            limit = min(limit, HISTORY_BYTES - terminal_bytes)
        else:
            valid = (len(relative.parts) == 2 and relative.parts[0] == 'terminal-acknowledgments'
                     and relative.parts[1].endswith('.json') and _SHA.fullmatch(relative.parts[1][:-5]))
            limit, suffix = ACK_LIMIT, '.terminal-ack.json'
            if len(acknowledgments) >= ACK_RECORDS:
                raise ValueError('Portable terminal evidence exceeds its acknowledgment budget.')
            limit = min(limit, ACK_BYTES - ack_bytes)
        digest = index['files'].get(object_path)
        if (not valid or name in index['files'] or not isinstance(digest, str) or not _SHA.fullmatch(digest)
                or object_path != 'objects/' + digest + suffix):
            raise ValueError('Unsafe or unverified portable terminal evidence mapping.')
        path = Path(source) / object_path
        if not path.resolve().is_relative_to(Path(source).resolve()):
            raise ValueError('Portable terminal object escaped its snapshot.')
        raw = evidence_bytes(path, digest, limit)
        if is_terminal:
            outcome = _outcome(raw)
            if outcome.payload['outcome_revision'] != relative.parts[1]:
                raise ValueError('Portable terminal revision does not match its mapping.')
            outcomes.append(outcome)
            terminal_bytes += len(raw)
        else:
            if sha256(raw).hexdigest() != relative.parts[1][:-5]:
                raise ValueError('Portable acknowledgment name does not match its bytes.')
            acknowledgments.append(load_acknowledgment(raw))
            ack_bytes += len(raw)
        if (len(outcomes) > HISTORY_RECORDS or len(acknowledgments) > ACK_RECORDS
                or terminal_bytes > HISTORY_BYTES or ack_bytes > ACK_BYTES):
            raise ValueError('Portable terminal evidence exceeds its aggregate budget.')
        rows.append((name, object_path, digest, limit, raw))
    _, conflicts = coalesce_run_outcomes(outcomes)
    if conflicts:
        raise ValueError('Conflicting portable terminal outcomes.')
    revisions = {outcome.payload['outcome_revision'] for outcome in outcomes}
    if any(record['outcome_revision'] not in revisions for record in acknowledgments):
        raise ValueError('Portable acknowledgment has no included terminal outcome.')
    reserved_objects = {name for name in index['files']
                        if isinstance(name, str) and name.endswith(('.terminal.json', '.terminal-ack.json'))}
    if reserved_objects != {row[1] for row in rows}:
        raise ValueError('Portable terminal object inventory has omitted or extra evidence mappings.')
    return tuple(rows)
