"""Bounded historical terminal loading; never install a current analysis."""
from dataclasses import dataclass
import os
import stat
from pathlib import Path

from seedvision.export.run_outcomes import LIMIT, FORMAT, NativeRunOutcome, _json, load_run_outcome
from seedvision.export.preflight_outcomes import PreflightRunOutcome


HISTORY_ENTRIES = 2048
HISTORY_RECORDS = 256
HISTORY_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class RunOutcomeHistory:
    outcomes: tuple
    diagnostics: tuple[str, ...]
    incomplete: bool
    bytes_read: int
    entries_examined: int


def _read(path, limit):
    path = Path(path)
    info = path.lstat()
    if (path.is_symlink() or getattr(info, 'st_file_attributes', 0) & 0x400
            or not stat.S_ISREG(info.st_mode)):
        raise ValueError('Linked terminal records are not loaded implicitly.')
    with path.open('rb') as stream:
        return stream.read(limit + 1)


def _outcome(raw):
    try:
        text = raw.decode('utf-8')
    except UnicodeError as error:
        raise ValueError('Terminal record is not UTF-8 JSON.') from error
    payload = load_run_outcome(text)
    if payload['format'] == FORMAT:
        return NativeRunOutcome(Path(payload['execution']['source_path']), text)
    return PreflightRunOutcome(Path(payload['execution']['subject_path']), text)


def load_terminal_record(path):
    """Read at most LIMIT+1 bytes; old source/artifact paths are never probed."""
    raw = _read(path, LIMIT)
    if len(raw) > LIMIT:
        raise ValueError('Terminal record exceeds its bounded JSON contract.')
    return _outcome(raw)


def coalesce_run_outcomes(outcomes):
    """Identical exports deduplicate; contradictory diagnostics/locators withhold."""
    unique, conflicting, diagnostics = {}, set(), []
    count, total = 0, 0
    for outcome in outcomes:
        count += 1
        if not isinstance(outcome, (NativeRunOutcome, PreflightRunOutcome)):
            raise ValueError('Unknown terminal history object.')
        payload = outcome.payload
        canonical = _json(payload)
        total += len(canonical.encode('utf-8'))
        if count > HISTORY_RECORDS or total > HISTORY_BYTES:
            return (), ('Terminal history exceeds its record/byte budget; select one record explicitly.',)
        revision = payload['outcome_revision']
        if revision in conflicting:
            continue
        previous = unique.get(revision)
        if previous is not None and _json(previous.payload) != canonical:
            conflicting.add(revision)
            unique.pop(revision)
            diagnostics.append(f'Conflicting terminal records for revision {revision}; all copies withheld.')
        else:
            unique[revision] = outcome
    return tuple(unique[key] for key in sorted(unique)), tuple(diagnostics)


def load_run_outcome_history(directory, *, entry_limit=HISTORY_ENTRIES,
                             record_limit=HISTORY_RECORDS, byte_limit=HISTORY_BYTES):
    """Inventory unique export directories with finite traversal/read budgets."""
    if any(type(value) is not int or value <= 0 for value in (entry_limit, record_limit, byte_limit)):
        raise ValueError('History limits must be positive integers.')
    root = Path(directory)
    diagnostics, candidates, outcomes = [], [], []
    examined = read_bytes = 0
    incomplete = False
    try:
        with os.scandir(root) as entries:
            for entry in entries:
                examined += 1
                if examined > entry_limit:
                    incomplete = True
                    diagnostics.append('Terminal history traversal budget exceeded; inventory withheld.')
                    break
                if entry.name.startswith('.run-outcome-'):
                    continue
                try:
                    info = entry.stat(follow_symlinks=False)
                except OSError as error:
                    diagnostics.append(f'{entry.name}: cannot inspect history entry: {error}')
                    continue
                if entry.is_symlink() or getattr(info, 'st_file_attributes', 0) & 0x400:
                    diagnostics.append(f'{entry.name}: linked history entry withheld.')
                elif entry.is_dir(follow_symlinks=False):
                    candidates.append(Path(entry.path) / 'run-outcome.json')
    except FileNotFoundError:
        return RunOutcomeHistory((), (), False, 0, 0)
    except OSError as error:
        return RunOutcomeHistory((), (f'Cannot inventory terminal history: {error}',), True, 0, examined)
    if len(candidates) > record_limit:
        incomplete = True
        diagnostics.append('Terminal history record budget exceeded; inventory withheld.')
    if incomplete:
        return RunOutcomeHistory((), tuple(diagnostics), True, 0, examined)
    for path in sorted(candidates):
        remaining = byte_limit - read_bytes
        try:
            raw = _read(path, min(LIMIT, remaining))
            read_bytes += len(raw)
            if len(raw) > remaining:
                incomplete = True
                diagnostics.append('Terminal history byte budget exceeded; inventory withheld.')
                break
            if len(raw) > LIMIT:
                raise ValueError('Terminal record exceeds its bounded JSON contract.')
            outcomes.append(_outcome(raw))
        except (ValueError, OSError) as error:
            diagnostics.append(f'{path.parent.name}: {type(error).__name__}: {error}')
    if incomplete:
        return RunOutcomeHistory((), tuple(diagnostics), True, read_bytes, examined)
    coherent, conflicts = coalesce_run_outcomes(outcomes)
    return RunOutcomeHistory(coherent, tuple(diagnostics) + conflicts, False, read_bytes, examined)
