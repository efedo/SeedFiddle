"""Compact, immutable facts for the native six-row, four-column card grid."""
from dataclasses import dataclass
import re


def normalize_neutral_patch_selection(value: str) -> str:
    """Accept automatic or unique local grid IDs, without Unicode coercion."""
    if not isinstance(value, str) or len(value) > 1024:
        raise ValueError('Neutral patches must be automatic or comma-separated IDs 1 through 24.')
    value = value.strip(' \t')
    if value == 'automatic':
        return value
    tokens = [token.strip(' \t') for token in value.split(',')]
    if any(re.fullmatch(r'(?:[1-9]|1[0-9]|2[0-4])', token) is None for token in tokens):
        raise ValueError('Use automatic or comma-separated decimal patch IDs 1 through 24.')
    ids = [int(token) for token in tokens]
    if len(set(ids)) != len(ids):
        raise ValueError('Neutral patch IDs must be unique.')
    return ','.join(map(str, sorted(ids)))


@dataclass(frozen=True, slots=True)
class NeutralPatchWitness:
    patch_id: int
    source_corners: tuple[tuple[float, float], ...]
    sample_corners: tuple[tuple[float, float], ...]
    source_mean_bgr: tuple[float, float, float]
    sampled_mean_bgr: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class NeutralBalanceFacts:
    enabled: bool
    requested_selection: str
    sample_frame: str
    state: str
    selection_mode: str
    available_patch_ids: tuple[int, ...]
    used_patch_ids: tuple[int, ...]
    patches: tuple[NeutralPatchWitness, ...]
    channel_gains_bgr: tuple[float, float, float]
    detection_sha256: str | None
    reason: str = ''


def neutral_balance_description(facts: NeutralBalanceFacts | None) -> str:
    if facts is None:
        return 'neutral balance state unrecorded'
    if facts.state != 'applied':
        return f'neutral balance {facts.state}'
    ids = ', '.join(map(str, facts.used_patch_ids))
    return f'neutral balance from {facts.sample_frame} patches {ids}'
