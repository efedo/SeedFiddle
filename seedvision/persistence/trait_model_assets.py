"""Bounded frozen trait-model objects for portable project snapshots."""
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
import re

from seedvision.learning.trait_classifier import FrozenTraitModel


MAX_TRAIT_MODEL_BYTES = 4 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class TraitModelAsset:
    path: str
    source_sha256: str
    source_bytes: int
    model_sha256: str
    node_id: str = 'trait_classifier'

    def __post_init__(self):
        if (self.node_id != 'trait_classifier'
                or type(self.source_bytes) is not int
                or not 0 < self.source_bytes <= MAX_TRAIT_MODEL_BYTES
                or any(not isinstance(value, str) or re.fullmatch('[0-9a-f]{64}', value) is None
                       for value in (self.source_sha256, self.model_sha256))
                or self.path != 'objects/' + self.source_sha256 + '.trait.json'):
            raise ValueError('Invalid portable trait-model identity.')

    def payload(self):
        return asdict(self)


def selected_trait_model(profile):
    for node in profile.nodes:
        if node.identifier == 'trait_classifier':
            value = dict(node.parameters)['trait_model_path']
            if not isinstance(value, str) or not value.strip():
                raise ValueError('Trait model selection must be explicit or None.')
            return None if value.strip().casefold() == 'none' else value
    return None


def trait_model_bytes(path, asset=None):
    """Validate the exact consumed bytes, with or without a recorded identity."""
    with Path(path).open('rb') as stream:
        raw = stream.read((asset.source_bytes if asset else MAX_TRAIT_MODEL_BYTES) + 1)
    if not raw or len(raw) > MAX_TRAIT_MODEL_BYTES:
        raise ValueError('Frozen trait model exceeds its supported size or is empty.')
    digest = sha256(raw).hexdigest()
    if asset and (len(raw) != asset.source_bytes or digest != asset.source_sha256):
        raise ValueError('Portable trait model differs from its recorded bytes.')
    model = FrozenTraitModel.from_json(raw.decode('utf-8'))
    if asset and model.model_sha256 != asset.model_sha256:
        raise ValueError('Portable trait model differs from its recorded model identity.')
    return raw, model.model_sha256


def capture_trait_model(root, profile):
    selected = selected_trait_model(profile)
    if selected is None:
        return None
    path = Path(selected).expanduser()
    path = path.resolve() if path.is_absolute() else (Path(root) / path).resolve()
    raw, model_digest = trait_model_bytes(path)
    digest = sha256(raw).hexdigest()
    asset = TraitModelAsset('objects/' + digest + '.trait.json', digest, len(raw), model_digest)
    return path, asset, raw


def snapshot_trait_assets(profile, index):
    rows = index.get('trait_model_assets', [])
    selected = selected_trait_model(profile)
    if not isinstance(rows, list) or len(rows) != (0 if selected is None else 1):
        raise ValueError('Portable trait-model closure is incomplete or inconsistent.')
    if selected is None:
        return {}
    try:
        if not isinstance(rows[0], dict):
            raise TypeError()
        asset = TraitModelAsset(**rows[0])
    except TypeError as error:
        raise ValueError('Invalid portable trait-model record.') from error
    if asset.path != selected or index['files'].get(asset.path) != asset.source_sha256:
        raise ValueError('Portable trait model is not in its selected verified inventory.')
    return {asset.path: asset}


def copy_trait_model(source, target, asset, raw=None):
    """Copy a validated immutable snapshot and reverify source and destination."""
    if raw is None:
        raw, _ = trait_model_bytes(source, asset)
    elif len(raw) != asset.source_bytes or sha256(raw).hexdigest() != asset.source_sha256:
        raise ValueError('Captured trait model differs from its recorded bytes.')
    target = Path(target)
    if not target.exists():
        target.write_bytes(raw)
    trait_model_bytes(target, asset)
    trait_model_bytes(source, asset)
